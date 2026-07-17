#!/usr/bin/env python3
"""Carding stage A: transplant teacher embedding geometry into a CHARKHA checkpoint.

This is the sane "eat an open model's weights" primitive. It does not pretend that raw bytes contain
portable skill. It learns a linear map from a teacher embedding space into the current CHARKHA
embedding space using anchor token pairs, then blends projected teacher vectors into selected student
embedding rows.

Why this is useful:
  * It is architecture-agnostic at the teacher side: any model with an embedding table can be a donor.
  * It is tokenizer-tolerant when you provide anchor/transfer pairs from text-token alignment.
  * It is reversible and checkpoint-local: write a new init checkpoint, then train normally.

The first production use should be conservative: small alpha (0.05-0.20), anchors from high-confidence
shared strings, then a short validation NLL smoke before any long run.

Anchor / transfer file format (JSONL):
  {"teacher_id": 123, "student_id": 456}

Usage:
  python scripts/card_embeddings.py --student runs/main/ckpt.pt --teacher-embed qwen_embed.pt \
    --anchors anchors.jsonl --transfer transfers.jsonl --alpha 0.10 --out runs/main/carded.pt
  python scripts/card_embeddings.py --selftest
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Iterable

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _state_and_cfg(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck, ck.get("model", ck), ck.get("cfg")


def _student_embed_key(state: dict[str, torch.Tensor]) -> str:
    if "embed.codes.weight" in state:  # factorized embedding, used by current production configs
        return "embed.codes.weight"
    if "embed.weight" in state:  # dense fallback/toy configs
        return "embed.weight"
    raise KeyError(
        "cannot find CHARKHA embedding key (expected embed.codes.weight or embed.weight)"
    )


def load_embedding_table(path: str, key: str | None = None) -> torch.Tensor:
    """Load a donor embedding table from .pt/.pth/.bin safetensors-ish payloads.

    For HF models too large to instantiate here, save just the embedding tensor first, e.g. from a
    cloud box: torch.save(model.get_input_embeddings().weight.cpu(), 'teacher_embed.pt')
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if torch.is_tensor(obj):
        return obj.detach().float()
    if isinstance(obj, dict):
        if key and key in obj:
            return obj[key].detach().float()
        for k in (
            "embed_tokens.weight",
            "model.embed_tokens.weight",
            "transformer.wte.weight",
            "gpt_neox.embed_in.weight",
            "embed.weight",
            "embed.codes.weight",
            "weight",
        ):
            if k in obj and torch.is_tensor(obj[k]):
                return obj[k].detach().float()
        if "model" in obj and isinstance(obj["model"], dict):
            return load_embedding_table_from_state(obj["model"], key)
    raise ValueError(f"could not find an embedding tensor in {path}; pass --teacher-key")


def load_embedding_table_from_state(
    state: dict[str, torch.Tensor], key: str | None = None
) -> torch.Tensor:
    if key:
        return state[key].detach().float()
    return state[_student_embed_key(state)].detach().float()


def read_pairs(path: str) -> list[tuple[int, int]]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            try:
                ti, si = int(d["teacher_id"]), int(d["student_id"])
            except KeyError as e:
                raise ValueError(f"{path}:{ln} missing {e}; expected teacher_id/student_id") from e
            pairs.append((ti, si))
    if not pairs:
        raise ValueError(f"{path} contained no pairs")
    return pairs


def fit_ridge(src: torch.Tensor, dst: torch.Tensor, l2: float = 1e-3) -> torch.Tensor:
    """Fit affine map [src,1] @ W ~= dst. Returns W with shape (src_dim+1, dst_dim)."""
    if src.ndim != 2 or dst.ndim != 2 or src.size(0) != dst.size(0):
        raise ValueError(f"bad fit shapes src={tuple(src.shape)} dst={tuple(dst.shape)}")
    ones = torch.ones(src.size(0), 1, dtype=src.dtype)
    X = torch.cat([src.float(), ones], dim=1)
    Y = dst.float()
    reg = torch.eye(X.size(1), dtype=torch.float32) * float(l2)
    reg[-1, -1] = 0.0  # do not penalize the bias term
    return torch.linalg.solve(X.T @ X + reg, X.T @ Y)


def project(vecs: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(vecs.size(0), 1, dtype=torch.float32)
    return torch.cat([vecs.float(), ones], dim=1) @ W.float()


def transplant(
    student_state: dict[str, torch.Tensor],
    teacher_embed: torch.Tensor,
    anchor_pairs: Iterable[tuple[int, int]],
    transfer_pairs: Iterable[tuple[int, int]] | None = None,
    alpha: float = 0.1,
    l2: float = 1e-3,
):
    """Return a new state dict with projected teacher rows blended into student embeddings."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("--alpha must be in [0,1]")
    if teacher_embed.ndim != 2:
        raise ValueError(f"teacher embedding must be rank-2, got {tuple(teacher_embed.shape)}")
    if not torch.isfinite(teacher_embed).all():
        raise ValueError("teacher embedding contains NaN/Inf; refusing to card a poisoned donor")
    ekey = _student_embed_key(student_state)
    student_embed = student_state[ekey].detach().float()
    if not torch.isfinite(student_embed).all():
        raise ValueError("student embedding contains NaN/Inf; fix the checkpoint before Carding")
    anchors = list(anchor_pairs)
    transfers = list(transfer_pairs) if transfer_pairs is not None else anchors
    if len({si for _, si in transfers}) != len(transfers):
        raise ValueError(
            "transfer pairs contain duplicate student_id rows; make the mapping one-to-one"
        )
    max_t = teacher_embed.size(0) - 1
    max_s = student_embed.size(0) - 1
    for ti, si in anchors + transfers:
        if not (0 <= ti <= max_t and 0 <= si <= max_s):
            raise ValueError(f"pair out of range teacher_id={ti} student_id={si}")
    src = torch.stack([teacher_embed[ti] for ti, _ in anchors]).float()
    dst = torch.stack([student_embed[si] for _, si in anchors]).float()
    W = fit_ridge(src, dst, l2=l2)
    t_ids = torch.tensor([ti for ti, _ in transfers], dtype=torch.long)
    s_ids = torch.tensor([si for _, si in transfers], dtype=torch.long)
    projected = project(teacher_embed[t_ids], W).to(student_embed.dtype)

    new_state = {k: v.detach().cpu().clone() for k, v in student_state.items()}
    updated = new_state[ekey].float()
    updated[s_ids] = (1.0 - alpha) * updated[s_ids] + alpha * projected
    new_state[ekey] = updated.to(student_state[ekey].dtype)
    return new_state, {
        "embed_key": ekey,
        "anchors": len(anchors),
        "transfers": len(transfers),
        "alpha": alpha,
        "l2": l2,
    }


def save_student(
    out_path: str, source_ckpt: dict, state: dict[str, torch.Tensor], meta_update: dict
):
    meta = dict(source_ckpt.get("meta", {}) or {})
    meta["card_embeddings"] = meta_update
    payload = {
        "model": state,
        "cfg": source_ckpt.get("cfg"),
        "step": int(source_ckpt.get("step", 0)),
        "meta": meta,
    }
    tmp = out_path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, out_path)


def selftest():
    from charkha import Charkha, CharkhaConfig

    print("CHARKHA Carding embedding self-test")
    torch.manual_seed(0)
    tmp = tempfile.mkdtemp(prefix="charkha_card_")
    cfg = CharkhaConfig.toy()
    model = Charkha(cfg)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    key = _student_embed_key(state)
    student_embed = state[key].float()

    teacher_dim = student_embed.size(1) + 7
    teacher = torch.randn(student_embed.size(0) + 11, teacher_dim)
    true_W = torch.randn(teacher_dim + 1, student_embed.size(1)) * 0.05
    # Make anchor destination exactly affine from teacher so the fit has a known answer.
    anchors = [(i, i) for i in range(32)]
    mapped = project(teacher[:32], true_W)
    state[key][:32] = mapped.to(state[key].dtype)
    transfers = [(40 + i, 40 + i) for i in range(8)]
    before = state[key].clone()
    new_state, meta = transplant(state, teacher, anchors, transfers, alpha=1.0, l2=1e-6)
    target = project(
        teacher[torch.tensor([ti for ti, _ in transfers])], fit_ridge(teacher[:32], mapped, l2=1e-6)
    )
    checks = {
        "uses expected embedding key": meta["embed_key"] == key,
        "transfer rows updated": not torch.allclose(
            new_state[key][40:48].float(), before[40:48].float()
        ),
        "projected rows match fitted map": torch.allclose(
            new_state[key][40:48].float(), target, atol=1e-3
        ),
        "non-transfer row unchanged": torch.allclose(
            new_state[key][70].float(), before[70].float()
        ),
    }
    try:
        transplant(state, teacher, anchors, [(40, 40), (41, 40)], alpha=0.1)
        checks["duplicate student transfer refused"] = False
    except ValueError:
        checks["duplicate student transfer refused"] = True
    bad_teacher = teacher.clone()
    bad_teacher[0, 0] = float("inf")
    try:
        transplant(state, bad_teacher, anchors, transfers, alpha=0.1)
        checks["non-finite teacher refused"] = False
    except ValueError:
        checks["non-finite teacher refused"] = True
    ckpt = {"model": state, "cfg": dict(cfg.__dict__), "step": 0, "meta": {}}
    out = os.path.join(tmp, "carded.pt")
    save_student(out, ckpt, new_state, meta)
    loaded = torch.load(out, map_location="cpu", weights_only=False)
    mdl = Charkha(CharkhaConfig.from_dict(loaded["cfg"]))
    mdl.load_state_dict(loaded["model"])
    with torch.no_grad():
        logits, _ = mdl(torch.randint(0, cfg.vocab_size, (1, 8)))
    checks["saved checkpoint loads and runs"] = tuple(logits.shape) == (1, 8, cfg.vocab_size)

    ok = True
    for name, passed in checks.items():
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(
        "\nSELFTEST",
        "PASS - teacher embedding geometry can be mapped into CHARKHA rows" if ok else "FAIL",
    )
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(
        description="Card teacher embedding geometry into a CHARKHA checkpoint"
    )
    ap.add_argument("--student", help="CHARKHA checkpoint to modify")
    ap.add_argument("--teacher-embed", help="torch-saved teacher embedding tensor or state dict")
    ap.add_argument("--teacher-key", help="optional tensor key inside --teacher-embed")
    ap.add_argument("--anchors", help="JSONL teacher_id/student_id pairs used to fit the map")
    ap.add_argument(
        "--transfer", help="JSONL teacher_id/student_id pairs to update; defaults to anchors"
    )
    ap.add_argument(
        "--alpha", type=float, default=0.1, help="blend strength for projected teacher rows"
    )
    ap.add_argument("--l2", type=float, default=1e-3, help="ridge regularization")
    ap.add_argument("--out", help="output CHARKHA checkpoint")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    for required in ("student", "teacher_embed", "anchors", "out"):
        if getattr(args, required) is None:
            ap.error(f"--{required.replace('_', '-')} is required")
    ck, state, _cfg = _state_and_cfg(args.student)
    teacher = load_embedding_table(args.teacher_embed, args.teacher_key)
    anchors = read_pairs(args.anchors)
    transfers = read_pairs(args.transfer) if args.transfer else anchors
    new_state, meta = transplant(state, teacher, anchors, transfers, alpha=args.alpha, l2=args.l2)
    save_student(args.out, ck, new_state, meta)
    print(f"[card] wrote {args.out}: {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

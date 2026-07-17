"""
CHARKHA consolidate - sleep-time personal-data consolidation.
=============================================================
The mechanism behind "slowly pump your own data into your own checkpoint": while the machine
is idle, this daemon turns what the owner did with the model into weights — safely.

Cycle (one `--once` run, or repeated under `--watch`):
  1. GATHER   new material since the last cycle:
                - serve.py conversation log (runs/serve/conversations.jsonl), keeping only
                  exchanges whose assistant turn passed the confidence gate (>= --min-conf) —
                  the model must not consolidate its own low-confidence guesses;
                - documents the owner dropped into data/personal/inbox/ (.txt/.md), tracked
                  by content hash so nothing is ingested twice.
  2. TOKENIZE the cumulative personal archive into data/personal-dd shards with the SAME
              tokenizer that produced the training shards (from the checkpoint config).
  3. UPDATE   run a bounded continual-training burst from the serving checkpoint: personal
              shards oversampled against a replay mix of the original pretraining dirs
              (replay is what prevents catastrophic forgetting), at the validated 8GB config.
  4. GATE     measure held-out NLL on the GENERAL replay data before and after. Promote the
              new checkpoint only if general ability did not regress beyond --gate-eps and
              personal NLL improved; otherwise discard the update and keep the old weights.

Nothing is uploaded anywhere; the archive, shards, and checkpoints stay on this machine.

  python src/consolidate.py --ckpt runs/model/ckpt.pt --replay data/replay-dd --once
  python src/consolidate.py --ckpt runs/model/ckpt.pt --replay data/replay-dd --watch 3600

"""

from __future__ import annotations
import argparse
import glob
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

CONV_LOG = os.path.join(ROOT, "runs", "serve", "conversations.jsonl")
PERSONAL = os.path.join(ROOT, "data", "personal")
INBOX = os.path.join(PERSONAL, "inbox")
ARCHIVE = os.path.join(PERSONAL, "archive.txt")
STATE = os.path.join(PERSONAL, "consolidate_state.json")
SHARDS = os.path.join(ROOT, "data", "personal-dd")


def _load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"conv_offset": 0, "ingested": [], "cycles": 0}


def _save_state(st):
    os.makedirs(PERSONAL, exist_ok=True)
    tmp = STATE + ".tmp"
    json.dump(st, open(tmp, "w"), indent=1)
    os.replace(tmp, STATE)


def gather(
    state, min_conf=0.35, conv_log=CONV_LOG, inbox=INBOX, world_model=None, contradiction_boost=2
):
    """Collect new gated material as a list of text documents. Mutates state offsets."""
    docs = []
    if os.path.exists(conv_log):
        with open(conv_log, encoding="utf-8") as f:
            f.seek(state.get("conv_offset", 0))
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # partial trailing line; re-read next cycle
                # confidence gate: never consolidate the model's own low-confidence guesses
                if float(rec.get("conf", 0.0)) < min_conf:
                    continue
                docs.append(f"user: {rec.get('user', '')}\nassistant: {rec.get('assistant', '')}\n")
            state["conv_offset"] = f.tell()
    seen = set(state.get("ingested", []))
    if os.path.isdir(inbox):
        for p in sorted(glob.glob(os.path.join(inbox, "*"))):
            if not p.lower().endswith((".txt", ".md")):
                continue
            try:
                text = open(p, encoding="utf-8", errors="replace").read().strip()
            except OSError:
                continue
            h = hashlib.sha1(text.encode()).hexdigest()
            if h in seen or not text:
                continue
            docs.append(text + "\n")
            seen.add(h)
    state["ingested"] = sorted(seen)
    if world_model:
        try:
            from worldmodel import WorldModel

            wm = WorldModel(world_model)
            rows = wm.contradiction_docs(since_ts=state.get("world_ts", 0.0))
            wm.close()
            for d in rows:
                for _ in range(max(1, int(contradiction_boost))):
                    docs.append(d)
            if rows:
                state["world_ts"] = time.time()
        except Exception:
            pass
    return docs


def build_shards(new_docs, tokenizer_path, vocab_size, archive=ARCHIVE, out_dir=SHARDS):
    """Append new docs to the cumulative archive, then (re)tokenize the archive into shards.
    The archive is tiny at personal scale, so a full rebuild each cycle keeps this simple and
    idempotent. Returns total tokens written."""
    from dataprep import ShardWriter
    from tokenizers import Tokenizer

    os.makedirs(PERSONAL, exist_ok=True)
    with open(archive, "a", encoding="utf-8") as f:
        for d in new_docs:
            f.write(d + "\n\x00DOC\x00\n")  # explicit doc separator for the re-split
    tok = Tokenizer.from_file(tokenizer_path)
    eos = tok.token_to_id("<|endoftext|>") or 0
    tmp = out_dir + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    writer = ShardWriter(tmp, shard_tokens=5_000_000, vocab_size=vocab_size)
    text = open(archive, encoding="utf-8").read()
    for doc in text.split("\n\x00DOC\x00\n"):
        if doc.strip():
            writer.add(tok.encode(doc).ids + [eos])
    writer.close()
    idx = {
        "vocab_size": vocab_size,
        "total_tokens": writer.total,
        "tokenizer": os.path.abspath(tokenizer_path),
        "shards": [{"file": s["file"], "tokens": s["tokens"]} for s in writer.shards],
        "accounting": {"source": "consolidate-personal"},
    }
    json.dump(idx, open(os.path.join(tmp, "index.json"), "w"), indent=1)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.replace(tmp, out_dir)
    return writer.total


def eval_nll(ckpt_path, data_dir, device, iters=12, T=512):
    """Held-out NLL of a checkpoint over one shard dir (its tail is used as val)."""
    from train import load_ckpt, evaluate, ShardLoader

    model, _cfg, _opts, _bases, _ck = load_ckpt(ckpt_path, device)
    loader = ShardLoader([data_dir], val_frac=0.05, split="val")  # held-out tail shards
    rng = random.Random(7)  # fixed batches -> comparable NLLs
    nll, _ppl, _bpt = evaluate(model, loader, 1, T, device, rng, iters=iters)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return nll


def run_update(ckpt_path, replay_dirs, steps, work_dir, personal_boost=3, extra_args=()):
    """Bounded continual burst from a COPY of the serving checkpoint (never in place). The
    personal dir is passed personal_boost times so the loader oversamples it against replay.
    Returns the path of the updated checkpoint."""
    os.makedirs(work_dir, exist_ok=True)
    work_ckpt = os.path.join(work_dir, "ckpt.pt")
    shutil.copy2(ckpt_path, work_ckpt)
    start = torch.load(work_ckpt, map_location="cpu", weights_only=False)["step"]
    data_args = []
    for _ in range(personal_boost):
        data_args += ["--data", SHARDS]
    for d in replay_dirs:
        data_args += ["--data", d]
    cmd = [
        sys.executable,
        os.path.join(HERE, "train.py"),
        "--resume",
        "--profile",
        "frontier",
        "--out",
        work_dir,
        "--steps",
        str(start + steps),
        "--seq-len",
        "512",
        "--batch-size",
        "1",
        "--accum-steps",
        "8",
        "--ce-chunk",
        "1024",
        "--gdn-chunk",
        "32",
        "--grad-checkpoint",
        "--offload-optim",
        "--8bit-optim",
        "--symmetry-opt",
        "--bptt-half",
        "--muon-lr",
        "0.004",
        "--adam-lr",
        "6e-4",  # ~5x below pretrain LR: consolidation
        "--eval-every",
        "0",
        "--save-every",
        str(steps),  # nudges, it does not re-cook the model
        *data_args,
        *extra_args,
    ]
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        raise RuntimeError(f"consolidation training burst failed (exit {r.returncode})")
    return work_ckpt


def cycle(args, device):
    state = _load_state()
    docs = gather(
        state,
        min_conf=args.min_conf,
        world_model=args.world_model,
        contradiction_boost=args.contradiction_boost,
    )
    if not docs and not args.force:
        print("[consolidate] no new gated material; nothing to do")
        return False
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    tok_path = ck["cfg"].get("tokenizer_name")
    vocab = ck["cfg"]["vocab_size"]
    assert tok_path and os.path.exists(tok_path), f"checkpoint has no usable tokenizer: {tok_path}"
    total = build_shards(docs, tok_path, vocab)
    print(f"[consolidate] gathered {len(docs)} docs -> {total:,} personal tokens")
    if total < args.min_tokens:
        print(f"[consolidate] under --min-tokens ({args.min_tokens}); deferring to a later cycle")
        _save_state(state)  # offsets advance; material stays in the archive
        return False

    gen_before = eval_nll(args.ckpt, args.replay[0], device)
    per_before = eval_nll(args.ckpt, SHARDS, device)
    work = os.path.join(ROOT, "runs", "consolidate", time.strftime("%Y%m%d-%H%M%S"))
    new_ckpt = run_update(args.ckpt, args.replay, args.steps, work)
    gen_after = eval_nll(new_ckpt, args.replay[0], device)
    per_after = eval_nll(new_ckpt, SHARDS, device)
    print(
        f"[consolidate] general nll {gen_before:.4f} -> {gen_after:.4f} | "
        f"personal nll {per_before:.4f} -> {per_after:.4f}"
    )

    if gen_after <= gen_before + args.gate_eps and per_after < per_before:
        shutil.copy2(args.ckpt, args.ckpt + ".prev")  # one-step rollback always available
        shutil.copy2(new_ckpt, args.ckpt)
        print(f"[consolidate] PROMOTED — serving checkpoint updated ({args.ckpt})")
        state["cycles"] = state.get("cycles", 0) + 1
        _save_state(state)
        return True
    print("[consolidate] REJECTED — general regression or no personal gain; weights unchanged")
    _save_state(state)  # material stays archived for the next attempt
    return False


def _busy():
    """True if a training/retokenization job owns the machine — never fight the day job."""
    try:
        out = subprocess.run(
            ["pgrep", "-f", "train.py|preflight.py"], capture_output=True, text=True
        ).stdout.strip()
        return bool(out)
    except OSError:
        return False


def selftest():
    """Hermetic: gathering, gating, offset/idempotence, and archive/doc-splitting logic."""
    import tempfile

    tmp = tempfile.mkdtemp(prefix="charkha_consol_")
    conv = os.path.join(tmp, "conv.jsonl")
    inbox = os.path.join(tmp, "inbox")
    os.makedirs(inbox)
    with open(conv, "w") as f:
        f.write(json.dumps({"user": "hi", "assistant": "hello", "conf": 0.9}) + "\n")
        f.write(json.dumps({"user": "q", "assistant": "bad guess", "conf": 0.1}) + "\n")
    open(os.path.join(inbox, "note.txt"), "w").write("my project notes")
    open(os.path.join(inbox, "skip.pdf"), "w").write("binary-ish")
    checks = {}
    st = {"conv_offset": 0, "ingested": []}
    docs = gather(st, min_conf=0.35, conv_log=conv, inbox=inbox)
    checks["high-conf exchange kept"] = any("hello" in d for d in docs)
    checks["low-conf exchange gated out"] = not any("bad guess" in d for d in docs)
    checks["inbox txt ingested"] = any("project notes" in d for d in docs)
    checks["non-text inbox file skipped"] = not any("binary-ish" in d for d in docs)
    docs2 = gather(st, min_conf=0.35, conv_log=conv, inbox=inbox)
    checks["second gather is a no-op (offsets/hashes)"] = docs2 == []
    with open(conv, "a") as f:
        f.write(json.dumps({"user": "new", "assistant": "fresh", "conf": 0.8}) + "\n")
    docs3 = gather(st, min_conf=0.35, conv_log=conv, inbox=inbox)
    checks["only new material on the next cycle"] = len(docs3) == 1 and "fresh" in docs3[0]
    try:
        from worldmodel import WorldModel

        wmp = os.path.join(tmp, "world.sqlite")
        wm = WorldModel(wmp)
        wm.remember("My GPU is a 4060 Ti.", ts=1)
        wm.remember("My GPU is an RTX 5090.", ts=2)
        wm.close()
        wdocs = gather(
            st, min_conf=0.35, conv_log=conv, inbox=inbox, world_model=wmp, contradiction_boost=2
        )
        checks["world contradictions enter replay twice"] = (
            len([d for d in wdocs if "CONTRADICTION" in d]) >= 2
        )
    except Exception:
        checks["world contradictions enter replay twice"] = False
    ok = all(checks.values())
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("consolidate selftest", "OK" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="CHARKHA sleep-time consolidation")
    ap.add_argument("--ckpt", help="serving checkpoint to consolidate into")
    ap.add_argument("--replay", nargs="+", default=[], help="pretraining shard dirs for replay mix")
    ap.add_argument("--steps", type=int, default=300, help="training burst length per cycle")
    ap.add_argument("--min-conf", type=float, default=0.35)
    ap.add_argument(
        "--min-tokens",
        type=int,
        default=2000,
        help="defer the burst until this much personal material has accumulated",
    )
    ap.add_argument(
        "--gate-eps",
        type=float,
        default=0.01,
        help="max tolerated general-NLL regression at promotion",
    )
    ap.add_argument("--once", action="store_true")
    ap.add_argument(
        "--watch",
        type=int,
        default=0,
        metavar="SECONDS",
        help="daemon mode: attempt a cycle every N seconds when the machine is idle",
    )
    ap.add_argument("--force", action="store_true", help="run a cycle even with no new material")
    ap.add_argument(
        "--world-model",
        default=None,
        help="serve.py --world-model sqlite path; contradictions are replayed during sleep",
    )
    ap.add_argument(
        "--contradiction-boost",
        type=int,
        default=2,
        help="oversampling multiplier for contradiction replay docs",
    )
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    assert args.ckpt and args.replay, "--ckpt and --replay are required (or use --selftest)"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.watch:
        print(f"[consolidate] watching every {args.watch}s (idle-only); Ctrl-C to stop")
        while True:
            if _busy():
                print("[consolidate] machine busy; skipping this tick")
            else:
                try:
                    cycle(args, device)
                except Exception as e:
                    print(f"[consolidate] cycle failed (weights untouched): {e}")
            time.sleep(args.watch)
    else:
        cycle(args, device)


if __name__ == "__main__":
    main()

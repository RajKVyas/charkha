"""
CHARKHA train - production training runner over tokenized shards.
==================================================================
Crash-survivable, run-forever training loop: consumes the .bin shards + index.json
that dataprep.py emits (uint16 or uint32, per vocab size) and feeds them to the
reference Charkha model.

What it adds over `python -m charkha._toy --toy`:
  * ShardLoader          mmap'd shards (dataprep output) + held-out val split
  * checkpoint/resume    full state - model + Muon + AdamW + step + RNG + metrics (atomic)
  * WSD branch-for-release   constant-LR cook; branch a linear decay+anneal for any release
  * eval                 periodic val NLL / perplexity / bits-per-token
  * metrics              JSONL log + console (loss EMA, tok/s, grad-norm, LR)
  * grad checkpoint      on the recurrent core (8GB invariant) via cfg.grad_checkpoint

Self-test (joins the regression baseline; pure CPU, no network, writes a temp shard set):
  python src/train.py --selftest

Real run (canonical env = WSL2 + CUDA + triton. GDN-1 can use the production fla
kernel; default GDN-2 is currently exact PyTorch recurrence). After
`python src/dataprep.py --manifest configs/sources.yaml --out /data/data`:
  python src/train.py --data /data/data --out runs/m1 --steps 200000 \
      --batch-size 8 --seq-len 1024 --save-every 1000 --eval-every 2000 --grad-checkpoint
  # branch a release once a checkpoint looks good (WSD decay+anneal to zero LR):
  python src/train.py --branch-from runs/m1/ckpt.pt --decay-steps 4000 --out runs/m1-release
"""

from __future__ import annotations
import array
import json
import math
import os
import random
import signal
import sys
import tempfile
import time
from argparse import Namespace

# Graceful pause: the autopilot (or any operator) can yield the GPU at ANY moment — e.g. to game —
# without losing progress. A SIGTERM, or a <out>/PAUSE flag file, is caught here; the training loop
# notices at the next step boundary, writes an atomic checkpoint, and exits 0 (clean) so the resilient
# launcher does NOT restart it. Resume later picks up exactly from that checkpoint. No mid-step kill,
# so no corruption; no extra memory held, so no leak.
_PAUSE_REQUESTED = False


def _request_pause(signum, _frame):
    global _PAUSE_REQUESTED
    _PAUSE_REQUESTED = True


import numpy as np
import torch
import torch.nn.functional as F

import charkha as charkha_mod
from charkha import (
    Charkha,
    CharkhaConfig,
    build_optimizers,
    build_symmetry_optimizers,
    wsd_lr_mult,
    have_fla,
    Muon,
    NormM,
    install_grad_release,
    clip_grads_mixed,
)

# --------------------------------------------------------------------------
# Data: stream (x, y) batches from dataprep's uint16 .bin shards.
# --------------------------------------------------------------------------

from data import ShardLoader


def _clear_release_grads(model):
    """Drop any grad-release CPU accumulators (the skip-a-poisoned-step paths must clear these
    too, or a non-finite gradient survives into the NEXT step's accumulation)."""
    for p in model.parameters():
        if getattr(p, "_cpu_grad_ready", False):
            p._cpu_grad.zero_()
            p._cpu_grad_ready = False


def build_opt_set(model, args, offload=False):
    """Return (opts, bases, mode): a LIST of optimizers + parallel base-LR list + a mode tag.
    Default -> [Muon, AdamW] (bit-identical to the old 2-tuple path).
    --symmetry-opt -> [Muon, RowNormM, ColNormM, AdamW] (arXiv:2605.18106): per-layer-correct
    update rule (Muon for attn/GDN, row/col-norm for embeddings+SwiGLU, AdamW for heads/1D).
    All downstream plumbing (save/load/set_lr/step) iterates the list, so N optimizers is uniform."""
    use_8bit = getattr(args, "eightbit_optim", False)
    if getattr(args, "symmetry_opt", False):
        opts = build_symmetry_optimizers(
            model,
            muon_lr=args.muon_lr,
            norm_lr=args.norm_lr,
            adam_lr=args.adam_lr,
            offload=offload,
            use_8bit_adam=use_8bit,
        )
        bases = [args.muon_lr, args.norm_lr, args.norm_lr, args.adam_lr]
        return opts, bases, "symmetry"
    muon, adam = build_optimizers(
        model, muon_lr=args.muon_lr, adam_lr=args.adam_lr, offload=offload, use_8bit_adam=use_8bit
    )
    return [muon, adam], [args.muon_lr, args.adam_lr], "default"


def save_ckpt(path, model, opts, opt_mode, step, cfg, meta, rng):
    # torch.compile wraps the model in an OptimizedModule whose state_dict keys gain a '_orig_mod.'
    # prefix; unwrap so the checkpoint loads cleanly into a fresh (uncompiled) Charkha on resume/serve.
    model = getattr(model, "_orig_mod", model)
    payload = {
        "model": model.state_dict(),
        "opts": [o.state_dict() for o in opts],  # N optimizers (2 default / 4 symmetry)
        "opt_mode": opt_mode,  # how to rebuild the optimizer structure on load
        "step": step,  # next step to run on resume
        "cfg": dict(cfg.__dict__),
        "meta": meta,
        "rng": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": rng.getstate(),
        },
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic: a half-written ckpt never clobbers a good one


def _maybe_cast_param_dtype(model, args, device):
    if getattr(args, "param_bf16", False):
        if device != "cuda":
            print("[warn] --param-bf16 ignored outside CUDA")
            return model
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("--param-bf16 requires a CUDA device with bf16 support")
        model.to(dtype=torch.bfloat16)
        print("[param-dtype] GPU parameters stored in bf16; CPU grad/momentum buffers remain fp32")
    return model


def load_ckpt(path, device, args=None, offload=False):
    # map_location='cpu' (NOT device): a checkpoint trained WITHOUT --offload-optim saves Muon
    # momentum as GPU tensors (~1.3GB at 0.42B). Loading that straight to CUDA dumps the whole 1.3GB
    # into the reserved pool at load time; with --offload-optim the momentum is meant to live in CPU
    # RAM, but Muon only re-homes it lazily on the first step() -- and PyTorch's reserved pool is
    # sticky, so the 1.3GB stays grabbed forever. That silently inflates a resumed run ~1.3GB over a
    # fresh build of the SAME config -> spill into shared memory on a tight 8GB card (a cloud->local
    # resume hit exactly this: probe said 6.88G, the resumed run pegged 8.0G.
    # Loading to CPU first keeps the GPU clean; only what each optimizer truly needs on-device is moved.
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = CharkhaConfig.from_dict(ck["cfg"])
    model = Charkha(cfg).to(device)
    model.load_state_dict(ck["model"])  # CPU state -> GPU params (cross-device copy is fine)
    model = _maybe_cast_param_dtype(model, args or Namespace(), device)
    # Rebuild the SAME optimizer structure the ckpt was trained with (from opt_mode, not args) so a
    # symmetry-trained ckpt resumes as symmetry even if the caller forgot the flag.
    mode = ck.get("opt_mode", "default")
    a = Namespace(**(args.__dict__ if args is not None else _toy_args().__dict__))
    a.symmetry_opt = mode == "symmetry"
    opts, bases, _ = build_opt_set(model, a, offload=offload)
    states = ck.get("opts")
    if states is None and "muon" in ck:  # legacy 2-tuple ckpt (pre-list format)
        states = [ck["muon"], ck["adam"]]
    # states may still be None (grow_init.py writes opts=None): resume weights with
    # fresh optimizers — exactly what a grown init wants.
    for o, s in zip(opts, states or []):
        o.load_state_dict(s)  # state lands on CPU (map_location was cpu)
        # Place each optimizer's state on the device it actually runs on. An offloading Muon KEEPS its
        # momentum on CPU (step() streams it per-matrix) -- the whole point of --offload-optim, so never
        # move it to GPU. Everything else (AdamW moments, a non-offloaded Muon, NormM) must sit on the
        # param device or its kernels hit a device mismatch / run slow.
        if getattr(o, "cpu_offload", False):
            continue
        for st in o.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(device)
    return model, cfg, opts, bases, ck


def restore_rng(ck, rng):
    if "rng" not in ck:  # grow_init checkpoints carry no RNG state — keep the fresh seed
        return
    torch.set_rng_state(ck["rng"]["torch"].to(torch.uint8).cpu())
    np.random.set_state(ck["rng"]["numpy"])
    rng.setstate(ck["rng"]["python"])


# --------------------------------------------------------------------------
# Eval: true next-token NLL (pure CE, not the composite train loss).
# --------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, loader, B, T, device, rng, iters=20, chunk=1024):
    """Memory-frugal validation. The old eval built the full (B,T,vocab) logits AND ran with autograd
    on — at T=2048 that ~400MB logit tensor + graph was the spike that tipped an 8GB card into shared
    memory mid-run. Fix: run under no_grad (no graph) and never materialize full logits — take post-norm
    hidden states once, then stream the tied-head CE over the sequence in chunks of `chunk` positions.
    Numerically identical to F.linear(hidden, embed.weight) + cross_entropy."""
    was = model.training
    model.eval()
    tot, ntok = 0.0, 0
    with torch.no_grad():
        for _ in range(iters):
            x, y = loader.batch(B, T, device, rng)
            h = model.hidden(x)  # (B,T,d) post-norm, routed exactly like forward
            h, W = model._head_hw(h)  # factorization-aware tied head
            hf = h[:, :-1].reshape(-1, h.size(-1))  # positions 0..T-2 predict y[:, :-1]
            yf = y[:, :-1].reshape(-1)
            for i in range(0, hf.size(0), chunk):
                lg = F.linear(hf[i : i + chunk], W)  # (<=chunk, vocab) — small, freed each chunk
                tot += F.cross_entropy(lg, yf[i : i + chunk], reduction="sum").item()
            ntok += yf.numel()
    if was:
        model.train()
    nll = tot / max(ntok, 1)
    return nll, math.exp(min(nll, 20)), nll / math.log(2)  # nll, perplexity, bits/token


# --------------------------------------------------------------------------
# Config + LR plumbing.
# --------------------------------------------------------------------------


def _apply_frontier_profile(cfg):
    """Enable the safe default experimental stack for fresh high-capability runs.

    This intentionally does not run on resumed checkpoints: checkpoint cfg/model
    shape is the source of truth there, and changing architecture flags on resume
    can make the state dict invalid.
    """
    cfg.use_gdn2 = True
    cfg.use_nitp = True
    cfg.use_deep_supervision = True
    cfg.cross_loop_consistency = True
    cfg.use_osdn = True
    cfg.use_bipolar_gate = True
    cfg.use_mtp_routing = True
    # use_orthogonal_loss / neuromod_gate / use_abacus_embed / fixed_weighted_halt:
    # REMOVED from the codebase (2026-07-05 config unification). Old checkpoints
    # load via CharkhaConfig.from_dict.
    # STAGED (not limbo): use_loop_adapters / use_latent_memory / use_subconscious are
    # exact-no-op-at-init growth features.
    cfg.use_task_rl = True
    # Experimental reasoning modules remain disabled by default because they caused training
    # divergence. Their internal op_router softmaxes and gate sigmoids have no logit
    # regularization while the output head has z-loss+softcap, so training sharpens
    # them without bound. The "delay but don't fix" signature: lowering LR only
    # postpones the blow-up. Off until stabilized and A/B-proven:
    # internal logit-softcap on routers/gates + NALU log-space guard.
    # Opt back in explicitly with --use-math-module / --use-logic-playground / etc.
    #   error_feedback: REMOVED from the codebase (training divergence + inference NaN, no win).
    cfg.use_thermostat = True
    cfg.track_convergence = True
    cfg.use_accel_exit = True
    cfg.sngp_enabled = True
    cfg.sngp_spectral_norm = True
    cfg.sngp_accumulate_train = False
    cfg.laplace_enabled = True
    cfg.per_seq_recurrence = True
    # inference or the halt head / convergence signals read states the training trajectory never
    # produced. Under grad_checkpoint the memory cost is recomputed, not stored.


def make_cfg(args):
    cfg = (
        CharkhaConfig.toy()
        if getattr(args, "toy", False)
        else CharkhaConfig.nano()
        if getattr(args, "nano", False)
        else CharkhaConfig.mini()
        if getattr(args, "mini", False)
        else CharkhaConfig.small()
        if getattr(args, "small", False)
        else CharkhaConfig.medium()
        if getattr(args, "medium", False)
        else CharkhaConfig()
    )
    if getattr(args, "profile", "frontier") == "frontier":
        _apply_frontier_profile(cfg)
    if args.no_recurrence:
        cfg.use_recurrence = False
    if args.no_halting:
        cfg.use_halting = False
    if getattr(args, "nitp", False):
        cfg.use_nitp = True
    if getattr(args, "use_deep_supervision", False):
        cfg.use_deep_supervision = True
    if getattr(args, "cross_loop_consistency", False):
        cfg.cross_loop_consistency = True
    if getattr(args, "use_gdn2", False):
        cfg.use_gdn2 = True
    if getattr(args, "no_gdn2", False):
        cfg.use_gdn2 = False
    if getattr(args, "use_granary", False):
        cfg.use_granary = True
        if getattr(args, "granary_slots", None) is not None:
            cfg.granary_slots = args.granary_slots
    # ── Experimental features ──
    if getattr(args, "use_bipolar_gate", False):
        cfg.use_bipolar_gate = True
    if getattr(args, "use_osdn", False):
        cfg.use_osdn = True
    if getattr(args, "use_mtp_routing", False):
        cfg.use_mtp_routing = True
    if getattr(args, "use_task_rl", False):
        cfg.use_task_rl = True
    if getattr(args, "awr_temperature", None) is not None:
        cfg.awr_temperature = args.awr_temperature
    if getattr(args, "value_weight", None) is not None:
        cfg.value_weight = args.value_weight
    if getattr(args, "task_replay_size", None) is not None:
        cfg.task_replay_size = args.task_replay_size
    if getattr(args, "use_thermostat", False):
        cfg.use_thermostat = True
    if getattr(args, "thermostat_conf_threshold", None) is not None:
        cfg.thermostat_conf_threshold = args.thermostat_conf_threshold
    if getattr(args, "track_convergence", False):
        cfg.track_convergence = True
    if getattr(args, "convergence_mode", None) is not None:
        cfg.convergence_mode = args.convergence_mode
    if getattr(args, "convergence_window", None) is not None:
        cfg.convergence_window = args.convergence_window
    if getattr(args, "use_accel_exit", False):
        cfg.use_accel_exit = True
    if getattr(args, "accel_exit_threshold", None) is not None:
        cfg.accel_exit_threshold = args.accel_exit_threshold
    if getattr(args, "sngp_enabled", False):
        cfg.sngp_enabled = True
    if getattr(args, "sngp_rff_dim", None) is not None:
        cfg.sngp_rff_dim = args.sngp_rff_dim
    if getattr(args, "sngp_scale", None) is not None:
        cfg.sngp_scale = args.sngp_scale
    if getattr(args, "sngp_ridge", None) is not None:
        cfg.sngp_ridge = args.sngp_ridge
    if getattr(args, "sngp_spectral_norm", False):
        cfg.sngp_spectral_norm = True
    if getattr(args, "sngp_accumulate_train", False):
        cfg.sngp_accumulate_train = True
    if getattr(args, "laplace_enabled", False):
        cfg.laplace_enabled = True
    if getattr(args, "use_process_head", False):
        cfg.use_process_head = True
    if getattr(args, "process_weight", None) is not None:
        cfg.process_weight = args.process_weight
    if getattr(args, "halt_granularity", None) is not None:
        cfg.halt_granularity = args.halt_granularity
    if getattr(args, "halt_segment_len", None) is not None:
        cfg.halt_segment_len = args.halt_segment_len
    cfg.grad_checkpoint = args.grad_checkpoint
    if getattr(args, "per_seq_recurrence", False):  # LoopWM per-sequence depth (helps batch>1)
        cfg.per_seq_recurrence = True
    if getattr(args, "ce_chunk", None):
        cfg.ce_chunk = args.ce_chunk
    if getattr(args, "gdn_chunk", None):
        cfg.gdn_chunk = args.gdn_chunk
    if getattr(args, "ce_vchunk", None):
        cfg.ce_vchunk = args.ce_vchunk
    if getattr(args, "rev_bptt", False):
        cfg.rev_bptt = True
    if getattr(args, "rev_anchor", None):
        cfg.rev_anchor = args.rev_anchor
    if getattr(args, "reversible", False):
        cfg.reversible = True
    if getattr(args, "embed_factor", None):
        cfg.embed_factor = args.embed_factor
    if getattr(args, "z_loss", None):
        cfg.z_loss = args.z_loss
    if getattr(args, "logit_softcap", None):
        cfg.logit_softcap = args.logit_softcap
    # generic config override (--set key=value, repeatable): the ablation lever — any config
    # field can be A/B'd from the CLI without a bespoke flag. Values parse as Python literals
    # ('true'/'false' accepted); unknown keys are a hard error, not a silent no-op.
    for kv in getattr(args, "set", None) or []:
        key, _, val = kv.partition("=")
        key = key.strip()
        if not hasattr(cfg, key):
            raise ValueError(f"--set {key}: no such CharkhaConfig field")
        import ast as _ast

        v = val.strip()
        try:
            v = _ast.literal_eval({"true": "True", "false": "False"}.get(v.lower(), v))
        except (ValueError, SyntaxError):
            pass  # keep as string
        setattr(cfg, key, v)
        print(f"[cfg] --set {key} = {v!r}")
    return cfg


def set_lr(opts, bases, mult):
    for opt, base in zip(opts, bases):
        for g in opt.param_groups:
            g["lr"] = base * mult


def head_grad_norms(model):
    """Grad-norm of each auxiliary head's weight AFTER backward (before zero_grad) — a free
    interference probe: shows how much gradient signal each head is pulling, no extra backward."""
    out = {}
    heads = [
        ("embed", getattr(model, "embed", None)),
        ("conf", getattr(model, "conf_head", None)),
        ("halt", getattr(model, "halt_head", None)),
        ("mtp", getattr(model, "mtp_proj", None)),
        ("nitp", getattr(model, "nitp_head", None)),
    ]
    sngp = getattr(model, "sngp_head", None)
    if sngp is not None:
        heads.append(("sngp", getattr(sngp, "beta", None)))
    for name, mod in heads:
        w = getattr(mod, "weight", None) if mod is not None else None
        if w is not None and w.grad is not None:
            out[name] = round(float(w.grad.norm()), 4)
    return out


def optimizer_state_summary(opts):
    """Report where optimizer state tensors live after the first optimizer step."""
    rows = []
    for opt in opts:
        dev_bytes = {}
        for st in opt.state.values():
            for v in st.values():
                if torch.is_tensor(v):
                    dev = v.device.type
                    dev_bytes[dev] = dev_bytes.get(dev, 0) + v.numel() * v.element_size()
        if dev_bytes:
            bits = ", ".join(f"{dev}:{sz / 1024**3:.2f}G" for dev, sz in sorted(dev_bytes.items()))
        else:
            bits = "no state yet"
        tag = opt.__class__.__name__
        if getattr(opt, "cpu_offload", False):
            tag += "(offload)"
        rows.append(f"{tag} {bits}")
    return "; ".join(rows)


# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------


def _gpu_mem():
    """(dedicated_used_G, total_G, shared_spill_G) on CUDA, else None. used = total-free (the TRUE
    dedicated footprint incl. CUDA context); spill = PyTorch reserved beyond dedicated total = the
    amount WDDM pushed into shared system RAM (the silent ~10-50x slowdown)."""
    try:
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        tg = total / 1024**3
        return (total - free) / 1024**3, tg, max(0.0, torch.cuda.memory_reserved() / 1024**3 - tg)
    except Exception:
        return None


def _fmt_dur(s):
    s = int(max(0, s))
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if d:
        return f"{d}d {h}h {m}m"
    if h:
        return f"{h}h {m}m"
    return f"{m}m {s}s"


def _fmt_tok(n):
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}K"
    return str(int(n))


def seed_everything(seed):
    """Deterministic init across processes/boxes. SAME seed -> IDENTICAL initial weights, which is
    exactly what makes two independently-trained checkpoints weight-MERGEABLE: same init keeps them
    in one loss basin (linear mode connectivity), so soup/slerp/ties actually compose. WITHOUT a
    shared seed, each box inits differently and only ensemble->distill (output-space, see bakeoff.py)
    can fuse them. Seeds python / numpy / torch / cuda RNG. (Also the symbol continual.py imports.)"""
    import random as _r

    _r.seed(seed)
    try:
        import numpy as _np

        _np.random.seed(seed % (2**32))
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "ckpt.pt")
    metrics_path = os.path.join(args.out, "metrics.jsonl")
    rng = random.Random(args.seed)
    seed_everything(args.seed)  # deterministic init: same --seed on every box => mergeable ckpts
    print(
        f"[seed] deterministic init from seed={args.seed} "
        f"(use the SAME --seed on every box for weight-mergeable runs; see bakeoff.py)"
    )

    # ---- build fresh, resume in place, or branch a release ----
    decay_start = None
    meta = {"loss_ema": None, "best_val": None}
    resume = args.branch_from or (
        ckpt_path if (args.resume and os.path.exists(ckpt_path)) else None
    )
    if resume:
        model, cfg, opts, bases, ck = load_ckpt(
            resume, device, args=args, offload=args.offload_optim
        )
        opt_mode = ck.get("opt_mode", "default")
        start_step = ck["step"]
        meta = {**meta, **(ck.get("meta") or {})}  # merge: sparse metas (grow_init) keep defaults
        restore_rng(ck, rng)
        # Runtime-tunable knobs (NOT architecture) re-applied from THIS launch's CLI args, not left as
        # whatever the checkpoint happened to save. Without this, resuming a checkpoint trained with
        # different flags (e.g. a bigger-VRAM cloud box using --ce-chunk 1024) silently keeps the OLD
        # box's settings forever -- --ce-chunk 512 on the CLI would be a no-op, quietly using more
        # backward-CE memory than intended and tipping a tight 8GB budget into shared-memory spill.
        cfg.grad_checkpoint = args.grad_checkpoint
        if getattr(args, "ce_chunk", None):
            cfg.ce_chunk = args.ce_chunk
        if getattr(args, "gdn_chunk", None):
            cfg.gdn_chunk = args.gdn_chunk
        # Memory-path knobs (same rationale as ce_chunk): identical weights + math, different
        # train-time memory strategy — always taken from THIS launch's CLI.
        if getattr(args, "ce_vchunk", None):
            cfg.ce_vchunk = args.ce_vchunk
        if getattr(args, "rev_bptt", False):
            cfg.rev_bptt = True
        if getattr(args, "rev_anchor", None):
            cfg.rev_anchor = args.rev_anchor
        if getattr(args, "reversible", False) and not getattr(cfg, "reversible", False):
            # Function-CHANGING on an existing checkpoint (same params, different dataflow):
            # allowed, but announce it loudly — expect a loss bump that must re-heal.
            cfg.reversible = True
            print(
                "[warn] --reversible flipped ON for a checkpoint trained single-stream: "
                "the computed function changes (measure with grow_init.py --to-reversible)."
            )
        tag = "branch" if args.branch_from else "resume"
        print(
            f"[{tag}] loaded {resume} @ step {start_step} (opt={opt_mode}) "
            f"ce_chunk={cfg.ce_chunk} grad_checkpoint={cfg.grad_checkpoint}"
        )
    else:
        cfg = make_cfg(args)
        torch.manual_seed(args.seed)  # re-seed at the construction site (exact init)
        model = Charkha(cfg).to(device)
        model = _maybe_cast_param_dtype(model, args, device)
        opts, bases, opt_mode = build_opt_set(model, args, offload=args.offload_optim)
        start_step = 0

    # ---- data (and reconcile vocab with the shards) ----
    dw = getattr(args, "data_weight", None)
    if dw:
        if len(dw) != len(args.data):
            raise ValueError(
                f"--data-weight count ({len(dw)}) must match --data count ({len(args.data)})"
            )
    train_loader = ShardLoader(args.data, val_frac=args.val_frac, split="train", dir_weights=dw)
    val_loader = (
        ShardLoader(args.data, val_frac=args.val_frac, split="val") if args.val_frac > 0 else None
    )
    ctx_loader = None
    if getattr(args, "retrieval_context_data", None):
        ctx_loader = ShardLoader(args.retrieval_context_data, val_frac=0.0, split="train")
        if ctx_loader.vocab_size != train_loader.vocab_size:
            raise ValueError(
                f"retrieval-context vocab {ctx_loader.vocab_size} != data vocab "
                f"{train_loader.vocab_size}"
            )
        print(
            f"[retrieval-pretrain] context={args.retrieval_context_data} "
            f"frac={args.retrieval_context_frac:g} ctx_tokens={args.retrieval_context_tokens}"
        )
    # tokenizer_name is metadata only (doesn't resize anything) -- set it whether or not a vocab
    # rebuild happens below, so serve.py always knows which tokenizer produced these shards.
    if train_loader.tokenizer_name and not cfg.tokenizer_name:
        cfg.tokenizer_name = train_loader.tokenizer_name
    if cfg.vocab_size < train_loader.vocab_size:
        if start_step > 0:
            raise ValueError(
                f"shard vocab {train_loader.vocab_size} exceeds checkpoint "
                f"vocab {cfg.vocab_size}; tokenizer mismatch"
            )
        cfg.vocab_size = ((train_loader.vocab_size + 127) // 128) * 128
        torch.manual_seed(args.seed)  # re-seed: rebuilt init must match across boxes
        model = Charkha(cfg).to(device)  # rebuild fresh model at the right vocab
        model = _maybe_cast_param_dtype(model, args, device)
        opts, bases, opt_mode = build_opt_set(model, args, offload=args.offload_optim)
        print(
            f"[data] bumped vocab_size -> {cfg.vocab_size} to fit shards "
            f"(shard vocab {train_loader.vocab_size})"
        )

    # ---- grad-release (LOMO-style): backward hooks stream each big-matrix gradient to a pinned
    # CPU accumulator as soon as it is produced, so the full fp32 gradient set never coexists on
    # the GPU (~1.6GB back at 0.42B). Applies to the Muon/NormM groups only — the AdamW group
    # (norms/heads/1D + possibly bnb 8-bit state) keeps the standard GPU path. ----
    if getattr(args, "grad_release", False):
        rel_params = [
            p
            for opt in opts
            if isinstance(opt, (Muon, NormM))
            for grp in opt.param_groups
            for p in grp["params"]
        ]
        n_rel = install_grad_release(rel_params)
        if n_rel:
            rel_bytes = sum(p.numel() for p in rel_params) * 4
            print(
                f"[grad-release] {n_rel} matrices ({rel_bytes / 1e9:.2f}GB fp32 grads) "
                "stream to CPU during backward"
            )
        else:
            print(
                "[grad-release] unavailable on this torch (no post_accumulate_grad_hook); "
                "continuing with standard grads"
            )

    # ---- optional torch.compile: fuse the eager graph for throughput (the binding constraint is
    # tokens/day, not VRAM). Opt-in; save_ckpt unwraps _orig_mod so checkpoints stay portable. Falls
    # back to eager if compile/inductor is unavailable so a run never dies on it. ----
    if getattr(args, "compile", False) and device == "cuda":
        try:
            model = torch.compile(model)
            print("[compile] torch.compile(model) enabled (first step pays a one-time graph cost)")
        except Exception as e:
            print(f"[compile] unavailable ({type(e).__name__}); continuing eager")

    # ---- branch-for-release: linearly decay LR to zero over decay-steps from here ----
    total_steps = args.steps
    if args.branch_from and args.decay_steps:
        decay_start = start_step
        total_steps = start_step + args.decay_steps

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"CHARKHA train | {n_params / 1e6:.1f}M params | device={device} | "
        f"recurrence={cfg.use_recurrence} halting={cfg.use_halting} "
        f"grad_ckpt={cfg.grad_checkpoint} | steps {start_step}->{total_steps} | "
        f"eff.batch {args.batch_size}x{max(1, args.accum_steps)}={args.batch_size * max(1, args.accum_steps)}"
        f" x {args.seq_len}tok | data {train_loader.total:,} tok ({len(train_loader.arrays)} shards) | "
        f"optim={opt_mode} offload={bool(args.offload_optim)} "
        f"adam={'8bit' if getattr(args, 'eightbit_optim', False) else 'fp32'}"
    )

    if getattr(args, "dashboard", False):
        _tps = args.batch_size * max(1, args.accum_steps) * args.seq_len
        _plan = _tps * max(0, total_steps - start_step)
        _mm = _gpu_mem()
        _fla_gdn2 = getattr(charkha_mod, "_HAVE_FLA_GDN2", False) and device == "cuda"
        _mixer = (
            "GDN-2 fla-kernel"
            if (getattr(cfg, "use_gdn2", False) and _fla_gdn2)
            else "GDN-2 exact-pytorch"
            if getattr(cfg, "use_gdn2", False)
            else "GDN-1 fla"
            if (device == "cuda" and have_fla())
            else "GDN-1 fallback"
        )
        print("  " + "=" * 72)
        print(
            f"  >> LIVE DASHBOARD | {n_params / 1e6:.0f}M params | {cfg.n_prelude}+{cfg.n_core}xr+{cfg.n_coda} "
            f"blocks (d={cfg.d_model}, ce_chunk={getattr(cfg, 'ce_chunk', '?')}) | "
            f"mixer={_mixer}"
        )
        print(
            f"  >> plan: {max(0, total_steps - start_step):,} steps x {_tps:,} tok = {_plan / 1e9:.2f}B tok "
            f"| {_plan / max(1, train_loader.total):.1f} epochs over {train_loader.total / 1e9:.2f}B-tok data"
        )
        if _mm:
            print(
                f"  >> vram at start: {_mm[0]:.2f} / {_mm[1]:.1f}G dedicated"
                + (
                    f"   [!] SPILL {_mm[2]:.2f}G already in shared"
                    if _mm[2] > 0.05
                    else "   [OK] no spill"
                )
            )
        print("  " + "=" * 72)

    B, T, accum = args.batch_size, args.seq_len, max(1, args.accum_steps)
    # bf16 autocast needs Ampere+ (Turing/Volta have no hardware bf16). Guard so a resume onto
    # an older GPU degrades to fp32 instead of crashing — slower, but the run survives a hardware swap.
    amp = device == "cuda"
    if device == "cuda":
        # TF32 matmuls: free throughput on Ampere+ for the fp32 paths outside the bf16
        # autocast region (optimizer math, norms, chunk-scan accumulators). ~1e-3 relative
        # precision, well inside training noise.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # conv shapes are fixed -> autotune once, reuse
    if amp and not torch.cuda.is_bf16_supported():
        amp = False
        print(
            "[warn] this GPU lacks bf16 (pre-Ampere); running fp32 (slower, more VRAM). "
            "An Ampere+ card (e.g. the 4060 Ti) is recommended for the real run."
        )
    # Training on CUDA without the fla/Triton kernel means the pure-PyTorch GDN fallback — far
    # slower and numerically distinct from production. Real runs belong in WSL2 with fla installed
    # (the canonical env, see README.md). Loud, not fatal: a deliberate fallback run still works.
    if device == "cuda" and not have_fla():
        print(
            "[warn] flash-linear-attention/triton NOT found — GDN is using the pure-PyTorch "
            "chunked fallback, NOT the production Triton kernel. Throughput will be much lower "
            "and numerics differ. For a real run: WSL2 + `pip install flash-linear-attention "
            "triton` (see scripts/setup_wsl.sh)."
        )

    # ---- optional teacher for soft-label distillation ----
    # Logit KD indexes the STUDENT's embedding rows with the TEACHER's token ids (_kd_topk), so it
    # is only meaningful when both share ONE tokenizer. Shards tokenized with a custom local
    # tokenizer (Sutra-131k, charkha_tokenizer.json) share it with NO stock teacher — the KD term would
    # be silent garbage. Hard-stop unless the operator asserts the teacher matches.
    if getattr(args, "teacher_url", None) or getattr(args, "distill", None):
        _tn = getattr(cfg, "tokenizer_name", None)
        if _tn and str(_tn).endswith(".json") and not getattr(args, "kd_tokenizer_ok", False):
            raise ValueError(
                f"logit KD requested, but the shards were tokenized with a custom tokenizer "
                f"({_tn}) that no stock teacher shares. Serve a teacher built on the SAME "
                f"tokenizer.json and re-launch with --kd-tokenizer-ok, or use sequence-level KD "
                f"(pipeline.py --kd-run) which is tokenizer-agnostic."
            )
    teacher = None
    kd_reader = None
    if getattr(args, "kd_shards", None):
        # accordion come-down: distill from a CACHED teacher (scripts/cache_logits.py)
        # with zero teacher inference — batches come from the KD cache itself so the
        # student's positions always align with the cached top-k.
        sys.path.insert(
            0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
        )
        from cache_logits import KDShardReader

        kd_reader = KDShardReader(args.kd_shards)
        if kd_reader.meta["vocab_size"] != cfg.vocab_size:
            raise SystemExit(
                f"KD cache vocab {kd_reader.meta['vocab_size']} != student vocab {cfg.vocab_size}"
            )
        print(
            f"[distill] cached KD: {kd_reader.n_windows} windows "
            f"(T={kd_reader.meta['seq_len']}, k={kd_reader.meta['topk']}, "
            f"frac={args.kd_shard_frac}) from {args.kd_shards}"
        )

    if getattr(args, "teacher_url", None):
        from distill import RemoteTeacher, teacher_kd_tuple

        teacher = RemoteTeacher(args.teacher_url)
        print(f"[distill] remote teacher at {args.teacher_url}")

    if teacher is None and getattr(args, "distill", None):
        from distill import HFTeacher, teacher_kd_tuple

        t_device = getattr(args, "teacher_device", None) or device
        teacher = HFTeacher(
            args.distill,
            device=t_device,
            k=args.distill_topk,
            temp=args.distill_temp,
            student_vocab=cfg.vocab_size,
            quant=getattr(args, "teacher_quant", None),
        )
        print(
            f"[distill] teacher={args.distill} k={args.distill_topk} temp={args.distill_temp} "
            f"weight={args.distill_weight} device={t_device} quant={getattr(args, 'teacher_quant', None)}"
        )

    # ---- optional frozen fluency prior (residual-logit training accelerant) ----
    # A small pretrained CHARKHA (CharkhaConfig.mini class) whose logits are ADDED to the
    # student's inside the fused CE while the prior's weight anneals to 0: gradient only flows
    # into what the fluency prior cannot already predict, so the big model never spends its
    # slow tokens learning English mechanics. The prior must share the tokenizer/vocab.
    prior_model = None
    if getattr(args, "logit_prior", None):
        pk = torch.load(args.logit_prior, map_location="cpu", weights_only=False)
        pcfg = CharkhaConfig.from_dict(pk["cfg"])
        if pcfg.vocab_size != cfg.vocab_size:
            raise ValueError(
                f"prior vocab {pcfg.vocab_size} != model vocab {cfg.vocab_size} (same tokenizer required)"
            )
        prior_model = Charkha(pcfg).to(device)
        prior_model.load_state_dict(pk["model"])
        prior_model.eval().requires_grad_(False)
        print(
            f"[prior] fluency prior {args.logit_prior} "
            f"({sum(p.numel() for p in prior_model.parameters()) / 1e6:.1f}M params), "
            f"weight {args.prior_weight} annealed to 0 over {args.prior_anneal} steps"
        )
    # ---- EMA of weights (CPU-resident: zero VRAM cost) ----
    # An exponential moving average of the weights consistently evaluates/serves better than
    # the raw last iterate (noise-averaged minimum). Kept in host RAM in fp32 and refreshed
    # every ema_every steps with the equivalent compounded decay; saved under meta['ema'] so
    # serve.py can prefer it. Never used for training itself.
    ema = None
    if getattr(args, "ema", 0):
        ema_model = getattr(model, "_orig_mod", model)
        prev = meta.get("ema")  # resume continues the same average
        if prev:
            prev = {n.removeprefix("_orig_mod."): t for n, t in prev.items()}
        ema = (
            {n: t.clone() for n, t in prev.items()}
            if prev
            else {
                n: p.detach().to("cpu", torch.float32).clone()
                for n, p in ema_model.named_parameters()
            }
        )
        meta["ema_decay"] = args.ema
        print(f"[ema] tracking weight EMA (decay {args.ema}, every {args.ema_every} steps, CPU)")
    model.train()
    log_f = open(metrics_path, "a")
    t0, tokens_done = time.time(), 0
    _win = [t0, 0]  # [last_log_time, last_log_tokens] for windowed tok/s
    _spill_warned = [False]  # one-time loud alarm when VRAM spills to shared RAM
    last_gnorm = 0.0
    # curric_r_end target: explicit flag wins; else the persisted meta value (resume); else
    # cfg.mean_recurrence captured NOW, before the loop mutates model.cfg.mean_recurrence (which
    # aliases cfg and is baked into the ckpt). Persisting it makes a mid-ramp resume keep the
    # original ceiling instead of collapsing it to the partially-ramped value.
    curric_r_end = (
        args.curric_r_end
        if getattr(args, "curric_r_end", None) is not None
        else meta.get("curric_r_end", cfg.mean_recurrence)
    )
    if args.recurrence_curriculum and meta.get("curric_r_end") is None:
        meta["curric_r_end"] = (
            curric_r_end  # persist the ceiling regardless of how the flag is (re)passed
        )
    if args.recurrence_curriculum:
        print(
            f"[curriculum] mean_recurrence {args.curric_r_start} -> {curric_r_end} "
            f"over {args.curric_steps} steps (retrofitted-recurrence)"
        )
    # halter phasing (spec): Phase 1 = fixed Poisson loops (halting OFF) for the first
    # --halt-start-step steps, then enable the configured halter; ponder cost ramps 0->target
    # over --ponder-anneal-steps starting at halt-start. Recomputed from `step` each iter so a
    # single run spans both phases and resumes correctly. halt_target derives from args (not the
    # possibly-baked cfg) so a Phase-1 ckpt still flips halting on in Phase 2 on resume.
    halt_target = cfg.use_recurrence and not args.no_halting
    ponder_target = cfg.ponder_cost
    if args.ponder_anneal_steps > 0:  # persist the target so resume mid-ramp is stable
        if meta.get("ponder_target") is None:
            meta["ponder_target"] = cfg.ponder_cost
        ponder_target = meta["ponder_target"]
    if args.halt_start_step > 0:
        print(
            f"[halter] fixed loops (halting off) until step {args.halt_start_step}, then halter on"
            + (
                f"; ponder 0->{ponder_target:g} over {args.ponder_anneal_steps} steps"
                if args.ponder_anneal_steps > 0
                else ""
            )
        )
    codec = (
        _build_codec(cfg)
        if (getattr(args, "sample_every", 0) or getattr(args, "dashboard", False))
        else None
    )
    if codec:
        print(
            f"[sample] every {args.sample_every} steps: {args.sample_tokens} tok from "
            f"{(args.sample_prompt or '<bos>')!r} (effort={args.sample_effort})"
        )
    pause_flag = os.path.join(args.out, "PAUSE")  # touch this file to pause-and-save anywhere
    signal.signal(signal.SIGTERM, _request_pause)  # `kill -TERM` => graceful pause too
    for step in range(start_step, total_steps):
        # Graceful pause check (top of step, before any work): save an atomic resume point at `step`
        # and exit 0 so the resilient launcher stays down until a resume. Resume re-runs this step.
        if _PAUSE_REQUESTED or os.path.exists(pause_flag):
            save_ckpt(ckpt_path, model, opts, opt_mode, step, cfg, meta, rng)
            print(
                f"[pause] checkpointed @ step {step} -> {ckpt_path}; exiting cleanly (resume to continue)."
            )
            log_f.write(json.dumps({"step": step, "paused": True}) + "\n")
            log_f.flush()
            log_f.close()
            return model, cfg
        mult = wsd_lr_mult(step, args.warmup, decay_start, args.decay_steps)
        set_lr(opts, bases, mult)
        if args.recurrence_curriculum:  # retrofitted-recurrence: anneal E[r] low->high
            prog = min(1.0, step / max(1, args.curric_steps))
            model.cfg.mean_recurrence = max(
                1, int(round(args.curric_r_start + (curric_r_end - args.curric_r_start) * prog))
            )
        # Auto-track BPTT depth = ceil(E[r]/2) so deeper recurrence carries
        # proportional gradient signal. Static backprop_depth (default 2) wastes
        # the first (r-k) loops under no_grad once E[r] exceeds ~3.
        # When --bptt-half is explicitly passed, use that as a signal the user
        # wants the more aggressive ceil(E[r]/2) tracking even without curriculum.
        if args.recurrence_curriculum or getattr(args, "bptt_half", False):
            model.cfg.backprop_depth = max(1, (model.cfg.mean_recurrence + 1) // 2)
        if args.halt_start_step > 0:  # halter phasing: Phase 1 fixed -> Phase 2 halt
            model.cfg.use_halting = halt_target and (step >= args.halt_start_step)
        if args.ponder_anneal_steps > 0:  # ramp the PonderNet penalty in after halt-start
            pp = min(1.0, max(0, step - args.halt_start_step) / max(1, args.ponder_anneal_steps))
            model.cfg.ponder_cost = ponder_target * pp

        # gradient accumulation: average the loss over `accum` micro-batches, one optimizer
        # step per `step`. Effective batch = B * accum - the only way to a sane batch at the
        # B=2 / T=4096 the 8GB budget forces. Grads are summed across micro-batches by autograd;
        # scaling each backward by 1/accum makes the summed grad the mean (matches a B*accum batch).
        lv = 0.0
        for _ in range(accum):
            if (
                ctx_loader is not None
                and T > args.retrieval_context_tokens + 8
                and rng.random() < args.retrieval_context_frac
            ):
                x, y = train_loader.batch_with_context(
                    ctx_loader, B, T, args.retrieval_context_tokens, device, rng, pin=amp
                )
            else:
                x, y = train_loader.batch(B, T, device, rng, pin=amp)
            kd = None
            if kd_reader is not None and rng.random() < args.kd_shard_frac:
                # cached-teacher step: window + top-k come from the KD cache (replay
                # mixing happens on the other 1-frac of micro-batches)
                x, y, kd = kd_reader.batch(
                    B, device, rng, temp=args.distill_temp, weight=args.distill_weight
                )
            elif teacher is not None:  # teacher top-k computed no-grad, off the graph
                kd = teacher_kd_tuple(
                    teacher,
                    x,
                    args.distill_weight,
                    args.distill_temp,
                    max_tokens=getattr(args, "esr_tokens", None),
                    refine=getattr(args, "refine_trajectory", False),
                )
            prior = None
            if prior_model is not None:
                pw = args.prior_weight * max(0.0, 1.0 - step / max(1, args.prior_anneal))
                if pw > 0:
                    with torch.no_grad():  # frozen prior: hidden + head, off the graph
                        php = prior_model.hidden(x, r=1)
                        php, pWp = prior_model._head_hw(php)
                    prior = (php[:, :-1], pWp, pw)
            # cache_enabled=False: with grad_checkpoint, autocast's bf16 weight-cache is populated
            # non-deterministically across the checkpoint forward vs its backward recompute (modules
            # with many Linears — logic/calculus/math — and the error-feedback path trip it), which
            # raises CheckpointError: "Recomputed values ... different metadata". Disabling the cache
            # makes every cast deterministic. Negligible cost vs the recompute itself.
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp, cache_enabled=False):
                _, loss = model(x, y, kd=kd, prior=prior)
            (loss / accum).backward()
            lv += loss.item() / accum
        tokens_done += B * T * accum

        if not math.isfinite(lv):
            # one bad batch must not silently NaN every weight for the rest of a 90-day run.
            print(f"[warn] non-finite loss ({lv}) at step {step}; skipping optimizer step")
            log_f.write(json.dumps({"step": step, "nonfinite_loss": True}) + "\n")
            log_f.flush()
            for o in opts:
                o.zero_grad(set_to_none=True)
            _clear_release_grads(model)
            last_gnorm = float("nan")
            continue

        last_gnorm = (
            float(clip_grads_mixed(model, args.grad_clip))
            if getattr(args, "grad_release", False)
            else float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
        )
        if not math.isfinite(last_gnorm):
            # A NaN/inf can originate in the BACKWARD pass (bf16 overflow, poisoned gradient) with a
            # perfectly finite forward loss, slipping the math.isfinite(lv) guard above. clip_grad_norm_
            # then scales grads by a nan coefficient and opt.step() would write nan into every weight
            # for the rest of a multi-week run. Skip the step like the forward-loss guard does.
            print(
                f"[warn] non-finite grad norm ({last_gnorm}) at step {step}; skipping optimizer step"
            )
            # name the source: which params actually carry the non-finite grad (first few) — a
            # multi-week run must be debuggable from its log alone.
            allbad = [
                n
                for n, p in model.named_parameters()
                if (p.grad is not None and not torch.isfinite(p.grad).all())
                or (getattr(p, "_cpu_grad_ready", False) and not torch.isfinite(p._cpu_grad).all())
            ]
            # report the DEEPEST poisoned params (backward propagates inf to everything earlier
            # in the forward order, so the tail of the list is the boundary near the source)
            bad = allbad[-6:]
            if bad:
                print(f"[warn]   non-finite grads: {len(allbad)} params, deepest: {', '.join(bad)}")
            log_f.write(json.dumps({"step": step, "nonfinite_grad": True, "params": bad}) + "\n")
            log_f.flush()
            for o in opts:
                o.zero_grad(set_to_none=True)
            _clear_release_grads(model)
            continue
        log_now = step % args.log_every == 0 or step == total_steps - 1
        # capture telemetry while grads still exist (before zero_grad), only on log steps
        hgnorms = head_grad_norms(model) if log_now else {}
        loss_parts = (
            {k: round(float(v), 4) for k, v in model._last_loss_parts.items()}
            if log_now and getattr(model, "_last_loss_parts", None)
            else {}
        )
        for o in opts:
            o.step()
        for o in opts:
            o.zero_grad(set_to_none=True)
        if ema is not None and step % max(1, args.ema_every) == 0:
            d = args.ema ** max(1, args.ema_every)  # compounded decay for the skipped steps
            ema_model = getattr(model, "_orig_mod", model)
            with torch.no_grad():
                for n, p in ema_model.named_parameters():
                    ema[n].mul_(d).add_(p.detach().to("cpu", torch.float32), alpha=1 - d)
            meta["ema"] = ema  # serialized with every checkpoint save
        if log_now and step == start_step and getattr(args, "dashboard", False):
            print(f"  [optim-state] {optimizer_state_summary(opts)}")
        meta["loss_ema"] = lv if meta["loss_ema"] is None else 0.98 * meta["loss_ema"] + 0.02 * lv

        if log_now:
            now = time.time()
            el = now - t0
            _wdt, _wtok = now - _win[0], tokens_done - _win[1]  # recent window -> live throughput
            tok_s = (_wtok / _wdt) if (_wdt > 0.05 and _wtok > 0) else (tokens_done / max(el, 1e-6))
            _win[0], _win[1] = now, tokens_done
            done_steps = max(1, step - start_step + 1)
            eta = max(0, total_steps - step - 1) * el / done_steps
            mm = _gpu_mem()
            vram = ""
            if mm:
                vram = f" | vram {mm[0]:.2f}/{mm[1]:.1f}G" + (
                    f" spill {mm[2]:.2f}G" if mm[2] > 0.05 else ""
                )
            parts_str = " ".join(f"{k}:{v:.3f}" for k, v in loss_parts.items())
            r_status = (
                "r-off"
                if not model.cfg.use_recurrence
                else f"r{model.cfg.mean_recurrence}{'h' if model.cfg.use_halting else 'f'}"
            )
            extra = ""
            if getattr(args, "dashboard", False):
                extra = (
                    f" | {100 * (step + 1) / total_steps:.1f}% | seen {_fmt_tok(tokens_done)}"
                    f" | ETA {_fmt_dur(eta)}" + vram
                )
            print(
                f"step {step:6d} | loss {lv:.4f} (ema {meta['loss_ema']:.4f}) | "
                f"lr x{mult:.3f} | gnorm {last_gnorm:.2f} | {r_status} | "
                f"{tok_s:,.0f} tok/s" + (f" | parts {parts_str}" if parts_str else "") + extra
            )
            row = {
                "step": step,
                "loss": round(lv, 4),
                "loss_ema": round(meta["loss_ema"], 4),
                "lr_mult": round(mult, 5),
                "grad_norm": round(last_gnorm, 3),
                "tok_s": round(tok_s),
                "recurrence": bool(model.cfg.use_recurrence),
                "mean_r": model.cfg.mean_recurrence if model.cfg.use_recurrence else 0,
                "halting": bool(model.cfg.use_halting),
            }
            if loss_parts:
                row["loss_parts"] = loss_parts
            if hgnorms:
                row["head_gnorm"] = hgnorms
            log_f.write(json.dumps(row) + "\n")
            log_f.flush()

        if val_loader and args.eval_every and step > start_step and step % args.eval_every == 0:
            nll, ppl, bpt = evaluate(model, val_loader, B, T, device, rng, args.eval_iters)
            improved = meta["best_val"] is None or nll < meta["best_val"]
            if improved:
                meta["best_val"] = nll
                save_ckpt(
                    os.path.join(args.out, "best.pt"),
                    model,
                    opts,
                    opt_mode,
                    step + 1,
                    cfg,
                    meta,
                    rng,
                )
            print(
                f"  [eval] step {step} | val nll {nll:.4f} | ppl {ppl:.2f} | "
                f"bits/tok {bpt:.3f}{'  *best' if improved else ''}"
            )
            if getattr(args, "dashboard", False):
                el = time.time() - t0
                done = step - start_step + 1
                eta = max(0, total_steps - step - 1) * el / max(1, done)
                pct = 100 * (step + 1) / total_steps
                fill = min(20, int(pct / 5))
                bar = "#" * fill + "." * (20 - fill)
                bestppl = (
                    math.exp(meta["best_val"]) if meta["best_val"] is not None else float("nan")
                )
                mm = _gpu_mem()
                print("  +" + "-" * 66)
                print(
                    f"  | [{bar}] {pct:5.1f}%   step {step:,}/{total_steps:,}   "
                    f"elapsed {_fmt_dur(el)}   ETA {_fmt_dur(eta)}"
                )
                print(
                    f"  | loss {lv:.3f} (ema {meta['loss_ema']:.3f})   val ppl {ppl:.2f} "
                    f"(best {bestppl:.2f})   {_fmt_tok(tokens_done)} tok seen   {tokens_done / max(el, 1e-6):,.0f} tok/s"
                )
                if mm:
                    print(
                        f"  | vram {mm[0]:.2f}/{mm[1]:.1f}G dedicated"
                        + (
                            f"   [!] SPILL {mm[2]:.2f}G shared (SLOW - scale down seq_len)"
                            if mm[2] > 0.05
                            else "   [OK] no spill"
                        )
                    )
                if codec:
                    _s = _gen_text(model, codec, args, device, "The ", n=48).replace("\n", " ")[:78]
                    print(f'  | writes: "The |{_s}"')  # see the prose itself improve, live
                    for _ln in _run_milestones(
                        model, codec, args, device, step, meta.setdefault("milestones", {})
                    ):  # can it do 1+1 yet?
                        print(_ln)
                print("  +" + "-" * 66)
            log_f.write(
                json.dumps(
                    {
                        "step": step,
                        "val_nll": round(nll, 4),
                        "val_ppl": round(ppl, 3),
                        "val_bits": round(bpt, 4),
                    }
                )
                + "\n"
            )
            log_f.flush()

        if codec and getattr(args, "sample_every", 0) and step and step % args.sample_every == 0:
            _emit_sample(model, codec, args, device, step)

        if args.save_every and step and step % args.save_every == 0:
            save_ckpt(ckpt_path, model, opts, opt_mode, step + 1, cfg, meta, rng)
        if getattr(args, "snapshot_every", 0) and step and step % args.snapshot_every == 0:
            # immortal, never-overwritten snapshot. ckpt.pt is the rolling resume point; a bad step
            # (silent divergence / NaN weights that slip the loss guard) overwrites it - snapshots
            # are the rewind points. Resume from one with --branch-from <snap> or copy it to ckpt.pt.
            snap = os.path.join(args.out, f"ckpt_{step:08d}.pt")
            save_ckpt(snap, model, opts, opt_mode, step + 1, cfg, meta, rng)
            print(f"  [snapshot] -> {snap}")

    # final eval: the in-loop check can never fire on the last step (range() is exclusive),
    # so short runs with --eval-every >= steps would otherwise end with no val_nll at all.
    if (
        val_loader
        and args.eval_every
        and not ((total_steps - 1) > start_step and (total_steps - 1) % args.eval_every == 0)
    ):
        nll, ppl, bpt = evaluate(model, val_loader, B, T, device, rng, args.eval_iters)
        if meta["best_val"] is None or nll < meta["best_val"]:
            meta["best_val"] = nll
        print(f"  [eval] final | val nll {nll:.4f} | ppl {ppl:.2f} | bits/tok {bpt:.3f}")
        log_f.write(
            json.dumps(
                {
                    "step": total_steps - 1,
                    "val_nll": round(nll, 4),
                    "val_ppl": round(ppl, 3),
                    "val_bits": round(bpt, 4),
                }
            )
            + "\n"
        )
        log_f.flush()
    save_ckpt(ckpt_path, model, opts, opt_mode, total_steps, cfg, meta, rng)
    log_f.close()
    print(f"done @ step {total_steps} | checkpoint -> {ckpt_path}")
    return model, cfg


# --------------------------------------------------------------------------
# Memory probe: report VRAM at each stage so the 8GB budget is measured, not guessed.
# Runs 3 steps because the Muon/Adam state allocates lazily on the FIRST opt.step() -
# i.e. step 1's backward is the real peak, not step 0's.
# --------------------------------------------------------------------------

from _mem_probe import (
    _build_codec,
    _emit_sample,
    _run_milestones,
    _gen_text,
)


def make_synthetic_shards(data_dir, n_shards=3, toks_per=4000):
    """Write a learnable byte-level corpus in dataprep's exact shard format."""
    os.makedirs(data_dir, exist_ok=True)
    text = b"the people build their own tools and learn the shape of their own freedom. "
    shards, total = [], 0
    for s in range(n_shards):
        buf = array.array("H")
        while len(buf) < toks_per:
            buf.extend(text)  # bytes 0..255
            buf.append(256)  # EOS, uint16-safe
        buf = buf[:toks_per]
        fname = f"shard_{s:05d}.bin"
        with open(os.path.join(data_dir, fname), "wb") as f:
            buf.tofile(f)
        shards.append({"file": fname, "tokens": len(buf)})
        total += len(buf)
    with open(os.path.join(data_dir, "index.json"), "w") as f:
        json.dump({"vocab_size": 257, "total_tokens": total, "shards": shards}, f, indent=2)


def _toy_args(**over):
    a = Namespace(
        toy=True,
        small=False,
        no_recurrence=False,
        no_halting=False,
        nitp=False,
        profile="base",
        grad_checkpoint=True,
        seed=0,
        muon_lr=0.02,
        adam_lr=3e-3,
        grad_clip=1.0,
        warmup=10,
        batch_size=8,
        accum_steps=1,
        seq_len=64,
        device="cpu",
        val_frac=0.34,
        eval_every=0,
        eval_iters=5,
        log_every=20,
        save_every=0,
        snapshot_every=0,
        offload_optim=False,
        resume=False,
        branch_from=None,
        decay_steps=0,
        steps=70,
        distill=None,
        distill_weight=0.5,
        distill_temp=2.0,
        distill_topk=64,
        teacher_quant=None,
        teacher_device=None,
        teacher_url=None,
        esr_tokens=None,
        refine_trajectory=False,
        retrieval_context_data=None,
        retrieval_context_frac=0.0,
        retrieval_context_tokens=64,
        use_process_head=False,
        process_weight=0.05,
        halt_granularity="token",
        halt_segment_len=16,
        symmetry_opt=False,
        norm_lr=0.02,
        recurrence_curriculum=False,
        curric_r_start=2,
        curric_r_end=None,
        curric_steps=10,
        halt_start_step=0,
        ponder_anneal_steps=0,
        sample_every=0,
        sample_tokens=16,
        sample_prompt="",
        sample_effort=None,
        sample_temp=0.8,
        sample_top_k=50,
        data=None,
        out=None,
    )
    a.__dict__.update(over)
    return a


def read_metrics(out_dir):
    rows = []
    with open(os.path.join(out_dir, "metrics.jsonl")) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def selftest():
    print("CHARKHA train self-test")
    tmp = tempfile.mkdtemp(prefix="charkha_train_")
    data_dir = os.path.join(tmp, "data")
    run_dir = os.path.join(tmp, "run")
    rel_dir = os.path.join(tmp, "release")
    make_synthetic_shards(data_dir)
    checks = {}

    # --- loader sanity: shapes, dtype, value range, shard isolation ---
    loader = ShardLoader(data_dir, val_frac=0.34, split="train")
    rng = random.Random(1)
    x, y = loader.batch(4, 32, "cpu", rng)
    checks["loader shapes (B,T)"] = tuple(x.shape) == (4, 32) and tuple(y.shape) == (4, 32)
    checks["loader y == x shifted"] = torch.equal(x[:, 1:], y[:, :-1])
    checks["loader ids in range"] = int(x.max()) < loader.vocab_size
    checks["val split held out"] = len(ShardLoader(data_dir, 0.34, "val").arrays) >= 1

    # --- phase 1: train from scratch (grad-checkpointed core) + periodic eval ---
    print("\n[phase 1] train from scratch (grad_checkpoint=True, eval on)")
    train(
        _toy_args(
            steps=70,
            save_every=30,
            snapshot_every=30,
            eval_every=30,
            eval_iters=4,
            out=run_dir,
            data=data_dir,
        )
    )
    m = read_metrics(run_dir)
    losses = [r["loss"] for r in m if "loss" in r]
    first, last = sum(losses[:2]) / 2, sum(losses[-2:]) / 2
    print(f"  loss {first:.3f} -> {last:.3f}  (ln 257 = {math.log(257):.3f})")
    checks["loss drops on real shards"] = last < first - 0.5
    checks["checkpoint written"] = os.path.exists(os.path.join(run_dir, "ckpt.pt"))
    ck0 = torch.load(os.path.join(run_dir, "ckpt.pt"), weights_only=False)
    checks["ckpt step == steps"] = ck0["step"] == 70
    checks["ckpt has optimizer state"] = bool(ck0["opts"][0]["state"]) and bool(
        ck0["opts"][-1]["state"]
    )
    checks["ckpt records opt_mode"] = ck0.get("opt_mode") == "default"
    evals = [r for r in m if "val_nll" in r]
    checks["eval produced val nll/ppl/bits"] = bool(evals) and all(
        math.isfinite(r["val_nll"]) and r["val_ppl"] > 0 and r["val_bits"] > 0 for r in evals
    )
    checks["best checkpoint written"] = os.path.exists(os.path.join(run_dir, "best.pt"))
    # interference probe: per-component loss + per-head grad-norm telemetry must reach the log
    checks["per-component loss telemetry logged"] = any(
        "loss_parts" in r and "ce" in r["loss_parts"] and "conf" in r["loss_parts"] for r in m
    )
    checks["per-head grad-norm telemetry logged"] = any(
        "head_gnorm" in r and "conf" in r["head_gnorm"] for r in m
    )

    # --- phase 2: resume in place; weights must load identically, then advance ---
    print("\n[phase 2] resume +20 steps")
    model_r, cfg_r, _, _, ck_r = load_ckpt(os.path.join(run_dir, "ckpt.pt"), "cpu")
    w_before = model_r.embed.weight.detach().clone()
    train(_toy_args(steps=90, save_every=0, resume=True, out=run_dir, data=data_dir))
    model_r2, _, _, _, ck_r2 = load_ckpt(os.path.join(run_dir, "ckpt.pt"), "cpu")
    checks["resume advanced step"] = ck_r2["step"] == 90
    checks["resume continued training (weights moved)"] = not torch.equal(
        w_before, load_ckpt(os.path.join(run_dir, "ckpt.pt"), "cpu")[0].embed.weight.detach()
    )
    checks["load_ckpt is deterministic"] = torch.equal(
        ck_r["model"]["embed.weight"], model_r.embed.weight.detach()
    )

    # --- phase 3: WSD branch-for-release; LR must decay to ~0 at the end ---
    print("\n[phase 3] branch-for-release (decay LR to 0)")
    train(
        _toy_args(
            branch_from=os.path.join(run_dir, "ckpt.pt"),
            decay_steps=12,
            log_every=4,
            out=rel_dir,
            data=data_dir,
        )
    )
    rm = read_metrics(rel_dir)
    last_mult = [r["lr_mult"] for r in rm if "lr_mult" in r][-1]
    print(f"  final lr multiplier = {last_mult:.4f}")
    checks["branch decays lr to ~0"] = last_mult < 0.15
    relck = torch.load(os.path.join(rel_dir, "ckpt.pt"), weights_only=False)
    checks["release ckpt at decay end"] = relck["step"] == 90 + 12

    # --- phase 4: gradient accumulation path (effective batch = batch_size * accum_steps) ---
    print("\n[phase 4] gradient accumulation (accum_steps=3)")
    acc_dir = os.path.join(tmp, "accum")
    train(_toy_args(steps=60, accum_steps=3, batch_size=4, out=acc_dir, data=data_dir))
    am = [r["loss"] for r in read_metrics(acc_dir) if "loss" in r]
    af, al = sum(am[:2]) / 2, sum(am[-2:]) / 2
    print(f"  accum loss {af:.3f} -> {al:.3f}")
    checks["accum path trains (loss drops)"] = al < af - 0.5
    checks["accum ckpt advanced steps"] = (
        torch.load(os.path.join(acc_dir, "ckpt.pt"), weights_only=False)["step"] == 60
    )

    # --- phase 5: distillation KD path trains (hermetic: ToyTeacher, no HF download) ---
    print("\n[phase 5] soft-label distillation (ToyTeacher top-k KD)")
    from distill import ToyTeacher, teacher_kd_tuple

    dcfg = CharkhaConfig.toy()
    dmodel = Charkha(dcfg)
    dmuon, dadam = build_optimizers(dmodel, muon_lr=0.02, adam_lr=3e-3)
    dloader = ShardLoader(data_dir, val_frac=0.0, split="train")
    if dcfg.vocab_size < dloader.vocab_size:
        dcfg.vocab_size = ((dloader.vocab_size + 127) // 128) * 128
        dmodel = Charkha(dcfg)
        dmuon, dadam = build_optimizers(dmodel, 0.02, 3e-3)
    dteach = ToyTeacher(dcfg.vocab_size, k=8, temp=2.0)
    drng = random.Random(3)
    dlosses = []
    dmodel.train()
    for _ in range(40):
        bx, by = dloader.batch(8, 32, "cpu", drng)
        kd = teacher_kd_tuple(dteach, bx, 0.5, 2.0)
        _, dl = dmodel(bx, by, kd=kd)
        dl.backward()
        torch.nn.utils.clip_grad_norm_(dmodel.parameters(), 1.0)
        dmuon.step()
        dadam.step()
        dmuon.zero_grad(set_to_none=True)
        dadam.zero_grad(set_to_none=True)
        dlosses.append(dl.item())
    df, dlast = sum(dlosses[:3]) / 3, sum(dlosses[-3:]) / 3
    print(f"  KD loss {df:.3f} -> {dlast:.3f}")
    checks["distillation KD loss is finite"] = all(math.isfinite(v) for v in dlosses)
    checks["distillation path trains (loss drops)"] = dlast < df - 0.3

    # --- phase 6: symmetry-compatible optimizer set (4 opts) trains + checkpoints + resumes ---
    print("\n[phase 6] symmetry-opt (Muon + Row/Col-NormM + AdamW)")
    sym_dir = os.path.join(tmp, "sym")
    train(_toy_args(steps=40, symmetry_opt=True, out=sym_dir, data=data_dir))
    sm = [r["loss"] for r in read_metrics(sym_dir) if "loss" in r]
    sf, sl = sum(sm[:2]) / 2, sum(sm[-2:]) / 2
    print(f"  symmetry loss {sf:.3f} -> {sl:.3f}")
    checks["symmetry-opt path trains (loss drops)"] = sl < sf - 0.4
    sck = torch.load(os.path.join(sym_dir, "ckpt.pt"), weights_only=False)
    checks["symmetry ckpt records mode + 4 opts"] = (
        sck.get("opt_mode") == "symmetry" and len(sck["opts"]) == 4
    )
    # resuming a symmetry ckpt must rebuild the 4-opt structure from opt_mode, not the caller's flag
    _sm, _sc, _sopts, _sb, _ = load_ckpt(os.path.join(sym_dir, "ckpt.pt"), "cpu")
    checks["symmetry ckpt resumes as 4 opts"] = len(_sopts) == 4
    train(_toy_args(steps=60, resume=True, symmetry_opt=True, out=sym_dir, data=data_dir))
    checks["symmetry resume advanced step"] = (
        torch.load(os.path.join(sym_dir, "ckpt.pt"), weights_only=False)["step"] == 60
    )

    # --- phase 7: recurrence curriculum anneals mean_recurrence low->high over steps ---
    print("\n[phase 7] recurrence curriculum (E[r] ramps up)")
    cur_dir = os.path.join(tmp, "curric")
    train(
        _toy_args(
            steps=40,
            recurrence_curriculum=True,
            curric_r_start=1,
            curric_r_end=4,
            curric_steps=40,
            log_every=5,
            out=cur_dir,
            data=data_dir,
        )
    )
    rs = [r["mean_r"] for r in read_metrics(cur_dir) if "mean_r" in r]
    print(f"  mean_r {rs[0]} -> {rs[-1]}")
    checks["recurrence curriculum ramps mean_r up"] = len(rs) > 1 and rs[-1] > rs[0]

    # --- phase 8: halter phasing (Phase 1 fixed loops -> Phase 2 halter + ponder anneal) ---
    print("\n[phase 8] halter phasing (fixed loops -> halting on, ponder annealed)")
    hp_dir = os.path.join(tmp, "haltphase")
    train(
        _toy_args(
            steps=40,
            halt_start_step=20,
            ponder_anneal_steps=10,
            log_every=5,
            out=hp_dir,
            data=data_dir,
        )
    )
    hrows = [r for r in read_metrics(hp_dir) if "halting" in r]
    early = [r["halting"] for r in hrows if r["step"] < 20]
    late = [r["halting"] for r in hrows if r["step"] >= 20]
    print(f"  halting Phase1={early} Phase2={late}")
    checks["halter phasing: fixed loops (halting off) in Phase 1"] = len(early) > 0 and not any(
        early
    )
    checks["halter phasing: halter on in Phase 2"] = len(late) > 0 and all(late)

    # --- phase 9: live sampling to console (watchable training) ---
    print("\n[phase 9] live sampling (--sample-every emits a generation)")
    import io
    import contextlib

    sp_dir = os.path.join(tmp, "sample")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        train(
            _toy_args(
                steps=20,
                sample_every=10,
                sample_tokens=8,
                sample_prompt="the ",
                out=sp_dir,
                data=data_dir,
            )
        )
    sample_lines = [ln for ln in buf.getvalue().splitlines() if "[sample @" in ln]
    print(
        f"  emitted {len(sample_lines)} sample line(s); e.g. {sample_lines[-1].strip() if sample_lines else '(none)'}"
    )
    checks["live sampling emits generations"] = len(sample_lines) > 0
    checks["live sampling does not crash run"] = os.path.exists(os.path.join(sp_dir, "ckpt.pt"))

    # --- crash-safety: immortal snapshots written alongside the rolling ckpt + resumable ---
    snaps = [f for f in os.listdir(run_dir) if f.startswith("ckpt_") and f.endswith(".pt")]
    checks["immortal snapshot written (--snapshot-every)"] = len(snaps) > 0
    if snaps:  # a snapshot must be a complete, resumable ckpt
        _sm, _sc, _so, _sb, _sk = load_ckpt(os.path.join(run_dir, sorted(snaps)[0]), "cpu")
        checks["snapshot is a full resumable ckpt"] = _sk.get("step", 0) > 0 and len(_so) >= 2

    # --- optimizer grouping: embeddings/heads on AdamW, hidden matrices on Muon ---
    _m, _c, _opts, _bases, _ = load_ckpt(os.path.join(run_dir, "ckpt.pt"), "cpu")
    muon_ids = {id(p) for grp in _opts[0].param_groups for p in grp["params"]}
    checks["embedding is NOT on Muon"] = id(_m.embed.weight) not in muon_ids
    checks["hidden matrix IS on Muon"] = id(_m.adapter.weight) in muon_ids
    print()
    ok = True
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= bool(passed)
    print("\nSELFTEST", "PASS - train loop survives checkpoint/resume/branch" if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------

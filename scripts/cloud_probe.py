#!/usr/bin/env python3
"""CHARKHA cloud probe — one short GPU session that answers, with real numbers:
   (1) how big a model can this card TRAIN (VRAM ceiling, per size),
   (2) how fast (steady tok/s + estimated MFU vs the card's achievable GEMM peak),
   (3) how many tokens that buys in the GPU-hour budget, and
   (4) does the recurrence earn its FLOP tax (equal-FLOP recurrent-vs-vanilla A/B).

Self-contained: uses synthetic data (random + a structured induction task), so there is no data
dependency and nothing to download. Every measurement is independently guarded — one OOM/error records
a row and the rest continue. Results stream to <out>/probe.json (machine) and the console (human); the
VM wrapper uploads them to GCS and powers the instance off.

  python scripts/cloud_probe.py --out runs/cloud-probe --budget-hours 100 --price 1.00

"""

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
from charkha import Charkha, CharkhaConfig


# --------------------------------------------------------------------------
# Result accumulation: write incrementally so a preemption mid-probe still leaves a usable file.
# --------------------------------------------------------------------------
class Report:
    def __init__(self, path):
        self.path = path
        self.data = {"sections": {}, "log": []}

    def log(self, msg):
        print(msg, flush=True)
        self.data["log"].append(msg)
        self.flush()

    def put(self, section, value):
        self.data["sections"][section] = value
        self.flush()

    def flush(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, default=str)
        os.replace(tmp, self.path)


def gpu_info():
    info = {"cuda": torch.cuda.is_available()}
    if not torch.cuda.is_available():
        return info
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    info.update(
        name=p.name,
        total_gib=round(total / 1024**3, 2),
        free_gib=round(free / 1024**3, 2),
        sm=f"{p.major}.{p.minor}",
        torch=torch.__version__,
        bf16=torch.cuda.is_bf16_supported(),
    )
    try:
        import triton

        info["triton"] = triton.__version__
    except Exception:
        info["triton"] = None
    try:
        import fla

        info["fla"] = getattr(fla, "__version__", "present")
    except Exception:
        info["fla"] = None
    return info


def gemm_peak_tflops(n=8192, iters=30):
    """Achievable bf16 matmul throughput — the realistic ceiling to compute MFU against (better than a
    spec sheet number). Returns TFLOP/s."""
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(5):
        c = a @ b
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize()
    dt = time.time() - t0
    del a, b, c
    torch.cuda.empty_cache()
    return (2 * n**3 * iters / dt) / 1e12


# --------------------------------------------------------------------------
# Config sizing. Scale d_model (head_dim fixed 64 -> n_heads = d/64, GQA 4:1) and d_ff ~2.75x.
# --------------------------------------------------------------------------
def apply_profile(cfg, profile):
    if profile == "base":
        return cfg
    if profile != "frontier":
        raise ValueError(f"unknown profile: {profile}")
    try:
        from train import _apply_frontier_profile

        _apply_frontier_profile(cfg)
    except Exception:
        # Keep this script usable even if train.py is refactored: mirror the current
        # frontier profile without importing the training CLI.
        cfg.use_gdn2 = True
        cfg.use_nitp = True
        cfg.use_deep_supervision = True
        cfg.cross_loop_consistency = True
        cfg.use_osdn = True
        cfg.use_bipolar_gate = True
        cfg.use_mtp_routing = True
        cfg.use_task_rl = True
        cfg.use_thermostat = True
        cfg.track_convergence = True
        cfg.use_accel_exit = True
        cfg.sngp_enabled = True
        cfg.sngp_spectral_norm = True
        cfg.sngp_accumulate_train = False
        cfg.laplace_enabled = True
        cfg.per_seq_recurrence = True
        cfg.reasoning_train_final_only = (
            False  # train/infer-consistent schedule (see charkha._model)
        )
    return cfg


def sized_cfg(d_model, n_layers_scale=1.0, profile="frontier", **over):
    cfg = CharkhaConfig()
    cfg.d_model = d_model
    cfg.n_heads = max(1, d_model // 64)
    cfg.n_kv_heads = max(1, cfg.n_heads // 4)
    cfg.d_ff = int(round(d_model * 2.75 / 64) * 64)
    if n_layers_scale != 1.0:
        cfg.n_prelude = max(1, int(round(cfg.n_prelude * n_layers_scale)))
        cfg.n_core = max(1, int(round(cfg.n_core * n_layers_scale)))
        cfg.n_coda = max(1, int(round(cfg.n_coda * n_layers_scale)))
    for k, v in over.items():
        setattr(cfg, k, v)
    cfg.max_seq_len = max(getattr(cfg, "max_seq_len", 4096), 4096)
    apply_profile(cfg, profile)
    return cfg


def build_optimizer(model):
    """Use the real CHARKHA optimizer set if importable (realistic VRAM), else AdamW."""
    try:
        from types import SimpleNamespace
        from train import build_opt_set

        a = SimpleNamespace(
            muon_lr=0.02,
            adam_lr=3e-3,
            norm_lr=0.02,
            symmetry_opt=True,
            eightbit_optim=True,
            offload_optim=False,
            grad_clip=1.0,
        )
        opts, _bases, mode = build_opt_set(model, a, offload=False)
        return opts, mode
    except Exception as e:
        return [
            torch.optim.AdamW(model.parameters(), lr=3e-4)
        ], f"adamw-fallback ({type(e).__name__})"


def flops_per_token(cfg, model):
    """Training FLOPs/token estimate: 2 FLOP per param-MAC, fwd over (non-core once + core×E[r]),
    ×3 for fwd+bwd. Ignores grad-checkpoint recompute and the aux reasoning modules, so it is a
    LOWER bound (true MFU is a bit higher). Good enough to rank configs."""
    core = sum(p.numel() for p in model.core.parameters())
    tot = sum(p.numel() for p in model.parameters())
    noncore = tot - core
    r = max(1, cfg.mean_recurrence)
    return 6 * (noncore + core * r)


def measure(cfg, B, T, grad_ckpt, steps=8):
    """One (size,batch,seq,grad_ckpt) point. Returns dict or raises (OOM caught by caller)."""
    cfg.grad_checkpoint = grad_ckpt
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = Charkha(cfg).to("cuda")
    model.train()
    opts, opt_mode = build_optimizer(model)
    params = sum(p.numel() for p in model.parameters())
    fpt = flops_per_token(cfg, model)

    def one():
        x = torch.randint(0, cfg.vocab_size, (B, T), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            _, loss = model(x, x)
        loss.backward()
        for o in opts:
            o.step()
        for o in opts:
            o.zero_grad(set_to_none=True)

    one()
    one()  # warmup: lazy optimizer state alloc
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        one()
    torch.cuda.synchronize()
    dt = time.time() - t0
    tok_s = steps * B * T / dt
    free, total = torch.cuda.mem_get_info()
    res = {
        "params_M": round(params / 1e6, 1),
        "batch": B,
        "seq": T,
        "grad_ckpt": grad_ckpt,
        "tok_s": round(tok_s),
        "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
        "ded_used_gib": round((total - free) / 1024**3, 2),
        "flops_per_tok": fpt,
        "opt": opt_mode,
    }
    model = None
    opts = None
    torch.cuda.empty_cache()
    return res


def max_batch(cfg, T, grad_ckpt, cap=64):
    """Doubling search for the largest power-of-two batch that fits (no spill/OOM)."""
    best = None
    B = 1
    while B <= cap:
        try:
            r = measure(cfg, B, T, grad_ckpt, steps=4)
            best = r
            B *= 2
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            torch.cuda.empty_cache()
            if "out of memory" not in str(e).lower():
                raise
            break
    return best


def recurrence_ab(rep, vocab=256, T=256, steps_r=120, d_model=512, mean_r=4):
    """Equal-FLOP A/B on a structured INDUCTION task (2nd half = copy of 1st half). Recurrent (E[r]=r)
    vs non-recurrent get the SAME training FLOPs (vanilla runs proportionally more steps), then we
    compare loss on a held-out induction batch. Directional only (synthetic task, short) — the final
    call should rerun on real data — but it tells us if depth-recurrence is paying for its tax."""

    def induction(B):
        half = T // 2
        x = torch.randint(0, vocab, (B, T), device="cuda")
        x[:, half : 2 * half] = x[:, :half].clone()  # second half copies the first
        return x

    def train_eval(recurrent):
        cfg = sized_cfg(
            d_model,
            vocab_size=vocab,
            mean_recurrence=(mean_r if recurrent else 1),
            max_recurrence_train=(mean_r * 2 if recurrent else 1),
            use_recurrence=recurrent,
            use_halting=False,
            grad_checkpoint=False,
        )
        torch.manual_seed(0)
        model = Charkha(cfg).to("cuda")
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        fpt = flops_per_token(cfg, model)
        B = 16
        # equal FLOPs: scale steps inversely to per-token cost
        base = recurrence_ab._base_fpt
        if base is None:
            recurrence_ab._base_fpt = fpt
            base = fpt
        steps = max(8, int(round(steps_r * base / fpt)))
        for _ in range(steps):
            x = induction(B)
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                _, loss = model(x, x)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            xe = induction(32)
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                _, l = model(xe, xe)
        out = {
            "recurrent": recurrent,
            "mean_r": cfg.mean_recurrence,
            "steps": steps,
            "flops_per_tok": fpt,
            "eval_loss": round(float(l), 4),
        }
        del model, opt
        torch.cuda.empty_cache()
        return out

    recurrence_ab._base_fpt = None
    nr = train_eval(False)  # establishes the FLOP baseline (more steps)
    rc = train_eval(True)
    verdict = (
        "recurrence WINS at equal FLOPs"
        if rc["eval_loss"] < nr["eval_loss"] - 1e-3
        else "recurrence does NOT beat vanilla at equal FLOPs (directional)"
    )
    return {"vanilla": nr, "recurrent": rc, "verdict": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/cloud-probe")
    ap.add_argument("--gcs", default=None, help="(informational) GCS dest the wrapper uploads to")
    ap.add_argument("--budget-hours", type=float, required=True)
    ap.add_argument("--price", type=float, required=True)
    ap.add_argument("--seqs", default="2048,4096")
    ap.add_argument("--dmodels", default="1280,1536,1792,2048")  # ~0.45B .. ~2B
    ap.add_argument(
        "--profile",
        choices=["frontier", "base"],
        default="frontier",
        help="frontier = safe default feature hooks; base = plain CharkhaConfig",
    )
    args = ap.parse_args()
    os.environ["CHARKHA_PROBE_OUT"] = args.out
    os.makedirs(args.out, exist_ok=True)
    rep = Report(os.path.join(args.out, "probe.json"))

    rep.put(
        "meta",
        {
            "budget_hours": args.budget_hours,
            "price_per_hr": args.price,
            "profile": args.profile,
            "gcs": args.gcs,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    rep.put("gpu", gpu_info())
    rep.log(f"GPU: {rep.data['sections']['gpu']}")
    if not torch.cuda.is_available():
        rep.log("NO CUDA — aborting.")
        return 1
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 1) achievable GEMM peak -> MFU denominator
    try:
        peak = gemm_peak_tflops()
        rep.put("gemm_peak_tflops", round(peak, 1))
        rep.log(f"achievable bf16 GEMM peak: {peak:.1f} TFLOP/s")
    except Exception as e:
        peak = None
        rep.put("gemm_peak_tflops", f"ERROR: {e}")
        rep.log(f"gemm peak failed: {e}")

    # 2) size x seq x grad_ckpt sweep: max batch + tok/s + MFU + affordable tokens
    seqs = [int(s) for s in str(args.seqs).split(",")]
    dmodels = [int(s) for s in str(args.dmodels).split(",")]
    budget_s = args.budget_hours * 3600 * 0.85  # 15% off for evals/ckpt/preemption
    sweep = []
    for d in dmodels:
        for T in seqs:
            for gc in (False, True):
                tag = f"d{d} seq{T} ckpt{int(gc)}"
                try:
                    cfg = sized_cfg(d, profile=args.profile)
                    best = max_batch(cfg, T, gc)
                    if best is None:
                        rep.log(f"  [{tag}] OOM even at batch=1")
                        continue
                    # remeasure the winning batch a bit longer for a stable tok/s
                    cfg = sized_cfg(d, profile=args.profile)
                    r = measure(cfg, best["batch"], T, gc, steps=10)
                    if peak:
                        r["mfu"] = round(100 * r["flops_per_tok"] * r["tok_s"] / (peak * 1e12), 1)
                    r["affordable_tokens_B"] = round(r["tok_s"] * budget_s / 1e9, 1)
                    r["chinchilla_tokens_B"] = round(20 * r["params_M"] / 1e3, 1)
                    r["overtrain_x"] = round(
                        r["affordable_tokens_B"] / max(r["chinchilla_tokens_B"], 1e-9), 2
                    )
                    sweep.append(r)
                    rep.put("sweep", sweep)
                    rep.log(
                        f"  [{tag}] {r['params_M']}M  maxB={r['batch']}  {r['tok_s']:,} tok/s  "
                        f"{r['ded_used_gib']}G  MFU={r.get('mfu', '?')}%  "
                        f"afford={r['affordable_tokens_B']}B ({r['overtrain_x']}x Chinchilla)"
                    )
                except Exception as e:
                    rep.log(f"  [{tag}] ERROR {type(e).__name__}: {str(e).splitlines()[0][:80]}")
                    rep.data["sections"].setdefault("errors", []).append(f"{tag}: {e}")
                    torch.cuda.empty_cache()

    # 3) equal-FLOP recurrence A/B
    try:
        ab = recurrence_ab(rep)
        rep.put("recurrence_ab", ab)
        rep.log(
            f"recurrence A/B: vanilla loss {ab['vanilla']['eval_loss']} vs "
            f"recurrent loss {ab['recurrent']['eval_loss']}  ->  {ab['verdict']}"
        )
    except Exception as e:
        rep.put("recurrence_ab", f"ERROR: {e}")
        rep.log(f"recurrence A/B failed: {type(e).__name__}: {e}\n{traceback.format_exc()[:500]}")

    # 4) recommendation
    try:
        ok = [r for r in sweep if r.get("mfu") is not None]
        if ok:
            # prefer configs that can hit >=20x Chinchilla (over-trained) with highest MFU
            good = [r for r in ok if r["overtrain_x"] >= 1.0] or ok
            best = max(good, key=lambda r: (r["overtrain_x"] >= 1.0, r["params_M"], r["mfu"]))
            rep.put(
                "recommendation",
                {
                    "pick": f"{best['params_M']}M params, seq {best['seq']}, batch {best['batch']}, "
                    f"grad_ckpt={best['grad_ckpt']}",
                    "tok_s": best["tok_s"],
                    "mfu_pct": best.get("mfu"),
                    "affordable_tokens_B": best["affordable_tokens_B"],
                    "overtrain_vs_chinchilla": best["overtrain_x"],
                    "note": "largest size still >=1x Chinchilla-optimal tokens within budget, max MFU; "
                    "spend slack on MORE TOKENS not more params.",
                },
            )
            rep.log(
                f"RECOMMENDATION: {rep.data['sections']['recommendation']['pick']}  "
                f"({best['affordable_tokens_B']}B tokens affordable, {best['overtrain_x']}x Chinchilla)"
            )
    except Exception as e:
        rep.log(f"recommendation step failed: {e}")

    rep.put("finished", time.strftime("%Y-%m-%d %H:%M:%S"))
    rep.log("PROBE COMPLETE.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        # last-resort: never crash without leaving a trace the wrapper can upload
        try:
            out = os.environ.get("CHARKHA_PROBE_OUT", "runs/cloud-probe")
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "FATAL.txt"), "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
        except Exception:
            pass
        traceback.print_exc()
        raise SystemExit(3)

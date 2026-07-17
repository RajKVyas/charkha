#!/usr/bin/env python3
"""CHARKHA feature ablation harness — the "prove it" gate for the unified config.

Runs short A/B training bursts at the REAL validated 8GB launch recipe (full model,
factorized embedding, seq 512, real v8 shards) — a baseline arm and one arm per candidate
feature — and renders a verdict table. Verdicts drive the all-in-or-remove decision:

  HARMFUL   non-finite gradients, EMA divergence, or val NLL clearly worse -> remove
  COSTLY    >10% tok/s hit without a loss win -> remove (throughput is capability here)
  NEUTRAL   within noise of baseline -> owner's call (default: keep OFF the unified config)
  HELPFUL   val NLL / loss EMA better beyond seed noise -> promote to always-on

Honest scope: an N-step burst detects harm, cost, and early-loss deltas. It cannot prove a
long-horizon win; anything promoted here still rides under the day-3 elasticity gate.

  python scripts/ablate.py --data data/corpus-dd --steps 80 --seeds 2
  python scripts/ablate.py --data data/corpus-dd --features use_bipolar_gate,use_subconscious

"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Candidate arms: each FLIPS one feature from its shipping (frontier-profile) default. --set is
# applied AFTER _apply_frontier_profile, so an arm that re-sets an already-on feature is a no-op
# (the original bipolar arm tested nothing — frontier already turns it on). Encode the flip against
# the actual default:
#   off-by-default features -> ADD arm  (does turning it ON help?)
#   on-by-default  features -> DROP arm (does removing it hurt? -> the all-in-or-remove question)
CANDIDATES = {
    # ADD arms — frontier leaves these off
    "add:subconscious": ["use_subconscious=true"],
    # DROP arms — frontier turns these on; does the model regress without them?
    "drop:bipolar_gate": ["use_bipolar_gate=false"],
    "drop:osdn": ["use_osdn=false"],
    "drop:mtp_routing": ["use_mtp_routing=false"],
    "drop:nitp": ["use_nitp=false"],
    "drop:thermostat": ["use_thermostat=false"],
}

LAUNCH = [
    "--profile",
    "frontier",
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
    "--embed-factor",
    "256",
    "--warmup",
    "20",
    "--val-frac",
    "0.02",
]


def run_arm(name, data_dirs, steps, seed, sets, out_root):
    out = os.path.join(out_root, f"{name}-s{seed}")
    cmd = [
        sys.executable,
        os.path.join(ROOT, "src", "train.py"),
        *LAUNCH,
        "--steps",
        str(steps),
        "--seed",
        str(seed),
        "--out",
        out,
        "--eval-every",
        str(steps),
        "--save-every",
        "0",
    ]
    for d in data_dirs:
        cmd += ["--data", os.path.abspath(d)]
    for kv in sets:
        cmd += ["--set", kv]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=os.path.join(ROOT, "src"), capture_output=True, text=True)
    rows = []
    mpath = os.path.join(out, "metrics.jsonl")
    if os.path.exists(mpath):
        for line in open(mpath):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    emas = [x["loss_ema"] for x in rows if x.get("loss_ema") is not None]
    gnorms = [x["grad_norm"] for x in rows if x.get("grad_norm") is not None]
    toks = [x["tok_s"] for x in rows if x.get("tok_s")]
    vals = [x["val_nll"] for x in rows if x.get("val_nll") is not None]
    return {
        "arm": name,
        "seed": seed,
        "exit": r.returncode,
        "wall_s": round(time.time() - t0),
        "ema_first": round(emas[0], 3) if emas else None,
        "ema_last": round(emas[-1], 3) if emas else None,
        "max_gnorm": round(max(gnorms), 1) if gnorms else None,
        "nonfinite": sum(1 for x in rows if x.get("nonfinite_loss") or x.get("nonfinite_grad")),
        "tok_s": round(sum(toks) / len(toks)) if toks else None,
        "val_nll": round(vals[-1], 4) if vals else None,
        "tail": r.stdout[-400:] if r.returncode != 0 else "",
    }


def verdict(base_arms, feat_arms):
    def agg(arms, k):
        v = [a[k] for a in arms if a.get(k) is not None]
        return sum(v) / len(v) if v else None

    def band(arms, k, floor):
        # seed-noise band = spread of the BASELINE arms on metric k. Every verdict must clear
        # this, not a hardcoded epsilon — at 80 steps/2 seeds val_nll alone spans ~3 points, so
        # a fixed 0.01 threshold cries HELPFUL/HARMFUL on pure noise.
        v = [a[k] for a in arms if a.get(k) is not None]
        return max((max(v) - min(v)) if len(v) > 1 else floor, floor)

    if any(a["exit"] != 0 for a in feat_arms) or any(a["nonfinite"] > 0 for a in feat_arms):
        return "HARMFUL (crash/non-finite)"
    b_ema, f_ema = agg(base_arms, "ema_last"), agg(feat_arms, "ema_last")
    b_tok, f_tok = agg(base_arms, "tok_s"), agg(feat_arms, "tok_s")
    b_val, f_val = agg(base_arms, "val_nll"), agg(feat_arms, "val_nll")
    ema_band, val_band = band(base_arms, "ema_last", 0.05), band(base_arms, "val_nll", 0.15)
    # instability check: a single seed diverging far past the worst baseline seed is a red flag
    # even when the OTHER seed masks it in the mean (e.g. neuromod: one seed val 15, one val 50).
    b_vals = [a["val_nll"] for a in base_arms if a.get("val_nll") is not None]
    f_vals = [a["val_nll"] for a in feat_arms if a.get("val_nll") is not None]
    if b_vals and f_vals and max(f_vals) > max(b_vals) + 3 * val_band:
        return f"HARMFUL (unstable: worst seed val {max(f_vals):.1f} vs baseline worst {max(b_vals):.1f})"
    if f_ema is not None and b_ema is not None and f_ema > b_ema + 3 * ema_band:
        return f"HARMFUL (ema +{f_ema - b_ema:.2f} vs noise {ema_band:.2f})"
    if b_tok and f_tok and f_tok < 0.9 * b_tok:
        loss_win = (f_val is not None and b_val is not None and f_val < b_val - val_band) or (
            f_ema is not None and b_ema is not None and f_ema < b_ema - ema_band
        )
        if not loss_win:
            return f"COSTLY ({f_tok}/{b_tok} tok/s, no loss win)"
    # a real val win must clear the baseline seed spread, not a token epsilon
    if f_val is not None and b_val is not None:
        if f_val < b_val - val_band:
            return f"HELPFUL (val {b_val:.2f} -> {f_val:.2f}, band {val_band:.2f})"
        if f_val > b_val + val_band:
            return f"HARMFUL (val {b_val:.2f} -> {f_val:.2f}, band {val_band:.2f})"
    if f_ema is not None and b_ema is not None and f_ema < b_ema - ema_band:
        return f"HELPFUL (ema {b_ema:.2f} -> {f_ema:.2f}, band {ema_band:.2f})"
    return "NEUTRAL (within seed noise)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--features", default=None, help="comma list (default: all candidates)")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "ablate"))
    a = ap.parse_args()
    feats = a.features.split(",") if a.features else list(CANDIDATES)
    for f in feats:
        assert f in CANDIDATES, f"unknown candidate {f}; add it to CANDIDATES"
    os.makedirs(a.out, exist_ok=True)
    # throwaway warm-up: the first run on a cold card reads 5-10% slower (clocks/caches), which
    # biased the first battery's tok/s comparison — baseline ran first and looked slowest.
    print("[ablate] warm-up burst (discarded) ...", flush=True)
    run_arm("warmup", a.data, max(10, a.steps // 8), 21, [], a.out)
    results = {"baseline": []}
    for seed in range(21, 21 + a.seeds):
        print(f"[ablate] baseline seed {seed} ...", flush=True)
        results["baseline"].append(run_arm("baseline", a.data, a.steps, seed, [], a.out))
        print(f"         {results['baseline'][-1]}", flush=True)
    for f in feats:
        results[f] = []
        for seed in range(21, 21 + a.seeds):
            print(f"[ablate] {f} seed {seed} ...", flush=True)
            results[f].append(run_arm(f, a.data, a.steps, seed, CANDIDATES[f], a.out))
            print(f"         {results[f][-1]}", flush=True)
    print("\n===== ABLATION VERDICTS =====")
    table = {}
    for f in feats:
        table[f] = verdict(results["baseline"], results[f])
        print(f"  {f:24s} {table[f]}")
    json.dump(
        {"results": results, "verdicts": table},
        open(os.path.join(a.out, "verdicts.json"), "w"),
        indent=1,
    )
    print(f"\nwrote {os.path.join(a.out, 'verdicts.json')}")


if __name__ == "__main__":
    main()

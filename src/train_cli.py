#!/usr/bin/env python3
"""CHARKHA training CLI — thin entry point for src/train.py."""

import argparse
import sys
from train import mem_probe, selftest, train


def main():
    try:
        # line_buffering=True: stdout is often captured by train_resilient.sh, so without it
        # Python block-buffers (~8KB) and train.log/the live dashboard
        # only update in bursts on buffer-fill or process exit, not per print - looks "stuck" for an
        # entire run even though training is progressing (metrics.jsonl is separately .flush()ed, so
        # it stayed live; this makes the human-readable log live too).
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except (AttributeError, ValueError):
        pass
    p = argparse.ArgumentParser(description="CHARKHA train - production runner")
    p.add_argument("--selftest", action="store_true", help="CPU end-to-end regression test")
    p.add_argument("--out", type=str, default="runs/charkha", help="run dir (ckpt + metrics)")
    p.add_argument(
        "--data",
        type=str,
        action="append",
        default=None,
        help="dir with index.json + shard_*.bin (dataprep output). Repeat for multiple dirs.",
    )
    p.add_argument(
        "--mini",
        action="store_true",
        help="~128M fluency-proxy config (d_model 640, same depth as the main model) — "
        "the grow-init / logit-prior source model (see CharkhaConfig.mini(), "
        "scripts/grow_init.py). Not for serving; a pretraining-speed stepping stone.",
    )
    p.add_argument(
        "--nano",
        action="store_true",
        help="~30M ladder-seed config (d_model 512; doubles exactly to small then 0.97B)",
    )
    p.add_argument(
        "--use-granary",
        action="store_true",
        help="STAGED: product-key memory layer, exact no-op at init (gate 0)",
    )
    p.add_argument(
        "--granary-slots",
        type=int,
        default=None,
        help="granary value slots (default from config; rounded to a square)",
    )
    p.add_argument(
        "--small", action="store_true", help="~0.3B config (d_model 1024) - more 8GB headroom"
    )
    p.add_argument(
        "--medium",
        action="store_true",
        help="~0.9B config (d_model 1536) - train on a 12-20GB card; still serves on 8GB (bf16 ~1.8GB)",
    )
    p.add_argument(
        "--profile",
        choices=["base", "frontier"],
        default="frontier",
        help="fresh-run architecture/training profile. frontier is the default and enables "
        "the safe default experimental stack in cfg/checkpoints. base is the minimal ablation. "
        "Ignored by resumed checkpoint architecture except for runtime knobs.",
    )
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--accum-steps",
        type=int,
        default=1,
        help="gradient accumulation micro-batches (effective batch = batch-size * accum-steps)",
    )
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument(
        "--ce-chunk",
        type=int,
        default=None,
        help="rows per fused-CE chunk (default cfg=2048). Lower (e.g. 512/256) trims the "
        "backward memory spike at long seq-len, at a small speed cost.",
    )
    p.add_argument(
        "--gdn-chunk",
        type=int,
        default=None,
        help="no-triton GDN scan chunk size (default cfg=32). Irrelevant when fla/triton "
        "is active (the real kernel ignores it).",
    )
    p.add_argument(
        "--embed-factor",
        type=int,
        default=None,
        help="factorized tied embedding rank (e.g. 256): codes (V,f) + up (f,d) replace "
        "the dense (V,d) table — ~134M params / ~1.1GB train-VRAM back at V=131072. "
        "FRESH RUNS ONLY (changes the state_dict); resumed checkpoints keep their "
        "baked-in embedding shape.",
    )
    p.add_argument("--warmup", type=int, default=250)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--adam-lr", type=float, default=3e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile(model) for throughput (CUDA only; ckpts stay portable). "
        "The fla GDN kernels are wrapped in torch.compiler.disable "
        "(see charkha._modules), so inductor cannot miscompile their Triton backward."
        "(it used to reschedule ops in fla/ops/gdn2/chunk_bwd.py, giving up to 30%% "
        "mixer-grad error). The mixer runs eager; the rest of the graph compiles. "
        "Validate grad-equivalence with scripts/check_compile_grads.py before a long run.",
    )
    p.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="checkpoint the recurrent core (needed at 0.4B on 8GB)",
    )
    p.add_argument(
        "--param-bf16",
        action="store_true",
        help="store trainable GPU parameters in bf16. This roughly halves resident weight VRAM; "
        "grad-release and offloaded Muon/NormM keep fp32 CPU grad/momentum buffers.",
    )
    p.add_argument(
        "--grad-release",
        action="store_true",
        help="LOMO-style: stream each big-matrix gradient to a pinned CPU accumulator "
        "during backward (Muon/NormM groups), so the full fp32 grad set (~1.6GB "
        "at 0.42B) never sits on the GPU. Composes with --offload-optim/--accum-steps; "
        "identical math (validated by the grad-release selftest).",
    )
    p.add_argument(
        "--ce-vchunk",
        type=int,
        default=None,
        help="CCE-style vocab streaming for the fused CE (arXiv:2411.09009 in spirit): "
        "chunk the VOCAB axis at this size so the (rows, V) logit block never "
        "materializes — kills the ~0.5GB backward spike at V=131072. Try 8192. "
        "0/unset = whole-vocab row chunks (existing path). Math identical.",
    )
    p.add_argument(
        "--rev-bptt",
        action="store_true",
        help="reversible-recurrence BPTT for GDN-2: backward reconstructs the scan "
        "states by closed-form inversion instead of storing them (novel; see "
        "_RevGDN2Scan). Weights untouched — safe across resumes. Uses the exact "
        "pytorch scan (not the fla kernel) for the mixers it covers.",
    )
    p.add_argument(
        "--rev-anchor",
        type=int,
        default=None,
        help="rev-bptt: store an exact anchor state every K chunks (default cfg=8) to "
        "bound inversion float error.",
    )
    p.add_argument(
        "--reversible",
        action="store_true",
        help="two-stream reversible residual coupling over the block stacks: O(1) "
        "activation memory in depth. Same parameter set, DIFFERENT function — "
        "flipping it on an existing checkpoint perturbs outputs (measure first "
        "with grow_init.py --to-reversible).",
    )
    p.add_argument(
        "--8bit-optim",
        dest="eightbit_optim",
        action="store_true",
        help="use bitsandbytes 8-bit AdamW for non-matrix weights (saves VRAM on 8GB cards)",
    )
    p.add_argument(
        "--no-recurrence", action="store_true", help="ablation: plain stack (no depth recurrence)"
    )
    p.add_argument(
        "--no-halting", action="store_true", help="ablation: fixed Poisson loops, no learned halter"
    )
    p.add_argument(
        "--nitp",
        action="store_true",
        help="enable NITP representation-space aux loss (arXiv:2605.24956); small train cost, none at inference",
    )
    p.add_argument(
        "--use-deep-supervision",
        action="store_true",
        help="A3: anytime deep supervision — supervise earlier core loops through coda+head "
        "(makes every effort level a valid answer). 0 params, +~2 coda passes/step.",
    )
    p.add_argument(
        "--cross-loop-consistency",
        action="store_true",
        help="A1: cross-loop consistency regularizer — maximize cosine similarity between "
        "consecutive core states so recurrence contracts toward a fixed direction. 0 params.",
    )
    p.add_argument(
        "--use-gdn2",
        action="store_true",
        help="force GatedDeltaNet-2 mixer on (default). Kept for old launch scripts.",
    )
    p.add_argument(
        "--no-gdn2",
        action="store_true",
        help="ablation: use original scalar-gate GatedDeltaNet instead of default GDN-2.",
    )
    p.add_argument(
        "--use-math-module",
        action="store_true",
        help="neural arithmetic module per core block (NAC/NALU weight-constrained FFN). "
        "Learns +,-,×,÷ as weight patterns that generalize to unseen numbers. "
        "~21K params per block (<0.01%% of 400M), per-token learned gate.",
    )

    # ── Experimental Features ──
    ex = p.add_argument_group("Experimental Features")
    # Architecture options
    ex.add_argument(
        "--use-bipolar-gate",
        action="store_true",
        help="bipolar sign-gating: STE forces GDN k,v to ±1 in forward pass (discrete facts), differentiable in backward. 0 params.",
    )
    ex.add_argument(
        "--use-osdn",
        action="store_true",
        help="OSDN per-dimension key preconditioning (arXiv:2605.13473). +H·head_dim params.",
    )
    ex.add_argument(
        "--use-mtp-routing",
        action="store_true",
        help="MTP-routed macro-states: feed predicted future rep into core loops for multi-token reasoning.",
    )

    # Training aux losses
    ex.add_argument(
        "--use-task-rl",
        action="store_true",
        help="task-based RL with advantage-weighted regression",
    )
    ex.add_argument(
        "--awr-temperature",
        type=float,
        default=0.5,
        help="AWR temperature (lower = more selective, only strong successes)",
    )
    ex.add_argument(
        "--value-weight", type=float, default=1.0, help="weight on the value-head critic loss (MSE)"
    )
    ex.add_argument(
        "--task-replay-size",
        type=int,
        default=1000,
        help="capacity of the task-trajectory replay buffer",
    )

    # Reasoning modules per core block
    ex.add_argument(
        "--use-logic-playground",
        action="store_true",
        help="differentiable boolean logic workspace per core block (AND/OR/NOT/XOR gates in recurrent workspace).",
    )
    ex.add_argument(
        "--playground-dim", type=int, default=64, help="dimension of the logic playground workspace"
    )
    ex.add_argument(
        "--max-play-steps",
        type=int,
        default=4,
        help="max recurrent iterations within the playground",
    )
    ex.add_argument(
        "--use-calculus-module",
        action="store_true",
        help="neural calculus module per core block: differentiation (∂x/∂t) + integration (∫x dt).",
    )
    ex.add_argument(
        "--calculus-bottleneck", type=int, default=16, help="hidden dim of the calculus module"
    )
    ex.add_argument(
        "--use-comparator-module",
        action="store_true",
        help="relational comparison of consecutive states (greater/less/equal/contrast).",
    )
    ex.add_argument(
        "--comparator-bottleneck", type=int, default=8, help="hidden dim of the comparator module"
    )
    ex.add_argument(
        "--reasoning-train-all-loops",
        action="store_true",
        help="ablation: run math/logic/calculus/comparator on every training loop. "
        "Default schedules them on final gradient-carrying loops to save VRAM/compute.",
    )

    ex.add_argument(
        "--use-thermostat",
        action="store_true",
        help="epistemic thermostat: override greedy halting when confidence variance is high or trajectory oscillates.",
    )
    ex.add_argument(
        "--thermostat-conf-threshold",
        type=float,
        default=0.02,
        help="confidence variance ceiling for forced continue",
    )

    # Neuromodulated gating

    # Convergence / early-exit
    ex.add_argument(
        "--track-convergence",
        action="store_true",
        help="expose per-token convergence (extrapolation-error) signal at inference.",
    )
    ex.add_argument(
        "--convergence-mode",
        type=str,
        default="extrapolation",
        choices=["extrapolation", "acceleration"],
        help="convergence signal mode: extrapolation (1st-order) or acceleration (2nd diff).",
    )
    ex.add_argument(
        "--convergence-window",
        type=int,
        default=3,
        help="rolling window size for convergence estimation (linear extrapolation).",
    )
    ex.add_argument(
        "--use-accel-exit",
        action="store_true",
        help="acceleration-curvature early exit at inference (two-scale-latent, arXiv:2509.23314).",
    )
    ex.add_argument(
        "--accel-exit-threshold",
        type=float,
        default=0.01,
        help="curvature threshold for acceleration-curvature early exit",
    )

    # SNGP (distance-aware epistemic uncertainty)
    ex.add_argument(
        "--sngp-enabled",
        action="store_true",
        help="enable SNGP head for distance-aware epistemic variance at inference (arXiv:2006.10108).",
    )
    ex.add_argument(
        "--sngp-rff-dim", type=int, default=256, help="random Fourier feature dimension for SNGP"
    )
    ex.add_argument("--sngp-scale", type=float, default=1.0, help="SNGP RFF scale parameter")
    ex.add_argument(
        "--sngp-ridge", type=float, default=1.0, help="SNGP precision ridge regularization"
    )
    ex.add_argument(
        "--sngp-spectral-norm",
        action="store_true",
        help="spectral norm on coda for distance-preserving features (bi-Lipschitz condition).",
    )
    ex.add_argument(
        "--sngp-accumulate-train",
        action="store_true",
        help="accumulate SNGP precision during training. Default off; prefer an explicit calibration pass.",
    )

    # Laplace post-hoc
    ex.add_argument(
        "--laplace-enabled",
        action="store_true",
        help="record Laplace-Redux intent (post-hoc last-layer Laplace on conf_head). No training change.",
    )
    ex.add_argument(
        "--use-process-head",
        action="store_true",
        help="train a process verifier over recurrent states against final token correctness.",
    )
    ex.add_argument(
        "--process-weight", type=float, default=0.05, help="weight for the process-head BCE loss"
    )
    ex.add_argument(
        "--halt-granularity",
        choices=["token", "segment", "sequence"],
        default="token",
        help="pool learned halt probabilities at token, segment, or sequence granularity",
    )
    ex.add_argument(
        "--halt-segment-len",
        type=int,
        default=16,
        help="segment length when --halt-granularity segment is used",
    )
    p.add_argument(
        "--set",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="generic CharkhaConfig override (repeatable) — the ablation lever",
    )
    p.add_argument(
        "--z-loss",
        type=float,
        default=0.0,
        help="PaLM z-loss weight (~1e-4): bounds the logit scale (bf16 stability)",
    )
    p.add_argument(
        "--logit-softcap",
        type=float,
        default=0.0,
        help="Gemma-2 final-logit softcap (~30); applied in training AND inference",
    )
    p.add_argument(
        "--ema",
        type=float,
        default=0.0,
        help="weight-EMA decay (~0.999); shadow kept on CPU, saved in meta[ema], "
        "preferred by serve.py at load",
    )
    p.add_argument(
        "--ema-every", type=int, default=10, help="update the EMA every N steps (compounded decay)"
    )
    p.add_argument(
        "--data-weight",
        type=float,
        action="append",
        default=None,
        help="per --data sampling multiplier (repeat, parallel to --data): the "
        "data-annealing lever — upweight high-quality dirs late in the run",
    )
    p.add_argument(
        "--retrieval-context-data",
        type=str,
        default=None,
        help="optional shard dir of retrieval/world-context text to splice before normal "
        "training windows for retrieval-aware pretraining",
    )
    p.add_argument(
        "--retrieval-context-frac",
        type=float,
        default=0.0,
        help="fraction of micro-batches that receive retrieval-context prefixes",
    )
    p.add_argument(
        "--retrieval-context-tokens",
        type=int,
        default=64,
        help="prefix length used for --retrieval-context-data",
    )
    p.add_argument(
        "--logit-prior",
        type=str,
        default=None,
        help="checkpoint of a small FROZEN fluency prior (CharkhaConfig.mini, same "
        "tokenizer): its logits are added to the student's during CE with the "
        "weight annealed to 0 — the student only learns what the prior cannot "
        "predict (residual-logit training accelerant)",
    )
    p.add_argument("--prior-weight", type=float, default=1.0, help="initial prior logit weight")
    p.add_argument(
        "--prior-anneal",
        type=int,
        default=20000,
        help="steps over which the prior weight anneals linearly to 0",
    )
    p.add_argument(
        "--distill",
        type=str,
        default=None,
        help="teacher HF id for soft-label KD — any teacher sharing the GPT-NeoX "
        "tokenizer (frontier KD in scope; artifact is non-distributable either way)",
    )
    p.add_argument("--distill-weight", type=float, default=0.5, help="KD loss weight")
    p.add_argument("--distill-temp", type=float, default=2.0, help="KD softmax temperature")
    p.add_argument("--distill-topk", type=int, default=64, help="teacher top-k tokens per position")
    p.add_argument(
        "--kd-shards",
        type=str,
        default=None,
        help="cached-teacher KD dir from scripts/cache_logits.py — distill with "
        "ZERO teacher inference (the accordion come-down)",
    )
    p.add_argument(
        "--kd-shard-frac",
        type=float,
        default=1.0,
        help="fraction of micro-batches drawn from --kd-shards (rest = normal "
        "data, i.e. replay mixing)",
    )
    p.add_argument(
        "--esr-tokens",
        type=int,
        default=None,
        help="efficient selective refinement: apply KD only to the first N target "
        "positions of each sequence. 0/None = all positions.",
    )
    p.add_argument(
        "--refine-trajectory",
        action="store_true",
        help="call teacher.refine(x) before top-k when the teacher provides it",
    )
    p.add_argument(
        "--teacher-quant",
        type=str,
        default=None,
        choices=["4bit", "8bit"],
        help="quantize the teacher (bitsandbytes) to fit a 7B in ~4-7GB",
    )
    p.add_argument(
        "--teacher-device",
        type=str,
        default=None,
        help="run the teacher off the training GPU (e.g. cpu, or cuda:1) to free VRAM",
    )
    p.add_argument(
        "--teacher-url",
        type=str,
        default=None,
        help="URL of a distill.py --serve teacher endpoint (e.g. http://teacher-host:8009). "
        "Uses RemoteTeacher over LAN instead of in-process HFTeacher.",
    )
    p.add_argument(
        "--kd-tokenizer-ok",
        action="store_true",
        help="assert the KD teacher uses the SAME custom tokenizer as the training shards "
        "(required to enable logit KD when shards were tokenized with a local "
        "tokenizer.json — otherwise KD trains on mismatched token ids)",
    )
    p.add_argument(
        "--val-frac", type=float, default=0.0, help="fraction of shards held out for eval"
    )
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="overwrite the rolling resume ckpt.pt every N steps (atomic)",
    )
    p.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="additionally write an immortal, never-overwritten ckpt_<step>.pt every N "
        "steps - a rewind point if the rolling ckpt.pt is overwritten with bad "
        "weights. 0 = off. Suggest a multiple of --save-every (e.g. 10000).",
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--resume", action="store_true", help="resume out/ckpt.pt if present")
    p.add_argument(
        "--branch-from", type=str, default=None, help="checkpoint to branch a release from"
    )
    p.add_argument("--decay-steps", type=int, default=0, help="WSD decay length for the branch")
    p.add_argument(
        "--mem-probe",
        action="store_true",
        help="VRAM-only probe: run real steps, report peak reserved vs DEDICATED vram + "
        "steady tok/s, and flag SHARED-memory spill (the silent perf killer). Then exit.",
    )
    p.add_argument(
        "--vram-budget",
        type=float,
        default=7.3,
        help="GB of DEDICATED vram a config may use and still count VRAM-only (8GB card minus "
        "Windows/WSL desktop overhead). Peak reserved above this risks shared-mem spill.",
    )
    p.add_argument(
        "--mem-steps",
        type=int,
        default=8,
        help="real steps per config in the mem-probe (first 2 are warmup, skipped for tok/s)",
    )
    p.add_argument(
        "--mem-sweep",
        type=str,
        default=None,
        help="comma-separated seq_lens to sweep, e.g. 2048,1536,1024,768,512. The tok/s CLIFF "
        "between rows is the spill point — pick the largest VRAM-ONLY row. "
        "Omit to probe just --seq-len.",
    )
    p.add_argument(
        "--offload-optim",
        dest="offload_optim",
        action="store_true",
        default=True,
        help="stream large matrix optimizer momentum from CPU RAM (default; required for the 8GB path)",
    )
    p.add_argument(
        "--no-offload-optim",
        dest="offload_optim",
        action="store_false",
        help="keep optimizer momentum on GPU for throughput on large-VRAM cloud cards",
    )
    p.add_argument(
        "--symmetry-opt",
        action="store_true",
        help="use the symmetry-compatible optimizer set (arXiv:2605.18106): Muon for "
        "attn/GDN, Row/Col-NormM for embeddings+SwiGLU, AdamW for heads. Default off "
        "(keeps the proven Muon+AdamW split); enable to A/B vs uniform Muon.",
    )
    p.add_argument(
        "--norm-lr",
        type=float,
        default=0.02,
        help="base LR for the Row/Col-NormM groups (--symmetry-opt)",
    )
    p.add_argument(
        "--recurrence-curriculum",
        action="store_true",
        help="retrofitted-recurrence (arXiv:2511.07384): anneal mean_recurrence (E[r]) "
        "low->high over --curric-steps; schedule-only, helps the fixed-Poisson phase.",
    )
    p.add_argument(
        "--curric-r-start", type=int, default=2, help="starting mean_recurrence for the curriculum"
    )
    p.add_argument(
        "--curric-r-end",
        type=int,
        default=None,
        help="ending mean_recurrence (default: cfg.mean_recurrence)",
    )
    p.add_argument("--curric-steps", type=int, default=10000, help="steps over which to ramp E[r]")
    p.add_argument(
        "--per-seq-recurrence",
        action="store_true",
        help="LoopWM (arXiv:2606.18208): sample the recurrent loop-count per SEQUENCE "
        "inside the micro-batch (not one per batch). No-op at batch=1; reduces "
        "objective variance / loss spikes at batch>1. Use for the big-batch cloud run.",
    )
    p.add_argument(
        "--bptt-half",
        action="store_true",
        help="LoopWM: truncate BPTT to ceil(E[r]/2) loops (tracks the curriculum), instead "
        "of the fixed cfg.backprop_depth. Saves activation memory at deep E[r].",
    )
    p.add_argument(
        "--halt-start-step",
        type=int,
        default=0,
        help="spec halter phasing: fixed Poisson loops (halting OFF) until this step, then "
        "enable the configured halter (Phase 1 -> Phase 2 in one resumable run). "
        "0 = halt per cfg from step 0 (default, bit-identical to old behavior).",
    )
    p.add_argument(
        "--ponder-anneal-steps",
        type=int,
        default=0,
        help="ramp ponder_cost linearly 0 -> cfg.ponder_cost over this many steps starting "
        "at --halt-start-step (anneals the PonderNet penalty in). 0 = constant (default).",
    )
    p.add_argument(
        "--sample-every",
        type=int,
        default=0,
        help="generate a sample from --sample-prompt every N steps so the run is watchable "
        "(see the same continuation sharpen from noise to text). 0 = off (default).",
    )
    p.add_argument("--sample-tokens", type=int, default=48, help="new tokens per live sample")
    p.add_argument(
        "--sample-prompt",
        type=str,
        default="",
        help="fixed seed text for live samples (empty = seed from BOS). A constant prompt "
        "makes improvement visible across steps.",
    )
    p.add_argument(
        "--sample-effort",
        type=int,
        default=None,
        help="fixed recurrence loops for live samples (default: adaptive halting)",
    )
    p.add_argument(
        "--sample-temp", type=float, default=0.8, help="sampling temperature for live samples"
    )
    p.add_argument("--sample-top-k", type=int, default=50, help="top-k for live samples")
    p.add_argument(
        "--no-dashboard",
        dest="dashboard",
        action="store_false",
        default=True,
        help="disable the live dashboard (specs banner + per-log %%/ETA/VRAM/spill + eval status "
        "block). On by default for real runs; ideal for multi-day/week monitoring.",
    )
    p.add_argument(
        "--probe",
        action="append",
        default=None,
        metavar="PROMPT=>EXPECTED",
        help="add a custom capability milestone to the dashboard checklist (repeatable): "
        "the prompt unlocks when the generation contains EXPECTED (case-insensitive). "
        "e.g. --probe 'Hi! =>hello' --probe 'The opposite of up is =>down'",
    )
    a = p.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    if not a.data:
        p.error("--data is required (or use --selftest)")
    if a.mem_probe:
        raise SystemExit(mem_probe(a))
    train(a)


if __name__ == "__main__":
    main()

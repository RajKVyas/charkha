#!/usr/bin/env python
"""
CHARKHA preflight — ONE command to verify everything before a multi-day run.
============================================================================
A superset of running the five selftests by hand. Three parts:

  1) COMPONENT SELF-TESTS   every file's own regression suite (subprocess, pass/fail rollup)
  2) PROBE-MODEL WARMUP     a quick toy training run (curriculum + halter phasing) that produces the
                            trained model the feature probes interrogate. The long, watchable
                            generation-sharpening showcase moved to demo.py --integrated.
  3) FEATURE PROBES         one line of evidence each for the headline behaviors — especially the
                            honesty thesis: confidence is HIGHER on in-distribution text than on
                            out-of-distribution garbage (the anti-hallucination substrate)

Then a FEATURE COVERAGE MAP ties every implemented idea to the test that exercises it, and a
single GREEN/RED verdict. Exit code 0 = cleared for launch.

Usage:
  python preflight.py                                # selftests + probes + benchmarks (CPU: a few min)
  python preflight.py --probes-only                  # skip the per-file selftest suite (fast iteration)
  python preflight.py --skip-probes                  # only the component selftests
  python preflight.py --scale-data <real-shard-dir>  # + AT-SCALE SMOKE: the true pre-launch gate (GPU)

The long, watchable SHOWCASES (generation-sharpening + the single-digit addition self-learning
curriculum) now live in `demo.py` — they are showcases, NOT launch gates.
preflight is the gate: component selftests + feature probes + the at-scale smoke (real shards,
real config, GDN-1 fla kernel available, loss drops, no NaN, resume continuity, fits 8GB). Capability
itself is proven by `pipeline.py --eval` / `--elasticity` on a trained checkpoint, not here.

"""

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import time

try:  # we print decoded model bytes; don't die on win32 cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def hr(t):
    print("\n" + "=" * 74 + f"\n  {t}\n" + "=" * 74)


def mark(b):
    return "PASS" if b else "FAIL"


def _norm_path(p):
    return os.path.realpath(os.path.abspath(p)).replace("\\", "/").lower()


def _is_wsl():
    try:
        return (
            "microsoft"
            in open("/proc/version", "r", encoding="utf-8", errors="ignore").read().lower()
        )
    except Exception:
        return False


def run_env_check():
    """Fail fast on broken venvs/deps instead of printing a dozen import tracebacks."""
    hr("0) ENVIRONMENT CHECK")
    checks = {}
    print(f"  python: {sys.executable}")
    print(f"  cwd:    {os.getcwd()}")
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        print(f"  venv:   {venv}")
    else:
        print("  venv:   (none)")

    if venv:
        root_n, venv_n = _norm_path(ROOT), _norm_path(venv)
        if _is_wsl() and root_n.startswith("/home/") and venv_n.startswith("/mnt/"):
            checks["WSL venv lives on Linux filesystem"] = False
            print("  [FAIL] WSL is using a /mnt/* virtualenv while repo code is on Linux FS.")
            print("         Run: deactivate")
            print(
                "         Then: cd ~/charkha && bash scripts/setup_wsl.sh && source .venv/bin/activate"
            )
        else:
            checks["WSL venv lives on Linux filesystem"] = True

    for mod in ("numpy", "torch"):
        spec = importlib.util.find_spec(mod)
        if spec is None:
            checks[f"import {mod}"] = False
            print(f"  [FAIL] import {mod}: module not installed")
            continue
        try:
            __import__(mod)
            checks[f"import {mod}"] = True
            print(f"  [PASS] import {mod}")
        except Exception as e:
            checks[f"import {mod}"] = False
            print(f"  [FAIL] import {mod}: {type(e).__name__}: {e}")

    if _is_wsl():
        root_n = _norm_path(ROOT)
        if root_n.startswith("/mnt/"):
            print(
                "  [WARN] repo is under /mnt/*. Training is supported but slower; prefer ~/charkha for code+venv."
            )
    if not all(checks.values()):
        print("\n  Environment is not launch-ready. Fix it before running preflight.")
        print("  Canonical WSL repair:")
        print("    deactivate 2>/dev/null || true")
        print("    cd ~/charkha")
        print("    rm -rf .venv")
        print("    bash scripts/setup_wsl.sh")
        print("    source .venv/bin/activate")
    return checks


# Each file's self-test is its own regression suite; exit code is the pass/fail signal
# (charkha --toy runs assert-based feature tests inside train(), so it too returns non-zero on fail).
COMPONENTS = [
    ("dataprep", ["dataprep.py", "--selftest"]),
    ("charkha --toy", ["charkha.py", "--toy"]),
    ("train", ["train.py", "--selftest"]),
    ("serve", ["serve.py", "--selftest"]),
    ("pipeline", ["pipeline.py", "--selftest"]),
    ("retrieval", ["retrieval.py", "--selftest"]),
    ("sft", ["sft.py", "--selftest"]),
    ("selfteach", ["selfteach.py", "--selftest"]),
    ("distill", ["distill.py", "--selftest"]),
    ("continual", ["continual.py", "--selftest"]),
    ("verify_shards", ["verify_shards.py", "--selftest"]),
    ("crossdedup", ["crossdedup.py", "--selftest"]),
    ("merge", ["merge.py", "--selftest"]),
    ("tasks", ["tasks.py"]),
    ("synth_seeds", ["synth_seeds.py"]),
]


def run_components():
    hr("1) COMPONENT SELF-TESTS  (each file's own regression suite)")
    res = {}
    for name, argv in COMPONENTS:
        t = time.time()
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, argv[0])] + argv[1:],
            capture_output=True,
            text=True,
            cwd=HERE,
        )
        dt = time.time() - t
        passed = proc.returncode == 0
        res[name] = passed
        print(f"\n  {'=' * 60}")
        print(f"  [{mark(passed)}] {name} ({dt:.1f}s)")
        print(f"  {'=' * 60}")
        # Print full stdout, every line
        for line in proc.stdout.splitlines():
            print(f"  {line}")
        # Surface stderr if present
        if proc.stderr.strip():
            print("  --- stderr ---")
            for line in proc.stderr.splitlines():
                print(f"  {line}")
    return res


def _decode(ids):
    return bytes(b & 0xFF for b in ids).decode("utf-8", errors="replace")


def _make_mixed_shards(data_dir, n_shards=3, toks_per=9000):
    """Corpus = a learnable sentence INTERLEAVED with runs of unpredictable bytes (fresh per block
    so the model can't memorize them). This gives the confidence head real WRONG examples on the
    noise spans (vs the pure-sentence corpus, where the model is ~100% right and conf saturates),
    so the calibration mechanism can actually be measured."""
    import array
    import random as _random

    os.makedirs(data_dir, exist_ok=True)
    sentence = list(b"the people build their own tools and learn the shape of their own freedom. ")
    rng = _random.Random(1234)
    shards, total = [], 0
    for s in range(n_shards):
        buf = array.array("H")
        while len(buf) < toks_per:
            buf.extend(sentence)
            buf.extend(
                rng.randint(0, 255) for _ in range(40)
            )  # unpredictable span -> model is wrong here
            buf.append(256)  # EOS (uint16-safe)
        buf = buf[:toks_per]
        fn = f"shard_{s:05d}.bin"
        with open(os.path.join(data_dir, fn), "wb") as f:
            buf.tofile(f)
        shards.append({"file": fn, "tokens": len(buf)})
        total += len(buf)
    import json

    with open(os.path.join(data_dir, "index.json"), "w") as f:
        json.dump({"vocab_size": 257, "total_tokens": total, "shards": shards}, f)


def _honesty_probe(tmp):
    """Train a tiny model on text+noise (quietly), then measure: is it CONFIDENT on learnable text
    and UNSURE on pure noise? Returns (conf_text, conf_noise). conf_text > conf_noise == calibrated."""
    import io
    import contextlib
    import random as _random
    import torch
    from train import train, _toy_args, load_ckpt

    hdir, hdata = os.path.join(tmp, "honesty"), os.path.join(tmp, "hdata")
    _make_mixed_shards(hdata)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):  # suppress the training spam
        train(
            _toy_args(
                steps=200,
                out=hdir,
                data=hdata,
                val_frac=0.2,
                eval_every=0,
                log_every=200,
                sample_every=0,
            )
        )
    model, cfg, *_ = load_ckpt(os.path.join(hdir, "ckpt.pt"), "cpu")
    model.eval()
    enc = lambda s: list(s.encode("utf-8"))
    rg = _random.Random(99)  # fresh noise, unseen in training
    text = torch.tensor(
        [enc("the people build their own tools and learn the shape of ")], dtype=torch.long
    )
    noise = torch.tensor([[rg.randint(0, 255) for _ in range(50)]], dtype=torch.long)
    with torch.no_grad():
        _, ct = model(text)
        _, cn = model(noise)
    return float(ct.mean()), float(cn.mean())


from _probes import COVERAGE, run_feature_probes


def print_coverage():
    hr("4) FEATURE COVERAGE MAP — every implemented idea -> the test that exercises it")
    for grp, items in COVERAGE.items():
        print(f"  {grp}:")
        for feat, by in items:
            print(f"      - {feat}  ({by})")


def run_scale_smoke(data_dir, steps=60, seq_len=2048, embed_factor=None):
    """The REAL pre-launch gate — what a toy test can't prove. Invokes the ACTUAL train.py CLI (the
    exact entrypoint the multi-day run uses) on REAL shards at the launch config (default 0.42B, NOT
    the toy model) for a short burst, then a resume burst, verifying:
      - the real config builds + runs without OOM/crash on THIS box (non-zero exit == it didn't fit),
      - the GDN-1 fla Triton kernel is available on CUDA,
      - loss decreases and stays finite (no NaN/Inf),
      - eval produces a finite val NLL,
      - checkpoint + resume continuity (the multi-day run depends on it).
    Short (~1-3 min on GPU). Meant for the GPU box. Returns a checks dict that GATES the verdict."""
    import json
    import math

    hr("AT-SCALE SMOKE — short REAL-config train.py run on real shards (the true launch gate)")
    checks = {}
    if data_dir:
        # train.py is launched with cwd=src/ below; a relative --scale-data like
        # 'data/foo-dd' (valid from the repo root) would silently stop resolving there.
        data_dir = os.path.abspath(data_dir)
    if (
        not data_dir
        or not os.path.isdir(data_dir)
        or not os.path.exists(os.path.join(data_dir, "index.json"))
    ):
        print(
            f"  [FAIL] --scale-data is not a shard dir (need index.json + shard_*.bin): {data_dir}"
        )
        checks["at-scale: shard dir exists"] = False
        return checks
    checks["at-scale: shard dir exists"] = True

    has_cuda, fla = False, None
    try:
        import torch

        has_cuda = torch.cuda.is_available()
        from charkha import have_fla

        fla = bool(have_fla())
    except Exception as e:
        print(f"  [warn] could not probe cuda/fla: {type(e).__name__}: {e}")
    print(f"  cuda={has_cuda}  fla/triton GDN-1 kernel available={fla}")
    print("  note: default frontier cfg uses GDN-2 exact PyTorch unless launched with --no-gdn2.")
    if has_cuda:
        checks["at-scale: GDN-1 fla kernel available on CUDA"] = fla is True
    else:
        print(
            "  NOTE: no CUDA here — this would run the REAL config on CPU (slow). Run it on the GPU box."
        )

    tmp = tempfile.mkdtemp(prefix="charkha_scale_")
    out = os.path.join(tmp, "run")
    mp = os.path.join(out, "metrics.jsonl")
    base = [
        sys.executable,
        os.path.join(HERE, "train.py"),
        "--profile",
        "frontier",
        "--data",
        data_dir,
        "--out",
        out,
        "--batch-size",
        "1",
        "--seq-len",
        str(seq_len),
        "--accum-steps",
        "1",
        "--ce-chunk",
        "512",
        "--gdn-chunk",
        "16",
        "--grad-checkpoint",
        "--offload-optim",
        "--8bit-optim",
        "--symmetry-opt",
        "--bptt-half",
        "--val-frac",
        "0.02",
        "--warmup",
        str(max(2, steps // 10)),
    ]
    if embed_factor:
        base += ["--embed-factor", str(embed_factor)]  # test the SAME embedding the launch will use
    run1 = base + [
        "--steps",
        str(steps),
        "--eval-every",
        str(max(2, steps // 2)),
        "--log-every",
        "5",
    ]
    print(f"  $ python train.py {' '.join(run1[2:])}")
    print(
        "  (streaming train.py output live below — a 0.42B step at this seq_len is heavy; watch tok/s."
    )
    print(
        "   If tok/s is tiny while dedicated VRAM is maxed, the config is SPILLING to shared memory —"
    )
    print(
        "   Ctrl-C and run `train.py --mem-probe --offload-optim --8bit-optim --symmetry-opt "
        "--mem-sweep 2048,1536,1024,768,512 ...` to find a fit.)"
    )
    p1 = subprocess.run(run1, cwd=HERE)  # stream live (no capture) so progress is visible
    checks["at-scale: real-config run did not OOM/crash"] = p1.returncode == 0
    if p1.returncode != 0:
        print(f"  [FAIL] train.py exited {p1.returncode} (see streamed output above)")
        for k, v in checks.items():
            print(f"  [{mark(v)}] {k}")
        return checks

    rows = [json.loads(l) for l in open(mp, encoding="utf-8")] if os.path.exists(mp) else []
    losses = [r["loss"] for r in rows if "loss" in r]
    finite = bool(losses) and all(math.isfinite(l) for l in losses)
    # judge the TREND on the EMA train.py logs, not two single-sample losses: at B=1 the
    # per-step loss is noisy enough that a healthy 60-step run can randomly end on a sample
    # above its first one (false RED on an otherwise all-green gate).
    emas = [r["loss_ema"] for r in rows if "loss_ema" in r] or losses
    dropped = len(emas) >= 2 and emas[-1] < emas[0]
    checks["at-scale: no NaN/Inf over the run"] = finite
    checks["at-scale: loss decreased"] = dropped
    if losses:
        print(f"  loss {losses[0]:.3f} -> {losses[-1]:.3f}   finite={finite}  dropped={dropped}")
    evals = [r for r in rows if "val_nll" in r]
    eval_ok = bool(evals) and math.isfinite(evals[-1]["val_nll"])
    checks["at-scale: eval produced finite val nll"] = eval_ok
    if evals:
        print(
            f"  val nll {evals[-1]['val_nll']:.3f}   ppl {evals[-1].get('ppl', float('nan')):.2f}"
        )
    step_before = max((r.get("step", 0) for r in rows), default=0)

    # resume burst — proves checkpoint/resume continuity (the multi-day run depends on it)
    run2 = base + ["--steps", str(steps + max(4, steps // 5)), "--eval-every", "0", "--resume"]
    p2 = subprocess.run(run2, cwd=HERE)  # stream live
    rows2 = [json.loads(l) for l in open(mp, encoding="utf-8")] if os.path.exists(mp) else []
    step_after = max((r.get("step", 0) for r in rows2), default=0)
    checks["at-scale: checkpoint written"] = os.path.exists(os.path.join(out, "ckpt.pt"))
    checks["at-scale: resume advances step"] = p2.returncode == 0 and step_after > step_before
    print(f"  resume: step {step_before} -> {step_after}  (exit {p2.returncode})")

    print()
    for k, v in checks.items():
        print(f"  [{mark(v)}] {k}")
    return checks


def ensure_hf_ckpt(local_path="hf_ckpt.pt"):
    import os
    import subprocess
    import shutil

    if os.path.exists(local_path):
        return local_path
    print("\n  [Downloading real trained checkpoint from CHARKHA_ORG/charkha-ckpt...]")
    try:
        subprocess.run(
            ["hf", "download", "CHARKHA_ORG/charkha-ckpt", "ckpt.pt", "--local-dir", "."],
            check=True,
        )
        if os.path.exists("ckpt.pt") and local_path != "ckpt.pt":
            shutil.move("ckpt.pt", local_path)
        return local_path
    except Exception as e:
        print(f"  [WARN] Failed to download via hf CLI: {e}")
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id="CHARKHA_ORG/charkha-ckpt", filename="ckpt.pt")
            shutil.copy(path, local_path)
            return local_path
        except Exception as e2:
            print(f"  [WARN] Failed fallback download via huggingface_hub: {e2}")
            return None


def run_post_training_checks(ckpt_path):
    hr("POST-TRAINING CHECKS — verifying the real HF checkpoint")
    checks = {}
    print(f"  Verifying {ckpt_path} ...")

    # 1. Ensure it loads properly
    try:
        from train import load_ckpt
        import torch

        model, cfg, opts, bases, step = load_ckpt(
            ckpt_path, "cuda" if torch.cuda.is_available() else "cpu"
        )
        checks["post-train: model loads cleanly"] = True
    except Exception as e:
        print(f"  [FAIL] Could not load checkpoint: {e}")
        checks["post-train: model loads cleanly"] = False
        return checks

    # 2. Check optimizers are intact
    has_opts = opts is not None and len(opts) > 0
    checks["post-train: optimizer state intact"] = has_opts

    # 3. Try to do a single resume step (verifies training can continue)
    try:
        model.train()
        device = next(model.parameters()).device
        xb = torch.randint(0, cfg.vocab_size, (1, 32), device=device)
        if has_opts:
            for o in opts:
                o.zero_grad(set_to_none=True)
            _, loss = model(xb, xb, r=1)
            loss.backward()
            for o in opts:
                o.step()
        checks["post-train: can take a training step (resume OK)"] = True
    except Exception as e:
        print(f"  [FAIL] Could not take a training step: {e}")
        checks["post-train: can take a training step (resume OK)"] = False

    return checks


def main():
    ap = argparse.ArgumentParser(description="CHARKHA preflight: full pre-launch verification.")
    ap.add_argument(
        "--skip-env-check",
        action="store_true",
        help="skip the dependency/venv doctor (debug only; normal preflight should keep it on)",
    )
    ap.add_argument(
        "--skip-components", action="store_true", help="skip the per-file selftest suite"
    )
    ap.add_argument(
        "--skip-probes",
        "--skip-demo",
        dest="skip_probes",
        action="store_true",
        help="skip the probe-model warmup + feature probes (--skip-demo kept as an alias)",
    )
    ap.add_argument(
        "--probes-only",
        "--demo-only",
        dest="probes_only",
        action="store_true",
        help="only the feature probes; skip the selftest suite (--demo-only kept as an alias)",
    )
    ap.add_argument(
        "--gpu-only",
        action="store_true",
        help="skip components+demo+coverage, go straight to GPU benchmark",
    )
    ap.add_argument(
        "--cpu-bench",
        action="store_true",
        help="run the long CPU training benchmark (also on by default)",
    )
    ap.add_argument("--skip-cpu-bench", action="store_true", help="skip the CPU training benchmark")
    ap.add_argument(
        "--skip-gpu-bench", action="store_true", help="skip the CUDA training benchmark"
    )
    ap.add_argument(
        "--bench-steps", type=int, default=200, help="steps for the CPU/GPU training benchmarks"
    )
    ap.add_argument(
        "--scale-data",
        type=str,
        default=None,
        help="REAL shard dir (index.json + shard_*.bin). Runs the AT-SCALE SMOKE gate: a short "
        "real-config run verifying GDN-1 fla availability, loss drops, no NaN, "
        'checkpoint+resume continuity, and finite eval — the true "ready to launch" check.',
    )
    ap.add_argument("--scale-steps", type=int, default=60, help="steps for the at-scale smoke gate")
    ap.add_argument(
        "--scale-seq-len",
        type=int,
        default=2048,
        help="seq-len for the at-scale smoke gate (defaults to the real launch T=2048)",
    )
    ap.add_argument(
        "--scale-embed-factor",
        type=int,
        default=None,
        help="pass --embed-factor to the at-scale smoke so it gates the SAME factorized-"
        "embedding config the launch will use (e.g. 256 on the 8GB box)",
    )
    ap.add_argument(
        "--hf-check",
        action="store_true",
        help="download and verify the optional CHARKHA_ORG/charkha-ckpt checkpoint",
    )
    ap.add_argument(
        "--probe-reasoning-modules",
        action="store_true",
        help="run the reasoning-module probe (opt-in)",
    )
    a = ap.parse_args()
    if a.gpu_only:
        a.skip_components = True
        a.skip_probes = True
        a.skip_cpu_bench = True
    if a.probes_only:
        a.skip_components = True
    t0 = time.time()

    has_cuda = False
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except Exception:
        pass
    mode = "GPU" if has_cuda else "CPU"
    hr(f"CHARKHA PREFLIGHT -- pre-launch verification harness ({mode} mode)")
    print("  CPU regression suite + GPU benchmark (auto-detected).")
    if not has_cuda:
        print("  NOTE: CUDA not available -- GPU benchmark will be skipped.")

    echecks = {} if a.skip_env_check else run_env_check()
    if echecks and not all(echecks.values()):
        hr("VERDICT")
        for n, p in echecks.items():
            print(f"  [{mark(p)}] environment: {n}")
        print("  PREFLIGHT RED -- environment/dependencies are broken; fix before launch.")
        return 1

    comp, dchecks = {}, {}
    if not a.skip_components:
        comp = run_components()
    else:
        print("\n  (component selftests skipped)")
    if not a.skip_probes:
        dchecks, _tmp = run_feature_probes()
    print_coverage()

    # CPU training benchmark — a proper long run on the CPU path (the no-triton chunked GDN scan),
    # so CPU throughput is measured the same way the GPU benchmark measures CUDA throughput.
    cchecks = {} if a.skip_cpu_bench else run_cpu_benchmark(steps=a.bench_steps)

    # GPU benchmark — always runs when CUDA is available
    gchecks = {} if a.skip_gpu_bench else run_gpu_benchmark(steps=a.bench_steps)

    # AT-SCALE SMOKE — the real launch gate; only when a real shard dir is provided.
    schecks = (
        run_scale_smoke(
            a.scale_data,
            steps=a.scale_steps,
            seq_len=a.scale_seq_len,
            embed_factor=a.scale_embed_factor,
        )
        if a.scale_data
        else {}
    )

    # POST-TRAINING VERIFICATION — runs on the HF checkpoint if allowed
    pchecks = {}
    if a.hf_check:
        ckpt = ensure_hf_ckpt()
        if ckpt:
            pchecks = run_post_training_checks(ckpt)

    hr("VERDICT")
    for n, p in echecks.items():
        print(f"  [{mark(p)}] environment: {n}")
    for n, p in comp.items():
        print(f"  [{mark(p)}] component: {n}")
    for group in (dchecks, cchecks, gchecks, schecks, pchecks):
        for n, p in group.items():
            print(f"  [{mark(p)}] {n}")
    ok = all(
        list(echecks.values())
        + list(comp.values())
        + list(dchecks.values())
        + list(cchecks.values())
        + list(gchecks.values())
        + list(schecks.values())
        + list(pchecks.values())
    )
    dt = time.time() - t0
    print()
    if not a.scale_data:
        print(
            "  NOTE: AT-SCALE SMOKE not run (no --scale-data). Before the REAL launch, run on the GPU box:"
        )
        print(
            "        python preflight.py --probes-only --skip-cpu-bench --scale-data <real-shard-dir>"
        )
    if ok:
        if a.scale_data:
            print(
                f"  PREFLIGHT GREEN -- all checks passed in {dt:.0f}s. Cleared for the multi-day run."
            )
        else:
            print(
                f"  PREFLIGHT GREEN -- requested checks passed in {dt:.0f}s. Run at-scale smoke before launch."
            )
    else:
        print(f"  PREFLIGHT RED -- fix the FAIL lines above before launching. ({dt:.0f}s)")
    return 0 if ok else 1


# Section 5 — GPU benchmark: Fable-5 mini-training with visible quality arc
# --------------------------------------------------------------------------

from _benchmarks import run_cpu_benchmark, run_gpu_benchmark

if __name__ == "__main__":
    raise SystemExit(main())

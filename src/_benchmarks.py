"""CHARKHA benchmarks — CPU and GPU training throughput measurements."""

import math
import os
import tempfile
import time
import torch
import torch.nn.functional as F
import numpy as np
from charkha import Charkha, CharkhaConfig, build_optimizers, wsd_lr_mult
from train import ShardLoader, make_synthetic_shards

HERE = os.path.dirname(os.path.abspath(__file__))


def hr(title):
    print("\n" + "=" * 74 + f"\n  {title}\n" + "=" * 74)


def mark(ok):
    return "PASS" if ok else "FAIL"


def run_cpu_benchmark(steps=200):
    """A proper CPU training run that mirrors the GPU benchmark: build a small-but-real model,
    train it for `steps` and report throughput + the loss arc + samples. This exercises the
    no-triton chunked GDN scan (the only GDN path that ever runs on CPU / native Windows), so
    CPU tok/s is measured, not guessed. Hermetic: learns a byte-level corpus, no network/HF."""
    import random as _random

    hr("CPU BENCHMARK — small-model training on the chunked-GDN path")
    checks = {}
    device = "cpu"
    # A real (non-toy) config sized so a few hundred CPU steps finish in minutes. Byte-level vocab
    # keeps the tied embedding/head cheap so the run measures the mixer/MLP path, not a 50K softmax.
    cfg = CharkhaConfig(
        vocab_size=384,
        d_model=256,
        n_heads=8,
        n_kv_heads=2,
        d_ff=768,
        n_prelude=2,
        n_core=4,
        n_coda=2,
        window=128,
        max_seq_len=512,
        mean_recurrence=2,
        max_recurrence_train=2,
        max_recurrence_infer=3,
        backprop_depth=1,
        use_halting=True,
        grad_checkpoint=False,
        mtp_weight=0.1,
        conf_weight=0.05,
    )
    tmp = tempfile.mkdtemp(prefix="charkha_cpubench_")
    data_dir = os.path.join(tmp, "data")
    make_synthetic_shards(data_dir, n_shards=4, toks_per=8000)
    loader = ShardLoader(data_dir)
    n_param = sum(p.numel() for p in Charkha(cfg).parameters())
    print(
        f"  model: {n_param:,} params, device=cpu | data {loader.total:,} tok, vocab={loader.vocab_size}"
    )
    model = Charkha(cfg).to(device)
    opts = build_optimizers(model, muon_lr=0.02, adam_lr=3e-3)
    B, T, warmup = 4, 256, 20
    print(f"  schedule: {steps} steps at batch={B}x{T}={B * T} tok/step (chunk={cfg.gdn_chunk})")
    rng = _random.Random(0)
    losses, tok_total, t0 = [], 0, time.time()
    enc = lambda s: list(s.encode("utf-8"))
    dec = lambda ids: bytes(b & 0xFF for b in ids).decode("utf-8", errors="replace")
    prompt = "the "
    for step in range(1, steps + 1):
        model.train()
        for o in opts:
            o.zero_grad(set_to_none=True)
        mult = wsd_lr_mult(step, warmup)
        for opt, base in zip(opts, (0.02, 3e-3)):
            for grp in opt.param_groups:
                grp["lr"] = base * mult
        xb, yb = loader.batch(B, T, device, np.random.RandomState(step))
        _, loss = model(xb, yb)
        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for o in opts:
                o.step()
            losses.append(loss.item())
        tok_total += B * T
        if step % max(1, steps // 5) == 0 or step == steps:
            valid = [v for v in losses[-50:] if math.isfinite(v)]
            dt = time.time() - t0
            avg = sum(valid) / len(valid) if valid else float("nan")
            print(f"\n  -- step {step}/{steps} | loss={avg:.3f} | {tok_total / dt:,.0f} tok/s --")
            model.eval()
            with torch.no_grad():
                out = model.generate(
                    torch.tensor([enc(prompt)], device=device), 32, temp=0.7, top_k=40, effort=None
                )
            print(f"    sample {prompt!r}│{dec(out[0, len(enc(prompt)) :].tolist())!r}")
    valid = [v for v in losses if math.isfinite(v)]
    total_t = time.time() - t0
    print(
        f"\n  -- DONE: {total_t:.1f}s, {tok_total / total_t:,.0f} tok/s, "
        f"loss {valid[0]:.2f} -> {valid[-1]:.2f} --"
        if valid
        else "\n  -- DONE: no finite losses --"
    )
    ok = bool(valid) and valid[-1] < valid[0]
    checks["CPU benchmark: loss decreases"] = ok
    print(
        f"  [{mark(ok)}] CPU benchmark loss decreases"
        + (f" ({valid[0]:.2f} -> {valid[-1]:.2f})" if valid else "")
    )
    return checks


def run_gpu_benchmark(steps=400, data_dir=None, tokenizer_path=None):
    """Run an optional GPU training benchmark against a user-supplied shard directory."""
    from tokenizers import Tokenizer

    checks = {}
    if os.name == "nt":
        print(
            "\n  [SKIP] GPU benchmark — native Windows is not the authoritative GDN-2 training path"
        )
        print("         Run WSL/Linux or the cloud at-scale smoke before launch.")
        checks["GPU benchmark: skipped on native Windows; use WSL/Linux scale smoke"] = True
        return checks
    if not torch.cuda.is_available():
        print("\n  [SKIP] GPU benchmark — CUDA not available")
        return checks

    device = "cuda"
    cfg = CharkhaConfig()
    cfg.d_model = 960
    cfg.n_heads = 15
    cfg.n_kv_heads = 5
    cfg.d_ff = 2560
    cfg.n_prelude = 3
    cfg.n_core = 6
    cfg.n_coda = 3
    cfg.window = 512
    cfg.max_seq_len = 2048
    cfg.vocab_size = 50277
    cfg.use_recurrence = True
    cfg.mean_recurrence = 2
    cfg.max_recurrence_train = 3
    cfg.max_recurrence_infer = 4
    cfg.backprop_depth = 1
    cfg.use_halting = True
    cfg.halt_threshold = 0.85
    cfg.grad_checkpoint = True
    cfg.mtp_weight = 0.1
    cfg.conf_weight = 0.05
    cfg.use_deep_supervision = True
    cfg.deepsup_weight = 0.05
    cfg.use_nitp = True
    cfg.nitp_weight = 0.05
    cfg.use_loop_embed = True
    cfg.effective_depth_scale = True
    cfg.use_bipolar_gate = True
    cfg.use_thermostat = True
    cfg.use_mtp_routing = True
    cfg.track_convergence = True

    hr("5) GPU BENCHMARK — user-supplied training shards")
    n_param = sum(p.numel() for p in Charkha(cfg).parameters())
    print(f"  model: {n_param:,} params, device={device}")

    data_dir = data_dir or os.environ.get("CHARKHA_BENCH_DATA")
    tokenizer_path = tokenizer_path or os.environ.get("CHARKHA_BENCH_TOKENIZER")
    import glob as _glob

    if (
        not data_dir
        or not os.path.isdir(data_dir)
        or not _glob.glob(os.path.join(data_dir, "shard_*.bin"))
    ):
        print(
            "\n  [SKIP] Set CHARKHA_BENCH_DATA to a prepared shard directory to enable "
            "the optional GPU benchmark."
        )
        return checks
    loader = ShardLoader(data_dir)
    print(
        f"  data:  {loader.total:,} tokens, {len(loader.arrays)} shard(s), vocab={loader.vocab_size}"
    )
    cfg.vocab_size = loader.vocab_size
    if tokenizer_path:
        tok = Tokenizer.from_file(tokenizer_path)
    else:

        class _TokenIds:
            @staticmethod
            def encode(text):
                return type("Encoded", (), {"ids": [0]})()

            @staticmethod
            def decode(ids):
                return " ".join(map(str, ids))

        tok = _TokenIds()

    model = Charkha(cfg).to(device)
    # Conservative LR: small batch (2048 tok) needs ~10x lower LR than full-scale (66K tok)
    opts = build_optimizers(model, muon_lr=0.003, adam_lr=3e-4)
    B, T = 4, 512
    print(f"  schedule: {steps} steps at batch={B}x{T}={B * T} tok/step, LR muon=0.003 adam=3e-4")

    rng = np.random.RandomState(42)
    losses, tok_total, t0 = [], 0, time.time()
    nan_streak = 0
    prompts = [
        "Think step by step: how would you",
        "Write a function that",
        "The bug is in the",
        "user: What is 5 plus 3?\nassistant:",
    ]

    # bf16 autocast (the real train path): tensor-core matmuls in bf16, ~1.5-2x faster than fp32.
    # The GDN scan internally forces fp32 for stability, so this is safe; mirrors train.py.
    amp = torch.cuda.is_bf16_supported()
    for step in range(1, steps + 1):
        model.train()
        for o in opts:
            o.zero_grad(set_to_none=True)
        xb, yb = loader.batch(B, T, device, rng, pin=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp, cache_enabled=False):
            _, loss = model(
                xb, yb, r=cfg.mean_recurrence
            )  # cache_enabled=False: checkpoint recompute determinism
        # NaN guard: skip step if loss exploded (prevents weight corruption). Throttle the
        # warning so a fully-diverged run doesn't print one line per step (400 lines of spam);
        # and bail early — if nothing has been finite for 50 steps, the config is broken, not
        # warming up. Better to fail fast and loud than burn the full GPU budget on NaNs.
        if torch.isnan(loss) or torch.isinf(loss):
            nan_streak += 1
            if nan_streak <= 3 or nan_streak % 50 == 0:
                print(
                    f"\n  [WARN] step {step}: NaN/inf loss, skipping optimizer step "
                    f"(streak={nan_streak})"
                )
            losses.append(float("nan"))
            tok_total += B * T
            if nan_streak >= 50 and not any(v == v for v in losses):
                print(
                    f"\n  [FAIL] {nan_streak} consecutive NaN/inf losses with no finite step — "
                    f"aborting GPU benchmark (model never trained)."
                )
                checks["GPU benchmark: loss decreases"] = False
                return checks
            continue
        nan_streak = 0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts:
            o.step()
        losses.append(loss.item())
        tok_total += B * T

        if step % 100 == 0 or step == steps:
            valid = [v for v in losses[-50:] if not math.isnan(v) and not math.isinf(v)]
            dt = time.time() - t0
            avg = sum(valid) / len(valid) if valid else float("nan")
            print(
                f"\n  -- step {step}/{steps} | loss={avg:.3f} | {tok_total / dt:,.0f} tok/s | "
                f"{tok_total / 1e6:.1f}M tok --"
            )
            # Skip sampling if model diverged
            if math.isnan(avg) or math.isinf(avg):
                print("    (sampling skipped — model diverged)")
                continue
            print(
                "    (samples below are a 200-step / ~0.4M-token smoke run — expect gibberish; "
                "this benchmark proves throughput + that loss falls, NOT capability. Coherence "
                "needs the real multi-day run.)"
            )
            model.eval()
            for p in prompts:
                pids = tok.encode(p).ids
                p_tok = torch.tensor([pids], dtype=torch.long, device=device)
                plen = len(pids)
                try:
                    with torch.no_grad():
                        s = model.generate(p_tok, 24, temp=0.7, top_k=40, effort=None)
                        lng = model.generate(p_tok, 120, temp=0.7, top_k=40, effort=None)
                    st = tok.decode(s[0, plen:].tolist(), skip_special_tokens=False)
                    lt = tok.decode(lng[0, plen:].tolist(), skip_special_tokens=False)
                    print(f"    prompt: {p!r}")
                    print(f"      24 tok: {st[:80]!r}")
                    print(f"      120 tok: {lt[:200]!r}")
                except Exception as e:
                    print(f"    prompt: {p!r}  (sample failed: {e})")
    valid_losses = [v for v in losses if not math.isnan(v) and not math.isinf(v)]
    total_t = time.time() - t0
    if not valid_losses:  # every step diverged — report, don't crash
        print(
            f"\n  -- DONE: {total_t / 60:.1f} min, {tok_total / total_t:,.0f} tok/s, "
            f"no finite losses --"
        )
        checks["GPU benchmark: loss decreases"] = False
        print(f"  [{mark(False)}] loss decreases (no finite losses — model diverged)")
        return checks
    fl = sum(valid_losses[-20:]) / min(20, len(valid_losses))
    # Contextualize the loss against the random baseline so the number is interpretable: a fresh
    # model over vocab V scores ln(V) nats (= log2(V) bits/tok); how far below that = real signal.
    rand_nats = math.log(loader.vocab_size)
    bits = fl / math.log(2)
    rand_bits = rand_nats / math.log(2)
    print(
        f"\n  -- DONE: {total_t / 60:.1f} min, {tok_total / total_t:,.0f} tok/s, "
        f"loss {valid_losses[0]:.2f} -> {fl:.2f} --"
    )
    print(
        f"     context: {bits:.1f} bits/tok vs {rand_bits:.1f} random (vocab {loader.vocab_size}); "
        f"{100 * (1 - fl / rand_nats):.0f}% below random after only {tok_total / 1e6:.1f}M tok — learning, "
        f"not trained. Real capability = the multi-day run + pipeline.py --elasticity."
    )
    train_ok = bool(valid_losses) and valid_losses[-1] < valid_losses[0]
    checks["GPU benchmark: loss decreases"] = train_ok
    print(f"  [{mark(train_ok)}] loss decreases ({valid_losses[0]:.2f} -> {valid_losses[-1]:.2f})")

    # NOTE: this model uses the GPT-NeoX BPE tokenizer (`tok`), NOT byte-level — encode/decode
    # through `tok`, never bytes() (generated ids exceed 255 and bytes() would raise).
    # Tiny Phase 2 check
    try:
        from tasks import TaskGenerator

        tg = TaskGenerator()
        task = tg.generate()
        tids = torch.tensor([tok.encode(task.prompt).ids[:200]], device=device)
        with torch.no_grad():
            lg, _ = model(tids, r=4)
        checks["GPU benchmark: task forward pass"] = lg is not None
        print(f"  [{mark(lg is not None)}] task RL forward pass (Phase 2)")
    except Exception as e:
        print(f"  [SKIP] task RL: {e}")

    # Tiny Phase 3-4 check
    sft = "user: What is 5 plus 3?\nassistant:"
    sids_list = tok.encode(sft).ids
    sids = torch.tensor([sids_list], device=device)
    model.eval()
    with torch.no_grad():
        a = model.generate(sids, 24, temp=0.6, top_k=30, effort=4)
    atxt = tok.decode(a[0, len(sids_list) :].tolist(), skip_special_tokens=False)
    checks["GPU benchmark: instruct generation"] = len(atxt) > 0
    print(f"  [{mark(len(atxt) > 0)}] instruct generation (Phases 3-4): {atxt[:80]!r}")

    # Effort dial — measured on a REAL in-distribution batch (the model trained on this
    # distribution). A short out-of-distribution string just pins CE at ln(vocab) on an undertrained
    # model and proves nothing. At 200 steps the recurrence hasn't learned to exploit depth yet, so
    # this is only a "no entropy pump" smoke check (more loops must not blow up loss), NOT the
    # capability curve — that's the CPU toy demo + pipeline.py --elasticity on a real checkpoint.
    ed_x, _ed_y = loader.batch(B, T, device, np.random.RandomState(7))
    rmax = cfg.max_recurrence_infer
    ecs = {}
    with torch.no_grad():
        for rt in (1, 2, rmax):
            lg, _ = model(ed_x, r=rt)
            ecs[rt] = float(
                F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)), ed_x[:, 1:].reshape(-1))
            )
    mon = ecs.get(rmax, 999) <= ecs.get(1, 999) + 0.2  # no entropy pump (lenient slack)
    checks["GPU benchmark: effort dial (no entropy pump)"] = mon
    print(
        f"  [{mark(mon)}] effort dial (no entropy pump): CE r1={ecs[1]:.3f} r2={ecs[2]:.3f} "
        f"r{rmax}={ecs[rmax]:.3f}  (real elasticity curve: pipeline.py --elasticity)"
    )

    # Overfit-one-batch sanity (Karpathy's recipe). The 200-step samples can't be coherent yet, so
    # they're weak evidence of learning. Driving ONE fixed batch's loss to ~0 in a few dozen steps is
    # strong, honest evidence that the model + optimizer + grad path can actually fit signal —
    # capacity, not capability. Run last: it overfits (corrupts) the weights, so nothing depends on
    # them afterward. LR is bumped 6x for this throwaway phase (weights are discarded), fp32 (no
    # autocast) to isolate the math from bf16 noise.
    model.train()
    for o in opts:
        for grp in o.param_groups:
            grp["lr"] *= 6
    ob_x, ob_y = loader.batch(B, T, device, np.random.RandomState(0))  # one fixed (real) batch
    ob0 = ob1 = None
    for _ in range(50):
        for o in opts:
            o.zero_grad(set_to_none=True)
        _, l = model(ob_x, ob_y, r=cfg.mean_recurrence)
        if torch.isnan(l) or torch.isinf(l):
            break
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts:
            o.step()
        ob1 = l.item()
        if ob0 is None:
            ob0 = ob1
    overfit_ok = ob0 is not None and ob1 is not None and ob1 < 0.5 * ob0
    checks["GPU benchmark: overfits a single batch"] = overfit_ok
    print(
        f"  [{mark(overfit_ok)}] overfit-one-batch sanity: loss {ob0:.2f} -> {ob1:.2f} "
        f"(model+optimizer+grad path can fit signal — learning capacity, not capability)"
        if ob0 is not None
        else "  [FAIL] overfit-one-batch: no finite loss"
    )

    return checks


__all__ = ["run_cpu_benchmark", "run_gpu_benchmark"]

"""CHARKHA memory probe — VRAM measurement and config validation."""

import random
import torch
from charkha import (
    Charkha,
    Muon,
    NormM,
    clip_grads_mixed,
    install_grad_release,
)
from data import ShardLoader


def _apply_probe_schedule(cfg, args, phase):
    """Mutate cfg to match a REAL scheduled training phase so the probe measures what the run
    actually executes — not the worst-case full-halting-from-step-0 config make_cfg() yields.
    The training loop defers halting (--halt-start-step) and ramps E[r] (--recurrence-curriculum),
    so a fresh make_cfg (halting on, mean_recurrence at its ceiling) over-states the launch peak and
    UNDER-states nothing — but it never matches the actual step-0 run. We reproduce the two phases:
      phase='launch'  = step-0 config: halting OFF if deferred, E[r] at the curriculum START.
      phase='ceiling' = eventual peak: halting ON (target), E[r] ramped to the curriculum END.
    Returns a short human label. Mirrors the schedule logic in train()."""
    curric = getattr(args, "recurrence_curriculum", False)
    r_start = getattr(args, "curric_r_start", None) or cfg.mean_recurrence
    r_end = getattr(args, "curric_r_end", None) or cfg.mean_recurrence
    halt_deferred = getattr(args, "halt_start_step", 0) > 0
    halt_target = cfg.use_recurrence and not getattr(args, "no_halting", False)
    if phase == "launch":
        if curric:
            cfg.mean_recurrence = max(1, int(r_start))
        if halt_deferred:
            cfg.use_halting = False
    else:  # 'ceiling' — the heaviest phase the run reaches
        if curric:
            cfg.mean_recurrence = max(1, int(r_end))
        cfg.use_halting = halt_target
    # train() auto-tracks BPTT depth = ceil(E[r]/2) under curriculum or --bptt-half
    if curric or getattr(args, "bptt_half", False):
        cfg.backprop_depth = max(1, (cfg.mean_recurrence + 1) // 2)
    return (
        f"{phase} E[r]={cfg.mean_recurrence} bptt={cfg.backprop_depth} "
        f"halt={'on' if cfg.use_halting else 'off'}"
    )


def _probe_one(args, loader, T, device, g, phase="launch"):
    """Run real steps + one eval forward at seq_len=T for a given scheduled `phase`. Returns
    (true_peak_G, pytorch_peak_G, overhead_G, steady_tok_s, ok, note). 'true_peak' = the DEDICATED vram
    the process actually holds = PyTorch's peak pool PLUS the CUDA context + cuBLAS/cuDNN/triton
    workspaces PyTorch's counters omit (~0.4-0.8G). That is the number Task Manager shows and what
    actually decides spill — reserved alone understates it (a '7.27G' run really sits at ~7.7G used).
    Prints VRAM after each phase (model.to / opt build / forward / backward / step / eval), flushed,
    so a mid-forward SPILL is visible on the spot instead of the process hanging with no row."""
    import time as _t
    from train import _maybe_cast_param_dtype, build_opt_set, evaluate, make_cfg

    cfg = make_cfg(args)
    if cfg.vocab_size < loader.vocab_size:
        cfg.vocab_size = ((loader.vocab_size + 127) // 128) * 128
    cfg.max_seq_len = max(
        getattr(cfg, "max_seq_len", T), T
    )  # ensure the RoPE table fits the probed T
    sched = _apply_probe_schedule(cfg, args, phase)
    rng = random.Random(args.seed)
    B, accum = args.batch_size, max(1, args.accum_steps)
    nsteps = max(4, getattr(args, "mem_steps", 8) or 8)

    def _mem(tag):
        torch.cuda.synchronize()
        free, tot = torch.cuda.mem_get_info()
        ded = (tot - free) / 1024**3  # what Task Manager shows == real spill trigger
        print(
            f"      [{T:>4} {phase:<7}] {tag:<14} reserved={g(torch.cuda.memory_reserved()):5.2f}G  "
            f"ded.used={ded:5.2f}G",
            flush=True,
        )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        print(f"    -- seq_len={T} [{sched}] --", flush=True)
        model = Charkha(cfg).to(device)
        model = _maybe_cast_param_dtype(model, args, device)
        _mem("model.to(cuda)")
        opts, bases, opt_mode = build_opt_set(model, args, offload=args.offload_optim)
        if getattr(args, "grad_release", False):
            rel_params = [
                p
                for opt in opts
                if isinstance(opt, (Muon, NormM))
                for grp in opt.param_groups
                for p in grp["params"]
            ]
            install_grad_release(rel_params)
        _mem("optimizer built")
        model.train()

        def one_step(probe=False):
            for i in range(accum):
                x, y = loader.batch(B, T, device, rng, pin=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    _, loss = model(
                        x, y
                    )  # cache_enabled=False: see training-loop note (checkpoint determinism)
                if probe and i == 0:
                    _mem("after forward")
                (loss / accum).backward()
                if probe and i == 0:
                    _mem("after backward")
            if getattr(args, "grad_release", False):
                clip_grads_mixed(model, args.grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            for o in opts:
                o.step()
            for o in opts:
                o.zero_grad(set_to_none=True)

        one_step(probe=True)
        _mem("after opt.step")  # step 1: lazy Muon/Adam state alloc + phase prints
        one_step()  # step 2: second warmup before timing
        torch.cuda.synchronize()
        # CUDA context + cuBLAS/cuDNN/triton workspaces live in DEDICATED vram but are NOT counted in
        # PyTorch's reserved pool. Measure that gap so the reported peak matches what Task Manager shows.
        _free_w, _total_b = torch.cuda.mem_get_info()
        overhead = max(0.0, (_total_b - _free_w) / 1024**3 - g(torch.cuda.memory_reserved()))
        # time the WHOLE block once (per-step timing underflows to ~0 on fast steps and explodes tok/s)
        t0 = _t.time()
        timed = max(2, nsteps - 2)
        for _ in range(timed):
            one_step()
        torch.cuda.synchronize()
        steady = timed * B * accum * T / max(_t.time() - t0, 1e-6)

        # measure the REAL (now memory-frugal) eval path too — the val pass is part of the run's peak.
        evaluate(model, loader, B, T, device, rng, iters=1)
        _mem("after eval")
        pytorch_peak = g(torch.cuda.max_memory_reserved())
        true_peak = pytorch_peak + overhead  # what the process actually holds in dedicated vram
        model = None
        opts = None
        torch.cuda.empty_cache()
        return true_peak, pytorch_peak, overhead, steady, True, ""
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        pp = g(torch.cuda.max_memory_reserved())
        torch.cuda.empty_cache()
        return (pp, pp, 0.0, 0.0, False, f"{type(e).__name__}: {str(e).splitlines()[0][:80]}")


def mem_probe(args):
    """VRAM-only probe + SHARED-MEMORY SPILL detector. On Windows/WSL2 (WDDM), a CUDA allocation that
    exceeds the card's DEDICATED vram does NOT OOM — the driver silently backs the overflow with SHARED
    system RAM, which is ~10-50x slower. So a too-big config 'works' but crawls. This runs real steps and
    reports peak reserved vs dedicated total + steady tok/s; with --mem-sweep it tries several seq_lens and
    the tok/s CLIFF marks the spill point. Pick the largest VRAM-ONLY row for the launch config."""
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device != "cuda":
        print("mem-probe needs a CUDA device (--device cuda)")
        return 1
    g = lambda b: b / 1024**3
    _free0, total = torch.cuda.mem_get_info()  # total = physical DEDICATED vram (e.g. ~8G card)
    total_g = g(total)
    budget = getattr(args, "vram_budget", 7.3) or 7.3
    loader = ShardLoader(args.data, split="train")
    sweep = getattr(args, "mem_sweep", None)
    seqs = [int(s) for s in str(sweep).split(",")] if sweep else [args.seq_len]
    B, accum = args.batch_size, max(1, args.accum_steps)
    print(
        f"VRAM PROBE | dedicated total={total_g:.2f}G | VRAM-only budget={budget:.2f}G | "
        f"B={B} accum={accum} ce_chunk={args.ce_chunk} grad_ckpt={getattr(args, 'grad_checkpoint', False)} "
        f"offload={bool(getattr(args, 'offload_optim', False))} "
        f"adam={'8bit' if getattr(args, 'eightbit_optim', False) else 'fp32'} "
        f"optim={'symmetry' if getattr(args, 'symmetry_opt', False) else 'default'}"
    )
    print(
        '  "ded.used" = the DEDICATED vram the process truly holds = PyTorch peak pool + CUDA context /'
    )
    print(
        '  cuBLAS / cuDNN / triton workspaces (the number Task Manager shows; "pytorch" alone understates'
    )
    print(
        "  it). Above the dedicated total -> WDDM spills to SHARED RAM (~10-50x slower). Keep real"
    )
    print(
        "  headroom under the total for the Windows desktop + fragmentation. (peak includes an eval forward)"
    )
    # Schedule-aware: probe the actual step-0 LAUNCH config; if the run defers halting / ramps E[r],
    # ALSO probe the eventual CEILING (halting on, E[r] at its end) so the future peak is visible too.
    phases = ["launch"]
    if getattr(args, "halt_start_step", 0) > 0 or getattr(args, "recurrence_curriculum", False):
        phases.append("ceiling")
        print(
            f"  schedule: halt_start_step={getattr(args, 'halt_start_step', 0):,} "
            f"curriculum r {getattr(args, 'curric_r_start', '?')}->{getattr(args, 'curric_r_end', None) or '?'}"
            f'  => probing BOTH the launch phase and the eventual halting "ceiling".'
        )
    print(
        f"  {'seq_len':>7} {'phase':>7} | {'ded.used':>8} {'pytorch':>8} {'ctx/ovh':>8} | {'tok/s':>8} | verdict"
    )
    print("  " + "-" * 78)
    rows = []
    for T in seqs:
        for phase in phases:
            print(
                f"  running seq_len={T} phase={phase} (warmup + timed train + eval forward)...",
                flush=True,
            )
            tp, pp, ovh, toks, ok, note = _probe_one(args, loader, T, device, g, phase=phase)
            spill = ok and tp > total_g - 0.1
            vram_only = ok and (not spill) and tp <= budget
            verdict = (
                "OOM/CRASH"
                if not ok
                else "SPILL->shared"
                if spill
                else "VRAM-ONLY"
                if vram_only
                else "TIGHT"
            )  # TIGHT = fits dedicated but past budget
            print(
                f"  {T:>7} {phase:>7} | {tp:>7.2f}G {pp:>7.2f}G {ovh:>7.2f}G | {toks:>8,.0f} | {verdict}"
                + ("  <--" if vram_only else "")
            )
            if note:
                print(f"          {note}")
            rows.append((T, phase, vram_only, toks, tp, ok, spill))
    print("  " + "-" * 78)

    def _flag_str():
        return (
            "--ce-chunk {c} --gdn-chunk {gc} --grad-checkpoint".format(
                c=args.ce_chunk, gc=getattr(args, "gdn_chunk", None)
            )
            + (
                " --offload-optim"
                if getattr(args, "offload_optim", False)
                else " --no-offload-optim"
            )
            + (" --8bit-optim" if getattr(args, "eightbit_optim", False) else "")
            + (" --symmetry-opt" if getattr(args, "symmetry_opt", False) else "")
            + (" --bptt-half" if getattr(args, "bptt_half", False) else "")
        )

    # A seq_len is "fully safe" only if EVERY probed phase has real headroom; "launch-safe" if at least
    # the step-0 phase does (you can start now, but the ceiling phase needs attention before it arrives).
    by_T = {}
    for T, phase, vo, toks, tp, ok, spill in rows:
        by_T.setdefault(T, {})[phase] = (vo, toks, tp, ok, spill)
    fully_safe = [T for T in by_T if all(by_T[T][p][0] for p in phases)]
    launch_safe = [T for T in by_T if by_T[T].get("launch", (False,))[0]]
    if fully_safe:
        best = max(fully_safe)
        lp = by_T[best]["launch"]
        print(
            f"  GREEN: largest config safe across ALL phases = seq_len={best}  "
            f"({lp[2]:.2f}G launch, {lp[1]:,.0f} tok/s)."
        )
        print(
            f"  -> launch: --batch-size {B} --seq-len {best} --accum-steps {accum} " + _flag_str()
        )
        return 0
    if launch_safe:
        best = max(launch_safe)
        cl = by_T[best].get("ceiling")
        warn = ""
        if cl is not None:
            cstate = "SPILLS" if cl[4] else ("TIGHT/over-budget" if not cl[0] else "ok")
            warn = (
                f"  but the halting CEILING phase is {cstate} ({cl[2]:.2f}G). You can START here, "
                f"but lower seq / chunk further BEFORE halt_start_step={getattr(args, 'halt_start_step', 0):,}."
            )
        print(f"  AMBER: seq_len={best} is safe to LAUNCH ({by_T[best]['launch'][2]:.2f}G).{warn}")
        print(
            f"  -> launch: --batch-size {B} --seq-len {best} --accum-steps {accum} " + _flag_str()
        )
        return 0
    # nothing safe under budget — fall back to "fits dedicated at all" (TIGHT) on the launch phase
    fits = [
        T
        for T in by_T
        if by_T[T].get("launch") and by_T[T]["launch"][3] and not by_T[T]["launch"][4]
    ]
    if fits:
        b = max(fits)
        print(
            f"  AMBER: nothing left real headroom under {budget:.1f}G, but seq_len={b} FITS dedicated "
            f"at {by_T[b]['launch'][2]:.2f}G on launch (TIGHT — one transient from spilling)."
        )
        print(
            f"  Safer to go smaller; or accept the risk: --seq-len {b} --accum-steps {accum} "
            + _flag_str()
        )
        return 0
    print(
        "  RED: even the smallest config spills into shared memory. Scale down further "
        "(--small for the 0.3B config, lower --seq-len / --gdn-chunk / --ce-chunk, or --offload-optim)."
    )
    return 2


# --------------------------------------------------------------------------
# Live sampling: generate from a fixed prompt every --sample-every steps so the
# run is watchable - you see the *same* continuation sharpen from noise to text.
# Opt-in, try/except-guarded (a sampling glitch must never kill a 90-day run).
# --------------------------------------------------------------------------


def _build_codec(cfg):
    """(encode, decode) for live samples. Small vocab => byte-level toy shards; otherwise the
    tokenizer that actually produced the shards (cfg.tokenizer_name, recorded by ShardLoader from
    dataprep's index.json), falling back to GPT-NeoX only for legacy shards predating that field.
    Hardcoding neox here made every dashboard sample and capability milestone on a custom-tokenizer
    run (Sutra-131k/v8) encode prompts with the WRONG vocab and decode ids as garbage — milestones could never
    unlock. Falls back to printing raw token ids if nothing loads."""
    if cfg.vocab_size <= 512:  # byte-level toy/selftest shards
        return (
            lambda s: list(s.encode("utf-8")),
            lambda ids: bytes(b & 0xFF for b in ids).decode("utf-8", errors="replace"),
        )
    name = getattr(cfg, "tokenizer_name", None)
    if name:
        try:
            from serve import load_tokenizer_for  # resolves a local tokenizer.json or a hub name

            tok = load_tokenizer_for(name)
            return (lambda s: tok.encode(s)), (lambda ids: tok.decode(ids))
        except Exception as e:
            print(f"[sample] cfg tokenizer {name!r} unavailable ({e}); will print token ids")
    return (lambda s: []), (lambda ids: " ".join(map(str, ids)))


def _emit_sample(model, codec, args, device, step):
    enc, dec = codec
    was_training = model.training
    try:
        ids = enc(args.sample_prompt) or [0]  # empty prompt -> seed from id 0 (BOS-ish)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(
            x,
            args.sample_tokens,
            effort=args.sample_effort,
            temp=args.sample_temp,
            top_k=args.sample_top_k,
        )
        gen = dec(out[0, len(ids) :].tolist())
        shown = (args.sample_prompt or "<bos>") + "│" + gen  # '|' marks prompt/gen boundary
        print(f"  [sample @ {step}] {shown!r}")
    except Exception as e:  # never let sampling crash the run
        print(f"  [sample @ {step}] skipped ({type(e).__name__}: {e})")
    finally:
        if was_training:
            model.train()  # generate() flips to eval(); restore


# --------------------------------------------------------------------------
# Live capability milestones: tiny prompts the model should eventually nail, ordered easy->hard.
# Each one "unlocks" the first eval where it passes, so a multi-day run is a checklist filling in --
# "can it do 1+1 yet? can it finish 'Hello wor'?" -- not just a loss number. Unlocked milestones keep
# getting RE-TESTED every eval (not skipped) and track a consistency %% -- a model that nails 1+1 once
# at step 4,500 then forgets it under later data/curriculum drift should show that, not a permanent [x].
# Heuristic + lenient on purpose (engagement, not a gate); guarded so a probe can never kill the run.
# --------------------------------------------------------------------------


def _firstint(s):
    import re

    m = re.search(r"-?\d+", s)
    return m.group(0) if m else None


_MILESTONES = [
    # (label, prompt, check(generated_text) -> bool)
    (
        "coherent text",
        "The ",
        lambda g: g.strip() != "" and sum(c.isalpha() or c.isspace() for c in g) >= 0.7 * len(g),
    ),
    ("repeats a pattern", "cat cat cat cat ", lambda g: "cat" in g.lower()),
    ("completes 'Hello wor'", "Hello wor", lambda g: g.lower().lstrip().startswith("ld")),
    ("1 + 1 = 2", "1 + 1 = ", lambda g: _firstint(g) == "2"),
    ("2 + 2 = 4", "2 + 2 = ", lambda g: _firstint(g) == "4"),
    ("7 + 5 = 12", "7 + 5 = ", lambda g: _firstint(g) == "12"),
    ('"Hi! How are" -> you', "Hi! How are ", lambda g: "you" in g.lower()),
    ("the sky is blue", "The sky is ", lambda g: "blue" in g.lower()),
    ("capital of France", "The capital of France is ", lambda g: "paris" in g.lower()),
    ("opposite of hot", "The opposite of hot is ", lambda g: "cold" in g.lower()),
]


def _milestone_list(args):
    """The built-in capability ladder PLUS any user goals from --probe 'PROMPT=>EXPECTED' (repeatable):
    e.g. --probe 'Hi! =>hello' --probe 'Q: 2+3? A:=>5'. Unlocks when the generation contains EXPECTED."""
    items = list(_MILESTONES)
    for spec in getattr(args, "probe", None) or []:
        if "=>" in spec:
            p, e = spec.split("=>", 1)
            e = e.strip()
            items.append(
                (f"you: {p.strip()!r} -> {e!r}", p, (lambda g, _e=e.lower(): _e in g.lower()))
            )
    return items


def _run_milestones(model, codec, args, device, step, unlocked, n_new=14):
    """Greedy-generate from EVERY probe -- including already-unlocked ones, not just locked ones --
    so a milestone keeps getting RE-TESTED after it first passes. Each record is {'first': step it
    first passed, 'checks': re-checks since (incl. the unlock check itself), 'passes': how many of
    those passed}, shown as a consistency %% next to the unlock step. A milestone that was a one-time
    fluke (passed once, then the model drifts off it under later data) shows up as a falling %%, not a
    permanent checkmark. Legacy checkpoints store a bare int (the old format, just the unlock step) --
    migrated to the new dict on first re-check. Returns the panel lines. Never raises -- a glitch must
    not kill a 90-day run."""
    enc, dec = codec
    mset = _milestone_list(args)
    was_training = model.training
    try:
        for label, prompt, check in mset:
            try:
                ids = enc(prompt) or [0]
                x = torch.tensor([ids], dtype=torch.long, device=device)
                out = model.generate(
                    x, n_new, effort=getattr(args, "sample_effort", None), temp=0.1, top_k=1
                )  # near-greedy, deterministic
                passed = bool(check(dec(out[0, len(ids) :].tolist())))
            except Exception:
                passed = False
            rec = unlocked.get(label)
            if rec is not None and not isinstance(rec, dict):  # legacy ckpt: bare int = unlock step
                rec = {"first": rec, "checks": 1, "passes": 1}
            if rec is None:
                if passed:
                    unlocked[label] = {"first": step, "checks": 1, "passes": 1}
            else:
                rec["checks"] += 1
                rec["passes"] += 1 if passed else 0
                unlocked[label] = rec
    finally:
        if was_training:
            model.train()
    got = sum(1 for lbl, _p, _c in mset if lbl in unlocked)
    lines = [f"  | MILESTONES {got}/{len(mset)} unlocked:"]
    for label, _p, _c in mset:
        rec = unlocked.get(label)
        if rec is not None:
            pct = 100.0 * rec["passes"] / max(1, rec["checks"])
            lines.append(
                f"  |   [x] {label}  (@ step {rec['first']:,}, "
                f"{pct:.0f}% consistent x{rec['checks']})"
            )
        else:
            lines.append(f"  |   [ ] {label}")
    return lines


def _gen_text(model, codec, args, device, prompt, n=48, temp=0.7, top_k=40):
    """Generate continuation text from a prompt (for the dashboard's live 'writes:' sample). Restores
    train mode; never raises. Returns the decoded continuation."""
    enc, dec = codec
    was = model.training
    try:
        ids = enc(prompt) or [0]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(
            x, n, effort=getattr(args, "sample_effort", None), temp=temp, top_k=top_k
        )
        return dec(out[0, len(ids) :].tolist())
    except Exception as e:
        return f"<gen failed: {type(e).__name__}>"
    finally:
        if was:
            model.train()


# --------------------------------------------------------------------------
# Self-test: synthetic shards -> train -> resume -> branch-decay, all on CPU.
# --------------------------------------------------------------------------


__all__ = ["mem_probe"]

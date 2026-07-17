"""CHARKHA feature probes — confidence, effort dial, convergence, reasoning."""

import os
import tempfile
import torch
import torch.nn.functional as F
from charkha import Charkha, CharkhaConfig, build_optimizers
from train import make_synthetic_shards, _toy_args, load_ckpt


def run_feature_probes():
    # Imported lazily to avoid a module cycle: preflight imports this function after defining
    # the shared reporting and probe helpers.
    from preflight import _decode, _honesty_probe, hr, mark
    import shutil
    from train import train

    checks = {}
    tmp = tempfile.mkdtemp(prefix="charkha_preflight_")
    rp_tmp = None
    data_dir, out_dir = os.path.join(tmp, "data"), os.path.join(tmp, "run")
    make_synthetic_shards(data_dir, n_shards=3, toks_per=6000)

    hr("2) PROBE-MODEL WARMUP — a quick toy run that produces the model the probes interrogate")
    print("  (The long, watchable generation-sharpening showcase moved to demo.py --integrated.)")
    print("  Trains a tiny model ~140 steps through the curriculum + halter phasing so the feature")
    print(
        "  probes below have a real trained model + confidence head to read. │ marks prompt | gen.\n"
    )
    try:  # untrained baseline so the noise->text arc is visible
        bcfg = CharkhaConfig.toy()
        bcfg.vocab_size = 384
        bmod = Charkha(bcfg).eval()
        with torch.no_grad():
            out = bmod.generate(
                torch.tensor([list(b"the ")], dtype=torch.long), 24, temp=0.8, top_k=50
            )
        print(f"  [sample @ untrained] {('the ' + '│' + _decode(out[0, 4:].tolist()))!r}")
    except Exception as e:
        print(f"  [baseline sample skipped: {type(e).__name__}: {e}]")

    # one rich run exercises: curriculum + halter phasing + ponder anneal + live sampling + eval + snapshot
    train(
        _toy_args(
            steps=140,
            out=out_dir,
            data=data_dir,
            val_frac=0.2,
            recurrence_curriculum=True,
            curric_r_start=1,
            curric_r_end=3,
            curric_steps=90,
            halt_start_step=70,
            ponder_anneal_steps=30,
            sample_every=20,
            sample_tokens=24,
            sample_prompt="the ",
            eval_every=40,
            eval_iters=4,
            log_every=40,
            snapshot_every=70,
        )
    )
    checks["integrated run completes + checkpoints"] = os.path.exists(
        os.path.join(out_dir, "ckpt.pt")
    )

    hr("3) FEATURE PROBES — evidence each headline behavior works")
    model, cfg, opts, bases, ck = load_ckpt(os.path.join(out_dir, "ckpt.pt"), "cpu")
    model.eval()
    enc = lambda s: list(s.encode("utf-8"))
    in_ids = torch.tensor([enc("the people build their own tools and learn ")], dtype=torch.long)

    # A. calibrated confidence = the anti-hallucination substrate. The clean demo model above is
    # ~100% on its corpus, so its conf head never saw an error and saturates - useless to test. We
    # train a SEPARATE tiny model on text+noise (the conf head then gets real wrong examples) and
    # check it is confident on learnable text but unsure on pure noise: "knows when it can't predict".
    print("  ... training a tiny calibration model on text+noise (~1 min, quiet) ...")
    ct, cn = _honesty_probe(tmp)
    sep = ct > cn
    checks["honesty: confident on learnable text, unsure on noise"] = sep
    print(
        f"  [{mark(sep)}] honesty / anti-hallucination: conf on text={ct:.3f} vs on noise={cn:.3f} "
        f"-> {'separates (knows when it cannot predict)' if sep else 'FLAT (conf undertrained)'}"
    )

    # B. effort dial runs at multiple loop counts (compute-elasticity mechanism)
    xb = torch.tensor(
        [enc("the people build their own tools and learn the shape of ")], dtype=torch.long
    )
    ces = {}
    with torch.no_grad():
        for r in (1, 2, 4):
            lg, _ = model(xb, r=r)
            ces[r] = float(
                F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)), xb[:, 1:].reshape(-1))
            )
    ran = all(c == c for c in ces.values())  # all finite
    checks["effort dial runs at r=1,2,4"] = ran
    print(
        f"  [{mark(ran)}] effort dial (elasticity mechanism): CE r1={ces[1]:.3f} r2={ces[2]:.3f} "
        f"r4={ces[4]:.3f}  (real curve: pipeline.py --elasticity)"
    )

    # C. convergence signal (A2 trajectory) exposed at inference
    cv_ok = False
    try:
        model.cfg.track_convergence = True
        model.cfg.use_halting = False
        with torch.no_grad():
            model(in_ids, r=4)
        cv = model._last_convergence
        cv_ok = cv is not None and bool(torch.isfinite(cv).all())
        m = float(cv.mean()) if cv is not None else float("nan")
        print(
            f"  [{mark(cv_ok)}] convergence signal (A2): per-token settle-error exposed & finite "
            f"(raw mean={m:.3g}; calibrate the error->confidence map on a real ckpt)"
        )
    except Exception as e:
        print(f"  [FAIL] convergence signal: {type(e).__name__}: {e}")
    finally:
        model.cfg.track_convergence = False
    checks["convergence signal (A2) exposed at inference"] = cv_ok

    # D. the honesty/effort heads are wired
    heads_ok = all(hasattr(model, h) for h in ("conf_head", "halt_head", "mtp_proj"))
    checks["heads wired (confidence / halter / MTP)"] = heads_ok
    print(
        f"  [{mark(heads_ok)}] heads wired: confidence (P(IK)) + PonderNet halter + multi-token-prediction"
    )

    # ── F. GDN2 speed probe: verify GDN2 forward pass is finite (NOT fast — pure Python for-loop) ──
    print("  ... building GDN2 model to verify correctness ...")
    gdn2cfg = CharkhaConfig.toy()
    gdn2cfg.use_gdn2 = True
    gdn2cfg.use_recurrence = False
    gdn2cfg.use_halting = False
    gdn2cfg.n_prelude = 1
    gdn2cfg.n_core = 1
    gdn2cfg.n_coda = 1
    gdn2model = Charkha(gdn2cfg).to(torch.device("cpu"))
    gdn2x = torch.randint(0, 256, (1, 8), dtype=torch.long)
    import time as _time

    t0 = _time.time()
    with torch.no_grad():
        gdn2lg, _ = gdn2model(gdn2x)
    gdn2_t = _time.time() - t0
    gdn2_finite = bool(torch.isfinite(gdn2lg).all()) and gdn2lg.shape[-1] == 256
    checks["GDN2: forward pass produces finite output"] = gdn2_finite
    print(
        f"  [{mark(gdn2_finite)}] GDN2 forward: finite={gdn2_finite} "
        f"({gdn2_t * 1000:.1f}ms @ T={gdn2x.size(-1)} — exact PyTorch recurrence. "
        f"Use --scale-data on Linux/CUDA before spending cloud time; --no-gdn2 is the ablation escape hatch.)"
    )

    # ── G. Safe frontier probe: all default frontier features ──
    print("  ... building safe-frontier model (default experimental stack) ...")
    ecfg = CharkhaConfig.toy()
    ecfg.vocab_size = 384
    ecfg.d_model = 320
    ecfg.n_heads = 8
    ecfg.n_kv_heads = 2
    ecfg.d_ff = 960
    ecfg.n_prelude = 2
    ecfg.n_core = 3
    ecfg.n_coda = 2
    ecfg.use_recurrence = True
    ecfg.mean_recurrence = 2
    ecfg.max_recurrence_train = 3
    ecfg.use_halting = True
    ecfg.backprop_depth = 2
    ecfg.use_loop_embed = True
    ecfg.effective_depth_scale = True
    ecfg.use_bipolar_gate = True
    ecfg.use_osdn = True
    ecfg.use_gdn2 = True
    ecfg.use_mtp_routing = True
    ecfg.use_nitp = True
    ecfg.use_deep_supervision = True
    ecfg.cross_loop_consistency = True
    ecfg.mtp_weight = 0.1
    ecfg.conf_weight = 0.05
    ecfg.playground_dim = 32
    ecfg.max_play_steps = 3
    ecfg.use_thermostat = True
    ecfg.track_convergence = True
    ecfg.use_accel_exit = True
    ecfg.sngp_enabled = True
    ecfg.sngp_rff_dim = 64
    ecfg.sngp_ridge = 1.0
    ecfg.sngp_scale = 1.0
    ecfg.grad_checkpoint = False
    ecfg.ce_chunk = 1024
    e_built = False
    try:
        emodel = Charkha(ecfg).to(torch.device("cpu"))
        e_built = True
        print(
            f"  [{mark(e_built)}] safe-frontier model built: {sum(p.numel() for p in emodel.parameters()):,} params"
        )
    except Exception as e:
        print(f"  [FAIL] safe-frontier model build crashed: {type(e).__name__}: {e}")
    checks["safe-frontier: model builds"] = e_built

    if e_built:
        ex = torch.randint(0, 384, (2, 16), dtype=torch.long)
        emodel.train()
        e_forw_ok = False
        try:
            with torch.no_grad():
                elog, econf = emodel(ex, targets=None, r=2)
            e_forw_ok = torch.isfinite(elog).all() and torch.isfinite(econf).all()
            # training pass returns (None, loss) — check loss is finite
            _, e_train_loss = emodel(ex, targets=ex, r=2)
            e_forw_ok = e_forw_ok and e_train_loss is not None and torch.isfinite(e_train_loss)
        except Exception as exx:
            print(f"      forward crashed: {type(exx).__name__}: {exx}")

        if e_forw_ok:
            emuon, eadam = build_optimizers(emodel, muon_lr=0.02, adam_lr=3e-3)
            elosses = []
            for _st in range(30):
                emuon.zero_grad(set_to_none=True)
                eadam.zero_grad(set_to_none=True)
                e_x = torch.randint(0, 384, (2, 16), dtype=torch.long)
                _, e_l = emodel(e_x, targets=e_x, r=2)
                if torch.isfinite(e_l):
                    e_l.backward()
                    torch.nn.utils.clip_grad_norm_(emodel.parameters(), 1.0)
                    emuon.step()
                    eadam.step()
                    elosses.append(e_l.item())
            # With all 20+ experimental features + SNGP running, the toy model has 10.5M params
            # and many competing regularizers. Check that it doesn't diverge rather than requiring
            # strict decrease in the first 30 noisy toy steps.
            e_trains = len(elosses) >= 15 and elosses[-1] < elosses[0] * 1.5
            checks["safe-frontier: trains stably (finite, no divergence)"] = e_trains
            print(
                f"  [{mark(e_trains)}] safe-frontier: trains stably "
                f"({elosses[0]:.3f} -> {elosses[-1]:.3f}, {len(elosses)}/30 finite)"
            )

        if hasattr(emodel, "sngp_head") and hasattr(emodel.sngp_head, "precision"):
            emodel.eval()
            with torch.no_grad():
                _ = emodel(torch.randint(0, 384, (1, 8), dtype=torch.long))
            sngp_prec = bool(torch.isfinite(emodel.sngp_head.precision).all())
            sngp_var = hasattr(emodel, "_last_sngp_var") and emodel._last_sngp_var is not None
            checks["safe-frontier: SNGP precision+variance"] = sngp_prec and sngp_var
            print(f"  [{mark(sngp_prec and sngp_var)}] safe-frontier: SNGP precision + variance")

    if rp_tmp:
        shutil.rmtree(rp_tmp, ignore_errors=True)
    return checks, tmp


COVERAGE = {
    "Architecture": [
        ("GDN sequence mixer (chunk_gated_delta_rule / CPU fallback)", "charkha --toy"),
        ("GQA attention + QK-RMSNorm + RoPE + FlashAttention", "charkha --toy"),
        ("Depth-recurrent core + loop-index embedding", "charkha --toy + demo"),
        ("GDN-2 channel-wise erase+write gates (arXiv:2605.22791)", "preflight probe F"),
        ("OSDN key preconditioning (arXiv:2605.13473)", "charkha --toy + experiment CLI"),
        ("Bipolar sign-gating (STE, discrete facts)", "experiment CLI + preflight G"),
        ("Effective depth scaling (residual variance control)", "charkha --toy + preflight G"),
        ("SwiGLU FFN + RMSNorm throughout", "charkha --toy"),
    ],
    "Adaptive compute / honesty": [
        ("PonderNet halting (learned per-token effort)", "demo (r..h) + charkha --toy"),
        ("Halter phasing Phase 1 fixed -> Phase 2 halt", "train #8 + demo"),
        ("Confidence head P(IK) = anti-hallucination", "preflight probe A"),
        ("Convergence signal A2 (extrapolation)", "preflight probe C + charkha --toy"),
        (
            "Acceleration early-exit (two-scale-latent, arXiv:2509.23314)",
            "experiment CLI + charkha --toy",
        ),
        ("SNGP epistemic variance (Random Fourier Features)", "experiment CLI + preflight G"),
        ("Laplace-Redux post-hoc uncertainty (arXiv:2106.14806)", "laplace-enabled CLI flag"),
        ("RLCM margin confidence (arXiv:2604.23333)", "selfteach selftest"),
        (
            "Epistemic thermostat (override greedy halting on oscillation)",
            "experiment CLI + preflight G",
        ),
        (
            "Neuromodulated gating (removed after audit)",
            "removed; stale checkpoints load via from_dict",
        ),
    ],
    "Recurrent core mechanisms": [
        ("Loop-index embedding (sinusoidal depth signal)", "charkha --toy"),
        ("Truncated BPTT (backpropagate last k loops)", "train #7 + preflight G"),
        ("Per-sequence recurrence (LoopWM, arXiv:2606.18208)", "--per-seq-recurrence CLI"),
        (
            "Active inference error feedback loop to loop (removed)",
            "removed after divergence/NaN audit",
        ),
        ("MTP-routed macro-states (future-rep into core)", "experiment CLI + preflight G"),
        ("Cross-loop consistency (A1 contractive trajectory)", "experiment CLI + preflight G"),
        ("Deep supervision (A3 anytime depth-recurrence)", "experiment CLI + preflight G"),
        ("Task RL: advantage-weighted regression + critic", "experiment CLI"),
    ],
    "Training": [
        ("Muon + AdamW (+ symmetry-opt set)", "train #6"),
        ("MTP / NITP / deep-supervision aux losses", "charkha --toy"),
        ("Recurrence curriculum (anneal E[r] low to high)", "train #7 + demo"),
        ("Auto-track backprop_depth = ceil(E[r]/2)", "train loop (fixed)"),
        ("Fused chunked CE (8GB backward) + grad-checkpoint", "train #1"),
        ("Grad accumulation + soft-label distillation", "train #4/#5"),
        ("Checkpoint / resume / branch-for-release", "train #1-3"),
        ("Immortal snapshots + bf16 guard + resilient launcher", "train #1 + preflight"),
        ("Live sampling + dashboard + capability milestones", "demo + train #9"),
        ("Style/intent orthogonal contrastive loss (removed)", "removed after config audit"),
        ("NITP next-implicit-token prediction (arXiv:2605.24956)", "experiment CLI"),
        ("All experimental features exposed via CLI flags", "train.py --help"),
    ],
    "Data / serve / eval": [
        ("quality -> dedup -> PII -> decontam -> tokenize (no license gate)", "dataprep selftest"),
        ("digit-split tokenization (math-from-the-start)", "dataprep selftest"),
        ("content_type:code filter + HF Xet + resume", "dataprep selftest"),
        ("Multi-dir loader + vocab-mismatch guard", "train selftest"),
        ("Serve: grounding, thinking, memory, calc tool, effort escalation", "serve selftest"),
        ("Retrieval (BM25 + MiniLM) + RAG in serve", "retrieval + serve selftest"),
        ("SFT: completion-masked instruction finetuning", "sft selftest"),
        ("Self-improvement loop (train->eval->promote)", "selfteach selftest"),
        ("Frontier seq-level KD turnkey (traces->tokenize->shards)", "pipeline selftest"),
        ("Compute-elasticity harness (E1) + train-shallow/infer-deep", "pipeline selftest"),
        ("Shard integrity verifier", "verify_shards selftest"),
        ("Same-arch weight merge (soup / slerp / TIES)", "merge selftest"),
        ("Exact document cross-dedup (blake2b, consis with dataprep)", "crossdedup selftest"),
    ],
}


__all__ = ["run_feature_probes"]

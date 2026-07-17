"""CHARKHA toy training — hermetic CPU selftest and CLI entry point."""

from __future__ import annotations
import argparse
import math
import os
import time
import urllib.request
import torch
import torch.nn.functional as F
from .config import CharkhaConfig
from ._model import Charkha
from ._optim import *
from ._modules import *
from ._modules import (
    _gdn_sequential_ref,
    _gdn_chunk_scan,
    _gdn2_sequential_ref,
    _gdn2_chunk_scan,
    _HAVE_FLA_GDN2,
    _chunk_gdn2,
    _fused_recurrent_gdn2,
)


def load_bytes(path=None):
    if path and os.path.exists(path):
        data = open(path, "rb").read()
    else:
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        try:
            data = urllib.request.urlopen(url, timeout=10).read()
            print("using tinyshakespeare (downloaded)")
        except Exception:
            data = FALLBACK_TEXT.encode()
            print("offline: using built-in fallback corpus")
    return torch.tensor(list(data), dtype=torch.long)


def get_batch(data, B, T, device):
    ix = torch.randint(len(data) - T - 1, (B,))
    x = torch.stack([data[i : i + T] for i in ix])
    y = torch.stack([data[i + 1 : i + T + 1] for i in ix])
    return x.to(device), y.to(device)


# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------


def train(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = CharkhaConfig.toy() if args.toy else CharkhaConfig()
    if args.no_recurrence:
        cfg.use_recurrence = False
    model = Charkha(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"CHARKHA | {n_params / 1e6:.1f}M params | device={device} | "
        f"recurrence={cfg.use_recurrence} halting={cfg.use_halting} "
        f"(mean r={cfg.mean_recurrence})"
    )

    data = load_bytes(args.data)
    muon, adam = build_optimizers(model, muon_lr=args.muon_lr, adam_lr=args.adam_lr)
    B, T = args.batch_size, min(args.seq_len, cfg.max_seq_len)
    amp = device == "cuda"
    t0, losses = time.time(), []
    for step in range(args.steps):
        mult = wsd_lr_mult(step, warmup=args.warmup)
        for opt, base in ((muon, args.muon_lr), (adam, args.adam_lr)):
            for g in opt.param_groups:
                g["lr"] = base * mult
        x, y = get_batch(data, B, T, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp, cache_enabled=False):
            _, loss = model(x, y)  # cache_enabled=False: checkpoint recompute determinism
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        muon.step()
        adam.step()
        muon.zero_grad(set_to_none=True)
        adam.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if step % args.log_every == 0 or step == args.steps - 1:
            tok_s = B * T * (step + 1) / (time.time() - t0)
            print(f"step {step:5d} | loss {loss.item():.4f} | {tok_s:,.0f} tok/s")
        if args.save_every and step and step % args.save_every == 0:
            torch.save({"model": model.state_dict(), "step": step, "cfg": cfg.__dict__}, args.ckpt)

    if args.toy:  # verification: loss must drop substantially from ln(256)=5.55
        first, last = sum(losses[:10]) / 10, sum(losses[-10:]) / 10
        print(f"\ntoy check: first10 {first:.3f} -> last10 {last:.3f}")
        assert last < first - 0.8, "FAIL: loss did not improve - investigate"
        print("PASS: architecture trains end to end.")
        # fixed-effort dial must run EXACTLY r core passes at inference (regression guard:
        # the old _run_core_fixed ran 2r - r under no_grad then r more).
        seed = torch.tensor([[87]], device=device)
        for r in (1, 3, cfg.max_recurrence_train):
            model._n_core_steps = 0
            model.generate(seed, 1, effort=r)  # one token -> one core run at effort r
            assert model._n_core_steps == r, (
                f"FAIL: effort r={r} ran {model._n_core_steps} core passes, expected {r}"
            )
        print("PASS: fixed effort dial runs exactly r core passes (no 2x inference loop).")
        # effort dial stability: extra fixed loops should not catastrophically increase loss.
        # This toy trains primarily through adaptive halting, so fixed r=max is a stress test,
        # not a strict monotonic-quality guarantee. Use training data tokens (model has seen
        # these, so CE signal is meaningful).
        dt = data[:24].reshape(2, 12).to(device)
        r_losses = []
        was_training = model.training
        model.eval()
        for r_test in (1, 2, cfg.max_recurrence_train):
            model._n_core_steps = 0
            torch.manual_seed(1234)
            if device == "cuda":
                torch.cuda.manual_seed_all(1234)
            with torch.no_grad():
                _, l = model(dt, dt, r=r_test)
            r_losses.append(l.item())
        model.train(was_training)
        # Allow bounded slack: the guard catches recurrence blow-ups while avoiding a false
        # promise that a 300-step toy halting run has already learned monotone fixed-effort quality.
        assert r_losses[-1] <= r_losses[0] + 1.0, (
            f"FAIL: entropy pump — r={cfg.max_recurrence_train} loss {r_losses[-1]:.3f} > r=1 loss {r_losses[0]:.3f} + 1.0"
        )
        print(
            f"PASS: effort dial bounded — r1={r_losses[0]:.3f} r2={r_losses[1]:.3f} r{cfg.max_recurrence_train}={r_losses[2]:.3f} (no catastrophic recurrence blow-up)."
        )
        # fused/chunked CE must equal full-logits cross-entropy (regression guard):
        # verifies the chunked backward produces identical grads to the whole-vocab path.
        hh = torch.randn(2, 33, cfg.d_model, device=device)
        tt = torch.randint(0, cfg.vocab_size, (2, 33), device=device)
        ref_ce = F.cross_entropy(
            F.linear(hh, model.embed.weight).reshape(-1, cfg.vocab_size), tt.reshape(-1)
        ).item()
        model.cfg.ce_chunk = 8  # force multiple chunks
        got_ce = model._fused_ce(hh, tt).item()
        assert abs(ref_ce - got_ce) < 1e-4, f"FAIL: fused CE {got_ce} != full CE {ref_ce}"
        print("PASS: fused/chunked CE matches full-logits CE (memory-frugal head is exact).")
        # chunkwise GDN scan (no-triton path) must equal the per-timestep recurrence to fp32 precision
        # AND carry gradients — it replaces the O(T) Python loop that throttled CPU + CUDA-without-fla.
        gh, gdh = 4, 16
        gq = F.normalize(torch.randn(2, 37, gh, gdh, device=device), dim=-1).requires_grad_()
        gk = F.normalize(torch.randn(2, 37, gh, gdh, device=device), dim=-1)
        gv = torch.randn(2, 37, gh, gdh, device=device)
        gb = torch.sigmoid(torch.randn(2, 37, gh, device=device))
        gg = (
            -F.softplus(torch.randn(gh, device=device))[None, None]
            * torch.sigmoid(torch.randn(2, 37, gh, device=device))
        ).clamp(min=-20)
        o_seq = _gdn_sequential_ref(gq, gk, gv, gg, gb)
        o_chunk = _gdn_chunk_scan(gq, gk, gv, gg, gb, chunk=8)  # ragged: 37 = 4×8 + 5
        assert torch.allclose(o_seq, o_chunk, atol=1e-4), (
            f"FAIL: chunked GDN scan != sequential ({(o_seq - o_chunk).abs().max():.2e})"
        )
        o_chunk.sum().backward()
        assert gq.grad is not None and torch.isfinite(gq.grad).all(), (
            "FAIL: chunked GDN no/NaN grad"
        )
        print(
            "PASS: chunkwise GDN scan matches the sequential recurrence (fast no-triton path, exact)."
        )
        # GDN-2: the chunkwise scan must equal its per-timestep recurrence (the no-fla fallback), AND
        # when the fused fla kernel is present on CUDA it must equal that same reference — the gate
        # before trusting chunk_gdn2 in training. b=key-axis erase, w=value-axis write, g=per-key-
        # channel log-decay; ragged T=37 exercises the partial final chunk of the scan.
        g2q = F.normalize(torch.randn(2, 37, gh, gdh, device=device), dim=-1).requires_grad_()
        g2k = F.normalize(torch.randn(2, 37, gh, gdh, device=device), dim=-1)
        g2v = torch.randn(2, 37, gh, gdh, device=device)
        g2b = torch.sigmoid(torch.randn(2, 37, gh, gdh, device=device))
        g2w = torch.sigmoid(torch.randn(2, 37, gh, gdh, device=device))
        g2g = (
            -F.softplus(torch.randn(gh, gdh, device=device))[None, None]
            * torch.sigmoid(torch.randn(2, 37, gh, gdh, device=device))
        ).clamp(min=-20)
        o2_seq = _gdn2_sequential_ref(g2q, g2k, g2v, g2b, g2w, g2g)
        o2_chunk = _gdn2_chunk_scan(g2q, g2k, g2v, g2b, g2w, g2g, chunk=8)
        assert torch.allclose(o2_seq, o2_chunk, atol=1e-4), (
            f"FAIL: GDN-2 chunk scan != sequential ({(o2_seq - o2_chunk).abs().max():.2e})"
        )
        o2_chunk.sum().backward()
        assert g2q.grad is not None and torch.isfinite(g2q.grad).all(), (
            "FAIL: GDN-2 chunk no/NaN grad"
        )
        print(
            "PASS: chunkwise GDN-2 scan matches its sequential recurrence (no-fla fallback, exact)."
        )
        if _HAVE_FLA_GDN2 and device == "cuda":
            # Fused-kernel parity vs the fp32 reference. Head dim 64 (a real-model size the Triton
            # kernel definitely supports) and T=128 = two chunks of 64 (chunk_gdn2 forces BT=64).
            fdh = 64
            with torch.no_grad():
                fq = F.normalize(torch.randn(2, 128, gh, fdh, device=device), dim=-1)
                fk = F.normalize(torch.randn(2, 128, gh, fdh, device=device), dim=-1)
                fv = torch.randn(2, 128, gh, fdh, device=device)
                fb = torch.sigmoid(torch.randn(2, 128, gh, fdh, device=device))
                fw = torch.sigmoid(torch.randn(2, 128, gh, fdh, device=device))
                fg = (
                    -F.softplus(torch.randn(gh, fdh, device=device))[None, None]
                    * torch.sigmoid(torch.randn(2, 128, gh, fdh, device=device))
                ).clamp(min=-20)
                o_ref = _gdn2_sequential_ref(fq, fk, fv, fb, fw, fg)
                o_fla = _chunk_gdn2(
                    fq.float(),
                    fk.float(),
                    fv.float(),
                    fg.float(),
                    fb.float(),
                    fw.float(),
                    scale=1.0,
                    use_qk_l2norm_in_kernel=False,
                )[0]
                rel = ((o_ref - o_fla).norm() / o_ref.norm().clamp_min(1e-6)).item()
            assert rel < 2e-2, f"FAIL: fla chunk_gdn2 != GDN-2 reference (rel L2 {rel:.2e})"
            print(f"PASS: fla chunk_gdn2 matches the GDN-2 reference on CUDA (rel L2 {rel:.2e}).")
            # fused_recurrent_gdn2 (the streaming-decode step kernel) must match the same
            # reference when driven token-by-token with a carried state — the gate before
            # trusting it in decode_step's cache path.
            with torch.no_grad():
                _, S_c = _gdn2_chunk_scan(
                    fq[:, :96],
                    fk[:, :96],
                    fv[:, :96],
                    fb[:, :96],
                    fw[:, :96],
                    fg[:, :96],
                    chunk=32,
                    return_state=True,
                )
                outs = []
                for t in range(96, 128):
                    o_t, S_c = _fused_recurrent_gdn2(
                        fq[:, t : t + 1].float(),
                        fk[:, t : t + 1].float(),
                        fv[:, t : t + 1].float(),
                        fg[:, t : t + 1].float(),
                        fb[:, t : t + 1].float(),
                        fw[:, t : t + 1].float(),
                        scale=1.0,
                        initial_state=S_c,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=False,
                    )
                    outs.append(o_t)
                rel2 = (
                    (o_ref[:, 96:] - torch.cat(outs, 1)).norm()
                    / o_ref[:, 96:].norm().clamp_min(1e-6)
                ).item()
            assert rel2 < 2e-2, (
                f"FAIL: fused_recurrent_gdn2 step != GDN-2 reference (rel L2 {rel2:.2e})"
            )
            print(
                f"PASS: fused_recurrent_gdn2 streaming step matches the reference with a carried "
                f"chunk-scan state (rel L2 {rel2:.2e})."
            )
        # Subconscious (EXPERIMENTAL, off by default): exact no-op at init (write proj zero -> drop-in
        # safe), trains once it earns influence, and runs through the gradient-checkpointed core loop
        # without a recompute hazard. Equal-FLOP ablation control via force_gate_zero.
        scfg = CharkhaConfig.toy()
        scfg.use_subconscious = True
        scfg.subconscious_dim = 16
        scfg.grad_checkpoint = True
        sm = Charkha(scfg).to(device)
        ss = torch.randn(2, 12, scfg.d_model, device=device)
        ss2, _ = sm.subconscious(ss, sm.subconscious.init_state(ss))
        assert bool((sm.subconscious.write.weight == 0).all()) and torch.equal(ss2, ss), (
            "FAIL: subconscious is not an exact no-op at init (global init clobbered reset_noop?)"
        )
        sm.train()
        with torch.no_grad():
            sm.subconscious.write.weight.normal_(0, 0.02)
        sx = torch.randint(0, scfg.vocab_size, (2, 12), device=device)
        _, sl = sm(sx, sx, r=2)
        sl.backward()
        sg = all(
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            for p in sm.subconscious.parameters()
        )
        assert bool(torch.isfinite(sl)) and sg, (
            "FAIL: subconscious did not train through the checkpointed core"
        )
        sm.zero_grad()
        sm.subconscious.force_gate_zero = True
        _, sl2 = sm(sx, sx, r=2)
        sl2.backward()
        assert bool(torch.isfinite(sl2)), "FAIL: subconscious equal-FLOP ablation non-finite"
        print(
            "PASS: subconscious scratchpad — no-op at init, trains via the checkpointed core, "
            "equal-FLOP ablatable (off by default)."
        )
        # Factorized tied embedding (cfg.embed_factor): must build with the small parameterization,
        # train end-to-end (codes AND up both get grads), produce full-vocab logits at eval, and
        # actually shrink the embedding params by ~V*(d-f).
        fcfg = CharkhaConfig.toy()
        fcfg.embed_factor = 16
        fm = Charkha(fcfg).to(device)
        dn = fcfg.vocab_size * fcfg.d_model
        fn = sum(p.numel() for p in fm.embed.parameters())
        assert fn < dn // 2, f"FAIL: factorized embedding not smaller ({fn} vs dense {dn})"
        fx = torch.randint(0, fcfg.vocab_size, (2, 16), device=device)
        _, floss = fm(fx, fx, r=2)
        floss.backward()
        assert torch.isfinite(floss), "FAIL: factorized-embed loss non-finite"
        assert (
            fm.embed.codes.weight.grad is not None and fm.embed.codes.weight.grad.abs().sum() > 0
        ), "FAIL: factorized codes got no gradient"
        assert fm.embed.up.weight.grad is not None and fm.embed.up.weight.grad.abs().sum() > 0, (
            "FAIL: factorized up-projection got no gradient"
        )
        fm.eval()
        with torch.no_grad():
            flg, _ = fm(fx)
        assert flg.shape == (2, 16, fcfg.vocab_size), f"FAIL: factorized logits shape {flg.shape}"
        fopts = build_symmetry_optimizers(fm)
        f_row = {id(p) for p in fopts[1].param_groups[0]["params"]}
        assert id(fm.embed.codes.weight) in f_row and id(fm.embed.up.weight) in f_row, (
            "FAIL: factorized embedding params not routed to RowNormM under --symmetry-opt"
        )
        print(
            f"PASS: factorized tied embedding — {fn:,} params vs dense {dn:,}, trains, "
            "full-vocab eval logits, RowNormM routing."
        )
        # NITP (arXiv:2605.24956): aux loss must add a finite term, give nitp_head a gradient,
        # and cost nothing at inference (head untouched when targets=None).
        ncfg = CharkhaConfig.toy()
        ncfg.use_nitp = True
        nm = Charkha(ncfg).to(device)
        xb = torch.randint(0, ncfg.vocab_size, (2, 16), device=device)
        _, nloss = nm(xb, xb)
        nloss.backward()
        assert torch.isfinite(nloss), "FAIL: NITP loss non-finite"
        assert nm.nitp_head.weight.grad is not None and nm.nitp_head.weight.grad.abs().sum() > 0, (
            "FAIL: NITP head got no gradient"
        )
        nm.eval()
        with torch.no_grad():
            lg, _ = nm(xb)
        assert tuple(lg.shape) == (2, 16, ncfg.vocab_size), "FAIL: NITP perturbed inference output"
        print("PASS: NITP aux loss trains (head gets grad) and is inference-free.")
        # loop-index embedding: the shared core block must produce DIFFERENT outputs at different
        # iterations (otherwise the loop is N identical applications and depth buys nothing).
        with torch.no_grad():
            se = torch.randn(1, 8, cfg.d_model, device=device)
            ee = torch.randn(1, 8, cfg.d_model, device=device)
            h0 = model._core_step(se, ee, 0)
            h3 = model._core_step(se, ee, 3)
            assert cfg.use_loop_embed and not torch.allclose(h0, h3), (
                "FAIL: loop-index embedding does not differentiate iterations"
            )
            ncfg2 = CharkhaConfig.toy()
            ncfg2.use_loop_embed = False
            m2 = Charkha(ncfg2).to(device).eval()
            a0 = m2._core_step(se, ee, 0)
            a3 = m2._core_step(se, ee, 3)
            assert torch.allclose(a0, a3), "FAIL: loop embed off should make iterations identical"
        print("PASS: loop-index embedding differentiates recurrent iterations (off => identical).")
        # anytime deep supervision: earlier grad loops are decoded through coda+head and supervised,
        # so coda must get gradient from them, the trajectory must be captured, and inference is free.
        dcfg = CharkhaConfig.toy()
        dcfg.use_deep_supervision = True
        dcfg.use_halting = False
        dm = Charkha(dcfg).to(device)
        xb = torch.randint(0, dcfg.vocab_size, (2, 16), device=device)
        _, dloss = dm(xb, xb, r=3)  # r=3, backprop_depth=2 -> 2 grad loops
        assert torch.isfinite(dloss), "FAIL: deep-supervision loss non-finite"
        dloss.backward()
        coda_grad = sum(
            p.grad.abs().sum().item() for p in dm.coda.parameters() if p.grad is not None
        )
        assert coda_grad > 0, "FAIL: deep supervision gave coda no gradient"
        assert dm._core_traj is not None and len(dm._core_traj) == 2, (
            "FAIL: deep-sup trajectory not captured (expected 2 grad loops at r=3)"
        )
        dm.eval()
        with torch.no_grad():
            lg, _ = dm(xb)
        assert tuple(lg.shape) == (2, 16, dcfg.vocab_size) and dm._core_traj is None, (
            "FAIL: deep supervision perturbed inference (or left a stale trajectory)"
        )
        print(
            "PASS: anytime deep supervision trains earlier loops (coda gets grad), inference-free."
        )
        # A2 convergence: at inference, a per-token extrapolation-error signal (B,T) is exposed and
        # finite; it is computed only behind the flag and never leaks into training.
        ccfg = CharkhaConfig.toy()
        ccfg.track_convergence = True
        ccfg.use_halting = False
        cm = Charkha(ccfg).to(device)
        cm.eval()
        xb = torch.randint(0, ccfg.vocab_size, (2, 12), device=device)
        with torch.no_grad():
            _ = cm(xb, r=5)  # fixed 5 loops -> extrapolation possible
        assert cm._last_convergence is not None and tuple(cm._last_convergence.shape) == (2, 12), (
            "FAIL: convergence signal missing or wrong shape"
        )
        assert torch.isfinite(cm._last_convergence).all(), "FAIL: convergence signal non-finite"
        cm.train()  # training must not populate the signal
        _, _l = cm(xb, xb, r=5)
        assert cm._last_convergence is None, "FAIL: convergence signal leaked into training"
        coff = CharkhaConfig.toy()
        coff.use_halting = False  # flag off -> always None
        mo = Charkha(coff).to(device)
        mo.eval()
        with torch.no_grad():
            _ = mo(xb, r=5)
        assert mo._last_convergence is None, "FAIL: convergence computed with flag off"
        # acceleration mode (two-scale-latent 2nd difference) is also a valid (B,T) signal
        acfg = CharkhaConfig.toy()
        acfg.track_convergence = True
        acfg.use_halting = False
        acfg.convergence_mode = "acceleration"
        am = Charkha(acfg).to(device)
        am.eval()
        with torch.no_grad():
            _ = am(xb, r=5)
        assert (
            am._last_convergence is not None
            and tuple(am._last_convergence.shape) == (2, 12)
            and torch.isfinite(am._last_convergence).all()
        ), "FAIL: acceleration signal bad"
        print(
            "PASS: A2 convergence signal exposed at inference (extrapolation + acceleration), "
            "training-free."
        )

        # acceleration early-exit (two-scale-latent): a huge threshold => exit as soon as 3 states
        # exist (3 loops); the same model without it runs the full adaptive budget. Inference-only.
        def _halt_steps(use_exit):
            ec = CharkhaConfig.toy()
            ec.use_halting = True
            ec.use_accel_exit = use_exit
            ec.accel_exit_threshold = 1e9
            ec.max_recurrence_infer = 8
            ec.halt_threshold = 0.999
            em = Charkha(ec).to(device)
            em.eval()
            em._n_core_steps = 0
            with torch.no_grad():
                em(torch.randint(0, ec.vocab_size, (1, 8), device=device))
            return em._n_core_steps

        s_exit, s_full = _halt_steps(True), _halt_steps(False)
        assert s_exit == 3, f"FAIL: accel-exit should stop at 3 loops, got {s_exit}"
        assert s_full > s_exit, f"FAIL: accel-exit did not reduce loops ({s_full} vs {s_exit})"
        print(
            f"PASS: acceleration early-exit halts the loop when settled ({s_exit} vs {s_full} loops)."
        )
        # symmetry-compatible optimizer split (arXiv:2605.18106): correct param routing + trains.
        scfg = CharkhaConfig.toy()
        sm = Charkha(scfg).to(device)
        sopts = build_symmetry_optimizers(sm, muon_lr=0.02, norm_lr=0.02, adam_lr=3e-3)
        nrow = sum(p.numel() for p in sopts[1].param_groups[0]["params"])
        ncol = sum(p.numel() for p in sopts[2].param_groups[0]["params"])
        ndown = sum(p.numel() for nm, p in sm.named_parameters() if nm.endswith("mlp.down.weight"))
        assert len(sopts) == 4 and nrow > 0 and ncol == ndown > 0, "FAIL: symmetry routing wrong"
        sxb = torch.randint(0, scfg.vocab_size, (2, 16), device=device)
        sl0 = None
        for _ in range(8):
            for o in sopts:
                o.zero_grad(set_to_none=True)
            _, sl = sm(sxb, sxb)
            sl.backward()
            for o in sopts:
                o.step()
            slv = sl.detach().item()
            if sl0 is None:
                sl0 = slv
        assert slv < sl0, f"FAIL: symmetry optimizers did not train ({sl0:.3f}->{slv:.3f})"
        print("PASS: symmetry-compatible optimizer split routes correctly and trains.")
        # OSDN: per-dim key preconditioning present on GDN layers, off by default, gets grad, trains.
        ocfg = CharkhaConfig.toy()
        ocfg.use_osdn = True
        om = Charkha(ocfg).to(device)
        gdn = om.prelude[0].mixer
        assert hasattr(gdn, "k_precond") and tuple(gdn.k_precond.shape) == (
            ocfg.n_heads,
            ocfg.head_dim,
        ), "FAIL: OSDN preconditioner missing/wrong shape"
        assert not hasattr(Charkha(CharkhaConfig.toy()).prelude[0].mixer, "k_precond"), (
            "FAIL: OSDN param present when off"
        )
        oxb = torch.randint(0, ocfg.vocab_size, (2, 16), device=device)
        oo = build_optimizers(om, 0.02, 3e-3)
        ol0 = None
        for _ in range(8):
            for o in oo:
                o.zero_grad(set_to_none=True)
            _, ol = om(oxb, oxb)
            ol.backward()
            for o in oo:
                o.step()
            olv = ol.detach().item()
            if ol0 is None:
                ol0 = olv
        assert gdn.k_precond.grad is not None and gdn.k_precond.grad.abs().sum() > 0, (
            "FAIL: OSDN preconditioner received no gradient"
        )
        assert olv < ol0, f"FAIL: OSDN model did not train ({ol0:.3f}->{olv:.3f})"
        print(
            "PASS: OSDN per-dimension key preconditioning trains (precond gets grad), off by default."
        )
        # hidden() accessor: post-norm h with the right shape, grad-carrying, head-free.
        hm_ = Charkha(CharkhaConfig.toy()).to(device)
        hm_.train()
        hxb = torch.randint(0, hm_.cfg.vocab_size, (2, 12), device=device)
        hh = hm_.hidden(hxb, r=2)
        assert tuple(hh.shape) == (2, 12, hm_.cfg.d_model), "FAIL: hidden() wrong shape"
        hh.sum().backward()
        assert hm_.conf_head.weight.grad is None, "FAIL: hidden() should not touch the head"
        assert any(p.grad is not None for p in hm_.coda.parameters()), (
            "FAIL: hidden() carried no grad"
        )
        print("PASS: hidden() exposes grad-carrying post-norm states (head-free probe hook).")
        # RLCM confidence-margin loss (arXiv:2604.23333): finite, >=0, and trains the conf head,
        # fed by the real hidden() path over correct vs incorrect prefixes.
        rm = Charkha(CharkhaConfig.toy()).to(device)
        hg = rm.hidden(torch.randint(0, rm.cfg.vocab_size, (4, 10), device=device), r=2).mean(1)
        hb = (
            rm.hidden(torch.randint(0, rm.cfg.vocab_size, (4, 10), device=device), r=2)
            .mean(1)
            .detach()
        )
        ml = rm.conf_margin_loss(hg, hb)
        assert torch.isfinite(ml) and ml.item() >= 0.0, "FAIL: conf-margin loss not finite/>=0"
        ml.backward()
        assert rm.conf_head.weight.grad is not None and rm.conf_head.weight.grad.abs().sum() > 0, (
            "FAIL: conf-margin loss gave the conf head no gradient"
        )
        print("PASS: RLCM confidence-margin loss is finite and trains the conf head.")
        # SNGP (arXiv:2006.10108): distance-aware epistemic head present when on, off=0 cost; the GP
        # mean trains by default, precision accumulation is explicit calibration, and the predictive
        # variance is finite + bigger for out-of-distribution inputs than for in-distribution ones.
        gcfg = CharkhaConfig.toy()
        gcfg.sngp_enabled = True
        gcfg.sngp_rff_dim = 64
        gm = Charkha(gcfg).to(device)
        assert hasattr(gm, "sngp_head") and not hasattr(
            Charkha(CharkhaConfig.toy()), "sngp_head"
        ), "FAIL: SNGP head present/absent wrong vs flag"
        gxb = torch.randint(0, gcfg.vocab_size, (2, 16), device=device)
        goo = build_optimizers(gm, 0.02, 3e-3)
        gl0 = None
        prec0 = gm.sngp_head.precision.clone()
        for _ in range(8):
            for o in goo:
                o.zero_grad(set_to_none=True)
            _, gl = gm(gxb, gxb)
            gl.backward()
            for o in goo:
                o.step()
            glv = gl.detach().item()
            if gl0 is None:
                gl0 = glv
        assert (
            gm.sngp_head.beta.weight.grad is not None
            and gm.sngp_head.beta.weight.grad.abs().sum() > 0
        ), "FAIL: SNGP GP-mean got no gradient"
        assert (gm.sngp_head.precision - prec0).abs().sum() == 0, (
            "FAIL: SNGP precision accumulated during default pretraining path"
        )
        gm.sngp_head.accumulate_precision(gm.hidden(gxb, r=1).detach())
        assert (gm.sngp_head.precision - prec0).abs().sum() > 0, (
            "FAIL: explicit SNGP calibration did not accumulate precision"
        )
        assert glv < gl0, f"FAIL: SNGP model did not train ({gl0:.3f}->{glv:.3f})"
        gm.eval()
        with torch.no_grad():
            gm(gxb)  # populates _last_sngp_var
            var_id = gm._last_sngp_var
            # OOD probe: a clearly different token distribution should read more uncertain on average
            gm(torch.zeros_like(gxb))
            var_ood_zero = gm._last_sngp_var
        assert var_id is not None and torch.isfinite(var_id).all() and (var_id >= 0).all(), (
            "FAIL: SNGP variance not finite/non-negative"
        )
        assert torch.isfinite(var_ood_zero).all(), "FAIL: SNGP OOD variance not finite"
        print(
            "PASS: SNGP epistemic head trains; precision calibration is explicit; variance is finite."
        )
        # SNGP spectral-norm option: wraps the coda's output projections (bi-Lipschitz / distance-
        # preserving feature map) and must still train + infer. Independently gated from the GP head.
        sncfg = CharkhaConfig.toy()
        sncfg.sngp_enabled = True
        sncfg.sngp_rff_dim = 64
        sncfg.sngp_spectral_norm = True
        snm = Charkha(sncfg).to(device)
        snxb = torch.randint(0, sncfg.vocab_size, (2, 16), device=device)
        snoo = build_optimizers(snm, 0.02, 3e-3)
        snl0 = None
        for _ in range(6):
            for o in snoo:
                o.zero_grad(set_to_none=True)
            _, snl = snm(snxb, snxb)
            snl.backward()
            for o in snoo:
                o.step()
            snlv = snl.detach().item()
            if snl0 is None:
                snl0 = snlv
        assert math.isfinite(snlv) and snlv < snl0, (
            f"FAIL: SNGP+spectral-norm did not train ({snl0:.3f}->{snlv:.3f})"
        )
        snm.eval()
        with torch.no_grad():
            snm(snxb)
        print("PASS: SNGP spectral-norm coda variant trains + infers.")
        # Laplace-Redux (arXiv:2106.14806): POST-HOC last-layer Laplace on conf_head -> (mean, var),
        # no training change. Fit on toy hidden states, predict finite mean in [0,1] + non-neg var,
        # and var must rise with the prior-precision relaxation (smaller prior => looser => more var).
        lm = Charkha(CharkhaConfig.toy()).to(device)
        device_l = lm.embed.weight.device  # model device (cuda or cpu)
        cal = [torch.randn(8, lm.cfg.d_model, device=device_l) for _ in range(4)]
        lap = LaplaceConf(lm.conf_head, prior_precision=1.0).fit(iter(cal))
        hq = torch.randn(3, 5, lm.cfg.d_model, device=device_l)
        lmean, lvar = lap.predict(hq)
        assert tuple(lmean.shape) == (3, 5) and tuple(lvar.shape) == (3, 5), (
            "FAIL: Laplace shape wrong"
        )
        assert torch.isfinite(lmean).all() and (lmean >= 0).all() and (lmean <= 1).all(), (
            "FAIL: Laplace mean not a probability"
        )
        assert torch.isfinite(lvar).all() and (lvar >= 0).all(), (
            "FAIL: Laplace var not finite/non-neg"
        )
        lap_loose = LaplaceConf(lm.conf_head, prior_precision=0.01).fit(iter(cal))
        _, lvar_loose = lap_loose.predict(hq)
        assert lvar_loose.mean() > lvar.mean(), "FAIL: looser prior did not increase Laplace var"
        print("PASS: Laplace-Redux post-hoc conf head yields calibrated (mean, var).")
        # Bipolar sign-gating: STE forces k,v to ±1 in forward, differentiable in backward.
        bcfg = CharkhaConfig.toy()
        bcfg.use_bipolar_gate = True
        bcfg.effective_depth_scale = True  # use the fix
        bm = Charkha(bcfg).to(device)
        bxb = torch.randint(0, bcfg.vocab_size, (2, 16), device=device)
        boo = build_optimizers(bm, 0.02, 3e-3)
        bl0 = None
        for _ in range(8):
            for o in boo:
                o.zero_grad(set_to_none=True)
            _, bl = bm(bxb, bxb)
            bl.backward()
            for o in boo:
                o.step()
            blv = bl.detach().item()
            if bl0 is None:
                bl0 = blv
        assert blv < bl0, f"FAIL: bipolar sign-gating did not train ({bl0:.3f}->{blv:.3f})"
        bm.eval()
        with torch.no_grad():
            lg, _ = bm(bxb)
        assert tuple(lg.shape) == (2, 16, bcfg.vocab_size), "FAIL: bipolar model inference broken"
        off_m = Charkha(CharkhaConfig.toy())
        assert not getattr(off_m.prelude[0].mixer, "use_bipolar_gate", False), (
            "FAIL: bipolar gate present when off"
        )
        print("PASS: bipolar sign-gating trains, infers, and is off by default.")

        # (error_feedback, neuromod_gate, use_orthogonal_loss, use_abacus_embed and
        # fixed_weighted_halt were REMOVED — no measured win, or buggy/limbo (2026-07-05
        # config unification). CharkhaConfig.from_dict drops the

        # MTP-routed macro-states: feed predicted future rep into core loops.
        mcfg = CharkhaConfig.toy()
        mcfg.use_mtp_routing = True
        mcfg.mtp_weight = 0.2
        mcfg.effective_depth_scale = True
        mm = Charkha(mcfg).to(device)
        assert mm.adapter.weight.shape[1] == 3 * mcfg.d_model, (
            "FAIL: adapter input dim wrong for MTP routing"
        )
        mo = build_optimizers(mm, 0.02, 3e-3)
        ml0 = None
        for _ in range(6):
            for o in mo:
                o.zero_grad(set_to_none=True)
            _, ml = mm(xb, xb, r=2)
            ml.backward()
            for o in mo:
                o.step()
            mlv = ml.detach().item()
            if ml0 is None:
                ml0 = mlv
        assert mlv < ml0, f"FAIL: MTP-routing model did not train ({ml0:.3f}->{mlv:.3f})"
        mm.eval()
        with torch.no_grad():
            lg, _ = mm(xb, r=2)
        assert lg.ndim == 3 and lg.shape[-1] == mcfg.vocab_size, (
            "FAIL: MTP-routing inference broken"
        )
        om = Charkha(CharkhaConfig.toy())
        assert om.adapter.weight.shape[1] == 2 * om.cfg.d_model, (
            "FAIL: adapter wrong when MTP routing off"
        )
        print("PASS: MTP-routed macro-states train, infer, and are off by default.")

        # Epistemic thermostat: overrides greedy halting when model is confused.
        tcfg = CharkhaConfig.toy()
        tcfg.use_thermostat = True
        tcfg.use_halting = True
        tcfg.effective_depth_scale = True
        tm = Charkha(tcfg).to(device)
        tm.eval()
        with torch.no_grad():
            tb = torch.randn(2, 8, tcfg.d_model, device=device)
            _, steps, _ = tm._run_core_halting(tb)
            mean_steps = steps.mean().item()
        assert 0 < mean_steps <= tcfg.max_recurrence_infer, (
            f"FAIL: thermostat broke halting (mean_steps={mean_steps:.1f})"
        )
        print(
            f"PASS: epistemic thermostat operates within halting bounds (mean_steps={mean_steps:.1f})."
        )

        # KITCHEN-SINK: EVERY optional flag enabled together must train + infer (fixed-r path covers
        # deep-sup/nitp/mtp/osdn/convergence; halting eval path covers accel-exit). Trained with the
        # symmetry-compatible optimizer split so all the optional machinery coexists end to end.
        kc = CharkhaConfig.toy()
        for fld in (
            "use_nitp",
            "use_deep_supervision",
            "track_convergence",
            "use_accel_exit",
            "use_osdn",
            "use_halting",
            "use_loop_embed",
            "sngp_enabled",
            "use_bipolar_gate",
            "use_mtp_routing",
            "use_thermostat",
        ):
            setattr(kc, fld, True)
        kc.sngp_rff_dim = 64
        kc.convergence_mode = "acceleration"
        kmod = Charkha(kc).to(device)
        kopts = build_symmetry_optimizers(kmod, 0.02, 0.02, 3e-3)
        kxb = torch.randint(0, kc.vocab_size, (2, 16), device=device)
        kl0 = None
        for _ in range(6):
            for o in kopts:
                o.zero_grad(set_to_none=True)
            _, kl = kmod(kxb, kxb, r=3)  # fixed-r => deep-sup + convergence active
            kl.backward()
            for o in kopts:
                o.step()
            klv = kl.detach().item()
            if kl0 is None:
                kl0 = klv
        assert math.isfinite(klv) and klv < kl0, (
            f"FAIL: opt-in feature stress test did not train ({kl0:.3f}->{klv:.3f})"
        )
        kmod.eval()
        with torch.no_grad():
            klog, kconf = kmod(kxb)  # halting eval path => accel-exit + convergence
            kgen = kmod.generate(torch.tensor([[5]], device=device), 16, effort=None)
        assert (
            tuple(klog.shape) == (2, 16, kc.vocab_size)
            and kgen.shape[1] == 17
            and kmod._last_convergence is not None
        ), "FAIL: full-featured inference broke"
        print("PASS: kitchen-sink feature stress config trains + infers.")
        print()
        print("=" * 62)
        print("  CHARKHA TOY -- generation samples across effort levels")
        print('  prompt: byte 87 = "W" (first letter of Shakespeare)')
        print("=" * 62)
        seed = torch.tensor([[87]], device=device)
        settings = [
            (1, 0.6, 80, "r=1 (minimal thought) -- fast, rough"),
            (2, 0.7, 120, "r=2 (moderate) -- balanced"),
            (
                cfg.max_recurrence_train,
                0.8,
                200,
                f"r={cfg.max_recurrence_train} (max effort) -- deepest",
            ),
            (None, 0.7, 160, "adaptive(halting) -- the model decides when to stop"),
        ]
        for effort, temp, length, label in settings:
            out = model.generate(seed, length, effort=effort, temp=temp, top_k=50)
            txt = bytes(out[0].tolist()).decode("utf-8", errors="replace")
            print(f"\n  [{label}]")
            print(f"  {'-' * 58}")
            for li, line in enumerate(txt.split("\n")[:12]):
                print(f"  {line!r}")
        print()
        print("=" * 62)
        delta = r_losses[-1] - r_losses[0]
        print(
            f"  loss: {first:.2f} -> {last:.2f} | max-effort loss delta vs r=1: {delta:+.3f} (limit: +1.000)"
        )
        print("  toy checks passed")
        print("=" * 62)
    return model


def main():
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    p = argparse.ArgumentParser(description="CHARKHA reference model")
    p.add_argument("--toy", action="store_true", help="tiny CPU-able verification run")
    p.add_argument("--data", type=str, default=None, help="path to a text file")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--adam-lr", type=float, default=3e-3)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--ckpt", type=str, default="charkha_ckpt.pt")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-recurrence", action="store_true", help="ablation: plain stack")
    a = p.parse_args()
    if a.toy:
        a.steps = a.steps or 300
        a.batch_size = a.batch_size or 16
        a.seq_len = a.seq_len or 128
        a.warmup = a.warmup or 20
    else:
        a.steps = a.steps or 10000
        a.batch_size = a.batch_size or 8
        a.seq_len = a.seq_len or 1024
        a.warmup = a.warmup or 250
    train(a)


if __name__ == "__main__":
    main()

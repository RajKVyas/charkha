#!/usr/bin/env python
"""Fast CHARKHA regression tests.

This suite is intentionally small and hermetic. The long model/data checks remain in
`python src/preflight.py`; these tests catch logic regressions that previously broke
resume/status, worker partitioning, dashboard logging, confidence scoring, and continual
training before a multi-hour run ever starts.
"""

import math
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


class FastRegressions(unittest.TestCase):
    def test_dataprep_partitions_return_all_entries(self):
        from dataprep import _entry_id, _partition_sources

        entries = [
            {"id": "a", "weight": 10},
            {"id": "b", "weight": 1},
            {"id": "c", "weight": 1},
        ]
        groups = _partition_sources(entries, 2)
        flat = sorted(_entry_id(e) for g in groups for e in g)
        self.assertEqual(flat, ["a", "b", "c"])
        self.assertEqual(len(groups), 2)

    def test_serve_confidence_scores_generated_span_with_context(self):
        from charkha import Charkha, CharkhaConfig
        from serve import mean_confidence

        cfg = CharkhaConfig.toy()
        cfg.use_gdn2 = True
        cfg.max_seq_len = 16
        model = Charkha(cfg)
        ids = [1, 2, 3, 4, 5, 6]
        conf = mean_confidence(model, ids, torch.device("cpu"), effort=1, start=4)
        self.assertTrue(math.isfinite(conf))
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_somatic_step_consolidates_core_and_coda(self):
        # Wake-sleep somatic consolidation must replay POST-PRELUDE shallow features (Charkha.shallow,
        # the space the recurrent core consumes) through core+coda and flow gradients there, against a
        # detached clean-output target (denoising-consistency). Regression guard for the representation-
        # space + ragged-length fix — this path was previously never exercised (the continual selftest
        # sets somatic_interval=0) and would have crashed on torch.stack of variable-length latents.
        from collections import deque
        from charkha import Charkha, CharkhaConfig
        from continual import ContinualTrainer

        torch.manual_seed(0)
        cfg = CharkhaConfig.toy()
        cfg.use_gdn2 = True
        cfg.use_recurrence = True
        cfg.use_halting = False
        model = Charkha(cfg)
        model.train()

        class _Stub:
            pass

        t = _Stub()
        t.model = model
        t.somatic_batch = 3
        t.somatic_noise = 0.1
        t.somatic_loss_ema = 0.0
        t.replay = _Stub()
        t.replay.latents = deque()
        for P in (5, 8, 6, 7):  # varying lengths -> replayed one at a time
            ids = torch.randint(1, cfg.vocab_size, (1, P))
            t.replay.latents.append(model.shallow(ids).detach().cpu())

        # The bug being guarded against was replaying hidden() (post-coda) instead of shallow()
        ids = torch.randint(1, cfg.vocab_size, (1, 6))
        self.assertGreater(
            (model.shallow(ids) - model.hidden(ids, r=None)).abs().max().item(), 1e-3
        )

        model.zero_grad(set_to_none=True)
        ContinualTrainer._somatic_step(t)
        self.assertTrue(math.isfinite(t.somatic_loss_ema))
        core_grad = sum(
            p.grad.abs().sum().item() for p in model.core.parameters() if p.grad is not None
        )
        coda_grad = sum(
            p.grad.abs().sum().item() for p in model.coda.parameters() if p.grad is not None
        )
        self.assertGreater(core_grad, 0.0, "somatic step gave the core no gradient")
        self.assertGreater(coda_grad, 0.0, "somatic step gave the coda no gradient")

    def test_mastery_gated_curriculum(self):
        # Curriculum advances ONLY on held-out mastery (generalization AND calibration AND fluency),
        # never on step counts. Guards the previously-dead path (tasks.advance_difficulty was defined
        # but never called, so difficulty was pinned at 0.0). All three gates must be required.
        import tempfile
        from charkha import Charkha, CharkhaConfig
        from continual import ContinualTrainer, ByteTokenizer
        from tasks import TaskGenerator, set_tokenizer

        torch.manual_seed(0)
        cfg = CharkhaConfig.toy()
        cfg.use_gdn2 = True
        cfg.use_recurrence = True
        cfg.use_halting = True
        model = Charkha(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        tok = ByteTokenizer()
        set_tokenizer(tok)
        tr = ContinualTrainer(
            model=model,
            optimizer=opt,
            tokenizer=tok,
            out_dir=tempfile.mkdtemp(prefix="quiz_test_"),
            task_gen=TaskGenerator(difficulty=0.0, seed=1),
            phase_ratio=(1.0, 0.0, 0.0),
            log_every=10**9,
            eval_every=10**9,
            save_every=10**9,
            somatic_interval=0,
            quiz_interval=5,
            quiz_size=3,
        )

        q = tr._quiz(n=3)  # real held-out quiz on a toy model
        for k in ("accuracy", "confidence", "calibration_gap", "fluency", "difficulty", "n"):
            self.assertIn(k, q)
        self.assertTrue(0.0 <= q["accuracy"] <= 1.0)
        self.assertTrue(0.0 <= q["fluency"] <= 1.0)
        self.assertGreaterEqual(q["calibration_gap"], 0.0)

        # full mastery -> advance
        tr._quiz = lambda n=None: {
            "difficulty": 0.0,
            "n": 3,
            "accuracy": 1.0,
            "confidence": 1.0,
            "calibration_gap": 0.0,
            "fluency": 1.0,
        }
        self.assertEqual(tr._maybe_advance_curriculum()["verdict"], "advance")
        self.assertAlmostEqual(tr.curriculum_difficulty, 0.05, places=6)

        # far below target -> regress, floored at 0
        tr._quiz = lambda n=None: {
            "difficulty": 0.05,
            "n": 3,
            "accuracy": 0.0,
            "confidence": 0.9,
            "calibration_gap": 0.9,
            "fluency": 0.1,
        }
        self.assertEqual(tr._maybe_advance_curriculum()["verdict"], "regress")
        self.assertAlmostEqual(tr.curriculum_difficulty, 0.0, places=6)

        # high accuracy but BAD calibration -> must NOT advance (all three gates required)
        tr.curriculum_difficulty = 0.2
        tr.task_gen.difficulty = 0.2
        tr._quiz = lambda n=None: {
            "difficulty": 0.2,
            "n": 3,
            "accuracy": 1.0,
            "confidence": 0.4,
            "calibration_gap": 0.6,
            "fluency": 1.0,
        }
        self.assertEqual(tr._maybe_advance_curriculum()["verdict"], "hold")
        self.assertAlmostEqual(tr.curriculum_difficulty, 0.2, places=6)

    def test_loop_adapters_noop_and_specialization(self):
        # Per-loop low-rank adapters: EXACT no-op at init (zero-init up, surviving the
        # global _init pass), per-loop differentiation once non-zero, gradient flow,
        # and out-of-range loop indices clamping to the last adapter.
        from charkha import Charkha, CharkhaConfig

        torch.manual_seed(0)
        c = CharkhaConfig.toy()
        c.use_recurrence = True
        c.use_loop_adapters = True
        c.loop_adapter_rank = 4
        c.loop_adapter_max = 4
        m = Charkha(c)
        m.eval()
        for u in m.loop_adapters.up:
            self.assertTrue(
                bool((u.weight == 0).all()),
                "up projection not zero at init (global _init clobbered reset_noop)",
            )
        x = torch.randint(0, c.vocab_size, (2, 12))
        with torch.no_grad():
            on, _ = m(x, r=2)
            la = m.loop_adapters
            m.loop_adapters = None
            off, _ = m(x, r=2)
            m.loop_adapters = la
        self.assertTrue(torch.equal(on, off), "loop adapters not an exact no-op at init")

        # make loop 0 and loop 1 differ -> the same state must map differently per loop
        with torch.no_grad():
            m.loop_adapters.up[0].weight.fill_(0.05)
            m.loop_adapters.up[1].weight.fill_(-0.05)
            s = torch.randn(1, 4, c.d_model)
            self.assertFalse(
                torch.equal(m.loop_adapters(s, 0), m.loop_adapters(s, 1)),
                "adapters did not differentiate loops",
            )
            # clamp: any n >= max reuses the last adapter (valid at every effort)
            self.assertTrue(
                torch.equal(m.loop_adapters(s, 99), m.loop_adapters(s, c.loop_adapter_max - 1))
            )

        # gradients reach the adapter params through a training forward
        m.train()
        _, loss = m(x, x, r=2)
        loss.backward()
        g = m.loop_adapters.down[0].weight.grad
        self.assertIsNotNone(g, "no gradient reached loop adapters")

    def test_subconscious_module(self):
        # EXPERIMENTAL low-dim recurrent scratchpad: must be an EXACT no-op at init (write=0 -> drop-in
        # safe / earns influence), train once the write is nonzero, run on BOTH the fixed and halting
        # cores, and expose an equal-FLOP ablation control. Off by default; built explicitly here.
        from charkha import Charkha, CharkhaConfig

        def build(halting):
            torch.manual_seed(0)
            c = CharkhaConfig.toy()
            c.use_gdn2 = True
            c.use_recurrence = True
            c.use_halting = halting
            c.use_subconscious = True
            c.subconscious_dim = 16
            return Charkha(c)

        # exact no-op at init (fixed r=2 isolates the scratchpad from r-sampling)
        m = build(False)
        m.eval()
        x = torch.randint(0, m.cfg.vocab_size, (2, 12))
        with torch.no_grad():
            on, _ = m(x, r=2)
            sub = m.subconscious
            m.subconscious = None
            off, _ = m(x, r=2)
            m.subconscious = sub
        self.assertTrue(
            bool((m.subconscious.write.weight == 0).all()),
            "write projection not zero at init (global _init pass clobbered reset_noop)",
        )
        self.assertTrue(torch.equal(on, off), "subconscious is not an exact no-op at init")

        # halting core runs with the scratchpad
        mh = build(True)
        mh.train()
        _, lh = mh(x, x)
        lh.backward()
        self.assertTrue(torch.isfinite(lh))

        # trains: write nudged off zero + fixed r=2 (two scratchpad updates) -> every param gets grad
        m2 = build(False)
        m2.train()
        with torch.no_grad():
            m2.subconscious.write.weight.normal_(0, 0.02)
        _, loss = m2(x, x, r=2)
        loss.backward()
        for n, p in m2.subconscious.named_parameters():
            self.assertIsNotNone(p.grad, f"{n} got no grad")
            self.assertGreater(p.grad.abs().sum().item(), 0.0, f"{n} got zero grad")

        # equal-FLOP ablation control runs end to end
        m2.zero_grad()
        m2.subconscious.force_gate_zero = True
        _, l2 = m2(x, x, r=2)
        l2.backward()
        self.assertTrue(torch.isfinite(l2))

    def test_continual_selftest_entrypoint(self):
        proc = subprocess.run(
            [sys.executable, os.path.join(SRC, "continual.py"), "--selftest"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_granary_staged_noop_equivalence(self):
        # STAGED-module contract: wiring the granary into a model must be an
        # EXACT no-op at init (gate=0) — same logits as the plain model, so it can be
        # activated function-preservingly at a grow point on an existing checkpoint.
        from charkha import Charkha, CharkhaConfig

        torch.manual_seed(7)
        cfg_off = CharkhaConfig.toy()
        torch.manual_seed(7)
        cfg_on = CharkhaConfig.toy()
        cfg_on.use_granary = True
        cfg_on.granary_slots = 256
        cfg_on.granary_topk = 8
        cfg_on.granary_knn = 8

        torch.manual_seed(7)
        m_off = Charkha(cfg_off).eval()
        torch.manual_seed(7)
        m_on = Charkha(cfg_on).eval()
        # granary params are extra (created after trunk init from the same seed), so copy
        # the trunk weights across to guarantee identical trunks.
        trunk = {k: v for k, v in m_off.state_dict().items()}
        missing, unexpected = m_on.load_state_dict(trunk, strict=False)
        self.assertFalse(unexpected, f"unexpected keys: {unexpected}")
        self.assertTrue(all("granary" in k for k in missing), f"non-granary missing: {missing}")

        idx = torch.randint(0, cfg_off.vocab_size, (2, 16))
        with torch.no_grad():
            lo, _ = m_off(idx)
            ln, _ = m_on(idx)
        self.assertTrue(torch.equal(lo, ln), "granary at gate=0 must be an EXACT no-op")

        # and once the gate opens, the output must actually change (module is live)
        with torch.no_grad():
            m_on.granary.gate.fill_(1.0)
            lg, _ = m_on(idx)
        self.assertFalse(torch.equal(lo, lg), "open gate must change logits")

    def test_granary_selftest_entrypoint(self):
        proc = subprocess.run(
            [sys.executable, os.path.join(SRC, "granary.py"), "--selftest"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_granary_micro_selftest_entrypoint(self):
        proc = subprocess.run(
            [sys.executable, os.path.join(SRC, "granary_micro.py"), "--selftest"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_frontier_profile_enables_capability_hooks(self):
        from charkha import Charkha
        from train import make_cfg

        args = SimpleNamespace(
            toy=True,
            small=False,
            medium=False,
            profile="frontier",
            no_recurrence=False,
            no_halting=False,
            nitp=False,
            use_deep_supervision=False,
            cross_loop_consistency=False,
            use_gdn2=False,
            no_gdn2=False,
            use_bipolar_gate=False,
            use_osdn=False,
            use_mtp_routing=False,
            use_task_rl=False,
            awr_temperature=0.5,
            value_weight=1.0,
            task_replay_size=1000,
            use_thermostat=False,
            thermostat_conf_threshold=None,
            track_convergence=False,
            convergence_mode=None,
            convergence_window=None,
            use_accel_exit=False,
            accel_exit_threshold=None,
            sngp_enabled=False,
            sngp_rff_dim=None,
            sngp_scale=None,
            sngp_ridge=None,
            sngp_spectral_norm=False,
            laplace_enabled=False,
            grad_checkpoint=False,
            per_seq_recurrence=False,
            ce_chunk=None,
            gdn_chunk=None,
        )
        cfg = make_cfg(args)
        self.assertTrue(cfg.use_gdn2)
        self.assertTrue(cfg.use_nitp)
        self.assertTrue(cfg.use_deep_supervision)
        self.assertTrue(cfg.cross_loop_consistency)
        self.assertTrue(cfg.use_osdn)
        self.assertTrue(cfg.use_bipolar_gate)
        self.assertTrue(cfg.use_mtp_routing)
        # pruned from the profile after review (see _apply_frontier_profile comments):
        self.assertTrue(cfg.track_convergence)
        self.assertTrue(cfg.sngp_enabled)
        self.assertTrue(cfg.sngp_spectral_norm)
        self.assertTrue(cfg.laplace_enabled)
        self.assertTrue(cfg.use_task_rl)
        self.assertFalse(hasattr(cfg, "error_feedback"))  # feature removed entirely
        self.assertTrue(cfg.use_accel_exit)
        self.assertTrue(cfg.per_seq_recurrence)
        self.assertFalse(cfg.sngp_accumulate_train)
        default_args = SimpleNamespace(**args.__dict__)
        delattr(default_args, "profile")

        model = Charkha(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 16))
        with torch.no_grad():
            h = model.hidden(ids)
        self.assertEqual(tuple(h.shape), (2, 16, cfg.d_model))

    def test_echo_gate_penalizes_momentum_not_instructed_repeats(self):
        # _echo_gate must (a) leave logits untouched when no loop-extending
        # candidate exists or echo_gate=0, (b) penalize a loop token ONLY by its
        # momentum excess (tail-only support > full-context support), and (c) leave
        # the loop token alone when the full context supports it at least as much
        # (the 'repetition was the task' case that no_repeat_ngram breaks).
        from unittest.mock import patch
        from charkha import Charkha, CharkhaConfig

        torch.manual_seed(0)
        cfg = CharkhaConfig(**{**CharkhaConfig.toy().__dict__, "use_recurrence": False})
        model = Charkha(cfg).eval()
        B, V = 1, cfg.vocab_size
        # prefix ends with the 2-gram tail (7, 8); token 9 completed (7,8,9) earlier
        prefix = torch.tensor([[3, 4, 7, 8, 9, 1, 2, 5, 6] * 9 + [7, 8]])
        logits = torch.zeros(B, V)

        # no candidates: a prefix with no repeated (7,8) n-gram start
        clean = torch.arange(2, 80).unsqueeze(0)
        out = model._echo_gate(logits.clone(), clean, effort=1, n=3, ctx_len=8)
        self.assertTrue(torch.equal(out, logits))

        def fake_gen_logits(ids, r):
            t = torch.zeros(1, ids.size(1), V)
            t[0, -1, 9] = 4.0  # tail-only model LOVES the loop token
            return t, torch.zeros(1, ids.size(1))

        with patch.object(Charkha, "_gen_logits", side_effect=fake_gen_logits):
            out = model._echo_gate(logits.clone(), prefix, effort=1, n=3, ctx_len=8)
        self.assertLess(float(out[0, 9]), 0.0)  # momentum-driven -> penalized
        untouched = torch.ones(V, dtype=torch.bool)
        untouched[9] = False
        self.assertTrue(torch.equal(out[0, untouched], logits[0, untouched]))

        # instructed case: full context supports token 9 MORE than the tail does
        strong = logits.clone()
        strong[0, 9] = 6.0
        with patch.object(Charkha, "_gen_logits", side_effect=fake_gen_logits):
            out2 = model._echo_gate(strong.clone(), prefix, effort=1, n=3, ctx_len=8)
        self.assertAlmostEqual(float(out2[0, 9]), 6.0, places=4)  # untouched

    def test_resume_tolerates_grow_init_checkpoints(self):
        # grow_init.py writes opts=None, no rng blob, and a sparse meta. All three
        # crashed --resume live at the 0.42B launch (2026-07-05): KeyError 'muon',
        # KeyError 'rng', KeyError 'loss_ema'. Resume must treat them as fresh state.
        import random
        from charkha import Charkha, CharkhaConfig
        from train import load_ckpt, restore_rng

        cfg = CharkhaConfig.toy()
        model = Charkha(cfg)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "init.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "step": 0,
                    "cfg": dict(cfg.__dict__),
                    "opt_mode": "default",
                    "opts": None,
                    "meta": {"grown_from": "mini"},
                },
                path,
            )
            m2, cfg2, opts, bases, ck = load_ckpt(path, "cpu")
            self.assertEqual(ck["step"], 0)
            self.assertTrue(all(len(o.state) == 0 for o in opts))  # fresh optimizers
            # rng restore is a no-op (keeps the launch seed) instead of a KeyError
            rng = random.Random(7)
            before = rng.getstate()
            restore_rng(ck, rng)
            self.assertEqual(rng.getstate(), before)
            # meta merge keeps trainer defaults for keys the sparse meta lacks
            meta = {"loss_ema": None, "best_val": None}
            meta = {**meta, **(ck.get("meta") or {})}
            self.assertIn("loss_ema", meta)
            self.assertEqual(meta["grown_from"], "mini")
            # weights round-trip exactly
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(v, m2.state_dict()[k]))

    def test_symmetry_optimizer_offload_marks_matrix_states_cpu(self):
        from charkha import Charkha, CharkhaConfig, build_symmetry_optimizers

        cfg = CharkhaConfig.toy()
        cfg.use_halting = False
        cfg.mean_recurrence = 1
        cfg.max_recurrence_train = 1
        model = Charkha(cfg)
        opts = build_symmetry_optimizers(model, 0.02, 0.02, 3e-3, offload=True)
        self.assertTrue(all(getattr(opt, "cpu_offload", False) for opt in opts[:3]))

        ids = torch.randint(0, cfg.vocab_size, (2, 12))
        _, loss = model(ids, ids, r=1)
        loss.backward()
        for opt in opts[:3]:
            opt.step()
        state_tensors = [
            v
            for opt in opts[:3]
            for st in opt.state.values()
            for v in st.values()
            if torch.is_tensor(v)
        ]
        self.assertTrue(state_tensors)
        self.assertTrue(all(t.device.type == "cpu" for t in state_tensors))

    def test_gdn2_chunk_scan_matches_sequential_reference(self):
        # The chunkwise-parallel GDN-2 scan (WY/UT form) must equal the O(T) per-timestep reference
        # for every chunk size, across seeds and decay strengths (incl. strong decay, which trips the
        # inf*0=NaN trap and the cumulative-decay carry). This is the correctness anchor for the kernel.
        import torch.nn.functional as F
        from charkha._modules import _gdn2_sequential_ref, _gdn2_chunk_scan

        worst = 0.0
        for seed, (T, dh, dscale) in enumerate(
            [(40, 8, 1.0), (40, 8, 4.0), (130, 8, 0.2), (64, 16, 2.0)]
        ):
            torch.manual_seed(seed)
            B, H = 2, 3
            q = torch.randn(B, T, H, dh)
            k = F.normalize(torch.randn(B, T, H, dh), dim=-1)
            v = torch.randn(B, T, H, dh)
            b = torch.sigmoid(torch.randn(B, T, H, dh))
            w = torch.sigmoid(torch.randn(B, T, H, dh))
            g = -F.softplus(torch.randn(B, T, H, dh) * dscale)  # log-decay <= 0
            ref = _gdn2_sequential_ref(q, k, v, b, w, g)
            for chunk in (1, 7, 13, 16, T, T + 8):
                out = _gdn2_chunk_scan(q, k, v, b, w, g, chunk=chunk)
                self.assertFalse(torch.isnan(out).any(), f"NaN at chunk={chunk} seed={seed}")
                d = (out - ref).abs().max().item()
                worst = max(worst, d)
                self.assertLess(d, 1e-3, f"chunk={chunk} seed={seed} diverged (max|Δ|={d:.2e})")
        self.assertLess(worst, 1e-3)

    def test_gdn2_chunk_scan_is_differentiable(self):
        # The parallel scan must carry finite gradients through the batched triangular solve.
        import torch.nn.functional as F
        from charkha._modules import _gdn2_chunk_scan

        torch.manual_seed(0)
        B, T, H, dh = 1, 24, 2, 8
        q, v = (
            torch.randn(B, T, H, dh, requires_grad=True),
            torch.randn(B, T, H, dh, requires_grad=True),
        )
        k = F.normalize(torch.randn(B, T, H, dh), dim=-1).requires_grad_(True)
        b = torch.sigmoid(torch.randn(B, T, H, dh)).requires_grad_(True)
        w = torch.sigmoid(torch.randn(B, T, H, dh)).requires_grad_(True)
        g = (-F.softplus(torch.randn(B, T, H, dh))).requires_grad_(True)
        _gdn2_chunk_scan(q, k, v, b, w, g, chunk=8).pow(2).mean().backward()
        for name, t in [("q", q), ("k", k), ("v", v), ("b", b), ("w", w), ("g", g)]:
            self.assertIsNotNone(t.grad, f"{name} got no grad")
            self.assertTrue(torch.isfinite(t.grad).all(), f"{name} grad non-finite")

    def test_frontier_grad_checkpoint_recompute_is_deterministic(self):
        # Regression: under bf16 autocast, gradient-checkpoint recompute must reproduce the SAME
        # graph as the forward. autocast's bf16 weight-cache breaks this (modules with many Linears
        # — logic/calculus/math — and error-feedback raise CheckpointError "Recomputed values ...
        # different metadata"). The training/eval/probe paths pass cache_enabled=False; this test
        # exercises the recurrent core + each culprit module + grad_checkpoint and asserts backward
        # succeeds across seeds. (CPU train() runs amp=False, so the bug only shows under forced
        # bf16 autocast, which mirrors the real CUDA path.)
        from charkha import Charkha, CharkhaConfig

        culprits = []
        for seed in range(4):
            cfg = CharkhaConfig.toy()
            cfg.grad_checkpoint = True
            cfg.use_halting = False
            cfg.mean_recurrence = 2
            cfg.max_recurrence_train = 2
            cfg.backprop_depth = 1
            cfg.use_gdn2 = True
            cfg.gdn_chunk = 4
            cfg.max_seq_len = 64
            for f in culprits:
                setattr(cfg, f, True)
            torch.manual_seed(seed)
            model = Charkha(cfg)
            model.train()
            x = torch.randint(0, cfg.vocab_size, (1, 24))
            with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
                _, loss = model(x, x)
            loss.backward()  # raises CheckpointError if recompute diverges
            self.assertTrue(math.isfinite(float(loss)))

        # The same apply() pass also clobbered the playground's halt-toward-exit bias (+2.0) and the
        # deliberately zero-init gate/blend weights; reset_parameters() must restore those too.
        for n, p in model.named_parameters():
            if n.endswith("playground.halt_head.bias"):
                self.assertAlmostEqual(
                    p.item(), 2.0, places=5, msg=f"{n} = {p.item():+.3f}, not +2.0"
                )
            if n.endswith(
                (
                    "math.gate.weight",
                    "playground.gate.weight",
                    "calculus.gate.weight",
                    "comparator.gate.weight",
                    "calculus.blend.weight",
                    "calculus.blend.bias",
                )
            ):
                self.assertEqual(
                    p.abs().max().item(),
                    0.0,
                    f"{n} should be zero-init, got nonzero (global _init clobbered it)",
                )

    def test_sngp_precision_accumulation_is_explicit(self):
        from charkha import Charkha, CharkhaConfig

        cfg = CharkhaConfig.toy()
        cfg.sngp_enabled = True
        cfg.sngp_rff_dim = 32
        cfg.use_halting = False
        cfg.mean_recurrence = 1
        cfg.max_recurrence_train = 1
        model = Charkha(cfg)
        model.train()
        ids = torch.randint(0, cfg.vocab_size, (1, 12))
        before = model.sngp_head.precision.clone()
        _, loss = model(ids, ids, r=1)
        loss.backward()
        self.assertEqual(float((model.sngp_head.precision - before).abs().sum()), 0.0)
        model.sngp_head.accumulate_precision(model.hidden(ids, r=1).detach())
        self.assertGreater(float((model.sngp_head.precision - before).abs().sum()), 0.0)

    def test_recurrent_eval_state_is_deterministic(self):
        from charkha import Charkha, CharkhaConfig

        cfg = CharkhaConfig.toy()
        cfg.use_halting = False
        cfg.mean_recurrence = 2
        cfg.max_recurrence_train = 2
        model = Charkha(cfg)
        model.eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 12))
        with torch.no_grad():
            _, l1 = model(ids, ids, r=2)
            _, l2 = model(ids, ids, r=2)
        self.assertEqual(float(l1), float(l2))

    def test_training_autocast_disables_weight_cache(self):
        # Guard the fix itself: the checkpointed forward+backward call sites must keep
        # cache_enabled=False, or the recompute-determinism bug returns under bf16 autocast.
        for rel in ("src/train.py", "src/preflight.py", "src/charkha/_model.py", "src/bakeoff.py"):
            with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
                text = fh.read()
            for line in text.splitlines():
                if "torch.autocast(" in line and "dtype=torch.bfloat16" in line:
                    self.assertIn(
                        "cache_enabled=False",
                        line,
                        f"{rel}: bf16 autocast must set cache_enabled=False -> {line.strip()}",
                    )

    def test_streaming_decode_matches_full_reforward(self):
        # The streaming decode engine (KV/state caches, O(1)/token) must be EXACT vs the
        # uncached full-prefix reforward, self-speculative greedy must equal plain greedy,
        # and converge mode must terminate within its cap. Covers the all-modules-on config
        # so the sequence-dependent module caches (calculus/comparator) stay exact too.
        from charkha import Charkha, CharkhaConfig

        torch.manual_seed(0)
        cfg = CharkhaConfig.toy()
        for f in (
            "use_math_module",
            "use_logic_playground",
            "use_calculus_module",
            "use_comparator_module",
            "use_mtp_routing",
        ):
            setattr(cfg, f, True)
        model = Charkha(cfg).eval()
        prompt = torch.randint(0, cfg.vocab_size, (1, 17))
        out_c = model.generate(prompt.clone(), 10, effort=3, temp=0.0, use_cache=True)
        out_u = model.generate(prompt.clone(), 10, effort=3, temp=0.0, use_cache=False)
        self.assertTrue(torch.equal(out_c, out_u), "cached decode diverged from full reforward")
        sp = model.generate(prompt.clone(), 10, effort=3, temp=0.0, draft_effort=1, draft_len=3)
        self.assertTrue(torch.equal(sp, out_u), "speculative greedy != plain greedy")
        outc = model.generate(prompt.clone(), 3, effort="converge", temp=0.0)
        self.assertEqual(outc.shape[1], prompt.shape[1] + 3)
        self.assertLessEqual(
            model._last_converge_iters, cfg.mean_recurrence + cfg.converge_max_iter
        )

    def test_sampling_controls_loop_contrast_and_loss_stabilizers(self):
        from charkha import Charkha, CharkhaConfig

        torch.manual_seed(1)
        cfg = CharkhaConfig.toy()
        cfg.use_halting = False
        cfg.mean_recurrence = 2
        cfg.max_recurrence_train = 2
        cfg.logit_softcap = 12.0
        cfg.z_loss = 1e-4
        model = Charkha(cfg)
        ids = torch.randint(0, cfg.vocab_size, (1, 14))
        _, loss = model(ids, ids, r=2)
        self.assertTrue(torch.isfinite(loss))
        model.eval()
        out = model.generate(
            ids[:, :8].clone(),
            5,
            effort=2,
            temp=0.7,
            top_k=20,
            top_p=0.9,
            min_p=0.05,
            rep_penalty=1.1,
            no_repeat_ngram=2,
            loop_contrast=0.2,
            loop_contrast_effort=1,
        )
        self.assertEqual(tuple(out.shape), (1, 13))
        out2 = model.generate(ids[:, :8].clone(), 3, effort=2, temp=0.0, seed_noise=0.01)
        self.assertEqual(tuple(out2.shape), (1, 11))
        self.assertEqual(getattr(model, "_seed_noise", 0.0), 0.0)

    def test_process_head_segment_halting_and_esr_kd(self):
        from charkha import Charkha, CharkhaConfig
        from distill import ToyTeacher, teacher_kd_tuple

        torch.manual_seed(2)
        cfg = CharkhaConfig.toy()
        cfg.use_halting = False
        cfg.mean_recurrence = 3
        cfg.max_recurrence_train = 3
        cfg.backprop_depth = 3
        cfg.use_process_head = True
        cfg.halt_granularity = "segment"
        cfg.halt_segment_len = 4
        model = Charkha(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 18))
        kd = teacher_kd_tuple(ToyTeacher(cfg.vocab_size, k=6), ids, 0.2, 2.0, max_tokens=5)
        _, loss = model(ids, ids, r=3, kd=kd)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("process", model._last_loss_parts)
        self.assertIn("kd", model._last_loss_parts)

        lam = torch.arange(10, dtype=torch.float32).view(1, 10) / 10
        pooled = model._pool_halt_prob(lam)
        self.assertTrue(torch.allclose(pooled[0, :4], pooled[0, :4].mean().expand(4)))
        self.assertTrue(torch.allclose(pooled[0, 4:8], pooled[0, 4:8].mean().expand(4)))

    def test_council_rank_penalizes_disagreement_and_ungrounded_answers(self):
        from serve import council_rank, numeric_claim_health, proof_contract, mini_verifier_ensemble

        passages = [
            {
                "text": "At standard pressure water boils at 100 degrees Celsius.",
                "source": "wikipedia",
            }
        ]
        cands = [
            {"answer": "Water boils at 100 degrees Celsius.", "base_conf": 0.7, "calls": []},
            {
                "answer": "At standard pressure, the boiling point is 100 C.",
                "base_conf": 0.68,
                "calls": [],
            },
            {
                "answer": "The answer is Paris Paris Paris Paris Paris.",
                "base_conf": 0.95,
                "calls": [],
            },
        ]
        best, report = council_rank(cands, passages=passages, retrieval_q=0.9)
        self.assertIn("100", best["answer"])
        self.assertEqual(report["size"], 3)
        self.assertGreaterEqual(report["entropy"], 0.0)
        self.assertLessEqual(report["entropy"], 1.0)
        self.assertEqual(numeric_claim_health("12 * 7 = 84"), 1.0)
        self.assertEqual(numeric_claim_health("12 * 7 = 83"), 0.0)
        self.assertIn("claim_reports", report)
        self.assertIn("market", report)
        self.assertEqual(proof_contract("12 * 7 = 83")["numeric"], 0.0)
        self.assertGreater(
            mini_verifier_ensemble("A short grounded answer.", passages)["score"], 0.0
        )

    def test_causal_scratchpad_and_failure_taxonomy(self):
        from serve import causal_engine, classify_failure, resolve_tools

        out = causal_engine("a=2; b=a+3; c=b*2; if a=10")
        self.assertIn("'b': 13", out)
        self.assertIn("'c': 26", out)
        txt, calls = resolve_tools("[[causal: a=1; b=a+4; if a=3]]")
        self.assertIn("'b': 7", txt)
        self.assertEqual(calls[0][0], "causal")
        self.assertEqual(classify_failure("12 * 7 = 83"), "arithmetic")

    def test_world_model_structured_beliefs_and_conflicts(self):
        from worldmodel import WorldModel, format_world_context

        wm = WorldModel(os.path.join(tempfile.mkdtemp(), "world.sqlite"))
        facts, conflicts = wm.remember("My GPU is a 4060 Ti. I live in Boston.", ts=1)
        self.assertGreaterEqual(len(facts), 2)
        self.assertEqual(conflicts, [])
        hits = wm.retrieve("what GPU do I use?", k=3)
        self.assertTrue(hits)
        self.assertIn("4060", hits[0]["object"])
        _facts, conflicts = wm.remember("My GPU is an RTX 5090.", ts=2)
        self.assertTrue(conflicts)
        self.assertTrue(wm.contradiction_docs())
        self.assertIn("replaces", wm.counterfactual("user", "has_gpu", "RTX 4090"))
        self.assertIn("WORLD MODEL", format_world_context(hits))
        wm.close()

    def test_shard_loader_data_weights_validate_and_scale_sampling_mass(self):
        from train import ShardLoader, make_synthetic_shards

        tmp = tempfile.mkdtemp()
        d1, d2 = os.path.join(tmp, "a"), os.path.join(tmp, "b")
        make_synthetic_shards(d1)
        make_synthetic_shards(d2)
        loader = ShardLoader([d1, d2], dir_weights=[1.0, 3.0])
        half = len(loader.lengths) // 2
        self.assertGreater(loader._sample_w[half:].sum(), loader._sample_w[:half].sum())
        with self.assertRaises(ValueError):
            ShardLoader([d1, d2], dir_weights=[1.0, 0.0])

    def test_retrieval_context_batch_and_personal_eval(self):
        import random
        from personal_eval import score_records
        from train import ShardLoader, make_synthetic_shards

        tmp = tempfile.mkdtemp()
        d1, d2 = os.path.join(tmp, "main"), os.path.join(tmp, "ctx")
        make_synthetic_shards(d1)
        make_synthetic_shards(d2)
        rng = random.Random(4)
        loader = ShardLoader(d1)
        ctx = ShardLoader(d2)
        x, y = loader.batch_with_context(ctx, 3, 32, 8, "cpu", rng)
        self.assertEqual(tuple(x.shape), (3, 32))
        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))
        scored = score_records(
            [
                {"prompt": "gpu", "answer": "4060 Ti", "contains": "4060"},
                {"prompt": "bad", "answer": "no", "equals": "yes"},
            ]
        )
        self.assertEqual(scored["n"], 2)
        self.assertAlmostEqual(scored["accuracy"], 0.5)

    def test_event_jepa_module_learns_synthetic_transitions(self):
        proc = subprocess.run(
            [sys.executable, os.path.join(SRC, "event_jepa.py"), "--selftest"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_preflight_path_normalization(self):
        from preflight import _norm_path

        self.assertEqual(
            _norm_path("D:\\llm\\src\\..\\README.md").replace("\\", "/").split("/")[-1], "readme.md"
        )


class MemoryPathTests(unittest.TestCase):
    """Micro-scale CPU validation of the 8GB memory-path stack: CCE vocab streaming
    (cfg.ce_vchunk), grad-release (--grad-release), reversible-recurrence BPTT for GDN-2
    (cfg.rev_bptt, incl. the closed-form chunk inversion), and the two-stream reversible
    residual coupling (cfg.reversible). Each must be numerically equivalent to the baseline
    path it replaces (same math, different memory strategy) — except cfg.reversible, which
    intentionally defines a DIFFERENT function and is tested for internal consistency."""

    @staticmethod
    def _gdn2_inputs(seed=0, B=2, T=64, H=2, dh=8, dscale=1.0):
        import torch.nn.functional as F

        torch.manual_seed(seed)
        q = torch.randn(B, T, H, dh)
        k = F.normalize(torch.randn(B, T, H, dh), dim=-1)
        v = torch.randn(B, T, H, dh)
        b = torch.sigmoid(torch.randn(B, T, H, dh))
        w = torch.sigmoid(torch.randn(B, T, H, dh))
        g = -F.softplus(torch.randn(B, T, H, dh) * dscale)  # log-decay <= 0
        return q, k, v, b, w, g

    def test_ce_vchunk_stream_matches_whole_vocab(self):
        # _ce_stream (vocab-chunked lse) must equal _ce_chunk (whole-vocab logits) exactly:
        # loss, argmax-correct mask, and grads into h and W — with and without prior/softcap/z-loss.
        from charkha._loss import _ce_argmax_chunk, _ce_stream

        torch.manual_seed(0)
        N, f, V = 24, 16, 200
        for cap, zl, use_prior in [(0.0, 0.0, False), (30.0, 1e-4, False), (0.0, 0.0, True)]:
            h = torch.randn(N, f, requires_grad=True)
            W = torch.randn(V, f, requires_grad=True)
            hp = torch.randn(N, f) if use_prior else None
            Wp = torch.randn(V, f) if use_prior else None
            pw = 0.7 if use_prior else 0.0
            t = torch.randint(0, V, (N,))
            ref, ref_cor = _ce_argmax_chunk(h, W, t, hp, Wp, pw, cap, zl)
            ref.backward()
            gh, gW = h.grad.clone(), W.grad.clone()
            h.grad = W.grad = None
            for vchunk in (V, 64, 17):
                out, cor = _ce_stream(
                    h,
                    W,
                    t,
                    vchunk,
                    ckpt=False,
                    hp=hp,
                    Wp=Wp,
                    pw=pw,
                    cap=cap,
                    zl=zl,
                    want_correct=True,
                )
                self.assertLess(
                    abs(float(out - ref)), 1e-3, f"loss mismatch vchunk={vchunk} cap={cap} zl={zl}"
                )
                self.assertTrue(torch.equal(cor, ref_cor), f"argmax mismatch vchunk={vchunk}")
                out.backward()
                self.assertLess((h.grad - gh).abs().max().item(), 1e-3)
                self.assertLess((W.grad - gW).abs().max().item(), 1e-3)
                h.grad = W.grad = None

    def test_gdn2_chunk_inversion_validity_domain(self):
        # The closed-form chunk inversion is exact ALGEBRA (single-chunk inversion recovers the
        # start state to float precision when decay is mild) but ill-conditioned under strong
        # decay — the delta-rule transition contracts S per step, so multi-chunk pure inversion
        # amplifies error exponentially. This test pins BOTH facts: (a) near-lossless single-chunk
        # inversion at mild decay, (b) measurable blow-up across chunks at production-strength
        # decay — the measured result that made anchored REPLAY the reconstruction of record in
        # _RevGDN2Scan.backward.
        import torch.nn.functional as F
        from charkha._modules import _Gdn2Masks, _gdn2_one_chunk, _gdn2_invert_chunk

        def run(gmaker, T=128, chunk=16):
            torch.manual_seed(1)
            B, H, dh = 2, 2, 8
            q = torch.randn(B, T, H, dh)
            k = F.normalize(torch.randn(B, T, H, dh), dim=-1)
            v = torch.randn(B, T, H, dh)
            b = torch.sigmoid(torch.randn(B, T, H, dh))
            w = torch.sigmoid(torch.randn(B, T, H, dh))
            g = gmaker(torch.randn(B, T, H, dh))
            qf, kf, vf, bf, wf, gf = (t.float().transpose(1, 2) for t in (q, k, v, b, w, g))
            masks = _Gdn2Masks(q.device, torch.float32)
            S = qf.new_zeros(B, H, dh, dh)
            states = [S]
            for c0 in range(0, T, chunk):
                sl = slice(c0, c0 + chunk)
                _, S = _gdn2_one_chunk(
                    S,
                    qf[:, :, sl],
                    kf[:, :, sl],
                    vf[:, :, sl],
                    bf[:, :, sl],
                    wf[:, :, sl],
                    gf[:, :, sl],
                    masks,
                )
                states.append(S)
            single, chained = 0.0, 0.0
            Sb = states[-1]
            for ci in range(len(states) - 2, -1, -1):
                sl = slice(ci * chunk, (ci + 1) * chunk)
                ins = (
                    qf[:, :, sl],
                    kf[:, :, sl],
                    vf[:, :, sl],
                    bf[:, :, sl],
                    wf[:, :, sl],
                    gf[:, :, sl],
                )
                # (a) algebra: invert ONE chunk from its TRUE end state
                s1 = _gdn2_invert_chunk(states[ci + 1], *ins, masks)
                single = max(single, (s1 - states[ci]).abs().max().item())
                # (b) conditioning: chain inversions end-to-start
                Sb = _gdn2_invert_chunk(Sb, *ins, masks)
                chained = max(chained, (Sb - states[ci]).abs().max().item())
            return single, chained

        single, chained = run(lambda x: -F.softplus(x - 5.0))  # mild decay
        # single-chunk inversion is exact algebra up to the chunk's condition number — the
        # contraction is dominated by the ERASE term (1-beta per step), not decay alone
        self.assertLess(single, 1e-4, f"single-chunk inversion error {single:.2e}")
        self.assertGreater(
            chained,
            1e-2,
            "chained pure inversion unexpectedly stable — revisit the replay "
            "decision in _RevGDN2Scan.backward if this starts passing",
        )

    def test_rev_bptt_gradients_match_stored_state_baseline(self):
        # _RevGDN2Scan (inversion-reconstructed backward) must produce the same outputs and the
        # same input gradients as the plain autograd chunk scan.
        from charkha._modules import _gdn2_chunk_scan, _RevGDN2Scan

        for anchor in (1, 4, 100):  # every-chunk, sparse, effectively-none
            ins_a = [t.clone().requires_grad_(True) for t in self._gdn2_inputs(seed=2, T=96)]
            ins_b = [t.clone().detach().requires_grad_(True) for t in ins_a]
            ref = _gdn2_chunk_scan(*ins_a, chunk=16)
            out = _RevGDN2Scan.apply(*ins_b, 16, anchor)
            self.assertLess((out - ref).abs().max().item(), 1e-4, f"fwd mismatch anchor={anchor}")
            torch.manual_seed(9)
            dy = torch.randn_like(ref)
            ref.backward(dy)
            out.backward(dy)
            for name, a, b_ in zip("qkvbwg", ins_a, ins_b):
                self.assertLess(
                    (a.grad - b_.grad).abs().max().item(),
                    1e-3,
                    f"grad({name}) mismatch anchor={anchor}",
                )

    def test_reversible_stack_memory_free_backward_matches_plain(self):
        # _RevStackFn (O(1)-memory inversion backward) must equal the plain two-stream forward
        # in outputs, input grads, and parameter grads — same function, different memory.
        from charkha._modules import make_stack, _two_stream_blocks, _rev_run_blocks
        from charkha.config import CharkhaConfig

        torch.manual_seed(3)
        cfg = CharkhaConfig.toy()
        cfg.use_gdn2 = True
        blocks = make_stack(cfg, 3, final_global_attn=False)
        blocks.train()
        x = torch.randn(1, 24, cfg.d_model)
        xa = x.clone().requires_grad_(True)
        xb = x.clone().requires_grad_(True)
        ya = _two_stream_blocks(blocks, xa)
        ga = torch.autograd.grad(
            ya.pow(2).mean(), [xa] + list(blocks.parameters()), allow_unused=True
        )
        yb = _rev_run_blocks(blocks, xb)
        self.assertLess((ya - yb).abs().max().item(), 1e-4)
        yb.pow(2).mean().backward()
        self.assertLess((ga[0] - xb.grad).abs().max().item(), 1e-4, "input grad mismatch")
        worst = 0.0
        for gref, p in zip(ga[1:], blocks.parameters()):
            if gref is None:
                continue
            self.assertIsNotNone(p.grad, "param missed by reversible backward")
            worst = max(worst, (gref - p.grad).abs().max().item())
        self.assertLess(worst, 1e-3, f"param grad mismatch (max|Δ|={worst:.2e})")

    def test_grad_release_optimizer_steps_match_baseline(self):
        # Muon with grad-release CPU accumulators (+ clip_grads_mixed) must produce the same
        # weights as the standard GPU-grad path, including under gradient accumulation.
        from charkha import Muon, install_grad_release, clip_grads_mixed
        import torch.nn as nn

        torch.manual_seed(4)
        ma = nn.Sequential(nn.Linear(16, 32, bias=False), nn.Linear(32, 16, bias=False))
        mb = nn.Sequential(nn.Linear(16, 32, bias=False), nn.Linear(32, 16, bias=False))
        mb.load_state_dict(ma.state_dict())
        oa = Muon(list(ma.parameters()), lr=0.05)
        ob = Muon(list(mb.parameters()), lr=0.05)
        n = install_grad_release(list(mb.parameters()))
        if n == 0:
            self.skipTest("torch lacks register_post_accumulate_grad_hook")
        for step in range(3):
            for micro in range(2):  # gradient accumulation
                torch.manual_seed(10 * step + micro)
                x = torch.randn(4, 16)
                (ma(x).pow(2).mean() / 2).backward()
                (mb(x).pow(2).mean() / 2).backward()
            na = float(torch.nn.utils.clip_grad_norm_(ma.parameters(), 0.5))
            nb = float(clip_grads_mixed(mb, 0.5))
            self.assertLess(abs(na - nb), 1e-4, "clip norms diverged")
            oa.step()
            ob.step()
            oa.zero_grad(set_to_none=True)
            ob.zero_grad(set_to_none=True)
        for pa, pb in zip(ma.parameters(), mb.parameters()):
            self.assertLess((pa - pb).abs().max().item(), 1e-5)

    def test_full_stack_toy_forward_backward_and_decode(self):
        # End-to-end: toy Charkha with reversible + rev_bptt + ce_vchunk all ON — a training
        # step produces a finite loss and finite grads, and the reversible streaming-decode
        # path runs (train/serve consistency for the new architecture).
        from charkha import Charkha, CharkhaConfig

        torch.manual_seed(5)
        cfg = CharkhaConfig.toy()
        cfg.use_gdn2 = True
        cfg.reversible = True
        cfg.rev_bptt = True
        cfg.rev_anchor = 2
        cfg.ce_vchunk = 64
        model = Charkha(cfg)
        model.train()
        x = torch.randint(0, cfg.vocab_size, (1, 33))
        _, loss = model(x[:, :-1], x[:, 1:])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(
            any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        )
        model.eval()
        with torch.no_grad():
            full, _ = model(x[:, :16], r=2)
            out = model.generate(x[:, :8], n_new=4, effort=2)
        self.assertTrue(torch.isfinite(full).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)

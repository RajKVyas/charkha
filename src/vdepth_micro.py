#!/usr/bin/env python3
"""Micro-E2: virtual-depth slab alignment, convergence + specialization (CPU-only).

The virtual-depth claim was the last of the four with zero empirical support:
to CONTIGUOUS SLABS of a deeper teacher, beating fixed-depth many-to-one alignment
at matched parameters? This file runs the registered E2 shape at micro scale, and is
also the first live exercise of the per-loop adapters (use_loop_adapters) on an
actually-recurrent model — the mainline 0.42B trains non-recurrent, so the adapters'
specialization story is only testable here.

Protocol (all arms share the pretrained teacher and identical align budgets):
  teacher   toy CHARKHA, 8 fixed core blocks, pretrained on the shared digit-sorting
            LM task until fluent; its 8 per-layer states define 4 slabs of 2.
  recur     toy student, 2 core blocks looped r=4 (8 virtual blocks), WITH per-loop
            adapters: loop pass t aligns to teacher slab t.
  recur0    identical student WITHOUT adapters (weight-tied purity control): can the
            same function play four different slabs?
  fixed     matched-parameter student, same 2 core blocks, NO recurrence: block 1
            must align to slabs 1+2, block 2 to slabs 3+4 (the many-to-one baseline
            the paper says depth-limits fixed students).
Alignment signal: MSE between position-space gram matrices (h_norm @ h_norm^T),
dimension-free per the claim (defined over positions, not embedding dims).

Readout:
  (i)  convergence — mean per-target align loss, recur vs fixed (registered E2-i).
  (ii) specialization — 4x4 similarity(pass t, slab u); diagonal argmax hits out
       of 4, recur vs recur0 (registered E2-ii, and the adapters' first real test).

Micro evidence only: positives here justify GPU-scale E2, negatives are reported
as micro-scale negatives, not claim-killers.

Usage:
    python src/vdepth_micro.py --selftest
    python src/vdepth_micro.py --run [--seeds 2 --pre-steps 300 --align-steps 200]

"""

import argparse
import json
import sys
import time

import torch
import torch.nn.functional as F

from charkha import Charkha, CharkhaConfig
from micro_task import make_split, SEP  # noqa: F401


def batch_from(split, i, bs):
    rows = [split[(i * bs + j) % len(split)] for j in range(bs)]
    seqs = [torch.tensor(p + a) for p, a in rows]
    T = max(len(s) for s in seqs)
    x = torch.zeros(bs, T, dtype=torch.long)
    for j, s in enumerate(seqs):
        x[j, : len(s)] = s
    return x


def teacher_cfg():
    return CharkhaConfig(
        vocab_size=64,
        d_model=96,
        n_heads=4,
        n_kv_heads=2,
        d_ff=256,
        n_prelude=1,
        n_core=8,
        n_coda=1,
        window=32,
        max_seq_len=64,
        use_recurrence=False,
    )


def student_cfg(adapters):
    return CharkhaConfig(
        vocab_size=64,
        d_model=96,
        n_heads=4,
        n_kv_heads=2,
        d_ff=256,
        n_prelude=1,
        n_core=2,
        n_coda=1,
        window=32,
        max_seq_len=64,
        use_recurrence=True,
        mean_recurrence=4,
        max_recurrence_train=4,
        backprop_depth=4,
        use_loop_adapters=adapters,
    )


def pretrain(model, split, steps, bs, lr, seed):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for i in range(steps):
        x = batch_from(split, int(torch.randint(0, 10**6, (1,), generator=g)), bs)
        _, loss = model(x[:, :-1], x[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    return float(loss)


def teacher_slab_grams(teacher, x, slab):
    """Per-layer core states -> gram of the state at each slab boundary."""
    with torch.no_grad():
        h = teacher._run_blocks(teacher.prelude, teacher._tok_embed(x))
        grams = []
        for i, blk in enumerate(teacher.core):
            h = blk(h)
            if (i + 1) % slab == 0:
                z = F.normalize(h, dim=-1)
                grams.append(z @ z.transpose(1, 2))
    return grams  # n_slabs x (B,T,T)


def student_pass_states(model, x, r):
    """Manual unroll of the recurrent core, keeping every pass's state."""
    e = model._run_blocks(model.prelude, model._tok_embed(x))
    s = model._initial_core_state(e)
    states = []
    for n in range(r):
        s = model._core_step(s, e, n)
        states.append(s)
    return states


def fixed_block_states(model, x):
    """The fixed-depth baseline's per-block states (2 of them)."""
    h = model._run_blocks(model.prelude, model._tok_embed(x))
    states = []
    for blk in model.core:
        h = blk(h)
        states.append(h)
    return states


def gram(h):
    z = F.normalize(h, dim=-1)
    return z @ z.transpose(1, 2)


def align(arm, student, teacher, split, steps, bs, lr, slab, seed):
    """Train the student on the alignment loss only; return per-target loss."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    student.train()
    last = None
    for i in range(steps):
        x = batch_from(split, int(torch.randint(0, 10**6, (1,), generator=g)), bs)
        tg = teacher_slab_grams(teacher, x, slab)
        if arm == "fixed":
            ss = fixed_block_states(student, x)
            # many-to-one: block j answers for slabs 2j and 2j+1
            losses = [F.mse_loss(gram(ss[j]), tg[2 * j + k]) for j in range(2) for k in range(2)]
        else:
            ss = student_pass_states(student, x, r=len(tg))
            losses = [F.mse_loss(gram(s_), t_) for s_, t_ in zip(ss, tg)]
        loss = torch.stack(losses).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        last = float(loss)
    student.eval()
    return last


@torch.no_grad()
def specialization(student, teacher, split, bs, slab, n_batches=4):
    """4x4 similarity: -MSE(pass t gram, slab u gram); diagonal argmax hits."""
    r = len(list(teacher.core)) // slab
    sim = torch.zeros(r, r)
    for i in range(n_batches):
        x = batch_from(split, i, bs)
        tg = teacher_slab_grams(teacher, x, slab)
        ss = student_pass_states(student, x, r)
        for t in range(r):
            for u in range(r):
                sim[t, u] -= float(F.mse_loss(gram(ss[t]), tg[u]))
    hits = int((sim.argmax(dim=1) == torch.arange(r)).sum())
    return sim / n_batches, hits


@torch.no_grad()
def heldout_nll(model, split, bs, n_batches=8, offset=1024):
    tot, n = 0.0, 0
    for i in range(n_batches):
        x = batch_from(split, offset + i, bs)
        h = model.hidden(x[:, :-1])
        h, W = model._head_hw(h)
        logits = F.linear(h.reshape(-1, h.size(-1)), W)
        tot += float(F.cross_entropy(logits, x[:, 1:].reshape(-1), reduction="sum"))
        n += logits.size(0)
    return tot / n


def kd_pretrain(student, teacher, split, steps, bs, lr, seed):
    """Token-matched logit-distillation control: same budget as `align`, but the
    signal is the teacher's OUTPUT distribution, no slab structure."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    student.train()
    for i in range(steps):
        x = batch_from(split, int(torch.randint(0, 10**6, (1,), generator=g)), bs)
        with torch.no_grad():
            th = teacher.hidden(x)
            th, tW = teacher._head_hw(th)
            tlog = F.linear(th.reshape(-1, th.size(-1)), tW)
        sh = student.hidden(x)
        sh, sW = student._head_hw(sh)
        slog = F.linear(sh.reshape(-1, sh.size(-1)), sW)
        loss = F.kl_div(F.log_softmax(slog, -1), F.softmax(tlog, -1), reduction="batchmean")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
    student.eval()


def dividend(args):
    """Micro-E4: recurrence dividend. Pretrain the recurrent student at its training
    depth (r=4), then evaluate held-out NLL at r in {1,2,4,8}. The premise of the
    virtual-depth section needs extra iterations to help at all — and r=8 probes
    extrapolation past the training depth."""
    t0 = time.time()
    results = {}
    for s in range(args.seeds):
        g = torch.Generator().manual_seed(500 + s)
        split = make_split(4096, 6, g)
        torch.manual_seed(200 + s)
        st = Charkha(student_cfg(False))
        pl = pretrain(
            st, split, args.pre_steps + args.align_steps + args.ft_steps, args.bs, 3e-3, seed=s
        )
        for r in (1, 2, 4, 8):
            tot, n = 0.0, 0
            for i in range(8):
                x = batch_from(split, 1024 + i, args.bs)
                with torch.no_grad():
                    h = st.hidden(x[:, :-1], r=r)
                    h, W = st._head_hw(h)
                    logits = F.linear(h.reshape(-1, h.size(-1)), W)
                    tot += float(F.cross_entropy(logits, x[:, 1:].reshape(-1), reduction="sum"))
                    n += logits.size(0)
            results.setdefault(r, []).append(tot / n)
            print(f"[seed {s}] r={r}  held-out NLL {tot / n:.4f}  ({time.time() - t0:.0f}s)")
    summary = {f"r{r}_mean_nll": round(sum(v) / len(v), 4) for r, v in results.items()}
    summary["seeds"] = args.seeds
    print(json.dumps(summary))
    return 0


def transfer(args):
    """Micro-E2-iii / E3: identical total budgets, identical CE fine-tune; the arms
    differ only in what the FIRST phase taught (slab grams / output logits / CE)."""
    t0 = time.time()
    results = {}
    for s in range(args.seeds):
        g = torch.Generator().manual_seed(500 + s)
        split = make_split(4096, 6, g)
        torch.manual_seed(100 + s)
        teacher = Charkha(teacher_cfg())
        tl = pretrain(teacher, split, args.pre_steps, args.bs, 3e-3, seed=s)
        print(f"[seed {s}] teacher pretrained (loss {tl:.3f}, {time.time() - t0:.0f}s)")
        for arm in ("slab", "kd", "none"):
            torch.manual_seed(200 + s)
            st = Charkha(student_cfg(False))
            if arm == "slab":
                align(
                    "recur",
                    st,
                    teacher,
                    split,
                    args.align_steps,
                    args.bs,
                    3e-3,
                    slab=2,
                    seed=300 + s,
                )
            elif arm == "kd":
                kd_pretrain(st, teacher, split, args.align_steps, args.bs, 3e-3, seed=300 + s)
            else:  # budget-matched: extra CE instead of phase 1
                pretrain(st, split, args.align_steps, args.bs, 3e-3, seed=300 + s)
            pretrain(st, split, args.ft_steps, args.bs, 3e-3, seed=400 + s)
            nll = heldout_nll(st, split, args.bs)
            results.setdefault(arm, []).append(nll)
            print(f"[seed {s}] {arm:4s} held-out NLL {nll:.4f}  ({time.time() - t0:.0f}s)")
    summary = {f"{a}_mean_nll": round(sum(v) / len(v), 4) for a, v in results.items()}
    summary.update(
        {
            "seeds": args.seeds,
            "align_steps": args.align_steps,
            "ft_steps": args.ft_steps,
            "per_seed": {a: [round(x, 4) for x in v] for a, v in results.items()},
        }
    )
    print(json.dumps(summary))
    return 0


def run(args):
    t0 = time.time()
    results = []
    for s in range(args.seeds):
        g = torch.Generator().manual_seed(500 + s)
        split = make_split(2048, 6, g)
        torch.manual_seed(100 + s)
        teacher = Charkha(teacher_cfg())
        tl = pretrain(teacher, split, args.pre_steps, args.bs, 3e-3, seed=s)
        print(f"[seed {s}] teacher pretrained (final loss {tl:.3f}, {time.time() - t0:.0f}s)")
        row = {"seed": s, "teacher_loss": round(tl, 3)}
        for arm, adapters in (("recur", True), ("recur0", False), ("fixed", False)):
            torch.manual_seed(200 + s)  # identical init where shapes match
            cfg = (
                student_cfg(adapters)
                if arm != "fixed"
                else CharkhaConfig(
                    **{
                        **student_cfg(False).__dict__,
                        "use_recurrence": False,
                        "use_loop_adapters": False,
                    }
                )
            )
            st = Charkha(cfg)
            al = align(
                arm, st, teacher, split, args.align_steps, args.bs, 3e-3, slab=2, seed=300 + s
            )
            row[f"{arm}_align"] = round(al, 5)
            if arm != "fixed":
                _, hits = specialization(st, teacher, split, args.bs, slab=2)
                row[f"{arm}_diag_hits"] = hits
            print(
                f"[seed {s}] {arm:6s} align {al:.5f}"
                + (f"  diag {row.get(arm + '_diag_hits')}/4" if arm != "fixed" else "")
                + f"   ({time.time() - t0:.0f}s)"
            )
        results.append(row)
    n = len(results)
    summary = {
        "recur_mean_align": round(sum(r["recur_align"] for r in results) / n, 5),
        "recur0_mean_align": round(sum(r["recur0_align"] for r in results) / n, 5),
        "fixed_mean_align": round(sum(r["fixed_align"] for r in results) / n, 5),
        "recur_diag_hits": [r["recur_diag_hits"] for r in results],
        "recur0_diag_hits": [r["recur0_diag_hits"] for r in results],
        "seeds": n,
        "pre_steps": args.pre_steps,
        "align_steps": args.align_steps,
    }
    print(json.dumps(summary))
    return 0


def _selftest():
    torch.manual_seed(0)
    checks = 0
    g = torch.Generator().manual_seed(1)
    split = make_split(64, 4, g)
    x = batch_from(split, 0, 2)
    assert x.dim() == 2 and x.size(0) == 2
    checks += 1

    teacher = Charkha(teacher_cfg())
    tg = teacher_slab_grams(teacher, x, slab=2)
    assert len(tg) == 4 and tg[0].shape == (2, x.size(1), x.size(1))
    # grams are symmetric with unit diagonal (normalized states)
    assert torch.allclose(tg[0], tg[0].transpose(1, 2), atol=1e-5)
    assert torch.allclose(tg[0].diagonal(dim1=1, dim2=2), torch.ones(2, x.size(1)), atol=1e-4)
    checks += 3

    st = Charkha(student_cfg(True))
    torch.manual_seed(42)  # _initial_core_state is random per call
    ss = student_pass_states(st, x, r=4)
    assert len(ss) == 4 and ss[0].shape[-1] == 96
    # adapters at init are exact no-ops, so pass states differ only via the loop
    # dynamics; after nudging an adapter, pass 1 must change while pass 0 does not
    base0, base1 = ss[0].clone(), ss[1].clone()
    with torch.no_grad():
        st.loop_adapters.up[1].weight.fill_(0.05)
    torch.manual_seed(42)
    ss2 = student_pass_states(st, x, r=4)
    assert torch.equal(ss2[0], base0) and not torch.equal(ss2[1], base1)
    checks += 2

    fx = Charkha(
        CharkhaConfig(
            **{**student_cfg(False).__dict__, "use_recurrence": False, "use_loop_adapters": False}
        )
    )
    al = align("fixed", fx, teacher, split, steps=2, bs=2, lr=1e-3, slab=2, seed=2)
    ar = align("recur", st, teacher, split, steps=2, bs=2, lr=1e-3, slab=2, seed=2)
    assert all(map(lambda v: v == v and v >= 0, (al, ar)))  # finite, non-negative
    checks += 1

    sim, hits = specialization(st, teacher, split, bs=2, slab=2, n_batches=1)
    assert sim.shape == (4, 4) and 0 <= hits <= 4
    checks += 1

    kd_pretrain(st, teacher, split, steps=2, bs=2, lr=1e-3, seed=4)
    nll = heldout_nll(st, split, bs=2, n_batches=1, offset=32)
    assert nll == nll and nll > 0
    checks += 1

    print(f"[selftest] vdepth_micro.py: all checks passed ({checks} groups)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--transfer", action="store_true")
    ap.add_argument("--dividend", action="store_true")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--pre-steps", type=int, default=300)
    ap.add_argument("--align-steps", type=int, default=200)
    ap.add_argument("--ft-steps", type=int, default=150)
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    if a.transfer:
        sys.exit(transfer(a))
    if a.dividend:
        sys.exit(dividend(a))
    if a.run:
        sys.exit(run(a))
    ap.print_help()

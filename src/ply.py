#!/usr/bin/env python3
"""Ply: test-time search over depth-recurrence trajectories (latent branch-and-select).

Mechanism (implemented + proofed here; capability claims are unverified):
a depth-recurrent model's forward pass contains a free search space that token-level
methods cannot reach: the trajectory of the looped core. Ply branches that trajectory
N ways inside one forward path (branch b perturbs the loop-injected input e by
Gaussian noise; branch 0 is always unperturbed), lets each branch settle over the
same r loops, scores every branch with the model's own internal signals, and emits
logits from the winner. Selection happens before any token exists:

    prompt -> prelude -> [core x N branches, r loops each] -> select -> coda -> token
                          (latent search, invisible in output)

Contrast with token-space search: best-of-N / self-consistency / tree-of-thought pay
per-token generation cost, produce N visible candidate texts, and need an external
scorer. Ply pays extra core-loop FLOPs, produces nothing visible, and can reuse the
confidence and margin signals the model already exposes. Nearest in-repo prior art is
serve.py's council (latent-seeded best-of-N over full generations, reranked at the
end); Ply moves the branch point inside the forward pass and the selection to
per-step granularity. If that delta proves empirically empty (E9), Ply demotes itself
to a council footnote.

Hard properties covered by --selftest:
  1. Exactness at N=1: ply_forward with one branch returns bit-identical logits to
     model(idx, r=r); the mechanism is a strict superset of the plain forward.
  2. Determinism: a caller-supplied generator fully determines the branch noise.
  3. Selection soundness: each batch row receives its own argmax-scoring branch.
  4. Zero token footprint: ply_generate transcripts contain ordinary vocab only and
     the interface is drop-in (prompt in, tokens out).

Score functions ('conf' default): 'conf' = trained P(top-1 correct) at the last
position; 'margin' = top1-top2 logit gap (no extra head needed); 'entropy' =
negative predictive entropy. 'target' is an oracle-only E9 measurement mode: it
selects the branch assigning highest log-probability to a supplied next token and
must never be used as a serving policy. Selecting the model's most-confident thought
risks confidently-wrong trajectories; that risk is E9's control arm, not a footnote.

Usage:
    python src/ply.py --selftest

"""

import argparse
import sys

import torch
import torch.nn.functional as F


def _score_matrix(logits_last, conf_last, how, target_next=None):
    """Return branch scores with shape (N, B), higher is better."""
    if how == "conf":
        return conf_last.float()
    if how == "margin":
        top2 = logits_last.float().topk(2, dim=-1).values
        return top2[..., 0] - top2[..., 1]
    if how == "entropy":
        p = F.softmax(logits_last.float(), dim=-1)
        return -(-(p * (p + 1e-12).log()).sum(-1))
    if how == "target":
        if target_next is None:
            raise ValueError("score='target' requires target_next")
        target = torch.as_tensor(target_next, device=logits_last.device, dtype=torch.long)
        if target.ndim == 0:
            target = target.view(1).expand(logits_last.size(1))
        if target.numel() != logits_last.size(1):
            raise ValueError("target_next must be scalar or have one token per batch row")
        target = target.view(1, -1, 1).expand(logits_last.size(0), -1, 1)
        return F.log_softmax(logits_last.float(), dim=-1).gather(-1, target).squeeze(-1)
    raise ValueError(f"score must be 'conf', 'margin', 'entropy', or 'target', got {how!r}")


def _branch_states(model, e_flat, r, mtp_flat, perturb, noise, n_branches, B, gen):
    """Run the core over the flattened branch batch. perturb='input' relies on the
    caller having already perturbed e; perturb='state' additionally gives branches
    1..N-1 a noisy INITIAL core state (in-distribution with training's
    recurrent_state_noise and the council's seed_noise), branch 0 stays exact."""
    if not model.cfg.use_recurrence:
        return model._run_blocks(model.core, e_flat)
    if perturb != "state" or n_branches == 1:
        return model._run_core_fixed(e_flat, r, mtp_future=mtp_flat)
    # branch 0 exact, branches 1.. with seeded initial-state noise
    s0 = model._run_core_fixed(e_flat[:B], r, mtp_future=None if mtp_flat is None else mtp_flat[:B])
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=gen))
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    model._seed_noise = float(noise)
    try:
        s_rest = model._run_core_fixed(
            e_flat[B:], r, mtp_future=None if mtp_flat is None else mtp_flat[B:]
        )
    finally:
        model._seed_noise = 0.0
        torch.random.set_rng_state(rng_state)
    return torch.cat([s0, s_rest], dim=0)


@torch.no_grad()
def ply_forward(
    model,
    idx,
    r=4,
    n_branches=4,
    noise=0.05,
    score="conf",
    gen=None,
    target_next=None,
    perturb="input",
):
    """One forward pass with latent branch-and-select over the recurrent core.

    Branch 0 is the unperturbed trajectory (N=1 is exactly the plain forward);
    branches 1..N-1 perturb the loop-injected input e (perturb='input') or the
    initial core state (perturb='state' — in-distribution with training's
    recurrent_state_noise, the micro-E9-motivated mode). Returns
    (logits, conf, info) where logits/conf come from the winning branch and
    info = {'chosen', 'scores'} with one chosen branch per batch row.
    """
    if n_branches < 1:
        raise ValueError("n_branches must be >= 1")
    if model.training:
        raise RuntimeError("ply_forward is inference-only; call model.eval() first")
    if not isinstance(r, int) or r < 1:
        raise ValueError("ply_forward requires a positive fixed integer r")
    if perturb not in ("input", "state"):
        raise ValueError(f"perturb must be 'input' or 'state', got {perturb!r}")

    model._core_traj = None
    model._last_convergence = None
    x = model._tok_embed(idx)
    e = model._run_blocks(model.prelude, x)
    mtp_future = (
        F.silu(model.mtp_proj(e))
        if (model.cfg.use_mtp_routing and model.cfg.mtp_weight > 0)
        else None
    )

    B, T, D = e.shape
    if n_branches == 1:
        e_flat = e
        mtp_flat = mtp_future
    else:
        if perturb == "input":
            eps = torch.randn((n_branches - 1, B, T, D), generator=gen, dtype=torch.float32)
            eps = eps.to(device=e.device, dtype=e.dtype) * float(noise)
            e_br = torch.cat([e.unsqueeze(0), e.unsqueeze(0) + eps], dim=0)
        else:  # 'state': e identical per branch
            e_br = e.unsqueeze(0).expand(n_branches, -1, -1, -1)
        e_flat = e_br.reshape(n_branches * B, T, D)
        mtp_flat = (
            None
            if mtp_future is None
            else mtp_future.unsqueeze(0)
            .expand(n_branches, -1, -1, -1)
            .reshape(n_branches * B, T, D)
        )

    s = _branch_states(model, e_flat, r, mtp_flat, perturb, noise, n_branches, B, gen)
    h = model.norm_f(model._run_blocks(model.coda, s))
    hh, Wh = model._head_hw(h)
    logits = model._softcap_logits(F.linear(hh, Wh))
    conf = torch.sigmoid(model.conf_head(h)).squeeze(-1)

    logits_nb = logits.reshape(n_branches, B, T, -1)
    conf_nb = conf.reshape(n_branches, B, T)
    if score == "consistency":
        # latent self-consistency: prefer the branch whose next-token argmax agrees
        # with the most other branches (majority vote in latent space — the
        # training-free selector; conf breaks ties). No extra tokens, no head.
        top = logits_nb[:, :, -1, :].argmax(-1)  # (N, B)
        agree = (top.unsqueeze(0) == top.unsqueeze(1)).sum(0)  # (N, B)
        scores = agree.float() + 1e-3 * conf_nb[:, :, -1].float()
    else:
        scores = _score_matrix(
            logits_nb[:, :, -1, :], conf_nb[:, :, -1], score, target_next=target_next
        )
    chosen = scores.argmax(dim=0)  # (B,)
    b_ix = chosen.view(1, B, 1, 1).expand(1, B, T, logits_nb.size(-1))
    c_ix = chosen.view(1, B, 1).expand(1, B, T)
    out_logits = logits_nb.gather(0, b_ix).squeeze(0)
    out_conf = conf_nb.gather(0, c_ix).squeeze(0)
    info = {
        "chosen": [int(x) for x in chosen.detach().cpu().tolist()],
        "scores": scores.transpose(0, 1).detach().cpu().tolist(),
        "selected_scores": scores.gather(0, chosen.view(1, B)).squeeze(0).detach().cpu().tolist(),
    }
    return out_logits, out_conf, info


@torch.no_grad()
def ply_generate(
    model, prompt_ids, n_tokens, r=4, n_branches=4, noise=0.05, score="conf", temp=0.0, gen=None
):
    """Greedy (temp=0) or sampled decode where every step runs ply_forward.
    Returns (tokens, info) with info['chosen'] listing the winning branch per step.
    The output is ordinary vocab; Ply leaves no trace in the transcript."""
    dev = next(model.parameters()).device
    idx = torch.tensor([list(prompt_ids)], dtype=torch.long, device=dev)
    out, chosen = [], []
    for _ in range(n_tokens):
        logits, _, info = ply_forward(
            model, idx, r=r, n_branches=n_branches, noise=noise, score=score, gen=gen
        )
        ll = logits[:, -1, :]
        if temp and temp > 0:
            t = torch.multinomial(F.softmax(ll / temp, dim=-1), 1, generator=gen)
        else:
            t = ll.argmax(-1, keepdim=True)
        out.append(int(t))
        chosen.append(info["chosen"][0])
        idx = torch.cat([idx, t], dim=1)
    return out, {"chosen": chosen}


@torch.no_grad()
def ply_serve_generate(
    model,
    idx,
    n_new,
    r,
    n_branches=4,
    noise=0.05,
    score="consistency",
    perturb="state",
    temp=0.8,
    top_k=50,
    gen=None,
):
    """Serve adapter mirroring Charkha.generate's (ids, conf) contract: returns
    (full sequence (B, T+n_new), per-generated-token conf (B, n_new)). Plain
    temp/top-k sampling only — ply pays its FLOPs in latent search, and the
    advanced sampling warps (min-p, rep-penalty, contrast) stay with the
    standard path. Uses a full forward per step (no streaming cache)."""
    model.eval()
    confs = []
    for _ in range(n_new):
        logits, conf, _ = ply_forward(
            model,
            idx,
            r=r,
            n_branches=n_branches,
            noise=noise,
            score=score,
            perturb=perturb,
            gen=gen,
        )
        ll = logits[:, -1, :].float()
        if top_k and top_k > 0:
            kth = ll.topk(min(top_k, ll.size(-1)), dim=-1).values[..., -1:]
            ll = ll.masked_fill(ll < kth, float("-inf"))
        if temp and temp > 0:
            t = torch.multinomial(F.softmax(ll / temp, dim=-1), 1, generator=gen)
        else:
            t = ll.argmax(-1, keepdim=True)
        confs.append(conf[:, -1])
        idx = torch.cat([idx, t], dim=1)
    return idx, torch.stack(confs, dim=1)


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------


def _selftest():
    sys.path.insert(0, __file__.replace(chr(92), "/").rsplit("/", 1)[0])
    from charkha import Charkha, CharkhaConfig

    torch.manual_seed(0)
    cfg = CharkhaConfig.toy()
    model = Charkha(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 10))
    checks = 0

    # 1. exactness at N=1: identical to the plain fixed-r forward
    base_logits, base_conf = model(x, r=3)
    p_logits, p_conf, info = ply_forward(model, x, r=3, n_branches=1)
    assert torch.equal(base_logits, p_logits) and torch.equal(base_conf, p_conf)
    assert info["chosen"] == [0, 0] and len(info["scores"]) == x.size(0)
    assert all(len(row) == 1 for row in info["scores"])
    checks += 1

    # 2. determinism: same generator seed -> same result; branches genuinely differ
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    l1, _, i1 = ply_forward(model, x, r=3, n_branches=4, noise=0.05, gen=g1)
    l2, _, i2 = ply_forward(model, x, r=3, n_branches=4, noise=0.05, gen=g2)
    assert torch.equal(l1, l2) and i1["scores"] == i2["scores"]
    flat_scores = [s for row in i1["scores"] for s in row]
    assert len(set(round(s, 6) for s in flat_scores)) > 1, (
        "branch perturbation produced identical trajectories"
    )
    checks += 2

    # 3. selection soundness: every row gets its own max-scoring branch
    g3 = torch.Generator().manual_seed(7)
    lw, cw, iw = ply_forward(model, x, r=3, n_branches=4, noise=0.05, gen=g3)
    for row, choice, selected in zip(iw["scores"], iw["chosen"], iw["selected_scores"]):
        assert row[choice] == max(row)
        assert selected == max(row)
    assert lw.shape == base_logits.shape and cw.shape == base_conf.shape
    checks += 1

    # 4. all score functions run and rank consistently with their definitions
    for how in ("conf", "margin", "entropy"):
        _, _, ii = ply_forward(
            model, x, r=2, n_branches=3, noise=0.05, score=how, gen=torch.Generator().manual_seed(1)
        )
        for row, choice in zip(ii["scores"], ii["chosen"]):
            assert row[choice] == max(row)
    _, _, ti = ply_forward(
        model,
        x,
        r=2,
        n_branches=3,
        noise=0.05,
        score="target",
        target_next=x[:, -1],
        gen=torch.Generator().manual_seed(1),
    )
    assert len(ti["chosen"]) == x.size(0)
    checks += 2

    # 5. zero token footprint: generated ids are ordinary vocab; greedy N=1 decode
    #    matches plain greedy decode exactly
    toks, tinfo = ply_generate(
        model,
        x[0, :6].tolist(),
        n_tokens=5,
        r=2,
        n_branches=3,
        noise=0.05,
        gen=torch.Generator().manual_seed(2),
    )
    assert len(toks) == 5 and all(0 <= t < cfg.vocab_size for t in toks)
    assert len(tinfo["chosen"]) == 5
    plain, _ = ply_generate(model, x[0, :6].tolist(), n_tokens=4, r=2, n_branches=1)
    idx = torch.tensor([x[0, :6].tolist()])
    ref = []
    for _ in range(4):
        lg, _ = model(idx, r=2)
        t = int(lg[0, -1].argmax())
        ref.append(t)
        idx = torch.cat([idx, torch.tensor([[t]])], dim=1)
    assert plain == ref, "N=1 ply_generate diverged from plain greedy decode"
    checks += 2

    # 6. state-perturb mode: branch 0 exact, branches differ, deterministic
    gs = torch.Generator().manual_seed(11)
    ls, cs, si = ply_forward(
        model, x, r=3, n_branches=3, noise=0.05, perturb="state", score="conf", gen=gs
    )
    gs2 = torch.Generator().manual_seed(11)
    ls2, _, si2 = ply_forward(
        model, x, r=3, n_branches=3, noise=0.05, perturb="state", score="conf", gen=gs2
    )
    assert torch.equal(ls, ls2) and si["scores"] == si2["scores"]
    flat = [s for row in si["scores"] for s in row]
    assert len(set(round(s, 6) for s in flat)) > 1, "state noise moved nothing"
    # a row that chose branch 0 must reproduce the plain forward exactly
    for b_row, choice in enumerate(si["chosen"]):
        if choice == 0:
            assert torch.allclose(ls[b_row], base_logits[b_row], atol=0), (
                "state-mode branch 0 not exact"
            )
    checks += 2

    # 7. consistency selector: chosen branch's argmax agrees with the majority
    gc = torch.Generator().manual_seed(13)
    lc, _, ci = ply_forward(
        model, x, r=2, n_branches=4, noise=0.3, perturb="state", score="consistency", gen=gc
    )
    for row, choice in zip(ci["scores"], ci["chosen"]):
        assert row[choice] == max(row)
    checks += 1

    print(f"[selftest] ply.py: all checks passed ({checks} groups)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    ap.print_help()

#!/usr/bin/env python3
"""Validate that torch.compile produces the SAME gradients as eager, now that the fla GDN
kernels are wrapped in torch.compiler.disable (charkha._modules _fla_* wrappers).

Background: inductor used to miscompile fla's hand-written Triton backward — rescheduling ops
in the GDN-2 backward ("operation scheduled before its operands") and corrupting mixer gradients
up to ~30% rel-L2, while the forward stayed bit-exact. The dynamo-disable graph break around each
fla entry point should make --compile gradient-correct again. This script proves it: it runs the
SAME model+batch through eager and compiled, backward on both, and reports the worst per-parameter
gradient rel-L2. It also re-checks eager-vs-eager determinism (must be ~0) so any residual gap is
attributable to compile, not bf16/checkpoint noise.

CUDA-only (torch.compile + the fla Triton path both require a GPU). Run AFTER the mini run frees
the card — this allocates its own model and will compete for VRAM:

    python scripts/check_compile_grads.py            # toy config, fast
    python scripts/check_compile_grads.py --mini     # 128M config, realistic

Exit 0 = pass (compiled grads match eager within tolerance). Exit 1 = the corruption is back.

"""

import argparse
import os
import sys

import torch

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")


def _grads(model, x, y):
    """One fwd+bwd; return {name: grad.clone()} and zero grads afterwards."""
    model.zero_grad(set_to_none=True)
    _logits, loss = model(x, targets=y)
    if not torch.is_tensor(loss):
        raise TypeError(
            f"expected Charkha training forward to return a loss tensor, got {type(loss)!r}"
        )
    loss.backward()
    g = {
        n: p.grad.detach().float().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    model.zero_grad(set_to_none=True)
    return loss.detach().float().item(), g


def _worst_rel(ga, gb):
    """Worst per-tensor rel-L2 ||a-b|| / ||a|| over the shared keys, plus the offending name."""
    worst, where = 0.0, None
    for n in ga:
        if n not in gb:
            continue
        a, b = ga[n], gb[n]
        denom = a.norm().clamp_min(1e-12)
        rel = ((a - b).norm() / denom).item()
        if rel > worst:
            worst, where = rel, n
    return worst, where


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mini", action="store_true", help="use CharkhaConfig.mini (128M) instead of toy"
    )
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument(
        "--tol", type=float, default=2e-2, help="max acceptable compiled-vs-eager grad rel-L2"
    )
    a = ap.parse_args()

    if not torch.cuda.is_available():
        print(
            "[skip] no CUDA — torch.compile + fla Triton path require a GPU. Nothing to validate."
        )
        return 0

    sys.path.insert(0, SRC)
    from charkha import Charkha, CharkhaConfig

    torch.manual_seed(0)
    cfg = CharkhaConfig.mini() if a.mini else CharkhaConfig.toy()
    dev = "cuda"
    model = Charkha(cfg).to(dev)
    model.train()
    x = torch.randint(0, cfg.vocab_size, (a.batch, a.seq), device=dev)
    y = torch.randint(0, cfg.vocab_size, (a.batch, a.seq), device=dev)

    # eager twice: determinism floor (rules out bf16/checkpoint noise as the explanation for any gap)
    l0, g0 = _grads(model, x, y)
    l1, g1 = _grads(model, x, y)
    eager_noise, eager_where = _worst_rel(g0, g1)
    print(
        f"[eager-vs-eager] loss {l0:.5f} / {l1:.5f} | worst grad rel-L2 {eager_noise:.2e} "
        f"({eager_where}) — this is the determinism floor"
    )

    # compiled vs eager
    cmodel = torch.compile(model)
    lc, gc = _grads(cmodel, x, y)
    comp_rel, comp_where = _worst_rel(g0, gc)
    print(
        f"[compiled-vs-eager] loss {l0:.5f} / {lc:.5f} | worst grad rel-L2 {comp_rel:.2e} ({comp_where})"
    )

    ok = comp_rel < a.tol
    print(
        f"\n{'PASS' if ok else 'FAIL'}: compiled grads {'match' if ok else 'DIVERGE from'} eager "
        f"(worst {comp_rel:.2e} vs tol {a.tol:.0e}). "
        f"{'The dynamo-disable fix holds.' if ok else 'The fla backward is STILL being miscompiled.'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

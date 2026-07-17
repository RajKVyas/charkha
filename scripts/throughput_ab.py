#!/usr/bin/env python3
"""Throughput regression proof: isolate the MIXER under the REAL production optimizer config
(8-bit + CPU-offload + symmetry split — exactly what charkha_start.sh launches with), not a naive
full-precision AdamW. Same minimal ~0.42B model (no reasoning modules, fixed r=2, halting off) —
only the mixer differs: GDN-1 (fla chunk_gated_delta_rule) vs GDN-2. As of the gdn2-kernel wiring,
GDN-2 now runs the fused fla.ops.gdn2.chunk_gdn2 Triton kernel on CUDA too (it fell back to a pure-
PyTorch scan before), so this A/B should now show the kernel gap CLOSED. ~60s on a 4060 Ti.

  python scripts/throughput_ab.py            # default seq 2048 batch 1
  python scripts/throughput_ab.py 1024 2     # seq_len batch
  python scripts/throughput_ab.py 2048 1 360M  # also bench a ~360M-sized config (see --size)
"""

import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import torch
from charkha import Charkha, CharkhaConfig, have_fla, _HAVE_FLA_GDN2
from train import build_opt_set

print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()} | fla {have_fla()}")
try:
    import fla.ops as _o

    print("fla.ops surface:", [x for x in dir(_o) if not x.startswith("_")])
except Exception as e:
    print("fla.ops introspection failed:", repr(e))
try:
    import fla.ops.gdn2 as _gdn2

    members = [x for x in dir(_gdn2) if not x.startswith("_")]
    print("fla.ops.gdn2 members:", members)
    for name in members:
        obj = getattr(_gdn2, name)
        if callable(obj):
            print(
                f"  {name}: {getattr(obj, '__doc__', None) and obj.__doc__.strip().splitlines()[0]}"
            )
except Exception as e:
    print("fla.ops.gdn2 introspection failed:", repr(e))

T = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
B = int(sys.argv[2]) if len(sys.argv) > 2 else 1
SIZE = sys.argv[3] if len(sys.argv) > 3 else None  # e.g. "360M" -> also bench a ~360M dims config

OPT_ARGS = SimpleNamespace(
    muon_lr=0.02, adam_lr=3e-3, norm_lr=0.02, grad_clip=1.0, symmetry_opt=True, eightbit_optim=True
)


def size_360m_cfg():
    # d_model 1152 (head_dim 64 -> 18 heads, GQA 3:1), d_ff 2.72x, same depth as default.
    # n_kv_heads MUST divide n_heads (GQA expand rep = n_heads // n_kv_heads); 18/4 = 4.5 was the
    # earlier "tensor a (18) must match b (16)" failure, so use 6 (18/6 = 3). ~362M params,
    # SmolLM2-360M-comparable scale.
    c = CharkhaConfig()
    c.d_model = 1152
    c.n_heads = 18
    c.n_kv_heads = 6
    c.d_ff = 3136
    return c


def bench(use_gdn2, size=None, steps=8, offload=True):
    cfg = size_360m_cfg() if size == "360M" else CharkhaConfig()  # ~0.42B default dims otherwise
    cfg.use_gdn2 = use_gdn2
    cfg.grad_checkpoint = True
    cfg.use_halting = False  # isolate the mixer: no halting, fixed shallow recurrence,
    cfg.mean_recurrence = 2  # NO reasoning modules — everything but the mixer is identical
    cfg.max_recurrence_train = 2
    cfg.backprop_depth = 2
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = Charkha(cfg).to("cuda")
    m.train()
    n_params = sum(p.numel() for p in m.parameters())
    # REAL production optimizer set: Muon/NormM/AdamW split, 8-bit Adam, CPU-offloaded momentum —
    # matches charkha_start.sh's --offload-optim --8bit-optim --symmetry-opt, NOT a naive full-fp32
    # AdamW-over-everything (which alone needs ~4x params bytes of optimizer state and was the
    # actual cause of the earlier 9.3G/spill reading — a benchmark bug, not an architecture one).
    opts, _bases, _mode = build_opt_set(m, OPT_ARGS, offload=offload)

    def step():
        x = torch.randint(0, cfg.vocab_size, (B, T), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            _, loss = m(x, x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), OPT_ARGS.grad_clip)
        for o in opts:
            o.step()
        for o in opts:
            o.zero_grad(set_to_none=True)

    step()
    step()
    torch.cuda.synchronize()  # warmup: lazy optimizer state alloc
    t0 = time.time()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    tok_s = steps * B * T / (time.time() - t0)
    peak = torch.cuda.max_memory_reserved() / 1024**3
    free, total = torch.cuda.mem_get_info()
    ded = (total - free) / 1024**3
    m = None
    opts = None
    torch.cuda.empty_cache()
    return n_params, tok_s, peak, ded


def main():
    configs = [(False, None), (True, None)] + ([(False, "360M"), (True, "360M")] if SIZE else [])
    print(
        f"\nmixer A/B @ seq={T} batch={B}, OFFLOAD+8BIT+SYMMETRY optimizer (matches the real launcher):"
    )

    for g2, sz in configs:
        gdn2_label = "GDN-2 (fla fused kernel)" if _HAVE_FLA_GDN2 else "GDN-2 (pure-pytorch scan)"
        name = (gdn2_label if g2 else "GDN-1 (fla kernel)") + (
            f" [{sz}]" if sz else " [default ~0.42B]"
        )
        try:
            n, ts, pk, ded = bench(g2, size=sz)
            print(
                f"  {name:42} {n / 1e6:6.1f}M  {ts:>9,.0f} tok/s   reserved={pk:.2f}G  ded.used={ded:.2f}G"
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"  {name:42} OOM: {str(e).splitlines()[0][:70]}")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {name:42} FAIL: {type(e).__name__}: {str(e).splitlines()[0][:70]}")
    if _HAVE_FLA_GDN2:
        print(
            "\nGDN-2 is now on the fused fla.ops.gdn2.chunk_gdn2 kernel (chunk size 64, Triton). If its "
            "tok/s is now within ~1.1-1.3x of GDN-1, the earlier regression (pure-PyTorch scan) is closed "
            "and GDN-2 can be the default mixer for the local pilot. If it is still multiples slower, fall "
            "back to GDN-1+fla and keep GDN-2 as an ablation."
        )
    else:
        print(
            "\nfla.ops.gdn2 not importable here -> GDN-2 ran the pure-PyTorch scan. If GDN-1 is multiples "
            "faster, the kernel gap IS the regression; install/upgrade fla to get chunk_gdn2."
        )


if __name__ == "__main__":
    main()

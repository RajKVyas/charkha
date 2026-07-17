#!/usr/bin/env python
"""
CHARKHA demos — the long, watchable showcases (split out of preflight.py).
============================================================================
preflight.py answers "is the machinery correct, will a real run survive?" — fast, gating.
THIS file answers "look what it can do" — slower, narrative, NOT gating. The showcase:

  --integrated   one toy run that ramps the recurrence curriculum AND flips the halter
                 Phase 1 -> Phase 2, sampling the SAME prompt throughout so you watch
                 generation sharpen from noise -> the sentence, then the effort dial.

(An addition self-learning demo was removed — a d=320 toy model can't reliably do arithmetic. Real capability is
`pipeline.py --eval` / `--elasticity` on a trained checkpoint.)

"""

import argparse
import os
import tempfile
import time
import sys

try:  # we print decoded model bytes; don't die on win32 cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

HERE = os.path.dirname(os.path.abspath(__file__))


def hr(t):
    print("\n" + "=" * 74 + f"\n  {t}\n" + "=" * 74)


def mark(b):
    return "PASS" if b else "FAIL"


def _decode(ids):
    return bytes(b & 0xFF for b in ids).decode("utf-8", errors="replace")


def run_integrated_demo(device=None):
    """The richer, longer generation-sharpening narrative. One toy model, ~320 steps, with the
    recurrence curriculum ramping E[r] 1->4 and the halter flipping Phase 1 (fixed loops) ->
    Phase 2 (learned halting). The SAME prompt is sampled throughout so you watch noise become
    the sentence; then the effort dial (CE at r=1/2/4) shows deeper recurrence buying lower loss.
    Returns a checks dict (non-gating)."""
    import torch
    import torch.nn.functional as F
    from charkha import Charkha, CharkhaConfig
    from train import make_synthetic_shards, train, _toy_args, load_ckpt

    checks = {}
    tmp = tempfile.mkdtemp(prefix="charkha_demo_")
    data_dir, out_dir = os.path.join(tmp, "data"), os.path.join(tmp, "run")
    make_synthetic_shards(data_dir, n_shards=4, toks_per=9000)

    hr("INTEGRATED DEMO — watch generation sharpen as the curriculum + halter phases fire")
    print("  corpus: a repeating sentence (byte-level, learnable in ~150 steps).")
    print('  the SAME prompt "the " is sampled every 25 steps. The step tag flips')
    print("  r..f (FIXED Poisson loops) -> r..h (HALTER on) at step 160, and the")
    print("  recurrence curriculum ramps E[r] 1 -> 4. │ marks prompt | generation.\n")
    try:  # untrained baseline so the noise->text arc is visible
        bcfg = CharkhaConfig.toy()
        bcfg.vocab_size = 384
        bmod = Charkha(bcfg).eval()
        with torch.no_grad():
            out = bmod.generate(
                torch.tensor([list(b"the ")], dtype=torch.long), 28, temp=0.8, top_k=50
            )
        print(f"  [sample @ untrained] {('the │' + _decode(out[0, 4:].tolist()))!r}")
    except Exception as e:
        print(f"  [baseline sample skipped: {type(e).__name__}: {e}]")

    train(
        _toy_args(
            steps=320,
            out=out_dir,
            data=data_dir,
            val_frac=0.2,
            recurrence_curriculum=True,
            curric_r_start=1,
            curric_r_end=4,
            curric_steps=200,
            halt_start_step=160,
            ponder_anneal_steps=60,
            sample_every=25,
            sample_tokens=28,
            sample_prompt="the ",
            eval_every=80,
            eval_iters=4,
            log_every=80,
            snapshot_every=160,
        )
    )
    checks["integrated demo completes + checkpoints"] = os.path.exists(
        os.path.join(out_dir, "ckpt.pt")
    )

    model, cfg, *_ = load_ckpt(os.path.join(out_dir, "ckpt.pt"), "cpu")
    model.eval()
    enc = lambda s: list(s.encode("utf-8"))
    xb = torch.tensor(
        [enc("the people build their own tools and learn the shape of ")], dtype=torch.long
    )
    print(
        "\n  EFFORT DIAL — same prompt, deeper recurrence costs more compute (lower CE = sharper):"
    )
    ces = {}
    with torch.no_grad():
        for r in (1, 2, 4):
            lg, _ = model(xb, r=r)
            ces[r] = float(
                F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)), xb[:, 1:].reshape(-1))
            )
            print(
                f"    r={r} (effort {'min' if r == 1 else 'max' if r == 4 else 'mid'}): CE={ces[r]:.3f}"
            )
    checks["effort dial produces finite CE at r=1,2,4"] = all(c == c for c in ces.values())
    with torch.no_grad():
        out = model.generate(
            torch.tensor([list(b"the ")], dtype=torch.long), 40, temp=0.7, top_k=40
        )
    print(f"  [sample @ trained]   {('the │' + _decode(out[0, 4:].tolist()))!r}")
    print("\n  The arc: untrained = noise -> trained = the sentence reconstructed. The effort dial")
    print(
        "  runs at r=1/2/4 on the SAME weights; a memorized toy corpus is too easy to separate the"
    )
    print(
        "  depths cleanly here — the real accuracy-vs-FLOPs curve is pipeline.py --elasticity on a ckpt."
    )
    return checks


def ensure_hf_ckpt(local_path="hf_ckpt.pt"):
    import os
    import subprocess
    import shutil

    if os.path.exists(local_path):
        return local_path
    print("\n  [Downloading real trained checkpoint from CHARKHA_ORG/charkha-ckpt...]")
    print("  (This might take a few minutes if it's not cached locally.)")
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


def run_trained_capability_demo(ckpt_path):
    import torch
    import torch.nn.functional as F
    from train import load_ckpt
    from tokenizers import Tokenizer

    hr("TRAINED CAPABILITY DEMO — showing off the real HF checkpoint")
    checks = {}
    print(f"  Loading {ckpt_path} ...")
    try:
        model, cfg, *_ = load_ckpt(ckpt_path, "cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        # Load the tokenizer recorded by the checkpoint, falling back to the included tokenizer.
        try:
            tokenizer_name = getattr(cfg, "tokenizer_name", None)
            if tokenizer_name and os.path.isfile(tokenizer_name):
                tok = Tokenizer.from_file(tokenizer_name)
            else:
                repo_tok = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), "charkha_tokenizer.json"
                )
                tok = Tokenizer.from_file(repo_tok)
            encode = lambda s: tok.encode(s).ids
            decode = lambda ids: tok.decode(ids, skip_special_tokens=False)
        except Exception:
            # Fallback to byte-level if tokenizers library fails
            encode = lambda s: list(s.encode("utf-8"))
            decode = lambda ids: bytes(b & 0xFF for b in ids).decode("utf-8", errors="replace")

        prompts = [
            "Write a poem about the ocean.",
            "user: What is 5 plus 3?\nassistant:",
            "Think step by step: how would you",
        ]

        print("\n  1) High-Quality Generation:")
        device = next(model.parameters()).device
        for p in prompts:
            xb = torch.tensor([encode(p)], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model.generate(xb, 60, temp=0.7, top_k=50, effort=None)
            res = decode(out[0, xb.shape[1] :].tolist())
            print(f"    prompt: {p!r}")
            print(f"    gen:    {res!r}\n")

        print("  2) Effort Dial (Compute Elasticity) on trained weights:")
        xb = torch.tensor(
            [encode("The quick brown fox jumps over the lazy dog.")],
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            for r in (1, 2, 4):
                lg, _ = model(xb, r=r)
                ce = float(
                    F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)), xb[:, 1:].reshape(-1))
                )
                print(f"    r={r}: CE={ce:.3f} (Lower = sharper representations)")
        checks["trained demo completed"] = True
    except Exception as e:
        print(f"  [FAIL] Trained capability demo crashed: {e}")
        checks["trained demo completed"] = False
    return checks


def main():
    ap = argparse.ArgumentParser(description="CHARKHA demos — the long, watchable showcases.")
    ap.add_argument(
        "--integrated",
        action="store_true",
        help="generation-sharpening curriculum demo (the default; currently the only demo)",
    )
    ap.add_argument(
        "--hf-check",
        action="store_true",
        help="download and run the optional CHARKHA_ORG/charkha-ckpt checkpoint demo",
    )
    a = ap.parse_args()
    t0 = time.time()
    checks = {}

    if a.integrated or True:
        checks.update(run_integrated_demo())

    if a.hf_check:
        ckpt = ensure_hf_ckpt()
        if ckpt:
            checks.update(run_trained_capability_demo(ckpt))

    hr("DEMOS COMPLETE")
    for n, p in checks.items():
        print(f"  [{mark(p)}] {n}")
    dt = time.time() - t0
    npass = sum(bool(v) for v in checks.values())
    print(f"\n  {npass}/{len(checks)} demo checks passed in {dt:.0f}s.")
    print(
        "  (Demos are showcases, not launch gates — real capability is pipeline.py --eval on a ckpt.)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

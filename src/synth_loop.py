"""CHARKHA 24/7 synthetic data generator — imports seed bank from synth_seeds.py.
Rotating 50M-token shards. Prints one sample every 25 generations."""

from __future__ import annotations
import argparse
import os
import random
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from synth_seeds import SEED_BANK


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="teacher model ID or local path")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--shard-tokens", type=int, default=50_000_000)
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    weights = [w for w, _, _ in SEED_BANK]

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    print(f"Loading {args.model} (4bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16
    ).eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    print(
        f"VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB  |  {len(SEED_BANK)} seeds  |  {args.shard_tokens / 1e6:.0f}M tok/shard\n"
    )

    shard_idx = 0
    while True:
        path = os.path.join(args.out, f"shard_{shard_idx:04d}.txt")
        print(f"[shard {shard_idx}] {path}  ", end="", flush=True)
        total, samples, shown = 0, 0, 0
        f = open(path, "w", encoding="utf-8")
        f.write(f"# CHARKHA synthetic — {args.model}\n# Shard {shard_idx}\n\n")
        t0 = time.time()

        while total < args.shard_tokens:
            try:
                # Weighted random selection from entire seed bank
                _, tmpl, opts = random.choices(SEED_BANK, weights=weights, k=1)[0]
                # Pick random option for each key
                fmt = {}
                for k, choices in opts.items():
                    fmt[k] = random.choice(choices)
                prompt = tmpl.format(**fmt)

                inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(
                    args.device
                )
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new,
                        temperature=args.temp,
                        do_sample=True,
                        top_p=0.95,
                        pad_token_id=tok.eos_token_id or 0,
                    )
                full = tok.decode(out[0], skip_special_tokens=True)
                response = full[len(prompt) :] if full.startswith(prompt) else full
                if len(response) < 15:
                    continue
                f.write(full + "\n\n")
                nt = out.shape[1] - inputs["input_ids"].shape[1]
                total += nt
                samples += 1

                # Show one sample every 25 generations
                if samples % 25 == 0:
                    shown += 1
                    summary = response[:120].replace("\n", "↵")
                    print(f"\n  ── sample {samples} ({total / 1e6:.1f}M tok) ──")
                    print(f"  prompt: {prompt[:100]}...")
                    print(f"  gen:    {summary}...")
                    print(f"  [{shard_idx}] ", end="", flush=True)

            except Exception:
                traceback.print_exc()
                time.sleep(5)

        f.close()
        dt = time.time() - t0
        tok_s = total / max(dt, 1)
        print(
            f"\n  ✓ {samples} samples, {total / 1e6:.1f}M tokens, {dt / 60:.1f}min ({tok_s:.0f} tok/s)\n"
        )
        shard_idx += 1


if __name__ == "__main__":
    main()

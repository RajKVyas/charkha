#!/usr/bin/env python3
"""CHARKHA pipeline CLI — evaluation, elasticity, synthetic data generation."""

import argparse
import os
import sys
import torch
from pipeline import (
    _selftest,
    generate_synthetic,
    generate_synthetic_api,
    kd_pipeline,
    run_elasticity,
    run_eval,
    run_risk_coverage,
)


def main():
    p = argparse.ArgumentParser(description="CHARKHA pipeline - eval + synthetic")
    p.add_argument("--eval", action="store_true", help="run lm-eval benchmarks")
    p.add_argument("--ckpt", type=str, default="out/ckpt.pt", help="checkpoint path")
    p.add_argument(
        "--tasks", type=str, default="hellaswag,arc_easy", help="comma-separated lm-eval tasks"
    )
    p.add_argument("--limit", type=int, default=None, help="max eval examples")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--risk-coverage",
        action="store_true",
        help="calibrated-abstention curve + conformal threshold on a val shard",
    )
    p.add_argument("--shard", type=str, default=None, help="tokenized uint16 val shard")
    p.add_argument("--max-tokens", type=int, default=200_000, help="predictions to score")
    p.add_argument("--target-risk", type=float, default=0.10, help="conformal risk target")
    p.add_argument("--delta", type=float, default=0.05, help="conformal confidence 1-delta")
    p.add_argument(
        "--effort", type=int, default=None, help="fixed recurrence loops (None=adaptive)"
    )
    p.add_argument(
        "--elasticity",
        action="store_true",
        help="E1: accuracy/NLL vs inference compute (loop count) on a val shard",
    )
    p.add_argument(
        "--efforts",
        type=str,
        default="1,2,4,8,16",
        help="comma-separated loop counts for --elasticity",
    )
    p.add_argument("--selftest", action="store_true", help="unit-test selective-prediction math")
    p.add_argument("--synth", action="store_true", help="generate synthetic data")
    p.add_argument(
        "--teacher",
        type=str,
        default=None,
        help="local Hugging Face teacher model ID; review its license and usage terms",
    )
    p.add_argument("--tokens", type=int, default=100_000, help="target tokens to generate")
    p.add_argument("--out", type=str, default="synthetic.txt", help="output file")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--temperature", type=float, default=0.8)
    # Frontier teacher via OpenAI-compatible API (DeepSeek-V4/GPT/Claude/Gemini through LiteLLM).
    # When --api-base is set, --synth uses sequence-level KD over HTTP instead of a local HF model.
    p.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="OpenAI-compatible base URL (e.g. http://localhost:4000/v1) → frontier seq-level KD",
    )
    p.add_argument(
        "--api-model",
        type=str,
        default=None,
        help="model name to request from the API teacher",
    )
    p.add_argument(
        "--api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="env var holding the API key (default OPENAI_API_KEY)",
    )
    p.add_argument("--api-max-new", type=int, default=1024, help="max_tokens per API completion")
    p.add_argument(
        "--kd-run",
        action="store_true",
        help="turnkey: generate teacher traces -> tokenize -> uint16 shards (train-ready)",
    )
    p.add_argument("--kd-out", type=str, default="data_kd", help="--kd-run output shard dir")
    a = p.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    elif a.risk_coverage:
        if not a.shard:
            p.error("--risk-coverage requires --shard <tokenized uint16 val shard>")
        run_risk_coverage(a.ckpt, a.shard, device, a.max_tokens, a.target_risk, a.delta, a.effort)
    elif a.elasticity:
        if not a.shard:
            p.error("--elasticity requires --shard <tokenized uint16 val shard>")
        run_elasticity(
            a.ckpt,
            a.shard,
            device,
            a.max_tokens,
            efforts=tuple(int(e) for e in a.efforts.split(",")),
        )
    elif a.eval:
        run_eval(a.ckpt, a.tasks.split(","), device, a.batch_size, a.limit)
    elif a.kd_run:
        if not a.api_base and not a.teacher:
            p.error("--kd-run requires --teacher or --api-base with --api-model")
        kd_pipeline(
            a.kd_out,
            a.tokens,
            api_base=a.api_base,
            api_model=a.api_model,
            api_key=os.environ.get(a.api_key_env, ""),
            teacher=a.teacher,
            device=device,
            temperature=a.temperature,
        )
    elif a.synth:
        if a.api_base:
            if not a.api_model:
                p.error("--api-base requires --api-model")
            generate_synthetic_api(
                a.api_base,
                a.api_model,
                a.out,
                a.tokens,
                api_key=os.environ.get(a.api_key_env, ""),
                temperature=a.temperature,
                max_new=a.api_max_new,
            )
        else:
            if not a.teacher:
                p.error("--synth requires --teacher or --api-base with --api-model")
            generate_synthetic(a.teacher, a.out, a.tokens, device, a.temperature)
    else:
        p.print_help()


if __name__ == "__main__":
    main()

# CHARKHA

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/RajKVyas/charkha/actions/workflows/ci.yml/badge.svg)](https://github.com/RajKVyas/charkha/actions/workflows/ci.yml)

A ~1B-parameter depth-recurrent language model that trains and serves on a
single consumer GPU (8 GB VRAM). Research code — not production software.

> **New here?** Follow the [Tutorial](docs/TUTORIAL.md) or browse the [file index](SUMMARY.md).
## What's here

A full training stack for a small, fast language model:

- **Model** (`src/charkha/`): GDN-2 + attention hybrid with depth-recurrent
  core. Loops the same transformer blocks multiple times per token, trading
  compute for depth at inference via an effort dial (1–16 loops). Default
  config at d_model=2048 produces ~994M parameters.
- **Tokenizer** (`charkha_tokenizer.json`): Included 65,535-token ByteLevel BPE tokenizer
  ([provenance](docs/TOKENIZER_PROVENANCE.md)).
- **Training** (`src/train.py`): Muon/NormM optimizers with 8-bit state and
  CPU offload, chunked fused cross-entropy, gradient checkpointing, truncated
  BPTT through recurrence. Crash-survivable checkpoints.
- **Data** (`src/dataprep.py`): Manifest-driven streaming pipeline for user-selected datasets
  through quality filtering, deduplication, PII scrubbing, decontamination,
  and tokenization into uint16/uint32 shards.

## Quickstart

```bash
git clone https://github.com/RajKVyas/charkha.git
cd charkha
bash scripts/setup_wsl.sh
source .venv/bin/activate

# Verify the model's CPU path
python -m charkha._toy --toy

# Full test suite
python tests/run_tests.py
```

Training on 8 GB GPU:

```bash
python src/train_cli.py --profile frontier --data data/mycorpus-dd --out runs/m1 \
  --steps 50000 --seq-len 512 --batch-size 1 --accum-steps 8 \
  --ce-chunk 1024 --gdn-chunk 32 --grad-checkpoint \
  --offload-optim --8bit-optim --symmetry-opt --embed-factor 256
```

Serving:

```bash
python src/serve.py --ckpt runs/m1/ckpt.pt
```

## Current status

**Implemented and covered by local regression tests** (default `--profile frontier` stack):
GDN-2 mixer, NITP, deep supervision, OSDN key preconditioning, MTP routing,
bipolar sign-gating, cross-loop consistency, per-sequence recurrence,
thermostat, convergence tracking, acceleration early-exit, SNGP uncertainty,
Laplace-Redux, task RL (AWR).

**Staged but inactive** (exact no-ops at init, for future grow points): loop
adapters, latent memory spine, subconscious scratchpad, product-key granary
memory layer.


No pretrained weights or public benchmark results are included. Full-scale training and serving
remain experimental; the serving path has partial test coverage and is exercised at toy scale.

## Structure

```
src/
  charkha/              Model package
    ├── _model.py          Charkha class
    ├── config.py          CharkhaConfig dataclass
    ├── _modules.py        GDN, attention, MLP, normalisation
    ├── _optim.py          Muon, NormM, LR schedules
    ├── _loss.py           Chunked fused CE helpers
    └── _toy.py            Hermetic CPU selftest + CLI
  train.py              Training loop, checkpointing
  train_cli.py          Training CLI entry point
  data.py               Streaming shard loader
  dataprep.py           Streaming data pipeline
  serve.py              Interactive serving: retrieval, tools, calibration
  _verify.py            Proof contracts, arithmetic verification
  _memory.py            SQLite conversation store
  pipeline.py           Evaluation and elasticity curves
  pipeline_cli.py       Pipeline CLI entry point
  preflight.py          Launch gate: env check + component tests
  _benchmarks.py        CPU/GPU throughput benchmarks
  _probes.py            Feature probes
  sft.py                Supervised instruction fine-tuning
  continual.py          Replay-mixed continual training
  distill.py            Logit-level knowledge distillation
  selfteach.py          Self-teaching with verifier/reward loops
  verify_shards.py      Data integrity checker
scripts/
  train_tokenizer.py    BPE tokenizer training
  train_resilient.sh    Auto-resume training wrapper
  setup_wsl.sh          WSL2 environment setup
configs/
  sources.example.yaml  Generic source-manifest template
tests/
  run_tests.py          Fast hermetic regression suite
docs/
  ARCHITECTURE.md       Package layout + dependency graph + API reference
```

## Prerequisites

- Python 3.10+
- PyTorch 2.x with CUDA (optional; CPU path works for development)
- For the production GDN kernel: `flash-linear-attention` + Triton (Linux only)
- See `requirements.txt` for the full dependency lock

## Hardware

- Development and testing: any CPU (all selftests are hermetic)
- Training at the documented full config: 8 GB VRAM was used during development; memory use
  depends on the installed kernels, sequence length, and optimizer settings
- Larger configurations require more memory; run `src/preflight.py` before a long job
- Cloud GPUs accelerate the same checkpoint lineage

## FAQ

**Does it need a GPU?** No — CPU selftests pass. Training at full config needs 8 GB VRAM.

**Can I use my own data?** Yes — write a `configs/sources.yaml` manifest and run `dataprep.py`.

**What's the minimum VRAM?** There is no universal minimum. Use `--nano` for smoke tests and run
the preflight checks with the exact configuration you plan to train.

**Does it work on Mac?** CPU path works. No MPS/CoreML support.

**Where are pretrained weights?** None yet — this is a training codebase.

**How do I contribute?** See [CONTRIBUTING.md](CONTRIBUTING.md) and [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

MIT — see [LICENSE](LICENSE). Third-party attributions in [NOTICE.md](NOTICE.md).

## Citation

If you use CHARKHA in your research, see [CITATION.cff](CITATION.cff).

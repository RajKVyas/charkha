# CHARKHA Architecture

## Package layout

```
src/
├── charkha/              # Model package
│   ├── __init__.py       # Public API: Charkha, CharkhaConfig, build_optimizers, etc.
│   ├── config.py         # CharkhaConfig dataclass + static constructors (toy, nano, small, etc.)
│   ├── _model.py         # Charkha class — forward pass, generation, halting, loss
│   ├── _modules.py       # NN layers: GDN, attention, MLP, normalisation, fla wrappers
│   ├── _loss.py          # Chunked fused cross-entropy helpers
│   ├── _optim.py         # Muon, NormM, build_optimizers, LR schedules, gradient utilities
│   └── _toy.py           # Hermetic CPU selftest + CLI entry point (`python -m charkha._toy --toy`)
│
├── train.py              # Training loop, checkpointing, evaluation
├── train_cli.py          # Training CLI (argparse + main)
├── data.py               # ShardLoader — streaming uint16/uint32 token shard reader
├── serve.py              # Interactive serving: Session, retrieval, tool calls, calibration
├── _memory.py            # Memory — SQLite-backed conversation store
├── _verify.py            # Proof contracts, arithmetic verification, tool resolution
├── _benchmarks.py        # CPU and GPU training throughput benchmarks
├── _probes.py            # Feature probes: confidence, effort dial, convergence
├── preflight.py          # Launch gate: environment check + component selftests + CLI
├── dataprep.py           # Streaming data pipeline: filter → dedup → PII → decontam → tokenize
├── pipeline.py           # Evaluation, elasticity curves, synthetic data generation
├── distill.py            # Logit-level knowledge distillation
├── selfteach.py          # Self-teaching with verifier/reward loops
├── continual.py          # Replay-mixed continual training
├── sft.py                # Supervised instruction fine-tuning
├── consolidate.py        # Sleep-time personal data consolidation
├── fold.py               # Latent context compression via gist vectors
├── granary.py            # Product-key memory layer (staged)
├── latent_memory.py      # Intrinsic latent memory spine (staged)
├── murmur.py             # Private reasoning register
├── ply.py                # Latent branch-and-select trajectory search
├── proofcarry.py         # Proof-carrying answers: claim ledger + arithmetic check
├── retrieval.py          # BM25 retrieval store with passage encoding
├── tasks.py              # Synthetic task generators for continual RL training
├── worldmodel.py         # Structured belief extraction from natural language
├── bakeoff.py            # Ensemble model comparison and distillation bake-off
├── curriculum.py         # Self-curating curriculum loop
├── crossdedup.py         # Cross-source shard deduplication
├── demo.py               # Long-form generation sharpening + reasoning module showcase
├── merge.py              # Model merging (SLERP, TIES, DARE)
├── verify_shards.py      # Data integrity checker for token shard directories
└── ...
```

## Key dependency graph

```
train.py ──────► charkha (model, optims)
                 data (ShardLoader)
                 dataprep (shard format)

serve.py ──────► charkha (model)
                 _memory (conversation store)
                 _verify (proof contracts, tools)
                 retrieval (BM25)

preflight.py ──► charkha (model)
                 _benchmarks (throughput)
                 _probes (feature tests)
                 train (ShardLoader)

pipeline.py ───► charkha (model)
                 dataprep (shard format)
```

## Public API surface

The `charkha` package exports a small, stable API:

| Symbol | Module | Purpose |
|---|---|---|
| `CharkhaConfig` | `config.py` | Model hyperparameters |
| `Charkha` | `_model.py` | Model class |
| `build_optimizers` | `_optim.py` | Muon + AdamW optimizer pair |
| `build_symmetry_optimizers` | `_optim.py` | Muon + NormM symmetry set |
| `wsd_lr_mult` | `_optim.py` | Warmup-stable-decay LR multiplier |
| `Muon`, `NormM` | `_optim.py` | Optimizer classes |
| `install_grad_release` | `_optim.py` | CPU gradient offload |
| `clip_grads_mixed` | `_optim.py` | Mixed-precision gradient clipping |
| `have_fla` | `_modules.py` | Whether fla Triton kernel is available |

## Stability notes

**Default experimental profile** (`--profile frontier`):
GDN-2, NITP, deep supervision, OSDN, MTP routing, bipolar gate,
cross-loop consistency, per-seq recurrence, thermostat, convergence,
acceleration early-exit, SNGP, Laplace-Redux, task RL.

**Staged (exact no-ops at init, for future grow points):**
Loop adapters, latent memory, subconscious, granary.

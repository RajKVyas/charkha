# CHARKHA — File Index

> **Docs:** [README](README.md) · [Tutorial](docs/TUTORIAL.md) · [Architecture](docs/ARCHITECTURE.md) · [Changelog](CHANGELOG.md)


## Core model (`src/charkha/`)

| File | Description |
|---|---|
| `config.py` | `CharkhaConfig` dataclass — all hyperparameters |
| `_model.py` | `Charkha` class — forward pass, generation, halting, training |
| `_modules.py` | NN layers — GDN, attention, MLP, RMSNorm, embedding |
| `_optim.py` | Muon, NormM, build_optimizers, LR schedules, gradient utilities |
| `_loss.py` | Chunked fused cross-entropy helpers |
| `_toy.py` | Hermetic CPU selftest — builds, trains, samples a tiny model |
| `__init__.py` | Public API re-exports |
| `cli.py` | Unified CLI — `charkha toy\|train\|serve\|preflight\|pipeline\|test` |

## Training

| File | Description |
|---|---|
| `train.py` | Training loop — checkpointing, evaluation, config profiles |
| `train_cli.py` | Training CLI entry point — argparse + main() |
| `data.py` | `ShardLoader` — streaming uint16/uint32 token shard reader |
| `_mem_probe.py` | VRAM memory probe — config validation and spill detection |
| `dataprep.py` | Streaming data pipeline — filter, dedup, PII, decontam, tokenize |
| `_dedup.py` | MinHash LSH + exact/near-deduplication |
| `verify_shards.py` | Data integrity checker for token shard directories |
| `crossdedup.py` | Cross-source shard deduplication |

## Serving

| File | Description |
|---|---|
| `serve.py` | Interactive serving — Session, retrieval, tools, calibration |
| `_verify.py` | Proof contracts, arithmetic verification, tool resolution |
| `_memory.py` | SQLite-backed conversation memory store |
| `_serve_utils.py` | Retrieval ingestion, checkpoint loading, datastore builder |
| `retrieval.py` | BM25 retrieval store with passage encoding |
| `murmur.py` | Private reasoning register — masked vocabulary band |
| `murmur_micro.py` | Micro-scale murmur capability probe |
| `ply.py` | Latent branch-and-select trajectory search |
| `ply_micro.py` | Micro-scale Ply capability probe |
| `proofcarry.py` | Proof-carrying answers — claim ledger + arithmetic check |
| `worldmodel.py` | Structured belief extraction from natural language |
| `personal_eval.py` | Private prompt benchmark harness |

## Post-training

| File | Description |
|---|---|
| `sft.py` | Supervised instruction fine-tuning |
| `continual.py` | Replay-mixed continual training with task RL |
| `distill.py` | Logit-level knowledge distillation |
| `selfteach.py` | Self-teaching with verifier/reward loops |
| `consolidate.py` | Sleep-time personal data consolidation |
| `fold.py` | Latent context compression via gist vectors |
| `curriculum.py` | Self-curating curriculum loop |
| `tasks.py` | Synthetic task generators for continual RL |
| `synth_loop.py` | Synthetic data generation loop |
| `synth_seeds.py` | Synthetic data seed bank — 3257 topic entries |
| `bakeoff.py` | Ensemble model comparison and distillation bake-off |
| `merge.py` | Model merging — SLERP, TIES, DARE |

## Staged modules

| File | Description |
|---|---|
| `granary.py` | Product-key memory layer — knowledge params in host RAM |
| `granary_micro.py` | Micro-scale granary capacity probe |
| `latent_memory.py` | Intrinsic latent memory spine |
| `micro_task.py` | Synthetic task definitions for micro-scale experiments |
| `accordion_micro.py` | Accordion invariant rehearsal — cached teacher KD |
| `vdepth_micro.py` | Virtual-depth slab alignment probe |

## Evaluation and debugging

| File | Description |
|---|---|
| `pipeline.py` | Evaluation, elasticity curves, synthetic data generation |
| `pipeline_cli.py` | Pipeline CLI entry point |
| `preflight.py` | Launch gate — environment check + component selftests |
| `_benchmarks.py` | CPU and GPU training throughput benchmarks |
| `_probes.py` | Feature probes — confidence, effort dial, convergence |
| `demo.py` | Long-form generation example |
| `event_jepa.py` | Event-based JEPA module |

## Scripts (`scripts/`)

| File | Description |
|---|---|
| `train_tokenizer.py` | ByteLevel BPE tokenizer training |
| `proof_bundle.py` | CPU proof bundle — syntax check + all selftests |
| `grow_init.py` | Width-growth init from a trained mini model |
| `cloud_probe.py` | Cloud GPU probe — launch config validation |
| `ablate.py` | Feature ablation CLI |
| `board.py` | File-based kanban board (CHK- card system) |
| `check_compile_grads.py` | Verify torch.compile gradient correctness |
| `cache_logits.py` | Pre-compute teacher top-k logits for offline KD |
| `card_embeddings.py` | Transplant teacher embedding geometry |
| `loom_merge.py` | DiLoCo-style same-lineage checkpoint relay |
| `throughput_ab.py` | Mixer A/B throughput regression proof |
| `news_ingest.py` | RSS/Atom feed ingestion pipeline |
| `curate_jsonl.py` | JSONL curator by model loss band |
| `recover_index.py` | Rebuild missing index.json for unindexed shard dirs |
| `dependency_audit.py` | Package dependency integrity check |
| `postprocess_tokenizer.py` | Post-process trained tokenizer for serving |
| `prep_textfile.py` | Tokenize a raw text file into shards |
| `scatter_progress.py` | Scatter plot of training metrics |
| `charkha_dash.py` | Live training dashboard |
| `build_re_dataset.py` | Reverse engineering dataset builder |

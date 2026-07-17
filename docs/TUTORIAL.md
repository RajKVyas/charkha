# CHARKHA Tutorial

A step-by-step walkthrough from zero to a trained model.

## 1. Clone and install

```bash
git clone https://github.com/RajKVyas/charkha.git
cd charkha
bash scripts/setup_wsl.sh
source .venv/bin/activate
pip install -e .
```

## 2. Verify your setup

```bash
charkha toy --toy
```

This runs a hermetic CPU selftest — builds a tiny model, trains it for a few steps,
generates samples, and checks the model's CPU execution path. Runtime depends on the machine.

If this fails, your environment isn't ready. Run `python src/preflight.py` for a full
diagnostic.

## 3. Prepare training data

CHARKHA trains on token shards produced by `dataprep.py`. You choose the input datasets in a
local manifest; the repository does not prescribe a training corpus.

Verify the pipeline works:

```bash
python src/dataprep.py --selftest
```

Copy the generic template, review the upstream licenses and terms, then add your own sources:

```bash
cp configs/sources.example.yaml configs/sources.local.yaml
# Edit configs/sources.local.yaml
python src/dataprep.py --manifest configs/sources.local.yaml \
  --out data/mycorpus-dd --workers 8 --resume
```

Or tokenize a single text file directly:

```bash
python scripts/prep_textfile.py --in my_text.txt --out data/mycorpus-dd
```

## 4. Train a nano model

Start with a ~30M parameter seed model:

```bash
charkha train --nano --data data/mycorpus-dd --out runs/nano --steps 2000
```

Monitor progress:

```bash
python src/_read_metrics.py runs/nano/metrics.jsonl
```

## 5. Grow to a larger model

Use the trained nano as a seed for a bigger model:

```bash
python scripts/grow_init.py --small runs/nano/ckpt.pt --mult 2 --out runs/grown/ckpt.pt
charkha train --data data/mycorpus-dd --out runs/grown --resume --steps 5000
```

## 6. Serve your model

```bash
charkha serve --ckpt runs/grown/ckpt.pt
```

Type questions. Try `what is 2+2?` — the model should use the tool calculator.
Try increasing the effort dial:

```bash
charkha serve --ckpt runs/grown/ckpt.pt --effort 8
```

## 7. Explore the effort dial

The core feature of CHARKHA is depth recurrence — the model loops its transformer
blocks multiple times per token, trading compute for depth.

```bash
# Compare accuracy at different effort levels
charkha pipeline --elasticity --ckpt runs/grown/ckpt.pt --shard data/mycorpus-dd/shard_00000.bin
```

Compare the measured curve across effort levels; improvement is an experimental result, not an
assumption of the architecture.

## Next steps

- **Evaluation**: `charkha pipeline --eval --ckpt runs/grown/ckpt.pt --tasks hellaswag`
- **SFT**: `python src/sft.py --run --ckpt runs/grown/ckpt.pt --data sft_data.jsonl`
- **Continual training**: `python src/continual.py` with task/reward configuration
- **Architecture reference**: `docs/ARCHITECTURE.md`

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: torch` | Activate venv: `source .venv/bin/activate` |
| CUDA out of memory | Reduce `--seq-len` or add `--embed-factor 256` |
| WSL `/mnt/` path warnings | Move repo to Linux filesystem: `~/charkha` |
| fla/triton not available | GDN falls back to pure PyTorch (slower, correct) |
| Training diverges (NaN) | Use `--profile frontier` (default) |

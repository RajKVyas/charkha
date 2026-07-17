"""CHARKHA data loading — streaming uint16/uint32 token shards."""

import json
import math
import os
import numpy as np
import torch
from dataprep import shard_format_for_vocab


class ShardLoader:
    """Reads the {index.json, shard_*.bin} layout dataprep.py writes.

    Accepts a single dir or a list of dirs; multi-dir mode pools shards from every dir
    # (length-weighted) so training reads directly from multiple data dirs
    The final `val_frac` of shards (at least one) is held out when split='val'."""

    def __init__(self, data_dirs, val_frac=0.0, split="train", dir_weights=None):
        # dir_weights: per-dir sampling multipliers (parallel to data_dirs). Sampling stays
        # length-weighted within a dir but a dir's mass is scaled by its weight — the lever for
        # mid-run data annealing (MiniCPM/OLMo style: upweight high-quality dirs late in the
        # cook) and for oversampling tiny personal corpora during consolidation.
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        if dir_weights is not None:
            dir_weights = [float(w) for w in dir_weights]
            bad = [w for w in dir_weights if not math.isfinite(w) or w <= 0]
            if bad:
                raise ValueError(f"dir_weights must be finite and > 0, got {bad}")
            if len(dir_weights) != len(data_dirs):
                raise ValueError(f"{len(dir_weights)} weights for {len(data_dirs)} data dirs")
        self._dirs = data_dirs
        vocab_sizes, all_arrays, all_lengths, all_files, tokenizer_names = [], [], [], [], []
        shard_w = []
        for di, dd in enumerate(data_dirs):
            with open(os.path.join(dd, "index.json")) as f:
                index = json.load(f)
            vocab_sizes.append(index["vocab_size"])
            tokenizer_names.append(index.get("tokenizer"))
            shards = index["shards"]
            if not shards:
                continue
            n_val = max(1, round(len(shards) * val_frac)) if val_frac > 0 else 0
            if split == "val":
                if n_val == 0:
                    raise ValueError("val split requested but val_frac == 0")
                use = shards[len(shards) - n_val :]
            else:
                use = shards[: len(shards) - n_val] if n_val else shards
            # dtype is per-dir (a dir's own index['vocab_size'] decides uint16 vs uint32), not a
            # module-wide constant -- correct even before the cross-dir uniformity check below runs.
            _, bytes_per_token = shard_format_for_vocab(index["vocab_size"])
            dtype = np.uint16 if bytes_per_token == 2 else np.uint32
            for s in use:
                fpath = os.path.join(dd, s["file"])
                all_arrays.append(np.memmap(fpath, dtype=dtype, mode="r"))
                all_lengths.append(s["tokens"])
                all_files.append(fpath)
                shard_w.append(dir_weights[di] if dir_weights is not None else 1.0)
        if not all_arrays:
            raise ValueError(f"no shards found in: {data_dirs}")
        if len(set(vocab_sizes)) != 1:
            raise ValueError(f"vocab size mismatch across data dirs: {set(vocab_sizes)}")
        self.vocab_size = vocab_sizes[0]
        # old shard dirs (pre tokenizer-field) have tokenizer=None; only enforce agreement among
        # dirs that actually recorded one, and prefer a non-None value as the loader's answer.
        named = {t for t in tokenizer_names if t}
        if len(named) > 1:
            raise ValueError(f"tokenizer mismatch across data dirs: {named}")
        self.tokenizer_name = named.pop() if named else None
        self.arrays = all_arrays
        self.lengths = np.array(all_lengths, dtype=np.int64)
        self._sample_w = self.lengths.astype(np.float64) * np.array(shard_w)
        self.total = int(self.lengths.sum())
        self.split = split

    def _pick_shard(self, T, rng):
        ok = np.where(self.lengths > T)[0]  # only shards that fit a full window
        pool = ok if len(ok) else np.arange(len(self.arrays))
        w = self._sample_w[pool]
        w = w / w.sum()
        r, c = rng.random(), 0.0
        for i, wi in enumerate(w):
            c += wi
            if r <= c:
                return int(pool[i])
        return int(pool[-1])

    def sample_tokens(self, B, L, device, rng, pin=False):
        """Sample raw token windows of length L."""
        z = torch.empty(B, L, dtype=torch.long, pin_memory=pin)
        for b in range(B):
            si = self._pick_shard(L, rng)
            a = self.arrays[si]
            hi = len(a) - L
            off = rng.randint(0, hi) if hi > 0 else 0
            z[b] = torch.from_numpy(a[off : off + L].astype(np.int64))
        return z.to(device, non_blocking=pin)

    def batch(self, B, T, device, rng, pin=False):
        chunk = self.sample_tokens(B, T + 1, device, rng, pin=pin)
        # pinned host buffers let the H2D copy overlap with compute (non_blocking) on CUDA.
        return chunk[:, :-1], chunk[:, 1:]

    def batch_with_context(self, context_loader, B, T, ctx_tokens, device, rng, pin=False):
        """Prefix ordinary training windows with retrieval/world-context tokens."""
        ctx_tokens = max(1, min(int(ctx_tokens), T - 2))
        main_tokens = T + 1 - ctx_tokens
        ctx = context_loader.sample_tokens(B, ctx_tokens, device, rng, pin=pin)
        main = self.sample_tokens(B, main_tokens, device, rng, pin=pin)
        chunk = torch.cat([ctx, main], dim=1)
        return chunk[:, :-1], chunk[:, 1:]


# --------------------------------------------------------------------------
# Checkpointing: full training state, atomic write.
# --------------------------------------------------------------------------

__all__ = ["ShardLoader"]

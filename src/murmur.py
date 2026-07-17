#!/usr/bin/env python3
"""Murmur: an imitation-free private reasoning register.

Mechanism (implemented + proofed here; capability claims are unverified):
a reserved band of vocabulary ids — disjoint from every id the tokenizer can emit, so
murmur tokens can NEVER occur in any corpus, prompt, or human text — in which the model
reasons privately before its visible thinking block:

    [BEGIN_MURMUR  <private-code tokens>  END_MURMUR]  ->  visible thinking  ->  answer
           (hidden, stripped from transcripts)              (shown)              (shown)

Four hard properties, each enforced by construction and covered by --selftest:
  1. Leak-impossibility: outside the murmur phase the entire band is masked to -inf,
     so P(band token) == 0 exactly. Visible text cannot contain murmur, ever.
  2. Imitation-freedom: band ids never appear in data, so no cross-entropy against any
     corpus can supervise them. Their only training signal is outcome pressure — the
     rejection-sampled bootstrap below keeps murmur blocks that measurably raise the
     likelihood of the subsequent answer (STaR-style, but over a private code space
     that has no human semantics to imitate).
  3. Ephemerality: strip_murmur() removes murmur spans from transcripts, and stripped
     transcripts are token-identical to never-murmured ones, so re-encoding on the next
     turn is exact. Murmur is per-response scratch, never carried forward.
  4. Exact migration: extend_vocab() grows an existing checkpoint's vocabulary by the
     band WITHOUT changing the model's distribution over real tokens (old embedding
     rows are copied; outside the murmur phase the band is masked, and softmax over the
     unmasked support is bit-identical to the original model's).

Two derived mechanisms (same band machinery, same guarantees, also proofed here):
  Vault  — a second, PERSISTENT private band: per-turn schedule is
           [carried vault] + prompt -> murmur (ephemeral) -> visible -> new vault.
           carry_forward() keeps the public transcript token-exact while the server
           privately carries the last vault span to the next turn. Guarantee is
           VERBATIM privacy only: vault ids can never be emitted or transcribed;
           semantic paraphrase of vault contents in visible text is NOT prevented
           by masking and is claimed nowhere here.
  Zip    — compression pressure on the bootstrap: hard code budgets, length-
           penalized acceptance (bootstrap_zip_step), budget annealing
           (anneal_budget), and the degenerate-code diagnostic
           (band_usage_entropy). A protocol, not a cost-reduction claim.

Adjacent work this is NOT:
Quiet-STaR (natural-language thoughts, shared vocab), Coconut (continuous latents, no
tokens), pause/filler tokens (fixed ids, pure compute, no code), planning tokens /
Token-Assorted (latent codes grounded in human traces via clustering/VQ — imitative).

Usage:
    python src/murmur.py --selftest

"""

import argparse
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Band definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MurmurBand:
    """A contiguous reserved id range [start, end): 2 markers + (end-start-2) codes."""

    start: int
    end: int

    @property
    def begin_id(self):
        return self.start

    @property
    def end_id(self):
        return self.start + 1

    @property
    def n_codes(self):
        return self.end - self.start - 2

    @staticmethod
    def for_extended_vocab(base_vocab, n_codes=1024):
        """Band appended above an existing tokenizer vocab (the production layout:
        v8 = 131072 real ids, murmur occupies [131072, 131072+2+n_codes))."""
        return MurmurBand(base_vocab, base_vocab + 2 + n_codes)

    def contains(self, ids):
        """Boolean mask over a tensor of token ids."""
        return (ids >= self.start) & (ids < self.end)


# ---------------------------------------------------------------------------
# Phase masks — the leak-impossibility mechanism
# ---------------------------------------------------------------------------


def mask_logits(logits, band, phase):
    """Return logits with the band masked for the given phase.

    phase='visible': every band id (markers included) -> -inf. Softmax then assigns
        the band exactly zero probability; visible text cannot emit murmur.
    phase='murmur': ONLY band codes and END_MURMUR are allowed; every real-vocab id
        and BEGIN_MURMUR -> -inf (BEGIN is inserted programmatically, never sampled).
    """
    out = logits.clone()
    if phase == "visible":
        out[..., band.start : band.end] = float("-inf")
    elif phase == "murmur":
        keep = out[..., band.start + 1 : band.end].clone()  # END marker + codes
        out[...] = float("-inf")
        out[..., band.start + 1 : band.end] = keep
    else:
        raise ValueError(f"phase must be 'visible' or 'murmur', got {phase!r}")
    return out


def strip_murmur(ids, band):
    """Remove [BEGIN .. END] murmur spans from a python list of ids (unclosed spans
    strip to the end). This is the transcript-level ephemerality contract."""
    out, inside = [], False
    for t in ids:
        if inside:
            if t == band.end_id:
                inside = False
        elif t == band.begin_id:
            inside = True
        elif band.contains(torch.tensor(t)).item():
            # stray band token outside a span — never legal, drop it defensively
            continue
        else:
            out.append(t)
    return out


def murmur_positions(ids, band):
    """Bool tensor marking positions inside murmur spans (markers included) — the
    complement is what corpus CE may supervise; the span is what bootstrap CE trains."""
    ids = torch.as_tensor(ids)
    pos = torch.zeros(ids.shape[-1], dtype=torch.bool)
    inside = False
    for i, t in enumerate(ids.tolist()):
        if t == band.begin_id:
            inside = True
        if inside:
            pos[i] = True
        if t == band.end_id:
            inside = False
    return pos


# ---------------------------------------------------------------------------
# Sampling + the two-register schedule
# ---------------------------------------------------------------------------


def _sample_step(model, idx, band, phase, temp, gen):
    logits, _ = model(idx)  # inference forward: (logits, conf)
    logits = mask_logits(logits[:, -1, :], band, phase)
    probs = F.softmax(logits / temp, dim=-1)
    return torch.multinomial(probs, 1, generator=gen)


@torch.no_grad()
def generate_murmur_block(model, prompt_ids, band, max_len=16, temp=1.0, gen=None):
    """Sample one murmur block: BEGIN + up to max_len band tokens, closed by END
    (forced if not sampled). Returns the block as a python list of ids."""
    dev = next(model.parameters()).device
    idx = torch.tensor([list(prompt_ids) + [band.begin_id]], dtype=torch.long, device=dev)
    block = [band.begin_id]
    for _ in range(max_len):
        t = _sample_step(model, idx, band, "murmur", temp, gen)
        block.append(int(t))
        idx = torch.cat([idx, t], dim=1)
        if int(t) == band.end_id:
            break
    if block[-1] != band.end_id:
        block.append(band.end_id)
    return block


@torch.no_grad()
def generate_two_register(
    model, prompt_ids, band, murmur_len=16, visible_len=16, temp=1.0, gen=None
):
    """The double-think schedule: private murmur, then visible tokens conditioned on
    it. Returns {'murmur', 'visible', 'transcript'} where transcript is the stripped
    (prompt + visible) sequence — what the outside world keeps."""
    dev = next(model.parameters()).device
    block = generate_murmur_block(model, prompt_ids, band, murmur_len, temp, gen)
    idx = torch.tensor([list(prompt_ids) + block], dtype=torch.long, device=dev)
    visible = []
    for _ in range(visible_len):
        t = _sample_step(model, idx, band, "visible", temp, gen)
        visible.append(int(t))
        idx = torch.cat([idx, t], dim=1)
    return {
        "murmur": block,
        "visible": visible,
        "transcript": strip_murmur(list(prompt_ids) + block + visible, band),
    }


# ---------------------------------------------------------------------------
# Outcome-pressure bootstrap (the ONLY training signal murmur can ever get)
# ---------------------------------------------------------------------------


@torch.no_grad()
def answer_logprob(model, ctx_ids, answer_ids):
    """Sum log P(answer | ctx) under the model (teacher-forced)."""
    dev = next(model.parameters()).device
    idx = torch.tensor([list(ctx_ids) + list(answer_ids)], dtype=torch.long, device=dev)
    logits, _ = model(idx)
    lp = F.log_softmax(logits[0].float(), dim=-1)
    n = len(ctx_ids)
    tot = 0.0
    for j, a in enumerate(answer_ids):
        tot += float(lp[n - 1 + j, a])
    return tot


@torch.no_grad()
def bootstrap_step(
    model, prompt_ids, answer_ids, band, k=4, max_len=8, margin=0.0, temp=1.0, gen=None
):
    """One rejection-sampling round: sample k murmur blocks, keep those that raise
    log P(answer) by at least `margin` over the no-murmur baseline. Accepted traces
    are the (only possible) supervised targets for the band. Returns (accepted, stats).
    """
    base = answer_logprob(model, prompt_ids, answer_ids)
    accepted, scores = [], []
    for _ in range(k):
        block = generate_murmur_block(model, prompt_ids, band, max_len, temp, gen)
        s = answer_logprob(model, list(prompt_ids) + block, answer_ids)
        scores.append(s)
        if s - base >= margin:
            trace = list(prompt_ids) + block + list(answer_ids)
            accepted.append(
                {"ids": trace, "uplift": s - base, "murmur_mask": murmur_positions(trace, band)}
            )
    return accepted, {"baseline": base, "scores": scores, "accept_rate": len(accepted) / max(k, 1)}


# ---------------------------------------------------------------------------
# Exact checkpoint migration: add the band to an existing model
# ---------------------------------------------------------------------------


def extend_vocab(model, cfg, band_codes=1024, init_std=1e-4, seed=0):
    """Grow a Charkha model's vocab by (2 + band_codes) rows, exactly.

    Every state tensor whose leading dim equals the old vocab (embedding codes,
    per-vocab buffers) is expanded: old rows copied verbatim, new rows ~N(0, init_std)
    for parameters (freshly-initialized buffers keep the new model's own values).
    Old-token logits are bit-identical because they depend only on copied rows, and
    outside the murmur phase the band is masked — so the visible-phase distribution
    is EXACTLY the original model's (proved in --selftest).

    Returns (new_model, new_cfg, band).
    """
    import copy
    from charkha import Charkha

    old_v = cfg.vocab_size
    new_cfg = copy.deepcopy(cfg)
    new_cfg.vocab_size = old_v + 2 + band_codes
    band = MurmurBand.for_extended_vocab(old_v, band_codes)

    new_model = Charkha(new_cfg)
    old_sd, new_sd = model.state_dict(), new_model.state_dict()
    g = torch.Generator().manual_seed(seed)
    param_names = {n for n, _ in new_model.named_parameters()}
    for key, new_t in new_sd.items():
        old_t = old_sd.get(key)
        if old_t is None:
            continue
        if old_t.shape == new_t.shape:
            new_t.copy_(old_t)
        elif (
            old_t.dim() == new_t.dim()
            and old_t.shape[0] == old_v
            and new_t.shape[0] == new_cfg.vocab_size
            and old_t.shape[1:] == new_t.shape[1:]
        ):
            new_t[:old_v] = old_t
            if key in param_names:  # buffers keep their freshly-constructed tail
                new_t[old_v:] = (
                    torch.randn(new_t[old_v:].shape, generator=g, dtype=torch.float32).to(
                        new_t.dtype
                    )
                    * init_std
                )
        else:
            raise RuntimeError(
                f"extend_vocab: unexpected shape change for {key}: "
                f"{tuple(old_t.shape)} -> {tuple(new_t.shape)}"
            )
    new_model.load_state_dict(new_sd)
    return new_model, new_cfg, band


# ---------------------------------------------------------------------------
# Vault: a second, PERSISTENT private band (compute is ephemeral; vault carries)
# ---------------------------------------------------------------------------
#
# Honesty note: masking makes VERBATIM
# leakage of vault tokens impossible — they can never be emitted in visible text
# and never appear in the public transcript. It does NOT prevent the model from
# PARAPHRASING vault contents in English during the visible phase. Verbatim
# privacy is a construction guarantee; semantic privacy is a training property
# and is claimed nowhere in this file.


@dataclass(frozen=True)
class MurmurLayout:
    """Two stacked reserved bands above the real vocab:
    compute  [base, base+2+cc)          — ephemeral per-response scratch
    vault    [base+2+cc, base+4+cc+vc)  — persistent private state, carried
                                          forward server-side between turns
    """

    compute: MurmurBand
    vault: MurmurBand

    @staticmethod
    def for_extended_vocab(base_vocab, compute_codes=1024, vault_codes=1024):
        c = MurmurBand.for_extended_vocab(base_vocab, compute_codes)
        v = MurmurBand.for_extended_vocab(c.end, vault_codes)
        return MurmurLayout(c, v)

    @property
    def n_new_rows(self):
        return (self.compute.end - self.compute.start) + (self.vault.end - self.vault.start)

    @property
    def vocab_end(self):
        return self.vault.end

    def contains(self, ids):
        return self.compute.contains(ids) | self.vault.contains(ids)


def mask_logits_layout(logits, layout, phase):
    """Phase masks over both bands.

    'visible': both bands -> -inf (verbatim leakage of either band impossible).
    'murmur' : only compute codes + compute END allowed.
    'vault'  : only vault codes + vault END allowed.
    """
    out = logits.clone()
    if phase == "visible":
        out[..., layout.compute.start : layout.compute.end] = float("-inf")
        out[..., layout.vault.start : layout.vault.end] = float("-inf")
    elif phase == "murmur":
        b = layout.compute
        keep = out[..., b.start + 1 : b.end].clone()
        out[...] = float("-inf")
        out[..., b.start + 1 : b.end] = keep
    elif phase == "vault":
        b = layout.vault
        keep = out[..., b.start + 1 : b.end].clone()
        out[...] = float("-inf")
        out[..., b.start + 1 : b.end] = keep
    else:
        raise ValueError(f"phase must be 'visible', 'murmur', or 'vault', got {phase!r}")
    return out


def carry_forward(ids, layout):
    """Split a raw turn sequence into (public_transcript, vault_block).

    public_transcript: both bands stripped — token-identical to a never-murmured
        turn, so the outside world (and next-turn re-encoding of visible history)
        is exact. The vault block is NEVER part of the public transcript.
    vault_block: the LAST closed vault span in `ids` (later state supersedes
        earlier), or [] if none — the server prepends it privately next turn.
    """
    public = strip_murmur(strip_murmur(ids, layout.compute), layout.vault)
    vault_block, cur, inside = [], [], False
    for t in ids:
        if inside:
            cur.append(t)
            if t == layout.vault.end_id:
                vault_block, inside = cur, False
        elif t == layout.vault.begin_id:
            cur, inside = [t], True
    return public, vault_block


@torch.no_grad()
def generate_stateful_turn(
    model,
    prompt_ids,
    layout,
    carried_vault=None,
    murmur_len=16,
    visible_len=16,
    vault_len=16,
    temp=1.0,
    gen=None,
):
    """One turn of the stateful schedule:

        [carried vault (private)] + prompt
          -> murmur block (private, ephemeral)
          -> visible tokens (shown)
          -> new vault block (private, persistent)

    Returns {'murmur','visible','vault','transcript'} where transcript is the
    public (prompt + visible) sequence and 'vault' is what the server carries to
    the next turn. carried_vault conditions everything but appears nowhere public.
    """
    dev = next(model.parameters()).device
    carried = list(carried_vault or [])
    ctx = carried + list(prompt_ids)

    def step(idx, phase):
        logits, _ = model(idx)
        logits = mask_logits_layout(logits[:, -1, :], layout, phase)
        probs = F.softmax(logits / temp, dim=-1)
        return torch.multinomial(probs, 1, generator=gen)

    def block(idx, band, phase, max_len):
        out = [band.begin_id]
        idx = torch.cat([idx, torch.tensor([[band.begin_id]], device=dev)], dim=1)
        for _ in range(max_len):
            t = step(idx, phase)
            out.append(int(t))
            idx = torch.cat([idx, t], dim=1)
            if int(t) == band.end_id:
                break
        if out[-1] != band.end_id:
            out.append(band.end_id)
            idx = torch.cat([idx, torch.tensor([[band.end_id]], device=dev)], dim=1)
        return out, idx

    idx = torch.tensor([ctx], dtype=torch.long, device=dev)
    murmur, idx = block(idx, layout.compute, "murmur", murmur_len)
    visible = []
    for _ in range(visible_len):
        t = step(idx, "visible")
        visible.append(int(t))
        idx = torch.cat([idx, t], dim=1)
    vault, idx = block(idx, layout.vault, "vault", vault_len)
    raw = ctx + murmur + visible + vault
    public, carried_out = carry_forward(raw, layout)
    return {"murmur": murmur, "visible": visible, "vault": carried_out, "transcript": public}


# ---------------------------------------------------------------------------
# Zip: compression pressure on the bootstrap (budget + length penalty + anneal)
# ---------------------------------------------------------------------------
#
# Zip is a PROTOCOL over the same band, not a new mechanism: force the murmur
# block into a hard token budget and make the bootstrap prefer the shortest
# block that still buys the uplift. Whether a tiny code block can replace long
# visible reasoning is an open empirical question (E7 in the paper) — nothing
# here claims a cost reduction; it only makes the pressure trainable.


@torch.no_grad()
def bootstrap_zip_step(
    model,
    prompt_ids,
    answer_ids,
    band,
    k=4,
    budget=8,
    margin=0.0,
    len_weight=0.1,
    temp=1.0,
    gen=None,
):
    """Length-penalized rejection sampling: sample k blocks capped at `budget`
    codes; accept those with uplift >= margin; rank accepted by
    (uplift - len_weight * n_codes) so shorter blocks that buy the same uplift
    win. Returns (accepted_sorted, stats)."""
    base = answer_logprob(model, prompt_ids, answer_ids)
    accepted, scores = [], []
    for _ in range(k):
        blk = generate_murmur_block(model, prompt_ids, band, budget, temp, gen)
        n_codes = sum(1 for t in blk if t not in (band.begin_id, band.end_id))
        assert n_codes <= budget
        s = answer_logprob(model, list(prompt_ids) + blk, answer_ids)
        scores.append(s)
        uplift = s - base
        if uplift >= margin:
            trace = list(prompt_ids) + blk + list(answer_ids)
            accepted.append(
                {
                    "ids": trace,
                    "uplift": uplift,
                    "n_codes": n_codes,
                    "zip_score": uplift - len_weight * n_codes,
                    "murmur_mask": murmur_positions(trace, band),
                }
            )
    accepted.sort(key=lambda a: -a["zip_score"])
    return accepted, {"baseline": base, "scores": scores, "accept_rate": len(accepted) / max(k, 1)}


def anneal_budget(budget, accept_rate, target=0.5, min_budget=2, step=1):
    """Budget annealing: when acceptance is comfortably above target, tighten
    the budget by `step`; when below, relax it. Ratchets murmur blocks toward
    the shortest length the model can still exploit."""
    if accept_rate > target:
        return max(min_budget, budget - step)
    if accept_rate < target:
        return budget + step
    return budget


def band_usage_entropy(blocks, band):
    """Shannon entropy (nats) of code-id usage across murmur blocks — the
    degenerate-code diagnostic: a collapsed band (one id repeated) scores ~0,
    a uniform band scores log(n_codes). Log this from the first real run."""
    counts = {}
    for blk in blocks:
        for t in blk:
            if t not in (band.begin_id, band.end_id) and band.contains(torch.tensor(t)).item():
                counts[t] = counts.get(t, 0) + 1
    tot = sum(counts.values())
    if tot == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / tot
        ent -= p * torch.log(torch.tensor(p)).item()
    return ent


def extend_vocab_layout(model, cfg, compute_codes=1024, vault_codes=1024, init_std=1e-4, seed=0):
    """extend_vocab for the two-band layout: grows by (4 + cc + vc) rows with the
    same exactness guarantee, returns (new_model, new_cfg, layout)."""
    # reuse extend_vocab's row logic by expressing the layout as one code count:
    # total new rows = 2 markers + N where N = 2 + cc + vc (vault markers+codes
    # live in the same appended region).
    new_model, new_cfg, _ = extend_vocab(
        model, cfg, band_codes=2 + compute_codes + vault_codes, init_std=init_std, seed=seed
    )
    layout = MurmurLayout.for_extended_vocab(cfg.vocab_size, compute_codes, vault_codes)
    assert layout.vocab_end == new_cfg.vocab_size
    return new_model, new_cfg, layout


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------


def _selftest():
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from charkha import Charkha, CharkhaConfig

    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    checks = 0

    # --- band bookkeeping ---------------------------------------------------
    band = MurmurBand.for_extended_vocab(256, n_codes=30)
    assert (band.begin_id, band.end_id, band.n_codes, band.end) == (256, 257, 30, 288)
    checks += 1

    # --- mask exactness: visible phase assigns the band probability 0 -------
    logits = torch.randn(2, band.end)
    pv = F.softmax(mask_logits(logits, band, "visible"), dim=-1)
    assert float(pv[:, band.start : band.end].abs().sum()) == 0.0
    assert torch.allclose(
        pv[:, : band.start], F.softmax(logits[:, : band.start], dim=-1), atol=1e-6
    )
    pm = F.softmax(mask_logits(logits, band, "murmur"), dim=-1)
    assert float(pm[:, : band.start + 1].abs().sum()) == 0.0  # real vocab + BEGIN = 0
    assert abs(float(pm.sum()) - 2.0) < 1e-5  # rows still normalize
    checks += 2

    # --- strip/ephemerality: stripped == never-murmured ----------------------
    clean = [5, 6, 7, 8]
    with_m = [5, 6, band.begin_id, 260, 261, band.end_id, 7, 8]
    assert strip_murmur(with_m, band) == clean
    assert strip_murmur([band.begin_id, 260], band) == []  # unclosed
    assert strip_murmur([270, 5], band) == [5]  # stray band id dropped
    mp = murmur_positions(with_m, band)
    assert mp.tolist() == [False, False, True, True, True, True, False, False]
    checks += 4

    # --- model-in-the-loop: toy Charkha extended with a real band -----------
    cfg = CharkhaConfig.toy()
    model = Charkha(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        base_logits, _ = model(x)

    model2, cfg2, band2 = extend_vocab(model, cfg, band_codes=30)
    model2.eval()
    with torch.no_grad():
        ext_logits, _ = model2(x)
    # exact preservation: old-token logits identical; masked distribution identical
    assert torch.allclose(base_logits, ext_logits[..., : cfg.vocab_size], atol=1e-5), (
        "old-token logits changed under vocab extension"
    )
    p_old = F.softmax(base_logits.float(), dim=-1)
    p_new = F.softmax(mask_logits(ext_logits.float(), band2, "visible"), dim=-1)
    assert torch.allclose(p_old, p_new[..., : cfg.vocab_size], atol=1e-5), (
        "visible-phase distribution not preserved by extension"
    )
    checks += 2

    # --- leak-impossibility under real sampling -----------------------------
    two = generate_two_register(model2, x[0].tolist(), band2, murmur_len=6, visible_len=24, gen=gen)
    assert all(band2.contains(torch.tensor(t)).item() for t in two["murmur"])
    assert not any(band2.contains(torch.tensor(t)).item() for t in two["visible"])
    assert two["murmur"][0] == band2.begin_id and two["murmur"][-1] == band2.end_id
    assert two["transcript"] == x[0].tolist() + two["visible"]  # ephemerality
    checks += 4

    # --- bootstrap: structure + margin logic ---------------------------------
    prompt, answer = x[0, :8].tolist(), x[0, 8:12].tolist()
    acc, stats = bootstrap_step(
        model2, prompt, answer, band2, k=3, max_len=5, margin=float("-inf"), gen=gen
    )
    assert len(acc) == 3 and stats["accept_rate"] == 1.0
    for a in acc:
        assert a["uplift"] >= float("-inf") and a["murmur_mask"].any()
        span = [t for t, m in zip(a["ids"], a["murmur_mask"].tolist()) if m]
        assert all(band2.contains(torch.tensor(t)).item() for t in span)
        # recompute the uplift independently — acceptance must be self-consistent
        blk = [t for t in a["ids"] if band2.contains(torch.tensor(t)).item()]
        s = answer_logprob(model2, prompt + blk, answer)
        assert abs((s - stats["baseline"]) - a["uplift"]) < 1e-4
    none, _ = bootstrap_step(
        model2, prompt, answer, band2, k=2, max_len=5, margin=float("inf"), gen=gen
    )
    assert none == []
    checks += 3

    # --- layout bookkeeping + layout masks -----------------------------------
    lay = MurmurLayout.for_extended_vocab(256, compute_codes=10, vault_codes=12)
    assert lay.compute.start == 256 and lay.compute.end == 268
    assert lay.vault.start == 268 and lay.vault.end == 282 and lay.vocab_end == 282
    logits3 = torch.randn(2, lay.vocab_end)
    pv3 = F.softmax(mask_logits_layout(logits3, lay, "visible"), dim=-1)
    assert float(pv3[:, lay.compute.start : lay.vault.end].abs().sum()) == 0.0
    pm3 = F.softmax(mask_logits_layout(logits3, lay, "murmur"), dim=-1)
    assert float(pm3[:, : lay.compute.start + 1].abs().sum()) == 0.0
    assert float(pm3[:, lay.vault.start : lay.vault.end].abs().sum()) == 0.0
    pva = F.softmax(mask_logits_layout(logits3, lay, "vault"), dim=-1)
    assert float(pva[:, : lay.vault.start + 1].abs().sum()) == 0.0
    checks += 2

    # --- carry_forward: public transcript clean; LAST vault span carries ------
    turn = [
        5,
        lay.compute.begin_id,
        258,
        lay.compute.end_id,
        6,
        lay.vault.begin_id,
        270,
        lay.vault.end_id,
        7,
        lay.vault.begin_id,
        271,
        272,
        lay.vault.end_id,
    ]
    public, vb = carry_forward(turn, lay)
    assert public == [5, 6, 7]
    assert vb == [lay.vault.begin_id, 271, 272, lay.vault.end_id]  # last span wins
    public2, vb2 = carry_forward([5, 6], lay)
    assert public2 == [5, 6] and vb2 == []
    checks += 2

    # --- stateful turn with a real model: three registers, no cross-leaks -----
    model3, cfg3, lay3 = extend_vocab_layout(model, cfg, compute_codes=10, vault_codes=12)
    model3.eval()
    with torch.no_grad():
        ext3, _ = model3(x)
    assert torch.allclose(base_logits, ext3[..., : cfg.vocab_size], atol=1e-5)
    turn1 = generate_stateful_turn(
        model3, x[0, :6].tolist(), lay3, murmur_len=4, visible_len=8, vault_len=4, gen=gen
    )
    assert all(lay3.compute.contains(torch.tensor(t)).item() for t in turn1["murmur"])
    assert all(lay3.vault.contains(torch.tensor(t)).item() for t in turn1["vault"])
    assert not any(lay3.contains(torch.tensor(t)).item() for t in turn1["visible"])
    assert not any(lay3.contains(torch.tensor(t)).item() for t in turn1["transcript"])
    assert turn1["transcript"] == x[0, :6].tolist() + turn1["visible"]
    # second turn conditioned on carried vault — still nothing private in public
    turn2 = generate_stateful_turn(
        model3,
        x[0, 6:10].tolist(),
        lay3,
        carried_vault=turn1["vault"],
        murmur_len=4,
        visible_len=8,
        vault_len=4,
        gen=gen,
    )
    assert not any(lay3.contains(torch.tensor(t)).item() for t in turn2["transcript"])
    assert turn2["vault"][0] == lay3.vault.begin_id
    checks += 3

    # --- zip bootstrap: budget honored, length penalty orders acceptances -----
    accz, statsz = bootstrap_zip_step(
        model2, prompt, answer, band2, k=4, budget=3, margin=float("-inf"), len_weight=0.5, gen=gen
    )
    assert len(accz) == 4 and statsz["accept_rate"] == 1.0
    for a in accz:
        assert a["n_codes"] <= 3
        assert abs(a["zip_score"] - (a["uplift"] - 0.5 * a["n_codes"])) < 1e-9
    assert all(accz[i]["zip_score"] >= accz[i + 1]["zip_score"] for i in range(len(accz) - 1))
    checks += 2

    # --- budget annealing + band-entropy diagnostic ---------------------------
    assert anneal_budget(8, accept_rate=0.9) == 7  # tighten
    assert anneal_budget(8, accept_rate=0.1) == 9  # relax
    assert anneal_budget(2, accept_rate=1.0, min_budget=2) == 2
    degen = band_usage_entropy([[band.begin_id, 260, 260, 260, band.end_id]], band)
    diverse = band_usage_entropy([[band.begin_id, 260, 261, 262, band.end_id]], band)
    assert degen == 0.0 and diverse > 1.0
    assert band_usage_entropy([], band) == 0.0
    checks += 2

    print(f"[selftest] murmur.py: all checks passed ({checks} groups)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    ap.print_help()

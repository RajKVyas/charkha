"""
CHARKHA dataprep - streaming data pipeline (REFERENCE).
=======================================================================
Reads a user-supplied source manifest, streams each dataset, and runs every document through:

  1. opt-out filter  Drops explicitly configured domains. Dataset selection and license review
                     happen when the source manifest is prepared.
  2. normalize       NFC, de-mojibake, strip HTML/boilerplate, collapse whitespace
  3. quality_filter  Gopher/C4-style heuristics (+ language id in real mode)
  4. dedup           exact (sha1) + near-dup (MinHash-LSH), document level
  5. pii_scrub       redact emails / phones / IPs / keys (Apertus-style)
  6. decontaminate   drop docs overlapping eval benchmarks (n-gram)
  7. tokenize_shard  BPE -> .bin shards (uint16/uint32 per vocab size) + provenance index.json

Design: the 7 stages are pure-stdlib so `python dataprep.py --selftest` verifies the
whole pipeline with no network and no third-party deps (mirrors charkha._toy --toy).
Real runs lazy-import `datasets`/`tokenizers`/`yaml`/`ftfy`.

quality/dedup/PII/decontam stages are engineering invariants and always run.

Usage:
  python dataprep.py --selftest                 # verify the pipeline (sandbox/CI)
  cp configs/sources.example.yaml configs/sources.local.yaml
  # Edit sources.local.yaml, then:
  python src/dataprep.py --manifest configs/sources.local.yaml --out data/corpus --limit 100000
"""

from __future__ import annotations
import argparse
import array
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict

# --------------------------------------------------------------------------
# Stage 1 - opt-out filter. License values are normalized as provenance metadata; callers remain
# responsible for selecting appropriately licensed sources in the manifest.
# --------------------------------------------------------------------------
_LICENSE_CANON = {
    "public domain": "public-domain",
    "pd": "public-domain",
    "cc0": "cc0",
    "cc-by": "cc-by",
    "cc by": "cc-by",
    "cc-by-4.0": "cc-by",
    "cc-by-sa": "cc-by-sa",
    "cc by-sa": "cc-by-sa",
    "mit": "mit",
    "apache": "apache-2.0",
    "apache-2.0": "apache-2.0",
    "bsd": "bsd",
    "gov": "gov-public",
    "cc-by-nc": "cc-by-nc",
    "cc-by-nd": "cc-by-nd",
    "gfdl": "gfdl-only",
    "none": "none-restrictive",
    "": "unknown",
}
_LICENSE_CANON_VALUES = set(_LICENSE_CANON.values())


def canon_license(s):
    # Canonicalize a license string for provenance metadata.
    s = (s or "").strip().lower()
    if s in _LICENSE_CANON_VALUES:
        return s  # IDEMPOTENT: already-canonical forms round-trip
    if s in _LICENSE_CANON:
        return _LICENSE_CANON[s]
    for k, v in _LICENSE_CANON.items():
        if k and k in s:
            return v
    return "unknown"


def domain_of(url):
    m = re.match(r"https?://([^/]+)", url or "")
    return (m.group(1).lower() if m else "").replace("www.", "")


def opted_out(doc, optout):
    """True if a document's domain is on the explicit opt-out list."""
    return bool(optout) and domain_of(doc.get("url", "")) in optout


# --------------------------------------------------------------------------
# Stage 2 - normalize
# --------------------------------------------------------------------------
_HTML = re.compile(r"<[^>]+>")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_WS_LINE = re.compile(r"[ \t]+")
_WS_NL = re.compile(r"\n{3,}")


def normalize(text):
    try:
        import ftfy

        text = ftfy.fix_text(text)  # lazy; real-mode only
    except Exception:
        pass
    text = unicodedata.normalize("NFC", text)
    text = _HTML.sub(" ", text)
    text = _CTRL.sub("", text)
    text = "\n".join(_WS_LINE.sub(" ", ln).rstrip() for ln in text.split("\n"))
    text = _WS_NL.sub("\n\n", text).strip()
    return text


# --------------------------------------------------------------------------
# Digit isolation (math-from-the-start). BPE fuses multi-digit numbers into arbitrary chunks
# ("1234" -> "12","34"; "1235" -> "123","5"), so a small model sees a *different* token for nearly
# every number and can't learn place-value arithmetic. Putting a space before each digit forces ONE
# token per digit — the standard fix in math-capable models (Llama, Minerva, etc.). Applied as a
# text normalization BEFORE tokenization, so the vocab is unchanged (GPT-NeoX already has 0-9 as
# single tokens) and KD stays consistent IF the teacher's text is normalized the same way. serve.py
# must apply split_digits to user input and join_digits to model output so train/infer match.
# --------------------------------------------------------------------------
_DIGIT_BOUND = re.compile(r"(?<=\S)(?=\d)")  # a non-space char immediately before a digit
_SPLIT_DIGITS = re.compile(r"(?<=\d) (?=\d)")  # inverse: a space flanked by digits


def split_digits(text):
    """Separate every digit so each tokenizes on its own ("ab12.3" -> "ab 1 2. 3"). Inserts a space
    only at a non-space→digit boundary, so already-whitespace-delimited numbers don't get doubled
    and join_digits is a clean inverse for them."""
    return _DIGIT_BOUND.sub(" ", text)


def join_digits(text):
    """Inverse of split_digits for display: re-glue single spaces that sit between two digits
    ("1 2 3" -> "123"). The spaces are non-overlapping so one global sub suffices."""
    return _SPLIT_DIGITS.sub("", text)


# --------------------------------------------------------------------------
# Stage 3 - quality filter (Gopher/C4-style)
# --------------------------------------------------------------------------
def quality_filter(text, min_words=10, max_words=100_000, target_langs=None, content_type="prose"):
    words = text.split()
    n = len(words)
    if n < min_words:
        return False, "too_short"
    if n > max_words:
        return False, "too_long"
    mean_wl = sum(len(w) for w in words) / n
    if content_type != "code" and not (2.5 <= mean_wl <= 12):
        return False, "mean_word_len"
    alpha = sum(c.isalpha() for c in text)
    if content_type != "code" and alpha / max(len(text), 1) < 0.6:
        return False, "low_alpha"
    sym = sum(text.count(c) for c in "#{}[]<>|^~")
    if content_type != "code" and sym / n > 0.10:
        return False, "symbol_ratio"
    lines = [l for l in text.split("\n") if l.strip()]
    if content_type != "code" and lines:
        dup = 1 - len(set(lines)) / len(lines)
        if dup > 0.30:
            return False, "line_repetition"
    # Fast character-ngram language detection (no C extensions needed)
    if target_langs and len(target_langs) == 1 and target_langs[0] == "en":
        if _detect_lang(text[:1024]) != "en":
            return False, "non_en"
    return True, "ok"


# Minimal character-trigram lang detection: English vs non-English.
# Accurate enough for filtering; replace with fastText if you need multi-lang.
_EN_TRIGRAMS: set = None


def _detect_lang(text: str) -> str:
    global _EN_TRIGRAMS
    if _EN_TRIGRAMS is None:
        common_en = "the and ing hat you tha was for ent ion her his tio ere ver all wit thi "
        common_en += "ed of in to it is re an on at ha ve me or as be he hi no so we if "
        common_en += "but not they have from are this that with which will their can had what "
        common_en += "when your said there use each how out then them these she some other "
        _EN_TRIGRAMS = set(common_en.lower().split())
    txt = text.lower().replace("\n", " ")
    trigrams = set(
        txt[i : i + 3]
        for i in range(len(txt) - 2)
        if txt[i].isalpha() and txt[i + 1].isalpha() and txt[i + 2].isalpha()
    )
    overlap = len(trigrams & _EN_TRIGRAMS) / max(len(trigrams), 1)
    return "en" if overlap > 0.04 else "other"


# --------------------------------------------------------------------------
# Stage 4 - dedup (exact sha1 + near-dup MinHash-LSH, pure python)
# --------------------------------------------------------------------------
from _dedup import Deduper, _PII


def pii_scrub(text):
    n = 0
    for rx, repl in _PII:
        text, c = rx.subn(repl, text)
        n += c
    return text, n


# --------------------------------------------------------------------------
# Stage 6 - decontamination vs eval benchmarks
# --------------------------------------------------------------------------
class Decontaminator:
    def __init__(self, bench_strings, n=8):
        self.n = n
        self.grams = set()
        for s in bench_strings:
            self.grams |= self._ngrams(s)

    def _ngrams(self, text):
        w = normalize(text).lower().split()
        return (
            {" ".join(w[i : i + self.n]) for i in range(len(w) - self.n + 1)}
            if len(w) >= self.n
            else set()
        )

    def contaminated(self, text):
        return bool(self._ngrams(text) & self.grams)


# --------------------------------------------------------------------------
# Stage 7 - tokenize + shard (uint16 .bin via stdlib array, no numpy)
# --------------------------------------------------------------------------
def _has_native_digit_pretokenizer(path):
    """True iff a saved tokenizer.json's pre_tokenizer already isolates every DECIMAL digit on its
    own — either a Digits component (scripts/train_tokenizer.py's individual_digits=True, the
    pre-hex-merge format) or a Split component whose regex pattern contains the bare single-digit
    branch ` ?[0-9]` that train_tokenizer.py's hex-merge GPT2_HEX_REGEX uses (hex literals like
    0x7fffffffe3a0 are carved out and merged whole there, but decimal place-value digits still
    isolate one-at-a-time, so the same --digit-split double-processing hazard applies). Either way,
    applying the --digit-split STRING hack on top would double-process: split_digits inserts a
    space before each digit, which the native pretokenizer then sees as a *different* input than
    the raw digit run it was trained on, silently changing (and likely hurting) the encoding. Hub
    names (no local file) return False — preserves prior behavior for tokenizers that don't do this
    natively."""
    if not os.path.exists(path):
        return False
    try:
        data = json.load(open(path))
    except Exception:
        return False

    def has_digits(node):
        if isinstance(node, dict):
            if node.get("type") == "Digits":
                return True
            if node.get("type") == "Split":
                pattern = node.get("pattern", {})
                regex = pattern.get("Regex", "") if isinstance(pattern, dict) else ""
                if "[0-9]" in regex:
                    return True
            return any(has_digits(v) for v in node.values())
        if isinstance(node, list):
            return any(has_digits(v) for v in node)
        return False

    return has_digits(data.get("pre_tokenizer"))


def load_tokenizer(name=None, digit_split=False):
    if name:
        try:
            from tokenizers import Tokenizer  # lazy; real-mode only

            # local tokenizer.json (e.g. scripts/train_tokenizer.py's output) vs an HF hub name
            tok = (
                Tokenizer.from_file(name)
                if os.path.exists(name)
                else Tokenizer.from_pretrained(name)
            )
            native_digits = _has_native_digit_pretokenizer(name)
            if digit_split and native_digits:
                print(
                    f"  [tokenizer] {name} already isolates digits natively; ignoring --digit-split"
                )
            pre = (
                (lambda t: t) if native_digits else (split_digits if digit_split else (lambda t: t))
            )
            eos = tok.token_to_id("<|endoftext|>") or 0
            return (lambda t: tok.encode(pre(t)).ids + [eos]), tok.get_vocab_size()
        except Exception as e:
            print(f"  [tokenizer] falling back to byte tokenizer ({e})")
    EOS = 256
    pre = split_digits if digit_split else (lambda t: t)
    return (lambda t: list(pre(t).encode("utf-8")) + [EOS]), 257  # uint16-safe


def shard_format_for_vocab(vocab_size):
    """Single source of truth for the .bin shard format, shared by every writer/reader in the repo.
    vocab<=65535 -> uint16 ('H', 2 bytes/token) -- the format of every shard prepared before the
    custom-tokenizer work. vocab>65535 (a bigger custom tokenizer) -> uint32 ('I', 4 bytes/token).
    'I' (C unsigned int), NOT 'L' (C unsigned long): on 64-bit Linux/macOS (LP64) array('L') is
    8 bytes, not 4, while array('I') has been 4 bytes on every mainstream platform (Windows/Linux/
    macOS, x86/x64/ARM) since the 1990s -- asserted defensively since a silent mismatch here would
    corrupt every shard's byte layout without erroring until something tries to read it back."""
    if vocab_size is not None and vocab_size > 65535:
        assert array.array("I").itemsize == 4, (
            "platform array('I') is not 4 bytes -- uint32 shard format would be wrong here"
        )
        return "I", 4
    return "H", 2


class ShardWriter:
    def __init__(self, out_dir, shard_tokens=100_000_000, resume=False, vocab_size=None):
        self.out, self.shard_tokens = out_dir, shard_tokens
        os.makedirs(out_dir, exist_ok=True)
        typecode, self.bytes_per_token = shard_format_for_vocab(vocab_size)
        self.buf = array.array(typecode)
        self.shards = []
        self.total = 0
        if resume:
            self._load_existing()

    def _load_existing(self):
        """Resume without clobbering: adopt the intact, contiguously-numbered full shards already on
        disk so _flush appends past them (shard counter = len(self.shards)). Stops at the first
        missing/short/renumbered shard so the numbering stays gap-free."""
        idx_path = os.path.join(self.out, "index.json")
        if not os.path.isfile(idx_path):
            return
        try:
            old = json.load(open(idx_path))
        except Exception as e:
            print(f"  [resume] cannot read existing index.json ({e}); starting fresh")
            return
        for s in old.get("shards", []):
            fp = os.path.join(self.out, s["file"])
            expect = f"shard_{len(self.shards):05d}.bin"
            if (
                s.get("file") == expect
                and os.path.isfile(fp)
                and os.path.getsize(fp) == s["tokens"] * self.bytes_per_token
            ):
                self.shards.append({"file": s["file"], "tokens": s["tokens"]})
                self.total += s["tokens"]
            else:
                break
        if self.shards:
            print(
                f"  [resume] keeping {len(self.shards)} intact shards ({self.total:,} tokens); "
                f"appending from shard_{len(self.shards):05d}"
            )

    def add(self, ids):
        self.buf.extend(ids)
        self.total += len(ids)
        while len(self.buf) >= self.shard_tokens:
            self._flush(self.shard_tokens)

    def _flush(self, count):
        path = os.path.join(self.out, f"shard_{len(self.shards):05d}.bin")
        with open(path, "wb") as f:
            self.buf[:count].tofile(f)
        self.shards.append({"file": os.path.basename(path), "tokens": count})
        del self.buf[:count]

    def close(self):
        if self.buf:
            self._flush(len(self.buf))
        return self.shards


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def _process_doc(doc, optout, dedup, decon, encode, min_words, tag=None):
    """Run one document through every cleaning stage.
    Returns (status, ids, n_pii): status is 'kept' or 'drop_<stage>'; ids is the token
    list only when kept; n_pii counts redactions (>0 once pii_scrub runs, even if the doc
    is later dropped by decontam - matching the original accounting order).
    Source selection and license review happen at manifest authoring time. This function also
    applies an explicit domain opt-out list before the remaining cleaning stages.

    If `tag` is set (e.g. '<src:wikipedia>\\n'), it is prepended to the
    doc text BEFORE encoding so every doc opens with its source tokens. Cleaning stages see
    the untagged text, so quality/dedup/decontam are unaffected. Evidence (Allen-Zhu & Li,
    Physics of LLMs 3.3): source tags let the model route storage budget to trustworthy
    sources, restoring knowledge capacity that mixed-quality data otherwise collapses. Plain
    text costs ~3-5 tokens/doc; a reserved special id would cost 1 (upgrade path)."""
    if opted_out(doc, optout):
        return "drop_optout", None, 0
    text = normalize(doc["text"])
    ok, _ = quality_filter(text, min_words, content_type=doc.get("content_type", "prose"))
    if not ok:
        return "drop_quality", None, 0
    d = dedup.check(text)
    if d:
        return f"drop_dedup_{d}", None, 0
    text, npii = pii_scrub(text)
    if decon.contaminated(text):
        return "drop_decontam", None, npii
    return "kept", encode((tag + text) if tag else text), npii


def run_pipeline(docs, out_dir, cfg):
    optout = set(cfg.get("optout", []))  # explicit takedown/opt-out list (empty = drop nothing)
    dedup = Deduper(cfg.get("dedup_mode", "near"))
    decon = Decontaminator(cfg.get("bench", []), n=cfg.get("decon_n", 8))
    encode, vocab = load_tokenizer(cfg.get("tokenizer"), cfg.get("digit_split", False))
    writer = ShardWriter(out_dir, cfg.get("shard_tokens", 100_000_000), vocab_size=vocab)
    mw = cfg.get("min_words", 10)
    # Explicit cfg['src_tag'] wins; otherwise derive a tag when provenance is enabled.
    # derive '<src:NAME>\n' from the source name / out dir. Default None => byte-identical to before.
    tag = cfg.get("src_tag")
    if tag is None and cfg.get("provenance"):
        name = cfg.get("name") or os.path.basename(out_dir.rstrip("/\\")).removesuffix("-dd")
        tag = f"<src:{name}>\n"
    acct = defaultdict(int)
    kept_ids = []
    for doc in docs:
        acct["seen"] += 1
        status, ids, npii = _process_doc(doc, optout, dedup, decon, encode, mw, tag=tag)
        acct["pii_redactions"] += npii
        if status == "kept":
            writer.add(ids)
            acct["kept"] += 1
            kept_ids.append(doc.get("id"))
        else:
            acct[status] += 1
    shards = writer.close()
    index = {
        "vocab_size": vocab,
        "total_tokens": writer.total,
        "tokenizer": cfg.get("tokenizer"),
        "shards": shards,
        "accounting": dict(acct),
        "src_tag": tag,
    }
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    return index, kept_ids


def _stream_file_shards(get_shard, indices):
    """Resilient FILE-level streaming: `get_shard(i)` returns an iterable over ONE underlying data
    file (via datasets' IterableDataset.shard). A file that raises — corrupt/empty chunk (e.g. Comma's
    arxiv_papers.chunk.16), a dropped connection, a parse error — is logged and skipped, so a single
    bad file can't crash a multi-billion-token run, and a transient mid-stream failure can't make the
    whole source end early (the bug that left CommonCorpus at 0.15B). Crucially we never re-read a bad
    file: a doc-count `.skip()` can't get *past* a bad FILE (it re-reads and re-raises), which is why
    row-level skip-ahead fails and file-level isolation is the correct fix. Catches Exception only;
    KeyboardInterrupt / GeneratorExit still stop cleanly."""
    for i in indices:
        try:
            for row in get_shard(i):
                yield row
        except Exception as e:
            print(f"  [stream] skipping bad file-shard {i} ({type(e).__name__}: {str(e)[:90]})")
            continue


def _iter_source(entry, optout=frozenset(), limit=None, skip_docs=0, shard_count=1, shard_index=0):
    """Yield docs for one manifest source so sources can be budgeted and sharded independently.
    License metadata travels with each document as provenance.

    If entry['datasets'] is a list of dataset IDs (or {id:, config:, data_dir:} dicts),
    iterates all of them, yielding from each — the parent entry's license/content_type/role
    are inherited by each sub-dataset.

    shard_count>1 splits the source's underlying files across N disjoint streams via
    IterableDataset.shard() — run N processes (across both machines) with shard_index=0..N-1
    to download ONE big source in parallel (each pulls a different subset of parquet files).

    HF's `datasets` already retries transient chunk reads internally; we just raise the per-read
    timeout (default 10s is too low for Comma's big .gz chunks) so those internal retries have time
    to succeed instead of surfacing. No outer retry wrapper — it would re-stream from the top on a
    deep failure (a perf footgun) and duplicate HF's own logic."""

    # Multi-dataset entry: chain all sub-datasets under one manifest entry. skip_docs is GLOBAL across
    # the whole chain (not per sub-dataset) — each sub streams from its start (skip=0) and we drop rows
    # until the combined count passes skip_docs, so resume skips exactly N docs total.
    datasets = entry.get("datasets")
    if datasets:
        seen = 0
        for sub in datasets:
            sub_entry = dict(entry)
            if isinstance(sub, str):
                sub_entry["id"] = sub
            else:
                sub_entry.update(sub)
            sub_entry.pop("datasets", None)  # prevent infinite recursion
            for row in _iter_source(sub_entry, optout, limit, 0, shard_count, shard_index):
                seen += 1
                if seen <= skip_docs:
                    continue
                yield row
        return
    import os as _os
    from datasets import load_dataset

    _os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    _os.environ.setdefault(
        "HF_XET_HIGH_PERFORMANCE", "1"
    )  # HF deprecated HF_HUB_ENABLE_HF_TRANSFER
    ds_id = entry["id"]
    if entry.get("verify_hf_path"):
        print(f"  [stream] skipping {ds_id} (verify_hf_path=True — confirm on HF first)")
        return
    config = entry.get("config")
    split = entry.get("split", "train")
    data_dir = entry.get("data_dir")
    data_files = entry.get("data_files")
    is_local = ds_id.startswith("local/")
    try:
        load_kw = {
            "split": split,
            "streaming": True,
            "trust_remote_code": bool(entry.get("trust_remote_code", False)),
        }
        if config:
            load_kw["name"] = config
        if data_dir:
            load_kw["data_dir"] = data_dir
        if data_files:
            if is_local:
                # Local file: resolve relative to the manifest's out dir (data/ by default)
                out_dir = os.environ.get(
                    "DATAPREP_OUT_DIR",
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data",
                )
                base = os.path.dirname(out_dir) if out_dir.endswith("-dd") else out_dir
                local_path = os.path.join(base, data_files)
                ds = load_dataset("json", data_files=local_path, **load_kw)
            else:
                # HF dataset blob
                ds = load_dataset(
                    "json",
                    data_files=f"https://huggingface.co/datasets/{ds_id}/resolve/main/{data_files}",
                    **load_kw,
                )
        else:
            ds = load_dataset(ds_id, **load_kw)
    except Exception as e:
        print(f"  [stream] WARNING: {ds_id} failed to load ({e})")
        return
    # File-level resilient streaming + sharding. Each underlying data file becomes its own sub-shard
    # (ds.shard keeps datasets' correct config/split resolution); we iterate it inside try/except and
    # skip the whole file if it errors. Cross-machine sharding = take this index's files (i::count).
    try:
        n = ds.n_shards if hasattr(ds, "shard") else 1
    except Exception:
        n = 1
    indices = (
        list(range(shard_index, n, shard_count))
        if shard_count and shard_count > 1
        else list(range(n))
    )
    print(
        f"  [stream] {ds_id} -> {entry.get('role', '')} "
        f"({len(indices)}/{n} files"
        f"{' shard %d/%d' % (shard_index, shard_count) if shard_count > 1 else ''}"
        f"{', skipping ' + str(skip_docs) + ' docs' if skip_docs else ''})"
    )

    def _get_shard(i):
        return ds.shard(num_shards=n, index=i) if n > 1 else ds

    gidx = 0
    count = 0
    for row in _stream_file_shards(_get_shard, indices):
        gidx += 1
        if gidx <= skip_docs:  # resume: skip already-processed docs
            continue
        text_fields = entry.get("text_fields", None)
        text_field = entry.get("text_field", None)
        text_subfield = entry.get("text_subfield", None)
        if text_fields:
            parts = []
            for key in text_fields:
                val = row.get(key, "")
                if val:
                    parts.append(f"{key}: {val}")
            text = "\n\n".join(parts)
        elif text_field:
            raw = row.get(text_field, "")
            if text_subfield and isinstance(raw, str) and raw.strip():
                try:
                    parsed = json.loads(raw)
                    text = str(parsed.get(text_subfield, ""))
                except (json.JSONDecodeError, TypeError, AttributeError):
                    text = str(raw)
            else:
                text = str(raw)
        else:
            text = row.get("text") or row.get("content") or row.get("title", "")
            if not text and "messages" in row:
                text = "\n\n".join(
                    f"{m.get('role', 'user')}: {m.get('content', '')}" for m in row["messages"]
                )
            if not text and "conversations" in row:
                text = "\n\n".join(
                    f"{m.get('from', 'user')}: {m.get('value', '')}" for m in row["conversations"]
                )
            if not text and "instruction" in row:
                text = f"Instruction: {row.get('instruction', '')}\nInput: {row.get('input', '')}\nOutput: {row.get('output', '')}"
            if not text and "code" in row:
                text = row.get("code", "")
            if not text:
                text = "\n\n".join(
                    str(v)
                    for k, v in row.items()
                    if isinstance(v, str)
                    and k not in ("id", "url", "meta", "repo_name", "license", "file_path")
                )
        url = row.get("url") or row.get("repo_name") or ""
        lic = row.get("license") or entry.get("license", "")
        if not text:
            text = ""  # pass empty through pipeline (will be quality-filtered)
        if opted_out({"url": url}, optout):
            continue
        yield {
            "id": ds_id,
            "text": text,
            "license": lic,
            "url": url,
            "content_type": entry.get("content_type", "prose"),
        }
        count += 1
        if limit and count >= limit:
            return


def stream_real(manifest_path, limit=None, filter_cfg=None):
    """Stream documents from every Hugging Face dataset listed in the manifest."""
    import yaml

    man = yaml.safe_load(open(manifest_path))
    optout = set((filter_cfg or {}).get("optout", []))
    for entry in man.get("pretrain", []):
        yield from _iter_source(entry, optout, limit)


# --------------------------------------------------------------------------
# Self-test - verifies every stage drops exactly the right documents
# --------------------------------------------------------------------------
def synthetic_corpus():
    base = (
        "The people who build their own tools learn the shape of their own freedom. "
        "A small model trained on consented text can still reason about the world, "
        "decompose a hard problem into parts, and explain each step it takes plainly."
    )
    near = base.replace("freedom", "liberty").replace("plainly", "clearly")  # ~0.95 Jaccard
    other = (
        "Spinning wheels turned cotton into thread in village homes for centuries, "
        "long before factories, letting families clothe themselves without a mill. "
        "Open weights aim at the same independence for computation today everywhere."
    )
    third = (
        "Public libraries digitized millions of out-of-copyright pages over decades, "
        "turning fragile paper into open text that anyone may read, copy and study. "
        "That patient archival work is the quiet backbone of every ethical training set."
    )
    fourth = (
        "A clearly documented source manifest helps researchers reproduce a corpus while "
        "preserving provenance and attribution requirements for each input collection. "
        "This synthetic document exercises the pipeline with a non-commercial license tag."
    )
    bench = "what is the boiling point of water at standard atmospheric pressure in celsius degrees exactly"
    return [
        {"id": "d1_good", "license": "cc-by", "url": "https://ok.org/a", "text": base},
        {"id": "d2_nc", "license": "cc-by-nc", "url": "https://x.org/b", "text": fourth},
        {"id": "d3_exact", "license": "cc-by", "url": "https://ok.org/c", "text": base},
        {"id": "d4_near", "license": "cc-by", "url": "https://ok.org/d", "text": near},
        {
            "id": "d5_lowq",
            "license": "cc0",
            "url": "https://ok.org/e",
            "text": "### {} <> | ^ ~ ###",
        },
        {
            "id": "d6_pii",
            "license": "mit",
            "url": "https://ok.org/f",
            "text": third + " contact email me@test.com or call 415 555 1234 anytime",
        },
        {
            "id": "d7_bench",
            "license": "cc-by",
            "url": "https://ok.org/g",
            "text": "Quick quiz follows. " + bench + " Please answer in one number only please.",
        },
        {
            "id": "d8_optout",
            "license": "cc-by",
            "url": "https://blocked.com/h",
            "text": base + " variant tail",
        },
        {"id": "d9_good2", "license": "public domain", "url": "https://ok.org/i", "text": other},
    ]


def selftest():
    print("CHARKHA dataprep self-test")
    out = os.path.join(sys.path[0] or ".", "_selftest_shards")
    cfg = {
        "optout": ["blocked.com"],
        "bench": [
            "what is the boiling point of water at standard atmospheric pressure in celsius degrees exactly"
        ],
        "decon_n": 8,
        "min_words": 10,
        "shard_tokens": 1000,
        "tokenizer": None,
    }
    index, kept = run_pipeline(synthetic_corpus(), out, cfg)
    a = index["accounting"]
    print("  accounting:", json.dumps(a))
    print("  kept docs :", kept)
    checks = {
        "manifest-selected document kept": "d2_nc" in kept,
        "optout drop (blocked.com)": a.get("drop_optout") == 1,
        "quality drop (symbols/short)": a.get("drop_quality") == 1,
        "exact dedup drop": a.get("drop_dedup_exact") == 1,
        "near dedup drop": a.get("drop_dedup_near") == 1,
        "decontam drop": a.get("drop_decontam") == 1,
        "pii redactions >=2": a.get("pii_redactions", 0) >= 2,
        "kept exactly d1,d2,d6,d9": set(kept) == {"d1_good", "d2_nc", "d6_pii", "d9_good2"},
        "license metadata does not add a hidden drop stage": "drop_license" not in a,
    }
    # shard reload round-trip
    total = 0
    for sh in index["shards"]:
        arr = array.array("H")
        with open(os.path.join(out, sh["file"]), "rb") as f:
            arr.fromfile(f, sh["tokens"])
        total += len(arr)
    checks["shard reload == accounting"] = total == index["total_tokens"] > 0
    checks["multiple shards written"] = len(index["shards"]) >= 1
    # PII really gone from d6 output (re-derive its scrubbed text)
    d6 = next(d for d in synthetic_corpus() if d["id"] == "d6_pii")
    scrubbed, _ = pii_scrub(normalize(d6["text"]))
    checks["no raw email remains"] = "@test.com" not in scrubbed and "<EMAIL>" in scrubbed
    # mix enforcement (offline): manifest weights normalize to the token target exactly
    fake = [
        {"id": "a", "weight": 0.35},
        {"id": "b", "weight": 0.20},
        {"id": "c", "weight": 0.20},
        {"id": "d", "weight": 0.07},
        {"id": "e", "weight": 0.07},
        {"id": "f", "weight": 0.05},
    ]  # sum 0.94
    bud = _compute_budgets(fake, 1_000_000)
    checks["mix budgets sum to target"] = abs(sum(bud.values()) - 1_000_000) <= len(fake)
    checks["mix proportions preserved"] = (
        bud["a"] == max(bud.values()) and bud["a"] > bud["b"] > bud["d"] >= bud["f"]
    )
    checks["mix normalizes 0.94 -> 1.0"] = abs(bud["a"] - round(0.35 / 0.94 * 1_000_000)) <= 2

    # resilient FILE-level streaming: a bad file-shard must be skipped, not crash the run nor end it
    def _mk_shards():
        def get(i):
            if i == 1:  # file 1 yields one row then dies

                def bad():
                    yield ("row", 1)
                    raise ValueError("corrupt file")

                return bad()
            return [("f%d" % i, j) for j in range(3)]  # files 0,2 yield 3 rows each

        return get

    fr = list(_stream_file_shards(_mk_shards(), [0, 1, 2]))
    checks["resilient stream skips a bad file-shard"] = len(fr) == 3 + 1 + 3
    # canon_license must be IDEMPOTENT: a manifest's declared canonical license (e.g. the hyphenated
    # 'public-domain' that Gutenberg's entry uses, since its rows carry no license field) must
    # round-trip to itself, not fall through to 'unknown' and get denied. Regression for the
    # silent-0-tokens Gutenberg bug.
    checks["canon_license idempotent on canonical forms"] = (
        canon_license("public-domain") == "public-domain"
        and canon_license("cc-by-sa") == "cc-by-sa"
        and canon_license("apache-2.0") == "apache-2.0"
        and canon_license("public domain") == "public-domain"
    )  # space form still maps too
    # digit-split (math-from-the-start): one token per digit, and join is its exact inverse.
    checks["digit_split isolates every digit"] = (
        split_digits("abc1234.56xy") == "abc 1 2 3 4. 5 6xy"
        and join_digits(split_digits("order 1234 of 56")) == "order 1234 of 56"
    )
    # digit-split actually changes the token stream (more tokens for a numeric doc); byte tokenizer.
    enc_plain, _ = load_tokenizer(None, digit_split=False)
    enc_split, _ = load_tokenizer(None, digit_split=True)
    checks["digit_split lengthens a numeric doc"] = len(enc_split("12345")) > len(
        enc_plain("12345")
    )
    # quality_filter must be code-aware: a code-like doc should PASS under 'code', FAIL under 'prose'.
    # The prose path's symbol_ratio (#{}[]<>|^~ >10%) and mean_word_len (2.5-12) are tuned for prose
    # and wrongly flag normal code (braces, brackets, operators, short variable names).
    code_text = "def foo(x):\n    return x+1\n\nclass Bar:\n    def __init__(self,y):\n        self.y={y}\n\ndef baz():\n    if x>0:\n        return [x]\n    else:\n        return None\n\nresult=foo(3);print(result)"
    code_ok_prose, code_reason_prose = quality_filter(
        normalize(code_text), min_words=10, content_type="prose"
    )
    code_ok_code, code_reason_code = quality_filter(
        normalize(code_text), min_words=10, content_type="code"
    )
    checks["quality_filter code doc dropped as prose"] = not code_ok_prose
    checks["quality_filter code doc kept as code"] = code_ok_code
    # Opt-in provenance tags must (a) lead every kept document's token stream,
    # (b) be recorded in the index, (c) NOT change which docs pass cleaning, (d) leave the
    # default (no tag) byte-identical.
    enc_b, _ = load_tokenizer(None)
    tag_str = "<src:wikipedia>\n"
    tag_ids = enc_b(tag_str)[:-1]  # drop the trailing EOS the byte tokenizer appends
    cfg_tag = dict(cfg)
    cfg_tag["src_tag"] = tag_str
    out_tag = os.path.join(sys.path[0] or ".", "_selftest_prov")
    idx_tag, kept_tag = run_pipeline(synthetic_corpus(), out_tag, cfg_tag)
    checks["provenance: src_tag recorded in index"] = idx_tag.get("src_tag") == tag_str
    checks["provenance: same docs kept as untagged"] = set(kept_tag) == set(kept)
    # first kept doc's shard must open with the tag ids
    first = array.array("H")
    with open(os.path.join(out_tag, idx_tag["shards"][0]["file"]), "rb") as f:
        first.fromfile(f, len(tag_ids) + 4)
    checks["provenance: tag ids lead the stream"] = list(first[: len(tag_ids)]) == tag_ids
    checks["provenance: tagged stream is longer"] = idx_tag["total_tokens"] > index["total_tokens"]
    checks["provenance: default (no tag) unchanged"] = index.get("src_tag") is None
    import shutil as _shutil

    _shutil.rmtree(out_tag, ignore_errors=True)
    # multi-dataset entries (a 'datasets:' list, no 'id' — CommonCorpus) must budget without KeyError
    multi = [
        {"datasets": ["Org/A", "Org/B"], "role": "breadth", "weight": 0.5},
        {"id": "Org/C", "weight": 0.5},
    ]
    checks["_entry_id handles datasets list"] = (
        _entry_id(multi[0]) == "Org/A+Org/B" and _entry_id(multi[1]) == "Org/C"
    )
    try:
        mbud = _compute_budgets(multi, 1_000_000)
        checks["budgets work on datasets-list entry"] = (
            mbud["Org/A+Org/B"] == 500_000 and mbud["Org/C"] == 500_000
        )
    except Exception as e:
        checks["budgets work on datasets-list entry"] = False
        print(f"  [budgets datasets-list] {type(e).__name__}: {e}")
    # ShardWriter resume must ADOPT existing intact shards (not clobber from shard_00000)
    rtmp = os.path.join(sys.path[0] or ".", "_selftest_resume")
    os.makedirs(rtmp, exist_ok=True)
    for f in os.listdir(rtmp):
        os.remove(os.path.join(rtmp, f))
    w1 = ShardWriter(rtmp, shard_tokens=100)
    w1.add(list(range(250)))  # -> 2 full shards (00000,00001) + 50 buffered
    w1.close()  # flush -> shard_00002 (50)
    _write_outputs(rtmp, w1, 257, defaultdict(int), {})
    n_before = len(w1.shards)
    w2 = ShardWriter(rtmp, shard_tokens=100, resume=True)
    checks["resume adopts existing shards"] = len(w2.shards) == n_before and w2.total == 250
    w2.add(list(range(100)))  # appends -> shard_00003
    w2.close()
    new_files = sorted(f for f in os.listdir(rtmp) if f.endswith(".bin"))
    checks["resume appends without clobber"] = (
        "shard_00000.bin" in new_files
        and "shard_00003.bin" in new_files
        and len(new_files) == n_before + 1
    )
    print()
    ok = True
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= passed
    print("\nSELFTEST", "PASS - pipeline drops exactly the right docs" if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------


def _entry_id(entry):
    """Stable budget/progress/accounting key for a manifest entry. Single-dataset entries use their
    'id'; multi-dataset entries (a 'datasets:' list, no top-level 'id', e.g. CommonCorpus) get a
    composite key joining the sub-dataset names so the whole entry shares one budget + progress slot."""
    if entry.get("id"):
        return entry["id"]
    ds = entry.get("datasets") or []
    names = [
        d if isinstance(d, str) else (d.get("id", "") if isinstance(d, dict) else "") for d in ds
    ]
    return "+".join(n for n in names if n) or entry.get("role", "source")


def _write_outputs(out_dir, writer, vocab, acct, progress, tokenizer=None):
    """Persist index.json + progress.json. Called at every shard-flush boundary (crash-safe resume)
    and at the end. total_tokens is the sum of the flushed shards, so it always matches `shards`."""
    total = sum(s["tokens"] for s in writer.shards)
    index = {
        "vocab_size": vocab,
        "total_tokens": total,
        "tokenizer": tokenizer,
        "shards": list(writer.shards),
        "accounting": dict(acct),
    }
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    with open(os.path.join(out_dir, "progress.json"), "w") as f:
        json.dump(progress, f, indent=2)
    return index


def _compute_budgets(entries, total_tokens):
    """Per-source token budgets from normalized manifest weights (weights need not sum to 1)."""
    weights = {_entry_id(e): float(e.get("weight", 0) or 0) for e in entries}
    s = sum(weights.values()) or 1.0
    return {k: int(round(v / s * total_tokens)) for k, v in weights.items()}


def _partition_sources(entries, n):
    """Split sources across n groups, greedily balancing by weight (longest-processing-time)."""
    n = max(1, int(n or 1))
    groups, loads = [[] for _ in range(n)], [0.0] * n
    for e in sorted(entries, key=lambda e: float(e.get("weight", 0) or 0), reverse=True):
        i = loads.index(min(loads))
        groups[i].append(e)
        loads[i] += float(e.get("weight", 0) or 0)
    return groups


def _run_source_group(entries, out_dir, cfg, budgets, skip_docs=None, resume=False):
    """Run the full pipeline over a group of sources into out_dir, capping each source at
    budgets[id] tokens (None/0 = unbounded). Global dedup within the group; one source at a
    time, so no shared-stream bandwidth multiplication. Writes index.json + progress.json at every
    shard-flush boundary so a killed run is resumable (resume=True adopts the on-disk shards instead
    of overwriting them; skip_docs replays the docs since the last flushed shard)."""
    os.makedirs(out_dir, exist_ok=True)
    optout = set(cfg.get("optout", []))  # explicit opt-outs only (empty = drop nothing)
    dedup = Deduper(cfg.get("dedup_mode", "near"))
    decon = Decontaminator(cfg.get("bench", []), n=cfg.get("decon_n", 8))
    encode, vocab = load_tokenizer(cfg.get("tokenizer"), cfg.get("digit_split", False))
    writer = ShardWriter(
        out_dir, cfg.get("shard_tokens", 100_000_000), resume=resume, vocab_size=vocab
    )
    mw, limit = cfg.get("min_words", 10), cfg.get("limit")
    acct = defaultdict(int)
    progress = {}
    if resume:  # accumulate stats/token-counts onto the prior partial run's accounting
        idx_path = os.path.join(out_dir, "index.json")
        if os.path.isfile(idx_path):
            try:
                for k, v in json.load(open(idx_path)).get("accounting", {}).items():
                    acct[k] += v
            except Exception:
                pass
    last_nshards = len(writer.shards)
    for entry in entries:
        ds_id = _entry_id(entry)
        nskip = (skip_docs or {}).get(ds_id, 0)
        budget = (budgets or {}).get(ds_id)

        prev_tokens = acct.get(f"tokens::{ds_id}", 0)
        if prev_tokens == 0 and len(entries) == 1:
            prev_tokens = writer.total
            acct[f"tokens::{ds_id}"] = prev_tokens

        src_tokens = 0
        docs_seen = 0

        if budget and prev_tokens >= budget:
            print(f"  [budget] {ds_id}: {prev_tokens:,}/{budget:,} tokens - already reached budget")
            # Ensure progress is saved even if we skip
            progress[ds_id] = nskip
            _write_outputs(out_dir, writer, vocab, acct, progress, tokenizer=cfg.get("tokenizer"))
            continue

        for doc in _iter_source(
            entry,
            optout,
            limit,
            skip_docs=nskip,
            shard_count=cfg.get("shard_count", 1),
            shard_index=cfg.get("shard_index", 0),
        ):
            docs_seen += 1
            acct["seen"] += 1
            status, ids, npii = _process_doc(doc, optout, dedup, decon, encode, mw)
            acct["pii_redactions"] += npii
            if status == "kept":
                writer.add(ids)
                acct["kept"] += 1
                src_tokens += len(ids)
                if (
                    len(writer.shards) > last_nshards
                ):  # a full shard just flushed -> crash-safe checkpoint
                    last_nshards = len(writer.shards)
                    progress[ds_id] = docs_seen + nskip
                    _write_outputs(
                        out_dir, writer, vocab, acct, progress, tokenizer=cfg.get("tokenizer")
                    )
                if budget and (prev_tokens + src_tokens) >= budget:
                    print(
                        f"  [budget] {ds_id}: {prev_tokens + src_tokens:,}/{budget:,} tokens - next source"
                    )
                    break
            else:
                acct[status] += 1
        acct[f"tokens::{ds_id}"] = prev_tokens + src_tokens
        # cumulative docs consumed (this run's seen + docs skipped on resume)
        progress[ds_id] = docs_seen + nskip
        _write_outputs(out_dir, writer, vocab, acct, progress)
    writer.close()  # flush the trailing partial shard, then write the final consistent index
    return _write_outputs(out_dir, writer, vocab, acct, progress, tokenizer=cfg.get("tokenizer"))


def build_corpus(manifest_path, out_dir, cfg, total_tokens=None, workers=1, resume=False):
    """Mix-enforced corpus build. Source-sharded across `workers` (each worker streams a
    DISJOINT set of sources - correct parallelism, unlike round-robin over one shared stream).
    total_tokens set -> per-source token budgets from manifest weights; else uncapped (--limit).
    resume=True -> loads progress.json to skip already-processed docs per source."""
    import yaml

    man = yaml.safe_load(open(manifest_path))
    entries = [e for e in man.get("pretrain", []) if not e.get("verify_hf_path")]
    budgets = _compute_budgets(entries, total_tokens) if total_tokens else {}
    if not budgets:
        for e in entries:
            b = e.get("budget_B")
            if b:
                budgets[_entry_id(e)] = int(b * 1e9)
    if budgets:
        print("  [mix] per-source token budgets (from manifest weights):")
        for e in entries:
            print(f"    {_entry_id(e)}: {budgets.get(_entry_id(e), 0):,}")
    # --- resume support ---
    skip_docs = None
    if resume:
        prog_path = os.path.join(out_dir, "progress.json")
        if os.path.exists(prog_path):
            skip_docs = json.load(open(prog_path, "r"))
            total_skip = sum(skip_docs.values())
            print(f"  [resume] skipping {total_skip:,} total docs from {len(skip_docs)} sources")
        else:
            print("  [resume] progress.json not found - starting from scratch")
    # ----------------------
    if workers <= 1:
        idx = _run_source_group(entries, out_dir, cfg, budgets, skip_docs, resume=resume)
        print(
            f"\n=== DONE === {len(idx['shards'])} shards, {idx['total_tokens']:,} tokens "
            f"({idx['total_tokens'] / 1e9:.2f}B)"
        )
        print(json.dumps(idx["accounting"], indent=2))
        return idx
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    if resume:
        print("  [resume] note: --resume with --workers only uses progress.json from out_dir")
        print("  [resume] (skip_docs passed to each worker; per-worker progress not merged)")
    groups = _partition_sources(entries, workers)
    procs = []
    for w, grp in enumerate(groups):
        wdir = os.path.join(out_dir, f"worker_{w:03d}")
        pr = mp.Process(target=_run_source_group, args=(grp, wdir, cfg, budgets, skip_docs, resume))
        pr.start()
        procs.append(pr)
    for pr in procs:
        pr.join()
    _merge_workers(out_dir, len(groups), cfg)


def _merge_workers(out_dir, n_workers, cfg):
    """Collect shards from all workers into the parent dir, write unified index."""
    import shutil

    all_shards = []
    total_tokens = 0
    merged_accounting = {}
    vocab_size = 0
    tokenizer_name = None
    exit_progress = {}
    for w in range(n_workers):
        wdir = os.path.join(out_dir, f"worker_{w:03d}")
        idx_path = os.path.join(wdir, "index.json")
        if not os.path.exists(idx_path):
            print(f"[merge] WARNING: worker {w} produced no output")
            continue
        idx = json.load(open(idx_path, "r"))
        vocab_size = vocab_size or int(idx.get("vocab_size") or 0)
        tokenizer_name = tokenizer_name or idx.get("tokenizer")
        # Re-number shards into parent directory
        for s in idx["shards"]:
            src = os.path.join(wdir, s["file"])
            dst_name = f"shard_{len(all_shards):05d}.bin"
            shutil.move(src, os.path.join(out_dir, dst_name))
            all_shards.append({"file": dst_name, "tokens": s["tokens"]})
            total_tokens += s["tokens"]
        for k, v in idx.get("accounting", {}).items():
            merged_accounting[k] = merged_accounting.get(k, 0) + v
        prog_path = os.path.join(wdir, "progress.json")
        if os.path.exists(prog_path):
            try:
                exit_progress.update(json.load(open(prog_path, "r")))
            except Exception:
                pass
    # Write unified index
    unified = {
        "vocab_size": vocab_size,
        "total_tokens": total_tokens,
        "tokenizer": tokenizer_name,
        "shards": all_shards,
        "accounting": merged_accounting,
    }
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(unified, f, indent=2)
    with open(os.path.join(out_dir, "progress.json"), "w") as f:
        json.dump(exit_progress, f, indent=2)
    print(
        f"\n=== DONE === {len(all_shards)} shards, {total_tokens:,} tokens "
        f"({total_tokens / 1e9:.1f}B)"
    )
    print(json.dumps(merged_accounting, indent=2))


if __name__ == "__main__":
    try:
        # Line-buffer stdout so progress shows promptly even when piped to `tee logs/x.log`
        # (a pipe makes Python block-buffer by default -> logs look frozen while shards write fine).
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    p = argparse.ArgumentParser(description="CHARKHA dataprep - streaming data pipeline")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--manifest", type=str, default=None, help="user-supplied YAML source manifest")
    p.add_argument("--out", type=str, default="data")
    p.add_argument("--limit", type=int, default=None, help="docs per source (smoke test cap)")
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel workers (source-sharded: each streams a disjoint set of sources)",
    )
    p.add_argument(
        "--target-tokens",
        type=int,
        default=None,
        help="total token budget; enforces the per-source mix from manifest weights",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="load progress.json to skip already-processed docs (300GB batch mode)",
    )
    p.add_argument(
        "--dedup",
        choices=["near", "exact", "none"],
        default="near",
        help="dedup mode. 'near'=sha1+MinHash-LSH (default, but ~88%% of CPU and OOMs "
        "at billions of tokens); 'exact'=sha1 only (fast, bounded — use for big "
        "curated runs); 'none'=skip dedup",
    )
    p.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="split each source into N parallel streams (run N procs, shard-index 0..N-1)",
    )
    p.add_argument(
        "--shard-index", type=int, default=0, help="this process index in [0, shard-count)"
    )
    p.add_argument(
        "--digit-split",
        action="store_true",
        help="math-from-the-start: put a space before every digit so the tokenizer emits "
        "ONE token per digit (place-value arithmetic instead of arbitrary multi-digit "
        "BPE merges). Recommended for the math-capable run. serve.py must use the same "
        "split_digits/join_digits so train and inference agree. Ignored (with a notice) "
        "when --tokenizer already isolates digits natively (e.g. train_tokenizer.py output).",
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "charkha_tokenizer.json"
        ),
        help="a tokenizer hub name or a local tokenizer.json path "
        "(default: the bundled 65,535-token ByteLevel BPE tokenizer, trained with "
        "scripts/train_tokenizer.py). The absolute default path lets serve.py and "
        "train.py resolve it regardless of cwd. --digit-split is auto-skipped when "
        "the selected tokenizer already isolates digits. Training and serving must "
        "use the same tokenizer and vocab_size.",
    )
    a = p.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.manifest:
        p.error("--manifest is required; start from configs/sources.example.yaml")
    # Fast downloads: hf_transfer parallelizes chunk downloads (Rust multi-threaded).
    # HF_XET_HIGH_PERFORMANCE for Xet-backed repos (harmless fallback).
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ["DATAPREP_OUT_DIR"] = os.path.abspath(a.out)
    cfg = {
        "optout": [],
        "bench": [],
        "tokenizer": a.tokenizer,
        "shard_tokens": 100_000_000,
        "min_words": 50,
        "limit": a.limit,
        "dedup_mode": a.dedup,
        "shard_count": a.shard_count,
        "shard_index": a.shard_index,
        "digit_split": a.digit_split,
    }

    if a.target_tokens or a.workers > 1 or a.resume:
        # mix-enforced and/or parallel build (source-sharded across workers)
        build_corpus(a.manifest, a.out, cfg, a.target_tokens, a.workers, a.resume)
        # GUARDRAIL (2026-07-06): a source whose manifest asked for a positive token
        # budget but produced 0 tokens is a SILENT FAILURE (dead repo, wrong config/
        # split, gated). Exiting non-zero stops the runner from marking it 'completed'
        # and writing an empty-but-valid index that every later pass skips. This is the
        # bug that quietly lost ~100B tokens across 21 sources.
        try:
            import yaml as _yaml

            want = 0.0
            for _e in (_yaml.safe_load(open(a.manifest, encoding="utf-8")) or {}).get(
                "pretrain", []
            ):
                want += float(_e.get("budget_B", 0) or 0)
            idxp = os.path.join(a.out, "index.json")
            got = json.load(open(idxp)).get("total_tokens", 0) if os.path.exists(idxp) else 0
            if want > 0 and got == 0:
                print(
                    f"  [guardrail] FAIL: budget {want}B requested but 0 tokens produced "
                    f"-> not marking complete (see load WARNING above)."
                )
                sys.exit(3)
        except SystemExit:
            raise
        except Exception as _ge:
            print(f"  [guardrail] (skipped: {_ge})")
    else:
        # simple single-worker stream (smoke test / no mix enforcement)
        idx, _ = run_pipeline(stream_real(a.manifest, a.limit), a.out, cfg)
        print(json.dumps(idx["accounting"], indent=2), "\n-> tokens:", idx["total_tokens"])

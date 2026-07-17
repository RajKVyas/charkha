"""CHARKHA serving utilities — retrieval ingestion, checkpoint loading."""

import os
import json
import re
import shutil
import torch
from charkha import Charkha, CharkhaConfig
from _verify import ByteTokenizer, DigitSplitTokenizer, load_tokenizer_for
from retrieval import Datastore


def _decode_ids_for_retrieval(tok, ids):
    try:
        return tok.decode([int(i) for i in ids]).replace("<|endoftext|>", "\n").strip()
    except TypeError:  # transformers tokenizers may accept kwargs only
        return (
            tok.decode([int(i) for i in ids], skip_special_tokens=False)
            .replace("<|endoftext|>", "\n")
            .strip()
        )


def _add_shard_dir_to_datastore(
    ds, path, tok=None, max_passages=20000, passage_tokens=192, stride_tokens=768
):
    """Add deterministic windows from a dataprep shard dir to the retrieval store.

    This gives serve-time RAG over the same on-disk curriculum/pretraining data without moving any
    tensors to GPU. The cap is deliberate: full corpora can be hundreds of millions of tokens, so the
    default builds a broad lexical sample; set --retrieve-max-passages 0 to index every window.
    """
    import numpy as np
    from dataprep import shard_format_for_vocab

    idx_path = os.path.join(path, "index.json")
    with open(idx_path, encoding="utf-8") as f:
        index = json.load(f)
    if tok is None:
        tok = load_tokenizer_for(index.get("tokenizer"))
    _, bpt = shard_format_for_vocab(index.get("vocab_size"))
    dtype = np.uint16 if bpt == 2 else np.uint32
    shards = index.get("shards", [])
    if not shards:
        return 0
    passage_tokens = max(16, int(passage_tokens))
    stride_tokens = max(passage_tokens, int(stride_tokens))
    total_windows = 0
    for sh in shards:
        total_windows += max(0, (int(sh.get("tokens", 0)) - passage_tokens) // stride_tokens + 1)
    keep_every = max(1, total_windows // max(1, int(max_passages))) if max_passages else 1
    added = seen = 0
    base = os.path.basename(os.path.normpath(path))
    for sh in shards:
        fp = os.path.join(path, sh["file"])
        arr = np.memmap(fp, dtype=dtype, mode="r")
        limit = max(0, len(arr) - passage_tokens)
        for off in range(0, limit + 1, stride_tokens):
            if max_passages and added >= max_passages:
                return added
            if seen % keep_every:
                seen += 1
                continue
            seen += 1
            text = _decode_ids_for_retrieval(tok, arr[off : off + passage_tokens])
            if len(text.split()) < 8:
                continue
            ds.add(text, source=f"{base}:{sh['file']}@{off}", license="training-data")
            added += 1
    return added


def build_datastore(path, use_dense=False, tokenizer=None, max_passages=20000, passage_tokens=192):
    """Build a provenanced retrieval datastore from:
      * .jsonl  — one object per line: {"text": ..., "source": ..., "license": ...}
      * .txt    — paragraphs split on blank lines; source defaults to the filename
      * shard dir — dataprep output with index.json + shard_*.bin, decoded into sampled passages
    Pure-stdlib BM25 by default; --dense fuses a sentence-transformers index if installed."""

    ds = Datastore(use_dense=use_dense)
    n = 0
    paths = [p for p in re.split(r"[,;]", path) if p.strip()]
    for one in paths:
        one = one.strip()
        base = os.path.basename(os.path.normpath(one))
        if os.path.isdir(one):
            if not os.path.exists(os.path.join(one, "index.json")):
                raise FileNotFoundError(f"{one} is a directory but has no index.json")
            n += _add_shard_dir_to_datastore(
                ds, one, tok=tokenizer, max_passages=max_passages, passage_tokens=passage_tokens
            )
            continue
        with open(one, "r", encoding="utf-8") as f:
            if one.endswith(".jsonl"):
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        o = json.loads(line)
                    except Exception:
                        continue
                    txt = (o.get("text") or "").strip()
                    if txt:
                        ds.add(txt, source=o.get("source", base), license=o.get("license", ""))
                        n += 1
            else:
                for para in re.split(r"\n\s*\n", f.read()):
                    para = para.strip()
                    if para:
                        ds.add(para, source=base, license="")
                        n += 1
    ds.finalize()
    print(
        f"[retrieval] datastore: {n} passages from {path}{' (+dense)' if use_dense else ' (BM25)'}"
    )
    return ds


def load_model(ckpt_path, device, toy=False, digit_split=False):

    if str(device).startswith("cuda"):
        # TF32 matmuls: faster decode on Ampere+ for fp32 paths, precision loss is far below
        # sampling noise. Mirrors the train.py setting so serve numerics match training.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if toy or not ckpt_path:
        cfg = CharkhaConfig.toy()
        tok = ByteTokenizer()
        return Charkha(cfg).to(device), (DigitSplitTokenizer(tok) if digit_split else tok), cfg
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = CharkhaConfig.from_dict(ck["cfg"]) if isinstance(ck["cfg"], dict) else ck["cfg"]
    model = Charkha(cfg).to(device)
    state = ck.get("model", ck)
    # Prefer the EMA weights when the run tracked them (train.py --ema): the noise-averaged
    # iterate consistently serves better than the raw last step. CHARKHA_RAW_WEIGHTS=1 opts out.
    ema = (ck.get("meta") or {}).get("ema")
    if ema and not os.environ.get("CHARKHA_RAW_WEIGHTS"):
        state = dict(state)
        ema = {n.removeprefix("_orig_mod."): t for n, t in ema.items()}
        used = {n: t.to(state[n].dtype) for n, t in ema.items() if n in state}
        state.update(used)
        print(
            f"[load] serving EMA weights ({len(used)}/{len(state)} tensors; "
            "set CHARKHA_RAW_WEIGHTS=1 for the raw iterate)"
        )
    model.load_state_dict(state)
    # Use whatever tokenizer produced this checkpoint's training shards (recorded by train.py from
    # dataprep's index.json into cfg.tokenizer_name) — falls back to gpt-neox-20b for legacy
    # checkpoints predating that field. Inference tokenization then always matches training.
    tok = load_tokenizer_for(getattr(cfg, "tokenizer_name", None))
    # Must match the training tokenization: a digit-split-trained model needs digit-split inference.
    return model, (DigitSplitTokenizer(tok) if digit_split else tok), cfg


# --------------------------------------------------------------------------
# REPL
# --------------------------------------------------------------------------


def _ingest(session, path, inbox=None):
    """Pull a text file (or every .txt/.md in a directory) into BOTH knowledge systems:
    the live retrieval datastore (available to the very next turn) and the consolidation
    inbox (data/personal/inbox), where the sleep-time daemon folds it into the weights."""
    if inbox is None:
        inbox = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "personal", "inbox"
        )
    files = (
        [
            os.path.join(path, f)
            for f in sorted(os.listdir(path))
            if f.lower().endswith((".txt", ".md"))
        ]
        if os.path.isdir(path)
        else [path]
        if os.path.isfile(path)
        else []
    )
    if not files:
        print(f"  nothing ingestible at {path} (.txt/.md)")
        return
    if session.retriever is None:
        session.retriever = Datastore()
    n = 0
    for fp in files:
        try:
            text = open(fp, encoding="utf-8", errors="replace").read()
        except OSError as e:
            print(f"  skip {fp}: {e}")
            continue
        for para in text.split("\n\n"):
            if para.strip():
                session.retriever.add(para.strip(), source=os.path.basename(fp), license="personal")
        os.makedirs(inbox, exist_ok=True)
        shutil.copy2(fp, os.path.join(inbox, os.path.basename(fp)))
        n += 1
    session.retriever.finalize()
    print(f"  ingested {n} file(s): retrievable now, consolidated into weights at next sleep")


def repl(session):
    print("CHARKHA ready. /effort N|converge | /ingest PATH | /mem | /quit")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/mem":
            for role, text in session.mem.recall():
                print(f"  {role}: {text[:80]}")
            continue
        if line.startswith("/ingest"):
            arg = line.split(None, 1)
            if len(arg) < 2:
                print("usage: /ingest <file-or-dir>")
            else:
                _ingest(session, os.path.expanduser(arg[1]))
            continue
        if line.startswith("/effort"):
            try:
                val = line.split()[1]
                session.base_effort = "converge" if val == "converge" else int(val)
                print(f"effort -> {session.base_effort}")
            except (IndexError, ValueError):
                print("usage: /effort N   or   /effort converge")
            continue
        r = session.respond(line)
        flag = (
            "  [ABSTAINED]"
            if r["abstained"]
            else ("  [low confidence - verify]" if r["low_confidence"] else "")
        )
        if r["thinking"]:
            print(f"  (thinking) {r['thinking'][:120]}")
        if r["tools"]:
            print(f"  (tools) {r['tools']}")
        if r.get("sources"):
            cites = ", ".join(s["source"] or "source" for s in r["sources"])
            print(f"  (sources) {cites}  [retrieval q={r['retrieval_quality']:.2f}]")
        if r["abstained"] and r["withheld_answer"]:
            print(f"  (withheld guess) {r['withheld_answer'][:120]}")
        print(f"{r['answer']}\n  [effort {r['effort']} | conf {r['confidence']:.2f}]{flag}")


# --------------------------------------------------------------------------
# Self-test - hermetic: stdlib + torch + a fresh toy model, no network, no HF.
# --------------------------------------------------------------------------


__all__ = ["build_datastore", "load_model", "repl"]

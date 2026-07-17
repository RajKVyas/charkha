"""CHARKHA SFT — supervised instruction finetuning with completion-only loss masking.

A base checkpoint trained on plain next-token prediction does NOT reliably follow the chat format
serve.py assumes (user:/assistant:, <thinking>…</thinking>, [[calc: …]]). SFT teaches that format:
we present (prompt, completion) pairs and train the LM loss ONLY on the completion tokens (the prompt
span is masked with IGNORE=-100), so the model learns to *produce* assistant turns, not to re-predict
the user's text. Any instruct mix (OASST, Aya, OpenHermes, Tulu, synthetic) works.

Design choice: SFT computes its masked loss via `model.hidden()` + the tied head, NOT the fused-CE
training path in charkha.forward (which divides by ALL tokens and can't mask). SFT batches are short
and small, so building per-batch logits is fine — and this keeps the 8GB pretraining hot path
untouched (all existing selftests stay green).

  python sft.py --selftest                                  # hermetic: toy model, masking + 1 step
  python sft.py --run --ckpt runs/charkha/release/ckpt.pt --data sft.jsonl --out runs/charkha-sft
"""

from __future__ import annotations
import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

IGNORE = -100  # standard cross-entropy ignore_index — masked positions contribute no loss

# --------------------------------------------------------------------------
# Formatting + masking (pure, torch-free — the testable core)
# --------------------------------------------------------------------------


def has_chat_tokens(tok) -> bool:
    """True iff the tokenizer bakes the CHARKHA chat specials (<|user|>/<|assistant|>) as real
    vocab entries — Sutra-131k / the v8-family tokenizers do. Then SFT must train the SAME special-token format
    serve.py's Session emits, or serving and finetuning formats silently diverge."""
    t2i = getattr(tok, "token_to_id", None)
    try:
        return bool(t2i) and all(t2i(t) is not None for t in ("<|user|>", "<|assistant|>"))
    except Exception:
        return False


def format_example(prompt: str, completion: str, eos: str = "\n", chat_tokens: bool = False):
    """Return (prompt_text, full_text) in CHARKHA chat format. The loss is masked to the part of
    full_text after prompt_text, i.e. the assistant completion. chat_tokens=True uses the native
    special-token format (matching serve.Session on a chat-token tokenizer, terminated with
    <|endoftext|>); False is the legacy plain-text format for tokenizers without the specials."""
    if chat_tokens:
        prompt_text = f"<|user|>{prompt.strip()}<|assistant|>"
        full_text = f"{prompt_text}{completion.strip()}<|endoftext|>"
        return prompt_text, full_text
    prompt_text = f"user: {prompt.strip()}\nassistant:"
    full_text = f"{prompt_text} {completion.strip()}{eos}"
    return prompt_text, full_text


def build_labels(prompt_ids, full_ids):
    """labels = full_ids with the prompt span (first len(prompt_ids) tokens) set to IGNORE, so the
    LM loss trains only on the completion continuation. Robust to a prompt longer than the (truncated)
    full sequence."""
    labels = list(full_ids)
    for i in range(min(len(prompt_ids), len(labels))):
        labels[i] = IGNORE
    return labels


def encode_example(
    tok, prompt: str, completion: str, max_len: int = None, chat_tokens: bool = None
):
    """(input_ids, labels) for one (prompt, completion). input_ids = full token ids; labels mask the
    prompt. Truncates both to max_len (keeping the head — the prompt + start of the answer).
    chat_tokens=None auto-detects from the tokenizer (see has_chat_tokens)."""
    if chat_tokens is None:
        chat_tokens = has_chat_tokens(tok)
    prompt_text, full_text = format_example(prompt, completion, chat_tokens=chat_tokens)
    prompt_ids = tok.encode(prompt_text)
    full_ids = tok.encode(full_text)
    labels = build_labels(prompt_ids, full_ids)
    if max_len:
        full_ids, labels = full_ids[:max_len], labels[:max_len]
    return full_ids, labels


def read_sft_jsonl(path):
    """Yield (prompt, completion) from a JSONL file. Accepts either:
      {"prompt": "...", "completion": "..."}
      {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    For multi-turn `messages`, the last user→assistant pair is used as the (prompt, completion)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if "prompt" in o and "completion" in o:
                yield str(o["prompt"]), str(o["completion"])
            elif "messages" in o:
                msgs = o["messages"]
                last_user = next(
                    (m["content"] for m in reversed(msgs) if m.get("role") == "user"), None
                )
                last_asst = next(
                    (m["content"] for m in reversed(msgs) if m.get("role") == "assistant"), None
                )
                if last_user is not None and last_asst is not None:
                    yield str(last_user), str(last_asst)


def collate(batch, pad_id=0):
    """Pad a list of (input_ids, labels) to a rectangular batch. input pad = pad_id (compute-only,
    masked from loss); label pad = IGNORE so padding never contributes to the loss."""
    maxlen = max(len(x) for x, _ in batch)
    xs, ys = [], []
    for x, y in batch:
        xs.append(x + [pad_id] * (maxlen - len(x)))
        ys.append(y + [IGNORE] * (maxlen - len(y)))
    return xs, ys


# --------------------------------------------------------------------------
# Masked loss (lazy torch) — via model.hidden() + tied head, not the fused-CE pretraining path
# --------------------------------------------------------------------------


def masked_ce(model, idx, labels, r=None):
    """Completion-masked next-token CE. idx/labels: LongTensors (B,T). Position t predicts token t+1,
    so we score logits[:, :-1] against labels[:, 1:] with ignore_index=IGNORE."""
    import torch.nn.functional as F

    h = model.hidden(idx, r=r)  # (B,T,d) grad-carrying post-norm states
    logits = F.linear(h[:, :-1], model.embed.weight)  # tied head; predict next token
    tgt = labels[:, 1:]
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=IGNORE
    )


def sft_step(model, opt, xs, ys, device, r=None, grad_clip=1.0):
    """One optimizer step over a collated (xs, ys) python batch. Returns the scalar loss."""
    import torch

    idx = torch.tensor(xs, dtype=torch.long, device=device)
    lab = torch.tensor(ys, dtype=torch.long, device=device)
    loss = masked_ce(model, idx, lab, r=r)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    opt.step()
    return float(loss.item())


def run_sft(
    ckpt,
    data_path,
    out,
    *,
    device=None,
    steps=2000,
    lr=1e-5,
    batch_size=4,
    max_len=1024,
    r=None,
    toy=False,
    save_every=500,
):
    """Finetune a checkpoint on (prompt, completion) pairs with completion-only loss. Saves to
    `out/ckpt.pt` in the same dict format train.py/serve.py expect (model + cfg)."""
    import torch
    from serve import load_model

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, cfg = load_model(ckpt, device, toy=toy)
    model.train()
    examples = [encode_example(tok, p, c, max_len) for p, c in read_sft_jsonl(data_path)]
    if not examples:
        raise SystemExit(f"no SFT examples read from {data_path}")
    print(f"[sft] {len(examples)} examples | steps={steps} bs={batch_size} lr={lr} device={device}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    os.makedirs(out, exist_ok=True)
    import random

    rng = random.Random(0)
    step = 0
    while step < steps:
        rng.shuffle(examples)
        for i in range(0, len(examples), batch_size):
            xs, ys = collate(examples[i : i + batch_size])
            loss = sft_step(model, opt, xs, ys, device, r=r)
            step += 1
            if step % 50 == 0 or step == 1:
                print(f"  step {step}/{steps}  loss {loss:.4f}")
            if step % save_every == 0 or step >= steps:
                tmp = os.path.join(out, "ckpt.pt.tmp")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "cfg": cfg.__dict__ if hasattr(cfg, "__dict__") else cfg,
                        "step": step,
                        "sft": True,
                    },
                    tmp,
                )
                os.replace(tmp, os.path.join(out, "ckpt.pt"))
            if step >= steps:
                break
    print(f"[sft] done -> {os.path.join(out, 'ckpt.pt')}")


# --------------------------------------------------------------------------
def _selftest():
    print("CHARKHA SFT self-test")
    ok = 0

    def ck(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        assert cond, name
        ok += 1

    # formatting + masking (pure)
    class _Tok:
        def encode(self, s):
            return list(s.encode("utf-8"))

        def decode(self, ids):
            return bytes(b & 0xFF for b in ids).decode("utf-8", "replace")

    tok = _Tok()
    ptext, full = format_example("2+2?", "It is 4.")
    ck("prompt text is the user/assistant stem", ptext == "user: 2+2?\nassistant:")
    ck("full text contains the completion", "It is 4." in full)
    ids, labels = encode_example(tok, "2+2?", "It is 4.")
    ck("input ids == full text ids", ids == tok.encode(full))
    n_prompt = len(tok.encode(ptext))
    ck("prompt span is masked (IGNORE)", all(l == IGNORE for l in labels[:n_prompt]))
    ck("completion span is supervised (not IGNORE)", any(l != IGNORE for l in labels[n_prompt:]))
    ck("masked count == prompt length", sum(1 for l in labels if l == IGNORE) == n_prompt)

    # truncation keeps lengths aligned
    ids2, lab2 = encode_example(tok, "x" * 50, "y" * 50, max_len=16)
    ck("truncation aligns input and labels", len(ids2) == 16 and len(lab2) == 16)

    # collate pads input with 0 and labels with IGNORE
    xs, ys = collate([([1, 2, 3], [IGNORE, 2, 3]), ([4, 5], [IGNORE, 5])])
    ck("collate pads to max length", len(xs[0]) == len(xs[1]) == 3)
    ck("input padded with 0", xs[1] == [4, 5, 0])
    ck("label padding is IGNORE", ys[1] == [IGNORE, 5, IGNORE])

    # jsonl reader accepts both schemas
    import tempfile

    jpath = os.path.join(tempfile.mkdtemp(), "sft.jsonl")
    with open(jpath, "w", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": "hi", "completion": "hello"}) + "\n")
        f.write(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "a"},
                    ]
                }
            )
            + "\n"
        )
    pairs = list(read_sft_jsonl(jpath))
    ck("reads prompt/completion schema", ("hi", "hello") in pairs)
    ck("reads messages schema (last user->assistant)", ("q", "a") in pairs)

    # end-to-end masked training on a toy model: loss is finite, ignores the prompt, and a few
    # AdamW steps reduce it on a tiny memorizable set (proves the masked path actually learns).
    import torch
    from charkha import Charkha, CharkhaConfig

    torch.manual_seed(0)
    device = "cpu"
    model = Charkha(CharkhaConfig.toy()).to(device)
    model.train()
    data = [encode_example(tok, "2+2?", "4"), encode_example(tok, "cap of japan?", "tokyo")]
    xs, ys = collate(data)
    loss0 = masked_ce(model, torch.tensor(xs, device=device), torch.tensor(ys, device=device))
    ck("masked CE is finite", torch.isfinite(loss0).item())

    # a sequence that is ENTIRELY masked has no supervised tokens -> loss carries no signal
    # (nan on most torch builds, 0.0 on a few). Documents the contract: never feed an all-prompt batch.
    all_masked = torch.full((1, len(xs[0])), IGNORE, device=device)
    lm = masked_ce(model, torch.tensor([xs[0]], device=device), all_masked)
    ck(
        "fully-masked batch carries no loss signal (nan or 0)",
        torch.isnan(lm).item() or float(lm.item()) == 0.0,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    last = None
    for _ in range(30):
        last = sft_step(model, opt, xs, ys, device)
    ck("SFT reduces masked loss over steps", last < float(loss0.item()))

    print(f"\nSFT selftest: {ok}/{ok} passed -- completion-masked instruction finetuning works")
    return True


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="CHARKHA SFT — instruction finetuning (completion-masked)"
    )
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--run", action="store_true", help="finetune a checkpoint on an SFT jsonl")
    p.add_argument(
        "--ckpt", type=str, default=None, help="base checkpoint (omit + --toy for a fresh toy)"
    )
    p.add_argument(
        "--data", type=str, default=None, help="SFT jsonl (prompt/completion or messages)"
    )
    p.add_argument("--out", type=str, default="runs/charkha-sft")
    p.add_argument("--toy", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument(
        "--effort", type=int, default=None, help="fixed recurrence loops during SFT (None=sampled)"
    )
    a = p.parse_args()
    if a.selftest:
        _selftest()
    elif a.run:
        if not a.data:
            p.error("--run requires --data <sft.jsonl>")
        run_sft(
            a.ckpt,
            a.data,
            a.out,
            device=a.device,
            steps=a.steps,
            lr=a.lr,
            batch_size=a.batch_size,
            max_len=a.max_len,
            r=a.effort,
            toy=a.toy,
        )
    else:
        p.print_help()

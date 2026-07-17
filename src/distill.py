"""CHARKHA distillation — soft-label (logit) knowledge distillation from a teacher.

This module implements soft-label knowledge distillation for local or remote teachers. It supplies the *teacher
side*: it runs a teacher model on a batch of token ids and returns, per position, the teacher's
**top-k** next-token distribution. The student (charkha.Charkha.forward(..., kd=...)) consumes
that as a memory-frugal KL term — gathering only k vocab columns, never building (B,T,V) logits.

TOKENIZER CONTRACT: logit KD requires the teacher and student to use exactly the same tokenizer.
A teacher on a different tokenizer cannot share logit targets directly; use sequence-level KD via
`pipeline.py --synth` instead. Select and review the teacher model explicitly for each run.


  python distill.py --selftest        # stdlib + torch; toy teacher, no HF download
"""

from __future__ import annotations
import argparse
import sys

import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def topk_from_logits(logits, k: int, temp: float):
    """Teacher logits (..., V) -> (idx, prob): the top-k token ids and a renormalized softmax
    over just those k at temperature `temp`. This defines the teacher's distribution on its own
    top-k support, which the student is trained to match (standard sparse-KD choice)."""
    k = min(k, logits.size(-1))
    top_logits, idx = torch.topk(logits, k, dim=-1)
    prob = F.softmax(top_logits / temp, dim=-1)  # renormalized over the k-subset
    return idx, prob


def kd_loss_topk_reference(student_logits_k, t_prob, temp: float):
    """Pure reference for the student's KD term, mirroring Charkha._kd_topk (for unit testing).
    student_logits_k: (..., k) student logits at the teacher's top-k ids. Returns scalar KL*temp^2."""
    slp = F.log_softmax(student_logits_k / temp, dim=-1)
    kl = (t_prob * (t_prob.clamp_min(1e-9).log() - slp)).sum(-1).mean()
    return kl * (temp * temp)


class ToyTeacher:
    """Hermetic stand-in for a real teacher: emits a valid top-k distribution for any batch.
    Two modes: wrap another module (self/peer-distillation) or synthesize a fixed distribution.
    Used by the self-tests so the KD path is exercised with no network/HF dependency."""

    def __init__(self, vocab_size, k=8, temp=1.0, model=None):
        self.vocab_size = vocab_size
        self.k = k
        self.temp = temp
        self.model = model

    @torch.no_grad()
    def topk(self, x):
        if self.model is not None:  # peer/self-distillation: real logits
            self.model.eval()
            logits, _ = self.model(x)
            return topk_from_logits(logits[:, :-1], self.k, self.temp)
        # synthetic: a smooth, valid per-position distribution seeded by the input ids
        B, T = x.shape
        base = torch.arange(self.k, device=x.device).float()
        logits = -(base.view(1, 1, -1)) + (x[:, :-1].float().unsqueeze(-1) % 3) * 0.5
        idx = (x[:, :-1].unsqueeze(-1) + torch.arange(self.k, device=x.device)) % self.vocab_size
        prob = F.softmax(logits / self.temp, dim=-1).expand(B, T - 1, self.k).contiguous()
        return idx, prob

    def refine(self, x):
        """Trajectory-refinement hook; identity in the hermetic toy teacher."""
        return x


class HFTeacher:
    """Real teacher: a Hugging Face causal LM run no-grad to produce per-position top-k.
    Loaded once and reused. Requires a SHARED tokenizer with the student (asserted on vocab).

    8GB NOTE: a 7B teacher does NOT fit on the training GPU beside the student+optimizer. Run it
    one of two ways (see ARCHITECTURE / distillation docs):
      * `quant='4bit'` (bitsandbytes) on a SEPARATE GPU/process/server — ~4GB for a 7B; or
      * `device='cpu'` — slow but zero VRAM contention ("willing to sacrifice time"); or
      * precompute top-k OFFLINE over a distillation subset and stream it (no teacher in-loop).
    """

    def __init__(
        self, model_id, device="cuda", k=64, temp=2.0, dtype=None, student_vocab=None, quant=None
    ):
        from transformers import AutoModelForCausalLM

        self.k, self.temp, self.device = k, temp, device
        self.student_vocab = student_vocab
        kw = {}
        if quant in ("4bit", "8bit"):  # bitsandbytes: fit a 7B teacher in ~4-7GB
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = (
                BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                if quant == "4bit"
                else BitsAndBytesConfig(load_in_8bit=True)
            )
            kw["device_map"] = {"": device}
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, use_safetensors=True, **kw
            ).eval()
        else:
            dt = dtype or (torch.bfloat16 if device == "cuda" else torch.float32)
            self.model = (
                AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dt, **kw)
                .to(device)
                .eval()
            )
        tv = self.model.get_output_embeddings().weight.size(0)
        if student_vocab is not None and tv < student_vocab:
            raise ValueError(
                f"teacher vocab {tv} < student {student_vocab}: tokenizer mismatch — "
                "use a teacher on the GPT-NeoX tokenizer (e.g. Comma), or sequence-level KD"
            )

    @torch.no_grad()
    def topk(self, x):
        # teacher logits at position i predict token i+1; [:, :-1] aligns with the student's
        # h[:, :-1] -> targets[:, :-1]. Filter any top-k id outside the student vocab (pad slack).
        out = self.model(x.to(self.device))
        logits = out.logits[:, :-1].float()
        idx, prob = topk_from_logits(logits, self.k, self.temp)
        if self.student_vocab is not None:
            bad = idx >= self.student_vocab
            if bad.any():  # zero out-of-range mass, renormalize
                prob = prob.masked_fill(bad, 0.0)
                idx = idx.masked_fill(bad, 0)
                prob = prob / prob.sum(-1, keepdim=True).clamp_min(1e-9)
        return idx.to(x.device), prob.to(x.device)

    def refine(self, x):
        """Optional trajectory-refinement hook. Real refiners can subclass this teacher."""
        return x


def teacher_kd_tuple(teacher, x, kd_weight, kd_temp, max_tokens=None, refine=False):
    """Build the (t_idx, t_prob, temp, weight) tuple Charkha.forward(kd=...) expects."""
    if refine and hasattr(teacher, "refine"):
        x = teacher.refine(x)
    t_idx, t_prob = teacher.topk(x)
    if max_tokens is not None and max_tokens > 0:
        t_idx = t_idx[:, :max_tokens]
        t_prob = t_prob[:, :max_tokens]
    return (t_idx, t_prob.to(x.device), kd_temp, kd_weight)


# LAN teacher server — run the teacher on a separate machine and fetch
# per-position top-k over the network. Frees the training GPU entirely.
# Custom endpoint (not OpenAI-compatible) because we need teacher-forced
# per-position top-k over the whole input, which the chat/completions
# logprobs API does not expose.
import base64
import json as _json


def _encode_array(arr):
    import numpy as np

    arr = np.ascontiguousarray(arr)
    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "b64": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def _decode_array(d):
    import numpy as np

    raw = base64.b64decode(d["b64"])
    return np.frombuffer(raw, dtype=d["dtype"]).reshape(d["shape"])


def compute_topk_payload(teacher, ids_array):
    """Core server logic (pure, no socket): decoded ids -> encoded top-k idx/prob.
    idx as int32, prob as float16 (compact on the wire, ample precision for KD)."""
    import numpy as np

    arr = np.asarray(ids_array)
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"ids must be integer dtype, got {arr.dtype}")
    # clip padding tokens (CHARKHA vocab 50,304 padded from GPT-NeoX 50,254)
    tv = teacher.model.get_input_embeddings().weight.size(0)
    arr = arr.clip(0, tv - 1)
    x = torch.from_numpy(arr.astype(np.int64))
    t_idx, t_prob = teacher.topk(x)
    return {
        "idx": _encode_array(t_idx.cpu().numpy().astype(np.int32)),
        "prob": _encode_array(t_prob.cpu().numpy().astype(np.float16)),
    }


def serve_teacher(
    model_id,
    host="0.0.0.0",
    port=8009,
    k=64,
    temp=2.0,
    quant="4bit",
    device="cuda",
    student_vocab=None,
):
    """Run an HTTP top-k teacher server. POST {'ids': <encoded (B,T)>} to /topk."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    teacher = HFTeacher(
        model_id, device=device, k=k, temp=temp, quant=quant, student_vocab=student_vocab
    )
    print(f"[teacher-server] {model_id} (quant={quant}, k={k}) on {host}:{port} — POST /topk")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):  # health check
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def do_POST(self):
            try:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                ids = _decode_array(_json.loads(body)["ids"])
                out = _json.dumps(compute_topk_payload(teacher, ids)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(out)
            except Exception as e:  # a bad request must not kill the server
                self.send_response(500)
                self.end_headers()
                self.wfile.write(_json.dumps({"error": str(e)}).encode())
                # CUDA errors are sticky — flush the error state so future requests survive
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

    ThreadingHTTPServer((host, port), H).serve_forever()


class RemoteTeacher:
    """Client for serve_teacher: same .topk(x) interface as HFTeacher, over the LAN."""

    def __init__(self, url):
        self.url = url.rstrip("/")

    def topk(self, x):
        import urllib.request
        import numpy as np

        payload = _json.dumps({"ids": _encode_array(x.cpu().numpy().astype(np.int64))}).encode()
        req = urllib.request.Request(
            self.url + "/topk", data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            d = _json.loads(r.read())
        idx = torch.from_numpy(_decode_array(d["idx"]).astype("int64")).to(x.device)
        prob = torch.from_numpy(_decode_array(d["prob"]).astype("float32")).to(x.device)
        return idx, prob

    def refine(self, x):
        return x


def _selftest():
    from charkha import Charkha, CharkhaConfig

    ok = 0

    def ck(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        assert cond, name
        ok += 1

    torch.manual_seed(0)
    # 1. topk_from_logits: shapes, probs normalize, picks the true argmax first.
    logits = torch.randn(2, 5, 100)
    idx, prob = topk_from_logits(logits, k=8, temp=2.0)
    ck("topk shapes", tuple(idx.shape) == (2, 5, 8) and tuple(prob.shape) == (2, 5, 8))
    ck("topk probs sum to 1", torch.allclose(prob.sum(-1), torch.ones(2, 5), atol=1e-5))
    ck("topk[0] is the argmax", bool((idx[..., 0] == logits.argmax(-1)).all()))

    # 2. KD reference: KL is >= 0, and ~0 when student logits match the teacher distribution.
    t_prob = torch.tensor([[0.7, 0.2, 0.1]])
    matched = t_prob.log()  # student logits == teacher log-probs
    kd0 = kd_loss_topk_reference(matched, t_prob, temp=1.0)
    ck("KD ~0 when student matches teacher", abs(kd0.item()) < 1e-4)
    worse = torch.tensor([[0.0, 0.0, 5.0]])  # mass on the teacher's least-likely token
    kd1 = kd_loss_topk_reference(worse, t_prob, temp=1.0)
    ck("KD larger when student disagrees", kd1.item() > kd0.item())
    ck("KD non-negative", kd0.item() > -1e-5 and kd1.item() > -1e-5)

    # 3. Model KD path matches the reference on real gathered logits.
    cfg = CharkhaConfig.toy()
    model = Charkha(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    teacher = ToyTeacher(cfg.vocab_size, k=8, temp=2.0)
    t_idx, t_prob = teacher.topk(x)
    h = torch.randn(2, 11, cfg.d_model)
    got = model._kd_topk(h, t_idx, t_prob, 2.0)
    W = model.embed.weight
    ref_logits_k = (h.unsqueeze(2) * W[t_idx]).sum(-1)  # (2,11,k) gathered student logits
    ref = kd_loss_topk_reference(ref_logits_k, t_prob, 2.0)
    ck("model._kd_topk matches reference", torch.allclose(got, ref, atol=1e-4))

    # 4. Full training forward with KD: finite loss, gradient flows.
    kd = teacher_kd_tuple(teacher, x, kd_weight=0.5, kd_temp=2.0)
    _, loss = model(x, x, kd=kd)
    loss.backward()
    ck("KD training loss is finite", bool(torch.isfinite(loss)))
    ck(
        "grad flows through the core under KD",
        any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.core.parameters()),
    )

    # 5. Self-distillation: teacher == the student's own distribution => KD term ~ 0.
    model.eval()
    selfteacher = ToyTeacher(cfg.vocab_size, k=8, temp=1.0, model=model)
    si, sp = selfteacher.topk(x)
    with torch.no_grad():
        slogits, _ = model(x)
        sl_k = slogits[:, :-1].gather(-1, si)
        self_kd = kd_loss_topk_reference(sl_k, sp, 1.0)
    ck("self-distillation KD ~0", self_kd.item() < 1e-3)

    print(
        f"\ndistill selftest: {ok}/{ok} passed -- top-k soft-label KD is exact, "
        "memory-frugal, and trains"
    )
    return True


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CHARKHA distillation (teacher side)")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--serve", action="store_true", help="start the LAN teacher server")
    p.add_argument(
        "--serve-model",
        type=str,
        default="common-pile/comma-v0.1-2t",
        help="HF model id for teacher",
    )
    p.add_argument(
        "--serve-quant",
        type=str,
        default="4bit",
        choices=["4bit", "8bit", None],
        help="bitsandbytes quantization",
    )
    p.add_argument("--serve-k", type=int, default=64, help="teacher top-k")
    p.add_argument("--serve-temp", type=float, default=2.0, help="KD temperature")
    p.add_argument("--serve-port", type=int, default=8009, help="listening port")
    p.add_argument("--serve-host", type=str, default="0.0.0.0", help="bind address")
    p.add_argument(
        "--serve-student-vocab", type=int, default=None, help="student vocab size for clipping"
    )
    a = p.parse_args()
    if a.serve:
        serve_teacher(
            a.serve_model,
            host=a.serve_host,
            port=a.serve_port,
            k=a.serve_k,
            temp=a.serve_temp,
            quant=a.serve_quant,
            device="cuda",
            student_vocab=a.serve_student_vocab,
        )
    elif a.selftest:
        _selftest()
    else:
        p.print_help()

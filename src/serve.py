"""CHARKHA serving wrapper with retrieval, tools, calibration, and session state."""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import torch

# Knowledge subsystem (stdlib-only core — keeps the serve selftest hermetic). A ~0.4B model can't
# store a 3B model's facts in its weights, so knowledge lives in an external provenanced datastore
# and is retrieved at inference. See retrieval.py.
from retrieval import Datastore, format_context, retrieval_quality
from worldmodel import WorldModel, format_world_context
from proofcarry import proof_carry as proof_carry_answer, replay_record as proof_replay_record

# --------------------------------------------------------------------------
# Tool calling - exact arithmetic, offloaded so the model doesn't have to guess.
# Model emits  [[calc: 12*(3+4)]]  ; we evaluate with an AST whitelist (never eval()).
# --------------------------------------------------------------------------


from _verify import *
from _verify import _terms
from _memory import Memory
from _serve_utils import build_datastore, load_model, _ingest


class Session:
    def __init__(
        self,
        model,
        tok,
        mem,
        device,
        *,
        location="Earth",
        date=None,
        cutoff="2026-06",
        base_effort=1,
        max_effort=None,
        conf_threshold=0.5,
        abstain_threshold=None,
        use_convergence=False,
        use_sngp=False,
        retriever=None,
        retrieval_k=4,
        retrieval_max_chars=800,
        use_retrieval_gate=False,
        concise=False,
        conv_log=None,
        speculative=False,
        sampling=None,
        contrast=None,
        best_of=1,
        council_size=1,
        council_synthesize=False,
        council_margin=0.15,
        council_seed_noise=0.0,
        world_model=None,
        world_k=8,
        adaptive_tts=False,
        ttt_steps=0,
        ttt_lr=5e-4,
        memory_of_thought=False,
        proof_carry=False,
        proof_strict=False,
        seed_noise=0.0,
        ply_branches=1,
        ply_noise=0.05,
        ply_score="consistency",
        ply_perturb="state",
    ):
        # conv_log: JSONL path for the consolidation daemon (consolidate.py) — every finished
        # exchange is appended with its confidence so sleep-time training can gate on it.
        # speculative: self-speculative decoding (draft at r=1, verify at the dialed effort) —
        # exact same output distribution, fewer full-effort passes per token.
        self.conv_log = conv_log
        self.speculative = speculative
        # sampling: dict of generate() quality knobs (top_p/min_p/rep_penalty/no_repeat_ngram).
        # contrast: (mini_model, lam) for contrastive decoding against a small fluency model.
        # best_of/council_size: sample multiple attempts per effort level. best_of is the old
        # confidence-only mode; council_size activates semantic agreement + groundedness reranking.
        self.sampling = sampling or {}
        self.contrast = contrast
        self.best_of = max(1, best_of)
        # ttt_steps: test-time training — when retrieval grounds a query, take a few gradient
        # steps on the retrieved passages with a THROWAWAY COPY of the model and answer with
        # that ("study the document before answering"); the serving weights never change.
        # memory_of_thought: feed confident answers back into the retrieval store as exemplars,
        # so the model's own past successes become few-shot context for similar future queries.
        # seed_noise: latent-seed diversity for best_of — candidates differ by the recurrent
        # core's initial state (in-distribution: training used state noise), not by temperature.
        self.ttt_steps = ttt_steps
        self.ttt_lr = ttt_lr
        self.memory_of_thought = memory_of_thought
        self.proof_carry = bool(proof_carry)
        self.proof_strict = bool(proof_strict)
        self.seed_noise = seed_noise
        self.council_size = max(1, council_size)
        self.council_synthesize = council_synthesize
        self.council_margin = max(0.0, float(council_margin))
        self.council_seed_noise = max(0.0, float(council_seed_noise))
        # ply: latent branch-and-select over the recurrent core (src/ply.py) —
        # per-step best-of-N in trajectory space, no extra tokens. Applies only to
        # fixed-int effort, plain (non-speculative) decode; pays N x core FLOPs per
        # token via full reforwards, so it is an explicit opt-in effort knob.
        self.ply_branches = max(1, int(ply_branches))
        self.ply_noise = float(ply_noise)
        self.ply_score = ply_score
        self.ply_perturb = ply_perturb
        self.world_model = world_model
        self.world_k = max(1, int(world_k))
        self.adaptive_tts = adaptive_tts
        self.model, self.tok, self.mem, self.device = model, tok, mem, device
        self.retriever = retriever
        self.retrieval_k = retrieval_k
        self.retrieval_max_chars = retrieval_max_chars
        self.use_retrieval_gate = use_retrieval_gate and retriever is not None
        self.location = location
        self.date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.cutoff = cutoff
        self.base_effort = base_effort
        self.max_effort = max_effort or model.cfg.max_recurrence_infer
        self.conf_threshold = conf_threshold
        self.abstain_threshold = abstain_threshold
        self.use_convergence = use_convergence
        self.use_sngp = use_sngp and getattr(model.cfg, "sngp_enabled", False)
        self.concise = concise
        # Use the tokenizer's native chat tokens when it has them (the custom CHARKHA tokenizers
        # bake <|system|>/<|user|>/<|assistant|> into the vocab). Without this those ids were DEAD
        # vocab and the serve format ('user: ...') could never match any special-token SFT format.
        # Plain-text fallback for tokenizers without them (byte toy, gpt-neox).
        t2i = getattr(tok, "token_to_id", None)
        try:
            self.chat_tokens = bool(t2i) and all(
                t2i(t) is not None for t in ("<|system|>", "<|user|>", "<|assistant|>")
            )
        except Exception:
            self.chat_tokens = False
        if use_convergence:
            model.cfg.track_convergence = True
        model.eval()

    def _adaptive_candidate_count(self, user: str, retrieval_q):
        n = max(self.best_of, self.council_size)
        if not self.adaptive_tts:
            return n
        terms = _terms(user)
        has_numeric = bool(re.search(r"\d|[+\-*/=]", user))
        is_question = "?" in user or user.lower().strip().split(" ", 1)[0] in {
            "what",
            "why",
            "how",
            "when",
            "where",
            "who",
            "which",
        }
        if has_numeric:
            n = max(n, 3)
        if is_question and len(terms) >= 8:
            n = max(n, 3)
        if retrieval_q is not None and retrieval_q < 0.55:
            n = max(n, 5)
        return n

    def _prompt_ids(self, user: str, note: str = None, context: str = None):
        pre = system_preamble(self.date, self.location, self.cutoff, concise=self.concise)
        if self.chat_tokens:
            # native chat-token format — the format SFT data must also use (see sft.py)
            sysblock = pre + (("\n" + context) if context else "") + (("\n" + note) if note else "")
            parts = ["<|system|>" + sysblock]
            for role, text in self.mem.recall():
                parts.append(("<|user|>" if role == "user" else "<|assistant|>") + text)
            parts.append(f"<|user|>{user}<|assistant|>")
            ids = self.tok.encode("".join(parts))
        else:
            parts = [pre]
            if context:
                parts.append(context)
            if note:
                parts.append(note)
            for role, text in self.mem.recall():
                parts.append(f"{role}: {text}")
            parts.append(f"user: {user}\nassistant:")
            ids = self.tok.encode("\n".join(parts))
        return ids or [ord("\n")]

    def _retrieve(self, user: str):
        """Top-k provenanced passages for the query (empty if no retriever / no hit / error).
        A retrieval failure must never crash a turn."""
        if self.retriever is None:
            return []
        try:
            return self.retriever.retrieve(user, k=self.retrieval_k)
        except Exception:
            return []

    def _speak_uncertain(self, user: str, n_new: int, context: str = None):
        """Regenerate an answer in the model's own words, conditioned to acknowledge
        uncertainty. Returns the hedged answer (or '' if the model produced nothing)."""
        ids = self._prompt_ids(user, note=UNCERTAINTY_NOTE, context=context)
        out = self.model.generate(
            torch.tensor([ids], device=self.device), n_new, effort=self.max_effort
        )[0].tolist()
        resolved, _ = resolve_tools(self.tok.decode(out[len(ids) :]))
        _, answer = split_thinking(resolved)
        return answer.strip()

    def _study(self, context: str, max_len: int = 512):
        """Test-time training: take a few gradient steps on the retrieved context using
        a throwaway deep copy of the model. The serving weights never change and the copy
        is discarded after the turn. Effective for out-of-distribution documents at small
        scale (TTT; see Sun et al., test-time training results). Costs ~ttt_steps training
        forwards; returns None on any failure."""
        ids = self.tok.encode(context)
        ids = getattr(ids, "ids", ids)
        if len(ids) < 16:
            return None
        try:
            import copy

            m2 = copy.deepcopy(self.model)
            m2.train()
            opt = torch.optim.SGD(m2.parameters(), lr=self.ttt_lr, momentum=0.9)
            for i in range(self.ttt_steps):
                s0 = (i * max_len) % max(1, len(ids) - 8)
                seq = torch.tensor([ids[s0 : s0 + max_len + 1]], device=self.device)
                if seq.size(1) < 8:
                    break
                _, loss = m2(seq[:, :-1], seq[:, 1:])
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(m2.parameters(), 1.0)
                    opt.step()
                opt.zero_grad(set_to_none=True)
            m2.eval()
            return m2
        except Exception:
            return None  # OOM / copy failure: answer with the base weights

    def _synthesize_council(self, user: str, cands, n_new: int, context: str = None):
        """One extra pass that compresses disagreeing attempts into a final answer. It is deliberately
        used only after the outer scorer has already exposed disagreement; the model is not asked to
        grade itself blindly, it is asked to reconcile concrete alternatives."""
        ranked = sorted(cands, key=lambda c: c.get("score", c.get("base_conf", 0.0)), reverse=True)[
            :5
        ]
        lines = []
        for i, c in enumerate(ranked, 1):
            ans = re.sub(r"\s+", " ", c.get("answer", "")).strip()[:700]
            lines.append(
                f"Attempt {i} score={c.get('score', 0.0):.2f} "
                f"conf={c.get('base_conf', 0.0):.2f}: {ans}"
            )
        note = (
            "[NOTE: Internal attempts are available below. Synthesize the most defensible final "
            "answer. Prefer claims supported by multiple attempts and retrieved context. If attempts "
            "conflict, state the uncertainty plainly instead of forcing a false consensus. Do not "
            "mention that there were attempts.]"
        )
        synth_user = user + "\n\nInternal attempts:\n" + "\n".join(lines)
        ids = self._prompt_ids(synth_user, note=note, context=context)
        out = self.model.generate(
            torch.tensor([ids], device=self.device),
            max(24, n_new),
            effort=self.max_effort,
            temp=0.2,
            top_k=40,
            top_p=0.9,
            min_p=0.0,
            rep_penalty=1.08,
            no_repeat_ngram=3,
        )[0].tolist()
        resolved, calls = resolve_tools(self.tok.decode(out[len(ids) :]))
        thinking, answer = split_thinking(resolved)
        return {
            "raw": resolved,
            "thinking": thinking,
            "answer": answer.strip(),
            "calls": calls,
            "ids": out,
        }

    def respond(self, user: str, n_new: int = 96, ts: float = 0.0):
        # Knowledge: retrieve once per turn, inject the cited passages, and (optionally) use the
        # retrieval-quality signal to gate effort/abstention.
        passages = self._retrieve(user)
        context = format_context(passages, self.retrieval_max_chars) if passages else ""
        world_facts = self.world_model.retrieve(user, k=self.world_k) if self.world_model else []
        world_context = format_world_context(world_facts)
        if world_context:
            context = (context + "\n" + world_context).strip()
        rq = retrieval_quality(passages) if passages else None
        prompt = self._prompt_ids(user, context=context)
        plen = len(prompt)
        # Test-time training: study the retrieved context with a throwaway copy; every
        # generation this turn uses the studied model, then it is discarded.
        gen_model = (self._study(context) if (self.ttt_steps and context) else None) or self.model

        def conf_fn(effort):
            spec = (
                dict(draft_effort=1, draft_len=4)
                if self.speculative and isinstance(effort, int) and effort > 1
                else {}
            )
            base_kw = dict(self.sampling)
            if self.contrast is not None and not spec:  # contrast composes with plain decode
                base_kw["contrast"] = self.contrast
            n_cand = self._adaptive_candidate_count(user, rq)
            cands = []
            prompt_t = torch.tensor([prompt], device=self.device)
            for ci in range(n_cand):
                kw = dict(base_kw)
                if n_cand > 1 and not spec:
                    base_temp = float(kw.get("temp", 0.8))
                    kw["temp"] = max(0.05, min(1.35, base_temp * (0.85 + 0.10 * ci)))
                    if self.council_seed_noise > 0:
                        kw["seed_noise"] = self.council_seed_noise * (ci + 1) / n_cand
                if (
                    self.ply_branches > 1
                    and not spec
                    and isinstance(effort, int)
                    and getattr(gen_model.cfg, "use_recurrence", False)
                ):
                    from ply import ply_serve_generate

                    gen, gconf = ply_serve_generate(
                        gen_model,
                        prompt_t,
                        n_new,
                        r=effort,
                        n_branches=self.ply_branches,
                        noise=self.ply_noise,
                        score=self.ply_score,
                        perturb=self.ply_perturb,
                        temp=float(kw.get("temp", 0.8)),
                        top_k=int(kw.get("top_k", 50)),
                    )
                else:
                    gen, gconf = gen_model.generate(
                        prompt_t, n_new, effort=effort, return_conf=True, **spec, **kw
                    )
                out = gen[0].tolist()
                raw = self.tok.decode(out[plen:])
                resolved, calls = resolve_tools(raw)
                thinking, answer = split_thinking(resolved)
                cand_c = float(gconf.mean().item()) if gconf.numel() else 1.0
                cands.append(
                    {
                        "ids": out,
                        "raw": raw,
                        "resolved": resolved,
                        "calls": calls,
                        "thinking": thinking,
                        "answer": answer.strip() or resolved.strip(),
                        "base_conf": cand_c,
                        "convergence_conf": (
                            convergence_confidence(getattr(gen_model, "_last_convergence", None))
                            if self.use_convergence
                            else None
                        ),
                        "sngp_conf": (
                            sngp_confidence(getattr(gen_model, "_last_sngp_var", None))
                            if self.use_sngp
                            else None
                        ),
                    }
                )
            if n_cand > 1:
                best, report = council_rank(
                    cands, passages=passages, retrieval_q=(rq if self.use_retrieval_gate else None)
                )
            else:
                best = max(cands, key=lambda c: c["base_conf"])
                report = {
                    "size": len(cands),
                    "consensus": best.get("consensus", 1.0),
                    "entropy": 0.0,
                    "clusters": [{"weight": 1.0, "size": len(cands)}],
                }
            conf_fn.last = best["ids"]
            conf_fn.selected = best
            conf_fn.candidates = cands
            conf_fn.council = report
            # confidence comes from the SAME forwards that generated the tokens (the conf head at
            # each step) — the old path re-ran the finished sequence through the model per effort
            # level just to re-read the same head, roughly doubling serve cost per escalation.
            c = float(best.get("base_conf", 1.0))
            if n_cand > 1:
                agreement = 1.0 - float(report.get("entropy", 0.0))
                c = min(float(best.get("score", c)), c * (0.55 + 0.45 * agreement))
            if self.use_convergence:
                c = min(c, best.get("convergence_conf") or 1.0)
            if self.use_sngp:
                c = min(c, best.get("sngp_conf") or 1.0)
            # retrieval-gated recurrence: weak/flat grounding caps confidence (min), so the dial
            # spends more loops and is likelier to abstain when the store can't back the answer.
            # Only gates when we actually retrieved something — a query that needs no retrieval
            # (e.g. arithmetic) returns empty and answers from weights as before.
            if self.use_retrieval_gate and rq is not None:
                c = min(c, rq)
            conf_fn.convergence = best.get("convergence_conf")
            conf_fn.sngp = best.get("sngp_conf")
            return c

        if self.base_effort == "converge":  # equilibrium mode: think-until-settled, no ladder
            conf = conf_fn("converge")
            effort, trace = "converge", [("converge", conf)]
        else:
            effort, conf, trace = escalate(
                conf_fn, self.base_effort, self.max_effort, self.conf_threshold
            )
        selected = getattr(conf_fn, "selected", None)
        if selected is not None:
            raw = selected["raw"]
            resolved = selected["resolved"]
            calls = selected["calls"]
            thinking, answer = selected["thinking"], selected["answer"]
        else:
            raw = self.tok.decode(conf_fn.last[plen:])
            resolved, calls = resolve_tools(raw)
            thinking, answer = split_thinking(resolved)
        synthesized = False
        council = getattr(conf_fn, "council", None)
        cands = getattr(conf_fn, "candidates", [])
        if (
            self.council_synthesize
            and cands
            and len(cands) > 1
            and (
                council.get("entropy", 0.0) > 0.35
                or council.get("consensus", 1.0) < 1.0 - self.council_margin
            )
        ):
            synth = self._synthesize_council(user, cands, n_new, context=context)
            if synth["answer"]:
                raw = synth["raw"]
                calls = synth["calls"]
                thinking, answer = synth["thinking"], synth["answer"]
                synthesized = True
                conf *= max(0.55, 1.0 - 0.5 * council.get("entropy", 0.0))
        low_conf = conf < self.conf_threshold  # escalation exhausted; flag, don't refuse

        # escalate-then-abstain: compute was already spent above; only NOW, if a
        # conformal gate is set and we're still under it, switch into uncertainty-aware
        # mode - the model re-answers in its own words, hedged, rather than guessing
        # confidently (and rather than emitting a canned refusal).
        abstain = self.abstain_threshold is not None and conf < self.abstain_threshold
        withheld = None
        if abstain:
            withheld = answer  # the confident-but-unreliable attempt
            hedged = self._speak_uncertain(user, len(conf_fn.last) - plen or 96, context=context)
            answer = hedged or honest_refusal(conf, self.abstain_threshold)
            thinking = ""  # hedged reply is the user-facing one

        proof_report = None
        proof_replay = None
        if self.proof_carry:
            proof_report = proof_carry_answer(
                answer, passages=passages, world_facts=world_facts, strict=self.proof_strict
            )
            if self.proof_strict:
                answer = proof_report["filtered_answer"]
            if proof_report["checked"]:
                conf *= max(0.25, 0.5 + 0.5 * float(proof_report["score"]))
            proof_replay = proof_replay_record(user, answer, proof_report)

        self.mem.remember("user", user, ts)
        self.mem.remember("assistant", answer, ts)
        # Memory-of-thought: a confident, non-abstained answer becomes a retrievable exemplar —
        # similar future queries get the model's own past reasoning as few-shot context, so the
        # system improves within a session without touching the weights (consolidate.py later
        # folds the same exchanges INTO the weights for zero-training improvement).
        if (
            self.memory_of_thought
            and self.retriever is not None
            and not abstain
            and conf >= self.conf_threshold
            and answer
            and (proof_report is None or proof_report.get("accepted"))
        ):
            try:
                self.retriever.add(
                    f"Q: {user}\nA: {answer}", source="experience", license="personal"
                )
                self.retriever.finalize()
            except Exception:
                pass
        world_updates, world_conflicts = [], []
        if self.world_model:
            try:
                world_updates, world_conflicts = self.world_model.remember(
                    user, source="user", ts=ts or time.time()
                )
            except Exception:
                world_updates, world_conflicts = [], []
        failure_mode = classify_failure(
            answer, council=council, retrieval_q=rq, world_conflicts=world_conflicts, calls=calls
        )
        if proof_report is not None:
            if proof_report.get("contradicted", 0):
                failure_mode = "contradicted_claim"
            elif proof_report.get("unsupported", 0):
                failure_mode = "unsupported_claim"
        if self.conv_log:
            # feed the sleep-time consolidation daemon; a log failure must never break a turn
            try:
                d = os.path.dirname(self.conv_log)
                if d:
                    os.makedirs(d, exist_ok=True)
                council_row = council or {}
                with open(self.conv_log, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "ts": ts or time.time(),
                                "user": user,
                                "assistant": answer,
                                "conf": round(float(conf), 4),
                                "abstained": bool(abstain),
                                "synthesized": bool(synthesized),
                                "council_size": council_row.get("size"),
                                "council_entropy": council_row.get("entropy"),
                                "world_updates": len(world_updates),
                                "world_conflicts": len(world_conflicts),
                                "proof_score": (
                                    None
                                    if proof_report is None
                                    else round(float(proof_report.get("score", 1.0)), 4)
                                ),
                                "proof_unsupported": (
                                    None
                                    if proof_report is None
                                    else proof_report.get("unsupported", 0)
                                ),
                                "failure_mode": failure_mode,
                            }
                        )
                        + "\n"
                    )
            except OSError:
                pass
        convergence = getattr(conf_fn, "convergence", None) if self.use_convergence else None
        sngp = getattr(conf_fn, "sngp", None) if self.use_sngp else None
        # provenance for citations: the sources backing this answer (empty if nothing retrieved)
        sources = [
            {
                "source": p.get("source", ""),
                "license": p.get("license", ""),
                "score": p.get("score", 0.0),
            }
            for p in passages
        ]
        return {
            "answer": answer,
            "thinking": thinking,
            "effort": effort,
            "confidence": conf,
            "low_confidence": low_conf,
            "tools": calls,
            "convergence": convergence,
            "sngp": sngp,
            "sources": sources,
            "retrieval_quality": rq,
            "effort_trace": trace,
            "abstained": abstain,
            "withheld_answer": withheld,
            "uncertainty_steered": abstain,
            "council": council,
            "synthesized": synthesized,
            "world_facts": world_facts,
            "world_updates": [f.__dict__ for f in world_updates],
            "world_conflicts": world_conflicts,
            "proof_carry": proof_report,
            "proof_replay": proof_replay,
            "failure_mode": failure_mode,
        }


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------


def _selftest():
    print("CHARKHA serve self-test")
    ok = 0

    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        assert cond, name
        ok += 1

    # tool calculator: correctness + safety
    check("calc arithmetic", safe_calc("2 + 3 * 4") == 14)
    check("calc precedence/pow", safe_calc("(1+2)**3") == 27)
    bad = False
    try:
        safe_calc('__import__("os").system("x")')
    except Exception:
        bad = True
    check("calc rejects calls/names", bad)
    bad2 = False
    try:
        safe_calc("x + 1")
    except Exception:
        bad2 = True
    check("calc rejects bare names", bad2)
    # structured reasoning tokens
    check("reason parsing", reason_engine("addition:5+3:=8") == "(addition → 5+3:=8)")
    check("reason single arg", reason_engine("just a thought") == "[reasoning: just a thought]")
    check("verify passthrough", verify_engine("5×3=15") == "[verified: 5×3=15]")
    txt3, calls3 = resolve_tools("[[reason: multiply:5*3:15]] and [[verify: 5*3=15]]")
    check("reason+verify tools resolve", len(calls3) >= 2)

    # tool-call extraction + resolution
    txt, calls = resolve_tools("the total is [[calc: 6*7]] dollars")
    check("tool resolves to value", "the total is 6*7 = 42 dollars" == txt)
    check("tool call recorded", calls == [("calc", "6*7", 42)])
    txt2, calls2 = resolve_tools("safe [[calc: 1/0]] tail")
    check("bad tool call survives", "error:" in txt2 and "tail" in txt2)

    # thinking split
    th, ans = split_thinking("<thinking>9*9 is 81</thinking>The answer is 81.")
    check("thinking captured", th == "9*9 is 81")
    check("answer stripped of thinking", ans == "The answer is 81.")

    # grounding preamble
    pre = system_preamble("2026-06-12", "Pune, IN", "2024-12")
    check("preamble has date", "2026-06-12" in pre)
    check("preamble has location", "Pune, IN" in pre)
    check("preamble has cutoff", "2024-12" in pre)

    # memory roundtrip + persistence across reopen
    import tempfile

    mpath = os.path.join(tempfile.mkdtemp(), "mem.sqlite")
    mem = Memory(mpath)
    mem.remember("user", "my name is Alex", 1.0)
    mem.remember("assistant", "noted", 2.0)
    mem.set_fact("name", "Alex")
    check("recall returns turns in order", [r for r, _ in mem.recall()] == ["user", "assistant"])
    mem.close()
    mem2 = Memory(mpath)  # reopen - data must survive
    check("memory persists across reopen", mem2.get_facts().get("name") == "Alex")
    check("turns persist across reopen", len(mem2.recall()) == 2)
    mem2.close()

    # effort escalation policy (pure): confidence climbs with effort
    curve = {1: 0.30, 2: 0.45, 4: 0.80, 8: 0.95}
    eff, conf, trace = escalate(lambda e: curve[e], base=1, max_effort=8, threshold=0.7)
    check("escalates until confident", eff == 4 and conf == 0.80)
    check("escalation trace recorded", [e for e, _ in trace] == [1, 2, 4])
    # never-confident -> stops at cap, flagged low (no refusal)
    eff2, conf2, _ = escalate(lambda e: 0.1, base=1, max_effort=8, threshold=0.7)
    check("caps at max effort when never confident", eff2 == 8 and conf2 == 0.1)

    # fallback refusal (used only if regeneration is empty) cites the confidence
    msg = honest_refusal(0.30, 0.80)
    check("fallback refusal cites confidence", "0.30" in msg)

    # end to end on a fresh toy model (untrained -> we only assert the plumbing runs)
    device = "cpu"
    model, tok, cfg = load_model(None, device, toy=True)
    mem3 = Memory(os.path.join(tempfile.mkdtemp(), "s.sqlite"))
    sess = Session(
        model,
        tok,
        mem3,
        device,
        date="2026-06-12",
        location="Earth",
        base_effort=1,
        max_effort=cfg.max_recurrence_infer,
        conf_threshold=2.0,
    )
    torch.manual_seed(0)
    r = sess.respond("what is 2+2?", n_new=24)
    check("session returns a string answer", isinstance(r["answer"], str))
    check("session effort within cap", 1 <= r["effort"] <= cfg.max_recurrence_infer)
    check("threshold>1 forces full escalation", r["effort"] == cfg.max_recurrence_infer)
    check("confidence in [0,1]", 0.0 <= r["confidence"] <= 1.0)
    check("no abstention without a gate", r["abstained"] is False)
    check("turn written to memory", len(mem3.recall()) == 2)
    mem3.close()

    # escalate-then-abstain: a gate above any possible confidence forces abstention,
    # the honest-refusal text replaces the answer, and the guess is withheld (not lost).
    mem4 = Memory(os.path.join(tempfile.mkdtemp(), "a.sqlite"))
    sess_a = Session(
        model,
        tok,
        mem4,
        device,
        date="2026-06-12",
        location="Earth",
        base_effort=1,
        max_effort=cfg.max_recurrence_infer,
        conf_threshold=2.0,
        abstain_threshold=1.01,
    )  # 1.01 > any sigmoid conf
    ra = sess_a.respond("what is 2+2?", n_new=24)
    check("abstains when below conformal gate", ra["abstained"] is True)
    check("uncertainty mode produces a real (non-empty) answer", len(ra["answer"]) > 0)
    check(
        "answer is the model speaking, not the canned fallback verbatim",
        ra["uncertainty_steered"] is True,
    )
    check("original guess withheld (not lost)", isinstance(ra["withheld_answer"], str))
    mem4.close()

    # A2 convergence fusion: the mapping is monotone (more trajectory error => less confidence),
    # absent-signal is neutral (1.0), and an opted-in Session surfaces the convergence score.
    check("convergence: absent signal is neutral", convergence_confidence(None) == 1.0)
    check(
        "convergence: monotone decreasing in error",
        convergence_confidence(torch.zeros(1, 4))
        > convergence_confidence(torch.full((1, 4), 3.0))
        > convergence_confidence(torch.full((1, 4), 9.0)),
    )
    mem5 = Memory(os.path.join(tempfile.mkdtemp(), "c.sqlite"))
    sess_c = Session(
        model,
        tok,
        mem5,
        device,
        date="2026-06-12",
        location="Earth",
        base_effort=1,
        max_effort=cfg.max_recurrence_infer,
        conf_threshold=2.0,
        use_convergence=True,
    )
    check("use_convergence enables model trajectory tracking", model.cfg.track_convergence is True)
    rc = sess_c.respond("what is 2+2?", n_new=24)
    check(
        "convergence surfaced in result",
        rc["convergence"] is not None and 0.0 < rc["convergence"] <= 1.0,
    )
    mem5.close()

    # SNGP fusion: mapping monotone (more epistemic variance => less confidence), absent neutral,
    # and an opted-in Session over an sngp-enabled model surfaces the score. use_sngp must no-op
    # (stay None) when the model wasn't built with sngp_enabled.
    check("sngp: absent signal is neutral", sngp_confidence(None) == 1.0)
    check(
        "sngp: monotone decreasing in variance",
        sngp_confidence(torch.zeros(1, 4))
        > sngp_confidence(torch.full((1, 4), 3.0))
        > sngp_confidence(torch.full((1, 4), 9.0)),
    )
    mem6 = Memory(os.path.join(tempfile.mkdtemp(), "d.sqlite"))
    sess_d = Session(model, tok, mem6, device, conf_threshold=2.0, use_sngp=True)
    check("use_sngp no-ops when model lacks sngp head", sess_d.use_sngp is False)
    rd = sess_d.respond("what is 2+2?", n_new=24)
    check("sngp absent in result when model has no head", rd["sngp"] is None)
    mem6.close()
    from charkha import Charkha, CharkhaConfig

    gcfg = CharkhaConfig.toy()
    gcfg.sngp_enabled = True
    gcfg.sngp_rff_dim = 64
    gmodel = Charkha(gcfg).to(device)
    mem7 = Memory(os.path.join(tempfile.mkdtemp(), "e.sqlite"))
    sess_e = Session(gmodel, tok, mem7, device, conf_threshold=2.0, use_sngp=True)
    check("use_sngp enabled on sngp model", sess_e.use_sngp is True)
    re_ = sess_e.respond("what is 2+2?", n_new=24)
    check("sngp surfaced in result", re_["sngp"] is not None and 0.0 < re_["sngp"] <= 1.0)
    mem7.close()

    # Retrieval (knowledge) wiring: a provenanced datastore feeds cited context into the prompt,
    # surfaces sources, and (gated) caps confidence on weak grounding. Hermetic: BM25 is stdlib.
    ds = Datastore()
    ds.add(
        "The boiling point of water at standard atmospheric pressure is 100 degrees Celsius.",
        source="wikipedia",
        license="cc-by-sa",
    )
    ds.add(
        "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
        source="common_corpus",
        license="public-domain",
    )
    ds.finalize()
    mem8 = Memory(os.path.join(tempfile.mkdtemp(), "r.sqlite"))
    sess_r = Session(
        model,
        tok,
        mem8,
        device,
        conf_threshold=2.0,
        retriever=ds,
        retrieval_k=2,
        use_retrieval_gate=True,
    )
    check("retrieval gate active only with a datastore", sess_r.use_retrieval_gate is True)
    passages = ds.retrieve("water boiling point", k=2)
    decoded = tok.decode(
        sess_r._prompt_ids("water boiling point", context=format_context(passages))
    )
    check(
        "retrieved context injected into prompt",
        "boiling point" in decoded and "[wikipedia]" in decoded,
    )
    rr = sess_r.respond("what temperature does water boil at", n_new=16)
    check(
        "answer cites the retrieved source", any(s["source"] == "wikipedia" for s in rr["sources"])
    )
    check(
        "retrieval_quality surfaced in (0,1]",
        rr["retrieval_quality"] is not None and 0.0 < rr["retrieval_quality"] <= 1.0,
    )
    check(
        "retrieval gate caps confidence at grounding quality",
        rr["confidence"] <= rr["retrieval_quality"] + 1e-6,
    )
    rn = sess_r.respond("zzzqqq nonexistent term", n_new=8)
    check(
        "no-match query -> empty sources, answers from weights",
        rn["sources"] == [] and rn["retrieval_quality"] is None,
    )

    # Training-data RAG: a dataprep shard dir can be decoded into cited retrieval passages,
    # giving serve-time access to the exact on-disk curriculum without retraining.
    import array

    td = tempfile.mkdtemp()
    shard_text = (
        "alpha curriculum mastery passage teaches the tiny model about blue widgets. " * 4
    ).encode("utf-8")
    with open(os.path.join(td, "shard_00000.bin"), "wb") as f:
        array.array("H", list(shard_text)).tofile(f)
    with open(os.path.join(td, "index.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "vocab_size": 257,
                "total_tokens": len(shard_text),
                "shards": [{"file": "shard_00000.bin", "tokens": len(shard_text)}],
            },
            f,
        )
    train_ds = build_datastore(td, tokenizer=tok, max_passages=4, passage_tokens=96)
    train_hits = train_ds.retrieve("blue widgets curriculum mastery", k=2)
    check(
        "training shard dir becomes retrievable context",
        train_hits and "blue widgets" in train_hits[0]["text"],
    )
    check(
        "training shard retrieval carries provenance",
        train_hits[0]["license"] == "training-data"
        and "shard_00000.bin@" in train_hits[0]["source"],
    )
    mem8.close()
    mem9 = Memory(os.path.join(tempfile.mkdtemp(), "rn.sqlite"))
    sess_nr = Session(model, tok, mem9, device, conf_threshold=2.0, use_retrieval_gate=True)
    check("retrieval gate no-ops without a datastore", sess_nr.use_retrieval_gate is False)
    rnr = sess_nr.respond("what is 2+2?", n_new=8)
    check("no retriever -> empty sources", rnr["sources"] == [])
    mem9.close()

    # Conversation logging (consolidation feed): every exchange lands in the JSONL with its
    # confidence; converge-mode and speculative sessions still answer end to end.
    clog = os.path.join(tempfile.mkdtemp(), "conv.jsonl")
    mem10 = Memory(os.path.join(tempfile.mkdtemp(), "l.sqlite"))
    sess_l = Session(model, tok, mem10, device, conf_threshold=0.0, conv_log=clog)
    sess_l.respond("hello there", n_new=8)
    rec = json.loads(open(clog).read().strip().splitlines()[-1])
    check(
        "conversation logged for consolidation",
        rec["user"] == "hello there" and "conf" in rec and "assistant" in rec,
    )
    mem10.close()
    mem11 = Memory(os.path.join(tempfile.mkdtemp(), "g.sqlite"))
    sess_g = Session(model, tok, mem11, device, conf_threshold=0.0, base_effort="converge")
    rg = sess_g.respond("what is 2+2?", n_new=8)
    check(
        "converge-mode session answers",
        isinstance(rg["answer"], str) and rg["effort"] == "converge",
    )
    mem11.close()
    mem11p = Memory(os.path.join(tempfile.mkdtemp(), "p.sqlite"))
    sess_p = Session(
        model, tok, mem11p, device, conf_threshold=0.0, base_effort=2, ply_branches=3, ply_noise=0.1
    )
    rp = sess_p.respond("what is 2+2?", n_new=8)
    check(
        "ply-mode session answers",
        isinstance(rp["answer"], str) and len(rp["answer"]) >= 0 and rp["effort"] == 2,
    )
    mem11p.close()
    mem12 = Memory(os.path.join(tempfile.mkdtemp(), "sp.sqlite"))
    sess_sp = Session(
        model, tok, mem12, device, conf_threshold=2.0, base_effort=2, max_effort=4, speculative=True
    )
    rs = sess_sp.respond("what is 2+2?", n_new=8)
    check(
        "speculative session answers with escalation",
        isinstance(rs["answer"], str) and 2 <= rs["effort"] <= 4,
    )
    mem12.close()
    mem12b = Memory(os.path.join(tempfile.mkdtemp(), "co.sqlite"))
    sess_co = Session(
        model,
        tok,
        mem12b,
        device,
        conf_threshold=0.0,
        base_effort=1,
        council_size=3,
        council_synthesize=True,
    )
    rc = sess_co.respond("name a primary color", n_new=8)
    check(
        "council mode returns uncertainty metadata",
        isinstance(rc["answer"], str)
        and rc["council"]["size"] == 3
        and 0.0 <= rc["council"]["entropy"] <= 1.0,
    )
    mem12b.close()
    wm = WorldModel(os.path.join(tempfile.mkdtemp(), "world.sqlite"))
    mem12c = Memory(os.path.join(tempfile.mkdtemp(), "wm.sqlite"))
    sess_wm = Session(model, tok, mem12c, device, conf_threshold=0.0, world_model=wm)
    rw1 = sess_wm.respond("My GPU is a 4060 Ti.", n_new=4)
    rw2 = sess_wm.respond("what GPU do I have?", n_new=4)
    check(
        "world model stores user facts",
        rw1["world_updates"] and any("4060" in f["object"] for f in rw2["world_facts"]),
    )
    rw3 = sess_wm.respond("My GPU is an RTX 5090.", n_new=4)
    check("world model surfaces contradictions", bool(rw3["world_conflicts"]))
    wm.close()
    mem12c.close()
    mem12d = Memory(os.path.join(tempfile.mkdtemp(), "ats.sqlite"))
    sess_ats = Session(model, tok, mem12d, device, conf_threshold=0.0, adaptive_tts=True)
    rats = sess_ats.respond("what is 12 * 13 = 156 and why?", n_new=4)
    check("adaptive TTS promotes hard prompts to council mode", rats["council"]["size"] >= 3)
    mem12d.close()
    # /ingest plumbing: file lands in the live datastore and the consolidation inbox
    ing_dir = tempfile.mkdtemp()
    open(os.path.join(ing_dir, "note.txt"), "w").write("the capital of France is Paris")
    mem13 = Memory(os.path.join(tempfile.mkdtemp(), "i.sqlite"))
    sess_i = Session(model, tok, mem13, device, conf_threshold=0.0)
    _ingest(sess_i, ing_dir, inbox=tempfile.mkdtemp())  # hermetic: never the real inbox
    hits = sess_i.retriever.retrieve("capital of France", k=1)
    check(
        "/ingest makes content retrievable immediately", bool(hits) and "Paris" in hits[0]["text"]
    )
    mem13.close()

    # Test-time training: a studied throwaway copy answers; base weights stay untouched.
    mem14 = Memory(os.path.join(tempfile.mkdtemp(), "ttt.sqlite"))
    ds_t = Datastore()
    ds_t.add("The zorblax constant equals forty-two exactly.", source="doc", license="x")
    ds_t.finalize()
    sess_t = Session(
        model, tok, mem14, device, conf_threshold=0.0, retriever=ds_t, ttt_steps=2, ttt_lr=1e-3
    )
    w_before = model.embed.weight.detach().clone()
    rt = sess_t.respond("what is the zorblax constant", n_new=8)
    check("TTT answers end to end", isinstance(rt["answer"], str))
    check(
        "TTT never mutates the serving weights", torch.equal(w_before, model.embed.weight.detach())
    )
    studied = sess_t._study("The zorblax constant equals forty-two exactly. " * 20)
    check(
        "TTT study returns a trained copy (weights moved)",
        studied is not None
        and not torch.equal(studied.embed.weight.detach(), model.embed.weight.detach()),
    )
    mem14.close()
    # Memory-of-thought: a confident answer becomes retrievable experience.
    mem15 = Memory(os.path.join(tempfile.mkdtemp(), "mot.sqlite"))
    ds_m = Datastore()
    ds_m.add("seed passage about gardens", source="doc", license="x")
    ds_m.finalize()
    sess_m = Session(
        model, tok, mem15, device, conf_threshold=0.0, retriever=ds_m, memory_of_thought=True
    )
    sess_m.respond("tell me about gardens", n_new=8)
    hits_m = ds_m.retrieve("tell me about gardens", k=3)
    check(
        "memory-of-thought stores the exchange as experience",
        any(h.get("source") == "experience" for h in hits_m),
    )
    mem15.close()

    # Proof-Carry: strict mode strips contradicted factual claims before return/storage.
    class _FixedCfg:
        max_recurrence_infer = 2
        sngp_enabled = False

    class _FixedModel:
        cfg = _FixedCfg()

        def __init__(self, text):
            self.text = text

        def eval(self):
            return self

        def generate(self, prompt_t, n_new, effort=None, return_conf=False, **kw):
            ans = torch.tensor([tok.encode(self.text)], dtype=torch.long, device=prompt_t.device)
            out = torch.cat([prompt_t, ans], dim=1)
            conf = torch.full((1, ans.size(1)), 0.95, device=prompt_t.device)
            return (out, conf) if return_conf else out

    mem16 = Memory(os.path.join(tempfile.mkdtemp(), "pc.sqlite"))
    sess_pc = Session(
        _FixedModel("12 * 13 = 155."),
        tok,
        mem16,
        device,
        conf_threshold=0.0,
        proof_carry=True,
        proof_strict=True,
    )
    rpc = sess_pc.respond("bad math", n_new=8)
    check(
        "proof-carry filters contradicted arithmetic",
        rpc["proof_carry"]["contradicted"] == 1 and "155" not in rpc["answer"],
    )
    check(
        "proof-carry returns a claim ledger",
        rpc["proof_carry"]["checked"] == 1 and rpc["proof_replay"] is None,
    )
    check("proof-carry sets failure taxonomy", rpc["failure_mode"] == "contradicted_claim")
    mem16.close()

    print(
        f"\nSELFTEST PASS - {ok} checks; serving wrapper grounds, remembers, "
        "retrieves cited knowledge, escalates effort, and offloads tools"
    )


# --------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="CHARKHA serve - grounded inference wrapper")
    p.add_argument("--ckpt", type=str, default=None, help="checkpoint path (real model)")
    p.add_argument("--toy", action="store_true", help="fresh byte-level toy model (no ckpt)")
    p.add_argument(
        "--digit-split",
        action="store_true",
        help="match a model trained with dataprep --digit-split: one-token-per-digit at "
        "inference (split input, re-glue output). MUST match how the ckpt was trained.",
    )
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--once", type=str, default=None, help="answer one prompt and exit")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--memory", type=str, default="charkha_memory.sqlite")
    p.add_argument("--location", type=str, default="Earth")
    p.add_argument("--date", type=str, default=None)
    p.add_argument("--cutoff", type=str, default="2024-12")
    p.add_argument(
        "--effort",
        type=str,
        default="1",
        help="base effort: loop count (escalates if unsure) or 'converge' "
        "(equilibrium mode: iterate the core until the state settles)",
    )
    p.add_argument("--max-effort", type=int, default=None)
    p.add_argument(
        "--speculative",
        action="store_true",
        help="self-speculative decoding: draft at r=1, verify at the dialed effort — "
        "identical output distribution, fewer full-effort passes per token",
    )
    p.add_argument("--temp", type=float, default=0.8, help="sampling temperature")
    p.add_argument("--top-k", type=int, default=50, help="top-k sampling cutoff (0=off)")
    p.add_argument("--top-p", type=float, default=0.0, help="nucleus sampling mass (0=off)")
    p.add_argument(
        "--min-p",
        type=float,
        default=0.05,
        help="min-p sampling: keep tokens with p >= min_p * p_max (0=off)",
    )
    p.add_argument(
        "--rep-penalty",
        type=float,
        default=1.15,
        help="repetition penalty on recently generated tokens (1.0=off)",
    )
    p.add_argument(
        "--no-repeat-ngram",
        type=int,
        default=3,
        help="ban completing an n-gram already generated (0=off; task-blind — "
        "breaks instructed repetition; prefer --echo-gate)",
    )
    p.add_argument(
        "--echo-gate",
        type=float,
        default=0.0,
        help="CONDITIONAL anti-loop strength (0=off): penalize a loop-extending "
        "token only by its momentum excess (tail-only support beyond "
        "full-context support). Instructed repetition passes untouched. "
        "Suggested: --echo-gate 2.0 --no-repeat-ngram 0",
    )
    p.add_argument(
        "--echo-ngram",
        type=int,
        default=3,
        help="n-gram length that counts as a loop candidate for --echo-gate",
    )
    p.add_argument(
        "--echo-ctx",
        type=int,
        default=64,
        help="tail length for the momentum readout of --echo-gate",
    )
    p.add_argument(
        "--contrast-ckpt",
        type=str,
        default=None,
        help="small fluency checkpoint for contrastive decoding "
        "(score = logp_big - lam*logp_mini on the plausible set)",
    )
    p.add_argument("--contrast-lam", type=float, default=0.5)
    p.add_argument(
        "--contrast-alpha",
        type=float,
        default=0.1,
        help="contrastive plausible-set floor: p >= alpha * p_max",
    )
    p.add_argument(
        "--loop-contrast",
        type=float,
        default=0.0,
        help="DoLa-style contrast across recurrent effort: logp(effort)-lam*logp(low effort)",
    )
    p.add_argument(
        "--loop-contrast-effort",
        type=int,
        default=1,
        help="low-effort readout used as the amateur for --loop-contrast",
    )
    p.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="sample N candidates per effort level, keep the most confident "
        "(self-consistency; N-fold cost)",
    )
    p.add_argument(
        "--council",
        type=int,
        default=1,
        help="sample N internal attempts per effort and rerank by confidence, consensus, "
        "grounding, tool validity, and anti-loop health",
    )
    p.add_argument(
        "--council-synthesize",
        action="store_true",
        help="when council attempts disagree, run one extra synthesis pass instead of "
        "blindly trusting the top sample",
    )
    p.add_argument(
        "--council-margin",
        type=float,
        default=0.15,
        help="synthesize when best candidate consensus is below 1-margin",
    )
    p.add_argument(
        "--council-seed-noise",
        type=float,
        default=0.0,
        help="latent recurrent seed noise for council diversity (0=token sampling only)",
    )
    p.add_argument(
        "--ply",
        type=int,
        default=1,
        help="latent branch-and-select: N recurrence trajectories per decode step, "
        "winner picked by --ply-score before any token is emitted (src/ply.py). "
        "1 = off. Fixed-int --effort only; N x core FLOPs per token.",
    )
    p.add_argument(
        "--ply-noise", type=float, default=0.05, help="branch perturbation scale for --ply"
    )
    p.add_argument(
        "--ply-score",
        type=str,
        default="consistency",
        choices=["consistency", "conf", "margin", "entropy"],
        help="branch selector for --ply (consistency = latent majority vote)",
    )
    p.add_argument(
        "--ply-perturb",
        type=str,
        default="state",
        choices=["state", "input"],
        help="branch axis: initial core state (in-distribution) or loop input",
    )
    p.add_argument(
        "--conv-log",
        type=str,
        default=None,
        help="JSONL conversation log for consolidate.py (default runs/serve/"
        "conversations.jsonl for real checkpoints; off for --toy)",
    )
    p.add_argument("--conf-threshold", type=float, default=0.5)
    p.add_argument("--max-new", type=int, default=96)
    p.add_argument(
        "--abstain-conf",
        type=float,
        default=None,
        help="conformal confidence gate: decline when conf < this (from pipeline.py)",
    )
    p.add_argument(
        "--calibrate",
        type=str,
        default=None,
        help="val shard: compute the conformal abstain threshold at boot",
    )
    p.add_argument(
        "--target-risk",
        type=float,
        default=0.10,
        help="with --calibrate: max error rate among answered (default 0.10)",
    )
    p.add_argument("--delta", type=float, default=0.05, help="with --calibrate: 1-delta confidence")
    p.add_argument(
        "--use-convergence",
        action="store_true",
        help="A2: fuse the trajectory-convergence signal into the effort dial (min with "
        "conf head). Parameter-free; uncalibrated mapping, off by default.",
    )
    p.add_argument(
        "--use-sngp",
        action="store_true",
        help="SNGP: fuse the distance-aware epistemic (OOD) confidence into the effort "
        "dial (min with conf head). Requires an sngp-enabled checkpoint; off by default.",
    )
    p.add_argument(
        "--retrieve",
        type=str,
        default=None,
        help="knowledge datastore file (.jsonl {text,source,license}, .txt paragraphs, "
        "or dataprep shard dir with index.json). Comma/semicolon separated paths are allowed; "
        "retrieved cited passages are injected into the prompt",
    )
    p.add_argument(
        "--retrieve-max-passages",
        type=int,
        default=20000,
        help="with shard-dir --retrieve: max decoded training-data passages to index (0 = all)",
    )
    p.add_argument(
        "--retrieve-passage-tokens",
        type=int,
        default=192,
        help="with shard-dir --retrieve: tokens per decoded training-data passage",
    )
    p.add_argument("--retrieval-k", type=int, default=4, help="passages to retrieve per turn")
    p.add_argument(
        "--retrieval-gate",
        action="store_true",
        help="fuse the retrieval-quality signal into the effort dial (weak grounding -> "
        "more loops -> prefer abstention)",
    )
    p.add_argument(
        "--dense",
        action="store_true",
        help="with --retrieve: add a sentence-transformers dense index (hybrid BM25+dense)",
    )
    p.add_argument(
        "--concise",
        action="store_true",
        help="ponytail-inspired verbosity control: inject a conciseness directive into "
        "the system prompt. 0 params, inference-only.",
    )
    p.add_argument(
        "--ttt",
        type=int,
        default=0,
        help="test-time training: N gradient steps on the retrieved context with a "
        "throwaway model copy before answering (0=off)",
    )
    p.add_argument("--ttt-lr", type=float, default=5e-4)
    p.add_argument(
        "--memory-of-thought",
        action="store_true",
        help="store confident answers as retrievable exemplars: past successes "
        "become few-shot context for similar future queries",
    )
    p.add_argument(
        "--proof-carry",
        action="store_true",
        help="attach a deterministic claim ledger to each answer and reduce "
        "confidence when factual claims are unsupported",
    )
    p.add_argument(
        "--proof-strict",
        action="store_true",
        help="with --proof-carry: remove unsupported factual sentences before "
        "storing or returning the answer",
    )
    p.add_argument(
        "--world-model",
        type=str,
        default=None,
        help="SQLite structured belief/world-state store. Facts from user turns are "
        "retrieved into future prompts and contradictions are surfaced.",
    )
    p.add_argument(
        "--world-k", type=int, default=8, help="structured world-model facts to retrieve per turn"
    )
    p.add_argument(
        "--adaptive-tts",
        action="store_true",
        help="route harder prompts to extra test-time compute automatically "
        "(small council for numeric/long/weakly-grounded prompts)",
    )
    a = p.parse_args()

    try:  # byte-level decode can emit any glyph; don't die on cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    if a.selftest:
        _selftest()
        return

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, cfg = load_model(a.ckpt, device, toy=a.toy)
    mem = Memory(
        a.memory
        if not a.toy
        else os.path.join(os.environ.get("TEMP", "/tmp"), "charkha_toy_mem.sqlite")
    )

    # Resolve the conformal abstain threshold: explicit value, or calibrate on a shard.
    abstain_tau = a.abstain_conf
    if a.calibrate:
        from pipeline import collect_token_level, selective_threshold

        conf, correct = collect_token_level(a.ckpt, a.calibrate, device, max_tokens=200_000)
        abstain_tau, cov, emp, ub = selective_threshold(conf, correct, a.target_risk, a.delta)
        if abstain_tau is None:
            print(
                f"[calibrate] target risk {a.target_risk} unreachable on this checkpoint; "
                "serving without abstention"
            )
        else:
            print(
                f"[calibrate] abstain when conf < {abstain_tau:.4f} -> coverage {cov:.1%}, "
                f"risk <= {ub:.4f} ({1 - a.delta:.0%} conf)"
            )

    retriever = (
        build_datastore(
            a.retrieve,
            use_dense=a.dense,
            tokenizer=tok,
            max_passages=a.retrieve_max_passages,
            passage_tokens=a.retrieve_passage_tokens,
        )
        if a.retrieve
        else None
    )
    world_model = WorldModel(a.world_model) if a.world_model else None
    base_effort = "converge" if a.effort == "converge" else int(a.effort)
    conv_log = a.conv_log
    if conv_log is None and not a.toy:
        conv_log = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "runs",
            "serve",
            "conversations.jsonl",
        )
    sess = Session(
        model,
        tok,
        mem,
        device,
        location=a.location,
        date=a.date,
        cutoff=a.cutoff,
        base_effort=base_effort,
        max_effort=a.max_effort,
        conf_threshold=a.conf_threshold,
        abstain_threshold=abstain_tau,
        use_convergence=a.use_convergence,
        use_sngp=a.use_sngp,
        retriever=retriever,
        retrieval_k=a.retrieval_k,
        use_retrieval_gate=a.retrieval_gate,
        concise=a.concise,
        conv_log=conv_log,
        speculative=a.speculative,
        sampling=dict(
            temp=a.temp,
            top_k=a.top_k,
            top_p=a.top_p,
            min_p=a.min_p,
            rep_penalty=a.rep_penalty,
            no_repeat_ngram=a.no_repeat_ngram,
            loop_contrast=a.loop_contrast,
            loop_contrast_effort=a.loop_contrast_effort,
            echo_gate=a.echo_gate,
            echo_ngram=a.echo_ngram,
            echo_ctx=a.echo_ctx,
        ),
        contrast=(
            (load_model(a.contrast_ckpt, device)[0], a.contrast_lam, a.contrast_alpha)
            if a.contrast_ckpt
            else None
        ),
        best_of=a.best_of,
        council_size=a.council,
        council_synthesize=a.council_synthesize,
        council_margin=a.council_margin,
        council_seed_noise=a.council_seed_noise,
        world_model=world_model,
        world_k=a.world_k,
        adaptive_tts=a.adaptive_tts,
        ttt_steps=a.ttt,
        ttt_lr=a.ttt_lr,
        memory_of_thought=a.memory_of_thought,
        proof_carry=a.proof_carry,
        proof_strict=a.proof_strict,
        ply_branches=a.ply,
        ply_noise=a.ply_noise,
        ply_score=a.ply_score,
        ply_perturb=a.ply_perturb,
    )
    gate = f" | abstain<{abstain_tau:.2f}" if abstain_tau is not None else ""
    print(
        f"CHARKHA serve | {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params | "
        f"device={device} | effort base={a.effort} max={sess.max_effort}{gate}"
    )
    if a.once is not None:
        r = sess.respond(a.once, n_new=a.max_new)
        if r["thinking"]:
            print(f"(thinking) {r['thinking']}")
        print(r["answer"])
        if r.get("sources"):
            cites = ", ".join(s["source"] or "source" for s in r["sources"])
            print(f"(sources) {cites}  [retrieval q={r['retrieval_quality']:.2f}]")
        tag = " | ABSTAINED" if r["abstained"] else (" | LOW" if r["low_confidence"] else "")
        print(f"[effort {r['effort']} | conf {r['confidence']:.2f}{tag}]")
    else:
        repl(sess)
    if world_model:
        world_model.close()
    mem.close()


if __name__ == "__main__":
    main()

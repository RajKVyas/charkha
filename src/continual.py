"""
Continual training loop: interleaves text pretraining, latent-space task RL,
and retrieval-augmented learning in a single run that can continue indefinitely.

Phases (scheduled by ratio):
  TEXT:    standard next-token CE on pretraining data (knowledge)
  TASK:    task → answer → reward → AWR on core trajectory (reasoning)
  RETRIEVE: BM25 retrieval → augmented context → CE (grounded knowledge)

Each step goes through the SAME model forward pass. Losses from all active
components are summed and backpropagated together — no separate optimizers,
no phase boundaries that reset momentum.

"""

import json
import time
import math
from pathlib import Path
from collections import deque

import torch
import torch.nn.functional as F

from charkha import CharkhaConfig, Charkha
from tasks import TaskGenerator, set_tokenizer


# ── Replay buffer for task trajectories ──


class ReplayBuffer:
    """Stores (task_prompt_ids, answer_ids, reward, traj_len) tuples.
    Uses a deque with maxlen so old experiences are automatically evicted.
    Positive and negative experiences stored separately for balanced sampling.
    Latent state storage for somatic wake-sleep consolidation."""

    def __init__(self, capacity=1000, latent_capacity=100):
        self.pos = deque(maxlen=capacity)  # successful episodes
        self.neg = deque(maxlen=capacity)  # failed episodes
        self.latents = deque(maxlen=latent_capacity)  # latent states for sleep phase

    def add(self, prompt_ids, answer_ids, reward, traj_len, latent=None):
        entry = (prompt_ids, answer_ids, reward, traj_len)
        if reward > 0:
            self.pos.append(entry)
        else:
            self.neg.append(entry)
        if latent is not None:
            self.latents.append(latent.detach().cpu())

    def sample(self, n, pos_ratio=0.5):
        """Sample n entries, pos_ratio fraction from successes."""
        n_pos = min(int(n * pos_ratio), len(self.pos))
        n_neg = min(n - n_pos, len(self.neg))
        n_pos = min(n - n_neg, len(self.pos))  # rebalance
        import random

        samples = []
        if n_pos > 0:
            samples.extend(random.sample(list(self.pos), n_pos))
        if n_neg > 0:
            samples.extend(random.sample(list(self.neg), n_neg))
        return samples

    def __len__(self):
        return len(self.pos) + len(self.neg)


# ── Continual trainer ──


class ContinualTrainer:
    """Unified training loop mixing text CE, task RL, and retrieval."""

    def __init__(
        self,
        model,
        optimizer,
        tokenizer,
        out_dir,
        text_loader=None,  # iterable of (idx, targets) batches
        retrieval_store=None,  # Datastore instance for BM25 (optional)
        task_gen=None,  # TaskGenerator (optional)
        phase_ratio=(0.7, 0.2, 0.1),  # text:task:retrieve ratios
        replay_capacity=1000,
        grad_clip=1.0,
        log_every=10,
        eval_every=500,
        save_every=1000,
        # Wake-sleep somatic consolidation
        somatic_interval=200,  # sleep every N active steps (0=off)
        somatic_batch=4,  # latent states per sleep step
        somatic_noise=0.1,  # reconstruction noise level
        latent_capacity=100,  # max latent states in buffer
        # SGS-style penalties
        overlong_threshold=0.8,  # fraction of max_seq_len triggering penalty
        overlong_penalty_scale=5.0,  # penalty multiplier
        stall_threshold=0.9,  # fraction of tokens at max depth => stall
        # Mastery-gated curriculum: difficulty advances ONLY on demonstrated mastery of a
        # HELD-OUT quiz (generalization + calibration + fluency), never on step/token counts.
        quiz_interval=0,  # run a held-out quiz every N active steps (0=off)
        quiz_size=16,  # fresh held-out tasks per quiz
        mastery_acc=0.8,  # min held-out exact-match accuracy to advance
        mastery_calib_gap=0.15,  # max |mean answer-confidence - accuracy| to advance
        mastery_fluency=0.5,  # min fluency = 1 - mean core loops / max loops, to advance
        curriculum_step=0.05,  # difficulty delta on advance / regress
    ):
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.text_loader = text_loader
        self.retrieval_store = retrieval_store
        self.task_gen = task_gen
        self.phase_ratio = phase_ratio
        self.replay = ReplayBuffer(replay_capacity, latent_capacity)
        self.grad_clip = grad_clip
        self.log_every = log_every
        self.eval_every = eval_every
        self.save_every = save_every
        # Wake-sleep
        self.somatic_interval = somatic_interval
        self.somatic_batch = somatic_batch
        self.somatic_noise = somatic_noise
        self.somatic_loss_ema = 0.0
        # SGS penalties
        self.overlong_threshold = overlong_threshold
        self.overlong_penalty_scale = overlong_penalty_scale
        self.stall_threshold = stall_threshold
        self._avg_steps = None
        # Mastery-gated curriculum
        self.quiz_interval = quiz_interval
        self.quiz_size = quiz_size
        self.mastery_acc = mastery_acc
        self.mastery_calib_gap = mastery_calib_gap
        self.mastery_fluency = mastery_fluency
        self.curriculum_step = curriculum_step
        self.last_quiz = None

        # state
        self.step_idx = 0
        self.curriculum_difficulty = 0.0
        self.task_successes = 0
        self.task_total = 0
        self.metrics = {
            "text_loss": deque(maxlen=100),
            "task_loss": deque(maxlen=100),
            "retrieve_loss": deque(maxlen=100),
            "task_success_rate": deque(maxlen=100),
        }

        self._running = False
        self._best_loss = float("inf")

        if self.task_gen:
            self.task_gen.difficulty = self.curriculum_difficulty
        self.device = next(model.parameters()).device
        self._text_iter = iter(text_loader) if text_loader is not None else None

    # ── Phase selection ──

    def _pick_phase(self):
        """Pick TEXT/TASK/RETRIEVE by ratio. TASK only if task_gen exists;
        RETRIEVE only if retrieval_store exists."""
        r = torch.rand(1).item()
        txt_r, task_r, ret_r = self.phase_ratio
        has_task = self.task_gen is not None
        has_ret = self.retrieval_store is not None

        if not has_task and not has_ret:
            return "text"
        if not has_task:
            # redistribute task ratio to text
            txt_r = txt_r + task_r
            task_r = 0.0
        if not has_ret:
            txt_r = txt_r + ret_r
            ret_r = 0.0

        if r < txt_r:
            return "text"
        elif r < txt_r + task_r:
            return "task"
        else:
            return "retrieve"

    # ── Main step ──

    def step(self):
        """One training step: pick phase, run forward, compute loss, backward, update."""
        phase = self._pick_phase()
        self.model.train()

        if phase == "text" and self.text_loader is not None:
            loss, parts = self._text_step()
        elif phase == "task" and self.task_gen is not None:
            loss, parts = self._task_step()
        elif phase == "retrieve" and self.retrieval_store is not None:
            loss, parts = self._retrieve_step()
        else:
            # fallback if requested phase unavailable
            if self.text_loader is not None:
                loss, parts = self._text_step()
            else:
                return None

        # backward + update
        self.optimizer.zero_grad()
        loss.backward()
        # somatic wake-sleep: every N steps, replay latent states. MUST run BEFORE optimizer.step()
        # so its reconstruction gradients accumulate into the SAME .grad buffers and actually get
        # applied — running it after step() (then zero_grad() next iter) silently discarded them,
        # making the whole consolidation a no-op that only burned compute.
        if (
            self.somatic_interval > 0
            and self.step_idx > 0
            and self.step_idx % self.somatic_interval == 0
        ):
            self._somatic_step()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        self.step_idx += 1

        # logging
        key = f"{phase}_loss"
        self.metrics[key].append(loss.item())
        if phase == "task":
            self.metrics["task_success_rate"].append(self.task_successes / max(1, self.task_total))

        if self.step_idx % self.log_every == 0:
            self._log(loss, parts, phase)

        if self.step_idx % self.eval_every == 0:
            self._eval()

        if (
            self.quiz_interval > 0
            and self.task_gen is not None
            and self.step_idx % self.quiz_interval == 0
        ):
            self._maybe_advance_curriculum()

        if self.step_idx % self.save_every == 0:
            self._save()

        return loss.item()

    # ── Wake-sleep somatic consolidation ──

    def _somatic_step(self):
        """Wake-sleep somatic consolidation: replay stored POST-PRELUDE shallow features (the space
        the recurrent core actually consumes) and train core+coda to be ROBUST to small perturbations
        of them — a denoising-CONSISTENCY objective: a noisy shallow feature must yield the same final
        representation as the clean one.

        Two earlier bugs are fixed here: (1) the latents were post-coda/post-norm_f states replayed
        back through the core, which never sees that representation space (out of distribution); they
        are now Charkha.shallow() outputs. (2) Reconstructing the latent *itself* would ask core+coda
        to behave as an identity autoencoder, fighting their real transform — the target is now the
        model's own CLEAN output (detached), so the signal is robustness, not identity. Stored latents
        have per-prompt-varying length, so they are replayed one at a time rather than stacked.
        Called every somatic_interval active steps; runs BEFORE optimizer.step() so its grads apply."""
        if len(self.replay.latents) < 4:
            return  # need a few latents to start
        import random

        n = min(self.somatic_batch, len(self.replay.latents))
        indices = random.sample(range(len(self.replay.latents)), n)
        dev = self.model.embed.weight.device

        def core_coda(s):
            x = (
                self.model._run_core_fixed(s, r=2)
                if self.model.cfg.use_recurrence
                else self.model._run_blocks(self.model.core, s)
            )
            return self.model.norm_f(self.model._run_blocks(self.model.coda, x))

        total = 0.0
        for i in indices:
            shallow = self.replay.latents[i].to(dev)  # (1, P_i, d) post-prelude shallow
            with torch.no_grad():  # clean target, detached
                target = core_coda(shallow)
            noisy = shallow + torch.randn_like(shallow) * self.somatic_noise
            loss_i = F.mse_loss(core_coda(noisy), target)  # noisy path must match clean output
            loss_i.backward()  # accumulate into the shared .grad
            total += loss_i.detach().item()
        self.somatic_loss_ema = 0.99 * self.somatic_loss_ema + 0.01 * (total / max(1, n))

    # ── Text pretraining step ──

    def _text_step(self):
        idx, targets = self._next_text_batch()
        idx = idx.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        _, loss = self.model(idx, targets=targets)
        parts = self.model._last_loss_parts or {}
        return loss, parts

    def _next_text_batch(self):
        if self.text_loader is None:
            raise RuntimeError("text_loader is required for text/retrieve phases")
        if self._text_iter is None:
            self._text_iter = iter(self.text_loader)
        try:
            return next(self._text_iter)
        except StopIteration:
            fresh = iter(self.text_loader)
            if fresh is self._text_iter:
                # text_loader IS an iterator/generator: iter() returns the same exhausted object,
                # so the old "restart" was a no-op and the retry re-raised StopIteration into the
                # trainer. Fail loud with the actual fix instead.
                raise RuntimeError(
                    "text_loader is exhausted and not re-iterable; pass a re-iterable "
                    "(list / Dataset / DataLoader) or an infinite generator"
                ) from None
            self._text_iter = fresh
            return next(self._text_iter)

    def _encode(self, text):
        ids = self.tokenizer.encode(text)
        if hasattr(ids, "ids"):
            ids = ids.ids
        return list(ids)

    # ── Task RL step ──

    def _task_step(self):
        """Generate task, run model forward, check answer, compute AWR loss."""
        task = self.task_gen.generate()
        prompt_ids = self._encode(task.prompt)
        answer_ids = self._encode(task.answer)
        eos = self.tokenizer.eos_token_id if hasattr(self.tokenizer, "eos_token_id") else 0

        # tokenize as full sequence: prompt + answer + eos
        full_ids = prompt_ids + answer_ids + [eos]
        full = torch.tensor([full_ids], dtype=torch.long, device=self.device)
        prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        # Greedy probe for reward only. The gradient path below is supervised CE + task_loss.
        with torch.no_grad():
            gen = self.model.generate(prompt_t, max(1, len(answer_ids)), effort=1)
        model_tokens = gen[0, len(prompt_ids) : len(prompt_ids) + len(answer_ids)].tolist()
        model_answer = self.tokenizer.decode(model_tokens)

        # reward: 1.0 if correct, -1.0 if wrong
        correct = task.answer.strip().lower() == model_answer.strip().lower()
        raw_reward = 1.0 if correct else -1.0

        # SGS overlong penalty: penalize trajectories consuming >80% context
        utilization = len(answer_ids) / self.model.cfg.max_seq_len
        if utilization > self.overlong_threshold:
            penalty = (
                (utilization - self.overlong_threshold) / (1.0 - self.overlong_threshold)
            ) * self.overlong_penalty_scale
            raw_reward = raw_reward - penalty

        # SGS stall penalty: if model hit max depth on most tokens, it's guessing blindly.
        # Capture from halting mode (uses _run_core_halting which tracks per-token steps).
        # In fixed-r mode, this is a no-op (all tokens at same depth).
        if hasattr(self.model, "_n_core_steps"):
            core_steps = getattr(self.model, "_n_core_steps", 0)
            max_allowed = self.model.cfg.max_recurrence_infer * 0.9
            # _n_core_steps is total passes across all tokens; approximate stall fraction
            # from the halting distribution stored in the model
            halts = getattr(self.model, "_last_halts", None)
            if halts is not None and halts.numel() > 0:
                # halts: (B, T, N) — halt probability at each loop. Stall = almost all prob
                # mass in the last loop => token didn't halt early, kept thinking at max depth.
                stall_frac = halts[:, :, -1].mean().item()  # frac of prob mass in last loop
                if stall_frac > self.stall_threshold:
                    raw_reward = 0.0  # no learning signal from blind guessing

        reward = torch.tensor(raw_reward, device=full.device)

        self.task_total += 1
        if correct:
            self.task_successes += 1

        # Store the POST-PRELUDE shallow feature (the representation the recurrent core consumes) for
        # the somatic sleep phase — NOT the post-coda hidden state. Replaying a post-coda/post-norm_f
        # state back through core+coda would be out of distribution (core never sees that space), which
        # is the bug this fixes. See Charkha.shallow() and _somatic_step().
        latent = self.model.shallow(prompt_t).detach() if hasattr(self.model, "shallow") else None

        # Teacher-forced CE keeps the task text path useful even before the policy can solve it.
        _, ce_loss = self.model(full[:, :-1], targets=full[:, 1:])
        self.model.task_forward(full[:, :-1])
        rl_loss, parts = self.model.task_loss(reward, targets=full[:, 1:])
        rl_loss = rl_loss.to(self.device)
        task_loss = ce_loss + 0.1 * rl_loss
        parts = dict(parts or {})
        parts["task_ce"] = ce_loss.detach()
        parts["task_rl"] = rl_loss.detach()

        # store in replay buffer (with latent for sleep)
        self.replay.add(
            prompt_ids, answer_ids, reward.item(), len(self.model._core_traj or []), latent=latent
        )
        if len(self.replay) >= 4:
            replay_loss = self._replay_step()
            task_loss = task_loss + 0.1 * replay_loss
            parts["replay"] = replay_loss.detach()
        return task_loss, parts

    def _replay_step(self):
        """Replay a batch from the buffer to prevent catastrophic forgetting."""
        samples = self.replay.sample(4, pos_ratio=0.5)
        total = torch.zeros((), device=self.device)
        for prompt_ids, answer_ids, reward_val, _traj_len in samples:
            full_ids = prompt_ids + answer_ids + [0]
            full = torch.tensor([full_ids], dtype=torch.long, device=self.device)
            prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
            reward = torch.tensor(reward_val, device=full.device)
            self.model.task_forward(prompt_t)
            rl, _ = self.model.task_loss(reward, targets=full[:, 1:])
            total = total + rl.to(self.device)
        return total / max(1, len(samples))

    # ── Retrieval-augmented step ──

    def _retrieve_step(self):
        """Text step with BM25 retrieval augmentation."""
        if self.text_loader is None:
            return self._text_step()
        idx, targets = self._next_text_batch()
        idx = idx.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        # decode a slice of the batch to use as retrieval query
        slice_t = idx[0, :64]
        query = self.tokenizer.decode(slice_t.tolist())

        # retrieve relevant passages
        if hasattr(self.retrieval_store, "search"):
            results = self.retrieval_store.search(query, k=2)
        else:
            results = self.retrieval_store.retrieve(query, k=2)

        # prepend to the sequence (simple approach: just log for now)
        # Full retrieval-augmented forward would require concatenating
        # retrieved passages. For now, we just compute standard CE.
        _, loss = self.model(idx, targets=targets)
        parts = self.model._last_loss_parts or {}
        parts["retrieved"] = torch.tensor(float(len(results)))
        return loss, parts

    # ── Logging / eval / save ──

    def _log(self, loss, parts, phase):
        avg_text = (
            sum(self.metrics["text_loss"]) / max(1, len(self.metrics["text_loss"]))
            if self.metrics["text_loss"]
            else 0
        )
        avg_task = (
            sum(self.metrics["task_loss"]) / max(1, len(self.metrics["task_loss"]))
            if self.metrics["task_loss"]
            else 0
        )
        sr = self.metrics["task_success_rate"][-1] if self.metrics["task_success_rate"] else 0
        diff = self.task_gen.difficulty if self.task_gen else 0

        print(
            f"step {self.step_idx:>6d} | {phase:>8s} loss {loss.item():.4f} | "
            f"text_avg {avg_text:.3f} task_avg {avg_task:.3f} | "
            f"sr {sr:.2f} diff {diff:.2f}"
        )
        for k, v in (parts or {}).items():
            print(f"  {k}: {v.item():.4f}")

        # write metrics
        entry = {
            "step": self.step_idx,
            "phase": phase,
            "loss": loss.item(),
            "parts": {k: v.item() for k, v in (parts or {}).items()},
            "text_avg": avg_text,
            "task_avg": avg_task,
            "task_sr": sr,
            "difficulty": diff,
            "timestamp": time.time(),
        }
        with open(self.out_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _eval(self):
        """Quick evaluation: average loss on a few text batches."""
        if self.text_loader is None:
            return
        self.model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(5):
                try:
                    idx, targets = self._next_text_batch()
                    idx = idx.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    _, loss = self.model(idx, targets=targets)
                    losses.append(loss.item())
                except StopIteration:
                    break
        self.model.train()
        avg = sum(losses) / max(1, len(losses))
        if avg < self._best_loss:
            self._best_loss = avg
            self._save(tag="best")
        print(f"  eval@{self.step_idx}: {avg:.4f} (best {self._best_loss:.4f})")

    # ── Mastery-gated curriculum ──

    def _quiz(self, n=None):
        """Held-out mastery probe: generate FRESH tasks at the current difficulty (never trained on
        or stored in replay) and measure three INDEPENDENT things — generalization (exact-match
        accuracy), confidence calibration (|mean answer-confidence - accuracy|, lower is better), and
        fluency (1 - mean core loops / max loops; high = solves without thrashing to max depth). Pure
        eval: no gradients, no replay writes, greedy decode for reproducibility. Returns a dict/None."""
        if self.task_gen is None:
            return None
        n = n or self.quiz_size
        was_training = self.model.training
        self.model.eval()
        loops_max = max(1, self.model.cfg.max_recurrence_infer)
        correct = conf_sum = loops_sum = 0.0
        counted = 0
        with torch.no_grad():
            for _ in range(n):
                task = self.task_gen.generate()
                prompt_ids = self._encode(task.prompt)
                answer_ids = self._encode(task.answer)
                if not prompt_ids or not answer_ids:
                    continue
                prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
                self.model._n_core_steps = 0  # reset the fluency counter
                gen = self.model.generate(prompt_t, len(answer_ids), effort=None, temp=0.0)
                loops_sum += self.model._n_core_steps / max(1, len(answer_ids))
                ans = gen[:, len(prompt_ids) : len(prompt_ids) + len(answer_ids)]
                ok = (
                    task.answer.strip().lower()
                    == self.tokenizer.decode(ans[0].tolist()).strip().lower()
                )
                correct += float(ok)
                # confidence on the model's own answer span (eval forward returns per-token conf)
                _, conf = self.model(torch.cat([prompt_t, ans], dim=1))
                conf_sum += conf[0, len(prompt_ids) :].mean().item()
                counted += 1
        if was_training:
            self.model.train()
        if counted == 0:
            return None
        acc = correct / counted
        conf = conf_sum / counted
        fluency = 1.0 - min(1.0, (loops_sum / counted) / loops_max)
        return {
            "difficulty": self.task_gen.difficulty,
            "n": counted,
            "accuracy": acc,
            "confidence": conf,
            "calibration_gap": abs(conf - acc),
            "fluency": fluency,
        }

    def _maybe_advance_curriculum(self):
        """Advance the curriculum ONLY on demonstrated mastery of a held-out quiz — generalization
        AND calibration AND fluency together — never on step/token counts. Regress when accuracy is
        far below target (level too hard); otherwise hold. Mutates difficulty in place + logs."""
        q = self._quiz()
        if q is None:
            return None
        mastered = (
            q["accuracy"] >= self.mastery_acc
            and q["calibration_gap"] <= self.mastery_calib_gap
            and q["fluency"] >= self.mastery_fluency
        )
        if mastered:
            self.curriculum_difficulty = min(1.0, self.curriculum_difficulty + self.curriculum_step)
            verdict = "advance"
        elif q["accuracy"] < self.mastery_acc * 0.5:
            self.curriculum_difficulty = max(0.0, self.curriculum_difficulty - self.curriculum_step)
            verdict = "regress"
        else:
            verdict = "hold"
        self.task_gen.difficulty = self.curriculum_difficulty
        self.last_quiz = {**q, "verdict": verdict, "new_difficulty": self.curriculum_difficulty}
        print(
            f"  quiz@{self.step_idx}: acc={q['accuracy']:.2f} calib_gap={q['calibration_gap']:.2f} "
            f"fluency={q['fluency']:.2f} -> {verdict} (difficulty {self.curriculum_difficulty:.2f})"
        )
        return self.last_quiz

    def _save(self, tag=None):
        name = f"{'best' if tag == 'best' else f'step_{self.step_idx}'}.pt"
        path = self.out_dir / name
        torch.save(
            {
                "step": self.step_idx,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "curriculum_difficulty": self.curriculum_difficulty,
                "best_loss": self._best_loss,
            },
            path,
        )
        print(f"  saved {path}")

    # ── Run loop ──

    def run(self, max_steps=None, stop_on=None):
        """Run the training loop. max_steps=None runs forever.
        stop_on: optional callable(stats) -> bool for early stopping."""
        self._running = True
        try:
            while self._running:
                if max_steps and self.step_idx >= max_steps:
                    break
                loss = self.step()
                if stop_on and stop_on(self.stats()):
                    break
        except KeyboardInterrupt:
            print(f"\nInterrupted at step {self.step_idx}. Saving...")
            self._save(tag="interrupt")
        return self.stats()

    def stats(self):
        return {
            "step": self.step_idx,
            "curriculum_difficulty": self.task_gen.difficulty if self.task_gen else 0,
            "task_success_rate": (
                sum(self.metrics["task_success_rate"])
                / max(1, len(self.metrics["task_success_rate"]))
            ),
            "best_loss": self._best_loss,
            "replay_size": len(self.replay),
        }


# ── Convenience factory ──


def create_trainer(
    out_dir,
    tokenizer,
    small=True,
    text_data_dirs=None,
    retrieval_store=None,
    phase_ratio=(0.7, 0.2, 0.1),
    lr=6e-4,
    steps=None,
    replay_capacity=1000,
    device="cuda",
):
    """Create and return a ContinualTrainer with sensible defaults."""
    from train import ShardLoader, seed_everything

    cfg = CharkhaConfig.small() if small else CharkhaConfig()
    cfg.use_gdn2 = True
    cfg.use_deep_supervision = True
    cfg.cross_loop_consistency = True
    cfg.use_halting = True
    cfg.use_task_rl = True
    cfg.grad_checkpoint = True

    seed_everything(42)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = Charkha(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True if device == "cuda" else False,
    )

    text_loader = None
    if text_data_dirs:
        shards = ShardLoader(text_data_dirs)

        def _batches():
            import random

            rng = random.Random(42)
            while True:
                yield shards.batch(
                    2, min(2048, cfg.max_seq_len), device, rng, pin=(device == "cuda")
                )

        text_loader = _batches()

    task_gen = TaskGenerator(difficulty=0.0)
    set_tokenizer(tokenizer)

    trainer = ContinualTrainer(
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        out_dir=out_dir,
        text_loader=text_loader,
        retrieval_store=retrieval_store,
        task_gen=task_gen,
        phase_ratio=phase_ratio,
        replay_capacity=replay_capacity,
    )
    return trainer


class ByteTokenizer:
    eos_token_id = 0

    def encode(self, text):
        return [b + 1 for b in text.encode("utf-8", errors="replace")]

    def decode(self, ids):
        return bytes(max(0, min(255, int(i) - 1)) for i in ids if int(i) > 0).decode(
            "utf-8", errors="replace"
        )


def selftest():
    import tempfile

    print("CHARKHA continual self-test")
    tok = ByteTokenizer()
    cfg = CharkhaConfig.toy()
    cfg.use_gdn2 = True
    cfg.use_recurrence = True
    cfg.use_halting = False
    cfg.use_task_rl = True
    cfg.use_deep_supervision = True
    cfg.cross_loop_consistency = True
    cfg.max_seq_len = 64
    model = Charkha(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    seq = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    text_loader = iter([(seq[:, :-1], seq[:, 1:])] * 8)
    trainer = ContinualTrainer(
        model=model,
        optimizer=opt,
        tokenizer=tok,
        out_dir=tempfile.mkdtemp(prefix="charkha_continual_"),
        text_loader=text_loader,
        task_gen=TaskGenerator(difficulty=0.0, seed=1),
        phase_ratio=(1.0, 0.0, 0.0),
        log_every=1000,
        eval_every=1000,
        save_every=1000,
        somatic_interval=0,
    )
    text_loss = trainer.step()
    task_loss, task_parts = trainer._task_step()
    trainer.optimizer.zero_grad()
    task_loss.backward()
    trainer.optimizer.step()
    losses = [text_loss, float(task_loss.detach())]
    checks = {
        "step method callable": trainer.step_idx == 1,
        "finite losses": all(l is not None and math.isfinite(float(l)) for l in losses),
        "task replay populated": len(trainer.replay) > 0,
        "task loss exposes CE": "task_ce" in task_parts,
        "stats reports step": trainer.stats()["step"] == 1,
    }
    ok = True
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= passed
    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="CHARKHA continual training loop")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    ap.error("no run CLI yet; use create_trainer(...) from Python or --selftest")

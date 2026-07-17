"""
Procedural task generator for latent-space RL training.
Produces infinite streams of verifiable tasks (arithmetic, logic, pattern-completion)
with curriculum difficulty. Each task has a text prompt, a verifiable answer,
and a scalar difficulty. No human data — purely synthetic.

Task types (extensible):
  Arithmetic:     "What is 17 + 24?"           →  "41"
  Algebra:        "Solve: 3x + 7 = 22"         →  "5"
  Logic puzzles:  "If A > B and B > C, is A > C?" → "yes"
  Pattern:        "2, 4, 8, 16, ?"             →  "32"
  Word scramble:  "Unscramble: elhlo"          →  "hello"
  Code snippets:  "What does 3 << 2 evaluate to?" → "12"
  Fraction:       "1/2 + 1/3 = ?"             →  "5/6"

"""

import random
import math
from dataclasses import dataclass, field

# ── reusable tokenizer reference (lazily set by caller) ──
_tokenizer = None
_eos_token_id = None


def set_tokenizer(tok, eos_id=0):
    global _tokenizer, _eos_token_id
    _tokenizer = tok
    _eos_token_id = eos_id


@dataclass
class Task:
    prompt: str  # natural-language prompt
    answer: str  # ground-truth answer string
    difficulty: float  # 0.0 (trivial) to 1.0 (hard)
    kind: str  # "arithmetic", "logic", "pattern", etc.
    tokens: list = field(default_factory=list)  # tokenized prompt (set after generation)


class TaskGenerator:
    """Procedural task generator with curriculum difficulty."""

    def __init__(self, difficulty=0.0, seed=None):
        self.difficulty = difficulty  # 0.0-1.0, controls operand ranges etc.
        self.rng = random.Random(seed)
        self.counters = {kind: 0 for kind in self._task_types()}

    @staticmethod
    def _task_types():
        return ["arithmetic", "algebra", "logic", "pattern", "scramble", "fraction", "spatial"]

    def generate(self, n=1, kind=None):
        """Generate n tasks. kind=None picks from all types weighted by difficulty."""
        tasks = []
        for _ in range(n):
            k = kind or self.rng.choice(self._task_types())
            t = self._generate_one(k)
            if _tokenizer:
                # tokenize prompt + answer for the model
                prompt_ids = _tokenizer.encode(t.prompt)
                answer_ids = _tokenizer.encode(t.answer)
                full = prompt_ids + answer_ids + [_eos_token_id]
                t.tokens = full
            tasks.append(t)
            self.counters[k] += 1
        return tasks if n > 1 else tasks[0]

    def _generate_one(self, kind):
        d = self.difficulty
        if kind == "arithmetic":
            return self._arithmetic(d)
        elif kind == "algebra":
            return self._algebra(d)
        elif kind == "logic":
            return self._logic(d)
        elif kind == "pattern":
            return self._pattern(d)
        elif kind == "scramble":
            return self._scramble(d)
        elif kind == "fraction":
            return self._fraction(d)
        elif kind == "spatial":
            return self._spatial(d)
        else:
            return self._arithmetic(d)

    # ── Arithmetic ──
    def _arithmetic(self, d):
        """Addition, multiplication, mixed. d scales operand size."""
        max_n = int(2 + d * 998)  # 2 → 1000
        ops = [(lambda a, b: a + b, "+"), (lambda a, b: a - b, "-"), (lambda a, b: a * b, "×")]
        if d < 0.3:
            ops = ops[:1]  # addition only at low difficulty
        elif d < 0.6:
            ops = ops[:2]  # add + sub
        fn, sym = self.rng.choice(ops)
        a = self.rng.randint(0, max_n)
        b = self.rng.randint(0, max_n)
        if sym == "-" and b > a:
            a, b = b, a  # keep non-negative
        result = fn(a, b)
        return Task(
            prompt=f"Calculate: {a} {sym} {b} = ?",
            answer=str(result),
            difficulty=d,
            kind="arithmetic",
        )

    # ── Algebra ──
    def _algebra(self, d):
        """Simple linear equations: ax + b = c."""
        max_coef = int(2 + d * 48)  # 2 → 50
        # ensure integer solution
        x = self.rng.randint(1, max(2, int(d * 100)))
        a = self.rng.randint(1, max_coef)
        b = self.rng.randint(-max_coef, max_coef)
        c = a * x + b
        eq = f"{a}x + {b} = {c}" if b >= 0 else f"{a}x - {abs(b)} = {c}"
        return Task(
            prompt=f"Solve for x: {eq}",
            answer=str(x),
            difficulty=d,
            kind="algebra",
        )

    # ── Logic ──
    _LOGIC_PAIRS = [
        ("A > B and B > C", "Is A > C?", "yes"),
        ("A < B and B < C", "Is A < C?", "yes"),
        ("A = B and B = C", "Is A = C?", "yes"),
        ("A > B and B < C", "Can we conclude A > C?", "no"),
        ("all cats are mammals; all mammals are animals", "Are all cats animals?", "yes"),
        ("some birds can fly; penguins are birds", "Can all birds fly?", "no"),
        ("if it rains, the ground is wet; the ground is wet", "Did it rain?", "not necessarily"),
    ]

    def _logic(self, d):
        premise, question, answer = self.rng.choice(self._LOGIC_PAIRS)
        return Task(
            prompt=f"Given: {premise}. {question}",
            answer=answer,
            difficulty=min(d + 0.3, 1.0),
            kind="logic",
        )

    # ── Pattern completion ──
    # Each generator draws its parameters ONCE per sequence (NOT inside the comprehension — that
    # redrew a fresh random per term, producing e.g. [5,2,12,8,64] which is no pattern at all and
    # gave the solver an unlearnable prompt with a meaningless gold answer). Each returns the full
    # 5-term sequence plus next_fn, which computes the genuine 6th term.
    _PATTERN_GENERATORS = [
        # geometric: a, a·ratio, a·ratio², …  (ratio drawn once)
        lambda rng, d: (
            lambda a, ratio: ([a * ratio**i for i in range(5)], lambda seq: str(seq[-1] * ratio))
        )(rng.randint(1, 5), rng.randint(2, 3)),
        # arithmetic: start, start+step, …  (start, step drawn once)
        lambda rng, d: (
            lambda start, step: (
                [start + step * i for i in range(5)],
                lambda seq: str(seq[-1] + step),
            )
        )(rng.randint(1, 9), rng.randint(2, 9)),
        # quadratic base + i²  (base drawn once); 2nd difference is a constant 2
        lambda rng, d: (
            lambda base: (
                [base + i**2 for i in range(5)],
                lambda seq: str(2 * seq[-1] - seq[-2] + 2),
            )
        )(rng.randint(1, 10)),
    ]

    def _pattern(self, d):
        gen = self.rng.choice(self._PATTERN_GENERATORS)
        seq, next_fn = gen(self.rng, d)
        # show ALL generated terms and ask for the NEXT one. next_fn computes the term after
        # seq[-1]; showing only seq[:4] made the prompt ask for the 5th term while the gold answer
        # was the 6th — an off-by-one-wrong answer on top of the unlearnable-sequence bug above.
        show = seq
        ans = next_fn(seq)
        return Task(
            prompt=f"Complete the pattern: {', '.join(map(str, show))}, ?",
            answer=ans,
            difficulty=d,
            kind="pattern",
        )

    # ── Word scramble ──
    _WORDS = [
        "hello",
        "world",
        "python",
        "think",
        "latent",
        "reward",
        "model",
        "depth",
        "recurrence",
        "compute",
        "value",
        "state",
        "cat",
        "dog",
        "run",
        "jump",
        "fast",
        "blue",
        "green",
    ]

    def _scramble(self, d):
        word = self.rng.choice(self._WORDS)
        chars = list(word)
        self.rng.shuffle(chars)
        scrambled = "".join(chars)
        if scrambled == word:
            scrambled = word[::-1]  # ensure it's actually scrambled
        return Task(
            prompt=f"Unscramble: {scrambled}",
            answer=word,
            difficulty=d * 0.5,
            kind="scramble",
        )

    # ── Fraction arithmetic ──
    def _fraction(self, d):
        max_den = int(2 + d * 18)  # 2 → 20
        n1 = self.rng.randint(1, max_den)
        d1 = self.rng.randint(2, max_den)
        n2 = self.rng.randint(1, max_den)
        d2 = self.rng.randint(2, max_den)
        # a/b + c/d = (ad + bc) / bd
        num = n1 * d2 + n2 * d1
        den = d1 * d2
        g = math.gcd(num, den)
        num //= g
        den //= g
        answer = str(num) if den == 1 else f"{num}/{den}"
        return Task(
            prompt=f"What is {n1}/{d1} + {n2}/{d2}?",
            answer=answer,
            difficulty=min(d + 0.2, 1.0),
            kind="fraction",
        )

    # ── Spatial Reasoning ──
    def _spatial(self, d):
        objects = ["coin", "key", "ring", "marble", "die", "button", "gem"]
        containers = ["cup", "box", "bowl", "mug", "jar", "glass", "bucket"]
        directions = ["left", "right", "forward", "backward", "north", "south"]

        obj = self.rng.choice(objects)
        container = self.rng.choice(containers)
        dir1 = self.rng.choice(directions)
        dist = self.rng.randint(2, 15)

        pattern = self.rng.randint(1, 3)
        if pattern == 1:
            prompt = f"A {container} is turned upside down. A {obj} is placed on top of the {container}. The {container} is then pushed {dist} inches to the {dir1}. Where is the {obj}?"
            answer = f"on top of the {container}"
        elif pattern == 2:
            prompt = f"A {obj} is placed inside a {container}. The {container} is moved {dist} inches to the {dir1}. Where is the {obj}?"
            answer = f"inside the {container}"
        else:
            # Transitive physical relation. The moved distance decides the truth: a 1-2 inch nudge
            # leaves the pair adjacent, a 2-5 foot move separates them. (The old hardcoded "no" was
            # wrong for small distances — it poisoned task-RL rewards and quiz ground truth.)
            obj2 = self.rng.choice([o for o in objects if o != obj])
            near = self.rng.random() < 0.5
            dist = self.rng.randint(1, 2) if near else self.rng.randint(24, 60)
            prompt = f"A {obj} is placed inside a {container}. A {obj2} is placed next to the {container}. The {container} is moved {dist} inches to the {dir1}. Is the {obj} still near the {obj2}?"
            answer = "yes" if near else "no"

        return Task(
            prompt=prompt,
            answer=answer,
            difficulty=min(d + 0.3, 1.0),
            kind="spatial",
        )

    def advance_difficulty(self, success_rate, threshold=0.7):
        """Auto-advance curriculum: increase difficulty when success rate > threshold.
        Returns new difficulty."""
        if success_rate > threshold:
            self.difficulty = min(1.0, self.difficulty + 0.05)
        elif success_rate < 0.3:
            self.difficulty = max(0.0, self.difficulty - 0.05)
        return self.difficulty

    def stats(self):
        return {
            "difficulty": self.difficulty,
            "counts": dict(self.counters),
            "total": sum(self.counters.values()),
        }


if __name__ == "__main__":
    print("Testing task generation...")
    tg = TaskGenerator()
    tasks = []
    for _ in range(100):
        tasks.append(tg.generate())

    counts = {}
    for t in tasks:
        counts[t.kind] = counts.get(t.kind, 0) + 1
        assert t.prompt and t.answer and t.kind
    print(f"Generated 100 tasks successfully across types: {counts}")
    print("PASS: tasks.py selftest")

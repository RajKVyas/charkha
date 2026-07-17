#!/usr/bin/env python3
"""Shared toy task for the CPU micro-experiment harnesses (murmur_micro, ply_micro).

Sort a short digit string: prompt = digits + [SEP], answer = sorted(digits) + [EOS].
Checkable answers, multi-step latent computation, real-vocab ids only (the murmur
band, if any, sits far above these) — the common substrate for micro-E5/E9 so the
experiments stay comparable.

"""

import torch

DIGITS = list(range(1, 9))  # token ids 1..8 are the digits
SEP, EOS = 11, 12  # separator / end markers (real-vocab ids)


def make_pair(n_digits, gen):
    d = [DIGITS[int(torch.randint(0, len(DIGITS), (1,), generator=gen))] for _ in range(n_digits)]
    return d + [SEP], sorted(d) + [EOS]


def make_split(n, n_digits, gen):
    return [make_pair(n_digits, gen) for _ in range(n)]

#!/usr/bin/env python3
"""CHARKHA live dashboard — watch the model get smarter, the teacher corpus grow, and disk stay sane.

Refreshes a compact panel: training loss + EMA + a loss sparkline, val bits/tok & perplexity, tok/s,
recurrence, the latest generated SAMPLE, KD tokens accumulated per teacher, and disk usage. Also
PRUNES old immortal snapshots (keeps the newest --keep) so 0.42B checkpoints don't eat the disk.

  python scripts/charkha_dash.py --run ~/charkha/runs/charkha \
      --kd data/kd-oss --kd data/kd-deepseek --keep 3

Pure stdlib; safe to run in a tmux pane next to training. Read-only except snapshot pruning.

"""

import argparse
import json
import os
import shutil
import sys
import time
import glob

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SPARK = "▁▂▃▄▅▆▇█"


def spark(vals, width=48):
    vals = vals[-width:]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return "".join(SPARK[min(7, int((v - lo) / rng * 7))] for v in vals)


def read_metrics(path):
    rows = []
    if not os.path.exists(path):
        return rows
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        pass
    return rows


def last_sample(log_path):
    if not os.path.exists(log_path):
        return None
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            tail = f.readlines()[-400:]
    except Exception:
        return None
    for line in reversed(tail):
        if "[sample @" in line:
            return line.strip()
    return None


def dir_tokens(d):
    p = os.path.join(d, "index.json")
    if not os.path.exists(p):
        return 0
    try:
        with open(p) as f:
            return int(json.load(f).get("total_tokens", 0))
    except Exception:
        return 0


def du(path):
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total


def human(n):
    for u in ("B", "K", "M", "G", "T"):
        if abs(n) < 1024:
            return f"{n:.0f}{u}" if u == "B" else f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}P"


def prune_snapshots(run, keep):
    snaps = sorted(glob.glob(os.path.join(run, "ckpt_*.pt")))
    removed = 0
    for s in snaps[:-keep] if keep > 0 else []:
        try:
            os.remove(s)
            removed += 1
        except OSError:
            pass
    return removed, max(0, len(snaps) - removed)


def render(a):
    rows = read_metrics(os.path.join(a.run, "metrics.jsonl"))
    train_rows = [r for r in rows if "loss" in r]
    val_rows = [r for r in rows if "val_bits" in r]
    last = train_rows[-1] if train_rows else {}
    val = val_rows[-1] if val_rows else {}
    ema_hist = [r["loss_ema"] for r in train_rows if "loss_ema" in r]
    removed, kept = prune_snapshots(a.run, a.keep)

    out = []
    out.append("═" * 64)
    out.append("  CHARKHA — live training dashboard")
    out.append("═" * 64)
    if last:
        out.append(f"  step      {last.get('step', 0):>10,}")
        out.append(
            f"  loss      {last.get('loss', float('nan')):>10.4f}   "
            f"ema {last.get('loss_ema', float('nan')):.4f}"
        )
        out.append(
            f"  tok/s     {last.get('tok_s', 0):>10,}   "
            f"r{last.get('mean_r', '?')}{'h' if last.get('halting') else 'f'}   "
            f"gnorm {last.get('grad_norm', float('nan')):.2f}"
        )
        if val:
            out.append(
                f"  val       bits/tok {val.get('val_bits', float('nan')):.3f}   "
                f"ppl {val.get('val_ppl', float('nan')):.1f}"
            )
        # bits/token context: random baseline = ln(vocab)/ln(2)
        out.append(f"  loss ↓    {spark(ema_hist)}")
    else:
        out.append("  (waiting for metrics.jsonl — training not started yet)")
    out.append("-" * 64)
    samp = last_sample(os.path.join(a.run, "train.log"))
    out.append("  latest sample:")
    if samp:
        s = samp.split("] ", 1)[-1]
        out.append("    " + (s[:300] + ("…" if len(s) > 300 else "")))
    else:
        out.append("    (no sample yet — appears every --sample-every steps)")
    out.append("-" * 64)
    out.append("  teacher corpus building up:")
    total_kd = 0
    for d in a.kd:
        t = dir_tokens(d)
        total_kd += t
        bar = "█" * min(40, t // max(1, a.kd_bar_unit))
        out.append(f"    {os.path.basename(d):<16} {t:>13,} tok  {bar}")
    out.append(f"    {'TOTAL KD':<16} {total_kd:>13,} tok")
    out.append("-" * 64)
    # disk
    try:
        usage = shutil.disk_usage(a.run)
        free_pct = 100 * usage.free / usage.total
        warn = "  ⚠ LOW" if free_pct < 10 else ""
        out.append(
            f"  disk      run {human(du(a.run))}   "
            f"kd {human(sum(du(d) for d in a.kd if os.path.isdir(d)))}   "
            f"free {human(usage.free)} ({free_pct:.0f}%){warn}"
        )
    except Exception:
        pass
    out.append(
        f"  snapshots kept {kept}"
        + (f", pruned {removed}" if removed else "")
        + f"  (keep={a.keep})"
    )
    out.append("═" * 64)
    out.append(time.strftime("  %Y-%m-%d %H:%M:%S") + f"   refresh {a.every}s   (Ctrl-C to quit)")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="CHARKHA live dashboard")
    ap.add_argument("--run", default="runs/charkha", help="training out dir")
    ap.add_argument("--kd", action="append", default=[], help="KD shard dir (repeatable)")
    ap.add_argument("--keep", type=int, default=3, help="snapshots to retain (prune older)")
    ap.add_argument("--every", type=int, default=10, help="refresh seconds")
    ap.add_argument("--kd-bar-unit", type=int, default=250_000, help="tokens per bar block")
    ap.add_argument("--once", action="store_true", help="render once and exit")
    a = ap.parse_args()
    if not a.kd:
        a.kd = ["data/kd-oss", "data/kd-deepseek"]
    while True:
        panel = render(a)
        if os.name == "nt":
            os.system("cls")
        else:
            # NOT os.system('clear') -- on modern terminfo/ncurses, `clear` emits \e[3J, which wipes
            # the terminal's SCROLLBACK buffer, not just the visible screen (kills your history).
            # \e[H (cursor home) + \e[2J (clear visible screen only) redraws in place, scrollback intact.
            print("\033[H\033[2J", end="")
        print(panel, flush=True)
        if a.once:
            break
        try:
            time.sleep(a.every)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()

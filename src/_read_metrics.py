"""Read training metrics JSONL and print summary."""

import json
import sys


def main():
    path = sys.argv[1]
    lines = [json.loads(l) for l in open(path)]
    first, last = lines[0], lines[-1]
    print(f"Steps: {first['step']} to {last['step']}  ({len(lines)} entries)")
    print(f"Loss: {first['loss']:.2f} -> {last['loss']:.2f}  (EMA {last.get('loss_ema', 0):.2f})")
    print(f"tok/s: {last.get('tok_s', 0):.0f}")
    for k in sorted(last.keys()):
        print(f"  {k}: {last[k]}")


if __name__ == "__main__":
    main()

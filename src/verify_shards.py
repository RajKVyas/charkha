"""
CHARKHA verify_shards - integrity check for dataprep's {index.json, shard_*.bin} output.
=======================================================================================
dataprep.py writes .bin shards (array('H' or 'I').tofile -> native little-endian on x86)
plus an index.json listing per-shard token counts and a
grand total. Bytes-per-token is decided by vocab_size via dataprep.shard_format_for_vocab: 2
(uint16) for vocab<=65535, 4 (uint32) above that (Sutra-131k / the legacy v8 131072 vocab). Before a multi-
day training run you want to know, on EVERY machine that holds shards, that:

  * every shard file named in index.json exists and is readable
  * its byte size is exactly tokens * bytes_per_token (no truncated/half-written shard from a
    killed run, and no shard written at the wrong width for its own index.json's vocab_size)
  * token ids are in range [0, vocab_size) (no corruption / wrong-endian / wrong dtype)
  * the per-shard counts sum to index.json's total_tokens

Cheap by default (structure + a sampled id-range check at the head/tail of each shard, which
catches truncation and dtype/endian corruption without reading 180GB). Pass --deep to scan the
full min/max of every shard (slow; reads every byte).

Usage:
  python verify_shards.py data/corpus-a data/corpus-b
  python verify_shards.py data/*            # shell-expanded list of source dirs
  python verify_shards.py --deep /mnt/data4/charkha/data/cc      # full per-shard min/max scan
  python verify_shards.py --selftest               # hermetic test (writes a temp shard set)

Exit code 0 = all shards in all dirs OK; 1 = at least one problem (printed per dir).
"""

from __future__ import annotations
import argparse
import array
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

from dataprep import shard_format_for_vocab

SAMPLE = 1_000_000  # ids sampled from each end of a shard for the range check (default mode)


def _rust_shardscan(fp, bytes_per_token, deep=False):
    """Optional Rust fast path for shard min/max scans.

    CHARKHA keeps Python as the orchestration layer, but raw shard scans are a clean Rust boundary:
    byte-heavy, deterministic, no Torch state, and easy to fall back from. Set CHARKHA_SHARDSCAN to
    an executable path, or build tools/charkha-shardscan and the verifier will discover it.
    """
    exe = os.environ.get("CHARKHA_SHARDSCAN")
    if not exe:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cand = os.path.join(
            root, "tools", "charkha-shardscan", "target", "release", "charkha-shardscan"
        )
        if os.path.exists(cand):
            exe = cand
    if not exe or not os.path.exists(exe):
        return None
    dtype = "u16" if bytes_per_token == 2 else "u32"
    cmd = [exe, "--dtype", dtype, "--deep" if deep else "--sample", str(SAMPLE), fp]
    if deep:
        cmd = [exe, "--dtype", dtype, "--deep", fp]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
        m = re.search(r"tokens=(\d+)\s+min=(\d+)\s+max=(\d+)", out)
        if not m:
            raise RuntimeError(f"unparseable shardscan output: {out}")
        return tuple(int(x) for x in m.groups())
    except Exception as e:
        raise RuntimeError(f"Rust shardscan failed for {fp}: {e}") from e


def verify_dir(d, deep=False):
    """Return (ok, total_tokens, [problems]) for one data dir."""
    problems = []
    idx_path = os.path.join(d, "index.json")
    if not os.path.isfile(idx_path):
        return False, 0, [f"no index.json in {d}"]
    with open(idx_path) as f:
        index = json.load(f)
    vocab = index.get("vocab_size", 0)
    shards = index.get("shards", [])
    if not shards:
        return False, 0, ["index.json lists 0 shards"]
    declared_total = index.get("total_tokens", 0)
    _, bytes_per_token = shard_format_for_vocab(vocab)
    np_dtype = np.uint16 if bytes_per_token == 2 else np.uint32
    summed = 0
    for s in shards:
        fp = os.path.join(d, s["file"])
        want_tok = s["tokens"]
        summed += want_tok
        if not os.path.isfile(fp):
            problems.append(f"{s['file']}: MISSING")
            continue
        size = os.path.getsize(fp)
        if size % bytes_per_token != 0:
            problems.append(
                f"{s['file']}: odd byte size {size} (not {bytes_per_token}-byte-token-aligned)"
            )
            continue
        have_tok = size // bytes_per_token
        if have_tok != want_tok:
            problems.append(
                f"{s['file']}: size says {have_tok:,} tok, index says {want_tok:,} "
                f"(truncated/partial?)"
            )
            continue
        # id-range check: prefer the optional Rust scanner when built; fall back to NumPy memmap.
        scanned = _rust_shardscan(fp, bytes_per_token, deep=deep)
        if scanned is not None:
            scan_tok, lo, hi = scanned
            if scan_tok != want_tok:
                problems.append(
                    f"{s['file']}: scanner saw {scan_tok:,} tok, index says {want_tok:,}"
                )
                continue
        else:
            a = np.memmap(fp, dtype=np_dtype, mode="r")
            if deep:
                lo, hi = int(a.min()), int(a.max())
            else:
                head = a[:SAMPLE]
                tail = a[-SAMPLE:]
                lo = int(min(head.min(), tail.min()))
                hi = int(max(head.max(), tail.max()))
            del a
        if vocab and hi >= vocab:
            problems.append(
                f"{s['file']}: max id {hi} >= vocab_size {vocab} (corruption / wrong dtype/endian)"
            )
    if declared_total and summed != declared_total:
        problems.append(f"per-shard sum {summed:,} != index total_tokens {declared_total:,}")
    return (len(problems) == 0), summed, problems


def main(argv):
    p = argparse.ArgumentParser(description="Verify CHARKHA dataprep shard dirs")
    p.add_argument(
        "dirs", nargs="*", help="one or more data dirs (each has index.json + shard_*.bin)"
    )
    p.add_argument(
        "--deep", action="store_true", help="full per-shard min/max scan (slow, reads all bytes)"
    )
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.dirs:
        p.error("give at least one data dir (or --selftest)")

    grand = 0
    all_ok = True
    vocabs = set()
    print(
        f"verify_shards | mode={'DEEP (full scan)' if a.deep else 'fast (sampled id-range)'} "
        f"| {len(a.dirs)} dir(s)\n"
    )
    for d in a.dirs:
        ok, toks, problems = verify_dir(d, deep=a.deep)
        grand += toks
        all_ok &= ok
        try:
            with open(os.path.join(d, "index.json")) as f:
                vocabs.add(json.load(f).get("vocab_size"))
        except Exception:
            pass
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {d:<48s} {toks / 1e9:7.3f}B tok")
        for pr in problems:
            print(f"           - {pr}")
    print(f"\n  {'-' * 60}")
    print(f"  TOTAL: {grand / 1e9:.3f}B tokens across {len(a.dirs)} dir(s)")
    if len(vocabs) > 1:
        print(
            f"  WARNING: vocab_size differs across dirs: {vocabs} "
            f"(ShardLoader requires them to match)"
        )
        all_ok = False
    elif vocabs:
        print(f"  vocab_size: {vocabs.pop()} (consistent)")
    print(f"\n{'ALL SHARDS OK' if all_ok else 'PROBLEMS FOUND - see above'}")
    return 0 if all_ok else 1


def selftest():
    print("verify_shards self-test")
    tmp = tempfile.mkdtemp(prefix="charkha_verify_")
    good = os.path.join(tmp, "good")
    os.makedirs(good)
    # write 2 valid shards: ids in [0, 300)
    shards, total = [], 0
    for s in range(2):
        buf = array.array("H", [(i % 300) for i in range(1000)])
        fn = f"shard_{s:05d}.bin"
        with open(os.path.join(good, fn), "wb") as f:
            buf.tofile(f)
        shards.append({"file": fn, "tokens": len(buf)})
        total += len(buf)
    with open(os.path.join(good, "index.json"), "w") as f:
        json.dump({"vocab_size": 512, "total_tokens": total, "shards": shards}, f)

    checks = {}
    ok, toks, probs = verify_dir(good)
    checks["clean dir passes"] = ok and toks == total and not probs

    # corrupt: truncate shard 0 -> size mismatch
    bad = os.path.join(tmp, "trunc")
    os.makedirs(bad)
    import shutil

    for fn in os.listdir(good):
        shutil.copy(os.path.join(good, fn), os.path.join(bad, fn))
    with open(os.path.join(bad, "shard_00000.bin"), "r+b") as f:
        f.truncate(500)
    ok2, _, probs2 = verify_dir(bad)
    checks["truncated shard is caught"] = (not ok2) and any(
        "truncated" in p or "size says" in p for p in probs2
    )

    # corrupt: id out of range (vocab too small)
    oor = os.path.join(tmp, "oor")
    os.makedirs(oor)
    buf = array.array("H", [1000] * 1000)  # ids 1000 >= vocab 512
    with open(os.path.join(oor, "shard_00000.bin"), "wb") as f:
        buf.tofile(f)
    with open(os.path.join(oor, "index.json"), "w") as f:
        json.dump(
            {
                "vocab_size": 512,
                "total_tokens": 1000,
                "shards": [{"file": "shard_00000.bin", "tokens": 1000}],
            },
            f,
        )
    ok3, _, probs3 = verify_dir(oor)
    checks["out-of-range id is caught"] = (not ok3) and any(">= vocab" in p for p in probs3)

    # total mismatch
    mis = os.path.join(tmp, "mismatch")
    os.makedirs(mis)
    shutil.copy(os.path.join(good, "shard_00000.bin"), os.path.join(mis, "shard_00000.bin"))
    with open(os.path.join(mis, "index.json"), "w") as f:
        json.dump(
            {
                "vocab_size": 512,
                "total_tokens": 999999,
                "shards": [{"file": "shard_00000.bin", "tokens": 1000}],
            },
            f,
        )
    ok4, _, probs4 = verify_dir(mis)
    checks["total-token mismatch is caught"] = (not ok4) and any(
        "total_tokens" in p for p in probs4
    )

    # vocab > 65535 -> shards are 4 bytes/token (uint32), not 2. A clean uint32 dir must verify
    # OK at the wider width, and a shard written at the WRONG width (uint16 truncation bug) must
    # be caught as a byte-size mismatch -- this is the regression guard for that exact class of bug.
    big = os.path.join(tmp, "big32")
    os.makedirs(big)
    bigbuf = array.array("I", [(i % 90000) for i in range(1000)])  # ids up to 89999 > 65535
    if array.array("I").itemsize != 4:
        raise RuntimeError("platform array('I') is not 4 bytes; uint32 shard selftest invalid")
    with open(os.path.join(big, "shard_00000.bin"), "wb") as f:
        bigbuf.tofile(f)
    with open(os.path.join(big, "index.json"), "w") as f:
        json.dump(
            {
                "vocab_size": 100000,
                "total_tokens": len(bigbuf),
                "shards": [{"file": "shard_00000.bin", "tokens": len(bigbuf)}],
            },
            f,
        )
    ok5, toks5, probs5 = verify_dir(big)
    checks["uint32 (vocab>65535) clean dir passes at 4 bytes/token"] = (
        ok5 and toks5 == len(bigbuf) and not probs5
    )

    wrong = os.path.join(tmp, "wrong_width")
    os.makedirs(wrong)
    array.array("H", [1] * 1000).tofile(
        open(os.path.join(wrong, "shard_00000.bin"), "wb")
    )  # uint16 bytes
    with open(os.path.join(wrong, "index.json"), "w") as f:
        json.dump(
            {
                "vocab_size": 100000,
                "total_tokens": 1000,  # but index claims uint32 vocab
                "shards": [{"file": "shard_00000.bin", "tokens": 1000}],
            },
            f,
        )
    ok6, _, probs6 = verify_dir(wrong)
    checks["wrong-width shard (uint16 bytes, uint32 vocab) is caught"] = (not ok6) and bool(probs6)

    allok = True
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        allok &= passed
    print("\nSELFTEST", "PASS" if allok else "FAIL")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

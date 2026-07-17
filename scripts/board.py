#!/usr/bin/env python3
"""CHARKHA kanban board — the single task tracker for humans and ALL agents.

Design: one JSON file per card under board/cards/ (git-friendly: parallel agents
touch different files, merges never conflict on a shared document). No daemon, no
database, no lock. The CLI is the agent interface; `html` renders a static board
for humans at board/board.html.

Columns: backlog -> todo -> doing -> review -> done  (+ dropped)

Usage:
    python scripts/board.py new "Title" [--col todo] [--pri P1] [--tags a,b] [--body "..."]
    python scripts/board.py move CHK-12 doing [--note "why"]
    python scripts/board.py note CHK-12 "progress note"
    python scripts/board.py list [--col doing] [--all]
    python scripts/board.py show CHK-12
    python scripts/board.py html            # render board/board.html
    python scripts/board.py doctor          # validate every card
    python scripts/board.py --selftest

Agent contract: before starting work, move your card to
`doing` (create it if missing); when done, move to `review` (or `done` if verified
by tests); leave a `note` for anything a follow-up session needs.

"""

import argparse
import datetime
import html as _html
import json
import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOARD = os.path.join(ROOT, "board")
CARDS = os.path.join(BOARD, "cards")
COLUMNS = ["backlog", "todo", "doing", "review", "done", "dropped"]
PRIORITIES = ["P0", "P1", "P2", "P3"]
_ID_RE = re.compile(r"^CHK-(\d+)$")


def _now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _card_path(cid):
    return os.path.join(CARDS, f"{cid}.json")


def _atomic_write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _load(cid):
    p = _card_path(cid)
    if not os.path.exists(p):
        raise SystemExit(f"[board] no such card: {cid}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _all_cards():
    if not os.path.isdir(CARDS):
        return []
    out = []
    for fn in sorted(os.listdir(CARDS)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(CARDS, fn), encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def _next_id():
    mx = 0
    if os.path.isdir(CARDS):
        for fn in os.listdir(CARDS):
            m = _ID_RE.match(fn[:-5]) if fn.endswith(".json") else None
            if m:
                mx = max(mx, int(m.group(1)))
    return f"CHK-{mx + 1}"


def _validate(c):
    errs = []
    for k in ("id", "title", "col", "pri", "created", "updated"):
        if k not in c:
            errs.append(f"missing key {k!r}")
    if c.get("col") not in COLUMNS:
        errs.append(f"bad column {c.get('col')!r}")
    if c.get("pri") not in PRIORITIES:
        errs.append(f"bad priority {c.get('pri')!r}")
    if "id" in c and not _ID_RE.match(c["id"]):
        errs.append(f"bad id {c['id']!r}")
    return errs


def cmd_new(a):
    cid = _next_id()
    card = {
        "id": cid,
        "title": a.title,
        "col": a.col,
        "pri": a.pri,
        "tags": [t for t in (a.tags or "").split(",") if t],
        "body": a.body or "",
        "notes": [],
        "created": _now(),
        "updated": _now(),
    }
    errs = _validate(card)
    if errs:
        raise SystemExit(f"[board] refusing invalid card: {errs}")
    _atomic_write(_card_path(cid), card)
    print(f"{cid}  [{a.col}]  {a.title}")
    return cid


def cmd_move(a):
    if a.dest not in COLUMNS:
        raise SystemExit(f"[board] bad column {a.dest!r}; choose from {COLUMNS}")
    c = _load(a.id)
    c.setdefault("notes", []).append(
        {"ts": _now(), "text": f"{c['col']} -> {a.dest}" + (f": {a.note}" if a.note else "")}
    )
    c["col"], c["updated"] = a.dest, _now()
    _atomic_write(_card_path(a.id), c)
    print(f"{a.id} -> {a.dest}")


def cmd_note(a):
    c = _load(a.id)
    c.setdefault("notes", []).append({"ts": _now(), "text": a.text})
    c["updated"] = _now()
    _atomic_write(_card_path(a.id), c)
    print(f"{a.id}: noted")


def cmd_list(a):
    cards = _all_cards()
    cols = (
        [a.col]
        if a.col
        else (COLUMNS if a.all else [c for c in COLUMNS if c not in ("done", "dropped")])
    )
    for col in cols:
        rows = sorted((c for c in cards if c["col"] == col), key=lambda c: (c["pri"], c["id"]))
        if not rows:
            continue
        print(f"-- {col} ({len(rows)}) " + "-" * max(1, 40 - len(col)))
        for c in rows:
            tags = (" [" + ",".join(c.get("tags", [])) + "]") if c.get("tags") else ""
            print(f"  {c['id']:>7}  {c['pri']}  {c['title']}{tags}")


def cmd_show(a):
    c = _load(a.id)
    print(json.dumps(c, indent=2, ensure_ascii=False))


def cmd_doctor(a):
    bad = 0
    for c in _all_cards():
        errs = _validate(c)
        if errs:
            bad += 1
            print(f"[doctor] {c.get('id', '?')}: {'; '.join(errs)}")
    print(f"[doctor] {'OK — all cards valid' if not bad else f'{bad} invalid card(s)'}")
    return 0 if not bad else 1


_HTML_HEAD = """<!doctype html><meta charset="utf-8"><title>CHARKHA board</title><style>
body{font:14px/1.4 system-ui,sans-serif;margin:16px;background:#111;color:#ddd}
h1{font-size:18px} .cols{display:flex;gap:12px;align-items:flex-start}
.col{flex:1;background:#1a1a1a;border-radius:8px;padding:8px;min-width:150px}
.col h2{font-size:13px;text-transform:uppercase;letter-spacing:1px;color:#888;margin:4px}
.card{background:#242424;border-radius:6px;padding:8px;margin:6px 0;border-left:3px solid #555}
.P0{border-color:#e5534b}.P1{border-color:#d4a72c}.P2{border-color:#57ab5a}.P3{border-color:#555}
.id{color:#888;font-size:11px}.tags{color:#6cb6ff;font-size:11px}
.note{color:#999;font-size:11px;margin-top:4px}</style>
"""


def cmd_html(a):
    cards = _all_cards()
    parts = [
        _HTML_HEAD,
        f'<h1>CHARKHA board <span class="id">rendered {_html.escape(_now())}'
        f' — regenerate with <code>python scripts/board.py html</code></span></h1><div class="cols">',
    ]
    for col in COLUMNS:
        rows = sorted((c for c in cards if c["col"] == col), key=lambda c: (c["pri"], c["id"]))
        parts.append(f'<div class="col"><h2>{col} ({len(rows)})</h2>')
        for c in rows:
            last = c["notes"][-1]["text"] if c.get("notes") else ""
            parts.append(
                f'<div class="card {c["pri"]}"><span class="id">{c["id"]} · {c["pri"]}</span> '
                f"{_html.escape(c['title'])}"
                + (
                    f'<div class="tags">{_html.escape(",".join(c["tags"]))}</div>'
                    if c.get("tags")
                    else ""
                )
                + (f'<div class="note">{_html.escape(last[:160])}</div>' if last else "")
                + "</div>"
            )
        parts.append("</div>")
    parts.append("</div>")
    out = os.path.join(BOARD, "board.html")
    os.makedirs(BOARD, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    print(f"[board] wrote {os.path.relpath(out, ROOT)} ({len(cards)} cards)")


def _selftest():
    global CARDS, BOARD
    import shutil

    tmp = tempfile.mkdtemp(prefix="board-selftest-")
    old_b, old_c = BOARD, CARDS
    BOARD, CARDS = tmp, os.path.join(tmp, "cards")
    try:
        n = argparse.Namespace(title="test card", col="todo", pri="P1", tags="a,b", body="hello")
        cid = cmd_new(n)
        assert cid == "CHK-1" and os.path.exists(_card_path(cid))
        cmd_move(argparse.Namespace(id=cid, dest="doing", note="starting"))
        c = _load(cid)
        assert c["col"] == "doing" and "starting" in c["notes"][-1]["text"]
        cmd_note(argparse.Namespace(id=cid, text="halfway"))
        assert _load(cid)["notes"][-1]["text"] == "halfway"
        assert _next_id() == "CHK-2"
        # invalid moves refused
        try:
            cmd_move(argparse.Namespace(id=cid, dest="nowhere", note=None))
            raise AssertionError("bad column accepted")
        except SystemExit:
            pass
        # doctor catches corruption
        bad = dict(_load(cid))
        bad["col"] = "limbo"
        _atomic_write(_card_path(cid), bad)
        assert cmd_doctor(None) == 1
        bad["col"] = "doing"
        _atomic_write(_card_path(cid), bad)
        assert cmd_doctor(None) == 0
        cmd_html(argparse.Namespace())
        assert os.path.exists(os.path.join(BOARD, "board.html"))
        print("[selftest] board.py: all checks passed")
        return 0
    finally:
        BOARD, CARDS = old_b, old_c
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("new")
    p.add_argument("title")
    p.add_argument("--col", default="todo", choices=COLUMNS)
    p.add_argument("--pri", default="P2", choices=PRIORITIES)
    p.add_argument("--tags")
    p.add_argument("--body")
    p = sub.add_parser("move")
    p.add_argument("id")
    p.add_argument("dest")
    p.add_argument("--note")
    p = sub.add_parser("note")
    p.add_argument("id")
    p.add_argument("text")
    p = sub.add_parser("list")
    p.add_argument("--col")
    p.add_argument("--all", action="store_true")
    p = sub.add_parser("show")
    p.add_argument("id")
    sub.add_parser("html")
    sub.add_parser("doctor")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not a.cmd:
        ap.print_help()
        return 0
    rc = {
        "new": cmd_new,
        "move": cmd_move,
        "note": cmd_note,
        "list": cmd_list,
        "show": cmd_show,
        "html": cmd_html,
        "doctor": cmd_doctor,
    }[a.cmd](a)
    return rc if isinstance(rc, int) else 0  # cmd_new returns the card id, not an exit code


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Backwards reverse-engineering ladder dataset: source -> binary -> asm -> pseudo.

Scraped binary/pseudo pairs can lack trustworthy ground truth. By compiling known
source, every rung of the ladder is exactly aligned by construction:

    C source  --gcc -O{0..3}-->  ELF/.o  --objdump-->  x86-64 asm
                                          --readelf-->  raw .text bytes (hex)
                                          --ghidra  -->  decompiled pseudo-C

Each function becomes a record carrying all rungs, so the model can be trained on
EVERY direction (lift asm->source, decompile bytes->pseudo, recompile source->asm,
repair pseudo->source) from one build. The original source is free supervision.

Ghidra is optional: set --ghidra $GHIDRA_HOME (an unpacked ghidra_11.x dir with
support/analyzeHeadless). Without it the pipeline still emits fully-aligned
(source, bytes, asm) triples — already high value — and marks pseudo as null.

Source rungs come from either --src-dir (real .c files, one function each is
ideal) or --gen N (built-in varied C generators: sorts, math, strings, structs,
recursion, pointers) for a self-contained proof and baseline volume.

Output: <out>/re_ladder.jsonl  (one JSON record per compiled function) and a
rendered training-text field the dataprep local loader can tokenize directly.

Usage:
    python scripts/build_re_dataset.py --selftest
    python scripts/build_re_dataset.py --gen 200 --opt 0,2 --out data/raw/re_ladder
    python scripts/build_re_dataset.py --src-dir some/c/files --ghidra ~/ghidra_11.3 \
        --out data/raw/re_ladder

"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

# ── built-in C generators: small, self-contained, varied real patterns ──
_TEMPLATES = [
    (
        "sum_array",
        "int {name}(const int *a, int n){{int s=0;for(int i=0;i<n;i++)s+=a[i];return s;}}",
    ),
    (
        "max_array",
        "int {name}(const int *a, int n){{int m=a[0];for(int i=1;i<n;i++)if(a[i]>m)m=a[i];return m;}}",
    ),
    ("factorial", "long {name}(int n){{long r=1;while(n>1)r*=n--;return r;}}"),
    (
        "fib",
        "long {name}(int n){{long a=0,b=1;for(int i=0;i<n;i++){{long t=a+b;a=b;b=t;}}return a;}}",
    ),
    ("gcd", "int {name}(int a,int b){{while(b){{int t=b;b=a%b;a=t;}}return a;}}"),
    ("strlen_c", "int {name}(const char *s){{int n=0;while(*s++)n++;return n;}}"),
    (
        "reverse",
        "void {name}(int *a,int n){{for(int i=0;i<n/2;i++){{int t=a[i];a[i]=a[n-1-i];a[n-1-i]=t;}}}}",
    ),
    (
        "bsearch_c",
        "int {name}(const int *a,int n,int x){{int lo=0,hi=n-1;while(lo<=hi){{int m=(lo+hi)/2;if(a[m]==x)return m;if(a[m]<x)lo=m+1;else hi=m-1;}}return -1;}}",
    ),
    (
        "bubble",
        "void {name}(int *a,int n){{for(int i=0;i<n;i++)for(int j=0;j+1<n-i;j++)if(a[j]>a[j+1]){{int t=a[j];a[j]=a[j+1];a[j+1]=t;}}}}",
    ),
    ("popcount", "int {name}(unsigned x){{int c=0;while(x){{c+=x&1;x>>=1;}}return c;}}"),
    (
        "dot",
        "double {name}(const double *a,const double *b,int n){{double s=0;for(int i=0;i<n;i++)s+=a[i]*b[i];return s;}}",
    ),
    (
        "is_prime",
        "int {name}(int n){{if(n<2)return 0;for(int i=2;(long)i*i<=n;i++)if(n%i==0)return 0;return 1;}}",
    ),
    ("str_upper", "void {name}(char *s){{for(;*s;s++)if(*s>=97&&*s<=122)*s-=32;}}"),
    ("clamp", "int {name}(int x,int lo,int hi){{return x<lo?lo:(x>hi?hi:x);}}"),
    (
        "matvec",
        "void {name}(const double *M,const double *v,double *o,int n){{for(int i=0;i<n;i++){{double s=0;for(int j=0;j<n;j++)s+=M[i*n+j]*v[j];o[i]=s;}}}}",
    ),
]


def gen_sources(n):
    """Yield (fn_name, c_source) — cycle templates with unique names for volume."""
    for i in range(n):
        base, tmpl = _TEMPLATES[i % len(_TEMPLATES)]
        name = f"{base}_{i}"
        yield name, tmpl.format(name=name)


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def compile_c(src, workdir, opt, name):
    """Compile one C function to a .o; return (obj_path, err|None)."""
    c_path = os.path.join(workdir, f"{name}.c")
    o_path = os.path.join(workdir, f"{name}.o")
    with open(c_path, "w") as f:
        f.write(src + "\n")
    r = _run(["gcc", f"-O{opt}", "-c", "-fno-asynchronous-unwind-tables", "-o", o_path, c_path])
    if r.returncode != 0:
        return None, r.stderr.strip()[:200]
    return o_path, None


def disasm(o_path):
    """objdump -> Intel-syntax asm text for the .text section (no raw bytes col)."""
    r = _run(["objdump", "-d", "-M", "intel", "--no-show-raw-insn", o_path])
    lines = []
    for ln in r.stdout.splitlines():
        s = ln.strip()
        if not s or s.startswith("/") or s.startswith("Disassembly") or s.startswith("..."):
            continue
        if s.endswith(":") and ("<" in s or "section" in s.lower()):
            lines.append(s)
        elif "\t" in ln:  # an instruction line
            lines.append(s)
    return "\n".join(lines)


def text_bytes_hex(o_path):
    """Raw .text section bytes as a hex string (readelf -x .text)."""
    r = _run(["readelf", "-x", ".text", o_path])
    hexes = []
    for ln in r.stdout.splitlines():
        parts = ln.split()
        if len(parts) >= 2 and parts[0].startswith("0x"):
            hexes += parts[1:5]  # 4 hex words per row
    return "".join(h for h in hexes if all(c in "0123456789abcdefABCDEF" for c in h))


class Ghidra:
    """Optional decompiler stage via headless analyzeHeadless + a Java postscript.
    Imports a WHOLE DIRECTORY of .o files in ONE session (JVM/analysis startup is
    amortized across every object) and returns {o_basename: pseudo_c}."""

    def __init__(self, ghidra_home, script_path):
        self.headless = os.path.join(ghidra_home, "support", "analyzeHeadless")
        self.script_path = script_path
        self.ok = os.path.exists(self.headless)

    def decompile_dir(self, obj_dir, workdir):
        if not self.ok:
            return {}
        proj = os.path.join(workdir, "ghp")
        os.makedirs(proj, exist_ok=True)
        out_jsonl = os.path.join(workdir, "pseudo.jsonl")
        if os.path.exists(out_jsonl):
            os.remove(out_jsonl)
        _run(
            [
                self.headless,
                proj,
                "p",
                "-import",
                obj_dir,
                "-recursive",
                "-scriptPath",
                os.path.dirname(self.script_path),
                "-postScript",
                os.path.basename(self.script_path),
                out_jsonl,
                "-deleteProject",
            ],
            timeout=3600,
        )
        by_prog = {}
        if os.path.exists(out_jsonl):
            for ln in open(out_jsonl):
                try:
                    d = json.loads(ln)
                    by_prog.setdefault(d["program"], []).append(d["pseudo"])
                except Exception:
                    continue
        return {k: "\n\n".join(v) for k, v in by_prog.items()}


def render(rec):
    """Training-text rendering of the ladder so the model sees every mapping."""
    parts = [
        f"// reverse-engineering ladder: opt=-O{rec['opt']} arch=x86-64",
        "// === SOURCE ===",
        rec["source"],
        "// === BYTES (.text, hex) ===",
        rec["bytes_hex"],
        "// === DISASSEMBLY (objdump, intel) ===",
        rec["asm"],
    ]
    if rec.get("pseudo"):
        parts += ["// === DECOMPILED (ghidra pseudo-C) ===", rec["pseudo"]]
    return "\n".join(parts)


def build(sources, opts, out_dir, ghidra=None, limit=0):
    """Two-phase: (1) compile every (source, opt) to a .o and grab asm+bytes;
    (2) if ghidra, decompile the whole .o directory in ONE session and match pseudo
    back by .o basename. Each .o is named {name}_O{opt}.o (unique) so the ghidra
    program name identifies its rung exactly."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "re_ladder.jsonl")
    n_ok = n_fail = 0
    with tempfile.TemporaryDirectory() as wd:
        obj_dir = os.path.join(wd, "objs")
        os.makedirs(obj_dir, exist_ok=True)
        recs = []
        for name, src in sources:
            for opt in opts:
                key = f"{name}_O{opt}"
                o_path, err = compile_c(src, obj_dir, opt, key)
                if err:
                    n_fail += 1
                    continue
                asm, hx = disasm(o_path), text_bytes_hex(o_path)
                if not asm or not hx:
                    n_fail += 1
                    continue
                recs.append(
                    {
                        "name": name,
                        "opt": opt,
                        "arch": "x86-64",
                        "source": src,
                        "bytes_hex": hx,
                        "asm": asm,
                        "pseudo": None,
                        "_obj": f"{key}.o",
                    }
                )
                if limit and len(recs) >= limit:
                    break
            if limit and len(recs) >= limit:
                break
        pseudo_map = ghidra.decompile_dir(obj_dir, wd) if ghidra else {}
        with open(out_path, "w") as out:
            for rec in recs:
                rec["pseudo"] = pseudo_map.get(rec.pop("_obj"))
                rec["text"] = render(rec)
                out.write(json.dumps(rec) + "\n")
                n_ok += 1
    return out_path, n_ok, n_fail


def iter_src_dir(d):
    for root, _dirs, files in os.walk(d):
        for fn in files:
            if fn.endswith(".c"):
                p = os.path.join(root, fn)
                try:
                    yield os.path.splitext(fn)[0], open(p, errors="ignore").read()
                except Exception:
                    continue


def _selftest():
    checks = 0
    if _run(["gcc", "--version"]).returncode != 0:
        print("[selftest] SKIP: no gcc")
        return 0
    with tempfile.TemporaryDirectory() as wd:
        name, src = next(gen_sources(1))
        o, err = compile_c(src, wd, 2, name)
        assert o and err is None and os.path.exists(o)
        checks += 1
        asm = disasm(o)
        assert "ret" in asm and len(asm) > 20  # real instructions present
        checks += 1
        hx = text_bytes_hex(o)
        assert len(hx) >= 8 and all(c in "0123456789abcdefABCDEF" for c in hx)
        checks += 1
    out_dir = tempfile.mkdtemp(prefix="re_")
    path, ok, fail = build(gen_sources(6), [0, 2], out_dir, limit=0)
    recs = [json.loads(l) for l in open(path)]
    assert ok == len(recs) >= 10  # 6 fns x 2 opts, minus any fails
    r0 = recs[0]
    assert set(("source", "asm", "bytes_hex", "text")) <= set(r0)
    assert "=== SOURCE ===" in r0["text"] and "=== DISASSEMBLY" in r0["text"]
    assert r0["pseudo"] is None  # no ghidra in selftest
    checks += 3
    print(
        f"[selftest] build_re_dataset.py: all checks passed ({checks} groups; "
        f"{ok} records built, {fail} compile/emit fails)"
    )
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gen", type=int, default=0, help="generate N built-in C functions")
    ap.add_argument("--src-dir", type=str, help="directory of real .c files instead")
    ap.add_argument("--opt", type=str, default="0,1,2,3", help="comma opt levels")
    ap.add_argument("--out", type=str, default="data/raw/re_ladder")
    ap.add_argument(
        "--ghidra",
        type=str,
        default=os.environ.get("GHIDRA_HOME"),
        help="GHIDRA_HOME (unpacked ghidra dir); enables the pseudo-C rung",
    )
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    opts = [int(x) for x in a.opt.split(",") if x.strip()]
    if a.src_dir:
        srcs = iter_src_dir(a.src_dir)
    elif a.gen:
        srcs = gen_sources(a.gen)
    else:
        ap.print_help()
        sys.exit(1)
    gh = None
    if a.ghidra:
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DecompileToJson.java")
        gh = Ghidra(a.ghidra, script)
        if not gh.ok:
            print(
                f"[re] WARNING: analyzeHeadless not found under {a.ghidra}; "
                f"emitting without the pseudo-C rung"
            )
            gh = None
    path, ok, fail = build(srcs, opts, a.out, ghidra=gh, limit=a.limit)
    print(
        f"[re] wrote {ok} ladder records ({fail} fails) -> {path}"
        + ("  [with ghidra pseudo]" if gh else "  [asm+bytes only; set --ghidra for pseudo]")
    )

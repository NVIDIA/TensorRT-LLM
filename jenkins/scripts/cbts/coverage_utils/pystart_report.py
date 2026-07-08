# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Merge per-process CBTS PY_START data files into the CBTS touch artifacts.

Reads all ``.cbtscov.*.json`` files (each ``{context: [[file, qualname], ...]}``) and unions
them per test context, then emits any of:
  --out-sqlite : indexed ``touch(test, file, qualname)`` DB -- the machine-readable artifact the
                 selector queries ("which tests touch file/function X"); single file, scales.
  --out-dir    : a browsable HTML report split by source file (index + one page per file), so no
                 single page is large and the whole tree compresses well (like coverage.py's report).
  --out-json   : the full per-test -> [file::qualname] map (kept for small/local runs).
stdlib-only; also prints a one-line touch-count summary (+ coverage rate when --source-root is given).
"""

import argparse
import ast
import glob
import gzip
import json
import os
import re
import sqlite3
from collections import defaultdict
from html import escape


def canon(path):
    """Collapse an absolute install/src path to the ``tensorrt_llm/...`` product-relative form."""
    m = re.search(r"(tensorrt_llm/.*)$", path)
    return m.group(1) if m else path


def enumerate_defs(source_root):
    """Enumerate the static coverage-rate denominator from the product source tree.

    Every product .py file and every top-level function / method defined in it (nested /
    comprehension frames are excluded to match what the PY_START tracker records).
    Returns (set(files), set((file, qualname))).
    """
    files, funcs = set(), set()
    for dirpath, _dirs, names in os.walk(source_root):
        for nm in names:
            if not nm.endswith(".py"):
                continue
            full = os.path.join(dirpath, nm)
            rel = canon(os.path.abspath(full))
            files.add(rel)
            try:
                tree = ast.parse(open(full, encoding="utf-8").read())
            except (OSError, SyntaxError):
                continue

            def visit(node, prefix):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        funcs.add((rel, prefix + child.name))
                    elif isinstance(child, ast.ClassDef):
                        for m in child.body:
                            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                funcs.add((rel, f"{prefix}{child.name}.{m.name}"))

            visit(tree, "")
    return files, funcs


def load_merged(pattern):
    """Union all per-process data files -> {test_context: set((file, qualname))}.

    Per-process files are SQLite (`.cbtscov.*.sqlite`, a `touch(test, file, qualname)` table);
    legacy `.json` / `.json.gz` are still accepted so older data merges too.
    """
    per_test = defaultdict(set)
    files = sorted(glob.glob(pattern))
    for fp in files:
        try:
            if fp.endswith(".sqlite"):
                con = sqlite3.connect(fp)
                try:
                    for ctx, file, qual in con.execute("SELECT test, file, qualname FROM touch"):
                        per_test[ctx].add((canon(file), qual))
                finally:
                    con.close()
            else:
                opener = gzip.open if fp.endswith(".gz") else open
                with opener(fp, "rt") as f:
                    data = json.load(f)
                for ctx, pairs in data.items():
                    for file, qual in pairs:
                        per_test[ctx].add((canon(file), qual))
        except (OSError, ValueError, sqlite3.Error):
            continue
    return per_test, len(files)


def write_sqlite(path, per_test, meta):
    """Indexed touch DB for the selector: touch(test, file, qualname) + a meta table."""
    if os.path.exists(path):
        os.remove(path)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE touch (test TEXT, file TEXT, qualname TEXT)")
    con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    rows = ((t, f, q) for t, pairs in per_test.items() if t for (f, q) in pairs)
    con.executemany("INSERT INTO touch VALUES (?, ?, ?)", rows)
    con.execute("CREATE INDEX ix_file ON touch(file)")
    con.execute("CREATE INDEX ix_func ON touch(file, qualname)")
    con.execute("CREATE INDEX ix_test ON touch(test)")
    con.executemany("INSERT INTO meta VALUES (?, ?)", list(meta.items()))
    con.commit()
    con.close()


_CSS = (
    "<style>body{font-family:-apple-system,Segoe UI,sans-serif;margin:24px;font-size:13px}"
    "h1{font-size:20px}a{color:#1a5fb4;text-decoration:none}a:hover{text-decoration:underline}"
    "table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:3px 8px;text-align:right}"
    "td.l,th.l{text-align:left}th{background:#eee}.mono{font-family:ui-monospace,Menlo,monospace;font-size:12px}"
    ".t{color:#557;font-size:11px}summary{cursor:pointer;font-weight:600}"
    "details{margin:5px 0;border:1px solid #ddd;border-radius:6px;padding:4px 10px}</style>"
)


def _page_name(file):
    return re.sub(r"[^A-Za-z0-9]", "_", file) + ".html"


def write_html_tree(out_dir, per_test, rate_line, n_tests, n_data_files):
    """Split HTML report: index.html (files grouped by directory) + files/<file>.html (per-file detail)."""
    files_dir = os.path.join(out_dir, "files")
    os.makedirs(files_dir, exist_ok=True)

    # file -> {qualname -> sorted[tests]}
    file_funcs = defaultdict(lambda: defaultdict(list))
    for ctx, pairs in per_test.items():
        if not ctx:
            continue
        for file, qual in pairs:
            file_funcs[file][qual].append(ctx)

    # per-file pages
    for file, funcs in file_funcs.items():
        h = [f"<!doctype html><meta charset='utf-8'><title>{escape(file)}</title>{_CSS}"]
        h.append(
            f"<p><a href='../index.html'>&larr; index</a></p><h1 class='mono'>{escape(file)}</h1>"
        )
        h.append(f"<p class='t'>{len(funcs)} functions touched</p>")
        for qual in sorted(funcs):
            tests = sorted(set(funcs[qual]))
            h.append(
                f"<details><summary class='mono'>{escape(qual)} "
                f"<span class='t'>&larr; {len(tests)} tests</span></summary>"
            )
            for t in tests:
                h.append(f"<div class='mono t'>{escape(t)}</div>")
            h.append("</details>")
        with open(os.path.join(files_dir, _page_name(file)), "w") as fh:
            fh.write("\n".join(h))

    # index: group files by directory
    by_dir = defaultdict(list)
    for file in file_funcs:
        by_dir[os.path.dirname(file)].append(file)
    h = [f"<!doctype html><meta charset='utf-8'><title>CBTS coverage</title>{_CSS}"]
    h.append("<h1>CBTS function-level coverage (PY_START)</h1>")
    h.append(
        f"<p>{n_tests} tests · {len(file_funcs)} product files touched · {n_data_files} data files merged.</p>"
    )
    if rate_line:
        h.append(f"<p><b>{escape(rate_line)}</b></p>")
    for d in sorted(by_dir):
        flist = sorted(by_dir[d])
        h.append(
            f"<details><summary class='mono'>{escape(d or '.')} <span class='t'>({len(flist)} files)</span></summary>"
        )
        h.append("<table><tr><th class='l'>file</th><th>funcs</th><th>tests</th></tr>")
        for file in flist:
            funcs = file_funcs[file]
            ntests = len({t for ts in funcs.values() for t in ts})
            h.append(
                f"<tr><td class='l mono'><a href='files/{_page_name(file)}'>{escape(os.path.basename(file))}</a></td>"
                f"<td>{len(funcs)}</td><td>{ntests}</td></tr>"
            )
        h.append("</table></details>")
    with open(os.path.join(out_dir, "index.html"), "w") as fh:
        fh.write("\n".join(h))


def main():
    ap = argparse.ArgumentParser(description="Merge CBTS PY_START data into touch artifacts.")
    ap.add_argument("--glob", default=".cbtscov.*.sqlite", help="glob for per-process data files")
    ap.add_argument(
        "--out-sqlite", default=None, help="indexed touch(test,file,qualname) DB for the selector"
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="directory for the split HTML report (index + per-file pages)",
    )
    ap.add_argument(
        "--out-json", default=None, help="optional full per-test -> [file::qualname] map"
    )
    ap.add_argument(
        "--source-root",
        default=None,
        help="product tree (…/tensorrt_llm); when given, compute file/function coverage rate",
    )
    a = ap.parse_args()

    per_test, n_data_files = load_merged(a.glob)

    # reverse counts (skip the empty context: import-time frames collected before any test)
    file_to_tests = defaultdict(set)
    func_to_tests = defaultdict(set)
    for ctx, pairs in per_test.items():
        if not ctx:
            continue
        for file, qual in pairs:
            file_to_tests[file].add(ctx)
            func_to_tests[(file, qual)].add(ctx)

    n_tests = len([c for c in per_test if c])
    print(
        f"CBTS PY_START: merged {n_data_files} data files -> "
        f"{n_tests} tests, {len(file_to_tests)} product files, {len(func_to_tests)} functions touched"
    )

    meta = {
        "tests": str(n_tests),
        "files": str(len(file_to_tests)),
        "functions": str(len(func_to_tests)),
    }
    rate_line = ""
    if a.source_root:
        _all_files, all_funcs = enumerate_defs(a.source_root)
        coverable_files = {f for f, _q in all_funcs}
        touched_files = {f for f in file_to_tests} & coverable_files
        touched_funcs = {k for k in func_to_tests} & all_funcs
        fr = 100.0 * len(touched_files) / len(coverable_files) if coverable_files else 0.0
        qr = 100.0 * len(touched_funcs) / len(all_funcs) if all_funcs else 0.0
        rate_line = (
            f"coverage rate: files {len(touched_files)}/{len(coverable_files)} "
            f"({fr:.1f}%), functions {len(touched_funcs)}/{len(all_funcs)} ({qr:.1f}%)"
        )
        print(rate_line)
        meta.update(
            {
                "file_rate_pct": f"{fr:.2f}",
                "func_rate_pct": f"{qr:.2f}",
                "total_files": str(len(coverable_files)),
                "total_functions": str(len(all_funcs)),
            }
        )

    if a.out_sqlite:
        write_sqlite(a.out_sqlite, per_test, meta)
        print(f"wrote {a.out_sqlite}")
    if a.out_json:
        touchmap = {
            ctx: sorted(f"{f}::{q}" for f, q in pairs)
            for ctx, pairs in sorted(per_test.items())
            if ctx
        }
        with open(a.out_json, "w") as f:
            json.dump(touchmap, f, indent=0)
        print(f"wrote {a.out_json}")
    if a.out_dir:
        write_html_tree(a.out_dir, per_test, rate_line, n_tests, n_data_files)
        print(f"wrote {a.out_dir}/index.html")


if __name__ == "__main__":
    main()

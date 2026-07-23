# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Merge per-process CBTS PY_START data files (streamed, deduped) into the touch DB / HTML report / JSON map."""

import argparse
import ast
import glob
import gzip
import hashlib
import json
import os
import re
import sqlite3
import tempfile
from collections import defaultdict
from html import escape


def canon(path):
    """Collapse an absolute install/src path to the ``tensorrt_llm/...`` product-relative form."""
    m = re.search(r"(tensorrt_llm/.*)$", path)
    return m.group(1) if m else path


def _iter_rows(fp):
    """Yield (test, file, qualname) rows from one per-process data file (SQLite or legacy JSON)."""
    if fp.endswith(".sqlite"):
        con = sqlite3.connect(f"file:{fp}?mode=ro", uri=True)
        try:
            yield from con.execute("SELECT test, file, qualname FROM touch")
        finally:
            con.close()
    else:
        opener = gzip.open if fp.endswith(".gz") else open
        with opener(fp, "rt") as f:
            data = json.load(f)
        for ctx, pairs in data.items():
            for file, qual in pairs:
                yield ctx, file, qual


def _safe_rows(fp):
    """Yield canon'd (test, file, qualname) rows from one input file; stop early on a corrupt/unreadable input."""
    try:
        for test, file, qual in _iter_rows(fp):
            yield test, canon(file), qual
    except (OSError, ValueError, sqlite3.Error):
        return


def _read_stage(fp):
    """Return the stage recorded inside a per-process sqlite, or '' for legacy files."""
    if not fp.endswith(".sqlite"):
        return ""
    try:
        con = sqlite3.connect(f"file:{fp}?mode=ro", uri=True)
        try:
            row = con.execute("SELECT stage FROM proc_meta LIMIT 1").fetchone()
            return row[0] if row and row[0] else ""
        finally:
            con.close()
    except sqlite3.Error:
        return ""


def _read_test_meta(fp):
    """Yield (test, outcome, expected_workers) from a per-process sqlite; nothing for legacy files."""
    if not fp.endswith(".sqlite"):
        return
    try:
        con = sqlite3.connect(f"file:{fp}?mode=ro", uri=True)
        try:
            yield from con.execute("SELECT test, outcome, expected_workers FROM test_meta")
        finally:
            con.close()
    except sqlite3.Error:
        return


def merge_to_sqlite(pattern, out_path):
    """Stream every per-process file into a deduped touch DB. Returns (connection, n_data_files)."""
    files = sorted(glob.glob(pattern))
    if os.path.exists(out_path):
        os.remove(out_path)
    con = sqlite3.connect(out_path)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute(
        "CREATE TABLE touch (test TEXT, file TEXT, qualname TEXT, stage TEXT, "
        "UNIQUE(test, file, qualname, stage))"
    )
    con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")

    # Per-(test, stage) completeness signal: saved_procs counts the per-process files that
    # contributed rows; outcome / expected_workers come from the coordinator's test_meta.
    test_procs = defaultdict(set)
    test_outcome = {}
    test_expected = defaultdict(int)

    for fp in files:
        stage = _read_stage(fp)
        batch = []
        seen_tests = set()
        # Input errors are recovered in _safe_rows; destination-write errors abort the merge.
        for test, file, qual in _safe_rows(fp):
            seen_tests.add(test)
            batch.append((test, file, qual, stage))
            if len(batch) >= 20000:
                con.executemany("INSERT OR IGNORE INTO touch VALUES (?, ?, ?, ?)", batch)
                batch = []
        if batch:
            con.executemany("INSERT OR IGNORE INTO touch VALUES (?, ?, ?, ?)", batch)
        for test in seen_tests:
            if test:
                test_procs[(test, stage)].add(fp)
        for test, outcome, expected in _read_test_meta(fp):
            if not test:
                continue
            if outcome is not None:
                test_outcome[(test, stage)] = outcome
            if expected:
                test_expected[(test, stage)] += int(expected)

    con.execute(
        "CREATE TABLE test_meta (test TEXT, stage TEXT, outcome TEXT, "
        "expected_workers INTEGER, saved_procs INTEGER, PRIMARY KEY(test, stage))"
    )
    con.executemany(
        "INSERT OR REPLACE INTO test_meta VALUES (?, ?, ?, ?, ?)",
        [
            (
                t,
                s,
                test_outcome.get((t, s)),
                test_expected.get((t, s), 0),
                len(test_procs.get((t, s), ())),
            )
            for (t, s) in set(test_procs) | set(test_outcome) | set(test_expected)
        ],
    )

    con.execute("CREATE INDEX ix_file ON touch(file)")
    con.execute("CREATE INDEX ix_func ON touch(file, qualname)")
    con.execute("CREATE INDEX ix_test ON touch(test)")
    con.execute("CREATE INDEX ix_stage ON touch(stage)")
    con.commit()
    return con, len(files)


def _collect_defs(node, prefix, rel, funcs):
    """Record functions/methods reachable through class and control-flow nesting (matches PY_START co_qualname)."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.add((rel, prefix + child.name))
        elif isinstance(child, ast.ClassDef):
            _collect_defs(child, f"{prefix}{child.name}.", rel, funcs)
        else:
            _collect_defs(child, prefix, rel, funcs)


def enumerate_defs(source_root):
    """Return (coverable_files, funcs) for the coverage-rate denominator, matching what the tracker records."""
    funcs = set()
    for dirpath, _dirs, names in os.walk(source_root):
        for nm in names:
            if not nm.endswith(".py"):
                continue
            full = os.path.join(dirpath, nm)
            rel = canon(os.path.abspath(full))
            try:
                tree = ast.parse(open(full, encoding="utf-8").read())
            except (OSError, SyntaxError):
                continue
            _collect_defs(tree, "", rel, funcs)
    return {f for f, _q in funcs}, funcs


_CSS = (
    "<style>body{font-family:-apple-system,Segoe UI,sans-serif;margin:24px;font-size:13px}"
    "h1{font-size:20px}a{color:#1a5fb4;text-decoration:none}a:hover{text-decoration:underline}"
    "table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:3px 8px;text-align:right}"
    "td.l,th.l{text-align:left}th{background:#eee}.mono{font-family:ui-monospace,Menlo,monospace;font-size:12px}"
    ".t{color:#557;font-size:11px}summary{cursor:pointer;font-weight:600}"
    "details{margin:5px 0;border:1px solid #ddd;border-radius:6px;padding:4px 10px}</style>"
)


def _page_name(file):
    # Readable slug plus a path digest so distinct files (a-b.py vs a_b.py) never share a page.
    slug = re.sub(r"[^A-Za-z0-9]", "_", file)
    return f"{slug}-{hashlib.sha1(file.encode('utf-8')).hexdigest()[:10]}.html"


def write_html_tree(out_dir, con, rate_line, n_tests, n_data_files):
    """Split HTML report, queried per file from the merged DB (bounded memory per page)."""
    files_dir = os.path.join(out_dir, "files")
    os.makedirs(files_dir, exist_ok=True)
    files = [
        r[0] for r in con.execute("SELECT DISTINCT file FROM touch WHERE test != '' ORDER BY file")
    ]

    for file in files:
        # one file's (qualname -> tests) held at a time
        funcs = defaultdict(list)
        for qual, test in con.execute(
            "SELECT DISTINCT qualname, test FROM touch WHERE file = ? AND test != '' "
            "ORDER BY qualname, test",
            (file,),
        ):
            funcs[qual].append(test)
        h = [f"<!doctype html><meta charset='utf-8'><title>{escape(file)}</title>{_CSS}"]
        h.append(
            f"<p><a href='../index.html'>&larr; index</a></p><h1 class='mono'>{escape(file)}</h1>"
        )
        h.append(f"<p class='t'>{len(funcs)} functions touched</p>")
        for qual in sorted(funcs):
            tests = funcs[qual]
            h.append(
                f"<details><summary class='mono'>{escape(qual)} "
                f"<span class='t'>&larr; {len(tests)} tests</span></summary>"
            )
            for t in tests:
                h.append(f"<div class='mono t'>{escape(t)}</div>")
            h.append("</details>")
        with open(os.path.join(files_dir, _page_name(file)), "w") as fh:
            fh.write("\n".join(h))

    # index: files grouped by directory, with per-file counts (SQL aggregate)
    counts = {
        f: (nf, nt)
        for f, nf, nt in con.execute(
            "SELECT file, COUNT(DISTINCT qualname), COUNT(DISTINCT test) "
            "FROM touch WHERE test != '' GROUP BY file"
        )
    }
    by_dir = defaultdict(list)
    for f in files:
        by_dir[os.path.dirname(f)].append(f)
    h = [f"<!doctype html><meta charset='utf-8'><title>CBTS coverage</title>{_CSS}"]
    h.append("<h1>CBTS function-level coverage (PY_START)</h1>")
    h.append(
        f"<p>{n_tests} tests · {len(files)} product files touched · {n_data_files} data files merged.</p>"
    )
    if rate_line:
        h.append(f"<p><b>{escape(rate_line)}</b></p>")
    for d in sorted(by_dir):
        flist = by_dir[d]
        h.append(
            f"<details><summary class='mono'>{escape(d or '.')} <span class='t'>({len(flist)} files)</span></summary>"
        )
        h.append("<table><tr><th class='l'>file</th><th>funcs</th><th>tests</th></tr>")
        for f in flist:
            nf, nt = counts.get(f, (0, 0))
            h.append(
                f"<tr><td class='l mono'><a href='files/{_page_name(f)}'>{escape(os.path.basename(f))}</a></td>"
                f"<td>{nf}</td><td>{nt}</td></tr>"
            )
        h.append("</table></details>")
    with open(os.path.join(out_dir, "index.html"), "w") as fh:
        fh.write("\n".join(h))


def write_json(con, out_path):
    """Stream the per-test touch map to JSON without holding it all in memory."""
    with open(out_path, "w") as f:
        f.write("{")
        first = True
        cur_test = None
        touched = []
        rows = con.execute(
            "SELECT DISTINCT test, file, qualname FROM touch WHERE test != '' "
            "ORDER BY test, file, qualname"
        )
        for test, file, qual in rows:
            if test != cur_test:
                if cur_test is not None:
                    f.write(
                        ("," if not first else "")
                        + json.dumps(cur_test)
                        + ":"
                        + json.dumps(sorted(touched))
                    )
                    first = False
                cur_test = test
                touched = []
            touched.append(f"{file}::{qual}")
        if cur_test is not None:
            f.write(
                ("," if not first else "")
                + json.dumps(cur_test)
                + ":"
                + json.dumps(sorted(touched))
            )
        f.write("}")


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

    # Merge into a SQLite DB (bounded memory): --out-sqlite directly, else a unique per-invocation temp file.
    keep = a.out_sqlite is not None
    if keep:
        db_path = a.out_sqlite
    else:
        fd, db_path = tempfile.mkstemp(prefix="cbts_merge_", suffix=".sqlite")
        os.close(fd)
    con, n_data_files = merge_to_sqlite(a.glob, db_path)

    n_tests = con.execute("SELECT COUNT(DISTINCT test) FROM touch WHERE test != ''").fetchone()[0]
    n_files = con.execute("SELECT COUNT(DISTINCT file) FROM touch WHERE test != ''").fetchone()[0]
    n_funcs = con.execute(
        "SELECT COUNT(*) FROM (SELECT DISTINCT file, qualname FROM touch WHERE test != '')"
    ).fetchone()[0]
    print(
        f"CBTS PY_START: merged {n_data_files} data files -> "
        f"{n_tests} tests, {n_files} product files, {n_funcs} functions touched"
    )

    # Completeness: a (test, stage) is safe to skip only if it passed and every expected
    # process saved (saved_procs >= expected_workers + 1, the +1 being the coordinator).
    n_meta = con.execute("SELECT COUNT(*) FROM test_meta WHERE test != ''").fetchone()[0]
    n_unsafe = con.execute(
        "SELECT COUNT(*) FROM test_meta WHERE test != '' AND "
        "(outcome IS NULL OR outcome != 'passed' OR saved_procs < expected_workers + 1)"
    ).fetchone()[0]
    print(
        f"CBTS completeness: {n_meta} (test, stage) rows, "
        f"{n_unsafe} not safe to skip (non-passed or saved_procs < expected_workers + 1)"
    )

    meta = {
        "schema_version": "2",
        "tests": str(n_tests),
        "files": str(n_files),
        "functions": str(n_funcs),
    }
    rate_line = ""
    if a.source_root:
        coverable_files, all_funcs = enumerate_defs(a.source_root)
        touched_files = {
            r[0] for r in con.execute("SELECT DISTINCT file FROM touch WHERE test != ''")
        } & coverable_files
        touched_funcs = {
            t for t in con.execute("SELECT DISTINCT file, qualname FROM touch WHERE test != ''")
        } & all_funcs
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

    con.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)", list(meta.items()))
    con.commit()
    if keep:
        print(f"wrote {a.out_sqlite}")
    if a.out_json:
        write_json(con, a.out_json)
        print(f"wrote {a.out_json}")
    if a.out_dir:
        write_html_tree(a.out_dir, con, rate_line, n_tests, n_data_files)
        print(f"wrote {a.out_dir}/index.html")

    con.close()
    if not keep:
        try:
            os.remove(db_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()

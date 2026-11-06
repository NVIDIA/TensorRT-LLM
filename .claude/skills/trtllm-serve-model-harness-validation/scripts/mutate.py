#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mutation gate: prove the properties can actually fail.

A property that no bug can break is worth nothing, and a green run of such a
property is worse than no run because it buys false confidence.  This plants
known bugs in the module under test, one at a time, and requires each property
to catch them.

Mutations live in a ``.mutations`` file next to the target, one TOML-ish block
per bug::

    [mutation]
    name = drop-trailing-args
    file = tensorrt_llm/serve/tool_parser/base_tool_parser.py
    find = argument_diff = cur_args_json[sent:]
    replace = argument_diff = ""

``find`` must match exactly once in the file.  Each mutation is written into the
file in place (the original is restored from a temporary backup afterwards), the
property runner is re-run against it, and the mutation is DETECTED when the run
fails and MISSED when it still passes.  Do not edit or test the target files
while the gate runs.

Usage (``references/verify-logic.md``)::

    python3 scripts/mutate.py <file.mutations> --runner <property runner> \
        [--runner-args "..."] [--python <interpreter with hypothesis>] [--timeout S]

A mutated run that does not end within ``--timeout`` seconds (a planted bug that loops) is
stopped and counts as DETECTED: the property did not hold.

``file`` paths are relative to the root of the repository that holds this skill.

Exit codes: 0 every mutation detected on a passing baseline, 1 a mutation was missed or the
baseline already fails (then nothing is proven), 2 usage error, 3 UNAVAILABLE.
"""

import argparse
import functools
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]


def parse_mutations(path: Path):
    muts, cur = [], None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line == "[mutation]":
            cur = {}
            muts.append(cur)
            continue
        if not line or line.startswith("#") or cur is None:
            continue
        key, _, value = line.partition("=")
        cur[key.strip()] = value.strip()
    for m in muts:
        for key in ("name", "file", "find", "replace"):
            if key not in m:
                raise ValueError(f"mutation {m.get('name', '?')} lacks {key!r}")
    return muts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mutations", type=Path)
    ap.add_argument(
        "--runner",
        required=True,
        help="the property runner: a script that exits 0 when every property holds",
    )
    ap.add_argument("--runner-args", default="")
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds per mutated run")
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="the interpreter that runs the runner (hypothesis importable)",
    )
    args = ap.parse_args()

    # A planted bug is restored in a ``finally`` block. SIGTERM (a ``timeout``, a job limit)
    # would end the process without running it, so turn SIGTERM into an exit that does.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    print_now = functools.partial(print, flush=True)

    if not args.mutations.exists():
        print(f"UNAVAILABLE: no mutations file at {args.mutations}")
        return 3
    if not Path(args.python).exists():
        print(f"UNAVAILABLE: no interpreter at {args.python}")
        return 3

    runner = Path(args.runner).resolve()
    if not runner.exists():
        print(f"UNAVAILABLE: no property runner at {runner}")
        return 3
    muts = parse_mutations(args.mutations)

    # Baseline: the property must pass on the unmutated tree, otherwise a
    # DETECTED result proves nothing.
    base = subprocess.run(
        [args.python, str(runner), *args.runner_args.split()], capture_output=True, text=True
    )
    baseline_clean = base.returncode == 0
    print_now(
        f"baseline: {'PASS' if baseline_clean else 'ALREADY FAILING'} (exit {base.returncode})"
    )
    if not baseline_clean:
        print(
            "  note: with a failing baseline a mutation that still fails is "
            "not evidence; fix or filter the baseline first."
        )

    missed = []
    for m in muts:
        target = REPO / m["file"]
        original = target.read_text()
        if original.count(m["find"]) != 1:
            print(
                f"MISSED {m['name']}: 'find' matches "
                f"{original.count(m['find'])} times, need exactly 1"
            )
            missed.append(m["name"])
            continue
        backup = Path(tempfile.mkdtemp()) / target.name
        shutil.copy2(target, backup)
        try:
            target.write_text(original.replace(m["find"], m["replace"]))
            try:
                res = subprocess.run(
                    [args.python, str(runner), *args.runner_args.split()],
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                )
            except subprocess.TimeoutExpired:
                print_now(f"DETECTED {m['name']} (no result within {args.timeout:g} s)")
                continue
            detected = res.returncode != 0 and res.returncode != 3
            if base.returncode == res.returncode and not baseline_clean:
                detected = res.stdout != base.stdout
            print_now(f"{'DETECTED' if detected else 'MISSED'} {m['name']}")
            if not detected:
                missed.append(m["name"])
        finally:
            shutil.copy2(backup, target)
            shutil.rmtree(backup.parent, ignore_errors=True)

    print(f"\nSUMMARY: {len(muts) - len(missed)} DETECTED, {len(missed)} MISSED")
    if missed:
        print("missed: " + ", ".join(missed))
        print(
            "A missed mutation means the property does not constrain that "
            "behaviour. Strengthen the property; do not delete the mutation."
        )
    if not baseline_clean:
        print("The baseline fails, so no DETECTED above proves anything: the gate does not pass.")
    return 1 if missed or not baseline_clean else 0


if __name__ == "__main__":
    sys.exit(main())

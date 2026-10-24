# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``trtllm-modeling-v2 intake derive|check|confirm|compile``.

derive   --checkpoint C --site S [--goal G] -o OUT   derive a request (derive.py); exit 1 on a block
check    TASK            run the feasibility checks; exit 1 on any block
confirm  TASK --by NAME  record the confirmation in TASK.confirmed (refused while a block stands)
compile  TASK --site S -o OUT   write the resolved spec the pipeline runs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ..task_schema import TaskSchemaError
from .checks import blocking, run_checks
from .compile import compile_files
from .derive import REPORT_FILENAME, derive_task, parse_goal, render_report
from .freeze import confirm
from .schema import IntakeError


def _derive(args: argparse.Namespace) -> int:
    try:
        site = yaml.safe_load(args.site.read_text(encoding="utf-8")) or {}
        derivation = derive_task(args.checkpoint, site, parse_goal(args.goal), site_path=args.site)
    except IntakeError as exc:
        print(f"error: {exc}")
        return 2
    findings = run_checks(derivation.task)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(derivation.task, sort_keys=False), encoding="utf-8")
    report = args.out.with_name(REPORT_FILENAME)
    report.write_text(render_report(derivation, args.out, findings), encoding="utf-8")
    for finding in findings:
        print(finding.render())
    print(f"wrote {args.out} and {report}")
    return 1 if blocking(findings) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trtllm-modeling-v2 intake")
    sub = parser.add_subparsers(dest="cmd", required=True)
    derive = sub.add_parser("derive")
    derive.add_argument("--checkpoint", required=True, type=Path)
    derive.add_argument("--site", required=True, type=Path)
    derive.add_argument("--goal", default=None)
    derive.add_argument("-o", "--out", required=True, type=Path)
    check = sub.add_parser("check")
    check.add_argument("task", type=Path)
    conf = sub.add_parser("confirm")
    conf.add_argument("task", type=Path)
    conf.add_argument("--by", required=True)
    comp = sub.add_parser("compile")
    comp.add_argument("task", type=Path)
    comp.add_argument("--site", required=True, type=Path)
    comp.add_argument("-o", "--out", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.cmd == "derive":
        return _derive(args)
    task = yaml.safe_load(args.task.read_text(encoding="utf-8"))
    findings = run_checks(task)
    if args.cmd == "check":
        for finding in findings:
            print(finding.render())
        return 1 if blocking(findings) else 0
    if args.cmd == "confirm":
        if blocking(findings):
            for finding in blocking(findings):
                print(finding.render())
            print("not confirmed: answer the blocks above first")
            return 1
        try:
            stamp = confirm(args.task, args.by)
        except IntakeError as exc:
            print(f"error: {exc}")
            return 2
        print(f"confirmed by {stamp['by']} at {stamp['at']} ({stamp['sha256'][:12]})")
        return 0
    try:
        compiled = compile_files(args.task, args.site, args.out)
    except (IntakeError, TaskSchemaError) as exc:
        print(f"error: {exc}")
        return 2
    for warning in compiled.warnings:
        print(f"warning: {warning}")
    print(f"wrote {args.out}")
    return 0

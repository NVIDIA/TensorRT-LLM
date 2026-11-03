# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``trtllm-modeling-v2 intake``: from a request to a spec, before anything is launched.

``intake derive`` writes the derived spec (:mod:`.derive`) and stops: no run,
no GPU, and no ``site.yaml`` required (without one the environment is left as
an open question). ``intake lint`` checks a spec, derived or hand-written
(:mod:`.lint`). INTAKE.md, beside this module, is the procedure the operator follows
between them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from . import derive as derive_mod
from . import lint as lint_mod

#: The lint report ``intake derive`` writes beside the derived spec.
REPORT_FILENAME = "intake.report.md"

USAGE = """usage: trtllm-modeling-v2 intake {derive,lint} ...

  derive  derive a spec from --checkpoint, --goal and an optional --site; write it, its
          derive.report.md and intake.report.md into --out, and stop
  lint    lint a spec before launch (exit 1 while a finding stands)

spec/INTAKE.md in the workflow's package is the procedure.
"""


def derive_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="trtllm-modeling-v2 intake derive",
        description="Derive a spec and stop: no run, no GPU. The derived spec is a starting "
        "point to edit, not a launch.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--goal", default=None, help="Features, members and workload in one line.")
    parser.add_argument("--site", type=Path, default=None, help="The cluster's site.yaml, if any.")
    parser.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="The TensorRT-LLM checkout, when there is no --site.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Directory to write into.")
    args = parser.parse_args(argv)
    if args.site is not None:
        site = yaml.safe_load(args.site.read_text(encoding="utf-8")) or {}
    elif args.repo is not None:
        site = {"trtllm_repo_path": str(args.repo.resolve()), "environment": {"kind": "local"}}
    else:
        parser.error("give --site, or --repo when the cluster has no site.yaml")
    try:
        derivation = derive_mod.derive_spec(
            args.checkpoint, site, derive_mod.parse_goal(args.goal), site_path=args.site
        )
    except derive_mod.DeriveError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.site is None:
        derivation.questions.append(
            "environment: no --site was given, so the spec carries `environment: {kind: local}`; "
            "replace it with the cluster's (session_spool, its sessions, image, mounts)"
        )
    path = derive_mod.write_derivation(args.out, derivation)
    task = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    report = lint_mod.lint_task(
        task, checkpoint=args.checkpoint, repo=site["trtllm_repo_path"], base=path.resolve().parent
    )
    (args.out / REPORT_FILENAME).write_text(report.render(), encoding="utf-8")
    print(f"→ {path}\n→ {args.out / derive_mod.REPORT_FILENAME}\n→ {args.out / REPORT_FILENAME}")
    print(f"lint: {report.summary()}")
    for question in derivation.questions:
        print(f"  open: {question}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0 if argv else 2
    command, rest = argv[0], argv[1:]
    if command == "derive":
        return derive_main(rest)
    if command == "lint":
        return lint_mod.main(rest)
    print(USAGE, file=sys.stderr)
    return 2


__all__ = ["REPORT_FILENAME", "derive_main", "main"]

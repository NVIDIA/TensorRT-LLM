# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``trtllm-modeling-v2 intake``: from a request to a spec, before anything is launched.

``intake derive`` writes the derived spec and each deployment's options
(:mod:`.derive`), prints what the spec is likely to cost or miss
(:mod:`.advice`) and stops: no run, no GPU, and no ``site.yaml`` required
(without one the environment is left as an open question). ``intake lint``
prints the same advice for a spec, derived or hand-written; only a spec that
does not load fails it. INTAKE.md, beside this module, is the procedure the
operator follows between them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from . import advice as advice_mod
from . import deployments as deployments_mod
from . import derive as derive_mod

#: The advice ``intake derive`` writes beside the derived spec.
REPORT_FILENAME = "intake.report.md"

USAGE = """usage: trtllm-modeling-v2 intake {derive,lint} ...

  derive  derive a spec from --checkpoint, --goal and an optional --site; write it, each
          deployment's options (engines/), its derive.report.md and intake.report.md into
          --out, print the advice, and stop
  lint    print what a spec is likely to cost or miss; advice, never a stop (exit 1 only
          for a spec that does not load)

spec/INTAKE.md in the workflow's package is the procedure.
"""


def derive_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="trtllm-modeling-v2 intake derive",
        description="Derive a spec and stop: no run, no GPU. The derived spec is a starting "
        "point to edit, not a launch.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--goal", default=None, help="Layouts, features and workload in one line.")
    parser.add_argument("--site", type=Path, default=None, help="The cluster's site.yaml, if any.")
    parser.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="The TensorRT-LLM checkout, when there is no --site.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Directory to write into.")
    args = parser.parse_args(argv)
    if args.site is None and args.repo is None:
        parser.error("give --site, or --repo when the cluster has no site.yaml")
    try:
        if args.site is not None:
            site = derive_mod.read_site(args.site)
        else:
            site = {"trtllm_repo_path": str(args.repo.resolve()), "environment": {"kind": "local"}}
        derivation = derive_mod.derive_spec(
            args.checkpoint, site, derive_mod.parse_goal(args.goal), site_path=args.site
        )
    except derive_mod.DeriveError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if (args.out / derive_mod.SPEC_FILENAME).is_file():
        # Said before it happens: an edit made in these files does not survive this.
        print(f"→ {args.out} holds a spec derived before: it and its engines/ are written anew")
    path = derive_mod.write_derivation(args.out, derivation)
    task = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    # Read as a launch would read it: each options file by its path beside the spec.
    deployments_mod.resolve_paths(task, path.resolve().parent)
    advice = advice_mod.advise(task, checkpoint=args.checkpoint, repo=site["trtllm_repo_path"])
    (args.out / REPORT_FILENAME).write_text(advice.render(), encoding="utf-8")
    print(f"→ {path}\n→ {args.out / derive_mod.REPORT_FILENAME}\n→ {args.out / REPORT_FILENAME}")
    if advice.deployments:
        print(advice.deployments)
    print(f"advice: {advice.summary()}")
    for line in (*advice.warnings, *advice.constraints):
        print(f"  advice: {line}")
    for question in derivation.questions:
        print(f"  open: {question}")
    if args.site is None:
        print(
            "→ no --site was given, so the spec is for this one host (`environment: {kind: "
            "local}`); for a cluster, derive with its site file (--site)"
        )
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
        return advice_mod.main(rest)
    print(USAGE, file=sys.stderr)
    return 2


__all__ = ["REPORT_FILENAME", "derive_main", "main"]

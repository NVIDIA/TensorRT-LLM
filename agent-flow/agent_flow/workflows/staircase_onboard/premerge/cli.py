# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``staircase-onboard premerge <step>``: the pre-merge pack, one step at a time.

Exit codes: 0 the step passed (or, for a dry run, would pass); 1 it ran and
found something; 2 it could not run (bad arguments, dirty tree, no git).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from ..committer import (
    DEFAULT_BASE_REFS,
    CommitPolicyError,
    audit_commits,
    resolve_base,
    resolve_commit_identity,
)
from ._git import GitError, git
from .ci_case import build_cases, describe, find_test_db_entries, write_case
from .copyright import check_copyright
from .lint import LintError, run_lint
from .pack import build_pack, write_pack
from .pr_open import PrOpenError, PrRequest, open_pr
from .resign import ResignError, resign_branch
from .split import SplitError, parse_stack, split_branch


def _base(repo: str, explicit: str) -> str:
    base = resolve_base(repo, [explicit] if explicit else list(DEFAULT_BASE_REFS))
    if not base:
        print(
            f"error: no base: {explicit or ' / '.join(DEFAULT_BASE_REFS)} does not resolve in {repo}",
            file=sys.stderr,
        )
        sys.exit(2)
    return base


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _default_out(repo: str) -> Path:
    """Inside the repository's git directory: kept, and never committable."""
    try:
        return Path(git(repo, "rev-parse", "--absolute-git-dir").strip()) / "staircase-premerge"
    except GitError:
        return Path(repo) / ".staircase-premerge"


def _cmd_lint(args) -> int:
    base = _base(args.repo, args.base)
    try:
        identity = resolve_commit_identity(args.repo)
    except CommitPolicyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    log_dir = Path(args.log_dir or _default_out(args.repo) / f"lint-{_stamp()}")
    try:
        result = run_lint(
            args.repo,
            base=base,
            log_dir=log_dir,
            pre_commit=args.pre_commit,
            dry_run=args.dry_run,
            message=args.message,
            skip_hooks=args.skip or (),
            auto_skip_type_check=not args.run_type_check,
            allow_semantic_changes=args.allow_semantic_changes,
            identity=identity,
        )
    except (LintError, GitError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(result.render())
    if args.copyright:
        print(check_copyright(args.repo, base, "HEAD", year=datetime.now().year).render())
    return 0 if result.ok else 1


def _cmd_ci(args) -> int:
    if args.list:
        print(
            describe(find_test_db_entries(args.repo, args.case))
            or f"no test-db line contains {args.case!r}"
        )
        return 0
    out_root = Path(args.out or _default_out(args.repo))
    cases = build_cases(
        args.repo,
        args.case,
        out_root=out_root,
        stage=args.stage,
        models_root=args.models_root,
        launcher=args.launcher,
        collect_only=args.collect_only,
        pythonpath=args.repo if args.pythonpath_tree else "",
    )
    if len(cases) != 1:
        print(
            f"error: expected exactly one test-db line containing {args.case!r}, found {len(cases)}:\n"
            + describe([c.entry for c in cases]),
            file=sys.stderr,
        )
        return 2
    script = write_case(cases[0])
    print(
        f"ci: {cases[0].entry.yml}: {cases[0].entry.line}\n    stage {cases[0].stage}\n    script {script}"
    )
    if not args.run:
        print("    not run (pass --run, or submit the script where the case's GPUs are)")
        return 0
    return subprocess.run(["bash", str(script)]).returncode


def _cmd_copyright(args) -> int:
    base = _base(args.repo, args.base)
    report = check_copyright(args.repo, base, args.head, year=args.year or datetime.now().year)
    print(report.render())
    return 0 if report.passed else 1


def _cmd_hygiene(args) -> int:
    base = _base(args.repo, args.base)
    findings = audit_commits(
        args.repo, f"{base}..{args.head}", allow_co_authors=args.allow_co_authors
    )
    for finding in findings:
        print(finding.render())
    print(f"hygiene: {len(findings)} finding(s) in {base[:10]}..{args.head}")
    return 0 if not findings else 1


def _cmd_split(args) -> int:
    try:
        stacks = [parse_stack(s) for s in args.stack]
        forbid = dict(f.split("=", 1) for f in args.forbid or ())
        result = split_branch(
            args.repo,
            base=args.base,
            src=args.src,
            stacks=stacks,
            drop_paths=tuple(args.drop_path) if args.drop_path else ("tensorrt_llm/bindings",),
            stacked=args.stacked,
            force=args.force,
            forbid=forbid,
        )
    except (SplitError, GitError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(result.render())
    return 0 if result.ok else 1


def _cmd_resign(args) -> int:
    try:
        result = resign_branch(args.repo, src=args.src, new_branch=args.new, base=args.base)
    except (ResignError, GitError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text = result.render_markdown(args.src)
    if args.map_out:
        Path(args.map_out).write_text(text, encoding="utf-8")
        print(f"map: {args.map_out}")
    print(text.split("\n| #", 1)[0])
    return 0 if result.ok else 1


def _cmd_pr_open(args) -> int:
    request = PrRequest(
        upstream=args.upstream,
        owner=args.owner,
        branch=args.branch,
        title=args.title,
        body=Path(args.body_file).read_text(encoding="utf-8"),
        base=args.base,
        draft=not args.ready,
    )
    try:
        report = open_pr(request, yes=args.yes, check_remote=args.check)
    except PrOpenError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    shown = dict(report)
    shown["payload"] = {**report["payload"], "body": report["payload"]["body"][:300] + " ..."}
    print(json.dumps(shown, indent=2))
    if report["dry_run"]:
        print("pr-open: DRY RUN. Nothing was sent. Add --yes to create the pull request.")
    return 0


def _cmd_pack(args) -> int:
    report = build_pack(args.repo, test_pattern=args.tests, base_ref=args.base)
    path = write_pack(report, args.out or _default_out(args.repo))
    print(f"pack: {path}")
    return 0 if report.hygiene_ok else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="staircase-onboard premerge",
        description="The pre-merge pack: lint, CI-case reproduction, copyright, commit "
        "audit, PR split, DCO re-sign, and an opt-in PR opener. Nothing here pushes.",
    )
    sub = parser.add_subparsers(dest="step", required=True)

    def repo_arg(p):
        p.add_argument("--repo", default=".", help="The checkout (default: the current directory).")

    p = sub.add_parser(
        "lint",
        help="pre-commit twice over the changed files; commit formatting only when pass 2 is clean",
    )
    repo_arg(p)
    p.add_argument(
        "--base", default="", help=f"Diff base (default: merge-base with {DEFAULT_BASE_REFS[0]})."
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Run both passes and the AST proof; commit nothing."
    )
    p.add_argument("--message", default="[None][chore] Apply pre-commit formatting")
    p.add_argument("--skip", action="append", help="A hook id to skip (repeatable).")
    p.add_argument(
        "--run-type-check",
        action="store_true",
        help="Do not auto-skip type-check where mypy is not importable.",
    )
    p.add_argument(
        "--allow-semantic-changes",
        action="store_true",
        help="Commit even when the AST proof finds a non-formatting change.",
    )
    p.add_argument("--pre-commit", default="pre-commit")
    p.add_argument("--log-dir", default="")
    p.add_argument("--copyright", action="store_true", help="Also report copyright headers.")
    p.set_defaults(func=_cmd_lint)

    p = sub.add_parser("ci", help="write (and with --run, run) one test-db line as CI runs it")
    repo_arg(p)
    p.add_argument("--case", required=True, help="Substring of exactly one test-db test id.")
    p.add_argument(
        "--stage",
        default="",
        help="CI stage name (default: from scripts/test_to_stage_mapping.py).",
    )
    p.add_argument(
        "--launcher",
        default="",
        help="Rank launcher prefix, e.g. 'mpirun -n 4 --oversubscribe --allow-run-as-root trtllm-llmapi-launch'.",
    )
    p.add_argument(
        "--models-root", default="", help="LLM_MODELS_ROOT for the case (default: inherited)."
    )
    p.add_argument(
        "--pythonpath-tree",
        action="store_true",
        help="Put the tree on PYTHONPATH (a worktree whose package is not the installed one).",
    )
    p.add_argument("--collect-only", action="store_true")
    p.add_argument(
        "--out",
        default="",
        help="Where case directories go (default: <git dir>/staircase-premerge).",
    )
    p.add_argument("--list", action="store_true", help="List matching test-db lines and stop.")
    p.add_argument(
        "--run", action="store_true", help="Run the script here instead of only writing it."
    )
    p.set_defaults(func=_cmd_ci)

    p = sub.add_parser(
        "copyright", help="NVIDIA headers on new files, current year on modified ones"
    )
    repo_arg(p)
    p.add_argument("--base", default="")
    p.add_argument("--head", default="HEAD")
    p.add_argument("--year", type=int, default=0)
    p.set_defaults(func=_cmd_copyright)

    p = sub.add_parser(
        "hygiene", help="sign-off by author, no co-author, no symlink, no placeholder, per commit"
    )
    repo_arg(p)
    p.add_argument("--base", default="")
    p.add_argument("--head", default="HEAD")
    p.add_argument("--allow-co-authors", action="store_true")
    p.set_defaults(func=_cmd_hygiene)

    p = sub.add_parser(
        "split", help="split base..src into PR stacks by path, with consistency checks"
    )
    repo_arg(p)
    p.add_argument("--base", required=True)
    p.add_argument("--src", required=True)
    p.add_argument(
        "--stack",
        action="append",
        required=True,
        help="NAME=BRANCH:prefix[,prefix...]; '*' for the rest. Repeatable; order matters with --stacked.",
    )
    p.add_argument(
        "--stacked", action="store_true", help="Build each stack on the previous one's tip."
    )
    p.add_argument("--force", action="store_true", help="Reset existing stack branches.")
    p.add_argument(
        "--drop-path",
        action="append",
        help="Remove a path from every commit (default tensorrt_llm/bindings).",
    )
    p.add_argument(
        "--forbid", action="append", help="NAME=REGEX that must not match in that stack's paths."
    )
    p.set_defaults(func=_cmd_split)

    p = sub.add_parser("resign", help="new branch with every commit signed off by its own author")
    repo_arg(p)
    p.add_argument("--src", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--map-out", default="")
    p.set_defaults(func=_cmd_resign)

    p = sub.add_parser(
        "pr-open", help="OPT-IN: open a draft PR via the GitHub API (dry run unless --yes)"
    )
    p.add_argument("--upstream", default="NVIDIA/TensorRT-LLM")
    p.add_argument(
        "--owner", required=True, help="The fork's owner (the branch must already be pushed there)."
    )
    p.add_argument("--branch", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--body-file", required=True)
    p.add_argument("--base", default="main")
    p.add_argument(
        "--ready", action="store_true", help="Open as ready for review instead of draft."
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Read-only GETs: does the fork branch exist, is a PR open already.",
    )
    p.add_argument("--yes", action="store_true", help="Actually create the pull request.")
    p.set_defaults(func=_cmd_pr_open)

    p = sub.add_parser(
        "pack", help="the cheap checks plus the remaining commands, as a markdown report"
    )
    repo_arg(p)
    p.add_argument("--base", default="")
    p.add_argument("--tests", default="", help="Substring naming the model's test-db lines.")
    p.add_argument("--out", default="")
    p.set_defaults(func=_cmd_pack)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.func(args) or 0)


__all__ = ["main"]

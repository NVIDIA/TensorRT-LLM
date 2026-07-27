#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Check whether a PR's base is too far behind the target branch.

Fails (or warns, in warn-only mode) when the merge-base between the PR head
and the target branch is older than configured thresholds. See
``reports/TRTLLM-12092-design.md`` for the rationale.
"""

import os
import subprocess
import sys


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"::warning::{name}='{raw}' is not an integer; using default {default}")
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes"}


def main() -> int:
    pr_head = os.environ.get("PR_HEAD_SHA", "").strip()
    target_ref = os.environ.get("TARGET_REF", "origin/main").strip()
    commits_limit = _env_int("COMMITS_BEHIND_LIMIT", 200)
    age_limit_days = _env_int("BASE_AGE_LIMIT_DAYS", 14)
    enforce = _env_bool("ENFORCE")

    if not pr_head:
        print("::error::PR_HEAD_SHA is not set")
        return 1

    merge_base = _git("merge-base", pr_head, target_ref)
    commits_behind = int(_git("rev-list", "--count", f"{merge_base}..{target_ref}"))

    target_ts = int(_git("show", "-s", "--format=%ct", target_ref))
    base_ts = int(_git("show", "-s", "--format=%ct", merge_base))
    age_days = max(0.0, (target_ts - base_ts) / 86400.0)

    base_summary = _git("show", "-s", "--format=%h %s", merge_base)
    target_summary = _git("show", "-s", "--format=%h %s", target_ref)

    commits_exceeded = commits_behind > commits_limit
    age_exceeded = age_days > age_limit_days
    stale = commits_exceeded or age_exceeded

    summary_lines = [
        "PR base freshness report",
        f"  target ref:            {target_ref}",
        f"  commits behind target: {commits_behind} (limit: {commits_limit})",
        f"  base commit age:       {age_days:.1f} days (limit: {age_limit_days})",
        f"  merge base:            {base_summary}",
        f"  target HEAD:           {target_summary}",
    ]
    print("\n".join(summary_lines))

    gh_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if gh_summary:
        with open(gh_summary, "a", encoding="utf-8") as fh:
            fh.write("## PR Base Freshness\n\n")
            fh.write("| metric | value | limit |\n")
            fh.write("| --- | --- | --- |\n")
            fh.write(f"| commits behind `{target_ref}` | {commits_behind} | {commits_limit} |\n")
            fh.write(f"| merge-base age (days) | {age_days:.1f} | {age_limit_days} |\n\n")
            fh.write(f"- merge base: `{base_summary}`\n")
            fh.write(f"- target HEAD: `{target_summary}`\n")

    if not stale:
        print("PR base freshness OK.")
        return 0

    reasons = []
    if commits_exceeded:
        reasons.append(f"{commits_behind} commits behind (limit {commits_limit})")
    if age_exceeded:
        reasons.append(f"base is {age_days:.1f} days old (limit {age_limit_days})")
    reason_str = "; ".join(reasons)

    guidance = (
        "To resolve: rebase onto the target branch (preferred) or merge it into this "
        "branch, then push again."
    )

    if not enforce:
        print(
            f"::warning::[warn-only] PR base is stale: {reason_str}. "
            "This check will start blocking merges once enforcement is enabled. "
            f"{guidance}"
        )
        return 0

    print(f"::error::PR base is stale: {reason_str}. {guidance}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

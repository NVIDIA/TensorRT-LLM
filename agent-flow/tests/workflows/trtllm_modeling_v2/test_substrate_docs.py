# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The operator-facing docs say what the substrate does, in the words its output uses."""

from __future__ import annotations

import re
from pathlib import Path

import agent_flow.workflows.trtllm_modeling_v2 as pkg
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate import scheduler

PACKAGE = Path(pkg.__file__).parent


def _doc(name: str) -> str:
    return (PACKAGE / name).read_text(encoding="utf-8")


def test_the_safe_kill_rule_is_written_down_where_an_operator_looks():
    """An operator facing a hung job must find the rule before reaching for kill."""
    text = _doc("TROUBLESHOOTING.md")
    assert "safe-kill rule" in text
    assert "spoolctl.py cancel" in text
    assert "_errhandler status=-61" in text


def test_every_verdict_reason_the_scheduler_writes_is_documented():
    """A receipt says `reason=<x>`; the operator greps for exactly that."""
    text = _doc("TROUBLESHOOTING.md")
    reasons = set(
        re.findall(r'Verdict\(\s*RC_[A-Z_]+,\s*"([a-z_]+)"', Path(scheduler.__file__).read_text())
    )
    reasons |= {"orphaned", "host_memory", "memory_pressure"}
    undocumented = sorted(r for r in reasons if f"reason={r}" not in text and f"`{r}`" not in text)
    assert not undocumented, f"verdict reasons missing from TROUBLESHOOTING.md: {undocumented}"


def test_every_return_code_the_docs_quote_is_the_schedulers():
    text = _doc("TROUBLESHOOTING.md")
    quoted = {int(code) for code in re.findall(r"\(rc (\d+)\)", text)}
    codes = {getattr(scheduler, name) for name in dir(scheduler) if name.startswith("RC_")}
    assert quoted and quoted <= codes, quoted - codes


def test_the_build_page_sends_rebuilds_and_worktrees_through_the_substrate():
    text = _doc("BUILD_AND_TEST.md")
    assert "incremental_build.sh" in text and "fresh inode" in text
    assert "--tree <worktree>" in text
    assert "trtllm-modeling-v2 session render" in text


def test_the_readme_describes_the_substrate():
    text = _doc("README.md")
    section = text[text.index("## Execution substrate\n") :].split("\n## ")[0]
    for fragment in ("--tree", "safe-kill rule", "trtllm-modeling-v2 driver", "session render"):
        assert fragment in section

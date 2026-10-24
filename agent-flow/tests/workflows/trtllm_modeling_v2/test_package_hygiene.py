# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The README's flags table is the CLI's."""

from __future__ import annotations

import re
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[3] / "agent_flow" / "workflows" / "trtllm_modeling_v2"


def test_the_readme_flags_table_is_the_cli():
    """Every flag the CLI accepts is in the README's table, and the table names no other."""
    import argparse

    from agent_flow.workflows.trtllm_modeling_v2 import cli

    captured: list[argparse.ArgumentParser] = []
    original = argparse.ArgumentParser.parse_args

    def capture(self, args=None, namespace=None):
        captured.append(self)
        return original(self, ["--task", "t.yaml"], namespace)

    argparse.ArgumentParser.parse_args = capture
    try:
        cli._parse_args([])
    finally:
        argparse.ArgumentParser.parse_args = original
    accepted = {
        option
        for action in captured[0]._actions
        for option in action.option_strings
        if option.startswith("--") and option != "--help"
    }

    readme = (PACKAGE / "README.md").read_text(encoding="utf-8")
    table = readme.split("## Flags", 1)[1].split("\n\n", 2)[1]
    documented = set(re.findall(r"`(--[a-z-]+)`", table))
    assert documented == accepted
    for subcommand in cli.SUBCOMMANDS:
        assert f"`trtllm-modeling-v2 {subcommand} " in readme

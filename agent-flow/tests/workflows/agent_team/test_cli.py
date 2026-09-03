"""Tests for the agent-team CLI argument surface."""

from __future__ import annotations

from agent_flow.workflows.agent_team import cli


def test_cli_defaults_to_in_process_tools():
    args = cli._parse_args(["--task", "t.yaml"])
    assert args.use_in_process_tools is True


def test_cli_no_mcp_tools_flag_disables_in_process_tools():
    args = cli._parse_args(["--task", "t.yaml", "--no-mcp-tools"])
    assert args.use_in_process_tools is False

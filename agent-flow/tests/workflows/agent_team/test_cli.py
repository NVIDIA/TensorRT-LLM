from __future__ import annotations

from agent_flow.workflows.agent_team import cli


def test_cli_concurrent_defaults_off():
    args = cli._parse_args(["--task", "t.yaml"])
    assert args.concurrent is False
    assert args.max_parallel == 8


def test_cli_concurrent_flag_and_max_parallel():
    args = cli._parse_args(["--task", "t.yaml", "--concurrent", "--max-parallel", "4"])
    assert args.concurrent is True
    assert args.max_parallel == 4

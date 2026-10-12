# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DSpark as its own speculator kind, and never a spelling of MTP.

DeepSeek-V4.1 ships its DSpark block drafter as `mtp.*` with
`num_nextn_predict_layers: 3`. Every test here guards one of the ways that
namespace leads a run to gate the wrong drafter, or the right drafter at the
wrong depth, or not at all.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.staircase_onboard import gates, workflow
from agent_flow.workflows.staircase_onboard.environment import resolve_environment
from agent_flow.workflows.staircase_onboard.gates import ShellResult

REPO_ROOT = Path(__file__).resolve().parents[4]
QUICKSTART = REPO_ROOT / "examples" / "llm-api" / "quickstart_advanced.py"


def _env():
    return resolve_environment(
        {"environment": {"kind": "local", "python": "/opt/venv/bin/python", "env": {}}},
        devices=(0,),
        world_size=1,
    )


def _options(tmp_path, **speculative) -> str:
    path = tmp_path / "dspark_llm_api.yaml"
    path.write_text(yaml.safe_dump({"speculative_config": speculative}), encoding="utf-8")
    return str(path)


# ------------------------------------------------------------------- flags


def test_dspark_is_its_own_algorithm_never_mtp():
    flags = gates.speculation_flags(["dspark"], max_draft_len=5)
    assert flags == ["--spec_decode_algo=DSPARK", "--spec_decode_max_draft_len=5"]
    assert not any("MTP" in flag for flag in flags)


@pytest.mark.parametrize(
    "speculative",
    [
        {"decoding_type": "DSpark", "max_draft_len": 5},
        {"decoding_type": "DSpark", "block_size": 5},
        {"decoding_type": "DSpark", "max_draft_len": 5, "block_size": 5},
    ],
)
def test_the_dspark_depth_comes_from_the_options_yaml(tmp_path, speculative):
    """Quickstart builds the config from the flag alone.

    A depth of 1 against a drafter trained for a 5-token block fails the
    `block_size == max_draft_len` validation before the engine starts, so the
    old default is not merely a weaker gate here, it is no gate.
    """
    accuracy = {"llm_api_options": {"dspark": _options(tmp_path, **speculative)}}
    assert gates.resolve_max_draft_len(accuracy, "dspark") == 5


def test_max_draft_len_wins_over_its_deprecated_alias(tmp_path):
    """The engine's own order: `MTPDecodingConfig` drops the alias when both are set."""
    both = _options(tmp_path, decoding_type="MTP", max_draft_len=2, num_nextn_predict_layers=3)
    assert gates.resolve_max_draft_len({"llm_api_options": {"mtp": both}}, "mtp") == 2


def test_dspark_is_acceptance_gated():
    assert "dspark" in workflow.ACCEPTANCE_GATED_UNITS


# ------------------------------------------------ the acceptance gate for dspark


def _task(tmp_path, floor):
    return {
        "checkpoint_path": str(tmp_path / "ckpt"),
        "trtllm_repo_path": str(tmp_path),
        "parallel": "tp1",
        "features": ["dspark"],
        "runtime": {"kv_cache_reuse": True, "extra_flags": []},
        "accuracy": {
            "llm_api_options": {
                "dspark": _options(tmp_path, decoding_type="DSpark", max_draft_len=5)
            },
            "acceptance": {"dspark": {"min_absolute": floor}},
        },
    }


def _runner(accept_line):
    """`=require` drafts; `=off` cannot load a new architecture at all."""
    seen: list[str] = []

    def run(command, env=None, timeout=0):
        seen.append(command)
        if "TRTLLM_MODELING_V2=off" in command:
            return ShellResult(
                returncode=1,
                stderr="RuntimeError: Unknown model architecture: DeepseekV41ForCausalLM",
            )
        return ShellResult(returncode=0, stdout=accept_line)

    run.seen = seen  # type: ignore[attr-defined]
    return run


def test_the_declared_floor_stands_in_for_the_missing_stock_leg(tmp_path):
    """A new architecture has no `=off` leg, so the declared floor decides.

    `min_absolute` must work for dspark exactly as it does for mtp, and the
    verdict must say it is the weaker claim.
    """
    run = _runner(
        "[0] Accept rate: 3.10 (tokens=62, iterations=20)\n"
        "[1] Accept rate: 2.90 (tokens=58, iterations=20)\n"
    )
    result = gates.run_acceptance_gate(
        _env(),
        task_data=_task(tmp_path, 1.2),
        runner=run,
        unit="dspark",
        features=["dspark"],
    )
    assert result.passed, result.detail
    assert result.measured == pytest.approx(120 / 40)
    assert result.reference == pytest.approx(1.2)
    assert "WEAKER CLAIM" in result.detail
    seen = run.seen  # type: ignore[attr-defined]
    assert "--spec_decode_algo=DSPARK" in seen[0]
    assert "--spec_decode_max_draft_len=5" in seen[0]


def test_below_the_declared_floor_fails(tmp_path):
    run = _runner("[0] Accept rate: 1.05 (tokens=42, iterations=40)\n")
    result = gates.run_acceptance_gate(
        _env(), task_data=_task(tmp_path, 1.2), runner=run, unit="dspark", features=["dspark"]
    )
    assert not result.passed
    # Long enough to be a measurement, so the floor is what failed it.
    assert "too short" not in result.detail
    assert result.measured == pytest.approx(1.05)


# --------------------------------------------------------------- quickstart


def _function_source(name: str) -> str:
    source = QUICKSTART.read_text(encoding="utf-8")
    node = next(
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return ast.get_source_segment(source, node) or ""


def test_quickstart_builds_a_dspark_config():
    """The gates emit `--spec_decode_algo=DSPARK`; quickstart has to honour it."""
    setup = _function_source("setup_llm")
    branch = setup[setup.index('spec_decode_algo == "DSPARK"') :]
    branch = branch[: branch.index("elif")]
    assert "DSparkDecodingConfig(" in branch
    assert "max_draft_len=args.spec_decode_max_draft_len" in branch


def test_the_accept_rate_line_does_not_depend_on_the_algorithm():
    """The acceptance gate parses `Accept rate: ... (tokens=T, iterations=I)`.

    It is printed from the streamed iterations in `main`, which never looks at
    which speculator is configured -- so DSpark feeds it exactly as MTP does.
    """
    main = _function_source("main")
    assert "Accept rate:" in main
    assert "spec_decode_algo" not in main

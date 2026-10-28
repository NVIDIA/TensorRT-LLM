# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The two-leg gates for features that are output-identical by construction."""

from __future__ import annotations

from agent_flow.workflows.trtllm_modeling_v2.checks.witness_disagg import (
    ABLATE_STATE,
    run_equivalence_witness,
)
from agent_flow.workflows.trtllm_modeling_v2.checks.witness_protocol import LEG_B_ABLATE_ENV
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment

_TASK = {
    "trtllm_repo_path": "/repo",
    "checkpoint_path": "/ckpt",
    "parallel": "tp1",
    "features": [],
    "runtime": {"cuda_graph": True, "kv_cache_reuse": True, "extra_flags": []},
}


def _env():
    return RunEnvironment(kind="local", python="/opt/venv/bin/python", world_size=1)


def _ids(text: str) -> list[int]:
    """Token ids for ``text``: one per character, so ids differ where the text does."""
    return [ord(c) for c in text]


#: In a command: it is leg B's self-test run.
_ABLATE = f"{LEG_B_ABLATE_ENV}={ABLATE_STATE}"


def _ablated(text: str = " lost the thread", what: str = "conv, ssm") -> str:
    """Leg B's self-test run: its one continuation, and the line saying what it destroyed."""
    return _whole([text]) + f"[gen] my_leg_b[gen]: Witness ablated: {what}\n"


def _whole(texts, ids=None) -> str:
    """Quickstart's output under TRTLLM_WITNESS_FULL_OUTPUT=1 (see the example script).

    ``ids`` are the token ids printed per continuation, one per character of
    its text when not given; ``False`` prints the whole text without them.
    """
    out = ""
    for i, text in enumerate(texts):
        out += f"[{i}] Prompt: 'p'..., Generated text: {text[:200]!r}...\n"
        if ids is not False:
            out += f"[{i}] Witness token ids: {list(_ids(text) if ids is None else ids[i])}\n"
        out += f"[{i}] Witness text: {text!r}\n"
    return out


def _run_equivalence(tmp_path, agg_out, dis_out, **kwargs):
    """Both legs, and leg B's self-test run, whose continuation moves."""
    calls: list[str] = []

    def runner(command, env=None, timeout=0):
        calls.append(command)
        if _ABLATE in command:
            return CommandResult(returncode=0, stdout=_ablated())
        return CommandResult(returncode=0, stdout=agg_out if len(calls) == 1 else dis_out)

    kwargs.setdefault("contrast_command", "/onboard/scripts/my_leg_b.py")
    kwargs.setdefault("workdir", tmp_path / "witness")
    kwargs.setdefault("task_data", _TASK)
    result = run_equivalence_witness(
        kwargs.pop("environment", _env()),
        contrast_env={"TRTLLM_DISAGG": "1"},
        runner=runner,
        **kwargs,
    )
    return result, calls


def test_a_failing_verdict_says_where_each_continuation_first_differs(tmp_path):
    """Lost state shows at the first decode step; a late flip is numerics."""
    agg = _whole([" one two three", " same"], ids=[(1, 2, 3, 4, 5), (7, 7)])
    dis = _whole([" one two four", " same"], ids=[(1, 2, 3, 9, 5), (7, 7)])
    result, _ = _run_equivalence(tmp_path, agg, dis)
    assert not result.passed
    assert "continuation(s) [0]" in result.detail
    assert "#0 first differs at token 3" in result.detail
    assert "logit" in result.detail and "fluent, wrong text" in result.detail
    assert "around token 3" in result.stdout_tail

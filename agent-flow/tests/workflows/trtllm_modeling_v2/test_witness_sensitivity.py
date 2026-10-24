# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The equivalence witness proves it could have failed, and crosses every rank.

Token identity across the hop is only evidence when the workload would have
shown a lost state. Before a pass, leg B runs once more on one prompt with
the received state destroyed, and that continuation must move. Under
attention DP a one-request-at-a-time leg crosses every hop from rank 0 to
rank 0 unless each hop is pinned, so the witness hands leg B the pairs.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest

from agent_flow.workflows.trtllm_modeling_v2.checks import gates
from agent_flow.workflows.trtllm_modeling_v2.checks.witness import (
    ABLATE_STATE,
    LEG_B_ABLATE_ENV,
    LEG_B_PROMPTS_FILE_ENV,
    LEG_B_RANK_PAIRS_ENV,
    attention_dp_size,
    natural_text,
    parse_witness_ablations,
    parse_witness_ranks,
    rank_pairs,
    run_equivalence_witness,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CommandResult

_TASK = {
    "trtllm_repo_path": "/repo",
    "checkpoint_path": "/ckpt",
    "parallel": "tp1",
    "features": [],
    "runtime": {"cuda_graph": True, "kv_cache_reuse": True, "extra_flags": []},
}
_ABLATE = f"{LEG_B_ABLATE_ENV}={ABLATE_STATE}"
_TEXTS = (" Paris, a city on the Seine.", " the process by which plants make sugar.")


def _env(world: int = 1) -> RunEnvironment:
    devices = tuple(range(world)) if world > 1 else ()
    return RunEnvironment(
        kind="local", python="/opt/venv/bin/python", world_size=world, devices=devices
    )


def _ids(text: str) -> list[int]:
    return [ord(c) for c in text]


def _whole(texts, ranks=None) -> str:
    """A leg's output under TRTLLM_WITNESS_FULL_OUTPUT=1; ``ranks`` adds leg B's hop lines."""
    out = ""
    for i, text in enumerate(texts):
        if ranks is not None:
            out += f"[{i}] Witness ranks: ctx={ranks[i][0]} gen={ranks[i][1]}\n"
        out += f"[{i}] Prompt: 'p'..., Generated text: {text[:200]!r}...\n"
        out += f"[{i}] Witness token ids: {_ids(text)}\n"
        out += f"[{i}] Witness text: {text!r}\n"
    return out


def _destroyed(text: str = " lost the thread entirely", what: str = "conv, ssm") -> str:
    """The self-test leg's output: its one continuation and what it destroyed, engine-tagged."""
    return _whole([text]) + f"[gen] disagg_leg_b[gen]: Witness ablated: {what}\n"


class _Legs:
    """Leg A prints ``agg``, leg B ``dis``, and leg B's self-test run ``ablated``."""

    def __init__(self, agg: str, dis: str | None = None, ablated: str | None = None, ok=True):
        self.agg, self.dis = agg, agg if dis is None else dis
        self.ablated = _destroyed() if ablated is None else ablated
        self.ok = ok
        self.calls: list[str] = []

    def __call__(self, command, env=None, timeout=0):
        self.calls.append(command)
        if _ABLATE in command:
            return CommandResult(returncode=0 if self.ok else 1, stdout=self.ablated)
        legs = [call for call in self.calls if _ABLATE not in call]
        return CommandResult(returncode=0, stdout=self.agg if len(legs) == 1 else self.dis)


def _witness(tmp_path, legs: _Legs, environment: RunEnvironment | None = None, **kwargs):
    kwargs.setdefault("contrast_command", "/onboard/scripts/disagg_leg_b.py")
    kwargs.setdefault("workdir", tmp_path / "witness")
    return run_equivalence_witness(
        environment or _env(),
        task_data=_TASK,
        contrast_env={"TRTLLM_DISAGG": "1"},
        runner=legs,
        **kwargs,
    )


def _assignment(command: str, name: str) -> str:
    for word in shlex.split(command):
        if word.startswith(f"{name}="):
            return word.split("=", 1)[1]
    raise AssertionError(f"{name} not exported in {command}")


def _handed(command: str) -> list[str]:
    """The prompts a leg-B run was handed in its prompts file."""
    return json.loads(Path(_assignment(command, LEG_B_PROMPTS_FILE_ENV)).read_text())["prompts"]


# ------------------------------------------------------------- the self-test


def test_a_pass_is_trusted_only_after_destroying_the_state_moves_the_output(tmp_path):
    legs = _Legs(_whole(_TEXTS))
    result = _witness(tmp_path, legs)
    assert result.passed, result.detail
    agg, dis, ablated = legs.calls
    assert _ABLATE not in agg and _ABLATE not in dis
    assert _assignment(ablated, "TRTLLM_DISAGG") == "1"
    assert "with the received state destroyed (conv, ssm) continuation #0" in result.detail
    assert "first differed at token 1" in result.detail
    assert "sees a hop that loses state" in result.detail


def test_the_self_test_sends_only_the_longest_prompt(tmp_path):
    """Where a counting pattern hid lost state, prose has to show it."""
    legs = _Legs(_whole([*_TEXTS, " c", " d"]))
    result = _witness(tmp_path, legs, prompt_tokens=[5, 40])
    assert result.passed, result.detail
    assert _handed(legs.calls[2]) == [natural_text(40)]
    # The main run keeps its own file: both prompts files stay to be read.
    assert len(_handed(legs.calls[1])) == 4
    assert "continuation #3 (40 words)" in result.detail


def test_identical_output_with_the_state_destroyed_is_insensitive_not_a_pass(tmp_path):
    """The workload would pass a hop that drops the recurrent state."""
    legs = _Legs(_whole(_TEXTS), ablated=_destroyed(_TEXTS[0]))
    result = _witness(tmp_path, legs)
    assert not result.passed
    assert result.detail.startswith("INSENSITIVE, not a pass: 2 continuation(s) token-identical")
    assert "continuation #0" in result.detail and "prose" in result.detail
    assert result.command == legs.calls[2]


def test_a_leg_b_that_destroys_nothing_cannot_run_the_self_test(tmp_path):
    """A launcher that ignores the variable prints the same continuation and no report."""
    legs = _Legs(_whole(_TEXTS), ablated=_whole([" moved, but nothing says why"]))
    result = _witness(tmp_path, legs)
    assert not result.passed
    assert "the self-test could not run" in result.detail
    assert "`Witness ablated: <what>`" in result.detail


def test_a_self_test_leg_that_dies_is_not_a_pass(tmp_path):
    legs = _Legs(_whole(_TEXTS), ok=False, ablated="Traceback: boom\n")
    result = _witness(tmp_path, legs)
    assert not result.passed
    assert "did not complete" in result.detail and _ABLATE in result.detail
    assert "boom" in result.stdout_tail


def test_a_self_test_leg_must_print_its_one_continuation(tmp_path):
    legs = _Legs(_whole(_TEXTS), ablated=_whole(_TEXTS[::-1]) + "Witness ablated: conv\n")
    result = _witness(tmp_path, legs)
    assert not result.passed
    assert "printed 2 `Witness token ids:` line(s) for its one prompt" in result.detail


@pytest.mark.parametrize(
    "dis",
    [
        _whole([_TEXTS[0], " a different answer"]),
        _whole(_TEXTS).replace("Witness token ids", "Witness ids"),
    ],
    ids=["difference", "no-token-ids"],
)
def test_no_self_test_runs_when_the_comparison_already_fails(tmp_path, dis):
    legs = _Legs(_whole(_TEXTS), dis)
    result = _witness(tmp_path, legs)
    assert not result.passed
    assert len(legs.calls) == 2


def test_the_one_switch_shape_runs_the_self_test_in_its_second_leg():
    legs = _Legs(_whole([*_TEXTS, " c"]))
    result = run_equivalence_witness(
        _env(), task_data=_TASK, contrast_env={"X": "1"}, runner=legs, prompt_tokens=[9]
    )
    assert result.passed, result.detail
    ablated = legs.calls[2]
    assert _assignment(ablated, "X") == "1" and _ABLATE in ablated
    assert ablated.endswith(f"--prompt {shlex.quote(natural_text(9))}")


def test_what_leg_b_destroyed_is_read_behind_any_prefix_and_named_once():
    out = (
        "[gen] Witness ablated: conv, ssm\n"
        "1: [gen] Witness ablated: conv, ssm\n"
        "[gen] Witness ablated: kv pages\n"
    )
    assert parse_witness_ablations(out) == ("conv, ssm", "kv pages")
    assert parse_witness_ablations("nothing") == ()


# ---------------------------------------------------------------- the ranks


def test_rank_pairs_put_every_rank_on_both_sides_first():
    assert rank_pairs(1) == ((0, 0),)
    assert rank_pairs(2) == ((0, 0), (1, 1), (0, 1), (1, 0))
    four = rank_pairs(4)
    assert len(set(four)) == 16
    assert four[:4] == ((0, 0), (1, 1), (2, 2), (3, 3))


@pytest.mark.parametrize(
    ("parallel", "member", "ranks"),
    [("tp1", "dep4", 4), ("tp1", "tep4", 1), ("tp1", "tp2", 1), ("dep2", None, 2)],
)
def test_only_a_dep_member_has_attention_dp_ranks(parallel, member, ranks):
    assert attention_dp_size({**_TASK, "parallel": parallel}, _env(4), member) == ranks


def test_at_dep2_every_rank_pair_is_pinned_and_reported(tmp_path):
    """Two boot prompts become four hops: 0:0, 1:1, 0:1, 1:0."""
    pairs = rank_pairs(2)
    texts = [*_TEXTS, *_TEXTS]
    legs = _Legs(_whole(texts), _whole(texts, ranks=pairs))
    result = _witness(tmp_path, legs, environment=_env(2), member="dep2")
    assert result.passed, result.detail
    agg, dis, ablated = legs.calls
    assert "--enable_attention_dp" in agg
    assert _assignment(dis, LEG_B_RANK_PAIRS_ENV) == "0:0,1:1,0:1,1:0"
    # Both legs run the repeated workload, one prompt per pair.
    handed = _handed(dis)
    assert handed == [*gates.BootSpec().prompts] * 2
    for prompt in handed:
        assert shlex.quote(prompt) in agg
    assert "every attention-DP rank pair crossed (4 pairs over 4 hops)" in result.detail
    # The self-test's hop goes through the pair its prompt had.
    assert _assignment(ablated, LEG_B_RANK_PAIRS_ENV) == "0:0"


def test_a_hop_leg_b_does_not_report_pinned_fails(tmp_path):
    texts = [*_TEXTS, *_TEXTS]
    reported = _whole(texts, ranks=rank_pairs(2)).replace("[3] Witness ranks: ctx=1 gen=0\n", "")
    legs = _Legs(_whole(texts), reported)
    result = _witness(tmp_path, legs, environment=_env(2), member="dep2")
    assert not result.passed
    assert "no `[i] Witness ranks:` line for continuation(s) [3]" in result.detail
    assert "from context rank 0 to generation rank 0" in result.detail
    assert len(legs.calls) == 2


def test_a_launcher_that_ignores_the_pairs_fails_on_every_unpinned_hop(tmp_path):
    texts = [*_TEXTS, *_TEXTS]
    legs = _Legs(_whole(texts), _whole(texts, ranks=[(0, 0)] * 4))
    result = _witness(tmp_path, legs, environment=_env(2), member="dep2")
    assert not result.passed
    assert "#2 reported ctx=0 gen=0 for 0:1" in result.detail
    assert "#3 reported ctx=0 gen=0 for 1:0" in result.detail


def test_more_prompts_than_pairs_cycle_the_pairs(tmp_path):
    prompts = [f"prompt number {word}" for word in ("one", "two", "three", "four", "five")]
    texts = [f" answer {i}" for i in range(5)]
    hops = [rank_pairs(2)[i % 4] for i in range(5)]
    legs = _Legs(_whole(texts), _whole(texts, ranks=hops))
    result = _witness(tmp_path, legs, environment=_env(2), member="dep2", prompts=prompts)
    assert result.passed, result.detail
    assert _assignment(legs.calls[1], LEG_B_RANK_PAIRS_ENV) == "0:0,1:1,0:1,1:0,0:0"
    assert _handed(legs.calls[1]) == prompts


def test_dep4_crosses_all_sixteen_pairs(tmp_path):
    pairs = rank_pairs(4)
    texts = [_TEXTS[i % 2] for i in range(16)]
    legs = _Legs(_whole(texts), _whole(texts, ranks=pairs))
    result = _witness(tmp_path, legs, environment=_env(4), member="dep4")
    assert result.passed, result.detail
    handed = _assignment(legs.calls[1], LEG_B_RANK_PAIRS_ENV).split(",")
    assert len(handed) == len(set(handed)) == 16
    assert len(_handed(legs.calls[1])) == 16


@pytest.mark.parametrize("member", ["tep2", "tp2"])
def test_without_attention_dp_nothing_is_pinned(tmp_path, member):
    legs = _Legs(_whole(_TEXTS))
    result = _witness(tmp_path, legs, environment=_env(2), member=member)
    assert result.passed, result.detail
    assert all(LEG_B_RANK_PAIRS_ENV not in call for call in legs.calls)
    assert "rank pair" not in result.detail


def test_the_one_switch_shape_pins_nothing_at_a_dep_member():
    """Only a launcher places a hop on a rank."""
    legs = _Legs(_whole(_TEXTS))
    result = run_equivalence_witness(
        _env(2), task_data=_TASK, contrast_env={"X": "1"}, runner=legs, member="dep2"
    )
    assert result.passed, result.detail
    assert all(LEG_B_RANK_PAIRS_ENV not in call for call in legs.calls)


def test_reported_ranks_are_read_per_prompt():
    out = "[0] Witness ranks: ctx=1 gen=0\n0: [2] Witness ranks: ctx=0, gen=1\n"
    assert parse_witness_ranks(out) == {0: (1, 0), 2: (0, 1)}

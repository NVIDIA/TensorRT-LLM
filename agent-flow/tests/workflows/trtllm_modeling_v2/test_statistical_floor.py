# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The statistical accuracy floor, the cross-member finding, and the statistics behind them."""

from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import gates, stats
from agent_flow.workflows.trtllm_modeling_v2.checks.gate_records import (
    KIND_FINDING,
    GateRecord,
    GateRecords,
    open_findings,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import resolve_environment
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import (
    TaskSchemaError,
    load_and_validate_task_yaml,
)

# A GSM8K anchor: 92.30 over 1319 questions.
_ANCHOR = 92.30
_TASK = {
    "checkpoint_path": "/ckpt/M",
    "parallel": "tep4",
    "trtllm_repo_path": "/repo",
    "accuracy": {"benchmark": "gsm8k", "reference_yaml": "r.yaml", "tolerance": 5.0},
}


def _env():
    return resolve_environment(
        {"environment": {"kind": "local", "python": "/py", "env": {}}}, devices=(0,), world_size=1
    )


def _eval(score: float, extra: str = ""):
    def run(command, env=None, timeout=0):
        run.seen.append(command)
        return CommandResult(
            returncode=0, stdout=f"{extra}lm-eval gsm8k average accuracy: {score}\n"
        )

    run.seen = []
    return run


# ------------------------------------------------------------------ the numbers


def test_the_gsm8k_standard_error_is_three_quarters_of_a_point():
    """Anchor - 2 SE is ~1.5 points for GSM8K-1319, where anchor - 5 is ~7 SE."""
    se = stats.binomial_se(_ANCHOR, 1319)
    assert se == pytest.approx(0.734, abs=1e-3)
    floor = stats.statistical_floor(_ANCHOR, 1319)
    assert _ANCHOR - floor.value == pytest.approx(1.47, abs=0.01)
    assert (5.0 / se) > 6.5


def test_a_perfect_score_still_has_a_nonzero_error():
    assert stats.binomial_se(100.0, 50) > 0


def test_student_t_matches_tabulated_values():
    assert stats.student_t_cdf(-2.262, 9) == pytest.approx(0.025, abs=2e-4)
    assert stats.student_t_cdf(2.0, 10) == pytest.approx(0.9633, abs=2e-4)
    assert stats.student_t_cdf(0.0, 5) == pytest.approx(0.5)
    assert stats.student_t_cdf(-1.4, 31) == pytest.approx(0.0857, abs=5e-4)


def test_the_wilcoxon_p_is_exact_for_small_samples():
    """Six drops out of six: the one sign assignment out of 64 at least this extreme."""
    result = stats.paired_test([1, 2, 3, 4, 5, 6], [0.5, 1.6, 2.9, 3.2, 4.8, 5.1])
    assert result.wilcoxon_p == pytest.approx(1 / 64)
    assert result.down == 6 and result.up == 0


def test_a_large_sample_uses_the_normal_approximation_and_agrees_in_direction():
    baseline = [float(i % 7) for i in range(100)]
    candidate = [b - 0.3 for b in baseline]
    result = stats.paired_test(baseline, candidate, test=stats.TEST_WILCOXON, min_effect=0.1)
    assert result.n > stats.WILCOXON_EXACT_MAX_N
    assert result.wilcoxon_p < 1e-6 and result.regressed


# ----------------------------------------------- the paired acceptance comparison


def test_the_f051_change_is_not_a_regression_on_32_prompts():
    """A change that only moves generated text: a small drop at paired t ~-1.4 is noise."""
    baseline = [2.6 + 0.4 * math.sin(i) for i in range(32)]
    candidate = [b - 0.06 + 0.22 * math.cos(3 * i) for i, b in enumerate(baseline)]
    result = stats.paired_test(baseline, candidate, min_effect=0.1)
    assert -2.0 < result.t < 0
    assert not result.regressed


def test_four_prompts_cannot_carry_a_finding():
    """A fixed 0.1 delta flags this four-prompt drop; the paired test cannot reach significance."""
    result = stats.paired_test([2.08, 4.00, 4.06, 1.92], [1.80, 3.95, 4.10, 1.85], min_effect=0.1)
    assert not result.significant


def test_a_real_drop_is_both_significant_and_large():
    baseline = [2.6 + 0.3 * math.sin(i) for i in range(32)]
    candidate = [b - 0.5 + 0.05 * math.cos(i) for i, b in enumerate(baseline)]
    result = stats.paired_test(baseline, candidate, min_effect=0.1)
    assert result.significant and result.regressed
    assert "paired over 32 prompt(s)" in result.describe()


def test_significance_alone_is_not_a_regression():
    """A tiny but consistent drop on a large workload is below the minimum effect."""
    baseline = [3.0 + 0.1 * (i % 5) for i in range(200)]
    candidate = [b - 0.01 for b in baseline]
    result = stats.paired_test(baseline, candidate, min_effect=0.1)
    assert result.significant and not result.regressed


def test_mismatched_pairs_raise():
    with pytest.raises(ValueError, match="differ in length"):
        stats.paired_test([1.0], [1.0, 2.0])


# ---------------------------------------------------------- the accuracy gate


def test_the_floor_is_statistical_by_default():
    """With no rule stated the floor is anchor - 2 SE, not a fixed tolerance."""
    result = gates.run_accuracy_gate(_env(), task_data=_TASK, reference=_ANCHOR, runner=_eval(90.0))
    assert not result.passed
    assert result.floor == pytest.approx(_ANCHOR - 2 * 0.734, abs=0.01)
    assert result.se == pytest.approx(0.734, abs=1e-3) and result.samples == 1319
    assert "2.00 x SE 0.73 (n=1319)" in result.detail
    assert "SE 0.73" in result.summary() and "floor 90.83" in result.summary()


def test_a_score_inside_two_se_passes():
    result = gates.run_accuracy_gate(_env(), task_data=_TASK, reference=_ANCHOR, runner=_eval(91.2))
    assert result.passed


def test_k_is_configurable():
    task = {**_TASK, "accuracy": {**_TASK["accuracy"], "floor": {"k": 4.0}}}
    result = gates.run_accuracy_gate(_env(), task_data=task, reference=_ANCHOR, runner=_eval(90.0))
    assert result.passed and "4.00 x SE" in result.detail


def test_the_tolerance_rule_is_still_available_explicitly():
    task = {**_TASK, "accuracy": {**_TASK["accuracy"], "floor": {"rule": "tolerance"}}}
    result = gates.run_accuracy_gate(_env(), task_data=task, reference=_ANCHOR, runner=_eval(90.0))
    assert result.passed and result.floor == pytest.approx(87.30)
    assert "fixed tolerance" in result.floor_rule and result.se is not None


def test_the_count_the_eval_printed_wins():
    """A run limited by --num_samples scores fewer questions, and its SE is wider."""
    out = "Partial scores after 200/200 responses (estimate, 0~100): x\n"
    result = gates.run_accuracy_gate(
        _env(), task_data=_TASK, reference=_ANCHOR, runner=_eval(89.0, extra=out)
    )
    assert result.samples == 200 and result.passed
    assert "the eval's own output" in result.detail


def test_num_samples_in_the_benchmark_args_sets_the_count():
    accuracy = {**_TASK["accuracy"], "benchmark_args": ["--num_samples=64"]}
    assert gates.resolve_samples(accuracy, None, "gsm8k")[0] == 64
    accuracy = {**_TASK["accuracy"], "benchmark_args": ["--num_samples", "48"]}
    assert gates.resolve_samples(accuracy, None, "gsm8k")[0] == 48


def test_a_declared_count_is_used_for_an_unknown_benchmark():
    accuracy = {"samples": {"default": 500, "multimodal": 900}}
    assert gates.resolve_samples(accuracy, "core", "longbench")[0] == 500
    assert gates.resolve_samples(accuracy, "multimodal", "custom")[0] == 900
    assert gates.resolve_samples({}, None, "longbench")[0] is None


def test_an_unknown_count_fails_rather_than_guessing():
    task = {**_TASK, "accuracy": {**_TASK["accuracy"], "benchmark": "longbench"}}
    result = gates.run_accuracy_gate(_env(), task_data=task, reference=50.0, runner=_eval(49.0))
    assert not result.passed and "declare `accuracy.samples`" in result.detail


# ----------------------------------------------------- the cross-member finding


def _verdict(member, measured, reference=_ANCHOR, se=0.734):
    return gates.GateResult(
        name="accuracy[gsm8k]",
        passed=True,
        detail="",
        member=member,
        measured=measured,
        reference=reference,
        se=se,
    )


def test_every_member_below_one_se_is_a_finding():
    results = [_verdict("tep4", 91.4), _verdict("dep4", 91.5), _verdict("tp4", 91.2)]
    finding = gates.check_systematic_shift(results, _TASK["accuracy"], "dspark")
    assert finding is not None and not finding.acknowledged
    assert "every member is below the anchor" in finding.describe()
    assert "accuracy.floor.acknowledge.dspark" in finding.describe()


def test_one_member_within_one_se_is_not():
    results = [_verdict("tep4", 91.4), _verdict("dep4", 92.0), _verdict("tp4", 91.2)]
    assert gates.check_systematic_shift(results, _TASK["accuracy"], "dspark") is None


def test_a_single_member_unit_gets_no_cross_member_check_by_default():
    assert gates.check_systematic_shift([_verdict("tp1", 80.0)], _TASK["accuracy"], "core") is None
    accuracy = {**_TASK["accuracy"], "floor": {"shift_min_members": 1}}
    assert gates.check_systematic_shift([_verdict("tp1", 90.0)], accuracy, "core") is not None


def test_an_acknowledged_shift_is_recorded_with_its_reason():
    accuracy = {**_TASK["accuracy"], "floor": {"acknowledge": {"dspark": "drafter rounding"}}}
    results = [_verdict("tep4", 91.4), _verdict("tp4", 91.2)]
    finding = gates.check_systematic_shift(results, accuracy, "dspark")
    assert finding.acknowledged == "drafter rounding"


def test_a_finding_blocks_delivery_until_acknowledged_or_resolved(tmp_path):
    records = gates.records_for(tmp_path)
    shifted = [_verdict("tep4", 91.4), _verdict("tp4", 91.2)]
    gates.note_systematic_shift(records, shifted, _TASK["accuracy"], "dspark")
    blocking = gates.blocking_findings(records, _TASK["accuracy"])
    assert len(blocking) == 1 and blocking[0].key == {
        "finding": "systematic_shift",
        "unit": "dspark",
    }
    # Acknowledged in the CURRENT task: no longer blocking.
    acknowledged = {**_TASK["accuracy"], "floor": {"acknowledge": {"dspark": "measured noise"}}}
    assert gates.blocking_findings(records, acknowledged) == []
    # A later clean gating of the same unit resolves it.
    clean = [_verdict("tep4", 92.4), _verdict("tp4", 91.2)]
    gates.note_systematic_shift(records, clean, _TASK["accuracy"], "dspark")
    assert gates.blocking_findings(records, _TASK["accuracy"]) == []


# ---------------------------------------------------------------- the records


def test_the_baseline_is_the_newest_passing_record_of_another_tree(tmp_path):
    records = GateRecords(tmp_path / "r.jsonl")
    key = {"unit": "core", "member": "tp1"}
    records.append(GateRecord(kind="perf", key=key, tree="a", passed=True, values={"v": 1}))
    records.append(GateRecord(kind="perf", key=key, tree="b", passed=False, values={"v": 2}))
    records.append(GateRecord(kind="perf", key={"unit": "x"}, tree="c", passed=True))
    assert records.latest("perf", key).tree == "a"
    assert records.latest("perf", key, passed_only=False).tree == "b"
    assert records.latest("perf", key, exclude_tree="a") is None
    assert records.all("perf")[0].recorded_at


def test_a_torn_line_is_skipped(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text(
        '{"kind": "finding", "key": {"unit": "u"}, "tree": "t", "passed": false}\n{"kind', "utf-8"
    )
    assert len(GateRecords(path).all()) == 1
    assert [r.kind for r in open_findings(GateRecords(path))] == [KIND_FINDING]


# ---------------------------------------------------------------- the schema


def _spec(tmp_path, accuracy):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(exist_ok=True)
    data = {
        "checkpoint_path": str(ckpt),
        "trtllm_repo_path": str(repo),
        "arch": "sm_100",
        "parallel": "tp1",
        "environment": {"kind": "local", "python": "/usr/bin/python3"},
        "accuracy": {"reference_yaml": "r.yaml", "model_key": "M", **accuracy},
    }
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_the_schema_accepts_a_floor_block(tmp_path):
    accuracy = {
        "benchmark": "gsm8k",
        "floor": {"rule": "statistical", "k": 2.5, "acknowledge": {"core": "why"}},
    }
    loaded = load_and_validate_task_yaml(_spec(tmp_path, accuracy))
    assert loaded["accuracy"]["floor"]["k"] == 2.5


@pytest.mark.parametrize(
    "accuracy,message",
    [
        ({"benchmark": "gsm8k", "floor": {"rule": "loose"}}, "accuracy.floor.rule"),
        ({"benchmark": "gsm8k", "floor": {"k": 0}}, "accuracy.floor.k"),
        ({"benchmark": "gsm8k", "floor": {"acknowledge": {"core": ""}}}, "acknowledge"),
        ({"benchmark": "gsm8k", "floor": {"width": 1}}, "width"),
        ({"benchmark": "gsm8k", "samples": 0}, "accuracy.samples"),
        ({"benchmark": "longbench"}, "declare `accuracy.samples`"),
    ],
)
def test_the_schema_refuses_a_floor_it_cannot_compute(tmp_path, accuracy, message):
    with pytest.raises(TaskSchemaError, match=message.replace(".", r"\.")):
        load_and_validate_task_yaml(_spec(tmp_path, accuracy))


def test_an_unknown_benchmark_is_fine_under_the_tolerance_rule_or_with_a_count(tmp_path):
    load_and_validate_task_yaml(
        _spec(tmp_path, {"benchmark": "longbench", "floor": {"rule": "tolerance"}})
    )
    load_and_validate_task_yaml(_spec(tmp_path, {"benchmark": "longbench", "samples": 503}))


# ---------------------------------------------------------------- the wiring


def test_the_planned_run_checks_the_members_together_after_accuracy():
    from agent_flow.workflows.trtllm_modeling_v2.run.stages.planned import PlannedStage

    assert "_note_shift" in inspect.getsource(PlannedStage._run_contract_check)
    assert "note_systematic_shift" in inspect.getsource(PlannedStage._note_shift)


def test_delivery_refuses_open_findings_first(tmp_path):
    """Before the delivery report, and as a remediable verdict a relaunch re-enters."""
    from .test_workflow import _workflow

    wf = _workflow(tmp_path)
    wf.task_data = {"accuracy": dict(_TASK["accuracy"])}
    gates.note_systematic_shift(
        gates.records_for(wf.workspace),
        [_verdict("tep4", 91.0), _verdict("tp4", 91.1)],
        wf.task_data["accuracy"],
        "dspark",
    )
    closed = []
    wf._close_run = lambda state, log, verdict: closed.append(verdict)
    wf._delivery_report = lambda state: pytest.fail("the report must not run")
    assert wf._run_deliver_phase(SimpleNamespace(units_done=["core"]), None) is False
    assert closed == ["delivery-failed:findings"]
    assert "delivery blocked by gate findings" in wf.journal.path.read_text(encoding="utf-8")

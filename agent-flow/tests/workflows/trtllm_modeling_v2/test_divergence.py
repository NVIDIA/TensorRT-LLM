# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A divergence from the reference is measured and recorded, and never fails a gate by itself."""

from __future__ import annotations

import json
import math

import pytest

from agent_flow.workflows.trtllm_modeling_v2.checks import boot as boot_checks
from agent_flow.workflows.trtllm_modeling_v2.checks import divergence
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import workload_io
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import (
    TaskSchemaError,
    _validate_accuracy,
)

from .test_gates import _TASK, _env, recorder

PARIS = {11: -0.05, 12: -3.2, 13: -4.0}


def test_identical_distributions_are_within_every_bound():
    found = divergence.assess({"p0": PARIS}, {"p0": dict(PARIS)}, divergence.Bounds())
    assert found.within and found.top1 == 1.0
    assert found.max_js == pytest.approx(0.0, abs=1e-9)
    assert found.min_cosine == pytest.approx(1.0)


def test_a_flip_inside_the_tie_margin_agrees_and_one_outside_it_is_recorded():
    tied = {11: -0.70, 12: -0.75}
    swapped = {11: -0.75, 12: -0.70}
    assert divergence.first_token_agrees(tied, swapped, 0.15)
    found = divergence.assess({"p0": PARIS}, {"p0": {12: -0.05, 11: -3.2, 13: -4.0}},
                              divergence.Bounds())  # fmt: skip
    assert not found.within and found.top1 == 0.0
    assert "first token differs" in found.crossed[0]
    assert 0 < found.max_js <= math.log(2)
    assert "divergence recorded" in found.describe()


def test_the_bounds_are_the_runs_to_loosen():
    loose = divergence.Bounds.from_block({"max_js": 0.7, "min_cosine": -1, "tie_margin": 5})
    found = divergence.assess({"p0": PARIS}, {"p0": {12: -0.05, 11: -3.2, 13: -4.0}}, loose)
    assert found.within


def _reference(tmp_path, first_tokens):
    path = tmp_path / "reference.json"
    path.write_text(json.dumps({
        "source": "transformers test", "top_k": 3,
        "prompts": {"p0": "The capital of France is"},
        "first_tokens": {"p0": {str(k): v for k, v in first_tokens.items()}},
    }))  # fmt: skip
    return {**_TASK, "accuracy": {**_TASK["accuracy"], "reference_logits": {"path": str(path)}}}


def _driver_output(texts, top):
    lines = []
    for n, text in enumerate(texts):
        record = {"index": n, "text": text, "token_ids": [1, 2], "tokens": 2, "iterations": 2}
        lines.append(f"{workload_io.RECORD_MARKER} {json.dumps(record)}")
    record = {"index": len(texts), "text": "x", "token_ids": [1], "tokens": 1, "iterations": 1,
              "top_logprobs": [[[t, lp] for t, lp in top.items()]]}  # fmt: skip
    lines.append(f"{workload_io.RECORD_MARKER} {json.dumps(record)}")
    return "\n".join(lines) + "\n"


def test_the_boot_records_a_divergence_and_still_passes(tmp_path):
    task = _reference(tmp_path, PARIS)
    texts = ("Paris is the capital", "the way plants make food")
    run = recorder(CommandResult(returncode=0, stdout=_driver_output(texts, PARIS)))
    result = boot_checks.run_boot_gate(_env(), task_data=task, runner=run, workdir=tmp_path)
    assert result.passed and not result.warnings
    assert "against the reference, first tokens agree on 100% of 1 prompt(s)" in result.detail
    flipped = {12: -0.05, 11: -3.2, 13: -4.0}
    run = recorder(CommandResult(returncode=0, stdout=_driver_output(texts, flipped)))
    result = boot_checks.run_boot_gate(_env(), task_data=task, runner=run, workdir=tmp_path)
    assert result.passed, result.detail
    assert "divergence from the reference recorded" in result.warnings[0]
    assert "p0 (first token differs" in result.detail


def test_an_unreadable_reference_is_unknown_not_a_failure(tmp_path):
    task = {**_TASK, "accuracy": {**_TASK["accuracy"],
                                  "reference_logits": {"path": str(tmp_path / "absent.json")}}}  # fmt: skip
    texts = ("Paris is the capital", "the way plants make food")
    run = recorder(CommandResult(returncode=0, stdout=_driver_output(texts, {})[:-1]))
    result = boot_checks.run_boot_gate(_env(), task_data=task, runner=run, workdir=tmp_path)
    assert result.passed and "divergence UNKNOWN" in result.detail


def test_the_spec_names_a_reference_by_its_file_and_bounds():
    rows = {"benchmark": "gsm8k", "reference_yaml": "gsm8k.yaml"}
    errors: list[str] = []
    _validate_accuracy({**rows, "reference_logits": {"path": "/r.json", "max_js": 0.3}}, errors)
    assert not errors
    _validate_accuracy({**rows, "reference_logits": {"max_js": -1}}, errors)
    assert errors == ["'accuracy.reference_logits' must be a mapping with a `path` to the "
                      "reference's file"]  # fmt: skip
    with pytest.raises(TaskSchemaError):
        _validate_accuracy([], [])

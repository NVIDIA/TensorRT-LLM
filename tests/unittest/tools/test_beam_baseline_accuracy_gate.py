# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the baseline gate's launch and verdict wiring without a model server."""

import importlib.util
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.cpu_only
GATE = (
    Path(__file__).resolve().parents[2]
    / "integration/defs/accuracy/modeling_v2/beam/full_model_nvfp4__sm_103__tp4/test_accuracy.py"
)
spec = importlib.util.spec_from_file_location("beam_baseline_accuracy_gate", GATE)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


@pytest.mark.parametrize("passed", [True, False])
@pytest.mark.parametrize("timeout,budget", [(3600, 2400), (2700, 2400), (600, 300)])
def test_gate_uses_baseline_and_propagates_public_verdict(
    tmp_path, monkeypatch, passed, timeout, budget
):
    root = tmp_path / "models"
    (root / "target").mkdir(parents=True)
    monkeypatch.setenv("TRTLLM_TEST_TARGET_RELATIVE_PATH", "target")
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return {"verdict": {"passed": passed, "problems": [] if passed else ["accuracy"]}}

    monkeypatch.setattr(gate.runner, "run", run)
    if passed:
        gate._run_first100(root, tmp_path / "results", timeout)
    else:
        with pytest.raises(AssertionError, match="accuracy"):
            gate._run_first100(root, tmp_path / "results", timeout)
    (call,) = calls
    assert (
        call["num_questions"],
        call["min_correct"],
        call["concurrency"],
        call["time_budget_s"],
    ) == (100, 95, 1, budget)
    managed = call["managed"]
    assert managed.argv[:3] == ("trtllm-serve", str(root / "target"), "--config")
    config = yaml.safe_load(Path(managed.argv[-1]).read_text())
    assert config == managed.settings == call["identity_extra"]["server_config"]
    assert config["modeling_v2"] == "require"
    assert config["checkpoint_format"] == "beam_nvfp4"
    assert config["tensor_parallel_size"] == 4
    assert config["cuda_graph_config"] is None and config["disable_overlap_scheduler"]
    assert "speculative_config" not in config


@pytest.mark.parametrize("timeout", [0, 300])
def test_timeout_must_allow_cleanup(tmp_path, monkeypatch, timeout):
    monkeypatch.setattr(gate.runner, "run", lambda **kwargs: pytest.fail("unexpected launch"))
    with pytest.raises(ValueError, match="cleanup reserve"):
        gate._run_first100(tmp_path, tmp_path / "results", timeout)


@pytest.mark.parametrize("relative", ["../outside", "/outside", "missing"])
def test_missing_or_invalid_checkpoint_fails_before_launch(tmp_path, monkeypatch, relative):
    monkeypatch.setenv("TRTLLM_TEST_TARGET_RELATIVE_PATH", relative)
    monkeypatch.setattr(gate.runner, "run", lambda **kwargs: pytest.fail("unexpected launch"))
    with pytest.raises(ValueError):
        gate._run_first100(tmp_path, tmp_path / "results", 2700)

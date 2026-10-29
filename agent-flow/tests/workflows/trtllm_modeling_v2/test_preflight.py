# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight's refusals: the checks that decide whether the run happens.

Every probe is injected, so none of this needs a GPU.
"""

from __future__ import annotations

import json

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.infra.compute import environment
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.run.startup import preflight
from agent_flow.workflows.trtllm_modeling_v2.run.startup.preflight import PreflightError


def _smi_rows(*rows: str) -> str:
    return "\n".join(rows) + "\n"


def _healthy_row(index: int, cap: str = "10.3") -> str:
    return f"{index}, 180000, 183359, 0, {cap}"


def make_runner(*, smi: str, probe_ok=lambda index: True):
    """Build a runner that answers nvidia-smi, the lone probe, and torchrun."""

    def runner(argv, env=None, timeout=0):
        joined = " ".join(argv)
        if argv and argv[0] == "nvidia-smi":
            return CommandResult(returncode=0, stdout=smi)
        if argv and argv[0] == "torchrun":
            return CommandResult(returncode=0, stdout="[preflight] OK\n")
        if "torch.cuda.set_device" in joined:
            index = int((env or {}).get("CUDA_VISIBLE_DEVICES", "0"))
            if probe_ok(index):
                return CommandResult(returncode=0, stdout="ok\n")
            return CommandResult(returncode=1, stderr="cudaErrorDevicesUnavailable")
        raise AssertionError(f"unexpected command: {argv}")

    return runner


# A declared scratch location that is not under /tmp. It is never created:
# `check_tmpdir` reads the declared value, it does not stat it — and pytest's
# own `tmp_path` lives under /tmp on some hosts, which is exactly what the
# check rejects.
_SCRATCH = "/scratch/mv2flow-run-tmp"


def _task(tmp_path, **overrides):
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "config.json").write_text(json.dumps({"architectures": ["Qwen3MoeForCausalLM"]}))
    task = {
        "checkpoint_path": str(checkpoint),
        "arch": "sm_103",
        "parallel": "tp1",
        "trtllm_repo_path": str(tmp_path),
        "environment": {
            "kind": "local",
            "python": "/opt/venv/bin/python",
            "env": {"TMPDIR": _SCRATCH, "CLAUDE_CODE_TMPDIR": _SCRATCH},
        },
    }
    task.update(overrides)
    return task


# ---------------------------------------------------------------- checkpoint


def test_checkpoint_must_parse(tmp_path):
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{not json")
    with pytest.raises(PreflightError, match="not valid JSON"):
        preflight.check_checkpoint(checkpoint)


# ------------------------------------------------------------------- tmpdir


def test_tmpdir_must_be_set_outside_tmp_and_short_enough():
    with pytest.raises(environment.EnvironmentError, match="TMPDIR is not set"):
        environment.check_tmpdir({"CLAUDE_CODE_TMPDIR": "/scratch/x"})
    for value in ("/tmp", "/tmp/mine"):
        with pytest.raises(environment.EnvironmentError, match="points into /tmp"):
            environment.check_tmpdir({"TMPDIR": value, "CLAUDE_CODE_TMPDIR": value})
    # 61 works and 62 does not: the RPC socket's path must fit in ``sun_path``.
    environment.check_tmpdir({"TMPDIR": "/s" + "a" * 59, "CLAUDE_CODE_TMPDIR": _SCRATCH})
    with pytest.raises(environment.EnvironmentError, match="at most 61"):
        environment.check_tmpdir({"TMPDIR": "/s" + "a" * 60, "CLAUDE_CODE_TMPDIR": _SCRATCH})


# ------------------------------------------------------------------ devices


def test_claim_skips_a_card_that_fails_the_lone_probe():
    """A card that fails alone is unusable, not contended — waiting never helps."""
    runner = make_runner(
        smi=_smi_rows(_healthy_row(0), _healthy_row(1), _healthy_row(2)),
        probe_ok=lambda index: index != 0,
    )
    notes: list[str] = []
    assert preflight.claim_devices(2, "sm_103", "/p", runner=runner, notes=notes) == (1, 2)
    assert any("lone-process probe" in n for n in notes)


def test_a_claim_it_cannot_fill_says_why():
    runner = make_runner(
        smi=_smi_rows(_healthy_row(0), "1, 20000, 183359, 0, 10.3", "2, 180000, 183359, 97, 10.3")
    )
    with pytest.raises(PreflightError) as excinfo:
        preflight.claim_devices(3, "sm_103", "/p", runner=runner)
    message = str(excinfo.value)
    assert "need 3 healthy sm_103" in message
    assert "device 1" in message and "device 2" in message
    # The wrong architecture is refused with the reason.
    runner = make_runner(smi=_smi_rows(_healthy_row(0, cap="10.0"), _healthy_row(1, cap="10.0")))
    with pytest.raises(PreflightError) as excinfo:
        preflight.claim_devices(1, "sm_103", "/p", runner=runner)
    assert "sm_100" in str(excinfo.value)
    assert "another machine's certification" in str(excinfo.value)


def test_nvidia_smi_failure_is_not_silently_an_empty_claim():
    def runner(argv, env=None, timeout=0):
        return CommandResult(returncode=127, stderr="no nvidia-smi")

    with pytest.raises(PreflightError, match="nvidia-smi did not answer"):
        preflight.query_devices(runner)


# ---------------------------------------------------------------- multirank


def test_multirank_timeout_explains_the_hang(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import (
        resolve_environment,
    )

    env = resolve_environment(_task(tmp_path), devices=[0, 1], world_size=2)

    def runner(argv, env_=None, timeout=0):
        return CommandResult(returncode=124, timed_out=True)

    with pytest.raises(PreflightError) as excinfo:
        preflight.check_multirank_substrate(2, env, runner)
    assert "hangs rather than fails" in str(excinfo.value)


# ---------------------------------------------------------- accuracy anchors


_REFERENCES_DIR = "tests/integration/defs/accuracy/references"


def _reference(tmp_path, name, payload):
    """Write a reference file where `trtllm_repo_path` will resolve it."""
    path = tmp_path / _REFERENCES_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return f"{_REFERENCES_DIR}/{name}"


# -------------------------------------------------------------- end to end


def test_run_preflight_resolves_identity(tmp_path):
    task = _task(tmp_path, parallel="dep4")
    (tmp_path / "ckpt").rename(tmp_path / "Qwen3-30B-A3B")
    task["checkpoint_path"] = str(tmp_path / "Qwen3-30B-A3B")
    runner = make_runner(smi=_smi_rows(*[_healthy_row(i) for i in range(4)]))
    result = preflight.run_preflight(task, runner=runner)
    assert result.slug == "qwen3-30b-a3b-sm_103-dep4"
    assert result.branch == "trtllm-modeling-v2/qwen3-30b-a3b-sm_103-dep4"
    assert result.devices == (0, 1, 2, 3)
    assert result.world_size == 4
    assert result.checkpoint_config["architectures"] == ["Qwen3MoeForCausalLM"]


def test_a_missing_model_key_stops_the_run_before_a_single_device_is_probed(tmp_path):
    """The anchor check is pure file reading, so it runs before any probe of the host."""
    reference = _reference(tmp_path, "gsm8k.yaml", {"openai/gpt-oss-120b": [{"accuracy": 90.0}]})
    task = _task(
        tmp_path,
        accuracy={"benchmark": "gsm8k", "reference_yaml": reference, "model_key": "XiaomiMiMo/x"},
    )

    def runner(argv, env=None, timeout=0):  # pragma: no cover - must not run
        raise AssertionError(f"probed the host before checking the spec: {argv}")

    with pytest.raises(PreflightError) as excinfo:
        preflight.run_preflight(task, runner=runner)
    message = str(excinfo.value)
    assert "unit `core`" in message
    assert "XiaomiMiMo/x" in message
    assert reference in message
    # `load_reference_score`'s own words, surfaced rather than flattened to
    # "not found" -- including the instruction that the anchor is never
    # invented, which is the part that says what to do next.
    assert "has no entry for 'XiaomiMiMo/x'" in message
    assert "never invented" in message


# --------------------------------------------------------------- a placeholder anchor


def test_an_anchor_is_in_percent_and_committed_where_the_gates_read_it(tmp_path):
    """Anchors are in percent: 1.0 under a floor of reference - k*SE passes any engine."""
    import subprocess

    reference = _reference(tmp_path, "gsm8k.yaml", {"x/y": [{"accuracy": 1.0}]})
    task = _task(
        tmp_path, accuracy={"benchmark": "gsm8k", "reference_yaml": reference, "model_key": "x/y"}
    )
    with pytest.raises(PreflightError, match="reads as a fraction"):
        preflight.check_accuracy_anchors(task)
    # A low placeholder is an anchor; uncommitted, no gate (each measures a commit) reads it.
    _reference(tmp_path, "gsm8k.yaml", {"x/y": [{"accuracy": 10.0}]})
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    with pytest.raises(PreflightError, match="commit it to the run's branch"):
        preflight.check_accuracy_anchors(task)
    subprocess.run([*git, "add", reference], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "anchor"], check=True)
    assert "= 10" in preflight.check_accuracy_anchors(task)[0]


def test_a_relative_variant_needs_its_units_benchmark_and_disagg_one_node(tmp_path):
    fp8 = {"llm_api_options": {"kv_cache_config": {"dtype": "fp8"}}, "members": ["dep4"]}
    task = _task(
        tmp_path,
        parallel="unified",
        members=["tp4", "dep4"],
        accuracy={"benchmark": "gsm8k", "reference_yaml": "x.yaml", "benchmark_members": "tp4"},
        variants={"kv_fp8": {**fp8, "gates": ["accuracy"]}},
    )
    with pytest.raises(PreflightError, match="`core`'s benchmark at none of dep4"):
        preflight.check_variant_anchors(task, 4)
    del task["accuracy"]["benchmark_members"]
    assert "relative, core's own score at dep4" in preflight.check_variant_anchors(task, 4)[0]
    task["features"] = ["disagg"]
    task["environment"]["gpus_per_node"] = 4
    task["accuracy"]["disagg"] = {"test": "tests/test_d.py::TestM::test_auto_dtype", "devices": 8}
    with pytest.raises(PreflightError, match="more than the 4 devices one node holds"):
        preflight.check_disagg_placement(task)

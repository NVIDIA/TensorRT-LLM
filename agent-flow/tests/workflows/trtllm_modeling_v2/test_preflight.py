# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for preflight — the checks that decide whether the run happens.

Every probe is injected, so none of this needs a GPU. That is the point:
the checks most worth getting right are the ones about broken hardware, and
those are exactly the ones a test machine cannot reproduce.
"""

from __future__ import annotations

import json
import os
import socket
import uuid
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.run import preflight
from agent_flow.workflows.trtllm_modeling_v2.run.preflight import DeviceReport, PreflightError

_SMI_ARGS = "--query-gpu=index,memory.free,memory.total,utilization.gpu,compute_cap"


def _smi_rows(*rows: str) -> str:
    return "\n".join(rows) + "\n"


def _healthy_row(index: int, cap: str = "10.3") -> str:
    return f"{index}, 180000, 183359, 0, {cap}"


def make_runner(*, smi: str, probe_ok=lambda index: True, multirank_ok: bool = True):
    """Build a runner that answers nvidia-smi, the lone probe, and torchrun."""

    def runner(argv, env=None, timeout=0):
        joined = " ".join(argv)
        if argv and argv[0] == "nvidia-smi":
            return CommandResult(returncode=0, stdout=smi)
        if argv and argv[0] == "torchrun":
            if multirank_ok:
                return CommandResult(returncode=0, stdout="[preflight] OK\n")
            return CommandResult(returncode=1, stderr="NCCL init failed")
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


def test_missing_config_json_is_named(tmp_path):
    with pytest.raises(PreflightError, match="not a HF checkpoint dir"):
        preflight.check_checkpoint(tmp_path)


# ------------------------------------------------------------------- tmpdir


def test_tmpdir_must_be_set():
    with pytest.raises(PreflightError, match="TMPDIR is not set"):
        preflight.check_tmpdir({"CLAUDE_CODE_TMPDIR": "/scratch/x"})


def test_tmpdir_does_not_fall_back_to_the_launching_shell(monkeypatch):
    """Agents run under the declared prefix, not under the orchestrator's env.

    A run whose scratch location is inherited rather than declared is one
    nobody can reproduce, so an omitted value is an error even when the
    launching shell happens to have a perfectly good one.
    """
    monkeypatch.setenv("TMPDIR", "/scratch/inherited")
    monkeypatch.setenv("CLAUDE_CODE_TMPDIR", "/scratch/inherited")
    with pytest.raises(PreflightError, match="TMPDIR is not set"):
        preflight.check_tmpdir({})


@pytest.mark.parametrize("value", ["/tmp", "/tmp/mine", "/tmp/a/b"])
def test_tmpdir_under_slash_tmp_is_rejected(value):
    with pytest.raises(PreflightError, match="points into /tmp"):
        preflight.check_tmpdir({"TMPDIR": value, "CLAUDE_CODE_TMPDIR": value})


def test_tmpdir_elsewhere_is_accepted():
    preflight.check_tmpdir({"TMPDIR": _SCRATCH, "CLAUDE_CODE_TMPDIR": _SCRATCH})


# ------------------------------------------------------ tmpdir, by length
#
# The property that breaks first on a cluster whose storage paths are deep,
# and the one an author is least likely to suspect: the directory is on the
# right filesystem, is writable, has space, and still fails because its NAME
# is long. It surfaces as a ZeroMQ traceback naming neither the workflow nor
# the spec line, after every rank has started and loaded weights.


def test_the_bound_is_derived_from_the_two_fixed_costs():
    """Not a literal. `107 - 45 - len("/")`, and the separator is the easy miss.

    An `AF_UNIX` bind succeeds at 107 characters and fails at 108; the
    basename `rpc_test_<uuid4>` is `len("rpc_test_") + 36` == 45; and
    `os.path.join` puts a separator between them. Dropping the separator
    yields 62 and admits one `TMPDIR` that does not work.
    """
    assert preflight.MAX_TMPDIR_LEN == 61
    assert (
        preflight.MAX_TMPDIR_LEN
        == preflight.UNIX_SOCKET_PATH_MAX - preflight.RPC_SOCKET_BASENAME_LEN - 1
    )


def test_a_tmpdir_too_long_for_a_unix_socket_is_rejected():
    """A 92-character value on a deep scratch path, and the arithmetic the message carries."""
    value = "/scratch/projects/example_org/example_project/inference/users/example_user/runs/mv2flow-temp"
    assert len(value) == 92
    with pytest.raises(PreflightError) as excinfo:
        preflight.check_tmpdir({"TMPDIR": value, "CLAUDE_CODE_TMPDIR": value})
    message = str(excinfo.value)
    # The arithmetic belongs in the message: a bare "too long" sends the
    # reader to the filesystem, which is not where the problem is.
    assert "92 characters" in message
    assert "rpc_test_<uuid4>" in message
    assert "sizeof(sockaddr_un.sun_path)" in message
    assert "at most 61" in message


def test_the_boundary_is_inclusive_on_the_side_that_binds():
    """61 works and 62 does not; the next test checks the bound against a real bind."""
    preflight.check_tmpdir({"TMPDIR": "/s" + "a" * 59, "CLAUDE_CODE_TMPDIR": _SCRATCH})
    with pytest.raises(PreflightError, match="at most 61"):
        preflight.check_tmpdir({"TMPDIR": "/s" + "a" * 60, "CLAUDE_CODE_TMPDIR": _SCRATCH})


def test_the_derived_bound_agrees_with_an_actual_af_unix_bind():
    """The arithmetic checked against the kernel rather than against itself.

    This is the one measurement behind the bound that can be taken locally
    with no GPU, so there is no reason to take it on trust: bind a socket at
    exactly the budget and one character past it, under the real basename.

    ``/tmp`` on purpose — it is the shortest directory that certainly exists,
    and the length of the prefix is the whole variable here.
    """
    root = Path("/tmp")
    if not os.access(root, os.W_OK):
        pytest.skip("no writable /tmp to measure the socket path limit against")

    basename = f"rpc_test_{uuid.uuid4()}"
    assert len(basename) == preflight.RPC_SOCKET_BASENAME_LEN

    def binds(tmpdir_len: int) -> bool:
        directory = root / ("a" * (tmpdir_len - len("/tmp/")))
        directory.mkdir(exist_ok=True)
        path = directory / basename
        assert len(str(path)) == tmpdir_len + 1 + len(basename)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.bind(str(path))
            return True
        except OSError:
            return False
        finally:
            sock.close()
            path.unlink(missing_ok=True)
            directory.rmdir()

    assert binds(preflight.MAX_TMPDIR_LEN)
    assert not binds(preflight.MAX_TMPDIR_LEN + 1)


def test_the_length_bound_does_not_reach_the_harness_scratch_dir():
    """`CLAUDE_CODE_TMPDIR` binds no socket, so a socket's budget is not its.

    Holding it to one would reject a value that works, for a reason that
    would not be true of it.
    """
    long_value = "/scratch/" + "a" * 100
    assert len(long_value) > preflight.MAX_TMPDIR_LEN
    preflight.check_tmpdir({"TMPDIR": _SCRATCH, "CLAUDE_CODE_TMPDIR": long_value})


# ------------------------------------------------------------------ devices


def test_compute_cap_maps_to_the_receipt_key():
    assert DeviceReport(0, 1, 1, 0, "10.3").arch == "sm_103"
    assert DeviceReport(0, 1, 1, 0, "10.0").arch == "sm_100"
    assert DeviceReport(0, 1, 1, 0, "9.0").arch == "sm_90"


def test_a_spinning_card_is_broken_not_busy():
    """100% utilization invalidates every number taken there."""
    report = DeviceReport(0, 180000, 183359, 100, "10.3")
    assert report.looks_idle is False
    assert "broken for our purposes" in report.why_not_idle()


def test_a_card_holding_memory_is_not_claimable():
    report = DeviceReport(0, 20000, 183359, 0, "10.3")
    assert report.looks_idle is False
    assert "free" in report.why_not_idle()


def test_claim_picks_healthy_devices(tmp_path):
    runner = make_runner(smi=_smi_rows(_healthy_row(0), _healthy_row(1), _healthy_row(2)))
    assert preflight.claim_devices(2, "sm_103", "/p", runner=runner) == (0, 1)


def test_claim_skips_a_card_that_fails_the_lone_probe():
    """A card that fails alone is unusable, not contended — waiting never helps."""
    runner = make_runner(
        smi=_smi_rows(_healthy_row(0), _healthy_row(1), _healthy_row(2)),
        probe_ok=lambda index: index != 0,
    )
    notes: list[str] = []
    assert preflight.claim_devices(2, "sm_103", "/p", runner=runner, notes=notes) == (1, 2)
    assert any("lone-process probe" in n for n in notes)


def test_claim_reports_why_when_it_cannot_fill_the_world_size():
    runner = make_runner(
        smi=_smi_rows(_healthy_row(0), "1, 20000, 183359, 0, 10.3", "2, 180000, 183359, 97, 10.3")
    )
    with pytest.raises(PreflightError) as excinfo:
        preflight.claim_devices(3, "sm_103", "/p", runner=runner)
    message = str(excinfo.value)
    assert "need 3 healthy sm_103" in message
    assert "device 1" in message and "device 2" in message


def test_wrong_architecture_is_refused_with_the_reason():
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


def test_unparsable_smi_row_is_rejected():
    def runner(argv, env=None, timeout=0):
        return CommandResult(returncode=0, stdout="0, 1, 2\n")

    with pytest.raises(PreflightError, match="unparsable"):
        preflight.query_devices(runner)


# ---------------------------------------------------------------- multirank


def test_multirank_probe_is_skipped_at_world_size_one():
    def runner(argv, env=None, timeout=0):
        raise AssertionError("should not run")

    preflight.check_multirank_substrate(1, None, runner)  # type: ignore[arg-type]


def test_multirank_timeout_explains_the_hang(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.infra.environment import resolve_environment

    env = resolve_environment(_task(tmp_path), devices=[0, 1], world_size=2)

    def runner(argv, env_=None, timeout=0):
        return CommandResult(returncode=124, timed_out=True)

    with pytest.raises(PreflightError) as excinfo:
        preflight.check_multirank_substrate(2, env, runner)
    assert "hangs rather than fails" in str(excinfo.value)


# ---------------------------------------------------------- accuracy anchors
#
# A unit's reference file can lack a row for the checkpoint. Unchecked, the
# unit is assembled, booted on every member and killed at its accuracy gate
# hours later, on a spec error readable at startup. So what these tests
# actually assert is *when* it is noticed.


_REFERENCES_DIR = "tests/integration/defs/accuracy/references"


def _reference(tmp_path, name, payload):
    """Write a reference file where `trtllm_repo_path` will resolve it."""
    path = tmp_path / _REFERENCES_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return f"{_REFERENCES_DIR}/{name}"


def test_a_missing_model_key_is_named_before_anything_is_built(tmp_path):
    reference = _reference(tmp_path, "gsm8k.yaml", {"openai/gpt-oss-120b": [{"accuracy": 90.0}]})
    task = _task(
        tmp_path,
        accuracy={"benchmark": "gsm8k", "reference_yaml": reference, "model_key": "XiaomiMiMo/x"},
    )
    with pytest.raises(PreflightError) as excinfo:
        preflight.check_accuracy_anchors(task)
    message = str(excinfo.value)
    assert "unit `core`" in message
    assert "XiaomiMiMo/x" in message
    assert reference in message
    # `load_reference_score`'s own words, surfaced rather than flattened to
    # "not found" -- including the instruction that the anchor is never
    # invented, which is the part that says what to do next.
    assert "has no entry for 'XiaomiMiMo/x'" in message
    assert "never invented" in message


def test_a_model_with_no_matching_row_lists_the_rows_that_are_there(tmp_path):
    """The discriminator case, which is the one a spec gets subtly wrong.

    A model key that is present reads as "the anchor exists", and the run
    then measures against a row that does not describe this checkpoint --
    or, here, against nothing at all.
    """
    reference = _reference(
        tmp_path,
        "gsm8k.yaml",
        {"openai/gpt-oss-120b": [{"quant_algo": "FP8", "accuracy": 90.0}]},
    )
    task = _task(
        tmp_path,
        accuracy={
            "benchmark": "gsm8k",
            "reference_yaml": reference,
            "model_key": "openai/gpt-oss-120b",
            "discriminators": {"quant_algo": "MIXED_PRECISION"},
        },
    )
    with pytest.raises(PreflightError) as excinfo:
        preflight.check_accuracy_anchors(task)
    message = str(excinfo.value)
    assert "discriminators: {'quant_algo': 'MIXED_PRECISION'}" in message
    # What was searched for AND what is available, because the gap between
    # the two is the edit the author has to make.
    assert "has no entry matching {'quant_algo': 'MIXED_PRECISION'}" in message
    assert "Available discriminator sets: [{'quant_algo': 'FP8'}]" in message


def test_every_unit_is_checked_not_just_core(tmp_path):
    """A valid multi-unit spec resolves, and each unit is reported.

    `benchmark`, `reference_yaml` and `discriminators` are all per-unit
    here, the shape a spec with a multimodal increment takes.
    """
    gsm8k = _reference(
        tmp_path,
        "gsm8k.yaml",
        {"XiaomiMiMo/MiMo-V2.6-Flash": [{"quant_algo": "MIXED_PRECISION", "accuracy": 91.2}]},
    )
    mmmu = _reference(
        tmp_path,
        "mmmu.yaml",
        {"XiaomiMiMo/MiMo-V2.6-Flash": [{"quant_algo": "MIXED_PRECISION", "accuracy": 64.8}]},
    )
    task = _task(
        tmp_path,
        features=["multimodal"],
        accuracy={
            "benchmark": {"default": "gsm8k", "multimodal": "mmmu"},
            "reference_yaml": {"default": gsm8k, "multimodal": mmmu},
            "model_key": "XiaomiMiMo/MiMo-V2.6-Flash",
            "discriminators": {"quant_algo": "MIXED_PRECISION"},
        },
    )
    notes = preflight.check_accuracy_anchors(task)
    assert len(notes) == 2
    assert "accuracy anchor [core]: gsm8k XiaomiMiMo/MiMo-V2.6-Flash" in notes[0]
    assert "= 91.2" in notes[0]
    assert "accuracy anchor [multimodal]: mmmu XiaomiMiMo/MiMo-V2.6-Flash" in notes[1]
    assert "= 64.8" in notes[1]


def test_the_unit_that_differs_is_the_one_that_fails(tmp_path):
    """Core resolving says nothing about the increment, and vice versa.

    The failure this guards: the text anchor resolves while the multimodal
    one -- a different benchmark, in a different file -- does not.
    """
    gsm8k = _reference(tmp_path, "gsm8k.yaml", {"XiaomiMiMo/M": [{"accuracy": 91.2}]})
    mmmu = _reference(tmp_path, "mmmu.yaml", {"some/other-model": [{"accuracy": 60.0}]})
    task = _task(
        tmp_path,
        features=["multimodal"],
        accuracy={
            "benchmark": {"default": "gsm8k", "multimodal": "mmmu"},
            "reference_yaml": {"default": gsm8k, "multimodal": mmmu},
            "model_key": "XiaomiMiMo/M",
        },
    )
    with pytest.raises(PreflightError, match="unit `multimodal`"):
        preflight.check_accuracy_anchors(task)


def test_a_unit_with_no_accuracy_block_is_skipped(tmp_path):
    """Two spellings of "there is nothing to check", neither a failure."""
    assert preflight.check_accuracy_anchors(_task(tmp_path)) == []

    task = _task(
        tmp_path,
        features=["multimodal"],
        accuracy={"benchmark": "gsm8k", "reference_yaml": {"core": "refs/gsm8k.yaml"}},
    )
    notes = preflight.check_accuracy_anchors(task)
    assert "no reference_yaml for this unit" in notes[1]


def test_a_reference_file_this_checkout_does_not_have_is_not_a_failure(tmp_path):
    """The check owns the ROW. A file it cannot read tells it nothing.

    A sparse checkout, or a path that resolves outside the repo entirely,
    must not stop a run -- but the note has to state the weaker claim rather
    than read as a pass.
    """
    task = _task(
        tmp_path,
        accuracy={
            "benchmark": "gsm8k",
            "reference_yaml": "/elsewhere/gsm8k.yaml",
            "model_key": "openai/gpt-oss-120b",
        },
    )
    notes = preflight.check_accuracy_anchors(task)
    assert "accuracy reference file not found" in notes[0]
    assert "unchecked here" in notes[0]


def test_the_model_key_falls_back_to_the_checkpoint_repo_name(tmp_path):
    """Because `accuracy.model_key` is optional, and the gate derives it too.

    Deriving a different key here than the gate derives would make this
    check pass on a row nothing ever reads.
    """
    task = _task(tmp_path)
    assert preflight.resolve_model_key(task) == f"{tmp_path.name}/ckpt"


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


def test_unified_segment_defaults_its_world_size(tmp_path):
    task = _task(tmp_path, parallel="unified")
    runner = make_runner(smi=_smi_rows(*[_healthy_row(i) for i in range(4)]))
    result = preflight.run_preflight(task, runner=runner)
    assert result.world_size == 4
    assert any("unified segment" in n for n in result.notes)


def test_a_bad_anchor_stops_the_run_before_a_single_device_is_probed(tmp_path):
    """Where this check sits in the order, asserted rather than described.

    It is pure file reading, so everything below it -- the device claim, the
    multi-rank probe, the substrate gate -- is work a failure here would
    have wasted.
    """
    reference = _reference(tmp_path, "gsm8k.yaml", {"someone/else": [{"accuracy": 90.0}]})
    task = _task(
        tmp_path,
        accuracy={
            "benchmark": "gsm8k",
            "reference_yaml": reference,
            "model_key": "openai/gpt-oss-120b",
        },
    )

    def runner(argv, env=None, timeout=0):  # pragma: no cover - must not run
        raise AssertionError(f"probed the host before checking the spec: {argv}")

    with pytest.raises(PreflightError, match="unit `core`"):
        preflight.run_preflight(task, runner=runner)


def test_the_resolved_anchor_reaches_the_report(tmp_path):
    reference = _reference(tmp_path, "gsm8k.yaml", {"openai/gpt-oss-120b": [{"accuracy": 90.5}]})
    task = _task(
        tmp_path,
        accuracy={
            "benchmark": "gsm8k",
            "reference_yaml": reference,
            "model_key": "openai/gpt-oss-120b",
        },
    )
    runner = make_runner(smi=_smi_rows(_healthy_row(0)))
    result = preflight.run_preflight(task, runner=runner)
    rendered = preflight.format_preflight_report(result)
    assert "accuracy anchor [core]: gsm8k openai/gpt-oss-120b {} = 90.5" in rendered


def test_format_report_carries_the_prefix(tmp_path):
    runner = make_runner(smi=_smi_rows(_healthy_row(0)))
    result = preflight.run_preflight(_task(tmp_path), runner=runner)
    rendered = preflight.format_preflight_report(result)
    assert result.environment.prefix in rendered
    assert result.slug in rendered


# ------------------------------------------------------- a pending or placeholder anchor


def test_a_pending_anchor_needs_the_logit_gate_as_its_oracle(tmp_path):
    accuracy = {"benchmark": "gsm8k", "anchor": "pending", "model_key": "x/y"}
    with pytest.raises(PreflightError, match="pending.*no logit"):
        preflight.check_accuracy_anchors(_task(tmp_path, accuracy=accuracy))
    accuracy["logits"] = {"reference": "logits.reference.json.gz"}
    notes = preflight.check_accuracy_anchors(_task(tmp_path, accuracy=accuracy))
    assert "pending; the logit gate is the oracle" in notes[0]


def test_an_anchor_that_reads_as_a_fraction_or_a_placeholder_is_refused(tmp_path):
    """Anchors are in percent: 1.0 under a floor of reference - k*SE passes any engine."""
    reference = _reference(tmp_path, "gsm8k.yaml", {"x/y": [{"accuracy": 1.0}]})
    task = _task(
        tmp_path, accuracy={"benchmark": "gsm8k", "reference_yaml": reference, "model_key": "x/y"}
    )
    with pytest.raises(PreflightError, match="fraction or a placeholder"):
        preflight.check_accuracy_anchors(task)

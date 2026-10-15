# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the substrate gates — the checks that come before any agent.

Both exist because one real onboarding spent roughly half its wall clock on
failures that had nothing to do with the model, and every one of them
surfaced mid-run inside an agent's turn.
"""

from __future__ import annotations

import pytest

from agent_flow.workflows.trtllm_modeling_v2.environment import resolve_environment
from agent_flow.workflows.trtllm_modeling_v2.gates import ShellResult
from agent_flow.workflows.trtllm_modeling_v2.preflight import (
    PreflightError,
    agent_plane_gate,
    substrate_gate,
)

from .test_workflow import driver_records

_TASK = {
    "checkpoint_path": "/ckpt",
    "trtllm_repo_path": "/repo",
    "parallel": "tp1",
    "runtime": {"cuda_graph": True, "extra_flags": ["--kv_cache_dtype fp8"]},
    "features": ["mtp"],
}


def _environment():
    return resolve_environment(
        {"environment": {"kind": "local", "python": "/py", "env": {"PATH": "/venv/bin"}}},
        world_size=1,
    )


def _runner(result, sink=None):
    def run(command, env=None, timeout=0):
        if sink is not None:
            sink.append((command, timeout))
        return result

    return run


def _scripted(*results, sink=None):
    """A runner that answers each call with the next result in turn.

    The fallback path runs the gate twice — the run's own checkpoint, then
    the proxy — so a single fixed answer cannot tell the two apart.
    """
    remaining = list(results)

    def run(command, env=None, timeout=0):
        if sink is not None:
            sink.append((command, timeout))
        return remaining.pop(0) if remaining else results[-1]

    return run


#: What the boot gate's engine driver prints for a host that generates: one
#: structured record per prompt.
_GENERATED = driver_records()

#: What a stock path that has never heard of this architecture actually
#: prints. Quoted from the raise sites named in
#: `preflight.UNSUPPORTED_ARCHITECTURE_MARKERS`.
_UNKNOWN_ARCH = "ValueError: Unknown architecture for AutoModelForCausalLM: MiMoV2ForCausalLM"


# ------------------------------------------------------------ GPU plane


def test_a_host_that_generates_passes():
    notes: list[str] = []
    substrate_gate(
        _TASK,
        _environment(),
        shell_runner=_runner(ShellResult(returncode=0, stdout=_GENERATED)),
        notes=notes,
    )
    assert any("substrate gate" in n for n in notes)


def test_a_host_that_cannot_generate_fails_before_any_agent():
    runner = _runner(ShellResult(returncode=1, stdout="MPI_ERR_SPAWN: could not spawn processes"))
    with pytest.raises(PreflightError, match="cannot generate a token"):
        substrate_gate(_TASK, _environment(), shell_runner=runner)


def test_the_failure_says_the_target_is_not_at_fault():
    """The whole point: these errors read as model defects and are not."""
    runner = _runner(ShellResult(returncode=1, stdout='could not open source file "cuda.h"'))
    with pytest.raises(PreflightError) as excinfo:
        substrate_gate(_TASK, _environment(), shell_runner=runner)
    message = str(excinfo.value)
    assert "no target exists yet" in message
    # And it must carry the command, or the operator cannot reproduce it.
    assert "quickstart_advanced.py" in message


def test_it_runs_the_stock_path_not_the_target():
    """There is no target at preflight time, so `require` would always fail."""
    sink: list[tuple[str, int]] = []
    substrate_gate(
        _TASK,
        _environment(),
        shell_runner=_runner(ShellResult(returncode=0, stdout=_GENERATED), sink),
    )
    command = sink[0][0]
    assert "TRTLLM_MODELING_V2=off" in command
    assert "TRTLLM_MODELING_V2=require" not in command


def test_it_carries_the_runs_own_runtime_block():
    """Each of the three motivating failures depended on these flags."""
    sink: list[tuple[str, int]] = []
    substrate_gate(
        _TASK,
        _environment(),
        shell_runner=_runner(ShellResult(returncode=0, stdout=_GENERATED), sink),
    )
    command = sink[0][0]
    assert "--use_cuda_graph" in command
    assert "--kv_cache_dtype fp8" in command
    assert "PATH=/venv/bin" in command


def test_it_does_not_carry_any_feature_flags():
    """No unit is assembled yet, so a speculator flag would be a lie."""
    sink: list[tuple[str, int]] = []
    substrate_gate(
        _TASK,
        _environment(),
        shell_runner=_runner(ShellResult(returncode=0, stdout=_GENERATED), sink),
    )
    assert "--spec_decode_algo" not in sink[0][0]


# ------------------------------------------- the architecture that has no stock path


def _with_proxy(proxy="/proxy-ckpt"):
    return _TASK | {
        "environment": {"kind": "local", "python": "/py", "substrate_checkpoint": proxy}
    }


@pytest.mark.parametrize(
    "message",
    [
        "ValueError: Unknown architecture for AutoModelForCausalLM: MiMoV2ForCausalLM",
        "ValueError: Unknown architecture for AutoModelForMultimodalEncoder: MiMoV2ForCausalLM",
        "NotImplementedError: The given huggingface model architecture MiMoV2ForCausalLM "
        "is not supported in TRT-LLM yet",
    ],
)
def test_every_way_the_stock_path_says_it_does_not_know_the_architecture(message):
    """Matched on the message text, because the gate reads a log and not an exception.

    All three come from the two architecture registries — `modeling_auto.py`'s
    `AutoModelForCausalLM.from_config` and `automodel.py`'s `MODEL_MAP`
    lookup — and all three mean the same thing about this run.
    """
    runner = _scripted(ShellResult(returncode=1, stdout=message))
    with pytest.raises(PreflightError, match="does not know this checkpoint's architecture"):
        substrate_gate(_TASK, _environment(), shell_runner=runner)


def test_an_unknown_architecture_with_no_proxy_says_to_declare_one():
    """Not the raw architecture error, which reads as a target defect.

    There is no target at preflight time, so a message naming the
    checkpoint's architecture as unsupported points the operator at exactly
    the thing that is supposed to be unsupported — the failure mode this
    gate exists to prevent, reproduced by the gate itself.
    """
    runner = _scripted(ShellResult(returncode=1, stdout=_UNKNOWN_ARCH))
    with pytest.raises(PreflightError) as excinfo:
        substrate_gate(_TASK, _environment(), shell_runner=runner)
    message = str(excinfo.value)
    assert "environment.substrate_checkpoint" in message
    assert "not a fault in the checkpoint or the host" in message
    # And it must steer away from the blunter escape hatch, which also gives
    # up the two checks that do not depend on the checkpoint at all.
    assert "--skip-substrate-gate" in message


def test_the_proxy_is_run_under_the_same_environment_and_runtime_block():
    sink: list[tuple[str, int]] = []
    notes: list[str] = []
    substrate_gate(
        _with_proxy(),
        _environment(),
        shell_runner=_scripted(
            ShellResult(returncode=1, stdout=_UNKNOWN_ARCH),
            ShellResult(returncode=0, stdout=_GENERATED),
            sink=sink,
        ),
        notes=notes,
    )
    assert len(sink) == 2
    first, second = sink[0][0], sink[1][0]
    assert "--model_dir=/ckpt " in first
    assert "--model_dir=/proxy-ckpt " in second
    # Same environment, same runtime block, same stock path — one variable.
    for command in (first, second):
        assert "TRTLLM_MODELING_V2=off" in command
        assert "--use_cuda_graph" in command
        assert "--kv_cache_dtype fp8" in command


def test_the_weaker_claim_is_recorded_verbatim():
    """The weaker claim is recorded verbatim.

    A note that said only "substrate gate passed" would be read later as
    the claim the unqualified gate makes, and the difference is precisely
    which failure this run has been protected from.
    """
    notes: list[str] = []
    substrate_gate(
        _with_proxy(),
        _environment(),
        shell_runner=_scripted(
            ShellResult(returncode=1, stdout=_UNKNOWN_ARCH),
            ShellResult(returncode=0, stdout=_GENERATED),
        ),
        notes=notes,
    )
    assert notes == [
        "substrate proven on `/proxy-ckpt`; this checkpoint's own geometry is "
        "unverified, so a kvCache sizing fault will surface at the first boot "
        "gate rather than here."
    ]


def test_a_real_host_failure_is_never_answered_by_measuring_another_model():
    """The whole reason the discriminator is narrow.

    Widening the fallback to "any failure" would let a genuine kvCache
    sizing fault on the target checkpoint be answered by silently booting a
    different model and reporting green — the same class of mistake as
    gating under `TRTLLM_MODELING_V2=auto`.
    """
    sink: list[tuple[str, int]] = []
    runner = _scripted(
        ShellResult(returncode=1, stdout="Impossible to fit in any sequence in kvCache"),
        ShellResult(returncode=0, stdout=_GENERATED),
        sink=sink,
    )
    with pytest.raises(PreflightError, match="cannot generate a token"):
        substrate_gate(_with_proxy(), _environment(), shell_runner=runner)
    assert len(sink) == 1, "the proxy was run for a failure that was not about the architecture"


def test_a_proxy_that_also_fails_fails_the_run_and_names_the_proxy():
    """The proxy is the control, so its failure is a host failure."""
    runner = _scripted(
        ShellResult(returncode=1, stdout=_UNKNOWN_ARCH),
        ShellResult(returncode=1, stdout="MPI_ERR_SPAWN: could not spawn processes"),
    )
    with pytest.raises(PreflightError) as excinfo:
        substrate_gate(_with_proxy(), _environment(), shell_runner=runner)
    message = str(excinfo.value)
    assert "failed on the declared proxy checkpoint /proxy-ckpt" in message
    assert "no target exists yet" in message


def test_a_checkpoint_the_stock_path_knows_never_reaches_the_proxy():
    """Nothing changes for the run that was already working."""
    sink: list[tuple[str, int]] = []
    notes: list[str] = []
    substrate_gate(
        _with_proxy(),
        _environment(),
        shell_runner=_scripted(ShellResult(returncode=0, stdout=_GENERATED), sink=sink),
        notes=notes,
    )
    assert len(sink) == 1
    assert any("stock path generated" in note for note in notes)


# ---------------------------------------------------------- agent plane


def test_one_good_agent_turn_passes():
    notes: list[str] = []
    agent_plane_gate(lambda: "OK", notes)
    assert any("agent plane" in n for n in notes)


def test_an_agent_that_cannot_start_fails_loudly():
    def probe():
        raise RuntimeError("EROFS: read-only file system, mkdir '.../credentials.lock'")

    with pytest.raises(PreflightError, match="not a model or target"):
        agent_plane_gate(probe)


def test_an_empty_reply_is_a_failure():
    """A role that cannot answer a trivial prompt will not write a decision."""
    with pytest.raises(PreflightError, match="empty reply"):
        agent_plane_gate(lambda: "   ")


def test_trust_remote_code_refusal_is_an_unsupported_architecture():
    """A checkpoint whose implementation travels with its weights.

    `transformers` raises this instead of any registry message when the
    checkpoint carries an `auto_map`, and that is the same condition: the
    stock path has no built-in architecture for it. Keying the discriminator
    only on TensorRT-LLM's registry missed it, and the run stopped on a
    checkpoint nobody expected the stock path to load.
    """
    from agent_flow.workflows.trtllm_modeling_v2.preflight import _reports_unsupported_architecture

    class _Result:
        detail = "exited 1"
        stdout_tail = (
            "ValueError: The repository /path/to/ckpt contains custom code which "
            "must be executed to correctly load the model. ... Please pass the "
            "argument `trust_remote_code=True` to allow custom code to be run."
        )

    assert _reports_unsupported_architecture(_Result())


def test_an_unknown_model_type_is_an_unsupported_architecture():
    """A checkpoint newer than the installed `transformers`, with no remote code.

    Measured on DeepSeek-V4.1-Flash-NVFP4 (`model_type: deepseek_v41`,
    transformers 5.5.4): `AutoConfig.from_pretrained` refuses before any
    registry is consulted, and the run stopped instead of falling back.
    """
    from agent_flow.workflows.trtllm_modeling_v2.preflight import _reports_unsupported_architecture

    class _Result:
        detail = "exited 1"
        stdout_tail = (
            "ValueError('The checkpoint you are trying to load has model type "
            "`deepseek_v41` but Transformers does not recognize this architecture. "
            "This could be because of an issue with the checkpoint, or because your "
            "version of Transformers is out of date.')"
        )

    assert _reports_unsupported_architecture(_Result())


def test_a_registration_shim_without_a_target_is_an_unsupported_architecture():
    """A registration shim without a target is an unsupported architecture.

    The stock path finds a class registered under the architecture name
    (so a multimodal input processor can be found) that raises when built
    without a modeling_v2 target: the acceptance gate's `=off` leg must fall
    back to the declared floor rather than hard-fail.
    """
    from agent_flow.workflows.trtllm_modeling_v2.preflight import _reports_unsupported_architecture

    class _Result:
        detail = "exited 1"
        stdout_tail = (
            "ValueError: DeepseekV41ForCausalLM has no built-in implementation: DeepSeek-V4.1 "
            "runs only as a modeling_v2 target (tensorrt_llm/_torch/_experimental/modeling_v2/"
            "models/deepseek_v41/), and this class only registers its multimodal input processor."
        )

    assert _reports_unsupported_architecture(_Result())


def test_a_kv_cache_failure_is_still_a_hard_stop():
    """The widening must not turn the fallback into "any failure"."""
    from agent_flow.workflows.trtllm_modeling_v2.preflight import _reports_unsupported_architecture

    class _Result:
        detail = "exited 1"
        stdout_tail = "Impossible to fit in any sequence in kvCache"

    assert not _reports_unsupported_architecture(_Result())

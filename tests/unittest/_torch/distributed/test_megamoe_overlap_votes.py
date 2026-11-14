# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vote transport for MegaMoE with the overlap scheduler under the Ray orchestrator.

TorchDist runs the executor's host votes as NCCL collectives on a side stream. Beside a MegaMoE
persistent kernel (every SM, spinning for its peers) NCCL kernels can deadlock, so for that
combination the executor creator moves the votes to TorchDist's SM-free transport, falls back to
the CPU backend where the stack cannot run it, and then refuses an explicit request for the NCCL
votes (by world consensus).
"""

import os
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor import py_executor_creator as creator

pytestmark = pytest.mark.cpu_only

ENV = creator.CUDA_OBJECT_COLLECTIVES_ENV_VAR


def _args(backend: str = "MEGAMOE_CUTEDSL", disable_overlap_scheduler: bool = False):
    return SimpleNamespace(
        moe_config=SimpleNamespace(backend=backend),
        disable_overlap_scheduler=disable_overlap_scheduler,
    )


class _Dist:
    """TorchDist's vote-transport surface."""

    def __init__(
        self, enabled: bool = True, sm_free: bool = True, peers_request_nccl: bool = False
    ):
        self.cuda_object_collectives_enabled = enabled
        self._sm_free = sm_free
        self._peers_request_nccl = peers_request_nccl
        self.sm_free_object_collectives_status = "off: the self-test failed"
        self.calls = []

    def allgather(self, obj):
        self.calls.append("allgather")
        return [obj, self._peers_request_nccl]

    def use_sm_free_object_collectives(self) -> bool:
        self.calls.append("use_sm_free")
        return self._sm_free

    def disable_cuda_object_collectives(self) -> None:
        self.calls.append("disable")
        self.cuda_object_collectives_enabled = False


@pytest.fixture
def env(monkeypatch):
    """The switch unset, restored after the test."""
    monkeypatch.setenv(ENV, "placeholder")
    monkeypatch.delenv(ENV)
    return monkeypatch


@pytest.fixture
def ray(env):
    env.setattr(creator, "mpi_disabled", lambda: True)
    return env


@pytest.mark.parametrize("backend", ["MEGAMOE_CUTEDSL", "MEGAMOE_DEEPGEMM", "megamoe_cutedsl"])
def test_megamoe_with_overlap_under_ray_votes_on_copy_engines(ray, backend):
    dist = _Dist()
    creator._keep_votes_off_sms_for_megamoe_overlap(_args(backend), dist)
    assert dist.calls == ["use_sm_free"]
    assert dist.cuda_object_collectives_enabled


@pytest.mark.parametrize("requested", [None, "1"])
def test_explicit_or_default_nccl_votes_run_on_copy_engines(ray, requested):
    if requested is not None:
        ray.setenv(ENV, requested)
    dist = _Dist()
    creator._keep_votes_off_sms_for_megamoe_overlap(_args(), dist)
    assert dist.calls == ["use_sm_free"]


def test_without_copy_engine_votes_they_take_the_cpu(ray):
    dist = _Dist(sm_free=False)
    creator._keep_votes_off_sms_for_megamoe_overlap(_args(), dist)
    assert dist.calls == ["use_sm_free", "allgather", "disable"]
    assert ENV not in os.environ


def test_without_copy_engine_votes_an_explicit_request_for_nccl_is_refused(ray):
    ray.setenv(ENV, "1")
    dist = _Dist(sm_free=False)
    with pytest.raises(ValueError, match="deadlocks.*the self-test failed"):
        creator._keep_votes_off_sms_for_megamoe_overlap(_args(), dist)
    assert dist.calls == ["use_sm_free", "allgather"]


def test_a_peers_explicit_request_for_nccl_is_refused_on_every_rank(ray):
    """The refusal is a world consensus: this rank leaves the switch unset."""
    dist = _Dist(sm_free=False, peers_request_nccl=True)
    with pytest.raises(ValueError, match="deadlocks"):
        creator._keep_votes_off_sms_for_megamoe_overlap(_args(), dist)
    assert dist.calls == ["use_sm_free", "allgather"]


def test_cpu_votes_already_requested_stay(ray):
    ray.setenv(ENV, "0")
    dist = _Dist(enabled=False)
    creator._keep_votes_off_sms_for_megamoe_overlap(_args(), dist)
    assert dist.calls == []


@pytest.mark.parametrize(
    "args",
    [
        _args("CUTLASS"),
        _args("TRTLLM"),
        _args("AUTO"),
        _args(disable_overlap_scheduler=True),
    ],
    ids=["cutlass", "trtllm", "auto", "overlap_off"],
)
def test_other_engines_keep_their_votes(ray, args):
    dist = _Dist()
    creator._keep_votes_off_sms_for_megamoe_overlap(args, dist)
    assert dist.calls == []


def test_mpi_engines_are_untouched(env):
    """MPI communicators vote on the host already."""
    env.setattr(creator, "mpi_disabled", lambda: False)
    env.setenv(ENV, "1")
    dist = _Dist()
    creator._keep_votes_off_sms_for_megamoe_overlap(_args(), dist)
    assert dist.calls == [] and os.environ[ENV] == "1"

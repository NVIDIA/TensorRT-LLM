# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The copy plan is opt-in: without a copier the shared call sites take their step-input pack path unchanged, and a
model uses one path or the other, never both."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.pyexecutor import device_copy, step_inputs
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.device_copy import CopyPlan, CopySite, DeviceCopier
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import check_one_copy_path
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks
from tensorrt_llm._torch.pyexecutor.step_inputs import StepInputPack
from tensorrt_llm._torch.speculative.interface import _upload_host_values

_PLAN = CopyPlan(merge={CopySite.GRAPH_INPUT_IDS, CopySite.SPEC_RNG_SEEDS})


def _scatter_op_registered() -> bool:
    return torch.cuda.is_available() and hasattr(torch.ops.trtllm, "step_inputs_scatter")


_needs_scatter = pytest.mark.skipif(
    not _scatter_op_registered(),
    reason="CUDA and torch.ops.trtllm.step_inputs_scatter (built from "
    "cpp/tensorrt_llm/thop/minimaxM3/stepInputsScatterOp.cpp) are required",
)


def _device():
    return torch.device("cuda", torch.cuda.current_device())


@pytest.mark.parametrize(
    "hooks", [RuntimeHooks(pack_step_inputs=True), RuntimeHooks(skip_unchanged_h2d=True)]
)
def test_a_model_with_a_plan_cannot_also_turn_on_the_step_input_pack(hooks):
    with pytest.raises(ValueError, match="as a whole"):
        check_one_copy_path(SimpleNamespace(), _PLAN, hooks)


@pytest.mark.parametrize(
    "plan, hooks",
    [
        (CopyPlan(), RuntimeHooks(pack_step_inputs=True, skip_unchanged_h2d=True)),
        (_PLAN, RuntimeHooks()),
        (CopyPlan(), RuntimeHooks()),
    ],
)
def test_either_path_alone_is_accepted(plan, hooks):
    check_one_copy_path(SimpleNamespace(), plan, hooks)


@pytest.mark.parametrize("pack", [False, True])
def test_a_routed_upload_without_a_copier_keeps_its_pack_choice(monkeypatch, pack):
    calls = []
    monkeypatch.setattr(step_inputs, "copy_", lambda dst, src: calls.append("pack"))
    monkeypatch.setattr(device_copy, "copy", lambda *args, **kwargs: calls.append("plan"))
    dst = torch.zeros(4, dtype=torch.int64)
    _upload_host_values(dst, np.arange(3, dtype=np.int64), pack, CopySite.SPEC_RNG_SEEDS)
    if pack:
        assert calls == ["pack"]
    else:
        assert calls == [] and dst.tolist() == [0, 1, 2, 0]


@pytest.mark.parametrize("pack", [False, True])
def test_a_routed_upload_with_a_copier_goes_through_its_site(monkeypatch, pack):
    calls = []
    monkeypatch.setattr(step_inputs, "copy_", lambda dst, src: calls.append("pack"))
    monkeypatch.setattr(device_copy, "copy", lambda site, dst, src: calls.append(site))
    monkeypatch.setattr(device_copy, "current", lambda: object())
    _upload_host_values(
        torch.zeros(4, dtype=torch.int64),
        np.arange(3, dtype=np.int64),
        pack,
        CopySite.SPEC_RNG_SEEDS,
    )
    assert calls == [CopySite.SPEC_RNG_SEEDS]


@_needs_scatter
def test_a_static_input_copy_without_a_copier_is_staged_into_the_open_step_input_pack():
    dst = torch.zeros(4, dtype=torch.int32, device="cuda")
    src = torch.arange(4, dtype=torch.int32, device="cuda")
    pack = StepInputPack(_device())
    with step_inputs.active(pack):
        step_inputs.begin()
        CUDAGraphRunner._copy_static_input(CopySite.GRAPH_INPUT_IDS, dst, src)
        torch.cuda.synchronize()
        # Staged into the pack, not issued: it lands at the pack's flush.
        assert dst.tolist() == [0, 0, 0, 0]
        step_inputs.flush()
    torch.cuda.synchronize()
    assert dst.tolist() == [0, 1, 2, 3]


@_needs_scatter
def test_a_static_input_copy_with_a_copier_is_merged_through_its_site():
    dst = torch.zeros(4, dtype=torch.int32, device="cuda")
    src = torch.arange(4, dtype=torch.int32, device="cuda")
    copier = DeviceCopier(_PLAN, _device())
    with device_copy.active(copier):
        device_copy.begin()
        CUDAGraphRunner._copy_static_input(CopySite.GRAPH_INPUT_IDS, dst, src)
        assert copier.pending_sites(dst) == [CopySite.GRAPH_INPUT_IDS]
        assert not step_inputs.is_open()
        device_copy.flush()
    torch.cuda.synchronize()
    assert dst.tolist() == [0, 1, 2, 3]

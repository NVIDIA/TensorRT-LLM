# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The DeepGEMM MegaMoE backend fences its warmup launches with an EP barrier."""

import types

import pytest
import torch
import torch.distributed as dist

from tensorrt_llm._torch.moe.fused_moe.impl_contract import MoERunContext
from tensorrt_llm._torch.moe.fused_moe.mega_moe import mega_moe_deepgemm
from tensorrt_llm._torch.moe.fused_moe.mega_moe.mega_moe_deepgemm import (
    DeepgemmCudaW4a8Mxfp4Mxfp8Impl,
    set_warmup_launch_fence,
)
from tensorrt_llm._torch.pyexecutor import model_engine

# Host-side control flow only. The marker is also what makes the file reachable:
# the CPU stage collects only files that carry it.
pytestmark = pytest.mark.cpu_only

_HIDDEN = 8


@pytest.fixture(autouse=True)
def _fence_off_after_test():
    yield
    set_warmup_launch_fence(False)


@pytest.fixture
def events(monkeypatch):
    """Barrier and kernel calls, in the order they happen."""
    recorded = []
    monkeypatch.setattr(
        dist, "barrier", lambda group=None, **kwargs: recorded.append(("barrier", group))
    )
    return recorded


def _backend(events, ep_size=4):
    """A stand-in carrying exactly what ``run_moe`` reads for a zero-token chunk."""
    backend = types.SimpleNamespace(
        ep_size=ep_size,
        _ep_pg=object(),
        layer_idx=0,
        mapping=types.SimpleNamespace(moe_ep_rank=0),
        dtype=torch.bfloat16,
        max_num_tokens=16,
        hidden_size=_HIDDEN,
        _symm_buffer=types.SimpleNamespace(x=torch.empty(0, _HIDDEN)),
        _dg=types.SimpleNamespace(
            fp8_fp4_mega_moe=lambda *args, **kwargs: events.append(("kernel", None))
        ),
        _t_l1=None,
        _t_l2=None,
        dg_activation="swiglu",
        act_clamp=None,
        fast_math=True,
        act_alpha=None,
        act_beta=None,
    )
    backend._fence_ep_ranks = types.MethodType(
        DeepgemmCudaW4a8Mxfp4Mxfp8Impl._fence_ep_ranks, backend
    )
    return backend


def _run_zero_token_chunk(backend):
    # Zero-token chunks still launch the kernel, so they exercise the launch path
    # without any of the input preparation.
    ctx = MoERunContext(
        token_selected_experts=torch.empty(0, 2, dtype=torch.int32),
        token_final_scales=torch.empty(0, 2),
        x=torch.empty(0, _HIDDEN, dtype=torch.bfloat16),
        x_sf=None,
        output_dtype=torch.bfloat16,
    )
    DeepgemmCudaW4a8Mxfp4Mxfp8Impl.run_moe(backend, ctx)


@pytest.mark.parametrize(
    ("enabled", "capturing", "expected"),
    [
        (True, False, ["barrier", "kernel"]),
        (True, True, ["kernel"]),
        (False, False, ["kernel"]),
    ],
)
def test_run_moe_fences_warmup_launches(monkeypatch, events, enabled, capturing, expected):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    backend = _backend(events)
    set_warmup_launch_fence(enabled)

    _run_zero_token_chunk(backend)

    assert [kind for kind, _ in events] == expected
    assert all(group is backend._ep_pg for kind, group in events if kind == "barrier")


def test_single_rank_ep_is_not_fenced(monkeypatch, events):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    set_warmup_launch_fence(True)

    _run_zero_token_chunk(_backend(events, ep_size=1))

    assert [kind for kind, _ in events] == ["kernel"]


@pytest.mark.parametrize("in_warmup", [True, False])
def test_engine_warmup_transitions_switch_the_fence(monkeypatch, in_warmup):
    monkeypatch.setattr(model_engine, "_set_moe_a2a_warmup", lambda value: None)
    set_warmup_launch_fence(not in_warmup)

    # The setter only stores attributes on the engine, so a plain object can stand
    # in for PyTorchModelEngine.
    model_engine.PyTorchModelEngine.is_warmup.fset(types.SimpleNamespace(), in_warmup)

    assert mega_moe_deepgemm._warmup_launch_fence is in_warmup

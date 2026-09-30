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
"""Each MoE communication timeout sink must reach its backend's native setting.

The native round trips need the built extensions; DeepGEMM and DeepEP also need a GPU.
"""

from types import SimpleNamespace

import pytest
import torch
from utils.util import skip_pre_hopper

from tensorrt_llm._torch.moe.fused_moe import deep_ep_utils
from tensorrt_llm._torch.moe.fused_moe.communication import nvlink_one_sided
from tensorrt_llm._torch.moe.fused_moe.mega_moe.mega_moe_deepgemm import _DeepGemmBarrierTimeoutSink
from tensorrt_llm._torch.moe.fused_moe.nccl_ep_utils import _NcclEpGroupTimeoutSink

_NVLINK_ONE_SIDED_DEFAULT_SEC = 300
_DEEP_EP_DEFAULT_SEC = 100


def test_nvlink_one_sided_sink_sets_the_completion_flag_budget() -> None:
    sink = nvlink_one_sided._TIMEOUT_SINK
    try:
        sink.set_timeout_seconds(1800)
        assert torch.ops.trtllm.moe_a2a_get_timeout() == 1800
        sink.set_timeout_seconds(None)
        assert torch.ops.trtllm.moe_a2a_get_timeout() == _NVLINK_ONE_SIDED_DEFAULT_SEC
    finally:
        sink.set_timeout_seconds(None)


@pytest.mark.parametrize("timeout_sec", [-1, 86401])
def test_moe_a2a_set_timeout_rejects_out_of_range(timeout_sec: int) -> None:
    before = torch.ops.trtllm.moe_a2a_get_timeout()

    with pytest.raises(RuntimeError, match="MoE all-to-all timeout"):
        torch.ops.trtllm.moe_a2a_set_timeout(timeout_sec)

    assert torch.ops.trtllm.moe_a2a_get_timeout() == before


def test_deepgemm_sink_maps_the_native_default_to_zero() -> None:
    calls: list[int] = []
    sink = _DeepGemmBarrierTimeoutSink(SimpleNamespace(set_barrier_timeout_seconds=calls.append))

    sink.set_timeout_seconds(1800)
    sink.set_timeout_seconds(None)

    assert calls == [1800, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DeepGEMM's runtime needs a GPU")
def test_deepgemm_sink_sets_the_barrier_timeout() -> None:
    dg = pytest.importorskip("tensorrt_llm.deep_gemm")
    sink = _DeepGemmBarrierTimeoutSink(dg)
    try:
        sink.set_timeout_seconds(1800)
        assert dg.get_barrier_timeout_seconds() == 1800
        sink.set_timeout_seconds(None)
        assert dg.get_barrier_timeout_seconds() == 0
        with pytest.raises(RuntimeError):
            dg.set_barrier_timeout_seconds(86401)
    finally:
        sink.set_timeout_seconds(None)


@skip_pre_hopper
@pytest.mark.skipif(not deep_ep_utils.deep_ep_installed, reason="DeepEP is not installed")
def test_deep_ep_sink_sets_the_kernel_timeout() -> None:
    sink = deep_ep_utils._TIMEOUT_SINK
    try:
        sink.set_timeout_seconds(1800)
        assert deep_ep_utils.deep_ep.get_timeout_seconds() == 1800
        sink.set_timeout_seconds(None)
        assert deep_ep_utils.deep_ep.get_timeout_seconds() == _DEEP_EP_DEFAULT_SEC
    finally:
        sink.set_timeout_seconds(None)


def test_nccl_ep_sink_converts_seconds_to_group_nanoseconds() -> None:
    calls: list[int] = []
    sink = _NcclEpGroupTimeoutSink(SimpleNamespace(set_timeout_ns=calls.append))

    sink.set_timeout_seconds(1800)
    sink.set_timeout_seconds(None)

    assert calls == [1_800_000_000_000, 0]

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
"""Byte-level parity with the linear-layout CUDA NVFP4 quantizer."""

import pytest
import torch
import triton
import triton.language as tl
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.kernels.nvfp4_quant import can_use_nvfp4_quant, nvfp4_quant

pytestmark = skip_pre_blackwell


def _check(x: torch.Tensor, scale: torch.Tensor) -> None:
    actual = nvfp4_quant(x, scale)
    expected = torch.ops.trtllm.fp4_quantize(x, scale, 16, False, False)
    for result, reference in zip(actual, expected):
        assert result.dtype == reference.dtype == torch.uint8
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [1, 3, 4, 5, 129, 1025, 4097])
@pytest.mark.parametrize("hidden", [1024, 2048, 4096, 7168])
@pytest.mark.parametrize("global_scale", [1e-6, 1.0, 100.0, 1e6])
def test_nvfp4_quant_parity(rows: int, hidden: int, global_scale: float) -> None:
    torch.manual_seed(42)
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([global_scale], device="cuda", dtype=torch.float32)
    _check(x, scale)


@pytest.mark.parametrize("global_scale", [1e-6, 1.0, 100.0, 1e6])
def test_nvfp4_quant_finite_bf16_patterns(global_scale: float) -> None:
    # All finite BF16 bit patterns, including signed zero and subnormals.
    x = torch.arange(65536, device="cuda").to(torch.int16).view(torch.bfloat16)
    x = torch.where(x.isfinite(), x, 0).reshape(64, 1024)
    scale = torch.tensor([global_scale], device="cuda", dtype=torch.float32)
    _check(x, scale)


@pytest.mark.parametrize("fill", [0.0, -0.0, 0.25, -0.75, 1.25, 2.5, 5.0, 6.0])
def test_nvfp4_quant_rounding_and_zero(fill: float) -> None:
    x = torch.full((5, 1024), fill, device="cuda", dtype=torch.bfloat16)
    # Max=6, global_scale=1 makes the E4M3 block scale exactly 1.
    x[:, 15::16] = 6.0
    _check(x, torch.ones(1, device="cuda"))


def test_nvfp4_quant_graph_replay() -> None:
    x = torch.randn(5, 4096, device="cuda", dtype=torch.bfloat16)
    scale = torch.ones(1, device="cuda")
    _check(x, scale)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = nvfp4_quant(x, scale)
    for value in [0.0, 1.0, -2.5]:
        x.fill_(value)
        graph.replay()
        expected = torch.ops.trtllm.fp4_quantize(x, scale, 16, False, False)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_nvfp4_quant_eligibility() -> None:
    x = torch.ones(5, 2048, device="cuda", dtype=torch.bfloat16)
    scale = torch.ones(1, device="cuda")
    assert can_use_nvfp4_quant(x, scale)
    for excluded in [x[:0], x[:, :0], x[:, :1024], x[:, ::2], x.float(), x.half(), x[0]]:
        assert not can_use_nvfp4_quant(excluded, scale)
    for excluded_scale in [scale.cpu(), scale.half(), scale.expand(2)]:
        assert not can_use_nvfp4_quant(x, excluded_scale)


@pytest.mark.parametrize("global_scale", [1.0, 100.0])
@pytest.mark.parametrize(
    "value", [float("inf"), -float("inf"), float("nan")], ids=["inf", "neg_inf", "nan"]
)
def test_nvfp4_quant_nonfinite_inputs(global_scale: float, value: float) -> None:
    scale = torch.tensor([global_scale], device="cuda", dtype=torch.float32)
    # Exercise each lane of a scale block, mixed and entirely non-finite blocks.
    for lane in range(16):
        x = torch.linspace(-6, 6, 1024, device="cuda", dtype=torch.bfloat16).repeat(5, 1)
        x[:, lane::16] = value
        _check(x, scale)
    _check(torch.full((5, 1024), value, device="cuda", dtype=torch.bfloat16), scale)


@pytest.mark.parametrize("global_scale", [1e-6, 1.0, 100.0, 1e6])
@pytest.mark.parametrize("pattern", ["odd", "even", "all", "mixed"])
def test_nvfp4_quant_nan_lane_groups(global_scale: float, pattern: str) -> None:
    x = torch.linspace(-6, 6, 4096, device="cuda", dtype=torch.bfloat16).repeat(5, 1)
    if pattern == "odd":
        x[:, 1::2] = float("nan")
    elif pattern == "even":
        x[:, 0::2] = float("nan")
    elif pattern == "all":
        x.fill_(float("nan"))
    else:
        blocks = x.reshape(5, 256, 16)
        blocks[:, 0::4, 1::2] = float("nan")
        blocks[:, 1::4, 0::2] = float("nan")
        blocks[:, 2::4, :] = float("nan")
    _check(x, torch.tensor([global_scale], device="cuda", dtype=torch.float32))


@pytest.mark.parametrize("global_scale", [1e-6, 1.0, 100.0, 1e6])
def test_nvfp4_quant_all_nan_position_masks(global_scale: float) -> None:
    mask_ids = torch.arange(65536, device="cuda")[:, None]
    positions = torch.arange(16, device="cuda")[None, :]
    nan_mask = ((mask_ids >> positions) & 1).bool()
    x = torch.randn(65536, 16, device="cuda", dtype=torch.bfloat16)
    x.masked_fill_(nan_mask, float("nan"))
    _check(x.reshape(1024, 1024), torch.tensor([global_scale], device="cuda", dtype=torch.float32))


@triton.jit
def _producer(src, dst, count: tl.constexpr, pdl: tl.constexpr) -> None:
    if pdl:
        tl.extra.cuda.gdc_wait()
    offsets = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    values = tl.load(src + offsets, offsets < count, other=0)
    tl.store(dst + offsets, values, offsets < count)
    if pdl:
        tl.inline_asm_elementwise(
            "fence.acq_rel.gpu; // dummy $0",
            "=r,~{memory}",
            [],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        tl.debug_barrier()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _consumer(q, sf, out_q, out_sf, nq: tl.constexpr, nsf: tl.constexpr, pdl: tl.constexpr) -> None:
    if pdl:
        tl.extra.cuda.gdc_wait()
    offsets = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    q_values = tl.load(q + offsets, offsets < nq, other=0)
    sf_values = tl.load(sf + offsets, offsets < nsf, other=0)
    tl.store(out_q + offsets, q_values, offsets < nq)
    tl.store(out_sf + offsets, sf_values, offsets < nsf)
    if pdl:
        tl.inline_asm_elementwise(
            "fence.acq_rel.gpu; // dummy $0",
            "=r,~{memory}",
            [],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        tl.debug_barrier()
        tl.extra.cuda.gdc_launch_dependents()


@pytest.mark.parametrize("rows", [5, 1025, 4097])
def test_nvfp4_quant_graph_dependency_chain(rows: int) -> None:
    pdl = get_env_enable_pdl()
    source = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
    x = torch.empty_like(source)
    scale = torch.tensor([100.0], device="cuda")

    def chain() -> tuple[torch.Tensor, torch.Tensor]:
        _producer[(triton.cdiv(x.numel(), 1024),)](source, x, x.numel(), pdl, launch_pdl=pdl)
        q, sf = nvfp4_quant(x, scale)
        out_q, out_sf = torch.empty_like(q), torch.empty_like(sf)
        _consumer[(triton.cdiv(q.numel(), 1024),)](
            q, sf, out_q, out_sf, q.numel(), sf.numel(), pdl, launch_pdl=pdl
        )
        return out_q, out_sf

    for _ in range(3):
        chain()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = chain()
    for iteration in range(50):
        source.normal_()
        if iteration % 5 == 0:
            source[:, 1::2] = float("nan")
        graph.replay()
        expected = torch.ops.trtllm.fp4_quantize(source, scale, 16, False, False)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result.flatten(), reference.flatten(), rtol=0, atol=0)

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
"""kernels/megamoe: the fused MegaMoE top-k reduce tail and the add + RMSNorm (MXFP8) kernels, bit for bit against
the stock MegaMoE reduce and FlashInfer's fused add + RMSNorm."""

import functools

import pytest
import torch

from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() < 100,
    reason="needs an SM100+ GPU",
)

EPS = 1e-6


@functools.cache
def _stock_reduce(num_rows: int, num_topk: int, hidden_size: int):
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.helpers.topk_reduce import TopkReduce
    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.quant_def import CombineFormat

    reducer = TopkReduce(hidden_size, num_topk, CombineFormat.parse("bf16"))

    def to_cute(tensor):
        return from_dlpack(tensor, assumed_align=16 if tensor.dtype == torch.bfloat16 else 4)

    def stream():
        return cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    plane = torch.empty(num_rows, num_topk, hidden_size, dtype=torch.bfloat16, device="cuda")
    scores = torch.empty(num_rows, num_topk, dtype=torch.float32, device="cuda")
    out = torch.empty(num_rows, hidden_size, dtype=torch.bfloat16, device="cuda")
    compiled = cute.compile(reducer, to_cute(plane), None, to_cute(out), to_cute(scores), stream())

    def run(plane, scores):
        out = torch.empty(plane.shape[0], hidden_size, dtype=torch.bfloat16, device="cuda")
        compiled(to_cute(plane), None, to_cute(out), to_cute(scores), stream())
        return out

    return run


def _inputs(num_rows, num_topk, hidden_size, case, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, device="cuda")

    plane = randn(num_rows, num_topk, hidden_size)
    if case == "wide_range":
        exponent = torch.randint(-30, 30, plane.shape, generator=generator, device="cuda").float()
        plane = plane * torch.exp2(exponent)
    elif case == "cancellation":
        plane[:, 1::2] = -plane[:, 0::2][:, : plane[:, 1::2].shape[1]] * (1 + 2**-7)
    elif case == "subnormal":
        plane = plane * 2.0**-130
    plane = plane.to(torch.bfloat16)
    raw = torch.rand(num_rows, num_topk, generator=generator, device="cuda")
    scores = (raw / raw.sum(dim=-1, keepdim=True) * 2.5).float()
    shared = randn(num_rows, hidden_size).to(torch.bfloat16)
    residual = (randn(num_rows, hidden_size) * 4).to(torch.bfloat16)
    weight = (1 + 0.1 * randn(hidden_size)).to(torch.bfloat16)
    return plane, scores, shared, residual, weight


def _stock_moe(plane, scores, shared):
    routed = _stock_reduce(*plane.shape)(plane, scores)
    return shared.clone().add_(routed)


def _bits(tensor):
    return tensor.view(torch.int16)


CASES = ("normal", "wide_range", "cancellation", "subnormal")
# Every case without PDL, plus one PDL launch.
CASE_PDL = [(case, False) for case in CASES] + [("normal", True)]
# 1..192 rows take the decode launch configuration; 1025 takes the prefill one.
NUM_ROWS = [1, 1025]


@pytest.mark.parametrize("hidden_size,num_topk", [(6144, 4)])
@pytest.mark.parametrize("num_rows", NUM_ROWS)
@pytest.mark.parametrize("case,enable_pdl", CASE_PDL)
def test_shared_add_form_matches_stock(hidden_size, num_topk, num_rows, case, enable_pdl):
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.kernels.megamoe.megamoe_topk_reduce_tail import (  # noqa: E501
        megamoe_topk_reduce_shared_add,
    )

    plane, scores, shared, _, _ = _inputs(num_rows, num_topk, hidden_size, case)
    expected = _stock_moe(plane, scores, shared)
    plane_before = plane.clone()
    megamoe_topk_reduce_shared_add(plane, scores, shared, enable_pdl=enable_pdl)
    torch.cuda.synchronize()
    assert torch.equal(_bits(shared), _bits(expected))
    assert torch.equal(_bits(plane), _bits(plane_before))


def _norm_matches():
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.kernels.megamoe.megamoe_topk_reduce_tail import (  # noqa: E501
        megamoe_topk_reduce_add_rmsnorm_matches_flashinfer,
    )

    return megamoe_topk_reduce_add_rmsnorm_matches_flashinfer(torch.device("cuda"), 6144)


@pytest.mark.parametrize("num_rows", NUM_ROWS)
@pytest.mark.parametrize("case,enable_pdl", CASE_PDL)
def test_norm_form_matches_stock(num_rows, case, enable_pdl):
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.kernels.megamoe.megamoe_topk_reduce_tail import (  # noqa: E501
        megamoe_topk_reduce_add_rmsnorm,
    )

    if not _norm_matches():
        pytest.skip("FlashInfer does not run FusedAddRMSNormKernel with one CTA per row here")
    plane, scores, shared, residual, weight = _inputs(num_rows, 4, 6144, case)
    expected_out = _stock_moe(plane, scores, shared)
    expected_residual = residual.clone()
    torch.ops.trtllm.flashinfer_fused_add_rmsnorm(expected_out, expected_residual, weight, EPS)
    megamoe_topk_reduce_add_rmsnorm(
        plane, scores, shared, residual, weight, EPS, enable_pdl=enable_pdl
    )
    torch.cuda.synchronize()
    assert torch.equal(_bits(residual), _bits(expected_residual))
    assert torch.equal(_bits(shared), _bits(expected_out))


# Widest first, so each narrower step's last scale tile still holds the previous step's rows.
MXFP8_ROWS = (448, 192, 129, 88, 64, 8, 3, 1)


def _mxfp8_out(num_rows, hidden_size=6144):
    """Garbage-filled MXFP8 outputs, so a byte the kernel leaves unwritten fails the comparison."""
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.kernels.megamoe.mxfp8_rows import (
        scale_bytes,
    )

    q = torch.full((num_rows, hidden_size), 0xAB, dtype=torch.uint8, device="cuda")
    sf = torch.full((scale_bytes(num_rows, hidden_size),), 0xAB, dtype=torch.uint8, device="cuda")
    return q.view(torch.float8_e4m3fn), sf


def _assert_stock_quantize(rows, q, sf):
    ref_q, ref_sf = torch.ops.trtllm.mxfp8_quantize(rows, True)
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))
    assert ref_sf.numel() == sf.numel()
    assert torch.equal(sf, ref_sf.view(torch.uint8).view(-1))


def _add_rmsnorm_inputs(num_rows, case, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, device="cuda")

    rows = randn(num_rows, 6144)
    residual = randn(num_rows, 6144) * 4
    if case == "wide_range":
        exponent = torch.randint(-30, 30, rows.shape, generator=generator, device="cuda").float()
        rows = rows * torch.exp2(exponent)
    rows, residual = rows.to(torch.bfloat16), residual.to(torch.bfloat16)
    if case == "zero_rows":
        rows[::3] = 0
        residual[::3] = 0
    weight = (1 + 0.1 * randn(6144)).to(torch.bfloat16)
    return rows, residual, weight


NORM_CASE_PDL = [(case, False) for case in ("normal", "wide_range", "zero_rows")] + [
    ("normal", True)
]


@pytest.mark.parametrize("case,enable_pdl", NORM_CASE_PDL)
def test_add_rmsnorm_mxfp8_is_flashinfer_then_the_quantize(case, enable_pdl):
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.kernels.megamoe.add_rmsnorm_mxfp8 import (
        add_rmsnorm_mxfp8,  # noqa: E501
    )

    if not _norm_matches():
        pytest.skip("FlashInfer does not run FusedAddRMSNormKernel with one CTA per row here")
    for num_rows in MXFP8_ROWS:
        rows, residual, weight = _add_rmsnorm_inputs(num_rows, case, num_rows)
        expected_out, expected_residual = rows.clone(), residual.clone()
        torch.ops.trtllm.flashinfer_fused_add_rmsnorm(expected_out, expected_residual, weight, EPS)
        q, sf = _mxfp8_out(num_rows)
        add_rmsnorm_mxfp8(rows, residual, weight, EPS, q, sf, enable_pdl=enable_pdl)
        torch.cuda.synchronize()
        assert torch.equal(_bits(residual), _bits(expected_residual)), num_rows
        assert torch.equal(_bits(rows), _bits(expected_out)), num_rows
        _assert_stock_quantize(expected_out, q, sf)


def _replicas_out(num_rows, copies, hidden_size=6144):
    """A garbage-filled replicas buffer, so an element the kernel leaves unwritten fails the comparison."""
    full = torch.full((num_rows, copies * hidden_size), -1, dtype=torch.int16, device="cuda")
    return full.view(torch.bfloat16)


@pytest.mark.parametrize("case,enable_pdl", NORM_CASE_PDL)
@pytest.mark.parametrize("copies", [3])
def test_add_rmsnorm_replicated_is_flashinfer_then_the_cat(case, enable_pdl, copies):
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.kernels.megamoe.add_rmsnorm_mxfp8 import (
        add_rmsnorm_replicated,  # noqa: E501
    )

    if not _norm_matches():
        pytest.skip("FlashInfer does not run FusedAddRMSNormKernel with one CTA per row here")
    for num_rows in MXFP8_ROWS:
        rows, residual, weight = _add_rmsnorm_inputs(num_rows, case, num_rows)
        expected_out, expected_residual = rows.clone(), residual.clone()
        torch.ops.trtllm.flashinfer_fused_add_rmsnorm(expected_out, expected_residual, weight, EPS)
        replicas = _replicas_out(num_rows, copies)
        add_rmsnorm_replicated(rows, residual, weight, EPS, replicas, enable_pdl=enable_pdl)
        torch.cuda.synchronize()
        assert torch.equal(_bits(residual), _bits(expected_residual)), num_rows
        assert torch.equal(_bits(rows), _bits(expected_out)), num_rows
        expected = torch.cat([expected_out] * copies, dim=-1)
        assert torch.equal(_bits(replicas), _bits(expected)), num_rows

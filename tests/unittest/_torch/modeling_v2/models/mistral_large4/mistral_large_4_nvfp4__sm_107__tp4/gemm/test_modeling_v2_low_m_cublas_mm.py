# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the sm_107 tp4 target's low_m_cublas_mm entry."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm as op
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp4.ops.gemm.low_m_cublas_mm import (  # noqa: E501
    _LOW_M_DIRECT_TACTICS,
    _low_m_direct_tactic,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp4.ops.gemm.low_m_cublas_mm import (  # noqa: E501
    low_m_cublas_mm as low_m_op,
)

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402,F401 — needs the path declared above
    ULP,
    assert_within_ulp,
    certifying,
    validating,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="low_m_cublas_mm requires a CUDA device"
)


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """The cublas_mm entry's band, derived in its test: a bf16 output within 4 ulp element-wise
    and 2 ulp of relative RMS, an fp32 output within 512 and 256 of fp32's ulp."""
    if out.dtype is torch.float32:
        element_ulp, rms_ulp, ulp = 512.0, 256.0, ULP[torch.float32]
    else:
        element_ulp, rms_ulp, ulp = 4.0, 2.0, ULP[torch.bfloat16]
    assert_within_ulp(out, ref, element_ulp=element_ulp, rms_ulp=rms_ulp, ulp=ulp)


@pytest.mark.parametrize("cell", low_m_op.CELLS, ids=[cell.why for cell in low_m_op.CELLS])
@pytest.mark.parametrize("seed", [42, 123, 2026])
def test_low_m_fixed_tactic_graph_and_offsets(cell, seed, certifying) -> None:  # noqa: F811
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Fixed low-M tactics require SM107")
    certifying(low_m_op)
    spec = cell.spec
    torch.manual_seed(seed)
    m, n, k = spec["m"], spec["n"], spec["k"]
    a = torch.randn(m + 1, k, device="cuda", dtype=torch.bfloat16)[1:]
    b = torch.randn(n + 1, k, device="cuda", dtype=torch.bfloat16)[1:].t()
    assert a.storage_offset() and b.storage_offset()
    with validating(low_m_op):
        _compare(low_m_op(a, b), low_m_op.reference(a, b))
    low_m_op(a, b)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = low_m_op(a, b)
    for replay in range(100):
        a.normal_()
        a[-1].zero_()
        graph.replay()
        _compare(out, low_m_op.reference(a, b))
        assert torch.equal(out, low_m_op(a, b))
        assert not out[-1].view(torch.int16).any()
    a.zero_()
    a[:, :3] = 1
    b.zero_()
    b[0].fill_(1)
    b[1].fill_(-1)
    b[2].fill_(2**-10)
    graph.replay()
    _compare(out, low_m_op.reference(a, b))


@pytest.mark.parametrize(
    "m,n",
    [(m, 160) for m in (1, 2, 3, 12, 32, 64, 128, 256, 2048, 16384)]
    + [(16, 512), (16, 576), (24, 576), (32, 576)],
)
def test_low_m_protection_fallback(m, n) -> None:
    assert (m, n, 6144) not in _LOW_M_DIRECT_TACTICS
    a = torch.randn(m, 6144, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, 6144, device="cuda", dtype=torch.bfloat16).t()
    actual = low_m_op(a, b)
    assert torch.equal(actual, op(a, b))
    _compare(actual, low_m_op.reference(a, b))


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("has_bias", [False, True])
def test_low_m_output_dtype_and_bias_fallback(out_dtype, has_bias) -> None:
    a = torch.randn(4, 6144, device="cuda", dtype=torch.bfloat16)
    n = 128 if out_dtype == torch.float32 else 160
    b = torch.randn(n, 6144, device="cuda", dtype=torch.bfloat16).t()
    bias = torch.randn(n, device="cuda", dtype=out_dtype) if has_bias else None
    actual = low_m_op(a, b, bias, out_dtype)
    if has_bias or n == 128:
        assert torch.equal(actual, op(a, b, bias, out_dtype))
    _compare(actual, low_m_op.reference(a, b, bias, out_dtype))


@pytest.mark.parametrize("buffer_kind,group", [(1, None), (0, []), (0, [0, 1])])
def test_low_m_collective_buffer_arguments_are_forwarded(monkeypatch, buffer_kind, group) -> None:
    a = torch.empty(4, 6144, device="cuda", dtype=torch.bfloat16)
    b = torch.empty(160, 6144, device="cuda", dtype=torch.bfloat16).t()
    calls = []

    def fallback(*args):
        calls.append(args)
        return a.new_empty((4, 160))

    monkeypatch.setattr(torch.ops.trtllm, "cublas_mm", fallback)
    low_m_op(a, b, output_buffer_kind=buffer_kind, group=group)
    assert len(calls) == 1
    assert calls[0][4] == buffer_kind and calls[0][5] is group


@pytest.mark.parametrize("offset", [1, 8, 16, 64])
def test_low_m_operand_alignment(offset) -> None:
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Fixed low-M tactics require SM107")
    a = torch.randn(4 * 6144 + offset, device="cuda", dtype=torch.bfloat16)[offset:].view(4, 6144)
    b = (
        torch.randn(160 * 6144 + offset, device="cuda", dtype=torch.bfloat16)[offset:]
        .view(160, 6144)
        .t()
    )
    selected = _low_m_direct_tactic(a, b, None, None, 0, None)
    assert (selected is not None) == (offset % 16 == 0)
    _compare(low_m_op(a, b), low_m_op.reference(a, b))

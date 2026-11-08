# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local VR200 parity checks for the DSA prefill scorer."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.ops.attention.cute_dsl_gvr_topk_prefill import (  # noqa: E501
    cute_dsl_gvr_topk_prefill,
    gvr_workspace,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.ops.attention.triton_fp8_mqa_logits import (  # noqa: E501
    triton_fp8_mqa_logits as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="requires a VR200/SM107 device",
)


@pytest.mark.parametrize("cell", op.CELLS, ids=[cell.why for cell in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest fixture
    certifying(op)
    m, n = cell.spec["m"], cell.spec["n"]
    generator = torch.Generator(device="cuda").manual_seed(cell.seed_base())
    q = torch.randn(m, 32, 128, device="cuda", generator=generator).to(torch.float8_e4m3fn)
    k = torch.randn(n, 128, device="cuda", generator=generator).to(torch.float8_e4m3fn)
    scale = torch.rand(n, device="cuda", generator=generator) + 0.01
    weight = torch.randn(m, 64, device="cuda", generator=generator)[:, ::2]
    start = torch.arange(m, device="cuda", dtype=torch.int32) % max(n // 2, 1)
    end = (start + torch.arange(m, device="cuda", dtype=torch.int32) * 71 + 1).clamp(max=n + 17)
    if m > 1:
        end[0] = start[0]
    if cell.spec.get("strided"):

        def strided_view(tensor: torch.Tensor) -> torch.Tensor:
            storage = torch.zeros(tensor.numel() * 2, dtype=tensor.dtype, device=tensor.device)
            storage[::2].copy_(tensor)
            return storage[::2]

        scale, start, end = (strided_view(tensor) for tensor in (scale, start, end))
    args = q, (k, scale), weight, start, end
    actual = op(*args)
    assert actual.stride(0) % 4 == 0
    reference = op.reference(*args)
    torch.testing.assert_close(torch.isneginf(actual), torch.isneginf(reference))
    live = torch.isfinite(reference)
    assert bool(torch.isfinite(actual[live]).all())
    magnitude = torch.zeros_like(reference)
    with true_fp32_matmul():
        for head in range(32):
            magnitude += (q[:, head].float().abs() @ k.float().abs().t()) * weight[
                :, head : head + 1
            ].abs()
    magnitude *= scale[None, :]
    error = (actual[live].double() - reference[live].double()).abs()
    assert float((error / magnitude[live].double().clamp_min(1e-30)).max()) <= 8 * 2**-23
    relative_rms = error.square().sum().sqrt() / reference[
        live
    ].double().square().sum().sqrt().clamp_min(1e-30)
    assert float(relative_rms) <= 8 * 2**-23


@pytest.mark.parametrize("n", [2051, 2530, 2533])
def test_prefill_topk_consumer(n: int) -> None:
    m, top_k = 17, 2048
    generator = torch.Generator(device="cuda").manual_seed(n)
    q = torch.randn(m, 32, 128, device="cuda", generator=generator).to(torch.float8_e4m3fn)
    k = torch.randn(n, 128, device="cuda", generator=generator).to(torch.float8_e4m3fn)
    scale = torch.rand(n, device="cuda", generator=generator) + 0.01
    weight = torch.randn(m, 32, device="cuda", generator=generator)
    start = torch.zeros(m, device="cuda", dtype=torch.int32)
    end = torch.full_like(start, n)
    end[:3] = torch.tensor([0, 31, top_k], device="cuda", dtype=torch.int32)
    start[-1] = 32
    logits = op(q, (k, scale), weight, start, end)
    output = torch.full((m, top_k), -7, device="cuda", dtype=torch.int32)
    cute_dsl_gvr_topk_prefill(logits, start, end, output, top_k, gvr_workspace(q.device))
    for row in range(m):
        lo, hi = int(start[row]), int(end[row])
        if hi - lo <= top_k:
            expected = torch.full_like(output[row], -1)
            expected[: hi - lo] = torch.arange(hi - lo, device="cuda", dtype=torch.int32)
            torch.testing.assert_close(output[row], expected, rtol=0, atol=0)
        else:
            expected = torch.argsort(logits[row, lo:hi], descending=True, stable=True)[:top_k]
            torch.testing.assert_close(
                output[row].long().sort().values, expected.sort().values, rtol=0, atol=0
            )

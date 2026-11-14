# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the target's activation/k3_draft_silu_and_mul op (sm_100).

Every cell runs through ``__call__`` (with ``is_valid`` interposed) and ``reference``, and also through the stock
``torch.ops.trtllm.silu_and_mul(x)`` the built-in drafter's GatedMLP calls: the entry must give the stock call's bits.
"""

import importlib
import zlib

import pytest
import torch

assert torch.cuda.is_available(), "k3_draft_silu_and_mul requires a CUDA device"

if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
    pytest.skip("k3_draft_silu_and_mul is certified on sm_100 and sm_103", allow_module_level=True)

k3_draft_silu_and_mul = importlib.import_module(
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl."
    "kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.ops.activation.k3_draft_silu_and_mul"
).k3_draft_silu_and_mul

__extra_import_path__ = ["../../.."]
from _validating import validating  # noqa: E402

# Both sides compute silu(gate) * up in fp32 and round once to bf16. The kernel's sigmoid may differ from torch's in
# the last fp32 bits, which moves the rounded result by at most one bf16 spacing: at most eps * |ref|.
RTOL = torch.finfo(torch.bfloat16).eps


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _check(x: torch.Tensor) -> None:
    with validating(k3_draft_silu_and_mul):
        out = k3_draft_silu_and_mul(x)
    assert out.shape == (x.shape[0], x.shape[1] // 2) and out.dtype == x.dtype
    assert out.is_contiguous() and out.device == x.device
    stock = torch.ops.trtllm.silu_and_mul(x)
    assert torch.equal(_bits(out), _bits(stock)), (
        "the entry differs from the stock silu_and_mul call"
    )
    ref = k3_draft_silu_and_mul.reference(x)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=0.0)


@pytest.mark.parametrize(
    "cell", k3_draft_silu_and_mul.CELLS, ids=[c.why for c in k3_draft_silu_and_mul.CELLS]
)
def test_k3_draft_silu_and_mul(cell) -> None:
    spec = cell.spec
    gen = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    x = (torch.randn(spec["rows"], spec["width"], generator=gen, device="cuda") * 4.0).bfloat16()
    _check(x)


def test_row_strided_input() -> None:
    """A view with a row stride wider than the row (any row stride is in the contract)."""
    gen = torch.Generator(device="cuda").manual_seed(7)
    wide = (torch.randn(56, 1792 + 256, generator=gen, device="cuda") * 4.0).bfloat16()
    x = wide[:, :1792]
    assert not x.is_contiguous() and x.stride(1) == 1
    _check(x)


_OUT_OF_CONTRACT = {
    "3-D input": lambda x: x.view(2, 4, 1792),
    "odd width": lambda x: x[:, :1791],
    "strided last dim": lambda x: x[:, ::2],
}


@pytest.mark.parametrize("case", list(_OUT_OF_CONTRACT))
def test_rejects_out_of_contract(case) -> None:
    """The guard refuses the inputs the kernel would answer wrongly (it raises before the kernel runs)."""
    x = _OUT_OF_CONTRACT[case](torch.randn(8, 1792, dtype=torch.bfloat16, device="cuda"))
    with validating(k3_draft_silu_and_mul), pytest.raises(AssertionError):
        k3_draft_silu_and_mul(x)

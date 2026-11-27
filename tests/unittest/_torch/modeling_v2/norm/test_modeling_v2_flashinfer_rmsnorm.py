# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_rmsnorm catalog entry.

The entry owns what is certified: `CELLS` is the list, `reference` and
`compare` are the gate, `is_valid` is the guard. This file owns only what a
cell cannot carry -- turning a spec into real tensors -- plus the inputs the
guard exists to refuse.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch.modeling_v2.catalog.norm.flashinfer_rmsnorm import flashinfer_rmsnorm as op

__extra_import_path__ = [".."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "flashinfer_rmsnorm requires a CUDA device"


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape, dtype = spec["shape"], spec["dtype"]
    x = torch.randn(shape, generator=g, device="cuda").to(dtype)
    weight = torch.randn(shape[-1], generator=g, device="cuda").to(dtype)
    return x, weight, spec["eps"]


# The shapes each shipped target calls this entry with, by target directory
# under `models/`. The entry's own `CELLS` are the op's whoever calls it.
TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "deepseek_v3/r1_0528_nvfp4__sm_10x__default": (
        Cell(
            why="decode-shaped, deepseek-r1 hidden 7168",
            spec=dict(shape=(1, 7168), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="deepseek q_lora rank 1536, the q_norm this op is called on",
            spec=dict(shape=(16, 1536), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="deepseek kv_lora rank 512, the narrowest row any target asks for",
            spec=dict(shape=(16, 512), dtype=torch.bfloat16, eps=1e-6),
        ),
    ),
    "gpt_oss/gpt_oss_120b__sm_10x__default": (
        Cell(
            why="decode-shaped, gpt-oss-120b hidden 2880",
            spec=dict(shape=(1, 2880), dtype=torch.bfloat16, eps=1e-5),
        ),
    ),
    "mistral_large4/mistral_large_4_nvfp4__sm_103__tep": (
        Cell(
            why="ML4 hidden 6144 rmsnorm",
            spec=dict(shape=(16, 6144), dtype=torch.bfloat16, eps=1e-5),
        ),
        Cell(
            why="ML4 kv_lora 512 rmsnorm",
            spec=dict(shape=(16, 512), dtype=torch.bfloat16, eps=1e-5),
        ),
    ),
}

CELLS = certified_cells(op, TARGET_CELLS)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    x, weight, eps = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(x, weight, eps)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert out.data_ptr() != x.data_ptr(), "the op must return a new tensor"
    torch.testing.assert_close(out, op.reference(x, weight, eps))


def test_guard_refuses_fp8_before_the_kernel_answers() -> None:
    """The hazard `is_valid` exists for: accepted, and silently plausible.

    Driving the op directly shows why the guard is not cosmetic -- it returns a
    tensor rather than raising, so a caller who forgot to dequantize gets a
    number back.
    """
    x = torch.randn(4, 2880, device="cuda").to(torch.float8_e4m3fn)
    weight = torch.randn(2880, device="cuda").to(torch.float8_e4m3fn)
    unguarded = torch.ops.trtllm.flashinfer_rmsnorm(x, weight, 1e-6)
    assert unguarded.dtype is torch.float8_e4m3fn, "the hazard this guard covers is gone"

    with pytest.raises(AssertionError, match="dequantize"):
        with validating(op):
            op(x, weight, 1e-6)


def test_strided_normalized_dim_raises() -> None:
    """`x.stride(-1) != 1` is rejected by the kernel's call wrapper."""
    buf = torch.randn(8, 2880, 2, dtype=torch.bfloat16, device="cuda")
    x = buf[..., 0]
    assert x.stride(-1) != 1
    weight = torch.randn(2880, dtype=torch.bfloat16, device="cuda")
    with pytest.raises((AssertionError, ValueError)):
        with validating(op):
            op(x, weight, 1e-6)


def test_out_inplace_strided_and_overlap_guard() -> None:
    """`out=` writes in place or into a strided latent view without touching the tail."""
    g = torch.Generator(device="cuda").manual_seed(0)
    kv = torch.randn(16, 576, generator=g, device="cuda").to(torch.bfloat16)
    weight = torch.randn(512, generator=g, device="cuda").to(torch.bfloat16)
    ref = op(kv[:, :512], weight, 1e-5)

    # In-place on the leading 512 columns preserves the 64-column RoPE tail.
    tail_before = kv[:, 512:].clone()
    inplace = kv.clone()
    with validating(op):
        res = op(inplace[:, :512], weight, 1e-5, out=inplace[:, :512])
    assert res.data_ptr() == inplace.data_ptr()
    assert torch.equal(res, ref)
    assert torch.equal(inplace[:, 512:], tail_before)

    # Strided destination buffer.
    dst = torch.zeros_like(kv)
    with validating(op):
        res2 = op(kv[:, :512], weight, 1e-5, out=dst[:, :512])
    assert res2.data_ptr() == dst.data_ptr()
    assert torch.equal(res2, ref)
    assert torch.count_nonzero(dst[:, 512:]) == 0

    # Partial overlap between x and out is refused by is_valid.
    with pytest.raises(AssertionError, match="overlap"):
        with validating(op):
            op(kv[:-1, :512], weight, 1e-5, out=kv[1:, :512])

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_fused_add_rmsnorm catalog entry.

The entry owns what is certified: `CELLS` is the list, `reference` and
`compare` are the gate, `is_valid` is the guard. This file owns only what a
cell cannot carry -- turning a spec into real tensors -- plus the inputs the
guard exists to refuse, which are claims about inputs no cell describes.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (
    flashinfer_fused_add_rmsnorm as op,
)

__extra_import_path__ = [".."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "flashinfer_fused_add_rmsnorm requires a CUDA device"


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape, dtype = spec["shape"], spec["dtype"]
    x = torch.randn(shape, generator=g, device="cuda").to(dtype)
    residual = torch.randn(shape, generator=g, device="cuda").to(dtype)
    weight = torch.randn(shape[-1], generator=g, device="cuda").to(dtype)
    return x, residual, weight, spec["eps"]


# The shapes each shipped target calls this entry with, by target directory
# under `models/`. The entry's own `CELLS` are the op's whoever calls it.
TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "deepseek_v3/r1_0528_nvfp4__sm_10x__default": (
        Cell(
            why="decode-shaped, deepseek-r1 hidden",
            spec=dict(shape=(1, 7168), dtype=torch.bfloat16, eps=1e-6),
        ),
    ),
    "gpt_oss/gpt_oss_120b__sm_10x__default": (
        Cell(
            why="decode-shaped, gpt-oss-120b hidden",
            spec=dict(shape=(1, 2880), dtype=torch.bfloat16, eps=1e-5),
        ),
    ),
}

CELLS = certified_cells(op, TARGET_CELLS)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:40] for c in CELLS])
def test_certified_cells(cell) -> None:
    """Every certified configuration, kernel against reference.

    Driven inside `validating`, so each cell also proves the guard admits the
    inputs a shipped target actually passes.
    """
    x, residual, weight, eps = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    ref_x, ref_residual = op.reference(x, residual, weight, eps)
    with validating(op):
        out_x, out_residual = op(x, residual, weight, eps)
        assert out_x is x and out_residual is residual, "the op writes in place"
    torch.testing.assert_close(residual, ref_residual)
    torch.testing.assert_close(x, ref_x)


def _one(**over):
    g = torch.Generator(device="cuda").manual_seed(7)
    kw = dict(rows=4, hidden=2880, dtype=torch.bfloat16)
    kw.update(over)
    shape = (kw["rows"], kw["hidden"])
    return (
        torch.randn(shape, generator=g, device="cuda").to(kw["dtype"]),
        torch.randn(shape, generator=g, device="cuda").to(kw["dtype"]),
        torch.randn(shape[-1], generator=g, device="cuda").to(kw["dtype"]),
    )


def test_guard_refuses_fp8() -> None:
    """Accepted, and silently plausible: the case a caller cannot detect."""
    x, residual, weight = _one(dtype=torch.float8_e4m3fn)
    with pytest.raises(AssertionError, match="dequantize"):
        with validating(op):
            op(x, residual, weight, 1e-6)


def test_guard_refuses_a_strided_normalized_dim() -> None:
    """`stride(-1) != 1` reads the wrong elements rather than failing."""
    buf = torch.randn(8, 2880, 2, dtype=torch.bfloat16, device="cuda")
    x, residual = buf[..., 0], buf[..., 1]
    assert x.stride(-1) != 1
    weight = torch.randn(2880, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="contiguous"):
        with validating(op):
            op(x, residual, weight, 1e-6)


def test_guard_refuses_aliased_buffers() -> None:
    """The op writes both tensors; one buffer passed twice yields neither."""
    x, _, weight = _one()
    with pytest.raises(AssertionError, match="distinct buffers"):
        with validating(op):
            op(x, x, weight, 1e-6)


def test_the_guard_is_not_on_the_hot_path() -> None:
    """Outside `validating`, the wrapper goes straight to the kernel.

    Stated as a test because it is the property that lets the guard be as
    strict as it likes: a served engine never runs it.
    """
    x, residual, weight = _one(dtype=torch.float8_e4m3fn)
    # fp8 is what the guard refuses; unguarded, the kernel answers anyway.
    op(x, residual, weight, 1e-6)

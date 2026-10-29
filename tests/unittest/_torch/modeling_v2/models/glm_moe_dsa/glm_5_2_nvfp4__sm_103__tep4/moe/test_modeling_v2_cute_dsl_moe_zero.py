# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_moe_zero entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("cute_dsl_moe_zero runs sm_103 CuTe DSL kernels", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_zero import (  # noqa: E402, E501
    DECODE_MAX_ROWS,
    HIDDEN,
    CompiledZero,
    build_decode,
    build_prefill,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_zero import (  # noqa: E402, E501
    cute_dsl_moe_zero as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

#: Rows past the cleared view, in the same allocation, that the clear must not touch.
_TAIL_ROWS = 8

#: Compiled clears, one per decode row count plus one prefill clear, built once a session.
_HANDLES: dict[object, CompiledZero] = {}


def _compiled(rows: int) -> CompiledZero:
    key = "prefill" if rows > DECODE_MAX_ROWS else rows
    if key not in _HANDLES:
        device = torch.cuda.current_device()
        _HANDLES[key] = build_prefill(device) if key == "prefill" else build_decode(rows, device)
    return _HANDLES[key]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    """Every element exactly zero, and nothing past the view written.

    A clear does no arithmetic: both sides store the bit pattern of +0.0, so
    the comparison is `torch.equal`. `out` starts as random bf16 rather than
    zeros, so an element the kernel skips keeps a nonzero value and fails.

    `out` is the first `rows` rows of a larger allocation whose remaining rows
    hold random values too. The decode clear writes rows `0..rows-1` only, and
    the prefill clear writes exactly `numel * 2` bytes; a grid or byte count
    that ran one vector too far would change the tail.
    """
    rows = cell.spec["rows"]
    compiled = _compiled(rows)
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    buf = torch.randn(rows + _TAIL_ROWS, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    out = buf[:rows]
    tail = buf[rows:].clone()
    with validating(op):
        assert op(out, compiled) is out, "the op clears `out` and returns it"
    assert torch.equal(out, op.reference(out, compiled)), "an element was not cleared"
    assert torch.equal(buf[rows:], tail), "the clear wrote past the rows it was given"


def _unreachable(*args) -> None:
    raise AssertionError("the guard should have refused before the kernel ran")


def test_guard_refuses_a_row_count_the_decode_clear_was_not_compiled_for() -> None:
    """The decode clear's row count is compiled in: 5 rows on a 4-row handle
    leave the last row uncleared."""
    out = torch.empty(5, HIDDEN, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="compiled for 4 rows, got 5"):
        with validating(op):
            op(out, CompiledZero("decode", 4, 0, _unreachable))


def test_guard_refuses_a_strided_view() -> None:
    """A column slice of a wider buffer: the kernel clears dense rows from its
    pointer, so it writes the gap columns between the view's rows and leaves
    the view's last columns uncleared."""
    wide = torch.empty(4, HIDDEN + 64, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="out must be contiguous"):
        with validating(op):
            op(wide[:, :HIDDEN], CompiledZero("decode", 4, 0, _unreachable))

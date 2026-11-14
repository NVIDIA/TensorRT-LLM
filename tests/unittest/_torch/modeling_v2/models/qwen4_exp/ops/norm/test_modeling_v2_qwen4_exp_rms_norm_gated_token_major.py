# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rms_norm_gated_token_major entry: every cell against the
entry's reference, the direct launch bitwise equal to the registered op, and the
weight-length guard."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.norm import (
    rms_norm_gated_token_major as op_module,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.norm.rms_norm_gated_token_major import (  # noqa: E501
    rms_norm_gated_token_major as op,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "rms_norm_gated_token_major requires a CUDA device"


#: The width of the concatenated GDN input projection `[q k v | z | a | b]` a cell names no
#: `z_row_stride` for: all 48 heads, 10240 | 6144 | 48 | 48. A tp4/tep4 rank's is 4120:
#: 2560 | 1536 | 12 | 12.
_IN_PROJ_WIDTH = 16480


def _z_offset(row_stride: int, heads: int, n: int) -> int:
    """First column of `z` in a `[q k v | z | a | b]` row: `a` and `b` are one column a head."""
    offset = row_stride - heads * (n + 2)
    assert offset > 0, (row_stride, heads, n)
    return offset


#: The checkpoint's GDN epilogue geometry over a packed `z`, for the guard test.
_PACKED = dict(op.CELLS[0].spec, tokens=4, z="packed")


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _build(spec, seed):
    """`__call__`'s positional arguments for one spec."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    t, heads, n, dtype = spec["tokens"], spec["heads"], spec["width"], spec["dtype"]

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda").to(dtype)

    x = randn(t * heads, n)

    layout = spec.get("z", "packed")
    if layout == "in_proj":
        width = spec.get("z_row_stride", _IN_PROJ_WIDTH)
        offset = _z_offset(width, heads, n)
        z = randn(t, width)[:, offset : offset + heads * n].view(t, heads, n)
    else:
        assert layout == "packed", layout
        z = randn(t, heads, n)

    weight = randn(n)
    return (
        x,
        z,
        weight,
        spec["eps"],
        None,
        spec["gate_activation"],
        spec.get("direct_launch", False),
    )


#: The FP8 target's tp2/tep2 rank, called with `direct_launch=True`: 24 heads, `z` in its
#: 8240-wide projection.
_FP8_TP2 = dict(
    heads=24,
    width=128,
    dtype=torch.bfloat16,
    eps=1e-6,
    gate_activation="sigmoid",
    z="in_proj",
    z_row_stride=8240,
    direct_launch=True,
)

#: The shapes each target calls this entry with beyond its own cells, by target directory.
TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "qwen4_exp/qwen3_8_flash_next_fp8__sm_103__unified": (
        Cell(
            why="fp8 tp2/tep2, one token: 24 heads, z at token stride 8240; 4-row kernel, PDL",
            spec=dict(tokens=1, **_FP8_TP2),
        ),
        Cell(
            why="fp8 tp2/tep2, 683 tokens (16392 rows): 4-row kernel, grid past the SM count",
            spec=dict(tokens=683, **_FP8_TP2),
        ),
        Cell(
            why="fp8 tp2/tep2 prefill at max_num_tokens 8192: 196608 rows on the 8-row kernel",
            spec=dict(tokens=8192, **_FP8_TP2),
        ),
    ),
}

CELLS = certified_cells(op, TARGET_CELLS)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _build(cell.spec, seed=_seed(cell.why))
    x, z, weight = args[:3]
    assert z.stride()[1:] == (cell.spec["width"], 1)
    if cell.spec["tokens"] > 1:
        # A one-token view reports the dense stride for its size-1 token dim (and is then
        # contiguous); the kernel never reads past token 0 there.
        row_stride = cell.spec.get("z_row_stride", _IN_PROJ_WIDTH)
        assert z.stride(0) == row_stride and not z.is_contiguous()
    before = [x.clone(), z.clone(), weight.clone()]

    out = op(*args)

    assert out.shape == x.shape and out.is_contiguous() and out.dtype == x.dtype
    for t, b in zip((x, z, weight), before):
        assert torch.equal(t, b), "inputs are read-only"
    torch.testing.assert_close(out, op.reference(*args))


#: The direct-launch cells below the 8-row threshold and more shapes for the pin test: the
#: last row count before it, a PDL grid at each width, the silu gate over a packed `z`.
_PIN_SPECS = tuple(
    c.spec
    for c in CELLS
    if c.spec.get("direct_launch")
    and c.spec["tokens"] * c.spec["heads"] < op_module._PREFILL_MIN_ROWS
) + (
    dict(_FP8_TP2, tokens=4),
    dict(_FP8_TP2, tokens=128),
    dict(_FP8_TP2, tokens=1365),
    dict(op.CELLS[3].spec, tokens=4, direct_launch=True),
    dict(op.CELLS[3].spec, tokens=2730, direct_launch=True),
    dict(op.CELLS[-1].spec, direct_launch=True),
    dict(_PACKED, tokens=16, gate_activation="silu", direct_launch=True),
)


@pytest.mark.parametrize(
    "spec",
    _PIN_SPECS,
    ids=[
        f"{s['tokens']}x{s['heads']}h-{s.get('z_row_stride', s['z'])}-{s['gate_activation']}"
        for s in _PIN_SPECS
    ],
)
def test_direct_launch_is_bitwise_the_registered_op(spec) -> None:
    """`_direct_launch` is a copy of `torch.ops.trtllm.rms_norm_gated_token_major`'s
    multi-row launch: it is taken at every shape here and returns the op's bytes, so a
    change to the op's launch (kernel, rows a program, warps) fails here."""
    x, z, weight, eps, _, act, _ = _build(spec, seed=_seed(f"pin {spec}"))
    assert x.shape[0] < op_module._PREFILL_MIN_ROWS and op_module._multirow(x, z, act)
    expected = torch.ops.trtllm.rms_norm_gated_token_major(x, z, weight, eps, None, act)

    assert torch.equal(op_module._direct_launch(x, z, weight, eps, act), expected)
    assert torch.equal(op(x, z, weight, eps, None, act, True), expected)


def test_guard_refuses_a_weight_of_the_wrong_length() -> None:
    """Unguarded, a length-192 weight is silently truncated to weight[:128]."""
    x, z, weight, eps, _, act, _ = _build(_PACKED, seed=12)
    long_weight = torch.cat([weight, torch.randn(64, dtype=weight.dtype, device="cuda")])
    assert torch.equal(
        torch.ops.trtllm.rms_norm_gated_token_major(x, z, long_weight, eps, None, act),
        torch.ops.trtllm.rms_norm_gated_token_major(x, z, weight, eps, None, act),
    ), "the hazard this guard covers is gone"

    with pytest.raises(AssertionError, match="weight must be 1-D of length"):
        with validating(op):
            op(x, z, long_weight, eps, None, act)

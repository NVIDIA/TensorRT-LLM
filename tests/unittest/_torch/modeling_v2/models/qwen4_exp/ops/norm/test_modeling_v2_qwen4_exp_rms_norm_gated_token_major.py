# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rms_norm_gated_token_major entry: every cell against the
entry's reference, the direct launch bitwise equal to the registered op, and the
weight-length guard."""

import zlib

import pytest
import torch
import triton

from tensorrt_llm._torch.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.norm import (
    rms_norm_gated_token_major as op_module,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.norm.rms_norm_gated_token_major import (  # noqa: E501
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
    """`_direct_launch` makes `torch.ops.trtllm.rms_norm_gated_token_major`'s multi-row
    launch without the custom-op dispatch: it is taken at every shape here and returns the
    op's bytes."""
    x, z, weight, eps, _, act, _ = _build(spec, seed=_seed(f"pin {spec}"))
    assert x.shape[0] < op_module._PREFILL_MIN_ROWS and op_module._multirow(x, z, act)
    expected = torch.ops.trtllm.rms_norm_gated_token_major(x, z, weight, eps, None, act)

    assert torch.equal(op_module._direct_launch(x, z, weight, eps, act), expected)
    assert torch.equal(op(x, z, weight, eps, None, act, True), expected)


@pytest.mark.parametrize(
    "spec", [_FP8_TP2, op_module._TP4], ids=["24h-8240-fp8-tp2", "12h-4120-tp4"]
)
def test_direct_launch_reuses_the_compiled_variant_across_batches(spec) -> None:
    """After one decode-sized batch, no token count `_direct_launch` takes compiles a new
    variant of the multi-row kernel. One token included: its view gives `z` the dense token
    stride (3072 or 1536), and 1536 is a multiple of 16 where the projection's 4120 is not."""
    from _torch.helpers import triton_num_compiled

    kernel = op_module._layernorm_gated._rms_norm_gated_fwd_multirow_kernel
    while isinstance(kernel, triton.runtime.Heuristics):
        kernel = kernel.fn  # the compiled variants live on the inner JITFunction
    # Start from no compiled variants, as a server process does, so that those of earlier
    # tests (such as the one-token cells) cannot stand in for the ones warmup compiles.
    for cache in getattr(kernel, "device_caches", {}).values():
        cache[0].clear()
    op_module._layernorm_gated._multirow_variants_with_siblings.clear()

    def check(tokens: int) -> None:
        x, z, weight, eps, _, act, _ = _build(
            dict(spec, tokens=tokens), seed=_seed(f"reuse {tokens}")
        )
        assert x.shape[0] < op_module._PREFILL_MIN_ROWS
        out = op_module._direct_launch(x, z, weight, eps, act)
        torch.testing.assert_close(out, op.reference(x, z, weight, eps, None, act))

    check(32)
    compiled = triton_num_compiled(kernel)
    max_tokens = (op_module._PREFILL_MIN_ROWS - 1) // spec["heads"]
    for tokens in (1, 2, 3, 4, 17, 128, 1000, max_tokens):
        check(tokens)
    assert triton_num_compiled(kernel) == compiled


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

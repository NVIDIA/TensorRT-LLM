# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fp4_quantize catalog entry (NVFP4: sf_vec_size=16, e4m3 scales)."""

import zlib

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (  # noqa: E501
    fp4_quantize as op,
)

__extra_import_path__ = [".."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "fp4_quantize requires a CUDA device"

_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device="cuda"
)


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(spec["shape"], generator=g, device="cuda").to(spec["dtype"])
    # The canonical NVFP4 activation global scale, 448*6/amax.
    gs = (448.0 * 6.0 / x.abs().max().float()).reshape(1)
    return x, gs, spec["vec"], spec["swizzled"]


def _unpack(packed):
    """[M, K/2] packed bytes -> [M, K] codes (element 2i in the low nibble)."""
    m, half = packed.shape
    codes = torch.empty(m, half * 2, dtype=torch.uint8, device=packed.device)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = packed >> 4
    return codes


# The shapes each shipped target calls this entry with, by target directory
# under `models/`. The entry's own `CELLS` are the op's whoever calls it.
TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "deepseek_v3/r1_0528_nvfp4__sm_10x__default": (
        Cell(
            why="deepseek default at attention TP8: the shared expert's down input, 2048/8 wide",
            spec=dict(shape=(16, 256), dtype=torch.bfloat16, vec=16, swizzled=True),
        ),
    ),
    "qwen4_exp/qwen3_8_flash_next_nvfp4__sm_103__tp1": (
        Cell(
            why="qwen3.8 decode: 1 token, hidden 2560, linear -- its routed-expert input",
            spec=dict(shape=(1, 2560), dtype=torch.bfloat16, vec=16, swizzled=False),
        ),
        Cell(
            why="qwen3.8, 7 rows at hidden 2560, linear: several rows, no row padding",
            spec=dict(shape=(7, 2560), dtype=torch.bfloat16, vec=16, swizzled=False),
        ),
        Cell(
            why="qwen3.8 prefill at max_num_tokens 8192, hidden 2560, linear",
            spec=dict(shape=(8192, 2560), dtype=torch.bfloat16, vec=16, swizzled=False),
        ),
    ),
    "glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4": (
        Cell(
            why="glm-5.2 decode: 1 token, hidden 6144, linear",
            spec=dict(shape=(1, 6144), dtype=torch.bfloat16, vec=16, swizzled=False),
        ),
        Cell(
            why="glm-5.2, 64 rows: its own quantizer's ceiling",
            spec=dict(shape=(64, 6144), dtype=torch.bfloat16, vec=16, swizzled=False),
        ),
        Cell(
            why="glm-5.2, 65 rows: the first row count past its own quantizer",
            spec=dict(shape=(65, 6144), dtype=torch.bfloat16, vec=16, swizzled=False),
        ),
        Cell(
            why="glm-5.2 prefill chunk, 2048 rows at hidden 6144, linear",
            spec=dict(shape=(2048, 6144), dtype=torch.bfloat16, vec=16, swizzled=False),
        ),
        Cell(
            why="glm-5.2 at its MoE chunk bound, 8192 rows at hidden 6144",
            spec=dict(shape=(8192, 6144), dtype=torch.bfloat16, vec=16, swizzled=False),
        ),
    ),
}

CELLS = certified_cells(op, TARGET_CELLS)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    """Codes bit-exact away from the rounding ties the entry declines to predict.

    `near_tie` is the entry's own statement about where the kernel's approximate
    reciprocals decide the direction. Everything outside it has to match; the
    count inside it is asserted small, so the window cannot quietly grow into a
    licence for the kernel to be wrong.
    """
    x, gs, vec, swizzled = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        packed, sf = op(x, gs, vec, False, swizzled)
    ref_packed, ref_sf = op.reference(x, gs, vec, False, swizzled)

    assert torch.equal(sf, ref_sf), "block scales are exact; a mismatch is a defect"

    tie = op.near_tie(op.scaled(x, gs, vec))
    codes, ref_codes = _unpack(packed), _unpack(ref_packed)
    assert torch.equal(codes[~tie], ref_codes[~tie]), "codes differ away from a rounding tie"
    # Measured across these cells: 0.13% to 0.24%. Higher than a random landing
    # would give by some three orders of magnitude, because most of these are
    # *exact* midpoint hits rather than near misses -- a bf16 input times an
    # e4m3 scale lands on a grid coarse enough to hit midpoints often. The bound
    # is 4x the worst measured, tight enough that a window quietly widening into
    # a licence for the kernel to be wrong would trip it.
    assert tie.sum().item() <= 0.01 * tie.numel(), (
        f"{tie.sum().item()} of {tie.numel()} elements sit in the tie window; "
        "a window this wide is no longer a statement about rounding"
    )


def test_the_quantization_round_trips() -> None:
    """Independent of the reference: dequantizing has to return the input.

    The cells compare two implementations of one arithmetic, which would both be
    wrong together if the arithmetic were. This bounds the error against the
    input instead -- e2m1 carries one mantissa bit, so at worst half a step of
    the block's own scale.
    """
    x = torch.randn(64, 7168, dtype=torch.bfloat16, device="cuda")
    gs = (448.0 * 6.0 / x.abs().max().float()).reshape(1)
    packed, sf = op(x, gs, 16, False, False)

    codes = _unpack(packed)
    value = _E2M1_VALUES[(codes & 7).long()]
    value = torch.where((codes & 8).bool(), -value, value)
    scale = sf.view(64, 7168 // 16).view(torch.float8_e4m3fn).float()
    deq = value * scale.repeat_interleave(16, dim=-1) / gs.item()

    block_amax = x.float().view(64, 7168 // 16, 16).abs().amax(-1, keepdim=True)
    err = (deq - x.float()).view(64, 7168 // 16, 16).abs().amax(-1, keepdim=True)
    assert torch.all(err <= block_amax * 0.5 + 1e-6)


def test_the_two_layouts_differ_at_equal_length() -> None:
    """At a multiple of 128 rows the linear and swizzled buffers are the same
    length, so a consumer handed the wrong one accepts it. The linear cells'
    exact scale gate is what tells them apart -- this shows it can."""
    x, gs, _, _ = _build(dict(shape=(128, 2560), dtype=torch.bfloat16, vec=16, swizzled=False), 0)
    _, linear = op(x, gs, 16, False, False)
    _, swizzled = op(x, gs, 16, False, True)
    assert linear.numel() == swizzled.numel() == 128 * 2560 // 16
    assert not torch.equal(linear, swizzled)
    assert torch.equal(linear, op.reference(x, gs, 16, False, False)[1])


def test_guard_refuses_a_per_token_global_scale() -> None:
    """Element 0 is applied to every row and the rest are ignored."""
    x = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="exactly one element"):
        with validating(op):
            op(x, torch.ones(8, device="cuda"), 16, False, True)


def test_guard_refuses_a_partial_block() -> None:
    x = torch.randn(8, 120, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="whole number of 16-element"):
        with validating(op):
            op(x, torch.ones(1, device="cuda"), 16, False, True)


def test_guard_refuses_a_non_contiguous_input() -> None:
    x = torch.randn(8, 256, dtype=torch.bfloat16, device="cuda")[:, ::2]
    with pytest.raises(AssertionError, match="contiguous"):
        with validating(op):
            op(x, torch.ones(1, device="cuda"), 16, False, True)

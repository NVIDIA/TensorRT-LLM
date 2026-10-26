# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fp4_quantize catalog entry (NVFP4: sf_vec_size=16, e4m3 scales)."""

import zlib

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (  # noqa: E501
    fp4_quantize as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "fp4_quantize requires a CUDA device"

_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device="cuda"
)

#: e4m3 bytes: 448, the saturated scale, and the first normal one.
_E4M3_MAX_BYTE = 0x7E
_E4M3_MIN_NORMAL_BYTE = 0x08

#: Outlier channels of a Step-5 cell: 8 of 4096, 512 apart.
_OUTLIER_STRIDE = 512
#: The 16 channels of the block sized to a subnormal e4m3 scale.
_TINY_BLOCK = slice(16, 32)


def _step5_input(spec, g):
    """A Step-5 MoE input: one layer's post-attention RMSNorm output.

    The norm output has unit RMS before the weight, so the body is
    `N(0, sigma)` with `sigma` the layer's norm-weight RMS. Eight outlier
    channels at std `outlier` and one block at std `tiny` put blocks in the
    saturated and the subnormal scale regimes.
    """
    rows, k = spec["shape"]
    std = torch.full((k,), spec["sigma"], device="cuda")
    std[_OUTLIER_STRIDE // 2 :: _OUTLIER_STRIDE] = spec["outlier"]
    std[_TINY_BLOCK] = spec["tiny"]
    x = torch.randn(rows, k, generator=g, device="cuda") * std
    # The engine holds the reciprocal of the checkpoint's input_scale, taken in
    # fp32 on device, as a 0-dim parameter.
    gs = torch.tensor(spec["input_scale"], dtype=torch.float32, device="cuda").reciprocal()
    return x.to(spec["dtype"]), gs


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if "input_scale" in spec:
        x, gs = _step5_input(spec, g)
    else:
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


def _assert_matches(x, gs, vec, packed, sf, ref_packed, ref_sf):
    """Scales bit-exact; codes bit-exact away from the e2m1 rounding ties.

    `near_tie` is the entry's own statement about where the kernel's approximate
    reciprocals decide the direction. Everything outside it has to match; the
    count inside it is asserted small, so the window cannot quietly grow into a
    licence for the kernel to be wrong.
    """
    assert torch.equal(sf, ref_sf), "block scales are exact; a mismatch is a defect"

    tie = op.near_tie(op.scaled(x, gs, vec))
    codes, ref_codes = _unpack(packed), _unpack(ref_packed)
    assert torch.equal(codes[~tie], ref_codes[~tie]), "codes differ away from a rounding tie"
    # Measured across the cells: 0.13% to 0.24%. Higher than a random
    # landing would give by some three orders of magnitude, because most of
    # these are *exact* midpoint hits rather than near misses -- a bf16 input
    # times an e4m3 scale lands on a grid coarse enough to hit midpoints often.
    # The bound is 4x the worst measured, tight enough that a window quietly
    # widening into a licence for the kernel to be wrong would trip it.
    assert tie.sum().item() <= 0.01 * tie.numel(), (
        f"{tie.sum().item()} of {tie.numel()} elements sit in the tie window; "
        "a window this wide is no longer a statement about rounding"
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    """The op against the entry's reference at every cell."""
    certifying(op)
    x, gs, vec, swizzled = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    packed, sf = op(x, gs, vec, False, swizzled)
    ref_packed, ref_sf = op.reference(x, gs, vec, False, swizzled)
    _assert_matches(x, gs, vec, packed, sf, ref_packed, ref_sf)

    if "input_scale" in cell.spec:
        # A Step-5 cell is only worth its name if it reached both scale regimes
        # the checkpoint's global scale produces. Swizzle padding is zero, so
        # it counts in neither.
        saturated = (ref_sf == _E4M3_MAX_BYTE).sum().item()
        subnormal = ((ref_sf > 0) & (ref_sf < _E4M3_MIN_NORMAL_BYTE)).sum().item()
        assert saturated > 0, "no block reached the saturated e4m3 scale"
        assert subnormal > 0, "no block reached a subnormal e4m3 scale"


def test_cuda_graph_capture_and_replay() -> None:
    """The decode site is captured: a replay on refilled input must equal eager.

    Warm up on a side stream, capture at the 128-row decode batch with the
    layer-3 global scale, refill the static input in place, replay, and compare
    with an eager call on the same values bit for bit.
    """
    spec = next(c.spec for c in op.CELLS if "input_scale" in c.spec and c.spec["shape"][0] == 128)
    g = torch.Generator(device="cuda").manual_seed(128)
    x, gs = _step5_input(spec, g)
    static_x = x.clone()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            op(static_x, gs, 16, False, True)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_packed, static_sf = op(static_x, gs, 16, False, True)

    for _ in range(2):
        fresh, _ = _step5_input(spec, g)
        static_x.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        eager_packed, eager_sf = op(fresh, gs, 16, False, True)
        assert torch.equal(static_packed, eager_packed), "replayed codes differ from eager"
        assert torch.equal(static_sf, eager_sf), "replayed scales differ from eager"
        ref_packed, ref_sf = op.reference(fresh, gs, 16, False, True)
        _assert_matches(fresh, gs, 16, static_packed, static_sf, ref_packed, ref_sf)


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
    """At 128 rows the linear and swizzled buffers have the same length.

    A consumer handed the wrong one accepts it. The linear cells' exact scale
    gate is what tells them apart -- this shows it can.
    """
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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the weight side of ``rubin_mxfp8_linear``: K32 scale relay, row stack, guard.

The relay is checked byte for byte against a port of the C++ offset function that the engine's
``block_scale_interleave`` and ``mxfp8_quantize`` write with (``get_sf_out_offset_128x4``). The
entry's reference is checked against a float64 product built from the checkpoint's own form:
E4M3 values and row-major K32 scales, never relaid.
"""

import pytest
import torch
from _rubin_gemm_util import mxfp8_weight_scales, stack_mxfp8_weights

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm import (
    rubin_mxfp8_gemm as G,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm import (
    rubin_mxfp8_linear as L,
)

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 7),
        reason="K3.1 Rubin (sm_107) entries; the CPU stage runs this file",
    ),
]

op = L.rubin_mxfp8_linear


def _sf_offset(m_idx, k_idx, num_col_vecs):
    """``get_sf_out_offset_128x4`` of ``cpp/tensorrt_llm/kernels/quantization.cuh``, unbatched,
    on integer tensors: the byte offset of the scale of row ``m_idx``, column ``k_idx``."""
    inner_k_idx, inner_k_stride = k_idx % 4, 1
    inner_m_idx, inner_m_stride = (m_idx % (32 * 4)) // 32, 4 * inner_k_stride
    # The [32, 4] M tile is column-major.
    outer_m_idx, outer_m_stride = m_idx % 32, 4 * inner_m_stride
    k_tile_idx, k_tile_stride = k_idx // 4, 32 * outer_m_stride
    num_k_tiles = (num_col_vecs + 4 - 1) // 4
    m_tile_idx, m_tile_stride = m_idx // (32 * 4), num_k_tiles * k_tile_stride
    return (
        m_tile_idx * m_tile_stride
        + k_tile_idx * k_tile_stride
        + outer_m_idx * outer_m_stride
        + inner_m_idx * inner_m_stride
        + inner_k_idx * inner_k_stride
    )


def _relay_by_offsets(scales: torch.Tensor) -> torch.Tensor:
    """Each byte at its C++ offset, zero elsewhere: the flat R128c4 buffer."""
    rows, cols = scales.shape
    padded_rows, padded_cols = -(-rows // 128) * 128, -(-cols // 4) * 4
    out = torch.zeros(padded_rows * padded_cols, dtype=torch.uint8)
    r = torch.arange(rows).unsqueeze(1).expand(rows, cols)
    c = torch.arange(cols).unsqueeze(0).expand(rows, cols)
    out[_sf_offset(r, c, cols).reshape(-1)] = scales.reshape(-1)
    return out


def _checkpoint_weight(rows: int, k: int, gen: torch.Generator):
    """E4M3 values and row-major K32 scales of a ``[rows, k]`` weight, as the export stores them.
    Column magnitudes span four decades, so the 32-value scales differ from block to block."""
    w = torch.randn(rows, k, generator=gen) * torch.logspace(-2, 2, k)
    return G.quantize_mxfp8_reference(w.to(torch.bfloat16))


def _activation(rows: int, k: int, gen: torch.Generator) -> torch.Tensor:
    x = torch.randn(rows, k, generator=gen) * 4.0
    x[:, ::7] *= 30.0
    return x.to(torch.bfloat16)


def _modelopt_float64(x: torch.Tensor, values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """The ModelOpt W8A8 product in float64, from the checkpoint form (``scales`` row-major).

    Each 32 input values take the smallest power of two at or above ``amax / 448`` (found with
    frexp, not with the entry's bit arithmetic) and round to E4M3. Weight value ``(r, j)`` is
    ``values[r, j] * 2 ** (scales[r, j // 32] - 127)``.
    """
    m, k = x.shape
    n = values.shape[0]
    blocks = x.float().view(m, k // 32, 32)
    ratio = blocks.abs().amax(-1) / 448.0
    mantissa, exponent = torch.frexp(ratio)
    exponent = torch.where(mantissa == 0.5, exponent - 1, exponent)
    scale = torch.ldexp(torch.ones_like(ratio, dtype=torch.float64), exponent)
    scale = torch.where(ratio > 0, scale, torch.full_like(scale, 2.0**-127))
    xq = (blocks.double() / scale.unsqueeze(-1)).float().to(torch.float8_e4m3fn)
    xd = xq.double() * scale.unsqueeze(-1)
    wd = values.double().view(n, k // 32, 32) * torch.exp2(scales.double() - 127.0).unsqueeze(-1)
    return xd.view(m, k) @ wd.view(n, k).t()


def _within_one_rounding(out: torch.Tensor, exact: torch.Tensor) -> None:
    """One BF16 rounding of an FP32 sum of exact terms: within 2^-8 of the row's largest value."""
    assert out.dtype == torch.bfloat16 and out.shape == exact.shape
    row_max = exact.abs().amax(dim=1, keepdim=True)
    assert torch.all((out.double() - exact).abs() <= row_max * 2.0**-8)


_RELAY_SHAPES = [
    (1, 1),
    (37, 5),
    (130, 7),
    (12, 224),  # KDA b_proj at TP8
    (96, 224),  # KDA b_proj at DEP16
    (6284, 224),  # the KDA stack's rows at TP8, before padding
    (7168, 48),  # KDA o_proj at TP8
    (7168, 384),  # KDA o_proj at DEP16
    (1536, 224),  # shared experts [gate | up] at TP8
    (7168, 24),  # shared experts down at TP8
]


@pytest.mark.parametrize("rows, cols", _RELAY_SHAPES)
def test_relay_bytes_are_the_cpp_swizzle(rows, cols) -> None:
    gen = torch.Generator().manual_seed(rows * 7919 + cols)
    scales = torch.randint(0, 256, (rows, cols), generator=gen, dtype=torch.int32)
    scales = scales.to(torch.uint8)
    relaid = mxfp8_weight_scales(scales)
    assert relaid.dtype == torch.uint8 and relaid.is_contiguous()
    assert tuple(relaid.shape) == L.mxfp8_weight_scales_shape(rows, cols * 32)
    assert torch.equal(relaid, _relay_by_offsets(scales)), "every byte at its C++ offset"
    # It is the GEMM's buffer, and the inverse gives the checkpoint bytes back.
    assert torch.equal(relaid, G.mxfp8_scales_to_r128c4(scales))
    assert torch.equal(G.mxfp8_scales_from_r128c4(relaid, rows, cols * 32), scales)


def test_relay_and_stack_run_on_meta_tensors() -> None:
    """The relay and the stack are plain torch: they run on meta tensors."""
    meta = torch.device("meta")
    relaid = mxfp8_weight_scales(torch.empty(6144, 224, dtype=torch.uint8, device=meta))
    assert relaid.is_meta and tuple(relaid.shape) == (6144 * 224,)
    parts = [
        (
            torch.empty(n, 7168, dtype=torch.float8_e4m3fn, device=meta),
            torch.empty(L.mxfp8_weight_scales_shape(n, 7168), dtype=torch.uint8, device=meta),
        )
        for n in (6144, 128, 12)
    ]
    weight, weight_scale = stack_mxfp8_weights(parts)
    assert weight.is_meta and weight.dtype == torch.float8_e4m3fn
    assert tuple(weight.shape) == (6288, 7168)
    assert tuple(weight_scale.shape) == L.mxfp8_weight_scales_shape(6288, 7168) == (6400 * 224,)


@pytest.mark.parametrize(
    "rows, n_multiple",
    [((256, 128, 12), 8), ((256, 128, 12), 16), ((384, 128, 96), 8), ((128, 128), 8), ((44,), 8)],
)
def test_stack_is_one_relay_of_the_stacked_checkpoint_rows(rows, n_multiple) -> None:
    k = 256
    gen = torch.Generator().manual_seed(sum(rows) + n_multiple)
    parts = [_checkpoint_weight(n, k, gen) for n in rows]
    weight, weight_scale = stack_mxfp8_weights(
        [(v, mxfp8_weight_scales(s)) for v, s in parts], n_multiple
    )
    total = sum(rows)
    pad = -(-total // n_multiple) * n_multiple - total
    assert weight.dtype == torch.float8_e4m3fn and tuple(weight.shape) == (total + pad, k)
    # The checkpoint rows in order, then zero rows; the scales relaid once after stacking, with
    # zero bytes on the padding rows.
    want_values = torch.cat(
        [v.view(torch.uint8) for v, _ in parts] + [torch.zeros(pad, k, dtype=torch.uint8)]
    )
    want_scales = _relay_by_offsets(
        torch.cat([s for _, s in parts] + [torch.zeros(pad, k // 32, dtype=torch.uint8)])
    )
    assert torch.equal(weight.view(torch.uint8), want_values)
    assert torch.equal(weight_scale, want_scales)
    # Each part stays a view at its rows in both tensors: in the scales, its 128-row tiles.
    cols, offset = -(-(k // 32) // 4) * 4, 0
    for v, s in parts:
        n = v.shape[0]
        assert torch.equal(weight[offset : offset + n].view(torch.uint8), v.view(torch.uint8))
        tiles = weight_scale[offset * cols : (offset + -(-n // 128) * 128) * cols]
        assert torch.equal(tiles, mxfp8_weight_scales(s))
        offset += n


def test_stack_refuses_what_it_cannot_stack() -> None:
    gen = torch.Generator().manual_seed(5)
    a, a_s = _checkpoint_weight(96, 128, gen)
    b, b_s = _checkpoint_weight(128, 128, gen)
    c, c_s = _checkpoint_weight(128, 256, gen)
    relay = mxfp8_weight_scales
    with pytest.raises(AssertionError, match="last part"):
        stack_mxfp8_weights([(a, relay(a_s)), (b, relay(b_s))])  # 96 rows end inside a tile
    with pytest.raises(AssertionError, match="E4M3"):
        stack_mxfp8_weights([(b, relay(b_s)), (c, relay(c_s))])  # another K
    with pytest.raises(AssertionError, match="relaid"):
        stack_mxfp8_weights([(a, a_s)])  # the checkpoint's [96, 4]
    with pytest.raises(AssertionError, match="relaid"):
        stack_mxfp8_weights([(b, relay(b_s).view(128, 4))])  # relaid bytes as a matrix
    with pytest.raises(AssertionError, match="divides 128"):
        stack_mxfp8_weights([(b, relay(b_s))], n_multiple=24)
    with pytest.raises(AssertionError, match="scale bytes"):
        relay(a_s.float())
    with pytest.raises(AssertionError, match="relaid already"):
        relay(relay(b_s))  # a second relay would move the bytes again


@pytest.mark.parametrize("m, n, k", [(8, 128, 7168), (5, 96, 1536), (3, 24, 768)])
def test_reference_is_the_modelopt_recipe(m, n, k) -> None:
    gen = torch.Generator().manual_seed(m * n + k)
    x = _activation(m, k, gen)
    values, scales = _checkpoint_weight(n, k, gen)
    out = op.reference(x, values, mxfp8_weight_scales(scales))
    _within_one_rounding(out, _modelopt_float64(x, values, scales))


@pytest.mark.parametrize("rows", [(256, 128, 12), (128, 128, 96)])
def test_a_stacked_reference_gives_each_projection_its_columns(rows) -> None:
    k = 512
    gen = torch.Generator().manual_seed(sum(rows))
    parts = [_checkpoint_weight(n, k, gen) for n in rows]
    weight, weight_scale = stack_mxfp8_weights([(v, mxfp8_weight_scales(s)) for v, s in parts])
    x = _activation(16, k, gen)
    out = op.reference(x, weight, weight_scale)
    assert tuple(out.shape) == (16, weight.shape[0])
    offset = 0
    for v, s in parts:
        _within_one_rounding(out[:, offset : offset + v.shape[0]], _modelopt_float64(x, v, s))
        offset += v.shape[0]
    assert torch.count_nonzero(out[:, offset:]) == 0, "zero rows give zero columns"


def test_guard_takes_relaid_scales_and_refuses_checkpoint_scales() -> None:
    gen = torch.Generator().manual_seed(3)
    x = _activation(4, 128, gen)
    values, scales = _checkpoint_weight(128, 128, gen)  # [128, 4]: the padded R128c4 shape
    relaid = mxfp8_weight_scales(scales)
    # Well formed but on the CPU: the guard reaches its last line.
    with pytest.raises(AssertionError, match="CUDA device"):
        op.is_valid(x, values, relaid)
    # The checkpoint's matrix, and the relaid bytes as a matrix of the same shape: nothing in a
    # 2-D scale tells the two apart, so both are refused.
    for matrix in (scales, relaid.view(128, 4)):
        with pytest.raises(AssertionError, match="row-major"):
            op.is_valid(x, values, matrix)
    for bad in (relaid.new_zeros(2 * relaid.numel())[::2], relaid.view(torch.int8)):
        with pytest.raises(AssertionError, match="contiguous UE8M0"):
            op.is_valid(x, values, bad)
    longer, _ = _checkpoint_weight(256, 128, gen)
    with pytest.raises(AssertionError, match="this weight's relaid scales"):
        op.is_valid(x, longer, relaid)  # 128 rows' scales for a 256-row weight


def test_guard_refuses_checkpoint_scales_at_every_cell() -> None:
    """On o_proj and the shared experts a checkpoint's [N, K / 32] has the padded R128c4 matrix's
    shape (N % 128 == 0, K / 32 % 4 == 0); only the flat form marks relaid scales."""
    meta = torch.device("meta")
    same_shape = set()
    for cell in op.CELLS:
        m, n, k = cell.spec["m"], cell.spec["n"], cell.spec["k"]
        x = torch.empty(m, k, dtype=torch.bfloat16, device=meta)
        weight = torch.empty(n, k, dtype=torch.float8_e4m3fn, device=meta)
        checkpoint = torch.empty(n, k // 32, dtype=torch.uint8, device=meta)
        if n % 128 == 0 and k // 32 % 4 == 0:
            same_shape.add((n, k))
        with pytest.raises(AssertionError, match="row-major"):
            op.is_valid(x, weight, checkpoint)
        with pytest.raises(AssertionError, match="CUDA device"):
            op.is_valid(x, weight, mxfp8_weight_scales(checkpoint))
    # o_proj at TP8 and DEP16; the shared [gate | up] and down at TP8 and whole; dep16's MXFP8
    # levers' latent_down and the drafter's MLP gate_up and down.
    assert same_shape == {
        (7168, 1536),
        (7168, 12288),
        (1536, 7168),
        (12288, 7168),
        (7168, 768),
        (7168, 6144),
        (3584, 7168),
        (28672, 7168),
        (7168, 14336),
    }


def test_call_hands_the_gemm_the_relaid_buffer(monkeypatch) -> None:
    """The GEMM gets the caller's flat buffer itself: no view and no copy on the way."""
    gen = torch.Generator().manual_seed(4)
    x = _activation(4, 128, gen)
    values, scales = _checkpoint_weight(16, 128, gen)
    relaid = mxfp8_weight_scales(scales)
    want = op.reference(x, values, relaid)
    seen, flags, reserves = [], [], []

    def quantize(input):
        act, act_scale = G.quantize_mxfp8_reference(input)
        return act, G.mxfp8_scales_to_r128c4(act_scale)

    def gemm(act, weight, act_scale, weight_scale, pinned, reserve_sms):
        seen.append(weight_scale)
        flags.append(pinned)
        reserves.append(reserve_sms)
        return G.rubin_mxfp8_gemm.reference(act, weight, act_scale, weight_scale)

    monkeypatch.setattr(L, "_quantize", lambda: quantize)
    monkeypatch.setattr(L, "rubin_mxfp8_gemm", gemm)
    assert torch.equal(op(x, values, relaid), want)
    assert len(seen) == 1 and seen[0] is relaid
    assert flags == [False], "a plain call is not pinned (dep16's MXFP8 levers pass pinned=True)"
    assert reserves == [0], (
        "a plain call keeps the full grid (the shared gate_up passes its reserve)"
    )


def test_cells_carry_the_stacked_kda_projection() -> None:
    """The KDA [q | k | v | g | f_a | b] stack at TP8 and at DEP16 (attention DP)."""

    def stacked_rows(tp: int) -> int:
        heads = 96 // tp  # q / k / v / g are heads x 128 rows each; f_a 128; b one per head
        rows = 4 * heads * 128 + 128 + heads
        return -(-rows // L.N_MULTIPLE) * L.N_MULTIPLE

    shapes = {(cell.spec["n"], cell.spec["k"]) for cell in op.CELLS}
    assert {(stacked_rows(8), 7168), (stacked_rows(1), 7168)} <= shapes

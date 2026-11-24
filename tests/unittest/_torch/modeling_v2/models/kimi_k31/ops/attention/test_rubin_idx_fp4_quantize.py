# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the rubin_idx_fp4_quantize entry (DSA indexer Q / K to MXFP4)."""

import zlib

import pytest
import torch

# l0_gb300 collects this directory: any GPU but sm_107 skips the module before a family import.
if torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("certified on sm_107 only", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_idx_fp4_quantize import (  # noqa: E402,E501
    E2M1_MAGNITUDES,
    mxfp4_values,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_idx_fp4_quantize import (  # noqa: E402,E501
    rubin_idx_fp4_quantize as op,
)

__extra_import_path__ = ["../../../.."]
if torch.cuda.is_available():  # the GPU cells' guard interposer: a CPU run needs neither
    from _validating import certifying  # noqa: E402,F401 -- path above; fixture by name

cpu = pytest.mark.cpu_only
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="drives the kernel on sm_107")

HEADS, DIM = 32, 128


def _rows(rows: int, data: str, gen: torch.Generator) -> torch.Tensor:
    """bf16 [rows, 128] in one of the cells' data modes."""
    if data == "randn":
        return torch.randn(rows, DIM, generator=gen).bfloat16()
    if data == "zeros_block":
        x = torch.randn(rows, DIM, generator=gen)
        x.view(rows, 4, 32)[:, 1] = 0.0  # block 1 of every row
        x[::3] = 0.0  # and every third row whole
        return x.bfloat16()
    if data == "wide":
        x = torch.randn(rows, DIM, generator=gen)
        scale = torch.pow(2.0, torch.randint(-60, 61, (rows, 4, 1), generator=gen).float())
        return (x.view(rows, 4, 32) * scale).view(rows, DIM).bfloat16()
    if data == "midpoints":
        # Blocks whose amax is 6 * 2^e (scale exactly 2^e) holding every code edge and its sign.
        edges = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, 7.0])
        picks = edges[torch.randint(0, len(edges), (rows, 4, 32), generator=gen)]
        picks = picks * torch.where(torch.rand(rows, 4, 32, generator=gen) < 0.5, -1.0, 1.0)
        picks[..., 0] = 6.0
        picks = picks.clamp(-6.0, 6.0)
        scale = torch.pow(2.0, torch.randint(-8, 9, (rows, 4, 1), generator=gen).float())
        return (picks * scale).view(rows, DIM).bfloat16()
    raise ValueError(data)


def _halves(spec, gen: torch.Generator, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """``(pe, nope)``: the two 64-wide halves of one row-major tensor, as the indexer passes them."""
    rows = spec["tokens"] * (spec["heads"] or 1)
    x = _rows(rows, spec["data"], gen).to(device)
    if spec["heads"]:
        x = x.view(spec["tokens"], spec["heads"], DIM)
    return x[..., : spec["pe_dim"]], x[..., spec["pe_dim"] :]


def _independent(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The format derived without the reference's bit tricks: per block the smallest power of two
    ``s >= amax * fp32(1/6)`` (frexp), then the nearest E2M1 level of ``min(|x| / s, 6)`` with
    midpoints to the smaller level. Returns (codes uint8 [M, 128], exponent bytes uint8 [M, 4])."""
    x = x.reshape(-1, DIM).float()
    blocks = x.view(-1, 4, 32)
    ratio = blocks.abs().amax(-1).clamp_min(1e-12) * torch.tensor(1.0 / 6.0, dtype=torch.float32)
    mantissa, exponent = torch.frexp(ratio)  # ratio = mantissa * 2^exponent, mantissa in [0.5, 1)
    power = torch.where(mantissa == 0.5, exponent - 1, exponent)
    scale = torch.pow(2.0, power.float())
    levels = torch.tensor(E2M1_MAGNITUDES)
    mag = (blocks / scale[..., None]).abs().clamp_max(6.0)
    dist = (mag[..., None] - levels).abs()
    code = dist.argmin(-1)  # argmin takes the first (smaller) level on a tie
    negative = (blocks < 0) & (code != 0)
    codes = (code | (negative.long() << 3)).view(-1, DIM).to(torch.uint8)
    return codes, (power + 127).to(torch.uint8)


def _unpack(packed: torch.Tensor, scales: torch.Tensor):
    data = packed.view(torch.uint8)
    codes = torch.stack((data & 15, data >> 4), dim=-1).reshape(data.shape[0], DIM)
    return codes, scales.view(torch.uint8).reshape(-1, 4)


@cpu
@pytest.mark.parametrize("data", ["randn", "zeros_block", "wide", "midpoints"])
def test_reference_is_the_block_scaled_nearest_e2m1_code(data):
    gen = torch.Generator().manual_seed(len(data))
    x = _rows(257, data, gen)
    packed, scales = op.reference(x[:, :64], x[:, 64:])
    assert packed.dtype == torch.int8 and packed.shape == (257, 64)
    assert scales.dtype == torch.int32 and scales.shape == (257, 1)
    codes, exponents = _unpack(packed, scales)
    want_codes, want_exponents = _independent(x)
    assert torch.equal(exponents, want_exponents)
    assert torch.equal(codes, want_codes)


@cpu
def test_reference_pins_known_rows():
    # Block 0 has amax 6, so its scale is 1 (6 * fp32(1/6) rounds to exactly 1.0): the edges round
    # down, the values just above them round up, a negative value of code 0 has no sign bit.
    block = [6.0, 5.0, 5.03125, 0.25, 0.255859375, -0.25, -0.3, -6.0, 3.5, 3.6, 1.25, -1.3]
    x = torch.zeros(1, DIM)
    x[0, : len(block)] = torch.tensor(block)
    x[0, 32] = 3.0  # block 1: amax 3, 3 * fp32(1/6) rounds to 0.5: scale 2^-1, code 7
    # block 2 stays zero: the 1e-12 amax floor, scale 2^-42, exponent byte 85
    x[0, 96] = 100.0  # block 3: amax 100, scale 2^ceil(log2(16.67)) = 32, 100 / 32 -> code 5
    x = x.bfloat16()
    packed, scales = op.reference(x[:, :64], x[:, 64:])
    codes, exponents = _unpack(packed, scales)
    assert codes[0, : len(block)].tolist() == [7, 6, 7, 0, 1, 0, 9, 15, 5, 6, 2, 11]
    assert codes[0, 32].item() == 7 and codes[0, 96].item() == 5
    assert int(codes[0, 64:96].max()) == 0
    assert exponents[0].tolist() == [127, 126, 85, 132]


@cpu
def test_reference_packs_the_low_nibble_first_and_block_bytes_little_endian():
    x = torch.zeros(1, DIM)
    x[0, 0], x[0, 1] = 6.0, -6.0  # block 0 at scale 1: byte 0 = code 7 low, code 15 high
    x[0, 40] = 24.0  # block 1: scale 4
    packed, scales = op.reference(x[:, :64].bfloat16(), x[:, 64:].bfloat16())
    assert packed.view(torch.uint8)[0, 0].item() == 7 | (15 << 4)
    word = int(scales[0, 0].item()) & 0xFFFFFFFF
    assert [(word >> (8 * g)) & 0xFF for g in range(4)] == [127, 129, 85, 85]


@cpu
def test_the_decoder_returns_the_quantized_values():
    gen = torch.Generator().manual_seed(4)
    x = _rows(64, "wide", gen)
    packed, scales = op.reference(x[:, :64], x[:, 64:])
    codes, exponents = _independent(x)
    sign = 1.0 - 2.0 * ((codes >> 3) & 1).float()
    level = torch.tensor(E2M1_MAGNITUDES)[(codes & 7).long()]
    scale = torch.pow(2.0, exponents.float() - 127.0).repeat_interleave(32, dim=1)
    assert torch.equal(mxfp4_values(packed, scales), sign * level * scale)
    # The same rows as one int32 per row (no trailing 1) and as uint8 bytes decode alike.
    assert torch.equal(
        mxfp4_values(packed.view(torch.uint8), scales.view(-1)), sign * level * scale
    )


@cpu
def test_quantization_error_is_within_half_a_step():
    gen = torch.Generator().manual_seed(9)
    x = _rows(512, "randn", gen).float()
    packed, scales = op.reference(x[:, :64].bfloat16(), x[:, 64:].bfloat16())
    err = (mxfp4_values(packed, scales) - x).abs().view(-1, 4, 32)
    # The widest gap between E2M1 levels is 2 (4 to 6): at most one scale unit off, and the block
    # scale is at most amax / 3.
    amax = x.view(-1, 4, 32).abs().amax(-1, keepdim=True)
    assert bool((err <= amax / 3 + 1e-30).all())


@cpu
def test_guard_admits_row_major_halves():
    gen = torch.Generator().manual_seed(1)
    pe, nope = _halves(dict(tokens=5, heads=32, pe_dim=64, data="randn"), gen)
    op.is_valid(pe, nope)
    pe, nope = _halves(dict(tokens=5, heads=0, pe_dim=64, data="randn"), gen)
    op.is_valid(pe, nope)


@cpu
def test_guard_refuses_a_head_slice():
    x = torch.zeros(4, HEADS, DIM, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="collapse"):
        op.is_valid(x[:, :16, :64], x[:, :16, 64:])


@gpu
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    gen = torch.Generator().manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    pe, nope = _halves(cell.spec, gen, device="cuda")
    packed, scales = op(pe, nope)
    again = op(pe, nope)
    ref_packed, ref_scales = op.reference(pe, nope)
    torch.cuda.synchronize()
    rows = pe.numel() // pe.shape[-1]
    assert packed.shape == (rows, 64) and packed.dtype == torch.int8
    assert scales.shape == (rows, 1) and scales.dtype == torch.int32
    assert torch.equal(packed, again[0]) and torch.equal(scales, again[1])
    assert torch.equal(scales, ref_scales), "block exponents differ from the reference"
    assert torch.equal(packed, ref_packed), "E2M1 codes differ from the reference"


@gpu
def test_graph_replay_quantizes_the_new_rows():
    gen = torch.Generator().manual_seed(21)
    spec = dict(tokens=96, heads=HEADS, pe_dim=64, data="randn")
    first = torch.cat(_halves(spec, gen, device="cuda"), -1).contiguous()
    second = torch.cat(_halves(dict(spec, data="wide"), gen, device="cuda"), -1).contiguous()
    static = first.clone()
    op(static[..., :64], static[..., 64:])  # warm up outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        packed, scales = op(static[..., :64], static[..., 64:])
    static.copy_(second)
    graph.replay()
    want_packed, want_scales = op(second[..., :64], second[..., 64:])
    torch.cuda.synchronize()
    assert torch.equal(packed, want_packed) and torch.equal(scales, want_scales)

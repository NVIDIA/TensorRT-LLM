# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the rubin_helix_kept_gather entry: its reference against a scalar model of the
conversion (Python floats, ``struct`` for the fp16 / fp32 roundings, an explicit E4M3 encoder) and
a loop over the remap rules, and its guard."""

import math
import struct

import pytest
import torch

# l0_gb300 collects this directory: any GPU but sm_107 skips the module before a family import.
if torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("certified on sm_107 only", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_helix_kept_gather import (  # noqa: E402,E501
    HEAD_DIM,
    dequantize_rows,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_helix_kept_gather import (  # noqa: E402,E501
    rubin_helix_kept_gather as op,
)

pytestmark = pytest.mark.cpu_only

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FINITE = [b for b in range(256) if (b & 0x7F) != 0x7F]


def _f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


def _f16(x: float) -> float:
    try:
        return struct.unpack("e", struct.pack("e", x))[0]
    except OverflowError:  # past the largest finite fp16 after rounding
        return math.copysign(math.inf, x)


def _e4m3_value(byte: int) -> float:
    e, m = (byte >> 3) & 15, byte & 7
    mag = m * 2.0**-9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
    return -mag if byte & 0x80 else mag


def _e4m3_byte(x: float) -> int:
    """Round to nearest even, saturate to 448; the sign of zero is kept."""
    sign = 0x80 if math.copysign(1.0, x) < 0 else 0
    a = min(abs(x), 448.0)
    if a < 2.0**-6:  # subnormal step 2^-9; a carry to 8 is the smallest normal's encoding
        return sign | round(a / 2.0**-9)
    frac, exp = math.frexp(a)  # a = frac * 2^exp, frac in [0.5, 1)
    m = round((frac * 2 - 1) * 8)
    e = exp - 1
    if m == 8:
        e, m = e + 1, 0
    return sign | ((e + 7) << 3) | m


def _code(row_bytes, v: int) -> float:
    c = (row_bytes[v // 2] >> (4 * (v % 2))) & 15
    return -_E2M1[c & 7] if c & 8 else _E2M1[c]


def _scalar_row(data, scale, g: float, residual: int):
    """One token's 576 E4M3 bytes, value by value."""
    out = []
    for v in range(HEAD_DIM):
        if residual == 0 or v < 512:
            s16 = _f16(_f32(_e4m3_value(scale[v // 16]) * g))
            out.append(_e4m3_byte(_f16(_code(data, v) * s16)))
            continue
        k, j = divmod(v - 512, 16)
        main = _code(data[256 + 16 * k : 256 + 16 * k + 8], j)
        res = _code(data[256 + 16 * k + 8 : 256 + 16 * k + 16], j)
        a = _f32(_e4m3_value(scale[32 + 2 * k]) * g)
        b = _f32(_e4m3_value(scale[33 + 2 * k]) * g)
        out.append(_e4m3_byte(_f32(main * a + _f32(res * b))))
    return out


@pytest.mark.parametrize("residual", [0, 64])
@pytest.mark.parametrize("g", [100.0 / (448.0 * 6.0), 4.0, 2.0**-12])
def test_dequantization_matches_the_scalar_model(residual, g):
    """Random rows and every finite scale byte; G large enough to saturate, small enough to
    underflow into fp16 and E4M3 subnormals."""
    gen = torch.Generator().manual_seed(residual + int(g * 1000))
    n = 40
    data = torch.randint(0, 256, (n, (HEAD_DIM + residual) // 2), generator=gen).to(torch.uint8)
    pick = torch.randint(0, len(_FINITE), (n, (HEAD_DIM + residual) // 16), generator=gen)
    scale = torch.tensor(_FINITE, dtype=torch.uint8)[pick]
    scale.view(-1)[: len(_FINITE)] = torch.tensor(_FINITE, dtype=torch.uint8)
    g32 = torch.tensor(g, dtype=torch.float32)
    got = dequantize_rows(data, scale, g32, residual)
    assert got.dtype == torch.float8_e4m3fn and got.shape == (n, HEAD_DIM)
    want = [
        _scalar_row(data[i].tolist(), scale[i].tolist(), float(g32), residual) for i in range(n)
    ]
    assert got.view(torch.uint8).tolist() == want


def _case(rows=6, top_k=512, num_req=3, tpb=16, num_layers=2, layer=1, residual=0, seed=0):
    gen = torch.Generator().manual_seed(seed)
    pages = 8
    num_blocks = num_req * pages + 4
    num_pool_tokens = num_blocks * num_layers * tpb
    table = torch.randperm(num_blocks, generator=gen)[: num_req * pages].view(num_req, pages)
    table = table.to(torch.int32).contiguous()
    table[0, 2] = -1  # a padded table entry
    table[1, 3] = num_blocks * 4  # a page past the pool
    topk = torch.full((rows, top_k), -1, dtype=torch.int32)
    lens = [0, 5, 16, 17, top_k, top_k + 9][:rows]
    for r, n in enumerate(lens):
        n = min(n, pages * tpb)
        topk[r, :n] = torch.randperm(pages * tpb, generator=gen)[:n].to(torch.int32)
    topk[1, 1] = -1  # a hole inside the row
    topk[2, 0] = pages * tpb + 3  # a block id past the table
    topk[3, 4] = 2 * tpb + 1  # row 3 is request 0: its page 2 is the padded entry
    data_bytes, scale_bytes = (HEAD_DIM + residual) // 2, (HEAD_DIM + residual) // 16
    return dict(
        topk_indices=topk,
        row_lens=torch.tensor(lens, dtype=torch.int32),
        req_idx=torch.tensor([r % num_req for r in range(rows)], dtype=torch.int32),
        block_table=table,
        tokens_per_block=tpb,
        stride_factor=num_layers * tpb,
        layer_offset=layer,
        data_pool=torch.randint(0, 256, (num_pool_tokens * data_bytes,), generator=gen).to(
            torch.uint8
        ),
        scale_pool=torch.randint(0, 0x7F, (num_pool_tokens * scale_bytes,), generator=gen).to(
            torch.uint8
        ),
        dequant_scale=torch.tensor([100.0 / (448.0 * 6.0)], dtype=torch.float32),
        num_pool_tokens=num_pool_tokens,
        scratch=torch.full((rows + 1, top_k, HEAD_DIM), 0x5A, dtype=torch.uint8).view(
            torch.float8_e4m3fn
        ),
        compact_indices=torch.full((rows, top_k), 12345, dtype=torch.int32),
        residual_dim=residual,
    )


def _remap_loop(a):
    """(compact, pool token or None) per (row, column), from the rules in the entry's docstring."""
    topk, table = a["topk_indices"].tolist(), a["block_table"].tolist()
    tpb = a["tokens_per_block"]
    rows, top_k = a["topk_indices"].shape
    out = {}
    for r in range(rows):
        n = max(0, min(int(a["row_lens"][r]), top_k))
        req = int(a["req_idx"][r])
        for c in range(top_k):
            t = topk[r][c]
            g = None
            if c < n and t >= 0 and t // tpb < len(table[req]):
                b = table[req][t // tpb]
                cand = b * a["stride_factor"] + t % tpb + a["layer_offset"] * tpb
                if b >= 0 and 0 <= cand < a["num_pool_tokens"]:
                    g = cand
            out[r, c] = (r * top_k + c if g is not None else -1, g)
    return out


def test_reference_remaps_and_gathers_by_the_rules():
    a = _case(seed=1)
    scratch_in, compact_in = a["scratch"].clone(), a["compact_indices"].clone()
    scratch, compact = op.reference(**a)
    # The inputs are left alone.
    assert torch.equal(a["scratch"].view(torch.uint8), scratch_in.view(torch.uint8))
    assert torch.equal(a["compact_indices"], compact_in)
    assert compact.dtype == torch.int32 and scratch.dtype == torch.float8_e4m3fn
    rules = _remap_loop(a)
    g = float(a["dequant_scale"][0])
    data, scale = a["data_pool"].view(-1, HEAD_DIM // 2), a["scale_pool"].view(-1, HEAD_DIM // 16)
    got = scratch.view(torch.uint8)
    written = 0
    for (r, c), (want, token) in rules.items():
        assert int(compact[r, c]) == want, (r, c)
        if token is None:
            assert bool((got[r, c] == 0x5A).all()), f"({r}, {c}) must not be written"
        else:
            row = _scalar_row(data[token].tolist(), scale[token].tolist(), g, 0)
            assert got[r, c].tolist() == row, (r, c)
            written += 1
    assert written > 0
    assert bool((got[-1] == 0x5A).all()), "the spare scratch row must not be written"


def test_guard_admits_a_well_formed_call():
    op.is_valid(**_case())


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda a: dict(a, scratch=a["scratch"].view(torch.uint8)), "E4M3"),
        (lambda a: dict(a, compact_indices=a["compact_indices"][:-1]), "compact_indices"),
        (lambda a: dict(a, compact_indices=a["compact_indices"].long()), "compact_indices"),
        (lambda a: dict(a, req_idx=a["req_idx"] + 5), "block table"),
        (lambda a: dict(a, row_lens=a["row_lens"][:2]), "one length"),
        (lambda a: dict(a, dequant_scale=torch.ones(1)), "other than 1"),
        (lambda a: dict(a, dequant_scale=a["dequant_scale"].double()), "fp32"),
        (lambda a: dict(a, stride_factor=2**30), "int32"),
    ],
    ids=["scratch", "short compact", "compact dtype", "req", "lens", "G == 1", "G dtype", "int32"],
)
def test_guard_refuses(change, match):
    with pytest.raises(AssertionError, match=match):
        op.is_valid(**change(_case()))


def test_the_cells_cover_both_nvfp4_latent_layouts():
    assert {0, 64} == {c.spec["residual_dim"] for c in op.CELLS}

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The rubin_megamoe_quantize entry: on a sample of its cells it writes fp4_quantize's bytes into a
MegaMoE bucket and empties the bucket's routing past the rows, on SM107."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__dep16.ops.moe.rubin_megamoe_quantize import (  # noqa: E501
    rubin_megamoe_quantize,
)

__extra_import_path__ = ["../../../../..", "../../.."]
from _k31_utils import cpu_or_sm, on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401 -- path above; fixture by name

pytestmark = cpu_or_sm(107)
gpu = on_sm(107)

HIDDEN, TOP_K = 3584, 16
# What the bucket holds before a call: bytes no quantization writes, and routing that still routes.
STALE_ACT, STALE_SF, STALE_ID, STALE_WEIGHT = 0xAB, 0xCD, 7, 3.0


def _x(rows, seed, mode="normal", device="cpu"):
    """Latent-like BF16 rows; ``edges``: all-zero blocks, tiny blocks (their scale underflows to
    0) and large ones (their scale saturates at 448) in every row."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(rows, HIDDEN, generator=g) * 0.15
    if mode == "edges":
        blocks = x.view(rows, HIDDEN // 16, 16)
        blocks[:, 0::7] = 0.0
        blocks[:, 1::7] *= 1e-30
        blocks[:, 2::7] *= 1e4
    return x.to(torch.bfloat16).to(device)


def _bucket(t, device):
    return (
        torch.full((t, HIDDEN // 2), STALE_ACT, dtype=torch.uint8, device=device),
        torch.full((t, HIDDEN // 16), STALE_SF, dtype=torch.uint8, device=device),
        torch.full((t, TOP_K), STALE_ID, dtype=torch.int32, device=device),
        torch.full((t, TOP_K), STALE_WEIGHT, device=device),
    )


def _unpack(packed):
    """[M, K/2] packed bytes -> [M, K] codes (element 2i in the low nibble)."""
    m, half = packed.shape
    codes = torch.empty(m, half * 2, dtype=torch.uint8, device=packed.device)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = packed >> 4
    return codes


def _today(x, gs):
    """fp4_quantize as dep16's staging path calls it: values and linear scales [n, hidden / 16]."""
    from tensorrt_llm._torch.modeling_v2.catalog.quantization.fp4_quantize import fp4_quantize

    x4, sf = fp4_quantize(x, gs, 16, False, False)
    return x4.view(torch.uint8), sf.view(torch.uint8).view(x.shape[0], HIDDEN // 16)


def _check_bucket(n, x, gs, act, sf, ids, weights):
    want_x4, want_sf = _today(x, gs)
    assert torch.equal(act[:n], want_x4), "values are not fp4_quantize's bits"
    assert torch.equal(sf[:n], want_sf), "block scales are not fp4_quantize's bits"
    assert bool((act[n:] == STALE_ACT).all()) and bool((sf[n:] == STALE_SF).all()), (
        "rows past the input were written"
    )
    assert bool((ids[:n] == STALE_ID).all()) and bool((weights[:n] == STALE_WEIGHT).all()), (
        "the routing of the rows was written"
    )
    assert bool((ids[n:] == -1).all()) and not weights[n:].any(), "the routing tail is not empty"


CELLS = sample_cells(rubin_megamoe_quantize.CELLS, size_key="rows")[:2]


@gpu
@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 -- fixture resolved by name
    from tensorrt_llm._torch.modeling_v2.catalog.quantization.fp4_quantize import fp4_quantize

    certifying(rubin_megamoe_quantize)
    spec = cell.spec
    n, t = spec["rows"], spec["bucket"]
    x = _x(n, zlib.crc32(cell.why.encode()), spec.get("mode", "normal"), "cuda")
    gs = torch.tensor([spec.get("scale", 1.0)], device="cuda")
    act, sf, ids, weights = _bucket(t, "cuda")
    rubin_megamoe_quantize(x, gs, act, sf, ids, weights)
    torch.cuda.synchronize()
    _check_bucket(n, x, gs, act, sf, ids, weights)
    # The catalog entry's own rule against its reference: scales exact, codes exact away from the
    # rounding ties its approximate reciprocals decide.
    ref_x4, ref_sf = rubin_megamoe_quantize.reference(x, gs, act, sf, ids, weights)
    assert torch.equal(sf[:n], ref_sf), "block scales differ from the reference"
    tie = fp4_quantize.near_tie(fp4_quantize.scaled(x, gs, 16))
    codes, ref_codes = _unpack(act[:n]), _unpack(ref_x4)
    assert torch.equal(codes[~tie], ref_codes[~tie]), "codes differ away from a rounding tie"

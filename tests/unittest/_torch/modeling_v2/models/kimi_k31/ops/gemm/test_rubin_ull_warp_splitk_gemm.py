# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's warp_splitk_gemm entry: the DSA in-projection's prepared engine at
every cell's row count against the FP64 product, and its output tiles 16 and 32 bit for bit."""

import dataclasses
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1.modeling import (  # noqa: E501
    HIDDEN,
    PITCH,
    DsaInProjection,
    _warp,
    smem_limits,
    warp_tactic,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_ull_warp_splitk_gemm import (  # noqa: E501
    warp_splitk_gemm as op,
)

__extra_import_path__ = ["../../../..", "."]
from _bf16_band import assert_bf16_gemm_close  # noqa: E402 — needs the path declared above
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the warp split-K kernel needs an SM10x GPU",
)

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)


@pytest.fixture(scope="module")
def projection():
    g = torch.Generator(device="cuda").manual_seed(7)
    weight = (torch.randn(PITCH, HIDDEN, generator=g, device="cuda") * HIDDEN**-0.5).to(
        torch.bfloat16
    )
    proj = DsaInProjection(weight)
    proj.prepare(c.spec["rows"] for c in CELLS)
    return proj


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, projection) -> None:
    rows = cell.spec["rows"]
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    x = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    out = torch.full((rows, PITCH), float("nan"), dtype=torch.bfloat16, device="cuda")
    engine = projection.engine
    with validating(op):
        result = op(engine, x, out)
    torch.cuda.synchronize()
    assert result is out
    assert_bf16_gemm_close(out, op.reference(engine, x, out), x, engine.weight)


def test_is_valid_refuses_a_strided_x(projection) -> None:
    wide = torch.zeros(8, 2 * HIDDEN, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(8, PITCH, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="packed"):
        op.is_valid(projection.engine, wide[:, :HIDDEN], out)


# (SMs, per-CTA opt-in smem, per-SM smem): Rubin (sm_107) and GB300 (sm_103).
RUBIN, GB300 = (212, 232448, 233472), (152, 232448, 233472)


@pytest.mark.parametrize("rows", [3, 8, 12, 16])
def test_warp_tactic_output_tile(rows) -> None:
    """Output tile 16 where its PITCH // 16 CTAs fit one per SM (Rubin), vLLM's 32 elsewhere."""
    w = _warp()
    for (sms, cta, sm), tile in ((RUBIN, 16), (GB300, 32)):
        t = warp_tactic(rows, sms, cta, sm)
        assert t is not None and t.output_tile == tile
        assert w._smem_bytes(t) <= min(cta, w._SMEM_CAPACITY)
        deeper = dataclasses.replace(t, stages=t.stages + 1)
        assert t.stages == w._MAX_STAGES or w._smem_bytes(deeper) > min(cta, w._SMEM_CAPACITY)


@pytest.mark.parametrize("rows", [3, 8, 12, 16])
def test_output_tile_16_matches_32_bitwise(rows, projection) -> None:
    """The served tactics of both devices on this one: every output bit is the same whichever tile
    width (and ring depth) computes it. Run in waves where 190 CTAs exceed the SMs."""
    w = _warp()
    cta, sm = smem_limits(projection.weight.device)
    tactics = {
        tile: warp_tactic(rows, sms, min(cta, c), min(sm, s))
        for tile, (sms, c, s) in ((16, RUBIN), (32, GB300))
    }
    assert {t: tactics[t].output_tile for t in tactics} == {16: 16, 32: 32}
    engines = {
        t: w.PreparedWarpSplitK(projection.weight, {rows: tac}) for t, tac in tactics.items()
    }
    g = torch.Generator(device="cuda").manual_seed(1000 + rows)
    for scale in (1.0, 64.0):
        x = (torch.randn(rows, HIDDEN, generator=g, device="cuda") * scale).to(torch.bfloat16)
        x[:, ::97] *= 1024  # wide dynamic range: any change of the K order shows in the bits
        outs = {}
        for t, e in engines.items():
            outs[t] = torch.full((rows, PITCH), float("nan"), dtype=torch.bfloat16, device="cuda")
            e.run(x, outs[t])
        torch.cuda.synchronize()
        assert not outs[16].isnan().any()
        assert torch.equal(outs[16].view(torch.int16), outs[32].view(torch.int16))

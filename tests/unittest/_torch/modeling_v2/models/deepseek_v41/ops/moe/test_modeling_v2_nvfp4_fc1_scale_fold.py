# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One FC1 global weight scale for gate + up (`nvfp4_fc1_scale_fold`), and the MegaMoE packing that uses it, on CPU.

An expert whose `w1` and `w3` `weight_scale_2` differ shares the LARGER one; the other half's e4m3 block scales are
scaled down onto it. Proved: an amax block scale (448) stays finite whichever half is larger (multiplying it up by
`ws3 / ws1 > 1.036` would round past e4m3's range, and the unsaturated cast gives NaN), each half's dequantized scales
(block scale x the shared global scale) equal the checkpoint's own (block scale x its own global scale) within one e4m3
rounding, and an expert whose two scales are equal keeps its bytes."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.megamoe_pack_experts import (
    megamoe_expert_scalars,
    megamoe_pack_experts,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.nvfp4_fc1_scale_fold import (
    fc1_global_scale,
    fold_block_scales,
)

pytestmark = pytest.mark.cpu_only

E4M3 = torch.float8_e4m3fn
RTOL = (
    2.0**-4 * 1.01
)  # one round-to-nearest of a normal e4m3 value (3 mantissa bits), plus float32 slack


def _block_scales(shape, seed):
    """Normal e4m3 block scales in [0.25, 448], the first block of every row the amax one (448)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.exp2(torch.rand(shape, generator=g) * (8.807 - (-2.0)) - 2.0).clamp(max=448.0)
    x[..., 0] = 448.0
    return x.to(E4M3)


# (ws1, ws3): up larger by 1.5 (the overflow case), gate larger by 1.5, a large ratio, equal
PAIRS = [(0.01, 0.015), (0.015, 0.01), (0.002, 0.03), (0.01, 0.01)]


@pytest.mark.parametrize("ws1,ws3", PAIRS, ids=["up-larger", "gate-larger", "ratio-15", "equal"])
def test_the_larger_scale_is_shared_and_the_other_half_scaled_down(ws1, ws3):
    w1, w3 = torch.tensor(ws1), torch.tensor(ws3)
    ws13, f1, f3 = fc1_global_scale(w1, w3)
    assert torch.equal(ws13, torch.maximum(w1, w3))
    assert float(f1) <= 1.0 and float(f3) <= 1.0 and max(float(f1), float(f3)) == 1.0
    s1, s3 = _block_scales((64, 32), 1), _block_scales((64, 32), 2)
    for sf, fold, own in ((s1, f1, w1), (s3, f3, w3)):
        folded = fold_block_scales(sf, fold)
        assert folded.dtype == E4M3 and folded.shape == sf.shape
        assert torch.isfinite(folded.float()).all()
        torch.testing.assert_close(
            folded.float() * ws13, sf.float() * own.float(), rtol=RTOL, atol=0
        )
        if float(fold) == 1.0:
            assert torch.equal(folded.view(torch.uint8), sf.view(torch.uint8))


def test_fold_block_scales_never_gives_nan():
    """Even a factor > 1 (which `fc1_global_scale` never produces) saturates instead of turning 448 into NaN."""
    sf = _block_scales((4, 16), 3)
    out = fold_block_scales(sf, torch.tensor(1.5))
    assert torch.isfinite(out.float()).all() and float(out.float().max()) == 448.0


def test_a_per_slot_fold_scales_each_slot_by_its_own_factor():
    sf = _block_scales((3, 8, 16), 4)
    fold = torch.tensor([1.0, 0.5, 0.25])
    out = fold_block_scales(sf.view(torch.uint8), fold)
    for s in range(3):
        want = fold_block_scales(sf[s], fold[s])
        assert torch.equal(out[s].view(torch.uint8), want.view(torch.uint8))
    assert torch.equal(out[0].view(torch.uint8), sf[0].view(torch.uint8))


def _unpack_fc1_sf(fc1_sf: torch.Tensor, inter: int, cols: int):
    """Invert `megamoe_pack_experts`'s FC1 scale layout -> (gate, up) `[S, I, cols]` e4m3: the 128x4 atoms by the
    destination offset its reference documents, then the 16-row gate / up interleave."""
    s, rows = fc1_sf.shape[0], 2 * inter
    col_tiles = -(-cols // 4)
    r = torch.arange(rows)[:, None]
    c = torch.arange(cols)[None, :]
    dest = (((r // 128) * col_tiles + c // 4) * 32 + r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4
    inter_sf = fc1_sf[:, dest.reshape(-1)].reshape(s, inter // 16, 2, 16, cols)
    gate = inter_sf[:, :, 0].reshape(s, inter, cols).view(E4M3)
    up = inter_sf[:, :, 1].reshape(s, inter, cols).view(E4M3)
    return gate, up


def test_megamoe_packs_both_halves_onto_the_larger_scale():
    """`megamoe_expert_scalars` + `megamoe_pack_experts`, one slot per pair of `PAIRS`: the alpha is built from the
    larger scale, the packed scales are finite and dequantize to the checkpoint's, equal scales keep the bytes."""
    slots, inter, hidden = len(PAIRS), 128, 256
    cols = hidden // 16
    g = torch.Generator().manual_seed(5)
    w1 = torch.randint(0, 256, (slots, inter, hidden // 2), generator=g, dtype=torch.uint8)
    w3 = torch.randint(0, 256, (slots, inter, hidden // 2), generator=g, dtype=torch.uint8)
    w2 = torch.randint(0, 256, (slots, hidden, inter // 2), generator=g, dtype=torch.uint8)
    s1 = _block_scales((slots, inter, cols), 6)
    s3 = _block_scales((slots, inter, cols), 7)
    s2 = _block_scales((slots, hidden, inter // 16), 8)
    ws1 = torch.tensor([p[0] for p in PAIRS])
    ws3 = torch.tensor([p[1] for p in PAIRS])
    ws2 = torch.full((slots,), 0.01)
    isc1, isc2 = torch.tensor(0.0017), torch.full((slots,), 0.003)

    sc = megamoe_expert_scalars(isc1, isc2, ws1, ws3, ws2)
    ws13 = torch.maximum(ws1, ws3)
    torch.testing.assert_close(sc["fc1_alpha"], isc1 * ws13, rtol=1e-6, atol=0)
    fc1, fc1_sf, _, _ = megamoe_pack_experts(
        w1, s1.view(torch.uint8), w3, s3.view(torch.uint8), w2, s2.view(torch.uint8),
        w3_scale_fold=sc["w3_scale_fold"], w1_scale_fold=sc["w1_scale_fold"],
    )  # fmt: skip
    gate, up = _unpack_fc1_sf(fc1_sf, inter, cols)
    for s in range(slots):
        for got, orig, own in ((gate[s], s1[s], ws1[s]), (up[s], s3[s], ws3[s])):
            assert torch.isfinite(got.float()).all(), PAIRS[s]
            torch.testing.assert_close(got.float() * ws13[s], orig.float() * own, rtol=RTOL, atol=0)
        if ws1[s] == ws3[s]:
            assert torch.equal(gate[s].view(torch.uint8), s1[s].view(torch.uint8))
            assert torch.equal(up[s].view(torch.uint8), s3[s].view(torch.uint8))
    # the codes are only moved
    ref = megamoe_pack_experts.reference(
        w1, s1.view(torch.uint8), w3, s3.view(torch.uint8), w2, s2.view(torch.uint8),
        sc["w3_scale_fold"], sc["w1_scale_fold"],
    )  # fmt: skip
    assert torch.equal(fc1, ref[0]) and torch.equal(fc1_sf, ref[1])

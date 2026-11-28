# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the megamoe_pack_experts entry, bit for bit against the engine's loader and the reference."""

import pytest
import torch

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.megamoe_pack_experts import (  # noqa: E402
    megamoe_expert_scalars,
    megamoe_pack_experts,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
DEV = "cuda"


def _stacks(s: int, inter: int, hidden: int, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)

    def one(n, k):
        w = torch.randint(0, 256, (s, n, k // 2), dtype=torch.uint8, device=DEV, generator=g)
        sf = torch.randint(0, 256, (s, n, k // 16), dtype=torch.uint8, device=DEV, generator=g)
        return w, sf  # arbitrary bytes: a layout transform must not care what they mean

    return one(inter, hidden), one(inter, hidden), one(hidden, inter)


@pytest.mark.parametrize(
    "cell", megamoe_pack_experts.CELLS, ids=[c.why[:44] for c in megamoe_pack_experts.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811
    """Every output of the op equals the reference's, byte for byte, with a scale fold on every slot."""
    certifying(megamoe_pack_experts)
    sp = cell.spec
    (w1, s1), (w3, s3), (w2, s2) = _stacks(sp["slots"], sp["inter"], sp["hidden"], 21)
    s3 = torch.randint(
        0x20, 0x50, s3.shape, dtype=torch.uint8, device=DEV
    )  # normal, small e4m3: the fold is visible
    s1f = torch.randint(0x20, 0x50, s1.shape, dtype=torch.uint8, device=DEV)
    fold = torch.rand(sp["slots"], device=DEV) + 0.5
    fold1 = torch.rand(sp["slots"], device=DEV) + 0.5
    for args in (
        (w1, s1, w3, s3, w2, s2),
        (w1, s1, w3, s3, w2, s2, fold),
        (w1, s1f, w3, s3, w2, s2, fold, fold1),
    ):
        got = megamoe_pack_experts(*args)
        ref = megamoe_pack_experts.reference(*args)
        for name, mine, want in zip(("fc1", "fc1_sf", "fc2", "fc2_sf"), got, ref):
            assert mine.shape == want.shape and mine.dtype == want.dtype, name
            assert torch.equal(mine, want), f"{name} differs from the reference"


def test_rejections():
    (w1, s1), (w3, s3), (w2, s2) = _stacks(2, 256, 512, 3)
    megamoe_pack_experts.is_valid(w1, s1, w3, s3, w2, s2)
    with pytest.raises(AssertionError, match="16 rows"):
        megamoe_pack_experts.is_valid(w1[:, :250], s1[:, :250], w3[:, :250], s3[:, :250], w2, s2)
    with pytest.raises(AssertionError, match="H/16"):
        megamoe_pack_experts.is_valid(w1, s1[:, :, :-1], w3, s3[:, :, :-1], w2, s2)
    with pytest.raises(AssertionError, match="per slot"):
        megamoe_pack_experts.is_valid(w1, s1, w3, s3, w2, s2, torch.ones(3, device=DEV))


@pytest.mark.parametrize("s,inter,hidden", [(3, 2304, 5120), (35, 256, 512), (1, 128, 256)])
def test_matches_the_engines_mega_format_transform_bit_for_bit(s, inter, hidden):
    from tensorrt_llm._torch.moe.fused_moe.quantization import NVFP4MegaMoECuteDslMethod

    (w1, s1), (w3, s3), (w2, s2) = _stacks(s, inter, hidden, 11)
    fc1, fc1_sf, fc2, fc2_sf = megamoe_pack_experts(w1, s1, w3, s3, w2, s2)

    method = NVFP4MegaMoECuteDslMethod()
    raw_w3_w1 = torch.cat([w3, w1], dim=1)  # the engine stores [up | gate]
    raw_sf = torch.cat([s3, s1], dim=1)
    # the engine hands the scales as int32 words (4 e4m3 scales along K)
    ref = method._build_mega_format_buffers(
        raw_w3_w1,
        raw_sf.view(torch.int32),
        w2,
        s2.view(torch.int32),
        num_slots=s,
        intermediate=inter,
        hidden=hidden,
        expand_intermediate=2 * inter,
    )
    for name, mine, theirs in zip(
        ("fc1", "fc1_sf", "fc2", "fc2_sf"), (fc1, fc1_sf, fc2, fc2_sf), ref
    ):
        assert mine.shape == theirs.shape and mine.dtype == theirs.dtype, name
        assert torch.equal(mine, theirs), f"{name} differs from the engine's layout"


def test_scale_fold_multiplies_the_up_block_scales_in_e4m3():
    (w1, s1), (w3, s3), (w2, s2) = _stacks(2, 256, 512, 5)
    s3 = torch.randint(0x20, 0x50, s3.shape, dtype=torch.uint8, device=DEV)  # normal, small e4m3
    fold = torch.tensor([2.0, 0.5], device=DEV)
    folded = megamoe_pack_experts(w1, s1, w3, s3, w2, s2, fold)
    s3_scaled = (s3.view(torch.float8_e4m3fn).float() * fold[:, None, None]).to(torch.float8_e4m3fn)
    expected = megamoe_pack_experts(w1, s1, w3, s3_scaled.view(torch.uint8), w2, s2)
    assert all(torch.equal(a, b) for a, b in zip(folded, expected))
    assert torch.equal(folded[2], w2) and folded[2].data_ptr() != w2.data_ptr(), (
        "fc2 is a copy of w2"
    )


def test_scalars_follow_the_nvfp4_loader_convention():
    """alpha = weight_scale_2 * input_scale, bit for bit as the engine's loader rounds it; norm const = 1 / the
    expert's own FC2 input_scale."""
    from tensorrt_llm._torch.moe.fused_moe.quantization import NVFP4MegaMoECuteDslMethod

    g = torch.Generator().manual_seed(1)
    isc1 = torch.tensor(0.0017)
    isc2, ws1, ws3, ws2 = (torch.rand(64, generator=g) * 0.1 + 0.01 for _ in range(4))
    out = megamoe_expert_scalars(isc1, isc2, ws1, ws3, ws2)
    assert not torch.equal(out["fc2_alpha"], isc2 * ws2), (
        "the plain product rounds differently somewhere"
    )
    method = NVFP4MegaMoECuteDslMethod()
    for e in range(64):
        # the engine's loader keeps reciprocals: alpha = 1 / (g * (1 / ws2)) with g = 1 / input_scale
        alpha1, alpha2 = torch.empty(()), torch.empty(())
        ws13 = torch.maximum(
            ws1[e], ws3[e]
        )  # one FC1 weight scalar, the larger (stock's CUTLASS choice)
        method.load_expert_fc31_alpha_nvfp4(ws13, ws13, isc1.reciprocal(), alpha1)
        method.load_expert_fc2_alpha_nvfp4(ws2[e], isc2[e].reciprocal(), alpha2)
        assert torch.equal(out["fc1_alpha"][e], alpha1), (out["fc1_alpha"][e], alpha1)
        assert torch.equal(out["fc2_alpha"][e], alpha2), (out["fc2_alpha"][e], alpha2)
    assert torch.equal(out["fc1_norm_const"], isc2.reciprocal())
    assert torch.equal(out["input_global_scale"], isc1.reciprocal().reshape(1))
    ws13 = torch.maximum(ws1, ws3)
    torch.testing.assert_close(out["w3_scale_fold"], ws3 / ws13)
    torch.testing.assert_close(out["w1_scale_fold"], ws1 / ws13)
    assert bool((out["w1_scale_fold"] <= 1).all()) and bool((out["w3_scale_fold"] <= 1).all())
    assert all(v.dtype == torch.float32 and v.is_contiguous() for v in out.values())

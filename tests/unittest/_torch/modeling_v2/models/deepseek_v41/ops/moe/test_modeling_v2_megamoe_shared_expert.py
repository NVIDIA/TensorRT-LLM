# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the megamoe_shared_expert helpers: the MXFP8 weight packing and the shared expert's input buffers (the
routing launch's MXFP8 of the input is checked in `test_modeling_v2_dsv4_route_nvfp4_quant.py`)."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.megamoe_shared_expert import (
    megamoe_pack_shared_expert,
    megamoe_shared_activation_buffers,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
DEV = "cuda"
E4M3 = torch.float8_e4m3fn


def _proj(g, n, k, block):
    w = (torch.randn(n, k, device=DEV, generator=g) * 2).clamp(-400, 400).to(E4M3)
    rb, cb = block
    s = torch.randint(
        118, 128, (-(-n // rb), -(-k // cb)), device=DEV, generator=g, dtype=torch.int32
    )
    return w, s.to(torch.uint8)


def _dequant(w, s, block):
    rb, cb = block
    scale = torch.exp2(s.float() - 127.0).repeat_interleave(rb, 0).repeat_interleave(cb, 1)
    return w.float() * scale[: w.shape[0], : w.shape[1]]


def _unblock(flat, rows, cols):
    """Undo the 128x4 atom layout of per-row, per-32 scale bytes."""
    r, c = -(-rows // 128) * 128, -(-cols // 4) * 4
    atoms = flat.view(r // 128, c // 4, 32, 4, 4).permute(0, 3, 2, 1, 4)
    return atoms.reshape(r, c)[:rows, :cols]


@pytest.mark.parametrize("block", [(32, 32), (128, 128)])
def test_packing_preserves_every_dequantized_weight(block):
    inter, hidden = 1024, 1024
    g = torch.Generator(device=DEV).manual_seed(4)
    w1, s1 = _proj(g, inter, hidden, block)
    w3, s3 = _proj(g, inter, hidden, block)
    w2, s2 = _proj(g, hidden, inter, block)
    out = megamoe_pack_shared_expert(w1, s1, w3, s3, w2, s2, 10.0, block)
    assert out["intermediate_size"] == inter and out["gate_up_clamp"] == 10.0
    # fc1: gate and up interleaved by 16 rows, each row's per-32 scales in 128x4 atoms
    fc1 = out["fc1_weight"].view(inter // 16, 2, 16, hidden)
    assert torch.equal(fc1[:, 0].reshape(inter, hidden).view(torch.uint8), w1.view(torch.uint8))
    assert torch.equal(fc1[:, 1].reshape(inter, hidden).view(torch.uint8), w3.view(torch.uint8))
    sf = _unblock(out["fc1_weight_sf"], 2 * inter, hidden // 32).view(
        inter // 16, 2, 16, hidden // 32
    )
    for half, (w, s) in enumerate(((w1, s1), (w3, s3))):
        got = torch.exp2(sf[:, half].reshape(inter, hidden // 32).float() - 127.0)
        want = torch.exp2(s.float() - 127.0).repeat_interleave(block[0], 0)[:inter]
        want = want.repeat_interleave(block[1] // 32, 1)[:, : hidden // 32]
        assert torch.equal(got, want)
    assert torch.equal(out["fc2_weight"].view(torch.uint8), w2.view(torch.uint8))
    sf2 = _unblock(out["fc2_weight_sf"], hidden, inter // 32).float()
    want2 = torch.exp2(s2.float() - 127.0).repeat_interleave(block[0], 0)[:hidden]
    want2 = want2.repeat_interleave(block[1] // 32, 1)[:, : inter // 32]
    assert torch.equal(torch.exp2(sf2 - 127.0), want2)


def test_matches_the_engines_packing_of_a_128_block_shared_expert():
    from tensorrt_llm._torch.moe.fused_moe.mega_moe.shared_expert import build_shared_expert_weights

    inter, hidden = 1024, 1024
    g = torch.Generator(device=DEV).manual_seed(8)
    w1, s1 = _proj(g, inter, hidden, (128, 128))
    w3, s3 = _proj(g, inter, hidden, (128, 128))
    w2, s2 = _proj(g, hidden, inter, (128, 128))
    pow2 = lambda s: torch.exp2(s.float() - 127.0)  # noqa: E731
    gate_up = SimpleNamespace(
        weight=torch.cat([w1, w3], 0), weight_scale=torch.cat([pow2(s1), pow2(s3)], 0), bias=None
    )
    down = SimpleNamespace(weight=w2, weight_scale=pow2(s2), bias=None)
    mlp = SimpleNamespace(
        gate_up_proj=gate_up, down_proj=down, swiglu_alpha=None, swiglu_beta=None,
        activation=F.silu, swiglu_limit=10.0,
    )  # fmt: skip
    theirs = build_shared_expert_weights(mlp, hidden)
    assert theirs is not None
    mine = megamoe_pack_shared_expert(w1, s1, w3, s3, w2, s2, 10.0, (128, 128))
    for key in ("fc1_weight", "fc1_weight_sf", "fc2_weight", "fc2_weight_sf"):
        assert torch.equal(mine[key].view(torch.uint8), theirs[key].view(torch.uint8)), key
    assert mine["intermediate_size"] == theirs["intermediate_size"]
    assert mine["gate_up_clamp"] == theirs["gate_up_clamp"]


def test_unfusable_shapes_and_limits():
    g = torch.Generator(device=DEV).manual_seed(2)
    w1, s1 = _proj(g, 1024, 1000, (32, 32))  # hidden not a multiple of 512
    w2, s2 = _proj(g, 1000, 1024, (32, 32))
    with pytest.raises(ValueError):
        megamoe_pack_shared_expert(w1, s1, w1, s1, w2, s2, 10.0)
    w1, s1 = _proj(g, 1024, 1024, (32, 32))
    w2, s2 = _proj(g, 1024, 1024, (32, 32))
    assert megamoe_pack_shared_expert(w1, s1, w1, s1, w2, s2, 0.0)["gate_up_clamp"] is None
    assert megamoe_pack_shared_expert(w1, s1, w1, s1, w2, s2, None)["gate_up_clamp"] is None


@pytest.mark.parametrize("block", [(32, 48), (32, 16), (0, 32), (32, 0)])
def test_scale_blocks_that_do_not_map_to_32_column_groups_are_refused(block):
    """A 48-column block would give column 32 the second exponent (it belongs to the first block): refused, as are
    blocks narrower than 32 columns and non-positive ones; 32 and 128 are the packing test's."""
    g = torch.Generator(device=DEV).manual_seed(5)
    w1, s1 = _proj(g, 1024, 1024, (32, 32))
    w2, s2 = _proj(g, 1024, 1024, (32, 32))
    with pytest.raises(ValueError, match="scale block"):
        megamoe_pack_shared_expert(w1, s1, w1, s1, w2, s2, 10.0, block)


def test_buffers_are_shared_per_bucket():
    a = megamoe_shared_activation_buffers(256, 5120, DEV)
    b = megamoe_shared_activation_buffers(256, 5120, DEV)
    c = megamoe_shared_activation_buffers(1024, 5120, DEV)
    assert a[0].data_ptr() == b[0].data_ptr() and a[0].data_ptr() != c[0].data_ptr()
    assert a[0].shape == (256, 5120) and a[0].dtype == E4M3
    assert a[1].numel() == 256 * 5120 // 32

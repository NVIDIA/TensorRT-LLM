# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MegaMoE weight packing of the dep target (`weights.load_megamoe_experts` / `load_megamoe_shared`) on one GPU.

A synthetic checkpoint dict in the checkpoint's own names (`<prefix>{e}.{w1,w2,w3}.{weight,weight_scale,weight_scale_2,
input_scale}`, random packed codes and e4m3 scales, a `weight_scale_2` that differs between w1 and w3 for some
experts) is loaded for several ranks of a placement with copies. Proved: each slot holds the bytes of the expert the
placement names (against the family's `megamoe_pack_experts` reference on stacks built by hand), the scalars are the
documented functions of the checkpoint scalars with ONE FC1 activation scale over all experts, a copy slot is
bit-identical to its twin on another rank, and the shared expert goes in as the engine's 128x128-requantized FP8
(bitwise what stock's loader and `build_shared_expert_weights` produce).
"""

import importlib
from types import SimpleNamespace as NS

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="sm_103 (GB300)",
)

_FAMILY = "tensorrt_llm._torch.modeling_v2.models.deepseek_v41"
mega = importlib.import_module(f"{_FAMILY}.flash_nvfp4__sm_103__dep.megamoe")
wts = importlib.import_module(f"{_FAMILY}.flash_nvfp4__sm_103__dep.weights")
pack_op = importlib.import_module(f"{_FAMILY}.ops.moe.megamoe_pack_experts")
shared_op = importlib.import_module(f"{_FAMILY}.ops.moe.megamoe_shared_expert")
slots_op = importlib.import_module(f"{_FAMILY}.ops.moe.megamoe_expert_slots")
DEV = "cuda"
PREFIX = "layers.5.ffn.experts."


def _config(hidden, inter, experts, slots, ep, rank, clamp=10.0):
    return mega.MegaMoeConfig(
        ep_size=ep, ep_rank=rank, num_experts=experts, num_slots=slots, top_k=6, hidden=hidden, inter=inter,
        clamp=clamp, combine_format="bf16", fused_shared=True, max_tokens=256, initial_assignments=None,
    )  # fmt: skip


def _checkpoint(hidden, inter, experts, seed=0):
    g = torch.Generator().manual_seed(seed)

    def scales(*shape):
        return (torch.randint(-3, 1, shape, generator=g).float().exp2()).to(torch.float8_e4m3fn)

    w = {}
    for e in range(experts):
        q = f"{PREFIX}{e}."
        for name, (n, k) in (
            ("w1", (inter, hidden)),
            ("w3", (inter, hidden)),
            ("w2", (hidden, inter)),
        ):
            w[q + f"{name}.weight"] = torch.randint(
                0, 256, (n, k // 2), generator=g, dtype=torch.uint8
            )
            w[q + f"{name}.weight_scale"] = scales(n, k // 16)
            w[q + f"{name}.input_scale"] = 0.002 + 0.01 * torch.rand((), generator=g)
            w[q + f"{name}.weight_scale_2"] = 0.01 * (0.8 + 0.4 * torch.rand((), generator=g))
        if (
            e % 3 == 0
        ):  # up and gate with different weight_scale_2: the gate's block scales go onto the larger
            w[q + "w3.weight_scale_2"] = w[q + "w1.weight_scale_2"] * 2.0
    return w


class _Moe:
    """What `_MegaMoe` offers the loader: the config, the placement, and the two setters."""

    def __init__(self, mc, assignments):
        self.mc, self.assignments, self.cfg = mc, assignments, NS(swiglu_limit=mc.clamp)
        self.experts = self.shared = None

    def set_experts(self, fc1, fc1_sf, fc2, fc2_sf, scalars, device):
        self.experts = (fc1, fc1_sf, fc2, fc2_sf, scalars)

    def set_shared(self, pack):
        self.shared = pack


def _load(ck, mc, assignments):
    moe = _Moe(mc, assignments)
    wts.load_megamoe_experts(moe, ck, PREFIX, DEV)
    return moe.experts


def _expected(ck, mc, assignments):
    """The slots' stacks built by hand from the placement, and the scalars from their definitions."""
    local = mc.local_slots
    mine = assignments[mc.ep_rank * local : (mc.ep_rank + 1) * local]

    def stack(name, key):
        return torch.stack([ck[f"{PREFIX}{e}.{name}.{key}"] for e in mine]).to(DEV)

    def sc(e, name, key):
        return float(ck[f"{PREFIX}{e}.{name}.{key}"])

    i1 = max(
        max(sc(e, "w1", "input_scale"), sc(e, "w3", "input_scale")) for e in range(mc.num_experts)
    )
    ws1 = torch.tensor([sc(e, "w1", "weight_scale_2") for e in mine])
    ws3 = torch.tensor([sc(e, "w3", "weight_scale_2") for e in mine])
    isc2 = torch.tensor([sc(e, "w2", "input_scale") for e in mine])
    ws2 = torch.tensor([sc(e, "w2", "weight_scale_2") for e in mine])
    ws13 = torch.maximum(ws1, ws3)  # the fused FC1's one weight scalar
    packed = pack_op.megamoe_pack_experts.reference(
        stack("w1", "weight"), stack("w1", "weight_scale").view(torch.uint8),
        stack("w3", "weight"), stack("w3", "weight_scale").view(torch.uint8),
        stack("w2", "weight"), stack("w2", "weight_scale").view(torch.uint8),
        (ws3 / ws13).to(DEV), (ws1 / ws13).to(DEV),
    )  # fmt: skip
    scalars = dict(
        input_global_scale=torch.tensor([1.0 / i1]),
        fc1_alpha=i1 * ws13,
        fc2_alpha=isc2 * ws2,
        fc1_norm_const=1.0 / isc2,
    )
    return packed, {k: v.to(DEV) for k, v in scalars.items()}


# (hidden, inter, experts, slots, ep): the deployed width with few experts, and a small width with copies
GEOMETRIES = [(5120, 2304, 8, 12, 4), (512, 256, 16, 20, 4)]


@pytest.mark.parametrize(
    "hidden,inter,experts,slots,ep", GEOMETRIES, ids=["deployed-width", "small-width"]
)
@pytest.mark.parametrize("rank", [0, 1, 3])
def test_every_slot_holds_the_bytes_and_scalars_of_its_expert(
    hidden, inter, experts, slots, ep, rank
):
    mc = _config(hidden, inter, experts, slots, ep, rank)
    assignments = slots_op.megamoe_default_placement(experts, slots, ep)
    ck = _checkpoint(hidden, inter, experts, seed=rank)
    fc1, fc1_sf, fc2, fc2_sf, scalars = _load(ck, mc, assignments)
    ref, ref_scalars = _expected(ck, mc, assignments)
    for got, want in zip((fc1, fc1_sf, fc2, fc2_sf), ref):
        assert got.dtype == torch.uint8 and torch.equal(got, want)
    for key, want in ref_scalars.items():
        torch.testing.assert_close(scalars[key], want, rtol=1e-6, atol=0)
        assert scalars[key].dtype == torch.float32
    assert fc1.shape[0] == mc.local_slots and scalars["fc1_alpha"].shape == (mc.local_slots,)


def test_a_copy_slot_is_bit_identical_to_its_twin():
    hidden, inter, experts, slots, ep = 512, 256, 16, 20, 4
    assignments = slots_op.megamoe_default_placement(experts, slots, ep)
    ck = _checkpoint(hidden, inter, experts)
    local = slots // ep
    # rank r's last slot holds the first expert of rank r + 1: the same expert is slot 0 there
    for r in range(ep - 1):
        a = _load(ck, _config(hidden, inter, experts, slots, ep, r), assignments)
        b = _load(ck, _config(hidden, inter, experts, slots, ep, r + 1), assignments)
        assert assignments[r * local + local - 1] == assignments[(r + 1) * local]
        for ta, tb in zip(a[:4], b[:4]):
            assert torch.equal(ta[local - 1], tb[0])
        for key in ("fc1_alpha", "fc2_alpha", "fc1_norm_const"):
            assert torch.equal(a[4][key][local - 1], b[4][key][0])
        # the layer-wide activation scale is the same on every rank
        assert torch.equal(a[4]["input_global_scale"], b[4]["input_global_scale"])


def test_a_configured_placement_moves_the_bytes_with_the_experts():
    hidden, inter, experts, slots, ep = 512, 256, 16, 16, 4
    ck = _checkpoint(hidden, inter, experts)
    one = list(range(experts))
    swapped = list(one)
    swapped[1], swapped[14] = swapped[14], swapped[1]  # expert 14 now on rank 0, expert 1 on rank 3
    a = _load(ck, _config(hidden, inter, experts, slots, ep, 0), one)
    b = _load(ck, _config(hidden, inter, experts, slots, ep, 0), swapped)
    c = _load(ck, _config(hidden, inter, experts, slots, ep, 3), one)
    assert torch.equal(b[0][1], c[0][2])  # expert 14: slot 1 of rank 0 now, slot 2 of rank 3 before
    assert not torch.equal(a[0][1], b[0][1]) and torch.equal(a[0][0], b[0][0])


def test_the_shared_expert_carries_the_engines_requantized_fp8():
    """The engine requantizes the shared expert's block-32 FP8 to 128x128 power-of-two tiles at load
    (`_V41_REQUANT_STEM_TAILS`), then `build_shared_expert_weights` packs it for MegaMoE: the target's kernel bytes and
    scales are exactly those (the checkpoint's own block-32 bytes would be a different, finer quantization)."""
    from torch.nn import functional as F

    from tensorrt_llm._torch.models.modeling_deepseekv41 import _requantize_dense_stem
    from tensorrt_llm._torch.moe.fused_moe.mega_moe.shared_expert import build_shared_expert_weights

    hidden, inter = 512, 256  # H % 512 == 0 and 2 * I % 512 == 0
    g = torch.Generator().manual_seed(3)
    ck = {}
    for name, (n, k) in (("w1", (inter, hidden)), ("w3", (inter, hidden)), ("w2", (hidden, inter))):
        ck[f"layers.5.ffn.shared_experts.{name}.weight"] = (
            torch.randn(n, k, generator=g).clamp(-4, 4).to(torch.float8_e4m3fn)
        )
        # block exponents that differ inside a 128x128 tile, so the requantization re-rounds some values
        ck[f"layers.5.ffn.shared_experts.{name}.scale"] = torch.randint(
            115, 127, (n // 32, k // 32), generator=g, dtype=torch.uint8
        )
    moe = _Moe(_config(hidden, inter, 16, 16, 4, 0, clamp=10.0), list(range(16)))
    wts.load_megamoe_shared(moe, ck, "layers.5.ffn.", DEV)
    stock = {}
    for name in ("w1", "w3", "w2"):
        scale = torch.exp2(ck[f"layers.5.ffn.shared_experts.{name}.scale"].float() - 127.0)
        stock[name] = [
            t.to(DEV)
            for t in _requantize_dense_stem(ck[f"layers.5.ffn.shared_experts.{name}.weight"], scale)
        ]
    mlp = NS(
        gate_up_proj=NS(
            weight=torch.cat([stock["w1"][0], stock["w3"][0]]),
            weight_scale=torch.cat([stock["w1"][1], stock["w3"][1]]),
            bias=None,
        ),
        down_proj=NS(weight=stock["w2"][0], weight_scale=stock["w2"][1], bias=None),
        swiglu_alpha=None, swiglu_beta=None, activation=F.silu, swiglu_limit=10.0,
    )  # fmt: skip
    theirs = build_shared_expert_weights(mlp, hidden)
    assert theirs is not None
    p = moe.shared
    for key in ("fc1_weight", "fc1_weight_sf", "fc2_weight", "fc2_weight_sf"):
        assert torch.equal(p[key].view(torch.uint8), theirs[key].view(torch.uint8)), key
    assert p["intermediate_size"] == inter and p["gate_up_clamp"] == 10.0
    w2 = ck["layers.5.ffn.shared_experts.w2.weight"].view(torch.uint8).to(DEV)
    assert not torch.equal(
        p["fc2_weight"].view(torch.uint8), w2
    )  # requantized, not the checkpoint's block-32 bytes

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 family's checkpoint load (``weights.py``), host-side: it rejects the configurations it does not take
(helix, attention DP, pipeline parallelism, FP8 attention, routed experts other than packed MXFP4) with the reason;
the KDA and MLA fused rows read each key through its own cut into its row range (the checks are
``_kimi_k3_loader_parity``); and a routed expert read on the host reaches the buffers' device as this rank's MoE-TP
share only, at both targets' expert splits, with the bytes of the whole-expert relayout."""

import types

import pytest
import torch

from tensorrt_llm._torch.modeling_v2._router_index import NoTarget

try:
    from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl import moe, weights
except NoTarget as no_target:
    pytest.skip(str(no_target), allow_module_level=True)

__extra_import_path__ = ["."]
from _kimi_k3_loader_parity import (  # noqa: E402
    REJECTIONS,
    check_fused_rows,
    check_rejects,
    check_rejects_nvfp4_experts,
    check_row_cuts,
)

pytestmark = pytest.mark.cpu_only

LATENT, MOE_INTERMEDIATE = 3584, 3072

#: Each target's MoE split and one MoE TP rank of it.
EXPERT_SPLITS = {
    "kimi_k3_mxfp4__sm_100__tp16_moetp4ep4": dict(moe_tp=4, moe_tp_rank=2, moe_ep=4, moe_ep_rank=1),
    "kimi_k3_mxfp4__sm_100__tp16_moetp16ep1": dict(
        moe_tp=16, moe_tp_rank=5, moe_ep=1, moe_ep_rank=0
    ),
}


@pytest.mark.parametrize("case", sorted(REJECTIONS))
def test_rejects(case):
    check_rejects(weights, case)


def test_rejects_nvfp4_experts():
    check_rejects_nvfp4_experts(weights)


def test_fused_rows_land_in_their_row_ranges():
    check_fused_rows(weights)


def test_row_cuts_per_key():
    check_row_cuts(weights)


def _recording(moved: list):
    """A tensor type that records the shape of every tensor of it handed to ``Tensor.to``."""

    class Recorded(torch.Tensor):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            if func is torch.Tensor.to:
                moved.append(tuple(args[0].shape))
            return super().__torch_function__(func, types, args, kwargs or {})

    return Recorded


def _checkpoint_expert(seed: int) -> dict[str, torch.Tensor]:
    """One checkpoint expert at Kimi K3's full width, on the host: packed MXFP4 w1 / w3 ``[3072, 1792]``, w2
    ``[3584, 1536]``, and their E8M0 scales."""
    g = torch.Generator().manual_seed(seed)

    def u8(*shape, lo=0, hi=256):
        return torch.randint(lo, hi, shape, generator=g, dtype=torch.uint8)

    return dict(
        w1_weight=u8(MOE_INTERMEDIATE, LATENT // 2),
        w1_weight_scale=u8(MOE_INTERMEDIATE, LATENT // 32, lo=118, hi=124),
        w2_weight=u8(LATENT, MOE_INTERMEDIATE // 2),
        w2_weight_scale=u8(LATENT, MOE_INTERMEDIATE // 32, lo=118, hi=124),
        w3_weight=u8(MOE_INTERMEDIATE, LATENT // 2),
        w3_weight_scale=u8(MOE_INTERMEDIATE, LATENT // 32, lo=118, hi=124),
    )


@pytest.mark.parametrize("split", sorted(EXPERT_SPLITS))
def test_routed_expert_moves_only_its_share(split):
    """The loader's routed-expert step (``_pack_routed_expert`` -> ``load_packed_mxfp4_expert``) hands ``Tensor.to``
    (the copy to the buffers' device) this rank's share of each of the six checkpoint tensors, cut on the intermediate
    axis and padded to the rank's width, never the full width; the slot then holds the bytes of the relayout of the
    whole expert (``pack_mxfp4_expert``)."""
    s = EXPERT_SPLITS[split]
    layout = moe.Mxfp4ExpertLayout(
        hidden=LATENT,
        intermediate=MOE_INTERMEDIATE,
        moe_tp_size=s["moe_tp"],
        moe_tp_rank=s["moe_tp_rank"],
    )
    # Four experts over the split's EP size: one or four local slots, each the layout's shapes.
    backend = moe.K3MxfpExpertWeights(
        num_experts=4,
        moe_ep_size=s["moe_ep"],
        moe_ep_rank=s["moe_ep_rank"],
        layout=layout,
        situ_betas=(4.0, 25.0),
        device=torch.device("cpu"),
    )
    expert, slot = backend.initial_local_expert_ids[0], 0
    tensors = _checkpoint_expert(expert)
    moved: list = []
    recorded_type = _recording(moved)
    recorded = {name: t.as_subclass(recorded_type) for name, t in tensors.items()}
    stand_in = types.SimpleNamespace(routed_experts=types.SimpleNamespace(backend=backend))
    weights._pack_routed_expert(stand_in, expert, slot, recorded)

    inter_pad, k_pad, rows = layout.inter_pad, layout.fc1_k_pad, layout.fc2_rows_pad
    fc1 = (inter_pad, k_pad // 2)
    fc1_scale = (inter_pad, k_pad // 32)
    share = [fc1, fc1_scale, (rows, inter_pad // 2), (rows, inter_pad // 32), fc1, fc1_scale]
    assert sorted(moved) == sorted(share), (moved, share)
    full = {tuple(t.shape) for t in tensors.values()}
    assert not full & set(moved), (moved, full)
    # Bytes per expert to the device: 1 / moe_tp of the six tensors, up to the zero padding of the share (none at
    # moe TP 4; 192 -> 256 columns, 4 / 3, at moe TP 16).
    assert (
        sum(r * c for r, c in moved) * s["moe_tp"]
        <= sum(t.numel() for t in tensors.values()) * 4 // 3
    )

    expected = moe.pack_mxfp4_expert(layout, **tensors)
    for name, value in expected.items():
        assert torch.equal(getattr(backend, name).data[slot], value), name

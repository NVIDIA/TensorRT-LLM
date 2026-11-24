# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The routed experts of ``kimi_k31_pro_nvfp4__sm_107__helix16_wide`` sum to the layer (CPU only):
the 16 partials of FC12's FP32 reference on the banks the loader packs (un-packed back to the
reference's stored form) sum to the whole layer's reference, at shrunk dimensions (16 experts,
intermediate 256, latent 128, top 4; the TP2 x EP8 split unchanged).
"""

from __future__ import annotations

import pytest
import torch
from _helix16_wide_cpu import (
    checkpoint,
    fc12_reference_module,
    load_weights_module,
    tiny_experts,
    ungate_up16,
    unswizzle,
)

pytestmark = pytest.mark.cpu_only

PREFIX = "language_model."
LAYER = 7
TOP_K = 4
SITU = dict(situ_beta=4.0, situ_linear_beta=25.0)


@pytest.fixture(scope="module")
def w():
    return load_weights_module()


@pytest.fixture
def reference():
    with fc12_reference_module() as module:
        yield module.nvfp4_situ_moe_reference


def _stored(ckpt, w, e, proj):
    """``Nvfp4Weight`` (codes, E4M3 block scales, weight_scale_2, input_scale) of one projection,
    as the reference takes it."""
    key = lambda kind: w.expert_key(PREFIX, LAYER, e, proj, kind)  # noqa: E731
    return (
        ckpt[key("weight")].contiguous(),
        ckpt[key("weight_scale")].contiguous().view(torch.uint8),
        ckpt[key("weight_scale_2")],
        ckpt[key("input_scale")],
    )


def _from_bank(bank, local, w):
    """The reference's stored form of local expert ``local`` of a packed FC12 bank: FC1 rows
    un-interleaved, block scales un-swizzled, ``weight_scale_2`` = alpha x the input scale
    (``alpha = input_scale_max * weight_scale_2`` in the packer), ``input_scale`` = 1 / scale."""
    gate, up = ungate_up16(bank["fc1_w"][local])
    gate_s, up_s = ungate_up16(unswizzle(bank["fc1_s"][local]))
    in1 = bank["fc31_in_scale"].reshape(()).reciprocal()
    in2 = bank["fc2_in_scale"].reshape(()).reciprocal()
    ws1 = bank["fc31_alpha"][local] * bank["fc31_in_scale"].reshape(())
    ws2 = bank["fc2_alpha"][local] * bank["fc2_in_scale"].reshape(())
    return {
        "w1": (gate, gate_s, ws1, in1),
        "w3": (up, up_s, ws1, in1),
        "w2": (bank["fc2_w"][local], unswizzle(bank["fc2_s"][local]), ws2, in2),
    }


def _routes(experts, rows, generator):
    ids = torch.stack([torch.randperm(experts, generator=generator)[:TOP_K] for _ in range(rows)])
    weights = torch.rand(rows, TOP_K, generator=generator).to(torch.bfloat16).float()
    return ids.to(torch.int32), weights


@pytest.mark.parametrize("rows", [8, 72])
def test_the_16_ep_partials_sum_to_the_layer(w, reference, rows):
    with tiny_experts(w) as dims:
        e_all, latent = dims["EXPERTS"], dims["LATENT"]
        ckpt = checkpoint(w, LAYER, seed=3)
        g = torch.Generator().manual_seed(rows)
        x = (torch.randn(rows, latent, generator=g) * 2).to(torch.bfloat16)
        ids, weights = _routes(e_all, rows, g)
        whole = [{p: _stored(ckpt, w, e, p) for p in ("w1", "w3", "w2")} for e in range(e_all)]
        layer = reference(x, ids, weights, whole, local_expert_offset=0, **SITU)
        assert layer.abs().max() > 0

        ep_sum = torch.zeros_like(layer)
        for rank in range(16):
            reads = w.ExpertReads()
            raw = w.read_routed_experts_ep(
                ckpt, LAYER, rank, prefix=PREFIX, device="cpu", reads=reads
            )
            reads.check(PREFIX, LAYER, rank)
            bank = w.pack_routed_experts_fc12(**raw)
            window = w.expert_window(rank)
            experts = [_from_bank(bank, j, w) for j in range(len(window))]
            ep_sum += reference(x, ids, weights, experts, local_expert_offset=window.start, **SITU)

        err = (ep_sum - layer).abs().max() / layer.abs().max()
        assert err < 1e-5, f"max |sum - layer| / max |layer| = {err:.3e}"

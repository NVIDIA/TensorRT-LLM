# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FC1 gate/up weight interleave contracts of the CuTe DSL MoE weight method.

The Rubin NVFP4 kernels read FC1 B and its block scales in ``[gate16, up16]``
row groups, the Blackwell kernels in ``[up64, gate64]`` groups. These tests pin
the helper that produces the Rubin order against the two other places that
define the same layout (MegaMoE's packing and the kernel author's reference
permutation), and check that the weight method picks the order by SM and
applies the identical permutation on the device weights and on the EPLB host
mirrors.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe import quantization as quant_mod
from tensorrt_llm._torch.moe.fused_moe.quantization import (
    NVFP4CuteDslFusedMoEMethod,
    interleave_gate_and_linear,
    interleave_linear_and_gate,
)
from tensorrt_llm._torch.utils import swizzle_sf, unswizzle_sf

# FC1 rows are [up (intermediate) | gate (intermediate)]; K is packed FP4 bytes.
INTERMEDIATE = 256
FC1_ROWS = 2 * INTERMEDIATE
HIDDEN = 512
SF_VEC = 16


def _raw_fc1(num_experts: int = 1, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, 256, (num_experts, FC1_ROWS, HIDDEN // 2), dtype=torch.uint8, generator=g
    )


def test_gate16_helper_matches_megamoe_inline_packing():
    raw = _raw_fc1(num_experts=3)
    ours = interleave_gate_and_linear(raw, group_size=16, dim=1)

    # The MegaMoE weight method packs the same layout inline
    # (NVFP4MegaMoECuteDslMethod._build_mega_format_buffers): gate and up are
    # split into 16-row atoms and stacked gate-first.
    num_slots = raw.shape[0]
    n_pairs = INTERMEDIATE // 16
    up_part = raw[:, :INTERMEDIATE, :]
    gate_part = raw[:, INTERMEDIATE:, :]
    gate_p = gate_part.reshape(num_slots, n_pairs, 16, -1)
    up_p = up_part.reshape(num_slots, n_pairs, 16, -1)
    theirs = torch.stack([gate_p, up_p], dim=2).reshape(num_slots, FC1_ROWS, -1)

    assert torch.equal(ours, theirs)


def test_gate16_helper_matches_kernel_reference_permutation():
    # The kernel author's host packer starts from the older [up64, gate64]
    # interleave and reorders every 128-row block through the index map
    # (N/128, [up, gate], 4, 16) -> flip(up/gate) -> transpose(groups, up/gate).
    raw = _raw_fc1()[0]
    assert FC1_ROWS % 128 == 0
    canonical = interleave_linear_and_gate(raw, group_size=64, dim=0)
    canonical_n = (
        torch.arange(FC1_ROWS)
        .reshape(FC1_ROWS // 128, 2, 4, 16)
        .flip(1)
        .transpose(1, 2)
        .reshape(FC1_ROWS)
    )
    expected = canonical.index_select(0, canonical_n)

    assert torch.equal(interleave_gate_and_linear(raw, group_size=16, dim=0), expected)


def test_gate16_helper_is_one_row_permutation_for_weights_and_scales():
    perm = interleave_gate_and_linear(
        torch.arange(FC1_ROWS).view(FC1_ROWS, 1), group_size=16, dim=0
    )
    perm = perm.view(FC1_ROWS)
    assert torch.equal(perm.sort().values, torch.arange(FC1_ROWS))
    # First 32 rows: gate rows 0..15 (offset INTERMEDIATE) then up rows 0..15.
    assert perm[:16].tolist() == list(range(INTERMEDIATE, INTERMEDIATE + 16))
    assert perm[16:32].tolist() == list(range(0, 16))

    weights = _raw_fc1()[0]
    scales = torch.randint(0, 256, (FC1_ROWS, HIDDEN // SF_VEC), dtype=torch.uint8)
    assert torch.equal(interleave_gate_and_linear(weights, group_size=16, dim=0), weights[perm])
    assert torch.equal(interleave_gate_and_linear(scales, group_size=16, dim=0), scales[perm])


def test_gate16_helper_dim_argument_is_per_expert():
    raw = _raw_fc1(num_experts=2)
    batched = interleave_gate_and_linear(raw, group_size=16, dim=1)
    per_expert = torch.stack(
        [interleave_gate_and_linear(raw[e], group_size=16, dim=0) for e in range(raw.shape[0])]
    )
    assert torch.equal(batched, per_expert)


def _expected_rows(x: torch.Tensor, sm: int) -> torch.Tensor:
    if sm == 107:
        return interleave_gate_and_linear(x, group_size=16, dim=0)
    return interleave_linear_and_gate(x, group_size=64, dim=0)


def _module_namespace(is_gated_activation: bool = True, **extra) -> SimpleNamespace:
    return SimpleNamespace(
        intermediate_size_per_partition=INTERMEDIATE,
        hidden_size=HIDDEN,
        scaling_vector_size=SF_VEC,
        is_gated_activation=is_gated_activation,
        **extra,
    )


def _random_swizzled_scales(num_experts: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, 256, (num_experts, FC1_ROWS, HIDDEN // SF_VEC), dtype=torch.uint8, generator=g
    ).view(quant_mod.float4_sf_dtype)


def _expected_scales(dst: torch.Tensor, sm: int) -> torch.Tensor:
    # Mirrors _interleave_w3_w1_weight_scale_cute_dsl with an explicit helper.
    unswizzled = unswizzle_sf(dst.cuda(), FC1_ROWS, HIDDEN).view(FC1_ROWS, HIDDEN // SF_VEC)
    return swizzle_sf(_expected_rows(unswizzled, sm), FC1_ROWS, HIDDEN).view(
        FC1_ROWS, HIDDEN // SF_VEC
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the weight method interleaves on the GPU"
)
@pytest.mark.parametrize("sm", [107, 100], ids=["rubin_gate16", "blackwell_up64"])
def test_weight_method_picks_the_interleave_by_sm(monkeypatch, sm):
    monkeypatch.setattr(quant_mod, "get_sm_version", lambda: sm)
    method = NVFP4CuteDslFusedMoEMethod()

    raw = _raw_fc1()[0]
    dst = raw.clone()
    method._interleave_w3_w1_weight(dst)
    assert torch.equal(dst, _expected_rows(raw, sm))

    scales = _random_swizzled_scales(1, seed=1)[0]
    dst_sf = scales.clone()
    method._interleave_w3_w1_weight_scale_cute_dsl(_module_namespace(), dst_sf)
    assert torch.equal(
        dst_sf.view(torch.uint8), _expected_scales(scales, sm).cpu().view(torch.uint8)
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the weight method interleaves on the GPU"
)
@pytest.mark.parametrize("sm", [107, 100], ids=["rubin_gate16", "blackwell_up64"])
def test_eplb_host_mirrors_get_the_same_interleave_as_device_weights(monkeypatch, sm):
    # Online EPLB copies experts into slots from the host mirrors, bypassing
    # process_weights_after_loading, so the mirror hooks must apply the exact
    # permutation the device path applies.
    monkeypatch.setattr(quant_mod, "get_sm_version", lambda: sm)
    method = NVFP4CuteDslFusedMoEMethod()
    num_experts = 3

    raw = _raw_fc1(num_experts, seed=2)
    scales = _random_swizzled_scales(num_experts, seed=3)
    module = _module_namespace(
        local_shared_w3_w1_tensors=raw.clone(),
        local_shared_w3_w1_scale_tensors=scales.clone(),
    )
    method._prepare_shared_weights_for_finalization(module)
    method._prepare_shared_weight_scales_for_finalization(module)

    for e in range(num_experts):
        device_w = raw[e].clone()
        method._interleave_w3_w1_weight(device_w)
        assert torch.equal(module.local_shared_w3_w1_tensors[e], device_w)
        assert torch.equal(module.local_shared_w3_w1_tensors[e], _expected_rows(raw[e], sm))

        device_sf = scales[e].clone()
        method._interleave_w3_w1_weight_scale_cute_dsl(_module_namespace(), device_sf)
        assert torch.equal(
            module.local_shared_w3_w1_scale_tensors[e].view(torch.uint8),
            device_sf.view(torch.uint8),
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the weight method interleaves on the GPU"
)
def test_eplb_hooks_skip_non_gated_modules(monkeypatch):
    monkeypatch.setattr(quant_mod, "get_sm_version", lambda: 107)
    method = NVFP4CuteDslFusedMoEMethod()
    raw = _raw_fc1(2, seed=4)
    module = _module_namespace(is_gated_activation=False, local_shared_w3_w1_tensors=raw.clone())
    method._prepare_shared_weights_for_finalization(module)
    assert torch.equal(module.local_shared_w3_w1_tensors, raw)

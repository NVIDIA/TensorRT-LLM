# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``prepare_mxfp4_weights_for_trtllm_gen``.

These tests mirror the gpt-oss-120b MoE/GEMM structure (small E/H/I) and pin
the kernel-layout invariants the trtllm-gen ``bf16_mxe2m1_block_scale_moe_runner``
relies on:

* fc1 / fc2 biases must go through the SAME row permutation as fc1 / fc2
  weights (gated-act-gemm interleave + epilogue-tile reorder for w3/w1; only
  the epilogue-tile reorder for w2). Without this the kernel adds the wrong
  bias to each post-shuffle output row and MoE output is garbage (~2% on
  gpt-oss-120b GSM8K instead of ~90%).
"""

import pytest
import torch

# Permute helpers are CUDA-only because shuffle_matrix is registered there.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="prepare_mxfp4_weights_for_trtllm_gen relies on torch.ops.trtllm.shuffle_matrix",
)


# Sized to match the gpt-oss-120b layout exactly (H=2880, I=2880) but with a
# small expert count to keep the test cheap.
GPTOSS_HIDDEN_SIZE = 2880
GPTOSS_INTERMEDIATE_SIZE = 2880
NUM_EXPERTS = 4


def _build_synthetic_mxfp4_inputs(
    e: int = NUM_EXPERTS,
    h: int = GPTOSS_HIDDEN_SIZE,
    i: int = GPTOSS_INTERMEDIATE_SIZE,
    device: str = "cuda",
):
    """Build deterministic uint8/bf16 expert tensors in the HF on-disk layout."""
    g = torch.Generator(device="cpu").manual_seed(0)
    gu_blocks = torch.randint(0, 256, (e, 2 * i, h // 32, 16), dtype=torch.uint8, generator=g).to(
        device
    )
    gu_scales = torch.randint(0, 256, (e, 2 * i, h // 32), dtype=torch.uint8, generator=g).to(
        device
    )
    gu_bias = torch.randn(e, 2 * i, dtype=torch.bfloat16, generator=g).to(device)
    dn_blocks = torch.randint(0, 256, (e, h, i // 32, 16), dtype=torch.uint8, generator=g).to(
        device
    )
    dn_scales = torch.randint(0, 256, (e, h, i // 32), dtype=torch.uint8, generator=g).to(device)
    dn_bias = torch.randn(e, h, dtype=torch.bfloat16, generator=g).to(device)
    return gu_blocks, gu_scales, gu_bias, dn_blocks, dn_scales, dn_bias


def test_fc1_bias_is_shuffled_with_same_row_permutation_as_fc1_weights():
    """Regression: fc1 bias must follow the gated-act-gemm + TMA row permute.

    Reproduces the gpt-oss-120b MoE/GEMM accuracy bug where bias was only
    padded (not shuffled), causing the trtllm-gen kernel to add the wrong
    bias to each output row.
    """
    from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe.mxfp4_weight_prep import (
        prepare_mxfp4_weights_for_trtllm_gen,
    )
    from tensorrt_llm._torch.modules.fused_moe.quantization import (
        trtllmgen_maybe_get_cached_w3_w1_permute_indices,
    )

    device = "cuda"
    gu_blocks, gu_scales, gu_bias, dn_blocks, dn_scales, dn_bias = _build_synthetic_mxfp4_inputs(
        device=device
    )
    e, two_i_pad = 4, 5888  # I_pad = 2944 (= ceil(2880/128)*128); 2*I_pad = 5888

    # Reconstruct the pre-shuffle bias the prep helper builds (after pad +
    # de-interleave + cat([up | gate])). Then derive the expected shuffled
    # bias by reusing PT's permute helpers and compare against the actual
    # output of ``prepare_mxfp4_weights_for_trtllm_gen``.
    gate_b = gu_bias[:, 0::2].contiguous()  # [E, I]
    up_b = gu_bias[:, 1::2].contiguous()  # [E, I]
    pad_amount = (128 - GPTOSS_INTERMEDIATE_SIZE % 128) % 128
    up_b_padded = torch.nn.functional.pad(up_b, (0, pad_amount)).float()
    gate_b_padded = torch.nn.functional.pad(gate_b, (0, pad_amount)).float()
    pre_shuffle_fc1_bias = torch.cat([up_b_padded, gate_b_padded], dim=1).contiguous()
    assert pre_shuffle_fc1_bias.shape == (e, two_i_pad)

    cache: dict = {}
    expected_fc1_bias_per_expert = []
    for k in range(e):
        slc = pre_shuffle_fc1_bias[k].contiguous()
        perm = trtllmgen_maybe_get_cached_w3_w1_permute_indices(slc, cache, 128)
        expected_fc1_bias_per_expert.append(torch.index_select(slc, 0, perm.to(slc.device)))
    expected_fc1_bias = torch.stack(expected_fc1_bias_per_expert, dim=0).contiguous()

    prep = prepare_mxfp4_weights_for_trtllm_gen(
        gu_blocks,
        gu_scales,
        gu_bias,
        dn_blocks,
        dn_scales,
        dn_bias,
        hidden_size=GPTOSS_HIDDEN_SIZE,
        intermediate_size=GPTOSS_INTERMEDIATE_SIZE,
        tp_size=1,
        tp_rank=0,
    )

    assert prep.fc1_bias_f32.shape == (e, two_i_pad)
    torch.testing.assert_close(prep.fc1_bias_f32, expected_fc1_bias, atol=0, rtol=0)

    # And it should NOT equal the unshuffled (just-padded) baseline — that's
    # the buggy state we are guarding against.
    assert not torch.equal(prep.fc1_bias_f32, pre_shuffle_fc1_bias), (
        "fc1 bias was not shuffled — this is the regression that wrecks gpt-oss-120b accuracy."
    )


def test_fc2_bias_is_shuffled_with_same_row_permutation_as_fc2_weights():
    """Regression: fc2 bias must follow the (non-gated) TMA row permute used by w2."""
    from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe.mxfp4_weight_prep import (
        prepare_mxfp4_weights_for_trtllm_gen,
    )
    from tensorrt_llm._torch.modules.fused_moe.quantization import (
        trtllmgen_maybe_get_cached_w2_permute_indices,
    )

    device = "cuda"
    gu_blocks, gu_scales, gu_bias, dn_blocks, dn_scales, dn_bias = _build_synthetic_mxfp4_inputs(
        device=device
    )
    e, h_pad = 4, 2944  # H_pad = ceil(2880/128)*128 = 2944

    pad_amount = (128 - GPTOSS_HIDDEN_SIZE % 128) % 128
    pre_shuffle_fc2_bias = torch.nn.functional.pad(dn_bias, (0, pad_amount)).float()
    assert pre_shuffle_fc2_bias.shape == (e, h_pad)

    cache: dict = {}
    expected_fc2_bias_per_expert = []
    for k in range(e):
        slc = pre_shuffle_fc2_bias[k].contiguous()
        perm = trtllmgen_maybe_get_cached_w2_permute_indices(slc, cache, 128)
        expected_fc2_bias_per_expert.append(torch.index_select(slc, 0, perm.to(slc.device)))
    expected_fc2_bias = torch.stack(expected_fc2_bias_per_expert, dim=0).contiguous()

    prep = prepare_mxfp4_weights_for_trtllm_gen(
        gu_blocks,
        gu_scales,
        gu_bias,
        dn_blocks,
        dn_scales,
        dn_bias,
        hidden_size=GPTOSS_HIDDEN_SIZE,
        intermediate_size=GPTOSS_INTERMEDIATE_SIZE,
        tp_size=1,
        tp_rank=0,
    )

    assert prep.fc2_bias_f32.shape == (e, h_pad)
    torch.testing.assert_close(prep.fc2_bias_f32, expected_fc2_bias, atol=0, rtol=0)

    assert not torch.equal(prep.fc2_bias_f32, pre_shuffle_fc2_bias), (
        "fc2 bias was not shuffled — this is the regression that wrecks gpt-oss-120b accuracy."
    )


def test_prep_against_pt_reference_loader_byte_identical():
    """Compare the AD prep output byte-for-byte against PT's MXFP4 loader.

    PT's ``MXFP4WeightTRTLLMGenFusedMoEMethod.{load_expert_w3_w1_weight,
    load_expert_w2_weight, load_expert_w3_w1_weight_scale_mxfp4,
    load_expert_w2_weight_scale_mxfp4}`` is the gold standard the AD prep
    helper must mirror.  Any divergence here is a kernel-layout bug.
    """
    from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe.mxfp4_weight_prep import (
        prepare_mxfp4_weights_for_trtllm_gen,
    )
    from tensorrt_llm._torch.modules.fused_moe.quantization import (
        _get_weight_alignment,
        maybe_pad_for_mxfp4,
        trtllmgen_maybe_get_cached_w2_permute_indices,
        trtllmgen_maybe_get_cached_w3_w1_permute_indices,
    )

    device = "cuda"
    gu_blocks, gu_scales, gu_bias, dn_blocks, dn_scales, dn_bias = _build_synthetic_mxfp4_inputs(
        device=device
    )
    e, h, i = NUM_EXPERTS, GPTOSS_HIDDEN_SIZE, GPTOSS_INTERMEDIATE_SIZE
    weight_alignment = 128
    input_hidden_alignment = 512
    scaling_vector_size = 32
    epilogue_tile_m = 128

    gu_blocks_3d = gu_blocks.contiguous().view(e, 2 * i, h // 2)  # [E, 2I, H/2]
    dn_blocks_3d = dn_blocks.contiguous().view(e, h, i // 2)  # [E, H, I/2]

    # PT-style per-expert reference for fc1 weight + bias.
    # Step 1: deinterleave gate/up halves so dst gets [up | gate] in the row dim.
    gate_w = gu_blocks_3d[:, 0::2, :].contiguous()  # [E, I, H/2]
    up_w = gu_blocks_3d[:, 1::2, :].contiguous()  # [E, I, H/2]
    gate_s = gu_scales[:, 0::2, :].contiguous()
    up_s = gu_scales[:, 1::2, :].contiguous()
    gate_b = gu_bias[:, 0::2].contiguous()
    up_b = gu_bias[:, 1::2].contiguous()

    cache: dict = {}

    fc1_weight_ref = []
    fc1_scale_ref = []
    fc1_bias_ref = []
    fc2_weight_ref = []
    fc2_scale_ref = []
    fc2_bias_ref = []

    for k in range(e):
        # ---- fc1 weight ----
        alignment_w = _get_weight_alignment(weight_alignment, scaling_vector_size, 1, i)
        u = maybe_pad_for_mxfp4(up_w[k], input_hidden_alignment // 2, alignment_w)
        gp = maybe_pad_for_mxfp4(gate_w[k], input_hidden_alignment // 2, alignment_w)
        dst = torch.cat([u, gp], dim=0).contiguous()  # [2*I_pad, H_pad/2]
        perm = trtllmgen_maybe_get_cached_w3_w1_permute_indices(dst, cache, epilogue_tile_m)
        fc1_weight_ref.append(torch.index_select(dst, 0, perm.to(dst.device)))

        # ---- fc1 weight scale ----
        u_s = maybe_pad_for_mxfp4(
            up_s[k], input_hidden_alignment // scaling_vector_size, alignment_w
        )
        gp_s = maybe_pad_for_mxfp4(
            gate_s[k], input_hidden_alignment // scaling_vector_size, alignment_w
        )
        dst_s = torch.cat([u_s, gp_s], dim=0).contiguous()  # [2*I_pad, H_pad/32]
        perm_s = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
            dst_s, cache, epilogue_tile_m, num_elts_per_sf=scaling_vector_size
        )
        shuffled_s = torch.index_select(dst_s, 0, perm_s.to(dst_s.device))
        fc1_scale_ref.append(
            torch.ops.trtllm.block_scale_interleave(shuffled_s).reshape(dst_s.shape)
        )

        # ---- fc1 bias ----
        ub = maybe_pad_for_mxfp4(up_b[k], alignment_w).float()
        gb = maybe_pad_for_mxfp4(gate_b[k], alignment_w).float()
        dst_b = torch.cat([ub, gb], dim=0).contiguous()  # [2*I_pad]
        perm_b = trtllmgen_maybe_get_cached_w3_w1_permute_indices(dst_b, cache, epilogue_tile_m)
        fc1_bias_ref.append(torch.index_select(dst_b, 0, perm_b.to(dst_b.device)))

        # ---- fc2 weight ----
        alignment_w2 = _get_weight_alignment(weight_alignment, scaling_vector_size, 1, i)
        d = maybe_pad_for_mxfp4(dn_blocks_3d[k], alignment_w2 // 2, weight_alignment)
        perm_w2 = trtllmgen_maybe_get_cached_w2_permute_indices(d, cache, epilogue_tile_m)
        fc2_weight_ref.append(torch.index_select(d, 0, perm_w2.to(d.device)))

        # ---- fc2 weight scale ----
        alignment_w2_s = _get_weight_alignment(
            weight_alignment, scaling_vector_size, 1, dn_scales[k].shape[-1]
        )
        d_s = maybe_pad_for_mxfp4(
            dn_scales[k], alignment_w2_s // scaling_vector_size, weight_alignment
        )
        perm_w2_s = trtllmgen_maybe_get_cached_w2_permute_indices(
            d_s, cache, epilogue_tile_m, num_elts_per_sf=scaling_vector_size
        )
        shuffled_s2 = torch.index_select(d_s, 0, perm_w2_s.to(d_s.device))
        fc2_scale_ref.append(
            torch.ops.trtllm.block_scale_interleave(shuffled_s2).reshape(d_s.shape)
        )

        # ---- fc2 bias ----
        db = maybe_pad_for_mxfp4(dn_bias[k], weight_alignment).float()
        perm_b2 = trtllmgen_maybe_get_cached_w2_permute_indices(db, cache, epilogue_tile_m)
        fc2_bias_ref.append(torch.index_select(db, 0, perm_b2.to(db.device)))

    fc1_weight_ref_t = torch.stack(fc1_weight_ref, dim=0).contiguous()
    fc1_scale_ref_t = torch.stack(fc1_scale_ref, dim=0).contiguous()
    fc1_bias_ref_t = torch.stack(fc1_bias_ref, dim=0).contiguous()
    fc2_weight_ref_t = torch.stack(fc2_weight_ref, dim=0).contiguous()
    fc2_scale_ref_t = torch.stack(fc2_scale_ref, dim=0).contiguous()
    fc2_bias_ref_t = torch.stack(fc2_bias_ref, dim=0).contiguous()

    prep = prepare_mxfp4_weights_for_trtllm_gen(
        gu_blocks,
        gu_scales,
        gu_bias,
        dn_blocks,
        dn_scales,
        dn_bias,
        hidden_size=h,
        intermediate_size=i,
        tp_size=1,
        tp_rank=0,
    )

    assert torch.equal(prep.fc1_weights_mxfp4, fc1_weight_ref_t)
    assert torch.equal(prep.fc1_weights_scale_ue8m0, fc1_scale_ref_t)
    torch.testing.assert_close(prep.fc1_bias_f32, fc1_bias_ref_t, atol=0, rtol=0)
    assert torch.equal(prep.fc2_weights_mxfp4, fc2_weight_ref_t)
    assert torch.equal(prep.fc2_weights_scale_ue8m0, fc2_scale_ref_t)
    torch.testing.assert_close(prep.fc2_bias_f32, fc2_bias_ref_t, atol=0, rtol=0)

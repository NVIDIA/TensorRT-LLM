# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test FlashInfer TRTLLM MLA backend operations (kv_lora_rank=256, Blackwell).

Tests the flashinfer_trtllm_mla_with_cache cached op against the torch_cached_mla_with_cache
reference implementation across prefill, decode, mixed batch, and multi-step scenarios.

Key features tested:
- Combined paged cache: [num_pages, page_size, kv_lora_rank + qk_rope_head_dim]
- Prefill: Reference torch path with causal masking
- Decode: FlashInfer Path 2 trtllm_batch_decode_with_kv_cache_mla on Blackwell (SM100+)
- Mixed batch: Prefill + decode in same call
- Multi-step: Prefill followed by multiple sequential decode steps
"""

from unittest.mock import patch

import pytest
import torch

import tensorrt_llm._torch.auto_deploy  # noqa: F401
from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import BatchInfo
from tensorrt_llm._torch.auto_deploy.custom_ops.mla import flashinfer_trtllm_mla as trtllm_mla_mod
from tests.unittest.auto_deploy.singlegpu.custom_ops.mla.test_flashinfer_mla_op import (
    _create_mla_inputs,
    _create_paged_cache_and_metadata,
    _create_unpaged_cache_and_metadata,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FlashInfer TRTLLM MLA tests require CUDA",
)

# MLA constants for rank-256 models
KV_LORA_RANK = 256
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
PAGE_SIZE = 64
DTYPE = torch.bfloat16
DEVICE = "cuda"


def _make_combined_cache(ckv_cache: torch.Tensor, kpe_cache: torch.Tensor) -> torch.Tensor:
    """Create 3D combined cache [num_pages, page_size, kv_lora_rank + qk_rope_head_dim]."""
    combined = torch.zeros(
        ckv_cache.shape[0],
        ckv_cache.shape[1],
        ckv_cache.shape[2] + kpe_cache.shape[2],
        dtype=ckv_cache.dtype,
        device=ckv_cache.device,
    )
    combined[:, :, : ckv_cache.shape[2]] = ckv_cache
    combined[:, :, ckv_cache.shape[2] :] = kpe_cache
    return combined


def _copy_unpaged_to_combined_paged(
    unpaged_cache: torch.Tensor,
    combined_cache: torch.Tensor,
    batch_size: int,
    tokens_per_seq: list,
    page_size: int,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
    kv_lora_rank: int,
):
    """Copy unpaged cache to combined paged cache format."""
    for batch_idx in range(batch_size):
        num_tokens = tokens_per_seq[batch_idx]
        if num_tokens == 0:
            continue
        page_start_idx = int(cu_num_pages[batch_idx].item())
        page_end_idx = int(cu_num_pages[batch_idx + 1].item())
        token_offset = 0
        for i in range(page_start_idx, page_end_idx):
            page_num = int(cache_loc[i].item())
            tokens_to_copy = min(page_size, num_tokens - token_offset)
            if tokens_to_copy <= 0:
                break
            unpaged_data = unpaged_cache[batch_idx, token_offset : token_offset + tokens_to_copy]
            combined_cache[page_num, :tokens_to_copy] = unpaged_data
            token_offset += tokens_to_copy


def _run_torch_reference(inputs, torch_meta, kv_lora_rank):
    """Run torch reference backend."""
    return torch.ops.auto_deploy.torch_cached_mla_with_cache(
        inputs["q_nope"],
        inputs["q_pe"],
        inputs["compressed_kv"],
        inputs["kpe"],
        inputs["kv_b_proj_weight"],
        torch_meta["batch_info_host"],
        torch_meta["seq_len"],
        torch_meta["input_pos"],
        torch_meta["slot_idx"],
        torch_meta["cu_seqlen"],
        torch_meta["mla_cache"],
        None,
        kv_lora_rank,
    )


def _run_trtllm_mla(inputs, flashinfer_meta, combined_cache, kv_lora_rank):
    """Run flashinfer_trtllm_mla backend."""
    return torch.ops.auto_deploy.flashinfer_trtllm_mla_with_cache(
        inputs["q_nope"],
        inputs["q_pe"],
        inputs["compressed_kv"],
        inputs["kpe"],
        inputs["kv_b_proj_weight"],
        flashinfer_meta["batch_info_host"],
        flashinfer_meta["cu_seqlen_host"],
        flashinfer_meta["cu_num_pages"],
        flashinfer_meta["cu_num_pages_host"],
        flashinfer_meta["cache_loc"],
        flashinfer_meta["last_page_len"],
        flashinfer_meta["last_page_len_host"],
        flashinfer_meta["seq_len_with_cache_host"],
        combined_cache,
        None,
        kv_lora_rank,
    )


# =============================================================================
# Test 1: Decode on Blackwell (real FlashInfer kernel) vs torch reference
# =============================================================================


def test_flashinfer_trtllm_mla_decode_matches_torch_reference():
    if torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("requires Hopper+ for MLA test setup")

    batch_size = 2
    seq_len = 1
    prefill_seq_length = 128
    num_heads = 4
    max_seq_len = prefill_seq_length + seq_len + 64
    max_num_pages = batch_size * (max_seq_len // PAGE_SIZE + 2)

    inputs = _create_mla_inputs(
        batch_size,
        seq_len,
        num_heads,
        QK_NOPE_HEAD_DIM,
        QK_ROPE_HEAD_DIM,
        KV_LORA_RANK,
        V_HEAD_DIM,
        DTYPE,
        DEVICE,
    )

    seq_lengths = [seq_len] * batch_size
    input_positions = [prefill_seq_length] * batch_size

    torch_meta = _create_unpaged_cache_and_metadata(
        batch_size,
        max_seq_len,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        DTYPE,
        DEVICE,
        seq_lengths,
        input_positions,
    )
    torch_meta["mla_cache"][:, :prefill_seq_length].normal_()

    flashinfer_meta = _create_paged_cache_and_metadata(
        batch_size,
        max_num_pages,
        PAGE_SIZE,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        DTYPE,
        DEVICE,
        seq_lengths,
        input_positions,
    )
    combined_cache = _make_combined_cache(
        flashinfer_meta["ckv_cache"], flashinfer_meta["kpe_cache"]
    )
    _copy_unpaged_to_combined_paged(
        torch_meta["mla_cache"],
        combined_cache,
        batch_size,
        [prefill_seq_length] * batch_size,
        PAGE_SIZE,
        flashinfer_meta["cu_num_pages"],
        flashinfer_meta["cache_loc"],
        KV_LORA_RANK,
    )

    torch_output = _run_torch_reference(inputs, torch_meta, KV_LORA_RANK)
    trtllm_output = _run_trtllm_mla(inputs, flashinfer_meta, combined_cache, KV_LORA_RANK)

    assert torch.allclose(trtllm_output.float(), torch_output.float(), atol=5e-2, rtol=5e-2), (
        f"Decode output mismatch. Max diff: {(trtllm_output - torch_output).abs().max():.6f}"
    )


# =============================================================================
# Test 2: Mock-based test for Blackwell kernel dispatch
# =============================================================================


def test_flashinfer_trtllm_mla_uses_blackwell_decode_kernel_when_available():
    batch_size = 2
    seq_len = 1
    num_heads = 4
    max_num_pages = batch_size * 4

    inputs = _create_mla_inputs(
        batch_size,
        seq_len,
        num_heads,
        QK_NOPE_HEAD_DIM,
        QK_ROPE_HEAD_DIM,
        KV_LORA_RANK,
        V_HEAD_DIM,
        DTYPE,
        DEVICE,
    )
    seq_lengths = [seq_len] * batch_size
    input_positions = [64] * batch_size
    flashinfer_meta = _create_paged_cache_and_metadata(
        batch_size,
        max_num_pages,
        PAGE_SIZE,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        DTYPE,
        DEVICE,
        seq_lengths,
        input_positions,
    )
    combined_cache = _make_combined_cache(
        flashinfer_meta["ckv_cache"], flashinfer_meta["kpe_cache"]
    )

    latent_out = torch.ones(batch_size, 1, num_heads, KV_LORA_RANK, dtype=DTYPE, device=DEVICE)
    weight_reshaped = inputs["kv_b_proj_weight"].view(
        num_heads, QK_NOPE_HEAD_DIM + V_HEAD_DIM, KV_LORA_RANK
    )
    w_v = weight_reshaped[:, QK_NOPE_HEAD_DIM:, :]
    expected = torch.einsum("bsnk,nvk->bsnv", latent_out, w_v)

    with patch.object(trtllm_mla_mod, "_is_blackwell_decode_supported", return_value=True):
        with patch.object(
            trtllm_mla_mod.flashinfer.mla,
            "trtllm_batch_decode_with_kv_cache_mla",
            return_value=latent_out,
        ) as kernel_mock:
            output = _run_trtllm_mla(inputs, flashinfer_meta, combined_cache, KV_LORA_RANK)

    kernel_mock.assert_called_once()
    assert torch.allclose(output.float(), expected.float(), atol=1e-4, rtol=1e-4)


# =============================================================================
# Test 3: Prefill (context phase)
# =============================================================================


@pytest.mark.parametrize("seq_length", [32, 128])
@pytest.mark.parametrize("num_heads", [1, 4])
@pytest.mark.parametrize("batch_size", [2, 8])
def test_flashinfer_trtllm_mla_prefill(seq_length, num_heads, batch_size):
    """Test pure prefill: seq_length > 1, no prior cache."""
    if torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("requires Hopper+")

    max_seq_len = 256
    max_num_pages = batch_size * (max_seq_len // PAGE_SIZE + 1)

    inputs = _create_mla_inputs(
        batch_size,
        seq_length,
        num_heads,
        QK_NOPE_HEAD_DIM,
        QK_ROPE_HEAD_DIM,
        KV_LORA_RANK,
        V_HEAD_DIM,
        DTYPE,
        DEVICE,
    )

    total_tokens = batch_size * seq_length
    q_nope_flat = inputs["q_nope"].view(1, total_tokens, num_heads, QK_NOPE_HEAD_DIM)
    q_pe_flat = inputs["q_pe"].view(1, total_tokens, num_heads, QK_ROPE_HEAD_DIM)
    compressed_kv_flat = inputs["compressed_kv"].view(1, total_tokens, KV_LORA_RANK)
    kpe_flat = inputs["kpe"].view(1, total_tokens, 1, QK_ROPE_HEAD_DIM)

    flat_inputs = {
        "q_nope": q_nope_flat,
        "q_pe": q_pe_flat,
        "compressed_kv": compressed_kv_flat,
        "kpe": kpe_flat,
        "kv_b_proj_weight": inputs["kv_b_proj_weight"],
    }

    seq_lengths = [seq_length] * batch_size
    input_positions = [0] * batch_size

    torch_meta = _create_unpaged_cache_and_metadata(
        batch_size,
        max_seq_len,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        DTYPE,
        DEVICE,
        seq_lengths,
        input_positions,
    )

    flashinfer_meta = _create_paged_cache_and_metadata(
        batch_size,
        max_num_pages,
        PAGE_SIZE,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        DTYPE,
        DEVICE,
        seq_lengths,
        input_positions,
    )
    combined_cache = _make_combined_cache(
        flashinfer_meta["ckv_cache"], flashinfer_meta["kpe_cache"]
    )

    torch_output = _run_torch_reference(flat_inputs, torch_meta, KV_LORA_RANK)
    trtllm_output = _run_trtllm_mla(flat_inputs, flashinfer_meta, combined_cache, KV_LORA_RANK)

    torch_out_r = torch_output.view(batch_size, seq_length, num_heads, V_HEAD_DIM)
    trtllm_out_r = trtllm_output.view(batch_size, seq_length, num_heads, V_HEAD_DIM)

    assert torch.allclose(trtllm_out_r.float(), torch_out_r.float(), atol=0.05, rtol=0.02), (
        f"Prefill output mismatch. Max diff: {(trtllm_out_r - torch_out_r).abs().max():.6f}"
    )


# =============================================================================
# Test 4: Decode with pre-filled cache (real computation, no mocking)
# =============================================================================


@pytest.mark.parametrize("prefill_seq_length", [64, 128])
@pytest.mark.parametrize("num_heads", [1, 4])
@pytest.mark.parametrize("batch_size", [2, 8])
def test_flashinfer_trtllm_mla_decode_with_cache(prefill_seq_length, num_heads, batch_size):
    """Test decode with pre-populated cache data."""
    if torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("requires Hopper+")

    seq_len = 1
    max_seq_len = 256
    max_num_pages = batch_size * (max_seq_len // PAGE_SIZE + 2)

    inputs = _create_mla_inputs(
        batch_size,
        seq_len,
        num_heads,
        QK_NOPE_HEAD_DIM,
        QK_ROPE_HEAD_DIM,
        KV_LORA_RANK,
        V_HEAD_DIM,
        DTYPE,
        DEVICE,
    )

    seq_lengths = [seq_len] * batch_size
    input_positions = [prefill_seq_length] * batch_size

    torch_meta = _create_unpaged_cache_and_metadata(
        batch_size,
        max_seq_len,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        DTYPE,
        DEVICE,
        seq_lengths,
        input_positions,
    )
    torch_meta["mla_cache"][:, :prefill_seq_length].normal_()

    flashinfer_meta = _create_paged_cache_and_metadata(
        batch_size,
        max_num_pages,
        PAGE_SIZE,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        DTYPE,
        DEVICE,
        seq_lengths,
        input_positions,
    )
    combined_cache = _make_combined_cache(
        flashinfer_meta["ckv_cache"], flashinfer_meta["kpe_cache"]
    )
    _copy_unpaged_to_combined_paged(
        torch_meta["mla_cache"],
        combined_cache,
        batch_size,
        [prefill_seq_length] * batch_size,
        PAGE_SIZE,
        flashinfer_meta["cu_num_pages"],
        flashinfer_meta["cache_loc"],
        KV_LORA_RANK,
    )

    torch_output = _run_torch_reference(inputs, torch_meta, KV_LORA_RANK)
    trtllm_output = _run_trtllm_mla(inputs, flashinfer_meta, combined_cache, KV_LORA_RANK)

    assert torch.allclose(trtllm_output.float(), torch_output.float(), atol=0.05, rtol=0.05), (
        f"Decode output mismatch. Max diff: {(trtllm_output - torch_output).abs().max():.6f}"
    )


# =============================================================================
# Test 5: Context (prefill) followed by generate (decode)
# =============================================================================


@pytest.mark.parametrize("prefill_seq_length", [32, 128])
@pytest.mark.parametrize("num_heads", [1, 4])
@pytest.mark.parametrize("batch_size", [2, 4])
def test_flashinfer_trtllm_mla_context_then_generate(prefill_seq_length, num_heads, batch_size):
    """Full workflow: prefill then a single decode step."""
    if torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("requires Hopper+")

    max_seq_len = 512
    max_num_pages = batch_size * (max_seq_len // PAGE_SIZE + 2)

    # --- Context phase ---
    inputs_ctx = _create_mla_inputs(
        batch_size,
        prefill_seq_length,
        num_heads,
        QK_NOPE_HEAD_DIM,
        QK_ROPE_HEAD_DIM,
        KV_LORA_RANK,
        V_HEAD_DIM,
        DTYPE,
        DEVICE,
    )

    total_tokens_ctx = batch_size * prefill_seq_length
    flat_inputs_ctx = {
        "q_nope": inputs_ctx["q_nope"].view(1, total_tokens_ctx, num_heads, QK_NOPE_HEAD_DIM),
        "q_pe": inputs_ctx["q_pe"].view(1, total_tokens_ctx, num_heads, QK_ROPE_HEAD_DIM),
        "compressed_kv": inputs_ctx["compressed_kv"].view(1, total_tokens_ctx, KV_LORA_RANK),
        "kpe": inputs_ctx["kpe"].view(1, total_tokens_ctx, 1, QK_ROPE_HEAD_DIM),
        "kv_b_proj_weight": inputs_ctx["kv_b_proj_weight"],
    }

    seq_lengths_ctx = [prefill_seq_length] * batch_size
    input_positions_ctx = [0] * batch_size

    torch_meta_ctx = _create_unpaged_cache_and_metadata(
        batch_size,
        max_seq_len,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        DTYPE,
        DEVICE,
        seq_lengths_ctx,
        input_positions_ctx,
    )

    flashinfer_meta_ctx = _create_paged_cache_and_metadata(
        batch_size,
        max_num_pages,
        PAGE_SIZE,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        DTYPE,
        DEVICE,
        seq_lengths_ctx,
        input_positions_ctx,
    )
    combined_cache = _make_combined_cache(
        flashinfer_meta_ctx["ckv_cache"], flashinfer_meta_ctx["kpe_cache"]
    )

    torch_out_ctx = _run_torch_reference(flat_inputs_ctx, torch_meta_ctx, KV_LORA_RANK)
    trtllm_out_ctx = _run_trtllm_mla(
        flat_inputs_ctx, flashinfer_meta_ctx, combined_cache, KV_LORA_RANK
    )

    torch_ctx_r = torch_out_ctx.view(batch_size, prefill_seq_length, num_heads, V_HEAD_DIM)
    trtllm_ctx_r = trtllm_out_ctx.view(batch_size, prefill_seq_length, num_heads, V_HEAD_DIM)
    assert torch.allclose(trtllm_ctx_r.float(), torch_ctx_r.float(), atol=0.01, rtol=0.01), (
        f"Context phase mismatch. Max diff: {(trtllm_ctx_r - torch_ctx_r).abs().max():.6f}"
    )

    # --- Generate phase ---
    inputs_gen = _create_mla_inputs(
        batch_size,
        1,
        num_heads,
        QK_NOPE_HEAD_DIM,
        QK_ROPE_HEAD_DIM,
        KV_LORA_RANK,
        V_HEAD_DIM,
        DTYPE,
        DEVICE,
    )
    inputs_gen["kv_b_proj_weight"] = inputs_ctx["kv_b_proj_weight"]

    seq_lengths_gen = [1] * batch_size
    input_positions_gen = [prefill_seq_length] * batch_size

    torch_meta_gen = _create_unpaged_cache_and_metadata(
        batch_size,
        max_seq_len,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        DTYPE,
        DEVICE,
        seq_lengths_gen,
        input_positions_gen,
    )
    # Copy the torch cache from context phase (it was mutated in-place)
    torch_meta_gen["mla_cache"] = torch_meta_ctx["mla_cache"]

    flashinfer_meta_gen = _create_paged_cache_and_metadata(
        batch_size,
        max_num_pages,
        PAGE_SIZE,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        DTYPE,
        DEVICE,
        seq_lengths_gen,
        input_positions_gen,
    )
    # Reuse the combined cache from context phase (it was mutated in-place)
    # But we need to update the metadata pointers to the same cache_loc mapping
    # The paged metadata for decode needs its own page assignments that match the
    # existing cache layout. We reconstruct by copying from the context cache.
    combined_cache_gen = _make_combined_cache(
        flashinfer_meta_gen["ckv_cache"], flashinfer_meta_gen["kpe_cache"]
    )
    _copy_unpaged_to_combined_paged(
        torch_meta_ctx["mla_cache"],
        combined_cache_gen,
        batch_size,
        [prefill_seq_length] * batch_size,
        PAGE_SIZE,
        flashinfer_meta_gen["cu_num_pages"],
        flashinfer_meta_gen["cache_loc"],
        KV_LORA_RANK,
    )

    torch_out_gen = _run_torch_reference(inputs_gen, torch_meta_gen, KV_LORA_RANK)
    trtllm_out_gen = _run_trtllm_mla(
        inputs_gen, flashinfer_meta_gen, combined_cache_gen, KV_LORA_RANK
    )

    assert torch.allclose(trtllm_out_gen.float(), torch_out_gen.float(), atol=0.05, rtol=0.02), (
        f"Generate phase mismatch. Max diff: {(trtllm_out_gen - torch_out_gen).abs().max():.6f}"
    )


# =============================================================================
# Test 6: Multi-step decode
# =============================================================================


def test_flashinfer_trtllm_mla_multi_step_decode():
    """Prefill followed by 5 sequential decode steps."""
    if torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("requires Hopper+")

    batch_size = 4
    num_heads = 4
    prefill_seq_length = 64
    num_decode_steps = 5
    max_seq_len = 512
    max_num_pages = batch_size * (max_seq_len // PAGE_SIZE + 2)

    # --- Context phase ---
    inputs_ctx = _create_mla_inputs(
        batch_size,
        prefill_seq_length,
        num_heads,
        QK_NOPE_HEAD_DIM,
        QK_ROPE_HEAD_DIM,
        KV_LORA_RANK,
        V_HEAD_DIM,
        DTYPE,
        DEVICE,
    )
    kv_b_proj_weight = inputs_ctx["kv_b_proj_weight"]

    total_tokens_ctx = batch_size * prefill_seq_length
    flat_inputs_ctx = {
        "q_nope": inputs_ctx["q_nope"].view(1, total_tokens_ctx, num_heads, QK_NOPE_HEAD_DIM),
        "q_pe": inputs_ctx["q_pe"].view(1, total_tokens_ctx, num_heads, QK_ROPE_HEAD_DIM),
        "compressed_kv": inputs_ctx["compressed_kv"].view(1, total_tokens_ctx, KV_LORA_RANK),
        "kpe": inputs_ctx["kpe"].view(1, total_tokens_ctx, 1, QK_ROPE_HEAD_DIM),
        "kv_b_proj_weight": kv_b_proj_weight,
    }

    torch_meta_ctx = _create_unpaged_cache_and_metadata(
        batch_size,
        max_seq_len,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        DTYPE,
        DEVICE,
        [prefill_seq_length] * batch_size,
        [0] * batch_size,
    )

    flashinfer_meta_ctx = _create_paged_cache_and_metadata(
        batch_size,
        max_num_pages,
        PAGE_SIZE,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        DTYPE,
        DEVICE,
        [prefill_seq_length] * batch_size,
        [0] * batch_size,
    )
    combined_cache_ctx = _make_combined_cache(
        flashinfer_meta_ctx["ckv_cache"], flashinfer_meta_ctx["kpe_cache"]
    )

    _run_torch_reference(flat_inputs_ctx, torch_meta_ctx, KV_LORA_RANK)
    _run_trtllm_mla(flat_inputs_ctx, flashinfer_meta_ctx, combined_cache_ctx, KV_LORA_RANK)

    # --- Decode steps ---
    for step in range(num_decode_steps):
        current_pos = prefill_seq_length + step

        inputs_dec = _create_mla_inputs(
            batch_size,
            1,
            num_heads,
            QK_NOPE_HEAD_DIM,
            QK_ROPE_HEAD_DIM,
            KV_LORA_RANK,
            V_HEAD_DIM,
            DTYPE,
            DEVICE,
        )
        inputs_dec["kv_b_proj_weight"] = kv_b_proj_weight

        torch_meta_dec = _create_unpaged_cache_and_metadata(
            batch_size,
            max_seq_len,
            KV_LORA_RANK,
            QK_ROPE_HEAD_DIM,
            DTYPE,
            DEVICE,
            [1] * batch_size,
            [current_pos] * batch_size,
        )
        torch_meta_dec["mla_cache"] = torch_meta_ctx["mla_cache"]

        flashinfer_meta_dec = _create_paged_cache_and_metadata(
            batch_size,
            max_num_pages,
            PAGE_SIZE,
            KV_LORA_RANK,
            QK_ROPE_HEAD_DIM,
            DTYPE,
            DEVICE,
            [1] * batch_size,
            [current_pos] * batch_size,
        )
        combined_cache_dec = _make_combined_cache(
            flashinfer_meta_dec["ckv_cache"], flashinfer_meta_dec["kpe_cache"]
        )
        _copy_unpaged_to_combined_paged(
            torch_meta_ctx["mla_cache"],
            combined_cache_dec,
            batch_size,
            [current_pos] * batch_size,
            PAGE_SIZE,
            flashinfer_meta_dec["cu_num_pages"],
            flashinfer_meta_dec["cache_loc"],
            KV_LORA_RANK,
        )

        torch_out = _run_torch_reference(inputs_dec, torch_meta_dec, KV_LORA_RANK)
        trtllm_out = _run_trtllm_mla(
            inputs_dec, flashinfer_meta_dec, combined_cache_dec, KV_LORA_RANK
        )

        assert torch.allclose(trtllm_out.float(), torch_out.float(), atol=0.05, rtol=0.05), (
            f"Multi-step decode step {step} mismatch. "
            f"Max diff: {(trtllm_out - torch_out).abs().max():.6f}"
        )


# =============================================================================
# Test 7: Variable sequence lengths in prefill
# =============================================================================


def test_flashinfer_trtllm_mla_variable_seq_lengths():
    """Prefill with different sequence lengths per sequence."""
    if torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("requires Hopper+")

    num_heads = 4
    seq_lengths = [16, 32, 64]
    batch_size = len(seq_lengths)
    max_seq_len = 256
    max_num_pages = batch_size * (max_seq_len // PAGE_SIZE + 2)

    # Create individual per-sequence inputs then concatenate
    q_nope_parts = []
    q_pe_parts = []
    ckv_parts = []
    kpe_parts = []
    for sl in seq_lengths:
        inp = _create_mla_inputs(
            1,
            sl,
            num_heads,
            QK_NOPE_HEAD_DIM,
            QK_ROPE_HEAD_DIM,
            KV_LORA_RANK,
            V_HEAD_DIM,
            DTYPE,
            DEVICE,
        )
        q_nope_parts.append(inp["q_nope"].view(sl, num_heads, QK_NOPE_HEAD_DIM))
        q_pe_parts.append(inp["q_pe"].view(sl, num_heads, QK_ROPE_HEAD_DIM))
        ckv_parts.append(inp["compressed_kv"].view(sl, KV_LORA_RANK))
        kpe_parts.append(inp["kpe"].view(sl, 1, QK_ROPE_HEAD_DIM))
        kv_b_proj_weight = inp["kv_b_proj_weight"]

    q_nope_cat = torch.cat(q_nope_parts, dim=0).unsqueeze(0)
    q_pe_cat = torch.cat(q_pe_parts, dim=0).unsqueeze(0)
    ckv_cat = torch.cat(ckv_parts, dim=0).unsqueeze(0)
    kpe_cat = torch.cat(kpe_parts, dim=0).unsqueeze(0)

    flat_inputs = {
        "q_nope": q_nope_cat,
        "q_pe": q_pe_cat,
        "compressed_kv": ckv_cat,
        "kpe": kpe_cat,
        "kv_b_proj_weight": kv_b_proj_weight,
    }

    input_positions = [0] * batch_size

    torch_meta = _create_unpaged_cache_and_metadata(
        batch_size,
        max_seq_len,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        DTYPE,
        DEVICE,
        seq_lengths,
        input_positions,
    )

    flashinfer_meta = _create_paged_cache_and_metadata(
        batch_size,
        max_num_pages,
        PAGE_SIZE,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        DTYPE,
        DEVICE,
        seq_lengths,
        input_positions,
    )
    combined_cache = _make_combined_cache(
        flashinfer_meta["ckv_cache"], flashinfer_meta["kpe_cache"]
    )

    torch_output = _run_torch_reference(flat_inputs, torch_meta, KV_LORA_RANK)
    trtllm_output = _run_trtllm_mla(flat_inputs, flashinfer_meta, combined_cache, KV_LORA_RANK)

    assert torch.allclose(trtllm_output.float(), torch_output.float(), atol=0.05, rtol=0.02), (
        f"Variable seq length prefill mismatch. "
        f"Max diff: {(trtllm_output - torch_output).abs().max():.6f}"
    )


# =============================================================================
# Test 8: Mixed batch (prefill + decode in same call)
# =============================================================================


def test_flashinfer_trtllm_mla_mixed_batch():
    """Mixed batch: some sequences in prefill, others in decode."""
    if torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("requires Hopper+")

    num_heads = 4
    num_prefill = 2
    prefill_seq_length = 32
    num_decode = 2
    prefill_cache_length = 64  # decode sequences have 64 cached tokens
    batch_size = num_prefill + num_decode
    max_seq_len = 256
    max_num_pages = batch_size * (max_seq_len // PAGE_SIZE + 2)

    # Sequence lengths: prefill seqs have seq_length tokens, decode seqs have 1 token
    seq_lengths = [prefill_seq_length] * num_prefill + [1] * num_decode

    # Input positions: prefill starts at 0, decode at prefill_cache_length
    input_positions = [0] * num_prefill + [prefill_cache_length] * num_decode

    # Create inputs per-sequence and concatenate
    q_nope_parts = []
    q_pe_parts = []
    ckv_parts = []
    kpe_parts = []
    for sl in seq_lengths:
        inp = _create_mla_inputs(
            1,
            sl,
            num_heads,
            QK_NOPE_HEAD_DIM,
            QK_ROPE_HEAD_DIM,
            KV_LORA_RANK,
            V_HEAD_DIM,
            DTYPE,
            DEVICE,
        )
        q_nope_parts.append(inp["q_nope"].view(sl, num_heads, QK_NOPE_HEAD_DIM))
        q_pe_parts.append(inp["q_pe"].view(sl, num_heads, QK_ROPE_HEAD_DIM))
        ckv_parts.append(inp["compressed_kv"].view(sl, KV_LORA_RANK))
        kpe_parts.append(inp["kpe"].view(sl, 1, QK_ROPE_HEAD_DIM))
    kv_b_proj_weight = inp["kv_b_proj_weight"]

    q_nope_cat = torch.cat(q_nope_parts, dim=0).unsqueeze(0)
    q_pe_cat = torch.cat(q_pe_parts, dim=0).unsqueeze(0)
    ckv_cat = torch.cat(ckv_parts, dim=0).unsqueeze(0)
    kpe_cat = torch.cat(kpe_parts, dim=0).unsqueeze(0)

    flat_inputs = {
        "q_nope": q_nope_cat,
        "q_pe": q_pe_cat,
        "compressed_kv": ckv_cat,
        "kpe": kpe_cat,
        "kv_b_proj_weight": kv_b_proj_weight,
    }

    # Torch reference metadata
    torch_meta = _create_unpaged_cache_and_metadata(
        batch_size,
        max_seq_len,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        DTYPE,
        DEVICE,
        seq_lengths,
        input_positions,
    )
    # Pre-fill cache for decode sequences
    for i in range(num_prefill, batch_size):
        torch_meta["mla_cache"][i, :prefill_cache_length].normal_()

    # Build mixed BatchInfo manually
    _bi = BatchInfo()
    num_prefill_tokens = num_prefill * prefill_seq_length
    _bi.update([num_prefill, num_prefill_tokens, 0, 0, num_decode, num_decode])
    torch_meta["batch_info_host"] = _bi.serialize()

    # Paged metadata
    flashinfer_meta = _create_paged_cache_and_metadata(
        batch_size,
        max_num_pages,
        PAGE_SIZE,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        DTYPE,
        DEVICE,
        seq_lengths,
        input_positions,
    )
    # Override batch_info for mixed batch
    flashinfer_meta["batch_info_host"] = _bi.serialize()

    combined_cache = _make_combined_cache(
        flashinfer_meta["ckv_cache"], flashinfer_meta["kpe_cache"]
    )
    # Copy decode sequences' pre-filled cache
    for i in range(num_prefill, batch_size):
        page_start = int(flashinfer_meta["cu_num_pages"][i].item())
        page_end = int(flashinfer_meta["cu_num_pages"][i + 1].item())
        token_offset = 0
        for flat_idx in range(page_start, page_end):
            page_idx = int(flashinfer_meta["cache_loc"][flat_idx].item())
            take = min(PAGE_SIZE, prefill_cache_length - token_offset)
            if take <= 0:
                break
            data = torch_meta["mla_cache"][i, token_offset : token_offset + take]
            combined_cache[page_idx, :take] = data
            token_offset += take

    torch_output = _run_torch_reference(flat_inputs, torch_meta, KV_LORA_RANK)
    trtllm_output = _run_trtllm_mla(flat_inputs, flashinfer_meta, combined_cache, KV_LORA_RANK)

    assert torch.allclose(trtllm_output.float(), torch_output.float(), atol=0.05, rtol=0.05), (
        f"Mixed batch mismatch. Max diff: {(trtllm_output - torch_output).abs().max():.6f}"
    )


# =============================================================================
# Test 9: Edge case - single sequence decode with long cache
# =============================================================================


@pytest.mark.parametrize("cache_length", [1, 63, 64, 127, 128, 255, 256])
def test_flashinfer_trtllm_mla_decode_various_cache_lengths(cache_length):
    """Decode with various cache lengths to test page boundary handling."""
    if torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("requires Hopper+")

    batch_size = 1
    num_heads = 4
    max_seq_len = 512
    max_num_pages = batch_size * (max_seq_len // PAGE_SIZE + 2)

    inputs = _create_mla_inputs(
        batch_size,
        1,
        num_heads,
        QK_NOPE_HEAD_DIM,
        QK_ROPE_HEAD_DIM,
        KV_LORA_RANK,
        V_HEAD_DIM,
        DTYPE,
        DEVICE,
    )

    torch_meta = _create_unpaged_cache_and_metadata(
        batch_size,
        max_seq_len,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        DTYPE,
        DEVICE,
        [1],
        [cache_length],
    )
    torch_meta["mla_cache"][:, :cache_length].normal_()

    flashinfer_meta = _create_paged_cache_and_metadata(
        batch_size,
        max_num_pages,
        PAGE_SIZE,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        DTYPE,
        DEVICE,
        [1],
        [cache_length],
    )
    combined_cache = _make_combined_cache(
        flashinfer_meta["ckv_cache"], flashinfer_meta["kpe_cache"]
    )
    _copy_unpaged_to_combined_paged(
        torch_meta["mla_cache"],
        combined_cache,
        batch_size,
        [cache_length],
        PAGE_SIZE,
        flashinfer_meta["cu_num_pages"],
        flashinfer_meta["cache_loc"],
        KV_LORA_RANK,
    )

    torch_output = _run_torch_reference(inputs, torch_meta, KV_LORA_RANK)
    trtllm_output = _run_trtllm_mla(inputs, flashinfer_meta, combined_cache, KV_LORA_RANK)

    assert torch.allclose(trtllm_output.float(), torch_output.float(), atol=0.05, rtol=0.05), (
        f"Decode cache_length={cache_length} mismatch. "
        f"Max diff: {(trtllm_output - torch_output).abs().max():.6f}"
    )

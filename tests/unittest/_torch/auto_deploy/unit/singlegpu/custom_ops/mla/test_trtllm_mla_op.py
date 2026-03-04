# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Test TRT-LLM MLA backend operations.

Tests the trtllm_mla_with_cache cached op by comparing its output against the
torch_mla source op (reference implementation).

Key features tested:
- 5 tensor arguments: q_nope, q_pe, compressed_kv, kpe, kv_b_proj_weight
- Paged latent cache with thop.attention (is_mla_enable=True)
- Prefill: expand compressed_kv, build fused QKV, call thop.attention
- Decode: weight absorption + latent-space attention + output projection
- Multi-step generation (prefill then multiple decode steps)
"""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy  # noqa: F401
from tensorrt_llm._torch.auto_deploy.custom_ops.mla.trtllm_mla import _GlobalTrtllmMLAPlanner

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (8, 0),
    reason="TRT-LLM MLA tests require GPU with compute capability >= 8.0",
)

_MLA_TOKENS_PER_BLOCK = 32


def _create_mla_inputs(
    batch_size,
    seq_len,
    num_heads,
    qk_nope_head_dim,
    qk_rope_head_dim,
    kv_lora_rank,
    v_head_dim,
    dtype,
    device,
):
    """Create MLA input tensors with Xavier-like initialization."""
    kv_head_dim = qk_nope_head_dim + v_head_dim
    q_scale = 1.0 / (qk_nope_head_dim**0.5)
    kv_scale = 1.0 / (kv_lora_rank**0.5)
    weight_scale = 1.0 / (kv_lora_rank**0.5)

    return {
        "q_nope": torch.randn(
            batch_size,
            seq_len,
            num_heads,
            qk_nope_head_dim,
            dtype=dtype,
            device=device,
        )
        * q_scale,
        "q_pe": torch.randn(
            batch_size,
            seq_len,
            num_heads,
            qk_rope_head_dim,
            dtype=dtype,
            device=device,
        )
        * q_scale,
        "compressed_kv": torch.randn(
            batch_size,
            seq_len,
            kv_lora_rank,
            dtype=dtype,
            device=device,
        )
        * kv_scale,
        "kpe": torch.randn(
            batch_size,
            seq_len,
            1,
            qk_rope_head_dim,
            dtype=dtype,
            device=device,
        )
        * q_scale,
        "kv_b_proj_weight": torch.randn(
            num_heads * kv_head_dim,
            kv_lora_rank,
            dtype=dtype,
            device=device,
        )
        * weight_scale,
    }


def _create_trtllm_paged_metadata(
    batch_size,
    max_num_pages,
    page_size,
    kv_lora_rank,
    qk_rope_head_dim,
    dtype,
    device,
    seq_lengths,
    input_positions,
):
    """Create paged cache and TRT-LLM-style metadata for MLA.

    Returns a dict with kv_cache, batch_info_host, seq_len, seq_len_host,
    input_pos_host, seq_len_with_cache, and max_seq_info_host tensors.
    """
    latent_dim = kv_lora_rank + qk_rope_head_dim
    num_kv_heads = 1

    # HND paged cache: [num_pages, 2, num_kv_heads, page_size, latent_dim]
    kv_cache = torch.zeros(
        max_num_pages,
        2,
        num_kv_heads,
        page_size,
        latent_dim,
        dtype=dtype,
        device=device,
    )

    kv_lengths = [pos + slen for pos, slen in zip(input_positions, seq_lengths)]
    pages_per_seq = [
        max(1, (kv_len - 1) // page_size + 1) if kv_len > 0 else 1 for kv_len in kv_lengths
    ]

    # Sequential page assignment
    page_assignments = []
    next_page = 0
    for np_ in pages_per_seq:
        page_assignments.append(list(range(next_page, next_page + np_)))
        next_page += np_

    total_tokens = sum(seq_lengths)
    is_decode = all(s == 1 for s in seq_lengths)

    if is_decode:
        batch_info_host = torch.tensor(
            [0, 0, batch_size],
            dtype=torch.int32,
            device="cpu",
        )
    else:
        batch_info_host = torch.tensor(
            [batch_size, total_tokens, 0],
            dtype=torch.int32,
            device="cpu",
        )

    seq_len_tensor = torch.tensor(seq_lengths, dtype=torch.int32, device=device)
    seq_len_host = seq_len_tensor.cpu()

    input_pos_host = torch.tensor(input_positions, dtype=torch.int32, device="cpu")

    seq_len_with_cache = torch.tensor(kv_lengths, dtype=torch.int32, device=device)
    seq_len_with_cache_host = seq_len_with_cache.cpu()

    # Page metadata for the planner
    num_pages_tensor = torch.tensor(pages_per_seq, dtype=torch.int32, device="cpu")
    cu_num_pages_host = torch.zeros(batch_size + 1, dtype=torch.int32, device="cpu")
    cu_num_pages_host[1:] = torch.cumsum(num_pages_tensor, dim=0)

    cache_loc = torch.tensor(
        [p for pages in page_assignments for p in pages],
        dtype=torch.int32,
        device=device,
    )

    # Pre-compute page_seq_indices and page_in_seq for the planner
    page_seq_indices_list = []
    page_in_seq_list = []
    for seq_idx, pages in enumerate(page_assignments):
        for pg_idx_in_seq, _ in enumerate(pages):
            page_seq_indices_list.append(seq_idx)
            page_in_seq_list.append(pg_idx_in_seq)
    page_seq_indices = torch.tensor(
        page_seq_indices_list,
        dtype=torch.int32,
        device=device,
    )
    page_in_seq = torch.tensor(
        page_in_seq_list,
        dtype=torch.int32,
        device=device,
    )

    # block_offset_multiplier: for HND with kv_factor=2, each page occupies 2 slots
    block_offset_multiplier = 2
    max_blocks_per_seq = max(pages_per_seq) if pages_per_seq else 1
    max_context_length = max(kv_lengths) if kv_lengths else 1

    max_seq_info_host = torch.tensor(
        [max_context_length, max_blocks_per_seq, block_offset_multiplier, batch_size],
        dtype=torch.int32,
        device="cpu",
    )

    return {
        "kv_cache": kv_cache,
        "batch_info_host": batch_info_host,
        "seq_len": seq_len_tensor,
        "seq_len_host": seq_len_host,
        "input_pos_host": input_pos_host,
        "seq_len_with_cache": seq_len_with_cache,
        "seq_len_with_cache_host": seq_len_with_cache_host,
        "max_seq_info_host": max_seq_info_host,
        "cu_num_pages_host": cu_num_pages_host,
        "cache_loc": cache_loc,
        "page_seq_indices": page_seq_indices,
        "page_in_seq": page_in_seq,
    }


def _run_trtllm_mla(inputs, meta, kv_lora_rank):
    """Run the trtllm_mla_with_cache op with proper planner setup."""
    # Reset planner so it re-initializes
    _GlobalTrtllmMLAPlanner.__init__()

    # Host-side prepare (fills block_offsets, request types, etc.)
    from tensorrt_llm._torch.auto_deploy.custom_ops.mla.trtllm_mla import (
        prepare_trtllm_mla_metadata_host,
    )

    prepare_trtllm_mla_metadata_host(
        meta["batch_info_host"],
        meta["max_seq_info_host"],
        meta["seq_len_with_cache_host"],
        meta["cu_num_pages_host"],
        meta["cache_loc"],
        meta["page_seq_indices"],
        meta["page_in_seq"],
        meta["input_pos_host"],
        meta["seq_len_host"],
    )

    return torch.ops.auto_deploy.trtllm_mla_with_cache(
        inputs["q_nope"],
        inputs["q_pe"],
        inputs["compressed_kv"],
        inputs["kpe"],
        inputs["kv_b_proj_weight"],
        meta["batch_info_host"],
        meta["seq_len"],
        meta["seq_len_host"],
        meta["input_pos_host"],
        meta["seq_len_with_cache"],
        meta["max_seq_info_host"],
        meta["kv_cache"],
        None,  # scale
        kv_lora_rank,
    )


def _run_torch_mla_reference(inputs):
    """Run the torch_mla source op (pure PyTorch reference)."""
    return torch.ops.auto_deploy.torch_mla(
        inputs["q_nope"],
        inputs["q_pe"],
        inputs["compressed_kv"],
        inputs["kpe"],
        inputs["kv_b_proj_weight"],
        True,  # is_causal
        None,  # scale
        "bsnd",
    )


@pytest.mark.parametrize("seq_length", [32, 64])
@pytest.mark.parametrize("num_heads", [1, 4])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ["cuda"])
def test_trtllm_mla_prefill(seq_length, num_heads, batch_size, dtype, device):
    """Test TRT-LLM MLA prefill against torch_mla source op."""
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    kv_lora_rank = 512
    v_head_dim = 128
    page_size = _MLA_TOKENS_PER_BLOCK
    max_seq_len = 256
    max_num_pages = batch_size * (max_seq_len // page_size + 2)

    inputs = _create_mla_inputs(
        batch_size,
        seq_length,
        num_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        kv_lora_rank,
        v_head_dim,
        dtype,
        device,
    )

    # Reference: run source op per-batch (source op expects [B, S, ...] layout)
    ref_output = _run_torch_mla_reference(inputs)

    # TRT-LLM: flatten batch into [1, total_tokens, ...] for the cached op
    total_tokens = batch_size * seq_length
    trtllm_inputs = {
        "q_nope": inputs["q_nope"].reshape(1, total_tokens, num_heads, qk_nope_head_dim),
        "q_pe": inputs["q_pe"].reshape(1, total_tokens, num_heads, qk_rope_head_dim),
        "compressed_kv": inputs["compressed_kv"].reshape(1, total_tokens, kv_lora_rank),
        "kpe": inputs["kpe"].reshape(1, total_tokens, 1, qk_rope_head_dim),
        "kv_b_proj_weight": inputs["kv_b_proj_weight"],
    }

    seq_lengths = [seq_length] * batch_size
    input_positions = [0] * batch_size

    meta = _create_trtllm_paged_metadata(
        batch_size,
        max_num_pages,
        page_size,
        kv_lora_rank,
        qk_rope_head_dim,
        dtype,
        device,
        seq_lengths,
        input_positions,
    )

    trtllm_output = _run_trtllm_mla(trtllm_inputs, meta, kv_lora_rank)
    trtllm_output = trtllm_output.view(batch_size, seq_length, num_heads, v_head_dim)

    torch.testing.assert_close(
        trtllm_output,
        ref_output,
        atol=5e-2,
        rtol=5e-2,
    )


@pytest.mark.parametrize("num_heads", [1, 4])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ["cuda"])
def test_trtllm_mla_multi_step(num_heads, batch_size, dtype, device):
    """Test multi-step: prefill followed by decode steps.

    Verifies that the decode output after populating the cache via prefill
    is consistent with the torch_mla reference that sees the full sequence.
    """
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    kv_lora_rank = 512
    v_head_dim = 128
    prefill_len = 16
    num_decode_steps = 4
    page_size = _MLA_TOKENS_PER_BLOCK
    max_seq_len = prefill_len + num_decode_steps + 8
    max_num_pages = batch_size * (max_seq_len // page_size + 2)

    # Generate all tokens upfront for reference
    total_len = prefill_len + num_decode_steps
    all_inputs = _create_mla_inputs(
        batch_size,
        total_len,
        num_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        kv_lora_rank,
        v_head_dim,
        dtype,
        device,
    )

    # --- Step 1: Prefill ---
    prefill_inputs = {
        "q_nope": all_inputs["q_nope"][:, :prefill_len],
        "q_pe": all_inputs["q_pe"][:, :prefill_len],
        "compressed_kv": all_inputs["compressed_kv"][:, :prefill_len],
        "kpe": all_inputs["kpe"][:, :prefill_len],
        "kv_b_proj_weight": all_inputs["kv_b_proj_weight"],
    }

    total_prefill_tokens = batch_size * prefill_len
    trtllm_prefill_inputs = {
        "q_nope": prefill_inputs["q_nope"].reshape(
            1,
            total_prefill_tokens,
            num_heads,
            qk_nope_head_dim,
        ),
        "q_pe": prefill_inputs["q_pe"].reshape(
            1,
            total_prefill_tokens,
            num_heads,
            qk_rope_head_dim,
        ),
        "compressed_kv": prefill_inputs["compressed_kv"].reshape(
            1,
            total_prefill_tokens,
            kv_lora_rank,
        ),
        "kpe": prefill_inputs["kpe"].reshape(
            1,
            total_prefill_tokens,
            1,
            qk_rope_head_dim,
        ),
        "kv_b_proj_weight": all_inputs["kv_b_proj_weight"],
    }

    prefill_meta = _create_trtllm_paged_metadata(
        batch_size,
        max_num_pages,
        page_size,
        kv_lora_rank,
        qk_rope_head_dim,
        dtype,
        device,
        [prefill_len] * batch_size,
        [0] * batch_size,
    )

    _run_trtllm_mla(trtllm_prefill_inputs, prefill_meta, kv_lora_rank)

    # --- Step 2: Decode steps ---
    for step in range(num_decode_steps):
        token_idx = prefill_len + step
        decode_inputs = {
            "q_nope": all_inputs["q_nope"][:, token_idx : token_idx + 1],
            "q_pe": all_inputs["q_pe"][:, token_idx : token_idx + 1],
            "compressed_kv": all_inputs["compressed_kv"][:, token_idx : token_idx + 1],
            "kpe": all_inputs["kpe"][:, token_idx : token_idx + 1],
            "kv_b_proj_weight": all_inputs["kv_b_proj_weight"],
        }

        decode_meta = _create_trtllm_paged_metadata(
            batch_size,
            max_num_pages,
            page_size,
            kv_lora_rank,
            qk_rope_head_dim,
            dtype,
            device,
            [1] * batch_size,
            [token_idx] * batch_size,
        )
        # Reuse the same kv_cache across steps
        decode_meta["kv_cache"] = prefill_meta["kv_cache"]

        trtllm_decode_out = _run_trtllm_mla(decode_inputs, decode_meta, kv_lora_rank)

        # Reference: run source op on the full sequence up to this point
        ref_inputs = {
            "q_nope": all_inputs["q_nope"][:, token_idx : token_idx + 1],
            "q_pe": all_inputs["q_pe"][:, token_idx : token_idx + 1],
            "compressed_kv": all_inputs["compressed_kv"][:, : token_idx + 1],
            "kpe": all_inputs["kpe"][:, : token_idx + 1],
            "kv_b_proj_weight": all_inputs["kv_b_proj_weight"],
        }

        # torch_mla expects matching Q and KV seq lengths for non-causal,
        # but for generate (Q_len=1, KV_len>1) we need to handle this specially.
        # Use the source op with full KV to get the expected output.
        ref_output = torch.ops.auto_deploy.torch_mla(
            ref_inputs["q_nope"],
            ref_inputs["q_pe"],
            ref_inputs["compressed_kv"],
            ref_inputs["kpe"],
            ref_inputs["kv_b_proj_weight"],
            False,  # is_causal=False since Q_len != KV_len
            None,
            "bsnd",
        )

        torch.testing.assert_close(
            trtllm_decode_out.view(batch_size, 1, num_heads, v_head_dim),
            ref_output,
            atol=5e-2,
            rtol=5e-2,
        )


@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ["cuda"])
def test_trtllm_mla_chunked_prefill(num_heads, batch_size, dtype, device):
    """Test chunked prefill: two prefill chunks followed by a decode step.

    Simulates long prompt processing where the prompt is split into chunks.
    The second chunk must attend to tokens cached from the first chunk.
    If chunked prefill is broken, chunk 2 only sees its own tokens and the
    decode output will be incorrect because the KV cache is stale.
    """
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    kv_lora_rank = 512
    v_head_dim = 128
    chunk1_len = 16
    chunk2_len = 16
    total_prefill = chunk1_len + chunk2_len
    page_size = _MLA_TOKENS_PER_BLOCK
    max_seq_len = total_prefill + 8
    max_num_pages = batch_size * (max_seq_len // page_size + 2)

    all_inputs = _create_mla_inputs(
        batch_size,
        total_prefill + 1,  # +1 for decode step
        num_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        kv_lora_rank,
        v_head_dim,
        dtype,
        device,
    )

    # --- Chunk 1: regular prefill (input_pos=0) ---
    total_chunk1_tokens = batch_size * chunk1_len
    chunk1_inputs = {
        "q_nope": all_inputs["q_nope"][:, :chunk1_len].reshape(
            1, total_chunk1_tokens, num_heads, qk_nope_head_dim
        ),
        "q_pe": all_inputs["q_pe"][:, :chunk1_len].reshape(
            1, total_chunk1_tokens, num_heads, qk_rope_head_dim
        ),
        "compressed_kv": all_inputs["compressed_kv"][:, :chunk1_len].reshape(
            1, total_chunk1_tokens, kv_lora_rank
        ),
        "kpe": all_inputs["kpe"][:, :chunk1_len].reshape(
            1, total_chunk1_tokens, 1, qk_rope_head_dim
        ),
        "kv_b_proj_weight": all_inputs["kv_b_proj_weight"],
    }

    chunk1_meta = _create_trtllm_paged_metadata(
        batch_size,
        max_num_pages,
        page_size,
        kv_lora_rank,
        qk_rope_head_dim,
        dtype,
        device,
        [chunk1_len] * batch_size,
        [0] * batch_size,
    )

    _run_trtllm_mla(chunk1_inputs, chunk1_meta, kv_lora_rank)

    # --- Chunk 2: chunked prefill (input_pos=chunk1_len, already has cached tokens) ---
    total_chunk2_tokens = batch_size * chunk2_len
    chunk2_inputs = {
        "q_nope": all_inputs["q_nope"][:, chunk1_len:total_prefill].reshape(
            1, total_chunk2_tokens, num_heads, qk_nope_head_dim
        ),
        "q_pe": all_inputs["q_pe"][:, chunk1_len:total_prefill].reshape(
            1, total_chunk2_tokens, num_heads, qk_rope_head_dim
        ),
        "compressed_kv": all_inputs["compressed_kv"][:, chunk1_len:total_prefill].reshape(
            1, total_chunk2_tokens, kv_lora_rank
        ),
        "kpe": all_inputs["kpe"][:, chunk1_len:total_prefill].reshape(
            1, total_chunk2_tokens, 1, qk_rope_head_dim
        ),
        "kv_b_proj_weight": all_inputs["kv_b_proj_weight"],
    }

    chunk2_meta = _create_trtllm_paged_metadata(
        batch_size,
        max_num_pages,
        page_size,
        kv_lora_rank,
        qk_rope_head_dim,
        dtype,
        device,
        [chunk2_len] * batch_size,
        [chunk1_len] * batch_size,  # input_pos > 0: chunked prefill
    )
    chunk2_meta["kv_cache"] = chunk1_meta["kv_cache"]

    chunk2_out = _run_trtllm_mla(chunk2_inputs, chunk2_meta, kv_lora_rank)

    # Reference: torch_mla on the full sequence for chunk2 query positions
    # For chunked prefill, each query token should attend to ALL preceding tokens
    # (from both chunk 1 and chunk 2).
    ref_full_output = torch.ops.auto_deploy.torch_mla(
        all_inputs["q_nope"][:, :total_prefill],
        all_inputs["q_pe"][:, :total_prefill],
        all_inputs["compressed_kv"][:, :total_prefill],
        all_inputs["kpe"][:, :total_prefill],
        all_inputs["kv_b_proj_weight"],
        True,  # is_causal
        None,
        "bsnd",
    )
    ref_chunk2 = ref_full_output[:, chunk1_len:total_prefill]

    actual_chunk2 = chunk2_out.view(batch_size, chunk2_len, num_heads, v_head_dim)

    torch.testing.assert_close(
        actual_chunk2,
        ref_chunk2,
        atol=5e-3,
        rtol=5e-3,
    )

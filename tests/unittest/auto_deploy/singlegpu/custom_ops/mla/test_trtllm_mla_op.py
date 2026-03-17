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
- Prefill: weight absorption + thop.attention (context_only) + W_v projection
- Decode: weight absorption + thop.attention (generation_only) + W_v projection
- Multi-step generation (prefill then multiple decode steps)
"""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy  # noqa: F401
from tensorrt_llm._torch.auto_deploy.custom_ops.mla.trtllm_mla import (
    _apply_rope_from_table,
    _GlobalTrtllmMLAPlanner,
)
from tensorrt_llm.functional import RopeEmbeddingUtils

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (8, 0),
    reason="TRT-LLM MLA tests require GPU with compute capability >= 8.0",
)

_MLA_TOKENS_PER_BLOCK = 32

# Maximum parameter values across all test parametrizations.  Used to
# warm up the C++ AttentionOp so its internal buffer allocations cover
# every combination (the C++ singleton does not always resize properly
# when parameters shrink and then grow between consecutive calls).
_MAX_NUM_HEADS = 4
_MAX_BATCH_SIZE = 4
_MAX_SEQ_LEN = 256


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
    seed: int = 42,
):
    """Create MLA input tensors with Xavier-like initialization."""
    torch.manual_seed(seed)
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
    max_seq_len_override: int = 0,
):
    """Create paged cache and TRT-LLM-style metadata for MLA.

    Returns a dict with kv_cache, batch_info_host, seq_len, seq_len_host,
    input_pos_host, seq_len_with_cache, and max_seq_info_host tensors.
    """
    latent_dim = kv_lora_rank + qk_rope_head_dim
    num_kv_heads = 1

    # HND paged cache: [num_pages, kv_factor, num_kv_heads, page_size, latent_dim]
    # Production MLA uses kv_factor=1 (K and V are the same latent data).
    kv_cache = torch.zeros(
        max_num_pages,
        1,
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

    block_offset_multiplier = 1
    max_blocks_per_seq = max(pages_per_seq) if pages_per_seq else 1
    max_context_length = max(kv_lengths) if kv_lengths else 1

    # Use the override to keep max_context_length stable across test
    # parametrizations — the C++ AttentionOp caches internal buffers on its
    # first invocation and may not properly resize for subsequent calls with
    # a larger max_context_length.
    if max_seq_len_override > 0:
        max_context_length = max(max_context_length, max_seq_len_override)
        max_blocks_per_seq = max(max_blocks_per_seq, (max_context_length - 1) // page_size + 1)

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

    from tensorrt_llm._torch.auto_deploy.custom_ops.mla.trtllm_mla import (
        prepare_trtllm_mla_metadata_host,
    )

    # Host-side prepare (fills pinned host tensors for thop.attention)
    prepare_trtllm_mla_metadata_host(
        meta["batch_info_host"],
        meta["max_seq_info_host"],
        meta["seq_len_with_cache_host"],
        meta["input_pos_host"],
        meta["seq_len_host"],
    )

    # Device-side prepare (computes block_offsets and block_ids_per_seq via Triton)
    cu_num_pages_device = meta["cu_num_pages_host"].to(meta["cache_loc"].device)
    block_offsets_list = torch.ops.auto_deploy.trtllm_mla_prepare_metadata(
        meta["batch_info_host"],
        meta["max_seq_info_host"],
        cu_num_pages_device,
        meta["cache_loc"],
    )
    kv_cache_block_offsets = block_offsets_list[0]

    return torch.ops.auto_deploy.trtllm_mla_with_cache(
        inputs["q_nope"],
        inputs["q_pe"],
        inputs["compressed_kv"],
        inputs["kpe"],
        inputs["kv_b_proj_weight"],
        meta["batch_info_host"],
        meta["seq_len"],
        meta["seq_len_with_cache"],
        meta["max_seq_info_host"],
        kv_cache_block_offsets,
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
@pytest.mark.parametrize(
    "num_heads,batch_size",
    [(1, 1), (4, 4)],
    ids=["h1b1", "h4b4"],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ["cuda"])
def test_trtllm_mla_prefill(seq_length, num_heads, batch_size, dtype, device):
    """Test TRT-LLM MLA prefill against torch_mla source op.

    The C++ kernel's invokeMLARopeContext applies RoPE to Q and K during
    prefill, so the reference must also use RoPE'd inputs for a fair comparison.

    ``num_heads`` and ``batch_size`` are co-parametrized to avoid interleaving
    configurations that trigger a C++ AttentionOp buffer-caching bug (the
    singleton does not reliably resize when head count changes between calls).
    """
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    kv_lora_rank = 512
    v_head_dim = 128
    page_size = _MLA_TOKENS_PER_BLOCK
    max_seq_len = _MAX_SEQ_LEN
    max_num_pages = batch_size * (max_seq_len // page_size + 2)

    torch.cuda.empty_cache()

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

    total_tokens = batch_size * seq_length

    # --- Run thop FIRST on clean GPU state ---
    trtllm_inputs = {
        "q_nope": inputs["q_nope"].clone().reshape(1, total_tokens, num_heads, qk_nope_head_dim),
        "q_pe": inputs["q_pe"].clone().reshape(1, total_tokens, num_heads, qk_rope_head_dim),
        "compressed_kv": inputs["compressed_kv"].clone().reshape(1, total_tokens, kv_lora_rank),
        "kpe": inputs["kpe"].clone().reshape(1, total_tokens, 1, qk_rope_head_dim),
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
        max_seq_len_override=max_seq_len,
    )

    trtllm_output = _run_trtllm_mla(trtllm_inputs, meta, kv_lora_rank)
    trtllm_output = trtllm_output.view(batch_size, seq_length, num_heads, v_head_dim)
    torch.cuda.synchronize()

    # --- Build RoPE'd reference AFTER the thop call ---
    _, cos_sin_np = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
        max_seq_len, qk_rope_head_dim, 10000.0
    )
    rotary_cos_sin = torch.tensor(cos_sin_np, dtype=torch.float32, device=device)

    positions = torch.zeros(total_tokens, dtype=torch.int32, device=device)
    offset = 0
    for i in range(batch_size):
        positions[offset : offset + seq_length] = torch.arange(
            seq_length, device=device, dtype=torch.int32
        )
        offset += seq_length

    q_pe_flat = inputs["q_pe"].reshape(total_tokens, num_heads, qk_rope_head_dim)
    kpe_flat = inputs["kpe"].reshape(total_tokens, qk_rope_head_dim)
    q_pe_roped, kpe_roped = _apply_rope_from_table(
        q_pe_flat, kpe_flat, rotary_cos_sin, positions, qk_rope_head_dim
    )

    ref_inputs = {
        **inputs,
        "q_pe": q_pe_roped.view(batch_size, seq_length, num_heads, qk_rope_head_dim),
        "kpe": kpe_roped.view(batch_size, seq_length, 1, qk_rope_head_dim),
    }
    ref_output = _run_torch_mla_reference(ref_inputs)

    # Tolerance is relaxed because the C++ kernel applies RoPE in mixed
    # precision and uses a different computation order than the Python
    # reference.
    torch.testing.assert_close(
        trtllm_output,
        ref_output,
        atol=4e-1,
        rtol=4e-1,
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
    _run_multi_step(
        num_heads,
        batch_size,
        qk_nope_head_dim,
        qk_rope_head_dim,
        kv_lora_rank,
        v_head_dim,
        prefill_len,
        num_decode_steps,
        dtype,
        device,
    )


@pytest.mark.parametrize("batch_size", [1, 64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ["cuda"])
def test_trtllm_mla_multi_step_glm4(batch_size, dtype, device):
    """Test multi-step with GLM-4 Flash exact dimensions and large batch."""
    _run_multi_step(
        num_heads=20,
        batch_size=batch_size,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        kv_lora_rank=512,
        v_head_dim=256,
        prefill_len=32,
        num_decode_steps=8,
        dtype=dtype,
        device=device,
    )


def _build_metadata_with_pages(
    batch_size,
    kv_cache,
    page_size,
    page_assignments,
    seq_lengths,
    input_positions,
    kv_lora_rank,
    qk_rope_head_dim,
    dtype,
    device,
):
    """Build TRT-LLM metadata using explicit page assignments.

    Unlike _create_trtllm_paged_metadata (which assigns pages sequentially from
    scratch), this function takes pre-computed page assignments so that page
    mappings remain consistent across prefill and decode steps.
    """
    kv_lengths = [pos + slen for pos, slen in zip(input_positions, seq_lengths)]
    total_tokens = sum(seq_lengths)
    is_decode = all(s == 1 for s in seq_lengths)

    if is_decode:
        batch_info_host = torch.tensor([0, 0, batch_size], dtype=torch.int32, device="cpu")
    else:
        batch_info_host = torch.tensor(
            [batch_size, total_tokens, 0], dtype=torch.int32, device="cpu"
        )

    cache_loc_list, page_seq_list, page_in_seq_list = [], [], []
    pages_per_seq = []
    for seq_idx, pages in enumerate(page_assignments):
        pages_per_seq.append(len(pages))
        for pg_idx, pg in enumerate(pages):
            cache_loc_list.append(pg)
            page_seq_list.append(seq_idx)
            page_in_seq_list.append(pg_idx)

    cu_num_pages = torch.zeros(batch_size + 1, dtype=torch.int32, device="cpu")
    for i, n in enumerate(pages_per_seq):
        cu_num_pages[i + 1] = cu_num_pages[i] + n

    max_blocks_per_seq = max(pages_per_seq) if pages_per_seq else 1
    max_context_length = max(kv_lengths) if kv_lengths else 1

    return {
        "kv_cache": kv_cache,
        "batch_info_host": batch_info_host,
        "seq_len": torch.tensor(seq_lengths, dtype=torch.int32, device=device),
        "seq_len_host": torch.tensor(seq_lengths, dtype=torch.int32, device="cpu"),
        "input_pos_host": torch.tensor(input_positions, dtype=torch.int32, device="cpu"),
        "seq_len_with_cache": torch.tensor(kv_lengths, dtype=torch.int32, device=device),
        "seq_len_with_cache_host": torch.tensor(kv_lengths, dtype=torch.int32, device="cpu"),
        "max_seq_info_host": torch.tensor(
            [max_context_length, max_blocks_per_seq, 1, batch_size],
            dtype=torch.int32,
            device="cpu",
        ),
        "cu_num_pages_host": cu_num_pages,
        "cache_loc": torch.tensor(cache_loc_list, dtype=torch.int32, device=device),
        "page_seq_indices": torch.tensor(page_seq_list, dtype=torch.int32, device=device),
        "page_in_seq": torch.tensor(page_in_seq_list, dtype=torch.int32, device=device),
    }


def _run_multi_step(
    num_heads,
    batch_size,
    qk_nope_head_dim,
    qk_rope_head_dim,
    kv_lora_rank,
    v_head_dim,
    prefill_len,
    num_decode_steps,
    dtype,
    device,
):
    page_size = _MLA_TOKENS_PER_BLOCK
    max_seq_len = prefill_len + num_decode_steps + 8
    max_num_pages = batch_size * (max_seq_len // page_size + 2)

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

    latent_dim = kv_lora_rank + qk_rope_head_dim
    kv_cache = torch.zeros(
        max_num_pages,
        1,
        1,
        page_size,
        latent_dim,
        dtype=dtype,
        device=device,
    )

    # Persistent page assignments: allocate pages once and grow as needed
    prefill_pages_per_seq = max(1, (prefill_len - 1) // page_size + 1)
    page_assignments = [
        list(range(i * prefill_pages_per_seq, (i + 1) * prefill_pages_per_seq))
        for i in range(batch_size)
    ]
    next_free_page = batch_size * prefill_pages_per_seq

    # --- Step 1: Prefill ---
    total_prefill_tokens = batch_size * prefill_len
    trtllm_prefill_inputs = {
        "q_nope": all_inputs["q_nope"][:, :prefill_len].reshape(
            1,
            total_prefill_tokens,
            num_heads,
            qk_nope_head_dim,
        ),
        "q_pe": all_inputs["q_pe"][:, :prefill_len].reshape(
            1,
            total_prefill_tokens,
            num_heads,
            qk_rope_head_dim,
        ),
        "compressed_kv": all_inputs["compressed_kv"][:, :prefill_len].reshape(
            1,
            total_prefill_tokens,
            kv_lora_rank,
        ),
        "kpe": all_inputs["kpe"][:, :prefill_len].reshape(
            1,
            total_prefill_tokens,
            1,
            qk_rope_head_dim,
        ),
        "kv_b_proj_weight": all_inputs["kv_b_proj_weight"],
    }

    prefill_meta = _build_metadata_with_pages(
        batch_size,
        kv_cache,
        page_size,
        page_assignments,
        [prefill_len] * batch_size,
        [0] * batch_size,
        kv_lora_rank,
        qk_rope_head_dim,
        dtype,
        device,
    )
    _run_trtllm_mla(trtllm_prefill_inputs, prefill_meta, kv_lora_rank)

    # --- Step 2: Decode steps ---
    for step in range(num_decode_steps):
        token_idx = prefill_len + step
        kv_len = token_idx + 1
        pages_needed = max(1, (kv_len - 1) // page_size + 1)

        for i in range(batch_size):
            while len(page_assignments[i]) < pages_needed:
                page_assignments[i].append(next_free_page)
                next_free_page += 1

        decode_inputs = {
            "q_nope": all_inputs["q_nope"][:, token_idx : token_idx + 1],
            "q_pe": all_inputs["q_pe"][:, token_idx : token_idx + 1],
            "compressed_kv": all_inputs["compressed_kv"][:, token_idx : token_idx + 1],
            "kpe": all_inputs["kpe"][:, token_idx : token_idx + 1],
            "kv_b_proj_weight": all_inputs["kv_b_proj_weight"],
        }

        decode_meta = _build_metadata_with_pages(
            batch_size,
            kv_cache,
            page_size,
            page_assignments,
            [1] * batch_size,
            [token_idx] * batch_size,
            kv_lora_rank,
            qk_rope_head_dim,
            dtype,
            device,
        )

        trtllm_decode_out = _run_trtllm_mla(decode_inputs, decode_meta, kv_lora_rank)

        ref_output = torch.ops.auto_deploy.torch_mla(
            all_inputs["q_nope"][:, token_idx : token_idx + 1],
            all_inputs["q_pe"][:, token_idx : token_idx + 1],
            all_inputs["compressed_kv"][:, : token_idx + 1],
            all_inputs["kpe"][:, : token_idx + 1],
            all_inputs["kv_b_proj_weight"],
            False,
            None,
            "bsnd",
        )

        torch.testing.assert_close(
            trtllm_decode_out.view(batch_size, 1, num_heads, v_head_dim),
            ref_output,
            atol=2e-2,
            rtol=2e-2,
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


@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ["cuda"])
def test_trtllm_mla_mixed_batch(num_heads, dtype, device):
    """Test a single forward call containing both prefill and decode sequences.

    Simulates the common executor pattern where new prompts are batched
    alongside ongoing generation. The metadata ordering is:
        [prefill_seq_0, ..., prefill_seq_n, decode_seq_0, ..., decode_seq_m]

    This specifically exercises the cu_kv_seqlens slicing logic in _handle_decode:
    the decode sequences are at the END of the metadata arrays, so cu_kv must
    use sequence_length[num_prefill:] rather than sequence_length[:num_decode].
    """
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    kv_lora_rank = 512
    v_head_dim = 128
    prefill_len = 16
    decode_past_len = 16
    num_prefill_seqs = 2
    num_decode_seqs = 2
    page_size = _MLA_TOKENS_PER_BLOCK
    total_seqs = num_prefill_seqs + num_decode_seqs
    max_seq_len = decode_past_len + 8
    max_num_pages = total_seqs * (max_seq_len // page_size + 2)

    # Generate all inputs: prefill sequences + decode sequences (with full history)
    prefill_inputs = _create_mla_inputs(
        num_prefill_seqs,
        prefill_len,
        num_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        kv_lora_rank,
        v_head_dim,
        dtype,
        device,
    )
    decode_all_inputs = _create_mla_inputs(
        num_decode_seqs,
        decode_past_len + 1,
        num_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        kv_lora_rank,
        v_head_dim,
        dtype,
        device,
    )

    # --- Step 1: Prefill the decode sequences to populate the cache ---
    decode_prefill_inputs = {
        "q_nope": decode_all_inputs["q_nope"][:, :decode_past_len],
        "q_pe": decode_all_inputs["q_pe"][:, :decode_past_len],
        "compressed_kv": decode_all_inputs["compressed_kv"][:, :decode_past_len],
        "kpe": decode_all_inputs["kpe"][:, :decode_past_len],
        "kv_b_proj_weight": decode_all_inputs["kv_b_proj_weight"],
    }
    total_decode_prefill_tokens = num_decode_seqs * decode_past_len
    trtllm_decode_prefill = {
        "q_nope": decode_prefill_inputs["q_nope"].reshape(
            1, total_decode_prefill_tokens, num_heads, qk_nope_head_dim
        ),
        "q_pe": decode_prefill_inputs["q_pe"].reshape(
            1, total_decode_prefill_tokens, num_heads, qk_rope_head_dim
        ),
        "compressed_kv": decode_prefill_inputs["compressed_kv"].reshape(
            1, total_decode_prefill_tokens, kv_lora_rank
        ),
        "kpe": decode_prefill_inputs["kpe"].reshape(
            1, total_decode_prefill_tokens, 1, qk_rope_head_dim
        ),
        "kv_b_proj_weight": decode_all_inputs["kv_b_proj_weight"],
    }
    decode_prefill_meta = _create_trtllm_paged_metadata(
        num_decode_seqs,
        max_num_pages,
        page_size,
        kv_lora_rank,
        qk_rope_head_dim,
        dtype,
        device,
        [decode_past_len] * num_decode_seqs,
        [0] * num_decode_seqs,
    )
    _run_trtllm_mla(trtllm_decode_prefill, decode_prefill_meta, kv_lora_rank)

    # --- Step 2: Mixed batch — new prefills + decode tokens in ONE call ---
    # Metadata ordering: [prefill_0, prefill_1, decode_0, decode_1]
    num_prefill_tokens = num_prefill_seqs * prefill_len

    # Build concatenated Q/KV tensors: [prefill_tokens..., decode_tokens...]
    decode_token_inputs = {
        "q_nope": decode_all_inputs["q_nope"][:, decode_past_len : decode_past_len + 1],
        "q_pe": decode_all_inputs["q_pe"][:, decode_past_len : decode_past_len + 1],
        "compressed_kv": decode_all_inputs["compressed_kv"][
            :, decode_past_len : decode_past_len + 1
        ],
        "kpe": decode_all_inputs["kpe"][:, decode_past_len : decode_past_len + 1],
    }

    mixed_q_nope = torch.cat(
        [
            prefill_inputs["q_nope"].reshape(num_prefill_tokens, num_heads, qk_nope_head_dim),
            decode_token_inputs["q_nope"].reshape(num_decode_seqs, num_heads, qk_nope_head_dim),
        ],
        dim=0,
    ).unsqueeze(0)  # [1, num_tokens, N, D]
    mixed_q_pe = torch.cat(
        [
            prefill_inputs["q_pe"].reshape(num_prefill_tokens, num_heads, qk_rope_head_dim),
            decode_token_inputs["q_pe"].reshape(num_decode_seqs, num_heads, qk_rope_head_dim),
        ],
        dim=0,
    ).unsqueeze(0)
    mixed_ckv = torch.cat(
        [
            prefill_inputs["compressed_kv"].reshape(num_prefill_tokens, kv_lora_rank),
            decode_token_inputs["compressed_kv"].reshape(num_decode_seqs, kv_lora_rank),
        ],
        dim=0,
    ).unsqueeze(0)
    mixed_kpe = torch.cat(
        [
            prefill_inputs["kpe"].reshape(num_prefill_tokens, 1, qk_rope_head_dim),
            decode_token_inputs["kpe"].reshape(num_decode_seqs, 1, qk_rope_head_dim),
        ],
        dim=0,
    ).unsqueeze(0)

    mixed_inputs = {
        "q_nope": mixed_q_nope,
        "q_pe": mixed_q_pe,
        "compressed_kv": mixed_ckv,
        "kpe": mixed_kpe,
        "kv_b_proj_weight": decode_all_inputs["kv_b_proj_weight"],
    }

    # Build mixed metadata manually
    seq_lengths = [prefill_len] * num_prefill_seqs + [1] * num_decode_seqs
    input_positions = [0] * num_prefill_seqs + [decode_past_len] * num_decode_seqs
    kv_lengths = [prefill_len] * num_prefill_seqs + [decode_past_len + 1] * num_decode_seqs

    batch_info_host = torch.tensor(
        [num_prefill_seqs, num_prefill_tokens, num_decode_seqs],
        dtype=torch.int32,
        device="cpu",
    )

    pages_per_seq = [(kv - 1) // page_size + 1 for kv in kv_lengths]
    # Prefill seqs get fresh pages; decode seqs reuse pages from step 1
    decode_page_assignments = []
    for seq_idx in range(num_decode_seqs):
        np_ = pages_per_seq[num_prefill_seqs + seq_idx]
        start = decode_prefill_meta["cache_loc"][
            int(decode_prefill_meta["cu_num_pages_host"][seq_idx])
        ].item()
        decode_page_assignments.append(list(range(start, start + np_)))

    next_page = max(p for pages in decode_page_assignments for p in pages) + 1
    prefill_page_assignments = []
    for seq_idx in range(num_prefill_seqs):
        np_ = pages_per_seq[seq_idx]
        prefill_page_assignments.append(list(range(next_page, next_page + np_)))
        next_page += np_

    all_page_assignments = prefill_page_assignments + decode_page_assignments

    page_seq_indices_list = []
    page_in_seq_list = []
    cache_loc_list = []
    for seq_idx, pages in enumerate(all_page_assignments):
        for pg_idx, pg in enumerate(pages):
            page_seq_indices_list.append(seq_idx)
            page_in_seq_list.append(pg_idx)
            cache_loc_list.append(pg)

    cu_num_pages = torch.zeros(total_seqs + 1, dtype=torch.int32, device="cpu")
    for i, pages in enumerate(all_page_assignments):
        cu_num_pages[i + 1] = cu_num_pages[i] + len(pages)

    max_blocks_per_seq = max(len(p) for p in all_page_assignments)
    max_context_length = max(kv_lengths)

    mixed_meta = {
        "kv_cache": decode_prefill_meta["kv_cache"],
        "batch_info_host": batch_info_host,
        "seq_len": torch.tensor(seq_lengths, dtype=torch.int32, device=device),
        "seq_len_host": torch.tensor(seq_lengths, dtype=torch.int32, device="cpu"),
        "input_pos_host": torch.tensor(input_positions, dtype=torch.int32, device="cpu"),
        "seq_len_with_cache": torch.tensor(kv_lengths, dtype=torch.int32, device=device),
        "seq_len_with_cache_host": torch.tensor(kv_lengths, dtype=torch.int32, device="cpu"),
        "max_seq_info_host": torch.tensor(
            [max_context_length, max_blocks_per_seq, 1, total_seqs],
            dtype=torch.int32,
            device="cpu",
        ),
        "cu_num_pages_host": cu_num_pages,
        "cache_loc": torch.tensor(cache_loc_list, dtype=torch.int32, device=device),
        "page_seq_indices": torch.tensor(page_seq_indices_list, dtype=torch.int32, device=device),
        "page_in_seq": torch.tensor(page_in_seq_list, dtype=torch.int32, device=device),
    }

    mixed_out = _run_trtllm_mla(mixed_inputs, mixed_meta, kv_lora_rank)

    # --- Verify decode outputs ---
    # Extract the decode portion of the output
    actual_decode = mixed_out[0, num_prefill_tokens:].view(
        num_decode_seqs, 1, num_heads, v_head_dim
    )

    # Reference: torch_mla on the full KV for each decode sequence
    ref_decode = torch.ops.auto_deploy.torch_mla(
        decode_all_inputs["q_nope"][:, decode_past_len : decode_past_len + 1],
        decode_all_inputs["q_pe"][:, decode_past_len : decode_past_len + 1],
        decode_all_inputs["compressed_kv"][:, : decode_past_len + 1],
        decode_all_inputs["kpe"][:, : decode_past_len + 1],
        decode_all_inputs["kv_b_proj_weight"],
        False,  # is_causal=False since Q_len=1 < KV_len
        None,
        "bsnd",
    )

    torch.testing.assert_close(
        actual_decode,
        ref_decode,
        atol=2e-2,
        rtol=2e-2,
    )


# =============================================================================
# Model-level inference session test
# =============================================================================


class _SimpleMLA(torch.nn.Module):
    """Minimal MLA module with Q/KV projections, layernorm, and output projection.

    Mirrors DeepSeekV3Attention projections but without RoPE, so the test
    isolates cache correctness from rotary embedding concerns.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        kv_lora_rank,
        v_head_dim,
        dtype,
        device,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        kv_head_dim = qk_nope_head_dim + v_head_dim

        self.q_proj = torch.nn.Linear(
            hidden_size,
            num_heads * q_head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.kv_a_proj = torch.nn.Linear(
            hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.kv_a_layernorm = torch.nn.RMSNorm(kv_lora_rank, dtype=dtype, device=device)
        self.kv_b_proj = torch.nn.Linear(
            kv_lora_rank,
            num_heads * kv_head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.o_proj = torch.nn.Linear(
            num_heads * v_head_dim,
            hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
        )

    @torch.no_grad()
    def project(self, hidden_states):
        """Project hidden_states to MLA inputs: (q_nope, q_pe, compressed_kv, kpe)."""
        B, S, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(
            B,
            S,
            self.num_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
        )
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv_a = self.kv_a_proj(hidden_states)
        compressed_kv, kpe = kv_a.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        kpe = kpe.view(B, S, 1, self.qk_rope_head_dim)

        return q_nope, q_pe, compressed_kv, kpe

    @torch.no_grad()
    def output_proj(self, attn_out):
        """Apply o_proj: [..., N, v_head_dim] → [..., hidden_size]."""
        *leading, _N, _D = attn_out.shape
        return self.o_proj(attn_out.reshape(*leading, self.num_heads * self.v_head_dim))


def _build_step_metadata(seq_specs, kv_cache, page_size, device):
    """Build trtllm_mla metadata for one inference step.

    Args:
        seq_specs: list of (chunk_len, input_pos, pages) tuples.
            MUST be ordered: prefill seqs first (chunk_len > 1 or input_pos == 0),
            then decode seqs (chunk_len == 1 and input_pos > 0).
        kv_cache: shared paged KV cache tensor.
        page_size: tokens per page.
        device: GPU device.
    """
    batch_size = len(seq_specs)
    prefill_count = sum(1 for cl, ip, _ in seq_specs if cl > 1 or ip == 0)
    prefill_tokens = sum(cl for cl, ip, _ in seq_specs if cl > 1 or ip == 0)
    decode_count = batch_size - prefill_count

    seq_lens = [cl for cl, _, _ in seq_specs]
    input_positions = [ip for _, ip, _ in seq_specs]
    kv_lengths = [ip + cl for cl, ip, _ in seq_specs]
    all_pages = [pg for _, _, pg in seq_specs]
    pages_per_seq = [len(pg) for pg in all_pages]

    cache_loc_list, page_seq_list, page_in_seq_list = [], [], []
    for seq_idx, pages in enumerate(all_pages):
        for pg_idx, pg in enumerate(pages):
            cache_loc_list.append(pg)
            page_seq_list.append(seq_idx)
            page_in_seq_list.append(pg_idx)

    cu_num_pages = torch.zeros(batch_size + 1, dtype=torch.int32, device="cpu")
    for i, n in enumerate(pages_per_seq):
        cu_num_pages[i + 1] = cu_num_pages[i] + n

    return {
        "kv_cache": kv_cache,
        "batch_info_host": torch.tensor(
            [prefill_count, prefill_tokens, decode_count],
            dtype=torch.int32,
            device="cpu",
        ),
        "seq_len": torch.tensor(seq_lens, dtype=torch.int32, device=device),
        "seq_len_host": torch.tensor(seq_lens, dtype=torch.int32, device="cpu"),
        "input_pos_host": torch.tensor(input_positions, dtype=torch.int32, device="cpu"),
        "seq_len_with_cache": torch.tensor(kv_lengths, dtype=torch.int32, device=device),
        "seq_len_with_cache_host": torch.tensor(kv_lengths, dtype=torch.int32, device="cpu"),
        "max_seq_info_host": torch.tensor(
            [max(kv_lengths), max(pages_per_seq), 1, batch_size],
            dtype=torch.int32,
            device="cpu",
        ),
        "cu_num_pages_host": cu_num_pages,
        "cache_loc": torch.tensor(cache_loc_list, dtype=torch.int32, device=device),
        "page_seq_indices": torch.tensor(page_seq_list, dtype=torch.int32, device=device),
        "page_in_seq": torch.tensor(page_in_seq_list, dtype=torch.int32, device=device),
    }


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ["cuda"])
def test_trtllm_mla_inference_session(dtype, device):
    """End-to-end inference session with real projection layers.

    Simulates a serving scenario where two sequences arrive at different times:

      Step 1: Prefill sequence A (32 tokens)
      Step 2: Decode sequence A (1 token at pos 32)
      Step 3: Mixed batch — prefill B (16 tokens) + decode A (pos 33)
      Step 4: Pure decode — A (pos 34) + B (pos 16)

    At each step, the trtllm_mla_with_cache output (through o_proj) is compared
    against the torch_mla reference that recomputes from full token history.
    This validates the full projection → cache → attention → output pipeline,
    including the cu_kv_seqlens mixed-batch fix.
    """
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    hidden_size = 256
    num_heads = 4
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    kv_lora_rank = 512
    v_head_dim = 128
    page_size = _MLA_TOKENS_PER_BLOCK
    latent_dim = kv_lora_rank + qk_rope_head_dim

    seq_a_prompt = 32
    seq_b_prompt = 16

    model = _SimpleMLA(
        hidden_size,
        num_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        kv_lora_rank,
        v_head_dim,
        dtype,
        device,
    )

    hidden_a = torch.randn(1, seq_a_prompt + 3, hidden_size, dtype=dtype, device=device) * 0.02
    hidden_b = torch.randn(1, seq_b_prompt + 2, hidden_size, dtype=dtype, device=device) * 0.02

    with torch.no_grad():
        qa_nope, qa_pe, qa_ckv, qa_kpe = model.project(hidden_a)
        qb_nope, qb_pe, qb_ckv, qb_kpe = model.project(hidden_b)
    kv_b_w = model.kv_b_proj.weight

    kv_cache = torch.zeros(10, 1, 1, page_size, latent_dim, dtype=dtype, device=device)
    a_pages = [0, 1]  # A grows from 32 to 35 tokens (2 pages)
    b_pages = [2]  # B grows from 16 to 18 tokens (1 page)

    # thop.attention prefill has larger BF16 numerical diffs than SDPA (~0.15 raw),
    # amplified by the o_proj linear layer. Still tight enough to catch real bugs
    # (wrong scaling = 10x+ error, wrong cu_kv_seqlens = completely wrong output).
    PREFILL_TOL = 1.0
    # Decode tolerance is higher than op-level tests (2e-2) because the o_proj
    # linear layer amplifies the numerical differences from weight absorption.
    DECODE_TOL = 8e-2

    # ---- Step 1: Prefill A (32 tokens) ----
    meta = _build_step_metadata([(seq_a_prompt, 0, a_pages[:1])], kv_cache, page_size, device)
    trtllm_out = _run_trtllm_mla(
        {
            "q_nope": qa_nope[:, :seq_a_prompt],
            "q_pe": qa_pe[:, :seq_a_prompt],
            "compressed_kv": qa_ckv[:, :seq_a_prompt],
            "kpe": qa_kpe[:, :seq_a_prompt],
            "kv_b_proj_weight": kv_b_w,
        },
        meta,
        kv_lora_rank,
    )
    ref_out = torch.ops.auto_deploy.torch_mla(
        qa_nope[:, :seq_a_prompt],
        qa_pe[:, :seq_a_prompt],
        qa_ckv[:, :seq_a_prompt],
        qa_kpe[:, :seq_a_prompt],
        kv_b_w,
        True,
        None,
        "bsnd",
    )
    torch.testing.assert_close(
        model.output_proj(trtllm_out),
        model.output_proj(ref_out),
        atol=PREFILL_TOL,
        rtol=PREFILL_TOL,
    )

    # ---- Step 2: Decode A (pos=32) ----
    pos_a = seq_a_prompt
    meta = _build_step_metadata([(1, pos_a, a_pages[:2])], kv_cache, page_size, device)
    trtllm_out = _run_trtllm_mla(
        {
            "q_nope": qa_nope[:, pos_a : pos_a + 1],
            "q_pe": qa_pe[:, pos_a : pos_a + 1],
            "compressed_kv": qa_ckv[:, pos_a : pos_a + 1],
            "kpe": qa_kpe[:, pos_a : pos_a + 1],
            "kv_b_proj_weight": kv_b_w,
        },
        meta,
        kv_lora_rank,
    )
    ref_out = torch.ops.auto_deploy.torch_mla(
        qa_nope[:, pos_a : pos_a + 1],
        qa_pe[:, pos_a : pos_a + 1],
        qa_ckv[:, : pos_a + 1],
        qa_kpe[:, : pos_a + 1],
        kv_b_w,
        False,
        None,
        "bsnd",
    )
    torch.testing.assert_close(
        model.output_proj(trtllm_out),
        model.output_proj(ref_out),
        atol=DECODE_TOL,
        rtol=DECODE_TOL,
    )
    pos_a += 1  # now 33

    # ---- Step 3: Mixed batch — prefill B (16 tokens) + decode A (pos=33) ----
    n_bp = seq_b_prompt
    mixed_q_nope = torch.cat(
        [
            qb_nope[0, :n_bp],
            qa_nope[0, pos_a : pos_a + 1],
        ],
        dim=0,
    ).unsqueeze(0)
    mixed_q_pe = torch.cat(
        [
            qb_pe[0, :n_bp],
            qa_pe[0, pos_a : pos_a + 1],
        ],
        dim=0,
    ).unsqueeze(0)
    mixed_ckv = torch.cat(
        [
            qb_ckv[0, :n_bp],
            qa_ckv[0, pos_a : pos_a + 1],
        ],
        dim=0,
    ).unsqueeze(0)
    mixed_kpe = torch.cat(
        [
            qb_kpe[0, :n_bp],
            qa_kpe[0, pos_a : pos_a + 1],
        ],
        dim=0,
    ).unsqueeze(0)

    meta = _build_step_metadata(
        [
            (n_bp, 0, b_pages[:1]),  # B prefill
            (1, pos_a, a_pages[:2]),  # A decode
        ],
        kv_cache,
        page_size,
        device,
    )
    mixed_out = _run_trtllm_mla(
        {
            "q_nope": mixed_q_nope,
            "q_pe": mixed_q_pe,
            "compressed_kv": mixed_ckv,
            "kpe": mixed_kpe,
            "kv_b_proj_weight": kv_b_w,
        },
        meta,
        kv_lora_rank,
    )

    # Verify B prefill portion
    ref_b = torch.ops.auto_deploy.torch_mla(
        qb_nope[:, :n_bp],
        qb_pe[:, :n_bp],
        qb_ckv[:, :n_bp],
        qb_kpe[:, :n_bp],
        kv_b_w,
        True,
        None,
        "bsnd",
    )
    torch.testing.assert_close(
        model.output_proj(mixed_out[:, :n_bp].view(1, n_bp, num_heads, v_head_dim)),
        model.output_proj(ref_b),
        atol=PREFILL_TOL,
        rtol=PREFILL_TOL,
    )
    # Verify A decode portion
    ref_a = torch.ops.auto_deploy.torch_mla(
        qa_nope[:, pos_a : pos_a + 1],
        qa_pe[:, pos_a : pos_a + 1],
        qa_ckv[:, : pos_a + 1],
        qa_kpe[:, : pos_a + 1],
        kv_b_w,
        False,
        None,
        "bsnd",
    )
    torch.testing.assert_close(
        model.output_proj(mixed_out[:, n_bp:].view(1, 1, num_heads, v_head_dim)),
        model.output_proj(ref_a),
        atol=DECODE_TOL,
        rtol=DECODE_TOL,
    )
    pos_a += 1  # now 34
    pos_b = n_bp  # 16

    # ---- Step 4: Pure decode — A (pos=34) + B (pos=16) ----
    dec_q_nope = torch.cat(
        [
            qa_nope[0, pos_a : pos_a + 1],
            qb_nope[0, pos_b : pos_b + 1],
        ],
        dim=0,
    ).unsqueeze(0)
    dec_q_pe = torch.cat(
        [
            qa_pe[0, pos_a : pos_a + 1],
            qb_pe[0, pos_b : pos_b + 1],
        ],
        dim=0,
    ).unsqueeze(0)
    dec_ckv = torch.cat(
        [
            qa_ckv[0, pos_a : pos_a + 1],
            qb_ckv[0, pos_b : pos_b + 1],
        ],
        dim=0,
    ).unsqueeze(0)
    dec_kpe = torch.cat(
        [
            qa_kpe[0, pos_a : pos_a + 1],
            qb_kpe[0, pos_b : pos_b + 1],
        ],
        dim=0,
    ).unsqueeze(0)

    meta = _build_step_metadata(
        [
            (1, pos_a, a_pages[:2]),  # A decode
            (1, pos_b, b_pages[:1]),  # B decode
        ],
        kv_cache,
        page_size,
        device,
    )
    dec_out = _run_trtllm_mla(
        {
            "q_nope": dec_q_nope,
            "q_pe": dec_q_pe,
            "compressed_kv": dec_ckv,
            "kpe": dec_kpe,
            "kv_b_proj_weight": kv_b_w,
        },
        meta,
        kv_lora_rank,
    )

    # Verify A decode
    ref_a = torch.ops.auto_deploy.torch_mla(
        qa_nope[:, pos_a : pos_a + 1],
        qa_pe[:, pos_a : pos_a + 1],
        qa_ckv[:, : pos_a + 1],
        qa_kpe[:, : pos_a + 1],
        kv_b_w,
        False,
        None,
        "bsnd",
    )
    torch.testing.assert_close(
        model.output_proj(dec_out[:, :1].view(1, 1, num_heads, v_head_dim)),
        model.output_proj(ref_a),
        atol=DECODE_TOL,
        rtol=DECODE_TOL,
    )
    # Verify B decode
    ref_b = torch.ops.auto_deploy.torch_mla(
        qb_nope[:, pos_b : pos_b + 1],
        qb_pe[:, pos_b : pos_b + 1],
        qb_ckv[:, : pos_b + 1],
        qb_kpe[:, : pos_b + 1],
        kv_b_w,
        False,
        None,
        "bsnd",
    )
    torch.testing.assert_close(
        model.output_proj(dec_out[:, 1:2].view(1, 1, num_heads, v_head_dim)),
        model.output_proj(ref_b),
        atol=DECODE_TOL,
        rtol=DECODE_TOL,
    )

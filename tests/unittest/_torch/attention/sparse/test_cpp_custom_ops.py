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
"""Unit tests for DSA C++ custom ops:

- ``torch.ops.trtllm.indexer_k_cache_gather_op``
- ``torch.ops.trtllm.convert_req_index_to_global``
- ``torch.ops.trtllm.fused_cat_fp4``
"""

import pytest
import torch
from utils.util import skip_pre_blackwell

# Import tensorrt_llm to load C++ custom operators
import tensorrt_llm  # noqa: F401

try:
    from tensorrt_llm.deep_gemm.utils.math import per_token_cast_to_fp4

    HAS_DEEPGEMM_REF = True
except ImportError:
    HAS_DEEPGEMM_REF = False

# ---------------------------------------------------------------------------
# Constants matching the C++ kernel (DeepSeek-V3.2 indexer config)
# ---------------------------------------------------------------------------
HEAD_DIM = 128
SCALE_BYTES = 4
BYTES_PER_TOKEN = HEAD_DIM + SCALE_BYTES


# ===================================================================
# Test 1: indexer_k_cache_gather_op
# ===================================================================


def _reference_indexer_k_cache_gather(
    k_cache: torch.Tensor,
    slot_mapping_fp8: torch.Tensor,
    slot_mapping_scale: torch.Tensor,
    k_token_start: int,
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Python reference for indexer_k_cache_gather_op.

    The C++ op treats the 4D k_cache as a flat byte buffer (via strides)
    and gathers HEAD_DIM bytes for FP8 data and SCALE_BYTES bytes for scales
    at positions given by the slot mappings.
    """
    device = k_cache.device

    if num_tokens == 0:
        return (
            torch.empty(0, HEAD_DIM, dtype=torch.float8_e4m3fn, device=device),
            torch.empty(0, 1, dtype=torch.float32, device=device),
        )

    # Flatten the cache to a 1D byte view using a contiguous copy
    k_cache_flat = k_cache.contiguous().reshape(-1)

    fp8_bases = slot_mapping_fp8[k_token_start : k_token_start + num_tokens]
    byte_offsets_fp8 = torch.arange(HEAD_DIM, device=device, dtype=torch.int64)
    gather_fp8 = fp8_bases.unsqueeze(1) + byte_offsets_fp8.unsqueeze(0)
    out_fp8 = k_cache_flat[gather_fp8]

    scale_bases = slot_mapping_scale[k_token_start : k_token_start + num_tokens]
    byte_offsets_scale = torch.arange(SCALE_BYTES, device=device, dtype=torch.int64)
    gather_scale = scale_bases.unsqueeze(1) + byte_offsets_scale.unsqueeze(0)
    out_scale = k_cache_flat[gather_scale]

    k_fp8 = out_fp8.view(torch.float8_e4m3fn)
    k_scale = out_scale.view(torch.float32).view(num_tokens, 1)
    return k_fp8, k_scale


def _create_4d_cache_and_mappings(
    total_kv_len: int,
    num_blocks: int,
    block_size: int,
    per_token_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a random 4D k_cache with valid, non-overlapping slot mappings.

    k_cache shape: [num_blocks, block_size, 1, per_token_size]
    Each token occupies ``per_token_size`` bytes at a unique (block, slot)
    position. Slot mappings point into the *contiguous flat* byte layout.
    """
    k_cache = torch.randint(
        0,
        256,
        (num_blocks, block_size, 1, per_token_size),
        dtype=torch.uint8,
        device=device,
    )

    total_slots = num_blocks * block_size
    assert total_kv_len <= total_slots, (
        f"total_kv_len ({total_kv_len}) exceeds total slots ({total_slots})"
    )

    # Pick random unique (block, slot) positions
    perm = torch.randperm(total_slots, device=device)[:total_kv_len]
    # Flat byte offset of each token in the contiguous view
    # Each token starts at perm[i] * per_token_size (within the [total_slots, 1, per_token_size] view)
    # But the cache is [num_blocks, block_size, 1, per_token_size], so flat offset is:
    # perm[i] * (1 * per_token_size) since dim2=1
    flat_starts = perm.to(torch.int64) * per_token_size

    slot_mapping_fp8 = flat_starts
    slot_mapping_scale = flat_starts + HEAD_DIM

    return k_cache, slot_mapping_fp8, slot_mapping_scale


@pytest.mark.parametrize(
    "total_kv_len,num_blocks,block_size,k_token_start,num_tokens",
    [
        # Gather all tokens
        (64, 8, 8, 0, 64),
        # Sub-range from the middle
        (128, 16, 8, 32, 64),
        # Single token
        (10, 2, 8, 3, 1),
        # Larger test
        (512, 64, 8, 100, 300),
        # Non-power-of-2 token count
        (100, 16, 8, 10, 37),
        # Very small
        (3, 1, 4, 0, 3),
        # Larger block_size
        (64, 4, 16, 0, 64),
    ],
)
def test_indexer_k_cache_gather_contiguous(
    total_kv_len,
    num_blocks,
    block_size,
    k_token_start,
    num_tokens,
):
    """Test C++ gather op on a contiguous 4D cache against Python reference."""
    device = torch.device("cuda")
    per_token_size = BYTES_PER_TOKEN  # 132

    k_cache, slot_fp8, slot_scale = _create_4d_cache_and_mappings(
        total_kv_len, num_blocks, block_size, per_token_size, device
    )

    # C++ op
    cpp_fp8, cpp_scale = torch.ops.trtllm.indexer_k_cache_gather_op(
        k_cache, slot_fp8, slot_scale, k_token_start, num_tokens, HEAD_DIM
    )

    # Reference
    ref_fp8, ref_scale = _reference_indexer_k_cache_gather(
        k_cache, slot_fp8, slot_scale, k_token_start, num_tokens
    )

    assert cpp_fp8.shape == ref_fp8.shape
    assert cpp_scale.shape == ref_scale.shape
    assert torch.equal(cpp_fp8.view(torch.uint8), ref_fp8.view(torch.uint8)), "FP8 data mismatch"
    assert torch.equal(cpp_scale.view(torch.uint8), ref_scale.view(torch.uint8)), (
        "Scale data mismatch"
    )


@pytest.mark.parametrize(
    "total_kv_len,num_blocks,block_size,k_token_start,num_tokens",
    [
        (64, 8, 8, 0, 64),
        (128, 16, 8, 32, 64),
        (512, 64, 8, 100, 300),
    ],
)
def test_indexer_k_cache_gather_noncontiguous(
    total_kv_len,
    num_blocks,
    block_size,
    k_token_start,
    num_tokens,
):
    """Test C++ gather op on a non-contiguous 4D cache.

    The real KV cache pool is typically a large buffer viewed with
    non-contiguous strides (layer interleaving). Simulate this by
    allocating a larger buffer and taking a strided view.
    """
    device = torch.device("cuda")
    per_token_size = BYTES_PER_TOKEN
    num_layers = 3  # simulate multi-layer interleaving
    target_layer = 1

    # Allocate the full interleaved pool:
    # [num_blocks, block_size, num_layers, per_token_size]
    full_pool = torch.randint(
        0,
        256,
        (num_blocks, block_size, num_layers, per_token_size),
        dtype=torch.uint8,
        device=device,
    )

    # Select a single layer → shape [num_blocks, block_size, 1, per_token_size]
    # This slice is non-contiguous because stride(1) spans across all layers.
    k_cache_nc = full_pool[:, :, target_layer : target_layer + 1, :]
    assert not k_cache_nc.is_contiguous(), "Expected non-contiguous cache"

    # Build slot mappings based on the *contiguous* copy (what the reference sees)
    k_cache_contig = k_cache_nc.contiguous()
    total_slots = num_blocks * block_size
    perm = torch.randperm(total_slots, device=device)[:total_kv_len]
    flat_starts = perm.to(torch.int64) * per_token_size
    slot_fp8 = flat_starts
    slot_scale = flat_starts + HEAD_DIM

    # C++ op (handles non-contiguous strides internally)
    cpp_fp8, cpp_scale = torch.ops.trtllm.indexer_k_cache_gather_op(
        k_cache_nc, slot_fp8, slot_scale, k_token_start, num_tokens, HEAD_DIM
    )

    # Reference uses contiguous copy
    ref_fp8, ref_scale = _reference_indexer_k_cache_gather(
        k_cache_contig, slot_fp8, slot_scale, k_token_start, num_tokens
    )

    assert cpp_fp8.shape == ref_fp8.shape
    assert cpp_scale.shape == ref_scale.shape
    assert torch.equal(cpp_fp8.view(torch.uint8), ref_fp8.view(torch.uint8)), (
        "FP8 data mismatch (non-contiguous)"
    )
    assert torch.equal(cpp_scale.view(torch.uint8), ref_scale.view(torch.uint8)), (
        "Scale data mismatch (non-contiguous)"
    )


@pytest.mark.parametrize(
    "total_kv_len,num_blocks,block_size,k_token_start,num_tokens",
    [
        # FP4 indexer K cache layout: 64 bytes of packed E2M1 codes (two per
        # byte) + 4 bytes of int32 scale (four UE8M0 exponents packed
        # little-endian) = 68 bytes per token. The gather op must accept
        # head_dim=64 and return (num_tokens, 64) bytes that the caller
        # reinterprets as int8 via .view(torch.int8).
        (64, 8, 8, 0, 64),
        (128, 16, 8, 32, 64),
        (512, 64, 8, 100, 300),
    ],
)
def test_indexer_k_cache_gather_contiguous_fp4(
    total_kv_len,
    num_blocks,
    block_size,
    k_token_start,
    num_tokens,
):
    """Exercise the FP4 cache layout (head_dim=64, per-token size=68 B)."""
    device = torch.device("cuda")
    fp4_head_dim = 64
    fp4_bytes_per_token = fp4_head_dim + SCALE_BYTES  # 68

    k_cache = torch.randint(
        0,
        256,
        (num_blocks, block_size, 1, fp4_bytes_per_token),
        dtype=torch.uint8,
        device=device,
    )
    total_slots = num_blocks * block_size
    perm = torch.randperm(total_slots, device=device)[:total_kv_len]
    flat_starts = perm.to(torch.int64) * fp4_bytes_per_token
    slot_fp8 = flat_starts
    slot_scale = flat_starts + fp4_head_dim

    cpp_fp8, cpp_scale = torch.ops.trtllm.indexer_k_cache_gather_op(
        k_cache, slot_fp8, slot_scale, k_token_start, num_tokens, fp4_head_dim
    )

    # Reference: flat-byte gather mirroring the C++ op.
    k_cache_flat = k_cache.contiguous().reshape(-1)
    fp8_bases = slot_fp8[k_token_start : k_token_start + num_tokens]
    byte_offsets_fp8 = torch.arange(fp4_head_dim, device=device, dtype=torch.int64)
    ref_fp8 = k_cache_flat[fp8_bases.unsqueeze(1) + byte_offsets_fp8.unsqueeze(0)]
    scale_bases = slot_scale[k_token_start : k_token_start + num_tokens]
    byte_offsets_scale = torch.arange(SCALE_BYTES, device=device, dtype=torch.int64)
    ref_scale = k_cache_flat[scale_bases.unsqueeze(1) + byte_offsets_scale.unsqueeze(0)]

    assert cpp_fp8.shape == (num_tokens, fp4_head_dim)
    assert cpp_scale.shape == (num_tokens, 1)
    assert torch.equal(cpp_fp8.view(torch.uint8), ref_fp8), "FP4 gather data mismatch"
    assert torch.equal(cpp_scale.view(torch.uint8), ref_scale), "FP4 gather scale mismatch"


def test_indexer_k_cache_gather_empty():
    """Zero-length gather should return correctly shaped empty tensors."""
    device = torch.device("cuda")
    k_cache = torch.randint(0, 256, (4, 8, 1, BYTES_PER_TOKEN), dtype=torch.uint8, device=device)
    slot_fp8 = torch.zeros(10, dtype=torch.int64, device=device)
    slot_scale = torch.zeros(10, dtype=torch.int64, device=device)

    k_fp8, k_scale = torch.ops.trtllm.indexer_k_cache_gather_op(
        k_cache, slot_fp8, slot_scale, k_token_start=5, num_tokens=0, head_dim=HEAD_DIM
    )

    assert k_fp8.shape == (0, HEAD_DIM)
    assert k_scale.shape == (0, 1)
    assert k_fp8.dtype == torch.float8_e4m3fn
    assert k_scale.dtype == torch.float32


# ===================================================================
# Test 2: convert_req_index_to_global
# ===================================================================


def _reference_convert_req_index_to_global(
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    block_size: int,
    stride_factor: int,
    layer_id: int,
) -> torch.Tensor:
    """Python reference for convert_req_index_to_global.

    For each (token_id, indice_id):
        tok = token_indices[token_id, indice_id]
        if tok == -1:
            out = -1
        else:
            block_id = tok // block_size
            inblock_off = tok % block_size + layer_id * block_size
            req = req_id[token_id]
            base = block_table[req, block_id]
            if block_id >= max_blocks_per_req or base < 0:
                out = -1
            else:
                out = base * stride_factor + inblock_off
    """
    num_tokens, num_topk = token_indices.shape
    max_blocks_per_req = block_table.shape[1]

    out = torch.empty_like(token_indices)
    # Move to CPU for the reference loop
    req_id_cpu = req_id.cpu()
    block_table_cpu = block_table.cpu()
    token_indices_cpu = token_indices.cpu()
    out_cpu = out.cpu()

    for i in range(num_tokens):
        req = req_id_cpu[i].item()
        for j in range(num_topk):
            tok = token_indices_cpu[i, j].item()
            if tok < 0:
                out_cpu[i, j] = -1
                continue
            block_id = tok // block_size
            inblock_off = tok % block_size + layer_id * block_size
            if block_id >= max_blocks_per_req:
                out_cpu[i, j] = -1
                continue
            base = block_table_cpu[req, block_id].item()
            if base < 0:
                out_cpu[i, j] = -1
                continue
            out_cpu[i, j] = base * stride_factor + inblock_off

    return out_cpu.to(token_indices.device)


@pytest.mark.parametrize(
    "num_tokens,num_requests,num_topk,block_size,stride_factor,layer_id",
    [
        # Basic case: contiguous pool (stride_factor == block_size), layer 0
        (4, 2, 128, 64, 64, 0),
        # Layer interleaving: stride_factor > block_size
        (4, 2, 128, 64, 192, 1),
        # Single token
        (1, 1, 128, 64, 64, 0),
        # Larger batch
        (32, 8, 256, 64, 64, 0),
        # Small block_size
        (8, 4, 128, 16, 16, 0),
        # layer_id > 0 with contiguous pool
        (4, 2, 128, 64, 64, 2),
        # Large topk
        (4, 2, 2048, 64, 192, 1),
    ],
)
def test_convert_req_index_to_global(
    num_tokens,
    num_requests,
    num_topk,
    block_size,
    stride_factor,
    layer_id,
):
    """Test C++ op against Python reference."""
    device = torch.device("cuda")
    torch.manual_seed(42)

    max_kv_len = 4096
    max_blocks_per_req = (max_kv_len + block_size - 1) // block_size

    # req_id: which request each token belongs to
    req_id = torch.randint(0, num_requests, (num_tokens,), dtype=torch.int32, device=device)

    # block_table: maps (request, block_id) → physical block index
    # Use positive values; some entries can be -1 (padding)
    block_table = torch.randint(
        0, 1000, (num_requests, max_blocks_per_req), dtype=torch.int32, device=device
    )
    # Add some padding (-1) in later blocks
    block_table[:, max_blocks_per_req // 2 :] = -1

    # token_indices: request-local token positions, some -1 (invalid)
    max_valid_token = (max_blocks_per_req // 2) * block_size - 1
    token_indices = torch.randint(
        0, max(max_valid_token, 1), (num_tokens, num_topk), dtype=torch.int32, device=device
    )
    # Sprinkle -1s for invalid tokens (~20%)
    invalid_mask = torch.rand(num_tokens, num_topk, device=device) < 0.2
    token_indices[invalid_mask] = -1

    # C++ op
    cpp_out = torch.ops.trtllm.convert_req_index_to_global(
        req_id, block_table, token_indices, block_size, num_topk, stride_factor, layer_id
    )

    # Reference
    ref_out = _reference_convert_req_index_to_global(
        req_id, block_table, token_indices, block_size, stride_factor, layer_id
    )

    assert cpp_out.shape == ref_out.shape
    assert torch.equal(cpp_out, ref_out), (
        f"Mismatch found.\n"
        f"  First diff at: {(cpp_out != ref_out).nonzero(as_tuple=False)[0].tolist()}\n"
        f"  cpp: {cpp_out[(cpp_out != ref_out)][:5]}\n"
        f"  ref: {ref_out[(cpp_out != ref_out)][:5]}"
    )


def test_convert_req_index_to_global_all_invalid():
    """All token_indices are -1 → entire output should be -1."""
    device = torch.device("cuda")
    num_tokens, num_topk, block_size = 4, 128, 64

    req_id = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    block_table = torch.ones(1, 16, dtype=torch.int32, device=device) * 100
    token_indices = torch.full((num_tokens, num_topk), -1, dtype=torch.int32, device=device)

    out = torch.ops.trtllm.convert_req_index_to_global(
        req_id, block_table, token_indices, block_size, num_topk, block_size, 0
    )

    assert (out == -1).all(), "All outputs should be -1 for all-invalid input"


def test_convert_req_index_to_global_block_table_padding():
    """block_table entries of -1 should produce -1 in output."""
    device = torch.device("cuda")
    num_tokens, num_topk, block_size = 2, 64, 16

    req_id = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    # All block_table entries are -1 (no valid blocks)
    block_table = torch.full((1, 8), -1, dtype=torch.int32, device=device)
    # Valid-looking token indices that would map to these blocks
    token_indices = torch.randint(0, 128, (num_tokens, num_topk), dtype=torch.int32, device=device)

    out = torch.ops.trtllm.convert_req_index_to_global(
        req_id, block_table, token_indices, block_size, num_topk, block_size, 0
    )

    assert (out == -1).all(), "All outputs should be -1 when block_table is all padding"


# ===================================================================
# Test 3: fused_cat_fp4 — bit-exact vs DeepGEMM per_token_cast_to_fp4
# ===================================================================


@pytest.mark.skipif(
    not HAS_DEEPGEMM_REF,
    reason="tensorrt_llm.deep_gemm.utils.math.per_token_cast_to_fp4 unavailable",
)
@skip_pre_blackwell
@pytest.mark.parametrize(
    "shape",
    [
        (4, 128),
        (1, 32, 128),
        (3, 7, 128),
        (2, 5, 4, 128),
    ],
)
@pytest.mark.parametrize("seed", [0, 42, 2026])
def test_fused_cat_fp4_matches_deepgemm(shape, seed):
    """Packed bytes and UE8M0 scale int32 must match DeepGEMM byte-for-byte."""
    torch.manual_seed(seed)
    head_dim = shape[-1]
    leading = shape[:-1]
    rope_dim = head_dim // 2
    pe = torch.randn(*leading, rope_dim, device="cuda", dtype=torch.bfloat16)
    nope = torch.randn(*leading, head_dim - rope_dim, device="cuda", dtype=torch.bfloat16)

    packed, scale = torch.ops.trtllm.fused_cat_fp4(pe, nope)

    # Reference expects 2D (M, head_dim); flatten leading dims.
    cat_2d = torch.cat([pe, nope], dim=-1).reshape(-1, head_dim).contiguous()
    ref_packed, ref_scale = per_token_cast_to_fp4(
        cat_2d, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )

    M = packed.size(0)
    assert packed.shape == (M, head_dim // 2)
    assert scale.shape == (M, 1)
    assert torch.equal(
        packed.view(-1).contiguous(),
        ref_packed.reshape(-1).to(torch.int8).contiguous(),
    ), "FP4 packed bytes mismatch vs DeepGEMM reference"
    assert torch.equal(
        scale.view(-1).contiguous(),
        ref_scale.reshape(-1).to(torch.int32).contiguous(),
    ), "UE8M0 scale int32 mismatch vs DeepGEMM reference"


@skip_pre_blackwell
@pytest.mark.parametrize(
    "pe_dim,nope_dim",
    [
        (64, 64),
        (32, 96),
        (16, 112),
        (96, 32),
    ],
)
def test_fused_cat_fp4_shape_dispatch(pe_dim, nope_dim):
    """Verify the op handles asymmetric pe/nope splits and returns correct shapes."""
    torch.manual_seed(0)
    M = 8
    pe = torch.randn(M, pe_dim, device="cuda", dtype=torch.bfloat16)
    nope = torch.randn(M, nope_dim, device="cuda", dtype=torch.bfloat16)

    packed, scale = torch.ops.trtllm.fused_cat_fp4(pe, nope)

    assert packed.dtype == torch.int8
    assert scale.dtype == torch.int32
    assert packed.shape == (M, (pe_dim + nope_dim) // 2)
    assert scale.shape == (M, 1)


@skip_pre_blackwell
def test_fused_cat_fp4_noncontiguous_split():
    """Op must accept non-contiguous pe/nope views from torch.split()."""
    torch.manual_seed(0)
    M = 16
    head_dim = 128
    x = torch.randn(M, head_dim, device="cuda", dtype=torch.bfloat16)
    pe, nope = x.split([head_dim // 2, head_dim // 2], dim=-1)
    assert not pe.is_contiguous()

    packed, scale = torch.ops.trtllm.fused_cat_fp4(pe, nope)

    # Contiguous reference via pre-materialized copies.
    packed_ref, scale_ref = torch.ops.trtllm.fused_cat_fp4(pe.contiguous(), nope.contiguous())

    assert torch.equal(packed, packed_ref)
    assert torch.equal(scale, scale_ref)


@skip_pre_blackwell
@pytest.mark.parametrize(
    "num_tokens,n_heads",
    [
        # DSV3.2 production shapes from the 8192-token prefill warmup that
        # caused CUDA illegal-memory-access in test_nvfp4_multi_gpus[fp4_indexer]:
        # q has shape (num_tokens, n_heads=64, head_dim=128) and is split into
        # q_pe / q_nope along dim -1 — pe/nope are 3D non-contiguous views with
        # stride(-2) = 128 (inherited from the pre-split tensor).
        (24, 64),  # generation-warmup batch size with n_heads=64
        (2048, 64),  # smaller prefill chunk
        (8192, 64),  # full prefill warmup
    ],
)
def test_fused_cat_fp4_dsv32_prefill_shape(num_tokens, n_heads):
    """Reproduces the 3D split pattern the real DSV3.2 model feeds to
    fused_cat_fp4 during prefill, across growing token counts.

    After `q.view(-1, n_heads, head_dim).split([64, 64], dim=-1)`:
      q_pe   shape (num_tokens, n_heads, 64), stride (n_heads*128, 128, 1)
      q_nope shape (num_tokens, n_heads, 64), stride (n_heads*128, 128, 1)
    """
    torch.manual_seed(0)
    head_dim = 128
    q = torch.randn(num_tokens, n_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    q_pe, q_nope = q.split([head_dim // 2, head_dim // 2], dim=-1)
    assert not q_pe.is_contiguous()
    assert q_pe.stride(-2) == head_dim, f"expected stride(-2)={head_dim}, got {q_pe.stride(-2)}"

    packed, scale = torch.ops.trtllm.fused_cat_fp4(q_pe, q_nope)

    assert packed.shape == (num_tokens * n_heads, head_dim // 2)
    assert scale.shape == (num_tokens * n_heads, 1)

    # Bit-exact vs contiguous copies — stride independence.
    packed_ref, scale_ref = torch.ops.trtllm.fused_cat_fp4(q_pe.contiguous(), q_nope.contiguous())
    assert torch.equal(packed, packed_ref)
    assert torch.equal(scale, scale_ref)

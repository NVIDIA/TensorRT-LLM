import torch
import triton
import triton.language as tl

########################################################
# Convert request-local indices to global KV cache pool indices
########################################################

@triton.jit
def _convert_req_index_to_global_index_kernel_with_stride_factor(
    req_id_ptr,  # int32 [num_tokens]
    block_table_ptr,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    out_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    # shapes
    max_num_blocks_per_req,
    BLOCK_SIZE,
    BLOCK_N: tl.constexpr,  # tile width along columns (used in tl.arange)
    stride_factor,  # for strided memory layout adjustment
    layer_id,  # for layer interleaving layout
    bt_stride0,
    bt_stride1,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
):
    """
    Triton kernel for converting request-local token indices to global KV cache pool indices.
    Derived from vllm's flashmla_sparse.py, with stride_factor fused in the kernel.
    """
    # program_id(0) -> token_id (row)
    # program_id(1) -> tile index along columns
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    # Each program covers BLOCK_N consecutive columns
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load request id for this token (no mask: grid is exact)
    req = tl.load(req_id_ptr + token_id)

    # Load token indices for this tile
    ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    tok = tl.load(ti_ptr)  # int32

    # Only token == -1 should propagate as -1
    is_invalid_tok = tok < 0

    # Clamp negative tokens to 0 for safe block_id computation.
    # The result is discarded via is_invalid_tok below, but this prevents
    # computing negative block_id which would cause OOB block_table loads.
    safe_tok = tl.maximum(tok, 0)

    # Compute block id and in-block offset
    block_id = safe_tok // BLOCK_SIZE
    inblock_off = safe_tok % BLOCK_SIZE + layer_id * BLOCK_SIZE

    # Guard block_table access
    valid_block = block_id < max_num_blocks_per_req
    bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
    base = tl.load(bt_ptr, mask=valid_block & (~is_invalid_tok), other=0)

    # If token == -1 OR block_id OOB OR block_table has -1 (padding), output -1
    # Otherwise: base * stride_factor + inblock_off
    # (stride_factor accounts for layer interleaving in strided KV cache pools)
    is_invalid = is_invalid_tok | (~valid_block) | (base < 0)
    out_val = tl.where(is_invalid, -1, base * stride_factor + inblock_off)

    # Store results
    out_ptr_ij = out_ptr + token_id * out_stride0 + indice_id * out_stride1
    tl.store(out_ptr_ij, out_val)


def triton_convert_req_index_to_global_index(
        req_id: torch.Tensor,  # int32 [num_tokens]
        block_table: torch.
    Tensor,  # int32 [num_requests, max_num_blocks_per_req]
        token_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS]
        BLOCK_SIZE: int,
        NUM_TOPK_TOKENS: int = 2048,
        BLOCK_N: int = 128,  # tile width along columns
        stride_factor:
    int = None,  # for strided memory layout (with layer interleaving), defaults to BLOCK_SIZE
        layer_id: int = 0,  # for layer interleaving layout
):
    """
    Convert request-local token indices to global KV cache pool indices.

    out[token_id, indice_id] =
        block_table[req_id[token_id],
            token_indices[token_id, indice_id] // BLOCK_SIZE] * stride_factor
        + token_indices[token_id, indice_id] % BLOCK_SIZE

    Args:
        stride_factor: Memory stride between consecutive blocks (default: BLOCK_SIZE).
                        For non-contiguous pools with layer interleaving, use
                        (num_layers * BLOCK_SIZE) to account for memory gaps.

    Only when token_indices[token_id, indice_id] == -1 do we output -1.
    For safety, we also output -1 if the derived block_id would be
        out-of-bounds.
    """
    if stride_factor is None:
        stride_factor = BLOCK_SIZE
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, \
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by" \
        f"BLOCK_N ({BLOCK_N})"

    num_tokens = req_id.shape[0]
    num_requests, max_num_blocks_per_req = block_table.shape
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    # Ensure contiguous tensors on the same device
    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()
    out = torch.empty_like(token_indices_c)

    # Strides in elements
    bt_stride0, bt_stride1 = block_table_c.stride()
    ti_stride0, ti_stride1 = token_indices_c.stride()
    out_stride0, out_stride1 = out.stride()

    # Exact 2D grid: tokens × column tiles
    grid = (num_tokens, tiles_per_row)

    _convert_req_index_to_global_index_kernel_with_stride_factor[grid](
        req_id_c,
        block_table_c,
        token_indices_c,
        out,
        # shapes / constexprs
        max_num_blocks_per_req,
        BLOCK_SIZE,
        BLOCK_N,
        stride_factor,
        # strides
        layer_id,
        bt_stride0,
        bt_stride1,
        ti_stride0,
        ti_stride1,
        out_stride0,
        out_stride1,
    )
    return out


########################################################
# Fused K cache gather kernel
########################################################


@triton.jit
def _triton_gather_k_cache_kernel(
    k_cache_ptr,
    slot_fp8_ptr,
    slot_scale_ptr,
    out_fp8_ptr,
    out_scale_ptr,
    k_token_start,
    num_k_tokens,
    HEAD_DIM: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    pid = tl.program_id(0)
    token_offsets = (pid * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)).to(
        tl.int64)
    token_mask = token_offsets < num_k_tokens

    fp8_base = tl.load(slot_fp8_ptr + k_token_start + token_offsets,
                       mask=token_mask,
                       other=0)
    scale_base = tl.load(slot_scale_ptr + k_token_start + token_offsets,
                         mask=token_mask,
                         other=0)

    byte_offsets = tl.arange(0, HEAD_DIM).to(tl.int64)
    src_fp8 = fp8_base[:, None] + byte_offsets[None, :]
    dst_fp8 = token_offsets[:, None] * HEAD_DIM + byte_offsets[None, :]
    gather_mask = token_mask[:, None]

    fp8_data = tl.load(k_cache_ptr + src_fp8, mask=gather_mask, other=0)
    tl.store(out_fp8_ptr + dst_fp8, fp8_data, mask=gather_mask)

    scale_byte_offsets = tl.arange(0, SCALE_BYTES).to(tl.int64)
    src_scale = scale_base[:, None] + scale_byte_offsets[None, :]
    dst_scale = token_offsets[:,
                              None] * SCALE_BYTES + scale_byte_offsets[None, :]

    scale_data = tl.load(k_cache_ptr + src_scale, mask=gather_mask, other=0)
    tl.store(out_scale_ptr + dst_scale, scale_data, mask=gather_mask)


def triton_gather_k_cache(
    k_cache: torch.Tensor,
    slot_mapping_fp8: torch.Tensor,
    slot_mapping_scale: torch.Tensor,
    k_token_start: int,
    k_token_end: int,
    head_dim: int,
):
    """Gather K FP8 values and scales from the indexer K cache for a chunk.

    Replaces ``_gather_k_cache_for_chunk``, fusing ~8-12 small PyTorch ops
    (arange, unsqueeze, broadcast add, _unravel_indices, advanced indexing)
    into a single Triton kernel that directly gathers from flat byte offsets.
    This is purely data movement — bit-exact with the original.

    Args:
        k_cache: Indexer K cache pool data (2D contiguous), uint8.
        slot_mapping_fp8: Flat byte indices for FP8 data
            ``[total_kv_len]``, int64.
        slot_mapping_scale: Flat byte indices for scale data
            ``[total_kv_len]``, int64.
        k_token_start: Start index into slot mapping arrays.
        k_token_end: End index into slot mapping arrays.
        head_dim: FP8 head dimension (typically 128).

    Returns:
        Tuple of (k_fp8, k_scale):
            k_fp8: ``[num_k_tokens, head_dim]``, float8_e4m3fn.
            k_scale: ``[num_k_tokens, 1]``, float32.
    """
    num_k_tokens = k_token_end - k_token_start
    device = k_cache.device

    if num_k_tokens == 0:
        return (
            torch.empty(0, head_dim, dtype=torch.float8_e4m3fn, device=device),
            torch.empty(0, 1, dtype=torch.float32, device=device),
        )

    SCALE_BYTES = 4
    BLOCK_TOKENS = 32

    k_cache_flat = k_cache.reshape(-1)
    out_fp8 = torch.empty(num_k_tokens,
                          head_dim,
                          dtype=torch.uint8,
                          device=device)
    out_scale = torch.empty(num_k_tokens,
                            SCALE_BYTES,
                            dtype=torch.uint8,
                            device=device)

    grid = (triton.cdiv(num_k_tokens, BLOCK_TOKENS), )
    _triton_gather_k_cache_kernel[grid](
        k_cache_flat,
        slot_mapping_fp8,
        slot_mapping_scale,
        out_fp8.view(-1),
        out_scale.view(-1),
        k_token_start,
        num_k_tokens,
        HEAD_DIM=head_dim,
        SCALE_BYTES=SCALE_BYTES,
        BLOCK_TOKENS=BLOCK_TOKENS,
    )

    k_fp8 = out_fp8.view(torch.float8_e4m3fn)
    k_scale = out_scale.view(torch.float32).view(num_k_tokens, 1)
    return k_fp8, k_scale

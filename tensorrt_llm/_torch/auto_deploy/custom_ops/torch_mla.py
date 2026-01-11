import math

import torch
import torch.nn as nn


# This is the original unmodified code from Hugging Face
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """

    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@torch.library.custom_op("auto_deploy::torch_deepseek_mla_no_cache", mutates_args=())
def torch_deepseek_mla_no_cache(
    q_normed_dn: torch.Tensor,  # [bsz, q_len, self.q_lora_rank=1536]
    compressed_kv: torch.Tensor,  # [bsz, q_len, 512]
    k_pe: torch.Tensor,  # [bsz, 1, q_len, 64]
    sin: torch.Tensor,
    cos: torch.Tensor,
    wkv_b: torch.Tensor,  # [128 * 256, 512]
    wq_b: torch.Tensor,  # [self.num_heads * self.q_head_dim, self.q_lora_rank]
    w_uq_ukv: torch.Tensor,
    wo_proj: torch.Tensor,
    w_uv_o: torch.Tensor,  # Absorbed weights (W^UV * W_O)
    # METADATA
    position_ids: torch.Tensor,
    # CONSTANTS
    softmax_scale: float,
    attention_mask: torch.Tensor = None,
) -> torch.Tensor:
    """MHA mode for MLA (i.e. no weight absorption)

    * No cache
    * Invoked from the deepseek.py patch and replaced by the MLA backend when optimizing the model.
    """
    v_head_dim = 128
    num_heads = 128
    qk_nope_head_dim = 128
    bsz, q_len, q_lora_rank = q_normed_dn.shape
    ckv_bsz, kv_seq_len, head_dim_ckv = compressed_kv.shape
    assert ckv_bsz == bsz
    assert kv_seq_len == q_len
    assert sin is not None
    assert cos is not None
    assert w_uq_ukv is None and w_uv_o is None, "w_uq_ukv and w_uv_o must be None"

    # q_normed_dn * (W^UQ and W^QR) (i.e. q up projection)
    wq_b = wq_b.reshape(num_heads, -1, q_lora_rank)
    q = torch.einsum("bsl,hdl->bhsd", q_normed_dn, wq_b)  # [bsz, 128, q_len, 192]
    q_head_dim = q.shape[-1]  # 192
    qk_rope_head_dim = q_head_dim - qk_nope_head_dim
    assert qk_rope_head_dim == 64

    # Separate q into the no-positional encoding part (q_nope) and the positional encoding part (q_pe)
    q_nope, q_pe = torch.split(
        q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1
    )  # q_nope ~ [bsz, 128, q_len, 128], q_pe ~ [bsz, 128, q_len, 64]

    # Ensure contiguous memory layout for CUDA operations
    q_nope = q_nope.contiguous()
    q_pe = q_pe.contiguous()

    # kv = c_K * W_UK (i.e. upward projection)
    kv = (
        torch.einsum("bsc,xc->bsx", compressed_kv, wkv_b)  # [bsz, q_len, 128*512]
        .view(bsz, q_len, num_heads, qk_nope_head_dim + v_head_dim)  # [bsz, q_len, 128, 256]
        .transpose(1, 2)  # [bsz, 128, q_len, 256]
    )

    k_nope, value_states = torch.split(
        kv, [qk_nope_head_dim, v_head_dim], dim=-1
    )  # k_nope ~ [bsz, 128, q_len, 128], value_states ~ [bsz, 128, q_len, 128]

    # Ensure contiguous memory layout for CUDA operations
    k_nope = k_nope.contiguous()
    value_states = value_states.contiguous()

    if position_ids is None:
        q_pe, k_pe = apply_rotary_pos_emb(
            q_pe, k_pe, cos[:kv_seq_len], sin[:kv_seq_len], position_ids
        )
    else:
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

    query_states = k_pe.new_empty(bsz, num_heads, q_len, q_head_dim)  # [bsz, 128, q_len, 192]
    query_states[:, :, :, :qk_nope_head_dim] = q_nope
    query_states[:, :, :, qk_nope_head_dim:] = q_pe

    key_states = k_pe.new_empty(bsz, num_heads, q_len, q_head_dim)  # [bsz, 128, q_len, 192]
    key_states[:, :, :, :qk_nope_head_dim] = k_nope
    key_states[:, :, :, qk_nope_head_dim:] = k_pe

    # Batched matmul: [bsz, num_heads, q_len, 192] @ [bsz, num_heads, 192, kv_seq_len].transpose(-1, -2)
    attn_weights = (
        torch.matmul(query_states, key_states.transpose(-1, -2)) * softmax_scale
    )  # [bsz, num_heads, q_len, kv_seq_len]

    if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )
    # Apply attention mask (which contains proper causal masking)
    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask
    else:
        causal_mask = (
            torch.triu(
                torch.ones(q_len, kv_seq_len, device=q_normed_dn.device, dtype=torch.bool),
                diagonal=1,  # Use diagonal=1 for standard causal masking
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        attn_weights.masked_fill_(causal_mask, float("-inf"))

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )

    # attn_output = torch.matmul(attn_weights, v_batched_t)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, num_heads, q_len, v_head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, num_heads, q_len, v_head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, num_heads * v_head_dim)
    attn_output = torch.einsum("bsd,td->bst", attn_output, wo_proj)
    return attn_output


@torch_deepseek_mla_no_cache.register_fake
def torch_deepseek_mla_no_cache_fake(
    q_normed_dn: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    wkv_b: torch.Tensor,
    wq_b: torch.Tensor,
    w_uq_ukv: torch.Tensor,
    wo_proj: torch.Tensor,
    w_uv_o: torch.Tensor,
    # METADATA
    position_ids: torch.Tensor,
    # CONSTANTS
    # attention_mask: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    hidden_size = wo_proj.shape[0]
    bsz, q_len, _ = q_normed_dn.shape

    attn_output = torch.empty(bsz, q_len, hidden_size).to(
        device=q_normed_dn.device, dtype=q_normed_dn.dtype
    )
    return attn_output


# @torch.library.custom_op("auto_deploy::torch_deepseek_prefill_no_absorb_attn", mutates_args=())
@torch.library.custom_op("auto_deploy::torch_deepseek_mla_with_kv_cache", mutates_args=())
def torch_deepseek_mla_with_kv_cache(
    q_normed_dn: torch.Tensor,  # [bsz, q_len, q_lora_rank]
    compressed_kv: torch.Tensor,  # [bsz, kv_len, 512]
    k_pe: torch.Tensor,  # [bsz, 1, kv_len, 64]
    sin_cache: torch.Tensor,
    cos_cache: torch.Tensor,
    wkv_b: torch.Tensor,  # [128 * 256, 512]
    wq_b: torch.Tensor,  # [num_heads * q_head_dim, q_lora_rank]
    w_uq_ukv: torch.Tensor,  # Absorbed weights (W_UQ * W_UK)
    wo_proj: torch.Tensor,  # [hidden_size, num_heads * v_head_dim]
    w_uv_o: torch.Tensor,  # Absorbed weights (W^UV * W_O)
    # METADATA
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    seq_start: torch.Tensor,
    position_ids: torch.Tensor,
    # CACHES
    ckv_cache: torch.Tensor,  # [page_num, page_size, ckv_dim]
    k_pe_cache: torch.Tensor,  # [page_num, page_size, k_pe_dim]
    # CONSTANTS
    # attention_mask: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """MHA mode for MLA (i.e. no weight absorption)

    * Cached
    * Serves both prefill and decode
    * Requests are either all decode or mixed prefill+decode.
    * When all requests are decode then input shape is [bsz=nb_seqs, q_seq_len=1, dim].
      I.e. request is its own sequence of len 1.
    * When requests are mixed prefill+decode then input shape is
      [bsz=1, q_seq_len=sum(seq_len), dim].
      I.e. all requests are flattened into a single sequence of length q_len.
      This is like a ragged tensor. Use seq_lens to determine the actual number of sequences
      and tokens per sequence. Use seq_start to determine the actual start position of each
      sequence.

    Expect one of two cases here:
    1. b > 0, s==1: this indicates a generate-only batch of tokens.
    2. b==1, s > 0: this indicates a mixed context+generate phase. The actual number of sequences
       and number of tokens per sequence are encoded in seq_len and seq_start.

    If q_seq_len == 1 ==> this is decode-only, otherwise it's mixed prefill+decode
    """

    original_bsz, q_seq_len = q_normed_dn.shape[0:2]

    if original_bsz != 1:
        # This is a decode-only batch
        assert q_seq_len == 1, "q_seq_len of each request must be 1 for decode-only batch"
        # (bsz, 1, dim) -> (1, bsz, dim) like in the case of mixed prefill+decode
        q_normed_dn = q_normed_dn.transpose(0, 1)
        compressed_kv = compressed_kv.transpose(0, 1)
        k_pe = k_pe.transpose(0, 2)
        if position_ids is not None:
            position_ids = position_ids.transpose(0, 1)

    #
    # From this point on, we treat decode-only and mixed prefill+decode the same
    #

    v_head_dim = 128
    num_heads = 128
    qk_nope_head_dim = 128
    # Todo: is q_len even a meaningful variable here?
    bsz, q_len, q_lora_rank = q_normed_dn.shape
    ckv_bsz, kv_seq_len, head_dim_ckv = compressed_kv.shape
    assert ckv_bsz == bsz
    assert kv_seq_len == q_len
    assert sin_cache is not None
    assert cos_cache is not None
    assert bsz == 1

    # q_normed_dn * (W^UQ and W^QR) (i.e. q up projection)
    wq_b = wq_b.reshape(num_heads, -1, q_lora_rank)
    # W_UKV: wq_b ~ [num_heads * q_head_dim, q_lora_rank]
    # q_normed_dn ~ [bsz, q_len, q_lora_rank]
    q = torch.einsum("bsl,hdl->bhsd", q_normed_dn, wq_b)  # [bsz, 128, q_len, 192]
    q_head_dim = q.shape[-1]  # 192
    qk_rope_head_dim = q_head_dim - qk_nope_head_dim
    assert qk_rope_head_dim == 64

    # Separate q into the no-positional encoding part (q_nope) and the positional encoding part (q_pe)
    q_nope, q_pe = torch.split(
        q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1
    )  # q_nope ~ [bsz, 128, q_len, 128], q_pe ~ [bsz, 128, q_len, 64]

    # Ensure contiguous memory layout for CUDA operations
    q_nope = q_nope.contiguous()
    q_pe = q_pe.contiguous()

    # Apply RoPE only on the new (uncached) tokens positional encodings
    if position_ids is None:
        q_pe, k_pe = apply_rotary_pos_emb(
            q_pe, k_pe, cos_cache[:kv_seq_len], sin_cache[:kv_seq_len], position_ids
        )
    else:
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos_cache, sin_cache, position_ids)

    # Store the tokens in the cache, after adding position information
    k_pe = k_pe.squeeze(0)  # why is this here?
    update_ckv_kpe_cache(
        compressed_kv, k_pe, ckv_cache, k_pe_cache, seq_len, input_pos, cache_loc, seq_start
    )

    # kv = c_K * W^UK (i.e. upward projection)
    # Use entire cache
    kv = (
        torch.einsum(
            "bsc,xc->bsx",
            ckv_cache,
            wkv_b,
        )  # [bsz, q_len, 128*512]
        .view(
            ckv_cache.shape[0], ckv_cache.shape[1], num_heads, qk_nope_head_dim + v_head_dim
        )  # [bsz, q_len, 128, 256]
        .transpose(1, 2)  # [bsz, 128, ckv_cache.shape[1], 256]
    )

    # Over entire cache
    k_nope, value_states = torch.split(
        kv, [qk_nope_head_dim, v_head_dim], dim=-1
    )  # k_nope ~ [bsz, 128, q_len, 128], value_states ~ [bsz, 128, q_len, 128]

    # Ensure contiguous memory layout for CUDA operations
    k_nope = k_nope.contiguous()
    value_states = value_states.contiguous()

    # Concatenate q_nope and q_pe
    query_states = k_pe.new_empty(bsz, num_heads, q_len, q_head_dim)  # [bsz, 128, q_len, 192]
    query_states[..., :qk_nope_head_dim] = q_nope
    query_states[..., qk_nope_head_dim:] = q_pe

    # Concatenate k_nope and k_pe
    key_states = k_pe.new_empty(
        # bsz, num_heads, q_len, q_head_dim
        kv.shape[:-1] + (q_head_dim,)
    )  # [bsz, 128, q_len, 192]
    key_states[..., :qk_nope_head_dim] = k_nope
    key_states[..., qk_nope_head_dim:] = k_pe_cache.unsqueeze(1)

    # Process one sequence at a time
    attn_outputs = []
    nb_seqs = seq_len.shape[0]
    for i in range(nb_seqs):
        seq_len_i = seq_len[i]
        seq_start_i = seq_start[i]
        input_pos_i = input_pos[i]
        cache_loc_i = cache_loc[i]

        # Batched matmul: [bsz, num_heads, q_len, 192] @ [bsz, num_heads, 192, kv_seq_len].transpose(-1, -2)
        attn_weights = (
            torch.matmul(
                query_states[:, :, seq_start_i : seq_start_i + seq_len_i, :],
                key_states[cache_loc_i, :, seq_start_i : seq_start_i + seq_len_i, :].transpose(
                    -1, -2
                ),
            )
            * softmax_scale
        )  # [bsz, num_heads, q_len, kv_seq_len]

        if attn_weights.size() != (bsz, num_heads, seq_len_i, seq_len_i):
            raise ValueError(
                f"Attention weights should be of size {(bsz, num_heads, seq_len_i, seq_len_i)}, but is"
                f" {attn_weights.size()}"
            )

        # Apply attention mask for prefill requests only
        if seq_len_i > 1:
            kv_seq_len = input_pos_i + seq_len_i
            causal_mask = (
                torch.triu(
                    torch.ones(seq_len_i, kv_seq_len, device=q_normed_dn.device, dtype=torch.bool),
                    diagonal=1,  # Use diagonal=1 for standard causal masking
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            attn_weights.masked_fill_(causal_mask, float("-inf"))

        # upcast attention to fp32 ???
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        # attn_output = torch.matmul(attn_weights, v_batched_t)
        attn_output = torch.matmul(
            attn_weights, value_states[cache_loc_i, :, seq_start_i : seq_start_i + seq_len_i, :]
        )

        if attn_output.size() != (bsz, num_heads, seq_len_i, v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, num_heads, seq_len_i, v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, seq_len_i, num_heads * v_head_dim)
        attn_outputs.append(attn_output)

    attn_output = torch.cat(attn_outputs, dim=-2)  # .unsqueeze(0)
    if original_bsz != 1:
        attn_output = attn_output.transpose(0, 1)
    attn_output = torch.einsum("bsd,td->bst", attn_output, wo_proj)
    return attn_output


@torch_deepseek_mla_with_kv_cache.register_fake
def torch_deepseek_mla_with_kv_cache_fake(
    q_normed_dn: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    sin_cache: torch.Tensor,
    cos_cache: torch.Tensor,
    wkv_b: torch.Tensor,
    wq_b: torch.Tensor,
    w_uq_ukv: torch.Tensor,
    wo_proj: torch.Tensor,
    w_uv_o: torch.Tensor,
    # METADATA
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    seq_start: torch.Tensor,
    position_ids: torch.Tensor,
    # CACHES
    ckv_cache: torch.Tensor,
    k_pe_cache: torch.Tensor,
    # CONSTANTS
    # attention_mask: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    hidden_size = wo_proj.shape[0]
    bsz, q_len, _ = q_normed_dn.shape

    attn_output = torch.empty(bsz, q_len, hidden_size).to(
        device=q_normed_dn.device, dtype=q_normed_dn.dtype
    )
    return attn_output


def update_ckv_kpe_cache(
    ckv: torch.Tensor,
    kpe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    seq_len: torch.Tensor,  # metadata
    input_pos: torch.Tensor,  # position in sequence or page
    cache_loc: torch.Tensor,  # batch or page
    seq_start: torch.Tensor,
) -> torch.Tensor:
    """
    Reference implementation for update kv cache function. Assumes KV cache layout to be [B,S,D].
    This function can be used to build reference attention implementations that use KV cache.
    """

    for idx in range(seq_len.shape[0]):
        ckv_cache[cache_loc[idx], input_pos[idx] : input_pos[idx] + seq_len[idx], :] = ckv[
            :, seq_start[idx] : seq_start[idx] + seq_len[idx], ...
        ]
        kpe_cache[cache_loc[idx], input_pos[idx] : input_pos[idx] + seq_len[idx], :] = kpe[
            :, seq_start[idx] : seq_start[idx] + seq_len[idx], ...
        ]


def update_ckv_kpe_cache_paged(
    ckv: torch.Tensor,
    kpe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    seq_len: torch.Tensor,  # metadata
    input_pos: torch.Tensor,  # metadata
    cache_loc: torch.Tensor,
    seq_start: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """
    Reference implementation for update kv cache function. Assumes KV cache layout to be [max_nb_pages,page_size,D].
    This function can be used to build reference attention implementations that use KV cache.
    """

    seq_start_loc = 0
    for seq_idx in range(seq_len.shape[0]):
        kv_page_indptr = math.cumsum(seq_len[:seq_idx])  # location of the first page in the block
        nb_pages_in_seq = math.ceil(seq_len[seq_idx] / page_size)
        tokens_to_copy = (
            seq_len[seq_idx] % page_size if seq_len[seq_idx] % page_size != 0 else page_size
        )

        # Copy the sequence tokens one page at a time
        for page_idx_in_block in range(nb_pages_in_seq):
            if page_idx_in_block == 0:
                tokens_to_copy -= input_pos[seq_idx]
                copy_start_pos = input_pos[seq_idx]
                src_start_pos = seq_start[seq_idx]
            else:
                copy_start_pos = 0
                tokens_to_copy = page_size
                src_start_pos += page_size
            if page_idx_in_block == nb_pages_in_seq - 1:
                tokens_to_copy = (
                    seq_len[seq_idx] % page_size if seq_len[seq_idx] % page_size != 0 else page_size
                )
            virtual_page_loc = kv_page_indptr + page_idx_in_block

            ckv_cache[
                cache_loc[virtual_page_loc], copy_start_pos : copy_start_pos + tokens_to_copy, :
            ] = ckv[:, src_start_pos : src_start_pos + tokens_to_copy, ...]
            kpe_cache[
                cache_loc[virtual_page_loc],
                input_pos[seq_idx] : input_pos[seq_idx] + seq_len[seq_idx],
                :,
            ] = kpe[:, :, src_start_pos : src_start_pos + seq_len[seq_idx], ...]
            seq_start_loc += seq_len[seq_idx]

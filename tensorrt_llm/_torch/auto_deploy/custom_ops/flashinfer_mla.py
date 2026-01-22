import flashinfer
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.torch_mla import apply_rotary_pos_emb

WORKSPACE_SIZE = 128 * 1024 * 1024
global_flashinfer_mla_workspace_buffer = None  # must be zero initialized


def get_workspace_buffer(device):
    global global_flashinfer_mla_workspace_buffer
    if global_flashinfer_mla_workspace_buffer is None:
        global_flashinfer_mla_workspace_buffer = torch.zeros(
            WORKSPACE_SIZE, dtype=torch.int8, device=device
        )
    return global_flashinfer_mla_workspace_buffer


def compute_w_uq_uk(w_uq, w_uk) -> torch.Tensor:
    """Compute W_UQ_UK <= W_UQ * W_UK"""
    w_uq_uk = torch.einsum("qhd,hdk->qhk", w_uq, w_uk)  # [q_lora_rank, num_heads, kv_lora_rank]
    return w_uq_uk


def compute_w_uv_o(w_uv, wo_proj, kv_lora_rank, num_heads, v_head_dim) -> torch.Tensor:
    """Compute W_UV_O <= W_UV * W_O"""
    wo_proj = wo_proj.reshape(-1, num_heads, v_head_dim)  # [hidden_size, num_heads, v_head_dim]
    w_uv_o = torch.einsum("hvl,dhv->hld", w_uv, wo_proj)  # [num_heads, kv_lora_rank, hidden_size]
    w_uv_o = w_uv_o.reshape(num_heads * kv_lora_rank, -1)
    return w_uv_o


@torch.library.custom_op("auto_deploy::flashinfer_deepseek_mla_with_kv_cache", mutates_args=())
def flashinfer_deepseek_mla_with_kv_cache(
    q_normed_dn: torch.Tensor,  # [bsz, q_len, self.q_lora_rank]
    compressed_kv: torch.Tensor,  # [bsz, kv_len, kv_lora_rank]
    k_pe: torch.Tensor,  # [bsz, 1, kv_len, qk_rope_head_dim]
    sin_cache: torch.Tensor,
    cos_cache: torch.Tensor,
    wkv_b: torch.Tensor,  # [num_heads * (qk_nope_head_dim+v_head_dim), kv_lora_rank]
    wq_b: torch.Tensor,  # [num_heads * q_head_dim, q_lora_rank]
    w_uq_uk: torch.Tensor,  # Absorbed weights (W_UQ * W_UK)
    wo_proj: torch.Tensor,  # [hidden_size, num_heads * v_head_dim]
    w_uv_o: torch.Tensor,  # Absorbed weights (W^UV * W_O)
    # METADATA
    q_indptr: torch.Tensor,
    kv_page_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    kv_lens: torch.Tensor,
    page_size: torch.Tensor,
    position_ids: torch.Tensor,
    # CACHES
    ckv_cache: torch.Tensor,  # [page_num, page_size, ckv_dim]
    k_pe_cache: torch.Tensor,  # [page_num, page_size, k_pe_dim]
    # CONSTANTS
    # attention_mask: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """MQA mode for MLA (i.e. with weight absorption)

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
        q_normed_dn = q_normed_dn.transpose(0, 1).contiguous()
        compressed_kv = compressed_kv.transpose(0, 1).contiguous()
        k_pe = k_pe.transpose(0, 2).contiguous()
        if position_ids is not None:
            position_ids = position_ids.transpose(0, 1).contiguous()

    q_normed_dn = q_normed_dn.contiguous()

    #
    # From this point on, we treat decode-only and mixed prefill+decode the same
    #

    v_head_dim = 128
    qk_nope_head_dim = 128
    qk_rope_head_dim = k_pe.shape[-1]
    assert qk_rope_head_dim == 64
    q_head_dim = qk_rope_head_dim + qk_nope_head_dim
    assert q_head_dim == 192
    num_heads = wq_b.shape[0] // q_head_dim
    assert num_heads == 128
    bsz, q_len, q_lora_rank = q_normed_dn.shape
    ckv_bsz, kv_seq_len, head_dim_ckv = compressed_kv.shape
    kv_lora_rank = head_dim_ckv
    assert ckv_bsz == bsz
    assert kv_seq_len == q_len
    assert sin_cache is not None
    assert cos_cache is not None
    assert bsz == 1

    # NEW - absorbed q_pe prep
    w_uq_qr = wq_b  # W_{UQ + QR}: wq_b ~ [num_heads * q_head_dim, q_lora_rank]
    w_uq_qr_t = (
        w_uq_qr.transpose(0, 1).reshape(q_lora_rank, num_heads, q_head_dim).contiguous()
    )  # [q_lora_rank, num_heads, q_head_dim]

    w_uq = w_uq_qr_t[
        :, :, :qk_nope_head_dim
    ].contiguous()  # [q_lora_rank, num_heads, qk_nope_head_dim]
    w_qr = w_uq_qr_t[
        :, :, qk_nope_head_dim:
    ].contiguous()  # [q_lora_rank, num_heads, qk_rope_head_dim]

    w_ukv = wkv_b
    w_ukv = w_ukv.view(
        num_heads, -1, kv_lora_rank
    )  # [num_heads, qk_nope_head_dim+v_head_dim, kv_lora_rank]
    w_uk = w_ukv[
        :, :qk_nope_head_dim, :
    ].contiguous()  # [num_heads, qk_nope_head_dim, kv_lora_rank]
    w_uv = w_ukv[:, -v_head_dim:]  # [num_heads, v_head_dim, kv_lora_rank]

    # q_pe = (x * W_DQ) * W_QR (i.e. q positional encoding))
    q_pe = torch.einsum("bsl,lhd->bhsd", q_normed_dn, w_qr)

    # Weight absorption: (c_Q * W^UQ) * W^UKV
    # q_nope = (x * W_DQ) * W_UQ (i.e. q_nope up projection)
    if w_uq_uk is None:
        # Weight absorption: (c_Q * W^UQ) * W^UKV
        # q_nope = (x * WDQ) * W^UQ (i.e. q_nope up projection)
        # q_normed_dn ~ [bsz, q_len, q_lora_rank]
        q_nope = torch.einsum(
            "bsl,lhd->bhsd", q_normed_dn, w_uq
        )  # [bsz, num_heads, q_len, qk_nope_head_dim]

        # Weight absorption: (c_Q * W^UQ) * W^UK
        # q_nope is already c_Q * W^UQ
        q_nope = q_nope.transpose(1, 2).contiguous()  # [bsz, q_len, num_heads, qk_nope_head_dim]
        q_nope_absorbed = torch.einsum(
            "bshd,hdc->bshc", q_nope, w_uk
        )  # [bsz, q_len, num_heads, kv_lora_rank]
        q_nope = None  # [bsz, q_len, 128, 512]
    else:
        q_nope_absorbed = torch.einsum(
            "bsq,qhk->bshk", q_normed_dn, w_uq_uk
        )  # [bsz, q_len, num_heads, kv_lora_rank]
        q_nope = None  # [bsz, q_len, 128, 512]

    assert position_ids is not None
    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos_cache, sin_cache, position_ids)
    q_pe = q_pe.transpose(1, 2).contiguous()
    k_pe = k_pe.squeeze(1)  # [bsz, kv_seq_len, 64]

    # Store the tokens in the cache, after adding position information
    # positions = torch.arange(0, 32, device=q_pe.device, dtype=torch.int32)  ## < THIS AINT RIGHT
    # kv_page_indptr = kv_page_indptr * 8
    flashinfer.append_paged_mla_kv_cache(
        compressed_kv.reshape(-1, head_dim_ckv),
        k_pe.reshape(-1, qk_rope_head_dim),
        # from qo_indptr - The batch indices of each entry in the appended KV pairs,
        # shape: [append_indptr[-1]].
        batch_indices,
        # from qo_indptr - The positions of each entry in the appended KV pairs,
        # shape: [append_indptr[-1]].
        positions,
        ckv_cache,
        k_pe_cache,
        kv_page_indices,  # The page indices of the paged kv-cache, shape: [kv_indptr[-1]].
        kv_page_indptr,  # The indptr of the paged kv-cache, shape: [batch_size + 1].
        # The number of entries in the last page of each request in the paged kv cache,
        # shape: [batch_size].
        kv_last_page_len,
    )

    # https://sourcegraph.com/github.com/flashinfer-ai/flashinfer/-/blob/tests/test_hopper.py?L130-132
    backend = "auto"
    use_cuda_graph = False
    causal = True
    workspace_buffer = get_workspace_buffer(device=k_pe.device)

    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        workspace_buffer,
        backend=backend,
        use_cuda_graph=use_cuda_graph,
    )

    head_dim_kpe = k_pe_cache.shape[-1]
    wrapper.plan(
        q_indptr,
        kv_page_indptr,
        kv_page_indices,
        kv_lens,
        num_heads,
        head_dim_ckv,
        head_dim_kpe,
        page_size.item(),
        causal,
        sm_scale=softmax_scale,
        q_data_type=q_nope_absorbed.dtype,
        kv_data_type=ckv_cache.dtype,
    )

    q_nope_absorbed = q_nope_absorbed.reshape(-1, num_heads, head_dim_ckv)
    q_pe_absorbed = q_pe.reshape(-1, num_heads, head_dim_kpe)
    q_pe = None
    o = wrapper.run(
        q_nope_absorbed, q_pe_absorbed, ckv_cache, k_pe_cache, return_lse=False
    )  # [num_tokens, num_heads, kv_lora_rank]
    assert not torch.isnan(o).any()  # wrong shapes will yield nans

    if False:
        # o ~ [bsz * q_len, num_heads, kv_lora_rank]
        # w_uv ~ [num_heads, v_head_dim, kv_lora_rank]
        # Weight absorption: W^UV_O
        attn_output = torch.einsum("bhl,hdl->bhd", o, w_uv)  # [bsz * q_len, num_heads, v_head_dim]
        attn_output = attn_output.reshape(bsz, q_len, num_heads * v_head_dim)
        # wo_proj ~ [hidden_size, num_heads * v_head_dim]
        attn_output = torch.einsum("bsd,td->bst", attn_output, wo_proj)  # [bsz, q_len, hidden_size]
        if original_bsz != 1:
            attn_output = attn_output.transpose(0, 1)
    else:
        if w_uv_o is None:
            w_uv_o = compute_w_uv_o(w_uv, wo_proj, kv_lora_rank, num_heads, v_head_dim)
        o = o.reshape(-1, num_heads * kv_lora_rank)
        attn_output = torch.einsum("tx,xh->th", o, w_uv_o)  # [bsz, q_len, hidden_size]
        attn_output = attn_output.reshape(bsz, q_len, -1)
        if original_bsz != 1:
            attn_output = attn_output.transpose(0, 1)
    return attn_output


@flashinfer_deepseek_mla_with_kv_cache.register_fake
def flashinfer_deepseek_mla_with_kv_cache_fake(
    q_normed_dn: torch.Tensor,  # [bsz, q_len, self.q_lora_rank]
    compressed_kv: torch.Tensor,  # [bsz, kv_len, kv_lora_rank]
    k_pe: torch.Tensor,  # [bsz, 1, kv_len, qk_rope_head_dim]
    sin_cache: torch.Tensor,
    cos_cache: torch.Tensor,
    wkv_b: torch.Tensor,  # [num_heads * (qk_nope_head_dim+v_head_dim), kv_lora_rank]
    wq_b: torch.Tensor,  # [num_heads * q_head_dim, q_lora_rank]
    w_uq_uk: torch.Tensor,  # Absorbed weights (W_UQ * W_UK)
    wo_proj: torch.Tensor,  # [hidden_size, num_heads * v_head_dim]
    w_uv_o: torch.Tensor,  # Absorbed weights (W^UV * W_O)
    # METADATA
    q_indptr: torch.Tensor,
    kv_page_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    kv_lens: torch.Tensor,
    page_size: torch.Tensor,
    position_ids: torch.Tensor,
    # CACHES
    ckv_cache: torch.Tensor,  # [page_num, page_size, ckv_dim]
    k_pe_cache: torch.Tensor,  # [page_num, page_size, k_pe_dim]
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


#############################################################################################
# Unused reference


@torch.library.custom_op("auto_deploy::flashinfer_deepseek_mla_no_cache", mutates_args=())
def flashinfer_deepseek_mla_no_cache(
    q_normed_dn: torch.Tensor,  # [bsz, q_len, self.q_lora_rank]
    compressed_kv: torch.Tensor,  # [bsz, kv_len, 512]
    k_pe: torch.Tensor,  # [bsz, 1, kv_len, 64]
    sin_cache: torch.Tensor,
    cos_cache: torch.Tensor,
    wkv_b: torch.Tensor,  # [128 * 256, 512]
    wq_b: torch.Tensor,  # [self.num_heads * self.q_head_dim, self.q_lora_rank]
    w_uq_ukv: torch.Tensor,  # Absorbed weights (W_UQ * W_UK)
    wo_proj: torch.Tensor,  # [hidden_size, num_heads * v_head_dim]
    w_uv_o: torch.Tensor,  # Absorbed weights (W^UV * W_O)
    position_ids: torch.Tensor,
    # CONSTANTS
    softmax_scale: float,
    attention_mask: torch.Tensor = None,
) -> torch.Tensor:
    """MHA mode for MLA (i.e. no weight absorption)

    * No cache
    * Serves both prefill and decode
    * Requests are either all decode or mixed prefill+decode.
    """

    v_head_dim = 128
    num_heads = 128
    qk_nope_head_dim = 128
    bsz, q_len, q_lora_rank = q_normed_dn.shape
    ckv_bsz, kv_seq_len, head_dim_ckv = compressed_kv.shape
    assert ckv_bsz == bsz
    assert kv_seq_len == q_len
    assert sin_cache is not None
    assert cos_cache is not None

    # q_normed_dn * (W^UQ and W^QR) (i.e. q up projection)
    wq_b = wq_b.reshape(num_heads, -1, q_lora_rank)  # [num_heads, q_head_dim, q_lora_rank]
    q = torch.einsum("bsl,hdl->bhsd", q_normed_dn, wq_b)  # [bsz, 128, q_len, 192]
    q_head_dim = q.shape[-1]  # 192
    qk_rope_head_dim = q_head_dim - qk_nope_head_dim
    assert qk_rope_head_dim == 64

    # Separate q into the no-positional encoding part (q_nope) and the positional encoding part (q_pe)
    q_nope, q_pe = torch.split(
        q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1
    )  # q_nope ~ [bsz, 128, q_len, 128], q_pe ~ [bsz, 128, q_len, 64]

    # kv = c_K * W^UK (i.e. upward projection)
    kv = (
        torch.einsum("bsc,xc->bsx", compressed_kv, wkv_b)  # [bsz, q_len, 128, 512]
        .view(bsz, q_len, num_heads, qk_nope_head_dim + v_head_dim)
        .transpose(1, 2)
    )

    k_nope, value_states = torch.split(kv, [qk_nope_head_dim, v_head_dim], dim=-1)

    q_pe, k_pe = apply_rotary_pos_emb(
        q_pe, k_pe, cos_cache[:kv_seq_len], sin_cache[:kv_seq_len], position_ids
    )

    query_states = k_pe.new_empty(bsz, num_heads, q_len, q_head_dim)
    query_states[:, :, :, :qk_nope_head_dim] = q_nope
    query_states[:, :, :, qk_nope_head_dim:] = q_pe

    key_states = k_pe.new_empty(bsz, num_heads, q_len, q_head_dim)
    key_states[:, :, :, :qk_nope_head_dim] = k_nope
    key_states[:, :, :, qk_nope_head_dim:] = k_pe

    # https://sourcegraph.com/github.com/flashinfer-ai/flashinfer/-/blob/tests/test_hopper.py?L130-132
    backend = "auto"
    workspace_buffer = get_workspace_buffer(query_states.device)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer,
        backend=backend,
        kv_layout="NHD",
        use_cuda_graph=False,
    )
    qo_indptr = torch.arange(0, bsz * q_len + 1, q_len).int().to(query_states.device)
    kv_indptr = torch.arange(0, bsz * q_len + 1, q_len).int().to(query_states.device)
    head_dim_qk = 192
    head_dim_vo = 128
    causal = True

    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_heads,
        num_heads,
        head_dim_qk,
        causal=causal,
        head_dim_vo=head_dim_vo,
        q_data_type=query_states.dtype,
        kv_data_type=value_states.dtype,
        sm_scale=softmax_scale,
    )
    query_states = query_states.transpose(1, 2).contiguous()
    key_states = key_states.transpose(1, 2).contiguous()
    value_states = value_states.transpose(1, 2).contiguous()
    query_states = query_states.reshape(-1, num_heads, head_dim_qk)
    key_states = key_states.reshape(-1, num_heads, head_dim_qk)
    value_states = value_states.reshape(-1, num_heads, head_dim_vo)

    attn_output = wrapper.run(query_states, key_states, value_states, return_lse=False)
    assert not torch.isnan(attn_output).any()

    if attn_output.size() != (bsz * q_len, num_heads, v_head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz * q_len, num_heads, v_head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.reshape(bsz, q_len, num_heads * v_head_dim)
    attn_output = torch.einsum("bsd,td->bst", attn_output, wo_proj)  # wo_proj(attn_output)
    return attn_output


@flashinfer_deepseek_mla_no_cache.register_fake
def flashinfer_deepseek_mla_no_cache_fake(
    q_normed_dn: torch.Tensor,  # [bsz, q_len, self.q_lora_rank]
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    sin_cache: torch.Tensor,
    cos_cache: torch.Tensor,
    wkv_b: torch.Tensor,
    wq_b: torch.Tensor,
    w_uq_ukv: torch.Tensor,  # Absorbed weights (W_UQ * W_UK)
    wo_proj: torch.Tensor,  # [hidden_size, num_heads * v_head_dim]
    w_uv_o: torch.Tensor,  # Absorbed weights (W^UV * W_O)
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

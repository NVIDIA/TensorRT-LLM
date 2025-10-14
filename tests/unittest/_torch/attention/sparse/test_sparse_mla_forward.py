"""
Test suite for deepseek sparse attention with kvcache_dtype=bf16 using FlashMLA backend.
"""
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, cast

import pytest
import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.attention_backend.interface import (
    PositionalEmbeddingParams, RopeParams)
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.modules.attention import MLA
from tensorrt_llm._torch.attention_backend.sparse.dsa import DSACacheManager
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._utils import str_dtype_to_binding, torch_dtype_to_str
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.functional import PositionEmbeddingType, RopeEmbeddingUtils
from tensorrt_llm.mapping import Mapping
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi.llm_args import DSASparseAttentionConfig


try:
    from tensorrt_llm.flash_mla import flash_mla_sparse_fwd
    HAS_FLASH_MLA = True
except ImportError:
    HAS_FLASH_MLA = False


@dataclass
class BatchSpec:
    """Batch specification for testing, following vLLM's pattern."""
    seq_lens: List[int]
    query_lens: List[int]

    @property
    def batch_size(self):
        return len(self.seq_lens)


# Unified batch specifications covering all scenarios
# seq_lens = total KV cache length (for decode/mixed) or current sequence length (for pure prefill)
# query_lens = number of new query tokens per request
BATCH_SPECS = {
    # Pure prefill scenarios (query_lens == seq_lens)
    "small_prefill": BatchSpec(
        seq_lens=[32, 48, 24],
        query_lens=[32, 48, 24]
    ),
    "medium_prefill": BatchSpec(
        seq_lens=[128, 256, 64],
        query_lens=[128, 256, 64]
    ),
    "single_prefill": BatchSpec(
        seq_lens=[512],
        query_lens=[512]
    ),

    # Pure decode/generation scenarios (query_lens == 1)
    # seq_lens includes the cached context + 1 new token
    "small_decode": BatchSpec(
        seq_lens=[33, 49, 25],  # Previous context + 1 new token
        query_lens=[1, 1, 1]
    ),
    "medium_decode": BatchSpec(
        seq_lens=[129, 257, 65],
        query_lens=[1, 1, 1]
    ),
    "single_decode": BatchSpec(
        seq_lens=[513],
        query_lens=[1]
    ),

    # Mixed scenarios (some prefill, some decode)
    # Format: [prefill_req1, prefill_req2, decode_req1, decode_req2, ...]
    "small_mixed": BatchSpec(
        seq_lens=[32, 48, 25, 33],  # First 2 are prefill, last 2 are decode (cached: 24, 32)
        query_lens=[32, 48, 1, 1]   # First 2 process full seq, last 2 generate 1 token
    ),
    "medium_mixed": BatchSpec(
        seq_lens=[128, 65, 257],    # 1 prefill, 1 decode, 1 decode
        query_lens=[128, 1, 1]
    ),
    "mixed_heavy_prefill": BatchSpec(
        seq_lens=[256, 128, 64, 33, 49],  # 3 prefill, 2 decode
        query_lens=[256, 128, 64, 1, 1]
    ),
    "mixed_heavy_decode": BatchSpec(
        seq_lens=[32, 33, 49, 65, 129],   # 1 prefill, 4 decode
        query_lens=[32, 1, 1, 1, 1]
    ),
}

def apply_rotary_embedding(
    x: torch.Tensor,  # [seq_len, ..., rope_dim]
    cos_sin: torch.Tensor,  # [seq_len, 2, rope_dim]
) -> torch.Tensor:
    """
    Apply rotary position embedding.
    """
    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    original_dtype = x.dtype
    cos, sin = cos_sin.chunk(2, dim=-2)
    cos = cos.squeeze(1)
    sin = sin.squeeze(1)

    # Convert to interleaved format
    x_interleaved = x.unflatten(-1, [-1, 2]).transpose(-2, -1).flatten(start_dim=-2)
    cos_expanded = cos.view(cos.shape[0], *([1] * (x.ndim - 2)), cos.shape[-1])
    sin_expanded = sin.view(sin.shape[0], *([1] * (x.ndim - 2)), sin.shape[-1])
    # Apply rotation
    x_rotated = (x_interleaved * cos_expanded) + (rotate_half(x_interleaved) * sin_expanded)

    # Convert back to original dtype
    return x_rotated.to(original_dtype)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value tensors n_rep times along the KV head dimension.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen,
                                 head_dim)

@dataclass
class RopeConfig:
    """Configuration for RoPE."""
    hidden_size: int
    num_attention_heads: int
    rope_scaling: dict
    max_position_embeddings: int
    rope_theta: float
    qk_rope_head_dim: int
    model_type: str

def calculate_reference_output_prefill_only(
    q_c, kv_c, k_pe, W_UK, W_UV, rope_cos_sin, sequence_lengths,
    num_heads, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim,
    v_head_dim, softmax_scale, device
):
    """Reference for pure prefill (unrotated inputs, applies RoPE internally)."""
    results, offset = [], 0
    for seq_len in sequence_lengths:
        q_seq = q_c[offset:offset+seq_len]
        kv_seq = kv_c[offset:offset+seq_len]
        k_pe_seq = k_pe[offset:offset+seq_len]

        q_nope, q_pe = q_seq.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
        cos_sin = rope_cos_sin[:seq_len]
        q_pe = apply_rotary_embedding(q_pe, cos_sin)
        k_pe_rot = apply_rotary_embedding(k_pe_seq, cos_sin)

        ql_nope = torch.einsum("qnh,lnh->qnl", q_nope, W_UK)
        q_mqa = torch.cat([ql_nope, q_pe], dim=-1)
        k_mqa = torch.cat([kv_seq.unsqueeze(1).expand(-1, num_heads, -1),
                          k_pe_rot.unsqueeze(1).expand(-1, num_heads, -1)], dim=-1)
        v_mqa = kv_seq.unsqueeze(1).expand(-1, num_heads, -1)

        attn_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        q_in = q_mqa.unsqueeze(0).transpose(1, 2)
        k_in = k_mqa.unsqueeze(0).transpose(1, 2)
        v_in = v_mqa.unsqueeze(0).transpose(1, 2)

        out = torch.nn.functional.scaled_dot_product_attention(
            q_in, k_in, v_in, attn_mask=attn_mask, scale=softmax_scale
        ).transpose(1, 2).squeeze(0)

        results.append(torch.einsum("qnl,lnv->qnv", out, W_UV).flatten(start_dim=-2))
        offset += seq_len
    return torch.cat(results, dim=0)


def calculate_reference_output_generation(
    q_c, kv_c, k_pe, W_UK, W_UV, kv_cache_lens,
    num_heads, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim,
    v_head_dim, softmax_scale, device
):
    """Reference for generation (rotated inputs, no RoPE application)."""
    results, kv_offset = [], 0
    for kv_len in kv_cache_lens:
        q_seq = q_c[len(results):len(results)+1]  # [1, num_heads, qk_head_dim]
        kv_seq = kv_c[kv_offset:kv_offset+kv_len]
        k_pe_seq = k_pe[kv_offset:kv_offset+kv_len]

        q_nope, q_pe = q_seq.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
        ql_nope = torch.einsum("qnh,lnh->qnl", q_nope, W_UK)
        q_mqa = torch.cat([ql_nope, q_pe], dim=-1)

        k_mqa = torch.cat([kv_seq.unsqueeze(1).expand(-1, num_heads, -1),
                          k_pe_seq.unsqueeze(1).expand(-1, num_heads, -1)], dim=-1)
        v_mqa = kv_seq.unsqueeze(1).expand(-1, num_heads, -1)

        q_in = q_mqa.unsqueeze(0).transpose(1, 2)
        k_in = k_mqa.unsqueeze(0).transpose(1, 2)
        v_in = v_mqa.unsqueeze(0).transpose(1, 2)

        out = torch.nn.functional.scaled_dot_product_attention(
            q_in, k_in, v_in, attn_mask=None, scale=softmax_scale
        ).transpose(1, 2).squeeze(0)

        results.append(torch.einsum("qnl,lnv->qnv", out, W_UV).flatten(start_dim=-2))
        kv_offset += kv_len
    return torch.cat(results, dim=0)


def calculate_reference_output_mixed(
    q_ctx, q_gen, kv_c_all, k_pe_all, W_UK, W_UV, rope_cos_sin,
    ctx_indices, gen_indices, seq_lens, query_lens,
    num_heads, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim,
    v_head_dim, softmax_scale, device
):
    """Reference for mixed batch (combines context and generation)."""
    ref_results = [None] * len(seq_lens)

    # Extract KV slices for context and generation (in [context...][generation...] layout)
    def extract_kv_slices(indices, start_offset=0):
        slices_kv, slices_kpe = [], []
        offset = start_offset
        for req_idx in indices:
            seq_len = seq_lens[req_idx]
            slices_kv.append(kv_c_all[offset:offset+seq_len])
            slices_kpe.append(k_pe_all[offset:offset+seq_len])
            offset += seq_len
        return torch.cat(slices_kv) if slices_kv else None, \
               torch.cat(slices_kpe) if slices_kpe else None

    # Process context requests (unrotated, apply RoPE)
    if ctx_indices:
        kv_c, k_pe = extract_kv_slices(ctx_indices, 0)
        ctx_results = calculate_reference_output_prefill_only(
            q_ctx, kv_c, k_pe, W_UK, W_UV, rope_cos_sin,
            [seq_lens[i] for i in ctx_indices],
            num_heads, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim,
            v_head_dim, softmax_scale, device
        )
        offset = 0
        for req_idx in ctx_indices:
            seq_len = seq_lens[req_idx]
            ref_results[req_idx] = ctx_results[offset:offset+seq_len]
            offset += seq_len

    # Process generation requests (rotated, no RoPE)
    if gen_indices:
        kv_c, k_pe = extract_kv_slices(gen_indices, sum(seq_lens[i] for i in ctx_indices))
        gen_results = calculate_reference_output_generation(
            q_gen, kv_c, k_pe, W_UK, W_UV, [seq_lens[i] for i in gen_indices],
            num_heads, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim,
            v_head_dim, softmax_scale, device
        )
        for idx, req_idx in enumerate(gen_indices):
            ref_results[req_idx] = gen_results[idx:idx+1]

    return torch.cat(ref_results, dim=0)


@pytest.mark.skipif(not HAS_FLASH_MLA, reason="FlashMLA not available")
@pytest.mark.skipif(get_sm_version() < 90, reason="FlashMLA requires SM90 (Hopper) or later")
@pytest.mark.parametrize("batch_name", list(BATCH_SPECS.keys()))
@pytest.mark.parametrize("kv_cache_dtype", ["auto"])  # TODO: Add "fp8" support
def test_forward_sparse_mla_unified(batch_name, kv_cache_dtype):
    """Test sparse MLA attention for pure prefill, pure decode, and mixed batches."""
    print(f"\n{'='*80}\nTesting: {batch_name}\n{'='*80}")

    device = torch.device('cuda')
    dtype = torch.bfloat16

    batch_spec = BATCH_SPECS[batch_name]
    seq_lens = batch_spec.seq_lens
    query_lens = batch_spec.query_lens

    # Identify context (query_len==seq_len) vs generation (query_len<seq_len) requests
    ctx_indices = [i for i, (q, s) in enumerate(zip(query_lens, seq_lens)) if q == s]
    gen_indices = [i for i, (q, s) in enumerate(zip(query_lens, seq_lens)) if q < s]
    num_contexts, num_generations = len(ctx_indices), len(gen_indices)

    print(f"Requests: {len(seq_lens)} total ({num_contexts} ctx, {num_generations} gen)")

    # Model configuration
    # Since topk=2048 > seq_lens, indexer selects all tokens anyway
    num_heads = 128
    q_lora_rank = 512
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    head_size = kv_lora_rank + qk_rope_head_dim  # 576
    topk_tokens = 2048
    hidden_size = 2048
    max_position_embeddings = 4096
    tokens_per_block = 64
    num_layers = 1
    layer_idx = 0

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Create RoPE config
    rope_config = RopeConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        rope_scaling={
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "yarn",
        },
        max_position_embeddings=max_position_embeddings,
        rope_theta=10000.0,
        qk_rope_head_dim=qk_rope_head_dim,
        model_type="deepseek_v2",
    )

    # Generate RoPE cos/sin embeddings for reference calculation
    rope_cos_sin = torch.tensor(
        RopeEmbeddingUtils.create_sinusoidal_positions_yarn(
            rope_config.max_position_embeddings,
            rope_config.qk_rope_head_dim,
            rope_config.rope_theta,
            rope_config.rope_scaling['factor'],
            rope_config.rope_scaling['original_max_position_embeddings'],
            rope_config.rope_scaling['beta_fast'],
            rope_config.rope_scaling['beta_slow'],
            rope_config.rope_scaling['mscale'],
            rope_config.rope_scaling['mscale_all_dim'],
        )[1],
        dtype=torch.float32,
        device=device,
    ).reshape(rope_config.max_position_embeddings, -1, 2).transpose(-2, -1)

    # Calculate scaling factors (aligned with vLLM)
    def yarn_get_mscale(scale=1, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    mscale_all_dim = rope_config.rope_scaling['mscale_all_dim']
    scaling_factor = rope_config.rope_scaling['factor']
    mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
    q_scaling = 1.0 / (mscale * mscale)

    # Softmax scale for sparse attention computation
    softmax_scale = 1.0 / (math.sqrt(qk_head_dim) * q_scaling)

    # Setup model config
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    max_seqlen = max(seq_lens)
    total_cache_tokens = sum(seq_lens)
    max_tokens = 16384

    # Create sparse attention config (DSA - DeepSeek Sparse Attention)
    sparse_config = DSASparseAttentionConfig(
        index_n_heads=64,  # Number of heads for indexer
        index_head_dim=128,  # Dimension of indexer heads
        index_topk=topk_tokens,  # Top-k tokens to select (2048)
    )

    if kv_cache_dtype == "auto":
        cache_dtype = dtype
    elif kv_cache_dtype == "fp8":
        # TODO: Implement FP8 quantization for MLA cache
        pytest.skip("fp8 cache dtype not yet supported with FlashMLA backend")
        cache_dtype = torch.float8_e4m3fn
    else:
        cache_dtype = dtype

    # Create pretrained_config for ModelConfig
    pretrained_config = SimpleNamespace(
        rms_norm_eps=1e-6,
    )

    model_config = ModelConfig(
        mapping=mapping,
        sparse_attention_config=sparse_config,
        pretrained_config=pretrained_config,
    )

    # Setup positional embedding params
    pos_embd_params = PositionalEmbeddingParams(
        type=PositionEmbeddingType.yarn,
        rope=RopeParams.from_config(rope_config),
        is_neox=False,
    )

    # Create MLA module first
    mla = MLA(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=1,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        predicted_tokens_per_seq=1,
        max_position_embeddings=max_position_embeddings,
        bias=False,
        pos_embd_params=pos_embd_params,
        layer_idx=layer_idx,
        dtype=dtype,
        config=model_config,
    ).to(device)

    # Initialize weights
    with torch.no_grad():
        # Initialize kv_b_proj weight
        nn_init_std = 0.02
        mla.kv_b_proj.weight.normal_(mean=0.0, std=nn_init_std)

        # Extract W_UK and W_UV for reference calculation
        kv_b_weight = mla.kv_b_proj.weight.data  # [num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
        kv_b_weight_reshaped = kv_b_weight.view(num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)
        W_UK = kv_b_weight_reshaped[:, :qk_nope_head_dim, :].permute(2, 0, 1).contiguous()  # [kv_lora_rank, num_heads, qk_nope_head_dim]
        W_UV = kv_b_weight_reshaped[:, qk_nope_head_dim:, :].permute(2, 0, 1).contiguous()  # [kv_lora_rank, num_heads, v_head_dim]

        # v_b_proj: [num_heads, v_head_dim, kv_lora_rank]
        mla.v_b_proj.data = kv_b_weight_reshaped[:, qk_nope_head_dim:, :].contiguous()

        # k_b_proj_trans: [num_heads, kv_lora_rank, qk_nope_head_dim]
        mla.k_b_proj_trans.data = kv_b_weight_reshaped[:, :qk_nope_head_dim, :].transpose(1, 2).contiguous()

        # Initialize indexer weights
        mla.mqa.indexer.wq_b.weight.normal_(mean=0.0, std=nn_init_std)
        mla.mqa.indexer.wk.weight.normal_(mean=0.0, std=nn_init_std)
        mla.mqa.indexer.weights_proj.weight.normal_(mean=0.0, std=nn_init_std)

    # Calculate cached token counts (context already in cache)
    cached_lens = [seq_lens[i] - query_lens[i] for i in range(len(seq_lens))]

    # Create KV cache manager
    kv_cache_manager = DSACacheManager(
        KvCacheConfig(max_tokens=max_tokens, enable_block_reuse=False),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=head_size,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seqlen,
        max_batch_size=batch_spec.batch_size,
        mapping=mapping,
        dtype=str_dtype_to_binding(torch_dtype_to_str(cache_dtype)),
        sparse_attn_config=sparse_config,
        model_config=model_config,
    )

    request_ids = list(range(batch_spec.batch_size))
    AttentionCls = get_attention_backend("TRTLLM", sparse_config)

    # Allocate and pre-populate KV cache in batch order [context...][generation...]
    all_cached_compressed_kv = {}
    all_cached_k_pe_rotated = {}

    print(f"  Allocating and pre-populating cache...")

    # Allocate context requests first
    for req_idx in ctx_indices:
        assert cached_lens[req_idx] == 0, f"Context request {req_idx} should have no cached tokens"
        kv_cache_manager.add_dummy_requests(
            request_ids=[req_idx],
            token_nums=[seq_lens[req_idx]],
            is_gen=False,
            prepare_resource=True,
        )
        print(f"    - Allocated cache for ctx request {req_idx}: {seq_lens[req_idx]} tokens")

    # Allocate and pre-populate generation requests (batched)
    if gen_indices:
        gen_cached_lens = [cached_lens[i] for i in gen_indices if cached_lens[i] > 0]
        gen_with_cache = [i for i in gen_indices if cached_lens[i] > 0]
        # Allocate all generation caches
        for req_idx in gen_with_cache:
            kv_cache_manager.add_dummy_requests(
                request_ids=[req_idx],
                token_nums=[cached_lens[req_idx]],
                is_gen=False,
                prepare_resource=True,
            )
            _allocate_kv_cache_for_generation(kv_cache_manager, [req_idx])

        # Generate batched cache data
        total_gen_cache_tokens = sum(gen_cached_lens)
        batched_latent = torch.empty(total_gen_cache_tokens, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)

        offset = 0
        for req_idx in gen_with_cache:
            cached_len = cached_lens[req_idx]
            all_cached_compressed_kv[req_idx] = torch.randn(cached_len, kv_lora_rank, dtype=dtype, device=device)
            cached_k_pe = torch.randn(cached_len, qk_rope_head_dim, dtype=dtype, device=device)
            batched_latent[offset:offset+cached_len] = torch.cat([all_cached_compressed_kv[req_idx], cached_k_pe], dim=-1)
            offset += cached_len

        # Single batched metadata for all generation cache population
        cached_metadata = AttentionCls.Metadata(
            seq_lens=torch.tensor(gen_cached_lens, dtype=torch.int),
            request_ids=gen_with_cache,
            max_num_requests=len(gen_with_cache),
            num_contexts=len(gen_with_cache),
            prompt_lens=gen_cached_lens,
            max_num_tokens=total_gen_cache_tokens,
            kv_cache_manager=kv_cache_manager,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0] * len(gen_with_cache)),
            mapping=mapping,
        )
        cached_metadata.prepare()

        dummy_q = torch.randn(total_gen_cache_tokens, num_heads * qk_head_dim, dtype=dtype, device=device)
        mla.mqa.mla_rope_append_paged_kv_assign_q(dummy_q, batched_latent, cached_metadata)

        # Extract rotated k_pe for each request
        offset = 0
        for req_idx in gen_with_cache:
            cached_len = cached_lens[req_idx]
            all_cached_k_pe_rotated[req_idx] = batched_latent[offset:offset+cached_len, kv_lora_rank:].clone()
            offset += cached_len
            print(f"    - Allocated+populated cache for gen request {req_idx}: {cached_len} cached + 1 new = {seq_lens[req_idx]} tokens")

    print(f"  ✓ KV cache allocated and pre-populated")

    # Generate inputs directly in batch order [context...][generation...]
    batch_order = ctx_indices + gen_indices
    total_query_tokens = sum(query_lens)
    batch_query_lens = [query_lens[i] for i in batch_order]

    q = torch.randn(total_query_tokens, num_heads * qk_head_dim, dtype=dtype, device=device)
    compressed_kv = torch.randn(total_query_tokens, kv_lora_rank, dtype=dtype, device=device)
    k_pe = torch.randn(total_query_tokens, qk_rope_head_dim, dtype=dtype, device=device)
    k_pe_original_for_ref = k_pe.clone()  # Save before kernel modifies it
    hidden_states = torch.randn(total_query_tokens, hidden_size, dtype=dtype, device=device)
    qr = torch.randn(total_query_tokens, q_lora_rank, dtype=dtype, device=device)

    latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)
    position_ids = torch.cat([torch.arange(cached_lens[i], cached_lens[i] + query_lens[i],
                                          device=device, dtype=torch.int32)
                             for i in batch_order])
    output = torch.empty(total_query_tokens, num_heads * v_head_dim, dtype=dtype, device=device)

    attn_metadata = AttentionCls.Metadata(
        seq_lens=torch.tensor(batch_query_lens, dtype=torch.int),
        request_ids=batch_order,
        max_num_requests=batch_spec.batch_size,
        num_contexts=num_contexts,
        prompt_lens=[seq_lens[i] if i in ctx_indices else cached_lens[i] for i in batch_order],
        max_num_tokens=total_query_tokens,
        kv_cache_manager=kv_cache_manager,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[cached_lens[i] for i in batch_order],
        ),
        mapping=mapping,
    )
    attn_metadata.prepare()

    assert hasattr(mla, 'softmax_scale') and abs(mla.softmax_scale - softmax_scale) < 1e-6
    print(f"  ✓ Inputs prepared: {total_query_tokens} query tokens")

    q_original_for_ref = q.clone()
    batch_token_offsets = [0] + [sum(batch_query_lens[:i+1]) for i in range(len(batch_order))]
    num_ctx_tokens = sum(query_lens[i] for i in ctx_indices)

    def create_causal_indices(req_indices, cache_offset_start=0):
        """Helper to create causal attention indices with padding."""
        indices = []
        kv_offset = cache_offset_start
        for req_idx in req_indices:
            for local_pos in range(query_lens[req_idx]):
                num_attend = min(cached_lens[req_idx] + local_pos + 1, topk_tokens)
                attend_indices = torch.arange(num_attend, dtype=torch.int32, device=device) + kv_offset
                if num_attend < topk_tokens:
                    padding = torch.full((topk_tokens - num_attend,), -1, dtype=torch.int32, device=device)
                    attend_indices = torch.cat([attend_indices, padding])
                indices.append(attend_indices)
            kv_offset += seq_lens[req_idx]
        return torch.stack(indices, dim=0)

    def local_to_global_indices(local_indices, req_indices, cache_offset_start=0):
        """
        Transform indexer's local indices to global indices.
        """
        global_indices = local_indices.clone()
        kv_offset = cache_offset_start
        token_idx = 0

        for req_idx in req_indices:
            num_tokens = query_lens[req_idx]
            # Add offset for this request's cache position
            for local_pos in range(num_tokens):
                # Only transform non-padding indices (>= 0)
                mask = global_indices[token_idx] >= 0
                global_indices[token_idx][mask] += kv_offset
                token_idx += 1
            kv_offset += seq_lens[req_idx]
        return global_indices

    topk_indices_local = mla.mqa.indexer(qr, hidden_states, attn_metadata, position_ids)

    # Validate indexer output against expected causal indices (since seq_len < topk=2048)
    if num_contexts > 0:
        # Transform context indices from local to global
        ctx_topk_local = topk_indices_local[:num_ctx_tokens]
        ctx_topk_global = local_to_global_indices(ctx_topk_local, ctx_indices, cache_offset_start=0)

         # Create expected global indices (sorted) for validation (not used but can be used for validation)
        expected_ctx_indices = create_causal_indices(ctx_indices, cache_offset_start=0)

        mla.forward_context_dsa(
            q=q[:num_ctx_tokens],
            compressed_kv=compressed_kv[:num_ctx_tokens],
            k_pe=k_pe[:num_ctx_tokens],
            attn_metadata=attn_metadata,
            output=output[:num_ctx_tokens],
            latent_cache=latent_cache[:num_ctx_tokens],
            topk_indices=ctx_topk_local,  # Use global indices
        )
        print(f"  ✓ Context forward: {num_ctx_tokens} tokens from {num_contexts} requests")

    if num_generations > 0:
        # Transform generation indices from local to global
        num_gen_tokens = sum(query_lens[i] for i in gen_indices)
        gen_topk_local = topk_indices_local[num_ctx_tokens:num_ctx_tokens + num_gen_tokens]
        gen_topk_global = local_to_global_indices(gen_topk_local, gen_indices, cache_offset_start=0)

        # Create expected global indices (sorted) for validation (not used but can be used for validation)
        expected_gen_indices = create_causal_indices(gen_indices, cache_offset_start=0)

        mla.forward_generation_dsa(
            q=q[num_ctx_tokens:],
            compressed_kv=compressed_kv[num_ctx_tokens:],
            k_pe=k_pe[num_ctx_tokens:],
            attn_metadata=attn_metadata,
            output=output[num_ctx_tokens:],
            latent_cache=latent_cache[num_ctx_tokens:],
            topk_indices=gen_topk_local,  # Use global indices
        )
        print(f"  ✓ Generation forward: {sum(query_lens[i] for i in gen_indices)} tokens from {num_generations} requests")

    print(f"  ✓ Forward pass complete: output shape {output.shape}")

    # Assemble reference in BATCH order (same as output)
    q_for_ref_list, kv_c_list, k_pe_list = [], [], []

    for batch_idx, orig_req_idx in enumerate(batch_order):
        batch_start = batch_token_offsets[batch_idx]
        batch_end = batch_token_offsets[batch_idx + 1]

        if orig_req_idx in ctx_indices:
            # Context: use unrotated q and k_pe from latent_cache
            q_req = q_original_for_ref[batch_start:batch_end]
            kv_c_list.append(latent_cache[batch_start:batch_end, :kv_lora_rank])
            k_pe_list.append(k_pe_original_for_ref[batch_start:batch_end])
        else:
            # Generation: use rotated q, combine cached + new KV from latent_cache
            q_req = q[batch_start:batch_end]
            cached_len = cached_lens[orig_req_idx]
            kv_c_list.append(torch.cat([all_cached_compressed_kv[orig_req_idx][:cached_len],
                                        latent_cache[batch_start:batch_end, :kv_lora_rank]]))
            k_pe_list.append(torch.cat([all_cached_k_pe_rotated[orig_req_idx][:cached_len],
                                        latent_cache[batch_start:batch_end, kv_lora_rank:]]))

        q_for_ref_list.append(q_req.view(-1, num_heads, qk_head_dim))

    q_for_ref = torch.cat(q_for_ref_list, dim=0)
    all_compressed_kv = torch.cat(kv_c_list, dim=0)
    all_k_pe_for_ref = torch.cat(k_pe_list, dim=0)

    print(f"  - Computing reference ({num_contexts} ctx, {num_generations} gen)...")
    q_ctx_ref = q_for_ref[:num_ctx_tokens] if ctx_indices else torch.empty(0, num_heads, qk_head_dim, device=device)
    q_gen_ref = q_for_ref[num_ctx_tokens:] if gen_indices else torch.empty(0, num_heads, qk_head_dim, device=device)

    reference_output = calculate_reference_output_mixed(
        q_ctx=q_ctx_ref,
        q_gen=q_gen_ref,
        kv_c_all=all_compressed_kv,
        k_pe_all=all_k_pe_for_ref,
        W_UK=W_UK,
        W_UV=W_UV,
        rope_cos_sin=rope_cos_sin,
        ctx_indices=list(range(num_contexts)),
        gen_indices=list(range(num_contexts, len(batch_order))),
        seq_lens=[seq_lens[i] for i in batch_order],
        query_lens=batch_query_lens,
        num_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        softmax_scale=softmax_scale,
        device=device,
    )

    assert output.shape == reference_output.shape and output.dtype == reference_output.dtype
    assert torch.isfinite(output).all() and torch.isfinite(reference_output).all()

    # Compare directly (both in batch order now)
    abs_error = (output - reference_output).abs()
    for batch_idx, orig_req_idx in enumerate(batch_order):
        req_error = abs_error[batch_token_offsets[batch_idx]:batch_token_offsets[batch_idx+1]]
        if req_error.max() > 0.1:
            req_type = "CTX" if orig_req_idx in ctx_indices else "GEN"
            print(f"  ⚠ Request {orig_req_idx} [{req_type}]: max error {req_error.max():.3f}")

    torch.testing.assert_close(output, reference_output, rtol=0.1, atol=0.1)
    print(f"  ✓ Validation passed: max_error={abs_error.max():.4f}, mean_error={abs_error.mean():.6f}")

    kv_cache_manager.shutdown()
    print(f"  ✓ Test '{batch_name}' completed\n")


def _allocate_kv_cache_for_generation(kv_cache_manager, request_ids):
    """
    Allocate KV cache blocks for generation phase following PyExecutor's pattern.

    This mimics the production flow: prepare_resources() calls add_token()
    for each generation request, which allocates blocks as needed.

    For DSACacheManager, we need to allocate for both:
    - Main KV cache (via impl.add_token)
    - Indexer K cache (via indexer_k_cache_manager.add_tokens)
    """
    for request_id in request_ids:
        # Allocate main KV cache block (mimics prepare_resources flow)
        kv_cache_manager.impl.add_token(request_id)

        # Allocate indexer K cache block for DSA sparse attention
        if hasattr(kv_cache_manager, 'indexer_k_cache_manager'):
            kv_cache_manager.indexer_k_cache_manager.add_tokens(request_id, 1)


# Old test_forward_sparse_mla_generation removed - unified into test_forward_sparse_mla_unified


if __name__ == "__main__":
    # Test pure prefill
    test_forward_sparse_mla_unified(batch_name="small_prefill", kv_cache_dtype="auto")

    # Test pure decode
    test_forward_sparse_mla_unified(batch_name="small_decode", kv_cache_dtype="auto")

    # TODO: Mixed batch test - generation reference needs sparse attention masking
    test_forward_sparse_mla_unified(batch_name="small_mixed", kv_cache_dtype="auto")

    print("All tests passed!")


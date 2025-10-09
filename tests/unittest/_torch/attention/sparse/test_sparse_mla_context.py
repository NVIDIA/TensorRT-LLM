"""
Test suite for deepseek sparse attention with kvcache_dtype=bf16 using FlashMLA backend.
"""
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List

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


# Batch specifications aligned with vLLM's SPARSE_BACKEND_BATCH_SPECS
BATCH_SPECS = {
    # TODO: Add mixed scenarios with decode and prefill
    # Mixed scenarios with decode and prefill

    # TODO: Add decode scenarios with decode and prefill

    # Pure prefill scenarios
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
}

class MockIndexer(torch.nn.Module):
    """Mock indexer for testing that provides topk_indices_buffer."""

    def __init__(self, topk_indices_buffer: torch.Tensor):
        super().__init__()
        self.topk_indices_buffer = topk_indices_buffer

    def forward(self, *args, **kwargs):
        """Mock forward that does nothing - indices are pre-computed."""
        return self.topk_indices_buffer

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
    q_c: torch.Tensor,  # [total_tokens, num_heads, qk_head_dim]
    kv_c: torch.Tensor,  # [total_tokens, kv_lora_rank]
    k_pe: torch.Tensor,  # [total_tokens, qk_rope_head_dim]
    W_UK: torch.Tensor,  # [kv_lora_rank, num_heads, qk_nope_head_dim]
    W_UV: torch.Tensor,  # [kv_lora_rank, num_heads, v_head_dim]
    rope_cos_sin: torch.Tensor,  # [max_seq_len, 2, qk_rope_head_dim]
    sequence_lengths: List[int],
    num_heads: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    softmax_scale: float,
    device: torch.device,
):
    """
    Calculate reference result using standard PyTorch attention in MQA latent space.

    Args:
        q_c: Q tensor [total_tokens, num_heads, qk_head_dim] where qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        kv_c: Compressed KV latent cache [total_tokens, kv_lora_rank]
        k_pe: Key RoPE embeddings (unrotated) [total_tokens, qk_rope_head_dim]
        W_UK: K projection weight [kv_lora_rank, num_heads, qk_nope_head_dim]
        W_UV: V projection weight [kv_lora_rank, num_heads, v_head_dim]
        rope_cos_sin: RoPE cos/sin values [max_seq_len, 2, qk_rope_head_dim]
        sequence_lengths: List of sequence lengths for each request
        softmax_scale: Scale factor for attention scores
    """
    num_requests = len(sequence_lengths)

    # Process each request separately
    ref_results = []
    total_tokens = 0

    for i in range(num_requests):
        seq_len = sequence_lengths[i]
        query_len = seq_len  # For context phase, query_len == seq_len

        # Extract tensors for this request
        q_seq = q_c[total_tokens:total_tokens + seq_len]  # [seq_len, num_heads, qk_head_dim]
        kv_c_seq = kv_c[total_tokens:total_tokens + seq_len]  # [seq_len, kv_lora_rank]
        k_pe_seq = k_pe[total_tokens:total_tokens + seq_len]  # [seq_len, qk_rope_head_dim]

        q_nope, q_pe = q_seq.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
        # q_nope: [seq_len, num_heads, qk_nope_head_dim]
        # q_pe: [seq_len, num_heads, qk_rope_head_dim] (rope unapplied)

        # Apply RoPE
        cos_sin_seq = rope_cos_sin[:seq_len]  # [seq_len, 2, qk_rope_head_dim]
        q_pe_rotated = apply_rotary_embedding(q_pe, cos_sin_seq)
        k_pe_rotated = apply_rotary_embedding(k_pe_seq, cos_sin_seq)  # [seq_len, qk_rope_head_dim]

        # Up project q_nope
        ql_nope = torch.einsum("qnh,lnh->qnl", q_nope, W_UK)  # [seq_len, num_heads, kv_lora_rank]
        q_mqa = torch.cat([ql_nope, q_pe_rotated], dim=-1)  # [seq_len, num_heads, kv_lora_rank + qk_rope_head_dim]
        # concat k_pe_rotated to kv_c_seq
        k_mqa = torch.cat([kv_c_seq.unsqueeze(1).expand(-1, num_heads, -1),
                          k_pe_rotated.unsqueeze(1).expand(-1, num_heads, -1)], dim=-1)
        # k_mqa: [seq_len, num_heads, kv_lora_rank + qk_rope_head_dim]

        # v_mqa is just the latent cache expanded
        v_mqa = kv_c_seq.unsqueeze(1).expand(-1, num_heads, -1)  # [seq_len, num_heads, kv_lora_rank]

        # Attention mask for context phase
        attn_mask = torch.ones(query_len, seq_len, dtype=torch.bool, device=device)
        causal_mask = torch.tril(torch.ones(query_len, query_len, device=device))
        ctx_len = seq_len - query_len  # For pure prefill, ctx_len = 0
        attn_mask[:, ctx_len:] = causal_mask

        # Prepare inputs for scaled_dot_product_attention
        q_sdpa_in = q_mqa.unsqueeze(0).transpose(1, 2)  # [1, num_heads, seq_len, kv_lora_rank + qk_rope_head_dim]
        k_sdpa_in = k_mqa.unsqueeze(0).transpose(1, 2)  # [1, num_heads, seq_len, kv_lora_rank + qk_rope_head_dim]
        v_sdpa_in = v_mqa.unsqueeze(0).transpose(1, 2)  # [1, num_heads, seq_len, kv_lora_rank]

        # Attention in latent space
        sdpa_out = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa_in, k_sdpa_in, v_sdpa_in,
            attn_mask=attn_mask,
            scale=softmax_scale
        )
        # sdpa_out: [1, num_heads, seq_len, kv_lora_rank]
        sdpa_out = sdpa_out.transpose(1, 2).squeeze(0)  # [seq_len, num_heads, kv_lora_rank]

        # Up project output
        attn_output = torch.einsum("qnl,lnv->qnv", sdpa_out, W_UV)  # [seq_len, num_heads, v_head_dim]
        attn_output = attn_output.flatten(start_dim=-2)  # [seq_len, num_heads * v_head_dim]

        ref_results.append(attn_output)
        total_tokens += seq_len

    return torch.cat(ref_results, dim=0)


@pytest.mark.skipif(not HAS_FLASH_MLA, reason="FlashMLA not available")
@pytest.mark.skipif(get_sm_version() < 90, reason="FlashMLA requires SM90 (Hopper) or later")
@pytest.mark.parametrize("batch_name", list(BATCH_SPECS.keys()))
@pytest.mark.parametrize("kv_cache_dtype", ["auto"])  # TODO: Add "fp8" support
def test_forward_sparse_mla_context(batch_name, kv_cache_dtype):
    """
    Test forward_sparse_mla_kvcache_bf16 for context phase (is_generation=False).

    This test:
    1. Creates an MLA/KVCache module with sparse attention
    2. Computes output using forward_sparse_mla_kvcache_bf16
    3. Computes reference output using standard PyTorch attention
    """
    device = torch.device('cuda')
    dtype = torch.bfloat16

    batch_spec = BATCH_SPECS[batch_name]
    seq_lens = batch_spec.seq_lens

    # Model configuration
    # q_lora_rank should not be used in this pure attention test
    # Since topk=2048 > seq_lens, indexer selects all tokens anyway
    num_heads = 128
    q_lora_rank = None
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
        q_lora_rank=None,
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

    # Add requests to KV cache manager
    request_ids = list(range(batch_spec.batch_size))
    token_nums = seq_lens  # For context phase, token_nums = seq_lens
    kv_cache_manager.add_dummy_requests(
        request_ids=request_ids,
        token_nums=token_nums,
        is_gen=False,  # Context phase
        prepare_resource=True,
    )

    # Create metadata
    AttentionCls = get_attention_backend("TRTLLM", sparse_config)
    attn_metadata = AttentionCls.Metadata(
        seq_lens=torch.tensor(seq_lens, dtype=torch.int),
        request_ids=request_ids,
        max_num_requests=batch_spec.batch_size,
        num_contexts=batch_spec.batch_size,
        prompt_lens=seq_lens,
        max_num_tokens=total_cache_tokens,
        kv_cache_manager=kv_cache_manager,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[0 for _ in seq_lens],
        ),
        mapping=mapping,
        # TODO: Required for sparse MLA to use load_paged_kv_cache_for_mla&mla_rope_append_paged_kv_assign_q kernels
        enable_context_mla_with_cached_kv=True,
    )
    attn_metadata.prepare()

    total_tokens = sum(seq_lens)
    # q: [total_tokens, num_heads * qk_head_dim]
    q = torch.randn(total_tokens, num_heads * qk_head_dim, dtype=dtype, device=device)

    # compressed_kv: [total_tokens, kv_lora_rank]
    compressed_kv = torch.randn(total_tokens, kv_lora_rank, dtype=dtype, device=device)
    k_pe = torch.randn(total_tokens, qk_rope_head_dim, dtype=dtype, device=device)

    # latent_cache for appending to KV cache
    latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)  # [total_tokens, kv_lora_rank + qk_rope_head_dim]

    # hidden_states: used by indexer to compute k via self.wk(hidden_states)
    hidden_states = torch.randn(total_tokens, hidden_size, dtype=dtype, device=device)

    # qr: compressed query used by indexer
    qr = torch.randn(total_tokens, kv_lora_rank, dtype=dtype, device=device)

    # position IDs for each sequence
    position_ids = torch.cat([torch.arange(seq_len, device=device, dtype=torch.int32)
                              for seq_len in seq_lens])

    # Create output buffer
    output = torch.empty(total_tokens, num_heads * v_head_dim, dtype=dtype, device=device)

    # Mock indexer with GLOBAL causal indices
    # Since topk_tokens=2048 > max(seq_lens), this effectively creates full causal attention
    topk_indices_global = []
    global_offset = 0
    for seq_len in seq_lens:
        for local_pos in range(seq_len):
            # For token at local position local_pos in this sequence,
            # it can attend to all previous tokens in the SAME sequence
            # Global indices: [global_offset, global_offset + local_pos]
            num_attend = min(local_pos + 1, topk_tokens)
            indices = torch.arange(global_offset, global_offset + num_attend,
                                  dtype=torch.int32, device=device)

            # Pad to topk_tokens with -1
            if len(indices) < topk_tokens:
                padding = torch.full((topk_tokens - len(indices),), -1,
                                    dtype=torch.int32, device=device)
                indices = torch.cat([indices, padding])

            topk_indices_global.append(indices)

        global_offset += seq_len  # Move to next sequence's global offset
    topk_indices_final = torch.stack(topk_indices_global, dim=0)  # [total_tokens, topk_tokens]

    # Create mock indexer as nn.Module
    mock_indexer = MockIndexer(topk_indices_final)
    mla.mqa.indexer = mock_indexer

    # Verify softmax_scale matches between MLA and test
    assert hasattr(mla, 'softmax_scale'), "MLA should have softmax_scale attribute"
    assert abs(mla.softmax_scale - softmax_scale) < 1e-6, \
        f"Scale mismatch: MLA={mla.softmax_scale:.6f} vs test={softmax_scale:.6f}"

    # Call the function under test
    result = mla.forward_sparse_mla_kvcache_bf16(
        q=q.clone(),  # Clone because it gets modified in-place
        compressed_kv=compressed_kv,
        k_pe=k_pe,
        attn_metadata=attn_metadata,
        output=output,
        latent_cache=latent_cache,
        hidden_states=hidden_states,
        qr=qr,
        position_ids=position_ids,
        is_generation=False,
    )

    # Calculate reference output
    q_reshaped = q.view(total_tokens, num_heads, qk_head_dim)
    reference_output = calculate_reference_output_prefill_only(
        q_c=q_reshaped,
        kv_c=compressed_kv,
        k_pe=k_pe,
        W_UK=W_UK,
        W_UV=W_UV,
        rope_cos_sin=rope_cos_sin,
        sequence_lengths=seq_lens,
        num_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        softmax_scale=softmax_scale,
        device=device,
    )

    assert result.shape == reference_output.shape, \
        f"Shape mismatch: result {result.shape} vs reference {reference_output.shape}"
    assert result.dtype == reference_output.dtype, \
        f"Dtype mismatch: result {result.dtype} vs reference {reference_output.dtype}"

    assert torch.isfinite(result).all(), "Result contains NaN or Inf"
    assert torch.isfinite(reference_output).all(), "Reference contains NaN or Inf"
    torch.testing.assert_close(result, reference_output, rtol=0.1, atol=0.1)
    print(f"  ✓ Accuracy test passed (rtol=0.1, atol=0.1)")

    kv_cache_manager.shutdown()


if __name__ == "__main__":
    test_forward_sparse_mla_context(batch_name="small_prefill", kv_cache_dtype="auto")


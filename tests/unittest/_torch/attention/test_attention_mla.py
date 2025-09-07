import math
import random
from dataclasses import dataclass
from typing import List

import pytest
import torch

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionInputType, MLAParams, PositionalEmbeddingParams, RopeParams)
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.llm_request import (LlmRequest,
                                                        LlmRequestState,
                                                        SamplingConfig)
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import str_dtype_to_binding, torch_dtype_to_str
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.functional import PositionEmbeddingType, RopeEmbeddingUtils
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.sampling_params import SamplingParams


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def calculate_ref_result_ctx(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                             compressed_kv: torch.Tensor, k_pe: torch.Tensor,
                             rope_cos_sin: torch.Tensor, num_heads: int,
                             num_kv_heads: int, qk_nope_head_dim: int,
                             qk_rope_head_dim: int, v_head_dim: int,
                             sequence_lengths: List[int], q_scaling: float):
    """
    use standard attention to calculate the reference result by iterating over each request
    q shape: (total_tokens, num_heads * qk_head_dim)
    k shape: (total_tokens, num_kv_heads * qk_head_dim)
    v shape: (total_tokens, num_kv_heads * v_head_dim)
    compressed_kv shape: (total_tokens, kv_lora_rank)
    k_pe shape: (total_tokens, qk_rope_head_dim)
    rope_cos_sin shape: (max_position_embeddings, 2, qk_rope_head_dim)
    """
    num_requests = len(sequence_lengths)
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    # Reshape inputs for reference calculation
    q_reshaped = []
    k_reshaped = []
    v_reshaped = []
    k_pe_ref_list = []
    total_tokens = 0
    for i in range(num_requests):
        q_seq = q[total_tokens:total_tokens + sequence_lengths[i]].unflatten(
            -1, [num_heads, qk_head_dim])
        k_seq = k[total_tokens:total_tokens + sequence_lengths[i]].unflatten(
            -1, [num_kv_heads, qk_head_dim])
        v_seq = v[total_tokens:total_tokens + sequence_lengths[i]].unflatten(
            -1, [num_kv_heads, v_head_dim])

        q_pe_seq = q_seq[..., -qk_rope_head_dim:]
        k_pe_seq = k_pe[total_tokens:total_tokens +
                        sequence_lengths[i]].unsqueeze(-2)
        cos, sin = rope_cos_sin[:sequence_lengths[i]].chunk(2, dim=-2)
        q_pe_seq = q_pe_seq.unflatten(-1, [-1, 2]).transpose(
            -2, -1).flatten(start_dim=-2)
        k_pe_seq = k_pe_seq.unflatten(-1, [-1, 2]).transpose(
            -2, -1).flatten(start_dim=-2)
        q_pe_seq = ((q_pe_seq * cos) +
                    (rotate_half(q_pe_seq) * sin)).to(dtype=q_seq.dtype)
        k_pe_seq = ((k_pe_seq * cos) +
                    (rotate_half(k_pe_seq) * sin)).to(dtype=k_seq.dtype)
        q_seq[..., -qk_rope_head_dim:] = q_pe_seq
        k_seq[..., -qk_rope_head_dim:] = k_pe_seq
        k_pe_ref_list.append(k_pe_seq)

        q_reshaped.append(q_seq.transpose(
            0, 1))  # (num_heads, seq_len, qk_head_dim)
        k_reshaped.append(k_seq.transpose(
            0, 1))  # (num_kv_heads, seq_len, qk_head_dim)
        v_reshaped.append(v_seq.transpose(
            0, 1))  # (num_kv_heads, seq_len, v_head_dim)

        total_tokens += sequence_lengths[i]

    # Calculate reference result batch by batch
    ref_results = []
    for i in range(num_requests):
        q = q_reshaped[i]  # (num_heads, seq_len, qk_head_dim)
        k = k_reshaped[i]  # (num_kv_heads, seq_len, qk_head_dim)
        v = v_reshaped[i]  # (num_kv_heads, seq_len, v_head_dim)

        # Handle grouped-query attention if num_heads > num_kv_heads
        if num_heads > num_kv_heads:
            assert num_heads % num_kv_heads == 0
            num_kv_groups = num_heads // num_kv_heads
            k = repeat_kv(k.unsqueeze(0), num_kv_groups).squeeze(0)
            v = repeat_kv(v.unsqueeze(0), num_kv_groups).squeeze(0)

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(
            1, 2)) / (q_scaling * math.sqrt(qk_head_dim))

        # For causal mask, we block future tokens (upper triangular above the diagonal)
        seq_len = q.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len,
                                            seq_len,
                                            device=q.device,
                                            dtype=torch.bool),
                                 diagonal=1)
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        # Apply softmax to get attention probabilities
        attn_weights = torch.nn.functional.softmax(attn_weights,
                                                   dim=-1,
                                                   dtype=torch.float32).to(
                                                       q.dtype)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights,
                                   v)  # (num_heads, seq_len, v_head_dim)

        # Reshape back to (seq_len, num_heads*v_head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(
            sequence_lengths[i], num_heads * v_head_dim)
        ref_results.append(attn_output)

    ref_result = torch.cat(ref_results)
    k_pe_ref = torch.cat(k_pe_ref_list).squeeze(-2)
    latent_cache = torch.cat([compressed_kv, k_pe_ref], dim=-1)
    return ref_result, latent_cache


def calculate_ref_result_gen(fused_q: torch.Tensor, q_pe: torch.Tensor,
                             compressed_kv: torch.Tensor, k_pe: torch.Tensor,
                             latent_cache: torch.Tensor,
                             rope_cos_sin: torch.Tensor, num_heads: int,
                             kv_lora_rank: int, qk_nope_head_dim: int,
                             qk_rope_head_dim: int, sequence_lengths: List[int],
                             q_scaling: float):
    """
    use standard attention to calculate the reference result by iterating over each request
    fused_q shape: (num_tokens, num_heads * (kv_lora_rank + qk_rope_head_dim))
    q_pe shape: (num_tokens, num_heads, qk_rope_head_dim)
    compressed_kv shape: (num_requests, kv_lora_rank)
    k_pe shape: (num_requests, qk_rope_head_dim)
    latent_cache shape: (total_tokens, kv_lora_rank + qk_rope_head_dim)
    rope_cos_sin shape: (max_position_embeddings, 2, qk_rope_head_dim)
    """
    num_requests = len(sequence_lengths)
    seq_len_q = fused_q.shape[0] // num_requests

    # Reshape inputs for reference calculation
    q_reshaped = []
    k_reshaped = []
    v_reshaped = []
    latent_cache_list = []
    total_tokens = 0
    for i in range(num_requests):
        fused_q_seq = fused_q[i * seq_len_q:(i + 1) * seq_len_q].unflatten(
            -1, [num_heads, kv_lora_rank + qk_rope_head_dim])
        q_pe_seq = q_pe[i * seq_len_q:(i + 1) * seq_len_q]
        compressed_kv_seq = compressed_kv[i * seq_len_q:(i + 1) *
                                          seq_len_q].unsqueeze(-2)
        k_pe_seq = k_pe[i * seq_len_q:(i + 1) * seq_len_q].unsqueeze(-2)
        latent_cache_seq = latent_cache[total_tokens:total_tokens +
                                        sequence_lengths[i]].unsqueeze(-2)

        cos, sin = rope_cos_sin[sequence_lengths[i]:sequence_lengths[i] +
                                seq_len_q].chunk(2, dim=-2)
        q_pe_seq = q_pe_seq.unflatten(-1, [-1, 2]).transpose(
            -2, -1).flatten(start_dim=-2)
        k_pe_seq = k_pe_seq.unflatten(-1, [-1, 2]).transpose(
            -2, -1).flatten(start_dim=-2)
        q_pe_seq = ((q_pe_seq * cos) +
                    (rotate_half(q_pe_seq) * sin)).to(dtype=q_pe_seq.dtype)
        k_pe_seq = ((k_pe_seq * cos) +
                    (rotate_half(k_pe_seq) * sin)).to(dtype=k_pe_seq.dtype)
        fused_q_seq[..., -qk_rope_head_dim:] = q_pe_seq
        latent_cache_seq = torch.cat([
            latent_cache_seq,
            torch.cat([compressed_kv_seq, k_pe_seq], dim=-1)
        ],
                                     dim=0)
        latent_cache_list.append(latent_cache_seq)

        q_reshaped.append(fused_q_seq.transpose(
            0, 1))  # (num_heads, seq_len_q, kv_lora_rank + qk_rope_head_dim)
        k_reshaped.append(latent_cache_seq.transpose(
            0, 1))  # (1, seq_len_kv, kv_lora_rank + qk_rope_head_dim)
        v_reshaped.append(latent_cache_seq[..., :kv_lora_rank].transpose(
            0, 1))  # (1, seq_len_kv, kv_lora_rank)

        total_tokens += sequence_lengths[i]

    # Calculate reference result batch by batch
    ref_results = []
    for i in range(num_requests):
        q = q_reshaped[
            i]  # (num_heads, seq_len_q, kv_lora_rank + qk_rope_head_dim)
        k = k_reshaped[i]  # (1, seq_len_kv, kv_lora_rank + qk_rope_head_dim)
        v = v_reshaped[i]  # (1, seq_len_kv, kv_lora_rank)

        # Handle grouped-query attention
        k = repeat_kv(k.unsqueeze(0), num_heads).squeeze(0)
        v = repeat_kv(v.unsqueeze(0), num_heads).squeeze(0)

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / (
            q_scaling * math.sqrt(qk_nope_head_dim + qk_rope_head_dim))

        # Use MTP mask by default if seqlen_q > 1.
        seq_len_q = q.shape[1]
        seq_len_kv = k.shape[1]
        mask = torch.zeros(seq_len_q,
                           seq_len_kv,
                           device=q.device,
                           dtype=torch.bool)
        for qi in range(seq_len_q):
            for ki in range(seq_len_kv - seq_len_q + 1 + qi, seq_len_kv):
                mask[qi, ki] = 1
        attn_weights = attn_weights.masked_fill(mask, float('-inf'))
        # Apply softmax to get attention probabilities
        attn_weights = torch.nn.functional.softmax(attn_weights,
                                                   dim=-1,
                                                   dtype=torch.float32).to(
                                                       q.dtype)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights,
                                   v)  # (num_heads, 1, kv_lora_rank)

        # Reshape back to (seq_len_q, num_heads*kv_lora_rank)
        attn_output = attn_output.transpose(0, 1).contiguous().view(
            -1, num_heads * kv_lora_rank)
        ref_results.append(attn_output)

    ref_result = torch.cat(ref_results)
    latent_cache = torch.cat(latent_cache_list).squeeze(-2)
    return ref_result, latent_cache


@dataclass(kw_only=True, frozen=True)
class Scenario:
    dtype: torch.dtype = torch.bfloat16
    kv_cache_dtype: torch.dtype = torch.bfloat16
    num_layers: int = 1
    num_heads: int = 128
    num_kv_heads: int = 128
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    hidden_size: int = 7168
    max_position_embeddings: int = 163840
    rope_theta: float = 10000.0
    rope_beta_fast: int = 32
    rope_beta_slow: int = 1
    rope_factor: float = 40.0
    rope_mscale: float = 1.0
    rope_mscale_all_dim: float = 1.0
    rope_original_max_position_embeddings: int = 4096
    rope_type: str = "yarn"
    model_type: str = "deepseek_v3"
    kv_cache_tokens_per_block: int = 64


@dataclass(kw_only=True, frozen=True)
class RopeConfig:
    hidden_size: int = 7168
    num_attention_heads: int = 128
    rope_scaling: dict = {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 40.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "type": "yarn",
    },
    max_position_embeddings: int = 163840
    rope_theta: float = 10000.0
    qk_rope_head_dim: int = 64
    model_type: str = "deepseek_v3"


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :,
                                  None, :, :].expand(batch, num_key_value_heads,
                                                     n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen,
                                 head_dim)


# Set seed for reproducibility.
random.seed(0)
min_context_sequence_length = 1
max_context_sequence_length = 1000
min_num_contexts = 1
max_num_contexts = 10
random_context_sequence_lengths = [
    random.randint(min_context_sequence_length, max_context_sequence_length)
    for _ in range(random.randint(min_num_contexts, max_num_contexts))
]

# Define test data
context_sequence_lengths = [
    [10, 12, 5],
    [100, 300, 20, 10],
    [253, 253, 253, 253],
    [100, 1110, 1000, 1000],
    random_context_sequence_lengths,
]
# Use MTP by default if seqlen_q > 1.
generation_seq_len_q = [1, 4]
num_generation_steps = [10]

kv_cache_dtype_list = [torch.bfloat16]
if torch.cuda.get_device_capability() in [(8, 9), (9, 0), (10, 0), (12, 0)]:
    kv_cache_dtype_list.append(torch.float8_e4m3fn)
scenarios = [
    Scenario(kv_cache_dtype=kv_cache_dtype, num_layers=num_layers)
    for kv_cache_dtype in kv_cache_dtype_list for num_layers in [1, 2]
]

accuracy_dict = {
    torch.bfloat16: (3e-2, 3e-3),
    torch.float8_e4m3fn: (4e-1, 4e-2),
}


# Convert parameterized tests to pytest parametrize
@pytest.mark.parametrize("scenario", scenarios, ids=lambda x: f"scenario: {x}")
@pytest.mark.parametrize("context_sequence_lengths",
                         context_sequence_lengths,
                         ids=lambda x: f"context_sequence_lengths: {x}")
@pytest.mark.parametrize("generation_seq_len_q",
                         generation_seq_len_q,
                         ids=lambda x: f"generation_seq_len_q: {x}")
@pytest.mark.parametrize("num_generation_steps",
                         num_generation_steps,
                         ids=lambda x: f"num_generation_steps: {x}")
def test_attention_mla(scenario: Scenario, context_sequence_lengths: List[int],
                       generation_seq_len_q: int,
                       num_generation_steps: List[int]):
    """Test MLA computation for both context and generation phases"""
    num_heads = scenario.num_heads
    num_kv_heads = scenario.num_kv_heads
    q_lora_rank = scenario.q_lora_rank
    kv_lora_rank = scenario.kv_lora_rank
    qk_nope_head_dim = scenario.qk_nope_head_dim
    qk_rope_head_dim = scenario.qk_rope_head_dim
    v_head_dim = scenario.v_head_dim
    rope_config = RopeConfig(
        hidden_size=scenario.hidden_size,
        num_attention_heads=scenario.num_heads,
        rope_scaling={
            "beta_fast": scenario.rope_beta_fast,
            "beta_slow": scenario.rope_beta_slow,
            "factor": scenario.rope_factor,
            "mscale": scenario.rope_mscale,
            "mscale_all_dim": scenario.rope_mscale_all_dim,
            "original_max_position_embeddings":
            scenario.rope_original_max_position_embeddings,
            "type": scenario.rope_type,
        },
        max_position_embeddings=scenario.max_position_embeddings,
        rope_theta=scenario.rope_theta,
        qk_rope_head_dim=scenario.qk_rope_head_dim,
        model_type=scenario.model_type,
    )
    kv_cache_tokens_per_block = scenario.kv_cache_tokens_per_block
    num_layers = scenario.num_layers
    device = torch.device('cuda')
    dtype = scenario.dtype
    kv_cache_dtype = scenario.kv_cache_dtype

    print(
        f"--------------------------------Test for scenario: {scenario} start--------------------------------"
    )

    _run_test_for_backend("TRTLLM", num_heads, num_kv_heads, num_layers,
                          q_lora_rank, kv_lora_rank, qk_nope_head_dim,
                          qk_rope_head_dim, v_head_dim, rope_config,
                          kv_cache_tokens_per_block, device, dtype,
                          kv_cache_dtype, context_sequence_lengths,
                          generation_seq_len_q, num_generation_steps)


def _run_test_for_backend(backend_name, num_heads, num_kv_heads, num_layers,
                          q_lora_rank, kv_lora_rank, qk_nope_head_dim,
                          qk_rope_head_dim, v_head_dim, rope_config,
                          kv_cache_tokens_per_block, device, dtype,
                          kv_cache_dtype, context_sequence_lengths,
                          generation_seq_len_q, num_generation_steps):
    AttentionCls = get_attention_backend(backend_name)
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    # Set seed for reproducibility.
    torch.manual_seed(123)

    # Create inputs
    inputs_per_layer = []
    for layer_idx in range(num_layers):
        ctx_compressed_kv = torch.cat([
            torch.empty(
                [ctx_len, kv_lora_rank],
                dtype=dtype,
                device=device,
            ).uniform_(-1, 1) for ctx_len in context_sequence_lengths
        ])
        ctx_k_pe = torch.cat([
            torch.empty(
                [ctx_len, qk_rope_head_dim],
                dtype=dtype,
                device=device,
            ).uniform_(-1, 1) for ctx_len in context_sequence_lengths
        ])
        ctx_q = torch.cat([
            torch.empty(
                [ctx_len, num_heads * qk_head_dim],
                dtype=dtype,
                device=device,
            ).uniform_(-1, 1) for ctx_len in context_sequence_lengths
        ])
        ctx_kv = torch.cat([
            torch.empty(
                [ctx_len, num_kv_heads * (qk_nope_head_dim + v_head_dim)],
                dtype=dtype,
                device=device,
            ).uniform_(-1, 1) for ctx_len in context_sequence_lengths
        ])
        # ctx_v.stride(0) == num_kv_heads * (qk_nope_head_dim + v_head_dim)
        ctx_k_nope, ctx_v = ctx_kv.split(
            [num_kv_heads * qk_nope_head_dim, num_kv_heads * v_head_dim],
            dim=-1)
        ctx_k_nope = ctx_k_nope.view(-1, num_kv_heads, qk_nope_head_dim)
        ctx_k = torch.cat([
            ctx_k_nope,
            ctx_k_pe.view(-1, 1, qk_rope_head_dim).expand(-1, num_kv_heads, -1)
        ],
                          dim=-1)
        ctx_k = ctx_k.view(-1, num_kv_heads * qk_head_dim)

        gen_compressed_kv_list = [
            torch.cat([
                torch.empty(
                    [generation_seq_len_q, kv_lora_rank],
                    dtype=dtype,
                    device=device,
                ).uniform_(-1, 1) for _ in context_sequence_lengths
            ]) for _ in range(num_generation_steps)
        ]
        gen_k_pe_list = [
            torch.cat([
                torch.empty(
                    [generation_seq_len_q, qk_rope_head_dim],
                    dtype=dtype,
                    device=device,
                ).uniform_(-1, 1) for _ in context_sequence_lengths
            ]) for _ in range(num_generation_steps)
        ]
        gen_fused_q_list = [
            torch.cat([
                torch.empty(
                    [
                        generation_seq_len_q, num_heads *
                        (kv_lora_rank + qk_rope_head_dim)
                    ],
                    dtype=dtype,
                    device=device,
                ).uniform_(-1, 1) for _ in context_sequence_lengths
            ]) for _ in range(num_generation_steps)
        ]
        gen_q_pe_list = [
            torch.cat([
                torch.empty(
                    [generation_seq_len_q, num_heads, qk_rope_head_dim],
                    dtype=dtype,
                    device=device,
                ).uniform_(-1, 1) for _ in context_sequence_lengths
            ]) for _ in range(num_generation_steps)
        ]

        inputs = {
            "ctx_compressed_kv": ctx_compressed_kv,
            "ctx_k_pe": ctx_k_pe,
            "ctx_q": ctx_q,
            "ctx_k": ctx_k,
            "ctx_v": ctx_v,
            "gen_compressed_kv_list": gen_compressed_kv_list,
            "gen_k_pe_list": gen_k_pe_list,
            "gen_fused_q_list": gen_fused_q_list,
            "gen_q_pe_list": gen_q_pe_list,
        }
        inputs_per_layer.append(inputs)
        print(f"context sequence lengths: {context_sequence_lengths}")
        for key, val in inputs.items():
            if key.endswith("_list"):
                print(f"{key}: [{val[0].shape}] * {len(val)}")
            else:
                print(f"{key}: {val.shape}")

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

    # Setup attention module and metadata
    pos_embd_params = PositionalEmbeddingParams(
        type=PositionEmbeddingType.yarn,
        rope=RopeParams.from_config(rope_config),
        is_neox=False,
    )
    mla_params = MLAParams(
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        v_head_dim=v_head_dim,
        predicted_tokens_per_seq=1,
    )

    def yarn_get_mscale(scale=1, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    mscale_all_dim = pos_embd_params.rope.mscale_all_dim
    scaling_factor = pos_embd_params.rope.scale
    mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
    q_scaling = 1.0 / (mscale * mscale)

    quant_config = None
    if kv_cache_dtype == torch.float8_e4m3fn:
        quant_config = QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8.value)
    ctx_layers = [
        AttentionCls(
            layer_idx=layer_idx,
            num_heads=num_heads,
            head_dim=qk_head_dim,
            num_kv_heads=num_kv_heads,
            quant_config=quant_config,
            q_scaling=q_scaling,
            pos_embd_params=pos_embd_params,
            mla_params=mla_params,
        ) for layer_idx in range(num_layers)
    ]
    gen_layers = [
        AttentionCls(
            layer_idx=layer_idx,
            num_heads=num_heads,
            head_dim=kv_lora_rank + qk_rope_head_dim,
            num_kv_heads=1,
            quant_config=quant_config,
            q_scaling=q_scaling,
            pos_embd_params=pos_embd_params,
            mla_params=mla_params,
        ) for layer_idx in range(num_layers)
    ]

    # NOTE: set up metadata, refer to tensorrt_llm/_torch/pyexecutor/model_engine.py
    # all layers share the same metadata
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    max_tokens = (
        max_context_sequence_length +
        (num_generation_steps + 1) * generation_seq_len_q +
        kv_cache_tokens_per_block - 1
    ) // kv_cache_tokens_per_block * kv_cache_tokens_per_block * max_num_contexts
    kv_cache_manager = KVCacheManager(
        KvCacheConfig(
            max_tokens=max_tokens,
            enable_block_reuse=False,
        ),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=kv_lora_rank + qk_rope_head_dim,
        tokens_per_block=kv_cache_tokens_per_block,
        max_seq_len=max(context_sequence_lengths) +
        (num_generation_steps + 1) * generation_seq_len_q,
        max_batch_size=len(context_sequence_lengths),
        mapping=mapping,
        dtype=str_dtype_to_binding(torch_dtype_to_str(kv_cache_dtype)),
    )
    request_list = []
    for req_id, ctx_len in enumerate(context_sequence_lengths):
        req = LlmRequest(
            request_id=req_id,
            max_new_tokens=num_generation_steps + 1,
            input_tokens=[1] * ctx_len,
            sampling_config=SamplingConfig(
                SamplingParams()._get_sampling_config()),
            is_streaming=False,
        )
        req.paged_kv_block_ids = []
        beam_width = 1
        kv_cache_manager.impl.add_sequence(req_id, ctx_len, beam_width, req)
        request_list.append(req)
    attn_metadata = AttentionCls.Metadata(
        seq_lens=torch.tensor(context_sequence_lengths, dtype=torch.int),
        request_ids=list(range(len(context_sequence_lengths))),
        max_num_requests=len(context_sequence_lengths),
        num_contexts=len(context_sequence_lengths),
        prompt_lens=context_sequence_lengths,
        max_num_tokens=max(context_sequence_lengths),
        kv_cache_manager=kv_cache_manager,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[0 for _ in context_sequence_lengths],
        ),
        mapping=mapping,
    )
    attn_metadata.prepare()

    # run forward for each step and each layer
    latent_cache_ref_all_list = [None for _ in range(num_layers)]
    for step in range(num_generation_steps + 1):
        if step > 0:
            for req_id in range(len(context_sequence_lengths)):
                for _ in range(generation_seq_len_q):
                    kv_cache_manager.impl.add_token(req_id)
            attn_metadata = AttentionCls.Metadata(
                seq_lens=torch.tensor([generation_seq_len_q] *
                                      len(context_sequence_lengths),
                                      dtype=torch.int),
                request_ids=list(range(len(context_sequence_lengths))),
                max_num_requests=len(context_sequence_lengths),
                num_contexts=0,
                prompt_lens=context_sequence_lengths,
                max_num_tokens=max(context_sequence_lengths),
                kv_cache_manager=kv_cache_manager,
                kv_cache_params=KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=[
                        ctx_len + (step - 1) * generation_seq_len_q
                        for ctx_len in context_sequence_lengths
                    ],
                ),
                mapping=mapping,
                enable_flash_mla=torch.cuda.get_device_capability() == (9, 0),
            )
            attn_metadata.prepare()
        for layer_idx in range(num_layers):
            print(
                f"--------------------------------step {step} layer {layer_idx} start--------------------------------"
            )
            if step == 0:
                q = inputs_per_layer[layer_idx]["ctx_q"]
                k = inputs_per_layer[layer_idx]["ctx_k"]
                v = inputs_per_layer[layer_idx]["ctx_v"]
                compressed_kv = inputs_per_layer[layer_idx]["ctx_compressed_kv"]
                k_pe = inputs_per_layer[layer_idx]["ctx_k_pe"]
                latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)
                # q/k will be modified in the forward pass, so we need to clone them
                # we should not clone v because we need to keep the stride of v
                result = ctx_layers[layer_idx].forward(
                    q.clone(),
                    k.clone(),
                    v,
                    attn_metadata,
                    attention_input_type=AttentionInputType.context_only,
                    latent_cache=latent_cache,
                )
                ref_result, latent_cache_ref = calculate_ref_result_ctx(
                    q,
                    k,
                    v,
                    compressed_kv,
                    k_pe,
                    rope_cos_sin,
                    num_heads,
                    num_kv_heads,
                    qk_nope_head_dim,
                    qk_rope_head_dim,
                    v_head_dim,
                    context_sequence_lengths,
                    q_scaling,
                )
                latent_cache_ref_all_list[layer_idx] = latent_cache_ref
                for req in request_list:
                    req.state = LlmRequestState.GENERATION_IN_PROGRESS
            else:
                fused_q = inputs_per_layer[layer_idx]["gen_fused_q_list"][step -
                                                                          1]
                q_pe = inputs_per_layer[layer_idx]["gen_q_pe_list"][step - 1]
                compressed_kv = inputs_per_layer[layer_idx][
                    "gen_compressed_kv_list"][step - 1]
                k_pe = inputs_per_layer[layer_idx]["gen_k_pe_list"][step - 1]
                latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)
                result = gen_layers[layer_idx].forward(
                    fused_q,
                    None,
                    None,
                    attn_metadata,
                    attention_input_type=AttentionInputType.generation_only,
                    latent_cache=latent_cache,
                    q_pe=q_pe,
                )
                ref_result, latent_cache_ref = calculate_ref_result_gen(
                    fused_q,
                    q_pe,
                    compressed_kv,
                    k_pe,
                    latent_cache_ref_all_list[layer_idx],
                    rope_cos_sin,
                    num_heads,
                    kv_lora_rank,
                    qk_nope_head_dim,
                    qk_rope_head_dim,
                    [
                        ctx_len + (step - 1) * generation_seq_len_q
                        for ctx_len in context_sequence_lengths
                    ],
                    q_scaling,
                )
                latent_cache_ref_all_list[layer_idx] = latent_cache_ref
            # Compare results
            print(
                f"{backend_name} output mean: {result.abs().mean().item()}, max: {result.abs().max().item()}"
            )
            print(
                f"Reference output mean: {ref_result.abs().mean().item()}, max: {ref_result.abs().max().item()}"
            )
            print(
                f"Difference mean: {(result - ref_result).abs().mean().item()}, max: {(result - ref_result).abs().max().item()}"
            )

            # Assert results are close
            atol, rtol = accuracy_dict[kv_cache_dtype]
            assert torch.allclose(result, ref_result, atol=atol, rtol=rtol), \
                f"Results for MLA in {backend_name} backend don't match reference implementation at layer {layer_idx} in step {step}"

            print(
                f"Test for MLA in {backend_name} backend passed at layer {layer_idx} in step {step}"
            )
            print(
                f"--------------------------------step {step} layer {layer_idx} end--------------------------------"
            )

    print(f"Test for MLA in {backend_name} backend passed")
    kv_cache_manager.shutdown()

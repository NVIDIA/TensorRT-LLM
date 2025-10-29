import itertools
import math
from dataclasses import dataclass
from typing import List, Tuple

import pytest
import torch
from utils.util import skip_blackwell

from tensorrt_llm._torch.attention_backend.interface import \
    PredefinedAttentionMask
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend


def generate_attn_scenarios(num_q_heads_kv_heads: List[Tuple[int, int]],
                            head_dim: List[int], num_layers: List[int],
                            dtype: List[torch.dtype]):
    scenarios = []
    product_iter = itertools.product(num_q_heads_kv_heads, head_dim, num_layers,
                                     dtype)
    for num_q_heads_kv_head, head_dim, num_layers, dtype in product_iter:
        num_q_heads, num_kv_heads = num_q_heads_kv_head
        scenarios.append(
            Scenario(num_heads=num_q_heads,
                     num_kv_heads=num_kv_heads,
                     head_dim=head_dim,
                     num_layers=num_layers,
                     dtype=dtype))
    return scenarios


def calculate_ref_result(q: torch.Tensor,
                         k: torch.Tensor,
                         v: torch.Tensor,
                         num_heads: int,
                         num_kv_heads: int,
                         head_dim: int,
                         sequence_lengths: List[int],
                         mask_type=PredefinedAttentionMask.FULL):
    """
    use standard attention to calculate the reference result by iterating over each request
    q shape: (total_tokens, num_heads * head_dim)
    k shape: (total_tokens, num_kv_heads * head_dim)
    v shape: (total_tokens, num_kv_heads * head_dim)
    mask_type: either PredefinedAttentionMask.FULL or PredefinedAttentionMask.CAUSAL
    """
    num_requests = len(sequence_lengths)
    # Reshape inputs for reference calculation
    q_reshaped = []
    k_reshaped = []
    v_reshaped = []
    total_tokens = 0

    # Reshape inputs for reference calculation
    for i in range(num_requests):
        q_seq = q[total_tokens:total_tokens + sequence_lengths[i]]
        k_seq = k[total_tokens:total_tokens + sequence_lengths[i]]
        v_seq = v[total_tokens:total_tokens + sequence_lengths[i]]

        # Reshape to (seq_len, num_heads, head_dim)
        q_seq = q_seq.view(sequence_lengths[i], num_heads, head_dim)
        k_seq = k_seq.view(sequence_lengths[i], num_kv_heads, head_dim)
        v_seq = v_seq.view(sequence_lengths[i], num_kv_heads, head_dim)

        q_reshaped.append(q_seq.transpose(0,
                                          1))  # (num_heads, seq_len, head_dim)
        k_reshaped.append(k_seq.transpose(
            0, 1))  # (num_kv_heads, seq_len, head_dim)
        v_reshaped.append(v_seq.transpose(
            0, 1))  # (num_kv_heads, seq_len, head_dim)

        total_tokens += sequence_lengths[i]

    # Calculate reference result batch by batch
    ref_results = []
    for i in range(num_requests):
        q = q_reshaped[i]  # (num_heads, seq_len, head_dim)
        k = k_reshaped[i]  # (num_kv_heads, seq_len, head_dim)
        v = v_reshaped[i]  # (num_kv_heads, seq_len, head_dim)

        # Handle grouped-query attention if num_heads > num_kv_heads
        if num_heads > num_kv_heads:
            num_kv_groups = num_heads // num_kv_heads
            k = repeat_kv(k.unsqueeze(0), num_kv_groups).squeeze(0)
            v = repeat_kv(v.unsqueeze(0), num_kv_groups).squeeze(0)

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(head_dim)

        if mask_type == PredefinedAttentionMask.CAUSAL:
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
                                   v)  # (num_heads, seq_len, head_dim)

        # Reshape back to (seq_len, num_heads*head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(
            sequence_lengths[i], num_heads * head_dim)
        ref_results.append(attn_output)

    ref_result = torch.cat(ref_results)
    return ref_result


@dataclass(kw_only=True, frozen=True)
class Scenario:
    dtype: torch.dtype = torch.float16
    num_layers: int = 1
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128

    @property
    def num_kv_groups(self) -> int:
        return self.num_heads // self.num_kv_heads

    # self-defined repr for pytest substring match
    def __repr__(self) -> str:
        return f"Scenario(num_heads_{self.num_heads}, num_kv_heads_{self.num_kv_heads}, head_dim_{self.head_dim}, num_layers_{self.num_layers}, dtype_{self.dtype})"


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


min_context_sequence_length = 1
max_context_sequence_length = 1000
min_num_contexts = 1
max_num_contexts = 10

# Define test data
context_sequence_lengths = [
    [10, 12, 5],
    [100, 300, 20, 10],
    [253, 253, 253, 253],
    [100, 1110, 1000, 1000],
]

num_q_heads_kv_heads = [
    (32, 32),
    (32, 8),
    (16, 16),
]
num_layers = [1, 2, 16]
head_dim = [64, 72, 128]
dtype = [torch.float16]

scenarios = generate_attn_scenarios(num_q_heads_kv_heads, head_dim, num_layers,
                                    dtype)


# skip for blackwell
@skip_blackwell
@pytest.mark.skip(reason="https://nvbugspro.nvidia.com/bug/5247232")
# Convert parameterized tests to pytest parametrize
@pytest.mark.parametrize("accuracy", [(1e-2, 1e-2)],
                         ids=lambda x: f"atol={x[0]} rtol={x[1]}")
@pytest.mark.parametrize("scenario", scenarios, ids=lambda x: f"scenario: {x}")
@pytest.mark.parametrize("context_sequence_lengths",
                         context_sequence_lengths,
                         ids=lambda x: f"context_sequence_lengths: {x}")
@pytest.mark.parametrize(
    "mask_type", [PredefinedAttentionMask.FULL, PredefinedAttentionMask.CAUSAL],
    ids=lambda x: f"mask_type: {x}")
def test_attention_no_cache(scenario: Scenario,
                            context_sequence_lengths: List[int], mask_type,
                            accuracy):
    """Test attention computation without using cache for both FULL and CAUSAL masks"""
    # set seed for reproducibility
    torch.manual_seed(720)

    num_heads = scenario.num_heads
    num_kv_heads = scenario.num_kv_heads
    head_dim = scenario.head_dim
    num_layers = scenario.num_layers
    device = torch.device('cuda')
    dtype = scenario.dtype
    print(
        f"--------------------------------Test for scenario: {scenario} with mask_type: {mask_type} start--------------------------------"
    )

    _run_test_for_backend("TRTLLM", num_heads, num_kv_heads, num_layers,
                          head_dim, device, dtype, context_sequence_lengths,
                          mask_type, accuracy)


def _run_test_for_backend(backend_name, num_heads, num_kv_heads, num_layers,
                          head_dim, device, dtype, context_sequence_lengths,
                          mask_type, accuracy):
    AttentionCls = get_attention_backend(backend_name)

    sequence_lengths = context_sequence_lengths
    token_nums = sequence_lengths  # Only current tokens, no past tokens

    contexts_per_layer = []
    for layer_idx in range(num_layers):
        # Create query, key, value tensors(only context phase)
        context_qs = torch.cat([
            torch.randn(seq_len,
                        num_heads * head_dim,
                        dtype=dtype,
                        device=device) for seq_len in sequence_lengths
        ])
        context_ks = torch.cat([
            torch.randn(seq_len,
                        num_kv_heads * head_dim,
                        dtype=dtype,
                        device=device) for seq_len in sequence_lengths
        ])
        context_vs = torch.cat([
            torch.randn(seq_len,
                        num_kv_heads * head_dim,
                        dtype=dtype,
                        device=device) for seq_len in sequence_lengths
        ])

        print(f"context sequence lengths: {sequence_lengths}")
        print(
            f"context_qs.shape: {context_qs.shape}, context_ks.shape: {context_ks.shape}, context_vs.shape: {context_vs.shape}"
        )

        contexts_per_layer.append((context_qs, context_ks, context_vs))

    # Setup attention module and metadata
    layers = [
        AttentionCls(layer_idx=layer_idx,
                     num_heads=num_heads,
                     head_dim=head_dim,
                     num_kv_heads=num_kv_heads)
        for layer_idx in range(num_layers)
    ]

    # NOTE: set up metadata, refer to tensorrt_llm/_torch/pyexecutor/model_engine.py
    # all layers share the same metadata
    attn_metadata = AttentionCls.Metadata(
        max_num_requests=len(context_sequence_lengths),
        max_num_tokens=8192,
        kv_cache_manager=None,
        mapping=None,
        runtime_features=None)

    # NOTE: set up metadata
    attn_metadata.seq_lens = torch.tensor(sequence_lengths, dtype=torch.int)
    attn_metadata.num_contexts = len(context_sequence_lengths)
    attn_metadata.request_ids = torch.tensor(range(
        len(context_sequence_lengths)),
                                             dtype=torch.int)
    attn_metadata.max_seq_len = max(sequence_lengths)
    attn_metadata.prepare()
    print(f"attn_metadata: {attn_metadata}")
    # run forward for each layer
    for layer_idx in range(num_layers):
        q_at_layer = contexts_per_layer[layer_idx][0]
        k_at_layer = contexts_per_layer[layer_idx][1]
        v_at_layer = contexts_per_layer[layer_idx][2]
        print(
            f"--------------------------------layer {layer_idx} start--------------------------------"
        )
        # qkv_at_layer shape (num_tokens, (num_heads + 2 * num_kv_heads) * head_dim)
        qkv_at_layer = torch.cat((q_at_layer, k_at_layer, v_at_layer), dim=-1)

        result = layers[layer_idx].forward(qkv_at_layer,
                                           None,
                                           None,
                                           attn_metadata,
                                           attention_mask=mask_type)

        # Calculate reference result for validation
        ref_result = calculate_ref_result(q_at_layer,
                                          k_at_layer,
                                          v_at_layer,
                                          num_heads,
                                          num_kv_heads,
                                          head_dim,
                                          sequence_lengths,
                                          mask_type=mask_type)

        # Compare results
        print(f"{backend_name} output mean: {result.abs().mean().item()}")
        print(f"Reference output mean: {ref_result.abs().mean().item()}")
        print(f"Difference mean: {(result - ref_result).abs().mean().item()}")

        # Assert results are close
        atol = accuracy[0]
        rtol = accuracy[1]
        assert torch.allclose(result, ref_result, atol=atol, rtol=rtol), \
            f"Results for {backend_name} backend don't match reference implementation at layer {layer_idx}"

        print(
            f"Test for {backend_name} backend without cache passed at layer {layer_idx}"
        )
        print(
            f"--------------------------------layer {layer_idx} end--------------------------------"
        )

    print(
        f"Test for {backend_name} backend without cache passed with mask_type: {mask_type}"
    )

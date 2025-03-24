import math
import random
from dataclasses import dataclass
from typing import List

import pytest
import torch

from tensorrt_llm._torch.attention_backend.interface import \
    PredefinedAttentionMask
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend


def calculate_ref_result(q: List[torch.Tensor], k: List[torch.Tensor],
                         v: List[torch.Tensor], num_heads: int,
                         num_kv_heads: int, head_dim: int,
                         sequence_lengths: List[int]):
    """
    use standard attention to calculate the reference result by iterating over each request
    q, k, v are list of tensors, each tensor is a request
    q[i] shape: (total_tokens, num_heads, head_dim)
    k[i] shape: (total_tokens, num_kv_heads, head_dim)
    v[i] shape: (total_tokens, num_kv_heads, head_dim)
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

scenarios = [
    Scenario(
        num_layers=1,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        dtype=torch.float16,
    ),
    Scenario(
        num_layers=2,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        dtype=torch.float16,
    ),
]


# Convert parameterized tests to pytest parametrize
@pytest.mark.parametrize("scenario", scenarios)
@pytest.mark.parametrize("context_sequence_lengths", context_sequence_lengths)
def test_attention_no_cache(scenario: Scenario,
                            context_sequence_lengths: List[int]):
    """Test attention computation without using cache"""
    num_heads = scenario.num_heads
    num_kv_heads = scenario.num_kv_heads
    head_dim = scenario.head_dim
    num_layers = scenario.num_layers
    device = torch.device('cuda')
    dtype = scenario.dtype
    print(
        f"--------------------------------Test for scenario: {scenario} start--------------------------------"
    )

    # Test with TRTLLM backend if available
    _run_test_for_backend("TRTLLM", num_heads, num_kv_heads, num_layers,
                          head_dim, device, dtype, context_sequence_lengths)


def _run_test_for_backend(backend_name, num_heads, num_kv_heads, num_layers,
                          head_dim, device, dtype, context_sequence_lengths):
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
    kwargs = {
        "max_num_requests": len(context_sequence_lengths),
        "max_num_tokens": 8192,
        "kv_cache_manager": None,
        "mapping": None,
        "runtime_features": None,
        "is_dummy_attention": False,
    }
    # # only trtllm attention metadata has is_no_cache for now
    # if AttentionCls.Metadata is TrtllmAttentionMetadata:
    #     kwargs.update({"is_no_cache": True})
    #     kwargs["max_seq_len"] = max(sequence_lengths)
    # all layers share the same metadata
    attn_metadata = AttentionCls.Metadata(**kwargs)

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
        print(
            f"q_at_layer.shape: {q_at_layer.shape}, k_at_layer.shape: {k_at_layer.shape}, v_at_layer.shape: {v_at_layer.shape}"
        )
        # qkv_at_layer shape (num_tokens, (num_heads + 2 * num_kv_heads) * head_dim)
        qkv_at_layer = torch.cat((q_at_layer, k_at_layer, v_at_layer), dim=-1)
        print(f"qkv_at_layer.shape: {qkv_at_layer.shape}")

        result = layers[layer_idx].forward(
            qkv_at_layer,
            None,
            None,
            attn_metadata,
            attention_mask=PredefinedAttentionMask.FULL)

        # Calculate reference result for validation
        ref_result = calculate_ref_result(q_at_layer, k_at_layer, v_at_layer,
                                          num_heads, num_kv_heads, head_dim,
                                          sequence_lengths)

        # Compare results
        print(f"{backend_name} output mean: {result.abs().mean().item()}")
        print(f"Reference output mean: {ref_result.abs().mean().item()}")
        print(f"Difference mean: {(result - ref_result).abs().mean().item()}")

        # Assert results are close
        atol = 1e-2
        rtol = 1e-3
        assert torch.allclose(result, ref_result, atol=atol, rtol=rtol), \
            f"Results for {backend_name} backend don't match reference implementation at layer {layer_idx}"

        print(
            f"Test for {backend_name} backend without cache passed at layer {layer_idx}"
        )
        print(
            f"--------------------------------layer {layer_idx} end--------------------------------"
        )

    print(f"Test for {backend_name} backend without cache passed")


# For direct testing without pytest
def _test_attention_no_cache_normal():
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    device = torch.device('cuda')
    dtype = torch.float16

    # Create query, key, value tensors(only context phase)
    _run_test_for_backend_normal("TRTLLM", num_heads, num_kv_heads, head_dim,
                                 device, dtype)


def _run_test_for_backend_normal(backend_name, num_heads, num_kv_heads,
                                 head_dim, device, dtype):
    AttentionCls = get_attention_backend(backend_name)
    AttentionCls.Metadata

    context_sequence_lengths = [10, 12, 1, 5]
    # context_sequence_lengths = [10,]
    sequence_lengths = context_sequence_lengths
    token_nums = sequence_lengths  # Only current tokens, no past tokens

    # Create query, key, value tensors(only context phase)
    context_1_q = torch.randn(context_sequence_lengths[0],
                              num_heads * head_dim,
                              dtype=dtype,
                              device=device)
    context_1_k = torch.randn(context_sequence_lengths[0],
                              num_kv_heads * head_dim,
                              dtype=dtype,
                              device=device)
    context_1_v = torch.randn(context_sequence_lengths[0],
                              num_kv_heads * head_dim,
                              dtype=dtype,
                              device=device)

    context_2_q = torch.randn(context_sequence_lengths[1],
                              num_heads * head_dim,
                              dtype=dtype,
                              device=device)
    context_2_k = torch.randn(context_sequence_lengths[1],
                              num_kv_heads * head_dim,
                              dtype=dtype,
                              device=device)
    context_2_v = torch.randn(context_sequence_lengths[1],
                              num_kv_heads * head_dim,
                              dtype=dtype,
                              device=device)
    context_3_q = torch.randn(context_sequence_lengths[2],
                              num_heads * head_dim,
                              dtype=dtype,
                              device=device)
    context_3_k = torch.randn(context_sequence_lengths[2],
                              num_kv_heads * head_dim,
                              dtype=dtype,
                              device=device)
    context_3_v = torch.randn(context_sequence_lengths[2],
                              num_kv_heads * head_dim,
                              dtype=dtype,
                              device=device)
    context_4_q = torch.randn(context_sequence_lengths[3],
                              num_heads * head_dim,
                              dtype=dtype,
                              device=device)
    context_4_k = torch.randn(context_sequence_lengths[3],
                              num_kv_heads * head_dim,
                              dtype=dtype,
                              device=device)
    context_4_v = torch.randn(context_sequence_lengths[3],
                              num_kv_heads * head_dim,
                              dtype=dtype,
                              device=device)
    # Setup attention module and metadata
    attention = AttentionCls(layer_idx=0,
                             num_heads=num_heads,
                             head_dim=head_dim,
                             num_kv_heads=num_kv_heads)

    # [context_1, context_2, context_3, context_4]
    attn_metadata = AttentionCls.Metadata(
        seq_lens=torch.tensor(sequence_lengths, dtype=torch.int),
        num_contexts=len(context_sequence_lengths),
        max_num_requests=len(context_sequence_lengths),
        max_num_tokens=8192,
        kv_cache_manager=None,
        mapping=None,
        runtime_features=None)
    # NOTE: set up metadata
    attn_metadata.prompt_lens = attn_metadata.context_lens
    attn_metadata.request_ids = torch.tensor(range(
        len(context_sequence_lengths)),
                                             dtype=torch.int)
    attn_metadata.prepare()
    print(attn_metadata)

    # Calculate output using the attention module
    combined_q = torch.cat((context_1_q, context_2_q, context_3_q, context_4_q),
                           dim=0)
    combined_k = torch.cat((context_1_k, context_2_k, context_3_k, context_4_k),
                           dim=0)
    combined_v = torch.cat((context_1_v, context_2_v, context_3_v, context_4_v),
                           dim=0)
    print(combined_q.shape, combined_k.shape, combined_v.shape)
    # qkv shape (num_tokens, (num_heads + 2 * num_kv_heads) * head_dim)
    qkv = torch.cat((combined_q, combined_k, combined_v), dim=-1)
    print(qkv.shape)

    # fused qkv test
    result = attention.forward(qkv,
                               None,
                               None,
                               attn_metadata,
                               attention_mask=PredefinedAttentionMask.FULL,
                               no_cache=True)

    # Calculate reference result for validation
    combined_q = torch.cat((context_1_q, context_2_q, context_3_q, context_4_q),
                           dim=0)
    combined_k = torch.cat((context_1_k, context_2_k, context_3_k, context_4_k),
                           dim=0)
    combined_v = torch.cat((context_1_v, context_2_v, context_3_v, context_4_v),
                           dim=0)

    # Reshape inputs for reference calculation
    q_reshaped = []
    k_reshaped = []
    v_reshaped = []
    total_tokens = 0

    for seq_len in context_sequence_lengths:
        q_seq = combined_q[total_tokens:total_tokens + seq_len]
        k_seq = combined_k[total_tokens:total_tokens + seq_len]
        v_seq = combined_v[total_tokens:total_tokens + seq_len]

        # Reshape to (seq_len, num_heads, head_dim)
        q_seq = q_seq.view(seq_len, num_heads, head_dim)
        k_seq = k_seq.view(seq_len, num_kv_heads, head_dim)
        v_seq = v_seq.view(seq_len, num_kv_heads, head_dim)

        q_reshaped.append(q_seq.transpose(0,
                                          1))  # (num_heads, seq_len, head_dim)
        k_reshaped.append(k_seq.transpose(
            0, 1))  # (num_kv_heads, seq_len, head_dim)
        v_reshaped.append(v_seq.transpose(
            0, 1))  # (num_kv_heads, seq_len, head_dim)

        total_tokens += seq_len

    # Calculate reference result batch by batch
    ref_results = []
    for i in range(len(context_sequence_lengths)):
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

        # Apply softmax to get attention probabilities
        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(dtype)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights,
                                   v)  # (num_heads, seq_len, head_dim)

        # Reshape back to (seq_len, num_heads*head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(
            context_sequence_lengths[i], num_heads * head_dim)
        ref_results.append(attn_output)

    ref_result = torch.cat(ref_results)

    # Compare results
    print(f"{backend_name} output mean: {result.abs().mean().item()}")
    print(f"Reference output mean: {ref_result.abs().mean().item()}")
    print(f"Difference mean: {(result - ref_result).abs().mean().item()}")
    print(f"result: {result}")
    print(f"ref_result: {ref_result}")

    # Assert results are close
    atol = 1e-2
    rtol = 1e-3
    assert torch.allclose(result, ref_result, atol=atol, rtol=rtol), \
        f"Results for {backend_name} backend don't match reference implementation"

    print(f"Test for {backend_name} backend without cache passed")


if __name__ == '__main__':
    test_attention_no_cache()

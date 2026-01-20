import json
import os

import pytest
import torch
from utils.llm_data import llm_models_root
from utils.util import getSMVersion

import tensorrt_llm
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.attention_backend.sparse.rocket import (
    RocketKVCacheManager, RocketTrtllmAttention, RocketTrtllmAttentionMetadata,
    RocketVanillaAttention, RocketVanillaAttentionMetadata)
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm.llmapi import (CudaGraphConfig, KvCacheConfig,
                                 RocketSparseAttentionConfig)
from tensorrt_llm.mapping import Mapping


@pytest.mark.skipif(getSMVersion() < 100,
                    reason="RocketKV requires SM100 (Blackwell)")
@pytest.mark.parametrize("backend", ["pytorch"])
@pytest.mark.parametrize("model_name",
                         ["llama-3.1-model/Llama-3.1-8B-Instruct"])
@pytest.mark.parametrize("attention_backend", ["VANILLA", "TRTLLM"])
def test_model(backend, model_name, attention_backend):
    model_dir = str(llm_models_root() / model_name)
    max_batch_size = 16
    max_output_tokens = 128
    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.7,
                                    enable_block_reuse=False)

    kt_cache_dtype = 'float8_e5m2' if attention_backend == "TRTLLM" else 'bfloat16'

    sparse_attention_config = RocketSparseAttentionConfig(
        window_size=32,
        kernel_size=63,
        prompt_budget=2048,
        kt_cache_dtype=kt_cache_dtype,
    )

    cuda_graph_config = CudaGraphConfig(
        batch_sizes=[1, 2, 4, 8, 16],
        enable_padding=True,
    )

    llm = LLM(
        model=model_dir,
        backend=backend,
        kv_cache_config=kv_cache_config,
        attn_backend=attention_backend,
        sparse_attention_config=sparse_attention_config,
        max_batch_size=max_batch_size,
        max_seq_len=20480,
        max_num_tokens=81920,
        cuda_graph_config=None
        if attention_backend == "VANILLA" else cuda_graph_config,
    )

    inputs, references = [], []
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(os.path.dirname(
        os.path.dirname(current_file)))
    input_file = f'{current_dir}/multi_gpu/NIAH_simple_data.jsonl'
    with open(input_file, 'r') as f:
        for line in f:
            sample = json.loads(line)
            inputs.append({
                'prompt':
                sample['input_context'] + sample['input_query'],
            })
            references.append(sample['outputs'][0])

    with llm:
        outputs = llm.generate(
            inputs,
            use_tqdm=True,
            sampling_params=SamplingParams(add_special_tokens=False,
                                           max_tokens=max_output_tokens,
                                           temperature=0.8,
                                           top_p=0.95),
        )

    count = 0
    for ref, ret in zip(references, outputs):
        print(f"ret: {ret.outputs[0].text}")
        print(f"ref: {ref}")
        if ref not in ret.outputs[0].text:
            print(f'reference {ref} is not in the output {ret.outputs[0].text}')
        else:
            count = count + 1
    acc = count / len(outputs)

    assert acc >= 0.9, 'accuracy test of rocketkv sparse attention failed'


def create_rocket_kv_cache_manager(num_layers, num_kv_heads, head_dim,
                                   tokens_per_block, max_seq_len,
                                   max_batch_size, dtype, sparse_attn_config):
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    num_blocks = 100
    kv_cache_config = KvCacheConfig(max_tokens=num_blocks * tokens_per_block,
                                    enable_block_reuse=False)

    kv_cache_manager = RocketKVCacheManager(
        kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        mapping=mapping,
        dtype=dtype,
        sparse_attn_config=sparse_attn_config,
    )
    return kv_cache_manager


def create_test_metadata(seq_lens, num_contexts, past_seen_tokens, request_ids,
                         kv_cache_manager, sparse_attn_config, metadata_cls):
    prompt_lens = []
    for i, (seq_len, past_token) in enumerate(zip(seq_lens, past_seen_tokens)):
        if i < num_contexts:
            prompt_lens.append(seq_len)
        else:
            prompt_lens.append(past_token + seq_len)

    metadata = metadata_cls(
        seq_lens=torch.tensor(seq_lens, dtype=torch.int),
        num_contexts=num_contexts,
        kv_cache_params=KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=past_seen_tokens),
        max_num_requests=len(seq_lens),
        max_num_sequences=len(seq_lens),
        max_num_tokens=8192,
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
        prompt_lens=prompt_lens,
        sparse_attention_config=sparse_attn_config,
    )
    metadata.prepare()
    return metadata


@pytest.mark.parametrize(
    "batch_size,num_contexts",
    [
        (1, 1),  # bs=1
        (4, 4),  # bs=2, context only (2 contexts)
        (6, 3),  # bs=6, mixed (3 contexts + 3 generations)
    ])
def test_sparse_kv_predict(batch_size, num_contexts):
    """
    Test sparse_kv_predict against vanilla _get_snapkv_indices.

    This test verifies that the batched implementation produces the same results
    as the single-request implementation for SnapKV sparse attention.
    """

    # Test configuration
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    device = torch.device('cuda')
    dtype = torch.bfloat16

    sparse_attn_config = RocketSparseAttentionConfig(
        window_size=32,
        kernel_size=3,
        prompt_budget=256,
        page_size=4,
        kt_cache_dtype='bfloat16',
    )

    # Create sequence lengths - mix short and long sequences in context phase
    seq_lens = []
    past_seen_tokens = []
    for i in range(batch_size):
        if i < num_contexts:
            # Context phase: mix sequences shorter and longer than prompt_budget
            if i % 2 == 1 and batch_size > 1:
                # Short sequence: seq_len < prompt_budget
                seq_lens.append(
                    torch.randint(sparse_attn_config.prompt_budget // 2,
                                  sparse_attn_config.prompt_budget - 10,
                                  (1, )).item())
            else:
                # Long sequence: seq_len > prompt_budget
                seq_lens.append(
                    torch.randint(sparse_attn_config.prompt_budget,
                                  sparse_attn_config.prompt_budget + 200,
                                  (1, )).item())
            past_seen_tokens.append(0)
        else:
            # Generation phase: single token
            seq_lens.append(1)
            past_seen_tokens.append(torch.randint(100, 200, (1, )).item())

    request_ids = list(range(batch_size))

    num_layers = 1
    tokens_per_block = 64
    max_seq_len = 4096
    if dtype == torch.float16:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
    elif dtype == torch.bfloat16:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
    else:
        raise ValueError("Invalid dtype")

    vanilla_tokens_per_block = max_seq_len  # Each sequence in one block

    trtllm_kv_cache_manager = create_rocket_kv_cache_manager(
        num_layers, num_kv_heads, head_dim, tokens_per_block, max_seq_len,
        batch_size, kv_cache_dtype, sparse_attn_config)
    vanilla_kv_cache_manager = create_rocket_kv_cache_manager(
        num_layers, num_kv_heads, head_dim, vanilla_tokens_per_block,
        max_seq_len, batch_size, kv_cache_dtype, sparse_attn_config)

    # Add dummy requests to both cache managers
    token_nums = [
        seq_len + past_token
        for seq_len, past_token in zip(seq_lens, past_seen_tokens)
    ]
    trtllm_kv_cache_manager.add_dummy_requests(request_ids, token_nums)
    vanilla_kv_cache_manager.add_dummy_requests(request_ids, token_nums)

    trtllm_attn = RocketTrtllmAttention(
        layer_idx=0,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        sparse_attention_config=sparse_attn_config,
    )

    vanilla_attn = RocketVanillaAttention(
        layer_idx=0,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        sparse_attention_config=sparse_attn_config,
    )

    trtllm_metadata = create_test_metadata(seq_lens, num_contexts,
                                           past_seen_tokens, request_ids,
                                           trtllm_kv_cache_manager,
                                           sparse_attn_config,
                                           RocketTrtllmAttentionMetadata)
    vanilla_metadata = create_test_metadata(seq_lens, num_contexts,
                                            past_seen_tokens, request_ids,
                                            vanilla_kv_cache_manager,
                                            sparse_attn_config,
                                            RocketVanillaAttentionMetadata)

    total_tokens = sum(seq_lens)
    qkv = torch.randn(total_tokens, (num_heads + 2 * num_kv_heads) * head_dim,
                      dtype=dtype,
                      device=device)

    trtllm_sparse_kv_indices, trtllm_sparse_kv_offsets = trtllm_attn.sparse_kv_predict(
        qkv, None, trtllm_metadata)

    vanilla_sparse_kv_indices_list = []
    offset = 0
    for i in range(num_contexts):
        seq_len = seq_lens[i]
        single_qkv = qkv[offset:offset + seq_len]
        q, k, _ = single_qkv.split([
            num_heads * head_dim, num_kv_heads * head_dim,
            num_kv_heads * head_dim
        ],
                                   dim=-1)
        q = q.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(1, seq_len, num_kv_heads, head_dim)

        if seq_len <= sparse_attn_config.prompt_budget:
            # Short sequences: vanilla returns None, but trtllm returns [0, 1, ..., seq_len-1]
            # Generate expected indices for comparison
            short_indices = torch.arange(seq_len,
                                         device=device,
                                         dtype=torch.int32).unsqueeze(0).expand(
                                             num_kv_heads, -1)
            vanilla_sparse_kv_indices_list.append(short_indices)
        else:
            vanilla_indices = vanilla_attn._get_snapkv_indices(q, k, i)
            if vanilla_indices is not None:
                vanilla_indices = vanilla_indices.squeeze(0).transpose(
                    0, 1).contiguous()
                vanilla_sparse_kv_indices_list.append(vanilla_indices)

        offset += seq_len

    if len(vanilla_sparse_kv_indices_list) > 0:
        vanilla_sparse_kv_indices = torch.cat(vanilla_sparse_kv_indices_list,
                                              dim=-1).contiguous()
    else:
        vanilla_sparse_kv_indices = None

    # Compare results
    if trtllm_sparse_kv_indices is not None:
        assert vanilla_sparse_kv_indices is not None, "Vanilla should also produce indices"

        assert trtllm_sparse_kv_indices.shape == vanilla_sparse_kv_indices.shape, \
            f"Shape mismatch: {trtllm_sparse_kv_indices.shape} vs {vanilla_sparse_kv_indices.shape}"

        # Check indices overlap per batch and per head
        num_kv_heads = trtllm_sparse_kv_indices.shape[0]

        # trtllm_sparse_kv_offsets tells where each batch's indices start/end
        trtllm_offsets = trtllm_sparse_kv_offsets.cpu().tolist()

        overlap_ratios = []
        batch_overlap_details = []

        for batch_idx in range(num_contexts):
            start_idx = trtllm_offsets[batch_idx]
            end_idx = trtllm_offsets[batch_idx + 1]
            end_idx - start_idx

            batch_overlaps = []
            for head_idx in range(num_kv_heads):
                trtllm_batch = trtllm_sparse_kv_indices[
                    head_idx, start_idx:end_idx].cpu().tolist()
                vanilla_batch = vanilla_sparse_kv_indices[
                    head_idx, start_idx:end_idx].cpu().tolist()

                trtllm_set = set(trtllm_batch)
                vanilla_set = set(vanilla_batch)

                # Calculate overlap
                overlap = len(vanilla_set & trtllm_set)
                overlap_ratio = overlap / len(vanilla_set) if len(
                    vanilla_set) > 0 else 1.0
                batch_overlaps.append(overlap_ratio)
                overlap_ratios.append(overlap_ratio)

            avg_batch_overlap = sum(batch_overlaps) / len(batch_overlaps)
            batch_overlap_details.append(
                f"Batch {batch_idx}: {avg_batch_overlap:.4f}")

        avg_overlap_ratio = sum(overlap_ratios) / len(overlap_ratios)
        print(f"Average overlap ratio: {avg_overlap_ratio:.4f}")
        print(f"Per-batch average: {batch_overlap_details}")

        assert avg_overlap_ratio >= 0.98, \
            f"Indices overlap ratio {avg_overlap_ratio:.4f} is too low (< 0.98)"
    else:
        assert vanilla_sparse_kv_indices is None, "Both should return None when no sparse attention is needed"


@pytest.mark.parametrize(
    "batch_size,num_contexts",
    [
        (1, 0),  # bs=1, generation only (1 generation)
        (2, 0),  # bs=2, generation only (2 generations)
        (3, 0),  # bs=3, generation only (3 generations)
        (5, 3),  # bs=5, mixed (3 contexts + 2 generations)
        (6, 2),  # bs=6, mixed (2 ctx + 4 gen)
    ])
def test_sparse_attn_predict(batch_size, num_contexts):
    """Test sparse_attn_predict against vanilla _rocketkv_selection."""
    num_generations = batch_size - num_contexts

    # Test configuration
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    device = torch.device('cuda')
    dtype = torch.bfloat16

    sparse_attn_config_vanilla = RocketSparseAttentionConfig(
        window_size=32,
        kernel_size=3,
        prompt_budget=256,
        page_size=2,
        topk=128,
        topr=96,
        kt_cache_dtype='bfloat16',
    )
    sparse_attn_config_trtllm = RocketSparseAttentionConfig(
        window_size=32,
        kernel_size=3,
        prompt_budget=256,
        page_size=2,
        topk=64,
        topr=96,
        kt_cache_dtype='bfloat16',
    )

    # Create sequence lengths
    seq_lens = []
    past_seen_tokens = []
    for i in range(batch_size):
        if i < num_contexts:
            # Context phase: longer sequences
            seq_lens.append(torch.randint(300, 400, (1, )).item())
            past_seen_tokens.append(0)
        else:
            # Generation phase: single token
            seq_lens.append(1)
            # 128 is the minimum number of tokens for shape alignment
            past_seen_tokens.append(torch.randint(128, 300, (1, )).item())

    request_ids = list(range(batch_size))

    num_layers = 1
    tokens_per_block = 64
    max_seq_len = 4096
    if dtype == torch.float16:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
    elif dtype == torch.bfloat16:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
    else:
        raise ValueError("Invalid dtype")

    vanilla_tokens_per_block = max_seq_len  # Each sequence in one block

    trtllm_kv_cache_manager = create_rocket_kv_cache_manager(
        num_layers, num_kv_heads, head_dim, tokens_per_block, max_seq_len,
        batch_size, kv_cache_dtype, sparse_attn_config_trtllm)
    vanilla_kv_cache_manager = create_rocket_kv_cache_manager(
        num_layers, num_kv_heads, head_dim, vanilla_tokens_per_block,
        max_seq_len, batch_size, kv_cache_dtype, sparse_attn_config_vanilla)

    # Add dummy requests to both cache managers
    token_nums = [
        seq_len + past_token
        for seq_len, past_token in zip(seq_lens, past_seen_tokens)
    ]
    trtllm_kv_cache_manager.add_dummy_requests(request_ids, token_nums)
    vanilla_kv_cache_manager.add_dummy_requests(request_ids, token_nums)

    trtllm_attn = RocketTrtllmAttention(
        layer_idx=0,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        sparse_attention_config=sparse_attn_config_trtllm,
    )

    vanilla_attn = RocketVanillaAttention(
        layer_idx=0,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        sparse_attention_config=sparse_attn_config_vanilla,
    )

    trtllm_metadata = create_test_metadata(seq_lens, num_contexts,
                                           past_seen_tokens, request_ids,
                                           trtllm_kv_cache_manager,
                                           sparse_attn_config_trtllm,
                                           RocketTrtllmAttentionMetadata)
    vanilla_metadata = create_test_metadata(seq_lens, num_contexts,
                                            past_seen_tokens, request_ids,
                                            vanilla_kv_cache_manager,
                                            sparse_attn_config_vanilla,
                                            RocketVanillaAttentionMetadata)

    total_tokens = sum(seq_lens)
    qkv = torch.randn(total_tokens, (num_heads + 2 * num_kv_heads) * head_dim,
                      dtype=dtype,
                      device=device)

    for layer_idx in range(num_layers):
        trtllm_kt_buf = trtllm_kv_cache_manager.get_kt_buffers(layer_idx)
        vanilla_kt_buf = vanilla_kv_cache_manager.get_kt_buffers(layer_idx)

        if trtllm_kt_buf.dtype == torch.float8_e5m2:
            temp_buf = torch.empty_like(trtllm_kt_buf, dtype=torch.float16)
            torch.nn.init.normal_(temp_buf)
            trtllm_kt_buf.copy_(temp_buf.to(trtllm_kt_buf.dtype))
        else:
            torch.nn.init.normal_(trtllm_kt_buf)

        # Map trtllm data to vanilla based on block offsets
        # TRTLLM: (num_blocks, kt_tokens_per_block, num_kv_heads, 2*head_dim)
        # VANILLA: (num_blocks, kt_tokens_per_block_vanilla, num_kv_heads, 2*head_dim)
        trtllm_kt_tokens_per_block = trtllm_kv_cache_manager.kt_tokens_per_block
        vanilla_kv_cache_manager.kt_tokens_per_block

        trtllm_block_offsets = trtllm_metadata.kt_cache_block_offsets
        vanilla_block_offsets = vanilla_metadata.kt_cache_block_offsets

        for req_idx in range(num_contexts, batch_size):
            # Get the number of KT tokens for this request
            past_token = past_seen_tokens[req_idx]
            num_kt_tokens = (past_token + 1 +
                             sparse_attn_config_trtllm.page_size -
                             1) // sparse_attn_config_trtllm.page_size

            # Get block offsets for this request
            trtllm_blocks = trtllm_block_offsets[req_idx]
            vanilla_blocks = vanilla_block_offsets[req_idx]

            # Copy data from trtllm blocks to vanilla blocks
            kt_token_idx = 0
            vanilla_block_idx = 0

            # For trtllm: iterate through blocks and copy KT tokens
            for trtllm_block_local_idx in range(len(trtllm_blocks)):
                if kt_token_idx >= num_kt_tokens:
                    break

                trtllm_block = trtllm_blocks[trtllm_block_local_idx]
                if trtllm_block < 0:
                    break

                # How many KT tokens in this trtllm block
                kt_tokens_in_this_block = min(trtllm_kt_tokens_per_block,
                                              num_kt_tokens - kt_token_idx)

                # Copy to vanilla buffer
                vanilla_block = vanilla_blocks[vanilla_block_idx]
                if vanilla_block >= 0:
                    vanilla_kt_buf[
                        vanilla_block, kt_token_idx:kt_token_idx +
                        kt_tokens_in_this_block].copy_(trtllm_kt_buf[
                            trtllm_block, :kt_tokens_in_this_block].to(
                                vanilla_kt_buf.dtype))

                kt_token_idx += kt_tokens_in_this_block

    trtllm_sparse_attn_indices, trtllm_sparse_attn_offsets = trtllm_attn.sparse_attn_predict(
        qkv, None, trtllm_metadata)

    vanilla_sparse_attn_indices_list = []
    offset = sum(seq_lens[:num_contexts])  # Skip context tokens

    for i in range(num_contexts, batch_size):
        seq_len = seq_lens[i]
        single_qkv = qkv[offset:offset + seq_len]
        q, k, _ = single_qkv.split([
            num_heads * head_dim, num_kv_heads * head_dim,
            num_kv_heads * head_dim
        ],
                                   dim=-1)
        q = q.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(1, seq_len, num_kv_heads, head_dim)

        past_seen_token = past_seen_tokens[i]
        vanilla_indices = vanilla_attn._rocketkv_selection(
            q, k, vanilla_metadata, past_seen_token, i)
        vanilla_sparse_attn_indices_list.append(vanilla_indices.squeeze(0))
        offset += seq_len

    if trtllm_sparse_attn_indices is not None:
        assert len(vanilla_sparse_attn_indices_list
                   ) > 0, "Vanilla should also produce indices"

        vanilla_sparse_attn_indices = torch.cat(
            vanilla_sparse_attn_indices_list, dim=0).transpose(0,
                                                               1).contiguous()

        # Apply interleave operation to trtllm indices
        # For each head, multiply indices by page_size and expand to include all tokens in each page
        page_size = sparse_attn_config_trtllm.page_size
        num_kv_heads, total_indices = trtllm_sparse_attn_indices.shape

        interleaved_indices_list = []
        for head_idx in range(num_kv_heads):
            head_indices = trtllm_sparse_attn_indices[
                head_idx]  # Shape: [total_indices]

            page_starts = head_indices * page_size  # Shape: [total_indices]

            expanded_indices = []
            for page_start in page_starts:
                page_indices = torch.arange(page_start,
                                            page_start + page_size,
                                            device=page_starts.device)
                expanded_indices.append(page_indices)

            head_interleaved = torch.cat(expanded_indices, dim=0)

            # Slice to match vanilla shape
            target_length = vanilla_sparse_attn_indices.shape[1]
            head_interleaved = head_interleaved[:target_length]

            interleaved_indices_list.append(head_interleaved)

        # Stack all heads
        trtllm_sparse_attn_indices = torch.stack(interleaved_indices_list,
                                                 dim=0)

        assert trtllm_sparse_attn_indices.shape == vanilla_sparse_attn_indices.shape, \
            f"Shape mismatch: {trtllm_sparse_attn_indices.shape} vs {vanilla_sparse_attn_indices.shape}"

        trtllm_sparse_attn_indices = trtllm_sparse_attn_indices.sort(
            dim=-1).values
        vanilla_sparse_attn_indices = vanilla_sparse_attn_indices.sort(
            dim=-1).values

        # Check indices overlap per batch and per head
        num_kv_heads = trtllm_sparse_attn_indices.shape[0]

        trtllm_offsets = trtllm_sparse_attn_offsets.cpu().tolist()

        overlap_ratios = []
        batch_overlap_details = []

        num_generations = batch_size - num_contexts
        for batch_idx in range(num_generations):
            start_idx = trtllm_offsets[batch_idx]
            end_idx = trtllm_offsets[batch_idx + 1]
            end_idx - start_idx

            batch_overlaps = []
            for head_idx in range(num_kv_heads):
                trtllm_batch = trtllm_sparse_attn_indices[
                    head_idx, start_idx:end_idx].cpu().tolist()
                vanilla_batch = vanilla_sparse_attn_indices[
                    head_idx, start_idx:end_idx].cpu().tolist()

                trtllm_set = set(trtllm_batch)
                vanilla_set = set(vanilla_batch)

                # Calculate overlap
                overlap = len(vanilla_set & trtllm_set)
                overlap_ratio = overlap / len(vanilla_set) if len(
                    vanilla_set) > 0 else 1.0
                batch_overlaps.append(overlap_ratio)
                overlap_ratios.append(overlap_ratio)

            avg_batch_overlap = sum(batch_overlaps) / len(batch_overlaps)
            batch_overlap_details.append(
                f"Batch {batch_idx}: {avg_batch_overlap:.4f}")

        avg_overlap_ratio = sum(overlap_ratios) / len(overlap_ratios)
        print(f"Average overlap ratio: {avg_overlap_ratio:.4f}")
        print(f"Per-batch average: {batch_overlap_details}")

        threshold = 0.94
        assert avg_overlap_ratio >= threshold, \
            f"Indices overlap ratio {avg_overlap_ratio:.4f} is too low (< {threshold})"
    else:
        assert len(
            vanilla_sparse_attn_indices_list
        ) == 0, "Both should return None when no sparse attention is needed"


if __name__ == '__main__':
    # RocketKV e2e tests
    print("=== Testing RocketKV E2E tests ===")
    test_model("pytorch", "llama-3.1-model/Llama-3.1-8B-Instruct", "VANILLA")
    test_model("pytorch", "llama-3.1-model/Llama-3.1-8B-Instruct", "TRTLLM")

    # Unit tests for sparse_kv_predict
    print("\n=== Testing sparse_kv_predict ===")
    test_sparse_kv_predict(1, 1)  # bs=1, context only
    test_sparse_kv_predict(2, 2)  # bs=2, context only
    test_sparse_kv_predict(6, 3)  # bs=6, mixed (3 ctx + 3 gen)

    # Unit tests for sparse_attn_predict
    print("\n=== Testing sparse_attn_predict ===")
    test_sparse_attn_predict(1, 0)  # bs=1, generation only
    test_sparse_attn_predict(2, 0)  # bs=2, generation only
    test_sparse_attn_predict(3, 0)  # bs=3, generation only
    test_sparse_attn_predict(5, 3)  # bs=5, mixed (3 ctx + 2 gen)
    test_sparse_attn_predict(6, 2)  # bs=6, mixed (2 ctx + 4 gen)

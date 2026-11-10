# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math
import os

import pytest
import torch
from utils.llm_data import llm_models_root
from utils.util import getSMVersion

import tensorrt_llm
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.attention.backends.interface import AttentionForwardArgs
from tensorrt_llm._torch.attention.backends.sparse.rocket import (
    RocketKVCacheManager,
    RocketKVCacheManagerV2,
    RocketTrtllmAttention,
    RocketTrtllmAttentionMetadata,
    RocketVanillaAttention,
    RocketVanillaAttentionMetadata,
)
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import get_size_in_bytes
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, RocketSparseAttentionConfig
from tensorrt_llm.mapping import Mapping


@pytest.mark.skipif(getSMVersion() < 100, reason="RocketKV requires SM100 (Blackwell)")
@pytest.mark.parametrize("backend", ["pytorch"])
@pytest.mark.parametrize("model_name", ["llama-3.1-model/Llama-3.1-8B-Instruct"])
@pytest.mark.parametrize("attention_backend", ["VANILLA", "TRTLLM"])
def test_model(backend, model_name, attention_backend, monkeypatch):
    # RocketKV is a single-GPU path. Keep this test independent of MPI
    # dynamic-process bootstrap so a cluster launch failure cannot mask the
    # attention result.
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    # The single-process worker builds the cache managers in this process;
    # record them to check that the default KV cache manager V2 is used.
    created_manager_types = []
    for manager_cls in (RocketKVCacheManager, RocketKVCacheManagerV2):

        def record_init(self, *args, _init=manager_cls.__init__, **kwargs):
            created_manager_types.append(type(self))
            _init(self, *args, **kwargs)

        monkeypatch.setattr(manager_cls, "__init__", record_init)
    model_dir = str(llm_models_root() / model_name)
    max_batch_size = 16
    max_output_tokens = 128
    # The answer takes a few tokens. On TRTLLM, decode past it so that every
    # request's KV and KT caches grow over several KV cache blocks; Vanilla
    # decodes too slowly for that.
    decode_all_tokens = attention_backend == "TRTLLM"
    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.7, enable_block_reuse=False)

    kt_cache_dtype = "float8_e5m2" if attention_backend == "TRTLLM" else "bfloat16"

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
        cuda_graph_config=None if attention_backend == "VANILLA" else cuda_graph_config,
    )

    inputs, references = [], []
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
    input_file = f"{current_dir}/multi_gpu/NIAH_simple_data.jsonl"
    with open(input_file, "r") as f:
        for line in f:
            sample = json.loads(line)
            inputs.append(
                {
                    "prompt": sample["input_context"] + sample["input_query"],
                }
            )
            references.append(sample["outputs"][0])

    with llm:
        outputs = llm.generate(
            inputs,
            use_tqdm=True,
            sampling_params=SamplingParams(
                add_special_tokens=False,
                max_tokens=max_output_tokens,
                temperature=0.8,
                top_p=0.95,
                ignore_eos=decode_all_tokens,
            ),
        )

    misses = [
        (ref, ret.outputs[0].text)
        for ref, ret in zip(references, outputs)
        if ref not in ret.outputs[0].text
    ]
    acc = 1 - len(misses) / len(outputs)

    assert created_manager_types
    assert set(created_manager_types) == {RocketKVCacheManagerV2}
    if decode_all_tokens:
        assert [len(ret.outputs[0].token_ids) for ret in outputs] == (
            [max_output_tokens] * len(outputs)
        )
    assert acc >= 0.9, (
        f"RocketKV NIAH accuracy {acc:.4f} < 0.9; missed (reference, output): {misses}"
    )


def create_rocket_kv_cache_manager(
    num_layers,
    num_kv_heads,
    head_dim,
    tokens_per_block,
    max_seq_len,
    max_batch_size,
    dtype,
    sparse_attn_config,
    use_kv_cache_manager_v2=False,
    **manager_kwargs,
):
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    num_blocks = 100
    kv_cache_config = KvCacheConfig(
        max_tokens=num_blocks * tokens_per_block, enable_block_reuse=False
    )

    manager_cls = RocketKVCacheManagerV2 if use_kv_cache_manager_v2 else RocketKVCacheManager
    kv_cache_manager = manager_cls(
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
        sparse_attention_config=sparse_attn_config,
        **manager_kwargs,
    )
    return kv_cache_manager


@pytest.fixture
def make_rocket_kv_cache_manager():
    """Create RocketKV cache managers that are shut down when the test ends."""
    managers = []

    def make(*args, **kwargs):
        manager = create_rocket_kv_cache_manager(*args, **kwargs)
        managers.append(manager)
        return manager

    yield make
    for manager in managers:
        manager.shutdown()


def create_test_metadata(
    seq_lens,
    num_contexts,
    past_seen_tokens,
    request_ids,
    kv_cache_manager,
    sparse_attn_config,
    metadata_cls,
):
    sparse_metadata_params = sparse_attn_config.to_sparse_metadata_params()
    prompt_lens = []
    for i, (seq_len, past_token) in enumerate(zip(seq_lens, past_seen_tokens)):
        if i < num_contexts:
            prompt_lens.append(seq_len)
        else:
            prompt_lens.append(past_token + seq_len)

    metadata = metadata_cls(
        seq_lens=torch.tensor(seq_lens, dtype=torch.int),
        num_contexts=num_contexts,
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=past_seen_tokens),
        max_num_requests=len(seq_lens),
        max_num_sequences=len(seq_lens),
        max_num_tokens=8192,
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
        prompt_lens=prompt_lens,
        sparse_metadata_params=sparse_metadata_params,
    )
    metadata.prepare()
    return metadata


@pytest.mark.parametrize(
    "batch_size,num_contexts",
    [
        (1, 1),  # bs=1
        (4, 4),  # bs=2, context only (2 contexts)
        (6, 3),  # bs=6, mixed (3 contexts + 3 generations)
    ],
)
@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True], ids=["v1", "v2"])
def test_sparse_kv_predict(
    batch_size, num_contexts, use_kv_cache_manager_v2, make_rocket_kv_cache_manager
):
    """
    Test sparse_kv_predict against vanilla _get_snapkv_indices.

    This test verifies that the batched implementation produces the same results
    as the single-request implementation for SnapKV sparse attention.
    """

    # Test configuration
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    device = torch.device("cuda")
    dtype = torch.bfloat16

    sparse_attn_config = RocketSparseAttentionConfig(
        window_size=32,
        kernel_size=3,
        prompt_budget=256,
        page_size=4,
        kt_cache_dtype="bfloat16",
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
                    torch.randint(
                        sparse_attn_config.prompt_budget // 2,
                        sparse_attn_config.prompt_budget - 10,
                        (1,),
                    ).item()
                )
            else:
                # Long sequence: seq_len > prompt_budget
                seq_lens.append(
                    torch.randint(
                        sparse_attn_config.prompt_budget,
                        sparse_attn_config.prompt_budget + 200,
                        (1,),
                    ).item()
                )
            past_seen_tokens.append(0)
        else:
            # Generation phase: single token
            seq_lens.append(1)
            past_seen_tokens.append(torch.randint(100, 200, (1,)).item())

    request_ids = list(range(batch_size))

    num_layers = 2
    tokens_per_block = 64
    max_seq_len = 4096
    if dtype == torch.float16:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
    elif dtype == torch.bfloat16:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
    else:
        raise ValueError("Invalid dtype")

    vanilla_tokens_per_block = max_seq_len  # Each sequence in one block

    trtllm_kv_cache_manager = make_rocket_kv_cache_manager(
        num_layers,
        num_kv_heads,
        head_dim,
        tokens_per_block,
        max_seq_len,
        batch_size,
        kv_cache_dtype,
        sparse_attn_config,
        use_kv_cache_manager_v2,
    )
    vanilla_kv_cache_manager = make_rocket_kv_cache_manager(
        num_layers,
        num_kv_heads,
        head_dim,
        vanilla_tokens_per_block,
        max_seq_len,
        batch_size,
        kv_cache_dtype,
        sparse_attn_config,
        use_kv_cache_manager_v2,
    )

    # Add dummy requests to both cache managers
    token_nums = [seq_len + past_token for seq_len, past_token in zip(seq_lens, past_seen_tokens)]
    trtllm_kv_cache_manager.add_dummy_requests(request_ids, token_nums)
    vanilla_kv_cache_manager.add_dummy_requests(request_ids, token_nums)
    sparse_params = sparse_attn_config.to_sparse_params()

    trtllm_attn = RocketTrtllmAttention(
        layer_idx=0,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        sparse_params=sparse_params,
    )

    vanilla_attn = RocketVanillaAttention(
        layer_idx=0,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        sparse_params=sparse_params,
    )

    trtllm_metadata = create_test_metadata(
        seq_lens,
        num_contexts,
        past_seen_tokens,
        request_ids,
        trtllm_kv_cache_manager,
        sparse_attn_config,
        RocketTrtllmAttentionMetadata,
    )
    _ = create_test_metadata(
        seq_lens,
        num_contexts,
        past_seen_tokens,
        request_ids,
        vanilla_kv_cache_manager,
        sparse_attn_config,
        RocketVanillaAttentionMetadata,
    )

    total_tokens = sum(seq_lens)
    qkv = torch.randn(
        total_tokens, (num_heads + 2 * num_kv_heads) * head_dim, dtype=dtype, device=device
    )
    forward_args = AttentionForwardArgs()

    trtllm_sparse_kv_indices, trtllm_sparse_kv_offsets = trtllm_attn.sparse_kv_predict(
        qkv, None, trtllm_metadata, forward_args
    )

    vanilla_sparse_kv_indices_list = []
    offset = 0
    for i in range(num_contexts):
        seq_len = seq_lens[i]
        single_qkv = qkv[offset : offset + seq_len]
        q, k, _ = single_qkv.split(
            [num_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim], dim=-1
        )
        q = q.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(1, seq_len, num_kv_heads, head_dim)

        if seq_len <= sparse_attn_config.prompt_budget:
            # Short sequences: vanilla returns None, but trtllm returns [0, 1, ..., seq_len-1]
            # Generate expected indices for comparison
            short_indices = (
                torch.arange(seq_len, device=device, dtype=torch.int32)
                .unsqueeze(0)
                .expand(num_kv_heads, -1)
            )
            vanilla_sparse_kv_indices_list.append(short_indices)
        else:
            vanilla_indices = vanilla_attn._get_snapkv_indices(q, k, i)
            if vanilla_indices is not None:
                vanilla_indices = vanilla_indices.squeeze(0).transpose(0, 1).contiguous()
                vanilla_sparse_kv_indices_list.append(vanilla_indices)

        offset += seq_len

    if len(vanilla_sparse_kv_indices_list) > 0:
        vanilla_sparse_kv_indices = torch.cat(vanilla_sparse_kv_indices_list, dim=-1).contiguous()
    else:
        vanilla_sparse_kv_indices = None

    # Compare results
    if trtllm_sparse_kv_indices is not None:
        assert vanilla_sparse_kv_indices is not None, "Vanilla should also produce indices"

        assert trtllm_sparse_kv_indices.shape == vanilla_sparse_kv_indices.shape, (
            f"Shape mismatch: {trtllm_sparse_kv_indices.shape} vs {vanilla_sparse_kv_indices.shape}"
        )

        # Check indices overlap per batch and per head
        num_kv_heads = trtllm_sparse_kv_indices.shape[0]

        # trtllm_sparse_kv_offsets tells where each batch's indices start/end
        trtllm_offsets = trtllm_sparse_kv_offsets.cpu().tolist()

        overlap_ratios = []
        batch_overlap_details = []

        for batch_idx in range(num_contexts):
            start_idx = trtllm_offsets[batch_idx]
            end_idx = trtllm_offsets[batch_idx + 1]

            batch_overlaps = []
            for head_idx in range(num_kv_heads):
                trtllm_batch = trtllm_sparse_kv_indices[head_idx, start_idx:end_idx].cpu().tolist()
                vanilla_batch = (
                    vanilla_sparse_kv_indices[head_idx, start_idx:end_idx].cpu().tolist()
                )

                trtllm_set = set(trtllm_batch)
                vanilla_set = set(vanilla_batch)

                # Calculate overlap
                overlap = len(vanilla_set & trtllm_set)
                overlap_ratio = overlap / len(vanilla_set) if len(vanilla_set) > 0 else 1.0
                batch_overlaps.append(overlap_ratio)
                overlap_ratios.append(overlap_ratio)

            avg_batch_overlap = sum(batch_overlaps) / len(batch_overlaps)
            batch_overlap_details.append(f"Batch {batch_idx}: {avg_batch_overlap:.4f}")

        avg_overlap_ratio = sum(overlap_ratios) / len(overlap_ratios)
        print(f"Average overlap ratio: {avg_overlap_ratio:.4f}")
        print(f"Per-batch average: {batch_overlap_details}")

        assert avg_overlap_ratio >= 0.98, (
            f"Indices overlap ratio {avg_overlap_ratio:.4f} is too low (< 0.98)"
        )
    else:
        assert vanilla_sparse_kv_indices is None, (
            "Both should return None when no sparse attention is needed"
        )


@pytest.mark.parametrize(
    "batch_size,num_contexts",
    [
        (1, 0),  # bs=1, generation only (1 generation)
        (2, 0),  # bs=2, generation only (2 generations)
        (3, 0),  # bs=3, generation only (3 generations)
        (5, 3),  # bs=5, mixed (3 contexts + 2 generations)
        (6, 2),  # bs=6, mixed (2 ctx + 4 gen)
    ],
)
@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True], ids=["v1", "v2"])
def test_sparse_attn_predict(
    batch_size, num_contexts, use_kv_cache_manager_v2, make_rocket_kv_cache_manager
):
    """Test sparse_attn_predict against vanilla _rocketkv_selection."""
    num_generations = batch_size - num_contexts

    # Test configuration
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    device = torch.device("cuda")
    dtype = torch.bfloat16

    sparse_attn_config_vanilla = RocketSparseAttentionConfig(
        window_size=32,
        kernel_size=3,
        prompt_budget=256,
        page_size=2,
        topk=128,
        topr=96,
        kt_cache_dtype="bfloat16",
    )
    sparse_attn_config_trtllm = RocketSparseAttentionConfig(
        window_size=32,
        kernel_size=3,
        prompt_budget=256,
        page_size=2,
        topk=64,
        topr=96,
        kt_cache_dtype="bfloat16",
    )

    # Create sequence lengths
    seq_lens = []
    past_seen_tokens = []
    for i in range(batch_size):
        if i < num_contexts:
            # Context phase: longer sequences
            seq_lens.append(torch.randint(300, 400, (1,)).item())
            past_seen_tokens.append(0)
        else:
            # Generation phase: single token
            seq_lens.append(1)
            # 128 is the minimum number of tokens for shape alignment
            past_seen_tokens.append(torch.randint(128, 300, (1,)).item())

    request_ids = list(range(batch_size))

    num_layers = 2
    tokens_per_block = 64
    max_seq_len = 4096
    if dtype == torch.float16:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
    elif dtype == torch.bfloat16:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
    else:
        raise ValueError("Invalid dtype")

    vanilla_tokens_per_block = max_seq_len  # Each sequence in one block

    trtllm_kv_cache_manager = make_rocket_kv_cache_manager(
        num_layers,
        num_kv_heads,
        head_dim,
        tokens_per_block,
        max_seq_len,
        batch_size,
        kv_cache_dtype,
        sparse_attn_config_trtllm,
        use_kv_cache_manager_v2,
    )
    vanilla_kv_cache_manager = make_rocket_kv_cache_manager(
        num_layers,
        num_kv_heads,
        head_dim,
        vanilla_tokens_per_block,
        max_seq_len,
        batch_size,
        kv_cache_dtype,
        sparse_attn_config_vanilla,
        use_kv_cache_manager_v2,
    )

    # Add dummy requests to both cache managers
    token_nums = [seq_len + past_token for seq_len, past_token in zip(seq_lens, past_seen_tokens)]
    trtllm_kv_cache_manager.add_dummy_requests(request_ids, token_nums)
    vanilla_kv_cache_manager.add_dummy_requests(request_ids, token_nums)
    trtllm_sparse_params = sparse_attn_config_trtllm.to_sparse_params()
    vanilla_sparse_params = sparse_attn_config_vanilla.to_sparse_params()

    trtllm_attn = RocketTrtllmAttention(
        layer_idx=0,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        sparse_params=trtllm_sparse_params,
    )

    vanilla_attn = RocketVanillaAttention(
        layer_idx=0,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        sparse_params=vanilla_sparse_params,
    )

    trtllm_metadata = create_test_metadata(
        seq_lens,
        num_contexts,
        past_seen_tokens,
        request_ids,
        trtllm_kv_cache_manager,
        sparse_attn_config_trtllm,
        RocketTrtllmAttentionMetadata,
    )
    vanilla_metadata = create_test_metadata(
        seq_lens,
        num_contexts,
        past_seen_tokens,
        request_ids,
        vanilla_kv_cache_manager,
        sparse_attn_config_vanilla,
        RocketVanillaAttentionMetadata,
    )

    total_tokens = sum(seq_lens)
    qkv = torch.randn(
        total_tokens, (num_heads + 2 * num_kv_heads) * head_dim, dtype=dtype, device=device
    )

    # Fill every layer before copying any of them: KVCacheManagerV2 layer views
    # interleave in one KT pool, so filling a whole view also writes the pages
    # of the other layers.
    for layer_idx in range(num_layers):
        trtllm_kt_buf = trtllm_kv_cache_manager.get_kt_buffers(layer_idx)
        if trtllm_kt_buf.dtype == torch.float8_e5m2:
            temp_buf = torch.empty_like(trtllm_kt_buf, dtype=torch.float16)
            torch.nn.init.normal_(temp_buf)
            trtllm_kt_buf.copy_(temp_buf.to(trtllm_kt_buf.dtype))
        else:
            torch.nn.init.normal_(trtllm_kt_buf)

    for layer_idx in range(num_layers):
        trtllm_kt_buf = trtllm_kv_cache_manager.get_kt_buffers(layer_idx)
        vanilla_kt_buf = vanilla_kv_cache_manager.get_kt_buffers(layer_idx)

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
            num_kt_tokens = (
                past_token + 1 + sparse_attn_config_trtllm.page_size - 1
            ) // sparse_attn_config_trtllm.page_size

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
                kt_tokens_in_this_block = min(
                    trtllm_kt_tokens_per_block, num_kt_tokens - kt_token_idx
                )

                # Copy to vanilla buffer
                vanilla_block = vanilla_blocks[vanilla_block_idx]
                if vanilla_block >= 0:
                    vanilla_kt_buf[
                        vanilla_block, kt_token_idx : kt_token_idx + kt_tokens_in_this_block
                    ].copy_(
                        trtllm_kt_buf[trtllm_block, :kt_tokens_in_this_block].to(
                            vanilla_kt_buf.dtype
                        )
                    )

                kt_token_idx += kt_tokens_in_this_block

    forward_args = AttentionForwardArgs()
    trtllm_sparse_attn_indices, trtllm_sparse_attn_offsets = trtllm_attn.sparse_attn_predict(
        qkv, None, trtllm_metadata, forward_args
    )

    vanilla_sparse_attn_indices_list = []
    offset = sum(seq_lens[:num_contexts])  # Skip context tokens

    for i in range(num_contexts, batch_size):
        seq_len = seq_lens[i]
        single_qkv = qkv[offset : offset + seq_len]
        q, k, _ = single_qkv.split(
            [num_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim], dim=-1
        )
        q = q.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(1, seq_len, num_kv_heads, head_dim)

        past_seen_token = past_seen_tokens[i]
        vanilla_indices = vanilla_attn._rocketkv_selection(
            q, k, vanilla_metadata, past_seen_token, i
        )
        vanilla_sparse_attn_indices_list.append(vanilla_indices.squeeze(0))
        offset += seq_len

    if trtllm_sparse_attn_indices is not None:
        assert len(vanilla_sparse_attn_indices_list) > 0, "Vanilla should also produce indices"

        vanilla_sparse_attn_indices = (
            torch.cat(vanilla_sparse_attn_indices_list, dim=0).transpose(0, 1).contiguous()
        )

        # Apply interleave operation to trtllm indices
        # For each head, multiply indices by page_size and expand to include all tokens in each page
        page_size = sparse_attn_config_trtllm.page_size
        num_kv_heads, total_indices = trtllm_sparse_attn_indices.shape

        interleaved_indices_list = []
        for head_idx in range(num_kv_heads):
            head_indices = trtllm_sparse_attn_indices[head_idx]  # Shape: [total_indices]

            page_starts = head_indices * page_size  # Shape: [total_indices]

            expanded_indices = []
            for page_start in page_starts:
                page_indices = torch.arange(
                    page_start, page_start + page_size, device=page_starts.device
                )
                expanded_indices.append(page_indices)

            head_interleaved = torch.cat(expanded_indices, dim=0)

            # Slice to match vanilla shape
            target_length = vanilla_sparse_attn_indices.shape[1]
            head_interleaved = head_interleaved[:target_length]

            interleaved_indices_list.append(head_interleaved)

        # Stack all heads
        trtllm_sparse_attn_indices = torch.stack(interleaved_indices_list, dim=0)

        assert trtllm_sparse_attn_indices.shape == vanilla_sparse_attn_indices.shape, (
            f"Shape mismatch: {trtllm_sparse_attn_indices.shape} vs {vanilla_sparse_attn_indices.shape}"
        )

        trtllm_sparse_attn_indices = trtllm_sparse_attn_indices.sort(dim=-1).values
        vanilla_sparse_attn_indices = vanilla_sparse_attn_indices.sort(dim=-1).values

        # Check indices overlap per batch and per head
        num_kv_heads = trtllm_sparse_attn_indices.shape[0]

        trtllm_offsets = trtllm_sparse_attn_offsets.cpu().tolist()

        overlap_ratios = []
        batch_overlap_details = []

        num_generations = batch_size - num_contexts
        for batch_idx in range(num_generations):
            start_idx = trtllm_offsets[batch_idx]
            end_idx = trtllm_offsets[batch_idx + 1]

            batch_overlaps = []
            for head_idx in range(num_kv_heads):
                trtllm_batch = (
                    trtllm_sparse_attn_indices[head_idx, start_idx:end_idx].cpu().tolist()
                )
                vanilla_batch = (
                    vanilla_sparse_attn_indices[head_idx, start_idx:end_idx].cpu().tolist()
                )

                trtllm_set = set(trtllm_batch)
                vanilla_set = set(vanilla_batch)

                # Calculate overlap
                overlap = len(vanilla_set & trtllm_set)
                overlap_ratio = overlap / len(vanilla_set) if len(vanilla_set) > 0 else 1.0
                batch_overlaps.append(overlap_ratio)
                overlap_ratios.append(overlap_ratio)

            avg_batch_overlap = sum(batch_overlaps) / len(batch_overlaps)
            batch_overlap_details.append(f"Batch {batch_idx}: {avg_batch_overlap:.4f}")

        avg_overlap_ratio = sum(overlap_ratios) / len(overlap_ratios)
        print(f"Average overlap ratio: {avg_overlap_ratio:.4f}")
        print(f"Per-batch average: {batch_overlap_details}")

        threshold = 0.94
        assert avg_overlap_ratio >= threshold, (
            f"Indices overlap ratio {avg_overlap_ratio:.4f} is too low (< {threshold})"
        )
    else:
        assert len(vanilla_sparse_attn_indices_list) == 0, (
            "Both should return None when no sparse attention is needed"
        )


@pytest.mark.skipif(getSMVersion() < 100, reason="RocketKV requires SM100 (Blackwell)")
def test_rocketkv_kt_cache_sized_by_kt_dtype_not_kv_dtype():
    """get_cache_bytes_per_token() must size the KT (key-landmark) cache from the KT
    pool's own dtype, matching the pool physically allocated in __init__ -- not the
    main KV cache dtype. The old code scaled the KT term by the main KV dtype byte
    width, under-counting the KT pool by 2x when the main KV cache was FP8. Using
    FP8 main KV + bfloat16 KT makes the two dtypes differ, so this fails without the
    fix and passes with it.
    """
    sparse_attn_config = RocketSparseAttentionConfig(
        prompt_budget=2048, page_size=4, kt_cache_dtype="bfloat16"
    )
    kv_cache_manager = create_rocket_kv_cache_manager(
        num_layers=4,
        num_kv_heads=8,
        head_dim=128,
        tokens_per_block=64,
        max_seq_len=4096,
        max_batch_size=8,
        dtype=DataType.FP8,
        sparse_attn_config=sparse_attn_config,
    )
    try:
        # KT contribution: ground truth from the physically allocated KT pool.
        kt_pool_bytes = sum(
            t.numel() * t.element_size() for t in kv_cache_manager.kt_cache_pool_per_layer
        )
        tokens = kv_cache_manager.num_blocks * kv_cache_manager.tokens_per_block
        # Main K/V contribution is unchanged (sized at the main KV dtype).
        main_bytes = get_size_in_bytes(
            math.ceil(
                kv_cache_manager.kv_factor
                * sum(kv_cache_manager.num_kv_heads_per_layer)
                * kv_cache_manager.head_dim
            ),
            kv_cache_manager.dtype,
        )
        assert kv_cache_manager.get_cache_bytes_per_token() == main_bytes + kt_pool_bytes // tokens
    finally:
        kv_cache_manager.shutdown()


def test_rocket_add_dummy_requests_forwards_capture_sampling_params(mocker):
    """RocketKVCacheManager overrides add_dummy_requests with an explicit
    signature that must stay in sync with KVCacheManager.add_dummy_requests's
    capture_sampling_params kwarg, or CUDA graph warmup breaks with a
    TypeError for sparse-attention models.
    """
    mgr = object.__new__(RocketKVCacheManager)
    base_add_dummy_requests = mocker.patch.object(
        KVCacheManager, "add_dummy_requests", return_value=[]
    )
    sampling_params = SamplingParams(temperature=0.9, top_p=0.95)

    mgr.add_dummy_requests([123], capture_sampling_params=sampling_params)

    assert base_add_dummy_requests.call_args.kwargs["capture_sampling_params"] is sampling_params


@pytest.mark.parametrize(
    "kt_cache_dtype,page_size",
    [
        ("float8_e5m2", 4),
        ("bfloat16", 4),
        # A KT page as large as a K/V page.
        ("bfloat16", 2),
        # A RocketKV page longer than a KV cache block: one KT row per block.
        ("bfloat16", 64),
    ],
)
def test_rocketkv_v2_kt_rows_live_in_request_pages(
    kt_cache_dtype, page_size, make_rocket_kv_cache_manager
):
    """KVCacheManagerV2 KT rows are addressed by the request's own KV pages.

    The KT block table must be the request's base page indices in KT-view units,
    and writes through one layer's KT view must not reach K/V data, another
    layer's KT rows, or another request's pages.
    """
    num_layers, num_kv_heads, head_dim, tokens_per_block = 2, 8, 128, 32
    sparse_attn_config = RocketSparseAttentionConfig(
        prompt_budget=256, page_size=page_size, kt_cache_dtype=kt_cache_dtype
    )
    kv_cache_manager = make_rocket_kv_cache_manager(
        num_layers,
        num_kv_heads,
        head_dim,
        tokens_per_block,
        max_seq_len=1024,
        max_batch_size=2,
        dtype=DataType.BF16,
        sparse_attn_config=sparse_attn_config,
        use_kv_cache_manager_v2=True,
    )
    request_ids = [0, 1]
    token_nums = [100, 200]
    kv_cache_manager.add_dummy_requests(request_ids, token_nums)
    kt_tokens_per_block = math.ceil(tokens_per_block / page_size)
    assert kv_cache_manager.kt_tokens_per_block == kt_tokens_per_block

    kt_dtype = torch.float8_e5m2 if kt_cache_dtype == "float8_e5m2" else torch.bfloat16
    kv_bytes_per_token = 2 * num_kv_heads * head_dim * 2
    kt_bytes_per_block = kt_tokens_per_block * 2 * num_kv_heads * head_dim * kt_dtype.itemsize
    kt_bytes_per_token = math.ceil(kt_bytes_per_block / tokens_per_block)
    assert kv_cache_manager.get_cache_bytes_per_token() == num_layers * (
        kv_bytes_per_token + kt_bytes_per_token
    )

    block_offsets = torch.full(
        (len(request_ids), kv_cache_manager.max_kt_blocks_per_seq), -7, dtype=torch.int32
    )
    kv_cache_manager.copy_kt_block_offsets(request_ids, block_offsets)
    base_page_indices = kv_cache_manager.get_batch_cache_indices(request_ids, raw_indices=True)
    kt_pages = []
    for row, token_num in enumerate(token_nums):
        num_blocks = math.ceil(token_num / tokens_per_block)
        # Every KT row of the request's tokens maps into one of its own pages.
        assert math.ceil(math.ceil(token_num / page_size) / kt_tokens_per_block) <= num_blocks
        expected = [
            page * kv_cache_manager.kt_cache_page_scale
            for page in base_page_indices[row][:num_blocks]
        ]
        assert block_offsets[row, :num_blocks].tolist() == expected
        # Unused table entries keep the invalid page index rather than a scaled one.
        assert torch.all(block_offsets[row, num_blocks:] == -1)
        kt_pages.append(expected)

    kv_buffers = [kv_cache_manager.get_buffers(layer) for layer in range(num_layers)]
    kt_buffers = [kv_cache_manager.get_kt_buffers(layer) for layer in range(num_layers)]
    for layer in range(num_layers):
        assert kt_buffers[layer].dtype == kt_dtype
        assert kt_buffers[layer].shape[1:] == (kt_tokens_per_block, num_kv_heads, 2 * head_dim)
        for pages in kt_pages:
            kt_buffers[layer][pages] = 0
    for kv_buffer in kv_buffers:
        kv_buffer.zero_()

    # Small integers are exact in both KT dtypes.
    def fill_value(layer, row):
        return float(1 + layer * len(request_ids) + row)

    for layer in range(num_layers):
        for row, pages in enumerate(kt_pages):
            kt_buffers[layer][pages] = fill_value(layer, row)
    torch.cuda.synchronize()

    for kv_buffer in kv_buffers:
        assert torch.count_nonzero(kv_buffer) == 0
    for layer in range(num_layers):
        for row, pages in enumerate(kt_pages):
            values = kt_buffers[layer][pages].float()
            assert torch.all(values == fill_value(layer, row))


@pytest.mark.parametrize("prompt_len", [40, 200])
def test_rocketkv_v2_context_completion_keeps_retained_prompt(
    prompt_len, make_rocket_kv_cache_manager
):
    """A finished prompt keeps only its retained RocketKV entries in V2 pages.

    The cache counts the retained entries as its prompt and history, so decode
    grows from them and its allocations count as generation, not missed context
    blocks.
    """
    tokens_per_block, prompt_budget = 32, 64
    sparse_attn_config = RocketSparseAttentionConfig(
        prompt_budget=prompt_budget, page_size=4, kt_cache_dtype="float8_e5m2"
    )
    kv_cache_manager = make_rocket_kv_cache_manager(
        num_layers=2,
        num_kv_heads=8,
        head_dim=128,
        tokens_per_block=tokens_per_block,
        max_seq_len=1024,
        max_batch_size=1,
        dtype=DataType.BF16,
        sparse_attn_config=sparse_attn_config,
        use_kv_cache_manager_v2=True,
        enable_stats=True,
    )
    request = LlmRequest(
        request_id=0,
        max_new_tokens=8,
        input_tokens=list(range(prompt_len)),
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )
    assert kv_cache_manager.prepare_context(request)
    assert kv_cache_manager.resize_context(request, request.context_chunk_size)
    kv_cache = kv_cache_manager.kv_cache_map[request.py_request_id]
    assert kv_cache.capacity >= prompt_len
    assert kv_cache.num_blocks == math.ceil(kv_cache.capacity / tokens_per_block)

    context_batch = ScheduledRequests()
    context_batch.context_requests_last_chunk = [request]
    request.context_current_position = prompt_len
    request.add_new_token(prompt_len, 0)
    kv_cache_manager.update_context_resources(context_batch)
    kv_cache_manager.commit_scheduled_kv_cache_stats(context_batch)
    prompt_stats = kv_cache_manager.get_kv_cache_stats()
    assert prompt_stats.missed_blocks > 0

    retained = min(prompt_len, prompt_budget)
    assert kv_cache.capacity == retained
    assert kv_cache.history_length == retained
    # Only the pages that hold retained entries stay with the request.
    assert kv_cache.num_blocks == math.ceil(retained / tokens_per_block)
    assert len(kv_cache_manager.get_batch_cache_indices([request.py_request_id])[0]) == (
        kv_cache.num_blocks
    )

    # Decode grows from the retained entries, and each generation update
    # advances history over the retained entries plus the decoded tokens.
    request.state = LlmRequestState.GENERATION_IN_PROGRESS
    generation_batch = ScheduledRequests()
    generation_batch.generation_requests = [request]
    num_decode_steps = 2 * tokens_per_block
    for step in range(1, num_decode_steps + 1):
        assert kv_cache_manager.try_allocate_generation(request)
        assert kv_cache.capacity == retained + step
        request.add_new_token(prompt_len + step, 0)
        kv_cache_manager.update_resources(generation_batch)
        assert kv_cache.capacity == retained + step
        assert kv_cache.history_length == retained + step
    num_decode_blocks = kv_cache.num_blocks - math.ceil(retained / tokens_per_block)
    assert num_decode_blocks == 2

    kv_cache_manager.commit_scheduled_kv_cache_stats(generation_batch)
    stats = kv_cache_manager.get_kv_cache_stats()
    # Decode pages are generation allocations, not missed context blocks.
    assert stats.alloc_new_blocks == prompt_stats.alloc_new_blocks + num_decode_blocks
    assert stats.missed_blocks == prompt_stats.missed_blocks


@pytest.mark.parametrize("kt_tokens_per_block", [3, 8])
def test_vanilla_kt_rows_follow_the_kt_block_table(kt_tokens_per_block):
    """Vanilla KT rows spread over KT blocks match the single-block layout."""
    page_size, num_kv_heads, head_dim = 4, 2, 16
    prompt_len, num_decode_steps = 37, 6
    device = torch.device("cuda")
    dtype = torch.bfloat16
    sparse_params = RocketSparseAttentionConfig(
        page_size=page_size, kt_cache_dtype="bfloat16"
    ).to_sparse_params()
    attn = RocketVanillaAttention(
        layer_idx=0,
        num_heads=2 * num_kv_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        sparse_params=sparse_params,
    )

    num_kt_rows = math.ceil((prompt_len + num_decode_steps) / page_size)
    kt_row_shape = (num_kv_heads, 2 * head_dim)
    single_block_cache = torch.zeros((1, num_kt_rows, *kt_row_shape), dtype=dtype, device=device)
    single_block_table = torch.zeros(1, dtype=torch.int32, device=device)
    num_blocks = math.ceil(num_kt_rows / kt_tokens_per_block)
    paged_cache = torch.zeros(
        (num_blocks + 3, kt_tokens_per_block, *kt_row_shape), dtype=dtype, device=device
    )
    # Give the request its blocks in reverse order, away from block 0.
    paged_table = torch.arange(num_blocks + 2, 1, -1, dtype=torch.int32, device=device)[:num_blocks]

    def update(k, seq_len, past_seen_tokens, is_generation):
        positions = torch.arange(
            past_seen_tokens // page_size, math.ceil(seq_len / page_size), device=device
        )
        expected = attn._single_request_update_kt_cache(
            k, single_block_cache, single_block_table, seq_len, positions, is_generation
        )
        actual = attn._single_request_update_kt_cache(
            k, paged_cache, paged_table, seq_len, positions, is_generation
        )
        assert actual.shape == (1, math.ceil(seq_len / page_size), *kt_row_shape)
        assert torch.equal(actual, expected)

    k = torch.randn((1, prompt_len, num_kv_heads, head_dim), dtype=dtype, device=device)
    update(k, prompt_len, 0, is_generation=False)
    for past_seen_tokens in range(prompt_len, prompt_len + num_decode_steps):
        k = torch.randn((1, 1, num_kv_heads, head_dim), dtype=dtype, device=device)
        update(k, past_seen_tokens + 1, past_seen_tokens, is_generation=True)

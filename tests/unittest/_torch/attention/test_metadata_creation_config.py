# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A constructed copy of a metadata reports the creation configuration of its source.

Covers the core families and the feature extensions that attach without a
family-specific KV cache manager; the sparse tests check their own metadata
(see ``utils.metadata_contract.CREATION_CONFIG_COVERAGE``).
"""

import pytest
import torch
from utils.metadata_contract import check_copy_reproduces_creation_config

import tensorrt_llm
from tensorrt_llm._torch.attention.backends.features import (
    FlashMlaMetadata,
    HelixMetadata,
    RecurrentConfig,
    RecurrentMetadata,
)
from tensorrt_llm._torch.attention.backends.features.spec_decoding import SpecDecodingConfig
from tensorrt_llm._torch.attention.backends.flashinfer import FlashInferAttentionMetadata
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionMetadata,
    AttentionRuntimeFeatures,
    BatchInputs,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_backend import (
    MiniMaxM3MsaMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.triton_metadata import (
    MiniMaxM3TritonMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.params import MTPIndexShareExtension
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.attention.backends.vanilla import VanillaAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.models.modeling_qwen4_exp import Qwen4ExpPleState
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

BATCH_SIZE = 4
TOKENS_PER_BLOCK = 32
HISTORY = 100
MAX_DRAFT_TOKENS = 1


@pytest.fixture(scope="module")
def manager():
    manager = KVCacheManager(
        KvCacheConfig(max_tokens=BATCH_SIZE * 8 * TOKENS_PER_BLOCK, enable_block_reuse=False),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=1,
        num_kv_heads=2,
        head_dim=64,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=512,
        max_batch_size=BATCH_SIZE,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.BF16,
    )
    manager.add_dummy_requests(
        request_ids=list(range(BATCH_SIZE)), token_nums=[HISTORY + 2] * BATCH_SIZE, is_gen=False
    )
    yield manager
    manager.shutdown()


def _batch(manager, query_len: int = 1 + MAX_DRAFT_TOKENS) -> BatchInputs:
    return BatchInputs(
        seq_lens=torch.full((BATCH_SIZE,), query_len, dtype=torch.int),
        num_contexts=0,
        request_ids=list(range(BATCH_SIZE)),
        prompt_lens=[HISTORY] * BATCH_SIZE,
        kv_cache_params=KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=[HISTORY] * BATCH_SIZE
        ),
    )


def test_trtllm_target_draft_and_graph_copies(manager):
    metadata = TrtllmAttentionMetadata(
        max_num_requests=BATCH_SIZE,
        max_num_tokens=(1 + MAX_DRAFT_TOKENS) * BATCH_SIZE,
        kv_cache_manager=manager,
        runtime_features=AttentionRuntimeFeatures(has_speculative_draft_tokens=True),
        spec_decoding=SpecDecodingConfig(
            max_draft_len=MAX_DRAFT_TOKENS, max_total_draft_tokens=MAX_DRAFT_TOKENS
        ),
        one_model_draft=True,
    )
    check_copy_reproduces_creation_config(metadata)
    metadata.prepare(_batch(manager))
    check_copy_reproduces_creation_config(metadata)
    check_copy_reproduces_creation_config(metadata.draft)
    graph = metadata.create_cuda_graph_metadata(BATCH_SIZE, max_draft_tokens=MAX_DRAFT_TOKENS)
    check_copy_reproduces_creation_config(graph)
    graph.prepare(_batch(manager))
    check_copy_reproduces_creation_config(graph)


@pytest.mark.parametrize(
    "extension_type",
    [
        HelixMetadata,
        FlashMlaMetadata,
        MTPIndexShareExtension,
        Qwen4ExpPleState,
        MiniMaxM3MsaMetadata,
        MiniMaxM3TritonMetadata,
    ],
    ids=lambda extension_type: extension_type.__name__,
)
def test_trtllm_feature_extensions(manager, extension_type):
    metadata = TrtllmAttentionMetadata(
        max_num_requests=BATCH_SIZE,
        max_num_tokens=(1 + MAX_DRAFT_TOKENS) * BATCH_SIZE,
        kv_cache_manager=manager,
        extension_types=(extension_type,),
    )
    check_copy_reproduces_creation_config(metadata)
    metadata.prepare(_batch(manager, query_len=1))
    check_copy_reproduces_creation_config(metadata)


def test_vanilla_core(manager):
    metadata = VanillaAttentionMetadata(
        max_num_requests=BATCH_SIZE,
        max_num_tokens=(1 + MAX_DRAFT_TOKENS) * BATCH_SIZE,
        kv_cache_manager=manager,
    )
    check_copy_reproduces_creation_config(metadata)
    metadata.prepare(_batch(manager))
    check_copy_reproduces_creation_config(metadata)


def test_flashinfer_core(manager):
    metadata = FlashInferAttentionMetadata(
        max_num_requests=BATCH_SIZE,
        max_num_tokens=(1 + MAX_DRAFT_TOKENS) * BATCH_SIZE,
        kv_cache_manager=manager,
    )
    check_copy_reproduces_creation_config(metadata)
    metadata.prepare(_batch(manager))
    check_copy_reproduces_creation_config(metadata)
    graph = metadata.create_cuda_graph_metadata(BATCH_SIZE)
    check_copy_reproduces_creation_config(graph)


def test_recurrent_extension():
    metadata = AttentionMetadata(
        max_num_requests=BATCH_SIZE,
        max_num_tokens=BATCH_SIZE,
        extension_types=(RecurrentMetadata,),
        recurrent_config=RecurrentConfig(chunk_size=64),
    )
    check_copy_reproduces_creation_config(metadata)


def test_cross_attention_decoder_and_sub_metadata():
    decoder = AttentionMetadata(max_num_requests=BATCH_SIZE, max_num_tokens=BATCH_SIZE)
    decoder.prepare(
        BatchInputs(
            seq_lens=torch.tensor([3, 1], dtype=torch.int32),
            num_contexts=1,
            request_ids=[0, 1],
            prompt_lens=[3, 5],
        )
    )
    cross = decoder.create_cross_metadata([7, 0])
    check_copy_reproduces_creation_config(decoder)
    check_copy_reproduces_creation_config(cross)

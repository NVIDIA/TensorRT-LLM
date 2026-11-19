# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for the Kimi K3.1 family DSA indexer and its attention metadata.

This file tests the indexer's chunked prefill on the KDA + MLA hybrid cache manager (request-level and
Q-block chunking) against the single-pass prefill.
"""

import random
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm import deep_gemm
from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import DSACacheManagerV2
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_indexer import K31Indexer
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_indexer_plan import (
    use_cute_dsl_paged_mqa_logits as resolve_cute_dsl_paged_mqa_logits,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_metadata import (
    K31DSAtrtllmAttentionMetadata,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_params import k31_dsa_params
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.kda_dsa_cache_manager import (
    KimiK3DSAHybridCacheManager,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization.utils import fp8_utils

__extra_import_path__ = ["~/tests/unittest"]

from utils.util import skip_pre_hopper


def has_deep_gemm():
    """Return True if the DeepGEMM library is available."""
    try:
        return deep_gemm is not None
    except Exception:
        return False


def create_dsa_cache_manager(
    batch_size: int,
    head_dim: int,
    tokens_per_block: int,
    max_seq_len: int,
    num_layers: int = 1,
    indexer_k_dtype: str = "fp8",
    index_topk: int = 2048,
    pretrained_config=None,
):
    """Helper to create a DSA cache manager for testing."""

    sparse_attn_config = DeepSeekSparseAttentionConfig(
        index_head_dim=head_dim,
        index_n_heads=32,  # Default number of heads for indexer
        index_topk=index_topk,
        indexer_k_dtype=indexer_k_dtype,
    )

    kv_cache_config = KvCacheConfig(
        enable_block_reuse=False,
        # The KDA + MLA hybrid (V2) places well under max_tokens tokens (four
        # 4096-token requests need max_tokens = 3 x 16384); give it 4x.
        max_tokens=max_seq_len * batch_size * 4,
    )

    # Create mapping (single GPU, no parallelism)
    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)

    # Kimi K3.1 serves DSA from its KDA + MLA hybrid manager on KV cache
    # manager V2 (K31_RUNTIME_HOOKS.kv_cache_manager_cls). Layers 0..num_layers-1
    # are the DSA MLA layers under test; one trailing KDA layer completes
    # the hybrid layout without shifting their layer indices.
    cache_manager = KimiK3DSAHybridCacheManager(
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_num_heads=4,
        mamba_n_groups=4,
        mamba_head_dim=8,
        mamba_num_layers=1,
        mamba_layer_mask=[False] * num_layers + [True],
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=torch.float32,
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=num_layers,
        num_kv_heads=1,  # MQA
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=DataType.HALF,
        layer_mask=[True] * num_layers + [False],
        sparse_attention_config=sparse_attn_config,
        pretrained_config=pretrained_config,
    )
    return cache_manager, sparse_attn_config


@pytest.fixture
def dsa_cache_manager():
    """Factory for DSA cache managers that shuts every one down on teardown.

    KV cache manager V2 owns CUDA reservations that must be released
    deterministically; a manager left to the garbage collector is destroyed on
    a thread without a current CUDA context.
    """
    managers = []

    def make(**kwargs):
        manager, sparse_attn_config = create_dsa_cache_manager(**kwargs)
        managers.append(manager)
        return manager, sparse_attn_config

    yield make
    for manager in reversed(managers):
        manager.shutdown()


def create_indexer(sparse_attn_config, layer_idx=0, pretrained_config=None):
    """Create a test indexer using checkpoint-derived parameters (position-free, as K3.1's)."""
    mla_params = SimpleNamespace(hidden_size=4096, q_lora_rank=512, qk_rope_head_dim=64)
    if pretrained_config is None:
        pretrained_config = SimpleNamespace(index_rope_dim=0)
    elif not hasattr(pretrained_config, "index_rope_dim"):
        pretrained_config = SimpleNamespace(**vars(pretrained_config), index_rope_dim=0)
    sparse_params = k31_dsa_params(
        sparse_attn_config.to_sparse_params(pretrained_config=pretrained_config), pretrained_config
    )
    return K31Indexer(mla_params, sparse_params, torch.bfloat16, layer_idx=layer_idx)


def _create_mock_metadata(
    request_ids,
    batch_size,
    num_contexts,
    num_generations,
    seq_lens,
    kv_lens,
    num_cached_tokens,
    cache_manager,
    num_ctx_tokens,
    num_tokens,
    indexer_max_chunk_size=8194,
    max_draft_tokens=0,
    enable_context_mla_with_cached_kv=False,
    index_topk=2048,
    enable_indexer_skip=False,
    use_cute_dsl_paged_mqa_logits=None,
    indexer_head_dim=128,
    max_num_tokens=8192,
):
    """Helper to create mock metadata for testing."""
    if use_cute_dsl_paged_mqa_logits is None:
        # Match the indexer built from the config default: the buffer layout
        # (separate full-next_n schedule) differs between the two kernels.
        use_cute_dsl_paged_mqa_logits = resolve_cute_dsl_paged_mqa_logits(
            DeepSeekSparseAttentionConfig.model_fields["use_cute_dsl_paged_mqa_logits"].default
        )

    class MockKVCacheParams:
        """Mock KV cache parameters holding per-sequence cached token counts."""

        def __init__(self):
            """Initialize with the provided num_cached_tokens."""
            self.num_cached_tokens_per_seq = num_cached_tokens

    class MockMetadata(K31DSAtrtllmAttentionMetadata):
        """Mock DSA attention metadata for unit testing indexer operations."""

        def __init__(self, cache_manager: DSACacheManagerV2) -> None:
            """Initialize mock metadata with buffers, block tables, and indexer state.

            `cache_manager` is a parameter rather than a capture of the enclosing
            variable: a capture puts it in a closure cell reachable from the class
            object, which is always part of a reference cycle, so only the cyclic
            GC would free the manager -- on whatever thread runs the collection,
            which need not have a CUDA context.
            """
            self.num_sms = deep_gemm.get_num_sms()
            self.max_num_tokens = max_num_tokens
            self.request_ids = request_ids
            self.num_contexts = num_contexts
            self.num_generations = num_generations
            self._num_seqs = num_contexts + num_generations
            self.max_num_sequences = batch_size
            self.max_draft_tokens = max_draft_tokens
            self.num_sparse_topk = index_topk
            self._radix_rows_per_sequence = 1 + self.max_draft_tokens
            self._create_radix_aux_buffers(capture_graph=False)
            self.enable_indexer_skip = enable_indexer_skip
            self.indexer_head_dim = indexer_head_dim
            self.indexer_quant_block_size = 128
            self.compress_ratios = [1]
            # Keep seq_lens on CPU for split_prefill_chunks and other CPU operations
            # CUDA kernels will convert to CUDA as needed
            self.seq_lens = seq_lens.cpu() if seq_lens.is_cuda else seq_lens
            self.kv_lens = kv_lens
            self.kv_cache_params = MockKVCacheParams()
            self.kv_cache_manager = cache_manager
            # The DSA indexer K cache is uncompressed: its slot mapping uses the
            # cache manager's tokens per block.
            self._indexer_compress_ratio = 1
            self._tokens_per_block = self.kv_cache_manager.tokens_per_block
            self.kv_lens_cuda_runtime = kv_lens.cuda()
            self.indexer_k_cache_block_offsets = torch.zeros(
                [batch_size, self.kv_cache_manager.max_blocks_per_seq],
                device="cuda",
                dtype=torch.int32,
            )
            self.host_indexer_k_cache_block_offsets = torch.zeros_like(
                self.indexer_k_cache_block_offsets,
                device="cpu",
                pin_memory=True,
            )
            block_ids = self.kv_cache_manager.get_batch_cache_indices(request_ids)
            for i in range(len(block_ids)):
                self.host_indexer_k_cache_block_offsets[i, : len(block_ids[i])] = torch.tensor(
                    block_ids[i], dtype=torch.int32
                )
            self.indexer_k_cache_block_offsets[:batch_size].copy_(
                self.host_indexer_k_cache_block_offsets[:batch_size], non_blocking=True
            )

            self.slot_mapping_fp8 = torch.zeros((num_tokens,), device="cuda", dtype=torch.int64)
            self.slot_mapping_scale = torch.zeros((num_tokens,), device="cuda", dtype=torch.int64)
            self.scheduler_metadata_buffer = torch.zeros(
                (self.num_sms + 1, 2), device="cuda", dtype=torch.int32
            )
            # DSL needs the next_n=1 schedule preserved (kNumNextNAtoms=1),
            # so allocate a separate buffer for the full-next_n schedule.
            # DeepGEMM expects the full-next_n schedule in
            # `scheduler_metadata_buffer` itself (the alias makes
            # `K31Indexer.prepare()`'s second populate overwrite the first).
            if use_cute_dsl_paged_mqa_logits:
                self.scheduler_metadata_buffer_full_next_n = torch.zeros(
                    (self.num_sms + 1, 2), device="cuda", dtype=torch.int32
                )
            else:
                self.scheduler_metadata_buffer_full_next_n = self.scheduler_metadata_buffer
            # Pre-allocated 2D kv_lens buffer for the DeepGEMM 2D context_lens API.
            self.kv_lens_cuda_2d = torch.zeros(
                (self.num_seqs, 1 + self.max_draft_tokens), device="cuda", dtype=torch.int32
            )
            if num_generations > 0:
                gen_kv_lens = (
                    kv_lens[num_contexts : num_contexts + num_generations].cuda().to(torch.int32)
                )
                next_n_cap = 1 + self.max_draft_tokens
                self.kv_lens_cuda_2d[:num_generations, :next_n_cap].copy_(
                    gen_kv_lens.unsqueeze(-1).expand(-1, next_n_cap)
                )
            self.cu_seqlen_ks = torch.zeros((num_tokens,), device="cuda", dtype=torch.int32)
            self.cu_seqlen_ke = torch.zeros((num_tokens,), device="cuda", dtype=torch.int32)
            self.ctx_kv_offsets = torch.zeros((num_tokens, 1), device="cuda", dtype=torch.int32)
            self.gen_kv_offsets = torch.zeros((num_tokens, 1), device="cuda", dtype=torch.int32)
            self.host_slot_mapping_fp8 = torch.zeros_like(
                self.slot_mapping_fp8, device="cpu", pin_memory=True
            )
            self.host_slot_mapping_scale = torch.zeros_like(
                self.slot_mapping_scale, device="cpu", pin_memory=True
            )
            self.host_ctx_kv_indptr = torch.zeros(
                (num_contexts + 1,), device="cpu", pin_memory=True, dtype=torch.int64
            )
            self.host_gen_kv_indptr = torch.zeros(
                (num_generations + 1,), device="cpu", pin_memory=True, dtype=torch.int64
            )
            # Add host_ctx_cached_token_indptr for prepare_one_prefill_chunk
            self.host_ctx_cached_token_indptr = torch.zeros(
                (num_contexts + 1,), device="cpu", pin_memory=True, dtype=torch.int64
            )
            self._num_ctx_tokens = num_ctx_tokens
            self._num_tokens = num_tokens
            self.num_gen_tokens = num_tokens - num_ctx_tokens
            # Also set private attributes used by DSAtrtllmAttentionMetadata
            self._num_contexts = num_contexts
            self._num_generations = num_generations
            self._num_ctx_tokens = num_ctx_tokens
            self._num_tokens = num_tokens

            torch.cumsum(
                kv_lens[:num_contexts],
                dim=0,
                dtype=torch.int64,
                out=self.host_ctx_kv_indptr[1 : num_contexts + 1],
            )
            torch.cumsum(
                kv_lens[num_contexts : num_contexts + num_generations],
                dim=0,
                dtype=torch.int64,
                out=self.host_gen_kv_indptr[1 : num_generations + 1],
            )
            # Compute host_ctx_cached_token_indptr from num_cached_tokens
            if num_contexts > 0:
                cached_lens = torch.tensor(num_cached_tokens[:num_contexts], dtype=torch.int64)
                torch.cumsum(
                    cached_lens,
                    dim=0,
                    dtype=torch.int64,
                    out=self.host_ctx_cached_token_indptr[1 : num_contexts + 1],
                )

            # Add indexer-specific attributes
            self.indexer_max_chunk_size = indexer_max_chunk_size
            self.indexer_prefill_chunks = None

            self.enable_context_mla_with_cached_kv = enable_context_mla_with_cached_kv

            # Add runtime_features for chunked prefill detection
            class RuntimeFeatures:
                """Mock runtime features for chunked prefill detection."""

                def __init__(self):
                    """Initialize runtime features from outer scope parameters."""
                    self.chunked_prefill = enable_context_mla_with_cached_kv
                    self.cache_reuse = False
                    self.has_speculative_draft_tokens = False
                    self.chunk_size = 512
                    self.chunked_prefill_buffer_batch_size = 4

            self.runtime_features = RuntimeFeatures()

            # The CuTe DSL logits' atom-split buffers (DeepGEMM runs any next_n natively).
            self.kv_lens_expanded_cuda = torch.zeros(
                (self.num_seqs * (1 + self.max_draft_tokens),), device="cuda", dtype=torch.int32
            )
            self.kv_lens_expanded_host = torch.zeros_like(
                self.kv_lens_expanded_cuda, device="cpu", pin_memory=True
            )
            self.block_table_expanded = torch.zeros(
                (
                    self.num_seqs * (1 + self.max_draft_tokens),
                    self.kv_cache_manager.max_blocks_per_seq,
                ),
                device="cuda",
                dtype=torch.int32,
            )
            self.host_block_table_expanded = torch.zeros_like(
                self.block_table_expanded, device="cpu", pin_memory=True
            )
            self.scheduler_metadata_buffer_expanded = torch.zeros(
                (self.num_sms + 1, 2), device="cuda", dtype=torch.int32
            )

            # Add skip indexer attributes
            self.topk_indices_buffer = torch.zeros(
                (num_tokens, self.num_sparse_topk), device="cuda", dtype=torch.int32
            )

            if self.num_contexts > 0 and self.enable_indexer_skip:
                self.skip_indexer_for_ctx_reqs = (
                    kv_lens[: self.num_contexts].max().item() <= self.num_sparse_topk
                )
            else:
                self.skip_indexer_for_ctx_reqs = False

            if self.num_generations > 0 and self.enable_indexer_skip:
                self.max_draft_tokens + 1
                self.skip_indexer_for_gen_reqs = (
                    kv_lens[self.num_contexts : self.num_seqs].max().item() <= self.num_sparse_topk
                )
            else:
                self.skip_indexer_for_gen_reqs = False
            self.prepare_dense_topk_indices(self.kv_lens_cuda_runtime, device=True)

        @property
        def num_seqs(self) -> int:
            """
            The number of sequences in the batch.
            """
            return self._num_seqs

    return MockMetadata(cache_manager)


def validate_topk_indices(topk_indices_0, topk_indices_1, total_tokens):
    """
    Validate the similarity between two topk indices.
    """
    num_exact_matches = 0
    total_similarity = 0.0
    min_similarity = 1.0

    for token_idx in range(total_tokens):
        valid_0 = topk_indices_0[token_idx][topk_indices_0[token_idx] != -1]
        valid_1 = topk_indices_1[token_idx][topk_indices_1[token_idx] != -1]

        if torch.equal(valid_0, valid_1):
            num_exact_matches += 1
            similarity = 1.0
            total_similarity += similarity
        else:
            valid_0_set = set(valid_0.cpu().tolist())
            valid_1_set = set(valid_1.cpu().tolist())
            intersection = len(valid_0_set & valid_1_set)
            union = len(valid_0_set | valid_1_set)
            similarity = intersection / union if union > 0 else 0.0
            total_similarity += similarity

        # Track min similarity
        min_similarity = min(min_similarity, similarity)

    return num_exact_matches, total_similarity, min_similarity


@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@skip_pre_hopper
@pytest.mark.parametrize(
    "chunk_size,seq_lens_list,chunking_type",
    [
        # Two-level chunking: intra-request Q-block splitting for large requests
        (1024, [512, 2500, 800], "two_level"),  # Middle request needs Q-blocks
    ],
)
def test_indexer_chunked_prefill(
    chunk_size,
    seq_lens_list,
    chunking_type,
    dsa_cache_manager,
):
    """
    Tests for indexer chunked prefill:
    1. Request-level chunking: Multiple small requests packed together
    2. Two-level chunking: Large requests split into Q-blocks + request-level chunking

    Validates that chunked and non-chunked (single-pass) execution produce identical topk indices.
    Tests:
    - Chunk metadata creation and usage
    - K-cache gathering for chunks
    - Variable-length sequence handling
    - Q-block splitting for large requests
    - Correct attention scope per Q-block
    """
    torch.manual_seed(42)
    random.seed(42)

    # Test parameters
    heads, head_dim = 32, 128
    block_size = 64
    index_topk = 2048
    max_model_len = 16384
    layer_idx = 0

    # Generate sequence lengths based on chunking type
    if chunking_type == "request_level":
        # Generate variable seq lens for request-level chunking
        # Each request fits in chunk, but batch exceeds chunk_size
        batch_size = 4
        seq_lens_list = []
        for _ in range(batch_size):
            seq_len = random.randint(int(0.3 * chunk_size), int(0.9 * chunk_size))
            seq_lens_list.append(seq_len)
    else:
        # Two-level chunking: use provided seq_lens with large requests
        batch_size = len(seq_lens_list)

    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device="cpu")
    total_tokens = seq_lens.sum().item()

    # Create cache manager and indexer
    cache_manager, sparse_attn_config = dsa_cache_manager(
        batch_size=batch_size,
        head_dim=head_dim,
        tokens_per_block=block_size,
        max_seq_len=max_model_len,
        num_layers=1,
        index_topk=index_topk,
    )
    indexer = create_indexer(sparse_attn_config, layer_idx=layer_idx)

    # Allocate blocks for all sequences
    request_ids = list(range(batch_size))
    cache_manager.add_dummy_requests(
        request_ids=request_ids, token_nums=seq_lens_list, is_gen=False
    )

    # Generate test data
    q = torch.randn((total_tokens, heads, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((total_tokens, head_dim), device="cuda", dtype=torch.bfloat16)
    weights = torch.randn((total_tokens, heads), device="cuda", dtype=torch.float32)
    hidden_states = torch.randn((total_tokens, 4096), device="cuda", dtype=torch.bfloat16)

    # Quantize inputs
    q_fp8 = q.to(torch.float8_e4m3fn)
    k_fp8, k_scale = fp8_utils.fp8_quantize_1x128_sf_transpose(k)

    # ========== Test Path 1: Chunked Prefill ==========

    metadata_chunked = _create_mock_metadata(
        request_ids,
        batch_size,
        num_contexts=batch_size,
        num_generations=0,
        seq_lens=seq_lens.clone(),
        kv_lens=seq_lens.clone(),
        num_cached_tokens=[0] * batch_size,
        cache_manager=cache_manager,
        num_ctx_tokens=total_tokens,
        num_tokens=total_tokens,
        indexer_max_chunk_size=chunk_size,
        indexer_head_dim=head_dim,
    )

    K31Indexer.prepare(metadata_chunked)

    assert metadata_chunked.indexer_prefill_chunks is not None

    indexer._update_k_cache(k_fp8, k_scale, metadata_chunked)
    topk_indices_chunked = indexer.sparse_attn_indexer(
        metadata_chunked, hidden_states, q_fp8, k_fp8, k_scale, weights
    )

    # ========== Test Path 2: Non-chunked Baseline ==========

    metadata_baseline = _create_mock_metadata(
        request_ids,
        batch_size,
        num_contexts=batch_size,
        num_generations=0,
        seq_lens=seq_lens.clone(),
        kv_lens=seq_lens.clone(),
        num_cached_tokens=[0] * batch_size,
        cache_manager=cache_manager,
        num_ctx_tokens=total_tokens,
        num_tokens=total_tokens,
        indexer_max_chunk_size=max_model_len,
        indexer_head_dim=head_dim,
    )

    K31Indexer.prepare(metadata_baseline)

    indexer._update_k_cache(k_fp8, k_scale, metadata_baseline)
    topk_indices_baseline = indexer.sparse_attn_indexer(
        metadata_baseline, hidden_states, q_fp8, k_fp8, k_scale, weights
    )

    # ========== Validation ==========

    # Use Jaccard similarity to handle ties (multiple indices with same value)
    _, total_similarity, _ = validate_topk_indices(
        topk_indices_chunked, topk_indices_baseline, total_tokens
    )
    avg_similarity = total_similarity / total_tokens

    # Use Jaccard similarity threshold instead of exact match
    assert avg_similarity >= 0.9, (
        f"Chunked and non-chunked results differ significantly: avg similarity {avg_similarity:.4f} < 0.9"
    )

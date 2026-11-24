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

This file tests:
1. Indexer construction from the checkpoint config (NoPE indexer, key-norm eps, BF16 projections)
2. Slot mappings and the FP8 indexer K cache in the KDA + MLA hybrid cache manager
3. Paged MQA logits on that cache, chunked prefill and decode Top-K
4. Metadata buffer swaps for draft replay
"""

import builtins
import functools
import importlib
import random
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm import deep_gemm
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention import (
    dsa_indexer,
    dsa_indexer_plan,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.dsa_indexer import (
    K31Indexer,
    transform_local_topk_and_prepare_pool_view,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.dsa_indexer_plan import (
    compute_cu_seqlen_kv_bounds_with_cache,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.dsa_indexer_plan import (
    use_cute_dsl_paged_mqa_logits as resolve_cute_dsl_paged_mqa_logits,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.dsa_metadata import (
    K31DSAtrtllmAttentionMetadata,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.dsa_params import (
    k31_dsa_params,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.kda_dsa_cache_manager import (
    KimiK3DSAHybridCacheManager,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import DSACacheManagerV2
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.speculative.interface import (
    prepare_attn_metadata_for_draft_replay,
    restore_attn_metadata_after_draft_replay,
)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.deep_gemm import fp8_paged_mqa_logits
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


def test_metadata_compiles_no_cute_dsl_radix_topk():
    """The engine warmup calls this on every DSA metadata; the K3.1 indexer runs no CuTe DSL decode
    Top-K (routing refuses use_cute_dsl_topk), so it compiles nothing."""
    with patch(
        "tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops.warmup_cute_dsl_radix_topk_decode"
    ) as cute_dsl_radix:
        K31DSAtrtllmAttentionMetadata.warmup_cute_dsl_radix_topk(SimpleNamespace(), 1)
    cute_dsl_radix.assert_not_called()


def test_index_rope_dim_comes_from_pretrained_config():
    sparse_config = DeepSeekSparseAttentionConfig(
        index_head_dim=128, index_n_heads=32, index_topk=2048
    )
    assert k31_dsa_params(sparse_config.to_sparse_params()).index_rope_dim is None
    nope_config = SimpleNamespace(index_rope_dim=0)
    assert (
        k31_dsa_params(
            sparse_config.to_sparse_params(pretrained_config=nope_config), nope_config
        ).index_rope_dim
        == 0
    )


@skip_pre_hopper
def test_index_k_norm_eps_comes_from_pretrained_config():
    sparse_config = DeepSeekSparseAttentionConfig(
        index_head_dim=128, index_n_heads=32, index_topk=2048
    )
    # DeepSeek-V3.2 checkpoints carry no key-norm epsilon: the indexer keeps 1e-6.
    assert k31_dsa_params(sparse_config.to_sparse_params()).index_k_norm_eps is None
    assert create_indexer(sparse_config).k_norm.variance_epsilon == 1e-6
    kimi_config = SimpleNamespace(index_k_norm_eps=1e-5)
    assert create_indexer(sparse_config, pretrained_config=kimi_config).k_norm.variance_epsilon == (
        1e-5
    )


@skip_pre_hopper
def test_position_free_indexer_skips_rope():
    """index_rope_dim=0 (Kimi K3.1): Q/K pass through unrotated, as two halves of each head."""
    sparse_config = DeepSeekSparseAttentionConfig(
        index_head_dim=128, index_n_heads=32, index_topk=2048
    )
    indexer = create_indexer(sparse_config, pretrained_config=SimpleNamespace(index_rope_dim=0))

    indexer = indexer.cuda()
    with torch.no_grad():
        indexer.wq_b.weight.normal_()
        indexer.k_norm.weight.normal_()
        indexer.k_norm.bias.normal_()
    qr = torch.randn(5, indexer.q_lora_rank, dtype=torch.bfloat16, device="cuda")
    indexer_k = torch.randn(5, indexer.head_dim, dtype=torch.bfloat16, device="cuda")

    q_pe, q_nope, k_pe, k_nope = indexer._project_q_norm_k(qr, indexer_k)

    q = (qr.float() @ indexer.wq_b.weight.float().t()).view(-1, indexer.n_heads, indexer.head_dim)
    torch.testing.assert_close(torch.cat([q_pe, q_nope], dim=-1).float(), q, rtol=2e-2, atol=2e-2)
    k = torch.nn.functional.layer_norm(
        indexer_k.float(),
        (indexer.head_dim,),
        indexer.k_norm.weight.float(),
        indexer.k_norm.bias.float(),
        indexer.k_norm.variance_epsilon,
    )
    torch.testing.assert_close(torch.cat([k_pe, k_nope], dim=-1).float(), k, rtol=2e-2, atol=2e-2)
    assert q_pe.shape[-1] == k_pe.shape[-1] == indexer.head_dim // 2


def test_indexer_projection_runs_in_model_dtype():
    """wq_b / wk / weights_proj hold model-dtype (BF16) weights and no bias, the precision of the reference indexer;
    create_indexer builds the indexer with dtype=torch.bfloat16."""
    sparse_config = DeepSeekSparseAttentionConfig(
        index_head_dim=128,
        index_n_heads=32,
        index_topk=128,
    )

    indexer = create_indexer(sparse_config)

    for projection in (indexer.wq_b, indexer.wk, indexer.weights_proj):
        assert projection.weight.dtype == torch.bfloat16
        assert dict(projection.named_parameters()).keys() == {"weight"}, "no bias"


def test_metadata_spec_update_resizes_radix_workspace():
    metadata = object.__new__(K31DSAtrtllmAttentionMetadata)
    metadata.max_num_sequences = 2
    metadata.max_draft_tokens = 0
    metadata._radix_rows_per_sequence = 1
    metadata.num_sparse_topk = 2
    metadata.is_cuda_graph = True
    metadata.nvfp4_mla_fp8_scratch = None
    metadata.kv_lens_cuda_2d = torch.empty(2, 1)
    metadata.kv_lens_expanded_host = torch.empty(2)
    metadata._create_kv_lens_2d_buffer = Mock()
    metadata.create_expanded_buffers = Mock()
    metadata._create_radix_aux_buffers = Mock()

    with patch(
        "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.dsa_metadata."
        "TrtllmAttentionMetadata.update_spec_dec_param"
    ):
        metadata.update_spec_dec_param(
            batch_size=2,
            is_spec_decoding_enabled=True,
            is_spec_dec_tree=False,
            is_spec_dec_dynamic_tree=False,
            max_draft_len=7,
            max_total_draft_tokens=7,
        )

    assert metadata._radix_rows_per_sequence == 8
    metadata._create_radix_aux_buffers.assert_called_once_with(True)


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


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
    # manager V2 (select_k31_kv_cache_manager_cls). Layers 0..num_layers-1
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


def test_transform_local_topk_uses_v2_page_mapping():
    """Use the V2 pool scale and offset when flattening DSA indices."""
    cache_manager = SimpleNamespace(
        _primary_pool_page_index_params=[(3, 1)],
        get_primary_pool_page_index_params=None,
    )
    cache_manager.get_primary_pool_page_index_params = MethodType(
        DSACacheManagerV2.get_primary_pool_page_index_params,
        cache_manager,
    )
    req_idx = torch.tensor([0], dtype=torch.int32)
    block_table = torch.tensor([[3, 7]], dtype=torch.int32)
    topk_indices = torch.tensor([[6]], dtype=torch.int32)
    metadata = SimpleNamespace(
        kv_cache_manager=cache_manager,
        _ensure_pool_view_cached=Mock(),
        _cached_block_table_ctx=block_table,
        _cached_req_idx_ctx=req_idx,
        _cached_tokens_per_block=4,
        _cached_pool_view=torch.empty(0),
    )
    expected_indices = torch.tensor([[90]], dtype=torch.int32)

    with patch(
        "torch.ops.trtllm.convert_req_index_to_global",
        return_value=expected_indices,
    ) as convert:
        actual, _ = transform_local_topk_and_prepare_pool_view(topk_indices, metadata, layer_idx=0)

    torch.testing.assert_close(actual, expected_indices)
    convert.assert_called_once_with(req_idx, block_table, topk_indices, 4, 1, 12, 1)


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


def _calc_diff(x: torch.Tensor, y: torch.Tensor):
    """Return a global difference metric for unit tests.

    DeepGEMM kernels on Blackwell/B200 currently exhibit noticeable per-element
    error, causing ``torch.testing.assert_close`` to fail.  Instead of checking
    every element, we compute a cosine-style similarity over the whole tensor
    and report ``1 - sim``.  Once kernel accuracy improves this helper can be
    removed.
    """

    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def _ref_fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    num_tokens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
):
    """
    Reference implementation of fp8_paged_mqa_logits (optimized version).

    Args:
        q: [batch_size, next_n, num_heads, head_dim]
        kv_cache: [num_blocks, block_size, 1, head_dim]
        weights: [batch_size * next_n, num_heads]
        num_tokens: [batch_size]
        block_tables: [batch_size, max_num_blocks]
        max_model_len: Maximum sequence length

    Returns:
        logits: [batch_size * next_n, max_model_len]
    """
    batch_size, next_n, _, _ = q.size()
    _, block_size, _, _ = kv_cache.size()

    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )

    num_tokens_list = num_tokens.tolist()

    for i in range(batch_size):
        num_token = num_tokens_list[i]

        # Query positions: [num_token - next_n, ..., num_token - 1]
        q_offsets = torch.arange(num_token - next_n, num_token, device="cuda")

        # Transpose weights for this sequence: [num_heads, next_n]
        weight_slice = weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()

        # Process each block in the sequence
        for block_rk in range(cdiv(num_token, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache[block_idx]

            # Key positions in this block
            k_offsets = torch.arange(
                block_rk * block_size,
                (block_rk + 1) * block_size,
                device="cuda",
            )

            # Causal mask: k_pos < num_token AND k_pos <= q_pos
            k_mask = k_offsets[None, :] < num_token
            causal_mask = k_offsets[None, :] <= q_offsets[:, None]
            mask = k_mask & causal_mask

            # Compute attention scores: [num_heads, next_n, block_size]
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(logits.dtype),
                float("-inf"),
            )

            # Apply ReLU, multiply by weights, and sum over heads
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)  # [next_n, block_size]

            # Write to output with additional causal mask
            logits[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = torch.where(causal_mask, s, float("-inf"))

    return logits


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


_TOP_K_OPS = (
    "indexer_topk_decode_gvr",
    "indexer_topk_decode",
    "indexer_topk_prefill_gvr",
    "indexer_topk_prefill",
)


def _use_torch_top_k(monkeypatch) -> None:
    """Run the indexer's Top-K ops through their plain-torch references (``torch.topk`` / a stable
    sort over each row's window): an independent selection over the same logits."""
    for name in _TOP_K_OPS:
        monkeypatch.setattr(dsa_indexer, name, getattr(dsa_indexer, name).reference)


@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@skip_pre_hopper
def test_recompute_slot_mappings_matches_prepare_with_cached_tokens(dsa_cache_manager):
    """Recompute slot mappings without re-running full K31Indexer.prepare()."""
    head_dim = 128
    block_size = 64
    request_ids = [0, 1]
    cached_tokens = [64, 128]
    seq_lens = torch.tensor([32, 4], dtype=torch.int32)
    kv_lens = torch.tensor([96, 132], dtype=torch.int32)
    num_ctx_tokens = seq_lens[0].item()
    num_tokens = seq_lens.sum().item()

    cache_manager, _ = dsa_cache_manager(
        batch_size=len(request_ids),
        head_dim=head_dim,
        tokens_per_block=block_size,
        max_seq_len=256,
        num_layers=1,
    )
    cache_manager.add_dummy_requests(request_ids, kv_lens.tolist(), is_gen=False)
    metadata = _create_mock_metadata(
        request_ids,
        len(request_ids),
        num_contexts=1,
        num_generations=1,
        seq_lens=seq_lens,
        kv_lens=kv_lens,
        num_cached_tokens=cached_tokens,
        cache_manager=cache_manager,
        num_ctx_tokens=num_ctx_tokens,
        num_tokens=num_tokens,
        indexer_head_dim=head_dim,
    )

    K31Indexer.prepare(metadata)
    expected_fp8 = metadata.slot_mapping_fp8[:num_tokens].clone()
    expected_scale = metadata.slot_mapping_scale[:num_tokens].clone()

    metadata.host_slot_mapping_fp8.zero_()
    metadata.host_slot_mapping_scale.zero_()
    metadata.slot_mapping_fp8.zero_()
    metadata.slot_mapping_scale.zero_()
    K31Indexer.recompute_slot_mappings(metadata)

    torch.testing.assert_close(metadata.slot_mapping_fp8[:num_tokens], expected_fp8)
    torch.testing.assert_close(metadata.slot_mapping_scale[:num_tokens], expected_scale)


@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@skip_pre_hopper
def test_fp8_k_cache_roundtrip(dsa_cache_manager):
    """Verify FP8 quantization scales survive write/read cycle for multiple requests."""
    torch.manual_seed(42)

    # Setup with 2 requests, each spanning multiple blocks
    batch_size = 2
    head_dim, block_size = 128, 64
    max_seq_len = 512
    num_tokens_per_req = [150, 100]  # Request 0: 3 blocks, Request 1: 2 blocks

    cache_manager, sparse_attn_config = dsa_cache_manager(
        batch_size=batch_size,
        head_dim=head_dim,
        tokens_per_block=block_size,
        max_seq_len=max_seq_len,
        num_layers=1,
    )
    indexer = create_indexer(sparse_attn_config, layer_idx=0)

    # Allocate blocks for both requests
    request_ids = [0, 1]
    cache_manager.add_dummy_requests(request_ids, num_tokens_per_req, is_gen=False)

    # Prepare and write data for each request
    total_tokens = sum(num_tokens_per_req)
    metadata = _create_mock_metadata(
        request_ids,
        batch_size,
        num_contexts=batch_size,
        num_generations=0,
        seq_lens=torch.tensor(num_tokens_per_req, dtype=torch.int32),
        kv_lens=torch.tensor(num_tokens_per_req, dtype=torch.int32),
        num_cached_tokens=[0] * batch_size,
        cache_manager=cache_manager,
        num_ctx_tokens=total_tokens,
        num_tokens=total_tokens,
        indexer_head_dim=head_dim,
    )
    K31Indexer.prepare(metadata)

    # Generate unique patterns for each request and quantize
    k_original = torch.randn((total_tokens, head_dim), device="cuda", dtype=torch.bfloat16)
    k_fp8, k_scale = fp8_utils.fp8_quantize_1x128_sf_transpose(k_original)

    # Write to cache
    indexer._update_k_cache(k_fp8, k_scale, metadata)

    # Verify scales for both requests
    cache_flat = cache_manager.indexer_k_cache_pool_per_layer[0]  # [num_blocks, flat_bytes]

    scale_offset = block_size * head_dim  # Scales start after FP8 data
    scale_size = 4  # float32
    original_scales = k_scale.cpu().numpy()

    # Verify scales for all requests
    global_token_idx = 0
    for req_idx, num_tokens in enumerate(num_tokens_per_req):
        for local_token_idx in range(num_tokens):
            # Compute block location
            block_idx_in_seq = local_token_idx // block_size
            pos_in_block = local_token_idx % block_size
            block_id = metadata.host_indexer_k_cache_block_offsets[req_idx, block_idx_in_seq].item()

            # Extract stored scale
            scale_bytes = cache_flat[
                block_id,
                scale_offset + pos_in_block * scale_size : scale_offset
                + (pos_in_block + 1) * scale_size,
            ]
            stored_scale = scale_bytes.view(torch.float32).item()

            # Compare with original
            orig_scale = original_scales[global_token_idx]
            assert abs(orig_scale - stored_scale) < 1e-6, (
                f"Request {req_idx}, token {local_token_idx} (block {block_idx_in_seq}, pos {pos_in_block}): "
                f"scale mismatch (orig={orig_scale:.6f}, stored={stored_scale:.6f})"
            )

            global_token_idx += 1


@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@skip_pre_hopper
# next_n=5 exercises the native paged-MQA path (no buffer expansion). See
# DSAtrtllmAttentionMetadata.prepare_for_spec_decode.
@pytest.mark.parametrize("batch_size,next_n", [(4, 1), (2, 2), (4, 3), (4, 4), (2, 5)])
@pytest.mark.parametrize(
    "backend",
    [
        "deepgemm",
        pytest.param(
            "dsl",
            marks=pytest.mark.skipif(
                get_sm_version() not in (100, 103),
                reason=f"CuTe DSL FP8 Paged MQA Logits only supports SM 100/103, got SM {get_sm_version()}",
            ),
        ),
    ],
)
def test_indexer_decode_with_paged_kv_cache(
    batch_size,
    next_n,
    backend,
    dsa_cache_manager,
):
    """
    Test FP8 paged KV cache with two-phase workflow and variable context lengths.

    Validates:
    - Variable-length sequences in the same batch
    - Slot mapping computation for non-interleaved FP8 cache layout
    - Vectorized scatter operations for FP8 data and scales
    - Block table management for paged attention
    - Multi-token generation phase (similar to prepare_resources)
    - Kernel accuracy vs reference implementation
    """
    torch.manual_seed(123)
    random.seed(123)

    use_dsl = backend == "dsl"
    heads, head_dim = (32, 128)
    block_size = 64
    avg_context_len = 2048
    num_gen_tokens = next_n  # Number of tokens to generate per sequence
    max_model_len = 4096
    layer_idx = 0

    # Generate variable context lengths per sequence
    context_lens_context = torch.randint(
        int(0.7 * avg_context_len),
        int(1.4 * avg_context_len),
        (batch_size,),
        dtype=torch.int32,
        device="cpu",
    )

    # Final lengths after generation phase
    final_lens = context_lens_context + num_gen_tokens
    max_seq_len = final_lens.max().item()
    total_context_tokens = context_lens_context.sum().item()
    total_gen_tokens = batch_size * num_gen_tokens

    # Setup: Create cache manager and indexer
    cache_manager, sparse_attn_config = dsa_cache_manager(
        batch_size=batch_size,
        head_dim=head_dim,
        tokens_per_block=block_size,
        max_seq_len=max_seq_len,
        num_layers=1,
    )
    indexer = create_indexer(sparse_attn_config, layer_idx=layer_idx)

    # Allocate blocks for all sequences (max final length)
    request_ids = list(range(batch_size))
    cache_manager.add_dummy_requests(
        request_ids=request_ids,
        token_nums=final_lens.tolist(),
        is_gen=False,
    )

    # Generate test data with variable lengths
    q = torch.randn((batch_size, next_n, heads, head_dim), device="cuda", dtype=torch.bfloat16)
    weights = torch.randn((batch_size * next_n, heads), device="cuda", dtype=torch.float32)
    k_context_bf16 = torch.randn(
        (total_context_tokens, head_dim), device="cuda", dtype=torch.bfloat16
    )
    k_gen_bf16 = torch.randn((total_gen_tokens, head_dim), device="cuda", dtype=torch.bfloat16)

    # Phase 1: Write context tokens (variable per sequence) as FP8
    metadata_context = _create_mock_metadata(
        request_ids,
        batch_size,
        num_contexts=batch_size,
        num_generations=0,
        seq_lens=context_lens_context.clone(),
        kv_lens=context_lens_context.clone(),
        num_cached_tokens=[0] * batch_size,
        cache_manager=cache_manager,
        num_ctx_tokens=total_context_tokens,
        num_tokens=total_context_tokens,
        max_draft_tokens=next_n - 1,
        use_cute_dsl_paged_mqa_logits=use_dsl,
        indexer_head_dim=head_dim,
    )
    K31Indexer.prepare(metadata_context)

    k_context_fp8, k_context_scale = fp8_utils.fp8_quantize_1x128_sf_transpose(k_context_bf16)

    indexer._update_k_cache(k_context_fp8, k_context_scale, metadata_context)

    # Phase 2: Write generation tokens (next_n per sequence) as FP8
    # Similar to prepare_resources: add_token() for each new token
    metadata_gen = _create_mock_metadata(
        request_ids,
        batch_size,
        num_contexts=0,
        num_generations=batch_size,
        seq_lens=torch.tensor([num_gen_tokens] * batch_size, dtype=torch.int32, device="cpu"),
        kv_lens=final_lens.clone(),
        num_cached_tokens=context_lens_context.tolist(),
        cache_manager=cache_manager,
        num_ctx_tokens=0,
        num_tokens=batch_size * num_gen_tokens,
        max_draft_tokens=next_n - 1,
        use_cute_dsl_paged_mqa_logits=use_dsl,
        indexer_head_dim=head_dim,
    )
    K31Indexer.prepare(metadata_gen)

    k_gen_fp8, k_gen_scale = fp8_utils.fp8_quantize_1x128_sf_transpose(k_gen_bf16)
    indexer._update_k_cache(k_gen_fp8, k_gen_scale, metadata_gen)

    # Run kernel: FP8 paged MQA with actual cache
    kv_cache_fp8_pool = cache_manager.get_indexer_k_cache_buffers(layer_idx)
    q_fp8 = q.to(torch.float8_e4m3fn)

    # DeepGEMM's 2D context_lens: shape (batch_size, next_n).
    context_lens = metadata_gen.kv_lens_cuda_2d[0:batch_size, 0:next_n].contiguous()
    block_table = metadata_gen.indexer_k_cache_block_offsets[0:batch_size]
    # The DeepGEMM paged MQA logits kernel takes the base scheduler layout for every next_n
    # it supports; the DSL path uses its own schedule buffer (see below).
    scheduler_metadata_buffer = metadata_gen.scheduler_metadata_buffer

    if use_dsl:
        # DSL design: kNumNextNAtoms = 1 for any real next_n. Reuse the base
        # `scheduler_metadata_buffer` which is also built with num_atoms=1
        # (from a (num_gen, 1) input shape). 1D contiguous kv_lens slice
        # suffices for context_lens (all next_n positions share the same KV
        # length on this path; kernel signature is 1D).
        dsl_context_lens = metadata_gen.gen_indexer_kv_lens_cuda_runtime
        assert dsl_context_lens is not None
        logits = torch.ops.trtllm.cute_dsl_fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8_pool,
            weights,
            dsl_context_lens,
            block_table,
            metadata_gen.scheduler_metadata_buffer,
            max_model_len,
        )
    else:
        logits = fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8_pool,
            weights,
            context_lens,
            block_table,
            scheduler_metadata_buffer,
            max_model_len,
        )

    # Reference: Reconstruct BF16 cache from original values
    num_blocks = kv_cache_fp8_pool.shape[0]
    kv_cache_bf16 = torch.zeros(
        (num_blocks, block_size, 1, head_dim), device="cuda", dtype=torch.bfloat16
    )

    # Populate cache with variable-length sequences
    context_offset = 0
    gen_offset = 0
    for seq_idx in range(batch_size):
        seq_context_len = context_lens_context[seq_idx].item()
        seq_gen_len = num_gen_tokens

        # Write context tokens
        for token_pos in range(seq_context_len):
            block_idx = token_pos // block_size
            pos_in_block = token_pos % block_size
            physical_block_id = metadata_gen.indexer_k_cache_block_offsets[
                seq_idx, block_idx
            ].item()
            if physical_block_id >= 0:
                kv_cache_bf16[physical_block_id, pos_in_block, 0, :] = k_context_bf16[
                    context_offset + token_pos
                ]

        # Write generation tokens
        for gen_token_idx in range(seq_gen_len):
            token_pos = seq_context_len + gen_token_idx
            block_idx = token_pos // block_size
            pos_in_block = token_pos % block_size
            physical_block_id = metadata_gen.indexer_k_cache_block_offsets[
                seq_idx, block_idx
            ].item()
            if physical_block_id >= 0:
                kv_cache_bf16[physical_block_id, pos_in_block, 0, :] = k_gen_bf16[
                    gen_offset + gen_token_idx
                ]

        context_offset += seq_context_len
        gen_offset += seq_gen_len

    num_tokens_cuda = final_lens.cuda()
    ref_logits = _ref_fp8_paged_mqa_logits(
        q,
        kv_cache_bf16,
        weights,
        num_tokens_cuda[0:batch_size],
        metadata_gen.indexer_k_cache_block_offsets,
        max_model_len,
    )

    # Validate: Compare masked outputs (handle variable lengths and next_n)

    # Expand context lens for each query: each sequence has next_n queries
    # Query at position i (where i = 0..next_n-1) attends to tokens up to (context_len - next_n + i)
    positions = torch.arange(max_model_len, device="cuda").unsqueeze(0)  # [1, max_model_len]

    # For each query, compute its end position
    # Shape: [batch_size * next_n]
    row_indices = torch.arange(batch_size * next_n, device="cuda") // next_n  # Which sequence
    next_n_offset = (
        torch.arange(batch_size * next_n, device="cuda") % next_n
    )  # Query offset within sequence
    query_end_positions = (
        num_tokens_cuda[row_indices] - next_n + next_n_offset
    )  # [batch_size * next_n]

    # Create mask: positions <= query_end_position
    # Shape: [batch_size * next_n, max_model_len]
    mask = positions <= query_end_positions.unsqueeze(1)

    diff = _calc_diff(logits.masked_fill(~mask, 0), ref_logits.masked_fill(~mask, 0))

    assert diff < 1e-3, f"Accuracy check failed: {diff=}"


@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@skip_pre_hopper
@pytest.mark.parametrize(
    "chunk_size,seq_lens_list,chunking_type",
    [
        # Request-level chunking: small requests that fit individually but need chunking as batch
        (512, None, "request_level"),  # Random seq lens, 5 requests
        (1024, None, "request_level"),  # Random seq lens, 4 requests
        (2048, None, "request_level"),  # Random seq lens, 4 requests
        # Two-level chunking: intra-request Q-block splitting for large requests
        (1024, [512, 2500, 800], "two_level"),  # Middle request needs Q-blocks
        (1500, [3200, 1200], "two_level"),  # First request needs 3 Q-blocks
        (1000, [400, 600, 2200, 500], "two_level"),  # Mixed: request 2 needs Q-blocks
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
        batch_size = 5 if chunk_size == 512 else 4
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
    total_similarity = 0.0

    for token_idx in range(total_tokens):
        chunked_indices = topk_indices_chunked[token_idx]
        baseline_indices = topk_indices_baseline[token_idx]

        # Filter out -1 (invalid) indices
        chunked_valid = chunked_indices[chunked_indices != -1]
        baseline_valid = baseline_indices[baseline_indices != -1]

        # Check if exactly the same
        if torch.equal(chunked_valid, baseline_valid):
            total_similarity += 1.0
            continue

        # Calculate set-based similarity (Jaccard index) to handle ties
        if chunked_valid.shape[0] > 0 or baseline_valid.shape[0] > 0:
            chunked_set = set(chunked_valid.cpu().tolist())
            baseline_set = set(baseline_valid.cpu().tolist())

            intersection = len(chunked_set & baseline_set)
            union = len(chunked_set | baseline_set)
            similarity = intersection / union if union > 0 else 0.0
            total_similarity += similarity

    avg_similarity = total_similarity / total_tokens

    # Use Jaccard similarity threshold instead of exact match
    assert avg_similarity >= 0.9, (
        f"Chunked and non-chunked results differ significantly: avg similarity {avg_similarity:.4f} < 0.9"
    )


@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@skip_pre_hopper
@pytest.mark.parametrize("batch_size", [1, 16, 64])
@pytest.mark.parametrize("next_n", [1, 2, 4])
@pytest.mark.parametrize("index_topk", [2048])
def test_indexer_decode_custom_vs_fallback(
    batch_size,
    next_n,
    index_topk,
    dsa_cache_manager,
    monkeypatch,
):
    """
    Test that the production Top-K kernels and the plain-torch Top-K references select
    the same keys in the decode phase of sparse_attn_indexer.

    This test validates:
    1. Custom CUDA top-k kernel (indexer_topk_decode) correctness
    2. Consistency with the plain-torch Top-K references (torch.topk over each row's window)
    3. Handling of decode scenarios with next_n > 1 (speculative decoding)
    4. Proper masking and handling of variable-length sequences

    Test scenarios:
    - Different batch sizes
    - Different next_n values (1, 2, 4 for speculative decode)
    - Short sequences (512, 1024): indexer skip must match the real decode Top-K
    """
    torch.manual_seed(42)
    random.seed(42)

    # Test parameters
    heads, head_dim = 32, 128
    block_size = 64
    max_model_len = 16384
    layer_idx = 0

    # KV cache lengths at most index_topk, so the indexer can be skipped
    kv_lens = torch.randint(512, 1024, (batch_size,), dtype=torch.int32)

    seq_lens = torch.full((batch_size,), next_n, dtype=torch.int32)
    num_gen_tokens = batch_size * next_n
    num_cached_tokens = kv_lens.tolist()

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

    # Allocate blocks for all sequences (including historical + new tokens)
    request_ids = list(range(batch_size))
    final_lens = kv_lens + next_n  # Historical + new decode tokens
    cache_manager.add_dummy_requests(
        request_ids=request_ids, token_nums=final_lens.tolist(), is_gen=False
    )

    # Populate KV cache with historical context
    total_context_tokens = kv_lens.sum().item()
    k_context_bf16 = torch.randn(
        (total_context_tokens, head_dim), device="cuda", dtype=torch.bfloat16
    )
    k_context_fp8, k_context_scale = fp8_utils.fp8_quantize_1x128_sf_transpose(k_context_bf16)

    metadata_context = _create_mock_metadata(
        request_ids=request_ids,
        batch_size=batch_size,
        num_contexts=batch_size,
        num_generations=0,
        seq_lens=kv_lens.clone(),
        kv_lens=kv_lens.clone(),
        num_cached_tokens=[0] * batch_size,
        cache_manager=cache_manager,
        num_ctx_tokens=total_context_tokens,
        num_tokens=total_context_tokens,
        max_draft_tokens=next_n - 1,
        indexer_head_dim=head_dim,
    )
    K31Indexer.prepare(metadata_context)
    indexer._update_k_cache(k_context_fp8, k_context_scale, metadata_context)

    # Generate decode phase test data
    q = torch.randn((num_gen_tokens, heads, head_dim), device="cuda", dtype=torch.bfloat16)
    k_gen_bf16 = torch.randn((num_gen_tokens, head_dim), device="cuda", dtype=torch.bfloat16)
    weights = torch.randn((num_gen_tokens, heads), device="cuda", dtype=torch.float32)
    hidden_states = torch.randn((num_gen_tokens, 4096), device="cuda", dtype=torch.bfloat16)

    q_fp8 = q.to(torch.float8_e4m3fn)
    k_fp8, k_scale = fp8_utils.fp8_quantize_1x128_sf_transpose(k_gen_bf16)

    metadata_gen_write = _create_mock_metadata(
        request_ids=request_ids,
        batch_size=batch_size,
        num_contexts=0,
        num_generations=batch_size,
        seq_lens=seq_lens.clone(),
        kv_lens=final_lens.clone(),
        num_cached_tokens=num_cached_tokens,
        cache_manager=cache_manager,
        num_ctx_tokens=0,
        num_tokens=num_gen_tokens,
        max_draft_tokens=next_n - 1,
        indexer_head_dim=head_dim,
    )
    K31Indexer.prepare(metadata_gen_write)
    indexer._update_k_cache(k_fp8, k_scale, metadata_gen_write)

    # Test with custom CUDA kernel
    metadata_custom = _create_mock_metadata(
        request_ids,
        batch_size,
        0,
        batch_size,
        seq_lens.clone(),
        final_lens.clone(),
        num_cached_tokens,
        cache_manager,
        0,
        num_gen_tokens,
        max_model_len,
        max_draft_tokens=next_n - 1,
        indexer_head_dim=head_dim,
    )

    K31Indexer.prepare(metadata_custom)
    indexer._update_k_cache(k_fp8, k_scale, metadata_custom)

    topk_indices_custom = indexer.sparse_attn_indexer(
        metadata_custom, hidden_states, q_fp8, k_fp8, k_scale, weights
    )

    # Test with the plain-torch Top-K references
    metadata_fallback = _create_mock_metadata(
        request_ids,
        batch_size,
        0,
        batch_size,
        seq_lens.clone(),
        final_lens.clone(),
        num_cached_tokens,
        cache_manager,
        0,
        num_gen_tokens,
        max_model_len,
        max_draft_tokens=next_n - 1,
        indexer_head_dim=head_dim,
    )

    K31Indexer.prepare(metadata_fallback)
    indexer._update_k_cache(k_fp8, k_scale, metadata_fallback)
    with monkeypatch.context() as torch_top_k:
        _use_torch_top_k(torch_top_k)
        topk_indices_fallback = indexer.sparse_attn_indexer(
            metadata_fallback, hidden_states, q_fp8, k_fp8, k_scale, weights
        ).clone()

    # Test with indexer skip enabled
    metadata_skip = _create_mock_metadata(
        request_ids,
        batch_size,
        0,
        batch_size,
        seq_lens.clone(),
        final_lens.clone(),
        num_cached_tokens,
        cache_manager,
        0,
        num_gen_tokens,
        max_model_len,
        max_draft_tokens=next_n - 1,
        enable_indexer_skip=True,
        indexer_head_dim=head_dim,
    )

    K31Indexer.prepare(metadata_skip)
    indexer._update_k_cache(k_fp8, k_scale, metadata_skip)
    topk_indices_skip = indexer.sparse_attn_indexer(
        metadata_skip, hidden_states, q_fp8, k_fp8, k_scale, weights
    )

    # Validation
    ## Custom vs fallback
    num_ctx_tokens = 0
    custom_decode = topk_indices_custom[num_ctx_tokens : num_ctx_tokens + num_gen_tokens, :]
    fallback_decode = topk_indices_fallback[num_ctx_tokens : num_ctx_tokens + num_gen_tokens, :]
    num_exact_matches, total_similarity, _ = validate_topk_indices(
        custom_decode, fallback_decode, num_gen_tokens
    )
    avg_similarity = total_similarity / num_gen_tokens
    assert avg_similarity >= 0.95, (
        f"Decode custom vs fallback differ: avg similarity {avg_similarity:.4f} < 0.95"
    )
    ## Custom vs skip
    skip_decode = topk_indices_skip[num_ctx_tokens : num_ctx_tokens + num_gen_tokens, :]
    num_exact_matches, total_similarity, _ = validate_topk_indices(
        custom_decode, skip_decode, num_gen_tokens
    )
    avg_similarity = total_similarity / num_gen_tokens
    assert avg_similarity >= 0.95, (
        f"Decode custom vs skip differ: avg similarity {avg_similarity:.4f} < 0.95"
    )


@skip_pre_hopper
@pytest.mark.parametrize("enable_indexer_skip", [True, False])
def test_indexer_topk_multi_request_with_different_cache(
    enable_indexer_skip, dsa_cache_manager, monkeypatch
):
    """
    Test that custom topk kernel handles multi-request batches with different cached amounts.
    """
    torch.manual_seed(42)

    # Parameters matching your problematic case
    batch_size = 2
    heads, head_dim = 64, 128
    block_size = 64
    index_topk = 2048
    max_model_len = 10240
    layer_idx = 0

    # Critical: different cached amounts
    seq_lens = [256, 237]  # NEW tokens
    if enable_indexer_skip:
        cached_tokens = [256, 584]  # Req0: no cache, Req1: short cache
    else:
        cached_tokens = [0, 3584]  # Req0: no cache, Req1: large cache
    total_kv_lens = [seq_lens[i] + cached_tokens[i] for i in range(batch_size)]
    total_tokens = sum(seq_lens)

    # Create cache manager
    cache_manager, sparse_attn_config = dsa_cache_manager(
        batch_size=batch_size,
        head_dim=head_dim,
        tokens_per_block=block_size,
        max_seq_len=max_model_len,
        num_layers=1,
        index_topk=index_topk,
    )
    indexer = create_indexer(sparse_attn_config, layer_idx=layer_idx)

    # Allocate blocks
    request_ids = [0, 1]
    cache_manager.add_dummy_requests(request_ids, total_kv_lens, is_gen=False)

    # Generate test data
    q = torch.randn((total_tokens, heads, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((total_tokens, head_dim), device="cuda", dtype=torch.bfloat16)
    weights = torch.randn((total_tokens, heads), device="cuda", dtype=torch.float32)
    hidden_states = torch.randn((total_tokens, 4096), device="cuda", dtype=torch.bfloat16)

    q_fp8 = q.to(torch.float8_e4m3fn)
    k_fp8, k_scale = fp8_utils.fp8_quantize_1x128_sf_transpose(k)

    # Create metadata with chunked prefill enabled
    metadata = _create_mock_metadata(
        request_ids,
        batch_size,
        batch_size,
        0,
        torch.tensor(seq_lens, dtype=torch.int32),
        torch.tensor(total_kv_lens, dtype=torch.int32),
        cached_tokens,
        cache_manager,
        total_tokens,
        total_tokens,
        indexer_max_chunk_size=32768,
        enable_context_mla_with_cached_kv=True,
        indexer_head_dim=head_dim,
    )

    K31Indexer.prepare(metadata)
    if not enable_indexer_skip:
        assert metadata.slot_mapping_fp8_fullkv is not metadata.slot_mapping_fp8
        assert metadata.slot_mapping_scale_fullkv is not metadata.slot_mapping_scale
    indexer._update_k_cache(k_fp8, k_scale, metadata)

    # Test custom kernel
    topk_custom = indexer.sparse_attn_indexer(
        metadata, hidden_states, q_fp8, k_fp8, k_scale, weights
    ).clone()

    # Test the plain-torch Top-K references
    with monkeypatch.context() as torch_top_k:
        _use_torch_top_k(torch_top_k)
        topk_fallback = indexer.sparse_attn_indexer(
            metadata, hidden_states, q_fp8, k_fp8, k_scale, weights
        ).clone()

    # Test with indexer skip enabled
    if enable_indexer_skip:
        metadata_skip = _create_mock_metadata(
            request_ids,
            batch_size,
            batch_size,
            0,
            torch.tensor(seq_lens, dtype=torch.int32),
            torch.tensor(total_kv_lens, dtype=torch.int32),
            cached_tokens,
            cache_manager,
            total_tokens,
            total_tokens,
            indexer_max_chunk_size=32768,
            enable_context_mla_with_cached_kv=True,
            enable_indexer_skip=True,
            indexer_head_dim=head_dim,
        )
        K31Indexer.prepare(metadata_skip)
        indexer._update_k_cache(k_fp8, k_scale, metadata_skip)
        topk_indices_skip = indexer.sparse_attn_indexer(
            metadata_skip, hidden_states, q_fp8, k_fp8, k_scale, weights
        )

    # Tokens with large windows (>= 2048) have exactly 2048 valid indices.

    host_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device="cpu")
    host_cached = torch.tensor(cached_tokens, dtype=torch.int32, device="cpu")
    cu_ks, cu_ke = compute_cu_seqlen_kv_bounds_with_cache(
        host_seq_lens, batch_size, total_tokens, host_cached
    )

    for tok_id in range(total_tokens):
        window_size = (cu_ke[tok_id] - cu_ks[tok_id]).item()
        if window_size >= index_topk:
            num_valid = (topk_custom[tok_id] >= 0).sum().item()
            num_valid_fallback = (topk_fallback[tok_id] >= 0).sum().item()
            assert num_valid_fallback == index_topk, (
                f"[Fallback] Token {tok_id}: window={window_size}, "
                f"but only {num_valid_fallback}/{index_topk} valid indices"
            )
            assert num_valid == index_topk, (
                f"[Custom Topk] Token {tok_id}: window={window_size}, but only {num_valid}/{index_topk} valid indices"
            )

    # Validation
    num_exact_matches, total_similarity, min_similarity = validate_topk_indices(
        topk_custom, topk_fallback, total_tokens
    )
    avg_similarity = total_similarity / total_tokens

    assert avg_similarity >= 0.95, (
        f"Custom vs fallback differ: avg similarity {avg_similarity:.4f} < 0.95"
    )

    if enable_indexer_skip:
        num_exact_matches, total_similarity, min_similarity = validate_topk_indices(
            topk_custom, topk_indices_skip, total_tokens
        )
        avg_similarity = total_similarity / total_tokens
        assert avg_similarity >= 0.95, (
            f"Custom vs indexer skip differ: avg similarity {avg_similarity:.4f} < 0.95"
        )


def _mla_prefill_reuse_case(monkeypatch, dsa_cache_manager):
    """Build a reuse-hit MLA chunked-prefill batch; return a runner over it.

    Four context requests bring few new tokens each on top of long cached
    prefixes. The runner prepares the batch with a given packed-logits budget
    (and optionally a TP mapping) and returns the planned chunks and top-k.
    """
    torch.manual_seed(0)
    heads, head_dim = 64, 128
    block_size = 64
    index_topk = 2048
    max_model_len = 10240

    seq_lens = [237, 300, 256, 199]
    cached_tokens = [3584, 3072, 0, 4096]
    batch_size = len(seq_lens)
    total_kv_lens = [s + c for s, c in zip(seq_lens, cached_tokens)]
    total_tokens = sum(seq_lens)

    cache_manager, sparse_attn_config = dsa_cache_manager(
        batch_size=batch_size,
        head_dim=head_dim,
        tokens_per_block=block_size,
        max_seq_len=max_model_len,
        num_layers=1,
        index_topk=index_topk,
    )
    indexer = create_indexer(sparse_attn_config, layer_idx=0)
    request_ids = list(range(batch_size))
    cache_manager.add_dummy_requests(request_ids, total_kv_lens, is_gen=False)

    # Write distinct K for every position first, so the cached prefixes hold
    # real values rather than ties that either chunking may break differently.
    k_full = torch.randn((sum(total_kv_lens), head_dim), device="cuda", dtype=torch.bfloat16)
    k_full_fp8, k_full_scale = fp8_utils.fp8_quantize_1x128_sf_transpose(k_full)
    fill_metadata = _create_mock_metadata(
        request_ids,
        batch_size,
        batch_size,
        0,
        torch.tensor(total_kv_lens, dtype=torch.int32),
        torch.tensor(total_kv_lens, dtype=torch.int32),
        [0] * batch_size,
        cache_manager,
        sum(total_kv_lens),
        sum(total_kv_lens),
        indexer_max_chunk_size=32768,
        indexer_head_dim=head_dim,
    )
    K31Indexer.prepare(fill_metadata)
    indexer._update_k_cache(k_full_fp8, k_full_scale, fill_metadata)

    # The new tokens are the tail of each request's sequence.
    kv_starts = [sum(total_kv_lens[:i]) for i in range(batch_size)]
    new_rows = torch.cat(
        [
            torch.arange(start + cached, start + cached + new)
            for start, cached, new in zip(kv_starts, cached_tokens, seq_lens)
        ]
    ).cuda()
    k = k_full[new_rows]
    q = torch.randn((total_tokens, heads, head_dim), device="cuda", dtype=torch.bfloat16)
    weights = torch.randn((total_tokens, heads), device="cuda", dtype=torch.float32)
    hidden_states = torch.randn((total_tokens, 4096), device="cuda", dtype=torch.bfloat16)
    q_fp8 = q.to(torch.float8_e4m3fn)
    k_fp8, k_scale = fp8_utils.fp8_quantize_1x128_sf_transpose(k)

    def run(elem_budget, mapping=None):
        monkeypatch.setattr(dsa_indexer_plan, "_INDEXER_PACKED_LOGITS_ELEM_BUDGET", elem_budget)
        metadata = _create_mock_metadata(
            request_ids,
            batch_size,
            batch_size,
            0,
            torch.tensor(seq_lens, dtype=torch.int32),
            torch.tensor(total_kv_lens, dtype=torch.int32),
            cached_tokens,
            cache_manager,
            total_tokens,
            total_tokens,
            indexer_max_chunk_size=32768,
            enable_context_mla_with_cached_kv=True,
            indexer_head_dim=head_dim,
        )
        metadata.mapping = mapping
        K31Indexer.prepare(metadata)
        indexer._update_k_cache(k_fp8, k_scale, metadata)
        topk = indexer.sparse_attn_indexer(
            metadata, hidden_states, q_fp8, k_fp8, k_scale, weights
        ).clone()
        return metadata, topk

    return run, seq_lens, total_kv_lens, total_tokens


@skip_pre_hopper
def test_indexer_mla_prefill_groups_requests_by_logits_budget(monkeypatch, dsa_cache_manager):
    """Grouping reuse-hit requests into separate chunks must not change top-k.

    Under MLA chunked prefill, packing every context request into one indexer
    chunk makes its logits [sum(new tokens) x sum(cached + new KV)]. With
    prefix reuse each request brings few new tokens but a long cached context,
    so that product grows with the square of the request count. The budget
    splits such batches; each query row still scores only its own request's
    window, so the selected indices must match the packed run.
    """
    run, seq_lens, total_kv_lens, total_tokens = _mla_prefill_reuse_case(
        monkeypatch, dsa_cache_manager
    )
    packed_metadata, topk_packed = run(1 << 62)
    split_metadata, topk_split = run(0)
    packed_chunks = packed_metadata.indexer_prefill_chunks
    split_chunks = split_metadata.indexer_prefill_chunks

    assert len(packed_chunks) == 1
    assert len(split_chunks) == len(seq_lens)
    q_start = kv_start = 0
    for chunk, q_len, kv_len in zip(split_chunks, seq_lens, total_kv_lens):
        assert (chunk.token_start, chunk.token_end) == (q_start, q_start + q_len)
        assert (chunk.k_token_start, chunk.k_token_end) == (kv_start, kv_start + kv_len)
        q_start += q_len
        kv_start += kv_len

    # Top-k order within a row is unspecified; the selected sets must be equal.
    _, _, min_similarity = validate_topk_indices(topk_split, topk_packed, total_tokens)
    assert min_similarity == 1.0, f"split vs packed select different indices: {min_similarity}"


@skip_pre_hopper
@pytest.mark.parametrize("tp_size", [2, 4])
def test_indexer_mla_prefill_tp_split_matches_unsplit(monkeypatch, tp_size, dsa_cache_manager):
    """Splitting a grouped MLA prefill batch across TP ranks must not change top-k.

    Each emulated rank plans chunks for its share of the context queries only,
    including requests cut at the share boundaries, and all-gathers its top-k
    rows once. Concatenating the gathered shares must reproduce the top-k of
    the unsplit run.
    """
    run, seq_lens, _, total_tokens = _mla_prefill_reuse_case(monkeypatch, dsa_cache_manager)
    # Pairs of requests fit the budget. With TP2 the second rank's share holds
    # a cut request followed by two whole ones grouped into one chunk; with TP4
    # every share is too short to hold two whole requests.
    elem_budget = 600 * 8000
    request_bounds = [sum(seq_lens[:i]) for i in range(1, len(seq_lens))]
    _, topk_unsplit = run(elem_budget)

    monkeypatch.setattr(dsa_indexer, "_prefill_q_split_threshold", lambda metadata: 1)
    gathered = []

    def fake_allgather(tensor, mapping, dim=-1, sizes=None):
        gathered.append((mapping.tp_rank, tensor.clone(), list(sizes)))
        return torch.full(
            (sum(sizes), *tensor.shape[1:]), -2, dtype=tensor.dtype, device=tensor.device
        )

    # The indexer gathers through the family allgather entry, which calls the engine allgather.
    allgather_module_mod = importlib.import_module(
        "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm.allgather_module"
    )

    monkeypatch.setattr(allgather_module_mod, "allgather", fake_allgather)
    expected_sizes = [
        (r + 1) * total_tokens // tp_size - r * total_tokens // tp_size for r in range(tp_size)
    ]
    has_grouped_chunk = False
    for rank in range(tp_size):
        mapping = Mapping(world_size=tp_size, rank=rank, tp_size=tp_size, pp_size=1)
        metadata, _ = run(elem_budget, mapping)
        q_begin, q_end, q_sizes = metadata.indexer_prefill_q_split
        assert q_sizes == expected_sizes
        assert (q_begin, q_end) == (sum(expected_sizes[:rank]), sum(expected_sizes[: rank + 1]))
        chunks = metadata.indexer_prefill_chunks
        assert [(c.token_start, c.token_end) for c in chunks] == list(
            zip([q_begin] + [c.token_end for c in chunks[:-1]], [c.token_end for c in chunks])
        )
        assert chunks[0].token_start == q_begin and chunks[-1].token_end == q_end
        has_grouped_chunk |= any(
            c.token_start < bound < c.token_end for c in chunks for bound in request_bounds
        )

    assert has_grouped_chunk == (tp_size == 2)
    assert [rank for rank, _, _ in gathered] == list(range(tp_size)), "one allgather per rank"
    topk_split = torch.cat([rows for _, rows, _ in gathered])
    _, _, min_similarity = validate_topk_indices(topk_split, topk_unsplit, total_tokens)
    assert min_similarity == 1.0, f"TP split vs unsplit select different indices: {min_similarity}"


@pytest.mark.cpu_only
class TestPrepareRestoreAttnMetadataForDraftReplay:
    """Tests for prepare_attn_metadata_for_draft_replay and
    restore_attn_metadata_after_draft_replay."""

    @staticmethod
    def _make_mock_metadata():
        """Create a mock attention metadata object with KV cache block offsets."""
        meta = Mock()
        meta.enable_flash_mla = False
        meta.kv_cache_manager = Mock(name="target_kv_cache_manager")
        meta.kv_cache_block_offsets = torch.tensor([10, 20, 30])
        meta.host_kv_cache_block_offsets = torch.tensor([10, 20, 30])
        meta.draft_kv_cache_block_offsets = torch.tensor([100, 200, 300])
        meta.prepare_for_draft_forward.return_value = None
        meta.draft_replay_swapped_attrs = {}
        for helper in ("record_draft_swap", "swap_for_draft", "restore_draft_swaps"):
            setattr(meta, helper, functools.partial(getattr(TrtllmAttentionMetadata, helper), meta))
        return meta

    @staticmethod
    def _make_mock_draft_manager():
        """Create a mock draft KV cache manager with host block offsets."""
        return SimpleNamespace(host_kv_cache_block_offsets=torch.tensor([100, 200, 300]))

    def test_native_dsa_replay_swaps_and_restores_buffers(self):
        """Switch native DSA metadata to draft buffers and restore it."""
        meta = self._make_mock_metadata()
        mgr = object.__new__(DSACacheManagerV2)
        mgr.host_kv_cache_block_offsets = torch.tensor([100, 200, 300])
        # The family DSA metadata recognizes a manager that declares the DSA indexer K cache.
        mgr.provides_dsa_indexer_k_cache = True
        meta.enable_context_mla_with_cached_kv = False
        meta._invalidate_pool_view_cache = Mock()
        buffer_pairs = {
            "host_indexer_k_cache_block_offsets": "host_draft_indexer_k_cache_block_offsets",
            "indexer_k_cache_block_offsets": "draft_indexer_k_cache_block_offsets",
            "host_slot_mapping_fp8": "host_draft_slot_mapping_fp8",
            "host_slot_mapping_scale": "host_draft_slot_mapping_scale",
            "slot_mapping_fp8": "draft_slot_mapping_fp8",
            "slot_mapping_scale": "draft_slot_mapping_scale",
            "block_table": "draft_block_table",
            "block_table_expanded": "draft_block_table_expanded",
            "host_block_table_expanded": "host_draft_block_table_expanded",
        }
        target_buffers = {}
        draft_buffers = {}
        for target_name, draft_name in buffer_pairs.items():
            target_buffers[target_name] = object()
            draft_buffers[target_name] = object()
            setattr(meta, target_name, target_buffers[target_name])
            setattr(meta, draft_name, draft_buffers[target_name])
        del meta.slot_mapping_fp8_fullkv
        del meta.slot_mapping_scale_fullkv

        meta.prepare_for_draft_forward.side_effect = (
            lambda: K31DSAtrtllmAttentionMetadata.prepare_for_draft_forward(meta)
        )
        meta.restore_after_draft_forward.side_effect = (
            lambda saved_state: K31DSAtrtllmAttentionMetadata.restore_after_draft_forward(
                meta, saved_state
            )
        )

        def is_attn_metadata(obj, cls):
            if cls in (TrtllmAttentionMetadata, K31DSAtrtllmAttentionMetadata):
                return obj is meta
            return builtins.isinstance(obj, cls)

        with (
            patch(
                "tensorrt_llm._torch.speculative.interface.isinstance",
                side_effect=is_attn_metadata,
            ),
            patch(
                "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.dsa_metadata.torch.cuda.is_current_stream_capturing",
                return_value=True,
            ),
        ):
            saved = prepare_attn_metadata_for_draft_replay(meta, mgr)

        assert "saved_backend_state" in saved
        assert meta.slot_mapping_fp8 is draft_buffers["slot_mapping_fp8"]
        restore_attn_metadata_after_draft_replay(meta, saved)
        assert meta.slot_mapping_fp8 is target_buffers["slot_mapping_fp8"]
        assert meta.slot_mapping_fp8_fullkv is target_buffers["slot_mapping_fp8"]


@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@skip_pre_hopper
def test_topk_indices_buffer_cuda_graph(dsa_cache_manager):
    from tensorrt_llm._torch.memory_buffer_utils import Buffers

    batch_size, next_n = 4, 3
    head_dim, block_size = 128, 64
    index_topk = 2048
    kv_len = 512
    layer_idx = 0
    num_tokens = batch_size * next_n

    cache_manager, sparse_attn_config = dsa_cache_manager(
        batch_size=batch_size,
        head_dim=head_dim,
        tokens_per_block=block_size,
        max_seq_len=kv_len,
        num_layers=1,
        index_topk=index_topk,
    )
    indexer = create_indexer(sparse_attn_config, layer_idx=layer_idx)

    request_ids = list(range(batch_size))
    kv_lens = torch.tensor([kv_len] * batch_size, dtype=torch.int32)
    cache_manager.add_dummy_requests(
        request_ids=request_ids,
        token_nums=kv_lens.tolist(),
        is_gen=False,
    )

    metadata = _create_mock_metadata(
        request_ids,
        batch_size,
        num_contexts=0,
        num_generations=batch_size,
        seq_lens=torch.tensor([next_n] * batch_size, dtype=torch.int32),
        kv_lens=kv_lens.clone(),
        num_cached_tokens=[kv_len - next_n] * batch_size,
        cache_manager=cache_manager,
        num_ctx_tokens=0,
        num_tokens=num_tokens,
        max_draft_tokens=next_n - 1,
        index_topk=index_topk,
        enable_indexer_skip=True,
    )
    assert metadata.skip_indexer_for_gen_reqs, (
        "test setup must hit the skip-indexer path (kv_len <= index_topk)"
    )

    metadata.is_cuda_graph = True
    metadata.cuda_graph_buffers = Buffers()

    hidden_states = torch.zeros((num_tokens, 1), device="cuda", dtype=torch.bfloat16)
    dummy = torch.zeros((num_tokens, 1), device="cuda", dtype=torch.bfloat16)

    def _run_indexer():
        return indexer.sparse_attn_indexer(
            metadata,
            hidden_states,
            q_fp8=dummy,
            k_fp8=dummy,
            k_scale=dummy,
            weights=dummy,
        )

    out1 = _run_indexer()
    ptr1 = out1.data_ptr()
    out2 = _run_indexer()
    ptr2 = out2.data_ptr()

    assert ptr1 == ptr2, (
        f"indexer topk-output buffer address changed across calls "
        f"({ptr1:#x} -> {ptr2:#x}); under CUDA-graph capture it must be a "
        f"persistent reserved-arena buffer to avoid stale-pointer IMA"
    )
    assert "indexer_topk_out_buffer" in metadata.cuda_graph_buffers.buffers, (
        "indexer topk-output buffer must be drawn from the cuda_graph_buffers arena"
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_deterministic_topk_order(device):
    """Rows keep their index set, ascending, with the unused (-1) slots last, whatever order the
    radix Top-K wrote them in."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    rows = torch.tensor(
        [[7, -1, 3, 9, -1, 0], [5, 4, 3, 2, 1, 0], [-1, -1, -1, -1, -1, -1]],
        dtype=torch.int32,
        device=device,
    )
    expected = torch.tensor(
        [[0, 3, 7, 9, -1, -1], [0, 1, 2, 3, 4, 5], [-1, -1, -1, -1, -1, -1]],
        dtype=torch.int32,
        device=device,
    )
    out = dsa_indexer.deterministic_topk_order(rows)
    assert out.dtype == torch.int32
    assert torch.equal(out, expected)
    shuffled = rows[:, torch.randperm(rows.shape[1], device=device)]
    assert torch.equal(dsa_indexer.deterministic_topk_order(shuffled), expected)

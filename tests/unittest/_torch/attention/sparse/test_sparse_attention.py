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
Unit tests for sparse attention with TrtllmAttention backend.
"""

import ast
import inspect
import math
from dataclasses import dataclass
from types import ModuleType
from typing import List, Optional, Tuple

import pytest
import torch
from utils.util import getSMVersion

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionBackend,
    AttentionForwardArgs,
    SparsePrediction,
)
from tensorrt_llm._torch.attention_backend.sparse.dsa.kernels import (
    triton_convert_req_index_to_global_index,
)
from tensorrt_llm._torch.attention_backend.sparse.hooks import (
    _SPARSE_ATTN_HOOK_MODULE_PATHS,
    SparseAttnHooks,
    _get_sparse_attn_hooks_for_algorithm,
    get_sparse_attn_hooks,
)
from tensorrt_llm._torch.attention_backend.sparse.hooks import __all__ as SPARSE_ATTN_HOOK_API
from tensorrt_llm._torch.attention_backend.sparse.params import SparseParams
from tensorrt_llm._torch.attention_backend.sparse.prediction import prepare_sparse_prediction
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.modules.attention import Attention
from tensorrt_llm._torch.modules.mla import MLA
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import str_dtype_to_binding, torch_dtype_to_str
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping

ATOL = 2e-2
RTOL = 2e-2


@dataclass(kw_only=True, frozen=False)
class SparseScenario:
    """Base configuration for sparse attention tests.

    NOTE: SparseMqaGqa trtllm-gen cubins are currently only available for BF16.
    FP16 cubins have not been generated yet, so tests default to BF16.
    """

    dtype: torch.dtype = torch.bfloat16
    kvcache_dtype: torch.dtype = torch.bfloat16
    num_layers: int = 1
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    page_size: int = 32
    num_pages: int = 16
    batch_size: int = 4
    num_sparse_topk: int = 64

    @property
    def num_kv_groups(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def kv_cache_len(self) -> int:
        return self.page_size * self.num_pages

    @property
    def max_num_pages(self) -> int:
        return self.batch_size * self.num_pages


@dataclass(kw_only=True, frozen=False)
class SparseContextScenario(SparseScenario):
    """Configuration for context phase tests with sparse kv cache write."""

    seq_lens: Tuple[int, ...] = (128,)

    def __post_init__(self):
        if len(self.seq_lens) != self.batch_size:
            raise ValueError(
                f"seq_lens length {len(self.seq_lens)} must match batch_size {self.batch_size}"
            )

    @property
    def max_seq_len(self) -> int:
        return max(self.seq_lens)

    @property
    def nnz_q(self) -> int:
        return sum(self.seq_lens)


@dataclass(kw_only=True, frozen=False)
class SparseGenerationScenario(SparseScenario):
    """Configuration for generation phase tests with sparse attention."""

    past_kv_lens: Tuple[int, ...] = (256,)
    num_contexts: int = 0

    def __post_init__(self):
        if len(self.past_kv_lens) != self.batch_size:
            raise ValueError(
                f"past_kv_lens length {len(self.past_kv_lens)} must match batch_size {self.batch_size}"
            )

    @property
    def num_generations(self) -> int:
        return self.batch_size - self.num_contexts

    @property
    def max_past_kv_len(self) -> int:
        return max(self.past_kv_lens)

    @property
    def nnz_q(self) -> int:
        return self.num_generations


class MockSparseParams(SparseParams):
    """Sparse params stub used to exercise generic sparse attention plumbing."""

    algorithm: str = "mqa_gqa"

    @property
    def indices_block_size(self) -> int:
        return 1


@dataclass
class TestSparseAttentionMetadata(TrtllmAttentionMetadata):
    """Metadata for testing sparse attention."""

    num_sparse_topk: int = 64


class TestSparseAttention(TrtllmAttention):
    """TrtllmAttention subclass for testing with predetermined sparse indices."""

    def __init__(
        self,
        *args,
        sparse_kv_indices: Optional[torch.Tensor] = None,
        sparse_kv_offsets: Optional[torch.Tensor] = None,
        sparse_attn_indices: Optional[torch.Tensor] = None,
        sparse_attn_offsets: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        kwargs["sparse_params"] = MockSparseParams()
        kwargs["pos_embd_params"] = None
        super().__init__(*args, **kwargs)

        self._sparse_kv_indices = sparse_kv_indices
        self._sparse_kv_offsets = sparse_kv_offsets
        self._sparse_attn_indices = sparse_attn_indices
        self._sparse_attn_offsets = sparse_attn_offsets

    def sparse_kv_predict(self, q, k, metadata, forward_args: AttentionForwardArgs):
        return self._sparse_kv_indices, self._sparse_kv_offsets

    def sparse_attn_predict(self, q, k, metadata, forward_args: AttentionForwardArgs):
        return self._sparse_attn_indices, self._sparse_attn_offsets


def test_prepare_sparse_prediction_uses_optional_hooks() -> None:
    attention = TestSparseAttention.__new__(TestSparseAttention)
    attention.sparse_params = MockSparseParams()
    attention._sparse_kv_indices = torch.tensor([1], dtype=torch.int32)
    attention._sparse_kv_offsets = torch.tensor([0, 1], dtype=torch.int32)
    attention._sparse_attn_indices = torch.tensor([2], dtype=torch.int32)
    attention._sparse_attn_offsets = None
    forward_args = AttentionForwardArgs(
        sparse_prediction=SparsePrediction(sparse_mla_topk_lens=torch.tensor([3]))
    )

    prediction = prepare_sparse_prediction(attention, torch.empty(0), None, None, forward_args)

    assert prediction.sparse_kv_indices is attention._sparse_kv_indices
    assert prediction.sparse_kv_offsets is attention._sparse_kv_offsets
    assert prediction.sparse_attn_indices is attention._sparse_attn_indices
    assert prediction.sparse_attn_offsets is None
    assert prediction.sparse_attn_indices_block_size == 1
    assert prediction.sparse_mla_topk_lens is forward_args.sparse_prediction.sparse_mla_topk_lens


def test_mla_forward_uses_sparse_hook_facade() -> None:
    mla_module = inspect.getmodule(MLA)
    attention_module = inspect.getmodule(Attention)
    assert mla_module is not None
    assert attention_module is not None
    mla_module_source = inspect.getsource(mla_module)
    attention_module_source = inspect.getsource(attention_module)
    mla_module_ast = ast.parse(mla_module_source)
    hook_imports = [
        alias.name
        for node in ast.walk(mla_module_ast)
        if isinstance(node, ast.ImportFrom) and node.module == "attention_backend.sparse.hooks"
        for alias in node.names
    ]
    attention_init_source = inspect.getsource(Attention.__init__)
    attention_forward_source = inspect.getsource(Attention.forward)
    attention_forward_impl_source = inspect.getsource(Attention.forward_impl)
    init_source = inspect.getsource(MLA.__init__)
    create_weights_source = inspect.getsource(MLA.create_weights)
    create_outputs_source = inspect.getsource(MLA._create_outputs)
    forward_source = inspect.getsource(MLA.forward)
    forward_impl_source = inspect.getsource(MLA.forward_impl)
    forward_custom_op_source = inspect.getsource(MLA._forward_custom_op)
    project_output_source = inspect.getsource(MLA._project_output)
    transform_weights_source = inspect.getsource(MLA.transform_weights)

    assert hook_imports == ["get_sparse_attn_hooks"]
    for removed_helper in (
        "_initialize_dense_mla_modules",
        "_create_default_mla_weights",
        "_transform_default_mla_weights",
        "_create_sparse_mla_outputs",
        "_validate_sparse_mla_custom_op_outputs",
    ):
        assert removed_helper not in mla_module_source
    for source in (init_source, forward_source, forward_impl_source, project_output_source):
        assert "is_dsa" not in source
        assert "is_deepseek_v4" not in source
    assert "forward_sparse_attn" in forward_impl_source
    assert "attn_output[0]" in forward_impl_source
    assert "self.mqa.forward_mla" not in forward_impl_source
    assert "self.sparse_attn_hooks = get_sparse_attn_hooks(self)" in init_source
    assert "self.sparse_attn_hooks = get_sparse_attn_hooks(self)" in attention_init_source
    assert "sparse_attn_hooks is not None" not in mla_module_source
    assert "sparse_attn_hooks is not None" not in attention_module_source
    assert "initialize_sparse_attn" in attention_init_source
    assert "RocketKV" not in attention_init_source
    assert 'algorithm == "rocket"' not in attention_init_source
    assert attention_init_source.index("initialize_sparse_attn") < attention_init_source.index(
        "self.rotary_emb = None"
    )
    assert attention_init_source.index("initialize_sparse_attn") < attention_init_source.index(
        "self.attn = create_attention"
    )
    assert "forward_sparse_attn" in attention_forward_impl_source
    assert "project_sparse_attn_output" in attention_forward_source
    assert "get_sparse_attn_hooks" not in create_weights_source
    assert "get_sparse_attn_hooks" not in create_outputs_source
    assert "get_sparse_attn_hooks" not in forward_impl_source
    assert "get_sparse_attn_hooks" not in forward_custom_op_source
    assert "get_sparse_attn_hooks" not in project_output_source
    assert "sparse_epilogue_output" not in mla_module_source
    assert "sparse_output" not in mla_module_source
    assert "topk_indices" not in mla_module_source
    assert "sparse_backend_overrides" not in mla_module_source
    assert "self.indexer = getattr(self.mqa" not in mla_module_source
    assert "self.compressor = getattr(self.mqa" not in mla_module_source
    assert init_source.count("sparse_params=self.sparse_params") == 1
    assert init_source.count("sparse_params=None") == 1
    assert "return [self.create_output" in create_outputs_source
    assert "initialize_sparse_attn" in init_source
    assert init_source.index("self.kv_b_proj = Linear") < init_source.index(
        "initialize_sparse_attn"
    )
    assert init_source.index("self.mqa = create_attention") < init_source.index(
        "initialize_sparse_attn"
    )
    assert init_source.index("self.mha = create_attention") < init_source.index(
        "initialize_sparse_attn"
    )
    assert create_weights_source.index("create_sparse_attn_weights") < create_weights_source.index(
        "self.k_b_proj_trans ="
    )
    assert transform_weights_source.index(
        "transform_sparse_attn_weights"
    ) < transform_weights_source.index("self.resmooth_parameters")
    assert "project_sparse_attn_output" in project_output_source
    assert set(SPARSE_ATTN_HOOK_API) == {"SparseAttnHooks", "get_sparse_attn_hooks"}
    assert set(_SPARSE_ATTN_HOOK_MODULE_PATHS) == {"rocket", "dsa", "deepseek_v4"}
    assert _get_sparse_attn_hooks_for_algorithm("dsa").algorithm == "dsa"
    assert _get_sparse_attn_hooks_for_algorithm("deepseek_v4").algorithm == "deepseek_v4"
    rocket_hooks = _get_sparse_attn_hooks_for_algorithm("rocket")
    assert rocket_hooks.algorithm == "rocket"
    assert rocket_hooks.initialize_sparse_attn is not None
    rocket_hook_module = inspect.getmodule(rocket_hooks.initialize_sparse_attn)
    assert rocket_hook_module is not None
    assert "self.rope_fusion = False" in inspect.getsource(rocket_hook_module)
    for algorithm in ("dsa", "deepseek_v4"):
        hook_module = inspect.getmodule(
            _get_sparse_attn_hooks_for_algorithm(algorithm).initialize_sparse_attn
        )
        assert hook_module is not None
        hook_module_source = inspect.getsource(hook_module)
        assert "create_attention(" not in hook_module_source
        assert "self.indexer = getattr(self.mqa" in hook_module_source
        if algorithm == "deepseek_v4":
            assert "self.compressor = getattr(self.mqa" in hook_module_source
    assert not hasattr(AttentionBackend, "forward_mla_module")
    assert not hasattr(AttentionBackend, "forward_mla_custom_op")
    assert not hasattr(AttentionBackend, "project_mla_output")


def test_sparse_attn_hook_contract_allows_optional_hooks() -> None:
    dense_module = ModuleType("dense_attention_module")
    dense_module.sparse_params = None
    dense_hooks = get_sparse_attn_hooks(dense_module)

    assert isinstance(dense_hooks, SparseAttnHooks)
    assert not dense_hooks
    assert dense_hooks.algorithm is None
    assert dense_hooks.initialize_sparse_attn is None

    hook_module = ModuleType("test_sparse_attn_module")

    def initialize_sparse_attn(
        module,
        *,
        config,
        mapping,
        mapping_o,
        rms_norm_eps,
        quant_config,
        q_scaling,
        bias,
        dtype,
        reduce_output,
        aux_stream,
    ):
        return None

    def forward_sparse_attn(
        module,
        position_ids,
        hidden_states,
        attn_metadata,
        attn_output,
    ):
        return None

    hook_module.initialize_sparse_attn = initialize_sparse_attn
    hook_module.forward_sparse_attn = forward_sparse_attn

    hooks = SparseAttnHooks.from_module("test", hook_module)

    assert hooks
    assert hooks.initialize_sparse_attn is initialize_sparse_attn
    assert hooks.create_sparse_attn_weights is None
    assert hooks.transform_sparse_attn_weights is None
    assert hooks.prepare_sparse_attn_outputs is None
    assert hooks.forward_sparse_attn_custom_op is None
    assert hooks.project_sparse_attn_output is None
    assert hooks.require("forward_sparse_attn") is forward_sparse_attn

    missing_forward = ModuleType("missing_forward_sparse_attn")
    missing_forward.initialize_sparse_attn = initialize_sparse_attn
    hooks_without_forward = SparseAttnHooks.from_module("test", missing_forward)
    with pytest.raises(NotImplementedError, match="forward_sparse_attn"):
        hooks_without_forward.require("forward_sparse_attn")

    invalid_forward = ModuleType("invalid_forward_sparse_attn")
    invalid_forward.forward_sparse_attn = lambda module: None
    with pytest.raises(TypeError, match="expected"):
        SparseAttnHooks.from_module("test", invalid_forward)

    attention_hook_module = ModuleType("test_attention_sparse_attn_module")

    def forward_attention_sparse_attn(
        module,
        q,
        k,
        v,
        attn_metadata,
        attention_mask,
        attention_window_size,
        attention_mask_data,
        mrope_config,
        attention_sinks,
        relative_attention_bias,
        relative_attention_max_distance,
        has_lora,
    ):
        return None

    def project_attention_sparse_attn_output(
        module,
        attn_output,
        attn_metadata,
        all_reduce_params,
        lora_params,
    ):
        return attn_output

    attention_hook_module.forward_sparse_attn = forward_attention_sparse_attn
    attention_hook_module.project_sparse_attn_output = project_attention_sparse_attn_output
    attention_hooks = SparseAttnHooks.from_module("attention_test", attention_hook_module)

    assert attention_hooks.forward_sparse_attn is forward_attention_sparse_attn
    assert attention_hooks.project_sparse_attn_output is project_attention_sparse_attn_output


def test_prepare_sparse_prediction_allows_backend_without_prediction() -> None:
    attention = TrtllmAttention.__new__(TrtllmAttention)
    attention.sparse_params = MockSparseParams()

    prediction = prepare_sparse_prediction(
        attention, torch.empty(0), None, None, AttentionForwardArgs()
    )

    assert prediction == SparsePrediction()


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat kv heads to match query heads."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def create_kv_cache_manager(
    s: SparseScenario, kv_cache: Optional[torch.Tensor] = None
) -> KVCacheManager:
    """Create kv cache manager for testing."""
    kv_cache_config = KvCacheConfig(max_tokens=s.max_num_pages * s.page_size)
    mapping = Mapping(world_size=1, tp_size=1, rank=0)

    manager = KVCacheManager(
        kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=s.num_layers,
        num_kv_heads=s.num_kv_heads,
        head_dim=s.head_dim,
        tokens_per_block=s.page_size,
        max_seq_len=s.max_num_pages * s.page_size,
        max_batch_size=s.batch_size,
        mapping=mapping,
        dtype=str_dtype_to_binding(torch_dtype_to_str(s.kvcache_dtype)),
    )

    if kv_cache is not None:
        for i in range(s.num_layers):
            manager.get_buffers(i, kv_layout="HND").copy_(kv_cache[i])

    return manager


def generate_sparse_kv_indices(
    seq_lens: Tuple[int, ...],
    num_kv_heads: int,
    num_sparse_topk: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate sparse kv indices for context phase.

    For each request, pick min(num_sparse_topk, seq_len) indices from [0, seq_len).
    Returns (indices [num_kv_heads, total_sparse], offsets [num_requests + 1]).
    """
    all_indices = []
    offsets = [0]

    for seq_len in seq_lens:
        pick = min(num_sparse_topk, seq_len)
        batch_indices = []
        for _ in range(num_kv_heads):
            indices = torch.randperm(seq_len, device=device)[:pick].sort().values
            batch_indices.append(indices)
        all_indices.append(torch.stack(batch_indices, dim=0))
        offsets.append(offsets[-1] + pick)

    indices = torch.cat(all_indices, dim=1).int()
    offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
    return indices, offsets


def generate_sparse_attn_ctx_indices(
    seq_lens: Tuple[int, ...],
    num_kv_heads: int,
    num_sparse_topk: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate causal sparse attention indices for context phase.

    For each token at position pos (0-indexed within its request), available_kv_len = pos + 1.
    If available_kv_len <= num_sparse_topk: select all [0..pos], pad rest with -1.
    Otherwise: randomly pick num_sparse_topk from [0..pos].

    Returns: [num_kv_heads, total_tokens, num_sparse_topk] with -1 padding.
    """
    total_tokens = sum(seq_lens)
    result = torch.full(
        (num_kv_heads, total_tokens, num_sparse_topk), -1, dtype=torch.int32, device=device
    )

    token_offset = 0
    for seq_len in seq_lens:
        for token_idx in range(seq_len):
            available_kv_len = token_idx + 1
            pick = min(num_sparse_topk, available_kv_len)

            for head_idx in range(num_kv_heads):
                indices = torch.randperm(available_kv_len, device=device)[:pick].sort().values
                result[head_idx, token_offset + token_idx, :pick] = indices

        token_offset += seq_len

    return result


def generate_sparse_attn_gen_indices(
    past_kv_lens: Tuple[int, ...],
    num_kv_heads: int,
    num_sparse_topk: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate causal sparse attention indices for generation phase.

    Each generation token has past_kv_len available KV positions.
    Pick min(num_sparse_topk, past_kv_len) indices, pad rest with -1.

    Returns: [num_kv_heads, num_generations, num_sparse_topk] with -1 padding.
    """
    num_gens = len(past_kv_lens)
    result = torch.full(
        (num_kv_heads, num_gens, num_sparse_topk), -1, dtype=torch.int32, device=device
    )

    for gen_idx, past_kv_len in enumerate(past_kv_lens):
        pick = min(num_sparse_topk, past_kv_len)
        for head_idx in range(num_kv_heads):
            indices = torch.randperm(past_kv_len, device=device)[:pick].sort().values
            result[head_idx, gen_idx, :pick] = indices

    return result


def convert_sparse_indices_to_global(
    sparse_indices: torch.Tensor,
    metadata: TrtllmAttentionMetadata,
    layer_idx: int = 0,
    kv_factor: int = 2,
) -> torch.Tensor:
    """
    Convert local sparse indices to global KV cache pool indices.

    Works for both context (variable-length Q packed) and generation (one token per request).
    sparse_indices shape: [num_kv_heads, num_tokens, num_sparse_topk]
    """
    num_kv_heads, num_tokens, num_sparse_tokens = sparse_indices.shape
    device = sparse_indices.device

    tokens_per_block = metadata.kv_cache_manager.tokens_per_block
    num_layers = metadata.kv_cache_manager.num_layers
    stride_factor = num_layers * kv_factor * num_kv_heads * tokens_per_block

    # Build req_idx_per_token: map each token to its request index.
    num_requests = len(metadata.request_ids)
    seq_lens = metadata.seq_lens[:num_requests]
    if hasattr(seq_lens, "cpu"):
        seq_lens_cpu = seq_lens.cpu()
    else:
        seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int32)
    req_idx_per_token = torch.repeat_interleave(
        torch.arange(num_requests, dtype=torch.int32), seq_lens_cpu, dim=0
    ).to(device)

    # Build 2D block table: [num_requests, max_pages]
    request_ids = metadata.request_ids
    page_indices = metadata.kv_cache_manager.get_batch_cache_indices(request_ids)
    max_pages = max(len(p) for p in page_indices) if page_indices else 1
    host_block_table = torch.full((num_requests, max_pages), -1, dtype=torch.int32)
    for i, pages in enumerate(page_indices):
        if len(pages) > 0:
            host_block_table[i, : len(pages)] = torch.tensor(pages, dtype=torch.int32)
    block_table = host_block_table.to(device)

    # Convert to global
    global_indices = triton_convert_req_index_to_global_index(
        req_idx_per_token,
        block_table,
        sparse_indices,
        BLOCK_SIZE=tokens_per_block,
        NUM_TOPK_TOKENS=num_sparse_tokens,
        BLOCK_N=min(64, num_sparse_tokens),
        stride_factor=stride_factor,
        layer_id=layer_idx,
        num_kv_heads=num_kv_heads,
        kv_factor=kv_factor,
    )

    return global_indices


def _extract_batch_tensors(
    tensor: torch.Tensor, offset: int, length: int, shape_per_token: Tuple
) -> torch.Tensor:
    """Extract and reshape tensors for a specific batch."""
    return tensor[offset : offset + length].view(length, *shape_per_token)


def build_expected_sparse_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    sparse_kv_indices: torch.Tensor,
    sparse_kv_offsets: torch.Tensor,
    s: SparseContextScenario,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Build expected sparse K and V values based on sparse indices."""
    expected_kvs = []
    token_offset = 0

    for batch_idx, seq_len in enumerate(s.seq_lens):
        sparse_len = sparse_kv_offsets[batch_idx + 1].item() - sparse_kv_offsets[batch_idx].item()
        k_batch = _extract_batch_tensors(k, token_offset, seq_len, (s.num_kv_heads, s.head_dim))
        v_batch = _extract_batch_tensors(v, token_offset, seq_len, (s.num_kv_heads, s.head_dim))

        expected_k = torch.zeros(
            sparse_len, s.num_kv_heads, s.head_dim, device=k.device, dtype=k.dtype
        )
        expected_v = torch.zeros_like(expected_k)

        start, end = sparse_kv_offsets[batch_idx].item(), sparse_kv_offsets[batch_idx + 1].item()
        for head_idx in range(s.num_kv_heads):
            indices = sparse_kv_indices[head_idx, start:end]
            expected_k[:, head_idx] = k_batch[indices, head_idx]
            expected_v[:, head_idx] = v_batch[indices, head_idx]

        expected_kvs.append((expected_k, expected_v))
        token_offset += seq_len

    return expected_kvs


def _extract_tokens_from_cache(
    kv_buffer: torch.Tensor,
    block_ids: List[int],
    num_tokens: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract tokens from paged kv cache."""
    device = kv_buffer.device
    k_cache = torch.zeros(num_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.zeros_like(k_cache)

    for token_idx in range(num_tokens):
        block_idx = token_idx // page_size
        offset_in_block = token_idx % page_size
        block_id = block_ids[block_idx]

        for head_idx in range(num_kv_heads):
            k_cache[token_idx, head_idx] = kv_buffer[block_id, 0, head_idx, offset_in_block, :].to(
                dtype
            )
            v_cache[token_idx, head_idx] = kv_buffer[block_id, 1, head_idx, offset_in_block, :].to(
                dtype
            )

    return k_cache, v_cache


def extract_kv_from_paged_cache(
    kv_cache_manager: KVCacheManager,
    request_ids: List[int],
    sparse_kv_offsets: torch.Tensor,
    s: SparseContextScenario,
    dtype: torch.dtype,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Extract K and V values from paged kv cache."""
    kv_buffer = kv_cache_manager.get_buffers(0, kv_layout="HND")
    kv_caches = []

    for batch_idx in range(s.batch_size):
        num_sparse_tokens = (
            sparse_kv_offsets[batch_idx + 1].item() - sparse_kv_offsets[batch_idx].item()
        )
        block_ids = kv_cache_manager.get_block_ids_per_seq([request_ids[batch_idx]])[0]
        k_cache, v_cache = _extract_tokens_from_cache(
            kv_buffer, block_ids, num_sparse_tokens, s.num_kv_heads, s.head_dim, s.page_size, dtype
        )
        kv_caches.append((k_cache, v_cache))

    return kv_caches


def _compute_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_kv_groups: int,
) -> torch.Tensor:
    """Compute causal attention for a single batch."""
    seq_len = q.shape[2]
    head_dim = q.shape[3]

    k_expanded = repeat_kv(k, num_kv_groups)
    v_expanded = repeat_kv(v, num_kv_groups)

    attn_weights = torch.matmul(q, k_expanded.transpose(-1, -2)) / math.sqrt(head_dim)
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=q.device), diagonal=1
    )
    attn_weights = attn_weights + causal_mask
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        q.dtype
    )
    output = torch.matmul(attn_weights, v_expanded)

    return output


def reference_context_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: SparseContextScenario,
) -> torch.Tensor:
    """Reference implementation for context phase."""
    outputs = []
    token_offset = 0

    for seq_len in s.seq_lens:
        q_batch = _extract_batch_tensors(q, token_offset, seq_len, (s.num_heads, s.head_dim))
        k_batch = _extract_batch_tensors(k, token_offset, seq_len, (s.num_kv_heads, s.head_dim))
        v_batch = _extract_batch_tensors(v, token_offset, seq_len, (s.num_kv_heads, s.head_dim))

        q_batch = q_batch.view(1, seq_len, s.num_heads, s.head_dim).transpose(1, 2)
        k_batch = k_batch.view(1, seq_len, s.num_kv_heads, s.head_dim).transpose(1, 2)
        v_batch = v_batch.view(1, seq_len, s.num_kv_heads, s.head_dim).transpose(1, 2)

        output_batch = _compute_causal_attention(q_batch, k_batch, v_batch, s.num_kv_groups)
        output_batch = output_batch.transpose(1, 2).reshape(seq_len, s.num_heads * s.head_dim)
        outputs.append(output_batch)
        token_offset += seq_len

    return torch.cat(outputs, dim=0)


def reference_context_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sparse_attn_ctx_indices: torch.Tensor,
    s: SparseContextScenario,
) -> torch.Tensor:
    """
    Reference implementation for context phase with sparse attention.
    Uses mask-based approach for each KV head.
    """
    total_tokens = sum(s.seq_lens)
    device = q.device
    dtype = q.dtype

    # Reshape inputs: [num_tokens, num_heads, head_dim]
    q_reshaped = q.view(total_tokens, s.num_heads, s.head_dim)
    k_reshaped = k.view(total_tokens, s.num_kv_heads, s.head_dim)
    v_reshaped = v.view(total_tokens, s.num_kv_heads, s.head_dim)

    outputs = []
    token_offset = 0

    for seq_len in s.seq_lens:
        q_batch = q_reshaped[
            token_offset : token_offset + seq_len
        ]  # [seq_len, num_heads, head_dim]
        k_batch = k_reshaped[
            token_offset : token_offset + seq_len
        ]  # [seq_len, num_kv_heads, head_dim]
        v_batch = v_reshaped[
            token_offset : token_offset + seq_len
        ]  # [seq_len, num_kv_heads, head_dim]

        batch_output = []

        # Process each KV head
        for kv_head_idx in range(s.num_kv_heads):
            k_head = k_batch[:, kv_head_idx, :]
            v_head = v_batch[:, kv_head_idx, :]

            # Build sparse mask for this head
            sparse_mask = torch.full(
                (seq_len, seq_len), float("-inf"), device=device, dtype=torch.float32
            )

            for token_idx in range(seq_len):
                global_token_idx = token_offset + token_idx
                # Get sparse indices for this token: [num_sparse_tokens]
                indices = sparse_attn_ctx_indices[kv_head_idx, global_token_idx]
                # Filter out -1 padding
                valid_indices = indices[indices >= 0]
                # Set mask values to 0 for valid positions
                sparse_mask[token_idx, valid_indices] = 0.0

            # Apply causal mask on top of sparse mask
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=torch.float32),
                diagonal=1,
            )
            combined_mask = sparse_mask + causal_mask

            # Process each query head in this KV group
            for group_idx in range(s.num_kv_groups):
                q_head_idx = kv_head_idx * s.num_kv_groups + group_idx
                q_head = q_batch[:, q_head_idx, :]  # [seq_len, head_dim]

                attn_scores = torch.matmul(q_head, k_head.T) / math.sqrt(s.head_dim)
                attn_scores = attn_scores + combined_mask
                attn_weights = torch.nn.functional.softmax(
                    attn_scores, dim=-1, dtype=torch.float32
                ).to(dtype)

                out_head = torch.matmul(attn_weights, v_head)
                batch_output.append(out_head)

        # Concatenate all heads: [seq_len, num_heads, head_dim] -> [seq_len, num_heads * head_dim]
        batch_output = torch.stack(batch_output, dim=1)
        batch_output = batch_output.reshape(seq_len, s.num_heads * s.head_dim)
        outputs.append(batch_output)

        token_offset += seq_len

    return torch.cat(outputs, dim=0)


def _get_selected_pages_tokens(
    token_indices: torch.Tensor,
    page_size: int,
    kv_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert token indices to page indices and gather all tokens from selected pages."""
    if len(token_indices) == 0:
        return torch.tensor([], dtype=torch.long, device=device)

    page_indices = torch.unique((token_indices // page_size).sort().values)
    selected_tokens = []

    for page_idx in page_indices:
        token_start = page_idx * page_size
        token_end = min(token_start + page_size, kv_len)
        selected_tokens.append(torch.arange(token_start, token_end, device=device))

    return (
        torch.cat(selected_tokens)
        if selected_tokens
        else torch.tensor([], dtype=torch.long, device=device)
    )


def _compute_sparse_attention_per_head(
    q_head: torch.Tensor,
    k_sparse: torch.Tensor,
    v_sparse: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Compute attention for a single query head."""
    if len(k_sparse) == 0:
        return torch.zeros(head_dim, device=q_head.device, dtype=q_head.dtype)

    attn_weights = torch.matmul(q_head, k_sparse.T) / math.sqrt(head_dim)
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        q_head.dtype
    )
    return torch.matmul(attn_weights, v_sparse)


def reference_generation_sparse_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    sparse_attn_indices: torch.Tensor,
    s: SparseGenerationScenario,
) -> torch.Tensor:
    """Reference implementation for generation phase with sparse attention.

    Args:
        sparse_attn_indices: [num_kv_heads, num_gens, num_sparse_topk] with -1 padding.
    """
    outputs = []

    for gen_idx in range(s.num_generations):
        batch_idx = s.num_contexts + gen_idx
        past_kv_len = s.past_kv_lens[batch_idx]
        kv_len = past_kv_len + 1

        k_full = k_cache[batch_idx, :kv_len].clone()
        v_full = v_cache[batch_idx, :kv_len].clone()
        k_full[past_kv_len] = k_new[gen_idx].view(s.num_kv_heads, s.head_dim)
        v_full[past_kv_len] = v_new[gen_idx].view(s.num_kv_heads, s.head_dim)

        q_batch = q[gen_idx].view(s.num_heads, s.head_dim)
        head_outputs = []

        for kv_head_idx in range(s.num_kv_heads):
            token_indices = sparse_attn_indices[kv_head_idx, gen_idx]
            valid_indices = token_indices[token_indices >= 0].long()

            if len(valid_indices) == 0:
                head_outputs.extend(
                    [torch.zeros(s.head_dim, device=q.device, dtype=q.dtype)] * s.num_kv_groups
                )
                continue

            k_sparse = k_full[valid_indices, kv_head_idx, :]
            v_sparse = v_full[valid_indices, kv_head_idx, :]

            for group_idx in range(s.num_kv_groups):
                q_head_idx = kv_head_idx * s.num_kv_groups + group_idx
                out_head = _compute_sparse_attention_per_head(
                    q_batch[q_head_idx], k_sparse, v_sparse, s.head_dim
                )
                head_outputs.append(out_head)

        outputs.append(torch.cat(head_outputs, dim=0))

    return torch.stack(outputs, dim=0)


def _setup_context_test(s: SparseContextScenario):
    """Setup common components for context test."""
    device = torch.device("cuda")
    torch.manual_seed(42)
    num_sparse_topk = s.num_sparse_topk

    q = torch.randn(s.nnz_q, s.num_heads * s.head_dim, device=device, dtype=s.dtype)
    k = torch.randn(s.nnz_q, s.num_kv_heads * s.head_dim, device=device, dtype=s.dtype)
    v = torch.randn(s.nnz_q, s.num_kv_heads * s.head_dim, device=device, dtype=s.dtype)
    sparse_kv_indices, sparse_kv_offsets = generate_sparse_kv_indices(
        s.seq_lens, s.num_kv_heads, num_sparse_topk, device
    )

    kv_cache = torch.zeros(
        s.num_layers,
        s.max_num_pages,
        2,
        s.num_kv_heads,
        s.page_size,
        s.head_dim,
        device=device,
        dtype=s.kvcache_dtype,
    )
    kv_cache_manager = create_kv_cache_manager(s, kv_cache)

    request_ids = list(range(s.batch_size))
    kv_cache_manager.add_dummy_requests(request_ids, list(s.seq_lens))

    metadata = TestSparseAttentionMetadata(
        num_contexts=s.batch_size,
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0] * s.batch_size),
        seq_lens=torch.tensor(s.seq_lens, dtype=torch.int32),
        max_num_requests=s.batch_size,
        max_num_tokens=s.nnz_q,
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
        prompt_lens=list(s.seq_lens),
        num_sparse_topk=num_sparse_topk,
    )
    metadata.prepare()

    attention = TestSparseAttention(
        layer_idx=0,
        num_heads=s.num_heads,
        head_dim=s.head_dim,
        num_kv_heads=s.num_kv_heads,
        sparse_kv_indices=sparse_kv_indices,
        sparse_kv_offsets=sparse_kv_offsets,
    )

    return (
        device,
        q,
        k,
        v,
        sparse_kv_indices,
        sparse_kv_offsets,
        kv_cache_manager,
        request_ids,
        metadata,
        attention,
    )


def _setup_generation_test(s: SparseGenerationScenario):
    """Setup common components for generation test."""
    device = torch.device("cuda")
    torch.manual_seed(42)
    num_sparse_topk = s.num_sparse_topk

    token_nums = [past_len + 1 for past_len in s.past_kv_lens]

    q = torch.randn(s.num_generations, s.num_heads * s.head_dim, device=device, dtype=s.dtype)
    k_new = torch.randn(
        s.num_generations, s.num_kv_heads * s.head_dim, device=device, dtype=s.dtype
    )
    v_new = torch.randn(
        s.num_generations, s.num_kv_heads * s.head_dim, device=device, dtype=s.dtype
    )

    gen_past_kv_lens = tuple(s.past_kv_lens[s.num_contexts + i] for i in range(s.num_generations))
    # Local sparse indices: [num_kv_heads, num_gens, num_sparse_topk]
    sparse_attn_indices = generate_sparse_attn_gen_indices(
        gen_past_kv_lens, s.num_kv_heads, num_sparse_topk, device
    )

    kv_cache = torch.randn(
        s.num_layers,
        s.max_num_pages,
        2,
        s.num_kv_heads,
        s.page_size,
        s.head_dim,
        device=device,
        dtype=s.kvcache_dtype,
    )
    kv_cache_manager = create_kv_cache_manager(s, kv_cache)

    request_ids = list(range(s.batch_size))
    kv_cache_manager.add_dummy_requests(request_ids, token_nums)

    metadata = TestSparseAttentionMetadata(
        num_contexts=s.num_contexts,
        kv_cache_params=KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=list(s.past_kv_lens)
        ),
        seq_lens=torch.tensor([1] * s.num_generations).int(),
        max_num_requests=s.batch_size,
        max_num_tokens=s.num_generations,
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
        prompt_lens=list(s.past_kv_lens),
        num_sparse_topk=num_sparse_topk,
    )
    metadata.prepare()

    # Convert local indices to global KV cache pool indices
    global_sparse_attn_indices = convert_sparse_indices_to_global(
        sparse_attn_indices, metadata, layer_idx=0
    )

    attention = TestSparseAttention(
        layer_idx=0,
        num_heads=s.num_heads,
        head_dim=s.head_dim,
        num_kv_heads=s.num_kv_heads,
        sparse_attn_indices=global_sparse_attn_indices,
    )

    return (
        device,
        q,
        k_new,
        v_new,
        sparse_attn_indices,
        kv_cache_manager,
        request_ids,
        metadata,
        attention,
    )


def _build_reference_kv_cache(
    kv_cache_manager, request_ids, s: SparseGenerationScenario, device, dtype
):
    """Build reference K, V cache from paged format."""
    k_cache_ref = torch.zeros(
        s.batch_size, s.kv_cache_len, s.num_kv_heads, s.head_dim, device=device, dtype=dtype
    )
    v_cache_ref = torch.zeros_like(k_cache_ref)

    kv_buffer = kv_cache_manager.get_buffers(0, kv_layout="HND")
    for batch_idx, past_kv_len in enumerate(s.past_kv_lens):
        block_ids = kv_cache_manager.get_block_ids_per_seq([request_ids[batch_idx]])[0]
        for block_local_idx, block_id in enumerate(block_ids):
            token_start = block_local_idx * s.page_size
            token_end = min(token_start + s.page_size, past_kv_len)
            tokens_in_block = token_end - token_start

            for head_idx in range(s.num_kv_heads):
                k_cache_ref[batch_idx, token_start:token_end, head_idx] = kv_buffer[
                    block_id, 0, head_idx, :tokens_in_block, :
                ].to(dtype)
                v_cache_ref[batch_idx, token_start:token_end, head_idx] = kv_buffer[
                    block_id, 1, head_idx, :tokens_in_block, :
                ].to(dtype)

    return k_cache_ref, v_cache_ref


@pytest.mark.skipif(getSMVersion() < 100, reason="Sparse MQA/GQA requires SM100 (Blackwell)")
@pytest.mark.parametrize(
    "s",
    [
        SparseContextScenario(batch_size=2, seq_lens=(48, 64), num_pages=8),
        SparseContextScenario(batch_size=4, seq_lens=(96, 112, 128, 144), num_pages=16),
        SparseContextScenario(batch_size=1, seq_lens=(256,), num_pages=8),
        SparseContextScenario(batch_size=3, seq_lens=(64, 96, 128), num_pages=12),
    ],
    ids=["batch2_var_seq", "batch4_var_seq", "batch1_seq256", "batch3_var_seq"],
)
def test_context_sparse_kv(s: SparseContextScenario):
    """Test context phase with sparse kv cache write."""
    (
        device,
        q,
        k,
        v,
        sparse_kv_indices,
        sparse_kv_offsets,
        kv_cache_manager,
        request_ids,
        metadata,
        attention,
    ) = _setup_context_test(s)

    ref_output = reference_context_attention(q.clone(), k.clone(), v.clone(), s)
    expected_kvs = build_expected_sparse_kv(
        k.clone(), v.clone(), sparse_kv_indices, sparse_kv_offsets, s
    )

    qkv = torch.cat([q, k, v], dim=1)
    output = attention.forward(qkv, None, None, metadata)

    assert output.shape == ref_output.shape, f"Shape mismatch: {output.shape} vs {ref_output.shape}"
    torch.testing.assert_close(output, ref_output, atol=ATOL, rtol=RTOL)
    print(f"Context sparse kv attention output test passed: {s}")

    actual_kvs = extract_kv_from_paged_cache(
        kv_cache_manager, request_ids, sparse_kv_offsets, s, s.dtype
    )

    for batch_idx in range(s.batch_size):
        actual_k, actual_v = actual_kvs[batch_idx]
        expected_k, expected_v = expected_kvs[batch_idx]
        torch.testing.assert_close(
            actual_k,
            expected_k,
            atol=ATOL,
            rtol=RTOL,
            msg=f"K cache mismatch for batch {batch_idx} after sparse compaction",
        )
        torch.testing.assert_close(
            actual_v,
            expected_v,
            atol=ATOL,
            rtol=RTOL,
            msg=f"V cache mismatch for batch {batch_idx} after sparse compaction",
        )

    print(f"Context sparse kv cache content test passed: {s}")
    kv_cache_manager.shutdown()


@pytest.mark.skipif(getSMVersion() < 100, reason="Sparse MQA/GQA requires SM100 (Blackwell)")
@pytest.mark.parametrize(
    "s",
    [
        # Basic scenarios
        SparseGenerationScenario(
            batch_size=2,
            past_kv_lens=(96, 128),
            num_pages=16,
        ),
        SparseGenerationScenario(
            batch_size=4,
            past_kv_lens=(192, 224, 256, 288),
            num_pages=32,
        ),
        SparseGenerationScenario(batch_size=1, past_kv_lens=(64,), num_pages=8),
        SparseGenerationScenario(
            batch_size=3,
            past_kv_lens=(128, 160, 192),
            num_pages=24,
        ),
        # GQA ratios: MQA (8Q/1KV), GQA 4:1, GQA 2:1
        SparseGenerationScenario(
            num_heads=8,
            num_kv_heads=1,
            batch_size=2,
            past_kv_lens=(96, 128),
            num_pages=16,
        ),
        SparseGenerationScenario(
            num_heads=16,
            num_kv_heads=4,
            batch_size=2,
            past_kv_lens=(128, 256),
            num_pages=16,
        ),
        # topk: minimum (4), topk exceeding some past_kv_lens
        SparseGenerationScenario(
            batch_size=1,
            past_kv_lens=(128,),
            num_pages=8,
            num_sparse_topk=4,
        ),
        SparseGenerationScenario(
            batch_size=2,
            past_kv_lens=(32, 256),
            num_pages=16,
            num_sparse_topk=128,
        ),
        # Large batch
        SparseGenerationScenario(
            batch_size=8,
            past_kv_lens=(64, 96, 128, 160, 192, 224, 256, 288),
            num_pages=64,
        ),
        # Page boundary: page_size=64
        SparseGenerationScenario(
            page_size=64,
            batch_size=2,
            past_kv_lens=(64, 192),
            num_pages=8,
        ),
    ],
    ids=[
        "batch2_var_kv",
        "batch4_var_kv",
        "batch1_kv64",
        "batch3_var_kv",
        "mqa_8q1kv",
        "gqa_16q4kv",
        "topk4_min",
        "topk128_exceeds_some",
        "batch8_varied",
        "page_size_64",
    ],
)
def test_generation_sparse_attention(s: SparseGenerationScenario):
    """Test generation phase with sparse attention computation."""
    (
        device,
        q,
        k_new,
        v_new,
        sparse_attn_indices,
        kv_cache_manager,
        request_ids,
        metadata,
        attention,
    ) = _setup_generation_test(s)

    k_cache_ref, v_cache_ref = _build_reference_kv_cache(
        kv_cache_manager, request_ids, s, device, s.dtype
    )
    ref_sparse_output = reference_generation_sparse_attention(
        q, k_cache_ref, v_cache_ref, k_new, v_new, sparse_attn_indices, s
    )

    qkv = torch.cat([q, k_new, v_new], dim=1)
    output = attention.forward(qkv, None, None, metadata)

    expected_shape = (s.num_generations, s.num_heads * s.head_dim)
    assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
    assert torch.isfinite(output).all(), "Output contains non-finite values"

    torch.testing.assert_close(output, ref_sparse_output, atol=ATOL, rtol=RTOL)
    print(f"Generation sparse attention test passed: {s}")
    kv_cache_manager.shutdown()


@pytest.mark.skipif(getSMVersion() < 100, reason="Sparse MQA/GQA requires SM100 (Blackwell)")
@pytest.mark.parametrize(
    "s",
    [
        # MQA (8Q/1KV)
        SparseContextScenario(
            batch_size=2,
            seq_lens=(128, 64),
            num_pages=8,
            num_kv_heads=1,
            num_heads=8,
        ),
        # GQA 4:1 (8Q/2KV)
        SparseContextScenario(
            batch_size=2,
            seq_lens=(128, 64),
            num_pages=8,
            num_kv_heads=2,
            num_heads=8,
        ),
        # GQA 4:1 (16Q/4KV) with 3 batches
        SparseContextScenario(
            batch_size=3,
            seq_lens=(64, 96, 128),
            num_pages=12,
            num_kv_heads=4,
            num_heads=16,
        ),
        # GQA 8:1 (32Q/4KV)
        SparseContextScenario(
            batch_size=2,
            seq_lens=(64, 128),
            num_pages=8,
            num_kv_heads=4,
            num_heads=32,
        ),
        # topk=4 (very sparse)
        SparseContextScenario(
            batch_size=2,
            seq_lens=(64, 128),
            num_pages=8,
            num_kv_heads=1,
            num_heads=8,
            num_sparse_topk=4,
        ),
        # topk=128 (near-dense, topk >= seq_len for some requests)
        SparseContextScenario(
            batch_size=2,
            seq_lens=(64, 128),
            num_pages=8,
            num_kv_heads=2,
            num_heads=8,
            num_sparse_topk=128,
        ),
    ],
    ids=[
        "mqa_8q1kv",
        "gqa_8q2kv",
        "gqa_16q4kv_batch3",
        "gqa_32q4kv",
        "topk4_very_sparse",
        "topk128_near_dense",
    ],
)
def test_context_sparse_attention_mqa(s: SparseContextScenario):
    """Test context phase with sparse attention using sparse_attn_ctx_indices."""
    (
        device,
        q,
        k,
        v,
        sparse_kv_indices,
        sparse_kv_offsets,
        kv_cache_manager,
        request_ids,
        metadata,
        _,
    ) = _setup_context_test(s)

    num_sparse_topk = metadata.num_sparse_topk

    # Generate causal sparse attention indices, padded to num_sparse_topk
    sparse_attn_ctx_indices = generate_sparse_attn_ctx_indices(
        s.seq_lens, s.num_kv_heads, num_sparse_topk, device
    )
    assert sparse_attn_ctx_indices.shape[-1] == num_sparse_topk

    # Convert to global indices for attentionOp
    global_sparse_attn_ctx_indices = convert_sparse_indices_to_global(
        sparse_attn_ctx_indices, metadata, layer_idx=0
    )

    # Compute reference output using local indices
    ref_output = reference_context_sparse_attention(
        q.clone(), k.clone(), v.clone(), sparse_attn_ctx_indices, s
    )

    # Verify reference output shape
    total_tokens = sum(s.seq_lens)
    expected_shape = (total_tokens, s.num_heads * s.head_dim)
    assert ref_output.shape == expected_shape, (
        f"Reference output shape mismatch: {ref_output.shape} vs {expected_shape}"
    )
    assert torch.isfinite(ref_output).all(), "Reference output contains non-finite values"

    print(f"Context sparse attention MQA reference test passed: {s}")

    attention = TestSparseAttention(
        layer_idx=0,
        num_heads=s.num_heads,
        head_dim=s.head_dim,
        num_kv_heads=s.num_kv_heads,
        sparse_kv_indices=sparse_kv_indices,
        sparse_kv_offsets=sparse_kv_offsets,
        sparse_attn_indices=global_sparse_attn_ctx_indices,
    )

    qkv = torch.cat([q, k, v], dim=1)
    output = attention.forward(qkv, None, None, metadata)
    torch.testing.assert_close(output, ref_output, atol=ATOL, rtol=RTOL)
    print(f"Context sparse attention MQA forward test passed: {s}")

    kv_cache_manager.shutdown()


if __name__ == "__main__":
    s = SparseContextScenario(
        batch_size=2,
        seq_lens=(128, 64),
        num_pages=8,
        num_kv_heads=1,
        num_heads=8,
        head_dim=128,
    )
    test_context_sparse_attention_mqa(s)

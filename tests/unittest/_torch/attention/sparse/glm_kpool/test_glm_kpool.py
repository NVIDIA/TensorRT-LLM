# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests for GLM sparse attention metadata and cache layouts."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm._torch.attention.backends.sparse.glm_kpool import (
    Glm5NextMamba2Metadata,
    GlmKpoolSparseAttention,
    GlmKpoolSparseParams,
    latent_pool_rows,
    paged_slot_indices,
)
from tensorrt_llm._torch.attention.backends.sparse.glm_kpool.params import (
    GlmKpoolBackendForwardArgs,
)

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("fp8", [False, True], ids=["bf16-kv", "fp8-kv"])
@pytest.mark.parametrize(
    "snapshot_options,retains_page",
    [
        ({}, False),
        ({"periodic_snapshot_interval": 32}, True),
        ({"periodic_snapshot_interval": 48}, True),
        ({"periodic_snapshot_interval": 1024}, False),
        ({"additional_snapshot_offsets_from_start": [512]}, True),
        ({"additional_snapshot_offsets_from_start": [513]}, False),
        ({"additional_snapshot_offsets_from_end": [511]}, True),
        ({"additional_snapshot_offsets_from_end": [512]}, False),
    ],
    ids=[
        "no-snapshots",
        "aligned-periodic",
        "unaligned-periodic",
        "unreachable-periodic",
        "start-boundary",
        "unreachable-start",
        "from-end",
        "unreachable-end",
    ],
)
@pytest.mark.parametrize(
    "layout,expected_sparse_layers",
    [("tp", 1), ("adp", 1), ("mtp", 2), ("pp-linear", 0), ("pp-mtp", 2), ("draft-only", 1)],
)
def test_indexer_static_cache_cost(
    fp8, snapshot_options, retains_page, layout, expected_sparse_layers
):
    from tensorrt_llm._torch.attention.backends.sparse.glm_kpool.cache_manager import (
        Glm5NextCacheManager,
    )
    from tensorrt_llm._torch.configs.glm5_next import Glm5NextTextConfig
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
        MambaHybridCacheManagerV2,
    )
    from tensorrt_llm.llmapi import KvCacheConfig, MambaStateConfig, MTPDecodingConfig
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig

    config = Glm5NextTextConfig(
        num_hidden_layers=2,
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        kv_lora_rank=512,
        index_head_dim=128,
        dtype="bfloat16",
        linear_attn_config={"num_heads": 4, "head_dim": 8, "short_conv_kernel_size": 4},
    )
    model_config = ModelConfig(
        pretrained_config=config,
        quant_config=QuantConfig(kv_cache_quant_algo="FP8" if fp8 else None),
    )
    pp_size = 2 if layout.startswith("pp-") else 1
    mapping = Mapping(
        world_size=4 * pp_size,
        tp_size=4,
        pp_size=pp_size,
        pp_partition=[1, 1] if pp_size == 2 else None,
        rank=4 if layout == "pp-mtp" else 0,
        enable_attention_dp=layout == "adp",
    )
    kwargs = dict(
        max_batch_size=4,
        tokens_per_block=32,
        max_seq_len=512,
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=bool(snapshot_options),
            mamba_state_config=MambaStateConfig(**snapshot_options),
        ),
        spec_config=MTPDecodingConfig(max_draft_len=3)
        if layout in ("mtp", "pp-mtp", "draft-only")
        else None,
        is_draft=layout == "draft-only",
    )
    base_slope, base_fixed = MambaHybridCacheManagerV2.get_cache_size_per_token(
        model_config, mapping, **kwargs
    )
    slope, fixed = Glm5NextCacheManager.get_cache_size_per_token(model_config, mapping, **kwargs)
    # The three 128-wide BF16 indexer sections are replicated on every TP/ADP rank.
    extra_per_token = expected_sparse_layers * 3 * 128 * 2
    assert slope - base_slope == extra_per_token
    retained_pages = 4 * pp_size if retains_page else 0
    assert fixed - base_fixed == retained_pages * 32 * extra_per_token
    # An equivalent number of latent bytes must get the same token and snapshot
    # budget from the base estimator, independently of where the bytes are stored.
    reference_config = deepcopy(model_config)
    reference_config.pretrained_config.kv_lora_rank += 768 if fp8 else 384
    assert (slope, fixed) == MambaHybridCacheManagerV2.get_cache_size_per_token(
        reference_config, mapping, **kwargs
    )


@pytest.mark.parametrize("fp8", [False, True], ids=["bf16-kv", "fp8-kv"])
@pytest.mark.parametrize("snapshot_interval", [0, 32])
def test_indexer_runtime_cache_cost_matches_registered_buffers(fp8, snapshot_interval):
    from tensorrt_llm._torch.attention.backends.sparse.glm_kpool.cache_manager import (
        Glm5NextCacheManager,
    )
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
    from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
    from tensorrt_llm.llmapi import KvCacheConfig, MambaStateConfig
    from tensorrt_llm.mapping import Mapping

    manager = object.__new__(Glm5NextCacheManager)
    manager.mapping = Mapping(world_size=1, tp_size=1)
    manager.pp_layers = [0, 1]
    manager.layer_offsets = {0: 0, 1: 1}
    manager.sparse_layer_ids = [1, 3]  # Layer 3 is on another PP rank.
    manager.index_state_dim = 384
    manager.num_local_layers = 2
    manager.local_num_mamba_layers = 1
    manager.num_kv_heads_per_layer = [0, 1]
    manager.head_dim_per_layer = [512, 512]
    manager.kv_cache_type = CacheTypeCpp.SELFKONLY
    manager.kv_factor = 1
    manager.dtype = DataType.FP8 if fp8 else DataType.BF16
    manager.tokens_per_block = 32
    manager.max_batch_size = 4
    manager.max_num_tokens = 16
    manager.max_seq_len = 512
    manager.max_attention_window_vec = [None, None]
    manager.enable_swa_scratch_reuse = False
    manager._generation_kv_capacity_headroom = 0
    manager._has_cp_helix = False
    manager._num_reserved_dummy_slots = 1
    manager._ple_layer_ids = []
    manager.ssm_bytes, manager.conv_bytes = 256, 144
    manager.kv_cache_config = KvCacheConfig(
        enable_block_reuse=bool(snapshot_interval),
        mamba_state_config=MambaStateConfig(periodic_snapshot_interval=snapshot_interval),
    )

    buffers = manager._extra_buffers_per_layer(tokens_per_block=32)
    assert set(buffers) == {1}
    assert buffers[1][0].role == Role.INDEX_KEY
    assert buffers[1][0].size == 768 * 32
    latent_bytes = 512 if fp8 else 1024
    assert manager.get_layer_bytes_per_token(0, Role.ALL) == 0
    assert manager.get_layer_bytes_per_token(0, Role.INDEX_KEY) == 0
    assert manager.get_layer_bytes_per_token(1, Role.KEY) == latent_bytes
    assert manager.get_layer_bytes_per_token(1, Role.INDEX_KEY) == 768
    assert manager.get_layer_bytes_per_token(1, Role.ALL) == latent_bytes + 768
    assert manager._attention_cache_bytes_per_token() == latent_bytes + 768
    snapshot_bytes_per_token = 400 // 32 if snapshot_interval else 0
    assert manager.get_cache_bytes_per_token() == latent_bytes + 768 + snapshot_bytes_per_token

    tokens = 128
    state_slots = 5 + (tokens // 32 if snapshot_interval else 0)
    retained_page_tokens = 4 * 32 if snapshot_interval else 0
    expected_quota = (tokens + retained_page_tokens) * (latent_bytes + 768) + state_slots * 400
    assert manager._get_quota_from_max_tokens(tokens) == expected_quota
    assert manager._get_max_tokens_from_quota(expected_quota) == tokens


def test_nope_mla_geometry_uses_shared_factory(monkeypatch):
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention
    from tensorrt_llm._torch.attention.backends.utils import create_attention

    def initialize_base(self, layer_idx, num_heads, head_dim, num_kv_heads, **kwargs):
        self.mla_params = kwargs["mla_params"]
        self.num_kv_heads = num_kv_heads
        for name in ("q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "v_head_dim"):
            setattr(self, name, getattr(self.mla_params, name))

    monkeypatch.setattr(TrtllmAttention, "__init__", initialize_base)
    kwargs = dict(
        backend_name="TRTLLM",
        layer_idx=0,
        num_heads=16,
        head_dim=512,
        num_kv_heads=1,
        is_mla_enable=True,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        qk_nope_head_dim=256,
        v_head_dim=256,
        rope_append=False,
        sparse_params=GlmKpoolSparseParams(),
    )
    backend = create_attention(**kwargs)
    assert isinstance(backend, GlmKpoolSparseAttention)
    assert backend.mla_params.qk_rope_head_dim == 0
    assert backend.mla_params.rope_append is False
    assert (backend.q_lora_rank, backend.kv_lora_rank, backend.v_head_dim) == (1536, 512, 256)
    assert backend.softmax_scale == 256**-0.5
    with pytest.raises(ValueError, match="must equal kv_lora_rank"):
        create_attention(**{**kwargs, "head_dim": 256})
    with pytest.raises(ValueError, match="fully NoPE"):
        create_attention(**{**kwargs, "qk_rope_head_dim": 64})
    with pytest.raises(AssertionError):
        create_attention(**{**kwargs, "qk_rope_head_dim": -1})


@pytest.fixture
def forward_case():
    backend = object.__new__(GlmKpoolSparseAttention)
    backend.num_heads = 2
    backend.head_dim = backend.kv_lora_rank = 4
    state = SimpleNamespace(
        latent_pool=torch.arange(64, dtype=torch.bfloat16).view(4, 4, 4),
        block_tables=torch.tensor([[3, 0], [2, 1], [0, 2]]),
        num_contexts=1,
        tokens_per_block=4,
    )
    backend._cache_state = Mock(return_value=state)
    backend._dispatch_sparse_core = Mock()
    return SimpleNamespace(
        backend=backend,
        state=state,
        q=torch.ones(2, 8, dtype=torch.bfloat16),
        indices=torch.tensor([[0, -1], [1, 2]], dtype=torch.int32),
        metadata=object(),
    )


@pytest.mark.parametrize("route", ["context", "generation", "verify"])
@pytest.mark.parametrize("supply_output", [False, True])
def test_forward_dispatches_latent_rows_and_preserves_output(forward_case, route, supply_output):
    case = forward_case
    context = route == "context"
    q = case.q.repeat_interleave(2, dim=0) if route == "verify" else case.q
    indices = case.indices.repeat_interleave(2, dim=0) if route == "verify" else case.indices
    core_output = torch.arange(q.shape[0] * 8, dtype=q.dtype).view(q.shape[0], 2, 4)
    case.backend._dispatch_sparse_core.return_value = core_output
    selection = GlmKpoolBackendForwardArgs(topk_rows=indices)
    output = torch.empty(q.shape[0], 8, dtype=q.dtype) if supply_output else None
    args = AttentionForwardArgs(
        attention_input_type=(
            AttentionInputType.context_only if context else AttentionInputType.generation_only
        ),
        sparse_backend_args=selection,
        output=output,
    )
    actual = case.backend.forward(q, None, None, case.metadata, args)
    torch.testing.assert_close(actual, core_output.flatten(1))
    if supply_output:
        assert actual is output
    case.backend._cache_state.assert_called_once_with(case.metadata)
    case.backend._dispatch_sparse_core.assert_called_once()
    dispatched_q, dispatched_k, dispatched_indices = (
        case.backend._dispatch_sparse_core.call_args.args
    )
    torch.testing.assert_close(dispatched_q, q.view(-1, 2, 4))
    torch.testing.assert_close(dispatched_k, case.state.latent_pool.view(-1, 1, 4))
    torch.testing.assert_close(dispatched_indices, indices)


@pytest.mark.parametrize(
    "invalid",
    [
        "selection",
        "selection_type",
        "missing_rows",
        "legacy_indices",
        "v",
        "out_scale",
        "out_scale_sf",
        "output_sf",
        "shape",
        "dtype",
        "device",
    ],
)
def test_forward_rejects_invalid_arguments_before_cache_access(forward_case, invalid):
    case = forward_case
    args = AttentionForwardArgs(
        attention_input_type=AttentionInputType.context_only,
        sparse_backend_args=GlmKpoolBackendForwardArgs(topk_rows=case.indices),
    )
    v = None
    error, message = ValueError, "quantized attention output"
    if invalid == "selection":
        args.sparse_backend_args = None
        error, message = AssertionError, "GlmKpoolBackendForwardArgs"
    elif invalid == "selection_type":
        args.sparse_backend_args = SimpleNamespace(topk_rows=case.indices, topk_indices=None)
        error, message = AssertionError, "GlmKpoolBackendForwardArgs"
    elif invalid == "missing_rows":
        args.sparse_backend_args = GlmKpoolBackendForwardArgs()
        message = "pool-expanded selection"
    elif invalid == "legacy_indices":
        args.sparse_backend_args.topk_indices = case.indices
        message = "not request-local topk_indices"
    elif invalid == "v":
        v = torch.zeros(1)
        message = "v must be None"
    elif invalid in ("out_scale", "out_scale_sf", "output_sf"):
        setattr(args, invalid, torch.tensor(1.0))
    else:
        args.output = torch.empty(
            2,
            9 if invalid == "shape" else 8,
            dtype=torch.float32 if invalid == "dtype" else case.q.dtype,
            device="meta" if invalid == "device" else "cpu",
        )
        message = "forward_args.output must be"
    with pytest.raises(error, match=message):
        case.backend.forward(case.q, None, v, case.metadata, args)
    case.backend._cache_state.assert_not_called()
    case.backend._dispatch_sparse_core.assert_not_called()


def test_forward_rejects_mixed_phase(forward_case):
    case = forward_case
    selection = GlmKpoolBackendForwardArgs(topk_rows=case.indices)
    args = AttentionForwardArgs(
        attention_input_type=AttentionInputType.mixed, sparse_backend_args=selection
    )
    with pytest.raises(ValueError, match="phase-explicit"):
        case.backend.forward(case.q, None, None, case.metadata, args)
    case.backend._dispatch_sparse_core.assert_not_called()


@pytest.mark.parametrize("context", [False, True])
def test_forward_rejects_explicit_latent_source(forward_case, context):
    case = forward_case
    args = AttentionForwardArgs(
        attention_input_type=(
            AttentionInputType.context_only if context else AttentionInputType.generation_only
        ),
        sparse_backend_args=GlmKpoolBackendForwardArgs(topk_rows=case.indices),
    )
    k = torch.zeros(3, 4, dtype=case.q.dtype)
    message = "k must be None"
    with pytest.raises(ValueError, match=message):
        case.backend.forward(case.q, k, None, case.metadata, args)
    case.backend._dispatch_sparse_core.assert_not_called()


@pytest.mark.parametrize("is_cuda_graph", [False, True])
def test_cache_state_uses_prepared_metadata_without_fallback(is_cuda_graph):
    backend = object.__new__(GlmKpoolSparseAttention)
    backend.layer_idx = 0
    latent = torch.zeros(4, 8, 1, 512, dtype=torch.bfloat16)
    index = torch.zeros(4, 8, 1, 384, dtype=torch.bfloat16)
    tables = torch.tensor([[2, 0], [3, 1]], dtype=torch.long)
    live_lengths = torch.tensor([5, 8], dtype=torch.int32)
    prepared = object.__new__(Glm5NextMamba2Metadata)
    prepared.glm_block_tables = tables
    metadata = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            tokens_per_block=8,
            get_latent_state_buffer=lambda _: latent,
            get_index_state_buffer=lambda _: index,
        ),
        seq_lens=torch.ones(2, dtype=torch.long),
        num_contexts=0,
        is_cuda_graph=is_cuda_graph,
        kv_lens_cuda=live_lengths,
        mamba_metadata=prepared,
    )
    state = backend._cache_state(metadata)
    assert state.block_tables.data_ptr() == tables.data_ptr()
    assert state.kv_lens.data_ptr() == live_lengths.data_ptr()
    torch.testing.assert_close(state.kv_lens, live_lengths)
    metadata.mamba_metadata.glm_block_tables = None
    with patch("torch.zeros", side_effect=AssertionError("must not allocate fallback tables")):
        with pytest.raises(RuntimeError, match="requires prepared glm_block_tables"):
            backend._cache_state(metadata)
    metadata.mamba_metadata.glm_block_tables = tables
    metadata.kv_lens_cuda = None
    with pytest.raises(ValueError, match="kv_lens_cuda"):
        backend._cache_state(metadata)
    metadata.kv_lens_cuda = live_lengths
    for invalid in (None, False, SimpleNamespace(glm_block_tables=tables)):
        metadata.mamba_metadata = invalid
        with pytest.raises(AssertionError, match="Glm5NextMamba2Metadata"):
            backend._cache_state(metadata)


def test_metadata_prepare_rejects_non_glm_cache_manager():
    prepared = object.__new__(Glm5NextMamba2Metadata)
    manager = SimpleNamespace(get_batch_slot_tables=Mock())
    with pytest.raises(AssertionError, match="Glm5NextCacheManager"):
        prepared.prepare(SimpleNamespace(kv_cache_manager=manager))
    manager.get_batch_slot_tables.assert_not_called()


@pytest.mark.parametrize("heads", [16, 64])
def test_backend_output_keeps_flat_contract_without_copy(heads):
    storage = torch.randn(3, 64, 512)
    per_head = storage[:, :heads]
    output = GlmKpoolSparseAttention._finalize_output(None, per_head, None)
    assert output.shape == (3, heads * 512)
    assert output.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr()
    torch.testing.assert_close(output.view(3, heads, 512), per_head)
    supplied = torch.empty(3, heads * 512)
    assert GlmKpoolSparseAttention._finalize_output(None, per_head, supplied) is supplied
    torch.testing.assert_close(supplied, output)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"index_kpool": 0},
        {"index_kpool": 3},
        {"index_topk": 0},
        {"index_topk": 7},
        {"index_always_select_tail": False},
    ],
)
def test_unsupported_pool_layout_fails_before_kernel_launch(kwargs):
    with pytest.raises(ValueError, match="glm_kpool requires"):
        GlmKpoolSparseParams(**kwargs)


def test_derived_geometry_follows_the_checkpoint_defaults():
    params = GlmKpoolSparseParams()
    assert params.algorithm == "glm_kpool"
    assert params.select_k == 2048 // 4
    # topk positions plus the always-visible incomplete tail (kpool - 1 rows).
    assert params.output_width == 2048 + 3
    # Padded to the FlashMLA top-k tile.
    assert params.kernel_output_width == 2112
    assert params.packed_state_dim == 2 * 128
    assert params.cache_row_dim == 3 * 128
    small = GlmKpoolSparseParams(index_topk=64, index_kpool=8)
    assert (small.select_k, small.output_width, small.kernel_output_width) == (8, 71, 128)


def test_paged_slot_indices_returns_page_and_offset_pairs():
    table = torch.tensor([[7, 2, 9], [4, 0, 1]])
    positions = torch.tensor([[0, 5, 16], [3, 8, 23]])
    page, offset = paged_slot_indices(table, positions, tokens_per_block=8)
    assert torch.equal(page, torch.tensor([[7, 7, 9], [4, 0, 1]]))
    assert torch.equal(offset, torch.tensor([[0, 5, 0], [3, 0, 7]]))


def test_latent_pool_rows_reinterprets_a_coalesced_pool_without_copying():
    dim, tpb, slots = 16, 4, 3
    # A wider shared storage: each slot holds this buffer's page plus another
    # buffer's payload, so the slot stride exceeds tpb * dim (V2 coalescing).
    storage = torch.arange(slots * 2 * tpb * dim, dtype=torch.float32).view(slots, 2 * tpb, dim)
    pool = storage[:, :tpb, :]
    rows, base_row, rows_per_slot = latent_pool_rows(pool)
    assert (base_row, rows_per_slot) == (0, 2 * tpb)
    assert rows.data_ptr() == pool.data_ptr()
    for slot in range(slots):
        for t in range(tpb):
            assert torch.equal(rows[base_row + slot * rows_per_slot + t, 0], pool[slot, t])
    # A storage offset that is a whole number of rows is folded into base_row.
    shifted = storage.view(-1)[2 * dim :].view(-1, dim)[: slots * tpb].view(slots, tpb, dim)
    _, base, per_slot = latent_pool_rows(shifted)
    assert (base, per_slot) == (2, tpb)


def test_latent_pool_rows_rejects_layouts_without_a_uniform_row_view():
    dim, tpb, slots = 16, 4, 3
    ragged = torch.zeros(slots, tpb, dim + 1)[..., :dim]  # rows not contiguous within a page
    with pytest.raises(ValueError, match="contiguous within a page"):
        latent_pool_rows(ragged)
    odd_stride = torch.zeros(slots * tpb * dim + slots * 3).as_strided(
        (slots, tpb, dim), (tpb * dim + 3, dim, 1)
    )
    with pytest.raises(ValueError, match="not multiples of dim"):
        latent_pool_rows(odd_stride)

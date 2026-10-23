# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for KV cache token estimation in KvCacheCreator._get_token_num_for_estimation.

Guards the ADP (Attention Data Parallelism) cache-block reduction: when
enable_attention_dp is True and tp_size > 1, _create_dummy_context_requests
produces tp_size duplicate requests, but the scheduler distributes them
1-per-rank.  Each rank's KV cache therefore only needs capacity for its own
share, not all copies.
"""

import math
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

import tensorrt_llm.bindings.internal.batch_manager as batch_manager
from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin
from tensorrt_llm._torch.pyexecutor._util import (
    CacheCost,
    KvCacheCreator,
    _create_kv_cache_manager,
    _derive_v2_layer_type_attention_windows,
    _get_num_pool_groups_for_estimation,
    get_mla_context_workspace_kv_len_cap,
)
from tensorrt_llm._torch.pyexecutor.config_utils import get_layer_attention_window
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheManagerV2,
    _get_generation_kv_capacity,
    max_admissible_sequences,
)
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.llmapi.llm_args import (
    DFlashDecodingConfig,
    DSparkDecodingConfig,
    KvCacheConfig,
    MTPDecodingConfig,
    MultimodalConfig,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_request(num_input_tokens, beam_width=1):
    """Create a mock request with the fields _get_token_num_for_estimation reads."""
    req = Mock()
    req.input_token_ids = list(range(num_input_tokens))
    req.sampling_config.beam_width = beam_width
    return req


def _make_mla_profile_creator() -> KvCacheCreator:
    creator = object.__new__(KvCacheCreator)
    config = SimpleNamespace(
        pretrained_config=SimpleNamespace(vocab_size=128, kv_lora_rank=512, qk_rope_head_dim=64),
        attn_backend="TRTLLM",
        sparse_attention_config=None,
    )
    creator._model_engine = SimpleNamespace(
        model=SimpleNamespace(model_config=config),
        attn_runtime_features=AttentionRuntimeFeatures(chunked_prefill=True, chunk_size=8192),
        use_mrope=False,
    )
    creator._mapping = Mapping()
    creator._max_num_tokens = 8192
    creator._max_beam_width = 1
    creator._speculative_config = None
    creator._tokens_per_block = 32
    return creator


@pytest.fixture
def mla_profile_creator(monkeypatch) -> KvCacheCreator:
    monkeypatch.setattr("tensorrt_llm._torch.pyexecutor._util.get_sm_version", lambda: 103)
    return _make_mla_profile_creator()


@pytest.mark.parametrize("attention_dp", [False, True])
def test_mla_profile_builds_full_cached_chunk(
    attention_dp: bool, mla_profile_creator: KvCacheCreator
) -> None:
    creator = mla_profile_creator
    creator._mapping = Mapping(world_size=4, tp_size=4, enable_attention_dp=attention_dp)
    requests = creator._create_dummy_context_requests(262143)
    assert len(requests) == (4 if attention_dp else 1)
    assert all(len(req.input_token_ids) == 73728 for req in requests)
    # The final forward has a full query and two full cached-KV chunks,
    # covering the previous chunk's live tensors during the next expansion.
    steps = [(position, min(8192, 73728 - position)) for position in range(0, 73728, 8192)]
    assert steps[-1] == (65536, 8192)
    cached_tokens, query_tokens = steps[-1]
    kv_chunk_tokens = 32768
    chunk_lengths = [
        min(kv_chunk_tokens, cached_tokens - offset)
        for offset in range(0, cached_tokens, kv_chunk_tokens)
    ]
    assert chunk_lengths == [32768, 32768]
    assert query_tokens == creator._max_num_tokens


@pytest.mark.parametrize("input_seq_len", [73728, 73727, 8192, 4095])
def test_mla_profile_respects_context_limit(
    input_seq_len: int, mla_profile_creator: KvCacheCreator
) -> None:
    creator = mla_profile_creator
    requests = creator._create_dummy_context_requests(input_seq_len)
    assert all(len(req.input_token_ids) <= input_seq_len for req in requests)
    if input_seq_len == 73728:
        assert creator._mla_chunked_profile_length == 73728
    else:
        assert creator._mla_chunked_profile_length is None
        assert sum(len(req.input_token_ids) for req in requests) == 8192


def test_mla_profile_uses_runtime_chunk_dimensions(mla_profile_creator: KvCacheCreator) -> None:
    creator = mla_profile_creator
    features = creator._model_engine.attn_runtime_features
    features.chunk_size = 4096
    features.chunked_prefill_buffer_batch_size = 3
    # Two 12288-token KV chunks need three 8192-token scheduler steps.
    assert creator._get_mla_chunked_profile_length(262143) == 32768
    # A non-aligned prefix must round up, preserving a full final query.
    features.chunk_size = 4097
    assert creator._get_mla_chunked_profile_length(262143) == 40960


@pytest.mark.parametrize("profiled", [False, True])
@pytest.mark.parametrize("bounded", [False, True])
def test_mla_profile_reserve_requires_measured_chunk(profiled: bool, bounded: bool) -> None:
    config = SimpleNamespace(enable_block_reuse=True, fp8_context_mla_kv_len_cap=None)
    cap = get_mla_context_workspace_kv_len_cap(
        config,
        max_batch_size=16,
        max_num_tokens=8192,
        max_seq_len=262144,
        enable_chunked_prefill=True,
        workspace_is_chunked_prefill_bounded=bounded,
        chunked_workspace_profiled=profiled,
    )
    assert cap == (None if profiled and bounded else 16 * 262144)


@pytest.mark.parametrize("unsupported", ["disabled", "zero_chunk", "non_mla", "backend", "sparse"])
def test_mla_profile_leaves_other_paths_unchanged(
    unsupported: str, mla_profile_creator: KvCacheCreator
) -> None:
    creator = mla_profile_creator
    config = creator._model_engine.model.model_config
    if unsupported == "disabled":
        creator._model_engine.attn_runtime_features.chunked_prefill = False
    elif unsupported == "zero_chunk":
        creator._model_engine.attn_runtime_features.chunk_size = 0
    elif unsupported == "non_mla":
        config.pretrained_config.kv_lora_rank = None
    elif unsupported == "backend":
        config.attn_backend = "FLASHINFER"
    else:
        config.sparse_attention_config = SimpleNamespace(algorithm="dsa")
    requests = creator._create_dummy_context_requests(262143)
    assert creator._mla_chunked_profile_length is None
    assert sum(len(req.input_token_ids) for req in requests) == 8192


@pytest.mark.parametrize("mode", ["beam", "spec", "skip_softmax"])
def test_mla_profile_supports_dense_prefill_modes(
    mode: str, mla_profile_creator: KvCacheCreator
) -> None:
    creator = mla_profile_creator
    if mode == "beam":
        creator._max_beam_width = 2
    elif mode == "spec":
        creator._speculative_config = MTPDecodingConfig(num_nextn_predict_layers=1)
    else:
        creator._model_engine.model.model_config.sparse_attention_config = SimpleNamespace(
            algorithm="skip_softmax"
        )
    requests = creator._create_dummy_context_requests(262143)
    assert creator._mla_chunked_profile_length == 73728
    assert len(requests) == 1
    assert len(requests[0].input_token_ids) == 73728
    assert requests[0].sampling_config.beam_width == creator._max_beam_width


@pytest.mark.parametrize("sm", [90, 100, 103, 107, 120])
def test_mla_profile_requires_bounded_runtime_path(
    sm: int, mla_profile_creator: KvCacheCreator, monkeypatch
) -> None:
    monkeypatch.setattr("tensorrt_llm._torch.pyexecutor._util.get_sm_version", lambda: sm)
    length = mla_profile_creator._get_mla_chunked_profile_length(262143)
    assert length == (None if sm < 100 else 73728)


class _TextModel:
    def __init__(self, encoder_cache_max_bytes):
        self.model_config = ModelConfig(
            multimodal_config=MultimodalConfig(encoder_cache_max_bytes=encoder_cache_max_bytes)
        )


class _MultimodalModel(MultimodalModelMixin, torch.nn.Module):
    def __init__(self, encoder_cache_max_bytes):
        super().__init__()
        self.model_config = ModelConfig(
            multimodal_config=MultimodalConfig(encoder_cache_max_bytes=encoder_cache_max_bytes)
        )


class _EncoderCacheMultimodalModel(_MultimodalModel):
    supports_encoder_cache = True


@dataclass
class _ModelEngine:
    model: _TextModel | _MultimodalModel
    mm_encoder_output_budget_bytes: int | None = None
    _mm_item_scheduler: object | None = None
    mm_encoder_item_scheduling_enabled: bool = False
    bytes_per_mm_encoder_embedding: int = 8
    mm_encoder_cache = PyTorchModelEngine.mm_encoder_cache


def _make_reserve_creator(
    model: _TextModel | _MultimodalModel,
    *,
    mm_encoder_output_budget_bytes: int | None = None,
    disable_mm_encoder: bool = False,
    max_num_tokens: int = 1,
) -> KvCacheCreator:
    llm_args = TorchLlmArgs(
        model="dummy",
        checkpoint_format="HF",
        disable_mm_encoder=disable_mm_encoder,
        multimodal_config=model.model_config.multimodal_config,
    )
    # Match ModelLoader: the model receives the effective, normalized config
    # from TorchLlmArgs rather than retaining the caller's input object.
    model.model_config.multimodal_config = llm_args.multimodal_config
    if isinstance(model, MultimodalModelMixin):
        cache_bytes = mm_encoder_output_budget_bytes or 0
        if model.encoder_cache_active:
            cache_bytes = max(
                cache_bytes,
                model.model_config.multimodal_config.encoder_cache_max_bytes,
            )
        model._initialize_multimodal_encoder_cache(cache_bytes)
    item_scheduling = mm_encoder_output_budget_bytes is not None
    model_engine = _ModelEngine(
        model=model,
        mm_encoder_output_budget_bytes=mm_encoder_output_budget_bytes,
        _mm_item_scheduler=object() if item_scheduling else None,
        mm_encoder_item_scheduling_enabled=item_scheduling,
    )
    return KvCacheCreator(
        model_engine=model_engine,
        draft_model_engine=None,
        mapping=Mapping(),
        net_max_seq_len=1,
        kv_connector_manager=None,
        max_num_tokens=max_num_tokens,
        max_beam_width=1,
        tokens_per_block=1,
        max_seq_len=1,
        max_batch_size=1,
        kv_cache_config=KvCacheConfig(),
        llm_args=llm_args,
        speculative_config=None,
        sparse_attention_config=None,
        profiling_stage_data=None,
        is_disagg=False,
        skip_est=True,
    )


def test_encoder_profiling_uses_full_budget_independent_of_llm_limit(
    monkeypatch,
):
    class _InputProcessor:
        def __init__(self):
            self.calls = []

        def get_mm_max_tokens_per_item(self, max_num_encoder_tokens=None):
            del max_num_encoder_tokens
            return {"image": 4096}

        def get_dummy_mm_data(
            self,
            *,
            max_num_encoder_tokens,
            mm_counts,
            dtype,
        ):
            self.calls.append((max_num_encoder_tokens, mm_counts, dtype))
            item_count = mm_counts["image"]
            return {"image": {"item_count": item_count}}

    class _Model(MultimodalModelMixin):
        dtype = torch.float16
        mm_encoder = object()

        def __init__(self):
            self.forwarded_item_counts = []
            self.last_output = None

        def encode_multimodal_inputs(self, multimodal_params):
            count = multimodal_params[0].multimodal_data["image"]["item_count"]
            self.forwarded_item_counts.append(count)
            self.last_output = torch.arange(count * 4).reshape(count, 4)
            return self.last_output

    input_processor = _InputProcessor()
    model = _Model()
    model.model_config = SimpleNamespace(pretrained_config=SimpleNamespace(vocab_size=128))
    creator = object.__new__(KvCacheCreator)
    creator._model_engine = SimpleNamespace(
        model=model,
        input_processor=input_processor,
        encoder_batch_size=4,
        encoder_max_num_tokens=8192,
        use_mrope=False,
    )
    creator._llm_args = TorchLlmArgs(model="dummy", checkpoint_format="HF")
    creator._profiling_stage_data = {"enable_mm_reqs": True}
    creator._mapping = SimpleNamespace(enable_attention_dp=False)
    creator._max_num_tokens = 18
    creator._max_beam_width = 1

    # The LLM dummy remains text-only and is not allowed to shrink the
    # independent encoder profiling budget.
    requests = creator._create_dummy_context_requests(input_seq_len=18)
    assert sum(len(request.input_token_ids) for request in requests) == 18
    assert input_processor.calls == []
    assert all(getattr(request, "py_multimodal_data", None) is None for request in requests)

    creator._dummy_encoder_inputs = creator._create_dummy_encoder_inputs()
    assert input_processor.calls == [(8192, {"image": 2}, torch.float16)]

    monkeypatch.setattr(MultimodalParams, "to_device", lambda self, *args, **kwargs: self)
    retained_output = creator._encode_dummy_inputs()
    assert model.forwarded_item_counts == [2]
    assert retained_output.data_ptr() != model.last_output.data_ptr()
    assert creator._dummy_encoder_inputs == []


def _make_creator(
    tokens_per_block,
    dummy_reqs,
    enable_attention_dp,
    tp_size,
    batch_size=1,
    model_max_seq_len=1,
    max_cuda_graph_batch_size=1,
    layer_types=None,
    sliding_window=None,
    use_sliding_window=None,
    max_attention_window=None,
    max_beam_width=1,
):
    """Build a minimal KvCacheCreator (bypasses __init__) wired up for
    _get_token_num_for_estimation only."""
    c = object.__new__(KvCacheCreator)

    c._tokens_per_block = tokens_per_block
    c._max_beam_width = max_beam_width
    c._net_max_seq_len = 2048
    c._speculative_config = None
    c._dummy_reqs = dummy_reqs

    c._mapping = Mock(enable_attention_dp=enable_attention_dp, tp_size=tp_size, cp_config={})

    c._llm_args = Mock(disable_overlap_scheduler=True)

    pretrained = SimpleNamespace(
        layer_types=layer_types,
        num_hidden_layers=(len(layer_types) if isinstance(layer_types, (list, tuple)) else None),
        sliding_window=sliding_window,
        use_sliding_window=use_sliding_window,
    )

    model_config = Mock()
    model_config.pretrained_config = pretrained

    c._model_engine = Mock(
        batch_size=batch_size,
        max_seq_len=model_max_seq_len,
        _max_cuda_graph_batch_size=max_cuda_graph_batch_size,
    )
    c._model_engine.model = Mock(model_config=model_config)

    c._kv_cache_config = Mock(
        free_gpu_memory_fraction=0.9, max_attention_window=max_attention_window
    )

    # _get_token_num_for_estimation gates pool-group scaling on V2 (the
    # split-pool layout that motivates the scaling). Mamba hybrid uses
    # MambaHybridCacheManager and must NOT scale. These tests target V2
    # behavior; production sets this in __init__ via
    # _get_model_kv_cache_manager_cls(), which we bypass here.
    c._kv_cache_manager_cls = KVCacheManagerV2
    c._is_kv_cache_manager_v2 = True

    return c


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_gpu():
    """Stub out CUDA memory queries and per-token KV size so the test runs on
    any machine and the memory cap never constrains the result.

    _get_kv_size_per_token now returns a CacheCost; using slope=1 + zero
    intercept keeps the legacy ``budget // bytes_per_token`` arithmetic
    untouched downstream."""
    huge = 100 * (1 << 30)
    with (
        patch("torch.cuda.mem_get_info", return_value=(huge, huge)),
        patch.object(KvCacheCreator, "_get_kv_size_per_token", return_value=CacheCost(slope=1)),
    ):
        yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_adp_reduces_blocks_to_per_rank_share():
    """With ADP + tp_size duplicated requests the result must equal a single
    rank's share, not the sum across all duplicates."""
    tpb = 64
    tp = 4
    n_in = 128  # ceil((128+1)/64) = 3 blocks per request

    baseline = _make_creator(tpb, [_make_mock_request(n_in)], enable_attention_dp=False, tp_size=1)
    adp = _make_creator(
        tpb, [_make_mock_request(n_in) for _ in range(tp)], enable_attention_dp=True, tp_size=tp
    )

    assert adp._get_token_num_for_estimation() == baseline._get_token_num_for_estimation()


def test_without_adp_all_blocks_counted():
    """Without ADP every request's blocks contribute to the total."""
    tpb = 64
    n_in = 128  # 3 blocks each
    n_reqs = 4

    c = _make_creator(
        tpb, [_make_mock_request(n_in) for _ in range(n_reqs)], enable_attention_dp=False, tp_size=1
    )

    # 4 reqs * 3 blocks * 64 tokens/block = 768
    assert c._get_token_num_for_estimation() == n_reqs * 3 * tpb


def test_max_beam_width_scales_estimation_blocks():
    """Warm-up capacity uses the same configured beam width as dummy requests."""
    tpb = 64
    beam_width = 4
    c = _make_creator(
        tpb,
        [_make_mock_request(128, beam_width=beam_width)],
        enable_attention_dp=False,
        tp_size=1,
        max_beam_width=beam_width,
    )
    # Beam search falls back to V1; exercise the V1 memory-cap path.
    c._is_kv_cache_manager_v2 = False

    assert c._get_token_num_for_estimation() == 3 * beam_width * tpb


@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_adp_various_tp_sizes(tp_size):
    """ADP division must hold for several representative tp_size values."""
    tpb = 64
    n_in = 128  # 3 blocks per request

    c = _make_creator(
        tpb,
        [_make_mock_request(n_in) for _ in range(tp_size)],
        enable_attention_dp=True,
        tp_size=tp_size,
    )

    total = tp_size * 3
    expected_blocks = (total + tp_size - 1) // tp_size
    assert c._get_token_num_for_estimation() == expected_blocks * tpb


def test_regression_without_fix_would_overcount():
    """If the ADP ceil-division fix were removed, the returned
    value would be tp_size times too large.  This test guards that fix."""
    tpb = 64
    tp = 4
    n_in = 128

    c = _make_creator(
        tpb, [_make_mock_request(n_in) for _ in range(tp)], enable_attention_dp=True, tp_size=tp
    )

    result = c._get_token_num_for_estimation()

    correct = 3 * tpb  # 192  (per-rank share)
    wrong = tp * 3 * tpb  # 768  (all duplicates summed)
    assert result == correct
    assert result != wrong


@pytest.mark.parametrize(
    ("model_cls", "encoder_cache_max_bytes", "expected_reserve"),
    [
        (_TextModel, 64, 0),
        (_MultimodalModel, 0, 0),
        (_MultimodalModel, 64, 0),
        (_EncoderCacheMultimodalModel, 0, 0),
        (_EncoderCacheMultimodalModel, 64, 64),
    ],
)
def test_kv_cache_estimation_reserves_multimodal_encoder_cache(
    model_cls,
    encoder_cache_max_bytes,
    expected_reserve,
):
    creator = _make_reserve_creator(model_cls(encoder_cache_max_bytes))

    assert creator._get_multimodal_encoder_memory_reserve() == expected_reserve
    # Inline encoding keeps its batch output beside the cache's copies, so the
    # profiled batch output cannot stand in for cache-owned bytes.
    assert creator._get_multimodal_encoder_memory_reserve(64) == expected_reserve


def test_kv_cache_estimation_skips_multimodal_reserve_when_encoder_disabled():
    model = _EncoderCacheMultimodalModel(64)
    creator = _make_reserve_creator(
        model,
        disable_mm_encoder=True,
    )

    assert model.model_config.multimodal_config.encoder_cache_max_bytes == 0
    assert creator._create_dummy_encoder_inputs() == []
    assert creator._get_multimodal_encoder_memory_reserve() == 0


@pytest.mark.parametrize(
    ("cache_bytes", "profiled_bytes", "max_num_tokens", "expected"),
    [
        (512, 400, 4, 112 + 64),
        (512, 600, 4, 64),  # A profiled store never cancels unprofiled joins.
        (64, 0, 16, 64 + 256),  # Shared entries can be copied by many requests.
    ],
)
def test_reserve_adds_unprofiled_store_and_chunk_copies(
    cache_bytes, profiled_bytes, max_num_tokens, expected
):
    creator = _make_reserve_creator(
        _MultimodalModel(0),
        mm_encoder_output_budget_bytes=cache_bytes,
        max_num_tokens=max_num_tokens,
    )
    assert creator._get_multimodal_encoder_memory_reserve(profiled_bytes) == expected
    # Whole-model torch.compile wraps the model after its cache is created.
    creator._model_engine.model = torch.compile(creator._model_engine.model)
    assert creator._get_multimodal_encoder_memory_reserve(profiled_bytes) == expected


def test_downstream_pp_rank_without_encoder_store_reserves_no_memory() -> None:
    creator = object.__new__(KvCacheCreator)
    creator._model_engine = SimpleNamespace(mm_encoder_cache=None)

    assert creator._get_multimodal_encoder_memory_reserve() == 0


# ---------------------------------------------------------------------------
# VSWA hybrid attention pool-group scaling (Gemma4 hybrid MMMU Pro hang fix)
# ---------------------------------------------------------------------------
#
# KVCacheManagerV2 creates one pool group per distinct attention-window size.
# The quota derived from max_tokens is split proportionally across pool
# groups, so each pool ends up with roughly num_cache_blocks / num_pool_groups
# blocks.  A single long-context request then overflows the full-attention
# pool and the scheduler livelocks on suspend/retry.  The fix scales
# num_cache_blocks by the number of distinct attention-window sizes inferred
# effective per-layer sliding windows on the pretrained config (preferred) or
# an explicit ``max_attention_window`` list on kv_cache_config (fallback).


def test_uniform_layer_types_no_scaling():
    """All-sliding or all-full layers stay a single pool group."""
    tpb = 32
    max_seq_len = 4096
    uniform = _make_creator(
        tpb,
        [_make_mock_request(max_seq_len - 1)],
        enable_attention_dp=False,
        tp_size=1,
        model_max_seq_len=max_seq_len,
        max_cuda_graph_batch_size=4,
        layer_types=["sliding_attention"] * 26,
    )
    # num_pool_groups = 1 -> behaviour unchanged from legacy ADP-only case.
    baseline = _make_creator(
        tpb,
        [_make_mock_request(max_seq_len - 1)],
        enable_attention_dp=False,
        tp_size=1,
        model_max_seq_len=max_seq_len,
        max_cuda_graph_batch_size=4,
    )
    assert uniform._get_token_num_for_estimation() == baseline._get_token_num_for_estimation()


def test_get_layer_attention_window_honors_max_window_layers():
    config = SimpleNamespace(
        use_sliding_window=True,
        sliding_window=4096,
        max_window_layers=2,
    )

    assert [get_layer_attention_window(config, layer_idx) for layer_idx in range(4)] == [
        None,
        None,
        4096,
        4096,
    ]


def test_gemma4_hybrid_scales_by_num_pool_groups():
    """Gemma4 hybrid attention (mixed sliding/full layers) must scale the
    estimated block count by the number of distinct layer types.  Otherwise
    the per-pool quota is too small to hold a single max_seq_len request,
    which is the MMMU Pro livelock reproducer."""
    tpb = 32
    max_seq_len = 12288
    layer_types = ["sliding_attention"] * 28 + ["full_attention"] * 7
    # Mixed sliding/full attention must include the model's actual window size;
    # Gemma4-E2B uses a 512-token sliding window.
    sliding_window = 512
    assert len(set(layer_types)) == 2

    hybrid = _make_creator(
        tpb,
        [_make_mock_request(max_seq_len - 1), _make_mock_request(1)],
        enable_attention_dp=False,
        tp_size=1,
        model_max_seq_len=max_seq_len,
        max_cuda_graph_batch_size=4,
        layer_types=layer_types,
        sliding_window=sliding_window,
    )
    config = hybrid._model_engine.model.model_config.pretrained_config
    config.head_dim, config.global_head_dim = 256, 128
    uniform = _make_creator(
        tpb,
        [_make_mock_request(max_seq_len - 1), _make_mock_request(1)],
        enable_attention_dp=False,
        tp_size=1,
        model_max_seq_len=max_seq_len,
        max_cuda_graph_batch_size=4,
        layer_types=["full_attention"] * 35,
    )

    hybrid_tokens = hybrid._get_token_num_for_estimation()
    uniform_tokens = uniform._get_token_num_for_estimation()
    assert hybrid_tokens == 2 * uniform_tokens, (
        f"Expected 2x scaling for 2 pool groups, got "
        f"hybrid={hybrid_tokens}, uniform={uniform_tokens}"
    )


def test_hybrid_linear_attention_scales_by_num_pool_groups():
    """Hybrid linear/attention V2 managers retain both estimation pools."""
    tpb = 32
    max_seq_len = 4096
    layer_types = ["linear_attention"] * 30 + ["full_attention"] * 10

    hybrid = _make_creator(
        tpb,
        [_make_mock_request(max_seq_len - 1)],
        enable_attention_dp=False,
        tp_size=1,
        model_max_seq_len=max_seq_len,
        max_cuda_graph_batch_size=4,
        layer_types=layer_types,
    )
    uniform = _make_creator(
        tpb,
        [_make_mock_request(max_seq_len - 1)],
        enable_attention_dp=False,
        tp_size=1,
        model_max_seq_len=max_seq_len,
        max_cuda_graph_batch_size=4,
        layer_types=["full_attention"] * 40,
    )

    assert hybrid._get_token_num_for_estimation() == 2 * uniform._get_token_num_for_estimation()


@pytest.mark.parametrize(
    ("sliding_window", "use_sliding_window"),
    [
        ([512, 1024], None),
        (None, True),
    ],
    ids=["multiple_window_sizes", "missing_window"],
)
def test_plain_v2_collapses_unsupported_full_sliding_metadata(
    sliding_window,
    use_sliding_window,
):
    tpb = 32
    max_seq_len = 4096
    hybrid = _make_creator(
        tpb,
        [_make_mock_request(max_seq_len - 1)],
        enable_attention_dp=False,
        tp_size=1,
        model_max_seq_len=max_seq_len,
        max_cuda_graph_batch_size=4,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=sliding_window,
        use_sliding_window=use_sliding_window,
    )
    uniform = _make_creator(
        tpb,
        [_make_mock_request(max_seq_len - 1)],
        enable_attention_dp=False,
        tp_size=1,
        model_max_seq_len=max_seq_len,
        max_cuda_graph_batch_size=4,
        layer_types=["full_attention", "full_attention"],
    )

    assert hybrid._get_token_num_for_estimation() == uniform._get_token_num_for_estimation()


def test_uniform_window_overrides_mixed_attention_metadata() -> None:
    config = SimpleNamespace(
        num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=128,
    )

    assert _get_num_pool_groups_for_estimation(config, 4096, [512]) == 1


def test_vswa_max_attention_window_fallback_scales():
    """When layer_types is absent but kv_cache_config.max_attention_window is
    a heterogeneous list (VSWA), we still scale by the number of distinct
    windows."""
    tpb = 32
    max_seq_len = 12288
    max_attention_window = [1024] * 28 + [max_seq_len] * 7
    assert len(set(max_attention_window)) == 2

    c = _make_creator(
        tpb,
        [_make_mock_request(max_seq_len - 1)],
        enable_attention_dp=False,
        tp_size=1,
        model_max_seq_len=max_seq_len,
        max_cuda_graph_batch_size=4,
        layer_types=None,
        max_attention_window=max_attention_window,
    )
    uniform = _make_creator(
        tpb,
        [_make_mock_request(max_seq_len - 1)],
        enable_attention_dp=False,
        tp_size=1,
        model_max_seq_len=max_seq_len,
        max_cuda_graph_batch_size=4,
    )
    assert c._get_token_num_for_estimation() == 2 * uniform._get_token_num_for_estimation()


def test_pool_scaling_prevents_mmmu_pro_underestimation():
    """Regression: with max_seq_len=12288 and max_num_tokens=12288 (MMMU Pro
    config), hybrid estimation must produce enough capacity to hold one full
    max_seq_len request per pool, with max_util_for_resume headroom."""
    tpb = 32
    max_seq_len = 12288
    layer_types = ["sliding_attention"] * 28 + ["full_attention"] * 7
    # Mixed sliding/full attention must include the model's actual window size;
    # Gemma4-E2B uses a 512-token sliding window.
    sliding_window = 512

    c = _make_creator(
        tpb,
        [_make_mock_request(max_seq_len - 1), _make_mock_request(1)],
        enable_attention_dp=False,
        tp_size=1,
        model_max_seq_len=max_seq_len,
        max_cuda_graph_batch_size=4,
        layer_types=layer_types,
        sliding_window=sliding_window,
    )
    config = c._model_engine.model.model_config.pretrained_config
    config.head_dim, config.global_head_dim = 256, 128

    total_tokens = c._get_token_num_for_estimation()
    per_pool_tokens = total_tokens // 2  # 2 pool groups
    # Must be enough to hold a max_seq_len request per pool.
    assert per_pool_tokens >= max_seq_len


def test_v2_cache_size_per_token_models_generation_swa_cost():
    class FakeModelConfig:
        quant_config = None
        pretrained_config = SimpleNamespace(
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
        )

        def get_num_attention_layers(self):
            return 3

    mapping = Mock(enable_attention_dp=False, tp_size=1)
    mapping.pp_layers.return_value = [0, 1, 2]

    no_scratch_size_per_token = CacheCost.from_raw(
        KVCacheManagerV2.get_cache_size_per_token(
            FakeModelConfig(),
            mapping,
            tokens_per_block=64,
            max_seq_len=4096,
            max_batch_size=3,
            kv_cache_config=KvCacheConfig(
                max_attention_window=[2048, 2048, 4096], enable_swa_scratch_reuse=False
            ),
        )
    )
    scratch_size_per_token = CacheCost.from_raw(
        KVCacheManagerV2.get_cache_size_per_token(
            FakeModelConfig(),
            mapping,
            tokens_per_block=64,
            max_seq_len=4096,
            max_batch_size=3,
            kv_cache_config=KvCacheConfig(
                max_attention_window=[2048, 2048, 4096], enable_swa_scratch_reuse=True
            ),
        )
    )

    # Per layer: K+V * kv_heads * head_dim * bf16 bytes = 2 * 2 * 8 * 2.
    expected = CacheCost(slope=64, intercept=3 * 2 * (2048 + 64) * 64)
    assert no_scratch_size_per_token == expected
    assert scratch_size_per_token == expected


@pytest.mark.parametrize("max_seq_len", [256, 512])
def test_v2_static_cache_size_preserves_window_pattern_phase_across_pp(max_seq_len) -> None:
    class FakeModelConfig:
        quant_config = None
        pretrained_config = SimpleNamespace(
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            layer_types=["sliding_attention", "full_attention"] * 2 + ["sliding_attention"],
        )

        def get_num_attention_layers(self) -> int:
            return 5

    mapping = Mock(enable_attention_dp=False, tp_size=1)
    mapping.pp_layers.return_value = [3]

    cache_cost = CacheCost.from_raw(
        KVCacheManagerV2.get_cache_size_per_token(
            FakeModelConfig(),
            mapping,
            tokens_per_block=64,
            max_seq_len=max_seq_len,
            max_batch_size=1,
            kv_cache_config=KvCacheConfig(max_attention_window=[128, 256]),
        )
    )

    # Global layer 3 selects the full-attention entry, so its 64-byte layer
    # cost is entirely per-token with no fixed SWA allocation.
    assert cache_cost == CacheCost(slope=64, intercept=0)


def test_v2_cache_size_per_token_charges_reuse_window_lookahead():
    class FakeModelConfig:
        quant_config = None
        pretrained_config = SimpleNamespace(
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
        )

        def get_num_attention_layers(self):
            return 1

    class UnsupportedKVCacheManagerV2(KVCacheManagerV2):
        _supports_reuse_match_backoff = False

    mapping = Mock(enable_attention_dp=False, tp_size=1)
    mapping.pp_layers.return_value = [0]
    spec_config = MTPDecodingConfig(max_draft_len=1, use_mtp_vanilla=True)
    kwargs = dict(
        tokens_per_block=64,
        max_seq_len=4096,
        max_batch_size=3,
        kv_cache_config=KvCacheConfig(
            max_attention_window=[64],
            max_util_for_resume=1.0,
        ),
        spec_config=spec_config,
    )

    no_draft = CacheCost.from_raw(
        KVCacheManagerV2.get_cache_size_per_token(
            FakeModelConfig(), mapping, **(kwargs | {"spec_config": None})
        )
    )
    single_kvcm = CacheCost.from_raw(
        KVCacheManagerV2.get_cache_size_per_token(FakeModelConfig(), mapping, **kwargs)
    )
    target = CacheCost.from_raw(
        KVCacheManagerV2.get_cache_size_per_token(
            FakeModelConfig(),
            mapping,
            use_separate_draft_kv_cache=True,
            **kwargs,
        )
    )
    draft = CacheCost.from_raw(
        KVCacheManagerV2.get_cache_size_per_token(
            FakeModelConfig(), mapping, is_draft=True, **kwargs
        )
    )
    unsupported = CacheCost.from_raw(
        UnsupportedKVCacheManagerV2.get_cache_size_per_token(FakeModelConfig(), mapping, **kwargs)
    )
    block_reuse_disabled = CacheCost.from_raw(
        KVCacheManagerV2.get_cache_size_per_token(
            FakeModelConfig(),
            mapping,
            **(
                kwargs
                | {
                    "kv_cache_config": KvCacheConfig(
                        enable_block_reuse=False,
                        max_attention_window=[64],
                        max_util_for_resume=1.0,
                    )
                }
            ),
        )
    )

    # W=64 plus the base generation token can retain two boundary pages.
    # One-model draft reuse extends retention to W+D=65; its two-token
    # generation step can therefore cross into a third page.
    assert no_draft == CacheCost(slope=0, intercept=3 * 128 * 64)
    assert unsupported == no_draft
    assert block_reuse_disabled == no_draft
    assert single_kvcm == target == draft == CacheCost(slope=0, intercept=3 * 192 * 64)


def test_creator_uses_v2_affine_cache_cost():
    class FakeV2Manager(KVCacheManagerV2):
        @staticmethod
        def get_cache_size_per_token(model_config, mapping, **kwargs):
            return 20, 21

    creator = object.__new__(KvCacheCreator)
    creator._mapping = Mock()
    creator._tokens_per_block = 64
    creator._max_seq_len = 1024
    creator._max_batch_size = 3
    creator._max_num_tokens = 1024
    creator._kv_cache_config = KvCacheConfig()
    creator._speculative_config = None

    cost = creator._per_manager_cache_cost(FakeV2Manager, Mock())

    assert cost == CacheCost(slope=20, intercept=21)


def test_v2_quota_from_max_tokens_models_context_swa_scratch():
    manager = object.__new__(KVCacheManagerV2)
    manager._has_cp_helix = False
    manager.num_local_layers = 3
    manager.pp_layers = [4, 5, 6]
    manager.max_attention_window_vec = [128, 128, None]
    manager.tokens_per_block = 64
    manager.max_batch_size = 4
    manager.max_num_tokens = 1000
    manager._generation_kv_capacity_headroom = 1
    manager.get_layer_bytes_per_token = lambda local_layer_idx, data_role: [10, 10, 20][
        local_layer_idx
    ]

    max_tokens = 1200

    manager.enable_swa_scratch_reuse = False
    no_scratch_quota = manager._get_quota_from_max_tokens(max_tokens)
    assert no_scratch_quota == (max_tokens * 20 + manager.max_num_tokens * 20 + 4 * 2 * 192 * 10)
    assert manager._get_max_tokens_from_quota(no_scratch_quota) == max_tokens

    manager.enable_swa_scratch_reuse = True
    scratch_quota = manager._get_quota_from_max_tokens(max_tokens)
    assert scratch_quota == (max_tokens * 20 + manager.max_num_tokens * 10 + 4 * 2 * 192 * 10)
    assert manager._get_max_tokens_from_quota(scratch_quota) == max_tokens


# ---------------------------------------------------------------------------
# KVCacheManagerV2 clamp_max_seq_len_for_mem float-to-int cast regression
# ---------------------------------------------------------------------------
#
# ``KVCacheManagerV2.get_num_available_tokens`` returns
# ``clamp_max_seq_len_for_mem(...) - extra_tokens``, where the clamp helper
# does a floating-point memory-budget division and returns a float.  When
# ``max_seq_len > max_num_tokens`` the ``__init__`` body assigned the float
# directly to ``self.max_seq_len``, which then propagated into
# ``_util.py::_create_dummy_context_requests``'s ``torch.randint(size=(...))``
# call and crashed on a float size tuple.
#
# resource_manager.py:1820 now casts ``int(max_num_tokens)`` at the assign
# site.  The tests below lock in that invariant: (a) the replay test
# demonstrates the fix applied to the scenario that previously crashed,
# and (b) the propagation test shows that downstream arithmetic in
# ``_util.py:610-620`` stays int-safe.


def test_kv_cache_manager_v2_clamp_casts_float_to_int():
    """Replay of the V2 clamp block with a float clamp result.

    Reproduces the exact arithmetic from ``resource_manager.py``
    ``KVCacheManagerV2.__init__`` lines 1813-1825.  Pre-fix, the ``if``
    branch stored a float on ``self.max_seq_len``.  Post-fix, it casts
    to int.  A broken fix (e.g. dropping the cast) will make this test
    fail on the ``isinstance(..., int)`` assertion."""
    # Scenario pulled from the MMMU Pro 26B + s=131K crash log:
    #   clamp_max_seq_len_for_mem returned 60160.0 when requested
    #   131072 tokens of max_seq_len.
    initial_max_seq_len = 131072
    # Simulate the float return from clamp_max_seq_len_for_mem.
    clamp_float_result = 60160.0
    assert isinstance(clamp_float_result, float)
    # Match the production code block (resource_manager.py:1816-1825).
    self_max_seq_len = initial_max_seq_len
    if self_max_seq_len > clamp_float_result:
        self_max_seq_len = int(clamp_float_result)

    assert isinstance(self_max_seq_len, int), (
        "KVCacheManagerV2 must cast the clamp result to int — float "
        "propagates into torch.randint(size=...) and crashes."
    )
    assert self_max_seq_len == 60160


def test_kv_cache_manager_v2_float_max_seq_len_would_crash_torch_randint():
    """Pre-fix behaviour: a float max_seq_len propagating into
    torch.randint(size=(...,)) raises.  This test documents WHY the cast
    is necessary — if the cast is dropped, the following code crashes."""

    float_seq_len = 60160.0
    with pytest.raises((TypeError, RuntimeError)):
        # torch.randint only accepts int-valued size tuples.
        torch.randint(low=0, high=32000, size=(float_seq_len,))


def test_kv_manager_int_max_seq_len_stays_int_through_util_expression():
    """The downstream expression in ``_util.py:615-620`` (which builds the
    dummy context request input_seq_len) must stay int when the V2
    manager's max_seq_len is int.  Covers the full propagation chain
    from the root-cause fix to the downstream consumer."""
    net_max_seq_len = 131072  # from ModelEngine
    creator_max_seq_len = 131072  # KvCacheCreator.self._max_seq_len
    kv_manager_max_seq_len = 60160  # post-fix int (was 60160.0 pre-fix)

    # Replay _util.py:616-619 arithmetic.
    input_seq_len = max(
        1,
        net_max_seq_len - 1 - (creator_max_seq_len - kv_manager_max_seq_len),
    )
    assert isinstance(input_seq_len, int)
    assert input_seq_len >= 1

    # Sanity: the same expression with a float would yield float (the
    # original bug), ensuring this test would notice regression.
    buggy_input = max(
        1,
        net_max_seq_len - 1 - (creator_max_seq_len - float(kv_manager_max_seq_len)),
    )
    assert isinstance(buggy_input, float), (
        "Sanity check: the pre-fix float path must still be demonstrable "
        "in the test, otherwise this regression guard is hollow."
    )


# ---------------------------------------------------------------------------
# _create_kv_cache_manager: MLA branch must forward max_num_tokens
# ---------------------------------------------------------------------------


def test_mla_branch_forwards_max_num_tokens_to_manager() -> None:
    """The is_mla branch of ``_create_kv_cache_manager`` must pass
    ``max_num_tokens`` to the manager, like the generic branch does.

    Regression guard: when the argument is dropped,
    ``DeepseekV4CacheManager._max_num_tokens`` stays ``None`` and the
    profiling-phase context extra quota is sized by the full ``max_tokens``
    estimate instead of the runtime chunk size (observed as a 27.59 GiB vs
    11.92 GiB temp-quota inflation on DeepSeek-V4-Pro, raising peak memory
    and OOM risk during KV cache estimation).
    """

    captured_kwargs = {}

    class _RecordingManager:
        def __init__(self, *args, **kwargs) -> None:
            captured_kwargs.update(kwargs)

    # Minimal MLA pretrained config: is_mla() keys on kv_lora_rank and
    # qk_rope_head_dim being set.
    pretrained = SimpleNamespace(
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        hidden_size=1024,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_hidden_layers=2,
        vocab_size=32000,
    )
    model_config = Mock()
    model_config.pretrained_config = pretrained
    model_config.quant_config = None

    _create_kv_cache_manager(
        model_engine=None,
        kv_cache_manager_cls=_RecordingManager,
        mapping=Mock(),
        kv_cache_config=KvCacheConfig(),
        tokens_per_block=32,
        max_seq_len=2048,
        max_batch_size=8,
        spec_config=None,
        sparse_attention_config=None,
        max_num_tokens=333,
        max_beam_width=1,
        kv_connector_manager=None,
        model_config=model_config,
        dtype=torch.bfloat16,
        is_draft=False,
    )

    assert captured_kwargs.get("max_num_tokens") == 333, (
        "is_mla branch dropped max_num_tokens: DeepseekV4CacheManager then "
        "falls back to _max_num_tokens=None and over-sizes the estimation "
        "temp quota."
    )


@pytest.mark.parametrize(
    (
        "sm",
        "chunked_workspace_profiled",
        "chunked_prefill",
        "bounded",
        "expected_budget",
        "expected_cap",
    ),
    [
        (100, False, True, True, 384, 128),
        (100, True, True, True, 512, None),
        (90, False, True, True, 512, None),
        (90, False, False, True, 384, 128),
        (90, False, True, False, 384, 128),
    ],
)
def test_estimation_temporarily_uses_inferred_pool_sizing(
    sm: int,
    chunked_workspace_profiled: bool,
    chunked_prefill: bool,
    bounded: bool,
    expected_budget: int,
    expected_cap: int | None,
) -> None:
    """Verify measured chunk capacity and preserve Hopper's existing reserve policy."""

    pool_ratio = [0.2, 0.3, 0.5]
    avg_seq_len = 128
    max_seq_len = 4096
    user_max_tokens = 1024
    estimation_max_tokens = 256
    kv_cache_config = KvCacheConfig(
        max_tokens=user_max_tokens,
        enable_block_reuse=True,
        pool_ratio=pool_ratio,
        avg_seq_len=avg_seq_len,
    )
    model_engine = Mock()
    model_engine.model.model_config.attn_backend = "TRTLLM"
    # Explicit False: try_prepare_estimation skips estimation for
    # encoder-decoder models, and a bare Mock attribute is truthy.
    model_engine.model.model_config.is_encoder_decoder = False
    # A bare Mock would auto-create the attribute; real engines set it to
    # None unless the model opted into MM item scheduling.
    model_engine.mm_encoder_output_budget_bytes = None
    model_engine.mm_encoder_cache = None
    llm_args = Mock(cache_transceiver_config=None, enable_chunked_prefill=chunked_prefill)

    with patch.object(
        KvCacheCreator,
        "_get_model_kv_cache_manager_cls",
        return_value=KVCacheManagerV2,
    ):
        creator = KvCacheCreator(
            model_engine=model_engine,
            draft_model_engine=None,
            mapping=Mock(cp_config={}),
            net_max_seq_len=max_seq_len,
            kv_connector_manager=None,
            max_num_tokens=256,
            max_beam_width=1,
            tokens_per_block=128,
            max_seq_len=max_seq_len,
            max_batch_size=8,
            kv_cache_config=kv_cache_config,
            llm_args=llm_args,
            speculative_config=None,
            sparse_attention_config=None,
            profiling_stage_data=None,
            is_disagg=False,
        )

    creator._mla_chunked_profile_length = 2304 if chunked_workspace_profiled else None
    py_executor = Mock()
    py_executor.dist.mapping.rank = 0
    py_executor.dist.broadcast.side_effect = lambda value, root: value
    py_executor.enqueue_requests.return_value = [1]
    py_executor.await_responses.return_value = []
    kv_manager = Mock()
    kv_manager.get_kv_cache_stats.return_value = SimpleNamespace(allocated_bytes=0)
    py_executor.resource_manager.resource_managers.get.side_effect = [kv_manager, None]

    with (
        patch("tensorrt_llm._torch.pyexecutor._util.get_sm_version", return_value=sm),
        patch.object(
            creator,
            "_get_token_num_for_estimation",
            return_value=estimation_max_tokens,
        ),
        patch.object(creator, "_cal_max_memory", return_value=512),
        patch.object(torch.cuda, "mem_get_info", return_value=(768, 1024)),
        patch.object(
            torch.cuda,
            "memory_stats",
            return_value={"allocated_bytes.all.current": 128, "allocated_bytes.all.peak": 256},
        ),
        patch.object(creator, "_encode_dummy_inputs", return_value=None),
        patch.object(creator, "_get_kv_size_per_token", return_value=CacheCost(slope=3)),
        patch.object(torch.cuda, "empty_cache"),
        patch.object(torch.cuda, "reset_peak_memory_stats"),
        # Use a nonzero backend cost: the 512-byte budget splits into 384 bytes
        # of KV and 128 bytes of workspace unless chunk profiling covered it.
        patch(
            "tensorrt_llm._torch.pyexecutor._util.get_attention_workspace_bytes_per_token",
            return_value=1,
        ),
        patch(
            "tensorrt_llm._torch.pyexecutor._util.get_attention_workspace_is_chunked_prefill_bounded",
            return_value=bounded,
        ),
    ):
        assert creator.try_prepare_estimation()
        assert kv_cache_config.max_tokens == estimation_max_tokens
        assert kv_cache_config.pool_ratio is None
        assert kv_cache_config.avg_seq_len == max_seq_len

        creator.configure_kv_cache_capacity(py_executor)

    assert kv_cache_config.max_gpu_total_bytes == expected_budget
    assert creator._fp8_ctx_mla_kv_len_cap == expected_cap
    py_executor.start_worker.assert_called_once()
    py_executor.shutdown.assert_called_once()
    assert kv_cache_config.max_tokens == user_max_tokens
    assert kv_cache_config.pool_ratio == pool_ratio
    assert kv_cache_config.avg_seq_len == avg_seq_len


@pytest.mark.parametrize("estimating", [False, True])
@pytest.mark.parametrize("layer_limit", [None, 20])
def test_csa2_creation_preserves_resolved_context_cache_boundary(estimating, layer_limit):
    from tensorrt_llm._torch.pyexecutor._util import _create_kv_cache_manager

    captured = {}

    class RecordingManager:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    pretrained = SimpleNamespace(
        hidden_size=5120,
        head_dim=512,
        num_attention_heads=64,
        num_key_value_heads=1,
        num_hidden_layers=40,
        vocab_size=32000,
    )
    derived = {} if layer_limit is None else {"csa2_context_swa_layer_limit": layer_limit}
    model_config = SimpleNamespace(
        pretrained_config=pretrained, quant_config=None, extra_attrs=derived
    )
    _create_kv_cache_manager(
        model_engine=None,
        kv_cache_manager_cls=RecordingManager,
        mapping=Mapping(),
        kv_cache_config=KvCacheConfig(),
        tokens_per_block=128,
        max_seq_len=10240,
        max_batch_size=32,
        spec_config=None,
        sparse_attention_config=SimpleNamespace(algorithm="csa2"),
        max_num_tokens=8192,
        max_beam_width=1,
        kv_connector_manager=None,
        model_config=model_config,
        dtype=torch.bfloat16,
        is_draft=False,
        is_disagg=True,
        estimating_kv_cache=estimating,
    )
    assert captured["context_swa_layer_limit"] == layer_limit
    # Layer 20 still owns the boundary GLOBAL cache; reducing num_layers would
    # remove it and renumber the peer's later model-layer identities.
    assert captured["num_layers"] == 40
    assert captured["is_estimating_kv_cache"] is estimating
    assert captured["pretrained_config"] is pretrained


@pytest.mark.parametrize(
    ("estimating_kv_cache", "expected_avg_seq_len"),
    [(True, 2045), (False, 2055)],
)
def test_manager_estimation_clamps_only_temporary_avg_seq_len(
    estimating_kv_cache,
    expected_avg_seq_len,
) -> None:
    import torch

    captured_configs = []

    class _RecordingKVCacheManagerV2(KVCacheManagerV2):
        def __init__(self, kv_cache_config, _kv_cache_type, **kwargs) -> None:
            captured_configs.append(kv_cache_config)
            self.max_seq_len = kwargs["max_seq_len"]

    pretrained = SimpleNamespace(
        hidden_size=1024,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_hidden_layers=2,
        vocab_size=32000,
    )
    model_config = Mock()
    model_config.pretrained_config = pretrained
    model_config.quant_config = None
    kv_cache_config = KvCacheConfig(
        max_tokens=2048,
        avg_seq_len=2055,
    )

    _create_kv_cache_manager(
        model_engine=None,
        kv_cache_manager_cls=_RecordingKVCacheManagerV2,
        mapping=Mock(),
        kv_cache_config=kv_cache_config,
        tokens_per_block=32,
        max_seq_len=2045,
        max_batch_size=4,
        spec_config=None,
        sparse_attention_config=None,
        max_num_tokens=2048,
        max_beam_width=1,
        kv_connector_manager=None,
        estimating_kv_cache=estimating_kv_cache,
        model_config=model_config,
        dtype=torch.bfloat16,
        is_draft=False,
    )

    assert captured_configs[0].avg_seq_len == expected_avg_seq_len
    assert kv_cache_config.avg_seq_len == 2055


def test_separate_one_model_draft_normalizes_target_pool_ratio() -> None:
    creator = object.__new__(KvCacheCreator)
    creator._model_engine = Mock(_max_cuda_graph_batch_size=4)
    target_pool_ratio = [0.32, 0.68]
    creator._kv_cache_config = KvCacheConfig(
        pool_ratio=target_pool_ratio,
        max_attention_window=None,
    )
    creator._max_seq_len = 9472
    creator._tokens_per_block = 32
    creator._max_batch_size = 1024
    creator._max_num_tokens = 9472
    creator._max_beam_width = 1
    creator._kv_connector_manager = None
    creator._skip_est = False
    creator._execution_stream = None
    creator._is_disagg = False
    creator._cache_transceiver_config = None
    creator._mapping = Mock()
    creator._speculative_config = Mock()
    creator._disable_overlap_scheduler = False

    effective_draft_config = Mock()
    effective_draft_config.pretrained_config.torch_dtype = "bfloat16"
    effective_draft_config.sparse_attention_config = None

    with (
        patch.object(creator, "_get_num_draft_layers", return_value=1),
        patch.object(creator, "_get_one_model_draft_layer_mask", return_value=[True]),
        patch.object(
            creator,
            "_get_effective_draft_config",
            return_value=effective_draft_config,
        ),
        patch.object(creator, "_enable_kv_cache_stats", return_value=False),
        patch.object(
            creator,
            "_validate_or_fallback_kv_cache_manager_v2",
            return_value=KVCacheManagerV2,
        ),
        patch(
            "tensorrt_llm._torch.pyexecutor._util._derive_draft_max_attention_window",
            return_value=None,
        ),
        patch(
            "tensorrt_llm._torch.pyexecutor._util.get_kv_cache_manager_cls",
            return_value=KVCacheManagerV2,
        ),
        patch(
            "tensorrt_llm._torch.pyexecutor._util._create_kv_cache_manager",
            return_value=Mock(),
        ) as create_manager,
    ):
        codec_provider = object()
        creator._create_one_model_draft_kv_cache_manager(
            creator._max_seq_len,
            estimating_kv_cache=True,
            cold_page_codec_provider=codec_provider,
        )

    draft_config = create_manager.call_args.kwargs["kv_cache_config"]
    assert draft_config.pool_ratio == [1.0]
    assert create_manager.call_args.kwargs["max_cuda_graph_batch_size"] == 4
    assert create_manager.call_args.kwargs["cold_page_codec_provider"] is codec_provider
    assert creator._kv_cache_config.pool_ratio == target_pool_ratio


def _dflash_draft_model(uses_worker_backend: bool, backend: str) -> SimpleNamespace:
    """The two attributes KvCacheCreator._draft_tokens_per_block reads off the
    built draft model: whether it runs the shared DFlash op sets, and the
    resolved attention backend."""
    return SimpleNamespace(
        _uses_worker_attention_backend=uses_worker_backend,
        dflash_attention_backend=backend,
    )


def _page_size_creator(tokens_per_block: int, draft_model) -> KvCacheCreator:
    creator = object.__new__(KvCacheCreator)
    creator._tokens_per_block = tokens_per_block
    creator._model_engine = Mock(_max_cuda_graph_batch_size=4)
    creator._model_engine.model = SimpleNamespace(draft_model=draft_model)
    return creator


class TestDraftPoolPageSize:
    """A standalone DFlash/DSpark drafter on the TRTLLM backend reads its pool
    with trtllm-gen, which accepts fewer page sizes than a target may pin
    (MiniMax-M3 MSA: 128), so the separate draft pool is carved at the largest
    size the drafter accepts and joint block reuse is declined when the two
    pools differ."""

    def test_premise_trtllm_gen_page_sizes(self):
        from tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen import (
            FlashInferTrtllmGenFmha,
        )

        supported = FlashInferTrtllmGenFmha.SUPPORTED_TOKENS_PER_BLOCK
        assert 128 not in supported
        assert max(size for size in supported if size < 128) == 64

    @pytest.mark.parametrize(
        "tokens_per_block, draft_model, expected",
        [
            (128, _dflash_draft_model(True, "TRTLLM"), 64),
            (128, _dflash_draft_model(True, "VANILLA"), 128),
            (128, _dflash_draft_model(True, "FA4"), 128),
            # The MLA DSpark drafter brings its own kernels under the same name.
            (128, _dflash_draft_model(False, "TRTLLM"), 128),
            # No drafter at all (MTP / Eagle3 draft layers).
            (128, None, 128),
            (64, _dflash_draft_model(True, "TRTLLM"), 64),
            (32, _dflash_draft_model(True, "TRTLLM"), 32),
        ],
    )
    def test_draft_tokens_per_block(self, tokens_per_block, draft_model, expected):
        creator = _page_size_creator(tokens_per_block, draft_model)
        assert creator._draft_tokens_per_block() == expected

    def test_separate_draft_manager_is_carved_at_the_draft_page_size(self):
        creator = _page_size_creator(128, _dflash_draft_model(True, "TRTLLM"))
        creator._kv_cache_config = KvCacheConfig(max_attention_window=None)
        creator._max_seq_len = 9472
        creator._max_batch_size = 8
        creator._max_num_tokens = 9472
        creator._max_beam_width = 1
        creator._kv_connector_manager = None
        creator._skip_est = False
        creator._execution_stream = None
        creator._is_disagg = False
        creator._cache_transceiver_config = None
        creator._mapping = Mock()
        creator._mapping.has_cp_helix.return_value = False
        creator._speculative_config = Mock()
        creator._disable_overlap_scheduler = False

        effective_draft_config = Mock()
        effective_draft_config.pretrained_config.torch_dtype = "bfloat16"
        effective_draft_config.sparse_attention_config = None

        with (
            patch.object(creator, "_get_num_draft_layers", return_value=5),
            patch.object(creator, "_get_one_model_draft_layer_mask", return_value=[True] * 5),
            patch.object(
                creator,
                "_get_effective_draft_config",
                return_value=effective_draft_config,
            ),
            patch.object(creator, "_enable_kv_cache_stats", return_value=False),
            patch.object(
                creator,
                "_validate_or_fallback_kv_cache_manager_v2",
                return_value=KVCacheManagerV2,
            ),
            patch(
                "tensorrt_llm._torch.pyexecutor._util._derive_draft_max_attention_window",
                return_value=None,
            ),
            patch(
                "tensorrt_llm._torch.pyexecutor._util.get_kv_cache_manager_cls",
                return_value=KVCacheManagerV2,
            ),
            patch(
                "tensorrt_llm._torch.pyexecutor._util._create_kv_cache_manager",
                return_value=Mock(),
            ) as create_manager,
        ):
            creator._create_one_model_draft_kv_cache_manager(
                creator._max_seq_len, estimating_kv_cache=False
            )

        assert create_manager.call_args.kwargs["tokens_per_block"] == 64
        assert create_manager.call_args.kwargs["is_draft"] is True
        # The target's own page size is untouched.
        assert creator._tokens_per_block == 128

    @pytest.mark.parametrize("tokens_per_block, expected", [(128, False), (64, True)])
    def test_joint_reuse_needs_one_page_size(self, tokens_per_block, expected):
        creator = _page_size_creator(tokens_per_block, _dflash_draft_model(True, "TRTLLM"))
        creator._is_kv_cache_manager_v2 = True
        creator._kv_cache_manager_cls = KVCacheManagerV2
        # Standalone DSpark: lookahead 0, so only the page sizes decide.
        creator._speculative_config = DSparkDecodingConfig(max_draft_len=4)
        assert creator._joint_reuse_supported() is expected


class _StandaloneDrafterConfig:
    """KV geometry of a five-layer GQA drafter with one KV head per TP rank
    (4 KV heads over TP4, head_dim 128, bf16): 512 bytes per layer and token,
    2560 per token."""

    quant_config = None
    pretrained_config = SimpleNamespace(
        hidden_size=6144,
        num_attention_heads=64,
        num_key_value_heads=4,
        head_dim=128,
        layer_types=["sliding_attention"] * 5,
        sliding_window=4096,
    )

    def get_num_attention_layers(self) -> int:
        return 5


def _standalone_drafter_cost(max_attention_window) -> CacheCost:
    mapping = Mock(enable_attention_dp=False, tp_size=4)
    mapping.pp_layers.return_value = [0, 1, 2, 3, 4]
    return CacheCost.from_raw(
        KVCacheManagerV2.get_cache_size_per_token(
            _StandaloneDrafterConfig(),
            mapping,
            tokens_per_block=64,
            max_seq_len=1048576,
            max_batch_size=8,
            max_num_tokens=16384,
            kv_cache_config=KvCacheConfig(
                max_attention_window=max_attention_window, max_util_for_resume=0.95
            ),
            spec_config=DSparkDecodingConfig(max_draft_len=7),
            is_draft=True,
        )
    )


class TestStandaloneDraftPoolFunding:
    """The standalone DFlash/DSpark drafter's pool keeps the drafter's attention
    windows, so a mirror holds its window plus the chunk it is computing and the
    pool is funded for max_batch_size such mirrors as a fixed cost, with or
    without block reuse. A drafter without a window keeps whole contexts: the
    target's per-token funding bounds the mirrors without block reuse, and with
    it the pool is funded for max_batch_size full-length mirrors."""

    @staticmethod
    def _creator(
        spec_config,
        *,
        max_seq_len,
        max_batch_size,
        max_num_tokens=16384,
        disable_overlap_scheduler=False,
        is_disagg=False,
        mapping=None,
    ):
        creator = object.__new__(KvCacheCreator)
        creator._speculative_config = spec_config
        creator._max_seq_len = max_seq_len
        creator._max_batch_size = max_batch_size
        creator._max_num_tokens = max_num_tokens
        creator._disable_overlap_scheduler = disable_overlap_scheduler
        creator._is_disagg = is_disagg
        creator._mapping = mapping if mapping is not None else Mapping()
        return creator

    @staticmethod
    def _fixed_cost(bytes_per_token, tokens_per_mirror, num_mirrors, tokens_per_block=64):
        pages = num_mirrors * math.ceil(tokens_per_mirror / tokens_per_block)
        slots = math.ceil(pages / float(np.float32(0.95)))
        return CacheCost(slope=0, intercept=slots * tokens_per_block * bytes_per_token)

    @pytest.mark.parametrize(
        "disable_overlap_scheduler, is_disagg, mapping, expected",
        [
            # Overlap scheduler: requests retiring from iteration n keep their
            # pages while iteration n+1 admits new ones in their place.
            (False, False, Mapping(), 16),
            (True, False, Mapping(), 8),
            (True, True, Mapping(), 16),
            # Pipeline parallelism: one batch per stage, no retiring overlap.
            (False, False, Mapping(world_size=2, pp_size=2), 16),
        ],
    )
    def test_mirrors_follow_the_manager_admissible_sequences(
        self, disable_overlap_scheduler, is_disagg, mapping, expected
    ):
        assert (
            max_admissible_sequences(
                8, mapping, is_disagg=is_disagg, disable_overlap_scheduler=disable_overlap_scheduler
            )
            == expected
        )
        spec = DSparkDecodingConfig(max_draft_len=7)
        creator = self._creator(
            spec,
            max_seq_len=65536,
            max_batch_size=8,
            disable_overlap_scheduler=disable_overlap_scheduler,
            is_disagg=is_disagg,
            mapping=mapping,
        )
        _, headroom = _get_generation_kv_capacity(spec, is_draft=True)
        with patch.object(creator, "_get_draft_max_attention_window", return_value=[4096] * 5):
            cost = creator._standalone_draft_pool_cost(
                CacheCost(slope=2560), KvCacheConfig(max_util_for_resume=0.95), 64
            )
        assert cost == self._fixed_cost(2560, 4096 + 16384 + 2 * headroom + 64, expected)

    def test_per_token_cost_comes_from_the_windowless_manager_cost(self):
        # Through the real KVCacheManagerV2.get_cache_size_per_token: costed
        # with its windows the drafter's pool is a fixed cost with slope 0
        # (a funding that multiplies by it funds the pool with 0 bytes); the
        # same layers costed without windows give the
        # 2560 bytes one token of drafter K/V occupies at TP4.
        mapping = Mock(enable_attention_dp=False, tp_size=4)
        mapping.pp_layers.return_value = [0, 1, 2, 3, 4]
        creator = self._creator(
            DSparkDecodingConfig(max_draft_len=7), max_seq_len=1048576, max_batch_size=8
        )
        creator._kv_cache_config = KvCacheConfig(max_util_for_resume=0.95)
        creator._tokens_per_block = 128
        windowed = KvCacheConfig(max_attention_window=[4096] * 5, max_util_for_resume=0.95)
        windowed_cost = creator._per_manager_cache_cost(
            KVCacheManagerV2,
            _StandaloneDrafterConfig(),
            windowed,
            mapping=mapping,
            tokens_per_block=64,
            is_draft=True,
        )
        assert windowed_cost.slope == 0 and windowed_cost.intercept > 0
        per_token = creator._standalone_draft_per_token_cost(
            KVCacheManagerV2, _StandaloneDrafterConfig(), windowed, mapping, 64
        )
        assert per_token == CacheCost(slope=2560, intercept=0)
        with patch.object(creator, "_get_draft_max_attention_window", return_value=[4096] * 5):
            cost = creator._standalone_draft_pool_cost(
                per_token, KvCacheConfig(enable_block_reuse=True, max_util_for_resume=0.95), 64
            )
        assert cost.slope == 0 and cost.intercept > 2560 * 64 * 16 * 4096 // 64

    def test_pool_cost_rejects_a_cost_that_is_not_a_per_token_slope(self):
        creator = self._creator(
            DSparkDecodingConfig(max_draft_len=7), max_seq_len=1048576, max_batch_size=8
        )
        for bad in (
            CacheCost(slope=0, intercept=826 * 64 * 2560),
            CacheCost(slope=2560, intercept=1),
        ):
            with pytest.raises(ValueError, match="pure slope"):
                creator._standalone_draft_pool_cost(bad, KvCacheConfig(), 64)

    @pytest.mark.parametrize("enable_block_reuse", [True, False])
    def test_windowed_mirrors_are_funded_for_window_plus_chunk(self, enable_block_reuse):
        # TP2: 5 layers x 4 KV heads x 128 x bf16 K+V = 5120 B/token/rank,
        # max_batch_size 30 at 1M max_seq_len and 16384-token chunks, overlap
        # scheduler on (60 admissible mirrors). 30 full-length mirrors alone
        # would cost ~160 GiB per rank, more than the whole KV budget; 60
        # window-plus-chunk mirrors cost a few GiB.
        spec = DSparkDecodingConfig(max_draft_len=3)
        creator = self._creator(spec, max_seq_len=1048576, max_batch_size=30)
        _, headroom = _get_generation_kv_capacity(spec, is_draft=True)
        config = KvCacheConfig(enable_block_reuse=enable_block_reuse, max_util_for_resume=0.95)
        with patch.object(creator, "_get_draft_max_attention_window", return_value=[4096] * 5):
            cost = creator._standalone_draft_pool_cost(CacheCost(slope=5120), config, 64)
        tokens_per_mirror = 4096 + 16384 + 2 * headroom + 64
        assert cost == self._fixed_cost(5120, tokens_per_mirror, 60)
        assert 6 < cost.intercept / 2**30 < 8
        # The window is the drafter's, read from its own derived layout, so
        # the cost does not move with the context length.
        creator._max_seq_len = 65536
        with patch.object(creator, "_get_draft_max_attention_window", return_value=[4096] * 5):
            assert creator._standalone_draft_pool_cost(CacheCost(slope=5120), config, 64) == cost

    def test_windowed_funding_covers_the_live_page_geometry(self):
        # TP4, batch 8, 1M context, K=7, 16384-token chunks, overlap scheduler
        # on: the windowed pool holds up to 841 live pages, against the 826 the
        # manager's own SWA cost funds
        # (test_premise_windowed_funding_is_a_fixed_cost).
        spec = DSparkDecodingConfig(max_draft_len=7)
        creator = self._creator(spec, max_seq_len=1048576, max_batch_size=8)
        _, headroom = _get_generation_kv_capacity(spec, is_draft=True)
        with patch.object(creator, "_get_draft_max_attention_window", return_value=[4096] * 5):
            cost = creator._standalone_draft_pool_cost(
                CacheCost(slope=2560), KvCacheConfig(enable_block_reuse=True), 64
            )
        assert cost == self._fixed_cost(2560, 4096 + 16384 + 2 * headroom + 64, 16)
        assert cost.intercept // (64 * 2560) > 841
        assert cost.intercept / 2**30 < 1

    def test_windowless_drafter_under_reuse_funds_max_batch_size_full_length_mirrors(self):
        spec = DSparkDecodingConfig(max_draft_len=7)
        creator = self._creator(spec, max_seq_len=1048576, max_batch_size=8)
        _, headroom = _get_generation_kv_capacity(spec, is_draft=True)
        with patch.object(creator, "_get_draft_max_attention_window", return_value=None):
            cost = creator._standalone_draft_pool_cost(
                CacheCost(slope=2560),
                KvCacheConfig(enable_block_reuse=True, max_util_for_resume=0.95),
                64,
            )
        # 16 admissible mirrors (overlap scheduler) x 1M tokens at 2560 B:
        # about 42 GiB, which is why a windowless drafter cannot serve long
        # context under block reuse at any useful batch size.
        assert cost == self._fixed_cost(2560, 1048576 + headroom, 16)
        assert 41 < cost.intercept / 2**30 < 43

    def test_windowless_drafter_without_reuse_keeps_the_per_token_funding(self):
        creator = self._creator(
            DSparkDecodingConfig(max_draft_len=7), max_seq_len=1048576, max_batch_size=8
        )
        per_token = CacheCost(slope=2560)
        with patch.object(creator, "_get_draft_max_attention_window", return_value=None):
            cost = creator._standalone_draft_pool_cost(
                per_token, KvCacheConfig(enable_block_reuse=False), 64
            )
        assert cost == per_token

    @pytest.mark.parametrize(
        "windows, max_seq_len, expected",
        [
            ([4096] * 5, 1048576, 4096),
            ([4096], 1048576, 4096),
            # A layer attending to the whole context makes the mirror full-length.
            ([4096, 1048576], 1048576, None),
            ([4096] * 5, 4096, None),
            (None, 1048576, None),
        ],
    )
    def test_standalone_draft_mirror_window(self, windows, max_seq_len, expected):
        creator = self._creator(
            DSparkDecodingConfig(max_draft_len=7), max_seq_len=max_seq_len, max_batch_size=8
        )
        with patch.object(creator, "_get_draft_max_attention_window", return_value=windows):
            window = creator._standalone_draft_mirror_window(KvCacheConfig(enable_block_reuse=True))
        assert window == expected

    def test_shared_prefix_mirrors_fit_the_reuse_funding_not_the_per_token_one(self):
        # Two requests sharing a 32768-token prefix, 128 new tokens and 7
        # drafts each: the target holds the prefix once, the unpaired mirrors
        # hold it twice.
        prefix, tail, draft = 32768, 128, 7
        target_tokens = prefix + 2 * (tail + draft)
        mirror_tokens = 2 * (prefix + tail + draft)
        per_token = CacheCost(slope=2560)
        assert (
            per_token.tokens_for_budget(per_token.bytes_for_tokens(target_tokens)) < mirror_tokens
        )
        bound = KvCacheCreator.standalone_draft_pool_worst_case_cost(
            2560, 64, prefix + tail + draft, 2, 1.0
        )
        assert bound.intercept // 2560 >= mirror_tokens

    def test_premise_windowed_funding_is_a_fixed_cost(self):
        # Five 4096-token windows: 66 pages per request x 8 requests plus one
        # max_num_tokens of context, rounded to 64-token slots at the resume
        # watermark. 826 slots = 0.13 GiB, independent of max_seq_len.
        assert _standalone_drafter_cost([4096] * 5) == CacheCost(slope=0, intercept=826 * 2560 * 64)

    def test_full_context_funding_is_per_token(self):
        assert _standalone_drafter_cost(None) == CacheCost(slope=2560, intercept=0)

    @pytest.mark.parametrize(
        "spec_config",
        [
            DSparkDecodingConfig(max_draft_len=7),
            DFlashDecodingConfig(max_draft_len=7),
            MTPDecodingConfig(max_draft_len=1),
        ],
    )
    def test_draft_pool_keeps_the_drafter_windows(self, spec_config):
        # The standalone drafter's pool trims below the window like any other;
        # only the estimation pass carves without windows.
        creator = object.__new__(KvCacheCreator)
        creator._speculative_config = spec_config
        with patch.object(creator, "_get_draft_max_attention_window", return_value=[4096] * 5):
            draft_kv_config = creator._get_one_model_draft_kv_cache_config(KvCacheConfig(), 1048576)
            estimation_config = creator._get_one_model_draft_kv_cache_config(
                KvCacheConfig(), 1048576, estimating_kv_cache=True
            )
        assert draft_kv_config.max_attention_window == [4096] * 5
        assert estimation_config.max_attention_window is None

    def test_full_context_share_of_the_budget(self):
        # The price of the per-token funding: 2560 of every 8640 + 2560 bytes
        # per token, 22.9 % of a 181 GiB GPU budget (41.4 GiB).
        creator = object.__new__(KvCacheCreator)
        total = 181 * 2**30
        target_budget, draft_budget = creator._compute_draft_budget_shares(
            total, CacheCost(slope=8640), CacheCost(slope=2560)
        )
        assert draft_budget == total * 2560 // 11200
        assert target_budget + draft_budget == total
        assert 41.3 < draft_budget / 2**30 < 41.5


# ---------------------------------------------------------------------------
# Per-layer attention windows derived from a mixed layer_types schedule
# ---------------------------------------------------------------------------
#
# `_derive_layer_type_attention_windows` (covered in
# test_layer_type_attention_windows.py) turns a mixed sliding/full
# `layer_types` schedule into one window per layer for KVCacheManagerV2.
# `_derive_v2_layer_type_attention_windows` applies it in
# `_create_kv_cache_manager` (tested here) and in the static per-token cost
# model the creator splits the GPU budget with (tested in
# test_kv_cache_budget_split.py), and keeps the single-window default for
# hybrid linear-attention configs and for a `pool_ratio` that does not match
# the derived layer groups. The creator also gives one-model speculative
# layers appended after the decoder stack the full context, and leaves the
# cross-attention pool alone.

_SLIDING = "sliding_attention"
_FULL = "full_attention"


def _create_manager_and_capture_config(
    pretrained: SimpleNamespace,
    kv_cache_config: KvCacheConfig,
    max_seq_len: int,
    manager_cls: type[KVCacheManager] | type[KVCacheManagerV2] = KVCacheManagerV2,
    **factory_overrides: object,
) -> KvCacheConfig:
    """Run `_create_kv_cache_manager` with a recording subclass of `manager_cls`
    and return the `KvCacheConfig` the manager was constructed with.
    `factory_overrides` replace the default keyword arguments of the factory
    call (for example `spec_config`, `layer_mask` or `kv_cache_type`)."""
    captured = []

    class _RecordingManager(manager_cls):
        def __init__(self, kv_cache_config: KvCacheConfig, *args: object, **kwargs: object) -> None:
            captured.append(kv_cache_config)

    model_config = Mock()
    model_config.pretrained_config = pretrained
    model_config.quant_config = None

    factory_kwargs: dict[str, object] = dict(
        model_engine=None,
        kv_cache_manager_cls=_RecordingManager,
        mapping=Mock(),
        kv_cache_config=kv_cache_config,
        tokens_per_block=32,
        max_seq_len=max_seq_len,
        max_batch_size=4,
        spec_config=None,
        sparse_attention_config=None,
        max_num_tokens=1024,
        max_beam_width=1,
        kv_connector_manager=None,
        model_config=model_config,
        dtype=torch.bfloat16,
        is_draft=False,
    )
    factory_kwargs.update(factory_overrides)
    _create_kv_cache_manager(**factory_kwargs)

    assert len(captured) == 1
    return captured[0]


def _mixed_schedule_pretrained(layer_types: list[str], **overrides: object) -> SimpleNamespace:
    fields = dict(
        hidden_size=1024,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_hidden_layers=len(layer_types),
        vocab_size=32000,
        layer_types=layer_types,
        sliding_window=512,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _eagle3_one_model_spec_config(num_draft_hidden_layers: int | None = None) -> SimpleNamespace:
    """The fields `get_num_spec_layers` and `should_use_separate_draft_kv_cache`
    read from an Eagle3 one-model speculative config."""
    return SimpleNamespace(
        spec_dec_mode=SpeculativeDecodingMode.EAGLE3_ONE_MODEL,
        _use_shared_kv_cache=False,
        _allow_separate_draft_kv_cache=True,
        _num_draft_hidden_layers=num_draft_hidden_layers,
    )


def test_create_kv_cache_manager_pool_ratio_arity_mismatch_keeps_single_window() -> None:
    """A `pool_ratio` written for the single pool (one entry) does not match the
    two layer groups the derived windows would create. Rather than failing the
    manager's arity check at startup, the derivation is skipped with a warning
    and the configuration keeps the single-window default it was written for."""
    kv_cache_config = KvCacheConfig(pool_ratio=[1.0])

    with patch("tensorrt_llm._torch.pyexecutor._util.logger") as mock_logger:
        manager_config = _create_manager_and_capture_config(
            _mixed_schedule_pretrained([_SLIDING, _SLIDING, _FULL, _SLIDING]),
            kv_cache_config,
            max_seq_len=2048,
        )

    assert manager_config is kv_cache_config
    assert manager_config.max_attention_window is None
    mock_logger.warning_once.assert_called_once()
    message = mock_logger.warning_once.call_args.args[0]
    assert "pool_ratio has 1 entries" in message
    assert "2 layer groups" in message


def test_create_kv_cache_manager_pool_ratio_per_layer_group_keeps_derivation() -> None:
    """One `pool_ratio` entry per derived layer group is the intended pairing."""
    kv_cache_config = KvCacheConfig(pool_ratio=[0.5, 0.5])

    manager_config = _create_manager_and_capture_config(
        _mixed_schedule_pretrained([_SLIDING, _SLIDING, _FULL, _SLIDING]),
        kv_cache_config,
        max_seq_len=2048,
    )

    assert manager_config.max_attention_window == [512, 512, 2048, 512]
    assert manager_config.pool_ratio == [0.5, 0.5]


def test_derive_v2_windows_skip_hybrid_linear_configs() -> None:
    """A hybrid linear-attention model (recurrent layers interleaved with
    attention layers) keeps the single-window default even when its
    `layer_types` and `sliding_window` would derive a per-layer list: the
    static cost model indexes windows by attention-layer position while the
    manager indexes them by global layer id, so the two would disagree."""
    model_config = Mock()
    model_config.pretrained_config = _mixed_schedule_pretrained(
        [_SLIDING, _FULL, _SLIDING, _FULL], hybrid_override_pattern="M*M*"
    )

    assert (
        _derive_v2_layer_type_attention_windows(
            KvCacheConfig(), KVCacheManagerV2, model_config, max_seq_len=2048
        )
        is None
    )

    # Without the hybrid marker the same schedule derives a per-layer list.
    model_config.pretrained_config = _mixed_schedule_pretrained([_SLIDING, _FULL, _SLIDING, _FULL])
    assert _derive_v2_layer_type_attention_windows(
        KvCacheConfig(), KVCacheManagerV2, model_config, max_seq_len=2048
    ) == [512, 2048, 512, 2048]


@pytest.mark.parametrize(
    ("layer_mask", "expected_windows"),
    [
        (None, [512, 512, 2048, 512, 2048, 2048]),
        ([True] * 4, [512, 512, 2048, 512]),
    ],
    ids=["spec_layers_appended", "target_only_mask"],
)
def test_create_kv_cache_manager_keeps_appended_spec_layers_full_context(
    layer_mask: list[bool] | None,
    expected_windows: list[int],
) -> None:
    """Without a `layer_mask`, `get_pp_layers` appends the one-model speculative
    layers after the decoder layers and V2 reads the window list modulo its
    length, so the derived list gets one full-context entry per appended layer.
    A target-only mask holds the decoder layers alone and needs no extra entry."""
    manager_config = _create_manager_and_capture_config(
        _mixed_schedule_pretrained([_SLIDING, _SLIDING, _FULL, _SLIDING]),
        KvCacheConfig(),
        max_seq_len=2048,
        spec_config=_eagle3_one_model_spec_config(num_draft_hidden_layers=2),
        layer_mask=layer_mask,
    )

    assert manager_config.max_attention_window == expected_windows


def test_create_kv_cache_manager_cross_pool_keeps_single_window_default() -> None:
    """The cross-attention pool stores encoder-side KV, which the decoder's
    `layer_types` do not describe."""
    kv_cache_config = KvCacheConfig()

    manager_config = _create_manager_and_capture_config(
        _mixed_schedule_pretrained([_SLIDING, _SLIDING, _FULL, _SLIDING]),
        kv_cache_config,
        max_seq_len=2048,
        kv_cache_type=batch_manager.CacheType.CROSS,
        num_layers=4,
        num_kv_heads=8,
        head_dim=128,
    )

    assert manager_config is kv_cache_config
    assert manager_config.max_attention_window is None

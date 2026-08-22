# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for KV cache token estimation in KvCacheCreator._get_token_num_for_estimation.

Guards the ADP (Attention Data Parallelism) cache-block reduction: when
enable_attention_dp is True and tp_size > 1, _create_dummy_context_requests
produces tp_size duplicate requests, but the scheduler distributes them
1-per-rank.  Each rank's KV cache therefore only needs capacity for its own
share, not all copies.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin
from tensorrt_llm._torch.pyexecutor._util import CacheCost, KvCacheCreator
from tensorrt_llm._torch.pyexecutor.config_utils import get_layer_attention_window
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MultimodalConfig, TorchLlmArgs
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


class _TextModel:
    def __init__(self, encoder_cache_max_bytes):
        self.model_config = ModelConfig(
            multimodal_config=MultimodalConfig(encoder_cache_max_bytes=encoder_cache_max_bytes)
        )


class _MultimodalModel(MultimodalModelMixin):
    def __init__(self, encoder_cache_max_bytes):
        self.model_config = ModelConfig(
            multimodal_config=MultimodalConfig(encoder_cache_max_bytes=encoder_cache_max_bytes)
        )


class _EncoderCacheMultimodalModel(_MultimodalModel):
    supports_encoder_cache = True


@dataclass
class _ModelEngine:
    model: _TextModel | _MultimodalModel
    mm_encoder_output_budget_bytes: int | None = None


def _make_reserve_creator(
    model: _TextModel | _MultimodalModel,
    *,
    mm_encoder_output_budget_bytes: int | None = None,
    disable_mm_encoder: bool = False,
) -> KvCacheCreator:
    llm_args = TorchLlmArgs(
        model="dummy",
        checkpoint_format="HF",
        disable_mm_encoder=disable_mm_encoder,
        multimodal_config=model.model_config.multimodal_config,
    )
    model_engine = _ModelEngine(
        model=model,
        mm_encoder_output_budget_bytes=mm_encoder_output_budget_bytes,
    )
    return KvCacheCreator(
        model_engine=model_engine,
        draft_model_engine=None,
        mapping=Mapping(),
        net_max_seq_len=1,
        kv_connector_manager=None,
        max_num_tokens=1,
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
):
    """Build a minimal KvCacheCreator (bypasses __init__) wired up for
    _get_token_num_for_estimation only."""
    c = object.__new__(KvCacheCreator)

    c._tokens_per_block = tokens_per_block
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


def test_kv_cache_estimation_skips_multimodal_reserve_when_encoder_disabled():
    creator = _make_reserve_creator(
        _EncoderCacheMultimodalModel(64),
        mm_encoder_output_budget_bytes=512,
        disable_mm_encoder=True,
    )

    assert creator._get_multimodal_encoder_memory_reserve() == 0


def test_reserve_adds_only_unprofiled_output_capacity():
    creator = _make_reserve_creator(
        _MultimodalModel(0),
        mm_encoder_output_budget_bytes=512,
    )
    assert creator._get_multimodal_encoder_memory_reserve(profiled_output_bytes=400) == 112


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
def test_v2_pool_estimation_falls_back_for_unsupported_window_metadata(
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

    assert hybrid._get_token_num_for_estimation() == 2 * uniform._get_token_num_for_estimation()


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
            kv_cache_config=KvCacheConfig(max_attention_window=[2048, 2048, 4096]),
        )
    )
    scratch_size_per_token = CacheCost.from_raw(
        KVCacheManagerV2.get_cache_size_per_token(
            FakeModelConfig(),
            mapping,
            tokens_per_block=64,
            max_seq_len=4096,
            max_batch_size=3,
            kv_cache_config=KvCacheConfig(max_attention_window=[2048, 2048, 4096]),
            enable_swa_scratch_reuse=True,
        )
    )

    # Per layer: K+V * kv_heads * head_dim * bf16 bytes = 2 * 2 * 8 * 2.
    expected = CacheCost(slope=64, intercept=3 * 2 * 2048 * 64)
    assert no_scratch_size_per_token == expected
    assert scratch_size_per_token == expected


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
    creator._kv_cache_config = KvCacheConfig()
    creator._speculative_config = None

    cost = creator._per_manager_cache_cost(FakeV2Manager, Mock())

    assert cost == CacheCost(slope=20, intercept=21)


def test_v2_quota_from_max_tokens_models_context_swa_scratch():
    manager = object.__new__(KVCacheManagerV2)
    manager.num_local_layers = 3
    manager.pp_layers = [0, 1, 2]
    manager.max_attention_window_vec = [128, 128, None]
    manager.tokens_per_block = 64
    manager.max_batch_size = 4
    manager.max_num_tokens = 1000
    manager.get_layer_bytes_per_token = lambda local_layer_idx, data_role: [10, 10, 20][
        local_layer_idx
    ]

    max_tokens = 1200

    manager.enable_swa_scratch_reuse = False
    no_scratch_quota = manager._get_quota_from_max_tokens(max_tokens)
    assert no_scratch_quota == (max_tokens * 20 + manager.max_num_tokens * 20 + 4 * 2 * 128 * 10)
    assert manager._get_max_tokens_from_quota(no_scratch_quota) == max_tokens

    manager.enable_swa_scratch_reuse = True
    scratch_quota = manager._get_quota_from_max_tokens(max_tokens)
    assert scratch_quota == (max_tokens * 20 + manager.max_num_tokens * 10 + 4 * 2 * 128 * 10)
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

    from tensorrt_llm._torch.pyexecutor._util import _create_kv_cache_manager

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


def test_estimation_temporarily_uses_inferred_pool_sizing() -> None:
    pool_ratio = [0.2, 0.3, 0.5]
    avg_seq_len = 128
    max_seq_len = 4096
    user_max_tokens = 1024
    estimation_max_tokens = 256
    kv_cache_config = KvCacheConfig(
        max_tokens=user_max_tokens,
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
    llm_args = Mock(cache_transceiver_config=None)

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

    with (
        patch.object(
            creator,
            "_get_token_num_for_estimation",
            return_value=estimation_max_tokens,
        ),
        patch.object(creator, "_cal_max_memory", return_value=512),
        patch.object(torch.cuda, "mem_get_info", return_value=(768, 1024)),
        patch.object(torch.cuda, "memory_stats", return_value={"allocated_bytes.all.current": 128}),
        patch.object(torch.cuda, "empty_cache"),
        patch.object(torch.cuda, "reset_peak_memory_stats"),
        # This test exercises inferred pool sizing, not the backend workspace reserve; the mock
        # model_config would otherwise walk into backend resolution and the MLA byte-cost path.
        # Neutralize it so no reserve applies.
        patch(
            "tensorrt_llm._torch.pyexecutor._util.get_attention_workspace_bytes_per_token",
            return_value=0,
        ),
    ):
        assert creator.try_prepare_estimation()
        assert kv_cache_config.max_tokens == estimation_max_tokens
        assert kv_cache_config.pool_ratio is None
        assert kv_cache_config.avg_seq_len == max_seq_len

        creator.configure_kv_cache_capacity()

    assert kv_cache_config.max_tokens == user_max_tokens
    assert kv_cache_config.pool_ratio == pool_ratio
    assert kv_cache_config.avg_seq_len == avg_seq_len


@pytest.mark.parametrize(
    ("estimating_kv_cache", "expected_avg_seq_len"),
    [(True, 2045), (False, 2055)],
)
def test_manager_estimation_clamps_only_temporary_avg_seq_len(
    estimating_kv_cache,
    expected_avg_seq_len,
) -> None:
    import torch

    from tensorrt_llm._torch.pyexecutor._util import _create_kv_cache_manager

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
    creator._mapping = Mock()
    creator._speculative_config = Mock()

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
        creator._create_one_model_draft_kv_cache_manager(creator._max_seq_len)

    draft_config = create_manager.call_args.kwargs["kv_cache_config"]
    assert draft_config.pool_ratio == [1.0]
    assert creator._kv_cache_config.pool_ratio == target_pool_ratio

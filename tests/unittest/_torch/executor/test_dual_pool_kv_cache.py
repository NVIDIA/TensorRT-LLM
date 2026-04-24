# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for dual-pool KVCacheManagerV2 construction (enc-dec Step 4).

Validates budget splitting, ResourceManagerType.CROSS_KV_CACHE_MANAGER
registration, and the cross pool wiring through KVCacheV2Scheduler.
"""

from unittest.mock import Mock, patch  # noqa: I001

from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_kv_cache_config(
    cross_kv_cache_fraction=None, max_gpu_total_bytes=None, use_kv_cache_manager_v2=True
):
    """Create a mock KvCacheConfig with the fields KvCacheCreator needs."""
    config = Mock()
    config.cross_kv_cache_fraction = cross_kv_cache_fraction
    config.max_gpu_total_bytes = max_gpu_total_bytes
    config.use_kv_cache_manager_v2 = use_kv_cache_manager_v2
    config.max_tokens = None
    config.max_attention_window = None
    config.event_buffer_max_size = 0

    def model_copy():
        c = Mock()
        c.cross_kv_cache_fraction = config.cross_kv_cache_fraction
        c.max_gpu_total_bytes = config.max_gpu_total_bytes
        c.use_kv_cache_manager_v2 = config.use_kv_cache_manager_v2
        c.max_tokens = config.max_tokens
        c.max_attention_window = config.max_attention_window
        c.event_buffer_max_size = config.event_buffer_max_size
        return c

    config.model_copy = model_copy
    return config


def _make_mock_model_config(
    is_encoder_decoder=False,
    is_generation=True,
    **pretrained_overrides,
):
    """Minimal mock ModelConfig for KvCacheCreator."""
    model_config = Mock()
    model_config.is_encoder_decoder = is_encoder_decoder
    model_config.is_generation = is_generation
    model_config.sparse_attention_config = None

    pretrained = Mock()
    pretrained.num_hidden_layers = 6
    pretrained.num_attention_heads = 8
    pretrained.num_key_value_heads = 8
    pretrained.hidden_size = 512
    pretrained.head_dim = 64
    pretrained.vocab_size = 32000
    pretrained.quantization = Mock()
    pretrained.quantization.quant_algo = None
    pretrained.quantization.kv_cache_quant_algo = None
    for key, value in pretrained_overrides.items():
        setattr(pretrained, key, value)
    if "encoder_attention_heads" not in pretrained_overrides:
        pretrained.encoder_attention_heads = pretrained.num_attention_heads
    if "decoder_attention_heads" not in pretrained_overrides:
        pretrained.decoder_attention_heads = pretrained.num_attention_heads
    if "encoder_layers" not in pretrained_overrides:
        pretrained.encoder_layers = pretrained.num_hidden_layers
    if "decoder_layers" not in pretrained_overrides:
        pretrained.decoder_layers = pretrained.num_hidden_layers
    if "d_model" not in pretrained_overrides:
        pretrained.d_model = pretrained.hidden_size
    if "max_position_embeddings" not in pretrained_overrides:
        pretrained.max_position_embeddings = 1024
    model_config.pretrained_config = pretrained
    model_config.quant_config = None
    return model_config


def _make_mock_model_engine(model_config):
    """Minimal mock PyTorchModelEngine."""
    engine = Mock()
    engine.model.model_config = model_config
    engine.dtype = "bfloat16"
    engine.is_draft_model = False
    engine.kv_cache_manager_key = ResourceManagerType.KV_CACHE_MANAGER
    return engine


def _make_creator(kv_cache_config, model_config=None, is_enc_dec=False):
    """Create a KvCacheCreator with minimal mocking."""
    if model_config is None:
        model_config = _make_mock_model_config(is_encoder_decoder=is_enc_dec)
    model_engine = _make_mock_model_engine(model_config)

    from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2

    with patch(
        "tensorrt_llm._torch.pyexecutor._util.get_kv_cache_manager_cls",
        return_value=KVCacheManagerV2,
    ):
        creator = KvCacheCreator.__new__(KvCacheCreator)
        creator._model_engine = model_engine
        creator._draft_model_engine = None
        creator._mapping = Mock()
        creator._mapping.enable_attention_dp = False
        creator._mapping.tp_size = 1
        creator._mapping.pp_size = 1
        creator._mapping.cp_config = {}
        creator._mapping.is_last_pp_rank.return_value = True
        creator._kv_cache_config = kv_cache_config
        creator._max_kv_tokens_in = kv_cache_config.max_tokens
        creator._max_num_tokens = 4096
        creator._max_beam_width = 1
        creator._kv_connector_manager = None
        creator._llm_args = Mock()
        creator._llm_args.extra_resource_managers = {}
        creator._cache_transceiver_config = None
        creator._speculative_config = None
        creator._sparse_attention_config = None
        creator._tokens_per_block = 64
        creator._max_seq_len = 2048
        creator._max_batch_size = 8
        creator._net_max_seq_len = 2048
        creator._dummy_reqs = None
        creator._profiling_stage_data = None
        creator._kv_cache_manager_cls = KVCacheManagerV2
        creator._execution_stream = None
        creator._draft_config = None
        creator._skip_est = True
    return creator


# ---------------------------------------------------------------------------
# Tests: _split_kv_cache_budget_for_cross
# ---------------------------------------------------------------------------


class TestSplitKvCacheBudgetForCross:
    """Test the budget splitting method directly."""

    def test_split_50_50(self):
        total = 10 * (1 << 30)  # 10 GiB
        config = _make_mock_kv_cache_config(cross_kv_cache_fraction=0.5, max_gpu_total_bytes=total)

        creator = _make_creator(config, is_enc_dec=True)
        cross_config = creator._split_kv_cache_budget_for_cross()

        assert cross_config is not None
        assert cross_config.max_gpu_total_bytes == total // 2
        assert config.max_gpu_total_bytes == total - total // 2

    def test_split_30_70(self):
        total = 10 * (1 << 30)
        config = _make_mock_kv_cache_config(cross_kv_cache_fraction=0.3, max_gpu_total_bytes=total)

        creator = _make_creator(config, is_enc_dec=True)
        cross_config = creator._split_kv_cache_budget_for_cross()

        expected_cross = int(total * 0.3)
        expected_self = total - expected_cross
        assert cross_config.max_gpu_total_bytes == expected_cross
        assert config.max_gpu_total_bytes == expected_self

    def test_no_split_when_fraction_is_none(self):
        total = 10 * (1 << 30)
        config = _make_mock_kv_cache_config(cross_kv_cache_fraction=None, max_gpu_total_bytes=total)

        creator = _make_creator(config)
        cross_config = creator._split_kv_cache_budget_for_cross()

        assert cross_config is None
        assert config.max_gpu_total_bytes == total

    def test_no_split_when_budget_is_none(self):
        config = _make_mock_kv_cache_config(cross_kv_cache_fraction=0.5, max_gpu_total_bytes=None)

        creator = _make_creator(config, is_enc_dec=True)
        cross_config = creator._split_kv_cache_budget_for_cross()

        assert cross_config is None

    def test_no_split_when_budget_is_zero(self):
        config = _make_mock_kv_cache_config(cross_kv_cache_fraction=0.5, max_gpu_total_bytes=0)

        creator = _make_creator(config, is_enc_dec=True)
        cross_config = creator._split_kv_cache_budget_for_cross()

        assert cross_config is None

    def test_is_encoder_decoder_helper(self):
        dec_config = _make_mock_model_config(is_encoder_decoder=False)
        dec_creator = _make_creator(_make_mock_kv_cache_config(), model_config=dec_config)
        assert not dec_creator._is_encoder_decoder()

        enc_dec_config = _make_mock_model_config(is_encoder_decoder=True)
        enc_dec_creator = _make_creator(_make_mock_kv_cache_config(), model_config=enc_dec_config)
        assert enc_dec_creator._is_encoder_decoder()

    def test_budgets_sum_to_total(self):
        """Self + cross budgets always sum to the original total."""
        total = 7 * (1 << 30) + 123  # non-round number
        config = _make_mock_kv_cache_config(cross_kv_cache_fraction=0.4, max_gpu_total_bytes=total)

        creator = _make_creator(config, is_enc_dec=True)
        cross_config = creator._split_kv_cache_budget_for_cross()

        assert (config.max_gpu_total_bytes + cross_config.max_gpu_total_bytes) == total


# ---------------------------------------------------------------------------
# Tests: ResourceManagerType enum
# ---------------------------------------------------------------------------


class TestResourceManagerType:
    """Verify CROSS_KV_CACHE_MANAGER exists in the enum."""

    def test_cross_kv_cache_manager_in_enum(self):
        assert hasattr(ResourceManagerType, "CROSS_KV_CACHE_MANAGER")
        assert ResourceManagerType.CROSS_KV_CACHE_MANAGER.value == "CROSS_KV_CACHE_MANAGER"


# ---------------------------------------------------------------------------
# Tests: Cross-pool geometry and build_managers coverage
# ---------------------------------------------------------------------------


class TestCrossKvCacheConstruction:
    """Exercise the Step 4 construction path beyond helper math."""

    def test_create_cross_kv_cache_manager_uses_encoder_geometry(self):
        config = _make_mock_kv_cache_config(
            cross_kv_cache_fraction=0.5, max_gpu_total_bytes=8 * (1 << 30)
        )
        model_config = _make_mock_model_config(
            is_encoder_decoder=True,
            num_hidden_layers=10,
            num_attention_heads=16,
            num_key_value_heads=16,
            hidden_size=768,
            head_dim=48,
            encoder_layers=8,
            decoder_layers=10,
            encoder_attention_heads=12,
            d_model=768,
            max_position_embeddings=1024,
        )
        creator = _make_creator(config, model_config=model_config)
        cross_cfg = config.model_copy()

        with patch(
            "tensorrt_llm._torch.pyexecutor._util._create_kv_cache_manager",
            return_value=Mock(),
        ) as create_mock:
            creator._create_cross_kv_cache_manager(cross_cfg)

        kwargs = create_mock.call_args.kwargs
        assert kwargs["num_layers"] == 10
        assert kwargs["num_kv_heads"] == 12
        assert kwargs["head_dim"] == 64
        assert kwargs["max_seq_len"] == 1024

        import tensorrt_llm

        assert kwargs["kv_cache_type"] == (
            tensorrt_llm.bindings.internal.batch_manager.CacheType.CROSS
        )

    def test_get_kv_size_per_token_includes_cross_pool_for_enc_dec(self):
        config = _make_mock_kv_cache_config(
            cross_kv_cache_fraction=0.5, max_gpu_total_bytes=8 * (1 << 30)
        )
        model_config = _make_mock_model_config(
            is_encoder_decoder=True,
            num_hidden_layers=10,
            num_attention_heads=16,
            num_key_value_heads=16,
            hidden_size=768,
            head_dim=48,
            encoder_attention_heads=12,
            decoder_layers=10,
            d_model=768,
        )
        creator = _make_creator(config, model_config=model_config)

        with patch.object(
            creator._kv_cache_manager_cls,
            "get_cache_size_per_token",
            side_effect=[100, 40],
        ) as get_size_mock:
            kv_size = creator._get_kv_size_per_token()

        assert kv_size == 140
        assert get_size_mock.call_count == 2

        cross_call = get_size_mock.call_args_list[1]
        proxy_model_config = cross_call.args[0]
        assert proxy_model_config.pretrained_config.num_key_value_heads == 12
        assert proxy_model_config.pretrained_config.num_attention_heads == 12
        assert proxy_model_config.pretrained_config.head_dim == 64
        assert cross_call.kwargs["num_layers"] == 10

    def test_build_managers_registers_cross_pool_for_enc_dec(self):
        creator = _make_creator(
            _make_mock_kv_cache_config(
                cross_kv_cache_fraction=0.5,
                max_gpu_total_bytes=8 * (1 << 30),
            ),
            is_enc_dec=True,
        )
        creator.configure_kv_cache_capacity = Mock()
        creator._should_create_separate_draft_kv_cache = Mock(return_value=False)
        creator._split_kv_cache_budget_for_cross = Mock(return_value=Mock())
        creator._create_kv_cache_manager = Mock(return_value=Mock())
        creator._create_cross_kv_cache_manager = Mock(return_value=Mock())

        resources = {}
        creator.build_managers(resources, estimating_kv_cache=False)

        assert resources[ResourceManagerType.KV_CACHE_MANAGER] is not None
        assert resources[ResourceManagerType.CROSS_KV_CACHE_MANAGER] is not None
        creator._create_cross_kv_cache_manager.assert_called_once()

    def test_build_managers_skips_cross_pool_for_decoder_only(self):
        creator = _make_creator(
            _make_mock_kv_cache_config(
                cross_kv_cache_fraction=None,
                max_gpu_total_bytes=8 * (1 << 30),
            ),
            is_enc_dec=False,
        )
        creator.configure_kv_cache_capacity = Mock()
        creator._should_create_separate_draft_kv_cache = Mock(return_value=False)
        creator._split_kv_cache_budget_for_cross = Mock()
        creator._create_kv_cache_manager = Mock(return_value=Mock())
        creator._create_cross_kv_cache_manager = Mock()

        resources = {}
        creator.build_managers(resources, estimating_kv_cache=False)

        creator._split_kv_cache_budget_for_cross.assert_not_called()
        creator._create_cross_kv_cache_manager.assert_not_called()
        assert resources[ResourceManagerType.CROSS_KV_CACHE_MANAGER] is None


# ---------------------------------------------------------------------------
# Tests: KVCacheV2Scheduler cross_kv_cache_manager parameter
# ---------------------------------------------------------------------------


class TestKVCacheV2SchedulerCrossParam:
    """KVCacheV2Scheduler should accept and store cross_kv_cache_manager."""

    def _make_mock_kv_mgr(self, tokens_per_block=64):
        from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2

        mgr = Mock(spec=KVCacheManagerV2)
        mgr.tokens_per_block = tokens_per_block
        return mgr

    def test_default_cross_is_none(self):
        from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler

        kv_mgr = self._make_mock_kv_mgr()
        scheduler = KVCacheV2Scheduler(
            max_batch_size=8,
            max_num_tokens=4096,
            kv_cache_manager=kv_mgr,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
        )
        assert scheduler.cross_kv_cache_manager is None

    def test_cross_kv_cache_manager_is_stored(self):
        from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler

        kv_mgr = self._make_mock_kv_mgr()
        cross_mgr = self._make_mock_kv_mgr()
        scheduler = KVCacheV2Scheduler(
            max_batch_size=8,
            max_num_tokens=4096,
            kv_cache_manager=kv_mgr,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
            cross_kv_cache_manager=cross_mgr,
        )
        assert scheduler.cross_kv_cache_manager is cross_mgr

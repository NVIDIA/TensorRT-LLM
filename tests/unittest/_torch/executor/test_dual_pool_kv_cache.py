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
"""Tests for dual-pool KV cache construction (enc-dec Steps 4 and 5).

Validates budget splitting, ResourceManagerType.CROSS_KV_CACHE_MANAGER
registration, and the cross pool wiring for both the V1 ``KVCacheManager``
(default and production target) and the V2 ``KVCacheManagerV2``
(additive secondary path) scheduler integrations.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from transformers import PretrainedConfig

import tensorrt_llm
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager, ResourceManagerType
from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy, KvCacheConfig, TorchLlmArgs
from tensorrt_llm.mapping import Mapping

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeCudaStream:
    cuda_stream = 0


class _FakeKVCacheManagerCpp:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.num_pools = 1
        self.max_blocks_per_seq = 1
        self.max_num_blocks = 1

    def allocate_pools(self, _use_uvm):
        pass

    def get_block_pool_pointers(self):
        return torch.empty(0, dtype=torch.int64)

    def get_block_scale_pool_pointers(self):
        return torch.empty(0, dtype=torch.int64)

    def get_layer_to_pool_mapping(self):
        return torch.empty(0, dtype=torch.int32)

    def release_pools(self):
        pass


_DUMMY_MODEL = "/tmp/dummy_model_dual_pool_test"


def _make_kv_cache_config(
    cross_kv_cache_fraction=None,
    max_gpu_total_bytes=None,
    use_kv_cache_manager_v2=True,
    max_tokens=None,
    free_gpu_memory_fraction=0.9,
    host_cache_size=None,
):
    """Create a KvCacheConfig with the fields KvCacheCreator needs."""
    return KvCacheConfig(
        cross_kv_cache_fraction=cross_kv_cache_fraction,
        max_gpu_total_bytes=max_gpu_total_bytes or 0,
        use_kv_cache_manager_v2=use_kv_cache_manager_v2,
        max_tokens=max_tokens,
        free_gpu_memory_fraction=free_gpu_memory_fraction,
        host_cache_size=host_cache_size,
    )


def _make_model_config(
    is_encoder_decoder=False,
    **pretrained_overrides,
):
    """Real ModelConfig wrapping a real HF PretrainedConfig.

    ``is_encoder_decoder`` is set on the HF config so that
    ``ModelConfig.__post_init__`` derives ``ModelConfig.is_encoder_decoder``
    through the production path.  Unknown kwargs are kept as attributes by
    ``transformers``, matching how enc-dec geometry fields appear on real
    BART/T5/Whisper configs.
    """
    pretrained_kwargs = {
        "is_encoder_decoder": is_encoder_decoder,
        "num_hidden_layers": 6,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "hidden_size": 512,
        "head_dim": 64,
        "vocab_size": 32000,
    }
    pretrained_kwargs.update(pretrained_overrides)
    pretrained_kwargs.setdefault(
        "architectures",
        ["T5ForConditionalGeneration"] if is_encoder_decoder else ["LlamaForCausalLM"],
    )
    pretrained_kwargs.setdefault(
        "encoder_attention_heads", pretrained_kwargs["num_attention_heads"]
    )
    pretrained_kwargs.setdefault(
        "decoder_attention_heads", pretrained_kwargs["num_attention_heads"]
    )
    pretrained_kwargs.setdefault("encoder_layers", pretrained_kwargs["num_hidden_layers"])
    pretrained_kwargs.setdefault("decoder_layers", pretrained_kwargs["num_hidden_layers"])
    pretrained_kwargs.setdefault("d_model", pretrained_kwargs["hidden_size"])
    pretrained_kwargs.setdefault("max_position_embeddings", 1024)
    return ModelConfig(pretrained_config=PretrainedConfig(**pretrained_kwargs))


def _make_mock_model_engine(model_config):
    """Mock PyTorchModelEngine shell; a real engine needs GPU + loaded weights."""
    engine = Mock()
    engine.model.model_config = model_config
    engine.dtype = torch.bfloat16
    engine.is_draft_model = False
    engine.kv_cache_manager_key = ResourceManagerType.KV_CACHE_MANAGER
    return engine


def _make_creator(
    kv_cache_config,
    model_config=None,
    is_enc_dec=False,
    manager_cls=None,
    mapping=None,
):
    """Create a KvCacheCreator from real config objects and a mock engine.

    ``manager_cls`` selects the KV cache manager class the creator binds to.
    Defaults to ``KVCacheManagerV2`` when ``kv_cache_config.use_kv_cache_manager_v2``
    is True, otherwise the V1 ``KVCacheManager``.  Tests can override
    explicitly via ``manager_cls`` to exercise either path independently.
    """
    if model_config is None:
        model_config = _make_model_config(is_encoder_decoder=is_enc_dec)
    model_engine = _make_mock_model_engine(model_config)

    if manager_cls is None:
        manager_cls = (
            KVCacheManagerV2 if kv_cache_config.use_kv_cache_manager_v2 else KVCacheManager
        )

    llm_args = TorchLlmArgs(model=_DUMMY_MODEL, skip_tokenizer_init=True)
    # TorchLlmArgs defaults max_input_len to 1024, which would silently cap
    # the cross-pool encoder capacity; clear it so each test controls the
    # cap explicitly.
    llm_args.max_input_len = None

    creator = KvCacheCreator.__new__(KvCacheCreator)
    creator._model_engine = model_engine
    creator._draft_model_engine = None
    creator._mapping = mapping if mapping is not None else Mapping()
    creator._kv_cache_config = kv_cache_config
    creator._max_kv_tokens_in = kv_cache_config.max_tokens
    creator._max_num_tokens = 4096
    creator._max_beam_width = 1
    creator._kv_connector_manager = None
    creator._llm_args = llm_args
    creator._cache_transceiver_config = None
    creator._speculative_config = None
    creator._sparse_attention_config = None
    creator._tokens_per_block = 64
    creator._max_seq_len = 2048
    creator._max_batch_size = 8
    creator._net_max_seq_len = 2048
    creator._dummy_reqs = None
    creator._profiling_stage_data = None
    creator._kv_cache_manager_cls = manager_cls
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
        config = _make_kv_cache_config(
            cross_kv_cache_fraction=0.5,
            max_gpu_total_bytes=total,
            free_gpu_memory_fraction=0.8,
        )

        creator = _make_creator(config, is_enc_dec=True)
        self_config, cross_config = creator._split_kv_cache_budget_for_cross()

        assert self_config is not config
        assert cross_config.max_gpu_total_bytes == total // 2
        assert self_config.max_gpu_total_bytes == total - total // 2
        assert cross_config.free_gpu_memory_fraction == pytest.approx(0.4)
        assert self_config.free_gpu_memory_fraction == pytest.approx(0.4)
        assert config.max_gpu_total_bytes == total
        assert config.free_gpu_memory_fraction == pytest.approx(0.8)

    def test_split_30_70(self):
        total = 10 * (1 << 30)
        config = _make_kv_cache_config(
            cross_kv_cache_fraction=0.3,
            max_gpu_total_bytes=total,
            free_gpu_memory_fraction=0.8,
        )

        creator = _make_creator(config, is_enc_dec=True)
        self_config, cross_config = creator._split_kv_cache_budget_for_cross()

        expected_cross = int(total * 0.3)
        expected_self = total - expected_cross
        assert cross_config.max_gpu_total_bytes == expected_cross
        assert self_config.max_gpu_total_bytes == expected_self
        assert cross_config.free_gpu_memory_fraction == pytest.approx(0.24)
        assert self_config.free_gpu_memory_fraction == pytest.approx(0.56)
        assert config.max_gpu_total_bytes == total
        assert config.free_gpu_memory_fraction == pytest.approx(0.8)

    def test_no_split_when_fraction_is_none(self):
        total = 10 * (1 << 30)
        config = _make_kv_cache_config(cross_kv_cache_fraction=None, max_gpu_total_bytes=total)

        creator = _make_creator(config, is_enc_dec=True)
        with pytest.raises(ValueError, match="cross_kv_cache_fraction"):
            creator._split_kv_cache_budget_for_cross()

    def test_split_free_fraction_when_budget_is_none(self):
        config = _make_kv_cache_config(
            cross_kv_cache_fraction=0.5,
            max_gpu_total_bytes=None,
            max_tokens=1000,
            free_gpu_memory_fraction=0.8,
        )

        creator = _make_creator(config, is_enc_dec=True)
        self_config, cross_config = creator._split_kv_cache_budget_for_cross()

        assert cross_config.max_tokens == 1000
        assert self_config.max_tokens == 1000
        assert cross_config.free_gpu_memory_fraction == pytest.approx(0.4)
        assert self_config.free_gpu_memory_fraction == pytest.approx(0.4)
        assert config.free_gpu_memory_fraction == pytest.approx(0.8)

    def test_split_free_fraction_when_budget_is_zero(self):
        config = _make_kv_cache_config(
            cross_kv_cache_fraction=0.5,
            max_gpu_total_bytes=0,
            max_tokens=1000,
            free_gpu_memory_fraction=0.8,
        )

        creator = _make_creator(config, is_enc_dec=True)
        self_config, cross_config = creator._split_kv_cache_budget_for_cross()

        assert cross_config.max_tokens == 1000
        assert self_config.max_tokens == 1000
        assert cross_config.free_gpu_memory_fraction == pytest.approx(0.4)
        assert self_config.free_gpu_memory_fraction == pytest.approx(0.4)
        assert config.free_gpu_memory_fraction == pytest.approx(0.8)

    def test_is_encoder_decoder_helper(self):
        dec_config = _make_model_config(is_encoder_decoder=False)
        dec_creator = _make_creator(_make_kv_cache_config(), model_config=dec_config)
        assert not dec_creator._is_encoder_decoder()

        enc_dec_config = _make_model_config(is_encoder_decoder=True)
        enc_dec_creator = _make_creator(_make_kv_cache_config(), model_config=enc_dec_config)
        assert enc_dec_creator._is_encoder_decoder()

    def test_budgets_sum_to_total(self):
        """Self + cross budgets always sum to the original total."""
        total = 7 * (1 << 30) + 123  # non-round number
        config = _make_kv_cache_config(cross_kv_cache_fraction=0.4, max_gpu_total_bytes=total)

        creator = _make_creator(config, is_enc_dec=True)
        self_config, cross_config = creator._split_kv_cache_budget_for_cross()

        assert (self_config.max_gpu_total_bytes + cross_config.max_gpu_total_bytes) == total
        assert config.max_gpu_total_bytes == total

    def test_host_cache_budget_is_split_without_mutating_base_config(self):
        """Self + cross host cache budgets sum to the original host budget."""
        total_host = 7 * (1 << 30) + 123
        config = _make_kv_cache_config(
            cross_kv_cache_fraction=0.4,
            max_gpu_total_bytes=8 * (1 << 30),
            host_cache_size=total_host,
        )

        creator = _make_creator(config, is_enc_dec=True)
        self_config, cross_config = creator._split_kv_cache_budget_for_cross()

        expected_cross_host = int(total_host * 0.4)
        expected_self_host = total_host - expected_cross_host
        assert cross_config.host_cache_size == expected_cross_host
        assert self_config.host_cache_size == expected_self_host
        assert (self_config.host_cache_size + cross_config.host_cache_size) == total_host
        assert config.host_cache_size == total_host

    def test_host_cache_budget_counts_as_split_budget_source(self):
        total_host = 4 * (1 << 30)
        config = _make_kv_cache_config(
            cross_kv_cache_fraction=0.25,
            max_gpu_total_bytes=None,
            free_gpu_memory_fraction=0.6,
            host_cache_size=total_host,
        )

        creator = _make_creator(config, is_enc_dec=True)
        self_config, cross_config = creator._split_kv_cache_budget_for_cross()

        assert cross_config.host_cache_size == total_host // 4
        assert self_config.host_cache_size == total_host - total_host // 4
        assert config.host_cache_size == total_host


# ---------------------------------------------------------------------------
# Tests: ResourceManagerType enum
# ---------------------------------------------------------------------------


class TestResourceManagerType:
    """Verify CROSS_KV_CACHE_MANAGER exists in the enum."""

    def test_cross_kv_cache_manager_in_enum(self):
        assert ResourceManagerType.CROSS_KV_CACHE_MANAGER.value == "CROSS_KV_CACHE_MANAGER"


# ---------------------------------------------------------------------------
# Tests: Cross-pool geometry and build_managers coverage
# ---------------------------------------------------------------------------


class TestCrossKvCacheConstruction:
    """Exercise the Steps 4 and 5 construction path beyond helper math."""

    @pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True])
    def test_create_cross_kv_cache_manager_uses_encoder_geometry(self, use_kv_cache_manager_v2):
        expected_cls = KVCacheManagerV2 if use_kv_cache_manager_v2 else KVCacheManager
        config = _make_kv_cache_config(
            cross_kv_cache_fraction=0.5,
            max_gpu_total_bytes=8 * (1 << 30),
            use_kv_cache_manager_v2=use_kv_cache_manager_v2,
        )
        model_config = _make_model_config(
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
        creator = _make_creator(config, model_config=model_config, manager_cls=expected_cls)
        cross_cfg = config.model_copy()

        with patch(
            "tensorrt_llm._torch.pyexecutor._util._create_kv_cache_manager",
            return_value=Mock(),
        ) as create_mock:
            creator._create_cross_kv_cache_manager(cross_cfg)

        kwargs = create_mock.call_args.kwargs
        # Cross pool must use the same manager class as the self pool so
        # both pools share the same runtime ABI.  V1 is the default and
        # production target; V2 is an additive secondary path.
        assert kwargs["kv_cache_manager_cls"] is expected_cls
        assert kwargs["num_layers"] == 10
        assert kwargs["num_kv_heads"] == 12
        assert kwargs["head_dim"] == 64
        assert kwargs["max_seq_len"] == 1024
        assert kwargs["kv_cache_type"] == (
            tensorrt_llm.bindings.internal.batch_manager.CacheType.CROSS
        )

    def test_create_cross_kv_cache_manager_uses_tp_sharded_encoder_heads(self):
        num_decoder_layers = 10
        encoder_num_attention_heads = 12
        encoder_num_kv_heads = 12
        encoder_hidden_size = 768
        tp_size = 2
        local_kv_heads = encoder_num_kv_heads // tp_size
        head_dim = encoder_hidden_size // encoder_num_attention_heads

        config = _make_kv_cache_config(
            cross_kv_cache_fraction=0.5,
            max_gpu_total_bytes=8 * (1 << 30),
            max_tokens=2048,
            use_kv_cache_manager_v2=False,
        )
        model_config = _make_model_config(
            is_encoder_decoder=True,
            decoder_layers=num_decoder_layers,
            encoder_attention_heads=encoder_num_attention_heads,
            encoder_num_key_value_heads=encoder_num_kv_heads,
            d_model=encoder_hidden_size,
            max_position_embeddings=1024,
        )
        mapping = Mapping(world_size=tp_size, tp_size=tp_size)
        creator = _make_creator(
            config,
            model_config=model_config,
            manager_cls=KVCacheManager,
            mapping=mapping,
        )
        creator._skip_est = False

        import tensorrt_llm

        with (
            patch(
                "tensorrt_llm._torch.pyexecutor.resource_manager.KVCacheManagerCpp",
                _FakeKVCacheManagerCpp,
            ),
            patch(
                "tensorrt_llm._torch.pyexecutor.resource_manager.torch.cuda.Stream",
                return_value=_FakeCudaStream(),
            ),
            patch(
                "tensorrt_llm._torch.pyexecutor.resource_manager.prefer_pinned",
                return_value=False,
            ),
        ):
            cross_kv_cache_manager = creator._create_cross_kv_cache_manager(
                config.model_copy(),
                estimating_kv_cache=True,
            )

        try:
            assert cross_kv_cache_manager.mapping is mapping
            assert cross_kv_cache_manager.kv_cache_type == (
                tensorrt_llm.bindings.internal.batch_manager.CacheType.CROSS
            )
            assert cross_kv_cache_manager.num_kv_heads == encoder_num_kv_heads
            assert (
                cross_kv_cache_manager.num_kv_heads_per_layer
                == [local_kv_heads] * num_decoder_layers
            )
            assert (
                cross_kv_cache_manager.total_num_kv_heads_per_layer
                == [local_kv_heads] * num_decoder_layers
            )
            assert cross_kv_cache_manager.head_dim == head_dim
        finally:
            cross_kv_cache_manager.shutdown()

        size_per_token = creator._get_cross_kv_size_per_token()
        kv_factor = 2  # key and value caches
        bytes_per_element = 2  # bfloat16
        assert size_per_token == (
            num_decoder_layers * kv_factor * local_kv_heads * head_dim * bytes_per_element
        )

    def test_cross_layout_uses_max_input_len_for_encoder_capacity(self):
        config = _make_kv_cache_config(
            cross_kv_cache_fraction=0.5,
            max_gpu_total_bytes=8 * (1 << 30),
        )
        model_config = _make_model_config(
            is_encoder_decoder=True,
            max_position_embeddings=4096,
        )
        creator = _make_creator(config, model_config=model_config)
        creator._llm_args.max_input_len = 1536
        creator._max_seq_len = 864

        _, _, _, max_seq_len = creator._get_cross_kv_cache_layout(fallback_max_seq_len=2048)

        assert max_seq_len == 1536

    def test_build_managers_cross_pool_ignores_mutated_self_max_seq_len(self):
        config = _make_kv_cache_config(
            cross_kv_cache_fraction=0.5,
            max_gpu_total_bytes=8 * (1 << 30),
        )
        model_config = _make_model_config(
            is_encoder_decoder=True,
            max_position_embeddings=4096,
        )
        creator = _make_creator(config, model_config=model_config)
        creator._llm_args.max_input_len = None
        creator._max_seq_len = 2048
        creator.configure_kv_cache_capacity = Mock()
        creator._should_create_separate_draft_kv_cache = Mock(return_value=False)

        def create_self_manager(*_args, **_kwargs):
            creator._max_seq_len = 864
            manager = Mock()
            manager.max_seq_len = 864
            return manager

        captured_cross_max_seq_lens = []

        def create_cross_manager(*_args, **kwargs):
            captured_cross_max_seq_lens.append(kwargs["max_seq_len"])
            manager = Mock()
            manager.max_seq_len = kwargs["max_seq_len"]
            return manager

        creator._create_kv_cache_manager = Mock(side_effect=create_self_manager)
        with patch(
            "tensorrt_llm._torch.pyexecutor._util._create_kv_cache_manager",
            side_effect=create_cross_manager,
        ):
            creator.build_managers({}, estimating_kv_cache=False)

        assert creator._max_seq_len == 864
        assert captured_cross_max_seq_lens == [2048]

    def test_get_kv_size_per_token_includes_cross_pool_for_enc_dec(self):
        config = _make_kv_cache_config(
            cross_kv_cache_fraction=0.5, max_gpu_total_bytes=8 * (1 << 30)
        )
        model_config = _make_model_config(
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

        assert kv_size.slope == 140
        assert kv_size.intercept == 0
        assert get_size_mock.call_count == 2

        cross_call = get_size_mock.call_args_list[1]
        proxy_model_config = cross_call.args[0]
        assert proxy_model_config.pretrained_config.num_key_value_heads == 12
        assert proxy_model_config.pretrained_config.num_attention_heads == 12
        assert proxy_model_config.pretrained_config.head_dim == 64
        assert cross_call.kwargs["num_layers"] == 10

    @pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True])
    def test_build_managers_registers_cross_pool_for_enc_dec(self, use_kv_cache_manager_v2):
        creator = _make_creator(
            _make_kv_cache_config(
                cross_kv_cache_fraction=0.5,
                max_gpu_total_bytes=8 * (1 << 30),
                use_kv_cache_manager_v2=use_kv_cache_manager_v2,
            ),
            is_enc_dec=True,
        )
        creator.configure_kv_cache_capacity = Mock()
        creator._should_create_separate_draft_kv_cache = Mock(return_value=False)
        creator._split_kv_cache_budget_for_cross = Mock(return_value=(Mock(), Mock()))
        creator._create_kv_cache_manager = Mock(return_value=Mock())
        creator._create_cross_kv_cache_manager = Mock(return_value=Mock())

        resources = {}
        creator.build_managers(resources, estimating_kv_cache=False)

        creator._create_cross_kv_cache_manager.assert_called_once()

    @pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True])
    def test_build_managers_registers_cross_pool_for_enc_dec_estimation(
        self, use_kv_cache_manager_v2
    ):
        creator = _make_creator(
            _make_kv_cache_config(
                cross_kv_cache_fraction=0.5,
                max_gpu_total_bytes=0,
                max_tokens=1024,
                free_gpu_memory_fraction=0.8,
                use_kv_cache_manager_v2=use_kv_cache_manager_v2,
            ),
            is_enc_dec=True,
        )
        creator.configure_kv_cache_capacity = Mock()
        creator._should_create_separate_draft_kv_cache = Mock(return_value=False)
        creator._split_kv_cache_budget_for_cross = Mock(return_value=(Mock(), Mock()))
        creator._create_kv_cache_manager = Mock(return_value=Mock())
        creator._create_cross_kv_cache_manager = Mock(return_value=Mock())

        resources = {}
        creator.build_managers(resources, estimating_kv_cache=True)

        creator._create_cross_kv_cache_manager.assert_called_once()

    @pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True])
    def test_build_managers_uses_split_cross_budget_without_mutating_base_config(
        self, use_kv_cache_manager_v2
    ):
        total_budget = 10 * (1 << 30)
        creator = _make_creator(
            _make_kv_cache_config(
                cross_kv_cache_fraction=0.5,
                max_gpu_total_bytes=total_budget,
                free_gpu_memory_fraction=0.9,
                use_kv_cache_manager_v2=use_kv_cache_manager_v2,
            ),
            is_enc_dec=True,
        )
        creator.configure_kv_cache_capacity = Mock()
        creator._should_create_separate_draft_kv_cache = Mock(return_value=False)

        self_budgets = []
        cross_budgets = []

        def create_self_manager(*_args, **kwargs):
            self_cfg = kwargs["kv_cache_config_override"]
            self_budgets.append(
                (
                    self_cfg.free_gpu_memory_fraction,
                    self_cfg.max_gpu_total_bytes,
                )
            )
            return Mock()

        def create_cross_manager(cross_cfg, *_args, **_kwargs):
            cross_budgets.append(
                (
                    cross_cfg.free_gpu_memory_fraction,
                    cross_cfg.max_gpu_total_bytes,
                )
            )
            return Mock()

        creator._create_kv_cache_manager = Mock(side_effect=create_self_manager)
        creator._create_cross_kv_cache_manager = Mock(side_effect=create_cross_manager)

        resources = {}
        creator.build_managers(resources, estimating_kv_cache=True)

        assert creator._kv_cache_config.free_gpu_memory_fraction == pytest.approx(0.9)
        assert creator._kv_cache_config.max_gpu_total_bytes == total_budget

        creator.build_managers(resources, estimating_kv_cache=False)

        expected_split = total_budget // 2
        assert self_budgets == [
            (pytest.approx(0.45), expected_split),
            (pytest.approx(0.45), expected_split),
        ]
        assert cross_budgets == [
            (pytest.approx(0.45), expected_split),
            (pytest.approx(0.45), expected_split),
        ]

    def test_build_managers_skips_cross_pool_for_decoder_only(self):
        creator = _make_creator(
            _make_kv_cache_config(
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

    def test_factory_forwards_encoder_init_until_state_for_cross_pool(self):
        """The executor factory must widen V2 scheduling to ENCODER_INIT.

        Without this, V2 enc-dec requests are filtered by the default
        CONTEXT_INIT state gate before the encoder loop can see them.
        """
        from tensorrt_llm._torch.pyexecutor._util import create_py_executor_instance
        from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState

        kv_mgr = Mock()
        kv_mgr.tokens_per_block = 64
        cross_mgr = Mock()
        resources = {
            ResourceManagerType.KV_CACHE_MANAGER: kv_mgr,
            ResourceManagerType.CROSS_KV_CACHE_MANAGER: cross_mgr,
            ResourceManagerType.DRAFT_KV_CACHE_MANAGER: None,
        }
        mapping = Mapping()
        model_engine = SimpleNamespace(
            spec_config=None,
            model=SimpleNamespace(model_config=_make_model_config()),
        )
        llm_args = TorchLlmArgs(
            model=_DUMMY_MODEL,
            skip_tokenizer_init=True,
            disable_overlap_scheduler=True,
        )

        with (
            patch(
                "tensorrt_llm._torch.pyexecutor._util.KVCacheManagerV2",
                new=Mock,
            ),
            patch(
                "tensorrt_llm._torch.pyexecutor._util.KVCacheV2Scheduler",
            ) as scheduler_cls,
            patch(
                "tensorrt_llm._torch.pyexecutor._util.create_kv_cache_transceiver",
                return_value=None,
            ),
            patch(
                "tensorrt_llm._torch.pyexecutor._util.PyExecutor",
            ),
        ):
            scheduler_cls.return_value = Mock()
            create_py_executor_instance(
                dist=Mock(),
                resources=resources,
                mapping=mapping,
                llm_args=llm_args,
                ctx_chunk_config=None,
                model_engine=model_engine,
                start_worker=False,
                sampler=Mock(),
                drafter=None,
                max_seq_len=128,
                max_batch_size=8,
                max_beam_width=1,
                max_num_tokens=4096,
            )

        kwargs = scheduler_cls.call_args.kwargs
        assert kwargs["cross_kv_cache_manager"] is cross_mgr
        assert kwargs["no_schedule_until_state"] == LlmRequestState.ENCODER_INIT


# ---------------------------------------------------------------------------
# Tests: V1 scheduler cross_kv_cache_manager wiring.
# ---------------------------------------------------------------------------


class TestBindCapacitySchedulerCrossParam:
    """C++-bound V1 ``BindCapacityScheduler`` exposes cross-KV wiring.

    The C++ ``CapacityScheduler`` already accepts a cross manager. The Python
    wrapper forwards the cross pool and the ENCODER_INIT gating.
    """

    def test_default_cross_is_none_and_default_until_state(self):
        from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
        from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import BindCapacityScheduler

        with patch(
            "tensorrt_llm._torch.pyexecutor.scheduler.scheduler.tb_internal.algorithms.CapacityScheduler"
        ) as cap_cls:
            cap_cls.return_value = Mock()
            scheduler = BindCapacityScheduler(
                max_num_requests=8,
                kv_cache_manager=Mock(),
                peft_cache_manager=None,
                scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
            )

        assert scheduler.cross_kv_cache_manager is None
        kwargs = cap_cls.call_args.kwargs
        assert kwargs["no_schedule_until_state"] == LlmRequestState.CONTEXT_INIT

    def test_cross_kv_cache_manager_and_until_state_are_forwarded(self):
        from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
        from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import BindCapacityScheduler

        cross_mgr = Mock()
        kv_mgr = Mock()
        with patch(
            "tensorrt_llm._torch.pyexecutor.scheduler.scheduler.tb_internal.algorithms.CapacityScheduler"
        ) as cap_cls:
            impl = Mock()
            cap_cls.return_value = impl
            scheduler = BindCapacityScheduler(
                max_num_requests=8,
                kv_cache_manager=kv_mgr,
                peft_cache_manager=None,
                scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
                cross_kv_cache_manager=cross_mgr,
                no_schedule_until_state=LlmRequestState.ENCODER_INIT,
            )

            # Construction forwarded the gating to the C++ binding.
            ctor_kwargs = cap_cls.call_args.kwargs
            assert ctor_kwargs["no_schedule_until_state"] == LlmRequestState.ENCODER_INIT

            # schedule_request must forward the cross manager to the C++
            # __call__ so the dual-pool scheduling logic activates.
            impl.return_value = ([], [], [])
            scheduler.schedule_request([])
            impl.assert_called_once_with([], kv_mgr, None, cross_mgr)


class TestSimpleUnifiedSchedulerCrossParam:
    """V1 Python ``SimpleUnifiedScheduler`` exposes cross-KV wiring."""

    def test_cross_kv_cache_manager_and_until_state_are_forwarded(self):
        from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
        from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import SimpleUnifiedScheduler

        kv_mgr = Mock()
        kv_mgr.is_variable_window = False
        kv_mgr.enable_block_reuse = False
        cross_mgr = Mock()
        cross_mgr.is_variable_window = False
        cross_mgr.enable_block_reuse = False

        scheduler = SimpleUnifiedScheduler(
            max_batch_size=8,
            max_num_tokens=4096,
            kv_cache_manager=kv_mgr,
            peft_cache_manager=None,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
            cross_kv_cache_manager=cross_mgr,
            no_schedule_until_state=LlmRequestState.ENCODER_INIT,
        )

        assert scheduler.capacity_scheduler.cross_kv_cache_manager is cross_mgr
        assert scheduler.capacity_scheduler.no_schedule_until_state == LlmRequestState.ENCODER_INIT
        assert (
            scheduler.micro_batch_scheduler.no_schedule_until_state == LlmRequestState.ENCODER_INIT
        )


# ---------------------------------------------------------------------------
# Tests: V1 dual-pool smoke test.
# ---------------------------------------------------------------------------


class TestV1DualPoolSmoke:
    """Smoke test exercising V1 dual-pool construction.

    Constructs both pools as V1 ``KVCacheManager`` instances with
    ``CacheType.SELF`` / ``CacheType.CROSS`` (via mocked
    ``_create_kv_cache_manager``) and verifies that ``build_managers``
    wires both pools into the resource map for the V1 production path.

    Running an actual encoder + decoder context iteration requires GPUs
    and a full model engine; that lives in the integration suite. Here
    we verify the V1 construction wiring with mocks consistent with the
    rest of this file.
    """

    def test_build_managers_uses_v1_kv_cache_manager_for_both_pools(self):
        kv_cache_config = _make_kv_cache_config(
            cross_kv_cache_fraction=0.5,
            max_gpu_total_bytes=8 * (1 << 30),
            use_kv_cache_manager_v2=False,
        )
        creator = _make_creator(kv_cache_config, is_enc_dec=True, manager_cls=KVCacheManager)
        creator.configure_kv_cache_capacity = Mock()
        creator._should_create_separate_draft_kv_cache = Mock(return_value=False)
        creator._split_kv_cache_budget_for_cross = Mock(return_value=(Mock(), Mock()))

        # Both _create_kv_cache_manager (self pool) and
        # _create_cross_kv_cache_manager are exercised through the
        # underlying free-function _create_kv_cache_manager so we can
        # assert the manager_cls and CacheType for each call.
        import tensorrt_llm

        cache_type_self = tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF
        cache_type_cross = tensorrt_llm.bindings.internal.batch_manager.CacheType.CROSS

        # Stub the self-pool path (_create_kv_cache_manager method) to
        # avoid invoking the heavyweight free function.
        self_mgr = Mock(spec=KVCacheManager)
        self_mgr.kv_cache_type = cache_type_self
        creator._create_kv_cache_manager = Mock(return_value=self_mgr)

        cross_mgr = Mock(spec=KVCacheManager)
        cross_mgr.kv_cache_type = cache_type_cross
        with patch(
            "tensorrt_llm._torch.pyexecutor._util._create_kv_cache_manager",
            return_value=cross_mgr,
        ) as create_mock:
            resources = {}
            creator.build_managers(resources, estimating_kv_cache=False)

        # Self pool: registered as KV_CACHE_MANAGER.
        assert resources[ResourceManagerType.KV_CACHE_MANAGER] is self_mgr

        # Cross pool: registered as CROSS_KV_CACHE_MANAGER and built
        # with the V1 KVCacheManager class + CacheType.CROSS.
        assert resources[ResourceManagerType.CROSS_KV_CACHE_MANAGER] is cross_mgr
        cross_kwargs = create_mock.call_args.kwargs
        assert cross_kwargs["kv_cache_manager_cls"] is KVCacheManager
        assert cross_kwargs["kv_cache_type"] == cache_type_cross

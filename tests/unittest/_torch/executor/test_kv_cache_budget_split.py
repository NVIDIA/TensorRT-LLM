# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for KV cache budget splitting between target and draft managers."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor._util import CacheCost, KvCacheCreator
from tensorrt_llm._torch.pyexecutor.config_utils import uses_vswa_kv_cache_layout
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.llmapi.llm_args import KvCacheConfig

pytestmark = pytest.mark.cpu_only


GB = 1 << 30

# Budgets that back an offload tier reserved in full at manager construction.
OFFLOAD_TIER_BUDGET_ATTRS = ("host_cache_size", "disk_cache_size")


def _make_creator(
    max_gpu_total_bytes: int,
    host_cache_size=None,
    disk_cache_size: int | None = None,
    disk_cache_path: str | None = None,
    total_kv_per_token: int = 100,
    target_kv_per_token: int = 80,
    total_kv_intercept: int = 0,
    target_kv_intercept: int = 0,
) -> KvCacheCreator:
    """Minimal KvCacheCreator for budget-split helpers.

    ``*_intercept`` model the affine fixed cost (e.g. mamba SSM state) that a
    manager pays per batch regardless of token count. The draft cost is derived
    as ``total - target`` for both slope and intercept.
    """
    c = object.__new__(KvCacheCreator)

    c._kv_cache_config = KvCacheConfig(
        max_gpu_total_bytes=max_gpu_total_bytes,
        host_cache_size=host_cache_size,
        disk_cache_size=disk_cache_size,
        disk_cache_path=disk_cache_path,
    )
    c._tokens_per_block = 64
    c._max_seq_len = 1024
    c._max_batch_size = 1
    c._speculative_config = None
    c._mapping = Mock()
    c._model_engine = Mock()
    c._llm_args = SimpleNamespace(kv_cache_compression_config=None)

    c._kv_cache_manager_cls = Mock()
    c._kv_cache_manager_cls.get_cache_size_per_token = Mock(
        return_value=(target_kv_per_token, target_kv_intercept)
    )

    c._get_kv_size_per_token = Mock(
        return_value=CacheCost(
            slope=total_kv_per_token,
            intercept=total_kv_intercept,
        )
    )
    c._should_create_separate_draft_kv_cache = Mock(return_value=True)

    return c


class TestSplitGpuBudgetForDraft:
    @pytest.mark.parametrize(
        "is_external_drafter",
        [True, False],
        ids=["external_dflash", "eagle3_mtp"],
    )
    def test_one_model_draft_cost_uses_derived_kv_config(
        self,
        mocker,
        is_external_drafter,
    ) -> None:
        class DraftModelConfig:
            quant_config = None
            pretrained_config = SimpleNamespace(
                num_hidden_layers=1,
                hidden_size=32,
                num_attention_heads=4,
                num_key_value_heads=2,
                sliding_window=512,
                layer_types=["sliding_attention"],
            )

            def get_num_attention_layers(self):
                return 1

        creator = object.__new__(KvCacheCreator)
        target_kv_config = KvCacheConfig(max_attention_window=[16384])
        mode = Mock()
        mode.is_external_drafter.return_value = is_external_drafter

        target_model_config = SimpleNamespace(is_encoder_decoder=False)
        draft_model_config = DraftModelConfig()
        draft_kv_configs = []

        class DraftCostKVCacheManager(KVCacheManagerV2):
            @staticmethod
            def get_cache_size_per_token(model_config, *args, **kwargs):
                if model_config is target_model_config:
                    return 10
                draft_kv_configs.append(kwargs["kv_cache_config"])
                return KVCacheManagerV2.get_cache_size_per_token(model_config, *args, **kwargs)

        creator._kv_cache_config = target_kv_config
        creator._tokens_per_block = 64
        creator._max_seq_len = 16384
        creator._max_batch_size = 1
        creator._max_num_tokens = 128
        creator._mapping = Mock(enable_attention_dp=False, tp_size=1)
        creator._mapping.pp_layers.return_value = [0]
        creator._mapping.is_last_pp_rank.return_value = True
        creator._speculative_config = SimpleNamespace(spec_dec_mode=mode)
        creator._model_engine = SimpleNamespace(
            model=SimpleNamespace(model_config=target_model_config)
        )
        creator._draft_model_engine = None
        creator._draft_config = draft_model_config
        creator._kv_cache_manager_cls = DraftCostKVCacheManager
        creator._is_disagg = False
        creator._should_create_separate_draft_kv_cache = Mock(return_value=True)
        creator._get_effective_draft_config = Mock(return_value=draft_model_config)
        creator._get_num_draft_layers = Mock(return_value=1)
        get_manager_cls = mocker.patch(
            "tensorrt_llm._torch.pyexecutor._util.get_kv_cache_manager_cls",
            return_value=DraftCostKVCacheManager,
        )

        # The draft layer stores 64 bytes/token in a fixed 512-token window.
        # Leaking the target's 16K window would instead count it as 64 bytes/token.
        cost = creator._get_kv_size_per_token()
        assert cost == CacheCost(slope=10, intercept=512 * 64)
        assert len(draft_kv_configs) == 1
        draft_kv_config = draft_kv_configs[0]
        assert draft_kv_config.max_attention_window == [512]
        assert target_kv_config.max_attention_window == [16384]
        if is_external_drafter:
            assert get_manager_cls.call_args.args[1] is draft_kv_config
        else:
            get_manager_cls.assert_not_called()
        mode.is_dflash.assert_not_called()

    def test_target_cost_uses_derived_layer_type_windows(self) -> None:
        """A target with a mixed sliding/full `layer_types` schedule on
        KVCacheManagerV2 is costed from the same derived per-layer windows
        `_create_kv_cache_manager` builds it with: its three sliding layers
        become a fixed per-request cost and only the full layer is charged per
        token, so the split matches the manager's pools. Without the derivation
        the target counted four full layers per token and no fixed cost."""

        class TargetModelConfig:
            quant_config = None
            is_encoder_decoder = False
            pretrained_config = SimpleNamespace(
                num_hidden_layers=4,
                hidden_size=1024,
                num_attention_heads=8,
                num_key_value_heads=8,
                sliding_window=512,
                layer_types=[
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention",
                    "sliding_attention",
                ],
            )

            def get_num_attention_layers(self) -> int:
                return 4

        class DraftModelConfig:
            """A one-layer full-attention draft head without window metadata."""

            quant_config = None
            pretrained_config = SimpleNamespace(
                num_hidden_layers=1,
                hidden_size=1024,
                num_attention_heads=8,
                num_key_value_heads=8,
            )

            def get_num_attention_layers(self) -> int:
                return 1

        target_model_config = TargetModelConfig()
        draft_model_config = DraftModelConfig()
        target_kv_config = KvCacheConfig()
        mode = Mock()
        mode.is_external_drafter.return_value = False
        costed_windows: list[tuple[object, list[int] | None]] = []

        class RecordingKVCacheManager(KVCacheManagerV2):
            @staticmethod
            def get_cache_size_per_token(
                model_config: object, *args: object, **kwargs: object
            ) -> tuple[int, int]:
                costed_windows.append(
                    (model_config, kwargs["kv_cache_config"].max_attention_window)
                )
                return KVCacheManagerV2.get_cache_size_per_token(model_config, *args, **kwargs)

        max_batch_size = 2
        creator = object.__new__(KvCacheCreator)
        creator._kv_cache_config = target_kv_config
        creator._tokens_per_block = 64
        creator._max_seq_len = 16384
        creator._max_batch_size = max_batch_size
        creator._max_num_tokens = 128
        creator._mapping = Mock(enable_attention_dp=False, tp_size=1)
        creator._mapping.pp_layers.return_value = [0, 1, 2, 3]
        creator._mapping.is_last_pp_rank.return_value = True
        creator._speculative_config = SimpleNamespace(spec_dec_mode=mode)
        creator._model_engine = SimpleNamespace(
            model=SimpleNamespace(model_config=target_model_config)
        )
        creator._draft_model_engine = None
        creator._draft_config = draft_model_config
        creator._kv_cache_manager_cls = RecordingKVCacheManager
        creator._is_disagg = False
        creator._should_create_separate_draft_kv_cache = Mock(return_value=True)
        creator._get_effective_draft_config = Mock(return_value=draft_model_config)
        creator._get_num_draft_layers = Mock(return_value=1)

        target_kv, draft_kv = creator._get_target_and_draft_cache_costs()

        # K and V, 8 heads x 128 dims, bf16.
        layer_bytes_per_token = 2 * 8 * 128 * 2
        # Each sliding layer keeps a 512-token window per request.
        sliding_bytes_per_request = 3 * 512 * layer_bytes_per_token
        assert target_kv == CacheCost(
            slope=layer_bytes_per_token,
            intercept=sliding_bytes_per_request * max_batch_size,
        )
        assert draft_kv == CacheCost(slope=layer_bytes_per_token, intercept=0)
        target_windows = [
            windows
            for model_config, windows in costed_windows
            if model_config is target_model_config
        ]
        assert target_windows and all(
            windows == [512, 512, 16384, 512] for windows in target_windows
        )
        draft_windows = [
            windows
            for model_config, windows in costed_windows
            if model_config is draft_model_config
        ]
        assert draft_windows == [None]
        # The creator's own config is left untouched.
        assert target_kv_config.max_attention_window is None

    def test_v1_mixed_draft_build_uses_original_max_seq_len(self, mocker):
        c = _make_creator(max_gpu_total_bytes=10 * GB)
        original_max_seq_len = 16384
        c._max_seq_len = original_max_seq_len
        c._skip_est = False
        c._is_kv_cache_manager_v2 = False
        c._is_encoder_decoder = Mock(return_value=False)
        c._draft_model_engine = None
        c._kv_connector_manager = None
        c._is_disagg = False
        c._max_num_tokens = 8192
        c._max_beam_width = 1
        c._execution_stream = None
        c._enable_kv_cache_stats = Mock(return_value=False)
        c._fp8_ctx_mla_kv_len_cap = None
        c._should_create_separate_draft_kv_cache = Mock(return_value=True)
        c._speculative_config = Mock()
        c._speculative_config.spec_dec_mode.is_external_drafter.return_value = True

        draft_pretrained_config = SimpleNamespace(
            num_hidden_layers=2,
            sliding_window=4096,
            layer_types=["sliding_attention", "full_attention"],
            torch_dtype=None,
        )
        effective_draft_config = SimpleNamespace(
            pretrained_config=draft_pretrained_config,
            sparse_attention_config=None,
        )
        c._get_effective_draft_config = Mock(return_value=effective_draft_config)
        c._get_num_draft_layers = Mock(return_value=2)
        c._model_engine.model.model_config.pretrained_config.num_hidden_layers = 40
        c._fallback_if_unsupported_kv_cache_manager_v2 = Mock(
            side_effect=lambda manager_cls, *_: manager_cls
        )

        target_manager = Mock()

        def create_target_manager(*_args, **_kwargs):
            c._max_seq_len = 4096
            return target_manager

        c._create_kv_cache_manager = Mock(side_effect=create_target_manager)
        mocker.patch(
            "tensorrt_llm._torch.pyexecutor._util.get_kv_cache_manager_cls", return_value=Mock
        )
        create_manager = mocker.patch(
            "tensorrt_llm._torch.pyexecutor._util._create_kv_cache_manager", return_value=Mock()
        )

        resources = {}
        c.build_managers(resources)

        draft_kwargs = create_manager.call_args.kwargs
        assert draft_kwargs["max_seq_len"] == original_max_seq_len
        assert draft_kwargs["kv_cache_config"].max_attention_window == [4096, 16384]
        assert uses_vswa_kv_cache_layout(draft_kwargs["kv_cache_config"].max_attention_window)

    @pytest.mark.parametrize(
        ("windows", "expected"),
        [([4096, 16384], True), ([-2147483647, 16384], False)],
    )
    def test_vswa_detection_excludes_recurrent_state_windows(self, windows, expected):
        assert uses_vswa_kv_cache_layout(windows) is expected

    def test_gpu_budget_split_proportionally(self):
        total_gpu = 10 * GB
        c = _make_creator(
            max_gpu_total_bytes=total_gpu,
            total_kv_per_token=100,
            target_kv_per_token=80,
        )

        target_config, draft_config = c._split_kv_cache_budget_for_draft("max_gpu_total_bytes")

        assert draft_config is not None
        assert target_config.max_gpu_total_bytes == 8 * GB
        assert draft_config.max_gpu_total_bytes == 2 * GB
        assert target_config.host_cache_size is None
        assert c._kv_cache_config.max_gpu_total_bytes == total_gpu
        assert c._kv_cache_config.host_cache_size is None

    def test_returns_none_when_no_gpu_budget(self):
        c = _make_creator(max_gpu_total_bytes=0)

        target_config, draft_config = c._split_kv_cache_budget_for_draft("max_gpu_total_bytes")

        assert target_config is c._kv_cache_config
        assert draft_config is None

    def test_returns_none_when_draft_kv_zero(self):
        c = _make_creator(
            max_gpu_total_bytes=10 * GB,
            total_kv_per_token=100,
            target_kv_per_token=100,
        )

        target_config, draft_config = c._split_kv_cache_budget_for_draft("max_gpu_total_bytes")

        assert target_config is c._kv_cache_config
        assert draft_config is None

    def test_fixed_only_draft_uses_manager_estimated_quota(self):
        total_gpu = 10 * GB
        slot_bytes = 327_680
        configured_slots = 2_561
        configured_bytes = configured_slots * slot_bytes
        c = _make_creator(
            max_gpu_total_bytes=total_gpu,
            total_kv_per_token=80,
            target_kv_per_token=80,
            total_kv_intercept=configured_bytes,
            target_kv_intercept=0,
        )

        target_config, draft_config = c._split_kv_cache_budget_for_draft("max_gpu_total_bytes")

        assert draft_config is not None
        assert draft_config.max_gpu_total_bytes == configured_bytes
        assert target_config.max_gpu_total_bytes + draft_config.max_gpu_total_bytes == total_gpu

    @pytest.mark.parametrize(
        ("total_gpu", "expected_target", "expected_draft"),
        [(500, 300, 200), (1_000, 800, 200)],
        ids=["exact_fixed_cost", "spare_budget"],
    )
    def test_both_managers_can_be_fixed_only(self, total_gpu, expected_target, expected_draft):
        c = _make_creator(
            max_gpu_total_bytes=total_gpu,
            total_kv_per_token=0,
            target_kv_per_token=0,
            total_kv_intercept=500,
            target_kv_intercept=300,
        )

        target_config, draft_config = c._split_kv_cache_budget_for_draft("max_gpu_total_bytes")

        assert draft_config is not None
        assert target_config.max_gpu_total_bytes == expected_target
        assert draft_config.max_gpu_total_bytes == expected_draft

    def test_target_can_be_fixed_only(self):
        c = _make_creator(
            max_gpu_total_bytes=1_000,
            total_kv_per_token=20,
            target_kv_per_token=0,
            total_kv_intercept=300,
            target_kv_intercept=300,
        )

        target_config, draft_config = c._split_kv_cache_budget_for_draft("max_gpu_total_bytes")

        assert draft_config is not None
        assert target_config.max_gpu_total_bytes == 300
        assert draft_config.max_gpu_total_bytes == 700

    def test_exact_fixed_budget_rejects_a_linear_manager(self):
        c = _make_creator(
            max_gpu_total_bytes=200,
            total_kv_per_token=80,
            target_kv_per_token=80,
            total_kv_intercept=200,
        )

        with pytest.raises(ValueError, match="GPU budget"):
            c._split_kv_cache_budget_for_draft("max_gpu_total_bytes")


class TestSplitHostCacheBudgetForDraft:
    def test_host_budget_split_proportionally(self):
        total_gpu = 10 * GB
        total_host = 20 * GB
        c = _make_creator(
            max_gpu_total_bytes=total_gpu,
            host_cache_size=total_host,
            total_kv_per_token=100,
            target_kv_per_token=80,
        )

        target_config, draft_config = c._split_kv_cache_budget_for_draft("host_cache_size")

        assert draft_config is not None
        assert target_config.host_cache_size == 16 * GB
        assert draft_config.host_cache_size == 4 * GB
        assert target_config.max_gpu_total_bytes == total_gpu
        assert c._kv_cache_config.host_cache_size == total_host
        assert c._kv_cache_config.max_gpu_total_bytes == total_gpu

    def test_host_budget_not_doubled(self):
        """Regression: both managers used to receive the full host_cache_size."""
        total_host = 20 * GB
        c = _make_creator(
            max_gpu_total_bytes=10 * GB,
            host_cache_size=total_host,
            total_kv_per_token=100,
            target_kv_per_token=80,
        )

        target_config, draft_config = c._split_kv_cache_budget_for_draft("host_cache_size")

        assert draft_config is not None
        assert (target_config.host_cache_size + draft_config.host_cache_size) == total_host

    def test_host_split_without_gpu_budget_uses_slope_ratio(self):
        """V1 non-VSWA: host split must not depend on max_gpu_total_bytes."""
        total_host = 20 * GB
        c = _make_creator(
            max_gpu_total_bytes=0,
            host_cache_size=total_host,
            total_kv_per_token=100,
            target_kv_per_token=80,
        )

        target_config, draft_config = c._split_kv_cache_budget_for_draft("host_cache_size")

        assert draft_config is not None
        assert target_config.host_cache_size == 16 * GB
        assert draft_config.host_cache_size == 4 * GB

    def test_host_split_merges_into_existing_draft_config(self):
        total_gpu = 10 * GB
        total_host = 20 * GB
        c = _make_creator(
            max_gpu_total_bytes=total_gpu,
            host_cache_size=total_host,
            total_kv_per_token=100,
            target_kv_per_token=80,
        )

        target_config, draft_config = c._split_kv_cache_budget_for_draft("max_gpu_total_bytes")
        target_config, draft_config = c._split_kv_cache_budget_for_draft(
            "host_cache_size", target_config, draft_config
        )

        assert draft_config is not None
        assert draft_config.max_gpu_total_bytes == 2 * GB
        assert target_config.max_gpu_total_bytes == 8 * GB
        assert draft_config.host_cache_size == 4 * GB
        assert target_config.host_cache_size == 16 * GB
        assert c._kv_cache_config.max_gpu_total_bytes == total_gpu
        assert c._kv_cache_config.host_cache_size == total_host

    def test_host_split_after_gpu_split_is_unaffected_by_target_only_gpu_budget(self):
        """Regression: host split used to read max_gpu_total_bytes (already
        overridden to target's share by the prior GPU split) instead of the
        host budget, producing a skewed ratio. Now host split uses the host
        budget directly and stays proportional to the cache costs."""
        total_gpu = 10 * GB
        total_host = 20 * GB
        c = _make_creator(
            max_gpu_total_bytes=total_gpu,
            host_cache_size=total_host,
            total_kv_per_token=100,
            target_kv_per_token=80,
        )

        target_config, draft_config = c._split_kv_cache_budget_for_draft("max_gpu_total_bytes")
        target_config, draft_config = c._split_kv_cache_budget_for_draft(
            "host_cache_size", target_config, draft_config
        )

        assert draft_config is not None
        assert target_config.host_cache_size == 16 * GB
        assert draft_config.host_cache_size == 4 * GB

    def test_no_host_cache_leaves_none(self):
        c = _make_creator(
            max_gpu_total_bytes=10 * GB,
            host_cache_size=None,
            total_kv_per_token=100,
            target_kv_per_token=80,
        )

        target_config, draft_config = c._split_kv_cache_budget_for_draft("host_cache_size")

        assert target_config is c._kv_cache_config
        assert draft_config is None

    def test_zero_host_cache_unchanged(self):
        c = _make_creator(
            max_gpu_total_bytes=10 * GB,
            host_cache_size=0,
            total_kv_per_token=100,
            target_kv_per_token=80,
        )

        target_config, draft_config = c._split_kv_cache_budget_for_draft("host_cache_size")

        assert target_config is c._kv_cache_config
        assert draft_config is None

    @pytest.mark.parametrize("target_frac", [0.5, 0.75, 0.9, 0.95])
    def test_various_ratios(self, target_frac):
        total_host = 20 * GB
        total_kv = 1000
        target_kv = int(total_kv * target_frac)

        c = _make_creator(
            max_gpu_total_bytes=10 * GB,
            host_cache_size=total_host,
            total_kv_per_token=total_kv,
            target_kv_per_token=target_kv,
        )

        target_config, draft_config = c._split_kv_cache_budget_for_draft("host_cache_size")

        assert draft_config is not None
        assert (target_config.host_cache_size + draft_config.host_cache_size) == total_host

    def test_budgets_sum_to_original_with_gpu_and_host(self):
        total_gpu = 15 * GB
        total_host = 30 * GB
        c = _make_creator(
            max_gpu_total_bytes=total_gpu,
            host_cache_size=total_host,
            total_kv_per_token=1000,
            target_kv_per_token=700,
        )

        target_config, draft_config = c._split_kv_cache_budget_for_draft("max_gpu_total_bytes")
        target_config, draft_config = c._split_kv_cache_budget_for_draft(
            "host_cache_size", target_config, draft_config
        )

        assert draft_config is not None
        assert (target_config.max_gpu_total_bytes + draft_config.max_gpu_total_bytes) == total_gpu
        assert (target_config.host_cache_size + draft_config.host_cache_size) == total_host
        assert c._kv_cache_config.max_gpu_total_bytes == total_gpu
        assert c._kv_cache_config.host_cache_size == total_host


class TestSplitDiskCacheBudgetForDraft:
    def test_disk_budget_split_proportionally(self, tmp_path):
        total_disk = 20 * GB
        c = _make_creator(
            max_gpu_total_bytes=10 * GB,
            disk_cache_size=total_disk,
            disk_cache_path=str(tmp_path),
            total_kv_per_token=100,
            target_kv_per_token=80,
        )

        target_config, draft_config = c._split_kv_cache_budget_for_draft("disk_cache_size")

        assert draft_config is not None
        assert target_config.disk_cache_size == 16 * GB
        assert draft_config.disk_cache_size == 4 * GB
        # Both managers write into the same folder; only the quota differs.
        assert draft_config.disk_cache_path == str(tmp_path)
        assert c._kv_cache_config.disk_cache_size == total_disk

    def test_no_disk_cache_leaves_none(self):
        c = _make_creator(max_gpu_total_bytes=10 * GB)

        target_config, draft_config = c._split_kv_cache_budget_for_draft("disk_cache_size")

        assert target_config is c._kv_cache_config
        assert draft_config is None


class TestHostSplitIgnoresGpuFixedCost:
    """The fixed cost models GPU-resident state and is not host memory."""

    def test_host_split_proportional_despite_large_intercept(self):
        total_host = 10 * GB
        c = _make_creator(
            max_gpu_total_bytes=0,
            host_cache_size=total_host,
            total_kv_per_token=100,
            target_kv_per_token=80,
            total_kv_intercept=50 * GB,
        )

        target_config, draft_config = c._split_kv_cache_budget_for_draft("host_cache_size")

        assert draft_config is not None
        assert target_config.host_cache_size == 8 * GB
        assert draft_config.host_cache_size == 2 * GB

    def test_host_split_sums_to_original_despite_large_intercept(self):
        total_host = 20 * GB
        c = _make_creator(
            max_gpu_total_bytes=0,
            host_cache_size=total_host,
            total_kv_per_token=100,
            target_kv_per_token=80,
            total_kv_intercept=100 * GB,
        )

        target_config, draft_config = c._split_kv_cache_budget_for_draft("host_cache_size")

        assert draft_config is not None
        assert (target_config.host_cache_size + draft_config.host_cache_size) == total_host


class TestGpuSplitChargesFixedCost:
    """``max_gpu_total_bytes`` carries the GPU-resident fixed cost."""

    def test_gpu_split_subtracts_intercept(self):
        total_gpu = 10 * GB
        c = _make_creator(
            max_gpu_total_bytes=total_gpu,
            total_kv_per_token=100,
            target_kv_per_token=80,
            total_kv_intercept=5 * GB,
            target_kv_intercept=0,
        )

        target_config, draft_config = c._split_kv_cache_budget_for_draft("max_gpu_total_bytes")

        assert draft_config is not None
        assert draft_config.max_gpu_total_bytes == 6 * GB
        assert target_config.max_gpu_total_bytes == 4 * GB

    def test_gpu_split_infeasible_raises(self):
        """A GPU budget too small for fixed cost must fail fast."""
        c = _make_creator(
            max_gpu_total_bytes=1 * GB,
            total_kv_per_token=100,
            target_kv_per_token=80,
            total_kv_intercept=2 * GB,
        )

        with pytest.raises(ValueError, match="GPU budget"):
            c._split_kv_cache_budget_for_draft("max_gpu_total_bytes")

    def test_gpu_raise_does_not_block_subsequent_host_split(self):
        total_host = 10 * GB
        c = _make_creator(
            max_gpu_total_bytes=0,
            host_cache_size=total_host,
            total_kv_per_token=100,
            target_kv_per_token=80,
            total_kv_intercept=2 * GB,
        )

        target_config, draft_config = c._split_kv_cache_budget_for_draft("host_cache_size")

        assert draft_config is not None
        assert target_config.host_cache_size == 8 * GB
        assert draft_config.host_cache_size == 2 * GB


class TestBuildManagersBudgetGates:
    """Which splits build_managers applies, as opposed to how they divide."""

    @staticmethod
    def _make_build_creator(
        total_offload: int, total_gpu: int, disk_path: str | None = None
    ) -> KvCacheCreator:
        """Creator whose host and disk tiers both carry total_offload."""
        c = _make_creator(
            max_gpu_total_bytes=total_gpu,
            host_cache_size=total_offload,
            disk_cache_size=total_offload,
            disk_cache_path=disk_path,
            total_kv_per_token=100,
            target_kv_per_token=80,
        )
        c._skip_est = False
        c._draft_model_engine = None
        c._kv_connector_manager = None
        c._is_kv_cache_manager_v2 = True
        # build_managers carries this onto the real manager, not the probes.
        c._fp8_ctx_mla_kv_len_cap = None
        c._model_engine.model.model_config.is_encoder_decoder = False
        c._should_create_separate_draft_kv_cache = Mock(return_value=True)
        c._needs_gpu_kv_cache_budget_split = Mock(return_value=True)
        c._create_kv_cache_manager = Mock(return_value=Mock())
        c._create_one_model_draft_kv_cache_manager = Mock(return_value=Mock())
        return c

    @staticmethod
    def _configs_passed_to_managers(c: KvCacheCreator):
        return (
            c._create_kv_cache_manager.call_args.kwargs["kv_cache_config_override"],
            c._create_one_model_draft_kv_cache_manager.call_args.kwargs["kv_cache_config_override"],
        )

    @pytest.mark.parametrize("budget_attr", OFFLOAD_TIER_BUDGET_ATTRS)
    @pytest.mark.parametrize("estimating_kv_cache", [False, True])
    def test_no_manager_receives_a_full_offload_budget(
        self, budget_attr, estimating_kv_cache, tmp_path
    ):
        """No manager may be handed a whole offload tier, estimating or not.

        Each one reserves what it is given, so two full-budget managers make a
        run need twice the configured host memory or disk space.
        """
        total_offload = 20 * GB
        c = self._make_build_creator(total_offload, total_gpu=10 * GB, disk_path=str(tmp_path))

        c.build_managers({}, estimating_kv_cache=estimating_kv_cache)

        target_config, draft_config = self._configs_passed_to_managers(c)
        assert getattr(target_config, budget_attr) != total_offload
        assert getattr(draft_config, budget_attr) != total_offload

    @pytest.mark.parametrize("budget_attr", OFFLOAD_TIER_BUDGET_ATTRS)
    def test_offload_budget_is_split_between_target_and_draft(self, budget_attr, tmp_path):
        total_offload = 20 * GB
        c = self._make_build_creator(total_offload, total_gpu=10 * GB, disk_path=str(tmp_path))

        c.build_managers({}, estimating_kv_cache=False)

        target_config, draft_config = self._configs_passed_to_managers(c)
        target_share = getattr(target_config, budget_attr)
        draft_share = getattr(draft_config, budget_attr)
        assert target_share + draft_share == total_offload
        assert draft_share < total_offload

    @pytest.mark.parametrize("budget_attr", OFFLOAD_TIER_BUDGET_ATTRS)
    def test_estimation_managers_drop_explicit_offload_tiers(self, budget_attr, tmp_path):
        """An explicit offload tier would reserve capacity a probe cannot fill."""
        c = self._make_build_creator(20 * GB, total_gpu=10 * GB, disk_path=str(tmp_path))

        c.build_managers({}, estimating_kv_cache=True)

        target_config, draft_config = self._configs_passed_to_managers(c)
        assert getattr(target_config, budget_attr) is None
        assert getattr(draft_config, budget_attr) is None

    def test_gpu_budget_split_stays_skipped_during_estimation(self, tmp_path):
        """Estimation sizes GPU pools from max_tokens, so it keeps the budget whole."""
        c = self._make_build_creator(20 * GB, total_gpu=10 * GB, disk_path=str(tmp_path))

        c.build_managers({}, estimating_kv_cache=True)

        target_config, _ = self._configs_passed_to_managers(c)
        assert target_config.max_gpu_total_bytes == 10 * GB
        c._needs_gpu_kv_cache_budget_split.assert_not_called()

    def _make_two_model_creator(self, total_offload: int, disk_path: str) -> KvCacheCreator:
        """V2 creator whose draft cache comes from a separate engine."""
        c = self._make_build_creator(total_offload, total_gpu=10 * GB, disk_path=disk_path)
        c._draft_model_engine = Mock()
        c._should_create_separate_draft_kv_cache = Mock(return_value=False)
        # V2 sizes two-model GPU pools per manager from the whole budget.
        c._needs_gpu_kv_cache_budget_split = Mock(return_value=False)
        return c

    @pytest.mark.parametrize("budget_attr", OFFLOAD_TIER_BUDGET_ATTRS)
    def test_two_model_offload_budget_is_split(self, budget_attr, tmp_path):
        """A separate draft engine divides the offload budgets too.

        Its manager lives alongside the target's, so handing both the whole
        budget would reserve it twice.
        """
        total_offload = 20 * GB
        c = self._make_two_model_creator(total_offload, str(tmp_path))

        c.build_managers({}, estimating_kv_cache=False)

        calls = c._create_kv_cache_manager.call_args_list
        target_config = calls[0].kwargs["kv_cache_config_override"]
        draft_config = calls[1].kwargs["kv_cache_config_override"]
        target_share = getattr(target_config, budget_attr)
        draft_share = getattr(draft_config, budget_attr)
        assert target_share + draft_share == total_offload
        assert 0 < draft_share < total_offload

    def test_two_model_keeps_the_gpu_budget_whole(self, tmp_path):
        """Each two-model manager sizes its GPU pools from the full budget."""
        c = self._make_two_model_creator(20 * GB, str(tmp_path))

        c.build_managers({}, estimating_kv_cache=False)

        calls = c._create_kv_cache_manager.call_args_list
        for call in calls:
            config = call.kwargs["kv_cache_config_override"]
            assert config.max_gpu_total_bytes == 10 * GB

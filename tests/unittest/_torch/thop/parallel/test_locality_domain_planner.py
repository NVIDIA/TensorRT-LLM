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
"""
Unit tests for locality domain Execution Planner.

Tests cover:
- Policy dataclass behavior (frozen, defaults)
- PartitionPlan / LinearPartitionPlan creation
- PartitionedTensorLayout metadata
- LocalityDomainExecutionPlanner enable/disable decisions
- Singleton DISABLED_PLAN
"""

from typing import Optional
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.locality_domain.layout import (
    make_bf16_linear_output_layout,
    make_nvfp4_linear_output_layout,
)
from tensorrt_llm._torch.locality_domain.policy import (
    DISABLED_PLAN,
    LinearPartitionPlan,
    LocalityDomainExecutionPlanner,
    LocalityDomainPolicy,
    PartitionPlan,
)
from tensorrt_llm._torch.utils import model_extra_attrs


@pytest.fixture(autouse=True)
def _enable_cutedsl_extended(monkeypatch):
    monkeypatch.setattr(
        "tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_EXTENDED_AVAILABLE", True
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeQuantMode:
    """Minimal stub for quant_config.layer_quant_mode."""

    def __init__(self, nvfp4: bool = True):
        self._nvfp4 = nvfp4

    def has_nvfp4(self):
        return self._nvfp4

    def has_any_quant(self, exclude_kv_cache=False):
        return self._nvfp4


class _FakeQuantConfig:
    """Minimal stub for QuantConfig."""

    def __init__(self, nvfp4: bool = True, quant_algo=None):
        self.layer_quant_mode = _FakeQuantMode(nvfp4)
        self.quant_algo = quant_algo


# Import WeightMode lazily to avoid heavy module load
def _vanilla_weight_mode():
    from tensorrt_llm._torch.modules.linear import WeightMode

    return WeightMode.VANILLA


def _fused_qkv_weight_mode():
    from tensorrt_llm._torch.modules.linear import WeightMode

    return WeightMode.FUSED_QKV_LINEAR


def _fused_gate_up_weight_mode():
    from tensorrt_llm._torch.modules.linear import WeightMode

    return WeightMode.FUSED_GATE_UP_LINEAR


# ---------------------------------------------------------------------------
# Policy dataclass tests
# ---------------------------------------------------------------------------


class TestLocalityDomainPolicy:
    """Tests for LocalityDomainPolicy frozen dataclass."""

    def test_default_values(self):
        policy = LocalityDomainPolicy()
        assert not policy.enabled
        assert policy.num_partitions == 2
        assert "nvfp4_linear" in policy.allowed_ops
        assert "bf16_linear" in policy.allowed_ops
        assert "bf16_bmm" in policy.allowed_ops
        assert "nvfp4_moe" in policy.allowed_ops
        assert "bf16_moe" in policy.allowed_ops

    def test_enabled(self):
        policy = LocalityDomainPolicy(enabled=True)
        assert policy.enabled

    def test_frozen(self):
        policy = LocalityDomainPolicy()
        with pytest.raises(AttributeError):
            policy.enabled = True

    def test_custom_partitions_rejected(self):
        # Runtime only supports num_partitions=2
        with pytest.raises(ValueError, match="only supports num_partitions=2"):
            LocalityDomainPolicy(num_partitions=4)

    def test_default_partitions(self):
        policy = LocalityDomainPolicy()
        assert policy.num_partitions == 2

    def test_custom_allowed_ops(self):
        policy = LocalityDomainPolicy(allowed_ops=frozenset({"nvfp4_linear"}))
        assert "nvfp4_moe" not in policy.allowed_ops


# ---------------------------------------------------------------------------
# PartitionPlan tests
# ---------------------------------------------------------------------------


class TestPartitionPlan:
    """Tests for PartitionPlan and LinearPartitionPlan."""

    def test_disabled_plan(self):
        plan = PartitionPlan(enabled=False)
        assert not plan.enabled
        assert plan.num_partitions == 2
        assert plan.backend == "cutedsl"

    def test_enabled_plan(self):
        plan = PartitionPlan(enabled=True, num_partitions=2)
        assert plan.enabled
        assert plan.merge_kind == "concat"

    def test_linear_partition_plan(self):
        plan = LinearPartitionPlan(enabled=True)
        assert plan.partition_axis == 0
        assert isinstance(plan, PartitionPlan)

    def test_singleton_disabled_plan(self):
        assert not DISABLED_PLAN.enabled
        assert DISABLED_PLAN.reason_if_disabled is not None

    def test_frozen(self):
        plan = LinearPartitionPlan(enabled=True)
        with pytest.raises(AttributeError):
            plan.enabled = False


# ---------------------------------------------------------------------------
# LocalityDomainExecutionPlanner tests
# ---------------------------------------------------------------------------


class TestPartitionedTensorLayout:
    """Tests for logical/padded partition metadata."""

    def test_nvfp4_linear_layout_slices(self):
        layout = make_nvfp4_linear_output_layout(
            out_features=7168,
            in_features=2048,
            num_partitions=2,
        )
        assert layout.logical_shape == (7168, 2048)
        assert layout.padded_shape == (7168, 2048)
        assert layout.partition_axis_slice(0, padded=False) == slice(0, 3584)
        assert layout.partition_axis_slice(1, padded=True) == slice(3584, 7168)
        assert layout.disabled_reason_for_padding_free_split() is None

    def test_nvfp4_linear_layout_reports_padding_gap(self):
        layout = make_nvfp4_linear_output_layout(
            out_features=7170,
            in_features=2048,
            num_partitions=2,
        )
        assert layout.logical_shape == (7170, 2048)
        assert layout.padded_shape == (7200, 2048)
        assert "NVFP4 row alignment" in layout.disabled_reason_for_padding_free_split()

    def test_bf16_linear_layout_requires_aligned_partition_rows(self):
        layout = make_bf16_linear_output_layout(
            out_features=2112,
            in_features=7168,
            num_partitions=2,
        )
        assert layout.logical_shape == layout.padded_shape
        assert layout.partition_axis_slice(0, padded=False) == slice(0, 1056)
        assert layout.partition_axis_slice(1, padded=False) == slice(1056, 2112)
        assert layout.disabled_reason_for_padding_free_split() is None

        unaligned = make_bf16_linear_output_layout(
            out_features=2114,
            in_features=7168,
            num_partitions=2,
        )
        assert "BF16 locality domain row alignment" in (
            unaligned.disabled_reason_for_padding_free_split()
        )


class TestLocalityDomainExecutionPlanner:
    """Tests for plan_linear() enable/disable decisions."""

    def _plan(
        self,
        policy=None,
        in_features=2048,
        out_features=7168,
        quant_config=None,
        weight_mode=None,
        **kwargs,
    ):
        """Helper: run plan_linear with sensible defaults."""
        if policy is None:
            policy = LocalityDomainPolicy(enabled=True)
        if quant_config is None:
            quant_config = _FakeQuantConfig(nvfp4=True)
        if weight_mode is None:
            weight_mode = _vanilla_weight_mode()
        planner = LocalityDomainExecutionPlanner(policy)
        return planner.plan_linear(in_features, out_features, quant_config, weight_mode, **kwargs)

    # --- Policy checks ---

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_disabled_policy_disables(self, mock_locality_domain):
        plan = self._plan(policy=LocalityDomainPolicy(enabled=False))
        assert not plan.enabled
        assert "disabled" in plan.reason_if_disabled

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_enabled_policy_enables_when_all_conditions_met(self, mock_locality_domain):
        plan = self._plan(policy=LocalityDomainPolicy(enabled=True))
        assert plan.enabled
        assert plan.backend == "cutedsl"
        assert plan.num_partitions == 2

    # --- Hardware check ---

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=False
    )
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_locality_domain_not_supported_disables(self, mock_locality_domain):
        plan = self._plan()
        assert not plan.enabled
        assert "hardware" in plan.reason_if_disabled.lower()

    # --- Op type check ---

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_op_not_allowed_disables(self, mock_locality_domain):
        policy = LocalityDomainPolicy(enabled=True, allowed_ops=frozenset({"nvfp4_moe"}))
        plan = self._plan(policy=policy)
        assert not plan.enabled
        assert "allowed_ops" in plan.reason_if_disabled

    # --- Quantization check ---

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_non_nvfp4_quant_disables(self, mock_locality_domain):
        plan = self._plan(quant_config=_FakeQuantConfig(nvfp4=False))
        assert not plan.enabled
        assert "NVFP4" in plan.reason_if_disabled

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_no_quant_config_disables(self, mock_locality_domain):
        """quant_config=None should disable locality domain (no quantization)."""
        planner = LocalityDomainExecutionPlanner(LocalityDomainPolicy(enabled=True))
        plan = planner.plan_linear(2048, 7168, None, _vanilla_weight_mode())
        assert not plan.enabled

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_nvfp4_arc_disables(self, mock_locality_domain):
        from tensorrt_llm.quantization.mode import QuantAlgo

        plan = self._plan(quant_config=_FakeQuantConfig(nvfp4=True, quant_algo=QuantAlgo.NVFP4_ARC))
        assert not plan.enabled
        assert "NVFP4_ARC" in plan.reason_if_disabled

    # --- CuteDSL availability ---

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", False)
    def test_cutedsl_unavailable_disables(self, mock_locality_domain):
        plan = self._plan()
        assert not plan.enabled
        assert "CuteDSL" in plan.reason_if_disabled

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_cutedsl_extended_unavailable_disables(self, mock_locality_domain, monkeypatch):
        monkeypatch.setattr(
            "tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_EXTENDED_AVAILABLE", False
        )
        plan = self._plan()
        assert not plan.enabled
        assert "extended" in plan.reason_if_disabled

    # --- Weight mode check ---

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_non_vanilla_weight_mode_disables(self, mock_locality_domain):
        plan = self._plan(weight_mode=_fused_qkv_weight_mode())
        assert not plan.enabled
        assert "VANILLA" in plan.reason_if_disabled

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_nvfp4_fused_gate_up_weight_mode_enables(self, mock_locality_domain):
        plan = self._plan(weight_mode=_fused_gate_up_weight_mode())
        assert plan.enabled
        assert plan.op_kind == "nvfp4_linear"

    # --- Divisibility check ---

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_odd_out_features_disables(self, mock_locality_domain):
        plan = self._plan(out_features=7169)
        assert not plan.enabled
        assert "divisible" in plan.reason_if_disabled

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_unaligned_partition_out_features_disables(self, mock_locality_domain):
        plan = self._plan(out_features=7170)
        assert not plan.enabled
        assert "row alignment" in plan.reason_if_disabled

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_unaligned_nvfp4_packed_k_disables(self, mock_locality_domain):
        plan = self._plan(in_features=2064)
        assert not plan.enabled
        assert "packed-K alignment" in plan.reason_if_disabled

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_custom_partition_count_rejected(self, mock_locality_domain):
        # Runtime only supports num_partitions=2
        with pytest.raises(ValueError, match="only supports num_partitions=2"):
            LocalityDomainPolicy(num_partitions=4)

    # --- Enabled plan properties ---

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_enabled_plan_properties(self, mock_locality_domain):
        plan = self._plan()
        assert plan.enabled
        assert plan.backend == "cutedsl"
        assert plan.merge_kind == "concat"
        assert plan.partition_axis == 0
        assert plan.layout == make_nvfp4_linear_output_layout(7168, 2048, 2)
        assert plan.op_kind == "nvfp4_linear"
        assert plan.reason_if_disabled is None

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    @pytest.mark.parametrize(
        (
            "weight_mode_factory",
            "in_features",
            "out_features",
            "enable_locality_domain_bf16_linear",
            "expected_enabled",
            "reason_substrings",
        ),
        [
            pytest.param(None, 7168, 2112, False, False, (), id="requires-opt-in"),
            pytest.param(None, 7168, 2112, True, True, (), id="vanilla"),
            pytest.param(
                _fused_gate_up_weight_mode,
                7168,
                2112,
                True,
                True,
                (),
                id="fused-gate-up",
            ),
            pytest.param(
                _fused_qkv_weight_mode,
                7168,
                2112,
                True,
                False,
                ("FUSED_QKV_LINEAR", "not supported for bf16_linear"),
                id="fused-qkv",
            ),
            pytest.param(None, 7169, 2112, True, False, ("alignment",), id="unaligned-k"),
            pytest.param(None, 7168, 2114, True, False, ("alignment",), id="unaligned-n"),
        ],
    )
    def test_bf16_linear_decision_table(
        self,
        mock_locality_domain,
        weight_mode_factory,
        in_features,
        out_features,
        enable_locality_domain_bf16_linear,
        expected_enabled,
        reason_substrings,
    ):
        weight_mode = (
            _vanilla_weight_mode() if weight_mode_factory is None else weight_mode_factory()
        )
        plan = self._plan(
            quant_config=_FakeQuantConfig(nvfp4=False),
            in_features=in_features,
            out_features=out_features,
            weight_mode=weight_mode,
            dtype=torch.bfloat16,
            use_cute_dsl_bf16_gemm=True,
            enable_locality_domain_bf16_linear=enable_locality_domain_bf16_linear,
        )

        assert plan.enabled is expected_enabled
        if expected_enabled:
            assert plan.op_kind == "bf16_linear"
            assert plan.layout == make_bf16_linear_output_layout(out_features, in_features, 2)
        else:
            for reason in reason_substrings:
                assert reason in plan.reason_if_disabled

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    @pytest.mark.skip(reason="requires the Linear/model_config wire-up (call-site PR)")
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_bf16_fused_gate_up_requires_canonical_mapping(self, mock_locality_domain):
        from tensorrt_llm._torch.modules.linear import Linear, WeightMode, WeightsLoadingConfig

        linear = Linear(
            in_features=7168,
            out_features=2112,
            bias=False,
            dtype=torch.bfloat16,
            quant_config=_FakeQuantConfig(nvfp4=False),
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_GATE_UP_LINEAR
            ),
            skip_create_weights_in_init=True,
            use_cute_dsl_bf16_gemm=True,
            enable_locality_domain_bf16_linear=True,
            fused_weight_shard_indices_mapping={
                "up": (0, 1056),
                "gate": (1056, 1056),
            },
            locality_domain_policy=LocalityDomainPolicy(enabled=True),
        )

        linear._replan_locality_domain()

        assert not linear.partition_plan.enabled
        assert "canonical [gate | up]" in linear.partition_plan.reason_if_disabled

    @pytest.mark.skip(reason="requires the Linear/model_config wire-up (call-site PR)")
    def test_linear_reads_policy_from_model_extra_attrs(self):
        from tensorrt_llm._torch.modules.linear import Linear

        with model_extra_attrs({"locality_domain_policy": LocalityDomainPolicy(enabled=False)}):
            linear = Linear(
                in_features=2048,
                out_features=7168,
                bias=False,
                quant_config=_FakeQuantConfig(nvfp4=True),
                skip_create_weights_in_init=True,
            )
            linear._replan_locality_domain()

        assert not linear.partition_plan.enabled
        assert "disabled" in linear.partition_plan.reason_if_disabled

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    @pytest.mark.skip(reason="requires the Linear/model_config wire-up (call-site PR)")
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_linear_copies_backend_list_before_locality_domain_append(self, mock_locality_domain):
        from tensorrt_llm._torch.modules.linear import Linear

        shared_backends = ["cutlass", "cublaslt", "cuda_core"]
        with model_extra_attrs(
            {
                "locality_domain_policy": LocalityDomainPolicy(enabled=True),
                "nvfp4_gemm_allowed_backends": shared_backends,
            }
        ):
            linear = Linear(
                in_features=2048,
                out_features=7168,
                bias=False,
                quant_config=_FakeQuantConfig(nvfp4=True),
                skip_create_weights_in_init=True,
            )
            linear._replan_locality_domain()

        assert shared_backends == ["cutlass", "cublaslt", "cuda_core"]
        assert linear.nvfp4_allowed_backends is not shared_backends
        assert linear.nvfp4_allowed_backends == [
            "cutlass",
            "cublaslt",
            "cuda_core",
            "cutedsl",
        ]

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    @pytest.mark.skip(reason="requires the Linear/model_config wire-up (call-site PR)")
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    @pytest.mark.parametrize("start_nvfp4", [True, False], ids=["nvfp4-to-bf16", "bf16-to-nvfp4"])
    def test_linear_replans_after_quant_override(self, mock_locality_domain, start_nvfp4):
        from tensorrt_llm._torch.modules.linear import Linear

        linear = Linear(
            in_features=7168,
            out_features=2112,
            bias=False,
            dtype=torch.bfloat16,
            quant_config=_FakeQuantConfig(nvfp4=start_nvfp4),
            skip_create_weights_in_init=True,
            use_cute_dsl_bf16_gemm=True,
            enable_locality_domain_bf16_linear=True,
            locality_domain_policy=LocalityDomainPolicy(enabled=True),
        )
        linear._replan_locality_domain()
        initial_runtime = linear._locality_domain_runtime
        initial_op_kind = "nvfp4_linear" if start_nvfp4 else "bf16_linear"
        final_op_kind = "bf16_linear" if start_nvfp4 else "nvfp4_linear"
        assert linear.partition_plan.op_kind == initial_op_kind
        assert linear._locality_domain_added_cutedsl_backend is start_nvfp4

        linear.quant_config = _FakeQuantConfig(nvfp4=not start_nvfp4)
        linear._replan_locality_domain()

        assert linear.partition_plan.op_kind == final_op_kind
        assert linear._locality_domain_runtime is not initial_runtime
        assert linear._locality_domain_added_cutedsl_backend is not start_nvfp4
        assert ("cutedsl" in linear.nvfp4_allowed_backends) is (not start_nvfp4)

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    @pytest.mark.skip(reason="requires the Linear/model_config wire-up (call-site PR)")
    @patch("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    def test_linear_replan_after_sharding_requires_reload(self, mock_locality_domain):
        from tensorrt_llm._torch.modules.linear import Linear

        linear = Linear(
            in_features=7168,
            out_features=2112,
            bias=False,
            dtype=torch.bfloat16,
            quant_config=_FakeQuantConfig(nvfp4=True),
            skip_create_weights_in_init=True,
            locality_domain_policy=LocalityDomainPolicy(enabled=True),
        )
        linear._replan_locality_domain()
        original_plan = linear.partition_plan
        original_runtime = linear._locality_domain_runtime
        shards = [{"weight": object()}, {"weight": object()}]
        linear._locality_domain_weight_shards = shards
        linear.quant_config = _FakeQuantConfig(nvfp4=False)

        with pytest.raises(RuntimeError, match="reload"):
            linear._replan_locality_domain()

        assert linear.partition_plan is original_plan
        assert linear._locality_domain_runtime is original_runtime
        assert linear._locality_domain_weight_shards is shards

    @pytest.mark.skip(reason="requires the Linear/model_config wire-up (call-site PR)")
    def test_model_config_exports_locality_domain_policy(self):
        from tensorrt_llm._torch.model_config import ModelConfig

        policy = LocalityDomainPolicy(enabled=False)
        config = ModelConfig(locality_domain_policy=policy)

        assert config.extra_attrs["locality_domain_policy"] is policy


# ---------------------------------------------------------------------------
# MoE planner tests
# ---------------------------------------------------------------------------


class _FakeMoeQuantMode:
    """Stub for MoE quant_config.quant_mode."""

    def __init__(self, nvfp4: bool = True, any_quant: Optional[bool] = None):
        self._nvfp4 = nvfp4
        self._any_quant = nvfp4 if any_quant is None else any_quant

    def has_nvfp4(self):
        return self._nvfp4

    def has_any_quant(self):
        return self._any_quant


class _FakeMoeQuantConfig:
    """Stub for MoE QuantConfig."""

    def __init__(self, nvfp4: bool = True, any_quant: Optional[bool] = None):
        self.quant_mode = _FakeMoeQuantMode(nvfp4, any_quant)


class TestLocalityDomainMoePlanner:
    """Tests for plan_moe() enable/disable decisions."""

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    def test_moe_enabled_when_nvfp4(self, mock_locality_domain):
        planner = LocalityDomainExecutionPlanner(LocalityDomainPolicy(enabled=True))
        plan = planner.plan_moe(_FakeMoeQuantConfig(nvfp4=True))
        assert plan.enabled
        assert plan.backend == "cutedsl"
        assert plan.merge_kind == "none"

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    def test_moe_enabled_when_bf16(self, mock_locality_domain):
        planner = LocalityDomainExecutionPlanner(LocalityDomainPolicy(enabled=True))
        plan = planner.plan_moe(None, dtype_activation=torch.bfloat16)
        assert plan.enabled
        assert plan.backend == "cutedsl"
        assert plan.merge_kind == "none"

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    def test_moe_disabled_without_fused_finalize(self, mock_locality_domain):
        planner = LocalityDomainExecutionPlanner(LocalityDomainPolicy(enabled=True))
        plan = planner.plan_moe(_FakeMoeQuantConfig(nvfp4=True), use_fused_finalize=False)
        assert not plan.enabled
        assert "fused finalize" in plan.reason_if_disabled

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    def test_moe_disabled_when_backend_not_cutedsl(self, mock_locality_domain):
        planner = LocalityDomainExecutionPlanner(LocalityDomainPolicy(enabled=True))
        plan = planner.plan_moe(_FakeMoeQuantConfig(nvfp4=True), moe_backend="CUTLASS")
        assert not plan.enabled
        assert "CuteDSL backend" in plan.reason_if_disabled

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    def test_moe_disabled_when_quantized_but_not_nvfp4(self, mock_locality_domain):
        planner = LocalityDomainExecutionPlanner(LocalityDomainPolicy(enabled=True))
        plan = planner.plan_moe(_FakeMoeQuantConfig(nvfp4=False, any_quant=True))
        assert not plan.enabled

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    def test_moe_disabled_when_bf16_not_in_allowed_ops(self, mock_locality_domain):
        policy = LocalityDomainPolicy(
            enabled=True, allowed_ops=frozenset({"nvfp4_linear", "nvfp4_moe"})
        )
        planner = LocalityDomainExecutionPlanner(policy)
        plan = planner.plan_moe(None, dtype_activation=torch.bfloat16)
        assert not plan.enabled
        assert "bf16_moe" in plan.reason_if_disabled

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    def test_moe_disabled_when_unquantized_not_bf16(self, mock_locality_domain):
        planner = LocalityDomainExecutionPlanner(LocalityDomainPolicy(enabled=True))
        plan = planner.plan_moe(None, dtype_activation=torch.float16)
        assert not plan.enabled
        assert "bfloat16" in plan.reason_if_disabled

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    def test_moe_disabled_when_policy_disabled(self, mock_locality_domain):
        planner = LocalityDomainExecutionPlanner(LocalityDomainPolicy(enabled=False))
        plan = planner.plan_moe(_FakeMoeQuantConfig(nvfp4=True))
        assert not plan.enabled
        assert "disabled" in plan.reason_if_disabled

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=False
    )
    def test_moe_disabled_when_no_hardware(self, mock_locality_domain):
        planner = LocalityDomainExecutionPlanner(LocalityDomainPolicy(enabled=True))
        plan = planner.plan_moe(_FakeMoeQuantConfig(nvfp4=True))
        assert not plan.enabled

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    def test_moe_disabled_when_not_in_allowed_ops(self, mock_locality_domain):
        policy = LocalityDomainPolicy(enabled=True, allowed_ops=frozenset({"nvfp4_linear"}))
        planner = LocalityDomainExecutionPlanner(policy)
        plan = planner.plan_moe(_FakeMoeQuantConfig(nvfp4=True))
        assert not plan.enabled

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    def test_moe_disabled_without_cutedsl_extended(self, mock_locality_domain, monkeypatch):
        monkeypatch.setattr(
            "tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_EXTENDED_AVAILABLE", False
        )
        planner = LocalityDomainExecutionPlanner(LocalityDomainPolicy(enabled=True))
        plan = planner.plan_moe(_FakeMoeQuantConfig(nvfp4=True))
        assert not plan.enabled
        assert "extended" in plan.reason_if_disabled


class TestLocalityDomainBf16BmmPlanner:
    """Tests for plan_bf16_bmm() enable/disable decisions."""

    @patch(
        "tensorrt_llm._torch.locality_domain_utils.is_locality_domain_enabled", return_value=True
    )
    @pytest.mark.parametrize(
        (
            "allowed_ops",
            "dtype",
            "use_cute_dsl_bf16_bmm",
            "expected_enabled",
            "reason",
        ),
        [
            pytest.param(None, torch.bfloat16, True, True, None, id="enabled"),
            pytest.param(
                frozenset({"bf16_linear"}),
                torch.bfloat16,
                True,
                False,
                "allowed_ops",
                id="op-not-allowed",
            ),
            pytest.param(None, torch.float16, True, False, None, id="wrong-dtype"),
            pytest.param(None, torch.bfloat16, False, False, None, id="backend-disabled"),
        ],
    )
    def test_bf16_bmm_decision_table(
        self,
        mock_locality_domain,
        allowed_ops,
        dtype,
        use_cute_dsl_bf16_bmm,
        expected_enabled,
        reason,
    ):
        policy_kwargs = {} if allowed_ops is None else {"allowed_ops": allowed_ops}
        planner = LocalityDomainExecutionPlanner(
            LocalityDomainPolicy(enabled=True, **policy_kwargs)
        )
        plan = planner.plan_bf16_bmm(
            dtype=dtype,
            use_cute_dsl_bf16_bmm=use_cute_dsl_bf16_bmm,
        )

        assert plan.enabled is expected_enabled
        if expected_enabled:
            assert plan.merge_kind == "concat"
        elif reason is not None:
            assert reason in plan.reason_if_disabled

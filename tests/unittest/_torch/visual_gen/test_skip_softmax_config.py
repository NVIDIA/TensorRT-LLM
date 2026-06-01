# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SkipSoftmax visual generation config and wiring."""

import math
from typing import Final

import pytest

from tensorrt_llm.llmapi.llm_args import SkipSoftmaxAttentionConfig
from tensorrt_llm.visual_gen.args import AttentionConfig, QuantAttentionConfig
from tensorrt_llm.visual_gen.sparse_attention import (
    SkipSoftmaxConfig,
    SkipSoftmaxFormula,
    apply_skip_softmax_overrides,
    set_skip_softmax_enabled,
)

# =============================================================================
# SkipSoftmaxFormula — accepts both log_a (diffusion) and a (LLM) formats
# =============================================================================


class TestSkipSoftmaxFormulaFormats:
    def test_accepts_log_a(self):
        """Diffusion format: log_a stored directly."""
        f = SkipSoftmaxFormula(log_a=-14.409, b=37.457)
        assert f.log_a == pytest.approx(-14.409)
        assert f.b == pytest.approx(37.457)

    def test_accepts_linear_a_and_normalizes(self):
        """LLM format: a is normalized to log_a = log(a)."""
        f = SkipSoftmaxFormula(a=7e-5, b=7.929109)
        assert f.log_a == pytest.approx(math.log(7e-5))
        assert f.b == pytest.approx(7.929109)

    def test_rejects_both_log_a_and_a(self):
        """Specifying both is ambiguous — error rather than silently pick one."""
        with pytest.raises(ValueError, match="not both"):
            SkipSoftmaxFormula(log_a=-10.0, a=999.0, b=5.0)

    def test_rejects_non_positive_a(self):
        """Linear 'a' must be positive (log of 0/negative is undefined)."""
        with pytest.raises(ValueError, match="must be positive"):
            SkipSoftmaxFormula(a=0.0, b=5.0)
        with pytest.raises(ValueError, match="must be positive"):
            SkipSoftmaxFormula(a=-1.0, b=5.0)


# =============================================================================
# SkipSoftmaxConfig construction
# =============================================================================


class TestSkipSoftmaxConfigConstruction:
    def test_attention_config_from_full_dict(self):
        """Discriminator dispatch from a raw dict populating every user-facing field.

        Subsumes the per-field assignment tests (Pydantic's contract) and the
        narrower dict-construction variants. Calibration coefficients
        (``formula``, ``layer_overrides``) are intentionally *not* user-facing
        — they are loaded from ModelOpt YAML via ``with_calibration`` and
        live in private attrs; passing them here would (correctly) fail
        StrictBaseModel validation.
        """
        cfg = AttentionConfig(
            **{
                "backend": "TRTLLM",
                "sparse_attention_config": {
                    "algorithm": "skip_softmax",
                    "threshold_scale_factor": 5000.0,
                    "target_sparsity": 0.5,
                    "first_dense_steps": 16,
                },
            }
        )
        sc = cfg.sparse_attention_config
        assert isinstance(sc, SkipSoftmaxConfig)
        assert sc.algorithm == "skip_softmax"
        assert sc.threshold_scale_factor == 5000.0
        assert sc.target_sparsity == 0.5
        assert sc.first_dense_steps == 16

    def test_attention_config_rejects_calibration_fields(self):
        """``formula`` and ``layer_overrides`` are not part of the user surface.

        Guards against accidentally re-exposing calibration internals on the
        Pydantic model. ``StrictBaseModel`` rejects unknown fields with
        ``ValidationError``.
        """
        from pydantic import ValidationError

        for field, value in [
            ("formula", {"log_a": math.log(0.0003), "b": 7.5}),
            ("layer_overrides", {"transformer_blocks.0.*": 0}),
        ]:
            with pytest.raises(ValidationError):
                AttentionConfig(
                    backend="TRTLLM",
                    sparse_attention_config={
                        "algorithm": "skip_softmax",
                        "threshold_scale_factor": 5000.0,
                        field: value,
                    },
                )

    def test_attention_config_round_trip(self):
        """``model_dump()`` → ``AttentionConfig(**dumped)`` preserves all fields.

        Guards against future serialization regressions on the user-facing
        surface: dropped discriminator, alias drift on the inherited llmapi
        fields. Calibration state is private and intentionally outside the
        dump/load contract.
        """
        original = AttentionConfig(
            backend="TRTLLM",
            sparse_attention_config=SkipSoftmaxConfig(
                threshold_scale_factor=5000.0,
                target_sparsity=0.5,
            ),
        )
        dumped = original.model_dump()
        rehydrated = AttentionConfig(**dumped)
        assert rehydrated.model_dump() == dumped

    def test_public_overrides_preserve_first_dense_steps_when_omitted(self):
        """ModelOpt calibration keeps its dense prefix unless the user sets one."""
        loaded = SkipSoftmaxConfig.with_calibration(
            target_sparsity=0.5,
            first_dense_steps=12,
            formula=SkipSoftmaxFormula(log_a=math.log(0.0003), b=7.5),
        )
        assert (
            loaded._with_public_overrides(SkipSoftmaxConfig(target_sparsity=0.7)).first_dense_steps
            == 12
        )
        assert (
            loaded._with_public_overrides(
                SkipSoftmaxConfig(target_sparsity=0.7, first_dense_steps=16)
            ).first_dense_steps
            == 16
        )

    def test_attention_config_no_sparse(self):
        cfg = AttentionConfig(backend="VANILLA")
        assert cfg.sparse_attention_config is None

    def test_base_class_inheritance(self):
        cfg = SkipSoftmaxConfig(threshold_scale_factor=5000.0)
        # Inherits from the LLM-shared SkipSoftmaxAttentionConfig (reuse, no duplication)
        assert isinstance(cfg, SkipSoftmaxAttentionConfig)
        assert cfg.algorithm == "skip_softmax"

    def test_quant_attention_requires_trtllm_backend(self):
        """Quantized attention requires backend='TRTLLM'."""
        with pytest.raises(ValueError, match="requires backend='TRTLLM'"):
            AttentionConfig(backend="VANILLA", quant_attention_config=QuantAttentionConfig())

    def test_quant_attention_rejects_unsupported_block_combo(self):
        """The validator must reject unsupported quantized-attention recipes."""
        with pytest.raises(ValueError, match="Unsupported quant_attention_config"):
            AttentionConfig(
                backend="TRTLLM",
                quant_attention_config=QuantAttentionConfig(
                    q_block_size=99,
                    k_block_size=99,
                    v_block_size=99,
                ),
            )


# =============================================================================
# Use case scenarios
# =============================================================================


class TestUseCaseScenarios:
    """End-to-end use case tests matching the PR documentation.

    Case 1: Normal HF checkpoint (no skip_softmax metadata in config.json)
      1a: User provides threshold_scale_factor → all layers get same threshold
      1b: User provides target_sparsity without formula → helpful error
      1c: User provides target_sparsity + formula → resolves correctly
      1d: User provides full config with layer_overrides → per-layer thresholds

    Case 2: ModelOpt checkpoint (has calibrated a, b in config.json)
      2a: User provides nothing → auto-enable from checkpoint
      2b: User provides threshold_scale_factor → user overrides checkpoint
      2c: User provides target_sparsity → uses checkpoint formula
    """

    MODELOPT_CHECKPOINT: Final = {
        "sparse_attention_config": {
            "config_groups": {
                "group_0": {
                    "sparse_algo": "softmax_skip",
                    "targets": ["Attention"],
                }
            },
            "threshold_scale_factor": {
                "formula": "a * exp(b * target_sparsity)",
                "prefill": {"a": 7.93, "b": 8.61},
                "decode": {"a": 0.12, "b": 9.85},
            },
            "producer": {"name": "modelopt", "version": "0.37.0"},
        }
    }

    # --- Case 1: Normal HF checkpoint ---

    def test_case_1a_user_threshold_only(self):
        """Normal checkpoint + user threshold → works."""
        cfg = SkipSoftmaxConfig(threshold_scale_factor=5000.0)
        result = cfg.resolve_threshold_scale_factor(checkpoint_formula=None)
        assert result == 5000.0

    def test_case_1b_user_target_sparsity_no_formula(self):
        """Normal checkpoint + target_sparsity without formula → helpful error."""
        cfg = SkipSoftmaxConfig(target_sparsity=0.5)
        with pytest.raises(ValueError, match="calibration formula"):
            cfg.resolve_threshold_scale_factor(checkpoint_formula=None)

    def test_case_1c_target_sparsity_with_attached_formula(self):
        """target_sparsity + attached calibration formula (from YAML/checkpoint) → resolves.

        The user surface only carries ``target_sparsity``; the formula is
        attached by the loader via :meth:`SkipSoftmaxConfig.with_calibration`.
        """
        cfg = SkipSoftmaxConfig.with_calibration(
            target_sparsity=0.5,
            formula=SkipSoftmaxFormula(log_a=math.log(0.0003), b=7.5),
        )
        result = cfg.resolve_threshold_scale_factor(checkpoint_formula=None)
        expected = 0.0003 * math.exp(7.5 * 0.5)
        assert result == pytest.approx(expected)

    def test_case_1d_layer_overrides_from_calibration(self):
        """Attached layer_overrides (from ModelOpt ``disabled_layers``) → per-layer thresholds."""
        cfg = SkipSoftmaxConfig.with_calibration(
            threshold_scale_factor=5000.0,
            layer_overrides={"blocks.0*": 0, "blocks.5*": 8000.0},
        )
        assert cfg.resolve_threshold("blocks.0.attn1") is None  # disabled
        assert cfg.resolve_threshold("blocks.5.attn1") == 8000.0  # override
        assert cfg.resolve_threshold("blocks.3.attn1") == 5000.0  # default

    # --- Case 2: ModelOpt checkpoint ---

    def test_case_2a_modelopt_checkpoint_auto_enable(self):
        """ModelOpt checkpoint + no user config → auto-enable from checkpoint.

        The pipeline should detect sparse_attention_config in checkpoint
        config.json and create a SkipSoftmaxConfig automatically.
        """
        from tensorrt_llm.visual_gen.sparse_attention import auto_detect_sparse_attention_config

        ckpt = self.MODELOPT_CHECKPOINT
        result = auto_detect_sparse_attention_config(ckpt)
        assert result is not None
        assert isinstance(result, SkipSoftmaxConfig)
        # Calibration formula is attached as a private attr (not user-facing).
        assert result._formula is not None
        assert result._formula.log_a == pytest.approx(math.log(7.93))
        assert result._formula.b == pytest.approx(8.61)

    def test_case_2b_modelopt_user_threshold_overrides(self):
        """ModelOpt checkpoint + user threshold → user wins."""
        cfg = SkipSoftmaxConfig(threshold_scale_factor=3000.0)
        ckpt_formula = self.MODELOPT_CHECKPOINT["sparse_attention_config"][
            "threshold_scale_factor"
        ]["prefill"]
        result = cfg.resolve_threshold_scale_factor(checkpoint_formula=ckpt_formula)
        # User threshold takes precedence, checkpoint formula ignored
        assert result == 3000.0

    def test_case_2c_modelopt_user_target_sparsity(self):
        """ModelOpt checkpoint + user target_sparsity → uses checkpoint formula."""
        cfg = SkipSoftmaxConfig(target_sparsity=0.5)
        ckpt_formula = self.MODELOPT_CHECKPOINT["sparse_attention_config"][
            "threshold_scale_factor"
        ]["prefill"]
        result = cfg.resolve_threshold_scale_factor(checkpoint_formula=ckpt_formula)
        expected = 7.93 * math.exp(8.61 * 0.5)
        assert result == pytest.approx(expected)

    def test_case_2a_no_sparse_config_returns_none(self):
        """Normal checkpoint (no sparse_attention_config) → returns None."""
        from tensorrt_llm.visual_gen.sparse_attention import auto_detect_sparse_attention_config

        result = auto_detect_sparse_attention_config({})
        assert result is None

        result = auto_detect_sparse_attention_config({"other_key": 123})
        assert result is None


# =============================================================================
# YAML loading
# =============================================================================


class TestYamlLoading:
    def test_load_modelopt_yaml(self, tmp_path):
        """Load from ModelOpt sparse YAML file."""
        from tensorrt_llm.visual_gen.sparse_attention import load_sparse_config_from_yaml

        yaml_content = """
config_groups:
  group_0:
    sparse_algo: softmax_skip
    targets:
    - WanAttention
    threshold_scale_factor:
      formula: log_a + b * target_sparsity
      prefill:
        log_a: -14.14
        b: 36.64
    disabled_layers:
    - blocks.0.attn1
    - blocks.0.attn2
    - blocks.39.attn2
"""
        yaml_file = tmp_path / "sparse.yaml"
        yaml_file.write_text(yaml_content)

        cfg = load_sparse_config_from_yaml(str(yaml_file))
        assert cfg is not None
        # Calibration formula + disabled-layer overrides land in private attrs,
        # not on the user-facing surface.
        assert cfg._formula.log_a == pytest.approx(-14.14)
        assert cfg._formula.b == pytest.approx(36.64)
        assert cfg._layer_overrides is not None
        assert cfg._layer_overrides["blocks.0.attn1"] == 0
        assert cfg._layer_overrides["blocks.0.attn2"] == 0
        assert cfg._layer_overrides["blocks.39.attn2"] == 0
        assert len(cfg._layer_overrides) == 3

    def test_load_modelopt_yaml_llm_format_a(self, tmp_path):
        """Load from LLM-format YAML where prefill uses 'a' instead of 'log_a'."""
        from tensorrt_llm.visual_gen.sparse_attention import load_sparse_config_from_yaml

        yaml_content = """
config_groups:
  group_0:
    sparse_algo: softmax_skip
    threshold_scale_factor:
      formula: a * exp(b * target_sparsity)
      prefill:
        a: 7.0e-5
        b: 7.929109
"""
        yaml_file = tmp_path / "sparse.yaml"
        yaml_file.write_text(yaml_content)

        cfg = load_sparse_config_from_yaml(str(yaml_file))
        assert cfg is not None
        # 'a' should be normalized to log_a = log(a)
        assert cfg._formula.log_a == pytest.approx(math.log(7e-5))
        assert cfg._formula.b == pytest.approx(7.929109)

    def test_load_modelopt_yaml_accepts_consistent_a_and_log_a(self, tmp_path):
        """ModelOpt may emit both 'a' and 'log_a'; accept them when consistent."""
        from tensorrt_llm.visual_gen.sparse_attention import load_sparse_config_from_yaml

        a = 2227.5528011152483
        yaml_content = f"""
config_groups:
  group_0:
    sparse_algo: softmax_skip
    threshold_scale_factor:
      formula: a * exp(b * target_sparsity)
      prefill:
        a: {a}
        b: 4.300334457820109
        log_a: {math.log(a)}
"""
        yaml_file = tmp_path / "sparse.yaml"
        yaml_file.write_text(yaml_content)

        cfg = load_sparse_config_from_yaml(str(yaml_file))
        assert cfg is not None
        assert cfg._formula.log_a == pytest.approx(math.log(a))
        assert cfg._formula.b == pytest.approx(4.300334457820109)

    def test_load_modelopt_yaml_rejects_inconsistent_a_and_log_a(self, tmp_path):
        """Reject dual-format ModelOpt YAML if the two coefficients disagree."""
        from tensorrt_llm.visual_gen.sparse_attention import load_sparse_config_from_yaml

        yaml_content = """
config_groups:
  group_0:
    sparse_algo: softmax_skip
    threshold_scale_factor:
      formula: a * exp(b * target_sparsity)
      prefill:
        a: 100.0
        b: 4.0
        log_a: 1.0
"""
        yaml_file = tmp_path / "sparse.yaml"
        yaml_file.write_text(yaml_content)

        with pytest.raises(ValueError, match="disagree"):
            load_sparse_config_from_yaml(str(yaml_file))

    def test_load_consolidated_modelopt_yaml_component_map(self, tmp_path):
        """Load current ModelOpt YAML with separate component calibration."""
        from tensorrt_llm.visual_gen.sparse_attention import load_sparse_config_from_yaml

        yaml_content = """
transformer:
  config_groups:
    group_0:
      sparse_algo: softmax_skip
      threshold_scale_factor:
        formula: log_a + b * target_sparsity
        prefill:
          log_a: -10.0
          b: 2.0
      disabled_layers:
      - blocks.0.attn1
transformer_2:
  config_groups:
    group_0:
      sparse_algo: softmax_skip
      threshold_scale_factor:
        formula: log_a + b * target_sparsity
        prefill:
          log_a: -20.0
          b: 4.0
      disabled_layers:
      - blocks.1.attn1
producer:
  name: modelopt
"""
        yaml_file = tmp_path / "sparse.yaml"
        yaml_file.write_text(yaml_content)

        cfg = load_sparse_config_from_yaml(str(yaml_file))
        assert cfg is not None
        assert cfg._formula is None
        assert cfg._component_configs is not None
        assert set(cfg._component_configs) == {"transformer", "transformer_2"}

        merged = cfg._with_public_overrides(SkipSoftmaxConfig(target_sparsity=0.5))
        transformer_threshold = math.exp(-10.0 + 2.0 * 0.5)
        transformer_2_threshold = math.exp(-20.0 + 4.0 * 0.5)
        assert merged.resolve_threshold("transformer.blocks.0.attn1") is None
        assert merged.resolve_threshold("transformer.blocks.2.attn1") == pytest.approx(
            transformer_threshold
        )
        assert merged.resolve_threshold("transformer_2.blocks.1.attn1") is None
        assert merged.resolve_threshold("transformer_2.blocks.2._orig_mod.attn1") == pytest.approx(
            transformer_2_threshold
        )

    def test_load_yaml_no_skip_softmax(self, tmp_path):
        """YAML without softmax_skip algo returns None."""
        from tensorrt_llm.visual_gen.sparse_attention import load_sparse_config_from_yaml

        yaml_content = """
config_groups:
  group_0:
    sparse_algo: other_algo
"""
        yaml_file = tmp_path / "sparse.yaml"
        yaml_file.write_text(yaml_content)

        cfg = load_sparse_config_from_yaml(str(yaml_file))
        assert cfg is None

    def test_auto_detect_yaml(self, tmp_path):
        """Auto-detect sparse YAML files in checkpoint directory."""
        from tensorrt_llm.visual_gen.sparse_attention import auto_detect_sparse_yaml

        yaml_content = """
config_groups:
  group_0:
    sparse_algo: softmax_skip
    threshold_scale_factor:
      prefill:
        log_a: -14.14
        b: 36.64
"""
        (tmp_path / "sparse.yaml").write_text(yaml_content)

        cfg = auto_detect_sparse_yaml(str(tmp_path))
        assert cfg is not None
        assert cfg._formula is not None
        assert cfg._formula.log_a == pytest.approx(-14.14)


# =============================================================================
# resolve_threshold_scale_factor
# =============================================================================


class TestResolveThresholdScaleFactor:
    def test_direct_threshold_returns_immediately(self):
        cfg = SkipSoftmaxConfig(threshold_scale_factor=5000.0)
        assert cfg.resolve_threshold_scale_factor() == 5000.0

    def test_direct_threshold_ignores_formula(self):
        cfg = SkipSoftmaxConfig.with_calibration(
            threshold_scale_factor=5000.0,
            target_sparsity=0.5,
            formula=SkipSoftmaxFormula(log_a=math.log(0.0003), b=7.5),
        )
        # threshold_scale_factor takes precedence
        assert cfg.resolve_threshold_scale_factor() == 5000.0

    def test_target_sparsity_with_attached_formula(self):
        cfg = SkipSoftmaxConfig.with_calibration(
            target_sparsity=0.5,
            formula=SkipSoftmaxFormula(log_a=math.log(7e-5), b=7.929109),
        )
        expected = 7e-5 * math.exp(7.929109 * 0.5)
        assert cfg.resolve_threshold_scale_factor() == pytest.approx(expected)

    def test_target_sparsity_with_checkpoint_formula(self):
        cfg = SkipSoftmaxConfig(target_sparsity=0.5)
        checkpoint = {"a": 7e-5, "b": 7.929109}
        expected = 7e-5 * math.exp(7.929109 * 0.5)
        assert cfg.resolve_threshold_scale_factor(checkpoint) == pytest.approx(expected)

    def test_attached_formula_overrides_checkpoint(self):
        cfg = SkipSoftmaxConfig.with_calibration(
            target_sparsity=0.5,
            formula=SkipSoftmaxFormula(log_a=math.log(0.001), b=5.0),  # attached
        )
        checkpoint = {"a": 7e-5, "b": 7.929109}  # checkpoint_formula arg (lower priority)
        expected = 0.001 * math.exp(5.0 * 0.5)  # attached formula wins
        assert cfg.resolve_threshold_scale_factor(checkpoint) == pytest.approx(expected)

    def test_modelopt_checkpoint_formula_format(self):
        """Test with the actual ModelOpt config.json format."""
        cfg = SkipSoftmaxConfig(target_sparsity=0.5)
        # ModelOpt format: sparse_attention_config.threshold_scale_factor.prefill
        modelopt_prefill = {"a": 7.93, "b": 8.61}
        expected = 7.93 * math.exp(8.61 * 0.5)
        assert cfg.resolve_threshold_scale_factor(modelopt_prefill) == pytest.approx(expected)

    def test_no_threshold_no_sparsity_returns_none(self):
        cfg = SkipSoftmaxConfig()
        assert cfg.resolve_threshold_scale_factor() is None

    def test_target_sparsity_no_formula_raises(self):
        cfg = SkipSoftmaxConfig(target_sparsity=0.5)
        with pytest.raises(ValueError, match="calibration formula"):
            cfg.resolve_threshold_scale_factor()

    def test_target_sparsity_zero(self):
        cfg = SkipSoftmaxConfig.with_calibration(
            target_sparsity=0.0,
            formula=SkipSoftmaxFormula(log_a=math.log(7e-5), b=7.929109),
        )
        # exp(0) = 1, so result = a
        assert cfg.resolve_threshold_scale_factor() == pytest.approx(7e-5)

    def test_target_sparsity_one(self):
        cfg = SkipSoftmaxConfig.with_calibration(
            target_sparsity=1.0,
            formula=SkipSoftmaxFormula(log_a=math.log(7e-5), b=7.929109),
        )
        expected = 7e-5 * math.exp(7.929109)
        assert cfg.resolve_threshold_scale_factor() == pytest.approx(expected)


# =============================================================================
# resolve_threshold (layer overrides)
# =============================================================================


class TestResolveThreshold:
    def test_no_overrides_returns_default(self):
        cfg = SkipSoftmaxConfig(threshold_scale_factor=5000.0)
        assert cfg.resolve_threshold("transformer_blocks.5.attn1") == 5000.0

    def test_matching_override(self):
        cfg = SkipSoftmaxConfig.with_calibration(
            threshold_scale_factor=5000.0,
            layer_overrides={"transformer_blocks.0.*": 0},
        )
        assert cfg.resolve_threshold("transformer_blocks.0.attn1") is None  # disabled

    def test_non_matching_returns_default(self):
        cfg = SkipSoftmaxConfig.with_calibration(
            threshold_scale_factor=5000.0,
            layer_overrides={"transformer_blocks.0.*": 0},
        )
        assert cfg.resolve_threshold("transformer_blocks.5.attn1") == 5000.0

    def test_override_with_custom_value(self):
        cfg = SkipSoftmaxConfig.with_calibration(
            threshold_scale_factor=5000.0,
            layer_overrides={"single_transformer_blocks.*": 8000.0},
        )
        assert cfg.resolve_threshold("single_transformer_blocks.10.attn") == 8000.0

    def test_first_match_wins(self):
        cfg = SkipSoftmaxConfig.with_calibration(
            threshold_scale_factor=5000.0,
            layer_overrides={
                "transformer_blocks.0.*": 0,
                "transformer_blocks.*": 3000.0,
            },
        )
        # First pattern matches
        assert cfg.resolve_threshold("transformer_blocks.0.attn1") is None
        # Second pattern matches for other blocks
        assert cfg.resolve_threshold("transformer_blocks.5.attn1") == 3000.0

    def test_wildcard_patterns(self):
        cfg = SkipSoftmaxConfig.with_calibration(
            threshold_scale_factor=5000.0,
            layer_overrides={"*.attn2": 0},  # disable all cross-attention
        )
        assert cfg.resolve_threshold("transformer.blocks.3.attn2") is None
        assert cfg.resolve_threshold("transformer.blocks.3.attn1") == 5000.0

    def test_modelopt_relative_patterns_match_transformer_components(self):
        cfg = SkipSoftmaxConfig.with_calibration(
            threshold_scale_factor=5000.0,
            layer_overrides={"blocks.0.attn1": 0},
        )
        assert cfg.resolve_threshold("transformer.blocks.0.attn1") is None
        assert cfg.resolve_threshold("transformer_2.blocks.0.attn1") is None
        assert cfg.resolve_threshold("transformer.blocks.0._orig_mod.attn1") is None
        assert cfg.resolve_threshold("transformer.blocks.1.attn1") == 5000.0

    def test_no_threshold_or_target_returns_none(self):
        cfg = SkipSoftmaxConfig()
        assert cfg.resolve_threshold("any_layer") is None

    def test_target_sparsity_without_formula_raises(self):
        cfg = SkipSoftmaxConfig(target_sparsity=0.5)
        with pytest.raises(ValueError, match="calibration formula"):
            cfg.get_or_resolve_threshold()
        with pytest.raises(ValueError, match="calibration formula"):
            cfg.resolve_threshold("any_layer")


# =============================================================================
# apply_skip_softmax_overrides
# =============================================================================


class TestApplySkipSoftmaxOverrides:
    def _make_mock_model(self):
        """Create a mock model with patched TrtllmAttention instances."""
        from unittest.mock import MagicMock

        import torch.nn as nn

        from tensorrt_llm._torch.visual_gen.attention_backend.trtllm import TrtllmAttention

        def make_mock_backend():
            mock = MagicMock(spec=TrtllmAttention)
            mock.sparse_attention_config = None
            return mock

        class MockAttentionModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = make_mock_backend()

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block0 = MockAttentionModule()
                self.block1 = MockAttentionModule()
                self.block2 = MockAttentionModule()

        return MockModel()

    def test_no_overrides_returns_zero(self):
        model = self._make_mock_model()
        cfg = SkipSoftmaxConfig(threshold_scale_factor=5000.0)
        assert apply_skip_softmax_overrides(model, cfg) == 0

    def test_set_skip_softmax_enabled_global_dense_prefix(self):
        model = self._make_mock_model()
        cfg = SkipSoftmaxConfig(threshold_scale_factor=5000.0)

        n = set_skip_softmax_enabled(model, cfg, enabled=True)
        assert n == 3
        assert model.block0.attn.sparse_attention_config is not None
        assert model.block1.attn.sparse_attention_config is not None
        assert model.block2.attn.sparse_attention_config is not None

        n = set_skip_softmax_enabled(model, cfg, enabled=False)
        assert n == 3
        assert model.block0.attn.sparse_attention_config is None
        assert model.block1.attn.sparse_attention_config is None
        assert model.block2.attn.sparse_attention_config is None

    def test_overrides_applied(self):
        model = self._make_mock_model()
        cfg = SkipSoftmaxConfig.with_calibration(
            threshold_scale_factor=5000.0,
            layer_overrides={"block0*": 0, "block2*": 8000.0},
        )
        n = apply_skip_softmax_overrides(model, cfg)
        assert n == 3

        # block0: disabled (threshold=0 → None)
        assert model.block0.attn.sparse_attention_config is None
        # block1: default threshold
        assert model.block1.attn.sparse_attention_config is not None
        assert model.block1.attn.sparse_attention_config.threshold_scale_factor_prefill == 5000.0
        # block2: overridden to 8000
        assert model.block2.attn.sparse_attention_config is not None
        assert model.block2.attn.sparse_attention_config.threshold_scale_factor_prefill == 8000.0

    def test_overrides_applied_to_compiled_module_names(self):
        from unittest.mock import MagicMock

        import torch.nn as nn

        from tensorrt_llm._torch.visual_gen.attention_backend.trtllm import TrtllmAttention

        class MockAttentionModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = MagicMock(spec=TrtllmAttention)
                self.attn.sparse_attention_config = None

        class MockCompiledBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self._orig_mod = nn.Module()
                self._orig_mod.attn1 = MockAttentionModule()

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([MockCompiledBlock()])

        model = MockModel()
        cfg = SkipSoftmaxConfig.with_calibration(
            threshold_scale_factor=5000.0,
            layer_overrides={"blocks.0.attn1": 0},
        )
        n = apply_skip_softmax_overrides(model, cfg)
        assert n == 1
        assert model.blocks[0]._orig_mod.attn1.attn.sparse_attention_config is None

    def test_overrides_applied_to_component_relative_names(self):
        from unittest.mock import MagicMock

        import torch.nn as nn

        from tensorrt_llm._torch.visual_gen.attention_backend.trtllm import TrtllmAttention

        class MockAttentionModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = MagicMock(spec=TrtllmAttention)
                self.attn.sparse_attention_config = None

        class MockCompiledBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self._orig_mod = nn.Module()
                self._orig_mod.attn1 = MockAttentionModule()

        class MockTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([MockCompiledBlock()])

        class MockPipeline(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = MockTransformer()
                self.transformer_2 = MockTransformer()

        model = MockPipeline()
        cfg = SkipSoftmaxConfig.with_calibration(
            threshold_scale_factor=5000.0,
            layer_overrides={"blocks.0.attn1": 0},
        )
        n = apply_skip_softmax_overrides(model, cfg)
        assert n == 2
        assert model.transformer.blocks[0]._orig_mod.attn1.attn.sparse_attention_config is None
        assert model.transformer_2.blocks[0]._orig_mod.attn1.attn.sparse_attention_config is None

    def test_component_configs_apply_distinct_thresholds(self):
        from unittest.mock import MagicMock

        import torch.nn as nn

        from tensorrt_llm._torch.visual_gen.attention_backend.trtllm import TrtllmAttention

        class MockAttentionModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = MagicMock(spec=TrtllmAttention)
                self.attn.sparse_attention_config = None

        class MockTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([MockAttentionModule()])

        class MockPipeline(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = MockTransformer()
                self.transformer_2 = MockTransformer()

        model = MockPipeline()
        cfg = SkipSoftmaxConfig.with_calibration(
            component_configs={
                "transformer": SkipSoftmaxConfig.with_calibration(
                    target_sparsity=0.5,
                    formula=SkipSoftmaxFormula(log_a=-10.0, b=2.0),
                ),
                "transformer_2": SkipSoftmaxConfig.with_calibration(
                    target_sparsity=0.5,
                    formula=SkipSoftmaxFormula(log_a=-20.0, b=4.0),
                ),
            }
        )
        n = apply_skip_softmax_overrides(model, cfg)
        assert n == 2
        assert model.transformer.blocks[
            0
        ].attn.sparse_attention_config.threshold_scale_factor_prefill == pytest.approx(
            math.exp(-10.0 + 2.0 * 0.5)
        )
        assert model.transformer_2.blocks[
            0
        ].attn.sparse_attention_config.threshold_scale_factor_prefill == pytest.approx(
            math.exp(-20.0 + 4.0 * 0.5)
        )

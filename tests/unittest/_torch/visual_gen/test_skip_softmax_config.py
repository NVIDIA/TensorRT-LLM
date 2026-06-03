# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SkipSoftmax visual generation config and wiring."""

import math
from typing import Final

import pytest

from tensorrt_llm._torch.attention_backend.sparse.skip_softmax import SkipSoftmaxFormula
from tensorrt_llm.llmapi.llm_args import SkipSoftmaxAttentionConfig as LlmSkipSoftmaxAttentionConfig
from tensorrt_llm.visual_gen.args import AttentionConfig, QuantAttentionConfig
from tensorrt_llm.visual_gen.sparse_attention import (
    SkipSoftmaxAttentionConfig,
    apply_skip_softmax_overrides,
)


def _scheduled_prefill_threshold(params):
    assert params is not None
    return params.scheduler.get_kernel_params().threshold_scale_factor_prefill


# =============================================================================
# SkipSoftmaxFormula — numexpr expression + named coefficients
# =============================================================================


class TestSkipSoftmaxFormulaFormats:
    def test_accepts_arbitrary_expression(self):
        """Checkpoint can ship any numexpr-evaluable expression along with
        its named coefficients; the runtime evaluates it verbatim."""
        f = SkipSoftmaxFormula(
            formula="a * exp(b * target_sparsity) + c",
            coefficients={"a": 2.0, "b": 1.5, "c": 0.1},
        )
        assert f.formula == "a * exp(b * target_sparsity) + c"
        assert f.coefficients == {"a": 2.0, "b": 1.5, "c": 0.1}

    def test_rejects_expression_missing_coefficient(self):
        """Coefficient set must cover everything the expression references."""
        with pytest.raises(ValueError, match="missing"):
            SkipSoftmaxFormula(
                formula="a * exp(b * target_sparsity)",
                coefficients={"a": 1.0},
            )

    def test_rejects_invalid_expression(self):
        """A non-numexpr-parseable expression is rejected at construction."""
        with pytest.raises(ValueError, match="invalid formula"):
            SkipSoftmaxFormula(
                formula="a * (b",
                coefficients={"a": 1.0, "b": 2.0},
            )

    def test_rejects_formula_without_target_sparsity(self):
        """A formula that never references target_sparsity is rejected."""
        with pytest.raises(ValueError, match="must reference 'target_sparsity'"):
            SkipSoftmaxFormula(formula="a * b", coefficients={"a": 1.0, "b": 2.0})


# =============================================================================
# SkipSoftmaxAttentionConfig construction
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
                },
            }
        )
        sc = cfg.sparse_attention_config
        assert isinstance(sc, SkipSoftmaxAttentionConfig)
        assert sc.algorithm == "skip_softmax"
        assert sc.threshold_scale_factor == 5000.0
        assert sc.target_sparsity == 0.5

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
            sparse_attention_config=SkipSoftmaxAttentionConfig(
                threshold_scale_factor=5000.0,
                target_sparsity=0.5,
            ),
        )
        dumped = original.model_dump()
        rehydrated = AttentionConfig(**dumped)
        assert rehydrated.model_dump() == dumped

    def test_attention_config_no_sparse(self):
        cfg = AttentionConfig(backend="VANILLA")
        assert cfg.sparse_attention_config is None

    def test_standalone_config_is_not_llm_config(self):
        cfg = SkipSoftmaxAttentionConfig(threshold_scale_factor=5000.0)
        # VG SkipSoftmaxAttentionConfig is intentionally decoupled from the LLM-side
        # SkipSoftmaxAttentionConfig: diffusion pipelines have no prefill/decode
        # split, and VG carries its own knobs (e.g., future ``warmup``).
        assert not isinstance(cfg, LlmSkipSoftmaxAttentionConfig)
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
                "coefficients": {"a": 7.93, "b": 8.61},
            },
            "producer": {"name": "modelopt", "version": "0.37.0"},
        }
    }

    # --- Case 1: Normal HF checkpoint ---

    def test_case_1a_user_threshold_only(self):
        """Normal checkpoint + user threshold → works."""
        cfg = SkipSoftmaxAttentionConfig(threshold_scale_factor=5000.0)
        result = cfg.resolve_threshold_scale_factor(checkpoint_formula=None)
        assert result == 5000.0

    def test_case_1b_user_target_sparsity_no_formula(self):
        """Normal checkpoint + target_sparsity without formula → helpful error."""
        cfg = SkipSoftmaxAttentionConfig(target_sparsity=0.5)
        with pytest.raises(ValueError, match="calibration formula"):
            cfg.resolve_threshold_scale_factor(checkpoint_formula=None)

    def test_case_1c_target_sparsity_with_attached_formula(self):
        """target_sparsity + attached calibration formula (from YAML/checkpoint) → resolves.

        The user surface only carries ``target_sparsity``; the formula is
        attached by the loader via :meth:`SkipSoftmaxAttentionConfig.with_calibration`.
        """
        cfg = SkipSoftmaxAttentionConfig.with_calibration(
            target_sparsity=0.5,
            formula=SkipSoftmaxFormula(
                formula="exp(log_a + b * target_sparsity)",
                coefficients={"log_a": math.log(0.0003), "b": 7.5},
            ),
        )
        result = cfg.resolve_threshold_scale_factor(checkpoint_formula=None)
        expected = 0.0003 * math.exp(7.5 * 0.5)
        assert result == pytest.approx(expected)

    def test_case_1d_layer_overrides_from_calibration(self):
        """Attached layer_overrides (from ModelOpt ``disabled_layers``) → per-layer thresholds."""
        cfg = SkipSoftmaxAttentionConfig.with_calibration(
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
        config.json and create a SkipSoftmaxAttentionConfig automatically.
        """
        from tensorrt_llm.visual_gen.sparse_attention import auto_detect_sparse_attention_config

        ckpt = self.MODELOPT_CHECKPOINT
        result = auto_detect_sparse_attention_config(ckpt)
        assert result is not None
        assert isinstance(result, SkipSoftmaxAttentionConfig)
        # Calibration formula is attached as a private attr (not user-facing).
        # The checkpoint ships a full numexpr expression; coefficients are
        # honored verbatim (no log_a normalization when the form is supplied).
        assert result._formula is not None
        assert result._formula.formula == "a * exp(b * target_sparsity)"
        assert result._formula.coefficients["a"] == pytest.approx(7.93)
        assert result._formula.coefficients["b"] == pytest.approx(8.61)

    @staticmethod
    def _modelopt_block():
        tsf = TestUseCaseScenarios.MODELOPT_CHECKPOINT["sparse_attention_config"][
            "threshold_scale_factor"
        ]
        return {"formula": tsf["formula"], **tsf["coefficients"]}

    def test_case_2b_modelopt_user_threshold_overrides(self):
        """ModelOpt checkpoint + user threshold → user wins."""
        cfg = SkipSoftmaxAttentionConfig(threshold_scale_factor=3000.0)
        result = cfg.resolve_threshold_scale_factor(checkpoint_formula=self._modelopt_block())
        # User threshold takes precedence, checkpoint formula ignored
        assert result == 3000.0

    def test_case_2c_modelopt_user_target_sparsity(self):
        """ModelOpt checkpoint + user target_sparsity → uses checkpoint formula."""
        cfg = SkipSoftmaxAttentionConfig(target_sparsity=0.5)
        result = cfg.resolve_threshold_scale_factor(checkpoint_formula=self._modelopt_block())
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
      formula: exp(log_a + b * target_sparsity)
      coefficients:
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
        # not on the user-facing surface. The loader honors the YAML's
        # `formula` field verbatim — there is no implicit fallback.
        assert cfg._formula.formula == "exp(log_a + b * target_sparsity)"
        assert cfg._formula.coefficients["log_a"] == pytest.approx(-14.14)
        assert cfg._formula.coefficients["b"] == pytest.approx(36.64)
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
      coefficients:
        a: 7.0e-5
        b: 7.929109
"""
        yaml_file = tmp_path / "sparse.yaml"
        yaml_file.write_text(yaml_content)

        cfg = load_sparse_config_from_yaml(str(yaml_file))
        assert cfg is not None
        # YAML carries a full numexpr expression containing `exp(...)`,
        # so coefficients are honored verbatim (no log_a normalization).
        assert cfg._formula.formula == "a * exp(b * target_sparsity)"
        assert cfg._formula.coefficients["a"] == pytest.approx(7e-5)
        assert cfg._formula.coefficients["b"] == pytest.approx(7.929109)

    def test_load_consolidated_modelopt_yaml_component_map(self, tmp_path):
        """Load current ModelOpt YAML with separate component calibration."""
        from tensorrt_llm.visual_gen.sparse_attention import load_sparse_config_from_yaml

        yaml_content = """
transformer:
  config_groups:
    group_0:
      sparse_algo: softmax_skip
      threshold_scale_factor:
        formula: exp(log_a + b * target_sparsity)
        coefficients:
          log_a: -10.0
          b: 2.0
      disabled_layers:
      - blocks.0.attn1
transformer_2:
  config_groups:
    group_0:
      sparse_algo: softmax_skip
      threshold_scale_factor:
        formula: exp(log_a + b * target_sparsity)
        coefficients:
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

        merged = cfg._with_public_overrides(SkipSoftmaxAttentionConfig(target_sparsity=0.5))
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
      formula: exp(log_a + b * target_sparsity)
      coefficients:
        log_a: -14.14
        b: 36.64
"""
        (tmp_path / "sparse.yaml").write_text(yaml_content)

        cfg = auto_detect_sparse_yaml(str(tmp_path))
        assert cfg is not None
        assert cfg._formula is not None
        assert cfg._formula.coefficients["log_a"] == pytest.approx(-14.14)


# =============================================================================
# resolve_threshold_scale_factor
# =============================================================================


class TestResolveThresholdScaleFactor:
    def test_direct_threshold_returns_immediately(self):
        cfg = SkipSoftmaxAttentionConfig(threshold_scale_factor=5000.0)
        assert cfg.resolve_threshold_scale_factor() == 5000.0

    def test_direct_threshold_ignores_formula(self):
        cfg = SkipSoftmaxAttentionConfig.with_calibration(
            threshold_scale_factor=5000.0,
            target_sparsity=0.5,
            formula=SkipSoftmaxFormula(
                formula="exp(log_a + b * target_sparsity)",
                coefficients={"log_a": math.log(0.0003), "b": 7.5},
            ),
        )
        # threshold_scale_factor takes precedence
        assert cfg.resolve_threshold_scale_factor() == 5000.0

    def test_target_sparsity_with_attached_formula(self):
        cfg = SkipSoftmaxAttentionConfig.with_calibration(
            target_sparsity=0.5,
            formula=SkipSoftmaxFormula(
                formula="exp(log_a + b * target_sparsity)",
                coefficients={"log_a": math.log(7e-5), "b": 7.929109},
            ),
        )
        expected = 7e-5 * math.exp(7.929109 * 0.5)
        assert cfg.resolve_threshold_scale_factor() == pytest.approx(expected)

    def test_target_sparsity_with_checkpoint_formula(self):
        cfg = SkipSoftmaxAttentionConfig(target_sparsity=0.5)
        checkpoint = {
            "formula": "a * exp(b * target_sparsity)",
            "a": 7e-5,
            "b": 7.929109,
        }
        expected = 7e-5 * math.exp(7.929109 * 0.5)
        assert cfg.resolve_threshold_scale_factor(checkpoint) == pytest.approx(expected)

    def test_attached_formula_overrides_checkpoint(self):
        cfg = SkipSoftmaxAttentionConfig.with_calibration(
            target_sparsity=0.5,
            formula=SkipSoftmaxFormula(
                formula="exp(log_a + b * target_sparsity)",
                coefficients={"log_a": math.log(0.001), "b": 5.0},
            ),  # attached
        )
        # checkpoint_formula arg (lower priority — attached _formula wins)
        checkpoint = {
            "formula": "a * exp(b * target_sparsity)",
            "a": 7e-5,
            "b": 7.929109,
        }
        expected = 0.001 * math.exp(5.0 * 0.5)  # attached formula wins
        assert cfg.resolve_threshold_scale_factor(checkpoint) == pytest.approx(expected)

    def test_modelopt_checkpoint_formula_format(self):
        """Test with the ModelOpt config.json phase block (formula + coefs)."""
        cfg = SkipSoftmaxAttentionConfig(target_sparsity=0.5)
        modelopt_prefill = {
            "formula": "a * exp(b * target_sparsity)",
            "a": 7.93,
            "b": 8.61,
        }
        expected = 7.93 * math.exp(8.61 * 0.5)
        assert cfg.resolve_threshold_scale_factor(modelopt_prefill) == pytest.approx(expected)

    def test_no_threshold_no_sparsity_returns_none(self):
        cfg = SkipSoftmaxAttentionConfig()
        assert cfg.resolve_threshold_scale_factor() is None

    def test_target_sparsity_no_formula_raises(self):
        cfg = SkipSoftmaxAttentionConfig(target_sparsity=0.5)
        with pytest.raises(ValueError, match="calibration formula"):
            cfg.resolve_threshold_scale_factor()

    def test_target_sparsity_zero(self):
        cfg = SkipSoftmaxAttentionConfig.with_calibration(
            target_sparsity=0.0,
            formula=SkipSoftmaxFormula(
                formula="exp(log_a + b * target_sparsity)",
                coefficients={"log_a": math.log(7e-5), "b": 7.929109},
            ),
        )
        # exp(0) = 1, so result = a
        assert cfg.resolve_threshold_scale_factor() == pytest.approx(7e-5)

    def test_target_sparsity_one(self):
        cfg = SkipSoftmaxAttentionConfig.with_calibration(
            target_sparsity=1.0,
            formula=SkipSoftmaxFormula(
                formula="exp(log_a + b * target_sparsity)",
                coefficients={"log_a": math.log(7e-5), "b": 7.929109},
            ),
        )
        expected = 7e-5 * math.exp(7.929109)
        assert cfg.resolve_threshold_scale_factor() == pytest.approx(expected)


# =============================================================================
# resolve_threshold (layer overrides)
# =============================================================================


class TestResolveThreshold:
    def test_no_overrides_returns_default(self):
        cfg = SkipSoftmaxAttentionConfig(threshold_scale_factor=5000.0)
        assert cfg.resolve_threshold("transformer_blocks.5.attn1") == 5000.0

    def test_matching_override(self):
        cfg = SkipSoftmaxAttentionConfig.with_calibration(
            threshold_scale_factor=5000.0,
            layer_overrides={"transformer_blocks.0.*": 0},
        )
        assert cfg.resolve_threshold("transformer_blocks.0.attn1") is None  # disabled

    def test_non_matching_returns_default(self):
        cfg = SkipSoftmaxAttentionConfig.with_calibration(
            threshold_scale_factor=5000.0,
            layer_overrides={"transformer_blocks.0.*": 0},
        )
        assert cfg.resolve_threshold("transformer_blocks.5.attn1") == 5000.0

    def test_override_with_custom_value(self):
        cfg = SkipSoftmaxAttentionConfig.with_calibration(
            threshold_scale_factor=5000.0,
            layer_overrides={"single_transformer_blocks.*": 8000.0},
        )
        assert cfg.resolve_threshold("single_transformer_blocks.10.attn") == 8000.0

    def test_first_match_wins(self):
        cfg = SkipSoftmaxAttentionConfig.with_calibration(
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
        cfg = SkipSoftmaxAttentionConfig.with_calibration(
            threshold_scale_factor=5000.0,
            layer_overrides={"*.attn2": 0},  # disable all cross-attention
        )
        assert cfg.resolve_threshold("transformer.blocks.3.attn2") is None
        assert cfg.resolve_threshold("transformer.blocks.3.attn1") == 5000.0

    def test_modelopt_relative_patterns_match_transformer_components(self):
        cfg = SkipSoftmaxAttentionConfig.with_calibration(
            threshold_scale_factor=5000.0,
            layer_overrides={"blocks.0.attn1": 0},
        )
        assert cfg.resolve_threshold("transformer.blocks.0.attn1") is None
        assert cfg.resolve_threshold("transformer_2.blocks.0.attn1") is None
        assert cfg.resolve_threshold("transformer.blocks.0._orig_mod.attn1") is None
        assert cfg.resolve_threshold("transformer.blocks.1.attn1") == 5000.0

    def test_no_threshold_or_target_returns_none(self):
        cfg = SkipSoftmaxAttentionConfig()
        assert cfg.resolve_threshold("any_layer") is None

    def test_target_sparsity_without_formula_raises(self):
        cfg = SkipSoftmaxAttentionConfig(target_sparsity=0.5)
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
            mock.sparse_params = None
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
        cfg = SkipSoftmaxAttentionConfig(threshold_scale_factor=5000.0)
        assert apply_skip_softmax_overrides(model, cfg) == 0

    def test_overrides_applied(self):
        model = self._make_mock_model()
        cfg = SkipSoftmaxAttentionConfig.with_calibration(
            threshold_scale_factor=5000.0,
            layer_overrides={"block0*": 0, "block2*": 8000.0},
        )
        n = apply_skip_softmax_overrides(model, cfg)
        assert n == 3

        # block0: disabled (threshold=0 → None)
        assert model.block0.attn.sparse_params is None
        # block1: default threshold
        assert _scheduled_prefill_threshold(model.block1.attn.sparse_params) == 5000.0
        # block2: overridden to 8000
        assert _scheduled_prefill_threshold(model.block2.attn.sparse_params) == 8000.0

    def test_overrides_applied_to_compiled_module_names(self):
        from unittest.mock import MagicMock

        import torch.nn as nn

        from tensorrt_llm._torch.visual_gen.attention_backend.trtllm import TrtllmAttention

        class MockAttentionModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = MagicMock(spec=TrtllmAttention)
                self.attn.sparse_params = None

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
        cfg = SkipSoftmaxAttentionConfig.with_calibration(
            threshold_scale_factor=5000.0,
            layer_overrides={"blocks.0.attn1": 0},
        )
        n = apply_skip_softmax_overrides(model, cfg)
        assert n == 1
        assert model.blocks[0]._orig_mod.attn1.attn.sparse_params is None

    def test_overrides_applied_to_component_relative_names(self):
        from unittest.mock import MagicMock

        import torch.nn as nn

        from tensorrt_llm._torch.visual_gen.attention_backend.trtllm import TrtllmAttention

        class MockAttentionModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = MagicMock(spec=TrtllmAttention)
                self.attn.sparse_params = None

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
        cfg = SkipSoftmaxAttentionConfig.with_calibration(
            threshold_scale_factor=5000.0,
            layer_overrides={"blocks.0.attn1": 0},
        )
        n = apply_skip_softmax_overrides(model, cfg)
        assert n == 2
        assert model.transformer.blocks[0]._orig_mod.attn1.attn.sparse_params is None
        assert model.transformer_2.blocks[0]._orig_mod.attn1.attn.sparse_params is None

    def test_component_configs_apply_distinct_thresholds(self):
        from unittest.mock import MagicMock

        import torch.nn as nn

        from tensorrt_llm._torch.visual_gen.attention_backend.trtllm import TrtllmAttention

        class MockAttentionModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = MagicMock(spec=TrtllmAttention)
                self.attn.sparse_params = None

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
        cfg = SkipSoftmaxAttentionConfig.with_calibration(
            component_configs={
                "transformer": SkipSoftmaxAttentionConfig.with_calibration(
                    target_sparsity=0.5,
                    formula=SkipSoftmaxFormula(
                        formula="exp(log_a + b * target_sparsity)",
                        coefficients={"log_a": -10.0, "b": 2.0},
                    ),
                ),
                "transformer_2": SkipSoftmaxAttentionConfig.with_calibration(
                    target_sparsity=0.5,
                    formula=SkipSoftmaxFormula(
                        formula="exp(log_a + b * target_sparsity)",
                        coefficients={"log_a": -20.0, "b": 4.0},
                    ),
                ),
            }
        )
        n = apply_skip_softmax_overrides(model, cfg)
        assert n == 2
        assert _scheduled_prefill_threshold(
            model.transformer.blocks[0].attn.sparse_params
        ) == pytest.approx(math.exp(-10.0 + 2.0 * 0.5))
        assert _scheduled_prefill_threshold(
            model.transformer_2.blocks[0].attn.sparse_params
        ) == pytest.approx(math.exp(-20.0 + 4.0 * 0.5))

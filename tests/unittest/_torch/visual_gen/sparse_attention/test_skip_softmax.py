# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for VisualGen Skip Softmax Attention config behavior."""

import json
import math
from typing import Optional

import pytest
import yaml
from pydantic import ValidationError

from tensorrt_llm._torch.attention_backend.sparse.skip_softmax import (
    SkipSoftmaxParams,
    SkipSoftmaxScheduler,
)
from tensorrt_llm.visual_gen.args import AttentionConfig, VisualGenArgs
from tensorrt_llm.visual_gen.sparse_attention import SkipSoftmaxAttentionConfig


def _ckpt_sparse_attention_config(
    *,
    log_a: float = -10.0,
    b: float = 2.0,
    target_sparsity: Optional[object] = 0.5,
    ignore: Optional[list[str]] = None,
    disabled_until_timestep: Optional[float] = None,
) -> dict:
    # ModelOpt stores sparse-attention metadata under config_groups. These
    # tests vary one skip-softmax group without requiring a real checkpoint.
    group = {
        "algorithm": "skip_softmax",
        "threshold_scale_factor": {
            "formula": "exp(log_a + b * target_sparsity)",
            "coefficients": {
                "log_a": log_a,
                "b": b,
            },
        },
    }
    if target_sparsity is not None:
        group["target_sparsity"] = target_sparsity
    if ignore is not None:
        group["ignore"] = ignore
    if disabled_until_timestep is not None:
        group["disabled_until_timestep"] = disabled_until_timestep
    return {
        "config_groups": {
            "group_0": group,
        },
    }


def _checkpoint_config(**kwargs) -> dict:
    return {"sparse_attention_config": _ckpt_sparse_attention_config(**kwargs)}


def _skip_softmax_group(checkpoint_config: dict) -> dict:
    return checkpoint_config["sparse_attention_config"]["config_groups"]["group_0"]


def _expected_threshold(log_a: float, b: float, target_sparsity: float) -> float:
    return math.exp(log_a + b * target_sparsity)


def _prefill_threshold(
    sparse_params: Optional[SkipSoftmaxParams],
    *,
    timestep: Optional[float] = None,
) -> float:
    assert isinstance(sparse_params, SkipSoftmaxParams)
    return sparse_params.scheduler.get_kernel_params(
        timestep=timestep
    ).threshold_scale_factor_prefill


class TestVisualGenSkipSoftmaxUserAPI:
    """User-facing config surface: VisualGen args only expose runtime knobs."""

    def test_python_api_parses_skip_softmax_config(self):
        # Python users configure Skip Softmax Attention through
        # AttentionConfig.sparse_attention_config.
        config = AttentionConfig(
            backend="TRTLLM",
            sparse_attention_config={
                "algorithm": "skip_softmax",
                "threshold_scale_factor": 5000.0,
                "target_sparsity": 0.5,
                "disabled_until_timestep": 0.6,
            },
        )

        sparse_config = config.sparse_attention_config

        assert isinstance(sparse_config, SkipSoftmaxAttentionConfig)
        assert sparse_config.threshold_scale_factor == 5000.0
        assert sparse_config.target_sparsity == 0.5
        assert sparse_config.disabled_until_timestep == 0.6
        assert AttentionConfig(**config.model_dump()).model_dump() == config.model_dump()

    def test_yaml_api_parses_skip_softmax_config(self):
        # YAML config should deserialize to the same public config object as
        # the Python API.
        config_dict = yaml.safe_load("""
attention_config:
  backend: TRTLLM
  sparse_attention_config:
    algorithm: skip_softmax
    target_sparsity: 0.5
    disabled_until_timestep: 0.6
""")

        args = VisualGenArgs(model="/tmp/dummy_model", **config_dict)

        sparse_config = args.attention_config.sparse_attention_config
        assert isinstance(sparse_config, SkipSoftmaxAttentionConfig)
        assert sparse_config.target_sparsity == 0.5
        assert sparse_config.disabled_until_timestep == 0.6

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("formula", "a * target_sparsity"),
            ("coefficients", {"a": 1.0}),
            ("ignore", ["blocks.0.attn1"]),
            ("config_groups", {}),
        ],
    )
    def test_checkpoint_metadata_fields_are_not_user_api_fields(self, field, value):
        # Calibration formula, coefficients, group selection, and layer ignore
        # patterns come from checkpoint config.json, not the public API.
        with pytest.raises(ValidationError):
            AttentionConfig(
                backend="TRTLLM",
                sparse_attention_config={
                    "algorithm": "skip_softmax",
                    "threshold_scale_factor": 5000.0,
                    field: value,
                },
            )

    def test_visualgen_target_sparsity_user_value_is_scalar(self):
        # VisualGen has no prefill/decode split, so target_sparsity is scalar.
        with pytest.raises(ValidationError):
            SkipSoftmaxAttentionConfig(
                target_sparsity={
                    "prefill": 0.5,
                    "decode": 0.3,
                }
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"target_sparsity": -0.1},
            {"target_sparsity": 1.1},
            {"disabled_until_timestep": -0.1},
            {"disabled_until_timestep": 1.1},
        ],
    )
    def test_normalized_user_values_must_be_in_unit_interval(self, kwargs):
        # target_sparsity and disabled_until_timestep are normalized values.
        with pytest.raises(ValidationError):
            SkipSoftmaxAttentionConfig(**kwargs)

    def test_direct_threshold_lowers_without_checkpoint_metadata(self):
        # Direct threshold configuration bypasses ModelOpt calibration.
        config = SkipSoftmaxAttentionConfig(threshold_scale_factor=5000.0)

        sparse_params = config.to_sparse_params()

        assert _prefill_threshold(sparse_params) == pytest.approx(5000.0)

    def test_threshold_scale_factor_takes_precedence_without_checkpoint_formula(self):
        # Raw threshold wins over target_sparsity, so no formula is needed.
        config = SkipSoftmaxAttentionConfig(
            threshold_scale_factor=5000.0,
            target_sparsity=1.0,
        )

        sparse_params = config.to_sparse_params()

        assert _prefill_threshold(sparse_params) == pytest.approx(5000.0)


class TestVisualGenSkipSoftmaxCheckpointConfig:
    """Checkpoint metadata: ModelOpt calibration is consumed at lowering time."""

    def test_user_target_sparsity_lowers_through_checkpoint_formula(self):
        # User target_sparsity overrides the checkpoint default but still uses
        # the checkpoint's calibrated formula.
        config = SkipSoftmaxAttentionConfig(target_sparsity=0.4)

        sparse_params = config.to_sparse_params(
            checkpoint_config=_checkpoint_config(
                log_a=math.log(7e-5),
                b=7.929109,
                target_sparsity=0.9,
            )
        )

        assert _prefill_threshold(sparse_params) == pytest.approx(7e-5 * math.exp(7.929109 * 0.4))

    def test_checkpoint_target_sparsity_default_lowers_through_formula(self):
        # If the user omits target_sparsity, the checkpoint default can drive
        # threshold resolution through the same formula.
        config = SkipSoftmaxAttentionConfig()

        sparse_params = config.to_sparse_params(
            checkpoint_config=_checkpoint_config(
                log_a=math.log(7e-5),
                b=7.929109,
                target_sparsity=0.5,
            )
        )

        assert _prefill_threshold(sparse_params) == pytest.approx(7e-5 * math.exp(7.929109 * 0.5))

    def test_checkpoint_formula_accepts_arbitrary_numexpr_expression(self):
        # The formula is evaluated by numexpr, so it is not restricted to the
        # exp(log_a + b * target_sparsity) shape used by the helper.
        config = SkipSoftmaxAttentionConfig()
        checkpoint_config = _checkpoint_config(target_sparsity=0.25)
        _skip_softmax_group(checkpoint_config)["threshold_scale_factor"] = {
            "formula": "sqrt(a + target_sparsity)",
            "coefficients": {
                "a": 0.75,
            },
        }

        sparse_params = config.to_sparse_params(checkpoint_config=checkpoint_config)

        assert _prefill_threshold(sparse_params) == pytest.approx(1.0)

    def test_target_sparsity_requires_checkpoint_formula(self):
        # target_sparsity is semantic; without a formula it cannot be converted
        # into the kernel-facing threshold_scale_factor.
        config = SkipSoftmaxAttentionConfig(target_sparsity=0.5)
        checkpoint_config = _checkpoint_config()
        del _skip_softmax_group(checkpoint_config)["threshold_scale_factor"]

        with pytest.raises(ValueError, match="calibration formula"):
            config.to_sparse_params(checkpoint_config=checkpoint_config)

    def test_checkpoint_phase_target_sparsity_dict_is_rejected(self):
        # prefill/decode target_sparsity dictionaries are LLM-only.
        config = SkipSoftmaxAttentionConfig()
        checkpoint_config = _checkpoint_config(target_sparsity={"prefill": 0.5})

        with pytest.raises(ValueError, match="prefill/decode phase dictionaries"):
            config.to_sparse_params(checkpoint_config=checkpoint_config)

    def test_other_sparse_attention_groups_are_ignored(self):
        # Checkpoints may include groups for several sparse algorithms. Only
        # the skip-softmax group should affect this config.
        config = SkipSoftmaxAttentionConfig(target_sparsity=0.5)
        checkpoint_config = _checkpoint_config(log_a=-10.0, b=2.0)
        groups = checkpoint_config["sparse_attention_config"]["config_groups"]
        groups["group_1"] = {
            "algorithm": "vsa",
            "target_sparsity": 0.8,
        }

        sparse_params = config.to_sparse_params(checkpoint_config=checkpoint_config)

        assert _prefill_threshold(sparse_params) == pytest.approx(
            _expected_threshold(-10.0, 2.0, 0.5)
        )

    def test_multiple_skip_softmax_checkpoint_groups_are_invalid(self):
        # Multiple skip-softmax groups make calibration ambiguous.
        config = SkipSoftmaxAttentionConfig(target_sparsity=0.5)
        checkpoint_config = _checkpoint_config()
        groups = checkpoint_config["sparse_attention_config"]["config_groups"]
        groups["group_1"] = {
            "algorithm": "vsa",
            "target_sparsity": 0.8,
        }
        groups["group_2"] = dict(groups["group_0"])

        with pytest.raises(ValueError, match="multiple skip-softmax"):
            config.to_sparse_params(checkpoint_config=checkpoint_config)


class TestVisualGenSkipSoftmaxLayerFiltering:
    """Layer filtering: checkpoint ignore patterns disable selected modules."""

    def test_checkpoint_ignore_patterns_disable_matching_attention_layers(self):
        # Patterns are matched against full names, component-relative names,
        # and names with torch.compile's _orig_mod wrapper removed.
        config = SkipSoftmaxAttentionConfig(threshold_scale_factor=5000.0)
        checkpoint_config = _checkpoint_config(ignore=["blocks.0.attn1", "*.attn2"])

        assert (
            config.to_sparse_params(
                module_name="transformer.blocks.0.attn1",
                checkpoint_config=checkpoint_config,
            )
            is None
        )
        assert (
            config.to_sparse_params(
                module_name="transformer_2.blocks.0._orig_mod.attn1",
                checkpoint_config=checkpoint_config,
            )
            is None
        )
        assert (
            config.to_sparse_params(
                module_name="transformer.blocks.3.attn2",
                checkpoint_config=checkpoint_config,
            )
            is None
        )
        assert _prefill_threshold(
            config.to_sparse_params(
                module_name="transformer.blocks.1.attn1",
                checkpoint_config=checkpoint_config,
            )
        ) == pytest.approx(5000.0)


class TestVisualGenSkipSoftmaxTimestepCutoff:
    """Timestep cutoff: early denoising can run with skip-softmax disabled."""

    def test_user_disabled_until_timestep_uses_normalized_timestep(self):
        # Denoising timesteps move from high to low. The cutoff disables
        # skip-softmax while timestep >= disabled_until_timestep.
        config = SkipSoftmaxAttentionConfig(
            threshold_scale_factor=5000.0,
            disabled_until_timestep=0.6,
        )
        sparse_params = config.to_sparse_params()

        assert _prefill_threshold(sparse_params, timestep=1.0) == pytest.approx(0.0)
        assert _prefill_threshold(sparse_params, timestep=0.6) == pytest.approx(0.0)
        assert _prefill_threshold(sparse_params, timestep=0.59) == pytest.approx(5000.0)

    def test_checkpoint_disabled_until_timestep_applies_when_user_omits_it(self):
        # Checkpoint metadata can provide a default cutoff when the user omits
        # the field.
        config = SkipSoftmaxAttentionConfig(threshold_scale_factor=5000.0)
        sparse_params = config.to_sparse_params(
            checkpoint_config=_checkpoint_config(disabled_until_timestep=0.6)
        )

        assert _prefill_threshold(sparse_params, timestep=1.0) == pytest.approx(0.0)
        assert _prefill_threshold(sparse_params, timestep=0.6) == pytest.approx(0.0)
        assert _prefill_threshold(sparse_params, timestep=0.59) == pytest.approx(5000.0)

    def test_user_disabled_until_timestep_overrides_checkpoint_default(self):
        # User config has higher priority than checkpoint defaults.
        config = SkipSoftmaxAttentionConfig(
            threshold_scale_factor=5000.0,
            disabled_until_timestep=0.8,
        )
        sparse_params = config.to_sparse_params(
            checkpoint_config=_checkpoint_config(disabled_until_timestep=0.4)
        )

        assert _prefill_threshold(sparse_params, timestep=0.7) == pytest.approx(5000.0)

    @pytest.mark.parametrize(
        ("timestep", "expected"),
        [
            (1.0, 0),
            (0.6, 0),
            (0.59, 1),
            (None, None),
        ],
    )
    def test_graph_phase_tracks_disabled_until_timestep_boundary(
        self,
        timestep,
        expected,
    ):
        # CUDA graph keys need a stable sparse-attention phase so captured
        # graphs are not reused across disabled and enabled skip-softmax states.
        assert (
            SkipSoftmaxScheduler.get_graph_phase_for_timestep(
                timestep,
                disabled_until_timestep=0.6,
            )
            == expected
        )


class TestVisualGenSkipSoftmaxPipelineConfig:
    """Pipeline config: multi-transformer checkpoints keep metadata separated."""

    def test_pipeline_config_keeps_checkpoint_metadata_per_model(self, tmp_path):
        from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig

        # Construct a minimal Diffusers-style layout with two transformer
        # components, each with its own config.json.
        (tmp_path / "transformer").mkdir()
        (tmp_path / "transformer_2").mkdir()
        (tmp_path / "model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "WanPipeline",
                    "transformer": ["diffusers", "WanTransformer3DModel"],
                    "transformer_2": ["diffusers", "WanTransformer3DModel"],
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "transformer" / "config.json").write_text(
            json.dumps(
                _checkpoint_config(
                    ignore=["blocks.0.attn1"],
                    log_a=-10.0,
                    b=2.0,
                )
            ),
            encoding="utf-8",
        )
        (tmp_path / "transformer_2" / "config.json").write_text(
            json.dumps(_checkpoint_config(log_a=-20.0, b=4.0)),
            encoding="utf-8",
        )

        pipeline_config = DiffusionPipelineConfig.from_pretrained(
            str(tmp_path),
            args=VisualGenArgs(
                model=str(tmp_path),
                attention_config=AttentionConfig(
                    backend="TRTLLM",
                    sparse_attention_config=SkipSoftmaxAttentionConfig(
                        target_sparsity=0.5,
                        disabled_until_timestep=0.6,
                    ),
                ),
            ),
        )

        sparse_config = pipeline_config.attention.sparse_attention_config
        assert isinstance(sparse_config, SkipSoftmaxAttentionConfig)
        assert sparse_config.model_dump() == {
            "algorithm": "skip_softmax",
            "threshold_scale_factor": None,
            "target_sparsity": 0.5,
            "disabled_until_timestep": 0.6,
        }

        # The public sparse config is shared at pipeline level, while
        # checkpoint calibration metadata stays attached to each component.
        transformer_config = pipeline_config.model_configs["transformer"]
        transformer_2_config = pipeline_config.model_configs["transformer_2"]
        assert transformer_config.pretrained_config is not transformer_2_config.pretrained_config

        transformer_params = sparse_config.to_sparse_params(
            module_name="transformer.blocks.1.attn1",
            pretrained_config=transformer_config.pretrained_config,
        )
        transformer_2_params = sparse_config.to_sparse_params(
            module_name="transformer_2.blocks.1.attn1",
            pretrained_config=transformer_2_config.pretrained_config,
        )
        transformer_disabled_params = sparse_config.to_sparse_params(
            module_name="transformer.blocks.0.attn1",
            pretrained_config=transformer_config.pretrained_config,
        )

        # The same public target_sparsity resolves through different per-model
        # formulas, and ignore patterns apply only to the owning component.
        assert _prefill_threshold(transformer_params, timestep=0.6) == pytest.approx(0.0)
        assert _prefill_threshold(transformer_params, timestep=0.59) == pytest.approx(
            _expected_threshold(-10.0, 2.0, 0.5)
        )
        assert _prefill_threshold(transformer_2_params, timestep=0.59) == pytest.approx(
            _expected_threshold(-20.0, 4.0, 0.5)
        )
        assert transformer_disabled_params is None

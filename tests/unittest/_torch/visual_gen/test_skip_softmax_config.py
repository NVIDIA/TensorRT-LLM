# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API tests for VisualGen skip-softmax sparse attention config."""

import json
import math
from typing import Final, Optional

import pytest

from tensorrt_llm._torch.attention_backend.sparse.skip_softmax import (
    SkipSoftmaxParams,
    SkipSoftmaxScheduler,
)
from tensorrt_llm.visual_gen.args import AttentionConfig, VisualGenArgs
from tensorrt_llm.visual_gen.sparse_attention import SkipSoftmaxAttentionConfig


def _checkpoint_config(
    *,
    log_a: float = -10.0,
    b: float = 2.0,
    target_sparsity: float = 0.5,
    disabled_layers: Optional[list[str]] = None,
    initial_disabled_steps: Optional[int] = None,
) -> dict:
    group = {
        "algorithm": "skip_softmax",
        "targets": ["WanAttention"],
        "threshold_scale_factor": {
            "formula": "exp(log_a + b * target_sparsity)",
            "prefill": {
                "log_a": log_a,
                "b": b,
            },
        },
        "target_sparsity": {
            "prefill": target_sparsity,
        },
    }
    if disabled_layers is not None:
        group["ignore"] = disabled_layers
    if initial_disabled_steps is not None:
        group["initial_disabled_steps"] = initial_disabled_steps
    return {
        "sparse_attention_config": {
            "config_groups": {
                "group_0": group,
            },
        },
    }


def _checkpoint_config_with_disabled_layers(
    *,
    disabled_layers: list[str],
    log_a: float = -10.0,
    b: float = 2.0,
) -> dict:
    return _checkpoint_config(
        log_a=log_a,
        b=b,
        disabled_layers=disabled_layers,
    )


def _prefill_threshold(
    sparse_params: Optional[SkipSoftmaxParams],
    *,
    step_index: Optional[int] = None,
) -> float:
    assert isinstance(sparse_params, SkipSoftmaxParams)
    return sparse_params.scheduler.get_kernel_params(
        step_index=step_index
    ).threshold_scale_factor_prefill


class TestPublicApi:
    def test_attention_config_accepts_user_surface(self):
        config = AttentionConfig(
            backend="TRTLLM",
            sparse_attention_config={
                "algorithm": "skip_softmax",
                "threshold_scale_factor": 5000.0,
                "target_sparsity": 0.5,
                "exclude_modules": ["blocks.0.*", "*.attn2"],
                "initial_disabled_steps": 2,
            },
        )

        sparse_config = config.sparse_attention_config
        assert isinstance(sparse_config, SkipSoftmaxAttentionConfig)
        assert sparse_config.threshold_scale_factor == 5000.0
        assert sparse_config.target_sparsity == 0.5
        assert sparse_config.exclude_modules == ["blocks.0.*", "*.attn2"]
        assert sparse_config.initial_disabled_steps == 2
        assert AttentionConfig(**config.model_dump()).model_dump() == config.model_dump()

    @pytest.mark.parametrize("field", ["formula", "component_configs"])
    def test_calibration_internals_are_not_user_fields(self, field):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AttentionConfig(
                backend="TRTLLM",
                sparse_attention_config={
                    "algorithm": "skip_softmax",
                    "threshold_scale_factor": 5000.0,
                    field: {},
                },
            )

    def test_direct_threshold_lowers_to_sparse_params(self):
        config = SkipSoftmaxAttentionConfig(threshold_scale_factor=5000.0)

        sparse_params = config.to_sparse_params()

        assert isinstance(sparse_params, SkipSoftmaxParams)
        assert _prefill_threshold(sparse_params) == 5000.0

    def test_target_sparsity_lowers_through_checkpoint_formula(self):
        config = SkipSoftmaxAttentionConfig(target_sparsity=0.5)

        sparse_params = config.to_sparse_params(
            checkpoint_config=_checkpoint_config(log_a=math.log(7e-5), b=7.929109)
        )

        assert _prefill_threshold(sparse_params) == pytest.approx(7e-5 * math.exp(7.929109 * 0.5))

    def test_checkpoint_target_sparsity_lowers_through_checkpoint_formula(self):
        config = SkipSoftmaxAttentionConfig()

        sparse_params = config.to_sparse_params(
            checkpoint_config=_checkpoint_config(
                log_a=math.log(7e-5), b=7.929109, target_sparsity=0.5
            )
        )

        assert _prefill_threshold(sparse_params) == pytest.approx(7e-5 * math.exp(7.929109 * 0.5))

    def test_target_sparsity_without_checkpoint_formula_raises(self):
        config = SkipSoftmaxAttentionConfig(target_sparsity=0.5)

        with pytest.raises(ValueError, match="calibration formula"):
            config.to_sparse_params()

    def test_threshold_scale_factor_wins_over_target_sparsity(self):
        config = SkipSoftmaxAttentionConfig(
            threshold_scale_factor=5000.0,
            target_sparsity=1.0,
        )

        assert _prefill_threshold(config.to_sparse_params()) == 5000.0

    def test_user_exclude_modules_use_fnmatch_and_component_relative_names(self):
        config = SkipSoftmaxAttentionConfig(
            threshold_scale_factor=5000.0,
            exclude_modules=["blocks.0.attn1", "*.attn2"],
        )

        assert config.to_sparse_params(module_name="transformer.blocks.0.attn1") is None
        assert config.to_sparse_params(module_name="transformer_2.blocks.0._orig_mod.attn1") is None
        assert config.to_sparse_params(module_name="transformer.blocks.3.attn2") is None
        assert (
            _prefill_threshold(config.to_sparse_params(module_name="transformer.blocks.3.attn1"))
            == 5000.0
        )

    def test_checkpoint_disabled_layers_are_applied_during_lowering(self):
        config = SkipSoftmaxAttentionConfig(threshold_scale_factor=5000.0)
        checkpoint_config = _checkpoint_config_with_disabled_layers(
            disabled_layers=["blocks.0.attn1"]
        )

        assert (
            config.to_sparse_params(
                module_name="transformer.blocks.0.attn1",
                checkpoint_config=checkpoint_config,
            )
            is None
        )
        assert (
            _prefill_threshold(
                config.to_sparse_params(
                    module_name="transformer.blocks.1.attn1",
                    checkpoint_config=checkpoint_config,
                )
            )
            == 5000.0
        )

    def test_initial_disabled_steps_use_explicit_step_index(self):
        config = SkipSoftmaxAttentionConfig(
            threshold_scale_factor=5000.0,
            initial_disabled_steps=2,
        )
        sparse_params = config.to_sparse_params()

        assert _prefill_threshold(sparse_params, step_index=0) == 0.0
        assert _prefill_threshold(sparse_params, step_index=1) == 0.0
        assert _prefill_threshold(sparse_params, step_index=2) == 5000.0

    def test_checkpoint_initial_disabled_steps_use_explicit_step_index(self):
        config = SkipSoftmaxAttentionConfig(threshold_scale_factor=5000.0)
        sparse_params = config.to_sparse_params(
            checkpoint_config=_checkpoint_config(initial_disabled_steps=2)
        )

        assert _prefill_threshold(sparse_params, step_index=0) == 0.0
        assert _prefill_threshold(sparse_params, step_index=1) == 0.0
        assert _prefill_threshold(sparse_params, step_index=2) == 5000.0

    def test_graph_phase_tracks_initial_disabled_step_boundary(self):
        config = SkipSoftmaxAttentionConfig(
            threshold_scale_factor=5000.0,
            initial_disabled_steps=2,
        )

        assert (
            SkipSoftmaxScheduler.get_graph_phase_for_step_index(
                0,
                initial_disabled_steps=config.initial_disabled_steps,
            )
            == 0
        )
        assert (
            SkipSoftmaxScheduler.get_graph_phase_for_step_index(
                1,
                initial_disabled_steps=config.initial_disabled_steps,
            )
            == 0
        )
        assert (
            SkipSoftmaxScheduler.get_graph_phase_for_step_index(
                2,
                initial_disabled_steps=config.initial_disabled_steps,
            )
            == 1
        )


class TestCheckpointMetadata:
    CHECKPOINT_CONFIG: Final = {
        "sparse_attention_config": {
            "config_groups": {
                "group_0": {
                    "algorithm": "skip_softmax",
                    "targets": ["WanAttention"],
                    "ignore": ["blocks.0.attn1"],
                    "threshold_scale_factor": {
                        "formula": "exp(log_a + b * target_sparsity)",
                        "prefill": {
                            "log_a": -10.0,
                            "b": 2.0,
                        },
                    },
                    "target_sparsity": {
                        "prefill": 0.5,
                    },
                },
            },
        },
    }

    def test_pipeline_config_keeps_checkpoint_metadata_per_model(self, tmp_path):
        from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig

        (tmp_path / "transformer").mkdir()
        (tmp_path / "transformer_2").mkdir()
        (tmp_path / "model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "WanPipeline",
                    "transformer": ["diffusers", "WanTransformer3DModel"],
                    "transformer_2": ["diffusers", "WanTransformer3DModel"],
                }
            )
        )
        (tmp_path / "transformer" / "config.json").write_text(
            json.dumps(
                _checkpoint_config_with_disabled_layers(
                    disabled_layers=["blocks.0.attn1"],
                    log_a=-10.0,
                    b=2.0,
                )
            )
        )
        (tmp_path / "transformer_2" / "config.json").write_text(
            json.dumps(_checkpoint_config(log_a=-20.0, b=4.0))
        )

        pipeline_config = DiffusionPipelineConfig.from_pretrained(
            str(tmp_path),
            args=VisualGenArgs(
                model=str(tmp_path),
                attention_config=AttentionConfig(
                    backend="TRTLLM",
                    sparse_attention_config=SkipSoftmaxAttentionConfig(
                        target_sparsity=0.5,
                        initial_disabled_steps=1,
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
            "exclude_modules": None,
            "initial_disabled_steps": 1,
        }

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
        assert _prefill_threshold(transformer_params, step_index=0) == 0.0
        assert _prefill_threshold(transformer_params, step_index=1) == pytest.approx(
            math.exp(-10.0 + 2.0 * 0.5)
        )
        assert _prefill_threshold(transformer_2_params, step_index=1) == pytest.approx(
            math.exp(-20.0 + 4.0 * 0.5)
        )
        assert transformer_disabled_params is None

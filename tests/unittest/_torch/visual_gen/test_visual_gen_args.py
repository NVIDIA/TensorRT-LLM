# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for VisualGenArgs construction, validation, and serialization."""

import pickle

import pytest
from pydantic import ValidationError

from tensorrt_llm._torch.visual_gen.config import (
    AttentionConfig,
    CompilationConfig,
    CudaGraphConfig,
    ParallelConfig,
    PipelineConfig,
    TeaCacheConfig,
    TorchCompileConfig,
    VisualGenArgs,
)


class TestVisualGenArgsStrictValidation:
    """extra='forbid' rejects unknown fields at every nesting level."""

    def test_unknown_top_level_field_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            VisualGenArgs(checkpoint_path="/tmp/model", unknown_field="bad")

    def test_typo_field_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            VisualGenArgs(checkpoint_path="/tmp/model", chekpoint_path="/typo")

    def test_nested_parallel_unknown_field_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            VisualGenArgs(
                checkpoint_path="/tmp/model",
                parallel={"dit_cfg_size": 1, "nonexistent_param": 42},
            )

    def test_nested_attention_unknown_field_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            AttentionConfig(backend="VANILLA", extra_key="bad")

    def test_nested_teacache_unknown_field_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            TeaCacheConfig(enable_teacache=True, unknown_opt=True)

    def test_nested_torch_compile_unknown_field_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            TorchCompileConfig(enable_torch_compile=True, bad_key=1)

    def test_nested_cuda_graph_unknown_field_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            CudaGraphConfig(enable_cuda_graph=False, extra=True)

    def test_nested_warmup_unknown_field_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            CompilationConfig(resolutions=[(480, 832)], bad_field=True)

    def test_nested_pipeline_unknown_field_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            PipelineConfig(fuse_qkv=True, invalid_flag=True)

    def test_legacy_linear_field_rejected(self):
        """The removed 'linear' YAML field must now cause an error."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            VisualGenArgs(
                checkpoint_path="/tmp/model",
                linear={"type": "default"},
            )


class TestVisualGenArgsFromDict:
    """from_dict now enforces strict validation (no silent drops)."""

    def test_valid_dict(self):
        args = VisualGenArgs.from_dict(
            {
                "checkpoint_path": "/tmp/model",
                "parallel": {"dit_cfg_size": 2, "dit_ulysses_size": 1},
            }
        )
        assert args.checkpoint_path == "/tmp/model"
        assert args.parallel.dit_cfg_size == 2

    def test_unknown_field_raises(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            VisualGenArgs.from_dict(
                {
                    "checkpoint_path": "/tmp/model",
                    "bad_key": 123,
                }
            )

    def test_nested_dict_auto_coerced(self):
        args = VisualGenArgs.from_dict(
            {
                "checkpoint_path": "/tmp/model",
                "attention": {"backend": "TRTLLM"},
                "teacache": {"enable_teacache": True, "teacache_thresh": 0.3},
            }
        )
        assert isinstance(args.attention, AttentionConfig)
        assert args.attention.backend == "TRTLLM"
        assert args.teacache.enable_teacache is True
        assert args.teacache.teacache_thresh == 0.3

    def test_quant_config_dict_coerced(self):
        args = VisualGenArgs.from_dict(
            {
                "checkpoint_path": "/tmp/model",
                "quant_config": {"quant_algo": "FP8", "dynamic": True},
            }
        )
        assert args.quant_config.quant_algo is not None
        assert args.dynamic_weight_quant is True


class TestVisualGenArgsFromYaml:
    """from_yaml round-trips through a YAML file."""

    def test_from_yaml_basic(self, tmp_path):
        yaml_path = tmp_path / "config.yml"
        yaml_path.write_text(
            "checkpoint_path: /tmp/model\nparallel:\n  dit_cfg_size: 2\n  dit_ulysses_size: 1\n"
        )
        args = VisualGenArgs.from_yaml(yaml_path)
        assert args.checkpoint_path == "/tmp/model"
        assert args.parallel.dit_cfg_size == 2

    def test_from_yaml_with_overrides(self, tmp_path):
        yaml_path = tmp_path / "config.yml"
        yaml_path.write_text("checkpoint_path: /tmp/model\ndtype: float16\n")
        args = VisualGenArgs.from_yaml(yaml_path, dtype="bfloat16")
        assert args.dtype == "bfloat16"

    def test_from_yaml_unknown_field_raises(self, tmp_path):
        yaml_path = tmp_path / "bad.yml"
        yaml_path.write_text("checkpoint_path: /tmp/model\nlinear:\n  type: default\n")
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            VisualGenArgs.from_yaml(yaml_path)


class TestParallelConfigValidation:
    """ParallelConfig no longer checks WORLD_SIZE at construction time."""

    def test_large_parallel_no_env_check(self):
        pc = ParallelConfig(dit_cfg_size=2, dit_ulysses_size=4)
        assert pc.total_parallel_size == 8
        assert pc.n_workers == 8

    def test_validate_world_size_passes(self):
        pc = ParallelConfig(dit_cfg_size=2, dit_ulysses_size=2)
        pc.validate_world_size(4)

    def test_validate_world_size_fails(self):
        pc = ParallelConfig(dit_cfg_size=2, dit_ulysses_size=4)
        with pytest.raises(ValueError, match="exceeds world_size"):
            pc.validate_world_size(4)


class TestVisualGenArgsPickle:
    """VisualGenArgs must survive pickle round-trip (mp.Process spawn)."""

    def test_pickle_roundtrip(self):
        args = VisualGenArgs(
            checkpoint_path="/tmp/model",
            dtype="float16",
            parallel=ParallelConfig(dit_cfg_size=2, dit_ulysses_size=1),
            attention=AttentionConfig(backend="TRTLLM"),
            quant_config={"quant_algo": "FP8", "dynamic": True},
        )
        data = pickle.dumps(args)
        restored = pickle.loads(data)

        assert restored.checkpoint_path == args.checkpoint_path
        assert restored.dtype == args.dtype
        assert restored.parallel.dit_cfg_size == 2
        assert restored.attention.backend == "TRTLLM"
        assert restored.quant_config.quant_algo is not None
        assert restored.dynamic_weight_quant is True

    def test_model_copy_device_override(self):
        args = VisualGenArgs(checkpoint_path="/tmp/model", device="cuda")
        updated = args.model_copy(update={"device": "cuda:3"})
        assert updated.device == "cuda:3"
        assert args.device == "cuda"

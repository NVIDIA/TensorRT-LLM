# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for VisualGenArgs construction, validation, and serialization."""

import itertools
import pickle
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from tensorrt_llm._torch.visual_gen.config import (
    AttentionConfig,
    CacheDiTConfig,
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
            TeaCacheConfig(unknown_opt=True)

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


class TestVisualGenArgsCacheBackend:
    def test_cache_dit_nested_config(self):
        args = VisualGenArgs(
            checkpoint_path="/tmp/model",
            cache=CacheDiTConfig(Fn_compute_blocks=2, max_warmup_steps=4),
        )
        assert isinstance(args.cache, CacheDiTConfig)
        assert args.cache_dit.Fn_compute_blocks == 2
        assert args.cache_dit.max_warmup_steps == 4

    def test_cache_union_discriminated_teacache(self):
        args = VisualGenArgs(
            checkpoint_path="/tmp/model",
            cache=TeaCacheConfig(teacache_thresh=0.11),
        )
        assert args.cache_backend == "teacache"
        assert isinstance(args.cache, TeaCacheConfig)
        assert args.teacache.teacache_thresh == 0.11

    def test_cache_default_is_none(self):
        args = VisualGenArgs(checkpoint_path="/tmp/model")
        assert args.cache is None
        assert args.cache_backend is None


class TestVisualGenArgsFromDict:
    """VisualGenArgs construction from dicts enforces strict validation."""

    def test_valid_dict(self):
        args = VisualGenArgs(
            **{
                "checkpoint_path": "/tmp/model",
                "parallel": {"dit_cfg_size": 2, "dit_ulysses_size": 1},
            }
        )
        assert args.checkpoint_path == "/tmp/model"
        assert args.parallel.dit_cfg_size == 2

    def test_unknown_field_raises(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            VisualGenArgs(
                **{
                    "checkpoint_path": "/tmp/model",
                    "bad_key": 123,
                }
            )

    def test_nested_dict_auto_coerced(self):
        args = VisualGenArgs(
            **{
                "checkpoint_path": "/tmp/model",
                "attention": {"backend": "TRTLLM"},
                "cache": {"cache_backend": "teacache", "teacache_thresh": 0.3},
            }
        )
        assert isinstance(args.attention, AttentionConfig)
        assert args.attention.backend == "TRTLLM"
        assert isinstance(args.cache, TeaCacheConfig)
        assert args.teacache.teacache_thresh == 0.3

    def test_quant_config_dict_coerced(self):
        args = VisualGenArgs(
            **{
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


# =============================================================================
# Batch Generation Input Parsing
# =============================================================================


class TestDiffusionRequestBatchPrompt:
    """DiffusionRequest accepts Union[str, List[str]] for prompt field."""

    def test_single_prompt(self):
        from tensorrt_llm._torch.visual_gen.executor import DiffusionRequest

        req = DiffusionRequest(request_id=0, prompt="hello")
        assert req.prompt == "hello"

    def test_list_prompt(self):
        from tensorrt_llm._torch.visual_gen.executor import DiffusionRequest

        prompts = ["a sunset", "a city"]
        req = DiffusionRequest(request_id=0, prompt=prompts)
        assert req.prompt == prompts
        assert len(req.prompt) == 2


class TestVisualGenBatchInputParsing:
    """Test that VisualGen.generate_async() correctly parses batch inputs.

    Uses mocking to avoid spawning GPU worker processes.
    """

    def _make_visual_gen_with_mock_executor(self):
        """Create a VisualGen instance with a mocked executor."""
        from tensorrt_llm.visual_gen import VisualGen

        # Patch __init__ to avoid spawning workers
        with patch.object(VisualGen, "__init__", lambda self, *a, **kw: None):
            vg = VisualGen.__new__(VisualGen)
            vg.model = "/tmp/fake"
            vg.args = VisualGenArgs(checkpoint_path="/tmp/fake")
            vg._req_counter = itertools.count()
            vg.executor = MagicMock()
            return vg

    def _make_params(self):
        from tensorrt_llm.visual_gen import VisualGenParams

        return VisualGenParams()

    def test_string_input(self):
        """String input → single-element list in DiffusionRequest."""
        vg = self._make_visual_gen_with_mock_executor()
        params = self._make_params()

        vg.generate_async(inputs="a cat", params=params)

        # Check the DiffusionRequest passed to enqueue_requests
        call_args = vg.executor.enqueue_requests.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0].prompt == ["a cat"]

    def test_dict_input(self):
        """Dict input → prompt wrapped in list + negative_prompt extracted."""
        vg = self._make_visual_gen_with_mock_executor()
        params = self._make_params()

        vg.generate_async(
            inputs={"prompt": "a cat", "negative_prompt": "blurry"},
            params=params,
        )

        call_args = vg.executor.enqueue_requests.call_args[0][0]
        assert call_args[0].prompt == ["a cat"]
        assert call_args[0].negative_prompt == "blurry"

    def test_list_of_strings_input(self):
        """List of strings → batch prompt in single DiffusionRequest."""
        vg = self._make_visual_gen_with_mock_executor()
        params = self._make_params()

        vg.generate_async(inputs=["a sunset", "a city"], params=params)

        call_args = vg.executor.enqueue_requests.call_args[0][0]
        assert len(call_args) == 1
        req = call_args[0]
        assert req.prompt == ["a sunset", "a city"]
        assert req.negative_prompt is None

    def test_list_of_dicts_input(self):
        """List of dicts → batch prompts with negative_prompt from first dict."""
        vg = self._make_visual_gen_with_mock_executor()
        params = self._make_params()

        vg.generate_async(
            inputs=[
                {"prompt": "a sunset", "negative_prompt": "dark"},
                {"prompt": "a city"},
            ],
            params=params,
        )

        call_args = vg.executor.enqueue_requests.call_args[0][0]
        req = call_args[0]
        assert req.prompt == ["a sunset", "a city"]
        assert req.negative_prompt == "dark"

    def test_mixed_list_input(self):
        """Mixed list of strings and dicts → batch prompts extracted."""
        vg = self._make_visual_gen_with_mock_executor()
        params = self._make_params()

        vg.generate_async(inputs=["a sunset", {"prompt": "a city"}], params=params)

        call_args = vg.executor.enqueue_requests.call_args[0][0]
        req = call_args[0]
        assert req.prompt == ["a sunset", "a city"]

    def test_conflicting_negative_prompt_raises(self):
        """Conflicting per-item negative_prompt raises ValueError."""
        vg = self._make_visual_gen_with_mock_executor()
        params = self._make_params()

        with pytest.raises(ValueError, match="Per-item negative_prompt is not supported"):
            vg.generate_async(
                inputs=[
                    {"prompt": "a sunset", "negative_prompt": "dark"},
                    {"prompt": "a city", "negative_prompt": "light"},
                ],
                params=params,
            )

    def test_empty_batch_raises(self):
        """Empty batch input raises ValueError."""
        vg = self._make_visual_gen_with_mock_executor()
        params = self._make_params()

        with pytest.raises(ValueError, match="at least one item"):
            vg.generate_async(inputs=[], params=params)

    def test_missing_prompt_in_dict_raises(self):
        """Dict without 'prompt' key raises ValueError."""
        vg = self._make_visual_gen_with_mock_executor()
        params = self._make_params()

        with pytest.raises(ValueError, match="missing 'prompt'"):
            vg.generate_async(
                inputs=[{"negative_prompt": "dark"}],
                params=params,
            )

    def test_invalid_input_raises(self):
        """Invalid input type raises ValueError."""
        vg = self._make_visual_gen_with_mock_executor()
        params = self._make_params()

        with pytest.raises(ValueError, match="Invalid inputs type"):
            vg.generate_async(inputs=12345, params=params)

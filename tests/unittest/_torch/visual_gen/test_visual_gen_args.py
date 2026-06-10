# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for VisualGenArgs construction, validation, and serialization."""

import itertools
import pickle
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from tensorrt_llm.visual_gen.args import (
    AttentionConfig,
    CacheDiTConfig,
    CompilationConfig,
    CudaGraphConfig,
    ParallelConfig,
    QuantAttentionConfig,
    TeaCacheConfig,
    TorchCompileConfig,
    VisualGenArgs,
)


class TestVisualGenArgsStrictValidation:
    """extra='forbid' rejects unknown fields at every nesting level."""

    def test_unknown_top_level_field_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            VisualGenArgs(model="/tmp/model", unknown_field="bad")

    def test_typo_field_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            VisualGenArgs(model="/tmp/model", chekpoint_path="/typo")

    def test_nested_parallel_unknown_field_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            VisualGenArgs(
                model="/tmp/model",
                parallel_config={"cfg_size": 1, "nonexistent_param": 42},
            )

    def test_nested_attention_unknown_field_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            AttentionConfig(backend="VANILLA", extra_key="bad")

    def test_nested_teacache_unknown_field_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            TeaCacheConfig(unknown_opt=True)

    def test_nested_torch_compile_unknown_field_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            TorchCompileConfig(enable=True, bad_key=1)

    def test_nested_cuda_graph_unknown_field_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            CudaGraphConfig(enable=False, extra=True)

    def test_nested_warmup_unknown_field_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            CompilationConfig(resolutions=[(480, 832)], bad_field=True)

    def test_legacy_linear_field_rejected(self):
        """The removed 'linear' YAML field must now cause an error."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            VisualGenArgs(
                model="/tmp/model",
                linear={"type": "default"},
            )


class TestAttentionConfigQuantValidation:
    """Unsupported quantized-attention recipes are rejected with ValueError."""

    def test_quant_config_rejected_on_unsupported_backend(self):
        with pytest.raises(ValidationError, match="requires backend in"):
            AttentionConfig(
                backend="VANILLA",
                quant_attention_config=QuantAttentionConfig(),
            )

    def test_quant_config_rejected_when_unsupported(self):
        with pytest.raises(ValidationError, match="Unsupported quant_attention_config"):
            AttentionConfig(
                backend="TRTLLM",
                quant_attention_config=QuantAttentionConfig(
                    qk_dtype="int8", q_block_size=1, k_block_size=127, v_block_size=1
                ),
            )

    def test_supported_quant_config_sage(self):
        attention = AttentionConfig(
            backend="TRTLLM",
            quant_attention_config=QuantAttentionConfig(
                qk_dtype="int8", q_block_size=1, k_block_size=16, v_block_size=1
            ),
        )

        assert attention.quant_attention_config is not None

    def test_supported_quant_config_cute(self):
        attention = AttentionConfig(
            backend="CUTEDSL",
            quant_attention_config=QuantAttentionConfig(qk_dtype="bf16", v_dtype="fp8"),
        )

        assert attention.quant_attention_config is not None


class TestPipelineRegistryUnique:
    """Guard against duplicate HF IDs across PIPELINE_REGISTRY entries.

    A single HF model id must dispatch to exactly one entry so
    ``VisualGen.supported_models()`` doesn't list duplicates and
    ``VisualGen.pipeline_config(id)`` returns a deterministic
    defaults dict. When a checkpoint has multiple runtime variants
    (e.g. LTX2 one-stage vs two-stage), only the user-facing entry
    should carry the hf_id — variants swap in at load time via
    ``BasePipeline.resolve_variant()``.
    """

    def test_no_duplicate_hf_ids(self):
        from tensorrt_llm._torch.visual_gen.pipeline_registry import PIPELINE_REGISTRY

        seen: dict[str, str] = {}
        for entry_name, entry in PIPELINE_REGISTRY.items():
            for hf_id in entry.hf_ids:
                assert hf_id not in seen, (
                    f"HF id {hf_id!r} registered on both {seen[hf_id]!r} and "
                    f"{entry_name!r}; only the user-facing entry should carry it."
                )
                seen[hf_id] = entry_name

    def test_supported_models_no_duplicates(self):
        from tensorrt_llm.visual_gen import VisualGen

        ids = VisualGen.supported_models()
        assert len(ids) == len(set(ids)), f"VisualGen.supported_models() returned duplicates: {ids}"


class TestVisualGenArgsCacheBackend:
    def test_cache_dit_nested_config(self):
        args = VisualGenArgs(
            model="/tmp/model",
            cache_config=CacheDiTConfig(Fn_compute_blocks=2, max_warmup_steps=4),
        )
        assert isinstance(args.cache_config, CacheDiTConfig)
        assert args.cache_dit.Fn_compute_blocks == 2
        assert args.cache_dit.max_warmup_steps == 4

    def test_cache_union_discriminated_teacache(self):
        args = VisualGenArgs(
            model="/tmp/model",
            cache_config=TeaCacheConfig(teacache_thresh=0.11),
        )
        assert args.cache_backend == "teacache"
        assert isinstance(args.cache_config, TeaCacheConfig)
        assert args.teacache.teacache_thresh == 0.11

    def test_cache_default_is_none(self):
        args = VisualGenArgs(model="/tmp/model")
        assert args.cache_config is None
        assert args.cache_backend is None


class TestVisualGenArgsFromDict:
    """VisualGenArgs construction from dicts enforces strict validation."""

    def test_valid_dict(self):
        args = VisualGenArgs(
            **{
                "model": "/tmp/model",
                "parallel_config": {"cfg_size": 2, "ulysses_size": 1},
            }
        )
        assert args.model == "/tmp/model"
        assert args.parallel_config.cfg_size == 2

    def test_unknown_field_raises(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            VisualGenArgs(
                **{
                    "model": "/tmp/model",
                    "bad_key": 123,
                }
            )

    def test_nested_dict_auto_coerced(self):
        args = VisualGenArgs(
            **{
                "model": "/tmp/model",
                "attention_config": {"backend": "TRTLLM"},
                "cache_config": {"cache_backend": "teacache", "teacache_thresh": 0.3},
            }
        )
        assert isinstance(args.attention_config, AttentionConfig)
        assert args.attention_config.backend == "TRTLLM"
        assert isinstance(args.cache_config, TeaCacheConfig)
        assert args.teacache.teacache_thresh == 0.3

    def test_quant_config_dict_passthrough(self):
        """ModelOpt-format dicts are accepted as-is — they parse in PipelineLoader."""
        from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig

        raw = {"quant_algo": "FP8", "dynamic": True}
        args = VisualGenArgs(model="/tmp/model", quant_config=raw)
        # The dict stays a dict on the public schema (no eager coercion to
        # QuantConfig), so the derived dynamic flags don't leak onto VisualGenArgs.
        assert isinstance(args.quant_config, dict)
        assert args.quant_config["quant_algo"] == "FP8"
        # The same dict is the source of truth for the derived flags; verify
        # the pipeline-config parser will run on it.
        qc, _, dwq, _ = DiffusionPipelineConfig.load_diffusion_quant_config(args.quant_config)
        assert qc.quant_algo is not None
        assert dwq is True

    def test_quant_config_dict_does_not_leak_dynamic_flags(self):
        """AC-5: dynamic flags are not part of the VisualGenArgs schema."""
        args = VisualGenArgs(
            model="/tmp/model",
            quant_config={"quant_algo": "FP8", "dynamic": True},
        )
        # Negative: removed fields raise ValidationError when set directly.
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            VisualGenArgs(model="/tmp/model", dynamic_weight_quant=True)
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            VisualGenArgs(model="/tmp/model", force_dynamic_quantization=True)
        # The names do not appear in model_fields or model_dump output.
        fields = set(args.model_fields)
        assert "dynamic_weight_quant" not in fields
        assert "force_dynamic_quantization" not in fields
        dumped = set(args.model_dump())
        assert "dynamic_weight_quant" not in dumped
        assert "force_dynamic_quantization" not in dumped


class TestVisualGenArgsFromYaml:
    """from_yaml round-trips through a YAML file."""

    def test_from_yaml_basic(self, tmp_path):
        yaml_path = tmp_path / "config.yml"
        yaml_path.write_text(
            "model: /tmp/model\nparallel_config:\n  cfg_size: 2\n  ulysses_size: 1\n"
        )
        args = VisualGenArgs.from_yaml(yaml_path)
        assert args.model == "/tmp/model"
        assert args.parallel_config.cfg_size == 2

    def test_from_yaml_with_overrides(self, tmp_path):
        yaml_path = tmp_path / "config.yml"
        yaml_path.write_text("model: /tmp/model\nrevision: original\n")
        args = VisualGenArgs.from_yaml(yaml_path, revision="override")
        assert args.revision == "override"

    def test_from_yaml_unknown_field_raises(self, tmp_path):
        yaml_path = tmp_path / "bad.yml"
        yaml_path.write_text("model: /tmp/model\nlinear:\n  type: default\n")
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            VisualGenArgs.from_yaml(yaml_path)


class TestParallelConfigValidation:
    """ParallelConfig no longer checks WORLD_SIZE at construction time."""

    def test_large_parallel_no_env_check(self):
        pc = ParallelConfig(cfg_size=2, ulysses_size=4)
        assert pc.total_parallel_size == 8
        assert pc.n_workers == 8

    def test_validate_world_size_passes(self):
        pc = ParallelConfig(cfg_size=2, ulysses_size=2)
        pc.validate_world_size(4)

    def test_validate_world_size_fails(self):
        pc = ParallelConfig(cfg_size=2, ulysses_size=4)
        with pytest.raises(ValueError, match="exceeds world_size"):
            pc.validate_world_size(4)

    def test_attn2d_config_basic(self):
        pc = ParallelConfig(attn2d_size=(2, 2))
        assert pc.seq_parallel_size == 4
        assert pc.total_parallel_size == 4
        assert pc.n_workers == 4

    def test_attn2d_with_cfg(self):
        pc = ParallelConfig(cfg_size=2, attn2d_size=(2, 2))
        assert pc.seq_parallel_size == 4
        assert pc.total_parallel_size == 8
        assert pc.n_workers == 8

    def test_attn2d_validate_world_size_passes(self):
        pc = ParallelConfig(cfg_size=2, attn2d_size=(2, 2))
        pc.validate_world_size(8)

    def test_attn2d_validate_world_size_fails(self):
        pc = ParallelConfig(cfg_size=2, attn2d_size=(2, 2))
        with pytest.raises(ValueError, match="exceeds world_size"):
            pc.validate_world_size(4)

    def test_attn2d_asymmetric_mesh(self):
        pc = ParallelConfig(attn2d_size=(1, 4))
        assert pc.seq_parallel_size == 4

    def test_seq_parallel_size_ulysses(self):
        pc = ParallelConfig(ulysses_size=4)
        assert pc.seq_parallel_size == 4

    def test_attn2d_default_is_disabled(self):
        pc = ParallelConfig()
        assert pc.seq_parallel_size == 1

    def test_parallel_vae_size_must_be_positive(self):
        with pytest.raises(ValidationError):
            ParallelConfig(parallel_vae_size=0)

    def test_attn2d_and_ulysses_seq_parallel_size(self):
        pc = ParallelConfig(
            attn2d_size=(2, 2),
            ulysses_size=2,
        )
        assert pc.seq_parallel_size == 8


class TestVisualGenArgsPickle:
    """VisualGenArgs must survive pickle round-trip (mp.Process spawn)."""

    def test_pickle_roundtrip(self):
        args = VisualGenArgs(
            model="/tmp/model",
            compilation_config=CompilationConfig(skip_warmup=True),
            parallel_config=ParallelConfig(cfg_size=2, ulysses_size=1),
            attention_config=AttentionConfig(backend="TRTLLM"),
            quant_config={"quant_algo": "FP8", "dynamic": True},
        )
        data = pickle.dumps(args)
        restored = pickle.loads(data)

        assert restored.model == args.model
        assert restored.compilation_config.skip_warmup is True
        assert restored.parallel_config.cfg_size == 2
        assert restored.attention_config.backend == "TRTLLM"
        # quant_config is the user dict (lazy-parsed in PipelineLoader)
        assert restored.quant_config["quant_algo"] == "FP8"

    def test_model_copy_override(self):
        args = VisualGenArgs(model="/tmp/model")
        updated = args.model_copy(update={"revision": "v2"})
        assert updated.revision == "v2"
        assert args.revision is None


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


class TestVisualGenInputParsing:
    """Test that VisualGen.generate_async() correctly parses inputs.

    Uses mocking to avoid spawning GPU worker processes.
    """

    def _make_visual_gen_with_mock_executor(self):
        """Create a VisualGen instance with a mocked executor."""
        from tensorrt_llm.visual_gen import VisualGen

        # Patch __init__ to avoid spawning workers
        with patch.object(VisualGen, "__init__", lambda self, *a, **kw: None):
            vg = VisualGen.__new__(VisualGen)
            vg.model = "/tmp/fake"
            vg.args = VisualGenArgs(model="/tmp/fake")
            vg._req_counter = itertools.count()
            vg.executor = MagicMock()
            return vg

    def test_string_input(self):
        """String input → single-element list in DiffusionRequest."""
        vg = self._make_visual_gen_with_mock_executor()

        vg.generate_async(inputs="a cat")

        call_args = vg.executor.enqueue_requests.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0].prompt == ["a cat"]

    def test_list_of_strings_input(self):
        """List of strings → batch prompt in single DiffusionRequest."""
        vg = self._make_visual_gen_with_mock_executor()

        vg.generate_async(inputs=["a sunset", "a city"])

        call_args = vg.executor.enqueue_requests.call_args[0][0]
        assert len(call_args) == 1
        req = call_args[0]
        assert req.prompt == ["a sunset", "a city"]

    def test_params_default_none(self):
        """Omitting params passes None; executor materializes defaults later."""
        vg = self._make_visual_gen_with_mock_executor()

        vg.generate_async(inputs="a cat")

        call_args = vg.executor.enqueue_requests.call_args[0][0]
        req = call_args[0]
        assert req.params is None

    def test_negative_prompt_via_params(self):
        """negative_prompt is passed through params, not inputs."""
        from tensorrt_llm.visual_gen import VisualGenParams

        vg = self._make_visual_gen_with_mock_executor()
        params = VisualGenParams(negative_prompt="blurry")

        vg.generate_async(inputs="a cat", params=params)

        call_args = vg.executor.enqueue_requests.call_args[0][0]
        assert call_args[0].params.negative_prompt == "blurry"

    def test_empty_batch_raises(self):
        """Empty batch input raises ValueError."""
        vg = self._make_visual_gen_with_mock_executor()

        with pytest.raises(ValueError, match="at least one item"):
            vg.generate_async(inputs=[])

    def test_invalid_input_raises(self):
        """Invalid input type raises ValueError."""
        vg = self._make_visual_gen_with_mock_executor()

        with pytest.raises(ValueError, match="Invalid inputs type"):
            vg.generate_async(inputs=12345)

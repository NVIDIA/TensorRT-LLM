# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU pipeline configuration tests for LTX-2.3 generation and retake."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from safetensors.torch import save_file

# Importing models applies the pipeline registration decorators.
from tensorrt_llm._torch.visual_gen import models  # noqa: F401
from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.models.ltx23.pipeline_ltx23 import LTX23Pipeline
from tensorrt_llm._torch.visual_gen.models.ltx23.pipeline_ltx23_retake import LTX23RetakePipeline
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm._torch.visual_gen.pipeline_registry import AutoPipeline
from tensorrt_llm.visual_gen.args import VisualGenArgs

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def ltx23_checkpoint(tmp_path: Path) -> str:
    """Write only the native metadata needed for detection and configuration."""
    checkpoint = tmp_path / "ltx23.safetensors"
    native_config = {
        "transformer": {
            "cross_attention_adaln": True,
            "caption_proj_before_connector": True,
            "caption_projection_first_linear": False,
            "caption_projection_second_linear": False,
            "caption_proj_input_norm": False,
        },
        "vae": {},
    }
    save_file({}, str(checkpoint), metadata={"config": json.dumps(native_config)})
    return str(checkpoint)


def test_ltx23_pipeline_config_defaults(ltx23_checkpoint: str) -> None:
    args = VisualGenArgs(model=ltx23_checkpoint)
    resolved = PipelineLoader(args, device="cpu")._resolve_pipeline_config(ltx23_checkpoint)

    assert AutoPipeline._detect_from_checkpoint(ltx23_checkpoint) == "LTX23Pipeline"
    assert resolved == {
        "workflow": "generation",
        "text_encoder_path": "google/gemma-3-12b-it-qat-q4_0-unquantized",
    }
    assert args.pipeline_config == {}


def test_ltx23_pipeline_config_user_overrides(ltx23_checkpoint: str) -> None:
    overrides = {"workflow": "retake", "text_encoder_path": "/local/gemma"}
    args = VisualGenArgs(model=ltx23_checkpoint, pipeline_config=overrides)

    assert (
        PipelineLoader(args, device="cpu")._resolve_pipeline_config(ltx23_checkpoint) == overrides
    )


def test_ltx23_pipeline_config_rejects_unknown_keys(ltx23_checkpoint: str) -> None:
    args = VisualGenArgs(model=ltx23_checkpoint, pipeline_config={"unknown_option": True})
    with pytest.raises(ValueError, match="Unknown pipeline_config keys for LTX23Pipeline"):
        PipelineLoader(args, device="cpu")._resolve_pipeline_config(ltx23_checkpoint)


@pytest.mark.parametrize(
    "workflow, expected", [("generation", LTX23Pipeline), ("retake", LTX23RetakePipeline)]
)
def test_ltx23_config_preserves_resolved_options(
    ltx23_checkpoint: str, workflow: str, expected: type
) -> None:
    args = VisualGenArgs(
        model=ltx23_checkpoint,
        pipeline_config={"workflow": workflow},
        attention_config={"backend": "VANILLA"},
    )
    resolved = PipelineLoader(args, device="cpu")._resolve_pipeline_config(ltx23_checkpoint)
    config = DiffusionPipelineConfig.from_pretrained(
        ltx23_checkpoint, args=args, pipeline_config=resolved, device="cpu"
    )

    assert config.extra_attrs["workflow"] == workflow
    assert config.extra_attrs["text_encoder_path"] == resolved["text_encoder_path"]
    assert (
        config.extra_attrs["monolithic_safetensors_config"]["transformer"]["cross_attention_adaln"]
        is True
    )
    assert LTX23Pipeline.resolve_variant(config) is expected


def test_ltx23_config_rejects_unknown_workflow(ltx23_checkpoint: str) -> None:
    args = VisualGenArgs(model=ltx23_checkpoint, pipeline_config={"workflow": "unknown"})
    with pytest.raises(ValueError, match="LTX-2.3 workflow must be"):
        DiffusionPipelineConfig.from_pretrained(ltx23_checkpoint, args=args, device="cpu")


@pytest.mark.parametrize("pipeline_class", [LTX23Pipeline, LTX23RetakePipeline])
@pytest.mark.parametrize("cfg_size", [1, 2])
def test_ltx23_config_requires_cfg_size_one(
    ltx23_checkpoint: str, pipeline_class: type, cfg_size: int
) -> None:
    """Reject CFG parallelism before initializing components; keep cfg_size=1 valid."""
    args = VisualGenArgs(model=ltx23_checkpoint, parallel_config={"cfg_size": cfg_size})
    config = DiffusionPipelineConfig.from_pretrained(ltx23_checkpoint, args=args, device="cpu")
    with patch.object(BasePipeline, "__init__", return_value=None) as initialize:
        if cfg_size == 1:
            pipeline_class(config)
            initialize.assert_called_once_with(config)
        else:
            with pytest.raises(NotImplementedError, match=r"requires parallel\.cfg_size=1"):
                pipeline_class(config)
            initialize.assert_not_called()


@pytest.mark.parametrize(
    "recipe, workflow, quant_algo",
    [
        ("ltx23-t2v-bf16-1gpu.yaml", "generation", None),
        ("ltx23-t2v-fp8-1gpu.yaml", "generation", "FP8_BLOCK_SCALES"),
        ("ltx23-t2v-fp4-1gpu.yaml", "generation", "NVFP4"),
        ("ltx23-retake-bf16-1gpu.yaml", "retake", None),
        ("ltx23-t2v-bf16-2gpu.yaml", "generation", None),
        ("ltx23-retake-bf16-2gpu.yaml", "retake", None),
        ("ltx23-retake-fp8-1gpu.yaml", "retake", "FP8_BLOCK_SCALES"),
        ("ltx23-retake-fp4-1gpu.yaml", "retake", "NVFP4"),
    ],
)
def test_ltx23_example_recipe_config(
    ltx23_checkpoint: str, recipe: str, workflow: str, quant_algo: str | None
) -> None:
    config_dir = Path(__file__).resolve().parents[4] / "examples" / "visual_gen" / "configs"
    args = VisualGenArgs.from_yaml(config_dir / recipe, model=ltx23_checkpoint)
    resolved = PipelineLoader(args, device="cpu")._resolve_pipeline_config(ltx23_checkpoint)

    assert resolved["workflow"] == workflow
    if isinstance(args.quant_config, dict):
        assert args.quant_config["quant_algo"] == quant_algo
        assert args.quant_config["dynamic"] is True
    else:
        assert args.quant_config.quant_algo == quant_algo
    assert args.torch_compile_config.enable is True
    assert args.cuda_graph_config.enable is True
    if recipe.endswith("2gpu.yaml"):
        assert args.parallel_config.cfg_size == 1
        assert args.parallel_config.ulysses_size == 2
        assert args.parallel_config.parallel_vae_size == 2
        assert args.attention_config.backend == "VANILLA"
    if quant_algo is not None:
        assert args.attention_config.backend == "FLASHINFER"
        expected_qk = "nvfp4" if quant_algo == "NVFP4" else "mxfp8"
        assert args.attention_config.quant_attention_config.qk_dtype == expected_qk
        assert args.attention_config.quant_attention_config.v_dtype == "fp8"
    if workflow == "retake":
        assert args.pipeline_config == {"workflow": "retake"}
    config = DiffusionPipelineConfig.from_pretrained(
        ltx23_checkpoint, args=args, pipeline_config=resolved, device="cpu"
    )
    assert config.quant_config.quant_algo == quant_algo
    assert config.dynamic_weight_quant is (quant_algo is not None)
    assert config.extra_attrs["workflow"] == workflow
    if workflow == "retake":
        assert LTX23Pipeline.resolve_variant(config) is LTX23RetakePipeline
        assert (
            config.attention.quant_attention_config == args.attention_config.quant_attention_config
        )

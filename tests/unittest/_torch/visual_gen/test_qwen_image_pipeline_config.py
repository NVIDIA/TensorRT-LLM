# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline-level configuration tests for Qwen-Image."""

import json
from types import SimpleNamespace

import pytest

# Importing the models package applies the Qwen-Image registration side effect.
from tensorrt_llm._torch.visual_gen import models  # noqa: F401
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.qwen_image import QwenJointAttention
from tensorrt_llm._torch.visual_gen.modules.attention import QKVMode
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.visual_gen.args import (
    AttentionConfig,
    CudaGraphConfig,
    ParallelConfig,
    TorchCompileConfig,
    VisualGenArgs,
)


def _write_minimal_qwen_checkpoint(tmp_path):
    """Create the minimum diffusers layout needed by PipelineLoader config code."""
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "QwenImagePipeline",
                "transformer": ["diffusers", "QwenImageTransformer2DModel"],
            }
        )
    )
    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "config.json").write_text(
        json.dumps({"_class_name": "QwenImageTransformer2DModel"})
    )
    return tmp_path


def test_qwen_pipeline_config_defaults_to_empty_dict(tmp_path):
    checkpoint_dir = _write_minimal_qwen_checkpoint(tmp_path)

    args = VisualGenArgs(model=str(checkpoint_dir))
    resolved = PipelineLoader(args)._resolve_pipeline_config(str(checkpoint_dir))

    assert resolved == {}


def test_qwen_pipeline_config_rejects_unknown_keys(tmp_path):
    checkpoint_dir = _write_minimal_qwen_checkpoint(tmp_path)

    args = VisualGenArgs(
        model=str(checkpoint_dir),
        pipeline_config={"text_encoder_path": "/tmp/not-a-qwen-knob"},
    )
    with pytest.raises(ValueError, match="Unknown pipeline_config keys for QwenImagePipeline"):
        PipelineLoader(args)._resolve_pipeline_config(str(checkpoint_dir))


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        pytest.param(
            VisualGenArgs(model="Qwen/Qwen-Image"),
            {
                "backend": "VANILLA",
                "compile": True,
                "autotune": True,
                "cuda_graph": False,
                "n_workers": 1,
            },
            id="default",
        ),
        pytest.param(
            VisualGenArgs(
                model="Qwen/Qwen-Image",
                torch_compile_config=TorchCompileConfig(enable=False, enable_autotune=False),
                cuda_graph_config=CudaGraphConfig(enable=True),
            ),
            {
                "backend": "VANILLA",
                "compile": False,
                "autotune": False,
                "cuda_graph": True,
                "n_workers": 1,
            },
            id="cuda-graph",
        ),
        pytest.param(
            VisualGenArgs(
                model="Qwen/Qwen-Image",
                attention_config=AttentionConfig(backend="FA4"),
                parallel_config=ParallelConfig(attn2d_size=(2, 2)),
            ),
            {
                "backend": "FA4",
                "compile": True,
                "autotune": True,
                "cuda_graph": False,
                "n_workers": 4,
            },
            id="attention2d",
        ),
        pytest.param(
            VisualGenArgs(
                model="Qwen/Qwen-Image",
                attention_config=AttentionConfig(backend="FA4"),
                parallel_config=ParallelConfig(ring_size=2),
            ),
            {
                "backend": "FA4",
                "compile": True,
                "autotune": True,
                "cuda_graph": False,
                "n_workers": 2,
            },
            id="ring",
        ),
        pytest.param(
            VisualGenArgs(
                model="Qwen/Qwen-Image",
                attention_config=AttentionConfig(backend="TRTLLM"),
            ),
            {
                "backend": "TRTLLM",
                "compile": True,
                "autotune": True,
                "cuda_graph": False,
                "n_workers": 1,
            },
            id="trtllm-backend",
        ),
    ],
)
def test_qwen_pipeline_feature_args(args, expected):
    assert args.attention_config.backend == expected["backend"]
    assert args.torch_compile_config.enable is expected["compile"]
    assert args.torch_compile_config.enable_autotune is expected["autotune"]
    assert args.cuda_graph_config.enable is expected["cuda_graph"]
    assert args.parallel_config.n_workers == expected["n_workers"]


@pytest.mark.parametrize(
    ("quant_config", "quant_algo", "group_size", "force_dynamic_quantization"),
    [
        pytest.param({}, None, 128, False, id="bf16"),
        pytest.param(
            {"quant_algo": "FP8", "dynamic": True},
            QuantAlgo.FP8,
            None,
            False,
            id="dynamic-fp8-tensor",
        ),
        pytest.param(
            {"quant_algo": "FP8_BLOCK_SCALES", "dynamic": True},
            QuantAlgo.FP8_BLOCK_SCALES,
            128,
            False,
            id="dynamic-fp8-blockwise",
        ),
        pytest.param(
            {"quant_algo": "NVFP4", "dynamic": True},
            QuantAlgo.NVFP4,
            16,
            True,
            id="dynamic-fp4",
        ),
    ],
)
def test_qwen_pipeline_quant_config_parses_from_args(
    tmp_path, quant_config, quant_algo, group_size, force_dynamic_quantization
):
    checkpoint_dir = _write_minimal_qwen_checkpoint(tmp_path)
    args = VisualGenArgs(model=str(checkpoint_dir), quant_config=quant_config)

    config = DiffusionModelConfig.from_pretrained(str(checkpoint_dir), args=args)

    assert config.quant_config.quant_algo == quant_algo
    assert config.quant_config.group_size == group_size
    assert config.dynamic_weight_quant is (quant_algo is not None)
    assert config.force_dynamic_quantization is force_dynamic_quantization


@pytest.mark.parametrize(
    ("visual_gen_mapping", "not_wrapped_as"),
    [
        pytest.param(
            SimpleNamespace(
                attn2d_row_size=2,
                attn2d_col_size=2,
                ring_size=1,
                ulysses_size=1,
            ),
            "Attention2DAttention",
            id="attention2d",
        ),
        pytest.param(
            SimpleNamespace(
                attn2d_row_size=1,
                attn2d_col_size=1,
                ring_size=2,
                ulysses_size=1,
            ),
            "RingAttention",
            id="ring",
        ),
    ],
)
def test_qwen_joint_attention_keeps_separate_qkv_path_unwrapped(visual_gen_mapping, not_wrapped_as):
    config = DiffusionModelConfig(
        attention=AttentionConfig(backend="VANILLA"),
        visual_gen_mapping=visual_gen_mapping,
    )

    attention = QwenJointAttention(
        dim=16,
        num_attention_heads=2,
        attention_head_dim=8,
        config=config,
    )

    assert attention.qkv_mode == QKVMode.SEPARATE_QKV
    assert attention.attn.__class__.__name__ != not_wrapped_as

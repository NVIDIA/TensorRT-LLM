# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline-level configuration tests for Qwen-Image."""

import json
from types import SimpleNamespace

import pytest
import torch

# Importing the models package applies the Qwen-Image registration side effect.
from tensorrt_llm._torch.visual_gen import models  # noqa: F401
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig, DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.models.qwen_image import (
    QwenImageTransformer2DModel,
    QwenJointAttention,
    apply_rotary_emb_qwen,
)
from tensorrt_llm._torch.visual_gen.models.qwen_image.transformer_qwen_image import (
    qwen_complex_freqs_to_cos_sin,
)
from tensorrt_llm._torch.visual_gen.modules.attention import QKVMode, apply_rotary_emb
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

    config = DiffusionPipelineConfig.from_pretrained(str(checkpoint_dir), args=args)

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


def test_qwen_complex_freqs_convert_to_shared_rope_format():
    torch.manual_seed(0)
    seq_len = 8
    head_dim = 16
    x = torch.randn(2, seq_len, 3, head_dim)
    phases = torch.randn(seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(phases), phases)

    freqs_cos, freqs_sin = qwen_complex_freqs_to_cos_sin(freqs_cis)

    ref = apply_rotary_emb_qwen(x, freqs_cis)
    out = apply_rotary_emb(x, freqs_cos, freqs_sin)
    torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)


def test_qwen_joint_attention_fused_rope_passes_2d_freqs_to_kernel(monkeypatch):
    torch.manual_seed(0)
    txt_seq = 5
    img_seq = 7
    batch_size = 2
    head_dim = 8
    attention = QwenJointAttention(
        dim=16,
        num_attention_heads=2,
        attention_head_dim=head_dim,
        config=DiffusionModelConfig(),
    )
    captured = {}

    def fake_apply_packed_qk_norm_rope(qkv, freqs_cos, freqs_sin, **kwargs):
        captured["cos_shape"] = tuple(freqs_cos.shape)
        captured["sin_shape"] = tuple(freqs_sin.shape)

    monkeypatch.setattr(attention, "apply_packed_qk_norm_rope", fake_apply_packed_qk_norm_rope)

    hidden_states = torch.randn(batch_size, img_seq, 16)
    encoder_hidden_states = torch.randn(batch_size, txt_seq, 16)
    img_phases = torch.randn(img_seq, head_dim // 2)
    txt_phases = torch.randn(txt_seq, head_dim // 2)
    image_rotary_emb = (
        torch.polar(torch.ones_like(img_phases), img_phases),
        torch.polar(torch.ones_like(txt_phases), txt_phases),
    )

    attention._prepare_qkv_fused(hidden_states, encoder_hidden_states, image_rotary_emb)

    assert captured == {
        "cos_shape": (batch_size * (txt_seq + img_seq), head_dim),
        "sin_shape": (batch_size * (txt_seq + img_seq), head_dim),
    }


def test_qwen_joint_attention_fused_rope_requires_qk_norm():
    attention = QwenJointAttention(
        dim=16,
        num_attention_heads=2,
        attention_head_dim=8,
        config=DiffusionModelConfig(),
    )
    hidden_states = SimpleNamespace(is_cuda=True, dtype=torch.bfloat16)
    image_rotary_emb = (object(), object())

    assert attention._use_fused_qk_norm_rope(hidden_states, image_rotary_emb)

    attention.qk_norm = False
    assert not attention._use_fused_qk_norm_rope(hidden_states, image_rotary_emb)


def test_qwen_transformer_cpu_fallback_uses_unfused_qk_norm_rope():
    torch.manual_seed(0)
    model = QwenImageTransformer2DModel(
        model_config=DiffusionModelConfig(),
        patch_size=1,
        in_channels=4,
        out_channels=4,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        caption_channels=16,
        axes_dims_rope=(4, 6, 6),
    ).eval()

    hidden_states = torch.randn(1, 4, 4)
    encoder_hidden_states = torch.randn(1, 5, 16)
    timestep = torch.tensor([1.0])

    assert model.transformer_blocks[0].attn.fuse_qk_norm_rope
    out = model(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_shapes=[(1, 2, 2)],
        txt_seq_lens=torch.tensor([5]),
    )

    assert out[0].shape == hidden_states.shape

# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry smoke tests for Qwen-Image.

These tests make sure Qwen-Image checkpoints route through VisualGen's
``AutoPipeline`` instead of failing with ``Unknown pipeline: ''`` from
registry detection.
"""

import json

import pytest
import torch

from tensorrt_llm._torch.modules.linear import NVFP4LinearMethod, UnquantizedLinearMethod

# Importing the models package side-effects the ``@register_pipeline``
# decorator on ``QwenImagePipeline`` being applied, which is what we are
# testing here.
from tensorrt_llm._torch.visual_gen import models  # noqa: F401
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.qwen_image import (
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)
from tensorrt_llm._torch.visual_gen.pipeline_registry import PIPELINE_REGISTRY, AutoPipeline
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.visual_gen.args import AttentionConfig


def test_qwen_image_pipeline_is_registered():
    """@register_pipeline("QwenImagePipeline") must have been applied."""
    assert "QwenImagePipeline" in PIPELINE_REGISTRY
    assert PIPELINE_REGISTRY["QwenImagePipeline"].pipeline_cls is QwenImagePipeline


def test_auto_pipeline_detects_qwen_image_class_name(tmp_path):
    """model_index.json with _class_name=QwenImagePipeline resolves."""
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "QwenImagePipeline"}))
    assert AutoPipeline._detect_from_checkpoint(str(tmp_path)) == "QwenImagePipeline"


def test_transformer_constructs_with_defaults():
    """The full transformer instantiates with the documented defaults.

    Doesn't touch real weights -- just verifies that the class graph is
    buildable and exposes the expected public attributes.
    """
    model = QwenImageTransformer2DModel(
        model_config=None,
        num_layers=2,  # tiny to keep CPU instantiation fast
    )
    assert len(model.transformer_blocks) == 2
    assert model.inner_dim == 24 * 128
    assert hasattr(model, "img_in")
    assert hasattr(model, "txt_in")
    assert hasattr(model, "txt_norm")
    assert hasattr(model, "norm_out")
    assert hasattr(model, "proj_out")
    assert hasattr(model, "pos_embed")


def test_transformer_load_weights_detects_mismatch():
    """load_weights should surface a clear RuntimeError on a bad dict."""
    model = QwenImageTransformer2DModel(model_config=None, num_layers=2)
    with pytest.raises(RuntimeError, match=r"Missing keys|Unexpected keys"):
        model.load_weights({})


def test_transformer_applies_quant_config_ignore_list() -> None:
    """Qwen-Image should honor selective dynamic quantization exclusions."""
    model_config = DiffusionModelConfig(
        quant_config=QuantConfig(
            quant_algo=QuantAlgo.NVFP4,
            exclude_modules=[
                "transformer_blocks.0*",
                "img_in",
                "proj_out",
            ],
        ),
        dynamic_weight_quant=True,
        force_dynamic_quantization=True,
    )
    model = QwenImageTransformer2DModel(model_config=model_config, num_layers=2)

    assert model.img_in.quant_config.quant_algo is None
    assert model.proj_out.quant_config.quant_algo is None
    assert model.transformer_blocks[0].attn.add_q_proj.quant_config.quant_algo is None
    assert model.transformer_blocks[0].img_mlp.up_proj.quant_config.quant_algo is None
    assert isinstance(model.img_in.quant_method, UnquantizedLinearMethod)
    assert isinstance(model.proj_out.quant_method, UnquantizedLinearMethod)
    assert isinstance(
        model.transformer_blocks[0].attn.add_q_proj.quant_method, UnquantizedLinearMethod
    )
    assert isinstance(
        model.transformer_blocks[0].img_mlp.up_proj.quant_method, UnquantizedLinearMethod
    )

    assert model.txt_in.quant_config.quant_algo == QuantAlgo.NVFP4
    assert model.transformer_blocks[1].attn.add_q_proj.quant_config.quant_algo == QuantAlgo.NVFP4
    assert isinstance(model.txt_in.quant_method, NVFP4LinearMethod)
    assert isinstance(model.transformer_blocks[1].attn.add_q_proj.quant_method, NVFP4LinearMethod)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("with_text_mask", [False, True])
def test_transformer_forward_sanity(with_text_mask):
    """A tiny Qwen-Image transformer runs both unmasked and text-masked paths."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    model_config = DiffusionModelConfig(
        attention=AttentionConfig(backend="VANILLA"),
        skip_create_weights_in_init=False,
    )
    model = (
        QwenImageTransformer2DModel(
            model_config=model_config,
            patch_size=1,
            in_channels=4,
            out_channels=4,
            num_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=12,
            axes_dims_rope=(2, 2, 4),
        )
        .to(device, dtype=dtype)
        .eval()
    )

    hidden_states = torch.randn(1, 4, 4, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(1, 5, 12, device=device, dtype=dtype)
    encoder_hidden_states_mask = None
    if with_text_mask:
        encoder_hidden_states_mask = torch.tensor([[True, True, True, False, False]], device=device)

    with torch.inference_mode():
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            timestep=torch.tensor([0.5], device=device, dtype=dtype),
            img_shapes=[[[1, 2, 2]]],
        )

    assert isinstance(output, tuple)
    assert output[0].shape == (1, 4, 4)

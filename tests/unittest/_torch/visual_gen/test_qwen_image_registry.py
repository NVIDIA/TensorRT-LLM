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
from tensorrt_llm._torch.visual_gen.quantization.loader import DynamicLinearWeightLoader
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.visual_gen.args import AttentionConfig


def _tiny_static_nvfp4_model() -> QwenImageTransformer2DModel:
    """Build a CPU-sized static NVFP4 model with representative exclusions."""
    quant_config = QuantConfig(
        quant_algo=QuantAlgo.NVFP4,
        kv_cache_quant_algo=QuantAlgo.FP8,
        group_size=16,
        exclude_modules=[
            "img_in",
            "txt_in",
            "proj_out",
            "norm_out*",
            "transformer_blocks.0*",
        ],
    )
    model_config = DiffusionModelConfig(
        quant_config=quant_config,
        skip_create_weights_in_init=False,
    )
    return QwenImageTransformer2DModel(
        model_config=model_config,
        patch_size=1,
        in_channels=16,
        out_channels=16,
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=1,
        joint_attention_dim=16,
        axes_dims_rope=(4, 6, 6),
    )


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


def test_static_quant_excludes_high_precision_layers():
    """Layers in the checkpoint's ``ignore`` list build as unquantized.

    A statically pre-quantized ModelOpt NVFP4 checkpoint stores the excluded
    layers (embedders, output projection, first/last transformer blocks) in
    BF16 with no scales, so they must fall back to the unquantized Linear
    method while the rest stay NVFP4.
    """
    model = _tiny_static_nvfp4_model()

    # Excluded layers materialize directly in their unquantized layout.
    excluded = (
        model.img_in,
        model.txt_in,
        model.proj_out,
        model.norm_out.linear,
        model.transformer_blocks[0].attn.to_q,
    )
    for module in excluded:
        assert module.quant_config is not None
        assert module.quant_config.quant_algo is None
        assert module.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
        assert isinstance(module.quant_method, UnquantizedLinearMethod)
    assert model.img_in._weights_created
    assert model.img_in.weight.shape == (16, 16)
    assert not hasattr(model.img_in, "weight_scale")

    # Included layers materialize in the NVFP4 layout.
    keep = model.transformer_blocks[1].attn.to_q.quant_config
    assert keep is not None and keep.quant_algo == QuantAlgo.NVFP4
    assert keep.kv_cache_quant_algo == QuantAlgo.FP8
    quantized = model.transformer_blocks[1].attn.to_q
    assert isinstance(quantized.quant_method, NVFP4LinearMethod)
    assert quantized._weights_created
    assert quantized.weight.shape == (16, 8)
    assert hasattr(quantized, "weight_scale")

    # Applying module-local exclusions must not mutate the global config.
    assert model.model_config.quant_config is not None
    assert model.model_config.quant_config.quant_algo == QuantAlgo.NVFP4
    assert model.model_config.quant_config.kv_cache_quant_algo == QuantAlgo.FP8


def test_static_quant_load_allows_only_missing_derived_parameters(monkeypatch):
    """Static loading ignores quant helpers but still requires real weights."""
    model = _tiny_static_nvfp4_model()
    expected = dict(model.named_parameters())
    derived_param_names = ("alpha", "inv_input_scale", "kv_scales", "inv_kv_scales")
    derived = {
        name
        for name in expected
        if any(name.endswith(f".{param_name}") for param_name in derived_param_names)
    }
    for param_name in derived_param_names:
        assert any(name.endswith(f".{param_name}") for name in derived)

    checkpoint = {name: param.detach() for name, param in expected.items() if name not in derived}
    monkeypatch.setattr(
        DynamicLinearWeightLoader,
        "get_linear_weights",
        lambda self, module, full_name, weights: [],
    )
    monkeypatch.setattr(
        DynamicLinearWeightLoader,
        "filter_weights",
        lambda self, prefix, weights: {},
    )

    model.load_weights(checkpoint)

    real_weight = "transformer_blocks.1.attn.to_q.weight"
    checkpoint_missing_weight = checkpoint.copy()
    checkpoint_missing_weight.pop(real_weight)
    with pytest.raises(RuntimeError) as exc_info:
        model.load_weights(checkpoint_missing_weight)
    assert real_weight in str(exc_info.value)
    assert not any(name in str(exc_info.value) for name in derived)


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

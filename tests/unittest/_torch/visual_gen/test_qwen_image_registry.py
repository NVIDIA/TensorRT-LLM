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

from tensorrt_llm._torch.modules.linear import (
    FP8QDQLinearMethod,
    Linear,
    NVFP4LinearMethod,
    UnquantizedLinearMethod,
)

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


def _tiny_static_quant_model(quant_algo: QuantAlgo) -> QwenImageTransformer2DModel:
    """Build a CPU-sized static FP8/NVFP4 model with representative exclusions."""
    quant_config = QuantConfig(
        quant_algo=quant_algo,
        group_size=16 if quant_algo == QuantAlgo.NVFP4 else None,
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
        dynamic_weight_quant=False,
        force_dynamic_quantization=False,
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


@pytest.mark.parametrize("quant_algo", [QuantAlgo.NVFP4, QuantAlgo.FP8], ids=["nvfp4", "fp8"])
def test_static_quant_excludes_high_precision_layers(quant_algo: QuantAlgo) -> None:
    """Layers in the checkpoint's ``ignore`` list build as unquantized.

    When a static ModelOpt checkpoint excludes layers, those layers are stored
    in BF16 without scales. Embedders, the output projection, and selected
    transformer blocks must therefore use the unquantized Linear method while
    included layers retain the checkpoint's quantization format.
    """
    model = _tiny_static_quant_model(quant_algo)

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
        assert module.quant_config.kv_cache_quant_algo is None
        assert isinstance(module.quant_method, UnquantizedLinearMethod)
    assert model.img_in._weights_created
    assert model.img_in.weight.shape == (16, 16)
    assert model.img_in.weight.dtype == torch.bfloat16
    assert not hasattr(model.img_in, "weight_scale")

    # Included layers materialize in the selected static-quant layout.
    keep = model.transformer_blocks[1].attn.to_q.quant_config
    assert keep is not None and keep.quant_algo == quant_algo
    assert keep.kv_cache_quant_algo is None
    quantized = model.transformer_blocks[1].attn.to_q
    assert quantized._weights_created
    assert hasattr(quantized, "weight_scale")
    assert hasattr(quantized, "input_scale")

    if quant_algo == QuantAlgo.NVFP4:
        assert isinstance(quantized.quant_method, NVFP4LinearMethod)
        assert quantized.weight.dtype == torch.uint8
        assert quantized.weight.shape == (16, 8)
        assert quantized.weight_scale.dtype == torch.uint8
        assert quantized.weight_scale_2.dtype == torch.float32
    else:
        assert isinstance(quantized.quant_method, FP8QDQLinearMethod)
        assert quantized.weight.dtype == torch.float8_e4m3fn
        assert quantized.weight.shape == (16, 16)
        assert quantized.input_scale.dtype == torch.float32
        assert quantized.input_scale.shape == torch.Size([])
        assert quantized.weight_scale.dtype == torch.float32
        assert quantized.weight_scale.shape == torch.Size([])
        assert not hasattr(quantized, "weight_scale_2")

    # Applying module-local exclusions must not mutate the global config.
    assert model.model_config.quant_config is not None
    assert model.model_config.quant_config.quant_algo == quant_algo
    assert model.model_config.quant_config.kv_cache_quant_algo is None


@pytest.mark.parametrize("quant_algo", [QuantAlgo.NVFP4, QuantAlgo.FP8], ids=["nvfp4", "fp8"])
def test_static_quant_load_allows_only_missing_non_serialized_parameters(
    monkeypatch: pytest.MonkeyPatch, quant_algo: QuantAlgo
) -> None:
    """Static loading allows only format-specific, non-serialized helpers to be missing."""
    model = _tiny_static_quant_model(quant_algo)
    expected = dict(model.named_parameters())
    non_serialized_suffixes = {
        QuantAlgo.NVFP4: ("alpha", "inv_input_scale", "kv_scales", "inv_kv_scales"),
        QuantAlgo.FP8: ("inv_input_scale", "kv_scales", "inv_kv_scales"),
    }[quant_algo]
    quantized_module_names = {
        name
        for name, module in model.named_modules()
        if isinstance(module, Linear)
        and module.quant_config is not None
        and module.quant_config.quant_algo == quant_algo
    }
    assert quantized_module_names
    assert "transformer_blocks.1.attn.to_q" in quantized_module_names
    assert all(name.startswith("transformer_blocks.1.") for name in quantized_module_names)

    non_serialized = {
        f"{module_name}.{suffix}"
        for module_name in quantized_module_names
        for suffix in non_serialized_suffixes
    }
    assert model._non_serialized_quant_parameter_names() == non_serialized
    assert non_serialized <= expected.keys()
    if quant_algo == QuantAlgo.FP8:
        assert not any(name.endswith(".alpha") for name in expected)

    checkpoint = {
        name: param.detach() for name, param in expected.items() if name not in non_serialized
    }
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

    # These keys are present in real static ModelOpt checkpoints and are not
    # derived by TensorRT-LLM. Strict loading must reject each one if absent.
    required_suffixes = ["weight", "bias", "input_scale", "weight_scale"]
    if quant_algo == QuantAlgo.NVFP4:
        required_suffixes.append("weight_scale_2")
    for suffix in required_suffixes:
        required_key = f"transformer_blocks.1.attn.to_q.{suffix}"
        checkpoint_missing_key = checkpoint.copy()
        checkpoint_missing_key.pop(required_key)
        with pytest.raises(RuntimeError, match="Missing keys") as exc_info:
            model.load_weights(checkpoint_missing_key)
        assert required_key in str(exc_info.value)


def test_static_fp8_loads_serialized_weights_and_derives_inverse_scale() -> None:
    """Static FP8 loading copies checkpoint tensors and derives inverse input scales."""
    model = _tiny_static_quant_model(QuantAlgo.FP8)
    expected = dict(model.named_parameters())
    non_serialized = model._non_serialized_quant_parameter_names()
    checkpoint = {
        name: torch.zeros_like(param)
        for name, param in expected.items()
        if name not in non_serialized
    }
    for name, value in checkpoint.items():
        if name.endswith(".input_scale"):
            value.fill_(2.0)
        elif name.endswith(".weight_scale"):
            value.fill_(0.25)

    model.load_weights(checkpoint)

    target_name = "transformer_blocks.1.attn.to_q"
    target = model.transformer_blocks[1].attn.to_q
    assert torch.equal(target.weight, checkpoint[f"{target_name}.weight"])
    assert target.input_scale.item() == pytest.approx(2.0)
    assert target.inv_input_scale.item() == pytest.approx(0.5)
    assert target.weight_scale.item() == pytest.approx(0.25)


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

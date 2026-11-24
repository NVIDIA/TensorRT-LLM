# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models import MiniMaxH3ImagePipeline
from tensorrt_llm._torch.visual_gen.models.minimax_h3_image.packing import (
    generation_canvas,
    text_image_positions,
)
from tensorrt_llm._torch.visual_gen.models.minimax_h3_image.schedule import ImagePDDSchedule
from tensorrt_llm._torch.visual_gen.models.minimax_h3_image.transformer_minimax_h3_image import (
    MiniMaxH3ImageTransformer2DModel,
)
from tensorrt_llm._torch.visual_gen.pipeline_registry import PIPELINE_REGISTRY, AutoPipeline
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.visual_gen.args import AttentionConfig
from tensorrt_llm.visual_gen.params import VisualGenParams


@pytest.mark.parametrize(
    ("filename", "name"),
    [
        ("model_index.json", "MiniMaxH3ImagePipeline"),
        ("modular_model_index.json", "MiniMaxH3ImageModularPipeline"),
    ],
)
def test_minimax_h3_image_manifest(tmp_path, filename, name):
    (tmp_path / filename).write_text(json.dumps({"_class_name": name}))
    assert AutoPipeline._detect_from_checkpoint(str(tmp_path)) == name
    assert PIPELINE_REGISTRY[name].pipeline_cls is MiniMaxH3ImagePipeline
    entries = [e for e in PIPELINE_REGISTRY.values() if "MiniMaxAI/MiniMax-H3-Image" in e.hf_ids]
    assert len(entries) == 1
    assert entries[0].pipeline_cls is MiniMaxH3ImagePipeline


def test_minimax_h3_image_request_defaults():
    # Request metadata must be discoverable without allocating model weights.
    pipeline = MiniMaxH3ImagePipeline.__new__(MiniMaxH3ImagePipeline)
    defaults = pipeline.default_generation_params
    assert defaults["height"] == defaults["width"] == 2048
    assert defaults["num_frames"] == 1
    assert defaults["num_inference_steps"] == 6
    assert defaults["guidance_scale"] == 1.0
    assert defaults["max_sequence_length"] == 4096
    specs = pipeline.extra_param_specs
    assert set(specs) == {"min_generation_area"}
    area = specs["min_generation_area"]
    assert area.type == "int"
    assert area.default == 2048**2
    assert area.range == (1024, 4096**2)


# modular_model_index.json: round each knot through clean-ward time in FP32,
# including the penultimate one (which doesn't survive this round trip exactly).
_SAMPLE_SIGMAS = [
    1.0,
    0.9675236940383911,
    0.8875485062599182,
    0.75,
    0.5327739119529724,
    0.23200887441635132,
    0.0,
]


def test_minimax_h3_image_schedule():
    schedule = ImagePDDSchedule(torch.device("cpu"))
    times = torch.tensor([1 - sigma for sigma in _SAMPLE_SIGMAS], dtype=torch.float32)
    sigmas = 1 - times
    torch.testing.assert_close(schedule.timesteps, times[:-1], rtol=0, atol=0)
    torch.testing.assert_close(schedule.sigmas, sigmas, rtol=0, atol=0)
    x = torch.tensor([[-1, 2], [3, -4]], dtype=torch.bfloat16)
    direction = torch.tensor([[0.5, -0.25], [-0.75, 2]], dtype=torch.bfloat16)
    for i in range(6):
        v = direction * (i + 1)
        expected = x.float() + (sigmas[i] - sigmas[i + 1]) * v.float()
        x = schedule.step(i, x, v)
        torch.testing.assert_close(x, expected, rtol=0, atol=0)
    for index in (-1, 6):
        with pytest.raises(ValueError, match="PDD head"):
            schedule.step(index, x, x)
    with pytest.raises(ValueError, match="velocity row"):
        schedule.step(0, x, x[:1])


def test_minimax_h3_image_canvas():
    assert generation_canvas(512, 1024, 1024) == (512, 1024)
    assert generation_canvas(512, 512, 2048**2) == (2048, 2048)
    with pytest.raises(ValueError, match="min_generation_area"):
        generation_canvas(512, 512, 0)


def test_minimax_h3_image_packed_positions():
    # A 1x4 grid spans 1/2 unit vertically and 2 units horizontally,
    # anchored on the text length on the temporal RoPE axis.
    positions = text_image_positions(2, 1, 4, torch.device("cpu"))
    expected = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [2, 8, -16], [2, 8, 0], [2, 8, 16], [2, 8, 32]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(positions, expected, atol=0, rtol=0)


def _image_config(**overrides):
    architecture = dict(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=32,
        ffn_dim=64,
        num_layers=1,
        num_refiner_layers=1,
        in_channels=2,
        patch_size=(1, 2, 2),
        text_dim=8,
        freq_dim=8,
        time_embed_hidden_dim=32,
        time_embed_dim=16,
        rope_freq_dim=2,
        rope_theta=10000.0,
        norm_eps=1e-5,
        qk_norm_eps=1e-5,
        final_norm_eps=1e-5,
        num_pdd_heads=6,
    )
    return DiffusionModelConfig(
        pretrained_config=SimpleNamespace(**(architecture | overrides)),
        attention=AttentionConfig(backend="VANILLA"),
    )


@pytest.mark.parametrize("override", [{"num_pdd_heads": 5}, {"patch_size": (1, 1, 1)}])
def test_image_transformer_rejects_incompatible_checkpoint(override):
    with pytest.raises(ValueError, match="six-head.*1x2x2"):
        MiniMaxH3ImageTransformer2DModel(_image_config(**override))


@pytest.mark.parametrize("algo", [QuantAlgo.FP8, QuantAlgo.FP8_BLOCK_SCALES, QuantAlgo.NVFP4])
def test_image_transformer_dynamic_quantization(algo):
    config = _image_config(
        hidden_size=128,
        ffn_dim=256,
        time_embed_dim=128,
        num_attention_heads=1,
        attention_head_dim=128,
    ).model_copy(
        update=dict(
            quant_config=QuantConfig(
                quant_algo=algo,
                group_size=16 if algo == QuantAlgo.NVFP4 else 128,
                exclude_modules=["norm_out.linear"],
            ),
            dynamic_weight_quant=True,
            force_dynamic_quantization=algo == QuantAlgo.NVFP4,
        )
    )
    model = MiniMaxH3ImageTransformer2DModel(config)
    qkv = model.transformer_blocks[0].attn.qkv_proj
    assert qkv.quant_config.quant_algo == algo
    assert qkv.weight.dtype == (torch.uint8 if algo == QuantAlgo.NVFP4 else torch.float8_e4m3fn)
    assert not model.norm_out.linear.has_any_quant
    assert model.norm_out.linear.weight.dtype == config.torch_dtype
    assert model.proj_in.weight.dtype == torch.float32
    assert all(head.weight.dtype == torch.float32 for head in model.pdd_proj_out)
    assert model.time_embedder.linear_1.weight.dtype == torch.float32
    assert model.context_embedder.weight.dtype == config.torch_dtype


@pytest.mark.parametrize("algo", [QuantAlgo.FP8, QuantAlgo.FP8_BLOCK_SCALES, QuantAlgo.NVFP4])
def test_image_transformer_rejects_static_quantization(algo):
    config = _image_config().model_copy(update=dict(quant_config=QuantConfig(quant_algo=algo)))
    with pytest.raises(ValueError, match="dynamically quantized"):
        MiniMaxH3ImageTransformer2DModel(config)


def test_image_transformer_nvfp4_requires_dynamic_activation_scales():
    config = _image_config().model_copy(
        update=dict(
            quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4),
            dynamic_weight_quant=True,
        )
    )
    with pytest.raises(ValueError, match="force_dynamic_quantization=True"):
        MiniMaxH3ImageTransformer2DModel(config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TRTLLM RMSNorm requires CUDA")
@torch.inference_mode()
def test_image_transformer_packed_forward_selects_one_head():
    model = MiniMaxH3ImageTransformer2DModel(_image_config()).to("cuda").eval()
    rng = torch.Generator(device="cpu").manual_seed(23)
    # TRTLLM Linears are storage for checkpoint tensors and need explicit weights.
    for name, param in model.named_parameters():
        if "norm" in name and param.ndim == 1 and name.endswith("weight"):
            param.fill_(1)
        else:
            param.copy_(torch.randn(param.shape, generator=rng) * 0.05)
    # Keep the projection nonconstant across rows, but make the selected head
    # observable: identical W on every head, with b_i = i.
    for i, head in enumerate(model.pdd_proj_out):
        if i:
            head.weight.copy_(model.pdd_proj_out[0].weight)
        head.bias.fill_(i)
    model.post_load_weights()
    rows = torch.randn(1, 4, 8, generator=rng).cuda()
    text = torch.randn(1, 2, 8, generator=rng).cuda()
    coords = text_image_positions(2, 2, 2, rows.device)
    context = model.prepare_static_context(text, coords)
    assert context.text_embeds.shape == (1, 2, 32)
    assert context.sequence_length == 6
    sigma = rows.new_tensor(0.75)
    base = model(rows, sigma, context, 0)
    assert base.dtype == torch.float32 and base.shape == rows.shape
    assert base.isfinite().all() and not torch.equal(base[:, 0], base[:, 1])
    for i in range(1, 6):
        torch.testing.assert_close(model(rows, sigma, context, i), base + i)
    with pytest.raises(ValueError, match="PDD head"):
        model(rows, sigma, context, 6)
    with pytest.raises(ValueError, match="one noise level"):
        model(rows, sigma.expand(2), context, 0)
    with pytest.raises(ValueError, match="RoPE coordinates"):
        model(rows[:, :1], sigma, context, 0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_frames": 2}, "single image"),
        ({"num_images_per_prompt": 2}, "single image"),
        ({"num_inference_steps": 30}, "exactly six"),
        ({"guidance_scale": 7.5}, "CFG"),
        ({"negative_prompt": "blur"}, "CFG"),
        ({"image_reference": {"format": "bytes", "content": b"unused"}}, "text-to-image"),
    ],
)
def test_image_pipeline_rejects_unsupported_request(overrides, message):
    pipeline = MiniMaxH3ImagePipeline.__new__(MiniMaxH3ImagePipeline)
    params = VisualGenParams(**(pipeline.default_generation_params | overrides))
    with pytest.raises(ValueError, match=message):
        pipeline.infer(SimpleNamespace(prompt="a red cube", params=params))


@pytest.mark.parametrize(
    ("prompt", "message"),
    [
        ([], "exactly one prompt"),
        (["red", "blue"], "exactly one prompt"),
        ("", "nonempty prompt string"),
        ([""], "nonempty prompt string"),
        ([None], "nonempty prompt string"),
        ([7], "nonempty prompt string"),
        (None, "nonempty prompt string"),
    ],
)
def test_image_pipeline_rejects_invalid_prompt(prompt, message):
    pipeline = MiniMaxH3ImagePipeline.__new__(MiniMaxH3ImagePipeline)
    params = VisualGenParams(**pipeline.default_generation_params)
    with pytest.raises(ValueError, match=message):
        pipeline.infer(SimpleNamespace(prompt=prompt, params=params))


@pytest.mark.parametrize("prompt", ["a red cube", ["a red cube"]], ids=["scalar", "singleton"])
def test_image_pipeline_six_step_latent_update(prompt):
    # Exercise infer, including seed/patch ordering, all six velocity updates,
    # RGB truncation and output sizing. Learned components are test doubles.
    pipeline = MiniMaxH3ImagePipeline.__new__(MiniMaxH3ImagePipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline._device = torch.device("cpu")
    pipeline._is_warmup = True  # skip the request profiler in this unit test
    pipeline.scheduler = ImagePDDSchedule(pipeline.device)
    pipeline.tokenizer = Mock(return_value={"input_ids": [3, 5, 8]})
    pipeline.text_encoder = Mock(spec=["encode"])
    pipeline.text_encoder.encode.return_value = torch.zeros(1, 2, 8)
    pipeline.transformer = Mock(spec=["config", "prepare_static_context"])
    pipeline.transformer.config = SimpleNamespace(in_channels=2)
    pipeline.transformer.side_effect = lambda rows, **kw: torch.full_like(
        rows, kw["pdd_step_index"]
    )
    # Normalized zero maps to ImageNet RGB means, truncated (not rounded).
    pipeline.vae = Mock(return_value=torch.zeros(1, 3, 32, 64))
    params = VisualGenParams(
        height=31,
        width=61,
        num_inference_steps=6,
        max_sequence_length=2,
        extra_params={"min_generation_area": 1024},
        seed=7,
    )
    output = pipeline.infer(SimpleNamespace(prompt=prompt, params=params))
    pipeline.tokenizer.assert_called_once_with("a red cube", add_special_tokens=False)
    torch.testing.assert_close(pipeline.text_encoder.encode.call_args.args[0], torch.tensor([3, 5]))
    expected = torch.randn(1, 2, 2, 4, generator=torch.Generator().manual_seed(7))
    calls = pipeline.transformer.call_args_list
    assert len(calls) == 6
    for i, call in enumerate(calls):
        assert call.kwargs["pdd_step_index"] == i
        assert call.kwargs["timestep"] == pipeline.scheduler.sigmas[i]
        assert (
            call.kwargs["static_context"]
            is pipeline.transformer.prepare_static_context.return_value
        )
        rows = expected.reshape(1, 2, 1, 2, 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(1, 2, 8)
        torch.testing.assert_close(call.args[0], rows)
        expected = expected + i * (pipeline.scheduler.sigmas[i] - pipeline.scheduler.sigmas[i + 1])
    torch.testing.assert_close(pipeline.vae.call_args.args[0], expected)
    assert output.image.shape == (1, 31, 61, 3) and output.image.dtype == torch.uint8
    rgb = torch.tensor([123, 116, 103], dtype=torch.uint8)
    torch.testing.assert_close(output.image, rgb.expand_as(output.image))

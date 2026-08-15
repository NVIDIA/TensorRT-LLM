# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Registry and transformer smoke tests for Qwen-Image-Layered."""

import json
import sys
from types import SimpleNamespace

import pytest
import torch

# Importing the models package side-effects the ``@register_pipeline``
# decorator on ``QwenImageLayeredPipeline`` being applied.
from tensorrt_llm._torch.visual_gen import models  # noqa: F401
from tensorrt_llm._torch.visual_gen.config import (
    AttentionConfig,
    DiffusionModelConfig,
    DiffusionPipelineConfig,
)
from tensorrt_llm._torch.visual_gen.cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig
from tensorrt_llm._torch.visual_gen.models.qwen_image import QwenImagePipeline
from tensorrt_llm._torch.visual_gen.models.qwen_image_layered import QwenImageLayeredPipeline
from tensorrt_llm._torch.visual_gen.models.qwen_image_layered.transformer_qwen_image_layered import (
    QwenEmbedLayer3DRope,
    QwenImageLayeredTransformer2DModel,
)
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline
from tensorrt_llm._torch.visual_gen.pipeline_registry import (
    PIPELINE_REGISTRY,
    AutoPipeline,
    PipelineComponent,
)


def _pipeline_config(model_config=None):
    return DiffusionPipelineConfig(
        model_configs={"transformer": model_config or DiffusionModelConfig()}
    )


def test_qwen_image_layered_pipeline_is_registered():
    """@register_pipeline("QwenImageLayeredPipeline") must have been applied."""
    assert "QwenImageLayeredPipeline" in PIPELINE_REGISTRY
    entry = PIPELINE_REGISTRY["QwenImageLayeredPipeline"]
    assert entry.pipeline_cls is QwenImageLayeredPipeline
    assert entry.hf_ids == ["Qwen/Qwen-Image-Layered"]


def test_qwen_image_layered_pipeline_has_separate_boundary():
    """Layered is registered as its own pipeline variant, not a QwenImagePipeline subclass."""
    assert issubclass(QwenImageLayeredPipeline, BasePipeline)
    assert not issubclass(QwenImageLayeredPipeline, QwenImagePipeline)


def test_auto_pipeline_detects_qwen_image_layered_class_name(tmp_path):
    """Qwen-Image-Layered checkpoints must resolve to their layered pipeline."""
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "QwenImageLayeredPipeline"})
    )
    assert AutoPipeline._detect_from_checkpoint(str(tmp_path)) == "QwenImageLayeredPipeline"


def test_auto_pipeline_routes_qwen_image_layered_variants(tmp_path):
    """Unknown layered variants stay on the layered pipeline, not the base Qwen pipeline."""
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "QwenImageLayeredEditPipeline"})
    )
    assert AutoPipeline._detect_from_checkpoint(str(tmp_path)) == "QwenImageLayeredPipeline"


def test_qwen_image_layered_processor_loads_from_subfolder(monkeypatch, tmp_path):
    """Layered checkpoints store the VL processor in the processor/ subfolder."""
    calls = []

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, checkpoint_dir, **kwargs):
            calls.append((checkpoint_dir, kwargs))
            return cls()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(Qwen2VLProcessor=FakeProcessor),
    )
    pipeline = QwenImageLayeredPipeline(_pipeline_config())
    pipeline.load_standard_components(
        str(tmp_path),
        torch.device("cpu"),
        skip_components=[
            PipelineComponent.TOKENIZER,
            PipelineComponent.TEXT_ENCODER,
            PipelineComponent.VAE,
            PipelineComponent.SCHEDULER,
            PipelineComponent.IMAGE_PROCESSOR,
        ],
    )

    assert calls == [
        (str(tmp_path), {"subfolder": "processor"}),
    ]
    assert isinstance(pipeline.processor, FakeProcessor)


def test_qwen_image_layered_default_params_match_runtime_inputs():
    """Layered derives size from image+resolution unless height/width are explicitly set."""
    pipeline = QwenImageLayeredPipeline(_pipeline_config())
    assert pipeline.default_generation_params["height"] is None
    assert pipeline.default_generation_params["width"] is None
    assert pipeline.extra_param_specs["resolution"].range is None
    assert pipeline.default_warmup_num_frames == [1]
    assert pipeline.warmup_cache_key(None, None, num_frames=1) == (640, 640)
    assert pipeline.warmup_cache_key(512, 768, num_frames=1) == (512, 768)


def test_qwen_image_layered_rejects_multi_frame_latent_input():
    """Layered latent input accepts one conditioning frame before packing."""
    pipeline = QwenImageLayeredPipeline(_pipeline_config())
    image = torch.zeros(1, pipeline.latent_channels, 2, 4, 4)

    assert pipeline._is_layered_latent_image(image)
    with pytest.raises(ValueError, match="exactly one conditioning frame"):
        pipeline._validate_single_conditioning_frame(image)


def test_qwen_image_layered_repeats_conditioning_in_prompt_order():
    """Conditioning images follow prompt expansion order for num_images_per_prompt > 1."""
    image_latents = torch.arange(2).view(2, 1, 1, 1, 1)

    repeated = QwenImageLayeredPipeline._repeat_conditioning_batch(image_latents, 4, "image")

    assert repeated[:, 0, 0, 0, 0].tolist() == [0, 0, 1, 1]


def test_qwen_image_layered_aligns_captions_to_expanded_prompts():
    """Per-image captions are repeated positionally for expanded prompt batches."""
    prompts = ["", "", "", ""]
    captions = ["caption 0", "caption 1"]

    aligned_prompts = QwenImageLayeredPipeline._align_prompts_to_image_batch(prompts, 2)
    aligned_captions = QwenImageLayeredPipeline._expand_values_to_batch(
        captions,
        len(aligned_prompts),
        "caption",
    )

    assert aligned_prompts == prompts
    assert aligned_captions == ["caption 0", "caption 0", "caption 1", "caption 1"]


def test_qwen_image_layered_layer_stack_to_image_grid():
    """Layer stacks are converted to a saveable image grid instead of video output."""
    layer_stack = torch.arange(12, dtype=torch.uint8).view(1, 3, 2, 2, 1)

    grid = QwenImageLayeredPipeline._layer_stack_to_image_grid(layer_stack)

    assert grid.shape == (1, 4, 4, 1)
    assert grid[0, :, :, 0].tolist() == [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 0, 0],
        [10, 11, 0, 0],
    ]


def test_transformer_constructs_with_layered_config():
    """Layered config fields select layer-aware RoPE and additional time conditioning."""
    model = QwenImageLayeredTransformer2DModel(
        model_config=None,
        num_layers=2,
        use_additional_t_cond=True,
        use_layer3d_rope=True,
    )
    assert isinstance(model.pos_embed, QwenEmbedLayer3DRope)
    assert model.time_text_embed.use_additional_t_cond is True
    assert hasattr(model.time_text_embed, "addition_t_embedding")


def test_transformer_from_config_dict_preserves_layered_fields():
    """from_config_dict should not drop Qwen-Image-Layered transformer fields."""
    model = QwenImageLayeredTransformer2DModel.from_config_dict(
        {
            "num_layers": 2,
            "use_additional_t_cond": True,
            "use_layer3d_rope": True,
        }
    )
    assert isinstance(model.pos_embed, QwenEmbedLayer3DRope)
    assert model.time_text_embed.use_additional_t_cond is True


def test_layered_transformer_cuda_graph_key_includes_img_shapes():
    """Layered RoPE shape metadata must be part of the CUDA graph key."""
    runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
    model = QwenImageLayeredTransformer2DModel(
        model_config=None,
        patch_size=1,
        in_channels=4,
        out_channels=4,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=12,
        axes_dims_rope=(2, 2, 4),
        use_layer3d_rope=True,
    )
    model.register_cuda_graph_extra_key_fns(runner)

    tensor_kwargs = {
        "hidden_states": torch.empty(1, 12, 4),
        "encoder_hidden_states": torch.empty(1, 5, 12),
        "timestep": torch.empty(1),
    }
    key_a = runner.get_graph_key(
        **tensor_kwargs,
        img_shapes=[[(1, 2, 3), (1, 2, 3)]],
    )
    key_b = runner.get_graph_key(
        **tensor_kwargs,
        img_shapes=[[(1, 3, 2), (1, 3, 2)]],
    )

    assert ("img_shapes", (((1, 2, 3), (1, 2, 3)),)) in key_a
    assert ("img_shapes", (((1, 3, 2), (1, 3, 2)),)) in key_b
    assert key_a != key_b


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("with_text_mask", [False, True])
def test_layered_transformer_forward_sanity(with_text_mask):
    """A tiny Qwen-Image-Layered transformer runs layer-aware RoPE and t-cond paths."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    model_config = DiffusionModelConfig(
        attention=AttentionConfig(backend="VANILLA"),
        skip_create_weights_in_init=False,
    )
    model = (
        QwenImageLayeredTransformer2DModel(
            model_config=model_config,
            patch_size=1,
            in_channels=4,
            out_channels=4,
            num_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=12,
            axes_dims_rope=(2, 2, 4),
            use_additional_t_cond=True,
            use_layer3d_rope=True,
        )
        .to(device, dtype=dtype)
        .eval()
    )

    hidden_states = torch.randn(1, 12, 4, device=device, dtype=dtype)
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
            additional_t_cond=torch.zeros(1, device=device, dtype=torch.long),
            img_shapes=[[(1, 2, 2), (1, 2, 2), (1, 2, 2)]],
        )

    assert isinstance(output, tuple)
    assert output[0].shape == (1, 12, 4)

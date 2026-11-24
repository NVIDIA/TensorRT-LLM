# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1's image input processor: processor geometry, encoder items, the image-only refusal.

The processor tests stub the HF image processor's ``media_proc_cfg``, so all but the last two tests run without a
checkpoint. The last two run the real processor through the LLM API's hashing wrapper when ``LLM_MODELS_ROOT`` holds the
Kimi-K3 checkpoint (K3.1's image input is K3's).
"""

import math
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from utils.llm_data import llm_models_root

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.input_processor import (
    K31InputProcessor,
    _navit_max_patches,
    moonvit_max_image_grid,
)
from tensorrt_llm._torch.models import modeling_kimi_k3_vl
from tensorrt_llm.inputs.multimodal import MULTIMODAL_ENCODER_ITEM_METADATA_KEY

pytestmark = pytest.mark.cpu_only

_PATCH = 14


def _media_proc_cfg(in_patch_limit=65536):
    return {
        "in_patch_limit": in_patch_limit,
        "patch_size": _PATCH,
        "merge_kernel_size": 2,
        "patch_limit_on_one_side": 512,
    }


def _processor(*, in_patch_limit=65536):
    """An input processor around a stub HF image processor."""
    processor = K31InputProcessor.__new__(K31InputProcessor)
    processor._config = SimpleNamespace(
        vision_config=SimpleNamespace(patch_size=_PATCH, merge_kernel_size=(2, 2))
    )
    processor._processor = SimpleNamespace(
        image_processor=SimpleNamespace(media_proc_cfg=_media_proc_cfg(in_patch_limit))
    )
    processor._dtype = torch.bfloat16
    return processor


def _navit_patches(width, height, *, in_patch_limit, side=512, merge=2):
    """Patches of the grid the HF ``navit_resize_image`` gives an image."""
    s1 = math.sqrt(in_patch_limit / (max(1.0, width // _PATCH) * max(1.0, height // _PATCH)))
    scale = min(1.0, s1, side * _PATCH / width, side * _PATCH / height)
    new_w = min(max(1, int(width * scale)), side * _PATCH)
    new_h = min(max(1, int(height * scale)), side * _PATCH)
    factor = merge * _PATCH
    return merge * math.ceil(new_h / factor) * merge * math.ceil(new_w / factor)


def _payload(*grids, pixel_rows=None):
    """Packed MoonViT media whose pixel rows count up."""
    if pixel_rows is None:
        pixel_rows = sum(t * h * w for t, h, w in grids)
    return {
        "pixel_values": torch.arange(pixel_rows, dtype=torch.float32)
        .view(-1, 1, 1, 1)
        .expand(-1, 3, _PATCH, _PATCH),
        "image_grid_thw": torch.tensor(grids, dtype=torch.int64),
    }


# ---------------------------------------------------------------------------
# Processor geometry
# ---------------------------------------------------------------------------


def test_padding_takes_real_images_past_the_patch_limit():
    assert _navit_patches(3600, 3600, in_patch_limit=65536) == 66564
    assert _navit_patches(1889, 7377, in_patch_limit=65536) == 67320
    assert _navit_max_patches(_PATCH, 2, 65536, 512) == 67320


def test_max_image_bounds_resized_images():
    bound = _navit_max_patches(_PATCH, 2, 65536, 512)
    sizes = list(range(1, 400, 7)) + list(range(400, 30000, 97))

    assert max(_navit_patches(w, h, in_patch_limit=65536) for w in sizes for h in sizes) <= bound


@pytest.mark.parametrize("in_patch_limit, grid", [(65536, (1, 220, 306)), (16384, (1, 124, 144))])
def test_largest_image_grid_holds_the_most_patches_the_processor_produces(in_patch_limit, grid):
    """The vision tower's attention capacity and KV-capacity profiling image."""
    t, h, w = moonvit_max_image_grid(_media_proc_cfg(in_patch_limit), (2, 2), _PATCH)

    assert (t, h, w) == grid
    assert t * h * w == _navit_max_patches(_PATCH, 2, in_patch_limit, 512)
    assert h % 2 == w % 2 == 0 and max(h, w) <= 512


def test_largest_image_grid_defaults_without_a_processor_config():
    assert moonvit_max_image_grid({}, (2, 2), _PATCH) == moonvit_max_image_grid(
        _media_proc_cfg(16384), (2, 2), _PATCH
    )


# ---------------------------------------------------------------------------
# Item metadata
# ---------------------------------------------------------------------------


def test_item_metadata_for_one_and_two_images():
    processor = _processor()

    one = processor.get_mm_item_metadata([], {"image": _payload((1, 400, 128))})
    assert one.item_refs == (("image", 0),)
    assert one.encoder_token_lengths == (51200,)
    assert one.output_embedding_lengths == (12800,)

    two = processor.get_mm_item_metadata([], {"image": _payload((1, 400, 128), (1, 72, 102))})
    two.validate()
    assert two.item_refs == (("image", 0), ("image", 1))
    assert two.encoder_token_lengths == (51200, 7344)
    assert two.output_embedding_lengths == (12800, 1836)


def test_item_metadata_follows_the_prompt_item_order():
    processor = _processor()
    mm_data = {"image": _payload((1, 4, 6), (1, 8, 4))}
    order = [{"modality": "image", "index": 1}, {"modality": "image", "index": 0}]

    metadata = processor.get_mm_item_metadata([], mm_data, order)
    assert metadata.item_refs == (("image", 1), ("image", 0))
    assert metadata.encoder_token_lengths == (32, 24)
    assert metadata.output_embedding_lengths == (8, 6)
    with pytest.raises(ValueError, match="do not match the 2 processed Kimi images"):
        processor.get_mm_item_metadata([], mm_data, order[:1])
    with pytest.raises(ValueError, match="do not match"):
        processor.get_mm_item_metadata([], mm_data, [{"modality": "video", "index": 0}] + order[:1])


def test_item_metadata_leaves_attached_embeddings_alone():
    processor = _processor()

    assert processor.get_mm_item_metadata([], {"multimodal_embedding": torch.zeros(4, 8)}) is None


@pytest.mark.parametrize("grid", [(1, 514, 2), (1, 6, 5), (2, 4, 4), (1, 512, 512)])
def test_item_metadata_rejects_grids_outside_the_processor_geometry(grid):
    with pytest.raises(ValueError, match="outside the processor geometry"):
        _processor().get_mm_item_metadata([], {"image": _payload(grid)})


def test_item_metadata_rejects_pixels_that_do_not_match_the_grids():
    with pytest.raises(ValueError, match="pixel_values holds 23 patches"):
        _processor().get_mm_item_metadata([], {"image": _payload((1, 4, 6), pixel_rows=23)})


# ---------------------------------------------------------------------------
# Prompt processing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modality", ["video", "audio"])
def test_prompts_with_other_modalities_are_refused(modality):
    with pytest.raises(ValueError, match="image inputs only"):
        _processor().call_with_text_prompt(
            {
                "prompt": "Describe it.",
                "multi_modal_data": {modality: [Image.new("RGB", (28, 28))]},
            },
            None,
        )


class _PilOnlyImageProcessor:
    """Count tokens like the HF processor, which rejects anything but PIL images."""

    def media_tokens_calculator(self, media):
        image = media["image"]
        if not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported data type: {type(image)}")
        width, height = image.size
        return (height // 28) * (width // 28)


def test_tensor_images_count_like_pil_images():
    # The serving media loader yields CHW float tensors.
    processor = _processor()
    processor._processor = SimpleNamespace(image_processor=_PilOnlyImageProcessor())

    assert processor.get_num_tokens_per_image(image=torch.rand(3, 56, 84)) == 6


# ---------------------------------------------------------------------------
# The real processor through the LLM API's hashing wrapper
# ---------------------------------------------------------------------------

_MODELS_ROOT = llm_models_root()
_KIMI_K3_DIR = None if _MODELS_ROOT is None else _MODELS_ROOT / "Kimi-K3"
_NEEDS_K3 = pytest.mark.skipif(
    _KIMI_K3_DIR is None or not _KIMI_K3_DIR.is_dir(), reason="needs the Kimi-K3 checkpoint"
)


def _process(processor_cls, prompt, multi_modal_data):
    """Run ``processor_cls`` on the checkpoint the way the LLM API does, with multimodal hashing."""
    from transformers import AutoTokenizer

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.inputs import create_input_processor_with_hash
    from tensorrt_llm.sampling_params import SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(str(_KIMI_K3_DIR), trust_remote_code=True)
    config = ModelConfig.from_pretrained(
        str(_KIMI_K3_DIR), trust_remote_code=True
    ).pretrained_config
    processor = processor_cls(str(_KIMI_K3_DIR), config, tokenizer, trust_remote_code=True)
    token_ids, extra = create_input_processor_with_hash(processor)(
        {"prompt": prompt, "multi_modal_data": multi_modal_data}, SamplingParams()
    )
    # Hashing falls back silently on an error, so require its output.
    assert extra.get("multimodal_input") is not None
    return processor, token_ids, extra


@_NEEDS_K3
@pytest.mark.parametrize("image_format", ["pil", "pt"])
def test_processor_items_match_the_prompt_and_the_largest_image(image_format):
    images = [Image.new("RGB", (224, 448)), Image.new("RGB", (3600, 3600))]
    if image_format == "pt":
        # trtllm-serve's default media format: CHW float tensors in [0, 1].
        images = [pil_to_tensor(image).float() / 255 for image in images]
    processor, token_ids, extra = _process(
        K31InputProcessor,
        "<|kimi_image_placeholder|><|kimi_image_placeholder|>Describe both.",
        {"image": images},
    )
    mm_data = extra["multimodal_data"]
    metadata = mm_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY]
    grids = mm_data["image"]["image_grid_thw"].tolist()

    assert sum(metadata.output_embedding_lengths) == token_ids.count(
        int(processor.get_mm_token_ids()[0])
    )
    assert metadata.item_refs == (("image", 0), ("image", 1))
    assert metadata.encoder_token_lengths == tuple(t * h * w for t, h, w in grids)
    assert metadata.output_embedding_lengths == tuple(
        processor.get_num_tokens_per_image(image=image) for image in images
    )
    # Padding takes the large image past in_patch_limit, but not past the largest image grid.
    t, h, w = moonvit_max_image_grid(
        processor.processor.image_processor.media_proc_cfg, (2, 2), _PATCH
    )
    assert 65536 < metadata.encoder_token_lengths[1] <= t * h * w


@_NEEDS_K3
def test_processor_matches_the_engine_k3_processor_on_a_fixed_image():
    """Token ids, pixels, grids and multimodal positions of one fixed image: the family's processor gives the
    engine K3 processor's."""
    image = Image.new("RGB", (448, 336))
    image.putdata(
        [
            ((x * 7) % 256, (y * 5) % 256, ((x + y) * 3) % 256)
            for y in range(336)
            for x in range(448)
        ]
    )
    prompt = "<|kimi_image_placeholder|>Describe the image."
    _, ours, our_extra = _process(K31InputProcessor, prompt, {"image": [image]})
    _, engine, engine_extra = _process(
        modeling_kimi_k3_vl.KimiK3InputProcessor, prompt, {"image": [image]}
    )

    assert ours == engine
    assert ours.count(163605) > 0
    for key in ("pixel_values", "image_grid_thw"):
        assert torch.equal(
            our_extra["multimodal_data"]["image"][key],
            engine_extra["multimodal_data"]["image"][key],
        )
    our_input, engine_input = our_extra["multimodal_input"], engine_extra["multimodal_input"]
    assert our_input.multimodal_positions == engine_input.multimodal_positions
    assert our_input.multimodal_lengths == engine_input.multimodal_lengths

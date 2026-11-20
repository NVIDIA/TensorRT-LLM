# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""uint8 images, the default `"np"` image format, through the model input processors.

HF image processors handle a uint8 array exactly as they handle the same PIL image. Against the
float `"pt"` image, their results differ only where the resize rounds to the uint8 grid and clamps
its overshoot. Processors that need PIL images or float tensors turn uint8 arrays into the same
values the other formats give. The Qwen-VL cases are in test_modeling_qwen3vl.py.
"""

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from transformers import (
    BatchFeature,
    Gemma3ImageProcessor,
    Llama4ImageProcessor,
    LlavaNextImageProcessor,
    PixtralImageProcessor,
)
from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor

from tensorrt_llm._torch.models import (
    modeling_kimi_k25,
    modeling_nemotron_h_multimodal,
    modeling_phi4mm,
    modeling_vila,
)
from tensorrt_llm._torch.models.modeling_gemma3vl import Gemma3InputProcessor
from tensorrt_llm._torch.models.modeling_gemma4_unified import Gemma4UnifiedImageProcessor
from tensorrt_llm._torch.models.modeling_gemma4mm import Gemma4InputProcessor
from tensorrt_llm._torch.models.modeling_llama import Llama4InputProcessor
from tensorrt_llm._torch.models.modeling_llava_next import LlavaNextInputProcessor
from tensorrt_llm._torch.models.modeling_mistral import MistralHFInputProcessor

pytestmark = pytest.mark.cpu_only

# (height, width)
SIZES = {
    "square": (512, 512),
    "1920x1080": (1080, 1920),
    "portrait": (1280, 720),
    "small": (48, 64),
}


def _test_image(height: int, width: int) -> np.ndarray:
    """Smooth gradients with a sharp-edged block, so the resize both rounds and overshoots."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    image = np.stack([x / width, y / height, (x + y) / (width + height)], axis=-1) * 255
    image[height // 4 : height // 2, width // 4 : width // 2] = (250, 10, 128)
    return np.ascontiguousarray(image.astype(np.uint8))


def _to_float(image: np.ndarray) -> torch.Tensor:
    """What the `"pt"` image loader gives for the same pixels."""
    return torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255)


def _all_formats(size: str):
    image = _test_image(*SIZES[size])
    return image, Image.fromarray(image), _to_float(image)


def _assert_uint8_matches(from_uint8, from_pil, from_float, std: float) -> None:
    # HF processors treat a uint8 array exactly like the PIL image.
    torch.testing.assert_close(from_uint8, from_pil, rtol=0, atol=0)
    # Against the float image, only the uint8 rounding and the clamped resize overshoot differ.
    steps = (from_uint8.float() - from_float.float()).abs() * 255 * std
    assert steps.mean() < 0.5
    assert (steps <= 1.01).float().mean() > 0.98


class _Processor:
    """An HF processor with real image (and video) processors and a one-token text output."""

    def __init__(self, image_processor, video_processor=None):
        self.image_processor = image_processor
        self.video_processor = video_processor

    def __call__(
        self,
        text=None,
        images=None,
        videos=None,
        do_rescale=None,
        images_kwargs=None,
        videos_kwargs=None,
        return_tensors="pt",
        **kwargs,
    ):
        output = BatchFeature(
            {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}
        )
        for processor, key, items, modality_kwargs in (
            (self.image_processor, "images", images, images_kwargs),
            (self.video_processor, "videos", videos, videos_kwargs),
        ):
            if items is None:
                continue
            modality_kwargs = dict(modality_kwargs or {})
            if do_rescale is not None:
                modality_kwargs["do_rescale"] = do_rescale
            output.update(processor(**{key: items}, return_tensors="pt", **modality_kwargs))
        return output


def _new(processor_cls, image_processor, video_processor=None):
    processor = processor_cls.__new__(processor_cls)
    processor._processor = _Processor(image_processor, video_processor)
    processor._dtype = torch.float32
    return processor


def _gemma3_pixels(image):
    processor = _new(Gemma3InputProcessor, Gemma3ImageProcessor())
    return processor._preprocess({"prompt": "<image>", "multi_modal_data": {"image": [image]}})[1]


def _gemma4_pixels(image, image_processor_cls=Gemma4ImageProcessor):
    processor = _new(Gemma4InputProcessor, image_processor_cls())
    processor._image_processor = processor._processor.image_processor
    processor._vision_gpu_normalize = False
    return processor._preprocess({"prompt": "<image>", "multi_modal_data": {"image": [image]}})[1]


def _image_data(processor, image, sampling_params=None):
    inputs = {"prompt": "<image>", "multi_modal_data": {"image": [image]}}
    _, extra = processor.call_with_text_prompt(inputs, sampling_params)
    return extra["multimodal_data"]["image"]["pixel_values"]


# The processors' default configs; `std` is the smallest normalization std.
_HF_MODELS = {
    "gemma3": (_gemma3_pixels, min(Gemma3ImageProcessor().image_std)),
    "gemma4": (_gemma4_pixels, 1.0),
    "gemma4_unified": (
        lambda image: _gemma4_pixels(image, Gemma4UnifiedImageProcessor),
        1.0,
    ),
    "llava_next": (
        lambda image: _image_data(_new(LlavaNextInputProcessor, LlavaNextImageProcessor()), image),
        min(LlavaNextImageProcessor().image_std),
    ),
    "llama4": (
        lambda image: _image_data(
            _new(Llama4InputProcessor, Llama4ImageProcessor()),
            image,
            SimpleNamespace(truncate_prompt_tokens=None, add_special_tokens=True),
        ),
        min(Llama4ImageProcessor().image_std),
    ),
    "mistral3": (
        lambda image: _image_data(_new(MistralHFInputProcessor, PixtralImageProcessor()), image),
        min(PixtralImageProcessor().image_std),
    ),
}


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("model", _HF_MODELS)
def test_hf_processor_models_take_uint8_images(model: str, size: str) -> None:
    pixels, std = _HF_MODELS[model]
    from_uint8, from_pil, from_float = (pixels(image) for image in _all_formats(size))
    _assert_uint8_matches(from_uint8, from_pil, from_float, std)


@pytest.mark.parametrize("size", SIZES)
def test_kimi_k25_hands_uint8_images_to_its_processor_as_pil(size: str) -> None:
    """Kimi K2.5 / K3 give their processor PIL images; uint8 arrays become the same PIL image."""

    class _Stop(Exception):
        pass

    captured = []

    def processor(*, medias: list, **kwargs):
        captured.extend(medias)
        raise _Stop

    image, _, float_image = _all_formats(size)
    processor_cls = modeling_kimi_k25.KimiK25InputProcessor
    kimi = processor_cls.__new__(processor_cls)
    kimi._processor = processor
    kimi._tokenizer = None
    pil_images = []
    for item in (image, float_image):
        captured.clear()
        with pytest.raises(_Stop):
            kimi.call_with_text_prompt(
                {"prompt": "<image>", "multi_modal_data": {"image": [item]}}, None
            )
        [media] = captured
        assert isinstance(media["image"], Image.Image)
        pil_images.append(np.asarray(media["image"]))
    np.testing.assert_array_equal(pil_images[0], image)
    np.testing.assert_array_equal(pil_images[1], np.asarray(to_pil_image(float_image)))

    # The token count reads the same PIL image.
    kimi._processor = SimpleNamespace(
        image_processor=SimpleNamespace(
            media_tokens_calculator=lambda media: media["image"].size[0] * media["image"].size[1]
        )
    )
    height, width = SIZES[size]
    assert kimi.get_num_tokens_per_image(image=image) == height * width


@pytest.mark.parametrize("size", SIZES)
def test_nemotron_h_takes_uint8_images(size: str) -> None:
    """The dynamic tiler and the HF path get the same values from uint8 and float images."""
    nemotron = modeling_nemotron_h_multimodal
    image, _, float_image = _all_formats(size)
    height, width = SIZES[size]
    assert nemotron._get_media_dimensions(image) == (width, height)
    assert nemotron._get_media_dimensions(float_image) == (width, height)
    raw = nemotron._media_to_raw_chw(image)
    assert raw.dtype == torch.float32 and raw.is_contiguous()
    torch.testing.assert_close(raw, nemotron._media_to_raw_chw(float_image), rtol=0, atol=0)

    seen = []
    processor_cls = nemotron.NemotronHMultimodalInputProcessor
    processor = processor_cls.__new__(processor_cls)
    processor._processor = lambda images, **kwargs: seen.extend(images) or BatchFeature(
        {"num_patches": []}
    )
    processor.device = "cpu"
    processor.img_context_token = "<image>"
    processor._process_images([image], "")
    processor._process_images([float_image], "")
    torch.testing.assert_close(seen[0], seen[1], rtol=0, atol=0)


@pytest.mark.parametrize("size", SIZES)
def test_vila_takes_uint8_images(monkeypatch: pytest.MonkeyPatch, size: str) -> None:
    seen = []
    monkeypatch.setattr(
        modeling_vila, "process_images", lambda images, *args, **kwargs: seen.append(images)
    )
    processor = modeling_vila.VilaInputProcessor.__new__(modeling_vila.VilaInputProcessor)
    processor.image_processor = None
    processor._config = None
    processor.vision_tower = SimpleNamespace(dtype=torch.float32)
    image, _, float_image = _all_formats(size)
    for item in (image, float_image):
        processor._preprocess({"image": [item]}, {})
    [uint8_input], [float_input] = seen
    torch.testing.assert_close(uint8_input, float_input, rtol=0, atol=0)


@pytest.mark.parametrize("size", SIZES)
def test_phi4mm_takes_uint8_images(size: str) -> None:
    """Phi-4-MM's local preprocessing gives the same inputs and token count for uint8 images."""
    image_processor = SimpleNamespace(
        dynamic_hd=36,
        pad_to_max_num_crops=lambda images, max_crops: images,
        pad_mask_to_max_num_crops=lambda masks, max_crops: masks,
    )
    image_processor.dynamic_preprocess = MethodType(
        modeling_phi4mm.dynamic_preprocess, image_processor
    )
    image, _, float_image = _all_formats(size)
    outputs = [
        modeling_phi4mm.image_preprocess(image_processor, [item], return_tensors="pt")
        for item in (image, float_image)
    ]
    for key in ("input_image_embeds", "image_sizes", "image_attention_mask", "num_img_tokens"):
        torch.testing.assert_close(outputs[0][key], outputs[1][key], rtol=0, atol=0)

    processor = modeling_phi4mm.Phi4MMInputProcessor.__new__(modeling_phi4mm.Phi4MMInputProcessor)
    processor._processor = SimpleNamespace(image_processor=image_processor)
    assert processor.get_num_tokens_per_image(image=image) == processor.get_num_tokens_per_image(
        image=float_image
    )

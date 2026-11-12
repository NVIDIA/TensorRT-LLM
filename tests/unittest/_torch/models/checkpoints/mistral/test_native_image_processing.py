# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Images through the Mistral-native input processor.

``checkpoint_format: mistral`` builds ``MistralNativeInputProcessor``, whose
``MistralCommonImageProcessor`` encodes each image for mistral-common. The media loaders
deliver a PIL image (``"pil"``), a float ``(C, H, W)`` tensor in [0, 1] (``"pt"``) or a
uint8 ``(H, W, C)`` array (``"np"``); each is converted losslessly. The server renders a
chat to token ids with the image grids in place; the processor checks those grids
against the images, or expands one ``[IMG]`` per image to its grid, without re-encoding
the text.

The conversion tests need nothing else. The processor tests read Mistral-Small-4's
native files (tekken.json, params.json) under LLM_MODELS_ROOT and need OpenCV, which
mistral-common's image encoder resizes with.
"""

from __future__ import annotations

import asyncio
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.models.modeling_mistral import MistralCommonImageProcessor

pytestmark = pytest.mark.cpu_only

_to_pil = MistralCommonImageProcessor.to_pil_image
_SMALL4 = "Mistral-Small-4-119B-2603/Mistral-Small-4-119B-2603"
_INPUT_FILES = Path(__file__).resolve().parents[5] / "integration" / "test_input_files"


def _rgb(seed: int = 0, size=(37, 23)) -> Image.Image:
    gen = np.random.default_rng(seed)
    return Image.fromarray(gen.integers(0, 256, size=(*size, 3), dtype=np.uint8), "RGB")


def _every_level() -> Image.Image:
    """Each of the 256 8-bit levels in every channel, so every level must come back."""
    levels = np.arange(256, dtype=np.uint8).reshape(16, 16)
    return Image.fromarray(np.stack([levels, levels.T, 255 - levels], -1), "RGB")


def _pt(image: Image.Image) -> torch.Tensor:
    """The ``"pt"`` format is ``ToTensor``: uint8 / 255 in fp32."""
    tensor = ToTensor()(image)
    assert tensor.dtype == torch.float32 and tensor.shape == (3, image.height, image.width)
    return tensor


def _uint8_chw(image: Image.Image, channels: int = 3) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image)[..., :channels]).permute(2, 0, 1).contiguous()


def _image_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


@pytest.mark.parametrize(
    "image,form,mode",
    [
        (_rgb(), lambda image: image, "RGB"),
        (_every_level(), _pt, "RGB"),
        (_rgb(1), _pt, "RGB"),
        (_rgb(2), np.asarray, "RGB"),
        (_rgb(2), _uint8_chw, "RGB"),
        (_rgb(2), lambda image: _uint8_chw(image, channels=1), "L"),
    ],
    ids=["pil", "pt_every_level", "pt", "uint8_hwc_array", "uint8_chw_tensor", "uint8_gray"],
)
def test_to_pil_image_takes_every_loader_form(image, form, mode):
    """A PIL image passes through; a ``"pt"`` tensor comes back at its exact 8-bit levels;
    a uint8 ``(H, W, C)`` array or ``(C, H, W)`` tensor comes back as it is, with one
    channel as mode "L"."""
    given = form(image)
    back = _to_pil(given)
    if isinstance(given, Image.Image):
        assert back is given
    assert back.mode == mode
    if mode == "RGB":
        assert back.size == image.size
        assert np.array_equal(np.asarray(back), np.asarray(image))


def test_an_unknown_type_is_refused():
    with pytest.raises(TypeError, match="unsupported image type"):
        _to_pil([1, 2, 3])


def test_the_server_places_an_image_in_its_message():
    """trtllm-serve's chat parsing for the Mistral-native model type keeps an image's place."""
    from tensorrt_llm.inputs.content_format import ContentFormat
    from tensorrt_llm.inputs.registry import MULTIMODAL_PLACEHOLDER_REGISTRY
    from tensorrt_llm.serve.chat_utils import parse_chat_messages_coroutines

    assert (
        MULTIMODAL_PLACEHOLDER_REGISTRY.get_content_format("mistral_common") == ContentFormat.OPENAI
    )
    prompt = "Describe this image in one sentence."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _image_url(_rgb(5))}},
            ],
        }
    ]
    config = type("Config", (), {"model_type": "mistral_common"})()
    conversation, media, counts, _ = parse_chat_messages_coroutines(messages, config)
    mm_data, _ = asyncio.run(media)
    assert counts == [{"[IMG]": 1}]
    assert conversation[0]["content"] == prompt
    assert len(conversation[0]["content_parts"]) == 2
    assert len(mm_data["image"]) == 1


@pytest.fixture(scope="module")
def native_processor():
    pytest.importorskip("cv2")
    from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import MistralConfigLoader
    from tensorrt_llm._torch.models.modeling_mistral import MistralNativeInputProcessor

    root = llm_models_root()
    path = None if root is None else root / _SMALL4
    if path is None or not (path / "tekken.json").is_file():
        pytest.skip(f"needs {_SMALL4} under LLM_MODELS_ROOT")
    config = MistralConfigLoader().load(str(path)).pretrained_config
    processor = MistralNativeInputProcessor(str(path), config, None)
    assert processor._has_vision_encoder()
    return processor


def test_every_loader_format_encodes_an_image_alike(native_processor):
    from tensorrt_llm.inputs.utils import load_image

    processor = native_processor.processor
    path = str(_INPUT_FILES / "merlion.png")
    pil = load_image(path, format="pil")
    images = {"pil": pil, "pt": load_image(path, format="pt"), "np": np.asarray(pil)}
    results = {fmt: processor(text="Describe.", images=[image]) for fmt, image in images.items()}
    for fmt in ("pt", "np"):
        assert torch.equal(results[fmt]["input_ids"], results["pil"]["input_ids"]), fmt
        assert torch.equal(results[fmt]["pixel_values"], results["pil"]["pixel_values"]), fmt
    flat = results["pil"]["input_ids"][0]
    assert int(flat[0]) == native_processor.tokenizer.bos_token_id
    assert int((flat == processor.image_end_token_id).sum()) == 1


@pytest.mark.parametrize("name", ["merlion.png", "excel_table_test.jpg"])
def test_pixels_are_what_mistral_common_makes_of_the_loaded_image(native_processor, name):
    """Nothing lossy lies between the loaded image and mistral-common's resize and normalize."""
    from mistral_common.protocol.instruct.chunk import ImageChunk

    processor = MistralCommonImageProcessor(native_processor.tokenizer, torch.float32)
    image = Image.open(_INPUT_FILES / name).convert("RGB")
    expected = torch.as_tensor(processor.image_processor(ImageChunk(image=image)).image)
    pixels = processor("", images=[image])["pixel_values"][0]
    assert torch.equal(pixels, expected.to(pixels.dtype))


def test_two_different_image_sizes_preserve_each_grid(native_processor):
    processor = native_processor.processor
    images = [_rgb(12, (224, 336)), _rgb(13, (448, 280))]
    singles = [processor("Describe.", images=[image]) for image in images]
    together = processor("Describe.", images=images)
    assert together["image_sizes"].tolist() == [one["image_sizes"][0].tolist() for one in singles]
    for i, one in enumerate(singles):
        h, w = one["image_sizes"][0].tolist()
        assert torch.equal(together["pixel_values"][i, :, :h, :w], one["pixel_values"][0])
    expected = sum(int((one["input_ids"] == processor.image_token_id).sum()) for one in singles)
    assert int((together["input_ids"] == processor.image_token_id).sum()) == expected


@pytest.mark.parametrize("multi_turn", [False, True])
def test_a_served_chat_keeps_roles_image_order_and_control_tokens(native_processor, multi_turn):
    from tensorrt_llm.inputs.utils import async_apply_chat_template_with_mm
    from tensorrt_llm.serve.chat_utils import parse_chat_messages_coroutines

    images = [_rgb(21, (56, 84)), _rgb(22, (112, 56))]
    messages = [
        {"role": "system", "content": "Describe images in order."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "First "},
                {"type": "image_url", "image_url": {"url": _image_url(images[0])}},
                {"type": "text", "text": " then "},
                {"type": "image_url", "image_url": {"url": _image_url(images[1])}},
                {"type": "text", "text": "; compare them."},
            ],
        },
    ]
    if multi_turn:
        messages += [
            {"role": "assistant", "content": "Two images."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Now compare the first to "},
                    {"type": "image_url", "image_url": {"url": _image_url(images[0])}},
                    {"type": "text", "text": " again."},
                ],
            },
        ]
    expected = native_processor.tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False, add_generation_prompt=True
    )
    conversation, media, counts, _ = parse_chat_messages_coroutines(
        messages, native_processor.config
    )
    ids, (mm_data, _) = asyncio.run(
        async_apply_chat_template_with_mm(
            media,
            model_type="mistral_common",
            tokenizer=native_processor.tokenizer,
            processor=None,
            conversation=conversation,
            add_generation_prompt=True,
            mm_placeholder_counts=counts,
        )
    )
    assert ids == expected
    actual_ids, extra = native_processor(
        {"prompt_token_ids": ids, "multi_modal_data": mm_data}, None
    )
    assert actual_ids == expected
    wanted_images = images + images[:1] if multi_turn else images
    payload = extra["multimodal_data"]["image"]
    for index, image in enumerate(wanted_images):
        single = native_processor.processor("", images=[image])
        h, w = single["image_sizes"][0].tolist()
        assert payload["image_sizes"][index] == [h, w]
        assert torch.equal(payload["pixel_values"][index, :, :h, :w], single["pixel_values"][0])


def test_raw_text_places_images_at_its_markers(native_processor):
    images = [_rgb(30, (56, 84)), _rgb(31, (112, 56))]
    prompt = "Before [IMG] between [IMG] after"
    ids, extra = native_processor({"prompt": prompt, "multi_modal_data": {"image": images}}, None)
    expected = native_processor.tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Before "},
                    {"type": "image_url", "image_url": {"url": _image_url(images[0])}},
                    {"type": "text", "text": " between "},
                    {"type": "image_url", "image_url": {"url": _image_url(images[1])}},
                    {"type": "text", "text": " after"},
                ],
            }
        ],
        tokenize=True,
        return_dict=False,
    )
    assert ids == expected
    assert len(extra["multimodal_data"]["image"]["image_sizes"]) == 2
    with pytest.raises(ValueError, match="placeholder count"):
        native_processor({"prompt": prompt, "multi_modal_data": {"image": images[:1]}}, None)


def test_token_ids_expand_one_marker_per_image_without_reencoding(native_processor):
    processor = native_processor.processor
    img, img_break, img_end = (
        processor.image_token_id,
        processor.image_break_token_id,
        processor.image_end_token_id,
    )
    images = [_rgb(40, (56, 84)), _rgb(41, (112, 56))]
    ids = [1, 1000, img, 2000, img, 3000]
    actual, _ = native_processor(
        {"prompt_token_ids": ids, "multi_modal_data": {"image": images}}, None
    )
    blocks = native_processor._image_blocks(actual)
    assert actual[: blocks[0][0]] == [1, 1000]
    assert actual[blocks[0][1] : blocks[1][0]] == [2000]
    assert actual[blocks[1][1] :] == [3000]
    again, _ = native_processor(
        {"prompt_token_ids": actual, "multi_modal_data": {"image": images}}, None
    )
    assert actual == again
    for invalid in (actual[:-2], actual + [img_end], [1, img], [1, img_break]):
        with pytest.raises(ValueError, match="[Ii]mage"):
            native_processor(
                {"prompt_token_ids": invalid, "multi_modal_data": {"image": images}}, None
            )
    with pytest.raises(ValueError, match="sizes and order"):
        native_processor(
            {"prompt_token_ids": actual, "multi_modal_data": {"image": list(reversed(images))}},
            None,
        )


def test_text_only_token_ids_are_unchanged(native_processor):
    processor = native_processor.processor
    ids = native_processor.tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Continue"},
        ],
        tokenize=True,
        return_dict=False,
    )
    assert native_processor({"prompt_token_ids": ids}, None) == (ids, None)
    with pytest.raises(ValueError, match="require image data"):
        native_processor(
            {"prompt_token_ids": [1, processor.image_token_id, processor.image_end_token_id]},
            None,
        )


def test_the_default_loader_returns_native_token_ids(native_processor):
    from tensorrt_llm.inputs.utils import default_multimodal_input_loader

    images = [_rgb(50, (56, 84)), _rgb(51, (112, 56))]
    inputs = default_multimodal_input_loader(
        tokenizer=None,
        model_dir=native_processor.model_path,
        model_type="mistral_common",
        modality="multiple_image",
        prompts=["Compare."],
        media=[_image_url(image) for image in images],
    )
    assert len(inputs) == 1 and "prompt_token_ids" in inputs[0]
    ids, extra = native_processor(inputs[0], None)
    assert ids == inputs[0]["prompt_token_ids"]
    singles = [native_processor.processor("", images=[image]) for image in images]
    assert extra["multimodal_data"]["image"]["image_sizes"] == [
        one["image_sizes"][0].tolist() for one in singles
    ]

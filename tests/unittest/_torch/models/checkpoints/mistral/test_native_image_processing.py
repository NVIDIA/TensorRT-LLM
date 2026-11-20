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
deliver a PIL image, a float ``(C, H, W)`` tensor in [0, 1] or a uint8 array; each is
converted losslessly. The server renders a chat to token ids with the image grids in
place; the processor checks those grids against the images without re-encoding the text.

The processor tests read Mistral-Small-4's native files under LLM_MODELS_ROOT and need
OpenCV, which mistral-common's image encoder resizes with.
"""

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


def _rgb(seed=0, size=(37, 23)):
    gen = np.random.default_rng(seed)
    return Image.fromarray(gen.integers(0, 256, size=(*size, 3), dtype=np.uint8), "RGB")


def _every_level():
    """Each of the 256 8-bit levels in every channel, so every level must come back."""
    levels = np.arange(256, dtype=np.uint8).reshape(16, 16)
    return Image.fromarray(np.stack([levels, levels.T, 255 - levels], -1), "RGB")


def _uint8_chw(image, channels=3):
    return torch.from_numpy(np.asarray(image)[..., :channels]).permute(2, 0, 1).contiguous()


def _image_url(image):
    buffer = BytesIO()
    image.save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _user(*parts):
    """A user message of text parts and images, in order."""
    content = [
        {"type": "text", "text": part}
        if isinstance(part, str)
        else {"type": "image_url", "image_url": {"url": _image_url(part)}}
        for part in parts
    ]
    return {"role": "user", "content": content}


@pytest.mark.parametrize(
    "image,form,mode",
    [
        (_rgb(), lambda image: image, "RGB"),
        (_every_level(), ToTensor(), "RGB"),  # the "pt" loader format
        (_rgb(2), np.asarray, "RGB"),
        (_rgb(2), _uint8_chw, "RGB"),
        (_rgb(2), lambda image: _uint8_chw(image, channels=1), "L"),
    ],
    ids=["pil", "pt_every_level", "uint8_hwc_array", "uint8_chw_tensor", "uint8_gray"],
)
def test_to_pil_image_takes_every_loader_form(image, form, mode):
    given = form(image)
    back = _to_pil(given)
    if isinstance(given, Image.Image):
        assert back is given
    assert back.mode == mode
    if mode == "RGB":
        assert np.array_equal(np.asarray(back), np.asarray(image))


def test_an_unknown_type_is_refused():
    with pytest.raises(TypeError, match="unsupported image type"):
        _to_pil([1, 2, 3])


@pytest.mark.parametrize("encoding", ["JPEG", "PNG"])
def test_the_preferred_loader_keeps_the_pixels_of_a_pil_decode(encoding: str) -> None:
    """The media loader format the processor asks for yields an array of Pillow's pixels.

    mistral-common decodes an image with Pillow; the loaders decode a JPEG with torchvision.
    """
    from tensorrt_llm._torch.models.modeling_mistral import MistralNativeInputProcessor
    from tensorrt_llm.inputs.media_io import ImageMediaIO

    preferred = MistralNativeInputProcessor.__new__(
        MistralNativeInputProcessor
    ).get_preferred_media_io_kwargs()
    buffer = BytesIO()
    _rgb(3, (480, 640)).save(buffer, encoding)
    loaded = ImageMediaIO.create(preferred.get("image"), None).load_bytes(buffer.getvalue())
    assert isinstance(loaded, np.ndarray) and loaded.dtype == np.uint8
    pillow = np.asarray(Image.open(BytesIO(buffer.getvalue())).convert("RGB"))
    assert np.array_equal(np.asarray(_to_pil(loaded)), pillow)


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


def _assert_grids(native_processor, payload, images):
    """``payload`` holds each image's own grid, in order."""
    for index, image in enumerate(images):
        single = native_processor.processor("", images=[image])
        h, w = single["image_sizes"][0].tolist()
        sizes = payload["image_sizes"][index]
        assert (sizes.tolist() if torch.is_tensor(sizes) else list(sizes)) == [h, w]
        assert torch.equal(payload["pixel_values"][index, :, :h, :w], single["pixel_values"][0])


@pytest.mark.parametrize(
    "form", [lambda image: image, np.asarray, ToTensor()], ids=["pil", "np", "pt"]
)
def test_pixels_are_what_mistral_common_makes_of_the_loaded_image(
    native_processor, form, monkeypatch
):
    """Nothing lies between the loaded image and mistral-common's resize and normalize.

    The server loads ``np`` arrays; other callers may pass PIL images or ``pt`` tensors. No
    image is serialized on the way.
    """
    from mistral_common.protocol.instruct.chunk import ImageChunk

    processor = MistralCommonImageProcessor(native_processor.tokenizer, torch.float32)
    image = Image.open(_INPUT_FILES / "excel_table_test.jpg").convert("RGB")
    expected = torch.as_tensor(processor.image_processor(ImageChunk(image=image)).image)
    loaded = form(image)

    def no_serialization(*args, **kwargs):
        raise AssertionError("the processor serialized an image")

    monkeypatch.setattr(Image.Image, "save", no_serialization)
    processed = processor("", images=[loaded])
    pixels = processed["pixel_values"][0]
    assert torch.equal(pixels, expected.to(pixels.dtype))
    assert processed["image_sizes"].tolist() == [list(expected.shape[-2:])]


@pytest.mark.parametrize(
    "image,size",
    [
        (_rgb(size=(23, 37)), (37, 23)),
        (ToTensor()(_rgb(size=(23, 37))), (37, 23)),
        (np.zeros((23, 37, 3), dtype=np.uint8), (37, 23)),
    ],
    ids=["pil", "chw_tensor", "hwc_array"],
)
def test_pil_image_size_is_the_size_of_the_pil_image(image, size):
    assert MistralCommonImageProcessor.pil_image_size(image) == size
    assert _to_pil(image).size == size


@pytest.mark.parametrize("served", [True, False], ids=["server_loader", "default_loader"])
def test_a_served_chat_keeps_roles_image_order_and_control_tokens(native_processor, served):
    """A chat renders and processes to the reference ids and grids.

    The server loads images in the format the processor prefers (``np`` arrays); a caller
    without that configuration gets the loaders' default (``pt`` tensors).
    """
    from tensorrt_llm.inputs.content_format import ContentFormat
    from tensorrt_llm.inputs.multimodal import MultimodalServerConfig
    from tensorrt_llm.inputs.registry import MULTIMODAL_PLACEHOLDER_REGISTRY
    from tensorrt_llm.inputs.utils import async_apply_chat_template_with_mm
    from tensorrt_llm.serve.chat_utils import parse_chat_messages_coroutines

    assert (
        MULTIMODAL_PLACEHOLDER_REGISTRY.get_content_format("mistral_common") == ContentFormat.OPENAI
    )
    first, second = _rgb(21, (56, 84)), _rgb(22, (112, 56))
    messages = [
        {"role": "system", "content": "Describe images in order."},
        _user("First ", first, " then ", second, "; compare them."),
        {"role": "assistant", "content": "Two images."},
        _user("Now compare the first to ", first, " again."),
    ]
    expected = native_processor.tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False, add_generation_prompt=True
    )
    server_config = (
        MultimodalServerConfig(media_io_kwargs=native_processor.get_preferred_media_io_kwargs())
        if served
        else None
    )
    conversation, media, counts, _ = parse_chat_messages_coroutines(
        messages, native_processor.config, multimodal_server_config=server_config
    )
    assert counts == [{}, {"[IMG]": 2}, {}, {"[IMG]": 1}]
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
    loaded_type = np.ndarray if served else torch.Tensor
    assert all(isinstance(image, loaded_type) for image in mm_data["image"])
    actual_ids, extra = native_processor(
        {"prompt_token_ids": ids, "multi_modal_data": mm_data}, None
    )
    assert actual_ids == expected
    _assert_grids(native_processor, extra["multimodal_data"]["image"], [first, second, first])


def test_raw_text_places_images_at_its_markers(native_processor):
    images = [_rgb(30, (56, 84)), _rgb(31, (112, 56))]
    prompt = "Before [IMG] between [IMG] after"
    ids, extra = native_processor({"prompt": prompt, "multi_modal_data": {"image": images}}, None)
    expected = native_processor.tokenizer.apply_chat_template(
        [_user("Before ", images[0], " between ", images[1], " after")],
        tokenize=True,
        return_dict=False,
    )
    assert ids == expected
    assert len(extra["multimodal_data"]["image"]["image_sizes"]) == 2
    with pytest.raises(ValueError, match="placeholder count"):
        native_processor({"prompt": prompt, "multi_modal_data": {"image": images[:1]}}, None)


def test_token_ids_carry_whole_image_grids_without_reencoding(native_processor):
    processor = native_processor.processor
    img, img_break, img_end = (
        processor.image_token_id,
        processor.image_break_token_id,
        processor.image_end_token_id,
    )
    images = [_rgb(40, (56, 84)), _rgb(41, (112, 56))]

    def process(ids, images=images):
        return native_processor(
            {"prompt_token_ids": ids, "multi_modal_data": {"image": images}}, None
        )

    assert native_processor({"prompt_token_ids": [1, 1000]}, None) == ([1, 1000], None)
    with pytest.raises(ValueError, match="require image data"):
        native_processor({"prompt_token_ids": [1, img, img_end]}, None)
    encoded = processor("", images=images)["input_ids"][0].tolist()
    first, second = (encoded[a:b] for a, b in native_processor._image_blocks(encoded))
    actual = [1, 1000, *first, 2000, *second, 3000]
    assert process(actual)[0] == actual
    # One [IMG] per image is not a grid: the ids are checked, never expanded.
    one_marker_each = [1, 1000, img, 2000, img, 3000]
    for invalid in (one_marker_each, actual[:-2], actual + [img_end], [1, img], [1, img_break]):
        with pytest.raises(ValueError, match="[Ii]mage"):
            process(invalid)
    with pytest.raises(ValueError, match="sizes and order"):
        process(actual, list(reversed(images)))


def test_the_default_loader_places_images_at_the_prompts_markers(native_processor):
    """As the input processor places a text prompt's images, one per [IMG] marker."""
    from tensorrt_llm.inputs.utils import default_multimodal_input_loader

    images = [_rgb(52, (56, 84)), _rgb(53, (112, 56))]
    prompt = "Before [IMG] between [IMG] after"

    def load(prompt):
        return default_multimodal_input_loader(
            tokenizer=None,
            model_dir=native_processor.model_path,
            model_type="mistral_common",
            modality="multiple_image",
            prompts=[prompt],
            media=[_image_url(image) for image in images],
        )

    (inputs,) = load(prompt)
    expected, _ = native_processor({"prompt": prompt, "multi_modal_data": {"image": images}}, None)
    assert inputs["prompt_token_ids"] == expected
    ids, extra = native_processor(inputs, None)
    assert ids == expected
    _assert_grids(native_processor, extra["multimodal_data"]["image"], images)
    with pytest.raises(ValueError, match="1 \\[IMG\\] placeholders for 2 images"):
        load("Only [IMG] one")


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
    _assert_grids(native_processor, extra["multimodal_data"]["image"], images)

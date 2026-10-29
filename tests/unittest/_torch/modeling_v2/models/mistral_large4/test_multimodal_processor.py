# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The mistral-native image processor takes an image in every form the media loaders
hand one over.

`checkpoint_format: mistral_large_4` builds `MistralNativeInputProcessor`, whose
`MistralCommonImageProcessor` encodes each image for mistral-common. The loaders
deliver a PIL image (`"pil"`), a float `(C, H, W)` tensor in [0, 1] (`"pt"`, the
default of `default_multimodal_input_loader` and of the boot gate) or a uint8
`(H, W, C)` array (`"np"`); before `to_pil_image` the processor took PIL only and a
tensor raised `AttributeError: 'Tensor' object has no attribute 'convert'`.

trtllm-serve resolves a chat request's image placeholder through the
placeholder registry under the config class's model_type, which for the
Mistral Large 4 composite config is "mistral_large_4"; with the metadata registered only
under "mistral_common" every image chat request was refused with
`TypeError: Unknown modality: image`.

The conversion and placeholder tests run on CPU. The end-to-end test runs the
processor on the boot image when MISTRAL_LARGE4_CHECKPOINT_DIR names the checkpoint
(its tekken.json) and OpenCV is installed (mistral-common's image encoder needs
it): every format gives the same token ids and pixels, and the geometry the
Mistral Large 4 tower and its splice expect.
"""

from __future__ import annotations

import asyncio
import base64
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from tensorrt_llm._torch.models.modeling_mistral import MistralCommonImageProcessor

pytestmark = pytest.mark.cpu_only

_to_pil = MistralCommonImageProcessor.to_pil_image

#: The boot gate's image, from the tree under test.
_BOOT_IMAGE = (
    Path(__file__).resolve().parents[5] / "integration" / "test_input_files" / "merlion.png"
)


def _rgb(seed: int = 0, size=(37, 23)) -> Image.Image:
    gen = np.random.default_rng(seed)
    return Image.fromarray(gen.integers(0, 256, size=(*size, 3), dtype=np.uint8), "RGB")


def _every_level() -> Image.Image:
    """Each of the 256 8-bit levels in every channel, so every level must come back."""
    levels = np.arange(256, dtype=np.uint8).reshape(16, 16)
    return Image.fromarray(np.stack([levels, levels.T, 255 - levels], -1), "RGB")


def _pt(image: Image.Image) -> torch.Tensor:
    """The `"pt"` format is `ToTensor`: uint8 / 255 in fp32."""
    tensor = ToTensor()(image)
    assert tensor.dtype == torch.float32 and tensor.shape == (3, image.height, image.width)
    return tensor


def _uint8_chw(image: Image.Image, channels: int = 3) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image)[..., :channels]).permute(2, 0, 1).contiguous()


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
    """A PIL image passes through; a `"pt"` tensor comes back at its exact 8-bit
    levels; a uint8 `(H, W, C)` array or `(C, H, W)` tensor comes back as it is,
    with one channel as mode "L"."""
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


def test_the_server_places_an_image_for_the_mistral_large4_config():
    """The chat parsing trtllm-serve runs on a request with an image, on the
    config the mistral loader builds for a Mistral Large 4 checkpoint."""
    from tensorrt_llm._torch.configs.mistral_large4 import MistralLarge4Config
    from tensorrt_llm.inputs.registry import MULTIMODAL_PLACEHOLDER_REGISTRY
    from tensorrt_llm.serve.chat_utils import (
        parse_chat_messages_coroutines,
        resolve_top_level_model_type,
    )

    config = MistralLarge4Config()
    assert resolve_top_level_model_type(config) == "mistral_large_4"
    from tensorrt_llm.inputs.content_format import ContentFormat

    assert (
        MULTIMODAL_PLACEHOLDER_REGISTRY.get_content_format("mistral_large_4")
        == ContentFormat.OPENAI
    )

    buffer = BytesIO()
    _rgb(5).save(buffer, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    prompt = "Describe this image in one sentence."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }
    ]
    conversation, media, counts, _ = parse_chat_messages_coroutines(messages, config)
    mm_data, _ = asyncio.run(media)
    assert counts == [{"[IMG]": 1}]
    # The native renderer receives the ordered content parts and resolved images.
    assert conversation[0]["content"] == prompt
    assert len(mm_data["image"]) == 1


@pytest.mark.skipif(
    not os.environ.get("MISTRAL_LARGE4_CHECKPOINT_DIR"),
    reason="MISTRAL_LARGE4_CHECKPOINT_DIR names no Mistral Large 4 checkpoint",
)
def test_every_loader_format_encodes_the_boot_image_alike():
    pytest.importorskip("cv2")
    from tensorrt_llm._torch.models.checkpoints.mistral.tokenizer import MistralTokenizer
    from tensorrt_llm.inputs.utils import load_image

    tokenizer = MistralTokenizer.from_pretrained(os.environ["MISTRAL_LARGE4_CHECKPOINT_DIR"])
    processor = MistralCommonImageProcessor(
        tokenizer=tokenizer, dtype=torch.bfloat16, native_chat=True
    )
    prompt = "Describe this image in one sentence."
    pil = load_image(str(_BOOT_IMAGE), format="pil")
    images = {
        "pil": pil,
        "pt": load_image(str(_BOOT_IMAGE), format="pt"),
        "np": np.asarray(pil),
    }
    results = {fmt: processor(text=prompt, images=[image]) for fmt, image in images.items()}
    ids = results["pil"]["input_ids"]
    pixels = results["pil"]["pixel_values"]
    for fmt in ("pt", "np"):
        assert torch.equal(results[fmt]["input_ids"], ids), fmt
        assert torch.equal(results[fmt]["pixel_values"], pixels), fmt
    # 1192 x 874 -> 1204 x 896 pixels: a 64 x 86 patch grid, 32 x 43 merged tiles,
    # one [IMG] per tile, [IMG_BREAK] after each tile row but the last, [IMG_END].
    assert tuple(pixels.shape) == (1, 3, 896, 1204)
    assert results["pil"]["image_sizes"].tolist() == [[896, 1204]]
    flat = ids[0]
    assert int((flat == processor.image_token_id).sum()) == 32 * 43
    assert int((flat == processor.image_break_token_id).sum()) == 31
    assert int((flat == processor.image_end_token_id).sum()) == 1
    assert int(flat[0]) == 1, "BOS"


@pytest.mark.skipif(
    not os.environ.get("MISTRAL_LARGE4_CHECKPOINT_DIR"), reason="needs checkpoint tokenizer"
)
def test_two_different_image_sizes_preserve_each_grid():
    pytest.importorskip("cv2")
    from tensorrt_llm._torch.models.checkpoints.mistral.tokenizer import MistralTokenizer

    tokenizer = MistralTokenizer.from_pretrained(os.environ["MISTRAL_LARGE4_CHECKPOINT_DIR"])
    processor = MistralCommonImageProcessor(tokenizer, torch.bfloat16, native_chat=True)
    images = [_rgb(12, (224, 336)), _rgb(13, (448, 280))]
    singles = [processor("Describe.", images=[image]) for image in images]
    together = processor("Describe.", images=images)
    assert together["image_sizes"].tolist() == [one["image_sizes"][0].tolist() for one in singles]
    for i, one in enumerate(singles):
        h, w = one["image_sizes"][0].tolist()
        assert torch.equal(together["pixel_values"][i, :, :h, :w], one["pixel_values"][0])
    expected = sum(int((one["input_ids"] == processor.image_token_id).sum()) for one in singles)
    assert int((together["input_ids"] == processor.image_token_id).sum()) == expected


@pytest.mark.skipif(
    not os.environ.get("MISTRAL_LARGE4_CHECKPOINT_DIR"), reason="needs checkpoint tokenizer"
)
@pytest.mark.parametrize("name", ["merlion.png", "excel_table_test.jpg"])
def test_pixels_are_what_mistral_common_makes_of_the_loaded_image(name):
    """Nothing lossy lies between the loaded image and mistral-common's resize and normalize."""
    pytest.importorskip("cv2")
    from mistral_common.protocol.instruct.chunk import ImageChunk

    from tensorrt_llm._torch.models.checkpoints.mistral.tokenizer import MistralTokenizer

    tokenizer = MistralTokenizer.from_pretrained(os.environ["MISTRAL_LARGE4_CHECKPOINT_DIR"])
    processor = MistralCommonImageProcessor(tokenizer, torch.float32, native_chat=True)
    image = Image.open(_BOOT_IMAGE.parent / name).convert("RGB")
    expected = torch.as_tensor(processor.image_processor(ImageChunk(image=image)).image)
    pixels = processor("", images=[image])["pixel_values"][0]
    assert torch.equal(pixels, expected.to(pixels.dtype))


@pytest.fixture(scope="module")
def native_processor():
    if not os.environ.get("MISTRAL_LARGE4_CHECKPOINT_DIR"):
        pytest.skip("needs checkpoint tokenizer")
    from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.routing import (
        serves_images,
    )
    from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import MistralConfigLoader
    from tensorrt_llm._torch.models.modeling_mistral import MistralNativeInputProcessor

    path = os.environ["MISTRAL_LARGE4_CHECKPOINT_DIR"]
    # The checkpoint's own configuration: its vision encoder and its NVFP4 export are what make
    # the processor take images.
    config = MistralConfigLoader().load_pretrained_config(path)
    if not serves_images(config):
        pytest.skip(f"{path}: this checkpoint's target serves text only")
    return MistralNativeInputProcessor(path, config, None)


def _image_url(image):
    buffer = BytesIO()
    image.save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


@pytest.mark.parametrize("multi_turn", [False, True])
def test_native_chat_keeps_roles_image_order_and_control_tokens(native_processor, multi_turn):
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
            model_type="mistral_large_4",
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
    assert ids[0] == 1
    wanted_images = images + images[:1] if multi_turn else images
    payload = extra["multimodal_data"]["image"]
    for index, image in enumerate(wanted_images):
        single = native_processor.processor("", images=[image])
        h, w = single["image_sizes"][0].tolist()
        assert payload["image_sizes"][index] == [h, w]
        assert torch.equal(payload["pixel_values"][index, :, :h, :w], single["pixel_values"][0])


def test_raw_text_interleaving_matches_native_chat(native_processor):
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


def test_token_ids_expand_placeholders_without_reencoding(native_processor):
    images = [_rgb(40, (56, 84)), _rgb(41, (112, 56))]
    ids = [1, 1000, 10, 2000, 10, 3000]
    actual, _ = native_processor(
        {"prompt_token_ids": ids, "multi_modal_data": {"image": images}}, None
    )
    blocks = native_processor._image_blocks(actual)
    assert actual[: blocks[0][0]] == [1, 1000]
    assert actual[blocks[0][1] : blocks[1][0]] == [2000]
    assert actual[blocks[1][1] :] == [3000]
    assert actual.count(10) == (2 * 3) + (4 * 2)
    again, _ = native_processor(
        {"prompt_token_ids": actual, "multi_modal_data": {"image": images}}, None
    )
    assert actual == again
    for invalid in (actual[:-2], actual + [13], [1, 10], [1, 12]):
        with pytest.raises(ValueError, match="[Ii]mage"):
            native_processor(
                {"prompt_token_ids": invalid, "multi_modal_data": {"image": images}}, None
            )
    with pytest.raises(ValueError, match="sizes and order"):
        native_processor(
            {"prompt_token_ids": actual, "multi_modal_data": {"image": list(reversed(images))}},
            None,
        )


def test_text_only_native_token_ids_are_unchanged(native_processor):
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
        native_processor({"prompt_token_ids": [1, 10, 13]}, None)


def test_default_loader_returns_native_token_ids(native_processor):
    from tensorrt_llm.inputs.utils import default_multimodal_input_loader

    images = [_rgb(50, (56, 84)), _rgb(51, (112, 56))]
    inputs = default_multimodal_input_loader(
        tokenizer=native_processor.tokenizer,
        model_dir=native_processor.model_path,
        model_type="mistral_large_4",
        modality="multiple_image",
        prompts=["Compare."],
        media=[_image_url(image) for image in images],
    )
    assert len(inputs) == 1 and "prompt_token_ids" in inputs[0]
    ids, extra = native_processor(inputs[0], None)
    assert ids == inputs[0]["prompt_token_ids"]
    assert extra["multimodal_data"]["image"]["image_sizes"] == [[56, 84], [112, 56]]


def test_image_hashes_and_positions_are_independent_of_companions(native_processor):
    from tensorrt_llm.inputs.registry import create_input_processor_with_hash

    wrapped = create_input_processor_with_hash(native_processor)
    a, b, c = [_rgb(seed, (56, 84)) for seed in (61, 62, 63)]

    def process(images, prompt):
        ids, extra = wrapped({"prompt": prompt, "multi_modal_data": {"image": images}}, None)
        assert native_processor.multimodal_hashing_supported is True
        mm = extra["multimodal_input"]
        blocks = native_processor._image_blocks(ids)
        assert mm.multimodal_positions == [begin for begin, _ in blocks]
        assert mm.multimodal_lengths == [end - begin for begin, end in blocks]
        counts = extra["multimodal_data"]["multimodal_embedding_lengths"]
        assert counts == [6] * len(images)
        return ids, mm.multimodal_hashes

    alone, ha = process([a], "Color [IMG]?")
    warm, ha_warm = process([a], "Color [IMG]?")
    different, hb = process([b], "Color [IMG]?")
    ab, hab = process([a, b], "First [IMG], then [IMG].")
    ac, hac = process([a, c], "First [IMG], then [IMG].")
    ba, hba = process([b, a], "First [IMG], then [IMG].")
    assert alone == warm == different
    assert ha == ha_warm and ha != hb
    assert ab == ac == ba
    assert hab[0] == hac[0] == hba[1] == ha[0]
    assert hab[1] == hba[0] == hb[0]
    assert hac[1] != hab[1]

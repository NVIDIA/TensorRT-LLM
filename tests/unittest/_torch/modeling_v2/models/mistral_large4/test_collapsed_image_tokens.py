# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chat token ids with one [IMG] per image expand to the ids trtllm-serve renders.

A frontend that renders Mistral-Large-4 chats itself, without resizing images, puts
one [IMG] where each image goes and hands the engine those token ids with the decoded
images. `MistralNativeInputProcessor` expands each [IMG] into the image's grid -- an
[IMG] per merged tile, [IMG_BREAK] after every row but the last, [IMG_END] -- from the
image as mistral-common resizes it.

These tests render the collapsed ids independently of that expansion: mistral-common's
own chat encoder with an image encoder that emits a single [IMG] per image. The
expansion must reproduce trtllm-serve's native chat path exactly -- token ids, pixels
and image sizes -- for one image, several images of different sizes, text interleaved
with images, image-only turns, multi-turn chats with earlier images, tools with
reasoning effort, an image larger than the encoder's maximum size, and JPEG / RGBA PNG /
palette PNG inputs. The images reach the processor as decoded PIL images in RGB,
transparency composited over white, while trtllm-serve loads the same bytes itself.

CPU only. Needs MISTRAL_LARGE4_CHECKPOINT_DIR (the checkpoint's tekken.json and
params.json) and OpenCV, which mistral-common's image resize uses.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import os
from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.cpu_only


@pytest.fixture(scope="module")
def native_processor():
    if not os.environ.get("MISTRAL_LARGE4_CHECKPOINT_DIR"):
        pytest.skip("needs the checkpoint's tokenizer and configuration")
    pytest.importorskip("cv2")
    from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.routing import (
        serves_images,
    )
    from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import MistralConfigLoader
    from tensorrt_llm._torch.models.modeling_mistral import MistralNativeInputProcessor

    path = os.environ["MISTRAL_LARGE4_CHECKPOINT_DIR"]
    config = MistralConfigLoader().load_pretrained_config(path)
    if not serves_images(config):
        pytest.skip(f"{path}: this checkpoint's target serves text only")
    return MistralNativeInputProcessor(path, config, None)


def _noise(seed: int, height: int, width: int, mode: str = "RGB") -> Image.Image:
    gen = np.random.default_rng(seed)
    channels = {"RGB": 3, "RGBA": 4}[mode]
    pixels = gen.integers(0, 256, size=(height, width, channels), dtype=np.uint8)
    return Image.fromarray(pixels, mode)


def _encoded(image: Image.Image, fmt: str) -> bytes:
    buffer = BytesIO()
    image.save(buffer, fmt)
    return buffer.getvalue()


def _png(seed: int, height: int, width: int) -> bytes:
    return _encoded(_noise(seed, height, width), "PNG")


def _data_url(data: bytes) -> str:
    kind = "jpeg" if data[:2] == b"\xff\xd8" else "png"
    return f"data:image/{kind};base64," + base64.b64encode(data).decode()


def _worker_image(data: bytes) -> Image.Image:
    """The image as a frontend's worker decodes it before handing it to the engine: RGB, with
    transparency composited over white as trtllm-serve's media loader does."""
    from tensorrt_llm.inputs.media_io import convert_image_mode

    return convert_image_mode(Image.open(BytesIO(data), formats=["JPEG", "PNG", "WEBP"]), "RGB")


def _image_part(data: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": _data_url(data)}}


def _text(text: str) -> dict:
    return {"type": "text", "text": text}


@contextlib.contextmanager
def _one_img_per_image(tokenizer):
    """mistral-common's chat encoder, with every image encoded as a single [IMG]."""
    from mistral_common.tokens.tokenizers.image import ImageEncoding

    instruct = tokenizer.mistral.instruct_tokenizer
    image_encoder = instruct.image_encoder
    image_token = image_encoder.special_ids.img

    def one_token(chunk):
        return ImageEncoding(tokens=[image_token], image=np.zeros((3, 1, 1), np.float32))

    instruct.image_encoder = one_token
    try:
        yield image_token
    finally:
        instruct.image_encoder = image_encoder


def _collapsed_ids(tokenizer, messages, tools, reasoning_effort):
    from mistral_common.protocol.instruct.request import ChatCompletionRequest

    with _one_img_per_image(tokenizer):
        request = ChatCompletionRequest.from_openai(
            messages=messages, tools=tools, reasoning_effort=reasoning_effort
        )
        return tokenizer.mistral.encode_chat_completion(request).tokens


def _serve_ids(native_processor, messages, tools, reasoning_effort):
    """Token ids and images of trtllm-serve's native chat path for `messages`."""
    from tensorrt_llm.inputs.utils import async_apply_chat_template_with_mm
    from tensorrt_llm.serve.chat_utils import parse_chat_messages_coroutines

    conversation, media, counts, _ = parse_chat_messages_coroutines(
        messages, native_processor.config
    )
    kwargs = {"reasoning_effort": reasoning_effort} if reasoning_effort else None
    ids, (mm_data, _) = asyncio.run(
        async_apply_chat_template_with_mm(
            media,
            model_type="mistral_large_4",
            tokenizer=native_processor.tokenizer,
            processor=None,
            conversation=conversation,
            add_generation_prompt=True,
            mm_placeholder_counts=counts,
            tools=tools,
            chat_template_kwargs=kwargs,
        )
    )
    return ids, mm_data


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "describe_region",
            "description": "Describe a region of an image.",
            "parameters": {
                "type": "object",
                "properties": {"image": {"type": "integer"}, "region": {"type": "string"}},
                "required": ["image", "region"],
            },
        },
    }
]


def _cases():
    one = _png(1, 280, 420)
    small = _png(2, 56, 84)
    tall = _png(3, 900, 300)
    eight = [_png(10 + i, 56 + 37 * i, 420 - 29 * i) for i in range(8)]
    large = _png(4, 2300, 1700)
    jpeg = _encoded(_noise(5, 333, 517), "JPEG")
    rgba = _encoded(_noise(6, 200, 300, "RGBA"), "PNG")
    palette = _encoded(_noise(7, 150, 240).convert("P"), "PNG")
    return {
        "single": (
            [{"role": "user", "content": [_text("Describe this image."), _image_part(one)]}],
            None,
            None,
        ),
        "image_only": ([{"role": "user", "content": [_image_part(small)]}], None, None),
        "interleaved": (
            [
                {"role": "system", "content": "Compare images in order."},
                {
                    "role": "user",
                    "content": [
                        _text("First "),
                        _image_part(small),
                        _text(" then "),
                        _image_part(tall),
                        _text("; which is taller?"),
                    ],
                },
            ],
            None,
            None,
        ),
        "eight_images": (
            [
                {
                    "role": "user",
                    "content": [_text("Count the images.")] + [_image_part(i) for i in eight],
                }
            ],
            None,
            None,
        ),
        "multi_turn": (
            [
                {"role": "user", "content": [_image_part(one), _text("What is this?")]},
                {"role": "assistant", "content": "Random noise."},
                {
                    "role": "user",
                    "content": [_text("And this one, next to the first?"), _image_part(tall)],
                },
            ],
            None,
            None,
        ),
        "tools_reasoning": (
            [
                {"role": "system", "content": "Use tools when helpful."},
                {"role": "user", "content": [_text("Describe the top half."), _image_part(one)]},
            ],
            _TOOLS,
            "high",
        ),
        "larger_than_encoder": (
            [{"role": "user", "content": [_text("Describe."), _image_part(large)]}],
            None,
            None,
        ),
        "jpeg_rgba_palette": (
            [
                {
                    "role": "user",
                    "content": [
                        _image_part(jpeg),
                        _image_part(rgba),
                        _image_part(palette),
                        _text("Which image has the most colors?"),
                    ],
                }
            ],
            None,
            "low",
        ),
    }


_CASES = _cases()


def _image_bytes(messages):
    images = []
    for message in messages:
        content = message.get("content")
        for part in content if isinstance(content, list) else []:
            if part["type"] == "image_url":
                url = part["image_url"]["url"]
                images.append(base64.b64decode(url.split(",", 1)[1]))
    return images


@pytest.mark.parametrize("name", list(_CASES))
def test_collapsed_ids_expand_to_the_served_chat(native_processor, name):
    messages, tools, reasoning_effort = _CASES[name]
    images = _image_bytes(messages)
    proc = native_processor.processor
    served, served_mm = _serve_ids(native_processor, messages, tools, reasoning_effort)
    collapsed = _collapsed_ids(native_processor.tokenizer, messages, tools, reasoning_effort)

    assert collapsed.count(proc.image_token_id) == len(images)
    assert proc.image_break_id not in collapsed and proc.image_end_id not in collapsed
    assert len(native_processor._image_blocks(served)) == len(images)

    expanded, extra = native_processor(
        {
            "prompt_token_ids": collapsed,
            "multi_modal_data": {"image": [_worker_image(data) for data in images]},
        },
        None,
    )
    assert expanded == served

    _, served_extra = native_processor(
        {"prompt_token_ids": served, "multi_modal_data": served_mm}, None
    )
    payload = extra["multimodal_data"]["image"]
    reference = served_extra["multimodal_data"]["image"]
    assert payload["image_sizes"] == reference["image_sizes"]
    assert torch.equal(payload["pixel_values"], reference["pixel_values"])


def test_collapsed_ids_hash_and_place_images_as_the_served_chat(native_processor):
    """KV-cache reuse keys images by content and position: both inputs must agree."""
    from tensorrt_llm.inputs.registry import create_input_processor_with_hash

    wrapped = create_input_processor_with_hash(native_processor)
    messages, tools, reasoning_effort = _CASES["interleaved"]
    images = [_worker_image(data) for data in _image_bytes(messages)]
    served, _ = _serve_ids(native_processor, messages, tools, reasoning_effort)
    collapsed = _collapsed_ids(native_processor.tokenizer, messages, tools, reasoning_effort)

    def process(ids):
        out_ids, extra = wrapped(
            {"prompt_token_ids": ids, "multi_modal_data": {"image": images}}, None
        )
        mm = extra["multimodal_input"]
        return out_ids, mm.multimodal_hashes, mm.multimodal_positions, mm.multimodal_lengths

    assert process(collapsed) == process(served)


def test_collapsed_ids_must_match_the_images(native_processor):
    messages, tools, reasoning_effort = _CASES["interleaved"]
    images = [_worker_image(data) for data in _image_bytes(messages)]
    collapsed = _collapsed_ids(native_processor.tokenizer, messages, tools, reasoning_effort)
    with pytest.raises(ValueError, match="placeholder count"):
        native_processor(
            {"prompt_token_ids": collapsed, "multi_modal_data": {"image": images[:1]}}, None
        )
    with pytest.raises(ValueError, match="require image data"):
        native_processor({"prompt_token_ids": collapsed}, None)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""prompt_embeds on Qwen-VL models, alone and batched with image requests."""

import json
import os
import uuid
from typing import Iterator, List, Tuple

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors import safe_open

from tensorrt_llm import LLM
from tensorrt_llm.inputs import default_multimodal_input_loader
from tensorrt_llm.inputs.multimodal_data import VideoData
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.sampling_params import SamplingParams

# isort: off
from .test_llm import get_model_path
# isort: on

pytestmark = pytest.mark.threadleak(enabled=False)

PROMPTS = ["The capital of France is", "Write a haiku about the ocean. The waves"]


def _greedy() -> SamplingParams:
    return SamplingParams(max_tokens=16, temperature=0.0)


def _run(llm: LLM, requests: list) -> List[List[int]]:
    outputs = llm.generate(requests, _greedy(), cache_salt=uuid.uuid4().hex)
    return [list(o.outputs[0].token_ids) for o in outputs]


def _text_embedding_table(model_dir: str) -> torch.Tensor:
    """Read the language model's input embedding table from the checkpoint."""
    for name in sorted(os.listdir(model_dir)):
        if not name.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(model_dir, name), "pt") as f:
            for key in f.keys():
                if key.endswith("embed_tokens.weight") and "visual" not in key:
                    return f.get_tensor(key)
    raise FileNotFoundError(f"no embed_tokens.weight under {model_dir}")


@pytest.fixture(scope="module", params=["Qwen3/Qwen3-VL-2B-Instruct", "Qwen2.5-VL-3B-Instruct"])
def model_dir(request) -> str:
    path = get_model_path(request.param)
    if not os.path.isdir(path):
        pytest.skip(f"{path} not found")
    return path


@pytest.fixture(scope="module")
def llm(model_dir: str) -> Iterator[LLM]:
    with LLM(
        model=model_dir,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4, enable_block_reuse=True),
        max_batch_size=4,
    ) as instance:
        yield instance


def _image_request(llm: LLM, model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json")) as f:
        model_type = json.load(f)["model_type"]
    image = Image.new("RGB", (224, 224), (200, 40, 40))
    (request,) = default_multimodal_input_loader(
        tokenizer=llm.tokenizer,
        model_dir=model_dir,
        model_type=model_type,
        modality="image",
        prompts=["Describe the color of this image."],
        media=[[image]],
        image_data_format="pt",
    )
    return request


def test_text_prompt_embeds_match_tokens(llm: LLM, model_dir: str) -> None:
    table = _text_embedding_table(model_dir)
    ids = [llm.tokenizer.encode(p) for p in PROMPTS]
    expected = _run(llm, ids)
    n = len(ids[1])
    mask = [p % 2 == 0 for p in range(n)]
    requests = [
        {"prompt_embeds": table[torch.tensor(ids[0])]},
        {
            "prompt_token_ids": ids[1],
            "prompt_embeds": table[torch.tensor([t for t, m in zip(ids[1], mask) if m])],
            "prompt_embeds_mask": mask,
        },
    ]
    assert _run(llm, requests) == expected


def test_batched_with_image_request(llm: LLM, model_dir: str) -> None:
    table = _text_embedding_table(model_dir)
    image = _image_request(llm, model_dir)
    ids = llm.tokenizer.encode(PROMPTS[0])
    expected = _run(llm, [image, ids])
    assert _run(llm, [image, {"prompt_embeds": table[torch.tensor(ids)]}]) == expected


def test_prompt_embeds_with_image_in_one_request(llm: LLM, model_dir: str) -> None:
    # The mask indexes the ids before the image placeholder expands.
    table = _text_embedding_table(model_dir)
    image = _image_request(llm, model_dir)
    ids = llm.tokenizer.encode(image["prompt"], add_special_tokens=False)
    plain = {"prompt_token_ids": ids, "multi_modal_data": image["multi_modal_data"]}
    expected = _run(llm, [plain])
    vision_ids = set(
        llm.tokenizer.convert_tokens_to_ids(["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"])
    )
    mask = [t not in vision_ids and p % 2 == 0 for p, t in enumerate(ids)]
    request = {
        **plain,
        "prompt_embeds": table[torch.tensor([t for t, m in zip(ids, mask) if m])],
        "prompt_embeds_mask": mask,
    }
    assert _run(llm, [request]) == expected

    image_pos = ids.index(llm.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    on_image = [p == image_pos for p in range(len(ids))]
    with pytest.raises(ValueError, match="multimodal placeholder"):
        llm.generate(
            {**plain, "prompt_embeds": table[:1], "prompt_embeds_mask": on_image}, _greedy()
        )


def _video_request(llm: LLM) -> dict:
    """Chat prompt with one video placeholder and a short red-then-blue clip."""
    indices = [0, 80, 160, 239]
    frames = []
    for index in indices:
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        frame[:, :, 0 if index < 120 else 2] = 255
        frames.append(frame)
    video = VideoData(
        frames=np.stack(frames),
        metadata={
            "fps": 24.0,
            "total_num_frames": 240,
            "duration": 10.0,
            "frames_indices": indices,
        },
    )
    prompt = llm.tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": [{"type": "video"}, {"type": "text", "text": "What happens?"}],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return {
        "prompt_token_ids": llm.tokenizer.encode(prompt, add_special_tokens=False),
        "multi_modal_data": {"video": [video]},
        "mm_processor_kwargs": {"do_sample_frames": False},
    }


def test_prompt_embeds_with_video_in_one_request(llm: LLM, model_dir: str) -> None:
    # Qwen3-VL frames each video frame with timestamp text the mask never indexes.
    if "Qwen3-VL" not in model_dir:
        pytest.skip("timestamp-framed video layout is Qwen3-VL specific")
    table = _text_embedding_table(model_dir)
    plain = _video_request(llm)
    ids = plain["prompt_token_ids"]
    expected = _run(llm, [plain])
    vision_ids = set(
        llm.tokenizer.convert_tokens_to_ids(["<|vision_start|>", "<|video_pad|>", "<|vision_end|>"])
    )
    mask = [t not in vision_ids and p % 2 == 1 for p, t in enumerate(ids)]
    request = {
        **plain,
        "prompt_embeds": table[torch.tensor([t for t, m in zip(ids, mask) if m])],
        "prompt_embeds_mask": mask,
    }
    assert _run(llm, [request]) == expected

    # Rows the model can tell apart: swap two words around the video through
    # embeddings only; the reference swaps the ids themselves.
    before, after = _swappable_positions(llm, ids, vision_ids)
    swapped_ids = list(ids)
    swapped_ids[before], swapped_ids[after] = ids[after], ids[before]
    swapped = {
        **plain,
        "prompt_embeds": table[torch.tensor([ids[after], ids[before]])],
        "prompt_embeds_mask": [p in (before, after) for p in range(len(ids))],
    }
    expected_swap = _run(llm, [{**plain, "prompt_token_ids": swapped_ids}])
    assert expected_swap != expected
    assert _run(llm, [swapped]) == expected_swap


def _swappable_positions(llm: LLM, ids: List[int], vision_ids: set) -> Tuple[int, int]:
    """Pick a word before and one after the vision tokens whose swap re-tokenizes stably."""
    first_vision = min(p for p, t in enumerate(ids) if t in vision_ids)
    last_vision = max(p for p, t in enumerate(ids) if t in vision_ids)
    words = [
        p
        for p, t in enumerate(ids)
        if t not in vision_ids and llm.tokenizer.decode([t]).strip().isalpha()
    ]
    for before in (p for p in words if p < first_vision):
        for after in (p for p in words if p > last_vision):
            swapped = list(ids)
            swapped[before], swapped[after] = ids[after], ids[before]
            text = llm.tokenizer.decode(swapped)
            if llm.tokenizer.encode(text, add_special_tokens=False) == swapped:
                return before, after
    pytest.skip("no stably re-tokenizing word swap in the prompt")

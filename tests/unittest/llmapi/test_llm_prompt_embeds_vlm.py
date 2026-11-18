# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""prompt_embeds on Qwen-VL models, alone and batched with image requests."""

import json
import os
import uuid
from typing import Iterator, List

import pytest
import torch
from PIL import Image
from safetensors import safe_open

from tensorrt_llm import LLM
from tensorrt_llm.inputs import default_multimodal_input_loader
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
        media=[image],
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


def test_rejects_prompt_embeds_with_image(llm: LLM, model_dir: str) -> None:
    table = _text_embedding_table(model_dir)
    image = _image_request(llm, model_dir)
    with pytest.raises(ValueError, match="cannot be combined"):
        llm.generate({**image, "prompt_embeds": table[:4]}, _greedy())

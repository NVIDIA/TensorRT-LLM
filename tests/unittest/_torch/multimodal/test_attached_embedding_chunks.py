# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Precomputed image embeddings larger than one prefill chunk, on Qwen3-VL-2B."""

import json
from collections.abc import Generator
from pathlib import Path

import pytest
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from utils.llm_data import llm_models_root

from tensorrt_llm import MultimodalEncoder
from tensorrt_llm._torch.pyexecutor.engine.multimodal import resolve_mm_encoder_token_budget
from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.inputs import default_multimodal_input_loader
from tensorrt_llm.inputs.registry import create_input_processor
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.llmapi.llm import LLM, SamplingParams

_MODEL_DIR = llm_models_root() / "Qwen3" / "Qwen3-VL-2B-Instruct"
_IMAGE = llm_models_root() / "multimodals" / "test_data" / "seashore.png"
_CHUNK_TOKENS = 1024


@pytest.fixture(scope="module")
def large_images(tmp_path_factory: pytest.TempPathFactory) -> list[str]:
    """Images of about 4K and 8K embedding rows."""
    out_dir = tmp_path_factory.mktemp("large_images")
    paths = []
    for index, size in enumerate([(2048, 2048), (4096, 2048)]):
        path = out_dir / f"image_{index}.png"
        Image.open(_IMAGE).convert("RGB").resize(size).save(path)
        paths.append(str(path))
    return paths


@pytest.fixture(scope="module")
def engines() -> Generator[tuple[LLM, LLM, MultimodalEncoder], None, None]:
    """An unchunked LLM, a chunked LLM with block reuse, and an encoder."""
    common = dict(model=_MODEL_DIR, trust_remote_code=True, max_batch_size=4)
    with (
        LLM(
            **common,
            kv_cache_config=KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.2),
            max_num_tokens=16384,
        ) as llm,
        LLM(
            **common,
            kv_cache_config=KvCacheConfig(enable_block_reuse=True, free_gpu_memory_fraction=0.2),
            enable_chunked_prefill=True,
            max_num_tokens=_CHUNK_TOKENS,
        ) as chunked_llm,
        MultimodalEncoder(model=_MODEL_DIR, max_batch_size=1) as encoder,
    ):
        yield llm, chunked_llm, encoder


def _load_inputs(llm: LLM, prompt: str, image: str) -> dict:
    model_type = json.loads((Path(llm._hf_model_dir) / "config.json").read_text())["model_type"]
    (inputs,) = default_multimodal_input_loader(
        tokenizer=llm.tokenizer,
        model_dir=llm._hf_model_dir,
        model_type=model_type,
        modality="image",
        prompts=[prompt],
        media=[image],
        image_data_format="pt",
    )
    return inputs


def _attached_inputs(encoder: MultimodalEncoder, raw_inputs: dict, image: str) -> dict:
    """The raw request with the encoder's output attached in place of the image."""
    (output,) = encoder.generate([raw_inputs])
    handles = output.disaggregated_params.multimodal_embedding_handles
    embedding = torch.cat([SharedTensorContainer.from_dict(h).get_local_view() for h in handles])
    (grid,) = (
        AutoProcessor.from_pretrained(_MODEL_DIR)
        .image_processor(images=[Image.open(image)], return_tensors="pt")["image_grid_thw"]
        .tolist()
    )
    return {
        "prompt": raw_inputs["prompt"],
        "multi_modal_embeddings": {"image": [{"mm_embeddings": embedding, "image_grid_thw": grid}]},
    }


def _old_attachment_bound_rows() -> int:
    """Rows the encoder-output budget held when attachments stayed resident whole."""
    processor = create_input_processor(str(_MODEL_DIR), AutoTokenizer.from_pretrained(_MODEL_DIR))
    atomic = max(processor.get_mm_max_tokens_per_item().values())
    return processor.get_max_mm_encoder_output_embeddings(
        resolve_mm_encoder_token_budget(_CHUNK_TOKENS, atomic)
    )


@pytest.mark.threadleak(enabled=False)
def test_chunked_long_attachment_matches_unchunked(
    engines: tuple[LLM, LLM, MultimodalEncoder], large_images: list[str]
) -> None:
    llm, chunked_llm, encoder = engines
    prompt = "Describe the image in detail."
    raw_inputs = _load_inputs(llm, prompt, large_images[0])
    inputs = _attached_inputs(encoder, raw_inputs, large_images[0])
    rows = inputs["multi_modal_embeddings"]["image"][0]["mm_embeddings"].shape[0]
    assert rows > 3 * _CHUNK_TOKENS
    params = SamplingParams(max_tokens=32)

    def tokens(engine: LLM, request: dict) -> list[int]:
        return engine.generate(request, params).outputs[0].token_ids

    expected = tokens(llm, raw_inputs)
    assert tokens(llm, inputs) == expected
    assert tokens(chunked_llm, raw_inputs) == expected
    # The first chunked run stores the prompt's blocks; the second starts
    # after the reused prefix, part way into the attachment.
    assert [tokens(chunked_llm, inputs) for _ in range(2)] == [expected, expected]


@pytest.mark.threadleak(enabled=False)
def test_sequences_above_the_old_attachment_bound_are_served(
    engines: tuple[LLM, LLM, MultimodalEncoder], large_images: list[str]
) -> None:
    _, chunked_llm, encoder = engines
    prompt = "Describe the image."
    raw_inputs = _load_inputs(chunked_llm, prompt, large_images[1])
    inputs = _attached_inputs(encoder, raw_inputs, large_images[1])
    rows = inputs["multi_modal_embeddings"]["image"][0]["mm_embeddings"].shape[0]
    num_sequences = 3
    assert num_sequences * rows > _old_attachment_bound_rows()

    output = chunked_llm.generate(
        inputs, SamplingParams(max_tokens=8, n=num_sequences, temperature=1.0, seed=0)
    )

    assert len(output.outputs) == num_sequences
    assert all(sequence.token_ids for sequence in output.outputs)

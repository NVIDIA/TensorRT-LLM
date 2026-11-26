# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Precomputed image embeddings larger than one prefill chunk, on Qwen3-VL-2B."""

import json
from collections.abc import Generator
from pathlib import Path

import pytest
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
_TEST_DATA = llm_models_root() / "multimodals" / "test_data"
_CHUNK_TOKENS = 1024
_IMAGE_TOKEN_ID = 151655
_COMMON = dict(model=_MODEL_DIR, trust_remote_code=True, max_batch_size=4)


@pytest.fixture(scope="module")
def images(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """Images of about 4K and 8K embedding rows."""
    out_dir = tmp_path_factory.mktemp("images")
    sources = {
        "large": ("seashore.png", (2048, 2048)),
        "larger": ("seashore.png", (4096, 2048)),
    }
    paths = {}
    for name, (source, size) in sources.items():
        paths[name] = str(out_dir / f"{name}.png")
        Image.open(_TEST_DATA / source).convert("RGB").resize(size).save(paths[name])
    return paths


@pytest.fixture(scope="module")
def engines() -> Generator[tuple[LLM, LLM, MultimodalEncoder], None, None]:
    """An unchunked LLM, a chunked LLM with block reuse, and an encoder."""
    with (
        LLM(
            **_COMMON,
            kv_cache_config=KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.2),
            max_num_tokens=16384,
        ) as llm,
        LLM(
            **_COMMON,
            kv_cache_config=KvCacheConfig(enable_block_reuse=True, free_gpu_memory_fraction=0.2),
            enable_chunked_prefill=True,
            max_num_tokens=_CHUNK_TOKENS,
        ) as chunked_llm,
        MultimodalEncoder(model=_MODEL_DIR, max_batch_size=1) as encoder,
    ):
        yield llm, chunked_llm, encoder


def _load_inputs(llm: LLM, prompt: str, images: list[str]) -> dict:
    model_type = json.loads((Path(llm._hf_model_dir) / "config.json").read_text())["model_type"]
    (inputs,) = default_multimodal_input_loader(
        tokenizer=llm.tokenizer,
        model_dir=llm._hf_model_dir,
        model_type=model_type,
        modality="multiple_image" if len(images) > 1 else "image",
        prompts=[prompt],
        media=[images] if len(images) > 1 else images,
        image_data_format="pt",
    )
    return inputs


def _attached_inputs(
    encoder: MultimodalEncoder, raw_inputs: dict, images: list[str], device: str
) -> dict:
    """The raw request with the encoder's outputs attached in place of its images."""
    (output,) = encoder.generate([raw_inputs])
    handles = output.disaggregated_params.multimodal_embedding_handles
    image_processor = AutoProcessor.from_pretrained(_MODEL_DIR).image_processor
    grids = image_processor(images=[Image.open(image) for image in images], return_tensors="pt")[
        "image_grid_thw"
    ].tolist()
    assert len(handles) == len(grids)
    attachments = [
        {
            # A tensor received over CUDA IPC cannot be shared again, so attach a copy.
            "mm_embeddings": SharedTensorContainer.from_dict(handle)
            .get_local_view()
            .to(device, copy=True),
            "image_grid_thw": grid,
        }
        for handle, grid in zip(handles, grids)
    ]
    return {"prompt": raw_inputs["prompt"], "multi_modal_embeddings": {"image": attachments}}


def _old_attachment_bound_rows() -> int:
    """Rows the encoder-output budget held when attachments stayed resident whole."""
    processor = create_input_processor(str(_MODEL_DIR), AutoTokenizer.from_pretrained(_MODEL_DIR))
    atomic = max(processor.get_mm_max_tokens_per_item().values())
    return processor.get_max_mm_encoder_output_embeddings(
        resolve_mm_encoder_token_budget(_CHUNK_TOKENS, atomic)
    )


@pytest.mark.threadleak(enabled=False)
def test_chunked_long_attachment_matches_unchunked(
    engines: tuple[LLM, LLM, MultimodalEncoder], images: dict[str, str]
) -> None:
    llm, chunked_llm, encoder = engines
    params = SamplingParams(max_tokens=32)
    media = [images["large"]]
    # The first attached request prefills cold, consuming every sliced row.
    # The second prompt shares the image but not the text after it, so it
    # reuses the KV cache the first one's chunks wrote. Reuse perturbs the
    # numerics, so the chunked reference is the raw image on the same engine,
    # run after the attached request so its blocks cannot warm it.
    for index, prompt in enumerate(
        ("Describe the image in detail.", "What colors are in this picture?")
    ):
        raw_inputs = _load_inputs(llm, prompt, media)
        # A host attachment reaches the GPU one chunk's rows at a time.
        inputs = _attached_inputs(encoder, raw_inputs, media, "cpu")
        rows = inputs["multi_modal_embeddings"]["image"][0]["mm_embeddings"].shape[0]
        assert rows > 3 * _CHUNK_TOKENS
        expected = llm.generate(raw_inputs, params)
        assert llm.generate(inputs, params).outputs[0].token_ids == expected.outputs[0].token_ids
        output = chunked_llm.generate(inputs, params)
        chunked_expected = chunked_llm.generate(raw_inputs, params)
        assert output.outputs[0].token_ids == chunked_expected.outputs[0].token_ids
        image_positions = [
            position
            for position, token in enumerate(expected.prompt_token_ids)
            if token == _IMAGE_TOKEN_ID
        ]
        assert len(image_positions) == rows
        if index == 0:
            assert output.cached_tokens < image_positions[0]
            assert output.outputs[0].token_ids == expected.outputs[0].token_ids
        else:
            assert output.cached_tokens > image_positions[-1]


@pytest.mark.threadleak(enabled=False)
def test_sequences_above_the_old_attachment_bound_are_served(
    engines: tuple[LLM, LLM, MultimodalEncoder], images: dict[str, str]
) -> None:
    _, chunked_llm, encoder = engines
    prompt = "Describe the image."
    raw_inputs = _load_inputs(chunked_llm, prompt, [images["larger"]])
    inputs = _attached_inputs(encoder, raw_inputs, [images["larger"]], "cuda")
    rows = inputs["multi_modal_embeddings"]["image"][0]["mm_embeddings"].shape[0]
    num_sequences = 3
    assert num_sequences * rows > _old_attachment_bound_rows()

    output = chunked_llm.generate(
        inputs, SamplingParams(max_tokens=8, n=num_sequences, temperature=1.0, seed=0)
    )

    assert len(output.outputs) == num_sequences
    assert all(sequence.token_ids for sequence in output.outputs)


@pytest.mark.threadleak(enabled=False)
def test_chunked_attachment_without_item_scheduling(
    engines: tuple[LLM, LLM, MultimodalEncoder], images: dict[str, str]
) -> None:
    """A prefill-only worker (no MM encoder, so no item scheduler) slices too."""
    llm, _, encoder = engines
    media = [images["large"]]
    params = SamplingParams(max_tokens=32)
    raw_inputs = _load_inputs(llm, "Describe the image in detail.", media)
    inputs = _attached_inputs(encoder, raw_inputs, media, "cpu")
    expected = llm.generate(raw_inputs, params)

    with LLM(
        **_COMMON,
        kv_cache_config=KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.1),
        enable_chunked_prefill=True,
        max_num_tokens=_CHUNK_TOKENS,
        disable_mm_encoder=True,
    ) as prefill_llm:
        output = prefill_llm.generate(inputs, params)

    assert output.outputs[0].token_ids == expected.outputs[0].token_ids

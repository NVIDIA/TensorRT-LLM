# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step-5-Preview's image input processor against its references.

* Without any file: the tiling of known sizes and the row count of each tile.
* With ``STEP5_PREVIEW_CHECKPOINT`` (the checkpoint directory) and
  ``STEP5_PREVIEW_VISION_ORACLE`` (the oracle safetensors, built by
  ``make_vision_oracle.py`` beside it): on ``merlion.png``, the processor's image span
  and pixel values equal the reference's exactly, from a PIL image and from the
  ``load_image(format="pt")`` tensor alike, its text tokens are the engine
  tokenizer's, and the multimodal hashing wrapper finds the image span. (The oracle's
  own text tokens came from a bare ``AutoTokenizer``, which drops this checkpoint's
  spaces, so only its image span is compared.)
* With ``STEP5_PREVIEW_CHECKPOINT`` and ``STEP3_VL_PROCESSOR_FILE`` (the family's
  ``processing_step3.py``): over sizes that take every branch of the tiling (no crop,
  small elongated, large, capped, thin padded, thin large), the tiles and pixels equal
  the family processor's.
* With the checkpoint: ``create_input_processor`` builds this processor for it, and the
  engine's memory profiler gets the largest tile set that fits its encoder budget.

The processor is built over the tokenizer the engine uses, ``tokenizer_factory``'s
(``tokenizer.json`` verbatim; ``test_modeling_v2_step5_tokenizer.py``).
The checkpoint is ``STEP5_PREVIEW_CHECKPOINT``, else ``$LLM_MODELS_ROOT/Step-5-Preview-NVFP4``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from tensorrt_llm._torch._experimental.modeling_v2.models.step5.step5_preview_nvfp4__sm_103__unified import (  # noqa: E501
    input_processor as ip,
)
from tensorrt_llm.inputs.registry import (
    MULTIMODAL_PLACEHOLDER_REGISTRY,
    create_input_processor,
    create_input_processor_with_hash,
    get_architecture_input_processor,
)
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.tokenizer import tokenizer_factory

pytestmark = pytest.mark.cpu_only

_MERLION = Path(__file__).resolve().parents[6] / "integration" / "test_input_files" / "merlion.png"


def _checkpoint_dir():
    explicit = os.environ.get("STEP5_PREVIEW_CHECKPOINT")
    if explicit:
        return explicit
    from utils.llm_data import llm_models_root

    root = llm_models_root()
    staged = None if root is None else Path(root) / "Step-5-Preview-NVFP4"
    return str(staged) if staged is not None and (staged / "config.json").is_file() else None


_CHECKPOINT = _checkpoint_dir()
_ORACLE = os.environ.get("STEP5_PREVIEW_VISION_ORACLE")
_FAMILY_PROCESSOR = os.environ.get("STEP3_VL_PROCESSOR_FILE")

needs_checkpoint = pytest.mark.skipif(
    not _CHECKPOINT,
    reason="set STEP5_PREVIEW_CHECKPOINT or stage Step-5-Preview-NVFP4 under LLM_MODELS_ROOT",
)
needs_oracle = pytest.mark.skipif(
    not (_CHECKPOINT and _ORACLE),
    reason="set STEP5_PREVIEW_CHECKPOINT and STEP5_PREVIEW_VISION_ORACLE",
)
needs_family_processor = pytest.mark.skipif(
    not (_CHECKPOINT and _FAMILY_PROCESSOR),
    reason="set STEP5_PREVIEW_CHECKPOINT and STEP3_VL_PROCESSOR_FILE",
)

OVERVIEW = 728


# --- without files -----------------------------------------------------------------


def test_tile_rows_follow_the_encoder_downsampling():
    assert ip._rows_after_downsampling(728, 14) == 169
    assert ip._rows_after_downsampling(504, 14) == 81


@pytest.mark.parametrize(
    "size, pad, scaled, canvas, boxes, row_ends",
    [
        # merlion.png: 3 x 2 windows of 504 on a canvas stretched from 1192 x 874.
        (
            (1192, 874),
            None,
            (1192, 874),
            (1512, 1008),
            [(x, y, x + 504, y + 504) for y in (0, 504) for x in (0, 504, 1008)],
            (False, False, True, False, False, False),
        ),
        # Small and near square: the overview alone.
        ((640, 480), None, (640, 480), (640, 480), [], ()),
        # Small and elongated: windows of the shorter side.
        (
            (700, 400),
            None,
            (700, 400),
            (800, 400),
            [(0, 0, 400, 400), (400, 0, 800, 400)],
            (False, False),
        ),
        # Large and thin: one row of windows of the shorter side; 4.33 windows round up.
        (
            (1300, 300),
            None,
            (1300, 300),
            (1500, 300),
            [(x, 0, x + 300, 300) for x in (0, 300, 600, 900, 1200)],
            (False,) * 5,
        ),
        # Thin and extreme: padded square first.
        (
            (1000, 20),
            1000,
            (1000, 1000),
            (1008, 1008),
            [(x, y, x + 504, y + 504) for y in (0, 504) for x in (0, 504)],
            (False, True, False, False),
        ),
    ],
)
def test_tile_plan(size, pad, scaled, canvas, boxes, row_ends):
    plan = ip.tile_plan(*size, OVERVIEW)
    assert plan.pad_side == pad
    assert plan.size == scaled
    assert plan.canvas == canvas
    assert list(plan.boxes) == boxes
    assert plan.row_ends == row_ends


def test_tile_plan_caps_the_longer_side():
    plan = ip.tile_plan(4000, 3000, OVERVIEW)
    assert plan.size == (3024, 2268)
    assert plan.canvas == (3024, 2520)
    assert len(plan.boxes) == 6 * 5
    assert sum(plan.row_ends) == 4


def test_the_architecture_reaches_this_processor():
    assert get_architecture_input_processor(ip.ARCHITECTURE) is ip.Step5PreviewInputProcessor
    assert MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholder(ip.MODEL_TYPE, "image") == "<im_patch>"


def test_serve_reads_no_media_io_preference():
    """``OpenAIServer.__init__`` calls ``get_preferred_media_io_kwargs()`` on every
    multimodal input processor; this one answers with an empty dict."""
    unbuilt = ip.Step5PreviewInputProcessor.__new__(ip.Step5PreviewInputProcessor)
    preference = unbuilt.get_preferred_media_io_kwargs()
    assert isinstance(preference, dict)
    assert preference == {}


# --- with the checkpoint -----------------------------------------------------------


def _checkpoint_config():
    """config.json as attributes, sub-configs as dicts: no config class needed."""
    return SimpleNamespace(**json.loads((Path(_CHECKPOINT) / "config.json").read_text()))


@pytest.fixture(scope="module")
def processor():
    return ip.Step5PreviewInputProcessor(
        _CHECKPOINT, _checkpoint_config(), tokenizer_factory(_CHECKPOINT)
    )


@pytest.fixture(scope="module")
def oracle():
    from safetensors import safe_open

    with safe_open(_ORACLE, framework="pt") as handle:
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        metadata = handle.metadata()
    assert metadata["image_sha256"] == hashlib.sha256(_MERLION.read_bytes()).hexdigest(), (
        f"{_MERLION} is not the image the oracle was built from (a git-lfs pointer?)"
    )
    return tensors, metadata


def _oracle_span(processor, oracle_ids: list) -> list:
    """The oracle prompt's image span: from its first ``<patch_start>`` to its last
    ``<im_end>``."""
    start = oracle_ids.index(processor._ids["patch_start"])
    end = len(oracle_ids) - 1 - oracle_ids[::-1].index(processor._ids["im_end"])
    return oracle_ids[start : end + 1]


def _expected_ids(processor, prompt: str, oracle_ids: list) -> list:
    """The engine tokenizer's ids for `prompt` with its one ``<im_patch>`` replaced by
    the oracle's image span."""
    text = processor.tokenizer.encode(prompt, add_special_tokens=True)
    at = text.index(processor._image_token_id)
    return text[:at] + _oracle_span(processor, oracle_ids) + text[at + 1 :]


def _merlion_pil():
    return Image.open(_MERLION).convert("RGB")


def _merlion_pt():
    from tensorrt_llm.inputs import load_image

    return load_image(str(_MERLION), format="pt")


@needs_oracle
@pytest.mark.parametrize("load", [_merlion_pil, _merlion_pt], ids=["pil", "pt"])
def test_merlion_matches_the_reference(processor, oracle, load):
    tensors, metadata = oracle
    inputs = {"prompt": metadata["prompt"], "multi_modal_data": {"image": [load()]}}
    token_ids, extra = processor(inputs, SamplingParams())

    assert token_ids == _expected_ids(processor, metadata["prompt"], tensors["input_ids"].tolist())
    image = extra["multimodal_data"]["image"]
    assert image["num_patches"] == tensors["num_patches"].tolist()
    assert image["pixel_values"].dtype == processor.dtype
    assert torch.equal(image["pixel_values"], tensors["pixel_values"].to(processor.dtype))
    assert torch.equal(
        image["patch_pixel_values"], tensors["patch_pixel_values"].to(processor.dtype)
    )

    overview, crops, plan = processor.tiles(load())
    assert torch.equal(overview, tensors["pixel_values"][0])
    assert torch.equal(crops, tensors["patch_pixel_values"])
    assert list(plan.row_ends) == tensors["patch_newline_mask"].tolist()
    # One row of vision features per placeholder, as the reference's merge requires.
    assert token_ids.count(int(metadata["image_token_id"])) == tensors["image_embeds"].shape[0]


@needs_oracle
def test_the_loader_feeds_the_oracle_inputs(processor, oracle):
    """The boot gate's path: ``default_multimodal_input_loader`` renders the oracle's
    prompt (one ``<im_patch>``, after the text) and hands over the image as a tensor."""
    from tensorrt_llm.inputs import default_multimodal_input_loader

    tensors, metadata = oracle
    inputs = default_multimodal_input_loader(
        tokenizer=processor.tokenizer,
        model_dir=_CHECKPOINT,
        model_type=ip.MODEL_TYPE,
        modality="image",
        prompts=[metadata["user_text"]],
        media=[str(_MERLION)],
        image_data_format="pt",
        device="cpu",
    )
    assert inputs[0]["prompt"] == metadata["prompt"]
    token_ids, extra = processor(inputs[0], SamplingParams())
    assert token_ids == _expected_ids(processor, metadata["prompt"], tensors["input_ids"].tolist())
    assert torch.equal(
        extra["multimodal_data"]["image"]["pixel_values"],
        tensors["pixel_values"].to(processor.dtype),
    )


@needs_oracle
def test_merlion_span_for_multimodal_hashing(processor, oracle):
    tensors, metadata = oracle
    inputs = {"prompt": metadata["prompt"], "multi_modal_data": {"image": [_merlion_pil()]}}
    token_ids, extra = create_input_processor_with_hash(processor)(inputs, SamplingParams())

    assert token_ids == _expected_ids(processor, metadata["prompt"], tensors["input_ids"].tolist())
    span = processor.get_num_tokens_per_image(image=_merlion_pil())
    assert span == processor.get_num_tokens_per_image(image=_merlion_pt())
    start = token_ids.index(processor._ids["patch_start"])
    assert token_ids[start + span - 1] == processor._ids["im_end"]
    mm_input = extra["multimodal_input"]
    assert mm_input.multimodal_positions == [start]
    assert mm_input.multimodal_lengths == [span]
    data = extra["multimodal_data"]
    assert data["multimodal_embedding_lengths"] == [tensors["image_embeds"].shape[0]]
    assert int(data["multimodal_embed_mask_cumsum"][-1]) == tensors["image_embeds"].shape[0]


@needs_checkpoint
def test_text_prompt_tokenizes_as_the_tokenizer_does(processor):
    prompt = "Question: what is 2 + 3?\nAnswer:"
    token_ids, extra = processor({"prompt": prompt}, SamplingParams())
    assert extra is None
    assert token_ids == processor.tokenizer.encode(prompt, add_special_tokens=True)


@needs_checkpoint
def test_placeholder_count_must_match_the_images(processor):
    inputs = {"prompt": "no placeholder here", "multi_modal_data": {"image": [_merlion_pil()]}}
    with pytest.raises(ValueError, match="image placeholders"):
        processor(inputs, SamplingParams())


# --- with the family's processor ---------------------------------------------------


def _family_module():
    spec = importlib.util.spec_from_file_location("_step3_family_processing", _FAMILY_PROCESSOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@needs_family_processor
@pytest.mark.parametrize(
    "size",
    [
        (1192, 874),
        (640, 480),
        (700, 400),
        (300, 1300),
        (2000, 300),
        (1000, 20),
        (504, 504),
        (1260, 600),
        (4000, 3000),
    ],
)
def test_tiles_match_the_family_processor(processor, size):
    family = _family_module()
    patcher = family.ImagePatcher()
    pixels = family.Step3VisionProcessor(OVERVIEW, "bilinear", 504)
    rng = np.random.default_rng(sum(size))
    image = Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8), mode="RGB")

    ref_image, ref_crops, ref_newlines = patcher(image)
    overview, crops, plan = processor.tiles(image)

    assert plan.size == ref_image.size
    assert len(plan.boxes) == len(ref_crops)
    assert list(plan.row_ends) == (ref_newlines or [])
    assert torch.equal(overview, pixels(ref_image)["pixel_values"][0])
    for mine, ref in zip(crops, ref_crops):
        assert torch.equal(mine, pixels(ref, is_patch=True)["pixel_values"][0])
    # The span length the hashing path reads equals the family's count.
    family_processor = family.Step3VLProcessor(tokenizer=processor.tokenizer.tokenizer)
    assert processor.get_num_tokens_per_image(image=image) == family_processor.get_num_image_tokens(
        *size
    )


# --- through the engine's lookup ---------------------------------------------------


@needs_checkpoint
def test_create_input_processor_builds_it():
    built = create_input_processor(_CHECKPOINT, None)
    assert isinstance(built, ip.Step5PreviewInputProcessor)
    assert built.dtype == torch.bfloat16
    assert built.get_vocab_size() == built.config.text_config.vocab_size
    span = built.get_num_tokens_per_image(image=_merlion_pil())
    assert span == 6 * (81 + 2) + 1 + 169 + 2
    assert built.get_preferred_media_io_kwargs() == {}
    # Built without a tokenizer, it loads the engine's: tokenizer.json verbatim.
    assert built.tokenizer.encode("The capital of France is") == [0, 671, 6102, 294, 8760, 344]


def test_the_tiling_bounds_no_image_size():
    """Why `get_mm_max_tokens_per_item()` states no startup maximum: a thin image is cut
    into windows of its shorter side, so the crop count grows as that side shrinks."""
    assert len(ip.tile_plan(3024, 32, OVERVIEW).boxes) == 95
    assert len(ip.tile_plan(4000, 32, OVERVIEW).boxes) == 126
    assert len(ip.tile_plan(100000, 40, OVERVIEW).boxes) == 3024


@needs_checkpoint
def test_profiling_gets_the_largest_tile_set_in_budget(processor):
    """The engine profiles the tower at its encoder budget (`max_num_tokens` by default):
    the processor answers with the largest image that fits and builds it."""
    overview, crop = 52 * 52, 36 * 36
    assert processor.get_mm_max_tokens_per_item() == {}
    assert processor.get_mm_max_tokens_per_item(8192) == {"image": overview + 4 * crop}
    assert processor.get_mm_max_tokens_per_item(overview - 1) == {}

    one = processor.get_dummy_mm_data(
        max_num_encoder_tokens=8192, mm_counts={"image": 1}, dtype=torch.bfloat16
    )["image"]
    assert one["num_patches"] == [4]
    assert one["pixel_values"].shape == (1, 3, 728, 728)
    assert one["patch_pixel_values"].shape == (4, 3, 504, 504)
    assert one["pixel_values"].dtype == torch.bfloat16

    two = processor.get_dummy_mm_data(max_num_encoder_tokens=8192, mm_counts={"image": 2})
    assert two["image"]["num_patches"] == [1, 1]
    bare = processor.get_dummy_mm_data(max_num_encoder_tokens=3000, mm_counts={"image": 1})
    assert bare["image"]["num_patches"] == [0] and "patch_pixel_values" not in bare["image"]
    assert processor.get_dummy_mm_data(max_num_encoder_tokens=8192, mm_counts={"image": 0}) == {}
    with pytest.raises(ValueError, match="do not fit"):
        processor.get_dummy_mm_data(max_num_encoder_tokens=8192, mm_counts={"image": 4})
    with pytest.raises(ValueError, match="image input only"):
        processor.get_dummy_mm_data(max_num_encoder_tokens=8192, mm_counts={"video": 1})

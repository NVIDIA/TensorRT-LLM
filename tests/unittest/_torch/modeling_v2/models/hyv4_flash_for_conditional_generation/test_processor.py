# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HY4-Flash processor: the checkpoint's own inputs, read on CPU.

``AutoProcessor.from_pretrained(<checkpoint>)`` must resolve the checkpoint's
``HYV4FlashProcessor`` (transformers 5.5.4 has no ``hy_v4_flash``), whose image
processor is the checkpoint's ``HYV4VLImageProcessor``: rescale + normalize with
mean/std 0.5, patch 16, 2x2 merge, min/max pixels 4096/4194304. ``inputs/utils.py``
must then build a text+media conversation whose image placeholder is the
checkpoint's start/pad/end token run.

Point ``HYV4_FLASH_CKPT`` at the checkpoint directory, or place it at
``LLM_MODELS_ROOT/Hy4-flash-FP8``.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.cpu_only

_CHECKPOINT_ENV = "HYV4_FLASH_CKPT"
_MODEL_DIR = "Hy4-flash-FP8"

# The checkpoint's own geometry, asserted below against processor_config.json.
_PATCH_SIZE = 16
_MERGE_SIZE = 2
_MIN_PIXELS = 4096
_MAX_PIXELS = 4194304


@pytest.fixture(scope="module")
def checkpoint_dir() -> str:
    """The checkpoint directory, from the environment or the models root."""
    override = os.environ.get(_CHECKPOINT_ENV)
    path = Path(override) if override else None
    if path is None:
        from utils.llm_data import llm_models_root

        root = llm_models_root()
        path = None if root is None else Path(root) / _MODEL_DIR
    if path is None or not (path / "processor_config.json").is_file():
        pytest.skip(
            f"set {_CHECKPOINT_ENV} to the HY4-Flash checkpoint directory, or "
            f"place it at LLM_MODELS_ROOT/{_MODEL_DIR}"
        )
    return str(path)


@pytest.fixture(scope="module")
def processor(checkpoint_dir):
    from transformers import AutoProcessor

    from tensorrt_llm.inputs.registry import MULTIMODAL_PLACEHOLDER_REGISTRY

    # The provider is registered lazily: a model-type lookup imports the module
    # through `MULTIMODAL_MODEL_TYPE_TO_MODULE`, and its import-time
    # `AutoProcessor.register` is what makes the checkpoint's
    # `processor_class` resolvable below.
    assert MULTIMODAL_PLACEHOLDER_REGISTRY.is_valid("hy_v4_flash", "image"), (
        "hy_v4_flash is not registered; is it missing from "
        "MULTIMODAL_MODEL_TYPE_TO_MODULE in _torch/models/_arch_index.py?"
    )
    return AutoProcessor.from_pretrained(checkpoint_dir)


@pytest.fixture(scope="module")
def checkpoint_tokens(checkpoint_dir) -> dict:
    with open(Path(checkpoint_dir) / "tokenizer_config.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def checkpoint_image_config(checkpoint_dir) -> dict:
    with open(Path(checkpoint_dir) / "processor_config.json", encoding="utf-8") as f:
        return json.load(f)["image_processor"]


def _expected_pixel_values(raw: torch.Tensor, image_processor) -> torch.Tensor:
    """Rescale/normalize math from the processor's own attributes.

    ``raw`` is a (3, H, W) uint8 tensor; the result is (3, H, W) float, the
    per-pixel value the image processor's fused rescale+normalize produces.
    """
    rescale = image_processor.rescale_factor
    mean = torch.tensor(list(image_processor.image_mean), dtype=torch.float32)
    std = torch.tensor(list(image_processor.image_std), dtype=torch.float32)
    return (raw.float() * rescale - mean.view(3, 1, 1)) / std.view(3, 1, 1)


def test_arch_index_registers_the_processor_on_demand(checkpoint_dir):
    """A fresh process reaches the processor through ``_arch_index`` alone.

    Nothing imports the modeling module eagerly: the registry's model-type
    lookup is what imports it, and its import-time ``AutoProcessor.register``
    is what makes the checkpoint's ``processor_class`` resolvable afterwards.
    """
    script = textwrap.dedent(
        f"""\
        import sys

        from tensorrt_llm.inputs.registry import MULTIMODAL_PLACEHOLDER_REGISTRY

        provider = "tensorrt_llm._torch.models.modeling_hyv4_flash"
        assert provider not in sys.modules, "provider imported before the lookup"
        assert MULTIMODAL_PLACEHOLDER_REGISTRY.is_valid("hy_v4_flash", "image")
        assert provider in sys.modules, "the arch index did not import the provider"

        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained({checkpoint_dir!r})
        assert type(processor).__name__ == "HYV4FlashProcessor", type(processor).__name__
        print("ON-DEMAND-OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        env=os.environ.copy(),
    )
    assert result.returncode == 0 and "ON-DEMAND-OK" in result.stdout, (
        f"on-demand registration failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_auto_processor_resolves_the_checkpoint_processor(checkpoint_dir, processor):
    """The registration makes AutoProcessor read the checkpoint's processor_class."""
    from transformers.models.auto.processing_auto import PROCESSOR_MAPPING

    from tensorrt_llm._torch.configs import HYV4FlashConfig
    from tensorrt_llm._torch.models.modeling_hyv4_flash import (
        HYV4FlashProcessor,
        HYV4VLImageProcessor,
        HYV4VLVideoProcessor,
    )

    assert PROCESSOR_MAPPING[HYV4FlashConfig] is HYV4FlashProcessor
    assert isinstance(processor, HYV4FlashProcessor)
    assert isinstance(processor.image_processor, HYV4VLImageProcessor)
    assert isinstance(processor.video_processor, HYV4VLVideoProcessor)
    assert "mm_token_type_ids" in processor.model_input_names


def test_image_processor_reads_the_checkpoint_config(checkpoint_image_config, processor):
    """Every image-processor value is the checkpoint's, not a library default."""
    image_processor = processor.image_processor

    assert image_processor.patch_size == checkpoint_image_config["patch_size"] == _PATCH_SIZE
    assert image_processor.merge_size == checkpoint_image_config["merge_size"] == _MERGE_SIZE
    assert (
        image_processor.temporal_patch_size == checkpoint_image_config["temporal_patch_size"] == 1
    )
    assert (
        list(image_processor.image_mean) == checkpoint_image_config["image_mean"] == [0.5, 0.5, 0.5]
    )
    assert (
        list(image_processor.image_std) == checkpoint_image_config["image_std"] == [0.5, 0.5, 0.5]
    )
    assert (
        image_processor.size["shortest_edge"]
        == checkpoint_image_config["min_pixels"]
        == _MIN_PIXELS
    )
    assert (
        image_processor.size["longest_edge"] == checkpoint_image_config["max_pixels"] == _MAX_PIXELS
    )
    assert image_processor.rescale_factor == pytest.approx(
        checkpoint_image_config["rescale_factor"]
    )
    assert image_processor.do_rescale and image_processor.do_normalize
    assert image_processor.do_resize and image_processor.do_convert_rgb


def test_placeholder_tokens_are_the_checkpoints(checkpoint_tokens, processor):
    """The pad/start/end/newline tokens resolve to the checkpoint's token ids."""
    tokenizer = processor.tokenizer

    assert processor.image_token == checkpoint_tokens["image_token"]
    assert processor.video_token == checkpoint_tokens["video_token"]
    assert processor.image_token_id == 120120
    assert processor.video_token_id == 120124
    for key, token_id in (
        ("image_start_token", 120118),
        ("image_end_token", 120119),
        ("image_newline_token", 120121),
        ("video_start_token", 120122),
        ("video_end_token", 120123),
    ):
        assert tokenizer.convert_tokens_to_ids(checkpoint_tokens[key]) == token_id


def test_placeholder_metadata_matches_the_checkpoint(checkpoint_tokens, processor):
    """The registered placeholder is the checkpoint's start+pad+end token run."""
    from tensorrt_llm._torch.models.modeling_hyv4_flash import PLACEHOLDER_METADATA
    from tensorrt_llm.inputs import ContentFormat, MultimodalPlaceholderPlacement
    from tensorrt_llm.inputs.registry import MULTIMODAL_PLACEHOLDER_REGISTRY

    image_placeholder = "".join(
        checkpoint_tokens[key] for key in ("image_start_token", "image_token", "image_end_token")
    )
    video_placeholder = "".join(
        checkpoint_tokens[key] for key in ("video_start_token", "video_token", "video_end_token")
    )
    assert PLACEHOLDER_METADATA.placeholder_map["image"] == image_placeholder
    assert PLACEHOLDER_METADATA.placeholder_map["video"] == video_placeholder
    assert (
        MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholder("hy_v4_flash", "image") == image_placeholder
    )
    assert (
        MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholder("hy_v4_flash", "video") == video_placeholder
    )
    assert MULTIMODAL_PLACEHOLDER_REGISTRY.get_content_format("hy_v4_flash") is ContentFormat.STRING
    assert (
        MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholder_placement("hy_v4_flash")
        is MultimodalPlaceholderPlacement.BEFORE_TEXT
    )


def test_conversation_carries_the_image_placeholder(
    checkpoint_dir, checkpoint_tokens, processor, tmp_path
):
    """``inputs/utils.py`` turns one image into text+media with one placeholder."""
    from tensorrt_llm.inputs.utils import default_multimodal_input_loader

    image_path = tmp_path / "synthetic.png"
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(image_path)

    prompt = "Describe the image."
    loaded = default_multimodal_input_loader(
        tokenizer=processor.tokenizer,
        model_dir=checkpoint_dir,
        model_type="hy_v4_flash",
        modality="image",
        prompts=[prompt],
        media=[str(image_path)],
    )[0]

    image_placeholder = "".join(
        checkpoint_tokens[key] for key in ("image_start_token", "image_token", "image_end_token")
    )
    assert loaded["chat_template_applied"] is True
    assert loaded["prompt"].count(image_placeholder) == 1
    # BEFORE_TEXT: the placeholder precedes the user text.
    assert loaded["prompt"].index(image_placeholder) < loaded["prompt"].index(prompt)
    assert loaded["mm_item_order"] == [
        {"modality": "image", "index": 0, "placeholder": image_placeholder}
    ]
    images = loaded["multi_modal_data"]["image"]
    assert len(images) == 1
    assert isinstance(images[0], torch.Tensor) and images[0].shape == (3, 32, 32)

    # One pad token, one start and one end; the input processor expands the pad
    # run to `replace_image_token`'s count from `image_grid_thw`.
    ids = processor.tokenizer.encode(loaded["prompt"])
    assert ids.count(120120) == 1
    assert ids.count(120118) == 1
    assert ids.count(120119) == 1


def test_image_pixels_have_the_patch_count_and_order(processor):
    """Patch count and order follow patch 16 with a 2x2 merge, row-major."""
    image_processor = processor.image_processor
    # 64x96 is already aligned to patch * merge and inside the pixel budget, so
    # the resize is a no-op and the raw pixel values reach the patch layout.
    height, width = 64, 96
    raw = (
        (torch.arange(3 * height * width, dtype=torch.int64) % 251)
        .to(torch.uint8)
        .reshape(3, height, width)
    )

    out = image_processor(images=raw, return_tensors="pt")
    pixel_values = out["pixel_values"]
    grid_h, grid_w = height // _PATCH_SIZE, width // _PATCH_SIZE
    num_patches = grid_h * grid_w
    assert out["image_grid_thw"].tolist() == [[1, grid_h, grid_w]]
    assert pixel_values.shape == (
        num_patches,
        3 * _PATCH_SIZE * _PATCH_SIZE,
    )

    # Independent expectation: patch row-major over the merged grid -- merged
    # block row, merge row inside it, merged block column, merge column -- then
    # channel-major pixels inside a patch. `patchify`'s permute keeps the merge
    # row inside the block, not the block's patches as a 2x2 unit.
    expected = _expected_pixel_values(raw, image_processor)
    patches = []
    for block_h in range(grid_h // _MERGE_SIZE):
        for merge_h in range(_MERGE_SIZE):
            for block_w in range(grid_w // _MERGE_SIZE):
                for merge_w in range(_MERGE_SIZE):
                    top = (block_h * _MERGE_SIZE + merge_h) * _PATCH_SIZE
                    left = (block_w * _MERGE_SIZE + merge_w) * _PATCH_SIZE
                    patches.append(
                        expected[:, top : top + _PATCH_SIZE, left : left + _PATCH_SIZE].reshape(-1)
                    )
    torch.testing.assert_close(pixel_values, torch.stack(patches), rtol=1e-5, atol=1e-5)


def test_image_pixel_values_are_rescaled_and_normalized(processor):
    """A constant image becomes (value * rescale - mean) / std in every pixel."""
    image_processor = processor.image_processor
    channel_values = torch.tensor([0, 51, 204], dtype=torch.uint8)
    raw = channel_values.view(3, 1, 1).expand(3, 64, 96).contiguous()

    out = image_processor(images=raw, return_tensors="pt")
    pixel_values = out["pixel_values"]
    assert pixel_values.shape == (4 * 6, 3 * _PATCH_SIZE * _PATCH_SIZE)

    expected = _expected_pixel_values(raw, image_processor)[:, 0, 0]  # (3,)
    per_patch = expected.repeat_interleave(_PATCH_SIZE * _PATCH_SIZE)  # (768,)
    torch.testing.assert_close(
        pixel_values, per_patch.expand_as(pixel_values), rtol=1e-5, atol=1e-5
    )


def test_placeholder_counts_follow_the_merged_grid(processor):
    """A 64x96 image needs 2 x (3 + 1) pad slots; a 4-frame video 4 x that."""
    # The image processor hands the model a tensor grid; the documented list
    # form works too.
    image_out = processor.image_processor(
        images=torch.zeros(3, 64, 96, dtype=torch.uint8), return_tensors="pt"
    )
    assert processor.replace_image_token(image_out, 0) == processor.image_token * 8
    assert (
        processor.replace_image_token({"image_grid_thw": [[1, 4, 6]]}, 0)
        == processor.image_token * 8
    )

    video_inputs = {"video_grid_thw": torch.tensor([[4, 4, 6]])}
    assert processor.replace_video_token(video_inputs, 0) == processor.video_token * (4 * 8)

    # The same geometry, derived by the video processor's own budget math.
    num_patches, grid_t, grid_h, grid_w = processor.video_processor.get_num_of_video_patches(
        4, 64, 96
    )
    assert (grid_t, grid_h, grid_w) == (4, 4, 6)
    assert num_patches == grid_t * grid_h * grid_w

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from tensorrt_llm._torch.configs.nemotron_shortcut import NemotronShortcutConfig
from tensorrt_llm._torch.modeling_v2.models.nemotron_shortcut.input_processor import (
    ShortcutInputProcessor,
)
from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config
from tensorrt_llm.inputs.model_config import resolve_top_level_model_type
from tensorrt_llm.inputs.multimodal import MULTIMODAL_ENCODER_ITEM_METADATA_KEY
from tensorrt_llm.inputs.registry import create_input_processor, create_input_processor_with_hash
from tensorrt_llm.inputs.utils import retrieve_multimodal_placeholder
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.tokenizer import TransformersTokenizer

pytestmark = pytest.mark.cpu_only


def test_video_retention_preserves_timestamps_and_empty_rows():
    from tensorrt_llm._torch.modeling_v2.models.nemotron_shortcut.vision import fuse_video_template

    # Only the bottom right patch changes in frame two. The first frame survives intact.
    projected = torch.tensor([[1.0, 0.0]] * 7 + [[-1.0, 0.0]])
    text = torch.arange(18, dtype=torch.float32).reshape(9, 2)
    patch, row_end, frame_end, timestamp = 1, 2, 3, 4
    frame = [patch, patch, row_end, patch, patch, frame_end]
    result = fuse_video_template(
        projected, (2, 4, 4), 2, 0.375, [*frame, timestamp, *frame], text.__getitem__
    )
    expected = torch.stack(
        [
            projected[0],
            projected[1],
            text[row_end],
            projected[2],
            projected[3],
            text[frame_end],
            text[timestamp],
            text[row_end],
            projected[7],
            text[frame_end],
        ]
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.fixture
def processor(tmp_path: Path) -> ShortcutInputProcessor:
    tags = ["<nemo:img>", "<row>", "<end>"]
    vocab = ["<unk>", *tags]
    backend = Tokenizer(
        models.WordLevel({token: i for i, token in enumerate(vocab)}, unk_token="<unk>")
    )
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", extra_special_tokens=tags
    )
    spec = {
        "image_token": tags[0],
        "image_break_token": tags[1],
        "image_end_token": tags[2],
        "patch_size": 2,
        "spatial_merge_size": 2,
        "image_processor": {
            "patch_size": {"width": 4, "height": 4},
            "size": {"longest_edge": 8},
            "do_resize": True,
            "do_rescale": True,
            "do_normalize": True,
            "rescale_factor": 1 / 255,
            "image_mean": [0, 0, 0],
            "image_std": [1, 1, 1],
        },
    }
    (tmp_path / "processor_config.json").write_text(json.dumps(spec))
    config = NemotronShortcutConfig(dtype="bfloat16", image_token_id=1)
    return ShortcutInputProcessor(str(tmp_path), config, TransformersTokenizer(tokenizer))


def test_image_grid_and_rgb_compositing(processor: ShortcutInputProcessor) -> None:
    transparent = Image.new("RGBA", (3, 7), (255, 0, 0, 0))
    opaque = Image.new("RGB", (8, 4), (0, 0, 0))
    ids, extra = processor(
        {"prompt": "<nemo:img><nemo:img>", "multi_modal_data": {"image": [transparent, opaque]}},
        SamplingParams(),
    )
    assert ids == [1, 2, 1, 3, 1, 1, 3]
    data = extra["multimodal_data"]
    metadata = data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY]
    assert metadata.item_refs == (("image", 0), ("image", 1))
    assert metadata.encoder_token_lengths == (8, 8) and metadata.output_embedding_lengths == (2, 2)
    assert data["image"]["image_sizes"] == [(8, 4), (4, 8)]
    patches = data["image"]["patches"]
    assert patches.dtype == torch.bfloat16 and patches.shape == (16, 12)
    torch.testing.assert_close(patches[:8], torch.ones_like(patches[:8]))
    assert patches[8:].count_nonzero() == 0
    with pytest.raises(ValueError, match="Unsupported Shortcut media"):
        processor({"prompt": "", "multi_modal_data": {"video": []}}, SamplingParams())


def test_patch_order_follows_learned_merge(processor: ShortcutInputProcessor) -> None:
    image = torch.arange(64, dtype=torch.float32).view(1, 8, 8) / 64
    patches, size = processor.preprocess_image(image)
    assert size == (8, 8)
    # Top-left pixels of the four patches in each tile, in raster tile order.
    expected = [0, 2, 16, 18, 4, 6, 20, 22, 32, 34, 48, 50, 36, 38, 52, 54]
    assert (patches[:, 0] * 64).tolist() == expected
    assert patches[0, :4].tolist() == [0, 1 / 64, 8 / 64, 9 / 64]


def test_video_frame_limit(processor: ShortcutInputProcessor) -> None:
    frames = [Image.new("RGB", (4, 4))] * 65
    with pytest.raises(ValueError, match="at most 64 frames"):
        processor.get_num_tokens_per_video(video=frames, video_metadata={"fps": 2})


def test_chat_placeholder_and_disabled_encoder(processor: ShortcutInputProcessor) -> None:
    path = Path(processor.model_path)
    config = NemotronShortcutConfig(vision_config={}, dtype="bfloat16", image_token_id=1)
    config.save_pretrained(path)
    chat_config = load_pretrained_config(str(path))
    enabled = create_input_processor(str(path), processor.tokenizer)
    assert isinstance(enabled, ShortcutInputProcessor)
    model_type = resolve_top_level_model_type(chat_config)
    assert retrieve_multimodal_placeholder(model_type, "image", 1) == "<nemo:img>"

    (path / "processor_config.json").unlink()
    disabled = create_input_processor(str(path), processor.tokenizer, disable_mm_encoder=True)
    process = create_input_processor_with_hash(disabled)
    assert process({"prompt": "<end>"}, SamplingParams(add_special_tokens=False)) == ([3], None)
    image = Image.new("RGB", (4, 4))
    for prompt in ({"prompt": "<nemo:img>"}, {"prompt_token_ids": [1]}):
        with pytest.raises(ValueError, match="disable_mm_encoder=True"):
            process({**prompt, "multi_modal_data": {"image": image}}, SamplingParams())
    assert process({"prompt_token_ids": [0], "multi_modal_data": {}}, SamplingParams()) == (
        [0],
        None,
    )

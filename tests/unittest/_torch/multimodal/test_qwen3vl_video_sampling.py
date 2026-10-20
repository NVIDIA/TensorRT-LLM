# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exercise the input adapter with real HF video processing on CPU.

Load just the adapter definitions so these tests do not require TensorRT,
CUDA, model weights, or the model module's GPU kernel imports.
"""

import ast
import copy
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen3_vl.processing_qwen3_vl import (
    Qwen3VLProcessor,
    Qwen3VLProcessorKwargs,
)
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
from transformers.video_utils import VideoMetadata

pytestmark = pytest.mark.cpu_only


@pytest.fixture(scope="module")
def adapter() -> SimpleNamespace:
    source = Path(__file__).resolve().parents[4] / "tensorrt_llm/_torch/models/modeling_qwen3vl.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    decide = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_decide_do_sample_frames"
    )
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "Qwen3VLInputProcessorBase"
    )
    preprocess = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_preprocess"
    )
    namespace = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "math": math,
        "np": np,
        "torch": torch,
        "VideoMetadata": VideoMetadata,
        "Qwen3VLProcessorKwargs": Qwen3VLProcessorKwargs,
    }
    exec(
        compile(ast.Module(body=[decide, preprocess], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return SimpleNamespace(
        preprocess=namespace["_preprocess"], decide=namespace["_decide_do_sample_frames"]
    )


@pytest.fixture()
def processor() -> Qwen3VLProcessor:
    special = [
        "[UNK]",
        "[PAD]",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
    ]
    backend = Tokenizer(WordLevel({s: i for i, s in enumerate(special)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=special[2:],
    )
    return Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(),
        video_processor=Qwen3VLVideoProcessor(size={"shortest_edge": 2048, "longest_edge": 4096}),
        tokenizer=tokenizer,
    )


def _video(indices: list[int], fps: float = 24.0, total: int = 240) -> SimpleNamespace:
    frames = torch.arange(len(indices), dtype=torch.float32)[:, None, None, None].expand(
        -1, 3, 32, 32
    )
    return SimpleNamespace(
        frames=list(frames),
        metadata={
            "fps": fps,
            "total_num_frames": total,
            "duration": total / fps,
            "frames_indices": indices,
        },
    )


def _run(adapter: SimpleNamespace, processor: Qwen3VLProcessor, videos: list, **kwargs) -> dict:
    return adapter.preprocess(
        SimpleNamespace(processor=processor),
        "<|video_pad|>" * len(videos),
        {"video": videos},
        {"return_metadata": True, "do_normalize": False, **kwargs},
    )


@pytest.mark.parametrize("fps,total", [(24.0, 240), (30.0, 900), (7.5, 75)])
def test_presampled_default_preserves_frames_and_source_timestamps(adapter, processor, fps, total):
    indices = np.linspace(0, total - 1, 40).astype(int).tolist()
    video = _video(indices, fps, total)
    original = copy.deepcopy(video.metadata)
    result = _run(adapter, processor, [video])
    assert result["video_grid_thw"][0, 0] == 20
    metadata = result["video_metadata"][0]
    assert list(metadata.frames_indices) == indices
    assert metadata.fps == fps
    assert processor._calculate_timestamps(metadata.frames_indices, fps) == pytest.approx(
        [(indices[i] + indices[i + 1]) / (2 * fps) for i in range(0, 40, 2)]
    )
    assert video.metadata == original


@pytest.mark.parametrize(
    "kwargs,count", [({"num_frames": 8}, 8), ({"fps": 2}, 20), ({"do_sample_frames": True}, 20)]
)
def test_explicit_sampling_maps_nonuniform_source_indices(adapter, processor, kwargs, count):
    indices = [int(239 * (i / 39) ** 2) for i in range(40)]
    video = _video(indices)
    original = copy.deepcopy(video.metadata)
    result = _run(adapter, processor, [video], **kwargs)
    selected = np.linspace(0, 39, count).round().astype(int)
    expected = [indices[i] for i in selected]
    metadata = result["video_metadata"][0]
    assert result["video_grid_thw"][0, 0] == count // 2
    assert list(metadata.frames_indices) == expected
    assert metadata.fps == 24.0
    assert processor._calculate_timestamps(metadata.frames_indices, metadata.fps) == pytest.approx(
        [(expected[i] + expected[i + 1]) / 48 for i in range(0, count, 2)]
    )
    # Pixel values encode decoded positions: ensure metadata describes the frames actually selected.
    patches = result["pixel_values_videos"].reshape(count // 2, -1, 3, 2, 16, 16)
    assert patches[:, 0, 0, :, 0, 0].flatten().tolist() == selected.tolist()
    assert video.metadata == original


def test_explicit_disable_overrides_frame_target(adapter, processor):
    video = _video(np.linspace(0, 239, 40).astype(int).tolist())
    result = _run(adapter, processor, [video], do_sample_frames=False, num_frames=8)
    assert result["video_grid_thw"][0, 0] == 20
    assert list(result["video_metadata"][0].frames_indices) == video.metadata["frames_indices"]


def test_mixed_batch_keeps_presampled_video_and_samples_full_video(adapter, processor):
    sampled = _video(np.linspace(0, 239, 40).astype(int).tolist())
    full = _video(list(range(48)), total=48)
    result = _run(adapter, processor, [sampled, full])
    assert result["video_grid_thw"][:, 0].tolist() == [20, 2]
    assert list(result["video_metadata"][0].frames_indices) == sampled.metadata["frames_indices"]
    assert list(result["video_metadata"][1].frames_indices) == [0, 16, 31, 47]


def test_full_video_retains_hf_default(adapter, processor):
    full = _video(list(range(48)), total=48)
    result = _run(adapter, processor, [full])
    reference = processor(
        text=["<|video_pad|>"],
        videos=[full.frames],
        video_metadata=[copy.deepcopy(full.metadata)],
        do_sample_frames=True,
        do_rescale=False,
        do_normalize=False,
        return_tensors="pt",
        return_metadata=True,
    )
    assert torch.equal(result["video_grid_thw"], reference["video_grid_thw"])
    assert torch.equal(result["pixel_values_videos"], reference["pixel_values_videos"])
    assert torch.equal(result["input_ids"], reference["input_ids"])


def test_metadata_without_source_indices_uses_decoded_offsets(adapter, processor):
    video = _video(list(range(48)), total=48)
    del video.metadata["frames_indices"]
    result = _run(adapter, processor, [video])
    assert list(result["video_metadata"][0].frames_indices) == [0, 16, 31, 47]


def test_malformed_source_indices_fail_before_sampling(adapter, processor):
    video = _video(list(range(40)))
    video.metadata["frames_indices"] = [0, 239]
    with pytest.raises(ValueError, match="frames_indices"):
        _run(adapter, processor, [video], num_frames=8)


@pytest.mark.parametrize("format", ["numpy", "pil"])
def test_uint8_frames_preserve_sampling_and_rescale(adapter, processor, format):
    video = _video([0, 2, 10, 30, 60, 90, 160, 239])
    pixels = np.full((8, 32, 32, 3), 128, dtype=np.uint8)
    video.frames = pixels if format == "numpy" else [Image.fromarray(frame) for frame in pixels]
    result = _run(adapter, processor, [video], num_frames=4)
    assert list(result["video_metadata"][0].frames_indices) == [0, 10, 90, 239]
    assert result["pixel_values_videos"].mean().item() == pytest.approx(128 / 255)


def test_odd_frame_count_padding_does_not_mutate_source_metadata(adapter, processor):
    video = _video([0, 1, 3, 8, 50, 100, 239])
    original = copy.deepcopy(video.metadata)
    result = _run(adapter, processor, [video])
    assert result["video_grid_thw"][0, 0] == 4
    assert processor._calculate_timestamps(result["video_metadata"][0].frames_indices, 24)[
        -1
    ] == pytest.approx(239 / 24)
    assert video.metadata == original


def test_no_source_mapping_retains_legacy_sampling_duration(adapter, processor):
    video = _video(list(range(40)))
    del video.metadata["frames_indices"]
    result = _run(adapter, processor, [video])
    assert result["video_grid_thw"][0, 0] == 2
    assert list(result["video_metadata"][0].frames_indices) == [0, 13, 26, 39]


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, [0.5, 2.5, 4.5, 6.5]),
        ({"num_frames": 4}, [1.0, 6.0]),
        ({"do_sample_frames": False}, [0.5, 2.5, 4.5, 6.5]),
    ],
)
def test_prompt_contains_source_timestamps(adapter, processor, monkeypatch, kwargs, expected):
    prompts = []
    tokenize = type(processor.tokenizer).__call__

    def capture_prompt(tokenizer, text=None, **tokenizer_kwargs):
        prompts.extend(text)
        return tokenize(tokenizer, text=text, **tokenizer_kwargs)

    monkeypatch.setattr(type(processor.tokenizer), "__call__", capture_prompt)
    _run(adapter, processor, [_video(list(range(0, 192, 24)))], **kwargs)
    timestamps = re.findall(r"<([0-9.]+) seconds>", prompts[0])
    assert [float(timestamp) for timestamp in timestamps] == pytest.approx(expected)


def test_explicit_fps_uses_duration_without_source_frame_count(adapter, processor):
    video = _video(np.linspace(0, 239, 40).astype(int).tolist())
    del video.metadata["total_num_frames"]
    result = _run(adapter, processor, [video], fps=2)
    assert result["video_grid_thw"][0, 0] == 10
    selected = np.linspace(0, 39, 20).round().astype(int)
    assert list(result["video_metadata"][0].frames_indices) == [
        video.metadata["frames_indices"][i] for i in selected
    ]
    assert result["video_metadata"][0].fps == 24.0
    assert result["video_metadata"][0].total_num_frames == 240


@pytest.mark.parametrize("kwargs", [{"num_frames": 40}, {"fps": 4}])
def test_matching_target_preserves_all_source_frames(adapter, processor, kwargs):
    video = _video(np.linspace(0, 239, 40).astype(int).tolist())
    result = _run(adapter, processor, [video], **kwargs)
    assert result["video_grid_thw"][0, 0] == 20
    assert list(result["video_metadata"][0].frames_indices) == video.metadata["frames_indices"]


@pytest.mark.parametrize("kwargs", [{}, {"num_frames": 8}])
def test_preprocessing_does_not_mutate_request_or_frames(adapter, processor, kwargs):
    video = _video(np.linspace(0, 239, 40).astype(int).tolist())
    request = {"return_metadata": True, "do_normalize": False, **kwargs}
    original_kwargs = copy.deepcopy(request)
    original_frames = torch.stack(video.frames).clone()
    adapter.preprocess(
        SimpleNamespace(processor=processor), "<|video_pad|>", {"video": [video]}, request
    )
    assert request == original_kwargs
    assert torch.equal(torch.stack(video.frames), original_frames)


@pytest.mark.parametrize("target", [{"fps": 4}, {"num_frames": 8, "fps": None}])
def test_nested_sampling_preserves_pixels_and_source_timestamps(adapter, processor, target):
    video = _video(list(range(48)), total=48)
    request = {
        "return_metadata": True,
        "videos_kwargs": {"do_normalize": False, **target},
    }
    original = copy.deepcopy(request)
    result = adapter.preprocess(
        SimpleNamespace(processor=processor), "<|video_pad|>", {"video": [video]}, request
    )
    expected = np.linspace(0, 47, 8).round().astype(int).tolist()
    metadata = result["video_metadata"][0]
    assert result["video_grid_thw"][0, 0].item() == 4
    assert list(metadata.frames_indices) == expected
    assert processor._calculate_timestamps(metadata.frames_indices, metadata.fps) == pytest.approx(
        [(expected[i] + expected[i + 1]) / 48 for i in range(0, 8, 2)]
    )
    patches = result["pixel_values_videos"].reshape(4, -1, 3, 2, 16, 16)
    assert patches[:, 0, 0, :, 0, 0].flatten().tolist() == pytest.approx(expected)
    assert request == original


@pytest.mark.parametrize(
    "kwargs,count",
    [
        ({}, 40),
        ({"fps": 2}, 20),
        ({"num_frames": 8}, 8),
        ({"do_sample_frames": True}, 20),
        ({"do_sample_frames": False, "num_frames": 8}, 40),
    ],
)
def test_nested_sampling_controls_presampled_video(adapter, processor, kwargs, count):
    indices = np.linspace(0, 239, 40).astype(int).tolist()
    video = _video(indices)
    result = _run(adapter, processor, [video], videos_kwargs=kwargs)
    selected = np.linspace(0, 39, count).round().astype(int)
    assert result["video_grid_thw"][0, 0].item() == count // 2
    assert list(result["video_metadata"][0].frames_indices) == [indices[i] for i in selected]


@pytest.mark.parametrize(
    "config",
    [{"num_frames": 8, "fps": None}, {"fps": 4}, {"do_sample_frames": False}],
)
def test_video_processor_sampling_defaults_match_hf(adapter, processor, config):
    processor.video_processor = Qwen3VLVideoProcessor(
        size={"shortest_edge": 2048, "longest_edge": 4096}, **config
    )
    video = _video(list(range(48)), total=48)
    reference = processor(
        text=["<|video_pad|>"],
        videos=[video.frames],
        video_metadata=[copy.deepcopy(video.metadata)],
        do_rescale=False,
        do_normalize=False,
        return_tensors="pt",
        return_metadata=True,
    )
    result = _run(adapter, processor, [video])
    for key in ("video_grid_thw", "pixel_values_videos", "input_ids"):
        assert torch.equal(result[key], reference[key])


def test_request_can_enable_sampling_disabled_in_processor(adapter, processor):
    processor.video_processor.do_sample_frames = False
    video = _video(list(range(48)), total=48)
    result = _run(adapter, processor, [video], do_sample_frames=True, num_frames=8)
    assert result["video_grid_thw"][0, 0].item() == 4


@pytest.mark.parametrize("presampled", [False, True])
@pytest.mark.parametrize("kwargs", [{}, {"num_frames": 8}])
def test_sampling_preserves_source_total_duration_and_effective_fps(
    adapter, processor, presampled, kwargs
):
    indices = np.linspace(0, 239, 40).astype(int).tolist() if presampled else list(range(48))
    video = _video(indices, total=240 if presampled else 48)
    result = _run(adapter, processor, [video], **kwargs)
    metadata = result["video_metadata"][0]
    assert metadata.total_num_frames == video.metadata["total_num_frames"]
    assert metadata.duration == video.metadata["duration"]
    assert metadata.sampled_fps == pytest.approx(
        len(metadata.frames_indices) / video.metadata["duration"]
    )
    assert max(metadata.frames_indices) < metadata.total_num_frames


@pytest.mark.parametrize("with_image", [False, True])
def test_non_video_processing_matches_hf(adapter, processor, with_image):
    processor.video_processor = None
    images = [torch.ones(3, 32, 32)] if with_image else None
    text = "<|image_pad|>" if with_image else "text"
    result = adapter.preprocess(
        SimpleNamespace(processor=processor), text, {"image": images}, {"do_normalize": False}
    )
    reference = processor(
        text=[text],
        images=images,
        padding=True,
        return_tensors="pt",
        do_rescale=not with_image,
        do_normalize=False,
    )
    assert result.keys() == reference.keys()
    for key in result:
        assert torch.equal(result[key], reference[key])


@pytest.mark.parametrize("kwargs", [{}, {"return_metadata": False}, {"return_metadata": True}])
def test_metadata_output_filter_preserves_prompt_timestamps(
    adapter, processor, monkeypatch, kwargs
):
    prompts = []
    tokenize = type(processor.tokenizer).__call__

    def capture_prompt(tokenizer, text=None, **tokenizer_kwargs):
        prompts.extend(text)
        return tokenize(tokenizer, text=text, **tokenizer_kwargs)

    monkeypatch.setattr(type(processor.tokenizer), "__call__", capture_prompt)
    video = _video(list(range(0, 192, 24)))
    result = adapter.preprocess(
        SimpleNamespace(processor=processor), "<|video_pad|>", {"video": [video]}, kwargs
    )
    assert result["video_grid_thw"][0, 0].item() == 4
    assert [float(t) for t in re.findall(r"<([0-9.]+) seconds>", prompts[0])] == pytest.approx(
        [0.5, 2.5, 4.5, 6.5]
    )
    assert ("video_metadata" in result) == kwargs.get("return_metadata", False)


@pytest.mark.parametrize("format", ["numpy", "pil"])
def test_nested_uint8_video_is_rescaled_once(adapter, processor, format):
    video = _video([0, 2, 10, 30, 60, 90, 160, 239])
    pixels = np.full((8, 32, 32, 3), 128, dtype=np.uint8)
    video.frames = pixels if format == "numpy" else [Image.fromarray(frame) for frame in pixels]
    result = adapter.preprocess(
        SimpleNamespace(processor=processor),
        "<|video_pad|>",
        {"video": [video]},
        {"videos_kwargs": {"num_frames": 4, "do_normalize": False}},
    )
    assert result["pixel_values_videos"].mean().item() == pytest.approx(128 / 255)


def test_nested_image_and_video_preserve_float_pixel_values(adapter, processor):
    images = [torch.ones(3, 32, 32)]
    video = _video(list(range(0, 192, 24)))
    result = adapter.preprocess(
        SimpleNamespace(processor=processor),
        "<|image_pad|><|video_pad|>",
        {"image": images, "video": [video]},
        {
            "images_kwargs": {"do_normalize": False},
            "videos_kwargs": {"num_frames": 4, "do_normalize": False},
        },
    )
    assert result["pixel_values"].mean().item() == pytest.approx(1.0)
    patches = result["pixel_values_videos"].reshape(2, -1, 3, 2, 16, 16)
    assert patches[:, 0, 0, :, 0, 0].flatten().tolist() == pytest.approx([0, 2, 5, 7])

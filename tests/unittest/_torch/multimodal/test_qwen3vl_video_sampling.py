# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exercise the production input adapter with real HF video processing on CPU."""

import copy
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLInputProcessorBase

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("n,target", [(10, 32), (40, 100)])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("explicit", [None, True, False])
def test_oversized_frame_target_keeps_available_frames(
    adapter, processor, n, target, nested, explicit
):
    """An upper frame budget must not cause duplicate encoder work."""
    video = _video(np.linspace(0, 239, n).astype(int).tolist())
    controls = {"num_frames": target}
    if explicit is not None:
        controls["do_sample_frames"] = explicit
    kwargs = {"videos_kwargs": controls} if nested else controls
    result = _run(adapter, processor, [video], **kwargs)
    reference = _run(adapter, processor, [video], do_sample_frames=False)
    for key in ("input_ids", "pixel_values_videos", "video_grid_thw"):
        assert torch.equal(result[key], reference[key])
    assert list(result["video_metadata"][0].frames_indices) == video.metadata["frames_indices"]
    assert (
        adapter.get_num_tokens_per_video(
            video=video.frames, video_metadata=video.metadata, **kwargs
        )
        == int(result["video_grid_thw"][0].prod()) // 4
    )


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("provenance,count", [(False, 46), (True, 699), (None, 699)])
def test_interior_drop_below_cap_uses_loader_provenance(
    adapter, processor, nested, provenance, count
):
    """Identical non-prefix indices alone cannot distinguish sampling from loss."""
    indices = [i for i in range(700) if i != 350]
    video = _video(indices, fps=30, total=700)
    video.is_sampled = provenance
    presampled = _video(np.linspace(0, 699, 32).astype(int).tolist(), fps=30, total=700)
    original = copy.deepcopy(video.metadata)
    result = _run(
        adapter, processor, [video, presampled], **({"videos_kwargs": {}} if nested else {})
    )
    selected = np.linspace(0, 698, count).round().astype(int).tolist()
    expected = [indices[i] for i in selected]
    # HF pads odd frame counts to a temporal pair, without additional sampling.
    if count % 2:
        expected.append(expected[-1])
    metadata = result["video_metadata"][0]
    assert list(metadata.frames_indices) == expected
    assert result["video_grid_thw"][:, 0].tolist() == [(count + 1) // 2, 16]
    assert metadata.fps == 30 and metadata.total_num_frames == 700
    assert metadata.duration == 700 / 30
    assert processor._calculate_timestamps(metadata.frames_indices, 30) == pytest.approx(
        [(expected[i] + expected[i + 1]) / 60 for i in range(0, len(expected), 2)]
    )
    assert list(result["video_metadata"][1].frames_indices) == presampled.metadata["frames_indices"]
    assert video.metadata == original
    assert (
        adapter.get_num_tokens_per_video(
            video=video.frames, video_metadata=video.metadata, video_is_sampled=provenance
        )
        == int(result["video_grid_thw"][0].prod()) // 4
    )
    n_patches = int(result["video_grid_thw"][0].prod())
    patches = result["pixel_values_videos"][:n_patches].reshape((count + 1) // 2, -1, 3, 2, 16, 16)
    padded = selected + ([selected[-1]] if count % 2 else [])
    assert patches[:, 0, 0, :, 0, 0].flatten().tolist() == padded


def test_trusted_single_frame_sample_is_preserved(adapter, processor):
    video = _video([0], fps=30, total=700)
    video.is_sampled = True
    result = _run(adapter, processor, [video])
    assert list(result["video_metadata"][0].frames_indices) == [0, 0]


@pytest.mark.parametrize(
    "kwargs,count",
    [
        ({"num_frames": 0}, 699),
        ({"fps": -1}, 699),
        ({"do_sample_frames": False}, 699),
        ({"num_frames": 1000, "fps": 1}, 23),
        ({"num_frames": 32}, 32),
    ],
)
def test_decode_provenance_respects_explicit_controls(adapter, processor, kwargs, count):
    """Full-decode provenance only changes the implicit sampling path."""
    video = _video([i for i in range(700) if i != 350], fps=30, total=700)
    video.is_sampled = False
    result = _run(adapter, processor, [video], **kwargs)
    assert int(result["video_grid_thw"][0, 0]) == (count + 1) // 2


def test_disabled_processor_respects_decode_provenance(adapter, processor):
    processor.video_processor.do_sample_frames = False
    video = _video([i for i in range(700) if i != 350], fps=30, total=700)
    video.is_sampled = False
    result = _run(adapter, processor, [video])
    assert int(result["video_grid_thw"][0, 0]) == 350


@pytest.fixture()
def adapter(processor) -> Qwen3VLInputProcessorBase:
    instance = object.__new__(Qwen3VLInputProcessorBase)
    instance._processor = processor
    instance._config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
    return instance


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


def _run(
    adapter: Qwen3VLInputProcessorBase, processor: Qwen3VLProcessor, videos: list, **kwargs
) -> dict:
    request = {"return_metadata": True, **kwargs}
    if "videos_kwargs" in request:
        request["videos_kwargs"] = {"do_normalize": False, **request["videos_kwargs"]}
    else:
        request.setdefault("do_normalize", False)
    return adapter._preprocess(
        "<|video_pad|>" * len(videos),
        {"video": videos},
        request,
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


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("processor_fps,count", [(2, 60), (30, 768)])
def test_dropped_frame_above_cap_samples_with_source_mapping(
    adapter, processor, nested, processor_fps, count
) -> None:
    """A failed retrieve must not let the default path bypass HF's frame cap."""
    processor.video_processor.fps = processor_fps
    indices = [i for i in range(900) if i != 450]
    video = _video(indices, fps=30, total=900)
    prefix = _video(list(range(450)), fps=30, total=900)
    presampled = _video(np.linspace(0, 899, 32).astype(int).tolist(), fps=30, total=900)
    original = copy.deepcopy(video.metadata)
    result = _run(
        adapter, processor, [video, prefix, presampled], **({"videos_kwargs": {}} if nested else {})
    )
    selected = np.linspace(0, 898, count).round().astype(int)
    expected = [indices[i] for i in selected]
    metadata = result["video_metadata"][0]
    assert result["video_grid_thw"][:, 0].tolist() == [
        count // 2,
        (30 if processor_fps == 2 else 450) // 2,
        16,
    ]
    assert list(metadata.frames_indices) == expected
    assert metadata.fps == 30
    assert metadata.total_num_frames == 900
    assert metadata.duration == 30
    assert processor._calculate_timestamps(metadata.frames_indices, metadata.fps) == pytest.approx(
        [(expected[i] + expected[i + 1]) / 60 for i in range(0, count, 2)]
    )
    # Recover the first video's pixels; they encode decoded positions, not source indices.
    n_patches = int(result["video_grid_thw"][0].prod().item())
    patches = result["pixel_values_videos"][:n_patches].reshape(count // 2, -1, 3, 2, 16, 16)
    assert patches[:, 0, 0, :, 0, 0].flatten().tolist() == selected.tolist()
    assert list(result["video_metadata"][2].frames_indices) == presampled.metadata["frames_indices"]
    assert video.metadata == original
    assert adapter.get_num_tokens_per_video(
        video=video.frames, video_metadata=video.metadata
    ) == adapter.get_num_tokens_per_video(
        video=video.frames, video_grid_thw=result["video_grid_thw"][0]
    )


@pytest.mark.parametrize("n_decoded", [768, 769])
def test_presampled_default_frame_cap_boundary(adapter, processor, n_decoded) -> None:
    """Preserve IO samples through the cap and use HF defaults above it."""
    video = _video(np.linspace(0, 2399, n_decoded).astype(int).tolist(), total=2400)
    result = _run(adapter, processor, [video])
    expected_count = n_decoded if n_decoded == 768 else 200
    assert result["video_grid_thw"][0, 0].item() == expected_count // 2
    selected = np.linspace(0, n_decoded - 1, expected_count).round().astype(int)
    assert list(result["video_metadata"][0].frames_indices) == [
        video.metadata["frames_indices"][i] for i in selected
    ]


@pytest.mark.parametrize(
    "kwargs,processor_disabled,count",
    [
        ({"do_sample_frames": False}, False, 899),
        ({}, True, 899),
        ({"num_frames": 899}, False, 899),
        ({"num_frames": 0}, False, 899),
        ({"fps": 0}, False, 899),
        ({"num_frames": 8, "fps": 2}, False, 8),
    ],
)
def test_above_cap_respects_explicit_sampling_controls(
    adapter, processor, kwargs, processor_disabled, count
) -> None:
    """The default cap guard must not override caller targets or disabled sampling."""
    processor.video_processor.do_sample_frames = not processor_disabled
    video = _video([i for i in range(900) if i != 450], fps=30, total=900)
    result = _run(adapter, processor, [video], **kwargs)
    assert result["video_grid_thw"][0, 0].item() == (count + 1) // 2
    selected = np.linspace(0, 898, count).round().astype(int)
    expected = [video.metadata["frames_indices"][i] for i in selected]
    if count % 2:
        expected.append(expected[-1])  # HF pads the final temporal patch.
    assert list(result["video_metadata"][0].frames_indices) == expected


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
    adapter._preprocess("<|video_pad|>", {"video": [video]}, request)
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
    result = adapter._preprocess("<|video_pad|>", {"video": [video]}, request)
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
    result = adapter._preprocess(text, {"image": images}, {"do_normalize": False})
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
    result = adapter._preprocess("<|video_pad|>", {"video": [video]}, kwargs)
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
    result = adapter._preprocess(
        "<|video_pad|>",
        {"video": [video]},
        {"videos_kwargs": {"num_frames": 4, "do_normalize": False}},
    )
    assert result["pixel_values_videos"].mean().item() == pytest.approx(128 / 255)


def test_nested_image_and_video_preserve_float_pixel_values(adapter, processor):
    images = [torch.ones(3, 32, 32)]
    video = _video(list(range(0, 192, 24)))
    result = adapter._preprocess(
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


@pytest.mark.parametrize("n_decoded", [32, 40])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("do_sample_frames", [None, True, False])
def test_combined_sampling_targets_use_smaller_source_time_target(
    adapter, processor, n_decoded, nested, do_sample_frames
):
    video = _video(np.linspace(0, 239, n_decoded).astype(int).tolist())
    kwargs = {"num_frames": 40, "fps": 2, "do_normalize": False}
    if do_sample_frames is not None:
        kwargs["do_sample_frames"] = do_sample_frames
    request = {"videos_kwargs": kwargs} if nested else kwargs
    if do_sample_frames is False:
        result = _run(adapter, processor, [video], **request)
        assert result["video_grid_thw"][0, 0].item() == n_decoded // 2
        assert list(result["video_metadata"][0].frames_indices) == video.metadata["frames_indices"]
    else:
        result = _run(adapter, processor, [video], **request)
        selected = np.linspace(0, n_decoded - 1, 20).round().astype(int)
        assert result["video_grid_thw"][0, 0].item() == 10
        assert list(result["video_metadata"][0].frames_indices) == [
            video.metadata["frames_indices"][i] for i in selected
        ]
        patches = result["pixel_values_videos"].reshape(10, -1, 3, 2, 16, 16)
        assert patches[:, 0, 0, :, 0, 0].flatten().tolist() == selected.tolist()


@pytest.mark.parametrize(
    "n_decoded,source_fps,count", [(40, 24.0, 4), (800, 24.0, 66), (800, 0.01, 768)]
)
def test_truncated_prefix_uses_hf_sampling_and_frame_cap(
    adapter, processor, n_decoded, source_fps, count
):
    video = _video(list(range(n_decoded)), fps=source_fps, total=24000)
    original = copy.deepcopy(video.metadata)
    result = _run(adapter, processor, [video])
    selected = np.linspace(0, n_decoded - 1, count).round().astype(int).tolist()
    assert result["video_grid_thw"][0, 0].item() == count // 2
    assert list(result["video_metadata"][0].frames_indices) == selected
    assert video.metadata == original


@pytest.mark.parametrize("count", [8, 10])
def test_default_io_counts_are_preserved(adapter, processor, count):
    video = _video(np.linspace(0, 239, count).astype(int).tolist())
    result = _run(adapter, processor, [video])
    assert result["video_grid_thw"][0, 0].item() == count // 2
    assert list(result["video_metadata"][0].frames_indices) == video.metadata["frames_indices"]


@pytest.mark.parametrize("target", ["num_frames", "fps"])
@pytest.mark.parametrize("value", [0, -1, -2])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("presampled", [False, True])
def test_nonpositive_single_target_preserves_available_frames(
    adapter, processor, target, value, nested, presampled
):
    indices = np.linspace(0, 239, 40).astype(int).tolist() if presampled else list(range(40))
    video = _video(indices)
    original = copy.deepcopy(video.metadata)
    targets = {target: value}
    kwargs = {"videos_kwargs": targets} if nested else targets
    result = _run(adapter, processor, [video], **kwargs)
    assert result["video_grid_thw"][0, 0].item() == 20
    assert list(result["video_metadata"][0].frames_indices) == indices
    patches = result["pixel_values_videos"].reshape(20, -1, 3, 2, 16, 16)
    assert patches[:, 0, 0, :, 0, 0].flatten().tolist() == list(range(40))
    assert video.metadata == original


@pytest.mark.parametrize("target", ["num_frames", "fps"])
def test_explicit_sampling_with_nonpositive_target_keeps_available_frames(
    adapter, processor, target
):
    video = _video(list(range(40)))
    result = _run(adapter, processor, [video], do_sample_frames=True, **{target: 0})
    assert result["video_grid_thw"][0, 0].item() == 20
    assert list(result["video_metadata"][0].frames_indices) == list(range(40))


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("num_frames", [None, 0, 40])
@pytest.mark.parametrize(
    "missing_prefix_field", [None, "fps", "duration", "frames_indices", "total_num_frames"]
)
def test_explicit_fps_uses_available_prefix_duration(
    adapter, processor, nested, num_frames, missing_prefix_field
):
    prefix = _video(list(range(40)))
    if missing_prefix_field is not None:
        del prefix.metadata[missing_prefix_field]
    presampled = _video(np.linspace(0, 239, 40).astype(int).tolist())
    originals = [copy.deepcopy(v.metadata) for v in (prefix, presampled)]
    targets = {"fps": 4}
    if num_frames is not None:
        targets["num_frames"] = num_frames
    kwargs = {"videos_kwargs": targets} if nested else targets
    result = _run(adapter, processor, [prefix, presampled], **kwargs)
    assert result["video_grid_thw"][:, 0].tolist() == [3, 20]
    selected = np.linspace(0, 39, 6).round().astype(int).tolist()
    metadata = result["video_metadata"]
    assert list(metadata[0].frames_indices) == selected
    assert list(metadata[1].frames_indices) == presampled.metadata["frames_indices"]
    patches = result["pixel_values_videos"].reshape(23, -1, 3, 2, 16, 16)
    assert patches[:3, 0, 0, :, 0, 0].flatten().tolist() == selected
    for actual, original, video in zip(metadata, originals, (prefix, presampled), strict=True):
        for key in ("fps", "duration", "total_num_frames"):
            if key in original:
                assert actual[key] == original[key]
        assert video.metadata == original


@pytest.mark.parametrize(
    "kwargs,count",
    [
        ({"num_frames": 8, "fps": 2}, 8),
        ({"num_frames": 40, "fps": 0.01}, 1),
        ({"num_frames": 80, "fps": 8}, 40),
        ({"num_frames": 8, "fps": 0}, 8),
        ({"num_frames": 0, "fps": 2}, 20),
        ({"num_frames": 0, "fps": 0}, 40),
    ],
)
def test_combined_targets_respect_frame_cap_and_one_frame_minimum(
    adapter, processor, kwargs, count
):
    video = _video(np.linspace(0, 239, 40).astype(int).tolist())
    result = _run(adapter, processor, [video], **kwargs)
    assert result["video_grid_thw"][0, 0].item() == (count + 1) // 2
    expected = np.linspace(0, 39, count).round().astype(int).tolist()
    metadata = result["video_metadata"][0]
    assert list(metadata.frames_indices)[:count] == [
        video.metadata["frames_indices"][i] for i in expected
    ]


def test_combined_targets_use_each_videos_source_duration(adapter, processor):
    short = _video(np.linspace(0, 47, 16).astype(int).tolist(), total=48)
    long = _video(np.linspace(0, 239, 40).astype(int).tolist())
    result = _run(adapter, processor, [short, long], num_frames=40, fps=2)
    assert result["video_grid_thw"][:, 0].tolist() == [2, 10]
    assert [len(m.frames_indices) for m in result["video_metadata"]] == [4, 20]


@pytest.mark.parametrize("nested", [False, True])
def test_disabled_processor_preserves_combined_target_override(adapter, processor, nested):
    processor.video_processor.do_sample_frames = False
    video = _video(np.linspace(0, 239, 40).astype(int).tolist())
    targets = {"num_frames": 8, "fps": 2}
    kwargs = {"videos_kwargs": targets} if nested else targets
    result = _run(adapter, processor, [video], **kwargs)
    assert result["video_grid_thw"][0, 0].item() == 20


@pytest.mark.parametrize("layout", ["presampled", "full", "prefix"])
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"num_frames": 8},
        {"videos_kwargs": {"fps": 2}},
        {"num_frames": 40, "fps": 2},
        {"num_frames": 0},
        {"fps": 0},
        {"fps": 4},
    ],
)
def test_token_count_fallback_matches_encoder_grid(adapter, processor, layout, kwargs):
    if layout == "presampled":
        video = _video(np.linspace(0, 239, 40).astype(int).tolist())
    elif layout == "full":
        video = _video(list(range(48)), total=48)
    else:
        video = _video(list(range(40)))
    original = copy.deepcopy(video.metadata)
    result = _run(adapter, processor, [video], **kwargs)
    expected = int(result["video_grid_thw"][0].prod().item()) // 4
    assert (
        adapter.get_num_tokens_per_video(
            video=video.frames, video_metadata=video.metadata, **kwargs
        )
        == expected
    )
    assert video.metadata == original


@pytest.mark.parametrize("format", ["numpy", "pil"])
def test_token_count_fallback_accepts_uint8_frames(adapter, processor, format):
    video = _video(np.linspace(0, 239, 10).astype(int).tolist())
    pixels = np.full((10, 32, 32, 3), 128, dtype=np.uint8)
    video.frames = pixels if format == "numpy" else [Image.fromarray(frame) for frame in pixels]
    result = _run(adapter, processor, [video])
    expected = int(result["video_grid_thw"][0].prod().item()) // 4
    assert (
        adapter.get_num_tokens_per_video(video=video.frames, video_metadata=video.metadata)
        == expected
    )


def test_token_count_without_metadata_matches_preprocess(adapter, processor):
    video = _video(list(range(48)), total=48)
    video.metadata = {}
    result = _run(adapter, processor, [video])
    assert (
        adapter.get_num_tokens_per_video(video=video.frames)
        == int(result["video_grid_thw"][0].prod().item()) // 4
    )


@pytest.mark.parametrize("batched", [False, True])
def test_token_count_with_grid_skips_preprocessing(adapter, monkeypatch, batched):
    def fail(*args, **kwargs):
        pytest.fail("A processed grid must avoid reprocessing video pixels")

    monkeypatch.setattr(adapter, "_preprocess", fail)
    grid = torch.tensor([[4, 6, 8], [2, 4, 6]]) if batched else torch.tensor([4, 6, 8])
    assert adapter.get_num_tokens_per_video(video=[], video_grid_thw=grid) == (
        60 if batched else 48
    )

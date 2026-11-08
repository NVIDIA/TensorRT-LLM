# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Regression tests for `VideoData.update_hash`.

Cover both branches:
  * `raw_bytes_hash` set  -> source-anchor path (must not walk frames)
  * `raw_bytes_hash` None -> frame-walk fallback (must include every frame)

Plus sampling-metadata and audio contributions to cache identity.
"""

import pytest
import torch
from blake3 import blake3

from tensorrt_llm.inputs import multimodal_data as md
from tensorrt_llm.inputs.multimodal_data import AudioData, VideoData

pytestmark = pytest.mark.cpu_only


def _hex(video: VideoData) -> str:
    h = blake3()
    video.update_hash(h)
    return h.hexdigest()


def _make_video(raw_bytes_hash=None, audio=None, frames=None, meta=None):
    frames = frames or [torch.zeros((3, 8, 8), dtype=torch.float32) for _ in range(3)]
    meta = meta or {
        "frames_indices": [0, 1, 2],
        "fps": 30.0,
        "duration": 1.0,
        "total_num_frames": 3,
    }
    return VideoData(frames=frames, metadata=meta, audio=audio, raw_bytes_hash=raw_bytes_hash)


def _tensor_serialize_calls(spy) -> int:
    return sum(1 for call in spy.call_args_list if isinstance(call.args[0], torch.Tensor))


def test_source_anchor_skips_frame_walk(mocker):
    """When `raw_bytes_hash` is set the frames must not be serialized."""
    spy = mocker.spy(md, "serialize_item")
    _hex(_make_video(raw_bytes_hash="a" * 64))
    assert _tensor_serialize_calls(spy) == 0


def test_frame_walk_fallback_serializes_every_frame(mocker):
    """When `raw_bytes_hash` is None the fallback hashes each frame."""
    spy = mocker.spy(md, "serialize_item")
    _hex(_make_video(raw_bytes_hash=None))
    assert _tensor_serialize_calls(spy) == 3


def test_source_anchor_hash_stable_across_frame_content():
    """Frame contents must not affect the digest when source anchor is set.

    Given identical `raw_bytes_hash` + metadata, the source anchor is
    authoritative and pixel data must not participate in the hash.
    """
    v1 = _make_video(
        raw_bytes_hash="deadbeef" * 8, frames=[torch.zeros((3, 4, 4)) for _ in range(2)]
    )
    v2 = _make_video(
        raw_bytes_hash="deadbeef" * 8, frames=[torch.ones((3, 4, 4)) for _ in range(2)]
    )
    assert _hex(v1) == _hex(v2)


def test_metadata_change_alters_hash():
    """`frames_indices` / `fps` are part of cache identity."""
    base_meta = {"frames_indices": [0, 1, 2], "fps": 30.0}
    v1 = _make_video(raw_bytes_hash="a" * 64, meta=dict(base_meta))
    v2 = _make_video(raw_bytes_hash="a" * 64, meta={**base_meta, "frames_indices": [0, 2, 4]})
    v3 = _make_video(raw_bytes_hash="a" * 64, meta={**base_meta, "fps": 15.0})
    assert _hex(v1) != _hex(v2)
    assert _hex(v1) != _hex(v3)


def test_audio_contributes_to_hash():
    """Adding an audio track changes the digest."""
    audio = AudioData(samples=torch.zeros(1024), sample_rate=16000)
    without_audio = _make_video(raw_bytes_hash="a" * 64)
    with_audio = _make_video(raw_bytes_hash="a" * 64, audio=audio)
    assert _hex(without_audio) != _hex(with_audio)


def test_different_source_hashes_differ():
    """Different `raw_bytes_hash` inputs must yield different digests."""
    v1 = _make_video(raw_bytes_hash="a" * 64)
    v2 = _make_video(raw_bytes_hash="b" * 64)
    assert _hex(v1) != _hex(v2)

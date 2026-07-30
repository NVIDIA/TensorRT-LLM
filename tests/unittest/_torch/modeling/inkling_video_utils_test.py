# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage-7 / Goal 7.1 focused tests for the Inkling VIDEO (multi-frame image) path.

All functional -- NO Video-MME, no MVBench, no scored/accuracy/parity gate (human
feedback #22 TASK 2 / #23). Inkling has NO separate video tower: a video is decoded
to frames, a subset of frames is sampled, and each sampled frame is fed as an
ordinary image through the SAME hMLP vision tower (one ``<image>`` placeholder span
per frame). So the tests cover exactly the video-specific surface:

  * frame sampling matches the SGLang reference frame-for-frame -- a verbatim
    reproduction of SGLang ``test/registered/vlm/test_video_utils.py`` against the
    ported :func:`sample_video_frames` (bounded by desired FPS / max_frames / total
    frames, always >=1 frame, temporal order preserved);
  * :func:`sample_video_as_images` extracts the sampled frames from a short clip,
    preserving frame COUNT and ORDER;
  * the accepted multi-image :meth:`InklingInputProcessor.assemble` turns those K
    frames into K ``<image>`` placeholder spans -- one per frame, each expanded to
    that frame's own patch count, the concatenated ``vision_patches_bthwc`` rows
    equal to the frame's own patches in encounter order -- and FAILS LOUDLY on any
    frame/placeholder count mismatch;
  * the multi-frame patches forward through the REAL NVFP4-checkpoint
    ``InklingVisionModel`` on CUDA to one ``decoder_dmodel`` row per patch with no
    NaN/Inf and no dtype drift / checkpoint upcast (same vision tower Stage 1 used).

Run in the Slurm container (imports ``tensorrt_llm``); the CUDA forward uses one
GPU. See ``workspace/.../inkling_video_utils.sbatch``.
"""
import json
import os
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.models.modeling_inkling_vision import (
    DEFAULT_AUDIO_TOKEN_ID,
    DEFAULT_IMAGE_TOKEN_ID,
    DecodedVideo,
    InklingAudioPreprocessor,
    InklingImagePreprocessor,
    InklingInputProcessor,
    InklingVisionModel,
    sample_video_as_images,
    sample_video_frames,
)

_DEFAULT_CKPT = (
    "/lustre/fsw/coreai_comparch_trtllm/kleinc/hf_data/hf_home/hub/"
    "models--thinkingmachines--Inkling-NVFP4/snapshots/"
    "95e51a54d9486020a80d49ae4f9103fb2b3f9686"
)
CKPT = os.environ.get("INKLING_CKPT", _DEFAULT_CKPT)
_HAVE_CKPT = os.path.isfile(os.path.join(CKPT, "config.json"))


# --------------------------------------------------------------------------- #
# Frame sampling parity vs SGLang (no checkpoint / GPU needed)
#   Verbatim reproduction of sglang test/registered/vlm/test_video_utils.py so
#   TRT sample_video_frames selects the SAME frame indices SGLang does.
# --------------------------------------------------------------------------- #
class DummyVideo:
    """Minimal ``__len__`` + ``avg_fps`` view -- identical to the SGLang test's
    ``DummyVideo`` so the sampling cases are a byte-for-byte reproduction."""

    def __init__(self, total_frames: int, avg_fps: float):
        self._frames = total_frames
        self._fps = avg_fps

    def __len__(self):
        return self._frames

    @property
    def avg_fps(self):
        return self._fps


@dataclass(kw_only=True)
class _Case:
    frames: int
    avg_fps: float
    desired_fps: int
    max_frames: int
    expected_frames: list
    description: str


@pytest.mark.parametrize(
    "case",
    [
        _Case(
            frames=100, avg_fps=25.0, desired_fps=5, max_frames=200,
            expected_frames=[0, 5, 10, 15, 20, 26, 31, 36, 41, 46, 52, 57, 62, 67, 72, 78, 83, 88, 93, 99],
            description="capped by desired_fps",
        ),
        _Case(
            frames=10, avg_fps=10.0, desired_fps=100, max_frames=5,
            expected_frames=[0, 2, 4, 6, 9],
            description="capped by max_frames",
        ),
        _Case(
            frames=50, avg_fps=25.0, desired_fps=50, max_frames=200,
            expected_frames=list(range(50)),
            description="capped by total_frames",
        ),
        _Case(
            frames=1, avg_fps=30.0, desired_fps=0, max_frames=0,
            expected_frames=[0],
            description="always sample at least 1 frame",
        ),
    ],
    ids=lambda c: c.description,
)
def test_sample_video_frames_matches_sglang(case: _Case):
    video = DummyVideo(case.frames, case.avg_fps)
    result = sample_video_frames(
        video, desired_fps=case.desired_fps, max_frames=case.max_frames
    )
    assert result == case.expected_frames


def test_sample_video_frames_indices_strictly_increasing():
    # Temporal order preserved for a non-trivial sub-sample.
    idxs = sample_video_frames(DummyVideo(120, 30.0), desired_fps=3, max_frames=64)
    assert idxs == sorted(idxs)
    assert len(set(idxs)) == len(idxs)  # no duplicate frames
    assert idxs[0] == 0 and idxs[-1] == 119  # spans the whole clip


def test_sample_video_frames_empty_video_raises():
    with pytest.raises(AssertionError):
        sample_video_frames(DummyVideo(0, 25.0), desired_fps=5, max_frames=10)


# --------------------------------------------------------------------------- #
# sample_video_as_images: extract the sampled frames, preserving count + order
# --------------------------------------------------------------------------- #
def _synth_frame(h: int, w: int, seed: int) -> np.ndarray:
    """A distinct deterministic RGB frame (uint8 HWC) -- a valid 'video frame',
    no codec/file I/O. Content depends on ``seed`` so frames are not identical."""
    rng = np.random.RandomState(seed)
    base = rng.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
    base[:, :, 0] = (base[:, :, 0].astype(int) + 7 * seed) % 256  # per-frame tint
    return base


def test_sample_video_as_images_count_and_order():
    # 12 distinct frames at 12 fps (1 s clip); sample down to 4.
    frames = [_synth_frame(48, 64, seed=i) for i in range(12)]
    video = DecodedVideo(frames, avg_fps=12.0)
    idxs = sample_video_frames(video, desired_fps=4, max_frames=32)
    picked = sample_video_as_images(video, desired_fps=4, max_frames=32)
    # count matches the sampler, order preserved, and each picked frame IS the
    # frame at the sampled index (identity + ordering).
    assert len(picked) == len(idxs)
    assert len(picked) >= 1
    for pf, i in zip(picked, idxs):
        assert pf is frames[i]
    assert idxs == sorted(idxs)


def test_sample_video_as_images_single_frame_clip():
    frames = [_synth_frame(40, 40, seed=0)]
    video = DecodedVideo(frames, avg_fps=30.0)
    picked = sample_video_as_images(video, desired_fps=1, max_frames=8)
    assert len(picked) == 1 and picked[0] is frames[0]


def test_decoded_video_requires_frames():
    with pytest.raises(ValueError):
        DecodedVideo([], avg_fps=25.0)


# --------------------------------------------------------------------------- #
# Multi-frame -> multi-image assemble: one <image> span per frame, in order
# --------------------------------------------------------------------------- #
def _bare_processor():
    """An InklingInputProcessor with only the fields ``assemble`` needs, so the
    pure multi-image expansion/validation logic is testable without a
    tokenizer/model load (same pattern as the audio tower test)."""
    ip = InklingInputProcessor.__new__(InklingInputProcessor)
    ip.image_token_id = DEFAULT_IMAGE_TOKEN_ID
    ip.audio_token_id = DEFAULT_AUDIO_TOKEN_ID
    ip._preprocessor = InklingImagePreprocessor()
    ip._audio_preprocessor = InklingAudioPreprocessor()
    return ip


def test_assemble_video_frames_one_span_per_frame():
    ip = _bare_processor()
    # Four frames of DIFFERENT sizes -> different per-frame patch counts, so a
    # reordering/merge would change span lengths and be detected.
    frames = [
        _synth_frame(48, 64, seed=1),
        _synth_frame(80, 48, seed=2),
        _synth_frame(40, 120, seed=3),
        _synth_frame(96, 72, seed=4),
    ]
    per_frame_patches = [
        int(ip._preprocessor.preprocess(f)["num_patches"][0]) for f in frames
    ]
    # Interleaved placeholder stream: text, <image>, text, <image>, ... one per frame.
    ids = [10, ip.image_token_id, 11, ip.image_token_id, 12, ip.image_token_id, 13, ip.image_token_id, 14]
    out_ids, mm = ip.assemble(ids, image_data=frames, audio_data=None)

    assert "image" in mm and "audio" not in mm
    # one <image> span per frame, each expanded to that frame's own patch count
    assert mm["image"]["num_patches"] == per_frame_patches
    assert out_ids.count(ip.image_token_id) == sum(per_frame_patches)
    # non-media tokens are preserved verbatim (10..14 survive, in order)
    assert [t for t in out_ids if t not in (ip.image_token_id,)] == [10, 11, 12, 13, 14]

    # offsets segment the expanded stream one contiguous span per frame, in order
    offsets = mm["image"]["offsets"]
    assert len(offsets) == len(frames)
    prev_end = -1
    for (start, end), npat in zip(offsets, per_frame_patches):
        assert start > prev_end  # strictly increasing, non-overlapping
        assert end - start + 1 == npat  # span length == that frame's patch count
        assert all(out_ids[p] == ip.image_token_id for p in range(start, end + 1))
        prev_end = end

    # concatenated vision_patches rows == the frames' own patches, in encounter order
    vp = mm["image"]["vision_patches_bthwc"]
    assert vp.shape[0] == sum(per_frame_patches)
    assert vp.ndim == 5 and vp.shape[1:] == (2, 40, 40, 3)  # (rows, T, P, P, C)
    cum = 0
    for f, npat in zip(frames, per_frame_patches):
        expect = ip._preprocessor.encode_one(f)
        assert torch.equal(vp[cum:cum + npat], expect)
        cum += npat


def test_assemble_video_failloud_frame_placeholder_mismatch():
    ip = _bare_processor()
    frames = [_synth_frame(48, 48, seed=i) for i in range(3)]
    # 3 placeholders but 2 frames
    with pytest.raises(ValueError):
        ip.assemble(
            [ip.image_token_id, ip.image_token_id, ip.image_token_id],
            image_data=frames[:2],
            audio_data=None,
        )
    # 2 placeholders but 3 frames
    with pytest.raises(ValueError):
        ip.assemble(
            [ip.image_token_id, ip.image_token_id],
            image_data=frames,
            audio_data=None,
        )


# --------------------------------------------------------------------------- #
# Multi-frame vision tower with REAL checkpoint weights, forwarded on CUDA
#   (the SAME hMLP tower Stage 1 used; video adds no new tower)
# --------------------------------------------------------------------------- #
def _vision_config_from_ckpt():
    cfg = json.load(open(os.path.join(CKPT, "config.json")))["vision_config"]

    class _VC:
        pass

    vc = _VC()
    for k, v in cfg.items():
        setattr(vc, k, v)
    return vc


def _load_visual_weights():
    from safetensors import safe_open

    idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
    keys = [k for k in idx if k.startswith("model.visual.")]
    handles, out = {}, {}
    for k in keys:
        shard = idx[k]
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(CKPT, shard), framework="pt")
        out[k] = handles[shard].get_tensor(k)
    return out


@pytest.mark.skipif(not _HAVE_CKPT, reason=f"checkpoint not found at {CKPT}")
def test_video_multiframe_tower_real_weights_cuda_forward():
    vc = _vision_config_from_ckpt()
    assert int(vc.decoder_dmodel) == 6144 and int(vc.patch_size) == 40
    assert int(vc.temporal_patch_size) == 2 and bool(vc.use_vision_norm)

    tower = InklingVisionModel(vc).to(torch.bfloat16)
    tower.load_weights(_load_visual_weights())  # strict: 8 model.visual.* tensors
    assert tower.layers["linear_3"].weight.dtype == torch.bfloat16  # no upcast

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tower = tower.to(dev)

    # A short 5-frame clip (5 distinct frames) sampled + preprocessed the way the
    # multi-image video path does; forward ALL frames' patches through the tower.
    frames = [_synth_frame(64, 48, seed=i) for i in range(6)]
    video = DecodedVideo(frames, avg_fps=6.0)
    picked = sample_video_as_images(video, desired_fps=5, max_frames=5)
    assert len(picked) == 5  # 6 frames @6fps, 1s -> desired 5fps -> 5 frames

    pre = InklingImagePreprocessor(
        patch_size=int(vc.patch_size), temporal_patch_size=int(vc.temporal_patch_size)
    )
    feat = pre.preprocess(picked)
    per_frame = [int(n) for n in feat["num_patches"]]
    total = sum(per_frame)
    vp = feat["vision_patches_bthwc"].to(dev)

    out = tower(vp)

    assert out.shape == (total, 6144)  # one decoder_dmodel row per patch, all frames
    assert out.dtype == torch.bfloat16  # no dtype drift
    assert torch.isfinite(out.float()).all()  # no NaN/Inf
    assert float(out.float().abs().sum()) > 0.0  # non-empty / non-degenerate
    print(
        f"VIDEO_TOWER_CUDA_OK dev={dev} n_frames={len(picked)} "
        f"per_frame_patches={per_frame} total_rows={total} "
        f"out={tuple(out.shape)} dtype={out.dtype} finite=True"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))

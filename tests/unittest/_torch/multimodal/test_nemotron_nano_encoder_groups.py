# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Encoder-group tests for `NemotronH_Nano_VL_V2` with stubbed vision /
sound encoders.

Runs in pre-merge CI — no weights required. Exercises the three
per-modality encoder groups and their `encoder_fn` plumbing directly on
the class:

* `mm_encoder_groups` returns one group per underlying encoder call
  (image, video, audio) — the invariant that keeps `EncoderGroup`'s
  "one encoder invocation per group" contract honest for RADIO, whose
  image and video sub-paths cannot share a single ViT forward.
* `_encode_image_group` / `_encode_video_group` wrap each incoming
  param in a single-modality virtual param (`modality_type` legacy tag
  set), which is the workaround for
  `NanoV2VLVisionEncoder.forward`'s exactly-one-of-image/video
  assertion.
* `_encode_video_group` stashes per-video EVS retained-token counts on
  the caller's `multimodal_data` for the downstream `merge_evs_mm_embeds`.
* `_encode_audio_group` concatenates `_encode_audio` outputs in
  request order.
"""

from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import torch

from tensorrt_llm._torch.models.modeling_nemotron_nano import NemotronH_Nano_VL_V2
from tensorrt_llm.inputs.multimodal import MultimodalParams


def _make_multimodal_params(
    *,
    image: Optional[Dict[str, Any]] = None,
    video: Optional[Dict[str, Any]] = None,
    audio: Optional[Dict[str, Any]] = None,
) -> MultimodalParams:
    """Build a `MultimodalParams` with only the requested buckets populated."""
    data: Dict[str, Any] = {}
    if image is not None:
        data["image"] = image
    if video is not None:
        data["video"] = video
    if audio is not None:
        data["audio"] = audio
    return MultimodalParams(multimodal_data=data, mm_item_order=None)


def _marker_tensor(marker: int, n: int, dim: int = 4) -> torch.Tensor:
    """Column-0 carries the marker so ordering can be asserted."""
    t = torch.zeros((n, dim))
    t[:, 0] = float(marker)
    return t


def _model_with_stubs() -> NemotronH_Nano_VL_V2:
    """Bypass `__init__` and inject stubbed encoders.

    Stubbed `vision_encoder` reads `marker` + `rows` off the virtual param's
    bucket and returns marker tensors. Stubbed `_encode_audio` does the same
    for audio inputs.
    """
    m = NemotronH_Nano_VL_V2.__new__(NemotronH_Nano_VL_V2)

    def _vision(
        virtuals: List[MultimodalParams],
    ) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]]]:
        embeds: List[torch.Tensor] = []
        for v in virtuals:
            mt = v.multimodal_data["modality_type"]
            bucket = v.multimodal_data[mt]
            embeds.append(_marker_tensor(bucket["marker"], bucket["rows"]))
        return embeds, [None] * len(virtuals)

    m.vision_encoder = MagicMock(side_effect=_vision)
    m._encode_audio = lambda audio_data_list: [
        (_marker_tensor(a["marker"], a["rows"]), [a["rows"]]) for a in audio_data_list
    ]
    m.sound_encoder = MagicMock()  # non-None so video-embedded-audio can run
    return m


class TestMmEncoderGroups:
    def test_property_registers_three_per_modality_groups(self) -> None:
        groups = _model_with_stubs().mm_encoder_groups
        assert [g.modalities for g in groups] == [
            ("image",),
            ("video",),
            ("audio",),
        ]

    def test_image_group_wraps_virtual_params_and_concats(self) -> None:
        m = _model_with_stubs()
        params = [
            _make_multimodal_params(image={"marker": 1, "rows": 3}),
            _make_multimodal_params(image={"marker": 2, "rows": 2}),
        ]
        out = m._encode_image_group(params)
        assert torch.equal(out[:, 0], torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0]))
        # Each virtual param carries only the image bucket + legacy modality_type.
        virtuals = m.vision_encoder.call_args_list[0][0][0]
        for v in virtuals:
            assert v.multimodal_data["modality_type"] == "image"
            assert set(v.multimodal_data.keys()) == {"modality_type", "image"}

    def test_video_group_stashes_num_tokens_in_video(self) -> None:
        m = _model_with_stubs()
        # Return per-video EVS retained counts alongside embeddings.
        m.vision_encoder = MagicMock(
            side_effect=lambda virtuals: ([_marker_tensor(9, 5)], [torch.tensor([3, 2])])
        )
        p = _make_multimodal_params(video={"marker": 9, "rows": 5, "video_size": []})
        m._encode_video_group([p])
        assert torch.equal(p.multimodal_data["num_tokens_in_video"], torch.tensor([3, 2]))

    def test_audio_group_concats_encode_audio_outputs(self) -> None:
        m = _model_with_stubs()
        params = [
            _make_multimodal_params(audio={"marker": 7, "rows": 4}),
            _make_multimodal_params(audio={"marker": 8, "rows": 3}),
        ]
        out = m._encode_audio_group(params)
        assert torch.equal(out[:, 0], torch.tensor([7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0]))

    def test_video_group_skips_audio_when_video_has_no_audio(self) -> None:
        # Video-only (no embedded audio) must not touch the sound encoder.
        m = _model_with_stubs()
        m._encode_audio = MagicMock()
        m._encode_video_group([_make_multimodal_params(video={"marker": 5, "rows": 4})])
        m._encode_audio.assert_not_called()

    def test_video_group_skips_audio_when_sound_encoder_missing(self) -> None:
        # Video carries embedded audio but the worker has no sound encoder —
        # the audio path must be skipped rather than crashing.
        m = _model_with_stubs()
        m.sound_encoder = None
        m._encode_audio = MagicMock()
        m._encode_video_group(
            [
                _make_multimodal_params(
                    video={
                        "marker": 5,
                        "rows": 4,
                        "audio": {"has_audio": [True], "audio_num_clips": torch.tensor([1])},
                    }
                )
            ]
        )
        m._encode_audio.assert_not_called()

    def test_video_group_interleaves_when_embedded_audio_present(self) -> None:
        # Video with an embedded audio stream must invoke both `_encode_audio`
        # and `_interleave_video_audio_embeddings` (the row-layout stitch that
        # matches the video item's `<img_context>*N <so_embedding>*M` prompt run).
        m = _model_with_stubs()
        m._encode_audio = MagicMock(return_value=[(_marker_tensor(0, 3), [3])])
        m._interleave_video_audio_embeddings = MagicMock(side_effect=lambda emb, *a, **kw: emb)
        m._encode_video_group(
            [
                _make_multimodal_params(
                    video={
                        "marker": 5,
                        "rows": 4,
                        "audio": {"has_audio": [True], "audio_num_clips": torch.tensor([1])},
                    }
                )
            ]
        )
        m._encode_audio.assert_called_once()
        m._interleave_video_audio_embeddings.assert_called_once()

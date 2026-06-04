# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen3VL multimodal item extractor and bucket adapters.

The extractor parses the two wire keys (`multimodal_item_order` +
`multimodal_embedding_lengths`) into one transient `MixedModalEncodeContext` via
`from_metadata`, walks `ctx.order`, and yields one canonical six-field
`ModalityItem` per prompt slot. A `ModalityItem` owns exactly one slot
(`prompt_pos`); its `rows` is both the encoder-output row count and its scatter
footprint. Per-item payload slicing (one image / one video out of an aggregate
`pixel_values` + `*_grid_thw` blob) goes through `_qwen3vl_slice_payload`; the
per-grid post-merge token count is `_qwen3vl_grid_rows`. When the extractor has a
`MixedModalEncodeContext`, rows come from `ctx.embedding_lengths`; the grid-row
helper is the fallback for pure single-modality requests that carry no
`multimodal_item_order` key (so `from_metadata` returns None and the extractor
synthesizes the modality-major default order).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from _mm_encode_helpers import _make_param

from tensorrt_llm._torch.models.mixed_modal_encode import ModalityItem
from tensorrt_llm._torch.models.modeling_qwen3vl import (
    Qwen3VisionModelBase,
    Qwen3VLInputProcessorBase,
    _qwen3vl_extract_items,
    _qwen3vl_grid_rows,
    _qwen3vl_slice_payload,
)
from tensorrt_llm.inputs.multimodal import MixedModalEncodeContext
from tensorrt_llm.sampling_params import SamplingParams


class TestQwen3VLPayloadSlicer:
    """The grid-row + slice helpers are the per-model source of truth for
    Qwen3VL: `_qwen3vl_slice_payload` slices the aggregate `pixel_values` (+
    `*_grid_thw`) blob into per-item encoder inputs by the raw-patch prefix sum
    `Sigma prod(grid_thw[:i])`, and `_qwen3vl_grid_rows` reports the per-grid
    post-merge token count `t * (h // merge) * (w // merge)`.
    """

    def test_rows_for_is_per_grid_post_merge_count(self):
        # Two image sub-items: grids [1,16,16] and [1,8,8] at merge=4 ->
        # 1*(16//4)*(16//4)=16 and 1*(8//4)*(8//4)=4 post-merge rows.
        payload = {
            "pixel_values": torch.randn(1 * 16 * 16 + 1 * 8 * 8, 1176),
            "image_grid_thw": torch.tensor([[1, 16, 16], [1, 8, 8]]),
        }
        param = _make_param({"image": payload})
        assert _qwen3vl_grid_rows(param, "image", 0, spatial_merge_size=4) == 16
        assert _qwen3vl_grid_rows(param, "image", 1, spatial_merge_size=4) == 4

    def test_slice_payload_slices_pixels_by_raw_patch_prefix_sum(self):
        # Raw-patch rows per grid = t*h*w: 256 then 64. The first item's pixel
        # slice is rows [0:256), the second is [256:320); each carries its own
        # single-row `*_grid_thw`.
        pv = torch.arange(320 * 2, dtype=torch.float32).reshape(320, 2)
        payload = {
            "pixel_values": pv,
            "image_grid_thw": torch.tensor([[1, 16, 16], [1, 8, 8]]),
        }
        param = _make_param({"image": payload})

        s0 = _qwen3vl_slice_payload(param, "image", 0, spatial_merge_size=4)
        s1 = _qwen3vl_slice_payload(param, "image", 1, spatial_merge_size=4)
        torch.testing.assert_close(s0["pixel_values"], pv[0:256])
        torch.testing.assert_close(s1["pixel_values"], pv[256:320])
        assert s0["image_grid_thw"].tolist() == [[1, 16, 16]]
        assert s1["image_grid_thw"].tolist() == [[1, 8, 8]]

    def test_slice_payload_video_uses_video_keys(self):
        pv = torch.arange(64 * 2, dtype=torch.float32).reshape(64, 2)
        payload = {
            "pixel_values_videos": pv,
            "video_grid_thw": torch.tensor([[1, 8, 8]]),
        }
        param = _make_param({"video": payload})

        s0 = _qwen3vl_slice_payload(param, "video", 0, spatial_merge_size=4)
        torch.testing.assert_close(s0["pixel_values_videos"], pv[0:64])
        assert s0["video_grid_thw"].tolist() == [[1, 8, 8]]
        # merge=4 -> 1*(8//4)*(8//4) = 4 post-merge rows.
        assert _qwen3vl_grid_rows(param, "video", 0, spatial_merge_size=4) == 4


class TestQwen3VLExtractItems:
    @pytest.mark.parametrize(
        "modality, payload, expected_rows",
        [
            (
                "image",
                {
                    # 20 patches -> 5 tokens at merge=4
                    "pixel_values": torch.randn(20, 1176),
                    "image_grid_thw": torch.tensor([[1, 16, 16]]),
                    "num_tokens": 5,  # explicit test convention, like Nano Task 7
                },
                5,
            ),
        ],
        ids=["image"],
    )
    def test_pure_single_modality(self, modality, payload, expected_rows):
        param = _make_param({modality: payload})
        items = list(_qwen3vl_extract_items(0, param))
        assert len(items) == 1
        assert items[0].modality == modality
        assert items[0].mm_idx_per_modality == 0
        assert items[0].prompt_pos == 0
        assert items[0].rows == expected_rows
        assert items[0].src_param_idx == 0

    def test_pure_single_modality_multi_item_enumerates_all(self):
        # A SINGLE modality (image) carrying TWO sub-items, with NO
        # `multimodal_item_order` key (a plain 2-image prompt on the
        # direct/non-hashing path). `from_metadata` returns None, so the
        # extractor synthesizes the modality-major default order. That default
        # must enumerate EVERY sub-item (one per `*_grid_thw` row), not collapse
        # to a single `(image, 0)` entry — otherwise the 2nd image is never
        # sliced or encoded.
        merge = 4
        # grids [1,16,16] -> 16 post-merge rows (256 raw patches) and
        #       [1, 8, 8] ->  4 post-merge rows ( 64 raw patches).
        payload = {
            "pixel_values": torch.randn(256 + 64, 1176),
            "image_grid_thw": torch.tensor([[1, 16, 16], [1, 8, 8]]),
        }
        param = _make_param({"image": payload})
        items = list(_qwen3vl_extract_items(0, param, spatial_merge_size=merge))
        assert len(items) == 2
        assert [(it.modality, it.mm_idx_per_modality, it.prompt_pos) for it in items] == [
            ("image", 0, 0),
            ("image", 1, 1),
        ]
        # Per-grid post-merge row counts (grid-driven fallback, no context).
        assert [it.rows for it in items] == [16, 4]
        # Each item carries its own sliced single-grid payload, so the bucket
        # adapter re-concatenation reproduces the original aggregate.
        assert items[0].payload["image_grid_thw"].tolist() == [[1, 16, 16]]
        assert items[0].payload["pixel_values"].shape[0] == 256
        assert items[1].payload["image_grid_thw"].tolist() == [[1, 8, 8]]
        assert items[1].payload["pixel_values"].shape[0] == 64

    @pytest.mark.parametrize(
        "item_order",
        [
            # Tuple(pair)-form order entries.
            [("video", 0), ("image", 0)],
            # Dict-form order entries (as the runtime registry emits them).
            # Regression guard for the tuple(pair) bug: the extractor must
            # normalize dict-form order identically to tuple-form.
            [{"modality": "video", "index": 0}, {"modality": "image", "index": 0}],
        ],
        ids=["tuple_form", "dict_form_regression"],
    )
    def test_mixed_image_video(self, item_order):
        payload_image = {
            "pixel_values": torch.randn(20, 1176),
            "image_grid_thw": torch.tensor([[1, 16, 16]]),
            "num_tokens": 5,
        }
        payload_video = {
            "pixel_values_videos": torch.randn(32, 1176),
            "video_grid_thw": torch.tensor([[2, 16, 16]]),
            "num_tokens": 8,
        }
        param = _make_param(
            {
                "image": payload_image,
                "video": payload_video,
                "multimodal_item_order": item_order,
                "multimodal_embedding_lengths": [8, 5],
            }
        )
        items = list(_qwen3vl_extract_items(0, param))
        assert len(items) == 2
        # video at prompt slot 0, image at slot 1 — distinct, no collapse.
        by_modality = {it.modality: it for it in items}
        assert by_modality["video"].prompt_pos == 0
        assert by_modality["image"].prompt_pos == 1
        assert by_modality["video"].mm_idx_per_modality == 0
        assert by_modality["image"].mm_idx_per_modality == 0
        assert by_modality["video"].rows == 8
        assert by_modality["image"].rows == 5

    def test_image_video_image_row_order(self):
        # Interleaved repeated modality: image -> video -> image. The trailing
        # image (prompt slot 2) must land AFTER the video (slot 1), not folded
        # into the leading image block. Each image sub-item is sliced out of the
        # aggregate `pixel_values`/`image_grid_thw` by grid; `rows` is the
        # per-grid post-merge count (grid-driven, the Qwen3VL source of truth).
        merge = 4
        # image grids: [1,16,16] -> 16 post-merge rows (256 raw patches);
        #              [1, 8, 8] ->  4 post-merge rows ( 64 raw patches).
        image_payload = {
            "pixel_values": torch.randn(256 + 64, 1176),
            "image_grid_thw": torch.tensor([[1, 16, 16], [1, 8, 8]]),
        }
        # video grid: [2,16,16] -> 2*(16//4)*(16//4) = 32 post-merge rows
        #             (2*16*16 = 512 raw patches).
        video_payload = {
            "pixel_values_videos": torch.randn(512, 1176),
            "video_grid_thw": torch.tensor([[2, 16, 16]]),
        }
        param = _make_param(
            {
                "image": image_payload,
                "video": video_payload,
                "multimodal_item_order": [
                    {"modality": "image", "index": 0},
                    {"modality": "video", "index": 0},
                    {"modality": "image", "index": 1},
                ],
                # The transient MixedModalEncodeContext requires the per-slot row
                # counts alongside the order (length-agreement is validated in
                # `__post_init__`). Rows still come from the grid via `rows_for`
                # until Task 5b sources them from the context; these match.
                "multimodal_embedding_lengths": [16, 32, 4],
            }
        )
        items = list(_qwen3vl_extract_items(0, param, spatial_merge_size=merge))
        assert len(items) == 3
        # Items are emitted in prompt order, one per slot.
        assert [(it.modality, it.mm_idx_per_modality, it.prompt_pos) for it in items] == [
            ("image", 0, 0),
            ("video", 0, 1),
            ("image", 1, 2),
        ]
        # Per-grid post-merge row counts.
        assert [it.rows for it in items] == [16, 32, 4]
        # The trailing image carries its own sliced single-grid payload (not the
        # leading image's), so the bucket adapter re-concatenation is faithful.
        assert items[2].payload["image_grid_thw"].tolist() == [[1, 8, 8]]
        assert items[2].payload["pixel_values"].shape[0] == 64
        assert items[0].payload["image_grid_thw"].tolist() == [[1, 16, 16]]
        assert items[0].payload["pixel_values"].shape[0] == 256

    def test_mixed_order_length_mismatch_raises(self):
        # The extractor now builds a transient MixedModalEncodeContext from the
        # two wire keys, so a `multimodal_item_order` whose length disagrees with
        # `multimodal_embedding_lengths` is rejected at construction time (the
        # context's `__post_init__` length-agreement check). The old
        # `MultimodalPromptOrder.from_metadata` path ignored the lengths entirely
        # and would have yielded items, so this guards the substitution.
        payload_image = {
            "pixel_values": torch.randn(20, 1176),
            "image_grid_thw": torch.tensor([[1, 16, 16]]),
            "num_tokens": 5,
        }
        payload_video = {
            "pixel_values_videos": torch.randn(32, 1176),
            "video_grid_thw": torch.tensor([[2, 16, 16]]),
            "num_tokens": 8,
        }
        param = _make_param(
            {
                "image": payload_image,
                "video": payload_video,
                # 2-entry order, 1-entry lengths -> the typed view rejects it.
                "multimodal_item_order": [("video", 0), ("image", 0)],
                "multimodal_embedding_lengths": [8],
            }
        )
        with pytest.raises(ValueError, match="embedding_lengths"):
            list(_qwen3vl_extract_items(0, param))


class TestQwen3VLBucketAdapters:
    def _make_encoder_stub(self, encode_visual_inputs_return):
        enc = MagicMock(spec=Qwen3VisionModelBase)
        enc._encode_visual_inputs = MagicMock(return_value=encode_visual_inputs_return)
        enc._image_encoder_adapter = Qwen3VisionModelBase._image_encoder_adapter.__get__(enc)
        enc._video_encoder_adapter = Qwen3VisionModelBase._video_encoder_adapter.__get__(enc)
        return enc

    @pytest.mark.parametrize(
        "adapter_attr, items, expected_grid_shape",
        [
            (
                # Image bucket: two items stack into one (32, 1176) call with a
                # (2, 3) grid (20 + 12 patches across two grid rows).
                "_image_encoder_adapter",
                [
                    ModalityItem(
                        0,
                        "image",
                        0,
                        0,
                        5,
                        {
                            "pixel_values": torch.randn(20, 1176),
                            "image_grid_thw": torch.tensor([[1, 16, 16]]),
                            "num_tokens": 5,
                        },
                    ),
                    ModalityItem(
                        1,
                        "image",
                        0,
                        1,
                        3,
                        {
                            "pixel_values": torch.randn(12, 1176),
                            "image_grid_thw": torch.tensor([[1, 12, 12]]),
                            "num_tokens": 3,
                        },
                    ),
                ],
                (2, 3),
            ),
            (
                # Video bucket: one item -> (32, 1176) call with a (1, 3) grid.
                "_video_encoder_adapter",
                [
                    ModalityItem(
                        0,
                        "video",
                        0,
                        0,
                        8,
                        {
                            "pixel_values_videos": torch.randn(32, 1176),
                            "video_grid_thw": torch.tensor([[2, 16, 16]]),
                            "num_tokens": 8,
                        },
                    ),
                ],
                (1, 3),
            ),
        ],
        ids=["image", "video"],
    )
    def test_adapter_stacks_pixel_values_and_grids(self, adapter_attr, items, expected_grid_shape):
        H = 1024
        out_tensor = torch.randn(8, H)  # total tokens across the bucket
        enc = self._make_encoder_stub(out_tensor)
        result = getattr(enc, adapter_attr)(items, [object()] * len(items))
        # _encode_visual_inputs is called once with the concatenated tensors.
        call_args = enc._encode_visual_inputs.call_args[0]
        pixel_values_arg, grid_arg = call_args[0], call_args[1]
        assert pixel_values_arg.shape == (32, 1176)
        assert grid_arg.shape == expected_grid_shape
        torch.testing.assert_close(result, out_tensor)


# --- Mixed image+video preprocess writes embedding lengths (non-hashing path) --

_QWEN_VOCAB_SIZE = 1000
_QWEN_IMAGE_TOKEN_ID = 10
_QWEN_VIDEO_TOKEN_ID = 11
# spatial_merge_size = 2. Image grid [1, 4, 4] -> 1*(4//2)*(4//2) = 4 tokens.
# Video grid [1, 2, 2] -> 1*(2//2)*(2//2) = 1 token.
_QWEN_IMAGE_GRID = [[1, 4, 4]]
_QWEN_VIDEO_GRID = [[1, 2, 2]]
_QWEN_IMAGE_TOKENS = 4
_QWEN_VIDEO_TOKENS = 1


def _make_mixed_qwen_processor():
    """Construct a Qwen3VL input processor that runs `call_with_text_prompt`
    without model weights.

    `__init__` loads an HF processor/tokenizer, so build via `__new__` and set
    only the attributes the mixed preprocess path reads, stubbing `_preprocess`
    (the HF call) and `get_mrope_config` (rope math, irrelevant to the lengths
    metadata under test). The image token-count hook is stubbed to the grid
    formula too, since the base default delegates to the HF processor.
    """
    proc = Qwen3VLInputProcessorBase.__new__(Qwen3VLInputProcessorBase)
    proc._config = SimpleNamespace(
        text_config=SimpleNamespace(vocab_size=_QWEN_VOCAB_SIZE, dtype=torch.float32),
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=_QWEN_IMAGE_TOKEN_ID,
        video_token_id=_QWEN_VIDEO_TOKEN_ID,
    )
    proc._dtype = torch.float32
    proc._processor = None
    proc.tllm_multimodal_token_id = proc.get_vocab_size() + 1

    # Prompt with image first then video, so the scanned item order is
    # [(image, 0), (video, 0)].
    text_prompt = (
        "<|vision_start|><|image_pad|><|vision_end|> <|vision_start|><|video_pad|><|vision_end|>"
    )
    # Stubbed HF output: input_ids carry one image token then one video token
    # (placeholders, pre-expansion is irrelevant — the embed mask keys off the
    # post-_postprocess tllm_multimodal_token_id). The image block is 4 tokens
    # and the video block is 1 token, matching the grid token counts.
    input_ids = torch.tensor(
        [
            [
                5,
                *([_QWEN_IMAGE_TOKEN_ID] * _QWEN_IMAGE_TOKENS),
                6,
                *([_QWEN_VIDEO_TOKEN_ID] * _QWEN_VIDEO_TOKENS),
                7,
            ]
        ]
    )
    processed = {
        "input_ids": input_ids,
        "pixel_values": torch.zeros(_QWEN_IMAGE_TOKENS, 1),
        "image_grid_thw": torch.tensor(_QWEN_IMAGE_GRID),
        "pixel_values_videos": torch.zeros(_QWEN_VIDEO_TOKENS, 1),
        "video_grid_thw": torch.tensor(_QWEN_VIDEO_GRID),
        "attention_mask": torch.ones_like(input_ids),
    }
    proc._preprocess = lambda text, mm_data, mm_processor_kwargs: processed
    proc.get_mrope_config = lambda *args, **kwargs: {}
    # The base get_num_tokens_per_image delegates to the HF processor; stub it to
    # the grid formula so it works weightless. get_num_tokens_per_video already
    # computes from the grid (no HF call needed).
    proc.get_num_tokens_per_image = lambda *, image, **kwargs: _QWEN_IMAGE_TOKENS
    return proc, text_prompt


def test_qwen3vl_mixed_preprocess_writes_matching_embedding_lengths():
    """Qwen3-VL mixed preprocess must bake multimodal_embedding_lengths.

    Regression for the mixed-modality crash: the `len(modalities) > 1` branch
    wrote `multimodal_item_order` but never `multimodal_embedding_lengths`, so on
    the non-hashing (direct preprocess) path nobody filled it. The per-item
    extractor then built a `MixedModalEncodeContext` via
    `from_metadata(..., multimodal_data.get("multimodal_embedding_lengths"))`,
    hitting `tuple(int(x) for x in None)` -> TypeError. The fix makes the
    preprocess path emit the SAME metadata the registry hashing path does.
    """
    proc, text_prompt = _make_mixed_qwen_processor()
    inputs = {
        "prompt": text_prompt,
        "multi_modal_data": {
            "image": [torch.zeros(3, 8, 8)],
            "video": [[torch.zeros(3, 4, 4)]],
        },
    }

    _, extra = proc.call_with_text_prompt(inputs, SamplingParams())
    multimodal_data = extra["multimodal_data"]

    # The mixed branch baked the prompt order: image first, then video.
    assert multimodal_data["multimodal_item_order"] == [
        {"modality": "image", "index": 0},
        {"modality": "video", "index": 0},
    ]
    # The crash fix: per-item embedding lengths are present and in prompt order
    # (image=4, then video=1). Embedding lengths are the embed-slot counts, which
    # equal the per-item token budgets here (no framing specials on Qwen3-VL).
    assert multimodal_data["multimodal_embedding_lengths"] == [
        _QWEN_IMAGE_TOKENS,
        _QWEN_VIDEO_TOKENS,
    ]
    # The lengths agree with what a transient MixedModalEncodeContext expects
    # (per-slot row counts aligned with the order) — i.e. the context the
    # extractor builds via from_metadata constructs without error.
    ctx = MixedModalEncodeContext.from_metadata(
        multimodal_data, multimodal_data["multimodal_embedding_lengths"]
    )
    assert ctx is not None
    assert ctx.embedding_lengths == (_QWEN_IMAGE_TOKENS, _QWEN_VIDEO_TOKENS)

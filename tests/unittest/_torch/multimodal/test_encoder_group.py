# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the generic encoder-group primitives.

Covers the pure helpers behind ``encode_multimodal_by_groups``:

* ``_lengths_by_modality`` — invert prompt-ordered ``multimodal_embedding_lengths``
  into per-modality per-item lists using ``mm_item_order``.
* ``_synthesize_single_modality_manifest`` — trivial manifest for single-modality
  requests that don't carry an explicit one.
* ``_reorder_embeds_by_manifest`` — slice per-modality tensors and concat in
  each request's prompt-order manifest.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_multimodal_mixin import (
    _lengths_by_modality,
    _reorder_embeds_by_manifest,
    _synthesize_single_modality_manifest,
)


def _mp(*, mm_item_order=None, embedding_lengths=None, buckets=None):
    """Minimal MultimodalParams stand-in — the helpers only touch
    ``multimodal_data`` and ``mm_item_order``."""
    data = dict(buckets or {})
    if embedding_lengths is not None:
        data["multimodal_embedding_lengths"] = embedding_lengths
    return SimpleNamespace(multimodal_data=data, mm_item_order=mm_item_order)


class TestLengthsByModality:
    def test_mixed_manifest_image_video_image(self):
        # Manifest order: image#0 (4 rows), video#0 (6), image#1 (2).
        mp = _mp(
            mm_item_order=[
                {"modality": "image", "index": 0},
                {"modality": "video", "index": 0},
                {"modality": "image", "index": 1},
            ],
            embedding_lengths=[4, 6, 2],
            buckets={"image": {}, "video": {}},
        )
        assert _lengths_by_modality([mp], ("image", "video")) == {
            "image": [4, 2],
            "video": [6],
        }

    def test_single_modality_no_manifest(self):
        # No mm_item_order; flat list is already in per-modality order.
        mp = _mp(embedding_lengths=[64, 64], buckets={"image": {}})
        assert _lengths_by_modality([mp], ("image", "video")) == {
            "image": [64, 64],
            "video": [],
        }

    def test_length_mismatch_raises(self):
        # ``strict=True`` inside the helper catches manifest/length divergence.
        mp = _mp(
            mm_item_order=[
                {"modality": "image", "index": 0},
                {"modality": "video", "index": 0},
            ],
            embedding_lengths=[4],  # one short
            buckets={"image": {}, "video": {}},
        )
        with pytest.raises(ValueError):
            _lengths_by_modality([mp], ("image", "video"))


class TestSynthesizeSingleModalityManifest:
    def test_image_only(self):
        mp = _mp(embedding_lengths=[10, 20, 30], buckets={"image": {}})
        assert _synthesize_single_modality_manifest(mp, ("image", "video")) == [
            {"modality": "image", "index": 0},
            {"modality": "image", "index": 1},
            {"modality": "image", "index": 2},
        ]

    def test_no_group_modality_returns_empty(self):
        # Request has audio but the group only covers image/video.
        mp = _mp(embedding_lengths=[10], buckets={"audio": {}})
        assert _synthesize_single_modality_manifest(mp, ("image", "video")) == []


class TestReorderEmbedsByManifest:
    @staticmethod
    def _marker_tensor(marker, rows, hidden=2):
        # Rows all share a marker so the reorder can be asserted by column-0.
        return torch.full((rows, hidden), float(marker))

    def test_mixed_image_video_image_in_one_request(self):
        # Prompt: image#0 (4 rows, marker 10), video#0 (6, 20), image#1 (2, 30).
        mp = _mp(
            mm_item_order=[
                {"modality": "image", "index": 0},
                {"modality": "video", "index": 0},
                {"modality": "image", "index": 1},
            ],
            buckets={"image": {}, "video": {}},
        )
        # Per-modality tensors are cat'd in encounter order — image items 0
        # and 1 concatenated, then video item 0.
        per_modality_embeds = {
            "image": torch.cat([self._marker_tensor(10, 4), self._marker_tensor(30, 2)], dim=0),
            "video": self._marker_tensor(20, 6),
        }
        per_modality_lengths = {"image": [4, 2], "video": [6]}
        out = _reorder_embeds_by_manifest([mp], per_modality_embeds, per_modality_lengths)
        # Expected column-0: 10 (×4), 20 (×6), 30 (×2) in prompt order.
        expected = torch.tensor([10.0] * 4 + [20.0] * 6 + [30.0] * 2)
        assert torch.equal(out[:, 0], expected)

    def test_cross_request_cursors_advance_per_modality(self):
        # Request A: [image#0 (2)]; Request B: [video#0 (3), image#0 (1)].
        # Global per-modality indexing: image=[A#0, B#0], video=[B#0].
        # Manifests use per-request indices — cursor must translate them.
        mp_a = _mp(
            mm_item_order=[{"modality": "image", "index": 0}],
            buckets={"image": {}},
        )
        mp_b = _mp(
            mm_item_order=[
                {"modality": "video", "index": 0},
                {"modality": "image", "index": 0},
            ],
            buckets={"image": {}, "video": {}},
        )
        per_modality_embeds = {
            "image": torch.cat([self._marker_tensor(1, 2), self._marker_tensor(3, 1)], dim=0),
            "video": self._marker_tensor(2, 3),
        }
        per_modality_lengths = {"image": [2, 1], "video": [3]}
        out = _reorder_embeds_by_manifest([mp_a, mp_b], per_modality_embeds, per_modality_lengths)
        # A: image#0 → marker 1 (×2). B: video#0 → 2 (×3), then image#0 → 3 (×1).
        expected = torch.tensor([1.0] * 2 + [2.0] * 3 + [3.0] * 1)
        assert torch.equal(out[:, 0], expected)

    def test_single_modality_falls_back_to_synthesized_manifest(self):
        # Two image items, no explicit manifest — reorder still works by
        # synthesizing a trivial per-modality manifest from the request's
        # multimodal_embedding_lengths.
        mp = _mp(embedding_lengths=[2, 3], buckets={"image": {}})
        per_modality_embeds = {
            "image": torch.cat([self._marker_tensor(7, 2), self._marker_tensor(9, 3)], dim=0),
        }
        per_modality_lengths = {"image": [2, 3]}
        out = _reorder_embeds_by_manifest([mp], per_modality_embeds, per_modality_lengths)
        expected = torch.tensor([7.0] * 2 + [9.0] * 3)
        assert torch.equal(out[:, 0], expected)

    def test_empty_bookkeeping_returns_typed_empty(self):
        # Mirrors the executor KV-cache profiling pass: the dummy batch runs the
        # encoder but carries no ``multimodal_embedding_lengths``, so per-modality
        # lengths are empty and the sliced embeds are zero-row. Reorder must
        # return a correctly-typed empty tensor, not crash on ``torch.cat([])``.
        mp = _mp(buckets={"image": {}})  # modality present, no manifest, no lengths
        hidden = 5
        per_modality_embeds = {"image": torch.empty((0, hidden), dtype=torch.float16)}
        per_modality_lengths = {"image": []}
        out = _reorder_embeds_by_manifest([mp], per_modality_embeds, per_modality_lengths)
        assert out.shape == (0, hidden)
        assert out.dtype == torch.float16

    def test_no_embeds_returns_empty(self):
        # Defensive: no group produced embeddings at all — return an empty
        # tensor rather than crashing.
        mp = _mp(buckets={})
        out = _reorder_embeds_by_manifest([mp], {}, {})
        assert out.numel() == 0

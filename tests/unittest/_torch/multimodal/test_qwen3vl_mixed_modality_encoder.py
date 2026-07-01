# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Encoder-level tests for Qwen3-VL mixed-modality interleaving.

Exercises `Qwen3VisionModelBase._parse_and_batch_multimodal_data` and
`forward` directly by constructing an instance without invoking __init__ and
stubbing `self.visual` with a marker function whose output row values encode
input row identities. This lets us assert that mixed image+video requests
produce a single ViT call whose output rows arrive in prompt order.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VisionModelBase


class _FakeVisual:
    """Marker ViT that returns a single-tensor pair mimicking Qwen3VL's
    (image_embeds, deepstack_list) contract. Row i of the output is a
    length-`hidden` tensor filled with a marker computed from the input
    row's marker (via pooling `patches_per_item` rows and preserving order).
    """

    def __init__(self, hidden=4, deepstack_layers=0):
        self.hidden = hidden
        self.deepstack_layers = deepstack_layers

    def __call__(self, pixel_values, grid_thw=None):
        # `pixel_values` is [P_total, marker_dim]; each row's first column
        # holds an integer identity marker. The "encoder" pools each item's
        # patches into a single row per item (grid_thw specifies item extent).
        assert grid_thw is not None
        out_rows = []
        offset = 0
        markers = pixel_values[:, 0].tolist()
        for thw in grid_thw.tolist():
            n = int(thw[0] * thw[1] * thw[2])
            # Use the first patch's marker as the item's identity.
            item_marker = markers[offset]
            offset += n
            # Emit ONE row per patch (real ViT output-length = P_total).
            out_rows.extend([item_marker] * n)
        base = torch.tensor(out_rows, dtype=torch.float32).unsqueeze(1).repeat(1, self.hidden)
        deepstack = [torch.zeros_like(base) for _ in range(self.deepstack_layers)]
        return base, deepstack


def _make_encoder():
    enc = object.__new__(Qwen3VisionModelBase)
    enc.model_dtype = torch.float32
    enc.visual = _FakeVisual(hidden=4, deepstack_layers=0)
    return enc


def _make_param(image=None, video=None, order=None):
    data = {}
    if image is not None:
        data["image"] = image
    if video is not None:
        data["video"] = video
    if order is not None:
        data["mm_item_order"] = order
    return SimpleNamespace(multimodal_data=data)


def _make_item_patches(marker: int, patches: int, dim: int = 2):
    # Row 0's first column encodes the item's marker.
    t = torch.zeros((patches, dim), dtype=torch.float32)
    t[:, 0] = float(marker)
    return t


def test_image_only_single_modality_fallback_unchanged():
    enc = _make_encoder()
    params = [
        _make_param(
            image={
                "pixel_values": _make_item_patches(marker=10, patches=4),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            },
        )
    ]
    embeds = enc.forward(params)
    assert len(embeds) == 1
    # 4 patches, all marker 10.
    assert embeds[0].shape == (4, 4)
    assert torch.equal(embeds[0][:, 0], torch.full((4,), 10.0))


def test_video_only_single_modality_fallback_unchanged():
    enc = _make_encoder()
    params = [
        _make_param(
            video={
                "pixel_values_videos": _make_item_patches(marker=20, patches=6),
                "video_grid_thw": torch.tensor([[2, 1, 3]]),
            },
        )
    ]
    embeds = enc.forward(params)
    assert len(embeds) == 1
    assert embeds[0].shape == (6, 4)
    assert torch.equal(embeds[0][:, 0], torch.full((6,), 20.0))


def test_mixed_image_video_image_prompt_order_interleave():
    enc = _make_encoder()
    # Prompt order: image#0 (marker=10, 4 patches), video#0 (marker=20, 6
    # patches), image#1 (marker=30, 2 patches).
    params = [
        _make_param(
            image={
                "pixel_values": torch.cat(
                    [
                        _make_item_patches(marker=10, patches=4),
                        _make_item_patches(marker=30, patches=2),
                    ],
                    dim=0,
                ),
                "image_grid_thw": torch.tensor([[1, 2, 2], [1, 1, 2]]),
            },
            video={
                "pixel_values_videos": _make_item_patches(marker=20, patches=6),
                "video_grid_thw": torch.tensor([[2, 1, 3]]),
            },
            order=[
                {"modality": "image", "index": 0},
                {"modality": "video", "index": 0},
                {"modality": "image", "index": 1},
            ],
        )
    ]
    embeds = enc.forward(params)
    assert len(embeds) == 1
    # 4 + 6 + 2 = 12 rows in prompt order.
    expected = torch.tensor([10.0] * 4 + [20.0] * 6 + [30.0] * 2, dtype=torch.float32)
    assert embeds[0].shape == (12, 4)
    assert torch.equal(embeds[0][:, 0], expected)


def test_video_then_image_interleave():
    enc = _make_encoder()
    params = [
        _make_param(
            image={
                "pixel_values": _make_item_patches(marker=99, patches=3),
                "image_grid_thw": torch.tensor([[1, 1, 3]]),
            },
            video={
                "pixel_values_videos": _make_item_patches(marker=77, patches=4),
                "video_grid_thw": torch.tensor([[1, 2, 2]]),
            },
            order=[
                {"modality": "video", "index": 0},
                {"modality": "image", "index": 0},
            ],
        )
    ]
    embeds = enc.forward(params)
    assert len(embeds) == 1
    expected = torch.tensor([77.0] * 4 + [99.0] * 3, dtype=torch.float32)
    assert torch.equal(embeds[0][:, 0], expected)


def test_mixed_without_manifest_raises_tripwire():
    """Guard: if both modalities show up but mm_item_order is absent the
    encoder must NOT silently produce misordered embeddings."""
    enc = _make_encoder()
    params = [
        _make_param(
            image={
                "pixel_values": _make_item_patches(marker=1, patches=1),
                "image_grid_thw": torch.tensor([[1, 1, 1]]),
            },
            video={
                "pixel_values_videos": _make_item_patches(marker=2, patches=1),
                "video_grid_thw": torch.tensor([[1, 1, 1]]),
            },
            # order deliberately omitted
        )
    ]
    with pytest.raises(ValueError, match="mm_item_order"):
        enc.forward(params)


def test_batch_of_two_requests_single_modality_each():
    """Batching still works when each request in the batch is
    single-modality (no manifest needed, no interleave path)."""
    enc = _make_encoder()
    params = [
        _make_param(
            image={
                "pixel_values": _make_item_patches(marker=1, patches=2),
                "image_grid_thw": torch.tensor([[1, 1, 2]]),
            }
        ),
        _make_param(
            image={
                "pixel_values": _make_item_patches(marker=2, patches=3),
                "image_grid_thw": torch.tensor([[1, 1, 3]]),
            }
        ),
    ]
    embeds = enc.forward(params)
    assert len(embeds) == 1
    expected = torch.tensor([1.0] * 2 + [2.0] * 3, dtype=torch.float32)
    assert torch.equal(embeds[0][:, 0], expected)

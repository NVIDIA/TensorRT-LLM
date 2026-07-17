# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Encoder-level tests for Qwen3-VL through the generic mm-encoder-group path.

Constructs a minimal ``MultimodalModelMixin`` with the Qwen3-VL
``EncoderGroup`` and a stub ``encode_batched`` marker function. Asserts that
mixed image+video requests, heterogeneous single-modality batches, and
cross-request batching all produce prompt-order embeddings via one ViT call.
"""

from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from tensorrt_llm._torch.models.modeling_multimodal_mixin import EncoderGroup, MultimodalModelMixin
from tensorrt_llm._torch.models.modeling_qwen3vl import _qwen3vl_build_batched_input


def _encode_batched(pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
    """Marker encoder: one output row per input patch, marker propagates via
    column 0. Emulates a modality-blind ViT with no compression."""
    return pixel_values.clone().repeat(1, 2)  # widen to hidden=marker_dim*2


class _StubModel(MultimodalModelMixin):
    def __init__(self) -> None:
        self.mm_encoder_groups = (
            EncoderGroup(
                modalities=("image", "video"),
                encoder_fn=_encode_batched,
                build_batched_input=_qwen3vl_build_batched_input,
            ),
        )


def _param(
    image: Optional[dict] = None,
    video: Optional[dict] = None,
    order: Optional[list] = None,
    lengths: Optional[list] = None,
) -> SimpleNamespace:
    data: dict = {}
    if image is not None:
        data["image"] = image
    if video is not None:
        data["video"] = video
    if lengths is not None:
        data["multimodal_embedding_lengths"] = lengths
    return SimpleNamespace(multimodal_data=data, mm_item_order=order)


def _patches(marker: int, n: int, dim: int = 2) -> torch.Tensor:
    t = torch.zeros((n, dim), dtype=torch.float32)
    t[:, 0] = float(marker)
    return t


def _encode(params: list) -> torch.Tensor:
    return _StubModel().encode_multimodal_by_groups(params)


def test_image_only_request() -> None:
    out = _encode(
        [
            _param(
                image={
                    "pixel_values": _patches(10, 4),
                    "image_grid_thw": torch.tensor([[1, 2, 2]]),
                },
                lengths=[4],
            ),
        ]
    )
    assert torch.equal(out[:, 0], torch.full((4,), 10.0))


def test_mixed_image_video_image_prompt_order() -> None:
    # Prompt: image#0 (4 rows), video#0 (6), image#1 (2).
    out = _encode(
        [
            _param(
                image={
                    "pixel_values": torch.cat([_patches(10, 4), _patches(30, 2)], dim=0),
                    "image_grid_thw": torch.tensor([[1, 2, 2], [1, 1, 2]]),
                },
                video={
                    "pixel_values_videos": _patches(20, 6),
                    "video_grid_thw": torch.tensor([[2, 1, 3]]),
                },
                order=[
                    {"modality": "image", "index": 0},
                    {"modality": "video", "index": 0},
                    {"modality": "image", "index": 1},
                ],
                lengths=[4, 6, 2],
            ),
        ]
    )
    expected = torch.tensor([10.0] * 4 + [20.0] * 6 + [30.0] * 2)
    assert torch.equal(out[:, 0], expected)


def test_mixed_without_manifest_raises() -> None:
    """Mixed request must carry mm_item_order; the shared helper enforces this
    once for every model that registers a multi-modality EncoderGroup."""
    with pytest.raises(ValueError, match="mm_item_order"):
        _encode(
            [
                _param(
                    image={
                        "pixel_values": _patches(1, 1),
                        "image_grid_thw": torch.tensor([[1, 1, 1]]),
                    },
                    video={
                        "pixel_values_videos": _patches(2, 1),
                        "video_grid_thw": torch.tensor([[1, 1, 1]]),
                    },
                    lengths=[1, 1],
                    # order deliberately omitted
                )
            ]
        )


def test_batch_of_two_image_only_requests() -> None:
    out = _encode(
        [
            _param(
                image={
                    "pixel_values": _patches(1, 2),
                    "image_grid_thw": torch.tensor([[1, 1, 2]]),
                },
                lengths=[2],
            ),
            _param(
                image={
                    "pixel_values": _patches(2, 3),
                    "image_grid_thw": torch.tensor([[1, 1, 3]]),
                },
                lengths=[3],
            ),
        ]
    )
    expected = torch.tensor([1.0] * 2 + [2.0] * 3)
    assert torch.equal(out[:, 0], expected)


def test_heterogeneous_batch_image_only_and_video_only() -> None:
    """One image-only request + one video-only request go through the same
    modality-blind ViT via the generic path — no per-request manifest needed
    because neither individual request is mixed."""
    out = _encode(
        [
            _param(
                image={
                    "pixel_values": _patches(1, 2),
                    "image_grid_thw": torch.tensor([[1, 1, 2]]),
                },
                lengths=[2],
            ),
            _param(
                video={
                    "pixel_values_videos": _patches(2, 3),
                    "video_grid_thw": torch.tensor([[1, 1, 3]]),
                },
                lengths=[3],
            ),
        ]
    )
    expected = torch.tensor([1.0] * 2 + [2.0] * 3)
    assert torch.equal(out[:, 0], expected)


def test_raw_inputs_route_through_encode_multimodal_by_groups() -> None:
    """Regression: ``Qwen3VLModelBase.forward``'s raw image/video branch
    routes through ``encode_multimodal_by_groups`` (via
    ``get_multimodal_embeddings``). Confirms the callable used there
    accepts a bare ``multimodal_params`` list and returns the same
    prompt-order embedding tensor that direct invocation produces."""
    stub = _StubModel()
    params = [
        _param(
            image={
                "pixel_values": _patches(11, 3),
                "image_grid_thw": torch.tensor([[1, 1, 3]]),
            },
            video={
                "pixel_values_videos": _patches(22, 5),
                "video_grid_thw": torch.tensor([[1, 1, 5]]),
            },
            order=[
                {"modality": "video", "index": 0},
                {"modality": "image", "index": 0},
            ],
            lengths=[5, 3],
        ),
    ]
    # Same callable signature `get_multimodal_embeddings` invokes at
    # `modeling_qwen3vl.py`'s raw-MM branch.
    out = stub.encode_multimodal_by_groups(params)
    expected = torch.tensor([22.0] * 5 + [11.0] * 3)
    assert torch.equal(out[:, 0], expected)


def test_mixed_request_alongside_ordered_one_still_rejects_unordered() -> None:
    """Even when a sibling request carries a valid manifest, a mixed request
    without its own manifest must still raise — validation is per-request."""
    with pytest.raises(ValueError, match="mm_item_order"):
        _encode(
            [
                _param(
                    image={
                        "pixel_values": _patches(10, 2),
                        "image_grid_thw": torch.tensor([[1, 1, 2]]),
                    },
                    video={
                        "pixel_values_videos": _patches(20, 2),
                        "video_grid_thw": torch.tensor([[1, 1, 2]]),
                    },
                    order=[
                        {"modality": "image", "index": 0},
                        {"modality": "video", "index": 0},
                    ],
                    lengths=[2, 2],
                ),
                _param(
                    image={
                        "pixel_values": _patches(30, 1),
                        "image_grid_thw": torch.tensor([[1, 1, 1]]),
                    },
                    video={
                        "pixel_values_videos": _patches(40, 1),
                        "video_grid_thw": torch.tensor([[1, 1, 1]]),
                    },
                    lengths=[1, 1],
                    # No manifest on this request even though it has both modalities.
                ),
            ]
        )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `MultimodalEncoderMixin`.

Verifies that the mixin's default `setup_attn_metadata` builds the encoder's
AttentionMetadata exactly once with the item/token budgets injected by the
engine and `kv_cache_manager=None`. The default mapping floors
`max_num_requests` at the encoder fallback because one attention segment per
vision tile can exceed the item count. These tests use a stub `metadata_cls` to
avoid pulling in any real attention backend (and any GPU/CUDA dependency).
"""

from types import SimpleNamespace

import torch.nn as nn

from tensorrt_llm._torch.models.modeling_multimodal_encoder import (
    _ENCODER_FALLBACK_MAX_NUM_REQUESTS,
    MultimodalEncoderMixin,
)
from tensorrt_llm._torch.models.modeling_pixtral import PixtralVisionModel
from tensorrt_llm._torch.models.modeling_qwen2vl import Qwen2_5_VisionModel
from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VisionModel


class _StubMetadata:
    """Minimal stand-in for an AttentionMetadata constructor."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummyEncoder(nn.Module, MultimodalEncoderMixin):
    def __init__(self):
        super().__init__()
        self.metadata_cls = _StubMetadata


def test_attn_metadata_is_none_before_setup():
    encoder = _DummyEncoder()
    assert encoder.attn_metadata is None


def test_setup_attn_metadata_builds_with_engine_sizes():
    """An item count above the encoder fallback passes through unchanged."""
    encoder = _DummyEncoder()
    big = _ENCODER_FALLBACK_MAX_NUM_REQUESTS + 100
    encoder.setup_attn_metadata(max_num_items=big, max_num_tokens=1234)

    assert isinstance(encoder.attn_metadata, _StubMetadata)
    assert encoder.attn_metadata.kwargs == {
        "max_num_requests": big,
        "max_num_tokens": 1234,
        "kv_cache_manager": None,
    }


def test_setup_attn_metadata_floors_small_item_count():
    """An item count below the encoder fallback is floored to it: the encoder
    runs one attention segment per vision tile, which can exceed the item
    count, so the per-segment buffers must not undersize."""
    encoder = _DummyEncoder()
    encoder.setup_attn_metadata(max_num_items=8, max_num_tokens=100)

    assert encoder.attn_metadata.kwargs == {
        "max_num_requests": _ENCODER_FALLBACK_MAX_NUM_REQUESTS,
        "max_num_tokens": 100,
        "kv_cache_manager": None,
    }


def test_setup_attn_metadata_accepts_processor_capacity():
    encoder = _DummyEncoder()
    encoder.setup_attn_metadata(
        max_num_items=8,
        max_num_tokens=100,
        attention_metadata_capacity={"attention": 7},
    )

    assert encoder.attn_metadata.kwargs == {
        "max_num_requests": 7,
        "max_num_tokens": 100,
        "kv_cache_manager": None,
    }


def test_setup_attn_metadata_is_idempotent_per_call():
    """Subsequent calls overwrite the previous AttentionMetadata, which
    matches how the engine drives `_set_up_multimodal_encoder_attn_metadata` (called
    once at engine init; tests that multiple calls don't crash and that the
    last sizes win)."""
    encoder = _DummyEncoder()
    big1 = _ENCODER_FALLBACK_MAX_NUM_REQUESTS + 8
    big2 = _ENCODER_FALLBACK_MAX_NUM_REQUESTS + 16
    encoder.setup_attn_metadata(max_num_items=big1, max_num_tokens=100)
    first = encoder.attn_metadata

    encoder.setup_attn_metadata(max_num_items=big2, max_num_tokens=200)
    second = encoder.attn_metadata

    assert first is not second
    assert second.kwargs == {
        "max_num_requests": big2,
        "max_num_tokens": 200,
        "kv_cache_manager": None,
    }


def test_pixtral_maps_each_atomic_item_to_one_attention_context():
    class _PixtralCapacityEncoder(_DummyEncoder):
        get_encoder_attention_metadata_capacity = (
            PixtralVisionModel.get_encoder_attention_metadata_capacity
        )

    encoder = _PixtralCapacityEncoder()
    encoder.setup_attn_metadata(max_num_items=3, max_num_tokens=100)

    assert encoder.attn_metadata.kwargs == {
        "max_num_requests": 3,
        "max_num_tokens": 100,
        "kv_cache_manager": None,
    }

    encoder.setup_attn_metadata(max_num_items=100, max_num_tokens=3)
    assert encoder.attn_metadata.kwargs["max_num_requests"] == 3


def test_qwen2_maps_token_budget_to_full_and_window_attention_contexts():
    encoder = SimpleNamespace(spatial_merge_unit=4)

    capacities = Qwen2_5_VisionModel.get_encoder_attention_metadata_capacity(
        encoder, max_num_items=8, max_num_tokens=100
    )

    assert capacities == {
        "full_attention": 25,
        "window_attention": 25,
    }


def test_qwen3_maps_token_budget_to_temporal_attention_contexts():
    encoder = SimpleNamespace(spatial_merge_unit=4)

    capacities = Qwen3VisionModel.get_encoder_attention_metadata_capacity(
        encoder, max_num_items=8, max_num_tokens=100
    )

    assert capacities == {"attention": 25}


def test_subclass_can_override_setup_attn_metadata():
    """Special encoders (e.g. RADIO with `kv_layout="NHD"`, Qwen2-VL with
    multiple metadata objects) override `setup_attn_metadata`. Verify that
    the override is honored and the default is not invoked."""
    captured = {}

    class _OverridingEncoder(nn.Module, MultimodalEncoderMixin):
        def __init__(self):
            super().__init__()
            self.metadata_cls = _StubMetadata

        def setup_attn_metadata(self, max_num_items, max_num_tokens):
            captured["max_num_items"] = max_num_items
            captured["max_num_tokens"] = max_num_tokens
            # Intentionally skip building self.attn_metadata to confirm
            # the default impl wasn't called.

    encoder = _OverridingEncoder()
    encoder.setup_attn_metadata(max_num_items=5, max_num_tokens=50)

    assert encoder.attn_metadata is None  # default impl not invoked
    assert captured == {"max_num_items": 5, "max_num_tokens": 50}

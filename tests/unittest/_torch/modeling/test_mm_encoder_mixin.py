"""Unit tests for `MmEncoderMixin`.

Verifies that the mixin's default `setup_attn_metadata` builds the encoder's
AttentionMetadata exactly once with the runtime sizes injected by the engine
(`max_num_requests`, `max_num_tokens`) and `kv_cache_manager=None`. These
tests use a stub `metadata_cls` to avoid pulling in any real attention
backend (and any GPU/CUDA dependency).
"""

import torch.nn as nn

from tensorrt_llm._torch.models.modeling_multimodal_encoder import MmEncoderMixin


class _StubMetadata:
    """Minimal stand-in for an AttentionMetadata constructor."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummyEncoder(nn.Module, MmEncoderMixin):
    def __init__(self):
        super().__init__()
        self.metadata_cls = _StubMetadata


def test_attn_metadata_is_none_before_setup():
    encoder = _DummyEncoder()
    assert encoder.attn_metadata is None


def test_setup_attn_metadata_builds_with_engine_sizes():
    encoder = _DummyEncoder()
    encoder.setup_attn_metadata(max_num_requests=42, max_num_tokens=1234)

    assert isinstance(encoder.attn_metadata, _StubMetadata)
    assert encoder.attn_metadata.kwargs == {
        "max_num_requests": 42,
        "max_num_tokens": 1234,
        "kv_cache_manager": None,
    }


def test_setup_attn_metadata_is_idempotent_per_call():
    """Subsequent calls overwrite the previous AttentionMetadata, which
    matches how the engine drives `_set_up_mm_encoder_attn_metadata` (called
    once at engine init; tests that multiple calls don't crash and that the
    last sizes win)."""
    encoder = _DummyEncoder()
    encoder.setup_attn_metadata(max_num_requests=8, max_num_tokens=100)
    first = encoder.attn_metadata

    encoder.setup_attn_metadata(max_num_requests=16, max_num_tokens=200)
    second = encoder.attn_metadata

    assert first is not second
    assert second.kwargs == {
        "max_num_requests": 16,
        "max_num_tokens": 200,
        "kv_cache_manager": None,
    }


def test_subclass_can_override_setup_attn_metadata():
    """Special encoders (e.g. RADIO with `kv_layout="NHD"`, Qwen2-VL with
    multiple metadata objects) override `setup_attn_metadata`. Verify that
    the override is honored and the default is not invoked."""
    captured = {}

    class _OverridingEncoder(nn.Module, MmEncoderMixin):
        def __init__(self):
            super().__init__()
            self.metadata_cls = _StubMetadata

        def setup_attn_metadata(self, max_num_requests, max_num_tokens):
            captured["max_num_requests"] = max_num_requests
            captured["max_num_tokens"] = max_num_tokens
            # Intentionally skip building self.attn_metadata to confirm
            # the default impl wasn't called.

    encoder = _OverridingEncoder()
    encoder.setup_attn_metadata(max_num_requests=5, max_num_tokens=50)

    assert encoder.attn_metadata is None  # default impl not invoked
    assert captured == {"max_num_requests": 5, "max_num_tokens": 50}

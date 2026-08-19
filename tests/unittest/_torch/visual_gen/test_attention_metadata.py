# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for VisualGen attention metadata sites.

VisualGen reuses the shared-core ``AttentionMetadata`` types rather than
defining its own; these tests pin the behaviour VisualGen depends on (no KV
cache, mixed Q/KV lengths, allocate-once/prepare-in-place) plus the thin
helpers in ``visual_gen/attention_backend/metadata.py``.
"""

import pytest
import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.visual_gen.attention_backend.metadata import (
    create_diffusion_attn_metadata,
    make_diffusion_attn_metadata,
    prepare_diffusion_attn_metadata,
)
from tensorrt_llm._torch.visual_gen.models.modeling import _attn_metadata_shape_key

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _site(max_batch_size=2, max_seq_len=4096, cls=TrtllmAttentionMetadata):
    return create_diffusion_attn_metadata(
        cls, max_batch_size=max_batch_size, max_seq_len=max_seq_len
    )


def test_self_attention_site_is_not_cross():
    md = _site()
    prepare_diffusion_attn_metadata(md, batch_size=1, q_seq_lens=1024)

    # `is_cross` upstream is an identity check, so a self-attention site must
    # leave seq_lens_kv unset rather than pass an equal-valued tensor.
    assert md.is_cross is False
    assert torch.equal(md.seq_lens, torch.tensor([1024], dtype=torch.int32))
    assert torch.equal(md.kv_lens[:1], torch.tensor([1024], dtype=torch.int32))


def test_mixed_site_carries_distinct_kv_length():
    md = _site()
    prepare_diffusion_attn_metadata(md, batch_size=1, q_seq_lens=1024, kv_seq_lens=1536)

    assert md.is_cross is True
    assert torch.equal(md.seq_lens, torch.tensor([1024], dtype=torch.int32))
    assert torch.equal(md.seq_lens_kv, torch.tensor([1536], dtype=torch.int32))
    # prepare() must derive the KV lengths from seq_lens_kv, not seq_lens.
    assert torch.equal(md.kv_lens[:1], torch.tensor([1536], dtype=torch.int32))


def test_no_kv_cache_path():
    md = _site()
    prepare_diffusion_attn_metadata(md, batch_size=1, q_seq_lens=256)

    assert md.kv_cache_manager is None
    assert md.kv_cache_params.use_cache is False


def test_reprepare_keeps_device_buffers_pointer_stable():
    """CUDA graphs capture pointers into these buffers; they must not move."""
    md = _site()
    prepare_diffusion_attn_metadata(md, batch_size=1, q_seq_lens=1024)
    ptr = md.seq_lens_cuda.data_ptr()

    prepare_diffusion_attn_metadata(md, batch_size=1, q_seq_lens=1024)  # unchanged
    assert md.seq_lens_cuda.data_ptr() == ptr

    prepare_diffusion_attn_metadata(md, batch_size=1, q_seq_lens=2048)  # new length
    assert md.seq_lens_cuda.data_ptr() == ptr
    assert torch.equal(md.seq_lens, torch.tensor([2048], dtype=torch.int32))


def test_batch_size_change_is_supported():
    md = _site(max_batch_size=2)
    prepare_diffusion_attn_metadata(md, batch_size=1, q_seq_lens=512)
    prepare_diffusion_attn_metadata(md, batch_size=2, q_seq_lens=512)

    assert md.num_seqs == 2
    assert md.num_contexts == 2


def test_batch_size_beyond_capacity_raises():
    md = _site(max_batch_size=1)
    with pytest.raises(ValueError, match="capacity"):
        prepare_diffusion_attn_metadata(md, batch_size=4, q_seq_lens=512)


def test_seq_lens_batch_mismatch_raises():
    md = _site()
    with pytest.raises(ValueError, match="batch"):
        prepare_diffusion_attn_metadata(md, batch_size=2, q_seq_lens=[512])


def test_base_metadata_type_works_for_backends_that_ignore_metadata():
    md = _site(max_seq_len=512, cls=AttentionMetadata)
    prepare_diffusion_attn_metadata(md, batch_size=1, q_seq_lens=512)
    assert md.num_tokens == 512


class TestMakeAttnMetadata:
    def test_each_call_returns_an_independent_site(self):
        self_site = make_diffusion_attn_metadata(
            TrtllmAttentionMetadata, batch_size=1, q_seq_lens=64
        )
        cross_site = make_diffusion_attn_metadata(
            TrtllmAttentionMetadata, batch_size=1, q_seq_lens=64, kv_seq_lens=32
        )
        assert self_site is not cross_site
        assert self_site.is_cross is False
        assert cross_site.is_cross is True

    def test_sizes_capacity_from_the_longer_of_q_and_kv(self):
        md = make_diffusion_attn_metadata(
            TrtllmAttentionMetadata, batch_size=2, q_seq_lens=128, kv_seq_lens=4096
        )
        assert md.max_seq_len >= 4096
        assert md.max_num_requests >= 2

    def test_accepts_per_sequence_lengths(self):
        md = make_diffusion_attn_metadata(
            TrtllmAttentionMetadata, batch_size=2, q_seq_lens=[64, 128]
        )
        assert torch.equal(md.seq_lens, torch.tensor([64, 128], dtype=torch.int32))
        assert md.max_seq_len >= 128

    def test_is_legal_outside_but_not_inside_a_cuda_graph_capture(self):
        """Metadata must be built before entering the captured region.

        ``prepare()`` stages lengths through pinned host memory, so building
        inside a capture raises. This pins the constraint that forces
        construction out of the CUDA-graph-wrapped model ``forward``.
        """
        make_diffusion_attn_metadata(TrtllmAttentionMetadata, batch_size=1, q_seq_lens=128)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            make_diffusion_attn_metadata(TrtllmAttentionMetadata, batch_size=1, q_seq_lens=128)
        torch.cuda.current_stream().wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with pytest.raises(RuntimeError, match="CUDA graph capture"):
            with torch.cuda.graph(graph):
                make_diffusion_attn_metadata(TrtllmAttentionMetadata, batch_size=1, q_seq_lens=128)


class TestCudaGraphShapeKey:
    """The graph key must see metadata, which reaches models as non-tensor args."""

    def test_skips_kv_for_self_attention(self):
        md = _site()
        prepare_diffusion_attn_metadata(md, batch_size=1, q_seq_lens=64)
        assert _attn_metadata_shape_key(attn_metadata=md) == (("attn_metadata", (64,), None),)

        prepare_diffusion_attn_metadata(md, batch_size=1, q_seq_lens=64, kv_seq_lens=96)
        assert _attn_metadata_shape_key(attn_metadata=md) == (("attn_metadata", (64,), (96,)),)

    def test_distinguishes_lengths_and_cross_ness(self):
        def key(**kw):
            return _attn_metadata_shape_key(
                attn_metadata=make_diffusion_attn_metadata(TrtllmAttentionMetadata, **kw)
            )

        base = key(batch_size=1, q_seq_lens=128)
        assert key(batch_size=1, q_seq_lens=256) != base
        assert key(batch_size=1, q_seq_lens=128, kv_seq_lens=512) != base
        assert key(batch_size=1, q_seq_lens=128) == base

    def test_expands_a_dict_of_sites(self):
        sites = {
            "video_self": make_diffusion_attn_metadata(
                TrtllmAttentionMetadata, batch_size=1, q_seq_lens=64
            ),
            "video_text_cross": make_diffusion_attn_metadata(
                TrtllmAttentionMetadata, batch_size=1, q_seq_lens=64, kv_seq_lens=32
            ),
        }
        assert _attn_metadata_shape_key(attn_metadata=sites) == (
            ("attn_metadata.video_self", (64,), None),
            ("attn_metadata.video_text_cross", (64,), (32,)),
        )

    def test_is_none_when_the_forward_carries_no_metadata(self):
        assert _attn_metadata_shape_key(hidden_states=torch.zeros(1)) is None

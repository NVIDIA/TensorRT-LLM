# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CUTEDSL and CUDNN attention over a CausalKVCacheManager: a causal video rollout, a denoising
pass and a clean pass per chunk, against a dense reference that knows nothing about pages."""

import pytest
import torch
from utils.kv_causal_reference import dense_window_reference, paged_window_reference

from tensorrt_llm._torch.attention.backends.interface import PredefinedAttentionMask
from tensorrt_llm._torch.visual_gen.attention.backends.cudnn import CuDNNAttention
from tensorrt_llm._torch.visual_gen.attention.backends.cudnn_kv_cache import CuDNNCausalKVPages
from tensorrt_llm._torch.visual_gen.attention.backends.cute_dsl import fmha
from tensorrt_llm._torch.visual_gen.attention.backends.cute_dsl.fmha import CuTeDSLAttention
from tensorrt_llm._torch.visual_gen.cache import CausalKVCacheManager
from tensorrt_llm.math_utils import ceil_div

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

NUM_LAYERS = 2
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 64
DTYPE = torch.bfloat16

# (tokens_per_page, pin_tokens, tokens_per_frame, window_frames, chunk_frames)
GEOMETRIES = [
    pytest.param(32, 48, 96, 4, 4, id="tpb32-pin48-frame96"),
    pytest.param(16, 17, 40, 3, 2, id="tpb16-pin17-frame40"),
    # h - window is the last slot of a page at every chunk boundary: the window's first token
    # is alone on a page the manager must keep.
    pytest.param(32, 63, 96, 3, 2, id="tpb32-window-start-on-last-slot"),
    pytest.param(64, 100, 200, 2, 3, id="tpb64-pin100-frame200"),
    # Cosmos frames of 924 tokens: no boundary is page aligned.
    pytest.param(64, 924, 924, 3, 2, id="tpb64-pin924-frame924"),
    # The fixed region ends on the page where the window starts.
    pytest.param(64, 70, 48, 2, 2, id="tpb64-fixed-end-and-window-start-share-a-page"),
]


_BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


@pytest.fixture(
    autouse=True,
    params=[
        "cutedsl-reference",
        pytest.param(
            "cutedsl",
            marks=pytest.mark.skipif(
                not _BLACKWELL, reason="the CuTe DSL FMHA kernel needs Blackwell"
            ),
        ),
        "cudnn",
    ],
)
def backend_kind(request, monkeypatch):
    """The backend under test. ``cutedsl-reference`` runs the CUTEDSL backend over the paged
    window kernel's reference, which checks the backend and the cache alone."""
    if request.param == "cutedsl-reference":
        monkeypatch.setattr(fmha, "cute_dsl_paged_window_fmha_fwd", paged_window_reference)
    return request.param


def make_cache(tpb, pin, frame, window_frames, chunk_frames, max_chunks=12):
    window, staged = (window_frames - 1) * frame, chunk_frames * frame
    # Private pages for cuDNN, enough for the clean pass's one block per frame.
    extra = CuDNNCausalKVPages.private_pages_for(tpb, max(chunk_frames, ceil_div(pin, frame)))
    return CausalKVCacheManager(
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
        tokens_per_page=tpb,
        pin_tokens=pin,
        window_tokens=window,
        max_staged_tokens=staged,
        max_seq_len=pin + max_chunks * staged,
        extra_pages=extra,
    )


def make_backends(kind):
    cls = CuDNNAttention if kind == "cudnn" else CuTeDSLAttention
    return [
        cls(layer_idx=i, num_heads=NUM_HEADS, head_dim=HEAD_DIM, num_kv_heads=NUM_KV_HEADS)
        for i in range(NUM_LAYERS)
    ]


def rand(*shape):
    return torch.randn(*shape, dtype=DTYPE, device="cuda")


class Rollout:
    """Runs forwards over a cache and the dense reference side by side."""

    def __init__(self, cache, kind):
        self.cache = cache
        self.backends = make_backends(kind)
        # Every committed token's K/V per layer, by logical position.
        self.keys = [
            torch.empty(0, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
        ] * NUM_LAYERS
        self.values = list(self.keys)

    def forward(self, num_tokens, block_size, num_rows=None):
        """One forward of ``num_tokens`` new tokens in every layer; checks each layer's
        output against the reference and returns the K/V it staged."""
        cache = self.cache
        rows = num_tokens if num_rows is None else num_rows
        staged = []
        for layer, backend in enumerate(self.backends):
            q, k, v = (
                rand(1, rows, NUM_HEADS, HEAD_DIM),
                rand(1, rows, NUM_KV_HEADS, HEAD_DIM),
                None,
            )
            v = rand(1, rows, NUM_KV_HEADS, HEAD_DIM)
            out = backend.forward(
                q,
                k,
                v,
                kv_cache=cache,
                seq_len=num_tokens,
                causal_block_size=block_size,
                batch_size=1,
                seq_len_kv=rows,
            )
            assert out.shape == (1, rows, NUM_HEADS, HEAD_DIM)
            all_k = torch.cat([self.keys[layer], k[0, :num_tokens]])
            all_v = torch.cat([self.values[layer], v[0, :num_tokens]])
            want = dense_window_reference(
                q[0, :num_tokens],
                all_k,
                all_v,
                start=cache.staging_offset,
                fixed=cache.fixed_tokens,
                window=cache.window_tokens,
                block_size=block_size or num_tokens,
                sm_scale=backend.scale,
            )
            torch.testing.assert_close(out[0, :num_tokens], want, rtol=2e-2, atol=2e-2)
            assert not out[0, num_tokens:].any(), "padding rows must come back zero"
            staged.append((k[0, :num_tokens], v[0, :num_tokens]))
        return staged

    def commit(self, staged):
        for layer, (k, v) in enumerate(staged):
            self.keys[layer] = torch.cat([self.keys[layer], k])
            self.values[layer] = torch.cat([self.values[layer], v])
        self.cache.commit(staged[0][0].shape[0])


@pytest.mark.parametrize("tpb,pin,frame,window_frames,chunk_frames", GEOMETRIES)
def test_rollout_matches_dense_reference(
    backend_kind, tpb, pin, frame, window_frames, chunk_frames
):
    cache = make_cache(tpb, pin, frame, window_frames, chunk_frames)
    try:
        cache.open()
        rollout = Rollout(cache, backend_kind)
        if pin:
            rollout.commit(rollout.forward(pin, None))
        chunk = chunk_frames * frame
        for _ in range(10):
            # Denoising steps restage the chunk as one block, the clean pass one frame per
            # block, causal across frames.
            for _ in range(2):
                rollout.forward(chunk, None)
            rollout.commit(rollout.forward(chunk, frame))
    finally:
        cache.shutdown()


def test_padding_rows_are_not_cached_or_attended(backend_kind):
    cache = make_cache(32, 48, 96, 4, 4)
    try:
        cache.open()
        rollout = Rollout(cache, backend_kind)
        rollout.commit(rollout.forward(48, None, num_rows=64))
        for _ in range(3):
            rollout.commit(rollout.forward(384, 96, num_rows=384 + 16))
        # A later forward still matches the reference, which never saw the padding rows.
        rollout.commit(rollout.forward(384, 96))
    finally:
        cache.shutdown()


def test_a_short_final_chunk(backend_kind):
    cache = make_cache(32, 48, 96, 4, 4)
    try:
        cache.open()
        rollout = Rollout(cache, backend_kind)
        rollout.commit(rollout.forward(48, None))
        for _ in range(4):
            rollout.commit(rollout.forward(384, 96))
        rollout.commit(rollout.forward(192, 96))
    finally:
        cache.shutdown()


def test_rejects_unsupported_calls(backend_kind):
    cache = make_cache(32, 48, 96, 4, 4)
    try:
        cache.open()
        backend = make_backends(backend_kind)[0]
        q, k, v = (
            rand(1, 96, NUM_HEADS, HEAD_DIM),
            rand(1, 96, 2, HEAD_DIM),
            rand(1, 96, 2, HEAD_DIM),
        )
        assert backend.support_kv_cache()
        with pytest.raises(NotImplementedError, match="full attention"):
            backend.forward(q, k, v, kv_cache=cache, attention_mask=PredefinedAttentionMask.CAUSAL)
        with pytest.raises(ValueError, match="one video"):
            backend.forward(rand(2, 96, NUM_HEADS, HEAD_DIM), k, v, kv_cache=cache)
        with pytest.raises(ValueError, match="heads"):
            backend.forward(q, rand(1, 96, 4, HEAD_DIM), rand(1, 96, 4, HEAD_DIM), kv_cache=cache)
        with pytest.raises(ValueError, match="seq_len"):
            backend.forward(q, k, v, kv_cache=cache, seq_len=97)
        with pytest.raises(ValueError, match="do not split"):
            backend.forward(q, k, v, kv_cache=cache, causal_block_size=40)
        with pytest.raises(NotImplementedError, match="forward_with_lse"):
            backend.forward_with_lse(q, k, v, kv_cache=cache)
        with pytest.raises(NotImplementedError, match="only implemented over a K/V cache"):
            backend.forward(q, k, v, causal_block_size=48)
    finally:
        cache.shutdown()

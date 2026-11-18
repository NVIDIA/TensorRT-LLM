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
"""`KvCachePageCodec`: how a connector reaches the cache manager's cold-page codec.

The native calls are replaced by a fake that reads the page-index pairs the
way the codec would, at call time, so these run without a GPU.
"""

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_llm._torch.pyexecutor.connectors.page_codec import KvCachePageCodec

pytestmark = pytest.mark.cpu_only


class FakeCache:
    """The four native calls, with the pairs copied out while the call runs."""

    def __init__(self, page_bytes, *, location=0, result=True):
        self.page_bytes = dict(page_bytes)
        self.location = location
        self.result = result
        self.calls = []

    def cold_page_bytes(self, layer_group_id):
        return self.page_bytes.get(layer_group_id, 0)

    def cold_page_index_location(self, layer_group_id):
        return self.location if layer_group_id in self.page_bytes else -1

    def _record(self, name, layer_group_id, base, page_indices, num_pages, stream):
        # A host array is only valid during the call, which is when a codec reads it.
        raw = ctypes.string_at(page_indices, num_pages * 8)
        pairs = np.frombuffer(raw, dtype=np.int32).reshape(num_pages, 2).tolist()
        self.calls.append((name, layer_group_id, base, [tuple(pair) for pair in pairs], stream))
        return self.result

    def encode_cold_pages(self, layer_group_id, dst_base, page_indices, num_pages, cuda_stream):
        return self._record(
            "encode", layer_group_id, dst_base, page_indices, num_pages, cuda_stream
        )

    def decode_cold_pages(self, layer_group_id, src_base, page_indices, num_pages, cuda_stream):
        return self._record(
            "decode", layer_group_id, src_base, page_indices, num_pages, cuda_stream
        )


def make_codec(cache, **kwargs):
    kwargs.setdefault("identity", "nvfp4-abc")
    kwargs.setdefault("layer_group_ids", cache.page_bytes.keys())
    return KvCachePageCodec(cache, **kwargs)


def test_encode_hands_the_codec_dst_src_pairs():
    """PageIndexPair is (dst, src): the arena slot first, the hot page second."""
    cache = FakeCache({0: 96})
    codec = make_codec(cache)
    codec.encode(0, 0x1000, [0, 1, 2], [7, 3, 9], stream=0x55)
    assert cache.calls == [("encode", 0, 0x1000, [(0, 7), (1, 3), (2, 9)], 0x55)]


def test_decode_hands_the_codec_hot_page_then_slot():
    cache = FakeCache({0: 96})
    codec = make_codec(cache)
    codec.decode(0, 0x2000, [7, 3], [0, 1], stream=0x66)
    assert cache.calls == [("decode", 0, 0x2000, [(7, 0), (3, 1)], 0x66)]


def test_an_empty_batch_reaches_no_codec():
    cache = FakeCache({0: 96})
    make_codec(cache).encode(0, 0x1000, [], [], stream=0)
    assert cache.calls == []


def test_the_legacy_default_stream_is_refused():
    """The default codec's batched copy rejects it, so no codec is handed it."""
    cache = FakeCache({0: 96})
    with pytest.raises(ValueError, match="legacy default stream"):
        make_codec(cache).encode(0, 0x1000, [0], [4], stream=0)
    assert cache.calls == []


def test_a_negative_index_is_refused_before_the_codec_sees_it():
    """The codec addresses memory with these unchecked: BAD_PAGE_INDEX is a wild write."""
    cache = FakeCache({0: 96})
    with pytest.raises(ValueError, match="non-negative"):
        make_codec(cache).encode(0, 0x1000, [0], [-1], stream=1)
    assert cache.calls == []


def test_mismatched_lists_are_refused():
    cache = FakeCache({0: 96})
    with pytest.raises(ValueError, match="destination slots"):
        make_codec(cache).decode(0, 0x1000, [1, 2], [0], stream=1)


def test_a_rejected_batch_raises():
    cache = FakeCache({0: 96}, result=False)
    with pytest.raises(RuntimeError, match="rejected encoding 1 pages of layer group 0"):
        make_codec(cache).encode(0, 0x1000, [0], [4], stream=1)


def test_bindings_from_an_older_build_are_named_as_the_problem():
    with pytest.raises(RuntimeError, match="rebuild or reinstall"):
        KvCachePageCodec(SimpleNamespace(), identity="x", layer_group_ids=[0])


def test_a_layer_group_the_codec_does_not_cover_is_refused():
    with pytest.raises(ValueError, match="no page size for layer group 1"):
        make_codec(FakeCache({0: 96}), layer_group_ids=[0, 1])


def test_an_unknown_index_location_is_refused():
    with pytest.raises(ValueError, match="page-index location 7"):
        make_codec(FakeCache({0: 96}, location=7))


def test_page_bytes_are_the_codecs():
    codec = make_codec(FakeCache({0: 96, 2: 160}))
    assert codec.page_bytes(0) == 96
    assert codec.page_bytes(2) == 160
    assert codec.layer_group_ids == (0, 2)


def test_the_encoding_names_the_codec_and_the_raw_page():
    """The same codec over a different hot layout must not read as a match."""
    codec = make_codec(FakeCache({0: 96, 1: 96}), raw_bytes_per_page={0: 512, 1: 256})
    assert codec.encoding(0) == "nvfp4-abc.r512"
    assert codec.encoding(1) == "nvfp4-abc.r256"
    assert codec.encoding(0) != codec.encoding(1)


def test_from_manager_returns_none_for_the_lossless_default():
    manager = SimpleNamespace(cold_page_codec_identity=None, impl=FakeCache({0: 96}))
    assert KvCachePageCodec.from_manager(manager, {0: 512}) is None


def test_from_manager_binds_the_managers_native_handle():
    cache = FakeCache({0: 96})
    manager = SimpleNamespace(cold_page_codec_identity="nvfp4-abc", impl=cache)
    codec = KvCachePageCodec.from_manager(manager, {0: 512})
    assert codec.identity == "nvfp4-abc"
    assert codec.encoding(0) == "nvfp4-abc.r512"
    codec.encode(0, 0x1000, [0], [5], stream=1)
    assert cache.calls == [("encode", 0, 0x1000, [(0, 5)], 1)]

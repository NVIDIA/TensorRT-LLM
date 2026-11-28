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
"""The KV cache manager's cold-page codec, for pages a connector stores outside it.

`KVCacheManagerV2` encodes a page when it leaves the GPU for a host, disk or
remote level, through the codec a `kv_cache_compression_config` method provides
(NVFP4 for `quantization_for_cold_page`). A connector whose store takes the place
of those levels, such as `mooncake-store`, can hold its pages in the same
representation through :class:`KvCachePageCodec`. The codec is the manager's own
instance, so a compression method plugged into the manager applies to the
connector's store as well, with nothing connector-specific to add.

One encoded page holds every buffer of one layer group's page slot and is
:meth:`KvCachePageCodec.page_bytes` long. Encoding reads hot pages by their page
slot, the index a connector gets from the cache's page-index lists, and writes
them at ``base + slot * page_bytes`` of caller-owned GPU-accessible memory, such
as a registered pinned buffer. Decoding is the reverse.
"""

import threading
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Sequence

import numpy as np
import torch

from tensorrt_llm._utils import maybe_pin_memory

if TYPE_CHECKING:
    from ..kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

__all__ = ["KvCachePageCodec"]

#: `cold_page_index_location` answers, as the bindings spell PageIndexLocation.
_INDEX_IN_HOST_MEMORY = 0
_INDEX_IN_DEVICE_MEMORY = 1


class KvCachePageCodec:
    """Encodes and decodes one cache manager's pages with its cold-page codec.

    Calls are serialized here: a codec is not required to be reentrant, and a
    connector typically saves on a background thread while it loads on the
    executor's. Each call only enqueues work on the stream it is given.
    """

    def __init__(
        self,
        cache,
        *,
        identity: str,
        layer_group_ids: Iterable[int],
        raw_bytes_per_page: Optional[Dict[int, int]] = None,
        kv_dtype: Optional[str] = None,
        num_slots: Optional[Dict[int, int]] = None,
    ):
        """Bind to a cache manager's native handle.

        Args:
            cache: The native `KVCacheManager` (`KVCacheManagerV2.impl`), or any
                object with its `cold_page_bytes`, `cold_page_index_location`,
                `encode_cold_pages` and `decode_cold_pages` methods.
            identity: Names the encoded representation; see :attr:`identity`.
            layer_group_ids: The layer groups a caller will encode.
            raw_bytes_per_page: Unencoded page size per layer group, folded into
                :meth:`encoding` so a geometry change is never read as a match.
            kv_dtype: Name of the hot cache's element type, folded into
                :meth:`encoding`: pages of the same size can hold FP16 or BF16.
            num_slots: GPU page slots per layer group. A hot page index at or
                past it is refused, since the codec writes through it unchecked.
        """
        missing = [
            name
            for name in (
                "cold_page_bytes",
                "cold_page_index_location",
                "encode_cold_pages",
                "decode_cold_pages",
            )
            if not hasattr(cache, name)
        ]
        if missing:
            raise RuntimeError(
                "The KV cache manager's bindings lack "
                f"{', '.join(missing)}, so a connector cannot run its cold-page "
                "codec. The Python package and the compiled bindings come from "
                "different builds; rebuild or reinstall TensorRT-LLM."
            )
        self._cache = cache
        self._identity = str(identity)
        self._lock = threading.Lock()
        self._page_bytes: Dict[int, int] = {}
        self._index_on_device: Dict[int, bool] = {}
        self._raw_bytes = dict(raw_bytes_per_page or {})
        self._kv_dtype = kv_dtype
        self._num_slots = {int(key): int(value) for key, value in (num_slots or {}).items()}
        self._closed = False
        for layer_group_id in layer_group_ids:
            layer_group_id = int(layer_group_id)
            page_bytes = int(cache.cold_page_bytes(layer_group_id))
            if page_bytes <= 0:
                raise ValueError(
                    f"the cold-page codec reports no page size for layer group "
                    f"{layer_group_id}; it does not cover the layout it was asked to encode"
                )
            location = int(cache.cold_page_index_location(layer_group_id))
            if location not in (_INDEX_IN_HOST_MEMORY, _INDEX_IN_DEVICE_MEMORY):
                raise ValueError(
                    f"the cold-page codec reports page-index location {location} for "
                    f"layer group {layer_group_id}; expected host (0) or device (1)"
                )
            self._page_bytes[layer_group_id] = page_bytes
            self._index_on_device[layer_group_id] = location == _INDEX_IN_DEVICE_MEMORY

    @classmethod
    def from_manager(
        cls,
        manager: "KVCacheManagerV2",
        raw_bytes_per_page: Dict[int, int],
        num_slots: Optional[Dict[int, int]] = None,
    ) -> Optional["KvCachePageCodec"]:
        """The manager's codec, or None when it keeps the lossless default.

        Args:
            manager: The cache manager whose pages will be encoded.
            raw_bytes_per_page: Unencoded page size of each layer group to cover.
            num_slots: GPU page slots of each layer group, bounding the hot
                page indices a call may pass.
        """
        identity = getattr(manager, "cold_page_codec_identity", None)
        if identity is None:
            return None
        return cls(
            manager.impl,
            identity=identity,
            layer_group_ids=raw_bytes_per_page.keys(),
            raw_bytes_per_page=raw_bytes_per_page,
            kv_dtype=getattr(getattr(manager, "dtype", None), "name", None),
            num_slots=num_slots,
        )

    @property
    def identity(self) -> str:
        """Names the encoded representation: the method and its settings.

        Two caches whose codecs would decode a page differently report
        different identities, so a store shared between them keys their pages
        apart.
        """
        return self._identity

    @property
    def layer_group_ids(self) -> Sequence[int]:
        """Layer groups this codec was bound to."""
        return tuple(self._page_bytes)

    def page_bytes(self, layer_group_id: int) -> int:
        """Bytes of one encoded page of `layer_group_id`."""
        return self._page_bytes[layer_group_id]

    def encoding(self, layer_group_id: int) -> str:
        """What a stored page of `layer_group_id` holds, for a store key.

        The codec identity plus the hot cache's element type and unencoded page
        size: cold pages of the same size can come from different hot layouts,
        for example the same method over BF16 and over FP8 KV, and decoding one
        into the other is wrong.
        """
        encoding = self._identity
        if self._kv_dtype is not None:
            encoding += f".{self._kv_dtype}"
        raw = self._raw_bytes.get(layer_group_id)
        if raw is not None:
            encoding += f".r{raw}"
        return encoding

    def close(self) -> None:
        """Refuse further calls, for a caller whose cache manager is shutting down.

        The manager frees its GPU pools at shutdown while keeping the codec, so
        a call after it would read or write freed memory. Work already enqueued
        is not affected; the caller drains its streams before the pools go.
        """
        with self._lock:
            self._closed = True

    def describe(self) -> str:
        """A one-line summary for startup logs."""
        sizes = ", ".join(
            f"lg{layer_group_id}: {self._raw_bytes.get(layer_group_id, '?')} -> {size} B"
            for layer_group_id, size in self._page_bytes.items()
        )
        return f"{self._identity} ({sizes})"

    def encode(
        self,
        layer_group_id: int,
        dst_base: int,
        dst_slots: Sequence[int],
        hot_pages: Sequence[int],
        stream: int,
    ) -> None:
        """Enqueue encoding of hot pages into slots at `dst_base`.

        Args:
            layer_group_id: The layer group the pages belong to.
            dst_base: GPU-accessible address of slot 0, `page_bytes` per slot.
            dst_slots: Destination slot of each page.
            hot_pages: GPU page slot of each page, as the cache's page-index
                lists name it.
            stream: Raw handle of a created CUDA stream (not the legacy
                default stream) the work is enqueued on. The pages must be
                complete on it, and the slots are written by the time it
                reaches the end of the enqueued work.
        """
        self._run(True, layer_group_id, dst_base, dst_slots, hot_pages, stream)

    def decode(
        self,
        layer_group_id: int,
        src_base: int,
        hot_pages: Sequence[int],
        src_slots: Sequence[int],
        stream: int,
    ) -> None:
        """Enqueue decoding of slots at `src_base` into hot pages.

        Args:
            layer_group_id: The layer group the pages belong to.
            src_base: GPU-accessible address of slot 0, `page_bytes` per slot.
            hot_pages: GPU page slot each decoded page is written to.
            src_slots: Source slot of each page.
            stream: Raw handle of a created CUDA stream the work is enqueued on.
        """
        self._run(False, layer_group_id, src_base, hot_pages, src_slots, stream)

    def _run(
        self,
        encode: bool,
        layer_group_id: int,
        base: int,
        dst: Sequence[int],
        src: Sequence[int],
        stream: int,
    ) -> None:
        if len(dst) != len(src):
            raise ValueError(f"{len(dst)} destination slots for {len(src)} source pages")
        if not dst:
            return
        if not stream:
            # The default codec copies with cuMemcpyBatchAsync, which refuses the
            # legacy default stream, so a codec may reject it.
            raise ValueError("pass a created CUDA stream, not the legacy default stream (0)")
        # (dst, src) int32 pairs: the PageIndexPair layout the codec reads.
        pairs = np.empty((len(dst), 2), dtype=np.int32)
        pairs[:, 0] = dst
        pairs[:, 1] = src
        if pairs.min() < 0:
            # The codec addresses memory with these unchecked, so a negative
            # index (BAD_PAGE_INDEX) would be a wild write, not an error.
            raise ValueError("page and slot indices must be non-negative")
        hot_limit = self._num_slots.get(layer_group_id)
        hot = pairs[:, 1] if encode else pairs[:, 0]
        if hot_limit is not None and int(hot.max()) >= hot_limit:
            raise ValueError(
                f"hot page index {int(hot.max())} is past the {hot_limit} GPU page slots "
                f"of layer group {layer_group_id}"
            )
        index_on_device = self._index_on_device.get(layer_group_id)
        if index_on_device is None:
            raise ValueError(
                f"layer group {layer_group_id} is not one this codec was bound to "
                f"({sorted(self._index_on_device)})"
            )
        keepalive = None
        if index_on_device:
            # The codec reads a device array on the stream, after this returns,
            # so the copy and the array's lifetime are both tied to the stream.
            target = torch.cuda.ExternalStream(stream)
            with torch.cuda.stream(target):
                keepalive = maybe_pin_memory(torch.from_numpy(pairs)).cuda(non_blocking=True)
            keepalive.record_stream(target)
            pointer = keepalive.data_ptr()
        else:
            # A host array only has to outlive the call.
            pointer = pairs.ctypes.data
        call = self._cache.encode_cold_pages if encode else self._cache.decode_cold_pages
        with self._lock:
            if self._closed:
                raise RuntimeError(
                    "the cold-page codec was closed because its cache manager is shutting down"
                )
            ok = call(int(layer_group_id), int(base), int(pointer), len(dst), int(stream))
        del keepalive
        if not ok:
            raise RuntimeError(
                f"the cold-page codec rejected {'encoding' if encode else 'decoding'} "
                f"{len(dst)} pages of layer group {layer_group_id}"
            )

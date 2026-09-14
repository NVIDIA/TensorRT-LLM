# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Cache manager and prepared metadata of the glm_kpool sparse backend.

``Glm5NextMamba2Metadata`` is the model's mamba metadata: the shared Kimi KDA
metadata (chunk schedule, aligned generation slot indices for the replay
verify kernel) plus the sparse layers' persistent buffers below.

CUDA graphs replay captured kernels over fixed buffer addresses; nothing
Python re-runs at replay. Every per-step request-derived value the sparse
layers consume in decode must therefore live in persistent device buffers
refreshed by ``prepare()`` -- which the engine (and the CUDA-graph runner,
before every replay) calls outside the captured region. This subclass adds
exactly those buffers to the mamba metadata the hybrid manager already
attaches:

* ``glm_block_tables`` -- ``[max_batch, max_blocks_per_seq]`` base-slot
  page ids (pinned staging + one async H2D per step);
* ``glm_kv_lens`` -- ``[max_batch]`` per-request visible lengths
  (cached + this step's tokens), same staging pattern;
* ``glm_cached_lens_host`` / ``glm_ctx_cu_seqlens`` -- plain host values
  for the prefill path, which is never captured (decode-only graphs).

The buffer width is fixed at first use from the manager's own
``max_blocks_per_seq`` so captured gathers can never need a wider table;
a mid-run widening request is a hard error rather than a silent
reallocation that stale graphs would keep reading.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from tensorrt_llm._torch.modules.kimi_kda.kimi_k3_mamba_metadata import KimiK3MambaMetadata
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm.runtime.kv_cache_manager_v2 import BufferConfig


class Glm5NextMamba2Metadata(KimiK3MambaMetadata):
    def __init__(
        self, max_batch_size: int, chunk_size: int, max_num_tokens: int | None = None
    ) -> None:
        # The KDA chunk-index buffer holds one row per 64-token chunk; a
        # harness that gives no token budget gets a generous fixed capacity
        # (two int64 per row -- negligible).
        super().__init__(
            max_batch_size, chunk_size, max_num_tokens if max_num_tokens is not None else 65536
        )
        from tensorrt_llm._utils import prefer_pinned

        self._glm_pin = prefer_pinned()
        self.glm_block_tables: torch.Tensor | None = None
        self._glm_block_tables_cpu: torch.Tensor | None = None
        self.glm_kv_lens = torch.zeros(max_batch_size, dtype=torch.long, device="cuda")
        self._glm_kv_lens_cpu = torch.zeros(
            max_batch_size, dtype=torch.long, pin_memory=self._glm_pin
        )
        self.glm_cached_lens_host: list[int] = []
        self.glm_ctx_cu_seqlens: list[int] = [0]

    def _glm_ensure_tables(self, width: int) -> None:
        width = max(1, int(width))
        if self.glm_block_tables is None:
            self.glm_block_tables = torch.zeros(
                self.max_batch_size, width, dtype=torch.long, device="cuda"
            )
            self._glm_block_tables_cpu = torch.zeros(
                self.max_batch_size, width, dtype=torch.long, pin_memory=self._glm_pin
            )
        elif self.glm_block_tables.shape[1] < width:
            raise RuntimeError(
                "glm5_next block-table buffer would need to grow from "
                f"{self.glm_block_tables.shape[1]} to {width} pages mid-run; "
                "captured CUDA graphs would keep reading the old buffer"
            )

    def prepare(self, attn_metadata) -> None:
        super().prepare(attn_metadata)
        manager = attn_metadata.kv_cache_manager
        kv_params = attn_metadata.kv_cache_params
        request_ids = attn_metadata.request_ids
        if (
            manager is None
            or not hasattr(manager, "get_batch_slot_tables")
            or kv_params is None
            or kv_params.num_cached_tokens_per_seq is None
            or request_ids is None
        ):
            return

        batch = attn_metadata.seq_lens.shape[0]
        num_contexts = int(attn_metadata.num_contexts)
        lens = [int(x) for x in attn_metadata.seq_lens[:batch]]
        cached_src = kv_params.num_cached_tokens_per_seq
        if isinstance(cached_src, torch.Tensor):
            cached = [int(x) for x in cached_src[:batch]]
        else:
            cached = [int(cached_src[i]) for i in range(batch)]

        self.glm_cached_lens_host = cached
        cu = [0]
        for length in lens[:num_contexts]:
            cu.append(cu[-1] + length)
        self.glm_ctx_cu_seqlens = cu

        self._glm_kv_lens_cpu[:batch].copy_(
            torch.as_tensor([c + n for c, n in zip(cached, lens)], dtype=torch.long)
        )
        self.glm_kv_lens[:batch].copy_(self._glm_kv_lens_cpu[:batch], non_blocking=True)

        width = int(getattr(manager, "max_blocks_per_seq", 0)) or 1
        self._glm_ensure_tables(width)
        pages = manager.get_batch_slot_tables(list(request_ids)[:batch])
        staging = self._glm_block_tables_cpu
        staging[:batch].zero_()
        for row, page_ids in enumerate(pages):
            if page_ids:
                staging[row, : len(page_ids)].copy_(torch.as_tensor(page_ids, dtype=torch.long))
        self.glm_block_tables[:batch].copy_(staging[:batch], non_blocking=True)


class Glm5NextCacheManager(MambaHybridCacheManagerV2):
    """One KVCacheManagerV2 lifecycle for all three kinds of GLM state.

    The three state families have genuinely different shapes and are kept
    that way -- none is padded into a faux common KV tensor:

    * linear-attention layers: the recurrent accumulator and the four-tap
      convolution history, carried by the inherited Mamba side with
      ``conv_state_layout='q_k_v'`` (the convolution's ``[q | k | v]``
      section order);
    * sparse-attention layers: the ``kv_lora_rank``-wide compressed latent,
      carried by the standard attention pages with ``SELFKONLY``;
    * sparse-attention layers again: the indexer's packed ``[k | gate]``
      state, added here as one extra ``Role.INDEX_KEY`` buffer per sparse
      layer.

    The extra buffer is registered through the base class's own
    ``_extra_buffers_per_layer`` hook, so allocation, block reuse, slot
    release, and disaggregated bookkeeping keep working. A second manager
    would duplicate request ownership and break exactly those lifecycles.
    """

    def __init__(
        self, *args, sparse_layer_ids: Sequence[int] = (), index_state_dim: int = 0, **kwargs
    ) -> None:
        # Set before super().__init__: the base _build_base_config calls
        # _extra_buffers_per_layer, which reads both of these.
        self.sparse_layer_ids = sorted(int(i) for i in sparse_layer_ids)
        self.index_state_dim = int(index_state_dim)
        if self.sparse_layer_ids and self.index_state_dim <= 0:
            raise ValueError(
                "glm5_next sparse layers need a positive index_state_dim "
                f"(got {self.index_state_dim})"
            )
        super().__init__(*args, **kwargs)

    def _extra_buffers_per_layer(self, *, tokens_per_block: int) -> dict[int, list[BufferConfig]]:
        """One ``Role.INDEX_KEY`` buffer per sparse layer, keyed by local id."""
        elem_bytes = torch.tensor([], dtype=torch.bfloat16).element_size()
        size_per_block = self.index_state_dim * elem_bytes * tokens_per_block
        return {
            self.layer_offsets[layer_id]: [BufferConfig(role=Role.INDEX_KEY, size=size_per_block)]
            for layer_id in self.sparse_layer_ids
            if layer_id in self.layer_offsets
        }

    def get_index_state_buffer(self, layer_idx: int) -> torch.Tensor | None:
        """Paged indexer state for ``layer_idx``, NHD-shaped."""
        return self.get_index_k_buffer(
            layer_idx,
            num_heads=1,
            head_dim=self.index_state_dim,
            dtype=torch.bfloat16,
            kv_layout="NHD",
        )

    def _sparse_pool_id(self) -> int:
        """The single V2 layer-group id that owns every sparse layer.

        Base page indices are per layer group, so one block table can
        address both the latent and the indexer views only because all
        sparse layers -- whose KEY and INDEX_KEY buffers share each
        layer's group -- resolve to one group. Asserted, not assumed:
        a future geometry change that splits the group must fail here
        rather than silently interleave two slot spaces.
        """
        pools = {
            self.layer_to_pool_mapping_dict[self.layer_offsets[layer_id]]
            for layer_id in self.sparse_layer_ids
            if layer_id in self.layer_offsets
        }
        if len(pools) != 1:
            raise ValueError(
                f"glm5_next sparse layers span V2 layer groups {sorted(pools)}; "
                "the slot-indexed latent/index views require a single group"
            )
        return pools.pop()

    def get_batch_slot_tables(self, request_ids: Sequence[int]) -> list[list[int]]:
        """Raw base-slot IDs per request, for the slot-major state views.

        ``get_batch_cache_indices`` is scaled for V2's flattened
        per-layer page views -- it returns ``base * scale // kv_factor``
        (scale is the coalesced buffers-per-slot count; 11 on the real
        checkpoint) -- so feeding its output to the slot-major views from
        :meth:`get_latent_state_buffer` / :meth:`get_index_state_buffer`
        addresses the wrong slots and eventually runs past the pool.
        Those views are indexed by the *base* page id itself, so this
        accessor requests the identity conversion from the same V2
        bookkeeping (``is_kv_aggregate=False, index_scale=1``).

        A pipeline-parallel rank whose local slice holds no sparse layer
        (e.g. the first three GLM layers are all linear attention) has no
        sparse pool at all; it gets empty rows, which no layer on that
        rank ever reads.
        """
        if not any(layer_id in self.layer_offsets for layer_id in self.sparse_layer_ids):
            return [[] for _ in request_ids]
        return self._get_batch_cache_indices_by_pool_id(
            list(request_ids),
            pool_id=self._sparse_pool_id(),
            is_kv_aggregate=False,
            index_scale=1,
        )

    def get_latent_state_buffer(self, layer_idx: int) -> torch.Tensor | None:
        """Paged latent state for ``layer_idx``, addressed by **slot** id.

        ``get_buffers`` hands back a view whose dim-0 stride is a *single*
        page, but V2 coalesces every layer's ``Role.KEY`` buffer into one
        pool and starts layer ``L``'s view ``L`` pages into it. Indexing two
        layers' views with the same block id therefore makes them overlap
        almost entirely -- measured on this checkpoint, eleven sparse layers
        share a 2046-page pool and each layer's page ``p`` is the next
        layer's page ``p - 1``.

        V2's own callers never index that view with a raw block id:
        ``_get_batch_cache_indices_by_pool_id`` multiplies every base page
        index by ``get_layer_page_index_scale(layer_idx)`` first. Folding
        that scale into the *view* rather than into each caller's block
        table is exactly what :meth:`get_index_k_buffer` already does for
        ``Role.INDEX_KEY`` (``[slots, scale, ...][:, 0]``), and it is the
        only convention under which one block table can address both pools:
        the coalesced INDEX_KEY view is expressible *only* slot-indexed.

        Returns ``[num_slots, tokens_per_block, num_kv_heads, head_dim]``,
        the same rank as :meth:`get_index_state_buffer`, so a caller
        decomposes a position into one ``(slot, within_slot)`` pair and uses
        it against both.
        """
        pages = self.get_buffers(layer_idx)
        if pages is None:
            return None
        flat = pages[:, 0]  # [pages, tokens, heads, dim]
        converter = self.impl.get_page_index_converter(self.layer_offsets[layer_idx], Role.KEY)
        # ``scale`` is the buffers-per-slot count of the coalesced pool;
        # ``within_slot`` is where this layer's buffer sits inside a slot.
        scale, within_slot = int(converter.scale), int(converter.layer_offset)
        kv_factor = int(self.kv_factor)
        if scale <= kv_factor:
            return flat
        if scale % kv_factor:
            raise ValueError(
                f"glm5_next layer {layer_idx}: page-index scale {scale} is not a "
                f"multiple of kv_factor {kv_factor}, so the latent pool has no "
                "slot-major view"
            )
        # ``get_buffers`` already folds the kv_factor axis in, so its dim 0
        # counts kv-aggregate pages and the slot stride is the scale in those
        # units. Slot counts are derived the way get_index_k_buffer derives
        # them, so both pools expose the same slot space.
        stride = scale // kv_factor
        slots = (flat.shape[0] + within_slot) // stride
        # Slot s lives at pool page ``within_slot + s * stride``; the last one
        # stays in range because a layer's offset inside a coalesced slot is
        # always smaller than the slot itself.
        if within_slot >= stride or (slots - 1) * stride >= flat.shape[0]:
            raise ValueError(
                f"glm5_next layer {layer_idx}: {slots} slots at stride {stride} do "
                f"not fit {flat.shape[0]} pages from slot offset {within_slot}"
            )
        return torch.as_strided(
            flat,
            size=(slots, *flat.shape[1:]),
            stride=(stride * flat.stride(0), *flat.stride()[1:]),
            storage_offset=flat.storage_offset(),
        )

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
"""Per-step decode metadata for the Inkling Triton attention path."""

from typing import Dict, List, Optional

import torch

from ....._utils import prefer_pinned
from ...trtllm import TrtllmAttentionMetadata


class InklingAttentionMetadata(TrtllmAttentionMetadata):
    """Per-step decode metadata for the Inkling Triton kernels.

    The decode kernel needs two things per generation request: the total KV
    length (``num_cached + 1``) and the physical page table. Building either from
    host lists inside ``model.forward`` is illegal under CUDA-graph capture, so
    both have to reach the kernel through fixed-pointer GPU buffers.

    Only the second is this class's to own. The first is already published by
    ``TrtllmAttentionMetadata.prepare`` into ``kv_lens_cuda`` -- same source
    values, same stability guarantee -- so :meth:`ink_gen_seq_lens` slices that
    buffer instead of staging a private copy. The page table cannot be borrowed
    the same way: the base's ``kv_cache_block_offsets`` is laid out for the C++
    attention op (pool-major, with a K/V pairing axis), while these kernels index
    logical page numbers, so :meth:`_prepare_inkling_decode` builds its own --
    but keyed by *page group* rather than by layer, so layers sharing a KV
    geometry share one build and one copy.

    ``prepare()`` is the framework's "before the forward step" hook and runs on
    every input-preparation path, after the padded batch is assembled and after
    ``super().prepare()`` has re-clamped ``num_cached_tokens_per_seq`` (and
    therefore refreshed ``kv_lens_cuda`` from the clamped values).
    ``ink_num_gen`` is reset at the top of each call, so a step that publishes
    nothing cannot leave the previous step's page table readable.

    ``TrtllmAttentionMetadata`` is the base both because Inkling's backend
    extends ``TrtllmAttention`` and because that is where the reusable per-step
    state lives.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.kv_layout = "HND"
        # Number of generation rows published this step. 0 means the decode
        # buffers hold nothing valid for this forward.
        self.ink_num_gen: int = 0
        self.ink_max_pages: Optional[int] = None
        self.ink_cap: int = 0
        # Keyed by page group, not by layer -- see :meth:`_ink_page_group`.
        self.ink_page_table: Dict[object, torch.Tensor] = {}  # group -> [cap, pages]
        self._ink_layer_groups: Dict[int, object] = {}  # layer -> group key
        self._ink_pt_host: Optional[torch.Tensor] = None
        # Short-conv pool + this forward's context/generation split, published
        # alongside the attention metadata because both are per-step host work
        # that must land in stable buffers before CUDA-graph capture.
        self.ink_conv_cache = None
        self.ink_conv_rt = None

    def _ink_layers(self) -> List[int]:
        """Global decoder-layer indices this rank owns.

        ``pp_layers`` is already the local slice, and the model addresses the
        cache by global layer index.
        """
        return list(getattr(self.kv_cache_manager, "pp_layers", []))

    def _ink_page_group(self, layer: int) -> object:
        """The key whose page table ``layer`` shares.

        ``KVCacheManagerV2.get_batch_cache_indices(ids, layer)`` resolves
        ``layer`` to ``(pool_id, index_scale)`` and returns block indices that
        depend on nothing else, so two layers landing on the same pair get
        byte-identical tables. Inkling has only two KV geometries (local layers
        carry more KV heads than global ones), so keying the page table by the
        pair instead of by the layer collapses one table, one host block-table
        build and one H2D copy *per layer per decode step* down to one per
        geometry. Decode is host-bound, so that ratio is the point.

        Falls back to the layer index when the manager does not expose the
        mapping -- the unit tests' stand-in managers do not, and a per-layer key
        is always correct, just not deduplicated.
        """
        mgr = self.kv_cache_manager
        try:
            offset = mgr.layer_offsets[layer]
            return (mgr.layer_to_pool_mapping_dict[offset], mgr.get_layer_page_index_scale(layer))
        except (AttributeError, KeyError, TypeError):
            return layer

    def _ink_ensure(self, num_gen: int) -> None:
        """Size the stable buffers, refusing to grow them under CUDA graph."""
        mgr = self.kv_cache_manager
        if self.ink_max_pages is None:
            self.ink_max_pages = max(1, int(mgr.max_blocks_per_seq))
        if self.ink_page_table and num_gen <= self.ink_cap:
            return
        if self.is_cuda_graph and self.ink_page_table:
            raise RuntimeError(
                f"InklingAttentionMetadata would grow its stable decode buffers "
                f"during CUDA graph capture/replay (num_gen={num_gen} > "
                f"cap={self.ink_cap}); the buffers are sized to the padded "
                f"scheduler batch, so this signals a capture-shape mismatch"
            )
        self.ink_cap = max(num_gen, self.ink_cap)
        device = self.seq_lens_cuda.device
        self._ink_layer_groups = {layer: self._ink_page_group(layer) for layer in self._ink_layers()}
        self.ink_page_table = {
            group: torch.zeros((self.ink_cap, self.ink_max_pages), dtype=torch.int32, device=device)
            for group in dict.fromkeys(self._ink_layer_groups.values())
        }

    # ---- backend-facing ---------------------------------------------------
    def ink_gen_page_table(self, layer: int) -> torch.Tensor:
        """This step's page table for ``layer`` (shared across its page group)."""
        return self.ink_page_table[self._ink_layer_groups[layer]]

    def ink_gen_seq_lens(self, num_gen: int) -> torch.Tensor:
        """Total-KV length per generation request, on device.

        Reuses the base ``kv_lens_cuda`` rather than staging a second copy:
        ``TrtllmAttentionMetadata.prepare`` already fills it with
        ``num_cached_tokens_per_seq + seq_lens_kv`` from the same (re-clamped)
        source Inkling would read, into a buffer the base already keeps
        CUDA-graph-stable. For a generation row ``seq_lens_kv`` is 1 -- Inkling's
        decode kernel is one query per request -- so this is exactly the
        ``num_cached + 1`` the kernel wants.

        Deliberately *not* ``kv_lens``: the host-side twin adds
        ``num_extra_kv_tokens`` (see ``TrtllmAttentionMetadata.prepare``), which
        this kernel must not see.
        """
        start = self.num_contexts
        return self.kv_lens_cuda[start : start + num_gen]

    def prepare(self) -> None:
        super().prepare()
        self._prepare_inkling_conv()
        self._prepare_inkling_decode()

    def _prepare_inkling_conv(self) -> None:
        """Publish the short-conv pool rows for this batch.

        Runs here rather than from PyTorchModelEngine because prepare() is the
        framework's pre-forward hook and is already called on every input-prep
        path, so the host->device slot write stays outside the captured region.

        Tests for the pool rather than for the concrete manager class: the
        manager owns the pool's *lifetime* (it is freed with the request's KV
        blocks) but nothing about a per-forward batch split, which is metadata
        work and belongs here. Asking for the capability also keeps the test
        honest for the fake managers the unit tests build.

        If a second short-conv model ever appears, the right move is to widen
        the framework's existing hook -- ``BaseMambaCacheManager`` already
        declares ``get_conv_states(layer_idx)`` and three models implement it --
        not to invent a parallel protocol beside it.
        """
        from .conv_state import InklingConvRuntime

        cache = getattr(self.kv_cache_manager, "conv_state_cache", None)
        if cache is None or self.request_ids is None:
            self.ink_conv_cache = self.ink_conv_rt = None
            return
        self.ink_conv_cache = cache
        self.ink_conv_rt = InklingConvRuntime.build(self, cache)

    def _prepare_inkling_decode(self) -> None:
        # Reset first: a step that returns early must not leave the previous
        # step's buffers advertised as current -- a page table one step out of
        # date silently drops a newly allocated page.
        self.ink_num_gen = 0
        mgr = self.kv_cache_manager
        if mgr is None or self.request_ids is None or self.kv_cache_params is None:
            return
        num_contexts = self.num_contexts
        num_gen = len(self.request_ids) - num_contexts
        if num_gen <= 0:
            return
        layers = self._ink_layers()
        if not layers:
            return
        # The captured decode kernel reads ``kv_lens_cuda`` at a fixed offset
        # (see :meth:`ink_gen_seq_lens`), so that offset has to be constant
        # across replays. Decode graphs are pure generation, which makes it 0.
        if self.is_cuda_graph and num_contexts:
            raise RuntimeError(
                f"InklingAttentionMetadata got {num_contexts} context requests in "
                "a CUDA-graph batch; the captured decode kernel reads seq lens at "
                "a fixed offset into kv_lens_cuda, which only holds for pure "
                "generation batches"
            )
        self._ink_ensure(num_gen)

        # One staging row per *page group*, not per layer -- layers sharing a
        # (pool_id, index_scale) pair get byte-identical block tables, so the
        # build and the H2D below run once for each of Inkling's two KV
        # geometries instead of once for each of its layers.
        #
        # A single buffer refilled per group would be torn by the non-blocking
        # copies below (the next group's fill racing the previous group's H2D),
        # and a fresh pinned allocation per group costs one cudaHostAlloc per
        # group per token -- hence one 3-D buffer indexed by group.
        groups = list(self.ink_page_table)
        n_groups = len(groups)
        if (
            self._ink_pt_host is None
            or self._ink_pt_host.shape[0] < n_groups
            or self._ink_pt_host.shape[1] < num_gen
        ):
            self._ink_pt_host = torch.zeros(
                (n_groups, max(num_gen, self.ink_cap), self.ink_max_pages),
                dtype=torch.int32,
                pin_memory=prefer_pinned(),
            )
        gen_ids = self.request_ids[num_contexts:]
        # Any layer of a group resolves to the same block indices; take the first.
        group_layer = {}
        for layer in layers:
            group_layer.setdefault(self._ink_layer_groups[layer], layer)
        pt_np = self._ink_pt_host.numpy()
        pt_np[:n_groups, :num_gen].fill(0)
        for gi, group in enumerate(groups):
            block_ids = mgr.get_batch_cache_indices(gen_ids, group_layer[group])
            # Clamp to num_gen: the staging row is sized to the generation
            # slice, so a manager returning more rows than request ids would
            # otherwise write past it.
            for i, blocks in enumerate(block_ids[:num_gen]):
                valid = [b for b in map(int, blocks) if b >= 0][: self.ink_max_pages]
                if valid:
                    pt_np[gi, i, : len(valid)] = valid
        # Issue the copies only after every row is filled, so no in-flight copy
        # can alias a row this loop still has to write.
        for gi, group in enumerate(groups):
            self.ink_page_table[group][:num_gen].copy_(
                self._ink_pt_host[gi, :num_gen], non_blocking=True
            )
        self.ink_num_gen = num_gen

    def create_cuda_graph_metadata(
        self, max_batch_size: int, *args, **kwargs
    ) -> "InklingAttentionMetadata":
        md = super().create_cuda_graph_metadata(max_batch_size, *args, **kwargs)
        if md is self or md.kv_cache_manager is None:
            return md
        # Same treatment interface.py gives block_ids_per_seq under
        # enable_flash_mla: create_cuda_graph_metadata is a shallow copy, so the
        # graph metadata would otherwise share and then resize the eager
        # metadata's buffers, stranding the captured pointers.
        md.ink_max_pages = max(1, int(md.kv_cache_manager.max_blocks_per_seq))
        md.ink_cap = max_batch_size
        md.ink_num_gen = 0
        md._ink_pt_host = None
        md.ink_conv_cache = None
        md.ink_conv_rt = None
        md._ink_layer_groups = {layer: md._ink_page_group(layer) for layer in md._ink_layers()}
        md.ink_page_table = {
            group: torch.zeros((max_batch_size, md.ink_max_pages), dtype=torch.int32, device="cuda")
            for group in dict.fromkeys(md._ink_layer_groups.values())
        }
        # ``kv_lens_cuda`` needs no treatment here: the base already gives the
        # graph metadata its own capture-pool buffer, which is the whole reason
        # to read seq lens from it instead of staging a private copy.
        return md

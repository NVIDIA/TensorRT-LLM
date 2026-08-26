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
"""Attention metadata for Inkling: the base plus one pre-capture hook."""

import torch

from ...trtllm import TrtllmAttentionMetadata
from .page_table import owned_layers, uses_pool_row


class InklingAttentionMetadata(TrtllmAttentionMetadata):
    """``TrtllmAttentionMetadata`` plus two per-step, pre-capture writes.

    Both are host->device copies into buffers the captured decode graph aliases,
    so they run every step and outside that region -- ``prepare()`` is the only
    hook that does both and still sees the CUDA-graph padding rows. The first is
    the short-conv pool's slot write; the second stages private decode page
    tables for layers whose page-index scale differs from their pool's shared row
    (see ``_stage_scale_fixed_page_tables``).
    """

    def prepare(self) -> None:
        super().prepare()
        cache = getattr(self.kv_cache_manager, "conv_state_cache", None)
        if cache is not None and self.request_ids is not None:
            cache.write_state_indices(self.request_ids)
        self._stage_scale_fixed_page_tables()

    def _stage_scale_fixed_page_tables(self) -> None:
        """Stage a private decode page table for every owned layer whose per-layer
        page-index scale differs from its pool's shared ``kv_cache_block_offsets``
        row.

        Main's V2 manager may group layers of different page-index scales into one
        pool (see ``KVCacheManagerV2.get_layer_page_index_scale``); such a layer
        cannot borrow the pool-scaled row, so its table is built from the
        *per-layer* scale via ``get_batch_cache_indices`` (already block indices)
        into a graph-stable buffer the captured decode graph aliases -- filled
        here, every step, outside that region, exactly like the conv write above.
        ``decode_page_table`` reads these back; scale-matched layers keep the
        zero-copy borrow.
        """
        self._scale_fixed_page_tables = {}
        mgr = self.kv_cache_manager
        if mgr is None or self.request_ids is None:
            return
        # SWA scratch reuse already gives every attention op its own row.
        if getattr(mgr, "enable_swa_scratch_reuse", False):
            return
        num_gen = self.num_generations
        if num_gen <= 0:
            return
        # Group mismatched layers by ``(pool, scale)`` space: a space's layers
        # share one pool's base indices and one scale, so their tables are
        # identical -- build each once and alias it (mirrors flashinfer's VSWA
        # per-space buffers).
        spaces = {}
        for layer in owned_layers(self):
            if uses_pool_row(self, layer):
                continue
            offset = mgr.layer_offsets[layer]
            key = (int(mgr.layer_to_pool_mapping_dict[offset]),
                   int(mgr.get_layer_page_index_scale(layer)))
            spaces.setdefault(key, []).append(layer)
        if not spaces:
            return
        gen_ids = self.request_ids[self.num_contexts : self.num_contexts + num_gen]
        capacity = self.max_num_sequences if self.max_num_sequences is not None else num_gen
        max_blocks = mgr.max_blocks_per_seq
        for (pool_id, scale), layers in spaces.items():
            buf = self.get_empty(
                self.cuda_graph_buffers,
                [capacity, max_blocks],
                dtype=torch.int32,
                cache_name=f"inkling_scale_fixed_pt_p{pool_id}_s{scale}",
                capture_graph=self.is_cuda_graph,
            )
            # Block indices per generation request, padded with 0 (never read: the
            # decode kernel bounds every access by the per-request seq len). Any
            # layer of the space gives the same table.
            host = torch.zeros((num_gen, max_blocks), dtype=torch.int32)
            for i, blocks in enumerate(mgr.get_batch_cache_indices(gen_ids, layers[0])):
                valid = [int(b) for b in blocks if int(b) >= 0]
                if valid:
                    host[i, : len(valid)] = torch.tensor(valid, dtype=torch.int32)
            buf[:num_gen].copy_(host, non_blocking=True)
            for layer in layers:
                self._scale_fixed_page_tables[layer] = buf

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

from ...._utils import prefer_pinned
from ..trtllm import TrtllmAttentionMetadata


class InklingAttentionMetadata(TrtllmAttentionMetadata):
    """Per-step decode metadata for the Inkling Triton kernels.

    The decode kernel needs, per generation request, the total KV length
    (``num_cached + 1``) and the physical page table. Building those from host
    lists inside ``model.forward`` raises ``Cannot copy between CPU and CUDA
    tensors during CUDA graph capture``, so they live in fixed-pointer GPU
    buffers that :meth:`prepare` overwrites each step -- the same shape as the
    base metadata's ``seq_lens_cuda`` in-place ``copy_``, and the same reason
    :meth:`AttentionMetadata.create_cuda_graph_metadata` exists.

    ``prepare()`` is the framework's documented "before the forward step of the
    model" hook and runs on all three input-preparation paths including the
    steady-generation fast path, which is exactly the set of places this used to
    be published from inside ``PyTorchModelEngine``. It runs after the padded
    batch is assembled (``CUDAGraphRunner.pad_batch`` wraps ``_prepare_inputs``)
    and after ``super().prepare()`` has re-clamped
    ``kv_cache_params.num_cached_tokens_per_seq``, so it sees exactly the data
    the model-engine hook saw.

    Freshness needs no epoch counter: the buffers belong to the metadata object
    for the step being prepared, and ``ink_num_gen`` is reset at the top of
    every ``prepare()``, so a step that publishes nothing cannot leave the
    previous step's page table readable.

    ``TrtllmAttentionMetadata`` is the base because that is the backend Inkling
    ran on before this class existed; Inkling reads only base fields, but
    inheriting keeps every other consumer's expectations intact.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.kv_layout = "HND"
        # Number of generation rows published this step. 0 means the decode
        # buffers hold nothing valid for this forward.
        self.ink_num_gen: int = 0
        self.ink_max_pages: Optional[int] = None
        self.ink_cap: int = 0
        self.ink_seq_lens: Optional[torch.Tensor] = None  # [cap] int32 total-KV
        self.ink_page_table: Dict[int, torch.Tensor] = {}  # layer -> [cap, pages]
        self._ink_sl_host: Optional[torch.Tensor] = None
        self._ink_pt_host: Optional[torch.Tensor] = None
        # Short-conv pool + this forward's context/generation split, published
        # alongside the attention metadata because both are per-step host work
        # that must land in stable buffers before CUDA-graph capture.
        self.ink_conv_cache = None
        self.ink_conv_rt = None

    def _ink_layers(self) -> List[int]:
        """Global decoder-layer indices this rank owns.

        ``get_batch_cache_indices`` is per-layer (per pool_id / index_scale), so
        the page table is too. ``pp_layers`` is already the local slice, and the
        model addresses the cache by global layer index.
        """
        return list(getattr(self.kv_cache_manager, "pp_layers", []))

    def _ink_ensure(self, num_gen: int) -> None:
        """Size the stable buffers, refusing to grow them under CUDA graph."""
        mgr = self.kv_cache_manager
        if self.ink_max_pages is None:
            self.ink_max_pages = max(1, int(mgr.max_blocks_per_seq))
        if self.ink_seq_lens is not None and num_gen <= self.ink_cap:
            return
        if self.is_cuda_graph and self.ink_seq_lens is not None:
            raise RuntimeError(
                f"InklingAttentionMetadata would grow its stable decode buffers "
                f"during CUDA graph capture/replay (num_gen={num_gen} > "
                f"cap={self.ink_cap}); the buffers are sized to the padded "
                f"scheduler batch, so this signals a capture-shape mismatch"
            )
        self.ink_cap = max(num_gen, self.ink_cap)
        device = self.seq_lens_cuda.device
        self.ink_seq_lens = torch.ones(self.ink_cap, dtype=torch.int32, device=device)
        self.ink_page_table = {
            layer: torch.zeros((self.ink_cap, self.ink_max_pages), dtype=torch.int32, device=device)
            for layer in self._ink_layers()
        }

    def prepare(self) -> None:
        super().prepare()
        self._prepare_inkling_conv()
        self._prepare_inkling_decode()

    def _prepare_inkling_conv(self) -> None:
        """Publish the short-conv pool rows for this batch.

        Runs here rather than from PyTorchModelEngine because prepare() is the
        framework's pre-forward hook and is already called on every input-prep
        path, so the host->device slot write stays outside the captured region.

        Type-tests the concrete manager rather than an abstract protocol. A
        protocol would have to be satisfied by exactly one class -- both its
        useful methods return Inkling's own pool and runtime types -- so it
        would abstract nothing while putting an Inkling file under
        ``pyexecutor``. The framework rule this looks like it breaks ("test the
        capability, not the model", the way _prepare_mamba_metadata tests
        BaseMambaCacheManager) is about not branching on model identity in
        SHARED code; this class only ever serves Inkling by construction.

        If a second short-conv model ever appears, the right move is to widen
        the framework's existing hook -- ``BaseMambaCacheManager`` already
        declares ``get_conv_states(layer_idx)`` and three models implement it --
        not to invent a parallel protocol beside it.
        """
        from .cache_manager import InklingHybridCacheManager

        mgr = self.kv_cache_manager
        if not isinstance(mgr, InklingHybridCacheManager) or self.request_ids is None:
            self.ink_conv_cache = self.ink_conv_rt = None
            return
        self.ink_conv_cache, self.ink_conv_rt = mgr.prepare_conv_runtime(self)

    def _prepare_inkling_decode(self) -> None:
        # Reset first: a step that returns early below must not leave the
        # previous step's buffers advertised as current. The same requests
        # advance num_cached_tokens_per_seq every step, so a page table one step
        # out of date silently drops a newly allocated page.
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
        self._ink_ensure(num_gen)

        # Total-KV lengths are layer-independent: staged once per step.
        num_cached = self.kv_cache_params.num_cached_tokens_per_seq[num_contexts:]
        if self._ink_sl_host is None or self._ink_sl_host.shape[0] < num_gen:
            self._ink_sl_host = torch.empty(num_gen, dtype=torch.int32, pin_memory=prefer_pinned())
        sl_host = self._ink_sl_host[:num_gen]
        sl_np = sl_host.numpy()
        for i in range(num_gen):
            sl_np[i] = int(num_cached[i]) + 1
        self.ink_seq_lens[:num_gen].copy_(sl_host, non_blocking=True)

        # Reused pinned staging, ONE ROW PER LAYER. It must not be a single
        # [cap, max_pages] buffer refilled per layer: the copies below are
        # non_blocking, so the next layer's fill would overwrite the host bytes
        # while the previous layer's H2D copy is still in flight, and every
        # layer past the first would land a torn page table -- attention then
        # reads the wrong KV pages and decode collapses to repeated tokens.
        # A fresh pinned allocation per layer is not an option either (66
        # cudaHostAllocs a token), hence one 3-D buffer indexed by layer.
        n_layers = len(layers)
        if (
            self._ink_pt_host is None
            or self._ink_pt_host.shape[0] < n_layers
            or self._ink_pt_host.shape[1] < num_gen
        ):
            self._ink_pt_host = torch.zeros(
                (n_layers, max(num_gen, self.ink_cap), self.ink_max_pages),
                dtype=torch.int32,
                pin_memory=prefer_pinned(),
            )
        gen_ids = self.request_ids[num_contexts:]
        pt_np = self._ink_pt_host.numpy()
        pt_np[:n_layers, :num_gen].fill(0)
        for li, layer in enumerate(layers):
            block_ids = mgr.get_batch_cache_indices(gen_ids, layer)
            # Clamp to num_gen: the staging row is sized to the generation
            # slice, so a manager returning more rows than request ids would
            # otherwise write past it.
            for i, blocks in enumerate(block_ids[:num_gen]):
                valid = [b for b in map(int, blocks) if b >= 0][: self.ink_max_pages]
                if valid:
                    pt_np[li, i, : len(valid)] = valid
        # Issue the copies only after every row is filled, so no in-flight copy
        # can alias a row this loop still has to write.
        for li, layer in enumerate(layers):
            self.ink_page_table[layer][:num_gen].copy_(
                self._ink_pt_host[li, :num_gen], non_blocking=True
            )
        self.ink_num_gen = num_gen

    def create_cuda_graph_metadata(
        self, max_batch_size: int, *args, **kwargs
    ) -> "InklingAttentionMetadata":
        md = super().create_cuda_graph_metadata(max_batch_size, *args, **kwargs)
        if md is self or md.kv_cache_manager is None:
            return md
        # Same treatment interface.py gives block_ids_per_seq under
        # enable_flash_mla: create_cuda_graph_metadata is a SHALLOW copy, so the
        # graph metadata would otherwise share -- and then resize -- the eager
        # metadata's buffers, stranding the captured pointers. Allocate at the
        # padded batch size up front so _ink_ensure never grows under capture.
        md.ink_max_pages = max(1, int(md.kv_cache_manager.max_blocks_per_seq))
        md.ink_cap = max_batch_size
        md.ink_num_gen = 0
        md._ink_sl_host = None
        md._ink_pt_host = None
        md.ink_conv_cache = None
        md.ink_conv_rt = None
        md.ink_seq_lens = torch.ones(max_batch_size, dtype=torch.int32, device="cuda")
        md.ink_page_table = {
            layer: torch.zeros((max_batch_size, md.ink_max_pages), dtype=torch.int32, device="cuda")
            for layer in md._ink_layers()
        }
        return md

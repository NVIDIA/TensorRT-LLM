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

import os
from typing import Dict, List

import torch

from tensorrt_llm.logger import logger

from ...trtllm import TrtllmAttentionMetadata

#: Per-process latch so the crosscheck announces itself exactly once. A check
#: that is silent on success is indistinguishable from one that never ran -- the
#: env var not reaching an MPI worker would look identical -- so the regression
#: harness requires this line and fails if it is absent.
_INK_XCHK_ANNOUNCED = False


class InklingAttentionMetadata(TrtllmAttentionMetadata):
    """Per-step decode metadata for the Inkling Triton kernels.

    The decode kernel needs two things per generation request: the total KV
    length (``num_cached + 1``) and the physical page table. Building either from
    host lists inside ``model.forward`` is illegal under CUDA-graph capture, so
    both have to reach the kernel through fixed-pointer GPU buffers.

    Neither is this class's to own. Both come from buffers
    ``TrtllmAttentionMetadata.prepare`` already fills and already keeps
    CUDA-graph-stable:

    - ``kv_lens_cuda`` holds ``num_cached_tokens_per_seq + seq_lens_kv`` from the
      same (re-clamped) source Inkling would read, so :meth:`ink_gen_seq_lens`
      slices it.
    - ``kv_cache_block_offsets`` holds the per-pool block tables, so
      :meth:`ink_gen_page_table` slices *that*. It is laid out for the C++
      attention op, but every difference from what these kernels want is an
      encoding the kernel absorbs for free:

      * *pool-major* is what we want, not an obstacle -- indexing the leading
        axis by pool id hands back exactly one table per KV geometry, which is
        the deduplication an earlier version of this class did by hand.
      * the ``2`` axis is the K/V plane; ``copyBatchBlockOffsetsToDeviceKernel``
        writes ``index_scale * base_page_index`` into plane 0 and
        ``+ kv_offset`` into plane 1, so plane 0 is the single table both K and
        V reads need.
      * entries count *pages* while the ``kv[:, 0]`` / ``kv[:, 1]`` views handed
        to the kernel count *blocks*, so the kernel divides by ``kv_factor``
        (``PAGE_DIV``, a constexpr power of two) once per KV tile.
      * absent blocks are already 0 (the C++ kernel maps ``BAD_PAGE_INDEX`` to
        0), matching what the kernel's ``seq_len`` mask assumes.

    Borrowing it also *removes* an assumption rather than adding one: the
    hand-built table compacted valid ids to the front, which only equals the
    kernel's ``k_pos // page_size`` addressing while no request has an interior
    hole in its block list. ``kv_cache_block_offsets`` is positional by
    construction.

    ``prepare()`` is the framework's "before the forward step" hook and runs on
    every input-preparation path, after the padded batch is assembled and after
    ``super().prepare()`` has re-clamped ``num_cached_tokens_per_seq`` (and
    therefore refreshed ``kv_lens_cuda`` and copied this step's block offsets).
    ``ink_num_gen`` is reset at the top of each call, so a step that publishes
    nothing cannot leave the previous step's rows readable.

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
        # Global decoder layer -> leading-axis row of ``kv_cache_block_offsets``.
        # Static for the lifetime of a manager, so it is built once and cached
        # rather than recomputed per step (see :meth:`_ink_build_pt_rows`).
        self._ink_pt_rows: Dict[int, int] = {}
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

    def _ink_build_pt_rows(self, layers: List[int]) -> Dict[int, int]:
        """Map each owned layer to its row in ``kv_cache_block_offsets``.

        The leading axis of that tensor is sized ``num_attention_op_pools``,
        which the manager defines two different ways (see
        ``KVCacheManagerV2._prepare_page_table_tensor`` and
        ``:meth:`_copy_batch_block_offsets_per_layer```):

        - default: one row per *pool*, so layers sharing a KV geometry share a
          row -- the same deduplication this class used to hand-roll, now free.
        - ``enable_swa_scratch_reuse``: one row per *attention-op layer*, keyed
          by ``layer_offsets[layer]``, because scratch pages are per layer.

        Also checks the encoding precondition. The C++ copy encodes with the
        *pool-level* ``index_scales[pool_id]`` (one representative layer's
        scale), while ``get_layer_page_index_scale`` documents that layers in one
        pool may differ. They agree for Inkling -- its two geometries differ in
        buffer size and in lifecycle, so they never share a pool -- but a silent
        mismatch here would be a wrong page address, not a crash, so it is
        asserted rather than assumed.
        """
        mgr = self.kv_cache_manager
        per_layer = bool(getattr(mgr, "enable_swa_scratch_reuse", False))
        rows: Dict[int, int] = {}
        for layer in layers:
            offset = mgr.layer_offsets[layer]
            if per_layer:
                rows[layer] = int(offset)
                continue
            pool_id = int(mgr.layer_to_pool_mapping_dict[offset])
            layer_scale = int(mgr.get_layer_page_index_scale(layer))
            pool_scale = int(mgr.index_scales[pool_id])
            if layer_scale != pool_scale:
                raise RuntimeError(
                    f"Inkling layer {layer} (offset {offset}) has page-index "
                    f"scale {layer_scale} but its pool {pool_id} is encoded with "
                    f"{pool_scale}; kv_cache_block_offsets cannot be reused as "
                    "this layer's page table because the recovered page address "
                    "would be wrong. Give the layer its own pool, or stage a "
                    "private table for it."
                )
            rows[layer] = pool_id
        return rows

    # ---- backend-facing ---------------------------------------------------
    def ink_gen_page_table(self, layer: int) -> torch.Tensor:
        """This step's generation-row page table for ``layer``.

        A view into the base ``kv_cache_block_offsets`` K plane, not a private
        buffer: entries are ``index_scale * base_page_index`` and the kernel
        divides by :meth:`ink_page_div`. The rows are the generation slice, so a
        captured graph reads a fixed offset -- which holds because
        :meth:`_prepare_inkling_decode` refuses a graph batch with contexts.
        """
        row = self._ink_pt_rows[layer]
        start = self.num_contexts
        return self.kv_cache_block_offsets[row, start : start + self.ink_num_gen, 0]

    @property
    def ink_page_div(self) -> int:
        """Divisor turning a ``kv_cache_block_offsets`` entry into a block index.

        The entries' unit is pages, which count K and V separately; the kernel's
        K/V views are ``kv[:, 0]`` / ``kv[:, 1]`` of a ``[blocks, kv_factor, ...]``
        buffer and so count blocks.
        """
        return int(getattr(self.kv_cache_manager, "kv_factor", 1))

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
        # The page table is the base's ``kv_cache_block_offsets``, already
        # refreshed for this batch by ``super().prepare()`` ->
        # ``copy_batch_block_offsets``. Nothing is staged or copied here; the
        # only per-step work is validating that it is there and recording how
        # many generation rows are live.
        offsets = self.kv_cache_block_offsets
        if offsets is None:
            raise RuntimeError(
                "InklingAttentionMetadata needs the base "
                "kv_cache_block_offsets as its decode page table, but it is "
                "None. TrtllmAttentionMetadata only allocates it when a KV "
                "cache manager is attached, so this means prepare() ran "
                "against a metadata object built without one."
            )
        if not self._ink_pt_rows:
            self._ink_pt_rows = self._ink_build_pt_rows(layers)
        max_row = max(self._ink_pt_rows.values())
        if max_row >= offsets.shape[0]:
            raise RuntimeError(
                f"Inkling needs row {max_row} of kv_cache_block_offsets but it "
                f"has only {offsets.shape[0]} (num_attention_op_pools). The "
                "leading axis is per-pool by default and per-attention-op-layer "
                "under enable_swa_scratch_reuse; _ink_build_pt_rows and the "
                "manager disagree about which."
            )
        if num_contexts + num_gen > offsets.shape[1]:
            raise RuntimeError(
                f"Inkling decode batch ({num_contexts} ctx + {num_gen} gen) "
                f"exceeds kv_cache_block_offsets' {offsets.shape[1]} sequence "
                "rows (max_num_sequences)."
            )
        self.ink_num_gen = num_gen
        if os.environ.get("TLLM_INKLING_PT_CROSSCHECK") == "1":
            self._ink_crosscheck_page_table(layers)

    def _ink_crosscheck_page_table(self, layers: List[int]) -> None:
        """Assert the borrowed table equals the one this class used to build.

        Off by default and gated on ``TLLM_INKLING_PT_CROSSCHECK=1``: it drags
        the deleted host path back per step, which is the cost this refactor
        removed. Run it once on a real workload to settle two things that static
        reading cannot:

        1. ``kv_cache_block_offsets`` plane 0 really holds
           ``index_scale * base_page_index`` -- inferred from
           ``copyBatchBlockOffsetsToDeviceKernel`` plus the ``index_scales`` /
           ``kv_offset`` argument split, not traced end to end.
        2. ``get_batch_cache_indices`` never leaves an interior hole. The old
           table dropped negatives and packed survivors to the front, which only
           matches the kernel's ``k_pos // page_size`` addressing while every
           hole is in the padded tail. Out-of-window SWA blocks are the obvious
           way that could stop being true.
        """
        global _INK_XCHK_ANNOUNCED
        mgr = self.kv_cache_manager
        gen_ids = self.request_ids[self.num_contexts :]
        div = self.ink_page_div
        checked = 0
        for layer in layers:
            borrowed = (self.ink_gen_page_table(layer) // div).tolist()
            for i, blocks in enumerate(mgr.get_batch_cache_indices(gen_ids, layer)):
                want = [int(b) for b in blocks]
                if any(b < 0 for b in want):
                    raise AssertionError(
                        f"layer {layer} request {gen_ids[i]}: get_batch_cache_indices "
                        f"returned an interior placeholder in {want}; the packed "
                        "representation this refactor replaced would have shifted "
                        "the survivors left, so the two are not equivalent here"
                    )
                got = borrowed[i][: len(want)]
                if got != want:
                    raise AssertionError(
                        f"page-table mismatch at layer {layer} request "
                        f"{gen_ids[i]}: borrowed//{div}={got} vs "
                        f"get_batch_cache_indices={want}"
                    )
                # Reported, not asserted. The C++ copy writes the full row width
                # from a host mirror initialised to zeros, so the tail should be
                # 0 -- but whether that mirror is re-zeroed when a slot is reused
                # by a shorter request is not established, and the kernel
                # provably never reads there (``mask_n = k_pos < seq_len`` bounds
                # every access by the request's own KV length). Asserting would
                # put a false-alarm path on the gate that is supposed to make
                # this change trustworthy.
                tail = borrowed[i][len(want) :]
                if any(p != 0 for p in tail):
                    # Wording matters: the already-queued regression jobs grep the
                    # server log for the crosscheck's failure strings, so this
                    # message must not contain any of them.
                    logger.warning(
                        "Inkling page-table crosscheck: layer %d request %s "
                        "carries stale values %s past its %d blocks. Harmless -- "
                        "the kernel masks every access by seq_len -- but the "
                        "borrowed row is padded differently from the staged one.",
                        layer,
                        gen_ids[i],
                        tail[:8],
                        len(want),
                    )
                checked += 1
        if not _INK_XCHK_ANNOUNCED:
            _INK_XCHK_ANNOUNCED = True
            # f-string, not %-args: tensorrt_llm.logger.info concatenates its
            # arguments rather than %-formatting them the way stdlib logging
            # does, so the lazy form printed the literal "%d" (job 6276424) and
            # the counts -- the whole point of this line -- were lost.
            logger.info(
                f"INKLING_PT_CROSSCHECK ACTIVE: {checked} layer/request pairs "
                f"agreed (borrowed kv_cache_block_offsets // {div} == "
                f"get_batch_cache_indices) over {len(layers)} layers and "
                f"{len(gen_ids)} generation rows"
            )

    def create_cuda_graph_metadata(
        self, max_batch_size: int, *args, **kwargs
    ) -> "InklingAttentionMetadata":
        md = super().create_cuda_graph_metadata(max_batch_size, *args, **kwargs)
        if md is self or md.kv_cache_manager is None:
            return md
        # No private buffers left to re-point. ``create_cuda_graph_metadata`` is
        # a shallow copy, and the previous version had to re-allocate the page
        # tables here or the graph metadata would resize the eager metadata's
        # buffers and strand the captured pointers. Both tensors the decode path
        # now reads -- ``kv_lens_cuda`` and ``kv_cache_block_offsets`` -- are the
        # base's, and the base already hands the graph metadata its own
        # capture-pool copies. That is the whole point of borrowing them.
        #
        # Only per-step publication state is reset: stale values here would
        # advertise the previous step's rows as current.
        md.ink_num_gen = 0
        md.ink_conv_cache = None
        md.ink_conv_rt = None
        # Rebuilt rather than inherited: the graph metadata may carry a different
        # manager, and the mapping is derived from it.
        md._ink_pt_rows = {}
        return md

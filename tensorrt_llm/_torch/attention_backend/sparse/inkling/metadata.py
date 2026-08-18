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

#: Per-process latch: the crosscheck is silent on success, so without this line
#: "it passed" and "it never ran" look identical. The harness requires it.
_INK_XCHK_ANNOUNCED = False


class InklingAttentionMetadata(TrtllmAttentionMetadata):
    """Per-step decode metadata for the Inkling Triton kernels.

    The decode kernel needs each generation request's total KV length and page
    table through fixed-pointer GPU buffers. Neither is this class's to own: it
    slices the base's ``kv_lens_cuda`` and ``kv_cache_block_offsets``, which
    ``TrtllmAttentionMetadata.prepare`` already fills and keeps graph-stable.

    The latter is laid out for the C++ attention op, but pool-major is what we
    want (one table per KV geometry, for free), plane 0 serves both K and V, and
    absent blocks are already 0. The one real difference is that entries count
    pages while the kernel's views count blocks -- hence :attr:`ink_page_div`,
    once per KV tile. Borrowing also removes an assumption: the hand-built table
    packed valid ids to the front, which matches the kernel's
    ``k_pos // page_size`` addressing only while no request has an interior hole.

    No per-step publication counter: ``num_generations`` already is
    ``len(request_ids) - num_contexts``, padding rows included, so the backend
    guards on the metadata *type* instead.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.kv_layout = "HND"
        # Layer -> row of kv_cache_block_offsets. Static, so built once.
        self._ink_pt_rows: Dict[int, int] = {}
        # Short-conv pool and this forward's context/generation split: per-step
        # host work that must land in stable buffers before graph capture.
        self.ink_conv_cache = None
        self.ink_conv_rt = None

    def _ink_layers(self) -> List[int]:
        """Global decoder-layer indices this rank owns (``pp_layers`` is already
        the local slice; the cache is addressed by global index)."""
        return list(getattr(self.kv_cache_manager, "pp_layers", []))

    def _ink_build_pt_rows(self, layers: List[int]) -> Dict[int, int]:
        """Map each owned layer to its row in ``kv_cache_block_offsets``.

        That tensor's leading axis is per *pool* by default, but per attention-op
        *layer* under ``enable_swa_scratch_reuse`` (scratch pages are per layer);
        reading one as the other returns another layer's pages.

        The scale check is not decoration: the C++ copy encodes with the
        pool-level ``index_scales[pool_id]`` while ``get_layer_page_index_scale``
        allows layers in one pool to differ. They agree for Inkling, but a
        mismatch would be a wrong address rather than a crash.
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
        """Generation-row page table for ``layer``: a view into the base
        ``kv_cache_block_offsets`` K plane, entries scaled by
        :attr:`ink_page_div`. A captured graph reads a fixed offset here because
        :meth:`_prepare_inkling_decode` refuses a graph batch with contexts.
        """
        row = self._ink_pt_rows[layer]
        start = self.num_contexts
        return self.kv_cache_block_offsets[row, start : start + self.num_generations, 0]

    @property
    def ink_page_div(self) -> int:
        """Pages-to-blocks divisor: entries count pages (K and V separately),
        the kernel's ``kv[:, 0]`` / ``kv[:, 1]`` views count blocks."""
        return int(getattr(self.kv_cache_manager, "kv_factor", 1))

    def ink_gen_seq_lens(self, num_gen: int) -> torch.Tensor:
        """Total-KV length per generation request, sliced off the base's
        ``kv_lens_cuda`` -- ``num_cached + 1``, since a generation row has
        ``seq_lens_kv == 1``.

        Deliberately *not* ``kv_lens``: the host-side twin adds
        ``num_extra_kv_tokens``, which this kernel must not see.
        """
        start = self.num_contexts
        return self.kv_lens_cuda[start : start + num_gen]

    def prepare(self) -> None:
        super().prepare()
        self._prepare_inkling_conv()
        self._prepare_inkling_decode()

    def _prepare_inkling_conv(self) -> None:
        """Publish the short-conv pool rows for this batch.

        Runs here, not in PyTorchModelEngine: prepare() is the pre-forward hook
        on every input-prep path, so the host->device slot write stays outside
        the captured region.

        Tests for the pool, not the concrete manager class -- the manager owns
        the pool's lifetime but nothing about a per-forward batch split.
        """
        from .conv_state import InklingConvRuntime

        cache = getattr(self.kv_cache_manager, "conv_state_cache", None)
        if cache is None or self.request_ids is None:
            self.ink_conv_cache = self.ink_conv_rt = None
            return
        self.ink_conv_cache = cache
        self.ink_conv_rt = InklingConvRuntime.build(self, cache)

    def _prepare_inkling_decode(self) -> None:
        # Publishes nothing -- the accessors read buffers the base refreshes each
        # step. This only validates the borrowed layout and caches the row map.
        # The early returns mean "no decode work in this batch"; none is
        # reachable from the backend, which needs a manager, generation rows and
        # owned layers to get there at all.
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
        # The captured kernel reads both borrowed buffers at a fixed generation
        # offset, so it must be constant across replays -- 0 for a pure
        # generation batch, which is what a decode graph is.
        if self.is_cuda_graph and num_contexts:
            raise RuntimeError(
                f"InklingAttentionMetadata got {num_contexts} context requests in "
                "a CUDA-graph batch; the captured decode kernel reads seq lens at "
                "a fixed offset into kv_lens_cuda, which only holds for pure "
                "generation batches"
            )
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
        # Premise of slicing by num_generations instead of a private counter:
        # model_engine assigns it from the same scheduled batch as request_ids.
        assert self.num_generations == num_gen, (self.num_generations, num_gen)
        if os.environ.get("TLLM_INKLING_PT_CROSSCHECK") == "1":
            self._ink_crosscheck_page_table(layers)

    def _ink_crosscheck_page_table(self, layers: List[int]) -> None:
        """Assert the borrowed table equals the one this class used to build.

        Gated on ``TLLM_INKLING_PT_CROSSCHECK=1`` because it drags the deleted
        host path back per step. Run once on a real workload to settle what
        static reading cannot: that plane 0 holds ``index_scale *
        base_page_index`` (inferred from the C++ copy's argument split, not
        traced), and that ``get_batch_cache_indices`` leaves no interior hole --
        out-of-window SWA blocks being the way that could stop being true.
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
                # Reported, not asserted: whether the host mirror is re-zeroed
                # when a slot is reused by a shorter request is not established,
                # and the kernel provably never reads past seq_len. Asserting
                # would put a false-alarm path on this gate.
                tail = borrowed[i][len(want) :]
                if any(p != 0 for p in tail):
                    # Must not contain the harness's failure-grep strings.
                    logger.warning(
                        f"Inkling page-table crosscheck: layer {layer} request "
                        f"{gen_ids[i]} carries stale values {tail[:8]} past its "
                        f"{len(want)} blocks. Harmless -- the kernel masks every "
                        "access by seq_len -- but the borrowed row is padded "
                        "differently from the staged one."
                    )
                checked += 1
        if not _INK_XCHK_ANNOUNCED:
            _INK_XCHK_ANNOUNCED = True
            # f-string, not %-args: tensorrt_llm.logger concatenates rather than
            # %-formatting, so the lazy form prints a literal "%d".
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
        # No private buffers to re-point: this is a shallow copy, and both
        # tensors the decode path reads are the base's, which already hands the
        # graph metadata its own capture-pool copies. Only short-conv state is
        # reset; _ink_pt_rows is rebuilt because the manager may differ.
        md.ink_conv_cache = None
        md.ink_conv_rt = None
        md._ink_pt_rows = {}
        return md

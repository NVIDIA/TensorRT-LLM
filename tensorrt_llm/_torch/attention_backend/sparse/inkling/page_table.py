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
"""Decode-time views into the attention metadata for Inkling's Triton kernels.

The kernel needs each generation request's total KV length and page table through
fixed-pointer GPU buffers. Both are slices of ``kv_lens_cuda`` and
``kv_cache_block_offsets``, which ``prepare()`` already fills and keeps
graph-stable, so these are free functions rather than fields on a subclass. The
page table is laid out for the C++ attention op: plane 0 serves both K and V, but
its entries count pages while the kernel's views count blocks, hence
:func:`page_div`.
"""

import os
from typing import List

import torch

from tensorrt_llm.logger import logger

# Per-process latch: the crosscheck is silent on success, so announce it once.
_INK_XCHK_ANNOUNCED = False


def owned_layers(md) -> List[int]:
    """Global decoder-layer indices this rank owns."""
    return list(getattr(md.kv_cache_manager, "pp_layers", []))


def pt_row(md, layer: int) -> int:
    """Row of ``kv_cache_block_offsets`` holding ``layer``'s page table.

    The tensor's leading axis is per pool by default but per attention-op layer
    under ``enable_swa_scratch_reuse``; reading one as the other returns another
    layer's pages. The scale check guards the same confusion: the C++ copy encodes
    with the pool-level scale, which need not equal the layer's.
    """
    mgr = md.kv_cache_manager
    offset = mgr.layer_offsets[layer]
    if getattr(mgr, "enable_swa_scratch_reuse", False):
        return int(offset)
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
    return pool_id


def gen_page_table(md, layer: int) -> torch.Tensor:
    """Generation-row page table for ``layer``: a view into the base
    ``kv_cache_block_offsets`` K plane, entries scaled by :func:`page_div`."""
    row = pt_row(md, layer)
    start = md.num_contexts
    return md.kv_cache_block_offsets[row, start : start + md.num_generations, 0]


def page_div(md) -> int:
    """Pages-to-blocks divisor: entries count pages (K and V separately), the
    kernel's ``kv[:, 0]`` / ``kv[:, 1]`` views count blocks."""
    return int(getattr(md.kv_cache_manager, "kv_factor", 1))


def uses_pool_row(md, layer: int) -> bool:
    """True when ``layer`` may borrow its pool's shared ``kv_cache_block_offsets``
    row as its page table.

    Two cases allow the borrow: ``enable_swa_scratch_reuse`` gives every attention
    op its own row (``pt_row`` returns the offset directly), or the layer's
    page-index scale already equals its pool's, so the pool-scaled row is correct
    for this layer. Otherwise main's V2 manager has placed the layer in a pool
    whose representative scale differs from the layer's (see
    ``get_layer_page_index_scale`` -- mixed-scale pools are allowed), the borrowed
    row would recover wrong page addresses, and ``InklingAttentionMetadata`` must
    stage a private table instead (see ``decode_page_table``).
    """
    mgr = md.kv_cache_manager
    if getattr(mgr, "enable_swa_scratch_reuse", False):
        return True
    offset = mgr.layer_offsets[layer]
    pool_id = int(mgr.layer_to_pool_mapping_dict[offset])
    return int(mgr.get_layer_page_index_scale(layer)) == int(mgr.index_scales[pool_id])


def decode_page_table(md, layer: int, num_req: int):
    """Decode-row page table and its page divisor for ``layer``.

    Scale-matched layers borrow the graph-stable per-pool row (page indices,
    divided by ``kv_factor`` downstream, hence divisor :func:`page_div`).
    Scale-mismatched layers read a private table staged in
    :meth:`InklingAttentionMetadata.prepare` from the *per-layer* scale via
    ``get_batch_cache_indices``, which already divides by ``kv_factor``; those are
    block indices, so their divisor is 1.
    """
    fixed = getattr(md, "_scale_fixed_page_tables", None)
    if fixed is not None and layer in fixed:
        return fixed[layer][:num_req], 1
    return gen_page_table(md, layer)[:num_req], page_div(md)


def gen_seq_lens(md, num_gen: int) -> torch.Tensor:
    """Total-KV length per generation request, sliced off the base's
    ``kv_lens_cuda``. Deliberately not ``kv_lens``: the host-side twin adds
    ``num_extra_kv_tokens``, which this kernel must not see."""
    start = md.num_contexts
    return md.kv_lens_cuda[start : start + num_gen]


def validate_decode_layout(md, layer: int, num_gen: int) -> None:
    """Check the borrowed layout before the decode kernel reads it.

    Cheap enough to run per layer: a dict lookup plus three comparisons.
    """
    num_contexts = md.num_contexts
    # The captured kernel reads both borrowed buffers at a fixed generation
    # offset, so it must be constant across replays -- 0 for a pure generation
    # batch, which is what a decode graph is.
    if md.is_cuda_graph and num_contexts:
        raise RuntimeError(
            f"Inkling got {num_contexts} context requests in a CUDA-graph "
            "batch; the captured decode kernel reads seq lens at a fixed offset "
            "into kv_lens_cuda, which only holds for pure generation batches"
        )
    offsets = md.kv_cache_block_offsets
    if offsets is None:
        raise RuntimeError(
            "Inkling has no kv_cache_block_offsets to borrow as its decode page "
            "table. prepare() fills it for any batch that reaches a kernel, so "
            "its absence is a setup error worth naming -- without this the next "
            "line reports it as AttributeError on NoneType."
        )
    if uses_pool_row(md, layer):
        row = pt_row(md, layer)
        if row >= offsets.shape[0]:
            raise RuntimeError(
                f"Inkling needs row {row} of kv_cache_block_offsets but it has "
                f"only {offsets.shape[0]} (num_attention_op_pools). The leading "
                "axis is per-pool by default and per-attention-op-layer under "
                "enable_swa_scratch_reuse; pt_row and the manager disagree about "
                "which."
            )
    elif getattr(md, "_scale_fixed_page_tables", {}).get(layer) is None:
        raise RuntimeError(
            f"Inkling layer {layer} needs a private decode page table -- its "
            "page-index scale differs from its pool's shared row -- but "
            "InklingAttentionMetadata.prepare did not stage one. prepare() must "
            "run (and see this layer as owned) before the decode kernel reads it."
        )
    if num_contexts + num_gen > offsets.shape[1]:
        raise RuntimeError(
            f"Inkling decode batch ({num_contexts} ctx + {num_gen} gen) exceeds "
            f"kv_cache_block_offsets' {offsets.shape[1]} sequence rows "
            "(max_num_sequences)."
        )
    if os.environ.get("TLLM_INKLING_PT_CROSSCHECK") == "1":
        layers = owned_layers(md)
        # Once per step, not once per layer: the crosscheck walks every layer.
        if layers and layer == layers[0]:
            crosscheck_page_table(md, layers)


def crosscheck_page_table(md, layers: List[int]) -> None:
    """Assert the borrowed table matches ``get_batch_cache_indices``.

    Gated on ``TLLM_INKLING_PT_CROSSCHECK=1`` because it rebuilds the host page
    table every step; run it once on a real workload after touching the borrowed
    layout.
    """
    global _INK_XCHK_ANNOUNCED
    mgr = md.kv_cache_manager
    gen_ids = md.request_ids[md.num_contexts :]
    num_gen = len(gen_ids)
    checked = 0
    for layer in layers:
        # Per-layer table + divisor: scale-mismatched layers read a private table
        # (divisor 1) instead of the pool row, so ``div`` is not layer-invariant.
        pt, div = decode_page_table(md, layer, num_gen)
        borrowed = (pt // div).tolist()
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
            # Reported, not asserted: the kernel masks every access by seq_len,
            # so stale padding past the last block is harmless.
            tail = borrowed[i][len(want) :]
            if any(p != 0 for p in tail):
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
        logger.info(
            f"INKLING_PT_CROSSCHECK ACTIVE: {checked} layer/request pairs "
            f"agreed (borrowed kv_cache_block_offsets // {div} == "
            f"get_batch_cache_indices) over {len(layers)} layers and "
            f"{len(gen_ids)} generation rows"
        )

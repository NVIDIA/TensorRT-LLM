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
"""Model-agnostic mixed-modality encode path: demux -> encode -> mux.

A single-modality request is the degenerate (identity-permutation) case.
Per-model code supplies (a) an extractor that walks each :class:`MultimodalParams`
and yields :class:`ModalityItem` instances, and (b) per-modality encoder adapters
that bridge a bucket of items to the model's existing encoder call. The pipeline
is three free functions:

  - `build_scatter_index` DEMULTIPLEXES the flat item list into per-modality
    buckets and computes each item's batch-relative destination range.
  - `encoder_dispatch` ENCODES one channel per active modality (one encoder
    launch each).
  - `scatter_to_prompt_order` MULTIPLEXES the per-modality outputs back into the
    canonical prompt-ordered buffer.

`encode_by_modality_and_scatter` is the model-facing orchestrator that chains the
three and keeps the per-param `sum(rows) == total_embeds_in_request` cross-check.

Every item owns exactly one prompt slot. There are no ghosts and no shared slots:
an item's `rows` is both the number of rows its encoder emits and the size of its
scatter destination range. Interleaved repeated modalities (`image -> video ->
image`, `video(+audio)` promoted to a first-class `(audio, k)` item) are handled
by the plain per-item scatter — no post-process re-placement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, NamedTuple

import torch

from tensorrt_llm.inputs.multimodal import MultimodalParams


@dataclass(frozen=True, slots=True)
class ModalityItem:
    """One single-modality payload (one image, video, or audio) as a unit of
    work in the mixed-modality encode pipeline.

    Canonical six-field item: every item owns exactly one prompt slot
    (`prompt_pos`) and its `rows` is both the encoder-output row count and its
    scatter-destination footprint. There are no flags, no ghosts, and no shared
    slots.

    Fields:
        src_param_idx: which request in the batch this item belongs to.
        modality: which encoder bucket the item rides (image / video / audio).
        mm_idx_per_modality: which blob in `multimodal_data[modality]` this item
            is, used to slice the per-item encoder payload.
        prompt_pos: this item's rank in the prompt-order stream; it OWNS this
            slot, so each `prompt_pos` appears at most once per source param.
        rows: encoder-output row count == this item's scatter footprint.
        payload: the single-item encoder input (already sliced).

    Placement (bucket slot, destination range) is derived by `build_scatter_index`
    and is never stored on the item.
    """

    src_param_idx: int
    modality: str
    mm_idx_per_modality: int
    prompt_pos: int
    rows: int
    payload: Mapping[str, Any]


ItemExtractor = Callable[[int, MultimodalParams], Iterable[ModalityItem]]


class ScatterIndices(NamedTuple):
    """Batch index tensors that drive the per-modality scatter. Plain data."""

    modality_slots: Dict[str, torch.Tensor]  # flat item indices per modality
    dst_indices: Dict[str, torch.Tensor]  # destination rows per modality
    param_lengths: torch.Tensor  # int64[num_params], rows per param
    total_tokens: int
    active_modalities: List[str]


def build_scatter_index(items: List[ModalityItem], num_params: int) -> ScatterIndices:
    """Partition items into per-modality buckets and compute batch-relative
    destination rows. Per-request uniqueness is already guaranteed by
    `MixedModalItemOrder.__post_init__`; this is pure index math.
    """
    n = len(items)
    if n == 0:
        return ScatterIndices({}, {}, torch.zeros(num_params, dtype=torch.int64), 0, [])

    # One pass lifts Python ModalityItem fields into parallel int64 tensors (the
    # Python-object -> tensor boundary); everything after is vectorized.
    src = torch.tensor([it.src_param_idx for it in items], dtype=torch.int64)
    rows = torch.tensor([it.rows for it in items], dtype=torch.int64)
    pos = torch.tensor([it.prompt_pos for it in items], dtype=torch.int64)
    active_modalities = list(dict.fromkeys(it.modality for it in items))  # first-seen order
    code = {m: i for i, m in enumerate(active_modalities)}
    mcode = torch.tensor([code[it.modality] for it in items], dtype=torch.int64)

    # Per-param row totals via scatter_add (no Python accumulation loop).
    param_lengths = torch.zeros(num_params, dtype=torch.int64).scatter_add_(0, src, rows)

    # Each item's destination start == exclusive prefix sum of `rows` in canonical
    # (src_param, prompt_pos) order. The canonical buffer is items concatenated by
    # param then by prompt order, so an item's start is the rows ordered before it.
    # One argsort + one cumsum replaces the per-param sort-and-accumulate loop.
    sort_key = src * (int(pos.max()) + 1) + pos
    order = torch.argsort(sort_key)
    excl_sorted = torch.cumsum(rows[order], dim=0) - rows[order]
    dst_start = torch.empty(n, dtype=torch.int64)
    dst_start[order] = excl_sorted  # unsort back to original item order

    # Buckets + destination rows. Only remaining loop is over distinct modalities
    # (<= 3); each body is a vectorized nonzero + _expand_ranges (no per-item loop).
    modality_slots: Dict[str, torch.Tensor] = {}
    dst_indices: Dict[str, torch.Tensor] = {}
    for modality, c in code.items():
        slots = (mcode == c).nonzero(as_tuple=True)[0]
        modality_slots[modality] = slots
        dst_indices[modality] = _expand_ranges(dst_start[slots], rows[slots])

    return ScatterIndices(
        modality_slots=modality_slots,
        dst_indices=dst_indices,
        param_lengths=param_lengths,
        total_tokens=int(param_lengths.sum().item()),
        active_modalities=active_modalities,
    )


def _expand_ranges(starts: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Concatenate per-segment ranges ``[s, s + l)`` for paired (starts, lengths).

    Vectorized equivalent of
    ``torch.cat([torch.arange(s, s + l) for s, l in zip(starts, lengths)])``,
    built with a single `repeat_interleave` + `arange` (no per-token Python
    loop). Both inputs are int64 1-D tensors of equal length; returns an empty
    int64 tensor when there are no rows.
    """
    total = int(lengths.sum()) if lengths.numel() > 0 else 0
    if total == 0:
        return torch.empty(0, dtype=torch.int64)
    # Exclusive prefix sum: each segment's start offset within the output.
    seg_start_in_out = torch.cumsum(lengths, dim=0) - lengths
    within = torch.arange(total, dtype=torch.int64) - seg_start_in_out.repeat_interleave(lengths)
    base = starts.repeat_interleave(lengths)
    return base + within


# Adapter contract: given a bucket's items and the source-param list,
# return one tensor of shape (sum(item.rows for item in items), hidden_dim)
# with rows in bucket order (= order in index.modality_slots[modality]).
ModalityEncoder = Callable[[List[ModalityItem], List[MultimodalParams]], torch.Tensor]


def encoder_dispatch(
    index: ScatterIndices,
    items: List[ModalityItem],
    encoders: Dict[str, ModalityEncoder],
    multimodal_params: List[MultimodalParams],
) -> Dict[str, torch.Tensor]:
    """Run one encoder per active modality; return per-modality output tensors."""
    outputs: Dict[str, torch.Tensor] = {}
    for modality in index.active_modalities:
        bucket_items = [items[i] for i in index.modality_slots[modality].tolist()]
        out = encoders[modality](bucket_items, multimodal_params)
        expected = index.dst_indices[modality].numel()  # == sum of this bucket's rows
        assert out.shape[0] == expected, (
            f"encoder for {modality!r} returned {out.shape[0]} rows; expected "
            f"{expected} (sum of item rows)"
        )
        outputs[modality] = out
    return outputs


def scatter_to_prompt_order(
    index: ScatterIndices,
    outputs_by_modality: Dict[str, torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
    hidden_dim: int,
) -> torch.Tensor:
    """Place per-modality outputs into the canonical prompt-ordered buffer."""
    if index.total_tokens == 0:
        return torch.empty((0, hidden_dim), dtype=dtype, device=device)
    final = torch.empty((index.total_tokens, hidden_dim), dtype=dtype, device=device)
    for modality in index.active_modalities:
        dst = index.dst_indices[modality].to(device)
        final.index_copy_(0, dst, outputs_by_modality[modality])
    return final


def encode_by_modality_and_scatter(
    multimodal_params: List[MultimodalParams],
    encoders: Dict[str, ModalityEncoder],
    extract: ItemExtractor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    hidden_dim: int,
) -> torch.Tensor:
    """Model-facing entry point: flatten -> build index -> dispatch -> scatter.

    Returns a tensor of shape `(index.total_tokens, hidden_dim)` in canonical
    layout (per-param slices in source-param order; each slice in prompt order).
    Caller wraps in `[final]` to satisfy the `get_multimodal_embeddings`
    single-tensor cache contract.

    Keeps the per-param cross-check that a param's summed item rows match its
    declared `total_embeds_in_request` (the param<->context seam; skipped when
    runtime metadata is absent, as for lightweight unit-test stubs).
    """
    items: List[ModalityItem] = [
        item
        for param_idx, param in enumerate(multimodal_params)
        for item in extract(param_idx, param)
    ]
    index = build_scatter_index(items, num_params=len(multimodal_params))

    # Vectorized cross-check: per-param row sums vs declared totals, masking params
    # whose runtime metadata is absent. Reuses index.param_lengths (no second
    # accumulation loop); the per-param attr read is the Python-object boundary.
    declared, present = [], []
    for param in multimodal_params:
        runtime = getattr(param, "multimodal_runtime", None)
        total = getattr(runtime, "total_embeds_in_request", None) if runtime else None
        declared.append(0 if total is None else int(total))
        present.append(total is not None)
    present_t = torch.tensor(present, dtype=torch.bool)
    declared_t = torch.tensor(declared, dtype=torch.int64)
    mismatch = present_t & (index.param_lengths != declared_t)
    if bool(mismatch.any()):
        bad = int(mismatch.nonzero(as_tuple=True)[0][0])
        raise ValueError(
            f"param {bad}: sum(rows)={int(index.param_lengths[bad])} != "
            f"total_embeds_in_request={int(declared_t[bad])}"
        )

    if index.total_tokens == 0:
        return torch.empty((0, hidden_dim), dtype=dtype, device=device)
    outputs_by_modality = encoder_dispatch(index, items, encoders, multimodal_params)
    return scatter_to_prompt_order(
        index, outputs_by_modality, device=device, dtype=dtype, hidden_dim=hidden_dim
    )

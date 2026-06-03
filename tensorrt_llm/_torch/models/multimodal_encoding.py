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
"""Embedding-assembly path for mixed-modality-capable models.

A single-modality request is the degenerate (identity-permutation) case.
Per-model code supplies (a) an extractor that walks each :class:`MultimodalParams`
and yields :class:`ModalityItem` instances, and (b) per-modality encoder adapters
that bridge a bucket of items to the model's existing encoder call. This module
owns partition, index-tensor build, and per-modality scatter assembly.

Every item owns exactly one prompt slot. There are no ghosts and no shared slots:
an item's `rows` is both the number of rows its encoder emits and the size of its
scatter destination range. Interleaved repeated modalities (`image -> video ->
image`, `video(+audio)` promoted to a first-class `(audio, k)` item) are handled
by the plain per-item scatter — no post-process re-placement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Tuple

import torch

from tensorrt_llm.inputs.multimodal import MultimodalParams, MultimodalPromptOrder  # noqa: F401


@dataclass(frozen=True, slots=True)
class ModalityItem:
    """One single-modality payload (one image, video, or audio) as a unit of
    work in a `MixedModalityAssembly`.

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

    Placement (bucket slot, destination range) is derived by the assembly and is
    never stored on the item.
    """

    src_param_idx: int
    modality: str
    mm_idx_per_modality: int
    prompt_pos: int
    rows: int
    payload: Mapping[str, Any]


ItemExtractor = Callable[[int, MultimodalParams], Iterable[ModalityItem]]


@dataclass
class MixedModalityAssembly:
    """Precomputed structure for assembling a multimodal embedding tensor.

    Used only by mixed-modality-capable models (Qwen3VL, Nemotron Nano); a
    single-modality request is the degenerate (identity-permutation) case.

    Built once via :meth:`from_params`; subsequently exposes precomputed
    index tensors that drive a vectorized per-modality scatter assembly
    in :func:`assemble_embeddings`. Every item contributes its `rows` to both its
    modality bucket (`_bucket_offsets`) and its source-param destination range
    (`_param_lengths` / `_dst_indices`) — there is no encoder-vs-destination
    distinction.
    """

    items: Tuple[ModalityItem, ...]
    _param_lengths: torch.Tensor  # int64[len(multimodal_params)]
    _modality_slots: Dict[str, torch.Tensor]
    _bucket_offsets: Dict[str, torch.Tensor]
    total_tokens: int
    active_modalities: List[str]
    _dst_indices: Dict[str, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def from_params(
        cls,
        multimodal_params: List[MultimodalParams],
        extract: ItemExtractor,
    ) -> "MixedModalityAssembly":
        items: List[ModalityItem] = []
        param_lengths: List[int] = [0] * len(multimodal_params)
        modality_slots: Dict[str, List[int]] = {}
        bucket_token_counts: Dict[str, List[int]] = {}
        per_param_items: List[List[int]] = [[] for _ in multimodal_params]

        for param_idx, param in enumerate(multimodal_params):
            for item in extract(param_idx, param):
                flat_idx = len(items)
                items.append(item)
                modality_slots.setdefault(item.modality, []).append(flat_idx)
                # `_bucket_offsets` indexes the per-modality encoder output, which
                # has `rows` rows per item. Each item owns its rows outright, so
                # the bucket offsets are simply the prefix sum of `rows`.
                bucket_token_counts.setdefault(item.modality, []).append(item.rows)
                param_lengths[param_idx] += item.rows
                per_param_items[param_idx].append(flat_idx)

        # Per-param uniqueness invariants. `mm_idx_per_modality` indexes into a
        # source param's own `multimodal_data[modality]`, so it is unique WITHIN
        # a param (the same `(image, 0)` legitimately recurs across batched
        # requests); `prompt_pos` is likewise a per-param prompt rank.
        for param_idx, flat_idxs in enumerate(per_param_items):
            seen_pos: set = set()
            seen_modality_group: set = set()
            for fi in flat_idxs:
                item = items[fi]
                if item.prompt_pos in seen_pos:
                    raise ValueError(f"duplicate prompt_pos={item.prompt_pos} in param {param_idx}")
                seen_pos.add(item.prompt_pos)
                key = (item.modality, item.mm_idx_per_modality)
                if key in seen_modality_group:
                    raise ValueError(
                        f"duplicate (modality, mm_idx_per_modality)={key} in param "
                        f"{param_idx}; each modality blob is owned by one item"
                    )
                seen_modality_group.add(key)

        # Cross-check: a param's summed item rows must equal its declared
        # `total_embeds_in_request` (the encoder-output row count). Skipped when
        # the runtime metadata is absent (e.g. lightweight unit-test stubs), per
        # the same contract as `_validate_primary_embedding_rows`.
        for param_idx, param in enumerate(multimodal_params):
            runtime = getattr(param, "multimodal_runtime", None)
            total = getattr(runtime, "total_embeds_in_request", None) if runtime else None
            if total is None:
                continue
            if param_lengths[param_idx] != int(total):
                raise ValueError(
                    f"param {param_idx}: sum(rows)={param_lengths[param_idx]} != "
                    f"total_embeds_in_request={int(total)}"
                )

        param_lengths_t = torch.tensor(param_lengths, dtype=torch.int64)
        if len(param_lengths) > 0:
            param_offsets_t = torch.cumsum(param_lengths_t, dim=0) - param_lengths_t
        else:
            param_offsets_t = torch.empty(0, dtype=torch.int64)

        slots_t = {m: torch.tensor(idxs, dtype=torch.int64) for m, idxs in modality_slots.items()}
        # `_bucket_offsets[m]` is the exclusive prefix sum of per-item `rows`
        # (length `len(counts) + 1`, leading 0): a zero prepended to the
        # cumulative sum of the bucket's row counts.
        offsets_t = {
            m: torch.cat(
                [
                    torch.zeros(1, dtype=torch.int64),
                    torch.cumsum(torch.tensor(counts, dtype=torch.int64), dim=0),
                ]
            )
            for m, counts in bucket_token_counts.items()
        }

        # Within-param destination offset for each item, by prompt-order rank.
        within_param_offsets: Dict[int, int] = {}
        for flat_idxs in per_param_items:
            sorted_idxs = sorted(flat_idxs, key=lambda fi: items[fi].prompt_pos)
            running = 0
            for fi in sorted_idxs:
                within_param_offsets[fi] = running
                running += items[fi].rows

        # Build `_dst_indices[m]`: per modality, expand each item's destination
        # row range `[start, start + rows)` with a single vectorized
        # repeat_interleave + arange (no per-token Python loop). Per-item Python
        # only assembles the small (start, length) lists.
        dst_indices_t: Dict[str, torch.Tensor] = {}
        param_offsets_list = param_offsets_t.tolist()
        for modality, slot_indices in modality_slots.items():
            starts: List[int] = []
            lengths: List[int] = []
            for fi in slot_indices:
                item = items[fi]
                starts.append(param_offsets_list[item.src_param_idx] + within_param_offsets[fi])
                lengths.append(item.rows)
            dst_indices_t[modality] = _expand_ranges(
                torch.tensor(starts, dtype=torch.int64),
                torch.tensor(lengths, dtype=torch.int64),
            )

        return cls(
            items=tuple(items),
            _param_lengths=param_lengths_t,
            _modality_slots=slots_t,
            _bucket_offsets=offsets_t,
            total_tokens=int(param_lengths_t.sum().item()) if len(param_lengths) > 0 else 0,
            active_modalities=list(modality_slots.keys()),
            _dst_indices=dst_indices_t,
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
# with rows in bucket order (= order in assembly._modality_slots[modality]).
ModalityEncoder = Callable[[List[ModalityItem], List[MultimodalParams]], torch.Tensor]


def assemble_embeddings(
    assembly: MixedModalityAssembly,
    encoders: Dict[str, ModalityEncoder],
    multimodal_params: List[MultimodalParams],
    *,
    device: torch.device,
    dtype: torch.dtype,
    hidden_dim: int,
) -> torch.Tensor:
    """Run one encoder call per active modality and assemble via scatter.

    Returns a tensor of shape `(assembly.total_tokens, hidden_dim)` in
    canonical layout (per-param slices in source-param order; each slice
    in MultimodalPromptOrder order). Caller wraps in `[final]` to satisfy the
    `get_multimodal_embeddings` single-tensor cache contract.

    Hot path: at most `len(assembly.active_modalities)` encoder launches +
    one `index_copy_` per modality + the destination allocation. No
    per-item Python loops on hot path.
    """
    if assembly.total_tokens == 0:
        return torch.empty((0, hidden_dim), dtype=dtype, device=device)

    bucket_outputs: Dict[str, torch.Tensor] = {}
    for modality in assembly.active_modalities:
        slot_indices = assembly._modality_slots[modality].tolist()
        bucket_items = [assembly.items[i] for i in slot_indices]
        out = encoders[modality](bucket_items, multimodal_params)
        expected_rows = sum(item.rows for item in bucket_items)
        assert out.shape[0] == expected_rows, (
            f"encoder for {modality!r} returned {out.shape[0]} rows; "
            f"assembly expected {expected_rows} (sum of item rows)"
        )
        bucket_outputs[modality] = out

    final = torch.empty((assembly.total_tokens, hidden_dim), dtype=dtype, device=device)
    for modality in assembly.active_modalities:
        dst = assembly._dst_indices[modality]
        final.index_copy_(0, dst.to(device), bucket_outputs[modality])

    assert final.shape[0] == int(assembly._param_lengths.sum().item()), (
        f"final shape {final.shape[0]} != sum(_param_lengths) "
        f"{int(assembly._param_lengths.sum().item())}"
    )
    return final

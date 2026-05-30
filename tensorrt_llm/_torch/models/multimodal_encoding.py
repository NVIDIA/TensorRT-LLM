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
"""Flat-item multimodal encoding plan.

Per-model code supplies (a) an extractor that walks each :class:`MultimodalParams`
and yields :class:`MultimodalItem` instances, and (b) per-modality encoder adapters
that bridge a bucket of items to the model's existing encoder call. This module
owns partition, index-tensor build, and per-modality scatter assembly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import torch

from tensorrt_llm.inputs.multimodal import MMItemOrder, MultimodalParams  # noqa: F401


@dataclass(frozen=True, slots=True)
class MultimodalItem:
    """One modality item in a flat encoding plan.

    Carries the join keys (``src_param_idx``, ``item_idx_in_param``) that
    let the plan reconcile modality-grouping (encoder batching) with
    source-param grouping (cache contract + MMItemOrder reassembly).

    Ghost items (e.g. audio extracted from a video payload) use
    ``item_idx_in_param == -1`` to indicate they have no MMItemOrder slot;
    their encoded rows are consumed by a model-specific post-process step
    rather than scattered into the final output.
    """

    src_param_idx: int
    item_idx_in_param: int
    modality: str
    token_count: int
    payload: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)


ItemExtractor = Callable[[int, MultimodalParams], Iterable[MultimodalItem]]


@dataclass
class EncodingPlan:
    """Flat-item plan for batched multimodal encoding.

    Built once via :meth:`from_params`; subsequently exposes precomputed
    index tensors that drive a vectorized per-modality scatter assembly
    in :func:`encode_with_plan` (added in Task 5).

    Ghost items (``item_idx_in_param == -1``) participate in the encoder
    bucket call but do NOT contribute to ``_param_lengths`` or (later)
    ``_dst_indices`` — their rows are consumed by the model-specific
    post-process step instead.
    """

    items: Tuple[MultimodalItem, ...]
    n_params: int
    _param_lengths: torch.Tensor  # int64[n_params]
    _param_offsets: torch.Tensor  # int64[n_params]
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
    ) -> "EncodingPlan":
        items: List[MultimodalItem] = []
        param_lengths: List[int] = [0] * len(multimodal_params)
        modality_slots: Dict[str, List[int]] = {}
        bucket_token_counts: Dict[str, List[int]] = {}
        per_param_non_ghost: List[List[int]] = [[] for _ in multimodal_params]

        for param_idx, param in enumerate(multimodal_params):
            for item in extract(param_idx, param):
                flat_idx = len(items)
                items.append(item)
                modality_slots.setdefault(item.modality, []).append(flat_idx)
                bucket_token_counts.setdefault(item.modality, []).append(item.token_count)
                if item.item_idx_in_param != -1:
                    param_lengths[param_idx] += item.token_count
                    per_param_non_ghost[param_idx].append(flat_idx)

        # Validate non-ghost item_idx_in_param uniqueness per param
        for param_idx, flat_idxs in enumerate(per_param_non_ghost):
            seen = set()
            for fi in flat_idxs:
                pos = items[fi].item_idx_in_param
                if pos in seen:
                    raise ValueError(f"duplicate item_idx_in_param={pos} in param {param_idx}")
                seen.add(pos)

        param_lengths_t = torch.tensor(param_lengths, dtype=torch.int64)
        if len(param_lengths) > 0:
            param_offsets_t = torch.cumsum(param_lengths_t, dim=0) - param_lengths_t
        else:
            param_offsets_t = torch.empty(0, dtype=torch.int64)

        slots_t = {m: torch.tensor(idxs, dtype=torch.int64) for m, idxs in modality_slots.items()}
        offsets_t = {
            m: torch.tensor([0] + list(_running_sum(counts)), dtype=torch.int64)
            for m, counts in bucket_token_counts.items()
        }

        # Compute within-param offsets by MMItemOrder rank for non-ghost items
        within_param_offsets: Dict[int, int] = {}
        for flat_idxs in per_param_non_ghost:
            sorted_idxs = sorted(flat_idxs, key=lambda fi: items[fi].item_idx_in_param)
            running = 0
            for fi in sorted_idxs:
                within_param_offsets[fi] = running
                running += items[fi].token_count

        # Build _dst_indices[m]: walk bucket order, emit row ranges for non-ghost items
        dst_indices_t: Dict[str, torch.Tensor] = {}
        param_offsets_list = param_offsets_t.tolist()
        for modality, slot_indices in modality_slots.items():
            rows: List[int] = []
            for fi in slot_indices:
                item = items[fi]
                if item.item_idx_in_param == -1:
                    continue
                start = param_offsets_list[item.src_param_idx] + within_param_offsets[fi]
                rows.extend(range(start, start + item.token_count))
            dst_indices_t[modality] = torch.tensor(rows, dtype=torch.int64)

        return cls(
            items=tuple(items),
            n_params=len(multimodal_params),
            _param_lengths=param_lengths_t,
            _param_offsets=param_offsets_t,
            _modality_slots=slots_t,
            _bucket_offsets=offsets_t,
            total_tokens=int(param_lengths_t.sum().item()) if len(param_lengths) > 0 else 0,
            active_modalities=list(modality_slots.keys()),
            _dst_indices=dst_indices_t,
        )


def _running_sum(xs: List[int]) -> Iterable[int]:
    s = 0
    for x in xs:
        s += x
        yield s


# Adapter contract: given a bucket's items and the source-param list,
# return one tensor of shape (sum(item.token_count for item in items), hidden_dim)
# with rows in bucket order (= order in plan._modality_slots[modality]).
ModalityEncoder = Callable[[List[MultimodalItem], List[MultimodalParams]], torch.Tensor]

# Post-process: runs after encode, before scatter. May mutate
# ``bucket_outputs`` in place (e.g. Nano video-audio interleave).
PostProcess = Callable[[Dict[str, torch.Tensor], "EncodingPlan", List[MultimodalParams]], None]


def encode_with_plan(
    plan: EncodingPlan,
    encoders: Dict[str, ModalityEncoder],
    multimodal_params: List[MultimodalParams],
    *,
    device: torch.device,
    dtype: torch.dtype,
    hidden_dim: int,
    post_process: Optional[PostProcess] = None,
) -> torch.Tensor:
    """Run one encoder call per active modality and assemble via scatter.

    Returns a tensor of shape `(plan.total_tokens, hidden_dim)` in
    canonical layout (per-param slices in source-param order; each slice
    in MMItemOrder order). Caller wraps in `[final]` to satisfy the
    `get_multimodal_embeddings` single-tensor cache contract.

    Hot path: at most `len(plan.active_modalities)` encoder launches +
    one `index_copy_` per modality + the destination allocation. No
    per-item Python loops on hot path.
    """
    if plan.total_tokens == 0:
        return torch.empty((0, hidden_dim), dtype=dtype, device=device)

    bucket_outputs: Dict[str, torch.Tensor] = {}
    for modality in plan.active_modalities:
        slot_indices = plan._modality_slots[modality].tolist()
        bucket_items = [plan.items[i] for i in slot_indices]
        out = encoders[modality](bucket_items, multimodal_params)
        expected_rows = int(plan._bucket_offsets[modality][-1].item())
        assert out.shape[0] == expected_rows, (
            f"encoder for {modality!r} returned {out.shape[0]} rows; "
            f"plan expected {expected_rows} (sum of item token_counts)"
        )
        bucket_outputs[modality] = out

    if post_process is not None:
        post_process(bucket_outputs, plan, multimodal_params)

    final = torch.empty((plan.total_tokens, hidden_dim), dtype=dtype, device=device)
    for modality in plan.active_modalities:
        dst = plan._dst_indices[modality]
        if dst.numel() == 0:
            continue  # all-ghost bucket; rows consumed by post_process
        bucket = bucket_outputs[modality]
        if dst.numel() != bucket.shape[0]:
            # Some items in this bucket are ghosts. Convention: non-ghost
            # items come first in bucket order (extractor's contract);
            # scatter only the leading non-ghost portion.
            bucket = bucket[: dst.numel()]
        final.index_copy_(0, dst.to(device), bucket)

    assert final.shape[0] == int(plan._param_lengths.sum().item()), (
        f"final shape {final.shape[0]} != sum(_param_lengths) "
        f"{int(plan._param_lengths.sum().item())}"
    )
    return final

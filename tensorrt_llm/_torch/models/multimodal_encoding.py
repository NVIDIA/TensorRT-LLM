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
from typing import Any, Callable, Dict, Iterable, List, Mapping, Tuple

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

        for param_idx, param in enumerate(multimodal_params):
            for item in extract(param_idx, param):
                flat_idx = len(items)
                items.append(item)
                modality_slots.setdefault(item.modality, []).append(flat_idx)
                bucket_token_counts.setdefault(item.modality, []).append(item.token_count)
                if item.item_idx_in_param != -1:
                    param_lengths[param_idx] += item.token_count

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

        return cls(
            items=tuple(items),
            n_params=len(multimodal_params),
            _param_lengths=param_lengths_t,
            _param_offsets=param_offsets_t,
            _modality_slots=slots_t,
            _bucket_offsets=offsets_t,
            total_tokens=int(param_lengths_t.sum().item()) if len(param_lengths) > 0 else 0,
            active_modalities=list(modality_slots.keys()),
        )


def _running_sum(xs: List[int]) -> Iterable[int]:
    s = 0
    for x in xs:
        s += x
        yield s

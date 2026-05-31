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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import torch

from tensorrt_llm.inputs.multimodal import MultimodalParams, MultimodalPromptOrder  # noqa: F401


@dataclass(frozen=True, slots=True)
class ModalityItem:
    """One single-modality payload (one image, video, or audio) as a unit of
    work in a `MixedModalityAssembly`, tagged with its source request
    (`src_param_idx`) and prompt-order position (`item_idx_in_param`).

    Carries the join keys (``src_param_idx``, ``item_idx_in_param``) that
    let the assembly reconcile modality-grouping (encoder batching) with
    source-param grouping (cache contract + MultimodalPromptOrder reassembly).

    Ghost items (`item_idx_in_param == -1`) exist in encoder space but have no
    independent slot in prompt-order space. A ghost is a bookkeeping handle: it
    says "encode me in my modality's bucket, but do NOT scatter my rows to my own
    prompt-order slot - a model-specific post-process will merge them into a paired
    host item's destination range instead." The ghost's embedding rows DO end up in
    the final tensor; they are just placed by interleaving into the host item's
    range rather than by a direct scatter to a slot of their own. The sole current
    instance is Nano's audio-extracted-from-video: the audio rides the audio encoder
    bucket, and `_nano_post_encode` interleaves its rows into the paired video
    item's destination range.

    ``token_count`` is the post-process row count contributed to the
    final scatter destination. ``encoder_token_count`` is the row count
    the per-modality encoder is expected to emit BEFORE any
    post-process expansion (defaults to ``token_count``). The two values
    differ for Nano video items whose post-process step interleaves
    audio rows from the paired ghost audio item — the video encoder
    only emits vision rows, but the final scatter destination for that
    item covers vision + audio.
    """

    src_param_idx: int
    item_idx_in_param: int
    modality: str
    token_count: int
    payload: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    encoder_token_count: Optional[int] = None

    @property
    def encoder_rows(self) -> int:
        return self.token_count if self.encoder_token_count is None else self.encoder_token_count


ItemExtractor = Callable[[int, MultimodalParams], Iterable[ModalityItem]]


@dataclass
class MixedModalityAssembly:
    """Precomputed structure for assembling a multimodal embedding tensor.

    Used only by mixed-modality-capable models (Qwen3VL, Nemotron Nano); a
    single-modality request is the degenerate (identity-permutation) case.

    Built once via :meth:`from_params`; subsequently exposes precomputed
    index tensors that drive a vectorized per-modality scatter assembly
    in :func:`assemble_embeddings`.

    Ghost items (``item_idx_in_param == -1``) participate in the encoder
    bucket call but do NOT contribute to ``_param_lengths`` or (later)
    ``_dst_indices`` — their rows are consumed by the model-specific
    post-process step instead.
    """

    items: Tuple[ModalityItem, ...]
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
    ) -> "MixedModalityAssembly":
        items: List[ModalityItem] = []
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

        # Enforce the per-bucket ghost-ordering invariant by construction:
        # stable-partition each modality bucket so non-ghost items
        # (`item_idx_in_param != -1`) lead and ghost items (`== -1`) form a
        # single TRAILING contiguous block. `_bucket_offsets` and `_dst_indices`
        # are derived from `modality_slots` below, so partitioning the slots here
        # makes "leading bucket rows == non-ghost" hold structurally — both the
        # scatter slice (`bucket[:dst.numel()]`) and the Nano post-process
        # truncation (`bucket[:n_non_ghost_rows]`) become correct without
        # depending on the extractor's yield order. The ghost-block start is
        # implicitly `dst.numel()`, so no extra index tensor is needed.
        for modality, slot_idxs in modality_slots.items():
            nonghost = [fi for fi in slot_idxs if items[fi].item_idx_in_param != -1]
            ghost = [fi for fi in slot_idxs if items[fi].item_idx_in_param == -1]
            partitioned = nonghost + ghost  # stable within each group
            modality_slots[modality] = partitioned
            bucket_token_counts[modality] = [items[fi].token_count for fi in partitioned]
            # Fail-loud guard: catch any future code path that re-introduces a
            # ghost-before-non-ghost ordering instead of silently mis-scattering.
            seen_ghost = False
            for fi in partitioned:
                is_ghost = items[fi].item_idx_in_param == -1
                if seen_ghost and not is_ghost:
                    raise ValueError(
                        f"bucket {modality!r}: non-ghost item follows a ghost; "
                        "non-ghost items must lead each modality bucket"
                    )
                seen_ghost = seen_ghost or is_ghost

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

        # Compute within-param offsets by MultimodalPromptOrder rank for non-ghost items
        within_param_offsets: Dict[int, int] = {}
        for flat_idxs in per_param_non_ghost:
            sorted_idxs = sorted(flat_idxs, key=lambda fi: items[fi].item_idx_in_param)
            running = 0
            for fi in sorted_idxs:
                within_param_offsets[fi] = running
                running += items[fi].token_count

        # Build _dst_indices[m]: per modality, expand each non-ghost item's
        # destination row range [start, start+token_count) with a single
        # vectorized repeat_interleave + arange (no per-token Python loop).
        # Per-item Python only assembles the small (start, length) lists.
        dst_indices_t: Dict[str, torch.Tensor] = {}
        param_offsets_list = param_offsets_t.tolist()
        for modality, slot_indices in modality_slots.items():
            starts: List[int] = []
            lengths: List[int] = []
            for fi in slot_indices:
                item = items[fi]
                if item.item_idx_in_param == -1:
                    continue
                starts.append(param_offsets_list[item.src_param_idx] + within_param_offsets[fi])
                lengths.append(item.token_count)
            dst_indices_t[modality] = _expand_ranges(
                torch.tensor(starts, dtype=torch.int64),
                torch.tensor(lengths, dtype=torch.int64),
            )

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


def _expand_ranges(starts: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Concatenate per-segment ranges ``[s, s + l)`` for paired (starts, lengths).

    Vectorized equivalent of
    ``torch.cat([torch.arange(s, s + l) for s, l in zip(starts, lengths)])``,
    built with a single `repeat_interleave` + `arange` (no per-token Python
    loop). Both inputs are int64 1-D tensors of equal length; returns an empty
    int64 tensor when there are no rows (e.g. an all-ghost bucket).
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
# return one tensor of shape (sum(item.token_count for item in items), hidden_dim)
# with rows in bucket order (= order in assembly._modality_slots[modality]).
ModalityEncoder = Callable[[List[ModalityItem], List[MultimodalParams]], torch.Tensor]

# Post-process: runs after encode, before scatter. May mutate
# ``bucket_outputs`` in place (e.g. Nano video-audio interleave).
PostProcess = Callable[
    [Dict[str, torch.Tensor], "MixedModalityAssembly", List[MultimodalParams]], None
]


def assemble_embeddings(
    assembly: MixedModalityAssembly,
    encoders: Dict[str, ModalityEncoder],
    multimodal_params: List[MultimodalParams],
    *,
    device: torch.device,
    dtype: torch.dtype,
    hidden_dim: int,
    post_process: Optional[PostProcess] = None,
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
        # Use ``encoder_token_count`` (defaults to ``token_count``) for the
        # encoder-output assertion. Items whose ``token_count`` reflects a
        # post-process expansion (e.g. Nano video + interleaved audio) must
        # set ``encoder_token_count`` to the encoder-only row count.
        expected_rows = sum(item.encoder_rows for item in bucket_items)
        assert out.shape[0] == expected_rows, (
            f"encoder for {modality!r} returned {out.shape[0]} rows; "
            f"assembly expected {expected_rows} (sum of item encoder_rows)"
        )
        bucket_outputs[modality] = out

    if post_process is not None:
        post_process(bucket_outputs, assembly, multimodal_params)

    final = torch.empty((assembly.total_tokens, hidden_dim), dtype=dtype, device=device)
    for modality in assembly.active_modalities:
        dst = assembly._dst_indices[modality]
        if dst.numel() == 0:
            continue  # all-ghost bucket; rows consumed by post_process
        bucket = bucket_outputs[modality]
        if dst.numel() != bucket.shape[0]:
            # Some items in this bucket are ghosts. Convention: non-ghost
            # items come first in bucket order (extractor's contract);
            # scatter only the leading non-ghost portion.
            bucket = bucket[: dst.numel()]
        final.index_copy_(0, dst.to(device), bucket)

    assert final.shape[0] == int(assembly._param_lengths.sum().item()), (
        f"final shape {final.shape[0]} != sum(_param_lengths) "
        f"{int(assembly._param_lengths.sum().item())}"
    )
    return final

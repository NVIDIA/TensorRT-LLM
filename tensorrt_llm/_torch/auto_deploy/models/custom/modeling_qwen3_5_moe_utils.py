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

from typing import List, Optional, Sequence, Tuple

import torch

MultimodalItemRun = Tuple[int, int, List[int]]
MultimodalItemRuns = List[List[MultimodalItemRun]]


def _normalize_special_offsets(
    req_special_offsets: Optional[Sequence[int]],
    total_mm_length: int,
) -> List[int]:
    if req_special_offsets is None:
        return []
    special_offsets = sorted(set(int(offset) for offset in req_special_offsets))
    if special_offsets and (special_offsets[0] < 0 or special_offsets[-1] >= total_mm_length):
        raise ValueError(
            "Qwen3.5 special multimodal offsets must be within the request "
            f"multimodal token range [0, {total_mm_length})"
        )
    return special_offsets


def _collect_run_special_offsets(
    special_offsets: Sequence[int],
    special_idx: int,
    run_flat_start: int,
    run_flat_end: int,
) -> Tuple[List[int], int]:
    non_embed_offsets: List[int] = []
    while special_idx < len(special_offsets) and special_offsets[special_idx] < run_flat_end:
        non_embed_offsets.append(int(special_offsets[special_idx]) - run_flat_start)
        special_idx += 1
    return non_embed_offsets, special_idx


def build_request_item_runs(
    req_mm_positions: Sequence[int],
    req_mm_lengths: Sequence[int],
    req_mm_item_run_cu_seqlen: Optional[Sequence[int]] = None,
    req_mm_run_positions: Optional[Sequence[int]] = None,
    req_mm_run_lengths: Optional[Sequence[int]] = None,
    req_special_offsets: Optional[Sequence[int]] = None,
) -> MultimodalItemRuns:
    total_mm_length = sum(int(mm_len) for mm_len in req_mm_lengths)
    special_offsets = _normalize_special_offsets(req_special_offsets, total_mm_length)
    if (
        req_mm_item_run_cu_seqlen is None
        and req_mm_run_positions is None
        and req_mm_run_lengths is None
    ):
        item_runs: MultimodalItemRuns = []
        special_idx = 0
        item_flat_start = 0
        for mm_start, mm_len in zip(req_mm_positions, req_mm_lengths):
            item_len = int(mm_len)
            non_embed_offsets, special_idx = _collect_run_special_offsets(
                special_offsets,
                special_idx,
                item_flat_start,
                item_flat_start + item_len,
            )
            item_runs.append([(int(mm_start), item_len, non_embed_offsets)])
            item_flat_start += item_len
        return item_runs
    if (
        req_mm_item_run_cu_seqlen is None
        or req_mm_run_positions is None
        or req_mm_run_lengths is None
    ):
        raise ValueError("Incomplete multimodal item-run metadata for Qwen3.5 chunk")
    if len(req_mm_item_run_cu_seqlen) != len(req_mm_positions) + 1:
        raise ValueError(
            "mm_item_run_cu_seqlen must have one more entry than request items: "
            f"cu={len(req_mm_item_run_cu_seqlen)}, items={len(req_mm_positions)}"
        )
    if len(req_mm_run_positions) != len(req_mm_run_lengths):
        raise ValueError(
            "Mismatch between mm_run_token_positions and mm_run_token_lengths: "
            f"positions={len(req_mm_run_positions)}, lengths={len(req_mm_run_lengths)}"
        )
    if int(req_mm_item_run_cu_seqlen[-1]) != len(req_mm_run_positions):
        raise ValueError(
            "mm_item_run_cu_seqlen final value must match number of run entries: "
            f"final={int(req_mm_item_run_cu_seqlen[-1])}, runs={len(req_mm_run_positions)}"
        )

    item_runs_by_item: MultimodalItemRuns = []
    special_idx = 0
    item_flat_start = 0
    for item_idx, mm_len in enumerate(req_mm_lengths):
        run_start_idx = int(req_mm_item_run_cu_seqlen[item_idx])
        run_end_idx = int(req_mm_item_run_cu_seqlen[item_idx + 1])
        if run_start_idx == run_end_idx:
            raise ValueError(f"Qwen3.5 multimodal item {item_idx} has no run entries")
        item_length = 0
        previous_run_end: Optional[int] = None
        item_runs: List[MultimodalItemRun] = []
        for run_idx in range(run_start_idx, run_end_idx):
            run_start = int(req_mm_run_positions[run_idx])
            run_length = int(req_mm_run_lengths[run_idx])
            if run_length <= 0:
                raise ValueError(
                    f"Qwen3.5 multimodal item {item_idx} run {run_idx} must have positive length"
                )
            if previous_run_end is not None and run_start < previous_run_end:
                raise ValueError(
                    f"Qwen3.5 multimodal item {item_idx} run {run_idx} overlaps "
                    "or is not sorted after the previous run"
                )
            run_flat_start = item_flat_start + item_length
            run_flat_end = run_flat_start + run_length
            non_embed_offsets, special_idx = _collect_run_special_offsets(
                special_offsets,
                special_idx,
                run_flat_start,
                run_flat_end,
            )
            item_runs.append((run_start, run_length, non_embed_offsets))
            item_length += run_length
            previous_run_end = run_start + run_length
        if item_length != int(mm_len):
            raise ValueError(
                "Qwen3.5 multimodal item run lengths do not match item length: "
                f"item={item_idx}, runs={item_length}, item_length={int(mm_len)}"
            )
        item_runs_by_item.append(item_runs)
        item_flat_start += item_length
    return item_runs_by_item


def select_request_chunk_multimodal_embeds(
    req_input_pos: int,
    req_seq_len: int,
    req_mm_item_types: Sequence[int],
    req_mm_positions: Sequence[int],
    req_mm_lengths: Sequence[int],
    req_special_offsets: Sequence[int],
    image_embeds_list: Optional[Sequence[torch.Tensor]],
    video_embeds_list: Optional[Sequence[torch.Tensor]],
    hidden_size: int,
    req_mm_item_run_cu_seqlen: Optional[Sequence[int]] = None,
    req_mm_run_positions: Optional[Sequence[int]] = None,
    req_mm_run_lengths: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    chunk_end = req_input_pos + req_seq_len
    img_idx = 0
    vid_idx = 0
    chunks: List[torch.Tensor] = []
    item_runs_by_item = build_request_item_runs(
        req_mm_positions,
        req_mm_lengths,
        req_mm_item_run_cu_seqlen,
        req_mm_run_positions,
        req_mm_run_lengths,
        req_special_offsets,
    )

    for item_idx, (item_type, mm_len) in enumerate(zip(req_mm_item_types, req_mm_lengths)):
        item_runs = item_runs_by_item[item_idx]

        if item_type == 0:
            if image_embeds_list is None:
                raise ValueError("Missing image embeddings for image multimodal item")
            item_embeds = image_embeds_list[img_idx]
            img_idx += 1
        elif item_type == 1:
            if video_embeds_list is None:
                raise ValueError("Missing video embeddings for video multimodal item")
            item_embeds = video_embeds_list[vid_idx]
            vid_idx += 1
        else:
            raise ValueError(f"Unsupported multimodal item type: {item_type}")

        expected_features = sum(
            run_length - len(non_embed_offsets) for _, run_length, non_embed_offsets in item_runs
        )
        if expected_features != item_embeds.shape[0]:
            raise ValueError(
                "Multimodal embedding length mismatch for Qwen3.5 item: "
                f"type={item_type}, expected={expected_features}, actual={item_embeds.shape[0]}, "
                f"mm_len={int(mm_len)}, item_runs={item_runs}, "
                f"special_offsets={sorted(set(int(x) for x in req_special_offsets))}"
            )

        selected_indices: List[int] = []
        feature_idx = 0
        for run_start, run_length, non_embed_offsets in item_runs:
            non_embed_offset_set = set(non_embed_offsets)
            overlap_start = max(req_input_pos, run_start)
            overlap_end = min(chunk_end, run_start + run_length)
            overlap_rel_start = overlap_start - run_start
            overlap_rel_end = overlap_end - run_start
            for rel in range(run_length):
                if rel in non_embed_offset_set:
                    continue
                if overlap_rel_start <= rel < overlap_rel_end:
                    selected_indices.append(feature_idx)
                feature_idx += 1
        if selected_indices:
            chunks.append(item_embeds[selected_indices])

    if chunks:
        return torch.cat(chunks, dim=0)

    device = None
    dtype = None
    if image_embeds_list:
        device = image_embeds_list[0].device
        dtype = image_embeds_list[0].dtype
    elif video_embeds_list:
        device = video_embeds_list[0].device
        dtype = video_embeds_list[0].dtype
    if device is None or dtype is None:
        raise ValueError("Cannot build empty multimodal chunk without image or video embeddings")
    return torch.empty(0, hidden_size, device=device, dtype=dtype)


def compute_request_chunk_mrope_positions(
    req_input_pos: int,
    req_seq_len: int,
    req_mm_item_types: Sequence[int],
    req_mm_positions: Sequence[int],
    req_mm_lengths: Sequence[int],
    req_special_offsets: Sequence[int],
    req_mm_item_run_cu_seqlen: Optional[Sequence[int]],
    req_mm_run_positions: Optional[Sequence[int]],
    req_mm_run_lengths: Optional[Sequence[int]],
    image_grid_thw: Optional[torch.Tensor],
    video_grid_thw: Optional[torch.Tensor],
    spatial_merge_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Compute chunk-local 3D mRoPE positions for one request in absolute coordinates."""
    chunk_end = req_input_pos + req_seq_len
    out = torch.empty((3, 1, req_seq_len), dtype=dtype, device=device)
    abs_cursor = 0
    comp_cursor = 0
    img_idx = 0
    vid_idx = 0
    item_runs_by_item = build_request_item_runs(
        req_mm_positions,
        req_mm_lengths,
        req_mm_item_run_cu_seqlen,
        req_mm_run_positions,
        req_mm_run_lengths,
        req_special_offsets,
    )

    def fill_text(abs_start: int, abs_end: int, comp_start: int) -> None:
        ov_start = max(req_input_pos, abs_start)
        ov_end = min(chunk_end, abs_end)
        if ov_start >= ov_end:
            return
        start_pos = comp_start + (ov_start - abs_start)
        text_pos = torch.arange(
            start_pos,
            start_pos + (ov_end - ov_start),
            device=device,
            dtype=dtype,
        )
        out[:, 0, ov_start - req_input_pos : ov_end - req_input_pos] = text_pos.unsqueeze(0).expand(
            3, -1
        )

    def build_vision_positions(grid: torch.Tensor, comp_start: int) -> Tuple[torch.Tensor, int]:
        t, h, w = [int(v) for v in grid.tolist()]
        llm_grid_t = int(t)
        llm_grid_h = int(h) // spatial_merge_size
        llm_grid_w = int(w) // spatial_merge_size

        t_index = (
            torch.arange(llm_grid_t, device=device, dtype=dtype)
            .view(-1, 1)
            .expand(-1, llm_grid_h * llm_grid_w)
            .flatten()
        )
        h_index = (
            torch.arange(llm_grid_h, device=device, dtype=dtype)
            .view(1, -1, 1)
            .expand(llm_grid_t, -1, llm_grid_w)
            .flatten()
        )
        w_index = (
            torch.arange(llm_grid_w, device=device, dtype=dtype)
            .view(1, 1, -1)
            .expand(llm_grid_t, llm_grid_h, -1)
            .flatten()
        )
        positions = torch.stack([t_index, h_index, w_index]) + comp_start
        return positions, comp_start + max(llm_grid_t, llm_grid_h, llm_grid_w)

    for item_idx, item_type in enumerate(req_mm_item_types):
        item_runs = item_runs_by_item[item_idx]
        vision_abs_positions: List[int] = []
        for run_start, run_length, non_embed_offsets in item_runs:
            non_embed_offset_set = set(non_embed_offsets)
            vision_abs_positions.extend(
                run_start + rel for rel in range(run_length) if rel not in non_embed_offset_set
            )

        if not vision_abs_positions:
            item_abs_end = max(run_start + run_length for run_start, run_length, _ in item_runs)
            fill_text(abs_cursor, item_abs_end, comp_cursor)
            comp_cursor += item_abs_end - abs_cursor
            abs_cursor = item_abs_end
            continue

        first_vision_abs_start = vision_abs_positions[0]
        fill_text(abs_cursor, first_vision_abs_start, comp_cursor)
        comp_cursor += first_vision_abs_start - abs_cursor

        if item_type == 0:
            if image_grid_thw is None:
                raise ValueError("Missing image_grid_thw for image multimodal item")
            grid = image_grid_thw[img_idx]
            img_idx += 1
        elif item_type == 1:
            if video_grid_thw is None:
                raise ValueError("Missing video_grid_thw for video multimodal item")
            grid = video_grid_thw[vid_idx]
            vid_idx += 1
        else:
            raise ValueError(f"Unsupported multimodal item type: {item_type}")

        vision_positions, next_comp_cursor = build_vision_positions(grid, comp_cursor)
        if vision_positions.shape[1] != len(vision_abs_positions):
            raise ValueError(
                "Qwen3.5 vision grid length does not match exact item runs: "
                f"grid_tokens={vision_positions.shape[1]}, run_tokens={len(vision_abs_positions)}"
            )

        text_cursor = first_vision_abs_start
        text_comp_cursor = next_comp_cursor
        for vision_idx, vision_abs_pos in enumerate(vision_abs_positions):
            fill_text(text_cursor, vision_abs_pos, text_comp_cursor)
            text_comp_cursor += vision_abs_pos - text_cursor
            if req_input_pos <= vision_abs_pos < chunk_end:
                out[:, 0, vision_abs_pos - req_input_pos] = vision_positions[:, vision_idx]
            text_cursor = vision_abs_pos + 1

        item_abs_end = max(run_start + run_length for run_start, run_length, _ in item_runs)
        fill_text(text_cursor, item_abs_end, text_comp_cursor)
        comp_cursor = text_comp_cursor + item_abs_end - text_cursor
        abs_cursor = item_abs_end

    fill_text(abs_cursor, chunk_end, comp_cursor)
    return out

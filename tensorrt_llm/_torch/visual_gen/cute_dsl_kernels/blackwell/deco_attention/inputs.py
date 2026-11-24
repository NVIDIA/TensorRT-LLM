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
"""Inputs of the DeCo preparation besides q, k and v, the same for every kernel family: sample_rows and hadamard. What
they need is built on first use and kept: the default rows per sequence length, rank and device, the priorities of the
tagged draw and the rotation per device. Building them is not capturable, so a first use under CUDA graph capture
raises: a capture follows one eager call of the shape. The last tagged draw is kept as well, for the calls of a forward
that pass the same tensors."""

import functools

import torch

from .constants import DEFAULT_RANK, HEAD_DIM, SAMPLE_FRAC

# Key of a position that the tagged draw does not take, above every priority.
_NOT_DRAWN = 1 << 30
# Smallest number of positions a priority table is built for.
_MIN_PRIORITIES = 1 << 16
# The last tagged draw outside a capture: its key (token_tags, tokens, their version counters, s, device, rank), its
# rows and the event of their completion on the stream that drew them.
_last_tagged_draw = None


def sample_count(s: int, rank: int = DEFAULT_RANK) -> int:
    """Return the number of sampled tokens for a sequence of s tokens and a basis of `rank` directions."""
    return max(s // SAMPLE_FRAC, 2 * rank)


def _indexed(device: torch.device | str) -> torch.device:
    """Return the device with its index ("cuda": the current CUDA device), so that cached tensors stay per device."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _not_under_capture(device: torch.device, what: str) -> None:
    """Raise when `what` is first needed while the current CUDA stream is capturing a graph."""
    if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"DeCo-Attention {what} first needed during CUDA graph capture; "
            "run the call eagerly once before capturing"
        )


def _version(tensor: torch.Tensor | None) -> int | None:
    """Return the version counter of a tensor, which a write in place advances; None for no tensor and for an
    inference tensor, which has none."""
    return None if tensor is None or tensor.is_inference() else tensor._version


def sample_rows(
    s: int,
    device: torch.device | str = "cuda",
    token_tags: torch.Tensor | None = None,
    rank: int = DEFAULT_RANK,
    tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the draw of the tokens that estimate the compensation basis of `rank` directions: int32 indices, [m] with
    m = sample_count(s, rank), or [B, m] with `tokens`.

    * Default draw (no token_tags): a permutation from a CUDA generator seeded 0, built once per length, rank and
      device: the same tokens every call.
    * token_tags (integer tensor [s]) marks the modality of every token of a packed multimodal sequence: 0 video,
      positive text or audio, negative padding (never drawn). The tagged draw takes half its rows from the text and
      audio tokens, so both groups weigh equally in the basis where the video tokens would otherwise leave the others
      almost unsampled.
        - A group with fewer tokens than its share gives them all and the other group fills the draw.
        - A sequence with fewer than m tokens repeats its draw; one without tokens gets row 0.
        - A group gives its tokens in a fixed pseudo-random order of the positions: the same tags, the same rows.
    * tokens (bool tensor [B, s], with token_tags): True where a batch element holds a token, whatever the tag of the
      position. Every batch element then gets the draw of its own tokens.
    * The tagged draw runs on the device and reads nothing back to the host: a CUDA graph can capture it. Outside a
      capture the last draw is kept and returned while the caller passes the same token_tags and tokens tensors,
      compared by identity and, where a tensor has one, by its version counter.
    """
    device = _indexed(device)
    if token_tags is None:
        rows, drawn = _sample_rows(s, device, rank)
        # A capture cannot wait on an outside event; the eager call before it did.
        if not torch.cuda.is_current_stream_capturing():
            torch.cuda.current_stream(device).wait_event(drawn)
        return rows
    global _last_tagged_draw
    capturing = device.type == "cuda" and torch.cuda.is_current_stream_capturing()
    key = (token_tags, tokens, _version(token_tags), _version(tokens), s, device, rank)
    if not capturing and _last_tagged_draw is not None:
        kept, rows, drawn = _last_tagged_draw
        if kept[0] is token_tags and kept[1] is tokens and kept[2:] == key[2:]:
            if drawn is not None:
                # Another stream may have drawn the rows and still be at it.
                torch.cuda.current_stream(device).wait_event(drawn)
            return rows
    tags = token_tags.to(device).reshape(-1)
    if tags.numel() != s:
        raise ValueError(f"token_tags must hold {s} tags, got {tuple(token_tags.shape)}")
    if tokens is not None:
        if tokens.dim() != 2 or tokens.shape[1] != s:
            raise ValueError(f"tokens must have shape [B, {s}], got {tuple(tokens.shape)}")
        tags = torch.where(tokens.to(device), tags, -1)
    m = sample_count(s, rank)
    k = min(m, s)
    priority = _priorities(device, max(_MIN_PRIORITIES, 1 << (s - 1).bit_length()))[:s]
    # Video tokens and the others, one group per leading index.
    groups = torch.stack([tags == 0, tags > 0])
    # The first k tokens of each group in priority order; behind the tokens of a shorter group come positions that
    # no slot below reads.
    video_rows, other_rows = (
        torch.where(groups, priority, _NOT_DRAWN).topk(k, largest=False).indices
    )
    num_video, num_other = groups.sum(-1, keepdim=True)
    take_other = num_other.clamp(max=m // 2)
    take_video = torch.minimum(num_video, m - take_other)
    take_other = torch.minimum(num_other, m - take_video)
    num_drawn = take_video + take_other
    # Slot j holds draw j modulo the number drawn, the video tokens first: a short draw repeats.
    slot = torch.arange(m, device=device) % num_drawn.clamp(min=1)
    index = torch.where(slot < take_video, slot, slot - take_video + k)
    rows = torch.cat([video_rows, other_rows], dim=-1).gather(-1, index)
    rows = rows.masked_fill(num_drawn == 0, 0).to(torch.int32)
    if not capturing:
        drawn = torch.cuda.current_stream(device).record_event() if device.type == "cuda" else None
        _last_tagged_draw = (key, rows, drawn)
    return rows


@functools.cache
def _priorities(device: torch.device, capacity: int) -> torch.Tensor:
    """Return the priorities of the positions below `capacity` in the tagged draw: int32, distinct, below _NOT_DRAWN.
    Position i has the 30-bit mix of i whatever the capacity, so a sequence draws the same rows in every process."""
    _not_under_capture(device, "priorities of a tagged sequence")
    # Every step is a bijection of the 30-bit integers.
    mask = _NOT_DRAWN - 1
    x = (torch.arange(capacity, dtype=torch.int64, device=device) + 0x9E3779B9) & mask
    x = ((x ^ (x >> 16)) * 0x85EBCA6B) & mask
    x = ((x ^ (x >> 13)) * 0xC2B2AE35) & mask
    return (x ^ (x >> 16)).to(torch.int32)


@functools.cache
def _sample_rows(s: int, device: torch.device, rank: int) -> tuple[torch.Tensor, torch.cuda.Event]:
    """Return the cached default draw and the event of its completion on the stream that drew it, which the streams
    of later callers wait on."""
    _not_under_capture(device, f"sampled rows of sequence length {s}")
    perm = torch.randperm(s, device=device, generator=torch.Generator(device=device).manual_seed(0))
    rows = perm[: sample_count(s, rank)].to(torch.int32).contiguous()
    return rows, torch.cuda.current_stream(device).record_event()


@functools.cache
def hadamard(device: torch.device | str) -> torch.Tensor:
    """Return the rotation of Q and K: the BF16 [HEAD_DIM, HEAD_DIM] Hadamard matrix with rows scaled by random signs
    (H H^T = HEAD_DIM I, exact in BF16; the signs of a generator seeded 0), built once per device (its host-to-device
    copy synchronizes)."""
    indexed = _indexed(device)
    if indexed != device:
        return hadamard(indexed)
    _not_under_capture(device, "rotation matrix")
    h = torch.ones(1, 1, device="cpu")
    while h.shape[0] < HEAD_DIM:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    signs = torch.randint(
        0, 2, (HEAD_DIM,), generator=torch.Generator("cpu").manual_seed(0), device="cpu"
    )
    signs = signs * 2 - 1
    return (signs[:, None] * h).to(device=device, dtype=torch.bfloat16)

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
"""Inputs of the DeCo preparation besides q, k and v, the same for every kernel family: rank_pair, sample_rows,
hadamard and the record of the compensation of a call that the host API hands to a family (Compensation). What the
functions need is built on first use and kept: the default rows per sequence length, row count and device,
the priorities of the drawn tokens and the rotation per device. Building them is not capturable, so a first use under
CUDA graph capture raises (the rule of a capture: api.py). The last two draws of given tokens are kept as well, for
the calls of a forward that pass the same tensors."""

import functools
from typing import NamedTuple

import torch

from . import target
from .constants import HEAD_DIM, MIN_SAMPLE_ROWS, QK_BLOCK, SAMPLE_FRAC
from .runtime import Basis, not_under_capture

# Key of a position that a draw of given tokens does not take, above every priority.
_NOT_DRAWN = 1 << 30
# Smallest number of positions a priority table is built for.
_MIN_PRIORITIES = 1 << 16
# The last draws of given tokens outside a capture, the newest first: per draw its key (token_tags, tokens, their
# version counters, s, device, row count), its rows and the event of their completion on the stream that drew them.
# Two are kept: the rows of Q and the rows of K of a call may be draws of two rates.
_KEPT_DRAWS = 2
_token_draws = []


class Compensation(NamedTuple):
    """The compensation of a call of prepare, as the host API resolved it for a kernel family.

    rank          (r_q, r_k): the channels of the Q-residual and of the K-residual term
    bases         the kind: "shared" (one basis for both terms) or, with two terms, "separate" (each term on the
                  basis of the other operand)
    gram_weight   bases "shared": the weight of the Gram matrix of Q in the basis; that of K has the rest of 1
    rows_q, rows_k    int32 [m] or [B, m]: the tokens that the Gram matrices of Q and of K are taken over, the same
                  tensor when one set serves both; None with a basis and without compensation channels
    map_rows      with valid lengths: the rows are the default draw over the padded sequence, which the eigensolver
                  maps into each valid prefix (row r stands for token floor(r n / S) of n valid tokens); otherwise
                  they are taken as given (a row at or past a length stands for the last valid token)
    basis         None, or the caller's basis, which the kernels read in place: no eigensolver runs and the rows are
                  not read
    """

    rank: tuple[int, int]
    bases: str
    gram_weight: float
    rows_q: torch.Tensor | None
    rows_k: torch.Tensor | None
    map_rows: bool
    basis: Basis | None


def rank_pair(rank: int | tuple[int, int] | list[int] | None) -> tuple[int, int]:
    """Return the rank (r_q, r_k), the channels of the Q-residual and of the K-residual compensation term, of a rank
    argument: None (half of the target's compensation channels each), an int r (r each) or a pair, a tuple or a list
    of two ints. ValueError for any other value, bools and floats among them, and unless both are multiples of
    QK_BLOCK from 0 up whose sum is within the target's max_comp_channels()."""
    limit = target.family().max_comp_channels()
    if rank is None:
        return limit // 2, limit // 2
    r_q = r_k = None
    if type(rank) is int:
        r_q = r_k = rank
    elif isinstance(rank, (tuple, list)) and len(rank) == 2:
        r_q, r_k = rank
    if (
        type(r_q) is not int
        or type(r_k) is not int
        or r_q < 0
        or r_k < 0
        or r_q % QK_BLOCK
        or r_k % QK_BLOCK
        or r_q + r_k > limit
    ):
        raise ValueError(
            f"rank {rank!r} is not offered on {target.arch()}: an int or a pair of ints, multiples of {QK_BLOCK} "
            f"from 0 up, r_q + r_k at most {limit}"
        )
    return r_q, r_k


def sample_count(
    s: int,
    rank: int | tuple[int, int] | list[int] | None = None,
    rate: float | None = None,
    min_rows: int | None = None,
) -> int:
    """Return the number of sampled tokens of a sequence of s tokens: floor(s rate), at least min_rows.

    * rank: a rank argument (rank_pair), checked whatever the other arguments.
    * rate: None is 1 / SAMPLE_FRAC; a fraction of the tokens in (0, 1], a float (or the int 1).
    * min_rows: None is twice the channels of the larger compensation term of the rank, at least MIN_SAMPLE_ROWS; an
      int from 1 up.
    """
    r_q, r_k = rank_pair(rank)
    if rate is None:
        rows = s // SAMPLE_FRAC
    elif type(rate) not in (float, int) or not 0.0 < rate <= 1.0:
        raise ValueError(f"rate must be a fraction of the tokens in (0, 1], got {rate!r}")
    else:
        rows = int(s * rate)
    if min_rows is None:
        min_rows = max(2 * r_q, 2 * r_k, MIN_SAMPLE_ROWS)
    elif type(min_rows) is not int or min_rows < 1:
        raise ValueError(f"min_rows must be an int from 1 up, got {min_rows!r}")
    return max(rows, min_rows)


def _indexed(device: torch.device | str) -> torch.device:
    """Return the device with its index ("cuda": the current CUDA device), so that cached tensors stay per device."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _version(tensor: torch.Tensor | None) -> int | None:
    """Return the version counter of a tensor, which a write in place advances; None for no tensor and for an
    inference tensor, which has none."""
    return None if tensor is None or tensor.is_inference() else tensor._version


def sample_rows(
    s: int,
    device: torch.device | str = "cuda",
    token_tags: torch.Tensor | None = None,
    rank: int | tuple[int, int] | list[int] | None = None,
    tokens: torch.Tensor | None = None,
    *,
    rate: float | None = None,
    min_rows: int | None = None,
) -> torch.Tensor:
    """Return the draw of the tokens that estimate a compensation basis: int32 indices, for
    m = sample_count(s, rank, rate, min_rows) [min(m, s)] of the default draw, [m] with token_tags, [B, m] with
    `tokens`.

    * Default draw (neither token_tags nor tokens): the first min(m, s) tokens of a permutation from a CUDA generator
      seeded 0, built once per length, m and device: the same tokens every call, and the draw of a smaller m is the
      start of a larger one's.
    * token_tags (integer tensor [s]) marks the modality of every token of a packed multimodal sequence: 0 video,
      positive text or audio, negative padding (never drawn). The tagged draw takes half its rows from the text and
      audio tokens, so both groups weigh equally in the basis where the video tokens would otherwise leave the others
      almost unsampled.
        - A group with fewer tokens than its share gives them all and the other group fills the draw.
        - A sequence with fewer than m tokens repeats its draw; one without tokens gets row 0.
        - A group gives its tokens in a fixed pseudo-random order of the positions: the same tags, the same rows.
    * tokens (bool tensor [B, s]): True where a batch element holds a token, whatever the tag of the position; without
      token_tags every token counts as video. Every batch element then gets the draw of its own tokens: for a call
      with valid prefix lengths n, tokens = arange(s) < n[:, None] draws the rows of each batch element within its
      valid prefix.
    * A draw with token_tags or tokens runs on the device and reads nothing back to the host: a CUDA graph can
      capture it. Outside a capture the last _KEPT_DRAWS of them are kept and returned while the caller passes the
      same token_tags and tokens tensors, compared by identity and, where a tensor has one, by its version counter:
      a mask is built once and passed again, and one rebuilt for every call is drawn anew every time.
    """
    device = _indexed(device)
    m = sample_count(s, rank, rate, min_rows)
    if token_tags is None and tokens is None:
        rows, drawn = _sample_rows(s, device, m)
        # A capture cannot wait on an outside event; the eager call before it did.
        if not torch.cuda.is_current_stream_capturing():
            torch.cuda.current_stream(device).wait_event(drawn)
        return rows
    capturing = device.type == "cuda" and torch.cuda.is_current_stream_capturing()
    key = (token_tags, tokens, _version(token_tags), _version(tokens), s, device, m)
    if not capturing:
        for kept, rows, drawn in _token_draws:
            if kept[0] is token_tags and kept[1] is tokens and kept[2:] == key[2:]:
                if drawn is not None:
                    # Another stream may have drawn the rows and still be at it.
                    torch.cuda.current_stream(device).wait_event(drawn)
                return rows
    if tokens is not None:
        if tokens.dtype != torch.bool or tokens.dim() != 2 or tokens.shape[1] != s:
            raise ValueError(f"tokens must be a bool tensor [B, {s}], got {tuple(tokens.shape)}")
        tokens = tokens.to(device)
    k = min(m, s)
    priority = _priorities(device, max(_MIN_PRIORITIES, 1 << (s - 1).bit_length()))[:s]
    slot = torch.arange(m, device=device)
    if token_tags is None:
        # Every token counts as video: one group, of which the draw takes the first k in priority order.
        drawn_rows = torch.where(tokens, priority, _NOT_DRAWN).topk(k, largest=False).indices
        num_drawn = tokens.sum(-1, keepdim=True).clamp(max=m)
        index = slot % num_drawn.clamp(min=1)
    else:
        tags = token_tags.to(device).reshape(-1)
        if tags.numel() != s:
            raise ValueError(f"token_tags must hold {s} tags, got {tuple(token_tags.shape)}")
        if tokens is not None:
            tags = torch.where(tokens, tags, -1)
        # Video tokens and the others, one group per leading index.
        groups = torch.stack([tags == 0, tags > 0])
        # The first k tokens of each group in priority order; behind the tokens of a shorter group come positions
        # that no slot below reads.
        video_rows, other_rows = (
            torch.where(groups, priority, _NOT_DRAWN).topk(k, largest=False).indices
        )
        num_video, num_other = groups.sum(-1, keepdim=True)
        take_other = num_other.clamp(max=m // 2)
        take_video = torch.minimum(num_video, m - take_other)
        take_other = torch.minimum(num_other, m - take_video)
        num_drawn = take_video + take_other
        slot = slot % num_drawn.clamp(min=1)
        index = torch.where(slot < take_video, slot, slot - take_video + k)
        drawn_rows = torch.cat([video_rows, other_rows], dim=-1)
    # Slot j holds draw j modulo the number drawn, the video tokens first: a short draw repeats.
    rows = drawn_rows.gather(-1, index)
    rows = rows.masked_fill(num_drawn == 0, 0).to(torch.int32)
    if not capturing:
        drawn = torch.cuda.current_stream(device).record_event() if device.type == "cuda" else None
        _token_draws.insert(0, (key, rows, drawn))
        del _token_draws[_KEPT_DRAWS:]
    return rows


@functools.cache
def _priorities(device: torch.device, capacity: int) -> torch.Tensor:
    """Return the priorities of the positions below `capacity` in a draw of given tokens: int32, distinct, below
    _NOT_DRAWN. Position i has the 30-bit mix of i whatever the capacity, so a sequence draws the same rows in every
    process."""
    not_under_capture(device, "priorities of the drawn tokens")
    # Every step is a bijection of the 30-bit integers.
    mask = _NOT_DRAWN - 1
    x = (torch.arange(capacity, dtype=torch.int64, device=device) + 0x9E3779B9) & mask
    x = ((x ^ (x >> 16)) * 0x85EBCA6B) & mask
    x = ((x ^ (x >> 13)) * 0xC2B2AE35) & mask
    return (x ^ (x >> 16)).to(torch.int32)


@functools.cache
def _sample_rows(s: int, device: torch.device, m: int) -> tuple[torch.Tensor, torch.cuda.Event]:
    """Return the cached default draw of m rows and the event of its completion on the stream that drew it, which the
    streams of later callers wait on."""
    not_under_capture(device, f"sampled rows of sequence length {s}")
    perm = torch.randperm(s, device=device, generator=torch.Generator(device=device).manual_seed(0))
    rows = perm[:m].to(torch.int32).contiguous()
    return rows, torch.cuda.current_stream(device).record_event()


@functools.cache
def hadamard(device: torch.device | str) -> torch.Tensor:
    """Return the rotation of Q and K: the BF16 [HEAD_DIM, HEAD_DIM] Hadamard matrix with rows scaled by random signs
    (H H^T = HEAD_DIM I, exact in BF16; the signs of a generator seeded 0), built once per device (its host-to-device
    copy synchronizes)."""
    indexed = _indexed(device)
    if indexed != device:
        return hadamard(indexed)
    not_under_capture(device, "rotation matrix")
    h = torch.ones(1, 1, device="cpu")
    while h.shape[0] < HEAD_DIM:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    signs = torch.randint(
        0, 2, (HEAD_DIM,), generator=torch.Generator("cpu").manual_seed(0), device="cpu"
    )
    signs = signs * 2 - 1
    return (signs[:, None] * h).to(device=device, dtype=torch.bfloat16)

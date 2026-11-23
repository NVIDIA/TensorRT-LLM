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
"""Inputs of the DeCo preparation besides q, k and v, the same for every kernel family: sample_rows and hadamard. Both
are built on first use and kept: the rows per sequence length, rank and device, the rotation per device. Building them
is not capturable, so a first use under CUDA graph capture raises: a capture follows one eager call of the shape."""

import functools

import torch

from .constants import DEFAULT_RANK, HEAD_DIM, SAMPLE_FRAC


def sample_count(s, rank=DEFAULT_RANK):
    """Return the number of sampled tokens for a sequence of s tokens and a basis of `rank` directions."""
    return max(s // SAMPLE_FRAC, 2 * rank)


def _indexed(device):
    """Return the device with its index ("cuda": the current CUDA device), so that cached tensors stay per device."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _not_under_capture(device, what):
    """Raise when `what` is first needed while the current CUDA stream is capturing a graph."""
    if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"DeCo-Attention {what} first needed during CUDA graph capture; "
            "run the call eagerly once before capturing"
        )


def sample_rows(s, device="cuda", token_tags=None, rank=DEFAULT_RANK):
    """Return the default draw of the tokens that estimate the compensation basis of `rank` directions: int32 indices.

    * A permutation from a CUDA generator seeded 0, built once per length, rank and device: the same tokens every call.
    * token_tags (integer [s]) marks the modality of every token of a packed multimodal sequence: 0 video, positive
      text or audio, negative padding (never drawn). The draw then takes half its rows from the text and audio tokens,
      so both groups weigh equally in the basis where the video tokens would otherwise leave the others almost
      unsampled; a group with fewer tokens than its share gives them all and the other group fills the draw.
    * The tagged draw reads the token counts back to the host and is not cached: callers keep the rows of a sequence
      and draw them outside CUDA graph capture.
    """
    device = _indexed(device)
    if token_tags is None:
        rows, drawn = _sample_rows(s, device, rank)
        # A capture cannot wait on an outside event; the eager call before it did.
        if not torch.cuda.is_current_stream_capturing():
            torch.cuda.current_stream(device).wait_event(drawn)
        return rows
    _not_under_capture(device, "sampled rows of a tagged sequence")
    tags = torch.as_tensor(token_tags, device=device).reshape(-1)[:s]
    gen = torch.Generator(device=device).manual_seed(0)
    video = torch.nonzero(tags == 0).squeeze(1)
    other = torch.nonzero(tags > 0).squeeze(1)
    n = sample_count(s, rank)
    n_video = min(n - min(n // 2, other.numel()), video.numel())
    n_other = min(n - n_video, other.numel())
    other_rows = other[torch.randperm(other.numel(), device=device, generator=gen)[:n_other]]
    video_rows = video[torch.randperm(video.numel(), device=device, generator=gen)[:n_video]]
    return torch.cat([video_rows, other_rows]).to(torch.int32).contiguous()


@functools.cache
def _sample_rows(s, device, rank):
    """Return the cached default draw and the event of its completion on the stream that drew it, which the streams
    of later callers wait on."""
    _not_under_capture(device, f"sampled rows of sequence length {s}")
    perm = torch.randperm(s, device=device, generator=torch.Generator(device=device).manual_seed(0))
    rows = perm[: sample_count(s, rank)].to(torch.int32).contiguous()
    return rows, torch.cuda.current_stream(device).record_event()


@functools.cache
def hadamard(device):
    """Return the rotation of Q and K: the BF16 [HEAD_DIM, HEAD_DIM] Hadamard matrix with rows scaled by random signs
    (H H^T = HEAD_DIM I, exact in BF16; the signs of a generator seeded 0), built once per device (its host-to-device
    copy synchronizes)."""
    indexed = _indexed(device)
    if indexed != device:
        return hadamard(indexed)
    _not_under_capture(device, "rotation matrix")
    h = torch.ones(1, 1)
    while h.shape[0] < HEAD_DIM:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    signs = torch.randint(0, 2, (HEAD_DIM,), generator=torch.Generator().manual_seed(0)) * 2 - 1
    return (signs[:, None] * h).to(device=device, dtype=torch.bfloat16)

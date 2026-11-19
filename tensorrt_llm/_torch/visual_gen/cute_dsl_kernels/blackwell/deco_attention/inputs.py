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
"""Inputs of the DeCo preparation besides q, k and v, the same for every kernel family: the sampled tokens whose q
and k rows estimate the compensation basis (1/256 of the sequence, at least twice the rank; on packed multimodal
sequences half of them from the text and audio tokens) and the sign-randomized Hadamard rotation of Q and K. Both
are built on first use and kept: the rows per sequence length, rank and device, the rotation per device. Building them
is not capturable, so a first use under CUDA graph capture raises: a capture follows one eager call of the shape."""

import functools

import torch

from .constants import DEFAULT_RANK, HEAD_DIM

SAMPLE_FRAC = 256


def sample_count(s, rank=DEFAULT_RANK):
    """The number of sampled tokens for a sequence of s tokens and a basis of `rank` directions."""
    return max(s // SAMPLE_FRAC, 2 * rank)


def _indexed(device):
    """The device with its index ("cuda" means the current CUDA device), so that cached tensors stay per device."""
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
    """int32 indices of the tokens that estimate the compensation basis of `rank` directions: the default draw (a
    permutation from a CUDA generator seeded 0), so every call on the same sequence length and rank uses the same
    tokens (built once per length, rank and device). token_tags (integer [s]) marks the modality of every token of a
    packed multimodal sequence: 0 video, positive text or audio, negative padding (never drawn). With it, the draw
    takes half its rows from the text and audio tokens (all of them when they are fewer) and the rest from the video
    tokens, so both groups weigh equally in the basis where the video tokens would otherwise leave the others almost
    unsampled. The tagged draw reads the token counts back to the host and is not cached: callers keep the rows of a
    sequence and draw them outside CUDA graph capture."""
    device = _indexed(device)
    if token_tags is None:
        return _sample_rows(s, device, rank)
    _not_under_capture(device, "sampled rows of a tagged sequence")
    tags = torch.as_tensor(token_tags, device=device).reshape(-1)[:s]
    gen = torch.Generator(device=device).manual_seed(0)
    video = torch.nonzero(tags == 0).squeeze(1)
    other = torch.nonzero(tags > 0).squeeze(1)
    n = sample_count(s, rank)
    n_other = min(n // 2, other.numel())
    o = other[torch.randperm(other.numel(), device=device, generator=gen)[:n_other]]
    w = video[torch.randperm(video.numel(), device=device, generator=gen)[: n - n_other]]
    return torch.cat([w, o]).to(torch.int32).contiguous()


@functools.cache
def _sample_rows(s, device, rank):
    _not_under_capture(device, f"sampled rows of sequence length {s}")
    perm = torch.randperm(s, device=device, generator=torch.Generator(device=device).manual_seed(0))
    return perm[: sample_count(s, rank)].to(torch.int32).contiguous()


def hadamard(device):
    """BF16 [HEAD_DIM, HEAD_DIM] Hadamard matrix with rows scaled by random signs (H H^T = HEAD_DIM I, exact in BF16;
    the signs of a generator seeded 0), built once per device (its host-to-device copy synchronizes)."""
    return _hadamard(_indexed(device))


@functools.cache
def _hadamard(device):
    _not_under_capture(device, "rotation matrix")
    h = torch.ones(1, 1)
    while h.shape[0] < HEAD_DIM:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    signs = torch.randint(0, 2, (HEAD_DIM,), generator=torch.Generator().manual_seed(0)) * 2 - 1
    return (signs[:, None] * h).to(device=device, dtype=torch.bfloat16)

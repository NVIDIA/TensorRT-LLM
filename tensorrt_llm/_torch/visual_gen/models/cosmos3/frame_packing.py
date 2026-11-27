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
"""Cosmos3 Nano Sim: frame-major token packing.

One latent frame is one block of ``A`` action rows followed by ``P`` vision tokens
(``P = ceil(H/2) * ceil(W/2)`` over the latent grid; with the published
checkpoint's ``A = 4``, 4 + 390 = 394 tokens per frame at 832x480). A chunk of
``n`` frames is ``n`` such blocks in time order, so every frame's tokens are
contiguous and the chunk can be cut into per-frame causal blocks and cached frame
by frame. The bidirectional model packs ``[all vision | all action]`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FramePacking:
    """Token layout of a chunk of ``num_frames`` latent frames."""

    num_frames: int
    action_tokens: int
    vision_tokens: int

    @property
    def tokens_per_frame(self) -> int:
        return self.action_tokens + self.vision_tokens

    @property
    def num_tokens(self) -> int:
        return self.num_frames * self.tokens_per_frame

    def interleave(self, vision: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """``vision [B, n*P, D]`` and ``action [B, n*A, D]`` (both frame-major) to the
        packed ``[B, n*(A+P), D]``, a new tensor."""
        n, a, p = self.num_frames, self.action_tokens, self.vision_tokens
        if vision.shape[1] != n * p or action.shape[1] != n * a:
            raise ValueError(
                f"packing {n} frames of {a} action + {p} vision tokens, got vision "
                f"{tuple(vision.shape)} and action {tuple(action.shape)}"
            )
        batch, _, dim = vision.shape
        packed = torch.cat([action.view(batch, n, a, dim), vision.view(batch, n, p, dim)], dim=2)
        return packed.view(batch, n * (a + p), dim)

    def vision(self, packed: torch.Tensor) -> torch.Tensor:
        """The vision tokens ``[B, n*P, D]`` of the packed ``[B, n*(A+P), D]``: a view
        for one frame, a copy for several."""
        n, a, p = self.num_frames, self.action_tokens, self.vision_tokens
        batch, _, dim = packed.shape
        return packed.view(batch, n, a + p, dim)[:, :, a:].reshape(batch, n * p, dim)

    def frame_slices(self, frame: int) -> tuple[slice, slice]:
        """Token ranges of ``frame``'s action rows and vision tokens within the chunk."""
        start = frame * self.tokens_per_frame
        return (
            slice(start, start + self.action_tokens),
            slice(start + self.action_tokens, start + self.tokens_per_frame),
        )

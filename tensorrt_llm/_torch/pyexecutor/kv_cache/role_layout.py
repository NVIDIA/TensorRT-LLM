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
"""The ways a buffer's bytes split across ranks, as the segments a cache manager gives its buffers.
Standard library only."""

from __future__ import annotations

from typing import Iterable

from .page_layout import Segment

__all__ = ["head_major", "replicated", "sectioned", "token_major"]


def head_major(size: int, heads: int) -> tuple[Segment, ...]:
    """``heads`` heads outermost, split across ranks: ``[head, ...]``."""
    return (Segment(size, (("head", heads),), "head"),)


def token_major(size: int, tokens: int, heads: int) -> tuple[Segment, ...]:
    """Tokens outermost, ``heads`` heads split across ranks within each token:
    ``[token, head, ...]``."""
    return (Segment(size, (("token", tokens), ("head", heads)), "head"),)


def replicated(size: int) -> tuple[Segment, ...]:
    """The same bytes on every rank of the layer group's grid."""
    return (Segment(size),)


def sectioned(section_bytes: Iterable[int]) -> tuple[Segment, ...]:
    """Consecutive sections of the given sizes, each split across ranks on its own, such as a
    recurrent convolution state ``[x | B | C]``."""
    return tuple(Segment(n, (("byte", n),), "byte") for n in section_bytes)

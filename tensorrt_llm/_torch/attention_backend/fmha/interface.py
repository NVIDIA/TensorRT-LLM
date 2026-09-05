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

import weakref
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


class FmhaPhase(str, Enum):
    """Attention phase checked by a phased FMHA library."""

    CONTEXT = "context"
    GENERATION = "generation"


class Fmha(ABC):
    """Common runtime contract for TRT-LLM attention FMHA libraries."""

    supports_skip_correction = False

    def __init__(self, attn: "TrtllmAttention"):
        self._attn_ref: weakref.ReferenceType["TrtllmAttention"] = weakref.ref(attn)

    @property
    def attn(self) -> "TrtllmAttention":
        attn = self._attn_ref()
        if attn is None:
            raise RuntimeError("The owning TrtllmAttention instance has been garbage collected.")
        return attn

    @classmethod
    def is_available(cls, attn: "TrtllmAttention") -> bool:
        """Return whether this library can serve the given attention layer.

        Evaluated once per ``FmhaManager`` construction, currently at the end
        of ``TrtllmAttention.update_quant_config()``. Conditions must depend
        only on state finalized before manager construction and invariant for
        its lifetime. Reading state that a model rewrites later, such as a
        remapped ``layer_idx``, silently leaves the library list stale because
        it is not revalidated. Request-varying conditions belong in
        ``is_supported`` instead.
        """
        return True

    def is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        """Return whether this library supports the request or requested phase.

        Forward-varying selection conditions must be represented in
        ``FmhaManager._make_cache_key``. Conditions omitted from that key must
        remain invariant for the attention instance. Size-based conditions
        must also preserve the same result throughout each FMHA cache grid
        cell or add the relevant boundary to the grid's candidate list.
        """
        return True

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> None:
        raise NotImplementedError

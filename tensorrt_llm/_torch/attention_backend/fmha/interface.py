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
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


class Fmha(ABC):
    """Common runtime contract for TRT-LLM attention FMHA libraries.

    Most FMHA backends are owned by a :class:`TrtllmAttention` layer and
    are driven by :meth:`TrtllmAttention.forward`. A small subset
    (indexer-style proxy FMHA libraries used by sparse-attention
    predictors) live in the same registry but have no owning attention
    layer; they pass ``None`` for ``attn``. The :meth:`attn` property
    raises only when callers actually dereference the owner, so the
    no-owner subclasses never trip it as long as they do not call into
    ``self.attn``.
    """

    def __init__(self, attn: Optional["TrtllmAttention"] = None):
        self._attn_ref: Optional[weakref.ReferenceType["TrtllmAttention"]] = (
            weakref.ref(attn) if attn is not None else None
        )

    @property
    def attn(self) -> "TrtllmAttention":
        if self._attn_ref is None:
            raise RuntimeError(
                f"{type(self).__name__} was constructed without an owning "
                "TrtllmAttention instance. This typically means an "
                "indexer-style FMHA backend was asked for a property "
                "that only makes sense for main-attention FMHA "
                "backends."
            )
        attn = self._attn_ref()
        if attn is None:
            raise RuntimeError("The owning TrtllmAttention instance has been garbage collected.")
        return attn

    @classmethod
    def is_available(cls, attn: Optional["TrtllmAttention"] = None) -> bool:
        return True

    def is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> bool:
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

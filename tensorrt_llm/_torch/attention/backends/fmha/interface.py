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
from typing import TYPE_CHECKING, ClassVar, Optional, final

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionForwardArgs
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


class FmhaPhase(str, Enum):
    """Attention phase checked by a phased FMHA library."""

    CONTEXT = "context"
    GENERATION = "generation"


class Fmha(ABC):
    """Common runtime contract for TRT-LLM attention FMHA libraries."""

    supports_skip_correction: ClassVar[bool] = False
    supports_block_sparse_inputs: ClassVar[bool] = False

    def __init__(self, attn: "TrtllmAttention"):
        self._attn_ref: weakref.ReferenceType["TrtllmAttention"] = weakref.ref(attn)

    @property
    def attn(self) -> "TrtllmAttention":
        attn = self._attn_ref()
        if attn is None:
            raise RuntimeError("The owning TrtllmAttention instance has been garbage collected.")
        return attn

    @classmethod
    @final
    def is_available(cls, attn: "TrtllmAttention") -> bool:
        """Return whether this library can serve the given attention layer.

        Check shared capabilities before the implementation's
        ``_is_available`` hook. Libraries declare their capabilities as class
        attributes and override only the hook for additional static checks.

        Evaluated once per ``FmhaManager`` construction, currently at the end
        of ``TrtllmAttention.update_quant_config()``. Conditions must depend
        only on state finalized before manager construction and invariant for
        its lifetime. Reading state that a model rewrites later, such as a
        remapped ``layer_idx``, silently leaves the library list stale because
        it is not revalidated. Request-varying conditions belong in
        ``is_supported`` instead.
        """
        if attn.skip_correction_threshold > 0.0 and not cls.supports_skip_correction:
            logger.debug(
                f"{cls.__name__} is unavailable: skip-correction is enabled and unsupported."
            )
            return False
        return cls._is_available(attn)

    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        """Check implementation-specific static restrictions after capability checks.

        Delegate to ``super()._is_available(attn)`` to reuse a parent hook;
        calling ``is_available`` here would re-enter the shared wrapper.
        """
        return True

    @final
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

        Shared request capability checks belong here, before delegating to
        ``_is_supported``. Libraries override only that hook for their
        request-specific restrictions.

        Forward-varying selection conditions must be represented in
        ``FmhaManager._make_cache_key``. Conditions omitted from that key must
        remain invariant for the attention instance. Size-based conditions
        must also preserve the same result throughout each FMHA cache grid
        cell or add the relevant boundary to the grid's candidate list.
        """
        if (
            forward_args.sparse_runtime_params.block_sparse_inputs is not None
            and not self.supports_block_sparse_inputs
        ):
            logger.debug(f"{type(self).__name__} does not support block-sparse inputs.")
            return False
        return self._is_supported(q, k, v, metadata, forward_args, phase=phase)

    def _is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        """Check implementation-specific request restrictions after capability checks.

        Delegate to ``super()._is_supported(...)`` to reuse a parent hook;
        calling ``is_supported`` here would re-enter the shared wrapper.
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

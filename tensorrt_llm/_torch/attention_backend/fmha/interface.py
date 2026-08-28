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
from typing import TYPE_CHECKING, NamedTuple, Optional, Protocol

import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


class _CuteDslMlaStagingKey(NamedTuple):
    """Identifies CuTe-DSL MLA inputs staged into a shared workspace.

    Attributes:
        is_capturing: Whether the staging occurred during CUDA graph capture.
        workspace_ptr: Address of the shared staging workspace.
        block_tables_ptr: Address of the source block tables.
        block_tables_shape: Shape of the source block tables.
        sequence_lengths_ptr: Address of the source sequence lengths.
        sequence_lengths_offset: Offset applied to the source sequence lengths.
        batch_beam: Number of generation sequences, including beam expansion.
        padded_num_pages: Page-table width after CuTe-DSL alignment padding.
    """

    is_capturing: bool
    workspace_ptr: int
    block_tables_ptr: int
    block_tables_shape: tuple[int, ...]
    sequence_lengths_ptr: int
    sequence_lengths_offset: int
    batch_beam: int
    padded_num_pages: int


class MlaBackendPolicy(Protocol):
    """Selects the MLA generation backend for one scheduler batch."""

    def __call__(
        self,
        requested_backend: str,
        metadata: "TrtllmAttentionMetadata",
        num_gen_tokens: int,
    ) -> str:
        """Return the backend to use for the supplied batch composition.

        Args:
            requested_backend: Backend selected by the attention instance.
            metadata: Runtime metadata for the current scheduler batch.
            num_gen_tokens: Number of generation tokens in the batch.

        Returns:
            Backend name to use for MLA generation in this batch.
        """
        ...


class FmhaPhase(str, Enum):
    """Attention phase checked by a phased FMHA library."""

    CONTEXT = "context"
    GENERATION = "generation"


class Fmha(ABC):
    """Common runtime contract for TRT-LLM attention FMHA libraries."""

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
        """Return whether this library supports the request or requested phase."""
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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashInfer SM120 sparse-MLA FMHA library."""

from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

from .interface import Fmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


FMHA_NAME = "flashinfer_sparse_mla"
_SUPPORTED_ALGORITHMS = frozenset({"deepseek_v4", "dsa"})


def _sparse_mla_op():
    from flashinfer.mla._sparse_mla_sm120 import _sparse_mla_sm120_paged_attention

    return _sparse_mla_sm120_paged_attention


def is_flashinfer_sparse_mla_enabled(algorithm: Optional[str]) -> bool:
    """Whether the selected FMHA list can own this sparse cache layout."""
    if algorithm not in _SUPPORTED_ALGORITHMS or get_sm_version() not in (120, 121):
        return False

    # Import locally to avoid a registry import cycle while this module is
    # itself being registered.
    from .registry import get_enabled_fmha_lib_names

    enabled_names = get_enabled_fmha_lib_names()
    # This FMHA owns an incompatible packed cache layout. Requiring it to be
    # first prevents an always-supported fallback from reading that cache.
    if not enabled_names or enabled_names[0] != FMHA_NAME:
        return False
    try:
        _sparse_mla_op()
    except (AttributeError, ImportError):
        return False
    return True


class FlashInferSparseMlaFmha(Fmha):
    """SM120/SM121 sparse MLA for TRTLLM's DSA and DeepSeek-V4 backends.

    This library owns a packed KV-cache layout, so supported layers must not
    fall through to the C++ attention op. Unsupported model-wide features are
    rejected by the algorithm-specific runner with a clear error instead.
    """

    @classmethod
    def is_available(cls, attn: "TrtllmAttention") -> bool:
        if not getattr(attn, "is_mla_enable", False):
            return False
        sparse_params = getattr(attn, "sparse_params", None)
        algorithm = getattr(sparse_params, "algorithm", None)
        available = is_flashinfer_sparse_mla_enabled(algorithm)
        if not available and algorithm in _SUPPORTED_ALGORITHMS and get_sm_version() in (120, 121):
            logger.debug(
                "FlashInfer sparse-MLA FMHA is unavailable: the library is disabled "
                "or flashinfer's SM120 sparse-MLA op cannot be imported."
            )
        return available

    def is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> bool:
        # Cache allocation and RoPE fusion use the same availability check.
        # Once selected, this library must own every call for the layer because
        # the regular fallback op cannot consume its packed KV-cache layout.
        return True

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> None:
        algorithm = self.attn.sparse_params.algorithm
        if algorithm == "deepseek_v4":
            from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.flashinfer import (
                run_flashinfer_sparse_mla,
            )

            run_flashinfer_sparse_mla(self.attn, q, metadata, forward_args)
            return
        if algorithm == "dsa":
            from tensorrt_llm._torch.attention_backend.sparse.dsa_flashinfer import (
                run_flashinfer_sparse_mla,
            )

            run_flashinfer_sparse_mla(self.attn, q, metadata, forward_args)
            return
        raise RuntimeError(f"Unsupported FlashInfer sparse-MLA algorithm: {algorithm!r}.")


__all__ = [
    "FMHA_NAME",
    "FlashInferSparseMlaFmha",
    "is_flashinfer_sparse_mla_enabled",
]

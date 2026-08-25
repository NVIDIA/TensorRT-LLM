# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashInfer SM120 sparse-MLA FMHA library."""

from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs
from tensorrt_llm._torch.attention_backend.sparse.flashinfer_utils import get_sparse_mla_op
from tensorrt_llm._torch.modules.rotary_embedding import RotaryEmbedding
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

from .interface import Fmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


_SUPPORTED_ALGORITHMS = frozenset({"deepseek_v4", "dsa"})


def is_flashinfer_sparse_mla_enabled(algorithm: Optional[str]) -> bool:
    """Whether FlashInfer sparse MLA is available for this model."""
    if algorithm not in _SUPPORTED_ALGORITHMS or get_sm_version() not in (120, 121):
        return False

    try:
        get_sparse_mla_op()
    except (AttributeError, ImportError) as error:
        logger.warning(
            "FlashInfer sparse MLA on SM120/SM121 requires private symbol "
            "flashinfer.mla._sparse_mla_sm120._sparse_mla_sm120_paged_attention "
            "from the pinned flashinfer-python==0.6.16 build; backend disabled: "
            f"{error}"
        )
        return False
    return True


class FlashInferSparseMlaFmha(Fmha):
    """SM120/SM121 sparse MLA for DSA and DeepSeek-V4."""

    def __init__(self, attn: "TrtllmAttention"):
        super().__init__(attn)
        self.rotary_emb: RotaryEmbedding = RotaryEmbedding(
            attn.rope_params,
            head_dim=attn.qk_rope_head_dim,
            is_neox=False,
        )

    @classmethod
    def is_available(cls, attn: "TrtllmAttention") -> bool:
        if not attn.is_mla_enable or getattr(attn, "kv_cache_dtype", None) != "fp8_ds_mla":
            return False
        return is_flashinfer_sparse_mla_enabled(getattr(attn.sparse_params, "algorithm", None))

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

            run_flashinfer_sparse_mla(self.attn, q, metadata, forward_args, self.rotary_emb)
            return
        if algorithm == "dsa":
            from tensorrt_llm._torch.attention_backend.sparse.dsa_flashinfer import (
                run_flashinfer_sparse_mla,
            )

            run_flashinfer_sparse_mla(self.attn, q, metadata, forward_args, self.rotary_emb)
            return
        raise RuntimeError(f"Unsupported FlashInfer sparse-MLA algorithm: {algorithm!r}.")


__all__ = [
    "FlashInferSparseMlaFmha",
]

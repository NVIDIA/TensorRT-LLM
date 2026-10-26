# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRT-LLM backend wrapper for QSA sparse metadata."""

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention, TrtllmAttentionMetadata

from .metadata import QSAMetadata


class QSATrtllmAttention(TrtllmAttention):
    """TRT-LLM backend using QSA's page and packed-row metadata.

    Dense-versus-sparse execution is selected later by ``QSASparseHooks``;
    this class only installs metadata required by both paths.
    """

    Metadata = TrtllmAttentionMetadata
    metadata_extensions = (QSAMetadata,)


__all__ = ["QSATrtllmAttention"]

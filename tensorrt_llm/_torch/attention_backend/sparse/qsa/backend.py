# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRT-LLM backend wrapper for QSA sparse metadata."""

from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention

from .metadata import QSAAttentionMetadata


class QSATrtllmAttention(TrtllmAttention):
    """Use the regular backend below threshold and QSA hooks above it."""

    Metadata = QSAAttentionMetadata


__all__ = ["QSATrtllmAttention"]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``POST /v1/embeddings`` on an encode-only ``LLM``."""

from typing import Any, Optional

from tensorrt_llm.serve._base.serving import BaseServing
from tensorrt_llm.serve.api.openai.embeddings.batcher import EncodeBatcher


class EmbeddingServing(BaseServing):
    """``POST /v1/embeddings`` on an encode-only ``LLM``.

    Concurrent requests are coalesced by ``EncodeBatcher``; the composition
    root keeps a reference to the batcher, because the lifespan starts and
    stops it and the health probe reports it.
    """

    def __init__(self, *, model: str, generator: Any,
                 embedding_max_queue_delay: float,
                 embedding_max_queue_size: int) -> None:
        self.model = model
        self.generator = generator
        self._embedding_max_queue_delay = embedding_max_queue_delay
        self._embedding_max_queue_size = embedding_max_queue_size
        self.embedding_batcher: Optional[EncodeBatcher] = None
        self._init_embedding_batcher()

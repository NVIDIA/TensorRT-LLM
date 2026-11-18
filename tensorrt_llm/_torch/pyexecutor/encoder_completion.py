# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How one forward of ``ScheduledEncoderExecutor`` completes its requests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .llm_request import LlmRequest

__all__ = ["Completion", "PoolingCompletion"]


class Completion(Protocol):
    """How one forward's outputs complete its requests; paired with the runner at initialization."""

    accepts_multimodal_inputs: bool
    """Whether a request may carry multimodal inputs."""

    def store(self, requests: list[LlmRequest], outputs: dict[str, Any]) -> None:
        """Attach one forward's results to its requests before their responses are created."""

    def retire(self, requests: list[LlmRequest]) -> None:
        """Release what ``store`` exported for requests whose responses will not be published."""


class PoolingCompletion:
    """Store each request's pooled row as its context logits, when the request asked for them."""

    accepts_multimodal_inputs = False

    def store(self, requests: list[LlmRequest], outputs: dict[str, Any]) -> None:
        for request, row in zip(requests, outputs["logits"]):
            if request.py_return_context_logits:
                request.py_result.append_context_logits(row.unsqueeze(0) if row.ndim == 1 else row)

    def retire(self, requests: list[LlmRequest]) -> None:
        pass

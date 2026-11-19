# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How one forward of ``ScheduledEncoderExecutor`` completes its requests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .llm_request import LlmRequest

__all__ = ["Completion", "MultimodalEmbeddingCompletion", "PoolingCompletion"]


class Completion(Protocol):
    """How one forward's outputs complete its requests; paired with the runner at initialization."""

    accepts_multimodal_inputs: bool
    """Whether a request may carry multimodal inputs."""

    def store(self, requests: list[LlmRequest], outputs: dict[str, Any]) -> None:
        """Attach one forward's results to its requests before their responses are created."""

    def retire(self, requests: list[LlmRequest]) -> None:
        """Withdraw what ``store`` attached, for requests that finish without their result."""


class PoolingCompletion:
    """Store each request's pooled row as its context logits, when the request asked for them."""

    accepts_multimodal_inputs = False

    def store(self, requests: list[LlmRequest], outputs: dict[str, Any]) -> None:
        for request, row in zip(requests, outputs["logits"]):
            if request.py_return_context_logits:
                request.py_result.append_context_logits(row.unsqueeze(0) if row.ndim == 1 else row)

    def retire(self, requests: list[LlmRequest]) -> None:
        for request in requests:
            request.py_result.drop_context_logits()


class MultimodalEmbeddingCompletion:
    """Export each request's multimodal embeddings, and its MRoPE positions, as CUDA IPC handles.

    A request without multimodal items in the batch completes without a result. Only the rank
    whose responses reach the frontend exports: an exported allocation stays alive until a
    consumer opens and releases its handle, or ``retire`` consumes it.
    """

    accepts_multimodal_inputs = True

    def __init__(self, *, exports: bool) -> None:
        self._exports = exports

    def store(self, requests: list[LlmRequest], outputs: dict[str, Any]) -> None:
        if not self._exports:
            return
        metadata = outputs.get("mm_embedding_metadata")
        mrope_position_ids = outputs.get("mrope_position_ids")
        mrope_position_deltas = outputs.get("mrope_position_deltas")
        for result, (request_index, embeddings, lengths) in enumerate(
            zip(
                outputs["mm_embedding_request_indices"],
                outputs["mm_embeddings"],
                outputs["mm_embedding_lengths"],
                strict=True,
            )
        ):
            py_result = requests[request_index].py_result
            py_result.append_mm_embeddings(
                embeddings, lengths, None if metadata is None else metadata[result]
            )
            if mrope_position_ids is not None and mrope_position_deltas is not None:
                py_result.set_mrope_position(
                    mrope_position_ids[result], mrope_position_deltas[result]
                )

    def retire(self, requests: list[LlmRequest]) -> None:
        for request in requests:
            request.py_result.release_exports()

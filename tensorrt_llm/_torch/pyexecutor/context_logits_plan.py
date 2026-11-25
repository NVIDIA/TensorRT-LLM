# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-owned storage of context logits and the eager projection plan."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from tensorrt_llm.executor.result import reduce_prompt_logprobs
from tensorrt_llm.logger import logger

from .ced_replay import encoder_replay_tokens

if TYPE_CHECKING:
    from ..modules.embedding import LMHead
    from .llm_request import LlmRequest
    from .scheduler import ScheduledRequests


def store_context_logits(
    request: LlmRequest,
    logits: torch.Tensor,
    *,
    offset: int = 0,
    reusable: bool = False,
    retain_owned: bool = False,
) -> None:
    """Consume context rows from either projection path using one result contract.

    Prompt logprobs are reduced on the logits' device; only the reduced rows and
    the final prompt row's log probabilities reach host memory.

    Args:
        request: Context request whose current chunk produced ``logits``.
        logits: ``[rows, vocab]`` FP32 logits of consecutive positions of the
            current chunk, over the full (gathered) vocabulary.
        offset: Row offset of ``logits`` within the current prefill chunk.
        reusable: ``logits`` is projection scratch that the caller overwrites
            next; host copies therefore complete before returning.
        retain_owned: Store a view of the result buffer in the diff, so it
            does not keep the whole logits batch alive.
    """
    if offset == 0 and request.prepopulated_prompt_len > 0:
        logger.warning(
            "Because of KV cache reuse, not all context logits could be produced "
            f"for request {request.request_id}."
        )
    compact = request.py_prompt_logprobs is not None
    if request.py_context_logits_explicit or not compact:
        request.py_result.append_context_logits(
            logits, reusable=reusable, retain_owned=retain_owned
        )
    if not compact:
        return
    begin = request.context_current_position + offset + 1
    end = min(request.py_prompt_len, begin + logits.shape[0])
    targets = request.get_tokens(0)[begin:end]
    # Appended even when empty, so a compact request always reports its rows.
    request.py_result.append_prompt_logprobs(
        reduce_prompt_logprobs(logits[: len(targets)], targets, request.py_prompt_logprobs)
    )
    if offset + logits.shape[0] == request.context_chunk_size and request.is_last_context_chunk:
        request.py_result.set_prompt_last_logprobs(_normalized_host_row(logits[-1:]))


def _normalized_host_row(row: torch.Tensor) -> torch.Tensor:
    """Log-softmax ``row`` on its own device, like the other prompt rows, and copy it to host.

    Args:
        row: ``[1, vocab]`` logits of any floating dtype.

    Returns:
        ``[1, vocab]`` FP32 host tensor of log probabilities.
    """
    try:
        logprobs = torch.log_softmax(row.to(dtype=torch.float32), dim=-1)
    except torch.OutOfMemoryError:
        if not row.is_cuda:
            raise
        logprobs = torch.log_softmax(row.to("cpu", dtype=torch.float32), dim=-1)
    return logprobs.to("cpu", copy=True)


def _keeps_context_logits(request: LlmRequest) -> bool:
    return request.py_return_context_logits and (
        request.py_context_logits_explicit or request.py_prompt_logprobs is None
    )


class ContextLogitsPlan:
    """Store context logits and select the rows needed by the sampler.

    Args:
        batch: Requests scheduled for the current step.
        project_in_forward: When True, project and store context logits during
            the forward. When False, normalize full-row logits after the forward.
    """

    def __init__(self, batch: ScheduledRequests, *, project_in_forward: bool = True) -> None:
        self.batch = batch
        self.project_in_forward = project_in_forward
        # The model consumes the replay plan during forward. Input packing can
        # reorder contexts, so key counts by request identity until projection.
        self.replay_rows = {
            id(request): count
            for request in batch.context_requests
            if (count := encoder_replay_tokens(request))
        }
        self.handled = False

    @classmethod
    def for_batch(
        cls,
        heads: Sequence[LMHead],
        batch: ScheduledRequests,
        *,
        is_spec_decode: bool,
    ) -> ContextLogitsPlan | None:
        """Select bounded projection or account for replay in full-row logits.

        Args:
            heads: ``modules.context_logits.context_logits_heads`` of the model,
                computed once.
            batch: The scheduled batch.
            is_spec_decode: Whether this step runs speculative decoding.
        """
        if is_spec_decode or not any(
            request.py_return_context_logits for request in batch.context_requests
        ):
            return None
        project_in_forward = any(
            head.context_logits_workspace is not None for head in heads
        ) and not any(
            getattr(request, "py_logits_post_processors", None) for request in batch.all_requests()
        )
        plan = cls(batch, project_in_forward=project_in_forward)
        return plan if project_in_forward or plan.replay_rows else None

    def _chunk_rows(self, contexts: Sequence[LlmRequest]) -> list[slice]:
        """New prompt rows in packing order, excluding each replayed prefix."""
        rows = []
        end = 0
        for request in contexts:
            start = end + self.replay_rows.get(id(request), 0)
            end = start + request.context_chunk_size
            rows.append(slice(start, end))
        return rows

    def store_full_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Store raw prompt logits and retain the rows needed for sampling.

        The stored prompt logits precede per-request logits post-processing.

        Args:
            logits: [physical_rows, vocab] logits in packing order, with each
                context request's replay rows before its new prompt rows.

        Returns:
            [num_contexts + tail_rows, vocab] logits containing the last new
            prompt row for each context request followed by the decode tail.
        """
        contexts = self.batch.context_requests
        chunks = self._chunk_rows(contexts)
        context_rows = chunks[-1].stop if chunks else 0
        if context_rows > logits.shape[0]:
            raise ValueError("Context logits input does not match scheduled token rows")
        kept = []
        for request, chunk in zip(contexts, chunks, strict=True):
            if request.py_return_context_logits:
                store_context_logits(request, logits[chunk], retain_owned=True)
            kept.append(logits[chunk.stop - 1 : chunk.stop])
        kept.append(logits[context_rows:])
        return torch.cat(kept)

    def project(self, hidden: torch.Tensor, head: LMHead) -> torch.Tensor:
        """Store requested context rows and project the rows the sampler needs.

        Args:
            hidden: ``[num_tokens, hidden_size]`` hidden states of the scheduled
                tokens, context chunks first and generation rows last.
            head: Prepared LM head.

        Returns:
            ``[num_contexts + generation_rows, vocab]`` FP32 rows: the last row
            of each context chunk, then every generation row.
        """
        workspace = head.context_logits_workspace
        assert workspace is not None
        contexts = self.batch.context_requests
        chunks = self._chunk_rows(contexts)
        context_rows = chunks[-1].stop if chunks else 0
        generation_rows = hidden.shape[0] - context_rows
        if generation_rows < 0:
            raise ValueError("Context logits input does not match scheduled token rows")
        # Own the sampling rows before result storage can consume the remaining
        # headroom. No view of a reusable projection buffer escapes this call.
        sampling = None
        storage_error = None
        try:
            sampling = torch.empty(
                (len(contexts) + generation_rows, head.num_embeddings),
                dtype=torch.float32,
                device=hidden.device,
            )
            for request in contexts:
                if _keeps_context_logits(request):
                    request.py_result.prepare_context_logits(head.num_embeddings, hidden.device)
        except RuntimeError as error:
            # CUDA OOM, or a failed host allocation after the GPU fallback.
            storage_error = error
        workspace.require_all(storage_error is None, "context-logits sampling rows and storage")
        assert sampling is not None

        # A local storage failure is reported at the next TP vote, so no rank
        # raises while its peers enter a collective.
        tail_rows: list[int] = []
        tail_slots: list[int] = []
        for index, (request, chunk) in enumerate(zip(contexts, chunks, strict=True)):
            count = request.context_chunk_size
            if not request.py_return_context_logits:
                tail_rows.append(chunk.stop - 1)
                tail_slots.append(index)
                continue
            rows = hidden[chunk]
            host_storage = (
                _keeps_context_logits(request) and request.py_result.context_logits_on_cpu
            )
            buffers = workspace.acquire(
                count, host_storage=host_storage, local_ok=storage_error is None
            )
            capacity = buffers[0].shape[0]
            for begin in range(0, count, capacity):
                end = min(begin + capacity, count)
                logits = workspace.project(
                    head, rows[begin:end], buffers, local_ok=storage_error is None
                )
                if storage_error is None:
                    try:
                        store_context_logits(request, logits, offset=begin, reusable=True)
                    except RuntimeError as error:
                        logger.error(
                            f"Storing context logits of request {request.request_id} failed: {error}"
                        )
                        storage_error = error
                if end == count:
                    sampling[index].copy_(logits[-1])
            del logits, buffers

        # Contexts without requested logits need only their last row; project
        # those together with the generation rows.
        tail_count = len(tail_rows) + generation_rows
        if tail_count == 0:
            workspace.require_all(storage_error is None, "context-logits storage")
        else:
            tail = slot_indices = None
            try:
                tail = hidden[context_rows:]
                slots = list(range(len(contexts), len(contexts) + generation_rows))
                if tail_rows:
                    indices = torch.tensor(tail_rows, dtype=torch.int64, device=hidden.device)
                    tail = torch.cat([hidden.index_select(0, indices), tail])
                    slots = tail_slots + slots
                slot_indices = torch.tensor(slots, dtype=torch.int64, device=hidden.device)
            except torch.OutOfMemoryError:
                # Reported to every TP rank by the capacity vote below.
                pass
            buffers = workspace.acquire(
                tail_count,
                host_storage=False,
                local_ok=storage_error is None and slot_indices is not None,
            )
            assert tail is not None and slot_indices is not None
            capacity = buffers[0].shape[0]
            for begin in range(0, tail_count, capacity):
                end = min(begin + capacity, tail_count)
                logits = workspace.project(
                    head, tail[begin:end], buffers, local_ok=storage_error is None
                )
                sampling.index_copy_(0, slot_indices[begin:end], logits)
        self.handled = True
        return sampling

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-owned storage of context logits and the eager projection plan."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from tensorrt_llm.executor.result import reduce_prompt_logprobs
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from ..modules.embedding import LMHead
    from .llm_request import LlmRequest
    from .scheduler import ScheduledRequests


def store_context_logits(
    request: LlmRequest, logits: torch.Tensor, *, offset: int = 0, reusable: bool = False
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
    """
    if offset == 0 and request.prepopulated_prompt_len > 0:
        logger.warning(
            "Because of KV cache reuse, not all context logits could be produced "
            f"for request {request.request_id}."
        )
    compact = request.py_prompt_logprobs is not None
    if request.py_context_logits_explicit or not compact:
        request.py_result.append_context_logits(logits, reusable=reusable)
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


def context_logits_rows(request: LlmRequest, *, context_logits_handled: bool) -> int:
    """Rows a context request occupies in a step's logits, as the sampler reads them.

    A request that returns context logits keeps its whole chunk, unless the
    context-logits plan already stored them; every other context request keeps
    only its last row, the one sampled from.

    Args:
        request: A context request of the step.
        context_logits_handled: Whether the plan stored the step's context logits.

    Returns:
        The number of rows.
    """
    if request.py_return_context_logits and not context_logits_handled:
        return request.context_chunk_size
    return 1


def store_context_logits_from_full_rows(
    context_requests: Sequence[LlmRequest], logits: torch.Tensor
) -> torch.Tensor:
    """Store context logits from logits that hold every context token; keep the sampling rows.

    Without the context-logits plan, a step that gathers context logits projects
    every scheduled context token, for every context request. Storing the
    requested chunks here, before any logits post-processor runs, keeps them the
    model's own scores, as the plan does; what remains is the plan's layout, one
    row per context request followed by the other rows.

    Args:
        context_requests: The step's context requests, in row order.
        logits: ``[context_tokens + other_rows, vocab]`` with all context rows
            first; rows after the context tokens are kept as they are.

    Returns:
        A new ``[num_contexts + other_rows, vocab]`` tensor: the last row of
        each context chunk, then the remaining rows.
    """
    num_context_tokens = sum(request.context_chunk_size for request in context_requests)
    if num_context_tokens > logits.shape[0]:
        raise ValueError(
            f"logits have {logits.shape[0]} rows, fewer than the "
            f"{num_context_tokens} scheduled context tokens"
        )
    keep: list[int] = []
    start = 0
    for request in context_requests:
        end = start + request.context_chunk_size
        if request.py_return_context_logits:
            # Copied into the request's own storage before returning: logits
            # may be a graph's static output that the next replay overwrites.
            store_context_logits(request, logits[start:end], reusable=True)
        keep.append(end - 1)
        start = end
    keep.extend(range(start, logits.shape[0]))
    return logits.index_select(0, torch.tensor(keep, device=logits.device))


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
    """Consume context chunks directly; return only rows needed by the sampler."""

    def __init__(self, batch: ScheduledRequests) -> None:
        self.batch = batch
        self.handled = False

    @classmethod
    def for_batch(
        cls,
        heads: Sequence[LMHead],
        batch: ScheduledRequests,
        *,
        is_spec_decode: bool,
    ) -> ContextLogitsPlan | None:
        """Select eager projection only when a prepared, supported head can use it.

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
        if any(
            getattr(request, "py_logits_post_processors", None) for request in batch.all_requests()
        ):
            return None
        if not any(head.context_logits_workspace is not None for head in heads):
            return None
        return cls(batch)

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
        context_rows = sum(request.context_chunk_size for request in contexts)
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
        offset = 0
        tail_rows: list[int] = []
        tail_slots: list[int] = []
        for index, request in enumerate(contexts):
            count = request.context_chunk_size
            if not request.py_return_context_logits:
                tail_rows.append(offset + count - 1)
                tail_slots.append(index)
                offset += count
                continue
            rows = hidden[offset : offset + count]
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
            offset += count

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

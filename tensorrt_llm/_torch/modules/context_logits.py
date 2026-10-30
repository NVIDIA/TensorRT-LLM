# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded eager projection and request-owned storage of context logits."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Iterator

import torch

if TYPE_CHECKING:
    from tensorrt_llm.mapping import Mapping

    from ..pyexecutor.llm_request import LlmRequest
    from ..pyexecutor.scheduler import ScheduledRequests
    from .embedding import LMHead


def context_logits_heads(
    model: torch.nn.Module, mapping: Mapping, *, is_spec_decode: bool
) -> tuple[LMHead, ...]:
    """Find heads whose untransformed context outputs can be consumed eagerly.

    Warmup and execution share this predicate. Only the standard processor opts
    into request-owned output storage; custom processors retain their own path.
    """
    from .embedding import LMHead
    from .logits_processor import LogitsProcessor

    if (
        is_spec_decode
        or mapping.tp_size != 1
        or mapping.cp_size != 1
        or mapping.enable_attention_dp
        or not mapping.is_last_pp_rank()
    ):
        return ()
    heads = []
    for module in model.modules():
        processor = getattr(module, "logits_processor", None)
        head = getattr(module, "lm_head", None)
        if (
            type(processor) is LogitsProcessor
            and processor.allow_context_logits_plan
            and isinstance(head, LMHead)
            and head.supports_context_logits_workspace
        ):
            heads.append(head)
    return tuple(heads)


def store_context_logits(
    request: LlmRequest, logits: torch.Tensor, *, offset: int = 0, reusable: bool = False
) -> None:
    """Consume context rows from either projection path using one result contract.

    ``offset`` is the row offset within the current prefill chunk. Reduction
    runs eagerly, after any model/output transforms. Reusable GPU-to-host
    copies must complete before the projection buffer is overwritten.
    """
    from ...executor.result import compute_prompt_logprobs

    compact = request.py_prompt_logprobs is not None
    if request.py_context_logits_explicit or not compact:
        request.py_result.append_context_logits(logits, reusable=reusable)
    if not compact:
        return
    begin = request.context_current_position + offset + 1
    end = min(request.py_prompt_len, begin + logits.shape[0])
    targets = request.get_tokens(0)[begin:end]
    values = compute_prompt_logprobs(
        request.py_prompt_logprobs,
        logits[: len(targets)],
        targets,
        request.py_prompt_logprobs_simple_format,
    ).prompt
    request.py_result.append_prompt_logprobs(values or [])
    if offset + logits.shape[0] == request.context_chunk_size and request.is_last_context_chunk:
        request.py_result.set_prompt_last_logits(logits[-1:].cpu())


class ContextLogitsWorkspace:
    """Resident output buffers for a dense, unsharded LM head.

    The output and FP32 conversion storage survive optional allocation failure.
    They are initialized during decoder warmup, before KV capacity estimation.
    """

    MIN_ROWS = 32
    MAX_ALLOCATION_ATTEMPTS = 4

    def __init__(self, head: LMHead) -> None:
        self.dtype = head.output_dtype or head.weight.dtype
        self.device = head.weight.device
        self.vocab_size = head.num_embeddings
        self.resident = self._allocate(self.MIN_ROWS)
        # Initialize the GEMM backend before requests can consume GPU headroom.
        hidden = torch.zeros(
            (self.MIN_ROWS, head.weight.shape[1]), dtype=head.weight.dtype, device=self.device
        )
        with torch.inference_mode():
            self.project(head, hidden, self.resident)

    def _allocate(self, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        output = torch.empty((rows, self.vocab_size), dtype=self.dtype, device=self.device)
        converted = (
            output
            if self.dtype == torch.float32
            else torch.empty((rows, self.vocab_size), dtype=torch.float32, device=self.device)
        )
        return output, converted

    def acquire(self, rows: int, *, host_storage: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Try larger output buffers without releasing the resident fallback."""
        if host_storage:
            return self.resident
        for _ in range(self.MAX_ALLOCATION_ATTEMPTS):
            if rows <= self.MIN_ROWS:
                break
            try:
                return self._allocate(rows)
            except torch.OutOfMemoryError:
                rows = max(self.MIN_ROWS, rows // 2)
        return self.resident

    def project(
        self, head: LMHead, hidden: torch.Tensor, buffers: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Write [rows, vocab] logits into caller-owned reusable storage."""
        rows = hidden.shape[0]
        output, converted = (buffer[:rows] for buffer in buffers)
        kwargs = {} if self.dtype == head.weight.dtype else {"out_dtype": self.dtype}
        torch.mm(hidden, head.weight.t(), out=output, **kwargs)
        if converted is not output and self.dtype != torch.float32:
            converted.copy_(output)
        return converted


class ContextLogitsPlan:
    """Consume context chunks directly; return only rows needed by the sampler."""

    def __init__(self, batch: ScheduledRequests) -> None:
        self.batch = batch
        self.handled = False

    @classmethod
    def for_batch(
        cls,
        model: torch.nn.Module,
        mapping: Mapping,
        batch: ScheduledRequests,
        *,
        is_spec_decode: bool,
    ) -> ContextLogitsPlan | None:
        """Select eager projection only when a prepared, supported head can use it."""
        if not any(request.py_return_context_logits for request in batch.context_requests):
            return None
        if any(
            getattr(request, "py_logits_post_processors", None) for request in batch.all_requests()
        ):
            return None
        heads = context_logits_heads(model, mapping, is_spec_decode=is_spec_decode)
        if not any(head.context_logits_workspace is not None for head in heads):
            return None
        return cls(batch)

    def project(self, hidden: torch.Tensor, head: LMHead) -> torch.Tensor:
        workspace = head.context_logits_workspace
        assert workspace is not None
        contexts = self.batch.context_requests
        context_rows = sum(request.context_chunk_size for request in contexts)
        generation_rows = hidden.shape[0] - context_rows
        if generation_rows < 0:
            raise ValueError("Context logits input does not match scheduled token rows")
        # Own the sampling rows before result storage can consume the remaining
        # headroom. No view of a reusable projection buffer escapes this call.
        sampling = torch.empty(
            (len(contexts) + generation_rows, head.num_embeddings),
            dtype=torch.float32,
            device=hidden.device,
        )
        for request in contexts:
            if request.py_return_context_logits and (
                request.py_context_logits_explicit or request.py_prompt_logprobs is None
            ):
                request.py_result.prepare_context_logits(head.num_embeddings, hidden.device)

        offset = 0
        for index, request in enumerate(contexts):
            count = request.context_chunk_size
            rows = hidden[offset : offset + count]
            wants_context = request.py_return_context_logits
            compact = wants_context and request.py_prompt_logprobs is not None
            keep_logits = wants_context and (request.py_context_logits_explicit or not compact)
            host_storage = keep_logits and request.py_result.context_logits_on_cpu
            if not wants_context:
                rows = rows[-1:]
            buffers = workspace.acquire(rows.shape[0], host_storage=host_storage)
            capacity = buffers[0].shape[0]
            for begin in range(0, rows.shape[0], capacity):
                end = min(begin + capacity, rows.shape[0])
                logits = workspace.project(head, rows[begin:end], buffers)
                if wants_context:
                    store_context_logits(request, logits, offset=begin, reusable=True)
                if end == rows.shape[0]:
                    sampling[index].copy_(logits[-1])
            del logits, buffers
            offset += count

        if generation_rows:
            buffers = workspace.acquire(generation_rows, host_storage=False)
            capacity = buffers[0].shape[0]
            for begin in range(0, generation_rows, capacity):
                end = min(begin + capacity, generation_rows)
                logits = workspace.project(head, hidden[offset + begin : offset + end], buffers)
                sampling[len(contexts) + begin : len(contexts) + end].copy_(logits)
        self.handled = True
        return sampling


_ACTIVE_PLAN: ContextVar[ContextLogitsPlan | None] = ContextVar("context_logits_plan", default=None)


@contextmanager
def context_logits_scope(plan: ContextLogitsPlan | None) -> Iterator[None]:
    token = _ACTIVE_PLAN.set(plan)
    try:
        yield
    finally:
        _ACTIVE_PLAN.reset(token)


@torch.compiler.disable
def project_context_logits(hidden: torch.Tensor, head: LMHead) -> torch.Tensor | None:
    """Use the executor's eager context-output plan when a supported head is prepared."""
    plan = _ACTIVE_PLAN.get()
    if plan is None or head.context_logits_workspace is None:
        return None
    return plan.project(hidden, head)

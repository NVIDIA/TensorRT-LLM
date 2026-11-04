# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded eager projection of context logits through the LM head.

The executor owns request and result bookkeeping
(``pyexecutor.context_logits_plan``); this module only provides the projection
buffers and the hook that lets an active plan replace the LM-head output.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Iterator, Protocol

import torch

if TYPE_CHECKING:
    from tensorrt_llm.mapping import Mapping

    from .embedding import LMHead


class ContextLogitsProjector(Protocol):
    """An executor plan that consumes context rows during the LM-head projection."""

    def project(self, hidden: torch.Tensor, head: LMHead) -> torch.Tensor: ...


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


class ContextLogitsWorkspace:
    """Resident projection, TP gather and FP32 conversion buffers.

    Warmup initializes the buffers and collectives before KV capacity estimation.
    Every TP rank agrees on capacity before entering a chunk's all-gather.
    """

    MIN_ROWS = 32
    MAX_ALLOCATION_ATTEMPTS = 4

    def __init__(self, head: LMHead) -> None:
        self.dtype = head.output_dtype or head.weight.dtype
        self.device = head.weight.device
        self.vocab_size = head.num_embeddings
        self.local_vocab_size = head.weight.shape[0]
        self.tp_size = head.tp_size
        self.mapping = head.mapping
        if self.tp_size > 1:
            self._vote = torch.empty(1, dtype=torch.int64, device=self.device)
            self._votes = torch.empty(self.tp_size, dtype=torch.int64, device=self.device)
        resident = hidden = None
        try:
            resident = self._allocate(self.MIN_ROWS)
            hidden = torch.zeros(
                (self.MIN_ROWS, head.weight.shape[1]), dtype=head.weight.dtype, device=self.device
            )
        except torch.OutOfMemoryError:
            pass
        self.require_all(resident is not None and hidden is not None, "context-logits warmup")
        assert resident is not None and hidden is not None
        self.resident = resident
        # Initialize GEMM and collective backends before serving consumes headroom.
        with torch.inference_mode():
            self.project(head, hidden, self.resident)

    def _allgather(self, source: torch.Tensor, destination: torch.Tensor) -> None:
        from tensorrt_llm._utils import mpi_disabled

        group_name = self.mapping.tp_group_name if mpi_disabled() else None
        torch.ops.trtllm.allgather_into_tensor(
            source, destination, self.mapping.tp_group, group_name
        )

    def _min_capacity(self, capacity: int) -> int:
        if self.tp_size == 1:
            return capacity
        self._vote.fill_(capacity)
        self._allgather(self._vote, self._votes)
        return min(self._votes.cpu().tolist())

    def require_all(self, succeeded: bool, operation: str) -> None:
        """Raise on every TP rank if any rank failed, before the next gather."""
        if not self._min_capacity(int(succeeded)):
            raise torch.OutOfMemoryError(f"Insufficient memory for {operation} on a TP rank")

    def _allocate(self, rows: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local = torch.empty((rows, self.local_vocab_size), dtype=self.dtype, device=self.device)
        gathered = (
            local
            if self.tp_size == 1
            else torch.empty(
                (self.tp_size * rows, self.local_vocab_size), dtype=self.dtype, device=self.device
            )
        )
        converted = (
            local
            if self.tp_size == 1 and self.dtype == torch.float32
            else torch.empty((rows, self.vocab_size), dtype=torch.float32, device=self.device)
        )
        return local, gathered, converted

    def acquire(
        self, rows: int, *, host_storage: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Agree on the smallest successful capacity, including CPU-storage fallback."""
        buffers = self.resident
        if not host_storage:
            for _ in range(self.MAX_ALLOCATION_ATTEMPTS):
                if rows <= self.MIN_ROWS:
                    break
                try:
                    buffers = self._allocate(rows)
                    break
                except torch.OutOfMemoryError:
                    rows = max(self.MIN_ROWS, rows // 2)
        capacity = self._min_capacity(buffers[0].shape[0])
        if capacity == self.MIN_ROWS:
            return self.resident
        local, gathered, converted = buffers
        return local[:capacity], gathered[: self.tp_size * capacity], converted[:capacity]

    def project(
        self,
        head: LMHead,
        hidden: torch.Tensor,
        buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        local_ok: bool = True,
    ) -> torch.Tensor:
        """Gather vocabulary shards into owned scratch without allocating a full output.

        Args:
            head: The LM head that owns this workspace.
            hidden: ``[rows, hidden_size]`` hidden states in the head's weight
                dtype, with ``rows`` at most the buffers' capacity.
            buffers: ``(local, gathered, converted)`` from ``acquire``:
                ``[capacity, local_vocab]`` GEMM output, ``[tp_size * capacity,
                local_vocab]`` gather receive buffer (``local`` when TP is 1)
                and ``[capacity, vocab]`` FP32 result (``local`` when it
                already is FP32 at TP 1).
            local_ok: False reports a local failure since the previous vote,
                so every TP rank raises together.

        Returns:
            ``[rows, vocab]`` FP32 view of ``converted``, overwritten by the
            next call with the same buffers.
        """
        rows = hidden.shape[0]
        local, gathered, converted = buffers
        local = local[:rows]
        converted = converted[:rows]
        kwargs = {} if self.dtype == head.weight.dtype else {"out_dtype": self.dtype}
        succeeded = False
        try:
            torch.mm(hidden, head.weight.t(), out=local, **kwargs)
            succeeded = True
        except torch.OutOfMemoryError:
            pass
        self.require_all(succeeded and local_ok, "context-logits projection or storage")
        if self.tp_size > 1:
            gathered = gathered[: self.tp_size * rows]
            self._allgather(local, gathered)
            shards = gathered.view(self.tp_size, rows, self.local_vocab_size)
            for rank in range(self.tp_size):
                begin = rank * self.local_vocab_size
                end = min(begin + self.local_vocab_size, self.vocab_size)
                if end > begin:
                    converted[:, begin:end].copy_(shards[rank, :, : end - begin])
        elif self.dtype != torch.float32:
            converted.copy_(local)
        return converted


_ACTIVE_PLAN: ContextVar[ContextLogitsProjector | None] = ContextVar(
    "context_logits_plan", default=None
)


@contextmanager
def context_logits_scope(plan: ContextLogitsProjector | None) -> Iterator[None]:
    token = _ACTIVE_PLAN.set(plan)
    try:
        yield
    finally:
        _ACTIVE_PLAN.reset(token)


@torch.compiler.disable
def project_context_logits(hidden: torch.Tensor, head: LMHead) -> torch.Tensor | None:
    """Use the executor's eager context-output plan when a supported head is prepared.

    Args:
        hidden: ``[num_tokens, hidden_size]`` hidden states of every scheduled
            token, context chunks first.

    Returns:
        None without an active plan; otherwise ``[num_contexts +
        generation_rows, vocab]`` FP32 rows for the sampler: the last row of
        each context chunk followed by every generation row.
    """
    plan = _ACTIVE_PLAN.get()
    if plan is None or head.context_logits_workspace is None:
        return None
    return plan.project(hidden, head)

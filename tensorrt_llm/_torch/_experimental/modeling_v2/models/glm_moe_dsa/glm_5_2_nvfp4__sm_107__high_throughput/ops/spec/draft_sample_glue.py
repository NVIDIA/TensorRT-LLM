# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ADP draft head's private gather and fused greedy draft step."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ... import native_ops
from ._draft_exchange import DraftExchangeWorkspace


@dataclass(frozen=True)
class DraftGlueWindow:
    """Unsharded BF16 embedding and norm operands for a contiguous eh_proj input."""

    table: torch.Tensor
    enorm: torch.Tensor
    hnorm: torch.Tensor
    eps: float
    col_lo: int
    col_hi: int


class DraftSampleGlueAdp:
    """Collective plan for a fixed LM-head group and padded rows per rank.

    Every world rank constructs both workspaces in the same eager draft call;
    the communicator split is collective over the world. Each group agrees on
    allocation success. Captured graphs only use previously constructed plans.
    """

    _instances: dict[tuple[Mapping, int, int], DraftSampleGlueAdp | None] = {}

    def __init__(self, mapping: Mapping, segment_rows: int, hidden_size: int) -> None:
        self.mapping = mapping
        self.segment_rows = segment_rows
        workspaces = []
        error = None
        # Attempt both world-collective splits even when the first group allocation fails.
        for size in (
            mapping.tp_size * segment_rows * hidden_size * 2,
            mapping.tp_size * segment_rows * mapping.tp_size * 16,
        ):
            try:
                workspaces.append(DraftExchangeWorkspace.for_mapping(mapping, size))
            except RuntimeError as exc:
                error = error or exc
        if error is not None:
            raise RuntimeError("MTP ADP workspaces unavailable") from error
        self._rows_gather, self._exchange = workspaces

    @classmethod
    def get(
        cls, mapping: Mapping, segment_rows: int, hidden_size: int
    ) -> DraftSampleGlueAdp | None:
        key = (mapping, segment_rows, hidden_size)
        if key in cls._instances:
            return cls._instances[key]
        if torch.cuda.is_current_stream_capturing():
            return None
        if not (
            mapping.enable_attention_dp
            and 2 <= mapping.tp_size <= 32
            and mapping.gpus_per_node % mapping.tp_size == 0
            and not mapping.has_cp()
            and 1 <= segment_rows <= 32
            and hidden_size == 6144
        ):
            return None
        try:
            plan = cls(mapping, segment_rows, hidden_size)
        except RuntimeError as exc:
            logger.warning(f"Rubin MTP ADP head unavailable: {exc}; using the shared draft loop")
            plan = None
        if plan is not None:
            logger.info(
                f"Rubin MTP ADP head armed: {mapping.tp_size} ranks, {segment_rows} rows per rank"
            )
        cls._instances[key] = plan
        return plan

    def gather(self, rows: torch.Tensor) -> torch.Tensor:
        """Gather contiguous BF16 [segment_rows, hidden] in group-rank order."""
        gathered = native_ops.glm_rubin_mtp_head_gather(
            rows.reshape(1, -1),
            self._rows_gather.buffer,
            self._rows_gather.flags,
            self.mapping.tp_size,
            False,
        )
        return gathered.view(self.mapping.tp_size * rows.shape[0], rows.shape[1])

    def run(
        self,
        logits: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_step: int,
        *,
        glue: DraftGlueWindow | None,
        glue_out: torch.Tensor | None,
        hidden: torch.Tensor,
        positions_in: torch.Tensor,
        positions_out: torch.Tensor | None,
        gather_ids: torch.Tensor | None,
        seq_lens: torch.Tensor,
        kv_lens: torch.Tensor,
        num_accepted: torch.Tensor,
        accepted_tokens: torch.Tensor | None = None,
        batch_indices: torch.Tensor | None = None,
        next_new_tokens: torch.Tensor | None = None,
    ) -> None:
        """Sample this rank's rows and update the next draft step in one launch.

        Logits are BF16 [group_size * segment_rows, vocabulary_shard]. Output
        tokens are contiguous int32 [local_requests, draft_length]. The shared
        metadata methods rebuild DSA metadata after the sequence/KV update.
        """
        has_glue = glue_out is not None
        native_ops.glm_rubin_mtp_sample_glue_adp(
            logits,
            self._exchange.buffer,
            self._exchange.flags,
            self.mapping.tp_size,
            self.segment_rows,
            draft_tokens,
            draft_step,
            glue.table if has_glue else None,
            hidden if has_glue else None,
            glue.enorm if has_glue else None,
            glue.hnorm if has_glue else None,
            glue.eps if has_glue else 0.0,
            glue.col_lo if has_glue else 0,
            glue.col_hi if has_glue else 0,
            glue_out,
            positions_in if positions_out is not None else None,
            positions_out,
            gather_ids,
            seq_lens,
            draft_step == 0,
            kv_lens,
            2 if draft_step == 0 else 1,
            num_accepted if draft_step == 0 else None,
            draft_tokens.shape[1],
            accepted_tokens,
            batch_indices,
            num_accepted if next_new_tokens is not None else None,
            next_new_tokens,
        )

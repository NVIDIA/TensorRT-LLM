# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small-batch BF16 sparse MLA with native TP-local query heads."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .kernels import compact_sparse_rows

if TYPE_CHECKING:
    from ...trtllm import TrtllmAttentionMetadata


class GlmKpoolNativeDecode:
    """Use shared attention scratch and backend-owned graph-stable counters.

    GLM selects physical latent row IDs, including holes for unavailable pools.
    TRTLLM-GEN consumes these as sparse indices into a 32-row paged view, with
    each query's valid selections packed before its length. No KV copy is made.
    """

    def __init__(self) -> None:
        self._counter: torch.Tensor | None = None
        self._workspace_size: int | None = None

    def prepare_workspace(self, q: torch.Tensor, metadata: TrtllmAttentionMetadata) -> None:
        from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes

        from ...fmha.flashinfer_trtllm_gen import _get_generation_workspace_layout

        capturing = torch.cuda.is_current_stream_capturing()
        if self._workspace_size is None:
            layout = _get_generation_workspace_layout(q.dtype, 1, 1, 16, 512, 1, 0)
            self._workspace_size = int(layout["trtllm_gen_workspace_size"])
        workspace = metadata.effective_workspace
        if workspace is None:
            raise RuntimeError("glm_kpool native decode requires prepared attention workspace")
        if workspace.numel() * workspace.element_size() < self._workspace_size:
            if capturing:
                raise RuntimeError("glm_kpool native decode workspace must be sized before capture")
            workspace.resize_(
                (self._workspace_size + workspace.element_size() - 1) // workspace.element_size()
            )
        if self._counter is None:
            if capturing:
                raise RuntimeError(
                    "glm_kpool native decode counters must be allocated before capture"
                )
            sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
            size = get_trtllm_gen_multi_ctas_kv_counter_bytes(8, 16, sm_count)
            # TRTLLM-GEN resets its counters after each invocation.
            self._counter = torch.zeros(size, dtype=torch.uint8, device=q.device)

    def __call__(
        self,
        q: torch.Tensor,
        kv_rows: torch.Tensor,
        topk_rows: torch.Tensor,
        scale: float,
        metadata: TrtllmAttentionMetadata,
    ) -> torch.Tensor:
        from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

        packed, lengths, nonempty = compact_sparse_rows(topk_rows)
        output = trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_rows.view(-1, 1, 32, 512),
            workspace_buffer=metadata.effective_workspace.view(torch.uint8),
            qk_nope_head_dim=256,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            block_tables=packed.unsqueeze(1),
            seq_lens=lengths,
            max_seq_len=packed.shape[1],
            sparse_mla_top_k=packed.shape[1],
            bmm1_scale=scale,
            backend="trtllm-gen",
            sparse_mla_top_k_lens=lengths,
            multi_ctas_kv_counter_buffer=self._counter,
        ).squeeze(1)
        # Empty (padded) requests use a dummy selection to keep the kernel's
        # length positive; discard that output, matching FlashMLA's empty row.
        return torch.where(nonempty[:, None, None], output, 0)

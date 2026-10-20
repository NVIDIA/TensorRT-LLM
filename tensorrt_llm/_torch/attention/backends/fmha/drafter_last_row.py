# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Context-phase FMHA of a one-layer Eagle3 drafter that attends only each sequence's last row.

At a context step the one-model Eagle3 loop reads the drafter's first pass at one row per context request, its last
row (``last_tokens_idx``), so the attention outputs of all other context rows are dead. Their K/V are not: the stock
context preprocess (RoPE, quantization and the KV-cache append) still runs for every row, so the drafter KV cache is
the stock one. Only the attention itself runs with one query per sequence, at the end of that sequence's KV, on the
trtllm-gen split-KV decode kernel (one query row per sequence attends to its whole KV, as a decode row does), or on
the context kernel at one query row when _DECODE_KERNEL is False. The other output rows are zeros, which keeps them
finite through the rest of the drafter layer.
"""

import dataclasses
import math
import os
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.logger import logger

from .flashinfer_trtllm_gen import FlashInferTrtllmGenFmha, _get_bmm1_scale_log2
from .interface import FmhaPhase
from .phased import FmhaParams

# The last rows run on the split-KV decode kernel (the context kernel gives one query row a single CTA per head group,
# so a long KV is read by a handful of CTAs). Kill: TRTLLM_CTX_DRAFTER_LAST_ROW_DECODE=0 in the deployment env (read
# once at import) runs them on the context kernel.
_DECODE_KERNEL = os.environ.get("TRTLLM_CTX_DRAFTER_LAST_ROW_DECODE", "1") == "1"

# Before the last rows attend, the K and V rows at and past each sequence's kv_len on its partial last page are zeroed:
# both kernels read that page whole, and a NaN code there turns the masked 0 * NaN into NaN output. Those slots hold no
# token of the sequence yet and are rewritten when it grows into them. Only on a KV cache whose partial last pages are
# private (see _v_tail_zero_safe). Kill: TRTLLM_CTX_DRAFTER_KV_TAIL_ZERO=0 in the deployment env (read once at import).
_KV_TAIL_ZERO = os.environ.get("TRTLLM_CTX_DRAFTER_KV_TAIL_ZERO", "1") == "1"

# Calls at which the taken line is logged.
_LOG_AT_CALLS = frozenset(10**i for i in range(7))

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.interface import AttentionForwardArgs
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


def _bmm1_scale_log2(bmm1_scale) -> Optional[torch.Tensor]:
    """The decode kernel's log2 softmax scale for a context-preprocess ``bmm1_scale``: its second entry when it carries
    one, else ``bmm1_scale * log2(e)`` rounded once to FP32 (as the trtllm-gen launcher derives it)."""
    if not isinstance(bmm1_scale, torch.Tensor):
        return None
    if bmm1_scale.numel() >= 2:
        return _get_bmm1_scale_log2(bmm1_scale)
    return (bmm1_scale[:1].double() * math.log2(math.e)).float()


class DrafterLastRowContextFmha(FlashInferTrtllmGenFmha):
    """The FlashInfer trtllm-gen context path with the attention run on the last row of each sequence only.

    Serves the context phase alone, so a mixed step composes it with the stock generation library through
    ``CombinedFmha``. Install it only on the attention of a drafter whose context outputs are read at the last row
    of each sequence and nowhere else.
    """

    def __init__(self, attn: "TrtllmAttention"):
        super().__init__(attn)
        # Separate log-once state per outcome, so a warmup "taken" does not hide a later "not taken"; the taken
        # count is logged at 1, 10, 100, ... calls so a log proves engagement past warmup.
        self._logged_not_taken = False
        self._num_taken = 0
        self._kv_tail_safe: Optional[bool] = None

    def _kv_tail_zero(self, meta: "TrtllmAttentionMetadata") -> bool:
        """Whether this step zeroes the partial last pages' tails (``_KV_TAIL_ZERO`` on a private-page cache)."""
        if not _KV_TAIL_ZERO:
            return False
        if self._kv_tail_safe is None:
            self._kv_tail_safe = _v_tail_zero_safe(meta.kv_cache_manager)
            if not self._kv_tail_safe:
                logger.warning(
                    "drafter_ctx_last_row: this KV cache can share a partially reused page without copying it; the "
                    "partial last pages' K/V tails are not zeroed"
                )
        return self._kv_tail_safe

    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        return bool(getattr(attn, _MARK, False))

    def _is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: "AttentionForwardArgs",
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        if phase is not FmhaPhase.CONTEXT:
            return False
        supported, reason = self._is_supported_with_reason(
            q, k, v, self.attn, metadata, forward_args, phase=phase
        )
        if not supported and not self._logged_not_taken:
            self._logged_not_taken = True
            logger.info(
                f"[M3 dispatch] rank {mpi_rank()}: drafter_ctx_last_row not taken: {reason}"
            )
        return supported

    def _run_context_attention(
        self,
        params: FmhaParams,
        q_processed: torch.Tensor,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        kv_scale_pool: Optional[torch.Tensor],
        fmha_workspace: torch.Tensor,
        cu_q_seqlens: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
        max_q_len: int,
        max_kv_len: int,
        window_left: int,
        bmm1_scale,
        bmm2_scale,
        causal: bool,
    ) -> None:
        if params.is_cross or not causal:
            super()._run_context_attention(
                params,
                q_processed,
                kv_pool,
                block_tables,
                kv_scale_pool,
                fmha_workspace,
                cu_q_seqlens,
                cu_kv_seqlens,
                max_q_len,
                max_kv_len,
                window_left,
                bmm1_scale,
                bmm2_scale,
                causal,
            )
            return
        batch = params.batch_size
        if self._kv_tail_zero(params.meta):
            _zero_kv_tails(kv_pool, block_tables, params.sequence_lengths, batch)
            logger.info_once(
                f"[CTX] drafter_kv_tail_zero taken (seqs={batch}, pages={2 * batch})",
                key="minimax_m3_drafter_kv_tail_zero",
            )
        last_rows = cu_q_seqlens[1 : batch + 1].to(torch.long) - 1
        q_last = q_processed.index_select(0, last_rows)
        out_last = params.output.new_empty((batch,) + tuple(params.output.shape[1:]))
        self._num_taken += 1
        if self._num_taken in _LOG_AT_CALLS:
            logger.info(
                f"[M3 dispatch] rank {mpi_rank()}: drafter_ctx_last_row taken at {params.num_tokens} rows, "
                f"{batch} sequences, {'decode' if _DECODE_KERNEL else 'context'} kernel "
                f"(call {self._num_taken})"
            )
        if _DECODE_KERNEL:
            import flashinfer

            flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                query=q_last,
                kv_cache=(kv_pool, kv_pool),
                workspace_buffer=fmha_workspace,
                block_tables=block_tables,
                seq_lens=params.sequence_lengths,
                max_seq_len=max_kv_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                window_left=window_left,
                out=out_last,
                sinks=params.fwd.attention_sinks,
                kv_layout=self._layout,
                enable_pdl=self._enable_pdl,
                backend="trtllm-gen",
                q_len_per_req=1,
                kv_cache_sf=(kv_scale_pool, kv_scale_pool) if kv_scale_pool is not None else None,
                uses_shared_paged_kv_idx=self.USE_SHARED_PAGED_KV_IDX,
                bmm1_scale_log2=_bmm1_scale_log2(bmm1_scale),
                multi_ctas_kv_counter_buffer=self._get_multi_ctas_kv_counter_buffer(),
            )
            params.output.zero_()
            params.output.index_copy_(0, last_rows, out_last)
            return
        params_last = dataclasses.replace(params, output=out_last)
        super()._run_context_attention(
            params_last,
            q_last,
            kv_pool,
            block_tables,
            kv_scale_pool,
            fmha_workspace,
            torch.arange(batch + 1, dtype=cu_q_seqlens.dtype, device=cu_q_seqlens.device),
            cu_kv_seqlens,
            1,
            max_kv_len,
            window_left,
            bmm1_scale,
            bmm2_scale,
            causal,
        )
        params.output.zero_()
        params.output.index_copy_(0, last_rows, out_last)


_MARK = "_drafter_ctx_last_row"


def _v_tail_zero_safe(kv_cache_manager) -> bool:
    """Whether every row's partial last page is its own, so zeroing past kv_len cannot touch another row's tokens.

    True without block reuse or partial reuse, or when a partially matched block is copied on reuse:
    KVCacheManagerV2 always copies it, and KVCacheManager does with copy_on_partial_reuse.
    """
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

    if not getattr(kv_cache_manager, "enable_block_reuse", True) or not getattr(
        kv_cache_manager, "enable_partial_reuse", True
    ):
        return True
    if isinstance(kv_cache_manager, KVCacheManagerV2):
        return True
    return bool(getattr(kv_cache_manager, "copy_on_partial_reuse", False))


def _zero_kv_tails(
    kv_pool: torch.Tensor, block_tables: torch.Tensor, kv_lens: torch.Tensor, batch: int
) -> None:
    """Zero the K and V rows at and past each sequence's kv_len on its partial last page, on the device (no host sync).

    kv_pool is the flat sub-block pool [blocks, kv_heads, page, d]; block_tables[b, 0 / 1, i] the K / V sub-block of
    sequence b's page i; kv_lens the sequences' K/V lengths. A sequence whose last page is full is left as it is.
    """
    page_size = int(kv_pool.shape[2])
    kv = kv_lens[:batch].to(torch.long)
    last = torch.div(kv - 1, page_size, rounding_mode="floor").clamp_min(0)
    blocks = (
        block_tables[:batch].to(torch.long).gather(2, last.view(-1, 1, 1).expand(-1, 2, 1)).view(-1)
    )
    first_free = (kv % page_size).repeat_interleave(2)
    past = (torch.arange(page_size, device=kv_pool.device)[None, :] >= first_free[:, None]) & (
        first_free[:, None] > 0
    )
    pool = kv_pool.view(torch.uint8)
    pages = pool[blocks]
    pages.masked_fill_(past[:, None, :, None], 0)
    pool[blocks] = pages


def mark_drafter_last_row_attention(draft_model: torch.nn.Module) -> int:
    """Make ``DrafterLastRowContextFmha`` available on the attention backend of a one-layer ``draft_model``.

    The FMHA registry lists the library first; it is available only on a backend marked here, so every FmhaManager the
    backend builds (one per quant-config update) has it. A drafter with more layers is left alone: there every
    context row's output feeds the next layer's K/V. Returns the number of backends marked (0 or 1).
    """
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention

    backends = [
        backend
        for backend in (getattr(module, "attn", None) for module in draft_model.modules())
        if isinstance(backend, TrtllmAttention)
    ]
    if len(backends) != 1:
        logger.info(
            f"drafter_ctx_last_row not enabled: the drafter has {len(backends)} attention layers"
        )
        return 0
    backend = backends[0]
    setattr(backend, _MARK, True)
    manager = getattr(backend, "_fmha_manager", None)
    if manager is not None and not any(
        isinstance(lib, DrafterLastRowContextFmha) for lib in manager.fmha_libs
    ):
        from .manager import FmhaManager

        backend._fmha_manager = FmhaManager(backend)
    logger.info(
        "drafter_ctx_last_row enabled ahead of "
        f"{[type(lib).__name__ for lib in backend._fmha_manager.fmha_libs[1:]]}"
    )
    return 1


__all__ = ["DrafterLastRowContextFmha", "mark_drafter_last_row_attention"]

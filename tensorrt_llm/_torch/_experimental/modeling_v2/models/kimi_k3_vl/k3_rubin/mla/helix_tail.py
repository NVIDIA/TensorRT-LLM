# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``TRTLLM_K3_HELIX_TAIL_FUSED=1``: the Kimi-K3 Helix generation tail as one kernel.

In a Helix generation batch whose V up-projection runs before the exchange
(``TRTLLM_HELIX_V_UP_BEFORE_A2A``), :class:`K3HelixTail` replaces the packed split-KV reduction,
the V up-projection BMM, the FIFO all-to-all and the combine by one launch of
``cute_dsl_kernels/blackwell/k3_helix_tail.py``: the decode's FP32 split partials are folded by
the senders (the reduction is deferred with ``DeferredPackedReduce``), every head's rows are
projected and pushed into the head owner's Lamport receive buffer (MNNVL fabric memory), and the
receivers combine them into the ``[T, H / P * 128]`` o_proj input. Decode backends that store
their rows directly (no split partials, the FP8 layers) take the same kernel without the fold.

Sizes and features:

* ``_TAIL_MB``: receive buffer per rank (two halves), 16 MiB; a batch whose records do not fit a
  half takes the unfused path. The kernel runs one CTA per SM: every CTA folds, projects and
  sends, then combines tokens.
* ``features.oproj_pc``: see :data:`OPROJ_PC`.
* ``features.ull2_mla_tail``: the senders stage the task plan before the grid wait, fold any
  number of split tasks with one code path and pad the latent rows for the V up-projection
  (``k3_helix_tail`` ``ull2``; bitwise the default). Read at every call; FP16 split partials
  (``TRTLLM_FP4_MLA_F16_PARTIALS``) always take that path. For a batch of at most one 16-token
  tile the decode before the tail then does not release it early
  (:func:`decode_releases_tail_early`).
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import torch

from tensorrt_llm._torch.distributed.mnnvl_memory import (
    HelixCpMnnvlMemory,
    MnnvlMemory,
    _check_cu_result,
    cuda,
)
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..features import active as _features

_F = _features()

HELIX_TAIL_FUSED = os.environ.get("TRTLLM_K3_HELIX_TAIL_FUSED", "0") == "1"
# TRTLLM_K3_OPROJ_PC=1 (with TRTLLM_K3_HELIX_TAIL_FUSED, the side-stream gate joined before the
# tail): the tail's receivers also store o_proj's operand, the MXFP8 form of bf16(out * gate),
# and o_proj runs the MXFP8 low-M GEMM with the all-reduce push epilogue on it (bitwise the gated
# low-M GEMM + push); the fused attn_res finish (TRTLLM_K3_AR_FINISH_FUSED) stays the consumer.
OPROJ_PC = _F.oproj_pc
# Batches of at most this many tokens (one tile of the tail kernel, one CTA per head) start the
# tail after the decode instead of at the decode's entry. TRTLLM_K3_ULL3_MLA_LATE_TAIL_MAX_TOKENS
# overrides it (read at every call), e.g. 32 so that helix n4 batches also release the tail late
# (with TRTLLM_K3_ULL3_MLA_TAIL_TRIGGER: from the decode's last tile instead of its entry).
_LATE_TAIL_MAX_TOKENS = 16


def ull2_tail_enabled() -> bool:
    return _features().ull2_mla_tail


def decode_releases_tail_early(num_tokens: int) -> bool:
    """Whether the FP4 decode before the fused tail releases it at the decode's entry
    (``DeferredPackedReduce.release_early``, with ``TRTLLM_FP4_MLA_REDUCE_PDL``).

    With ``TRTLLM_K3_ULL2_MLA_TAIL`` and at most :data:`_LATE_TAIL_MAX_TOKENS` tokens it does not:
    a tail grid that is resident during the whole decode folds and projects 2-4 us slower once
    the decode completes than one launched at its end, more than the early launch hides. Larger
    batches launch most of their tail CTAs at the decode's end anyway and gain from the others.
    """
    limit = _features().ull3_mla_late_tail_max_tokens
    return not (ull2_tail_enabled() and num_tokens <= limit)


# Receive buffer per rank in MiB (two halves).
_TAIL_MB = 16


class K3HelixTail:
    """The fused Helix tail over the CP group of ``mapping`` (one receive workspace per mapping)."""

    _cache: Dict[Mapping, "K3HelixTail"] = {}

    def __init__(self, mapping: Mapping):
        from ..kernels.k3_helix_tail import SENTINEL

        self.mapping = mapping
        MnnvlMemory.initialize()
        nbytes = _TAIL_MB << 20
        logger.info(f"Rank {mapping.cp_rank} allocating a {nbytes >> 20} MiB Helix tail buffer")
        self.workspace = HelixCpMnnvlMemory(mapping, nbytes)
        self.ws_addr = self.workspace.ptr
        self.rank_stride = self.workspace.rank_stride
        self.half_bytes = nbytes // 2 // 16 * 16
        # Armed through the driver: the kernel only uses raw addresses (see HelixAllToAllFused).
        _check_cu_result(
            cuda.cuMemsetD8Async(
                cuda.CUdeviceptr(self.ws_addr + mapping.cp_rank * self.rank_stride),
                SENTINEL & 0xFF,
                nbytes,
                cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            )
        )
        self.state = torch.zeros(8, dtype=torch.int64, device="cuda")
        num_sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        self.max_ctas = num_sms
        self.use_pdl = get_env_enable_pdl()
        # Tests on one GPU: this rank's records go to its own buffer (k3_helix_tail loopback).
        self.loopback = False
        torch.cuda.synchronize()
        HelixCpMnnvlMemory.get_comm(mapping).barrier()

    @staticmethod
    def get(mapping: Mapping) -> "K3HelixTail":
        if mapping not in K3HelixTail._cache:
            K3HelixTail._cache[mapping] = K3HelixTail(mapping)
        return K3HelixTail._cache[mapping]

    @staticmethod
    def applies(
        mapping: Mapping, num_tokens: int, num_heads: int, latent_dim: int, v_head_dim: int
    ) -> bool:
        """Whether the fused tail serves a generation batch of ``num_tokens`` tokens."""
        from ..kernels import k3_helix_tail as kernel

        if not HELIX_TAIL_FUSED or torch.compiler.is_compiling():
            return False
        if get_sm_version() not in (100, 103, 107) or num_tokens <= 0:
            return False
        if not mapping.has_cp_helix() or mapping.cp_config.get("use_nccl_for_alltoall", True):
            return False
        if not kernel.can_run(mapping.cp_size, num_heads, latent_dim, v_head_dim):
            return False
        nbytes = _TAIL_MB << 20
        record = kernel.record_bytes(num_heads // mapping.cp_size, v_head_dim)
        return num_tokens * mapping.cp_size * record <= nbytes // 2 // 16 * 16

    def __call__(
        self,
        rows: torch.Tensor,
        stats: torch.Tensor,
        w_uv: torch.Tensor,
        zero_kv_mask: Optional[torch.Tensor],
        fold=None,
        out: Optional[torch.Tensor] = None,
        mx_out=None,
        timestamps: Optional[torch.Tensor] = None,
        mx_atom: bool = False,
    ) -> torch.Tensor:
        """Fold, project, exchange and combine one layer's partials.

        Args:
            rows: BF16 ``[T, H * 512]`` decode output (latent rows ``(token, head)``); with
                ``fold`` only the rows the decode stored directly are read.
            stats: FP32 ``[T, H, 2]`` ``(max, sum)`` of ``rows``; read only without ``fold``.
            w_uv: BF16 ``[H, 128, 512]`` V up-projection of every head.
            zero_kv_mask: Optional bool ``[T]``, True where this rank owns no KV of the token.
            fold: ``k3_helix_tail.FoldInputs`` of the deferred split reduction, or None.
            out: Optional BF16 ``[T, H / P * 128]`` output.
            mx_out: Optional ``(gate, xq, xs)``: also store the MXFP8 form of
                ``bf16(out * gate)``, o_proj's gated operand (see ``k3_helix_tail``).
            timestamps: Optional int64 phase stamps (microbenchmarks, see ``k3_helix_tail``).
            mx_atom: ``xs`` in the mid-M GEMM's scale layout (see ``k3_helix_tail``).

        Returns:
            The combined BF16 ``[T, H / P * 128]`` output of this rank's heads.
        """
        from ..kernels.k3_helix_tail import k3_helix_tail

        num_tokens = rows.shape[0]
        num_heads = w_uv.shape[0]
        cp_size = self.mapping.cp_size
        if out is None:
            out = rows.new_empty((num_tokens, num_heads // cp_size * w_uv.shape[1]))
        if zero_kv_mask is not None:
            zero_kv_mask = zero_kv_mask[:num_tokens].contiguous()
        k3_helix_tail(
            rows.reshape(num_tokens * num_heads, -1),
            stats.reshape(num_tokens * num_heads, 2),
            w_uv,
            out,
            fold=fold,
            zero_kv_mask=zero_kv_mask,
            cp_rank=self.mapping.cp_rank,
            cp_size=cp_size,
            ws_addr=self.ws_addr,
            rank_stride=self.rank_stride,
            half_bytes=self.half_bytes,
            state=self.state,
            max_ctas=self.max_ctas,
            use_pdl=self.use_pdl,
            mx_out=mx_out,
            timestamps=timestamps,
            loopback=self.loopback,
            mx_atom=mx_atom,
            ull2=ull2_tail_enabled(),
        )
        return out


def fold_inputs_of(deferred, rows: torch.Tensor, stats: torch.Tensor):
    """``FoldInputs`` of the reduction a ``DeferredPackedReduce`` recorded for this decode (its
    output must be ``rows`` and its statistics ``stats``), or None if it recorded none. A recorded
    call on other buffers is launched instead (the caller then reads ``rows`` and ``stats``)."""
    from ..kernels.k3_helix_tail import FoldInputs

    args = deferred.args
    if args is None:
        return None
    own_buffers = (
        args["output"].data_ptr() == rows.data_ptr()
        and args["output"].numel() == rows.numel()
        and args["stats"] is not None
        and args["stats"].data_ptr() == stats.data_ptr()
    )
    if not own_buffers:
        deferred.run()
        return None
    return FoldInputs(
        args["partial"],
        args["maxima"],
        args["sums"],
        args["plan"],
        args["q_scale"],
        args["kv_scale"],
        num_requests=args["num_requests"],
        valid_m=args["valid_m"],
        physical_m=args["physical_m"],
        sm_scale=args["sm_scale"],
        output_scale=args["output_scale"],
    )

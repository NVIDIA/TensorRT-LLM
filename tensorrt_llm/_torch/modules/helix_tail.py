# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Helix MLA generation tail as one kernel (``kernels/blackwell/k3_helix_tail.py``).

In a Helix generation batch whose V up-projection runs before the exchange, :class:`HelixTail`
replaces the packed split-KV reduction, the V up-projection BMM, the all-to-all and the combine
by one launch: the decode's FP32 split partials are folded by the senders, every head's rows are
projected and pushed into the head owner's Lamport receive buffer (MNNVL fabric memory), and the
receivers combine them into the ``[T, H / P * 128]`` o_proj input. Decodes that store their rows
directly (no split partials) take the same kernel without the fold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..distributed.mnnvl_memory import HelixCpMnnvlMemory, MnnvlMemory, _check_cu_result, cuda
from ..flashinfer_utils import get_env_enable_pdl

if TYPE_CHECKING:
    from ..attention.backends.fp4_mla.fp4_mla_packed_reduce import PackedSplitPartials

# Receive buffer per rank, in two halves; max_tokens derives the tokens one launch fits.
_RECV_BUFFER_BYTES = 16 << 20
_HALF_BYTES = _RECV_BUFFER_BYTES // 2 // 16 * 16


class HelixTail:
    """The fused Helix tail over the CP group of ``mapping`` and its receive buffer.

    Its owner builds it at construction (a collective allocation) and may share it between
    layers whose launches are ordered on one stream.
    """

    def __init__(self, mapping: Mapping):
        from ..kernels.blackwell.k3_helix_tail import SENTINEL

        self.mapping = mapping
        MnnvlMemory.initialize()
        logger.info(
            f"Rank {mapping.cp_rank} allocating a {_RECV_BUFFER_BYTES >> 20} MiB Helix tail buffer"
        )
        self.workspace = HelixCpMnnvlMemory(mapping, _RECV_BUFFER_BYTES)
        self.ws_addr = self.workspace.ptr
        self.rank_stride = self.workspace.rank_stride
        # Armed through the driver: the kernel only uses raw addresses.
        _check_cu_result(
            cuda.cuMemsetD8Async(
                cuda.CUdeviceptr(self.ws_addr + mapping.cp_rank * self.rank_stride),
                SENTINEL & 0xFF,
                _RECV_BUFFER_BYTES,
                cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            )
        )
        self.state = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.max_ctas = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        self.use_pdl = get_env_enable_pdl()
        torch.cuda.synchronize()
        HelixCpMnnvlMemory.get_comm(mapping).barrier()

    @staticmethod
    def max_tokens(mapping: Mapping, num_heads: int, latent_dim: int, v_head_dim: int) -> int:
        """Largest generation batch (tokens) the fused tail serves for ``num_heads`` decoded heads
        per rank; 0 where it never does: it needs Helix with one attention-TP rank per CP group,
        no attention DP (whose o_proj reduce-scatters) and the MNNVL all-to-all transport."""
        if get_sm_version() not in (100, 103, 107):
            return 0
        if not mapping.has_cp_helix() or mapping.tp_size != 1 or mapping.enable_attention_dp:
            return 0
        if mapping.cp_config.get("use_nccl_for_alltoall", True):
            return 0
        from ..kernels.blackwell import k3_helix_tail as kernel

        if not kernel.can_run(mapping.cp_size, num_heads, latent_dim, v_head_dim):
            return 0
        record = kernel.record_bytes(num_heads // mapping.cp_size, v_head_dim)
        return _HALF_BYTES // (mapping.cp_size * record)

    def __call__(
        self,
        rows: torch.Tensor,
        stats: torch.Tensor,
        w_uv: torch.Tensor,
        zero_kv_mask: Optional[torch.Tensor],
        *,
        fold: Optional[PackedSplitPartials],
        out: torch.Tensor,
        mx_out: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Fold, project, exchange and combine one layer's partials into ``out``.

        Args:
            rows: BF16 ``[T, H * 512]`` decode output (latent rows ``(token, head)``); with
                ``fold`` only the rows the decode stored directly are read.
            stats: FP32 ``[T, H, 2]`` ``(max, sum)`` of ``rows``; read only without ``fold``.
            w_uv: BF16 ``[H, 128, 512]`` V up-projection of every head.
            zero_kv_mask: Optional bool ``[T]``, True where this rank owns no KV of the token.
            fold: The decode's unreduced split state, or None.
            out: BF16 ``[T, H / P * 128]`` output of this rank's heads.
            mx_out: Optional ``(gate, xq, xs)``: also store the MXFP8 form of
                ``bf16(out * gate)`` (see ``k3_helix_tail``).
        """
        from ..kernels.blackwell.k3_helix_tail import k3_helix_tail

        num_tokens = rows.shape[0]
        num_heads = w_uv.shape[0]
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
            cp_size=self.mapping.cp_size,
            ws_addr=self.ws_addr,
            rank_stride=self.rank_stride,
            half_bytes=_HALF_BYTES,
            state=self.state,
            max_ctas=self.max_ctas,
            use_pdl=self.use_pdl,
            mx_out=mx_out,
        )
        return out

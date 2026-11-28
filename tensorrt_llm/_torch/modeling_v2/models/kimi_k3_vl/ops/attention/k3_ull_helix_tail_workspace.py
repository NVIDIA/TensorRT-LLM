# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fused Helix tail's receive workspace of the Kimi K3 SM107 ULL target
(``kimi_k3_nvfp4__sm_107__helix16_moetp16ep1``), from the target's ``mla.py``.

:class:`K3HelixTail` keeps one receive workspace per mapping (``K3HelixTail._cache``); the target's MLA
layers and the drafter's DSpark kernel (``kernels/k3r_spec/dspark_ull2.py``) launch on it.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from tensorrt_llm._torch.distributed.mnnvl_memory import HelixCpMnnvlMemory, MnnvlMemory
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..common.k3_ull_features import active as _features
from ..common.sm107_launch import USE_PDL

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda


def ull2_tail_enabled() -> bool:
    return _features().ull2_mla_tail


def _check_cu_result(cu_func_ret):
    """The values a CUDA driver call returned after its status, raising if it failed."""
    if isinstance(cu_func_ret, tuple):
        cu_result, *others = cu_func_ret
        if cu_result != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(cu_result)
        if len(others) == 1:
            return others[0]
        elif len(others) > 1:
            return tuple(others)
        else:
            return None
    else:
        if cu_func_ret != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(cu_func_ret)
        return None


# Receive buffer per rank in MiB (two halves).
_TAIL_MB = 16


class K3HelixTail:
    """The fused Helix tail over the CP group of ``mapping`` (one receive workspace per mapping)."""

    _cache: Dict[Mapping, "K3HelixTail"] = {}

    def __init__(self, mapping: Mapping):
        from ...kernels.k3r_helix.k3_helix_tail_ull import SENTINEL

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
        self.use_pdl = USE_PDL
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
        from ...kernels.k3r_helix import k3_helix_tail_ull as kernel

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
            mx_atom: (c240) ``xs`` in the mid-M GEMM's scale layout (see ``k3_helix_tail``).

        Returns:
            The combined BF16 ``[T, H / P * 128]`` output of this rank's heads.
        """
        from .k3_helix_tail_ull import k3_helix_tail

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

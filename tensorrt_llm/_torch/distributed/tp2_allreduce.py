# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workspace and launcher of the TP2 large-message all-reduce + residual RMSNorm (+ FP8) kernel
(``kernels/blackwell/tp2_allreduce_norm.py``), one per TP2 mapping.

``AllReduce`` builds it (one per mapping, ``AllReduce.tp2_fused_norm_kernels``) when the model declares
``allreduce_tp2_fused_norm``, the strategy is AUTO or TWOSHOT and the TP2 group has no MNNVL all-reduce;
``AllReduce.forward`` routes the residual RMSNorm fusions of BF16 calls of ``MIN_TOKENS`` to ``max_tokens`` tokens
here. The workspace (each rank's two landing buffers of max_tokens / 2 rows and its flags: max_tokens x hidden x 2
bytes, e.g. 128 MiB for 16384 tokens at hidden 4096) is peer memory without a multicast object
(``ll_exchange._PeerBuffers``: IPC inside a node, unicast MNNVL memory across nodes; the kernel reads only the two
unicast mappings), allocated when the first AllReduce of the mapping is built, before the KV cache is sized and any
CUDA graph is captured. Built only under MPI (``AllReduce`` skips it with MPI disabled).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from tensorrt_llm.functional import AllReduceFusionOp
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from .ll_exchange import _PeerBuffers
from .mnnvl_memory import TpMnnvlMemory
from .ops import MNNVLAllReduce

# Fusion op -> (quantized output, normalized output): the fusions of the CuTe DSL norm kernels.
_OUTPUTS = MNNVLAllReduce._FUSED_NORM_OUTPUTS


class Tp2AllReduceNorm:
    """The TP2 all-reduce + residual RMSNorm of one mapping, for calls of ``MIN_TOKENS`` to ``max_tokens`` tokens.
    Construction is collective and raises RuntimeError on every rank where the workspace cannot be set up."""

    # Smaller calls stay on the library all-reduce, whose one-shot kernel is faster there.
    MIN_TOKENS = 2048

    @classmethod
    def is_supported(cls, mapping: Mapping, hidden: int, max_tokens: int) -> bool:
        """CuTe DSL, a TP2 mapping, calls that reach MIN_TOKENS, and a row the kernel's reduction tree spans."""
        from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

        if not (IS_CUTLASS_DSL_AVAILABLE and mapping.tp_size == 2 and max_tokens >= cls.MIN_TOKENS):
            return False
        from ..kernels.blackwell.tp2_allreduce_norm import Tp2WarpAllReduceNormKernel

        return Tp2WarpAllReduceNormKernel.supports_hidden(hidden)

    def __init__(self, mapping: Mapping, hidden: int, max_tokens: int) -> None:
        self.mapping = mapping
        self.hidden = hidden
        self.max_tokens = max_tokens
        dev = torch.device("cuda", torch.cuda.current_device())
        self.sms = torch.cuda.get_device_properties(dev).multi_processor_count
        self.warps = 8
        half = (max_tokens + 1) // 2
        lbytes = half * hidden * 2
        flag_bytes = 2 * self.sms * self.warps * 4
        size = 2 * lbytes + flag_bytes + 4096
        # The TP pair's communicator, ordered by TP rank (the one the buffers agree over).
        comm = TpMnnvlMemory.get_comm(mapping)
        bufs = None
        error = None
        try:
            # Collective: every rank's buffer is zero (the flags too) before any rank returns; None on every rank
            # when the pair has no peer memory or the fill failed on a rank.
            bufs = _PeerBuffers.create(mapping, size)
            if bufs is None:
                raise RuntimeError("no peer memory")
            # The rest is not collective, so it is agreed on below.
            self.ctl = torch.zeros((4,), dtype=torch.int32, device=dev)  # epoch, arrival counter
            o, q = bufs.local_ptr, bufs.peer_ptrs[1 - mapping.tp_rank]
            self.table = torch.tensor(
                [
                    o,
                    o + lbytes,
                    o + 2 * lbytes,
                    q,
                    q + lbytes,
                    q + 2 * lbytes,
                    self.ctl.data_ptr(),
                    self.ctl.data_ptr() + 4,
                ],
                dtype=torch.int64,
                device=dev,
            )
            self._bufs = bufs
            self._fns = {}
            self._dummy_scale = torch.ones((1,), dtype=torch.float32, device=dev)
            # CuTe views of the operands fixed across calls, built at the first call: the pointer table, the
            # dummy scale, and each layer's norm weight (keyed by its data pointer; the view keeps it alive).
            self._table_view = None
            self._dummy_scale_view = None
            self._norm_weight_views = {}
            torch.cuda.synchronize()
        # Any failure on this rank only (a CUDA error, out of memory, a host allocation) enters the agreement
        # below; an exception escaping before it would leave the peer waiting in the collective.
        except Exception as e:  # noqa: BLE001 - re-raised on every rank below as RuntimeError
            error = e
        # A failure on one rank fails all of them.
        causes = _PeerBuffers.failures(comm, error)
        if causes:
            if bufs is not None:
                bufs.close()
            raise RuntimeError(
                f"TP2 all-reduce workspace setup failed ({'; '.join(causes)}); the library all-reduce serves these "
                "calls"
            ) from error
        logger.info(
            f"[TP2 all-reduce] workspace {size >> 20} MiB for {max_tokens} tokens, H {hidden}"
        )

    def supports(
        self,
        input: torch.Tensor,
        fusion_op: AllReduceFusionOp,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
    ) -> bool:
        """BF16 input, residual and norm weight of the kernel's shapes, for one of its fusions."""
        return (
            fusion_op in _OUTPUTS
            and input.dim() == 2
            and input.dtype == torch.bfloat16
            and input.shape[1] == self.hidden
            and self.MIN_TOKENS <= input.shape[0] <= self.max_tokens
            and input.is_contiguous()
            and residual.dtype == torch.bfloat16
            and residual.shape == input.shape
            and norm_weight.dtype == torch.bfloat16
            and norm_weight.numel() == self.hidden
        )

    def __call__(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        fusion_op: AllReduceFusionOp,
        scale: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        """Outputs of the library op for ``fusion_op``: [norm, residual], [fp8, residual] or [norm, fp8, residual]."""
        import cuda.bindings.driver as cuda
        import cutlass

        from ..cute_dsl_utils import graph_safe_cute_tensor
        from ..flashinfer_utils import get_env_enable_pdl
        from ..kernels.blackwell.tp2_allreduce_norm import Tp2WarpAllReduceNormKernel

        quant, norm_out = _OUTPUTS[fusion_op]
        key = (norm_out, quant, float(eps))
        t = lambda a: graph_safe_cute_tensor(a, dynamic_leading_dim=a.dim() - 1)  # noqa: E731
        residual = residual.contiguous()
        res_out = torch.empty_like(residual)
        norm = torch.empty_like(residual) if norm_out else res_out
        q = (
            torch.empty(residual.shape, dtype=torch.float8_e4m3fn, device=residual.device)
            if quant
            else None
        )
        if self._table_view is None:
            self._table_view = graph_safe_cute_tensor(self.table)
            self._dummy_scale_view = graph_safe_cute_tensor(self._dummy_scale)
        weight_view = self._norm_weight_views.get(norm_weight.data_ptr())
        if weight_view is None:
            weight_view = graph_safe_cute_tensor(norm_weight.reshape(-1))
            self._norm_weight_views[norm_weight.data_ptr()] = weight_view
        sc = scale.reshape(1).float() if quant else self._dummy_scale
        sc_view = graph_safe_cute_tensor(sc) if quant else self._dummy_scale_view
        fn = self._fns.get(key)
        if fn is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "compile the TP2 all-reduce + norm kernel before CUDA graph capture"
                )
            k = Tp2WarpAllReduceNormKernel(
                hidden=self.hidden,
                norm_eps=float(eps),
                ctas=self.sms,
                warps=self.warps,
                norm_out=norm_out,
                quant=quant,
                use_pdl=get_env_enable_pdl(),
            )
            ex = torch.empty((2, self.hidden), dtype=torch.bfloat16, device=input.device)
            exq = torch.empty((2, self.hidden), dtype=torch.uint8, device=input.device)
            fn = self._fns[key] = cutlass.cute.compile(
                k,
                t(ex),
                t(ex),
                graph_safe_cute_tensor(norm_weight.reshape(-1)),
                graph_safe_cute_tensor(sc),
                t(ex),
                t(ex),
                t(exq),
                graph_safe_cute_tensor(self.table),
                cutlass.Int32(0),
                cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            )
        qv = q.view(torch.uint8) if quant else res_out.view(torch.uint8)[:, : self.hidden]
        fn(
            t(input),
            t(residual),
            weight_view,
            sc_view,
            t(res_out),
            t(norm),
            t(qv),
            self._table_view,
            cutlass.Int32(self.mapping.tp_rank),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        if quant:
            return (norm, q, res_out) if norm_out else (q, res_out)
        return norm, res_out

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
"""CuTe DSL kernels ``MNNVLAllReduce`` runs for its residual RMSNorm fusions (kernels in ``kernels/blackwell``).

* ``MnnvlAllReduceNorm``: all-reduce + residual + RMSNorm (+ FP8 output), optionally with the MoE fold
  ``routed + sigmoid(gate) * shared`` computed in-kernel (``routed`` finalized, or finalized in-kernel from
  the fc2 rows). Calls below ``TWO_SHOT_ROWS`` rows run the
  one-shot slot kernel on its own multicast workspace, larger ones the two-shot kernel on the library's Lamport
  workspace and flags. Both produce the library op's outputs.
* ``GemmAllReduceNorm``: FP8 per-tensor row-parallel GEMM (low-M warp MMA) + one-shot all-reduce + residual +
  RMSNorm in one launch, on the library's Lamport workspace.

The all-reduce kernels take the row count at run time and compile per static geometry; the GEMM kernel compiles
per row count (at most ``GemmAllReduceNorm.MAX_ROWS``).
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch

from tensorrt_llm._utils import is_blackwell_datacenter

from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE, graph_safe_cute_tensor
from ..flashinfer_utils import get_env_enable_pdl

# Rows from which a call takes the two-shot kernel.
TWO_SHOT_ROWS = 64


def _ctas_per_token(rows: int, hidden: int, sms: int) -> int:
    """CTAs per token, as the library one-shot's ``adjustGridConfig`` picks them, so the norm's reduction
    tree (and output) match the library bit for bit."""
    threads = hidden // 8
    ctas = 8
    while ctas > 1 and (threads // ctas < 64 or (threads // ctas) % 32 or rows * ctas > sms):
        ctas //= 2
    return ctas


def _tokens_per_cta(rows: int, ctas: int, sms: int) -> int:
    """Tokens per CTA that keep a call within one wave of CTAs (two 512-thread CTAs per SM, which the kernel's
    <= 64 registers allow), so it never runs several back-to-back cross-rank latency chains."""
    if ctas != 1:
        return 1
    wave = 2 * sms
    return 1 if rows <= wave else -(-rows // wave)


def _rows(t: torch.Tensor):
    """CuTe view of ``t`` [M, ...] / [M] with M a runtime value."""
    return graph_safe_cute_tensor(t, dynamic_leading_dim=t.dim() - 1)


@functools.lru_cache(maxsize=None)
def _compiled_two_shot(
    hidden: int,
    eps: float,
    nranks: int,
    ctas: int,
    tpc: int,
    use_pdl: bool,
    se_fold: bool,
    norm_out: bool,
    quant: bool,
    moe_topk: int,
    fc2_row: int,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from ..kernels.blackwell.mnnvl_allreduce_norm import MnnvlAllReduceNormKernel

    kernel = MnnvlAllReduceNormKernel(
        hidden=hidden,
        norm_eps=eps,
        nranks=nranks,
        ctas=ctas,
        use_pdl=use_pdl,
        se_fold=se_fold,
        norm_out=norm_out,
        quant=quant,
        moe_topk=moe_topk,
        fc2_row=fc2_row,
        tpc=tpc,
    )
    x, gate, w, scale, q, table = _example_operands(hidden, nranks)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute.compile(
        kernel,
        _rows(x),
        _rows(x),
        _rows(x),
        _rows(gate),
        _rows(x),
        graph_safe_cute_tensor(w),
        graph_safe_cute_tensor(scale),
        _rows(x),
        _rows(x),
        _rows(q),
        graph_safe_cute_tensor(table),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int32(0),
        stream,
    )


@functools.lru_cache(maxsize=None)
def _compiled_slot(
    hidden: int,
    eps: float,
    nranks: int,
    ctas: int,
    use_pdl: bool,
    se_fold: bool,
    norm_out: bool,
    quant: bool,
    moe_topk: int,
    fc2_row: int,
    region_bytes: int,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from ..kernels.blackwell.mnnvl_slot_allreduce_norm import MnnvlSlotAllReduceNormKernel

    kernel = MnnvlSlotAllReduceNormKernel(
        hidden=hidden,
        norm_eps=eps,
        nranks=nranks,
        ctas=ctas,
        use_pdl=use_pdl,
        se_fold=se_fold,
        norm_out=norm_out,
        quant=quant,
        region_bytes=region_bytes,
        moe_topk=moe_topk,
        fc2_row=fc2_row,
    )
    x, gate, w, scale, q, _ = _example_operands(hidden, nranks)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    z = cutlass.Int64(0)
    return cute.compile(
        kernel,
        _rows(x),
        _rows(x),
        _rows(x),
        _rows(gate),
        _rows(x),
        graph_safe_cute_tensor(w),
        graph_safe_cute_tensor(scale),
        _rows(x),
        _rows(x),
        _rows(q),
        z,
        z,
        z,
        z,
        z,
        z,
        cutlass.Int32(0),
        stream,
    )


def _example_operands(hidden: int, nranks: int):
    """Operands with the compiled layouts (rows are runtime): x [M, H], gate [M], weight [H], scale [1],
    FP8 out [M, H] as uint8, workspace table [3 + W]."""
    dev = torch.device("cuda", torch.cuda.current_device())
    rows = 2
    return (
        torch.empty((rows, hidden), dtype=torch.bfloat16, device=dev),
        torch.empty((rows,), dtype=torch.bfloat16, device=dev),
        torch.empty((hidden,), dtype=torch.bfloat16, device=dev),
        torch.empty((1,), dtype=torch.float32, device=dev),
        torch.empty((rows, hidden), dtype=torch.uint8, device=dev),
        torch.empty((3 + nranks,), dtype=torch.int64, device=dev),
    )


class _LamportTable:
    """Device table (local buffer, multicast buffer, flags, every rank's unicast buffer) of the library's MNNVL
    workspace of ``mapping``, refreshed when the workspace is replaced (only outside CUDA-graph capture)."""

    def __init__(self, mapping, dtype: torch.dtype):
        self.mapping = mapping
        self.dtype = dtype
        self.table = torch.zeros(
            (3 + mapping.tp_size,),
            dtype=torch.int64,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        self._ws = None

    def __call__(self, nbytes: int) -> torch.Tensor:
        from .ops import get_or_scale_allreduce_mnnvl_workspace

        ws = get_or_scale_allreduce_mnnvl_workspace(
            self.mapping, self.dtype, buffer_size_bytes=nbytes
        )
        if ws is not self._ws:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("the MNNVL workspace changed during CUDA graph capture")
            local = ws["uc_buffer"]
            mcast = ws["handle"].get_mc_buffer(tuple(local.shape), local.dtype, 0)
            peers = [
                ws["handle"].get_uc_buffer(r, (1,), torch.float32, 0)
                for r in range(self.mapping.tp_size)
            ]
            self.table.copy_(
                torch.tensor(
                    [local.data_ptr(), mcast.data_ptr(), ws["buffer_flags"].data_ptr()]
                    + [p.data_ptr() for p in peers],
                    dtype=torch.int64,
                )
            )
            self._ws = ws
        return self.table


class _SlotWorkspace:
    """Multicast workspace of the slot kernel: 3 regions of ``max_rows * ctas`` CTA slot blocks (one slot of
    16-byte vectors per rank each) plus per-CTA participation counters. Collective over the TP group."""

    def __init__(self, mapping, comm, hidden: int, ctas: int, max_rows: int):
        """``comm``: the TP-group communicator of the library's MNNVL workspace."""
        from .ops import (
            _make_mnnvl_mcast_buffer,
            _mnnvl_workspace_all_succeeded,
            use_mnnvl_fabric_handle,
        )

        threads = hidden // 8 // ctas
        max_ctas = max_rows * ctas
        self.region_bytes = max_ctas * mapping.tp_size * threads * 16
        size = 3 * self.region_bytes
        dev = torch.device("cuda", torch.cuda.current_device())
        self.handle = _make_mnnvl_mcast_buffer(
            comm, size, mapping, use_mnnvl_fabric_handle(mapping)
        )
        uc = self.handle.get_uc_buffer(mapping.tp_rank, (size // 4,), torch.float32, 0)
        uc.fill_(-0.0)  # every 32-bit word = 0x80000000 = "not written"
        mc = self.handle.get_mc_buffer((size // 4,), torch.float32, 0)
        self.state = torch.zeros((max_ctas * 32,), dtype=torch.int32, device=dev)
        torch.cuda.synchronize()
        # Every rank has cleared its slots before any rank can write into them.
        if not _mnnvl_workspace_all_succeeded(comm, True):
            raise RuntimeError("slot all-reduce workspace setup failed on at least one rank")
        self._buffers = (uc, mc)
        self.own, self.mcast, self.ctl = uc.data_ptr(), mc.data_ptr(), self.state.data_ptr()


class MnnvlAllReduceNorm:
    """All-reduce + residual + RMSNorm of one TP group and hidden size (see the module docstring)."""

    # The slot kernel's CTAs per token: its workspace geometry is fixed at construction.
    SLOT_CTAS = 8

    @staticmethod
    def is_supported(mapping, dtype: torch.dtype, hidden_size: int) -> bool:
        """BF16, TP 2 / 4 / 8 / 16 without attention DP, hidden sizes the slot kernel's 8 CTAs per token split
        into whole warps; SM 10.0 / 10.3."""
        return (
            IS_CUTLASS_DSL_AVAILABLE
            and is_blackwell_datacenter()
            and dtype == torch.bfloat16
            and mapping.tp_size in (2, 4, 8, 16)
            and not mapping.enable_attention_dp
            and hidden_size % 2048 == 0
            and hidden_size // 64 <= 1024
        )

    def __init__(self, mapping, comm, hidden_size: int, table: _LamportTable):
        """Collective over the TP group ``comm`` (allocates the slot workspace)."""
        self.mapping = mapping
        self.hidden = hidden_size
        self.table = table
        self._pdl = get_env_enable_pdl()
        props = torch.cuda.get_device_properties(table.table.device)
        self._sms = props.multi_processor_count
        self._smem_bytes = props.shared_memory_per_block_optin
        self.slot = _SlotWorkspace(mapping, comm, hidden_size, self.SLOT_CTAS, TWO_SHOT_ROWS)
        dev = table.table.device
        # Never-read operands of the compiled signature (gate, FP8 out, scale); rows are runtime values.
        # zeros, not empty: models are built under MetaInitMode, which turns torch.empty into meta tensors
        # and materializes only parameters and buffers.
        self._unused_gate = torch.zeros((1,), dtype=torch.bfloat16, device=dev)
        self._unused_quant = torch.zeros((1, hidden_size), dtype=torch.uint8, device=dev)
        self._unused_scale = torch.zeros((1,), dtype=torch.float32, device=dev)

    def supports_rows(self, rows: int) -> bool:
        """Whether one launch takes ``rows`` tokens: below TWO_SHOT_ROWS the slot workspace holds them, above it
        the shared memory of a CTA holds its tokens' residuals."""
        if rows <= 0:
            return False
        if rows < TWO_SHOT_ROWS:
            return True
        tpc = _tokens_per_cta(rows, _ctas_per_token(rows, self.hidden, self._sms), self._sms)
        return tpc == 1 or 1024 + tpc * self.hidden * 2 <= self._smem_bytes

    @staticmethod
    def moe_fold_supported(
        fc2: torch.Tensor, topk_weights: torch.Tensor, perm: torch.Tensor, hidden: int
    ) -> bool:
        """Whether the un-finalized trtllm-gen MoE output can be finalized in-kernel."""
        return (
            fc2.dtype == torch.bfloat16
            and fc2.dim() == 2
            and fc2.stride(1) == 1
            and fc2.shape[1] >= hidden
            and fc2.stride(0) % 8 == 0
            and fc2.data_ptr() % 16 == 0
            and topk_weights.dtype == torch.bfloat16
            and perm.dtype == torch.int32
            and topk_weights.shape == perm.shape
            and perm.dim() == 2
            and 0 < perm.shape[1] <= 16
        )

    def __call__(
        self,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        x: Optional[torch.Tensor] = None,
        moe_fold: Optional[Tuple] = None,
        scale: Optional[torch.Tensor] = None,
        norm_out: bool = True,
    ) -> Tuple[torch.Tensor, ...]:
        """Input ``x`` [M, H] BF16, or ``moe_fold`` = (routed, shared [M, H], gate logits [M] / [M, 1]) giving
        ``routed + sigmoid(gate) * shared``. ``routed`` is the finalized routed output [M, H] BF16, or the
        un-finalized (fc2 [P, >= H] BF16, topk_weights [M, K] BF16, expanded_idx_to_permuted_idx [M, K] int32),
        finalized in-kernel.

        Returns ``(norm, residual_out)``, ``(norm, fp8, residual_out)`` with ``scale``, or ``(fp8, residual_out)``
        with ``scale`` and ``norm_out=False``, as the library op."""
        import cuda.bindings.driver as cuda
        import cutlass

        hidden = self.hidden
        nranks = self.mapping.tp_size
        m = residual.shape[0]
        topk = fc2_row = 0
        fc2_ptr = topw_ptr = perm_ptr = 0
        if moe_fold is not None:
            routed, shared, gate = moe_fold
            if isinstance(routed, torch.Tensor):
                assert routed.shape == shared.shape and routed.dtype == torch.bfloat16
            else:
                fc2, topw, perm = routed
                assert perm.shape[0] == m and self.moe_fold_supported(fc2, topw, perm, hidden)
                topk, fc2_row = perm.shape[1], fc2.stride(0)
                fc2_ptr, topw_ptr, perm_ptr = (
                    fc2.data_ptr(),
                    topw.contiguous().data_ptr(),
                    perm.contiguous().data_ptr(),
                )
                routed = shared
            xin, shared, gate = routed, shared.contiguous(), gate.reshape(-1)
        else:
            xin, routed, shared, gate = x, x, x, self._unused_gate
        res_out = torch.empty_like(residual)
        norm = torch.empty_like(residual) if norm_out else res_out
        quant = (
            torch.empty(residual.shape, dtype=torch.float8_e4m3fn, device=residual.device)
            if scale is not None
            else None
        )
        operands = (
            _rows(xin.contiguous()),
            _rows(routed.contiguous()),
            _rows(shared),
            _rows(gate.contiguous()),
            _rows(residual.contiguous()),
            graph_safe_cute_tensor(norm_weight.reshape(-1)),
            graph_safe_cute_tensor(scale.reshape(1) if scale is not None else self._unused_scale),
            _rows(res_out),
            _rows(norm),
            _rows(quant.view(torch.uint8) if quant is not None else self._unused_quant),
        )
        folds = moe_fold is not None
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        if m < TWO_SHOT_ROWS:
            slot = self.slot
            fn = _compiled_slot(
                hidden, float(eps), nranks, self.SLOT_CTAS, self._pdl, folds, norm_out,
                scale is not None, topk, fc2_row, slot.region_bytes,
            )  # fmt: skip
            fn(
                *operands,
                cutlass.Int64(slot.own),
                cutlass.Int64(slot.mcast),
                cutlass.Int64(slot.ctl),
                cutlass.Int64(fc2_ptr),
                cutlass.Int64(topw_ptr),
                cutlass.Int64(perm_ptr),
                cutlass.Int32(self.mapping.tp_rank),
                stream,
            )
        else:
            # Two stages of max(scatter, broadcast) each.
            table = self.table(2 * ((m + nranks - 1) // nranks) * nranks * hidden * 2)
            ctas = _ctas_per_token(m, hidden, self._sms)
            fn = _compiled_two_shot(
                hidden, float(eps), nranks, ctas, _tokens_per_cta(m, ctas, self._sms), self._pdl,
                folds, norm_out, scale is not None, topk, fc2_row,
            )  # fmt: skip
            fn(
                *operands,
                graph_safe_cute_tensor(table),
                cutlass.Int64(fc2_ptr),
                cutlass.Int64(topw_ptr),
                cutlass.Int64(perm_ptr),
                cutlass.Int32(self.mapping.tp_rank),
                stream,
            )
        if scale is None:
            return norm, res_out
        if norm_out:
            return norm, quant, res_out
        return quant, res_out


@functools.lru_cache(maxsize=None)
def _compiled_gemm_norm(
    m: int,
    k: int,
    hidden: int,
    k_warps: int,
    n_tiles: int,
    swap: bool,
    fp8_act: str,
    a_stride: int,
    eps: float,
    nranks: int,
    use_pdl: bool,
    min_blocks: int,
    norm_out: bool,
    quant: bool,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from ..kernels.blackwell.gemm_allreduce_norm import GemmAllReduceNormKernel

    kernel = GemmAllReduceNormKernel(
        num_rows=m,
        k_extent=k,
        k_warps=k_warps,
        n_tiles=n_tiles,
        swap_ab=swap,
        fp8_act=fp8_act,
        hidden=hidden,
        norm_eps=eps,
        nranks=nranks,
        use_pdl=use_pdl,
        min_blocks=min_blocks,
        norm_out=norm_out,
        quant=quant,
    )
    dev = torch.device("cuda", torch.cuda.current_device())
    a16 = (
        torch.empty((m, a_stride), dtype=torch.bfloat16, device=dev)[:, :k]
        if fp8_act != "fp8"
        else torch.empty((1, 8), dtype=torch.bfloat16, device=dev)
    )
    a8 = (
        torch.empty((m, a_stride), dtype=torch.uint8, device=dev)[:, :k]
        if fp8_act == "fp8"
        else torch.empty((1, 16), dtype=torch.uint8, device=dev)
    )
    b = torch.empty((hidden, k), dtype=torch.uint8, device=dev)
    s = torch.empty((1,), dtype=torch.float32, device=dev)
    x = torch.empty((m, hidden), dtype=torch.bfloat16, device=dev)
    w = torch.empty((hidden,), dtype=torch.bfloat16, device=dev)
    qo = torch.empty((m, hidden), dtype=torch.uint8, device=dev)
    table = torch.empty((3 + nranks,), dtype=torch.int64, device=dev)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    t = graph_safe_cute_tensor
    return cute.compile(
        kernel,
        t(a16),
        t(a8),
        t(b),
        t(s),
        t(s),
        t(x),
        t(w),
        t(s),
        t(x),
        t(x),
        t(qo),
        t(table),
        cutlass.Int32(0),
        stream,
    )


class GemmAllReduceNorm:
    """Row-parallel FP8 ``x @ W.T`` (low-M warp MMA) + one-shot all-reduce + residual + RMSNorm in one launch."""

    # Rows of one launch: one row per CTA of the row all-reduce.
    MAX_ROWS = 32

    @staticmethod
    def is_supported(mapping, dtype: torch.dtype, hidden_size: int) -> bool:
        """As ``MnnvlAllReduceNorm`` and TP <= 8 (a TP16 projection's K is too short to hide the all-reduce)."""
        return mapping.tp_size <= 8 and MnnvlAllReduceNorm.is_supported(mapping, dtype, hidden_size)

    def __init__(self, mapping, hidden_size: int, table: _LamportTable):
        self.mapping = mapping
        self.hidden = hidden_size
        self.table = table
        self._pdl = get_env_enable_pdl()
        self._sms = torch.cuda.get_device_properties(table.table.device).multi_processor_count
        dev = table.table.device
        # Never-read operands of the compiled signature (rows are compile-time values here); zeros, not
        # empty, as in MnnvlAllReduceNorm (MetaInitMode).
        self._unused_quant = torch.zeros(
            (self.MAX_ROWS, hidden_size), dtype=torch.uint8, device=dev
        )
        self._unused_scale = torch.zeros((1,), dtype=torch.float32, device=dev)
        self._unused_a16 = torch.zeros((1, 8), dtype=torch.bfloat16, device=dev)
        self._unused_a8 = torch.zeros((1, 16), dtype=torch.uint8, device=dev)

    def tactic(self, m: int, k: int, k_warps: int, n_tiles: int, ksplit: int):
        """(k_warps, n_tiles, swap, min_blocks) for ``m`` rows from the GEMM tactic of the projection, or None.

        k_warps is the GEMM's (same FP32 summation order). n_tiles grows until the grid fits one CTA per SM:
        the row all-reduce waits for every CTA's tile, and with at most one CTA per SM (each fits an SM) the
        whole grid is co-resident."""
        hidden = self.hidden
        if not 0 < m <= self.MAX_ROWS or ksplit != 1:
            return None
        while n_tiles < 4 and hidden // (8 * n_tiles) > self._sms:
            n_tiles *= 2
        ctas = hidden // (8 * n_tiles)
        if (
            hidden % (8 * n_tiles)
            or ctas > self._sms
            or m * n_tiles > 32 * k_warps
            or k % (64 * k_warps)
        ):
            return None
        vpt = (hidden // 8) // (32 * k_warps)  # row all-reduce: 16-byte vectors per thread
        if (hidden // 8) % (32 * k_warps) or not 1 <= vpt <= 4 or m > ctas:
            return None
        return k_warps, n_tiles, m <= 8 and n_tiles % 2 == 0, 1

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        input_scale: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        tactic,
        scale: Optional[torch.Tensor] = None,
        norm_out: bool = True,
    ) -> Tuple[torch.Tensor, ...]:
        """x [M, K] BF16 (quantized in registers with ``input_scale``) or E4M3; weight [H, K] E4M3; ``tactic``
        from ``tactic()``. Returns the outputs of ``MnnvlAllReduceNorm.__call__``."""
        import cuda.bindings.driver as cuda
        import cutlass

        m, k = x.shape
        hidden = self.hidden
        fp8_act = "fp8" if x.dtype == torch.float8_e4m3fn else "quant"
        k_warps, n_tiles, swap, min_blocks = tactic
        nranks = self.mapping.tp_size
        table = self.table(m * hidden * nranks * 2)
        fn = _compiled_gemm_norm(
            m, k, hidden, k_warps, n_tiles, swap, fp8_act, x.stride(0), float(eps), nranks, self._pdl,
            min_blocks, norm_out, scale is not None,
        )  # fmt: skip
        res_out = torch.empty_like(residual)
        norm = torch.empty_like(residual) if norm_out else res_out
        quant = (
            torch.empty(residual.shape, dtype=torch.float8_e4m3fn, device=residual.device)
            if scale is not None
            else None
        )
        if fp8_act == "fp8":
            a16, a8 = self._unused_a16, x.view(torch.uint8)
        else:
            a16, a8 = x, self._unused_a8
        t = graph_safe_cute_tensor
        fn(
            t(a16),
            t(a8),
            t(weight.view(torch.uint8)),
            t(weight_scale.reshape(1).float()),
            t(input_scale.reshape(1).float()),
            t(residual.contiguous()),
            t(norm_weight.reshape(-1)),
            t(scale.reshape(1) if scale is not None else self._unused_scale),
            t(res_out),
            t(norm),
            t(quant.view(torch.uint8) if quant is not None else self._unused_quant[:m]),
            t(table),
            cutlass.Int32(self.mapping.tp_rank),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        if scale is None:
            return norm, res_out
        if norm_out:
            return norm, quant, res_out
        return quant, res_out

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
"""Qwen3.5 decode all-reduces on the library's MNNVL workspace (CuTe DSL kernels).

* ``Q35CuteAllReduce``: all-reduce + residual + RMSNorm (+ FP8 out, + shared-expert / MoE-finalize fold),
  bitwise ``mnnvl_fusion_allreduce`` where both are one-shot. It shares the library's workspace, Lamport
  triple buffer and control flags, so its launches may interleave with the library kernel's in one graph.
* ``Q35GemmArNorm``: row-parallel FP8 o_proj / out_proj GEMM + the above in one launch, same workspace.
* ``q35_zero_residual`` / ``q35_mnnvl_fusable``: fold a standalone RMSNorm after a plain all-reduce into a
  RESIDUAL_RMS_NORM all-reduce with a zero residual.

Decode shapes on MNNVL without attention DP only (``get_q35_cute_allreduce`` returns None otherwise).
"""

from __future__ import annotations

import functools
import math
import os
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch

from tensorrt_llm._torch.q35_env import register as _q35_register

if TYPE_CHECKING:
    from .ops import AllReduce

# Rows from which the fused all-reduces go two-shot (tuned for Qwen3.5-397B decode, SM 10.3, TP8 / TP16).
FUSED_AR_TWOSHOT_ROWS = 64
# Larger calls use the library op: above ~2 x SMs rows the CuTe two-shot needs a second wave and loses, unless
# TRTLLM_Q35_FUSED_AR_TPC keeps it one wave; deployments raise the bound with
# TRTLLM_Q35_FUSED_AR_MAX_ROWS (e.g. 512 / 1024 for TEP8 at batch 64 / 128).
FUSED_AR_MAX_ROWS = int(os.environ.get("TRTLLM_Q35_FUSED_AR_MAX_ROWS", "256"))
ZERO_RESIDUAL_MAX_TOKENS = 1024

_zero_residuals: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}


def q35_zero_residual(shape, dtype: torch.dtype, device: torch.device) -> Optional[torch.Tensor]:
    """A shared read-only zero tensor of ``shape`` (<= ``ZERO_RESIDUAL_MAX_TOKENS`` rows), or None.

    As the residual of RESIDUAL_RMS_NORM it yields RMSNorm(allreduce(x)) and allreduce(x) as residual output.
    Allocated only outside CUDA-graph capture (a capture would record the memset without running it); a first
    need inside a capture gets None and must take the unfused path.
    """
    shape = tuple(shape)
    hidden = shape[-1]
    numel = math.prod(shape)
    if numel > ZERO_RESIDUAL_MAX_TOKENS * hidden:
        return None
    key = (torch.device(device), dtype)
    buf = _zero_residuals.get(key)
    if buf is None or buf.numel() < numel:
        if torch.cuda.is_current_stream_capturing():
            return None
        buf = torch.zeros(max(numel, ZERO_RESIDUAL_MAX_TOKENS * hidden), dtype=dtype, device=device)
        _zero_residuals[key] = buf
    return buf[:numel].view(shape)


def q35_mnnvl_fusable(allreduce: Optional["AllReduce"], num_tokens: int) -> bool:
    """Whether ``allreduce`` serves fused calls of ``num_tokens`` rows on MNNVL (the zero-residual folds
    require it, so they never land on the C++ / NCCL fallback)."""
    return (
        allreduce is not None
        and allreduce.mnnvl_allreduce is not None
        and 0 < num_tokens <= ZERO_RESIDUAL_MAX_TOKENS
    )


def _two_shot(rows: int) -> bool:
    return rows >= FUSED_AR_TWOSHOT_ROWS


class _GraphSafeDlpack:
    """DLPack export without a stream sync (safe under CUDA-graph capture)."""

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor.detach()

    def __dlpack__(self, stream=None):
        return self._tensor.__dlpack__(stream=-1)

    def __dlpack_device__(self):
        return self._tensor.__dlpack_device__()


def _cute(t: torch.Tensor):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(_GraphSafeDlpack(t), assumed_align=16)


def _ctas_per_token(rows: int, hidden: int) -> int:
    """CTAs per token, as the library one-shot's ``adjustGridConfig`` picks them, so the norm's reduction
    tree (and output) match the library bit for bit."""
    threads = hidden // 8
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    ctas = 8
    while ctas > 1 and (threads // ctas < 64 or (threads // ctas) % 32 or rows * ctas > sms):
        ctas //= 2
    return ctas


@_q35_register
def _q35_read_switches() -> None:
    """This module's TRTLLM_Q35_* switches; q35_env.activate() re-reads them."""
    global _FUSED_AR_FP8, _FUSED_AR_TPC
    # TRTLLM_Q35_FUSED_AR_FP8=1 (default off): the CuTe kernel's two-shot calls carry E4M3 + per-256-element FP32
    # scales in both stages instead of BF16 (half the NVLink bytes; changes the reduced values, not bitwise the
    # library). It does not change which calls take the CuTe kernel: the TEP8 MTP-6 verify of batches 40-64
    # (280-448 rows) needs FUSED_AR_MAX_ROWS 512.
    _FUSED_AR_FP8 = os.environ.get("TRTLLM_Q35_FUSED_AR_FP8", "0") == "1"
    # TRTLLM_Q35_FUSED_AR_TPC=1 (default off): a BF16 two-shot call larger than one wave of CTAs runs
    # ceil(rows / wave) consecutive tokens per CTA with its phases decoupled across them (q35_allreduce_norm.py,
    # ``tpc``), so it stays one wave instead of two or three back-to-back cross-rank latency chains (TEP8 MTP-6
    # verify: b48-b64 = 336-448 rows, b80-b128 = 560-896 rows); bitwise the one-token kernel. The wave is
    # TRTLLM_Q35_FUSED_AR_WAVE_ROWS rows (default two 512-thread CTAs per SM, which the kernel's <= 64 registers
    # allow: 304 rows on 152 SMs).
    _FUSED_AR_TPC = os.environ.get("TRTLLM_Q35_FUSED_AR_TPC", "0") == "1"


def _tokens_per_cta(rows: int, ctas: int, two_shot: bool, fp8: bool) -> int:
    if not (_FUSED_AR_TPC and two_shot and not fp8 and ctas == 1):
        return 1
    wave = (
        int(os.environ.get("TRTLLM_Q35_FUSED_AR_WAVE_ROWS", "0"))
        or 2 * torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    )
    return 1 if rows <= wave else -(-rows // wave)


@functools.lru_cache(maxsize=None)
def _compiled_cute_ar(
    m: int,
    hidden: int,
    eps: float,
    nranks: int,
    ctas: int,
    use_pdl: bool,
    se_fold: bool,
    norm_out: bool,
    quant: bool,
    moe_topk: int = 0,
    fc2_row: int = 0,
    two_shot: bool = False,
    fp8: bool = False,
    tpc: int = 1,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from ..kernels.blackwell.q35_allreduce_norm import Q35AllReduceNormKernel

    kernel = Q35AllReduceNormKernel(
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
        two_shot=two_shot,
        fp8=fp8,
        tpc=tpc,
    )
    dev = torch.device("cuda")
    x = torch.empty((m, hidden), dtype=torch.bfloat16, device=dev)
    gate = torch.empty((m,), dtype=torch.bfloat16, device=dev)
    w = torch.empty((hidden,), dtype=torch.bfloat16, device=dev)
    scale = torch.empty((1,), dtype=torch.float32, device=dev)
    q = torch.empty((m, hidden), dtype=torch.uint8, device=dev)
    table = torch.empty((3 + nranks,), dtype=torch.int64, device=dev)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute.compile(
        kernel,
        _cute(x),
        _cute(x),
        _cute(x),
        _cute(gate),
        _cute(x),
        _cute(w),
        _cute(scale),
        _cute(x),
        _cute(x),
        _cute(q),
        _cute(table),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int32(0),
        stream,
    )


class Q35CuteAllReduce:
    """CuTe all-reduce + residual + RMSNorm on the ``MNNVLAllReduce`` workspace of ``mapping``'s TP group."""

    def __init__(self, mapping, dtype: torch.dtype = torch.bfloat16):
        from .ops import MNNVLAllReduce

        if dtype != torch.bfloat16 or mapping.tp_size not in (2, 4, 8, 16):
            raise ValueError("Q35CuteAllReduce supports BF16 and TP 2/4/8/16")
        self.mapping = mapping
        self.dtype = dtype
        self.mnnvl = MNNVLAllReduce(mapping, dtype)  # creates / shares the workspace
        self._ws = None
        self._views = None
        # (local buffer, multicast buffer, flags, every rank's unicast buffer)
        self.table = torch.zeros(
            (3 + mapping.tp_size,),
            dtype=torch.int64,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        from ..flashinfer_utils import get_env_enable_pdl

        self._pdl = get_env_enable_pdl()
        self._dummies = {}

    def _dummy(self, m: int, hidden: int):
        """Never-read operands of the compiled signature (gate [m], FP8 out [m, H])."""
        d = self._dummies.get((m, hidden))
        if d is None:
            dev = self.table.device
            d = (
                torch.empty((m,), dtype=torch.bfloat16, device=dev),
                torch.empty((m, hidden), dtype=torch.uint8, device=dev),
                self.table.view(torch.float32)[:1],
            )
            self._dummies[(m, hidden)] = d
        return d

    def _table(self, nbytes: int) -> torch.Tensor:
        from .ops import get_or_scale_allreduce_mnnvl_workspace

        ws = get_or_scale_allreduce_mnnvl_workspace(
            self.mapping, self.dtype, buffer_size_bytes=nbytes
        )
        if ws is None:
            raise RuntimeError(f"the MNNVL workspace (sized once) is too small for {nbytes} B")
        if ws is not self._ws:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("the MNNVL workspace changed during CUDA graph capture")
            local = ws["uc_buffer"]
            mcast = ws["handle"].get_mc_buffer(tuple(local.shape), local.dtype, 0)
            flags = ws["buffer_flags"]
            peers = [
                ws["handle"].get_uc_buffer(r, (1,), torch.float32, 0)
                for r in range(self.mapping.tp_size)
            ]
            self.table.copy_(
                torch.tensor(
                    [local.data_ptr(), mcast.data_ptr(), flags.data_ptr()]
                    + [p.data_ptr() for p in peers],
                    dtype=torch.int64,
                )
            )
            self._ws = ws
            self._views = (local, mcast, flags, peers)
        return self.table

    @staticmethod
    def moe_fold_supported(
        fc2: torch.Tensor, topk_weights: torch.Tensor, perm: torch.Tensor, hidden: int
    ) -> bool:
        """Whether ``shared_fold``'s routed operand can be this un-finalized trtllm-gen MoE output."""
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
        shared_fold: Optional[Tuple] = None,
        scale: Optional[torch.Tensor] = None,
        norm_out: bool = True,
        two_shot: Optional[bool] = None,
        fp8: Optional[bool] = None,
        tpc: Optional[int] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Input ``x`` [M, H] BF16, or ``shared_fold`` = (routed [M, H], shared [M, H], gate [M] / [M, 1])
        giving ``routed + sigmoid(gate) * shared``; ``routed`` may be the un-finalized MoE output
        ``(fc2 [P, >= H] BF16, topk_weights [M, K] BF16, expanded_idx_to_permuted_idx [M, K] int32)``.
        ``fp8``: E4M3 payload of a two-shot call (None = TRTLLM_Q35_FUSED_AR_FP8); ``tpc``: tokens per CTA of a
        BF16 two-shot call (None = ``_tokens_per_cta``).

        Returns ``(norm, residual_out)``, ``(norm, fp8, residual_out)`` with ``scale``, or
        ``(fp8, residual_out)`` with ``scale`` and ``norm_out=False``, as the library op."""
        import cuda.bindings.driver as cuda
        import cutlass

        moe = None
        if shared_fold is not None and isinstance(shared_fold[0], (tuple, list)):
            moe = shared_fold[0]
        src = x if x is not None else shared_fold[1]
        m, hidden = src.shape
        nranks = self.mapping.tp_size
        if two_shot is None:
            two_shot = _two_shot(m)
        fp8 = bool(_FUSED_AR_FP8 if fp8 is None else fp8) and bool(two_shot)
        if fp8:
            from tensorrt_llm.logger import logger

            logger.info_once(
                "TRTLLM_Q35_FUSED_AR_FP8: two-shot all-reduce with an E4M3 payload",
                key="q35_fused_ar_fp8",
            )
        # one-shot: [rank][token][H] in one stage; two-shot: two stages of max(scatter, broadcast) each (BF16 sizes,
        # which also hold the smaller FP8 rows). A two-shot call asks for the two-shot size only: the MNNVL workspace
        # (sized once) is sealed at >= 16 MiB per buffer, which the one-shot size of a call above 256 rows (TP8,
        # H 4096) exceeds while its two-shot size (2 x 8 KiB per row) does not.
        if two_shot:
            table = self._table(2 * ((m + nranks - 1) // nranks) * nranks * hidden * 2)
        else:
            table = self._table(m * hidden * nranks * 2)
        ctas = _ctas_per_token(m, hidden)
        if tpc is None:
            tpc = _tokens_per_cta(m, ctas, bool(two_shot), fp8)
        topk = moe[2].shape[1] if moe is not None else 0
        fc2_row = moe[0].stride(0) if moe is not None else 0
        fn = _compiled_cute_ar(
            m,
            hidden,
            float(eps),
            nranks,
            ctas,
            self._pdl,
            shared_fold is not None,
            norm_out,
            scale is not None,
            topk,
            fc2_row,
            bool(two_shot),
            fp8,
            tpc,
        )
        res_out = torch.empty_like(residual)
        norm = torch.empty_like(residual) if norm_out else res_out
        quant = (
            torch.empty(residual.shape, dtype=torch.float8_e4m3fn, device=residual.device)
            if scale is not None
            else None
        )
        d_gate, d_quant, d_scale = self._dummy(m, hidden)
        fc2_ptr = topw_ptr = perm_ptr = 0
        if shared_fold is not None:
            routed, shared, gate = shared_fold
            if moe is not None:
                fc2, topw, perm = moe
                assert perm.shape[0] == m and self.moe_fold_supported(fc2, topw, perm, hidden)
                fc2_ptr, topw_ptr, perm_ptr = (
                    fc2.data_ptr(),
                    topw.contiguous().data_ptr(),
                    perm.contiguous().data_ptr(),
                )
                routed = shared  # unread
            xin, gate = routed, gate.reshape(-1)
        else:
            xin, routed, shared, gate = x, x, x, d_gate
        w = norm_weight.reshape(-1)
        fn(
            _cute(xin.contiguous()),
            _cute(routed.contiguous()),
            _cute(shared.contiguous()),
            _cute(gate.contiguous()),
            _cute(residual.contiguous()),
            _cute(w),
            _cute(scale.reshape(1) if scale is not None else d_scale),
            _cute(res_out),
            _cute(norm),
            _cute(quant.view(torch.uint8) if quant is not None else d_quant),
            _cute(table),
            cutlass.Int64(fc2_ptr),
            cutlass.Int64(topw_ptr),
            cutlass.Int64(perm_ptr),
            cutlass.Int32(self.mapping.tp_rank),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        if scale is None:
            return norm, res_out
        if norm_out:
            return norm, quant, res_out
        return quant, res_out


_CUTE_INSTANCES = {}


def get_q35_cute_allreduce(mapping) -> Optional[Q35CuteAllReduce]:
    """The group's CuTe all-reduce (lazy), or None with attention DP / without MNNVL.

    Every rank of the group must decide the same way (same mapping, same shapes)."""
    key = (tuple(mapping.tp_group), mapping.tp_rank)
    if key not in _CUTE_INSTANCES:
        from .ops import MNNVLAllReduce

        inst = None
        if (
            mapping.tp_size in (2, 4, 8, 16)
            and not mapping.enable_attention_dp
            and MNNVLAllReduce.is_mnnvl(mapping, torch.bfloat16)
        ):
            inst = Q35CuteAllReduce(mapping)
        _CUTE_INSTANCES[key] = inst
    return _CUTE_INSTANCES[key]


@functools.lru_cache(maxsize=None)
def _compiled_gemm_ar(
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
    two_shot: bool = False,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from ..kernels.blackwell.q35_gemm_ar_norm import Q35GemmArNormKernel

    kernel = Q35GemmArNormKernel(
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
        two_shot=two_shot,
    )
    dev = torch.device("cuda")
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
    return cute.compile(
        kernel,
        _cute(a16),
        _cute(a8),
        _cute(b),
        _cute(s),
        _cute(s),
        _cute(x),
        _cute(w),
        _cute(s),
        _cute(x),
        _cute(x),
        _cute(qo),
        _cute(table),
        cutlass.Int32(0),
        stream,
    )


class Q35GemmArNorm:
    """Row-parallel FP8 ``x @ W.T`` (low-M warp MMA) + all-reduce + residual + RMSNorm in one launch."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.ar = Q35CuteAllReduce(mapping)  # owns the workspace table refresh
        self._sms = torch.cuda.get_device_properties(self.ar.table.device).multi_processor_count

    def tactic(self, m: int, hidden: int, k: int, fp8_act: str):
        """(k_warps, n_tiles, swap, min_blocks), or None when the fused kernel cannot run this shape.

        k_warps is the LowM GEMM's tactic (same FP32 summation order). n_tiles grows until the grid fits one
        CTA per SM: the row all-reduce waits for every CTA's tile, so all CTAs must be co-resident."""
        from ..kernels.blackwell.q35_gemm_ar_norm import MAX_ROWS
        from ..modules.lowm_linear import default_tactic

        if not 0 < m <= MAX_ROWS:
            return None
        k_warps, n_tiles, ksplit, _, _ = default_tactic(m, hidden, k, True, False, fp8_act)
        if ksplit != 1:
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
        # TP16 (K = 512): only the two-shot with >= 256-thread CTAs beats LowM + CuTe all-reduce (~0.7 us).
        if self.mapping.tp_size > 8 and not (_two_shot(m) and k_warps >= 8):
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
        scale: Optional[torch.Tensor] = None,
        norm_out: bool = True,
        tactic=None,
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        """x [M, K] BF16 (quantized in registers with ``input_scale``) or E4M3; weight [H, K] E4M3.

        Returns the outputs of ``Q35CuteAllReduce.__call__``, or None when the shape is not supported."""
        import cuda.bindings.driver as cuda
        import cutlass

        m, k = x.shape
        hidden = weight.shape[0]
        fp8_act = "fp8" if x.dtype == torch.float8_e4m3fn else "quant"
        tac = tactic or self.tactic(m, hidden, k, fp8_act)
        if tac is None:
            return None
        k_warps, n_tiles, swap, min_blocks = tac
        nranks = self.mapping.tp_size
        two_shot = _two_shot(m)
        table = self.ar._table(
            max(m * hidden * nranks * 2, 2 * ((m + nranks - 1) // nranks) * nranks * hidden * 2)
        )
        fn = _compiled_gemm_ar(
            m,
            k,
            hidden,
            k_warps,
            n_tiles,
            swap,
            fp8_act,
            x.stride(0),
            float(eps),
            nranks,
            self.ar._pdl,
            min_blocks,
            norm_out,
            scale is not None,
            two_shot,
        )
        res_out = torch.empty_like(residual)
        norm = torch.empty_like(residual) if norm_out else res_out
        quant = (
            torch.empty(residual.shape, dtype=torch.float8_e4m3fn, device=residual.device)
            if scale is not None
            else None
        )
        d_gate, d_quant, d_scale = self.ar._dummy(m, hidden)
        dev = x.device
        if fp8_act == "fp8":
            a16 = torch.empty((1, 8), dtype=torch.bfloat16, device=dev)
            a8 = x.view(torch.uint8)
        else:
            a16 = x
            a8 = torch.empty((1, 16), dtype=torch.uint8, device=dev)
        fn(
            _cute(a16),
            _cute(a8),
            _cute(weight.view(torch.uint8)),
            _cute(weight_scale.reshape(1).float()),
            _cute(input_scale.reshape(1).float()),
            _cute(residual.contiguous()),
            _cute(norm_weight.reshape(-1)),
            _cute(scale.reshape(1) if scale is not None else d_scale),
            _cute(res_out),
            _cute(norm),
            _cute(quant.view(torch.uint8) if quant is not None else d_quant),
            _cute(table),
            cutlass.Int32(self.mapping.tp_rank),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        if scale is None:
            return norm, res_out
        if norm_out:
            return norm, quant, res_out
        return quant, res_out


_GEMM_AR_INSTANCES = {}


def get_q35_gemm_ar(mapping) -> Optional[Q35GemmArNorm]:
    """The group's fused GEMM all-reduce (lazy; None wherever ``get_q35_cute_allreduce`` is None)."""
    key = (tuple(mapping.tp_group), mapping.tp_rank)
    if key not in _GEMM_AR_INSTANCES:
        inst = None
        if get_q35_cute_allreduce(mapping) is not None:
            inst = Q35GemmArNorm(mapping)
        _GEMM_AR_INSTANCES[key] = inst
    return _GEMM_AR_INSTANCES[key]

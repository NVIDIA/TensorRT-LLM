# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode step's input boundary as one kernel: embedding gather, ``embed_norm``, residual, layer 0's pre-QKV norm.

On a decode step ``DecodeTarget`` runs, before layer 0's QKV projection:

1. ``replicated_embedding_gather`` (``ops/embedding/replicated_gather.py``): one row of the
   replicated table per id, -0.0 written as +0.0, a +0.0 row for an id outside the table;
2. ``embed_norm``: flashinfer's CuTe DSL ``RMSNormKernel`` (``weight_bias = 0``) on the BF16 rows;
3. ``m_emb`` (1.0 for this checkpoint, so no kernel) and the FP32 copy, the residual stream;
4. layer 0's ``pre_qkv_proj_norm`` on ``residual.to(bf16)`` -- the BF16 rows of step 2 again,
   since they are exactly representable -- flashinfer's ``RMSNormKernel`` with
   ``weight_bias = 1`` (gemma).

That is five launches (a gather, two norms, two copies) of 8..128 CTAs each. This kernel does all
of it in one launch of one 128-thread CTA per row: it gathers the row as step 1 does, then runs
flashinfer's ``RMSNormKernel`` body twice on the same register fragment -- the same thread-value
layout (``_make_tv_layout``), the same per-thread ``TensorSSA.reduce``, the same
``row_reduce_sum_multirow`` (imported from flashinfer, not copied), the same ``rsqrt`` and the same
product order ``x * rstd * (w + weight_bias)`` -- and stores the FP32 residual between the two.
So every output is the five-launch chain's, bit for bit, for every table value the
replicated-embedding install admits.

PDL: every thread waits (``griddepcontrol.wait``) before its first global read (the ids are the
step's input, written by the kernels before this one). The served trigger is ``"exit"``,
flashinfer's placement after the last store, so every dependent the replaced chain's last norm
had sees the same launch order. ``"entry"`` releases the dependent at once; it is correct only
when every kernel that may launch behind this one waits before reading its outputs.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass import BFloat16, Float32, Int32, Int64
    from flashinfer.norm.kernels.rmsnorm import RMSNormKernel
    from flashinfer.norm.utils import row_reduce_sum_multirow

    HAS_CUTE_DSL = True
except ImportError:  # pragma: no cover - the served image has the CuTe DSL and flashinfer
    HAS_CUTE_DSL = False

#: Width of the residual stream this kernel is compiled for.
HIDDEN = 8192
#: Where the grid lets its dependent launch.
EXIT, ENTRY = 0, 1
TRIGGERS = {"exit": EXIT, "entry": ENTRY}
_ID_DTYPES = {torch.int32: "int32", torch.int64: "int64"}


if HAS_CUTE_DSL:

    class _InputBoundary:
        """One CTA per row; flashinfer ``RMSNormKernel``'s geometry at ``HIDDEN`` columns."""

        def __init__(self, hidden: int, trigger: int, id_dtype):
            # The geometry flashinfer picks for a BF16 row of this width on this device.
            ref = RMSNormKernel(BFloat16, hidden, 0.0)
            if ref.cluster_n != 1 or ref.rows_per_block != 1 or ref.cols_per_tile != hidden:
                raise ValueError(
                    f"[musespark_open] model_boundary: flashinfer's RMSNormKernel at H={hidden} uses "
                    f"cluster_n={ref.cluster_n}, rows_per_block={ref.rows_per_block}, "
                    f"cols_per_tile={ref.cols_per_tile}; the input boundary is built for one CTA "
                    "per row whose tile is the whole row"
                )
            self.hidden = hidden
            self.trigger = trigger
            self.id_dtype = id_dtype
            self.vec_size = ref.vec_size
            self.copy_bits = ref.copy_bits
            self.threads_per_row = ref.threads_per_row
            self.num_threads = ref.num_threads
            self.rows_per_block = ref.rows_per_block
            self.warps_per_row = ref.warps_per_row
            self.num_vec_blocks = ref.num_vec_blocks
            self.cols_per_tile = ref.cols_per_tile

        @cute.jit
        def __call__(self, mT, mIds, mWe, mWq, mR, mY, eps_e: Float32, eps_q: Float32, stream):
            tv_shape, tv_stride = RMSNormKernel._make_tv_layout(
                self.threads_per_row, self.rows_per_block, self.vec_size, self.num_vec_blocks
            )
            tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
            tiler_mn = (self.rows_per_block, self.cols_per_tile)
            self.kernel(mT, mIds, mWe, mWq, mR, mY, eps_e, eps_q, tv_layout, tiler_mn).launch(
                grid=[mIds.shape[0], 1, 1],
                block=[self.num_threads, 1, 1],
                smem=self.rows_per_block * self.warps_per_row * 4,
                stream=stream,
                use_pdl=True,
            )

        @cute.kernel
        def kernel(
            self,
            mT: cute.Tensor,
            mIds: cute.Tensor,
            mWe: cute.Tensor,
            mWq: cute.Tensor,
            mR: cute.Tensor,
            mY: cute.Tensor,
            eps_e: Float32,
            eps_q: Float32,
            tv_layout: cute.Layout,
            tiler_mn: cute.Shape,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            row, _, _ = cute.arch.block_idx()
            if cutlass.const_expr(self.trigger == ENTRY):
                cute.arch.griddepcontrol_launch_dependents()
            # Before any global read: the ids are written by the kernels before this one.
            cute.arch.griddepcontrol_wait()

            H = self.hidden
            threads_per_row = tv_layout.shape[0][0]
            rows_per_block = tiler_mn[0]
            warps_per_row = max(threads_per_row // 32, 1)

            smem = cutlass.utils.SmemAllocator()
            reduction_buffer = smem.allocate_tensor(
                Float32, cute.make_layout((rows_per_block, warps_per_row)), byte_alignment=4
            )

            # The gather's row choice (ops/embedding/replicated_gather.py): an id outside the
            # table reads nothing.
            token = Int64(mIds[row])
            valid = (token >= Int64(0)) & (token < Int64(mT.shape[0]))
            source = Int64(0)
            if valid:
                source = token

            gX = cute.local_tile(mT, tiler_mn, (source, 0))
            gR = cute.local_tile(mR, tiler_mn, (row, 0))
            gY = cute.local_tile(mY, tiler_mn, (row, 0))
            we_layout = cute.prepend(mWe.layout, cute.make_layout((tiler_mn[0],), stride=(0,)))
            wq_layout = cute.prepend(mWq.layout, cute.make_layout((tiler_mn[0],), stride=(0,)))
            gWe = cute.local_tile(cute.make_tensor(mWe.iterator, we_layout), tiler_mn, (0, 0))
            gWq = cute.local_tile(cute.make_tensor(mWq.iterator, wq_layout), tiler_mn, (0, 0))

            copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=self.copy_bits
            )
            tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler_mn)
            thr_copy = tiled_copy.get_slice(tidx)

            tXgX = thr_copy.partition_S(gX)
            tXrX = cute.make_fragment_like(tXgX)
            tWgWe = thr_copy.partition_S(gWe)
            tWrWe = cute.make_fragment_like(tWgWe)
            tWgWq = thr_copy.partition_S(gWq)
            tWrWq = cute.make_fragment_like(tWgWq)
            tXgR = thr_copy.partition_D(gR)
            tXgY = thr_copy.partition_D(gY)
            tXrY = cute.make_fragment_like(tXgY)
            tXrR = cute.make_fragment_like(tXgY, Float32)
            tXrS = cute.make_fragment_like(tXgY, Float32)

            tXrX.store(cute.zeros_like(tXrX, dtype=BFloat16))
            if valid:
                cute.copy(copy_atom, tXgX, tXrX)
            cute.copy(copy_atom, tWgWe, tWrWe)
            cute.copy(copy_atom, tWgWq, tWrWq)

            # The gather's sign rule: a zero of either sign is +0.0.
            tXrS.store(tXrX.load().to(Float32))
            for i in cutlass.range_constexpr(cute.size(tXrS)):
                if tXrS[i] == Float32(0.0):
                    tXrS[i] = Float32(0.0)

            # embed_norm: flashinfer RMSNormKernel, weight_bias 0.
            x = tXrS.load()
            x_sq = x * x
            sum_sq = row_reduce_sum_multirow(x_sq, threads_per_row, reduction_buffer, None, 1)
            mean_sq = sum_sq / Float32(H)
            rstd = cute.math.rsqrt(mean_sq + eps_e, fastmath=True)
            cute.arch.barrier()
            w = tWrWe.load().to(Float32)
            embeds = (x * rstd * (w + Float32(0.0))).to(BFloat16)

            # The FP32 residual stream (m_emb == 1).
            tXrR.store(embeds.to(Float32))
            cute.autovec_copy(tXrR, tXgR)

            # Layer 0's pre_qkv_proj_norm on residual.to(bf16) == embeds: RMSNormKernel, weight_bias 1.
            x2 = embeds.to(Float32)
            x2_sq = x2 * x2
            sum_sq2 = row_reduce_sum_multirow(x2_sq, threads_per_row, reduction_buffer, None, 1)
            mean_sq2 = sum_sq2 / Float32(H)
            rstd2 = cute.math.rsqrt(mean_sq2 + eps_q, fastmath=True)
            wq = tWrWq.load().to(Float32)
            tXrY.store((x2 * rstd2 * (wq + Float32(1.0))).to(BFloat16))
            cute.copy(copy_atom, tXrY, tXgY)

            if cutlass.const_expr(self.trigger == EXIT):
                cute.arch.griddepcontrol_launch_dependents()

    @functools.lru_cache(maxsize=None)
    def _compile(hidden: int, trigger: int, id_dtype: str, device_index: int):
        ids_type = Int32 if id_dtype == "int32" else Int64
        rows = cute.sym_int64()
        vocab = cute.sym_int64()
        table = cute.runtime.make_fake_compact_tensor(
            BFloat16, (vocab, hidden), stride_order=(1, 0), assumed_align=16
        )
        ids = cute.runtime.make_fake_compact_tensor(ids_type, (rows,), assumed_align=4)
        weight_e = cute.runtime.make_fake_compact_tensor(BFloat16, (hidden,), assumed_align=16)
        weight_q = cute.runtime.make_fake_compact_tensor(BFloat16, (hidden,), assumed_align=16)
        residual = cute.runtime.make_fake_compact_tensor(
            Float32, (rows, hidden), stride_order=(1, 0), assumed_align=16
        )
        out = cute.runtime.make_fake_compact_tensor(
            BFloat16, (rows, hidden), stride_order=(1, 0), assumed_align=16
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        with torch.cuda.device(device_index):
            return cute.compile(
                _InputBoundary(hidden, trigger, ids_type),
                table,
                ids,
                weight_e,
                weight_q,
                residual,
                out,
                Float32(1e-6),
                Float32(1e-6),
                stream,
                options="--enable-tvm-ffi",
            )


def available() -> Optional[str]:
    """Why this kernel cannot run here, or ``None``."""
    return (
        None if HAS_CUTE_DSL else "the CuTe DSL or flashinfer's CuTe DSL RMSNorm is not importable"
    )


def takes(
    table: torch.Tensor, ids: torch.Tensor, embed_weight: torch.Tensor, qkv_weight: torch.Tensor
) -> bool:
    """Whether a call fits the kernel (shapes, dtypes, contiguity; no pointer values)."""
    return (
        table.is_cuda
        and table.dtype == torch.bfloat16
        and table.dim() == 2
        and table.shape[1] == HIDDEN
        and table.is_contiguous()
        and ids.device == table.device
        and ids.dtype in _ID_DTYPES
        and ids.dim() == 1
        and ids.shape[0] > 0
        and ids.is_contiguous()
        and all(
            w.device == table.device
            and w.dtype == torch.bfloat16
            and tuple(w.shape) == (HIDDEN,)
            and w.is_contiguous()
            for w in (embed_weight, qkv_weight)
        )
    )


def input_boundary(
    table: torch.Tensor,
    ids: torch.Tensor,
    embed_weight: torch.Tensor,
    embed_eps: float,
    qkv_weight: torch.Tensor,
    qkv_eps: float,
    trigger: str = "exit",
) -> tuple:
    """``(residual FP32 [rows, H], layer 0's pre-QKV norm output BF16 [rows, H])`` in one launch.

    ``table`` is the whole replicated ``[vocab, H]`` BF16 embedding, ``ids`` the step's token
    ids, ``embed_weight`` ``embed_norm``'s (unit) weight and ``qkv_weight`` layer 0's gemma
    ``pre_qkv_proj_norm`` weight. The caller checks ``takes`` first.
    """
    rows = ids.shape[0]
    residual = torch.empty((rows, HIDDEN), dtype=torch.float32, device=table.device)
    out = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=table.device)
    _compile(HIDDEN, TRIGGERS[trigger], _ID_DTYPES[ids.dtype], table.device.index)(
        table, ids, embed_weight, qkv_weight, residual, out, float(embed_eps), float(qkv_eps)
    )
    return residual, out


def reference(
    table: torch.Tensor,
    ids: torch.Tensor,
    embed_weight: torch.Tensor,
    embed_eps: float,
    qkv_weight: torch.Tensor,
    qkv_eps: float,
) -> tuple:
    """The five-launch chain this kernel replaces, through the served ops."""
    from tensorrt_llm._torch.custom_ops import flashinfer_gemma_rmsnorm, flashinfer_rmsnorm

    from ...ops.embedding.replicated_gather import replicated_embedding_gather

    embeds = replicated_embedding_gather(table, ids)
    embeds = flashinfer_rmsnorm(embeds, embed_weight, embed_eps)
    residual = embeds.to(torch.float32)
    return residual, flashinfer_gemma_rmsnorm(residual.to(torch.bfloat16), qkv_weight, qkv_eps)

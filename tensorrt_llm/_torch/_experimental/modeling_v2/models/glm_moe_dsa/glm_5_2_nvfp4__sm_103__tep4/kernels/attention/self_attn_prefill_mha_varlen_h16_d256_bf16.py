# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.2 TP4 context attention in the NON-absorbed MLA form: causal varlen BF16 MHA, d = 256.

On a dense-causal context step (every request's window <= top-k 2048) the
absorbed chain (K-B q-base BMM -> sparse-MLA FMHA over the 576-wide FP8 latent -> V-B BMM) is
replaced by one expansion GEMM (latent [T, 576] -> K [T, 16, 256] = [W_UK_h c_kv | k_pe] and
V [T, 16, 256] = W_UV_h c_kv) and this kernel: standard causal attention per request with
d_qk = d_v = 256 (GLM-5.2: nope 192 + rope 64, v 256) over packed token-major operands::

    q, o : [T_q, 16, 256] BF16 (contiguous; q = the q_b projection output with rope(q_pe) in place)
    k, v : [T_k, 16, 256] BF16 (contiguous; the expansion GEMM outputs)
    cu_seqlens : int32 [B + 1], max_seqlen : int   (request b's query rows cu[b] .. cu[b+1]-1)
    cu_seqlens_k : int32 [B + 1] (optional)        (request b's K/V rows cu_k[b] .. cu_k[b+1]-1)

A request's K/V rows are its cached prefix (KV block reuse) followed by its new rows, so
``kv_len >= q_len``; query ``i`` attends K/V positions ``<= i + (kv_len - q_len)`` (causal,
bottom-right aligned). Without ``cu_seqlens_k`` (no cached prefix) K/V use ``cu_seqlens``.

Device code: the uniform-precision hd256 prefill FMHA (``_deps/fmha_prefill_d256.py``: 2-CTA
M=256 UMMA, P in TMEM, softmax / correction / MMA / load warps), imported for every helper;
``__call__`` and ``kernel`` are overridden for varlen:

* layouts are token-major with batch stride 0; every work tile reads ``cu_seqlens[b]`` and
  re-bases the Q/K/V TMA coordinate tensors and the O tensor at that row (``domain_offset``) --
  the scheme of the generic ``fmha.py`` varlen path;
* the grid is sized by ``max_seqlen``; a 2-CTA tile pair whose first row is past its request's
  length is skipped by every role (same decision on both CTAs);
* the causal / residual masks and trip counts use the request's own q and kv lengths (the
  base's WINDOW_MASK_INFERENCE: bottom-right alignment, ``offset = kv_len - q_len``); K/V are
  re-based at ``cu_seqlens_k[b]``. With ``cu_seqlens_k == cu_seqlens`` the code path and the
  output bytes are those of the q == kv kernel;
* the epilogue keeps its per-row predicate on request-local row coordinates, so a partial last
  tile never writes the next request's rows. Q/K/V rows past a request's end that a 128-row TMA
  box still loads belong to the next request (or are OOB zero-fill past T): they only feed
  masked scores or rows that are not stored.

Scheduling (``sched="clc"``, default): one 2-CTA cluster is launched per (q-tile pair, head,
request); the first wave fills every SM and the remaining clusters are claimed in launch order by
hardware cluster launch control (``clusterlaunchcontrol.try_cancel``, multicast to both CTAs; the
leader's first idle warp issues the queries, every role consumes the responses through a
``PipelineClcFetchAsync``). Launch order = (head, request) groups in chunks of
``max(4, 256 // m_cta_tiles)`` (about 32 MiB of K/V, L2-resident), inside a chunk the q-tile pairs
from the last (longest causal kv loop) to the first -- longest-processing-time-first list
scheduling. The base persistent round-robin over (m, h, b) (``sched="static"``) hands every pair
two alternating q-tile classes, e.g. 8x987: half the pairs 44 kv steps, the mean 34.6. Tile math
is unchanged, so the output bytes are identical to ``static``.

Conditional rescale (``rescale_threshold``, default 8.0; 0 = the base exact-max schedule):
FA4-style, the softmax keeps a per-row reference max and only moves it when a kv tile raises the
row max by more than the threshold (log2 units, so P <= 2^8 in BF16); the correction warps skip
the O read-scale-write of a warp's 32 rows when none of them moved. Exact after the final
1 / row_sum; not bitwise vs 0 (BF16 rounding of the larger P).

Numerics: FP32 QK and PV accumulation, BF16 P, exp2 softmax with the 1/16 scale folded in
(``scale_softmax_log2``); ``reference`` is the FP32 per-request causal attention.

FP8 variant (``build(fp8=True)``, ``GlmPrefillMhaVarlenD256Fp8``): Q, K, V are E4M3 (the q_b /
expansion GEMMs' E4M3 epilogues; dequantized values ``q8 * s_q``, ``k8 * s_k[h]``, ``v8 * s_v[h]``),
O stays BF16. Both MMAs run in E4M3 with FP32 accumulation; P is E4M3 with the base kernel's 2^8
prescale (``p_prescale_log2``, cancelled by 1 / row_sum). The dequantization folds into two
per-head factors read from ``head_scales`` [2, 16] FP32 at each tile's start (row 0: the softmax
``scale_softmax_log2 = log2(e) / 16 * s_q * s_k[h]``, row 1: the output ``scale_output = s_v[h]``),
so the inner loops are those of the BF16 kernel. The conditional rescale threshold is capped at
``FP8_RESCALE_THRESHOLD_MAX`` (P <= 2^(8 + 0.75) ~ 430 < 448, the E4M3 maximum). CLC schedule only
(the per-head factor is indexed by the CLC tile's head).
MHA epilogue options (FP8 launchers; ``build(..., variant=...)``, kernel symbol suffix ``_sol``; same
output bytes): ``cvt4`` converts P to E4M3 four values per register (one packed
``cvt.rn.satfinite.e4m3x2.f32`` pair, no PRMT re-packing); ``epi_fast`` (MXF8-O launcher) gives the
correction warps 208 registers, reads a whole 128-column O chunk from TMEM, hands it back to the MMA
and then scales / BF16-rounds (packed ``cvt.rn.bf16x2.f32`` on the ALU pipe instead of one XU ``F2F``
per value) / block-scales / stores from registers, the four 32-value sub-chunks interleaved stage by
stage. The served q8 epilogue was latency-bound on the correction warps (~6k cycles per work tile at
the 1.03 GHz base clock) and delays the next tile's per-step O handoff.
Tail-first Q tiling (``tail_first``, kernel symbol suffix ``_tail``; same output bytes): a request's
query rows are tiled shifted back by ``s`` rows -- tile pair t covers rows ``[256 t - s, 256 (t + 1) -
s)`` -- so the partial tile pair is the FIRST one, whose causal kv loop is the shortest, instead of the
last one, which runs the request's whole kv length; every other pair's kv loop also ends up to ``s``
rows earlier. ``s`` is the largest shift ``<= pad = 256 n - q_len`` (n tile pairs) that keeps each
pair's causal diagonal on the 128-row kv-tile grid, ``s = pad - ((pad - c) mod 128)`` for c cached rows
(0 if negative), so a pair crosses two masked kv steps instead of three; ``tail_align=False`` shifts by
``pad`` itself (symbol suffix ``_tailraw``). The tile code runs unchanged in a shifted request frame --
first row ``cu_seqlens[b] - s``, ``q_len + s`` query rows, the same K/V rows -- whose bottom-right
causal alignment is the original one. The ``s`` leading rows (the previous request's rows, or TMA
zero-fill below row 0) only feed rows that are not stored: the epilogue predicate keeps rows
``s <= r < q_len + s``. A stored row sees the same kv tiles with the same masks in the same order as
without the option (the extra fully masked tiles past its diagonal add exact zeros, and a rescale of a
row whose max did not move multiplies by exp2(0) = 1), so the output bytes are the same.
Epilogue offload (``epi_offload``, MXF8-O launchers only, kernel symbol suffix ``_epio``; same output
bytes): at a work tile's end the correction warps only read O from TMEM (each 128-column chunk handed
back to the MMA right after the read), scale it by ``scale_output / row_sum``, round it to BF16 (the
served packed RN cvt) and stage the 128 x 256 BF16 rows in SMEM (67.6 KB, one row per correction
thread, 132-word rows); four added epilogue warps (12-15, one per SM sub-partition) then block-scale /
convert / store the staged rows (the same unpack, 32-value amax, MXF8 code and E4M3 conversion as the
served epilogue) while the correction warps already serve the next tile's per-step O handoffs. In the
served kernel the epilogue ran on the correction warps and stalled the next tile once its two
softmax-statistics stages were full. Measured on one B300: cycles per CTA kv step -8..-10 % at
7-8-request context shapes; per launch after a GEMM (power-capped, the serving regime) -4.6..-5.5 %,
1-request steps unchanged; sustained only -1..-2.5 % because the denser kernel runs at a lower
power-capped clock. Register split (512 threads): softmax 256, correction 152, epilogue 72, MMA /
load / scheduler / idle 32.

Strided K/V (``kv_strided``, kernel symbol suffix ``_kvs``; same output bytes): K and V token rows at
their own row strides (multiples of 256 elements; head stride 256, unit inner stride) instead of
H * 256, e.g. the two halves of the one-launch K | V expansion's ``[T_k, 2 * H * 256]`` output. Only the
K / V TMA descriptors' strides change; the tile code is the same.
Launch: TVM-FFI (``--enable-tvm-ffi``, environment stream); one compiled function serves every
T / B / max_seqlen. Kernel symbol prefix ``glm52_self_attn_prefill_mha``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass._mlir import ir as _mlir_ir
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir.dialects import nvvm as _nvvm
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cute.runtime import from_dlpack, make_fake_stream
from cutlass.cute.typing import Boolean, Float32, Int32, Int64
from cutlass.cutlass_dsl import T as _T
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.memory import SmemAllocator, TmemAllocator
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.tensor_utils import LayoutEnum

from ..common import self_attn_prefill_act_quant_mxf8 as _mxf8
from . import fmha_prefill_d256 as _d256  # noqa: E402

fmha_utils = _d256.fmha_utils
load_qk_uniform = _d256.load_qk_uniform
load_v_uniform = _d256.load_v_uniform
mma_qk_uniform = _d256.mma_qk_uniform

NUM_HEADS = 16
HEAD_DIM = 256
# FP8 P = 2^(p_prescale_log2 + growth) under the conditional rescale: 2^(8 + 0.75) ~ 430 <= 448.
FP8_RESCALE_THRESHOLD_MAX = 0.75
SOFTMAX_SCALE = 1.0 / 16.0  # 1/sqrt(nope 192 + rope 64), the served MLA bmm1 scale
KERNEL_NAME_PREFIX = "glm52_self_attn_prefill_mha"


class GlmPrefillMhaVarlenD256(_d256.FusedMultiHeadAttentionPrefillD256):
    """Causal varlen BF16 hd256 MHA over packed token-major operands (see module doc)."""

    def __init__(
        self,
        sched: str = "clc",
        clc_stages: int = 1,
        clc_budget_div: int = 256,
        rescale_threshold: float = 8.0,
        fp8: bool = False,
        cvt4: bool = False,
        epi_fast: bool = False,
        tail_first: bool = False,
        tail_align: bool = True,
        epi_offload: bool = False,
        kv_strided: bool = False,
    ) -> None:
        """``sched``: ``clc`` (default; see the module doc) or ``static`` (the base persistent
        round-robin over (m, h, b) tiles). ``clc_budget_div`` sizes the CLC order's (h, b)
        chunk: ``max(4, clc_budget_div // m_cta_tiles)`` groups, about
        ``clc_budget_div * 128 KiB`` of K/V. ``rescale_threshold``: conditional rescale (module
        doc); 0 restores the base exact-max schedule. ``cvt4`` / ``epi_fast``: the MHA epilogue
        options (module doc; same output bytes). ``tail_first``: tail-first Q tiling, its shift kept
        on the kv-tile grid unless ``tail_align=False`` (module doc; same output bytes).
        ``epi_offload``: the MXF8-O epilogue on four added warps (module doc; MXF8-O launchers;
        same output bytes);
        ``kv_strided``: K / V rows at their tensors' own row strides (module doc)."""
        super().__init__(
            cutlass.Float32,
            cutlass.Float32,
            True,
            fmha_utils.MaskEnum.WINDOW_MASK_INFERENCE,
        )
        if sched not in ("static", "clc"):
            raise ValueError(f"sched must be 'static' or 'clc', got {sched!r}")
        self.sched = sched
        self.clc_stages = int(clc_stages)
        self.clc_budget_div = int(clc_budget_div)
        self.rescale_threshold = float(rescale_threshold)
        self.fp8 = bool(fp8)
        self.cvt4 = bool(cvt4)
        self.epi_fast = bool(epi_fast)
        self.tail_first = bool(tail_first)
        self.tail_align = bool(tail_align)
        # epi_offload: correction warps stage BF16 O rows in SMEM, warps 12-15 quantize / store
        self.epi_offload = bool(epi_offload)
        self.epi_warp_ids = ()
        # stage row: 256 BF16 (128 words) + 4 words (row state, global row, head, pad); the
        # 4-word pad spreads a warp's 16-byte row accesses over all banks
        self.stage_row_words = 132
        # 512 threads: softmax 4 x 256 + correction 4 x 152 + epilogue 4 x 72 +
        # MMA / load / scheduler / idle 4 x 32 = 16 warps x 128 launch registers
        self.num_regs_correction_offload = 152
        self.num_regs_epi = 72
        if self.epi_offload:
            self.epi_warp_ids = (12, 13, 14, 15)
            self.threads_per_cta += self.threads_per_warp * len(self.epi_warp_ids)
        self.kv_strided = bool(kv_strided)
        # epi_fast holds a 128-column FP32 O chunk per thread: softmax 4 x 256 + correction
        # 4 x 208 + mma / load / sched / empty 4 x 32 <= 12 warps x 168 launch registers
        self.num_regs_correction_epi = 208
        if self.fp8:
            if sched != "clc":
                raise ValueError(
                    "the FP8 variant indexes its per-head scales by the CLC tile's head: "
                    f"sched must be 'clc', got {sched!r}"
                )
            self.rescale_threshold = min(self.rescale_threshold, FP8_RESCALE_THRESHOLD_MAX)
        # the first idle warp issues the CLC queries (leader CTA only)
        self.sched_warp_id = self.empty_warp_ids[0]

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        cu_seqlens: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        max_seqlen: Int32,
        scale_softmax_log2: Float32,
        scale_output: Float32,
        stream: cuda.CUstream,
    ):
        """Packed varlen launch: q/o ``[T_q, H_q, 256]``, k/v ``[T_k, H_k, 256]``, all
        contiguous token-major (token stride H * 256, head stride 256; the launcher checks
        this), ``cu_seqlens`` / ``cu_seqlens_k`` int32 ``[B + 1]`` (query / K-V rows per
        request), causal within each request, bottom-right aligned (``max_seqlen``: q)."""
        self._launch(
            q,
            k,
            v,
            o,
            cu_seqlens,
            cu_seqlens_k,
            max_seqlen,
            scale_softmax_log2,
            scale_output,
            None,
            stream,
            None,
            None,
        )

    @cute.jit
    def _launch(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        cu_seqlens: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        max_seqlen: Int32,
        scale_softmax_log2: Float32,
        scale_output: Float32,
        head_scales: Optional[cute.Tensor],
        stream: cuda.CUstream,
        q8: Optional[cute.Tensor],
        sf8: Optional[cute.Tensor],
    ):
        """``__call__``'s body; ``head_scales`` (FP8 variant) replaces the two scalars per head.
        ``q8`` / ``sf8`` (``GlmPrefillMhaVarlenD256Fp8Q8``): O is also written as the o_proj
        operand in MXF8 (uint8 ``[T_o, H * 256]`` e4m3, rows ``>= T_q`` left to the caller, and
        the blocked e8m0 scales)."""
        self._setup_attributes()
        if cutlass.const_expr(self.epi_offload and q8 is None):
            raise ValueError("epi_offload is an MXF8-O (q8) launcher option")
        t = q.shape[0]
        t_k = k.shape[0]
        h_q = q.shape[1]
        h_k = k.shape[1]
        d = self.cta_tiler[2]
        h_r = h_q // h_k
        b = cu_seqlens.shape[0] - 1
        window_size_left = None
        window_size_right = Int32(0)  # causal
        # (s, d, ((h_r, h_k), b)): packed rows, batch stride 0 (rows re-based per tile)
        q = cute.make_tensor(
            q.iterator,
            cute.make_layout(
                (t, d, ((h_r, h_k), b)),
                stride=(cute.assume(h_q * d, divby=256), 1, ((d, d * h_r), 0)),
            ),
        )
        # K / V token row strides: packed (H_k * d), or the tensors' own (``kv_strided``)
        if cutlass.const_expr(self.kv_strided):
            k_row = cute.assume(k.stride[0], divby=256)
            v_row = cute.assume(v.stride[0], divby=256)
        else:
            k_row = cute.assume(h_k * d, divby=256)
            v_row = cute.assume(h_k * d, divby=256)
        # (s, d, ((h_r, h_k), b)), 0-stride for h_r to broadcast
        k = cute.make_tensor(
            k.iterator,
            cute.make_layout(
                (t_k, d, ((h_r, h_k), b)),
                stride=(k_row, 1, ((0, d), 0)),
            ),
        )
        # (d, s, ((h_r, h_k), b))
        v = cute.make_tensor(
            v.iterator,
            cute.make_layout(
                (d, t_k, ((h_r, h_k), b)),
                stride=(1, v_row, ((0, d), 0)),
            ),
        )
        o = cute.make_tensor(
            o.iterator,
            cute.make_layout(
                (t, d, ((h_r, h_k), b)),
                stride=(cute.assume(h_q * d, divby=256), 1, ((d, d * h_r), 0)),
            ),
        )
        lse = None
        q8_t = None
        if cutlass.const_expr(q8 is not None):
            # the O layout over the e4m3 bytes (the operand's own row count)
            q8_t = cute.make_tensor(
                cute.recast_ptr(q8.iterator, dtype=cutlass.Float8E4M3FN),
                cute.make_layout(
                    (q8.shape[0], d, ((h_r, h_k), b)),
                    stride=(cute.assume(h_q * d, divby=256), 1, ((d, d * h_r), 0)),
                ),
            )

        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.o_dtype = o.element_type
        self.p_dtype = self.q_dtype  # pv should has the same dtype
        # FP8 P pre-scale (fmha.py-style): bias the exp2 exponent so P fills
        # E4M3's [0, 448] range (2^8 = 256 <= 448); the bias cancels in
        # O = P@V / row_sum and only shifts a log2-space LSE by -8.
        self.p_prescale_log2 = 8.0 if self.p_dtype.width == 8 else 0.0
        self.tilePlikeFP32 = self.qk_mma_tiler[1] // Float32.width * self.p_dtype.width

        self.tile_sched_params, grid = fmha_utils.compute_grid(
            cute.shape((max_seqlen, d, ((h_r, h_k), b))),
            self.cta_tiler,
            self.is_persistent,
            is_2cta=True,
        )
        if cutlass.const_expr(self.sched == "clc"):
            # one 2-CTA cluster per (q-tile pair, head, request): (m_cta, H, B) CTAs
            grid = (cute.size(self.tile_sched_params.problem_shape_mhb), 1, 1)

        self.q_major_mode = LayoutEnum.from_tensor(q).mma_major_mode()
        self.k_major_mode = LayoutEnum.from_tensor(k).mma_major_mode()
        self.v_major_mode = LayoutEnum.from_tensor(v).mma_major_mode()
        self.o_layout = LayoutEnum.from_tensor(o)
        cta_group = tcgen05.CtaGroup.TWO
        p_major_mode = OperandMajorMode.K
        p_source = tcgen05.OperandSource.TMEM
        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.qk_mma_tiler[:2],
        )
        pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            p_major_mode,
            self.v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.pv_mma_tiler[:2],
            p_source,
        )
        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )
        self.epi_tile = self.pv_block_tiler[:2]

        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.q_dtype,
            self.q_stage,
        )
        # Uniform precision: K/V are loaded directly into the MMA-B SMEM layout
        # (proper swizzle for the dtype) with no separate dequant/transform buffer.
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.q_dtype,
            self.kv_stage,
        )
        p_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            pv_tiled_mma,
            self.pv_mma_tiler,
            self.p_dtype,
            self.qk_acc_stage,
        )
        p_tmem_layout = cute.select(p_tmem_layout_staged, mode=[0, 1, 2])
        v_smem_layout_staged = sm100_utils.make_smem_layout_b(
            pv_tiled_mma,
            self.pv_mma_tiler,
            self.q_dtype,
            self.kv_stage,
        )

        tma_load_q_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)
        # K/V synchronization workaround: K/V are consumed by
        # the 2-CTA UMMA via PipelineTmaUmma, where only the LEADER CTA arms
        # and waits the transaction barrier. The donor's CtaGroup.ONE op (per-
        # CTA sync, correct for its per-CTA transform-warp consumers) leaves
        # the PEER CTA's K/V arrival unsynchronized with the leader's MMA ->
        # the UMMA can read a stale K/V half. Use the pair-scoped TMA like Q.
        tma_load_kv_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)
        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_q_op,
            q,
            q_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        # TMA load for K
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_kv_op,
            k,
            k_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        # TMA load for V
        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_kv_op,
            v,
            v_smem_layout,
            self.pv_mma_tiler,
            pv_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        self.tma_copy_q_bytes = cute.size_in_bytes(self.q_dtype, q_smem_layout) * cute.size(
            qk_tiled_mma.thr_id.shape
        )
        # Companion to the K/V workaround above: with the pair-scoped K/V TMA,
        # the leader's transaction barrier receives BOTH CTAs' bytes per
        # stage (as Q already does).
        self.tma_copy_kv_bytes = cute.size_in_bytes(self.k_dtype, k_smem_layout) * cute.size(
            qk_tiled_mma.thr_id.shape
        )

        @cute.struct
        class SharedStorage:
            """Static SMEM plan consumed by the kernel via struct offsets:
            per-stage mbarrier pairs for each producer/consumer pipeline,
            the TMEM-allocation holding buffer and dealloc mbarrier, and
            the swizzle-aligned tensor SMEM regions."""

            # Pipeline barriers
            load_q_mbar_ptr: cute.struct.MemRange[Int64, self.q_stage * 2]
            load_kv_mbar_ptr: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mma_s_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
            p_mma_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
            s_corr_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
            sum_mbar_ptr: cute.struct.MemRange[Int64, 2]
            mma_o_mbar_ptr: cute.struct.MemRange[Int64, self.pv_acc_stage * 2]
            tmem_dealloc_mbar: Int64
            tmem_holding_buf: Int32
            # CLC work-claim pipeline (sched == "clc"; tiny and unused otherwise)
            clc_mbar_ptr: cute.struct.MemRange[Int64, self.clc_stages * 2]
            clc_response: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int128, self.clc_stages], 16
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            qk_tiled_mma,
            pv_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            o,
            cu_seqlens,
            cu_seqlens_k,
            lse,
            scale_softmax_log2,
            scale_output,
            head_scales,
            window_size_left,
            window_size_right,
            self.cluster_layout_vmnk,
            q_smem_layout_staged,
            k_smem_layout_staged,
            p_tmem_layout,
            v_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            q8_t,
            sf8,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    # softmax_step / correction_rescale: the base's (fmha_prefill_d256.py) with an optional
    # FA4-style conditional rescale (``rescale_threshold`` > 0); 0 keeps the base behaviour.
    @cute.jit
    def softmax_step(
        self,
        mask_args: Tuple,
        value_args: Tuple,
        tensor_args: Tuple,
        pipeline_args: Tuple,
    ) -> Tuple[Float32, Float32, pipeline.PipelineConsumer, pipeline.PipelineProducer]:
        need_apply_mask, window_size_left, window_size_right = mask_args
        row_max, row_sum, seqlen_q, seqlen_k, scale_softmax_log2 = value_args
        tStS, tScS, tScS_static = tensor_args
        mma_s_consumer, p_mma_producer, s_corr_producer = pipeline_args
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
        s_handle = mma_s_consumer.wait_and_advance()
        tStS_slice = tStS[(None, None), 0, 0, s_handle.index]
        tScS_slice = tScS[(None, None), 0, 0]
        tScS_static_slice = tScS_static[(None, None), 0, 0]
        if cutlass.const_expr(self.use_ldstat):
            # tcgen05.ld.red (>sm_100f): the load itself reduces (MAX) across the
            # loaded elements, delivering one partial max per atom call.
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.LdRed32x32bOp(
                    tcgen05.copy.Repetition(32), redOp=tcgen05.TmemLoadRedOp.MAX
                ),
                self.qk_acc_dtype,
            )
        else:
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), self.qk_acc_dtype
            )
        tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS_slice)
        thr_load = tmem_tiled_load.get_slice(thread_idx)
        tTMEM_LOADtS = thr_load.partition_S(tStS_slice)
        tTMEM_LOADcS = thr_load.partition_D(tScS_slice)
        tTMEM_LOADcS_static = thr_load.partition_D(tScS_static_slice)
        tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcS.shape, self.qk_acc_dtype)
        num_chunks = cute.size(tTMEM_LOADrS, mode=[2])
        old_row_max = row_max
        if need_apply_mask:
            # Masked steps (tile boundary / causal edge): load all of S, mask,
            # then reduce. Minority of steps; keep the simple schedule.
            if cutlass.const_expr(self.use_ldstat):
                # A red atom needs copy_atom_call with a (data, red) tuple
                # destination; the fused max is discarded here since masking
                # changes the values — the max is recomputed post-mask.
                tTMEM_LOADrMax = cute.make_rmem_tensor(
                    cute.make_layout((1, num_chunks)), self.qk_acc_dtype
                )
                for c in cutlass.range_constexpr(num_chunks):
                    cute.copy_atom_call(
                        tmem_tiled_load,
                        tTMEM_LOADtS[None, 0, c],
                        (tTMEM_LOADrS[None, 0, c], tTMEM_LOADrMax[None, c]),
                    )
            else:
                cute.copy(tmem_tiled_load, tTMEM_LOADtS, tTMEM_LOADrS)
            cute.arch.fence_view_async_tmem_load()
            _d256._apply_mask_hoisted(
                self.mask_type,
                tTMEM_LOADrS,
                tTMEM_LOADcS,
                tTMEM_LOADcS_static,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
            )
            row_max = tTMEM_LOADrS.load().reduce(cute.ReductionOp.MAX, row_max, 0)
        elif cutlass.const_expr(self.use_ldstat):
            # Unmasked fused-reduction path: each chunk load delivers its own MAX,
            # so the register FMAX chains disappear entirely — only the
            # num_chunks partial maxes are reduced at the end (fmha.py's
            # >sm_100f softmax path).
            tTMEM_LOADrMax = cute.make_rmem_tensor(
                cute.make_layout((1, num_chunks)), self.qk_acc_dtype
            )
            for c in cutlass.range_constexpr(num_chunks):
                cute.copy_atom_call(
                    tmem_tiled_load,
                    tTMEM_LOADtS[None, 0, c],
                    (tTMEM_LOADrS[None, 0, c], tTMEM_LOADrMax[None, c]),
                )
            cute.arch.fence_view_async_tmem_load()
            tile_row_max = tTMEM_LOADrMax.load().reduce(
                cute.ReductionOp.MAX, -cutlass.Float32.inf, 0
            )
            row_max = cute.arch.fmax(row_max, tile_row_max)
        else:
            # Unmasked steps: load S in four 32-element chunks and interleave
            # the row-max with the in-flight loads. tcgen05.ld results are
            # register-scoreboarded, so the FMAX on chunk i only waits for
            # chunk i's registers while later chunks stay in flight. Two max
            # chains, mirroring fmha.py's sm100 softmax schedule.
            cute.copy(
                tmem_tiled_load,
                tTMEM_LOADtS[None, None, 0],
                tTMEM_LOADrS[None, None, 0],
            )
            cute.copy(
                tmem_tiled_load,
                tTMEM_LOADtS[None, None, 1],
                tTMEM_LOADrS[None, None, 1],
            )
            row_max_a = row_max
            row_max_b = -cutlass.Float32.inf
            for j in cutlass.range_constexpr(0, 32, 4):
                row_max_a = cute.arch.fmax(row_max_a, tTMEM_LOADrS[j])
                row_max_a = cute.arch.fmax(row_max_a, tTMEM_LOADrS[j + 1])
                row_max_b = cute.arch.fmax(row_max_b, tTMEM_LOADrS[j + 2])
                row_max_b = cute.arch.fmax(row_max_b, tTMEM_LOADrS[j + 3])
            cute.copy(
                tmem_tiled_load,
                tTMEM_LOADtS[None, None, 2],
                tTMEM_LOADrS[None, None, 2],
            )
            for j in cutlass.range_constexpr(32, 64, 4):
                row_max_a = cute.arch.fmax(row_max_a, tTMEM_LOADrS[j])
                row_max_a = cute.arch.fmax(row_max_a, tTMEM_LOADrS[j + 1])
                row_max_b = cute.arch.fmax(row_max_b, tTMEM_LOADrS[j + 2])
                row_max_b = cute.arch.fmax(row_max_b, tTMEM_LOADrS[j + 3])
            cute.copy(
                tmem_tiled_load,
                tTMEM_LOADtS[None, None, 3],
                tTMEM_LOADrS[None, None, 3],
            )
            for j in cutlass.range_constexpr(64, 96, 4):
                row_max_a = cute.arch.fmax(row_max_a, tTMEM_LOADrS[j])
                row_max_a = cute.arch.fmax(row_max_a, tTMEM_LOADrS[j + 1])
                row_max_b = cute.arch.fmax(row_max_b, tTMEM_LOADrS[j + 2])
                row_max_b = cute.arch.fmax(row_max_b, tTMEM_LOADrS[j + 3])
            cute.arch.fence_view_async_tmem_load()
            for j in cutlass.range_constexpr(96, 128, 4):
                row_max_a = cute.arch.fmax(row_max_a, tTMEM_LOADrS[j])
                row_max_a = cute.arch.fmax(row_max_a, tTMEM_LOADrS[j + 1])
                row_max_b = cute.arch.fmax(row_max_b, tTMEM_LOADrS[j + 2])
                row_max_b = cute.arch.fmax(row_max_b, tTMEM_LOADrS[j + 3])
            row_max = cute.arch.fmax(row_max_a, row_max_b)
        if cutlass.const_expr(self.rescale_threshold > 0.0):
            # Conditional rescale (FA4): keep the previous reference max unless this
            # tile raised the row max by more than the threshold (log2 units), so most
            # steps leave O untouched; P <= 2^threshold, exact after the final 1/row_sum.
            grow = (row_max - old_row_max) * scale_softmax_log2
            row_max = cutlass.select_(grow > Float32(self.rescale_threshold), row_max, old_row_max)
        s_handle.release()
        row_max_safe = row_max
        if row_max == -cutlass.Float32.inf:
            row_max_safe = 0.0

        stats_handle = s_corr_producer.acquire_and_advance()
        stats_layout = cute.composition(
            tStS_slice.layout, cute.make_layout((tStS_slice.shape[0], 2))
        )
        stats_c_layout = cute.composition(
            tScS_slice.layout, cute.make_layout((tScS_slice.shape[0], 2))
        )
        tOtStats = cute.make_tensor(tStS_slice.iterator + self.tilePlikeFP32, stats_layout)
        tOcStats = cute.make_tensor(tScS_slice.iterator, stats_c_layout)
        tmem_store_stats_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(2)),
            self.qk_acc_dtype,
        )
        tiled_tmem_store_stats = tcgen05.make_tmem_copy(tmem_store_stats_atom, tOtStats)
        thr_tmem_store_stats = tiled_tmem_store_stats.get_slice(thread_idx)
        tTMEM_STOREcStats = thr_tmem_store_stats.partition_S(tOcStats)
        tTMEM_STORErStats = cute.make_rmem_tensor(tTMEM_STOREcStats.shape, self.qk_acc_dtype)
        tTMEM_STORErStats[0] = old_row_max
        tTMEM_STORErStats[1] = row_max_safe
        tTMEM_STOREtStats = thr_tmem_store_stats.partition_D(tOtStats)
        cute.copy(tiled_tmem_store_stats, tTMEM_STORErStats, tTMEM_STOREtStats)
        cute.arch.fence_view_async_tmem_store()
        stats_handle.commit()

        scale = scale_softmax_log2
        minus_row_max_scale = (0.0 - row_max_safe) * scale
        if cutlass.const_expr(self.p_prescale_log2 != 0.0):
            minus_row_max_scale = minus_row_max_scale + self.p_prescale_log2
        tTMEM_STORErP = cute.make_rmem_tensor(tTMEM_LOADrS.shape, self.p_dtype)

        # Software-pipelined exp2 loop (fmha.py-style): the scale FMA runs
        # FMA_COUNT elements ahead of the exp2 pair, while the P cvt and
        # the row-sum trail CVT_COUNT behind, so FMA / XU(MUFU) / ALU work
        # interleaves instead of serializing into phases.
        FMA_COUNT = 8
        CVT_COUNT = 4
        CVT_PER_STEP = 2
        SIZE = cute.size(tTMEM_LOADrS)
        # cvt4 (E4M3 P): four FP32 -> one register of four E4M3, issued every other loop
        # iteration for the elements CVT_COUNT + 2 .. CVT_COUNT + 5 behind (all exp'd by then)
        CVT4 = cutlass.const_expr(self.cvt4 and self.p_dtype.width == 8)
        tTMEM_LOADrS_cvt = cute.logical_divide(tTMEM_LOADrS, cute.make_layout(CVT_PER_STEP))
        tTMEM_STORErP_cvt = cute.logical_divide(tTMEM_STORErP, cute.make_layout(CVT_PER_STEP))
        tTMEM_LOADrS_cvt4 = cute.logical_divide(tTMEM_LOADrS, cute.make_layout(4))
        tTMEM_STORErP_cvt4 = cute.logical_divide(tTMEM_STORErP, cute.make_layout(4))
        for i in cutlass.range_constexpr(0, FMA_COUNT, 2):
            tTMEM_LOADrS[i], tTMEM_LOADrS[i + 1] = cute.arch.fma_packed_f32x2(
                (tTMEM_LOADrS[i], tTMEM_LOADrS[i + 1]),
                (scale, scale),
                (minus_row_max_scale, minus_row_max_scale),
            )
        acc_scale_ = scale * (old_row_max - row_max_safe)
        acc_scale = cute.math.exp2(acc_scale_, fastmath=True) * 0.5
        row_sum *= acc_scale
        local_row_sum = (row_sum, row_sum)
        for i in cutlass.range_constexpr(0, SIZE, 2):
            if cutlass.const_expr(i >= CVT_COUNT):
                if cutlass.const_expr(CVT4):
                    if cutlass.const_expr(i >= CVT_COUNT + 2 and (i - CVT_COUNT - 2) % 4 == 0):
                        g4 = (i - CVT_COUNT - 2) // 4
                        fmha_utils.cvt_f32x4_to_f8x4(
                            tTMEM_LOADrS_cvt4[None, g4], tTMEM_STORErP_cvt4[None, g4]
                        )
                else:
                    g = (i - CVT_COUNT) // CVT_PER_STEP
                    s_vec = tTMEM_LOADrS_cvt[None, g].load()
                    tTMEM_STORErP_cvt[None, g].store(s_vec.to(self.p_dtype))
                local_row_sum = cute.arch.add_packed_f32x2(
                    local_row_sum,
                    (
                        tTMEM_LOADrS[i - CVT_COUNT],
                        tTMEM_LOADrS[i - CVT_COUNT + 1],
                    ),
                )
            tTMEM_LOADrS[i] = cute.math.exp2(tTMEM_LOADrS[i], fastmath=True)
            if cutlass.const_expr(i + FMA_COUNT < SIZE):
                (
                    tTMEM_LOADrS[i + FMA_COUNT],
                    tTMEM_LOADrS[i + FMA_COUNT + 1],
                ) = cute.arch.fma_packed_f32x2(
                    (
                        tTMEM_LOADrS[i + FMA_COUNT],
                        tTMEM_LOADrS[i + FMA_COUNT + 1],
                    ),
                    (scale, scale),
                    (minus_row_max_scale, minus_row_max_scale),
                )
            tTMEM_LOADrS[i + 1] = cute.math.exp2(tTMEM_LOADrS[i + 1], fastmath=True)
        # Trailing cvt + row-sum for the last CVT_COUNT elements.
        if cutlass.const_expr(CVT4):
            # the 4-groups past the last one converted in the loop
            g4_next = (SIZE - 2 - CVT_COUNT - 2) // 4 + 1
            for g4 in cutlass.range_constexpr(g4_next, SIZE // 4):
                fmha_utils.cvt_f32x4_to_f8x4(
                    tTMEM_LOADrS_cvt4[None, g4], tTMEM_STORErP_cvt4[None, g4]
                )
            for i in cutlass.range_constexpr(SIZE - CVT_COUNT, SIZE, 2):
                local_row_sum = cute.arch.add_packed_f32x2(
                    local_row_sum, (tTMEM_LOADrS[i], tTMEM_LOADrS[i + 1])
                )
        else:
            for i in cutlass.range_constexpr(SIZE - CVT_COUNT, SIZE, 2):
                g = i // CVT_PER_STEP
                s_vec = tTMEM_LOADrS_cvt[None, g].load()
                tTMEM_STORErP_cvt[None, g].store(s_vec.to(self.p_dtype))
                local_row_sum = cute.arch.add_packed_f32x2(
                    local_row_sum, (tTMEM_LOADrS[i], tTMEM_LOADrS[i + 1])
                )

        p_handle = p_mma_producer.acquire_and_advance()
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition(32)), self.qk_acc_dtype
        )
        tilePlikeFP32 = tStS_slice.shape[1] // Float32.width * self.p_dtype.width
        tStS_P_layout = cute.composition(
            tStS_slice.layout, cute.make_layout((tStS_slice.shape[0], tilePlikeFP32))
        )
        tStS_P = cute.make_tensor(tStS_slice.iterator, tStS_P_layout)
        tScS_P_layout = cute.composition(
            tScS_slice.layout, cute.make_layout((tScS_slice.shape[0], tilePlikeFP32))
        )
        tScS_P = cute.make_tensor(tScS_slice.iterator, tScS_P_layout)
        tmem_tiled_store = tcgen05.make_tmem_copy(tmem_store_atom, tStS_P)
        thr_store = tmem_tiled_store.get_slice(thread_idx)
        tTMEM_STOREtP = thr_store.partition_D(tStS_P)
        tTMEM_STOREcS = thr_store.partition_S(tScS_P)
        tTMEM_STORErP_ = cute.make_tensor(
            cute.recast_ptr(tTMEM_STORErP.iterator, dtype=self.qk_acc_dtype),
            tTMEM_STOREcS.shape,
        )
        cute.copy(tmem_tiled_store, tTMEM_STORErP_, tTMEM_STOREtP)
        cute.arch.fence_view_async_tmem_store()

        p_handle.commit()
        row_sum = local_row_sum[0] + local_row_sum[1]
        return row_max, row_sum, mma_s_consumer, p_mma_producer, s_corr_producer

    @cute.jit
    def correction_rescale(
        self,
        scale_softmax_log2: Float32,
        stats_args: tuple,
        o_args: tuple,
        epi_tile: cute.Tile,
    ) -> pipeline.PipelineConsumer:
        (s_corr_consumer, tStS, tScS) = stats_args
        (mma_o_consumer, tOtO_staged, cO_staged) = o_args
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))

        stats_handle = s_corr_consumer.wait_and_advance()
        tStS_slice = tStS[(None, None), 0, 0, stats_handle.index]
        tScS_slice = tScS[(None, None), 0, 0]
        stats_layout = cute.composition(
            tStS_slice.layout, cute.make_layout((tStS_slice.shape[0], 2))
        )
        stats_c_layout = cute.composition(
            tScS_slice.layout, cute.make_layout((tScS_slice.shape[0], 2))
        )
        tOtStats = cute.make_tensor(tStS_slice.iterator + self.tilePlikeFP32, stats_layout)
        tOcStats = cute.make_tensor(tScS_slice.iterator, stats_c_layout)
        tmem_load_stats_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(2)),
            self.qk_acc_dtype,
        )
        tiled_tmem_load_stats = tcgen05.make_tmem_copy(tmem_load_stats_atom, tOtStats)
        thr_tmem_load_stats = tiled_tmem_load_stats.get_slice(thread_idx)
        tTMEM_LOADtStats = thr_tmem_load_stats.partition_S(tOtStats)
        tTMEM_LOADcStats = thr_tmem_load_stats.partition_D(tOcStats)
        tTMEM_LOADrStats = cute.make_rmem_tensor(tTMEM_LOADcStats.shape, self.qk_acc_dtype)
        cute.copy(tiled_tmem_load_stats, tTMEM_LOADtStats, tTMEM_LOADrStats)

        scale = scale_softmax_log2 * (tTMEM_LOADrStats[0] - tTMEM_LOADrStats[1])
        scale = cute.math.exp2(scale, fastmath=True)
        need = Boolean(True)
        if cutlass.const_expr(self.rescale_threshold > 0.0):
            # rescale this warp's 32 rows only if one of them moved its reference max
            need = cute.arch.vote_any_sync(tTMEM_LOADrStats[0] != tTMEM_LOADrStats[1])
        stats_handle.release()
        # Per-chunk O handoff: wait/rescale/release one dv chunk at a time so
        # chunk `iter` is back in the MMA's hands while chunk `iter+1` is still
        # being rescaled. The Rep16 double-buffered load/store pipeline inside
        # each chunk is unchanged.
        for iter in cutlass.range(self.iterations_pv, unroll_full=True):
            o_handle = mma_o_consumer.wait_and_advance()
            if need:
                tOtO = tOtO_staged[(None, None), 0, 0, iter]
                cO = cO_staged[None, None, iter]
                tOtO_epi = cute.zipped_divide(tOtO, epi_tile)
                cO_epi = cute.zipped_divide(cO, epi_tile)
                tmem_load_atom = cute.make_copy_atom(
                    tcgen05.Ld32x32bOp(tcgen05.Repetition(16)),
                    self.pv_acc_dtype,
                )
                tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_epi)
                thr_load = tmem_tiled_load.get_slice(thread_idx)
                tmem_store_atom = cute.make_copy_atom(
                    tcgen05.St32x32bOp(tcgen05.Repetition(16)),
                    self.pv_acc_dtype,
                )
                tmem_store_atom = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_epi)
                thr_store = tmem_store_atom.get_slice(thread_idx)
                tTMEM_LOADtO = thr_load.partition_S(tOtO_epi)
                tTMEM_LOADcO = thr_load.partition_D(cO_epi)
                tTMEM_STOREtO = thr_store.partition_D(tOtO_epi)
                tTMrO = cute.make_rmem_tensor_like(
                    cute.append(
                        cute.make_layout(tTMEM_LOADcO[None, 0, 0].shape),
                        cute.make_layout(2, stride=cute.size(tTMEM_LOADcO[None, 0, 0].shape)),
                    ),
                    self.pv_acc_dtype,
                )
                tTMEM_LOADtO_0 = tTMEM_LOADtO[None, 0, 0]
                cute.copy(tmem_tiled_load, tTMEM_LOADtO_0, tTMrO[None, 0])
                iter_num = cute.size(tTMEM_LOADtO, mode=[1])
                for i in cutlass.range(1, iter_num, unroll_full=True):
                    tTMEM_LOADtO_i = tTMEM_LOADtO[None, i, 0]
                    cute.copy(tmem_tiled_load, tTMEM_LOADtO_i, tTMrO[None, i % 2])
                    for j in cutlass.range(
                        cute.size(tTMrO, mode=[0]), unroll_full=True, vectorize=True
                    ):
                        tTMrO[j, (i - 1) % 2] = tTMrO[j, (i - 1) % 2] * scale
                    tTMEM_STOREtO_prev_i = tTMEM_STOREtO[None, i - 1, 0]
                    cute.copy(tmem_store_atom, tTMrO[None, (i - 1) % 2], tTMEM_STOREtO_prev_i)

                for j in cutlass.range(
                    cute.size(tTMrO, mode=[0]), unroll_full=True, vectorize=True
                ):
                    tTMrO[j, (iter_num - 1) % 2] = tTMrO[j, (iter_num - 1) % 2] * scale
                cute.copy(
                    tmem_store_atom,
                    tTMrO[None, (iter_num - 1) % 2],
                    tTMEM_STOREtO[None, iter_num - 1, 0],
                )
                cute.arch.fence_view_async_tmem_store()
            o_handle.release()
        return mma_o_consumer, s_corr_consumer

    def _tail_frame(self, cu_b, seqlen_b, seqlen_k):
        """Tail-first frame of one request (module doc): ``(first row, query rows, s)`` =
        ``(cu_b - s, seqlen_b + s, s)``; rows ``< s`` of the frame are not stored. ``s = pad`` (``pad =
        256 n - seqlen_b``) or, with ``tail_align``, the largest ``s <= pad`` with ``s = c (mod 128)``
        for the ``c = seqlen_k - seqlen_b`` cached rows (0 if none). Traced on the kernel's Int32
        values, signed arithmetic only, no control flow."""
        tm = self.qk_mma_tiler[0]
        tk = self.qk_mma_tiler[1]
        pad = (seqlen_b + (tm - 1)) // tm * tm - seqlen_b
        s = pad
        if cutlass.const_expr(self.tail_align):
            # (pad - c) mod tk, non-negative for either remainder sign convention
            r = ((pad - (seqlen_k - seqlen_b)) % tk + tk) % tk
            s = cutlass.max(pad - r, Int32(0))
        return cu_b - s, seqlen_b + s, s

    @cute.jit
    def correction_epilog_tail(
        self,
        value_args: Tuple,
        sum_args: Tuple,
        o_args: Tuple,
        epi_tile: cute.Tile,
    ) -> Tuple[pipeline.PipelineConsumer, pipeline.PipelineProducer]:
        """The base ``correction_epilog`` for ``tail_first``: ``value_args`` carries a 4th entry,
        the frame's first stored row, and the row predicate also rejects the leading rows."""
        (seqlen_q, scale_output, scale_softmax_log2, row_lo) = value_args
        (sum_consumer, sSum, sMax) = sum_args
        (mma_o_consumer, gO_staged, cO_staged, tOtO_staged, gLSE) = o_args
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
        sum_handle = sum_consumer.wait_and_advance()
        row_sum = sSum[thread_idx]
        row_max = Float32(0.0)
        if cutlass.const_expr(sMax is not None):
            row_max = sMax[thread_idx]
        cute.arch.fence_view_async_shared()
        sum_handle.release()
        scale = scale_output / row_sum
        if cutlass.const_expr(gLSE is not None):
            lse = cute.math.log2(row_sum, fastmath=True) + scale_softmax_log2 * row_max
            if cutlass.const_expr(self.p_prescale_log2 != 0.0):
                lse = lse - self.p_prescale_log2
            if _row_stored(cO_staged[thread_idx, 0, 0][0], seqlen_q, row_lo):
                gLSE[thread_idx] = lse
        for iter in cutlass.range(self.iterations_pv):
            o_handle = mma_o_consumer.wait_and_advance()
            gO = gO_staged[None, None, iter]
            cO = cO_staged[None, None, iter]
            tOtO = tOtO_staged[(None, None), 0, 0, iter]
            tOtO_epi = cute.zipped_divide(tOtO, epi_tile)
            cO_epi = cute.zipped_divide(cO, epi_tile)
            gO_epi = cute.zipped_divide(gO, epi_tile)
            tidx, _, _ = cute.arch.thread_idx()
            thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
            tmem_copy_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.pv_acc_dtype
            )
            tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_epi)
            thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
            tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_epi)
            tTMEM_LOADgO = thr_tmem_load.partition_D(gO_epi)
            tTMEM_LOADcO = thr_tmem_load.partition_D(cO_epi)
            for i in cutlass.range(cute.size(tTMEM_LOADtO, mode=[1]), unroll_full=True):
                tTMEM_LOADtO_i = tTMEM_LOADtO[None, i, 0]
                tTMEM_LOADgO_i = tTMEM_LOADgO[None, i, 0]
                tTMEM_LOADcO_i = tTMEM_LOADcO[None, i, 0]
                tTMrO = cute.make_rmem_tensor(tTMEM_LOADcO[None, 0, i].shape, self.pv_acc_dtype)
                cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO)
                for j in cutlass.range(cute.size(tTMrO), unroll_full=True, vectorize=True):
                    tTMrO[j] = tTMrO[j] * scale
                tSMrO = cute.make_rmem_tensor(tTMrO.shape, self.o_dtype)
                o_vec = tTMrO.load()
                tSMrO.store(o_vec.to(self.o_dtype))
                if _row_stored(tTMEM_LOADcO_i[0][0], seqlen_q, row_lo):
                    cute.autovec_copy(tSMrO, tTMEM_LOADgO_i)
            o_handle.release()
        return mma_o_consumer, sum_consumer

    @cute.jit
    def correction_epilog_q8(
        self,
        value_args: Tuple,
        sum_args: Tuple,
        o_args: Tuple,
        epi_tile: cute.Tile,
        q8_args: Tuple,
    ) -> Tuple[pipeline.PipelineConsumer, pipeline.PipelineProducer]:
        """The base ``correction_epilog`` that also writes O as the o_proj MXF8 operand.

        A correction thread's TMEM load (32x32b, 32 repetitions) is 32 consecutive columns of
        one row -- one whole 1x32 scale block -- so the block max needs no shuffle; the scale
        rule and the conversion are ``self_attn_prefill_act_quant_mxf8``'s, applied to the
        BF16 O values just stored, so the bytes are that kernel's for this O.  ``q8_args =
        (gQ8_staged, sf8_base, row_base, head)``: the e4m3 partition of the operand (O's
        layout), the scale bytes, the request's first global row and the tile's head."""
        seqlen_q = value_args[0]
        scale_output = value_args[1]
        scale_softmax_log2 = value_args[2]
        row_lo = None
        if cutlass.const_expr(len(value_args) > 3):
            row_lo = value_args[3]
        (sum_consumer, sSum, sMax) = sum_args
        (mma_o_consumer, gO_staged, cO_staged, tOtO_staged, gLSE) = o_args
        (gQ8_staged, sf8_base, row_base, head) = q8_args
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
        sum_handle = sum_consumer.wait_and_advance()
        row_sum = sSum[thread_idx]
        row_max = Float32(0.0)
        if cutlass.const_expr(sMax is not None):
            row_max = sMax[thread_idx]
        cute.arch.fence_view_async_shared()
        sum_handle.release()
        scale = scale_output / row_sum
        if cutlass.const_expr(gLSE is not None):
            lse = cute.math.log2(row_sum, fastmath=True) + scale_softmax_log2 * row_max
            if cutlass.const_expr(self.p_prescale_log2 != 0.0):
                lse = lse - self.p_prescale_log2
            if _row_stored(cO_staged[thread_idx, 0, 0][0], seqlen_q, row_lo):
                gLSE[thread_idx] = lse
        k_tiles = Int64(NUM_HEADS * HEAD_DIM // 128)
        col_base = Int64(head) * HEAD_DIM
        for iter in cutlass.range(self.iterations_pv):
            o_handle = mma_o_consumer.wait_and_advance()
            gO = gO_staged[None, None, iter]
            gQ8 = gQ8_staged[None, None, iter]
            cO = cO_staged[None, None, iter]
            tOtO = tOtO_staged[(None, None), 0, 0, iter]
            tOtO_epi = cute.zipped_divide(tOtO, epi_tile)
            cO_epi = cute.zipped_divide(cO, epi_tile)
            gO_epi = cute.zipped_divide(gO, epi_tile)
            gQ8_epi = cute.zipped_divide(gQ8, epi_tile)
            tidx, _, _ = cute.arch.thread_idx()
            thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
            tmem_copy_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.pv_acc_dtype
            )
            tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_epi)
            thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
            tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_epi)
            tTMEM_LOADgO = thr_tmem_load.partition_D(gO_epi)
            tTMEM_LOADgQ8 = thr_tmem_load.partition_D(gQ8_epi)
            tTMEM_LOADcO = thr_tmem_load.partition_D(cO_epi)
            if cutlass.const_expr(self.epi_fast):
                EPI_N = cute.size(tTMEM_LOADtO, mode=[1])
                # read the whole chunk, return it to the MMA, then work from registers
                tTMrO_all = _rmem_list(EPI_N, tTMEM_LOADcO[None, 0, 0].shape, self.pv_acc_dtype)
                for i in cutlass.range_constexpr(EPI_N):
                    cute.copy(tiled_tmem_load, tTMEM_LOADtO[None, i, 0], tTMrO_all[i])
                cute.arch.fence_view_async_tmem_load()
                o_handle.release()
                # stage 1 (all sub-chunks): output scale, BF16 rounding by packed cvt
                tSMrO_all = _rmem_list(EPI_N, tTMrO_all[0].shape, self.o_dtype)
                for i in cutlass.range_constexpr(EPI_N):
                    tTMrO = tTMrO_all[i]
                    assert cute.size(tTMrO) == 32, "the q8 epilogue needs 32 columns per thread"
                    for j in cutlass.range(cute.size(tTMrO), unroll_full=True, vectorize=True):
                        tTMrO[j] = tTMrO[j] * scale
                    tSMrO_i32 = cute.recast_tensor(tSMrO_all[i], Int32)
                    for k in cutlass.range_constexpr(16):
                        pk = _cvt_bf16x2_rn(tTMrO[2 * k + 1], tTMrO[2 * k])
                        tSMrO_i32[k] = pk
                        tTMrO[2 * k] = (pk << Int32(16)).bitcast(Float32)
                        tTMrO[2 * k + 1] = (pk & Int32(-65536)).bitcast(Float32)
                # stage 2: the four 32-value block scales (independent fmax trees)
                codes = []
                inv_scales = []
                for i in cutlass.range_constexpr(EPI_N):
                    c_i, s_i = _mxf8.mxf8_code_and_inv_scale(_absmax32(tTMrO_all[i]))
                    codes.append(c_i)
                    inv_scales.append(s_i)
                # stage 3: quantize + store
                for i in cutlass.range_constexpr(EPI_N):
                    tTMEM_LOADgO_i = tTMEM_LOADgO[None, i, 0]
                    tTMEM_LOADgQ8_i = tTMEM_LOADgQ8[None, i, 0]
                    tTMEM_LOADcO_i = tTMEM_LOADcO[None, i, 0]
                    tTMrO = tTMrO_all[i]
                    for j in cutlass.range(cute.size(tTMrO), unroll_full=True, vectorize=True):
                        tTMrO[j] = tTMrO[j] * inv_scales[i]
                    tQ8r = cute.make_rmem_tensor(tTMrO.shape, cutlass.Float8E4M3FN)
                    tTMrO_4 = cute.logical_divide(tTMrO, cute.make_layout(4))
                    tQ8r_4 = cute.logical_divide(tQ8r, cute.make_layout(4))
                    for g in cutlass.range_constexpr(8):
                        fmha_utils.cvt_f32x4_to_f8x4(tTMrO_4[None, g], tQ8r_4[None, g])
                    if _row_stored(tTMEM_LOADcO_i[0][0], seqlen_q, row_lo):
                        if cutlass.const_expr(self.q8_store_o):
                            cute.autovec_copy(tSMrO_all[i], tTMEM_LOADgO_i)
                        cute.autovec_copy(tQ8r, tTMEM_LOADgQ8_i)
                        m = Int64(row_base) + Int64(tTMEM_LOADcO_i[0][0])
                        col = col_base + Int64(tTMEM_LOADcO_i[0][1])
                        (sf8_base + _mxf8.mxf8_sf_offset(m, col // 32, k_tiles)).store(
                            cutlass.Uint8(codes[i])
                        )
            else:
                for i in cutlass.range(cute.size(tTMEM_LOADtO, mode=[1]), unroll_full=True):
                    tTMEM_LOADtO_i = tTMEM_LOADtO[None, i, 0]
                    tTMEM_LOADgO_i = tTMEM_LOADgO[None, i, 0]
                    tTMEM_LOADgQ8_i = tTMEM_LOADgQ8[None, i, 0]
                    tTMEM_LOADcO_i = tTMEM_LOADcO[None, i, 0]
                    tTMrO = cute.make_rmem_tensor(tTMEM_LOADcO[None, 0, i].shape, self.pv_acc_dtype)
                    assert cute.size(tTMrO) == 32, "the q8 epilogue needs 32 columns per thread"
                    cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO)
                    for j in cutlass.range(cute.size(tTMrO), unroll_full=True, vectorize=True):
                        tTMrO[j] = tTMrO[j] * scale
                    tSMrO = cute.make_rmem_tensor(tTMrO.shape, self.o_dtype)
                    o_vec = tTMrO.load()
                    tSMrO.store(o_vec.to(self.o_dtype))
                    # MXF8 of the BF16 values (act quant's rule, the 32 values are one block);
                    # the FP32 accumulator registers are reused for them (no extra 32 registers)
                    tTMrO.store(tSMrO.load().to(Float32))
                    block_max = Float32(0.0)
                    for j in cutlass.range_constexpr(32):
                        a = cute.math.absf(tTMrO[j])
                        if a > block_max:
                            block_max = a
                    code, inv_scale = _mxf8.mxf8_code_and_inv_scale(block_max)
                    tQ8r = cute.make_rmem_tensor(tTMrO.shape, cutlass.Float8E4M3FN)
                    tQ8r.store((tTMrO.load() * inv_scale).to(cutlass.Float8E4M3FN))
                    if _row_stored(tTMEM_LOADcO_i[0][0], seqlen_q, row_lo):
                        if cutlass.const_expr(self.q8_store_o):
                            cute.autovec_copy(tSMrO, tTMEM_LOADgO_i)
                        cute.autovec_copy(tQ8r, tTMEM_LOADgQ8_i)
                        m = Int64(row_base) + Int64(tTMEM_LOADcO_i[0][0])
                        col = col_base + Int64(tTMEM_LOADcO_i[0][1])
                        (sf8_base + _mxf8.mxf8_sf_offset(m, col // 32, k_tiles)).store(
                            cutlass.Uint8(code)
                        )
                o_handle.release()
        return mma_o_consumer, sum_consumer

    @cute.jit
    def correction_epilog_stage(
        self,
        value_args: Tuple,
        sum_args: Tuple,
        o_args: Tuple,
        epi_tile: cute.Tile,
        stage_args: Tuple,
        meta: Tuple,
    ):
        """``epi_offload``, a work tile's end on the correction warps: O (each 128-column chunk read
        whole and handed back to the MMA) times ``scale_output / row_sum``, rounded to BF16 by the
        served packed RN cvt into this thread's stage row (row = TMEM lane = thread), plus the row's
        state (1 stored / 2 not stored), global row and head; then handed to the epilogue warps."""
        seqlen_q = value_args[0]
        scale_output = value_args[1]
        row_lo = None
        if cutlass.const_expr(len(value_args) > 3):
            row_lo = value_args[3]
        (sum_consumer, sSum) = sum_args
        (mma_o_consumer, cO_staged, tOtO_staged) = o_args
        (sStage, epi_producer) = stage_args
        (row_base, head) = meta
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.correction_warp_ids))
        sum_handle = sum_consumer.wait_and_advance()
        row_sum = sSum[thread_idx]
        cute.arch.fence_view_async_shared()
        sum_handle.release()
        scale = scale_output / row_sum
        # the stage is free once the epilogue warps released the previous tile's rows
        stage_handle = epi_producer.acquire_and_advance()
        row = Int32(0)
        for iter in cutlass.range_constexpr(self.iterations_pv):
            o_handle = mma_o_consumer.wait_and_advance()
            cO = cO_staged[None, None, iter]
            tOtO = tOtO_staged[(None, None), 0, 0, iter]
            tOtO_epi = cute.zipped_divide(tOtO, epi_tile)
            cO_epi = cute.zipped_divide(cO, epi_tile)
            tmem_copy_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.pv_acc_dtype
            )
            tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_epi)
            thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
            tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_epi)
            tTMEM_LOADcO = thr_tmem_load.partition_D(cO_epi)
            EPI_N = cute.size(tTMEM_LOADtO, mode=[1])
            tTMrO_all = _rmem_list(EPI_N, tTMEM_LOADcO[None, 0, 0].shape, self.pv_acc_dtype)
            for i in cutlass.range_constexpr(EPI_N):
                cute.copy(tiled_tmem_load, tTMEM_LOADtO[None, i, 0], tTMrO_all[i])
            cute.arch.fence_view_async_tmem_load()
            o_handle.release()
            if cutlass.const_expr(iter == 0):
                row = Int32(tTMEM_LOADcO[None, 0, 0][0][0])
            for i in cutlass.range_constexpr(EPI_N):
                tTMrO = tTMrO_all[i]
                assert cute.size(tTMrO) == 32, "the offloaded epilogue needs 32 columns per thread"
                for j in cutlass.range(cute.size(tTMrO), unroll_full=True, vectorize=True):
                    tTMrO[j] = tTMrO[j] * scale
                packed = cute.make_rmem_tensor(cute.make_layout(16), Int32)
                for k in cutlass.range_constexpr(16):
                    packed[k] = _cvt_bf16x2_rn(tTMrO[2 * k + 1], tTMrO[2 * k])
                w0 = Int32(tTMEM_LOADcO[None, i, 0][0][1]) // 2
                dst_s = cute.make_tensor(
                    sStage.iterator + cute.assume(thread_idx * self.stage_row_words + w0, divby=16),
                    cute.make_layout(16),
                )
                cute.autovec_copy(packed, dst_s)
        state = Int32(2)
        if _row_stored(row, seqlen_q, row_lo):
            state = Int32(1)
        sStage[thread_idx, 128] = state
        sStage[thread_idx, 129] = Int32(row_base) + row
        sStage[thread_idx, 130] = Int32(head)
        stage_handle.commit()
        return mma_o_consumer, sum_consumer, epi_producer

    @cute.jit
    def epi_signal_end(self, sStage: cute.Tensor, epi_producer):
        """``epi_offload``: after the correction warps' last tile, a stage whose rows are all in
        state 0 tells the epilogue warps to exit."""
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.correction_warp_ids))
        stage_handle = epi_producer.acquire_and_advance()
        sStage[thread_idx, 128] = Int32(0)
        stage_handle.commit()

    @cute.jit
    def epi_rows(self, sStage: cute.Tensor, epi_consumer, out_args: Tuple):
        """``epi_offload``, the epilogue warps: one staged tile's row per thread (row = thread
        index, as staged), its eight 32-value blocks block-scaled and converted by
        ``self_attn_prefill_act_quant_mxf8``'s rule (the served epilogue's unpack, amax, code
        and E4M3 conversion), stored as o_proj's operand plus the e8m0 bytes (and BF16 O when
        ``q8_store_o``). Returns the row state (0: no more tiles)."""
        (mQ8, mO, mSF8) = out_args
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.epi_warp_ids))
        k_cols = NUM_HEADS * HEAD_DIM
        k_tiles = Int64(k_cols // 128)
        stage_handle = epi_consumer.wait_and_advance()
        state = sStage[thread_idx, 128]
        if state == Int32(1):
            m = Int64(sStage[thread_idx, 129])
            col_base = Int64(sStage[thread_idx, 130]) * HEAD_DIM
            for sub in cutlass.range(HEAD_DIM // 32, unroll=1):
                w = cute.make_rmem_tensor(cute.make_layout(16), Int32)
                src_s = cute.make_tensor(
                    sStage.iterator
                    + cute.assume(thread_idx * self.stage_row_words + sub * 16, divby=16),
                    cute.make_layout(16),
                )
                cute.autovec_copy(src_s, w)
                x = cute.make_rmem_tensor(cute.make_layout(32), Float32)
                for k in cutlass.range_constexpr(16):
                    x[2 * k] = (w[k] << Int32(16)).bitcast(Float32)
                    x[2 * k + 1] = (w[k] & Int32(-65536)).bitcast(Float32)
                code, inv_scale = _mxf8.mxf8_code_and_inv_scale(_absmax32(x))
                for k in cutlass.range(32, unroll_full=True, vectorize=True):
                    x[k] = x[k] * inv_scale
                q8r = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float8E4M3FN)
                x4 = cute.logical_divide(x, cute.make_layout(4))
                q4 = cute.logical_divide(q8r, cute.make_layout(4))
                for g in cutlass.range_constexpr(8):
                    fmha_utils.cvt_f32x4_to_f8x4(x4[None, g], q4[None, g])
                col = col_base + Int64(sub * 32)
                off = cute.assume(m * k_cols + col, divby=32)
                dst = cute.make_tensor(mQ8.iterator + off, cute.make_layout(32))
                cute.autovec_copy(q8r, dst)
                if cutlass.const_expr(self.q8_store_o):
                    dst_o = cute.make_tensor(
                        cute.recast_ptr(mO.iterator + off, dtype=Int32),
                        cute.make_layout(16),
                    )
                    cute.autovec_copy(w, dst_o)
                (mSF8.iterator + _mxf8.mxf8_sf_offset(m, col // 32, k_tiles)).store(
                    cutlass.Uint8(code)
                )
        stage_handle.release()
        return epi_consumer, state

    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_kdl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_dkl: cute.Tensor,
        mO_qdl: cute.Tensor,
        cu_seqlens: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        mLSE_ql: Optional[cute.Tensor],
        scale_softmax_log2: Float32,
        scale_output: Float32,
        head_scales: Optional[cute.Tensor],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        cluster_layout_vmnk: cute.Layout,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        p_tmem_layout: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        tile_sched_params: fmha_utils.FmhaStaticTileSchedulerParams,
        mQ8_qdl: Optional[cute.Tensor],
        mSF8: Optional[cute.Tensor],
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # Prefetch tma desc
        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(qk_tiled_mma.thr_id.shape)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        # Alloc
        smem = SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.q_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.load_warp_id])
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.mma_warp_id])
            ),
            tx_count=self.tma_copy_q_bytes,
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        # Uniform precision: K/V are TMA-loaded straight into the MMA-B buffer and
        # consumed by the MMA (UMMA) warp directly -- no dequant/scale stages.
        load_kv_producer, load_kv_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.kv_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.load_warp_id])
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.mma_warp_id])
            ),
            tx_count=self.tma_copy_kv_bytes,
            barrier_storage=storage.load_kv_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        mma_s_producer, mma_s_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.qk_acc_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.mma_warp_id])
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len(self.softmax_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            barrier_storage=storage.mma_s_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        p_mma_producer, p_mma_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.qk_acc_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len(self.softmax_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.mma_warp_id])
            ),
            barrier_storage=storage.p_mma_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        s_corr_producer, s_corr_consumer = pipeline.PipelineAsync.create(
            num_stages=self.qk_acc_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.softmax_warp_ids),
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.correction_warp_ids),
            ),
            barrier_storage=storage.s_corr_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        sum_producer, sum_consumer = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.softmax_warp_ids),
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.correction_warp_ids),
            ),
            barrier_storage=storage.sum_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        mma_o_producer, mma_o_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.pv_acc_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.mma_warp_id])
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len(self.correction_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            barrier_storage=storage.mma_o_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        clc_pipeline = None
        if cutlass.const_expr(self.sched == "clc"):
            # Consumers: load + mma + 4 softmax + 4 correction warps in both CTAs, plus
            # the leader's scheduler warp (dense_gemm_persistent_dynamic.py's pattern).
            clc_pipeline = pipeline.PipelineClcFetchAsync.create(
                barrier_storage=storage.clc_mbar_ptr.data_ptr(),
                num_stages=self.clc_stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.threads_per_warp
                    * (
                        1
                        + self.cluster_shape_mnk[0]
                        * (2 + len(self.softmax_warp_ids) + len(self.correction_warp_ids))
                    ),
                ),
                tx_count=16,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=self.threads_per_warp
            * len((self.mma_warp_id, *self.softmax_warp_ids, *self.correction_warp_ids)),
        )
        tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_dealloc_sync_bar_id,
            num_threads=self.threads_per_warp * len(self.correction_warp_ids),
        )
        # Tensor memory dealloc barrier init
        tmem = TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.correction_warp_ids[0],
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        sQ = smem.allocate_tensor(
            element_type=self.q_dtype,
            layout=q_smem_layout_staged.outer,
            swizzle=q_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sK = smem.allocate_tensor(
            element_type=self.k_dtype,
            layout=k_smem_layout_staged.outer,
            swizzle=k_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sV_ptr = cute.recast_ptr(sK.iterator, v_smem_layout_staged.inner)
        sV = cute.make_tensor(sV_ptr, v_smem_layout_staged.outer)

        sSum = smem.allocate_tensor(
            element_type=self.qk_acc_dtype,
            layout=cute.make_layout(len(self.softmax_warp_ids) * self.threads_per_warp),
            byte_alignment=128,
        )
        # Row-max rides the same softmax->correction handoff when LSE is on.
        sMax = None
        if cutlass.const_expr(mLSE_ql is not None):
            sMax = smem.allocate_tensor(
                element_type=self.qk_acc_dtype,
                layout=cute.make_layout(len(self.softmax_warp_ids) * self.threads_per_warp),
                byte_alignment=128,
            )

        # epi_offload: the BF16 O stage (one row per correction / epilogue thread) and its
        # one-stage correction -> epilogue-warp pipeline (mbarriers in the dynamic SMEM so the
        # default build's static SMEM plan is unchanged; initialized here, made visible below)
        sStage = None
        epi_producer = None
        epi_consumer = None
        if cutlass.const_expr(self.epi_offload):
            sStage = smem.allocate_tensor(
                element_type=Int32,
                layout=cute.make_layout(
                    (self.cta_tiler[0], self.stage_row_words),
                    stride=(self.stage_row_words, 1),
                ),
                byte_alignment=128,
            )
            sEpiBar = smem.allocate_tensor(
                element_type=Int64, layout=cute.make_layout(2), byte_alignment=16
            )
            epi_producer, epi_consumer = pipeline.PipelineAsync.create(
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.threads_per_warp * len(self.correction_warp_ids),
                ),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.threads_per_warp * len(self.epi_warp_ids),
                ),
                barrier_storage=sEpiBar.iterator,
                defer_sync=True,
            ).make_participants()
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()
        qk_thr_mma = qk_tiled_mma.get_slice(mma_tile_coord_v)
        pv_thr_mma = pv_tiled_mma.get_slice(mma_tile_coord_v)
        tSrQ = qk_thr_mma.make_fragment_A(sQ)
        tSrK = qk_thr_mma.make_fragment_B(sK)
        tOrV = pv_thr_mma.make_fragment_B(sV)
        qk_acc_shape = pv_thr_mma.partition_shape_C((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
        # (atomV, restM, restN, accStage)
        tStS = qk_tiled_mma.make_fragment_C(cute.append(qk_acc_shape, self.qk_acc_stage))
        pv_acc_shape = pv_thr_mma.partition_shape_C(cute.select(self.pv_mma_tiler, mode=[0, 1]))
        # (atomV, restM, restN)
        tOtO = pv_thr_mma.make_fragment_C(pv_acc_shape)
        tOtO_layout = cute.append(
            tOtO.layout,
            cute.make_layout(
                self.iterations_pv,
                stride=self.pv_mma_tiler[1] // self.tmem_warp_shape_mn[1],
            ),
        )
        tStS = cute.make_tensor(tStS.iterator + self.tmem_s_offset, tStS.layout)
        tOtO_staged = cute.make_tensor(tOtO.iterator + self.tmem_o_offset, tOtO_layout)
        # Varlen (GLM): Q/K/V/O are packed token-major [T, H, D]; the batch mode has
        # stride 0 and every work tile re-bases its tensors at cu_seqlens[b]
        # (domain_offset), so the global-memory partitions are built per tile.
        q_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        kv_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        # Row coordinates of O stay request-local: the epilogue predicates
        # row < seqlen_b, so rows of a partial tile never touch the next request.
        cO_qdl = cute.flat_divide(
            cute.make_identity_tensor(mO_qdl.shape),
            cute.select(self.pv_block_tiler, mode=[0, 1]),
        )
        gLSE_ql = None
        tile_sched = None
        clc_state = None
        clc_resp = None
        clc_map = None
        if cutlass.const_expr(self.sched == "clc"):
            clc_resp = storage.clc_response.data_ptr()
            clc_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.clc_stages
            )
            m_cta_tiles = Int32(tile_sched_params.problem_shape_mhb[0])
            clc_map = (
                m_cta_tiles // 2,
                Int32(tile_sched_params.problem_shape_mhb[1]),
                Int32(tile_sched_params.problem_shape_mhb[2]),
                cutlass.max(Int32(4), Int32(self.clc_budget_div) // m_cta_tiles),
            )
            work_tile = _clc_work_tile(Int32(bidx // 2), Int32(mma_tile_coord_v), clc_map, Int32(1))
        else:
            tile_sched = fmha_utils.create_fmha_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
        # Cluster wait
        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Load
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                batch_coord = curr_block_coord[2][1]
                cu_b = cu_seqlens[batch_coord]
                seqlen_b = cu_seqlens[batch_coord + 1] - cu_b
                cu_kb = cu_seqlens_k[batch_coord]
                seqlen_q = seqlen_b
                seqlen_k = cu_seqlens_k[batch_coord + 1] - cu_kb
                row_lo = None
                if cutlass.const_expr(self.tail_first):
                    cu_b, seqlen_q, row_lo = self._tail_frame(cu_b, seqlen_b, seqlen_k)
                # A 2-CTA tile pair is skipped as a unit (both CTAs see the same
                # mma tile), so every role makes the same decision.
                tile_valid = mma_block_coord[0] * self.qk_mma_tiler[0] < seqlen_q
                if tile_valid:
                    mQ_b = cute.domain_offset((cu_b, 0, ((0, 0), 0)), mQ_qdl)
                    mK_b = cute.domain_offset((cu_kb, 0, ((0, 0), 0)), mK_kdl)
                    mV_b = cute.domain_offset((0, cu_kb, ((0, 0), 0)), mV_dkl)
                    gQ_qdl = cute.flat_divide(mQ_b, cute.select(self.qk_mma_tiler, mode=[0, 2]))
                    tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
                    tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_q,
                        block_in_cluster_coord_vmnk[2],
                        q_cta_layout,
                        cute.group_modes(sQ, 0, 3),
                        cute.group_modes(tSgQ_qdl, 0, 3),
                    )
                    gK_kdl = cute.flat_divide(mK_b, cute.select(self.qk_mma_tiler, mode=[1, 2]))
                    tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
                    tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_k,
                        block_in_cluster_coord_vmnk[1],
                        kv_cta_layout,
                        cute.group_modes(sK, 0, 3),
                        cute.group_modes(tSgK_kdl, 0, 3),
                    )
                    gV_dkl = cute.flat_divide(mV_b, cute.select(self.pv_mma_tiler, mode=[1, 2]))
                    tOgV_dkl = pv_thr_mma.partition_B(gV_dkl)
                    tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_v,
                        block_in_cluster_coord_vmnk[1],
                        kv_cta_layout,
                        cute.group_modes(sV, 0, 3),
                        cute.group_modes(tOgV_dkl, 0, 3),
                    )
                    seqlen_kv_loop_steps = fmha_utils.FusedMask.get_trip_count(
                        self.mask_type,
                        mma_block_coord,
                        self.qk_mma_tiler,
                        seqlen_q,
                        seqlen_k,
                        window_size_left,
                        window_size_right,
                    )
                    # The kv loop covers tiles [start, start + count);
                    # get_trip_count is start-relative.
                    kv_tile_start = fmha_utils.FusedMask.get_trip_start(
                        self.mask_type,
                        mma_block_coord,
                        self.qk_mma_tiler,
                        seqlen_q,
                        seqlen_k,
                        window_size_left,
                        window_size_right,
                    )
                    # ((atom_v, rest_v), RestK)
                    tQgQ = tQgQ_qdl[None, mma_block_coord[0], None, mma_block_coord[2]]
                    # ((atom_v, rest_v), RestN, RestK)
                    tKgK = tKgK_kdl[None, None, None, mma_block_coord[2]]
                    # ((atom_v, rest_v), RestN, RestK)
                    tVgV = tVgV_dkl[None, None, None, mma_block_coord[2]]
                    load_kv_producer, load_q_producer = load_qk_uniform(  # Q & K0
                        self.iterations_qk,
                        kv_step=kv_tile_start,
                        k_args=(tKgK, tKsK, tma_atom_k, load_kv_producer),
                        q_args=(tQgQ, tQsQ, tma_atom_q, load_q_producer),
                    )
                    for step in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
                        load_kv_producer = load_qk_uniform(  # Ki
                            self.iterations_qk,
                            kv_step=kv_tile_start + step,
                            k_args=(tKgK, tKsK, tma_atom_k, load_kv_producer),
                        )
                        load_kv_producer = load_v_uniform(  # Vi-1
                            self.iterations_pv,
                            kv_step=kv_tile_start + step - 1,
                            v_args=(tVgV, tVsV, tma_atom_v, load_kv_producer),
                        )
                    load_kv_producer = load_v_uniform(  # Vend
                        self.iterations_pv,
                        kv_step=kv_tile_start + seqlen_kv_loop_steps - 1,
                        v_args=(tVgV, tVsV, tma_atom_v, load_kv_producer),
                    )
                if cutlass.const_expr(self.sched == "clc"):
                    clc_pipeline.consumer_wait(clc_state)
                    work_tile = _clc_response_tile(
                        clc_resp + clc_state.index, Int32(mma_tile_coord_v), clc_map
                    )
                    clc_pipeline.consumer_release(clc_state)
                    clc_state.advance()
                else:
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
            load_kv_producer.tail()
            load_q_producer.tail()

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem.wait_for_alloc()
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                batch_coord = curr_block_coord[2][1]
                cu_b = cu_seqlens[batch_coord]
                seqlen_b = cu_seqlens[batch_coord + 1] - cu_b
                cu_kb = cu_seqlens_k[batch_coord]
                seqlen_q = seqlen_b
                seqlen_k = cu_seqlens_k[batch_coord + 1] - cu_kb
                row_lo = None
                if cutlass.const_expr(self.tail_first):
                    cu_b, seqlen_q, row_lo = self._tail_frame(cu_b, seqlen_b, seqlen_k)
                # A 2-CTA tile pair is skipped as a unit (both CTAs see the same
                # mma tile), so every role makes the same decision.
                tile_valid = mma_block_coord[0] * self.qk_mma_tiler[0] < seqlen_q
                if tile_valid:
                    seqlen_kv_loop_steps = fmha_utils.FusedMask.get_trip_count(
                        self.mask_type,
                        mma_block_coord,
                        self.qk_mma_tiler,
                        seqlen_q,
                        seqlen_k,
                        window_size_left,
                        window_size_right,
                    )
                    load_q_releaser = load_q_consumer.clone()
                    pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    if seqlen_kv_loop_steps > 1:
                        mma_s_producer, load_q_consumer, load_kv_consumer = mma_qk_uniform(  # QK0
                            self.iterations_qk,
                            qk_tiled_mma,
                            (tStS, tSrQ, tSrK),
                            (
                                mma_s_producer,
                                load_q_consumer,
                                None,
                                load_kv_consumer,
                            ),
                        )
                        for i in cutlass.range(1, seqlen_kv_loop_steps - 1, 1, unroll=1):
                            mma_s_producer, _, load_kv_consumer = mma_qk_uniform(  # QKi
                                self.iterations_qk,
                                qk_tiled_mma,
                                (tStS, tSrQ, tSrK),
                                (mma_s_producer, None, None, load_kv_consumer),
                            )
                            (
                                pv_tiled_mma,
                                p_mma_consumer,
                                mma_o_producer,
                                load_kv_consumer,
                            ) = self.mma_pv(  # PVi
                                (pv_tiled_mma, pv_thr_mma),
                                (tOtO_staged, tStS, tOrV, p_tmem_layout),
                                (p_mma_consumer, mma_o_producer, load_kv_consumer),
                            )
                        mma_s_producer, _, load_kv_consumer = mma_qk_uniform(  # QKend (release Q)
                            self.iterations_qk,
                            qk_tiled_mma,
                            (tStS, tSrQ, tSrK),
                            (
                                mma_s_producer,
                                None,
                                load_q_releaser,
                                load_kv_consumer,
                            ),
                        )
                        (
                            pv_tiled_mma,
                            p_mma_consumer,
                            mma_o_producer,
                            load_kv_consumer,
                        ) = self.mma_pv(  # PVend-1
                            (pv_tiled_mma, pv_thr_mma),
                            (tOtO_staged, tStS, tOrV, p_tmem_layout),
                            (p_mma_consumer, mma_o_producer, load_kv_consumer),
                        )
                    else:
                        mma_s_producer, load_q_consumer, load_kv_consumer = mma_qk_uniform(  # QK0
                            self.iterations_qk,
                            qk_tiled_mma,
                            (tStS, tSrQ, tSrK),
                            (
                                mma_s_producer,
                                load_q_consumer,
                                load_q_releaser,
                                load_kv_consumer,
                            ),
                        )
                    pv_tiled_mma, p_mma_consumer, mma_o_producer, load_kv_consumer = (
                        self.mma_pv(  # PVend
                            (pv_tiled_mma, pv_thr_mma),
                            (tOtO_staged, tStS, tOrV, p_tmem_layout),
                            (p_mma_consumer, mma_o_producer, load_kv_consumer),
                        )
                    )
                if cutlass.const_expr(self.sched == "clc"):
                    clc_pipeline.consumer_wait(clc_state)
                    work_tile = _clc_response_tile(
                        clc_resp + clc_state.index, Int32(mma_tile_coord_v), clc_map
                    )
                    clc_pipeline.consumer_release(clc_state)
                    clc_state.advance()
                else:
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
            mma_s_producer.tail()
            mma_o_producer.tail()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx < self.correction_warp_ids[0] and warp_idx >= self.softmax_warp_ids[0]:
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            tmem.wait_for_alloc()
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                batch_coord = curr_block_coord[2][1]
                cu_b = cu_seqlens[batch_coord]
                seqlen_b = cu_seqlens[batch_coord + 1] - cu_b
                cu_kb = cu_seqlens_k[batch_coord]
                seqlen_q = seqlen_b
                seqlen_k = cu_seqlens_k[batch_coord + 1] - cu_kb
                row_lo = None
                if cutlass.const_expr(self.tail_first):
                    cu_b, seqlen_q, row_lo = self._tail_frame(cu_b, seqlen_b, seqlen_k)
                # A 2-CTA tile pair is skipped as a unit (both CTAs see the same
                # mma tile), so every role makes the same decision.
                tile_valid = mma_block_coord[0] * self.qk_mma_tiler[0] < seqlen_q
                if tile_valid:
                    seqlen_kv_loop_steps = fmha_utils.FusedMask.get_trip_count(
                        self.mask_type,
                        mma_block_coord,
                        self.qk_mma_tiler,
                        seqlen_q,
                        seqlen_k,
                        window_size_left,
                        window_size_right,
                    )
                    # A left window clips the FIRST visited tile too, so the loop
                    # is [ld_mask][unmask][tl_mask] (fmha.py's structure).
                    (
                        kv_tile_start,
                        _end_cnt,
                        ld_mask_steps,
                        unmask_steps,
                        _tl_mask_steps,
                    ) = fmha_utils.FusedMask.get_masked_info(
                        self.mask_type,
                        mma_block_coord,
                        self.qk_mma_tiler,
                        seqlen_q,
                        seqlen_k,
                        window_size_left,
                        window_size_right,
                    )
                    cS_base = cute.make_identity_tensor(
                        (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
                    )
                    cS = cute.domain_offset((mma_block_coord[0] * self.qk_mma_tiler[0], 0), cS_base)
                    tScS = qk_thr_mma.partition_C(cS)
                    tScS_static = qk_thr_mma.partition_C(cS_base)
                    # FP8: this head's dequantized softmax scale (s_q * s_k[h] folded in)
                    sm_scale = scale_softmax_log2
                    if cutlass.const_expr(head_scales is not None):
                        sm_scale = head_scales[0, curr_block_coord[2][0]]
                    row_max = -Float32.inf
                    row_max_prev = -Float32.inf
                    row_sum = 0.0
                    for step in cutlass.range(seqlen_kv_loop_steps, unroll=1):
                        # Mask coordinates must be ABSOLUTE kv positions.
                        cS_iter = cute.domain_offset(
                            (0, (kv_tile_start + step) * self.qk_mma_tiler[1]), cS
                        )
                        tScS_iter = qk_thr_mma.partition_C(cS_iter)
                        # Mask unless step is inside the clean middle segment.
                        # Unsigned-wrap trick: (step - ld) as Uint32 is huge for
                        # step < ld, so one compare covers both masked edges.
                        need_apply_mask = cutlass.Uint32(step - ld_mask_steps) >= cutlass.Uint32(
                            unmask_steps
                        )
                        # Si -> Pi
                        (
                            row_max,
                            row_sum,
                            mma_s_consumer,
                            p_mma_producer,
                            s_corr_producer,
                        ) = self.softmax_step(
                            (need_apply_mask, window_size_left, window_size_right),
                            (
                                row_max_prev,
                                row_sum,
                                seqlen_q,
                                seqlen_k,
                                sm_scale,
                            ),
                            (tStS, tScS_iter, tScS_static),
                            (mma_s_consumer, p_mma_producer, s_corr_producer),
                        )
                        row_max_prev = row_max
                    sum_producer = self.store_sum(row_sum, row_max_prev, sSum, sMax, sum_producer)
                if cutlass.const_expr(self.sched == "clc"):
                    clc_pipeline.consumer_wait(clc_state)
                    work_tile = _clc_response_tile(
                        clc_resp + clc_state.index, Int32(mma_tile_coord_v), clc_map
                    )
                    clc_pipeline.consumer_release(clc_state)
                    clc_state.advance()
                else:
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
            p_mma_producer.tail()
            s_corr_producer.tail()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx < self.mma_warp_id and warp_idx >= self.correction_warp_ids[0]:
            # With the reduced warp count the launch default regs/thread is higher
            # than num_regs_correction, so this is a decrease (cf. base fmha.py).
            if cutlass.const_expr(self.epi_offload):
                cute.arch.setmaxregister_increase(self.num_regs_correction_offload)
            elif cutlass.const_expr(self.epi_fast and mQ8_qdl is not None):
                cute.arch.setmaxregister_increase(self.num_regs_correction_epi)
            else:
                cute.arch.setmaxregister_decrease(self.num_regs_correction)
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                batch_coord = curr_block_coord[2][1]
                cu_b = cu_seqlens[batch_coord]
                seqlen_b = cu_seqlens[batch_coord + 1] - cu_b
                cu_kb = cu_seqlens_k[batch_coord]
                seqlen_q = seqlen_b
                seqlen_k = cu_seqlens_k[batch_coord + 1] - cu_kb
                row_lo = None
                if cutlass.const_expr(self.tail_first):
                    cu_b, seqlen_q, row_lo = self._tail_frame(cu_b, seqlen_b, seqlen_k)
                # A 2-CTA tile pair is skipped as a unit (both CTAs see the same
                # mma tile), so every role makes the same decision.
                tile_valid = mma_block_coord[0] * self.qk_mma_tiler[0] < seqlen_q
                if tile_valid:
                    mO_b = cute.domain_offset((cu_b, 0, ((0, 0), 0)), mO_qdl)
                    gO_qdl = cute.flat_divide(mO_b, cute.select(self.pv_block_tiler, mode=[0, 1]))
                    seqlen_kv_loop_steps = fmha_utils.FusedMask.get_trip_count(
                        self.mask_type,
                        mma_block_coord,
                        self.qk_mma_tiler,
                        seqlen_q,
                        seqlen_k,
                        window_size_left,
                        window_size_right,
                    )
                    gO_staged = gO_qdl[None, None, curr_block_coord[0], None, curr_block_coord[2]]
                    cO_staged = cO_qdl[None, None, curr_block_coord[0], None, curr_block_coord[2]]
                    gQ8_staged = None
                    if cutlass.const_expr(mQ8_qdl is not None):
                        mQ8_b = cute.domain_offset((cu_b, 0, ((0, 0), 0)), mQ8_qdl)
                        gQ8_qdl = cute.flat_divide(
                            mQ8_b, cute.select(self.pv_block_tiler, mode=[0, 1])
                        )
                        gQ8_staged = gQ8_qdl[
                            None, None, curr_block_coord[0], None, curr_block_coord[2]
                        ]
                    cS = cute.make_identity_tensor((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
                    tScS = qk_thr_mma.partition_C(cS)
                    # FP8: this head's dequantized softmax / output scales
                    sm_scale = scale_softmax_log2
                    out_scale = scale_output
                    if cutlass.const_expr(head_scales is not None):
                        sm_scale = head_scales[0, curr_block_coord[2][0]]
                        out_scale = head_scales[1, curr_block_coord[2][0]]
                    # Empty step as the first step is no need for correction
                    stats_handle = s_corr_consumer.wait_and_advance()
                    stats_handle.release()
                    for step in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
                        # Oi-1 -> Oi
                        mma_o_consumer, s_corr_consumer = self.correction_rescale(
                            sm_scale,
                            (s_corr_consumer, tStS, tScS),
                            (mma_o_consumer, tOtO_staged, cO_staged),
                            epi_tile,
                        )
                    # O_partial -> O_final
                    gLSE_staged = None
                    if cutlass.const_expr(gLSE_ql is not None):
                        gLSE_staged = gLSE_ql[None, curr_block_coord[0], curr_block_coord[2]]
                    epi_values = (seqlen_q, out_scale, sm_scale)
                    if cutlass.const_expr(row_lo is not None):
                        epi_values = (seqlen_q, out_scale, sm_scale, row_lo)
                    if cutlass.const_expr(self.epi_offload):
                        mma_o_consumer, sum_consumer, epi_producer = self.correction_epilog_stage(
                            epi_values,
                            (sum_consumer, sSum),
                            (mma_o_consumer, cO_staged, tOtO_staged),
                            epi_tile,
                            (sStage, epi_producer),
                            (cu_b, curr_block_coord[2][0]),
                        )
                    elif cutlass.const_expr(mQ8_qdl is None and row_lo is None):
                        mma_o_consumer, sum_consumer = self.correction_epilog(
                            epi_values,
                            (sum_consumer, sSum, sMax),
                            (
                                mma_o_consumer,
                                gO_staged,
                                cO_staged,
                                tOtO_staged,
                                gLSE_staged,
                            ),
                            epi_tile,
                        )
                    elif cutlass.const_expr(mQ8_qdl is None):
                        mma_o_consumer, sum_consumer = self.correction_epilog_tail(
                            epi_values,
                            (sum_consumer, sSum, sMax),
                            (
                                mma_o_consumer,
                                gO_staged,
                                cO_staged,
                                tOtO_staged,
                                gLSE_staged,
                            ),
                            epi_tile,
                        )
                    else:
                        mma_o_consumer, sum_consumer = self.correction_epilog_q8(
                            epi_values,
                            (sum_consumer, sSum, sMax),
                            (
                                mma_o_consumer,
                                gO_staged,
                                cO_staged,
                                tOtO_staged,
                                gLSE_staged,
                            ),
                            epi_tile,
                            (gQ8_staged, mSF8.iterator, cu_b, curr_block_coord[2][0]),
                        )
                if cutlass.const_expr(self.sched == "clc"):
                    clc_pipeline.consumer_wait(clc_state)
                    work_tile = _clc_response_tile(
                        clc_resp + clc_state.index, Int32(mma_tile_coord_v), clc_map
                    )
                    clc_pipeline.consumer_release(clc_state)
                    clc_state.advance()
                else:
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
            if cutlass.const_expr(self.epi_offload):
                self.epi_signal_end(sStage, epi_producer)
            tmem.relinquish_alloc_permit()
            # Synchronize before TMEM dealloc (done by the caller)
            tmem_dealloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  CLC scheduler (leader CTA) / empty
        # ///////////////////////////////////////////////////////////////////////////////
        if cutlass.const_expr(self.sched == "clc"):
            if warp_idx == self.sched_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_other)
                if cta_rank_in_cluster == 0:
                    clc_prod = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.ProducerConsumer, self.clc_stages
                    )
                    while work_tile.is_valid_tile:
                        clc_pipeline.producer_acquire(clc_prod)
                        clc_mbar = clc_pipeline.producer_get_barrier(clc_prod)
                        with cute.arch.elect_one():
                            cute.arch.issue_clc_query(clc_mbar, clc_resp + clc_prod.index)
                        clc_prod.advance()
                        clc_pipeline.consumer_wait(clc_state)
                        work_tile = _clc_response_tile(
                            clc_resp + clc_state.index, Int32(mma_tile_coord_v), clc_map
                        )
                        clc_pipeline.consumer_release(clc_state)
                        clc_state.advance()
                    clc_pipeline.producer_tail(clc_prod)
            if cutlass.const_expr(self.epi_offload):
                if warp_idx > self.sched_warp_id and warp_idx < self.epi_warp_ids[0]:
                    cute.arch.setmaxregister_decrease(self.num_regs_other)
            else:
                if warp_idx > self.sched_warp_id:
                    cute.arch.setmaxregister_decrease(self.num_regs_other)
        else:
            if warp_idx > self.load_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_other)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue warps (epi_offload): quantize / store the staged O rows
        # ///////////////////////////////////////////////////////////////////////////////
        if cutlass.const_expr(self.epi_offload):
            if warp_idx >= self.epi_warp_ids[0]:
                cute.arch.setmaxregister_decrease(self.num_regs_epi)
                state = Int32(1)
                while state != Int32(0):
                    epi_consumer, state = self.epi_rows(
                        sStage, epi_consumer, (mQ8_qdl, mO_qdl, mSF8)
                    )

        return


class GlmPrefillMhaVarlenD256Fp8(GlmPrefillMhaVarlenD256):
    """E4M3 Q/K/V, BF16 O, per-head dequantization factors ``head_scales`` (module doc)."""

    def __init__(self, **variant) -> None:
        super().__init__(fp8=True, **variant)

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        cu_seqlens: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        max_seqlen: Int32,
        head_scales: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """As the BF16 launch; ``head_scales`` FP32 ``[2, H]``: row 0 the softmax
        ``scale_softmax_log2`` per head, row 1 the output scale per head."""
        self._launch(
            q,
            k,
            v,
            o,
            cu_seqlens,
            cu_seqlens_k,
            max_seqlen,
            Float32(0.0),
            Float32(0.0),
            head_scales,
            stream,
            None,
            None,
        )


class GlmPrefillMhaVarlenD256Fp8Q8(GlmPrefillMhaVarlenD256):
    """The FP8 variant that also writes O as the o_proj MXF8 operand (``correction_epilog_q8``):
    ``q8`` uint8 ``[T_o, H * 256]`` (``T_o >= T_q``; rows ``>= T_q`` are not touched) and
    ``sf8`` its blocked e8m0 scales (``self_attn_prefill_act_quant_mxf8`` layout; the scale rows
    ``>= T_q``, padding included, are the caller's -- ``quantize_rows``)."""

    def __init__(self, store_o: bool = True, **variant) -> None:
        """``store_o=False``: the BF16 O rows are not written (the operand is O's only
        reader); ``o`` is then only the row-count / layout carrier."""
        super().__init__(fp8=True, **variant)
        self.q8_store_o = bool(store_o)

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        cu_seqlens: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        max_seqlen: Int32,
        head_scales: cute.Tensor,
        q8: cute.Tensor,
        sf8: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self._launch(
            q,
            k,
            v,
            o,
            cu_seqlens,
            cu_seqlens_k,
            max_seqlen,
            Float32(0.0),
            Float32(0.0),
            head_scales,
            stream,
            q8,
            sf8,
        )


def _row_stored(row, seqlen_q, row_lo):
    """Epilogue row predicate in the request frame: ``row < seqlen_q``, and with ``row_lo`` (the
    tail-first frame's first stored row) also ``row >= row_lo``, as one SIGNED compare
    ``min(row - row_lo, seqlen_q - 1 - row) >= 0``. (An ``Uint32(row - row_lo) < Uint32(...)``
    range trick is lowered as a signed compare here and lets the frame's leading rows through:
    a debug run caught their stores below the request start.)"""
    if row_lo is None:
        return cute.elem_less(row, seqlen_q)
    return cutlass.min(row - row_lo, seqlen_q - 1 - row) >= 0


@dsl_user_op
def _cvt_bf16x2_rn(hi: Float32, lo: Float32, *, loc=None, ip=None) -> Int32:
    """Round two FP32 values to BF16 (RN) with one packed ``cvt.rn.bf16x2.f32`` (ALU pipe; the
    scalar ``F2F.BF16.F32`` runs on the XU pipe the softmax exp2 shares): ``hi`` in bits
    31..16, ``lo`` in bits 15..0."""
    v = _nvvm.convert_f32x2_to_bf16x2(
        _mlir_ir.VectorType.get([2], cutlass.BFloat16.mlir_type, loc=loc),
        Float32(hi).ir_value(loc=loc, ip=ip),
        Float32(lo).ir_value(loc=loc, ip=ip),
        rnd=_nvvm.FPRoundingMode.RN,
        loc=loc,
        ip=ip,
    )
    return Int32(_llvm.bitcast(_T.i32(), v, loc=loc, ip=ip))


def _rmem_list(n: int, shape, dtype) -> list:
    """``n`` register tensors of ``shape`` (trace-time Python, no device loop)."""
    return [cute.make_rmem_tensor(shape, dtype) for _ in range(n)]


def _absmax32(t: cute.Tensor) -> Float32:
    """max |t[j]| over 32 FP32 registers as a balanced fmax tree (trace-time Python); equal to
    the compare-select chain's value (|x| >= 0, and fmax ignores a NaN as ``a > m`` does)."""
    vals = [cute.math.absf(t[j]) for j in range(32)]
    while len(vals) > 1:
        vals = [cute.arch.fmax(vals[2 * j], vals[2 * j + 1]) for j in range(len(vals) // 2)]
    return vals[0]


@cute.jit
def _clc_work_tile(cluster_idx: Int32, cta_v: Int32, clc_map, valid: Int32):
    """Work tile of launch-order cluster ``cluster_idx`` (CLC order).

    ``clc_map = (m_pairs, heads, batch, chunk)``. The (head, request) groups are cut into
    chunks of ``chunk`` groups (K/V locality); inside a chunk the q-tile pairs run from the
    last (longest causal kv loop) to the first, every group of the chunk per level. Launch /
    claim order then approximates longest-processing-time-first list scheduling, so the
    tail of the kernel is made of the short (first) q-tiles. Tiles past a request's length
    stay in the order and are skipped by every role (``tile_valid``)."""
    mp, n_heads, n_batch, chunk = clc_map
    ng = n_heads * n_batch
    per_chunk = chunk * mp
    ch = cluster_idx // per_chunk
    i = cluster_idx - ch * per_chunk
    gc = cutlass.max(Int32(1), cutlass.min(chunk, ng - ch * chunk))
    m_pair = mp - Int32(1) - i // gc
    g = ch * chunk + i % gc
    # Same shape as FmhaStaticTileScheduler's tile: (m, 0, (h, b)) with a Python 0, i.e. 3 IR
    # values + valid. In serving, ``helpers`` may come from another CuTeDSL tree loaded first
    # (the pinned dense-GEMM root), whose WorkTileInfo rebuilds scf regions from exactly 4 IR
    # values; an Int32(0) would make 5 and fail there.
    return fmha_utils.WorkTileInfo(
        (m_pair * 2 + cta_v, 0, (g % n_heads, g // n_heads)), valid != Int32(0)
    )


@cute.jit
def _clc_response_tile(resp_ptr, cta_v: Int32, clc_map):
    """Decode one multicast CLC response (first CTA id of the claimed cluster)."""
    x, _y, _z, vld = cute.arch.clc_response(resp_ptr)
    # the response is written through the async proxy and read through the generic one
    cute.arch.fence_proxy("async.shared", space="cta")
    return _clc_work_tile(x // 2, cta_v, clc_map, vld)


def _tvm_ffi_tensor(t: torch.Tensor):
    return from_dlpack(t, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(
        leading_dim=t.dim() - 1
    )


def fp8_head_scales(s_q: float, s_k: torch.Tensor, s_v: torch.Tensor) -> torch.Tensor:
    """The FP8 variant's ``head_scales`` [2, H] FP32: row 0 = log2(e) * SOFTMAX_SCALE * s_q *
    s_k[h] (the dequantized QK scale in exp2 units), row 1 = s_v[h] (the dequantized O scale)."""
    s_k = s_k.detach().float().reshape(-1)
    s_v = s_v.detach().float().reshape(-1).to(s_k.device)
    if s_k.numel() != NUM_HEADS or s_v.numel() != NUM_HEADS:
        raise ValueError(f"s_k / s_v must have {NUM_HEADS} entries")
    return torch.stack([s_k * (math.log2(math.e) * SOFTMAX_SCALE * float(s_q)), s_v]).contiguous()


class PrefillMha:
    """Launcher: ``__call__(q, k, v, o, cu_seqlens, max_seqlen, cu_seqlens_k=None)``; one compile
    serves every shape (q and K/V row counts are independent symbols). ``fp8=True``: E4M3 q/k/v,
    BF16 o, and ``head_scales`` (``fp8_head_scales``) on every call."""

    def __init__(
        self,
        device: int,
        fp8: bool = False,
        q8: bool = False,
        q8_store_o: bool = True,
        **variant,
    ) -> None:
        """``variant``: kernel-class options (``sched``, ``rescale_threshold``, the MHA epilogue options
        ``cvt4`` / ``epi_fast``, ``tail_first`` / ``tail_align``, ``epi_offload``); any MHA
        epilogue option adds ``_sol`` to the kernel symbol, ``tail_first`` adds ``_tail``
        (``_tailraw`` with ``tail_align=False``), ``epi_offload`` (q8 launchers) adds ``_epio``."""
        dev = torch.device("cuda", device)
        self.device = dev
        self.fp8 = bool(fp8)
        # q8: the FP8 variant that also writes O as the o_proj MXF8 operand (``q8_out``)
        self.q8 = bool(q8)
        if self.q8 and not self.fp8:
            raise ValueError("the MXF8 O epilogue is an FP8-variant option")
        if variant.get("epi_offload") and not self.q8:
            raise ValueError("epi_offload is an MXF8-O (q8) launcher option")
        if self.q8:
            fmha = GlmPrefillMhaVarlenD256Fp8Q8(store_o=q8_store_o, **variant)
        elif self.fp8:
            fmha = GlmPrefillMhaVarlenD256Fp8(**variant)
        else:
            fmha = GlmPrefillMhaVarlenD256(**variant)
        fmha.kernel.set_name_prefix(
            KERNEL_NAME_PREFIX
            + (
                ("_fp8q8" if q8_store_o else "_fp8q8only")
                if self.q8
                else "_fp8"
                if self.fp8
                else ""
            )
            + ("_sol" if (variant.get("cvt4") or variant.get("epi_fast")) else "")
            + (
                ("_tail" if variant.get("tail_align", True) else "_tailraw")
                if variant.get("tail_first")
                else ""
            )
            + ("_epio" if variant.get("epi_offload") else "")
            + ("_kvs" if variant.get("kv_strided") else ""),
            remove_cutlass_symbol=True,
        )
        # K / V rows at their own strides (the halves of one K | V row), not packed
        self.kv_strided = bool(variant.get("kv_strided"))
        self.in_dtype = torch.float8_e4m3fn if self.fp8 else torch.bfloat16
        # Distinct placeholder tensors: TVM-FFI ties the shape symbols of one tensor object
        # passed twice, and q / o rows (T_q) differ from k / v rows (T_k) with a cached prefix.
        q = torch.empty(256, NUM_HEADS, HEAD_DIM, dtype=self.in_dtype, device=dev)
        k = torch.empty(384, NUM_HEADS, HEAD_DIM, dtype=self.in_dtype, device=dev)
        v = torch.empty(384, NUM_HEADS, HEAD_DIM, dtype=self.in_dtype, device=dev)
        if self.kv_strided:
            # the served operand: K and V the two halves of one [T_k, 2 * H * 256] row
            kv = torch.empty(384, 2, NUM_HEADS, HEAD_DIM, dtype=self.in_dtype, device=dev)
            k, v = kv[:, 0], kv[:, 1]
        o = torch.empty(256, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        cu = torch.tensor([0, 256], dtype=torch.int32, device=dev)
        cu_k = torch.tensor([0, 384], dtype=torch.int32, device=dev)
        self.scale_log2 = SOFTMAX_SCALE * math.log2(math.e)
        head_args = (Float32(self.scale_log2), Float32(1.0))
        if self.fp8:
            hs = torch.ones(2, NUM_HEADS, dtype=torch.float32, device=dev)
            head_args = (from_dlpack(hs, assumed_align=16, enable_tvm_ffi=True),)
        if self.q8:
            q8 = torch.empty(320, NUM_HEADS * HEAD_DIM, dtype=torch.uint8, device=dev)
            sf8 = torch.empty(
                _mxf8.sf_bytes(320, NUM_HEADS * HEAD_DIM), dtype=torch.uint8, device=dev
            )
            head_args = head_args + (
                _tvm_ffi_tensor(q8),
                from_dlpack(sf8, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(),
            )
        self.compiled = cute.compile(
            fmha,
            _tvm_ffi_tensor(q),
            _tvm_ffi_tensor(k),
            _tvm_ffi_tensor(v),
            _tvm_ffi_tensor(o),
            from_dlpack(cu, assumed_align=4, enable_tvm_ffi=True).mark_layout_dynamic(),
            from_dlpack(cu_k, assumed_align=4, enable_tvm_ffi=True).mark_layout_dynamic(),
            Int32(256),
            *head_args,
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    def check(self, q, k, v, o, cu_seqlens, cu_seqlens_k=None, head_scales=None) -> None:
        if self.fp8 and (
            head_scales is None
            or head_scales.dtype != torch.float32
            or tuple(head_scales.shape) != (2, NUM_HEADS)
            or not head_scales.is_contiguous()
            or head_scales.device != q.device
        ):
            raise ValueError(
                f"the FP8 launcher needs head_scales FP32 [2, {NUM_HEADS}] on the q device"
            )
        for name, t in (("q", q), ("k", k), ("v", v), ("o", o)):
            want = torch.bfloat16 if name == "o" else self.in_dtype
            # kv_strided: K / V rows at any stride that is a multiple of 256 elements, past
            # one row's H * 256 (the head stride and the inner stride stay 256 / 1)
            layout_ok = t.is_contiguous() or (
                self.kv_strided
                and name in ("k", "v")
                and t.dim() == 3
                and t.stride(1) == HEAD_DIM
                and t.stride(2) == 1
                and t.stride(0) % 256 == 0
                and t.stride(0) >= NUM_HEADS * HEAD_DIM
            )
            if (
                t.dtype != want
                or t.dim() != 3
                or tuple(t.shape[1:]) != (NUM_HEADS, HEAD_DIM)
                or not layout_ok
            ):
                raise ValueError(
                    f"{name} must be contiguous {want} [T, {NUM_HEADS}, {HEAD_DIM}]"
                    + (" (k / v: or row-strided)" if self.kv_strided else "")
                    + f", got {t.dtype} {tuple(t.shape)} {t.stride()}"
                )
            if t.data_ptr() % 16:
                raise ValueError(f"{name} must be 16-byte aligned")
        if q.shape[0] != o.shape[0] or k.shape[0] != v.shape[0]:
            raise ValueError("q/o or k/v row counts differ")
        if cu_seqlens_k is None and q.shape[0] != k.shape[0]:
            raise ValueError("k/v rows differ from q rows: cu_seqlens_k is required")
        for name, cu in (("cu_seqlens", cu_seqlens), ("cu_seqlens_k", cu_seqlens_k)):
            if cu is not None and (cu.dtype != torch.int32 or cu.dim() != 1 or cu.numel() < 2):
                raise ValueError(f"{name} must be int32 [B + 1]")
        if cu_seqlens_k is not None and cu_seqlens_k.numel() != cu_seqlens.numel():
            raise ValueError("cu_seqlens and cu_seqlens_k must have the same B + 1 entries")

    def __call__(
        self,
        q,
        k,
        v,
        o,
        cu_seqlens,
        max_seqlen: int,
        validate: bool = False,
        cu_seqlens_k=None,
        head_scales=None,
        q8_out=None,
    ):
        """``max_seqlen``: the longest query (new-row) count. ``cu_seqlens_k`` (K/V rows per
        request, cached prefix first; ``kv_len >= q_len``) defaults to ``cu_seqlens``.

        Contract (checked only with ``validate=True``, which syncs the device):
        ``cu_seqlens`` starts at 0, is nondecreasing and ends at ``q.shape[0]``;
        ``cu_seqlens_k`` likewise ends at ``k.shape[0]``; ``max_seqlen`` is at least
        the longest request. Offsets past the tensors read or write out of bounds,
        and a short ``max_seqlen`` leaves the tail tiles of ``o`` unwritten. The
        serving route builds all three from one host prefix sum.
        ``head_scales``: FP8 launcher only (``fp8_head_scales``). ``q8_out = (q8, sf8)``: the
        q8 launcher's o_proj MXF8 operand (``q8 [T_o >= T_q, H * 256]`` uint8 contiguous,
        ``sf8`` >= ``sf_bytes(T_o, H * 256)`` bytes); rows ``< T_q`` are written."""
        if validate:
            self.check(q, k, v, o, cu_seqlens, cu_seqlens_k, head_scales)
        cu_k = cu_seqlens if cu_seqlens_k is None else cu_seqlens_k
        if self.q8:
            q8, sf8 = q8_out
            if (
                q8.dtype != torch.uint8
                or q8.dim() != 2
                or q8.shape[1] != NUM_HEADS * HEAD_DIM
                or q8.shape[0] < q.shape[0]
                or not q8.is_contiguous()
                or q8.data_ptr() % 16
                or sf8.dtype != torch.uint8
                or not sf8.is_contiguous()
                or sf8.numel() < _mxf8.sf_bytes(q8.shape[0], NUM_HEADS * HEAD_DIM)
            ):
                raise ValueError(
                    "q8_out must be uint8 [T_o >= T_q, 4096] + >= sf_bytes scale bytes"
                )
            self.compiled(q, k, v, o, cu_seqlens, cu_k, int(max_seqlen), head_scales, q8, sf8)
            return o
        if self.fp8:
            self.compiled(q, k, v, o, cu_seqlens, cu_k, int(max_seqlen), head_scales)
        else:
            self.compiled(q, k, v, o, cu_seqlens, cu_k, int(max_seqlen), self.scale_log2, 1.0)
        return o


_BUILT: dict[tuple, PrefillMha] = {}


def build(
    device: int | None = None,
    fp8: bool = False,
    q8: bool = False,
    q8_store_o: bool = True,
    variant: dict | None = None,
) -> PrefillMha:
    """Registry entry: one compiled launcher per device (and per BF16 / FP8 / FP8 + MXF8-O
    variant; ``q8_store_o=False``: MXF8 O only, no BF16 O rows). ``variant``: kernel-class
    options (e.g. the MHA epilogue options ``{"cvt4": True, "epi_fast": True}``, ``tail_first``,
    ``kv_strided``); same output bytes."""
    d = torch.cuda.current_device() if device is None else int(device)
    v = dict(variant or {})
    key = (d, bool(fp8), bool(q8), bool(q8_store_o) or not q8) + (
        (tuple(sorted(v.items())),) if v else ()
    )
    if key not in _BUILT:
        _BUILT[key] = PrefillMha(d, fp8=bool(fp8), q8=bool(q8), q8_store_o=bool(q8_store_o), **v)
    return _BUILT[key]


def reference(q, k, v, cu_seqlens, cu_seqlens_k=None) -> torch.Tensor:
    """FP32 causal attention per request (scale 1/16), bottom-right aligned: query i of a
    request with ``c = kv_len - q_len`` cached rows attends K/V positions ``<= c + i``."""
    cu = [int(x) for x in cu_seqlens.tolist()]
    cuk = cu if cu_seqlens_k is None else [int(x) for x in cu_seqlens_k.tolist()]
    out = torch.zeros(q.shape[0], q.shape[1], v.shape[2], dtype=torch.float32, device=q.device)
    for a, b, ka, kb in zip(cu[:-1], cu[1:], cuk[:-1], cuk[1:]):
        if b <= a:
            continue
        n = b - a
        c = (kb - ka) - n
        for s0 in range(0, n, 1024):
            s1 = min(n, s0 + 1024)
            s = (
                torch.einsum(
                    "qhd,khd->hqk",
                    q[a + s0 : a + s1].float(),
                    k[ka : ka + c + s1].float(),
                )
                * SOFTMAX_SCALE
            )
            qi = torch.arange(c + s0, c + s1, device=q.device)[:, None]
            ki = torch.arange(0, c + s1, device=q.device)[None, :]
            s = s.masked_fill((ki > qi)[None], float("-inf"))
            out[a + s0 : a + s1] = torch.einsum(
                "hqk,khd->qhd", torch.softmax(s, -1), v[ka : ka + c + s1].float()
            )
    return out

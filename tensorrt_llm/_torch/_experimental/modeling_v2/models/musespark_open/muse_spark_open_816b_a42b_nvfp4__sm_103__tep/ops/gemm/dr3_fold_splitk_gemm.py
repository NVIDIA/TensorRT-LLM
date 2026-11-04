# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The drafter's split-K GEMVs with an elementwise epilogue: flashinfer's cluster split-K GEMM, subclassed.

Two kernels, each the GEMM that precedes an elementwise kernel of the DFlash 2 drafter, with that kernel's
statements applied by the owner CTA to its reduced tile before the single store:

* ``conv_splitk_gemm`` (CF): the convolution coefficient projection (``kernel_projection``, ``[2 * taps *
  groups, hidden]``) followed by the prepare side of the block convolution (``ops/spec/dflash2_block_conv``).
  The weight's side-0 rows are permuted per kernel-M tile (``permute_conv_rows``): tile ``t`` of ``mma_m`` rows
  holds the tap-0 rows of groups ``t * h ... t * h + h - 1`` followed by the same groups' tap-1 rows,
  ``h = mma_m / 2``; the side-1 rows (the coefficients the drafter hop's ``finish`` reads) stay in place. A
  side-0 tile's owner then holds both taps of ``h`` groups, i.e. every coefficient of ``h * group_size``
  hidden columns, and writes the convolution of those columns. A side-1 tile stores its coefficients as the
  plain kernel does.
* ``silu_splitk_gemm`` (SF): ``gate_up_proj`` followed by ``silu(gate) * up``, on a weight whose gate / up
  rows are interleaved per kernel-M tile (``interleave_gate_up``: ``h`` gate rows, then the matching ``h`` up
  rows), for any ``mma_m`` in (64, 128) and ``mma_n`` in (8, 16), with or without a split-K reduction.

A row permutation of the weight changes no output element's K-ordered sum, so the reduced, BF16-rounded GEMM
values are the plain kernel's for the original weight, column for column. Everything up to that rounding is
the parent kernel's: the DMA warps (the weight's TMA loads before ``griddepcontrol.wait``, the activation's
after it, the dependents released after the weight's last load), the MMA warp, the TMEM accumulator, the
DSMEM split-K reduction into the cluster's rank-0 CTA and the launch. The owner's epilogue then rounds the
reduced FP32 tile to BF16, stages it widened back to FP32 (exact) in shared memory with the parent's TMEM
partition, synchronizes its 128 epilogue threads on a named barrier, and applies the elementwise kernel's
statements as inline PTX with that kernel's rounding points, storing BF16 with ``cvt.rn``. So each output is
bitwise the plain GEMM followed by the elementwise kernel.

CF's epilogue also reads the GEMM's own activation rows (the convolution's ``x``) and the base kernel. Its
epilogue warps execute ``griddepcontrol.wait`` before those loads (the activation's producer is the GEMM's
programmatic predecessor) and issue them before waiting for the MMA, so they are in registers when the tile is.

Only the no-bias ``run_splitk_dense`` layouts at public ``M <= mma_n`` are served. The compiled kernels are
cached per dtype, tactic, PDL flag, operand layout (and CF's block size), as flashinfer caches its own.

``entry_trigger`` (``kernels/dr3_folds.GEMV_ENTRY_TRIGGER``, off in every composition): the
weight DMA warp releases the GEMV's programmatic dependents right after it has issued its first ring of weight
loads (``min(stages, k_tiles)`` TMAs) instead of after its last one. The dependent (the next projection GEMV)
then becomes resident on the SMs this GEMV leaves free while this GEMV still streams, and fills its own weight
ring there. Nothing else moves: the dependent reads this GEMV's outputs only after its own
``griddepcontrol.wait``, i.e. after this grid completed, and it launches only after every CTA of this grid
triggered (so every CTA of this grid is resident and none can be held back). Value-identical by construction:
no instruction of the arithmetic changes.
"""

from __future__ import annotations

import functools
from typing import Tuple

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as _cute_ext
from cutlass.cutlass_dsl import T, dsl_user_op
from flashinfer.gemm.kernels import dense_bf16_gemm_sm100_splitk as _splitk

#: Named barrier of the owner CTA's 128 epilogue threads around the staged tile (id 0 is the CTA-wide barrier).
STAGE_BARRIER_ID = 1

#: The parent kernel's epilogue threads (warps 4-7).
_EPILOG_THREADS = 128

#: The drafter's block convolution geometry (``ops/spec/dflash2_block_conv``): two taps, 16 channels per group.
CONV_TAPS, CONV_GROUP = 2, 16

#: CF's spread epilogue (``spread=True``): a warp's lanes, the consecutive columns per lane, and the rows per
#: epilogue warp of the 8-row tile (four epilogue warps).
_SPREAD_LANES, _SPREAD_COLS, _SPREAD_ROWS = 32, 4, 2

# ---- inline PTX ------------------------------------------------------------------------------------------------
#: ``dflash2_block_conv``'s statements for one column of one row, BF16 operands (``ROUNDING == 1``):
#: ``c0 = r(base0 + d0)``, ``acc = r(c0 * x)``, ``c1 = r(base1 + d1)``, ``term = r(c1 * shifted)``,
#: ``term = r(term * keep)``, ``acc = r(acc + term)``, with ``r`` = round to BF16 (``cvt.rn.bf16.f32``) and widen.
#: The Triton kernel computes each product and sum in FP32 between its ``cvt`` pairs with FMA contraction off;
#: the ``.rn`` forms here cannot contract either. Operands: $1 x, $2 shifted, $3 base0, $4 base1, $5 d0, $6 d1,
#: $7 keep (0.0 or 1.0).
CONV_PTX = (
    "{\n"
    ".reg .f32 c0, c1, a, t;\n"
    ".reg .b16 h;\n"
    "add.rn.f32 c0, $3, $5;\n"
    "cvt.rn.bf16.f32 h, c0;\ncvt.f32.bf16 c0, h;\n"
    "mul.rn.f32 a, c0, $1;\n"
    "cvt.rn.bf16.f32 h, a;\ncvt.f32.bf16 a, h;\n"
    "add.rn.f32 c1, $4, $6;\n"
    "cvt.rn.bf16.f32 h, c1;\ncvt.f32.bf16 c1, h;\n"
    "mul.rn.f32 t, c1, $2;\n"
    "cvt.rn.bf16.f32 h, t;\ncvt.f32.bf16 t, h;\n"
    "mul.rn.f32 t, t, $7;\n"
    "cvt.rn.bf16.f32 h, t;\ncvt.f32.bf16 t, h;\n"
    "add.rn.f32 a, a, t;\n"
    "cvt.rn.bf16.f32 h, a;\ncvt.f32.bf16 $0, h;\n"
    "}"
)

#: The Triton ``silu_and_mul`` statement sequence for one element, as the PTX that kernel compiles to:
#: ``1.0 * a`` and ``-x`` fold into one multiply of ``a`` by ``-log2(e)``, ``exp`` is
#: ``ex2.approx.f32``, ``1 / (1 + e)`` is ``div.full.f32``, then ``sigmoid * a``, ``b + 0.0`` (keeps the sign of
#: zero) and their product.
SILU_PTX = (
    "{\n"
    ".reg .f32 t, e, d, s, p, q;\n"
    "mul.f32 t, $1, 0fBFB8AA3B;\n"
    "ex2.approx.f32 e, t;\n"
    "add.f32 d, e, 0f3F800000;\n"
    "div.full.f32 s, 0f3F800000, d;\n"
    "mul.f32 p, $1, s;\n"
    "add.f32 q, $2, 0f00000000;\n"
    "mul.f32 $0, p, q;\n"
    "}"
)


@dsl_user_op
def _conv_f32(x, shifted, base0, base1, d0, d1, keep, *, loc=None, ip=None) -> "cutlass.Float32":
    """One output of ``dflash2_block_conv`` (two taps, BF16 rounding points) in FP32 (a BF16 value)."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [v.ir_value(loc=loc, ip=ip) for v in (x, shifted, base0, base1, d0, d1, keep)],
            CONV_PTX,
            "=f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _silu_and_mul_f32(a, b, *, loc=None, ip=None) -> "cutlass.Float32":
    """``a * sigmoid(a) * (b + 0.0)`` in FP32 with the Triton kernel's instructions."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            SILU_PTX,
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _load_bf16_f32(address, *, loc=None, ip=None) -> "cutlass.Float32":
    """The BF16 value at the global ``address``, widened to FP32 (exact).

    Volatile (``has_side_effects``) so it stays after the ``griddepcontrol.wait`` it is written after.
    """
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [address.ir_value(loc=loc, ip=ip)],
            "{\n.reg .b16 h;\nld.global.b16 h, [$1];\ncvt.f32.bf16 $0, h;\n}",
            "=f,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _store_bf16_rn(value, address, *, loc=None, ip=None) -> None:
    """Round ``value`` (FP32) to BF16 to nearest even and store it at the global ``address``."""
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "{\n.reg .b16 h;\ncvt.rn.bf16.f32 h, $1;\nst.global.b16 [$0], h;\n}",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _address(tensor: cute.Tensor, coord):
    return (tensor.iterator + cute.crd2idx(coord, tensor.layout)).toint()


# ---- weight layouts --------------------------------------------------------------------------------------------
def permute_conv_rows(weight: torch.Tensor, mma_m: int, taps: int = CONV_TAPS) -> torch.Tensor:
    """``kernel_projection`` rows ``[side][tap][group]`` -> side 0 per ``mma_m``-row tile ``[tile][tap][h groups]``.

    Side 1 (the second half of the rows) is left in place. Returns a new tensor; ``unpermute_conv_rows`` is its
    inverse.
    """
    rows = weight.shape[0]
    side, h = rows // 2, mma_m // taps
    groups = side // taps
    s0 = weight[:side].reshape(taps, groups // h, h, *weight.shape[1:])
    return torch.cat((s0.transpose(0, 1).reshape(side, *weight.shape[1:]), weight[side:]), dim=0)


def unpermute_conv_rows(weight: torch.Tensor, mma_m: int, taps: int = CONV_TAPS) -> torch.Tensor:
    """The inverse of ``permute_conv_rows`` on the rows."""
    rows = weight.shape[0]
    side, h = rows // 2, mma_m // taps
    groups = side // taps
    s0 = weight[:side].reshape(groups // h, taps, h, *weight.shape[1:])
    return torch.cat((s0.transpose(0, 1).reshape(side, *weight.shape[1:]), weight[side:]), dim=0)


def unpermute_conv_columns(y: torch.Tensor, mma_m: int, taps: int = CONV_TAPS) -> torch.Tensor:
    """``[..., 2 * taps * groups]`` GEMM output of the permuted weight -> the built-in column order."""
    width = y.shape[-1]
    side, h = width // 2, mma_m // taps
    groups = side // taps
    lead = y.shape[:-1]
    s0 = y[..., :side].reshape(*lead, groups // h, taps, h).transpose(-3, -2).reshape(*lead, side)
    return torch.cat((s0, y[..., side:]), dim=-1)


def interleave_gate_up(weight: torch.Tensor, mma_m: int) -> torch.Tensor:
    """``[gate; up]`` rows -> per ``mma_m``-row tile: ``mma_m / 2`` gate rows, then the matching up rows."""
    rows = weight.shape[0]
    half, h = rows // 2, mma_m // 2
    gate, up = weight[:half], weight[half:]
    return torch.stack(
        (
            gate.reshape(half // h, h, *weight.shape[1:]),
            up.reshape(half // h, h, *weight.shape[1:]),
        ),
        dim=1,
    ).reshape(weight.shape)


def deinterleave_gate_up(weight: torch.Tensor, mma_m: int) -> torch.Tensor:
    """The inverse of ``interleave_gate_up`` on the rows."""
    rows = weight.shape[0]
    half, h = rows // 2, mma_m // 2
    tiles = weight.reshape(half // h, 2, h, *weight.shape[1:])
    return torch.cat(
        (
            tiles[:, 0].reshape(half, *weight.shape[1:]),
            tiles[:, 1].reshape(half, *weight.shape[1:]),
        ),
        dim=0,
    )


def deinterleave_columns(y: torch.Tensor, mma_m: int) -> torch.Tensor:
    """``[..., 2 * d]`` GEMM output of the interleaved weight -> the ``[gate | up]`` column order."""
    width = y.shape[-1]
    h = mma_m // 2
    tiles = y.reshape(*y.shape[:-1], width // mma_m, 2, h)
    return torch.cat(
        (
            tiles[..., 0, :].reshape(*y.shape[:-1], width // 2),
            tiles[..., 1, :].reshape(*y.shape[:-1], width // 2),
        ),
        dim=-1,
    )


# ---- operand checks --------------------------------------------------------------------------------------------
def _gemm_operands_ok(x: torch.Tensor, weight: torch.Tensor, tactic) -> bool:
    return (
        x.dim() == 2
        and weight.dim() == 2
        and x.dtype == weight.dtype == torch.bfloat16
        and x.is_cuda
        and x.device == weight.device
        and x.is_contiguous()
        and weight.is_contiguous()
        and 1 <= x.shape[0] <= tactic.mma_n
        and x.shape[1] == weight.shape[1]
        and x.data_ptr() % 32 == 0
        and weight.data_ptr() % 32 == 0
        and not torch.is_grad_enabled()
    )


def silu_takes(x: torch.Tensor, weight: torch.Tensor, tactic) -> bool:
    """Whether ``silu_splitk_gemm`` takes ``x @ weight.T`` at ``tactic`` (else the caller runs the plain path)."""
    return (
        tactic.mma_m in (64, 128)
        and tactic.mma_n in (8, 16)
        and _gemm_operands_ok(x, weight, tactic)
        and weight.shape[0] % (2 * tactic.mma_m) == 0
    )


def spread_takes(tactic) -> bool:
    """Whether CF's spread epilogue (``spread=True``) covers ``tactic``'s tile.

    Each of the cluster's ``split_k`` CTAs convolves ``h * 16 / split_k`` columns of a side-0 tile; the epilogue's
    four warps own two of the eight rows each and a warp's 32 lanes four consecutive columns each, so a CTA's share
    must be exactly 128 columns of an 8-row tile.
    """
    columns = (tactic.mma_m // CONV_TAPS) * CONV_GROUP
    return (
        tactic.mma_n == 8
        and tactic.split_k > 1
        and columns % tactic.split_k == 0
        and columns // tactic.split_k == _SPREAD_LANES * _SPREAD_COLS
    )


def conv_takes(
    x: torch.Tensor,
    weight: torch.Tensor,
    base0: torch.Tensor,
    block_size: int,
    tactic,
    spread: bool = False,
) -> bool:
    """Whether ``conv_splitk_gemm`` takes this projection + prepare convolution (else the plain path).

    ``weight`` is the ``[2 * taps * groups, hidden]`` projection (permuted), ``base0`` the side-0 base kernel
    ``[taps, hidden]``; ``spread`` asks for CF's spread epilogue (``spread_takes``).
    """
    if not (
        tactic.mma_m in (64, 128) and tactic.mma_n == 8 and _gemm_operands_ok(x, weight, tactic)
    ):
        return False
    if spread and not spread_takes(tactic):
        return False
    hidden = x.shape[1]
    groups = hidden // CONV_GROUP
    h = tactic.mma_m // CONV_TAPS
    columns = h * CONV_GROUP
    return (
        hidden % CONV_GROUP == 0
        and weight.shape[0] == 2 * CONV_TAPS * groups
        and groups % h == 0
        and columns % _EPILOG_THREADS == 0
        and CONV_GROUP % (columns // _EPILOG_THREADS) == 0
        and tuple(base0.shape) == (CONV_TAPS, hidden)
        and base0.dtype == torch.bfloat16
        and base0.is_contiguous()
        and base0.data_ptr() % 32 == 0
        and block_size >= 1
    )


# ---- the entry trigger: the dependents released after the first weight ring --------------------------------------
class _EntryTriggerDma:
    """Mixin over ``SplitKDenseGemmKernel``: its ``dma_warp`` with the entry trigger
    (``kernels/dr3_folds.GEMV_ENTRY_TRIGGER``, off in every composition).

    The parent's loop statement for statement, except that the weight warp executes
    ``griddepcontrol.launch_dependents`` right after the TMA of k-tile ``min(stages, k_tiles) - 1`` (the last
    load it can issue before any stage is consumed) and not after the loop. Mixed only into the entry-trigger
    kernels (``_ConvEntryTriggerKernel`` / ``_SiluEntryTriggerKernel``); the other kernels keep the parent's.
    """

    @cute.experimental.jit
    def dma_warp(
        self,
        bar_full,
        bar_empty,
        bar_tma_epilog,
        g_tile: cute.Tensor,
        s_tile: cute.Tensor,
        cta_v_map: cute.Layout,
        k_tile_start: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        is_a: cutlass.Constexpr,
    ):
        stages = self.num_ab_stage
        if cutlass.const_expr(not is_a and self.use_pdl):
            cute.arch.griddepcontrol_wait()

        trigger_at = k_tile_count - 1
        if k_tile_count > stages:
            trigger_at = cutlass.Int32(stages - 1)
        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar_full + stage,
                    cute.size_in_bytes(
                        s_tile.element_type,
                        cute.slice_(s_tile.layout, (None, None, None, 0)),
                    ),
                )
            _cute_ext.tma_load(
                g_tile[None, None, k_tile_start + k_tile],
                s_tile[None, None, None, stage],
                (bar_full + stage).value,
                cta_v_map=cta_v_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
            )
            if cutlass.const_expr(is_a and self.use_pdl):
                if k_tile == trigger_at:
                    cute.arch.griddepcontrol_launch_dependents()
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1

        if cutlass.const_expr(not is_a and self.has_bias):
            cute.arch.mbarrier_arrive(bar_tma_epilog)
        self._drain_producer(bar_empty, empty_phase, k_tile_count)


# ---- CF: the coefficient projection + prepare convolution -------------------------------------------------------
class ConvSplitKDenseGemmKernel(_splitk.SplitKDenseGemmKernel):
    """``SplitKDenseGemmKernel`` whose side-0 owner CTAs store the prepare convolution of their columns.

    ``c`` is the coefficient output (the GEMM's own ``[M, 2 * taps * groups]``, written for the side-1 tiles
    only), ``base`` the side-0 base kernel ``[taps, hidden]``, ``out`` the convolution ``[M, hidden]``.

    ``spread`` (``spread_takes``): in a side-0 cluster every CTA, not only the owner, ends with the reduced
    tile and convolves ``1 / split_k`` of its columns. Each CTA sends its FP32 partial to every peer (slot = its
    rank in a ``split_k``-slot mailbox, its own slot written locally) and sums the ``split_k`` slots in rank order,
    ``((p0 + p1) + p2) + ...``: the owner's sum of the plain kernel, operand for operand (FP32 addition of two
    values is commutative), so every CTA holds the owner's FP32 values, rounds them as the owner does, and applies
    the same convolution statements to its share of the columns. Side-1 clusters keep the owner-only reduction.
    """

    def __init__(
        self,
        *,
        tactic: "_splitk.SplitKTactic",
        use_pdl: bool,
        block_size: int,
        spread: bool = False,
    ) -> None:
        super().__init__(tactic=tactic, use_pdl=use_pdl, has_bias=False)
        self.half_m = tactic.mma_m // CONV_TAPS
        self.tile_columns = self.half_m * CONV_GROUP
        self.cols_per_thread = self.tile_columns // _EPILOG_THREADS
        self.block_size = block_size
        self.spread = bool(spread)
        if self.spread and not spread_takes(tactic):
            raise ValueError(
                f"ConvSplitKDenseGemmKernel: the spread epilogue does not cover {tactic}"
            )
        values_per_thread = (tactic.mma_m * tactic.mma_n) // _EPILOG_THREADS
        # The spread epilogue's mailbox has a slot per rank (its own included); the plain one a slot per peer.
        self.conv_mailbox_elements = (
            tactic.split_k * _EPILOG_THREADS * values_per_thread
            if self.spread
            else self.mailbox_elements
        )

    @cute.experimental.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        base: cute.Tensor,
        out: cute.Tensor,
        stream: _cuda.CUstream,
    ):
        # The parent's grid, cluster, shared-memory request and PDL flag.
        self.kernel(a, b, c, base, out).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(utils.get_smem_capacity_in_bytes("sm_100")),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mA: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L), K-major: the permuted projection weight
        mB: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L), K-major: the activation (the convolution's x)
        mC: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L), M-major: the coefficients
        mBase: cute.Tensor,  # (taps, hidden): the side-0 base kernel
        mOut: cute.Tensor,  # (Gemm_N, hidden): the prepare convolution
    ):
        """The parent's kernel body; the epilogue and the staging buffer differ."""
        stages = self.num_ab_stage

        ab_dtype = mA.element_type
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype,
            ab_dtype,
            utils.LayoutEnum.from_tensor(mA).mma_major_mode(),
            utils.LayoutEnum.from_tensor(mB).mma_major_mode(),
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )

        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        block_idx = cute.arch.block_idx()
        bidx = block_idx[0]
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_idx = block_idx[1] // self.split_k
        l_idx = block_idx[2]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        sA = _cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_a(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_splitk._AB_BUFFER_ALIGN_BYTES,
        )
        sB = _cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_splitk._AB_BUFFER_ALIGN_BYTES,
        )

        acc_layout = _cute_ext.make_tmem_layout_acc(tiled_mma, self.mma_tiler_mn, acc_stage=1)

        def barriers(count):
            return _cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(count),
                alignment=_splitk._MBARRIER_BYTES,
            ).iterator

        bar_full = barriers(stages)
        bar_empty = barriers(stages)
        bar_tma_epilog = barriers(1)
        bar_mma_epilog = barriers(1)
        bar_tmem_alloc = barriers(1)
        tmem_base_ptr = _cute_ext.allocate(
            cutlass.Int32,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_splitk._TMEM_POINTER_BYTES,
        ).iterator

        if cutlass.const_expr(self.split_k > 1):
            mailbox = _cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout(self.conv_mailbox_elements),
                alignment=_splitk._MAILBOX_ALIGN_BYTES,
            )
            bar_reduce = barriers(1)
        else:
            # Dummy operands for the compile-time-elided reduction.
            mailbox = sA
            bar_reduce = bar_mma_epilog

        # The owner's rounded tile, widened to FP32, in the accumulator's (M, N) layout.
        sStage = _cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout((self.cta_m, self.cta_n)),
            alignment=_splitk._MAILBOX_ALIGN_BYTES,
        )

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_tma_epilog, 32)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)

                if cutlass.const_expr(self.split_k > 1):
                    # Owner arrival plus peer transaction-byte credits.
                    cute.arch.mbarrier_init(bar_reduce, 1)

        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.split_k > 1):
            # Publish peer barriers before cross-CTA stores.
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.barrier()

        # Host validation guarantees an equal, tail-free K partition.
        k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
        k_tile_start = split_rank * k_tile_count

        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_wait()

        # Warp 3 is idle; warps 4-7 run the epilogue.
        if warp_idx == 0:
            self.dma_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                cute.local_tile(mA, (self.cta_m, self.cta_k), (bidx, None, l_idx)),
                sA,
                _cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A"),
                k_tile_start,
                k_tile_count,
                True,
            )
        elif warp_idx == 1:
            self.dma_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                cute.local_tile(mB, (self.cta_n, self.cta_k), (n_idx, None, l_idx)),
                sB,
                _cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B"),
                k_tile_start,
                k_tile_count,
                False,
            )
        elif warp_idx == 2:
            self.mma_warp(
                bar_full,
                bar_empty,
                bar_mma_epilog,
                bar_tmem_alloc,
                tiled_mma,
                sA,
                sB,
                tmem_base_ptr,
                acc_layout,
                self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2]),
                k_tile_count,
            )
        elif warp_idx >= 4:
            if cutlass.const_expr(self.spread):
                self.conv_spread_epilog_warp(
                    bar_mma_epilog,
                    bar_tmem_alloc,
                    tmem_base_ptr,
                    acc_layout,
                    cute.local_tile(mC, (self.cta_m, self.cta_n), (bidx, n_idx, l_idx)),
                    cute.size(mC, mode=[1]) - n_idx * self.cta_n,
                    sStage,
                    cute.arch.thread_idx()[0] - 128,
                    utils.LayoutEnum.from_tensor(mC),
                    mailbox,
                    bar_reduce,
                    split_rank,
                    bidx,
                    cute.size(mC, mode=[0]) // 2 // self.cta_m,
                    mB,
                    mBase,
                    mOut,
                )
            else:
                self.conv_epilog_warp(
                    bar_mma_epilog,
                    bar_tmem_alloc,
                    tmem_base_ptr,
                    acc_layout,
                    cute.local_tile(mC, (self.cta_m, self.cta_n), (bidx, n_idx, l_idx)),
                    cute.size(mC, mode=[1]) - n_idx * self.cta_n,
                    sStage,
                    cute.arch.thread_idx()[0] - 128,
                    utils.LayoutEnum.from_tensor(mC),
                    mailbox,
                    bar_reduce,
                    split_rank,
                    bidx,
                    cute.size(mC, mode=[0]) // 2 // self.cta_m,
                    mB,
                    mBase,
                    mOut,
                )

    @cute.experimental.jit
    def conv_epilog_warp(
        self,
        bar_mma_epilog,
        bar_tmem_alloc,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        gD_tile: cute.Tensor,
        rows_left: cutlass.Int32,
        sStage: cute.Tensor,
        epi_tid: cutlass.Int32,
        d_layout: cutlass.Constexpr,
        mailbox,
        bar_reduce,
        split_rank: cutlass.Int32,
        bidx: cutlass.Int32,
        conv_tiles: cutlass.Int32,
        mB: cute.Tensor,
        mBase: cute.Tensor,
        mOut: cute.Tensor,
    ):
        c_dtype = cutlass.BFloat16
        cpt = self.cols_per_thread
        rows = self.cta_n
        col0 = bidx * self.tile_columns + epi_tid * cpt
        group = (epi_tid * cpt) // CONV_GROUP

        # The convolution's operands for this thread's columns: base kernel taps and every row of x.
        rX = _cute_ext.allocate(
            cutlass.Float32, cute.AddressSpace.rmem, cute.make_layout(rows * cpt), alignment=32
        )
        rBase = _cute_ext.allocate(
            cutlass.Float32, cute.AddressSpace.rmem, cute.make_layout(2 * cpt), alignment=32
        )
        if split_rank == _splitk.OWNER_RANK:
            if bidx < conv_tiles:
                for e in cutlass.range_constexpr(cpt):
                    rBase[e] = _load_bf16_f32(_address(mBase, (0, col0 + e)))
                    rBase[cpt + e] = _load_bf16_f32(_address(mBase, (1, col0 + e)))
                if cutlass.const_expr(self.use_pdl):
                    # x is the activation the GEMM's programmatic predecessor wrote.
                    cute.arch.griddepcontrol_wait()
                for n in cutlass.range_constexpr(rows):
                    if n < rows_left:
                        for e in cutlass.range_constexpr(cpt):
                            rX[n * cpt + e] = _load_bf16_f32(_address(mB, (n, col0 + e, 0)))

        # Wait until MMA publishes the TMEM base pointer.
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)

        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr),
            acc_layout,
        )[((None, None), 0, 0, 0)]

        epi_tile = (self.cta_m, self.cta_n)
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (self.cta_m, self.cta_n, self.cta_k),
                d_layout,
                c_dtype,
                self.acc_dtype,
                epi_tile,
                False,
            ),
            acc_view,
        )
        gD_epi = cute.flat_divide(gD_tile, epi_tile)
        sStage_epi = cute.flat_divide(sStage, epi_tile)

        # Match each epilogue thread's TMEM partition in RMEM.
        rmem_layout = _cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
        rAcc = _cute_ext.allocate(self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32)
        rD = _cute_ext.allocate(c_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32)
        rWide = _cute_ext.allocate(
            self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32
        )
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        _cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        # Make tcgen05.ld visible before TMEM release and RMEM use.
        cute.arch.fence_view_async_tmem_load()
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        # Peers publish FP32 partials; only rank 0 reduces and stores (the parent's code).
        if cutlass.const_expr(self.split_k > 1):
            values_per_thread = cutlass.const_expr(cute.size(rmem_layout))
            values_per_peer = cutlass.const_expr(_EPILOG_THREADS * values_per_thread)
            if split_rank != _splitk.OWNER_RANK:
                for value_idx in cutlass.range_constexpr(0, values_per_thread, 4):
                    _splitk._store_shared_remote_v4(
                        rAcc[value_idx],
                        rAcc[value_idx + 1],
                        rAcc[value_idx + 2],
                        rAcc[value_idx + 3],
                        mailbox.iterator
                        + (split_rank - Int32(1)) * values_per_peer
                        + epi_tid * values_per_thread
                        + value_idx,
                        bar_reduce,
                        Int32(_splitk.OWNER_RANK),
                    )
            else:
                if epi_tid == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        bar_reduce, self.expected_transaction_bytes
                    )
                cute.arch.mbarrier_wait(bar_reduce, 0)
                for peer in cutlass.range_constexpr(self.split_k - 1):
                    for value_idx in cutlass.range_constexpr(values_per_thread):
                        rAcc[value_idx] = (
                            rAcc[value_idx]
                            + mailbox[
                                peer * values_per_peer + epi_tid * values_per_thread + value_idx
                            ]
                        )

        if split_rank == _splitk.OWNER_RANK:
            # The plain kernel's BF16 values.
            rD.store(rAcc.load().to(c_dtype))
            if bidx < conv_tiles:
                # Widened back (exact), staged in TMEM coordinates: row i < h of the tile is tap 0 of group
                # bidx * h + i, row h + i tap 1 of the same group.
                rWide.store(rD.load().to(self.acc_dtype))
                _cute_ext.partition_and_copy(thr_t2r, rWide, sStage_epi[None, None, 0, 0])
                cute.arch.barrier(barrier_id=STAGE_BARRIER_ID, number_of_threads=_EPILOG_THREADS)
                for n in cutlass.range_constexpr(rows):
                    if n < rows_left:
                        d0 = sStage[group, n]
                        d1 = sStage[self.half_m + group, n]
                        keep = cutlass.Float32(1.0 if n % self.block_size >= 1 else 0.0)
                        for e in cutlass.range_constexpr(cpt):
                            shifted = rX[(n - 1) * cpt + e] if n >= 1 else cutlass.Float32(0.0)
                            value = _conv_f32(
                                rX[n * cpt + e], shifted, rBase[e], rBase[cpt + e], d0, d1, keep
                            )
                            _store_bf16_rn(value, _address(mOut, (n, col0 + e)))
            else:
                # Side 1: the coefficients, stored as the plain kernel stores them.
                _cute_ext.partition_and_copy(thr_t2r, rD, gD_epi[None, None, 0, 0])

        # The reduction mbarrier covers remote stores; no cluster barrier needed.

    @cute.experimental.jit
    def conv_spread_epilog_warp(
        self,
        bar_mma_epilog,
        bar_tmem_alloc,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        gD_tile: cute.Tensor,
        rows_left: cutlass.Int32,
        sStage: cute.Tensor,
        epi_tid: cutlass.Int32,
        d_layout: cutlass.Constexpr,
        mailbox,
        bar_reduce,
        split_rank: cutlass.Int32,
        bidx: cutlass.Int32,
        conv_tiles: cutlass.Int32,
        mB: cute.Tensor,
        mBase: cute.Tensor,
        mOut: cute.Tensor,
    ):
        """The spread epilogue: ``conv_epilog_warp`` with the side-0 convolution spread over the cluster's CTAs.

        Epilogue warp ``w`` (0-3) owns rows ``2w`` and ``2w + 1``; lane ``l`` four consecutive columns of the CTA's
        128-column share (``split_rank * 128 + 4 * l``); each output is ``_conv_f32`` of the same operands as in
        ``conv_epilog_warp`` and is stored the same way.
        """
        c_dtype = cutlass.BFloat16
        cpt = _SPREAD_COLS
        rpw = _SPREAD_ROWS
        share = self.tile_columns // self.split_k
        lane = epi_tid % _SPREAD_LANES
        quad = epi_tid // _SPREAD_LANES
        col0 = bidx * self.tile_columns + split_rank * share + lane * cpt
        group = (split_rank * share + lane * cpt) // CONV_GROUP

        # This thread's convolution operands: both base taps of its columns, and x of rows 2w - 1 .. 2w + 1.
        rX = _cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.rmem,
            cute.make_layout((rpw + 1) * cpt),
            alignment=32,
        )
        rBase = _cute_ext.allocate(
            cutlass.Float32, cute.AddressSpace.rmem, cute.make_layout(2 * cpt), alignment=32
        )
        if bidx < conv_tiles:
            for e in cutlass.range_constexpr(cpt):
                rBase[e] = _load_bf16_f32(_address(mBase, (0, col0 + e)))
                rBase[cpt + e] = _load_bf16_f32(_address(mBase, (1, col0 + e)))
            if cutlass.const_expr(self.use_pdl):
                # x is the activation the GEMM's programmatic predecessor wrote.
                cute.arch.griddepcontrol_wait()
            for q in cutlass.range_constexpr(self.cta_n // rpw):
                if quad == q:
                    for lr in cutlass.range_constexpr(rpw + 1):
                        n = q * rpw - 1 + lr
                        if cutlass.const_expr(n >= 0):
                            if n < rows_left:
                                for e in cutlass.range_constexpr(cpt):
                                    rX[lr * cpt + e] = _load_bf16_f32(
                                        _address(mB, (n, col0 + e, 0))
                                    )

        # Wait until MMA publishes the TMEM base pointer.
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)

        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr),
            acc_layout,
        )[((None, None), 0, 0, 0)]

        epi_tile = (self.cta_m, self.cta_n)
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (self.cta_m, self.cta_n, self.cta_k),
                d_layout,
                c_dtype,
                self.acc_dtype,
                epi_tile,
                False,
            ),
            acc_view,
        )
        gD_epi = cute.flat_divide(gD_tile, epi_tile)
        sStage_epi = cute.flat_divide(sStage, epi_tile)

        # Match each epilogue thread's TMEM partition in RMEM.
        rmem_layout = _cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
        rAcc = _cute_ext.allocate(self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32)
        rD = _cute_ext.allocate(c_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32)
        rWide = _cute_ext.allocate(
            self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32
        )
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        _cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        # Make tcgen05.ld visible before TMEM release and RMEM use.
        cute.arch.fence_view_async_tmem_load()
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        # Rank r's partial goes to slot r of the mailboxes that reduce: every CTA's in a side-0 cluster (the
        # remote slots by st.async, its own slot by a local store), the owner's only in a side-1 cluster. A CTA
        # that reduces sums slots 0 .. split_k - 1 in that order, the owner's order of the plain kernel.
        values_per_thread = cutlass.const_expr(cute.size(rmem_layout))
        values_per_peer = cutlass.const_expr(_EPILOG_THREADS * values_per_thread)
        own = split_rank * values_per_peer + epi_tid * values_per_thread
        if bidx < conv_tiles:
            for value_idx in cutlass.range_constexpr(values_per_thread):
                mailbox[own + value_idx] = rAcc[value_idx]
            for peer in cutlass.range_constexpr(self.split_k):
                if split_rank != peer:
                    for value_idx in cutlass.range_constexpr(0, values_per_thread, 4):
                        _splitk._store_shared_remote_v4(
                            rAcc[value_idx],
                            rAcc[value_idx + 1],
                            rAcc[value_idx + 2],
                            rAcc[value_idx + 3],
                            mailbox.iterator + own + value_idx,
                            bar_reduce,
                            Int32(peer),
                        )
            if epi_tid == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(bar_reduce, self.expected_transaction_bytes)
            cute.arch.mbarrier_wait(bar_reduce, 0)
            for value_idx in cutlass.range_constexpr(values_per_thread):
                rAcc[value_idx] = mailbox[epi_tid * values_per_thread + value_idx]
                for rank in cutlass.range_constexpr(1, self.split_k):
                    rAcc[value_idx] = (
                        rAcc[value_idx]
                        + mailbox[rank * values_per_peer + epi_tid * values_per_thread + value_idx]
                    )
        else:
            if split_rank != _splitk.OWNER_RANK:
                for value_idx in cutlass.range_constexpr(0, values_per_thread, 4):
                    _splitk._store_shared_remote_v4(
                        rAcc[value_idx],
                        rAcc[value_idx + 1],
                        rAcc[value_idx + 2],
                        rAcc[value_idx + 3],
                        mailbox.iterator + own + value_idx,
                        bar_reduce,
                        Int32(_splitk.OWNER_RANK),
                    )
            else:
                if epi_tid == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        bar_reduce, self.expected_transaction_bytes
                    )
                cute.arch.mbarrier_wait(bar_reduce, 0)
                for value_idx in cutlass.range_constexpr(values_per_thread):
                    for rank in cutlass.range_constexpr(1, self.split_k):
                        rAcc[value_idx] = (
                            rAcc[value_idx]
                            + mailbox[
                                rank * values_per_peer + epi_tid * values_per_thread + value_idx
                            ]
                        )

        if bidx < conv_tiles:
            # Every CTA: the plain kernel's BF16 values, widened back (exact) and staged in TMEM coordinates
            # (row i < h of the tile is tap 0 of group bidx * h + i, row h + i tap 1 of the same group).
            rD.store(rAcc.load().to(c_dtype))
            rWide.store(rD.load().to(self.acc_dtype))
            _cute_ext.partition_and_copy(thr_t2r, rWide, sStage_epi[None, None, 0, 0])
            cute.arch.barrier(barrier_id=STAGE_BARRIER_ID, number_of_threads=_EPILOG_THREADS)
            for q in cutlass.range_constexpr(self.cta_n // rpw):
                if quad == q:
                    for j in cutlass.range_constexpr(rpw):
                        n = q * rpw + j
                        if n < rows_left:
                            d0 = sStage[group, n]
                            d1 = sStage[self.half_m + group, n]
                            keep = cutlass.Float32(1.0 if n % self.block_size >= 1 else 0.0)
                            for e in cutlass.range_constexpr(cpt):
                                shifted = rX[j * cpt + e] if n >= 1 else cutlass.Float32(0.0)
                                value = _conv_f32(
                                    rX[(j + 1) * cpt + e],
                                    shifted,
                                    rBase[e],
                                    rBase[cpt + e],
                                    d0,
                                    d1,
                                    keep,
                                )
                                _store_bf16_rn(value, _address(mOut, (n, col0 + e)))
        elif split_rank == _splitk.OWNER_RANK:
            # Side 1: the coefficients, stored as the plain kernel stores them.
            rD.store(rAcc.load().to(c_dtype))
            _cute_ext.partition_and_copy(thr_t2r, rD, gD_epi[None, None, 0, 0])

        # Each reducing CTA waited for every byte sent to it; a sender's st.async completes on the receiver's
        # mbarrier, so no cluster barrier is needed.


@cute.experimental.jit
def _bmm_conv(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    base: cute.Tensor,
    out: cute.Tensor,
    stream: _cuda.CUstream,
):
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0])),
        base,
        out,
        stream,
    )


class _ConvEntryTriggerKernel(_EntryTriggerDma, ConvSplitKDenseGemmKernel):
    """CF with the entry trigger (``entry_trigger``); otherwise ``ConvSplitKDenseGemmKernel``."""


@functools.cache
def _compiled_conv(
    dtype: torch.dtype,
    tactic: "_splitk.SplitKTactic",
    use_pdl: bool,
    leading_dims: Tuple[int, int, int],
    block_size: int,
    spread: bool = False,
    entry_trigger: bool = False,
):
    cls = _ConvEntryTriggerKernel if entry_trigger else ConvSplitKDenseGemmKernel
    kernel = cls(tactic=tactic, use_pdl=use_pdl, block_size=block_size, spread=spread)
    a, b, c = _splitk._make_compile_repr_tensors(dtype, False, *leading_dims)[:3]
    base = _splitk._from_dlpack_dynamic(
        torch.empty((CONV_TAPS, 128), dtype=dtype, device="cuda"), 1
    )
    out = _splitk._from_dlpack_dynamic(torch.empty((8, 128), dtype=dtype, device="cuda"), 1)
    stream = _cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return _cute_ext.compile(_bmm_conv, kernel, a, b, c, base, out, stream)


def conv_splitk_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    base0: torch.Tensor,
    block_size: int,
    tactic: "_splitk.SplitKTactic",
    enable_pdl: bool,
    spread: bool = False,
    *,
    entry_trigger: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(prepare convolution [M, hidden], coefficients [M, 2 * taps * groups])`` for a ``permute_conv_rows`` weight.

    The convolution is ``dflash2_block_conv(x, delta_0, base0, block_size, 16)`` with ``delta_0`` the side-0
    coefficients of ``x @ W.T`` (``W`` = the unpermuted weight); the returned coefficients hold the side-1
    columns (``[:, half:]``) of ``x @ W.T``, bitwise, and leave the side-0 columns unwritten. ``spread`` selects
    the spread epilogue (the same values; ``ConvSplitKDenseGemmKernel``). The first call per tactic, PDL flag, layout,
    block size and epilogue compiles (CuTe-DSL JIT) and must not be inside a CUDA-graph capture.

    Raises:
        ValueError: on the operands or tactic ``conv_takes`` / ``run_splitk_dense`` refuse.
    """
    if not conv_takes(x, weight, base0, block_size, tactic, spread):
        raise ValueError(
            f"conv_splitk_gemm: unsupported operands x {tuple(x.shape)} {x.dtype}, weight "
            f"{tuple(weight.shape)} {weight.dtype}, base {tuple(base0.shape)}, tactic {tactic}"
        )
    m, k = x.shape
    n = weight.shape[0]
    _splitk.validate_tactic(tactic, m, n, k)
    coefficients = torch.empty((m, n), dtype=x.dtype, device=x.device)
    out = torch.empty((m, k), dtype=x.dtype, device=x.device)
    # run_splitk_dense's operand swap (``_to_cute_swap``).
    a_swap = weight.detach().t().unsqueeze(0).transpose(-2, -1)
    b_swap = x.detach().unsqueeze(0).transpose(-2, -1)
    c_swap = coefficients.unsqueeze(0).transpose(-2, -1)
    leading_dims = tuple(_splitk._detect_leading_dim(t) for t in (a_swap, b_swap, c_swap))
    cute_tensors = tuple(
        _splitk._from_dlpack_dynamic(t, d)
        for t, d in zip((a_swap, b_swap, c_swap), leading_dims, strict=True)
    )
    compiled = _compiled_conv(
        x.dtype,
        tactic,
        bool(enable_pdl),
        leading_dims,
        int(block_size),
        bool(spread),
        bool(entry_trigger),
    )
    stream = _cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    compiled(
        *cute_tensors,
        _splitk._from_dlpack_dynamic(base0.detach(), 1),
        _splitk._from_dlpack_dynamic(out, 1),
        stream,
    )
    return out, coefficients


# ---- SF: gate_up + silu(gate) * up ------------------------------------------------------------------------------
class SiluSplitKDenseGemmKernel(_splitk.SplitKDenseGemmKernel):
    """``SplitKDenseGemmKernel`` whose owner CTA stores ``silu(gate) * up`` of its tile.

    ``c`` is the activation output (kernel-M = half the GEMM's rows). ``mma_m`` 64 or 128 (two halves per tile),
    ``mma_n`` 8 or 16, any split-K.
    """

    def __init__(self, *, tactic: "_splitk.SplitKTactic", use_pdl: bool) -> None:
        super().__init__(tactic=tactic, use_pdl=use_pdl, has_bias=False)
        self.half_m = tactic.mma_m // 2

    @cute.experimental.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        stream: _cuda.CUstream,
    ):
        # The parent's grid over the GEMM's kernel-M (twice the output's), cluster,
        # shared-memory request and PDL flag.
        self.kernel(a, b, c).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.half_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(utils.get_smem_capacity_in_bytes("sm_100")),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mA: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L), K-major: the interleaved weight
        mB: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L), K-major: the activation
        mC: cute.Tensor,  # (Gemm_M / 2, Gemm_N, Gemm_L), M-major: the SwiGLU output
    ):
        """The parent's kernel body; only the output tile and the staging buffer differ."""
        stages = self.num_ab_stage

        ab_dtype = mA.element_type
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype,
            ab_dtype,
            utils.LayoutEnum.from_tensor(mA).mma_major_mode(),
            utils.LayoutEnum.from_tensor(mB).mma_major_mode(),
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )

        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        block_idx = cute.arch.block_idx()
        bidx = block_idx[0]
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_idx = block_idx[1] // self.split_k
        l_idx = block_idx[2]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        sA = _cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_a(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_splitk._AB_BUFFER_ALIGN_BYTES,
        )
        sB = _cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_splitk._AB_BUFFER_ALIGN_BYTES,
        )

        acc_layout = _cute_ext.make_tmem_layout_acc(tiled_mma, self.mma_tiler_mn, acc_stage=1)

        def barriers(count):
            return _cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(count),
                alignment=_splitk._MBARRIER_BYTES,
            ).iterator

        bar_full = barriers(stages)
        bar_empty = barriers(stages)
        bar_tma_epilog = barriers(1)
        bar_mma_epilog = barriers(1)
        bar_tmem_alloc = barriers(1)
        tmem_base_ptr = _cute_ext.allocate(
            cutlass.Int32,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_splitk._TMEM_POINTER_BYTES,
        ).iterator

        if cutlass.const_expr(self.split_k > 1):
            mailbox = _cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout(self.mailbox_elements),
                alignment=_splitk._MAILBOX_ALIGN_BYTES,
            )
            bar_reduce = barriers(1)
        else:
            mailbox = sA
            bar_reduce = bar_mma_epilog

        sStage = _cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout((self.cta_m, self.cta_n)),
            alignment=_splitk._MAILBOX_ALIGN_BYTES,
        )

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_tma_epilog, 32)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)

                if cutlass.const_expr(self.split_k > 1):
                    cute.arch.mbarrier_init(bar_reduce, 1)

        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.barrier()

        k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
        k_tile_start = split_rank * k_tile_count

        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_wait()

        if warp_idx == 0:
            self.dma_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                cute.local_tile(mA, (self.cta_m, self.cta_k), (bidx, None, l_idx)),
                sA,
                _cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A"),
                k_tile_start,
                k_tile_count,
                True,
            )
        elif warp_idx == 1:
            self.dma_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                cute.local_tile(mB, (self.cta_n, self.cta_k), (n_idx, None, l_idx)),
                sB,
                _cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B"),
                k_tile_start,
                k_tile_count,
                False,
            )
        elif warp_idx == 2:
            self.mma_warp(
                bar_full,
                bar_empty,
                bar_mma_epilog,
                bar_tmem_alloc,
                tiled_mma,
                sA,
                sB,
                tmem_base_ptr,
                acc_layout,
                self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2]),
                k_tile_count,
            )
        elif warp_idx >= 4:
            self.silu_epilog_warp(
                bar_mma_epilog,
                bar_tmem_alloc,
                tmem_base_ptr,
                acc_layout,
                cute.local_tile(mC, (self.half_m, self.cta_n), (bidx, n_idx, l_idx)),
                cute.size(mC, mode=[1]) - n_idx * self.cta_n,
                sStage,
                cute.arch.thread_idx()[0] - 128,
                utils.LayoutEnum.from_tensor(mC),
                mailbox,
                bar_reduce,
                split_rank,
            )

    @cute.experimental.jit
    def silu_epilog_warp(
        self,
        bar_mma_epilog,
        bar_tmem_alloc,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        gOut_tile: cute.Tensor,
        rows_left: cutlass.Int32,
        sStage: cute.Tensor,
        epi_tid: cutlass.Int32,
        d_layout: cutlass.Constexpr,
        mailbox,
        bar_reduce,
        split_rank: cutlass.Int32,
    ):
        c_dtype = cutlass.BFloat16
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)

        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr),
            acc_layout,
        )[((None, None), 0, 0, 0)]

        epi_tile = (self.cta_m, self.cta_n)
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (self.cta_m, self.cta_n, self.cta_k),
                d_layout,
                c_dtype,
                self.acc_dtype,
                epi_tile,
                False,
            ),
            acc_view,
        )
        sStage_epi = cute.flat_divide(sStage, epi_tile)

        rmem_layout = _cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, sStage_epi, epi_tid)
        rAcc = _cute_ext.allocate(self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32)
        rD = _cute_ext.allocate(c_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32)
        rWide = _cute_ext.allocate(
            self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32
        )
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        _cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        cute.arch.fence_view_async_tmem_load()
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        if cutlass.const_expr(self.split_k > 1):
            values_per_thread = cutlass.const_expr(cute.size(rmem_layout))
            values_per_peer = cutlass.const_expr(_EPILOG_THREADS * values_per_thread)
            if split_rank != _splitk.OWNER_RANK:
                for value_idx in cutlass.range_constexpr(0, values_per_thread, 4):
                    _splitk._store_shared_remote_v4(
                        rAcc[value_idx],
                        rAcc[value_idx + 1],
                        rAcc[value_idx + 2],
                        rAcc[value_idx + 3],
                        mailbox.iterator
                        + (split_rank - Int32(1)) * values_per_peer
                        + epi_tid * values_per_thread
                        + value_idx,
                        bar_reduce,
                        Int32(_splitk.OWNER_RANK),
                    )
            else:
                if epi_tid == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        bar_reduce, self.expected_transaction_bytes
                    )
                cute.arch.mbarrier_wait(bar_reduce, 0)
                for peer in cutlass.range_constexpr(self.split_k - 1):
                    for value_idx in cutlass.range_constexpr(values_per_thread):
                        rAcc[value_idx] = (
                            rAcc[value_idx]
                            + mailbox[
                                peer * values_per_peer + epi_tid * values_per_thread + value_idx
                            ]
                        )

        if split_rank == _splitk.OWNER_RANK:
            rD.store(rAcc.load().to(c_dtype))
            rWide.store(rD.load().to(self.acc_dtype))
            _cute_ext.partition_and_copy(thr_t2r, rWide, sStage_epi[None, None, 0, 0])
            cute.arch.barrier(barrier_id=STAGE_BARRIER_ID, number_of_threads=_EPILOG_THREADS)

            column = epi_tid % self.half_m
            first_row = epi_tid // self.half_m
            row_step = _EPILOG_THREADS // self.half_m
            for j in cutlass.range_constexpr(self.cta_n // row_step):
                row = first_row + j * row_step
                if row < rows_left:
                    out = _silu_and_mul_f32(sStage[column, row], sStage[self.half_m + column, row])
                    _store_bf16_rn(
                        out,
                        (
                            gOut_tile.iterator + cute.crd2idx((column, row), gOut_tile.layout)
                        ).toint(),
                    )


@cute.experimental.jit
def _bmm_silu(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    stream: _cuda.CUstream,
):
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0])),
        stream,
    )


class _SiluEntryTriggerKernel(_EntryTriggerDma, SiluSplitKDenseGemmKernel):
    """SF with the entry trigger (``entry_trigger``); otherwise ``SiluSplitKDenseGemmKernel``."""


@functools.cache
def _compiled_silu(
    dtype: torch.dtype,
    tactic: "_splitk.SplitKTactic",
    use_pdl: bool,
    leading_dims: Tuple[int, int, int],
    entry_trigger: bool = False,
):
    cls = _SiluEntryTriggerKernel if entry_trigger else SiluSplitKDenseGemmKernel
    kernel = cls(tactic=tactic, use_pdl=use_pdl)
    compile_tensors = _splitk._make_compile_repr_tensors(dtype, False, *leading_dims)
    stream = _cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return _cute_ext.compile(_bmm_silu, kernel, *compile_tensors[:3], stream)


def silu_splitk_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    tactic: "_splitk.SplitKTactic",
    enable_pdl: bool,
    entry_trigger: bool = False,
) -> torch.Tensor:
    """``silu(x @ gate.T) * (x @ up.T)`` -> ``[M, N / 2]`` bf16 for an ``interleave_gate_up`` weight.

    The first call per tactic, PDL flag and layout compiles (CuTe-DSL JIT) and must not be inside a CUDA-graph
    capture.

    Raises:
        ValueError: on the operands or tactic ``silu_takes`` / ``run_splitk_dense`` refuse.
    """
    if not silu_takes(x, weight, tactic):
        raise ValueError(
            f"silu_splitk_gemm: unsupported operands x {tuple(x.shape)} {x.dtype}, "
            f"weight {tuple(weight.shape)} {weight.dtype}, tactic {tactic}"
        )
    m, k = x.shape
    n = weight.shape[0]
    _splitk.validate_tactic(tactic, m, n, k)
    out = torch.empty((m, n // 2), dtype=x.dtype, device=x.device)
    # run_splitk_dense's operand swap (``_to_cute_swap``).
    a_swap = weight.detach().t().unsqueeze(0).transpose(-2, -1)
    b_swap = x.detach().unsqueeze(0).transpose(-2, -1)
    c_swap = out.unsqueeze(0).transpose(-2, -1)
    leading_dims = tuple(_splitk._detect_leading_dim(t) for t in (a_swap, b_swap, c_swap))
    cute_tensors = tuple(
        _splitk._from_dlpack_dynamic(t, d)
        for t, d in zip((a_swap, b_swap, c_swap), leading_dims, strict=True)
    )
    compiled = _compiled_silu(x.dtype, tactic, bool(enable_pdl), leading_dims, bool(entry_trigger))
    stream = _cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    compiled(*cute_tensors, stream)
    return out

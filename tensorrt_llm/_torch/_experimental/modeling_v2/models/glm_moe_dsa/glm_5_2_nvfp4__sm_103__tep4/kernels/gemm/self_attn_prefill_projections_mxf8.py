# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 context-step projections (kv_a, q_b, o_proj; the MoE shared expert's gate|up and
down; the dense MLP's gate|up and down; the MTP module's eh_proj) as MXF8 block-scaled GEMMs.

On B300 the context step's BF16 projection GEMMs sit at the 1.08 kW power cap; e4m3
operands with hardware block scaling (tcgen05 ``kind::mxf8f6f4``, e8m0 scale per 32
elements of K) run them at 0.55-0.62x the BF16 time at context T 7255-8018.  Numerics:
a real-prompt 78-layer forward with these projections in MXF8 keeps perplexity at the BF16
level (3.54 vs 3.59), top-1 agreement 0.911.

``C[M, N] = (A_q * SFA) @ (B_q * SFB)^T`` with A the step's activation quantized by
``self_attn_prefill_act_quant_mxf8`` (one launch) and B the weight quantized once at
load by the same kernel; FP32 accumulation; BF16 output, or E4M3 output for the FP8
context MHA's Q (``q_b_proj_e4m3``: the caller folds the power-of-two Q scale into the
weight before quantizing it, so the output is already the MHA operand).

The GEMM is the CuTe DSL ``Sm100BlockScaledPersistentDenseGemmKernel`` (vendored as
``_deps/dense_blockscaled_gemm_persistent.py``), wrapped so it takes tensors and runs through
TVM-FFI (``GlmMxf8Gemm``); the problem shape is a runtime argument, so one compiled
instance per (op, tile) serves every row count.  Each op compiles its own instance
named ``glm52_ctx_proj_mxf8_<op>`` so a census can tell them apart.

The FP8 context MHA's K / V expansion GEMMs (``kv_b_expand_k_mxf8`` K = 576,
``kv_b_expand_v512_mxf8`` K = 512, both E4M3 out) take an A operand that is already
MXF8 -- the latent rows written by ``self_attn_prefill_mha_kv_gather_mxf8`` (cached
rows as their pool bytes at scale 2^0, new rows block-quantized) -- so they run
without the activation quant (``Mxf8PrequantProjection``; the caller's SFA, or a
constant all-2^0 one).  K = 576 is not a multiple of the 128-element k-block: TMA
zero-fills the last block's upper half for A and B, and the weight is quantized
zero-padded to 640 columns, whose blocked scale layout is the one the kernel
derives for K = 576 (5 k-groups).

``kv_b_expand_kv_mxf8`` runs both expansions in ONE launch: N = 8192 = K rows 0..4095 |
V rows 4096..8191 against the K operand ``a_k [T, 576]``, with the GEMM's ``n_split_k =
(4096, 512)``: the N tiles of the V half issue only the first 4 k-tiles (c_kv, 512 columns),
so every tile issues exactly the MMAs of the two-launch path on the same bytes (``a_v`` is
``a_k``'s first 512 columns and ``sfa_v``'s codes ``sfa_k``'s first four k-groups, both written
by the same gather; the merged weight is the two folded weights stacked, the V rows zero-padded
to 576, quantized per row, so its K rows' bytes are ``kv_b_expand_k``'s and its V rows' first
512 columns ``kv_b_expand_v512``'s) and the output ``[T, 8192]`` holds the two outputs side
by side, bitwise.  One persistent tile schedule over 2 x 16 N tiles instead of two launches
of 16: one launch tail and one partial wave fewer per layer.

Pattern: glm-ctx-proj-mxf8
Device: SM100+ (Blackwell)
Shape: runtime M; (N, K) = kv_a (2624, 6144), q_b (4096, 2048), o_proj (6144, 4096),
    kv_b_expand_k (4096, 576), kv_b_expand_v512 (4096, 512), shared_up (1024, 6144),
    shared_down (6144, 512), dense_gate_up (6144, 6144), dense_down (6144, 3072),
    mtp_eh_proj (6144, 12288)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor

from ..common import self_attn_prefill_act_quant_mxf8 as _quant
from ..common.cuda_stream import raw_stream as _raw_stream
from . import dense_blockscaled_gemm_persistent as _bs

_SF_VEC = 32
_ALIGN = 16
_SF_ALIGN = 32


class GlmMxf8Gemm(_bs.Sm100BlockScaledPersistentDenseGemmKernel):
    """The CuTe DSL block-scaled persistent GEMM taking tensors (uint8 storage for the e4m3 /
    e8m0 operands) and a runtime ``(m, n, k)``; A, B K-major, C row-major."""

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        c: cute.Tensor,
        m: cutlass.Int32,
        n: cutlass.Int32,
        k: cutlass.Int32,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        a_ptr = cute.recast_ptr(a.iterator, dtype=cutlass.Float8E4M3FN)
        b_ptr = cute.recast_ptr(b.iterator, dtype=cutlass.Float8E4M3FN)
        sfa_ptr = cute.recast_ptr(sfa.iterator, dtype=cutlass.Float8E8M0FNU)
        sfb_ptr = cute.recast_ptr(sfb.iterator, dtype=cutlass.Float8E8M0FNU)
        c_ptr = c.iterator
        if cutlass.const_expr(c.element_type == cutlass.Uint8):
            c_ptr = cute.recast_ptr(c.iterator, dtype=cutlass.Float8E4M3FN)
        _bs.Sm100BlockScaledPersistentDenseGemmKernel.__call__(
            self,
            a_ptr,
            b_ptr,
            sfa_ptr,
            sfb_ptr,
            c_ptr,
            (
                OperandMajorMode.K,
                OperandMajorMode.K,
                cutlass.tensor_utils.LayoutEnum.ROW_MAJOR,
            ),
            (m, n, k, cutlass.Int32(1)),
            max_active_clusters,
            stream,
        )


@dataclass(frozen=True)
class Mxf8Abi:
    name: str
    out_features: int
    in_features: int
    out_e4m3: bool = False
    # A is given as e4m3 bytes at scale 2^0 (no activation quant; constant unit SFA)
    a_prequant: bool = False
    # ``(n_split, k_split)``: output columns >= n_split accumulate only K < k_split (the GEMM's
    # ``n_split_k``); the weight's ``[n_split:, k_split:]`` block must be zero
    n_split_k: tuple[int, int] | None = None


ABIS: dict[str, Mxf8Abi] = {
    "kv_a_proj_with_mqa": Mxf8Abi("kv_a_proj_with_mqa", 2_624, 6_144),
    "q_b_proj": Mxf8Abi("q_b_proj", 4_096, 2_048),
    "q_b_proj_e4m3": Mxf8Abi("q_b_proj_e4m3", 4_096, 2_048, out_e4m3=True),
    "o_proj": Mxf8Abi("o_proj", 6_144, 4_096),
    "kv_b_expand_k_mxf8": Mxf8Abi("kv_b_expand_k_mxf8", 4_096, 576, out_e4m3=True, a_prequant=True),
    "kv_b_expand_v512_mxf8": Mxf8Abi(
        "kv_b_expand_v512_mxf8", 4_096, 512, out_e4m3=True, a_prequant=True
    ),
    # Both expansions in one launch: [T, 576] -> K | V [T, 4096 + 4096] (module doc).
    "kv_b_expand_kv_mxf8": Mxf8Abi(
        "kv_b_expand_kv_mxf8",
        8_192,
        576,
        out_e4m3=True,
        a_prequant=True,
        n_split_k=(4_096, 512),
    ),
    # MoE shared expert gate|up (gate rows first, the SwiGLU half order), this rank's
    # 2 x 512 of the 2048-wide shared intermediate; A is the pre-MoE AR+norm's x8 operand.
    "shared_up_mxf8": Mxf8Abi("shared_up_mxf8", 1_024, 6_144),
    # MoE shared expert down, [T, 512] -> [T, 6144]; A is the SwiGLU's MXF8 operand
    # (``mlp_prefill_swiglu_mxf8``).
    "shared_down_mxf8": Mxf8Abi("shared_down_mxf8", 6_144, 512),
    # Dense MLP (layers 0-2) gate|up, this rank's 2 x 3072 of the 12288-wide intermediate
    # (gate rows first); A is the pre-MLP AR+norm's x8 operand or the act quant's.
    "dense_gate_up_mxf8": Mxf8Abi("dense_gate_up_mxf8", 6_144, 6_144),
    # Dense MLP down, [T, 3072] -> [T, 6144]; A is the SwiGLU's MXF8 operand.
    "dense_down_mxf8": Mxf8Abi("dense_down_mxf8", 6_144, 3_072),
    # MTP eh_proj at context row counts (replicated on every rank), [T, 12288] -> [T, 6144];
    # A is concat(enorm(e), hnorm(h)) written as MXF8 by ``mtp_eh_norm_cat_mxf8``.
    "mtp_eh_proj_mxf8": Mxf8Abi("mtp_eh_proj_mxf8", 6_144, 12_288),
}

# (mma tiler (M, N), cluster (M, N)).  256x256 2-CTA was the best tile at every op at
# context T >= 7k in a power-capped B300 selection; 256x128 2-CTA wins at small T (more
# CTAs on the N side: kv_a T=65 10.8 vs 17.8 us, o_proj T=1024 26.5 vs 28.2 us), so each op
# takes it up to ``SMALL_TILE_MAX_ROWS[op]`` rows.
TILES: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "256x256": ((256, 256), (2, 1)),
    "256x128": ((256, 128), (2, 1)),
    "128x256": ((128, 256), (1, 1)),
}
DEFAULT_TILE = "256x256"
SMALL_TILE = "256x128"
SMALL_TILE_MAX_ROWS: dict[str, int] = {
    "kv_a_proj_with_mqa": 4_096,
    "q_b_proj": 512,
    "q_b_proj_e4m3": 512,
    "o_proj": 1_024,
    "kv_b_expand_k_mxf8": 1_024,
    "kv_b_expand_v512_mxf8": 1_024,
    # the two-launch path's tile per row count, so the merged launch's tiles are its tiles
    "kv_b_expand_kv_mxf8": 1_024,
    "shared_up_mxf8": 1_024,
    "shared_down_mxf8": 1_024,
    "dense_gate_up_mxf8": 1_024,
    "dense_down_mxf8": 1_024,
    "mtp_eh_proj_mxf8": 1_024,
}

Compiled = Callable[..., object]
_COMPILED: dict[tuple[str, str, int, bool], Compiled] = {}


def compile_gemm(
    op: str, tile: str = DEFAULT_TILE, device_index: int = 0, pdl: bool = False
) -> Compiled:
    """``fn(a[M,K] u8, b[N,K] u8, sfa u8, sfb u8, c[M,N] bf16|u8, M, N, K, stream)``.

    ``pdl``: the instance launched as a programmatic dependent that issues its weight (B /
    SFB) boxes of every pipeline stage before ``griddepcontrol.wait`` (``pdl_weights_first``;
    same MMAs, bitwise the plain instance's output), named ``..._pdl``: for a decode chain
    whose predecessor writes A. The context path keeps the plain instance."""
    key = (op, tile, device_index, pdl)
    if key not in _COMPILED:
        abi = ABIS[op]
        if pdl and abi.n_split_k is not None:
            raise ValueError(f"{op}: the PDL instance does not take n_split_k")
        tiler, cluster = TILES[tile]
        gemm = GlmMxf8Gemm(_SF_VEC, tiler, cluster)
        gemm.n_split_k = abi.n_split_k
        gemm.pdl_weights_first = pdl
        gemm.kernel.set_name_prefix(
            f"glm52_ctx_proj_mxf8_{op}" + ("_pdl" if pdl else ""), remove_cutlass_symbol=True
        )
        mac = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster[0] * cluster[1])
        c_dtype = cutlass.Uint8 if abi.out_e4m3 else cutlass.BFloat16
        _COMPILED[key] = cute.compile(
            gemm,
            make_fake_tensor(
                cutlass.Uint8,
                (cute.sym_int32(), abi.in_features),
                stride=(abi.in_features, 1),
                assumed_align=_ALIGN,
            ),
            make_fake_tensor(
                cutlass.Uint8,
                (abi.out_features, abi.in_features),
                stride=(abi.in_features, 1),
                assumed_align=_ALIGN,
            ),
            make_fake_compact_tensor(cutlass.Uint8, (cute.sym_int32(),), assumed_align=_SF_ALIGN),
            make_fake_compact_tensor(cutlass.Uint8, (cute.sym_int32(),), assumed_align=_SF_ALIGN),
            make_fake_tensor(
                c_dtype,
                (cute.sym_int32(), abi.out_features),
                stride=(abi.out_features, 1),
                assumed_align=_ALIGN,
            ),
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int32(0),
            mac,
            make_fake_stream(),
            options="--enable-tvm-ffi",
        )
    return _COMPILED[key]


def can_implement(op: str, rows: int, tile: str = DEFAULT_TILE) -> bool:
    abi = ABIS[op]
    tiler, cluster = TILES[tile]
    return _bs.Sm100BlockScaledPersistentDenseGemmKernel.can_implement(
        (rows, abi.out_features, abi.in_features, 1),
        cutlass.Float8E4M3FN,
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        cutlass.Float8E4M3FN if abi.out_e4m3 else cutlass.BFloat16,
        "k",
        "k",
        "n",
        _SF_VEC,
        tiler,
        cluster,
    )


@dataclass
class QuantizedWeight:
    """A projection weight in MXF8: e4m3 bytes ``[N, K]`` and blocked e8m0 scales."""

    q: torch.Tensor
    sf: torch.Tensor


def _k_padded(k: int) -> int:
    return (k + 127) // 128 * 128


def quantize_weight(weight: torch.Tensor) -> QuantizedWeight:
    """Quantize a BF16 ``[N, K]`` weight once (same kernel as the activations).  A K that
    is not a multiple of 128 is quantized zero-padded to the next multiple (the pad
    blocks get finite scales and zero values); ``q`` keeps the first K columns
    (contiguous ``[N, K]``) and ``sf`` the padded blocked layout, which is the one the
    GEMM derives for K."""
    fn = _quant.compile_quant(weight.device.index)
    n, k = weight.shape
    kp = _k_padded(k)
    if kp == k:
        q, sf = _quant.quantize(fn, weight.contiguous())
        return QuantizedWeight(q, sf)
    wp = torch.zeros(n, kp, dtype=weight.dtype, device=weight.device)
    wp[:, :k] = weight
    q, sf = _quant.quantize(fn, wp)
    return QuantizedWeight(q[:, :k].contiguous(), sf)


def unit_sf_bytes(rows: int, k: int) -> int:
    """Bytes of an all-2^0 blocked SFA for ``rows x k`` (k padded to 128)."""
    return _quant.sf_bytes(rows, _k_padded(k))


def unit_sf(rows: int, k: int, device: torch.device) -> torch.Tensor:
    """The constant SFA of an e4m3 operand at scale 2^0 (e8m0 code 127 everywhere)."""
    return torch.full((unit_sf_bytes(rows, k),), 127, dtype=torch.uint8, device=device)


class Mxf8Projection:
    """One op's launcher: ``__call__(x, w, out)`` quantizes the BF16 activation ``x [T, K]``
    (one launch) and runs the block-scaled GEMM against the quantized weight ``w`` into
    ``out [T, N]`` (BF16, or uint8 e4m3 bytes for ``q_b_proj_e4m3``).  The weight is an
    argument (one launcher serves every layer).  The tile is chosen per row count
    (``SMALL_TILE`` up to ``SMALL_TILE_MAX_ROWS[op]``, ``DEFAULT_TILE`` above);
    ``choice_hits`` counts launches per tile."""

    def __init__(self, op: str, device_index: int = 0):
        self.abi = ABIS[op]
        self.device_index = device_index
        self.small_max_rows = SMALL_TILE_MAX_ROWS[op]
        self.tiles = (SMALL_TILE, DEFAULT_TILE)
        self._quant = _quant.compile_quant(device_index)
        self._gemms = {t: compile_gemm(op, t, device_index) for t in self.tiles}
        self.choice_hits = {t: 0 for t in self.tiles}
        # launches whose operand came pre-quantized (``pre``)
        self.pre_hits = 0

    def tile_for(self, rows: int) -> str:
        return SMALL_TILE if rows <= self.small_max_rows else DEFAULT_TILE

    def warm_rows(self) -> tuple[int, ...]:
        """One row count per compiled tile (each tile is its own kernel to load)."""
        return (min(128, self.small_max_rows), self.small_max_rows + 1)

    def __call__(
        self,
        x: torch.Tensor,
        w: QuantizedWeight,
        out: torch.Tensor,
        pre: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """``pre = (xq, sfa)``: ``x`` already quantized by its producer (a fused-quant
        epilogue writing ``quantize``'s exact bytes); the quant launch is skipped and ``x``
        is only read for its row count."""
        rows = int(x.shape[0])
        k, n = self.abi.in_features, self.abi.out_features
        if x.shape[1] != k or tuple(out.shape) != (rows, n) or not out.is_contiguous():
            raise ValueError(
                f"{self.abi.name} mxf8: x [T,{k}] -> out [T,{n}] contiguous required, "
                f"got {tuple(x.shape)} -> {tuple(out.shape)}"
            )
        if out.dtype != (torch.uint8 if self.abi.out_e4m3 else torch.bfloat16):
            raise ValueError(f"{self.abi.name} mxf8: out dtype {out.dtype}")
        if pre is not None:
            xq, sfa = pre
            if (
                xq.dtype != torch.uint8
                or tuple(xq.shape) != (rows, k)
                or not xq.is_contiguous()
                or xq.data_ptr() % _ALIGN
                or sfa.dtype != torch.uint8
                or not sfa.is_contiguous()
                or sfa.numel() < _quant.sf_bytes(rows, k)
                or sfa.data_ptr() % _SF_ALIGN
            ):
                raise ValueError(
                    f"{self.abi.name} mxf8: pre-quantized operand must be uint8 [{rows},{k}] "
                    f"contiguous + >= sf_bytes scale bytes, got {tuple(xq.shape)} {xq.dtype} / "
                    f"{sfa.numel()} {sfa.dtype}"
                )
            self.pre_hits += 1
        else:
            xq = torch.empty(rows, k, dtype=torch.uint8, device=x.device)
            sfa = torch.empty(_quant.sf_bytes(rows, k), dtype=torch.uint8, device=x.device)
            _quant.quantize(self._quant, x, xq, sfa)
        tile = self.tile_for(rows)
        self.choice_hits[tile] += 1
        self._gemms[tile](
            xq,
            w.q,
            sfa,
            w.sf,
            out,
            rows,
            n,
            k,
            _raw_stream(x.device),
        )
        return out


class Mxf8PrequantProjection:
    """A launcher for an op whose A operand is already MXF8 (``Mxf8Abi.a_prequant``):
    ``__call__(a8, w, out, sfa=None)`` runs the block-scaled GEMM of the uint8 ``a8 [T,
    K]`` (contiguous) with its blocked scales ``sfa`` (``unit_sf_bytes(T, K)`` bytes at
    least; None = a constant all-2^0 buffer sized for ``max_rows``) against the
    quantized weight into ``out [T, N]`` (uint8 e4m3 bytes).  T <= ``max_rows`` (the
    caller slices).  Same tile choice and counters as ``Mxf8Projection``."""

    def __init__(self, op: str, device_index: int = 0, max_rows: int = 16_384):
        self.abi = ABIS[op]
        if not self.abi.a_prequant:
            raise ValueError(f"{op}: not a pre-quantized-A op")
        self.device_index = device_index
        self.max_rows = max_rows
        self.small_max_rows = SMALL_TILE_MAX_ROWS[op]
        self.tiles = (SMALL_TILE, DEFAULT_TILE)
        self._gemms = {t: compile_gemm(op, t, device_index) for t in self.tiles}
        self.choice_hits = {t: 0 for t in self.tiles}
        self.sfa = unit_sf(max_rows, self.abi.in_features, torch.device("cuda", device_index))

    def tile_for(self, rows: int) -> str:
        return SMALL_TILE if rows <= self.small_max_rows else DEFAULT_TILE

    def warm_rows(self) -> tuple[int, ...]:
        return (min(128, self.small_max_rows), self.small_max_rows + 1)

    def __call__(
        self,
        a8: torch.Tensor,
        w: QuantizedWeight,
        out: torch.Tensor,
        sfa: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rows = int(a8.shape[0])
        k, n = self.abi.in_features, self.abi.out_features
        if sfa is None:
            sfa = self.sfa
        elif (
            sfa.dtype != torch.uint8
            or not sfa.is_contiguous()
            or sfa.numel() < unit_sf_bytes(rows, k)
            or sfa.data_ptr() % _SF_ALIGN != 0
        ):
            raise ValueError(
                f"{self.abi.name} mxf8: sfa must be contiguous uint8 of >= "
                f"{unit_sf_bytes(rows, k)} bytes, {_SF_ALIGN}-byte aligned"
            )
        if (
            a8.dtype != torch.uint8
            or a8.ndim != 2
            or a8.shape[1] != k
            or not a8.is_contiguous()
            or out.dtype != torch.uint8
            or tuple(out.shape) != (rows, n)
            or not out.is_contiguous()
            or not 1 <= rows <= self.max_rows
        ):
            raise ValueError(
                f"{self.abi.name} mxf8: a8 u8 [T,{k}] -> out u8 [T,{n}] contiguous, "
                f"1 <= T <= {self.max_rows}; got {tuple(a8.shape)} {a8.dtype} -> "
                f"{tuple(out.shape)} {out.dtype}"
            )
        tile = self.tile_for(rows)
        self.choice_hits[tile] += 1
        self._gemms[tile](
            a8,
            w.q,
            sfa,
            w.sf,
            out,
            rows,
            n,
            k,
            _raw_stream(a8.device),
        )
        return out


def build(op: str, device_index: int = 0) -> Mxf8Projection | Mxf8PrequantProjection:
    if op not in ABIS:
        raise ValueError(f"unknown mxf8 projection {op!r}; known: {tuple(ABIS)}")
    if ABIS[op].a_prequant:
        return Mxf8PrequantProjection(op, device_index)
    return Mxf8Projection(op, device_index)


def dequantize_weight(w: QuantizedWeight) -> torch.Tensor:
    """FP32 values of a quantized weight (K padded to 128 for the scale lookup)."""
    n, k = w.q.shape
    kp = _k_padded(k)
    q = w.q
    if kp != k:
        q = torch.zeros(n, kp, dtype=torch.uint8, device=w.q.device)
        q[:, :k] = w.q
    return _quant.dequantize(q, w.sf, n, kp)[:, :k]


def emulate(
    x: torch.Tensor,
    w: QuantizedWeight,
    k: int,
    sfa: torch.Tensor | None = None,
    n_split_k: tuple[int, int] | None = None,
) -> torch.Tensor:
    """FP32 product of the MXF8 operands the GEMM reads: a bf16 ``x`` quantized by the
    activation kernel's torch twin, or a pre-quantized uint8 ``x`` with its blocked scales
    ``sfa`` (None: all 2^0), against the dequantized weight. ``n_split_k = (n, kk)``: output
    columns >= n read K < kk only (the weight's ``[n:, kk:]`` block taken as zero)."""
    b = dequantize_weight(w)
    if n_split_k is not None:
        b = b.clone()
        b[n_split_k[0] :, n_split_k[1] :] = 0.0
    if x.dtype == torch.uint8:
        if sfa is None:
            return x.view(torch.float8_e4m3fn).float() @ b.t()
        rows, kp = x.shape[0], _k_padded(k)
        q = torch.zeros(rows, kp, dtype=torch.uint8, device=x.device)
        q[:, :k] = x
        a = _quant.dequantize(q, sfa, rows, kp)[:, :k]
        return a @ b.t()
    xq, sfa = _quant.reference(x)
    a = _quant.dequantize(xq, sfa, x.shape[0], k)
    return a @ b.t()

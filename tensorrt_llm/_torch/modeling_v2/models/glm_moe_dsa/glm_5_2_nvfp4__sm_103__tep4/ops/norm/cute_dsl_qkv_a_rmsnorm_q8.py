# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MLA q_a and kv_a RMSNorms in one CuTe DSL launch that also writes the normalized q_a
rows as the MXF8 operand of the q_b GEMM that reads them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.common import self_attn_prefill_act_quant_mxf8 as _quant
from ...kernels.norm import self_attn_qkv_a_rmsnorm as _kernel
from .cute_dsl_qkv_a_rmsnorm import _Q, _ROW, WARP_ROW_MIN_ROWS, _served_layout, _stream

#: Decode rows the one-CTA-per-row q8 kernel serves: the CUDA-graph batches, 1..64.
DECODE_MAX_ROWS = 64


@dataclass(frozen=True)
class CompiledQkvARmsnormQ8:
    """The three q8 kernels `compile` built for one `(device, eps)`.

    `warp_row`: context rows, the MXF8 copy of the first `R8` rows. `warp_row_ctx`: the same,
    without storing the bf16 q rows below `q_store_from` (a caller whose only bf16 reader is
    past them). `cta`: decode rows, one CTA per row, releasing its dependent (the q_b GEMM) at
    entry. `dummies`: the context variant's pool / block-offset / prefix-sum operands, read
    only by a kv-append build this entry does not compile."""

    warp_row: Callable
    warp_row_ctx: Callable
    cta: Callable
    dummies: tuple
    #: `warp_row_ctx` is the ILP build.
    ilp: bool = False


def compile(device_id: int, eps: float, ilp: bool = True) -> CompiledQkvARmsnormQ8:  # noqa: A001
    """Compile the three kernels for `device_id` with `eps` baked in. Call once, at load; the
    kernel module caches each per key, so a second call is free. `ilp` builds the variant that
    skips the low bf16 q rows with fewer instructions per element (packed multiplies and bf16
    rounding, a min/max tree for the block amax, the power-of-two scale's reciprocal from its
    exponent): same output bytes, -1.3 .. -1.5 us per launch at 7-8k rows on B300."""
    dev = torch.device("cuda", device_id)
    return CompiledQkvARmsnormQ8(
        warp_row=_kernel.compile_qkv_a_rmsnorm_warp_row_q8(
            device_id, float(eps), inplace_kv_view=True
        ),
        warp_row_ctx=_kernel.compile_qkv_a_rmsnorm_warp_row_q8_ctx(
            device_id, float(eps), ilp=bool(ilp)
        ),
        cta=_kernel.compile_qkv_a_rmsnorm_q8(
            device_id, float(eps), inplace_kv_view=True, early_signal=True
        ),
        dummies=(
            torch.zeros(16, dtype=torch.uint8, device=dev),
            torch.zeros(1, 2, 1, dtype=torch.int32, device=dev),
            torch.zeros(2, dtype=torch.int32, device=dev),
            torch.zeros(1, dtype=torch.int32, device=dev),
        ),
        ilp=bool(ilp),
    )


def operand(rows: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Buffers of a q_a MXF8 operand for `rows` rows: e4m3 bytes `[rows, 2048]` and the
    blocked e8m0 scales (rows padded to 128), the act quant's layout."""
    return (
        torch.empty(rows, _Q, dtype=torch.uint8, device=device),
        torch.empty(_quant.sf_bytes(rows, _Q), dtype=torch.uint8, device=device),
    )


def takes(q_raw, q_weight, ckv_raw, kv_weight) -> bool:
    """Whether a call can take this entry: the target's in-place layout, 16-byte aligned
    weights, and on a context row count the warp-row kernel's 16-byte contract."""
    rows = int(q_raw.shape[0])
    if not _served_layout(q_raw, q_weight, ckv_raw, kv_weight):
        return False
    if q_weight.data_ptr() % 16 or kv_weight.data_ptr() % 16:
        return False
    if rows <= DECODE_MAX_ROWS:
        return True
    return rows >= WARP_ROW_MIN_ROWS and _kernel._use_warp_row_path(
        q_raw, ckv_raw, q_weight, kv_weight
    )


class CuteDslQkvARmsnormQ8(OpWrapper):
    """`cute_dsl_qkv_a_rmsnorm` plus the MXF8 bytes of the first `R8 = q8.shape[0]` normalized
    q_a rows, one launch.

    `q_raw` `[M, 2048]` and `ckv_raw` `[M, 512]` are the first two column slices of one
    row-major `[M, 2624]` projection output; the q result is a fresh compact `[M, 2048]` bf16
    tensor and the kv result is written into `ckv_raw` (the latent's head), as in
    `cute_dsl_qkv_a_rmsnorm`. `q8` (uint8 `[R8, 2048]`, contiguous, `R8 <= M`) and `sf` (at
    least `sf_bytes(R8, 2048)` bytes, 32-byte aligned) receive the activation quant's exact
    bytes of those q rows: per 32-element block, a power-of-two scale from the block's amax and
    e4m3 values. The bf16 q rows are the same bytes as `cute_dsl_qkv_a_rmsnorm` writes.

    Decode rows (`M <= 64`) run the one-CTA-per-row kernel and quantize every row (`R8 = M`);
    it releases its PDL dependent at entry, so the q_b GEMM that reads `q8` can load its weight
    under the norm. Context rows run the warp-per-row kernel; with `q_store_from > 0` the bf16
    q rows below it are not stored (they hold garbage), for a caller whose only bf16 reader
    is past them.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="decode, one token", spec=dict(rows=1, r8=1)),
        Cell(why="the full decode batch, 32 sequences x MTP 2", spec=dict(rows=64, r8=64)),
        Cell(
            why="the first warp-row context count, every row quantized",
            spec=dict(rows=WARP_ROW_MIN_ROWS, r8=WARP_ROW_MIN_ROWS),
        ),
        Cell(
            why="a mixed step: 7239 context rows quantized, 16 generation rows after them",
            spec=dict(rows=7255, r8=7239),
        ),
        Cell(
            why="the bf16 q rows below q_store_from skipped, the last 64 context rows kept",
            spec=dict(rows=7255, r8=7239, q_store_from=7175),
        ),
        Cell(why="max_num_tokens", spec=dict(rows=8192, r8=8192)),
    )

    note = """
    The target compiles one handle at load with the checkpoint's eps. On a pure decode step
    (the CUDA graphs) the q_a norm writes the decode q_b GEMM's MXF8 operand straight into
    that GEMM's scratch, so q_b runs without its own activation quant. On the fp8 context
    attention's steps the norm writes the context rows' q_a as the MXF8 operand of q_b's
    e4m3 GEMM; when no indexer reads q_a that step, `q_store_from` skips the bf16 rows only
    the generation rows' q_b tail could read (it reads at most the last 64 rows).
    """

    def __call__(
        self,
        q_raw: torch.Tensor,
        q_weight: torch.Tensor,
        ckv_raw: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
        q8: torch.Tensor,
        sf: torch.Tensor,
        compiled: CompiledQkvARmsnormQ8,
        q_store_from: int = 0,
    ) -> torch.Tensor:
        rows = int(q_raw.shape[0])
        q_out = torch.empty((rows, _Q), dtype=torch.bfloat16, device=q_raw.device)
        stream = _stream(q_raw.device)
        if rows <= DECODE_MAX_ROWS:
            compiled.cta(
                q_raw, q_weight, ckv_raw, kv_weight, q_out, ckv_raw, q8, sf,
                _ROW, _ROW, _Q, _ROW, stream,
            )  # fmt: skip
        elif q_store_from > 0:
            pool, offsets, cu, cached = compiled.dummies
            compiled.warp_row_ctx(
                q_raw, q_weight, ckv_raw, kv_weight, q_out, ckv_raw, q8, sf,
                pool.data_ptr(), offsets, cu, cached, _ROW, _ROW, _Q, _ROW,
                int(q_store_from), stream,
            )  # fmt: skip
        else:
            compiled.warp_row(
                q_raw, q_weight, ckv_raw, kv_weight, q_out, ckv_raw, q8, sf,
                _ROW, _ROW, _Q, _ROW, stream,
            )  # fmt: skip
        return q_out

    def reference(
        self,
        q_raw: torch.Tensor,
        q_weight: torch.Tensor,
        ckv_raw: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
        q8: torch.Tensor,
        sf: torch.Tensor,
        compiled: CompiledQkvARmsnormQ8,
        q_store_from: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(q_a, ckv, q8, sf)` as new tensors: the norms in fp64 rounded once to bf16, and the
        activation quant's torch twin applied to the first `R8` rows of that q_a. Does not
        write `ckv_raw`; call it before the op. `compiled` and `q_store_from` are ignored."""

        def rms(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            xd = x.double()
            inv = torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + eps)
            return (xd * inv * w.double()).to(x.dtype)

        q_a = rms(q_raw, q_weight)
        ref_q8, ref_sf = _quant.reference(q_a[: q8.shape[0]].contiguous())
        return q_a, rms(ckv_raw, kv_weight), ref_q8, ref_sf

    def is_valid(
        self,
        q_raw: torch.Tensor,
        q_weight: torch.Tensor,
        ckv_raw: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
        q8: torch.Tensor,
        sf: torch.Tensor,
        compiled: CompiledQkvARmsnormQ8,
        q_store_from: int = 0,
    ) -> None:
        rows = int(q_raw.shape[0])
        assert isinstance(compiled, CompiledQkvARmsnormQ8), (
            "compiled must be the handle this entry's compile() returns"
        )
        assert takes(q_raw, q_weight, ckv_raw, kv_weight), (
            "the q8 kernels need the target's in-place [M, 2624] layout, 16-byte aligned "
            f"weights, and M <= {DECODE_MAX_ROWS} or M >= {WARP_ROW_MIN_ROWS}"
        )
        r8 = int(q8.shape[0])
        assert q8.dtype == torch.uint8 and q8.is_contiguous() and tuple(q8.shape) == (r8, _Q), (
            f"q8 must be a contiguous uint8 [R8, {_Q}], got {tuple(q8.shape)} {q8.dtype}"
        )
        assert 0 < r8 <= rows, f"R8 must be in 1..{rows}, got {r8}"
        if rows <= DECODE_MAX_ROWS:
            assert r8 == rows, "the decode kernel quantizes every row"
            assert q_store_from == 0, "the decode kernel stores every bf16 q row"
        assert q8.data_ptr() % 16 == 0, "q8 must be 16-byte aligned"
        need = _quant.sf_bytes(r8, _Q)
        assert (
            sf.dtype == torch.uint8 and sf.is_contiguous() and sf.numel() >= need
            and sf.data_ptr() % 32 == 0
        ), f"sf must be contiguous uint8 of at least {need} bytes, 32-byte aligned"  # fmt: skip
        assert 0 <= q_store_from <= r8, f"q_store_from must be in 0..{r8}, got {q_store_from}"
        for cache, fn, name in (
            (_kernel._WARP_ROW_Q8_COMPILE_CACHE, compiled.warp_row, "warp_row"),
            (_kernel._WARP_ROW_Q8_CTX_COMPILE_CACHE, compiled.warp_row_ctx, "warp_row_ctx"),
            (_kernel._Q8_COMPILE_CACHE, compiled.cta, "cta"),
        ):
            keys = [k for k, f in cache.items() if f is fn]
            assert keys, f"compiled.{name} must be a handle from this entry's compile()"
            device_id, compiled_eps = keys[0][:2]
            assert compiled_eps == float(eps), (
                f"compiled.{name} has eps={compiled_eps} baked in, the call passes {eps}"
            )
            assert device_id == q_raw.device.index, (
                f"compiled.{name} is for device {device_id}, the tensors are on {q_raw.device}"
            )


cute_dsl_qkv_a_rmsnorm_q8 = CuteDslQkvARmsnormQ8()

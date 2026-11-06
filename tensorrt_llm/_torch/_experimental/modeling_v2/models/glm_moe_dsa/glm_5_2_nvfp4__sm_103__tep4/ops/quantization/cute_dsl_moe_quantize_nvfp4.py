# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4-quantize the routed experts' bf16 input, linear e4m3 scales, in CuTe DSL: an
exact-row kernel per decode width and a row-segmented kernel for context steps."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cutlass.torch as cutlass_torch
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)
from tensorrt_llm._torch.kernels.blackwell.utils import TRTLLM_ENABLE_PDL as _USE_PDL

from ...kernels.moe import moe_prefill_frontend_schedule as _sched
from ...kernels.quantization import moe_frontend_quantize_nvfp4 as _decode
from ...kernels.quantization import moe_prefill_quantize_segmented as _prefill

HIDDEN = 6144
SF_VEC = 16
DECODE_MAX_ROWS = 64
PREFILL_MIN_ROWS = DECODE_MAX_ROWS + 1
PREFILL_MAX_ROWS = 8192


def _prefill_flags(segments: int) -> dict:
    """Waits for its producer under PDL but does not release FC1 early: the prefill FC1
    reads A right after its own wait. The zero plane is not fused (the output clear is
    its own launch, skipped when the shared expert already wrote the output)."""
    return {
        "wait_for_prior_pdl": _USE_PDL,
        "signal_gg1_pdl": False,
        "fuse_zero": False,
        "segments_per_row": segments,
    }


def quantize_reference(
    x: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(packed codes [rows, 3072], scale bytes [rows, 384], scaled values [rows, 6144])`
    in the kernels' scale arithmetic, plain torch.

    Both kernels form the block scale as `global_scale * amax`, then times the fp32
    constant `1/6`, each an fp32 multiply rounded to nearest, then round it to e4m3 with
    saturation at 448. `fp4_quantize.reference` forms `global_scale * (amax / 6)`
    instead: one rounding in another place, which moves a product sitting within an fp32
    ulp or two of an e4m3 rounding boundary to the other code. Mirroring the order makes
    the scales exact. The codes are the e2m1 rounding (to nearest even) of `x *
    global_scale / scale`, computed exactly here; `scaled` is that pre-rounding value,
    which `fp4_quantize.near_tie` reads.
    """
    rows = int(x.shape[0])
    blocks = x.float().reshape(rows, HIDDEN // SF_VEC, SF_VEC)
    amax = blocks.abs().amax(-1)
    gs = global_scale.float().reshape(())
    sixth = torch.tensor(1.0 / 6.0, dtype=torch.float32, device=x.device)
    sf = ((gs * amax) * sixth).clamp(max=448.0).to(torch.float8_e4m3fn)
    out_scale = torch.where(amax != 0, gs / sf.float(), torch.zeros_like(amax))
    scaled = (blocks * out_scale.unsqueeze(-1)).reshape(rows, HIDDEN)
    codes = fp4_quantize._e2m1_codes(scaled)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed, sf.view(torch.uint8), scaled


@dataclass(frozen=True)
class CompiledMoeQuantize:
    """A compiled quantizer. Decode: pinned to `rows`, no PDL role. Prefill: `rows` None
    (65..8192), PDL wait only, the row segments chosen per call by row count.
    `launch(x, global_scale, fp4_out, scale_out)` is the kernel call."""

    kind: str
    rows: int | None
    launch: Callable[..., None]


def build_decode(rows: int, device) -> CompiledMoeQuantize:
    """Compile the exact-row quantizer for `rows` rows (1..64), no PDL role, no zero
    plane. Call at load. Compiles only (cached per process by the kernel module)."""
    rows = int(rows)
    bf = {"dtype": torch.bfloat16, "device": device}
    u8 = {"dtype": torch.uint8, "device": device}
    x = torch.zeros(rows, HIDDEN, **bf)
    with torch.cuda.device(device):
        compiled = _decode.compile_quantize_linear(
            x,
            torch.ones(1, dtype=torch.float32, device=device),
            torch.zeros(rows, HIDDEN // 2, **u8),
            torch.zeros(rows, HIDDEN // SF_VEC, **u8),
            x,
        )
    as_t = _decode._as_tensor

    def launch(x, global_scale, fp4_out, scale_out):
        # The unfused zero plane is in the ABI but never touched; `x` stands in for it.
        compiled(
            as_t(x, 1),
            as_t(global_scale),
            as_t(fp4_out, 1),
            as_t(scale_out, 1),
            as_t(x, 1),
            cutlass_torch.current_stream(),
        )

    return CompiledMoeQuantize("decode", rows, launch)


def build_prefill(device) -> CompiledMoeQuantize:
    """Compile the segmented quantizer per row-segment count its domain selects
    (`propose_schedule(rows).quant_segments`: 3 up to 2048 rows, 1 above), runtime row
    count, and launch each once on zeros. Call at load. Raises if a compile fails: the
    target then calls `fp4_quantize`."""
    first: dict[int, int] = {}
    for rows in range(PREFILL_MIN_ROWS, PREFILL_MAX_ROWS + 1):
        first.setdefault(_sched.propose_schedule(rows).quant_segments, rows)
    handles: dict[int, Callable[..., None]] = {}
    bf = {"dtype": torch.bfloat16, "device": device}
    u8 = {"dtype": torch.uint8, "device": device}
    g1 = torch.ones(1, dtype=torch.float32, device=device)
    with torch.cuda.device(device):
        for seg, rows in first.items():
            args = (
                torch.zeros(rows, HIDDEN, **bf),
                g1,
                torch.zeros(rows, HIDDEN // 2, **u8),
                torch.zeros(rows, HIDDEN // SF_VEC, **u8),
                torch.zeros(rows, HIDDEN, **bf),
            )
            # Compiles on a symbolic row count; the run is the warm launch, the compile
            # call then returns the cached handle.
            _prefill.run_quantize_linear(*args, **_prefill_flags(seg))
            handles[seg] = _prefill.compile_quantize_linear(*args, **_prefill_flags(seg))

    def launch(x, global_scale, fp4_out, scale_out):
        rows = int(x.shape[0])
        h = handles[_sched.propose_schedule(rows).quant_segments]
        # The unfused zero plane is in the ABI but never touched; `x` stands in for it.
        h(x, global_scale.reshape(1), fp4_out, scale_out, x, rows, cutlass_torch.current_stream())

    return CompiledMoeQuantize("prefill", None, launch)


class CuteDslMoeQuantizeNvfp4(OpWrapper):
    """NVFP4 quantization of `x` into caller-owned buffers. Returns `(fp4_out, scale_out)`.

    `x` is bf16 `[rows, 6144]`; `global_scale` one fp32 element; `fp4_out` uint8
    `[rows, 3072]` (two e2m1 codes per byte, element `2i` in the low nibble);
    `scale_out` uint8 `[rows, 384]`, one e4m3 scale per 16 elements in row-major
    (linear) order -- the layout the routed experts' first grouped GEMM gathers from.
    The scale is `e4m3((global_scale * amax) * fp32(1/6))`: the product formed in that
    order (see `quantize_reference`). The stock `fp4_quantize(x, global_scale, 16, False,
    False)` writes the same layout but forms `global_scale * (amax * 1/6)`, which can
    land on the neighbouring e4m3 code when the exact product sits within a few fp32 ulps
    of an e4m3 rounding boundary.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="one decode row on the exact-row kernel", spec=dict(rows=1)),
        Cell(why="64 rows, the exact-row kernel's ceiling", spec=dict(rows=64)),
        Cell(why="65 rows, the first prefill count: three segments per row", spec=dict(rows=65)),
        Cell(why="2048 rows, the last three-segment count", spec=dict(rows=2048)),
        Cell(why="2049 rows, the first one-segment count", spec=dict(rows=2049)),
        Cell(why="8192 rows, the token budget", spec=dict(rows=8192)),
    )

    note = """
    The trunk's decode rows arrive already quantized by the closing collective's
    epilogue, so the decode handles (built per power-of-two width up to 64) serve only a
    remainder chunk past the 8192-row budget, which no configured step produces. A
    context step takes the prefill handle when it built, else `fp4_quantize` at the call
    site (which allocates its own outputs and returns 1-D scales; the grouped GEMMs read
    either form).

    Scales match `quantize_reference` bit for bit. Codes match it outside
    `fp4_quantize.TIE_WINDOW` of an e2m1 midpoint: the kernels scale the values through
    approximate reciprocals, as the stock op does. The handles skip the shape checks the
    kernel modules' runners repeat per call; `is_valid` states them.
    """

    def __call__(
        self,
        x: torch.Tensor,
        global_scale: torch.Tensor,
        fp4_out: torch.Tensor,
        scale_out: torch.Tensor,
        compiled: CompiledMoeQuantize,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        compiled.launch(x, global_scale, fp4_out, scale_out)
        return fp4_out, scale_out

    def reference(
        self,
        x: torch.Tensor,
        global_scale: torch.Tensor,
        fp4_out: torch.Tensor,
        scale_out: torch.Tensor,
        compiled: CompiledMoeQuantize,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`quantize_reference(x, global_scale)`'s codes and scales. Fresh tensors; the
        buffers and `compiled` are not read."""
        packed, sf, _ = quantize_reference(x, global_scale)
        return packed, sf

    def is_valid(
        self,
        x: torch.Tensor,
        global_scale: torch.Tensor,
        fp4_out: torch.Tensor,
        scale_out: torch.Tensor,
        compiled: CompiledMoeQuantize,
    ) -> None:
        # The handles read raw extents: a strided or short buffer is written as if dense,
        # and a per-row scale tensor applies its first element to every row.
        rows = int(x.shape[0])
        if compiled.kind == "decode":
            assert rows == compiled.rows, f"compiled for {compiled.rows} rows, got {rows}"
        else:
            assert PREFILL_MIN_ROWS <= rows <= PREFILL_MAX_ROWS
        assert global_scale.dtype == torch.float32 and global_scale.numel() == 1, (
            "global_scale must hold one fp32 element"
        )
        for name, t, shape, dtype in (
            ("x", x, (rows, HIDDEN), torch.bfloat16),
            ("fp4_out", fp4_out, (rows, HIDDEN // 2), torch.uint8),
            ("scale_out", scale_out, (rows, HIDDEN // SF_VEC), torch.uint8),
        ):
            assert tuple(t.shape) == shape and t.dtype == dtype, (
                f"{name} must be {dtype} {list(shape)}, got {t.dtype} {tuple(t.shape)}"
            )
            assert t.is_contiguous(), f"{name} must be contiguous"
            assert t.data_ptr() % 16 == 0, f"{name} must be 16-byte aligned"


cute_dsl_moe_quantize_nvfp4 = CuteDslMoeQuantizeNvfp4()

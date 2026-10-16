# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Clear the routed experts' bf16 output before the second grouped GEMM accumulates into it,
in CuTe DSL: an exact-row kernel per decode width and a byte-clear for context steps."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cutlass.torch as cutlass_torch
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.kernels.blackwell.utils import TRTLLM_ENABLE_PDL as _USE_PDL

from ...kernels.moe import moe_frontend_zero as _decode
from ...kernels.moe import moe_prefill_memory as _prefill
from ...kernels.moe import moe_prefill_schedule as _geo

HIDDEN = 6144
DECODE_MAX_ROWS = 64
PREFILL_MIN_ROWS = DECODE_MAX_ROWS + 1
PREFILL_MAX_ROWS = 8192


@dataclass(frozen=True)
class CompiledZero:
    """A compiled clear. Decode: pinned to `rows`. Prefill: `rows` None (65..8192), at most
    `max_ctas` CTAs (eight waves of the device). `launch(out)` is the kernel call."""

    kind: str
    rows: int | None
    max_ctas: int
    launch: Callable[[torch.Tensor], None]


def build_decode(rows: int, device) -> CompiledZero:
    """Compile the exact-row clear for `rows` rows (1..64), no PDL role. Call at load.
    Compiles only (cached per process by the kernel module)."""
    out = torch.zeros(int(rows), HIDDEN, dtype=torch.bfloat16, device=device)
    with torch.cuda.device(device):
        compiled = _decode.compile_zero_rows(out)

    def launch(o):
        compiled(_decode._as_tensor(o), cutlass_torch.current_stream())

    return CompiledZero("decode", int(rows), 0, launch)


def build_prefill(device) -> CompiledZero:
    """Compile the runtime-size byte clear (PDL as the engine sets it) and clear 65 rows
    once. Call at load. Raises if the compile fails: the target then calls `zero_()`."""
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    cap = _geo.memory_cta_cap(num_sms)
    with torch.cuda.device(device):
        h = _prefill.compile_zero(use_pdl=_USE_PDL)

        def launch(o):
            h(o.view(torch.uint8).view(-1), o.numel() * 2, cap)

        launch(torch.zeros(PREFILL_MIN_ROWS, HIDDEN, dtype=torch.bfloat16, device=device))
    return CompiledZero("prefill", None, cap, launch)


class CuteDslMoeZero(OpWrapper):
    """`out.zero_()` for the routed output: bf16 `[rows, 6144]`, contiguous. Returns `out`."""

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="a 4-row decode width, one CTA per row", spec=dict(rows=4)),
        Cell(why="the full decode batch", spec=dict(rows=64)),
        Cell(why="65 rows, the first prefill count of the byte clear", spec=dict(rows=65)),
        Cell(why="8192 rows: the byte clear capped at eight CTA waves", spec=dict(rows=8192)),
    )

    note = """
    The trunk clears its routed output only when the shared expert did not write it
    first: at a padded width. Decode widths there are powers of two up to 64 (the
    target builds 1, 2, ..., 64); context steps take the prefill clear, or `zero_()` at
    the call site when it did not build. The MTP layer clears through the fused top-k's
    zero plane at decode widths and `zero_()` on context steps.

    The decode clear runs with no PDL role; the prefill one waits for its predecessor and
    releases its dependent after its stores.
    """

    def __call__(self, out: torch.Tensor, compiled: CompiledZero) -> torch.Tensor:
        compiled.launch(out)
        return out

    def reference(self, out: torch.Tensor, compiled: CompiledZero) -> torch.Tensor:
        """Zeros shaped like `out`; `out` is not written, `compiled` not read."""
        return torch.zeros_like(out)

    def is_valid(self, out: torch.Tensor, compiled: CompiledZero) -> None:
        # The decode handle clears exactly its compiled rows of a dynamic-layout view; a
        # strided view would be cleared as dense rows of someone else's memory.
        rows = int(out.shape[0])
        assert out.dim() == 2 and out.shape[1] == HIDDEN and out.dtype == torch.bfloat16
        assert out.is_contiguous(), "out must be contiguous"
        assert out.data_ptr() % 16 == 0, "out must be 16-byte aligned (128-bit stores)"
        if compiled.kind == "decode":
            assert rows == compiled.rows, f"compiled for {compiled.rows} rows, got {rows}"
        else:
            assert PREFILL_MIN_ROWS <= rows <= PREFILL_MAX_ROWS, (
                f"the prefill clear serves {PREFILL_MIN_ROWS}..{PREFILL_MAX_ROWS} rows, got {rows}"
            )


cute_dsl_moe_zero = CuteDslMoeZero()

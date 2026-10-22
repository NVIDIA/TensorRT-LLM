# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Add the shared expert's bf16 output into the routed experts', in place, in CuTe DSL."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.kernels.blackwell.utils import TRTLLM_ENABLE_PDL as _USE_PDL

from ...kernels.moe import moe_prefill_memory as _kernel
from ...kernels.moe import moe_prefill_schedule as _geo

HIDDEN = 6144
PREFILL_MIN_ROWS = 65
PREFILL_MAX_ROWS = 8192


@dataclass(frozen=True)
class CompiledMerge:
    """The compiled add, runtime-size, at most `max_ctas` CTAs (eight waves of the
    device). `launch(routed, shared)` is the kernel call."""

    max_ctas: int
    launch: Callable[[torch.Tensor, torch.Tensor], None]


def build(device) -> CompiledMerge:
    """Compile the add (PDL as the engine sets it) and run it once on 65 zero rows. Call
    at load. Raises if the compile fails: the target then calls `torch.add`."""
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    cap = _geo.memory_cta_cap(num_sms)
    with torch.cuda.device(device):
        h = _kernel.compile_merge(use_pdl=_USE_PDL)

        def launch(routed, shared):
            r = routed.view(-1)
            h(shared.view(-1), r, r, r.numel(), cap)

        bf = {"dtype": torch.bfloat16, "device": device}
        launch(
            torch.zeros(PREFILL_MIN_ROWS, HIDDEN, **bf),
            torch.zeros(PREFILL_MIN_ROWS, HIDDEN, **bf),
        )
    return CompiledMerge(cap, launch)


class CuteDslMoeMerge(OpWrapper):
    """`routed += shared` in bf16 with fp32 arithmetic, written into `routed`. Returns
    `routed`. Both bf16 `[rows, 6144]`, contiguous, 16-byte aligned, not overlapping."""

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="65 rows, the first context count the add serves", spec=dict(rows=65)),
        Cell(why="8192 rows: the add capped at eight CTA waves", spec=dict(rows=8192)),
    )

    note = """
    The trunk adds the shared expert separately only when it could not write into the
    routed output first: a padded row count. At a context row count it takes this op
    when it built, else `torch.add` at the call site; a padded decode width always takes
    `torch.add`. The call site joins the shared expert's stream before the call: the
    kernel's PDL wait covers its same-stream predecessor only.
    """

    def __call__(
        self, routed: torch.Tensor, shared: torch.Tensor, compiled: CompiledMerge
    ) -> torch.Tensor:
        compiled.launch(routed, shared)
        return routed

    def reference(
        self, routed: torch.Tensor, shared: torch.Tensor, compiled: CompiledMerge
    ) -> torch.Tensor:
        """The fp32 sum rounded once to bf16; fresh, `routed` not written."""
        return (routed.float() + shared.float()).to(torch.bfloat16)

    def is_valid(self, routed: torch.Tensor, shared: torch.Tensor, compiled: CompiledMerge) -> None:
        # The kernel adds flat element streams: a strided view, or two tensors of
        # different shapes with the same element count, add the wrong elements.
        rows = int(routed.shape[0])
        assert PREFILL_MIN_ROWS <= rows <= PREFILL_MAX_ROWS, (
            f"the add serves {PREFILL_MIN_ROWS}..{PREFILL_MAX_ROWS} rows, got {rows}"
        )
        for name, t in (("routed", routed), ("shared", shared)):
            assert tuple(t.shape) == (rows, HIDDEN) and t.dtype == torch.bfloat16, (
                f"{name} must be bf16 [{rows}, {HIDDEN}], got {t.dtype} {tuple(t.shape)}"
            )
            assert t.is_contiguous(), f"{name} must be contiguous"
            assert t.data_ptr() % 16 == 0, f"{name} must be 16-byte aligned"
        lo, hi = shared.data_ptr(), shared.data_ptr() + shared.numel() * 2
        r_lo, r_hi = routed.data_ptr(), routed.data_ptr() + routed.numel() * 2
        assert (lo, hi) == (r_lo, r_hi) or hi <= r_lo or r_hi <= lo, (
            "shared and routed may alias exactly but not partially overlap"
        )


cute_dsl_moe_merge = CuteDslMoeMerge()

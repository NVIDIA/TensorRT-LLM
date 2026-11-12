# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""L2 prefetch of byte ranges (CuTe DSL ``cp.async.bulk.prefetch.L2``): the next KDA layer's
projection weights, pulled into L2 on a side stream during a verify step."""

from __future__ import annotations

from typing import Sequence

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_kda import k3_l2_prefetch as _kernel

DEFAULT_MAX_CTAS = _kernel.DEFAULT_MAX_CTAS


class K3L2Prefetch(OpWrapper):
    """Issue L2 prefetches for ``ranges`` (``(address, bytes)``, 16-byte aligned, at most eight;
    empty ones dropped) on the current stream. Returns None and writes nothing."""

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why=(
                "helix8 (12 heads), up to 16 verify rows: the next KDA layer's QKVG and o_proj "
                "FP8 weights and scales, [f_a | b] and f_b"
            ),
            spec=dict(
                nbytes=(
                    4 * 12 * 128 * 7168,  # qkvg_proj weight, fp8
                    4 * 12 * 7168 // 128 * 4,  # its 128 x 128 block scales, fp32
                    7168 * 12 * 128,  # o_proj weight, fp8
                    7168 // 128 * 12 * 4,  # its block scales, fp32
                    144 * 7168 * 2,  # [f_a | b], bf16, 128 + 12 rows padded to 144
                    12 * 128 * 128 * 2,  # f_b, bf16
                )
            ),
        ),
    )
    note = """
    The only effect is on the L2 contents: no memory is written and nothing waits for the
    prefetches, so the reference is a no-op and the test proves the ranges' bytes and their
    neighbours unchanged. Kimi K3's KDA verify (the target's kda_l2_prefetch, the helix8 target, up to
    16 rows) launches it on the projection aux stream behind an event recorded after this
    layer's QKVG and [f_a | b] kernels, without PDL; it triggers its programmatic dependents at
    once. CTAs: one warp each, about one 8 KiB chunk per thread, at most max_ctas. Compiled
    kernels are cached per (range count, CTAs). Importing the entry needs CuTe DSL.
    """

    def __call__(self, ranges: Sequence[tuple[int, int]], max_ctas: int = DEFAULT_MAX_CTAS) -> None:
        _kernel.run_k3_l2_prefetch(ranges, max_ctas=max_ctas)

    def reference(
        self, ranges: Sequence[tuple[int, int]], max_ctas: int = DEFAULT_MAX_CTAS
    ) -> None:
        """Nothing: prefetches have no result."""
        return None


k3_l2_prefetch = K3L2Prefetch()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""L2 prefetch of weight byte ranges on sm_107: a hint that computes nothing.

``cp.async.bulk.prefetch.L2`` over up to 8 ``(address, bytes)`` ranges in 8 KiB chunks, one warp
per CTA (``kernels/gemm/rubin_l2_prefetch.py``). The decode issues it on a side stream to pull the
next KDA layer's projection weights into L2 while the recurrence and the all-reduce leave HBM
idle.
"""

from __future__ import annotations

import functools
from typing import Sequence

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The kernel's default CTA cap (``DEFAULT_MAX_CTAS``); the prefetch rate stops growing near it.
MAX_CTAS = 64


@functools.cache
def _kernel():
    """The kernel module; imported on first use (it needs the GPU stack)."""
    from ...kernels.gemm import rubin_l2_prefetch as kernel

    return kernel


def _sizes(sizes: tuple[int, ...], why: str, max_ctas: int = MAX_CTAS) -> Cell:
    return Cell(why=why, spec=dict(sizes=sizes, max_ctas=max_ctas))


# One TP8 KDA layer (12 heads x 128 per rank, hidden 7168): E4M3 values and R128c4 scale bytes.
_TP8_KDA = (
    6144 * 7168,  # q/k/v/g
    6144 * 224,
    128 * 7168,  # f_a
    128 * 224,
    12 * 7168,  # b
    128 * 224,
    7168 * 1536,  # o_proj
    7168 * 48,
)


class RubinL2Prefetch(OpWrapper):
    """Prefetch ``ranges`` into L2 on the current stream; returns ``None``.

    ``ranges`` are 16-byte aligned ``(address, bytes)`` pairs, at most 8 that are not empty (empty
    ones are dropped). ``max_ctas`` caps the CTAs, one warp each. The kernel only reads: no byte
    any later kernel reads changes.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        _sizes(_TP8_KDA, "the next TP8 KDA layer's weights and scales: eight ranges, the most"),
        _sizes(
            (49152 * 7168, 49152 * 224, 7168 * 12288),
            "a DEP16 KDA layer, all 96 heads: 450 MB, many chunks per thread",
        ),
        _sizes((4096, 3 * 8192 + 48), "under one 8 KiB chunk, and a range ending mid-chunk"),
        _sizes((0, 65536), "an empty range beside a full one (dropped before the launch)"),
        _sizes((1 << 20,), "one CTA: a single warp walks all 128 chunks", max_ctas=1),
    )

    note = """
    A prefetch is a hint: nothing waits for it and no result depends on it, so the reference is
    "no change". The test checks that every byte of the ranges is unchanged, eagerly and in a
    replayed CUDA graph. Launch it on a side stream behind an event recorded after the last
    kernel that streams weights from HBM. The first call per (range count, CTA count) compiles.
    """

    def __call__(self, ranges: Sequence[tuple[int, int]], max_ctas: int = MAX_CTAS) -> None:
        _kernel().run_k3_l2_prefetch(ranges, max_ctas=max_ctas)

    def reference(self, ranges: Sequence[tuple[int, int]], max_ctas: int = MAX_CTAS) -> None:
        """Nothing: the prefetch leaves every byte as it was; it only warms L2."""
        return None

    def is_valid(self, ranges: Sequence[tuple[int, int]], max_ctas: int = MAX_CTAS) -> None:
        """Nothing to add: the launcher drops 0-byte ranges, then raises on more than 8 ranges and
        on unaligned ones, and no range can change a value."""
        return None


rubin_l2_prefetch = RubinL2Prefetch()

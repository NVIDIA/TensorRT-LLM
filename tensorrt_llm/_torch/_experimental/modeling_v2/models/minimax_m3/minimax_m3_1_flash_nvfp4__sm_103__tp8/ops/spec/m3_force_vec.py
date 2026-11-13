# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-request forced accepted counts of a fractional forced acceptance, drawn as the in-tree worker draws them."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ..m3_force_vec import force_vec_kernel


class M3ForceVec(OpWrapper):
    """Bump ``counter`` (int64 [1]) by one, then for each of the ``batch_size`` requests ``b`` write
    ``min(base_total + (pool[(c * counter_stride + b * slot_stride) & pool_mask] < frac), max_total)`` into ``out``
    (int32), ``c`` the bumped counter: the in-tree ``_apply_force_accepted_tokens`` draw over its fixed-seed fp32
    ``pool``. ``block`` is the launch width (a power of two >= batch_size; 16, 32 or 64 are compiled at warmup).

    Only the forced-acceptance testing knob (``TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`` with a fractional part)
    runs it.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="one request, fraction 0.5", spec=dict(batch=1, block=16, frac=0.5)),
        Cell(why="eight requests, fraction 0.3", spec=dict(batch=8, block=16, frac=0.3)),
    )

    note = """
    Integer arithmetic and one fp32 comparison: `reference` must match exactly.
    """

    def __call__(
        self,
        pool,
        counter,
        out,
        batch_size,
        base_total,
        max_total,
        frac,
        counter_stride,
        slot_stride,
        pool_mask,
        block,
    ):
        force_vec_kernel[(1,)](
            pool,
            counter,
            out,
            batch_size,
            base_total,
            max_total,
            frac,
            counter_stride,
            slot_stride,
            pool_mask,
            BLOCK=block,
        )
        return out

    def reference(
        self,
        pool,
        counter,
        out,
        batch_size,
        base_total,
        max_total,
        frac,
        counter_stride,
        slot_stride,
        pool_mask,
        block,
    ):
        """The counts (a new tensor; ``counter`` and ``out`` are not written)."""
        c = int(counter.item()) + 1
        slots = torch.arange(batch_size, dtype=torch.int64, device=pool.device)
        idx = (c * counter_stride + slots * slot_stride) & pool_mask
        r = pool[idx]
        return torch.clamp(base_total + (r < frac).to(torch.int32), max=max_total).to(torch.int32)


m3_force_vec = M3ForceVec()

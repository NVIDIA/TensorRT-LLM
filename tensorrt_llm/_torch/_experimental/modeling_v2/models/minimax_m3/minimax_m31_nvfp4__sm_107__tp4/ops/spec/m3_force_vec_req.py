# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-request forced accepted counts of a fractional forced acceptance, each request drawn at its own key."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.m3_force_vec import force_vec_req_kernel


class M3ForceVecReq(OpWrapper):
    """For each of the ``batch_size`` requests ``b`` write ``min(base_total + (pool[(keys[b] * key_stride) &
    pool_mask] < frac), max_total)`` into ``out`` (int32): the forced-acceptance draw from the fixed-seed fp32 ``pool``
    at the request's own key (int32 ``keys``, its KV length at this verify), so a request's draws depend on neither the
    server's history nor the other requests. ``block`` is the launch width (a power of two >= batch_size; 16, 32 or
    64 are compiled at warmup).

    Only the forced-acceptance testing knob (``TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`` with a fractional part)
    runs it, with ``FORCE_AL_PER_REQUEST``.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(why="one request, fraction 0.5", spec=dict(batch=1, block=16, frac=0.5)),
        Cell(why="eight requests, fraction 0.3", spec=dict(batch=8, block=16, frac=0.3)),
    )

    note = """
    Integer arithmetic and one fp32 comparison: `reference` must match exactly.
    """

    def __call__(
        self, pool, keys, out, batch_size, base_total, max_total, frac, key_stride, pool_mask, block
    ):
        force_vec_req_kernel[(1,)](
            pool,
            keys,
            out,
            batch_size,
            base_total,
            max_total,
            frac,
            key_stride,
            pool_mask,
            BLOCK=block,
        )
        return out

    def reference(
        self, pool, keys, out, batch_size, base_total, max_total, frac, key_stride, pool_mask, block
    ):
        """The counts (a new tensor; ``out`` is not written)."""
        idx = (keys[:batch_size].to(torch.int64) * key_stride) & pool_mask
        r = pool[idx]
        return torch.clamp(base_total + (r < frac).to(torch.int32), max=max_total).to(torch.int32)


m3_force_vec_req = M3ForceVecReq()

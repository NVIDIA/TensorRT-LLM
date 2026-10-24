# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-request forced accepted counts of a fractional forced acceptance (the in-tree speculative worker's
``force_num_accepted_tokens``), drawn exactly as its ``_apply_force_accepted_tokens`` does, in one launch.

Triton: one CTA bumps the shared step counter and draws each request's count from the fixed-seed pool with the
in-tree (counter, slot) hash. Reached through the ``ops/spec`` entry ``m3_force_vec``.
"""

import triton
import triton.language as tl


@triton.jit
def force_vec_kernel(
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
    BLOCK: tl.constexpr,
):
    c = tl.load(counter) + 1
    slots = tl.arange(0, BLOCK)
    tl.store(counter, c)
    idx = (c * counter_stride + slots.to(tl.int64) * slot_stride) & pool_mask
    live = slots < batch_size
    r = tl.load(pool + idx, mask=live, other=1.0)
    v = tl.minimum(base_total + (r < frac).to(tl.int32), max_total)
    tl.store(out + slots, v, mask=live)

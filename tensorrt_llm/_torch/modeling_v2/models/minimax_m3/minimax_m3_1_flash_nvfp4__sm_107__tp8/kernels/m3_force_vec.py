# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-request forced accepted counts of a fractional forced acceptance (the in-tree speculative worker's
``force_num_accepted_tokens``), in one launch. Reached through the ``ops/spec`` entries ``m3_force_vec`` and
``m3_force_vec_req``.

Triton. ``force_vec_kernel``: one CTA bumps the shared step counter and draws each request's count from the fixed-seed
pool with the in-tree (counter, slot) hash, exactly as ``_apply_force_accepted_tokens`` does. ``force_vec_req_kernel``:
one draw per request from its own key (its KV length at this verify), no shared counter.
"""

import triton
import triton.language as tl


# One compiled variant for every batch and accept count: specializing them (== 1, divisible by 16) would compile
# mid-serving the first time a step reaches a new one, past the worker's warm-up.
@triton.jit(do_not_specialize=["batch_size", "base_total", "max_total"])
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


@triton.jit(do_not_specialize=["batch_size", "base_total", "max_total"])
def force_vec_req_kernel(
    pool,
    keys,
    out,
    batch_size,
    base_total,
    max_total,
    frac,
    key_stride,
    pool_mask,
    BLOCK: tl.constexpr,
):
    # One draw per request from its own key (its KV length at this verify): no shared counter, so a request's forced
    # counts depend only on its own prompt and progress, not on what else the server ran before or beside it.
    slots = tl.arange(0, BLOCK)
    live = slots < batch_size
    k = tl.load(keys + slots, mask=live, other=0).to(tl.int64)
    idx = (k * key_stride) & pool_mask
    r = tl.load(pool + idx, mask=live, other=1.0)
    v = tl.minimum(base_total + (r < frac).to(tl.int32), max_total)
    tl.store(out + slots, v, mask=live)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The drafter fc fork's release of the first draft input by a device counter instead of a stream join (modeling's
``DI_FC_FLAG``).

Triton. ``fc_wait_kernel``: one thread, queued on the main stream after the acceptance. It lets the next kernel (the
first draft input) launch, waits for the acceptance before it, then spins until the fork's counter reaches the number
of forks waited for. Ordering only: no value changes.
"""

import torch
import triton
import triton.language as tl

# Spins of fc_wait_kernel before it gives up on the fc fork (~ms): a broken fork / wait pairing then counts an overrun
# in counts[2] instead of hanging the step (the draft input may read a stale fc output: draft tokens only, the verify
# stays exact).
MAX_SPINS = 1 << 22  # capacity: fixed: a spin bound, not a size


@triton.jit
def fc_wait_kernel(counts, MAX_SPINS: tl.constexpr):
    # counts[0]: fc forks finished (bumped on the aux stream after each fc); counts[1]: forks waited for (here).
    # Lets the next kernel (the first draft input) launch now; waits for the acceptance before it, then for the fork.
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    target = tl.load(counts + 1) + 1
    done = tl.load(counts, volatile=True)
    spins = 0
    while (done < target) & (spins < MAX_SPINS):
        done = tl.load(counts, volatile=True)
        spins += 1
    tl.store(counts + 1, target)
    if done < target:
        tl.atomic_add(counts + 2, 1)


def fc_wait(counts: torch.Tensor) -> None:
    """Queue the wait on the current stream for the fork ``counts`` (int64 [3]) last counted."""
    fc_wait_kernel[(1,)](counts, MAX_SPINS=MAX_SPINS, num_warps=1, launch_pdl=True)

# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Device step validation for continuous TopK."""

import cutlass
import cutlass.cute as cute
from cutlass.utils.smem_allocator import SmemAllocator

_REFRESH_THREADS = 512
_REFRESH_ROWS_PER_CTA = 2048


class TopKCapacityRefreshKernel:
    """Validate continuous prefixes and publish safe per-row bounds.

    A single capturable launch checks extents, prefixes and capacity, then
    publishes sanitized bounds. Invalid inputs produce zero active rows and
    owner -1; error bits 1, 2 and 4 identify extent, prefix and capacity errors.
    """

    def __init__(self, max_context, max_rows, max_batch, ratio, num_keys):
        self.max_context = max_context
        self.max_rows = max_rows
        self.max_batch = max_batch
        self.ratio = ratio
        self.num_keys = num_keys
        self.publish_ctas = max(
            1, (max_rows + _REFRESH_ROWS_PER_CTA - 1) // _REFRESH_ROWS_PER_CTA
        )

    @cute.jit
    def _block_or(self, value, shared, tidx):
        """OR one Int32 across the CTA; ``shared`` is a one-word scratch."""
        cute.arch.sync_threads()
        if tidx == 0:
            shared[0] = 0
        cute.arch.sync_threads()
        if value != 0:
            cute.arch.atomic_or(shared.iterator, value, sem="relaxed", scope="cta")
        cute.arch.sync_threads()
        return shared[0]

    @cute.jit
    def _total_capacity_code(self, kv, batches, tidx):
        return cutlass.Int32(0)

    @cute.jit
    def _prefix_code(self, qo, kv, step, shared, tidx, next_n):
        """Extent bit 1, or the OR of prefix bit 2 and capacity bit 4."""
        supplied_rows = step[0]
        batches = step[1]
        code = cutlass.Int32(0)
        if (
            (supplied_rows < 0)
            | (supplied_rows > self.max_rows)
            | (batches < 0)
            | (batches > self.max_batch)
        ):
            code = cutlass.Int32(1)
        local = cutlass.Int32(0)
        if code == 0:
            if tidx == 0:
                if (qo[0] != 0) | (kv[0] != 0):
                    local = local | 2
            request = tidx
            while request < batches:
                qb, qe = (cutlass.Int64(qo[request]), cutlass.Int64(qo[request + 1]))
                kb, ke = (cutlass.Int64(kv[request]), cutlass.Int64(kv[request + 1]))
                if (qb < 0) | (qe < qb) | (kb < 0) | (ke < kb) | (ke - kb < qe - qb):
                    local = local | 2
                if (next_n != 0) & (qe - qb != next_n):
                    local = local | 2
                if (qe > supplied_rows) | (ke - kb > self.max_context):
                    local = local | 4
                request = request + _REFRESH_THREADS
            local = local | self._total_capacity_code(kv, batches, tidx)
        return code | self._block_or(local, shared, tidx)

    @cute.jit
    def _block_inclusive_sum(self, value, warp_totals, tidx):
        """Inclusive CTA sum of one Int32 per thread, plus the CTA total."""
        lane = tidx % 32
        warp = tidx // 32
        inclusive = value
        for step in cutlass.range_constexpr(5):
            prior = cute.arch.shuffle_sync_up(
                inclusive, 1 << step, mask=4294967295, mask_and_clamp=0
            )
            if lane >= 1 << step:
                inclusive = inclusive + prior
        cute.arch.sync_threads()
        if lane == 31:
            warp_totals[warp] = inclusive
        cute.arch.sync_threads()
        before = cutlass.Int32(0)
        total = cutlass.Int32(0)
        for other in cutlass.range_constexpr(_REFRESH_THREADS // 32):
            if other < warp:
                before = before + warp_totals[other]
            total = total + warp_totals[other]
        return (before + inclusive, total)

    @cute.kernel
    def publish_kernel(
        self,
        qo: cute.Tensor,
        kv: cute.Tensor,
        step: cute.Tensor,
        safe_qo: cute.Tensor,
        physical_kv: cute.Tensor,
        active: cute.Tensor,
        row_ends: cute.Tensor,
        row_requests: cute.Tensor,
        error: cute.Tensor,
        next_n: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        smem = SmemAllocator()
        shared = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((1,), order=(0,)),
            byte_alignment=16,
        )
        warp_totals = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((_REFRESH_THREADS // 32,), order=(0,)),
            byte_alignment=16,
        )
        code = self._prefix_code(qo, kv, step, shared, tidx, next_n)
        batches = step[1]
        active_rows = cutlass.Int32(0)
        active_batches = cutlass.Int32(0)
        if code == 0:
            active_rows = qo[batches]
            active_batches = batches
        if cta == 0:
            if tidx == 0:
                error[0] = code
                active[0] = active_rows
                active[1] = active_batches
                safe_qo[0] = cutlass.Int32(0)
                physical_kv[0] = cutlass.Int32(0)
            carry = cutlass.Int32(0)
            base = cutlass.Int32(0)
            while base < self.max_batch:
                request = base + tidx
                physical = cutlass.Int32(0)
                query_end = active_rows
                if request < active_batches:
                    query_end = qo[request + 1]
                    physical = cutlass.Int32(
                        (cutlass.Int64(kv[request + 1]) - cutlass.Int64(kv[request]))
                        // self.ratio
                    )
                inclusive, total = self._block_inclusive_sum(
                    physical, warp_totals, tidx
                )
                if request < self.max_batch:
                    safe_qo[request + 1] = query_end
                    physical_kv[request + 1] = carry + inclusive
                carry = carry + total
                base = base + _REFRESH_THREADS
        row = cta * _REFRESH_ROWS_PER_CTA + tidx
        row_limit = min((cta + 1) * _REFRESH_ROWS_PER_CTA, self.max_rows)
        while row < row_limit:
            row_end = cutlass.Int32(0)
            owner = cutlass.Int32(-1)
            if row < active_rows:
                low = cutlass.Int32(0)
                high = active_batches - 1
                while low < high:
                    middle = (low + high + 1) // 2
                    if qo[middle] <= row:
                        low = middle
                    else:
                        high = middle - 1
                owner = low
                q_length = qo[owner + 1] - qo[owner]
                kv_length = kv[owner + 1] - kv[owner]
                row_end = cutlass.Int32(
                    (cutlass.Int64(kv_length) - q_length + row - qo[owner] + 1)
                    // self.ratio
                )
            row_ends[row] = row_end
            row_requests[row] = owner
            row = row + _REFRESH_THREADS

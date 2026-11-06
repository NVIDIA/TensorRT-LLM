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
"""Device step validation for continuous selective TopK."""

import cutlass
import cutlass.cute as cute

from .topk_metadata import (
    TopKCapacityRefreshKernel,
    _REFRESH_THREADS,
)


class SelectiveContinuousCapacityRefreshKernel(TopKCapacityRefreshKernel):
    """Validate ragged continuous prefixes and publish bounds in one launch.

    The key capacity bounds the sum of pooled request lengths.
    """

    @cute.jit
    def __call__(
        self,
        qo,
        kv,
        step,
        safe_qo,
        physical_kv,
        active,
        row_ends,
        row_requests,
        error,
        next_n: cutlass.Int32,
        stream,
    ):
        self.publish_kernel(
            qo,
            kv,
            step,
            safe_qo,
            physical_kv,
            active,
            row_ends,
            row_requests,
            error,
            next_n,
        ).launch(
            grid=(self.publish_ctas, 1, 1),
            block=(_REFRESH_THREADS, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _total_capacity_code(self, kv, batches, tidx):
        code = cutlass.Int32(0)
        if tidx == 0:
            total = cutlass.Int64(0)
            for request in cutlass.range(batches):
                total += (
                    cutlass.Int64(kv[request + 1]) - cutlass.Int64(kv[request])
                ) // self.ratio
            if total > self.num_keys:
                code = cutlass.Int32(4)
        return code

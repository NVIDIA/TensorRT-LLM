# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Graph-safe unit-scale E4M3 query staging for CSA2's native PrimTS operands."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.compiler import CompiledKernel

__all__ = ["UnitScaleQuery"]

_BLOCK = 1024


# The row count is a run-time argument so every row count shares one compiled
# kernel; do_not_specialize also drops Triton's ==1 and %16 variants.
@triton.autotune(configs=[triton.Config({"BLOCK_SIZE": _BLOCK}, num_warps=4)], key=[])
@triton.jit(do_not_specialize=["rows"])
def _quantize_query(
    q_ptr, q8_pairs, rows, QUERY_WIDTH: tl.constexpr, BLOCK_SIZE: tl.constexpr
) -> None:
    # Q8 = RN-satfinite(Q): each pair takes the PTX conversion of the native
    # query converter, so out-of-range values saturate to +-448.
    q_values = rows.to(tl.int64) * QUERY_WIDTH
    columns = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    pairs = tl.arange(0, BLOCK_SIZE // 2).to(tl.int64)
    program = tl.program_id(0).to(tl.int64)
    for tile in range(program, tl.cdiv(q_values, BLOCK_SIZE), tl.num_programs(0).to(tl.int64)):
        values = tile * BLOCK_SIZE + columns
        q = tl.load(q_ptr + values, values < q_values, other=0).to(tl.float32)
        low, high = tl.split(tl.reshape(q, (BLOCK_SIZE // 2, 2)))
        packed = tl.inline_asm_elementwise(
            "cvt.rn.satfinite.e4m3x2.f32 $0, $2, $1;",
            "=h,r,r",
            [low, high],
            dtype=tl.int16,
            is_pure=True,
            pack=1,
        )
        pair = tile * (BLOCK_SIZE // 2) + pairs
        tl.store(q8_pairs + pair, packed, pair < q_values // 2)


@dataclass(frozen=True)
class UnitScaleQuery:
    """Precompiled BF16 to E4M3 query staging at unit scale.

    Q and the E4M3 destination remain live run-time arguments. The attention
    kernel then reads Q8 with a Q descale of one, so the caller's host softmax
    scale is the effective one.
    """

    kernel: CompiledKernel
    rows: int
    blocks: int

    @classmethod
    def plan(
        cls, device: torch.device, rows: int, heads: int, q8: torch.Tensor
    ) -> "UnitScaleQuery":
        """Precompile the converter, without reading inputs.

        The kernel depends on the head count, not on ``rows``, so plans for other
        row counts reuse the compiled kernel.
        """
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        blocks = max(1, min(8 * sm_count, triton.cdiv(rows * heads * 512, _BLOCK)))
        with torch.cuda.device(device):
            kernel = _quantize_query.warmup(
                q8.view(torch.bfloat16),
                q8.view(torch.int16),
                rows,
                QUERY_WIDTH=heads * 512,
                grid=(blocks,),
            )[0]
        return cls(kernel, rows, blocks)

    def run(self, q: torch.Tensor, q8: torch.Tensor) -> None:
        """Write RN-satfinite E4M3 Q into ``q8`` without allocation or host reads."""
        with torch.cuda.device(q.device):
            self.kernel[(self.blocks, 1, 1)](q, q8.view(torch.int16), self.rows)

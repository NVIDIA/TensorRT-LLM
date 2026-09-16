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

"""Exact native BF16 top-K when the row contains at most K+3 elements."""

from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass.cute import runtime

from . import gvr_topk_decode_self_sampling_bf16 as primitives

__all__ = ["get_compiled"]


@cute.jit
def _composite_key(raw, index):
    """Return ordered BF16 value bits with a unique low-16 index suffix."""
    bits = raw & cutlass.Uint32(0xFFFF)
    sign = bits >> cutlass.Uint32(15)
    ordered = (bits ^ ((cutlass.Uint32(0) - sign) | cutlass.Uint32(0x8000))) & cutlass.Uint32(
        0xFFFF
    )
    # torch.topk ranks NaNs above all numerical values, including +infinity.
    if (bits & cutlass.Uint32(0x7F80)) == cutlass.Uint32(0x7F80):
        if (bits & cutlass.Uint32(0x007F)) != cutlass.Uint32(0):
            ordered = cutlass.Uint32(0xFFFF)
    return (ordered << cutlass.Uint32(16)) | cutlass.Uint32(index)


class ComplementKernel:
    """Find up to three excluded indices, then emit their exact complement."""

    def __init__(self, k: int, envelope: int, next_n: int, cr_shift: int) -> None:
        self.k = k
        self.envelope = envelope
        self.next_n = next_n
        self.cr_shift = cr_shift
        self.excluded_max = envelope - k
        self.vector_batches = ((envelope // 4) + 255) // 256
        self.has_tail = envelope % 4 != 0
        self.slots = 4 * self.vector_batches + int(self.has_tail)

    @cute.kernel
    def kernel(
        self,
        logits: cute.Tensor,
        kv_lens: cute.Tensor,
        output: cute.Tensor,
        envelope: cutlass.Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane = tid & cutlass.Int32(31)
        warp = tid >> cutlass.Int32(5)
        request = row // cutlass.Int32(self.next_n)
        position = row % cutlass.Int32(self.next_n)
        valid = (
            kv_lens[request] - cutlass.Int32(self.next_n) + position + cutlass.Int32(1)
        ) >> cutlass.Int32(self.cr_shift)
        if valid < cutlass.Int32(0):
            valid = cutlass.Int32(0)
        if valid > envelope:
            valid = envelope
        if valid <= cutlass.Int32(self.k):
            for block in cutlass.range_constexpr((self.k + 255) // 256):
                index = tid + cutlass.Int32(block * 256)
                if index < cutlass.Int32(self.k):
                    selected = cutlass.Int32(-1)
                    if index < valid:
                        selected = index
                    output[row, index] = selected
        else:
            sentinel = cutlass.Uint32(0xFFFFFFFF)
            keys = cute.make_rmem_tensor((self.slots,), cutlass.Uint32)
            for slot in cutlass.range_constexpr(self.slots):
                keys[slot] = sentinel
            base = logits[row, None].iterator.toint()
            for batch in cutlass.range_constexpr(self.vector_batches):
                vector = tid + cutlass.Int32(batch * 256)
                if vector < cutlass.Int32(self.envelope // 4):
                    low, high = primitives._ld_g_nc_v2_b32(
                        base + cutlass.Int64(vector) * cutlass.Int64(8)
                    )
                    words = [low, low >> cutlass.Uint32(16), high, high >> cutlass.Uint32(16)]
                    for item in cutlass.range_constexpr(4):
                        index = vector * cutlass.Int32(4) + cutlass.Int32(item)
                        if index < valid:
                            keys[batch * 4 + item] = _composite_key(words[item], index)
            if cutlass.const_expr(self.has_tail):
                index = cutlass.Int32((self.envelope // 4) * 4) + tid
                if index < valid:
                    raw = primitives._ld_g_nc_u16(base + cutlass.Int64(index) * cutlass.Int64(2))
                    keys[self.slots - 1] = _composite_key(raw, index)
            shared = cute.arch.get_dyn_smem(cutlass.Uint32, alignment=16)
            warp_candidates = cute.make_tensor(shared, cute.make_layout((24,)))
            excluded = cute.make_tensor(shared + 24, cute.make_layout((3,)))
            for pick in cutlass.range_constexpr(self.excluded_max):
                local_minimum = sentinel
                for slot in cutlass.range_constexpr(self.slots):
                    if keys[slot] < local_minimum:
                        local_minimum = keys[slot]
                warp_minimum = primitives.warp_min_u32(local_minimum)
                if lane == cutlass.Int32(0):
                    warp_candidates[
                        warp * cutlass.Int32(self.excluded_max) + cutlass.Int32(pick)
                    ] = warp_minimum
                for slot in cutlass.range_constexpr(self.slots):
                    if keys[slot] == warp_minimum:
                        keys[slot] = sentinel
            cute.arch.barrier()
            if tid < cutlass.Int32(32):
                candidate = sentinel
                if tid < cutlass.Int32(8 * self.excluded_max):
                    candidate = cutlass.Uint32(warp_candidates[tid])
                for pick in cutlass.range_constexpr(self.excluded_max):
                    minimum = primitives.warp_min_u32(candidate)
                    if lane == cutlass.Int32(0):
                        excluded[cutlass.Int32(pick)] = minimum & cutlass.Uint32(0xFFFF)
                    if candidate == minimum:
                        candidate = sentinel
            cute.arch.barrier()
            exclude_count = valid - cutlass.Int32(self.k)
            for block in cutlass.range_constexpr((self.envelope + 255) // 256):
                index = tid + cutlass.Int32(block * 256)
                if index < valid:
                    before = cutlass.Int32(0)
                    omit = cutlass.Int32(0)
                    for pick in cutlass.range_constexpr(self.excluded_max):
                        if cutlass.Int32(pick) < exclude_count:
                            removed = cutlass.Int32(excluded[cutlass.Int32(pick)])
                            if removed < index:
                                before = before + cutlass.Int32(1)
                            if removed == index:
                                omit = cutlass.Int32(1)
                    if omit == cutlass.Int32(0):
                        output[row, index - before] = index

    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        kv_lens: cute.Tensor,
        output: cute.Tensor,
        envelope: cutlass.Int32,
        stream,
    ):
        self.kernel(logits, kv_lens, output, envelope).launch(
            grid=(logits.shape[0], 1, 1),
            block=(256, 1, 1),
            stream=stream,
            smem=128,
            min_blocks_per_mp=4,
        )


_CACHE: dict[tuple[int, int, int, int], Any] = {}


def get_compiled(k: int, envelope: int, next_n: int, cr_shift: int) -> Any:
    """Compile the capture-stable near-identity endpoint specialization."""
    key = (k, envelope, next_n, cr_shift)
    compiled = _CACHE.get(key)
    if compiled is None:
        kernel = ComplementKernel(k, envelope, next_n, cr_shift)
        logits = runtime.make_fake_compact_tensor(
            cutlass.BFloat16,
            (cute.sym_int(), cute.sym_int()),
            stride_order=(1, 0),
            assumed_align=16,
        )
        indices = runtime.make_fake_compact_tensor(
            cutlass.Int32, (cute.sym_int(), cute.sym_int()), stride_order=(1, 0), assumed_align=16
        )
        lengths = runtime.make_fake_compact_tensor(
            cutlass.Int32, (cute.sym_int(),), stride_order=(0,), assumed_align=4
        )
        stream = runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        with primitives._no_carveout():
            compiled = cute.compile(
                kernel,
                logits,
                indices,
                lengths,
                indices,
                cutlass.Int32(0),
                stream=stream,
                options="--enable-tvm-ffi",
            )
        _CACHE[key] = compiled
    return compiled

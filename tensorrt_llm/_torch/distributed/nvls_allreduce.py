# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NVLink SHARP (multicast) all-reduce of BF16 ``[M, H]`` messages.

The reduction runs in the NVSwitch. Each program owns ``ROWS`` rows; rank
``r`` reduces the ``r``-th contiguous slice of the program's 16-byte vectors
with ``multimem.ld_reduce`` on the multicast address (the switch reads the
vector from every rank and returns the FP32-accumulated sum rounded to BF16)
and writes the result to every rank with ``multimem.st``. Per rank and
direction this moves about ``(1 + 1/world_size)`` times the message over
NVLink, against about two times for a two-shot all-reduce.

Per program:

1. copy the input rows into this rank's input buffer (symmetric memory);
2. arrive on the program's flag (one relaxed ``multimem.red`` that adds 1 on
   every rank) and wait until every rank has arrived;
3. reduce this rank's slice in the switch and multicast it into the output
   buffer of every rank;
4. poll the program's rows in the local output buffer until none of the BF16
   lanes holds the sentinel ``0xFFFF`` (a NaN the reducer never stores), copy
   them out and re-arm them with the sentinel.

The arrival in step 2 is ordered after the input stores by a GPU-scope fence:
the input lives in this rank's memory, which the switch reads through this
GPU's L2, so stores that are visible GPU-wide are visible to the switch.
``TRTLLM_NVLS_AR_FENCE_SYS=1`` uses the system-scope fence of the PTX memory
model instead (slower).

Step 4 needs no fence: the output buffer is a Lamport buffer. A rank re-arms an
output row only after reading the single store each slice receives per call,
and the next store into that row follows the next call's arrival, which this
rank makes only after this call's kernel (re-arming included) has finished.
The input buffer is rewritten by the next call only after every slice of this
call has arrived, i.e. after every reducer has read it.

The barrier target comes from a per-program call counter kept on the rank, so
the kernel keeps no host state and replays inside CUDA graphs. Every rank
receives the same bits (each vector is reduced once), so the ranks stay
bitwise identical. The switch rounds the FP32 sum to BF16 faithfully (within
one BF16 ulp of the exact sum; it differs from the round-to-nearest result of
the Lamport two-shot kernel in about a quarter of the elements) and
deterministically.

With ``TRTLLM_NVLS_AR_V2=1``, messages of at least
``TRTLLM_NVLS_AR_V2_MIN_BYTES`` use a chunk-pipelined kernel instead
(``_nvls_allreduce_v2_kernel``). The steps are the same, but they run per
chunk (16 KiB by default) in three groups of persistent programs: stagers copy
chunks in and arrive on per-chunk flags, reducers reduce and multicast a chunk
as soon as every rank has staged it, and drainers copy complete chunks out of
the Lamport buffer. The input copy, the barrier, the NVLink traffic and the
output copy of different chunks overlap instead of running one after another.
Each vector is still reduced once by the switch, so all ranks receive the same
bits (in tests they also match the one-shot kernel bit for bit). The
per-chunk flags and call counters are separate from the one-shot kernel's, so
both kernels can share a workspace in any call order.
"""

import math
import os
from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._dlpack_utils import pack_strided_memory
from tensorrt_llm._torch.distributed.mnnvl_memory import _check_cu_result, cuda
from tensorrt_llm._utils import get_sm_version, mpi_comm
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..flashinfer_utils import get_env_enable_pdl

# Message size window (bytes of the BF16 input) that AllReduce routes to this
# kernel when TRTLLM_NVLS_AR=1.
NVLS_AR_MIN_BYTES = int(os.environ.get("TRTLLM_NVLS_AR_MIN_BYTES", 1 << 20))
NVLS_AR_MAX_BYTES = int(os.environ.get("TRTLLM_NVLS_AR_MAX_BYTES", 32 << 20))

# Pipelined kernel (TRTLLM_NVLS_AR_V2=1) knobs: name -> (environment variable,
# default). ``stagers``, ``reducers`` or ``drainers`` <= 0 means one program of
# that kind per SM.
_V2_KNOBS = {
    "min_bytes": ("TRTLLM_NVLS_AR_V2_MIN_BYTES", 4 << 20),
    "warps": ("TRTLLM_NVLS_AR_V2_WARPS", 8),
    "chunk_bytes": ("TRTLLM_NVLS_AR_V2_CHUNK_BYTES", 16384),
    "group": ("TRTLLM_NVLS_AR_V2_GROUP", 1),
    "stagers": ("TRTLLM_NVLS_AR_V2_STAGERS", 0),
    "reducers": ("TRTLLM_NVLS_AR_V2_REDUCERS", 0),
    "drainers": ("TRTLLM_NVLS_AR_V2_DRAINERS", 0),
    "probe": ("TRTLLM_NVLS_AR_V2_PROBE", 1),
}

# Kernel versions whose first call has been logged (True: pipelined).
_FIRST_CALL_LOGGED = set()

# 0xFFFF as int16: a BF16 NaN pattern the reducer never stores.
_SENTINEL_BITS = -1
_SENTINEL: tl.constexpr = tl.constexpr(_SENTINEL_BITS)


@triton.jit
def _arrive(flag_mc_addr, FENCE_SYS: tl.constexpr):
    """Thread 0: publish the CTA's staged input and add 1 to the flag on every rank."""
    if FENCE_SYS:
        return tl.inline_asm_elementwise(
            """
            {
            .reg .u32 %ltid;
            .reg .pred %p_skip;
            mov.u32 %ltid, %tid.x;
            mov.u32 $0, 0;
            setp.ne.u32 %p_skip, %ltid, 0;
            @%p_skip bra NVLS_ARRIVE_DONE;
            fence.acq_rel.sys;
            multimem.red.relaxed.sys.global.add.u32 [$1], 1;
            NVLS_ARRIVE_DONE:
            }
            """,
            "=r,l,~{memory}",
            args=[flag_mc_addr],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )
    else:
        return tl.inline_asm_elementwise(
            """
            {
            .reg .u32 %ltid;
            .reg .pred %p_skip;
            mov.u32 %ltid, %tid.x;
            mov.u32 $0, 0;
            setp.ne.u32 %p_skip, %ltid, 0;
            @%p_skip bra NVLS_ARRIVE_DONE;
            fence.acq_rel.gpu;
            multimem.red.relaxed.sys.global.add.u32 [$1], 1;
            NVLS_ARRIVE_DONE:
            }
            """,
            "=r,l,~{memory}",
            args=[flag_mc_addr],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )


@triton.jit
def _wait(flag_uc_addr, target):
    """Thread 0: spin until the local flag has reached ``target`` (mod 2^32)."""
    return tl.inline_asm_elementwise(
        """
        {
        .reg .u32 %ltid, %cur;
        .reg .s32 %diff;
        .reg .pred %p_skip, %p_wait;
        mov.u32 %ltid, %tid.x;
        mov.u32 $0, 0;
        setp.ne.u32 %p_skip, %ltid, 0;
        @%p_skip bra NVLS_WAIT_DONE;
        NVLS_WAIT_LOOP:
        ld.acquire.sys.global.u32 %cur, [$1];
        sub.s32 %diff, %cur, $2;
        setp.lt.s32 %p_wait, %diff, 0;
        @%p_wait bra NVLS_WAIT_LOOP;
        NVLS_WAIT_DONE:
        }
        """,
        "=r,l,r,~{memory}",
        args=[flag_uc_addr, target],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _multimem_ld_reduce_bf16x8(addrs, mask):
    """Switch-side sum (FP32 accumulate) of 8 BF16 values per address."""
    return tl.inline_asm_elementwise(
        """
        {
        .reg .pred %p_ld;
        setp.ne.s32 %p_ld, $5, 0;
        mov.b32 $0, 0;
        mov.b32 $1, 0;
        mov.b32 $2, 0;
        mov.b32 $3, 0;
        @%p_ld multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 {$0, $1, $2, $3}, [$4];
        }
        """,
        "=r,=r,=r,=r,l,r,~{memory}",
        args=[addrs, mask.to(tl.int32)],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def _multimem_st_b128(addrs, a, b, c, d, mask):
    """Store 16 bytes to the address on every rank of the multicast group."""
    return tl.inline_asm_elementwise(
        """
        {
        .reg .pred %p_st;
        setp.ne.s32 %p_st, $6, 0;
        mov.u32 $0, 0;
        @%p_st multimem.st.relaxed.sys.global.v4.f32 [$1], {$2, $3, $4, $5};
        }
        """,
        "=r,l,r,r,r,r,r,~{memory}",
        args=[addrs, a, b, c, d, mask.to(tl.int32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _no_sentinel(v):
    """Replace the 0xFFFF lanes of a BF16x2 word by 0x7FFF (both NaN)."""
    lo = v & 0xFFFF
    hi = v >> 16
    lo = tl.where(lo == 0xFFFF, 0x7FFF, lo)
    hi = tl.where(hi == 0xFFFF, 0x7FFF, hi)
    return lo | (hi << 16)


@triton.jit(do_not_specialize=["rank", "num_rows"])
def _nvls_allreduce_kernel(
    inp_ptr,
    out_ptr,
    in_buf_ptr,
    out_buf_ptr,
    in_mc_addr,
    out_mc_addr,
    flag_uc_addr,
    flag_mc_addr,
    epoch_ptr,
    rank,
    num_rows,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROWS: tl.constexpr,
    VEC_BLOCK: tl.constexpr,
    FLAG_SPACING: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    FENCE_SYS: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    pid = tl.program_id(0)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    epoch = tl.load(epoch_ptr + pid).to(tl.uint32)

    rows = pid * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_H)
    mask = (rows[:, None] < num_rows) & (cols[None, :] < H)
    offs = rows.to(tl.int64)[:, None] * H + cols[None, :]

    # 1. Stage this program's rows in the rank's input buffer.
    x = tl.load(inp_ptr + offs, mask=mask)
    tl.store(in_buf_ptr + offs, x, mask=mask)
    tl.debug_barrier()

    # 2. Wait until every rank has staged these rows.
    flag_off = pid.to(tl.int64) * (FLAG_SPACING * 4)
    _arrive(flag_mc_addr + flag_off, FENCE_SYS)
    _wait(flag_uc_addr + flag_off, (epoch + 1) * WORLD_SIZE)
    tl.debug_barrier()

    # 3. Reduce rank r's slice (the r-th contiguous share of the rows' 16-byte
    #    vectors) in the switch and multicast it.
    VPR: tl.constexpr = H // 8
    valid_rows = tl.minimum(num_rows - pid * ROWS, ROWS)
    total = valid_rows * VPR
    share = (total + WORLD_SIZE - 1) // WORLD_SIZE
    for v0 in range(0, share, VEC_BLOCK):
        j = rank * share + v0 + tl.arange(0, VEC_BLOCK)
        vmask = (v0 + tl.arange(0, VEC_BLOCK) < share) & (j < total)
        r_in = j // VPR
        voff = ((pid * ROWS + r_in).to(tl.int64) * H + (j - r_in * VPR) * 8) * 2
        a, b, c, d = _multimem_ld_reduce_bf16x8(in_mc_addr + voff, vmask)
        _multimem_st_b128(
            out_mc_addr + voff,
            _no_sentinel(a),
            _no_sentinel(b),
            _no_sentinel(c),
            _no_sentinel(d),
            vmask,
        )

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    # 4. Wait for the rows in the local output buffer, copy them out, re-arm.
    y = tl.load(out_buf_ptr + offs, mask=mask, other=0, volatile=True)
    pending = tl.max(tl.max(((y == _SENTINEL) & mask).to(tl.int32), axis=1), axis=0)
    while pending != 0:
        y = tl.load(out_buf_ptr + offs, mask=mask, other=0, volatile=True)
        pending = tl.max(tl.max(((y == _SENTINEL) & mask).to(tl.int32), axis=1), axis=0)
    tl.store(out_ptr + offs, y, mask=mask)
    tl.store(out_buf_ptr + offs, tl.full([ROWS, BLOCK_H], _SENTINEL, tl.int16), mask=mask)
    tl.store(epoch_ptr + pid, (epoch + 1).to(tl.int32))


@triton.jit
def _globaltimer(dummy):
    """Nanosecond global timer (debug timestamps)."""
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;",
        "=l,r,~{memory}",
        args=[dummy],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _spin_while_sentinel(ptrs, mask):
    """Spin until the 32-bit word at each masked pointer holds no 0xFFFF half."""
    return tl.inline_asm_elementwise(
        """
        {
        .reg .u32 %w, %h;
        .reg .pred %p_skip, %p_wait;
        setp.eq.s32 %p_skip, $2, 0;
        mov.u32 $0, 0;
        @%p_skip bra NVLS_SPIN_DONE;
        NVLS_SPIN_LOOP:
        ld.volatile.global.u32 %w, [$1];
        and.b32 %h, %w, 65535;
        setp.eq.u32 %p_wait, %h, 65535;
        shr.u32 %h, %w, 16;
        setp.eq.or.u32 %p_wait, %h, 65535, %p_wait;
        @%p_wait bra NVLS_SPIN_LOOP;
        NVLS_SPIN_DONE:
        }
        """,
        "=r,l,r,~{memory}",
        args=[ptrs, mask.to(tl.int32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _has_sentinel(w):
    """Whether either BF16 half of the 32-bit word ``w`` holds the sentinel 0xFFFF."""
    return ((w & 0xFFFF) == 0xFFFF) | (((w >> 16) & 0xFFFF) == 0xFFFF)


@triton.jit(do_not_specialize=["rank", "n_vec", "ch", "n_chunks", "p_a", "p_r", "p_c"])
def _nvls_allreduce_v2_kernel(
    inp_ptr,
    out_ptr,
    in_buf_ptr,
    out_buf_ptr,
    in_mc_addr,
    out_mc_addr,
    flag_uc_addr,
    flag_mc_addr,
    epoch_ptr,
    dbg_ptr,
    rank,
    n_vec,
    ch,
    n_chunks,
    p_a,
    p_r,
    p_c,
    VRT: tl.constexpr,
    GROUP: tl.constexpr,
    RTILE: tl.constexpr,
    CB: tl.constexpr,
    PROBE: tl.constexpr,
    FLAG_SPACING: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    USE_PDL: tl.constexpr,
    DEBUG: tl.constexpr,
):
    """Chunk-pipelined multicast all-reduce of ``n_vec`` 16-byte vectors.

    The message is cut into chunks of ``ch`` vectors (a multiple of
    ``WORLD_SIZE``, at most ``CB``); rank ``r`` reduces the ``r``-th
    ``ch / WORLD_SIZE`` vectors (at most ``VRT``) of every chunk. Three groups
    of persistent programs walk the chunks in order:

    * stagers (``p_a`` programs, chunks ``pid, pid + p_a, ...``) copy a chunk
      into this rank's input buffer and arrive on the chunk's flag;
    * reducers (``p_r`` programs) take ``GROUP`` consecutive chunks at a time,
      wait until every rank has arrived on them, reduce this rank's slices in
      the switch and multicast the sums into every rank's output buffer;
    * drainers (``p_c`` programs) wait until a chunk is complete in the local
      output buffer (no sentinel left), copy it out, re-arm it and count the
      chunk's use in its epoch.

    A chunk is reduced as soon as every rank has staged it, so the input copy,
    the NVLink traffic and the output copy of different chunks overlap.

    A chunk's flag target is ``(epoch + 1) * WORLD_SIZE``. The drainer bumps the
    epoch only after the chunk is complete, i.e. after this rank's reducer has
    read it. The input and output buffers are reused by the next call only after
    this call's kernel has finished on this rank, which needs every chunk
    complete, i.e. every rank's reduction of this call done. Stagers and
    reducers never wait on drainers, and programs start in ``pid`` order, so a
    grid that is not fully resident still makes progress.
    """
    pid = tl.program_id(0)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    FLAG_BYTES: tl.constexpr = FLAG_SPACING * 4
    if pid < p_a:
        for c in range(pid, n_chunks, p_a):
            c0 = c * ch
            c1 = tl.minimum(c0 + ch, n_vec)
            for b in range(c0, c1, CB):
                v = b + tl.arange(0, CB)
                m = (v < c1)[:, None]
                offs = v.to(tl.int64)[:, None] * 4 + tl.arange(0, 4)[None, :]
                x = tl.load(inp_ptr + offs, mask=m)
                tl.store(in_buf_ptr + offs, x, mask=m)
            tl.debug_barrier()
            _arrive(flag_mc_addr + c.to(tl.int64) * FLAG_BYTES, False)
            if DEBUG:
                tl.store(dbg_ptr + c * 8, _globaltimer(c))
        if USE_PDL:
            tl.extra.cuda.gdc_launch_dependents()
    elif pid < p_a + p_r:
        # RTILE >= threads: a smaller tile would be replicated across threads,
        # and every multimem instruction must run exactly once per vector.
        lane = tl.arange(0, RTILE)
        sub = lane // VRT
        li = lane % VRT
        n_groups = (n_chunks + GROUP - 1) // GROUP
        for g in range(pid - p_a, n_groups, p_r):
            for i in range(GROUP):
                c = g * GROUP + i
                if c < n_chunks:
                    epoch = tl.load(epoch_ptr + c)
                    _wait(flag_uc_addr + c.to(tl.int64) * FLAG_BYTES, (epoch + 1) * WORLD_SIZE)
                    if DEBUG:
                        tl.store(dbg_ptr + c * 8 + 1, _globaltimer(c))
            tl.debug_barrier()
            cl = g * GROUP + sub
            c0 = cl * ch
            chunk_len = tl.minimum(ch, n_vec - c0)
            share = (chunk_len + WORLD_SIZE - 1) // WORLD_SIZE
            j = rank * share + li
            m = (sub < GROUP) & (cl < n_chunks) & (li < share) & (j < chunk_len)
            voff = (c0 + j).to(tl.int64) * 16
            a, b, cc, d = _multimem_ld_reduce_bf16x8(in_mc_addr + voff, m)
            _multimem_st_b128(
                out_mc_addr + voff,
                _no_sentinel(a),
                _no_sentinel(b),
                _no_sentinel(cc),
                _no_sentinel(d),
                m,
            )
            if DEBUG:
                for i in range(GROUP):
                    c = g * GROUP + i
                    if c < n_chunks:
                        tl.store(dbg_ptr + c * 8 + 2, _globaltimer(c))
        if USE_PDL:
            tl.extra.cuda.gdc_launch_dependents()
    else:
        SLICES: tl.constexpr = triton.next_power_of_2(WORLD_SIZE)
        k = tl.arange(0, SLICES)
        for c in range(pid - p_a - p_r, n_chunks, p_c):
            c0 = c * ch
            c1 = tl.minimum(c0 + ch, n_vec)
            if PROBE:
                # Wait on the first word of every rank's slice before reading
                # the whole chunk, so that waiting costs little L2 bandwidth.
                share = (c1 - c0 + WORLD_SIZE - 1) // WORLD_SIZE
                pm = (k < WORLD_SIZE) & (k * share < c1 - c0)
                _spin_while_sentinel(out_buf_ptr + (c0 + k * share).to(tl.int64) * 4, pm)
                tl.debug_barrier()
            for b in range(c0, c1, CB):
                v = b + tl.arange(0, CB)
                m = (v < c1)[:, None]
                offs = v.to(tl.int64)[:, None] * 4 + tl.arange(0, 4)[None, :]
                y = tl.load(out_buf_ptr + offs, mask=m, other=0, volatile=True)
                pending = tl.max(tl.max((_has_sentinel(y) & m).to(tl.int32), axis=1), axis=0)
                while pending != 0:
                    y = tl.load(out_buf_ptr + offs, mask=m, other=0, volatile=True)
                    pending = tl.max(tl.max((_has_sentinel(y) & m).to(tl.int32), axis=1), axis=0)
                tl.store(out_ptr + offs, y, mask=m)
                tl.store(out_buf_ptr + offs, tl.full([CB, 4], -1, tl.int32), mask=m)
            epoch = tl.load(epoch_ptr + c)
            tl.store(epoch_ptr + c, epoch + 1)
            if DEBUG:
                tl.store(dbg_ptr + c * 8 + 3, _globaltimer(c))
        if USE_PDL:
            tl.extra.cuda.gdc_launch_dependents()


class _MulticastBuffer:
    """One rank's slice of a multicast object, mapped unicast and multicast.

    Uses fabric handles and, unless TRTLLM_NVLS_AR_MC_GRANULARITY=recommended,
    the minimum multicast granularity (2 MiB on GB300) instead of the
    recommended one (512 MiB), so a workspace costs its own size rather than a
    512 MiB page.
    """

    def __init__(self, size: int, comm, dev_id: int):
        handle_type = cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        prop = cuda.CUmemAllocationProp()
        prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = dev_id
        prop.requestedHandleTypes = handle_type
        alloc_gran = _check_cu_result(
            cuda.cuMemGetAllocationGranularity(
                prop, cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
            )
        )
        mc_prop = cuda.CUmulticastObjectProp()
        mc_prop.numDevices = comm.Get_size()
        mc_prop.handleTypes = handle_type
        mc_prop.size = size
        mc_gran_flag = (
            cuda.CUmulticastGranularity_flags.CU_MULTICAST_GRANULARITY_RECOMMENDED
            if os.environ.get("TRTLLM_NVLS_AR_MC_GRANULARITY", "min") == "recommended"
            else cuda.CUmulticastGranularity_flags.CU_MULTICAST_GRANULARITY_MINIMUM
        )
        mc_gran = _check_cu_result(cuda.cuMulticastGetGranularity(mc_prop, mc_gran_flag))
        gran = max(alloc_gran, mc_gran)
        self.size = (size + gran - 1) // gran * gran
        mc_prop.size = self.size

        self.mem = _check_cu_result(cuda.cuMemCreate(self.size, prop, 0))
        if comm.Get_rank() == 0:
            self.mc = _check_cu_result(cuda.cuMulticastCreate(mc_prop))
            fabric = _check_cu_result(cuda.cuMemExportToShareableHandle(self.mc, handle_type, 0))
            fabric_data = comm.bcast(fabric.data, root=0)
        else:
            fabric_data = comm.bcast(None, root=0)
            self.mc = _check_cu_result(
                cuda.cuMemImportFromShareableHandle(fabric_data, handle_type)
            )
        _check_cu_result(cuda.cuMulticastAddDevice(self.mc, dev_id))
        comm.Barrier()
        _check_cu_result(cuda.cuMulticastBindMem(self.mc, 0, self.mem, 0, self.size, 0))

        access = cuda.CUmemAccessDesc()
        access.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = dev_id
        access.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self.uc_ptr = int(_check_cu_result(cuda.cuMemAddressReserve(self.size, gran, 0, 0)))
        _check_cu_result(cuda.cuMemMap(self.uc_ptr, self.size, 0, self.mem, 0))
        _check_cu_result(cuda.cuMemSetAccess(self.uc_ptr, self.size, [access], 1))
        self.mc_ptr = int(_check_cu_result(cuda.cuMemAddressReserve(self.size, gran, 0, 0)))
        _check_cu_result(cuda.cuMemMap(self.mc_ptr, self.size, 0, self.mc, 0))
        _check_cu_result(cuda.cuMemSetAccess(self.mc_ptr, self.size, [access], 1))
        self.dev_id = dev_id
        comm.Barrier()

    def uc_tensor(self, offset: int, nbytes: int, dtype: torch.dtype) -> torch.Tensor:
        """Unicast view of ``[offset, offset + nbytes)`` of this rank's slice."""
        return pack_strided_memory(
            self.uc_ptr + offset, nbytes, nbytes, 1, dtype, self.dev_id
        ).view(-1)


class NVLSAllReduce:
    """Multicast (NVLink SHARP) all-reduce of BF16 tensors over the TP group.

    Workspaces are shared per ``(mapping, slot)``. Calls on one slot must be
    ordered by one CUDA stream on every rank; use different slots for
    all-reduces that may run concurrently on different streams.

    ``version=2`` (default from ``TRTLLM_NVLS_AR_V2``) sends messages of at least
    ``v2["min_bytes"]`` to the chunk-pipelined kernel and smaller ones to the
    one-shot kernel; ``v2_config`` overrides the ``TRTLLM_NVLS_AR_V2_*`` knobs.
    """

    _workspaces: Dict[Tuple[Mapping, str], Dict] = {}

    MAX_PROGRAMS = 1 << 13
    # Flag words per program: every flag in its own 128-byte line.
    FLAG_SPACING = 32

    def __init__(
        self,
        mapping: Mapping,
        slot: str = "main",
        max_bytes: Optional[int] = None,
        rows_per_program: Optional[int] = None,
        num_warps: Optional[int] = None,
        fence_sys: Optional[bool] = None,
        version: Optional[int] = None,
        v2_config: Optional[Dict[str, int]] = None,
    ):
        self.mapping = mapping
        self.world_size = mapping.tp_size
        self.rank = mapping.tp_rank
        self.max_bytes = max_bytes or NVLS_AR_MAX_BYTES
        self.rows_per_program = rows_per_program or int(os.environ.get("TRTLLM_NVLS_AR_ROWS", "1"))
        self.num_warps = num_warps or int(os.environ.get("TRTLLM_NVLS_AR_WARPS", "8"))
        if fence_sys is None:
            fence_sys = os.environ.get("TRTLLM_NVLS_AR_FENCE_SYS", "0") == "1"
        self.fence_sys = fence_sys
        if version is None:
            version = 2 if os.environ.get("TRTLLM_NVLS_AR_V2", "0") == "1" else 1
        self.version = version
        num_sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        cfg = {key: int(os.environ.get(env, default)) for key, (env, default) in _V2_KNOBS.items()}
        cfg.update(v2_config or {})
        for key, default in (
            ("stagers", num_sms // 3),
            ("reducers", num_sms),
            ("drainers", num_sms),
        ):
            if cfg[key] <= 0:
                cfg[key] = max(1, default)
        self.v2 = cfg
        # Optional int64 tensor of per-chunk timestamps (benchmarks only).
        self.debug_timestamps: Optional[torch.Tensor] = None
        self.use_pdl = get_env_enable_pdl()
        self._ws = self._get_workspace(mapping, slot, self.max_bytes)
        if self.version == 2:
            kernels = (
                f"pipelined kernel (v2) from {cfg['min_bytes']} bytes, one-shot kernel (v1) below"
            )
        else:
            kernels = "one-shot kernel (v1)"
        logger.info_once(
            f"[NVLS AR] active on tp_size={self.world_size}: {kernels}; AllReduce routes plain "
            f"BF16 sums of [{NVLS_AR_MIN_BYTES}, {NVLS_AR_MAX_BYTES}] bytes here ({__file__})",
            key=("nvls_ar_active", self.world_size, self.version),
        )

    @staticmethod
    def is_supported(mapping: Mapping) -> bool:
        return (
            mapping.tp_size >= 2
            and get_sm_version() >= 90
            and not mapping.has_cp()
            and not mapping.enable_attention_dp
        )

    @classmethod
    def _get_workspace(cls, mapping: Mapping, slot: str, max_bytes: int) -> Dict:
        key = (mapping, slot)
        ws = cls._workspaces.get(key)
        if ws is not None:
            if ws["data_bytes"] < max_bytes:
                raise ValueError(
                    f"NVLS all-reduce workspace '{slot}' holds {ws['data_bytes']} bytes, "
                    f"{max_bytes} requested"
                )
            return ws
        data_bytes = math.ceil(max_bytes / (1 << 20)) * (1 << 20)
        # Two flag arrays: per-program flags of the one-shot kernel, per-chunk
        # flags of the pipelined kernel.
        flag_bytes = 2 * cls.MAX_PROGRAMS * cls.FLAG_SPACING * 4
        comm = mpi_comm().Split(
            int(mapping.pp_rank * mapping.cp_size + mapping.cp_rank), mapping.tp_rank
        )
        buf = _MulticastBuffer(2 * data_bytes + flag_bytes, comm, torch.cuda.current_device())
        in_buf = buf.uc_tensor(0, data_bytes, torch.int16)
        out_buf = buf.uc_tensor(data_bytes, data_bytes, torch.int16)
        flags = buf.uc_tensor(2 * data_bytes, flag_bytes, torch.int32)
        out_buf.fill_(_SENTINEL_BITS)
        flags.zero_()
        # Per-program (one-shot) and per-chunk (pipelined) call counters, local
        # to this rank.
        epoch = torch.zeros(cls.MAX_PROGRAMS, dtype=torch.int32, device="cuda")
        epoch_v2 = torch.zeros(cls.MAX_PROGRAMS, dtype=torch.int32, device="cuda")
        torch.cuda.synchronize()
        comm.Barrier()
        ws = {
            "buffer": buf,
            "comm": comm,
            "in_buf": in_buf,
            "out_buf": out_buf,
            "flags": flags,
            "epoch": epoch,
            "epoch_v2": epoch_v2,
            "data_bytes": data_bytes,
            "in_mc_addr": buf.mc_ptr,
            "out_mc_addr": buf.mc_ptr + data_bytes,
            "flag_uc_addr": flags.data_ptr(),
            "flag_mc_addr": buf.mc_ptr + 2 * data_bytes,
            "flag_v2_uc_addr": flags.data_ptr() + flag_bytes // 2,
            "flag_v2_mc_addr": buf.mc_ptr + 2 * data_bytes + flag_bytes // 2,
        }
        logger.info(
            f"[NVLS AR] workspace slot={slot} tp={mapping.tp_size} rank={mapping.tp_rank} "
            f"{buf.size} bytes"
        )
        cls._workspaces[key] = ws
        return ws

    def can_use(self, inp: torch.Tensor) -> bool:
        if inp.dtype != torch.bfloat16 or not inp.is_contiguous() or inp.dim() == 0:
            return False
        hidden = inp.shape[-1]
        rows = inp.numel() // max(hidden, 1)
        if self._uses_v2(inp):
            fits = triton.cdiv(inp.numel() // 8, self._v2_chunk_vectors()) <= self.MAX_PROGRAMS
        else:
            fits = hidden <= 16384 and triton.cdiv(rows, self.rows_per_program) <= self.MAX_PROGRAMS
        return hidden % 8 == 0 and rows > 0 and fits and inp.numel() * 2 <= self._ws["data_bytes"]

    def _uses_v2(self, inp: torch.Tensor) -> bool:
        """Whether ``inp`` goes to the pipelined kernel (else the one-shot kernel)."""
        return self.version == 2 and inp.numel() * inp.element_size() >= self.v2["min_bytes"]

    def _v2_chunk_vectors(self) -> int:
        """16-byte vectors per chunk of the pipelined kernel (a multiple of the group size)."""
        return max(1, self.v2["chunk_bytes"] // 16 // self.world_size) * self.world_size

    def __call__(self, inp: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sum the contiguous BF16 tensor ``inp`` over the group."""
        hidden = inp.shape[-1]
        rows = inp.numel() // hidden
        if out is None:
            out = torch.empty_like(inp)
        ws = self._ws
        use_v2 = self._uses_v2(inp)
        if use_v2 not in _FIRST_CALL_LOGGED:
            _FIRST_CALL_LOGGED.add(use_v2)
            capturing = " (CUDA graph capture)" if torch.cuda.is_current_stream_capturing() else ""
            logger.info(
                f"[NVLS AR] first {'pipelined (v2)' if use_v2 else 'one-shot (v1)'} kernel "
                f"call{capturing}: shape {tuple(inp.shape)}, {inp.numel() * 2} bytes"
            )
        if use_v2:
            self._launch_v2(inp, out)
            return out
        rpp = self.rows_per_program
        _nvls_allreduce_kernel[(triton.cdiv(rows, rpp),)](
            inp.view(torch.int16),
            out.view(torch.int16),
            ws["in_buf"],
            ws["out_buf"],
            ws["in_mc_addr"],
            ws["out_mc_addr"],
            ws["flag_uc_addr"],
            ws["flag_mc_addr"],
            ws["epoch"],
            self.rank,
            rows,
            H=hidden,
            BLOCK_H=triton.next_power_of_2(hidden),
            ROWS=rpp,
            VEC_BLOCK=64 * self.num_warps,
            FLAG_SPACING=self.FLAG_SPACING,
            WORLD_SIZE=self.world_size,
            FENCE_SYS=self.fence_sys,
            USE_PDL=self.use_pdl,
            num_warps=self.num_warps,
            launch_pdl=self.use_pdl,
        )
        return out

    def _launch_v2(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        ws = self._ws
        cfg = self.v2
        n_vec = inp.numel() // 8
        ch = self._v2_chunk_vectors()
        n_chunks = triton.cdiv(n_vec, ch)
        threads = 32 * cfg["warps"]
        vrt = triton.next_power_of_2(triton.cdiv(ch, self.world_size))
        group = max(1, min(cfg["group"], threads // vrt))
        p_a = min(n_chunks, cfg["stagers"])
        p_r = min(triton.cdiv(n_chunks, group), cfg["reducers"])
        p_c = min(n_chunks, cfg["drainers"])
        debug = self.debug_timestamps is not None
        _nvls_allreduce_v2_kernel[(p_a + p_r + p_c,)](
            inp.view(torch.int32),
            out.view(torch.int32),
            ws["in_buf"].view(torch.int32),
            ws["out_buf"].view(torch.int32),
            ws["in_mc_addr"],
            ws["out_mc_addr"],
            ws["flag_v2_uc_addr"],
            ws["flag_v2_mc_addr"],
            ws["epoch_v2"],
            self.debug_timestamps if debug else ws["epoch_v2"],
            self.rank,
            n_vec,
            ch,
            n_chunks,
            p_a,
            p_r,
            p_c,
            VRT=vrt,
            GROUP=group,
            RTILE=max(threads, group * vrt),
            CB=max(threads, triton.next_power_of_2(ch)),
            PROBE=bool(cfg["probe"]),
            FLAG_SPACING=self.FLAG_SPACING,
            WORLD_SIZE=self.world_size,
            USE_PDL=self.use_pdl,
            DEBUG=debug,
            num_warps=cfg["warps"],
            launch_pdl=self.use_pdl,
        )

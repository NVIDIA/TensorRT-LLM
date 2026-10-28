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
"""Host side of the Qwen3.5 fused CP-lite partial decode attention (``kernels/blackwell/q35_cp_partial_attn.py``).

One launch computes this CP rank's partial attention over the pages it owns for every q row of the KV group, written
straight into alltoall_merge's input layout (contract: q35_cp.q35_cp_fused_partial_ref). One CTA per SM; an in-kernel
plan shares the CTAs among requests (balanced_split_plan is its host mirror); above 64 requests the two-slot plan
(balanced_split_plan_two) lets a CTA compute two requests. The per-device grid-barrier counters and workspace must
be allocated by q35_cp_fused_prepare() or an eager call before CUDA graph capture.
"""

import functools
import os
from typing import Dict, Optional, Union

import torch

from tensorrt_llm.logger import logger

PAGE = 64
HEAD_DIM = 256
MAX_ROWS = 128
# Plan variants by batch: one request per lane of the plan warp up to 32, two per lane up to 64 (wide plan), and above
# 64 the two-slot kernel (kernel_two, four requests per lane): a CTA computes up to two requests, so no request is held
# to whole CTAs. The one-slot plans give every request at least one CTA; past ~SMs / 2 uniform long requests then get
# exactly one each and the slowest CTA runs a whole request (64k contexts, b96: 128 tiles where 81 would balance).
# The quad plan (the one-slot plan for 65..128 requests) is reachable through ``plan`` only.
_LANE_PLAN_MAX_BATCH = 32
_WIDE_PLAN_MAX_BATCH = 64
_QUAD_PLAN_MAX_BATCH = 128
_TWO_SLOT_MAX_BATCH = 128
_PLAN_MAX_BATCH = dict(
    lane=_LANE_PLAN_MAX_BATCH,
    wide=_WIDE_PLAN_MAX_BATCH,
    quad=_QUAD_PLAN_MAX_BATCH,
    two=_TWO_SLOT_MAX_BATCH,
)
# Largest batch of one call: the lane plan's 32 unless raised with TRTLLM_Q35_ATTN_CP_FUSED_MAX_BATCH (up to 128,
# e.g. for TEP8 at batch 64 / 128); single-token (MTP draft) calls up to
# TRTLLM_Q35_ATTN_CP_FUSED_MAX_BATCH_Q1 (default: the same).
MAX_BATCH = max(
    1, min(_TWO_SLOT_MAX_BATCH, int(os.environ.get("TRTLLM_Q35_ATTN_CP_FUSED_MAX_BATCH", "32")))
)
MAX_BATCH_Q1 = max(
    1, min(MAX_BATCH, int(os.environ.get("TRTLLM_Q35_ATTN_CP_FUSED_MAX_BATCH_Q1", str(MAX_BATCH))))
)
LOG2E = 1.4426950408889634

# Prefix pages below the new tokens' first page that are read only after griddepcontrol.wait (MTP passes write them
# earlier in the step); older pages are requested before the wait, overlapping the q all-gather.
_FRESH_MARGIN = 1
# E4M3 P shift (log2 units): P = exp2(x - m + shift) before the FP8 conversion keeps probabilities down to
# 2^-(6 + shift) of the row max E4M3 normals; 5 matches trtllm-gen's error level.
_P_SHIFT = 5.0

# Declared pointer alignment (bytes), used at compile and call time: q / KV are TMA sources, the workspace takes
# 16-byte stores; scalars may be 4-byte aligned views such as scales[2:3]. out_o is 16 when its rows allow 8-element
# vector stores, else 4 (alltoall_merge reads it as int32).
_ALIGN = dict(q=16, kv=16, tab=4, seq=4, bmm1=4, bmm2=4, lse=4, ws=16, cnt=8)


def cp_partial_balanced_ws_bytes(rows: int, n_ctas: int, slots: int = 1) -> int:
    """Mirror of q35_cp_partial_attn.cp_partial_balanced_geometry()["ws_bytes"].

    Per CTA: one FP16 O slot (largest per-CTA share over c = 1..n_ctas CTAs per request) and 16 * rows bytes of LSE;
    ``slots`` per CTA (2 for the two-slot plan).
    """
    total = rows * HEAD_DIM
    slot_o = 0
    for c in range(1, n_ctas + 1):
        chunk = 8 * ((total + 8 * c - 1) // (8 * c))
        chunk_pad = chunk + (8 if (chunk // 8) % 2 == 0 else 0)
        slot_o = max(slot_o, ((total + chunk - 1) // chunk) * chunk_pad * 2)
    return slots * n_ctas * (16 * ((slot_o + 15) // 16) + 16 * rows)


def balanced_split_plan(seq_lens, cp_rank: int, cp_size: int, n_ctas: int):
    """Host mirror of the balanced split's in-kernel plan (Q35CpPartialAttnKernel._plan).

    Returns (T, c, ctas): T = the largest tile count of a CTA's block, c[r] = CTAs of request r, ctas[i] = (request,
    index in the request, first tile, tiles) of CTA i (tile k of a request = its owned pages 2k, 2k + 1 on this rank).
    """
    t = [
        ((((int(L) + PAGE - 1) // PAGE) - cp_rank + cp_size - 1) // cp_size + 1) // 2
        for L in seq_lens
    ]
    b = len(t)
    assert 1 <= b < n_ctas and b <= _QUAD_PLAN_MAX_BATCH
    total, tmax = sum(t), max(t)
    lo = max(-(-total // n_ctas), 1)
    hi = max(min(tmax, -(-total // (n_ctas - b))), lo)
    while lo < hi:
        mid = (lo + hi) // 2
        if sum(max(-(-x // mid), 1) for x in t) <= n_ctas:
            hi = mid
        else:
            lo = mid + 1
    c = [max(-(-x // lo), 1) for x in t]
    top = max(r for r in range(b) if t[r] == tmax)
    c[top] += n_ctas - sum(c)
    ctas = []
    for r in range(b):
        base, rem = divmod(t[r], c[r])
        for j in range(c[r]):
            ctas.append((r, j, j * base + min(j, rem), base + (1 if j < rem else 0)))
    return lo, c, ctas


def balanced_split_plan_two(seq_lens, cp_rank: int, cp_size: int, n_ctas: int):
    """Host mirror of the two-slot plan (q35_cp_partial_attn.Q35CpPartialAttnKernel._plan_two).

    T = the smallest tile count with sum_r max(t_r, T) <= n_ctas * T; request r takes the virtual tile range [V_r, V_r
    + max(t_r, T)) (requests in order), CTA i the range [i * T, (i + 1) * T). Returns (T, slots): slots[i] = [slot 0,
    slot 1] of CTA i, each None or (request, index among the request's CTAs, its CTAs, first workspace slot, first
    tile, tiles, CTAs with work); slot 0 is the request holding the CTA's first position, slot 1 the one starting
    inside its range.
    """
    t = [
        ((((int(L) + PAGE - 1) // PAGE) - cp_rank + cp_size - 1) // cp_size + 1) // 2
        for L in seq_lens
    ]
    b = len(t)
    assert 1 <= b < n_ctas and b <= _TWO_SLOT_MAX_BATCH
    total, tmax = sum(t), max(t)
    lo = max(-(-total // n_ctas), 1)
    hi = max(tmax, lo)
    while lo < hi:
        mid = (lo + hi) // 2
        if sum(max(x, mid) for x in t) <= n_ctas * mid:
            hi = mid
        else:
            lo = mid + 1
    T = lo
    slots = [[None, None] for _ in range(n_ctas)]
    V = s0 = 0
    for r, x in enumerate(t):
        te = max(x, 1)
        f, last = V // T, (V + te - 1) // T
        c = last - f + 1
        for j in range(c):
            i = f + j
            lo_t, hi_t = max(V, i * T) - V, min(V + te, (i + 1) * T) - V
            slot = (r, j, c, s0, lo_t if x else 0, hi_t - lo_t if x else 0, c if x else 0)
            k = 0 if V <= i * T else 1
            assert slots[i][k] is None
            slots[i][k] = slot
        s0 += c
        V += max(x, T)
    assert V <= n_ctas * T and s0 <= 2 * n_ctas
    return T, slots


def _num_ctas(sm_count: int) -> int:
    """Grid of the kernel (any batch): one CTA per SM."""
    return max(sm_count, 1)


def q35_cp_fused_supported(q_len: int, cp_size: int, hq: int, batch: int, sm_count: int) -> bool:
    """Shapes the kernel handles (D 256, FP8 q / KV and 64-token pages are checked by the caller)."""
    rows = q_len * cp_size * hq
    max_batch = MAX_BATCH if q_len > 1 else MAX_BATCH_Q1
    # the plan gives every request at least one CTA
    return 1 <= rows <= MAX_ROWS and 1 <= batch <= max_batch and batch < _num_ctas(sm_count)


class _DeviceState:
    """Per-device grid-barrier counters and partial workspace (sized for every supported call).

    counters: int64 [MAX_BATCH + 1, MAX_BATCH], row b serves batch size b; zero-initialized once, never reset.
    """

    def __init__(self, device: torch.device, sm_count: int):
        self.sm_count = sm_count
        need = cp_partial_balanced_ws_bytes(
            MAX_ROWS, _num_ctas(sm_count), 2 if MAX_BATCH > _WIDE_PLAN_MAX_BATCH else 1
        )
        self.workspace = torch.empty(need // 4 + 64, dtype=torch.float32, device=device)
        self.counters = torch.zeros(MAX_BATCH + 1, MAX_BATCH, dtype=torch.int64, device=device)


_STATES: Dict[int, _DeviceState] = {}


def _sm_count(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def q35_cp_fused_prepare(device: Optional[torch.device] = None, sm_count: int = 0) -> None:
    """Allocate this device's counters / workspace (call before CUDA graph capture)."""
    device = torch.device("cuda", torch.cuda.current_device()) if device is None else device
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _STATES:
        _STATES[idx] = _DeviceState(torch.device("cuda", idx), sm_count or _sm_count(device))


def _state(device: torch.device, sm_count: int) -> _DeviceState:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    st = _STATES.get(idx)
    if st is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "q35_cp_fused_partial: call q35_cp_fused_prepare() (or one eager call) before CUDA "
                "graph capture: the grid-barrier counters must be zero-initialized persistent memory"
            )
        q35_cp_fused_prepare(device, sm_count)
        st = _STATES[idx]
    return st


def _ptr(t: torch.Tensor, dtype, align: int):
    import cutlass.cute as cute

    return cute.runtime.make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


@functools.lru_cache(maxsize=None)
def _compiled(
    q_len: int,
    gh: int,
    hql: int,
    cp_size: int,
    n_ctas: int,
    bmm1_from_ptr: bool,
    bmm2_from_ptr: bool,
    o_vec16: bool,
    use_pdl: bool,
    plan: str,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from ....kernels.blackwell.q35_cp_partial_attn import Q35CpPartialAttnKernel

    kernel = Q35CpPartialAttnKernel(
        q_len=q_len,
        gh=gh,
        hql=hql,
        cp_size=cp_size,
        n_ctas=n_ctas,
        bmm1_from_ptr=bmm1_from_ptr,
        bmm2_from_ptr=bmm2_from_ptr,
        o_vec16=o_vec16,
        use_pdl=use_pdl,
        fresh_margin_pages=_FRESH_MARGIN,
        p_shift=_P_SHIFT,
        wide_plan=plan == "wide",
        quad_plan=plan == "quad",
        two_slot=plan == "two",
    )
    assert kernel.ws_bytes == cp_partial_balanced_ws_bytes(
        q_len * gh, n_ctas, 2 if plan == "two" else 1
    )
    dev = torch.device("cuda")
    f32 = torch.empty(1024, dtype=torch.float32, device=dev)
    i32 = torch.zeros(64, dtype=torch.int32, device=dev)
    u8 = torch.empty(64 * HEAD_DIM, dtype=torch.uint8, device=dev)
    bf = torch.empty(1024, dtype=torch.bfloat16, device=dev)
    i64 = torch.zeros(16, dtype=torch.int64, device=dev)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fp8 = cutlass.Float8E4M3FN
    a = _ALIGN
    I32 = cutlass.Int32
    return cute.compile(
        kernel,
        _ptr(u8, fp8, a["q"]),
        _ptr(u8, fp8, a["kv"]),
        I32(64),
        _ptr(i32, I32, a["tab"]),
        I32(2),
        I32(1),
        _ptr(i32, I32, a["seq"]),
        I32(0),
        _ptr(f32, cutlass.Float32, a["bmm1"]),
        cutlass.Float32(1.0),
        _ptr(f32, cutlass.Float32, a["bmm2"]),
        cutlass.Float32(1.0),
        _ptr(bf, cutlass.BFloat16, 16 if o_vec16 else 4),
        I32(HEAD_DIM),
        I32(HEAD_DIM),
        I32(HEAD_DIM),
        _ptr(f32, cutlass.Float32, a["lse"]),
        I32(1),
        I32(1),
        I32(1),
        _ptr(f32, cutlass.Float32, a["ws"]),
        _ptr(i64, cutlass.Int64, a["cnt"]),
        I32(1),
        stream,
    )


def q35_cp_fused_partial(
    q_all: torch.Tensor,
    kv_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    q_len: int,
    cp_rank: int,
    cp_size: int,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    out_o: torch.Tensor,
    out_lse: torch.Tensor,
    sm_count: int = 0,
    enable_pdl: bool = False,
    plan: Optional[str] = None,
) -> None:
    """This CP rank's partial attention of every group head over the pages it owns (q35_cp_fused_partial_ref).

    Args:
        q_all: [cp, rows, hq, D] FP8, the all-gather output viewed member-major. The native layout, strides (hq * D,
            cp * hq * D, D, 1) at a 16-byte aligned address, is read in place; any other view through a copy.
        kv_pool: [blocks, 1, 64, D] FP8.
        block_tables: [>= b, 2, max_pages] int32 K / V block ids, unit stride on pages.
        seq_lens: [>= b] int32 KV lengths including the q_len new tokens.
        q_len: Tokens per request; rows = b * q_len.
        cp_rank: This rank's index in the KV group.
        cp_size: KV group size.
        bmm1_scale: Float (natural-log domain) or a [>= 2] FP32 tensor (raw, raw * log2e).
        bmm2_scale: Float or a [>= 1] FP32 tensor.
        out_o: [cp, rows, hq, D] bf16, unit stride on D.
        out_lse: [cp, rows, hq] FP32 (base 2).
        sm_count: SMs of the device (0: queried).
        enable_pdl: Launch as a programmatic dependent.
        plan: The plan variant ("lane" <= 32 requests, "wide" <= 64, "quad" or "two" <= 128) instead of the one the
            batch selects (lane, wide, then two); lane / wide / quad compute the same one-slot plan (tests).
    """
    import cuda.bindings.driver as cuda
    import cutlass

    cp, rows, hq, D = q_all.shape
    assert cp == cp_size and D == HEAD_DIM and q_all.dtype == torch.float8_e4m3fn
    assert rows % q_len == 0
    b = rows // q_len
    gh = cp * hq
    R = q_len * gh
    assert 1 <= R <= MAX_ROWS, f"q_len x group heads = {R} > {MAX_ROWS}"
    assert 0 <= cp_rank < cp_size
    assert (
        kv_pool.dim() == 4
        and kv_pool.shape[1] == 1
        and kv_pool.shape[2] == PAGE
        and kv_pool.shape[3] == HEAD_DIM
    )
    assert kv_pool.is_contiguous() and kv_pool.dtype in (torch.float8_e4m3fn, torch.uint8)
    assert block_tables.dim() == 3 and block_tables.shape[0] >= b and block_tables.shape[1] == 2
    assert block_tables.stride(-1) == 1 and block_tables.dtype == torch.int32
    assert seq_lens.dtype == torch.int32 and seq_lens.numel() >= b and seq_lens.stride(0) == 1
    assert (
        tuple(out_o.shape) == (cp, rows, hq, D)
        and out_o.dtype == torch.bfloat16
        and out_o.stride(3) == 1
    )
    assert tuple(out_lse.shape) == (cp, rows, hq) and out_lse.dtype == torch.float32
    assert 1 <= b <= MAX_BATCH
    if plan is None:
        plan = (
            "two" if b > _WIDE_PLAN_MAX_BATCH else ("wide" if b > _LANE_PLAN_MAX_BATCH else "lane")
        )
    assert b <= _PLAN_MAX_BATCH[plan], f"batch {b} above the {plan} plan's {_PLAN_MAX_BATCH[plan]}"
    slots = 2 if plan == "two" else 1
    device = q_all.device
    sms = sm_count or _sm_count(device)
    state = _state(device, sms)
    n_ctas = _num_ctas(state.sm_count)
    assert b < n_ctas <= sms, "more CTAs than requests, every CTA co-resident (one per SM)"
    cnt = state.counters
    ws = state.workspace
    assert ws.numel() * 4 >= cp_partial_balanced_ws_bytes(R, n_ctas, slots), (
        "CP partial workspace too small"
    )
    # q: the native all-gather layout is one [b * rows_per_request, 256] matrix read by TMA in place
    native = (
        q_all.stride(3) == 1
        and q_all.stride(2) == D
        and q_all.stride(0) == hq * D
        and q_all.stride(1) == cp * hq * D
        and q_all.data_ptr() % 16 == 0
    )
    q_src = q_all if native else q_all.permute(1, 0, 2, 3).contiguous()
    b1_ptr = isinstance(bmm1_scale, torch.Tensor)
    b2_ptr = isinstance(bmm2_scale, torch.Tensor)
    if b1_ptr:
        assert (
            bmm1_scale.numel() >= 2
            and bmm1_scale.dtype == torch.float32
            and bmm1_scale.stride(0) == 1
        )
    if b2_ptr:
        assert bmm2_scale.numel() >= 1 and bmm2_scale.dtype == torch.float32
    o_vec16 = out_o.data_ptr() % 16 == 0 and all(out_o.stride(i) % 8 == 0 for i in range(3))
    assert out_o.data_ptr() % 4 == 0 and all(out_o.stride(i) % 2 == 0 for i in range(3)), (
        "out_o: 4-byte aligned rows (alltoall_merge reads it as int32)"
    )
    if plan != "lane":
        logger.info_once(
            f"Qwen3.5 CP-lite fused partial attention: {plan} plan above {_LANE_PLAN_MAX_BATCH} requests "
            f"(first call: batch {b}, q_len {q_len})",
            key=f"q35_cp_fused_{plan}",
        )
    fn = _compiled(q_len, gh, hq, cp, n_ctas, b1_ptr, b2_ptr, o_vec16, bool(enable_pdl), plan)
    fp8 = cutlass.Float8E4M3FN
    f32 = cutlass.Float32
    I32 = cutlass.Int32
    a = _ALIGN
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    dummy = ws
    fn(
        _ptr(q_src, fp8, a["q"]),
        _ptr(kv_pool, fp8, a["kv"]),
        I32(kv_pool.shape[0] * PAGE),
        _ptr(block_tables, I32, a["tab"]),
        I32(block_tables.stride(0)),
        I32(block_tables.stride(1)),
        _ptr(seq_lens, I32, a["seq"]),
        I32(cp_rank),
        _ptr(bmm1_scale if b1_ptr else dummy, f32, a["bmm1"]),
        f32(0.0 if b1_ptr else float(bmm1_scale) * LOG2E),
        _ptr(bmm2_scale if b2_ptr else dummy, f32, a["bmm2"]),
        f32(0.0 if b2_ptr else float(bmm2_scale)),
        _ptr(out_o, cutlass.BFloat16, 16 if o_vec16 else 4),
        I32(out_o.stride(0)),
        I32(out_o.stride(1)),
        I32(out_o.stride(2)),
        _ptr(out_lse, f32, a["lse"]),
        I32(out_lse.stride(0)),
        I32(out_lse.stride(1)),
        I32(out_lse.stride(2)),
        _ptr(ws, f32, a["ws"]),
        _ptr(cnt[b], cutlass.Int64, a["cnt"]),
        I32(b),
        stream,
    )

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
"""Host side of the fused KV-replica split partial decode attention
(attention/kernels/blackwell/gqa_kv_replica_partial_attn.py).

One launch computes this member's partial attention over the pages it owns for every q row of the KV-replica group,
written straight into alltoall_merge's input layout (contract: kv_replica_fused_partial_ref in the unit test). One CTA
per SM (n_ctas <= SMs), all co-resident for the in-kernel grid barrier; an in-kernel plan shares the CTAs among
requests, and above 64 requests the two-slot plan lets a CTA compute two requests. The grid-barrier counters and
workspace live in a KvReplicaFusedState, allocated by kv_replica_fused_state() before CUDA graph capture.
"""

import functools
from typing import Dict, Union

import torch

from tensorrt_llm.logger import logger

from ....cute_dsl_utils import gmem_ptr

PAGE = 64
HEAD_DIM = 256
MAX_ROWS = 128
MAX_Q_LEN = 16
# Plan variants by batch: one request per lane of the plan warp up to 32, two per lane up to 64 (wide plan), and above
# 64 the two-slot kernel (kernel_two, four requests per lane): a CTA computes up to two requests, so no request is held
# to whole CTAs.
_LANE_PLAN_MAX_BATCH = 32
_WIDE_PLAN_MAX_BATCH = 64
_TWO_SLOT_MAX_BATCH = 128
# Single-token (MTP draft) calls above this batch take the merge-free path (gqa_kv_replica.py).
_SINGLE_TOKEN_MAX_BATCH = _WIDE_PLAN_MAX_BATCH
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
    """Mirror of gqa_kv_replica_partial_attn.cp_partial_balanced_geometry()["ws_bytes"].

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


def _num_ctas(sm_count: int) -> int:
    """Grid of the kernel (any batch): one CTA per SM."""
    return max(sm_count, 1)


def kv_replica_fused_supported(
    q_len: int, cp_size: int, hq: int, batch: int, sm_count: int
) -> bool:
    """Shapes the kernel handles (D 256, FP8 q / KV and 64-token pages are checked by the caller)."""
    rows = q_len * cp_size * hq
    max_batch = _TWO_SLOT_MAX_BATCH if q_len > 1 else _SINGLE_TOKEN_MAX_BATCH
    # the plan gives every request at least one CTA
    return 1 <= rows <= MAX_ROWS and 1 <= batch <= max_batch and batch < _num_ctas(sm_count)


class KvReplicaFusedState:
    """Grid-barrier counters and partial workspace of the kernel on one device, for batches up to ``max_batch``.

    Shared by every layer, like the attention workspace: the layers' calls are ordered on one stream. counters: int64
    [max_batch + 1, max_batch], row b serves batch size b; zero-initialized once, never reset.
    """

    def __init__(self, device: torch.device, sm_count: int, max_batch: int):
        self.sm_count = sm_count
        self.max_batch = max_batch
        need = cp_partial_balanced_ws_bytes(
            MAX_ROWS, _num_ctas(sm_count), 2 if max_batch > _WIDE_PLAN_MAX_BATCH else 1
        )
        self.workspace = torch.empty(need // 4 + 64, dtype=torch.float32, device=device)
        self.counters = torch.zeros(max_batch + 1, max_batch, dtype=torch.int64, device=device)


_STATES: Dict[int, KvReplicaFusedState] = {}


def kv_replica_fused_state(
    device: torch.device, sm_count: int, max_num_requests: int
) -> KvReplicaFusedState:
    """This device's state for batches up to ``max_num_requests`` (capped at the kernel's 128).

    A state is allocated or replaced by a larger one only outside CUDA graph capture, so captured graphs keep valid
    addresses; callers request it before capture with the engine's ``max_num_requests``.
    """
    max_batch = max(1, min(_TWO_SLOT_MAX_BATCH, max_num_requests))
    idx = device.index if device.index is not None else torch.cuda.current_device()
    state = _STATES.get(idx)
    if state is None or state.max_batch < max_batch:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "The KV-replica split counters and workspace must be allocated before CUDA graph capture."
            )
        state = KvReplicaFusedState(torch.device("cuda", idx), sm_count, max_batch)
        _STATES[idx] = state
    return state


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

    from ...kernels.blackwell.gqa_kv_replica_partial_attn import GqaKvReplicaPartialAttnKernel

    kernel = GqaKvReplicaPartialAttnKernel(
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
        gmem_ptr(u8, fp8, a["q"]),
        gmem_ptr(u8, fp8, a["kv"]),
        I32(64),
        gmem_ptr(i32, I32, a["tab"]),
        I32(2),
        I32(1),
        gmem_ptr(i32, I32, a["seq"]),
        I32(0),
        gmem_ptr(f32, cutlass.Float32, a["bmm1"]),
        cutlass.Float32(1.0),
        gmem_ptr(f32, cutlass.Float32, a["bmm2"]),
        cutlass.Float32(1.0),
        gmem_ptr(bf, cutlass.BFloat16, 16 if o_vec16 else 4),
        I32(HEAD_DIM),
        I32(HEAD_DIM),
        I32(HEAD_DIM),
        gmem_ptr(f32, cutlass.Float32, a["lse"]),
        I32(1),
        I32(1),
        I32(1),
        gmem_ptr(f32, cutlass.Float32, a["ws"]),
        gmem_ptr(i64, cutlass.Int64, a["cnt"]),
        I32(1),
        stream,
    )


def kv_replica_fused_partial(
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
    state: KvReplicaFusedState,
    enable_pdl: bool = False,
) -> None:
    """This CP rank's partial attention of every group head over the pages it owns (contract: the unit test's
    kv_replica_fused_partial_ref).

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
        state: The device's counters and workspace (kv_replica_fused_state), sized for this batch.
        enable_pdl: Launch as a programmatic dependent.
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
    assert 1 <= b <= state.max_batch
    plan = "two" if b > _WIDE_PLAN_MAX_BATCH else ("wide" if b > _LANE_PLAN_MAX_BATCH else "lane")
    slots = 2 if plan == "two" else 1
    n_ctas = _num_ctas(state.sm_count)
    assert b < n_ctas, "more CTAs than requests, every CTA co-resident (one per SM)"
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
            f"KV-replica split fused partial attention: {plan} plan above {_LANE_PLAN_MAX_BATCH} requests "
            f"(first call: batch {b}, q_len {q_len})",
            key=f"kv_replica_fused_{plan}",
        )
    fn = _compiled(q_len, gh, hq, cp, n_ctas, b1_ptr, b2_ptr, o_vec16, bool(enable_pdl), plan)
    fp8 = cutlass.Float8E4M3FN
    f32 = cutlass.Float32
    I32 = cutlass.Int32
    a = _ALIGN
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    dummy = ws
    fn(
        gmem_ptr(q_src, fp8, a["q"]),
        gmem_ptr(kv_pool, fp8, a["kv"]),
        I32(kv_pool.shape[0] * PAGE),
        gmem_ptr(block_tables, I32, a["tab"]),
        I32(block_tables.stride(0)),
        I32(block_tables.stride(1)),
        gmem_ptr(seq_lens, I32, a["seq"]),
        I32(cp_rank),
        gmem_ptr(bmm1_scale if b1_ptr else dummy, f32, a["bmm1"]),
        f32(0.0 if b1_ptr else float(bmm1_scale) * LOG2E),
        gmem_ptr(bmm2_scale if b2_ptr else dummy, f32, a["bmm2"]),
        f32(0.0 if b2_ptr else float(bmm2_scale)),
        gmem_ptr(out_o, cutlass.BFloat16, 16 if o_vec16 else 4),
        I32(out_o.stride(0)),
        I32(out_o.stride(1)),
        I32(out_o.stride(2)),
        gmem_ptr(out_lse, f32, a["lse"]),
        I32(out_lse.stride(0)),
        I32(out_lse.stride(1)),
        I32(out_lse.stride(2)),
        gmem_ptr(ws, f32, a["ws"]),
        gmem_ptr(cnt[b], cutlass.Int64, a["cnt"]),
        I32(b),
        stream,
    )

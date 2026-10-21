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
"""Host side of the Qwen3.5 fused split-KV decode attention (CuTe DSL kernel
``cute_dsl_kernels/blackwell/q35_decode_attn.py``): support check, compile cache, workspace and launch.

One launch replaces the segmented chain (prep + trtllm-gen FMHA + LSE + merge + output gate) for uniform-q_len
generation batches of one KV head per rank (TRTLLM_Q35_ATTN_FUSED_DECODE=1, default off).
"""

import atexit
import functools
import os
import threading
from typing import Dict, Optional, Union

import torch

from tensorrt_llm._torch.q35_env import register as _q35_register
from tensorrt_llm._utils import prefer_pinned

# Grid splits per request: SMs // batch (one CTA per SM, all CTAs co-resident for the in-kernel grid barrier).
_SPLITS_OVERRIDE = int(os.environ.get("TRTLLM_Q35_ATTN_FUSED_SPLITS", "0"))


@_q35_register
def _q35_read_switches() -> None:
    """This module's TRTLLM_Q35_* switches; q35_env.activate() re-reads them."""
    global _TRIGGER
    # Where the grid releases its dependents (o_proj): "entry" (safe with the in-kernel grid barrier: the dependent
    # grid launches only after every CTA of this grid ran the trigger, i.e. after all of them are resident) or "end".
    # Follows TRTLLM_Q35_PDL_EARLY_TRIGGER unless TRTLLM_Q35_ATTN_FUSED_TRIGGER is set.
    _TRIGGER = os.environ.get(
        "TRTLLM_Q35_ATTN_FUSED_TRIGGER",
        "entry" if os.environ.get("TRTLLM_Q35_PDL_EARLY_TRIGGER", "0") == "1" else "end",
    )


_KV_PRE_WAIT = os.environ.get("TRTLLM_Q35_ATTN_FUSED_KV_PRE_WAIT", "1") == "1"
_STAGES = int(os.environ.get("TRTLLM_Q35_ATTN_FUSED_STAGES", "2"))
# Pages below the new tokens that are still read only after griddepcontrol.wait (default 1).
_FRESH_MARGIN = int(os.environ.get("TRTLLM_Q35_ATTN_FUSED_FRESH_MARGIN", "1"))
# E4M3 P shift (log2 units): P = exp2(x - m + shift), so the row's reference max maps to 2^shift instead of 1 and small
# probabilities neither flush to 0 (below 2^-10 of the max with shift 0) nor fall into E4M3's coarse subnormals
# (trtllm-gen scales P by 448 for the same reason). Shift 0 (unshifted P): with an attention sink 5-10x the error of
# trtllm-gen at short contexts. 6 matches trtllm-gen's error; 8 costs up to +4 us/call in rescales.
_P_SHIFT = float(os.environ.get("TRTLLM_Q35_ATTN_FUSED_P_SHIFT", "6"))
# SMs left out of the grid (default 0): keeps the in-kernel grid barrier co-resident when other kernels hold SMs.
_SM_RESERVE = int(os.environ.get("TRTLLM_Q35_ATTN_FUSED_SM_RESERVE", "0"))
# In-model diagnostics (default off), as TRTLLM_Q35_ATTN_CP_FUSED_DBG_SLOTS: per-launch / per-CTA stamps into a ring of
# the last n launches, saved after every burst of launches and at exit to
# TRTLLM_Q35_ATTN_FUSED_DBG_DIR/q35_fused_decode_dbg_r<rank>_p<pid>_{d<k>,final}.pt (start_debug_ring_dump).
_DBG_SLOTS = int(os.environ.get("TRTLLM_Q35_ATTN_FUSED_DBG_SLOTS", "0"))
_DBG_DIR = os.environ.get("TRTLLM_Q35_ATTN_FUSED_DBG_DIR", ".")
_DBG_RING_CTAS = 256
_DBG_RINGS: Dict[int, torch.Tensor] = {}


_RELAX_CAPTURE = True  # tests turn it off for a negative control


def _relax_thread_capture_mode() -> bool:
    """Put the calling thread in relaxed stream-capture mode (its CUDA calls then never invalidate a global-mode capture
    running on another thread). False if unavailable."""
    if not _RELAX_CAPTURE:
        return True
    try:
        try:
            from cuda.bindings import driver as cu
        except ImportError:
            from cuda import cuda as cu
        err, _ = cu.cuThreadExchangeStreamCaptureMode(
            cu.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_RELAXED
        )
        return err == cu.CUresult.CUDA_SUCCESS
    except Exception:  # noqa: BLE001 (diagnostics only)
        return False


def start_debug_ring_dump(
    ring: torch.Tensor, device: torch.device, path_prefix: str, slots: int, ctas: int, name: str
) -> None:
    """Save a kernel debug ring ([slots, ctas, 8] int64, field 6 = launch index * 64 + batch) to
    <path_prefix>_d<k>.pt at the end of every burst of launches (no new launch for one TRTLLM_Q35_ATTN_DBG_PERIOD_S
    period, default 5 s: e.g. one decode phase of a benchmark) and to <path_prefix>_final.pt at exit, from a side thread
    / stream. The ring may be an inference tensor (allocated under torch.inference_mode), so it is read in inference
    mode. The dump thread runs in relaxed stream-capture mode: PyTorch captures CUDA graphs in global mode, where a
    CUDA call from any other thread invalidates the capture (a side-stream copy during the warmup captures can hang
    the process); the pinned buffer is allocated here, on the calling (eager) thread. Diagnostics only: never raises."""
    period = float(os.environ.get("TRTLLM_Q35_ATTN_DBG_PERIOD_S", "5"))
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    side = torch.cuda.Stream(device=device)
    with torch.inference_mode():
        host = torch.empty(ring.shape, dtype=ring.dtype, pin_memory=prefer_pinned())
    st = dict(host=host, last=None, dirty=False, k=0)
    lock = threading.Lock()

    def snap():
        with torch.inference_mode():
            with torch.cuda.stream(side):
                st["host"].copy_(ring, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(side)
            ev.synchronize()
            v = st["host"].view(slots, ctas, 8)
            return int(v[:, :, 6].max()), v

    def save(v, tag):
        # clone outside inference mode: a normal tensor (the pinned buffer is reused by the next snapshot)
        path = f"{path_prefix}_{tag}.pt"
        torch.save(dict(ring=v.clone(), sms=sms, slots=slots, ctas=ctas), path + ".tmp")
        os.replace(path + ".tmp", path)

    def tick(final=False):
        with lock:
            try:
                mx, v = snap()
                if final:
                    if mx > 0:
                        save(v, "final")
                elif mx != st["last"]:
                    st["last"], st["dirty"] = mx, True
                elif st["dirty"] and mx > 0:
                    st["k"] += 1
                    save(v, f"d{st['k']:02d}")
                    st["dirty"] = False
            except Exception as exc:  # noqa: BLE001 (diagnostics only: never break the run)
                print(f"[{name}] debug ring dump failed: {exc}", flush=True)

    stop = threading.Event()

    def loop():
        torch.cuda.set_device(device)
        if not _relax_thread_capture_mode():
            print(
                f"[{name}] debug ring: relaxed capture mode unavailable; only the exit snapshot is saved",
                flush=True,
            )
            return
        while not stop.wait(period):
            tick()

    th = threading.Thread(target=loop, daemon=True)
    th.start()

    def finish():
        # stop the thread before interpreter finalization (a daemon thread inside a torch op at that point aborts
        # the process), then take the last snapshot from this thread
        stop.set()
        th.join(timeout=30)
        tick(final=True)

    atexit.register(finish)


def _dbg_ring(device: torch.device) -> torch.Tensor:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    ring = _DBG_RINGS.get(idx)
    if ring is None:
        assert not torch.cuda.is_current_stream_capturing(), (
            "allocate the debug ring with an eager call first"
        )
        ring = torch.zeros(_DBG_SLOTS * _DBG_RING_CTAS * 8, dtype=torch.int64, device=device)
        _DBG_RINGS[idx] = ring
        rank = next(
            (
                os.environ[k]
                for k in ("SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "RANK")
                if k in os.environ
            ),
            "x",
        )
        start_debug_ring_dump(
            ring,
            device,
            os.path.join(_DBG_DIR, f"q35_fused_decode_dbg_r{rank}_p{os.getpid()}"),
            _DBG_SLOTS,
            _DBG_RING_CTAS,
            "q35_fused_decode",
        )
    return ring


PAGE = 64
HEAD_DIM = 256
MAX_ROWS = 32
DEBUG_COLS = 24  # int64 stamps per CTA of the debug_ts test hook (Q35DecodeAttnKernel DBG_COLS)


def q35_fused_decode_supported(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_len: int,
    batch: int,
    window_left: int,
    has_sinks: bool,
    tokens_per_block: int,
    q_dtype: torch.dtype,
    out_dtype: torch.dtype,
    kv_dtype: Optional[torch.dtype],
    sm_count: int,
) -> bool:
    rows = q_len * (num_heads // max(num_kv_heads, 1))
    return (
        num_kv_heads == 1
        and head_dim == HEAD_DIM
        and tokens_per_block == PAGE
        and 1 <= q_len <= 16
        and 1 <= rows <= MAX_ROWS
        and window_left < 0
        and not has_sinks
        and q_dtype == torch.float8_e4m3fn
        and (kv_dtype is None or kv_dtype in (torch.float8_e4m3fn, torch.uint8))
        and out_dtype == torch.bfloat16
        and batch >= 1
        and _num_splits(batch, sm_count, rows) >= rows
    )


def _num_splits(batch: int, sm_count: int, rows: int) -> int:
    if _SPLITS_OVERRIDE > 0:
        return _SPLITS_OVERRIDE
    return max((sm_count - _SM_RESERVE) // batch, 1)


def _ws_req_floats(rows: int, n_splits: int) -> int:
    """Workspace floats per request (mirrors Q35DecodeAttnKernel: chunk-major O partials + row-major (m, l))."""
    total = rows * HEAD_DIM
    chunk = 4 * ((total + 4 * n_splits - 1) // (4 * n_splits))
    chunk_pad = chunk + (4 if (chunk // 4) % 2 == 0 else 0)
    n_chunks = (total + chunk - 1) // chunk
    return n_chunks * n_splits * chunk_pad + rows * n_splits * 4


def workspace_bytes(batch: int, sm_count: int, rows: int) -> int:
    return batch * _ws_req_floats(rows, _num_splits(batch, sm_count, rows)) * 4


def q35_fused_workspace_floats(sm_count: int) -> int:
    """Workspace large enough for every supported call (batch * splits <= SMs, splits <= 256, <= 32 rows)."""
    return sm_count * (MAX_ROWS * HEAD_DIM + 4 * MAX_ROWS + 4 * 256 + 256) + 64


class _GraphSafeDlpack:
    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor

    def __dlpack__(self, stream=None):
        return self._tensor.__dlpack__(stream=-1)

    def __dlpack_device__(self):
        return self._tensor.__dlpack_device__()


# Declared alignment (bytes) of each pointer argument, used at compile and at call time (make_ptr checks the address).
# q and the KV pool are TMA sources and the workspace takes 16-byte stores / bulk copies; everything else is accessed
# element by element, so only the element size is assumed (callers hand over views such as bmm2 = scales[2:3]).
_ALIGN = dict(
    q=16,
    kv=16,
    tab=4,
    seq=4,
    bmm1=4,
    bmm2=4,
    gate=2,
    out=2,
    ws=16,
    cnt=8,
    dbg=8,
    qkv=4,
    nw=4,
    cs=4,
    pos=4,
    kvs=4,
)


def _ptr(t: torch.Tensor, dtype, align: int):
    import cutlass.cute as cute

    return cute.runtime.make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


@functools.lru_cache(maxsize=None)
def _compiled(
    q_len: int,
    group: int,
    has_gate: bool,
    bmm1_from_ptr: bool,
    bmm2_from_ptr: bool,
    use_pdl: bool,
    trigger: str,
    kv_pre_wait: bool,
    debug_ts: bool,
    unsafe_skip_wait: bool,
    stages: int,
    n_splits: int,
    fresh_margin: int,
    p_shift: float = 0.0,
    fold: bool = False,
    use_mrope: bool = False,
    mrope_s1: int = 0,
    mrope_s2: int = 0,
    pos_i64: bool = False,
    debug_ring: int = 0,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from ...cute_dsl_kernels.blackwell.q35_decode_attn import Q35DecodeAttnKernel

    kernel = Q35DecodeAttnKernel(
        q_len=q_len,
        group=group,
        has_gate=has_gate,
        bmm1_from_ptr=bmm1_from_ptr,
        bmm2_from_ptr=bmm2_from_ptr,
        use_pdl=use_pdl,
        trigger=trigger,
        kv_pre_wait=kv_pre_wait,
        debug_ts=debug_ts,
        unsafe_skip_wait=unsafe_skip_wait,
        stages=stages,
        n_splits=n_splits,
        fresh_margin_pages=fresh_margin,
        p_shift=p_shift,
        fold=fold,
        use_mrope=use_mrope,
        mrope_s1=mrope_s1,
        mrope_s2=mrope_s2,
        pos_i64=pos_i64,
        debug_ring=debug_ring,
    )
    dev = torch.device("cuda")
    f32 = torch.empty(
        max(64, _ws_req_floats(q_len * group, n_splits)), dtype=torch.float32, device=dev
    )
    i32 = torch.zeros(64, dtype=torch.int32, device=dev)
    u8 = torch.empty(64 * HEAD_DIM, dtype=torch.uint8, device=dev)
    bf = torch.empty(64, dtype=torch.bfloat16, device=dev)
    i64 = torch.zeros(16, dtype=torch.int64, device=dev)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fp8 = cutlass.Float8E4M3FN
    a = _ALIGN
    return cute.compile(
        kernel,
        _ptr(u8, fp8, a["q"]),
        _ptr(u8, fp8, a["kv"]),
        cutlass.Int32(64),
        _ptr(i32, cutlass.Int32, a["tab"]),
        cutlass.Int32(2),
        cutlass.Int32(1),
        _ptr(i32, cutlass.Int32, a["seq"]),
        _ptr(f32, cutlass.Float32, a["bmm1"]),
        cutlass.Float32(1.0),
        _ptr(f32, cutlass.Float32, a["bmm2"]),
        cutlass.Float32(1.0),
        _ptr(bf, cutlass.BFloat16, a["gate"]),
        cutlass.Int32(HEAD_DIM),
        _ptr(bf, cutlass.BFloat16, a["out"]),
        cutlass.Int32(HEAD_DIM),
        _ptr(f32, cutlass.Float32, a["ws"]),
        _ptr(i64, cutlass.Int64, a["cnt"]),
        _ptr(i64, cutlass.Int64, a["dbg"]),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(0),
        *_fold_args(bf, bf, bf, f32, i64 if pos_i64 else i32, f32, f32, 0, 0, 0.0, 0.0),
        stream,
    )


def _fold_args(
    qkv,
    qw,
    kw,
    cos_sin,
    positions,
    kvq,
    kvd,
    qkv_stride: int,
    max_pos: int,
    host_bmm1: float,
    eps: float,
):
    """Trailing kernel arguments of the fold mode (dummies on the stock path)."""
    import cutlass

    a = _ALIGN
    pos_t = cutlass.Int64 if positions.dtype == torch.int64 else cutlass.Int32
    return (
        _ptr(qkv, cutlass.BFloat16, a["qkv"]),
        cutlass.Int32(qkv_stride),
        _ptr(qw, cutlass.BFloat16, a["nw"]),
        _ptr(kw, cutlass.BFloat16, a["nw"]),
        _ptr(cos_sin, cutlass.Float32, a["cs"]),
        cutlass.Int32(max_pos),
        _ptr(positions, pos_t, a["pos"]),
        _ptr(kvq, cutlass.Float32, a["kvs"]),
        _ptr(kvd, cutlass.Float32, a["kvs"]),
        cutlass.Float32(host_bmm1),
        cutlass.Float32(eps),
    )


def q35_fused_decode(
    q: torch.Tensor,
    kv_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    q_len: int,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    out: torch.Tensor,
    workspace: torch.Tensor,
    counters: torch.Tensor,
    sm_count: int,
    enable_pdl: bool,
    gate: Optional[torch.Tensor] = None,
    debug: Optional[torch.Tensor] = None,
    unsafe_skip_wait: bool = False,
    n_splits: Optional[int] = None,
    fresh_margin: Optional[int] = None,
    p_shift: Optional[float] = None,
) -> None:
    """Fused decode attention; writes ``out`` [b * q_len, Hq, D] (bf16, optionally times sigmoid(gate)).

    q: [b * q_len, Hq, D] E4M3 contiguous (one KV head per rank, Hq = group); kv_pool: [blocks, 1, 64, 256] E4M3;
    block_tables: [b, 2, max_pages] int32 K / V block indices (stride(-1) == 1); seq_lens: [b] int32 KV lengths
    including the q_len new tokens. bmm1_scale: float (natural-log domain QK scale) or the trtllm-gen [raw, log2]
    device tensor; bmm2_scale: float or a 1-element device tensor. workspace: >= workspace_bytes() bytes;
    counters: int64 [>= b + 1, >= b], zero-initialized once: grid-barrier epoch counters, row b serves batch b (each
    row always sees the same split count), never reset. gate: [b * q_len, Hq * D] (or [b * q_len, Hq, D]) bf16
    with unit stride along D.
    """
    import cuda.bindings.driver as cuda
    import cutlass

    num_tokens, num_heads, head_dim = q.shape
    b = num_tokens // q_len
    group = num_heads
    rows = q_len * group
    assert head_dim == HEAD_DIM and q.dtype == torch.float8_e4m3fn and q.is_contiguous()
    assert (
        kv_pool.dim() == 4
        and kv_pool.shape[1] == 1
        and kv_pool.shape[2] == PAGE
        and kv_pool.shape[3] == HEAD_DIM
    )
    assert kv_pool.is_contiguous()
    assert block_tables.dim() == 3 and block_tables.shape[1] == 2 and block_tables.stride(-1) == 1
    assert block_tables.dtype == torch.int32 and seq_lens.dtype == torch.int32
    assert out.dtype == torch.bfloat16 and out.stride(-1) == 1
    out2 = out.view(num_tokens, num_heads * HEAD_DIM) if out.dim() == 3 else out
    assert out2.stride(1) == 1
    splits = n_splits if n_splits else _num_splits(b, sm_count, rows)
    assert splits * b <= sm_count, (
        "the in-kernel grid barrier needs every CTA co-resident (one per SM)"
    )
    assert splits >= rows, "the combine phase assumes at most two rows per CTA"
    need = b * _ws_req_floats(rows, splits) * 4
    assert workspace.numel() * workspace.element_size() >= need, "fused decode workspace too small"
    assert (
        counters.dtype == torch.int64
        and counters.dim() == 2
        and counters.shape[0] > b
        and counters.shape[1] >= b
    )
    assert b <= 8
    assert counters.is_contiguous()
    assert debug is None or (
        debug.dim() == 2 and debug.shape[1] >= DEBUG_COLS and debug.is_contiguous()
    )
    cnt_row = counters[b]
    has_gate = gate is not None
    if has_gate:
        gate2 = gate.view(gate.shape[0], num_heads * HEAD_DIM) if gate.dim() == 3 else gate
        assert gate2.stride(1) == 1 and gate2.dtype == torch.bfloat16
    b1_ptr = isinstance(bmm1_scale, torch.Tensor)
    b2_ptr = isinstance(bmm2_scale, torch.Tensor)
    if b1_ptr:
        assert bmm1_scale.numel() >= 2 and bmm1_scale.dtype == torch.float32
    assert splits <= 256, "the combine reads at most 256 splits"
    ring = _dbg_ring(q.device) if (_DBG_SLOTS > 0 and debug is None) else None
    fn = _compiled(
        q_len,
        group,
        has_gate,
        b1_ptr,
        b2_ptr,
        bool(enable_pdl),
        _TRIGGER,
        _KV_PRE_WAIT,
        debug is not None,
        unsafe_skip_wait,
        _STAGES,
        splits,
        _FRESH_MARGIN if fresh_margin is None else fresh_margin,
        _P_SHIFT if p_shift is None else float(p_shift),
        debug_ring=_DBG_SLOTS if ring is not None else 0,
    )
    fp8 = cutlass.Float8E4M3FN
    f32 = cutlass.Float32
    dummy_f = workspace.view(torch.float32)
    dummy_bf = workspace.view(torch.bfloat16)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    a = _ALIGN
    fn(
        _ptr(q, fp8, a["q"]),
        _ptr(kv_pool, fp8, a["kv"]),
        cutlass.Int32(kv_pool.shape[0] * PAGE),
        _ptr(block_tables, cutlass.Int32, a["tab"]),
        cutlass.Int32(block_tables.stride(0)),
        cutlass.Int32(block_tables.stride(1)),
        _ptr(seq_lens, cutlass.Int32, a["seq"]),
        _ptr(bmm1_scale if b1_ptr else dummy_f, f32, a["bmm1"]),
        f32(0.0 if b1_ptr else float(bmm1_scale) * 1.4426950408889634),
        _ptr(bmm2_scale if b2_ptr else dummy_f, f32, a["bmm2"]),
        f32(0.0 if b2_ptr else float(bmm2_scale)),
        _ptr(gate2 if has_gate else out2, cutlass.BFloat16, a["gate"]),
        cutlass.Int32(gate2.stride(0) if has_gate else out2.stride(0)),
        _ptr(out2, cutlass.BFloat16, a["out"]),
        cutlass.Int32(out2.stride(0)),
        _ptr(workspace, f32, a["ws"]),
        _ptr(cnt_row, cutlass.Int64, a["cnt"]),
        _ptr(
            debug if debug is not None else (ring if ring is not None else cnt_row),
            cutlass.Int64,
            a["dbg"],
        ),
        cutlass.Int32(b),
        cutlass.Int32(splits),
        cutlass.Int32(0),
        *_fold_args(
            dummy_bf, dummy_bf, dummy_bf, dummy_f, seq_lens, dummy_f, dummy_f, 0, 0, 0.0, 0.0
        ),
        stream,
    )


# fold mode (TRTLLM_Q35_ATTN_FOLD): the rotary dimension the kernel's lane layout covers (Qwen3.5: 0.25 * 256).
FOLD_ROTARY_DIM = 64


def q35_fused_decode_fold_supported(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    q_len: int,
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
) -> bool:
    """Shape / layout conditions of q35_fused_decode_fold on top of q35_fused_decode_supported."""
    t = qkv.shape[0] if qkv.dim() == 2 else -1
    return (
        num_kv_heads == 1
        and head_dim == HEAD_DIM
        and rotary_dim == FOLD_ROTARY_DIM
        and qkv.dim() == 2
        and qkv.dtype == torch.bfloat16
        and qkv.shape[1] == (2 * num_heads + 2 * num_kv_heads) * head_dim
        and qkv.stride(1) == 1
        and qkv.stride(0) % 2 == 0
        and qkv.data_ptr() % 4 == 0
        and t > 0
        and t % q_len == 0
        and positions.dtype in (torch.int32, torch.int64)
        and positions.is_contiguous()
        and positions.numel() in (t, 3 * t)
        and cos_sin.dtype == torch.float32
        and cos_sin.is_contiguous()
        and cos_sin.dim() == 3
        and tuple(cos_sin.shape[1:]) == (2, rotary_dim // 2)
    )


def q35_fused_decode_fold(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    rotary_dim: int,
    mrope_section,
    kv_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    q_len: int,
    kv_scale_orig_quant: torch.Tensor,
    kv_scale_quant_orig: torch.Tensor,
    host_bmm1_scale: float,
    out: torch.Tensor,
    workspace: torch.Tensor,
    counters: torch.Tensor,
    sm_count: int,
    enable_pdl: bool,
    debug: Optional[torch.Tensor] = None,
    unsafe_skip_wait: bool = False,
    n_splits: Optional[int] = None,
    fresh_margin: Optional[int] = None,
    p_shift: Optional[float] = None,
) -> None:
    """Fused decode attention with the qk-norm / RoPE / gate / KV-append prologue folded in (TRTLLM_Q35_ATTN_FOLD).

    Same result as ``q35_fused_qkv_rope_gate_append`` (fused_qk_norm_rope_gate.py) followed by ``q35_fused_decode``
    with its FP8 query, scales and gate, bit for bit, in one launch that reads the raw QKV projection:
    qkv [b * q_len, (2 * Hq + 2) * 256] bf16 ((q, gate) per q head, then K, V of the one KV head); q_weight /
    k_weight [256] bf16 Gemma RMSNorm weights (scale by 1 + w); cos_sin [max_pos, 2, rotary_dim / 2] fp32 (neox);
    positions [T] or, for interleaved mrope (mrope_section = (s0, s1, s2)), [3, T], int32 / int64; kv scales: the
    FP8 KV cache's per-tensor [1] fp32 tensors; host_bmm1_scale: the softmax scale. The new tokens' FP8 K / V go to
    position seq_lens[r] - q_len + t of request r in kv_pool / block_tables (as the Triton kernel); ``out``
    [b * q_len, Hq, 256] bf16 receives the attention output times sigmoid(gate). Other arguments as
    ``q35_fused_decode``.
    """
    import cuda.bindings.driver as cuda
    import cutlass

    num_tokens = qkv.shape[0]
    num_heads = (qkv.shape[1] - 2 * HEAD_DIM) // (2 * HEAD_DIM)
    b = num_tokens // q_len
    rows = q_len * num_heads
    use_mrope = positions.numel() == 3 * num_tokens and mrope_section is not None
    assert q35_fused_decode_fold_supported(
        num_heads, 1, HEAD_DIM, rotary_dim, q_len, qkv, positions, cos_sin
    )
    assert q_weight.dtype == torch.bfloat16 and k_weight.dtype == torch.bfloat16
    assert q_weight.numel() == HEAD_DIM and k_weight.numel() == HEAD_DIM
    assert q_weight.is_contiguous() and k_weight.is_contiguous()
    assert kv_scale_orig_quant.dtype == torch.float32 and kv_scale_quant_orig.dtype == torch.float32
    assert (
        kv_pool.dim() == 4
        and kv_pool.shape[1] == 1
        and kv_pool.shape[2] == PAGE
        and kv_pool.shape[3] == HEAD_DIM
    )
    assert kv_pool.is_contiguous() and kv_pool.dtype == torch.float8_e4m3fn
    assert block_tables.dim() == 3 and block_tables.shape[1] == 2 and block_tables.stride(-1) == 1
    assert block_tables.dtype == torch.int32 and seq_lens.dtype == torch.int32
    assert out.dtype == torch.bfloat16 and out.stride(-1) == 1
    out2 = out.view(num_tokens, num_heads * HEAD_DIM) if out.dim() == 3 else out
    assert out2.stride(1) == 1
    splits = n_splits if n_splits else _num_splits(b, sm_count, rows)
    assert splits * b <= sm_count, (
        "the in-kernel grid barrier needs every CTA co-resident (one per SM)"
    )
    assert splits >= rows, "the combine phase assumes at most two rows per CTA"
    assert workspace.numel() * workspace.element_size() >= b * _ws_req_floats(rows, splits) * 4
    assert (
        counters.dtype == torch.int64
        and counters.dim() == 2
        and counters.shape[0] > b
        and counters.shape[1] >= b
    )
    assert b <= 8 and counters.is_contiguous() and splits <= 256
    assert debug is None or (
        debug.dim() == 2 and debug.shape[1] >= DEBUG_COLS and debug.is_contiguous()
    )
    cnt_row = counters[b]
    ring = _dbg_ring(qkv.device) if (_DBG_SLOTS > 0 and debug is None) else None
    fn = _compiled(
        q_len,
        num_heads,
        True,
        False,
        False,
        bool(enable_pdl),
        _TRIGGER,
        _KV_PRE_WAIT,
        debug is not None,
        unsafe_skip_wait,
        _STAGES,
        splits,
        _FRESH_MARGIN if fresh_margin is None else fresh_margin,
        _P_SHIFT if p_shift is None else float(p_shift),
        True,
        use_mrope,
        int(mrope_section[1]) if use_mrope else 0,
        int(mrope_section[2]) if use_mrope else 0,
        positions.dtype == torch.int64,
        debug_ring=_DBG_SLOTS if ring is not None else 0,
    )
    fp8 = cutlass.Float8E4M3FN
    f32 = cutlass.Float32
    dummy_f = workspace.view(torch.float32)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    a = _ALIGN
    fn(
        _ptr(workspace, fp8, a["q"]),  # q: computed in the kernel
        _ptr(kv_pool, fp8, a["kv"]),
        cutlass.Int32(kv_pool.shape[0] * PAGE),
        _ptr(block_tables, cutlass.Int32, a["tab"]),
        cutlass.Int32(block_tables.stride(0)),
        cutlass.Int32(block_tables.stride(1)),
        _ptr(seq_lens, cutlass.Int32, a["seq"]),
        _ptr(dummy_f, f32, a["bmm1"]),
        f32(0.0),
        _ptr(dummy_f, f32, a["bmm2"]),
        f32(0.0),
        _ptr(out2, cutlass.BFloat16, a["gate"]),  # gate: read from qkv in the kernel
        cutlass.Int32(0),
        _ptr(out2, cutlass.BFloat16, a["out"]),
        cutlass.Int32(out2.stride(0)),
        _ptr(workspace, f32, a["ws"]),
        _ptr(cnt_row, cutlass.Int64, a["cnt"]),
        _ptr(
            debug if debug is not None else (ring if ring is not None else cnt_row),
            cutlass.Int64,
            a["dbg"],
        ),
        cutlass.Int32(b),
        cutlass.Int32(splits),
        cutlass.Int32(0),
        *_fold_args(
            qkv,
            q_weight,
            k_weight,
            cos_sin,
            positions,
            kv_scale_orig_quant,
            kv_scale_quant_orig,
            qkv.stride(0),
            cos_sin.shape[0],
            float(host_bmm1_scale),
            float(eps),
        ),
        stream,
    )

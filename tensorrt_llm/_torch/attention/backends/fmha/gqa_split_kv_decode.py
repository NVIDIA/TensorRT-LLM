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
"""Host side of the fused split-KV decode attention (``attention/kernels/blackwell/gqa_split_kv_decode_attn.py``).

One launch for uniform-q_len generation batches of one KV head per rank, output gate included.
"""

import functools
from typing import Dict, Optional, Tuple, Union

import torch

from ....cute_dsl_utils import gmem_ptr

# Under PDL the grid releases its dependents (o_proj) at entry; safe with the in-kernel grid barrier because the
# dependent grid launches only after every CTA ran the trigger, i.e. all are resident.

# K / V ring stages of the TMA producer.
STAGES = 2
# Pages below the new tokens that are still read only after griddepcontrol.wait.
FRESH_MARGIN = 1
# E4M3 P shift (log2 units): P = exp2(x - m + shift) keeps small probabilities out of E4M3's flush-to-zero and coarse
# subnormal range (trtllm-gen scales P by 448 for the same reason). 6 matches trtllm-gen's error.
P_SHIFT = 6.0

PAGE = 64
HEAD_DIM = 256
MAX_ROWS = 32
# Largest batch of a call: one counter row per batch size.
MAX_BATCH = 4


def split_kv_decode_supported(
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
        and 1 <= batch <= MAX_BATCH
        and _num_splits(batch, sm_count, rows) >= rows
    )


def _num_splits(batch: int, sm_count: int, rows: int) -> int:
    """Grid splits per request: one CTA per SM, all CTAs co-resident for the in-kernel grid barrier."""
    del rows  # the split count depends on the batch only (one counter row per batch size)
    return max(sm_count // batch, 1)


def _ws_req_floats(rows: int, n_splits: int) -> int:
    """Workspace floats per request (mirrors GqaSplitKvDecodeAttnKernel: chunk-major O partials + row-major (m, l))."""
    total = rows * HEAD_DIM
    chunk = 4 * ((total + 4 * n_splits - 1) // (4 * n_splits))
    chunk_pad = chunk + (4 if (chunk // 4) % 2 == 0 else 0)
    n_chunks = (total + chunk - 1) // chunk
    return n_chunks * n_splits * chunk_pad + rows * n_splits * 4


def split_kv_decode_workspace_floats(sm_count: int) -> int:
    """Workspace large enough for every supported call (batch * splits <= SMs, splits <= 256, <= 32 rows)."""
    return sm_count * (MAX_ROWS * HEAD_DIM + 4 * MAX_ROWS + 4 * 256 + 256) + 64


# device index -> (workspace, counters), shared by every layer: the calls are serialized on the decode stream.
_BUFFERS: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}


def split_kv_decode_buffers(
    device: torch.device, sm_count: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """This device's workspace and grid-barrier counters, allocated by the first call (which must precede CUDA graph
    capture) and never reallocated. Shared by every layer, like the attention workspace: the layers' calls are
    ordered on one stream. Counter row b serves batch size b; zeroed once, never reset (the kernel keeps them
    consistent)."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    bufs = _BUFFERS.get(idx)
    if bufs is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "The split-KV decode workspace must be allocated before CUDA graph capture."
            )
        device = torch.device("cuda", idx)
        ws = torch.zeros(
            split_kv_decode_workspace_floats(sm_count), dtype=torch.float32, device=device
        )
        cnt = torch.zeros(MAX_BATCH + 1, 8, dtype=torch.int64, device=device)
        bufs = (ws, cnt)
        _BUFFERS[idx] = bufs
    return bufs


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
    qkv=4,
    nw=4,
    cs=4,
    pos=4,
    kvs=4,
)


@functools.lru_cache(maxsize=None)
def _compiled(
    q_len: int,
    group: int,
    has_gate: bool,
    bmm1_from_ptr: bool,
    bmm2_from_ptr: bool,
    use_pdl: bool,
    n_splits: int,
    fold: bool = False,
    use_mrope: bool = False,
    mrope_s1: int = 0,
    mrope_s2: int = 0,
    pos_i64: bool = False,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from ...kernels.blackwell.gqa_split_kv_decode_attn import GqaSplitKvDecodeAttnKernel

    kernel = GqaSplitKvDecodeAttnKernel(
        q_len=q_len,
        group=group,
        has_gate=has_gate,
        bmm1_from_ptr=bmm1_from_ptr,
        bmm2_from_ptr=bmm2_from_ptr,
        use_pdl=use_pdl,
        stages=STAGES,
        n_splits=n_splits,
        fresh_margin_pages=FRESH_MARGIN,
        p_shift=P_SHIFT,
        fold=fold,
        use_mrope=use_mrope,
        mrope_s1=mrope_s1,
        mrope_s2=mrope_s2,
        pos_i64=pos_i64,
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
        gmem_ptr(u8, fp8, a["q"]),
        gmem_ptr(u8, fp8, a["kv"]),
        cutlass.Int32(64),
        gmem_ptr(i32, cutlass.Int32, a["tab"]),
        cutlass.Int32(2),
        cutlass.Int32(1),
        gmem_ptr(i32, cutlass.Int32, a["seq"]),
        gmem_ptr(f32, cutlass.Float32, a["bmm1"]),
        cutlass.Float32(1.0),
        gmem_ptr(f32, cutlass.Float32, a["bmm2"]),
        cutlass.Float32(1.0),
        gmem_ptr(bf, cutlass.BFloat16, a["gate"]),
        cutlass.Int32(HEAD_DIM),
        gmem_ptr(bf, cutlass.BFloat16, a["out"]),
        cutlass.Int32(HEAD_DIM),
        gmem_ptr(f32, cutlass.Float32, a["ws"]),
        gmem_ptr(i64, cutlass.Int64, a["cnt"]),
        cutlass.Int32(1),
        cutlass.Int32(1),
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
    """Trailing kernel arguments of the fold mode (dummies on the unfolded path)."""
    import cutlass

    a = _ALIGN
    pos_t = cutlass.Int64 if positions.dtype == torch.int64 else cutlass.Int32
    return (
        gmem_ptr(qkv, cutlass.BFloat16, a["qkv"]),
        cutlass.Int32(qkv_stride),
        gmem_ptr(qw, cutlass.BFloat16, a["nw"]),
        gmem_ptr(kw, cutlass.BFloat16, a["nw"]),
        gmem_ptr(cos_sin, cutlass.Float32, a["cs"]),
        cutlass.Int32(max_pos),
        gmem_ptr(positions, pos_t, a["pos"]),
        gmem_ptr(kvq, cutlass.Float32, a["kvs"]),
        gmem_ptr(kvd, cutlass.Float32, a["kvs"]),
        cutlass.Float32(host_bmm1),
        cutlass.Float32(eps),
    )


def split_kv_decode(
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
) -> None:
    """Fused decode attention; writes ``out`` [b * q_len, Hq, D] bf16, optionally times sigmoid(gate).

    Args:
        q: [b * q_len, Hq, D] E4M3, contiguous (one KV head per rank).
        kv_pool: [blocks, 1, 64, 256] E4M3; block_tables: [b, 2, max_pages] int32 K / V block ids, unit last stride.
        seq_lens: [b] int32 KV lengths including the q_len new tokens.
        bmm1_scale: float (natural-log domain) or the trtllm-gen [raw, log2] tensor; bmm2_scale: float or [1] tensor.
        workspace: >= split_kv_decode_workspace_floats() floats.
        counters: int64 [>= b + 1, >= b] grid-barrier epochs, row b serves batch b; zeroed once, never reset.
        gate: [b * q_len, Hq * D] or [b * q_len, Hq, D] bf16, unit stride along D.
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
    splits = _num_splits(b, sm_count, rows)
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
    assert b <= MAX_BATCH
    assert counters.is_contiguous()
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
    fn = _compiled(q_len, group, has_gate, b1_ptr, b2_ptr, bool(enable_pdl), splits)
    fp8 = cutlass.Float8E4M3FN
    f32 = cutlass.Float32
    dummy_f = workspace.view(torch.float32)
    dummy_bf = workspace.view(torch.bfloat16)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    a = _ALIGN
    fn(
        gmem_ptr(q, fp8, a["q"]),
        gmem_ptr(kv_pool, fp8, a["kv"]),
        cutlass.Int32(kv_pool.shape[0] * PAGE),
        gmem_ptr(block_tables, cutlass.Int32, a["tab"]),
        cutlass.Int32(block_tables.stride(0)),
        cutlass.Int32(block_tables.stride(1)),
        gmem_ptr(seq_lens, cutlass.Int32, a["seq"]),
        gmem_ptr(bmm1_scale if b1_ptr else dummy_f, f32, a["bmm1"]),
        f32(0.0 if b1_ptr else float(bmm1_scale) * 1.4426950408889634),
        gmem_ptr(bmm2_scale if b2_ptr else dummy_f, f32, a["bmm2"]),
        f32(0.0 if b2_ptr else float(bmm2_scale)),
        gmem_ptr(gate2 if has_gate else out2, cutlass.BFloat16, a["gate"]),
        cutlass.Int32(gate2.stride(0) if has_gate else out2.stride(0)),
        gmem_ptr(out2, cutlass.BFloat16, a["out"]),
        cutlass.Int32(out2.stride(0)),
        gmem_ptr(workspace, f32, a["ws"]),
        gmem_ptr(cnt_row, cutlass.Int64, a["cnt"]),
        cutlass.Int32(b),
        cutlass.Int32(splits),
        *_fold_args(
            dummy_bf, dummy_bf, dummy_bf, dummy_f, seq_lens, dummy_f, dummy_f, 0, 0, 0.0, 0.0
        ),
        stream,
    )


# fold mode: the rotary dimension the kernel's lane layout covers.
FOLD_ROTARY_DIM = 64


def split_kv_decode_fold_supported(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    q_len: int,
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
) -> bool:
    """Shape / layout conditions of split_kv_decode_fold on top of split_kv_decode_supported."""
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


def split_kv_decode_fold(
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
) -> None:
    """Fused decode attention with the qk-norm / RoPE / gate / KV-append prologue folded in.

    Bit-identical to ``fused_qkv_rope_gate_fp8_append`` followed by ``split_kv_decode`` with the gate.

    Args:
        qkv: [b * q_len, (2 * Hq + 2) * 256] bf16 raw projection ((q, gate) per q head, then K, V).
        q_weight / k_weight: [256] bf16 Gemma RMSNorm weights (scale by 1 + w).
        cos_sin: [max_pos, 2, rotary_dim / 2] fp32 (neox).
        positions: [T], or [3, T] for interleaved mrope, int32 / int64.
        kv_scale_orig_quant / kv_scale_quant_orig: [1] fp32 per-tensor FP8 KV scales.
        Others: as ``split_kv_decode``; the new K / V go to position seq_lens[r] - q_len + t.
    """
    import cuda.bindings.driver as cuda
    import cutlass

    num_tokens = qkv.shape[0]
    num_heads = (qkv.shape[1] - 2 * HEAD_DIM) // (2 * HEAD_DIM)
    b = num_tokens // q_len
    rows = q_len * num_heads
    use_mrope = positions.numel() == 3 * num_tokens and mrope_section is not None
    assert split_kv_decode_fold_supported(
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
    splits = _num_splits(b, sm_count, rows)
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
    assert b <= MAX_BATCH and counters.is_contiguous() and splits <= 256
    cnt_row = counters[b]
    fn = _compiled(
        q_len,
        num_heads,
        True,
        False,
        False,
        bool(enable_pdl),
        splits,
        True,
        use_mrope,
        int(mrope_section[1]) if use_mrope else 0,
        int(mrope_section[2]) if use_mrope else 0,
        positions.dtype == torch.int64,
    )
    fp8 = cutlass.Float8E4M3FN
    f32 = cutlass.Float32
    dummy_f = workspace.view(torch.float32)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    a = _ALIGN
    fn(
        gmem_ptr(workspace, fp8, a["q"]),  # q: computed in the kernel
        gmem_ptr(kv_pool, fp8, a["kv"]),
        cutlass.Int32(kv_pool.shape[0] * PAGE),
        gmem_ptr(block_tables, cutlass.Int32, a["tab"]),
        cutlass.Int32(block_tables.stride(0)),
        cutlass.Int32(block_tables.stride(1)),
        gmem_ptr(seq_lens, cutlass.Int32, a["seq"]),
        gmem_ptr(dummy_f, f32, a["bmm1"]),
        f32(0.0),
        gmem_ptr(dummy_f, f32, a["bmm2"]),
        f32(0.0),
        gmem_ptr(out2, cutlass.BFloat16, a["gate"]),  # gate: read from qkv in the kernel
        cutlass.Int32(0),
        gmem_ptr(out2, cutlass.BFloat16, a["out"]),
        cutlass.Int32(out2.stride(0)),
        gmem_ptr(workspace, f32, a["ws"]),
        gmem_ptr(cnt_row, cutlass.Int64, a["cnt"]),
        cutlass.Int32(b),
        cutlass.Int32(splits),
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

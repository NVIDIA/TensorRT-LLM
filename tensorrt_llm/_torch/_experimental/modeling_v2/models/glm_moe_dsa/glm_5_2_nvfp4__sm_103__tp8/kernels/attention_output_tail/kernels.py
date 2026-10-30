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
"""attention_output_tail (the attention_output_tail contract): kernels and launch entry point.

The tail maps the latent attention output to the rank-local o_proj partial::

    v[r, h, :] = bf16(latent[r, h, :] @ w_uv[h].T)  # W_UV, FP32 accumulate
    partial[r] = bf16(v[r].flatten() @ w_o.T)  # local o_proj, FP32 accumulate

Both rounding points match the native ``bmm_out`` + ``F.linear`` sequence.
The default engine (``cute_stream``) runs two attention_output_tail-owned kernels:

* ``_wuv_kernel`` (Triton) computes ``v`` into a small scratch tensor. With
  PDL it is launched as a dependent of the latent's producer, exactly like
  the native cuBLAS kernel it replaces: it loads its ``w_uv`` tile, then
  executes ``griddepcontrol.wait`` before reading the latent or writing.
  It triggers its own dependent at entry.
* ``cute_oproj.stream_oproj`` (CuTe DSL) computes the partial. It is a PDL
  dependent of ``_wuv_kernel``: one persistent CTA per SM streams its rows of
  ``w_o`` into shared memory (a bounded number before
  ``griddepcontrol.wait``, the rest after), runs ``mma.sync`` with FP32
  accumulation, adds the per-warp partials in fixed order and stores BF16
  once. It lets its own PDL consumer launch once all copies are issued; the
  consumer still waits for the whole grid.

``_oproj_kernel`` (Triton) and ``cute_fused`` (one CuTe DSL
kernel for both GEMMs) remain selectable through ``TailConfig`` for
experiments. Every engine is deterministic; rows ``[num_rows, out_rows)`` of
the partial are written as zeros.
"""

import os
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import triton
import triton.language as tl

ROW_TILE = 16
"""Row tile of both kernels; it is also the largest supported row count."""

_PREFETCH_SPLITS = 128


@dataclass(frozen=True)
class TailConfig:
    """Launch configuration of the two tail kernels."""

    wuv_block_v: int = 16
    wuv_warps: int = 4
    oproj_block_n: int = 16
    oproj_block_k: int = 256
    oproj_warps: int = 4
    oproj_stages: int = 4
    prefetch: bool = False
    """Triton engine: prefetch the w_o slab into L2 before waiting (PDL)."""
    pdl_entry: bool = True
    """Launch the first attention_output_tail kernel as a PDL dependent of the latent's producer
    (as the native cuBLAS kernels of this region are). Before its
    ``griddepcontrol.wait`` it touches only immutable weights and its own
    shared memory; every latent read and every global write follows the
    wait. Effective when ``TRTLLM_ENABLE_PDL`` is on."""
    early_rows: int = 16
    """``cute_stream`` engine: ``w_o`` rows per CTA streamed before the wait
    on the W_UV kernel."""
    trigger_dependents: bool = True
    """``cute_stream`` engine: let the PDL-launched consumer of the partial
    (the all-reduce) start its prologue once all ``w_o`` copies are issued,
    as the native cuBLAS kernels do; it still waits for completion."""
    oproj_grid: int = 0
    """``cute_stream`` engine: CTAs of the o_proj kernel (0: one per SM).
    Fewer CTAs leave SMs free for a successor that needs whole SMs; each CTA
    then holds more ``w_o`` rows in shared memory."""
    trigger_late: bool = False
    """``cute_stream`` engine, experiments: with ``trigger_dependents``,
    trigger only after each CTA's MMAs."""
    into_hop_vector_publish: bool = True
    """``attention_output_into_hop`` publish: 16-byte multicast stores when every CTA owns a
    multiple of 8 output columns (False: 32-bit stores only, to compare)."""
    oproj_engine: str = "cute_stream"
    """``"cute_stream"`` (Triton W_UV, then a CuTe DSL o_proj that streams
    w_o to shared memory around its wait on v; default), ``"fused"`` (one
    CuTe DSL kernel; kept for experiments), ``"triton"`` (two
    Triton kernels) or ``"native"`` (``attention_output_tail_prefetch``: the native W_UV BMM and
    o_proj GEMM run unchanged, on weights the ``attention_output_tail_prefetch`` prefetch staged
    into L2; the tail runs the native projection so that the prefetch stream can
    be joined after it)."""
    prefetch_l2: bool = False
    """``attention_output_tail_prefetch``: when the attention of an armed call starts, prefetch this
    layer's ``w_uv`` and ``w_o`` into L2 from a private auxiliary stream
    (see ``prefetch.py``)."""
    successor_prefetch: bool = False
    """``attention_output_tail_prefetch``: also stage the successor's weights (set with
    ``AttnTail.set_successor_weights``, e.g. the MoE router weight that the
    post-attention hop or the native router GEMM loads right after the
    tail) into L2 with the same off-path prefetch. Works with or without
    ``prefetch_l2``."""
    prefetch_layers: str = "all"
    """``attention_output_tail_prefetch``: layers that prefetch: ``"all"``, ``"shared"`` (layers without
    an indexer of their own) or ``"full"`` (layers with an indexer)."""
    native_oproj: str = "cublas"
    """``native`` engine: the o_proj kernel run by ``project``: ``"cublas"``
    (the native GEMM) or ``"ring"`` (``cute_oproj_ring``: ``ring_grid`` CTAs,
    a ``ring_slots``-tile shared-memory ring, ``ring_early`` tiles requested
    before its PDL wait)."""
    ring_grid: int = 96
    ring_slots: int = 3
    ring_early: int = 3
    prefetch_fork: str = "attn"
    """``attention_output_tail_prefetch``: where the prefetch is forked: ``"attn"`` (right before the
    attention backend call, i.e. after the absorption BMM and the RoPE /
    FP8-q / KV-append kernel; the adapter's hook), ``"start"`` (start of
    the generation attention; experiments) or ``"prev"`` (forked ahead by
    the previous layer, right before its hop, via ``AttnTail.fork_ahead``;
    experiments)."""
    prefetch_join: str = "after_oproj"
    """``attention_output_tail_prefetch``: where the caller's stream joins the prefetch stream:
    ``"before_wuv"``, ``"before_oproj"`` or ``"after_oproj"``."""
    prefetch_chunk: int = 4096
    prefetch_ctas: int = 52
    prefetch_threads: int = 32
    prefetch_gap_ns: int = 0


DEFAULT_CONFIG = TailConfig()
TAIL_PREFETCH_STAGED_NATIVE_SCHEDULE = TailConfig(oproj_engine="native", prefetch_l2=True)
"""Experiment: the native W_UV and o_proj on tail weights staged into L2 by
the off-path prefetch."""

TAIL_PREFETCH_SCHEDULE = TailConfig(oproj_grid=128, successor_prefetch=True)
"""``attention_output_tail_prefetch`` with the native neighbours (TRTLLM-Gen FMHA before the tail, the
native AR + router GEMM after it): the attention_output_tail kernels with the o_proj on 128
CTAs, plus the router weight staged into L2 for the router GEMM. See
``tail.tail_prefetch_config`` for the other neighbour sets."""

TAIL_PREFETCH_SPARSE_DECODE_SCHEDULE = TailConfig(successor_prefetch=True)
"""``attention_output_tail_prefetch`` behind ``sparse_mla_decode``'s sparse attention (which releases its PDL
dependents early) with the native successor."""

TAIL_PREFETCH_SPARSE_DECODE_HOP_SCHEDULE = TailConfig(oproj_grid=128, early_rows=32)
"""``attention_output_tail_prefetch`` behind ``sparse_mla_decode`` with the ``post_attention_hop``
hop as successor (staging the router weight hurts next to the hop). ``sparse_mla_decode`` releases
its dependents early enough for each CTA to stream 32 of its 48 ``w_o`` rows before its wait."""

INTO_HOP_SCHEDULE = TailConfig(oproj_grid=128)
"""``attention_output_into_hop``: the attention_output_tail kernels with the o_proj on 128 CTAs (48 output
columns each, so every published row chunk is a whole number of 16-byte
stores, and 24 SMs stay free for the hop's CTAs); when the decoder layer
requests it, the o_proj also publishes its partial into the post_attention_hop's
transport. Used behind the native FMHA (streaming more rows before the
wait loses there)."""

INTO_HOP_SPARSE_DECODE_SCHEDULE = TailConfig(oproj_grid=128, early_rows=32)
"""``attention_output_into_hop`` behind ``sparse_mla_decode``: as ``INTO_HOP_SCHEDULE`` with 32 of each CTA's 48 ``w_o``
rows streamed before the PDL wait (as ``TAIL_PREFETCH_SPARSE_DECODE_HOP_SCHEDULE``)."""
# Shape checks use the coarsest tiles any config may use.
_MAX_BLOCK_C = 512
_MAX_BLOCK_V = 64
_MAX_BLOCK_N = 64
_MAX_BLOCK_K = 512


def pdl_enabled() -> bool:
    """Return whether attention_output_tail may use programmatic dependent launch at all."""
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


def pdl_entry_enabled() -> bool:
    """Return whether the first attention_output_tail kernel is a PDL dependent of its producer
    (whenever PDL is enabled)."""
    return pdl_enabled()


@triton.jit
def _gdc_wait_with_memory_clobber():
    tl.inline_asm_elementwise(
        "griddepcontrol.wait; // dummy $0",
        "=r,~{memory}",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _prefetch_l2(addr, nbytes):
    tl.inline_asm_elementwise(
        "cp.async.bulk.prefetch.L2.global [$1], $2; // dummy $0",
        "=r,l,r",
        [addr, nbytes],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _wuv_kernel(
    latent_ptr,
    w_uv_ptr,
    v_ptr,
    num_rows,
    stride_lat_row,
    stride_lat_head,
    stride_wuv_head,
    stride_wuv_v,
    stride_v_row,
    C: tl.constexpr,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_R: tl.constexpr,
    USE_PDL: tl.constexpr,
    WAIT_PRODUCER: tl.constexpr,
):
    head = tl.program_id(0)
    v_block = tl.program_id(1)
    if USE_PDL:
        # Let the dependent o_proj kernel start its weight stream now; it
        # still waits for this whole grid before reading v.
        tl.extra.cuda.gdc_launch_dependents()

    offs_v = v_block * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_r = tl.arange(0, BLOCK_R)
    offs_c = tl.arange(0, C)
    row_mask = offs_r < num_rows
    # Weights are immutable: load them before waiting for the producer.
    w = tl.load(
        w_uv_ptr + head * stride_wuv_head + offs_v[:, None] * stride_wuv_v + offs_c[None, :]
    )
    if WAIT_PRODUCER:
        # Launched as a PDL dependent of the latent's producer: every read of
        # the latent and every global write follows this wait.
        _gdc_wait_with_memory_clobber()
    lat_t = tl.load(
        latent_ptr + offs_r[None, :] * stride_lat_row + head * stride_lat_head + offs_c[:, None],
        mask=row_mask[None, :],
        other=0.0,
    )
    acc = tl.dot(w, lat_t)
    v_out = acc.to(v_ptr.dtype.element_ty)
    tl.store(
        v_ptr + offs_r[None, :] * stride_v_row + head * V + offs_v[:, None],
        v_out,
        mask=row_mask[None, :],
    )


@triton.jit
def _oproj_kernel(
    v_ptr,
    w_o_ptr,
    w_o_addr,
    out_ptr,
    num_rows,
    out_rows,
    stride_v_row,
    stride_out_row,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
    PREFETCH_SPLITS: tl.constexpr,
    PREFETCH: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    if PREFETCH:
        # The [BLOCK_N, K] slab of w_o is contiguous: split it into equal
        # pieces, one per lane, and pull them into L2 while W_UV runs.
        slab_bytes: tl.constexpr = BLOCK_N * K * 2
        piece: tl.constexpr = slab_bytes // PREFETCH_SPLITS
        offs_p = tl.arange(0, PREFETCH_SPLITS).to(tl.int64)
        addr = w_o_addr + pid.to(tl.int64) * slab_bytes + offs_p * piece
        _prefetch_l2(addr, tl.full([PREFETCH_SPLITS], piece, tl.int32))
    if USE_PDL:
        _gdc_wait_with_memory_clobber()

    offs_r = tl.arange(0, BLOCK_R)
    acc = tl.zeros([BLOCK_N, BLOCK_R], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        w = tl.load(w_o_ptr + offs_n[:, None] * K + offs_k[None, :])
        v_t = tl.load(
            v_ptr + offs_r[None, :] * stride_v_row + offs_k[:, None],
            mask=offs_r[None, :] < num_rows,
            other=0.0,
        )
        acc = tl.dot(w, v_t, acc)
    tl.store(
        out_ptr + offs_r[None, :] * stride_out_row + offs_n[:, None],
        acc.to(out_ptr.dtype.element_ty),
        mask=offs_r[None, :] < out_rows,
    )


def check_weights(w_uv: torch.Tensor, w_o: torch.Tensor) -> Optional[str]:
    """Return why the kernels cannot take these weights, or ``None``.

    Args:
        w_uv: ``[H, V, C]`` BF16 with a unit inner stride.
        w_o: ``[N, H * V]`` contiguous BF16.
    """
    if w_uv.dim() != 3 or w_o.dim() != 2:
        return "unexpected weight rank"
    heads, v_dim, c_dim = w_uv.shape
    n_dim, k_dim = w_o.shape
    if w_uv.dtype != torch.bfloat16 or w_o.dtype != torch.bfloat16:
        return "weight dtype is not bfloat16"
    if not (w_uv.is_cuda and w_o.is_cuda):
        return "weight not on CUDA"
    if k_dim != heads * v_dim:
        return "w_o in_features != heads * v_head_dim"
    if w_uv.stride(2) != 1 or not w_o.is_contiguous():
        return "weights not packed"
    if c_dim & (c_dim - 1) or c_dim > _MAX_BLOCK_C or v_dim % _MAX_BLOCK_V:
        return "W_UV shape not supported by the tiles"
    if k_dim % _MAX_BLOCK_K or n_dim % _MAX_BLOCK_N:
        return "o_proj shape not a tile multiple"
    if (16 * k_dim * 2) % (_PREFETCH_SPLITS * 16):
        return "prefetch slab not a 16-byte multiple per lane"
    if w_o.data_ptr() % 16:
        return "w_o is not 16-byte aligned"
    from .cute_oproj import _sm_count
    from .cute_oproj import check_shapes as check_stream_shapes

    grid = _sm_count(w_o.device.index or 0)
    reason = check_stream_shapes(n_dim, k_dim, grid)
    if reason is not None:
        return reason
    from .cute_fused import check_fused_shapes

    reason = check_fused_shapes(heads, v_dim, c_dim, n_dim, grid)
    if reason is not None:
        return reason
    if not w_uv.is_contiguous():
        return "w_uv is not contiguous"
    return None


def check_tail_shapes(
    latent: torch.Tensor, w_uv: torch.Tensor, w_o: torch.Tensor, num_rows: int, out_rows: int
) -> Optional[str]:
    """Return why the kernels cannot take these tensors, or ``None``.

    Args:
        latent: ``[>= num_rows, H, C]`` or ``[>= num_rows, H * C]`` BF16 with
            a unit inner stride.
        w_uv: ``[H, V, C]`` BF16 with a unit inner stride.
        w_o: ``[N, H * V]`` contiguous BF16.
        num_rows: live rows to compute, at most ``ROW_TILE``.
        out_rows: rows of the partial; rows past ``num_rows`` are zeroed.
    """
    reason = check_weights(w_uv, w_o)
    if reason is not None:
        return reason
    heads, _, c_dim = w_uv.shape
    if latent.dtype != torch.bfloat16 or not latent.is_cuda:
        return "latent is not a CUDA bfloat16 tensor"
    if not 1 <= num_rows <= ROW_TILE or out_rows < num_rows:
        return f"rows {num_rows}/{out_rows} outside [1, {ROW_TILE}]"
    if latent.shape[0] < num_rows:
        return "latent has fewer rows than num_rows"
    if latent.dim() == 2:
        if latent.shape[1] != heads * c_dim or latent.stride(1) != 1:
            return "latent is not [R, H * C] with unit inner stride"
    elif latent.dim() == 3:
        if (
            tuple(latent.shape[1:]) != (heads, c_dim)
            or latent.stride(2) != 1
            or latent.stride(1) != c_dim
        ):
            return "latent is not [R, H, C] with packed heads"
    else:
        return "unexpected latent rank"
    if latent.stride(0) != heads * c_dim or latent.data_ptr() % 16:
        return "latent rows are not packed and 16-byte aligned"
    return None


def launch_wuv(
    latent: torch.Tensor,
    w_uv: torch.Tensor,
    v: torch.Tensor,
    num_rows: int,
    config: TailConfig = DEFAULT_CONFIG,
    trigger_dependents: bool = False,
    wait_producer: bool = False,
) -> None:
    """Launch the W_UV kernel: ``v[:num_rows] = bf16(latent @ w_uv^T)`` per head.

    ``wait_producer`` launches it as a PDL dependent of the previous kernel
    on the stream; it loads ``w_uv`` first and executes
    ``griddepcontrol.wait`` before reading the latent or writing ``v``.
    """
    heads, v_dim, c_dim = w_uv.shape
    if latent.dim() == 2:
        stride_lat_row, stride_lat_head = latent.stride(0), c_dim
    else:
        stride_lat_row, stride_lat_head = latent.stride(0), latent.stride(1)
    _wuv_kernel[(heads, v_dim // config.wuv_block_v)](
        latent,
        w_uv,
        v,
        num_rows,
        stride_lat_row,
        stride_lat_head,
        w_uv.stride(0),
        w_uv.stride(1),
        v.stride(0),
        C=c_dim,
        V=v_dim,
        BLOCK_V=config.wuv_block_v,
        BLOCK_R=ROW_TILE,
        USE_PDL=trigger_dependents,
        WAIT_PRODUCER=wait_producer,
        num_warps=config.wuv_warps,
        num_stages=1,
        **({"launch_pdl": True} if wait_producer else {}),
    )


def launch_oproj(
    v: torch.Tensor,
    w_o: torch.Tensor,
    out: torch.Tensor,
    num_rows: int,
    out_rows: int,
    config: TailConfig = DEFAULT_CONFIG,
    use_pdl: bool = False,
    prefetch: Optional[bool] = None,
) -> None:
    """Launch the o_proj kernel: ``out[:out_rows] = bf16(v @ w_o^T)``.

    ``use_pdl`` launches it as a programmatic dependent of the previous
    kernel on the stream; ``prefetch`` (default: ``config.prefetch`` when
    ``use_pdl``) issues the L2 prefetch of its ``w_o`` slab first.
    """
    n_dim, k_dim = w_o.shape
    if prefetch is None:
        prefetch = config.prefetch and use_pdl
    launch_kwargs = {"launch_pdl": True} if use_pdl else {}
    _oproj_kernel[(n_dim // config.oproj_block_n,)](
        v,
        w_o,
        w_o.data_ptr(),
        out,
        num_rows,
        out_rows,
        v.stride(0),
        out.stride(0),
        K=k_dim,
        BLOCK_N=config.oproj_block_n,
        BLOCK_K=config.oproj_block_k,
        BLOCK_R=ROW_TILE,
        PREFETCH_SPLITS=_PREFETCH_SPLITS,
        PREFETCH=prefetch,
        USE_PDL=use_pdl,
        num_warps=config.oproj_warps,
        num_stages=config.oproj_stages,
        **launch_kwargs,
    )


_DEFAULT_STATES = {}


def default_state(device: torch.device) -> torch.Tensor:
    """Per-device fused-kernel counters that a benchmark passes explicitly.

    Only for strictly sequential callers on one stream that opt in by passing
    it as ``state``; ``attn_tail`` never falls back to it. The MLA adapter
    owns one state per layer instead.
    """
    from .cute_fused import new_state

    key = device.index or 0
    if key not in _DEFAULT_STATES:
        _DEFAULT_STATES[key] = new_state(device)
    return _DEFAULT_STATES[key]


def attn_tail(
    latent: torch.Tensor,
    w_uv: torch.Tensor,
    w_o: torch.Tensor,
    num_rows: int,
    out_rows: Optional[int] = None,
    use_pdl: Optional[bool] = None,
    v_scratch: Optional[torch.Tensor] = None,
    config: TailConfig = DEFAULT_CONFIG,
    state: Optional[torch.Tensor] = None,
    between: Optional[Callable[[], None]] = None,
    hop_transport=None,
) -> torch.Tensor:
    """Compute the rank-local o_proj partial from the latent attention output.

    The caller validates the arguments with ``check_tail_shapes`` first.

    Args:
        latent: ``[>= num_rows, H, C]`` or ``[>= num_rows, H * C]`` BF16.
        w_uv: ``[H, V, C]`` BF16 (the module's ``v_b_proj``).
        w_o: ``[N, H * V]`` BF16 (the local ``o_proj`` weight).
        num_rows: live rows.
        out_rows: rows of the returned partial (default ``num_rows``); rows
            past ``num_rows`` are zero.
        use_pdl: launch the o_proj kernel as a PDL dependent of the W_UV
            kernel (default: ``TRTLLM_ENABLE_PDL``).
        v_scratch: optional ``[num_rows, H * V]`` BF16 buffer for the
            intermediate (tests read it back); allocated per call otherwise.
        config: kernel launch configuration.
        state: fused-engine counters (``cute_fused.new_state``) owned by the
            caller. When omitted, the fused engine gets zeroed counters
            allocated for this call only (an invocation-local packet, never
            module-global storage).
        between: optional host callback run after the W_UV launch and before
            the o_proj launch (two-kernel engines only).
        hop_transport: the ``IntoHopTransport`` of ``attention_output_into_hop``; the
            ``cute_stream`` o_proj then also publishes all ``out_rows`` rows of the partial
            into it, the zero rows past ``num_rows`` included (``cute_oproj_into_hop``; at
            most 8).

    Returns:
        ``[out_rows, N]`` BF16 partial on ``latent``'s device.
    """
    if out_rows is None:
        out_rows = num_rows
    if use_pdl is None:
        use_pdl = pdl_enabled()
    n_dim, k_dim = w_o.shape
    v = v_scratch
    if v is None:
        v = torch.empty((num_rows, k_dim), dtype=w_uv.dtype, device=latent.device)
    out = torch.empty((out_rows, n_dim), dtype=w_o.dtype, device=latent.device)
    if out_rows > ROW_TILE:
        out[ROW_TILE:].zero_()
    if config.oproj_engine == "fused":
        from .cute_fused import fused_tail, new_state

        if state is None:
            state = new_state(latent.device)
        fused_tail(
            latent[:num_rows].reshape(num_rows, -1),
            w_uv,
            w_o,
            out,
            state,
            v,
            pdl_entry=config.pdl_entry and pdl_entry_enabled(),
        )
        return out
    launch_wuv(
        latent,
        w_uv,
        v,
        num_rows,
        config,
        trigger_dependents=use_pdl,
        wait_producer=config.pdl_entry and pdl_entry_enabled(),
    )
    if between is not None:
        between()
    if config.oproj_engine == "cute_stream" and hop_transport is not None:
        from .into_hop import stream_oproj_into_hop

        stream_oproj_into_hop(
            v,
            w_o,
            out,
            use_pdl=use_pdl,
            mc_words=hop_transport.mc_words,
            flags=hop_transport.flags,
            rank=hop_transport.rank,
            grid=config.oproj_grid or None,
            early_rows=config.early_rows,
            trigger_dependents=config.trigger_dependents and pdl_enabled(),
            vector_publish=config.into_hop_vector_publish,
        )
    elif config.oproj_engine == "cute_stream":
        from .cute_oproj import check_shapes as stream_shapes
        from .cute_oproj import stream_oproj

        grid = config.oproj_grid or None
        if grid is not None and stream_shapes(n_dim, k_dim, grid):
            raise ValueError(f"oproj_grid={grid}: {stream_shapes(n_dim, k_dim, grid)}")
        stream_oproj(
            v,
            w_o,
            out,
            use_pdl=use_pdl,
            grid=grid,
            early_rows=config.early_rows,
            trigger_dependents=config.trigger_dependents and pdl_enabled(),
            trigger_late=config.trigger_late,
        )
    else:
        launch_oproj(v, w_o, out, num_rows, out_rows, config, use_pdl=use_pdl)
    return out

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
"""oproj_weight_staging ``oproj_weight_staging``: staging configuration and the functional attention tail.

``stage_tail`` computes exactly what attention_output_tail's ``attn_tail`` computes with a
``cute_stream`` ``TailConfig`` (served: ``INTO_HOP_SPARSE_DECODE_SCHEDULE``)::

    v = bf16(latent @ w_uv ^ T)  # attention_output_tail's _wuv_kernel (or its fork)
    partial = bf16(v @ w_o ^ T)  # cute_oproj_stage (fork of the
    # served streaming o_proj)

with the same kernels' arithmetic, so the partial (and the attention_output_into_hop words)
are bit-identical to attention_output_tail's. ``StageConfig`` only decides how the o_proj rows
that the served schedule streams after its PDL wait are staged before it:

* host ``"oproj"``: the o_proj kernel itself, parked in
  ``griddepcontrol.wait``, prefetches them into L2 (``stage="l2"``) or
  copies them into its shared-memory slab (``stage="smem"``), right after
  its early copies (``when="entry"``), once they landed
  (``when="landed"``) or tile by tile behind them (``when="chain"``);
* host ``"wuv"``: a fork of attention_output_tail's ``_wuv_kernel`` prefetches them into L2
  before its own wait on the attention kernel (one o_proj CTA's rows per
  W_UV CTA), i.e. earlier than the o_proj becomes resident.

No launch and no stream is added in any configuration.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

from ..attention_output_tail.kernels import (
    ROW_TILE,
    TailConfig,
    launch_wuv,
    pdl_enabled,
    pdl_entry_enabled,
)
from .cute_oproj_stage import check_shapes, check_stage, plan, stream_oproj_stage

HOSTS = ("oproj", "wuv")
_WUV_PREFETCH_PIECES = 128


@dataclass(frozen=True)
class StageConfig:
    """How the after-wait ``w_o`` rows of the o_proj are staged (oproj_weight_staging)."""

    stage: str = "l2"
    """``"none"`` (served schedule, fork only), ``"l2"`` (L2 prefetch before
    the wait; the rows are still copied after it) or ``"smem"`` (copied into
    the slab before the wait)."""
    host: str = "oproj"
    """Kernel that stages: ``"oproj"`` or ``"wuv"`` (``stage="l2"`` only)."""
    when: str = "landed"
    """``host="oproj"``: ``"entry"`` (right after the early copies),
    ``"landed"`` (once the early-only tiles are in shared memory) or
    ``"chain"`` (``stage="smem"``: tile by tile, each once the previous tile
    is in shared memory)."""
    rows: int = -1
    """After-wait rows per CTA to stage (-1: all of them)."""
    prefetch_policy: str = "normal"
    """L2 policy of the staging prefetch: ``"normal"`` or ``"evict_last"``
    (never evict-first)."""
    copy_policy: str = "none"
    """L2 hint of the o_proj's ``w_o`` copies: ``"none"`` or
    ``"evict_first"`` (consumer read of a single-use stream)."""
    early_rows: Optional[int] = None
    """Override of ``TailConfig.early_rows`` (None: the tail's)."""


DEFAULT_STAGE = StageConfig(stage="smem", when="landed", early_rows=24, copy_policy="evict_first")
"""Default schedule: 24 rows per CTA issued at entry, the other 24 copied
into shared memory by the second warp once tile 0 has landed, all before
the wait; every ``w_o`` copy (the single-use consumer read) with an L2
``evict_first`` hint."""


def check_stage_config(stage: StageConfig) -> Optional[str]:
    """Return why ``stage`` is invalid, or ``None``."""
    if stage.host not in HOSTS:
        return f"host {stage.host!r} not in {HOSTS}"
    if stage.host == "wuv" and stage.stage not in ("l2", "none"):
        return "the W_UV host stages into L2 only"
    return check_stage(stage.stage, stage.when, stage.prefetch_policy, stage.copy_policy)


def check_tail_config(config: TailConfig, n_dim: int, k_dim: int, sm_count: int) -> Optional[str]:
    """Return why oproj_weight_staging cannot replace the o_proj of this attention_output_tail config."""
    if config.oproj_engine != "cute_stream":
        return f"o_proj engine {config.oproj_engine!r} is not cute_stream"
    if config.prefetch_l2 or config.successor_prefetch:
        return "attention_output_tail_prefetch off-path prefetch configs keep the attention_output_tail tail"
    if config.trigger_late:
        return "trigger_late is an experiment of the attention_output_tail tail"
    return check_shapes(n_dim, k_dim, config.oproj_grid or sm_count)


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
def _prefetch_l2_evict_last(addr, nbytes):
    tl.inline_asm_elementwise(
        "{ .reg .b64 pol; "
        "createpolicy.fractional.L2::evict_last.b64 pol, 1.0; "
        "cp.async.bulk.prefetch.L2.global.L2::cache_hint [$1], $2, pol; } "
        "// dummy $0",
        "=r,l,r",
        [addr, nbytes],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _wuv_stage_kernel(
    latent_ptr,
    w_uv_ptr,
    v_ptr,
    w_o_addr,
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
    OPROJ_GRID: tl.constexpr,
    HALF_N: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    STAGE_FIRST: tl.constexpr,
    STAGE_BYTES: tl.constexpr,
    PIECES: tl.constexpr,
    EVICT_LAST: tl.constexpr,
):
    """attention_output_tail's ``_wuv_kernel`` plus, before its wait, the L2 prefetch of the
    after-wait ``w_o`` rows of o_proj CTA ``pid`` (rows ``n0 + STAGE_FIRST``
    onward, ``STAGE_BYTES`` bytes; ``n0`` as in the o_proj's even split)."""
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
    pid = head * tl.num_programs(1) + v_block
    if pid < OPROJ_GRID:
        # oproj_weight_staging: stage o_proj CTA pid's after-wait rows into L2 (immutable
        # weight; a prefetch has no architectural effect).
        n0 = 2 * ((pid * HALF_N) // OPROJ_GRID)
        piece: tl.constexpr = STAGE_BYTES // PIECES
        offs_p = tl.arange(0, PIECES).to(tl.int64)
        addr = w_o_addr + (n0 + STAGE_FIRST).to(tl.int64) * ROW_BYTES + offs_p * piece
        if EVICT_LAST:
            _prefetch_l2_evict_last(addr, tl.full([PIECES], piece, tl.int32))
        else:
            _prefetch_l2(addr, tl.full([PIECES], piece, tl.int32))
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


def wuv_stage_geometry(
    n_dim: int, k_dim: int, grid: int, early_rows: int, rows: int
) -> Optional[tuple]:
    """``(first_row, nbytes)`` of the per-CTA prefetch of the W_UV host, or
    ``None`` when every CTA's after-wait rows cannot be covered evenly."""
    geo = plan(n_dim, k_dim, grid)
    if geo["rows_min"] != geo["rows_max"]:
        return None
    late = geo["rows_max"] - min(early_rows, geo["rows_max"])
    if rows < 0 or rows > late:
        rows = late
    nbytes = rows * k_dim * 2
    if nbytes == 0 or nbytes % (_WUV_PREFETCH_PIECES * 16):
        return None
    return early_rows, nbytes


def launch_wuv_stage(
    latent: torch.Tensor,
    w_uv: torch.Tensor,
    v: torch.Tensor,
    num_rows: int,
    w_o: torch.Tensor,
    oproj_grid: int,
    early_rows: int,
    stage: StageConfig,
    config: TailConfig,
    trigger_dependents: bool,
    wait_producer: bool,
) -> None:
    """Launch the W_UV fork that also stages the o_proj's after-wait rows."""
    heads, v_dim, c_dim = w_uv.shape
    n_dim, k_dim = w_o.shape
    if latent.dim() == 2:
        stride_lat_row, stride_lat_head = latent.stride(0), c_dim
    else:
        stride_lat_row, stride_lat_head = latent.stride(0), latent.stride(1)
    geo = wuv_stage_geometry(n_dim, k_dim, oproj_grid, early_rows, stage.rows)
    if geo is None:
        raise ValueError("W_UV host cannot cover the o_proj's after-wait rows")
    first, nbytes = geo
    _wuv_stage_kernel[(heads, v_dim // config.wuv_block_v)](
        latent,
        w_uv,
        v,
        w_o.data_ptr(),
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
        OPROJ_GRID=oproj_grid,
        HALF_N=n_dim // 2,
        ROW_BYTES=k_dim * 2,
        STAGE_FIRST=first,
        STAGE_BYTES=nbytes,
        PIECES=_WUV_PREFETCH_PIECES,
        EVICT_LAST=stage.prefetch_policy == "evict_last",
        num_warps=config.wuv_warps,
        num_stages=1,
        **({"launch_pdl": True} if wait_producer else {}),
    )


def stage_tail(
    latent: torch.Tensor,
    w_uv: torch.Tensor,
    w_o: torch.Tensor,
    num_rows: int,
    out_rows: Optional[int] = None,
    use_pdl: Optional[bool] = None,
    v_scratch: Optional[torch.Tensor] = None,
    config: TailConfig = TailConfig(oproj_grid=128, early_rows=32),
    stage: StageConfig = DEFAULT_STAGE,
    hop_transport=None,
    trace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """attention_output_tail's tail (``cute_stream`` engine) with oproj_weight_staging of the o_proj rows.

    Arguments as ``attention_output_tail.kernels.attn_tail`` (the caller validates them with
    ``check_tail_shapes`` and the config with ``check_tail_config``), plus
    the staging configuration and an optional ``trace`` buffer for the
    o_proj's ``%globaltimer`` stamps (diagnostics only).

    Returns:
        ``[out_rows, N]`` BF16 partial, bit-identical to ``attn_tail``'s.
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
    grid = config.oproj_grid or torch.cuda.get_device_properties(w_o.device).multi_processor_count
    early = config.early_rows if stage.early_rows is None else stage.early_rows
    wait_producer = config.pdl_entry and pdl_entry_enabled()
    wuv_host = stage.host == "wuv" and stage.stage == "l2" and use_pdl
    if wuv_host:
        launch_wuv_stage(
            latent,
            w_uv,
            v,
            num_rows,
            w_o,
            grid,
            early,
            stage,
            config,
            trigger_dependents=use_pdl,
            wait_producer=wait_producer,
        )
    else:
        launch_wuv(
            latent,
            w_uv,
            v,
            num_rows,
            config,
            trigger_dependents=use_pdl,
            wait_producer=wait_producer,
        )
    stream_oproj_stage(
        v,
        w_o,
        out,
        use_pdl=use_pdl,
        early_rows=early,
        grid=grid,
        trigger_dependents=config.trigger_dependents and pdl_enabled(),
        stage="none" if wuv_host else stage.stage,
        when=stage.when,
        stage_rows=stage.rows,
        prefetch_policy=stage.prefetch_policy,
        copy_policy=stage.copy_policy,
        hop_transport=hop_transport,
        vector_publish=config.into_hop_vector_publish,
        trace=trace,
    )
    return out


__all__ = [
    "DEFAULT_STAGE",
    "HOSTS",
    "StageConfig",
    "check_stage_config",
    "check_tail_config",
    "launch_wuv_stage",
    "stage_tail",
    "wuv_stage_geometry",
]

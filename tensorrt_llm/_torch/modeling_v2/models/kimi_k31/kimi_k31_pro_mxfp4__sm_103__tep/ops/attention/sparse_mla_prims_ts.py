# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA (D576, 96 heads, Top-K 2048) on the PrimTS kernels (CuTe DSL).

Plans are cached per (device, planned rows, heads, Top-K, KV format) and created by the first eager call; CUDA
graph capture replays plans made before it. All plans of a device share one workspace (one call in flight per
stream); a plan that needs more allocates a larger workspace and keeps the earlier ones alive, so graphs captured
on them stay valid.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.attention.mla_query_e4m3 import NOPE_DIM, QUERY_DIM

#: KV formats: ``native`` E4M3 ``[N, 576]`` rows.
KV_FORMATS = ("native",)

_PLANS: dict = {}
_WORKSPACES: dict = {}  # device index -> list of buffers, the last one the largest
_LENGTHS: dict = {}  # (device index, top_k) -> int32 buffer of top_k


def _workspace(device: torch.device, nbytes: int) -> torch.Tensor:
    buffers = _WORKSPACES.setdefault(device.index, [])
    if buffers and buffers[-1].numel() >= nbytes:
        return buffers[-1]
    buffers.append(torch.empty((max(nbytes, 1),), dtype=torch.uint8, device=device))
    return buffers[-1]


def _plan(device: torch.device, plan_rows: int, heads: int, top_k: int, kv_format: str):
    key = (device.index, plan_rows, heads, top_k, kv_format)
    wrapper = _PLANS.get(key)
    if wrapper is not None:
        return wrapper
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"sparse MLA plan for {plan_rows} rows ({kv_format}) must be made by an eager call before "
            "CUDA graph capture"
        )
    from ...kernels.attention import mla_decode_plan as mla_decode

    kwargs = dict(
        max_topk=top_k,
        max_q_len=1,
        packed_query=True,
        q_dtype=torch.float8_e4m3fn,
        qk_rope_head_dim=QUERY_DIM - NOPE_DIM,
        kv_cache_format=kv_format,
        return_lse=False,
    )
    need = int(
        mla_decode.get_prims_ts_sparse_mla_decode_workspace_size(device, plan_rows, heads, **kwargs)
    )
    wrapper = mla_decode.BatchSparseMLADecodePagedTSWrapper(
        workspace_buffer=_workspace(device, need)
    )
    wrapper.plan(device, plan_rows, heads, **kwargs)
    _PLANS[key] = wrapper
    return wrapper


def full_lengths(device: torch.device, rows: int, top_k: int) -> torch.Tensor:
    """A stable int32 ``[rows]`` view of ``top_k``: the live length of a ``-1``-padded Top-K row. Grows (eagerly)
    to the largest row count asked for; a graph keeps the buffer it captured."""
    key = (device.index, top_k)
    buf = _LENGTHS.get(key)
    if buf is None or buf.numel() < rows:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("sparse MLA lengths must be allocated before CUDA graph capture")
        buf = torch.full((max(rows, 1 << 16),), top_k, dtype=torch.int32, device=device)
        _LENGTHS[key] = buf
    return buf[:rows]


def _spec(rows, **overrides) -> dict:
    spec = dict(rows=rows, kv_format="native", pool_rows=262_144, holes=0, plan_rows=None)
    spec.update(overrides)
    return spec


class K31SparseMlaPrimTS(OpWrapper):
    """Token-sparse MLA over one KV pool: for every row ``r`` and head ``h``,
    ``out[r, h] = output_scale * softmax_j(softmax_scale * q[r, h] . K[j]) V[j]`` over the pool rows
    ``j = indices[r, c]``, ``c < lengths[r]``, ``indices[r, c] >= 0`` (``-1`` entries are masked), where a pool row
    is ``K = [latent 512 | rope 64]`` and ``V`` its latent part. ``query`` E4M3 ``[rows, 96, 576]``; ``kv_pool``
    E4M3 ``[N, 576]`` (``native``); ``indices`` int32 ``[rows, 2048]`` with a unit last stride; ``lengths`` int32
    ``[rows]``; ``out`` BF16 ``[rows, 96, 512]``. ``plan_rows`` (>= rows) is the row capacity the call runs on:
    one compiled kernel per planned capacity of at least 32 rows (for smaller power-of-two plans the schedule policy
    picks kernels the vendored package does not carry). Returns ``out``.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(why="one row on the smallest plan (32 rows)", spec=_spec(1, plan_rows=32)),
        Cell(why="ShardQ 98 rows on the 128-row plan", spec=_spec(98, plan_rows=128)),
        Cell(why="ShardQ 224 rows on the 256-row plan", spec=_spec(224, plan_rows=256)),
        Cell(why="-1 holes in the Top-K rows (short contexts)", spec=_spec(32, holes=600)),
        Cell(
            why="fewer rows than planned (one plan serves smaller calls)",
            spec=_spec(40, plan_rows=64),
        ),
        Cell(why="a 2048-row context chunk", spec=_spec(2048)),
    )
    note = """
    ARCHS lists what was driven: sm_103 (GB300).

    The TEP perf target runs its token-sharded context prefill (ShardQ: a rank's share of the step's context tokens
    with all 96 heads) through this, with plan_rows = the next power of two of the rows, at least 32. The query is
    the absorbed query quantized per tensor (k31_mla_query_e4m3); softmax_scale carries the softmax scale with the
    Q and KV descales, output_scale the KV descale. Not bitwise to TRTLLM-Gen (different split-KV schedule and
    accumulation order); the reference gates within a tolerance. Kernel: the vendored PrimTS D576 sparse MLA
    (kernels/attention/mla_decode_*), 96 heads and Top-K 2048 only. First call per plan compiles (seconds;
    persistent cache), so it must run eagerly.
    """

    def __call__(
        self,
        query: torch.Tensor,
        kv_pool: torch.Tensor,
        indices: torch.Tensor,
        lengths: torch.Tensor,
        softmax_scale: float,
        output_scale: float,
        out: torch.Tensor,
        kv_format: str,
        plan_rows: int,
    ) -> torch.Tensor:
        rows, heads, _ = query.shape
        wrapper = _plan(query.device, plan_rows, heads, int(indices.shape[1]), kv_format)
        return wrapper.run(
            query,
            kv_pool,
            indices,
            lengths,
            softmax_scale=softmax_scale,
            output_scale=output_scale,
            out=out,
        )

    def reference(
        self,
        query: torch.Tensor,
        kv_pool: torch.Tensor,
        indices: torch.Tensor,
        lengths: torch.Tensor,
        softmax_scale: float,
        output_scale: float,
        out: torch.Tensor,
        kv_format: str,
        plan_rows: int,
    ) -> torch.Tensor:
        """FP32 attention over the selected E4M3 rows as the kernel reads them; ``plan_rows`` is ignored."""
        rows = query.shape[0]
        q = query.float()
        for r in range(rows):
            n = int(lengths[r])
            idx = indices[r, :n].long()
            idx = idx[idx >= 0]
            kv = kv_pool.view(-1, QUERY_DIM)[idx].view(torch.float8_e4m3fn).float()
            s = (q[r] @ kv.t()) * softmax_scale
            p = torch.softmax(s, dim=-1)
            out[r] = ((p @ kv[:, :NOPE_DIM]) * output_scale).to(out.dtype)
        return out

    def is_valid(
        self,
        query: torch.Tensor,
        kv_pool: torch.Tensor,
        indices: torch.Tensor,
        lengths: torch.Tensor,
        softmax_scale: float,
        output_scale: float,
        out: torch.Tensor,
        kv_format: str,
        plan_rows: int,
    ) -> None:
        assert query.shape[0] <= plan_rows, "more rows than planned"
        assert kv_format in KV_FORMATS
        rows_in_pool = kv_pool.numel() // QUERY_DIM
        if indices.numel():
            # Out-of-pool rows read another allocation.
            assert int(indices.max()) < rows_in_pool, "an index past the pool"
        # A live row with no valid key divides by zero.
        assert bool((lengths > 0).all()), "every row needs a live length"


k31_sparse_mla_prims_ts = K31SparseMlaPrimTS()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The last stage's drafter-context store, with one paged store launch per context step.

On a context step the built-in ``DFlashWorker._store_prefill_context`` runs, for every
context request of the step, ``DFlashForCausalLM.precompute_context_kv`` (the fused K/V
GEMM, a K copy and a V copy out of its ``[K | V]`` rows, the stacked k_norm, the positions
as int32 repeated per drafter layer, the RoPE) and then ``_store_context_kv_paged`` (a row
tensor, two int32 casts, and per drafter layer a K copy, a V copy and flashinfer's
``append_paged_kv_cache``): several launches per request and drafter layer.

``store_runs`` keeps the per-request GEMM, K copy, RMS norm, k_norm multiply and RoPE: the
same ops on the same tensors as ``precompute_context_kv``, so every K and V value is the
built-in's. A run with lead rows (a drafter context window's request) runs the GEMM, K copy,
RMS norm and k_norm multiply over its lead and stored rows and the RoPE and store over its
stored rows. It drops the V copy, the position and row tensors and every per-layer copy and
append. The step's RoPE positions and the store's per-request table (the addresses of each
request's K and of its K/V GEMM output, its rows, its block-table row and its first
position) go to the device in one host-to-device copy, and ``page_store.store`` writes the
K and V rows of every drafter layer of every request of the step into the paged pool in one
launch, at the page and offset ``append_paged_kv_cache`` writes them to. Copies only: the
pool holds the built-in's bytes.

A pool of FP8 draft layers (``TRTLLM_DFLASH_FP8_DRAFT_KV``) holds them as FP8 E4M3: the K/V
are still computed in the weights' dtype (``compute_dtype``), and the store converts each
value with the built-in's saturating cast (``to_cache_dtype``: clamp to +-448, round to
nearest even, NaN kept).

``refusal`` names what keeps a worker and drafter on the built-in store; the worker then
runs the built-in ``_store_prefill_context`` unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

#: int64 fields of one request in the store table: K address, K/V address, rows, block-table
#: row, first position, padding.
META_FIELDS = 6
#: The dtype the drafter K/V are computed in (its weights, activations and K/V outputs).
DTYPE = torch.bfloat16
#: The dtypes of the draft pages the store writes: the compute dtype, or FP8 E4M3.
CACHE_DTYPES = (DTYPE, torch.float8_e4m3fn)
#: Largest finite magnitude of FP8 E4M3: the store saturates there.
E4M3_MAX = 448.0


def compute_dtype(draft_model: Any, cache_dtype: torch.dtype) -> torch.dtype:
    """The dtype the drafter K/V of a ``cache_dtype`` pool are computed in.

    The cache dtype itself, except for an FP8 pool: its K/V are computed in the drafter's
    fused K/V weight dtype and converted on store. The fused K/V tensors are built here when
    they are not built yet: a step that keeps the built-in store without asking ``refusal``
    (``store_override``, stream capture) may reach this before ``precompute_context_kv`` ran.
    """
    if cache_dtype == torch.float8_e4m3fn:
        if draft_model._fused_kv_weight is None:
            draft_model._build_fused_kv_buffers()
        return draft_model._fused_kv_weight.dtype
    return cache_dtype


def to_cache_dtype(x: torch.Tensor, cache_dtype: torch.dtype) -> torch.Tensor:
    """``x`` as the draft pages of ``cache_dtype`` hold it: ``x`` itself for its own dtype,
    else (FP8 E4M3) clamped to +-``E4M3_MAX`` and rounded to nearest even, NaN kept."""
    if x.dtype == cache_dtype:
        return x
    return x.clamp(-E4M3_MAX, E4M3_MAX).to(cache_dtype)


@dataclass(frozen=True)
class Run:
    """One context request's rows of a step, as the store writes them.

    ``offset`` is the first of its rows among the step's context rows, ``table_row`` the
    row of the block table its pages are read from (its batch index with the manager's
    tables, its slot on the private arena), ``first`` the drafter-local position of its
    first stored row. ``lead`` rows just ahead of the stored ones (from ``offset`` on) run
    through the K/V GEMM with them and are not stored (``window.WindowPlan.leads``).
    """

    offset: int
    rows: int
    table_row: int
    first: int
    lead: int = 0


def rope_positions(runs: Sequence[Run], layers: int) -> torch.Tensor:
    """The step's RoPE positions, host int32: per run ``arange(first, first + rows)`` with
    every position repeated ``layers`` times (``positions.to(int32).repeat_interleave``)."""
    if not runs:
        return torch.zeros(0, dtype=torch.int32)
    rows = torch.tensor([run.rows for run in runs], dtype=torch.int64)
    starts = torch.cumsum(rows, 0) - rows
    firsts = torch.tensor([run.first for run in runs], dtype=torch.int64)
    total = int(rows.sum())
    base = torch.repeat_interleave(firsts - starts, rows)
    positions = (base + torch.arange(total, dtype=torch.int64)).to(torch.int32)
    return positions.repeat_interleave(layers) if layers > 1 else positions


def host_table(
    runs: Sequence[Run],
    addresses: Sequence[Tuple[int, int]],
    layers: int,
    pin: bool,
) -> torch.Tensor:
    """The host buffer of one step: the store table (``META_FIELDS`` int64 per run), then the
    RoPE positions as int32, two per remaining int64 word."""
    num_runs = len(runs)
    positions = rope_positions(runs, layers)
    words = num_runs * META_FIELDS + (positions.numel() + 1) // 2
    buffer = torch.zeros(words, dtype=torch.int64, pin_memory=pin)
    table = [
        value
        for run, (k_address, kv_address) in zip(runs, addresses)
        for value in (k_address, kv_address, run.rows, run.table_row, run.first, 0)
    ]
    buffer[: num_runs * META_FIELDS] = torch.tensor(table, dtype=torch.int64)
    buffer[num_runs * META_FIELDS :].view(torch.int32)[: positions.numel()] = positions
    return buffer


def split_table(buffer: torch.Tensor, num_runs: int, num_positions: int):
    """``(store table, RoPE positions)`` views of a step buffer ``host_table`` built."""
    meta = buffer[: num_runs * META_FIELDS]
    positions = buffer[num_runs * META_FIELDS :].view(torch.int32)[:num_positions]
    return meta, positions


def layer_caches(worker: Any) -> List[torch.Tensor]:
    """The drafter layers' pages, one ``[pages, 2, kv_heads, page, head_dim]`` view each.

    A view of the manager's pool may span more than its layer's pages, so the page counts
    of the views can differ; the store reads only their addresses and strides.
    """
    buffer = worker._ctx_kv_buf
    return [buffer[layer] for layer in range(len(buffer))]


def block_table(worker: Any) -> Optional[torch.Tensor]:
    """The table the context write reads its pages from (``_ctx_paged_index_args``)."""
    if worker._ctx_block_tables is not None:
        return worker._ctx_block_tables
    return worker._ctx_page_table


def refusal(worker: Any, draft_model: Any) -> Optional[str]:
    """Why ``worker`` and ``draft_model`` keep the built-in store (None when they need not).

    Host checks only. Builds the drafter's fused K/V tensors when they are not built yet
    (``precompute_context_kv`` builds them on its first call).
    """
    if not getattr(worker, "_ctx_paged", False) or getattr(worker, "_ctx_kv_buf", None) is None:
        return "the drafter context is not paged"
    if not hasattr(draft_model, "fc") or not hasattr(draft_model, "hidden_norm"):
        return "the drafter has no fc / hidden_norm"
    from tensorrt_llm._torch.models import modeling_dflash

    base = modeling_dflash.DFlashForCausalLM
    for name in ("precompute_context_kv", "_fused_rope_inplace"):
        if getattr(type(draft_model), name, None) is not getattr(base, name):
            return f"the drafter overrides {name}"
    if modeling_dflash._flashinfer_rope is None:
        return "flashinfer RoPE is unavailable"
    draft_model._build_fused_kv_buffers()
    if getattr(draft_model, "_input_ln_eps", None) is not None:
        return "the drafter normalizes its context input"
    weight = draft_model._fused_kv_weight
    if weight.dtype != DTYPE or not weight.is_cuda:
        return f"the drafter K/V weight is {weight.dtype} on {weight.device}"
    for tensor in (draft_model._fused_kv_bias, draft_model._k_norm_stacked):
        if tensor is not None and tensor.dtype != DTYPE:
            return f"a drafter K/V bias or k_norm weight is {tensor.dtype}"
    layers = draft_model._num_attn_layers
    heads = draft_model._num_kv_heads
    head_dim = draft_model._head_dim
    caches = layer_caches(worker)
    if len(caches) != layers:
        return f"{len(caches)} draft layers in the pool for {layers} drafter layers"
    first = caches[0]
    if first.dtype not in CACHE_DTYPES:
        return f"the draft layers' pages are {first.dtype}, not one of {CACHE_DTYPES}"
    for cache in caches:
        if (
            cache.dim() != 5
            or cache.dtype != first.dtype
            or not cache.is_cuda
            or cache.shape[1] != 2
            or cache.shape[2] != heads
            or cache.shape[4] != head_dim
            or cache.stride(4) != 1
            or cache.shape[1:] != first.shape[1:]
            or cache.stride() != first.stride()
        ):
            return (
                f"a draft layer's pages are {tuple(cache.shape)} / {cache.stride()} / {cache.dtype}"
            )
    table = block_table(worker)
    if table is None or table.dtype != torch.int32 or not table.is_contiguous():
        return "the drafter block table is not a contiguous int32 table"
    return None


def layer_addresses(worker: Any) -> torch.Tensor:
    """The device int64 table of the drafter layers' page addresses, rebuilt when they move."""
    caches = layer_caches(worker)
    key = tuple(cache.data_ptr() for cache in caches)
    held = getattr(worker, "_page_store_layers", None)
    if held is None or held[0] != key:
        held = (key, torch.tensor(key, dtype=torch.int64, device=caches[0].device))
        worker._page_store_layers = held
    return held[1]


def context_k(draft_model: Any, projected: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``precompute_context_kv``'s ops before its RoPE: ``(K, K/V GEMM output)``.

    K is ``[rows, layers, kv_heads, head_dim]`` after the stacked RMS norm and k_norm
    multiply; the GEMM output ``[rows, layers * 2 * kv_heads * head_dim]`` holds V in place.
    """
    rows = projected.shape[0]
    layers = draft_model._num_attn_layers
    heads = draft_model._num_kv_heads
    head_dim = draft_model._head_dim
    weight = draft_model._fused_kv_weight
    if projected.dtype != weight.dtype:
        projected = projected.to(weight.dtype)
    kv_flat = F.linear(projected, weight, draft_model._fused_kv_bias)
    k = kv_flat.view(rows, layers, 2, heads, head_dim)[:, :, 0].contiguous()
    if draft_model._k_norm_stacked is not None:
        k = F.rms_norm(k, (head_dim,), eps=draft_model._k_norm_eps)
        k = k * draft_model._k_norm_stacked.view(1, layers, 1, head_dim)
    return k, kv_flat


def run_k(
    draft_model: Any, ctx_proj: torch.Tensor, run: Run, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``context_k`` of a run: the K/V GEMM over its lead and stored rows of ``ctx_proj`` (cast
    to ``dtype``), then the stored rows' K and K/V GEMM output (views of the run's tensors)."""
    chunk = ctx_proj[run.offset : run.offset + run.lead + run.rows].detach()
    k, kv_flat = context_k(draft_model, chunk.to(dtype))
    return k[run.lead :], kv_flat[run.lead :]


def rope_inplace(draft_model: Any, k: torch.Tensor, positions: torch.Tensor) -> None:
    """``_fused_rope_inplace``'s flashinfer call on ``k``, with its int32 ``positions`` ready."""
    from tensorrt_llm._torch.models import modeling_dflash

    rows, layers = k.shape[0], k.shape[1]
    head_dim = draft_model._head_dim
    k_flat = k.view(rows * layers, -1)
    need_rows = k_flat.shape[0]
    dummy_q = draft_model._rope_dummy_q
    if dummy_q is None or dummy_q.dtype != k_flat.dtype or dummy_q.shape[0] < need_rows:
        dummy_q = k_flat.new_empty(need_rows, head_dim)
        draft_model._rope_dummy_q = dummy_q
    modeling_dflash._flashinfer_rope(
        positions,
        dummy_q[:need_rows],
        k_flat,
        head_dim,
        draft_model._get_cos_sin_cache(),
        draft_model._is_neox,
    )


def store_runs(worker: Any, draft_model: Any, ctx_proj: torch.Tensor, runs: Sequence[Run]) -> None:
    """Compute and store the drafter K/V of a step's ``runs`` (after ``refusal`` returned None).

    ``ctx_proj`` is the step's ``project_target_hidden`` output over its context rows.
    """
    from tensorrt_llm._utils import prefer_pinned

    from . import page_store

    if not runs:
        return
    layers = draft_model._num_attn_layers
    dtype = compute_dtype(draft_model, worker._ctx_kv_buf[0].dtype)
    # Every request's K and GEMM output stay referenced until the store is queued: the store
    # reads them through the addresses in its table.
    held = [run_k(draft_model, ctx_proj, run, dtype) for run in runs]
    total = sum(run.rows for run in runs)
    buffer = host_table(
        runs,
        [(k.data_ptr(), kv_flat.data_ptr()) for k, kv_flat in held],
        layers,
        prefer_pinned(),
    )
    device = buffer.to(ctx_proj.device, non_blocking=True)
    meta, positions = split_table(device, len(runs), total * layers)
    start = 0
    for run, (k, _) in zip(runs, held):
        rope_inplace(draft_model, k, positions[start * layers : (start + run.rows) * layers])
        start += run.rows
    page_store.store(
        meta,
        layer_addresses(worker),
        block_table(worker),
        layer_caches(worker)[0],
        num_runs=len(runs),
        max_rows=max(run.rows for run in runs),
        layers=layers,
        heads=draft_model._num_kv_heads,
        head_dim=draft_model._head_dim,
    )


def reference_store(
    runs: Sequence[Run],
    ks: Sequence[torch.Tensor],
    kv_flats: Sequence[torch.Tensor],
    table: torch.Tensor,
    caches: Sequence[torch.Tensor],
) -> None:
    """What ``page_store.store`` writes, in plain torch, with ``append_paged_kv_cache``'s
    addressing: page ``table.flatten()[table_row * width + position // page]`` and offset
    ``position % page``, for every run, drafter layer, K and V, each value converted with
    ``to_cache_dtype``."""
    flat = table.flatten()
    width = table.shape[1]
    page_size = caches[0].shape[3]
    for run, k, kv_flat in zip(runs, ks, kv_flats):
        rows, layers, heads, head_dim = k.shape
        positions = torch.arange(run.first, run.first + run.rows, device=flat.device)
        pages = flat[run.table_row * width + positions // page_size].long()
        offsets = positions % page_size
        v = kv_flat.view(rows, layers, 2, heads, head_dim)[:, :, 1]
        for layer, cache in enumerate(caches):
            cache[pages, 0, :, offsets, :] = to_cache_dtype(k[:, layer], cache.dtype)
            cache[pages, 1, :, offsets, :] = to_cache_dtype(v[:, layer], cache.dtype)

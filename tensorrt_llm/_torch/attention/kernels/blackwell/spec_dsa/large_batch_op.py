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
"""specDSA large-batch kernels, host side: DeepSeek-V4-Flash decode for many rows on B200 (SM100, 148 SMs).

Two launches per call, back to back on the current stream. The first streams each row's FP8 indexer K cache once,
computes the indexer logits and selects the exact top-512 compressed positions (sampled threshold with a count
certificate and exact fallbacks), maps them to compressed-KV rows and publishes finished rows to a ready queue. The
second, started with programmatic dependent launch, runs the sparse MLA attention over the selected rows plus the
128-row sliding window and the attention sink, claiming rows from the queue as they complete.

``prepare`` builds the per-step state (the CTA schedule from the compressed lengths, the expanded sinks);
``run`` launches both kernels and writes ``topk`` (int32 compressed positions) and ``out`` (bf16). The kernels are
compiled on first use per device, outside CUDA-graph capture, or ahead of it by ``precompile``.

Contract:
- A state belongs to one batch size and one caller. Its workspace (ready queue, selection scratch) is reset by the
  kernels at the end of every launch, so launches that share a state, and ``prepare`` calls that refresh it, must be
  stream-ordered.
- ``topk`` and ``out`` must not alias any input.
- The indexer launch is persistent and needs all 148 SMs; it fails at launch on a partitioned device (MPS limits,
  green contexts).
- Block tables must be contiguous int32 and hold valid page ids for every page a row touches: ``ceil(n / 32)``
  entries for the indexer and the compressed-KV tables, ``ceil(kv_len / 128)`` for the window table.
- The kernels do not trap. A schedule that does not match ``ctx_lens`` (a stale ``prepare``) makes the launch refuse
  every row: ``topk`` is -1, ``out`` is zero and ``error_bits`` is non-zero.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import torch

from .config import DS_V4_FLASH, NUM_SMS, SpecDsaConfig, check_device, supported_spec

__all__ = [
    "ERR_SCHEDULE",
    "SpecDsaLargeBatchState",
    "error_bits",
    "is_supported",
    "precompile",
    "prepare",
    "run",
]

# Bit of ``error_bits`` (the kernel status word): the schedule does not match ctx_lens, i.e. prepare() was not
# refreshed for these lengths.
ERR_SCHEDULE = 1

# Schedule granule: a row costs ceil(n / SCHED_UNIT) units, at least SCHED_MIN_UNITS; the CTAs split the total evenly.
SCHED_UNIT = 256
SCHED_MIN_UNITS = 128
SELECT_BINS = 512
SEGMENT_WORDS = 520  # int64 words per CTA and slot of the selection scratch
# int32 words per row of the ready queue; word AQ_ENTRY + k holds the k-th finished row.
READY_QUEUE_WORDS = 16

_KERNEL_CONFIG = dict(
    phys_block_kv=DS_V4_FLASH.index_page_tokens,
    desc=True,
    sel_fin=True,
    gath=True,
    rot=True,
    ahint=1,
    late_release=False,
    apace=10,
    sel_idle_ns=128,
    rsort=True,
    policy=1,
    cbins=SELECT_BINS,
    trig0=768,
    trig=1536,
    attn=False,
    kv_stages=4,
    q_stages=2,
)

# The knobs of the indexer kernel per spec (kernel_key).
_KNOBS = {DS_V4_FLASH.kernel_key: _KERNEL_CONFIG}

_lock = threading.Lock()
_engines: dict[tuple, dict] = {}


def _kernel_modules() -> tuple:
    from . import large_batch, large_batch_lib

    return large_batch, large_batch_lib


def is_supported(
    device: torch.device | None = None, config: SpecDsaConfig | None = None
) -> tuple[bool, str]:
    """Whether the large-batch kernels serve ``config`` (default: DS-V4-Flash) on ``device``.

    That takes a supported spec with large-batch kernels, CuTe DSL installed and SM100 with all 148 SMs (one CTA per
    SM).
    """
    try:
        if not supported_spec(config).large_batch:
            return False, "the spec has no large-batch kernels"
    except NotImplementedError as e:
        return False, str(e)
    return check_device(device)


def _check_batch_size(batch_size: int, config: SpecDsaConfig) -> None:
    if not 1 <= batch_size <= config.max_batch:
        raise ValueError(f"batch size {batch_size} not in [1, {config.max_batch}]")


def _device_index(device: torch.device) -> int:
    device = torch.device(device)
    return device.index if device.index is not None else torch.cuda.current_device()


def _engine(device: torch.device, config: SpecDsaConfig | None = None) -> dict:
    config = supported_spec(config)
    if not config.large_batch:
        raise ValueError(f"{config.name} has no large-batch kernels")
    key = (_device_index(device), config.kernel_key)
    with _lock:
        if key not in _engines:
            lb, _ = _kernel_modules()
            _engines[key] = {
                "indexer": lb.IndexerKernel(
                    num_sms=NUM_SMS, config=config, **_KNOBS[config.kernel_key]
                ),
                "indexer_fn": None,
                "tail": lb.AttentionTail(config=config),
                "tail_fn": None,
            }
        return _engines[key]


def precompile(device: torch.device, config: SpecDsaConfig | None = None) -> None:
    """Compile both kernels for ``device`` (the indexer kernel takes about a minute).

    The compiled kernels serve every batch size. Must run outside CUDA-graph capture; ``run`` compiles on first use
    otherwise.
    """
    lb, _ = _kernel_modules()
    e = _engine(device, config)
    with _lock:
        if e["indexer_fn"] is None or e["tail_fn"] is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "spec_dsa (large batch): compile the kernels (precompile) before "
                    "CUDA-graph capture"
                )
            if e["indexer_fn"] is None:
                e["indexer_fn"] = lb.compile_indexer(e["indexer"])
            if e["tail_fn"] is None:
                e["tail_fn"] = lb.compile_tail(e["tail"])


@dataclass
class SpecDsaLargeBatchState:
    """Per-step metadata and workspace of one caller; see ``prepare``.

    ``sched`` is the CTA schedule (row and unit offset where each CTA starts, one extra entry for the end),
    ``sinks`` the per-row copy of the attention sink. ``ready_queue`` carries finished rows from the indexer to the
    attention kernel. ``selection_ws`` and ``counters`` are per-row selection scratch; ``gidx`` holds the 640 KV row
    indices of each row (128 window rows, then the selection). ``status`` is the kernel status word (``error_bits``).
    ``lg``, ``req`` and ``stamps`` are kernel arguments that carry no state. ``config`` is the spec the state was built
    for.
    """

    batch_size: int
    sched: torch.Tensor
    sinks: torch.Tensor
    req: torch.Tensor
    gidx: torch.Tensor
    selection_ws: torch.Tensor
    counters: torch.Tensor
    ready_queue: torch.Tensor
    stamps: torch.Tensor
    lg: torch.Tensor
    status: torch.Tensor
    config: SpecDsaConfig


def _schedule(ctx_lens: list[int], num_ctas: int) -> np.ndarray:
    """CTA schedule: CTA ``s`` starts at unit ``s * quota + min(s, rem)`` of the rows' concatenated unit ranges.

    Row ``r`` spans ``max(ceil(n_r / SCHED_UNIT), SCHED_MIN_UNITS)`` units; entry ``s`` is ``(row, unit offset in
    the row)`` of that start (offset 0 for rows with fewer than ``SCHED_MIN_UNITS`` real units, which are never
    split) and entry ``num_ctas`` is ``(batch_size, 0)``.
    """
    n = np.asarray(ctx_lens, dtype=np.int64)
    batch = n.size
    units = -(-n // SCHED_UNIT)
    cost = np.maximum(units, SCHED_MIN_UNITS)
    prefix = np.concatenate([[0], np.cumsum(cost)])
    quota, rem = divmod(int(prefix[-1]), num_ctas)
    s = np.arange(num_ctas + 1, dtype=np.int64)
    start = s * quota + np.minimum(s, rem)
    row = np.searchsorted(prefix[1:], start, side="right")
    inside = row < batch
    row_c = np.minimum(row, batch - 1)
    offset = np.where(inside & (units[row_c] >= SCHED_MIN_UNITS), start - prefix[row_c], 0)
    sched = np.stack([np.where(inside, row, batch), offset], axis=1)
    return sched.astype(np.int32)


def _host_ints(t: torch.Tensor, count: int, name: str) -> list[int]:
    v = t.reshape(-1)
    if v.numel() < count:
        raise ValueError(f"{name} has {v.numel()} entries, need {count}")
    return [int(x) for x in v[:count].tolist()]


def prepare(
    ctx_lens: torch.Tensor,
    sinks: torch.Tensor,
    state: SpecDsaLargeBatchState | None = None,
    *,
    config: SpecDsaConfig | None = None,
) -> SpecDsaLargeBatchState:
    """Build (or refresh in place) the per-step state for ``B = ctx_lens.numel()`` rows.

    Args:
        ctx_lens: int32 [B], compressed (indexer) length of each row, in [512, 2**18] for DS-V4-Flash.
        sinks: float32 [64] on the target device, per-head attention sink.
        state: a state of the same batch size to refresh; keeps its workspace and its spec.
        config: the spec (default: DS-V4-Flash, the only one the kernels serve).

    The lengths are read on the host: pass a host (CPU) tensor to avoid a device-to-host copy; a device tensor works
    but synchronizes.
    """
    if state is not None:
        if config is not None and config.kernel_key != state.config.kernel_key:
            raise ValueError(f"state is for {state.config.name!r}, got {config.name!r}")
        config = state.config
    config = supported_spec(config)
    batch_size = int(ctx_lens.numel())
    _check_batch_size(batch_size, config)
    device = sinks.device
    if device.type != "cuda":
        raise ValueError("sinks must be a CUDA tensor")
    if sinks.numel() != config.attn_heads:
        raise ValueError(f"sinks must have {config.attn_heads} entries, got {tuple(sinks.shape)}")
    ctx_host = _host_ints(ctx_lens, batch_size, "ctx_lens")
    if any(not config.min_index_len <= n <= config.max_index_len for n in ctx_host):
        raise ValueError(
            f"compressed lengths must lie in [{config.min_index_len}, {config.max_index_len}]"
        )
    if state is not None and state.batch_size != batch_size:
        raise ValueError(f"state is for {state.batch_size} rows, got {batch_size}")
    _, lib = _kernel_modules()
    if state is None:
        ready_queue = torch.zeros((batch_size, READY_QUEUE_WORDS), dtype=torch.int32, device=device)
        ready_queue.view(-1)[lib.AQ_ENTRY : lib.AQ_ENTRY + batch_size] = -1
        state = SpecDsaLargeBatchState(
            batch_size=batch_size,
            sched=torch.empty((NUM_SMS + 1, 2), dtype=torch.int32, device=device),
            sinks=torch.empty((batch_size, config.attn_heads), dtype=torch.float32, device=device),
            req=torch.arange(batch_size, dtype=torch.int32, device=device),
            gidx=torch.full(
                (batch_size, config.selected_rows), -9, dtype=torch.int32, device=device
            ),
            selection_ws=torch.zeros(3 * NUM_SMS * SEGMENT_WORDS, dtype=torch.int64, device=device),
            counters=torch.zeros(batch_size * 16, dtype=torch.int64, device=device),
            ready_queue=ready_queue,
            stamps=torch.zeros(NUM_SMS * 256, dtype=torch.int64, device=device),
            lg=torch.empty((1, 256), dtype=torch.float32, device=device),
            status=torch.full((4,), -7, dtype=torch.int32, device=device),
            config=config,
        )
    state.sched.copy_(torch.from_numpy(_schedule(ctx_host, NUM_SMS)), non_blocking=True)
    state.sinks.copy_(sinks.reshape(1, config.attn_heads).expand(batch_size, config.attn_heads))
    return state


def error_bits(state: SpecDsaLargeBatchState) -> torch.Tensor:
    """Status word of the last launch of ``state`` (``ERR_*`` bits): an int32 [1] device view, no sync.

    Zero means the last launch ran normally; a fresh state reads -7 until its first launch.
    """
    return state.status[:1]


def _check_numel(name: str, t: torch.Tensor, numel: int, dtype: torch.dtype | None = None) -> None:
    if t.numel() != numel or (dtype is not None and t.dtype != dtype) or not t.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous with {numel} elements"
            + (f" of {dtype}" if dtype is not None else "")
            + f", got {t.dtype} {tuple(t.shape)}"
        )


def _check_rows(name: str, t: torch.Tensor, row_bytes: int) -> None:
    if t.element_size() != 1 or t.dim() < 2 or t.shape[-1] != row_bytes or not t.is_contiguous():
        raise ValueError(
            f"{name} must be a contiguous 1-byte tensor with {row_bytes}-byte rows, "
            f"got {t.dtype} {tuple(t.shape)}"
        )


def _check_block_table(name: str, t: torch.Tensor, batch_size: int) -> None:
    if t.dtype != torch.int32 or t.dim() != 2 or t.shape[0] < batch_size or not t.is_contiguous():
        raise ValueError(
            f"{name} must be a contiguous int32 [B, pages] tensor with B >= {batch_size}, "
            f"got {t.dtype} {tuple(t.shape)}"
        )


def run(
    state: SpecDsaLargeBatchState,
    q_idx: torch.Tensor,
    w_idx: torch.Tensor,
    k_cache: torch.Tensor,
    idx_block_table: torch.Tensor,
    ctx_lens: torch.Tensor,
    kv_lens: torch.Tensor,
    q_att: torch.Tensor,
    swa_pool: torch.Tensor,
    bt_swa: torch.Tensor,
    c_pool: torch.Tensor,
    bt_c: torch.Tensor,
    topk: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float | None = None,
) -> None:
    """Launch the two kernels on the current stream.

    Args:
        state: from ``prepare`` for these lengths.
        q_idx: float8_e4m3fn [B, 64, 128] (or [B, 1, 64, 128]), indexer queries.
        w_idx: float32 [B, 64], indexer head weights.
        k_cache: uint8 [pages, ...] with 4224 bytes per page, indexer K pages (32 x 128 FP8 values, then 32 float32
            scales).
        idx_block_table: int32 [B, pages], indexer K pages of each row.
        ctx_lens: int32 [B] on the device, the same lengths as given to ``prepare``.
        kv_lens: int32 [B] on the device, uncompressed length of each row (>= 128); the window is its last 128
            positions.
        q_att: float8_e4m3fn [B, 64, 512], attention queries.
        swa_pool: float8_e4m3fn [..., 512], sliding-window KV rows (128 rows per page); bt_swa: int32 [B, pages].
        c_pool: float8_e4m3fn [..., 512], compressed KV rows (32 rows per page); bt_c: int32 [B, pages].
        topk: int32 [B, 512] output, the selected compressed positions (in no particular order).
        out: bfloat16 [B, 64 * 512] output.
        softmax_scale: attention softmax scale (default: ``1 / sqrt(kv_dim)`` of the spec).

    The shapes are those of the DS-V4-Flash spec; ``state.config`` gives the sizes the checks use.
    """
    config = state.config
    b = state.batch_size
    if softmax_scale is None:
        softmax_scale = config.default_softmax_scale
    _check_numel("q_idx", q_idx, b * config.index_heads * config.index_head_dim)
    if q_idx.element_size() != 1:
        raise ValueError(f"q_idx must be FP8 (1 byte per value), got {q_idx.dtype}")
    _check_numel("w_idx", w_idx, b * config.index_heads, torch.float32)
    _check_numel("q_att", q_att, b * config.attn_heads * config.kv_dim)
    if q_att.element_size() != 1:
        raise ValueError(f"q_att must be FP8 (1 byte per value), got {q_att.dtype}")
    _check_numel("ctx_lens", ctx_lens, b, torch.int32)
    _check_numel("kv_lens", kv_lens, b, torch.int32)
    _check_numel("topk", topk, b * config.topk, torch.int32)
    _check_numel("out", out, b * config.attn_heads * config.kv_dim, torch.bfloat16)
    _check_rows("swa_pool", swa_pool, config.kv_dim)
    _check_rows("c_pool", c_pool, config.kv_dim)
    for name, table in (("idx_block_table", idx_block_table), ("bt_swa", bt_swa), ("bt_c", bt_c)):
        _check_block_table(name, table, b)
    pages = int(k_cache.shape[0])
    if k_cache.element_size() != 1 or k_cache.numel() != pages * config.index_page_bytes:
        raise ValueError(
            f"k_cache must hold {config.index_page_bytes} bytes per page, got {tuple(k_cache.shape)}"
        )
    e = _engine(q_idx.device, config)
    if e["indexer_fn"] is None or e["tail_fn"] is None:
        precompile(q_idx.device, config)
    q_att8 = q_att.view(torch.uint8).view(b, config.attn_heads, config.kv_dim)
    swa8 = swa_pool.view(torch.uint8).view(-1, config.kv_dim)
    c8 = c_pool.view(torch.uint8).view(-1, config.kv_dim)
    out3 = out.view(b, config.attn_heads, config.kv_dim)
    e["indexer_fn"](
        k_cache.view(pages, -1),
        q_idx.reshape(-1)
        .as_strided(
            (config.index_heads, config.index_head_dim, b),
            (config.index_head_dim, 1, config.index_heads * config.index_head_dim),
        )
        .view(torch.uint8),
        w_idx.reshape(-1).as_strided((config.index_heads, b), (1, config.index_heads)),
        state.lg,
        idx_block_table[:b],
        ctx_lens,
        state.sched,
        kv_lens,
        state.req,
        bt_swa[:b],
        bt_c[:b],
        topk,
        state.gidx,
        state.selection_ws,
        state.counters,
        state.ready_queue,
        state.stamps,
        q_att8,
        swa8,
        c8,
        state.sinks,
        out3,
        state.status,
        softmax_scale,
        pages,
        b,
    )
    e["tail_fn"](
        q_att8, swa8, c8, state.gidx, state.sinks, out3, state.ready_queue, softmax_scale, NUM_SMS
    )

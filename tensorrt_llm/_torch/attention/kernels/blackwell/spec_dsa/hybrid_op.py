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
"""specDSA tail-attention hybrid, host side: decode of 5 to 26 rows (``tail_hybrid_range`` of the spec) of an MXFP4
indexer K cache on B200 (SM100, 148 SMs), behind the interface of the other implementations.

Three launches per call, back to back on the current stream: the indexer logits kernel of the production MXFP4 chain
(``torch.ops.trtllm.cute_dsl_fp4_paged_mqa_logits``, called as the chain calls it), the multi-CTA exact top-512 of
``topk_multicta_op`` and the tail attention of ``tail_op``. That is ``tail_op.run_hybrid`` with the multi-CTA top-K and
the defaults of its ``pdl`` and ``poll`` (``tail_op.default_pdl`` / ``default_poll``: up to 24 rows the tail attention
is launched programmatically behind the top-K, up to 23 rows it also polls the top-K's output entries). The logits
kernel is the repository's production one and not a kernel of this package: it needs
``tensorrt_llm._torch.custom_ops`` with CuTe DSL and ``tensorrt_llm.deep_gemm.get_paged_mqa_logits_metadata``. They are
resolved when first used; a missing one is a ``RuntimeError`` that names it, and there is no fallback
(``op.prepare(..., implementation="megakernel")`` serves 5 to 20 rows without them).

``prepare`` builds the per-step state (the compressed lengths on the device, the scheduler metadata of the logits kernel
built from them, the sinks copy); ``run`` launches the three kernels and writes ``topk`` (int32 compressed positions)
and ``out`` (bf16). The kernels are compiled on first use, outside CUDA-graph capture, or ahead of it by
``precompile``, which also launches the logits kernel once on dummy data: its first call compiles it.

The logits buffer has ``config.max_index_len`` columns whatever the lengths (the logits kernel writes each row's valid
length), so a captured launch has fixed shapes: the production op returns a view of its persistent logits buffer
([B, 2**18] float32). The hybrid is tuned for rows of 1M tokens (2**18 compressed positions). Its top-K is exact for
every row length, but it is the faster stage only for rows of 16,384 positions or more
(``topk_multicta_op.is_profitable``); the megakernel stays selectable at 5 to 20 rows.

Contract:
- A state belongs to one batch size and one caller. ``prepare`` refreshes it in place (a launch captured earlier reads
  the same addresses), so launches that share a state and ``prepare`` calls on it must be stream-ordered.
- The logits live in the process-global buffer of the production op: hybrid launches, and the other users of that op,
  must be stream-ordered with respect to each other, also across states.
- ``ctx_lens`` and ``kv_lens`` are read on the host by ``prepare``: pass host (CPU) tensors to avoid a device-to-host
  copy; device tensors work but synchronize. The logits kernel and the top-K read the compressed lengths from the state
  (the device copy that ``prepare`` keeps and built the schedule from), so ``run`` checks its ``ctx_lens`` and does not
  read it. The tail attention reads the ``kv_lens`` of ``run`` at every launch.
- ``idx_block_table``, ``bt_swa`` and ``bt_c`` must be contiguous int32 [B, pages] tensors and ``kv_lens`` a contiguous
  int32 [B] tensor: the kernels are compiled for compact tensors. The megakernel also takes strided block tables.
- ``topk`` and ``out`` must not alias any input. The multi-CTA top-K marks the entries of ``topk`` "not written yet"
  when it starts and writes each one once; the tail attention polls them.
- The launches need all 148 SMs and clusters of up to 8 CTAs; they fail at launch on a partitioned device (MPS limits,
  green contexts).
- Block tables must hold valid page ids for every page a row touches: the indexer table covers whole 128-position tiles
  (``4 * ceil(n / 128)`` entries per row), the compressed-KV table ``ceil(n / 32)`` entries.
- There is no device-side error flag (``error_bits`` is a zero view). A state that was not refreshed for the step is not
  detected: the launch computes for the lengths of the last ``prepare`` and its outputs are undefined.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import torch

from . import tail_op, topk_multicta_op
from .config import (
    DS_V4_FLASH_FP4,
    INDEX_TILE,
    NUM_SMS,
    SpecDsaConfig,
    check_device,
    supported_spec,
)

__all__ = [
    "SpecDsaHybridState",
    "error_bits",
    "is_supported",
    "precompile",
    "prepare",
    "run",
    "supported_batch_sizes",
]

# Positions per block of the scheduler metadata of the production paged-MQA logits kernels (``_DG_SCHEDULE_BLOCK_KV``
# of the DSA indexer, ``INDEX_PAGE_ROWS`` of the CSA2 one): a property of the schedule, not of the K cache page.
_SCHEDULE_BLOCK_KV = 64

_lock = threading.Lock()
# Device indexes on which the logits kernel has been launched outside a capture, i.e. is compiled.
_k1_ready: set[int] = set()


def _k1_op():
    """The production MXFP4 paged-MQA logits op, or a RuntimeError if this build does not register it."""
    try:
        import tensorrt_llm._torch.custom_ops  # noqa: F401  (registers the trtllm::cute_dsl_* ops)
    except ImportError as e:
        raise RuntimeError(
            "the specDSA tail hybrid needs tensorrt_llm._torch.custom_ops for the production indexer logits op "
            f"torch.ops.trtllm.cute_dsl_fp4_paged_mqa_logits, which failed to import: {e}"
        ) from e
    try:
        return torch.ops.trtllm.cute_dsl_fp4_paged_mqa_logits
    except AttributeError as e:
        raise RuntimeError(
            "the specDSA tail hybrid needs the production indexer logits op "
            "torch.ops.trtllm.cute_dsl_fp4_paged_mqa_logits, which is not registered (it needs CuTe DSL, "
            "nvidia-cutlass-dsl, on SM100)"
        ) from e


def _metadata_fn():
    """The scheduler metadata function of the production paged-MQA logits kernels, or a RuntimeError."""
    try:
        from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata
    except ImportError as e:
        raise RuntimeError(
            "the specDSA tail hybrid needs tensorrt_llm.deep_gemm.get_paged_mqa_logits_metadata, the scheduler "
            f"metadata of the production indexer logits op, which failed to import: {e}"
        ) from e
    return get_paged_mqa_logits_metadata


def _spec(config: SpecDsaConfig | None) -> SpecDsaConfig:
    """``config`` (default: DS-V4-Flash-FP4, the spec with a tail-hybrid range) if the kernels serve it."""
    return supported_spec(DS_V4_FLASH_FP4 if config is None else config)


def supported_batch_sizes(config: SpecDsaConfig | None = None) -> range:
    """Batch sizes of the tail-hybrid range of ``config`` (default: DS-V4-Flash-FP4); empty without a range."""
    rng = _spec(config).tail_hybrid_range
    return range(0) if rng is None else range(rng[0], rng[1] + 1)


def is_supported(
    device: torch.device | None = None, config: SpecDsaConfig | None = None
) -> tuple[bool, str]:
    """Whether the tail hybrid serves ``config`` (default: DS-V4-Flash-FP4) on ``device``, and why not if it does not.

    That takes a spec with a tail-hybrid range and an MXFP4 indexer K cache, CuTe DSL installed, SM100 with all 148 SMs
    and the production logits op and its scheduler metadata function. The batch size is checked by ``prepare``.
    """
    try:
        spec = _spec(config)
    except NotImplementedError as e:
        return False, str(e)
    if spec.tail_hybrid_range is None:
        return False, f"{spec.name} has no tail-hybrid range"
    if spec.index_k_dtype != "fp4":
        return (
            False,
            f"the tail hybrid serves the MXFP4 indexer K cache, {spec.name} has {spec.index_k_dtype}",
        )
    ok, why = check_device(device)
    if not ok:
        return False, why
    try:
        _k1_op()
        _metadata_fn()
    except RuntimeError as e:
        return False, str(e)
    return True, ""


def _check_batch_size(batch_size: int, config: SpecDsaConfig) -> None:
    rng = config.tail_hybrid_range
    if rng is None:
        raise ValueError(f"{config.name} has no tail-hybrid range")
    if config.index_k_dtype != "fp4":
        raise ValueError(
            f"the tail hybrid serves the MXFP4 indexer K cache, {config.name} has {config.index_k_dtype}"
        )
    if not rng[0] <= batch_size <= rng[1]:
        raise ValueError(f"batch size {batch_size} not in [{rng[0]}, {rng[1]}]")


def _device_index(device: torch.device) -> int:
    device = torch.device(device)
    return device.index if device.index is not None else torch.cuda.current_device()


def _warm_k1(device: torch.device, batch_size: int, config: SpecDsaConfig) -> None:
    """Launch the production logits op once on dummy data (one tile per row): its first call compiles it and sizes its
    logits buffer, and the scheduler metadata function builds its kernel for ``batch_size`` rows."""
    k1, metadata = _k1_op(), _metadata_fn()
    heads, q_values = config.index_heads, config.index_head_dim // 2
    page_tokens = config.index_page_tokens
    pages = config.pages_per_tile
    ctx = torch.full((batch_size,), INDEX_TILE, dtype=torch.int32, device=device)
    table = torch.arange(pages, dtype=torch.int32, device=device).repeat(batch_size, 1)
    k1(
        torch.zeros((batch_size, 1, heads, q_values), dtype=torch.uint8, device=device),
        torch.zeros((batch_size, 1, heads), dtype=torch.int32, device=device),
        torch.zeros(
            (pages, page_tokens, 1, config.index_page_bytes // page_tokens),
            dtype=torch.uint8,
            device=device,
        ),
        torch.zeros((batch_size, heads), dtype=torch.float32, device=device),
        ctx,
        table,
        metadata(ctx.view(-1, 1), _SCHEDULE_BLOCK_KV, NUM_SMS),
        config.max_index_len,
    )
    torch.cuda.current_stream(device).synchronize()
    with _lock:
        _k1_ready.add(_device_index(device))


def precompile(device: torch.device, batch_size: int, config: SpecDsaConfig | None = None) -> None:
    """Compile what ``batch_size`` rows need on ``device``: the multi-CTA top-K and the tail attention in the variant
    ``run`` launches (``tail_op.default_pdl`` / ``default_poll``), and the production logits kernel.

    Must run outside CUDA-graph capture; ``run`` compiles on first use otherwise.
    """
    config = _spec(config)
    _check_batch_size(batch_size, config)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("spec_dsa tail hybrid: compile (precompile) before CUDA-graph capture")
    _k1_op()
    _metadata_fn()
    pdl, poll = tail_op.default_pdl(batch_size), tail_op.default_poll(batch_size)
    topk_multicta_op.precompile(device, batch_size, config, trigger=pdl, arm=poll)
    tail_op.precompile(device, batch_size, config, pdl=pdl, poll=poll)
    _warm_k1(device, batch_size, config)


@dataclass
class SpecDsaHybridState:
    """Per-step metadata of one caller; see ``prepare``.

    ``ctx_lens`` is the int32 [B] copy of the compressed lengths that the logits kernel and the top-K read, ``sched``
    the int32 [NUM_SMS + 1, 2] scheduler metadata of the production logits kernel built from them, ``sinks`` the
    float32 [attn_heads] copy of the attention sinks that the tail attention reads, ``status`` an int32 [1] zero that
    ``error_bits`` returns and ``config`` the spec the state was built for.
    """

    batch_size: int
    ctx_lens: torch.Tensor
    sched: torch.Tensor
    sinks: torch.Tensor
    status: torch.Tensor
    config: SpecDsaConfig


def _host_ints(t: torch.Tensor, count: int, name: str) -> list[int]:
    v = t.reshape(-1)
    if v.numel() != count:
        raise ValueError(f"{name} must have {count} entries, got {v.numel()}")
    return [int(x) for x in v.tolist()]


def prepare(
    ctx_lens: torch.Tensor,
    kv_lens: torch.Tensor,
    idx_block_table: torch.Tensor,
    sinks: torch.Tensor,
    state: SpecDsaHybridState | None = None,
    *,
    config: SpecDsaConfig | None = None,
) -> SpecDsaHybridState:
    """Build (or refresh in place) the per-step state for ``B = ctx_lens.numel()`` rows.

    Args:
        ctx_lens: int32 [B], compressed (indexer) length of each row, in [512, 2**18]; the state keeps a device copy
            that the logits kernel and the top-K read.
        kv_lens: int32 [B], uncompressed length of each row (>= 128); exactly B entries.
        idx_block_table: contiguous int32 [B, pages] on the target device, indexer K pages of each row (32 compressed
            positions per page; whole 128-position tiles, so at least ``4 * ceil(n / 128)`` valid entries).
        sinks: float32 [64] on the target device, per-head attention sink.
        state: a state of the same batch size to refresh; keeps its buffers (a captured launch reads them) and its spec.
        config: the spec (default: DS-V4-Flash-FP4). A new state is built for it; a refreshed state keeps its own.

    The lengths are checked on the host: pass host (CPU) tensors to avoid a device-to-host copy; device tensors work
    but synchronize.
    """
    if state is not None:
        if config is not None and config.kernel_key != state.config.kernel_key:
            raise ValueError(f"state is for {state.config.name!r}, got {config.name!r}")
        config = state.config
    config = _spec(config)
    batch_size = int(ctx_lens.numel())
    _check_batch_size(batch_size, config)
    device = idx_block_table.device
    if device.type != "cuda":
        raise ValueError("idx_block_table must be a CUDA tensor")
    if sinks.numel() != config.attn_heads:
        raise ValueError(f"sinks must have {config.attn_heads} entries, got {tuple(sinks.shape)}")
    ctx_host = _host_ints(ctx_lens, batch_size, "ctx_lens")
    kv_host = _host_ints(kv_lens, batch_size, "kv_lens")
    if any(not config.min_index_len <= n <= config.max_index_len for n in ctx_host):
        raise ValueError(
            f"compressed lengths must lie in [{config.min_index_len}, {config.max_index_len}], got {ctx_host}"
        )
    if any(n < config.window for n in kv_host):
        raise ValueError(f"uncompressed lengths must be at least {config.window}, got {kv_host}")
    need_pages = config.pages_per_tile * math.ceil(max(ctx_host) / INDEX_TILE)
    _check_block_table("idx_block_table", idx_block_table, batch_size)
    if idx_block_table.shape[1] < need_pages:
        raise ValueError(
            f"idx_block_table has {idx_block_table.shape[1]} pages per row, need {need_pages}"
        )
    if state is not None and state.batch_size != batch_size:
        raise ValueError(f"state is for {state.batch_size} rows, got {batch_size}")
    _k1_op()
    metadata = _metadata_fn()
    if state is None:
        state = SpecDsaHybridState(
            batch_size=batch_size,
            ctx_lens=torch.empty(batch_size, dtype=torch.int32, device=device),
            sched=torch.empty((NUM_SMS + 1, 2), dtype=torch.int32, device=device),
            sinks=torch.empty(config.attn_heads, dtype=torch.float32, device=device),
            status=torch.zeros(1, dtype=torch.int32, device=device),
            config=config,
        )
    # one copy of the lengths feeds the schedule, the logits kernel and the top-K
    state.ctx_lens.copy_(torch.tensor(ctx_host, dtype=torch.int32), non_blocking=True)
    state.sched.copy_(metadata(state.ctx_lens.view(-1, 1), _SCHEDULE_BLOCK_KV, NUM_SMS))
    state.sinks.copy_(sinks.reshape(-1))
    return state


def error_bits(state: SpecDsaHybridState) -> torch.Tensor:
    """An int32 [1] device zero: the tail hybrid has no device-side error flag. No sync.

    A state that was not refreshed for the step (``prepare`` not called with its lengths) is not detected.
    """
    return state.status


def _check_numel(name: str, t: torch.Tensor, numel: int, dtype: torch.dtype | None = None) -> None:
    if t.numel() != numel or (dtype is not None and t.dtype != dtype) or not t.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous with {numel} elements"
            + (f" of {dtype}" if dtype is not None else "")
            + f", got {t.dtype} {tuple(t.shape)}"
        )


def _check_block_table(name: str, t: torch.Tensor, batch_size: int) -> None:
    if t.dtype != torch.int32 or t.dim() != 2 or t.shape[0] < batch_size or not t.is_contiguous():
        raise ValueError(
            f"{name} must be a contiguous int32 [B, pages] tensor with B >= {batch_size} (the tail hybrid takes no "
            f"strided block table, the megakernel does), got {t.dtype} {tuple(t.shape)} with strides {t.stride()}"
        )


def run(
    state: SpecDsaHybridState,
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
    *,
    q_scale: torch.Tensor | None = None,
) -> None:
    """Launch the logits kernel, the top-K and the tail attention on the current stream.

    Args:
        state: from ``prepare`` for these lengths and block tables.
        q_idx: packed E2M1 indexer queries, 1 byte per two values ([B, 64, 64], or [B, 1, 64, 64]); contiguous.
        w_idx: float32 [B, 64], indexer head weights.
        k_cache: uint8 [pages, ...] with ``config.index_page_bytes`` (2176) bytes per page, indexer K pages (32 x 64
            bytes of E2M1 pairs, then 32 int32 of four UE8M0 scales, one per 32 values).
        idx_block_table: contiguous int32 [B, pages], as given to ``prepare``.
        ctx_lens: int32 [B] on the device, the lengths given to ``prepare``; checked, not read (the launch uses the
            lengths of the state).
        kv_lens: contiguous int32 [B] on the device, uncompressed lengths (the window is the last 128 positions).
        q_att: float8_e4m3fn [B, 64, 512], attention queries.
        swa_pool: float8_e4m3fn [..., 512], sliding-window KV rows (128 rows per page); bt_swa: contiguous int32
            [B, pages].
        c_pool: float8_e4m3fn [..., 512], compressed KV rows (32 rows per page); bt_c: contiguous int32 [B, pages].
        topk: int32 [B, 512] output, the selected compressed positions (in no particular order).
        out: bfloat16 [B, 64 * 512] output.
        softmax_scale: attention softmax scale (default: ``1 / sqrt(kv_dim)`` of the spec).
        q_scale: int32 [B, 64] (or [B, 1, 64]), the four UE8M0 query scales of each head in one int32.

    The arguments are checked before the first launch. Inside a CUDA-graph capture the logits kernel has to be
    compiled already (``precompile``, or an earlier launch outside the capture).
    """
    config = state.config
    b = state.batch_size
    heads, q_values = config.index_heads, config.index_head_dim // 2
    if q_scale is None:
        raise ValueError("an FP4 spec needs q_scale (the UE8M0 scales of the FP4 queries)")
    _check_numel("q_idx", q_idx, b * heads * q_values)
    if q_idx.element_size() != 1:
        raise ValueError(f"q_idx must be packed FP4 (1 byte per two values), got {q_idx.dtype}")
    _check_numel("q_scale", q_scale, b * heads, torch.int32)
    _check_numel("w_idx", w_idx, b * heads, torch.float32)
    _check_numel("ctx_lens", ctx_lens, b, torch.int32)
    if ctx_lens.device != q_idx.device:
        raise ValueError(
            f"ctx_lens must be on the device of q_idx ({q_idx.device}), got {ctx_lens.device}"
        )
    _check_numel("kv_lens", kv_lens, b, torch.int32)
    _check_numel("topk", topk, b * config.topk, torch.int32)
    for name, table in (("idx_block_table", idx_block_table), ("bt_swa", bt_swa), ("bt_c", bt_c)):
        _check_block_table(name, table, b)
    pages = int(k_cache.shape[0])
    if k_cache.element_size() != 1 or k_cache.numel() != pages * config.index_page_bytes:
        raise ValueError(
            f"k_cache must hold {config.index_page_bytes} bytes per page, got {tuple(k_cache.shape)}"
        )
    topk = topk.view(b, config.topk)
    kv_lens = kv_lens.view(b)
    # the tail attention's own checks, so that a bad argument is refused before the first launch
    tail_op._check_arguments(
        config, b, topk, q_att, swa_pool, bt_swa, c_pool, bt_c, kv_lens, state.sinks, out
    )
    k1 = _k1_op()
    device_index = _device_index(q_idx.device)
    if torch.cuda.is_current_stream_capturing() and device_index not in _k1_ready:
        raise RuntimeError(
            "spec_dsa tail hybrid: compile the logits kernel (precompile, or a launch outside the capture) before "
            "CUDA-graph capture"
        )
    page_tokens = config.index_page_tokens

    def logits_fn() -> torch.Tensor:
        logits = k1(
            q_idx.reshape(b, 1, heads, q_values).view(torch.uint8),
            q_scale.reshape(b, 1, heads),
            k_cache.view(pages, page_tokens, 1, config.index_page_bytes // page_tokens),
            w_idx.reshape(b, heads),
            state.ctx_lens,
            idx_block_table[:b],
            state.sched,
            config.max_index_len,
        )
        if not torch.cuda.is_current_stream_capturing():
            with _lock:
                _k1_ready.add(device_index)
        return logits

    tail_op.run_hybrid(
        logits_fn,
        None,
        topk,
        q_att,
        swa_pool,
        bt_swa,
        c_pool,
        bt_c,
        kv_lens,
        state.sinks,
        out,
        softmax_scale,
        config=config,
        ctx_lens=state.ctx_lens,
    )

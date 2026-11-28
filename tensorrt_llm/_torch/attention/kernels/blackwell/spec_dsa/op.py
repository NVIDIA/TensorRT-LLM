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
"""specDSA: fused decode sparse attention of a DeepSeek DSA model on B200 (SM100), any batch size from 1 to 1,024.

One decode step of a CSA layer: FP8 or MXFP4 paged-MQA indexer logits over the compressed indexer K cache, exact top-K
selection, mapping of the selected positions to compressed-KV rows, and the sparse MLA attention over those rows, the
sliding window and the attention sink. One call replaces the four kernels of the unfused chain. The megakernel guesses
the top-K threshold from a sample of the row's own logits (the sampling ladder of the GVR V2 top-K) and checks it with a
count certificate; a row where the guess does not hold takes an exact fallback. The large-batch kernels take no sample:
they learn a floor from the candidates already admitted in the row and select the exact top-K from the ones kept. The
tail-attention hybrid keeps the production indexer logits kernel and replaces the rest of the chain. The selection is
exact in all three.

The kernels are specialized for a ``SpecDsaConfig`` (``config.py``). The default spec is ``DS_V4_FLASH`` (top-512 over a
64 x 128 FP8 indexer, 64 attention heads over 512-byte E4M3 KV rows, a 128-row window); ``DS_V4_FLASH_FP4`` is the same
model with the MXFP4 indexer K cache. A config without kernels is refused with ``NotImplementedError``
(``is_supported`` and ``config.supported_spec`` tell beforehand) and its caller keeps the existing kernels.

Three implementations sit behind one interface; ``implementation_for`` names the one a batch size uses by default:

- ``megakernel`` (``megakernel.py`` / ``megakernel_op.py``, 1 to ``megakernel_max_batch`` rows): one cooperative launch
  that streams the indexer K cache once and does the selection and the attention on dedicated CTAs while the others
  keep streaming; compiled per batch size.
- ``tail_hybrid`` (``hybrid_op.py``, the batch sizes of ``tail_hybrid_range``, FP4 spec only): the indexer logits kernel
  of the production MXFP4 chain, the multi-CTA exact top-512 (``topk_multicta_op.py``) and the tail attention
  (``tail_op.py``), the last two by default with programmatic dependent launch up to 24 rows and a polling handshake
  up to 23 rows.
- ``large_batch`` (``large_batch.py`` / ``large_batch_op.py``, more rows): a persistent indexer kernel and an attention
  kernel launched behind it with programmatic dependent launch; one compile per knob range (``large_batch_op``)
  serves every batch size in the range.

| Spec | megakernel | tail hybrid | large batch |
| --- | --- | --- | --- |
| DS-V4-Flash (FP8) | 1 to 47 | none | 48 to 1,024 |
| DS-V4-Flash-FP4 | 1 to 4 | 5 to 26 | 27 to 1,024 |

The default is the hybrid inside the spec's ``tail_hybrid_range``, else the megakernel up to ``megakernel_max_batch``,
else the large-batch kernels; for the FP4 spec these are the fastest ones in the measurements against the production
FP4 chain (1M-token captures, B200; ``README.md``). ``prepare`` and ``precompile`` take ``implementation=`` to force one
that can serve the batch size, e.g. the megakernel (which serves up to 20 FP4 rows) at 5 to 20 rows; a state remembers
its implementation.

The hybrid is not self-contained: its logits stage is the production op
``torch.ops.trtllm.cute_dsl_fp4_paged_mqa_logits`` with the scheduler metadata of ``tensorrt_llm.deep_gemm``. Both are
resolved when first used; a missing one is a ``RuntimeError`` (``hybrid_op.is_supported``, or ``is_supported`` with a
``batch_size``, tells beforehand), with no fallback. The hybrid is tuned for rows of 1M tokens: its top-K is exact for
every length, but the faster top-K stage only for rows of 16,384 positions or more
(``topk_multicta_op.is_profitable``). For shorter rows the megakernel stays selectable at 5 to 20 rows
(``implementation="megakernel"``).

Per step, ``prepare`` does the host work (descriptors, CTA schedule or logits scheduler metadata from the lengths, the
sinks) and ``run`` launches on the current stream; the launch can be captured in a CUDA graph and replayed while
``prepare`` refreshes the state in place.

Contract (the numbers are those of DS-V4-Flash):
- A state belongs to one batch size, one spec, one implementation and one caller. ``prepare`` calls that refresh it and
  launches that share it must be stream-ordered. The tail hybrid's logits live in the process-global buffer of the
  production logits op: its launches, and that op's other users, must be stream-ordered also across states.
- ``ctx_lens`` (compressed indexer length of each row, in [512, 2**18]) and ``kv_lens`` (uncompressed length, at least
  128; the window is its last 128 positions) are read on the host by ``prepare``: pass host (CPU) tensors to avoid a
  device-to-host copy; device tensors work but synchronize. ``run`` takes both on the device. The megakernel does not
  read the ``kv_lens`` of ``run``, and the tail hybrid does not read its ``ctx_lens``: it uses the copy of the lengths
  that ``prepare`` keeps in the state.
- ``topk`` and ``out`` must not alias any input.
- The kernels need all 148 SMs; they fail at launch on a partitioned device (MPS limits, green contexts).
- Block tables must hold valid page ids for every page a row touches: the indexer table covers whole 128-position tiles
  (``4 * ceil(n / 128)`` entries per row), the compressed-KV table ``ceil(n / 32)`` entries. The tail hybrid and the
  large-batch kernels take contiguous ``idx_block_table``, ``bt_swa`` and ``bt_c`` only; the megakernel also takes
  strided ones.
- A launch whose state does not match ``ctx_lens`` (``prepare`` not refreshed) is refused or flagged through
  ``error_bits`` and its outputs are undefined. The tail hybrid has no device flag and does not detect it: it computes
  for the lengths ``prepare`` saw.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import hybrid_op, large_batch_op, megakernel_op
from .config import SpecDsaConfig, check_device, supported_spec

__all__ = [
    "ERR_STALE",
    "IMPLEMENTATIONS",
    "SpecDsaState",
    "error_bits",
    "implementation_for",
    "is_supported",
    "precompile",
    "prepare",
    "run",
]

# Error bit shared by the megakernel and the large-batch kernels: the state's lengths differ from the launch's
# ``ctx_lens`` (``prepare`` was not refreshed). The megakernel reports further bits per row (``megakernel_op.ERR_*``).
ERR_STALE = 1

_MEGAKERNEL = "megakernel"
_TAIL_HYBRID = "tail_hybrid"
_LARGE_BATCH = "large_batch"
IMPLEMENTATIONS = (_MEGAKERNEL, _TAIL_HYBRID, _LARGE_BATCH)


@dataclass
class SpecDsaState:
    """Per-step metadata and workspace of one caller; see ``prepare``.

    ``implementation`` names the kernels the state is for (``implementation_for`` of the batch size, or the one forced
    at ``prepare``), ``state`` is their state, ``sinks`` the float32 [attn_heads] attention-sink copy the megakernel
    launch reads (the large-batch and tail-hybrid states hold their own copy) and ``config`` the spec the state was
    built for.
    """

    batch_size: int
    implementation: str
    state: object
    sinks: torch.Tensor | None
    config: SpecDsaConfig


def implementation_for(batch_size: int, config: SpecDsaConfig | None = None) -> str:
    """Name of the implementation that serves ``batch_size`` rows of ``config`` (default: DS-V4-Flash) by default.

    ``"tail_hybrid"`` inside ``config.tail_hybrid_range``, else ``"megakernel"`` up to ``config.megakernel_max_batch``,
    else ``"large_batch"``. ``prepare`` and ``precompile`` can force another implementation that serves the batch size.
    """
    config = supported_spec(config)
    if not 1 <= batch_size <= config.max_batch:
        raise ValueError(f"batch size {batch_size} not in [1, {config.max_batch}]")
    rng = config.tail_hybrid_range
    if rng is not None and rng[0] <= batch_size <= rng[1]:
        return _TAIL_HYBRID
    return _MEGAKERNEL if batch_size <= config.megakernel_max_batch else _LARGE_BATCH


def _batch_sizes(implementation: str, config: SpecDsaConfig) -> range:
    """Batch sizes ``implementation`` can serve for ``config``."""
    if implementation == _MEGAKERNEL:
        return megakernel_op.supported_batch_sizes(config)
    if implementation == _TAIL_HYBRID:
        return hybrid_op.supported_batch_sizes(config)
    return range(1, config.max_batch + 1) if config.large_batch else range(0)


def _select(batch_size: int, config: SpecDsaConfig, implementation: str | None) -> str:
    """``implementation`` if it is given and can serve ``batch_size`` rows, else the default one."""
    default = implementation_for(batch_size, config)
    if implementation is None or implementation == default:
        return default
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(f"implementation must be one of {IMPLEMENTATIONS}, got {implementation!r}")
    sizes = _batch_sizes(implementation, config)
    if not sizes:
        raise ValueError(f"{config.name} has no {implementation} implementation")
    if batch_size not in sizes:
        raise ValueError(
            f"the {implementation} implementation of {config.name} serves {sizes.start} to {sizes.stop - 1} rows, "
            f"got {batch_size}"
        )
    return implementation


def is_supported(
    device: torch.device | None = None,
    config: SpecDsaConfig | None = None,
    batch_size: int | None = None,
) -> tuple[bool, str]:
    """Whether the kernels serve ``config`` (default: DS-V4-Flash) on ``device``, and the reason when they do not.

    That takes a spec with kernels, CuTe DSL installed and SM100 with all 148 SMs (one CTA per SM). The batch sizes of
    the tail hybrid also need the production logits op (``hybrid_op.is_supported``): pass ``batch_size`` to include that
    check when the hybrid is its default implementation; without it a True answer does not cover those sizes.
    """
    try:
        config = supported_spec(config)
    except NotImplementedError as e:
        return False, str(e)
    ok, why = check_device(device)
    if not ok or batch_size is None:
        return ok, why
    try:
        implementation = implementation_for(batch_size, config)
    except ValueError as e:
        return False, str(e)
    if implementation == _TAIL_HYBRID:
        return hybrid_op.is_supported(device, config)
    return True, ""


def precompile(
    device: torch.device,
    batch_size: int,
    config: SpecDsaConfig | None = None,
    *,
    implementation: str | None = None,
) -> None:
    """Compile what ``batch_size`` rows need on ``device`` (the megakernel per batch size, the large-batch kernels
    once, the tail hybrid's top-K and tail attention per cluster size and its logits kernel).

    ``implementation`` forces one that can serve the batch size (default: ``implementation_for``). Must run outside
    CUDA-graph capture; ``run`` compiles on first use otherwise.
    """
    config = supported_spec(config)
    implementation = _select(batch_size, config, implementation)
    if implementation == _MEGAKERNEL:
        megakernel_op.precompile(device, batch_size, config)
    elif implementation == _TAIL_HYBRID:
        hybrid_op.precompile(device, batch_size, config)
    else:
        large_batch_op.precompile(device, config)


def prepare(
    ctx_lens: torch.Tensor,
    kv_lens: torch.Tensor,
    idx_block_table: torch.Tensor,
    sinks: torch.Tensor,
    state: SpecDsaState | None = None,
    *,
    config: SpecDsaConfig | None = None,
    implementation: str | None = None,
) -> SpecDsaState:
    """Build (or refresh in place) the per-step state for ``B = ctx_lens.numel()`` rows.

    Args:
        ctx_lens: int32 [B], compressed (indexer) length of each row, in [512, 2**18].
        kv_lens: int32 [B], uncompressed length of each row (>= 128).
        idx_block_table: int32 [B, pages] on the target device, indexer K pages of each row (32 compressed positions
            per page; whole 128-position tiles).
        sinks: float32 [64] on the target device, per-head attention sink.
        state: a state of the same batch size to refresh; keeps its workspace, its spec and its implementation.
        config: the spec (default: DS-V4-Flash). A new state is built for it; a refreshed state keeps its own.
        implementation: ``"megakernel"``, ``"tail_hybrid"`` or ``"large_batch"``: force that one for a new state (it has
            to be able to serve the batch size, e.g. the megakernel for 5 to 20 rows of the FP4 spec). Default: the
            implementation of ``implementation_for``, or, when refreshing, the one of ``state``. A request that differs
            from the implementation of ``state`` is an error.
    """
    if state is not None:
        if config is not None and config.kernel_key != state.config.kernel_key:
            raise ValueError(f"state is for {state.config.name!r}, got {config.name!r}")
        config = state.config
    config = supported_spec(config)
    if implementation is not None and implementation not in IMPLEMENTATIONS:
        raise ValueError(f"implementation must be one of {IMPLEMENTATIONS}, got {implementation!r}")
    batch_size = int(ctx_lens.numel())
    if implementation is None and state is not None and state.batch_size == batch_size:
        implementation = state.implementation
    else:
        implementation = _select(batch_size, config, implementation)
    if sinks.numel() != config.attn_heads:
        raise ValueError(f"sinks must have {config.attn_heads} entries, got {tuple(sinks.shape)}")
    if state is not None and (
        state.batch_size != batch_size or state.implementation != implementation
    ):
        raise ValueError(
            f"state is for {state.batch_size} rows ({state.implementation}), got {batch_size} rows ({implementation})"
        )
    inner = state.state if state is not None else None
    sinks_copy = None
    if implementation == _MEGAKERNEL:
        inner = megakernel_op.prepare(ctx_lens, kv_lens, idx_block_table, inner, config=config)
        sinks_copy = (
            state.sinks
            if state is not None
            else torch.empty(config.attn_heads, dtype=torch.float32, device=sinks.device)
        )
        sinks_copy.copy_(sinks.reshape(-1))
    elif implementation == _TAIL_HYBRID:
        inner = hybrid_op.prepare(ctx_lens, kv_lens, idx_block_table, sinks, inner, config=config)
    else:
        if int(kv_lens.numel()) != batch_size:
            raise ValueError(f"kv_lens must have {batch_size} entries, got {int(kv_lens.numel())}")
        if idx_block_table.dim() != 2 or idx_block_table.shape[0] < batch_size:
            raise ValueError(f"idx_block_table must be [B, pages] with B >= {batch_size}")
        inner = large_batch_op.prepare(ctx_lens, sinks, inner, config=config)
    if state is not None:
        return state
    return SpecDsaState(
        batch_size=batch_size,
        implementation=implementation,
        state=inner,
        sinks=sinks_copy,
        config=config,
    )


def error_bits(state: SpecDsaState) -> torch.Tensor:
    """Error bits of the launches of ``state``: a device view, no sync; zero means no error.

    The megakernel keeps one sticky int64 word per row (``megakernel_op.ERR_*``); the large-batch kernels keep one int32
    status word of the last launch (``ERR_STALE``; a fresh state reads -7 until its first launch); the tail hybrid has
    no device flag and returns an int32 [1] zero (a stale state is not detected).
    """
    if state.implementation == _MEGAKERNEL:
        return megakernel_op.error_bits(state.state)
    if state.implementation == _TAIL_HYBRID:
        return hybrid_op.error_bits(state.state)
    return large_batch_op.error_bits(state.state)


def run(
    state: SpecDsaState,
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
    """Launch the fused kernels on the current stream.

    Args:
        state: from ``prepare`` for these lengths and block tables.
        q_idx: float8_e4m3fn [B, 64, 128] (or [B, 1, 64, 128]), indexer queries.
        w_idx: float32 [B, 64], indexer head weights.
        k_cache: uint8 [pages, ...] with 4224 bytes per page, indexer K pages (32 x 128 FP8 values, then 32 float32
            scales).
        idx_block_table: int32 [B, pages], as given to ``prepare``.
        ctx_lens: int32 [B] on the device, the lengths given to ``prepare`` (not read by the tail hybrid).
        kv_lens: int32 [B] on the device, uncompressed lengths (the window is the last 128 positions); not read by the
            megakernel.
        q_att: float8_e4m3fn [B, 64, 512], attention queries.
        swa_pool: float8_e4m3fn [..., 512], sliding-window KV rows (128 rows per page); bt_swa: int32 [B, pages].
        c_pool: float8_e4m3fn [..., 512], compressed KV rows (32 rows per page); bt_c: int32 [B, pages]. The megakernel
            takes strided block tables (``bt_c``: each row 16-byte aligned, row stride a multiple of 4 entries), the
            tail hybrid and the large-batch kernels contiguous ones.
        topk: int32 [B, 512] output, the selected compressed positions (in no particular order).
        out: bfloat16 [B, 64 * 512] output.
        softmax_scale: attention softmax scale (default: ``1 / sqrt(kv_dim)`` of the spec of ``state``).
        q_scale: int32 [B, 64], the packed UE8M0 scales of an FP4 ``q_idx`` (FP4 specs only: ``q_idx`` and ``k_cache``
            are FP4 then).

    The shapes are those of DS-V4-Flash; ``state.config`` gives the sizes the checks use. For the FP4 spec ``k_cache``
    has 2176 bytes per page (32 x 64 bytes of E2M1 pairs, then 32 int32 of four UE8M0 scales) and ``q_idx`` holds
    1 byte per two values ([B, 64, 64]).
    """
    if state.implementation == _MEGAKERNEL:
        megakernel_op.run(
            state.state,
            q_idx,
            w_idx,
            k_cache,
            idx_block_table,
            ctx_lens,
            q_att,
            swa_pool,
            bt_swa,
            c_pool,
            bt_c,
            state.sinks,
            topk,
            out,
            softmax_scale,
            q_scale=q_scale,
        )
    elif state.implementation == _TAIL_HYBRID:
        hybrid_op.run(
            state.state,
            q_idx,
            w_idx,
            k_cache,
            idx_block_table,
            ctx_lens,
            kv_lens,
            q_att,
            swa_pool,
            bt_swa,
            c_pool,
            bt_c,
            topk,
            out,
            softmax_scale,
            q_scale=q_scale,
        )
    else:
        large_batch_op.run(
            state.state,
            q_idx,
            w_idx,
            k_cache,
            idx_block_table,
            ctx_lens,
            kv_lens,
            q_att,
            swa_pool,
            bt_swa,
            c_pool,
            bt_c,
            topk,
            out,
            softmax_scale,
            q_scale=q_scale,
        )

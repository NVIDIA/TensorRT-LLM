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
selection is exact in both.

The kernels are specialized for a ``SpecDsaConfig`` (``config.py``). The default spec is ``DS_V4_FLASH`` (top-512 over a
64 x 128 FP8 indexer, 64 attention heads over 512-byte E4M3 KV rows, a 128-row window); ``DS_V4_FLASH_FP4`` is the same
model with the MXFP4 indexer K cache. A config without kernels is refused with ``NotImplementedError``
(``is_supported`` and ``config.supported_spec`` tell beforehand) and its caller keeps the existing kernels.

Two implementations sit behind one interface; ``implementation_for`` names the one a batch size uses:

- ``megakernel`` (``megakernel.py`` / ``megakernel_op.py``, 1 to ``megakernel_max_batch`` rows): one cooperative launch
  that streams the indexer K cache once and does the selection and the attention on dedicated CTAs while the others
  keep streaming; compiled per batch size.
- ``large_batch`` (``large_batch.py`` / ``large_batch_op.py``, more rows): a persistent indexer kernel and an attention
  kernel launched behind it with programmatic dependent launch; one compile per knob range (``large_batch_op``)
  serves every batch size in the range.

Per step, ``prepare`` does the host work (descriptors or CTA schedule from the lengths, the sinks) and ``run`` launches
on the current stream; the launch can be captured in a CUDA graph and replayed while ``prepare`` refreshes the state in
place.

Contract (the numbers are those of DS-V4-Flash):
- A state belongs to one batch size, one spec and one caller. ``prepare`` calls that refresh it and launches that share
  it must be stream-ordered.
- ``ctx_lens`` (compressed indexer length of each row, in [512, 2**18]) and ``kv_lens`` (uncompressed length, at least
  128; the window is its last 128 positions) are read on the host by ``prepare``: pass host (CPU) tensors to avoid a
  device-to-host copy; device tensors work but synchronize. ``run`` takes both on the device.
- ``topk`` and ``out`` must not alias any input.
- The kernels need all 148 SMs; they fail at launch on a partitioned device (MPS limits, green contexts).
- Block tables must hold valid page ids for every page a row touches: the indexer table covers whole 128-position tiles
  (``4 * ceil(n / 128)`` entries per row), the compressed-KV table ``ceil(n / 32)`` entries.
- A launch whose state does not match ``ctx_lens`` (``prepare`` not refreshed) is refused or flagged through
  ``error_bits`` and its outputs are undefined.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import large_batch_op, megakernel_op
from .config import SpecDsaConfig, check_device, supported_spec

__all__ = [
    "ERR_STALE",
    "SpecDsaState",
    "error_bits",
    "implementation_for",
    "is_supported",
    "precompile",
    "prepare",
    "run",
]

# Error bit shared by both implementations: the state's lengths differ from the launch's ``ctx_lens`` (``prepare`` was
# not refreshed). The megakernel reports further bits per row (``megakernel_op.ERR_*``).
ERR_STALE = 1

_MEGAKERNEL = "megakernel"
_LARGE_BATCH = "large_batch"


@dataclass
class SpecDsaState:
    """Per-step metadata and workspace of one caller; see ``prepare``.

    ``implementation`` names the kernels the batch size uses, ``state`` is their state, ``sinks`` the float32
    [attn_heads] attention-sink copy the megakernel launch reads (the large-batch state holds its own per-row copy) and
    ``config`` the spec the state was built for.
    """

    batch_size: int
    implementation: str
    state: object
    sinks: torch.Tensor | None
    config: SpecDsaConfig


def implementation_for(batch_size: int, config: SpecDsaConfig | None = None) -> str:
    """Name of the implementation that serves ``batch_size`` rows of ``config`` (default: DS-V4-Flash).

    ``"megakernel"`` or ``"large_batch"``.
    """
    config = supported_spec(config)
    if not 1 <= batch_size <= config.max_batch:
        raise ValueError(f"batch size {batch_size} not in [1, {config.max_batch}]")
    return _MEGAKERNEL if batch_size <= config.megakernel_max_batch else _LARGE_BATCH


def is_supported(
    device: torch.device | None = None, config: SpecDsaConfig | None = None
) -> tuple[bool, str]:
    """Whether the kernels serve ``config`` (default: DS-V4-Flash) on ``device``, and the reason when they do not.

    That takes a spec with kernels, CuTe DSL installed and SM100 with all 148 SMs (one CTA per SM).
    """
    try:
        supported_spec(config)
    except NotImplementedError as e:
        return False, str(e)
    return check_device(device)


def precompile(device: torch.device, batch_size: int, config: SpecDsaConfig | None = None) -> None:
    """Compile what ``batch_size`` rows need on ``device`` (the megakernel per batch size, the large-batch
    kernels once).

    Must run outside CUDA-graph capture; ``run`` compiles on first use otherwise.
    """
    config = supported_spec(config)
    if implementation_for(batch_size, config) == _MEGAKERNEL:
        megakernel_op.precompile(device, batch_size, config)
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
) -> SpecDsaState:
    """Build (or refresh in place) the per-step state for ``B = ctx_lens.numel()`` rows.

    Args:
        ctx_lens: int32 [B], compressed (indexer) length of each row, in [512, 2**18].
        kv_lens: int32 [B], uncompressed length of each row (>= 128).
        idx_block_table: int32 [B, pages] on the target device, indexer K pages of each row (32 compressed positions
            per page; whole 128-position tiles).
        sinks: float32 [64] on the target device, per-head attention sink.
        state: a state of the same batch size to refresh; keeps its workspace and its spec.
        config: the spec (default: DS-V4-Flash). A new state is built for it; a refreshed state keeps its own.
    """
    if state is not None:
        if config is not None and config.kernel_key != state.config.kernel_key:
            raise ValueError(f"state is for {state.config.name!r}, got {config.name!r}")
        config = state.config
    config = supported_spec(config)
    batch_size = int(ctx_lens.numel())
    implementation = implementation_for(batch_size, config)
    if sinks.numel() != config.attn_heads:
        raise ValueError(f"sinks must have {config.attn_heads} entries, got {tuple(sinks.shape)}")
    if state is not None and (
        state.batch_size != batch_size or state.implementation != implementation
    ):
        raise ValueError(
            f"state is for {state.batch_size} rows ({state.implementation}), got {batch_size} rows ({implementation})"
        )
    inner = state.state if state is not None else None
    if implementation == _MEGAKERNEL:
        inner = megakernel_op.prepare(ctx_lens, kv_lens, idx_block_table, inner, config=config)
        sinks_copy = (
            state.sinks
            if state is not None
            else torch.empty(config.attn_heads, dtype=torch.float32, device=sinks.device)
        )
        sinks_copy.copy_(sinks.reshape(-1))
    else:
        if int(kv_lens.numel()) != batch_size:
            raise ValueError(f"kv_lens must have {batch_size} entries, got {int(kv_lens.numel())}")
        if idx_block_table.dim() != 2 or idx_block_table.shape[0] < batch_size:
            raise ValueError(f"idx_block_table must be [B, pages] with B >= {batch_size}")
        inner = large_batch_op.prepare(ctx_lens, sinks, inner, config=config)
        sinks_copy = None
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
    status word of the last launch (``ERR_STALE``; a fresh state reads -7 until its first launch).
    """
    if state.implementation == _MEGAKERNEL:
        return megakernel_op.error_bits(state.state)
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
        ctx_lens: int32 [B] on the device, the lengths given to ``prepare``.
        kv_lens: int32 [B] on the device, uncompressed lengths (the window is the last 128 positions).
        q_att: float8_e4m3fn [B, 64, 512], attention queries.
        swa_pool: float8_e4m3fn [..., 512], sliding-window KV rows (128 rows per page); bt_swa: int32 [B, pages].
        c_pool: float8_e4m3fn [..., 512], compressed KV rows (32 rows per page); bt_c: int32 [B, pages], each row
            16-byte aligned (row stride a multiple of 4 entries).
        topk: int32 [B, 512] output, the selected compressed positions (in no particular order).
        out: bfloat16 [B, 64 * 512] output.
        softmax_scale: attention softmax scale (default: ``1 / sqrt(kv_dim)`` of the spec of ``state``).
        q_scale: int32 [B, 64], the packed UE8M0 scales of an FP4 ``q_idx`` (FP4 specs only: ``q_idx`` and ``k_cache``
            are FP4 then).

    The shapes are those of DS-V4-Flash; ``state.config`` gives the sizes the checks use.
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

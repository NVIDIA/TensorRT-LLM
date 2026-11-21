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
"""specDSA multi-CTA exact top-512, host side: the top-K stage of the tail-attention hybrid (``tail_op.run_hybrid``).

``run`` selects the exact top-512 positions of every row of the fp32 indexer logits (``logits`` [B, npad], row ``r``
valid up to ``ctx_lens[r]``) with one cluster of CTAs per row (``topk_multicta.py``) and writes them to ``topk``
[B, 512] int32, the input of the tail attention: local positions in no particular order, and ``0 .. n - 1`` followed
by -1 for a row of ``n <= 512`` positions, as the production top-K does. The result is exact for every input (its
values are the 512 largest of the row; which positions that share the 512th value are taken is not specified): a row
whose sampled threshold does not certify the fast path takes an in-kernel exact fallback (a radix select over the
row), and rows of at most 8192 positions, which have no sample, always do.

Dispatch. Rows of 16,384 positions or more (``MIN_PROFITABLE_ROW_LEN``, 64K tokens at the compression ratio 4) take
the fast path, and the kernel is faster than the production top-K behind the same logits kernel (B200, FP4 indexer,
1M-token captures: 0.8 to 2.8 us of the 9 to 26 row hybrid, +1.3 to +3.5 % of its step; one top-K call +0.8 to +3.7
us in 20 of 21 cases of BS 12 / 16 / 24 x 7 lengths, -0.3 us in one). Below that length the sample has few windows
(one per 896 positions) or none (rows of at most 8192 positions always take the radix path), and the kernel is not
reliably faster: at BS 24 it is 2 to 4 us slower for rows up to 12K positions, at BS 12 to 16 it is up to 1 us slower
for 8193 to 12K positions and 0.6 to 2.5 us faster for rows of at most 8192. A row that takes the fallback costs ~6
us more (rare on real logits: none of 630 captured rows) and a batch is as slow as its slowest row, so
``is_profitable`` asks for every row to be long enough. The caller keeps its production top-K for the other batches
(``tail_op.run_hybrid`` takes it as ``topk_fn``).

Contract: ``logits`` float32 [B, npad] with unit column stride, a row stride that is a multiple of 128 floats (the
logits arena of the indexer kernel has 2**18) and a 16-byte aligned base, ``ctx_lens`` at most ``npad`` (the lengths
the logits kernel was given); ``topk`` int32 [B, 512] contiguous; ``ctx_lens`` int32 [B]; all on one CUDA device. B
is 1 to 26 rows (the batch sizes of the tail attention): a cluster of 8 / 6 / 5 / 4 CTAs serves a row for 1-12 /
13-16 / 17-24 / 25-26 rows, and the clusters of a launch have to fit one wave of co-resident clusters on the 148 SMs.
The kernel reads the logits only, so both indexer K dtypes are served (``config``: either supported spec, default
DS-V4-Flash). ``topk`` must not alias an input.
"""

from __future__ import annotations

import threading

import torch

from .config import SpecDsaConfig, check_device, supported_spec

__all__ = [
    "MAX_BATCH",
    "MIN_PROFITABLE_ROW_LEN",
    "cluster_size",
    "is_batch_size_supported",
    "is_profitable",
    "is_supported",
    "precompile",
    "run",
]

# (largest batch size, CTAs per row): the clusters of one launch (one CTA per SM, ~170 KB of shared memory each) have to
# co-reside in one wave. A B200 holds (measured) 15 clusters of 8 CTAs, 22 of 6, 26 of 5 and 33 of 4; these sizes put
# 96, 96, 120 and 104 CTAs at the largest batch of each.
_CLUSTER_LIMITS = ((12, 8), (16, 6), (24, 5), (26, 4))
MAX_BATCH = _CLUSTER_LIMITS[-1][0]
# Compressed positions per row from which the kernel is faster than the production top-K (see the module docstring).
MIN_PROFITABLE_ROW_LEN = 16384
_ROW_ALIGN = 128  # floats: the row stride of ``logits``

_lock = threading.Lock()
_compiled: dict[tuple, object] = {}


def cluster_size(batch_size: int) -> int | None:
    """CTAs per row for ``batch_size`` rows, None when the batch is not in [1, ``MAX_BATCH``]."""
    for max_batch, g in _CLUSTER_LIMITS:
        if 1 <= batch_size <= max_batch:
            return g
    return None


def _cluster(batch_size: int) -> int:
    g = cluster_size(batch_size)
    if g is None:
        raise ValueError(f"batch size {batch_size} not in [1, {MAX_BATCH}]")
    return g


def is_batch_size_supported(batch_size: int) -> tuple[bool, str]:
    """Whether the kernel serves ``batch_size`` rows (1 to ``MAX_BATCH``), and why not if it does not."""
    if cluster_size(batch_size) is None:
        return False, f"batch size {batch_size} not in [1, {MAX_BATCH}]"
    return True, ""


def is_supported(
    device: torch.device | None = None, config: SpecDsaConfig | None = None
) -> tuple[bool, str]:
    """Whether the kernel serves ``config`` (default: DS-V4-Flash) on ``device``, and why not if it does not.

    That takes a supported spec (either indexer K dtype), CuTe DSL installed and SM100 with all 148 SMs (clusters of
    up to 8 CTAs at one CTA per SM). The batch size is checked by ``is_batch_size_supported``.
    """
    try:
        supported_spec(config)
    except NotImplementedError as e:
        return False, str(e)
    return check_device(device)


def is_profitable(ctx_lens: torch.Tensor) -> bool:
    """Whether the kernel is the faster top-K for rows of these lengths: every row has at least
    ``MIN_PROFITABLE_ROW_LEN`` positions (see the module docstring).

    Pass the host (CPU) tensor of the step's lengths: a device tensor is read back and synchronizes.
    """
    return ctx_lens.numel() > 0 and bool((ctx_lens >= MIN_PROFITABLE_ROW_LEN).all())


def _device_index(device: torch.device) -> int:
    device = torch.device(device)
    return device.index if device.index is not None else torch.cuda.current_device()


def _fn(device: torch.device, g: int, force_fallback: bool = False, trigger: bool = False):
    """The compiled kernel of cluster size ``g`` on ``device``, compiled on first use (one compile per device, cluster
    size and variant). ``force_fallback`` selects the variant that runs the exact radix select on every row: a hook
    for the tests, ``run`` never uses it. ``trigger`` selects the variant that launches the dependent grid after the
    stream phase (see ``run``)."""
    from . import topk_multicta as TM

    key = (_device_index(device), g, force_fallback, trigger)
    with _lock:
        if key not in _compiled:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "spec_dsa multi-CTA top-K: compile (precompile) before CUDA-graph capture"
                )
            _compiled[key] = TM.compile_topk_multicta(TM.TopkMultiCta(g, force_fallback, trigger))
        return _compiled[key]


def precompile(
    device: torch.device,
    batch_size: int,
    config: SpecDsaConfig | None = None,
    trigger: bool = False,
) -> None:
    """Compile the kernel of ``batch_size`` rows on ``device`` (one compile per cluster size, 8, 6, 5 and 4 CTAs, and
    per ``trigger`` value: precompile the variant ``run`` will launch).

    Must run outside CUDA-graph capture; ``run`` compiles on first use otherwise.
    """
    supported_spec(config)
    _fn(device, _cluster(batch_size), False, trigger)


def _require(ok: bool, what: str, t: torch.Tensor) -> None:
    if not ok:
        raise ValueError(f"{what}, got {t.dtype} {tuple(t.shape)} on {t.device}")


def _check_arguments(config, logits, topk, ctx_lens) -> int:
    """Raise ValueError for an argument the kernel cannot take (before any device work); returns the batch size."""
    _require(
        logits.dtype == torch.float32 and logits.dim() == 2 and logits.stride(1) == 1,
        "logits must be a float32 [B, npad] tensor with unit column stride",
        logits,
    )
    b, cols = logits.shape
    if not 1 <= b <= MAX_BATCH:
        raise ValueError(
            f"batch size {b} not in [1, {MAX_BATCH}]: the clusters of more rows no longer fit one wave, "
            "use the production top-K"
        )
    _require(
        1 <= cols <= config.max_index_len,
        f"logits must have 1 to {config.max_index_len} columns",
        logits,
    )
    _require(
        logits.stride(0) % _ROW_ALIGN == 0 and logits.stride(0) >= cols,
        f"logits row stride must be a multiple of {_ROW_ALIGN} floats and at least the row length",
        logits,
    )
    _require(
        topk.dtype == torch.int32
        and tuple(topk.shape) == (b, config.topk)
        and topk.is_contiguous(),
        f"topk must be a contiguous int32 [{b}, {config.topk}] tensor",
        topk,
    )
    _require(
        ctx_lens.dtype == torch.int32
        and tuple(ctx_lens.shape) == (b,)
        and ctx_lens.is_contiguous(),
        f"ctx_lens must be a contiguous int32 [{b}] tensor",
        ctx_lens,
    )
    if logits.device.type != "cuda":
        raise ValueError(f"the tensors must be CUDA tensors, got logits on {logits.device}")
    for name, t in (("topk", topk), ("ctx_lens", ctx_lens)):
        if t.device != logits.device:
            raise ValueError(
                f"{name} must be on the device of logits ({logits.device}), got {t.device}"
            )
    return b


def run(
    logits: torch.Tensor,
    topk: torch.Tensor,
    ctx_lens: torch.Tensor,
    *,
    config: SpecDsaConfig | None = None,
    trigger: bool = False,
) -> None:
    """Launch the exact top-512 selection on the current stream (see the module docstring for the contract).

    Allocates nothing, never synchronizes with the host and can be captured in a CUDA graph once compiled
    (``precompile``; outside a capture the first call compiles).

    Args:
        logits: float32 [B, npad], the indexer logits; unit column stride, row stride a multiple of 128 floats,
            16-byte aligned base, B in [1, ``MAX_BATCH``].
        topk: int32 [B, 512] output: the positions of the 512 largest values of every row (see the module
            docstring for the order and the ties).
        ctx_lens: int32 [B] on the device, the valid length of each row in positions (at most ``npad``); a row of
            at most 512 positions gets ``0 .. n - 1`` and -1 entries.
        config: the spec (default: DS-V4-Flash); either indexer K dtype: the launch is the same.
        trigger: let the dependent grid launch as soon as the stream phase of every CTA is done (programmatic dependent
            launch): set it only when the next kernel on the stream is launched with the programmatic attribute and
            waits for this grid before it reads ``topk`` (``tail_op.run_attention(pdl=True)``); the results are the
            same.
    """
    config = supported_spec(config)
    b = _check_arguments(config, logits, topk, ctx_lens)
    _fn(logits.device, _cluster(b), False, trigger)(logits, ctx_lens, topk)

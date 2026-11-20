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
"""specDSA tail attention, host side: the multi-CTA sparse MLA (K4') that replaces the K3 (local-to-global mapping) +
K4 (sparse MLA) + K4b (reduction) kernels of the unfused decode chain behind any exact top-K kernel, and ``run_hybrid``
(the caller's indexer logits kernel and top-K kernel, then K4').

``run_attention`` takes the exact top-512 selection (``topk`` [B, 512] int32, local compressed positions in no
particular order, -1 for an unused entry) and writes ``out`` [B, 64 * 512] bf16 like ``AttentionTail`` of the
large-batch kernels does. One cluster of 5 CTAs serves a row, up to 26 rows (the clusters of 5 that a B200 holds at
once at one CTA per SM). The launch is a plain stream-ordered launch (no programmatic dependent launch: the production
top-K kernels never trigger it, and the attribute then costs 0.7 to 0.8 us).

The kernel reads neither the indexer nor its K cache, so it is the same for the specs with an FP8 and an MXFP4 indexer
K cache: one compile per device serves both, and ``config`` (default: DS-V4-Flash) only has to be a supported spec.

Contract: ``q_att`` float8_e4m3fn [B, 64, 512] (any shape with that many elements); ``swa_pool`` / ``c_pool``
float8_e4m3fn [..., 512] rows (128 / 32 rows per page); ``bt_swa`` / ``bt_c`` contiguous int32 [B, pages]; ``kv_lens``
int32 [B] on the device (the window is the last min(128, kv_len) positions); ``sinks`` float32 [64]; ``out`` bfloat16
with B * 64 * 512 elements; all of them on one CUDA device. Row b of the batch is request b (the block tables are
indexed by the row): the specDSA operators have the same contract. ``out`` must not alias an input.
"""

from __future__ import annotations

import threading

import torch

from .config import SpecDsaConfig, check_device, supported_spec

__all__ = [
    "TAIL_MAX_BATCH",
    "is_batch_size_supported",
    "is_supported",
    "precompile",
    "run_attention",
    "run_hybrid",
    "tail_cluster_size",
]

# (largest batch size, CTAs per row): the co-resident clusters of a 148-SM B200 at one CTA per SM (measured) are 26
# clusters of 5. Clusters of 4 / 3 would fit 33 / 45 rows but win nothing over the large-batch kernels there and leave
# only 0.5 to 0.8 % of the numerical headroom: not offered.
_CLUSTER_LIMITS = ((26, 5),)
TAIL_MAX_BATCH = _CLUSTER_LIMITS[-1][0]

_lock = threading.Lock()
_compiled: dict[tuple, object] = {}


def tail_cluster_size(batch_size: int) -> int | None:
    """CTAs per row of the tail attention for ``batch_size`` rows, None when the batch is above ``TAIL_MAX_BATCH``."""
    for max_batch, g in _CLUSTER_LIMITS:
        if 1 <= batch_size <= max_batch:
            return g
    return None


def _cluster_size(batch_size: int) -> int:
    g = tail_cluster_size(batch_size)
    if g is None:
        raise ValueError(f"batch size {batch_size} not in [1, {TAIL_MAX_BATCH}]")
    return g


def is_batch_size_supported(batch_size: int) -> tuple[bool, str]:
    """Whether the tail attention serves ``batch_size`` rows (1 to ``TAIL_MAX_BATCH``), and why not if it does not."""
    if tail_cluster_size(batch_size) is None:
        return False, f"batch size {batch_size} not in [1, {TAIL_MAX_BATCH}]"
    return True, ""


def is_supported(
    device: torch.device | None = None, config: SpecDsaConfig | None = None
) -> tuple[bool, str]:
    """Whether the tail attention serves ``config`` (default: DS-V4-Flash) on ``device``, and why not if it does not.

    That takes a supported spec (either indexer K dtype), CuTe DSL installed and SM100 with all 148 SMs (one CTA per
    SM, clusters of 5). The batch size is checked by ``is_batch_size_supported``.
    """
    try:
        supported_spec(config)
    except NotImplementedError as e:
        return False, str(e)
    return check_device(device)


def _device_index(device: torch.device) -> int:
    device = torch.device(device)
    return device.index if device.index is not None else torch.cuda.current_device()


def _fn(device: torch.device, g: int):
    from . import tail_attention as TA

    key = (_device_index(device), g)
    with _lock:
        if key not in _compiled:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "spec_dsa tail attention: compile (precompile) before CUDA-graph capture"
                )
            _compiled[key] = TA.compile_tail_attention(TA.TailAttention(g))
        return _compiled[key]


def precompile(device: torch.device, batch_size: int, config: SpecDsaConfig | None = None) -> None:
    """Compile the tail attention kernel of ``batch_size`` rows on ``device``.

    The compiled kernel serves every batch size with the same cluster size and both indexer K dtypes. Must run outside
    CUDA-graph capture; ``run_attention`` compiles on first use otherwise.
    """
    supported_spec(config)
    _fn(device, _cluster_size(batch_size))


def _require(ok: bool, what: str, t: torch.Tensor) -> None:
    if not ok:
        raise ValueError(f"{what}, got {t.dtype} {tuple(t.shape)} on {t.device}")


def _check_arguments(config, b, topk, q_att, swa_pool, bt_swa, c_pool, bt_c, kv_lens, sinks, out):
    """Raise ValueError for an argument the kernel cannot take (before any device work)."""
    heads, dim = config.attn_heads, config.kv_dim
    _require(
        topk.dtype == torch.int32
        and tuple(topk.shape) == (b, config.topk)
        and topk.is_contiguous(),
        f"topk must be a contiguous int32 [B, {config.topk}] tensor",
        topk,
    )
    _require(
        q_att.element_size() == 1 and q_att.numel() == b * heads * dim and q_att.is_contiguous(),
        f"q_att must be a contiguous FP8 tensor with {b} x {heads} x {dim} elements",
        q_att,
    )
    for name, pool in (("swa_pool", swa_pool), ("c_pool", c_pool)):
        _require(
            pool.element_size() == 1
            and pool.dim() >= 2
            and pool.shape[-1] == dim
            and pool.is_contiguous(),
            f"{name} must be a contiguous FP8 tensor with {dim}-byte rows",
            pool,
        )
    for name, table in (("bt_swa", bt_swa), ("bt_c", bt_c)):
        _require(
            table.dtype == torch.int32
            and table.dim() == 2
            and table.shape[0] >= b
            and table.is_contiguous(),
            f"{name} must be a contiguous int32 [B, pages] tensor with B >= {b}",
            table,
        )
    _require(
        kv_lens.dtype == torch.int32 and kv_lens.numel() == b and kv_lens.is_contiguous(),
        f"kv_lens must be a contiguous int32 tensor with {b} elements",
        kv_lens,
    )
    _require(
        sinks.dtype == torch.float32 and sinks.numel() == heads and sinks.is_contiguous(),
        f"sinks must be a contiguous float32 tensor with {heads} elements",
        sinks,
    )
    _require(
        out.dtype == torch.bfloat16 and out.numel() == b * heads * dim and out.is_contiguous(),
        f"out must be a contiguous bfloat16 tensor with {b} x {heads} x {dim} elements",
        out,
    )
    if q_att.device.type != "cuda":
        raise ValueError(f"the tensors must be CUDA tensors, got q_att on {q_att.device}")
    named = (
        ("topk", topk),
        ("swa_pool", swa_pool),
        ("bt_swa", bt_swa),
        ("c_pool", c_pool),
        ("bt_c", bt_c),
        ("kv_lens", kv_lens),
        ("sinks", sinks),
        ("out", out),
    )
    for name, t in named:
        if t.device != q_att.device:
            raise ValueError(
                f"{name} must be on the device of q_att ({q_att.device}), got {t.device}"
            )


def run_attention(
    topk: torch.Tensor,
    q_att: torch.Tensor,
    swa_pool: torch.Tensor,
    bt_swa: torch.Tensor,
    c_pool: torch.Tensor,
    bt_c: torch.Tensor,
    kv_lens: torch.Tensor,
    sinks: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float | None = None,
    *,
    config: SpecDsaConfig | None = None,
) -> None:
    """Launch the tail attention on the current stream (see the module docstring for the contract).

    Args:
        topk: int32 [B, 512], the exact top-512 selection of each row: local compressed positions in no particular
            order, -1 for an unused entry. B = ``topk.shape[0]`` must be in [1, ``TAIL_MAX_BATCH``].
        q_att: float8_e4m3fn [B, 64, 512] (or [B, 64 * 512]), attention queries.
        swa_pool: float8_e4m3fn [..., 512], sliding-window KV rows (128 rows per page); bt_swa: int32 [B, pages].
        c_pool: float8_e4m3fn [..., 512], compressed KV rows (32 rows per page); bt_c: int32 [B, pages].
        kv_lens: int32 [B] on the device, uncompressed length of each row; the window is its last min(128, kv_len)
            positions.
        sinks: float32 [64], per-head attention sink.
        out: bfloat16 [B, 64 * 512] output.
        softmax_scale: attention softmax scale (default: ``1 / sqrt(kv_dim)`` of the spec).
        config: the spec (default: DS-V4-Flash); either indexer K dtype: the launch is the same.
    """
    config = supported_spec(config)
    if topk.dim() != 2:
        raise ValueError(
            f"topk must be a contiguous int32 [B, {config.topk}] tensor, got {tuple(topk.shape)}"
        )
    b = int(topk.shape[0])
    g = _cluster_size(b)
    _check_arguments(config, b, topk, q_att, swa_pool, bt_swa, c_pool, bt_c, kv_lens, sinks, out)
    if softmax_scale is None:
        softmax_scale = config.default_softmax_scale
    fn = _fn(q_att.device, g)
    fn(
        q_att.view(torch.uint8).view(b, config.attn_heads, config.kv_dim),
        swa_pool.view(torch.uint8).view(-1, config.kv_dim),
        c_pool.view(torch.uint8).view(-1, config.kv_dim),
        topk,
        kv_lens,
        bt_swa[:b],
        bt_c[:b],
        sinks.reshape(-1),
        out.view(b, config.attn_heads, config.kv_dim),
        softmax_scale,
        b,
    )


def run_hybrid(
    k1_logits_fn,
    topk_fn,
    topk: torch.Tensor,
    q_att: torch.Tensor,
    swa_pool: torch.Tensor,
    bt_swa: torch.Tensor,
    c_pool: torch.Tensor,
    bt_c: torch.Tensor,
    kv_lens: torch.Tensor,
    sinks: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float | None = None,
    *,
    config: SpecDsaConfig | None = None,
) -> None:
    """The decode chain with K3 + K4 + K4b replaced: ``k1_logits_fn()``, ``topk_fn(logits)`` (fills ``topk``), then K4'.

    The arguments of the tail attention and ``config`` are those of ``run_attention``.
    """
    logits = k1_logits_fn()
    topk_fn(logits)
    run_attention(
        topk,
        q_att,
        swa_pool,
        bt_swa,
        c_pool,
        bt_c,
        kv_lens,
        sinks,
        out,
        softmax_scale,
        config=config,
    )

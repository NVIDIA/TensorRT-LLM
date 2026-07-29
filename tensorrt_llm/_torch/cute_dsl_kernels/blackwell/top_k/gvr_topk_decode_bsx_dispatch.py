# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BSX top-K tier dispatcher — routes ``cute_dsl_gvr_topk_decode`` calls to
the BSX CuTe DSL tiers (direct / reg / tp).

Exact transcription of the original CUDA implementation's dispatch
(``gvr_topk_launch_batched``)
(``gvr_topk_launch_batched`` + ``launch_dense`` + ``launch_tp<512>``).
The gvr streaming tier (cs=16 fallback for npad > 262144) is intentionally
NOT ported: it is unreachable inside the deployment envelope
(npad <= 262144), which :func:`is_bsx_supported` enforces.

Dispatcher guard (anything else falls back to the in-tree
``GvrTopKKernel`` path in ``CuteDSLGvrTopKDecodeRunner.forward``):
  dtype == fp32, next_n >= 1 (MTP; num_rows divisible by next_n),
  compress_ratio in {1, 4}, order_row is None, counters is None,
  K in {512, 1024, 2048}, npad <= 262144, npad % 64 == 0, contiguous
  16B-aligned tensors, and the routed tier's cluster size within the
  queried hardware max (never silently degrade the cluster shape).
pre_idx / seq_lens are request-level ([num_rows // next_n, K] /
[num_rows // next_n]) — the in-tree contract. Per-row degeneracy
(N_eff <= K) and ragged N are handled INSIDE the tiers, so the guard
needs no device sync.

Env knobs (identical semantics to the CUDA arm's ``GVR_BSX_*`` static locals,
renamed for the production tree; cached at first use like the CUDA static
locals; :func:`_reset_env_cache` re-reads for tests):
  TRTLLM_BSX_TP_BS    unset/-1 -> baked per-npad bands; 0 -> disable (2^30);
                      else the bs threshold at which the tp tier takes over.
  TRTLLM_BSX_DENSE_BS same, for the dense (tb=1024) reg tiers.
"""

import os

import torch

from .gvr_topk_decode_bsx_direct import DKCMAX, direct_topk
from .gvr_topk_decode_bsx_reg import reg_topk
from .gvr_topk_decode_bsx_tp import tp_cluster_size, tp_topk
from .single_pass_multi_cta_radix_topk_cluster import _query_max_cluster_size

_BIG = 1 << 30

_ENV = {}  # cached thresholds, mirrors the CUDA static-local caching


def _env_threshold(name):
    t = _ENV.get(name)
    if t is None:
        e = os.environ.get(name)
        t = int(e) if e not in (None, "") else -1
        if t == 0:
            t = _BIG
        _ENV[name] = t
    return t


def _reset_env_cache():
    _ENV.clear()
    _DISPATCH_CACHE.clear()  # routes depend on the env thresholds


def _thresholds(npad):
    tpb = _env_threshold("TRTLLM_BSX_TP_BS")
    if tpb < 0:
        tpb = 256 if npad <= 20480 else (128 if npad < 32768 else 16)
    dnb = _env_threshold("TRTLLM_BSX_DENSE_BS")
    if dnb < 0:
        dnb = 8 if npad >= 163840 else (64 if npad < 32768 else _BIG)
    return tpb, dnb


# Measured fallback band table (full-grid calibration, 2026-07-28):
# (npad, bs) buckets where the in-tree kernel is faster than every bsx
# tier by >1.10x on at least one production layer (865 real decode cells
# x 11 BS, same-rep cold-L2 nsys pairs; to recalibrate, re-run that
# paired sweep and route every (npad, bs) bucket whose per-case floor
# drops below 0.909). These are the L2-resident mid-N shapes where the
# in-tree exact-count ladder admits a leaner candidate set and its
# row-slice cluster split keeps every CTA busy through P4. Routing them
# to the in-tree kernel caps the worst case at 1.10x while keeping the
# bsx win elsewhere (full-grid gm 1.40 vs the in-tree head).
# Keys are the nearest power-of-two of npad; values are inclusive bs
# ranges. TRTLLM_BSX_FALLBACK_BANDS=0 disables the table (bsx serves
# every guarded shape).
# 2026-07-29 recalibration: the 131072 band was extended down to bs=8 for
# TRUE 128K shapes only (see _BAND_LOW_BS_NPAD_MAX). At bs=8 the reg tier
# dips to 0.82-0.91x vs the in-tree kernel on 5 production layers at
# npad=131136, which the floor guarantee cannot admit; the same bucket at
# npad~163776 wins 1.36-1.60x, and those shapes only land in this bucket
# through the nearest-power-of-two rounding, so the extension excludes them.
_FALLBACK_BANDS = {
    8192: (256, 1 << 30),
    16384: (256, 1 << 30),
    32768: (16, 1 << 30),
    65536: (16, 1 << 30),
    131072: (8, 255),
    262144: (16, 127),
}

# The bs=8 end of the 131072 band applies only up to this npad: shapes above
# it (e.g. npad 163776) round INTO the 131072 bucket but behave like the next
# tier band, where the bsx reg tier is far ahead. They keep the calibrated
# bs>=16 routing.
_BAND_LOW_BS_NPAD_MAX = {131072: 147456}


def _in_fallback_band(bs: int, npad: int) -> bool:
    if _env_threshold("TRTLLM_BSX_FALLBACK_BANDS") == _BIG:  # "0" -> off
        return False
    up = 1 << max(npad - 1, 1).bit_length()  # pow2 >= npad
    np2 = up if 4 * npad >= 3 * up else up >> 1  # arithmetic-midpoint nearest
    band = _FALLBACK_BANDS.get(np2)
    if band is None:
        return False
    lo, hi = band
    npad_max = _BAND_LOW_BS_NPAD_MAX.get(np2)
    if npad_max is not None and npad > npad_max and lo < 16:
        lo = 16
    return lo <= bs <= hi


# tier name -> (kind, params). reg params = (cs, tb, maxv, ar).
def route(bs: int, npad: int, K: int) -> str:
    """Tier the CUDA gvr_topk_launch_batched would take. Returns
    'tp' | 'direct' | 'reg(cs=C,tb=T,maxv=M,ar=A)' | 'gvr(cs=16,tb=512)'
    ('gvr' is unreachable through :func:`bsx_topk` — see module docstring)."""
    tpb, dnb = _thresholds(npad)
    if bs >= tpb:
        return "tp"
    if npad > DKCMAX and bs >= dnb:
        # launch_dense table
        if npad <= 20480:
            return "reg(cs=1,tb=1024,maxv=5,ar=8)"
        if npad < 32768:
            return "reg(cs=1,tb=1024,maxv=8,ar=8)"
        if npad <= 65536:
            return "reg(cs=2,tb=1024,maxv=8,ar=8)"
        if npad <= 131072:
            return "reg(cs=4,tb=1024,maxv=8,ar=8)"
        if npad <= 262144:
            return "reg(cs=8,tb=1024,maxv=8,ar=8)"
        return "gvr(cs=16,tb=512)"
    # latency ladder
    if npad <= DKCMAX:
        return "direct"
    if npad < 16384:
        return "reg(cs=1,tb=512,maxv=8,ar=8)"
    if npad < 32768:
        return "reg(cs=4,tb=512,maxv=4,ar=8)"
    if npad <= 49152:
        return "reg(cs=8,tb=512,maxv=3,ar=8)"
    if npad <= 65536:
        return "reg(cs=8,tb=512,maxv=4,ar=8)"
    if npad <= 131072:
        return "reg(cs=8,tb=512,maxv=8,ar=8)"
    if npad <= 163840:
        if K >= 2048:
            return "reg(cs=16,tb=512,maxv=5,ar=6)"
        return "reg(cs=16,tb=512,maxv=5,ar=8)"
    if npad <= 262144:
        if K == 2048:
            return "reg(cs=16,tb=512,maxv=8,ar=8)"
        return "reg(cs=16,tb=512,maxv=8,ar=6)"
    return "gvr(cs=16,tb=512)"


def _parse_reg(tier):
    body = tier[tier.index("(") + 1 : -1]
    d = dict(kv.split("=") for kv in body.split(","))
    return int(d["cs"]), int(d["tb"]), int(d["maxv"]), int(d["ar"])


def route_cluster_size(bs: int, npad: int, K: int) -> int:
    """Cluster size the routed tier would launch with (host-only helper for
    the hardware cluster-cap guard)."""
    tier = route(bs, npad, K)
    if tier == "tp":
        return tp_cluster_size(bs, npad)
    if tier == "direct":
        return 1
    if tier.startswith("reg"):
        return _parse_reg(tier)[0]
    return 16  # gvr


# Per-call route()+_parse_reg() string work costs ~2.5-3us host-submit wall
# on <20us kernels (measured). The routing decision is pure in (bs, npad, K)
# [env thresholds are cached at first use, like the CUDA static locals], so
# bind it once per key to a closure.
_DISPATCH_CACHE = {}  # (bs, npad, K, next_n, cr) -> callable(logits, pre, seq_lens, out)


def _bind(bs, npad, K, next_n, cr):
    tier = route(bs, npad, K)
    if tier == "tp":

        def fn(lg, pre, sl, out):
            tp_topk(lg, pre, sl, out, K, next_n, cr)
    elif tier == "direct":

        def fn(lg, pre, sl, out):
            direct_topk(lg, sl, out, K, next_n, cr)
    elif tier.startswith("gvr"):
        raise ValueError(
            f"bsx gvr tier is not ported (npad beyond the deployment "
            f"envelope); is_bsx_supported must gate this out (bs={bs}, "
            f"npad={npad}, K={K})"
        )
    else:
        cs, tb, maxv, ar = _parse_reg(tier)

        def fn(lg, pre, sl, out):
            reg_topk(lg, pre, sl, out, K, cs, tb, maxv, ar, next_n, cr)

    return fn


def is_bsx_supported(
    logits: torch.Tensor,
    pre_idx: torch.Tensor,
    seq_lens: torch.Tensor,
    output_indices: torch.Tensor,
    top_k: int,
    next_n: int,
    compress_ratio: int,
    order_row,
    counters,
) -> bool:
    """Host-only guard for the bsx tiers (no device sync; see module
    docstring). Returns False -> caller uses the in-tree kernel."""
    if logits.dtype != torch.float32:
        return False
    if next_n < 1 or compress_ratio not in (1, 4):
        return False
    if order_row is not None or counters is not None:
        return False
    if top_k not in (512, 1024, 2048):
        return False
    bs, npad = logits.shape
    if bs % next_n != 0:
        return False
    n_req = bs // next_n
    if npad > 262144 or npad % 64 != 0:
        return False
    if _in_fallback_band(bs, npad):
        return False
    if not (
        logits.is_contiguous()
        and pre_idx.is_contiguous()
        and output_indices.is_contiguous()
        and seq_lens.is_contiguous()
    ):
        return False
    if pre_idx.shape != (n_req, top_k) or output_indices.shape != (bs, top_k):
        return False
    if seq_lens.shape != (n_req,) or seq_lens.dtype != torch.int32:
        return False
    if pre_idx.dtype != torch.int32 or output_indices.dtype != torch.int32:
        return False
    if (
        logits.data_ptr() % 16 != 0
        or pre_idx.data_ptr() % 16 != 0
        or output_indices.data_ptr() % 16 != 0
    ):
        return False
    # Cluster cap: fall back to the in-tree kernel (dispatcher-level) rather
    # than silently degrading the tier's cluster shape.
    if route_cluster_size(bs, npad, top_k) > _query_max_cluster_size():
        return False
    return True


def bsx_topk(
    logits: torch.Tensor,
    pre_idx: torch.Tensor,
    seq_lens: torch.Tensor,
    output_indices: torch.Tensor,
    top_k: int,
    next_n: int = 1,
    compress_ratio: int = 4,
) -> None:
    """Unified bsx tier dispatch, replicating gvr_topk_launch_batched.

    logits [BS, npad] fp32 (BS = num_requests * next_n; npad multiple of
    64; per-row tail beyond N_eff may be garbage — masked in-kernel),
    pre_idx [BS // next_n, K] int32 (request-level), seq_lens
    [BS // next_n] int32 (request-level, uncompressed-token space),
    output_indices [BS, K] int32. Caller must have passed
    :func:`is_bsx_supported`.
    """
    bs, npad = logits.shape
    key = (bs, npad, top_k, next_n, compress_ratio)
    fn = _DISPATCH_CACHE.get(key)
    if fn is None:
        fn = _DISPATCH_CACHE[key] = _bind(bs, npad, top_k, next_n, compress_ratio)
    fn(logits, pre_idx, seq_lens, output_indices)


__all__ = [
    "bsx_topk",
    "is_bsx_supported",
    "route",
    "route_cluster_size",
    "_reset_env_cache",
]

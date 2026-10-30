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

"""Tiered-GVR top-K dispatcher — routes ``cute_dsl_gvr_topk_decode`` calls
to the CuTe DSL GVR tiers (direct / reg / tp).

Exact transcription of the original CUDA implementation's dispatch
(``gvr_topk_launch_batched``)
(``gvr_topk_launch_batched`` + ``launch_dense`` + ``launch_tp<512>``).
The gvr streaming tier (cs=16 fallback for npad > 262144) is intentionally
NOT ported: it is unreachable inside the deployment envelope
(npad <= 262144), which :func:`is_tiered_topk_supported` enforces.

Dispatcher guard (anything else falls back to the in-tree
``GvrTopKKernel`` path in ``CuteDSLGvrTopKDecodeRunner.forward``):
  dtype == fp32, next_n >= 1 (MTP; num_rows divisible by next_n),
  compress_ratio in {1, 4}, counters is None,
  K in {512, 1024, 2048}, npad <= 262144, npad % 64 == 0, contiguous
  16B-aligned tensors, and the routed tier's cluster size within the
  queried hardware max (never silently degrade the cluster shape).
``order_row`` is accepted and IGNORED: it is the LJF scheduling hint the
caller (``dsa.py``) computes for the in-tree persistent kernel whenever
num_rows >= 2 * num_sms, and the GVR tiers launch per-row CTAs that the
hardware schedules directly — the permutation affects neither their
correctness nor their measured performance (the tier mesh was benched in
natural row order). Rejecting it would silently turn the tiers off for every
large batch in production. ``counters`` (LB mode) still falls back: the
LB partition contract belongs to the in-tree kernel.
pre_idx / seq_lens are request-level ([num_rows // next_n, K] /
[num_rows // next_n]) — the in-tree contract. Per-row degeneracy
(N_eff <= K) and ragged N are handled INSIDE the tiers, so the guard
needs no device sync.

Env knobs (identical semantics to the CUDA development arm's static locals;
cached at first use like those; :func:`_reset_env_cache` re-reads for tests):
  TRTLLM_GVR_TP_BS    unset/-1 -> baked per-npad bands; 0 -> disable (2^30);
                      else the bs threshold at which the tp tier takes over.
  TRTLLM_GVR_DENSE_BS same, for the dense (tb=1024) reg tiers.
  TRTLLM_GVR_FALLBACK_BANDS
                      0 -> disable the measured fallback-band table (the
                      tiers serve every guarded shape); unset/other -> active.
  TRTLLM_GVR_TIERS_DISABLE  any value other than unset/""/"0" -> the guard
                      rejects everything (kill switch: every call takes the
                      in-tree kernel path).
Malformed numeric values fail soft: a warning is logged once and the knob
falls back to unset (-1) instead of raising on the decode path.
"""

import os

import torch

from tensorrt_llm.logger import logger

from .gvr_topk_decode_direct import DKCMAX, direct_topk
from .gvr_topk_decode_reg import reg_topk
from .gvr_topk_decode_tp import tp_cluster_size, tp_topk
from .single_pass_multi_cta_radix_topk_cluster import _query_max_cluster_size

_BIG = 1 << 30

_ENV = {}  # cached thresholds, mirrors the CUDA static-local caching


def _env_threshold(name):
    t = _ENV.get(name)
    if t is None:
        e = os.environ.get(name)
        if e is None or e.strip() == "":
            t = -1
        else:
            try:
                t = int(e.strip())
            except ValueError:
                # Tuning override with a safe baked default: fail soft
                # (treat as unset) instead of killing the decode path.
                logger.warning(
                    f"{name}={e!r} is not an integer; ignoring the override "
                    f"and using the baked default."
                )
                t = -1
        if t == 0:
            t = _BIG
        _ENV[name] = t
    return t


def _env_flag(name):
    t = _ENV.get(name)
    if t is None:
        t = os.environ.get(name, "").strip() not in ("", "0")
        _ENV[name] = t
    return t


def _reset_env_cache():
    _ENV.clear()
    _DISPATCH_CACHE.clear()  # routes depend on the env thresholds
    _CAP_OK_CACHE.clear()  # ditto (verdict embeds the routed tier)


def _thresholds(npad):
    tpb = _env_threshold("TRTLLM_GVR_TP_BS")
    if tpb < 0:
        tpb = 256 if npad <= 20480 else (128 if npad < 32768 else 16)
    dnb = _env_threshold("TRTLLM_GVR_DENSE_BS")
    if dnb < 0:
        dnb = 8 if npad >= 163840 else (64 if npad < 32768 else _BIG)
    return tpb, dnb


# Measured fallback band table (full-grid calibration, 2026-07-28):
# (npad, bs) buckets where the in-tree kernel is faster than every GVR
# tier by >1.10x on at least one production layer (865 real decode cells
# x 11 BS, same-rep cold-L2 nsys pairs; to recalibrate, re-run that
# paired sweep and route every (npad, bs) bucket whose per-case floor
# drops below 0.909). These are the L2-resident mid-N shapes where the
# in-tree exact-count ladder admits a leaner candidate set and its
# row-slice cluster split keeps every CTA busy through P4. Routing them
# to the in-tree kernel caps the worst case at 1.10x while keeping the
# tier win elsewhere (full-grid gm 1.40 vs the in-tree head).
# Keys are the nearest power-of-two of npad; values are inclusive bs
# ranges. TRTLLM_GVR_FALLBACK_BANDS=0 disables the table (the tiers
# serve every guarded shape).
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
# tier band, where the reg tier is far ahead. They keep the calibrated
# bs>=16 routing.
_BAND_LOW_BS_NPAD_MAX = {131072: 147456}


def _in_fallback_band(bs: int, npad: int) -> bool:
    if _env_threshold("TRTLLM_GVR_FALLBACK_BANDS") == _BIG:  # "0" -> off
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
    'tp' | 'direct' | 'reg(cs=C,tb=T,maxv=M,ar=A)' | 'cluster(cs=16,tb=512)'
    ('cluster' is unreachable through :func:`tiered_topk` — see module docstring)."""
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
        return "cluster(cs=16,tb=512)"
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
    return "cluster(cs=16,tb=512)"


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
    return 16  # cluster


# Per-call route()+_parse_reg() string work costs ~2.5-3us host-submit wall
# on <20us kernels (measured). The routing decision is pure in (bs, npad, K)
# [env thresholds are cached at first use, like the CUDA static locals], so
# bind it once per key to a closure.
_DISPATCH_CACHE = {}  # (bs, npad, K, next_n, cr) -> callable(logits, pre, seq_lens, out)
_CAP_OK_CACHE = {}  # (bs, npad, K) -> routed tier's cluster size within hw max


def _bind(bs, npad, K, next_n, cr):
    tier = route(bs, npad, K)
    if tier == "tp":

        def fn(lg, pre, sl, out):
            tp_topk(lg, pre, sl, out, K, next_n, cr)
    elif tier == "direct":

        def fn(lg, pre, sl, out):
            direct_topk(lg, sl, out, K, next_n, cr)
    elif tier.startswith("cluster"):
        raise ValueError(
            f"the cluster tier is not ported (npad beyond the deployment "
            f"envelope); is_tiered_topk_supported must gate this out (bs={bs}, "
            f"npad={npad}, K={K})"
        )
    else:
        cs, tb, maxv, ar = _parse_reg(tier)

        def fn(lg, pre, sl, out):
            reg_topk(lg, pre, sl, out, K, cs, tb, maxv, ar, next_n, cr)

    return fn


def is_tiered_topk_supported(
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
    """Host-only guard for the GVR tiers (no device sync; see module
    docstring). Returns False -> caller uses the in-tree kernel."""
    if _env_flag("TRTLLM_GVR_TIERS_DISABLE"):
        return False
    if logits.dtype != torch.float32:
        return False
    if next_n < 1 or compress_ratio not in (1, 4):
        return False
    # order_row is accepted and ignored (in-tree scheduling hint; see the
    # module docstring). counters (LB mode) keeps the in-tree path.
    if counters is not None:
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
    # than silently degrading the tier's cluster shape. The verdict is pure
    # in (bs, npad, K) once the env thresholds are cached, and computing it
    # re-runs route() + the _parse_reg string parse (the ~2.5-3us host cost
    # _DISPATCH_CACHE exists to avoid) — so memoize it the same way.
    key = (bs, npad, top_k)
    ok = _CAP_OK_CACHE.get(key)
    if ok is None:
        ok = _CAP_OK_CACHE[key] = route_cluster_size(bs, npad, top_k) <= _query_max_cluster_size()
    return ok


def tiered_topk(
    logits: torch.Tensor,
    pre_idx: torch.Tensor,
    seq_lens: torch.Tensor,
    output_indices: torch.Tensor,
    top_k: int,
    next_n: int = 1,
    compress_ratio: int = 4,
) -> None:
    """Unified GVR tier dispatch, replicating gvr_topk_launch_batched.

    logits [BS, npad] fp32 (BS = num_requests * next_n; npad multiple of
    64; per-row tail beyond N_eff may be garbage — masked in-kernel),
    pre_idx [BS // next_n, K] int32 (request-level), seq_lens
    [BS // next_n] int32 (request-level, uncompressed-token space),
    output_indices [BS, K] int32. Caller must have passed
    :func:`is_tiered_topk_supported`.
    """
    bs, npad = logits.shape
    key = (bs, npad, top_k, next_n, compress_ratio)
    fn = _DISPATCH_CACHE.get(key)
    if fn is None:
        fn = _DISPATCH_CACHE[key] = _bind(bs, npad, top_k, next_n, compress_ratio)
    fn(logits, pre_idx, seq_lens, output_indices)


__all__ = [
    "tiered_topk",
    "is_tiered_topk_supported",
    "route",
    "route_cluster_size",
    "_reset_env_cache",
]

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
the op43 CuTe DSL tiers (direct / reg / tp).

Port of the op43 ``ct_bsx.py`` unified launcher, itself an exact
transcription of the CUDA dispatch in op42 ``gvr_bsx.cu``
(``gvr_topk_launch_batched`` + ``launch_dense`` + ``launch_tp<512>``).
The gvr streaming tier (cs=16 fallback for npad > 262144) is intentionally
NOT ported: it is unreachable inside the deployment envelope
(npad <= 262144), which :func:`is_bsx_supported` enforces.

v1 dispatcher guard (anything else falls back to the in-tree
``GvrTopKKernel`` path in ``CuteDSLGvrTopKDecodeRunner.forward``):
  dtype == fp32, next_n == 1, compress_ratio == 4, order_row is None,
  counters is None, K in {512, 1024, 2048}, npad <= 262144, npad % 64 == 0,
  contiguous 16B-aligned tensors, and the routed tier's cluster size within
  the queried hardware max (never silently degrade the cluster shape).
Per-row degeneracy (N_eff <= K) and ragged N are handled INSIDE the tiers,
so the guard needs no device sync.

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
# on <20us kernels (op43 S-E). The routing decision is pure in (bs, npad, K)
# [env thresholds are cached at first use, like the CUDA static locals], so
# bind it once per key to a closure.
_DISPATCH_CACHE = {}  # (bs, npad, K) -> callable(logits, pre, seq_lens, out)


def _bind(bs, npad, K):
    tier = route(bs, npad, K)
    if tier == "tp":

        def fn(lg, pre, sl, out):
            tp_topk(lg, pre, sl, out, K)
    elif tier == "direct":

        def fn(lg, pre, sl, out):
            direct_topk(lg, sl, out, K)
    elif tier.startswith("gvr"):
        raise ValueError(
            f"bsx gvr tier is not ported (npad beyond the deployment "
            f"envelope); is_bsx_supported must gate this out (bs={bs}, "
            f"npad={npad}, K={K})"
        )
    else:
        cs, tb, maxv, ar = _parse_reg(tier)

        def fn(lg, pre, sl, out):
            reg_topk(lg, pre, sl, out, K, cs, tb, maxv, ar)

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
    """Host-only v1 guard for the bsx tiers (no device sync; see module
    docstring). Returns False -> caller uses the in-tree kernel."""
    if logits.dtype != torch.float32:
        return False
    if next_n != 1 or compress_ratio != 4:
        return False
    if order_row is not None or counters is not None:
        return False
    if top_k not in (512, 1024, 2048):
        return False
    bs, npad = logits.shape
    if npad > 262144 or npad % 64 != 0:
        return False
    if not (
        logits.is_contiguous()
        and pre_idx.is_contiguous()
        and output_indices.is_contiguous()
        and seq_lens.is_contiguous()
    ):
        return False
    if pre_idx.shape != (bs, top_k) or output_indices.shape != (bs, top_k):
        return False
    if seq_lens.shape != (bs,) or seq_lens.dtype != torch.int32:
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
) -> None:
    """Unified bsx tier dispatch, replicating gvr_topk_launch_batched.

    logits [BS, npad] fp32 (npad multiple of 64; per-row tail beyond N_eff
    may be garbage — masked in-kernel), pre_idx [BS, K] int32, seq_lens
    [BS] int32 (request-level, uncompressed-token space), output_indices
    [BS, K] int32. Caller must have passed :func:`is_bsx_supported`.
    """
    bs, npad = logits.shape
    key = (bs, npad, top_k)
    fn = _DISPATCH_CACHE.get(key)
    if fn is None:
        fn = _DISPATCH_CACHE[key] = _bind(bs, npad, top_k)
    fn(logits, pre_idx, seq_lens, output_indices)


__all__ = [
    "bsx_topk",
    "is_bsx_supported",
    "route",
    "route_cluster_size",
    "_reset_env_cache",
]

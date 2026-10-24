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
"""Shared helpers of the fused top-p rejection-sampling tests: a float64 model of the Philox
streams and of the inverse-CDF / chain-verify decisions, the row generators, and the float64 nucleus oracle
(flashinfer's ``top_p_renorm_probs`` semantics: the pivot is the first token, in descending order of the scaled
logit ``z``, at which the cumulative mass reaches ``top_p * sum``; every token with ``z >= z_pivot`` is kept,
ties included).

Every disagreement with flashinfer or with the oracle must be explained by float32 rounding: a draw within
``_CDF_TOL`` of an inverse-CDF boundary, an accept threshold within ``_ACCEPT_TOL``, or a nucleus inside the
oracle's sandwich: ``target_error`` is the relative change of the target mass ``fl32(p) * S`` that would make a
kept-set the exact nucleus (0 inside the exact interval, infinite for a set that is not a ``z >= threshold`` set,
so one that splits a tie group), and the kernels' sets must stay within ``_NUC_MASS_TOL`` of it (flashinfer's
approximate float32 search within ``_FI_MASS_TOL``). That is exact for tie groups, and it cannot excuse a pivot
that is off by more than float32 rounding, however thin the tokens around it are (a flat row of 163,840 tokens
has ~6e-6 per token).
"""

import math

import numpy as np
import torch

from tensorrt_llm._torch.speculative import fast_rejection_sampling as frs
from tensorrt_llm._torch.speculative import fast_topp_rejection_sampling as ftp

DEV = "cuda"
V = 163840
K = 5
_CDF_TOL = 2e-5
_ACCEPT_TOL = 1e-5
# Float32 rounding of a sum of ~1e5 weights is a few 1e-7 of the sum (tile-order sums here, float
# atomics in flashinfer): a target mass within this fraction of the row's total of a cumulative
# boundary can fall on either side of it, and two nuclei that differ only by tokens whose total mass
# is below it are the same nucleus to float precision (at p = 0.999 the boundary tokens of a
# 1e5-token nucleus weigh 4e-8 each, so one to three of them differ in most rows).
_NUC_MASS_TOL = 4e-6
# flashinfer's top_p_renorm_probs keeps one extra token beyond the exact nucleus (1.1e-5 of the
# mass: a flat row has 6e-6 per token) in ~1% of the flat rows at top_p 0.5 (41 of ~3,900 replayed
# verify-test rows; the target error of its set is at most 3.0e-5, three tokens), while the kernels'
# sets never miss the exact nucleus by more than 6e-7. Its nuclei are not under test, the kernels'
# are: a wider sandwich only checks that a row where the two differ differs by a token or two at
# the pivot, not by a bug.
_FI_MASS_TOL = 1e-4

_M32 = 0xFFFFFFFF
_M64 = (1 << 64) - 1


def _philox4x32_10(ctr, key):
    c0, c1, c2, c3 = ctr
    k0, k1 = key
    for r in range(10):
        p0 = 0xD2511F53 * c0
        p1 = 0xCD9E8D57 * c2
        c0, c1, c2, c3 = (
            ((p1 >> 32) ^ c1 ^ k0) & _M32,
            p1 & _M32,
            ((p0 >> 32) ^ c3 ^ k1) & _M32,
            p0 & _M32,
        )
        if r < 9:
            k0 = (k0 + 0x9E3779B9) & _M32
            k1 = (k1 + 0xBB67AE85) & _M32
    return c0, c1, c2, c3


def curand_uniform(seed, offset, subsequence, draw):
    """Draw ``draw`` of ``curand_init(seed, subsequence, offset)`` + ``curand_uniform``."""
    seed &= _M64
    g = (offset & 3) + draw
    ctr = (offset >> 2) + (g >> 2)
    out = _philox4x32_10(
        (ctr & _M32, (ctr >> 32) & _M32, subsequence & _M32, (subsequence >> 32) & _M32),
        (seed & _M32, (seed >> 32) & _M32),
    )
    x = np.float32(out[g & 3])
    return float(np.float32(x * np.float32(2.0**-32) + np.float32(2.0**-33)))


def _draft_key(seed):
    return (seed & _M64) ^ frs.DRAFT_KEY_XOR


def _as_int64(x):
    x &= _M64
    return x - (1 << 64) if x >> 63 else x


def _cdf_gap(weights, u, a, b):
    """Smallest delta for which ``u`` lies within delta of both tokens' inverse-CDF intervals."""
    c = np.cumsum(np.asarray(weights, dtype=np.float64))

    def dist(t):
        lo = c[t - 1] if t > 0 else 0.0
        hi = c[t]
        if lo <= u < hi:
            return 0.0
        return lo - u if u < lo else u - hi

    return max(dist(a), dist(b))


def _inv_cdf(weights, u):
    """flashinfer's ``sampling_from_probs`` rule on float64 sums."""
    w = np.asarray(weights, dtype=np.float64)
    idx = int(np.searchsorted(np.cumsum(w), u, side="right"))
    return idx if idx < len(w) else int(np.flatnonzero(w > 0)[-1])


def _token_gap(weights, u, got):
    want = _inv_cdf(weights, u)
    if int(got) == want:
        return 0.0
    return _cdf_gap(weights, u, int(got), want) / max(float(np.sum(weights)), 1e-300)


def _verify_model(p, q, drafts, seed, offset, b, k):
    """Request ``b``'s accept position and drawn-token weights from float64 target nuclei ``p``
    [k + 1, V] and stored proposals ``q`` [k, V]: (pos, weights, u scaled by their mass)."""
    pos = k
    for i in range(k):
        d = int(drafts[i])
        if not curand_uniform(seed, offset, b, i) * q[i, d] < p[i, d]:
            pos = i
            break
    w = np.maximum(p[pos] - q[pos], 0.0) if pos < k else p[pos]
    return pos, w, curand_uniform(seed, offset, b, k) * float(w.sum())


def _verify_row_gap(acc_row, num, p, q, drafts, seed, offset, b, k):
    pos_m, w, u = _verify_model(p, q, drafts, seed, offset, b, k)
    pos = int(num) - 1
    if pos != pos_m:
        i = min(pos, pos_m)
        d = int(drafts[i])
        u_i = curand_uniform(seed, offset, b, i)
        return "accept", abs(u_i * q[i, d] - p[i, d]) / max(p[i, d], 1e-300)
    row = [int(x) for x in acc_row]
    assert row[:pos] == [int(x) for x in drafts[:pos]]
    assert all(x == -1 for x in row[pos + 1 : k + 1])
    return "token", _token_gap(w, u, row[pos])


class _Gaps:
    """Worst rounding size per kind and the number of rows that needed one."""

    def __init__(self):
        self.worst = {"accept": 0.0, "token": 0.0, "nucleus": 0.0}
        self.count = 0

    def add(self, kind, size):
        self.worst[kind] = max(self.worst[kind], size)
        self.count += size > 0.0

    def check(self, total):
        assert self.worst["accept"] <= _ACCEPT_TOL and self.worst["token"] <= _CDF_TOL
        assert self.worst["nucleus"] <= _NUC_MASS_TOL, self.worst
        assert self.count <= max(2, int(0.02 * total)), (self.count, total)


# ----------------------------------------------------------------------------------------------
# Rows and the float64 nucleus oracle.
# ----------------------------------------------------------------------------------------------
KINDS = [
    "peaked",
    "flat",
    "medium",
    "very_flat",
    "heavy",
    "one_hot",
    "constant",
    "two_level",
    "uniform01",
    "ramp",
]


def peaked_logits(rows, vocab, gen, spread=2.0, ncand=48, top=14.0):
    base = torch.randn(rows, vocab, generator=gen, device=DEV) * spread
    idx = torch.randint(0, vocab, (rows, ncand), generator=gen, device=DEV)
    boost = torch.linspace(top, top - 8.0, ncand, device=DEV).unsqueeze(0).expand(rows, -1)
    base.scatter_add_(1, idx, boost + torch.randn(rows, ncand, generator=gen, device=DEV))
    return base.float().contiguous()


def make_rows(kind, rows, vocab, gen, bf16=False):
    """Logits rows of one family (the nucleus of ``peaked`` holds tens of tokens, of ``flat`` and
    ``medium`` thousands, of ``very_flat`` most of the vocabulary; ``zipf<alpha>`` rows follow a
    power law, p_r ~ r^-alpha: ~1 token for alpha 6, ~5 for 2.5, ~80 for 1.6, ~1000 for 1.4,
    most of the vocabulary below 1.35; ``constant`` / ``two_level`` / ``bf16`` rows are made of exact
    ties)."""
    if kind.startswith("zipf"):
        alpha = float(kind[4:])
        rank = torch.rand(rows, vocab, generator=gen, device=DEV).argsort(1).float() + 1.0
        noise = torch.randn(rows, vocab, generator=gen, device=DEV) * 0.05
        x = -alpha * torch.log(rank) + noise
    elif kind == "peaked":
        x = peaked_logits(rows, vocab, gen)
    elif kind == "medium":
        x = torch.randn(rows, vocab, generator=gen, device=DEV) * 1.5
        idx = torch.randint(0, vocab, (rows, 300), generator=gen, device=DEV)
        x.scatter_add_(1, idx, 7.0 + torch.randn(rows, 300, generator=gen, device=DEV))
    elif kind == "flat":
        x = torch.randn(rows, vocab, generator=gen, device=DEV)
        n = max(vocab // 400, 4)
        idx = torch.randint(0, vocab, (rows, n), generator=gen, device=DEV)
        x.scatter_add_(1, idx, 4.0 + torch.randn(rows, n, generator=gen, device=DEV))
    elif kind == "very_flat":
        x = torch.randn(rows, vocab, generator=gen, device=DEV) * 0.3
    elif kind == "heavy":
        x = torch.randn(rows, vocab, generator=gen, device=DEV)
        x = x / torch.sqrt(torch.rand(rows, vocab, generator=gen, device=DEV) + 0.05)
        idx = torch.randint(0, vocab, (rows, 20), generator=gen, device=DEV)
        x.scatter_add_(1, idx, torch.full((rows, 20), 8.0, device=DEV))
    elif kind == "one_hot":
        x = torch.randn(rows, vocab, generator=gen, device=DEV) * 2.0
        x[torch.arange(rows), torch.randint(0, vocab, (rows,), generator=gen, device=DEV)] += 30.0
    elif kind == "constant":
        x = torch.full((rows, vocab), 1.25, device=DEV)
    elif kind == "two_level":
        x = torch.where(torch.rand(rows, vocab, generator=gen, device=DEV) < 0.3, 2.0, 1.0).to(
            torch.float32
        )
    elif kind == "uniform01":
        x = torch.rand(rows, vocab, generator=gen, device=DEV)
    elif kind == "ramp":
        base = torch.linspace(0.0, 20.0, vocab, device=DEV)
        x = torch.stack(
            [base[torch.randperm(vocab, generator=gen, device=DEV)] for _ in range(rows)]
        )
    else:
        raise ValueError(kind)
    x = x.float()
    if bf16:
        x = x.to(torch.bfloat16).float()
    return x.contiguous()


def scaled(logits, temps):
    """``z = x * fl32(1 / T)`` as the kernels (and flashinfer's softmax) compute it."""
    inv_t = torch.where(temps == 0, torch.zeros_like(temps), 1.0 / temps.clamp_min(1e-30))
    return logits * inv_t.unsqueeze(1)


class Nucleus:
    """float64 oracle of the flashinfer top-p rule for each row of ``logits``: the pivot is the
    first token, descending in ``z``, whose cumulative float64 mass reaches ``fl32(p) * S``; every
    token with ``z >= z_pivot`` is kept. ``gap`` is the relative distance of the target from the
    cumulative boundaries of the pivot's group (how close the decision was to rounding; diagnostics
    only). ``target`` is ``fl32(p) * S`` and ``full`` marks the rows that keep the whole vocabulary
    (``top_p >= 1``); ``target_error`` measures a kept-set against them."""

    def __init__(self, logits, temps, top_ps):
        self.z = scaled(logits, temps)
        rows, vocab = self.z.shape
        zd = self.z.double()
        zmax = zd.max(dim=1).values
        self.w = torch.exp(zd - zmax.unsqueeze(1))
        self.s = self.w.sum(1)
        zs, order = torch.sort(zd, dim=1, descending=True, stable=True)
        cum = torch.cumsum(torch.gather(self.w, 1, order), dim=1)
        target = self.target = top_ps.float().double() * self.s
        k = torch.searchsorted(cum, target.unsqueeze(1)).clamp(max=vocab - 1)
        self.z_pivot = torch.gather(zs, 1, k).squeeze(1)
        full = self.full = top_ps >= 1.0
        self.z_pivot = torch.where(full, torch.full_like(self.z_pivot, -math.inf), self.z_pivot)
        first = (zs > self.z_pivot.unsqueeze(1)).sum(1)
        last = (zs >= self.z_pivot.unsqueeze(1)).sum(1) - 1
        before = torch.where(
            first > 0,
            torch.gather(cum, 1, (first - 1).clamp(min=0).unsqueeze(1)).squeeze(1),
            torch.zeros_like(self.s),
        )
        through = torch.gather(cum, 1, last.clamp(min=0).unsqueeze(1)).squeeze(1)
        self.gap = torch.minimum((before - target).abs(), (through - target).abs()) / self.s
        self.gap = torch.where(full, torch.ones_like(self.gap), self.gap)
        self.kept = zd >= self.z_pivot.unsqueeze(1)
        self.mass = torch.where(self.kept, self.w, torch.zeros_like(self.w)).sum(1)

    def probs(self, kept=None):
        """float64 renormalised probabilities of the oracle's nucleus (or of ``kept``)."""
        kept = self.kept if kept is None else kept
        q = torch.where(kept, self.w, torch.zeros_like(self.w))
        return q / q.sum(1, keepdim=True)


def kernel_kept(logits, temps, top_ps, cfg=None):
    """The kernels' nucleus of each row (``ftp.fast_topp_nucleus``): (kept mask, rec [rows, 4])."""
    rec = ftp.fast_topp_nucleus(logits, temps, top_ps, cfg or ftp.DEFAULT_NUCLEUS)
    return scaled(logits, temps) >= rec[:, 0:1], rec


def diff_mass(nuc, r, kept_a, kept_b):
    """Mass (relative to the row's total) of the tokens on which two kept-sets of row ``r`` differ."""
    diff = kept_a != kept_b
    return float(nuc.w[r][diff].sum() / nuc.s[r])


def target_error(nuc, r, kept):
    """How far row ``r``'s kept-set ``kept`` is from being the oracle's nucleus, as a relative change of the
    target mass: a set ``{z >= z_k}`` is the nucleus of every target in (mass strictly above ``z_k``, mass
    through ``z_k``], so the error is how far the oracle's target lies outside that interval (0 inside it).
    A set that is not a threshold set (a hole, a tie group that is split, nothing kept) is infinitely far; the
    rows with ``top_p >= 1`` must keep every token."""
    if bool(nuc.full[r]):
        return 0.0 if bool(kept.all()) else math.inf
    if not bool(kept.any()):
        return math.inf
    z, w = nuc.z[r], nuc.w[r]
    z_k = z[kept].min()
    if bool((kept != (z >= z_k)).any()):
        return math.inf
    target = float(nuc.target[r])
    excess = max(float(w[z > z_k].sum()) - target, target - float(w[kept].sum()), 0.0)
    if excess <= 0.0:
        return 0.0
    return excess / target if target > 0.0 else math.inf


def sandwich_ok(nuc, r, kept, tol=_NUC_MASS_TOL):
    """Whether the kept-set ``kept`` of row ``r`` is the nucleus of the oracle's target to float32 rounding:
    it holds every token of the nucleus of the target ``(1 - tol)`` times the exact one and none outside the
    nucleus of ``(1 + tol)`` times it (nuclei grow with the target, ties included)."""
    return target_error(nuc, r, kept) <= tol


def support_diff_ok(nuc, r, kept_a, kept_b):
    """Whether two kept-sets of row ``r`` (a kernel's, the oracle's, another launch configuration's)
    both lie in the oracle's sandwich (``sandwich_ok``), so they agree to float32 rounding."""
    return sandwich_ok(nuc, r, kept_a) and sandwich_ok(nuc, r, kept_b)


# flashinfer's top_p_renorm_probs at top_p == 1 is not "keep everything": it zeroes low-probability tokens
# (tokens of probability <= 2**-23, in a "medium" row 23,000 of them = 2e-3 of the mass; a
# "flat" row loses up to 1e-5). The kernels keep the whole vocabulary there (the exact softmax), so against
# flashinfer such rows are compared on that rule instead of the nucleus tolerance.
_FI_FULL_DROP_P = 2.0**-23 * 1.01


def fi_support_ok(nuc, r, top_p, kept_a, kept_fi):
    """The kernel's kept-set ``kept_a`` (strict sandwich) against flashinfer's ``kept_fi`` for a row with ``top_p``
    (the wider ``_FI_MASS_TOL`` sandwich; at ``top_p == 1`` the rule of ``_FI_FULL_DROP_P``)."""
    if top_p < 1.0:
        return sandwich_ok(nuc, r, kept_a) and sandwich_ok(nuc, r, kept_fi, _FI_MASS_TOL)
    if not bool((kept_a != kept_fi).any()):
        return True
    diff = kept_a != kept_fi
    tiny = bool(((nuc.w[r][diff] / nuc.s[r]) <= _FI_FULL_DROP_P).all())
    return (tiny and bool(kept_a[diff].all())) or diff_mass(
        nuc, r, kept_a, kept_fi
    ) <= _NUC_MASS_TOL


_TOPP_POOL = (0.95, 0.9, 0.99, 0.5, 1.0)


def _request_top_ps(num_gens, trial):
    pool = torch.tensor(_TOPP_POOL, dtype=torch.float32, device=DEV)
    return pool[(torch.arange(num_gens, device=DEV) + trial) % pool.numel()].contiguous()


def _correlated_case(num_gens, gen, accept_bias, kind="peaked", k=K, vocab=V, bf16=False):
    """Target logits [G*(k+1), V] and draft logits [G, k, V] that agree often. ``bf16`` rounds both
    to bfloat16 (what the lm_head produces before the upcast): exact ties abound."""
    tgt = make_rows(kind, num_gens * (k + 1), vocab, gen, bf16)
    noise = torch.randn(num_gens, k, vocab, generator=gen, device=DEV) * accept_bias
    drf = (tgt.view(num_gens, k + 1, vocab)[:, :k] + noise).contiguous()
    if bf16:
        drf = drf.to(torch.bfloat16).float().contiguous()
    return tgt, drf


def _check_drafts(tokens, stored, seeds, offsets, k, gaps):
    """Draft tokens [G, k] vs the model on the stored proposals [G, k, V] (float64)."""
    for b in range(tokens.shape[0]):
        for i in range(k):
            u = curand_uniform(_draft_key(seeds[b]), offsets[b] + 1, b * k + i, 0)
            gaps.add("token", _token_gap(stored[b, i], u, tokens[b, i]))


def _check_verify(acc, num, p, q, drafts, seeds, offsets, k, gaps):
    for b in range(acc.shape[0]):
        gaps.add(
            *_verify_row_gap(acc[b], num[b], p[b], q[b], drafts[b], seeds[b], offsets[b], b, k)
        )

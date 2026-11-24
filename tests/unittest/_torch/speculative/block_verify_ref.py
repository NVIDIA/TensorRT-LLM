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
"""float64 numpy model of block verification (Sun et al., arXiv:2403.10444, Algorithm 2) and of the
token-wise chain verification it replaces, batched over independent requests.

Shapes (``B`` requests, draft length ``K``, vocabulary ``V``): target rows ``p [B, K + 1, V]``
(``p[:, i]`` is the target distribution of the token at draft position ``i`` given the draft prefix,
``p[:, K]`` the bonus distribution), stored proposals ``q [B, K, V]``, drafts ``[B, K]``. This is
the oracle of ``fast_block_verification`` (kernel decisions) and of the exactness tests (exact
enumeration over small trees, Monte Carlo against the target). Nothing here needs a GPU.
"""

import itertools

import numpy as np

_M32 = 0xFFFFFFFF
_M64 = (1 << 64) - 1


# --- cuRAND-compatible uniforms (Philox4x32-10), the stream the kernels draw from ---------------
def philox4x32_10(ctr, key):
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
    """Draw ``draw`` of ``curand_init(seed, subsequence, offset)`` + ``curand_uniform`` (in (0, 1])."""
    seed &= _M64
    g = (offset & 3) + draw
    ctr = (offset >> 2) + (g >> 2)
    out = philox4x32_10(
        (ctr & _M32, (ctr >> 32) & _M32, subsequence & _M32, (subsequence >> 32) & _M32),
        (seed & _M32, (seed >> 32) & _M32),
    )
    x = np.float32(out[g & 3])
    return float(np.float32(x * np.float32(2.0**-32) + np.float32(2.0**-33)))


# Key of the block-verification uniform stream: ``seed ^ BV_KEY_XOR`` (the kernel module's constant,
# written out here so that the model does not import the code it checks).
BV_KEY_XOR = 0x1F83D9ABFB41BD6B


def bv_uniform(seed, offset, subsequence, draw):
    """Draw ``draw`` of a request's block-verification stream (draw ``i - 1``: the uniform of the
    sub-block of length ``i``; draw ``K``: the correction token)."""
    return curand_uniform((seed & _M64) ^ BV_KEY_XOR, offset, subsequence, draw)


# --- the algorithms --------------------------------------------------------------------------
def block_weights(p, q, drafts):
    """Weights ``w [B, K + 1]`` (``w_0 = 1``, ``w_{i+1} = min(w_i p_i(d_i) / q_i(d_i), 1)``).

    A drawn token has ``q_i(d_i) > 0``; for a stale or mismatched row that excludes it the weight
    is the limit of ``min(w_i p_i(d_i) / q, 1)`` as ``q`` goes to 0: 1 where the target allows the
    token and the prefix is possible (``w_i > 0``), else 0."""
    b, k = drafts.shape
    idx = np.arange(b)
    w = np.ones((b, k + 1))
    for i in range(k):
        d = drafts[:, i]
        p_d, q_d = p[idx, i, d], q[idx, i, d]
        ratio = np.divide(p_d, q_d, out=np.zeros_like(p_d), where=q_d > 0.0)
        w[:, i + 1] = np.where(
            q_d > 0.0,
            np.minimum(w[:, i] * ratio, 1.0),
            np.where((p_d > 0.0) & (w[:, i] > 0.0), 1.0, 0.0),
        )
    return w


def block_residual_mass(p, q, w):
    """``R_i = sum_x max(w_i p_i(x) - q_i(x), 0)`` for ``i < K``; ``R_K = w_K`` (``q_K = 0``)."""
    b, kp1, _ = p.shape
    k = kp1 - 1
    r = np.empty((b, kp1))
    r[:, :k] = np.maximum(w[:, :k, None] * p[:, :k] - q, 0.0).sum(-1)
    r[:, k] = w[:, k] * p[:, k].sum(-1)
    return r


def block_accept_probs(p, q, drafts):
    """Sub-block acceptance probabilities ``h [B, K + 1]`` (``h_0`` unused, set to 1) and the
    weights: ``h_i = R_i / (R_i + 1 - w_i)`` (1 where that is 0 / 0), ``h_K = w_K``."""
    w = block_weights(p, q, drafts)
    r = block_residual_mass(p, q, w)
    den = r + (1.0 - w)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(den > 0.0, r / den, 1.0)
    h[:, -1] = w[:, -1]
    h[:, 0] = 1.0
    return h, w


def block_tau_pmf(h):
    """``P(tau = i)`` for ``tau = max({0} + {i >= 1 : eta_i <= h_i})`` with independent uniforms."""
    b, kp1 = h.shape
    pmf = np.zeros((b, kp1))
    surv = np.ones(b)  # prod_{j > i} (1 - h_j)
    for i in range(kp1 - 1, 0, -1):
        pmf[:, i] = h[:, i] * surv
        surv = surv * (1.0 - h[:, i])
    pmf[:, 0] = surv
    return pmf


def block_expected_tau(h):
    """``E[tau | drafts] = sum_{i >= 1} P(tau >= i)`` (the accepted drafts, excluding the extra
    token), exactly, from the acceptance probabilities."""
    b, kp1 = h.shape
    out = np.zeros(b)
    surv = np.ones(b)
    for i in range(kp1 - 1, 0, -1):
        surv = surv * (1.0 - h[:, i])
        out += 1.0 - surv
    return out


def token_accept_probs(p, q, drafts):
    """Per-position token-wise acceptance probabilities ``min(1, p_i(d_i) / q_i(d_i))`` [B, K]."""
    b, k = drafts.shape
    idx = np.arange(b)[:, None]
    pos = np.arange(k)[None, :]
    return np.minimum(1.0, p[idx, pos, drafts] / q[idx, pos, drafts])


def token_expected_tau(p, q, drafts):
    """``E[tau | drafts]`` of token-wise verification: ``sum_i prod_{j <= i} min(1, r_j)``."""
    return np.cumprod(token_accept_probs(p, q, drafts), axis=1).sum(1)


def residual_weights(p, q, w, tau):
    """Unnormalised correction-token weights per request: ``max(w_tau p_tau - q_tau, 0)`` for
    ``tau < K``, ``p_K`` for ``tau = K`` [B, V]."""
    b, kp1, v = p.shape
    k = kp1 - 1
    idx = np.arange(b)
    out = p[idx, tau].copy()
    live = tau < k
    qt = q[idx, np.minimum(tau, k - 1)] if k > 0 else np.zeros_like(out)
    out[live] = np.maximum(w[idx, tau][live, None] * p[idx, tau][live] - qt[live], 0.0)
    return out


def inv_cdf(weights, u):
    """The kernels' rule on float64 sums: the first token whose running mass exceeds ``u * total``
    (it has nonzero weight), else the last nonzero token, else token 0 (an all-zero row, which only
    rounding can produce). ``weights`` [B, V], ``u`` [B]."""
    v = weights.shape[-1]
    cdf = np.cumsum(weights, axis=-1)
    thr = u * cdf[:, -1]
    tok = (cdf <= thr[:, None]).sum(-1)
    nonzero = weights > 0.0
    last = np.where(nonzero.any(-1), v - 1 - np.argmax(nonzero[:, ::-1], axis=-1), 0)
    return np.where(tok < v, tok, last)


def _assemble(drafts, tau, y):
    """Output rows ``[B, K + 1]``: the first ``tau`` drafts, the token ``y``, then ``-1``."""
    b, k = drafts.shape
    padded = np.concatenate([drafts, np.zeros((b, 1), dtype=drafts.dtype)], axis=1)
    acc = np.where(np.arange(k + 1)[None, :] < tau[:, None], padded, -1).astype(np.int64)
    acc[np.arange(b), tau] = y
    return acc


def block_verify(p, q, drafts, eta, u, accept_probs=block_accept_probs, residual=residual_weights):
    """Block verification with explicit uniforms. ``eta [B, K]``: ``eta[:, i - 1]`` decides the
    sub-block of length ``i``; ``u [B]`` picks the correction token. Returns ``accepted [B, K + 1]``
    (accepted drafts, the correction token, ``-1``), ``num_accepted [B]`` and ``tau [B]``.
    ``accept_probs`` / ``residual`` are the replaceable decision functions (see
    :func:`exact_block_output_joint`)."""
    b, k = drafts.shape
    h, w = accept_probs(p, q, drafts)
    hit = eta <= h[:, 1:]
    tau = np.where(hit.any(1), k - np.argmax(hit[:, ::-1], axis=1), 0)
    y = inv_cdf(residual(p, q, w, tau), u)
    return _assemble(drafts, tau, y), tau + 1, tau


def token_verify(p, q, drafts, u_accept, u):
    """Token-wise chain verification with explicit uniforms (accept ``i`` iff
    ``u_i q_i(d_i) < p_i(d_i)``, stop at the first rejection). ``u_accept [B, K]``."""
    b, k = drafts.shape
    idx = np.arange(b)[:, None]
    pos = np.arange(k)[None, :]
    ok = u_accept * q[idx, pos, drafts] < p[idx, pos, drafts]
    tau = np.where(ok.all(1), k, np.argmin(ok, axis=1))
    w = np.ones((b, k + 1))  # the token-wise residual has weight 1
    y = inv_cdf(residual_weights(p, q, w, tau), u)
    return _assemble(drafts, tau, y), tau + 1, tau


# --- small autoregressive models for exactness tests -------------------------------------------
class TreeModel:
    """A random autoregressive model over ``V`` tokens: ``dist(prefix)`` is a fixed random
    distribution per prefix tuple (tables per length, so batches of prefixes are gathers)."""

    def __init__(self, vocab, depth, seed, spread=1.0):
        rng = np.random.default_rng(seed)
        tables = []
        for length in range(depth + 1):
            z = rng.normal(0.0, spread, size=(vocab**length, vocab))
            e = np.exp(z - z.max(-1, keepdims=True))
            tables.append(e / e.sum(-1, keepdims=True))
        self._set_tables(tables)

    def _set_tables(self, tables):
        self.tables = [np.asarray(t, dtype=np.float64) for t in tables]
        self.vocab = self.tables[0].shape[-1]
        self.depth = len(self.tables) - 1

    @classmethod
    def from_tables(cls, tables):
        model = cls.__new__(cls)
        model._set_tables(tables)
        return model

    def map_tables(self, fn):
        """A model whose every conditional is ``fn(table)`` (rows stay distributions)."""
        return TreeModel.from_tables([fn(t) for t in self.tables])

    @classmethod
    def mixture(cls, a, b, weight_b):
        """The model with conditionals ``(1 - weight_b) * a + weight_b * b``."""
        return cls.from_tables(
            [(1.0 - weight_b) * ta + weight_b * tb for ta, tb in zip(a.tables, b.tables)]
        )

    def prefix_index(self, prefix):
        """Index of a prefix (``[..., length]`` int array) in its length's table."""
        idx = np.zeros(prefix.shape[:-1], dtype=np.int64)
        for j in range(prefix.shape[-1]):
            idx = idx * self.vocab + prefix[..., j]
        return idx

    def dist(self, prefix):
        prefix = np.asarray(prefix, dtype=np.int64)
        return self.tables[prefix.shape[-1]][self.prefix_index(prefix)]

    def sample_path(self, n, k, rng):
        """``n`` sampled paths of ``k`` tokens and the distribution rows along each: tokens
        ``[n, k]`` and rows ``[n, k + 1, V]`` (row ``i`` conditions on the path's first ``i``
        tokens, so row ``i`` is the distribution token ``i`` was drawn from)."""
        toks = np.zeros((n, 0), dtype=np.int64)
        rows = []
        for _ in range(k + 1):
            d = self.dist(toks)
            rows.append(d)
            if toks.shape[1] == k:
                break
            nxt = (rng.random((n, 1)) > np.cumsum(d, -1)).sum(-1, keepdims=True)
            toks = np.concatenate([toks, np.minimum(nxt, self.vocab - 1)], axis=1)
        return toks, np.stack(rows, axis=1)

    def rows_along(self, toks, k):
        """The first ``k`` conditional rows ``[n, k, V]`` along given paths ``toks [n, >= k]``
        (a draft model's proposals; ``k + 1`` rows are the target's)."""
        return np.stack([self.dist(toks[:, :i]) for i in range(k)], axis=1)


def target_joint(target, k):
    """Exact joint distribution ``[V] * (k + 1)`` of the first ``k + 1`` target tokens."""
    v = target.vocab
    out = np.zeros((v,) * (k + 1))
    for o in itertools.product(range(v), repeat=k + 1):
        out[o] = np.prod([target.dist(np.array(o[:i]))[o[i]] for i in range(k + 1)])
    return out


def draft_blocks(target, draft, k):
    """Every draft block of positive probability: ``(x, q(x), p [1, K + 1, V], q [1, K, V])``
    with ``p`` / ``q`` the target / draft rows along ``x``."""
    for x in itertools.product(range(target.vocab), repeat=k):
        qx = float(np.prod([draft.dist(np.array(x[:i]))[x[i]] for i in range(k)]))
        if qx <= 0.0:
            continue
        p = np.stack([target.dist(np.array(x[:i])) for i in range(k + 1)])[None]
        q = np.stack([draft.dist(np.array(x[:i])) for i in range(k)])[None]
        yield x, qx, p, q


def exact_block_output_joint(
    target, draft, k, accept_probs=block_accept_probs, residual=residual_weights
):
    """Exact distribution of ``(X^tau, Y, Z^{K - tau})`` (``Z`` completed by the target) for block
    verification, enumerating every draft block: Theorem 1 of the paper says it is the target's
    joint. ``accept_probs`` / ``residual`` are the two decision functions (replaceable, so a test
    can show that a wrong variant is caught)."""
    v = target.vocab
    out = np.zeros((v,) * (k + 1))
    for x, qx, p, q in draft_blocks(target, draft, k):
        h, w = accept_probs(p, q, np.array([x]))
        pmf = block_tau_pmf(h)[0]
        for tau in range(k + 1):
            # A length with probability at rounding level (p == q up to rounding gives tau < K
            # with ~1e-16) can have an empty residual; its mass is below the check tolerance.
            if pmf[tau] <= 1e-13:
                continue
            res = residual(p, q, w, np.array([tau]))[0]
            res = res / res.sum()
            for y in range(v):
                for z in itertools.product(range(v), repeat=k - tau):
                    o = x[:tau] + (y,) + z
                    pz = np.prod(
                        [target.dist(np.array(o[:j]))[o[j]] for j in range(tau + 1, k + 1)]
                    )
                    out[o] += qx * pmf[tau] * res[y] * pz
    return out


def expected_accepted_over_drafts(target, draft, k):
    """``(E[tau] block, E[tau] token-wise)`` averaged over the draft block, exactly."""
    block = token = 0.0
    for x, qx, p, q in draft_blocks(target, draft, k):
        h, _ = block_accept_probs(p, q, np.array([x]))
        block += qx * block_expected_tau(h)[0]
        token += qx * token_expected_tau(p, q, np.array([x]))[0]
    return block, token

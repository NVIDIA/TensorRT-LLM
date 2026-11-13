# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The TP8 one-request draft-probability path (switch id TP8_PROBLL).

What it replaces. At TP8 under rejection-sampling acceptance, the DRAFT_PROB1 op
(``mv2_minimax_m3::minimax_m3_sm107tp8_draft_prob1``) runs the K Markov steps of the draft and writes the proposal
``draft_probs[slot][k][:V] = softmax(lg_k / T)`` in five launches (prep, the Markov kernel, the fault agreement, the
softmax statistics and the proposal write).

This path (only at one generation request, TP world 8, the shapes checked below; anything else runs DRAFT_PROB1):

1. noise (Triton): this rank's Gumbel noise ``g[k, j]`` for its own vocabulary columns, keyed by the request's seed
   and this step's draft offset (``request_offsets + rng_draft_slot(0) * 32``, PROB1's), and ``1 / T`` (1 when
   ``T <= 0``: the greedy draw, noise 0);
2. chain (CuTe DSL, ``kernels/draft_probs_one_request_kernel``): the stock one-launch LL Markov chain kernel
   (``dspark_markov_chain_kernel_b1_ll``: markov_w2 in shared memory, the LL multicast exchange) with one change per
   row: ``lg = bf16(base + bf16(w2 . w1[prev]))`` and the draw compares ``fma(lg, 1 / T, g)`` instead of ``lg``, the
   Gumbel-max draw of ``softmax(lg / T)`` over the whole vocabulary; it also stores ``lg`` (bf16) for the proposal;
3. gather (CuTe DSL, the stock ``allgather_bf16_kernel`` on the ``allgather_cast`` workspace): every rank's ``lg``
   columns into a full ``[K, V]`` bf16 copy on every rank;
4. statistics and 5. write (Triton): per (position, 4096-column chunk) max and exp-sum of ``lg / T``, then
   ``draft_probs[slot][k][:V] = exp(lg / T - M) / S`` (one-hot at the drawn token for ``T <= 0``; ``1 / V`` for the
   CUDA-graph padding slot, as PROB1).

Exactness. The verifier is exact rejection sampling, so its output is the target's distribution for any proposal as
long as the stored q is the distribution each draft token was drawn from (PROB1's own argument). Here every token is
the Gumbel-max draw over ``lg / T`` of the same bf16 ``lg`` the proposal normalizes, with the same stored ``1 / T``.
Not bitwise PROB1's tokens (another random stream) and not its q bit for bit (another reduction order): drafts and so
acceptance can change, while the verified output keeps the target's distribution.

Checks at set-up (every rank, collective, before serving; any failure on any rank = off on every rank, DRAFT_PROB1
serves): greedy tokens bitwise against the stock LL chain kernel with the bf16 sum; ``lg`` against an fp64
recomputation (every element inside the interval any fp32 summation order of the dot allows, at most 1% off the
nominal value); the gathered copy's own columns bitwise and every rank's copy equal; q against torch softmax at T 1
and 0.7, exact one-hot at T 0, exact 1 / V for padding; 2,048 draws at T 1 from 16 candidate tokens (2 per rank)
against q by a chi-square test at every position.

Default ON; kill switch ``TRTLLM_M31_DISABLE_TP8_PROBLL=1`` (DRAFT_PROB1 serves every step). The kernels compile with
plain ``cute.compile`` / Triton at set-up (no AOT cache entry).

Waits. The gather traps after 30 s without a peer (the stock allgather_bf16 kernel). The chain's LL exchange polls
without a bound, exactly as the stock batch-1 LL chain (MARKOV_LL) already serves greedy steps; PROB1's bounded waits
and fault agreement are not carried over, so a dead peer hangs the step there instead of a sticky abort. A fallback
to DRAFT_PROB1 inside a captured step is not possible (the choice is made at capture).

Engagement lines (host only): ``[M3.1 PROBLL] set up on rank r``, ``check on rank r``, ``taken on rank r at
1 rows`` (once, at capture), ``not taken on rank r at <rows> rows: <why>`` (once per reason), ``off on
rank r: <why>``.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

SWITCH = "TRTLLM_M31_DISABLE_TP8_PROBLL"
TAG = "[M3.1 PROBLL]"
WORLD = 8  # the TP8 decode worker only
ROWS = 1  # one generation request (TP8 serves decode batch 1)
MAX_FAST_BATCH = 8  # tp8 dspark_markov_worker.MAX_FAST_BATCH: the S4 buffers' padded shard is padded_rows(shard, 8)
SALT = 0x56524D33_31504C4C  # keeps this Philox key space apart from every other stream
CH = 4096  # proposal columns per program
NOISE_BLOCK = 1024
CHECK_DRAWS = 2048
CHECK_CANDIDATES_PER_RANK = 2
CHI2_Z = 3.719  # one-sided normal quantile of p = 1e-4 (Wilson-Hilferty)
Q_TOL = 2e-5  # max |q - softmax| (fp32 exp of the Triton kernels against torch)
SUM_TOL = 1e-4
LG_OFF_RATE = 0.01  # check (a): at most 1% of the lg elements off their nominal value (expected: far below 0.1%)
LG_CHUNK = 4096  # check (a): shard rows per fp64 recomputation block (8 MB of fp64 w2)

_ON = ("1", "true", "on", "yes")
_OFF = ("", "0", "false", "off", "no")


def _read_switch(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in _ON:
        return True
    if v in _OFF:
        return False
    raise ValueError(f"{name}={raw!r}: use 1/true/on/yes or 0/false/off/no")


ENABLED = not _read_switch(
    SWITCH, False
)  # read once, at import: every step of a process sees the same setting

try:  # the image ships Triton (its sparse-attention kernels use it); a build without it leaves this path off
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _mix64(x):
        """splitmix64's finalizer (uint64 in, uint64 out)."""
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
        x = (x ^ (x >> 27)) * 0x94D049BB133111EB
        return x ^ (x >> 31)

    # Integer operands that may differ between the eager warm-up and the graph capture are not specialized on, so the
    # capture never compiles (do_not_specialize).
    @triton.jit(do_not_specialize=["rng_base", "salt"])
    def _probll_noise(
        noise_ptr,  # fp32 [K * rows_pad], out
        scale_ptr,  # fp32 [1], out: 1 / T (1 when T <= 0)
        temp_ptr,  # fp32 [>= 1]: the request's temperature
        seed_ptr,  # int64 [>= 1]: the request's seed
        off_ptr,  # int64 [>= 1]: the request's offset window base
        rng_base,  # the draft slot's offset inside the window (rng_draft_slot(0) * span)
        salt,
        voff,  # first vocabulary index of this rank's shard
        shard,  # real columns of the shard (the padded tail gets noise 0)
        rows_pad,
        total,  # K * rows_pad
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        e = pid * BLOCK + tl.arange(0, BLOCK)
        live = e < total
        k = e // rows_pad
        row = e - k * rows_pad
        t = tl.load(temp_ptr)
        sampled = t > 0.0
        inv_t = tl.where(sampled, 1.0 / t, 1.0)
        seed = tl.load(seed_ptr).to(tl.uint64, bitcast=True)
        off = (tl.load(off_ptr) + rng_base).to(tl.uint64, bitcast=True)
        key = _mix64(_mix64(seed ^ salt.to(tl.uint64)) ^ off)
        # One Philox4x32-10 counter per (position, vocabulary index): k < 16 and j < 2^18 fit 22 bits.
        ctr = (k * 262144 + voff + row).to(tl.uint32)
        r0, r1, r2, r3 = tl.randint4x(key, ctr)
        # A uniform strictly inside (0, 1) in fp32: 23 bits and a half step. (r0 >> 9) + 0.5 <= 2^23 - 0.5 is exact in
        # fp32 (spacing 0.5 below 2^23) and the scale is a power of two, so u lies in [2^-24, 1 - 2^-24]. (24 bits do
        # not: 2^24 - 0.5 rounds to 2^24 in fp32, so u = 1 and g = +inf.)
        u = ((r0 >> 9).to(tl.float32) + 0.5) * (1.0 / 8388608.0)
        # -log(u) >= -log(1 - 2^-24) ~ 5.96e-8 for an exact log; the floor 2^-25 is reached only by an inexact log near
        # 1 and keeps g finite (<= 17.33) whatever that log's error.
        g = -tl.log(tl.maximum(-tl.log(u), 2.9802322387695312e-08))
        g = tl.where(sampled & (row < shard), g, 0.0)
        tl.store(noise_ptr + e, g, mask=live)
        tl.store(scale_ptr, inv_t, mask=pid == 0)

    @triton.jit
    def _probll_stats(
        lg_ptr,  # bf16 [K, V]: the gathered lg
        scale_ptr,  # fp32 [1]: 1 / T
        pm_ptr,  # fp32 [K, NCH], out: the chunk max of lg / T
        ps_ptr,  # fp32 [K, NCH], out: the chunk sum of exp(lg / T - chunk max)
        V,
        NCH,
        CH_: tl.constexpr,
    ):
        k = tl.program_id(0)
        c = tl.program_id(1)
        cols = c * CH_ + tl.arange(0, CH_)
        m = cols < V
        inv_t = tl.load(scale_ptr)
        x = tl.load(lg_ptr + k * V + cols, mask=m, other=float("-inf")).to(tl.float32) * inv_t
        mx = tl.max(x, axis=0)
        safe = tl.where(mx == float("-inf"), 0.0, mx)
        s = tl.sum(tl.exp(x - safe), axis=0)
        tl.store(pm_ptr + k * NCH + c, mx)
        tl.store(ps_ptr + k * NCH + c, s)

    @triton.jit(do_not_specialize=["dummy_slot", "stride_slot", "stride_k"])
    def _probll_write(
        lg_ptr,  # bf16 [K, V]
        scale_ptr,  # fp32 [1]
        temp_ptr,  # fp32 [>= 1]
        tok_ptr,  # int32 [K]: the drawn tokens
        slot_ptr,  # int64 [>= 1]: the request's slot
        pm_ptr,
        ps_ptr,
        out_ptr,  # fp32 draft_probs [slots, >= K, >= V]
        dummy_slot,  # the CUDA-graph padding slot (0: none published)
        stride_slot,
        stride_k,
        V,
        NCH,
        CH_: tl.constexpr,
        NCHP: tl.constexpr,
    ):
        k = tl.program_id(0)
        c = tl.program_id(1)
        j = tl.arange(0, NCHP)
        jm = j < NCH
        pm = tl.load(pm_ptr + k * NCH + j, mask=jm, other=float("-inf"))
        ps = tl.load(ps_ptr + k * NCH + j, mask=jm, other=0.0)
        big = tl.max(pm, axis=0)
        w = tl.where(pm == float("-inf"), 0.0, tl.exp(pm - big))
        total = tl.sum(ps * w, axis=0)
        cols = c * CH_ + tl.arange(0, CH_)
        m = cols < V
        inv_t = tl.load(scale_ptr)
        x = tl.load(lg_ptr + k * V + cols, mask=m, other=float("-inf")).to(tl.float32) * inv_t
        q = tl.exp(x - big) / total
        t = tl.load(temp_ptr)
        tok = tl.load(tok_ptr + k)
        q = tl.where(t > 0.0, q, tl.where(cols == tok, 1.0, 0.0))
        slot = tl.load(slot_ptr)
        pad = (dummy_slot > 0) & (slot == dummy_slot)
        q = tl.where(pad, 1.0 / V, q)
        tl.store(out_ptr + slot * stride_slot + k * stride_k + cols, q, mask=m)


# ---------------------------------------------------------------------------------------------------- agreement


def _agree(comm, ok: bool) -> bool:
    from tensorrt_llm._torch.distributed.ops import _mnnvl_workspace_all_succeeded

    return bool(_mnnvl_workspace_all_succeeded(comm, bool(ok)))


def _all_equal(comm, value: int, bits: int = 32) -> bool:
    """Whether every rank holds the same ``value`` (low ``bits`` bits), through the boolean agreement alone: for every
    bit, all ranks have it set or all have it clear."""
    v = int(value) & ((1 << bits) - 1)
    same = True
    for b in range(bits):
        one = (v >> b) & 1 == 1
        all_one = _agree(comm, one)
        all_zero = _agree(comm, not one)
        same = same and (all_one or all_zero)
    return same


def _chi2_limit(df: int, z: float = CHI2_Z) -> float:
    """Upper quantile of chi-square(df) at p 1e-4: exact for 1 and 2 degrees of freedom, Wilson-Hilferty above (within
    about 1% from 3 up)."""
    if df == 1:
        return 15.137  # (two-sided normal quantile of 1e-4)^2
    if df == 2:
        return -2.0 * math.log(1e-4)
    a = 2.0 / (9.0 * df)
    return df * (1.0 - a + z * math.sqrt(a)) ** 3


def _chi2(observed: List[float], expected: List[float]) -> Tuple[float, int]:
    """Pearson's statistic with the cells of expected count < 5 merged into one; (statistic, degrees of freedom)."""
    o2, e2, ro, re = [], [], 0.0, 0.0
    for o, e in zip(observed, expected):
        if e >= 5.0:
            o2.append(o)
            e2.append(e)
        else:
            ro += o
            re += e
    if re > 0.0:
        o2.append(ro)
        e2.append(max(re, 1e-12))
    stat = sum((o - e) ** 2 / e for o, e in zip(o2, e2))
    return stat, max(len(o2) - 1, 1)


def _dot_gamma(n: int) -> float:
    """gamma_n = n u / (1 - n u), u = 2^-24: for ANY fp32 summation order of n products (an FMA chain, a tree, cuBLAS
    split-K), |computed - exact| <= gamma_n * sum |products| (Higham, Accuracy and Stability of Numerical Algorithms,
    sections 3.1 and 3.5; exact products from FMA only lower it)."""
    u = 2.0**-24
    if not 0 < n * u < 1:
        raise ValueError(f"n {n}")
    return n * u / (1.0 - n * u)


def _ordered(x: torch.Tensor) -> torch.Tensor:
    """bf16 bit patterns as integers in value order (+0 and -0 equal; one bf16 step = 1), int32."""
    b = x.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(b >= 0, b, -32768 - b)


def _checksum(t: torch.Tensor) -> int:
    """A position-weighted sum of the bits (int16 / int32 views), mod 2^32."""
    flat = t.reshape(-1)
    if flat.dtype in (torch.bfloat16, torch.float16):
        flat = flat.view(torch.int16)
    elif flat.dtype == torch.float32:
        flat = flat.view(torch.int32)
    x = flat.to(torch.int64) & 0xFFFFFFFF
    w = (torch.arange(flat.numel(), device=flat.device, dtype=torch.int64) % 65521) + 1
    return int(((x * w) % (1 << 32)).sum().item()) & 0xFFFFFFFF


# ---------------------------------------------------------------------------------------------------- the path


def _arg(t: torch.Tensor, align: int = 16):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t.detach(), assumed_align=align).mark_layout_dynamic(leading_dim=0)


class ProbLL:
    """The workspace of one TP8 rank's one-request path and its per-step call."""

    def __init__(
        self, p1, mapping, w1: torch.Tensor, w2_shard: torch.Tensor, voff: int, vocab: int, k: int
    ):
        from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
            dspark_markov_chain as chain,
        )

        self.p1 = p1
        self.mapping = mapping
        self.rank = int(mapping.tp_rank)
        self.world = int(mapping.tp_size)
        self.k = int(k)
        self.vocab = int(vocab)
        self.voff = int(voff)
        self.shard = int(w2_shard.shape[0])
        dev = w1.device
        self.device = dev
        self.w1 = w1
        self.rows_pad = int(chain.padded_rows(self.shard, MAX_FAST_BATCH, dev))
        self.grid = int(chain.pick_grid(self.rows_pad, self.k, ROWS, dev))
        self.taken: set = set()
        self.not_taken: set = set()
        self._fn = {}

    # -- local checks (shapes only: every rank decides alike) ---------------------------------------------------
    def why_not(self) -> Optional[str]:
        from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
            dspark_markov_chain as chain,
        )

        if triton is None:
            return "Triton is not importable"
        if self.world != WORLD:
            return f"world {self.world} (PROB-LL is TP8 only)"
        if self.shard * self.world != self.vocab:
            return f"shard {self.shard} x world {self.world} != vocab {self.vocab}"
        if self.shard % 8 != 0:
            return f"shard {self.shard} not a multiple of 8 (the bf16 gather)"
        if not 1 <= self.k <= chain.MAX_BLOCK:
            return f"K {self.k}"
        if self.rows_pad <= 0 or self.grid <= 0:
            return f"no chain geometry (padded rows {self.rows_pad}, grid {self.grid})"
        ll = chain._kernel_module(batch1=True, ll=True)
        if not ll.supports(self.rows_pad, self.grid, self.k, ROWS):
            return f"the LL kernel does not take padded rows {self.rows_pad} at grid {self.grid}"
        if self.k * self.shard * 2 > (1 << 31):
            return "lg too large"
        return None

    # -- build (local + the cached / collective workspaces) -------------------------------------------------------
    def build(self, w2_shard: torch.Tensor) -> None:
        from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.comm import (
            allgather_cast as ag,
        )
        from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
            dspark_markov_chain as chain,
        )

        dev, k = self.device, self.k
        # Collective on first use (every rank, here, before anything of ours can raise); its own agreement inside.
        self.ag_ws = ag.workspace_args(self.mapping, self.shard)
        self.ws = chain.workspace_args(
            self.mapping
        )  # built by prebuild_s4's chain.precompile: cached, no collective
        ll = chain._kernel_module(batch1=True, ll=True)
        if ll.ll_region_words(k, self.ws["slots"], self.grid) > self.ws["buf_words"]:
            raise RuntimeError("the LL exchange does not fit the chain workspace")
        self.ag_grid = int(ag.pick_grid(k, dev, self.world))
        self.w2p = chain.pad_w2_shard(w2_shard.contiguous(), self.rows_pad)
        f32, i32, i64 = torch.float32, torch.int32, torch.int64
        self.noise = torch.zeros(k * self.rows_pad, dtype=f32, device=dev)
        self.scale = torch.ones(1, dtype=f32, device=dev)
        self.tokens = torch.zeros(ROWS, k, dtype=i32, device=dev)
        self.next_new = torch.zeros(ROWS, k + 1, dtype=i32, device=dev)
        self.dummy = torch.zeros(4, dtype=i32, device=dev)
        self.nacc = torch.ones(ROWS, dtype=i32, device=dev)
        self.acc_rows = torch.arange(ROWS, dtype=i32, device=dev)
        self.acc = torch.zeros(ROWS, k + 1, dtype=i32, device=dev)
        self.lg_out = torch.zeros(k, self.shard, dtype=torch.bfloat16, device=dev)
        self.lg_full = torch.zeros(k, self.vocab, dtype=torch.bfloat16, device=dev)
        self.nch = -(-self.vocab // CH)
        self.nchp = 1 << max(0, (self.nch - 1).bit_length())
        self.pm = torch.zeros(k * self.nch, dtype=f32, device=dev)
        self.ps = torch.zeros(k * self.nch, dtype=f32, device=dev)
        self.salt = SALT
        self.live = False  # set-up checks run quiet: the taken line counts serving only
        del i64

    # -- launches ------------------------------------------------------------------------------------------------
    def _stream(self):
        import cuda.bindings.driver as cuda_driver

        return cuda_driver.CUstream(torch.cuda.current_stream(self.device).cuda_stream)

    def _chain_args(
        self, base_flat_i16: Optional[torch.Tensor], prev: Optional[torch.Tensor], lg: bool
    ) -> tuple:
        ws = self.ws
        if (
            base_flat_i16 is None
        ):  # compile only: stand-ins of the served shapes (the layouts are dynamic)
            base_flat_i16 = torch.zeros(
                self.k * self.rows_pad, dtype=torch.int16, device=self.device
            )
            prev = torch.zeros(1, dtype=torch.int64, device=self.device)
        args = [
            _arg(base_flat_i16),
            _arg(prev, align=8),
            _arg(self.w1.contiguous().view(-1).view(torch.int32)),
            _arg(self.w2p.view(-1).view(torch.int32)),
            _arg(self.dummy.view(torch.float32)),
            _arg(self.tokens.view(-1)),
            _arg(self.next_new.view(-1)),
            _arg(self.acc.view(-1), align=4),
            _arg(self.nacc, align=4),
            _arg(self.acc_rows, align=4),
            _arg(self.dummy, align=4),
            _arg(self.dummy, align=4),
            _arg(ws["ws_uc"].view(-1)),
            _arg(ws["ws_mc"].view(-1)),
            _arg(ws["ws_flags"].view(-1)),
        ]
        if lg:
            args += [
                _arg(self.noise),
                _arg(self.scale, align=4),
                _arg(self.lg_out.view(-1).view(torch.int32)),
            ]
        scalars = (int(self.w1.shape[0]), self.voff, self.rank, self.k + 1, 0, 0)
        return tuple(args), scalars

    def _consts(self, last) -> tuple:
        return (
            self.grid,
            self.rows_pad // self.grid,
            self.rows_pad,
            self.k,
            ROWS,
            int(self.ws["slots"]),
            1,
            int(self.ws["buf_words"]),
            True,
            last,
        )

    def launch_chain(self, base_flat_i16: torch.Tensor, prev: torch.Tensor) -> None:
        """The one-request chain: tokens, lg_out (needs noise / scale of this step)."""
        import cutlass.cute as cute

        from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

        from ..kernels import draft_probs_one_request_kernel as pk

        only = base_flat_i16 is None
        args, scalars = self._chain_args(base_flat_i16, prev, True)
        use_pdl = bool(get_env_enable_pdl())
        stream = self._stream()
        key = ("probll", use_pdl)
        fn = self._fn.get(key)
        if fn is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("PROB-LL: the chain must compile outside CUDA-graph capture")
            fn = self._fn[key] = cute.compile(
                pk.k3_markov, *args, *scalars, *self._consts(self.shard), use_pdl, stream
            )
        if only:
            return  # compile only
        fn(*args, *scalars, stream)

    def launch_reference_chain(self, base_flat_i16: torch.Tensor, prev: torch.Tensor) -> None:
        """The stock LL chain kernel with the bf16 sum (fp32_sum off): the greedy parity reference (set-up
        checks only; it shares the LL workspace and its sequence numbers, as every LL call does)."""
        import cutlass.cute as cute

        from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
            dspark_markov_chain as chain,
        )
        from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

        ll = chain._kernel_module(batch1=True, ll=True)
        only = base_flat_i16 is None
        args, scalars = self._chain_args(base_flat_i16, prev, False)
        use_pdl = bool(get_env_enable_pdl())
        stream = self._stream()
        key = ("reference", use_pdl)
        fn = self._fn.get(key)
        if fn is None:
            fn = self._fn[key] = cute.compile(
                ll.k3_markov, *args, *scalars, *self._consts(False), use_pdl, stream
            )
        if only:
            return  # compile only
        fn(*args, *scalars, stream)

    def launch_gather(self, compile_only: bool = False) -> None:
        import cutlass.cute as cute

        from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.comm import (
            allgather_cast as ag,
        )
        from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
            allgather_bf16_kernel as agk,
        )

        w = self.ag_ws
        args = (
            ag._arg(self.lg_out.view(-1).view(torch.int32)),
            ag._arg(self.lg_full.view(-1).view(torch.int32)),
            ag._arg(w["ws_uc"]),
            ag._arg(w["ws_mc"]),
            ag._arg(w["ws_flags"], align=4),
        )
        stream = self._stream()
        fn = self._fn.get("gather")
        if fn is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("PROB-LL: the gather must compile outside CUDA-graph capture")
            fn = self._fn["gather"] = cute.compile(
                agk.allgather_cast,
                *args,
                int(w["rank"]),
                *ag._consts(self.k, self.shard, self.world, self.ag_grid),
                False,
                stream,
            )
        if compile_only:
            return
        fn(*args, int(w["rank"]), stream)

    def launch_noise(self, temperature, seeds, offsets, rng_base: int) -> None:
        total = self.k * self.rows_pad
        _probll_noise[(triton.cdiv(total, NOISE_BLOCK),)](
            self.noise,
            self.scale,
            temperature,
            seeds,
            offsets,
            int(rng_base),
            self.salt,
            self.voff,
            self.shard,
            self.rows_pad,
            total,
            BLOCK=NOISE_BLOCK,
        )

    def launch_proposal(
        self, temperature, slot_ids, dummy_slot_row: int, draft_probs: torch.Tensor
    ) -> None:
        grid = (self.k, self.nch)
        _probll_stats[grid](
            self.lg_full, self.scale, self.pm, self.ps, self.vocab, self.nch, CH_=CH, num_warps=8
        )
        _probll_write[grid](
            self.lg_full,
            self.scale,
            temperature,
            self.tokens,
            slot_ids,
            self.pm,
            self.ps,
            draft_probs,
            int(dummy_slot_row),
            draft_probs.stride(0),
            draft_probs.stride(1),
            self.vocab,
            self.nch,
            CH_=CH,
            NCHP=self.nchp,
            num_warps=8,
        )

    def _decline(self, rows: int, why: str) -> None:
        if not self.live:
            raise RuntimeError(f"set-up call declined: {why}")
        if why not in self.not_taken:
            self.not_taken.add(why)
            logger.info(
                f"{TAG} not taken on rank {self.rank} at {rows} rows: {why}; DRAFT_PROB1 runs"
            )
        return None

    def run(
        self,
        rows,
        base2d,
        prev_in,
        slot_ids,
        dummy_slot_row,
        offsets,
        rng_base,
        span,
        temperature,
        seeds,
        draft_probs,
    ) -> Optional[torch.Tensor]:
        """DraftProb1.run's contract at one request: the int32 tokens [1, K] (a persistent buffer) and
        draft_probs[slot_ids[0], :K, :V]; None = declined (Python-level shape checks, fixed into the captured graph;
        every rank sees the same metadata, so every rank declines alike)."""
        if rows != ROWS:
            return None  # more than one request: DRAFT_PROB1 (TP8 serves one; warmup shapes may differ)
        k = self.k
        if (
            tuple(base2d.shape) != (k, self.rows_pad)
            or base2d.dtype != torch.bfloat16
            or not base2d.is_contiguous()
        ):
            return self._decline(
                rows,
                f"base {tuple(base2d.shape)} {base2d.dtype} (want ({k}, {self.rows_pad}) bf16)",
            )
        if (
            not isinstance(draft_probs, torch.Tensor)
            or draft_probs.dtype != torch.float32
            or draft_probs.dim() != 3
            or draft_probs.stride(2) != 1
            or draft_probs.shape[1] < k
            or draft_probs.shape[2] < self.vocab
            or draft_probs.device != base2d.device
        ):
            return self._decline(rows, "draft_probs layout")
        for name, t, dt in (
            ("temperature", temperature, torch.float32),
            ("seeds", seeds, torch.int64),
            ("offsets", offsets, torch.int64),
            ("slot_ids", slot_ids, torch.int64),
            ("prev", prev_in, torch.int64),
        ):
            if (
                not isinstance(t, torch.Tensor)
                or t.dtype != dt
                or t.numel() < 1
                or t.device != base2d.device
            ):
                return self._decline(rows, f"{name} {getattr(t, 'dtype', None)}")
        if int(span) <= 0:
            return self._decline(rows, f"span {span}")
        self.launch_noise(temperature, seeds, offsets, int(rng_base))
        self.launch_chain(base2d.view(-1).view(torch.int16), prev_in.contiguous().view(-1)[:1])
        self.launch_gather()
        self.launch_proposal(temperature, slot_ids, int(dummy_slot_row), draft_probs)
        if self.live and rows not in self.taken:
            self.taken.add(rows)
            logger.info(f"{TAG} taken on rank {self.rank} at {rows} rows")
        return self.tokens

    # -- set-up checks (collective: every rank, same order) ------------------------------------------------------
    def _scratch(self, temp: float, offset: int = 0, slot: int = 0):
        dev = self.device
        return dict(
            temperature=torch.tensor([temp], dtype=torch.float32, device=dev),
            seeds=torch.tensor([0x1234567], dtype=torch.int64, device=dev),
            offsets=torch.tensor([offset], dtype=torch.int64, device=dev),
            slot_ids=torch.tensor([slot], dtype=torch.int64, device=dev),
        )

    def _base(self, gen: torch.Generator, candidates: Optional[List[int]] = None) -> torch.Tensor:
        """A [K, rows_pad] bf16 test base: this rank's columns normal x 3 (or -inf but at ``candidates``, local
        columns), the padded tail -inf."""
        k, s = self.k, self.shard
        base = torch.full(
            (k, self.rows_pad), float("-inf"), dtype=torch.float32, device=self.device
        )
        vals = torch.randn(k, s, generator=gen, device=self.device) * 3.0
        if candidates is None:
            base[:, :s] = vals
        else:
            for i, c in enumerate(candidates):
                base[:, c] = vals[:, i] * 0.5
        return base.to(torch.bfloat16).contiguous()

    def self_check(self, comm) -> Tuple[bool, str]:
        k, s, v, dev = self.k, self.shard, self.vocab, self.device
        gen = torch.Generator(device=dev)
        gen.manual_seed(20261108 + self.rank)
        notes: List[str] = []
        ok = True
        probs = torch.zeros(4, k, v, dtype=torch.float32, device=dev)

        # (a) greedy parity: T = 0 (noise 0, 1 / T = 1) against the stock LL chain with the bf16 sum.
        base = self._base(gen)
        prev = torch.tensor([4242 % v], dtype=torch.int64, device=dev)
        sc = self._scratch(0.0)
        self.launch_noise(sc["temperature"], sc["seeds"], sc["offsets"], 0)
        self.launch_chain(base.view(-1).view(torch.int16), prev)
        mine = self.tokens.clone()
        lg_mine = self.lg_out.clone()
        self.launch_reference_chain(base.view(-1).view(torch.int16), prev)
        ref = self.tokens.clone()
        torch.cuda.synchronize()
        same = int((mine == ref).sum().item())
        ok &= same == k
        prevs = [int(prev.item())] + [int(t) for t in ref.view(-1)[: k - 1].tolist()]
        # lg against an fp64 recomputation: an analytic tolerance and a rate, not bitwise. The kernel sums the n
        # products of w2 . w1[prev] in fp32 in its own order, so its bias = bf16(dot) may sit one bf16 step from
        # another order's, and through lg = bf16(base + bias) one bias step becomes several lg steps where base and
        # bias nearly cancel. Any fp32 order lies within E = 2 gamma_n sum |w2 w1| of the fp64 dot (the factor 2
        # also covers the fp64 reference), and every later rounding is monotone, so each lg must lie in
        # [lg(dot - E), lg(dot + E)], lg(x) = bf16(base + bf16(fp32(x))): every element inside, and at most
        # LG_OFF_RATE of them off the nominal lg(dot) (a rounding-mode or order bug moves far more). torch's fp32
        # GEMV (another order) is reported in bf16 steps, INFO only.
        gdot = 2.0 * _dot_gamma(int(self.w2p.shape[1]))
        w1d = self.w1.index_select(
            0, torch.tensor(prevs, dtype=torch.long, device=dev)
        ).double()  # [K, n]
        w1a, w1f = w1d.abs(), w1d.float()
        inside = off_nom = bitwise = one = further = 0
        for r0 in range(
            0, s, LG_CHUNK
        ):  # 8 MB of fp64 w2 per block; markov_w1 holds only the K rows read
            r1 = min(s, r0 + LG_CHUNK)
            w2d = self.w2p[r0:r1].double()  # [c, n]
            dot = w1d @ w2d.T  # [K, c]
            err = (w1a @ w2d.abs().T) * gdot
            b = base[:, r0:r1].float()
            lo = (b + (dot - err).float().to(torch.bfloat16).float()).to(torch.bfloat16).float()
            nom = (b + dot.float().to(torch.bfloat16).float()).to(torch.bfloat16).float()
            hi = (b + (dot + err).float().to(torch.bfloat16).float()).to(torch.bfloat16).float()
            mine = lg_mine[:, r0:r1]
            mf = mine.float()
            inside += int(((mf >= lo) & (mf <= hi)).sum().item())
            off_nom += int((mf != nom).sum().item())
            t32 = (b + (w1f @ w2d.float().T).to(torch.bfloat16).float()).to(torch.bfloat16)
            steps = (_ordered(mine) - _ordered(t32)).abs()
            bitwise += int((steps == 0).sum().item())
            one += int((steps == 1).sum().item())
            further += int((steps > 1).sum().item())
        total = k * s
        lim = int(LG_OFF_RATE * total)
        ok &= inside == total and off_nom <= lim
        notes.append(
            f"greedy tokens {same}/{k} bitwise vs the stock LL chain (bf16 sum); lg {inside}/{total} "
            f"inside the fp32-order interval of an fp64 recomputation (E = {gdot:.2e} x sum|w2 w1|), "
            f"{off_nom} off its nominal value (limit {lim}); vs torch fp32 (INFO) {bitwise} bitwise, {one} "
            f"one bf16 step, {further} further"
        )

        # (b) + (c) the full path at T = 1, 0.7, 0 and the padding slot.
        errs = []
        for idx, temp in enumerate((1.0, 0.7, 0.0)):
            sc = self._scratch(temp, offset=512 * (idx + 1), slot=idx)
            self.run(
                1,
                base.view(k, self.rows_pad),
                prev,
                sc["slot_ids"],
                0,
                sc["offsets"],
                0,
                32,
                sc["temperature"],
                sc["seeds"],
                probs,
            )
            torch.cuda.synchronize()
            own = torch.equal(self.lg_full[:, self.voff : self.voff + s], self.lg_out)
            ok &= own
            q = probs[idx, :k, :v]
            if temp > 0:
                ref_q = torch.softmax(self.lg_full.float() * self.scale.item(), dim=-1)
                err = float((q - ref_q).abs().max().item())
                sums = float((q.sum(-1) - 1.0).abs().max().item())
                at_tok = bool((q.gather(1, self.tokens.view(k, 1).long()) > 0).all().item())
                ok &= err <= Q_TOL and sums <= SUM_TOL and at_tok
                errs.append(
                    f"T {temp}: q max err {err:.1e}, sum err {sums:.1e}, drawn tokens q > 0 {at_tok}"
                )
            else:
                onehot = torch.zeros_like(q)
                onehot.scatter_(1, self.tokens.view(k, 1).long(), 1.0)
                exact1 = torch.equal(q, onehot)
                ok &= exact1
                errs.append(f"T 0: one-hot exact {exact1}")
            if idx == 0:
                eq = _all_equal(
                    comm, _checksum(self.lg_full) ^ _checksum(self.tokens) ^ _checksum(q)
                )
                ok &= eq
                errs.append(f"gathered copy own columns bitwise {own}, every rank equal {eq}")
        sc = self._scratch(1.0, offset=4096, slot=3)
        self.run(
            1,
            base.view(k, self.rows_pad),
            prev,
            sc["slot_ids"],
            3,
            sc["offsets"],
            0,
            32,
            sc["temperature"],
            sc["seeds"],
            probs,
        )
        torch.cuda.synchronize()
        pad_ok = bool(((probs[3, :k, :v] - 1.0 / v).abs() <= 1e-6 / v).all().item())
        ok &= pad_ok
        errs.append(f"padding slot 1/V exact {pad_ok}")
        notes += errs

        # (d) draws: 16 candidate tokens (2 per rank), T = 1, CHECK_DRAWS draws with fresh offsets.
        cands_local = [(s // 3) * (i + 1) for i in range(CHECK_CANDIDATES_PER_RANK)]
        dgen = torch.Generator(device=dev)
        dgen.manual_seed(77 + self.rank)
        base_d = self._base(dgen, cands_local)
        cand = torch.tensor(
            [r * s + c for r in range(self.world) for c in cands_local],
            dtype=torch.long,
            device=dev,
        )
        n = CHECK_DRAWS
        toks = torch.zeros(n, k, dtype=torch.int64, device=dev)
        qc = torch.zeros(n, k, cand.numel(), dtype=torch.float32, device=dev)
        for d in range(n):
            sc = self._scratch(1.0, offset=8192 + 1024 * d, slot=0)
            self.run(
                1,
                base_d.view(k, self.rows_pad),
                prev,
                sc["slot_ids"],
                0,
                sc["offsets"],
                0,
                32,
                sc["temperature"],
                sc["seeds"],
                probs,
            )
            toks[d].copy_(self.tokens.view(-1))
            qc[d].copy_(probs[0, :k].index_select(1, cand))
        torch.cuda.synchronize()
        in_cand = bool(torch.isin(toks, cand).all().item())
        ok &= in_cand
        lines = []
        for pos in range(k):
            obs = [float((toks[:, pos] == c).sum().item()) for c in cand.tolist()]
            exp_ = [float(x) for x in qc[:, pos].sum(0).tolist()]
            stat, df = _chi2(obs, exp_)
            lim = _chi2_limit(df)
            ok &= stat <= lim
            lines.append(f"{stat:.1f}/{lim:.1f} (df {df})")
        notes.append(
            f"draws {n} at T 1 over {cand.numel()} candidates: all drawn tokens candidates {in_cand}; "
            f"chi2/limit by position {', '.join(lines)}"
        )
        return bool(ok), "; ".join(notes)

    def describe(self) -> str:
        return (
            f"world {self.world} K {self.k} shard {self.shard} (padded {self.rows_pad}, chain grid {self.grid}) "
            f"vocab {self.vocab}; gather grid {self.ag_grid}; proposal {self.nch} x {CH} columns; Gumbel key "
            f"stream salt {SALT:#x}"
        )


def attach(p1, mapping, comm, w1, w2_shard, voff: int, vocab: int, k: int) -> Optional[ProbLL]:
    """Set the one-request path up for this DraftProb1 (collective: every rank of the TP group, inside
    DraftProb1.create, same order). None = off (DRAFT_PROB1 serves every step)."""
    world = int(getattr(mapping, "tp_size", 0))
    rank = int(getattr(mapping, "tp_rank", -1))
    if world != WORLD:
        return None  # TP2 / TP4 share draft_prob1.py: no line, no collective
    if not ENABLED:
        logger.info(f"{TAG} off on rank {rank}: {SWITCH} set")
        return None
    st, why = None, None
    try:
        st = ProbLL(p1, mapping, w1, w2_shard, voff, vocab, k)
        why = st.why_not()
    except Exception as exc:  # noqa: BLE001 -- reported through the agreement
        why = f"set-up: {type(exc).__name__}: {exc}"
    if not _agree(comm, why is None):
        logger.info(f"{TAG} off on rank {rank}: {why or 'another rank declined'}")
        return None
    # Collective allocations next (the bf16 gather workspace), then the local build and compiles.
    try:
        st.build(w2_shard)
        sc = st._scratch(1.0)
        st.launch_noise(
            sc["temperature"], sc["seeds"], sc["offsets"], 0
        )  # Triton compiles (local kernels)
        st.launch_proposal(
            sc["temperature"],
            sc["slot_ids"],
            0,
            torch.zeros(1, st.k, st.vocab, dtype=torch.float32, device=st.device),
        )
        st.launch_chain(None, None)  # CuTe compiles only (no launch: these kernels are collective)
        st.launch_reference_chain(None, None)
        st.launch_gather(compile_only=True)
        torch.cuda.synchronize()
        why = None
    except Exception as exc:  # noqa: BLE001
        why = f"build: {type(exc).__name__}: {exc}"
    if not _agree(comm, why is None):
        logger.info(f"{TAG} off on rank {rank}: {why or 'another rank failed its build'}")
        return None
    try:
        ok, line = st.self_check(comm)
    except Exception as exc:  # noqa: BLE001 -- a rank that raises mid-check is reported through the agreement
        ok, line = False, f"check raised {type(exc).__name__}: {exc}"
    logger.info(f"{TAG} check on rank {rank}: {line} -> {'PASS' if ok else 'FAIL'}")
    if not _agree(comm, ok):
        logger.info(
            f"{TAG} off on rank {rank}: {'the set-up check failed on another rank' if ok else 'check FAIL'}"
        )
        return None
    st.taken.clear()
    st.not_taken.clear()
    st.live = True
    logger.info(f"{TAG} set up on rank {rank}: {st.describe()}")
    return st

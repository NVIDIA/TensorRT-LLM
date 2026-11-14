# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Flag-free gather for the TP8 decode worker (TP8_FLAGFREE_GATHER).

Two all-gathers run every TP8 decode step on the one-launch multicast gather (ops/comm/allgather_cast):
  1. the verify logits (dspark.target_logits, N3 A1): bf16 [8, 25008] per rank -> fp32 [8, 200064];
  2. the drafter fc output (dspark._f7_gather, FOLD_F7 on FC_SHARD): bf16 [8, 768] per rank, gathered as fp32 words
     [8, 384] -> [8, 3072].
Their cost is the gather's fixed protocol latency (a system-scope release reduction on a flag, then acquire polls of the
peers' flags), not the bytes: the fc output moves 12 KB per rank against the verify logits' 400 KB at about the same
time.

What this does: both calls run on ``kernels/flagfree_gather_kernel`` instead (the same gather with a flag-free
exchange: the pushed words carry their own readiness, the reader polls them and re-arms them; one one-way NVLink
latency). Same inputs, same output tensor, the same bits except NaN words of one pattern (the kernel file says
which). Only on the TP8 decode worker (tensor-parallel world 8), only for the shapes compiled here (verify rows
(K + 1) x b up to VERIFY_CAP_ROWS, fc rows 1..FC_CAP_ROWS) and only where the stock gather is set up for the same call
(the call sites ask this module only inside the stock gather's taken branch); any other call runs the stock gather.

Set-up (prebuild, every rank, collective, before any capture; any failure on any rank = off on every rank): the
multicast workspaces (filled with EMPTY), the compiles (plain ``cute.compile``), then the CHECK: CHECK_CALLS rounds
over every compiled shape of both gathers, interleaved (both parities, every row count, re-arm across shapes), each on
fresh seeded data that every rank regenerates for every peer (random bit patterns + every special word: +-0, +-inf,
NaNs, subnormals, max, EMPTY itself, CANON); the flag-free output must equal the expected words (the reference with
every EMPTY word as CANON) everywhere, and the stock gather run on the same data must equal the reference (which
validates the reference).

Default ON; kill switch ``TRTLLM_M31_DISABLE_TP8_FLAGFREE_GATHER=1`` (the stock gathers serve every step).

Engagement lines (host only): ``[M3.1 FLAGFREE_GATHER] set up on rank r``, ``check on rank r: ... -> PASS|FAIL``,
``taken on rank r: verify logits at <rows> rows`` / ``fc output at <rows> rows`` (once each,
at warm-up or capture), ``not taken on rank r: <site> at <rows> rows: <why>`` (once per reason),
``off on rank r: <why>``.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

SWITCH = "TRTLLM_M31_DISABLE_TP8_FLAGFREE_GATHER"
TAG = "[M3.1 FLAGFREE_GATHER]"
WORLD = 8  # the TP8 decode worker only
VERIFY_COLS = 25008  # bf16 shard columns of the verify logits (vocabulary 200064 / 8)
FC_COLS = 384  # fp32 words per row of the fc output shard (768 bf16 columns: hidden 6144 / 8)
VERIFY_CAP_ROWS = 16  # verify rows the workspace holds (K7 at batch 1 and 2; the TP8 decode worker serves batch 1)
FC_CAP_ROWS = 8  # fc rows (FC_SHARD's own bound: the K7 verify rows at batch 1)
MAX_GRID = 152  # CTAs per call at most (the stock gather's MAX_GRID; every CTA co-resident)
CHECK_CALLS = 6  # check rounds (both parities three times)
SPECIAL_WORDS = (
    0x00000000,
    0x80000000,
    0x00008000,
    0x80008000,  # +-0.0 in every half
    0xFFFFFFFF,
    0x7FFF7FFF,
    0xFFFF0000,
    0x0000FFFF,  # EMPTY, CANON, half-EMPTY patterns
    0x7F807F80,
    0xFF80FF80,
    0x7FC07FC0,
    0xFFC1FFC1,  # +-inf, NaNs
    0x00010001,
    0x807F807F,
    0x7F7F7F7F,
    0xFF7FFF7F,  # subnormals, max finite
)

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


def _arg(t: torch.Tensor, align: int = 16):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t.detach(), assumed_align=align).mark_layout_dynamic(leading_dim=0)


def _agree(comm, ok: bool) -> bool:
    from tensorrt_llm._torch.distributed.ops import _mnnvl_workspace_all_succeeded

    return bool(_mnnvl_workspace_all_succeeded(comm, bool(ok)))


def _i32(words) -> int:
    """An unsigned 32-bit word as the int32 torch stores."""
    w = int(words) & 0xFFFFFFFF
    return w - (1 << 32) if w >= (1 << 31) else w


def pick_grid(n_vec: int, sms: int) -> int:
    """CTAs for a call of n_vec vectors: one vector per thread up to MAX_GRID CTAs (and the SM count), at least 1."""
    from ..kernels import flagfree_gather_kernel as kern

    return max(1, min(MAX_GRID, sms, (n_vec + kern.THREADS - 1) // kern.THREADS))


class _Gather:
    """One flag-free gather: its multicast workspace, flags and compiled shapes (rows -> (fn, grid))."""

    def __init__(
        self, site: str, mapping, comm, cols: int, fp32: bool, cap_rows: int, rows: Tuple[int, ...]
    ):
        from ..kernels import flagfree_gather_kernel as kern

        self.kern = kern
        self.site, self.mapping, self.comm = site, mapping, comm
        self.cols, self.fp32, self.cap_rows = int(cols), bool(fp32), int(cap_rows)
        self.rows = tuple(int(r) for r in rows if 0 < int(r) <= cap_rows)
        self.world, self.rank = int(mapping.tp_size), int(mapping.tp_rank)
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.slot = kern.slot_words(self.cap_rows, self.cols, self.fp32)
        self.words = kern.workspace_words(self.cap_rows, self.cols, self.world, self.fp32)
        self.fn: Dict[int, tuple] = {}
        self.handle = self.uc = self.mc = self.flags = None

    def why_not(self) -> Optional[str]:
        if not self.rows:
            return f"{self.site}: no row count within the cap {self.cap_rows}"
        if self.cols <= 0 or self.cols % (4 if self.fp32 else 8):
            return f"{self.site}: {self.cols} shard columns are not whole 16-byte vectors"
        return None

    def build(self) -> None:
        """The multicast workspace (collective: every rank of the TP group, same order), filled with EMPTY."""
        from tensorrt_llm._torch.distributed.ops import _make_mnnvl_mcast_buffer

        use_fabric = (
            os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1" or self.mapping.is_multi_node()
        )
        self.handle = _make_mnnvl_mcast_buffer(self.comm, self.words * 4, self.mapping, use_fabric)
        self.uc = self.handle.get_uc_buffer(self.rank, (self.words,), torch.int32, 0)
        self.mc = self.handle.get_mc_buffer((self.words,), torch.int32, 0)
        with torch.inference_mode():
            self.uc.fill_(self.kern.EMPTY_WORD)
            self.flags = torch.zeros(self.kern.FLAG_WORDS, dtype=torch.int32, device=self.uc.device)
        torch.cuda.synchronize()

    def _args(self, shard: torch.Tensor, out: torch.Tensor) -> tuple:
        return (
            _arg(shard.view(-1).view(torch.int32)),
            _arg(out.view(-1).view(torch.int32)),
            _arg(self.uc.view(-1)),
            _arg(self.mc.view(-1)),
            _arg(self.flags.view(-1), align=4),
        )

    def _shard_dtype(self):
        return torch.float32 if self.fp32 else torch.bfloat16

    def compile_all(self) -> None:
        """Compile every row count (launches nothing; outside capture)."""
        import cuda.bindings.driver as cuda_driver
        import cutlass.cute as cute

        stream = cuda_driver.CUstream(torch.cuda.current_stream(self.device).cuda_stream)
        for r in self.rows:
            grid = pick_grid(self.kern.vectors(r, self.cols, self.fp32), self.sms)
            shard = torch.zeros(r, self.cols, dtype=self._shard_dtype(), device=self.device)
            out = torch.empty(r, self.world * self.cols, dtype=torch.float32, device=self.device)
            fn = cute.compile(
                self.kern.flagfree_gather,
                *self._args(shard, out),
                int(self.rank),
                grid,
                r,
                self.cols,
                self.world,
                self.slot,
                self.fp32,
                stream,
            )
            self.fn[r] = (fn, grid)

    def takes(self, shard: torch.Tensor) -> Optional[str]:
        """None when this gather takes the shard, else why not (shapes and set-up only: every rank alike)."""
        if shard.dim() != 2 or shard.dtype != self._shard_dtype() or not shard.is_cuda:
            return f"layout {tuple(shard.shape)} {shard.dtype}"
        if int(shard.shape[1]) != self.cols:
            return f"{int(shard.shape[1])} columns (set up for {self.cols})"
        if int(shard.shape[0]) not in self.fn:
            return f"rows not compiled here (compiled {sorted(self.fn)})"
        if not shard.is_contiguous() or shard.data_ptr() % 16:
            return "not contiguous or not 16-byte aligned"
        return None

    def run(self, shard: torch.Tensor) -> torch.Tensor:
        """The gathered fp32 [rows, world * cols] (collective: every rank, same order); the caller checked takes()."""
        import cuda.bindings.driver as cuda_driver

        rows = int(shard.shape[0])
        fn, _ = self.fn[rows]
        out = torch.empty(rows, self.world * self.cols, dtype=torch.float32, device=shard.device)
        stream = cuda_driver.CUstream(torch.cuda.current_stream(shard.device).cuda_stream)
        fn(*self._args(shard, out), int(self.rank), stream)
        return out

    def describe(self) -> str:
        grids = sorted({g for _, g in self.fn.values()})
        return (
            f"{self.site} rows {list(self.rows)} x {self.cols} {'fp32' if self.fp32 else 'bf16->fp32'} words, "
            f"grids {grids}, workspace {self.words * 4 / 2**20:.2f} MiB"
        )


class FlagFreeGather:
    """The two TP8 decode gathers on the flag-free kernel. ``verify_logits`` / ``fc_output`` return the gathered tensor
    or None (the stock gather then runs; shapes decide, so every rank declines alike)."""

    def __init__(
        self, rank: int, verify: Optional[_Gather], fc: Optional[_Gather], stock_verify, stock_fc
    ):
        self.rank = rank
        self.verify, self.fc = verify, fc
        self.stock_verify, self.stock_fc = (
            stock_verify,
            stock_fc,
        )  # the stock gathers' workspace args (check)
        self.live = False
        self._logged = set()

    def _log_once(self, key, msg: str) -> None:
        if key not in self._logged:
            self._logged.add(key)
            logger.info(msg)

    def _serve(
        self, g: Optional[_Gather], site: str, shard: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if not self.live or g is None:
            return None
        rows = int(shard.shape[0]) if shard.dim() == 2 else -1
        why = g.takes(shard)
        if why is not None:
            self._log_once(
                (site, rows, why),
                f"{TAG} not taken on rank {self.rank}: {site} at {rows} rows: {why}; "
                "the stock gather runs",
            )
            return None
        self._log_once((site, rows), f"{TAG} taken on rank {self.rank}: {site} at {rows} rows")
        return g.run(shard)

    def verify_logits(self, shard: torch.Tensor) -> Optional[torch.Tensor]:
        """fp32 [rows, world * 25008]: allgather_cast(shard)'s output, from the bf16 verify shard."""
        return self._serve(self.verify, "verify logits", shard)

    def fc_output(self, shard_f32: torch.Tensor) -> Optional[torch.Tensor]:
        """fp32 words [rows, world * 384]: gather_last_dim_f32(shard)'s output (the bf16 fc columns as
        fp32 words)."""
        return self._serve(self.fc, "fc output", shard_f32)

    # --------------------------------------------------------------------------------------------------- check

    def _gen(self, g: _Gather, peer: int, rows: int, call: int) -> torch.Tensor:
        """Peer ``peer``'s check shard for (gather, rows, call): seeded random 16-bit patterns (every rank regenerates
        every peer's), the special words at the start of every row and EMPTY at a few seeded places."""
        site_id = 1 if g is self.verify else 2
        seed = ((site_id * 1000003 + peer) * 1009 + rows) * 7919 + call
        gen = torch.Generator(device=g.device)
        gen.manual_seed(seed)
        rw = g.kern.row_words(g.cols, g.fp32)
        half = torch.randint(
            -32768, 32768, (rows, 2 * rw), dtype=torch.int16, device=g.device, generator=gen
        )
        words = half.view(torch.int32)  # [rows, rw]
        sp = torch.tensor([_i32(w) for w in SPECIAL_WORDS], dtype=torch.int32, device=g.device)
        k = min(rw, sp.numel())
        words[:, :k] = sp[:k]
        pos = torch.randint(k, rw, (rows, 3), device=g.device, generator=gen) if rw > k else None
        if pos is not None:
            words.scatter_(1, pos, _i32(0xFFFFFFFF))
        return words.view(torch.float32) if g.fp32 else words.view(torch.bfloat16)

    def _expected(self, g: _Gather, rows: int, call: int):
        """(reference, expected flag-free, this rank's shard): reference and expected as int32 [rows, world * cols]
        (expected = the reference with every rank's EMPTY words as CANON)."""
        shards = [self._gen(g, p, rows, call) for p in range(g.world)]
        if g.fp32:
            ref = torch.cat([s.view(torch.int32) for s in shards], dim=-1)
        else:
            parts = []
            for s in shards:
                x = s.view(torch.int16).to(torch.int64) & 0xFFFF
                y = x << 16
                parts.append(torch.where(y >= (1 << 31), y - (1 << 32), y).to(torch.int32))
            ref = torch.cat(parts, dim=-1)
        exp = ref.clone().view(rows, g.world, g.cols)
        for p in range(
            g.world
        ):  # every rank's columns, this rank's own included (the kernel canonicalizes both)
            hit = shards[p].view(torch.int32) == -1  # [rows, words per row]
            if g.fp32:
                exp[:, p, :][hit] = _i32(g.kern.CANON_WORD)
            else:
                exp[:, p, :][hit.repeat_interleave(2, dim=1)] = _i32(0x7FFF0000)
        return ref, exp.view(rows, g.world * g.cols), shards[g.rank]

    def _stock(self, g: _Gather, shard: torch.Tensor) -> torch.Tensor:
        from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.comm import (
            allgather_cast as ag,
        )

        if g is self.verify:
            return ag.allgather_cast(shard, **self.stock_verify)
        return ag.gather_last_dim_f32(shard, **self.stock_fc)

    def self_check(self) -> Tuple[bool, str]:
        """CHECK_CALLS rounds over every compiled shape of both gathers (collective: every rank, same order)."""
        ok = True
        notes = []
        gathers = [g for g in (self.verify, self.fc) if g is not None]
        stats = {
            g.site: {"calls": 0, "words": 0, "bad": 0, "stock_bad": 0, "canon": 0} for g in gathers
        }
        for call in range(CHECK_CALLS):
            for g in gathers:
                order = g.rows if call % 2 == 0 else tuple(reversed(g.rows))
                for rows in order:
                    ref, exp, mine_in = self._expected(g, rows, call)
                    got = g.run(mine_in).view(torch.int32)
                    stock = self._stock(g, mine_in).view(torch.int32)
                    torch.cuda.synchronize()
                    st = stats[g.site]
                    st["calls"] += 1
                    st["words"] += exp.numel()
                    st["bad"] += int((got != exp).sum().item())
                    st["stock_bad"] += int((stock != ref).sum().item())
                    st["canon"] += int((exp != ref).sum().item())
        for g in gathers:
            st = stats[g.site]
            ok &= st["bad"] == 0 and st["stock_bad"] == 0 and st["calls"] > 0
            notes.append(
                f"{g.site} rows {list(g.rows)} x {CHECK_CALLS} rounds ({st['calls']} calls, {st['words']} words): "
                f"{st['bad']} differ from the expected ({st['canon']} EMPTY-pattern NaN words expected as CANON), "
                f"stock gather vs reference {st['stock_bad']} differ"
            )
        return bool(ok), "; ".join(notes)

    def describe(self) -> str:
        return "; ".join(g.describe() for g in (self.verify, self.fc) if g is not None)


def attach(worker, lm_head) -> Optional[FlagFreeGather]:
    """Set the flag-free gather up for this TP8 DSpark worker (collective: every rank, in prebuild, right after the
    stock gathers are set up, same order). None = off (the stock gathers serve every step)."""
    mapping = getattr(worker, "mapping", None)
    world = int(getattr(mapping, "tp_size", 0) or 0)
    rank = int(getattr(mapping, "tp_rank", -1))
    if world != WORLD:
        return None  # no line, no collective
    if not ENABLED:
        logger.info(f"{TAG} off on rank {rank}: {SWITCH} set")
        return None
    a1 = getattr(worker, "_a1", None)
    f7 = getattr(worker, "_f7_ws", None)
    if a1 is None and f7 is None:
        # Both decisions were collective (every rank alike), so this one is too.
        logger.info(f"{TAG} off on rank {rank}: neither stock gather is set up (A1 / FOLD_F7 off)")
        return None
    from tensorrt_llm._torch.distributed.ops import _get_mnnvl_workspace_comm

    comm = _get_mnnvl_workspace_comm(mapping)  # collective (every TP rank, same point)
    st, why = None, None
    try:
        k1 = int(worker.max_draft_len) + 1
        verify = fc = None
        if a1 is not None:
            cols = int(getattr(lm_head, "out_features", 0))
            verify = _Gather(
                "verify logits",
                mapping,
                comm,
                cols,
                False,
                VERIFY_CAP_ROWS,
                tuple(b * k1 for b in range(1, VERIFY_CAP_ROWS // k1 + 1)),
            )
        if f7 is not None:
            fc = _Gather(
                "fc output",
                mapping,
                comm,
                int(f7[0]),
                True,
                FC_CAP_ROWS,
                tuple(range(1, FC_CAP_ROWS + 1)),
            )
        why = next((w for w in (g.why_not() for g in (verify, fc) if g is not None) if w), None)
        if why is None and verify is not None and verify.cols != VERIFY_COLS:
            why = f"verify shard {verify.cols} columns (built for {VERIFY_COLS})"
        if why is None and fc is not None and fc.cols != FC_COLS:
            why = f"fc shard {fc.cols} fp32 words (built for {FC_COLS})"
        st = FlagFreeGather(rank, verify, fc, a1, f7[1] if f7 is not None else None)
    except Exception as exc:  # noqa: BLE001 -- reported through the agreement
        why = f"set-up: {type(exc).__name__}: {exc}"
    if not _agree(comm, why is None):
        logger.info(f"{TAG} off on rank {rank}: {why or 'another rank declined'}")
        return None
    # The collective allocations, in the same order on every rank.
    try:
        for g in (st.verify, st.fc):
            if g is not None:
                g.build()
        why = None
    except Exception as exc:  # noqa: BLE001
        why = f"workspace: {type(exc).__name__}: {exc}"
    # Also the barrier that keeps any rank from pushing before every peer filled its workspace with EMPTY.
    if not _agree(comm, why is None):
        logger.info(f"{TAG} off on rank {rank}: {why or 'another rank failed its workspace'}")
        return None
    try:
        for g in (st.verify, st.fc):
            if g is not None:
                g.compile_all()
        torch.cuda.synchronize()
        why = None
    except Exception as exc:  # noqa: BLE001
        why = f"compile: {type(exc).__name__}: {exc}"
    if not _agree(comm, why is None):
        logger.info(f"{TAG} off on rank {rank}: {why or 'another rank failed its compiles'}")
        return None
    try:
        ok, line = st.self_check()
    except Exception as exc:  # noqa: BLE001 -- a rank that raises mid-check is reported through the agreement
        ok, line = False, f"check raised {type(exc).__name__}: {exc}"
    logger.info(f"{TAG} check on rank {rank}: {line} -> {'PASS' if ok else 'FAIL'}")
    if not _agree(comm, ok):
        logger.info(
            f"{TAG} off on rank {rank}: {'the set-up check failed on another rank' if ok else 'check FAIL'}"
        )
        return None
    st.live = True
    logger.info(f"{TAG} set up on rank {rank}: {st.describe()}")
    return st

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DRAFT_PROB1: the probabilistic DSpark Markov draft (dk_mk_prob1, adapted from an NVIDIA reference implementation)
in the TP4 / TP8 DSpark worker.

One draft (all K Markov steps of up to 8 requests at TP2 / TP4, 4 at TP8) is one op,
``mv2_minimax_m3::minimax_m3_sm107tp4_draft_prob1`` (the world-4 kernel) or
``mv2_minimax_m3::minimax_m3_sm107tp8_draft_prob1``
(the world-templated w8 kernel, dk_mk_prob1_w8; world <= 8):

1. prep: the per-row request index (-1 = CUDA-graph padding) and Philox position of each (request, step);
2. the reference dk_mk_prob1 (unchanged math, full-width exchange tags and a uniform abort gate): step s adds w2s
@ w1[token s-1] to this rank's shard of the block
   logits, writes the bf16 sum into every rank's draft_logits cache (mirror) and draws the step's
token by Gumbel-max
   over the full vocab (one (key, id) word per row and step across the TP ranks) -- q = softmax(lg /
T), temperature
   only;
3. agree: the reference kernel's waits are bounded (they set err instead of trapping); every rank publishes its
err to every rank
   and a rank0 system-CAS decision linearizes success or sticky abort for each sequence;
4. the adapter: draft_probs[slot][s][:V] = softmax(lg / T) of the same bf16 logits (one-hot for T ==
0, uniform for
   padding), tokens32 = the draws; on an agreed fault, token 0 and a one-hot q at 0 for every
position of every row.

Correctness: the verifier is exact rejection sampling, so its output is the target's distribution
for ANY proposal as
long as the stored q is the distribution each draft token was drawn from; it is (the same bf16
logits, the same
temperature). Dropping the request's draft-side top-k / top-p / min-p and conditioning step s on the
drafted token
s-1 (the in-tree S4-logits path conditions on the greedy chain) moves acceptance only. So it engages
only when the
verifier is rejection sampling (``engages``), never under strict / exact-match acceptance.

Workspace (``create``, collective, outside capture): one MNNVL allocation for the exchange slots +
fault/control words (peers
write them), one for the draft_logits cache [rows, K, V] bf16 (peers write their columns), the rest local.
Kill switch: TRTLLM_M31_DISABLE_DRAFT_PROB1=1, read at import. Hang cap of the fault agreement: (1 << 31) * 200 ns,
the boundary all-reduces' cap."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.m31_switches import (
    m31_disabled,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops

DISABLE_ENV = "TRTLLM_M31_DISABLE_DRAFT_PROB1"
_DISABLED = m31_disabled("DRAFT_PROB1")
# Hang cap of the fault agreement, in ns: the boundary all-reduces' cap.
HANG_CAP_NS = (1 << 31) * 200

MAX_ROWS = 8  # dkmkp::kMaxC (TP2 / TP4)
MAX_WORLD = 4  # dkmkp::kMaxW
MAX_ROWS_TP8 = 4  # dkmkpw::Lim<8>::kMaxC
MAX_WORLD_TP8 = 8  # dkmkpw::Lim<8>::kMaxW
MAX_K = 16  # dkmkp1::kMaxK
MARKOV_RANK = 256  # dkmkp::kR
MAX_GRID = 512  # dkmkp::kMaxGrid
EXCHANGE_STORAGE_WORDS = 256  # slots/fault controls/full64 epoch tags, in uint64 words
EPOCH_TAG_OFFSET = 128  # full generation tags parallel the64 payload slots
SLOT_WORDS = 2 * MAX_ROWS * MAX_WORLD  # [2 parity][kMaxC][kMaxW] u64
STAT_SPLIT = 8  # draftProb1::kStatSplit
MAX_VOCAB = 1 << 18  # the exchange packs token ids < 2^18
OP = "minimax_m3_sm107tp4_draft_prob1"
OP_TP8 = "minimax_m3_sm107tp8_draft_prob1"
# world -> (op, max rows, max world) of the peer kernel instantiated by that op's translation unit
TOPOLOGIES = {
    2: (OP, MAX_ROWS, MAX_WORLD),
    4: (OP, MAX_ROWS, MAX_WORLD),
    8: (OP_TP8, MAX_ROWS_TP8, MAX_WORLD_TP8),
}
assert all(2 * r * w == SLOT_WORDS for _, r, w in TOPOLOGIES.values())
TAG = "[M31_DRAFT_PROB1]"


def _op(op: str, suffix: str = ""):
    """The family op ``<op><suffix>`` of a topology's op (OP or OP_TP8), or None when the library lacks it."""
    return getattr(family_ops(), (OP_TP8 if op == OP_TP8 else OP) + suffix, None)


def disabled() -> bool:
    return _DISABLED


def supported(
    world: int, k: int, vocab: int, w1: torch.Tensor, w2_shard: torch.Tensor
) -> Optional[str]:
    """None if this rank can set the draft up, else the reason (shapes / dtypes only: every rank decides alike)."""
    if _DISABLED:
        return f"off ({DISABLE_ENV} set)"
    if world not in TOPOLOGIES:
        return f"world {world} (the peer exchange is world 2 / 4 / 8)"
    op = TOPOLOGIES[world][0]
    if _op(op) is None:
        return f"{op} is not built"
    if not 1 <= k <= MAX_K:
        return f"K {k} > {MAX_K}"
    if not 1 <= vocab <= MAX_VOCAB:
        return f"vocab {vocab} >= 2^18"
    if w1.dtype != torch.bfloat16 or w1.dim() != 2 or tuple(w1.shape) != (vocab, MARKOV_RANK):
        return "w1 is not bf16 [V, 256]"
    if (
        w2_shard.dtype != torch.bfloat16
        or w2_shard.dim() != 2
        or w2_shard.shape[1] != MARKOV_RANK
        or not w2_shard.is_contiguous()
    ):
        return "w2 shard is not a contiguous bf16 [Vs, 256]"
    return None


def engages(spec_metadata, num_gens: int, k: int, block: int, max_rows: int = MAX_ROWS) -> bool:
    """Per step (Python, fixed into the captured graph; every rank sees the same metadata): rejection-sampling
    acceptance only (the stored q must be the proposal the verifier reads), 1..max_rows gen requests (8 at TP2 / TP4,
    4 at TP8), K == the worker's block, the slot-indexed draft_probs buffer present."""
    return (
        bool(getattr(spec_metadata, "use_rejection_sampling", False))
        and bool(getattr(spec_metadata, "wants_advanced_draft_sampling", False))
        and getattr(spec_metadata, "draft_probs", None) is not None
        and getattr(spec_metadata, "batch_slot_ids", None) is not None
        and getattr(spec_metadata, "request_seeds", None) is not None
        and getattr(spec_metadata, "request_offsets", None) is not None
        and 1 <= num_gens <= max_rows
        and k == block
    )


class _Mnnvl:
    """Collective MNNVL allocations over the TP group: every rank allocates in the same order; ``alloc`` returns this
    rank's view and every rank's unicast base pointer."""

    def __init__(self, mapping, comm):
        self.mapping, self.comm = mapping, comm
        self.keep: List[object] = []

    def alloc(self, numel: int, dtype: torch.dtype):
        from tensorrt_llm._torch.distributed.ops import (
            _make_mnnvl_mcast_buffer,
            _mnnvl_workspace_all_succeeded,
        )

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("m31 draft prob1: MNNVL allocation during graph capture")
        if not _mnnvl_workspace_all_succeeded(self.comm, True):
            raise RuntimeError("m31 draft prob1: MNNVL allocation skipped on another rank")
        esz = torch.empty((), dtype=dtype).element_size()
        handle = _make_mnnvl_mcast_buffer(
            self.comm, numel * esz, self.mapping, self.mapping.is_multi_node()
        )
        world, rank = self.mapping.tp_size, self.mapping.tp_rank
        ucs = [handle.get_uc_buffer(r, (numel,), dtype, 0) for r in range(world)]
        if not _mnnvl_workspace_all_succeeded(self.comm, True):
            raise RuntimeError("m31 draft prob1: MNNVL allocation failed on another rank")
        self.keep += [handle, ucs]
        return ucs[rank], [int(u.data_ptr()) for u in ucs]


class DraftProb1:
    """The workspace of one TP4 / TP8 DSpark worker and its per-step call."""

    def __init__(self) -> None:
        self.rank = 0
        self.world = 1
        self.op = OP
        self.max_rows = MAX_ROWS
        self.k = 0
        self.vocab = 0
        self.voff = 0
        self.w1: Optional[torch.Tensor] = None
        self.w2s: Optional[torch.Tensor] = None
        self.grids: Dict[int, int] = {}
        self.ws: List[int] = []
        self.peers: List[int] = []
        self.fault_peers: List[int] = []
        self.caches: List[int] = []
        self.buf: Dict[str, torch.Tensor] = {}
        self.taken: set = set()
        self.faults_logged = 0
        self._mn: Optional[_Mnnvl] = None
        self._probll = None  # TP8_PROBLL: the one-request draft-probability path (ops/draft_probs_one_request.py)

    @classmethod
    def create(
        cls, mapping, comm, w1: torch.Tensor, w2_shard: torch.Tensor, voff: int, vocab: int, k: int
    ):
        """Collective (every rank of the TP group, same order), outside capture."""
        self = cls()
        dev = w1.device
        self.rank, self.world = int(mapping.tp_rank), int(mapping.tp_size)
        self.op, self.max_rows, _ = TOPOLOGIES[self.world]
        rows_max = self.max_rows
        self.k, self.vocab, self.voff = int(k), int(vocab), int(voff)
        self.w1, self.w2s = w1, w2_shard
        mn = _Mnnvl(mapping, comm)
        # slots[0:64], fault/control[64:75], full epoch tags[128:192]; zero before collective ready
        xs, xs_bases = mn.alloc(EXCHANGE_STORAGE_WORDS, torch.int64)
        cache_flat, cache_bases = mn.alloc(rows_max * self.k * self.vocab, torch.bfloat16)
        xs.zero_()
        cache_flat.zero_()
        self._mn = mn
        z = lambda n, dt: torch.zeros(n, dtype=dt, device=dev)  # noqa: E731
        b = self.buf
        b["part"] = z(rows_max * MAX_GRID, torch.int64)
        b["part_ms"] = z(rows_max * MAX_GRID * 2, torch.float32)
        b["counter"] = z(1, torch.int32)
        b["epoch"] = z(1, torch.int64)
        b["err"] = z(1, torch.int32)
        b["seq"] = z(1, torch.int64)
        b["fault"] = z(1, torch.int32)
        b["fault_count"] = z(1, torch.int64)
        b["prev0"] = z(rows_max, torch.int64)
        b["prev1"] = z(rows_max, torch.int64)
        b["tok_out"] = z(rows_max * self.k, torch.int64).view(rows_max, self.k)
        b["idx_map"] = z(rows_max * self.k, torch.int32)
        b["sample_pos"] = z(rows_max * self.k, torch.int64)
        b["stats"] = z(rows_max * self.k * STAT_SPLIT * 2, torch.float32)
        b["tokens32"] = z(rows_max * self.k, torch.int32)
        b["xs"] = xs
        b["cache"] = cache_flat.view(rows_max, self.k, self.vocab)
        slots_ptr = int(xs.data_ptr())
        fault_ptr = slots_ptr + SLOT_WORDS * 8
        self.ws = [
            int(b["part"].data_ptr()),
            int(b["part_ms"].data_ptr()),
            int(b["counter"].data_ptr()),
            int(b["epoch"].data_ptr()),
            slots_ptr,
            int(b["err"].data_ptr()),
            fault_ptr,
        ]
        self.peers = list(xs_bases)
        self.fault_peers = [p + SLOT_WORDS * 8 for p in xs_bases]
        self.caches = list(cache_bases)
        if self.peers[self.rank] != slots_ptr or self.caches[self.rank] != int(
            b["cache"].data_ptr()
        ):
            raise RuntimeError("m31 draft prob1: MNNVL unicast base of this rank != its local view")
        max_grid = _op(self.op, "_max_grid")
        for c in range(1, rows_max + 1):
            self.grids[c] = int(max_grid(c))
            if self.grids[c] < 1:
                raise RuntimeError(f"m31 draft prob1: no co-resident grid at {c} rows")
        torch.cuda.synchronize()
        # A slow rank must finish zeroing before another rank can commit a decision.
        from tensorrt_llm._torch.distributed.ops import _mnnvl_workspace_all_succeeded

        if not _mnnvl_workspace_all_succeeded(comm, True):
            raise RuntimeError("m31 draft prob1: peer workspace initialization did not complete")
        # TP8_PROBLL: the one-request draft-probability path, set up here on every rank of the TP group in the same
        # order (collective); None unless TP world 8, not killed, and every rank built it and passed its set-up checks.
        # TP2 / TP4 (world != 8) return None at once, no collective.
        from ....ops import draft_probs_one_request as _draft_probll

        self._probll = _draft_probll.attach(self, mapping, comm, w1, w2_shard, voff, vocab, k)
        return self

    def describe(self) -> str:
        tp = "tp8" if self.op == OP_TP8 else "tp4"
        return (
            f"{TAG} engaged: {tp} rank {self.rank} world {self.world} rows<={self.max_rows} K {self.k} grid "
            f"{self.grids.get(1)}..{self.grids.get(self.max_rows)} | mirror mnnvl | adapter softmax->draft_probs | "
            f"hang cap {HANG_CAP_NS} ns"
        )

    def run(
        self,
        base2d: torch.Tensor,
        prev_in: torch.Tensor,
        slot_ids: torch.Tensor,
        dummy_slot_row: int,
        offsets: torch.Tensor,
        rng_base: int,
        span: int,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        draft_probs: torch.Tensor,
    ) -> torch.Tensor:
        """One draft of C = base2d.shape[0] // K requests; returns the int32 tokens [C, K] (a view of a persistent
        buffer) and writes draft_probs[slot_ids[r], :K, :V]."""
        b, k = self.buf, self.k
        rows = base2d.shape[0] // k
        if rows * k != base2d.shape[0] or not 1 <= rows <= self.max_rows:
            raise RuntimeError(f"m31 draft prob1: base rows {base2d.shape[0]} not C * K")
        if (
            self._probll is not None
        ):  # TP8_PROBLL: one request -> the one-request path; None = declined, the op below
            tokens = self._probll.run(
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
            )
            if tokens is not None:
                return tokens
        _op(self.op)(
            base2d,
            self.w1,
            self.w2s,
            self.voff,
            k,
            prev_in,
            b["prev0"],
            b["prev1"],
            b["tok_out"],
            self.ws,
            self.peers,
            self.caches,
            self.rank,
            self.world,
            self.grids[rows],
            True,
            slot_ids,
            int(dummy_slot_row),
            offsets,
            int(rng_base),
            int(span),
            b["idx_map"],
            b["sample_pos"],
            temperature,
            seeds,
            b["cache"],
            b["fault"],
            b["fault_count"],
            b["seq"],
            self.fault_peers,
            HANG_CAP_NS,
            b["stats"],
            draft_probs,
            b["tokens32"],
            self.vocab,
        )
        return b["tokens32"][: rows * k].view(rows, k)

    def taken_line(self, rows: int) -> Optional[str]:
        """The per-rank dispatch line, once per batch size."""
        if rows in self.taken:
            return None
        self.taken.add(rows)
        return f"[M3 dispatch] rank {self.rank}: m31_draft_prob1 taken at {rows} rows"

    def faults(self) -> int:
        """The agreed-fault count (synchronizes: call at a safe point, outside capture)."""
        if torch.cuda.is_current_stream_capturing():
            return self.faults_logged
        return int(self.buf["fault_count"].item())

    def fault_line(self) -> Optional[str]:
        """``[M31_DRAFT_PROB1] faults N`` when the count is non-zero and changed since the last line."""
        n = self.faults()
        if n == 0 or n == self.faults_logged:
            return None
        self.faults_logged = n
        return (
            f"{TAG} faults {n} (rank {self.rank}; sticky abort: token 0 and exact one-hot q; "
            "collectively rebuild workspace and captured graphs before reuse)"
        )

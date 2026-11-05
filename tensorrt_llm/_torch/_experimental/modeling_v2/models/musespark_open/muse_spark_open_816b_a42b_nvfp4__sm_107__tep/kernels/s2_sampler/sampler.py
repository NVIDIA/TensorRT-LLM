# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The target head's sampler fold on an interleaved vocabulary shard, exchanging states instead of logits.

Part of the ``distributed_head_sampler`` fast path. Without the interleaved copy, the
``split_sampler.target_head`` head all-gathers (with the fast path's one-shot logits gather) the
``[8, 202048]`` BF16 logits row and
folds it with the cooperative kernel A (``ops/spec/head_sampler``): virtual thread ``vt`` (0 .. 1023) folds the
vectors ``v = vt + 1024 j`` (4 logits each) into one online-softmax state. If rank ``r`` holds the vectors
``v = world * m + r`` instead of a contiguous slice -- local row ``4m + q`` = global row ``4(world m + r) + q``
(``interleaved_rows``) -- it holds every vector of the virtual threads ``vt = world * a + r`` whole, so it can fold
them itself (``s2_sampler.cu``: the served statements, the same elements in the same order, so the same states bit
for bit), and only the 12-byte states move: 12 KB per rank instead of the 3.2 MB row.

The pieces:

* ``interleaved_rows`` / ``unpermute_index`` / ``unpermute``: the permutation and its inverse on a gathered row of
  interleaved shards (one ``index_select``), for a step that needs the full row after the interleaved GEMM ran.
* ``build_interleaved_shard``: this rank's interleaved copy of the LM-head weight, built from the eight contiguous
  shards (collective; a copy of the same bytes). The LM-head module keeps its own weight: the drafter shares that
  module (``modeling_dflash``: ``draft_model_full.lm_head = target_model.lm_head``) and every other reader of the
  column-parallel logits (prefill, mixed and non-steady steps, the draft head) keeps the contiguous layout.
* ``S2States``: the fold + LL state publish, the every-rank state poll, and the every-rank draw (the served kernel B's
  tokens: the walk on the rank holding the walker's vectors, the result through an LL token slot; a filtered row's
  vectors through an LL row area). Collective construction: library load, the multicast areas, agreements.
* rejection sampling (the ``rejection_sampler`` fast path, ``rs_sampler``): ``S2States`` built with ``rs_areas``
  also reserves the candidate-slot and digest areas (``_layout``); ``fold_publish_rs`` + ``rs_draw``
  (``S2Head.sample_rejection``) run the verify rows' chain test and residual / plain race against the slot's q record.
* ``S2Head``: the target head's 8-row decode calls (``split_sampler.target_head.TargetHead`` hands them
  over once ``S2Head`` is armed): the LM-head GEMM on the copy with the dispatcher's own runner and tactic, the
  context-K/V head fork point where the stock call fires it, ``sample`` (fold + draw) for the fused sampler, and
  ``full_row`` (the one-shot gather + the inverse permutation) for every other consumer. Armed only when every rank's
  M = 8 LM-head choice is in ``BIT_EQUIVALENT_TACTICS``; otherwise the stock head runs.

Every token, state and logit is the served one bit for bit (value-identical by construction; the 1-GPU and 8-rank
tests check it). Nothing imports this module's kernels outside the ``distributed_head_sampler`` install
(``install``).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, Tuple

import torch

#: Rows of the served step: one request's K = 7 verify block plus its last accepted token.
ROWS = 8
#: Virtual threads of the served sampler's row (``head_sampler.VIRTUAL_THREADS``).
VIRTUAL_THREADS = 1024
#: 32-bit words of one state (max, sum, argmax), as ``head_sampler.STATE_WORDS``.
STATE_WORDS = 3
#: 64-bit LL elements of one state in the areas (max, sum, argmax, pad).
STATE_ELEMS = 4
#: Fold shape (virtual threads per CTA, threads per CTA) and loop form (0: the coop kernel's two passes as written;
#: 1: both passes in one loop; 2: form 1 with phase 1's loads issued together and phase 2's next slot read ahead);
#: every shape and form gives the same states (``s2_sampler.cu``). Default: form 2 with one phase-1 slot per thread,
#: the fastest in the 8-rank bench (``_s2_sampler_rank_job.py --bench``).
FOLD_SHAPE = (4, 256)
FOLD_FORM = 2
#: A poll waiting longer than this traps (a peer that never published).
POLL_TIMEOUT_S = 600
LIBRARY_NAME = "musespark_open_s2_sampler"
SOURCE = Path(__file__).resolve().parent / "s2_sampler.cu"
#: ``fusedSamplingKernelsCommon.cuh`` (``OnlineSoftmax``, ``loadRowParams``) lives with the served sampler.
INCLUDE_DIR = Path(__file__).resolve().parents[2] / "ops" / "spec" / "_head_sampler"
#: Rows with a draft (K = 7) and candidate lanes per row (``rs_common``).
RS_POS = 7
RS_LANES = 16
#: Words of one row of ``rs_draw``'s test export and their offsets (``s2_sampler.cu`` ``kExport*``; ints are stored
#: as their bits): scalars, then per candidate lane ``w``, ``qZ``, the race key, ``x`` and ``log E``.
RS_EXPORT_WORDS = 96
RS_EXPORT = {
    "m": 0,
    "z": 1,
    "koff": 2,
    "ioff": 3,
    "kd": 4,
    "acc": 5,
    "y": 6,
    "out": 7,
    "usable": 8,
    "reason": 9,
    "ua": 10,
    "kstar": 11,
    "w": 16,
    "qz": 32,
    "kc": 48,
    "x": 64,
    "loge": 80,
}
#: ``RS_EXPORT["reason"]`` values (``s2_sampler.cu`` ``RsReason``), in the order the draw tests them.
RS_REASONS = ("usable", "kind", "poison", "drafts", "qdiv", "qinvalid", "kd", "dup", "qzero")
#: Words of one row of ``rs_draw``'s filtered-row owner export and their offsets (``s2_sampler.cu`` ``kFExport*``;
#: ints stored as their bits): scalars, then per candidate lane ``w [w >= tau]``, ``fl(qf K)`` and the residual.
RS_FEXPORT_WORDS = 80
RS_FEXPORT = {
    "phi": 0,
    "raw_s": 1,
    "tau": 2,
    "kept": 3,
    "s1": 4,
    "n": 5,
    "nmax": 6,
    "r_off": 7,
    "r_c": 8,
    "target": 9,
    "ur": 10,
    "claim": 11,
    "out": 12,
    "flags": 13,
    "ua": 14,
    "kd": 15,
    "y": 16,
    "m": 17,
    "z": 18,
    "case": 19,
    "kept_count": 20,
    "wk": 32,
    "qk": 48,
    "rho": 64,
}
#: ``RS_FEXPORT["case"]`` values: the screen's floor (none, top-p only, min-p, top-k without min-p).
RS_SCREEN_CASES = ("none", "top_p", "min_p", "top_k")
#: Bits of a filtered row's flags word (``s2_sampler.cu`` ``kFlag*``).
RS_FLAG = {
    "acc": 1 << 0,
    "usable": 1 << 1,
    "reason_shift": 2,
    "d_unkept": 1 << 6,
    "screen_fallback": 1 << 7,
    "resid_fallback": 1 << 8,
    "plain": 1 << 9,
    "hash_shift": 10,
    "hash_mask": 0x3FFFF,
}
#: The served sampler library's flags (``head_sampler.NVCC_FLAGS``): no fast math, so ``tanhf`` and the fold's
#: arithmetic compile as there.
NVCC_FLAGS = ("-O3", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-lineinfo")

_LOCK = threading.Lock()
_LOADED: Optional[str] = None


def arch_target() -> str:
    capability = torch.cuda.get_device_capability()
    if capability == (10, 7):
        return "107a"
    raise RuntimeError(
        f"the s2 sampler is built for sm_107a, not sm_{capability[0]}{capability[1]}"
    )


def load_library() -> str:
    """Build (if needed) and load the library once per process; its path."""
    global _LOADED
    with _LOCK:
        if _LOADED is None:
            from torch.utils.cpp_extension import load

            from ..rs_sampler import rs_common as _rs_common

            target = arch_target()
            path = load(
                name=f"{LIBRARY_NAME}_sm{target}",
                sources=[str(SOURCE)],
                extra_cuda_cflags=list(NVCC_FLAGS)
                + [f"-gencode=arch=compute_{target},code=sm_{target}"],
                extra_include_paths=[str(INCLUDE_DIR), str(_rs_common.INCLUDE_DIR)],
                is_python_module=False,
                verbose=False,
            )
            _LOADED = str(path) if path else LIBRARY_NAME
        return _LOADED


# ---- the permutation ----------------------------------------------------------------------------------------------


def geometry_reason(world: int, shard: int) -> Optional[str]:
    """Why the interleaved fold cannot reproduce the served fold at this geometry, or None."""
    if world < 1 or VIRTUAL_THREADS % world:
        return f"{VIRTUAL_THREADS} virtual threads do not split over {world} ranks"
    if shard % 4:
        return f"a shard of {shard} is not a whole number of 4-element vectors"
    local_threads = VIRTUAL_THREADS // world
    if -(-(shard // 4) // local_threads) > 64:
        return "a virtual thread would fold more than the walk's 64 vectors"
    return None


def interleaved_rows(world: int, rank: int, shard: int, device=None) -> torch.Tensor:
    """Global vocabulary rows of rank ``rank``'s interleaved shard: local row ``4m + q`` is
    ``4(world m + rank) + q``."""
    j = torch.arange(shard, dtype=torch.int64, device=device)
    return 4 * (world * (j // 4) + rank) + (j % 4)


def unpermute_index(world: int, shard: int, device=None) -> torch.Tensor:
    """For every global row ``g``, its column in the rank-major gather of the interleaved shards."""
    g = torch.arange(world * shard, dtype=torch.int64, device=device)
    v = g // 4
    return (v % world) * shard + 4 * (v // world) + (g % 4)


def unpermute(gathered: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """The vocabulary-order row of a ``[rows, world * shard]`` gather of interleaved shards (one ``index_select``)."""
    return gathered.index_select(-1, index)


def build_interleaved_shard(
    weight: torch.Tensor, mapping, column_chunk: int = 512, group=None
) -> torch.Tensor:
    """This rank's interleaved copy of a column-parallel weight ``[shard, hidden]`` (rank ``p`` holds rows
    ``p * shard ..``). Collective over the TP group: ``torch.distributed`` all-gathers of column chunks over the
    WORLD group (``target_head._ps_torch_dist`` creates it when none exists), then this rank's rows. A copy of the
    same bytes. Peak extra memory ``world * shard * column_chunk`` elements."""
    import torch.distributed as dist

    from ..split_sampler import target_head as _th

    if group is None:
        _th._torch_dist_world(mapping)
    world = int(mapping.tp_size)
    rank = int(mapping.tp_rank)
    shard, hidden = int(weight.shape[0]), int(weight.shape[1])
    rows = interleaved_rows(world, rank, shard, weight.device)
    out = torch.empty((shard, hidden), dtype=weight.dtype, device=weight.device)
    full = torch.empty(
        (world * shard, min(column_chunk, hidden)), dtype=weight.dtype, device=weight.device
    )
    for c0 in range(0, hidden, column_chunk):
        c1 = min(hidden, c0 + column_chunk)
        part = weight[:, c0:c1].contiguous()
        dest = (
            full[:, : c1 - c0]
            if c1 - c0 == full.shape[1]
            else torch.empty((world * shard, c1 - c0), dtype=weight.dtype, device=weight.device)
        )
        dist.all_gather_into_tensor(dest, part, group=group)
        out[:, c0:c1] = dest.index_select(0, rows)
        del part, dest
    torch.cuda.synchronize(weight.device)
    del full, rows
    return out


# ---- the states ---------------------------------------------------------------------------------------------------


def area_bytes(rows: int) -> int:
    """Bytes of one state area: every rank's states of ``rows`` rows (two areas alternate by epoch parity)."""
    return rows * VIRTUAL_THREADS * STATE_ELEMS * 8


def slot_bytes(rows: int, world: int) -> int:
    """Bytes of one token-slot area: one entry (two 64-bit LL elements) per (row, rank)."""
    return rows * world * 16


def row_area_bytes(rows: int, world: int, shard: int) -> int:
    """Bytes of one row area: every rank's local vectors of ``rows`` rows, one pair of 64-bit LL elements per
    vector (a filtered row's logits travel here)."""
    return rows * world * (shard // 4) * 16


def rs_cand_bytes(world: int) -> int:
    """Bytes of one candidate-slot area: one entry (two 64-bit LL elements) per (row t < 7, lane, rank)."""
    return RS_POS * RS_LANES * world * 16


def rs_dig_bytes(world: int) -> int:
    """Bytes of one digest area: one entry (two 64-bit LL elements) per (row t < 7, rank)."""
    return RS_POS * world * 16


def rs_list_cap(vocab: int) -> int:
    """Entries of one filtered row's survivor list (``s2_sampler.cu`` ``rsListCap``): every element of the row,
    ``4 * ceil(vocab / 4 / 1024)`` per thread of the draw CTA."""
    return 4 * -(-(vocab // 4) // VIRTUAL_THREADS) * VIRTUAL_THREADS


def kernel_attributes(device) -> dict:
    """Resolve the RS kernels (and the non-RS fold / draw) and return ``{name: (registers, local bytes, static shared
    bytes, max threads per block)}``, so nothing loads during a later CUDA-graph capture."""
    load_library()
    anchor = torch.empty(1, dtype=torch.int32, device=device)
    rows = torch.ops.musespark_open_s2_sampler.rs_attributes(anchor).tolist()
    names = ("rs_fold", "rs_draw", "rs_draw_trigger", "fold", "draw")
    return {n: tuple(int(v) for v in r) for n, r in zip(names, rows)}


def new_states(rows: int, device) -> torch.Tensor:
    """Scratch for ``rows`` rows of states at their virtual thread (``head_sampler.new_states``'s layout)."""
    return torch.empty(rows * VIRTUAL_THREADS * STATE_WORDS, dtype=torch.int32, device=device)


class S2States:
    """This rank's fold + state publish and its poll of every rank's states."""

    def __init__(
        self,
        mapping,
        shard: int,
        device: torch.device,
        rows: int = ROWS,
        *,
        fold_shape: tuple = FOLD_SHAPE,
        form: int = FOLD_FORM,
        emulated_world: Optional[int] = None,
        rs_areas: bool = False,
    ) -> None:
        """Collective over the TP group of ``mapping``.

        ``emulated_world`` (tests only, no ``mapping``): one process stands in for that many ranks on a plain device
        buffer that is both the multicast and the receive area; ``fold_publish(..., rank=r)`` publishes rank ``r``'s
        states with that rank's own epoch counters. ``rs_areas``: reserve the rejection-sampling areas; every rank must
        decide alike.
        """
        self.rs_areas = bool(rs_areas)
        self.shard = int(shard)
        self.rows = int(rows)
        self.device = device
        self.vts, self.threads = (int(fold_shape[0]), int(fold_shape[1]))
        self.form = int(form)
        self.timeout_ns = int(POLL_TIMEOUT_S * 1e9)
        self.area = area_bytes(self.rows)
        self.calls = 0
        if emulated_world is not None:
            self.world = int(emulated_world)
            self.rank = 0
            reason = geometry_reason(self.world, self.shard)
            if reason is not None:
                raise ValueError(f"[musespark_open] s2 sampler: {reason}")
            self._layout()
            load_library()
            self._area = torch.zeros(self.total // 4, dtype=torch.int32, device=device)
            self._mc_ptr = self._uc_ptr = int(self._area.data_ptr())
            self.fold_counters = [self._new_fold_counters() for _ in range(self.world)]
            self.poll_counters = torch.zeros(self.rows, dtype=torch.int32, device=device)
            self.draw_counters = [
                torch.zeros(self.rows, dtype=torch.int32, device=device) for _ in range(self.world)
            ]
            self._new_rs_lists(self.world)
            torch.ops.musespark_open_s2_sampler.dry(self.poll_counters)
            torch.cuda.synchronize(device)
            return
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
            _mnnvl_workspace_all_succeeded,
        )

        self.world = int(mapping.tp_size)
        self.rank = int(mapping.tp_rank)
        self._comm = _get_mnnvl_workspace_comm(mapping)
        error: Optional[BaseException] = None
        reason = geometry_reason(self.world, self.shard)
        try:
            if reason is not None:
                raise ValueError(reason)
            load_library()
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                "[musespark_open] s2 sampler: the library failed to load (or the geometry is refused) on at least "
                f"one rank{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error
        self._layout()
        size = self.total
        try:
            self._handle = _make_mnnvl_mcast_buffer(
                self._comm, size, mapping, mapping.is_multi_node()
            )
            self._uc = self._handle.get_uc_buffer(self.rank, (size // 4,), torch.int32, 0)
            self._mc = self._handle.get_mc_buffer((size // 4,), torch.int32, 0)
            self._mc_ptr = int(self._mc.data_ptr())
            self._uc_ptr = int(self._uc.data_ptr())
            with torch.cuda.device(device):
                self._uc.zero_()
                self.fold_counters = [self._new_fold_counters()]
                # One counter per row, shared by the state poll and the draw (a call runs one of them).
                self.poll_counters = torch.zeros(self.rows, dtype=torch.int32, device=device)
                self.draw_counters = None
                self._new_rs_lists(1)
                torch.ops.musespark_open_s2_sampler.dry(self.poll_counters)
                torch.cuda.synchronize(device)
        except Exception as exc:  # noqa: BLE001
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                "[musespark_open] s2 sampler: the state areas failed on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error

    def _layout(self) -> None:
        """Byte offsets in the fabric buffer: two state areas, two token-slot areas, two row areas; with ``rs_areas``
        then two candidate-slot areas and two digest areas (the offsets before them unchanged; 0 when not reserved)."""
        self.slot_area = slot_bytes(self.rows, self.world)
        self.row_area = row_area_bytes(self.rows, self.world, self.shard)
        self.slot_offset = 2 * self.area
        self.rows_offset = self.slot_offset + 2 * self.slot_area
        self.total = self.rows_offset + 2 * self.row_area
        self.rs_cand_offset = self.rs_cand_bytes = self.rs_dig_offset = self.rs_dig_bytes = 0
        if self.rs_areas:
            self.rs_cand_bytes = rs_cand_bytes(self.world)
            self.rs_dig_bytes = rs_dig_bytes(self.world)
            self.rs_cand_offset = self.total
            self.rs_dig_offset = self.rs_cand_offset + 2 * self.rs_cand_bytes
            self.total = self.rs_dig_offset + 2 * self.rs_dig_bytes

    def _new_rs_lists(self, copies: int) -> None:
        """With the RS areas: the filtered rows' survivor lists of ``rs_draw`` (``s2_sampler.cu`` ``rsFilteredRow``),
        FP32 values and int32 ids ``[ceil(rows / world), rs_list_cap(vocab)]``, one pair per emulated rank."""
        self.rs_list_x = self.rs_list_id = None
        if not self.rs_areas:
            return
        shape = (-(-self.rows // self.world), rs_list_cap(self.world * self.shard))
        self.rs_list_x = [
            torch.empty(shape, dtype=torch.float32, device=self.device) for _ in range(copies)
        ]
        self.rs_list_id = [
            torch.empty(shape, dtype=torch.int32, device=self.device) for _ in range(copies)
        ]

    def _new_fold_counters(self) -> torch.Tensor:
        ctas = self.rows * (VIRTUAL_THREADS // self.world) // self.vts
        return torch.zeros(ctas, dtype=torch.int32, device=self.device)

    def fold_publish(
        self,
        local: torch.Tensor,
        temperatures: torch.Tensor,
        multiplier: float,
        soft_cap: float,
        *,
        rank: Optional[int] = None,
        own_states: Optional[torch.Tensor] = None,
        stamps: Optional[torch.Tensor] = None,
        slot: int = 0,
        pdl: bool = True,
        top_ks: Optional[torch.Tensor] = None,
        top_ps: Optional[torch.Tensor] = None,
        min_ps: Optional[torch.Tensor] = None,
    ) -> None:
        """Fold this rank's ``[rows, shard]`` interleaved BF16 logits and publish the states to every rank.

        ``rank`` (emulated instances only) selects the emulated rank; ``own_states`` (tests) also receives this rank's
        states at their virtual thread; ``stamps`` (bench) ``[slots, 4]`` int64 globaltimer stamps. With ``top_ks`` /
        ``top_ps`` / ``min_ps`` (the draw's row params; all three or none) a row they filter also publishes its
        local vectors into the row area, which ``draw`` reads for that row.
        """
        from ...ops.spec import head_sampler as _hs

        if rank is None:
            rank, counters = self.rank, self.fold_counters[0]
        else:
            counters = self.fold_counters[int(rank)]
        m, inv, cap = _hs.cap_constants(multiplier, soft_cap)
        rows = top_ks is not None
        torch.ops.musespark_open_s2_sampler.fold_publish(
            local,
            temperatures,
            counters,
            self._mc_ptr,
            self.area,
            int(rank),
            self.world,
            m,
            inv,
            cap,
            self.vts,
            self.threads,
            self.form,
            pdl,
            own_states,
            stamps,
            int(slot),
            top_ks,
            top_ps,
            min_ps,
            self._mc_ptr + self.rows_offset if rows else 0,
            self.row_area if rows else 0,
        )

    def draw(
        self,
        local: torch.Tensor,
        scratch: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        min_ps: torch.Tensor,
        seed: Optional[torch.Tensor],
        offset: Optional[torch.Tensor],
        multiplier: float,
        soft_cap: float,
        *,
        rank: Optional[int] = None,
        stamps: Optional[torch.Tensor] = None,
        slot: int = 0,
        pdl: bool = True,
        trigger: bool = False,
        force_multi: bool = False,
    ) -> torch.Tensor:
        """The served kernel B's tokens of this call (int32 ``[rows]``), drawn on every rank from the polled states
        (``s2_sampler.cu`` ``s2DrawKernel``); follows ``fold_publish`` of the same call, which must have been given the
        row params. ``scratch`` (``[rows, world * shard]`` BF16) receives a filtered row in vocabulary order.

        ``rank`` (emulated instances only) selects the emulated rank, which then uses its own counters; ``trigger``
        releases the dependent launch at entry (``TargetHead.verify_pdl``); ``force_multi`` (tests) sends a
        one-claimant row through the several-claimants path.
        """
        from ...ops.spec import head_sampler as _hs

        if rank is None:
            rank, counters = self.rank, self.poll_counters
        else:
            counters = self.draw_counters[int(rank)]
        m, inv, cap = _hs.cap_constants(multiplier, soft_cap)
        tokens = torch.ops.musespark_open_s2_sampler.draw(
            local,
            scratch,
            temperatures,
            top_ks,
            top_ps,
            min_ps,
            seed,
            offset,
            counters,
            self._uc_ptr,
            self.area,
            self._uc_ptr + self.slot_offset,
            self._mc_ptr + self.slot_offset,
            self.slot_area,
            self._uc_ptr + self.rows_offset,
            self.row_area,
            int(rank),
            self.world,
            m,
            inv,
            cap,
            self.timeout_ns,
            pdl,
            trigger,
            stamps,
            int(slot),
            int(bool(force_multi)),
        )
        self.calls += 1
        return tokens

    def _require_rs(self) -> None:
        if not self.rs_areas:
            raise RuntimeError(
                "[musespark_open] s2 sampler: the RS areas are not reserved (S2States built without rs_areas)"
            )

    def fold_publish_rs(
        self,
        local: torch.Tensor,
        temperatures: torch.Tensor,
        multiplier: float,
        soft_cap: float,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        min_ps: torch.Tensor,
        seed: torch.Tensor,
        offset: torch.Tensor,
        rs,
        *,
        rank: Optional[int] = None,
        own_states: Optional[torch.Tensor] = None,
        stamps: Optional[torch.Tensor] = None,
        slot: int = 0,
        pdl: bool = True,
        export_e: Optional[torch.Tensor] = None,
        no_exclude: bool = False,
    ) -> None:
        """The rejection-sampling fold of this call (``s2_sampler.cu`` ``s2FoldPublishKernel<4, 256, 2, true>``):
        ``fold_publish`` with the row params (mandatory) + the race words, the candidate slots and the digests.

        ``rs`` carries ``records``, ``num_slots``, ``slot_ids`` (request 0's slot first), ``draft_in`` (int32 /
        int64 ``[1, 7]``, read in place) and ``vocab``; ``seed`` / ``offset`` are the per-row int64 RNG state.
        ``rank`` (emulated instances) selects the emulated rank. Tests only: ``export_e`` (FP32 ``[2, rows, vocab]``)
        receives every folded element's ``log E`` ([0]) and ``x`` ([1]); ``no_exclude`` keeps the candidates in the
        race.
        """
        from ...ops.spec import head_sampler as _hs

        self._require_rs()
        if rank is None:
            rank, counters = self.rank, self.fold_counters[0]
        else:
            counters = self.fold_counters[int(rank)]
        m, inv, cap = _hs.cap_constants(multiplier, soft_cap)
        torch.ops.musespark_open_s2_sampler.fold_publish_rs(
            local,
            temperatures,
            counters,
            self._mc_ptr,
            self.area,
            int(rank),
            self.world,
            m,
            inv,
            cap,
            self.vts,
            self.threads,
            self.form,
            pdl,
            own_states,
            stamps,
            int(slot),
            top_ks,
            top_ps,
            min_ps,
            self._mc_ptr + self.rows_offset,
            self.row_area,
            seed,
            offset,
            rs.records,
            int(rs.num_slots),
            rs.slot_ids,
            rs.draft_in,
            self._mc_ptr + self.rs_cand_offset,
            self.rs_cand_bytes,
            self._mc_ptr + self.rs_dig_offset,
            self.rs_dig_bytes,
            int(rs.vocab),
            export_e,
            bool(no_exclude),
        )

    def rs_draw(
        self,
        local: torch.Tensor,
        scratch: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        min_ps: torch.Tensor,
        seed: torch.Tensor,
        offset: torch.Tensor,
        multiplier: float,
        soft_cap: float,
        rs,
        *,
        rank: Optional[int] = None,
        stamps: Optional[torch.Tensor] = None,
        slot: int = 0,
        pdl: bool = True,
        trigger: bool = False,
        force_ua: Optional[torch.Tensor] = None,
        force_loge: Optional[torch.Tensor] = None,
        exports: Optional[torch.Tensor] = None,
        force_screen: int = 0,
        fexports: Optional[torch.Tensor] = None,
        fstamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The rejection-sampling draw of this call on every rank (``s2_sampler.cu`` ``s2RsDrawKernel``): int32
        ``[rows]``, ``out_t = accepted ? d_t : y_t`` for rows t < 7 and the plain race for row 7; follows
        ``fold_publish_rs`` of the same call. Clears the record's valid bit of each row t < 7 at its end. A row with
        top-k / top-p / min-p is decided by exact filtered rejection sampling on the rank ``row % world`` (or by the
        strict body under ``rs.filter_mode`` strict).

        ``rs`` carries ``records``, ``num_slots``, ``dummy_slot``, ``slot_ids``, ``draft_in``, ``vocab``, ``check``,
        ``diag`` (int64 ``[rs_common.DIAG_WORDS]``) and optionally ``ring`` / ``ring_head`` and ``filter_mode``
        (``rs_common.FilterMode``, default exact). ``local`` holds one request's K + 1 = 8 rows. Tests only:
        ``force_ua`` (FP32 ``[rows]``), ``force_loge`` (FP32 ``[rows, 16]``), ``exports`` (FP32
        ``[rows, RS_EXPORT_WORDS]``, layout ``RS_EXPORT``), ``force_screen`` (1: no raw-logit screen, 2: no screen),
        ``fexports`` (FP32 ``[rows, RS_FEXPORT_WORDS]``, layout ``RS_FEXPORT``, written by each filtered row's owner);
        bench: ``fstamps`` (int64 ``[slots, 4]``, the filtered rows' stamps).
        """
        from ...ops.spec import head_sampler as _hs

        self._require_rs()
        if rank is None:
            rank, counters = self.rank, self.poll_counters
            lists = 0
        else:
            counters = self.draw_counters[int(rank)]
            lists = int(rank)
        m, inv, cap = _hs.cap_constants(multiplier, soft_cap)
        tokens = torch.ops.musespark_open_s2_sampler.rs_draw(
            local,
            scratch,
            temperatures,
            top_ks,
            top_ps,
            min_ps,
            seed,
            offset,
            counters,
            self._uc_ptr,
            self.area,
            self._uc_ptr + self.slot_offset,
            self._mc_ptr + self.slot_offset,
            self.slot_area,
            self._uc_ptr + self.rows_offset,
            self.row_area,
            int(rank),
            self.world,
            m,
            inv,
            cap,
            self.timeout_ns,
            pdl,
            trigger,
            stamps,
            int(slot),
            rs.records,
            int(rs.num_slots),
            int(rs.dummy_slot),
            rs.slot_ids,
            rs.draft_in,
            self._uc_ptr + self.rs_cand_offset,
            self.rs_cand_bytes,
            self._uc_ptr + self.rs_dig_offset,
            self.rs_dig_bytes,
            int(rs.vocab),
            int(rs.check),
            rs.diag,
            getattr(rs, "ring", None),
            getattr(rs, "ring_head", None),
            force_ua,
            force_loge,
            exports,
            self.rs_list_x[lists],
            self.rs_list_id[lists],
            int(getattr(rs, "filter_mode", 0)),
            int(force_screen),
            fexports,
            fstamps,
        )
        self.calls += 1
        return tokens

    def poll(
        self,
        states: torch.Tensor,
        *,
        stamps: Optional[torch.Tensor] = None,
        slot: int = 0,
        pdl: bool = True,
    ) -> torch.Tensor:
        """Every rank's states of this call into ``states`` (``new_states(rows)``), in the served kernel B's layout."""
        torch.ops.musespark_open_s2_sampler.poll_states(
            states,
            self.poll_counters,
            self._uc_ptr,
            self.area,
            self.rows,
            self.world,
            self.timeout_ns,
            pdl,
            stamps,
            int(slot),
        )
        self.calls += 1
        return states


# ---- the head -------------------------------------------------------------------------------------------------------

#: The LM head's M = 8 split-K tactics ``(mma_m, mma_n, split_k, ab_stages)`` shown to give every logit identical bits
#: whatever the row's position in the shard (a 1-GPU test and an 8-rank end-to-end run). The interleaved head computes
#: logit ``g`` on rank ``(g / 4) % world`` instead of ``g / shard``, so it is value-identical only when every rank's
#: choice is in this set.
BIT_EQUIVALENT_TACTICS = frozenset({(128, 16, 1, 5), (128, 16, 1, 6), (128, 32, 1, 5)})


def _gemm_choice(lm_head, weight: torch.Tensor, x: torch.Tensor):
    """``(dispatcher, runner, tactic, weight_t)``: the low-M dispatcher's runner and tactic for ``x @ weight.T`` --
    ``split_sampler.target_head.lm_head_gemm_into``'s statements (``LowMGemmDispatcher.apply``'s query and its
    fallback of an incompatible cached direct tactic) with ``weight`` given."""
    from tensorrt_llm._torch.autotuner import AutoTuner
    from tensorrt_llm._torch.modules import low_m_gemm as _low_m

    dispatcher = getattr(lm_head, "_low_m_gemm_dispatcher", _low_m._DISPATCHER)
    if not dispatcher._prepared:
        dispatcher._force = False
        dispatcher._init_runners()
    if not (
        _low_m._should_apply_low_m_gemm(x)
        and not getattr(lm_head, "use_custom_cublas_mm", False)
        and dispatcher._is_candidate_shape(x, weight, None)
    ):
        raise RuntimeError(
            "[musespark_open] s2 sampler: the LM-head call does not take the low-M dispatcher on this rank"
        )
    x2d = x.view(-1, x.shape[-1])
    weight_t = weight.detach().t()
    cublas = dispatcher._cublas_no_bias
    runners = [cublas, dispatcher._runner_no_bias, dispatcher._direct]
    runner, tactic = AutoTuner.get().choose_one(
        _low_m._AUTOTUNER_OP_NAME, runners, dispatcher._tuning_config, [x2d, weight_t], bias=None
    )
    if runner is dispatcher._direct and not dispatcher._direct.is_tactic_compatible(
        [x2d, weight_t], tactic
    ):
        runner, tactic = cublas, -1
    return dispatcher, runner, tactic, weight_t


def _choice_state(lm_head, x: torch.Tensor) -> str:
    """This rank's verdict on the stock LM-head choice for ``x``: ``ok``, ``wait`` (autotuner tuning, or no cached
    entry yet) or a refusal reason."""
    from tensorrt_llm._torch.autotuner import AutoTuner
    from tensorrt_llm._torch.modules import low_m_gemm as _low_m

    tuner = AutoTuner.get()
    if tuner.is_tuning_mode:
        return "wait"
    dispatcher, runner, tactic, weight_t = _gemm_choice(lm_head, lm_head.weight, x)
    runners = [dispatcher._cublas_no_bias, dispatcher._runner_no_bias, dispatcher._direct]
    hit = tuner.profiling_cache.search_cache(
        _low_m._AUTOTUNER_OP_NAME,
        runners,
        tuple(tuner._get_input_sizes([x.view(-1, x.shape[-1]), weight_t])),
        dispatcher._tuning_config,
        apply_map_to_tuning_buckets=True,
    )[0]
    if not hit:
        return "wait"
    if runner is not dispatcher._runner_no_bias or tuple(tactic) not in BIT_EQUIVALENT_TACTICS:
        return f"M = {x.shape[0]} LM-head choice {type(runner).__name__} {tactic} is outside the bit-equivalent set"
    return "ok"


class S2Head:
    """The interleaved-shard path of the target head on this rank: the interleaved copy of the LM-head weight, its
    GEMM output, the state / token exchange (``S2States``) and the scratch row the step's pending ``logits`` object is.

    ``TargetHead`` (``split_sampler.target_head``) hands it the 8-row decode calls once ``takes`` says so: ``produce``
    runs the LM-head GEMM on the copy (the dispatcher's own runner and tactic, as the stock call) and fires
    ``context_kv_fork``'s head fork point where the stock call fires it; ``sample`` replaces the one-shot gather + the
    split sampler (fold + publish, then the every-rank draw); ``full_row`` gives the step's ``[8, vocab]`` row in
    vocabulary order for every other consumer (the one-shot gather of the interleaved shards + the inverse
    permutation).
    """

    def __init__(
        self, mapping, lm_head, weight: torch.Tensor, states: S2States, build_seconds: float
    ) -> None:
        self.mapping = mapping
        self.weight = weight
        self.states = states
        self.build_seconds = build_seconds
        self.shard = int(weight.shape[0])
        self.world = int(mapping.tp_size)
        device = weight.device
        self.local = torch.empty(ROWS, self.shard, dtype=torch.bfloat16, device=device)
        self.scratch = torch.empty(
            ROWS, self.world * self.shard, dtype=torch.bfloat16, device=device
        )
        self.index = unpermute_index(self.world, self.shard, device)
        #: None until every rank agreed on the stock M = 8 choice (``_decide``); then whether the interleaved path runs.
        self.armed: Optional[bool] = None
        self._tactic = None
        self.calls = 0

    def _decide(self, lm_head, x: torch.Tensor) -> bool:
        """Whether this 8-row call takes the interleaved path; decided once, collectively, when every rank's autotuner
        has a cached choice (until then the call runs the stock path). Every rank reaches this at the same call."""
        from tensorrt_llm.logger import logger

        from ... import dispatch as _dispatch

        if self.armed is not None:
            return self.armed
        try:
            local = _choice_state(lm_head, x)
        except RuntimeError as exc:
            local = str(exc)
        if self.world > 1:
            everyone = _dispatch.Distributed.get(self.mapping).tp_allgather(
                local, small_payload=True
            )
        else:
            everyone = [local]
        if any(v == "wait" for v in everyone):
            return False
        refused = [(r, v) for r, v in enumerate(everyone) if v != "ok"]
        self.armed = not refused
        if self.armed:
            _, runner, tactic, _ = _gemm_choice(lm_head, self.weight, x)
            self._tactic = (runner, tuple(tactic))
            logger.info(
                f"[musespark_open] s2 sampler armed: LM-head tactic {tuple(tactic)} on this rank; 8-row decode steps "
                "run the interleaved GEMM + fold + every-rank draw"
            )
        else:
            logger.warning(
                f"[musespark_open] s2 sampler not armed (the stock head runs): {refused} -- "
                "the s2 sampler is value-identical only when every rank's M = 8 LM-head choice is in "
                f"{sorted(BIT_EQUIVALENT_TACTICS)}"
            )
        return self.armed

    def takes(self, hidden_states: torch.Tensor, lm_head) -> bool:
        return (
            hidden_states.dim() == 2
            and hidden_states.shape[0] == ROWS
            and int(lm_head.weight.shape[0]) == self.shard
            and self._decide(lm_head, hidden_states)
        )

    def produce(self, lm_head, hidden_states: torch.Tensor) -> torch.Tensor:
        """The LM-head GEMM of this call on the interleaved copy into ``local``; the pending ``logits`` object."""
        from ..split_sampler import target_head as _th

        admitted, fork = _th.context_kv_head_fork(lm_head)
        if not admitted:
            raise RuntimeError(
                "[musespark_open] s2 sampler: the LM head's apply_linear was replaced on the instance by something "
                "other than context_kv_fork's head fork point; the s2 head runs the LM-head GEMM itself and would "
                "bypass it"
            )
        _, runner, tactic, weight_t = _gemm_choice(lm_head, self.weight, hidden_states)
        if (
            runner is not self._tactic[0]
            or isinstance(tactic, int)
            or tuple(tactic) != self._tactic[1]
        ):
            raise RuntimeError(
                f"[musespark_open] s2 sampler: the LM-head choice changed after arming ({self._tactic} -> "
                f"{(runner, tactic)})"
            )
        runner(
            [hidden_states.view(-1, hidden_states.shape[-1]), weight_t],
            tactic=tactic,
            bias=None,
            out=self.local,
        )
        if fork is not None:
            # What context_kv_fork's apply_linear does right after the GEMM: fork its side chain.
            fork.fire("head")
        self.calls += 1
        return self.scratch

    def sample(
        self,
        temperatures,
        top_ks,
        top_ps,
        min_ps,
        seed,
        offset,
        multiplier: float,
        soft_cap: float,
        pdl: bool,
        verify_pdl: bool = False,
    ) -> torch.Tensor:
        """The served split sampler's tokens of this call: the fold + state publish, then the every-rank draw.

        ``verify_pdl`` (``TargetHead.verify_pdl``): the draw releases its dependent at entry.
        """
        self.states.fold_publish(
            self.local,
            temperatures,
            multiplier,
            soft_cap,
            pdl=pdl,
            top_ks=top_ks,
            top_ps=top_ps,
            min_ps=min_ps,
        )
        return self.states.draw(
            self.local,
            self.scratch,
            temperatures,
            top_ks,
            top_ps,
            min_ps,
            seed,
            offset,
            multiplier,
            soft_cap,
            pdl=pdl,
            trigger=verify_pdl and pdl,
        )

    def sample_rejection(
        self,
        temperatures,
        top_ks,
        top_ps,
        min_ps,
        seed,
        offset,
        multiplier: float,
        soft_cap: float,
        pdl: bool,
        rs,
        verify_pdl: bool = False,
    ) -> torch.Tensor:
        """The rejection-sampling verify tokens of this call (int32 ``[8]``): ``out_t = accepted ? d_t : y_t`` for the
        draft rows, the plain race for the bonus row (``S2States.fold_publish_rs`` then ``rs_draw``); ``rs`` is the
        step's ``RsVerifyArgs``. ``verify_pdl`` (``TargetHead.verify_pdl``): the draw releases its dependent at entry.
        """
        self.states.fold_publish_rs(
            self.local,
            temperatures,
            multiplier,
            soft_cap,
            top_ks,
            top_ps,
            min_ps,
            seed,
            offset,
            rs,
            pdl=pdl,
        )
        return self.states.rs_draw(
            self.local,
            self.scratch,
            temperatures,
            top_ks,
            top_ps,
            min_ps,
            seed,
            offset,
            multiplier,
            soft_cap,
            rs,
            pdl=pdl,
            trigger=verify_pdl and pdl,
        )

    def full_row(self, gather) -> torch.Tensor:
        """This call's ``[8, vocab]`` BF16 row in vocabulary order, in ``scratch``: ``gather`` (the head's one-shot
        ``PvLogitsGather``, collective) of the interleaved shards, then the inverse permutation."""
        gathered = gather.gather(self.local)
        torch.index_select(gathered, 1, self.index, out=self.scratch)
        return self.scratch


def unsupported_reason(llm, mapping) -> Optional[str]:
    """Why the s2 pieces cannot be built for this model, or None. Rank-local facts (the caller agrees them)."""
    from tensorrt_llm._torch.modules import low_m_gemm as _low_m
    from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
    from tensorrt_llm._utils import mpi_world_size

    from ..split_sampler import target_head as _th

    processor = getattr(llm, "logits_processor", None)
    if not isinstance(processor, _th.SplitSamplerLogitsProcessor):
        return "the split_sampler head is not installed"
    head = processor._split_head
    if head.workspace is not None:
        return (
            "lm_head_inbox is installed (the distributed head sampler replaces the split_sampler head's 8-row path, "
            "not the inbox's)"
        )
    if head.pv_gather is None:
        return (
            "the one-shot logits gather is not installed (distributed_head_sampler; the distributed head sampler's "
            "full row uses it)"
        )
    lm_head = getattr(llm, "lm_head", None)
    if not isinstance(lm_head, Linear) or lm_head.tp_mode != TensorParallelMode.COLUMN:
        return "the LM head is not a column-parallel Linear"
    if (
        lm_head.bias is not None
        or lm_head.weight.dtype != torch.bfloat16
        or lm_head.weight.dim() != 2
    ):
        return "the LM head is not an unbiased 2-D BF16 projection"
    if not lm_head.weight.is_contiguous():
        return "the LM-head weight is not contiguous"
    if getattr(lm_head, "lora", None) is not None:
        return "the LM head has LoRA"
    if not _low_m.LOW_M_GEMM_ACTIVE or getattr(lm_head, "use_custom_cublas_mm", False):
        return "the LM head does not take the low-M dispatcher (TRTLLM_LOW_M_GEMM_BACKEND)"
    if type(lm_head).apply_linear is not Linear.apply_linear:
        return "the LM head's class replaces Linear.apply_linear (the s2 head runs the dispatcher's GEMM itself)"
    if mpi_world_size() != mapping.tp_size or list(mapping.tp_group) != list(
        range(mapping.tp_size)
    ):
        return "the TP group is not the whole MPI world (the copy is built over the WORLD group)"
    return geometry_reason(int(mapping.tp_size), int(lm_head.weight.shape[0]))


def install(
    llm, mapping, rs_facts: Optional[Tuple[bool, Optional[str]]] = None
) -> Optional[S2Head]:
    """``distributed_head_sampler``: build this rank's interleaved LM-head copy and the s2 sampler's fabric areas and
    attach the ``S2Head`` to the split_sampler head as ``_split_head.s2``, which hands it the 8-row decode calls once
    it is armed (``S2Head._decide``). Collective over the TP group (the agreed refusal, the fabric areas, the copy's
    all-gathers). The LM-head module and every other path that calls it (prefill, mixed and other-size steps, the
    drafter's head) are unchanged.

    Call after ``split_sampler.target_head.install``. Idempotent like the other drafter installs (the shell runs
    them from ``load_draft_weights`` and again from ``post_load_weights``): once a head is attached, a repeated call
    returns it and builds nothing. Every rank takes the same branch, since an
    install that fails raises on every rank alike.

    ``rs_facts``: None when the composition has no ``rejection_sampler`` path (no rejection-sampling areas, and the
    refusal agrees nothing else); otherwise this rank's ``rs_sampler.rs.agreement_facts`` ``(wanted, forced mode)``,
    which the refusal's collective also agrees: ranks that differ refuse before the fabric is sized, and the RS install
    that follows then branches alike on every rank. The RS areas are reserved when rejection sampling is wanted and
    not inert under forced acceptance.

    Raises:
        ValueError: if some rank cannot build them (on every rank alike, before any other collective).
    """
    attached = getattr(getattr(llm.logits_processor, "_split_head", None), "s2", None)
    if attached is not None:
        return attached
    import time

    from tensorrt_llm.logger import logger

    from ... import dispatch as _tp_agreement

    reason = _tp_agreement.agreed_refusal(mapping, unsupported_reason(llm, mapping), facts=rs_facts)
    if reason is not None:
        raise ValueError(f"[musespark_open] s2 sampler cannot run here: {reason}")
    head = llm.logits_processor._split_head
    lm_head = llm.lm_head
    device = lm_head.weight.device
    free_before, _ = torch.cuda.mem_get_info(device)
    start = time.perf_counter()
    rs_areas = rs_facts is not None and rs_facts[0] and rs_facts[1] != "inert"
    states = S2States(mapping, int(lm_head.weight.shape[0]), device, rs_areas=rs_areas)
    weight = build_interleaved_shard(lm_head.weight.detach(), mapping)
    torch.cuda.empty_cache()
    seconds = time.perf_counter() - start
    free_after, _ = torch.cuda.mem_get_info(device)
    head.s2 = S2Head(mapping, lm_head, weight, states, seconds)
    logger.info(
        "[musespark_open] s2 sampler: interleaved LM-head copy "
        f"{weight.numel() * weight.element_size() / 2**20:.1f} MiB + fabric areas {states.total / 2**20:.2f} MiB "
        f"built in {seconds:.1f} s (fold shape {states.vts}x{states.threads} form {states.form}); device free "
        f"{free_before / 2**30:.3f} -> {free_after / 2**30:.3f} GiB; arms at the first 8-row decode step outside "
        "autotuning"
    )
    return head.s2

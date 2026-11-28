# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The per-request integer bookkeeping of a sampled DFlash2 decode step, one launch each.

* ``k31_dflash2_accept_commit``: the end of ``dflash2_sample_and_accept`` for a step of generation requests: the
  greedy / block verification pick by temperature, the emitted tokens, the counts and the engine's forced
  acceptance count (``SpecWorkerBase._apply_force_accepted_tokens``).
* ``k31_dflash2_accept_record``: the KDA replay state's accepted-draft record (``KDAReplayState.record_acceptance``
  through ``update_mamba_states``) and the drafter's KV-length preparation (``_prepare_kv_for_draft_forward``).
* ``k31_dflash2_draft_tail``: the deferred KV-length rewind (``_apply_kv_rewind_after_draft``) and the next step's
  input tokens (``_prepare_next_new_tokens``).
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.spec.dflash2_step_bookkeeping import (
    FORCE_CONSTANT,
    FORCE_OFF,
    FORCE_POOL,
    MAX_REQUESTS,
    accept_commit,
    accept_record,
    draft_tail,
)

#: The engine's forced-acceptance pool (``_FORCE_ACCEPT_RNG_*`` of ``speculative/interface.py``).
FORCE_POOL_SIZE = 1 << 16
FORCE_COUNTER_STRIDE = 6007
FORCE_SLOT_STRIDE = 1009


def _commit_spec(
    n, K, *, force=FORCE_OFF, forced=0.0, temps="mixed", draft_dtype=torch.int32
) -> dict:
    return dict(n=n, K=K, force=force, forced=forced, temps=temps, draft_dtype=draft_dtype)


def _record_spec(n, *, dummies="none", state_dtype=torch.int32, slots=None, prev_stride=1) -> dict:
    return dict(
        n=n,
        dummies=dummies,
        state_dtype=state_dtype,
        slots=slots or 4 * n + 3,
        prev_stride=prev_stride,
    )


def _tail_spec(b, K, *, rewind=True, batch_dtype=torch.int32, counts="valid") -> dict:
    return dict(b=b, K=K, rewind=rewind, batch_dtype=batch_dtype, counts=counts)


class K31DFlash2AcceptCommit(OpWrapper):
    """The verified block of each generation request: ``(accepted [n, K + 1] int32, num_accepted [n] int32)``.

    ``tau`` / ``bonus`` come from the greedy rule where ``temperatures <= threshold`` (``torch.where`` of the two
    verifications), the tokens are the drafts with the last draft repeated and the bonus at ``tau``
    (``cat(drafts, drafts[:, -1:]).scatter(1, tau, bonus)``), the count ``tau + 1``. ``force`` then overrides the
    counts as ``_apply_force_accepted_tokens`` does: ``FORCE_CONSTANT`` writes ``base_total``; ``FORCE_POOL`` bumps
    ``counter`` once and draws ``extra = pool[(counter * counter_stride + r * slot_stride) & (len(pool) - 1)] <
    frac`` for request ``r``, the count ``min(base_total + extra, max_total)``.

    Args:
        tau, bonus, greedy_tau, greedy_bonus: ``[n]`` int64, the block and the greedy verifications.
        temperatures: ``[n]`` FP32 request temperatures; ``threshold`` the greedy one.
        draft_tokens: ``[n, K]`` int32 / int64 verified drafts.
        force, base_total, max_total, frac: the forced count (``FORCE_*``) and its parameters.
        counter: ``[1]`` int64, bumped in place (``FORCE_POOL``); pool: ``[2^k]`` FP32 uniforms.
        counter_stride, slot_stride: the pool index strides.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="TEP8 B7, K = 4, sampled and greedy rows", spec=_commit_spec(7, 4)),
        Cell(why="TEP16 B16, K = 4, all sampled", spec=_commit_spec(16, 4, temps="hot")),
        Cell(
            why="64 requests, K = 7, int64 drafts",
            spec=_commit_spec(64, 7, draft_dtype=torch.int64),
        ),
        Cell(
            why="forced 2.21 at K = 4: the pool draw (the bench's fractional acceptance)",
            spec=_commit_spec(7, 4, force=FORCE_POOL, forced=2.21),
        ),
        Cell(
            why="forced 3.68 at K = 7, 16 requests: the pool draw",
            spec=_commit_spec(16, 7, force=FORCE_POOL, forced=3.68),
        ),
        Cell(
            why="forced 3.0 at K = 4: the constant count",
            spec=_commit_spec(8, 4, force=FORCE_CONSTANT, forced=3.0),
        ),
        Cell(
            why="forced 4.5 at K = 4: capped at K + 1, constant",
            spec=_commit_spec(5, 4, force=FORCE_CONSTANT, forced=4.5),
        ),
        Cell(
            why="one request, K = 1, the threshold's edges", spec=_commit_spec(1, 1, temps="edges")
        ),
        Cell(why="300 requests, K = 5 (three rounds of the CTA)", spec=_commit_spec(300, 5)),
    )

    note = """
    Kernel: ``kernels/spec/dflash2_step_bookkeeping.py`` (``AcceptCommitKernel``), one CTA a thread a request: the
    integer arithmetic of the torch path in its types (int64 picks and pool index, int32 outputs, the FP32
    temperature and pool comparisons). Replaces about 25 one-CTA launches and three copies of the sampled verify's
    tail (with the forced count) or about 12 (without). Bitwise on every cell.
    """

    def __call__(
        self,
        tau: torch.Tensor,
        bonus: torch.Tensor,
        greedy_tau: torch.Tensor,
        greedy_bonus: torch.Tensor,
        temperatures: torch.Tensor,
        threshold: float,
        draft_tokens: torch.Tensor,
        force: int,
        base_total: int,
        max_total: int,
        frac: float,
        counter: torch.Tensor,
        pool: torch.Tensor,
        counter_stride: int,
        slot_stride: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n, K = draft_tokens.shape
        accepted = torch.empty((n, K + 1), dtype=torch.int32, device=draft_tokens.device)
        num_accepted = torch.empty((n,), dtype=torch.int32, device=draft_tokens.device)
        accept_commit(
            tau,
            bonus,
            greedy_tau,
            greedy_bonus,
            temperatures,
            threshold,
            draft_tokens,
            accepted,
            num_accepted,
            force,
            base_total,
            max_total,
            frac,
            counter,
            pool,
            counter_stride,
            slot_stride,
        )
        return accepted, num_accepted

    def reference(
        self,
        tau: torch.Tensor,
        bonus: torch.Tensor,
        greedy_tau: torch.Tensor,
        greedy_bonus: torch.Tensor,
        temperatures: torch.Tensor,
        threshold: float,
        draft_tokens: torch.Tensor,
        force: int,
        base_total: int,
        max_total: int,
        frac: float,
        counter: torch.Tensor,
        pool: torch.Tensor,
        counter_stride: int,
        slot_stride: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``dflash2_sample_and_accept``'s torch tail and ``_apply_force_accepted_tokens``' override (``counter``
        bumped in place, as the engine does)."""
        n = draft_tokens.shape[0]
        greedy = temperatures <= threshold
        tau = torch.where(greedy, greedy_tau, tau)
        bonus = torch.where(greedy, greedy_bonus, bonus)
        tokens = torch.cat((draft_tokens.long(), draft_tokens[:, -1:].long()), dim=1)
        tokens = tokens.scatter(1, tau.unsqueeze(1), bonus.unsqueeze(1))
        accepted = tokens.to(torch.int32)
        num_accepted = (tau + 1).to(torch.int32)
        if force == FORCE_CONSTANT:
            num_accepted[:] = base_total
        elif force == FORCE_POOL:
            counter += 1
            slots = torch.arange(n, device=draft_tokens.device, dtype=torch.int64)
            indices = (counter * counter_stride + slots * slot_stride) & (pool.shape[0] - 1)
            extra = (pool[indices] < frac).to(num_accepted.dtype)
            num_accepted[:] = (base_total + extra).clamp_(max=max_total)
        return accepted, num_accepted

    def is_valid(
        self,
        tau: torch.Tensor,
        bonus: torch.Tensor,
        greedy_tau: torch.Tensor,
        greedy_bonus: torch.Tensor,
        temperatures: torch.Tensor,
        threshold: float,
        draft_tokens: torch.Tensor,
        force: int,
        base_total: int,
        max_total: int,
        frac: float,
        counter: torch.Tensor,
        pool: torch.Tensor,
        counter_stride: int,
        slot_stride: int,
    ) -> None:
        for t in (tau, bonus, greedy_tau, greedy_bonus):
            assert t.dtype == torch.int64 and t.is_contiguous(), "int64 verification picks"
        assert temperatures.dtype == torch.float32 and temperatures.is_contiguous(), (
            "FP32 temperatures"
        )
        assert draft_tokens.dtype in (torch.int32, torch.int64) and draft_tokens.is_contiguous(), (
            "int drafts"
        )
        assert force in (FORCE_OFF, FORCE_CONSTANT, FORCE_POOL), "a FORCE_* mode"
        if force == FORCE_POOL:
            assert counter.dtype == torch.int64 and counter.numel() == 1, "an int64 [1] counter"
            assert pool.dtype == torch.float32 and pool.shape[0] & (pool.shape[0] - 1) == 0, (
                "a 2^k pool"
            )


class K31DFlash2AcceptRecord(OpWrapper):
    """The accepted-draft record of the KDA replay state and the drafter's KV-length preparation of ``n`` generation
    requests; returns the deferred rewind ``1 - num_accepted`` ([n] int32).

    ``prev_num_accepted[state_indices[r]]`` becomes ``max(num_accepted[r] - 1, 0)`` (a dummy request's slot keeps
    the value read before any write: ``record_acceptance``'s gather, ``where``, ``index_put``), and ``kv_lens`` gains
    one, in place.

    Args:
        num_accepted: ``[n]`` int32 accepted counts.
        state_indices: ``[n]`` int32 / int64 state slots; is_dummy: ``[n]`` bool.
        prev_num_accepted: ``[slots]`` int32, the replay state's counts (in place; any slot stride: the V2 manager
            keeps them in the KV pool at its slot pitch).
        kv_lens: ``[n]`` int32, the generation requests' KV lengths (in place).
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="TEP8 B7, no dummy requests", spec=_record_spec(7)),
        Cell(why="TEP16 B16, two CUDA-graph dummies", spec=_record_spec(16, dummies="some")),
        Cell(why="every request a dummy", spec=_record_spec(5, dummies="all")),
        Cell(why="int64 state slots, 64 requests", spec=_record_spec(64, state_dtype=torch.int64)),
        Cell(why="300 requests (three a thread)", spec=_record_spec(300, dummies="some")),
        Cell(why="counts of 0 clamp to 0 drafts", spec=_record_spec(8, dummies="zero")),
        Cell(
            why="replay counts in the KV pool: one int32 per slot at the slot pitch",
            spec=_record_spec(16, dummies="some", prev_stride=1040),
        ),
    )

    note = """
    Kernel: ``kernels/spec/dflash2_step_bookkeeping.py`` (``AcceptRecordKernel``), one CTA: every slot read, one
    barrier, every slot written (torch's gather before its index_put), then the rewind and the KV lengths. Replaces
    nine one-CTA launches. Bitwise on every cell (slots of distinct requests are distinct, as the manager assigns
    them; a slot shared by a dummy and a real request is undefined in torch's index_put and here alike).
    """

    def __call__(
        self,
        num_accepted: torch.Tensor,
        state_indices: torch.Tensor,
        is_dummy: torch.Tensor,
        prev_num_accepted: torch.Tensor,
        kv_lens: torch.Tensor,
    ) -> torch.Tensor:
        rewind = torch.empty_like(num_accepted)
        accept_record(num_accepted, state_indices, is_dummy, prev_num_accepted, kv_lens, rewind)
        return rewind

    def reference(
        self,
        num_accepted: torch.Tensor,
        state_indices: torch.Tensor,
        is_dummy: torch.Tensor,
        prev_num_accepted: torch.Tensor,
        kv_lens: torch.Tensor,
    ) -> torch.Tensor:
        """``KDAReplayState.record_acceptance(state_indices, num_accepted - 1, is_dummy)`` and
        ``_prepare_kv_for_draft_forward``'s device half."""
        slots = state_indices.to(torch.long)
        accepted = (num_accepted - 1).to(torch.int32).clamp(min=0)
        current = prev_num_accepted[slots]
        accepted = torch.where(is_dummy, current, accepted)
        prev_num_accepted[slots] = accepted
        rewind = 1 - num_accepted
        kv_lens += 1
        return rewind

    def is_valid(
        self,
        num_accepted: torch.Tensor,
        state_indices: torch.Tensor,
        is_dummy: torch.Tensor,
        prev_num_accepted: torch.Tensor,
        kv_lens: torch.Tensor,
    ) -> None:
        assert num_accepted.shape[0] <= MAX_REQUESTS, f"at most {MAX_REQUESTS} requests a launch"
        assert num_accepted.dtype == torch.int32 and kv_lens.dtype == torch.int32, (
            "int32 counts and lengths"
        )
        assert prev_num_accepted.dtype == torch.int32, "int32 replay counts"
        assert state_indices.dtype in (torch.int32, torch.int64) and is_dummy.dtype == torch.bool
        for t in (num_accepted, state_indices, is_dummy, kv_lens):
            assert t.is_contiguous(), "contiguous vectors"
        assert prev_num_accepted.dim() == 1, "a vector of replay counts (any stride)"


class K31DFlash2DraftTail(OpWrapper):
    """The KV-length rewind after the drafter and the next step's input tokens: ``next_new_tokens [b, 1 + K]``
    int32, ``accepted[batch_rows, num_accepted - 1]`` then ``next_drafts``; with ``apply_rewind``, ``kv_lens`` becomes
    ``max(kv_lens - rewind, 0)`` in place.

    Args:
        kv_lens, rewind: ``[n]`` int32 KV lengths of the generation requests and their deferred rewind.
        accepted: ``[rows, K + 1]`` int32 accepted tokens; batch_rows: ``[b]`` int32 / int64 rows of it.
        num_accepted: ``[b]`` int32; next_drafts: ``[b, K]`` int32.
        apply_rewind: the rewind applies (not a warm-up step, a rewind pending).
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="TEP8 B7, K = 4", spec=_tail_spec(7, 4)),
        Cell(
            why="TEP16 B16, K = 4, int64 batch rows",
            spec=_tail_spec(16, 4, batch_dtype=torch.int64),
        ),
        Cell(why="warm-up: no rewind", spec=_tail_spec(8, 7, rewind=False)),
        Cell(
            why="rewind below zero clamps; a count of 0 reads the last column",
            spec=_tail_spec(9, 4, counts="edge"),
        ),
        Cell(why="300 requests, K = 5", spec=_tail_spec(300, 5)),
    )

    note = """
    Kernel: ``kernels/spec/dflash2_step_bookkeeping.py`` (``DraftTailKernel``), one CTA, a thread a request. Replaces
    seven one-CTA launches. Bitwise on every cell.
    """

    def __call__(
        self,
        kv_lens: torch.Tensor,
        rewind: torch.Tensor,
        accepted: torch.Tensor,
        batch_rows: torch.Tensor,
        num_accepted: torch.Tensor,
        next_drafts: torch.Tensor,
        apply_rewind: bool,
    ) -> torch.Tensor:
        b, K = next_drafts.shape
        out = torch.empty((b, 1 + K), dtype=torch.int32, device=next_drafts.device)
        if b > 0 or apply_rewind:
            draft_tail(
                kv_lens, rewind, accepted, batch_rows, num_accepted, next_drafts, out, apply_rewind
            )
        return out

    def reference(
        self,
        kv_lens: torch.Tensor,
        rewind: torch.Tensor,
        accepted: torch.Tensor,
        batch_rows: torch.Tensor,
        num_accepted: torch.Tensor,
        next_drafts: torch.Tensor,
        apply_rewind: bool,
    ) -> torch.Tensor:
        """``_apply_kv_rewind_after_draft``'s device half and ``_prepare_next_new_tokens``."""
        if apply_rewind:
            kv_lens -= rewind
            kv_lens.clamp_(min=0)
        last = accepted[batch_rows, num_accepted - 1].unsqueeze(1)
        return torch.concat([last, next_drafts], dim=1)

    def is_valid(
        self,
        kv_lens: torch.Tensor,
        rewind: torch.Tensor,
        accepted: torch.Tensor,
        batch_rows: torch.Tensor,
        num_accepted: torch.Tensor,
        next_drafts: torch.Tensor,
        apply_rewind: bool,
    ) -> None:
        assert accepted.dtype == torch.int32 and next_drafts.dtype == torch.int32, "int32 tokens"
        assert num_accepted.dtype == torch.int32 and kv_lens.dtype == torch.int32, "int32 counts"
        assert batch_rows.dtype in (torch.int32, torch.int64), "int batch rows"
        for t in (kv_lens, rewind, accepted, batch_rows, num_accepted, next_drafts):
            assert t.is_contiguous(), "contiguous tensors"


k31_dflash2_accept_commit = K31DFlash2AcceptCommit()
k31_dflash2_accept_record = K31DFlash2AcceptRecord()
k31_dflash2_draft_tail = K31DFlash2DraftTail()

__all__ = [
    "FORCE_CONSTANT",
    "FORCE_COUNTER_STRIDE",
    "FORCE_OFF",
    "FORCE_POOL",
    "FORCE_POOL_SIZE",
    "FORCE_SLOT_STRIDE",
    "MAX_REQUESTS",
    "K31DFlash2AcceptCommit",
    "K31DFlash2AcceptRecord",
    "K31DFlash2DraftTail",
    "k31_dflash2_accept_commit",
    "k31_dflash2_accept_record",
    "k31_dflash2_draft_tail",
]

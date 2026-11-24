# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The verify step's target selection, strict acceptance and the bookkeeping after it in one
launch: ``kernels/k3/spec_tail_kernels.py``'s ``fused_select_accept`` (one Triton program)."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.spec import rubin_ull_k3_spec_tail_kernels as _k3tail
from ._rubin_ull_pairs import pair_argmax


def _cell(why: str, gens: int, **spec) -> Cell:
    base = dict(
        contexts=0,
        gens=gens,
        draft_len=7,
        slots=80,
        width=256,
        force=0,
        kda=True,
        kv=True,
        matches=None,
        ties=False,
        count=False,
        real=None,
    )
    base.update(spec)
    return Cell(why=why, spec=base)


class SelectAccept(OpWrapper):
    """From ``pairs = (buffer, row stride, 2, n_slots)``, every rank's exchanged (value, FP32
    global index) candidates of the ``num_contexts + num_gens * (K + 1)`` verify rows:

    * each row's token: the global index ``torch.argmax`` of the gathered logits picks (NaN
      first, the lowest index among equal values);
    * a context request accepts its token (``accepted[c, 0]``, count 1); generation request ``g``
      writes its ``K + 1`` tokens ``t`` to ``accepted[nc + g]`` and accepts ``1 +`` the leading run
      of ``draft_tokens[g, j] == t[j]``, then ``force``: ``(base,)`` sets every count to ``base``;
      ``(base, frac, max, pool, counter, counter_stride, slot_stride)`` advances ``counter`` by one
      and sets ``min(base + (pool[(counter * counter_stride + g * slot_stride) & (len - 1)] < frac),
      max)``; with ``real_rows`` (bool per generation row, ``SpecMetadata.real_draft_rows``)
      only the rows marked True are forced, the others keep their strict count;
    * ``kda_replay = (prev, state_indices, dummy_mask)``: ``prev[state_indices[nc + g]] =
      max(n - 1, 0)`` unless the row is a dummy (``KDAReplayState.record_acceptance``);
    * ``kv_lens``: ``kv_lens[nc + g] += 1`` and the rewind ``1 - n`` returned
      (``_prepare_kv_for_draft_forward``);
    * ``count``: int64 ``[2]`` running totals, ``+= num_gens`` and ``+= sum(n)`` (the forced
      acceptance's device count); with ``real_rows``, over the forced rows only.

    Returns ``(accepted [batch, K + 1] int32, num_accepted [batch] int32, rewind [num_gens] int32 or
    None)``; the context rows' columns past 0 are not written.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    # The flat head's exchange: 16 ranks x 5 splits = 80 slots in a 256-float row; K 7.
    CELLS: tuple[Cell, ...] = (
        _cell("served b1: one request, KDA record and kv_lens prep", 1, matches=(3,)),
        _cell(
            "served b4: accepted 1, 8, 5, 2 (all and none of the drafts)", 4, matches=(0, 7, 4, 1)
        ),
        _cell("b2 with a dummy row (its KDA record kept)", 2, dummy=1, matches=(6, 2)),
        _cell("forced count 3 (mode 1)", 2, force=1, matches=(0, 7)),
        _cell("forced 2.91 (mode 2): the pool draw and the counter advance", 4, force=2),
        _cell(
            "a context request first (warm-up), no bookkeeping folded",
            2,
            contexts=1,
            kda=False,
            kv=False,
        ),
        _cell("16 requests: the largest fused batch", 16),
        _cell(
            "served b1, forced draw (mode 2) and the forced-count totals", 1, force=2, count=True
        ),
        _cell(
            "forced count 3 and the totals, a context request first",
            3,
            contexts=1,
            force=1,
            kda=False,
            kv=False,
            count=True,
        ),
        _cell("NaN and equal values across ranks", 2, ties=True),
        _cell(
            "forced 2.91 with every row real: as without real_rows",
            4,
            force=2,
            count=True,
            real=(True, True, True, True),
        ),
        _cell(
            "forced 2.91, placeholder rows keep their strict count and are not counted",
            4,
            force=2,
            count=True,
            matches=(0, 7, 4, 1),
            real=(False, True, False, True),
        ),
        _cell(
            "forced count 3, no real row: strict counts, totals unchanged",
            3,
            contexts=1,
            force=1,
            kda=False,
            kv=False,
            count=True,
            matches=(2, 0, 7),
            real=(False, False, False),
        ),
        _cell(
            "served b1, the first step after the handoff (placeholder drafts)",
            1,
            force=2,
            count=True,
            matches=(0,),
            real=(False,),
        ),
    )

    note = """
    The helix16 target's DFlash2 worker launches it once per all-greedy verify step for a uniform
    batch of at most 16 requests of each kind with the head's exchanged argmax pairs
    (``FlatVocabParallelHead.selection_pairs``: 16 ranks x 5 splits at one request's 8 rows),
    folding the KDA replay record and the drafter's kv_lens prep where the manager allows
    (``DFlashWorker._fused_select_accept``); the helix8 target's worker does the same under its
    draft_select_accept constant with the engine's vocab-parallel head (8 ranks; the test's helix8
    cells hold its pair layouts); with forced acceptance it passes the engine's
    ``real_draft_rows``, so the first step after a disaggregated handoff (placeholder drafts) keeps
    its strict count, as ``SpecWorkerBase._apply_force_accepted_tokens`` does. One
    program, 4 warps; the force counter is read by every warp before it is written, the replay
    slots all read before any is written. Integer and FP32-compare work only: exact. Compiles per
    (K, force mode, folds, block sizes).
    """

    def __call__(
        self,
        pairs: tuple,
        draft_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
        force: Optional[tuple] = None,
        kda_replay: Optional[tuple] = None,
        kv_lens: Optional[torch.Tensor] = None,
        count: Optional[torch.Tensor] = None,
        real_rows: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return _k3tail.fused_select_accept(
            pairs,
            draft_tokens,
            num_contexts,
            batch_size,
            force,
            kda_replay=kda_replay,
            kv_lens=kv_lens,
            count=count,
            real_rows=real_rows,
        )

    def reference(
        self,
        pairs: tuple,
        draft_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
        force: Optional[tuple] = None,
        kda_replay: Optional[tuple] = None,
        kv_lens: Optional[torch.Tensor] = None,
        count: Optional[torch.Tensor] = None,
        real_rows: Optional[torch.Tensor] = None,
    ) -> dict:
        """``tokens`` (each verify row's selected id), ``num_accepted``, ``rewind`` and the state
        after the call: ``prev``, ``kv_lens``, ``counter``, ``count`` (None where not passed);
        nothing passed in is modified."""
        buf, stride_row, _, n_slots = pairs
        nc, ng = num_contexts, batch_size - num_contexts
        k = draft_tokens.shape[-1]
        rows = nc + ng * (k + 1)
        p = buf.as_strided((rows, 2 * n_slots), (stride_row, 1)).float()
        tokens = pair_argmax(p[:, 0::2], p[:, 1::2].to(torch.int64)).to(torch.int32)
        dev = buf.device
        num = torch.ones(batch_size, dtype=torch.int32, device=dev)
        gen_tokens = tokens[nc:].view(ng, k + 1)
        if ng:
            match = (draft_tokens.to(torch.int32) == gen_tokens[:, :k]).int()
            num[nc:] = 1 + match.cumprod(-1).sum(-1).int()
        counter = None
        forced = None
        if force is not None and len(force) == 1:
            forced = torch.full((ng,), int(force[0]), dtype=torch.int32, device=dev)
        elif force is not None:
            base, frac, top, pool, cnt, cs, ss = force
            counter = cnt + 1
            g = torch.arange(ng, device=dev, dtype=torch.int64)
            idx = (counter.to(torch.int64) * cs + g * ss) & (pool.numel() - 1)
            forced = torch.clamp(base + (pool[idx] < frac).int(), max=top).int()
        real = None
        if forced is not None:
            if real_rows is not None:
                real = real_rows[:ng]
                forced = torch.where(real, forced, num[nc:])
            num[nc:] = forced
        prev_after = None
        if kda_replay is not None and ng:
            prev, state_idx, dummy = kda_replay
            prev_after = prev.clone()
            slots = state_idx[nc:batch_size].long()
            live = dummy[nc:batch_size] == 0
            prev_after[slots[live]] = (num[nc:][live] - 1).clamp(min=0)
        kv_after, rewind = None, None
        if kv_lens is not None:
            kv_after = kv_lens.clone()
            kv_after[nc:batch_size] += 1
            rewind = 1 - num[nc:]
        count_after = None
        if count is not None:
            count_after = count.clone()
            if real is None:
                count_after[0] += ng
                count_after[1] += num[nc:].long().sum()
            else:
                count_after[0] += real.long().sum()
                count_after[1] += torch.where(real, num[nc:].long(), 0).sum()
        return dict(
            tokens=tokens,
            num_accepted=num,
            rewind=rewind,
            prev=prev_after,
            kv_lens=kv_after,
            counter=counter,
            count=count_after,
        )

    def is_valid(
        self,
        pairs: tuple,
        draft_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
        force: Optional[tuple] = None,
        kda_replay: Optional[tuple] = None,
        kv_lens: Optional[torch.Tensor] = None,
        count: Optional[torch.Tensor] = None,
        real_rows: Optional[torch.Tensor] = None,
    ) -> None:
        """The one program holds every row in registers and reads the operands unguarded."""
        buf, stride_row, _, n_slots = pairs
        nc, ng = num_contexts, batch_size - num_contexts
        k = draft_tokens.shape[-1]
        if ng and draft_tokens.shape[0] < ng:
            raise ValueError("select_accept: a draft row per generation request")
        if (
            buf.dtype != torch.float32
            or buf.numel() < (nc + ng * (k + 1) - 1) * stride_row + 2 * n_slots
        ):
            raise ValueError("select_accept: FP32 pairs covering every verify row")
        if force is not None and len(force) != 1:
            pool = force[3]
            if pool.numel() & (pool.numel() - 1):
                raise ValueError("select_accept: the force pool's size must be a power of two")
        if kda_replay is not None:
            prev, state_idx, dummy = kda_replay
            if state_idx.shape[0] < batch_size or dummy.shape[0] < batch_size:
                raise ValueError("select_accept: state indices and dummy mask cover the batch")
            if ng and int(state_idx[nc:batch_size].max()) >= prev.shape[0]:
                raise ValueError("select_accept: a state index past the replay records")
        if kv_lens is not None and kv_lens.shape[0] < batch_size:
            raise ValueError("select_accept: kv_lens shorter than the batch")
        if count is not None and (
            count.dtype != torch.int64 or count.numel() < 2 or not count.is_contiguous()
        ):
            raise ValueError("select_accept: count is a contiguous int64 [2]")
        if real_rows is not None and (
            real_rows.dtype != torch.bool or not real_rows.is_contiguous() or real_rows.numel() < ng
        ):
            raise ValueError("select_accept: real_rows is a contiguous bool per generation row")


select_accept = SelectAccept()

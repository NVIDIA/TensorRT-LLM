# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's speculative acceptance of one decode step and the block drafter's inputs in one launch: greedy strict
acceptance of B generation requests of K drafts each, the draft pool's block table, the KDA replay record, kv_lens + 1,
and the drafter's bonus tokens, positions and noise embedding. The target logits are fp32 and whole, or this rank's
bf16 vocabulary shard whose row maxima the TP group exchanges over a multicast Lamport buffer."""

from typing import List, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.kernels.k3_spec_accept import (  # noqa: F401, E501 — registers the op
    op as _k3_spec_accept_op,
)

# SpecWorkerBase._apply_force_accepted_tokens' draw from the fixed RNG pool (interface._FORCE_ACCEPT_RNG_*).
_RNG_COUNTER_STRIDE = 6007
_RNG_SLOT_STRIDE = 1009
_RNG_POOL_SIZE = 1 << 16


class K3SpecAccept(OpWrapper):
    """Return ``[accepted [B, K + 1] int32, num_accepted [B] int32, rewind [B] int32, bonus [B] int64,
    query_positions [B, block] int64, ctx_positions [B, K + 1] int64, noise [B, block, H] bf16]`` of the step whose
    target logits are ``logits`` and drafts ``draft`` [B, K]; updates ``block_counts``, ``block_tables``,
    ``prev_acc``, ``kv_lens`` and ``rng_counter`` in place."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS = (
        Cell(
            why="one request, seven drafts, DSpark's block of K, natural acceptance over six carried steps",
            spec=dict(batch=1, drafts=7, block=7, logits="plain", force=0.0, steps=6),
        ),
        Cell(
            why="eight requests (the kernel's maximum, 64 logits rows) with every accepted-prefix length 0..7",
            spec=dict(batch=8, drafts=7, block=7, logits="plain", force=0.0, steps=6),
        ),
        Cell(
            why="three requests at a block of K + 1 (a drafter whose block holds the bonus and every draft)",
            spec=dict(batch=3, drafts=7, block=8, logits="plain", force=0.0, steps=6),
        ),
        Cell(
            why="five requests with tied maxima across CTA boundaries (the lowest column wins) and two CUDA-graph "
            "padding requests sharing one KDA slot and one context slot",
            spec=dict(batch=5, drafts=7, block=7, logits="ties", force=0.0, steps=6),
        ),
        Cell(
            why="six requests under a fractional forced acceptance (5.4): the per-request draw from the RNG pool and "
            "the counter it advances",
            spec=dict(batch=6, drafts=7, block=7, logits="plain", force=5.4, steps=6),
        ),
        Cell(
            why="TP16's 10240-column bf16 shard on four MNNVL ranks filling four exchange slots each (the 16-slot "
            "exchange of TP16), one request",
            spec=dict(
                batch=1,
                drafts=7,
                block=7,
                logits="plain",
                force=0.0,
                steps=6,
                columns=10240,
                push_copies=4,
            ),
        ),
        Cell(
            why="TP16's shard with eight requests and maxima tied across CTA and shard boundaries",
            spec=dict(
                batch=8,
                drafts=7,
                block=7,
                logits="ties",
                force=0.0,
                steps=6,
                columns=10240,
                push_copies=4,
            ),
        ),
        Cell(
            why="a quarter of the vocabulary (40960 columns) per rank of a four-rank group, eight requests",
            spec=dict(
                batch=8,
                drafts=7,
                block=7,
                logits="plain",
                force=0.0,
                steps=6,
                columns=40960,
                push_copies=1,
            ),
        ),
        Cell(
            why="a quarter of the vocabulary per rank under a fractional forced acceptance (5.4), five requests",
            spec=dict(
                batch=5,
                drafts=7,
                block=7,
                logits="plain",
                force=5.4,
                steps=6,
                columns=40960,
                push_copies=1,
            ),
        ),
    )
    note = """
    Cells without ``columns`` pass fp32 logits of the whole 163840-token
    vocabulary; cells with it pass a rank's bf16 shard on four ranks, and
    the result equals the unsharded call on the gathered logits in fp32.
    The kernel compiles on the first call per shape and allocates its
    arrival counter on the first call per grid; both must happen outside
    CUDA-graph capture. Every call on a device with the same logits width
    shares that counter, so calls must be ordered on one stream. The
    sharded form is collective: every rank of the group makes the same
    calls in the same order, and the workspace (``op.workspace``) is
    allocated collectively outside capture. A request marked dummy keeps
    its KDA replay record. Forced acceptance reads ``rng_pool`` and
    advances ``rng_counter`` only for a fractional value below K; other
    values leave both untouched.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        draft: torch.Tensor,
        block_off: torch.Tensor,
        pool_idx: int,
        divisor: int,
        block_counts: torch.Tensor,
        block_tables: torch.Tensor,
        prev_acc: torch.Tensor,
        state_idx: torch.Tensor,
        dummy: torch.Tensor,
        kv_lens: torch.Tensor,
        batch_to_slot: torch.Tensor,
        ctx_len: torch.Tensor,
        max_ctx: int,
        embed: torch.Tensor,
        mask_row: torch.Tensor,
        rng_pool: torch.Tensor,
        rng_counter: torch.Tensor,
        force: float,
        block: int,
        ws_uc: Optional[torch.Tensor] = None,
        ws_mc: Optional[torch.Tensor] = None,
        ws_flags: Optional[torch.Tensor] = None,
        rank: int = 0,
        slots: int = 1,
        push_copies: int = 1,
        shard_offset: int = 0,
    ) -> List[torch.Tensor]:
        return torch.ops.trtllm.k3_spec_accept(
            logits,
            draft,
            block_off,
            pool_idx,
            divisor,
            block_counts,
            block_tables,
            prev_acc,
            state_idx,
            dummy,
            kv_lens,
            batch_to_slot,
            ctx_len,
            max_ctx,
            embed,
            mask_row,
            rng_pool,
            rng_counter,
            force,
            block,
            ws_uc,
            ws_mc,
            ws_flags,
            rank,
            slots,
            push_copies,
            shard_offset,
        )

    def reference(
        self,
        logits: Union[torch.Tensor, Sequence[torch.Tensor]],
        draft: torch.Tensor,
        block_off: torch.Tensor,
        pool_idx: int,
        divisor: int,
        block_counts: torch.Tensor,
        block_tables: torch.Tensor,
        prev_acc: torch.Tensor,
        state_idx: torch.Tensor,
        dummy: torch.Tensor,
        kv_lens: torch.Tensor,
        batch_to_slot: torch.Tensor,
        ctx_len: torch.Tensor,
        max_ctx: int,
        embed: torch.Tensor,
        mask_row: torch.Tensor,
        rng_pool: torch.Tensor,
        rng_counter: torch.Tensor,
        force: float,
        block: int,
        ws_uc: Optional[torch.Tensor] = None,
        ws_mc: Optional[torch.Tensor] = None,
        ws_flags: Optional[torch.Tensor] = None,
        rank: int = 0,
        slots: int = 1,
        push_copies: int = 1,
        shard_offset: int = 0,
    ) -> List[torch.Tensor]:
        """The step as the speculative worker's torch ops compute it: the draft pool's block table, greedy
        acceptance with the forced-acceptance override, the KDA replay record, kv_lens + 1 and the drafter's inputs
        up to its fc. Updates the same state tensors in place.

        Collective form (``slots`` > 1): ``logits`` is the group's bf16 shards in rank order, and the reference is
        the unsharded step on their concatenation along the columns in fp32. ``ws_uc``, ``ws_mc``, ``ws_flags``,
        ``rank``, ``push_copies`` and ``shard_offset`` only route the exchange and are ignored."""
        if slots > 1:
            logits = torch.cat(list(logits), dim=1).float()
        batch, drafts = draft.shape
        dev = logits.device
        # The draft pool's block table, decoded from the encoded block offsets (placeholders are -1).
        encoded = block_off[pool_idx, :batch, 0].to(torch.int64).clone()
        block_counts[:batch].copy_((encoded >= 0).sum(dim=1))
        decoded = encoded.clamp_(min=0).div_(divisor, rounding_mode="floor")
        block_tables[:batch].copy_(decoded.to(torch.int32))
        # Greedy strict acceptance: the accepted prefix of drafts equal to the target tokens, plus the bonus token.
        target = torch.argmax(logits, dim=-1).reshape(batch, drafts + 1)
        accepted = target.to(torch.int32)
        num_acc = torch.ones(batch, dtype=torch.int32, device=dev)
        num_acc += torch.cumprod((draft == target[:, :drafts]).int(), dim=-1).sum(1).to(torch.int32)
        # Forced acceptance: an integer total, or a fractional one drawn per request from the fixed RNG pool.
        if force != 0.0:
            int_part = int(force)
            frac = force - int_part
            max_total = drafts + 1
            base_total = min(int_part + 1, max_total)
            if frac > 0.0 and base_total < max_total:
                rng_counter += 1
                slot_ids = torch.arange(batch, device=dev, dtype=torch.int64)
                hashed = rng_counter * _RNG_COUNTER_STRIDE + slot_ids * _RNG_SLOT_STRIDE
                indices = hashed & (_RNG_POOL_SIZE - 1)
                extra = (rng_pool[indices] < frac).to(num_acc.dtype)
                num_acc[:] = (base_total + extra).clamp_(max=max_total)
            else:
                num_acc[:] = base_total
        # The KDA replay record: accepted drafts per replay slot (state_idx holds the replay slots); a dummy request
        # keeps its slot's record.
        slots_b = state_idx[:batch].to(torch.long)
        acc = (num_acc - 1).clamp(min=0)
        acc = torch.where(dummy[:batch], prev_acc[slots_b], acc)
        prev_acc[slots_b] = acc
        rewind = 1 - num_acc
        kv_lens[:batch] += 1
        # The drafter's inputs: the bonus token, the block's query positions, the context positions and the noise
        # embedding (the bonus token's embedding in front of the mask rows).
        bonus = accepted.gather(1, (num_acc - 1).clamp_min(0).long().unsqueeze(1)).squeeze(1).long()
        ctx = ctx_len[batch_to_slot[:batch]]
        now = (ctx + num_acc.long()).clamp_(max=max_ctx)
        qpos = now.unsqueeze(1) + torch.arange(block, dtype=torch.long, device=dev).unsqueeze(0)
        cpos = ctx.unsqueeze(1) + torch.arange(drafts + 1, dtype=torch.long, device=dev).unsqueeze(
            0
        )
        noise = mask_row.expand(batch, block, -1).clone()
        noise[:, 0, :] = F.embedding(bonus, embed)
        return [accepted, num_acc, rewind, bonus, qpos, cpos, noise]

    def is_valid(
        self,
        logits: torch.Tensor,
        draft: torch.Tensor,
        block_off: torch.Tensor,
        pool_idx: int,
        divisor: int,
        block_counts: torch.Tensor,
        block_tables: torch.Tensor,
        prev_acc: torch.Tensor,
        state_idx: torch.Tensor,
        dummy: torch.Tensor,
        kv_lens: torch.Tensor,
        batch_to_slot: torch.Tensor,
        ctx_len: torch.Tensor,
        max_ctx: int,
        embed: torch.Tensor,
        mask_row: torch.Tensor,
        rng_pool: torch.Tensor,
        rng_counter: torch.Tensor,
        force: float,
        block: int,
        ws_uc: Optional[torch.Tensor] = None,
        ws_mc: Optional[torch.Tensor] = None,
        ws_flags: Optional[torch.Tensor] = None,
        rank: int = 0,
        slots: int = 1,
        push_copies: int = 1,
        shard_offset: int = 0,
    ) -> None:
        batch, drafts = draft.shape
        assert 0 <= pool_idx < block_off.shape[0], "pool_idx must name a pool of block_off"
        assert block_off.shape[1] >= batch, "block_off must hold a row per request"
        assert divisor > 0, "divisor must be positive"
        for name, t in (
            ("block_counts", block_counts),
            ("block_tables", block_tables),
            ("state_idx", state_idx),
            ("dummy", dummy),
            ("kv_lens", kv_lens),
            ("batch_to_slot", batch_to_slot),
        ):
            assert t.shape[0] >= batch, f"{name} must hold a row per request"
        state = state_idx[:batch].long()
        assert bool(((state >= 0) & (state < prev_acc.numel())).all()), (
            "state_idx must name slots of prev_acc"
        )
        written = state[~dummy[:batch].bool()]
        assert int((state.unsqueeze(1) == written.unsqueeze(0)).sum()) == written.numel(), (
            "a KDA slot a live request writes must not be named by another request"
        )
        ctx_slots = batch_to_slot[:batch]
        assert bool(((ctx_slots >= 0) & (ctx_slots < ctx_len.numel())).all()), (
            "batch_to_slot must name slots of ctx_len"
        )
        world = slots // push_copies if slots > 1 else 1
        assert embed.shape[0] >= logits.shape[1] * world, "embed must cover the whole vocabulary"
        if slots > 1:
            assert shard_offset == rank * logits.shape[1], (
                "shard_offset must be this rank's first column"
            )
        mode, _, _ = _k3_spec_accept_op.force_mode(force, drafts)
        if _k3_spec_accept_op.force_draws_from_pool(mode):
            assert rng_pool.numel() == _RNG_POOL_SIZE, (
                "a fractional force draws from the full RNG pool"
            )
            assert rng_counter.numel() >= 1, "a fractional force advances rng_counter"


k3_spec_accept = K3SpecAccept()

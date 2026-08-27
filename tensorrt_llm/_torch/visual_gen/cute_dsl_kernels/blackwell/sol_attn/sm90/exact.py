"""Exact-block stream for the Hopper mainloop."""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr

from sol_attn.common import selector


@cute.jit
def _consume_exact_block(
    n_block: Int32,
    next_n_block: Int32,
    seqlen,
    producer_state,
    consumer_state,
    load_next: Callable,
    pipeline_k,
    issue_load,
    mma_pv: Callable,
    mma_one_n_block: Callable,
    mask: Callable,
    score_mod: Optional[Callable],
    accumulate,
    mask_seqlen: cutlass.Constexpr[bool],
    is_first: cutlass.Constexpr[bool],
    last_n_block: Int32,
    mask_last_only: cutlass.Constexpr[bool],
):
    if const_expr(mask_seqlen):
        next_state = mma_one_n_block(
            consumer_state,
            n_block=n_block,
            seqlen=seqlen,
            mma_pv_fn=partial(mma_pv, zero_init=not accumulate),
            mask_fn=partial(mask, mask_mod=None, mask_seqlen=True),
            score_mod_fn=score_mod,
            is_first_n_block=is_first,
            prefetch_next=True,
            next_n_block=next_n_block,
            kv_producer_state=producer_state,
            load_K=load_next,
            issue_load=issue_load,
        )
    elif const_expr(mask_last_only):
        next_state = mma_one_n_block(
            consumer_state,
            n_block=n_block,
            seqlen=seqlen,
            mma_pv_fn=partial(mma_pv, zero_init=not accumulate),
            mask_fn=partial(mask, mask_mod=None, mask_seqlen=False),
            last_block_mask_fn=partial(mask, mask_mod=None, mask_seqlen=True),
            last_n_block=last_n_block,
            score_mod_fn=score_mod,
            is_first_n_block=is_first,
            prefetch_next=True,
            next_n_block=next_n_block,
            kv_producer_state=producer_state,
            load_K=load_next,
            issue_load=issue_load,
        )
    else:
        next_state = mma_one_n_block(
            consumer_state,
            n_block=n_block,
            seqlen=seqlen,
            mma_pv_fn=partial(mma_pv, zero_init=not accumulate),
            mask_fn=partial(mask, mask_mod=None, mask_seqlen=False),
            score_mod_fn=score_mod,
            is_first_n_block=is_first,
            prefetch_next=True,
            next_n_block=next_n_block,
            kv_producer_state=producer_state,
            load_K=load_next,
            issue_load=issue_load,
        )
    return next_state


@cute.jit
def consume_exact_blocks(
    mask0: Int32,
    mask1: Int32,
    mask2: Int32,
    mask3: Int32,
    group_start: Int32,
    seqlen,
    producer_state,
    consumer_state,
    load_k: Callable,
    load_v: Callable,
    pipeline_k,
    pipeline_v,
    issue_load,
    mma_pv: Callable,
    mma_one_n_block: Callable,
    mask: Callable,
    score_mod: Optional[Callable],
    accumulate,
    scheduler_sync: Callable,
    scheduler_arrive: Callable,
    mask_first: cutlass.Constexpr[bool] = True,
    first_is_first: cutlass.Constexpr[bool] = True,
    group_words: cutlass.Constexpr[int] = 2,
    last_n_block: Int32 = Int32(-1),
    mask_last_only: cutlass.Constexpr[bool] = False,
    first_n_block: Int32 = Int32(0),
    has_exact=False,
    next_route_tile: Int32 = Int32(-1),
    load_next_route: Optional[Callable] = None,
):
    """Consume selected blocks while overlapping K loads with softmax and PV."""

    processed = False
    if has_exact:
        first = True
        pending_n_block = first_n_block

        for word in cutlass.range_constexpr(group_words):
            bits = selector.sol_attn_mask_word_constexpr(
                mask0, mask1, mask2, mask3, word
            )
            while bits != Int32(0):
                lowbit = bits & (Int32(0) - bits)
                next_n_block = (
                    group_start
                    + Int32(word * 32)
                    + selector.sol_attn_bfind_b32(lowbit)
                )

                if issue_load:
                    pipeline_v.producer_acquire(producer_state)
                    load_v(src_idx=pending_n_block, producer_state=producer_state)
                    producer_state.advance()

                if first:
                    scheduler_sync()
                    consumer_state = _consume_exact_block(
                        pending_n_block,
                        next_n_block,
                        seqlen,
                        producer_state,
                        consumer_state,
                        load_k,
                        pipeline_k,
                        issue_load,
                        mma_pv,
                        mma_one_n_block,
                        mask,
                        score_mod,
                        accumulate,
                        mask_first,
                        first_is_first,
                        last_n_block,
                        mask_last_only,
                    )
                else:
                    consumer_state = _consume_exact_block(
                        pending_n_block,
                        next_n_block,
                        seqlen,
                        producer_state,
                        consumer_state,
                        load_k,
                        pipeline_k,
                        issue_load,
                        mma_pv,
                        mma_one_n_block,
                        mask,
                        score_mod,
                        accumulate,
                        False,
                        False,
                        last_n_block,
                        mask_last_only,
                    )
                accumulate = True
                processed = True
                first = False
                pending_n_block = next_n_block
                bits = bits & (bits - Int32(1))

        if issue_load:
            pipeline_v.producer_acquire(producer_state)
            load_v(src_idx=pending_n_block, producer_state=producer_state)
            producer_state.advance()

        if first:
            scheduler_sync()
            consumer_state = _consume_exact_block(
                pending_n_block,
                next_route_tile,
                seqlen,
                producer_state,
                consumer_state,
                load_next_route,
                pipeline_k,
                issue_load,
                mma_pv,
                mma_one_n_block,
                mask,
                score_mod,
                accumulate,
                mask_first,
                first_is_first,
                last_n_block,
                mask_last_only,
            )
        else:
            consumer_state = _consume_exact_block(
                pending_n_block,
                next_route_tile,
                seqlen,
                producer_state,
                consumer_state,
                load_next_route,
                pipeline_k,
                issue_load,
                mma_pv,
                mma_one_n_block,
                mask,
                score_mod,
                accumulate,
                False,
                False,
                last_n_block,
                mask_last_only,
            )
        accumulate = True
        processed = True

    if processed:
        scheduler_arrive()

    return producer_state, consumer_state, accumulate, processed


__all__ = ["consume_exact_blocks"]

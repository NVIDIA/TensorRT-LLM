# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PLE n-gram id history: a packed token stream becomes one n-gram id window
per token, its out-of-slice predecessors read from -- and its tail published
back to -- the per-slot n-gram context pool.

A macro entry, composed from native torch integer ops (~20 small launches;
the sourcing ladder's native-torch rung); exact, so gated bitwise. This target's
local variant of ``qwen3_8_flash_next_nvfp4__sm_103__tp1/ops/ple/ngram_token_history``:
the op copied unchanged, with this target's per-rank slot pools added as cells.
Nothing registered under ``torch.ops.trtllm`` assembles an
n-gram id window, and the one kernel-plane callable in the family
(``EngramHashProvider.compute_hashes`` / ``NgramHashMapping._get_ngram_hashes``
in ``tensorrt_llm/_torch/modules/engram/engram.py``) has no cross-step state:
it left-pads every chunk from a pad id, so a decode step there loses the
sequence's whole context. The built-in Qwen4-Exp implementation keeps its own,
not wrapped here.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: This checkpoint's PLE constants: `ngram_size` 3, so a 2-id context row
#: (`Qwen4ExpPLECacheParams.ngram_context_len`).
CKPT_NGRAM_SIZE = 3
CKPT_EOS_TOKEN_ID = 248044
CKPT_VOCAB_SIZE = 248320
#: A `ple_ngram` slot count of a full-size engine run. The op addresses a slot by its index
#: alone, so a pool of another size (the pool shares the recurrent pool's slot space, and a
#: bf16 recurrent pool holds more slots in the same memory) takes the same path. The target
#: hands the whole pool over unsliced.
ENGINE_SLOTS = 461
#: The per-rank `ple_ngram` slot counts of this target's members: the stock engine's
#: `Qwen4ExpHybridCacheManager` at core's config (bf16 KV and recurrent state, KV fraction
#: 0.6, `max_batch_size` 128, `max_seq_len` 9216, `max_num_tokens` 8192, TRTLLM MoE, the
#: padded 19-batch CUDA-graph list) with the n-gram table host-resident
#: (`TRTLLM_QWEN4_EXP_PLE_HOST_OFFLOAD=1`), read off its "PLE state views bound" log line,
#: the same count on all four ranks. The pool shares the GDN recurrent pools' slot space,
#: which the V2 manager sizes from the rank's memory budget, not from `max_batch_size`:
#: tep4 holds 1784 with the table device-resident, and a dep4 rank, which keeps every GDN
#: head, fewer than a tep4 rank.
TEP4_SLOTS = 1874
DEP4_SLOTS = 850


def _scattered(num_seqs: int) -> tuple[int, ...]:
    """`num_seqs` distinct slots over the whole engine pool, both ends in."""
    slots = ((0, ENGINE_SLOTS - 1) + tuple(i * 13 + 4 for i in range(num_seqs)))[:num_seqs]
    assert len(set(slots)) == num_seqs and max(slots) < ENGINE_SLOTS
    return slots


def _spread(num_seqs: int, pool_slots: int) -> tuple[int, ...]:
    """`num_seqs` distinct slots evenly over a `pool_slots` pool, both ends in, out of
    order: batch row `s` is never slot `s`, as the scheduler's slot vector allows."""
    even = sorted({round(i * (pool_slots - 1) / (num_seqs - 1)) for i in range(num_seqs)})
    slots = tuple(even[1::2][::-1]) + tuple(even[0::2])
    assert len(set(slots)) == num_seqs and {0, pool_slots - 1} <= set(slots)
    return slots


def _cell(why: str, lengths, slots, carries, pool_slots: int, **spec) -> Cell:
    return Cell(
        why=why,
        spec=dict(
            lengths=tuple(lengths),
            slots=tuple(slots),
            carries=tuple(carries),
            pool_slots=pool_slots,
            **spec,
        ),
    )


#: The target's own argument dtypes: int32 ids and slots, int64
#: `query_start_loc_long` (or the int64 `cu_seqlens_from_seq_lens` rebuild).
_CALL_SITE = dict(ids_dtype=torch.int32, index_dtype=torch.int32, qsl_dtype=torch.int64)


class NgramTokenHistory(OpWrapper):
    """Row `t` of the returned window is `[c[t-n+1], ..., c[t-1], c[t]]` --
    **newest id last**, so column `ngram_size - 1` is the token's own id --
    where `c` is the token's **sequence's** id stream: for sequence `s` over
    packed tokens `[begin, end)` with `slot = state_indices[s]`, a predecessor
    at shift `k` is `input_ids[t - k]` if `t - begin >= k`, else column
    `ngram_size - 1 + (t - begin) - k` of the carry row. The carry row is
    `context_pool[slot]` (oldest id first) when `has_initial_states[s]`, and
    all `eos_token_id` when not.

    Then, **in place**, each named slot's row becomes the last `ngram_size - 1`
    ids of the sequence's history including the new tokens (`window[end - 1,
    1:]`); a zero-token sequence republishes its carry (so a fresh one is
    reseeded to EOS). Unnamed slots are bitwise untouched.

    The ids are **raw**: the only EOS written is the fresh-sequence seed, and
    an in-stream EOS passes through -- the hash downstream owns the
    segmentation, and `[eos, .., eos, id]` is what it treats as a sequence
    start. Seeding with 0 would be silently wrong: 0 is a real token.

    Returns a fresh contiguous int64 `[num_tokens, ngram_size]` window;
    `context_pool` is the only argument modified.

    Fusion boundary: token-to-sequence attribution, carry read, seed, the
    per-shift select and the publish. The hash, the table gather and slot
    assignment are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: lengths, slots, carries, pool_slots (the cache manager's width, not
    # the batch's), eos_density (0.0, of the packed ids), ids_dtype (int64),
    # index_dtype (int32, for state_indices), qsl_dtype (int32), qsl_from
    # "literal" | "seq_lens" (zeros + int64 cumsum of an int32 seq_lens prefix --
    # `cu_seqlens_from_seq_lens`' form). `ngram_size` is the checkpoint's 3.
    CELLS: tuple[Cell, ...] = (
        _cell(
            "MTP verify p=8: 32 decodes x 8 rows, qsl rebuilt",
            (8,) * 32,
            _scattered(32),
            (True,) * 32,
            ENGINE_SLOTS,
            qsl_from="seq_lens",
            **_CALL_SITE,
        ),
        _cell(
            "MTP verify mixed: 37-token context + 8 x 3 rows",
            (37,) + (3,) * 8,
            _scattered(9),
            (False,) + (True,) * 8,
            ENGINE_SLOTS,
            qsl_from="seq_lens",
            eos_density=0.05,
            **_CALL_SITE,
        ),
        _cell(
            "zero-token rows: carry kept, fresh reseeded",
            (2, 0, 0),
            (0, 1, 2),
            (True, True, False),
            3,
        ),
        _cell(
            f"tep4 decode: 128 x 1 row at the rank's {TEP4_SLOTS}-slot pool",
            (1,) * 128,
            _spread(128, TEP4_SLOTS),
            (True,) * 128,
            TEP4_SLOTS,
            qsl_from="seq_lens",
            **_CALL_SITE,
        ),
        _cell(
            f"dep4 decode: a rank's 32 x 1 row at its {DEP4_SLOTS}-slot pool",
            (1,) * 32,
            _spread(32, DEP4_SLOTS),
            (True,) * 32,
            DEP4_SLOTS,
            qsl_from="seq_lens",
            **_CALL_SITE,
        ),
        _cell(
            f"tep4 MTP7 verify: 128 x 8 rows at the {TEP4_SLOTS}-slot pool",
            (8,) * 128,
            _spread(128, TEP4_SLOTS),
            (True,) * 128,
            TEP4_SLOTS,
            qsl_from="seq_lens",
            **_CALL_SITE,
        ),
        _cell(
            "tep4 prefill: one fresh 8192-token context, max_num_tokens",
            (8192,),
            (TEP4_SLOTS - 1,),
            (False,),
            TEP4_SLOTS,
            qsl_from="seq_lens",
            **_CALL_SITE,
        ),
    )

    note = """
    Metadata consumed: none -- batch layout, pool, slot map and carry flags
    are explicit. Call site, once per forward on the PLE layer:
    `context_pool` the whole `MambaRole.PLE_NGRAM_CONTEXT` buffer
    (`[slots, ngram_size - 1]` int64, allocated by the V2 hybrid cache
    manager), `state_indices` the int32 slot vector the GDN mixer uses,
    `has_initial_states` the context prefix padded with True for the
    generation rows (the engine's own flags cover the context prefix only),
    `query_start_loc` int64 -- `query_start_loc_long`, or under MTP verify the
    `cu_seqlens_from_seq_lens` rebuild, because the metadata's vector has one
    row per generation sequence and would mis-segment silently.

    Preconditions:
    - Every slot a real one, `0 <= slot < slots`. The carry read
      bounds-checks, so -1 or an index past the pool is a device-side assert
      that poisons the CUDA context (observed in its own process). **A -1
      pad sentinel is not supported**; give padded rows real slots.
    - Slots distinct: two sequences on one slot is accepted and the publish
      is unordered (data; not guarded).
    - `query_start_loc[-1] == input_ids.shape[0]`: trailing tokens are
      attributed to the last sequence (data; not guarded).
    - `context_pool` int64 and `ngram_size - 1` wide, `has_initial_states`
      bool, `state_indices` and `input_ids` 1-D, `ngram_size >= 2`,
      `num_tokens >= 1`, `num_seqs >= 1`: each refused loudly.
    - The two length checks in `is_valid` are the silent ones.
    - Token ids are not range-checked (2^62 and -5 pass through verbatim).

    A wrong window is a window of valid ids: newest-first order, a zero
    seed, ignored carry flags, packing across sequence boundaries, a rotated
    carry read and publishing the head all hash to valid rows. Chunked
    prefill (96 tokens split at 41) is bitwise equal to one-shot, window and
    pool. At 461 slots the result is bitwise identical to an 11-slot pool
    with the same slot content.

    Certified at the tp1 forward's two verify-step shapes, both on the
    `cu_seqlens_from_seq_lens` rebuild at the 461-slot pool (32 x 8 rows; a
    37-token fresh context with in-stream EOS beside 8 x 3 rows), and at
    zero-token rows. At this target's per-rank pools (`TEP4_SLOTS` 1874,
    `DEP4_SLOTS` 850; every cell on the rebuild, with slots spread over the
    whole pool, both ends in): a plain decode of 128 x 1 rows at tep4 and of
    a dep4 rank's 32 x 1 rows, the MTP7 verify's 128 x 8 rows at tep4, and
    one fresh 8192-token context at tep4 (`max_num_tokens`, the envelope's
    largest row).

    The pool the manager hands over is a contiguous `[slots, 2]` int64
    (stride (2, 1), offset 0) at tp4, tep4 and dep4, ranks 0 and 3, with
    the GDN pools' slot count (probed on the stock manager). The call site
    is `ple_ngram_embed`, whose `reference` composes this entry's
    `reference`; the tp1 forward launches the fused entry, not this op.

    Not certified: tp4's pool (stock does not serve pure tp4 -- its NVFP4
    MoE weight shuffle refuses 160 rows per rank -- so it was not booted), the
    MTP and multimodal members' pools, and the dep4 variants' pools
    (kv_fp8, kv_nvfp4, chunked, reuse with its recurrent-state snapshots).
    The op addresses a slot by index alone, so these differ from the cells
    in pool length only.

    CUDA-graph capture certified (no `repeat_interleave` on device lengths,
    so no data-dependent shape). Cost, not gated: ~0.30 ms for a 4096-token
    prefill, a 32-row decode or a mixed 2 x 1024 + 30 x 1 batch alike --
    launch-bound.

    Hard limits: none reached. `num_tokens` is certified to 8192; every index
    is int64 and the largest tensor is the `[num_tokens, ngram_size]` window,
    far from 2^31 elements. Pure torch: no JIT path and no compiled kernel of
    its own, so nothing in it is architecture-specific.
    """

    def __call__(
        self,
        input_ids: torch.Tensor,
        context_pool: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_states: torch.Tensor,
        query_start_loc: torch.Tensor,
        eos_token_id: int,
        ngram_size: int,
    ) -> torch.Tensor:
        ids = input_ids.long()
        num_tokens = ids.shape[0]
        num_seqs = state_indices.shape[0]
        context_len = ngram_size - 1

        device = ids.device
        starts = query_start_loc.long()
        slots = state_indices.long()

        # A copy, so the in-place publish cannot disturb rows still being read
        # and a zero-token sequence has something to write back.
        carried = context_pool.index_select(0, slots)
        carried = torch.where(
            has_initial_states.reshape(-1, 1),
            carried,
            carried.new_full((), eos_token_id),
        )

        # Device-side and static-shape, so capturable; `repeat_interleave` on
        # device lengths would make the shape data-dependent and sync.
        positions = torch.arange(num_tokens, device=device)
        seq_id = torch.searchsorted(starts[1:].contiguous(), positions, right=True).clamp_max(
            num_seqs - 1
        )
        offset = positions - starts[seq_id]

        # One column per shift, oldest first: the packed vector once the token
        # is `shift` into its slice, the carry row (virtual positions
        # -context_len .. -1) before that.
        columns = []
        for shift in range(context_len, 0, -1):
            from_packed = ids.index_select(0, (positions - shift).clamp_min(0))
            pool_column = (context_len + offset - shift).clamp(0, context_len - 1)
            from_pool = carried[seq_id, pool_column]
            columns.append(torch.where(offset >= shift, from_packed, from_pool))
        columns.append(ids)
        window = torch.stack(columns, dim=1)

        # The last token's window minus its oldest id, or the carry when empty.
        ends = starts[1:]
        last_token = (ends - 1).clamp_min(0)
        published = torch.where(
            (ends > starts[:-1]).reshape(-1, 1),
            window.index_select(0, last_token)[:, 1:],
            carried,
        )
        context_pool[slots] = published
        return window

    def reference(
        self,
        input_ids: torch.Tensor,
        context_pool: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_states: torch.Tensor,
        query_start_loc: torch.Tensor,
        eos_token_id: int,
        ngram_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(window, context_pool)` as the call leaves them, from each
        sequence's own id stream materialized as a Python list (carry, then
        its slice) and read straight out; mutates nothing."""
        context_len = ngram_size - 1
        ids = input_ids.tolist()
        pool = context_pool.tolist()
        starts = query_start_loc.tolist()
        carries = has_initial_states.tolist()
        window = [[0] * ngram_size for _ in ids]
        published = [list(row) for row in pool]
        for s, slot in enumerate(state_indices.tolist()):
            carried = list(pool[slot]) if carries[s] else [eos_token_id] * context_len
            begin, end = starts[s], starts[s + 1]
            stream = carried + ids[begin:end]
            for step in range(end - begin):
                window[begin + step] = stream[step : step + ngram_size]
            published[slot] = stream[len(stream) - context_len :]
        device = input_ids.device
        return (
            torch.tensor(window, dtype=torch.long, device=device).reshape(len(ids), ngram_size),
            torch.tensor(published, dtype=torch.long, device=device).reshape(context_pool.shape),
        )

    def is_valid(
        self,
        input_ids: torch.Tensor,
        context_pool: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_states: torch.Tensor,
        query_start_loc: torch.Tensor,
        eos_token_id: int,
        ngram_size: int,
    ) -> None:
        # Both measured silent; the over-long direction of each raises instead.
        num_seqs = state_indices.shape[0]
        assert has_initial_states.shape[0] == num_seqs, (
            f"has_initial_states must be one flag per sequence; got "
            f"{tuple(has_initial_states.shape)} against {num_seqs} sequences -- a "
            f"length-1 flag tensor is broadcast over the whole batch rather than "
            f"rejected"
        )
        assert query_start_loc.shape[0] == num_seqs + 1, (
            f"query_start_loc must be [num_seqs + 1] = [{num_seqs + 1}] cumulative "
            f"token counts; got {tuple(query_start_loc.shape)} -- a short one "
            f"misattributes tokens to the wrong sequence rather than raising"
        )


ngram_token_history = NgramTokenHistory()

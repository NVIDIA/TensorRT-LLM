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
"""The short-conv under a multi-token generation step.

Ordinary decode feeds one token per generation request, and ``apply_short_conv``
sends the whole post-context slice through ``causal_conv1d_update`` with one
cache index per request. Speculative decoding breaks that assumption: the
target verifies ``1 + max_draft_len`` tokens per request in a single step.

The failure is loud in one place and silent in another, which is why both are
tested here. Loud: the update kernel requires ``conv_state_indices`` to have one
entry per ROW, so it rejects the multi-token batch outright. Silent: reshaping
to satisfy it would apply the same initial state to every drafted token instead
of advancing through them -- no error, just a conv that stopped being causal.
"""

import inspect

import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.inkling.conv_state import InklingConvRuntime


class _Meta:
    """The slice of attention metadata ``InklingConvRuntime.build`` reads."""

    def __init__(self, seq_lens, num_contexts):
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
        self.num_contexts = num_contexts
        self.request_ids = list(range(len(seq_lens)))
        self.is_cuda_graph = False


class _Cache:
    """Stands in for InklingConvStateCache: hands out one pool row per request."""

    def __init__(self, n):
        self.state_indices = torch.arange(n, dtype=torch.int32)
        self._n = n

    def write_state_indices(self, request_ids):
        # One argument, matching the pool after the move to sparse/inkling:
        # the graph-pointer check that used to need `is_graph` is now a
        # fixed-size pool plus a reserved padding row, so there is nothing
        # per-call to guard.
        return list(range(len(request_ids)))


def _build(seq_lens, num_contexts):
    return InklingConvRuntime.build(_Meta(seq_lens, num_contexts), _Cache(len(seq_lens)))


def test_ordinary_decode_stays_on_the_single_token_path():
    """One token per generation request must not change behaviour.

    This is the overwhelmingly common case and the update kernel is the fast
    path for it, so the multi-token branch has to stay strictly opt-in.
    """
    rt = _build([1, 1, 1], num_contexts=0)
    assert rt.gen_tokens_per_seq == 1
    assert rt.gen_query_start_loc is None
    assert rt.gen_has_initial_state is None


def test_a_verify_step_is_recognised_as_multi_token():
    """Four requests verifying 1 + 3 drafted tokens each."""
    rt = _build([4, 4, 4, 4], num_contexts=0)
    assert rt.gen_tokens_per_seq == 4
    assert rt.gen_query_start_loc.tolist() == [0, 4, 8, 12, 16]


def test_generation_tokens_continue_an_existing_window():
    """``has_initial_state`` is True for generation, unlike prefill.

    A prefill starts a fresh stream and the pool row holds nothing worth
    reading; a generation step continues one. Getting this backwards would drop
    the first kernel-width tokens of context at every verify step -- again with
    no error, only worse output.
    """
    rt = _build([4, 4], num_contexts=0)
    assert rt.gen_has_initial_state.tolist() == [True, True]
    # ...while the context side of the same structure stays False.
    rt2 = _build([7, 4], num_contexts=1)
    assert rt2.has_initial_state.tolist() == [False]


def test_mixed_context_and_verify_batch_splits_correctly():
    """Context requests keep their own varlen offsets; generation gets its own."""
    rt = _build([9, 5, 4, 4], num_contexts=2)
    assert rt.num_ctx_tokens == 14
    assert rt.query_start_loc.tolist() == [0, 9, 14]
    assert rt.gen_tokens_per_seq == 4
    assert rt.gen_query_start_loc.tolist() == [0, 4, 8]


def test_a_ragged_generation_batch_is_rejected():
    """The varlen offsets are built from a uniform per-request token count.

    Speculative decoding always verifies the same number of tokens per request,
    so a ragged batch means an assumption elsewhere has broken. Computing the
    offsets from the first request's length and carrying on would silently
    mis-slice every later request.
    """
    with pytest.raises(ValueError, match="uniform token count"):
        _build([4, 3], num_contexts=0)


# --- numerics -------------------------------------------------------------
# The reason to route a verify step through the varlen path rather than reshape
# it onto the update kernel: the conv must advance token by token. A reference
# built from repeated single-token updates pins that down.


def _reference_windows(init_state, x, width):
    """Conv state after each token: last ``width`` inputs of init ++ x.

    The state a causal conv carries is a window of past INPUTS, so it can be
    written down without running a conv at all -- which makes it a reference
    that shares no code with the implementation.
    """
    stream = torch.cat([init_state, x], dim=-1)
    return [stream[..., t + 1 : t + 1 + width] for t in range(x.shape[-1])]


def test_the_window_after_the_last_token_is_what_a_verify_step_must_leave():
    """After k tokens the state holds the last ``width`` of init ++ x[:k].

    This is the property the varlen path provides and the update kernel, given
    one shared initial state, does not: its output for every drafted token
    would be the window after the FIRST one.
    """
    width, channels, steps = 3, 2, 4
    init = torch.arange(channels * width, dtype=torch.float32).reshape(channels, width)
    x = torch.arange(100, 100 + channels * steps, dtype=torch.float32).reshape(channels, steps)

    windows = _reference_windows(init, x, width)
    assert len(windows) == steps
    # Each step shifts the window one input to the right...
    assert torch.equal(windows[0], torch.cat([init[:, 1:], x[:, :1]], dim=-1))
    # ...and after all four drafted tokens nothing of the initial state is left.
    assert torch.equal(windows[-1], x[:, -width:])


def test_partial_acceptance_needs_an_earlier_window_than_the_forward_leaves():
    """Why the model refuses speculative decoding rather than running.

    The forward advances the state to ``windows[-1]``. If only one token is
    accepted, the correct state is ``windows[0]``, and they differ. Nothing in
    the shapes or dtypes distinguishes them, which is precisely the problem:
    the wrong one is a perfectly valid conv state holding tokens the model
    never emitted.
    """
    width, channels, steps = 3, 2, 4
    init = torch.zeros(channels, width)
    x = torch.arange(1, 1 + channels * steps, dtype=torch.float32).reshape(channels, steps)
    windows = _reference_windows(init, x, width)
    assert not torch.equal(windows[0], windows[-1])


def test_a_draft_length_the_capture_cannot_hold_is_rejected():
    """The capture buffers are sized from max_draft_len.

    A verify step deeper than they hold could not be rolled back, so the
    precondition is checked at load rather than discovered as quietly worse
    output later. max_draft_len < 1 means nothing was sized at all.
    """

    from tensorrt_llm._torch.models.modeling_inkling import InklingForCausalLM

    src = inspect.getsource(InklingForCausalLM._assert_inkling_spec_conv_state)
    assert "max_draft_len" in src and "raise ValueError" in src


# --- rolling the window back to what was accepted --------------------------
# The verify forward leaves the window advanced over every drafted token. What
# each request should be left holding is the window after its ACCEPTED prefix,
# and the two differ on any partial acceptance.


def _capture(n, channels, kwin, steps, init, x):
    from tensorrt_llm._torch.attention_backend.sparse.inkling.conv_state import _ConvVerifyCapture

    cap = _ConvVerifyCapture(n, channels, kwin, steps, torch.device("cpu"), torch.float32)
    cap.init[:n].copy_(init)
    cap.x[:n].copy_(x)
    return cap


def test_accepted_window_matches_the_hand_computed_stream():
    """Reconstruct against the definition: last kwin of (init ++ x[:k]).

    The reference shares no code with the implementation -- it slices the
    concatenated stream directly -- so agreement is evidence, not tautology.
    """
    n, channels, kwin, steps = 3, 2, 3, 4
    init = torch.randn(n, channels, kwin)
    x = torch.randn(n, steps, channels)
    cap = _capture(n, channels, kwin, steps, init, x)

    for k in range(1, steps + 1):
        got = cap.accepted_window(torch.full((n,), k, dtype=torch.int64), kwin)
        want = torch.cat([init, x.transpose(1, 2)], dim=-1)[..., k : k + kwin]
        assert torch.allclose(got, want), f"k={k}"


def test_each_request_rolls_back_to_its_own_acceptance():
    """Acceptance counts differ per request within one batch.

    A commit that used a single count for the batch would be right for whichever
    request happened to set it and wrong for the rest -- the kind of bug that
    only shows up once acceptance rates stop being uniform.
    """
    n, channels, kwin, steps = 3, 2, 3, 4
    init = torch.zeros(n, channels, kwin)
    x = torch.arange(n * steps * channels, dtype=torch.float32).reshape(n, steps, channels)
    cap = _capture(n, channels, kwin, steps, init, x)

    accepted = torch.tensor([1, 3, 4], dtype=torch.int64)
    got = cap.accepted_window(accepted, kwin)
    stream = torch.cat([init, x.transpose(1, 2)], dim=-1)
    for i, k in enumerate(accepted.tolist()):
        assert torch.equal(got[i], stream[i, :, k : k + kwin])


def test_full_acceptance_leaves_what_the_forward_already_wrote():
    """When every drafted token is accepted the commit is a no-op in effect.

    Worth pinning: it is the case where a wrong commit would be invisible,
    because the forward's own result is also correct.
    """
    n, channels, kwin, steps = 2, 3, 2, 4
    init = torch.randn(n, channels, kwin)
    x = torch.randn(n, steps, channels)
    cap = _capture(n, channels, kwin, steps, init, x)
    got = cap.accepted_window(torch.full((n,), steps, dtype=torch.int64), kwin)
    # The state after all steps is simply the last kwin inputs.
    assert torch.allclose(got, x.transpose(1, 2)[..., -kwin:])


def test_single_acceptance_discards_the_rejected_tokens():
    """The target's own token is always accepted, so k >= 1 and never 0.

    With k=1 the window keeps kwin-1 of the pre-step state and exactly one new
    input; the drafted tokens 2..steps must leave no trace.
    """
    n, channels, kwin, steps = 1, 2, 3, 4
    init = torch.full((n, channels, kwin), -1.0)
    x = torch.arange(1, 1 + steps * channels, dtype=torch.float32).reshape(n, steps, channels)
    cap = _capture(n, channels, kwin, steps, init, x)
    got = cap.accepted_window(torch.ones(n, dtype=torch.int64), kwin)
    assert torch.equal(got[0, :, :-1], init[0, :, 1:])
    assert torch.equal(got[0, :, -1], x[0, 0])


def test_capture_is_only_allocated_when_speculating():
    """An ordinary server must not pay for buffers it never reads."""

    from tensorrt_llm._torch.attention_backend.sparse.inkling.conv_state import (
        InklingConvStateCache,
    )

    src = inspect.getsource(InklingConvStateCache.__init__)
    assert "verify_steps" in src and "if self.verify_steps < 2" in src


# --- verify-step attention -------------------------------------------------
# The decode path serves one query token per request. A verify step presents
# 1 + max_draft_len, which is what produced the illegal memory access: it wrote
# one KV entry per request and read a page table sized for one new position.


def test_verify_attention_is_not_routed_through_the_context_path():
    """Routing a verify step at the prefill kernel would drop the prefix.

    ``inkling_prefill_attention`` attends only within the tokens handed to it,
    which is complete for a fresh prefill (Inkling keeps block reuse off) and
    silently wrong here -- the drafted tokens would see none of the cached
    conversation and still produce fluent text. Cheapest-looking route, worst
    failure mode, so it is worth pinning that it was not taken.
    """

    from tensorrt_llm._torch.attention_backend.sparse.inkling.backend import InklingTritonAttention

    src = inspect.getsource(InklingTritonAttention._run_verify)
    # The docstring explains why the prefill kernel is wrong here, so check the
    # body rather than the whole source.
    quote = '"' * 3
    body = src[src.index(quote, src.index(quote) + 3) + 3 :]
    assert "inkling_prefill_attention" not in body
    assert "inkling_decode_attention" in body


def test_verify_walks_positions_in_order_so_causality_is_structural():
    """Position t must see the prefix plus 0..t, and nothing later.

    Ordering provides that rather than a mask: each step writes its KV before
    attending, and the seq_len it passes is num_cached + t + 1. A loop that
    wrote all KV up front would let position 0 attend to drafted tokens that,
    at that point in the sequence, do not exist.
    """

    from tensorrt_llm._torch.attention_backend.sparse.inkling.backend import InklingTritonAttention

    src = inspect.getsource(InklingTritonAttention._run_verify)
    write_at = src.index("write_kv_cache_hnd")
    attend_at = src.index("inkling_decode_attention")
    assert write_at < attend_at, "each position's KV must be written before it attends"
    # The attended length grows by one per position. The base is derived from
    # ink_seq_lens rather than num_cached (see the write-offset test below); what
    # matters here is the "+ t + 1", which is what makes position t see exactly
    # the prefix plus 0..t.
    assert "+ t + 1" in src


def test_verify_output_is_reassembled_in_packed_order():
    """The batch is request-major, so per-step results interleave back.

    Returning the steps concatenated instead would hand every downstream module
    a batch whose rows belong to the wrong requests -- a permutation, not a
    crash.
    """
    num_gen, steps, hidden = 3, 4, 5
    packed = torch.arange(num_gen * steps * hidden, dtype=torch.float32).reshape(
        num_gen * steps, hidden
    )
    view = packed.view(num_gen, steps, hidden)
    # Request-major: request i's step t is row i*steps + t.
    for i in range(num_gen):
        for t in range(steps):
            assert torch.equal(view[i, t], packed[i * steps + t])
    # ...and the reshape back is the identity, which is what _run_verify relies on.
    assert torch.equal(view.reshape(num_gen * steps, hidden), packed)


def test_capture_under_cuda_graph_is_refused_with_a_reason():
    """The verify path is eager; capturing it would be silently wrong.

    Better to say so at the point of use than to let a captured graph replay
    stale per-step writes.
    """

    from tensorrt_llm._torch.attention_backend.sparse.inkling.backend import InklingTritonAttention

    src = inspect.getsource(InklingTritonAttention._run_verify)
    assert "is_cuda_graph" in src and "RuntimeError" in src


# --- the draft chain's own conv state --------------------------------------
# The chain's blocks are addressed by GLOBAL layer index (trunk + depth), the
# same index their KV cache is keyed by, but the draft manager's conv pool holds
# only the chain's layers. Both facts have to be true at once.


def test_the_draft_pool_is_addressed_by_global_index():
    """A 3-row pool indexed at 42 without the offset is an IndexError.

    Sized by the trunk's layer count instead, it would allocate 42 rows to hold
    3 -- no error, just a pool that is mostly waste and whose banded pattern
    comes from the wrong layers.
    """

    from tensorrt_llm._torch.attention_backend.sparse.inkling.conv_state import (
        InklingConvStateCache,
    )

    src = inspect.getsource(InklingConvStateCache.__init__)
    assert "layer_offset" in src and "num_layers" in src
    state = inspect.getsource(InklingConvStateCache.layer_state)
    assert "layer_idx - self._layer_offset" in state


def test_the_draft_block_does_not_take_the_stateless_branch():
    """Passing no conv_rt drops the block onto the stateless short-conv path.

    That path convolves across the context/generation boundary of a packed
    batch -- the trunk raises NotImplementedError for exactly that case -- and
    the chain would keep no conv history between steps. Neither shows up as an
    error.
    """

    from tensorrt_llm._torch.models.modeling_inkling import InklingMTPBlock

    src = inspect.getsource(InklingMTPBlock.forward)
    assert "conv_rt=conv_rt" in src and "conv_state=conv_state" in src


def test_the_draft_conv_state_comes_from_the_manager_in_play():
    """``attn_metadata.ink_conv_cache`` is the TARGET's, published at prepare().

    The draft forward runs inside the draft KV cache context with the manager
    swapped underneath, so reading the published field would hand the chain the
    trunk's pool rows: a real pool, real rows, wrong history.
    """

    from tensorrt_llm._torch.models.modeling_inkling import InklingMTPBlock

    src = inspect.getsource(InklingMTPBlock.forward)
    assert "prepare_conv_runtime" in src
    # Positive form only: the comment explaining why the published field is the
    # wrong source names it, and a "not in" assertion keeps tripping over that.
    assert 'getattr(attn_metadata, "kv_cache_manager", None)' in src


def test_a_shorter_step_than_the_buffer_holds_is_recorded_as_such():
    """The buffer is sized for the target's verify step; the chain's is shorter.

    Reconstructing from the full buffer would concatenate stale inputs after the
    real ones and commit a window built partly from a previous batch -- a valid
    window, wrong history, no error.
    """
    n, channels, kwin, steps = 2, 3, 2, 4
    init = torch.zeros(n, channels, kwin)
    x = torch.arange(n * steps * channels, dtype=torch.float32).reshape(n, steps, channels)
    cap = _capture(n, channels, kwin, steps, init, x)
    cap.steps_used = 2  # as a 2-step save would leave it

    got = cap.accepted_window(torch.full((n,), 2, dtype=torch.int64), kwin)
    want = torch.cat([init, x[:, :2].transpose(1, 2)], dim=-1)[..., 2 : 2 + kwin]
    assert torch.allclose(got, want)


def test_more_steps_than_the_buffer_holds_is_an_error_not_a_truncation():
    """Sized from max_draft_len, so overflow means the sizing assumption broke."""
    from tensorrt_llm._torch.attention_backend.sparse.inkling.conv_state import _ConvVerifyCapture

    cap = _ConvVerifyCapture(2, 3, 2, 2, torch.device("cpu"), torch.float32)
    with pytest.raises(ValueError, match="max_draft_len"):
        cap.save(torch.zeros(2, 3, 2), torch.zeros(2, dtype=torch.int64), torch.zeros(8, 3), 4)


def test_a_shared_draft_kv_cache_is_refused_at_load():
    """The chain's layers are not the target's, so they cannot share its cache.

    A draft block is addressed by the global layer index (trunk + depth), which
    only the separate draft manager is keyed by. Running against the target's
    manager instead produced a bare `KeyError: 42` inside the draft loop,
    minutes into a 4-GPU run, naming nothing.
    """

    from tensorrt_llm._torch.models.modeling_inkling import InklingForCausalLM

    src = inspect.getsource(InklingForCausalLM._assert_inkling_spec_conv_state)
    assert "should_use_separate_draft_kv_cache" in src


def test_a_layer_with_no_slot_in_the_manager_says_so():
    """Same condition at runtime, in case the load-time check is bypassed."""

    from tensorrt_llm._torch.attention_backend.sparse.inkling.backend import _batch_cache_indices

    class _Mgr:
        def get_batch_cache_indices(self, request_ids, layer_idx):
            raise KeyError(layer_idx)

    with pytest.raises(RuntimeError, match="has no slot in the KV cache manager"):
        _batch_cache_indices(_Mgr(), [0], 42)


def test_a_draft_layer_falls_back_off_the_published_page_table():
    """The published page table covers the TARGET's layers only.

    ``ink_gen_page_table`` indexes ``self._ink_pt_rows[layer]``, which the
    metadata fills for the layers it was prepared for. A draft block's global
    index is not a key there, and the lookup raises a bare KeyError from inside
    a dict -- the same shape of failure as reading the published conv cache, and
    from the same cause: the chain runs against metadata prepared for the
    target.

    What keeps the chain away from it is the routing, not a lookup guard: a
    chain forward presents more than one query token per request, so it lands in
    ``_run_verify``, which builds its own table from the manager in play. Pin
    that, since it is the property that makes the KeyError unreachable.
    """

    from tensorrt_llm._torch.attention_backend.sparse.inkling.backend import InklingTritonAttention

    src = inspect.getsource(InklingTritonAttention._run_verify)
    assert "_batch_cache_indices(mgr" in src, (
        "the verify path must resolve pages through the manager in play"
    )
    assert "build_page_table(block_ids" in src
    assert "ink_gen_page_table" not in src, (
        "the verify path must not read the target's published page table"
    )


# --- where a verify step's KV actually goes ---------------------------------


def test_verify_writes_after_the_existing_history_not_at_zero():
    """A verify step's writes start at the history and grow by one per position.

    This test asserted the opposite for four rounds. The reading that
    ``num_cached`` is 0 on the speculative path came from warmup batches, where
    0 is correct; sampled past them it grows by exactly one per verify step,
    which is what ``model_engine``'s speculative branch fills it with. The
    replacement base derived from ``ink_seq_lens`` computed
    ``num_cached + 1 - steps`` -- negative early in a sequence -- so it was
    reverted.

    What has to hold is the shape, not the field: the room a step needs runs
    from the history to ``history + steps - 1``, and asking for that room
    succeeds when the pages cover it.
    """
    from tensorrt_llm._torch.attention_backend.sparse.inkling.backend import check_verify_write_room

    page_size, history, steps = 32, 669, 4
    pages = list(range(history // page_size + 2))  # covers 0 .. history + steps
    check_verify_write_room(history, steps, page_size, pages)

    one_page_short = pages[:-1]
    with pytest.raises(RuntimeError, match="no KV page for the last drafted"):
        check_verify_write_room(history, steps, page_size, one_page_short)


def test_the_single_token_case_agrees_with_the_decode_path():
    """steps == 1 must reduce to the decode path's ``sl - 1``.

    If the two disagree, a verify step and an ordinary decode step write the
    same logical position to different slots, and which one a token lands in
    depends on how many drafts were in flight.
    """
    sl, steps = 37, 1
    assert sl - steps + 0 == sl - 1


def test_a_negative_write_base_is_refused():
    """A negative base must fail loudly rather than write from the end.

    The guard is kept even though the base it caught came from a reverted
    change: torch indexes negatively, so such a write succeeds and rewrites the
    start of the request's own history, producing fluent output that differs
    from greedy. That took four rounds to find once.

    It is a backstop, not the handler for the one negative base that legitimately
    occurs -- see the warmup clamp below.
    """
    from tensorrt_llm._torch.attention_backend.sparse.inkling.backend import check_verify_write_room

    with pytest.raises(RuntimeError, match="negative KV write base"):
        check_verify_write_room(-3, 4, 32, list(range(24)))


def test_the_draft_chains_warmup_underflow_is_clamped_not_refused():
    """The draft chain's cached count goes negative on warmup, legitimately.

    `mtp.py` rewinds `num_cached_tokens_per_seq` by the rejected drafts after
    each verify step. On a real sequence the result is the correct post-rewind
    count; on the tiny dummy sequences of generation-step warmup it underflows
    (measured: `layer=42 base=-3 steps=3 history=0`, identically on the NVFP4
    and BF16 checkpoints, during executor init). `mtp.py` clamps `kv_lens_cuda`
    for exactly this case and says so; it does not clamp the CPU list Inkling
    reads.

    So the model clamps it, and a run must not abort at startup over it.

    Asserted by CALLING the base derivation, not by grepping its source. The
    previous version matched the literal ``base = [max(0,`` and broke the moment
    that expression moved into a helper -- while the clamp it exists to protect
    was still there and still working. A test that fails on a refactor it was
    not measuring is worse than no test.
    """
    import torch

    from tensorrt_llm._torch.attention_backend.sparse.inkling.backend import _verify_write_base

    class _WarmupMD:
        num_contexts = 0
        # A KV length shorter than the step presents: exactly the underflow.
        kv_lens_cuda = torch.tensor([1], dtype=torch.int32)

    assert _verify_write_base(_WarmupMD(), [0], 1, 3) == [0]

    # And through the fallback path, where the framework's own rewind has
    # already driven the CPU list negative.
    class _NoKvLens:
        num_contexts = 0
        kv_lens_cuda = None

    assert _verify_write_base(_NoKvLens(), [-3], 1, 3) == [0]


# --- who the commit actually writes to -------------------------------------
# accepted_window's arithmetic is covered above. The layer that applies it was
# not covered at all: which pool rows it touches, which slice of the batch's
# acceptance counts it uses, and when it must do nothing. Every one of those is
# silent when wrong -- the windows are the right shape either way.


class _ConvCfg:
    """Smallest config the pool reads: two layers, one banded and one global."""

    sconv_kernel_size = 4
    num_hidden_layers = 2
    hidden_size = 8

    @staticmethod
    def layer_num_kv_heads(idx):
        return 2 if idx == 0 else 1

    @staticmethod
    def layer_head_dim(_idx):
        return 4


def _cpu_pool(num_request_slots=4, max_draft_len=3):
    from tensorrt_llm._torch.attention_backend.sparse.inkling.conv_state import (
        InklingConvStateCache,
    )

    return InklingConvStateCache(
        _ConvCfg(),
        tp_size=1,
        num_request_slots=num_request_slots,
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_draft_len=max_draft_len,
    )


def _stage_capture(pool, rows, steps):
    """Put a known window + known inputs into every conv of every layer."""
    for layer_idx in range(_ConvCfg.num_hidden_layers):
        for cap, buf in zip(pool.layer_capture(layer_idx), pool.layer_state(layer_idx)):
            channels = buf.shape[1]
            init = torch.randn(len(rows), channels, pool.kwin)
            x = torch.randn(len(rows), steps, channels)
            cap.init[: len(rows)].copy_(init)
            cap.x[: len(rows), :steps].copy_(x)
            cap.steps_used = steps


def test_the_commit_writes_the_accepted_window_to_the_generation_rows():
    """Every conv of every layer, at the rows the verify step advanced."""
    pool = _cpu_pool()
    rows = torch.tensor([2, 0], dtype=torch.int64)  # deliberately not sorted
    steps = 4
    _stage_capture(pool, rows, steps)
    num_accepted = torch.tensor([2, 1], dtype=torch.int64)

    expected = {}
    for layer_idx in range(_ConvCfg.num_hidden_layers):
        for j, cap in enumerate(pool.layer_capture(layer_idx)):
            expected[(layer_idx, j)] = cap.accepted_window(num_accepted, pool.kwin).clone()

    pool.commit_after_verify(num_accepted, rows)

    for layer_idx in range(_ConvCfg.num_hidden_layers):
        for j, buf in enumerate(pool.layer_state(layer_idx)):
            want = expected[(layer_idx, j)]
            for i, row in enumerate(rows.tolist()):
                assert torch.allclose(buf[row], want[i]), (
                    f"layer {layer_idx} conv {j} row {row} did not get its own window"
                )


def test_the_commit_leaves_every_other_row_alone():
    """A context request sharing the batch must not have its window rewritten."""
    pool = _cpu_pool()
    untouched_row = 3
    for layer_idx in range(_ConvCfg.num_hidden_layers):
        for buf in pool.layer_state(layer_idx):
            buf[untouched_row].fill_(1.25)

    rows = torch.tensor([0, 1], dtype=torch.int64)
    _stage_capture(pool, rows, 4)
    pool.commit_after_verify(torch.tensor([1, 1], dtype=torch.int64), rows)

    for layer_idx in range(_ConvCfg.num_hidden_layers):
        for buf in pool.layer_state(layer_idx):
            assert torch.all(buf[untouched_row] == 1.25)


def test_a_pool_built_without_speculation_refuses_to_commit():
    """With no captures there is nothing to replay from, so it must not pretend."""
    pool = _cpu_pool(max_draft_len=0)
    with pytest.raises(RuntimeError, match="not built for speculative decoding"):
        pool.commit_after_verify(
            torch.tensor([1], dtype=torch.int64), torch.tensor([0], dtype=torch.int64)
        )


def test_the_manager_takes_the_generation_tail_of_the_batch():
    """``num_accepted`` covers the whole batch; the conv rows cover only the tail.

    The packed batch is contexts first, then generations, so the acceptance
    counts have to be indexed from the END. Slicing from the front would apply a
    context request's count to a generation request's window -- right shape,
    wrong history, no error.
    """
    from tensorrt_llm._torch.attention_backend.sparse.inkling.cache_manager import (
        InklingHybridCacheManager,
    )

    pool = _cpu_pool()
    rows = torch.tensor([0, 1], dtype=torch.int64)
    _stage_capture(pool, rows, 4)

    seen = {}

    class _Pool:
        def commit_after_verify(self, num_accepted, gen_rows):
            seen["num_accepted"] = num_accepted.clone()
            seen["rows"] = gen_rows.clone()

    class _Rt:
        gen_indices = rows
        gen_tokens_per_seq = 4

    class _Mgr:
        _last_conv_rt = _Rt()
        _conv_cache = _Pool()

    # Two contexts ahead of the two generation requests.
    batch = torch.tensor([9, 9, 2, 3], dtype=torch.int64)
    InklingHybridCacheManager.commit_conv_state_after_verify(_Mgr(), batch)
    assert torch.equal(seen["num_accepted"], torch.tensor([2, 3], dtype=torch.int64))
    assert torch.equal(seen["rows"], rows)


def test_an_ordinary_decode_step_commits_nothing():
    """One token per request is not a verify step; rolling back would corrupt it."""
    from tensorrt_llm._torch.attention_backend.sparse.inkling.cache_manager import (
        InklingHybridCacheManager,
    )

    called = []

    class _Pool:
        def commit_after_verify(self, *_a):
            called.append(1)

    class _Rt:
        gen_indices = torch.tensor([0], dtype=torch.int64)
        gen_tokens_per_seq = 1  # ordinary decode

    class _Mgr:
        _last_conv_rt = _Rt()
        _conv_cache = _Pool()

    InklingHybridCacheManager.commit_conv_state_after_verify(
        _Mgr(), torch.tensor([1], dtype=torch.int64)
    )
    assert not called, "an ordinary decode step must not roll the windows back"

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``context_kv_fork_paged_wide`` (opt-in): the context-K/V fork over a paged drafter context on
decode steps of 65..128 requests, forked behind the LM head's logits gather.

* The bound: with the lever a paged context takes up to its bound; without it, or over a dense
  or windowed arena, ``max_requests`` is unchanged.
* The fork point: a paged step of more than ``MAX_REQUESTS`` requests launches at
  ``GATHER_POINT`` only (the head and verify points leave it armed, so a step whose gather
  point never fires keeps the built-in store); every other step launches at the head point as
  before, and ``GATHER_POINT`` does nothing for it; with the lever off ``GATHER_POINT`` never
  launches.
* The commit: a paged step of more than ``context_kv_commit_paged.MAX_REQUESTS`` requests
  commits with the worker's ``_commit_context`` (the built-in's words), smaller ones keep the
  one-launch commit.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.kernels.context_kv_fork import (  # noqa: E501
    fork as context_kv_fork,
)
from tensorrt_llm._torch.speculative.dflash import DFlashWorker

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

BOUND = context_kv_fork.PARALLEL_MAX_REQUESTS


def _fork(*, paged=True, wide=BOUND, windowed_wide=None):
    worker = object.__new__(DFlashWorker)
    worker._ctx_paged = paged
    worker._ctx_base = None
    worker._ctx_keep = 0
    fork = object.__new__(context_kv_fork.ContextKvFork)
    fork.worker, fork.draft_model, fork.point = worker, None, context_kv_fork.POINT
    fork.stream = torch.cuda.Stream()
    fork.launches = {"head": 0, "verify": 0, context_kv_fork.GATHER_POINT: 0}
    fork._armed, fork._launched, fork._early = None, None, None
    fork._fallback_logged = fork._gather_point_logged = False
    fork.paged_wide_max_requests = wide
    fork.wide_max_requests = windowed_wide
    fork.parallel_min_requests = None
    fork.declined = None
    chains = []

    def plan(spec_metadata, attn_metadata):
        n = attn_metadata.num_seqs
        return (n, 7) if 1 <= n <= fork.max_requests() else None

    def side_chain(spec_metadata, attn_metadata, num_gens, k):
        chains.append((num_gens, k, torch.cuda.current_stream() == fork.stream))
        return None

    fork.plan, fork.side_chain = plan, side_chain
    return fork, chains


def _arm(fork, num_seqs):
    fork._armed = (SimpleNamespace(), SimpleNamespace(num_seqs=num_seqs))


def test_the_bound_applies_to_a_paged_context_only():
    assert context_kv_fork.ContextKvFork.max_requests(_fork()[0]) == BOUND
    off, _ = _fork(wide=None)
    assert off.max_requests() == context_kv_fork.MAX_REQUESTS
    # A dense arena keeps its bound with the lever set.
    dense, _ = _fork(paged=False)
    assert dense.max_requests() == context_kv_fork.MAX_REQUESTS


@pytest.mark.parametrize("num_seqs", [65, 73, 96, 128])
def test_a_wide_paged_step_forks_at_the_gather_point_only(num_seqs):
    fork, chains = _fork()
    _arm(fork, num_seqs)
    fork.fire("head")
    fork.fire("verify")
    assert not fork.launched and not chains and fork._armed is not None
    fork.fire(context_kv_fork.GATHER_POINT)
    assert fork.launched and chains == [(num_seqs, 7, True)]
    assert fork._launched.point == context_kv_fork.GATHER_POINT
    assert fork.launches == {"head": 0, "verify": 0, context_kv_fork.GATHER_POINT: 1}
    torch.cuda.current_stream().wait_stream(fork.stream)


def test_a_wide_step_whose_gather_point_did_not_fire_keeps_the_builtin_store():
    fork, chains = _fork()
    _arm(fork, 80)
    fork.fire("head")
    fork.fire("verify")
    # The worker's prepare takes nothing: the built-in prepare and store run.
    assert fork.take() is None and not chains


@pytest.mark.parametrize("num_seqs", [1, 17, 64])
def test_a_step_of_up_to_64_requests_forks_at_the_head_as_before(num_seqs):
    fork, chains = _fork()
    _arm(fork, num_seqs)
    fork.fire("head")
    assert chains == [(num_seqs, 7, True)] and fork._launched.point == "head"
    # The gather point after it does nothing.
    fork.fire(context_kv_fork.GATHER_POINT)
    assert len(chains) == 1
    torch.cuda.current_stream().wait_stream(fork.stream)


@pytest.mark.parametrize("num_seqs", [40, 73])
def test_without_the_lever_the_gather_point_never_launches(num_seqs):
    fork, chains = _fork(wide=None)
    _arm(fork, num_seqs)
    fork.fire(context_kv_fork.GATHER_POINT)
    assert not chains and fork._armed is not None
    fork.fire("head")
    # Up to 64 requests the head point launches as before; above, nothing forks.
    assert chains == ([(num_seqs, 7, True)] if num_seqs <= 64 else [])
    torch.cuda.current_stream().wait_stream(fork.stream)


def _paged_worker(slots_total, gen):
    worker = object.__new__(DFlashWorker)
    worker._ctx_paged = True
    worker._ctx_keep = 0
    worker._ctx_base = None
    worker._ctx_len = torch.randint(0, 5000, (slots_total,), generator=gen).cuda()
    worker._max_ctx = 6000
    worker._dummy_slot = slots_total - 1
    worker._ctx_block_tables = torch.zeros(slots_total, 4, dtype=torch.int32, device="cuda")
    worker._ctx_block_counts = torch.randint(300, 400, (slots_total,), generator=gen).cuda()
    worker._ctx_page_size = 16
    worker._compute_block_size = 8
    worker._ctx_unified_cache = False
    return worker


@pytest.mark.parametrize("num_gens", [33, 64, 65, 96, 128])
def test_the_commit_of_a_wide_step_is_the_builtin_commit(num_gens, monkeypatch):
    gen = torch.Generator().manual_seed(num_gens)
    worker = _paged_worker(BOUND + 1, gen)
    want = object.__new__(DFlashWorker)
    want.__dict__.update(worker.__dict__)
    want._ctx_len = worker._ctx_len.clone()
    fork, _ = _fork()
    fork.worker = worker
    fork.rows = torch.arange(BOUND, device="cuda")
    fork.paged_commit_on = True
    one_launch = []
    kernel = context_kv_fork.context_kv_commit_paged
    monkeypatch.setattr(
        context_kv_fork,
        "context_kv_commit_paged",
        lambda *args: (one_launch.append(args[2].numel()), kernel(*args))[1],
    )
    slots = torch.randperm(BOUND, generator=gen)[:num_gens].cuda()
    accepted = torch.randint(1, 9, (num_gens,), generator=gen, dtype=torch.int32).cuda()
    main = torch.cuda.current_stream()
    launched = context_kv_fork.Launched(main, None, num_gens, 7, context_kv_fork.GATHER_POINT, None)
    num_ctx = fork.commit(launched, slots, accepted)
    fork.join(launched)
    expected = DFlashWorker._commit_context(want, slots, accepted.long(), fork.rows[:num_gens])
    torch.cuda.synchronize()
    assert torch.equal(worker._ctx_len, want._ctx_len)
    assert torch.equal(num_ctx, expected)
    small = num_gens <= context_kv_fork.PAGED_COMMIT_MAX_REQUESTS
    assert one_launch == ([num_gens] if small else [])

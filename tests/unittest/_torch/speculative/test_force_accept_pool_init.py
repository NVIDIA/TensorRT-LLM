# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The forced-acceptance RNG pool is built when the worker is built.

``TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`` with a fractional part draws
from a fixed-seed pool and a device counter. CUDA graph warmup skips forced
acceptance, so a pool built on first use could be built inside a graph
capture. ``SpecWorkerBase.__init__`` builds it whenever forced acceptance is
set; without it the worker allocates nothing.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
from tensorrt_llm._torch.speculative import interface
from tensorrt_llm._torch.speculative.interface import (
    FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR,
    SpecWorkerBase,
)


class _StubSpecWorker(SpecWorkerBase):
    @property
    def max_draft_len(self) -> int:
        return 4

    def _forward_impl(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError


def _expected_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seeded_pool() -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(interface._FORCE_ACCEPT_RNG_SEED)
    return torch.rand(
        interface._FORCE_ACCEPT_RNG_POOL_SIZE, dtype=torch.float32, generator=generator
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("value", ["2.68", "2.43", "0.5", "3"])
def test_forced_acceptance_builds_the_pool_at_init(value, monkeypatch):
    monkeypatch.setenv(FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR, value)
    worker = _StubSpecWorker()

    pool, counter = worker._force_accept_rng_pool, worker._force_accept_rng_counter
    assert pool is not None and counter is not None
    assert pool.device.type == counter.device.type == _expected_device().type
    assert pool.dtype == torch.float32 and pool.shape == (interface._FORCE_ACCEPT_RNG_POOL_SIZE,)
    # The same table on every rank: the fixed seed, not the default generator.
    torch.testing.assert_close(pool.cpu(), _seeded_pool(), rtol=0, atol=0)
    assert counter.dtype == torch.int64 and counter.tolist() == [0]


@pytest.mark.cpu_only
@pytest.mark.parametrize("value", [None, "0", "0.0", "not-a-number"])
def test_without_forced_acceptance_nothing_is_built(value, monkeypatch):
    if value is None:
        monkeypatch.delenv(FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR, value)
    worker = _StubSpecWorker()

    assert worker.force_num_accepted_tokens == 0.0
    assert worker._force_accept_rng_pool is None
    assert worker._force_accept_rng_counter is None


@pytest.mark.cpu_only
def test_init_builds_what_the_first_use_would_have_built(monkeypatch):
    monkeypatch.setenv(FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR, "2.68")
    worker = _StubSpecWorker()
    lazy = SimpleNamespace(_force_accept_rng_pool=None, _force_accept_rng_counter=None)
    SpecWorkerBase._ensure_force_accept_rng_state(lazy, _expected_device())

    torch.testing.assert_close(worker._force_accept_rng_pool, lazy._force_accept_rng_pool)
    torch.testing.assert_close(worker._force_accept_rng_counter, lazy._force_accept_rng_counter)


@pytest.mark.cpu_only
def test_forced_acceptance_uses_the_pool_built_at_init(monkeypatch):
    monkeypatch.setenv(FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR, "2.5")
    worker = _StubSpecWorker()
    pool = worker._force_accept_rng_pool
    num_accepted = torch.ones(8, dtype=torch.int32, device=_expected_device())

    out = worker._apply_force_accepted_tokens(num_accepted, num_contexts=0, runtime_draft_len=4)

    assert worker._force_accept_rng_pool is pool
    assert worker._force_accept_rng_counter.tolist() == [1]
    assert set(out.tolist()) <= {3, 4}


@pytest.mark.cpu_only
def test_the_pool_is_real_under_meta_init(monkeypatch):
    """Models are built under ``MetaInitMode``; the pool must not become a meta
    tensor, and building it must not make the loader fall back to a full
    initialization."""
    monkeypatch.setenv(FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR, "2.68")
    with MetaInitMode():
        worker = _StubSpecWorker()

    assert not worker._force_accept_rng_pool.is_meta
    assert not worker._force_accept_rng_counter.is_meta
    torch.testing.assert_close(worker._force_accept_rng_pool.cpu(), _seeded_pool(), rtol=0, atol=0)


def test_a_graph_captures_forced_acceptance_without_an_eager_call(monkeypatch):
    """The first forced acceptance can be the captured one: warmup skips it."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    monkeypatch.setenv(FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR, "2.6")
    batch_size, runtime_draft_len, replays = 16, 4, 6

    reference = _StubSpecWorker()
    expected = []
    for _ in range(replays):
        out = reference._apply_force_accepted_tokens(
            torch.ones(batch_size, dtype=torch.int32, device="cuda"),
            num_contexts=0,
            runtime_draft_len=runtime_draft_len,
        )
        expected.append(out.clone())

    worker = _StubSpecWorker()
    static = torch.ones(batch_size, dtype=torch.int32, device="cuda")
    snapshot = torch.empty_like(static)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        worker._apply_force_accepted_tokens(
            static, num_contexts=0, runtime_draft_len=runtime_draft_len
        )
        snapshot.copy_(static)
    # Capture records the counter bump without running it.
    assert worker._force_accept_rng_counter.item() == 0

    for i in range(replays):
        static.fill_(1)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(snapshot, expected[i]), f"replay {i}"

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for ``drafter_step_prep`` (``MUSESPARK_OPEN_DRAFTER_FUSED_PREP``): the switch, what the fused
path takes and declines, the pool geometry, and its wiring into ``context_kv_fork``. The bitwise check of the
launches against the built-in methods is a separate GPU harness (not part of this suite)."""

import inspect
import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep8.kernels.context_kv_fork import (  # noqa: E501
    fork as fork_mod,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep8.ops.spec import (  # noqa: E501
    drafter_step_prep as prep,
)
from tensorrt_llm._torch.speculative.dflash import DFlashWorker

PAGE, HD, NKV, L, K = 32, 128, 1, 5, 7


def _pool(layers=L, pages=16, gap=64, dtype=torch.bfloat16):
    per = pages * 2 * NKV * PAGE * HD
    buf = torch.zeros(layers * (per + gap), dtype=dtype)
    order = list(reversed(range(layers)))
    return buf, [
        buf[o * (per + gap) : o * (per + gap) + per].view(pages, 2, NKV, PAGE, HD) for o in order
    ]


def _worker(n_slots=9, width=64):
    w = DFlashWorker.__new__(DFlashWorker)
    w._ctx_paged = True
    w._ctx_block_tables = torch.zeros(n_slots, width, dtype=torch.int32)
    w._ctx_block_counts = torch.zeros(n_slots, dtype=torch.int64)
    w._ctx_len = torch.zeros(n_slots, dtype=torch.int64)
    w._ctx_base = torch.zeros(n_slots, dtype=torch.int64)
    w._batch_to_slot = torch.zeros(n_slots - 1, dtype=torch.int64)
    w._dummy_slot = n_slots - 1
    w.spec_config = types.SimpleNamespace(
        max_draft_len=K
    )  # _draft_tokens_per_req = K + 1 (a property)
    w._ctx_kv_buf = _pool()[1]
    return w


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(prep.ENV, raising=False)
    assert prep.from_env() is None
    assert fork_mod._fused_prep_from_env() is None
    for value in ("0", "", "true", "2"):
        monkeypatch.setenv(prep.ENV, value)
        assert prep.from_env() is None
        assert fork_mod._fused_prep_from_env() is None


def test_on_and_min_requests(monkeypatch):
    monkeypatch.setenv(prep.ENV, "1")
    monkeypatch.delenv(prep.MIN_REQUESTS_ENV, raising=False)
    p = prep.from_env()
    assert p is not None and (p.min_requests, p.max_requests) == (
        prep.DEFAULT_MIN_REQUESTS,
        prep.MAX_REQUESTS,
    )
    assert isinstance(fork_mod._fused_prep_from_env(), prep.DrafterStepPrep)
    monkeypatch.setenv(prep.MIN_REQUESTS_ENV, "2")
    assert prep.from_env().min_requests == 2
    for bad in ("0", "65", "x"):
        monkeypatch.setenv(prep.MIN_REQUESTS_ENV, bad)
        with pytest.raises(ValueError):
            prep.from_env()


def test_max_requests_is_the_forks():
    assert prep.MAX_REQUESTS == fork_mod.MAX_REQUESTS


def test_pool_geometry_of_interleaved_views():
    buf, views = _pool()
    g = prep.pool_geometry(views)
    assert g is not None
    first, offsets, s_page, s_kv, s_head, s_entry, heads, dim = g
    assert first is views[0] and (heads, dim) == (NKV, HD)
    assert (s_page, s_kv, s_head, s_entry) == views[0].stride()[:4]
    for i, v in enumerate(views):
        assert first.data_ptr() + offsets[i] * first.element_size() == v.data_ptr()
    assert offsets[L:] == (0,) * (prep.MAX_LAYERS - L)
    assert any(o < 0 for o in offsets[:L])  # layer 0 is not the lowest address here


def test_pool_geometry_of_one_tensor():
    t = torch.zeros(L, 4, 2, NKV, PAGE, HD, dtype=torch.bfloat16)
    g = prep.pool_geometry(t)
    assert g is not None and g[1][:L] == tuple(i * t.stride(0) for i in range(L))


def test_pool_geometry_declines():
    _, views = _pool()
    assert prep.pool_geometry(views[:0]) is None
    assert prep.pool_geometry(_pool(layers=prep.MAX_LAYERS + 1)[1]) is None
    other = views[:2] + [views[2].to(torch.float16)]
    assert prep.pool_geometry(other) is None
    strided = views[:2] + [torch.zeros(16, 2, NKV, HD, PAGE, dtype=torch.bfloat16).transpose(3, 4)]
    assert prep.pool_geometry(strided) is None
    mla = [v[:, :1] for v in views]
    assert prep.pool_geometry(mla) is None


def test_takes_the_served_layout_and_declines_the_rest():
    p = prep.DrafterStepPrep(1)
    w = _worker()
    assert p.unsupported_reason(w, 4, K + 1) is None and p.takes(w, 4, K + 1)
    assert "outside" in p.unsupported_reason(w, 0, K + 1)
    assert "outside" in prep.DrafterStepPrep(2).unsupported_reason(w, 1, K + 1)
    assert "outside" in p.unsupported_reason(w, prep.MAX_REQUESTS + 1, K + 1)
    assert "K + 1" in p.unsupported_reason(w, 4, K)
    w2 = _worker()
    w2._ctx_paged = False
    assert "not paged" in p.unsupported_reason(w2, 4, K + 1)
    w2 = _worker()
    w2._ctx_block_tables = None
    assert "pages" in p.unsupported_reason(w2, 4, K + 1)
    w2 = _worker()
    w2._ctx_block_tables = w2._ctx_block_tables.long()
    assert "int32" in p.unsupported_reason(w2, 4, K + 1)
    w2 = _worker()
    w2._ctx_len = w2._ctx_len.int()
    assert "_ctx_len" in p.unsupported_reason(w2, 4, K + 1)
    w2 = _worker()
    w2._ctx_base = None
    assert p.unsupported_reason(w2, 4, K + 1) is None
    w2 = _worker()
    w2._dummy_slot = None
    assert "dummy" in p.unsupported_reason(w2, 4, K + 1)
    w2 = _worker()
    w2._ctx_kv_buf = [v.to(torch.float16) if i == 1 else v for i, v in enumerate(w2._ctx_kv_buf)]
    assert "geometry" in p.unsupported_reason(w2, 4, K + 1)


def test_a_worker_class_with_its_own_bookkeeping_is_refused():
    assert prep.methods_unsupported_reason(DFlashWorker.__new__(DFlashWorker)) is None
    for name in prep.REPLACED_METHODS:
        cls = type("Custom", (DFlashWorker,), {name: lambda self, *a, **k: None})
        assert name in prep.methods_unsupported_reason(cls.__new__(cls))
    w = DFlashWorker.__new__(DFlashWorker)
    w._commit_context = types.MethodType(lambda self, *a: None, w)
    assert "replaced" in prep.methods_unsupported_reason(w)
    # The fork's own worker class overrides only the table refresh, which the fused path calls by class.
    assert "_refresh_ctx_block_tables" not in prep.REPLACED_METHODS
    for name in prep.REPLACED_METHODS:
        assert name not in vars(fork_mod.ContextKvForkDFlashWorker)


def test_fork_wiring():
    assert "fused" in {f.name for f in fork_mod.Launched.__dataclass_fields__.values()}
    assert fork_mod.Launched.__dataclass_fields__["fused"].default is False
    assert "fused" in inspect.signature(fork_mod.ContextKvFork.side_chain).parameters
    src = inspect.getsource(fork_mod.ContextKvFork.side_chain)
    assert "self.prep.refresh_block_tables" in src and "self.prep.store_step_context" in src
    commit = inspect.getsource(fork_mod.ContextKvFork.commit)
    assert "worker._ctx_paged and launched.fused" in commit and "self.prep.commit_context" in commit
    fire = inspect.getsource(fork_mod.ContextKvFork.fire)
    assert "self.prep.takes(self.worker, num_gens, k + 1)" in fire
    init = inspect.getsource(fork_mod.ContextKvFork.__init__)
    assert "self.prep = _fused_prep_from_env()" in init

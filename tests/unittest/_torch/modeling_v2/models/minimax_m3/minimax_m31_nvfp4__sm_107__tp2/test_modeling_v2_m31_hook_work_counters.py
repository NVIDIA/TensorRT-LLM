# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""runtime_hooks.hook_work: the log-only work counters of the shared step-loop items.

* The counter: the first call logs "[M3 host] <ITEM> engaged: work 1 (...)", every power of two logs the running count,
  nothing else is logged, the exit hook is registered once, and the final lines ("[M3 host] <ITEM> engaged: final work
  N") are logged once per process. Every line says the count is host executions (graph replays not counted), and the
  lines match the engagement patterns.
* The call sites: each of the seven items counts only on the path its RuntimeHooks flag takes, never when the flag is
  off (the sampler, metadata and H2D items run on small stubs; the three DecoderRunner items are checked at their
  source: each count sits under its own flag). CPU only.
"""

import ast
import inspect
import re
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._utils as trtllm_utils
from tensorrt_llm._torch.pyexecutor import h2d_cache
from tensorrt_llm._torch.pyexecutor import runtime_hooks as rh
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import runner as runner_module
from tensorrt_llm._torch.pyexecutor.h2d_cache import H2DSkipCache
from tensorrt_llm._torch.pyexecutor.runtime_hooks import hook_work
from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata
from tensorrt_llm._torch.speculative.interface import SpecMetadata
from tensorrt_llm._torch.speculative.spec_sampler_base import SpecSampler
from tensorrt_llm.logger import logger

pytestmark = pytest.mark.cpu_only

ITEMS = (
    "HOST_D2H_SIDE",
    "SMALL_H2D",
    "PHILOX_PACK",
    "KV_LENS_ONCE",
    "H2D_CHANGED_SPAN",
    "BATCH_CHANGED_ONLY",
    "SKIP_ACCEPTED_UPLOAD",
)


class _Lines(list):
    """The logged lines, plus the functions hook_work registered at exit."""

    registered: list


@pytest.fixture
def lines(monkeypatch):
    """Every line hook_work logs, with rank 0, fresh counters and the exit registrations recorded instead of made."""
    out = _Lines()
    registered = []
    monkeypatch.setattr(logger, "info", out.append)
    monkeypatch.setattr(trtllm_utils, "mpi_rank", lambda: 0)
    monkeypatch.setattr(rh.atexit, "register", registered.append)
    monkeypatch.setattr(rh, "HOOK_WORK_COUNTS", {})
    monkeypatch.setattr(rh, "_hook_work_exit", [False, False])
    out.registered = registered
    return out


def _line(text: str) -> str:
    return f"{text} (rank 0; {rh.HOOK_WORK_HOST_COUNT})"


# ---------------------------------------------------------------------------
# The counter
# ---------------------------------------------------------------------------


def test_first_call_and_powers_of_two_log_the_count(lines):
    for _ in range(9):
        hook_work("UT_ITEM")
    assert lines == [
        _line("[M3 host] UT_ITEM engaged: work 1"),
        _line("[M3 host] UT_ITEM engaged: work 2"),
        _line("[M3 host] UT_ITEM engaged: work 4"),
        _line("[M3 host] UT_ITEM engaged: work 8"),
    ]
    assert rh.HOOK_WORK_COUNTS == {"UT_ITEM": 9}
    assert "graph replays not counted" in rh.HOOK_WORK_HOST_COUNT


def test_items_count_independently(lines):
    hook_work("X1")
    hook_work("X2")
    hook_work("X1")
    assert rh.HOOK_WORK_COUNTS == {"X1": 2, "X2": 1}


def test_the_final_lines_are_logged_once_per_process(lines):
    for item, n in (("B_ITEM", 3), ("A_ITEM", 1)):
        for _ in range(n):
            hook_work(item)
    assert lines.registered == [rh.log_hook_work_final]
    lines.clear()
    rh.log_hook_work_final()
    rh.log_hook_work_final()
    assert lines == [
        _line("[M3 host] A_ITEM engaged: final work 1"),
        _line("[M3 host] B_ITEM engaged: final work 3"),
    ]


def test_nothing_counted_logs_nothing_at_exit(lines):
    rh.log_hook_work_final()
    assert lines == [] and lines.registered == []


@pytest.mark.parametrize("item", ITEMS)
def test_the_lines_match_the_eng_kit(lines, item):
    """eng_tp2_v4's first and final patterns for each item."""
    hook_work(item)
    rh.log_hook_work_final()
    first, final = lines
    assert re.search(rf"\[M3 host\] {item} engaged: work 1 \(", first)
    assert re.search(rf"\[M3 host\] {item} engaged: final work [0-9]+", final)
    assert not re.search(rf"\[M3 host\] {item} engaged: final work", first)


# ---------------------------------------------------------------------------
# HOST_D2H_SIDE: SpecSampler._copy_outputs_to_host
# ---------------------------------------------------------------------------


class _Copier:
    event = None

    def stage_copy_to_host(self, t):
        return t


def _sampler(**attrs):
    @contextmanager
    def copier():
        yield _Copier()

    store = SimpleNamespace(
        new_tokens=torch.zeros(2), new_tokens_lens=torch.zeros(2), next_draft_tokens=torch.zeros(2)
    )
    return SimpleNamespace(
        store=store,
        _copy_to_host=lambda t: t,
        _make_side_stream_copier=copier,
        _async_worker_active=lambda: False,
        **attrs,
    )


@pytest.mark.parametrize(
    "flag,on_side_stream,counted", [(True, True, 1), (True, False, 0), (False, True, 0)]
)
def test_host_d2h_side_counts_the_side_stream_copies(lines, flag, on_side_stream, counted):
    sampler = _sampler(d2h_side_stream_copies=flag)
    SpecSampler._copy_outputs_to_host(sampler, None, store_copies_on_side_stream=on_side_stream)
    assert rh.HOOK_WORK_COUNTS.get("HOST_D2H_SIDE", 0) == counted


# ---------------------------------------------------------------------------
# SMALL_H2D: the sampler's static slot ids (SpecSampler._store_outputs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", [True, False])
def test_small_h2d_counts_the_static_slot_ids(lines, monkeypatch, flag):
    # A fold that takes the store, so _store_outputs returns right after the slot ids.
    fold = SimpleNamespace(scatter=lambda *args: True)
    sampler = _sampler(
        static_slot_ids=flag, spec_step_fold=fold, _device_slot_ids=lambda slots: "device"
    )
    # The stock path uploads to "cuda"; keep it on the host here.
    monkeypatch.setattr(
        torch, "as_tensor", lambda *args, **kwargs: SimpleNamespace(to=lambda **kw: "uploaded")
    )
    SpecSampler._store_outputs(sampler, [0, 1], None, None, None, None)
    assert rh.HOOK_WORK_COUNTS.get("SMALL_H2D", 0) == int(flag)


# ---------------------------------------------------------------------------
# SKIP_ACCEPTED_UPLOAD, SMALL_H2D (accepted counts packed) and KV_LENS_ONCE: DecoderRunner
# ---------------------------------------------------------------------------


def _guarding_flags(module, item: str) -> list[set]:
    """For each ``hook_work(item)`` call in ``module``: the ``self.runtime_hooks.<flag>`` names its enclosing ``if``
    tests (the branch the count sits on)."""
    tree = ast.parse(inspect.getsource(module))
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    found = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "hook_work"
            and node.args[0].value == item
        ):
            continue
        flags, child, parent = set(), node, parents.get(node)
        while parent is not None:
            if isinstance(parent, ast.If) and child in parent.body:
                flags |= {
                    a.attr
                    for a in ast.walk(parent.test)
                    if isinstance(a, ast.Attribute)
                    and isinstance(a.value, ast.Attribute)
                    and a.value.attr == "runtime_hooks"
                }
            child, parent = parent, parents.get(parent)
        found.append(flags)
    return found


@pytest.mark.parametrize(
    "item,flag",
    [
        ("SKIP_ACCEPTED_UPLOAD", "skip_accepted_draft_upload"),
        ("SMALL_H2D", "spec_small_h2d"),
        ("KV_LENS_ONCE", "kv_lens_update_once"),
    ],
)
def test_the_runner_items_count_only_under_their_flag(item, flag):
    (flags,) = _guarding_flags(runner_module, item)
    assert flag in flags, (item, flags)


# ---------------------------------------------------------------------------
# H2D_CHANGED_SPAN: H2DSkipCache._copy_changed_span
# ---------------------------------------------------------------------------


@pytest.fixture
def staged(monkeypatch):
    copies = []
    monkeypatch.setattr(h2d_cache.step_inputs, "is_open", lambda: True)
    monkeypatch.setattr(
        h2d_cache.step_inputs, "copy_", lambda dst, src: copies.append(src.tolist())
    )
    return copies


def _span(dst: torch.Tensor, old: torch.Tensor, src: torch.Tensor) -> bool:
    key = (dst.storage_offset(), tuple(dst.shape), tuple(dst.stride()), dst.dtype)
    return H2DSkipCache(changed_span=True)._copy_changed_span(dst, src, (key, old))


def test_changed_span_counts_a_staged_span(lines, staged):
    dst = torch.zeros(4, 16, dtype=torch.int32)
    old = torch.zeros(4, 16, dtype=torch.int32)
    src = old.clone()
    src[1, 3] = 7
    assert _span(dst, old, src) is True
    assert staged == [[7]]
    assert rh.HOOK_WORK_COUNTS == {"H2D_CHANGED_SPAN": 1}


def test_changed_span_does_not_count_a_declined_span(lines, staged):
    dst = torch.zeros(2, 4, dtype=torch.int32)
    old = torch.zeros(2, 4, dtype=torch.int32)
    src = torch.ones(2, 4, dtype=torch.int32)  # every column changed: over the fraction
    assert _span(dst, old, src) is False
    assert staged == [] and rh.HOOK_WORK_COUNTS == {}


def test_without_the_flag_the_span_is_never_tried():
    """The span path (the only caller of the counter in h2d_cache) is reached only with changed_span on."""
    src = inspect.getsource(H2DSkipCache.copy)
    assert "and self.changed_span" in src and "self._copy_changed_span(" in src
    assert H2DSkipCache().changed_span is False


# ---------------------------------------------------------------------------
# PHILOX_PACK: SpecMetadata._populate_request_rng_state
# ---------------------------------------------------------------------------


def _rng_meta(pack: bool, greedy_skip: bool = False) -> SpecMetadata:
    meta = SpecMetadata(max_num_requests=4, max_draft_len=3, max_total_draft_tokens=3)
    meta.pack_rng_uploads = pack
    meta.skip_greedy_rng_upload = greedy_skip
    # Host buffers large enough that nothing is allocated on the device.
    for name in ("request_seeds", "request_offsets", "seeds", "offsets"):
        setattr(meta, name, torch.zeros(16, dtype=torch.int64))
    return meta


def _rng_request(slot: int, seed):
    return SimpleNamespace(
        py_seq_slot=slot,
        py_decoding_iter=0,
        py_request_id=slot,
        sampling_config=SimpleNamespace(seed=seed),
    )


@pytest.mark.parametrize(
    "pack,greedy_skip,all_greedy,counted",
    [(True, False, False, 1), (False, False, False, 0), (True, True, True, 0)],
    ids=["packed", "flag off", "all-greedy skip"],
)
def test_philox_pack_counts_a_packed_step(lines, pack, greedy_skip, all_greedy, counted):
    meta = _rng_meta(pack, greedy_skip)
    meta.is_all_greedy_sample = all_greedy
    requests = [_rng_request(2, 1234), _rng_request(0, None)]
    meta._populate_request_rng_state(requests, [(0.7, 0, 0.9, 0.0, 4) for _ in requests])
    assert rh.HOOK_WORK_COUNTS.get("PHILOX_PACK", 0) == counted


# ---------------------------------------------------------------------------
# BATCH_CHANGED_ONLY: DFlashSpecMetadata.prepare
# ---------------------------------------------------------------------------


def _dflash_meta(changed_only: bool, worker=None):
    return SimpleNamespace(
        request_ids=[11, 12],
        num_generations=2,
        batch_map_changed_only=changed_only,
        _batch_indices_uploaded=0,
        batch_indices_cuda=torch.full((8,), -5, dtype=torch.int),
        _dflash_worker=worker,
    )


def _dflash_worker():
    w = SimpleNamespace(
        _ctx_buf_inited=True,
        _req_to_slot={11: 0, 12: 1},
        _req_ctx_pos={},
        _free_slots=[],
        _graph_dummy_id_floor=1 << 40,
        _dummy_slot=8,
        _batch_to_slot=torch.full((8,), -7, dtype=torch.long),
        _batch_to_slot_uploaded=None,
        _write_ctx_len=lambda evicted: None,
    )
    return w


@pytest.mark.parametrize("changed_only", [True, False])
def test_batch_changed_only_counts_each_skipped_upload(lines, changed_only):
    md = _dflash_meta(changed_only, _dflash_worker())
    DFlashSpecMetadata.prepare(md)  # first step: both maps uploaded
    assert rh.HOOK_WORK_COUNTS.get("BATCH_CHANGED_ONLY", 0) == 0
    DFlashSpecMetadata.prepare(md)  # steady: both skipped with the flag
    assert rh.HOOK_WORK_COUNTS.get("BATCH_CHANGED_ONLY", 0) == (2 if changed_only else 0)
    assert md.batch_indices_cuda[:2].tolist() == [0, 1]
    assert md._dflash_worker._batch_to_slot[:2].tolist() == [0, 1]


# ---------------------------------------------------------------------------
# Every item has exactly its call sites (the tests above check each one's branch).
# ---------------------------------------------------------------------------


def _hook_work_items(module) -> list[str]:
    tree = ast.parse(inspect.getsource(module))
    return sorted(
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "hook_work"
    )


def test_the_seven_items_have_their_call_sites():
    from tensorrt_llm._torch.speculative import dflash, interface, spec_sampler_base

    sites = {
        h2d_cache: ["H2D_CHANGED_SPAN"],
        runner_module: ["KV_LENS_ONCE", "SKIP_ACCEPTED_UPLOAD", "SMALL_H2D"],
        spec_sampler_base: ["HOST_D2H_SIDE", "SMALL_H2D"],
        interface: ["PHILOX_PACK"],
        dflash: ["BATCH_CHANGED_ONLY", "BATCH_CHANGED_ONLY"],
    }
    for module, items in sites.items():
        assert _hook_work_items(module) == items, module.__name__
    assert {item for items in sites.values() for item in items} == set(ITEMS)

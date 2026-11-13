# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MARKOV_GRID (dspark_markov_grid + dspark_markov_worker): the S4 chain's batch-1 calls on this target's wider grid.

- The grid plan: 198 CTAs x 256 rows for the M3.1 TP4 padded shard on a 212-SM SM107 (the catalog: 132 x 384), none
  when the GPU cannot keep a wider grid co-resident, the batch-1 variant is off, or the grid would not be wider; the
  exchange fits the catalog workspace.
- The worker routes only batch-1 calls there (with the KV-length rewind when it applies), sets the grid up at prebuild
  only when every rank agrees, logs its state and engagement, and the kill switch keeps the catalog grid.
- The prebuild agreement on the grid (``ranks_agree_value``) over lockstep ranks, and the opt-in switch.

The 4-rank check (the tokens and the rewind bitwise the catalog op's on 4 GPUs of one SM107 tray) is its own file,
``test_modeling_v2_m31_r6_markov_grid_matrix.py``, so a gate can budget it apart from these single-process tests.
"""

import ast
import importlib.util
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (  # noqa: E501
    dspark_markov_grid as grid_mod,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (  # noqa: E501
    dspark_markov_worker as s4,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
    dspark_markov_chain as chain,
)

_DIR = Path(s4.__file__).resolve().parent
ROWS = 50688  # the M3.1 TP4 shard (50,016) padded for the catalog op
VR200_SMS = 212


@pytest.fixture
def switches():
    saved = (chain._batch1_variant[0], chain._ll_exchange[0])
    yield
    chain._batch1_variant[0], chain._ll_exchange[0] = saved


@pytest.mark.parametrize("ll", [True, False], ids=["ll", "lamport"])
@pytest.mark.parametrize("block", [3, 7])
def test_pick_grid_on_vr200(ll, block):
    assert grid_mod.pick_grid(ROWS, block, VR200_SMS, ll) == 198
    assert ROWS // 198 == 256
    assert grid_mod.pick_grid(ROWS, block, 256, ll) == 198
    # Fewer SMs than the 198 co-resident CTAs: no grid of <= 256 rows per CTA fits.
    assert grid_mod.pick_grid(ROWS, block, 160, ll) == 0
    assert not grid_mod.grid_ok(
        ROWS, 132, block, ll
    )  # 384 rows per CTA: the catalog's, not this launcher's


def test_plan_needs_a_wider_grid_and_the_batch1_variant(monkeypatch, switches):
    monkeypatch.setattr(chain, "_sm_count", lambda device: VR200_SMS)
    chain.enable_batch1_variant(True)
    for ll in (True, False):
        chain.enable_ll_exchange(ll)
        assert grid_mod.catalog_grid(ROWS, 7, "cuda") == 132
        assert grid_mod.plan(ROWS, 7, "cuda", sm_count=VR200_SMS) == 198
        assert grid_mod.plan(ROWS, 7, "cuda", sm_count=160) == 0
        assert grid_mod.plan(ROWS, chain.MAX_BLOCK + 1, "cuda", sm_count=VR200_SMS) == 0
    chain.enable_batch1_variant(False)
    assert grid_mod.plan(ROWS, 7, "cuda", sm_count=VR200_SMS) == 0
    # Not wider than the catalog's own pick: no MARKOV_GRID.
    chain.enable_batch1_variant(True)
    monkeypatch.setattr(grid_mod, "catalog_grid", lambda rows, block, device: 198)
    assert grid_mod.plan(ROWS, 7, "cuda", sm_count=VR200_SMS) == 0


def test_the_exchange_fits_the_catalog_workspace(switches):
    kern = chain._kernel_module()
    world = 4
    buf_words = kern.buffer_words(chain.MAX_BLOCK, kern.MAX_BATCH, world, chain.MAX_GRID)
    ll = chain._kernel_module(batch1=True, ll=True)
    ws = dict(
        slots=world,
        buf_words=buf_words,
        ws_uc=torch.empty(ll.ll_workspace_words(buf_words), dtype=torch.int32),
        ws_flags=torch.empty(ll.LL_FLAG_WORDS, dtype=torch.int32),
    )
    for on in (True, False):
        chain.enable_ll_exchange(on)
        for block in (3, 7, chain.MAX_BLOCK):
            assert grid_mod.fits_workspace(block, 198, ws)
    chain.enable_ll_exchange(True)
    assert not grid_mod.fits_workspace(7, 198, dict(ws, ws_uc=torch.empty(16, dtype=torch.int32)))


# ------------------------------------------------------------------------------------------- the worker


class _Base:
    def __init__(self):
        self.calls = []

    def _refine_block_logits(self, draft_model, gen_logits, inputs, spec_metadata):
        self.calls.append("refine")
        return gen_logits

    def _apply_kv_rewind_after_draft(self, attn_metadata, spec_metadata):
        self.calls.append("stock_rewind")
        self._kv_rewind_pending = False


class _Worker(s4.S4WorkerMixin, _Base):
    pass


def _state(grid_b1, rows=ROWS, shard=8, block=3):
    st = s4._S4State(SimpleNamespace(tp_rank=1), rows, shard, 0, "w1", "w2s", block, {"ws": 1})
    for b in range(1, s4.MAX_FAST_BATCH + 1):
        st.bufs[(b, block)] = torch.full((b, block, rows), float("-inf"))
    st.grid_b1 = grid_b1
    return st


def _routing(monkeypatch, grid_b1=198, wide_ok=True):
    launches, lines = [], []

    def wide(base, prev, w1, w2s, offset, grid, *, kv_lens=None, rewind=None, rewind_first=0, **ws):
        launches.append(("wide", base.shape[0], grid, kv_lens is not None, rewind_first))
        return torch.zeros(base.shape[0], base.shape[1], dtype=torch.int32)

    def plain(base, *a, **k):
        launches.append(("catalog", base.shape[0]))
        return torch.zeros(base.shape[0], base.shape[1], dtype=torch.int32)

    def with_rewind(base, *a, **k):
        launches.append(("catalog_rewind", base.shape[0]))
        return torch.zeros(base.shape[0], base.shape[1], dtype=torch.int32)

    monkeypatch.setattr(
        s4, "markov_grid", SimpleNamespace(supports=lambda *a: wide_ok, markov_chain_b1=wide)
    )
    monkeypatch.setattr(s4, "chain", SimpleNamespace(supports=lambda *a: True, supports_rewind=lambda *a: True,
                                                     dspark_markov_chain=plain, dspark_markov_chain_rewind=with_rewind,
                                                     ll_exchange_engaged=lambda b: b == 1))  # fmt: skip
    monkeypatch.setattr(s4.logger, "info", lambda msg, *a, **k: lines.append(msg))
    monkeypatch.setattr(s4, "TAILBK_SPEC", True)
    wk = _Worker()
    wk.mapping = SimpleNamespace(tp_rank=1)
    wk._s4_step = _state(grid_b1)
    return wk, launches, lines


def _refine(wk, batch, meta=None):
    inputs = {"first_prev_tokens": torch.zeros(batch, dtype=torch.long)}
    wk._refine_block_logits(None, torch.zeros(batch, 3, 8), inputs, meta)


def test_only_batch1_calls_take_the_wide_grid(monkeypatch):
    wk, launches, lines = _routing(monkeypatch)
    _refine(wk, 1)
    _refine(wk, 2)
    _refine(wk, 1)
    assert launches == [("wide", 1, 198, False, 0), ("catalog", 2), ("wide", 1, 198, False, 0)]
    assert [m for m in lines if "MARKOV_GRID" in m] == [
        "[M3 dispatch] MARKOV_GRID taken: k3_markov batch-1 grid 198 x 256 rows (LL exchange) on rank 1"
    ]


def test_wide_grid_applies_the_pending_rewind(monkeypatch):
    wk, launches, _ = _routing(monkeypatch)
    meta = SimpleNamespace(is_cuda_graph=False)
    wk._s4_step_attn = SimpleNamespace(kv_lens_cuda=torch.zeros(4, dtype=torch.int32))
    wk._kv_rewind_pending = True
    wk._kv_rewind_amount = torch.zeros(1, dtype=torch.int32)
    wk._kv_rewind_nc, wk._kv_rewind_bs = 2, 3
    _refine(wk, 1, meta)
    wk._apply_kv_rewind_after_draft(None, meta)
    assert launches == [("wide", 1, 198, True, 2)]
    assert "stock_rewind" not in wk.calls and not wk._kv_rewind_pending


def test_no_grid_or_declined_keeps_the_catalog(monkeypatch):
    wk, launches, _ = _routing(monkeypatch, grid_b1=0)
    _refine(wk, 1)
    wk, launches2, _ = _routing(monkeypatch, wide_ok=False)
    _refine(wk, 1)
    assert launches == [("catalog", 1)] and launches2 == [("catalog", 1)]


def _prebuild(monkeypatch, plan_grid, agree=True, on=True):
    """prebuild_s4 with the catalog op and the collectives stubbed: the MARKOV_GRID set-up decision and log line."""
    lines, precompiled = [], []
    fake_chain = SimpleNamespace(
        MAX_BLOCK=8,
        padded_rows=lambda shard, max_batch, device: 12,
        pad_w2_shard=lambda w2, rows: w2,
        workspace_args=lambda mapping: {"slots": 4},
        pad_logits_buffer=lambda b, k, rows, device: torch.zeros(b, k, rows),
        enable_fp32_sum=lambda on: on,
        enable_batch1_variant=lambda on: on,
        enable_cached_inputs=lambda on: on,
        enable_ll_exchange=lambda on: on,
        precompile=lambda *a, **k: None,
    )
    fake_grid = SimpleNamespace(
        plan=lambda rows, block, device: plan_grid,
        fits_workspace=lambda block, grid, ws: True,
        precompile=lambda *a: precompiled.append(a[-2]),
        catalog_grid=lambda rows, block, device: 4,
    )
    monkeypatch.setattr(s4, "chain", fake_chain)
    monkeypatch.setattr(s4, "markov_grid", fake_grid)
    monkeypatch.setattr(s4, "MARKOV_GRID", on)
    monkeypatch.setattr(s4, "_head_is_sharded", lambda head, vocab, tp: True)
    monkeypatch.setattr(s4.logger, "info", lambda msg, *a, **k: lines.append(msg))
    monkeypatch.setattr(s4.atexit, "register", lambda fn: None)
    wk = _Worker()
    wk.mapping = SimpleNamespace(tp_size=4, tp_rank=0, enable_attention_dp=False)
    wk.max_draft_len = 3
    monkeypatch.setattr(wk, "_s4_ranks_agree", lambda local: local, raising=False)
    monkeypatch.setattr(wk, "_s4_ranks_agree_value", lambda v: agree and v != 0, raising=False)
    w = torch.zeros(32, s4.MARKOV_RANK, dtype=torch.bfloat16)
    draft = SimpleNamespace(has_markov_head=True, markov_w1=w, markov_w2=w)
    assert wk.prebuild_s4(SimpleNamespace(), draft)
    return (
        wk._s4.grid_b1,
        precompiled,
        [m for m in lines if "one-launch Markov chain set up" in m][0],
    )


def test_prebuild_sets_the_grid_only_when_ranks_agree(monkeypatch):
    grid, pre, line = _prebuild(monkeypatch, 6)
    assert (
        grid == 6
        and pre == [6]
        and line.endswith("MARKOV_GRID batch-1 grid 6 x 2 rows (catalog 4))")
    )
    grid, pre, line = _prebuild(monkeypatch, 6, agree=False)
    assert grid == 0 and pre == [] and line.endswith("MARKOV_GRID batch-1 grid off)")
    grid, pre, line = _prebuild(monkeypatch, 0)
    assert grid == 0 and pre == []
    grid, pre, _ = _prebuild(monkeypatch, 6, on=False)
    assert grid == 0 and pre == []


def _eval_switch(monkeypatch, env):
    """dspark_markov_worker.py's MARKOV_GRID assignment evaluated against a private copy of m31_switches under
    ``env``."""
    for name in list(os.environ):
        if name.startswith(("TRTLLM_M31_DISABLE_", "TRTLLM_M31_ENABLE_")):
            monkeypatch.delenv(name)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location(
        f"_m31_switches_markov_grid_{id(env)}", _DIR / "m31_switches.py"
    )
    switches = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(switches)
    tree = ast.parse((_DIR / "dspark_markov_worker.py").read_text())
    stmts = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "MARKOV_GRID" for t in n.targets)
    ]
    assert len(stmts) == 1
    ns = {"m31_disabled": switches.m31_disabled, "m31_enabled": switches.m31_enabled}
    exec(compile(ast.Module(stmts, []), "dspark_markov_worker.py", "exec"), ns)
    return ns["MARKOV_GRID"]


@pytest.mark.parametrize(
    "env,want",
    [
        ({}, False),  # opt-in until its micro bench and a disaggregated GEN run back it
        ({"TRTLLM_M31_ENABLE_MARKOV_GRID": "1"}, True),
        ({"TRTLLM_M31_ENABLE_MARKOV_GRID": "0"}, False),
        ({"TRTLLM_M31_ENABLE_MARKOV_GRID": "1", "TRTLLM_M31_DISABLE_MARKOV_GRID": "1"}, False),
        ({"TRTLLM_M31_DISABLE_MARKOV_GRID": "1"}, False),
    ],
)
def test_markov_grid_switch(monkeypatch, env, want):
    assert _eval_switch(monkeypatch, env) is want


class _Lockstep:
    """``world`` ranks as threads over a barrier-based all-reduce(AND): a rank that made a different number of
    collective calls than its peers breaks the barrier (BrokenBarrierError) instead of hanging."""

    def __init__(self, world: int):
        self.barrier = threading.Barrier(world, timeout=20)
        self.local = [False] * world
        self.calls = [0] * world

    def all_ranks(self, rank: int):
        def collective(local: bool) -> bool:
            self.local[rank] = bool(local)
            self.barrier.wait()
            result = all(self.local)
            self.barrier.wait()
            self.calls[rank] += 1
            return result

        return collective


def _agree_on(values):
    lock = _Lockstep(len(values))
    results, errors = [None] * len(values), []

    def run(rank):
        try:
            results[rank] = s4.ranks_agree_value(lock.all_ranks(rank), values[rank])
        except BaseException as e:  # noqa: BLE001 - reported below
            errors.append(e)

    threads = [threading.Thread(target=run, args=(r,)) for r in range(len(values))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not errors, errors
    assert len(set(lock.calls)) == 1, (
        f"ranks made different numbers of collective calls: {lock.calls}"
    )
    assert len(set(results)) == 1, results
    return results[0], lock.calls[0]


@pytest.mark.parametrize(
    "values,want",
    [
        ([198, 198, 198, 198], True),
        ([198, 198, 132, 198], False),  # one rank plans another grid
        ([198, 198, 198, 0], False),  # one rank plans none
        ([0, 0, 0, 0], False),
        ([198, 198 | 1 << 15, 198, 198], False),  # differ only in the top bit compared
        (
            [198 | 1 << 16, 198, 198, 198],
            False,
        ),  # past the compared bits on one rank: counted as disagreement
        ([1, 1, 1, 1], True),
        ([256, 256], True),
    ],
)
def test_ranks_agree_value_in_lockstep(values, want):
    """MARKOV_GRID's prebuild agreement on the batch-1 grid (``_s4_ranks_agree_value``'s logic over a lockstep
    collective): the same answer on every rank, the same number of collective rounds on every rank."""
    got, calls = _agree_on(values)
    assert got is want
    assert calls <= 1 + 2 * s4.AGREE_VALUE_BITS


def test_worker_agreement_uses_the_tp_collective(monkeypatch):
    """``_s4_ranks_agree_value`` runs ranks_agree_value over the TP group's workspace collective."""
    import tensorrt_llm._torch.distributed.ops as ops

    seen = []
    monkeypatch.setattr(ops, "_get_mnnvl_workspace_comm", lambda mapping: ("comm", mapping))
    monkeypatch.setattr(
        ops, "_mnnvl_workspace_all_succeeded", lambda comm, ok: seen.append((comm, ok)) or ok
    )
    wk = _Worker()
    wk.mapping = "tp-mapping"
    assert wk._s4_ranks_agree_value(198) is True
    # One round for "nonzero", then per bit one ("1 everywhere") or two ("0 everywhere" after "1 everywhere" failed).
    rounds = 1 + sum(1 if (198 >> b) & 1 else 2 for b in range(s4.AGREE_VALUE_BITS))
    assert seen[0] == (("comm", "tp-mapping"), True) and len(seen) == rounds
    seen.clear()
    assert wk._s4_ranks_agree_value(0) is False and seen == [(("comm", "tp-mapping"), False)]

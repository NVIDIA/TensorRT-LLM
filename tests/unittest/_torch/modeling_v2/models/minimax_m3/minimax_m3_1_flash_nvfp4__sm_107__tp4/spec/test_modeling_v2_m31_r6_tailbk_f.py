# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TAILBK_F at TP4: the drafter's k_norm weight multiply inside the DRAFTER_CTX_KV_STORE_FUSED ctx store kernel (catalog
dspark_ctx_store.fold_k_norm_weight, as TP2 and TP8 set it).

At the live TP4 drafter shapes (5 layers, 1 KV head per rank, head dim 128, rotary 64, hidden 6144, 8 rows = one
request x (K7 + 1)) the folded store -- TAIL_FORK_KV's prep on an aux stream + the store kernel, and the whole
fused_ctx_store -- must write the paged cache byte-identical to the unfolded store (torch's bf16 weight multiply), eager
and in a CUDA graph; the prep must really leave the multiply to the kernel; the worker's prebuild sets the fold from
the switch and the store's installation; the engagement line is logged once per rank.
"""

import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
    dspark_ctx_store as store,
)

TARGET = "minimax_m3_1_flash_nvfp4__sm_107__tp4"
_DIR = Path(_family.__file__).resolve().parent / TARGET

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

L, NKV, HD, ROT, H, PS = 5, 1, 128, 64, 6144, 32
KP1 = 8


@pytest.fixture
def fold_flag():
    """The process-wide fold flag, restored after the test."""
    saved = store._fold_w[0]
    yield store._fold_w
    store._fold_w[0] = saved


def _fake_drafter(device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = (torch.randn(L * 2 * NKV * HD, H, generator=g) * 0.02).to(torch.bfloat16).to(device)
    # A k_norm weight away from 1 (bf16 rounding of the product matters).
    kn = (1.0 + 0.3 * torch.randn(L, HD, generator=g)).to(torch.bfloat16).to(device)
    pos = torch.arange(4096, dtype=torch.float32)
    inv = 1.0 / (10000.0 ** (torch.arange(0, ROT, 2, dtype=torch.float32) / ROT))
    f = pos[:, None] * inv[None, :]
    cos_sin = torch.cat([f.cos(), f.sin()], dim=1).contiguous().to(device)
    return SimpleNamespace(
        _fused_kv_weight=w,
        _fused_kv_bias=None,
        _num_attn_layers=L,
        _num_kv_heads=NKV,
        _head_dim=HD,
        _k_norm_stacked=kn,
        _k_norm_eps=1e-6,
        _is_neox=True,
        _input_ln_eps=None,
        _get_cos_sin_cache=lambda: cos_sin,
        _build_fused_kv_buffers=lambda: None,
    )


def _operands(device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    pages = 64
    projected = (torch.randn(KP1, H, generator=g) * 2).to(torch.bfloat16).to(device)
    kv_buf = [
        torch.zeros(pages, 2, NKV, PS, HD, dtype=torch.bfloat16, device=device) for _ in range(L)
    ]
    table = torch.randperm(pages, generator=g)[:16].to(torch.int32).view(1, 16).to(device)
    counts = torch.tensor([16], dtype=torch.int64, device=device)
    positions = (torch.arange(KP1) + 301 + seed).to(torch.int64).to(device)
    num_accepted = torch.tensor([1 + seed % KP1], dtype=torch.int32, device=device)
    rows = torch.zeros(1, dtype=torch.int64, device=device)
    return projected, kv_buf, table, counts, positions, num_accepted, rows


def _bits(bufs):
    return [b.view(torch.int16) for b in bufs]


@needs_gpu
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_whole_store_folded_equals_unfolded(fold_flag, seed):
    dev = torch.device("cuda")
    draft = _fake_drafter(dev, seed)
    projected, kv_ref, table, counts, positions, nacc, rows = _operands(dev, seed)
    kv_new = [t.clone() for t in kv_ref]
    fold_flag[0] = False
    assert store.fused_ctx_store(draft, projected, positions, nacc, rows, None, table, kv_ref, KP1,
                                 capacity_counts=counts, max_col=None)  # fmt: skip
    fold_flag[0] = True
    assert store.fused_ctx_store(draft, projected, positions, nacc, rows, None, table, kv_new, KP1,
                                 capacity_counts=counts, max_col=None)  # fmt: skip
    torch.cuda.synchronize()
    for a, b in zip(_bits(kv_ref), _bits(kv_new)):
        assert torch.equal(a, b)
    assert any(t.abs().sum() > 0 for t in kv_new), "the store wrote nothing"


@needs_gpu
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_forked_prep_folded_equals_unfolded(fold_flag, graph):
    """TAIL_FORK_KV's split (prep on an aux stream, then the store kernel) with the fold == the unfolded whole store."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (  # noqa: E501
        dspark_tailfork as tf,
    )

    dev = torch.device("cuda")
    draft = _fake_drafter(dev, 4)
    projected, kv_ref, table, counts, positions, nacc, rows = _operands(dev, 4)
    kv_new = [t.clone() for t in kv_ref]
    fold_flag[0] = False
    assert store.fused_ctx_store(draft, projected, positions, nacc, rows, None, table, kv_ref, KP1,
                                 capacity_counts=counts, max_col=None)  # fmt: skip
    fold_flag[0] = True
    store._cache_arg(kv_new)
    # The prep leaves the weight multiply to the kernel: its K is the bare rms_norm.
    prep = tf.ctx_store_prep(draft, projected)
    assert prep.fold
    k_bare = torch.nn.functional.rms_norm(
        prep.kv_flat.view(KP1, L, 2, NKV, HD)[:, :, 0].contiguous(), (HD,), eps=draft._k_norm_eps
    )
    assert torch.equal(prep.k, k_bare)
    aux = torch.cuda.Stream()

    def step():
        main = torch.cuda.current_stream()
        aux.wait_stream(main)
        with torch.cuda.stream(aux):
            p = tf.ctx_store_prep(draft, projected)
        main.wait_stream(aux)
        return tf.ctx_store_launch(p, draft, projected, positions, nacc, rows, None, table, kv_new, KP1,
                                   capacity_counts=counts, max_col=None)  # fmt: skip

    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            assert step() is True
        g.replay()
    else:
        assert step() is True
    torch.cuda.synchronize()
    for a, b in zip(_bits(kv_ref), _bits(kv_new)):
        assert torch.equal(a, b)


def _dspark():
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (  # noqa: E501
        dspark,
    )

    return dspark


def _dspark_with_hooks():
    """The target's dspark module, when the engine's DSpark worker carries the drafter hooks prebuild arms
    (``enable_fused_ctx_store`` among them); without them prebuild sets up nothing, so these tests skip."""
    dspark = _dspark()
    if not hasattr(dspark.M3DSparkWorker, "enable_fused_ctx_store"):
        pytest.skip("the engine's DSpark worker has no drafter hooks (enable_fused_ctx_store)")
    return dspark


@pytest.mark.parametrize("on", [True, False])
def test_prebuild_sets_the_fold(monkeypatch, fold_flag, on):
    """The worker's prebuild (no head, no drafter: the shell's bare call) sets the fold when DRAFTER_CTX_KV_STORE_FUSED
    is installed and TAILBK_F is on, and logs its state."""
    dspark = _dspark_with_hooks()
    lines = []
    monkeypatch.setattr(dspark.logger, "info", lambda msg, *a, **k: lines.append(msg))
    monkeypatch.setattr(dspark, "TAILBK_F", on)
    fold_flag[0] = not on
    w = dspark.M3DSparkWorker.__new__(dspark.M3DSparkWorker)
    assert w.prebuild(None, None) is None
    assert fold_flag[0] is on
    want = f"M3.1 DSpark TAILBK_F: k_norm weight multiply in the ctx store kernel {'on' if on else 'off'} on rank None"
    assert want in lines


def test_prebuild_without_the_store_keeps_the_fold_off(monkeypatch, fold_flag):
    dspark = _dspark_with_hooks()
    monkeypatch.setattr(dspark.logger, "info", lambda msg, *a, **k: None)
    monkeypatch.setattr(dspark, "TAILBK_F", True)
    monkeypatch.setattr(dspark.M3DSparkWorker, "enable_fused_ctx_store", lambda self, s: False)
    fold_flag[0] = True
    w = dspark.M3DSparkWorker.__new__(dspark.M3DSparkWorker)
    w.prebuild(None, None)
    assert fold_flag[0] is False


_LEAK: dict = {}


def test_prebuild_fold_leak_a_prebuild_without_restoring(monkeypatch):
    """A test that runs prebuild and does not restore the process-wide fold itself (as the w0 / tailopt files do)."""
    dspark = _dspark_with_hooks()
    monkeypatch.setattr(dspark.logger, "info", lambda msg, *a, **k: None)
    monkeypatch.setattr(dspark, "TAILBK_F", True)
    _LEAK["before"] = store._fold_w[0]
    w = dspark.M3DSparkWorker.__new__(dspark.M3DSparkWorker)
    w.prebuild(None, None)
    assert store._fold_w[0] is True


def test_prebuild_fold_leak_b_restored_by_the_conftest():
    """The tp4 conftest's autouse fixture restored the fold after the previous test."""
    if "before" not in _LEAK:
        pytest.skip(
            "runs after test_prebuild_fold_leak_a_prebuild_without_restoring in the same process"
        )
    assert store._fold_w[0] == _LEAK.pop("before")


def test_engagement_logged_once_per_rank(monkeypatch):
    dspark = _dspark()
    lines = []
    monkeypatch.setattr(dspark.logger, "info", lambda msg, *a, **k: lines.append(msg))
    monkeypatch.setattr(dspark, "ctx_store_launch", lambda *a, **k: True)
    w = dspark.M3DSparkWorker.__new__(dspark.M3DSparkWorker)
    w.mapping = SimpleNamespace(tp_rank=3)
    projected = torch.zeros(KP1, 4)
    for fold in (False, True, True):
        w._tf = SimpleNamespace(prep=SimpleNamespace(fold=fold), joined=True)
        assert w._tf_ctx_store(None, projected, None, None, None, None, None, None, KP1) is True
    assert [m for m in lines if "TAILBK_F" in m] == [
        "[M3 dispatch] TAILBK_F taken: k_norm weight multiply in the ctx store kernel (fork prep, 8 rows) on rank 3"
    ]


_WHOLE = "[M3 dispatch] TAILBK_F taken: k_norm weight multiply in the ctx store kernel (whole store, 8 rows) on rank 2"
_FORK = "[M3 dispatch] TAILBK_F taken: k_norm weight multiply in the ctx store kernel (fork prep, 8 rows) on rank 2"


def _logging_worker(dspark, monkeypatch, lines):
    monkeypatch.setattr(dspark.logger, "info", lambda msg, *a, **k: lines.append(msg))
    w = dspark.M3DSparkWorker.__new__(dspark.M3DSparkWorker)
    w.mapping = SimpleNamespace(tp_rank=2)
    w._tf = None
    return w


def test_engagement_logged_once_per_kind_whole_store_first(monkeypatch, fold_flag):
    """Serving order: the first decode step is not forked (the drafter's context buffers are set up inside it), so
    its whole store comes first; the forked steps' prep line must still be logged once, and each kind only once."""
    dspark = _dspark()
    lines = []
    monkeypatch.setattr(dspark, "fused_ctx_store", lambda *a, **k: True)
    monkeypatch.setattr(dspark, "ctx_store_launch", lambda *a, **k: True)
    w = _logging_worker(dspark, monkeypatch, lines)
    fold_flag[0] = True
    draft = _fake_drafter("cpu")
    projected = torch.zeros(KP1, 4)
    for _ in range(2):
        w._tf = None
        assert w._tf_ctx_store(draft, projected, None, None, None, None, None, None, KP1) is True
    for _ in range(2):
        w._tf = SimpleNamespace(prep=SimpleNamespace(fold=True), joined=True)
        assert w._tf_ctx_store(draft, projected, None, None, None, None, None, None, KP1) is True
    w._tf = None
    assert w._tf_ctx_store(draft, projected, None, None, None, None, None, None, KP1) is True
    assert [m for m in lines if "TAILBK_F" in m] == [_WHOLE, _FORK]


@pytest.mark.parametrize(
    "case",
    [
        "flag_off",
        "fp32_weight",
        "strided_weight",
        "short_weight",
        "no_rows",
        "not_stored",
        "no_kv_weight",
    ],
)
def test_whole_store_line_follows_the_stores_own_fold(monkeypatch, fold_flag, case):
    """The whole-store line is logged only when fused_ctx_store itself folded (its fold condition on the operands it
    used), not on the process-wide flag alone."""
    dspark = _dspark()
    lines = []
    monkeypatch.setattr(dspark, "fused_ctx_store", lambda *a, **k: case != "not_stored")
    w = _logging_worker(dspark, monkeypatch, lines)
    fold_flag[0] = case != "flag_off"
    draft = _fake_drafter("cpu")
    projected = torch.zeros(0 if case == "no_rows" else KP1, 4)
    kn = draft._k_norm_stacked
    if case == "fp32_weight":
        draft._k_norm_stacked = kn.float()
    elif case == "strided_weight":
        draft._k_norm_stacked = torch.cat([kn, kn], dim=1)[:, ::2]
    elif case == "short_weight":
        draft._k_norm_stacked = kn[:-1]
    elif case == "no_kv_weight":
        draft._fused_kv_weight = None
    w._tf_ctx_store(draft, projected, None, None, None, None, None, None, KP1)
    assert [m for m in lines if "TAILBK_F" in m] == []
    if case != "not_stored":
        assert not dspark._whole_store_folded(draft, projected)
    # A drafter that meets the store's fold condition, with the flag on.
    fold_flag[0] = True
    assert dspark._whole_store_folded(_fake_drafter("cpu"), torch.zeros(KP1, 4))


@needs_gpu
def test_whole_store_through_the_worker_folds_and_logs(monkeypatch, fold_flag):
    """The worker's store on an unforked step (the real fused_ctx_store): the folded cache bytes equal the unfolded
    store's and the whole-store line is logged."""
    dspark = _dspark()
    lines = []
    w = _logging_worker(dspark, monkeypatch, lines)
    dev = torch.device("cuda")
    draft = _fake_drafter(dev, 6)
    projected, kv_ref, table, counts, positions, nacc, rows = _operands(dev, 6)
    kv_new = [t.clone() for t in kv_ref]
    fold_flag[0] = False
    assert store.fused_ctx_store(draft, projected, positions, nacc, rows, None, table, kv_ref, KP1,
                                 capacity_counts=counts, max_col=None)  # fmt: skip
    fold_flag[0] = True
    assert w._tf_ctx_store(draft, projected, positions, nacc, rows, None, table, kv_new, KP1,
                           capacity_counts=counts, max_col=None)  # fmt: skip
    torch.cuda.synchronize()
    for a, b in zip(_bits(kv_ref), _bits(kv_new)):
        assert torch.equal(a, b)
    assert [m for m in lines if "TAILBK_F" in m] == [_WHOLE]


def _eval_switch(monkeypatch, env):
    for name in list(os.environ):
        if name.startswith(("TRTLLM_M31_DISABLE_", "TRTLLM_M31_ENABLE_")):
            monkeypatch.delenv(name)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location(
        f"_m31_switches_tailbkf_{id(env)}", _DIR / "m31_switches.py"
    )
    switches = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(switches)
    tree = ast.parse((_DIR / "dspark.py").read_text())
    stmts = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "TAILBK_F" for t in n.targets)
    ]
    assert len(stmts) == 1
    ns = {"m31_disabled": switches.m31_disabled}
    exec(compile(ast.Module(stmts, []), "dspark.py", "exec"), ns)
    return ns["TAILBK_F"]


@pytest.mark.parametrize(
    "env,want",
    [
        ({}, True),
        ({"TRTLLM_M31_DISABLE_TAILBK_F": "1"}, False),
        ({"TRTLLM_M31_DISABLE_TAIL_BOOKKEEP_SPEC": "1"}, False),
        ({"TRTLLM_M31_DISABLE_TAILBK_F": "0"}, True),
    ],
)
def test_tailbk_f_switch(monkeypatch, env, want):
    assert _eval_switch(monkeypatch, env) is want

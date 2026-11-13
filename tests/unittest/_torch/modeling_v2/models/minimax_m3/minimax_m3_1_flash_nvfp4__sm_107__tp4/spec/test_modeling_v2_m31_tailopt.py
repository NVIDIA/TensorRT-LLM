# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The TP4 b1 drafter + verify tail items of the sm_107 TP4 target (tailopt), each bitwise to the path it
replaces:

- BLOCKOFF_GUARD / BLOCKOFF_PATCH (modeling.py): the runtime hooks of the separate draft KV cache's block-table copy,
  on by default, each with its kill switch (CPU: the constants evaluated from modeling.py's source).
- TAIL_DSFOLD_FWD (modeling.py): the DSpark step fold with the in-forward store + staging, on by default, kill switch.
- TAIL_FORK / TAIL_FORK_KV / TAIL_DEFER_CAPTURE / TAIL_FC_GATHER / TAIL_FORK_REFRESH / TAIL_FORCEVEC /
  TAIL_LEAN_WORKER (dspark.py): default on, kill switches.
- dspark_tailfork.ctx_store_prep + ctx_store_launch == catalog fused_ctx_store, the prep on another stream (eager and
  forked inside a captured CUDA graph), at the live TP4 drafter shapes (5 layers, 1 KV head per rank, head dim 128,
  rotary dim 64, 8 rows = one request x (K7 + 1)).
- dspark_tailfork.copy_captures == DFlash's per-layer inplace_slice_copy (concatenation and slice forms).
- M3DSparkWorker._apply_force_accepted_tokens under TAIL_FORCEVEC == the in-tree draw (counts and counter), over many
  steps and fractions.
"""

import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family

TARGET = "minimax_m3_1_flash_nvfp4__sm_107__tp4"
_DIR = Path(_family.__file__).resolve().parent / TARGET
_SWITCHES = _DIR / "m31_switches.py"


def _eval_constants(monkeypatch, source: Path, names, env):
    """The module-level assignments ``names`` of ``source`` evaluated (in file order) against a private copy of
    m31_switches loaded under ``env`` (every other TRTLLM_M31_* variable cleared)."""
    for name in list(os.environ):
        if name.startswith(("TRTLLM_M31_DISABLE_", "TRTLLM_M31_ENABLE_", "TRTLLM_M31_TAIL_")):
            monkeypatch.delenv(name)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location(f"_m31_switches_tailopt_{id(env)}", _SWITCHES)
    switches = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(switches)
    tree = ast.parse(source.read_text(), filename=str(source))
    statements = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in names
    ]
    assert sorted(n.targets[0].id for n in statements) == sorted(names)
    namespace = {
        "m31_disabled": switches.m31_disabled,
        "m31_enabled": switches.m31_enabled,
        "os": os,
    }
    exec(compile(ast.Module(statements, []), str(source), "exec"), namespace)
    return {n: namespace[n] for n in names}


_BLOCKOFF = ("STEP_BLOCKOFF_GUARD", "STEP_BLOCKOFF_PATCH")
_TAIL = (
    "TAIL_FORK",
    "TAIL_FORK_KV",
    "TAIL_DEFER_CAPTURE",
    "TAIL_FC_GATHER",
    "TAIL_FORK_REFRESH",
    "TAIL_FORCEVEC",
    "TAIL_LEAN_WORKER",
)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "env, want",
    [
        ({}, (True, True)),
        ({"TRTLLM_M31_DISABLE_BLOCKOFF_GUARD": "1"}, (False, False)),
        ({"TRTLLM_M31_DISABLE_BLOCKOFF_PATCH": "1"}, (True, False)),
        ({"TRTLLM_M31_DISABLE_BLOCKOFF_GUARD": "0"}, (True, True)),
    ],
)
def test_blockoff_switches(monkeypatch, env, want):
    got = _eval_constants(monkeypatch, _DIR / "modeling.py", _BLOCKOFF, env)
    assert (got["STEP_BLOCKOFF_GUARD"], got["STEP_BLOCKOFF_PATCH"]) == want


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "env, want",
    [
        ({}, True),
        ({"TRTLLM_M31_DISABLE_TAIL_DSFOLD_FWD": "1"}, False),
        ({"TRTLLM_M31_ENABLE_TAIL_DSFOLD_FWD": "1"}, True),
        ({"TRTLLM_M31_DISABLE_STEP_GATHER_FUSED": "1"}, True),
    ],
)
def test_dsfold_fwd_switch(monkeypatch, env, want):
    got = _eval_constants(monkeypatch, _DIR / "modeling.py", ("TAIL_DSFOLD_FWD",), env)
    assert got["TAIL_DSFOLD_FWD"] is want


@pytest.mark.cpu_only
def test_dsfold_fwd_fold_is_the_in_forward_dsfold():
    """The DSpark fold built under TAIL_DSFOLD_FWD: DSFOLD's gather-counting fold with the in-forward store + staging
    of the Eagle3 fold (scatter still declines, so unarmed rows keep the sampler's own stores)."""
    tree = ast.parse((_DIR / "modeling.py").read_text())
    cls = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    fwd = cls["M3DSparkStepFoldFwd"]
    assert [b.id for b in fwd.bases] == ["M3DSparkStepFold"]
    assert "scatter" not in {n.name for n in fwd.body if isinstance(n, ast.FunctionDef)}
    init = next(n for n in fwd.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    call = next(
        n
        for n in ast.walk(init)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "__init__"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "M3SpecStepFold"
    )
    assert {k.arg: k.value for k in call.keywords}["in_forward"].value is True


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "env, want",
    [
        # TAIL_DEFER_CAPTURE is opt-in on this target (FRONT_CAPFORK keeps the capture copies off the PDL chain).
        ({}, (True, True, False, True, True, True, True)),
        ({"TRTLLM_M31_ENABLE_TAIL_DEFER_CAPTURE": "1"}, (True, True, True, True, True, True, True)),
        (
            {
                "TRTLLM_M31_ENABLE_TAIL_DEFER_CAPTURE": "1",
                "TRTLLM_M31_DISABLE_TAIL_DEFER_CAPTURE": "1",
            },
            (True, True, False, True, True, True, True),
        ),
        (
            {"TRTLLM_M31_ENABLE_TAIL_DEFER_CAPTURE": "1", "TRTLLM_M31_DISABLE_TAIL_FORK": "1"},
            (False, False, False, False, False, True, True),
        ),
        ({"TRTLLM_M31_DISABLE_TAIL_FORK_KV": "1"}, (True, False, False, True, True, True, True)),
        ({"TRTLLM_M31_DISABLE_TAIL_FC_GATHER": "1"}, (True, True, False, False, True, True, True)),
        (
            {"TRTLLM_M31_DISABLE_TAIL_FORK_REFRESH": "1"},
            (True, True, False, True, False, True, True),
        ),
        ({"TRTLLM_M31_DISABLE_TAIL_FORCEVEC": "1"}, (True, True, False, True, True, False, True)),
        (
            {"TRTLLM_M31_DISABLE_TAIL_LEAN_WORKER": "1"},
            (True, True, False, True, True, True, False),
        ),
    ],
)
def test_tail_switches(monkeypatch, env, want):
    got = _eval_constants(monkeypatch, _DIR / "dspark.py", _TAIL, env)
    assert tuple(got[n] for n in _TAIL) == want


# ----------------------------------------------------------------------------------------------------- GPU tests

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# The live TP4 drafter: 5 layers, 4 KV heads / TP4 = 1 per rank, head dim 128, rotary 64, hidden 6144, K7 c1.
L, NKV, HD, ROT, H, PS = 5, 1, 128, 64, 6144, 32
KP1 = 8


def _fake_drafter(device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = (torch.randn(L * 2 * NKV * HD, H, generator=g) * 0.02).to(torch.bfloat16).to(device)
    kn = (1.0 + 0.1 * torch.randn(L, HD, generator=g)).to(torch.bfloat16).to(device)
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


def _store_operands(device, seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    pages = 64
    projected = (torch.randn(KP1, H, generator=g)).to(torch.bfloat16).to(device)
    kv_buf = [
        torch.zeros(pages, 2, NKV, PS, HD, dtype=torch.bfloat16, device=device) for _ in range(L)
    ]
    table = torch.randperm(pages, generator=g)[:16].to(torch.int32).view(1, 16).to(device)
    counts = torch.tensor([16], dtype=torch.int64, device=device)
    positions = (torch.arange(KP1) + 301).to(torch.int64).to(device)
    num_accepted = torch.tensor([5], dtype=torch.int32, device=device)
    rows = torch.zeros(1, dtype=torch.int64, device=device)
    return projected, kv_buf, table, counts, positions, num_accepted, rows


@needs_gpu
@pytest.mark.parametrize("graph", [False, True])
def test_ctx_store_split_on_a_forked_stream_is_bitwise(graph):
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (  # noqa: E501
        dspark_tailfork as tf,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark.dspark_ctx_store import (
        fused_ctx_store,
    )

    dev = torch.device("cuda")
    draft = _fake_drafter(dev)
    projected, kv_ref, table, counts, positions, num_accepted, rows = _store_operands(dev)
    kv_new = [t.clone() for t in kv_ref]
    assert fused_ctx_store(
        draft, projected, positions, num_accepted, rows, None, table, kv_ref, KP1,
        capacity_counts=counts, max_col=None,
    )  # fmt: skip
    # the per-layer pointer array of kv_new, built outside capture (as the warmup step does in-model)
    tf._store._cache_arg(kv_new)
    aux = torch.cuda.Stream()

    def step():
        main = torch.cuda.current_stream()
        aux.wait_stream(main)
        with torch.cuda.stream(aux):
            prep = tf.ctx_store_prep(draft, projected)
        main.wait_stream(aux)
        return tf.ctx_store_launch(
            prep, draft, projected, positions, num_accepted, rows, None, table, kv_new, KP1,
            capacity_counts=counts, max_col=None,
        )  # fmt: skip

    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            assert step() is True
        g.replay()
    else:
        assert step() is True
    torch.cuda.synchronize()
    for a, b in zip(kv_ref, kv_new):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))
    assert any(t.abs().sum() > 0 for t in kv_new), "the store wrote nothing"


@needs_gpu
def test_ctx_store_launch_declines_foreign_rows():
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (  # noqa: E501
        dspark_tailfork as tf,
    )

    dev = torch.device("cuda")
    draft = _fake_drafter(dev)
    projected, kv, table, counts, positions, num_accepted, rows = _store_operands(dev)
    prep = tf.ctx_store_prep(draft, projected)
    other = projected.clone()
    assert tf.ctx_store_launch(
        prep, draft, other, positions, num_accepted, rows, None, table, kv, KP1,
        capacity_counts=counts, max_col=None,
    ) is None  # fmt: skip
    assert tf.ctx_store_prep(SimpleNamespace(_fused_kv_weight=None), projected) is None


@needs_gpu
@pytest.mark.parametrize("all_layers", [True, False])
def test_copy_captures_is_the_slice_copy(all_layers):
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (  # noqa: E501
        dspark_tailfork as tf,
    )
    from tensorrt_llm._torch.custom_ops import inplace_slice_copy

    dev = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(7)
    layers, rows, cap = 5, 8, 64
    want = torch.randn(cap, layers * H, generator=g).to(torch.bfloat16).to(dev)
    got = want.clone()
    parts = [
        (i, torch.randn(rows, H, generator=g).to(torch.bfloat16).to(dev)) for i in range(layers)
    ]
    if not all_layers:
        parts = parts[1:4]
    for i, t in parts:
        inplace_slice_copy(want, t, i * H, (i + 1) * H)
    used_cat = tf.copy_captures(got, rows, H, layers, parts)
    assert used_cat is all_layers
    assert torch.equal(want.view(torch.int16), got.view(torch.int16))


def _worker(force: float, warm: bool, device):
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (  # noqa: E501
        dspark,
    )

    w = dspark.M3DSparkWorker.__new__(dspark.M3DSparkWorker)
    w.force_num_accepted_tokens = force
    w._force_accept_rng_pool = None
    w._force_accept_rng_counter = None
    w.mapping = SimpleNamespace(tp_rank=0)
    if warm:
        assert w._tfv_prebuild(device) is (force - int(force) > 0.0)
    else:
        w._tfv_warm = False
    # Both workers hold the same pool and counter state (an integer force never builds it in prebuild, and the in-tree
    # path never touches it).
    w._ensure_force_accept_rng_state(device)
    return w


@needs_gpu
@pytest.mark.parametrize("force", [3.04, 0.5, 2.6, 6.99, 3.0, 7.5])
@pytest.mark.parametrize("num_contexts,num_gens", [(0, 1), (0, 8), (1, 1), (0, 3)])
def test_forcevec_is_the_in_tree_draw(force, num_contexts, num_gens):
    dev = torch.device("cuda")
    ref = _worker(force, False, dev)
    new = _worker(force, True, dev)
    k = 7
    spec = SimpleNamespace(is_cuda_graph=False)
    for step in range(400):
        base = torch.randint(1, k + 2, (num_contexts + num_gens,), dtype=torch.int32, device=dev)
        a = ref._apply_force_accepted_tokens(base.clone(), num_contexts, k, spec_metadata=spec)
        b = new._apply_force_accepted_tokens(base.clone(), num_contexts, k, spec_metadata=spec)
        assert torch.equal(a, b), step
        assert torch.equal(ref._force_accept_rng_counter, new._force_accept_rng_counter), step
    # the eager warmup of a graph batch returns the counts untouched on both paths
    warm = SimpleNamespace(is_cuda_graph=True)
    base = torch.ones(num_contexts + num_gens, dtype=torch.int32, device=dev)
    assert torch.equal(
        ref._apply_force_accepted_tokens(base.clone(), num_contexts, k, spec_metadata=warm),
        new._apply_force_accepted_tokens(base.clone(), num_contexts, k, spec_metadata=warm),
    )


@needs_gpu
def test_forcevec_in_a_captured_graph():
    dev = torch.device("cuda")
    ref = _worker(3.04, False, dev)
    new = _worker(3.04, True, dev)
    spec = SimpleNamespace(is_cuda_graph=True)
    na_ref = torch.ones(1, dtype=torch.int32, device=dev)
    na_new = torch.ones(1, dtype=torch.int32, device=dev)
    graphs = []
    for w, na in ((ref, na_ref), (new, na_new)):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            w._apply_force_accepted_tokens(na, 0, 7, spec_metadata=spec)
        graphs.append(g)
    for step in range(300):
        na_ref.fill_(1)
        na_new.fill_(1)
        graphs[0].replay()
        graphs[1].replay()
        torch.cuda.synchronize()
        assert torch.equal(na_ref, na_new), step
        assert torch.equal(ref._force_accept_rng_counter, new._force_accept_rng_counter), step

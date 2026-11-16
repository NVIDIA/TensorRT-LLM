# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``head_sampler_coop_fold``'s VERIFY_GRID: ``dflash_verify_fold`` over one program per request.

CPU: ``configure`` and the grid kernel's statements against the loop kernel's (from the file). GPU (CUDA; not run on
CPU): the grid launch == the loop launch bit for bit on every output, the counter and kv_lens, for 1..64 requests,
every force mode, consecutive calls (the fractional mode's counter), with and without kv_lens / seq_lens; plus the
graph-replay time of both (``python test_modeling_v2_verify_fold_grid.py --bench``).
"""

from __future__ import annotations

import ast
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[9]
_FOLD = (
    _ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/models/musespark_open"
    "/muse_spark_open_816b_a42b_nvfp4__sm_107__tep/ops/spec/dflash_verify_fold.py"
)


def _host():
    tree = ast.parse(_FOLD.read_text())
    body = [
        n
        for n in tree.body
        if (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("GRID", "_GRID_LOGGED") for t in n.targets))
        or (isinstance(n, ast.FunctionDef) and n.name == "configure")
    ]
    mod = types.ModuleType("_fold_host")
    exec(compile(ast.Module(body=body, type_ignores=[]), str(_FOLD), "exec"), mod.__dict__)
    return mod


def test_configure():
    m = _host()
    assert m.GRID is False
    m.configure(grid=True)
    assert m.GRID is True
    m.configure()
    assert m.GRID is False
    for bad in (1, "1", None):
        with pytest.raises(ValueError, match="grid switch"):
            m.configure(grid=bad)


def _kernel_body(name: str) -> str:
    tree = ast.parse(_FOLD.read_text())
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    return ast.unparse(fn)


def _without_given_branch(body: str) -> list:
    """The loop body's lines minus upstream #21789's ACCEPTED_GIVEN branch (``FORCE == 3``), which
    the grid kernel does not take: ``__call__`` keeps those steps on the loop kernel (gint4-zf)."""
    out, skip_indent = [], None
    for line in body.splitlines():
        indent = len(line) - len(line.lstrip())
        if skip_indent is not None:
            if line.strip() and indent <= skip_indent:
                skip_indent = None
            else:
                continue
        if line.strip() == "elif FORCE == 3:":
            skip_indent = indent
            continue
        out.append(line)
    return out


def test_the_grid_body_is_the_loop_body():
    """Every statement of the loop kernel's per-request body appears in the grid kernel (with g its program),
    but the ACCEPTED_GIVEN branch, whose steps the grid launch never takes."""
    loop = _kernel_body("_dflash_verify_fold_kernel").split("for g in range(num_gens):")[1]
    grid = _kernel_body("_dflash_verify_fold_grid_kernel")
    assert "if GRID and not pre_wait and force_mode != ACCEPTED_GIVEN:" in _FOLD.read_text()
    for line in _without_given_branch(loop):
        stmt = line.strip()
        if not stmt or stmt.startswith("#") or "embed_ids_ptr + num_gens" in stmt:
            continue
        assert stmt in grid, stmt
    assert "tl.store(embed_ids_ptr + num_gens, mask_id)" in grid
    assert "old = tl.atomic_add(counter_ptr, ticket_unit)" in grid
    assert "counter = (old & 4294967295) + 1" in grid or "counter = (old & 0xFFFFFFFF) + 1" in _FOLD.read_text()


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------


def _gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA")
    import importlib

    mod = importlib.import_module(
        "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open."
        "muse_spark_open_816b_a42b_nvfp4__sm_107__tep.ops.spec.dflash_verify_fold"
    )
    return torch, mod


def _case(torch, g, k, block, seed, mode, with_kv, with_seq):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    target = torch.randint(0, 50, (g * (k + 1),), device=dev, generator=gen, dtype=torch.int64).to(torch.int32)
    drafts = target.view(g, k + 1)[:, :k].clone()
    flip = torch.rand((g, k), device=dev, generator=gen) < 0.3
    drafts[flip] += 1
    slots = torch.randperm(64, device=dev, generator=gen)[:g].to(torch.int64)
    ctx_len = torch.randint(0, 100000, (64,), device=dev, generator=gen, dtype=torch.int64)
    kv = torch.randint(0, 1000, (g + 3,), device=dev, generator=gen, dtype=torch.int64).to(torch.int32) if with_kv else None
    seq = torch.randint(0, 1000, (g + 200,), device=dev, generator=gen, dtype=torch.int64).to(torch.int32) if with_seq else None
    pool = torch.rand((1 << 12,), device=dev, generator=gen)
    counter = torch.tensor([seed * 7], dtype=torch.int64, device=dev)
    return target, drafts, slots, ctx_len, kv, seq, pool, counter


@pytest.mark.parametrize("g", [1, 2, 7, 12, 16, 33, 64])
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_gpu_grid_matches_the_loop(g, mode):
    torch, mod = _gpu()
    k, block = 7, 8
    for with_kv, with_seq in ((True, True), (False, False)):
        target, drafts, slots, ctx_len, kv, seq, pool, counter = _case(torch, g, k, block, 11 * g + mode, mode, with_kv, with_seq)
        kv_a = kv.clone() if kv is not None else None
        kv_b = kv.clone() if kv is not None else None
        c_a, c_b = counter.clone(), counter.clone()
        for call in range(3):
            args = lambda kvx, cx: (  # noqa: E731
                target, drafts, mode, 2, k + 1, 0.05, pool, cx, slots, ctx_len, 100000, block, 99, kvx, seq
            )
            mod.configure(grid=False)
            ref = mod.dflash_verify_fold(*args(kv_a, c_a))
            mod.configure(grid=True)
            got = mod.dflash_verify_fold(*args(kv_b, c_b))
            mod.configure(grid=False)
            torch.cuda.synchronize()
            for name in ref._fields:
                a, b = getattr(ref, name), getattr(got, name)
                if a is None:
                    assert b is None, name
                    continue
                assert torch.equal(a, b), (g, mode, call, name)
            assert torch.equal(c_a, c_b), (g, mode, call, c_a.item(), c_b.item())
            if kv is not None:
                assert torch.equal(kv_a, kv_b), (g, mode, call)


def _bench(iters=200):
    torch, mod = _gpu()
    for g in (8, 12, 16):
        target, drafts, slots, ctx_len, kv, seq, pool, counter = _case(torch, g, 7, 8, 3, 2, True, True)
        line = [f"requests {g}:"]
        for grid in (False, True):
            mod.configure(grid=grid)

            def step():
                return mod.dflash_verify_fold(
                    target, drafts, 2, 2, 8, 0.05, pool, counter, slots, ctx_len, 100000, 8, 99, kv, seq
                )

            step()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            line.append(f"{'grid' if grid else 'loop'} {start.elapsed_time(end) * 1000 / iters:.1f} us")
        mod.configure(grid=False)
        print(" ".join(line), flush=True)


if __name__ == "__main__":
    import sys

    if "--bench" in sys.argv:
        _bench()

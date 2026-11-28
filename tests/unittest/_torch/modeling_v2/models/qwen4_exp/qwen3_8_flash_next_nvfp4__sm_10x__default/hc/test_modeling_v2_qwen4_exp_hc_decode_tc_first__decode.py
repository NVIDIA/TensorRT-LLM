# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for this target's qwen4_exp_hc_decode_tc_first variant
(ops/hc/qwen4_exp_hc_decode_tc_first__decode): cells in the band against the float64
reference at the decode rows the members run, the row-count guard, graph replay at the
note's 1, 4, 16 and 256 rows (every tile variant)."""

import ast
import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default.ops.hc import (  # noqa: E501
    qwen4_exp_hc_decode_tc_first__decode as kernel_module,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default.ops.hc.qwen4_exp_hc_decode_tc_first__decode import (  # noqa: E501
    qwen4_exp_hc_decode_tc_first as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "qwen4_exp_hc_decode_tc_first requires a CUDA device"

DEV = "cuda"
RANK = 320


def _ratio(out, ref) -> float:
    """Worst element's distance from `ref` as a fraction of the band: one ulp of
    the element (`eps * |ref|`) plus `eps / 16` of the tensor's largest value.
    An fp32-accumulated 10240-term projection rounded once can sit one ulp off
    the float64 value at a rounding midpoint; the absolute term covers SiLU's
    cancellation near zero, where an fp32 logit error of ~1e-6 relative to the
    row's scale is many ulps of a tiny output."""
    eps = torch.finfo(ref.dtype).eps
    r = ref.double()
    return ((out.double() - r).abs() / (eps * r.abs() + eps / 16 * r.abs().max())).max().item()


def _compare(out, ref) -> None:
    for got, want in zip(out, ref):
        assert (got is None) == (want is None)
        if got is None:
            continue
        assert got.dtype == want.dtype and got.shape == want.shape and got.is_contiguous()
        assert _ratio(got, want) <= 1.0, f"{_ratio(got, want):.4g}x the band"


def _build(spec, seed):
    hc, hidden = 4, 2560
    rows, width, inject = spec["rows"], hc * hidden, spec.get("inject", True)
    g = torch.Generator(device=DEV).manual_seed(seed)
    # This checkpoint's layer statistics: down ~N(0, 0.0296), block_inject ~N(0, 0.0107),
    # the normed stream unit-RMS per stream times (1 + gamma), gamma ~N(0, 0.477).
    down = torch.randn(RANK, width, generator=g, device=DEV) * 0.0296
    parts = [down] + ([torch.randn(hc, width, generator=g, device=DEV) * 0.0107] if inject else [])
    w = torch.cat(parts)
    w = torch.cat([w, torch.zeros(384 - w.shape[0], width, device=DEV)])
    gamma = torch.randn(width, generator=g, device=DEV) * 0.477
    x = torch.randn(rows, hc, hidden, generator=g, device=DEV)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    normed = (x.flatten(-2) * (1.0 + gamma)).to(torch.bfloat16)
    return normed, w.to(torch.bfloat16).t(), RANK, hc, inject


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % (2**31))
    low, inject = op(*args)
    assert low.shape == (cell.spec["rows"], RANK)
    _compare((low, inject), op.reference(*args))


def test_guard_refuses_more_than_two_tiles_of_rows() -> None:
    normed, first, rank, c, wi = _build(dict(rows=257), seed=4)
    with validating(op), pytest.raises(AssertionError, match="at most 256 rows"):
        op(normed, first, rank, c, wi)


@pytest.mark.parametrize("rows", [1, 4, 16, 256])
def test_cuda_graph_capture_and_replay(rows) -> None:
    normed, first, rank, c, wi = _build(dict(rows=rows), seed=5)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(normed, first, rank, c, wi)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_low, g_inject = op(normed, first, rank, c, wi)
    fresh = _build(dict(rows=rows), seed=6)[0]
    normed.copy_(fresh)
    graph.replay()
    torch.cuda.synchronize()
    low, inject = op(fresh, first, rank, c, wi)
    assert torch.equal(g_low, low) and torch.equal(g_inject, inject)


_PEER_LOADS = ("_ld_dsmem_f32", "_ld_dsmem_f32_if")


def _callee(call):
    return call.func.attr if isinstance(call.func, ast.Attribute) else call.func.id


def _parents(tree):
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def _outer(node, parents, kind, stop):
    """The outermost `kind` ancestor of `node` below `stop`."""
    found = None
    while node is not stop:
        node = parents[node]
        if isinstance(node, kind):
            found = node
    return found


def _same(a, b) -> bool:
    return ast.dump(a) == ast.dump(b)


def _assigned(stmts, name):
    """Values assigned to `name` by the statements of one block (not nested ones)."""
    return [
        s.value
        for s in stmts
        if isinstance(s, ast.Assign)
        and any(_same(t, ast.Name(name, ast.Store())) for t in s.targets)
    ]


def _predicate_problems(func, call, parents):
    """Problems of one `_ld_dsmem_f32_if` call: its predicate must be a name set to 1 under
    exactly the condition of the `if` that sums the loaded value, in a loop over the same
    steps, with the condition's step-local names spelled the same in both loops."""
    where = f"line {call.lineno}"
    pred = call.args[1] if len(call.args) > 1 else None
    if not isinstance(pred, ast.Name):
        return [f"{where}: predicate {ast.unparse(pred) if pred else None} is not a share flag"]
    load_loop = _outer(call, parents, ast.For, func)
    store = parents[call]
    if load_loop is None or not (
        isinstance(store, ast.Assign) and isinstance(store.targets[0], ast.Subscript)
    ):
        return [f"{where}: the peer load is not stored into a buffer inside a step loop"]
    buf = store.targets[0].value
    zero, one = (ast.parse(f"cutlass.Int32({v})", mode="eval").body for v in (0, 1))
    sets = [
        n
        for n in ast.walk(func)
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign))
        and any(
            isinstance(t, ast.Name) and t.id == pred.id
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
        )
    ]
    guard = [
        s
        for s in load_loop.body
        if isinstance(s, ast.If)
        and not s.orelse
        and len(s.body) == 1
        and s.body[0] in sets
        and _same(s.body[0].value, one)
    ]
    init = [s for s in load_loop.body if s in sets and _same(s.value, zero)]
    if (
        len(sets) != 2
        or len(guard) != 1
        or len(init) != 1
        or load_loop.body.index(init[0]) > load_loop.body.index(guard[0])
    ):
        return [
            f"{where}: `{pred.id}` is not `cutlass.Int32(0)` then `cutlass.Int32(1)` under one "
            "`if` of the step loop (and set nowhere else)"
        ]
    cond = guard[0].test
    sums = [
        n
        for n in ast.walk(func)
        if isinstance(n, ast.If)
        and _outer(n, parents, ast.For, func) not in (None, load_loop)
        and any(
            isinstance(r, ast.Subscript) and isinstance(r.ctx, ast.Load) and _same(r.value, buf)
            for s in n.body
            for r in ast.walk(s)
        )
    ]
    if len(sums) != 1:
        return [f"{where}: {len(sums)} `if` blocks read `{ast.unparse(buf)}`, expected one"]
    sum_if = sums[0]
    sum_loop = _outer(sum_if, parents, ast.For, func)
    problems = []
    if not _same(cond, sum_if.test):
        problems.append(
            f"{where}: loads issued under `{ast.unparse(cond)}`, summed under "
            f"`{ast.unparse(sum_if.test)}`"
        )
    if not (_same(load_loop.target, sum_loop.target) and _same(load_loop.iter, sum_loop.iter)):
        problems.append(f"{where}: the load and sum loops run different steps")
    for name in sorted({n.id for n in ast.walk(cond) if isinstance(n, ast.Name)}):
        here, there = _assigned(load_loop.body, name), _assigned(sum_loop.body, name)
        if [ast.dump(v) for v in here] != [ast.dump(v) for v in there]:
            problems.append(f"{where}: `{name}` differs between the load and the sum loop")
    return problems


def _peer_load_problems(source):
    """`{class name: [problem]}` for every class of `source` that reads its cluster peers'
    shared memory: a CTA must not leave its exit barrier while one of its threads still has a
    peer load in flight."""
    tree = ast.parse(source)
    parents = _parents(tree)
    out = {}
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        calls = sorted(
            (c for c in ast.walk(cls) if isinstance(c, ast.Call)),
            key=lambda c: (c.lineno, c.col_offset),
        )
        calls = [c for c in calls if isinstance(c.func, (ast.Attribute, ast.Name))]
        loads = [c for c in calls if _callee(c) in _PEER_LOADS]
        if not loads:
            continue
        last_read = max(c.lineno for c in loads)
        after = [c for c in calls if c.lineno > last_read and "cluster_arrive" in _callee(c)]
        if not after:
            out[cls.name] = ["no cluster barrier after the peer reads (a CTA may exit while read)"]
            continue
        if all(_callee(c) == "cluster_arrive" for c in after):
            out[cls.name] = []  # a release arrive waits for every load of the thread
            continue
        problems = [
            f"line {c.lineno}: unpredicated peer load before a relaxed exit arrive"
            for c in loads
            if _callee(c) == "_ld_dsmem_f32"
        ]
        for c in loads:
            if _callee(c) == "_ld_dsmem_f32_if":
                problems += _predicate_problems(
                    _outer(c, parents, ast.FunctionDef, cls), c, parents
                )
        out[cls.name] = problems
    return out


_DSMEM_KERNELS = sorted(_peer_load_problems(open(kernel_module.__file__).read()).items())


@pytest.mark.parametrize("name,problems", _DSMEM_KERNELS, ids=[k[0] for k in _DSMEM_KERNELS])
def test_no_peer_load_in_flight_at_exit(name, problems) -> None:
    """A CTA leaves the cluster barrier after its reduce and exits, releasing its shared memory.
    A thread that reaches that barrier with a peer load still in flight (issued, never summed)
    lets the barrier release that peer under it: CUDBG_EXCEPTION_CLUSTER_BLOCK_NOT_PRESENT, a
    sticky launch failure of the whole process. ptxas hoists a later step's unconditional loads
    ahead of the share check, so either the arrive after them is a release `cluster_arrive()`,
    which waits for them, or every peer load is predicated on exactly the condition under which
    the thread sums it (`_ld_dsmem_f32_if`). A wider predicate (always true, say) keeps the
    output bitwise the same and brings the fault back, so the check compares the predicate's
    condition with the sum's; a narrower one would drop partials and fail the bitwise tests."""
    assert _DSMEM_KERNELS, "no kernel of the module reads peer shared memory"
    assert not problems, f"{name}: " + "; ".join(problems)


_REDUCE_FIXTURE = """
class K:
    def kernel(self):
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        for it in cutlass.range_constexpr(self.red_iters):
            local = tidx + it * nthr
            f = base + local
            fs = cutlass.min(f, total - 1)
            live = cutlass.Int32(0)
            if local < per and f < total:
                live = cutlass.Int32(1)
            for j in cutlass.range_constexpr(self.ks):
                vals[it * self.ks + j] = _ld_dsmem_f32_if(peers[j] + fs, live)
        for it in cutlass.range_constexpr(self.red_iters):
            local = tidx + it * nthr
            f = base + local
            if local < per and f < total:
                acc = vals[it * self.ks]
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()
"""

_BAD_FIXTURES = {
    "always_true": ("_if(peers[j] + fs, live)", "_if(peers[j] + fs, cutlass.Int32(1))"),
    "unconditional_flag": (
        "            if local < per and f < total:\n                live = cutlass.Int32(1)\n",
        "            live = cutlass.Int32(1)\n",
    ),
    "flag_set_twice": (
        "            for j in cutlass.range_constexpr(self.ks):\n                vals",
        "            live = cutlass.Int32(1)\n"
        "            for j in cutlass.range_constexpr(self.ks):\n                vals",
    ),
    "wider_condition": (
        "            if local < per and f < total:\n                live",
        "            if f < total:\n                live",
    ),
    "renamed_step": (
        "            local = tidx + it * nthr\n            f = base + local\n            fs",
        "            local = tidx\n            f = base + local\n            fs",
    ),
    "unpredicated": ("_ld_dsmem_f32_if(peers[j] + fs, live)", "_ld_dsmem_f32(peers[j] + fs)"),
    "no_exit_barrier": (
        "        cute.arch.cluster_arrive_relaxed()\n        cute.arch.cluster_wait()\n",
        "",
    ),
}


def test_peer_load_check_accepts_matched_predicate_and_release_exit() -> None:
    assert _peer_load_problems(_REDUCE_FIXTURE) == {"K": []}
    release = _REDUCE_FIXTURE.replace(
        "_ld_dsmem_f32_if(peers[j] + fs, live)", "_ld_dsmem_f32(peers[j] + fs)"
    ).replace("cluster_arrive_relaxed()", "cluster_arrive()")
    assert _peer_load_problems(release) == {"K": []}


@pytest.mark.parametrize("old,new", _BAD_FIXTURES.values(), ids=list(_BAD_FIXTURES))
def test_peer_load_check_rejects(old, new) -> None:
    """Each mutation of the matched fixture issues a peer load the thread may not sum before
    the relaxed exit arrive (or drops the exit barrier); the check must flag it."""
    assert _REDUCE_FIXTURE.count(old) == 1
    problems = _peer_load_problems(_REDUCE_FIXTURE.replace(old, new))
    assert problems["K"], problems

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The split sampler's multi-CTA top-p test (``head_sampler_coop_fold``'s TOPP_ROUNDS, ``draw_topp``).

CPU (no torch, no device):

* ``head_sampler.configure`` / ``topp_scratch_words`` / ``topp_takes`` and ``target_head.topp_setup`` (loaded
  from their files' ASTs): the knob's range, the scratch sizes, the loud failure when the library has no
  ``draw_topp``, the engagement log;
* ``head_sampler.cu`` against ``head_sampler.py``: the record slot, the round cap, the op schema and its call;
* a float32 model of the shipped tokens-only top-p rejection loop (one CTA, two sweeps per round) against the
  restructured one (kernel B's round 0, then ``rounds`` sweep / decide pairs, the last finishing in the CTA,
  each rejected round's next support taken from the test's partial masses): the same token on every row, for
  every ``rounds``, with the same Philox draws consumed.

GPU (SM107 only; not run on CPU): ``split_sample`` with ``topp_rounds`` 1..4 against ``topp_rounds`` 0 (kernel
B's in-CTA shipped body) bit for bit, on rows that reject often (low top_p, flat rows) and on the served
sampling (temperature 1, top_p 0.95) at 8..128 rows, with neutral / top-k / min-p rows mixed in; and the
timing of both under graph replay (``python test_modeling_v2_head_sampler_topp.py --bench``).
"""

from __future__ import annotations

import ast
import math
import random
import re
import struct
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[9]
_TARGET = (
    _ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/models/musespark_open"
    "/muse_spark_open_816b_a42b_nvfp4__sm_107__tep"
)
_HEAD_SAMPLER = _TARGET / "ops/spec/head_sampler.py"
_HEAD_SAMPLER_CU = _TARGET / "ops/spec/_head_sampler/head_sampler.cu"
_TARGET_HEAD = _TARGET / "kernels/split_sampler/target_head.py"

_HS_NAMES = {
    "MAX_ROWS",
    "VIRTUAL_THREADS",
    "VP_MIN_ROWS",
    "TOPP_ROUNDS",
    "TOPP_MAX_ROUNDS",
    "TOPP_RECORD_WORDS",
    "TOPP_PARTIAL_WORDS",
}
_HS_FUNCTIONS = {"configure", "topp_scratch_words", "topp_takes"}


def _host(path: Path, names: set, functions: set, **attrs) -> types.ModuleType:
    """The module-level assignments to ``names`` and the functions ``functions`` of ``path``, run alone."""
    tree = ast.parse(path.read_text())
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in names for t in node.targets
        ):
            body.append(node)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in names:
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            body.append(node)
    module = types.ModuleType(f"_{path.stem}_host")
    module.__dict__.update(attrs)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


def _hs_host() -> types.ModuleType:
    return _host(_HEAD_SAMPLER, _HS_NAMES, _HS_FUNCTIONS, Tuple=tuple)


# ---------------------------------------------------------------------------
# head_sampler: the knob, the scratch
# ---------------------------------------------------------------------------


def test_the_topp_rounds_default_off_and_configure_range():
    hs = _hs_host()
    assert hs.TOPP_ROUNDS == 0 and hs.TOPP_MAX_ROUNDS == 4
    hs.configure()
    assert hs.TOPP_ROUNDS == 0 and hs.VP_MIN_ROWS == 16
    for rounds in range(1, 5):
        hs.configure(topp_rounds=rounds)
        assert hs.TOPP_ROUNDS == rounds and hs.VP_MIN_ROWS == 16
    hs.configure(vp_min_rows=8, topp_rounds=2)
    assert (hs.VP_MIN_ROWS, hs.TOPP_ROUNDS) == (8, 2)
    for bad in (-1, 5, True, 1.0, "2"):
        with pytest.raises(ValueError, match="top-p test's rounds"):
            hs.configure(topp_rounds=bad)
    # A refused call changes nothing.
    assert (hs.VP_MIN_ROWS, hs.TOPP_ROUNDS) == (8, 2)
    hs.configure()
    assert (hs.VP_MIN_ROWS, hs.TOPP_ROUNDS) == (16, 0)


def test_the_topp_scratch_and_the_rows_it_takes():
    hs = _hs_host()
    assert hs.topp_scratch_words(128) == (128 * 32, 128 * 1024 * 2)
    assert hs.topp_scratch_words(1) == (32, 2048)
    for bad in (0, 129):
        with pytest.raises(ValueError, match="1..128 rows"):
            hs.topp_scratch_words(bad)
    assert not hs.topp_takes(96, 0)
    assert hs.topp_takes(96, 1) and hs.topp_takes(128, 4) and hs.topp_takes(1, 2)
    assert not hs.topp_takes(129, 2)  # wide calls keep kernel B (the 512-thread block)


# ---------------------------------------------------------------------------
# target_head.topp_setup: loud when the op is missing, the engagement log
# ---------------------------------------------------------------------------


class _Log:
    def __init__(self) -> None:
        self.records = []

    def info(self, message, *args, **kwargs) -> None:
        self.records.append(("info", str(message)))

    def error(self, message, *args, **kwargs) -> None:
        self.records.append(("error", str(message)))


class _Scratch:
    def __init__(self, n: int) -> None:
        self.n = n

    def numel(self) -> int:
        return self.n


def _topp_setup(has_op: bool):
    ops = types.SimpleNamespace()
    if has_op:
        ops.draw_topp = object()
    torch = types.SimpleNamespace(ops=types.SimpleNamespace(musespark_open_sm107_head_sampler=ops))
    made = []

    def new_topp_scratch(device):
        made.append(device)
        return _Scratch(128 * 32), _Scratch(128 * 1024 * 2)

    hs = types.SimpleNamespace(MAX_ROWS=128, new_topp_scratch=new_topp_scratch)
    log = _Log()
    host = _host(_TARGET_HEAD, set(), {"topp_setup"}, torch=torch, _hs=hs, logger=log)
    return host.topp_setup, made, log


def test_topp_setup_off_allocates_nothing():
    setup, made, log = _topp_setup(has_op=False)
    assert setup(0, "cuda:0") == (0, None, None)
    assert made == [] and log.records == []


def test_topp_setup_raises_when_the_library_has_no_draw_topp():
    setup, made, log = _topp_setup(has_op=False)
    with pytest.raises(RuntimeError, match="TOPP_ROUNDS=2.*no draw_topp op"):
        setup(2, "cuda:0")
    assert made == []


def test_topp_setup_allocates_and_logs_the_engagement():
    setup, made, log = _topp_setup(has_op=True)
    rounds, records, partials = setup(3, "cuda:0")
    assert rounds == 3 and records.numel() == 4096 and partials.numel() == 262144
    assert made == ["cuda:0"]
    (record,) = log.records
    assert record[0] == "info" and "head_sampler_coop_fold TOPP_ROUNDS=3" in record[1]
    assert "multi-CTA top-p test" in record[1]


def test_the_head_passes_the_scratch_to_split_sample():
    """``TargetHead.sample``'s split_sample call carries the knob and the scratch; the head builds them in
    ``__init__`` (before any capture) from ``head_sampler.TOPP_ROUNDS``."""
    tree = ast.parse(_TARGET_HEAD.read_text())
    (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TargetHead"]
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    init = ast.unparse(methods["__init__"])
    assert "topp_setup(_hs.TOPP_ROUNDS, self.device)" in init
    calls = [
        n
        for n in ast.walk(methods["sample"])
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "_hs.split_sample"
    ]
    (call,) = calls
    keywords = {k.arg: ast.unparse(k.value) for k in call.keywords}
    assert keywords["topp_rounds"] == "self.topp_rounds"
    assert keywords["topp_records"] == "self.topp_records"
    assert keywords["topp_partials"] == "self.topp_partials"


# ---------------------------------------------------------------------------
# head_sampler.cu against head_sampler.py
# ---------------------------------------------------------------------------


def _cu_constant(src: str, name: str) -> int:
    (value,) = re.findall(rf"constexpr int {name} = (\d+);", src)
    return int(value)


def test_the_cuda_constants_match_the_python_ones():
    src = _HEAD_SAMPLER_CU.read_text()
    hs = _hs_host()
    assert _cu_constant(src, "kTopPRecordBytes") == 4 * hs.TOPP_RECORD_WORDS
    assert _cu_constant(src, "kTopPMaxRounds") == hs.TOPP_MAX_ROUNDS
    assert hs.VIRTUAL_THREADS % _cu_constant(src, "kTopPCountThreads") == 0
    assert 'm.impl("draw_topp", &drawTopP);' in src


def test_the_draw_topp_schema_matches_its_call():
    src = _HEAD_SAMPLER_CU.read_text()
    (schema,) = re.findall(r'"draw_topp\((.*?)\) -> Tensor"\);', src.replace('"\n        "', ""), re.S)
    params = [p.strip().split()[-1].split("=")[0] for p in schema.split(",")]
    assert params == [
        "logits",
        "temperatures",
        "top_ks",
        "top_ps",
        "min_ps",
        "seed",
        "offset",
        "states",
        "records",
        "partials",
        "multiplier",
        "inv_cap",
        "cap",
        "rounds",
        "pdl",
        "early_trigger",
    ]
    tree = ast.parse(_HEAD_SAMPLER.read_text())
    (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SplitSample"]
    (call_fn,) = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__"]
    (call,) = [
        n
        for n in ast.walk(call_fn)
        if isinstance(n, ast.Call) and ast.unparse(n.func).endswith(".draw_topp")
    ]
    assert [ast.unparse(a) for a in call.args] == [
        "logits",
        "temperatures",
        "top_ks",
        "top_ps",
        "min_ps",
        "seed",
        "offset",
        "states",
        "topp_records",
        "topp_partials",
        "m",
        "inv",
        "cap",
        "topp_rounds",
        "pdl",
        "verify_pdl and pdl",
    ]


def test_the_test_sweep_uses_the_shipped_statements():
    """The sweep kernel's and the in-CTA sweep's weight, membership and accumulation are the shipped
    CountMass sweep's (fusedSamplingBody): the same expression, ``w >= floor(0) && w > candWeight``,
    count then mass."""
    src = _HEAD_SAMPLER_CU.read_text()
    weight = "__expf(__fmul_rn(capOf(__bfloat162float(elems[j]), cap), tempInv) - maxScaled)"
    count = src.split("splitTopPCountKernel(FusedSamplingParams")[1].split("partials[static_cast")[0]
    decide = src.split("splitTopPDecideKernel(FusedSamplingParams")[1].split(
        "if constexpr (!kFinal)\n    {\n        gridDependencyTrigger"
    )[0]
    for body in (count, decide):
        assert f"float const w = {weight};" in body
        assert "if (w >= 0.0f && w > candWeight)" in body
        assert body.index("local.count += 1.0f;") < body.index("local.mass += w;")
        # The skip leaves the partial at its initial {0, 0} and is decided before any row read.
        assert "CountMass{0.0f, 0.0f}" in body or "CountMass local{0.0f, 0.0f};" in body
        assert body.index("splitTopPSkips(states[") < body.index(f"float const w = {weight};")
        assert "!skip && v" in body
    shipped = (_TARGET / "ops/spec/_head_sampler/fusedSamplingKernelsCommon.cuh").read_text()
    assert "if (w >= supportFloor && w > candWeight)" in shipped
    assert "float const supportFloor = threshold;" in shipped


def test_the_sweep_skip_is_guarded():
    """splitTopPSkips: NaN-guarded (a NaN element makes the fold's sum NaN and its max no bound), a strict
    comparison against the candidate's weight with a margin far above __expf's error."""
    src = _HEAD_SAMPLER_CU.read_text()
    (margin,) = re.findall(r"constexpr float kTopPSkipMargin = ([0-9.]+)f;", src)
    assert float(margin) == 1.0 + 2.0**-10
    body = src.split("__device__ __forceinline__ bool splitTopPSkips(")[1].split("}")[0]
    assert "st.sum == st.sum && __expf(st.max - maxScaled) * kTopPSkipMargin < candWeight" in body


# ---------------------------------------------------------------------------
# The float32 model: the shipped loop against the restructured one
# ---------------------------------------------------------------------------

_MAX_REJECT_ROUNDS = 32


def _f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


def _sum(values) -> float:
    acc = 0.0
    for v in values:
        acc = _f32(acc + v)
    return acc


def _local(ws, above: float) -> float:
    """A thread's mass over its elements w >= 0 && w > above, in its order (both sweeps' statement)."""
    return _sum(w for w in ws if w >= 0.0 and w > above)


def _reduce(values) -> float:
    """Stands for cub's block reduction: a fixed pairwise tree (the same in both models)."""
    values = list(values)
    while len(values) > 1:
        values = [
            _f32(values[i] + values[i + 1]) if i + 1 < len(values) else values[i]
            for i in range(0, len(values), 2)
        ]
    return values[0]


def _scan(values):
    bases, acc = [], 0.0
    for v in values:
        bases.append(acc)
        acc = _f32(acc + v)
    return bases, acc


def _draw(threads, local, pivot, uniform, vb):
    """Scan, target, claimants' walk (kernel B's rule): (candidate index, weight) or None."""
    bases, support = _scan(local)
    target = _f32(uniform * support)
    best = None  # ((position, thread + 1), index, w): the later crossing, then the higher thread
    for t, ws in enumerate(threads):
        if not (bases[t] <= target < _f32(bases[t] + local[t])):
            continue
        running = bases[t]
        for position, w in enumerate(ws, 1):
            if not (w >= 0.0 and w > pivot):
                continue
            running = _f32(running + w)
            if running > target:
                index = (t + vb * ((position - 1) // 4)) * 4 + (position - 1) % 4
                if best is None or (position, t + 1) > best[0]:
                    best = ((position, t + 1), index, w)
                break
    if best is None:
        return None, support
    return (best[1], best[2]), support


def _shipped(threads, top_p, uniforms, argmax, vb):
    """fusedSamplingBody's tokens-only top-p rejection loop (one CTA): two sweeps per round."""
    pivot, mass_target = -1.0, None
    for rnd in range(_MAX_REJECT_ROUNDS):
        local = [_local(ws, pivot) for ws in threads]
        drawn, support = _draw(threads, local, pivot, next(uniforms), vb)
        if rnd == 0:
            mass_target = _f32(top_p * support)
        if drawn is None:
            return argmax
        cand, cand_w = drawn
        above = _reduce(_local(ws, cand_w) for ws in threads)
        if above <= mass_target:
            return cand
        pivot = cand_w
    return argmax


def _restructured(threads, top_p, uniforms, argmax, vb, rounds):
    """draw_topp: kernel B's round 0 and record; ``rounds`` sweep / decide pairs, the last finishing in-CTA."""
    local = [_local(ws, -1.0) for ws in threads]
    drawn, support = _draw(threads, local, -1.0, next(uniforms), vb)
    mass_target = _f32(top_p * support)
    if drawn is None:
        return argmax
    cand, cand_w = drawn
    rnd = 0
    for k in range(rounds):
        final = k == rounds - 1
        partials = [_local(ws, cand_w) for ws in threads]  # the sweep kernel (one per virtual thread)
        while True:
            if _reduce(partials) <= mass_target:
                return cand
            pivot = cand_w
            rnd += 1
            if rnd >= _MAX_REJECT_ROUNDS:
                return argmax
            drawn, _ = _draw(threads, partials, pivot, next(uniforms), vb)  # no sweep: the partials' mass
            if drawn is None:
                return argmax
            cand, cand_w = drawn
            if not final:
                break  # the record; the next sweep kernel
            partials = [_local(ws, cand_w) for ws in threads]  # the in-CTA sweep
    raise AssertionError("unreachable")


def _row(rng: random.Random, vb: int, vecs_per_thread: int, sharp: float):
    """Weights w in (0, 1] (the max element's is 1), thread t's elements of vectors t + vb j."""
    n = vb * vecs_per_thread * 4
    logits = [rng.gauss(0.0, 1.0) * sharp for _ in range(n)]
    top = max(range(n), key=lambda i: logits[i])
    ws = [_f32(math.exp(logits[i] - logits[top])) for i in range(n)]
    threads = [
        [ws[(t + vb * j) * 4 + e] for j in range(vecs_per_thread) for e in range(4)] for t in range(vb)
    ]
    return threads, top


class _Uniforms:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.used = 0

    def __iter__(self):
        return self

    def __next__(self) -> float:
        self.used += 1
        return _f32(1.0 - self.rng.random())  # curand_uniform: (0, 1]


@pytest.mark.parametrize("rounds", [1, 2, 3, 4])
def test_the_restructured_loop_draws_the_shipped_token(rounds):
    rng = random.Random(1234 + rounds)
    vb, vecs = 16, 3
    rejected_twice = 0
    for case in range(400):
        sharp = rng.choice([0.3, 1.0, 3.0, 8.0])
        top_p = rng.choice([0.05, 0.3, 0.6, 0.9, 0.95, 0.99])
        threads, argmax = _row(rng, vb, vecs, sharp)
        seed = rng.randrange(1 << 30)
        u_ref, u_new = _Uniforms(random.Random(seed)), _Uniforms(random.Random(seed))
        ref = _shipped(threads, top_p, u_ref, argmax, vb)
        new = _restructured(threads, top_p, u_new, argmax, vb, rounds)
        assert new == ref, (case, rounds, top_p, sharp)
        # The same Philox draws, so the row's stream ends where the shipped loop's does.
        assert u_new.used == u_ref.used, (case, rounds)
        rejected_twice += u_ref.used >= 3
    assert rejected_twice > 20  # the in-CTA continuation and the later pairs are exercised


# ---------------------------------------------------------------------------
# GPU (SM107): bit for bit against kernel B, and the timing
# ---------------------------------------------------------------------------

VOCAB = 202048  # the target's vocabulary (8 shards of 25256)
#: Soft-cap constants for the GPU cases (any FP32 values: both paths read the same ones).
_MULTIPLIER = 1.0
_SOFT_CAP = 30.0


def _gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the head sampler library is built for SM107")
    import importlib

    hs = importlib.import_module(
        "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open."
        "muse_spark_open_816b_a42b_nvfp4__sm_107__tep.ops.spec.head_sampler"
    )
    hs.load_library()
    return torch, hs


def _inputs(torch, rows, mode, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    scale = {"flat": 0.5, "served": 4.0, "sharp": 12.0}[mode]
    logits = (torch.randn(rows, VOCAB, device="cuda", generator=g) * scale).to(torch.bfloat16)
    temps = torch.ones(rows, device="cuda")
    top_ks = torch.zeros(rows, dtype=torch.int32, device="cuda")
    top_ps = torch.full((rows,), 0.95, device="cuda")
    min_ps = torch.zeros(rows, device="cuda")
    if mode == "flat":
        top_ps.fill_(0.3)  # many rejections, several rounds
    seeds = torch.randint(0, 1 << 40, (rows,), device="cuda", generator=g)
    offsets = torch.randint(0, 1 << 20, (rows,), device="cuda", generator=g)
    return logits, temps, top_ks, top_ps, min_ps, seeds, offsets


@pytest.mark.parametrize("rows", [8, 16, 96, 128])
@pytest.mark.parametrize("mode", ["served", "flat", "sharp", "mixed"])
def test_gpu_draw_topp_matches_kernel_b(rows, mode):
    torch, hs = _gpu()
    states = hs.new_states(rows, torch.device("cuda"))
    records, partials = hs.new_topp_scratch(torch.device("cuda"))
    for seed in range(4):
        logits, temps, top_ks, top_ps, min_ps, seeds, offsets = _inputs(
            torch, rows, "served" if mode == "mixed" else mode, seed
        )
        if mode == "mixed":
            # neutral, top-k + top-p, min-p + top-p, low temperature and top-p-only rows in one call
            top_ps[0::5] = 1.0
            top_ks[1::5] = 50
            min_ps[2::5] = 0.05
            temps[3::5] = 0.6
        args = (logits, temps, top_ks, top_ps, min_ps, seeds, offsets, _MULTIPLIER, _SOFT_CAP, states)
        ref = hs.split_sample(*args)
        for rounds in range(1, hs.TOPP_MAX_ROUNDS + 1):
            got = hs.split_sample(*args, topp_rounds=rounds, topp_records=records, topp_partials=partials)
            assert torch.equal(got, ref), (rows, mode, seed, rounds)


def _bench(rows_list=(16, 96, 128), iters=200):
    torch, hs = _gpu()
    dev = torch.device("cuda")
    states = hs.new_states(128, dev)
    records, partials = hs.new_topp_scratch(dev)
    for rows in rows_list:
        logits, temps, top_ks, top_ps, min_ps, seeds, offsets = _inputs(torch, rows, "served", 0)
        line = [f"rows {rows}:"]
        for rounds in range(0, hs.TOPP_MAX_ROUNDS + 1):

            def step():
                return hs.split_sample(
                    logits,
                    temps,
                    top_ks,
                    top_ps,
                    min_ps,
                    seeds,
                    offsets,
                    _MULTIPLIER,
                    _SOFT_CAP,
                    states,
                    pdl=True,
                    topp_rounds=rounds,
                    topp_records=records,
                    topp_partials=partials,
                )

            step()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            graph.replay()
            start.record()
            for _ in range(iters):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            line.append(f"R{rounds} {start.elapsed_time(end) * 1000 / iters:.1f} us")
        print(" ".join(line))


if __name__ == "__main__":
    import sys

    if "--bench" in sys.argv:
        _bench()

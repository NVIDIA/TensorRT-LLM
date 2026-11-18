# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The distributed verify sampler (``head_sampler_coop_fold``'s DIST_MIN_ROWS, ``head_sampler.ds_sample``).

CPU (no torch, no device):

* ``head_sampler.configure`` (the floor's range), ``DistLayout`` (the LL area and staging layout) and
  ``target_head.dist_setup`` (off / refused next to vocab_parallel_head / built and logged), loaded from their
  files' ASTs;
* ``head_sampler.cu`` against ``head_sampler.py``: the LL sizes, the ranks, the op schema and its call;
* ``TargetHead``'s routing (the distributed step before the vocabulary-parallel one; greedy batches and other
  consumers gather the shards first);
* the index map the kernel relies on: rank r's vectors i (global v = world i + r) are exactly the vectors of its
  virtual threads vt = world a + r, in kernel A's order (i = a + L j), on the contiguous shard v // shard_vecs.

GPU (SM107; not run on CPU): ``DistEmulation`` (8 ranks' CTAs in one launch on one GPU, rows <= SMs / 8) against
``split_sample`` on the gathered row (today's path) bit for bit, on every emulated rank, over consecutive calls
(both call parities, changing row counts), served / flat / sharp / mixed rows (the top-k / min-p fallback).
"""

from __future__ import annotations

import ast
import re
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
    "WALK_VECTORS",
    "VP_MIN_ROWS",
    "TOPP_ROUNDS",
    "TOPP_MAX_ROUNDS",
    "DIST_MIN_ROWS",
    "DIST_FLOOR",
    "DIST_MAX_ROWS",
    "DIST_MAX_RANKS",
    "DIST_STATE_BYTES",
    "DIST_PAIR_BYTES",
    "DIST_SLOT_BYTES",
    "DIST_STAMPS",
    "DIST_FABRIC",
}


def _host(
    path: Path, names: set, functions: set, classes: set = frozenset(), **attrs
) -> types.ModuleType:
    tree = ast.parse(path.read_text())
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in names for t in node.targets
        ):
            body.append(node)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in names
        ):
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in classes:
            body.append(node)
    module = types.ModuleType(f"_{path.stem}_host")
    module.__dict__.update(attrs)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


def _hs_host() -> types.ModuleType:
    return _host(
        _HEAD_SAMPLER,
        _HS_NAMES,
        {"configure", "_neutral_or_filtered_rows_ok"},
        {"DistLayout"},
        Tuple=tuple,
    )


# ---------------------------------------------------------------------------
# configure, the layout
# ---------------------------------------------------------------------------


def test_the_dist_floor_defaults_off_and_its_range():
    hs = _hs_host()
    assert hs.DIST_MIN_ROWS == 0 and (hs.DIST_FLOOR, hs.DIST_MAX_ROWS) == (8, 128)
    for floor in (8, 9, 16, 64, 128):
        hs.configure(dist_min_rows=floor)
        assert hs.DIST_MIN_ROWS == floor and hs.TOPP_ROUNDS == 0 and hs.VP_MIN_ROWS == 16
    for bad in (1, 7, 129, -8, True, 16.0):
        with pytest.raises(ValueError, match="distributed sampler's row floor"):
            hs.configure(dist_min_rows=bad)
    assert hs.DIST_MIN_ROWS == 128  # a refused call changes nothing
    hs.configure(topp_rounds=2, dist_min_rows=16)
    assert (hs.TOPP_ROUNDS, hs.DIST_MIN_ROWS) == (2, 16)
    hs.configure(dist_min_rows=16, dist_fabric=True)
    assert hs.DIST_FABRIC is True
    with pytest.raises(ValueError, match="fabric switch"):
        hs.configure(dist_fabric=1)
    hs.configure()
    assert (hs.VP_MIN_ROWS, hs.TOPP_ROUNDS, hs.DIST_MIN_ROWS, hs.DIST_FABRIC) == (16, 0, 0, False)


def test_the_dist_layout():
    hs = _hs_host()
    lay = hs.DistLayout(8, 25256, 128)
    assert (lay.vocab, lay.shard_vecs) == (202048, 6314)
    assert lay.ready_offset == 0
    assert lay.states_offset == 2 * 128 * 8 * 16
    assert lay.cand_offset == lay.states_offset + 2 * 128 * 1024 * 32
    assert lay.partial_offset == lay.cand_offset + 4 * 128 * 8 * 32
    assert lay.nbytes == lay.partial_offset + 4 * 128 * 1024 * 16
    for off in (lay.states_offset, lay.cand_offset, lay.partial_offset, lay.nbytes):
        assert off % 16 == 0
    assert lay.staging_bytes == 2 * 128 * 25256 * 2
    assert hs.DistLayout(8, 25256, 128, ll_staging=True).staging_bytes == 2 * lay.staging_bytes
    assert (
        lay.smem_bytes == 50 * 128 * 16 <= 227 * 1024 - 48 * 1024
    )  # 50 stripes of 1024 vectors, 128 slots each
    assert set(lay.offsets()) == {"ready_offset", "states_offset", "cand_offset", "partial_offset"}
    for bad in ((3, 25256, 128), (8, 25255, 128), (8, 25256, 129), (8, 25256, 0), (16, 25256, 128)):
        with pytest.raises(ValueError):
            hs.DistLayout(*bad)
    with pytest.raises(ValueError, match="does not fit the walk"):
        hs.DistLayout(8, 4 * 64 * 1024 // 8 + 4, 16)


# ---------------------------------------------------------------------------
# target_head.dist_setup and the head's routing
# ---------------------------------------------------------------------------


class _Log:
    def __init__(self) -> None:
        self.records = []

    def info(self, message, *args, **kwargs) -> None:
        self.records.append(("info", str(message)))


class _Workspace:
    built = []

    def __init__(self, mapping, shard, max_rows, ll_staging=False) -> None:
        _Workspace.built.append((mapping, shard, max_rows))
        self.max_rows = max_rows

    def nbytes(self) -> int:
        return 29 * 2**20


def _dist_setup():
    _Workspace.built = []
    hs = types.SimpleNamespace(DistWorkspace=_Workspace, DIST_MAX_ROWS=128, DIST_FABRIC=False)
    log = _Log()
    host = _host(_TARGET_HEAD, set(), {"dist_setup"}, _hs=hs, logger=log)
    return host.dist_setup, log


def test_dist_setup_off():
    setup, log = _dist_setup()
    assert setup(0, "mapping", 25256, False) == (0, None)
    assert setup(0, "mapping", 25256, True) == (0, None)
    assert _Workspace.built == [] and log.records == []


def test_dist_setup_refuses_the_vocab_parallel_head():
    setup, log = _dist_setup()
    with pytest.raises(ValueError, match="DIST_MIN_ROWS=16.*vocab_parallel_head"):
        setup(16, "mapping", 25256, True)
    assert _Workspace.built == []


def test_dist_setup_builds_and_logs():
    setup, log = _dist_setup()
    floor, workspace = setup(16, "mapping", 25256, False)
    assert floor == 16 and isinstance(workspace, _Workspace)
    assert _Workspace.built == [("mapping", 25256, 128)]
    ((level, message),) = log.records
    assert level == "info" and "head_sampler_coop_fold DIST_MIN_ROWS=16" in message
    assert "decode steps of 16..128 rows" in message and "no logits all-gather" in message


def _methods():
    tree = ast.parse(_TARGET_HEAD.read_text())
    (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TargetHead"]
    return {n.name: ast.unparse(n) for n in cls.body if isinstance(n, ast.FunctionDef)}


def test_the_head_routes_dist_steps():
    m = _methods()
    init = m["__init__"]
    assert (
        "dist_setup(_hs.DIST_MIN_ROWS, mapping, int(lm_head.weight.shape[0]), vocab_parallel)"
        in init
    )
    produce = m["produce"]
    # The distributed step comes first (it also takes 8-row steps at a floor of 8), with the built-in local GEMM.
    assert produce.index("self.dist_min_rows <= rows <= self.dist.max_rows") < produce.index(
        "self.vp is not None"
    )
    assert produce.index("self.dist_min_rows <= rows") < produce.index(
        "self.workspace is not None and rows == "
    )
    dist_branch = produce.split("self.dist_min_rows <= rows <= self.dist.max_rows")[1].split(
        "return logits"
    )[0]
    assert "lm_head.apply_linear(hidden_states, lm_head.bias, None, None)" in dist_branch
    assert "self.pending_dist = True" in dist_branch
    sample = m["sample"]
    assert (
        sample.index("if dist_row:")
        < sample.index("if s2_row:")
        < sample.index("if local is not None:")
    )
    assert "_hs.dist_sample(self.dist.view, logits, local" in sample.replace("\n", " ").replace(
        "    ", ""
    )
    # Greedy batches do not take the vocabulary-parallel argmax: they gather the shards (materialize).
    assert "not self.pending_dist" in m["vocab_parallel"]
    for name in ("_gather_now", "drain"):
        assert "self.pending_dist = False" in m[name], name


# ---------------------------------------------------------------------------
# head_sampler.cu against head_sampler.py
# ---------------------------------------------------------------------------


def _cu() -> str:
    return _HEAD_SAMPLER_CU.read_text()


def test_the_cuda_constants_match():
    src = _cu()
    hs = _hs_host()
    assert re.search(r"constexpr int kDsMaxRanks = (\d+);", src).group(1) == str(hs.DIST_MAX_RANKS)
    for name, value in (
        ("kDsStateBytes", hs.DIST_STATE_BYTES),
        ("kDsPairBytes", hs.DIST_PAIR_BYTES),
        ("kDsSlotBytes", hs.DIST_SLOT_BYTES),
    ):
        assert re.search(rf"constexpr unsigned long long {name} = (\d+);", src).group(1) == str(
            value
        ), name
    assert 'm.impl("ds_sample", &dsSample);' in src
    assert re.search(r"constexpr int kDsStamps = (\d+);", src).group(1) == str(hs.DIST_STAMPS)
    # the stamps exist only in the instrumented instantiation (the served one compiles none)
    assert (
        "if constexpr (kStamp)" in src
        and "template <bool kTrigger, bool kStamp = false, bool kFabric = false>" in src
    )


def test_the_ds_sample_schema_matches_its_call():
    src = _cu().replace('"\n        "', "")
    (schema,) = re.findall(r'"ds_sample\((.*?)\) -> Tensor"\);', src, re.S)
    params = [p.strip().split()[-1].split("=")[0] for p in schema.split(",")]
    tree = ast.parse(_HEAD_SAMPLER.read_text())
    (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_DistRank"]
    (sample,) = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "sample"]
    (call,) = [
        n
        for n in ast.walk(sample)
        if isinstance(n, ast.Call) and ast.unparse(n.func).endswith(".ds_sample")
    ]
    args = [ast.unparse(a) for a in call.args]
    assert len(params) == len(args) == 29
    assert args == [
        "local",
        "pending",
        "temperatures",
        "top_ks",
        "top_ps",
        "min_ps",
        "seed",
        "offset",
        "self.staging",
        "self.publish",
        "self.copies",
        "self.counters",
        "self.status",
        "self.rank",
        "lay.ranks",
        "self.emulated",
        "lay.max_rows",
        "lay.ready_offset",
        "lay.states_offset",
        "lay.cand_offset",
        "lay.partial_offset",
        "POLL_TIMEOUT_S",
        "m",
        "inv",
        "cap",
        "pdl",
        "early_trigger",
        "stamps",
        "bool(fabric)",
    ]
    assert params[:8] == [
        "local",
        "pending",
        "temperatures",
        "top_ks",
        "top_ps",
        "min_ps",
        "seed",
        "offset",
    ]
    assert params[-4:] == ["pdl", "early_trigger", "stamps", "fabric"]


def test_the_kernel_uses_kernel_a_and_b_statements():
    src = _cu()
    body = src.split("__global__ void __launch_bounds__(kDsThreads, 1) dsSampleKernel(")[1].split(
        "} // namespace fusedSampling"
    )[0]
    # kernel A's scaled logit and fold statement, kernel B's rebase / draw / test statements
    assert "__fmul_rn(capOf(__bfloat162float(elems[0]), p.cap), rp.tempInv)" in body
    # the fold: splitFoldCoopKernel's statements for combineOnlineSoftmax(st, {x, 1, index}) (FORM 1, one loop)
    coop = src.split("splitFoldCoopKernel(SplitFoldParams p)")[1].split("p.states[")[0]
    for stmt in (
        "bool const keep = runMax >= xs[e];",
        "float const w = keep ? __expf(xs[e] - runMax) : __expf(runMax - xs[e]);",
        "runArg = keep ? runArg : base + e;",
        "runMax = keep ? runMax : xs[e];",
    ):
        assert stmt in coop and stmt in body, stmt
    assert "runSum = keep ? runSum + 1.0f * w : 1.0f + runSum * w;" in body
    assert "runSum = keep ? runSum + 1.0f * w : 1.0f + runSum * w;" in coop
    assert (
        "float runMax = -FLT_MAX;" in body
        and "float runSum = 0.0f;" in body
        and "int runArg = 0;" in body
    )
    assert "cachedThreadStats.sum * __expf(cachedThreadStats.max - maxScaled)" in body
    assert "shared.massTarget = rp.topP * supportMass;" in body
    assert "shared.target = curand_uniform(&shared.rejectRng) * supportMass;" in body
    assert "if (w >= 0.0f && w > candWeight)" in body
    assert "sKeepToken = above.mass <= shared.massTarget ? 1 : 0;" in body
    assert "BlockReduceCountMass(shared.temp.reduceCountMass).Reduce(part, CountMassOp())" in body
    assert "fusedSamplingBody<CapBf16, kDsThreads, true, false>(fs, shared);" in body
    assert "if (filtered && !splitTopPOnly(rp))" in body
    # the fabric variant: LL staging written with the call's flag, pulled with an inline flag check (no fence /
    # ready flag on that path); the third state word as one LL element
    assert "template <bool kTrigger, bool kStamp = false, bool kFabric = false>" in src
    assert (
        "dsStorePair(mineLL + static_cast<unsigned long long>(v) * 16u, tag | w.x, tag | w.y);"
        in body
    )
    assert body.count("if (!dsFresh(a[u], b[u], flag0))") == 2  # the pull and the fallback gather
    assert "dsAwaitPair(llAt(v), flag0, p.status, p.timeoutNs, a[u], b[u]);" in body
    ready = body.split("if constexpr (!kFabric)\n    {\n        // The CTA's staging stores")[
        1
    ].split("stamp(2);")[0]
    assert "__threadfence_system();" in ready and "dsAwaitPair(copy + readyRow" in ready
    assert "dsPublishOne(p, at + 16u, dsTag(flag0) | static_cast<uint32_t>(own.argmax));" in body
    assert "e2 = dsAwaitOne(at + 16u, flag0, p.status, p.timeoutNs);" in body
    # every poll has a timeout; the epoch skips the zeroed flag
    assert "dsTimedOut(status)" in src and "(epoch & 0x07FFFFFFu) == 0u" in body


# ---------------------------------------------------------------------------
# The index map
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_rank_vectors_are_its_virtual_threads_in_kernel_a_order(world):
    """Rank r owns vt = L r + a; its shared-memory slot i = L j + a holds global vector v = 1024 j + L r + a (runs of
    L contiguous vectors, so the peers' staging is read coalesced); thread t reads entry t (owner t // L)."""
    vb = 1024
    for vec_count in (50512, 1024 * 7 + 336, 1024 * 3):
        if vec_count % world:
            continue
        sv = vec_count // world
        local_threads = vb // world
        stripes = -(-vec_count // vb)
        seen = set()
        for r in range(world):
            slot_vector = {
                i: (i // local_threads) * vb + local_threads * r + i % local_threads
                for i in range(stripes * local_threads)
            }
            for a in range(local_threads):
                vt = local_threads * r + a
                served = list(range(vt, vec_count, vb))  # kernel A: v = vt + 1024 j, increasing
                # the fold / sweep: slot a + L j while vt + 1024 j < vec_count
                ours = [
                    slot_vector[a + local_threads * j]
                    for j in range(stripes)
                    if vt + vb * j < vec_count
                ]
                assert ours == served, (world, vec_count, r, a)
                # the walk: element j of the walker at slot a + L j, j < ceil((vec_count - vt) / 1024)
                walk = -(-(vec_count - vt) // vb)
                assert [slot_vector[a + local_threads * j] for j in range(walk)] == served
                for v in ours:
                    q, off = divmod(v, sv)  # the contiguous shard holding v
                    assert 0 <= q < world and 0 <= off < sv
                    seen.add(v)
            # pulled slots in thread order are runs of L contiguous vectors
            run = [slot_vector[i] for i in range(local_threads)]
            assert run == list(range(local_threads * r, local_threads * (r + 1)))
        assert seen == set(range(vec_count))  # every vector once, over the ranks
        for t in range(
            vb
        ):  # the reader: thread t is virtual thread t, owner t // L, entry (owner, t % L) == t
            assert (t // local_threads) * local_threads + t % local_threads == t


# ---------------------------------------------------------------------------
# GPU (SM107): the emulation against today's split sampler, bit for bit
# ---------------------------------------------------------------------------

VOCAB = 202048
SHARD = VOCAB // 8
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
    scale = {"flat": 0.5, "served": 4.0, "sharp": 12.0, "mixed": 4.0}[mode]
    logits = (torch.randn(rows, VOCAB, device="cuda", generator=g) * scale).to(torch.bfloat16)
    temps = torch.ones(rows, device="cuda")
    top_ks = torch.zeros(rows, dtype=torch.int32, device="cuda")
    top_ps = torch.full((rows,), 0.95, device="cuda")
    min_ps = torch.zeros(rows, device="cuda")
    if mode == "flat":
        top_ps.fill_(0.3)  # many rejections, several rounds
    if mode == "mixed":
        top_ps[0::5] = 1.0  # neutral
        top_ks[1::5] = 50  # top-k + top-p: the exact fallback
        min_ps[2::5] = 0.05  # min-p + top-p: the exact fallback
        temps[3::5] = 0.6
    seeds = torch.randint(0, 1 << 40, (rows,), device="cuda", generator=g)
    offsets = torch.randint(0, 1 << 20, (rows,), device="cuda", generator=g)
    return logits, temps, top_ks, top_ps, min_ps, seeds, offsets


def test_gpu_dist_emulation_matches_the_split_sampler():
    torch, hs = _gpu()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    most = min(16, sms // 8)
    emus = {f: hs.DistEmulation(8, SHARD, most, "cuda", ll_staging=f) for f in (False, True)}
    states = hs.new_states(most, torch.device("cuda"))
    calls = 0
    for mode in ("served", "flat", "sharp", "mixed"):
        for rows in sorted({1, most // 2 or 1, most, 8 if most >= 8 else most}):
            for seed in range(3):
                logits, temps, top_ks, top_ps, min_ps, seeds, offsets = _inputs(
                    torch, rows, mode, 100 * calls + seed
                )
                ref = hs.split_sample(
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
                )
                locals_ = [logits[:, r * SHARD : (r + 1) * SHARD].contiguous() for r in range(8)]
                for fabric in (False, True):
                    emu = emus[fabric]
                    tokens, pending = emu.sample(
                        locals_,
                        temps,
                        top_ks,
                        top_ps,
                        min_ps,
                        seeds,
                        offsets,
                        _MULTIPLIER,
                        _SOFT_CAP,
                        fabric=fabric,
                    )
                    torch.cuda.synchronize()
                    assert emu.status.item() == 0
                    for r in range(8):
                        assert torch.equal(tokens[r], ref), (mode, rows, seed, r, fabric)
                    calls += 1
    assert calls >= 48  # both call parities, several epochs per row, both forms


def test_gpu_dist_emulation_fallback_rows_gather_the_row():
    torch, hs = _gpu()
    for fabric in (False, True):
        emu = hs.DistEmulation(8, SHARD, 8, "cuda", ll_staging=fabric)
        logits, temps, top_ks, top_ps, min_ps, seeds, offsets = _inputs(torch, 8, "mixed", 7)
        top_ks.fill_(40)  # every row: top-k + top-p
        locals_ = [logits[:, r * SHARD : (r + 1) * SHARD].contiguous() for r in range(8)]
        tokens, pending = emu.sample(
            locals_,
            temps,
            top_ks,
            top_ps,
            min_ps,
            seeds,
            offsets,
            _MULTIPLIER,
            _SOFT_CAP,
            fabric=fabric,
        )
        torch.cuda.synchronize()
        for r in range(8):
            assert torch.equal(pending[r], logits), (
                r,
                fabric,
            )  # the gathered row, vocabulary order, every rank


def _bench(iters=200):
    """Graph-replay us of ds_sample in emulation (8 ranks' CTAs in one launch: the exchanges stay on one GPU, so
    this is the kernel's floor, not its fabric time) vs split_sample on the gathered row (fold + kernel B)."""
    torch, hs = _gpu()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    most = min(16, sms // 8)
    emus = {f: hs.DistEmulation(8, SHARD, most, "cuda", ll_staging=f) for f in (False, True)}
    states = hs.new_states(most, torch.device("cuda"))
    for rows in sorted({1, 8 if most >= 8 else most, most}):
        for mode in ("served", "sharp", "flat"):
            logits, temps, top_ks, top_ps, min_ps, seeds, offsets = _inputs(torch, rows, mode, 5)
            locals_ = [logits[:, r * SHARD : (r + 1) * SHARD].contiguous() for r in range(8)]
            local = torch.cat(locals_, dim=0).contiguous()
            pending = torch.zeros((8 * rows, VOCAB), dtype=torch.bfloat16, device="cuda")

            def dist():
                return emus[False].view.sample(
                    local,
                    pending,
                    temps,
                    top_ks,
                    top_ps,
                    min_ps,
                    seeds,
                    offsets,
                    _MULTIPLIER,
                    _SOFT_CAP,
                    True,
                    False,
                )

            def dist_fabric():
                return emus[True].view.sample(
                    local,
                    pending,
                    temps,
                    top_ks,
                    top_ps,
                    min_ps,
                    seeds,
                    offsets,
                    _MULTIPLIER,
                    _SOFT_CAP,
                    True,
                    False,
                    None,
                    True,
                )

            def today():
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
                )

            line = [f"rows {rows} {mode}:"]
            for name, fn in (
                ("split_sample", today),
                ("ds_sample(emulated x8)", dist),
                ("fabric", dist_fabric),
            ):
                fn()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    fn()
                graph.replay()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(iters):
                    graph.replay()
                end.record()
                torch.cuda.synchronize()
                line.append(f"{name} {start.elapsed_time(end) * 1000 / iters:.1f} us")
            print(" ".join(line), flush=True)


if __name__ == "__main__":
    import sys

    if "--bench" in sys.argv:
        _bench()

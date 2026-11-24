# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Distributed autotuning must keep the ranks of a tuning group in lockstep.

choose_one runs host collectives (the per-op tactic merge or broadcast) only on
its cache-miss branch. Weak progress can leave the ranks out of step, which is
reproduced here on CPU with four forked ranks: MPI only progresses a send while
the sender is inside an MPI call. A rank that returns from the merge with its
last bytes to a peer still queued, and then waits in a device collective,
leaves that peer in the merge for good (the exchange barrier keeps every rank
in a small collective until the merge completed everywhere). The test transport
models that: a send it marks as stranded is delivered only on the sender's next
call.

The device collective after the tuned call is modeled by a collective the
tuner never touches (a second gloo group, or a multiprocessing barrier), so a
missing rank shows up as a timeout instead of a hang.
"""

import faulthandler
import os
import pickle
import sys
import tempfile
import time
import traceback
from datetime import timedelta

import pytest
import torch

from tensorrt_llm._torch.autotuner import (
    AutoTuner,
    DistributedTuningStrategy,
    DynamicTensorSpec,
    TunableRunner,
    TuningConfig,
)

WORLD = 4
N_TACTICS = 8
BUCKETS = (1, 2, 4, 8, 16, 32)
TIMEOUT_S = 12.0

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="forks the ranks"),
]


def _last_pow2(x: int) -> int:
    p = 1
    while p * 2 <= min(x, BUCKETS[-1]):
        p *= 2
    return p


class _Mapping:
    """The mapping fields the tuner reads: a pure TP tuning group."""

    def __init__(self, rank: int, size: int):
        self.rank = rank
        self.tp_rank = rank
        self.tp_size = size
        self.cp_rank = 0
        self.cp_size = 1
        self.world_size = size

    def has_cp_helix(self):
        return False

    def has_pp(self):
        return False


class _GlooDist:
    """Distributed over a gloo group; counts the collectives it runs."""

    def __init__(self, group):
        self.group = group
        self.collectives = []

    def tp_cp_allgather(self, obj, *, small_payload: bool = False):
        import torch.distributed as dist

        self.collectives.append(("allgather", small_payload))
        out = [None] * dist.get_world_size(self.group)
        dist.all_gather_object(out, obj, group=self.group)
        return out

    def tp_cp_broadcast(self, obj, root=0, **kwargs):
        import torch.distributed as dist

        self.collectives.append(("broadcast", False))
        box = [obj]
        dist.broadcast_object_list(box, src=root, group=self.group)
        return box[0]


class _WeakProgressDist:
    """File-backed allgather with MPI's weak-progress rule.

    A rank delivers its contribution to every rank, then waits for all
    contributions addressed to it, delivering anything it still has queued
    while it waits. The send this rank makes to `victim` in its first large
    allgather (the tactic merge) is queued instead of delivered, and this rank
    enters that allgather `delay_s` late, so it finds everything it needs and
    returns with the send still queued: only its next call delivers it.
    """

    def __init__(self, root, rank, size, stranded_to=None, delay_s=0.0):
        self.root, self.rank, self.size = root, rank, size
        self.stranded_to, self.delay_s = stranded_to, delay_s
        self.calls = 0
        self.large_calls = 0
        self.queued = []
        self.collectives = []

    def _write(self, path, obj):
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "wb") as f:
            pickle.dump(obj, f)
        os.rename(tmp, path)

    def _progress(self):
        for path, obj in self.queued:
            self._write(path, obj)
        self.queued.clear()

    def tp_cp_allgather(self, obj, *, small_payload: bool = False):
        self.collectives.append(("allgather", small_payload))
        self._progress()
        call = self.calls
        self.calls += 1
        strand = False
        if not small_payload:
            self.large_calls += 1
            strand = self.stranded_to is not None and self.large_calls == 1
            if strand:
                time.sleep(self.delay_s)
        for dst in range(self.size):
            path = os.path.join(self.root, f"c{call}_to{dst}_from{self.rank}")
            if strand and dst == self.stranded_to:
                self.queued.append((path, obj))
            else:
                self._write(path, obj)
        out = [None] * self.size
        missing = set(range(self.size))
        deadline = time.monotonic() + TIMEOUT_S
        while True:
            for src in list(missing):
                path = os.path.join(self.root, f"c{call}_to{self.rank}_from{src}")
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        out[src] = pickle.load(f)
                    missing.discard(src)
            if not missing:
                return out
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"rank {self.rank}: allgather #{call} still waits for ranks {sorted(missing)}"
                )
            self._progress()
            time.sleep(0.002)


class _FakeGemmRunner(TunableRunner):
    """A GEMM-like runner with 8 tactics; tactic t costs |t - M % 8| + 1.

    `kernel_cache` makes the tuner treat it as a JIT runner, so the winners
    are primed after each exchange as for the CuTe DSL GEMMs.
    """

    kernel_cache = {}

    def __init__(self, strategy, log):
        self.strategy = strategy
        self.log = log
        self.tuning_config = TuningConfig(
            dynamic_tensor_specs=(DynamicTensorSpec(0, 0, BUCKETS, _last_pow2),),
            distributed_tuning_strategy=strategy,
            use_cuda_graph=False,
        )

    def unique_id(self):
        return ("fake_gemm", self.strategy.value)

    def get_valid_tactics(self, inputs, profile, **kwargs):
        return list(range(N_TACTICS))

    def forward(self, inputs, *, tactic=-1, do_preparation=False, **kwargs):
        self.log.append(("forward", int(inputs[0].shape[0]), tactic))
        return inputs[0] @ inputs[1].t()


def _fake_profile_single_kernel(
    self, runner, inputs, tactic, tuning_config, use_cuda_graph=False, **kwargs
):
    runner(inputs, tactic=tactic, **kwargs)
    return float(abs(tactic - inputs[0].shape[0] % N_TACTICS) + 1)


def _patch_device_queries():
    torch.cuda.get_device_name = lambda *a, **k: "cpu-test"
    torch.cuda.get_device_capability = lambda *a, **k: (0, 0)


def _make_tuner(mapping, dist_obj):
    AutoTuner._instance = None
    tuner = AutoTuner.get()
    tuner.setup_distributed_state(mapping, dist_obj)
    tuner._profile_single_kernel = _fake_profile_single_kernel.__get__(tuner)
    tuner.is_tuning_mode = True
    return tuner


def _tensors(m: int):
    return [torch.randn(m, 16), torch.randn(24, 16)]


def _run_calls(tuner, runner, ms):
    """The tuned calls of a forward: returns the tactic each call runs."""
    chosen = []
    for m in ms:
        _, tactic = tuner.choose_one("test::fake_gemm", [runner], runner.tuning_config, _tensors(m))
        chosen.append(tactic)
    return chosen


def _disable_guards(barrier: bool) -> None:
    """The exchange barrier is always on in the autotuner; a test turns it off
    by replacing the method (in the forked rank, or under monkeypatch)."""
    if not barrier:
        AutoTuner._exchange_barrier = lambda self, what: 0.0


def _rank_main(rank, scenario, guards, tmpdir, queue):
    os.environ["TLLM_PROFILING_TIMER"] = "cuda_event"
    os.environ["TLLM_AUTOTUNER_HANG_DUMP_S"] = "0"
    _disable_guards(**guards)
    result = {"rank": rank}
    try:
        import torch.distributed as dist

        if scenario == "equivalence":
            dist.init_process_group(
                "gloo",
                init_method=f"file://{tmpdir}/store",
                rank=rank,
                world_size=WORLD,
                timeout=timedelta(seconds=TIMEOUT_S),
            )
            timeout = timedelta(seconds=TIMEOUT_S)
            tuning_group = dist.new_group(backend="gloo", timeout=timeout)
            device_group = dist.new_group(backend="gloo", timeout=timeout)
            dist_obj = _GlooDist(tuning_group)
        else:
            dist_obj = _WeakProgressDist(
                tmpdir, rank, WORLD, stranded_to=1 if rank == 3 else None, delay_s=0.5
            )
        log = []
        _patch_device_queries()
        tuner = _make_tuner(_Mapping(rank, WORLD), dist_obj)
        runner = _FakeGemmRunner(DistributedTuningStrategy.PARALLEL, log)

        ms = (16, 16, 3, 16, 32) if scenario == "equivalence" else (16,)
        result["chosen"] = _run_calls(tuner, runner, ms)
        # The next device collective of the forward.
        if scenario == "weak_progress":
            queue_barrier.wait(timeout=TIMEOUT_S)
        else:
            dist.barrier(group=device_group)
        result["cache"] = sorted((repr(k), v[1]) for k, v in tuner.profiling_cache.cache.items())
        result["forwards"] = log
        result["collectives"] = dist_obj.collectives
        result["ok"] = True
    except Exception as e:  # with a guard off, the expected outcome is a timeout
        result["ok"] = False
        result["error"] = f"{type(e).__name__}: {e}"
        result["trace"] = traceback.format_exc(limit=4)
    queue.put(result)
    queue.close()
    queue.join_thread()
    sys.stdout.flush()
    os._exit(0)


queue_barrier = None


def _run_ranks(scenario, guards):
    """Fork WORLD ranks, return their results ordered by rank."""
    global queue_barrier
    import multiprocessing

    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    queue_barrier = ctx.Barrier(WORLD)
    with tempfile.TemporaryDirectory() as tmpdir:
        procs = [
            ctx.Process(target=_rank_main, args=(r, scenario, guards, tmpdir, queue))
            for r in range(WORLD)
        ]
        for p in procs:
            p.start()
        results = []
        deadline = time.monotonic() + 3 * TIMEOUT_S + 30
        while len(results) < WORLD and time.monotonic() < deadline:
            try:
                results.append(queue.get(timeout=1.0))
            except Exception:
                pass
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join()
    assert len(results) == WORLD, f"only {len(results)} ranks reported"
    return sorted(results, key=lambda r: r["rank"])


def _env(barrier):
    return {"barrier": barrier}


def _assert_all_ok_and_same(results):
    for r in results:
        assert r["ok"], f"rank {r['rank']}: {r.get('error')}\n{r.get('trace')}"
    assert len({tuple(r["chosen"]) for r in results}) == 1
    assert len({tuple(r["cache"]) for r in results}) == 1


def test_weak_progress_hangs_without_exchange_barrier():
    results = _run_ranks("weak_progress", _env(barrier=False))
    assert not results[1]["ok"], results[1]
    assert "TimeoutError" in results[1]["error"], results[1]


def test_weak_progress_passes_with_exchange_barrier():
    results = _run_ranks("weak_progress", _env(barrier=True))
    _assert_all_ok_and_same(results)


def _large_collectives(result):
    return [c for c in result["collectives"] if not c[1]]


def test_lockstep_changes_no_tuning_result():
    # The barrier only adds small collectives: the same tactics are profiled
    # on each rank and the same winners are chosen.
    off = _run_ranks("equivalence", _env(barrier=False))
    on = _run_ranks("equivalence", _env(barrier=True))
    _assert_all_ok_and_same(off)
    _assert_all_ok_and_same(on)
    for a, b in zip(off, on):
        assert a["chosen"] == b["chosen"]
        assert a["cache"] == b["cache"]
        assert a["forwards"] == b["forwards"]
        assert _large_collectives(a) == _large_collectives(b)


def test_single_rank_runs_no_collective(monkeypatch):
    # A tuning group of one rank runs no barrier, and picks the same tactics as
    # with the barrier off.
    monkeypatch.setenv("TLLM_PROFILING_TIMER", "cuda_event")
    monkeypatch.setenv("TLLM_AUTOTUNER_HANG_DUMP_S", "0")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: "cpu")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (0, 0))
    saved = AutoTuner._instance
    chosen = {}
    try:
        for guards in (True, False):
            with pytest.MonkeyPatch.context() as mp:
                if not guards:
                    mp.setattr(AutoTuner, "_exchange_barrier", lambda self, what: 0.0)
                log = []
                dist_obj = _GlooDist(None)
                tuner = _make_tuner(_Mapping(0, 1), dist_obj)
                runner = _FakeGemmRunner(DistributedTuningStrategy.PARALLEL, log)
                chosen[guards] = (_run_calls(tuner, runner, (16, 16, 3, 16, 32)), list(log))
                assert dist_obj.collectives == []
    finally:
        AutoTuner._instance = saved
    assert chosen[True] == chosen[False]


@pytest.mark.parametrize("value, armed", [("0.5", 0.5), ("0", None), ("bad", 600.0)])
def test_hang_dump_timeout_comes_from_the_environment(monkeypatch, value, armed):
    """TLLM_AUTOTUNER_HANG_DUMP_S arms the stack dump around a guarded collective: 0 turns it
    off and a value that is not a number falls back to the 600 s default."""
    calls = []
    monkeypatch.setattr(faulthandler, "dump_traceback_later", lambda t, **kw: calls.append(t))
    monkeypatch.setattr(faulthandler, "cancel_dump_traceback_later", lambda: calls.append("off"))
    monkeypatch.setenv("TLLM_AUTOTUNER_HANG_DUMP_S", value)
    with AutoTuner.__new__(AutoTuner)._hang_dump_guard("merge"):
        assert calls == ([] if armed is None else [armed])
    assert calls == ([] if armed is None else [armed, "off"])

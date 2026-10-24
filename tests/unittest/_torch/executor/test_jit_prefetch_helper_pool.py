# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""JIT compile ahead's helper pool and the builds it starts, on the CPU.

The helpers here are small stand-in scripts speaking the helpers' line
protocol, so the pool's bookkeeping (a helper that exits, waits, stopping
idle helpers) is tested without a compiler.
"""

import contextlib
import os
import queue
import sys
import textwrap
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch import jit_prefetch as jp

_FAKE_HELPER = textwrap.dedent(
    """
    import json, sys
    mode = sys.argv[1]
    if mode == "die_at_start":
        sys.exit(3)
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        if mode == "die_on_request":
            sys.exit(4)
        print(json.dumps({"tag": req["tag"], "ok": True, "s": 0.0, "err": ""}), flush=True)
    """
)

_SWITCHES = (
    "TLLM_JIT_AHEAD",
    "TLLM_JIT_PREFETCH",
    "TLLM_JIT_TUNE_AHEAD",
    "TLLM_JIT_STATS",
    "TLLM_WARMUP_JIT_COVERAGE",
    "TLLM_JIT_PREFETCH_WAIT_TIMEOUT_S",
)


@pytest.fixture
def prefetcher(monkeypatch):
    """A JitPrefetcher with compile ahead on and no helpers of its own; its
    process-wide Triton hooks are restored afterwards."""
    from triton import knobs

    for var in _SWITCHES:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TLLM_JIT_AHEAD", "1")
    monkeypatch.setattr(jp.JitPrefetcher, "_start_helpers", lambda self: None)
    saved = (knobs.runtime.jit_cache_hook, knobs.compilation.listener,
             knobs.autotuning.listener)  # fmt: skip
    p = jp.JitPrefetcher(rank=0)
    yield p
    for proc in p._procs:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    (knobs.runtime.jit_cache_hook, knobs.compilation.listener,
     knobs.autotuning.listener) = saved  # fmt: skip


@contextlib.contextmanager
def _helpers(p, tmp_path, mode: str, n: int = 1, kind: str = "fake"):
    """Start ``n`` stand-in helpers of ``kind``; on exit, stop them and wait
    for the pool's threads to end, so none outlives the test."""
    script = tmp_path / "fake_helper.py"
    script.write_text(_FAKE_HELPER)
    p._queues.setdefault(kind, queue.PriorityQueue())
    p._spawn(kind, [sys.executable, "-u", str(script), mode], dict(os.environ), n)
    try:
        yield p._procs_by_kind[kind][-n:]
    finally:
        for proc in p._procs:
            if proc.poll() is None:
                proc.kill()
        for t in [*p._helper_threads, *threading.enumerate()]:
            if t in p._helper_threads or t.name == "jit_prefetch_retire":
                t.join(10.0)
        assert not any(t.is_alive() for t in p._helper_threads)


def _wait_until(cond, timeout_s: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


@pytest.mark.cpu_only
def test_a_helper_that_exits_mid_build_releases_the_waiting_launch(prefetcher, tmp_path):
    """A helper that exits while building a variant releases that build: a
    launch waiting for it stops waiting at once (and builds the variant in
    place) instead of after the wait timeout. As the kind's last helper, its
    exit also releases what is still queued, and a later request is never
    started, so no launch waits for it."""
    p = prefetcher
    with _helpers(p, tmp_path, "die_on_request"):
        assert p._submit("k1", "spec1", "label", jp._PRIO_BATCH, kind="fake")
        assert p._key_event["k1"].wait(30)
        assert _wait_until(lambda: not p._helpers_alive("fake"))
        assert p.stats.helper_fail == 1
        assert p._submit("k2", "spec2", "label", jp._PRIO_BATCH, kind="fake")
        time.sleep(0.2)
        assert "k2" not in p._key_started


@pytest.mark.cpu_only
def test_wait_ready_does_not_wait_for_helpers_that_exited(prefetcher, tmp_path):
    """Helpers that exit before they are ready (an import failure, say) end
    ``wait_ready``'s wait for them; the wait is one deadline for all
    helpers, not one per helper."""
    p = prefetcher
    with _helpers(p, tmp_path, "die_at_start", n=3) as procs:
        assert _wait_until(lambda: all(proc.poll() is not None for proc in procs))
        t0 = time.monotonic()
        p.wait_ready(timeout_s=60.0)
        assert time.monotonic() - t0 < 10.0
        assert _wait_until(lambda: len(p._exited) == 3)


@pytest.mark.cpu_only
def test_idle_helpers_are_stopped_once_serving_runs(prefetcher, tmp_path, monkeypatch):
    """Once the executor thread is bound (serving), helpers whose queue has
    drained are stopped; until then they stay up, as warmup still queues
    work for them."""
    p = prefetcher
    monkeypatch.setattr(jp.JitPrefetcher, "_take_deep_gemm_compiles", lambda self, report: None)
    with _helpers(p, tmp_path, "ok", n=2) as procs:
        assert p._submit("k", "spec", "label", jp._PRIO_BATCH, kind="fake")
        assert p._key_event["k"].wait(30)
        time.sleep(0.3)
        assert all(proc.poll() is None for proc in procs)
        p.bind_executor_thread()
        assert _wait_until(lambda: all(proc.poll() is not None for proc in procs))
        assert p._retired == {"fake"}
        assert p.stats.helper_fail == 0


@pytest.mark.cpu_only
def test_a_wait_that_times_out_gives_up_on_the_build(monkeypatch, prefetcher):
    """A launch waits for a helper's build at most the wait timeout; then
    that build is given up on, and later launches of the variant do not
    wait for it. A timeout that is not a number is ignored."""
    monkeypatch.setenv("TLLM_JIT_PREFETCH_WAIT_TIMEOUT_S", "not-a-number")
    assert jp._wait_timeout_s() == 300.0
    p = prefetcher
    p._wait_timeout = 0.1
    ev = threading.Event()
    p._wait_for_build(ev, "variant")
    assert ev.is_set()
    assert p.stats.wait_n == 1


@pytest.mark.cpu_only
def test_helper_env(monkeypatch, tmp_path):
    """Helpers see no launcher variables and no GPU, the JIT cache locations
    as absolute paths (they run in /tmp), and PYTHONPATH only when asked."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DG_JIT_CACHE_DIR", "rel/dg")
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "3")
    monkeypatch.setenv("SLURM_PROCID", "3")
    monkeypatch.setenv("PYTHONPATH", "/src")
    env = jp._helper_env(EXTRA="1")
    assert env["DG_JIT_CACHE_DIR"] == str(tmp_path / "rel" / "dg")
    assert "OMPI_COMM_WORLD_RANK" not in env and "SLURM_PROCID" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == "" and env["EXTRA"] == "1"
    assert "PYTHONPATH" not in env
    assert jp._helper_env(keep_pythonpath=True)["PYTHONPATH"] == "/src"


@pytest.mark.cpu_only
def test_plan_warmup_batch_plans_the_providers_that_opt_in(prefetcher):
    """A warmup forward's batch is planned by the providers that plan
    warmup batches, each with one dedup set across the warmup's batches;
    other providers are not asked."""
    p = prefetcher
    seen_sets = []

    class Planned:
        plans_warmup_batches = True

        def __call__(self, batch_ctx, seen=None):
            seen_sets.append(seen)
            return []

    class NotPlanned:
        def __call__(self, batch_ctx, seen=None):
            raise AssertionError("not a warmup batch provider")

    p.register("planned", Planned())
    p.register("not_planned", NotPlanned())
    ctx = SimpleNamespace(ctx_chunk_lens=[2048], any_ctx_cached=False, gen_tokens=0)
    assert p.plan_warmup_batch(ctx) == 0
    assert p.plan_warmup_batch(ctx) == 0
    assert len(seen_sets) == 2 and seen_sets[0] is seen_sets[1]


@pytest.mark.cpu_only
def test_flashinfer_norm_builds_queue_only_new_variants(prefetcher, monkeypatch):
    """The FlashInfer norm variants are queued once the model is constructed
    and the loader installed once; the call at the end of warmup queues
    only variants not queued before."""
    from tensorrt_llm._torch import jit_prefetch_cute_dsl as jcd

    p = prefetcher
    installed = []
    monkeypatch.setattr(jp.JitPrefetcher, "_start_cute_dsl_helpers", lambda self: True)
    monkeypatch.setattr(
        jcd, "install_flashinfer_norm_loader", lambda loader: installed.append(1) or ["getter"]
    )
    p._queues[jcd.KIND] = queue.PriorityQueue()
    provider = SimpleNamespace(shapes=[("bfloat16", 4096, 0.0)], specs=["a", "b"])
    provider.enumerate_specs = lambda: list(provider.specs)
    assert p.enable_flashinfer_norm(provider)
    assert not p.enable_flashinfer_norm(provider)
    provider.specs.append("c")
    assert p.enable_flashinfer_norm(provider)
    assert installed == [1]
    assert p._queues[jcd.KIND].qsize() == 3


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "mla,has_trtllm_gen,expected",
    [(False, True, ["fmha_gen"]), (True, True, []), (False, False, [])],
    ids=["gqa", "mla", "no_trtllm_gen"],
)
def test_fmha_gen_is_predicted_for_layers_on_trtllm_gen(mla, has_trtllm_gen, expected):
    """A model whose regular (non-MLA) attention layers can run on
    FlashInfer's TRTLLM-GEN FMHA library builds FlashInfer's fmha_gen module
    in its first forward, so compile ahead builds it from model
    construction; other models do not."""
    from tensorrt_llm._torch.attention.backends.fmha import jit_prefetch as fmha_plan
    from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
    from tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen import (
        FlashInferTrtllmGenFmha,
    )

    libs = [object.__new__(FallbackFmha)]
    if has_trtllm_gen:
        libs.insert(0, object.__new__(FlashInferTrtllmGenFmha))
    layer = torch.nn.Module()
    layer.attn = SimpleNamespace(is_mla_enable=mla, _fmha_manager=SimpleNamespace(fmha_libs=libs))
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), layer)
    assert fmha_plan.flashinfer_modules(model) == expected


@pytest.mark.cpu_only
@pytest.mark.parametrize("running", [True, False])
def test_swap_ab_tuning_pass_runs_without_deep_gemm_helpers(monkeypatch, running):
    """With compile ahead on, the swap-AB DeepGEMM ops skip their JIT
    tuning pass only while DeepGEMM helpers build the layouts; without them
    (they did not start, or all exited during warmup) the pass compiles the
    layouts as without compile ahead."""
    from tensorrt_llm._torch import jit_prefetch_deep_gemm as jdg
    from tensorrt_llm._torch.custom_ops import torch_custom_ops as ops

    for var in _SWITCHES:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TLLM_JIT_AHEAD", "1")
    monkeypatch.setattr(jdg, "supported", lambda: True)
    monkeypatch.setattr(jdg, "helpers_running", running)
    ops._deep_gemm_jit_coverage_pass.cache_clear()
    try:
        assert ops._deep_gemm_jit_tuning_needed() is (not running)
    finally:
        ops._deep_gemm_jit_coverage_pass.cache_clear()


@pytest.mark.cpu_only
def test_process_compiles_are_reported_at_most_once_a_second(prefetcher, monkeypatch):
    """Serving forwards report the DeepGEMM kernels compiled in this process
    since the last report, at most once a second."""
    p = prefetcher
    reports = []
    monkeypatch.setattr(
        jp.JitPrefetcher, "_take_deep_gemm_compiles", lambda self, report: reports.append(report)
    )
    for _ in range(5):
        p.report_process_compiles()
    assert reports == [True]
    p._next_compile_report = 0.0
    p.report_process_compiles()
    assert reports == [True, True]


@pytest.mark.cpu_only
def test_layer_module_helpers_are_joined_before_the_first_forward(monkeypatch):
    """The FlashInfer helpers building a module of the model's layers are
    joined on their own (before warmup's first forward); the others stay
    pending until their join before graph capture."""
    import subprocess

    def helper(name):
        line = {"module": name, "ok": False, "s": 0.0, "err": "stand-in"}
        return subprocess.Popen(
            [sys.executable, "-c", f"import json; print(json.dumps({line!r}))"],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )

    layer, other = helper("fmha_gen"), helper("sampling")
    monkeypatch.setattr(jp, "_fi_builds", [(("fmha_gen",), layer), (("sampling",), other)])
    monkeypatch.setattr(jp, "_fi_layer_modules", {"fmha_gen"})
    assert jp.flashinfer_layer_builds_pending()
    assert jp.wait_flashinfer_layer_builds() is not None
    assert jp._fi_builds == [(("sampling",), other)]
    assert not jp.flashinfer_layer_builds_pending()
    assert jp.wait_flashinfer_layer_builds() is None
    assert jp.wait_flashinfer_build() is not None
    assert jp._fi_builds == []


@pytest.mark.cpu_only
def test_recorded_launches_are_planned_on_a_thread_of_their_own(prefetcher):
    """A launch class warmup records is planned off the launching thread, on
    a thread the idle-helper retirement waits for."""
    p = prefetcher
    seen = []

    class Recorded:
        def plan_all(self):
            seen.append(threading.current_thread().name)
            return []

    p.plan_recorded(Recorded())
    threads = list(p._enum_threads)
    assert [t.name for t in threads] == ["jit_prefetch_recorded"]
    for t in threads:
        t.join(10.0)
    assert seen == ["jit_prefetch_recorded"]


@pytest.mark.cpu_only
def test_top_k_and_csa2_builds_are_queued_once(prefetcher, monkeypatch):
    """The radix top-k and CSA2 publication variants are queued before
    warmup's first forward. A later call queues only top-k variants not
    queued before (a TopK a sparse backend builds in its first forward), on
    a thread of its own when asked, and installs no loader twice; CSA2's
    later calls do nothing."""
    from tensorrt_llm._torch import jit_prefetch_cute_dsl as jcd

    p = prefetcher
    installed = []
    monkeypatch.setattr(jp.JitPrefetcher, "_start_cute_dsl_helpers", lambda self: True)
    monkeypatch.setattr(
        jcd, "install_topk_loader", lambda loader: installed.append("topk") or ["r"]
    )
    monkeypatch.setattr(jcd, "install_csa2_loader", lambda loader: installed.append("csa2") or True)
    p._queues[jcd.KIND] = queue.PriorityQueue()
    topk = SimpleNamespace(enumerate_specs=lambda next_ns: ["t1", "t2"])
    csa2 = SimpleNamespace(enumerate_specs=lambda: ["c1"])
    assert p.enable_cute_dsl_topk(topk, next_ns=[1]) and p.enable_csa2_publish(csa2)
    assert p.enable_cute_dsl_topk(topk, next_ns=[1]) and not p.enable_csa2_publish(csa2)
    assert p._queues[jcd.KIND].qsize() == 3
    later = SimpleNamespace(enumerate_specs=lambda next_ns: ["t2", "t3"])
    assert p.enable_cute_dsl_topk(later, next_ns=[1], urgent=True, background=True)
    for t in list(p._enum_threads):
        t.join(10.0)
    assert installed == ["topk", "csa2"]
    assert p._queues[jcd.KIND].qsize() == 4


@pytest.mark.cpu_only
def test_a_topk_reports_itself_at_its_first_forward(monkeypatch):
    """A TopK reaches the prefetcher's hook at its first forward, once, with
    the implementation its owner chose after building it, so a TopK a sparse
    backend builds in its first forward has its variants queued then."""
    from tensorrt_llm._torch.modules.top_k import TopK, TopKImplementation

    seen = []
    monkeypatch.setattr(
        TopK, "on_first_forward", lambda mod: seen.append(mod.prefill_implementation)
    )
    mod = TopK(2)
    mod.prefill_implementation = TopKImplementation.TORCH
    scores = torch.tensor([[0.1, 0.9, 0.5, 0.3]])
    out = torch.zeros(1, 2, dtype=torch.int32)
    for _ in range(2):
        mod(scores, out, is_prefill=True, row_starts=torch.tensor([0]), row_ends=torch.tensor([4]))
    assert seen == [TopKImplementation.TORCH]

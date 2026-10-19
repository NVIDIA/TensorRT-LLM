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
"""The Mamba SSD JIT-prefetch planner must predict the real launches' keys.

Records the Triton launches of one real Mamba2 SSD prefill (on CUDA tensors
built the way Mamba2Mixer.forward_core builds them) and the launches the
planner predicts on meta tensors, and checks that every real launch's Triton
cache key is among the planned ones. A mismatch means prefetch compiles a
variant nobody launches and the real one still compiles on the executor.
"""

import types

import pytest
import torch

from tensorrt_llm._torch import jit_prefetch as jp
from tensorrt_llm._torch.modules.mamba import jit_prefetch as mjp


def _shape(token_major: bool):
    s = mjp._MixerShape.__new__(mjp._MixerShape)
    s.nheads, s.head_dim, s.ngroups, s.d_state, s.chunk_size = 16, 64, 2, 64, 64
    s.d_inner = s.nheads * s.head_dim
    s.conv_dim = s.d_inner + 2 * s.ngroups * s.d_state
    s.delta_softplus = True
    s.state_dtype = torch.bfloat16
    s.token_major_conv = token_major
    s.has_dt_bias = True
    s.has_d = True
    s.io_dtype = torch.bfloat16
    s.zxbcdt_width = 2 * s.d_inner + 2 * s.ngroups * s.d_state + s.nheads
    return s


def _keys(calls):
    keys = set()
    for call in calls:
        variants, tuned = jp._expand(call)
        jit_fn, _, _ = jp._unwrap_jit(call.fn)
        for kw in variants:
            from triton import knobs
            from triton.runtime.driver import driver
            from triton.runtime.jit import compute_cache_key

            kw = dict(kw)
            kw["debug"] = kw.get("debug", jit_fn.debug) or knobs.runtime.debug
            kw["instrumentation_mode"] = knobs.compilation.instrumentation_mode
            dev = driver.active.get_current_device()
            _, kkc, _, _, binder = jit_fn.device_caches[dev]
            _, spec, opts = binder(*call.args, **kw)
            keys.add((jit_fn.fn.__name__, compute_cache_key(kkc, spec, opts)))
    return keys


def _real_calls(s, ctx_lens, any_cached):
    """Record the launches of a real forward on CUDA tensors, laid out like
    Mamba2Mixer.forward_core."""
    from tensorrt_llm._torch.modules.mamba.mamba2_metadata import (
        compute_extra_chunks_cpu,
        cu_seqlens_to_chunk_indices_offsets_triton,
    )
    from tensorrt_llm._torch.modules.mamba.ssd_combined import _mamba_chunk_scan_combined_fwd

    dev = torch.device("cuda")
    n, T = len(ctx_lens), sum(ctx_lens)
    zxbcdt = torch.empty(T, s.zxbcdt_width, dtype=s.io_dtype, device=dev)
    dt = zxbcdt[:, s.d_inner + s.conv_dim :].unsqueeze(0)
    bc = s.ngroups * s.d_state
    if s.token_major_conv:
        xbc = torch.empty(T, s.conv_dim, dtype=s.io_dtype, device=dev).t().t()
        x = xbc[:, : s.d_inner].view(T, s.nheads, s.head_dim).unsqueeze(0)
        B = xbc[:, s.d_inner : s.d_inner + bc].view(T, s.ngroups, s.d_state).unsqueeze(0)
        C = xbc[:, s.d_inner + bc :].view(T, s.ngroups, s.d_state).unsqueeze(0)
    else:
        x = torch.empty(T, s.d_inner, dtype=s.io_dtype, device=dev).view(1, T, s.nheads, s.head_dim)
        B = torch.empty(T, bc, dtype=s.io_dtype, device=dev).view(1, T, s.ngroups, s.d_state)
        C = torch.empty(T, bc, dtype=s.io_dtype, device=dev).view(1, T, s.ngroups, s.d_state)
    f32 = dict(dtype=torch.float32, device=dev)
    A, D, dt_bias = (torch.empty(s.nheads, **f32) for _ in range(3))
    cu = torch.tensor([0] + list(torch.tensor(ctx_lens).cumsum(0)), dtype=torch.int, device=dev)
    seq_idx = torch.repeat_interleave(
        torch.arange(n, device=dev, dtype=torch.int), torch.tensor(ctx_lens, device=dev)
    ).unsqueeze(0)
    out = torch.empty(T, s.nheads * s.head_dim, dtype=s.io_dtype, device=dev).view(
        1, T, -1, s.head_dim
    )
    init = ci = co = None
    with jp.shadow_launches() as rec:
        if any_cached:
            extra = compute_extra_chunks_cpu(ctx_lens, n, s.chunk_size)
            ci, co = cu_seqlens_to_chunk_indices_offsets_triton(
                cu, s.chunk_size, total_seqlens=T, extra_chunks=extra
            )
            init = torch.empty(n, s.nheads, s.head_dim, s.d_state, dtype=s.state_dtype, device=dev)
        _mamba_chunk_scan_combined_fwd(
            x,
            dt,
            A,
            B,
            C,
            s.chunk_size,
            D=D,
            z=None,
            dt_bias=dt_bias,
            initial_states=init,
            seq_idx=seq_idx,
            chunk_indices=ci,
            chunk_offsets=co,
            cu_seqlens=cu,
            dt_softplus=s.delta_softplus,
            dt_limit=(0.0, float("inf")),
            out=out,
            state_dtype=s.state_dtype,
        )
    return list(rec)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("token_major", [True, False])
@pytest.mark.parametrize(
    "ctx_lens,any_cached",
    [
        ([512], False),
        ([300, 212], False),
        ([300, 213], True),
        ([1, 511], True),
    ],
)
def test_planned_keys_cover_real_launches(token_major, ctx_lens, any_cached):
    s = _shape(token_major)
    prov = mjp.MambaSSDProvider.__new__(mjp.MambaSSDProvider)
    prov.shapes, prov._seen_keys = [s], set()
    planned = _keys(prov(types.SimpleNamespace(ctx_chunk_lens=ctx_lens, any_ctx_cached=any_cached)))
    real = _keys(_real_calls(s, ctx_lens, any_cached))
    assert real, "recorded no real launches"
    missing = real - planned
    assert not missing, f"real launches the planner did not predict: {missing}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("token_major", [True, False])
@pytest.mark.parametrize(
    "ctx_lens,any_cached",
    [([512], False), ([300, 212], False), ([300, 213], True), ([1, 511], True), ([17] * 3, True)],
)
def test_enumeration_covers_real_launches(token_major, ctx_lens, any_cached):
    """The background enumeration alone must plan every real launch of any
    batch within the engine limits."""
    s = _shape(token_major)
    prov = mjp.MambaSSDProvider.__new__(mjp.MambaSSDProvider)
    prov.shapes, prov._seen_keys = [s], set()
    prov.max_num_tokens, prov.max_batch_size = 512, 64
    seen = set()
    planned = set()
    for batch in prov.enumerate_batches():
        planned |= _keys(prov(batch, seen=seen))
    real = _keys(_real_calls(s, ctx_lens, any_cached))
    missing = real - planned
    assert not missing, f"real launches the enumeration did not predict: {missing}"


def _tune_keys(calls):
    """(kernel, autotune key) of every autotuned launch with more than one
    candidate config, i.e. every launch that can run the benchmark."""
    keys = set()
    for call in calls:
        jit_fn, autotuner, _ = jp._unwrap_jit(call.fn)
        if autotuner is not None and len(autotuner.configs) > 1:
            keys.add(
                (jit_fn.fn.__name__, jp._autotune_key(autotuner, None, call.args, call.kwargs))
            )
    return keys


def _tune_provider(token_major, max_num_tokens=512, max_batch_size=64, max_seq_len=0):
    s = _shape(token_major)
    prov = mjp.MambaSSDProvider.__new__(mjp.MambaSSDProvider)
    prov.shapes, prov._seen_keys = [s], set()
    prov.max_num_tokens, prov.max_batch_size = max_num_tokens, max_batch_size
    prov.max_seq_len = max_seq_len
    return s, prov


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("token_major", [True, False])
@pytest.mark.parametrize(
    "ctx_lens,any_cached",
    [([512], False), ([300, 212], False), ([300, 213], True), ([1, 511], True), ([17] * 3, True)],
)
def test_tune_batches_reach_every_autotune_key(token_major, ctx_lens, any_cached):
    """The batches tune ahead runs reach the autotune key of every autotuned
    launch any batch within the engine limits makes: a key does not depend
    on the batch's size or sequence count."""
    s, prov = _tune_provider(token_major)
    batches = list(prov.tune_batches())
    assert [b.any_ctx_cached for b in batches] == [False, True]
    for b in batches:
        assert sum(b.ctx_chunk_lens) == 512 and len(b.ctx_chunk_lens) == 64
    reached = set()
    for b in batches:
        reached |= _tune_keys(prov(b, seen=set()))
    missing = _tune_keys(_real_calls(s, ctx_lens, any_cached)) - reached
    assert not missing, f"autotune keys tune ahead does not reach: {missing}"


def test_tune_batches_respect_engine_limits(monkeypatch):
    """The tune batch stays within max_num_tokens, the token cap, the batch
    size and the per-sequence length limit; none without a Triton SSD layer."""
    monkeypatch.setattr(mjp._MixerShape, "triton_ssd", False)
    _, prov = _tune_provider(True)
    assert list(prov.tune_batches()) == []
    monkeypatch.setattr(mjp._MixerShape, "triton_ssd", True)
    _, prov = _tune_provider(True, max_num_tokens=100_000, max_batch_size=8, max_seq_len=300)
    lens = next(prov.tune_batches()).ctx_chunk_lens
    assert sum(lens) == 8 * 299 and len(lens) == 8 and max(lens) <= 299
    _, prov = _tune_provider(True, max_num_tokens=100_000, max_batch_size=512)
    lens = next(prov.tune_batches()).ctx_chunk_lens
    assert sum(lens) == mjp.MambaSSDProvider.TUNE_MAX_TOKENS
    _, prov = _tune_provider(True, max_num_tokens=3, max_batch_size=64)
    assert [b.ctx_chunk_lens for b in prov.tune_batches()] == [[1, 1, 1]] * 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.usefixtures("restore_triton_hooks")
@pytest.mark.parametrize("prefetch", [True, False], ids=["with_prefetch", "tune_ahead_alone"])
def test_tune_ahead_benchmarks_untuned_keys(prefetch, monkeypatch, tmp_path):
    """JitPrefetcher.tune runs the autotune benchmark of every untuned key of
    the provider's tune batches on the GPU, so later batches of other sizes,
    with and without cached prefixes, find each key tuned; a second call
    runs no batch. Without prefetch, its own thread pool compiles the
    candidates first, so no launch compiles one."""
    import queue

    from triton import knobs

    monkeypatch.setattr(knobs.cache, "dir", str(tmp_path))  # empty cache: candidates to compile
    monkeypatch.setenv("TLLM_JIT_TUNE_AHEAD", "1")
    if prefetch:
        monkeypatch.setenv("TLLM_JIT_PREFETCH", "1")
        # No helpers here: the queued variants compile in process at launch.
        monkeypatch.setenv("TLLM_JIT_PREFETCH_WAIT_TIMEOUT_S", "0")
        monkeypatch.setattr(jp.JitPrefetcher, "_start_helpers", lambda self: None)
    else:
        monkeypatch.delenv("TLLM_JIT_PREFETCH", raising=False)
        monkeypatch.delenv("TLLM_JIT_STATS", raising=False)
    s, prov = _tune_provider(True, max_num_tokens=512, max_batch_size=8)
    batches = list(prov.tune_batches())
    tune_keys = set()
    for batch in batches:
        for call in prov(batch, seen=set()):
            jit_fn, autotuner, _ = jp._unwrap_jit(call.fn)
            if autotuner is not None:
                autotuner.cache.clear()
            # Forget kernels earlier tests compiled in this process: with the
            # empty on-disk cache above, every candidate is then uncompiled.
            for kernel_cache, *_ in jit_fn.device_caches.values():
                kernel_cache.clear()
        tune_keys |= _tune_keys(prov(batch, seen=set()))
    p = jp.JitPrefetcher(rank=0)
    assert p.prefetch == prefetch
    if prefetch:
        p._queues["triton"] = queue.PriorityQueue()
    p.register("mamba_ssd", prov)
    p.tune()
    assert p.stats.ta_batches == len(batches) == 2
    assert p.stats.ta_tunes == len(tune_keys) > 0
    if not prefetch:
        assert p.stats.ta_inproc_compiles > 0 and p.stats.ta_inproc_failed == 0
        # The pool delivered every candidate: the benchmark compiled nothing.
        assert p.stats.ta_compiles == 0
    for lens, cached in (([37, 120, 211], True), ([37, 120, 211], False), ([300], False)):
        later = _real_calls(s, lens, cached)
        assert not [c.label for c in later if jp._untuned(c)], (lens, cached)
    p.tune()
    assert p.stats.ta_batches == 2
    assert p.stats.ta_tunes == len(tune_keys)


@pytest.mark.parametrize(
    "env,prefetch,tune,engine",
    [
        ({}, False, False, False),
        ({"TLLM_JIT_PREFETCH": "1"}, True, True, True),
        ({"TLLM_JIT_PREFETCH": "1", "TLLM_JIT_TUNE_AHEAD": "0"}, True, False, True),
        ({"TLLM_JIT_TUNE_AHEAD": "1"}, False, True, True),
        ({"TLLM_JIT_STATS": "1"}, False, False, True),
    ],
)
def test_prefetch_and_tune_ahead_are_independent(env, prefetch, tune, engine, monkeypatch):
    """TLLM_JIT_PREFETCH and TLLM_JIT_TUNE_AHEAD switch independently; tune
    ahead follows prefetch only while TLLM_JIT_TUNE_AHEAD is unset."""
    for var in ("TLLM_JIT_PREFETCH", "TLLM_JIT_TUNE_AHEAD", "TLLM_JIT_STATS"):
        monkeypatch.delenv(var, raising=False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    assert jp.prefetch_enabled() == prefetch
    assert jp.tune_ahead_enabled() == tune
    assert jp.jit_engine_enabled() == engine


@pytest.mark.usefixtures("restore_triton_hooks")
def test_tune_ahead_off(monkeypatch):
    """TLLM_JIT_TUNE_AHEAD=0 runs no tune batch, with prefetch on."""
    monkeypatch.setenv("TLLM_JIT_PREFETCH", "1")
    monkeypatch.setenv("TLLM_JIT_TUNE_AHEAD", "0")
    monkeypatch.setattr(jp.JitPrefetcher, "_start_helpers", lambda self: None)
    ran = []

    class Provider:
        def __call__(self, batch_ctx, seen=None):
            raise AssertionError("tune ahead is off")

        def tune_batches(self):
            ran.append(1)
            return iter(())

    p = jp.JitPrefetcher(rank=0)
    p.register("p", Provider())
    p.tune()
    assert not ran and p.stats.ta_batches == 0


@pytest.fixture
def restore_triton_hooks():
    """A JitPrefetcher installs process-wide Triton hooks; restore them so a
    prefetcher built by one test does not observe another test's compiles."""
    from triton import knobs

    saved = (knobs.runtime.jit_cache_hook, knobs.compilation.listener,
             knobs.autotuning.listener)  # fmt: skip
    yield
    (knobs.runtime.jit_cache_hook, knobs.compilation.listener,
     knobs.autotuning.listener) = saved  # fmt: skip


@pytest.mark.usefixtures("restore_triton_hooks")
def test_executor_compiles_triton_variant_no_helper_started(monkeypatch):
    """A variant still queued (no helper started it) is compiled on the
    executor at once; only a variant a helper is compiling is waited for."""
    import queue
    import threading

    from triton import knobs

    from tensorrt_llm._torch import jit_prefetch as jp

    monkeypatch.setenv("TLLM_JIT_STATS", "1")
    monkeypatch.delenv("TLLM_JIT_PREFETCH", raising=False)
    p = jp.JitPrefetcher(rank=0)
    p._queues["triton"] = queue.PriorityQueue()
    assert p._submit("k-queued", '{"v": 1}', "f", jp._PRIO_ENUM)
    assert p._submit("k-started", '{"v": 2}', "f", jp._PRIO_ENUM)
    p._key_started.add("k-started")
    fn = type("F", (), {"name": "f"})()
    hook = knobs.runtime.jit_cache_hook
    kw = dict(repr="", fn=fn, compile={}, is_manual_warmup=False, already_compiled=False)
    hook(key="k-queued", **kw)
    assert p.stats.wait_n == 0
    threading.Timer(0.05, p._key_event["k-started"].set).start()
    hook(key="k-started", **kw)
    assert p.stats.wait_n == 1


@pytest.mark.usefixtures("restore_triton_hooks")
def test_background_enumeration_reruns_only_warmup_recorded_providers(monkeypatch):
    """After a second warmup, only providers that plan from warmup records
    are enumerated again; the others are enumerated once per process."""
    import threading

    monkeypatch.setenv("TLLM_JIT_PREFETCH", "1")
    monkeypatch.setattr(jp.JitPrefetcher, "_start_helpers", lambda self: None)
    p = jp.JitPrefetcher(rank=0)
    runs = []
    done = threading.Semaphore(0)

    class Provider:
        def __init__(self, name, plans_from_warmup):
            self.name, self.plans_from_warmup = name, plans_from_warmup

        def __call__(self, batch_ctx, seen=None):
            return []

        def enumerate_batches(self):
            runs.append(self.name)
            done.release()
            return iter(())

    p.register("static", Provider("static", False))
    p.register("recorded", Provider("recorded", True))
    p.start_background()  # end of the KV-cache estimation warmup
    for _ in range(2):
        assert done.acquire(timeout=10)
    p.start_background()  # end of the final executor's warmup
    assert done.acquire(timeout=10)
    assert not done.acquire(timeout=0.2)
    assert sorted(runs) == ["recorded", "recorded", "static"]


@pytest.mark.usefixtures("restore_triton_hooks")
def test_cute_dsl_load_waits_only_for_a_compile_under_way(tmp_path, monkeypatch):
    """A CuTe DSL variant still queued (no helper started it) is compiled in
    place at once, as without prefetch; only one a helper is compiling is
    waited for."""
    import queue
    import threading

    from tensorrt_llm._torch import jit_prefetch_cute_dsl as jcd

    monkeypatch.setenv("TLLM_JIT_PREFETCH", "1")
    monkeypatch.setattr(jp.JitPrefetcher, "_start_helpers", lambda self: None)
    monkeypatch.setattr(jcd, "load", lambda spec, out_dir: None)
    p = jp.JitPrefetcher(rank=0)
    p._cd_dir = str(tmp_path)
    p._queues[jcd.KIND] = queue.PriorityQueue()
    for spec in ("queued", "started"):
        assert p._submit(spec, spec, "cute_dsl", jp._PRIO_ENUM, kind=jcd.KIND)
    p._key_started.add("started")
    assert p._cute_dsl_load("queued") is None
    assert p.stats.wait_n == 0
    threading.Timer(0.05, p._key_event["started"].set).start()
    p._cute_dsl_load("started")
    assert p.stats.wait_n == 1


def test_helper_preload_then_compile(tmp_path):
    """A helper given a preload request imports the module then still
    compiles that module's kernels (preload sends no reply)."""
    import json
    import os
    import subprocess
    import sys

    from tensorrt_llm._torch.modules.mamba import jit_prefetch_linear_attn as la

    helper = os.path.join(os.path.dirname(os.path.abspath(jp.__file__)), "jit_prefetch_helper.py")
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(jp.__file__))))
    s = la._KdaShape.__new__(la._KdaShape)
    s.heads, s.head_dim, s.dtype = 4, 128, torch.bfloat16
    p = la.LinearAttnProvider.__new__(la.LinearAttnProvider)
    p.shapes, p.max_num_tokens, p.device_index = [s], 256, torch.cuda.current_device()
    p._seen, p._num_sms, p.chunk_size = set(), None, 64
    calls = p(type("B", (), {"ctx_chunk_lens": [100], "gen_tokens": 0})(), seen=set())
    specs = []
    for call in calls:
        variants, _ = jp._expand(call)
        jit_fn, _, _ = jp._unwrap_jit(call.fn)
        found = jp._lookup_or_serialize(jit_fn, call.args, variants[0])
        if found:
            specs.append(found[1])
    assert specs
    lines = [json.dumps({"preload": list(la.LinearAttnProvider.kernel_modules)})]
    lines += [json.dumps({"tag": i, "spec": spec}) for i, spec in enumerate(specs)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1",
               TRITON_CACHE_DIR=str(tmp_path))  # fmt: skip
    out = subprocess.run([sys.executable, "-u", helper, pkg_root], input="\n".join(lines) + "\n",
                         capture_output=True, text=True, env=env, timeout=600).stdout  # fmt: skip
    replies = [json.loads(line) for line in out.splitlines()]
    assert replies[0] == {"ready": True}
    assert [r["tag"] for r in replies[1:]] == list(range(len(specs)))
    assert all(r["ok"] for r in replies[1:]), replies


def test_cute_dsl_build_skips_a_variant_another_helper_built(tmp_path, monkeypatch):
    """Ranks on a node share the CuTe DSL output dir: a variant already
    exported (or exported while waiting for the lock) is not compiled."""
    from tensorrt_llm._torch import jit_prefetch_cute_dsl as jcd

    calls = []
    monkeypatch.setattr(jcd, "_compile_export", lambda *a: calls.append(a))
    spec = jcd.k123_spec(96, True, True, True, "int64", "int64", True, "float32")
    (tmp_path / f"{jcd.spec_key(spec)}.o").write_bytes(b"")
    assert jcd.build(spec, "sm_103a", str(tmp_path)) is False
    other = jcd.k123_spec(96, True, True, False, "int64", "int64", True, "float32")
    assert jcd.build(other, "sm_103a", str(tmp_path)) is True
    assert len(calls) == 1


@pytest.mark.usefixtures("restore_triton_hooks")
def test_deep_gemm_compile_counted_once_per_shape(monkeypatch):
    """Only the first slow launch of an (m, n, k) counts as an executor
    compile; repeats of the same shape are host stalls."""
    from tensorrt_llm._torch import jit_prefetch as jp
    from tensorrt_llm._torch import jit_prefetch_deep_gemm as jdg

    monkeypatch.setenv("TLLM_JIT_STATS", "1")
    monkeypatch.delenv("TLLM_JIT_PREFETCH", raising=False)
    monkeypatch.setattr(jdg, "done_observer", None)
    monkeypatch.setattr(jdg, "launch_observer", None)
    # DeepGEMM without the compile record: host time is the only signal.
    monkeypatch.setattr(jp.JitPrefetcher, "_deep_gemm_module", staticmethod(lambda: None))
    p = jp.JitPrefetcher(rank=0)
    p.bind_executor_thread()
    p.observe_deep_gemm()
    slow = jdg.COMPILE_HOST_S + 0.1
    for _ in range(3):
        jdg.done_observer(128, 7168, 2048, slow)
    jdg.done_observer(64, 7168, 2048, slow)
    jdg.done_observer(32, 7168, 2048, 0.0)  # fast first launch: a cache hit
    jdg.done_observer(32, 7168, 2048, slow)  # a stall, not a compile
    assert p.stats.dg_executor_compiles == 2
    # Each one is reported, whatever the logger's level.
    assert sum("DeepGEMM compile on executor" in e for e in p.stats.events) == 2


@pytest.mark.usefixtures("restore_triton_hooks")
def test_deep_gemm_slow_launch_without_compile_not_counted(monkeypatch):
    """With DeepJIT's record of disk-cache misses, a slow first launch that
    compiled nothing (it waited on the GPU or loaded a helper-built cubin) is
    not an executor compile; one that did compile is, with its kernel name."""
    from tensorrt_llm._torch import jit_prefetch as jp
    from tensorrt_llm._torch import jit_prefetch_deep_gemm as jdg

    compiled = []

    def take_compiled_kernel_names():
        names = list(compiled)
        compiled.clear()
        return names

    module = types.SimpleNamespace(take_compiled_kernel_names=take_compiled_kernel_names)
    monkeypatch.setenv("TLLM_JIT_STATS", "1")
    monkeypatch.delenv("TLLM_JIT_PREFETCH", raising=False)
    monkeypatch.setattr(jdg, "done_observer", None)
    monkeypatch.setattr(jdg, "launch_observer", None)
    monkeypatch.setattr(jp.JitPrefetcher, "_deep_gemm_module", staticmethod(lambda: module))
    p = jp.JitPrefetcher(rank=0)
    p.bind_executor_thread()
    p.observe_deep_gemm()
    slow = jdg.COMPILE_HOST_S + 0.1
    jdg.done_observer(316, 4608, 5120, slow)  # slow, nothing compiled
    compiled.append("sm100_fp8_fp4_gemm_1d1d")
    jdg.done_observer(413, 4608, 5120, slow)
    assert p.stats.dg_executor_compiles == 1
    assert p.stats.dg_process_compiles == 1
    events = [e for e in p.stats.events if "DeepGEMM compile on executor" in e]
    assert len(events) == 1 and "m=413" in events[0] and "sm100_fp8_fp4_gemm_1d1d" in events[0]

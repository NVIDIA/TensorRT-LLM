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


@pytest.fixture
def restore_triton_hooks():
    """A JitPrefetcher installs process-wide Triton hooks; restore them so a
    prefetcher built by one test does not observe another test's compiles."""
    from triton import knobs

    saved = (knobs.runtime.jit_cache_hook, knobs.runtime.jit_post_compile_hook,
             knobs.compilation.listener, knobs.autotuning.listener)  # fmt: skip
    yield
    (knobs.runtime.jit_cache_hook, knobs.runtime.jit_post_compile_hook,
     knobs.compilation.listener, knobs.autotuning.listener) = saved  # fmt: skip


@pytest.mark.usefixtures("restore_triton_hooks")
def test_record_roundtrip(tmp_path, monkeypatch):
    """A record written by one prefetcher is read back for replay, and a
    record from another Triton version or GPU target is discarded."""
    import json

    from tensorrt_llm._torch import jit_prefetch as jp

    monkeypatch.setenv("TLLM_JIT_RECORD_DIR", str(tmp_path))
    monkeypatch.setenv("TLLM_JIT_STATS", "1")
    monkeypatch.delenv("TLLM_JIT_PREFETCH", raising=False)
    rec = tmp_path / "jit_record.rank0.jsonl"

    p = jp.JitPrefetcher(rank=0)
    header = json.loads(rec.read_text().splitlines()[0])
    spec = json.dumps({"key": "k1", "name": "m.f"})
    p._record_fh.write(json.dumps({"name": "f", "spec": spec}) + "\n")
    p._record_fh.close()

    p2 = jp.JitPrefetcher(rank=0)
    assert [r[0] for r in p2._replay] == ["k1"]
    p2._record_fh.close()

    rec.write_text(
        json.dumps({**header, "triton": "0.0.0"}) + "\n" + rec.read_text().split("\n", 1)[1]
    )
    p3 = jp.JitPrefetcher(rank=0)
    assert p3._replay == []
    p3._record_fh.close()


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
    monkeypatch.delenv("TLLM_JIT_RECORD_DIR", raising=False)
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
    monkeypatch.delenv("TLLM_JIT_RECORD_DIR", raising=False)
    monkeypatch.setattr(jdg, "done_observer", None)
    monkeypatch.setattr(jdg, "launch_observer", None)
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

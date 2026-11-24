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
"""Kimi-K3 ULL MLA attention side (``TRTLLM_K3_ULL2_MLA_*``) against today's path, one GPU.

``TRTLLM_K3_ULL2_MLA_DEC_PDL``: the fused query (``k3_mla_q_fused``) releases its dependents at
entry and the FP4 packed decode (K3's kernel module ``k3_rubin/kernels/fp4_mla_mufu16``, installed
as the FP4 MLA backend's CuTe DSL kernel module), launched as its programmatic dependent, issues
everything but the query load (barriers, TMEM, plan, page table, the first KV tile) before
``griddepcontrol.wait``. The chain is the module's: KV-only update on a side stream joined before
the decode, fused query on the main stream (its launch marked for the decode). Checks: decode
output and Helix statistics bitwise those of the normal launch, eager and over CUDA graph replays
with new inputs, with and without split-KV; the decode falls back to a normal launch when a
metadata kernel (plan, lengths) runs in its own call or the query rows come from the cache update.

``TRTLLM_K3_ULL2_MLA_TAIL`` launch policy: at most 16 tokens, the decode whose reduction the fused
tail folds (``DeferredPackedReduce(release_early=False)``) does not release its dependents at
entry; the recorded reduction and the decode outputs are unchanged.

The FP4 cache helpers are those of ``test_k3_mla_q_fused.py`` (same directory, importable by name
under pytest's default ``prepend`` import mode).
"""

import math

import pytest
import test_k3_mla_q_fused as qf
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="needs an SM107 GPU",
)

H, LORA, ROPE, Q_LEN = qf.H, qf.LORA, qf.ROPE, qf.Q_LEN
PDL_ENV = "TRTLLM_K3_ULL2_MLA_DEC_PDL"
TRIGGER_ENV = "TRTLLM_K3_ULL3_MLA_TAIL_TRIGGER"


@pytest.fixture(autouse=True)
def _k3_decode_kernel(monkeypatch):
    qf.use_k3_decode_kernel(monkeypatch)


class _Recorder:
    """Records the ``pdl_wait`` and ``launch_dependents`` of every packed decode launch."""

    def __init__(self, module):
        self.module = module
        self.inner = module.run_trtllm_fp4_mla_decode_packed
        self.calls = []
        self.releases = []

    def __call__(self, *args, **kwargs):
        self.calls.append(bool(kwargs.get("pdl_wait", False)))
        self.releases.append(bool(kwargs.get("launch_dependents", False)))
        return self.inner(*args, **kwargs)


def _record_decode(monkeypatch):
    """Wrap the installed CuTe DSL kernel module's packed decode in a :class:`_Recorder`."""
    from tensorrt_llm._torch.attention.backends import fp4_mla

    module = fp4_mla._fp4_mla_cutedsl_kernel_module()
    assert getattr(module, "LAUNCH_DEPENDENTS_ENTRY", 0), module.__name__
    recorder = _Recorder(module)
    monkeypatch.setattr(module, "run_trtllm_fp4_mla_decode_packed", recorder)
    return recorder


class _Chain:
    """The fused-query generation chain of one layer on a real FP4 cache: KV-only update on a
    side stream, ``k3_mla_q_fused`` on the main stream, join, FP4 packed decode (+ split
    reduction) with Helix statistics."""

    def __init__(self, n, kv, seed):
        self.mgr, self.md = qf._fp4_case(n, kv, seed)
        self.w = qf._weights()
        self.T = n * Q_LEN
        self.rot = qf._rotary_table(self.md.max_seq_len, "identity")
        self.x, self.latent = qf._inputs(self.T, seed + 1)
        self.side = torch.cuda.Stream()
        self.fork, self.ready = torch.cuda.Event(), torch.cuda.Event()
        self.out = torch.empty(self.T, H, LORA, device="cuda", dtype=torch.bfloat16)
        self.stats = torch.empty(self.T, H, 2, device="cuda", dtype=torch.float32)
        self.q_stub = torch.empty(self.T, H, LORA + ROPE, device="cuda", dtype=torch.bfloat16)
        # The first update writes the Q buffers the fused query fills.
        qf._external_scatter(self.md, self.latent, self.rot)
        torch.cuda.synchronize()

    def decode(self):
        from tensorrt_llm._torch.attention.backends import fp4_mla

        state = self.md.fp4_mla_state
        fp4_mla.run_fp4_mla_attention_decode(
            self.md,
            layer_idx=0,
            local_layer=0,
            q=self.q_stub,
            output=self.out,
            sm_scale=1.0 / math.sqrt(192.0),
            kv_lora_rank=LORA,
            qk_rope_head_dim=ROPE,
            prequantized_q=state.prequantized_q,
            prequantized_q_sf=state.prequantized_q_sf,
            q_batch_capacity=state.q_batch_capacity,
            softmax_stats_tensor=self.stats,
        )

    def run(self, early_trigger, before_decode=None):
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.kernels import (
            k3_mla_q_fused as fused,
        )
        from tensorrt_llm._torch.attention.backends.fp4_mla import mark_fp4_mla_query_launch

        md = self.md
        state = md.fp4_mla_state
        self.fork.record()
        with torch.cuda.stream(self.side):
            self.fork.wait()
            qf._external_scatter(md, self.latent, self.rot)
            self.ready.record()
        fused.k3_mla_q_fused(
            self.x,
            self.w.w_qb,
            self.w.qb_exp,
            self.w.w_uk,
            state.q_global_scale,
            self.rot,
            md.helix_position_offsets,
            state.prequantized_q,
            state.prequantized_q_sf.view(torch.uint8),
            gen_len=Q_LEN,
            rows_per_seq=qf._rows_per_seq(md),
            norm=(self.w.q_norm, qf.EPS),
            early_trigger=early_trigger,
        )
        mark_fp4_mla_query_launch(md)
        self.ready.wait()
        if before_decode is not None:
            before_decode()
        self.decode()

    def result(self):
        torch.cuda.synchronize()
        return self.out.clone(), self.stats.clone()


def _assert_bitwise(got, ref):
    assert torch.equal(got[0].view(torch.int16), ref[0].view(torch.int16))
    assert torch.equal(got[1].view(torch.int32), ref[1].view(torch.int32))


@qf.needs_k3_decode
@pytest.mark.parametrize("split", ["auto", "1"])
@pytest.mark.parametrize("n", [1, 2])
def test_decode_pdl_matches_normal_launch(n, split, monkeypatch):
    monkeypatch.setenv("TRTLLM_FP4_MLA_SPLIT_KV", split)
    recorder = _record_decode(monkeypatch)
    chain = _Chain(n, 3000, seed=50 + n)
    try:
        monkeypatch.delenv(PDL_ENV, raising=False)
        chain.run(early_trigger=False)
        ref = chain.result()
        assert recorder.calls[-1] is False
        monkeypatch.setenv(PDL_ENV, "1")
        for _ in range(2):
            chain.out.fill_(torch.nan)
            chain.run(early_trigger=True)
            _assert_bitwise(chain.result(), ref)
            assert recorder.calls[-1] is True
        # A normal decode after the early-released query is also exact.
        monkeypatch.delenv(PDL_ENV)
        chain.run(early_trigger=True)
        _assert_bitwise(chain.result(), ref)
        assert recorder.calls[-1] is False
    finally:
        chain.mgr.shutdown()


@qf.needs_k3_decode
@pytest.mark.parametrize("n", [1, 2])
def test_decode_pdl_graph_replay(n, monkeypatch):
    """Captured with the programmatic edge, replayed on new queries and new KV rows (the decode's
    early KV tile loads include this step's tokens: about two tiles per split task): each replay
    equals the eager normal-launch chain on the same inputs."""
    monkeypatch.setenv(PDL_ENV, "1")
    chain = _Chain(n, 5000, seed=60 + n)
    try:
        chain.run(early_trigger=True)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            chain.run(early_trigger=True)
        torch.cuda.synchronize()
        gen = torch.Generator(device="cuda").manual_seed(9)
        for _ in range(3):
            chain.x.copy_(torch.randn(chain.x.shape, device="cuda", generator=gen).bfloat16())
            chain.latent.copy_(
                (torch.randn(chain.latent.shape, device="cuda", generator=gen) * 0.5).bfloat16()
            )
            chain.out.fill_(torch.nan)
            graph.replay()
            got = chain.result()
            monkeypatch.delenv(PDL_ENV)
            chain.run(early_trigger=False)
            ref = chain.result()
            monkeypatch.setenv(PDL_ENV, "1")
            _assert_bitwise(got, ref)
    finally:
        chain.mgr.shutdown()


@qf.needs_k3_decode
def test_decode_pdl_falls_back(monkeypatch):
    """No programmatic launch when the decode call itself launches a metadata kernel (a stale
    plan) or when the cache update wrote the query rows."""
    monkeypatch.setenv(PDL_ENV, "1")
    recorder = _record_decode(monkeypatch)
    chain = _Chain(1, 3000, seed=70)
    try:
        chain.run(early_trigger=True)
        ref = chain.result()
        assert recorder.calls[-1] is True
        # A stale plan noticed by the decode call itself: the plan kernel runs right before it.
        chain.run(
            early_trigger=True,
            before_decode=lambda: setattr(chain.md.fp4_mla_state, "packed_plan_key", None),
        )
        _assert_bitwise(chain.result(), ref)
        assert recorder.calls[-1] is False
        chain.md.fp4_mla_state.q_rows_external = False
        chain.decode()
        _assert_bitwise(chain.result(), ref)
        assert recorder.calls[-1] is False
    finally:
        chain.mgr.shutdown()


@qf.needs_k3_decode
@pytest.mark.parametrize("release_early", [True, False])
def test_deferred_reduce_release(release_early, monkeypatch):
    """``DeferredPackedReduce(release_early=False)``: the decode is launched without releasing its
    dependents at entry (``TRTLLM_FP4_MLA_REDUCE_PDL=1``); the reduction it recorded, launched
    afterwards, gives the outputs of the plain decode bitwise."""
    from tensorrt_llm._torch.attention.backends.fp4_mla.fp4_mla_packed_reduce import (
        DeferredPackedReduce,
    )

    monkeypatch.setenv("TRTLLM_FP4_MLA_REDUCE_PDL", "1")
    monkeypatch.setenv("TRTLLM_FP4_MLA_SPLIT_KV", "8")
    monkeypatch.delenv(TRIGGER_ENV, raising=False)
    recorder = _record_decode(monkeypatch)
    chain = _Chain(1, 6000, seed=90)
    try:
        chain.run(early_trigger=False)
        ref = chain.result()
        assert recorder.releases[-1] is True
        chain.out.fill_(torch.nan)
        with DeferredPackedReduce(release_early=release_early) as deferred:
            chain.run(early_trigger=False)
        assert recorder.releases[-1] is release_early
        assert deferred.args is not None, "the split reduction was not deferred"
        deferred.run()
        _assert_bitwise(chain.result(), ref)
    finally:
        chain.mgr.shutdown()


def test_late_tail_policy(k3_features):
    """The fused tail (``TRTLLM_K3_ULL2_MLA_TAIL``) is released by the decode's completion at most one 16-token tile."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.mla.helix_tail import (
        decode_releases_tail_early,
    )

    k3_features(ull2_mla_tail=False, ull3_mla_late_tail_max_tokens=16)
    assert decode_releases_tail_early(8) and decode_releases_tail_early(32)
    k3_features(ull2_mla_tail=True)
    assert not decode_releases_tail_early(8)
    assert not decode_releases_tail_early(16)
    assert decode_releases_tail_early(17)

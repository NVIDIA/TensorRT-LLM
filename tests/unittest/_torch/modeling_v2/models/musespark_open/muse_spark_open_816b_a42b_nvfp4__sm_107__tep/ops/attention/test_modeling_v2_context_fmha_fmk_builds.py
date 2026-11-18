# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the multi-instance context FMHA's optional builds (``context_fmha_fmk``).

Host (any machine):
  - the op's environment knobs (``MUSESPARK_OPEN_FMK_EARLY_QK``, ``MUSESPARK_OPEN_FMK_EXP_F16X2``)
    are read once at import, only ``"1"`` turns a build on, and both default off;
  - the kernel object refuses the combinations it does not build.

sm107:
  - ``p_smem`` (P in shared memory, early QK) writes the served build's output bit for bit, on
    whole items, cut items (merged pieces), several requests, masked tails and both head
    layouts (a TEP4 rank's 32 / 4 and a pipeline stage's 128 / 16);
  - ``exp_f16x2`` (fp16-pair exponent) stays as close to an fp32 softmax reference over the same
    E4M3 inputs as the served build (relative L2 at most 1.15 x the served build's + 1e-3), and
    its partial results merge (cut items);
  - the op's defaults follow the knobs;
  - a CUDA-graph replay of each build matches its eager call.
"""

import importlib
import math
import os
import subprocess
import sys
import textwrap

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.ops.attention import (  # noqa: E501
    context_fmha_fmk as fmk,
)

D, P = 64, 32
OP = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open."
    "muse_spark_open_816b_a42b_nvfp4__sm_107__tep.ops.attention.context_fmha_fmk"
)


def _sm() -> int:
    if not torch.cuda.is_available():
        return 0
    major, minor = torch.cuda.get_device_capability()
    return 10 * major + minor


needs_sm107 = pytest.mark.skipif(_sm() != 107, reason="the kernel is written for sm107")


# --------------------------------------------------------------------------- host


@pytest.mark.parametrize(
    "env, early, f16",
    [
        ({}, False, False),
        ({fmk.EARLY_QK_ENV: "1"}, True, False),
        ({fmk.EXP_F16X2_ENV: "1"}, False, True),
        ({fmk.EARLY_QK_ENV: "1", fmk.EXP_F16X2_ENV: " 1 "}, True, True),
        ({fmk.EARLY_QK_ENV: "0", fmk.EXP_F16X2_ENV: "true"}, False, False),
    ],
)
def test_knobs_read_at_import(env, early, f16):
    child_env = {
        k: v for k, v in os.environ.items() if k not in (fmk.EARLY_QK_ENV, fmk.EXP_F16X2_ENV)
    }
    child_env.update(env)
    code = textwrap.dedent(
        f"""
        import importlib
        m = importlib.import_module({OP!r})
        print(int(m.EARLY_QK), int(m.EXP_F16X2))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env=child_env, capture_output=True, text=True, check=True
    )
    assert out.stdout.split()[-2:] == [str(int(early)), str(int(f16))]


def test_kernel_refuses_unbuilt_combinations():
    pytest.importorskip("cutlass")
    fk = importlib.import_module(OP + "_kernel")
    with pytest.raises(AssertionError):
        fk.PagedContextFmhaFmk(32, 4, tile_n=32, s_stages=2, p_smem=True)
    with pytest.raises(AssertionError):
        fk.PagedContextFmhaFmk(32, 4, ablate_mufu=True, exp_f16x2=True)
    kernel = fk.PagedContextFmhaFmk(32, 4, p_smem=True, exp_f16x2=True)
    assert kernel.p_smem and kernel.exp_f16x2


# --------------------------------------------------------------------------- sm107

MIXES = {
    "tep_deep": (32, 4, [(86000, 8192)]),
    "tep_three": (32, 4, [(86000, 2900), (0, 2800), (150000, 2492)]),
    "tep_odd": (32, 4, [(63, 129), (64, 64), (1000, 255), (12345, 1000)]),
    "tep_short": (32, 4, [(0, 300)]),
    "stage_deep": (128, 16, [(106000, 4088)]),
    "stage_two": (128, 16, [(106000, 2500), (40000, 1588)]),
    "stage_fresh": (128, 16, [(0, 4088)]),
}


class _Call:
    """One context call on a synthetic paged pool: random pages, E4M3 inputs."""

    def __init__(self, heads, kv_heads, mix, seed):
        dev = torch.device("cuda")
        gen = torch.Generator(device=dev).manual_seed(seed)
        self.heads, self.kv_heads, self.mix = heads, kv_heads, mix
        self.q_lens = [q for _, q in mix]
        self.kv_lens = [c + q for c, q in mix]
        pages = [-(-k // P) for k in self.kv_lens]
        total = sum(pages)
        self.pool = torch.randn(2 * total, kv_heads, P, D, generator=gen, device=dev).to(
            torch.float8_e4m3fn
        )
        perm = torch.randperm(total, generator=gen, device=dev).to(torch.int32)
        self.table = torch.zeros(len(mix), 2, max(pages), dtype=torch.int32, device=dev)
        base = 0
        for b, n in enumerate(pages):
            self.table[b, 0, :n] = 2 * perm[base : base + n]
            self.table[b, 1, :n] = 2 * perm[base : base + n] + 1
            base += n
        self.cu_q = torch.tensor(
            [0] + torch.tensor(self.q_lens).cumsum(0).tolist(), dtype=torch.int32, device=dev
        )
        self.kvl = torch.tensor(self.kv_lens, dtype=torch.int32, device=dev)
        self.scales = torch.tensor(
            [math.log2(math.e) / math.sqrt(D), 1.0], dtype=torch.float32, device=dev
        )
        tq = sum(self.q_lens)
        self.q = (2.0 * torch.randn(tq, heads, D, generator=gen, device=dev)).to(
            torch.float8_e4m3fn
        )
        self.out = torch.empty(tq, heads, D, dtype=torch.bfloat16, device=dev)
        num_ctas = torch.cuda.get_device_properties(dev).multi_processor_count
        self.plan = fmk.plan_context_fmha(
            self.q_lens,
            self.kv_lens,
            num_ctas=num_ctas,
            num_heads=heads,
            num_kv_heads=kv_heads,
            device=dev,
        )
        self.ws = fmk.FmkWorkspace()
        self.ws.reserve(fmk.max_slots(num_ctas), 4, dev)

    def launch(self, **kw):
        fmk.paged_context_fmha_fmk(
            self.q,
            self.pool,
            self.table,
            self.cu_q,
            self.kvl,
            self.plan,
            self.scales[0:1],
            self.scales[1:2],
            self.out,
            workspace=self.ws,
            **kw,
        )

    def run(self, **kw):
        self.out.fill_(float("nan"))
        self.launch(**kw)
        torch.cuda.synchronize()
        return self.out.clone()

    def rel_l2(self, out, b=0):
        """Relative L2 error of ``out`` vs an fp32 softmax over the same E4M3 inputs, on
        sampled rows and heads of request ``b``."""
        heads, kv_heads = self.heads, self.kv_heads
        kl, ql = self.kv_lens[b], self.q_lens[b]
        row0 = sum(self.q_lens[:b])
        n = -(-kl // P)
        kp, vp = self.table[b, 0, :n].long(), self.table[b, 1, :n].long()
        k = self.pool[kp].float().permute(0, 2, 1, 3).reshape(n * P, kv_heads, D)[:kl]
        v = self.pool[vp].float().permute(0, 2, 1, 3).reshape(n * P, kv_heads, D)[:kl]
        err = ref2 = 0.0
        for r in sorted({0, ql // 3, ql // 2, ql - 1}):
            lim = kl - ql + r + 1
            for h in (0, heads // 2 - 1, heads - 1):
                g = h // (heads // kv_heads)
                s = (k[:lim, g] @ self.q[row0 + r, h].float()) / math.sqrt(D)
                ref = torch.softmax(s, 0) @ v[:lim, g]
                err += ((out[row0 + r, h].float() - ref) ** 2).sum().item()
                ref2 += (ref**2).sum().item()
        return math.sqrt(err / ref2)


@needs_sm107
@pytest.mark.parametrize("name", list(MIXES))
def test_p_smem_is_bitwise_the_served_build(name):
    heads, kv_heads, mix = MIXES[name]
    call = _Call(heads, kv_heads, mix, seed=len(name))
    served = call.run(p_smem=False, exp_f16x2=False)
    early = call.run(p_smem=True, exp_f16x2=False)
    assert bool(torch.isfinite(served).all())
    assert torch.equal(early, served)


@needs_sm107
@pytest.mark.parametrize("name", list(MIXES))
@pytest.mark.parametrize("p_smem", [False, True])
def test_exp_f16x2_accuracy(name, p_smem):
    heads, kv_heads, mix = MIXES[name]
    call = _Call(heads, kv_heads, mix, seed=3 + len(name))
    served = call.run(p_smem=False, exp_f16x2=False)
    f16 = call.run(p_smem=p_smem, exp_f16x2=True)
    assert bool(torch.isfinite(f16).all())
    for b in range(len(mix)):
        rel_served, rel_f16 = call.rel_l2(served, b), call.rel_l2(f16, b)
        print(f"{name} request {b}: rel L2 vs fp32 served {rel_served:.4e} f16x2 {rel_f16:.4e}")
        assert rel_f16 <= 1.15 * rel_served + 1e-3, (b, rel_f16, rel_served)
    if p_smem:
        # The P.V order and operands are the same with P in shared memory.
        assert torch.equal(f16, call.run(p_smem=False, exp_f16x2=True))


@needs_sm107
def test_knob_defaults_select_the_builds(monkeypatch):
    call = _Call(32, 4, [(5007, 300), (777, 1), (33019, 4097)], seed=11)
    served = call.run(p_smem=False, exp_f16x2=False)
    both = call.run(p_smem=True, exp_f16x2=True)
    monkeypatch.setattr(fmk, "EARLY_QK", True)
    monkeypatch.setattr(fmk, "EXP_F16X2", True)
    assert torch.equal(call.run(), both)
    monkeypatch.setattr(fmk, "EXP_F16X2", False)
    assert torch.equal(call.run(), served)
    monkeypatch.setattr(fmk, "EARLY_QK", False)
    assert torch.equal(call.run(), served)


@needs_sm107
@pytest.mark.parametrize("p_smem, f16", [(True, False), (False, True), (True, True)])
def test_graph_replay_matches_eager(p_smem, f16):
    call = _Call(32, 4, [(5007, 300), (777, 1), (33019, 4097)], seed=5)
    assert call.plan.merge_meta is not None
    kw = dict(p_smem=p_smem, exp_f16x2=f16)
    call.launch(**kw)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call.launch(**kw)
    eager = call.run(**kw)
    call.out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(call.out, eager)

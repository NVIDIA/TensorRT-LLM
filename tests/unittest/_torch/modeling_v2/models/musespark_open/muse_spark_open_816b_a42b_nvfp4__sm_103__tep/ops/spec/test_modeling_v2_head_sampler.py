# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-GPU tests of ``ops/spec/head_sampler.py`` (fast paths ``split_sampler`` / ``lm_head_inbox``).

Both entries claim the built-in head's tokens bit for bit: the processor's
soft cap on the gathered row, then ``torch.ops.trtllm.fused_sample_from_logits``.
Every cell of ``SplitSample.CELLS`` and ``GatherSplitSample.CELLS`` is driven
over ``MODELING_V2_SPLIT_SAMPLER_TRIALS`` logit sets (default 16) and compared token for token;
neutral rows also compare every virtual thread's online-softmax state with
the shipped pass 1 run on the FP32 capped row (``ref_fold``), as raw bits.

The gather is exercised on one GPU with a plain buffer standing in for this
rank's inbox copy: the test publishes the eight shards into it before the
call, or while kernel A already polls (from another stream, shard by shard
behind device-side delays), and checks the gathered row, the reset inbox and
the status word. The 8-rank fabric run (GEMMs writing through the multicast
address) is ``_head_sampler_rank_job.py``.

The library is built under torch's extension build root when it has none for
the current sources (minutes).
"""

from __future__ import annotations

import os

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec import (  # noqa: E501
    head_sampler as hs,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="SM103 only",
)

#: The checkpoint's metap multiplier and output soft cap.
MULTIPLIER = 0.1767766952966369
SOFT_CAP = 20.0
SHARD = 25256
RANKS = 8
VOCAB = SHARD * RANKS
TRIALS = int(os.environ.get("MODELING_V2_SPLIT_SAMPLER_TRIALS", "16"))
DEV = "cuda"


@pytest.fixture(scope="module", autouse=True)
def library():
    return hs.load_library()


def _raw(kind: str, rows: int, g: torch.Generator) -> torch.Tensor:
    """BF16 LM-head logits of one of the test's logit distributions."""
    if kind == "n10":
        r = torch.randn(rows, VOCAB, generator=g, device=DEV) * 10
    elif kind == "n40":
        r = torch.randn(rows, VOCAB, generator=g, device=DEV) * 40
    elif kind == "n120":
        r = torch.randn(rows, VOCAB, generator=g, device=DEV) * 120
    elif kind == "spiky":
        r = torch.randn(rows, VOCAB, generator=g, device=DEV) * 20
        idx = torch.randint(0, VOCAB, (rows, 64), generator=g, device=DEV)
        r.scatter_(1, idx, 150 + 200 * torch.rand(rows, 64, generator=g, device=DEV))
    elif kind == "peaked":
        r = torch.randn(rows, VOCAB, generator=g, device=DEV) * 8
        idx = torch.randint(0, VOCAB, (rows, 1), generator=g, device=DEV)
        r.scatter_(1, idx, torch.full((rows, 1), 120.0, device=DEV))
    elif kind == "flat":
        r = torch.randn(rows, VOCAB, generator=g, device=DEV)
    elif kind == "sat":
        r = torch.randn(rows, VOCAB, generator=g, device=DEV) * 600
    else:
        raise ValueError(kind)
    return r.to(torch.bfloat16)


def _with_zeros(raw: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    pick = torch.rand(raw.shape, generator=g, device=DEV)
    raw = raw.clone()
    raw[pick < 0.01] = 0.0
    raw[(pick >= 0.01) & (pick < 0.02)] = -0.0
    return raw


def _params(rows: int, spec: dict, g: torch.Generator, gs: torch.Generator):
    temperature = spec.get("temperature", 1.0)
    temps = torch.full((rows,), temperature, device=DEV)
    top_ks = torch.zeros(rows, dtype=torch.int32, device=DEV)
    top_ps = torch.ones(rows, device=DEV)
    min_ps = torch.zeros(rows, device=DEV)
    if spec.get("filters") == "mixed":
        # rows 0..: neutral, top-k, top-p, min-p, top-k + top-p, INT32_MAX top-k, neutral, top-p 0.5
        if rows >= 8:
            top_ks[1] = 50
            top_ps[2] = 0.9
            min_ps[3] = 0.05
            top_ks[4] = 64
            top_ps[4] = 0.8
            top_ks[5] = 2**31 - 1
            top_ps[7] = 0.5
    n = rows if spec.get("rng", "row") == "row" else 1
    seed = torch.randint(0, 2**62, (n,), generator=gs, dtype=torch.int64).to(DEV)
    offset = torch.randint(0, 2**40, (n,), generator=gs, dtype=torch.int64).to(DEV)
    return temps, top_ks, top_ps, min_ps, seed, offset


def _neutral(top_ks, top_ps, min_ps, vocab) -> torch.Tensor:
    return ((top_ks <= 0) | (top_ks >= vocab)) & (top_ps >= 1.0) & (min_ps <= 0.0)


def _state_bits(states: torch.Tensor, rows: int) -> torch.Tensor:
    return states[: rows * hs.VIRTUAL_THREADS * hs.STATE_WORDS].view(rows, hs.VIRTUAL_THREADS, 3)


def test_soft_cap_is_the_torch_chain():
    g = torch.Generator(device=DEV).manual_seed(5)
    m, inv, cap = hs.cap_constants(MULTIPLIER, SOFT_CAP)
    for kind in ("n10", "n40", "n120", "sat", "flat"):
        raw = _with_zeros(_raw(kind, 8, g), g)
        ours = torch.ops.musespark_open_head_sampler.cap(raw, m, inv, cap)
        ref = hs.reference_capped(raw, MULTIPLIER, SOFT_CAP)
        assert torch.equal(ours.view(torch.int32), ref.view(torch.int32)), kind
    special = torch.tensor(
        [0.0, -0.0, float("inf"), float("-inf"), 1e-30, -1e-30, 3e38, -3e38], device=DEV
    ).to(torch.bfloat16)
    ours = torch.ops.musespark_open_head_sampler.cap(special.view(1, -1).contiguous(), m, inv, cap)
    ref = hs.reference_capped(special.view(1, -1), MULTIPLIER, SOFT_CAP)
    assert torch.equal(ours.view(torch.int32), ref.view(torch.int32))


@pytest.mark.parametrize("cell", hs.SplitSample.CELLS, ids=lambda c: c.why[:48])
def test_split_sample_cells(cell):
    spec = cell.spec
    rows = spec["rows"]
    g = torch.Generator(device=DEV).manual_seed(11 + rows)
    gs = torch.Generator().manual_seed(13)
    states = hs.new_states(rows, torch.device(DEV))
    ref_states = hs.new_states(rows, torch.device(DEV))
    m, inv, cap = hs.cap_constants(MULTIPLIER, SOFT_CAP)
    mismatches = draws = state_mismatches = 0
    for _ in range(TRIALS):
        raw = _raw(spec["kind"], rows, g)
        temps, top_ks, top_ps, min_ps, seed, offset = _params(rows, spec, g, gs)
        args = (raw, temps, top_ks, top_ps, min_ps, seed, offset, MULTIPLIER, SOFT_CAP, states)
        hs.split_sample.is_valid(*args)
        tokens = hs.split_sample(*args)
        ref = hs.split_sample.reference(*args)
        mismatches += int((tokens != ref).sum())
        draws += rows
        neutral = _neutral(top_ks, top_ps, min_ps, VOCAB)
        capped = hs.reference_capped(raw, MULTIPLIER, SOFT_CAP)
        torch.ops.musespark_open_head_sampler.ref_fold(capped, temps, ref_states)
        a = _state_bits(states, rows)[neutral]
        b = _state_bits(ref_states, rows)[neutral]
        state_mismatches += int((a != b).sum())
    print(
        f"split_sampler cell {cell.why[:40]!r}: {draws} draws, {mismatches} token mismatches, "
        f"{state_mismatches} state-word mismatches"
    )
    assert mismatches == 0
    assert state_mismatches == 0


#: The LM-head tactic per rank in the served autotuner cache (as ``_head_sampler_rank_job.py``).
SERVED_TACTICS = {0: (64, 16, 1, 11), 3: (128, 8, 2, 6), 7: (64, 8, 1, 11)}
DEFAULT_TACTIC = (64, 8, 1, 12)


def test_split_sample_on_lm_head_rows():
    """Rows an 8-rank LM head produces, as ``_head_sampler_rank_job.py --steps`` samples them.

    Rank ``r``'s shard is ``randn`` seeded ``1000 + r`` and its served split-K
    tactic, so every gathered row of that job is rebuilt here on one GPU: 17
    variants x 8 replays x 24 steps x 8 rows = 26,112 draws, each sampled twice
    by the split sampler and compared with the shipped sampler. Among them is
    one draw (tag 4005, step 17, row 7) whose target falls where two
    neighbouring spans of the float scan overlap, so two threads claim it: the
    shipped kernel lets both walk and keeps the crossing written last, which a
    single elected walker did not reproduce (it returned one of two tokens,
    run to run).
    """
    from tensorrt_llm._torch.modules.low_m_gemm import _SplitKGemmRunner

    hidden, steps, replays, rows = 8192, 24, 8, 8
    runner = _SplitKGemmRunner(has_bias=False, pdl=False)
    weights = []
    for r in range(RANKS):
        g = torch.Generator(device=DEV).manual_seed(1000 + r)
        w = (torch.randn(SHARD, hidden, generator=g, device=DEV) * 0.03).to(torch.bfloat16)
        weights.append(w.t())
    temps = torch.ones(rows, device=DEV)
    top_ks = torch.zeros(rows, dtype=torch.int32, device=DEV)
    top_ps = torch.ones(rows, device=DEV)
    min_ps = torch.zeros(rows, device=DEV)
    states = hs.new_states(rows, torch.device(DEV))
    shard_out = torch.empty(rows, SHARD, dtype=torch.bfloat16, device=DEV)
    draws = mismatches = unstable = 0
    for tag in (1000 * (v + 1) + rep for v in range(17) for rep in range(replays)):
        gc = torch.Generator(device=DEV).manual_seed(tag)
        xs = (torch.randn(steps, rows, hidden, generator=gc, device=DEV) * 2.0).to(torch.bfloat16)
        gs = torch.Generator().manual_seed(tag + 1)
        seeds = torch.randint(0, 2**62, (steps, rows), generator=gs, dtype=torch.int64).to(DEV)
        offsets = torch.randint(0, 2**40, (steps, rows), generator=gs, dtype=torch.int64).to(DEV)
        for s in range(steps):
            parts = []
            for r in range(RANKS):
                runner(
                    [xs[s], weights[r]], tactic=SERVED_TACTICS.get(r, DEFAULT_TACTIC), out=shard_out
                )
                parts.append(shard_out.clone())
            row = torch.cat(parts, dim=1)
            args = (
                temps,
                top_ks,
                top_ps,
                min_ps,
                seeds[s],
                offsets[s],
                MULTIPLIER,
                SOFT_CAP,
                states,
            )
            ref = hs.split_sample.reference(row, *args)
            first = hs.split_sample(row, *args)
            second = hs.split_sample(row, *args)
            draws += rows
            mismatches += int((first != ref).sum())
            unstable += int((first != second).sum())
            if tag == 4005 and s == 17:
                for _ in range(100):
                    again = hs.split_sample(row, *args)
                    unstable += int(again[7] != first[7])
                    mismatches += int(again[7] != ref[7])
    print(f"LM-head rows: {draws} draws, {mismatches} token mismatches, {unstable} unstable")
    assert draws == 26112
    assert mismatches == 0
    assert unstable == 0


def _inbox_from(raw: torch.Tensor, slot_rows: int) -> torch.Tensor:
    """The inbox bytes the eight GEMMs would publish: ``[ranks][slot_rows][shard]`` BF16."""
    rows = raw.shape[0]
    inbox = torch.full((RANKS, slot_rows, SHARD), -1, dtype=torch.int16, device=DEV)
    inbox[:, :rows] = raw.view(rows, RANKS, SHARD).permute(1, 0, 2).view(torch.int16)
    return inbox


@pytest.mark.parametrize("cell", hs.GatherSplitSample.CELLS, ids=lambda c: c.why[:48])
def test_gather_split_sample_cells(cell):
    spec = cell.spec
    rows, slot_rows = spec["rows"], hs.MAX_ROWS
    sample = spec.get("sample", True)
    g = torch.Generator(device=DEV).manual_seed(17 + rows)
    gs = torch.Generator().manual_seed(19)
    states = hs.new_states(slot_rows, torch.device(DEV))
    status = torch.zeros(1, dtype=torch.int32, device=DEV)
    inbox = torch.full((RANKS, slot_rows, SHARD), -1, dtype=torch.int16, device=DEV)
    gathered = torch.empty((rows, VOCAB), dtype=torch.bfloat16, device=DEV)
    mismatches = gather_mismatches = left = 0
    side = torch.cuda.Stream()
    if spec["publish"] == "during":
        # Load the producer's kernels before kernel A spins: with lazy module loading, a
        # first launch while A polls may wait for the context to go idle, which A never
        # does before the producer has run.
        with torch.cuda.stream(side):
            torch.cuda._sleep(1)
            inbox[0, :rows].copy_(inbox[0, :rows].clone())
        torch.cuda.synchronize()
    for trial in range(TRIALS):
        raw = _raw(spec["kind"], rows, g)
        if spec.get("zeros"):
            raw = _with_zeros(raw, g)
        temps, top_ks, top_ps, min_ps, seed, offset = _params(rows, spec, g, gs)
        published = _inbox_from(raw, slot_rows)
        args = (
            inbox,
            gathered,
            temps,
            top_ks,
            top_ps,
            min_ps,
            seed,
            offset,
            MULTIPLIER,
            SOFT_CAP,
            states,
            status,
            SHARD,
            slot_rows,
            sample,
        )
        ref = hs.gather_split_sample.reference(
            published,
            gathered,
            temps,
            top_ks,
            top_ps,
            min_ps,
            seed,
            offset,
            MULTIPLIER,
            SOFT_CAP,
            states,
            status,
            SHARD,
            slot_rows,
            sample,
        )
        hs.gather_split_sample.is_valid(*args)
        if spec["publish"] == "before":
            inbox.copy_(published)
            out = hs.gather_split_sample(*args)
        else:
            # Kernel A starts polling an empty inbox; the shards land later, from the side
            # stream, in reverse rank order, each behind a device-side delay.
            torch.cuda.synchronize()
            out = hs.gather_split_sample(*args)
            with torch.cuda.stream(side):
                for r in reversed(range(RANKS)):
                    torch.cuda._sleep(20000 * (1 + (trial + r) % 3))
                    inbox[r, :rows].copy_(published[r, :rows])
            torch.cuda.synchronize()
        torch.cuda.synchronize()
        if sample:
            mismatches += int((out != ref).sum())
            expect = published[:, :rows].permute(1, 0, 2).reshape(rows, VOCAB)
        else:
            expect = ref.view(torch.int16)
        gather_mismatches += int((gathered.view(torch.int16) != expect).sum())
        left += int((inbox != -1).sum())
    assert int(status.item()) == 0
    print(
        f"lm_head_inbox cell {cell.why[:40]!r}: {TRIALS * rows} rows, {mismatches} token mismatches, "
        f"{gather_mismatches} gathered-element mismatches, {left} inbox words not reset"
    )
    assert gather_mismatches == 0
    assert left == 0
    assert mismatches == 0


def test_graph_replay_with_pdl():
    """Both entries captured in one CUDA graph with PDL edges, replayed on new inputs."""
    rows = 8
    g = torch.Generator(device=DEV).manual_seed(23)
    gs = torch.Generator().manual_seed(29)
    spec = dict(rows=rows, temperature=1.0, rng="row")
    raw = _raw("n40", rows, g)
    temps, top_ks, top_ps, min_ps, seed, offset = _params(rows, spec, g, gs)
    states_a = hs.new_states(rows, torch.device(DEV))
    states_b = hs.new_states(rows, torch.device(DEV))
    status = torch.zeros(1, dtype=torch.int32, device=DEV)
    inbox = torch.full((RANKS, rows, SHARD), -1, dtype=torch.int16, device=DEV)
    staged = torch.full_like(inbox, -1)
    gathered = torch.empty((rows, VOCAB), dtype=torch.bfloat16, device=DEV)
    x = raw.clone()

    def step():
        t1 = hs.split_sample(
            x, temps, top_ks, top_ps, min_ps, seed, offset, MULTIPLIER, SOFT_CAP, states_a, True
        )
        inbox.copy_(staged)
        t2 = hs.gather_split_sample(
            inbox,
            gathered,
            temps,
            top_ks,
            top_ps,
            min_ps,
            seed,
            offset,
            MULTIPLIER,
            SOFT_CAP,
            states_b,
            status,
            SHARD,
            rows,
            True,
            True,
        )
        return t1, t2

    staged.copy_(_inbox_from(x, rows))
    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        t1, t2 = step()
    bad = 0
    for _ in range(max(TRIALS, 8)):
        x.copy_(_raw("n10", rows, g))
        staged.copy_(_inbox_from(x, rows))
        seed.copy_(torch.randint(0, 2**62, (rows,), generator=gs, dtype=torch.int64).to(DEV))
        graph.replay()
        torch.cuda.synchronize()
        ref = hs.split_sample.reference(
            x, temps, top_ks, top_ps, min_ps, seed, offset, MULTIPLIER, SOFT_CAP, states_a
        )
        bad += int((t1 != ref).sum()) + int((t2 != ref).sum())
        assert int((inbox != -1).sum()) == 0
    assert int(status.item()) == 0
    assert bad == 0


#: A poller for ``test_every_lm_head_producer_publishes_each_word_once``: it raises
#: ``started`` once it polls, the GEMM is launched behind a kernel that waits for that
#: flag, and it records per BF16 half of the output: the sentinel seen before any value
#: (bit 0), a later value different from the first (bit 1), the sentinel again after a
#: value (bit 2), and the first value seen.
_WAITER_CUDA = r"""
#include <cstdint>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ uint32_t loadRelaxed(const uint32_t* p) {
  uint32_t v; asm volatile("ld.relaxed.sys.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ uint32_t loadAcquire(const int* p) {
  uint32_t v; asm volatile("ld.acquire.sys.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ unsigned long long globalTimer() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
__global__ void pollerKernel(const uint32_t* buf, uint16_t* first, uint8_t* flags, int64_t words,
                             int* started, const int* done, unsigned long long timeoutNs) {
  int64_t stride = (int64_t)gridDim.x * blockDim.x;
  int64_t start = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (start == 0) { asm volatile("st.release.sys.global.b32 [%0], %1;" :: "l"(started), "r"(1) : "memory"); }
  unsigned long long t0 = globalTimer();
  int tail = -1;
  for (;;) {
    for (int64_t i = start; i < words; i += stride) {
      uint32_t v = loadRelaxed(buf + i);
      for (int h = 0; h < 2; ++h) {
        uint16_t x = (uint16_t)(v >> (16 * h));
        int64_t k = 2 * i + h;
        uint16_t f = first[k];
        if (x == 0xFFFFu) { flags[k] |= (f == 0xFFFFu) ? 1 : 4; }
        else if (f == 0xFFFFu) { first[k] = x; }
        else if (x != f) { flags[k] |= 2; }
      }
    }
    if (tail >= 0) { if (++tail >= 3) return; }
    else if (loadAcquire(done) == 1) { tail = 0; }
    if (globalTimer() - t0 > timeoutNs) return;
  }
}
__global__ void waitStartedKernel(const int* started) { while (loadAcquire(started) == 0) { __nanosleep(100); } }
__global__ void setDoneKernel(int* done) {
  asm volatile("st.release.sys.global.b32 [%0], %1;" :: "l"(done), "r"(1) : "memory");
}

void poll(torch::Tensor buf, torch::Tensor first, torch::Tensor flags, torch::Tensor started,
          torch::Tensor done, int64_t blocks, int64_t timeoutMs) {
  pollerKernel<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const uint32_t*)buf.data_ptr(), (uint16_t*)first.data_ptr(), (uint8_t*)flags.data_ptr(),
      buf.numel() * buf.element_size() / 4, started.data_ptr<int>(), done.data_ptr<int>(),
      (unsigned long long)timeoutMs * 1000000ull);
}
void wait_started(torch::Tensor started) {
  waitStartedKernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(started.data_ptr<int>());
}
void set_done(torch::Tensor done) {
  setDoneKernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(done.data_ptr<int>());
}
"""

_WAITER_CPP = r"""
#include <torch/extension.h>
void poll(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t);
void wait_started(torch::Tensor);
void set_done(torch::Tensor);
"""


def _lm_head_producers(x, w_t):
    """Every (name, runner, tactic) the lm_head_inbox LM-head call can run at this shape (``lm_head_gemm_into``)."""
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
    from tensorrt_llm._torch.modules.low_m_gemm import (
        _CuBLASGemmRunner,
        _DirectGemmRunner,
        _SplitKGemmRunner,
    )

    pdl = get_env_enable_pdl()
    cublas = _CuBLASGemmRunner(has_bias=False)
    splitk = _SplitKGemmRunner(has_bias=False, pdl=pdl)
    direct = _DirectGemmRunner(pdl=pdl)
    producers = [("cublas", cublas, t) for t in cublas.get_valid_tactics([x, w_t], None)]
    producers += [("splitk", splitk, tuple(t)) for t in splitk.get_valid_tactics([x, w_t], None)]
    producers += [("direct", direct, tuple(t)) for t in direct.get_valid_tactics([x, w_t], None)]
    return producers


def test_every_lm_head_producer_publishes_each_word_once(tmp_path):
    """Every runner and tactic the lm_head_inbox LM-head call can pick stores each output half once, finally.

    The inbox protocol needs it: a poller sees a word as published as soon as
    neither half is the sentinel, so a producer that stored a half twice, or
    a provisional value first, would hand the sampler a wrong logit. The
    poller is proven to run while the GEMM writes (the GEMM is gated on its
    start flag); the sentinel must be seen before each half's value, never a
    second value or the sentinel again, and the final value equals a plain
    call's (deterministic).
    """
    from torch.utils.cpp_extension import load_inline

    ext = load_inline(
        "lm_head_inbox_test_poller",
        _WAITER_CPP,
        cuda_sources=_WAITER_CUDA,
        functions=["poll", "wait_started", "set_done"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_103a,code=sm_103a"],
        build_directory=str(tmp_path),
    )
    rows, hidden = 8, 8192
    g = torch.Generator(device=DEV).manual_seed(4242)
    x = (torch.randn(rows, hidden, generator=g, device=DEV) * 2.0).to(torch.bfloat16)
    w_t = (torch.randn(SHARD, hidden, generator=g, device=DEV) * 0.03).to(torch.bfloat16).t()
    producers = _lm_head_producers(x, w_t)
    assert {name for name, _, _ in producers} == {"cublas", "splitk", "direct"}
    halves = rows * SHARD
    first = torch.empty(halves, dtype=torch.int16, device=DEV)
    flags = torch.empty(halves, dtype=torch.uint8, device=DEV)
    started = torch.zeros(1, dtype=torch.int32, device=DEV)
    done = torch.zeros(1, dtype=torch.int32, device=DEV)
    out = torch.empty((rows, SHARD), dtype=torch.bfloat16, device=DEV)
    ref = torch.empty_like(out)
    side = torch.cuda.Stream()
    # Load the three kernels before the first gated launch: with lazy module loading,
    # a first launch while the poller spins may wait for the context to go idle.
    done.fill_(1)
    with torch.cuda.stream(side):
        ext.poll(out, first, flags, started, done, 16, 5000)
        ext.set_done(done)
    torch.cuda.synchronize()
    ext.wait_started(started)
    torch.cuda.synchronize()
    bad = []
    for name, runner, tactic in producers:
        runner([x, w_t], tactic=tactic, out=ref)  # compile and warm
        torch.cuda.synchronize()
        totals = dict(pre=0, multi=0, back=0, first_ne_final=0, unseen=0, final_ne_ref=0)
        for _ in range(4):
            out.view(torch.int32).fill_(-1)
            first.fill_(-1)
            flags.zero_()
            started.zero_()
            done.zero_()
            torch.cuda.synchronize()
            with torch.cuda.stream(side):
                ext.poll(out, first, flags, started, done, 16, 5000)
            ext.wait_started(started)
            runner([x, w_t], tactic=tactic, out=out)
            ext.set_done(done)
            torch.cuda.synchronize()
            final = out.view(torch.int16).reshape(-1)
            fl = flags.to(torch.int32)
            seen = first != -1
            totals["pre"] += int(((fl & 1) != 0).sum())
            totals["multi"] += int(((fl & 2) != 0).sum())
            totals["back"] += int(((fl & 4) != 0).sum())
            totals["first_ne_final"] += int((seen & (first != final)).sum())
            totals["unseen"] += int((~seen).sum())
            totals["final_ne_ref"] += int((out.view(torch.int16) != ref.view(torch.int16)).sum())
        print(f"{name} {tactic}: {totals} of {4 * halves} halves")
        if totals["multi"] or totals["back"] or totals["first_ne_final"] or totals["final_ne_ref"]:
            bad.append((name, tactic, totals))
        # The poller ran while the GEMM wrote: it saw (nearly) every half unwritten first.
        assert totals["pre"] >= 0.9 * 4 * halves, (name, tactic, totals)
    print(f"{len(producers)} producers, {len(bad)} with a transient or non-final store")
    assert not bad, bad

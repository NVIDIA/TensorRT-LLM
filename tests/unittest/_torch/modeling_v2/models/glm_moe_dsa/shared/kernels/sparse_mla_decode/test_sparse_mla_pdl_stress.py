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
"""Delayed-producer and delayed-consumer stress of the sparse_mla_decode kernel under PDL.

The sparse_mla_decode kernel initializes its mbarriers and arrives on the cluster barrier
before ``griddepcontrol.wait``, issues ``griddepcontrol.launch_dependents``
before its compute, gathers pool rows with bulk copies and pushes partial
results to its cluster peers with ``st.async``. This test checks that none of
that reads an input early or writes the output early.

Every iteration is a chain without a plain kernel boundary:

  producer (PDL)  -> sparse_mla_decode (PDL) -> consumer (PDL) -> next producer ...

The producer lets its dependents launch at once, waits for its predecessor,
spins, and only then writes everything sparse_mla_decode reads: the FP8 query, the index
list, the KV lengths, the bmm scales and a block of pool rows the index list
points into. The consumer lets its dependents launch at once, waits for sparse_mla_decode,
optionally spins, and copies the output out. Consecutive iterations use
different inputs, and each input set alternates between two versions of its
pool rows, so an early read of any input, a late write of the output, or a
stale cache line gives a different result. Each output must be bit-identical
to the output of the same inputs computed in isolation (sparse_mla_decode is deterministic;
this is checked too); the isolated outputs are checked against the FP64 oracle.

Variants: eager chain with and without PDL on the helper kernels, the chain
captured in a CUDA graph and replayed, and the eager chain while a second
stream saturates HBM. ``%globaltimer`` stamps of the helper kernels and a
CUPTI trace show whether the early launches happened.
"""

import os
import statistics

import pytest
import torch
from sparse_mla_harness import E4M3_MAX, error_stats, oracle_attention, record
from sparse_mla_pdl_helpers import load_pdl_helpers

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.sparse_mla_decode import (  # noqa: E501
    sparse_mla,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 10
    or not sparse_mla.op_available(),
    reason="needs an SM10x GPU and the sparse_mla_decode kernel",
)

NUM_HEADS = 8
HEAD_DIM = 576
KV_LORA_RANK = 512
TOP_K = 2048
STATIC_ROWS = 120_000
OWNED_ROWS = 512
NUM_SETS = 6
ITERS = int(os.environ.get("GLM_MOE_DSA_SPARSE_MLA_DECODE_STRESS_ITERS", "400"))
GRAPH_REPLAYS = int(os.environ.get("GLM_MOE_DSA_SPARSE_MLA_DECODE_STRESS_GRAPH_REPLAYS", "8"))
PRODUCER_SPINS_NS = (0, 1000, 5000, 20000)
CONSUMER_SPINS_NS = (0, 0, 3000, 10000)
RESULTS = "sparse_mla_stress.jsonl"
# compute-sanitizer serializes and slows the instrumented kernel; the early-launch evidence is
# then not expected, only the correctness of every iteration.
UNDER_SANITIZER = os.environ.get("GLM_MOE_DSA_SPARSE_MLA_DECODE_UNDER_SANITIZER") == "1"


def _e4m3_bytes(shape, std: float, gen: torch.Generator) -> torch.Tensor:
    values = torch.randn(*shape, device="cuda", generator=gen) * std
    return values.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)


class StressCase:
    """Live buffers read by sparse_mla_decode, staged input sets and their isolated reference outputs."""

    def __init__(self, ctx_lens, seq_len_q: int, seed: int):
        self.seq_len_q = seq_len_q
        self.num_seqs = len(ctx_lens)
        assert self.num_seqs <= 4
        self.rows = self.num_seqs * seq_len_q
        gen = torch.Generator(device="cuda").manual_seed(seed)
        cpu_gen = torch.Generator(device="cpu").manual_seed(seed)
        num_pool_rows = STATIC_ROWS + NUM_SETS * OWNED_ROWS
        self.pool = _e4m3_bytes((num_pool_rows, HEAD_DIM), 1.0, gen)
        self.q = torch.zeros(self.rows, NUM_HEADS * HEAD_DIM, dtype=torch.uint8, device="cuda")
        self.indices = torch.zeros(self.rows, TOP_K, dtype=torch.int32, device="cuda")
        self.kv_lens = torch.zeros(4, dtype=torch.int32, device="cuda")
        self.scales = torch.zeros(4, dtype=torch.float32, device="cuda")
        self.out = torch.zeros(
            self.rows, NUM_HEADS * KV_LORA_RANK, dtype=torch.bfloat16, device="cuda"
        )
        self.sets = []
        for k in range(NUM_SETS):
            kv_lens = [c + seq_len_q + 3 * k for c in ctx_lens]
            owned_base = STATIC_ROWS + k * OWNED_ROWS
            limits = []
            idx = torch.empty(self.rows, TOP_K, dtype=torch.int64)
            for row in range(self.rows):
                seq, pos = divmod(row, seq_len_q)
                usable = kv_lens[seq] - (seq_len_q - 1 - pos)
                limits.append(usable)
                entries = torch.randint(0, STATIC_ROWS, (TOP_K,), generator=cpu_gen)
                owned = torch.rand(TOP_K, generator=cpu_gen) < 0.125
                entries[owned] = owned_base + torch.randint(
                    0, OWNED_ROWS, (int(owned.sum()),), generator=cpu_gen
                )
                negative = torch.rand(TOP_K, generator=cpu_gen) < 0.02
                entries[negative] = -1
                entries[64:96] = entries[0:32]  # repeated rows count per occurrence
                idx[row] = entries
            kv_tensor = torch.zeros(4, dtype=torch.int32)
            kv_tensor[: self.num_seqs] = torch.tensor(kv_lens, dtype=torch.int32)
            softmax_scale = 0.0625 * (1.0 + 0.05 * k)
            scales = torch.tensor(
                [softmax_scale, softmax_scale * 1.4426950408889634, 1.0 + 0.01 * k, 0.0],
                dtype=torch.float32,
            )
            variants = [
                _e4m3_bytes((OWNED_ROWS, HEAD_DIM), 2.0, gen),
                _e4m3_bytes((OWNED_ROWS, HEAD_DIM), 0.5, gen),
            ]
            self.sets.append(
                {
                    "q": _e4m3_bytes((self.rows, NUM_HEADS * HEAD_DIM), 1.0 + 0.25 * k, gen),
                    "indices": idx.to(torch.int32).cuda(),
                    "kv_lens": kv_tensor.cuda(),
                    "scales": scales.cuda(),
                    "owned_base": owned_base,
                    "variants": variants,
                    "limits": limits,
                }
            )
        self.segments = {}
        for k, s in enumerate(self.sets):
            for v in range(2):
                owned = self.pool[s["owned_base"] : s["owned_base"] + OWNED_ROWS]
                pairs = [
                    (self.q, s["q"]),
                    (self.indices, s["indices"]),
                    (self.kv_lens, s["kv_lens"]),
                    (self.scales, s["scales"]),
                    (owned, s["variants"][v]),
                ]
                for dst, src in pairs:
                    assert dst.is_contiguous() and src.is_contiguous() and dst.nbytes == src.nbytes
                    assert dst.nbytes % 16 == 0
                self.segments[(k, v)] = tuple(
                    torch.tensor(values, dtype=torch.int64, device="cuda")
                    for values in (
                        [d.data_ptr() for d, _ in pairs],
                        [s_.data_ptr() for _, s_ in pairs],
                        [d.nbytes // 16 for d, _ in pairs],
                    )
                )
        self.references = {}

    def apply_plain(self, k: int, v: int) -> None:
        s = self.sets[k]
        self.q.copy_(s["q"])
        self.indices.copy_(s["indices"])
        self.kv_lens.copy_(s["kv_lens"])
        self.scales.copy_(s["scales"])
        self.pool[s["owned_base"] : s["owned_base"] + OWNED_ROWS].copy_(s["variants"][v])

    def launch_sparse_mla(self) -> None:
        sparse_mla.sparse_mla_decode(
            self.q.view(self.rows, NUM_HEADS, HEAD_DIM),
            self.pool.data_ptr(),
            self.indices,
            self.scales[0:2],
            self.scales[2:3],
            self.kv_lens,
            self.seq_len_q,
            self.out,
        )

    def build_references(self, case: str) -> None:
        """Isolated outputs of every (set, version); determinism and FP64-oracle checks."""
        for k in range(NUM_SETS):
            for v in range(2):
                runs = []
                for _ in range(2):
                    self.apply_plain(k, v)
                    torch.cuda.synchronize()
                    self.out.fill_(float("nan"))
                    self.launch_sparse_mla()
                    torch.cuda.synchronize()
                    runs.append(self.out.clone())
                assert torch.equal(runs[0], runs[1]), (
                    "sparse_mla_decode output is not deterministic"
                )
                s = self.sets[k]
                oracle = oracle_attention(
                    self.q,
                    self.pool.view(torch.float8_e4m3fn),
                    self.indices,
                    float(s["scales"][0]),
                    float(s["scales"][2]),
                    list_limit=s["limits"],
                )
                err = error_stats(runs[0].double(), oracle)
                record(
                    RESULTS,
                    f"{case}:reference",
                    {"set": k, "version": v, "sparse_mla_vs_oracle": err},
                )
                assert err["nonfinite"] == 0
                assert err["nrms"] < 0.05, err
                assert torch.allclose(
                    runs[0].double().reshape(oracle.shape), oracle, atol=0.12, rtol=0.01
                )
                self.references[(k, v)] = runs[0]
        # The two versions of each set's pool rows must change the output, or a stale read of
        # those rows would go unnoticed.
        for k in range(NUM_SETS):
            assert not torch.equal(self.references[(k, 0)], self.references[(k, 1)])


def _schedule(iterations: int, seed: int):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    plan = []
    for i in range(iterations):
        k = i % NUM_SETS
        v = (i // NUM_SETS) % 2
        p = PRODUCER_SPINS_NS[int(torch.randint(0, len(PRODUCER_SPINS_NS), (1,), generator=gen))]
        c = CONSUMER_SPINS_NS[int(torch.randint(0, len(CONSUMER_SPINS_NS), (1,), generator=gen))]
        plan.append((k, v, p, c))
    return plan


# A spin kernel ahead of an eager chain (about 0.2 s), so that the host enqueues the whole
# chain before the GPU reaches it and the GPU, not the host, paces the chain.
GATE_CYCLES = int(4e8)


def _enqueue_chain(case: StressCase, helpers, plan, outs, stamps, use_pdl: bool) -> None:
    for i, (k, v, producer_spin, consumer_spin) in enumerate(plan):
        dsts, srcs, counts = case.segments[(k, v)]
        helpers.delayed_scatter(dsts, srcs, counts, producer_spin, stamps[i, 0:3], use_pdl, 64)
        case.launch_sparse_mla()
        helpers.pdl_consumer(outs[i], case.out, consumer_spin, stamps[i, 3:5], use_pdl)


def _mismatches(case: StressCase, plan, outs) -> list:
    bad = []
    for i, (k, v, _, _) in enumerate(plan):
        if not torch.equal(outs[i], case.references[(k, v)]):
            diff = (outs[i].float() - case.references[(k, v)].float()).abs()
            bad.append(
                {
                    "iteration": i,
                    "set": k,
                    "version": v,
                    "max_abs": float(diff.nan_to_num(1e30).max()),
                }
            )
    return bad


def _stamp_summary(stamps: torch.Tensor, plan) -> dict:
    """Early-launch evidence from the helper kernels' ``%globaltimer`` stamps.

    A consumer that started well before its ``griddepcontrol.wait`` returned was
    launched while sparse_mla_decode was still running, i.e. through sparse_mla_decode's early
    ``launch_dependents``.
    """
    s = stamps.cpu().tolist()
    consumer_wait = [row[4] - row[3] for row in s]
    producer_wait = [row[1] - row[0] for row in s]
    return {
        "consumer_wait_ns_median": statistics.median(consumer_wait),
        "consumer_started_before_sparse_mla_done": sum(1 for x in consumer_wait if x > 500)
        / len(s),
        "producer_wait_ns_median": statistics.median(producer_wait),
        "iterations": len(plan),
    }


@pytest.fixture(scope="module")
def helpers():
    return load_pdl_helpers()


CASES = {
    "one_seq_r6": ([40000], 6),
    "three_seqs_r2": ([3000, 100, 9000], 2),
}


@pytest.mark.parametrize("use_pdl", [True, False], ids=["pdl", "no_pdl"])
@pytest.mark.parametrize("case_name", list(CASES))
def test_sparse_mla_delayed_producer_consumer_eager(helpers, case_name, use_pdl):
    ctx_lens, seq_len_q = CASES[case_name]
    case = StressCase(ctx_lens, seq_len_q, seed=1)
    case.build_references(case_name)
    plan = _schedule(ITERS, seed=2)
    outs = torch.full(
        (ITERS, case.rows, NUM_HEADS * KV_LORA_RANK),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )
    stamps = torch.zeros(ITERS, 8, dtype=torch.int64, device="cuda")
    # Start from a different set than the first iteration's, so iteration 0 also overwrites.
    # Warm-up chain (module loading, launch attributes), then the measured chain.
    _enqueue_chain(case, helpers, plan[:2], outs, stamps, use_pdl)
    case.apply_plain(NUM_SETS - 1, 1)
    outs.fill_(float("nan"))
    stamps.zero_()
    torch.cuda.synchronize()
    torch.cuda._sleep(GATE_CYCLES)
    _enqueue_chain(case, helpers, plan, outs, stamps, use_pdl)
    torch.cuda.synchronize()
    bad = _mismatches(case, plan, outs)
    summary = _stamp_summary(stamps, plan)
    record(
        RESULTS,
        f"{case_name}:eager",
        {
            "use_pdl_helpers": use_pdl,
            "trtllm_enable_pdl_env": os.environ.get("TRTLLM_ENABLE_PDL"),
            "mismatches": len(bad),
            "first_mismatches": bad[:5],
            **summary,
        },
    )
    assert not bad, bad[:5]
    if use_pdl and os.environ.get("TRTLLM_ENABLE_PDL", "1") != "0" and not UNDER_SANITIZER:
        # The chain must actually exercise the early launch, or this test proves nothing.
        assert summary["consumer_started_before_sparse_mla_done"] > 0.25, summary


@pytest.mark.parametrize("case_name", list(CASES))
def test_sparse_mla_delayed_producer_consumer_graph(helpers, case_name):
    ctx_lens, seq_len_q = CASES[case_name]
    case = StressCase(ctx_lens, seq_len_q, seed=3)
    case.build_references(f"{case_name}:graph")
    length = 4 * NUM_SETS  # the version of each set alternates across replays too
    plan = _schedule(length, seed=4)
    outs = torch.full(
        (length, case.rows, NUM_HEADS * KV_LORA_RANK),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )
    stamps = torch.zeros(length, 8, dtype=torch.int64, device="cuda")
    case.apply_plain(NUM_SETS - 1, 1)
    # Warm up outside capture (attribute setup, lazy loading).
    _enqueue_chain(case, helpers, plan[:2], outs, stamps, True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _enqueue_chain(case, helpers, plan, outs, stamps, True)
    total_bad = 0
    summaries = []
    for replay in range(GRAPH_REPLAYS):
        outs.fill_(float("nan"))
        stamps.zero_()
        graph.replay()
        torch.cuda.synchronize()
        bad = _mismatches(case, plan, outs)
        total_bad += len(bad)
        summaries.append(_stamp_summary(stamps, plan))
        assert not bad, (replay, bad[:5])
    record(
        RESULTS,
        f"{case_name}:graph",
        {
            "replays": GRAPH_REPLAYS,
            "chain_length": length,
            "mismatches": total_bad,
            "consumer_started_before_sparse_mla_done": [
                x["consumer_started_before_sparse_mla_done"] for x in summaries
            ],
            "consumer_wait_ns_median": [x["consumer_wait_ns_median"] for x in summaries],
        },
    )


def test_sparse_mla_delayed_producer_consumer_under_hbm_contention(helpers):
    ctx_lens, seq_len_q = CASES["one_seq_r6"]
    case = StressCase(ctx_lens, seq_len_q, seed=5)
    case.build_references("contention")
    plan = _schedule(ITERS, seed=6)
    outs = torch.full(
        (ITERS, case.rows, NUM_HEADS * KV_LORA_RANK),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )
    stamps = torch.zeros(ITERS, 8, dtype=torch.int64, device="cuda")
    big_src = torch.ones(256 * 1024 * 1024, dtype=torch.float32, device="cuda")
    big_dst = torch.empty_like(big_src)
    case.apply_plain(NUM_SETS - 1, 1)
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        for _ in range(160):
            big_dst.copy_(big_src)
    torch.cuda._sleep(GATE_CYCLES // 4)
    _enqueue_chain(case, helpers, plan, outs, stamps, True)
    torch.cuda.synchronize()
    bad = _mismatches(case, plan, outs)
    record(
        RESULTS,
        "contention:eager",
        {"mismatches": len(bad), "first_mismatches": bad[:5], **_stamp_summary(stamps, plan)},
    )
    assert not bad, bad[:5]


@pytest.mark.skipif(UNDER_SANITIZER, reason="CUPTI tracing is not combined with compute-sanitizer")
def test_sparse_mla_early_launch_trace(helpers):
    """CUPTI kernel intervals: sparse_mla_decode becomes resident before its producer finishes."""
    ctx_lens, seq_len_q = CASES["one_seq_r6"]
    case = StressCase(ctx_lens, seq_len_q, seed=7)
    case.build_references("trace")
    iterations = 48
    plan = [(i % NUM_SETS, (i // NUM_SETS) % 2, 20000, 3000) for i in range(iterations)]
    outs = torch.full(
        (iterations, case.rows, NUM_HEADS * KV_LORA_RANK),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )
    stamps = torch.zeros(iterations, 8, dtype=torch.int64, device="cuda")
    case.apply_plain(NUM_SETS - 1, 1)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        torch.cuda._sleep(GATE_CYCLES // 4)
        _enqueue_chain(case, helpers, plan, outs, stamps, True)
        torch.cuda.synchronize()
    assert not _mismatches(case, plan, outs)
    kernels = sorted(
        (
            (e.time_range.start, e.time_range.end, e.name)
            for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA
        ),
        key=lambda x: x[0],
    )
    producers = [k for k in kernels if "delayedScatter" in k[2]]
    sparse_kernels = [k for k in kernels if "sparseMlaDecode" in k[2]]
    consumers = [k for k in kernels if "pdlConsumer" in k[2]]
    count = min(len(producers), len(sparse_kernels), len(consumers))
    sparse_mla_early = sum(1 for i in range(count) if sparse_kernels[i][0] < producers[i][1])
    consumer_early = sum(1 for i in range(count) if consumers[i][0] < sparse_kernels[i][1])
    record(
        RESULTS,
        "early_launch_trace",
        {
            "kernels": {
                "producer": len(producers),
                "sparse_mla_decode": len(sparse_kernels),
                "consumer": len(consumers),
            },
            "sparse_mla_started_before_producer_end": sparse_mla_early / max(count, 1),
            "consumer_started_before_sparse_mla_end": consumer_early / max(count, 1),
            "sparse_mla_start_minus_producer_end_us_median": statistics.median(
                sparse_kernels[i][0] - producers[i][1] for i in range(count)
            )
            if count
            else None,
            **_stamp_summary(stamps, plan),
        },
    )
    assert count == iterations

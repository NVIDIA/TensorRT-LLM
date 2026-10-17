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
"""Real-neighbour benchmark of the routed MoE inside the served MoE block.

Runs one process per GPU of the two-host TP8 group (``srun --mpi=pmix``, 8
tasks). Every call is one target MoE layer of the served graph, built from the
model's own modules with real GLM-5.2 NVFP4 weights:

    router GEMM (main stream) || BF16 shared expert (aux stream)
    -> routed MoE (native TRT-LLM-Gen runner, or routed_moe through the target's MoE)
    -> stream join -> post-MoE hop

The MoE is the built-in ``Deepseekv3MoE`` adopted by the target's ``GlmMoE``; its forward
runs inside a decode step, with multi-stream on (as the CUDA-graph runner captures it) and
``TRTLLM_ENABLE_PDL`` as served. The post-MoE hop is

* ``fin`` (post_moe_hop off): ``shared + routed`` inside ``Deepseekv3MoE.forward``, then
  the native MNNVL ``AllReduce(RESIDUAL_RMS_NORM)`` with the next layer's norm;
* ``raw`` (post_moe_hop on): ``forward(do_finalize=False)`` and the post_moe_hop on the raw
  tuple, exactly as the target's decoder layer calls it in ``forward_MoE``.

Each call's hop is the predecessor of the next call's router GEMM. Calls
rotate over several layers' weights (``--layers``), so a call never finds its
experts in L2. One CUDA graph holds ``--calls`` calls; a trial replays it
``--replays`` times after an MPI barrier; per-call time = trial time /
(replays * calls); the summary is the median trial of the slowest rank.

Inputs: ``--mode real`` uses the captured post-norm hidden states (real
routing through the real router GEMM, consecutive-token windows of R rows,
drawn from every position of the captured prompts);
``--mode sweep`` replaces the router GEMM's result (the GEMM still runs) by
constructed logits that route to exactly D distinct experts; ``--mode mix``
draws every call's D from a served distribution (``--mix``, see ``MIXES``)
and constructs its logits the same way, with fresh experts per call.

Router-logit producer (``--logits``):

* ``gate``: the MoE's own router GEMM runs on the main stream before the
  routed MoE (native, router_gemm/W2 off). The shared expert starts during that GEMM.
* ``supplied``: the router contract with logits present (post_attention_hop/router_gemm on): the
  pre-MoE hop already produced the FP32 logits, the MoE's gate is skipped (it returns the
  call's logits without launching anything), so the routed MoE and the
  shared expert become ready at the same moment. Config labels get a
  ``+sup`` suffix. The logits are the real router GEMM's output for the
  call's hidden states (computed once before capture), or the sweep logits.

Optional served neighbours before every call (identical in every arm):
``--attn-mb N`` streams N MB of weights (a reduction over a rotating buffer,
the stand-in for the attention block's weight reads, which also flushes L2
as the served graph does), then ``--pre-hop ar`` runs the native two-host
``AllReduce(RESIDUAL_RMS_NORM, trigger_completion_at_end=False)`` on a
rank-local partial (the stand-in for the pre-MoE hop that precedes the MoE
block; with ``--logits supplied`` it plays post_attention_hop's hop, which emits the logits).
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(
    0, os.path.join(os.path.dirname(HERE), "..", "..", "shared", "kernels", "post_moe_hop")
)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "..", "..", "shared", "kernels"))
import plan_testlib  # noqa: E402
import post_moe_hop_mpi_utils as U  # noqa: E402
import routed_moe_fixtures as fx  # noqa: E402

ACCUMULATION_ENV = "TRTLLM_STAIRCASE_GLM_LL_ROUTED_MOE_INTO_HOP_ACCUMULATION"
# routed_moe_into_hop routed-input modes (see routed_moe.routed_moe.routed_output_into_hop); the
# routed kernels run at fixed launch parameters, so the routed-input mode is the only arm knob.
ARM_ENV = {
    "routed_moe_into_hop_I": {ACCUMULATION_ENV: "identity"},
    "routed_moe_into_hop_F": {ACCUMULATION_ENV: "fp32"},
    "routed_moe_into_hop_R": {ACCUMULATION_ENV: "row"},
}
ARM_KNOBS = (ACCUMULATION_ENV,)
# Generic arm syntax ``routed_moe[K=V+K=V]`` / ``routed_moe_into_hop[...]`` with these short keys.
ARM_KEYS = {"MODE": ACCUMULATION_ENV}
# Served distinct-expert distributions at R = 6, from a server profile of
# the routed_moe arms (fixture routed_moe-server-profile,
# xc_end_histogram_rank0.csv: the share of two-round XC calls, i.e. D above
# the cluster count, at 24 / 28 / 32 / 36 clusters). Bins (lo, hi, share);
# inside a bin D is uniform, except the open top bin, which decays by 0.7
# per expert from 37 (the real prefill windows reach p90 38, max 47).
MIXES = {
    # late in a decode segment: the distribution of D over its bins
    "late": ((17, 24, 0.058), (25, 28, 0.077), (29, 32, 0.213), (33, 36, 0.195), (37, 48, 0.457)),
    # early in a segment: fewer large D; the split below 33 follows the late
    # window's proportions
    "early": ((17, 24, 0.084), (25, 28, 0.111), (29, 32, 0.308), (33, 36, 0.380), (37, 48, 0.117)),
}
ARMS_REPORTED = tuple(["routed_moe", "routed_moe_into_hop"] + list(ARM_ENV))
NUM_LAYERS = 78


class _PassAllReduce(torch.nn.Module):
    """Pass-through stand-in for the MoE module's own AllReduce.

    The decoder layer disables that all-reduce (``enable_allreduce=False``
    under POST_MOE_FUSION), so it only ever passes its input through.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    @property
    def output_buffer_kind(self) -> int:
        from tensorrt_llm.bindings.internal.thop import BufferKind

        return int(BufferKind.DEFAULT)

    def forward(self, input, *, all_reduce_params=None):
        assert all_reduce_params is not None and not all_reduce_params.enable_allreduce
        return input


def arm_env(impl: str) -> dict:
    """Environment of an arm: ``ARM_ENV[impl]`` or the generic ``base[K=V+...]``."""
    if "[" not in impl:
        return dict(ARM_ENV.get(impl, {}))
    base, _, spec = impl.partition("[")
    assert base in ("routed_moe", "routed_moe_into_hop") and spec.endswith("]"), impl
    env = {}
    for item in [v for v in spec[:-1].split("+") if v]:
        key, _, val = item.partition("=")
        env[ARM_KEYS[key]] = val
    return env


def mix_distinct(name: str, calls: int, rows: int, seed: int):
    """Per-call distinct-expert counts drawn from ``MIXES[name]``."""
    values, weights = [], []
    for lo, hi, share in MIXES[name]:
        if hi == fx.TOP_K * rows and lo > 32:
            span = [0.7 ** (d - lo) for d in range(lo, hi + 1)]
        else:
            span = [1.0] * (hi - lo + 1)
        total = sum(span)
        for d, w in zip(range(lo, hi + 1), span):
            values.append(d)
            weights.append(share * w / total)
    g = torch.Generator().manual_seed(seed)
    idx = torch.multinomial(torch.tensor(weights), calls, replacement=True, generator=g)
    return [values[i] for i in idx.tolist()]


def mix_logits(moes, order, rows: int, distinct_per_call, seed: int):
    """Constructed logits routing call ``i`` to ``distinct_per_call[i]`` experts,
    drawn fresh per call (the attention stand-in evicts L2 between calls)."""
    out = []
    for i, (layer, d) in enumerate(zip(order, distinct_per_call)):
        sets = fx.expert_sets_with_distinct(rows, d, seed=seed + i)
        perm = torch.randperm(
            fx.NUM_EXPERTS, generator=torch.Generator().manual_seed(seed + 7919 * i)
        ).tolist()
        uniq = sorted(set(e for row in sets for e in row))
        remap = {e: perm[j] for j, e in enumerate(uniq)}
        sets = [[remap[e] for e in row] for row in sets]
        bias = moes[layer].gate.e_score_correction_bias.to(torch.bfloat16)
        lg = fx.logits_for_expert_sets(sets, bias, seed=seed + i)
        ids, _ = fx.routing_reference(lg, bias)
        assert len(set(ids.flatten().tolist())) == d
        out.append(lg)
    return out


def distinct_stats(moes, order, logits_per_call):
    """Distinct experts per call (native routing of the logits the MoE reads)."""
    ds = []
    for layer, lg in zip(order, logits_per_call):
        bias = moes[layer].gate.e_score_correction_bias.to(torch.bfloat16)
        ids, _ = fx.routing_reference(lg.float().cpu(), bias.cpu())
        ds.append(len(set(ids.flatten().tolist())))
    n = len(ds)
    above = {c: round(100.0 * sum(d > c for d in ds) / n, 1) for c in (24, 26, 28, 32, 36)}
    return {"mean": round(sum(ds) / n, 2), "median": sorted(ds)[n // 2], "pct_above": above}


def set_features(ids: str) -> None:
    """Install, for this process, the plan that runs exactly the fast paths ``ids`` (comma-separated)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

    plan_testlib.select(dispatch, ids)


def next_norm_weight(layer: int) -> torch.Tensor:
    name = (
        "model.norm.weight"
        if layer + 1 >= NUM_LAYERS
        else f"model.layers.{layer + 1}.input_layernorm.weight"
    )
    return fx._Reader().full(name).to(torch.bfloat16).cuda()


class Engagement:
    """Counts routed_moe adapter calls that engaged (returned a result)."""

    def __init__(self):
        from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

        dispatch.get_impl("routed_moe", "routed_output")  # import + register
        self.counts = {}
        for name in ("routed_output", "routed_output_into_hop"):
            inner = dispatch._REGISTRY["routed_moe"].get(name)
            if inner is None:
                continue

            def counted(*args, _inner=inner, _name=name, **kwargs):
                out = _inner(*args, **kwargs)
                key = (_name, out is not None)
                self.counts[key] = self.counts.get(key, 0) + 1
                return out

            dispatch.register("routed_moe", name, counted)

    def engaged(self, name: str) -> int:
        return self.counts.get((name, True), 0)


class Neighbours:
    """Served kernels that precede each MoE block (identical in every arm).

    ``attn_mb``: a weight-streaming stand-in for the attention block (one
    reduction over ``attn_mb`` MB, two buffers used alternately so the reads
    miss L2 as the served attention weights do). ``pre_hop``: the native
    two-host fused all-reduce + residual + RMSNorm on a rank-local partial.
    """

    def __init__(self, env, rows: int, calls: int, attn_mb: int, pre_hop: str):
        g = torch.Generator(device="cuda").manual_seed(500 + env.rank)
        self.attn = []
        if attn_mb > 0:
            numel = attn_mb * (1 << 20) // 2
            self.attn = [
                torch.randn(numel, device="cuda", generator=g).bfloat16() for _ in range(2)
            ]
        self.pre = None
        if pre_hop == "ar":
            common = torch.Generator(device="cuda").manual_seed(600)
            self.pre = [
                (
                    (torch.randn(rows, fx.HIDDEN, device="cuda", generator=g) * 0.05).bfloat16(),
                    (torch.randn(rows, fx.HIDDEN, device="cuda", generator=common)).bfloat16(),
                )
                for _ in range(calls)
            ]
            self.gamma = (torch.rand(fx.HIDDEN, device="cuda", generator=common) + 0.5).bfloat16()
        self.sink = []

    def step(self, i: int):
        if self.attn:
            self.sink.append(self.attn[i % 2].amax())
        if self.pre is not None:
            partial, residual = self.pre[i]
            self.sink.append(U.native_fused_allreduce(partial, residual, self.gamma, fx.RMS_EPS))


class Chain:
    """The per-rank call sequence of one arm, captured in one CUDA graph."""

    def __init__(
        self,
        env,
        moes,
        gammas,
        order,
        xs,
        residuals,
        logits,
        hop,
        mode,
        impl,
        supplied=None,
        neighbours=None,
    ):
        self.env, self.moes, self.gammas = env, moes, gammas
        self.order, self.xs, self.residuals, self.logits = order, xs, residuals, logits
        self.hop, self.mode, self.impl = hop, mode, impl
        self.supplied, self.neighbours = supplied, neighbours
        self.outs = []

    def run(self):
        from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

        with dispatch.step(dispatch.Phase.DECODE):
            self._run()

    def _run(self):
        from tensorrt_llm._torch.distributed import AllReduceParams

        self.outs = []
        if self.neighbours is not None:
            self.neighbours.sink = []
        for i, layer in enumerate(self.order):
            moe = self.moes[layer]
            moe._routed_moe_bench_logits = None if self.logits is None else self.logits[i]
            moe._routed_moe_bench_supplied = None if self.supplied is None else self.supplied[i]
            if self.neighbours is not None:
                self.neighbours.step(i)
            x, res, gamma = self.xs[i], self.residuals[i], self.gammas[layer]
            if self.mode == "fin":
                h = moe.forward(
                    x,
                    do_finalize=True,
                    final_all_reduce_params=AllReduceParams(enable_allreduce=False),
                )
                out = U.native_fused_allreduce(h, res, gamma, fx.RMS_EPS)
            else:
                res_moe = moe.forward(x, do_finalize=False)
                if len(res_moe) == 2:
                    # routed_moe_into_hop: [shared, routed] with routed finalized into the hop's
                    # buffer
                    shared, routed = res_moe
                    out = self.hop.hop_routed(routed, shared, res, gamma, fx.RMS_EPS)
                else:
                    shared, fc2, scale, perm = res_moe
                    out = self.hop(fc2, scale, perm, shared, res, gamma, fx.RMS_EPS)
            self.outs.append(out)


def build_layers(env, layers):
    import tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe  # noqa: E501, F401  (registers routed_moe)

    saved = fx._NoAllReduce
    fx._NoAllReduce = _PassAllReduce
    moes, gammas = {}, {}
    try:
        for layer in layers:
            t0 = time.time()
            full = fx.load_layer_full(layer)
            moes[layer] = fx.build_deepseek_moe(layer, env.rank, full)
            del full
            gammas[layer] = next_norm_weight(layer)
            if env.rank == 0:
                print(f"[chain] layer {layer} built in {time.time() - t0:.1f}s", flush=True)
    finally:
        fx._NoAllReduce = saved
    for moe in moes.values():
        gate = moe.gate
        orig = gate.forward

        def forward(h, moe=moe, orig=orig):
            if moe._routed_moe_bench_supplied is not None:
                # the router contract with logits present: the gate is skipped
                return moe._routed_moe_bench_supplied
            out = orig(h)
            if moe._routed_moe_bench_logits is not None:
                # sweep: the router GEMM still runs as served; the routed MoE
                # reads the constructed logits instead (no extra kernel)
                out = moe._routed_moe_bench_logits
            return out

        gate.forward = forward
        moe._routed_moe_bench_logits = None
        moe._routed_moe_bench_supplied = None
        moe._routed_moe_bench_router = orig
    return moes, gammas


def supplied_logits(moes, order, xs, logits):
    """Per call, the FP32 logits a pre-MoE hop would hand over (static buffers)."""
    out = []
    with torch.no_grad():
        for i, layer in enumerate(order):
            if logits is not None:
                lg = logits[i].float().contiguous().clone()
            else:
                lg = moes[layer]._routed_moe_bench_router(xs[i]).float().contiguous().clone()
            out.append(lg)
    torch.cuda.synchronize()
    return out


def real_inputs(layers, rows: int, calls: int, seed: int):
    """Per call: (layer, x) from consecutive-token windows of the capture."""
    per_layer = {}
    for layer in layers:
        win = [x for x, _ in fx.real_windows(fx.load_real_capture(layer), rows)]
        assert win and win[0] is not None, f"no hidden states captured for layer {layer}"
        g = torch.Generator().manual_seed(seed + layer)
        per_layer[layer] = [win[i] for i in torch.randperm(len(win), generator=g).tolist()]
    order, xs = [], []
    for i in range(calls):
        layer = layers[i % len(layers)]
        order.append(layer)
        xs.append(per_layer[layer][(i // len(layers)) % len(per_layer[layer])].cuda())
    return order, xs


def sweep_logits(moes, order, rows: int, distinct: int, seed: int):
    """Constructed logits routing each call to ``distinct`` experts.

    Consecutive uses of one layer draw from disjoint expert pools.
    """
    pools = max(1, fx.NUM_EXPERTS // distinct)
    perm = torch.randperm(fx.NUM_EXPERTS, generator=torch.Generator().manual_seed(seed)).tolist()
    out = []
    for i, layer in enumerate(order):
        sets = fx.expert_sets_with_distinct(rows, distinct, seed=seed + i)
        p = (i // len(set(order))) % pools
        pool = perm[p * distinct : (p + 1) * distinct]
        uniq = sorted(set(e for row in sets for e in row))
        remap = {e: pool[j] for j, e in enumerate(uniq)}
        sets = [[remap[e] for e in row] for row in sets]
        bias = moes[layer].gate.e_score_correction_bias.to(torch.bfloat16)
        lg = fx.logits_for_expert_sets(sets, bias, seed=seed + i)
        ids, _ = fx.routing_reference(lg, bias)
        assert len(set(ids.flatten().tolist())) == distinct
        out.append(lg)
    return out


def capture(env, chain):
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            chain.run()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    env.comm.barrier()
    graph = torch.cuda.CUDAGraph()
    with with_multi_stream(True):
        with torch.cuda.graph(graph, stream=stream):
            chain.run()
    torch.cuda.synchronize()
    for _ in range(2):
        env.comm.barrier()
        graph.replay()
        torch.cuda.synchronize()
    return graph


def time_graph(env, graph, replays: int):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    env.comm.barrier()
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0


def compare(env, a_outs, b_outs, label):
    """Max |a - b| of the final hidden and residual over every call (sanity)."""
    worst = [0.0, 0.0]
    for a, b in zip(a_outs, b_outs):
        for j in range(2):
            worst[j] = max(worst[j], (a[j].float() - b[j].float()).abs().max().item())
    allw = env.comm.allgather(worst)
    if env.rank == 0:
        print(
            f"[chain] check {label}: max|d norm_out| {max(w[0] for w in allw):.3e} "
            f"max|d residual| {max(w[1] for w in allw):.3e}",
            flush=True,
        )


def run_config(
    env, args, moes, gammas, hop, eng, label, order, xs, logits, supplied=None, neighbours=None
):
    torch.manual_seed(0)
    g = torch.Generator(device="cuda").manual_seed(11)
    residuals = [
        (torch.randn(args.rows, fx.HIDDEN, device="cuda", generator=g) * 4.0).bfloat16()
        for _ in order
    ]
    arms = [a for a in args.arms.split(",") if a]
    graphs, chains = {}, {}
    # Every arm is captured ``--captures`` times, in forward and then reverse
    # arm order: an occasional graph replays with a per-call inter-kernel gap
    # that is a property of that one instantiation (identical arms captured
    # elsewhere do not show it), so an arm reports its fastest capture.
    passes = [arms if p % 2 == 0 else arms[::-1] for p in range(args.captures)]
    for p, seq in enumerate(passes):
        for arm in seq:
            impl, mode = arm.split(":")
            # "routed_moe" is the routed_moe schedule, "routed_moe_into_hop" the same schedule
            # writing into the post_moe_hop (raw mode)
            feats = (
                ""
                if impl == "native"
                else (
                    "post_moe_hop,routed_moe,routed_moe_into_hop"
                    if impl.startswith("routed_moe_into_hop")
                    else "routed_moe"
                )
            )
            if supplied is not None:
                # the composition's logit producer (post_attention_hop's hop) is selected
                feats = ",".join(f for f in (feats, "post_attention_hop") if f)
            set_features(feats)
            for knob in ARM_KNOBS:
                os.environ.pop(knob, None)
            os.environ.update(arm_env(impl))
            if impl.startswith("routed_moe_into_hop"):
                # collective, outside capture: the hop's routed-input mode of this arm
                assert hop.enable_routed_input(os.environ.get(ACCUMULATION_ENV) or None)
            entry = (
                "routed_output_into_hop"
                if impl.startswith("routed_moe_into_hop")
                else "routed_output"
            )
            e0 = eng.engaged(entry)
            chain = Chain(
                env,
                moes,
                gammas,
                order,
                xs,
                residuals,
                None if supplied is not None else logits,
                hop,
                mode,
                impl,
                supplied=supplied,
                neighbours=neighbours,
            )
            graphs[(arm, p)] = capture(env, chain)
            if p == 0:
                chains[arm] = chain
            engaged = eng.engaged(entry) - e0
            want = 0 if impl == "native" else 3 * len(order)  # two warmups + capture
            ok = U.all_ranks(engaged == want)
            if env.rank == 0:
                print(
                    f"[chain] {label} arm {arm} capture {p}: routed_moe engaged {engaged}/{want} "
                    f"all_ranks={ok}"
                )
            assert ok, f"{arm}: unexpected routed_moe engagement"
    set_features("")
    for knob in ARM_KNOBS:
        os.environ.pop(knob, None)
    per = {key: [] for key in graphs}
    for _ in range(args.trials):
        for key in graphs:
            per[key].append(time_graph(env, graphs[key], args.replays))
    if args.profile_config == label:
        # one replay per arm inside a cudaProfilerApi capture range, each in
        # an NVTX range named after the arm (nsys --capture-range=cudaProfilerApi)
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        for arm, p in graphs:
            env.comm.barrier()
            torch.cuda.nvtx.range_push(f"{label}:{arm}" + (f"#{p}" if p else ""))
            graphs[(arm, p)].replay()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
        torch.cuda.profiler.stop()
    rows = []
    lg_seen = supplied if supplied is not None else logits
    dstat = None
    if lg_seen is not None:
        dstat = distinct_stats(moes, order, lg_seen)
    elif env.rank == 0:
        dstat = distinct_stats(
            moes,
            order,
            [moes[layer]._routed_moe_bench_router(x).float() for layer, x in zip(order, xs)],
        )
    if env.rank == 0:
        print(f"CHAIN_DISTINCT {label} {json.dumps(dstat)}", flush=True)
    for arm in arms:
        caps = []
        for p in range(args.captures):
            per_call = [t / (args.replays * len(order)) for t in per[(arm, p)]]
            meds = env.comm.allgather(statistics.median(per_call))
            caps.append(
                {
                    "median": max(meds),
                    "spread": [min(meds), max(meds)],
                    "best": max(env.comm.allgather(min(per_call))),
                    "mean": max(env.comm.allgather(statistics.mean(per_call))),
                }
            )
        best = min(caps, key=lambda c: c["median"])
        rows.append(
            {
                "config": label,
                "arm": arm,
                "per_call_us_median_slowest_rank": best["median"],
                "per_call_us_median_rank_spread": best["spread"],
                "per_call_us_best_trial_slowest_rank": best["best"],
                "per_call_us_mean_slowest_rank": best["mean"],
                "per_capture_median_slowest_rank": [c["median"] for c in caps],
                "distinct": dstat,
                "calls_per_graph": len(order),
                "replays": args.replays,
                "trials": args.trials,
            }
        )
        if env.rank == 0:
            print("CHAIN_BENCH " + json.dumps(rows[-1]), flush=True)
            if len(caps) > 1:
                m = [c["median"] for c in caps]
                flag = "  <-- capture spread > 1 us" if max(m) - min(m) > 1.0 else ""
                print(
                    f"CHAIN_CAPTURES {label} {arm}: " + " ".join(f"{v:.2f}" for v in m) + flag,
                    flush=True,
                )
    by = {r["arm"]: r["per_call_us_median_slowest_rank"] for r in rows}
    by_mean = {r["arm"]: r["per_call_us_mean_slowest_rank"] for r in rows}
    impls = [a.split(":")[0] for a in arms if not a.startswith("native:")]
    impls = list(dict.fromkeys(impls))
    if env.rank == 0:
        for mode in ("fin", "raw"):
            for impl in impls:
                if f"native:{mode}" in by and f"{impl}:{mode}" in by:
                    d = by[f"native:{mode}"] - by[f"{impl}:{mode}"]
                    dm = by_mean[f"native:{mode}"] - by_mean[f"{impl}:{mode}"]
                    print(
                        f"CHAIN_DELTA {label} {mode}: native - {impl} = {d:+.2f} us/call "
                        f"(trial mean {dm:+.2f})",
                        flush=True,
                    )
        for impl in ("routed_moe", "routed_moe_into_hop"):
            if "native:fin" in by and f"{impl}:raw" in by:
                d = by["native:fin"] - by[f"{impl}:raw"]
                print(f"CHAIN_DELTA {label} native:fin - {impl}:raw = {d:+.2f} us/call", flush=True)
        if "routed_moe:raw" in by and "routed_moe_into_hop:raw" in by:
            d = by["routed_moe:raw"] - by["routed_moe_into_hop:raw"]
            print(
                f"CHAIN_DELTA {label} routed_moe:raw - routed_moe_into_hop:raw = {d:+.2f} us/call (routed input)",
                flush=True,
            )
    for mode in ("fin", "raw"):
        for impl in impls:
            if f"native:{mode}" in chains and f"{impl}:{mode}" in chains:
                compare(
                    env,
                    chains[f"native:{mode}"].outs,
                    chains[f"{impl}:{mode}"].outs,
                    f"{label} {mode} {impl}",
                )
    del graphs
    torch.cuda.synchronize()
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="10,30,50,77")
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--calls", type=int, default=48)
    ap.add_argument("--replays", type=int, default=10)
    ap.add_argument("--trials", type=int, default=9)
    ap.add_argument("--captures", type=int, default=2, help="graphs per arm (best one reported)")
    ap.add_argument("--arms", default="native:fin,routed_moe:fin,native:raw,routed_moe:raw")
    ap.add_argument("--mode", default="real", help="real | sweep | mix (comma list)")
    ap.add_argument("--mix", default="early,late", help="served D distributions (MIXES)")
    ap.add_argument("--distinct", default="8,14,21,25,31,38,48")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--logits", default="gate", help="gate | supplied | gate,supplied (router-logit producer)"
    )
    ap.add_argument("--attn-mb", type=int, default=0, help="attention weight-stream stand-in")
    ap.add_argument("--pre-hop", default="none", choices=("none", "ar"))
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--profile-config", default="", help="config label replayed once per arm under nsys"
    )
    args = ap.parse_args(argv)

    env = U.mpi_env()
    assert env.world == 8 and env.multi_node, "needs the two-host TP8 group"
    os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
    set_features("")
    layers = [int(v) for v in args.layers.split(",")]
    moes, gammas = build_layers(env, layers)
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHop,
    )

    # as the target's decoder layer builds it: the layer's MoE module declares the
    # raw-tuple contract the hop consumes
    hop = PostMoeHop(
        env.mapping,
        fx.HIDDEN,
        fx.TOP_K,
        torch.bfloat16,
        raw_producer=next(iter(moes.values())).experts,
    )
    assert hop.engaged, hop.reason
    if any(a.startswith("routed_moe_into_hop") for a in args.arms.split(",")):
        assert hop.enable_routed_input(), (
            "routed_moe_into_hop routed input not enabled on every rank"
        )
        for moe in moes.values():
            moe._routed_into_hop = hop
    eng = Engagement()
    if env.rank == 0:
        print(
            f"[chain] ranks={env.world} layers={layers} rows={args.rows} calls={args.calls} "
            f"pdl={os.environ.get('TRTLLM_ENABLE_PDL')} post_moe_hop={hop.impl}",
            flush=True,
        )
    results = []
    modes = args.mode.split(",")
    producers = [p for p in args.logits.split(",") if p]
    assert set(producers) <= {"gate", "supplied"}, args.logits
    order, xs = real_inputs(layers, args.rows, args.calls, args.seed)
    neighbours = None
    if args.attn_mb > 0 or args.pre_hop != "none":
        neighbours = Neighbours(env, args.rows, len(order), args.attn_mb, args.pre_hop)
    if env.rank == 0:
        print(
            f"[chain] logits={producers} attn_mb={args.attn_mb} pre_hop={args.pre_hop}",
            flush=True,
        )

    def configs(label, lg):
        out = []
        for producer in producers:
            sup = supplied_logits(moes, order, xs, lg) if producer == "supplied" else None
            name = label + ("+sup" if producer == "supplied" else "")
            out += run_config(
                env, args, moes, gammas, hop, eng, name, order, xs, lg, sup, neighbours
            )
        return out

    if "real" in modes:
        results += configs("real", None)
    if "mix" in modes:
        for name in [v for v in args.mix.split(",") if v]:
            ds = mix_distinct(name, len(order), args.rows, args.seed + 5000)
            lg = mix_logits(moes, order, args.rows, ds, args.seed + 6000)
            results += configs(f"mix{name}", lg)
    if "sweep" in modes:
        for d in [int(v) for v in args.distinct.split(",") if v]:
            if d < fx.TOP_K or d > fx.TOP_K * args.rows:
                continue
            lg = sweep_logits(moes, order, args.rows, d, args.seed + 1000 * d)
            results += configs(f"D{d}", lg)
    if env.rank == 0 and args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
    hop.raise_if_error()


if __name__ == "__main__":
    main()

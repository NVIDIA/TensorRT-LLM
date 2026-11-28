# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 MegaMoE layers on four GPUs (attention DP, EP4, 384 experts) with and without an adaptive load
balancer (512 slots, home route tables, no runtime), at decode (8 to 448 rows per rank: on Blackwell the one-round
router up to 256 and the metadata-push router above; on SM107 one row count in each generation-phase bucket 32 to
512, the 1024 bucket by the first prefill size) and prefill (1024 / 4096 / 8192, the metadata-push router) sizes.

Bit for bit where both layers add the top-k picks in one fixed order (_same): an adaptive layer whose dispatch
router maps and counts its routes returns what the same layer returns on the staged route, and what a layer without
the balancer returns; its statistics are the routing's picks of the live rows, nothing while the recording flag is
clear; the picks a malformed table drops add nothing and are not waited for. A graph captured on home tables
replays an install as eager routes it, and the server's mixed forward captures after its warmup and replays as
eager. Under ``balance`` off the 512-slot layers return what the layers without a balancer return and launch the
same kernels.
"""

import functools
import pickle
import sys
from typing import Optional

import cloudpickle
import pytest
import torch
from _torch.moe.quantize_utils import NVFP4QuantizeUtil
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4MoE
from tensorrt_llm._torch.utils import AuxStreamType
from tensorrt_llm._utils import get_sm_version, mpi_allgather, mpi_rank, mpi_world_size
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

_WORLD_SIZE = 4
_NUM_EXPERTS = 384
_NUM_SLOTS = 512
_TOP_K = 6
_HIDDEN_SIZE = 7168
_INTERMEDIATE_SIZE = 3072
_CLAMP = 10.0
_SCALING = 2.5
# One row count per SM107 generation-phase bucket (32, 64, 128, 192, 256, 384, 512; 1024 is the first prefill size).
_DECODE_ROWS = (8, 64, 128, 160, 256, 320, 448)
_PREFILL_ROWS = (1024, 4096, 8192)
_MAX_NUM_TOKENS = 8192
_LAYERS = 8  # distinct layers per arm

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)


def _init_worker(paths: list[str]) -> None:
    sys.path[:] = paths


def _assert_on_all_ranks(problem: Optional[str]) -> None:
    problems = mpi_allgather(problem)
    assert not any(problems), problems


# Unit roundoff of bf16 (8 significand bits): a rounding moves a value by at most 2^-8 of it.
_BF16_UNIT_ROUNDOFF = 2.0**-8


def _fixed_order(backend, rows: int, other=None) -> bool:
    """Whether two runs of ``rows`` rows, on ``backend`` and on ``other`` (default the same layer), add
    each row's top-k terms in one fixed order: neither sums them in its FC2 epilogue in arrival order
    (the SM107 32-row bucket), and both run one kernel tactic (on SM107 the layer without a balancer,
    96 experts per GPU, takes combine dedup at 8192 rows where the 128-slot layer does not)."""
    if backend._use_in_kernel_fc2_reduce(backend._select_launch_max_tokens(rows)):
        return False
    return (
        other is None
        or other is backend
        or backend._decode_tactic(rows) == other._decode_tactic(rows)
    )


def _pick_terms(backends, run, positions=range(_TOP_K)) -> list:
    """The terms the top-k reduce adds, as the kernels compute them: ``run()``'s outputs once per pick
    position j, with every other pick of every row set to -1, which the route drops (position None
    keeps no pick: what a fused shared expert adds alone). A row's FC2 output depends on no other row
    or pick, so the j-th output is the j-th pick's term of each row. The fused route also wrote the
    picks into the launch bucket, which the launch then reads as they are: both are masked."""
    terms = []
    for j in positions:
        fused = [backend.try_fused_route_quant for backend in backends]

        def keep_one(*args, _fused, _j=j, **kwargs):
            backend = _fused.__self__
            result = _fused(*args, **kwargs)
            assert result is not None, "splitting the reduce into terms needs the fused route"
            ids = result.token_selected_experts
            picks = [ids]
            staged = backend._prestaged
            if staged is not None and staged[1] is ids:
                picks.append(staged[0][: ids.shape[0]])
            for tensor in picks:
                kept = None if _j is None else tensor[:, _j].clone()
                tensor.fill_(-1)
                if kept is not None:
                    tensor[:, _j] = kept
            return result

        for backend, original in zip(backends, fused):
            backend.try_fused_route_quant = functools.partial(keep_one, _fused=original)
        try:
            terms.append(run())
        finally:
            for backend, original in zip(backends, fused):
                backend.try_fused_route_quant = original
    return terms


def _order_bound(terms, layer: int) -> torch.Tensor:
    """Elementwise bound on the difference of two bf16 sums of one row's k = len(terms) terms (the
    top-k picks', and a fused shared expert's when the layer has one) taken in different orders.
    Each sum rounds at most k times: k - 1 additions and its conversion to the output, or, for a
    layer that adds the terms in fp32, the terms' own roundings. Each rounding moves the partial
    result by at most u times its magnitude, and every partial result is at most sum |t_j|, so each
    sum lies within gamma_k * sum |t_j| of the exact sum, gamma_k = k u / (1 - k u) (Higham,
    Accuracy and Stability of Numerical Algorithms, Lemma 3.1 and Section 4.2), and the two within
    twice that. The terms come back from the kernel rounded to bf16 (|t| <= |t_hat| / (1 - u))."""
    u, k = _BF16_UNIT_ROUNDOFF, len(terms)
    gamma = k * u / (1 - k * u)
    magnitude = torch.stack([t[layer].float().abs() for t in terms]).sum(0)
    return 2 * gamma / (1 - u) * magnitude


def _same(got, want, terms=None) -> bool:
    """Bit for bit; with ``terms`` (_pick_terms of the run that gave ``want``) within _order_bound."""
    if terms is None:
        return all(torch.equal(a, b) for a, b in zip(got, want))
    return all(
        bool(((a.float() - b.float()).abs() <= _order_bound(terms, i)).all())
        for i, (a, b) in enumerate(zip(got, want))
    )


def _check_terms(terms, want) -> Optional[str]:
    """The split is right and the bound discriminates: the terms add up to ``want`` within the bound,
    and ``want`` without its first pick (what a pick summed into the wrong slot loses) breaks it."""
    for i, out in enumerate(want):
        bound = _order_bound(terms, i)
        total = torch.stack([t[i].float() for t in terms]).sum(0)
        if not bool(((total - out.float()).abs() <= bound).all()):
            return "the pick terms do not add up to the output"
        if not bool((terms[0][i].float().abs() > bound).any()):
            return "dropping a pick stays within the order bound"
    return None


def _replica_row(packed: list) -> torch.Tensor:
    """A route table row with replicas, as an install publishes one: the 128 spare slots each take a copy of one
    of 32 hot experts, with shares that include 0 and 1 copies (policy.build_route_tables)."""
    from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.policy import (
        ExpertGeometry,
        build_route_tables,
    )

    geometry = ExpertGeometry(_NUM_EXPERTS, _NUM_SLOTS, _WORLD_SIZE)
    layout = geometry.home_layout().tolist()
    free = [slot for slot, expert in enumerate(layout) if expert < 0]
    for k, slot in enumerate(free):
        layout[slot] = (k * 12) % _NUM_EXPERTS
    shares = [1.0] * _NUM_SLOTS
    for k, slot in enumerate(free):
        shares[slot] = (0.0, 1.0, 0.5, 2.0)[k % 4]
    return torch.tensor(
        build_route_tables(layout, shares, _NUM_EXPERTS, _NUM_SLOTS).packed(), dtype=torch.int32
    )


def _run() -> None:
    from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.transport import (
        find_backends,
        prepare_layers,
    )
    from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import maybe_create_moe_load_balancer
    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.kernel_src.blackwell.inference.mega import (
        block_scaled_swap_ab_mega_moe_kernel as kernel_module,
    )
    from tensorrt_llm.llmapi.llm_args import MoeAdaptiveLoadBalancerConfig

    # The one-round router engages from EP16; lower its gate so EP4 takes it at decode sizes, as EP32 does.
    kernel_module._ONE_ROUND_MIN_WORLD_SIZE = 1
    rank, world_size = mpi_rank(), mpi_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=world_size,
        tp_size=world_size,
        moe_ep_size=world_size,
        moe_tp_size=1,
        enable_attention_dp=True,
    )
    pretrained = PretrainedConfig(
        architectures=["DeepseekV41ForCausalLM"],
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=_INTERMEDIATE_SIZE,
        num_experts=_NUM_EXPERTS,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=_SCALING,
        n_hash_layers=0,
        n_shared_experts=1,
        vocab_size=129280,
        swiglu_limit=_CLAMP,
        torch_dtype=torch.bfloat16,
    )
    routed_quant = QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16)

    def model_config(adaptive: bool) -> ModelConfig:
        config = ModelConfig(
            pretrained_config=pretrained,
            mapping=mapping,
            quant_config=routed_quant,
            moe_backend="MEGAMOE_CUTEDSL",
            max_num_tokens=_MAX_NUM_TOKENS,
            moe_max_num_tokens=_MAX_NUM_TOKENS,
        )
        if adaptive:
            config.moe_load_balancer = MoeAdaptiveLoadBalancerConfig(num_slots=_NUM_SLOTS)
        return config

    with device, torch.inference_mode():
        torch.manual_seed(1701)
        quantizer = NVFP4QuantizeUtil(
            num_experts=_NUM_EXPERTS,
            dtype=torch.bfloat16,
            intermediate_size=_INTERMEDIATE_SIZE,
            hidden_size=_HIDDEN_SIZE,
            quant_config=routed_quant,
        )
        weights = quantizer.create_weights(x_sf_global=torch.tensor(448.0 * 6.0))
        gate_weight = torch.randn(_NUM_EXPERTS, _HIDDEN_SIZE) / _HIDDEN_SIZE**0.5
        bias = torch.randn(_NUM_EXPERTS, dtype=torch.float32) * 0.05

        def build(config: ModelConfig, layer_idx: int) -> DeepseekV4MoE:
            moe = DeepseekV4MoE(
                num_experts=_NUM_EXPERTS,
                top_k=_TOP_K,
                hidden_size=_HIDDEN_SIZE,
                intermediate_size=_INTERMEDIATE_SIZE,
                shared_expert_intermediate_size=_INTERMEDIATE_SIZE,
                aux_stream_dict={
                    AuxStreamType.MoeShared: torch.cuda.Stream(),
                    AuxStreamType.MoeChunkingOverlap: torch.cuda.Stream(),
                },
                dtype=torch.bfloat16,
                model_config=config,
                override_quant_config=routed_quant,
                layer_idx=layer_idx,
            )
            moe.experts.load_weights([weights])
            moe.experts.post_load_weights()
            moe.gate.load_weights([{"weight": gate_weight, "e_score_correction_bias": bias}])
            # The routed experts are what differs; the shared expert stays out of every arm.
            moe.shared_experts = None
            return moe

        plain = torch.nn.ModuleList(build(model_config(False), i) for i in range(_LAYERS))
        adaptive_config = model_config(True)
        with maybe_create_moe_load_balancer(adaptive_config, mapping) as balancer:
            adaptive = torch.nn.ModuleList(build(adaptive_config, i) for i in range(_LAYERS))
        balancer.register_weight_slots_after_to_cuda()
        balancer.finalize_model()
        prepare_layers(find_backends(adaptive, balancer.layers), balancer.layers, device)
        backends = [layer.experts.backend for layer in adaptive]
        plain_backend = plain[0].experts.backend
        plain_backends = [layer.experts.backend for layer in plain[:2]]
        router_route = type(backends[0]).routes_adaptive_in_router

        def use_router(on: bool) -> None:
            for backend in backends:
                backend.routes_adaptive_in_router = (
                    router_route.__get__(backend) if on else (lambda num_tokens: False)
                )

        def record(on: bool) -> None:
            balancer.observe_scheduled_batch(
                1 if on else 0, True, False, lambda start, stop: (0, 0, 0)
            )

        def forward(layers, x, counts):
            return [layer(x, all_rank_num_tokens=counts).clone() for layer in layers[:2]]

        # Bit for bit, decode and prefill sizes, with the statistics of the router's route.
        for rows in (*_DECODE_ROWS, *_PREFILL_ROWS):
            counts = [rows] * world_size
            torch.manual_seed(2000 + rows + rank)
            x = torch.randn((rows, _HIDDEN_SIZE), dtype=torch.bfloat16)
            want = forward(plain, x, counts)
            use_router(False)
            staged = forward(adaptive, x, counts)
            stats = balancer.statistics(device)
            stats["storage"].zero_()
            use_router(True)
            record(True)
            # The picks the first layer's fused route returns, as the router reads them from the bucket.
            fused, picked = backends[0].try_fused_route_quant, []

            def capture(*args, _fused=fused, **kwargs):
                result = _fused(*args, **kwargs)
                if result is not None:
                    picked.append(result.token_selected_experts.clone())
                return result

            backends[0].try_fused_route_quant = capture
            routed = forward(adaptive, x, counts)
            backends[0].try_fused_route_quant = fused
            record(False)
            torch.cuda.synchronize()
            if not picked:
                _assert_on_all_ranks(
                    f"{rows} rows: the adaptive layer did not take the fused route"
                )
            picks = torch.bincount(picked[0].reshape(-1).long(), minlength=_NUM_EXPERTS).float()
            problem = None
            # Outside one fixed order, and at the smallest bucket on every arch so the split and the bound
            # are checked where the comparison is exact too: the terms of the router's route.
            terms = None
            if rows == _DECODE_ROWS[0] or not _fixed_order(backends[0], rows, plain_backend):
                terms = _pick_terms(backends[:2], lambda: forward(adaptive, x, counts))
                problem = _check_terms(terms, routed)
                problem = problem and f"{rows} rows: {problem}"
            for name, got in (("staged", staged), ("router", routed)):
                exact = _fixed_order(backends[0], rows, plain_backend)
                if problem is None and not _same(got, want, None if exact else terms):
                    problem = f"{rows} rows: the {name} route's output differs from the layer without a balancer"
            exact = _fixed_order(backends[0], rows)
            if problem is None and not _same(routed, staged, None if exact else terms):
                problem = f"{rows} rows: the router route's output differs from the staged route's"
            if problem is None and not torch.equal(stats["rows"][0].cpu(), picks.cpu()):
                problem = f"{rows} rows: the router's statistics differ from the routing's picks"
            if problem is None and int(stats["forwards"][0].item()) != 1:
                problem = (
                    f"{rows} rows: the router counted {int(stats['forwards'][0].item())} forwards"
                )
            _assert_on_all_ranks(problem)

        # balance off: layers of the 512-slot configuration run as the layers without a balancer, on their
        # home slots: the same outputs (bit for bit in one fixed order, else within the order bound of the
        # plain layers' terms) and the same kernel launches (name, grid, block, shared memory), with no
        # route op, statistics or runtime. Before any capture: their home kernel's first eager launch
        # fences the adaptive layers' shared workspace.
        off_config = model_config(False)
        off_config.moe_load_balancer = MoeAdaptiveLoadBalancerConfig(
            num_slots=_NUM_SLOTS, balance="off"
        )
        with maybe_create_moe_load_balancer(off_config, mapping) as off_balancer:
            off = torch.nn.ModuleList(build(off_config, i) for i in range(2))
        off_balancer.register_weight_slots_after_to_cuda()
        off_balancer.finalize_model()
        off_balancer.prepare_runtime(off, list(_DECODE_ROWS))
        detached = all(layer.experts.layer_load_balancer is None for layer in off)
        problem = (
            None if detached and off_balancer.runtime is None else "balance off kept its balancer"
        )
        for rows in (*_DECODE_ROWS, *_PREFILL_ROWS):
            counts = [rows] * world_size
            x = torch.randn((rows, _HIDDEN_SIZE), dtype=torch.bfloat16)
            want, got = forward(plain, x, counts), forward(off, x, counts)
            terms = None
            if problem is None and not _fixed_order(off[0].experts.backend, rows, plain_backend):
                terms = _pick_terms(plain_backends, lambda: forward(plain, x, counts))
                problem = _check_terms(terms, want)
                problem = problem and f"{rows} rows: balance off: {problem}"
            if problem is None and not _same(got, want, terms):
                problem = f"{rows} rows: balance off differs from the layer without a balancer"
        for rows in _DECODE_ROWS:
            if problem is None and _launches(off[0], rows) != _launches(plain[0], rows):
                problem = f"{rows} rows: balance off launches other kernels than the plain layer"
        _assert_on_all_ranks(problem)

        # The route tables a runtime install rewrites in place (fixed address), and a replicated version.
        tables = balancer.route_tables(device)
        home_tables = tables.clone()
        replica_tables = torch.stack(
            [_replica_row(home_tables[i].tolist()) for i in range(_LAYERS)]
        ).to(device)
        # A table with malformed entries (every 7th expert: no copy, or its slot outside the slots): the router
        # drops those picks, which add nothing and are not waited for, as the staged route's out-of-range slots
        # do. A well-formed forward first leaves the dropped picks' combine cells holding real rows.
        malformed = home_tables.clone()
        bad = torch.arange(0, _NUM_EXPERTS, 7, device=device)
        malformed[:, bad[::2]] = 0
        for row in malformed:
            row[2 * _NUM_EXPERTS + row[_NUM_EXPERTS + bad[1::2]].long()] = _NUM_SLOTS
        for rows in (64, 1024):
            counts = [rows] * world_size
            x = torch.randn((rows, _HIDDEN_SIZE), dtype=torch.bfloat16)
            got = {}
            for router in (False, True):
                use_router(router)
                tables.copy_(home_tables)
                got["full", router] = forward(adaptive, x, counts)
                tables.copy_(malformed)
                got["dropped", router] = forward(adaptive, x, counts)
            torch.cuda.synchronize()
            problem = None
            assert _fixed_order(backends[0], rows), (
                "the malformed-table rows sum in one fixed order"
            )
            if _same(got["dropped", False], got["full", False]):
                problem = f"{rows} rows: the malformed table dropped no pick"
            elif not _same(got["dropped", True], got["dropped", False]):
                problem = f"{rows} rows: the router's dropped picks add to the output"
            _assert_on_all_ranks(problem)
        tables.copy_(home_tables)
        use_router(True)
        # Kernels per layer call (torch profiler): the router's route launches no route op or staging.
        names = {}
        for arm, router in (("adaptive_staged", False), ("adaptive_router", True)):
            use_router(router)
            x = torch.randn((64, _HIDDEN_SIZE), dtype=torch.bfloat16)
            adaptive[0](x, all_rank_num_tokens=[64] * world_size)
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                adaptive[0](x, all_rank_num_tokens=[64] * world_size)
                torch.cuda.synchronize()
            names[arm] = sorted(
                e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
            )
        staged_only = [
            n for n in names["adaptive_router"] if "StageInputs" in n or "adaptiveEplbRoute" in n
        ]
        _assert_on_all_ranks(
            None if not staged_only else f"the router's route still launches {staged_only}"
        )

        # After every eager pairing has run (as the executor captures after its warmup): a graph captured
        # on the home tables replays an install's replicated tables as an eager forward routes them (replica
        # slots hold no copied weights here, so the check is replay against eager, not against the plain layer).
        tables.copy_(home_tables)
        use_router(True)
        inputs = {}
        for rows in (1, 64, 1024):
            inputs[rows] = torch.randn((rows, _HIDDEN_SIZE), dtype=torch.bfloat16)
            for _ in range(2):
                adaptive[0](inputs[rows], all_rank_num_tokens=[rows] * world_size)
        torch.cuda.synchronize()
        graphs = {}
        for rows, x in inputs.items():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                slot = adaptive[0](x, all_rank_num_tokens=[rows] * world_size).clone()
            graphs[rows] = (graph, slot)
        tables.copy_(replica_tables)
        torch.cuda.synchronize()
        for rows, (graph, slot) in graphs.items():
            slot.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            eager = adaptive[0](inputs[rows], all_rank_num_tokens=[rows] * world_size).clone()
            torch.cuda.synchronize()
            terms = None
            if not _fixed_order(backends[0], rows):
                terms = _pick_terms(
                    backends[:1],
                    lambda rows=rows: [
                        adaptive[0](inputs[rows], all_rank_num_tokens=[rows] * world_size).clone()
                    ],
                )
            same = _same([slot], [eager], terms)
            _assert_on_all_ranks(
                None if same else f"{rows} rows: the replay after an install differs from eager"
            )
        del graphs
        # The server's order: a forward whose first layer routes as a DeepSeek-V4 hash layer does (no router's
        # route: the staged launch's kernel) and whose other layers take the router's route, all on one bucket's
        # shared workspace; an eager forward, then the CUDA graph runner's warmup step and its capture, replayed
        # against eager. It runs after every eager arm: the layers without a balancer may get the same shared
        # workspace with another local layout, and an eager fence of theirs after a capture would zero a captured one.
        from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner

        use_router(True)
        backends[0].routes_adaptive_in_router = lambda num_tokens: False
        for rows in (64, 1024):
            counts = [rows] * world_size
            x = torch.randn((rows, _HIDDEN_SIZE), dtype=torch.bfloat16)
            forward(adaptive, x, counts)
            torch.cuda.synchronize()
            for _ in range(CUDAGraphRunner.WARMUP_STEPS):
                want = [layer(x, all_rank_num_tokens=counts).clone() for layer in adaptive]
            torch.cuda.synchronize()
            graph, problem = torch.cuda.CUDAGraph(), None
            try:
                with torch.cuda.graph(graph):
                    got = [layer(x, all_rank_num_tokens=counts).clone() for layer in adaptive]
            except RuntimeError as error:
                problem = f"{rows} rows: capture after the eager warmup failed: {error}"
            if problem is None:
                graph.replay()
                torch.cuda.synchronize()
                assert _fixed_order(backends[0], rows), (
                    "the mixed-forward rows sum in one fixed order"
                )
                if not _same(got, want):
                    problem = f"{rows} rows: the captured mixed forward differs from eager"
            del graph
            _assert_on_all_ranks(problem)
        use_router(True)


def _launches(layer, rows: int) -> list:
    """(name, grid, block, dynamic shared memory) of every kernel one forward of ``layer`` launches, in order."""
    import json
    import tempfile

    x = torch.randn((rows, _HIDDEN_SIZE), dtype=torch.bfloat16)
    counts = [rows] * mpi_world_size()
    layer(x, all_rank_num_tokens=counts)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        layer(x, all_rank_num_tokens=counts)
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json") as trace:
        prof.export_chrome_trace(trace.name)
        events = json.load(open(trace.name))["traceEvents"]
    kernels = sorted((e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    keys = ("grid", "block", "shared memory")
    return [(e["name"], *(str(e.get("args", {}).get(k)) for k in keys)) for e in kernels]


def test_the_rows_cover_every_sm107_generation_phase_bucket(monkeypatch) -> None:
    """The rows of the decode sizes and the first prefill size launch in every SM107 bucket with a
    generation-phase tactic, one each: the backend's SM107 launch ladder, then its tactic bucket."""
    from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as op
    from tensorrt_llm._torch.moe.fused_moe.mega_moe import mega_moe_cute_dsl as backend

    monkeypatch.setattr(backend, "get_sm_version", lambda: 107)
    ladder = backend.TrtllmCutedslMegaMoeNvfp4Impl._resolve_maxt_buckets(_MAX_NUM_TOKENS)
    rows = (*_DECODE_ROWS, _PREFILL_ROWS[0])
    launched = [min(bucket for bucket in ladder if bucket >= count) for count in rows]
    assert [op._sm107_tactic_bucket(bucket) for bucket in launched] == sorted(
        op._SM107_GENPHASE_TACTICS
    ), launched


@pytest.mark.threadleak(enabled=False)
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD_SIZE,
    reason=f"Requires {_WORLD_SIZE} GPUs",
)
@pytest.mark.skipif(
    get_sm_version() not in (100, 103, 107), reason="MegaMoE CuteDSL on SM100/SM103/SM107"
)
def test_megamoe_adaptive_route_matches() -> None:
    # The workers spawn before this process touches CUDA, as test_megamoe_e2e's do: the run lists this
    # module first and nothing here launches work on the parent's device.
    if mpi_world_size() > 1:
        pytest.skip("Spawns its own worker processes")
    with MPIPoolExecutor(
        max_workers=_WORLD_SIZE, initializer=_init_worker, initargs=(sys.path,)
    ) as executor:
        futures = [executor.submit(_run) for _ in range(_WORLD_SIZE)]
        for future in futures:
            future.result()

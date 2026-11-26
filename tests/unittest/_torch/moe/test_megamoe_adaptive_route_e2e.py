# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 MegaMoE layers on four GPUs (attention DP, EP4, 384 experts) with and without an adaptive load
balancer (512 slots, home route tables, no runtime), at decode (8 / 64 / 128 rows per rank, the one-round router;
on SM107 the generation-phase kernel up to 1024) and prefill (1024 / 4096 / 8192, the metadata-push router) sizes.

Bit for bit: an adaptive layer whose dispatch router maps and counts its routes returns what the same layer returns
on the staged route, and what a layer without the balancer returns; its statistics are the routing's picks of the
live rows, nothing while the recording flag is clear. A graph captured on home tables replays an install as eager
routes it, and the server's mixed forward captures after its warmup and replays as eager. Under ``balance`` off the
512-slot layers return what the layers without a balancer return and launch the same kernels.
"""

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
_DECODE_ROWS = (8, 64, 128)
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


def _same(backend, rows: int, got, want) -> bool:
    """Bit for bit, except in a bucket whose kernel sums the top-k picks in its FC2 epilogue (the SM107
    32-row bucket): there the picks add in arrival order, so the outputs agree within bf16 rounding."""
    bucket = backend._select_launch_max_tokens(rows)
    if not backend._use_in_kernel_fc2_reduce(bucket):
        return all(torch.equal(a, b) for a, b in zip(got, want))
    return all(torch.allclose(a, b, rtol=1.6e-2, atol=1e-5) for a, b in zip(got, want))


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
            for name, got in (("staged", staged), ("router", routed)):
                if not _same(backends[0], rows, got, want):
                    problem = f"{rows} rows: the {name} route's output differs from the layer without a balancer"
            if problem is None and not torch.equal(stats["rows"][0].cpu(), picks.cpu()):
                problem = f"{rows} rows: the router's statistics differ from the routing's picks"
            if problem is None and int(stats["forwards"][0].item()) != 1:
                problem = (
                    f"{rows} rows: the router counted {int(stats['forwards'][0].item())} forwards"
                )
            _assert_on_all_ranks(problem)

        # balance off: layers of the 512-slot configuration run as the layers without a balancer, on their
        # home slots: the same outputs bit for bit and the same kernel launches (name, grid, block, shared
        # memory), with no route op, statistics or runtime. Before any capture: their home kernel's first
        # eager launch fences the adaptive layers' shared workspace.
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
            if problem is None and not all(torch.equal(a, b) for a, b in zip(got, want)):
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
            same = _same(backends[0], rows, [slot], [eager])
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
                if not _same(backends[0], rows, got, want):
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

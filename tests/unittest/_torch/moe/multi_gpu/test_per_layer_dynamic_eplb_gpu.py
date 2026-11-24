# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-layer dynamic EPLB on four GPUs: HALO-Q routes, helper copies, TRTLLM-Gen MoE.

Requirements: four SM100-family GPUs of one NVLink domain with fabric memory
(MNNVL), and a build whose dynamic EPLB kernels were compiled (SM100/103/107 in
the CUDA architectures, CUDA 13.1 or newer for the TMA copy). The MPI workers
start with ``CUDA_SCALE_LAUNCH_QUEUES=4x``, which per-layer dynamic EPLB needs
before CUDA initializes.

* ``test_halo_q_routes_and_helper_copies`` drives the layer handle directly on
  synthetic per-expert planes whose rows identify their expert. Each rank
  checks every route of every rank that lands on it: a resident slot serves its
  own expert, and a helper row holds the routed expert's bytes after the READY
  wait. It also checks that the busiest rank never gets busier.
* ``test_trtllm_nvfp4_moe_matches_reference`` runs a ConfigurableMoE on the
  TRTLLM-Gen NVFP4 backend with NVLink all-to-all under per-layer dynamic EPLB
  against the dequantized reference, over several routings.

Every check is agreed on by all ranks before anything raises, so a failure on
one rank cannot leave the others inside a collective.
"""

import pickle
import sys

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

pytestmark = pytest.mark.threadleak(enabled=False)

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

_WORLD_SIZE = 4
_WORKER_ENV = {
    "CUDA_SCALE_LAUNCH_QUEUES": "4x",
    # A stuck cross-rank wait then fails within a minute instead of five.
    "TRTLLM_MOE_COMM_TIMEOUT_SEC": "60",
    "TRTLLM_MOE_COMM_WARMUP_TIMEOUT_SEC": "120",
}
_ROW_BYTES = 4096
_SCALE_BYTES = 64


def _init_worker(custom_paths):
    for custom_path in custom_paths:
        if custom_path.endswith("tests/unittest") and custom_path not in sys.path:
            sys.path.append(custom_path)


@pytest.fixture(scope="module")
def executor():
    if torch.cuda.device_count() < _WORLD_SIZE:
        pytest.skip(f"requires {_WORLD_SIZE} GPUs")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("per-layer dynamic EPLB kernels need SM100-family GPUs")
    with MPIPoolExecutor(
        initializer=_init_worker,
        initargs=(sys.path,),
        max_workers=_WORLD_SIZE,
        env=_WORKER_ENV,
    ) as pool:
        yield pool


def _run(executor, worker, *args):
    results = list(executor.map(worker, *zip(*[args] * _WORLD_SIZE)))
    skipped = [result["skip"] for result in results if "skip" in result]
    if skipped:
        assert len(skipped) == _WORLD_SIZE, "inconsistent platform support across ranks"
        pytest.skip(skipped[0])
    # Task order is independent of the MPI rank that runs it.
    assert sorted(result["rank"] for result in results) == list(range(_WORLD_SIZE))
    return results


def _platform_skip_reason(comm):
    """The same skip reason on every rank, or None."""
    from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory

    MnnvlMemory.initialize()
    if all(comm.allgather(MnnvlMemory.supports_mnnvl())):
        return None
    return "per-layer dynamic EPLB needs fabric (MNNVL) memory on every GPU"


def _raise_on_any_rank(comm, errors):
    """Raise on every rank if any rank recorded an error."""
    gathered = comm.allgather(list(errors))
    failures = [f"rank {rank}: {error}" for rank, items in enumerate(gathered) for error in items]
    if failures:
        raise AssertionError("\n".join(failures))


def _producer_errors(group):
    """Fault flags of the layer's last HALO-Q launch and copy, as messages."""
    errors = []
    try:
        group.scheduler.check_status()
    except RuntimeError as error:
        errors.append(f"HALO-Q status: {error}")
    copy_error = int(torch.ops.trtllm.moe_rebalance_tma_result(group.broadcaster.ctx)[0])
    if copy_error:
        errors.append(f"TMA copy error {copy_error}")
    return errors


def _weight_pattern(experts):
    columns = torch.arange(_ROW_BYTES, device=experts.device)
    return ((experts[:, None] * 131 + columns[None, :] * 7) % 251).to(torch.uint8)


def _scale_pattern(experts):
    # FP8 E4M3 bit patterns in [1, 0x77]: no NaN encoding; compared as bytes.
    columns = torch.arange(_SCALE_BYTES, device=experts.device)
    return ((experts[:, None] * 3 + columns[None, :]) % 0x77 + 1).to(torch.uint8)


class _ExpertPlanes(torch.nn.Module):
    """A backend stand-in whose per-expert operands identify their expert."""

    PLANES = ("weight", "scale", "alpha")

    def __init__(self, layer, mapping, layer_idx, first_expert, resident, max_tokens):
        super().__init__()
        self.layer_load_balancer = layer
        self.mapping = mapping
        self.layer_idx = layer_idx
        self.expert_size_per_partition = resident
        self.ep_process_group = None
        self.max_tokens_per_rank_capacity = max_tokens
        experts = torch.arange(first_expert, first_expert + resident, device="cuda")
        self.weight = torch.nn.Parameter(_weight_pattern(experts), requires_grad=False)
        self.register_buffer(
            "scale", _scale_pattern(experts).view(torch.float8_e4m3fn), persistent=False
        )
        self.alpha = torch.nn.Parameter(experts.to(torch.float32), requires_grad=False)


def _routes(num_tokens, top_k, num_experts, hot, generator):
    logits = torch.randn(num_tokens, num_experts, device="cuda", generator=generator)
    if hot:
        logits[:, list(hot)] += 4.0
    return torch.topk(logits, top_k, dim=-1).indices.to(torch.int32).contiguous()


def _generation_errors(comm, rank, world, resident, compute, logical, physical, rows, skewed):
    """Check one layer generation from every rank's routes and this rank's rows."""
    errors = []

    def expect(condition, message):
        if not condition:
            errors.append(message)

    routes = comm.allgather((logical.cpu(), physical.cpu()))
    weight = rows["weight"].cpu()
    scale = rows["scale"].view(torch.uint8).cpu()
    alpha = rows["alpha"].cpu()
    for source, (source_logical, source_physical) in enumerate(routes):
        expect(source_physical.shape == source_logical.shape, f"rank {source}: route shape")
        if source_physical.numel() == 0 or source_physical.shape != source_logical.shape:
            continue
        logical64 = source_logical.to(torch.int64)
        physical64 = source_physical.to(torch.int64)
        expect(
            bool(((physical64 >= 0) & (physical64 < world * compute)).all()),
            f"rank {source}: physical slot out of range",
        )
        owner = physical64 // compute
        local = physical64 % compute
        resident_route = local < resident
        # A resident slot serves exactly its own expert.
        expect(
            torch.equal(logical64[resident_route], (owner * resident + local)[resident_route]),
            f"rank {source}: a resident slot serves another expert",
        )
        # A helper slot on this rank holds the bytes of the expert routed to it.
        mine = (owner == rank) & ~resident_route
        if bool(mine.any()):
            experts = logical64[mine]
            helper_rows = local[mine]
            expect(
                torch.equal(alpha[helper_rows].to(torch.int64), experts),
                f"rank {source}: helper scalar plane does not hold the routed expert",
            )
            expect(
                torch.equal(weight[helper_rows], _weight_pattern(experts)),
                f"rank {source}: helper weight plane does not hold the routed expert",
            )
            expect(
                torch.equal(scale[helper_rows], _scale_pattern(experts)),
                f"rank {source}: helper scale plane does not hold the routed expert",
            )
    # The copy never writes resident rows.
    home = torch.arange(rank * resident, (rank + 1) * resident)
    expect(torch.equal(weight[:resident], _weight_pattern(home)), "resident weight rows changed")
    expect(torch.equal(alpha[:resident].to(torch.int64), home), "resident scalar rows changed")

    every_logical = torch.cat([item[0].flatten() for item in routes]).to(torch.int64)
    every_physical = torch.cat([item[1].flatten() for item in routes]).to(torch.int64)
    if every_logical.numel() and every_logical.numel() == every_physical.numel():
        before = torch.bincount(every_logical // resident, minlength=world)
        after = torch.bincount(every_physical // compute, minlength=world)
        loads = f"rank loads {before.tolist()} -> {after.tolist()}"
        expect(int(after.max()) <= int(before.max()), f"busiest rank got busier: {loads}")
        if skewed:
            expect(int(after.max()) < int(before.max()), f"no rebalancing: {loads}")
    return errors


def _halo_q_worker(num_experts, helper_slots, top_k, num_tokens, helper_slot_mode):
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import (
        PerLayerDynamicMoeLoadBalancer,
    )
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb.live_planes import (
        bind_live_weight_planes,
    )
    from tensorrt_llm._utils import mpi_comm
    from tensorrt_llm.llmapi.llm_args import MoePerLayerDynamicLoadBalancerConfig
    from tensorrt_llm.mapping import Mapping

    comm = mpi_comm()
    rank, world = comm.Get_rank(), comm.Get_size()
    torch.cuda.set_device(rank)
    skip = _platform_skip_reason(comm)
    if skip is not None:
        return {"rank": rank, "skip": skip}

    mapping = Mapping(
        world_size=world,
        rank=rank,
        tp_size=world,
        moe_ep_size=world,
        moe_tp_size=1,
        enable_attention_dp=True,
    )
    config = MoePerLayerDynamicLoadBalancerConfig(
        num_slots=num_experts + world * helper_slots,
        auxiliary_sms=4,
        helper_slot_mode=helper_slot_mode,
    )
    config.setup(ep_rank=rank, ep_size=world)
    manager = PerLayerDynamicMoeLoadBalancer(
        ep_rank=rank, ep_size=world, config=config, mapping=mapping
    )
    resident = num_experts // world
    compute = resident + helper_slots
    # MAIN, as in the executor: a stream of lower priority than the copy stream.
    main = torch.cuda.Stream()
    generator = torch.Generator(device="cuda").manual_seed(1234 + rank)
    layers = []

    def plan_and_check(layer, backend, logical, skewed, overlap_fn=None):
        with torch.cuda.stream(main):
            if overlap_fn is None:
                physical = layer.route(logical)
            else:
                physical = layer.route(logical, overlap_fn=overlap_fn)
            plan = layer.require_replica_plan()
            plan.wait_for_routes()
            # The routes view is the scheduler's output buffer: keep this generation's.
            physical = physical.clone()
            plan.wait_for_helper_weights()
            rows = {name: getattr(backend, name).clone() for name in _ExpertPlanes.PLANES}
            flags = plan.ready_flags.view(torch.int64).clone()
            generation = plan.ready_generation
            layer.finish_replica_plan(plan)
        main.synchronize()
        errors = _producer_errors(layer._scheduler_group)
        if not bool((flags.cpu() >= generation).all()):
            errors.append(f"READY terminals {flags.tolist()} below generation {generation}")
        errors += _generation_errors(
            comm, rank, world, resident, compute, logical, physical, rows, skewed
        )
        _raise_on_any_rank(comm, errors)

    released = False
    try:
        for layer_idx in range(2):
            layer = manager.add_layer(
                expert_count=num_experts, top_k=top_k, slot_count_per_rank=compute
            )
            backend = _ExpertPlanes(
                layer, mapping, layer_idx, rank * resident, resident, max_tokens=num_tokens
            )
            arena = bind_live_weight_planes(
                backend, _ExpertPlanes.PLANES, hidden=_ROW_BYTES, intermediate=_SCALE_BYTES
            )
            assert backend.weight.shape == (compute, _ROW_BYTES)
            assert backend.alpha.shape == (compute,)
            layer.bind_backend(backend)
            assert layer.live_weight_arena is arena
            layers.append((layer, backend))

        lhs = torch.randn(256, 256, device="cuda")
        # (hot experts, whether they make one rank the busiest)
        routings = [
            ((), False),
            (tuple(range(top_k)), True),  # experts of the first rank
            (tuple(range(num_experts - top_k, num_experts)), True),  # of the last rank
            (tuple(range(0, num_experts, num_experts // top_k))[:top_k], False),  # spread
        ]
        for iteration, (hot, skewed) in enumerate(routings):
            # An empty rank still joins every collective launch.
            tokens = 0 if (iteration == 2 and rank == 1) else num_tokens
            for layer, backend in layers:
                logical = _routes(tokens, top_k, num_experts, hot, generator)
                # Independent MAIN work may run inside the route, beside the copy.
                overlap_fn = (lambda: lhs @ lhs) if iteration == 1 else None
                plan_and_check(layer, backend, logical, skewed, overlap_fn)

        # A failed overlap releases its generation, and the layer plans again.
        layer, backend = layers[0]

        def fail():
            raise RuntimeError("overlapped work failed")

        logical = _routes(num_tokens, top_k, num_experts, (), generator)
        with torch.cuda.stream(main):
            try:
                layer.route(logical, overlap_fn=fail)
            except RuntimeError as error:
                assert "overlapped work failed" in str(error)
            else:
                raise AssertionError("the overlap failure was swallowed")
        assert layer.replica_plan is None
        plan_and_check(layer, backend, logical, skewed=False)
    finally:
        torch.cuda.synchronize()
        released = manager.shutdown()
    assert released
    return {"rank": rank}


@pytest.mark.parametrize(
    ("num_experts", "helper_slots", "top_k", "num_tokens", "helper_slot_mode"),
    [
        pytest.param(64, 2, 4, 512, "per_layer_resident", id="e64-s2"),
        # Kimi K3.1 routing: 896 experts, top-16, 8 helper slots per rank.
        pytest.param(896, 8, 16, 2048, "per_layer_resident", id="e896-s8"),
        pytest.param(896, 8, 16, 2048, "shared_banks", id="e896-s8-shared-banks"),
        # The native expert bound.
        pytest.param(1024, 4, 8, 1024, "per_layer_resident", id="e1024-s4"),
    ],
)
def test_halo_q_routes_and_helper_copies(
    executor, num_experts, helper_slots, top_k, num_tokens, helper_slot_mode
):
    _run(executor, _halo_q_worker, num_experts, helper_slots, top_k, num_tokens, helper_slot_mode)


def _reset_nvlink_one_sided_workspace():
    """Release the one-sided workspace singleton so the next case sizes its own."""
    import gc

    from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided

    for name in ("_WORKSPACES", "_WORKSPACE_REFCOUNTS"):
        cache = getattr(NVLinkOneSided, name, None)
        if isinstance(cache, dict):
            cache.clear()
    if hasattr(NVLinkOneSided, "_WORKSPACE"):
        NVLinkOneSided._WORKSPACE = None
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _trtllm_moe_worker(
    num_experts,
    top_k,
    hidden_size,
    intermediate_size,
    helper_slots,
    seq_len,
    moe_max_num_tokens,
    communication_method,
):
    from _torch.moe.quantize_utils import get_test_quant_params
    from transformers.configuration_utils import PretrainedConfig

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.moe.fused_moe import RenormalizeMoeRoutingMethod, create_moe
    from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import MoeLoadBalancerIterContext
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import (
        PerLayerDynamicMoeLoadBalancer,
    )
    from tensorrt_llm._torch.nccl_window_tensor_scope import nccl_window_tensor_scope
    from tensorrt_llm._utils import mpi_comm
    from tensorrt_llm.llmapi.llm_args import MoePerLayerDynamicLoadBalancerConfig
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantAlgo

    comm = mpi_comm()
    rank, world = comm.Get_rank(), comm.Get_size()
    torch.cuda.set_device(rank)
    skip = _platform_skip_reason(comm)
    if skip is not None:
        return {"rank": rank, "skip": skip}

    dtype = torch.bfloat16
    mapping = Mapping(
        world_size=world,
        rank=rank,
        tp_size=world,
        moe_ep_size=world,
        moe_tp_size=1,
        enable_attention_dp=True,
    )
    config = MoePerLayerDynamicLoadBalancerConfig(
        num_slots=num_experts + world * helper_slots, auxiliary_sms=4
    )
    config.setup(ep_rank=rank, ep_size=world)
    manager = PerLayerDynamicMoeLoadBalancer(
        ep_rank=rank, ep_size=world, config=config, mapping=mapping
    )
    resident = num_experts // world
    compute = resident + helper_slots
    all_rank_num_tokens = [seq_len] * world
    main = torch.cuda.Stream()
    released = False
    try:
        with torch.device(f"cuda:{rank}"):
            # Every rank draws the same inputs and weights.
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            routing_method = RenormalizeMoeRoutingMethod(top_k=top_k, force_enable_pytorch_op=True)
            x = torch.randn((seq_len, hidden_size), dtype=dtype, device="cuda")
            quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
                QuantAlgo.NVFP4, x, "TRTLLM"
            )
            quantize_util = quantize_util_cls(
                num_experts=num_experts,
                dtype=dtype,
                intermediate_size=intermediate_size,
                hidden_size=hidden_size,
                quant_config=quant_config,
                num_local_experts=resident,
            )
            pretrained_config = PretrainedConfig()
            pretrained_config.num_experts = num_experts
            pretrained_config.hidden_size = hidden_size
            pretrained_config.intermediate_size = intermediate_size
            pretrained_config.torch_dtype = dtype
            model_kwargs = dict(
                pretrained_config=pretrained_config,
                mapping=mapping,
                quant_config=quant_config,
                moe_backend="TRTLLM",
                moe_disable_finalize_fusion=False,
                moe_load_balancer=config,
                max_num_tokens=max(256, seq_len),
            )
            if moe_max_num_tokens is not None:
                model_kwargs["moe_max_num_tokens"] = moe_max_num_tokens
            model_config = ModelConfig(**model_kwargs)

            with (
                nccl_window_tensor_scope(x),
                manager,
                create_moe(
                    routing_method=routing_method,
                    reduce_results=True,
                    model_config=model_config,
                    communication_method=communication_method,
                ) as fused_moe,
            ):
                fused_moe.create_weights()
                weights = quantize_util.create_weights(**quant_kwargs)
                fused_moe.load_weights([weights])
                # Moves the expert operands onto the H + S arena and builds the producer.
                fused_moe.post_load_weights()
                backend = fused_moe.backend
                layer = fused_moe.layer_load_balancer
                assert layer is backend.layer_load_balancer
                assert layer.live_weight_arena is not None
                for name in ("w3_w1_weight", "w3_w1_weight_scale", "w2_weight", "w2_weight_scale"):
                    assert getattr(backend, name).shape[0] == compute, name
                assert fused_moe.num_slots == world * compute

                ref_fused_moe = quantize_util.create_ref_module(routing_method)
                ref_fused_moe.load_weights([weights])
                ref_fused_moe.cuda(f"cuda:{rank}")

                helper_routes = 0
                shared = []
                lhs = torch.randn(256, 256, device="cuda")
                routings = [(), tuple(range(top_k)), tuple(range(num_experts - top_k, num_experts))]
                for iteration, hot in enumerate(routings):
                    router_logits = torch.randn((seq_len, num_experts), dtype=dtype, device="cuda")
                    if hot:
                        router_logits[:, list(hot)] += 4.0
                    with torch.inference_mode(), torch.cuda.stream(main):
                        main.wait_stream(torch.cuda.default_stream())
                        with MoeLoadBalancerIterContext(manager):
                            if iteration == 1:
                                # The Kimi K3 shared-expert hook: runs inside the first route.
                                with fused_moe.load_balancer_overlap(
                                    lambda: shared.append(lhs @ lhs)
                                ):
                                    output = fused_moe.forward(
                                        x, router_logits, all_rank_num_tokens=all_rank_num_tokens
                                    )
                            else:
                                output = fused_moe.forward(
                                    x, router_logits, all_rank_num_tokens=all_rank_num_tokens
                                )
                        torch.cuda.default_stream().wait_stream(main)
                    torch.cuda.synchronize()
                    group = layer._scheduler_group
                    errors = _producer_errors(group)
                    # The physical routes of the last chunk stay in the scheduler output.
                    physical = group.scheduler.outputs.physical_slot_ids
                    helper_routes += int(((physical >= 0) & (physical % compute >= resident)).sum())
                    with torch.inference_mode():
                        ref_output = ref_fused_moe.forward(x, router_logits)
                    try:
                        ref_fused_moe.check_accuracy(output, ref_output)
                    except AssertionError as error:
                        errors.append(f"routing {iteration}: {error}")
                    _raise_on_any_rank(comm, errors)
                assert manager.iter_id == len(routings)
                assert len(shared) == 1
                torch.testing.assert_close(shared[0], lhs @ lhs)
                # The hot experts were served from helper slots somewhere.
                assert sum(comm.allgather(helper_routes)) > 0
    finally:
        torch.cuda.synchronize()
        released = manager.shutdown()
        _reset_nvlink_one_sided_workspace()
    assert released
    return {"rank": rank}


@pytest.mark.parametrize(
    (
        "num_experts",
        "top_k",
        "hidden_size",
        "intermediate_size",
        "helper_slots",
        "seq_len",
        "moe_max_num_tokens",
        "communication_method",
    ),
    [
        pytest.param(64, 8, 1024, 512, 4, 64, None, "NVLINK_ONE_SIDED", id="e64-one-sided"),
        pytest.param(64, 8, 1024, 512, 4, 64, None, "NVLINK_TWO_SIDED", id="e64-two-sided"),
        # Four MoE chunks per forward: each plans and releases its own generation.
        pytest.param(64, 8, 1024, 512, 4, 64, 64, "NVLINK_ONE_SIDED", id="e64-chunked"),
        # Kimi K3.1 routing width (896 experts, top-16, 8 helpers) on small experts.
        pytest.param(896, 16, 512, 512, 8, 128, None, "NVLINK_ONE_SIDED", id="e896-top16"),
    ],
)
def test_trtllm_nvfp4_moe_matches_reference(
    executor,
    num_experts,
    top_k,
    hidden_size,
    intermediate_size,
    helper_slots,
    seq_len,
    moe_max_num_tokens,
    communication_method,
):
    _run(
        executor,
        _trtllm_moe_worker,
        num_experts,
        top_k,
        hidden_size,
        intermediate_size,
        helper_slots,
        seq_len,
        moe_max_num_tokens,
        communication_method,
    )

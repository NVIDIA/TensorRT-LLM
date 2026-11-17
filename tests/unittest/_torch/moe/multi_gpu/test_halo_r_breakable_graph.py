# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HALO-R under ``BreakableCUDAGraphRunner``: capture, replay and abort on a small MoE model.

The model is two MegaMoE CuteDSL layers (NVFP4, random weights) with per-layer
dynamic EPLB and ``planner="halo_r"`` at its defaults, one EP rank per GPU.
Each forward runs both layers on the same input, so each layer's output is
checked on its own against the NVFP4 reference MoE. Three quarters of each
rank's tokens pick the experts of one hot rank, so every plan uses helper slots.
The runner is driven as the decoder engine drives it: the forward enters the
EPLB iteration context and, while the runner captures, captures only the body.

Every rank checks, in order:
- eager forwards: the output of each layer, and that every route reaches its
  expert's home slot or a helper slot holding a byte-exact copy of the
  expert's weights, with some route on a helper slot;
- the runner warms up and captures two token buckets (largest first);
- replays of both buckets with fresh inputs in the static buffers, and an
  eager forward at an uncaptured token count between them: the outputs, that
  no replay called the layers' routes on the host, and that each layer's
  kernel launched once per eager forward and once per replay with its relay
  using helper slots during the replays;
- abort: a failure inside the second layer's MoE kernel call while the runner
  captures a third bucket. The capture raises, the runner keeps its two
  graphs, and the second layer refuses every later launch;
- the EPLB manager shuts down.

The path needs an SM100-family GPU with an NVLink fabric, four GPUs and a
build with the HALO-R kernels. HALO_R_TEST_NO_SKIP=1 turns a skip into a
failure, as in ``test_halo_r_relay.py``.
"""

import faulthandler
import os
import pickle
import sys
import traceback
from typing import Any

import cloudpickle
import pytest
import torch
import torch.distributed as dist
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

_EP_SIZE = 4
_LAYERS = 2
# HALO-R needs at least 4 experts per rank.
_NUM_EXPERTS = 16
_TOP_K = 4
_HIDDEN = 512
_INTERMEDIATE = 512
_HELPER_SLOTS_PER_RANK = 2
_MAX_NUM_TOKENS = 256
# Captured token buckets, largest first as the runner requires; an eager
# forward runs at the uncaptured count, and the abort case captures the last.
_BUCKETS = (128, 64)
_EAGER_TOKENS = 96
_ABORT_TOKENS = 32
# The hot rank of each replay step and of each eager forward.
_REPLAY_HOT_RANKS = (1, 2, 3, 0)
_EAGER_HOT_RANKS = (0, 1)
_WORKER_TIMEOUT_S = 900


class _CaptureFailure(Exception):
    """Stands for an error raised inside a captured MoE kernel call, such as a graph-pool OOM."""


def _skip_reason() -> str | None:
    from _torch.moe.moe_test_utils import (
        MoeBackendType,
        MoeModelConfig,
        should_skip_megamoe_cutedsl,
    )

    from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
    from tensorrt_llm.models.modeling_utils import QuantAlgo

    if torch.cuda.get_device_capability()[0] != 10:
        return "HALO-R requires an SM100-family GPU"
    MnnvlMemory.initialize()
    if not MnnvlMemory.supports_mnnvl():
        return "HALO-R requires an NVLink fabric"
    return should_skip_megamoe_cutedsl(
        MoeBackendType.MEGAMOE_CUTEDSL,
        quant_algo=QuantAlgo.NVFP4,
        dtype=torch.bfloat16,
        model_config=MoeModelConfig(_NUM_EXPERTS, _TOP_K, _HIDDEN, _INTERMEDIATE),
    )


def _init_worker(paths: list[str]) -> None:
    # Per-layer dynamic EPLB needs the larger launch queues from CUDA initialization on.
    os.environ["CUDA_SCALE_LAUNCH_QUEUES"] = "4x"
    for path in paths:
        if path.endswith("tests/unittest") and path not in sys.path:
            sys.path.append(path)


def _init_process_group(rank: int, world: int) -> None:
    """Join the NCCL WORLD group, which MegaMoE resolves as its EP group."""
    host = "127.0.0.1"
    store = None
    if rank == 0:
        store = dist.TCPStore(host, 0, world, is_master=True, wait_for_workers=False)
    port = MPI.COMM_WORLD.bcast(store.port if store is not None else None, root=0)
    if store is None:
        store = dist.TCPStore(host, port, world, is_master=False, wait_for_workers=False)
    os.environ.update(
        MASTER_ADDR=host,
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world),
        LOCAL_RANK=str(rank),
    )
    dist.init_process_group(backend="nccl", store=store, rank=rank, world_size=world)


def _skewed_logits(logits: torch.Tensor, hot_rank: int) -> torch.Tensor:
    """Point three quarters of the tokens at the experts of ``hot_rank``."""
    home = _NUM_EXPERTS // _EP_SIZE
    skewed = logits.clone()
    skewed[: logits.shape[0] * 3 // 4, hot_rank * home : (hot_rank + 1) * home] += 8
    return skewed


def _record_routes(layer: Any) -> list[tuple[torch.Tensor, torch.Tensor, bool]]:
    """Record (logical, physical, captured) for every route the layer plans on the host."""
    records = []
    route = layer.route

    def recording_route(
        token_selected_experts: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        captured = torch.cuda.is_current_stream_capturing()
        physical = route(token_selected_experts, *args, **kwargs)
        records.append((token_selected_experts, physical, captured))
        return physical

    layer.route = recording_route
    return records


def _route_errors(layer: Any, logical: torch.Tensor, physical: torch.Tensor) -> list[str]:
    """Check one eager forward's routes against the slot weights. Collective over the EP group."""
    world = dist.get_world_size()
    logical = logical.flatten().to(torch.int64)
    physical = physical.flatten().to(torch.int64)
    arena = layer.live_weight_arena
    home, slots = arena.home_experts, arena.slot_count
    errors = []
    in_range = (physical >= 0) & (physical < world * slots)
    if not in_range.all():
        errors.append(f"{int((~in_range).sum())} routes fall outside the physical slots")
    home_row = logical // home * slots + logical % home
    is_home = in_range & (physical % slots < home)
    if (is_home & (physical != home_row)).any():
        errors.append("a route reaches a home slot of another expert")
    is_helper = in_range & (physical % slots >= home)
    pairs = torch.unique(torch.stack([physical[is_helper], home_row[is_helper]], dim=1), dim=0)
    for name, plane in zip(arena.plane_names, arena.tekit_alias_views):
        rows = torch.stack([row.contiguous().reshape(-1).view(torch.uint8) for row in plane])
        gathered = rows.new_empty((world * rows.shape[0], rows.shape[1]))
        dist.all_gather_into_tensor(gathered, rows)
        if (gathered[pairs[:, 0]] != gathered[pairs[:, 1]]).any():
            errors.append(f"a helper slot read by a route holds another expert's {name}")
    helper_routes = is_helper.sum().reshape(1)
    dist.all_reduce(helper_routes)
    if helper_routes.item() == 0:
        errors.append("no route used a helper slot")
    return errors


def _relay_counts(layer: Any) -> tuple[int, int]:
    """(launches, receiver copies + skips) of the layer's HALO-R kernel, eager or replayed."""
    torch.cuda.synchronize()
    launches, copies, skips = layer._scheduler_group.broadcaster.relay_counts()
    return launches, copies + skips


def _run_rank() -> tuple[int, list[str]]:
    """Run every step on this EP rank; return (rank, errors)."""
    from tensorrt_llm._utils import mpi_rank

    faulthandler.dump_traceback_later(_WORKER_TIMEOUT_S, exit=True)
    rank = mpi_rank()
    try:
        torch.cuda.set_device(rank)
        reason = _skip_reason()
        reasons = MPI.COMM_WORLD.allgather(reason)
        if any(reasons):
            return rank, ["skip: " + "; ".join(sorted({r for r in reasons if r}))]
        _init_process_group(rank, _EP_SIZE)
        # The test utilities create weights and scales on the default device.
        with torch.device(f"cuda:{rank}"):
            return rank, _run_steps(rank)
    except Exception:
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        faulthandler.cancel_dump_traceback_later()


def _run_steps(rank: int) -> list[str]:
    from _torch.moe.moe_test_utils import MoeBackendType
    from _torch.moe.quantize_utils import get_test_quant_params
    from transformers.configuration_utils import PretrainedConfig

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.moe.fused_moe import RenormalizeMoeRoutingMethod, create_moe
    from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import (
        MoeLoadBalancerIterContext,
        maybe_create_moe_load_balancer,
    )
    from tensorrt_llm._torch.nccl_window_tensor_scope import nccl_window_tensor_scope
    from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph_runner import (
        BreakableCUDAGraphRunner,
        BreakableCUDAGraphRunnerState,
    )
    from tensorrt_llm.llmapi.llm_args import MoePerLayerDynamicLoadBalancerConfig
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantAlgo

    dtype = torch.bfloat16
    # Every rank draws every step's inputs, so all ranks quantize the weights
    # with one activation scale; each rank takes its own rows.
    torch.manual_seed(0)
    # The eager forwards, the captures (one input), the replays, the eager forward between
    # them and the failed capture.
    steps = len(_EAGER_HOT_RANKS) + 1 + len(_REPLAY_HOT_RANKS) + 1 + 1
    all_x = torch.randn((steps, _EP_SIZE, _MAX_NUM_TOKENS, _HIDDEN), dtype=dtype, device="cuda")
    all_logits = torch.randn(
        (steps, _EP_SIZE, _MAX_NUM_TOKENS, _NUM_EXPERTS), dtype=dtype, device="cuda"
    )
    inputs = [(all_x[step, rank], all_logits[step, rank]) for step in range(steps)]

    routing_method = RenormalizeMoeRoutingMethod(top_k=_TOP_K, force_enable_pytorch_op=True)
    quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
        QuantAlgo.NVFP4, all_x, MoeBackendType.MEGAMOE_CUTEDSL
    )
    ref_cls = quant_kwargs.pop("ref_cls")
    quantize_util = quantize_util_cls(
        num_experts=_NUM_EXPERTS,
        dtype=dtype,
        intermediate_size=_INTERMEDIATE,
        hidden_size=_HIDDEN,
        quant_config=quant_config,
        num_local_experts=_NUM_EXPERTS // _EP_SIZE,
    )
    weights = [quantize_util.create_weights(**quant_kwargs) for _ in range(_LAYERS)]

    mapping = Mapping(
        world_size=_EP_SIZE,
        tp_size=_EP_SIZE,
        moe_ep_size=_EP_SIZE,
        moe_tp_size=1,
        enable_attention_dp=True,
    )
    mapping.rank = rank
    pretrained_config = PretrainedConfig()
    # Any architecture with EPLB support; this one adds no MoE specifics.
    pretrained_config.architectures = ["Qwen3MoeForCausalLM"]
    pretrained_config.num_experts = _NUM_EXPERTS
    pretrained_config.hidden_size = _HIDDEN
    pretrained_config.intermediate_size = _INTERMEDIATE
    pretrained_config.torch_dtype = dtype
    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        mapping=mapping,
        quant_config=quant_config,
        moe_backend=MoeBackendType.MEGAMOE_CUTEDSL.value,
        # HALO-R at its defaults: 16 auxiliary SMs, per-layer resident helper slots.
        moe_load_balancer=MoePerLayerDynamicLoadBalancerConfig(
            num_slots=_NUM_EXPERTS + _EP_SIZE * _HELPER_SLOTS_PER_RANK, planner="halo_r"
        ),
        max_num_tokens=_MAX_NUM_TOKENS,
    )

    # The static inputs the captured graphs read; replays refill them.
    static_x = torch.zeros((_MAX_NUM_TOKENS, _HIDDEN), dtype=dtype, device="cuda")
    static_logits = torch.zeros((_MAX_NUM_TOKENS, _NUM_EXPERTS), dtype=dtype, device="cuda")
    errors = []
    with (
        nccl_window_tensor_scope(static_x),
        maybe_create_moe_load_balancer(model_config, mapping) as balancer,
    ):
        moes, refs = [], []
        for layer_idx in range(_LAYERS):
            moe = create_moe(
                routing_method=routing_method,
                reduce_results=True,
                model_config=model_config,
                layer_idx=layer_idx,
            )
            moe.create_weights()
            moe.load_weights([weights[layer_idx]])
            moe.post_load_weights()
            ref = quantize_util.create_ref_module(routing_method, ref_cls=ref_cls)
            ref.load_weights([weights[layer_idx]])
            ref.cuda(f"cuda:{rank}")
            moes.append(moe)
            refs.append(ref)
        layers = [moe.layer_load_balancer for moe in moes]
        if (balancer.planner, balancer.auxiliary_sms) != ("halo_r", 16):
            errors.append(f"rank {rank}: planner {balancer.planner}, {balancer.auxiliary_sms} SMs")
        records = [_record_routes(layer) for layer in layers]
        body = _TwoLayerBody(moes)
        runner = BreakableCUDAGraphRunner(body)
        num_tokens = {"value": _MAX_NUM_TOKENS}

        def run_model() -> torch.Tensor:
            tokens = num_tokens["value"]

            def run_body() -> torch.Tensor:
                return body(static_x[:tokens], static_logits[:tokens])

            with torch.inference_mode(), MoeLoadBalancerIterContext(balancer):
                if runner.is_capturing:
                    return runner.capture_model_body(run_body)
                return run_body()

        def load(step: int, hot_rank: int, tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
            x, logits = inputs[step]
            x, logits = x[:tokens], _skewed_logits(logits[:tokens], hot_rank)
            static_x[:tokens].copy_(x)
            static_logits[:tokens].copy_(logits)
            num_tokens["value"] = tokens
            return x, logits

        def check_output(
            where: str, output: torch.Tensor, x: torch.Tensor, logits: torch.Tensor
        ) -> None:
            for layer_idx, ref in enumerate(refs):
                try:
                    with torch.inference_mode():
                        ref_output = ref.forward(x, logits)
                    ref.check_accuracy(output[:, layer_idx], ref_output)
                except AssertionError as error:
                    errors.append(f"rank {rank} {where} layer {layer_idx}: output: {error}")

        def synchronize() -> None:
            torch.cuda.synchronize()
            dist.barrier()

        step = 0
        eager_launches = [0] * _LAYERS

        def eager_forward(where: str, hot_rank: int, tokens: int) -> None:
            nonlocal step
            x, logits = load(step, hot_rank, tokens)
            step += 1
            output = run_model().clone()
            synchronize()
            check_output(where, output, x, logits)
            for layer_idx, layer in enumerate(layers):
                calls = records[layer_idx]
                if len(calls) == 1 and not calls[0][2]:
                    logical, physical, _ = calls.pop()
                    eager_launches[layer_idx] += 1
                else:
                    errors.append(f"rank {rank} {where} layer {layer_idx}: route calls {calls}")
                    calls.clear()
                    # _route_errors is collective: every rank calls it, with no routes if need be.
                    logical = physical = torch.empty(0, dtype=torch.int64, device="cuda")
                errors.extend(
                    f"rank {rank} {where} layer {layer_idx}: {error}"
                    for error in _route_errors(layer, logical, physical)
                )

        # Eager forwards outside the runner.
        for hot_rank in _EAGER_HOT_RANKS:
            eager_forward(f"eager (hot rank {hot_rank})", hot_rank, _MAX_NUM_TOKENS)

        # Warm up and capture each bucket, on one input.
        for tokens in _BUCKETS:
            load(step, 0, tokens)
            runner.capture(tokens, run_model)
            synchronize()
            for layer_idx in range(_LAYERS):
                calls = records[layer_idx]
                captured = [c for _, _, c in calls]
                if captured.count(True) != 1 or captured[-1] is not True:
                    errors.append(
                        f"rank {rank} capture {tokens} layer {layer_idx}: route calls "
                        f"captured {captured}, expected eager warmups then one capture"
                    )
                eager_launches[layer_idx] += captured.count(False)
                calls.clear()
        if runner.captured_num_tokens != sorted(_BUCKETS):
            errors.append(f"rank {rank}: captured buckets {runner.captured_num_tokens}")
        step += 1

        # Replays with fresh inputs, and one eager forward between them.
        replays = 0
        # Per layer, the helper receivers (copied or already resident) of the replays.
        replay_receivers = [0] * _LAYERS
        for index, hot_rank in enumerate(_REPLAY_HOT_RANKS):
            tokens = _BUCKETS[index % len(_BUCKETS)]
            x, logits = load(step, hot_rank, tokens)
            step += 1
            before = [_relay_counts(layer)[1] for layer in layers]
            output = runner.execute(tokens, run_model).clone()
            replays += 1
            synchronize()
            for layer_idx, layer in enumerate(layers):
                replay_receivers[layer_idx] += _relay_counts(layer)[1] - before[layer_idx]
            check_output(f"replay {index} ({tokens} tokens)", output, x, logits)
            if any(records):
                errors.append(f"rank {rank} replay {index}: the host planned routes")
                for calls in records:
                    calls.clear()
            if index == 1:
                eager_forward("eager between replays", hot_rank, _EAGER_TOKENS)
        for layer_idx, layer in enumerate(layers):
            launches, _ = _relay_counts(layer)
            if launches != eager_launches[layer_idx] + replays:
                errors.append(
                    f"rank {rank} layer {layer_idx}: {launches} kernel launches, expected "
                    f"{eager_launches[layer_idx]} eager + {replays} replays"
                )
        receivers = torch.tensor(replay_receivers, device="cuda")
        dist.all_reduce(receivers)
        if (receivers == 0).any():
            errors.append(f"rank {rank}: replays used no helper slot: {receivers.tolist()}")

        # Abort: the second layer's MoE kernel call fails while the runner captures.
        backend = moes[1].backend
        run_moe = backend.run_moe

        def failing_run_moe(*args: Any, **kwargs: Any) -> torch.Tensor:
            if torch.cuda.is_current_stream_capturing():
                raise _CaptureFailure()
            return run_moe(*args, **kwargs)

        backend.run_moe = failing_run_moe
        previous_stream = torch.cuda.current_stream()
        load(step, 0, _ABORT_TOKENS)
        try:
            runner.capture(_ABORT_TOKENS, run_model)
            errors.append(f"rank {rank}: the capture with a failing MoE call did not raise")
        except Exception as error:  # noqa: BLE001
            if not _caused_by(error, _CaptureFailure):
                errors.append(f"rank {rank}: the failed capture raised {error!r}")
        finally:
            backend.run_moe = run_moe
        # A failed capture end can leave the capture stream current.
        torch.cuda.set_stream(previous_stream)
        synchronize()
        if torch.cuda.is_current_stream_capturing():
            errors.append(f"rank {rank}: a stream is still capturing after the failed capture")
        if runner.state != BreakableCUDAGraphRunnerState.IDLE or runner.has_graph(_ABORT_TOKENS):
            errors.append(f"rank {rank}: runner {runner.state} after the failed capture")
        if runner.captured_num_tokens != sorted(_BUCKETS):
            errors.append(f"rank {rank}: the failed capture changed the captured buckets")
        if layers[1].replica_plan is not None:
            errors.append(f"rank {rank}: the failed capture left layer 1's plan pending")
        try:
            with torch.inference_mode(), MoeLoadBalancerIterContext(balancer):
                moes[1].forward(
                    static_x[:_ABORT_TOKENS],
                    static_logits[:_ABORT_TOKENS],
                    all_rank_num_tokens=[_ABORT_TOKENS] * _EP_SIZE,
                )
            errors.append(f"rank {rank}: layer 1 launched again after its capture failed")
        except RuntimeError as error:
            if "restart the process" not in str(error):
                errors.append(f"rank {rank}: layer 1 relaunch raised {error!r}")
        synchronize()
        runner.clear()
        if not balancer.shutdown():
            errors.append(f"rank {rank}: per-layer dynamic EPLB did not shut down")
    return errors


def _caused_by(error: BaseException, cause: type) -> bool:
    """Whether ``cause`` is ``error`` or in its chain of causes and contexts."""
    seen = set()
    while error is not None and id(error) not in seen:
        if isinstance(error, cause):
            return True
        seen.add(id(error))
        error = error.__cause__ or error.__context__
    return False


class _TwoLayerBody(torch.nn.Module):
    """The captured body: every MoE layer on one input; outputs as (tokens, layers, hidden)."""

    def __init__(self, moes: list) -> None:
        super().__init__()
        self.moes = torch.nn.ModuleList(moes)

    def forward(self, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        tokens = [int(x.shape[0])] * _EP_SIZE
        # Tokens stay the first dimension, which the runner's shared output is sized by.
        return torch.stack([moe(x, logits, all_rank_num_tokens=tokens) for moe in self.moes], dim=1)


@pytest.mark.threadleak(enabled=False)  # the MPIPoolExecutor manager thread outlives the test
def test_halo_r_breakable_cuda_graph() -> None:
    if torch.cuda.device_count() < _EP_SIZE:
        _skip(f"needs {_EP_SIZE} GPUs")
    if torch.cuda.get_device_capability(0)[0] != 10:
        _skip("HALO-R requires an SM100-family GPU")
    with MPIPoolExecutor(
        initializer=_init_worker, initargs=(sys.path,), max_workers=_EP_SIZE
    ) as executor:
        results = list(executor.starmap(_run_rank, [()] * _EP_SIZE))
    skips = {error for _, errors in results for error in errors if error.startswith("skip: ")}
    if skips:
        _skip("; ".join(sorted(skips)))
    failures = [error for _, errors in sorted(results) for error in errors]
    assert not failures, "\n".join(failures[:20])


def _skip(reason: str) -> None:
    if os.environ.get("HALO_R_TEST_NO_SKIP") == "1":
        pytest.fail(f"HALO_R_TEST_NO_SKIP=1 but the test would skip: {reason}")
    pytest.skip(reason)

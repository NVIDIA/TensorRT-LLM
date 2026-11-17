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

import atexit
import ctypes
import faulthandler
import os
import pickle
import signal
import sys
import time
import traceback
from concurrent.futures import FIRST_EXCEPTION, wait
from pathlib import Path
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
# The hot rank of each eager forward; None leaves the routing as drawn (HALO-R may still move a
# few routes to helper slots, but no helper route is required).
_EAGER_HOT_RANKS = (None, 0, 1)
# A rank dumps its stacks and exits after this long, and the pytest process fails with the rank
# logs within one poll (or 120 s later if no rank dumps): both inside the GB200 entry's TIMEOUT (10),
# so a hang in CI still ends with them.
_WORKER_TIMEOUT_S = 420
# A rank's log starts with its pid; a failed rank writes the marker after its traceback.
_PID_LINE = "HALO_R_RANK_PID "
_FAILED_MARKER = "HALO_R_RANK_FAILED"
# Log text that means the rank is failing or gone: the marker, a fatal signal, or the timeout dump.
_FAILURE_TEXT = (_FAILED_MARKER, "Fatal Python error", "\nTimeout (")
_POLL_S = 5
_LOG_TAIL_LINES = 60


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


def _skewed_logits(logits: torch.Tensor, hot_rank: int | None) -> torch.Tensor:
    """Point three quarters of the tokens at the experts of ``hot_rank``; None leaves them as drawn."""
    home = _NUM_EXPERTS // _EP_SIZE
    skewed = logits.clone()
    if hot_rank is None:
        return skewed
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


def _mismatch_by_route(
    output: torch.Tensor,
    ref_output: torch.Tensor,
    physical: torch.Tensor,
    home: int,
    slots: int,
    rtol: float = 0.1,
    atol: float = 0.15,
) -> dict[str, tuple[int, float]]:
    """Split one layer's output mismatch by where each token was routed.

    ``output`` and ``ref_output`` are (tokens, hidden); ``physical`` is the (tokens, top_k) slot of
    every route. A token is "helper" if any of its routes reaches a helper slot, else "home". For
    each class: (tokens, fraction of its elements outside atol + rtol * |ref|), the element count
    check_accuracy uses.
    """
    out, ref = output.float(), ref_output.float()
    bad = (out - ref).abs() > atol + rtol * ref.abs()
    on_helper = (physical.to(torch.int64) % slots >= home).any(dim=-1)
    split = {}
    for name, rows in (("helper", on_helper), ("home", ~on_helper)):
        count = int(rows.sum())
        split[name] = (count, round(float(bad[rows].float().mean()), 4) if count else 0.0)
    return split


def _route_errors(
    layer: Any, logical: torch.Tensor, physical: torch.Tensor, expect_helpers: bool = True
) -> list[str]:
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
    if expect_helpers and helper_routes.item() == 0:
        errors.append("no route used a helper slot")
    return errors


def _relay_counts(layer: Any) -> tuple[int, int]:
    """(launches, receiver copies + skips) of the layer's HALO-R kernel, eager or replayed."""
    torch.cuda.synchronize()
    launches, copies, skips = layer._scheduler_group.broadcaster.relay_counts()
    return launches, copies + skips


_RANK_LOG = None


def _redirect_output(log_dir: str) -> None:
    """Send this rank's stdout and stderr, C++ and NCCL output included, to its own log file.

    A rank's output otherwise reaches only the pytest process's file descriptors, which pytest
    captures and drops if that process is killed. Fatal signals and the timeout dump go there too.
    """
    path = os.path.join(log_dir, f"rank{MPI.COMM_WORLD.Get_rank()}.log")
    global _RANK_LOG
    log = _RANK_LOG = open(path, "a", buffering=1)  # kept open for the life of the rank
    log.write(f"{_PID_LINE}{os.getpid()}\n")
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    atexit.register(log.flush)
    faulthandler.enable(file=log, all_threads=True)
    faulthandler.dump_traceback_later(_WORKER_TIMEOUT_S, exit=True, file=log)


def _run_rank(unittest_root: str, log_dir: str) -> tuple[int, list[str]]:
    """Run every step on this EP rank; return (rank, errors).

    ``unittest_root`` (tests/unittest) goes on sys.path for the shared MoE test utilities; this
    rank's output goes to ``log_dir``/rank<N>.log, which the pytest process reads on a failure.
    """
    _redirect_output(log_dir)
    from tensorrt_llm._utils import mpi_rank

    if unittest_root not in sys.path:
        sys.path.append(unittest_root)

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
    except BaseException:
        # Traceback first, marker last: the pytest process reads the log once it sees the marker.
        # It then fails at once and kills the other ranks, which may wait for this one in a
        # collective. No MPI abort: the pytest process is not in these ranks' MPI world, so an
        # abort would end the ranks without ending its wait for their results.
        traceback.print_exc()
        sys.stderr.flush()
        print(_FAILED_MARKER, flush=True)
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

        def load(step: int, hot_rank: int | None, tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
            x, logits = inputs[step]
            x, logits = x[:tokens], _skewed_logits(logits[:tokens], hot_rank)
            static_x[:tokens].copy_(x)
            static_logits[:tokens].copy_(logits)
            num_tokens["value"] = tokens
            return x, logits

        def check_output(
            where: str,
            output: torch.Tensor,
            x: torch.Tensor,
            logits: torch.Tensor,
            physical: list | None = None,
        ) -> None:
            """Check each layer's output; with the eager routes, split a mismatch by route class."""
            for layer_idx, ref in enumerate(refs):
                with torch.inference_mode():
                    ref_output = ref.forward(x, logits)
                try:
                    ref.check_accuracy(output[:, layer_idx], ref_output)
                except Exception as error:  # noqa: BLE001 - check_accuracy raises a bare Exception
                    detail = ""
                    if physical is not None and physical[layer_idx] is not None:
                        arena = layers[layer_idx].live_weight_arena
                        split = _mismatch_by_route(
                            output[:, layer_idx],
                            ref_output,
                            physical[layer_idx],
                            arena.home_experts,
                            arena.slot_count,
                        )
                        detail = f"; element mismatch by route (tokens, fraction): {split}"
                    errors.append(f"rank {rank} {where} layer {layer_idx}: output: {error}{detail}")

        def synchronize() -> None:
            torch.cuda.synchronize()
            dist.barrier()

        step = 0
        eager_launches = [0] * _LAYERS

        def eager_forward(where: str, hot_rank: int | None, tokens: int) -> None:
            nonlocal step
            x, logits = load(step, hot_rank, tokens)
            step += 1
            output = run_model().clone()
            synchronize()
            routes = []
            for layer_idx in range(_LAYERS):
                calls = records[layer_idx]
                if len(calls) == 1 and not calls[0][2]:
                    routes.append(calls.pop()[:2])
                    eager_launches[layer_idx] += 1
                else:
                    errors.append(f"rank {rank} {where} layer {layer_idx}: route calls {calls}")
                    calls.clear()
                    routes.append(None)
            check_output(
                where, output, x, logits, [r[1] if r is not None else None for r in routes]
            )
            for layer_idx, layer in enumerate(layers):
                # _route_errors is collective: every rank calls it, with no routes if need be.
                empty = torch.empty(0, dtype=torch.int64, device="cuda")
                logical, physical = routes[layer_idx] or (empty, empty)
                errors.extend(
                    f"rank {rank} {where} layer {layer_idx}: {error}"
                    for error in _route_errors(
                        layer, logical, physical, expect_helpers=hot_rank is not None
                    )
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
        except Exception as error:  # noqa: BLE001
            if not isinstance(error, RuntimeError) or "restart the process" not in str(error):
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
        # MegaMoE returns a view of its combine buffer, which MoE layers of one shape share and the
        # next layer overwrites: copy each layer's output before the next layer runs. Tokens stay
        # the first dimension, which the runner's shared output is sized by.
        outputs = [moe(x, logits, all_rank_num_tokens=tokens).clone() for moe in self.moes]
        return torch.stack(outputs, dim=1)


def _rank_logs(log_dir: Path) -> dict[int, str]:
    logs = {}
    for rank in range(_EP_SIZE):
        path = log_dir / f"rank{rank}.log"
        if path.exists():
            logs[rank] = path.read_text(errors="replace")
    return logs


def _dead_ranks(logs: dict[int, str]) -> list[int]:
    """Ranks whose log says they failed, or whose process is gone."""
    dead = []
    for rank, text in logs.items():
        pids = [line[len(_PID_LINE) :] for line in text.splitlines() if line.startswith(_PID_LINE)]
        gone = False
        if pids:
            try:
                os.kill(int(pids[-1]), 0)
            except ProcessLookupError:
                gone = True
        if gone or any(marker in text for marker in _FAILURE_TEXT):
            dead.append(rank)
    return dead


def _end_ranks(logs: dict[int, str]) -> None:
    """Kill the ranks that are still alive, so that a failure frees the GPUs at once."""
    for text in logs.values():
        for line in text.splitlines():
            if line.startswith(_PID_LINE):
                try:
                    os.kill(int(line[len(_PID_LINE) :]), signal.SIGKILL)
                except OSError:
                    pass


def _stop_pool_threads() -> None:
    """End the MPIPoolExecutor manager thread after the ranks are killed.

    It polls for results that never come and is joined at interpreter exit, so without this the
    pytest process would hang on exit after a failure.
    """
    import threading

    for thread in threading.enumerate():
        target = getattr(thread, "_target", None)
        if thread.ident and getattr(target, "__name__", "").startswith("_manager"):
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(thread.ident), ctypes.py_object(SystemExit)
            )


def _wait_for_ranks(futures: list, log_dir: Path) -> list:
    """Return every rank's result; fail fast, with the ranks' log tails, if a rank fails or hangs."""
    deadline = time.monotonic() + _WORKER_TIMEOUT_S + 120
    while True:
        done, pending = wait(futures, timeout=_POLL_S, return_when=FIRST_EXCEPTION)
        if not pending and not any(future.exception() for future in done):
            return [future.result() for future in futures]
        raised = [future.exception() for future in done if future.exception() is not None]
        dead = _dead_ranks(_rank_logs(log_dir))
        if raised or dead or time.monotonic() > deadline:
            _end_ranks(_rank_logs(log_dir))
            # Read the logs again after the kill, so a traceback written meanwhile is included.
            logs = _rank_logs(log_dir)
            tails = "\n".join(
                f"--- rank {rank} ({log_dir / f'rank{rank}.log'}), last {_LOG_TAIL_LINES} lines:\n"
                + "\n".join(text.splitlines()[-_LOG_TAIL_LINES:])
                for rank, text in sorted(logs.items())
            )
            if raised:
                reason = f"a rank raised {raised[0]!r}"
            elif dead:
                reason = f"ranks {dead} failed or exited"
            else:
                reason = "the ranks did not finish in time"
            pytest.fail(f"{reason}; {len(logs)} of {_EP_SIZE} rank logs:\n{tails}", pytrace=False)


@pytest.mark.threadleak(enabled=False)  # the MPIPoolExecutor manager thread outlives the test
def test_halo_r_breakable_cuda_graph(tmp_path: Path) -> None:
    """Single-node CI runs it as one plain pytest process, which starts the four ranks itself.

    The ranks start as test_halo_r_relay.py starts them: an MPIPoolExecutor given the launch-queue
    environment. The SM and fabric checks run in the ranks; this process only counts devices.
    Each rank writes its output to its own log under ``tmp_path``; this process polls the ranks and
    fails with those logs as soon as one rank fails, exits or overruns.
    """
    if os.environ.get("TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR") or MPI.COMM_WORLD.Get_size() > 1:
        _skip("starts its own four ranks; run it without trtllm-llmapi-launch or mpirun")
    if torch.cuda.device_count() < _EP_SIZE:
        _skip(f"needs {_EP_SIZE} GPUs")
    unittest_root = str(Path(__file__).resolve().parents[3])
    # The last resort if this process itself hangs, for runs outside CI (CI's TIMEOUT ends first):
    # it outlasts the rank timers and the poll's deadline, and writes its stacks to a file, since
    # pytest drops captured output on an exit.
    parent_log = open(tmp_path / "pytest_process.log", "a", buffering=1)
    faulthandler.dump_traceback_later(_WORKER_TIMEOUT_S + 240, exit=True, file=parent_log)
    # Per-layer dynamic EPLB needs the larger launch queues from CUDA initialization on.
    executor = MPIPoolExecutor(_EP_SIZE, env={"CUDA_SCALE_LAUNCH_QUEUES": "4x"})
    finished = False
    futures = []
    try:
        futures = [
            executor.submit(_run_rank, unittest_root, str(tmp_path)) for _ in range(_EP_SIZE)
        ]
        results = _wait_for_ranks(futures, tmp_path)
        finished = True
    finally:
        if finished:
            executor.shutdown(wait=True)
            faulthandler.cancel_dump_traceback_later()
            parent_log.close()
        else:
            # Kill any rank still alive (also when the test ends some other way) and do not wait
            # for them. The timer stays armed, so a hang while this process exits still ends,
            # with its stacks in pytest_process.log; the entry runs ISOLATION so that cannot
            # end another test.
            _end_ranks(_rank_logs(tmp_path))
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False)
            _stop_pool_threads()
    skips = {error for _, errors in results for error in errors if error.startswith("skip: ")}
    if skips:
        _skip("; ".join(sorted(skips)))
    failures = [error for _, errors in sorted(results) for error in errors]
    assert not failures, "\n".join(failures[:20])


class _SharedBufferLayer(torch.nn.Module):
    """Stands for one MegaMoE layer in the CPU test: writes its input times a factor into a buffer
    it shares with the other layers and returns a view of it, as MegaMoE returns its combine buffer."""

    def __init__(self, buffer: torch.Tensor, factor: float) -> None:
        super().__init__()
        self.buffer = buffer
        self.factor = factor

    def forward(self, x: torch.Tensor, logits: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del logits, kwargs
        self.buffer[: x.shape[0]] = x * self.factor
        return self.buffer[: x.shape[0]]


@pytest.mark.cpu_only
def test_layer_outputs_and_route_split() -> None:
    """Each layer's column holds that layer's output even when the layers share an output buffer,
    and the mismatch split attributes bad elements to the route class of their token."""
    tokens, hidden = 8, 4
    x = torch.arange(tokens * hidden, dtype=torch.float32).reshape(tokens, hidden)
    shared = torch.empty(16, hidden)
    body = _TwoLayerBody([_SharedBufferLayer(shared, 1.0), _SharedBufferLayer(shared, 2.0)])
    output = body(x, torch.zeros(tokens, _NUM_EXPERTS))
    assert output.shape == (tokens, _LAYERS, hidden)
    for layer_idx, factor in enumerate((1.0, 2.0)):
        torch.testing.assert_close(output[:, layer_idx], x * factor)

    # 4 home + 2 helper slots per rank: slots 4-5 of every rank are helpers.
    home, slots = 4, 6
    physical = torch.tensor([[0, 1], [6, 7], [4, 0], [11, 12], [1, 2], [5, 9], [3, 6], [12, 17]])
    on_helper = torch.tensor([False, False, True, True, False, True, False, True])
    ref = torch.ones(tokens, hidden)
    assert _mismatch_by_route(ref.clone(), ref, physical, home, slots) == {
        "helper": (4, 0.0),
        "home": (4, 0.0),
    }
    wrong = ref.clone()
    wrong[on_helper] += 1.0
    assert _mismatch_by_route(wrong, ref, physical, home, slots) == {
        "helper": (4, 1.0),
        "home": (4, 0.0),
    }
    wrong = ref.clone()
    wrong[0, 0] = 5.0
    assert _mismatch_by_route(wrong, ref, physical, home, slots) == {
        "helper": (4, 0.0),
        "home": (4, 0.0625),
    }


def _skip(reason: str) -> None:
    if os.environ.get("HALO_R_TEST_NO_SKIP") == "1":
        pytest.fail(f"HALO_R_TEST_NO_SKIP=1 but the test would skip: {reason}")
    pytest.skip(reason)

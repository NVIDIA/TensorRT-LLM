"""Atomic scheduler-to-MegaMoE dynamic-load-balance binding.

This scheduler-side adapter owns the per-launch copy/consumer lifecycle:
it installs one live-weight generation, validates
the borrowed scheduler routes, enqueues the complete MegaMoE pipeline, and
records consumption on the bound CUDA stream.  Copy submissions, leases, and
partially prepared kernel arguments are intentionally not public API.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from inspect import Parameter, signature
from typing import Callable, Optional, TypeVar


_MAX_GENERATION = (1 << 63) - 1
_EP_ARGUMENT_NAMES = (
    "activation",
    "activation_sf",
    "topk_indices",
    "topk_scores",
    "fc1_weight",
    "fc1_weight_sf",
    "fc2_weight",
    "fc2_weight_sf",
    "output_activation",
    "local_workspace",
    "shared_workspace",
    "peer_rank_ptr_mapper_host",
    "stream",
    "fc1_alpha",
    "fc2_alpha",
    "fc1_norm_const",
    "hot_expert_weight_ready_flags",
    "hot_expert_weight_ready_generation",
)
_EACH_GENERATION = "each_generation"
_BIND_ONCE = "bind_once"
_KERNEL_WEIGHT_KEYS = (
    "fc1_weight",
    "fc1_weight_sf",
    "fc2_weight",
    "fc2_weight_sf",
    "fc1_alpha",
    "fc2_alpha",
    "fc1_norm_const",
)
_KERNEL_LAUNCH_KEYS = (
    "topk_indices",
    *_KERNEL_WEIGHT_KEYS,
    "hot_expert_weight_ready_flags",
    "hot_expert_weight_ready_generation",
)
_PIPELINE_ARGUMENT_KEYS = (
    "activation",
    "activation_sf",
    "topk_scores",
    "output_activation",
    "local_workspace",
    "shared_workspace",
    "peer_rank_ptr_mapper_host",
)
_Result = TypeVar("_Result")


def _validation_mode(value: object) -> str:
    if type(value) is not str or value not in (
        _EACH_GENERATION,
        _BIND_ONCE,
    ):
        raise ValueError("validation_mode must be 'each_generation' or 'bind_once'")
    return value


def _dtype_name(value: object) -> str:
    return str(value).lower().removeprefix("torch.")


def _launch_generation(value: object) -> int:
    """Validate the CPU epoch lowered as the by-value int64 AOT tail scalar."""

    if type(value) is not int:
        raise TypeError(
            "launch generation must be an exact by-value Python int; "
            "bool, tensor, and pointer values are forbidden."
        )
    if not 1 <= value <= _MAX_GENERATION:
        raise ValueError(f"launch generation must be in [1, {_MAX_GENERATION}].")
    return value


def _shape(value: object, *, name: str) -> tuple[int, ...]:
    try:
        return tuple(int(extent) for extent in value.shape)  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(f"{name} must expose an integer shape.") from exc


def _require_contiguous_cuda_int32(value: object, *, name: str) -> tuple[int, ...]:
    shape = _shape(value, name=name)
    if _dtype_name(getattr(value, "dtype", None)) != "int32":
        raise TypeError(f"{name} must have dtype int32.")
    is_contiguous = getattr(value, "is_contiguous", None)
    if not callable(is_contiguous) or not bool(is_contiguous()):
        raise ValueError(f"{name} must be contiguous.")
    device = getattr(value, "device", None)
    if device is None or not str(device).lower().startswith("cuda"):
        raise ValueError(f"{name} must reside on CUDA memory.")
    return shape


def _data_pointer(value: object, *, name: str) -> int:
    pointer = getattr(value, "data_ptr", None)
    if callable(pointer):
        pointer = pointer()
    if isinstance(pointer, bool) or not isinstance(pointer, int) or pointer <= 0:
        raise TypeError(f"{name} must expose a positive integer data_ptr.")
    return pointer


def _cuda_device_ordinal(value: object, *, name: str) -> int:
    device = getattr(value, "device", value)
    if device is None or not str(device).lower().startswith("cuda"):
        raise ValueError(f"{name} must expose a CUDA device.")
    index = getattr(device, "index", None)
    if type(index) is not int:
        text = str(device).lower()
        separator = text.find(":")
        if separator >= 0:
            try:
                index = int(text[separator + 1 :])
            except ValueError as exc:
                raise ValueError(f"{name} exposes an invalid CUDA device.") from exc
        else:
            try:
                import torch

                index = int(torch.cuda.current_device())
            except (ImportError, RuntimeError, ValueError) as exc:
                raise ValueError(
                    f"{name} must expose an explicit CUDA device ordinal."
                ) from exc
    if index < 0:
        raise ValueError(f"{name} exposes an invalid CUDA device ordinal.")
    return index


def _stream_handle(stream: object, *, name: str) -> int:
    handle = getattr(stream, "cuda_stream", None)
    if type(handle) is not int or handle < 0:
        raise TypeError(f"{name} must expose a non-negative cuda_stream handle.")
    return handle


def _compiled_stream(consumer_stream: object) -> object:
    """Lower the bound framework stream to the explicit CUTE runtime stream."""

    try:
        import cuda.bindings.driver as cuda
    except ImportError as exc:
        raise RuntimeError("cuda.bindings is required to launch MegaMoE") from exc
    return cuda.CUstream(_stream_handle(consumer_stream, name="consumer_stream"))


def _cutlass_version() -> tuple[int, int]:
    try:
        import cutlass
    except ImportError as exc:
        raise RuntimeError("CUTLASS DSL is required to launch MegaMoE") from exc
    fields = str(cutlass.__version__).split(".")
    try:
        return int(fields[0]), int(fields[1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError("CUTLASS DSL exposes an invalid version") from exc


def _workspace_pointer(workspace: object, *, name: str) -> int:
    pointer = workspace if type(workspace) is int else _data_pointer(workspace, name=name)
    if pointer <= 0 or pointer % 128:
        raise ValueError(f"{name} must expose a positive 128-byte-aligned pointer")
    return pointer


def _peer_mapper_argument(mapper: object) -> tuple[object, ...]:
    try:
        base_address = mapper.base_address  # type: ignore[attr-defined]
        offsets = tuple(mapper.offsets)  # type: ignore[attr-defined]
        rank = mapper.rank  # type: ignore[attr-defined]
        max_ranks = int(mapper.max_ranks)  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(
            "peer_rank_ptr_mapper_host must expose base_address, offsets, "
            "rank, and integer max_ranks"
        ) from exc
    if max_ranks <= 0 or len(offsets) != max_ranks:
        raise ValueError("peer_rank_ptr_mapper_host offsets differ from max_ranks")
    fields = (base_address, offsets, rank, max_ranks)
    if (4, 5) <= _cutlass_version() < (4, 6):
        return fields[:3]
    return fields


def _validated_pipeline_arguments(arguments: object) -> dict[str, object]:
    if not isinstance(arguments, Mapping):
        raise TypeError("pipeline_arguments must be a mapping")
    supplied = set(arguments)
    expected = set(_PIPELINE_ARGUMENT_KEYS)
    if supplied != expected:
        missing = sorted(expected - supplied)
        unexpected = sorted(supplied - expected)
        raise ValueError(
            "pipeline_arguments must contain exactly the production MegaMoE "
            f"inputs; missing={missing}, unexpected={unexpected}"
        )
    return {name: arguments[name] for name in _PIPELINE_ARGUMENT_KEYS}


def _validate_consumer_stream(consumer_stream: object, route_tensor: object) -> None:
    _stream_handle(consumer_stream, name="consumer_stream")
    if _cuda_device_ordinal(
        consumer_stream, name="consumer_stream"
    ) != _cuda_device_ordinal(route_tensor, name="physical_slot_ids"):
        raise ValueError(
            "consumer_stream and SchedulerOutputs must share one CUDA device."
        )


def _terminal_count(terminal_flags: object) -> int:
    count = getattr(terminal_flags, "count", None)
    if count is None:
        numel = getattr(terminal_flags, "numel", None)
        count = numel() if callable(numel) else None
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("terminal_flags must expose integer count or numel().")
    return count


def _positive_integer(owner: object, attribute: str) -> int:
    value = getattr(owner, attribute, None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"live_weight_bridge.{attribute} must be a positive integer.")
    return value


def _validated_scheduler_outputs(
    scheduler_outputs: object,
    *,
    helper_experts_per_rank: int,
    topk: Optional[int] = None,
    max_tokens_per_rank: Optional[int] = None,
) -> tuple[object, object, object, tuple[int, int]]:
    try:
        physical_slot_ids = scheduler_outputs.physical_slot_ids  # type: ignore[attr-defined]
        hot_expert_ids = scheduler_outputs.hot_expert_ids  # type: ignore[attr-defined]
        hot_expert_source_ranks = scheduler_outputs.hot_expert_source_ranks  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise TypeError(
            "scheduler_outputs must expose physical_slot_ids, hot_expert_ids, "
            "and hot_expert_source_ranks."
        ) from exc

    route_shape = _require_contiguous_cuda_int32(
        physical_slot_ids, name="physical_slot_ids"
    )
    if len(route_shape) != 2 or route_shape[0] <= 0 or route_shape[1] <= 0:
        raise ValueError("physical_slot_ids must have positive shape [tokens, topk].")
    if topk is not None and route_shape[1] != topk:
        raise ValueError(
            f"physical_slot_ids topk extent must remain {topk}, got {route_shape[1]}."
        )
    if max_tokens_per_rank is not None and route_shape[0] != max_tokens_per_rank:
        raise ValueError(
            "physical_slot_ids token extent changed after binding: "
            f"expected {max_tokens_per_rank}, got {route_shape[0]}."
        )
    if _data_pointer(physical_slot_ids, name="physical_slot_ids") % 16:
        raise ValueError(
            "physical_slot_ids must be 16-byte aligned for Router vector loads."
        )

    expected_hot_shape = (helper_experts_per_rank,)
    for name, value in (
        ("hot_expert_ids", hot_expert_ids),
        ("hot_expert_source_ranks", hot_expert_source_ranks),
    ):
        shape = _require_contiguous_cuda_int32(value, name=name)
        if shape != expected_hot_shape:
            raise ValueError(
                f"{name} must have shape {expected_hot_shape}, got {shape}."
            )

    devices = {
        str(physical_slot_ids.device),  # type: ignore[attr-defined]
        str(hot_expert_ids.device),  # type: ignore[attr-defined]
        str(hot_expert_source_ranks.device),  # type: ignore[attr-defined]
    }
    if len(devices) != 1:
        raise ValueError(
            "all SchedulerOutputs tensors must reside on the same CUDA device."
        )
    return (
        physical_slot_ids,
        hot_expert_ids,
        hot_expert_source_ranks,
        (route_shape[0], route_shape[1]),
    )


def _validate_bridge(
    bridge: object,
    *,
    scheduler_outputs: object,
    world_size: int,
    home_experts_per_rank: int,
    helper_experts_per_rank: int,
) -> None:
    methods = (
        "submit_and_install",
        "validate_scheduler_outputs_binding",
        "validate_live_weight_lease",
        "claim_live_weight_lease",
        "abort_live_weight_lease",
    )
    if any(not callable(getattr(bridge, name, None)) for name in methods):
        raise TypeError("live_weight_bridge lacks the atomic production contract")
    if bridge.validate_scheduler_outputs_binding(scheduler_outputs) is not True:
        raise ValueError("SamiLiveWeightBridge is not bound to these SchedulerOutputs")
    geometry = (
        ("world_size", world_size),
        ("home_count", home_experts_per_rank),
        ("helper_count", helper_experts_per_rank),
    )
    for attribute, expected in geometry:
        if getattr(bridge, attribute, None) != expected:
            raise ValueError(f"live_weight_bridge.{attribute} changed after binding")
    if getattr(bridge, "weight_planes", None) is None:
        raise ValueError("live_weight_bridge exposes no live MegaMoE weight planes")


def _validate_kernel(
    kernel: object,
    *,
    world_size: int,
    home_experts_per_rank: int,
    helper_experts_per_rank: int,
    topk: int,
    max_tokens_per_rank: int,
    hidden_size: int,
    intermediate_size: int,
) -> None:
    try:
        parameters = tuple(signature(kernel.__call__).parameters.values())
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("kernel must expose the MegaMoE EP call signature") from exc
    if tuple(parameter.name for parameter in parameters) != _EP_ARGUMENT_NAMES or any(
        parameter.kind not in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
        for parameter in parameters
    ):
        raise TypeError("kernel must expose the 18-argument MegaMoE EP call signature")
    expected = {
        "world_size": world_size,
        "home_expert_count": home_experts_per_rank,
        "helper_expert_count": helper_experts_per_rank,
        "local_expert_count": home_experts_per_rank + helper_experts_per_rank,
        "topk": topk,
        "max_tokens_per_rank": max_tokens_per_rank,
        "hidden_size": hidden_size,
        "intermediate_gateup_size": 2 * intermediate_size,
    }
    for attribute, required in expected.items():
        observed = getattr(kernel, attribute, None)
        if observed != required or (
            type(required) is int and type(observed) is not int
        ):
            raise ValueError(
                f"kernel {attribute} must be {required!r}, "
                f"got {observed!r}"
            )
    if str(getattr(kernel, "quant_kind", None)) != "nvfp4":
        raise ValueError("kernel quant_kind must be 'nvfp4'")
    index_dtype = getattr(kernel, "topk_index_dtype", None)
    if _dtype_name(getattr(index_dtype, "__name__", index_dtype)) != "int32":
        raise ValueError("kernel topk_index_dtype must be int32")


@dataclass(frozen=True)
class _PreparedLaunch:
    lease: object = field(repr=False, compare=False)
    consumer_stream: object = field(repr=False, compare=False)
    kernel_values: tuple[object, ...] = field(repr=False, compare=False)

    def record_consumed(self) -> object:
        release = getattr(self.lease, "release", None)
        if not callable(release):
            raise TypeError("attested live-weight lease lacks release()")
        return release(self.consumer_stream)


@dataclass(frozen=True)
class _BindOnceLaunchContract:
    physical_slot_ids: object = field(repr=False, compare=False)
    weight_values: tuple[object, ...] = field(repr=False, compare=False)
    terminal_flags: object = field(repr=False, compare=False)
    terminal_pointer: int


def _capture_bind_once_launch_contract(
    bridge: object,
    scheduler_routes: object,
    *,
    world_size: int,
) -> _BindOnceLaunchContract:
    weight_planes = bridge.weight_planes
    lower_weights = getattr(weight_planes, "kernel_kwargs", None)
    if not callable(lower_weights):
        raise TypeError("live_weight_bridge weight planes lack kernel_kwargs()")
    weight_kwargs = lower_weights()
    if not isinstance(weight_kwargs, dict) or tuple(weight_kwargs) != _KERNEL_WEIGHT_KEYS:
        raise ValueError("live weight planes must lower exactly seven ordered planes")

    # The scheduler tensor was already validated; only its identity is needed.
    physical_slot_ids = getattr(bridge, "physical_slot_ids", None)
    if physical_slot_ids is not scheduler_routes:
        raise ValueError("MegaMoE routes must be the bound scheduler tensor")

    terminal_flags = getattr(bridge, "terminal_flags", None)
    if _dtype_name(getattr(terminal_flags, "dtype", None)) != "uint64":
        raise TypeError("terminal_flags must have dtype uint64.")
    if _terminal_count(terminal_flags) != world_size:
        raise ValueError(
            f"terminal_flags must contain one cell per source rank ({world_size})."
        )
    terminal_pointer = _data_pointer(terminal_flags, name="terminal_flags")
    if terminal_pointer % 8:
        raise ValueError("terminal_flags must be 8-byte aligned.")
    terminal_device = getattr(terminal_flags, "device", None)
    if terminal_device is not None and _cuda_device_ordinal(
        terminal_flags, name="terminal_flags"
    ) != _cuda_device_ordinal(physical_slot_ids, name="physical_slot_ids"):
        raise ValueError("terminal flags and scheduler routes devices differ.")
    return _BindOnceLaunchContract(
        physical_slot_ids,
        tuple(weight_kwargs.values()),
        terminal_flags,
        terminal_pointer,
    )


def _prepare_launch(
    scheduler_outputs: object,
    lease: object,
    *,
    bridge: object,
    consumer_stream: object,
    world_size: int,
    home_experts_per_rank: int,
    helper_experts_per_rank: int,
    topk: int,
    max_tokens_per_rank: int,
    expected_scheduler_outputs: object,
    validation_mode: str = _EACH_GENERATION,
    bound_contract: Optional[_BindOnceLaunchContract] = None,
) -> _PreparedLaunch:
    if scheduler_outputs is not expected_scheduler_outputs:
        raise ValueError(
            "scheduler_outputs is not the object bound during initialization."
        )
    if validation_mode == _BIND_ONCE:
        if bound_contract is None:
            raise RuntimeError("bind-once launch contract was not initialized")
        claimed = bridge.claim_live_weight_lease(lease, consumer_stream)
        if claimed is not lease:
            raise RuntimeError("live-weight claimant must return the exact lease object")
        launch_generation = _launch_generation(
            getattr(claimed, "generation", None)
        )
        kernel_values = (
            bound_contract.physical_slot_ids,
            *bound_contract.weight_values,
            bound_contract.terminal_pointer,
            launch_generation,
        )
        return _PreparedLaunch(claimed, consumer_stream, kernel_values)

    scheduler_routes, _, _, _ = _validated_scheduler_outputs(
        scheduler_outputs,
        helper_experts_per_rank=helper_experts_per_rank,
        topk=topk,
        max_tokens_per_rank=max_tokens_per_rank,
    )
    _validate_consumer_stream(consumer_stream, scheduler_routes)
    _validate_bridge(
        bridge,
        scheduler_outputs=scheduler_outputs,
        world_size=world_size,
        home_experts_per_rank=home_experts_per_rank,
        helper_experts_per_rank=helper_experts_per_rank,
    )

    ready = bridge.validate_live_weight_lease(lease)
    if ready is not lease:
        raise RuntimeError("live-weight attestor must return the exact lease object")
    weight_planes = bridge.weight_planes
    if getattr(ready, "weight_planes", None) is not weight_planes:
        raise ValueError("live-weight lease does not own the bridge's weight planes")

    lower_weights = getattr(ready, "kernel_weight_kwargs", None)
    if not callable(lower_weights):
        raise TypeError("live-weight lease lacks kernel_weight_kwargs()")
    weight_kwargs = lower_weights()
    if not isinstance(weight_kwargs, dict) or tuple(weight_kwargs) != _KERNEL_WEIGHT_KEYS:
        raise ValueError("live-weight lease must lower exactly seven ordered planes")

    # The scheduler tensor was already validated; only its identity is needed.
    physical_slot_ids = getattr(ready, "physical_slot_ids", None)
    if physical_slot_ids is not scheduler_routes:
        raise ValueError("MegaMoE routes must be the bound scheduler tensor")

    launch_generation = _launch_generation(getattr(ready, "generation", None))
    if getattr(ready, "remote_visibility_guaranteed", None) is not True:
        raise ValueError(
            "live-weight lease must guarantee system-scope remote visibility."
        )

    terminal_flags = getattr(ready, "terminal_flags", None)
    if _dtype_name(getattr(terminal_flags, "dtype", None)) != "uint64":
        raise TypeError("terminal_flags must have dtype uint64.")
    if _terminal_count(terminal_flags) != world_size:
        raise ValueError(
            f"terminal_flags must contain one cell per source rank ({world_size})."
        )
    terminal_pointer = _data_pointer(terminal_flags, name="terminal_flags")
    if terminal_pointer % 8:
        raise ValueError("terminal_flags must be 8-byte aligned.")
    terminal_device = getattr(terminal_flags, "device", None)
    if terminal_device is not None and _cuda_device_ordinal(
        terminal_flags, name="terminal_flags"
    ) != _cuda_device_ordinal(physical_slot_ids, name="physical_slot_ids"):
        raise ValueError("terminal flags and scheduler routes devices differ.")

    claimed = bridge.claim_live_weight_lease(ready, consumer_stream)
    if claimed is not ready:
        raise RuntimeError("live-weight claimant must return the exact lease object")
    kernel_values = (
        physical_slot_ids,
        *(weight_kwargs[name] for name in _KERNEL_WEIGHT_KEYS),
        terminal_pointer,
        launch_generation,
    )
    return _PreparedLaunch(ready, consumer_stream, kernel_values)


class DynamicLoadBalanceBinding:
    """One production binding for scheduler, SAMI, and a MegaMoE stream.

    Geometry and the reusable ``SchedulerOutputs`` identity are derived from
    ``live_weight_bridge`` once.  :meth:`launch` is the only per-iteration API;
    it encloses the direct router/body/optional-TopKReduce AOT invocation in the
    live-bank lease and poisons the bridge on any partial failure. The caller
    supplies the kernel used to compile or load the pipeline so its EP
    signature and specialization can be checked without annotating the callable.
    """

    __slots__ = (
        "_bridge",
        "_consumer_stream",
        "_compiled_pipeline",
        "_scheduler_outputs",
        "_world_size",
        "_home_experts_per_rank",
        "_helper_experts_per_rank",
        "_topk",
        "_max_tokens_per_rank",
        "_validation_mode",
        "_bound_launch_contract",
    )

    def __init__(
        self,
        live_weight_bridge: object,
        consumer_stream: object,
        compiled_pipeline: Callable[..., _Result],
        *,
        kernel: object,
        validation_mode: str = _EACH_GENERATION,
    ) -> None:
        validation = _validation_mode(validation_mode)
        if live_weight_bridge is None:
            raise TypeError("live_weight_bridge is required")
        if not callable(compiled_pipeline):
            raise TypeError("compiled_pipeline must be callable")
        scheduler_outputs = getattr(
            live_weight_bridge, "bound_scheduler_outputs", None
        )
        if scheduler_outputs is None:
            raise TypeError("live_weight_bridge exposes no bound SchedulerOutputs")
        bridge_validation = getattr(
            live_weight_bridge, "validation_mode", _EACH_GENERATION
        )
        if bridge_validation != validation:
            raise ValueError("binding and live-weight bridge validation modes differ")
        world_size = _positive_integer(live_weight_bridge, "world_size")
        home_count = _positive_integer(live_weight_bridge, "home_count")
        helper_count = _positive_integer(live_weight_bridge, "helper_count")
        hidden_size = _positive_integer(live_weight_bridge, "hidden_size")
        intermediate_size = _positive_integer(
            live_weight_bridge, "intermediate_size"
        )
        routes, _, _, route_shape = _validated_scheduler_outputs(
            scheduler_outputs, helper_experts_per_rank=helper_count
        )
        _validate_consumer_stream(consumer_stream, routes)
        _validate_bridge(
            live_weight_bridge,
            scheduler_outputs=scheduler_outputs,
            world_size=world_size,
            home_experts_per_rank=home_count,
            helper_experts_per_rank=helper_count,
        )
        self._bridge = live_weight_bridge
        self._consumer_stream = consumer_stream
        self._compiled_pipeline = compiled_pipeline
        self._scheduler_outputs = scheduler_outputs
        self._world_size = world_size
        self._home_experts_per_rank = home_count
        self._helper_experts_per_rank = helper_count
        self._max_tokens_per_rank, self._topk = route_shape
        self._validation_mode = validation
        self._bound_launch_contract = (
            _capture_bind_once_launch_contract(
                live_weight_bridge,
                routes,
                world_size=world_size,
            )
            if validation == _BIND_ONCE
            else None
        )
        _validate_kernel(
            kernel,
            world_size=world_size,
            home_experts_per_rank=home_count,
            helper_experts_per_rank=helper_count,
            topk=self._topk,
            max_tokens_per_rank=self._max_tokens_per_rank,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    @property
    def validation_mode(self) -> str:
        return self._validation_mode

    def launch(
        self,
        scheduler_outputs: object,
        pipeline_arguments: Mapping[str, object],
    ) -> _Result:
        """Install one generation and enqueue router/body/reduce exactly once.

        The caller supplies only the seven non-DLB pipeline operands.  This
        binding injects the bridge-owned ten-key DLB ABI and the exact consumer
        stream before invoking the compiled router/body/reduce entry directly.
        """

        if scheduler_outputs is not self._scheduler_outputs:
            raise ValueError(
                "scheduler_outputs is not the object bound during initialization."
            )
        if self.validation_mode == _BIND_ONCE:
            if not isinstance(pipeline_arguments, Mapping):
                raise TypeError("pipeline_arguments must be a mapping")
            try:
                pipeline_operands = (
                    pipeline_arguments["activation"],
                    pipeline_arguments["activation_sf"],
                    pipeline_arguments["topk_scores"],
                    pipeline_arguments["output_activation"],
                    _workspace_pointer(
                        pipeline_arguments["local_workspace"],
                        name="local_workspace",
                    ),
                    _workspace_pointer(
                        pipeline_arguments["shared_workspace"],
                        name="shared_workspace",
                    ),
                    _peer_mapper_argument(
                        pipeline_arguments["peer_rank_ptr_mapper_host"]
                    ),
                    _compiled_stream(self._consumer_stream),
                )
            except KeyError as exc:
                raise ValueError(
                    f"bind_once pipeline_arguments miss {exc.args[0]!r}"
                ) from exc
        else:
            call_arguments = _validated_pipeline_arguments(pipeline_arguments)
            pipeline_operands = None

        lease = None
        try:
            lease = self._bridge.submit_and_install(scheduler_outputs)
            prepared = _prepare_launch(
                scheduler_outputs,
                lease,
                bridge=self._bridge,
                consumer_stream=self._consumer_stream,
                world_size=self._world_size,
                home_experts_per_rank=self._home_experts_per_rank,
                helper_experts_per_rank=self._helper_experts_per_rank,
                topk=self._topk,
                max_tokens_per_rank=self._max_tokens_per_rank,
                expected_scheduler_outputs=self._scheduler_outputs,
                validation_mode=self.validation_mode,
                bound_contract=self._bound_launch_contract,
            )
            if pipeline_operands is None:
                pipeline_operands = (
                    call_arguments["activation"],
                    call_arguments["activation_sf"],
                    call_arguments["topk_scores"],
                    call_arguments["output_activation"],
                    _workspace_pointer(
                        call_arguments["local_workspace"], name="local_workspace"
                    ),
                    _workspace_pointer(
                        call_arguments["shared_workspace"], name="shared_workspace"
                    ),
                    _peer_mapper_argument(
                        call_arguments["peer_rank_ptr_mapper_host"]
                    ),
                    _compiled_stream(self._consumer_stream),
                )
            (
                activation,
                activation_sf,
                topk_scores,
                output_activation,
                local_workspace,
                shared_workspace,
                peer_mapper,
                compiled_stream,
            ) = pipeline_operands
            (
                topk_indices,
                fc1_weight,
                fc1_weight_sf,
                fc2_weight,
                fc2_weight_sf,
                fc1_alpha,
                fc2_alpha,
                fc1_norm_const,
                terminal_pointer,
                generation,
            ) = prepared.kernel_values
            positional_arguments = (
                activation,
                activation_sf,
                topk_indices,
                topk_scores,
                fc1_weight,
                fc1_weight_sf,
                fc2_weight,
                fc2_weight_sf,
                output_activation,
                local_workspace,
                shared_workspace,
                peer_mapper,
                compiled_stream,
                fc1_alpha,
                fc2_alpha,
                fc1_norm_const,
                terminal_pointer,
                generation,
            )
            result = self._compiled_pipeline(*positional_arguments)
            prepared.record_consumed()
            return result
        except BaseException:
            self._bridge.abort_live_weight_lease(lease)
            raise


__all__ = ["DynamicLoadBalanceBinding"]

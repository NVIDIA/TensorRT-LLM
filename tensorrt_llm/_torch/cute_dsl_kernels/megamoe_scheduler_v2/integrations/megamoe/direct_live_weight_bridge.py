"""Direct SAMI live-arena binding for production MegaMoE.

SAMI writes the seven final MegaMoE helper planes in place.  This module owns
only route borrowing and the single-live-bank consumption lease: there is no
raw receive pool, parity selection, weight D2D unpack, or second READY array.
The bridge captures a CPU submission epoch from the bound copy before submit.
The copy submission contributes only the persistent terminal tensor and enqueue
metadata; the lease carries the captured epoch to MegaMoE as a launch scalar.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Mapping, Optional


CANONICAL_WEIGHT_PLANE_NAMES = (
    "mega_fc1_weight",
    "mega_fc1_weight_sf",
    "mega_fc2_weight",
    "mega_fc2_weight_sf",
    "fc31_alpha",
    "fc2_alpha",
    "fc1_norm_const",
)

TERMINAL_ERROR_BIT = 1 << 63
TERMINAL_GENERATION_MASK = TERMINAL_ERROR_BIT - 1

COMPLETION_CHECKED = "completion_checked"
STREAM_ORDERED = "stream_ordered"
EACH_GENERATION = "each_generation"
BIND_ONCE = "bind_once"
_REUSE_MODES = (COMPLETION_CHECKED, STREAM_ORDERED)
_VALIDATION_MODES = (EACH_GENERATION, BIND_ONCE)


def _reuse_mode(value: object) -> str:
    if type(value) is not str or value not in _REUSE_MODES:
        raise ValueError(
            "reuse_mode must be 'completion_checked' or 'stream_ordered'"
        )
    return value


def _validation_mode(value: object) -> str:
    if type(value) is not str or value not in _VALIDATION_MODES:
        raise ValueError("validation_mode must be 'each_generation' or 'bind_once'")
    return value


def _exact_positive_int(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive exact int")
    return value


def _copy_next_generation(copy_module: object) -> int:
    """Read the bound copy's exact CPU-side submission epoch."""

    generation = getattr(copy_module, "next_generation", None)
    if type(generation) is not int or not (
        1 <= generation <= TERMINAL_GENERATION_MASK + 1
    ):
        raise RuntimeError(
            "bound copy next_generation must be an exact CPU int in [1, 2**63]"
        )
    return generation


def _pointer(value: object, *, name: str) -> int:
    pointer = getattr(value, "data_ptr", None)
    pointer = pointer() if callable(pointer) else pointer
    if type(pointer) is not int or pointer <= 0:
        raise TypeError(f"{name} must expose a positive integer data_ptr")
    return pointer


def _shape(value: object, *, name: str) -> tuple[int, ...]:
    try:
        return tuple(int(extent) for extent in value.shape)  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(f"{name} must expose an integer shape") from exc


def _stride(value: object, *, name: str) -> tuple[int, ...]:
    stride = getattr(value, "stride", None)
    stride = stride() if callable(stride) else stride
    try:
        return tuple(int(step) for step in stride)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must expose an integer stride") from exc


def _dtype_name(value: object) -> str:
    return str(value).lower().removeprefix("torch.")


def _canonical_cuda_device(device: object, *, name: str) -> str:
    if device is None or not str(device).lower().startswith("cuda"):
        raise ValueError(f"{name} must reside on CUDA memory")
    index = getattr(device, "index", None)
    if type(index) is not int:
        text = str(device).lower()
        if ":" in text:
            try:
                index = int(text.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"{name} exposes an invalid CUDA device") from exc
        else:
            try:
                import torch

                index = int(torch.cuda.current_device())
            except (ImportError, RuntimeError, ValueError) as exc:
                raise ValueError(
                    f"{name} must expose an explicit CUDA device ordinal"
                ) from exc
    if index < 0:
        raise ValueError(f"{name} exposes an invalid CUDA device ordinal")
    return f"cuda:{index}"


def _device_name(value: object, *, name: str) -> str:
    result = _canonical_cuda_device(getattr(value, "device", None), name=name)
    if getattr(value, "is_cuda", True) is not True:
        raise ValueError(f"{name} must reside on CUDA memory")
    return result


def _stream_handle(stream: object) -> int:
    if type(stream) is int:
        if stream < 0:
            raise ValueError("CUDA stream handle cannot be negative")
        return stream
    handle = getattr(stream, "cuda_stream", None)
    if type(handle) is not int or handle < 0:
        raise TypeError("CUDA stream must expose a non-negative cuda_stream handle")
    return handle


def _stream_device_name(stream: object, *, name: str) -> str:
    return _canonical_cuda_device(getattr(stream, "device", None), name=name)


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class _PlaneSpec:
    name: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    expert_nbytes: int
    pointer_alignment: int


def _nvfp4_plane_specs(
    slot_count: int,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[_PlaneSpec, ...]:
    gate_up = 2 * intermediate_size
    fc1_sf_columns = _align_up(gate_up, 128) * _align_up(
        (hidden_size + 15) // 16, 4
    )
    fc2_sf_columns = _align_up(hidden_size, 128) * _align_up(
        (intermediate_size + 15) // 16, 4
    )
    return (
        _PlaneSpec(
            "mega_fc1_weight",
            (slot_count, hidden_size // 2, gate_up),
            (hidden_size * gate_up // 2, 1, hidden_size // 2),
            "float4_e2m1fn_x2",
            hidden_size * gate_up // 2,
            16,
        ),
        _PlaneSpec(
            "mega_fc1_weight_sf",
            (slot_count, fc1_sf_columns),
            (fc1_sf_columns, 1),
            "float8_e4m3fn",
            fc1_sf_columns,
            16,
        ),
        _PlaneSpec(
            "mega_fc2_weight",
            (slot_count, intermediate_size // 2, hidden_size),
            (intermediate_size * hidden_size // 2, 1, intermediate_size // 2),
            "float4_e2m1fn_x2",
            intermediate_size * hidden_size // 2,
            16,
        ),
        _PlaneSpec(
            "mega_fc2_weight_sf",
            (slot_count, fc2_sf_columns),
            (fc2_sf_columns, 1),
            "float8_e4m3fn",
            fc2_sf_columns,
            16,
        ),
        _PlaneSpec("fc31_alpha", (slot_count,), (1,), "float32", 4, 4),
        _PlaneSpec("fc2_alpha", (slot_count,), (1,), "float32", 4, 4),
        _PlaneSpec("fc1_norm_const", (slot_count,), (1,), "float32", 4, 4),
    )


@dataclass(frozen=True)
class _ValidatedPlane:
    name: str
    tensor: object = field(repr=False, compare=False)
    base_pointer: int
    expert_nbytes: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class _ValidatedWeightPlanes:
    planes: tuple[_ValidatedPlane, ...]
    device: str
    slot_count: int


@dataclass(frozen=True)
class MegaMoeWeightPlanes:
    """The seven allocator-owned NVFP4 tensors consumed by MegaMoE."""

    mega_fc1_weight: object = field(repr=False, compare=False)
    mega_fc1_weight_sf: object = field(repr=False, compare=False)
    mega_fc2_weight: object = field(repr=False, compare=False)
    mega_fc2_weight_sf: object = field(repr=False, compare=False)
    fc31_alpha: object = field(repr=False, compare=False)
    fc2_alpha: object = field(repr=False, compare=False)
    fc1_norm_const: object = field(repr=False, compare=False)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "MegaMoeWeightPlanes":
        missing = tuple(name for name in CANONICAL_WEIGHT_PLANE_NAMES if name not in values)
        extras = tuple(name for name in values if name not in CANONICAL_WEIGHT_PLANE_NAMES)
        if missing or extras:
            raise ValueError(
                f"weight-plane keys mismatch: missing={missing}, extras={extras}"
            )
        return cls(**{name: values[name] for name in CANONICAL_WEIGHT_PLANE_NAMES})

    def items(self) -> tuple[tuple[str, object], ...]:
        return tuple((name, getattr(self, name)) for name in CANONICAL_WEIGHT_PLANE_NAMES)

    def kernel_kwargs(self) -> dict[str, object]:
        return {
            "fc1_weight": self.mega_fc1_weight,
            "fc1_weight_sf": self.mega_fc1_weight_sf,
            "fc2_weight": self.mega_fc2_weight,
            "fc2_weight_sf": self.mega_fc2_weight_sf,
            "fc1_alpha": self.fc31_alpha,
            "fc2_alpha": self.fc2_alpha,
            "fc1_norm_const": self.fc1_norm_const,
        }

    def validate(
        self,
        *,
        home_experts_per_rank: int,
        helper_experts_per_rank: int,
        hidden_size: int,
        intermediate_size: int,
        bundle_planes: object,
    ) -> _ValidatedWeightPlanes:
        home = _exact_positive_int("home_experts_per_rank", home_experts_per_rank)
        helper = _exact_positive_int("helper_experts_per_rank", helper_experts_per_rank)
        hidden = _exact_positive_int("hidden_size", hidden_size)
        intermediate = _exact_positive_int("intermediate_size", intermediate_size)
        if hidden % 128 or intermediate % 64:
            raise ValueError(
                "SAMI NVFP4 layout requires hidden % 128 == 0 and intermediate % 64 == 0"
            )
        bundle = tuple(bundle_planes)
        if tuple(getattr(plane, "name", None) for plane in bundle) != CANONICAL_WEIGHT_PLANE_NAMES:
            raise ValueError("SAMI bundle must contain the seven canonical planes in order")

        specs = _nvfp4_plane_specs(home + helper, hidden, intermediate)
        tensors = dict(self.items())
        validated: list[_ValidatedPlane] = []
        devices: set[str] = set()
        for spec, layout in zip(specs, bundle):
            tensor = tensors[spec.name]
            shape = _shape(tensor, name=spec.name)
            stride = _stride(tensor, name=spec.name)
            dtype = _dtype_name(getattr(tensor, "dtype", None))
            if shape != spec.shape:
                raise ValueError(f"{spec.name} must have shape {spec.shape}, got {shape}")
            if stride != spec.stride:
                raise ValueError(f"{spec.name} must have stride {spec.stride}, got {stride}")
            if dtype != spec.dtype:
                raise TypeError(f"{spec.name} must have dtype {spec.dtype}, got {dtype}")
            layout_name = _dtype_name(getattr(tensor, "layout", "strided"))
            if layout_name != "strided":
                raise ValueError(f"{spec.name} must use strided storage")
            devices.add(_device_name(tensor, name=spec.name))
            pointer = _pointer(tensor, name=spec.name)
            if pointer % spec.pointer_alignment:
                raise ValueError(
                    f"{spec.name} base pointer must be {spec.pointer_alignment}-byte aligned"
                )
            observed_nbytes = getattr(layout, "nbytes", None)
            if type(observed_nbytes) is not int or observed_nbytes != spec.expert_nbytes:
                raise ValueError(
                    f"SAMI {spec.name} bytes/expert must be {spec.expert_nbytes}, "
                    f"got {observed_nbytes}"
                )
            validated.append(
                _ValidatedPlane(
                    spec.name,
                    tensor,
                    pointer,
                    spec.expert_nbytes,
                    shape,
                    stride,
                    dtype,
                )
            )
        if len(devices) != 1:
            raise ValueError("all seven live weight planes must share one CUDA device")
        ranges = sorted(
            (
                plane.base_pointer,
                plane.base_pointer + (home + helper) * plane.expert_nbytes,
                plane.name,
            )
            for plane in validated
        )
        for (_, previous_end, previous), (begin, _, current) in zip(ranges, ranges[1:]):
            if begin < previous_end:
                raise ValueError(f"live weight planes overlap: {previous} and {current}")
        return _ValidatedWeightPlanes(tuple(validated), devices.pop(), home + helper)


@dataclass(frozen=True)
class _TensorStorage:
    tensor: object = field(repr=False, compare=False)
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str
    pointer: int
    nbytes: int


def _capture_int32_tensor(tensor: object, *, name: str, dimensions: int) -> _TensorStorage:
    shape = _shape(tensor, name=name)
    stride = _stride(tensor, name=name)
    if len(shape) != dimensions or any(extent <= 0 for extent in shape):
        raise ValueError(f"{name} must have a positive {dimensions}D shape")
    running = 1
    expected_reversed = []
    for extent in reversed(shape):
        expected_reversed.append(running)
        running *= extent
    if stride != tuple(reversed(expected_reversed)):
        raise ValueError(f"{name} must be row-major contiguous")
    dtype = _dtype_name(getattr(tensor, "dtype", None))
    if dtype != "int32":
        raise TypeError(f"{name} must have dtype int32")
    pointer = _pointer(tensor, name=name)
    if pointer % (16 if dimensions == 2 else 4):
        raise ValueError(f"{name} has insufficient pointer alignment")
    return _TensorStorage(
        tensor=tensor,
        shape=shape,
        stride=stride,
        dtype=dtype,
        device=_device_name(tensor, name=name),
        pointer=pointer,
        nbytes=running * 4,
    )


@dataclass(frozen=True)
class _ProviderConsumerLease:
    cookie: object = field(repr=False, compare=False)
    generation: int
    copy_ticket: object = field(repr=False, compare=False)


class TorchDistributedLiveBankLeaseProvider:
    """Single-bank reuse authority with selectable proof and validation modes."""

    collective_safe = True
    producer_reuse_guarded = True
    live_bank_count = 1

    def __init__(
        self,
        copy_module: object,
        *,
        process_group: Optional[object] = None,
        collective_timeout_seconds: float = 30.0,
        distributed_module: Optional[object] = None,
        torch_module: Optional[object] = None,
        reuse_mode: str = COMPLETION_CHECKED,
        validation_mode: str = EACH_GENERATION,
    ) -> None:
        reuse = _reuse_mode(reuse_mode)
        validation = _validation_mode(validation_mode)
        if isinstance(collective_timeout_seconds, bool) or not isinstance(
            collective_timeout_seconds, (int, float)
        ) or collective_timeout_seconds <= 0:
            raise ValueError("collective_timeout_seconds must be positive")
        submit = getattr(copy_module, "submit", None)
        backend = getattr(copy_module, "backend", None)
        broadcaster = getattr(backend, "broadcaster", None)
        if not callable(submit) or broadcaster is None:
            raise TypeError("copy_module must be a SAMI-backed InSwitchWeightCopy")
        if not callable(getattr(copy_module, "mark_collective_reuse_safe", None)):
            raise TypeError("copy module lacks mark_collective_reuse_safe()")
        if getattr(backend, "remote_visibility_release", None) is not True:
            raise RuntimeError("bound copy backend lacks remote visibility release")
        if _copy_next_generation(copy_module) != 1:
            raise RuntimeError("live-bank provider requires a fresh copy module")
        outputs = getattr(copy_module, "_bound_outputs", None)
        scheduler = getattr(copy_module, "_bound_scheduler", None)
        if outputs is None or scheduler is None or getattr(scheduler, "outputs", None) is not outputs:
            raise RuntimeError("copy module must be bound to one scheduler first")
        if getattr(backend, "_bound_planner", None) is not scheduler:
            raise RuntimeError("copy module and SAMI backend scheduler identities differ")
        if getattr(broadcaster, "live_bank_count", None) != 1:
            raise RuntimeError("SAMI broadcaster must expose one direct live bank")
        producer_stream = getattr(copy_module, "scheduler_stream", None)
        producer_stream_handle = _stream_handle(getattr(backend, "stream", None))
        if _stream_handle(producer_stream) != producer_stream_handle:
            raise RuntimeError("scheduler and SAMI producer streams differ")
        if reuse == STREAM_ORDERED and type(producer_stream) is int:
            raise TypeError(
                "stream_ordered reuse requires the exact framework scheduler stream object"
            )
        producer_device = f"cuda:{int(broadcaster.device)}"

        contracts = tuple(
            (
                name,
                dimensions,
                _capture_int32_tensor(
                    getattr(outputs, name),
                    name=f"SchedulerOutputs.{name}",
                    dimensions=dimensions,
                ),
            )
            for name, dimensions in (
                ("physical_slot_ids", 2),
                ("hot_expert_ids", 1),
                ("hot_expert_source_ranks", 1),
            )
        )
        if any(contract.device != producer_device for _, _, contract in contracts):
            raise RuntimeError("SchedulerOutputs device differs from SAMI")
        helper_count = int(broadcaster.helper_count)
        if any(
            contract.shape != (helper_count,)
            for _, dimensions, contract in contracts
            if dimensions == 1
        ):
            raise ValueError("SchedulerOutputs hot tensors differ from SAMI geometry")

        if reuse == COMPLETION_CHECKED:
            if distributed_module is None:
                try:
                    import torch.distributed as distributed_module
                except ImportError as exc:
                    raise RuntimeError(
                        "torch.distributed is required for live-bank leases"
                    ) from exc
            if torch_module is None:
                try:
                    import torch as torch_module
                except ImportError as exc:
                    raise RuntimeError("torch is required for live-bank leases") from exc
            is_initialized = getattr(distributed_module, "is_initialized", None)
            if not callable(is_initialized) or is_initialized() is not True:
                raise RuntimeError("torch.distributed must be initialized first")
            world = int(distributed_module.get_world_size(group=process_group))
            rank = int(distributed_module.get_rank(group=process_group))
            if world != int(broadcaster.comm.world) or rank != int(broadcaster.comm.rank):
                raise ValueError("process-group geometry differs from SAMI broadcaster")

        self.bound_copy_module = copy_module
        self.bound_scheduler_outputs = outputs
        self.broadcaster = broadcaster
        self._reuse_mode = reuse
        self._validation_mode = validation
        self.producer_stream = producer_stream
        self.producer_stream_handle = producer_stream_handle
        self.producer_stream_device = producer_device
        self.process_group = process_group
        self.collective_timeout_seconds = float(collective_timeout_seconds)
        self._scheduler_contracts = contracts
        self._dist = distributed_module
        self._torch = torch_module
        self._device_index = int(broadcaster.device)
        self._lock = threading.Lock()
        self._cookie = object()
        self._poisoned = False
        self._phase = "idle"
        self._active_ticket: object | None = None
        self._active_generation: int | None = None
        self._released_generation = 0
        self._collective_safe_generation = 0
        self._completion_event: object | None = None
        self._bridge_owner: object | None = None
        bind = getattr(copy_module, "bind_generation_reuse_authority", None)
        if not callable(bind):
            raise TypeError("copy module lacks bind_generation_reuse_authority()")
        bind(self)

    @property
    def reuse_mode(self) -> str:
        return self._reuse_mode

    @property
    def validation_mode(self) -> str:
        return self._validation_mode

    @property
    def next_generation(self) -> int:
        return _copy_next_generation(self.bound_copy_module)

    @property
    def released_generation(self) -> int:
        return self._released_generation

    @property
    def collective_safe_generation(self) -> int:
        return self._collective_safe_generation

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def _require_healthy(self) -> None:
        if self._poisoned:
            raise RuntimeError("collective live-bank provider is poisoned; rebuild the group")

    def _poison(self) -> None:
        with self._lock:
            self._poisoned = True
            self._phase = "poisoned"

    def _validate_scheduler_storage(self) -> None:
        for name, dimensions, expected in self._scheduler_contracts:
            tensor = getattr(self.bound_scheduler_outputs, name)
            observed = _capture_int32_tensor(
                tensor, name=f"SchedulerOutputs.{name}", dimensions=dimensions
            )
            if tensor is not expected.tensor or observed != expected:
                raise ValueError(f"SchedulerOutputs.{name} storage changed")

    def validate_binding(self, copy_module: object, broadcaster: object) -> bool:
        with self._lock:
            try:
                generation_is_fresh = self.next_generation == 1
            except RuntimeError:
                return False
            return (
                not self._poisoned
                and self._phase == "idle"
                and generation_is_fresh
                and self._bridge_owner is None
                and copy_module is self.bound_copy_module
                and broadcaster is self.broadcaster
            )

    def bind_bridge_owner(
        self, bridge: object, copy_module: object, broadcaster: object
    ) -> None:
        with self._lock:
            self._require_healthy()
            if self._bridge_owner is not None:
                raise RuntimeError("live-bank provider bridge binding is single-assignment")
            try:
                generation_is_fresh = self.next_generation == 1
            except RuntimeError:
                self._poisoned = True
                self._phase = "poisoned"
                raise
            if (
                self._phase != "idle"
                or not generation_is_fresh
                or copy_module is not self.bound_copy_module
                or broadcaster is not self.broadcaster
            ):
                self._poisoned = True
                self._phase = "poisoned"
                raise RuntimeError("live-bank provider bridge binding is invalid")
            self._bridge_owner = bridge

    def validate_scheduler_outputs_binding(self, scheduler_outputs: object) -> bool:
        if self._poisoned or scheduler_outputs is not self.bound_scheduler_outputs:
            return False
        if self.validation_mode == BIND_ONCE:
            return True
        try:
            self._validate_scheduler_storage()
        except (AttributeError, TypeError, ValueError):
            return False
        return True

    def validate_copy_stream(self, copy_stream: object) -> bool:
        if self._poisoned:
            return False
        try:
            scheduler_stream = getattr(self.bound_copy_module, "scheduler_stream", None)
            backend = getattr(self.bound_copy_module, "backend", None)
            matches = (
                _stream_handle(copy_stream) == self.producer_stream_handle
                and (
                    type(copy_stream) is int
                    or _stream_device_name(copy_stream, name="copy_stream")
                    == self.producer_stream_device
                )
                and _stream_handle(scheduler_stream)
                == self.producer_stream_handle
                and _stream_handle(getattr(backend, "stream", None))
                == self.producer_stream_handle
            )
            return matches and (
                self.reuse_mode != STREAM_ORDERED
                or (
                    copy_stream is self.producer_stream
                    and scheduler_stream is self.producer_stream
                )
            )
        except (TypeError, ValueError):
            return False

    def _bounded_barrier(self) -> None:
        assert self.reuse_mode == COMPLETION_CHECKED
        assert self._dist is not None
        work = self._dist.barrier(
            group=self.process_group,
            async_op=True,
            device_ids=[self._device_index],
        )
        wait = getattr(work, "wait", None)
        if not callable(wait):
            raise RuntimeError("asynchronous distributed barrier returned no Work")
        completed = wait(timeout=timedelta(seconds=self.collective_timeout_seconds))
        if completed is False:
            raise TimeoutError("collective live-bank barrier timed out")

    def submit_bound_copy(self, scheduler_outputs: object) -> object:
        """Submit after the bound scheduler has been launched on its producer stream.

        Stream-ordered reuse requires that scheduler to exchange this generation
        across every EP rank before publishing its copy plan. Its stream already
        waits on the previous consumer event. The collective mark records this
        ordering contract; it does not enqueue a barrier or synchronize the CPU.
        """
        if scheduler_outputs is not self.bound_scheduler_outputs:
            raise RuntimeError("SchedulerOutputs are not provider-bound")
        if self.validation_mode == EACH_GENERATION:
            try:
                self._validate_scheduler_storage()
            except BaseException:
                self._poison()
                raise
        with self._lock:
            self._require_healthy()
            if self._phase != "idle":
                raise RuntimeError(f"live-bank provider phase is {self._phase}, expected idle")
            try:
                generation = self.next_generation
                if generation > TERMINAL_GENERATION_MASK:
                    raise RuntimeError(
                        "bound copy launch generation space is exhausted"
                    )
            except RuntimeError:
                self._poisoned = True
                self._phase = "poisoned"
                raise
            previous_event = self._completion_event
            previous_generation = generation - 1
            if generation > 1 and (
                self._released_generation != previous_generation
                or previous_event is None
            ):
                self._poisoned = True
                self._phase = "poisoned"
                raise RuntimeError("previous live generation has not been released")
            if self.reuse_mode == STREAM_ORDERED and self._bridge_owner is None:
                self._poisoned = True
                self._phase = "poisoned"
                raise RuntimeError("live-bank provider has no bound bridge owner")
            if not self.validate_copy_stream(self.producer_stream):
                self._poisoned = True
                self._phase = "poisoned"
                raise RuntimeError("bound producer stream identity changed")
            self._phase = "preparing"
        try:
            if self.reuse_mode == COMPLETION_CHECKED:
                assert self._torch is not None
                with self._torch.cuda.device(self._device_index):
                    if generation > 1:
                        synchronize = getattr(previous_event, "synchronize", None)
                        if not callable(synchronize):
                            raise TypeError("completion event must support synchronize()")
                        synchronize()
                        self._bounded_barrier()
                        self.bound_copy_module.mark_collective_reuse_safe(
                            self, previous_generation
                        )
                    ticket = self.bound_copy_module.submit(scheduler_outputs)
            else:
                if generation > 1:
                    self.bound_copy_module.mark_collective_reuse_safe(
                        self, previous_generation
                    )
                ticket = self.bound_copy_module.submit(scheduler_outputs)
            if self.next_generation != generation + 1:
                raise RuntimeError(
                    "bound copy next_generation did not advance exactly once"
                )
            if getattr(ticket, "remote_visibility_guaranteed", None) is not True:
                raise RuntimeError("copy ticket lacks remote visibility release")
            if getattr(ticket, "terminal_flags", None) is not self.broadcaster.terminal_flags:
                raise RuntimeError("copy ticket replaced the native terminal tensor")
            command_count = getattr(ticket, "command_count", None)
            if type(command_count) is not int or command_count < 1:
                raise RuntimeError("copy ticket command_count must include its terminal")
        except BaseException:
            self._poison()
            raise
        with self._lock:
            self._require_healthy()
            try:
                generation_advanced_once = self.next_generation == generation + 1
            except RuntimeError:
                self._poisoned = True
                self._phase = "poisoned"
                raise
            if self._phase != "preparing" or not generation_advanced_once:
                self._poisoned = True
                self._phase = "poisoned"
                raise RuntimeError("live-bank provider state changed during submit")
            self._active_ticket = ticket
            self._active_generation = generation
            self._phase = "leased"
            if generation > 1:
                self._completion_event = None
                self._collective_safe_generation = previous_generation
        return ticket

    def acquire_consumer(self, copy_ticket: object) -> _ProviderConsumerLease:
        with self._lock:
            self._require_healthy()
            if self._phase != "leased" or copy_ticket is not self._active_ticket:
                raise RuntimeError("copy ticket was not submitted through this provider")
            assert self._active_generation is not None
            lease = _ProviderConsumerLease(
                self._cookie, self._active_generation, copy_ticket
            )
            self._phase = "acquired"
            return lease

    def release_generation_after(
        self, lease: object, completion_event: object
    ) -> None:
        with self._lock:
            self._require_healthy()
            if (
                self._phase != "acquired"
                or not isinstance(lease, _ProviderConsumerLease)
                or lease.cookie is not self._cookie
                or lease.copy_ticket is not self._active_ticket
                or lease.generation != self._active_generation
            ):
                self._poisoned = True
                self._phase = "poisoned"
                raise RuntimeError("live-bank consumer lease is invalid")
            if not callable(getattr(completion_event, "synchronize", None)):
                self._poisoned = True
                self._phase = "poisoned"
                raise TypeError("completion event must support synchronize()")
            self._phase = "releasing"
            generation = lease.generation
        try:
            self.bound_copy_module.release_generation_after(generation, completion_event)
        except BaseException:
            self._poison()
            raise
        with self._lock:
            self._require_healthy()
            if self._phase != "releasing":
                self._poisoned = True
                self._phase = "poisoned"
                raise RuntimeError("live-bank provider state changed during release")
            self._released_generation = generation
            self._completion_event = completion_event
            self._active_ticket = None
            self._active_generation = None
            self._phase = "idle"

    def abort_generation(self, lease_or_ticket: object) -> None:
        del lease_or_ticket
        self._poison()


class _TorchStreamOperations:
    def __init__(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for the direct-live bridge") from exc
        self.torch = torch

    def record_event(self, stream: object) -> object:
        with self.torch.cuda.device(stream.device):
            event = self.torch.cuda.Event(blocking=False, interprocess=False)
            event.record(stream)
        return event

    def wait_event(self, stream: object, event: object) -> None:
        with self.torch.cuda.device(stream.device):
            stream.wait_event(event)


@dataclass(frozen=True)
class MegaMoeLiveWeightLease:
    """One immutable CPU launch epoch of routes and the direct live weight bank."""

    generation: int
    terminal_flags: object = field(repr=False)
    weight_planes: MegaMoeWeightPlanes = field(repr=False)
    physical_slot_ids: object = field(repr=False)
    route_ready_event: object = field(repr=False)
    ready_event: object = field(repr=False)
    command_count: int
    remote_visibility_guaranteed: bool = True
    _bridge: "SamiLiveWeightBridge" = field(repr=False, compare=False, default=None)  # type: ignore[assignment]
    _attestation_cookie: object = field(repr=False, compare=False, default=None)
    _copy_ticket: object = field(repr=False, compare=False, default=None)
    _provider_lease: object = field(repr=False, compare=False, default=None)
    _state: str = field(init=False, default="installed", repr=False, compare=False)

    def kernel_weight_kwargs(self) -> dict[str, object]:
        return self.weight_planes.kernel_kwargs()

    def release(self, consumer_stream: object) -> object:
        if self._state == "released":
            raise RuntimeError("live weight lease was already released")
        if self._state != "claimed":
            raise RuntimeError("live weight lease must be claimed before release")
        return self._bridge._release_live_lease(self, consumer_stream)


class SamiLiveWeightBridge:
    """Atomic adapter from one scheduler/copy producer to MegaMoE."""

    live_bank_count = 1

    def __init__(
        self,
        *,
        weight_planes: MegaMoeWeightPlanes,
        copy_module: object,
        copy_stream: object,
        collective_live_bank_lease_provider: object,
        stream_operations: Optional[object] = None,
        reuse_mode: str = COMPLETION_CHECKED,
        validation_mode: str = EACH_GENERATION,
    ) -> None:
        reuse = _reuse_mode(reuse_mode)
        validation = _validation_mode(validation_mode)
        if not isinstance(weight_planes, MegaMoeWeightPlanes):
            raise TypeError("weight_planes must be MegaMoeWeightPlanes")
        provider = collective_live_bank_lease_provider
        backend = getattr(copy_module, "backend", None)
        broadcaster = getattr(backend, "broadcaster", None)
        if broadcaster is None:
            raise TypeError("copy_module must expose a SAMI broadcaster")
        if getattr(provider, "collective_safe", None) is not True:
            raise RuntimeError("live-bank provider is not collective-safe")
        if getattr(provider, "producer_reuse_guarded", None) is not True:
            raise RuntimeError("live-bank provider does not guard producer reuse")
        if getattr(provider, "live_bank_count", None) != 1:
            raise ValueError("live-bank provider must manage exactly one bank")
        if getattr(provider, "reuse_mode", None) != reuse:
            raise ValueError("bridge and live-bank provider reuse modes differ")
        if getattr(provider, "validation_mode", None) != validation:
            raise ValueError("bridge and live-bank provider validation modes differ")
        if not callable(getattr(provider, "validate_binding", None)) or provider.validate_binding(
            copy_module, broadcaster
        ) is not True:
            raise RuntimeError("live-bank provider is not bound to this producer")
        if type(copy_stream) is int:
            raise TypeError("copy_stream must be a framework CUDA stream object")
        if provider.validate_copy_stream(copy_stream) is not True:
            raise RuntimeError("copy_stream is not the scheduler/SAMI producer stream")

        self.weight_planes = weight_planes
        self._bound_weight_planes = weight_planes
        self.copy_module = copy_module
        self.copy_stream = copy_stream
        self.broadcaster = broadcaster
        self._reuse_mode = reuse
        self._validation_mode = validation
        self._provider = provider
        self._ops = stream_operations or _TorchStreamOperations()
        self._lock = threading.Lock()
        self._aborted = False
        self._submission_in_progress = False
        self._active_live_lease: MegaMoeLiveWeightLease | None = None
        self._active_state: str | None = None
        self._active_consumer_stream_handle: int | None = None
        self._active_consumer_stream_device: str | None = None
        self._active_attestation: tuple[object, ...] | None = None
        self._attestation_cookie = object()

        self._validate_broadcaster()
        self.bound_scheduler_outputs = provider.bound_scheduler_outputs
        if provider.validate_scheduler_outputs_binding(self.bound_scheduler_outputs) is not True:
            raise RuntimeError("provider rejects its bound SchedulerOutputs")
        self._scheduler_contracts = tuple(
            (
                name,
                dimensions,
                _capture_int32_tensor(
                    getattr(self.bound_scheduler_outputs, name),
                    name=f"SchedulerOutputs.{name}",
                    dimensions=dimensions,
                ),
            )
            for name, dimensions in (
                ("physical_slot_ids", 2),
                ("hot_expert_ids", 1),
                ("hot_expert_source_ranks", 1),
            )
        )
        self._bound_physical_slot_ids = self.bound_scheduler_outputs.physical_slot_ids
        self._bound_hot_expert_ids = self.bound_scheduler_outputs.hot_expert_ids
        self._bound_hot_expert_source_ranks = (
            self.bound_scheduler_outputs.hot_expert_source_ranks
        )
        self._validated = weight_planes.validate(
            home_experts_per_rank=self.home_count,
            helper_experts_per_rank=self.helper_count,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            bundle_planes=self.bundle_planes,
        )
        if self._validated.device != self.device:
            raise ValueError("live weight planes and SAMI devices differ")
        supplied = tuple(tensor for _, tensor in weight_planes.items())
        if len(supplied) != len(self.broadcaster.live_plane_tensors) or any(
            actual is not expected
            for actual, expected in zip(supplied, self.broadcaster.live_plane_tensors)
        ):
            raise RuntimeError("MegaMoE must consume the allocator's exact live tensors")

        bind_bridge = getattr(provider, "bind_bridge_owner", None)
        if not callable(bind_bridge):
            raise TypeError("live-bank provider lacks bind_bridge_owner()")
        bind_bridge(self, copy_module, broadcaster)

    def _validate_broadcaster(self) -> None:
        try:
            self.world_size = int(self.broadcaster.comm.world)
            self.local_rank = int(self.broadcaster.comm.rank)
            self.helper_count = int(self.broadcaster.helper_count)
            global_experts = int(self.broadcaster.global_expert_count)
            self.hidden_size = int(self.broadcaster.bundle.hidden)
            self.intermediate_size = int(self.broadcaster.bundle.intermediate)
            self.bundle_planes = tuple(self.broadcaster.bundle.planes)
            self.arena = self.broadcaster.arena
            self.device = f"cuda:{int(self.broadcaster.device)}"
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError("broadcaster does not expose direct-live SAMI geometry") from exc
        _exact_positive_int("world_size", self.world_size)
        _exact_positive_int("global_experts", global_experts)
        _exact_positive_int("hidden_size", self.hidden_size)
        _exact_positive_int("intermediate_size", self.intermediate_size)
        if self.helper_count <= 0:
            raise ValueError(
                "direct-live DLB bridge requires S > 0; bypass DLB for S == 0"
            )
        self.home_count = global_experts // self.world_size
        _exact_positive_int("home_count", self.home_count)
        if (
            not 0 <= self.local_rank < self.world_size
            or global_experts % self.world_size
            or self.helper_count > self.home_count
        ):
            raise ValueError("invalid SAMI expert geometry")
        if getattr(self.broadcaster, "live_bank_count", None) != 1:
            raise RuntimeError("SAMI broadcaster must expose exactly one live bank")
        self._validate_arena_identity()
        if getattr(self.arena, "arena", None) is not self.broadcaster.live_arena:
            raise RuntimeError("bound arena does not own the broadcaster live arena")
        expected_plane_names = tuple(plane.name for plane in self.bundle_planes)
        expected_plane_bytes = tuple(int(plane.nbytes) for plane in self.bundle_planes)
        arena_geometry = (
            getattr(self.arena, "world", None),
            getattr(self.arena, "rank", None),
            getattr(self.arena, "home_count", None),
            getattr(self.arena, "helper_count", None),
            getattr(self.arena, "device", None),
            tuple(getattr(self.arena, "plane_names", ())),
            tuple(getattr(self.arena, "plane_bytes", ())),
        )
        expected_geometry = (
            self.world_size,
            self.local_rank,
            self.home_count,
            self.helper_count,
            int(self.broadcaster.device),
            expected_plane_names,
            expected_plane_bytes,
        )
        if arena_geometry != expected_geometry:
            raise RuntimeError("bound live arena geometry differs from SAMI")
        live_planes = tuple(self.broadcaster.live_plane_tensors)
        arena_planes = tuple(self.arena.local_plane_views)
        if len(live_planes) != len(arena_planes) or any(
            actual is not expected
            for actual, expected in zip(live_planes, arena_planes)
        ):
            raise RuntimeError("broadcaster live-plane identities are inconsistent")
        self.terminal_flags = self.broadcaster.terminal_flags
        if self.terminal_flags is not self.broadcaster.live_arena.terminals.local_view:
            raise RuntimeError("SAMI terminal tensor identity is inconsistent")
        if _pointer(self.terminal_flags, name="SAMI terminal_flags") != self.arena.terminal_local_ptr:
            raise RuntimeError("SAMI terminal pointer is inconsistent")
        if _shape(self.terminal_flags, name="SAMI terminal_flags") != (self.world_size,):
            raise ValueError("SAMI terminal flags must have shape [EP]")
        if _dtype_name(getattr(self.terminal_flags, "dtype", None)) != "uint64":
            raise TypeError("SAMI terminal flags must have dtype uint64")
        if _device_name(self.terminal_flags, name="SAMI terminal_flags") != self.device:
            raise ValueError("SAMI terminal flags are on the wrong device")

    @property
    def reuse_mode(self) -> str:
        return self._reuse_mode

    @property
    def validation_mode(self) -> str:
        return self._validation_mode

    @property
    def next_generation(self) -> int:
        return _copy_next_generation(self.copy_module)

    @property
    def physical_slot_ids(self) -> object:
        """Borrow the scheduler's routes until this generation's consumer completes."""
        return self._bound_physical_slot_ids

    @property
    def active_live_lease(self) -> MegaMoeLiveWeightLease | None:
        return self._active_live_lease

    def validate_scheduler_outputs_binding(self, scheduler_outputs: object) -> bool:
        if scheduler_outputs is not self.bound_scheduler_outputs:
            return False
        if self.validation_mode == BIND_ONCE:
            return (
                self.bound_scheduler_outputs.physical_slot_ids
                is self._bound_physical_slot_ids
                and self._provider.validate_scheduler_outputs_binding(
                    scheduler_outputs
                )
                is True
            )
        return (
            self._provider.validate_scheduler_outputs_binding(scheduler_outputs) is True
        )

    def _validate_bound_identity(self, scheduler_outputs: object) -> None:
        if scheduler_outputs is not self.bound_scheduler_outputs:
            raise RuntimeError("bound SchedulerOutputs identity changed")
        if scheduler_outputs.physical_slot_ids is not self._bound_physical_slot_ids:
            raise RuntimeError("SchedulerOutputs route tensor identity changed")
        if scheduler_outputs.hot_expert_ids is not self._bound_hot_expert_ids:
            raise RuntimeError("SchedulerOutputs hot expert tensor identity changed")
        if (
            scheduler_outputs.hot_expert_source_ranks
            is not self._bound_hot_expert_source_ranks
        ):
            raise RuntimeError("SchedulerOutputs source-rank tensor identity changed")
        if self.weight_planes is not self._bound_weight_planes:
            raise RuntimeError("live weight-plane bundle identity changed")
        if (
            self.broadcaster.terminal_flags is not self.terminal_flags
            or self.broadcaster.live_arena.terminals.local_view
            is not self.terminal_flags
        ):
            raise RuntimeError("native terminal tensor identity changed")

    def _validate_arena_identity(self) -> None:
        """Recheck the copy pointer tables through the public arena binder."""
        live_arena = self.broadcaster.live_arena
        if self.arena.arena is not live_arena:
            raise RuntimeError("bound arena does not own the broadcaster live arena")
        observed = live_arena.bind(
            world=self.world_size,
            rank=self.local_rank,
            home_count=self.home_count,
            helper_count=self.helper_count,
            bundle=self.broadcaster.bundle,
            device=int(self.broadcaster.device),
        )
        for name in (
            "world", "rank", "home_count", "helper_count", "device",
            "plane_names", "plane_bytes", "owner_source_ptrs",
            "scatter_source_ptrs", "destination_ptrs",
            "terminal_local_ptr", "terminal_mc_ptr",
        ):
            if getattr(observed, name) != getattr(self.arena, name):
                raise RuntimeError(f"bound live arena {name} changed since bind")
        expected_planes = tuple(self.arena.local_plane_views)
        observed_planes = tuple(observed.local_plane_views)
        if len(observed_planes) != len(expected_planes) or any(
            actual is not expected
            for actual, expected in zip(observed_planes, expected_planes)
        ):
            raise RuntimeError("bound live arena local view identity changed")

    def _validate_static_storage(self) -> None:
        self._validate_arena_identity()
        for name, dimensions, expected in self._scheduler_contracts:
            tensor = getattr(self.bound_scheduler_outputs, name)
            observed = _capture_int32_tensor(
                tensor, name=f"SchedulerOutputs.{name}", dimensions=dimensions
            )
            if tensor is not expected.tensor or observed != expected:
                raise RuntimeError(f"SchedulerOutputs.{name} storage changed")
        live_planes = tuple(self.broadcaster.live_plane_tensors)
        expected_planes = tuple(tensor for _, tensor in self.weight_planes.items())
        arena_planes = tuple(self.arena.local_plane_views)
        if (
            len(live_planes) != 7
            or any(a is not b for a, b in zip(live_planes, expected_planes))
            or any(a is not b for a, b in zip(live_planes, arena_planes))
        ):
            raise RuntimeError("live weight tensor identity changed")
        observed_planes = self.weight_planes.validate(
            home_experts_per_rank=self.home_count,
            helper_experts_per_rank=self.helper_count,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            bundle_planes=self.bundle_planes,
        )
        if observed_planes != self._validated:
            raise RuntimeError("live weight plane storage changed")
        if (
            self.broadcaster.terminal_flags is not self.terminal_flags
            or self.broadcaster.live_arena.terminals.local_view is not self.terminal_flags
            or _pointer(self.terminal_flags, name="SAMI terminal_flags")
            != self.arena.terminal_local_ptr
        ):
            raise RuntimeError("native terminal tensor identity changed")

    def _validate_submission(self, submission: object) -> None:
        if getattr(submission, "terminal_flags", None) is not self.terminal_flags:
            raise RuntimeError(
                "copy submission did not pass through the native terminal tensor"
            )
        if getattr(submission, "remote_visibility_guaranteed", None) is not True:
            raise RuntimeError("copy submission lacks system-scope remote visibility")
        command_count = getattr(submission, "command_count", None)
        if type(command_count) is not int or command_count < 1:
            raise ValueError(
                "copy submission command_count must include the terminal copy"
            )

    def submit_and_install(self, scheduler_outputs: object) -> MegaMoeLiveWeightLease:
        if scheduler_outputs is not self.bound_scheduler_outputs:
            raise ValueError("SchedulerOutputs are not bound to this producer")
        with self._lock:
            if self._aborted:
                raise RuntimeError("SAMI live-weight bridge is aborted; rebuild the group")
        try:
            if self._provider.validate_copy_stream(self.copy_stream) is not True:
                raise RuntimeError("bound producer stream identity changed")
            if self.validate_scheduler_outputs_binding(scheduler_outputs) is not True:
                raise RuntimeError("bound SchedulerOutputs storage changed")
            if self.validation_mode == EACH_GENERATION:
                self._validate_static_storage()
            else:
                self._validate_bound_identity(scheduler_outputs)
        except BaseException:
            with self._lock:
                self._aborted = True
                self._active_state = "aborted"
            self._provider.abort_generation(None)
            raise
        with self._lock:
            if self._aborted:
                raise RuntimeError("SAMI live-weight bridge is aborted; rebuild the group")
            if self._active_live_lease is not None:
                raise RuntimeError("single live bank is still leased")
            if self._submission_in_progress:
                raise RuntimeError("copy submission is already in progress")
            try:
                generation = self.next_generation
                if generation > TERMINAL_GENERATION_MASK:
                    raise RuntimeError(
                        "bound copy launch generation space is exhausted"
                    )
            except RuntimeError:
                self._aborted = True
                self._active_state = "aborted"
                self._provider.abort_generation(None)
                raise
            self._submission_in_progress = True
        ticket = None
        provider_lease = None
        try:
            record_event = getattr(self._ops, "record_event", None)
            if not callable(record_event):
                raise TypeError("stream operations lack record_event()")
            # A consumer on another stream must wait for scheduler completion,
            # even though it reads the original routes without a D2D copy.
            route_event = record_event(self.copy_stream)
            ticket = self._provider.submit_bound_copy(scheduler_outputs)
            if self.next_generation != generation + 1:
                raise RuntimeError(
                    "bound copy next_generation did not advance exactly once"
                )
            self._validate_submission(ticket)
            provider_lease = self._provider.acquire_consumer(ticket)
            if getattr(provider_lease, "generation", None) != generation:
                raise RuntimeError("provider lease launch generation is out of sequence")
            lease = MegaMoeLiveWeightLease(
                generation=generation,
                terminal_flags=ticket.terminal_flags,
                weight_planes=self.weight_planes,
                physical_slot_ids=self.physical_slot_ids,
                route_ready_event=route_event,
                ready_event=ticket.ready_event,
                command_count=ticket.command_count,
                remote_visibility_guaranteed=ticket.remote_visibility_guaranteed,
                _bridge=self,
                _attestation_cookie=self._attestation_cookie,
                _copy_ticket=ticket,
                _provider_lease=provider_lease,
            )
        except BaseException:
            with self._lock:
                self._submission_in_progress = False
                self._aborted = True
            self._provider.abort_generation(provider_lease or ticket)
            raise
        with self._lock:
            if self._aborted or not self._submission_in_progress:
                self._provider.abort_generation(provider_lease)
                raise RuntimeError("bridge state changed during submission")
            self._submission_in_progress = False
            self._active_live_lease = lease
            self._active_state = "installed"
            self._active_attestation = (
                generation,
                ticket,
                provider_lease,
                route_event,
                ticket.ready_event,
                ticket.command_count,
                ticket.terminal_flags,
            )
        return lease

    def _validated_active(
        self, lease: object, *, states: tuple[str, ...]
    ) -> MegaMoeLiveWeightLease:
        if self._aborted:
            raise RuntimeError("SAMI live-weight bridge is aborted; rebuild the group")
        active = self._active_live_lease
        if type(lease) is not MegaMoeLiveWeightLease or lease is not active:
            raise ValueError("object is not this bridge's active live-weight lease")
        assert active is not None
        try:
            if self.validation_mode == EACH_GENERATION:
                self._validate_static_storage()
            else:
                self._validate_bound_identity(self.bound_scheduler_outputs)
            attestation = self._active_attestation
            checks = (
                (attestation is not None, "attestation"),
                (active._bridge is self, "bridge identity"),
                (active._attestation_cookie is self._attestation_cookie, "attestation cookie"),
                (active._state == self._active_state and active._state in states, "state"),
                (
                    type(active.generation) is int
                    and 1 <= active.generation <= TERMINAL_GENERATION_MASK,
                    "launch generation",
                ),
                (type(active.command_count) is int and active.command_count > 0, "command count type"),
                (active.terminal_flags is self.terminal_flags, "terminal tensor"),
                (active.weight_planes is self.weight_planes, "weight planes"),
                (active.physical_slot_ids is self.physical_slot_ids, "scheduler routes"),
                (active.remote_visibility_guaranteed is True, "remote visibility"),
            )
            if attestation is not None:
                (
                    attested_generation,
                    attested_ticket,
                    attested_provider_lease,
                    attested_route_event,
                    attested_ready_event,
                    attested_command_count,
                    attested_terminal,
                ) = attestation
                checks += (
                    (
                        type(attested_generation) is int
                        and attested_generation == active.generation
                        and attested_ticket is active._copy_ticket
                        and attested_provider_lease is active._provider_lease
                        and getattr(attested_provider_lease, "generation", None)
                        == active.generation
                        and attested_route_event is active.route_ready_event
                        and attested_ready_event is active.ready_event
                        and type(attested_command_count) is int
                        and attested_command_count == active.command_count
                        and attested_terminal is active.terminal_flags,
                        "ticket attestation",
                    ),
                )
            for valid, name in checks:
                if not valid:
                    raise ValueError(f"active live-weight lease has invalid {name}")
            self._validate_submission(active._copy_ticket)
        except BaseException:
            self._aborted = True
            self._active_state = "aborted"
            object.__setattr__(active, "_state", "aborted")
            self._provider.abort_generation(active._provider_lease)
            raise
        return active

    def validate_live_weight_lease(self, lease: object) -> MegaMoeLiveWeightLease:
        with self._lock:
            return self._validated_active(lease, states=("installed", "claimed"))

    def claim_live_weight_lease(
        self, lease: object, consumer_stream: object
    ) -> MegaMoeLiveWeightLease:
        handle = _stream_handle(consumer_stream)
        device = _stream_device_name(consumer_stream, name="consumer_stream")
        if device != self.device:
            raise ValueError("consumer_stream and live arena devices differ")
        with self._lock:
            active = self._validated_active(lease, states=("installed",))
            try:
                self._ops.wait_event(consumer_stream, active.route_ready_event)
            except BaseException:
                self._aborted = True
                self._provider.abort_generation(active._provider_lease)
                raise
            self._active_consumer_stream_handle = handle
            self._active_consumer_stream_device = device
            self._active_state = "claimed"
            object.__setattr__(active, "_state", "claimed")
            return active

    def _release_live_lease(
        self, lease: MegaMoeLiveWeightLease, consumer_stream: object
    ) -> object:
        handle = _stream_handle(consumer_stream)
        device = _stream_device_name(consumer_stream, name="consumer_stream")
        with self._lock:
            active = self._validated_active(lease, states=("claimed",))
            if (
                handle != self._active_consumer_stream_handle
                or device != self._active_consumer_stream_device
            ):
                raise ValueError("live weight lease must be released on its bound stream")
            try:
                completion_event = self._ops.record_event(consumer_stream)
                self._provider.release_generation_after(
                    active._provider_lease, completion_event
                )
            except BaseException:
                self._aborted = True
                self._provider.abort_generation(active._provider_lease)
                raise
            self._active_state = "released"
            object.__setattr__(active, "_state", "released")
            self._active_live_lease = None
            self._active_attestation = None
            self._active_state = None
            self._active_consumer_stream_handle = None
            self._active_consumer_stream_device = None
            return completion_event

    def abort_live_weight_lease(self, lease: object) -> None:
        with self._lock:
            if self._aborted:
                return
            self._aborted = True
            active = self._active_live_lease
            self._active_state = "aborted"
            if active is not None:
                object.__setattr__(active, "_state", "aborted")
            provider_lease = getattr(active or lease, "_provider_lease", lease)
        self._provider.abort_generation(provider_lease)

    def install(self, copy_ticket: object) -> MegaMoeLiveWeightLease:
        self.abort_live_weight_lease(copy_ticket)
        raise RuntimeError("CopyTicket bypass is forbidden; use submit_and_install()")


__all__ = [
    "BIND_ONCE",
    "COMPLETION_CHECKED",
    "EACH_GENERATION",
    "MegaMoeLiveWeightLease",
    "MegaMoeWeightPlanes",
    "STREAM_ORDERED",
    "SamiLiveWeightBridge",
    "TorchDistributedLiveBankLeaseProvider",
]

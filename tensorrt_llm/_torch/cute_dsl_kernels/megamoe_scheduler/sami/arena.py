# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""External single-bank MegaMoE weight arena consumed directly by SAMI.

The production arena is plane-major.  Each of the seven tensors has local
physical slots ``[0, H)`` for canonical experts and ``[H, H + S)`` for live
helpers.  A plane exposes every rank's unicast mapping plus the multicast
mapping of the same backing allocation.  This is the shape naturally exposed
by TEKit ``McastGPUBuffer.get_uc_buffer``/``get_mc_buffer``; callers without
that class can provide equivalent typed CUDA tensor views.

This module performs no allocation and deliberately does not import TEKit.
It validates geometry, pointer and object identity once at bind time and
resolves the pointer tables the copy backend uses from then on; identity is
not re-checked per submit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .geometry import BundleLayout


def _exact_positive_int(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive exact int")
    return value


def _pointer(value: object, *, name: str) -> int:
    accessor = getattr(value, "data_ptr", None)
    pointer = accessor() if callable(accessor) else accessor
    if type(pointer) is not int or pointer <= 0:
        raise TypeError(f"{name} must expose a positive integer data_ptr")
    return pointer


def _shape(value: object, *, name: str) -> tuple[int, ...]:
    try:
        result = tuple(int(extent) for extent in value.shape)  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(f"{name} must expose an integer shape") from exc
    if not result or any(extent <= 0 for extent in result):
        raise ValueError(f"{name} must have a non-empty positive shape")
    return result


def _stride(value: object, *, name: str) -> tuple[int, ...]:
    accessor = getattr(value, "stride", None)
    raw = accessor() if callable(accessor) else accessor
    try:
        result = tuple(int(step) for step in raw)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must expose an integer stride") from exc
    if any(step <= 0 for step in result):
        raise ValueError(f"{name} must have positive strides")
    return result


def _dtype(value: object) -> str:
    return str(getattr(value, "dtype", None)).lower().removeprefix("torch.")


def _element_size(value: object, *, name: str) -> int:
    accessor = getattr(value, "element_size", None)
    raw = accessor() if callable(accessor) else accessor
    if type(raw) is not int or raw <= 0:
        raise TypeError(f"{name} must expose a positive integer element_size")
    return raw


def _device_index(value: object, *, name: str) -> int:
    if getattr(value, "is_cuda", True) is not True:
        raise ValueError(f"{name} must reside on CUDA memory")
    device = getattr(value, "device", None)
    text = str(device).lower()
    if not text.startswith("cuda"):
        raise ValueError(f"{name} must reside on CUDA memory")
    index = getattr(device, "index", None)
    if type(index) is not int:
        if ":" not in text:
            raise ValueError(f"{name} must expose an explicit CUDA device ordinal")
        try:
            index = int(text.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"{name} exposes an invalid CUDA device") from exc
    if index < 0:
        raise ValueError(f"{name} exposes an invalid CUDA device")
    return index


def _expected_plane_metadata(
    bundle: BundleLayout, slot_count: int
) -> dict[str, tuple[tuple[int, ...], tuple[int, ...], str, int, int]]:
    hidden = bundle.hidden
    intermediate = bundle.intermediate
    by_name = {plane.name: plane for plane in bundle.planes}
    gate_up = 2 * intermediate
    fc1_sf = by_name["mega_fc1_weight_sf"].nbytes
    fc2_sf = by_name["mega_fc2_weight_sf"].nbytes
    return {
        "mega_fc1_weight": (
            (slot_count, hidden // 2, gate_up),
            (hidden * gate_up // 2, 1, hidden // 2),
            "float4_e2m1fn_x2",
            1,
            16,
        ),
        "mega_fc1_weight_sf": (
            (slot_count, fc1_sf),
            (fc1_sf, 1),
            "float8_e4m3fn",
            1,
            16,
        ),
        "mega_fc2_weight": (
            (slot_count, intermediate // 2, hidden),
            (intermediate * hidden // 2, 1, intermediate // 2),
            "float4_e2m1fn_x2",
            1,
            16,
        ),
        "mega_fc2_weight_sf": (
            (slot_count, fc2_sf),
            (fc2_sf, 1),
            "float8_e4m3fn",
            1,
            16,
        ),
        "fc31_alpha": ((slot_count,), (1,), "float32", 4, 4),
        "fc2_alpha": ((slot_count,), (1,), "float32", 4, 4),
        "fc1_norm_const": ((slot_count,), (1,), "float32", 4, 4),
    }

@dataclass(frozen=True)
class LivePlaneView:
    """One live plane's rank-ordered UC views and its MC alias.

    ``aliases_same_backing`` is an explicit provider attestation.  Pointer
    arithmetic alone cannot prove a CUDA UC/MC alias relationship, so an
    un-attested collection of unrelated tensors is rejected.
    """

    name: str
    uc_views: tuple[object, ...] = field(repr=False, compare=False)
    mc_view: object = field(repr=False, compare=False)
    aliases_same_backing: bool = False
    backing_owner: object | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class LiveTerminalView:
    """Local ``uint64[EP]`` terminals and the matching MC alias."""

    local_view: object = field(repr=False, compare=False)
    mc_view: object = field(repr=False, compare=False)
    aliases_same_backing: bool = False
    backing_owner: object | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class _TensorIdentity:
    object_id: int
    pointer: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    element_size: int
    device: int
    name: str

    @classmethod
    def capture(cls, value: object, *, name: str) -> "_TensorIdentity":
        shape = _shape(value, name=name)
        stride = _stride(value, name=name)
        if len(shape) != len(stride):
            raise ValueError(f"{name} shape/stride rank mismatch")
        return cls(
            object_id=id(value),
            pointer=_pointer(value, name=name),
            shape=shape,
            stride=stride,
            dtype=_dtype(value),
            element_size=_element_size(value, name=name),
            device=_device_index(value, name=name),
            name=name,
        )

    @property
    def metadata(self) -> tuple:
        """The layout a view must match; deliberately excludes the pointer."""
        return (self.shape, self.stride, self.dtype, self.element_size,
                self.device)


@dataclass(frozen=True)
class BoundLiveWeightArena:
    """Validated pointer tables and strong references for one SAMI rank."""

    arena: "LiveWeightArena" = field(repr=False, compare=False)
    world: int
    rank: int
    home_count: int
    helper_count: int
    device: int
    plane_names: tuple[str, ...]
    plane_bytes: tuple[int, ...]
    local_plane_views: tuple[object, ...] = field(repr=False, compare=False)
    owner_source_ptrs: tuple[int, ...]
    scatter_source_ptrs: tuple[int, ...]
    destination_ptrs: tuple[int, ...]
    terminal_local_ptr: int
    terminal_mc_ptr: int

    @property
    def local_slot_count(self) -> int:
        return self.home_count + self.helper_count


@dataclass(frozen=True)
class LiveWeightArena:
    """Seven plane-major live weight views plus one terminal allocation."""

    planes: tuple[LivePlaneView, ...]
    terminals: LiveTerminalView

    def bind(
        self,
        *,
        world: int,
        rank: int,
        home_count: int,
        helper_count: int,
        bundle: BundleLayout,
        device: int,
    ) -> BoundLiveWeightArena:
        world = _exact_positive_int("world", world)
        home_count = _exact_positive_int("home_count", home_count)
        helper_count = _exact_positive_int("helper_count", helper_count)
        if type(rank) is not int or not 0 <= rank < world:
            raise ValueError("rank must be in [0, world)")
        if type(device) is not int or device < 0:
            raise ValueError("device must be a non-negative exact int")

        expected_names = tuple(plane.name for plane in bundle.planes)
        observed_names = tuple(plane.name for plane in self.planes)
        if observed_names != expected_names:
            raise ValueError(
                f"live arena plane order mismatch: {observed_names} != {expected_names}"
            )
        if len(self.planes) != 7:
            raise ValueError("live arena must contain exactly seven planes")

        slot_count = home_count + helper_count
        expected_metadata_by_name = _expected_plane_metadata(bundle, slot_count)
        local_planes: list[object] = []
        plane_bytes: list[int] = []
        owner_sources: list[int] = []
        scatter_sources: list[int] = []
        destinations: list[int] = []
        uc_ranges: list[list[tuple[int, int, str]]] = [
            [] for _ in range(world)
        ]
        mc_ranges: list[tuple[int, int, str]] = []

        plane_identities: list[tuple[_TensorIdentity, ...]] = []
        mc_identities: list[_TensorIdentity] = []
        for plane_index, (view, layout) in enumerate(zip(self.planes, bundle.planes)):
            if view.aliases_same_backing is not True:
                raise ValueError(f"{view.name} UC/MC alias is not provider-attested")
            if len(view.uc_views) != world:
                raise ValueError(f"{view.name} must expose exactly world UC views")

            captured = tuple(
                _TensorIdentity.capture(value, name=f"{view.name}.uc[{peer}]")
                for peer, value in enumerate(view.uc_views)
            )
            mc = _TensorIdentity.capture(view.mc_view, name=f"{view.name}.mc")
            reference = captured[rank]
            (
                expected_shape,
                expected_stride,
                expected_dtype,
                expected_element_size,
                expected_pointer_alignment,
            ) = expected_metadata_by_name[view.name]
            expected_metadata = (
                expected_shape,
                expected_stride,
                expected_dtype,
                expected_element_size,
                device,
            )
            for identity in (*captured, mc):
                if identity.metadata != expected_metadata:
                    raise ValueError(
                        f"{view.name} must have exact MegaMoE "
                        f"shape={expected_shape}, stride={expected_stride}, "
                        f"dtype={expected_dtype} on cuda:{device}"
                    )
                if identity.pointer % expected_pointer_alignment:
                    raise ValueError(
                        f"{view.name} pointer must be "
                        f"{expected_pointer_alignment}-byte aligned"
                    )
            if len({identity.pointer for identity in captured}) != world:
                raise ValueError(f"{view.name} UC rank mappings must have distinct pointers")
            if mc.pointer in {identity.pointer for identity in captured}:
                raise ValueError(f"{view.name} MC and UC mappings must be distinct")
            if layout.nbytes % reference.element_size:
                raise ValueError(f"{view.name} bytes/expert are not dtype aligned")

            expert_elements = layout.nbytes // reference.element_size
            tail_span = 1 + sum(
                (extent - 1) * stride
                for extent, stride in zip(reference.shape[1:], reference.stride[1:])
            )
            if reference.stride[0] != expert_elements or tail_span != expert_elements:
                raise ValueError(
                    f"{view.name} must be dense per expert with byte stride {layout.nbytes}"
                )

            plane_identities.append(captured)
            mc_identities.append(mc)
            local_planes.append(view.uc_views[rank])
            plane_bytes.append(layout.nbytes)
            for peer, identity in enumerate(captured):
                uc_ranges[peer].append(
                    (
                        identity.pointer,
                        identity.pointer + slot_count * layout.nbytes,
                        view.name,
                    )
                )
            mc_ranges.append(
                (mc.pointer, mc.pointer + slot_count * layout.nbytes, view.name)
            )

        weight_ranges = [
            (begin, end, f"{name}.uc[{peer}]")
            for peer, peer_ranges in enumerate(uc_ranges)
            for begin, end, name in peer_ranges
        ]
        weight_ranges.extend(
            (begin, end, f"{name}.mc") for begin, end, name in mc_ranges
        )
        self._reject_overlaps(weight_ranges)

        for local_expert in range(home_count):
            for plane_index, layout in enumerate(bundle.planes):
                owner_sources.append(
                    plane_identities[plane_index][rank].pointer
                    + local_expert * layout.nbytes
                )
        for expert in range(world * home_count):
            owner, local_expert = divmod(expert, home_count)
            for plane_index, layout in enumerate(bundle.planes):
                scatter_sources.append(
                    plane_identities[plane_index][owner].pointer
                    + local_expert * layout.nbytes
                )
        for helper_index in range(helper_count):
            physical_slot = home_count + helper_index
            for plane_index, layout in enumerate(bundle.planes):
                destinations.append(
                    mc_identities[plane_index].pointer
                    + physical_slot * layout.nbytes
                )

        if self.terminals.aliases_same_backing is not True:
            raise ValueError("terminal UC/MC alias is not provider-attested")
        terminal_local = _TensorIdentity.capture(
            self.terminals.local_view, name="terminal.local"
        )
        terminal_mc = _TensorIdentity.capture(self.terminals.mc_view, name="terminal.mc")
        for terminal in (terminal_local, terminal_mc):
            if terminal.shape != (world,) or terminal.stride != (1,):
                raise ValueError("terminal views must have contiguous shape [EP]")
            if terminal.dtype != "uint64" or terminal.element_size != 8:
                raise TypeError("terminal views must have dtype uint64")
            if terminal.device != device:
                raise ValueError("terminal views are on the wrong CUDA device")
            if terminal.pointer % 8:
                raise ValueError("terminal pointers must be 8-byte aligned")
        self._reject_terminal_overlap(terminal_local, weight_ranges, "local")
        self._reject_terminal_overlap(terminal_mc, weight_ranges, "multicast")
        terminal_bytes = world * 8
        if (
            terminal_local.pointer < terminal_mc.pointer + terminal_bytes
            and terminal_mc.pointer < terminal_local.pointer + terminal_bytes
        ):
            raise ValueError("terminal local and multicast mappings overlap")

        return BoundLiveWeightArena(
            arena=self,
            world=world,
            rank=rank,
            home_count=home_count,
            helper_count=helper_count,
            device=device,
            plane_names=expected_names,
            plane_bytes=tuple(plane_bytes),
            local_plane_views=tuple(local_planes),
            owner_source_ptrs=tuple(owner_sources),
            scatter_source_ptrs=tuple(scatter_sources),
            destination_ptrs=tuple(destinations),
            terminal_local_ptr=terminal_local.pointer,
            terminal_mc_ptr=terminal_mc.pointer,
        )

    @staticmethod
    def _reject_overlaps(ranges: list[tuple[int, int, str]]) -> None:
        ordered = sorted(ranges)
        for (_, previous_end, previous), (begin, _, current) in zip(
            ordered, ordered[1:]
        ):
            if begin < previous_end:
                raise ValueError(
                    f"live arena mappings overlap: {previous} and {current}"
                )

    @staticmethod
    def _reject_terminal_overlap(
        terminal: _TensorIdentity,
        ranges: list[tuple[int, int, str]],
        label: str,
    ) -> None:
        begin, end = terminal.pointer, terminal.pointer + terminal.shape[0] * 8
        for plane_begin, plane_end, name in ranges:
            if begin < plane_end and plane_begin < end:
                raise ValueError(
                    f"{label} terminal overlaps live weight plane {name}"
                )


@runtime_checkable
class LiveWeightArenaProvider(Protocol):
    """Provider boundary implemented by TEKit or an equivalent allocator."""

    def build_live_arena(
        self,
        *,
        world: int,
        rank: int,
        home_count: int,
        helper_count: int,
        bundle: BundleLayout,
        device: int,
        comm: object,
    ) -> LiveWeightArena:
        """Return one fully constructed and provider-attested live arena."""


__all__ = [
    "BoundLiveWeightArena",
    "LivePlaneView",
    "LiveTerminalView",
    "LiveWeightArena",
    "LiveWeightArenaProvider",
]

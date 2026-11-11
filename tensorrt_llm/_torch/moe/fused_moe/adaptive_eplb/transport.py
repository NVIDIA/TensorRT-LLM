# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expert weights of adaptive EPLB in MNNVL memory, and the copies between ranks.

After loading, every balanced layer's slot planes (backend parameters whose leading axis is the
local slot) move into one MNNVL block per layer that every EP rank maps. An install copies experts
from peers as one batch on the copy engines, overlapping the forward's kernels.
"""

from __future__ import annotations

import functools
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Protocol

import torch

from tensorrt_llm._torch.distributed.mnnvl_memory import EpMnnvlMemory, MnnvlMemory
from tensorrt_llm.logger import logger
from tensorrt_llm.math_utils import pad_up

from ..impl_base import MoEImplBase

if TYPE_CHECKING:
    from tensorrt_llm.mapping import Mapping as ParallelMapping

    from .balancer import AdaptiveMoeLayerLoadBalancer

MIN_PLANE_ALIGNMENT = 512
"""int: Least byte alignment of a plane's start: the block size of the MNNVL allocator."""


@dataclass(frozen=True)
class SlotPlane:
    """A backend parameter whose leading axis is the local slot; it starts on ``alignment`` bytes."""

    name: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    alignment: int

    @property
    def slots(self) -> int:
        return self.shape[0]

    @property
    def slot_bytes(self) -> int:
        """Bytes from one slot's row to the next."""
        return self.stride[0] * self.dtype.itemsize


class LayerBlockLayout:
    """Byte ``offsets`` of one layer's slot planes in its block, each aligned, and the block size."""

    def __init__(
        self,
        planes: Sequence[SlotPlane],
        slots: int,
        *,
        min_alignment: int = MIN_PLANE_ALIGNMENT,
    ) -> None:
        self.planes = tuple(planes)
        self.slots = int(slots)
        self.offsets: dict[str, int] = {}
        cursor = 0
        for plane in self.planes:
            alignment = math.lcm(plane.alignment, min_alignment)
            cursor = pad_up(cursor, alignment)
            self.offsets[plane.name] = cursor
            cursor += self.slots * plane.slot_bytes
        self.nbytes = pad_up(cursor, min_alignment)

    @property
    def expert_bytes(self) -> int:
        """Bytes of one slot over all planes: what one expert copy moves."""
        return sum(plane.slot_bytes for plane in self.planes)

    def signature(self) -> tuple[int, int, tuple[tuple[str, int, int], ...]]:
        """A description of the layout that compares equal on ranks with the same blocks."""
        planes = tuple((p.name, p.slot_bytes, self.offsets[p.name]) for p in self.planes)
        return self.slots, self.nbytes, planes


@functools.cache
def _batch_attributes() -> tuple[ModuleType, list[object]]:
    """The CUDA runtime bindings and batched-copy attributes; needs CUDA 12.8 or later bindings."""
    try:
        from cuda.bindings import runtime as cudart

        cudart.cudaMemcpyBatchAsync
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "adaptive EPLB copies experts with cudaMemcpyBatchAsync, which needs CUDA 12.8 or "
            f"later bindings: {exc}"
        ) from exc
    attributes = cudart.cudaMemcpyAttributes()
    attributes.srcAccessOrder = cudart.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderStream
    attributes.flags = cudart.cudaMemcpyFlags.cudaMemcpyFlagPreferOverlapWithCompute
    return cudart, [attributes]


def _copy_batch(
    dst: list[int], src: list[int], sizes: list[int], stream: torch.cuda.Stream
) -> None:
    """One ``cudaMemcpyBatchAsync`` on ``stream``, after its earlier work, on the copy engines."""
    cudart, attributes = _batch_attributes()
    (error,) = cudart.cudaMemcpyBatchAsync(
        dst, src, sizes, len(dst), attributes, [0], 1, stream.cuda_stream
    )
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaMemcpyBatchAsync of {len(dst)} copies failed: {error}")


def dedicated_copy_stream(device: torch.device) -> torch.cuda.Stream:
    """A non-blocking stream of the copies' own on ``device``, kept for the life of the process.

    ``torch.cuda.Stream()`` hands out the streams of a small per-device pool in turn, and other
    components take theirs from the same pool: a pooled copy stream can be a stream that also runs
    kernels, and the copies would then wait behind them instead of overlapping them.
    """
    cudart, _ = _batch_attributes()
    with torch.cuda.device(device):
        error, stream = cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking)
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaStreamCreateWithFlags failed: {error}")
    return torch.cuda.ExternalStream(int(stream), device=device)


class PeerSlotTransport:
    """Copies experts from a peer rank's layer block into this rank's slots.

    The blocks are ``memory`` of the MoE EP group of ``mapping``; ``copy_batch`` issues the byte
    copies of one batch on a stream.
    """

    def __init__(
        self,
        mapping: ParallelMapping,
        *,
        memory: type[MnnvlMemory] = EpMnnvlMemory,
        copy_batch: Callable[[list[int], list[int], list[int], torch.cuda.Stream], None] = (
            _copy_batch
        ),
    ) -> None:
        self._comm = memory.get_comm(mapping)
        self.ep_rank = self._comm.Get_rank()
        self.ep_size = self._comm.Get_size()
        self._mapping = mapping
        self._memory = memory
        self._copy_batch = copy_batch
        self._blocks: list[MnnvlMemory] = []
        self._layouts: dict[int, LayerBlockLayout] = {}
        self._plane_rows: dict[int, dict[str, torch.Tensor]] = {}
        self._addresses: dict[int, tuple[tuple[tuple[int, ...], int], ...]] = {}

    def add_layer(self, layer: int, layout: LayerBlockLayout) -> MnnvlMemory:
        """Allocate one layer's block on every rank and map them all; collective over EP.

        Raises ``ValueError`` when a rank lays the block out differently.
        """
        signature = layout.signature()
        differ = [
            r for r, theirs in enumerate(self._comm.allgather(signature)) if theirs != signature
        ]
        if differ:
            raise ValueError(
                f"layer {layer}: ranks {differ} lay out the block differently from rank "
                f"{self.ep_rank}, which has {signature}"
            )
        block = self._memory(self._mapping, layout.nbytes)
        self._blocks.append(block)
        rows_by_rank = block.as_torch_strided_tensor(torch.uint8)
        table = {}
        addresses = []
        for plane in layout.planes:
            start = layout.offsets[plane.name]
            rows = rows_by_rank[:, start : start + layout.slots * plane.slot_bytes]
            table[plane.name] = rows.unflatten(1, (layout.slots, plane.slot_bytes))
            addresses.append((tuple(row.data_ptr() for row in rows), plane.slot_bytes))
        self._plane_rows[layer] = table
        self._addresses[layer] = tuple(addresses)
        self._layouts[layer] = layout
        return block

    def allgather(self, value: object) -> list[object]:
        """Every rank's ``value``, by rank; collective over the EP group."""
        return self._comm.allgather(value)

    def expert_bytes(self, layer: int) -> int:
        """Bytes one expert copy of ``layer`` moves."""
        return self._layouts[layer].expert_bytes

    def local_planes(self, layer: int) -> dict[str, torch.Tensor]:
        """Plane name to this rank's rows, ``uint8 [slots, slot_bytes]``."""
        return {name: rows[self.ep_rank] for name, rows in self._plane_rows[layer].items()}

    def copy(
        self,
        experts: Iterable[tuple[int, int, int, int]],
        stream: torch.cuda.Stream,
        start: torch.cuda.Event,
        end: torch.cuda.Event,
    ) -> None:
        """Copy ``(layer, dst_local_slot, src_rank, src_local_slot)`` experts as one batch on
        ``stream`` between ``start`` and ``end``; neither side may be touched until ``end``."""
        dst = []
        src = []
        sizes = []
        for layer, dst_local_slot, src_rank, src_local_slot in experts:
            for bases, slot_bytes in self._addresses[layer]:
                dst.append(bases[self.ep_rank] + dst_local_slot * slot_bytes)
                src.append(bases[src_rank] + src_local_slot * slot_bytes)
                sizes.append(slot_bytes)
        start.record(stream)
        if dst:
            self._copy_batch(dst, src, sizes, stream)
        end.record(stream)


class AdaptiveEplbBackend(Protocol):
    """What a MoE backend whose capabilities include ``supports_adaptive_eplb`` provides.

    Such a backend initializes ``torch.distributed`` by the end of its construction. It may also
    provide ``route_staging(num_tokens, top_k)``: the staging of its next run that the route op
    fills (see ``AdaptiveMoeLayerLoadBalancer.route``).
    """

    slot_id_dtype: torch.dtype
    """The integer dtype the backend reads slot ids in."""

    adaptive_eplb_block: object
    """The MNNVL memory the slot parameters point into once moved; the backend keeps it alive."""

    @classmethod
    def has_decode_time_model(cls, device_name: str) -> bool:
        """Whether :meth:`decode_time_model` has a calibration for the device."""

    def slot_weight_planes(self) -> list[tuple[str, int]]:
        """``(parameter name, byte alignment)`` of every parameter whose leading axis is the slot."""

    def check_weights_movable(self) -> None:
        """Raise if the backend already holds views of its slot planes, e.g. after a forward."""

    def decode_time_model(self, expert_bytes: int, device_name: str) -> dict[str, float]:
        """The :class:`~.policy.TimeModel` fields of the backend's decode kernels on the device.

        ``expert_bytes`` is the size of one expert's weights. Raises ``ValueError`` when the device
        has no calibration.
        """


def _plan_layer(layer: int, backend: AdaptiveEplbBackend) -> LayerBlockLayout:
    backend.check_weights_movable()
    planes = []
    for name, alignment in backend.slot_weight_planes():
        param = backend.get_parameter(name)
        nbytes = param.numel() * param.element_size()
        if (
            not param.is_contiguous()
            or param.storage_offset() != 0
            or param.untyped_storage().nbytes() != nbytes
        ):
            raise ValueError(
                f"MoE layer {layer}, plane {name!r}: the parameter is not alone in one "
                "contiguous storage"
            )
        planes.append(
            SlotPlane(name, tuple(param.shape), tuple(param.stride()), param.dtype, alignment)
        )
    slots = {plane.slots for plane in planes}
    if len(slots) != 1:
        raise ValueError(f"MoE layer {layer}: the slot planes disagree on the slots: {slots}")
    return LayerBlockLayout(planes, slots.pop())


def migrate_backend_weights(
    backends: Mapping[int, AdaptiveEplbBackend],
    transport: PeerSlotTransport,
    device: torch.device,
) -> int:
    """Move every layer's slot planes into a block of ``transport``; collective over EP.

    All ranks check all layers first, so a failure on any rank raises on every rank before any block
    is allocated. Returns the bytes moved on this rank.
    """
    plans: dict[int, LayerBlockLayout] = {}
    error = None
    try:
        plans = {layer: _plan_layer(layer, backends[layer]) for layer in sorted(backends)}
    except (RuntimeError, ValueError) as exc:
        error = str(exc)
    errors = transport.allgather(error)
    failed = [f"rank {rank}: {message}" for rank, message in enumerate(errors) if message]
    if failed:
        what = "adaptive EPLB weight migration failed checking the backend planes"
        raise RuntimeError(f"{what}: {'; '.join(failed)}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)  # the checkpoint load may still be writing the planes
    moved = 0
    for layer, layout in plans.items():
        module = backends[layer]
        block = transport.add_layer(layer, layout)
        rows = transport.local_planes(layer)
        # Assigning ``data`` keeps every Parameter object, so holders of the parameters (the quant
        # scales hold the per-expert scale planes) see the copy in the block as well. The backend
        # keeps the block its parameters now point into, so the block outlives the transport.
        for plane in layout.planes:
            param = module.get_parameter(plane.name)
            alias = rows[plane.name].view(-1).view(plane.dtype)
            alias = alias.as_strided(plane.shape, plane.stride).copy_(param.detach())
            param.data = alias
        module.adaptive_eplb_block = block
        torch.cuda.empty_cache()
        moved += layout.slots * layout.expert_bytes
    return moved


def find_backends(
    model: torch.nn.Module, layers: Sequence[AdaptiveMoeLayerLoadBalancer]
) -> dict[int, AdaptiveEplbBackend]:
    """The backend module of every balanced layer: the ``MoEImplBase`` that holds the layer's handle
    and supports adaptive EPLB. Raises ``ValueError`` when a layer has none, or two."""
    wanted = {id(layer): layer for layer in layers}
    found: dict[int, AdaptiveEplbBackend] = {}
    for module in model.modules():
        layer = wanted.get(id(getattr(module, "layer_load_balancer", None)))
        if (
            layer is None
            or not isinstance(module, MoEImplBase)
            or not module.capabilities.supports_adaptive_eplb
        ):
            continue
        if layer.layer_idx in found:
            raise ValueError(f"balanced layer {layer.layer_idx} has two backend modules")
        found[layer.layer_idx] = module
    missing = sorted(layer.layer_idx for layer in layers if layer.layer_idx not in found)
    if missing:
        raise ValueError(
            f"balanced layers {missing} have no MoE backend that supports adaptive EPLB"
        )
    return found


def peer_copy_gap(mapping: ParallelMapping) -> str | None:
    """Why this rank cannot copy experts from every EP rank, or ``None``.

    The copies read the peers' memory over NVLink, so the EP ranks must share a node or one MNNVL
    domain, and they run as batches of CUDA 12.8 bindings.
    """
    nodes = {rank // mapping.gpus_per_node for rank in mapping.moe_ep_group}
    if len(nodes) > 1 and not MnnvlMemory.supports_mnnvl():
        return f"the EP ranks span {len(nodes)} nodes outside one MNNVL domain"
    try:
        _batch_attributes()
    except RuntimeError as exc:
        return str(exc)
    return None


def prepare_layers(
    backends: Mapping[int, AdaptiveEplbBackend],
    layers: Sequence[AdaptiveMoeLayerLoadBalancer],
    device: torch.device,
) -> None:
    """Bind every layer handle's statistics and route tables to its backend on ``device``."""
    for layer in layers:
        backend = backends[layer.layer_idx]
        route_staging = getattr(backend, "route_staging", None)
        layer.prepare_device_state(backend.slot_id_dtype, device, route_staging)


def migrate_model_weights(
    backends: Mapping[int, AdaptiveEplbBackend],
    transport: PeerSlotTransport,
    device: torch.device,
) -> torch.cuda.Stream:
    """Move the slot planes of every balanced layer into MNNVL blocks of ``transport``.

    Every EP rank must call it. Returns the copy stream (:func:`dedicated_copy_stream`).
    """
    free = torch.cuda.mem_get_info(device)[0]
    start = time.perf_counter()
    moved = migrate_backend_weights(backends, transport, device)
    gib = float(1 << 30)
    logger.info(
        f"adaptive EPLB moved {moved / gib:.2f} GiB of slot planes of {len(backends)} MoE layers "
        f"into MNNVL blocks in {time.perf_counter() - start:.2f} s; free device memory "
        f"{free / gib:.2f} -> {torch.cuda.mem_get_info(device)[0] / gib:.2f} GiB"
    )
    return dedicated_copy_stream(device)

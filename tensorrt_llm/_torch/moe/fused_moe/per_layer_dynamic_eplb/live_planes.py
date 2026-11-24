# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live-weight planes of a backend whose kernel reads per-expert tensors.

A grouped-GEMM backend keeps every expert-indexed operand of its kernel as one
``[slots, ...]`` tensor: weights, block scales and per-expert scalars. Once
those tensors hold their final kernel layout for the ``H`` resident experts,
``bind_live_weight_planes`` describes them as a ``BundleLayout``, allocates the
layer's ``H + S`` arena through its layer handle, copies the resident rows into
the arena and rebinds each tensor to its ``[H + S, ...]`` arena view. Rows
``[H, H + S)`` are the helper slots that the HALO-Q copy fills on every
forward; they start as copies of resident rows, so a kernel that reads them
before the first plan (autotuning) sees valid weights.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import torch

__all__ = ["bind_live_weight_planes", "live_weight_bundle"]

# Large planes back TMA descriptors and vectorized loads of the GEMM kernels;
# the arena must give them the alignment a caching-allocator block has.
_LARGE_PLANE_POINTER_ALIGNMENT = 256
_BUNDLE_ALIGNMENT = 512


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def live_weight_bundle(
    planes: Sequence[Tuple[str, torch.Tensor]],
    *,
    hidden: int,
    intermediate: int,
) -> Any:
    """Describe ``[H, ...]`` per-expert tensors as one live-weight bundle.

    Each plane's per-slot view is the tensor's trailing shape, dense; its byte
    size is the per-slot byte stride of the arena. Plane order is the given
    order, which every EP rank must agree on.
    """
    from ...kernels.megamoe_scheduler.sami.geometry import BundleLayout, Plane, PlaneViewLayout

    if not planes:
        raise ValueError("a live-weight bundle needs at least one plane")
    names = [name for name, _ in planes]
    if len(set(names)) != len(names):
        raise ValueError(f"live-weight plane names must be unique; got {names}")
    resident = int(planes[0][1].shape[0])
    built = []
    cursor = 0
    for name, tensor in planes:
        if not isinstance(tensor, torch.Tensor) or tensor.dim() < 1:
            raise TypeError(f"live-weight plane {name!r} must be a [slots, ...] tensor")
        if int(tensor.shape[0]) != resident:
            raise ValueError(
                f"live-weight plane {name!r} has {tensor.shape[0]} rows; expected {resident}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"live-weight plane {name!r} must be contiguous")
        shape = tuple(int(extent) for extent in tensor.shape[1:])
        element_size = int(tensor.element_size())
        steps = []
        running = 1
        for extent in reversed(shape):
            steps.append(running)
            running *= extent
        stride = tuple(reversed(steps))
        alignment = _LARGE_PLANE_POINTER_ALIGNMENT if shape else element_size
        view = PlaneViewLayout(
            shape=shape,
            stride=stride,
            dtype=_dtype_name(tensor.dtype),
            element_size=element_size,
            pointer_alignment=alignment,
        )
        if shape and view.nbytes % 16:
            raise ValueError(
                f"live-weight plane {name!r} spans {view.nbytes} bytes per slot; the "
                "kernels need a multiple of 16"
            )
        cursor = _align_up(cursor, alignment)
        built.append(Plane(name, cursor, view.nbytes, alignment, storage=view, kernel=view))
        cursor += view.nbytes
    return BundleLayout(
        hidden=int(hidden),
        intermediate=int(intermediate),
        planes=tuple(built),
        total_bytes=sum(plane.nbytes for plane in built),
        stride_bytes=_align_up(cursor, _BUNDLE_ALIGNMENT),
        alignment=_BUNDLE_ALIGNMENT,
    )


def _rebind(module: torch.nn.Module, name: str, view: torch.Tensor) -> None:
    current = getattr(module, name)
    if isinstance(current, torch.nn.Parameter):
        # Keep the Parameter object; only its storage moves to the arena.
        current.data = view
    elif name in module._buffers:
        module._buffers[name] = view
    else:
        setattr(module, name, view)


def bind_live_weight_planes(
    module: torch.nn.Module,
    plane_names: Sequence[str],
    *,
    hidden: int,
    intermediate: int,
    ep_process_group: Optional[Any] = None,
) -> Any:
    """Move ``module``'s per-expert tensors onto its layer's ``H + S`` arena.

    Collective across the EP group: every rank calls it for the same layers in
    the same order with the same planes. ``module`` carries the per-layer
    dynamic EPLB layer handle (``layer_load_balancer``), its ``mapping``,
    ``layer_idx`` and ``expert_size_per_partition`` (``H``). Returns the
    arena; afterwards every named tensor is a ``[H + S, ...]`` arena view
    whose first ``H`` rows equal the tensor it replaced.
    """
    layer = module.layer_load_balancer
    resident = int(module.expert_size_per_partition)
    if resident != int(layer.resident_experts_per_rank):
        raise ValueError(
            f"backend holds {resident} resident experts per rank; the EPLB layer "
            f"expects {layer.resident_experts_per_rank}"
        )
    planes = [(name, getattr(module, name)) for name in plane_names]
    for name, tensor in planes:
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
            raise TypeError(f"live-weight plane {name!r} must be a CUDA tensor")
    bundle = live_weight_bundle(planes, hidden=hidden, intermediate=intermediate)
    device = planes[0][1].device.index
    arena = layer.allocate_arena(
        bundle=bundle,
        ep_process_group=ep_process_group,
        mapping=module.mapping,
        device=int(device if device is not None else torch.cuda.current_device()),
        layer_idx=module.layer_idx,
    )
    helpers = int(arena.helper_slots)
    with torch.no_grad():
        for (name, source), alias in zip(planes, arena.tekit_alias_views):
            if tuple(alias.shape[1:]) != tuple(source.shape[1:]) or alias.dtype != source.dtype:
                raise RuntimeError(f"arena view of {name!r} does not match its tensor")
            alias[:resident].copy_(source)
            for helper in range(helpers):
                alias[resident + helper].copy_(source[helper % resident])
    # The sources are released below; their copies must have read them first.
    torch.cuda.current_stream().synchronize()
    for (name, _), alias in zip(planes, arena.tekit_alias_views):
        _rebind(module, name, alias)
    return arena

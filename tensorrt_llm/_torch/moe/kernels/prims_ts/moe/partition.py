# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Two-way Prims-TS MoE partitioning and prepared-weight sharding.

The caller owns weight allocation/localization, including placement in Rubin
uGPU locality domains, and two green-context streams. This module only
describes the execution contract, validates it, and materializes ordinary
contiguous shards from already prepared Prims-TS weights.

FC1 row sharding relies on two block-local permutations: gated rows are first
interleaved as ``gate0, up0, gate1, up1, ...`` and then shuffled independently
inside 16- or 32-row blocks.  Consequently, slicing a prepared FC1 at an
intermediate-partition boundary commutes with slicing the logical gate and up
halves *only when that boundary contains whole shuffle blocks*.  The same rule
applies to FC2 output-row shards.  K sharding commutes with row permutation, but
packed data and scale-factor groups must not be split.
"""

from __future__ import annotations

import contextlib
import functools
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

import torch

from flashinfer.tllm_enums import WeightLayout
from flashinfer.utils import get_compute_capability


PartitionStrategy = Literal["tp_n_tp_n", "tp_n_tp_k"]
PreparedWeightLayoutKind = Literal["major_k", "block_major_k"]
ScaleFactorLayout = Literal["r128c4"]

_PARTITION_COUNT = 2
_SUPPORTED_STRATEGIES = ("tp_n_tp_n", "tp_n_tp_k")
_SCALE_POINTER_ALIGNMENT_BYTES = 32


def _require_green_context_streams(streams: Sequence[torch.Stream]) -> None:
    """Reject CUDA streams that are not owned by green contexts."""

    try:
        import cuda.bindings.driver as cuda_drv

        from flashinfer.cuda_utils import checkCudaErrors
    except ImportError as exc:
        raise RuntimeError(
            "Partitioned Prims-TS MoE requires cuda-python>=12.9 to validate "
            "green-context streams"
        ) from exc
    if not hasattr(cuda_drv, "cuStreamGetGreenCtx"):
        raise RuntimeError(
            "Partitioned Prims-TS MoE requires cuda-python>=12.9 with "
            "cuStreamGetGreenCtx support"
        )

    for partition_id, torch_stream in enumerate(streams):
        stream = cuda_drv.CUstream(int(torch_stream.cuda_stream))
        try:
            green_ctx = checkCudaErrors(cuda_drv.cuStreamGetGreenCtx(stream))
        except RuntimeError as exc:
            raise ValueError(
                f"partition stream {partition_id} must belong to a green context "
                "on its intended memory-local SM partition"
            ) from exc
        if green_ctx is None or not int(green_ctx):
            raise ValueError(
                f"partition stream {partition_id} must belong to a green context "
                "on its intended memory-local SM partition"
            )


def require_partition_pair(name: str, seq: Sequence[Any] | None) -> None:
    """Reject a per-partition sequence that does not hold exactly two shards."""

    if seq is not None and len(seq) != _PARTITION_COUNT:
        raise ValueError(
            f"{name} must contain exactly {_PARTITION_COUNT} partition shards, "
            f"got {len(seq)}"
        )


def validate_partition_arch(tensor: torch.Tensor) -> None:
    """Reject CUDA devices that cannot run partitioned Prims-TS MoE."""

    if tensor.device.type != "cuda":
        return
    major, minor = get_compute_capability(tensor.device)
    if (major, minor) != (10, 7):
        raise NotImplementedError(
            "Prims-TS partitioned MoE (uGPU locality domains) is only supported on "
            f"SM107 (Rubin); got SM{major}{minor}"
        )


@dataclass(frozen=True)
class PreparedWeightLayout:
    """Physical description of one prepared Prims-TS weight tensor.

    ``storage_bits`` is the number of physical payload bits per logical K
    element.  It is 4 for packed FP4, 8 for FP8, and 16 for BF16/FP16.
    MajorK storage is ``[expert, Mn, K_storage]``.  BlockMajorK storage is
    ``[expert, K_bytes / block_major_k_bytes, Mn,
    block_major_k_bytes / element_size]``.

    When ``scale_layout`` is set, the associated scale tensor is flattened
    R128c4 MMA storage.  ``scale_k_group_size`` gives the logical K
    elements represented by one one-byte scale factor.  With no scale layout,
    an optional opaque per-tensor/per-expert scale is replicated into each
    shard instead of being interpreted.
    """

    kind: PreparedWeightLayoutKind
    storage_bits: int
    weight_dtype: torch.dtype | None = None
    block_major_k_bytes: int | None = None
    scale_layout: ScaleFactorLayout | None = None
    scale_k_group_size: int | None = None
    scale_dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("major_k", "block_major_k"):
            raise ValueError(f"Unsupported prepared weight layout {self.kind!r}")
        if self.storage_bits not in (4, 8, 16):
            raise ValueError(
                f"storage_bits must be one of 4, 8, or 16, got {self.storage_bits}"
            )
        if self.kind == "block_major_k":
            if self.block_major_k_bytes is None or self.block_major_k_bytes <= 0:
                raise ValueError(
                    "block_major_k_bytes is required for BlockMajorK storage"
                )
        elif self.block_major_k_bytes is not None:
            raise ValueError(
                "block_major_k_bytes is only valid for BlockMajorK storage"
            )
        if (self.scale_layout is None) != (self.scale_k_group_size is None):
            raise ValueError(
                "scale_layout and scale_k_group_size must be specified together"
            )
        if self.scale_layout not in (None, "r128c4"):
            raise ValueError(f"Unsupported scale-factor layout {self.scale_layout!r}")
        if self.scale_k_group_size is not None and self.scale_k_group_size <= 0:
            raise ValueError(
                f"scale_k_group_size must be positive, got {self.scale_k_group_size}"
            )
        if self.scale_layout is None and self.scale_dtype is not None:
            raise ValueError("scale_dtype is only valid for block-scaled weights")


def prepared_weight_layouts(
    *,
    weight_dtype: torch.dtype,
    weight_layout: int | WeightLayout,
    use_shuffled_weight: bool,
    scale_k_group_size: int | None = None,
    scale_dtype: torch.dtype | None = None,
    fc1_block_major_k_bytes: int | None = None,
    fc2_block_major_k_bytes: int | None = None,
) -> tuple[PreparedWeightLayout, PreparedWeightLayout]:
    """Derive FC1/FC2 prepared layouts from the public kernel arguments.

    Callers preparing shards should use this helper rather than duplicating the
    runtime mapping. Block-scaled Prims-TS weights use R128c4 scale storage;
    per-tensor formats leave ``scale_k_group_size`` unset. ``torch.uint8``
    weights are packed FP4 (two elements per byte); FP8 weights must use a
    ``float8`` dtype.
    """

    try:
        layout = WeightLayout(int(weight_layout))
    except ValueError as exc:
        raise ValueError(
            f"Unsupported prepared weight layout {weight_layout!r}"
        ) from exc
    if layout == WeightLayout.MajorMn:
        raise ValueError("Partitioned Prims-TS MoE does not support MajorMn weights")
    if layout == WeightLayout.MajorK and not use_shuffled_weight:
        raise ValueError("Partitioned MajorK weights must use shuffled storage")

    if weight_dtype in (torch.bfloat16, torch.float16):
        storage_bits = 16
    elif weight_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        storage_bits = 8
    elif weight_dtype == torch.uint8:
        storage_bits = 4
    else:
        raise ValueError(f"Unsupported prepared weight dtype {weight_dtype}")

    kind: PreparedWeightLayoutKind = (
        "block_major_k" if layout == WeightLayout.BlockMajorK else "major_k"
    )
    if kind == "block_major_k":
        if fc1_block_major_k_bytes is None or fc2_block_major_k_bytes is None:
            raise ValueError(
                "BlockMajorK layouts require the prepared FC1 and FC2 block sizes"
            )
    elif fc1_block_major_k_bytes is not None or fc2_block_major_k_bytes is not None:
        raise ValueError("BlockMajorK block sizes are only valid for BlockMajorK")

    def make_layout(block_bytes: int | None) -> PreparedWeightLayout:
        return PreparedWeightLayout(
            kind=kind,
            storage_bits=storage_bits,
            weight_dtype=weight_dtype,
            block_major_k_bytes=block_bytes,
            scale_layout="r128c4" if scale_k_group_size is not None else None,
            scale_k_group_size=scale_k_group_size,
            scale_dtype=scale_dtype,
        )

    return make_layout(fc1_block_major_k_bytes), make_layout(fc2_block_major_k_bytes)


def prepared_block_major_k_bytes(
    weight: torch.Tensor, weight_layout: int | WeightLayout
) -> int | None:
    """Return the physical K-block byte width encoded by a prepared tensor."""

    layout = WeightLayout(int(weight_layout))
    if layout != WeightLayout.BlockMajorK:
        return None
    if weight.ndim != 4:
        raise ValueError(
            "BlockMajorK prepared weights must have shape "
            "[experts, K_blocks, M, block_elements]"
        )
    block_bytes = int(weight.shape[-1]) * weight.element_size()
    if block_bytes <= 0:
        raise ValueError("BlockMajorK prepared weight block must be non-empty")
    return block_bytes


@dataclass(frozen=True)
class MoeShardedWeights:
    """Prepared tensors grouped by partition for direct op consumption."""

    fc1: list[torch.Tensor]
    fc2: list[torch.Tensor]
    fc1_scale: list[torch.Tensor] | None = None
    fc2_scale: list[torch.Tensor] | None = None


@dataclass(eq=False)
class MoePartitionResources:
    """Caller-owned streams and reusable synchronization events.

    Construct this handle before CUDA graph capture and retain it for the
    lifetime of every eager launch or captured graph that uses it. Weights and
    strategy are per-op inputs and deliberately do not live on this handle.
    """

    streams: Sequence[torch.Stream]
    fork_join: CudaForkJoin | None = field(
        init=False, default=None, repr=False, compare=False
    )
    _stream_topology: tuple | None = field(
        init=False, default=None, repr=False, compare=False
    )
    _launch_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        # Derived stream metadata is cached on the handle.
        self.streams = tuple(self.streams)
        if len(self.streams) != _PARTITION_COUNT:
            raise ValueError(
                f"Expected exactly two partition streams, got {len(self.streams)}"
            )
        # Events must exist before CUDA graph capture. Keeping them on the
        # long-lived handle also guarantees their lifetime covers eager launches
        # and every replay of a captured graph.
        cuda_streams = all(
            torch.device(getattr(stream, "device", "cpu")).type == "cuda"
            for stream in self.streams
        )
        if cuda_streams:
            _require_green_context_streams(self.streams)
        self.fork_join = CudaForkJoin(self.streams) if cuda_streams else None

    @contextlib.contextmanager
    def launch_guard(self):
        """Serialize host enqueue that records reusable events.

        The host lock prevents two Python threads from interleaving event
        records. Independently concurrent CUDA graph replays must use distinct
        resource handles because their event records execute asynchronously.
        """

        with self._launch_lock:
            yield


class CudaForkJoin:
    """Reusable two-stream CUDA fork/join events.

    Construct this object before CUDA graph capture and keep it alive for at
    least as long as any eager work or captured graph using it.  The object
    retains all synchronization events, which avoids relying on temporary
    Python event lifetimes. Streams remain caller-owned.

    ``fork`` records readiness on the caller stream and makes both partition
    streams wait.  Callers may then loop over
    ``with torch.cuda.stream(partition_stream):`` to enqueue each launch.
    ``join`` records completion explicitly on each partition stream, so it is
    unaffected by Python stream-context restoration, and makes the caller
    stream wait for both. No tensor ``record_stream`` calls are performed.

    These ordinary events impose full dependencies during eager execution.
    During graph capture, a PDL-attributed consumer launch turns incoming
    kernel dependencies carried through event waits into programmatic edges.
    Such consumers must wait before their first predecessor-dependent access.
    """

    def __init__(self, streams: Sequence[torch.Stream]) -> None:
        if len(streams) != _PARTITION_COUNT:
            raise ValueError(
                f"CudaForkJoin requires exactly two streams, got {len(streams)}"
            )
        if torch.device(streams[0].device) != torch.device(streams[1].device):
            raise ValueError("Partition streams must be on the same CUDA device")
        self.streams = streams
        self._device = torch.device(streams[0].device)
        self._fork_event = torch.cuda.Event()
        self._done_events = (torch.cuda.Event(), torch.cuda.Event())

    def fork(self, caller_stream: torch.Stream | None = None) -> None:
        """Make both partition streams wait for prior caller-stream work."""

        if caller_stream is None:
            caller_stream = torch.cuda.current_stream(self._device)
        if torch.device(caller_stream.device) != self._device:
            raise ValueError("Caller and partition streams must use the same device")
        self._fork_event.record(caller_stream)
        for stream in self.streams:
            stream.wait_event(self._fork_event)

    def join(self, caller_stream: torch.Stream | None = None) -> None:
        """Make the caller stream wait for both partition streams."""

        if caller_stream is None:
            caller_stream = torch.cuda.current_stream(self._device)
        if torch.device(caller_stream.device) != self._device:
            raise ValueError("Caller and partition streams must use the same device")
        for stream, event in zip(self.streams, self._done_events, strict=True):
            event.record(stream)
            caller_stream.wait_event(event)


def _shuffle_block_size(epilogue_tile_m: int) -> int:
    if epilogue_tile_m <= 0:
        raise ValueError(f"epilogue_tile_m must be positive, got {epilogue_tile_m}")
    return 32 if epilogue_tile_m % 128 == 0 else 16


def _scale_rows_per_block(layout: ScaleFactorLayout) -> int:
    if layout != "r128c4":
        raise ValueError(f"Unsupported weight scale-factor layout {layout!r}")
    return 128


def _scale_storage_numel(
    *,
    num_experts: int,
    logical_rows: int,
    logical_k: int,
    layout: ScaleFactorLayout,
    k_group_size: int,
) -> int:
    if logical_k % k_group_size:
        raise ValueError(
            f"logical K={logical_k} must be divisible by scale group {k_group_size}"
        )
    sf_cols = logical_k // k_group_size
    if sf_cols % 4:
        raise ValueError(
            f"K scale groups={sf_cols} must be divisible by the four-SF MMA pack"
        )
    rows_per_block = _scale_rows_per_block(layout)
    if logical_rows % rows_per_block:
        raise ValueError(
            f"weight scale rows per expert must be {rows_per_block}-row aligned"
        )
    row_blocks = num_experts * (logical_rows // rows_per_block)
    return row_blocks * (sf_cols // 4) * rows_per_block * 4


def _validate_prepared_weight(
    name: str,
    tensor: torch.Tensor,
    *,
    num_experts: int,
    logical_rows: int,
    logical_k: int,
    layout: PreparedWeightLayout,
    pointer_alignment_bytes: int,
) -> None:
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if layout.weight_dtype is not None and tensor.dtype != layout.weight_dtype:
        raise ValueError(
            f"{name} must have dtype {layout.weight_dtype}, got {tensor.dtype}"
        )
    if tensor.data_ptr() % pointer_alignment_bytes:
        raise ValueError(
            f"{name} data pointer must be {pointer_alignment_bytes}-byte aligned"
        )
    element_bits = tensor.element_size() * 8
    k_bits = logical_k * layout.storage_bits
    if k_bits % element_bits:
        raise ValueError(
            f"{name} logical K storage is not integral for dtype {tensor.dtype}"
        )
    if layout.kind == "major_k":
        expected = (num_experts, logical_rows, k_bits // element_bits)
    else:
        block_bytes = int(layout.block_major_k_bytes)
        k_bytes = k_bits // 8
        if k_bits % 8 or k_bytes % block_bytes:
            raise ValueError(
                f"{name} logical K storage must contain whole "
                f"{block_bytes}-byte BlockMajorK blocks"
            )
        if block_bytes % tensor.element_size():
            raise ValueError(
                f"{name} element size does not divide BlockMajorK block bytes"
            )
        expected = (
            num_experts,
            k_bytes // block_bytes,
            logical_rows,
            block_bytes // tensor.element_size(),
        )
    if tuple(map(int, tensor.shape)) != expected:
        raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {expected}")


def _partition_geometry(
    strategy: PartitionStrategy,
    partition_id: int,
    *,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Return FC1-row, FC2-row, and FC2-K half-open logical ranges."""

    fc1_rows = (
        partition_id * intermediate_size,
        (partition_id + 1) * intermediate_size,
    )
    if strategy == "tp_n_tp_n":
        fc2_rows = (
            partition_id * (hidden_size // 2),
            (partition_id + 1) * (hidden_size // 2),
        )
        fc2_k = (0, intermediate_size)
    else:
        fc2_rows = (0, hidden_size)
        fc2_k = (
            partition_id * (intermediate_size // 2),
            (partition_id + 1) * (intermediate_size // 2),
        )
    return fc1_rows, fc2_rows, fc2_k


def _shard_weight(
    tensor: torch.Tensor,
    *,
    row_range: tuple[int, int],
    k_range: tuple[int, int],
    layout: PreparedWeightLayout,
) -> torch.Tensor:
    row_start, row_stop = row_range
    k_start, k_stop = k_range
    if layout.kind == "major_k":
        element_bits = tensor.element_size() * 8
        storage_start_bits = k_start * layout.storage_bits
        storage_stop_bits = k_stop * layout.storage_bits
        if storage_start_bits % element_bits or storage_stop_bits % element_bits:
            raise ValueError("MajorK K partition cuts through a packed storage element")
        return tensor[
            :,
            row_start:row_stop,
            storage_start_bits // element_bits : storage_stop_bits // element_bits,
        ]

    block_bytes = int(layout.block_major_k_bytes)
    logical_k_per_block = block_bytes * 8 // layout.storage_bits
    if k_start % logical_k_per_block or k_stop % logical_k_per_block:
        raise ValueError(
            "BlockMajorK K partition cuts through a physical K block of "
            f"{logical_k_per_block} logical elements"
        )
    return tensor[
        :,
        k_start // logical_k_per_block : k_stop // logical_k_per_block,
        row_start:row_stop,
        :,
    ]


def _shard_scale(
    scale: torch.Tensor | None,
    *,
    num_experts: int,
    full_rows: int,
    full_k: int,
    row_range: tuple[int, int],
    k_range: tuple[int, int],
    layout: PreparedWeightLayout,
) -> torch.Tensor | None:
    if scale is None:
        return None
    if layout.scale_layout is None:
        return scale
    k_group_size = layout.scale_k_group_size
    if k_group_size is None:
        raise ValueError("Block-scale layout requires scale_k_group_size")
    expected = _scale_storage_numel(
        num_experts=num_experts,
        logical_rows=full_rows,
        logical_k=full_k,
        layout=layout.scale_layout,
        k_group_size=k_group_size,
    )
    if not scale.is_contiguous() or scale.numel() != expected:
        raise ValueError(
            "Prepared block-scale tensor must be contiguous flattened MMA "
            f"storage with {expected} elements, got shape {tuple(scale.shape)}"
        )
    if layout.scale_dtype is not None and scale.dtype != layout.scale_dtype:
        raise ValueError(
            f"Prepared block-scale tensor must have dtype {layout.scale_dtype}, "
            f"got {scale.dtype}"
        )
    row_start, row_stop = row_range
    k_start, k_stop = k_range
    if k_start % k_group_size or k_stop % k_group_size:
        raise ValueError(
            "K partition cuts through a block-scale group of "
            f"{k_group_size} logical elements"
        )
    sf_start = k_start // k_group_size
    sf_stop = k_stop // k_group_size
    if row_start % 128 or row_stop % 128:
        raise ValueError("R128c4 row shard must contain whole 128-row blocks")
    if sf_start % 4 or sf_stop % 4:
        raise ValueError("R128c4 K shard must contain whole four-scale blocks")
    row_blocks = full_rows // 128
    col_blocks = (full_k // k_group_size) // 4
    return scale.view(num_experts, row_blocks, col_blocks, 32, 4, 4)[
        :,
        row_start // 128 : row_stop // 128,
        sf_start // 4 : sf_stop // 4,
        :,
        :,
        :,
    ]


def shard_prepared_moe_weights(
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    *,
    strategy: PartitionStrategy,
    hidden_size: int,
    intermediate_size: int,
    fc1_layout: PreparedWeightLayout,
    fc2_layout: PreparedWeightLayout,
    fc1_scale: torch.Tensor | None = None,
    fc2_scale: torch.Tensor | None = None,
    fc1_epilogue_tile_m: int = 128,
    fc2_epilogue_tile_m: int = 128,
    alloc: Callable[[int, tuple[int, ...], torch.dtype], torch.Tensor] | None = None,
) -> MoeShardedWeights:
    """Materialize two contiguous shards from prepared gated-MoE weights.

    FC1 input is prepared storage for logical ``[E, 2I, H]``.  FC2 input is
    prepared storage for logical ``[E, H, I]``.  MajorK packed FP4 therefore
    has a final storage dimension of ``K / 2`` bytes, while BlockMajorK moves
    that packed K dimension to physical dimension 1.

    ``alloc`` receives ``(partition_id, shape, dtype)`` for each materialized
    tensor. It can return storage from a caller-managed locality pool; the
    shard is copied directly into that storage. Block-scale storage returned by
    ``alloc`` must be 32-byte aligned. When omitted, ordinary
    contiguous tensors are returned on the source device. Use
    :func:`prepared_weight_layouts` to derive ``fc1_layout`` and ``fc2_layout``
    from the same public arguments used by the runtime op.
    """

    if strategy not in _SUPPORTED_STRATEGIES:
        raise ValueError(f"Unsupported MoE partition strategy {strategy!r}")
    if hidden_size <= 0 or intermediate_size <= 0:
        raise ValueError("hidden_size and intermediate_size must be positive")
    num_experts = int(fc1_weight.shape[0])
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if hidden_size % 2 or intermediate_size % 2:
        raise ValueError(
            "Two-way MoE partitioning requires hidden_size and "
            "intermediate_size divisible by two"
        )
    _validate_prepared_weight(
        "fc1_weight",
        fc1_weight,
        num_experts=num_experts,
        logical_rows=2 * intermediate_size,
        logical_k=hidden_size,
        layout=fc1_layout,
        pointer_alignment_bytes=16,
    )
    _validate_prepared_weight(
        "fc2_weight",
        fc2_weight,
        num_experts=num_experts,
        logical_rows=hidden_size,
        logical_k=intermediate_size,
        layout=fc2_layout,
        pointer_alignment_bytes=16,
    )
    if fc2_weight.device != fc1_weight.device:
        raise ValueError("FC1 and FC2 prepared weights must be on the same device")
    for name, scale in (("fc1_scale", fc1_scale), ("fc2_scale", fc2_scale)):
        if scale is not None:
            if scale.device != fc1_weight.device:
                raise ValueError(f"{name} must be on the weight device")
            if not scale.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
            if scale.data_ptr() % _SCALE_POINTER_ALIGNMENT_BYTES:
                raise ValueError(
                    f"{name} data pointer must be "
                    f"{_SCALE_POINTER_ALIGNMENT_BYTES}-byte aligned"
                )

    fc1_shuffle = _shuffle_block_size(fc1_epilogue_tile_m)
    if intermediate_size % fc1_shuffle:
        raise ValueError(
            "FC1 prepared-row partition boundary must contain whole "
            f"{fc1_shuffle}-row shuffle blocks"
        )
    if strategy == "tp_n_tp_n":
        fc2_shuffle = _shuffle_block_size(fc2_epilogue_tile_m)
        if (hidden_size // 2) % fc2_shuffle:
            raise ValueError(
                "FC2 prepared-row partition boundary must contain whole "
                f"{fc2_shuffle}-row shuffle blocks"
            )
    for name, scale, layout, shard_rows in (
        ("fc1_scale", fc1_scale, fc1_layout, intermediate_size),
        (
            "fc2_scale",
            fc2_scale,
            fc2_layout,
            hidden_size // 2 if strategy == "tp_n_tp_n" else None,
        ),
    ):
        if (
            scale is not None
            and shard_rows is not None
            and layout.scale_layout is not None
        ):
            scale_rows = _scale_rows_per_block(layout.scale_layout)
            if shard_rows % scale_rows:
                raise ValueError(
                    f"{name} row shard has {shard_rows} rows per expert; "
                    f"{layout.scale_layout} requires {scale_rows}-row alignment"
                )

    def place(
        partition_id: int,
        tensor: torch.Tensor | None,
        *,
        is_block_scale: bool = False,
    ) -> torch.Tensor | None:
        if tensor is None:
            return tensor
        output_shape = (tensor.numel(),) if is_block_scale else tuple(tensor.shape)
        if alloc is None:
            return tensor.clone(memory_format=torch.contiguous_format).reshape(
                output_shape
            )
        target = alloc(partition_id, output_shape, tensor.dtype)
        if tuple(target.shape) != output_shape or target.dtype != tensor.dtype:
            raise ValueError(
                "partition allocator returned shape/dtype "
                f"{tuple(target.shape)}/{target.dtype}, expected "
                f"{output_shape}/{tensor.dtype}"
            )
        if not target.is_contiguous():
            raise ValueError("partition allocator must return contiguous storage")
        if is_block_scale and target.data_ptr() % _SCALE_POINTER_ALIGNMENT_BYTES:
            raise ValueError(
                "partition allocator must return "
                f"{_SCALE_POINTER_ALIGNMENT_BYTES}-byte aligned block-scale storage"
            )
        target.view(tensor.shape).copy_(tensor)
        return target

    fc1_shards: list[torch.Tensor] = []
    fc2_shards: list[torch.Tensor] = []
    fc1_scale_shards: list[torch.Tensor] = []
    fc2_scale_shards: list[torch.Tensor] = []
    for partition_id in range(_PARTITION_COUNT):
        fc1_rows, fc2_rows, fc2_k = _partition_geometry(
            strategy,
            partition_id,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        fc1_shard = place(
            partition_id,
            _shard_weight(
                fc1_weight,
                row_range=fc1_rows,
                k_range=(0, hidden_size),
                layout=fc1_layout,
            ),
        )
        fc2_shard = place(
            partition_id,
            _shard_weight(
                fc2_weight,
                row_range=fc2_rows,
                k_range=fc2_k,
                layout=fc2_layout,
            ),
        )
        assert fc1_shard is not None and fc2_shard is not None
        fc1_shards.append(fc1_shard)
        fc2_shards.append(fc2_shard)
        fc1_scale_shard = place(
            partition_id,
            _shard_scale(
                fc1_scale,
                num_experts=num_experts,
                full_rows=2 * intermediate_size,
                full_k=hidden_size,
                row_range=fc1_rows,
                k_range=(0, hidden_size),
                layout=fc1_layout,
            ),
            is_block_scale=fc1_layout.scale_layout is not None,
        )
        fc2_scale_shard = place(
            partition_id,
            _shard_scale(
                fc2_scale,
                num_experts=num_experts,
                full_rows=hidden_size,
                full_k=intermediate_size,
                row_range=fc2_rows,
                k_range=fc2_k,
                layout=fc2_layout,
            ),
            is_block_scale=fc2_layout.scale_layout is not None,
        )
        if fc1_scale_shard is not None:
            fc1_scale_shards.append(fc1_scale_shard)
        if fc2_scale_shard is not None:
            fc2_scale_shards.append(fc2_scale_shard)

    return MoeShardedWeights(
        fc1=fc1_shards,
        fc2=fc2_shards,
        fc1_scale=fc1_scale_shards if fc1_scale is not None else None,
        fc2_scale=fc2_scale_shards if fc2_scale is not None else None,
    )


def validate_partition_shape(
    resources: MoePartitionResources,
    *,
    strategy: PartitionStrategy,
    fc1_weights: Sequence[torch.Tensor],
    fc2_weights: Sequence[torch.Tensor],
    fc1_scales: Sequence[torch.Tensor] | None,
    fc2_scales: Sequence[torch.Tensor] | None,
    fc1_layout: PreparedWeightLayout,
    fc2_layout: PreparedWeightLayout,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    partition_alignment_elements: int = 32,
    pointer_alignment_bytes: int = 16,
) -> None:
    """Validate the invariant shape/layout portion of a partition contract."""

    if strategy not in _SUPPORTED_STRATEGIES:
        raise ValueError(f"Unsupported MoE partition strategy {strategy!r}")

    if len(fc1_weights) != _PARTITION_COUNT or len(fc2_weights) != _PARTITION_COUNT:
        raise ValueError("Expected exactly two FC1 and two FC2 partition weight shards")
    if fc1_scales is not None and len(fc1_scales) != _PARTITION_COUNT:
        raise ValueError("Expected exactly two FC1 scale shards")
    if fc2_scales is not None and len(fc2_scales) != _PARTITION_COUNT:
        raise ValueError("Expected exactly two FC2 scale shards")

    def require_uniform_dtype(
        name: str,
        tensors: Sequence[torch.Tensor] | None,
        expected: torch.dtype | None,
    ) -> torch.dtype | None:
        if tensors is None:
            return None
        resolved = expected if expected is not None else tensors[0].dtype
        for partition_id, tensor in enumerate(tensors):
            if tensor.dtype != resolved:
                raise ValueError(
                    f"{name}[{partition_id}] must have dtype {resolved}, "
                    f"got {tensor.dtype}"
                )
        return resolved

    fc1_weight_dtype = require_uniform_dtype(
        "fc1_weights", fc1_weights, fc1_layout.weight_dtype
    )
    fc2_weight_dtype = require_uniform_dtype(
        "fc2_weights", fc2_weights, fc2_layout.weight_dtype
    )
    if fc1_weight_dtype != fc2_weight_dtype:
        raise ValueError("FC1 and FC2 weight shards must share a dtype")
    fc1_scale_dtype = require_uniform_dtype(
        "fc1_scales", fc1_scales, fc1_layout.scale_dtype
    )
    fc2_scale_dtype = require_uniform_dtype(
        "fc2_scales", fc2_scales, fc2_layout.scale_dtype
    )
    if fc1_scale_dtype != fc2_scale_dtype:
        raise ValueError("FC1 and FC2 scale shards must share a dtype")

    if hidden_size <= 0 or intermediate_size <= 0:
        raise ValueError("hidden_size and intermediate_size must be positive")
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if hidden_size % 2 or intermediate_size % 2:
        raise ValueError(
            "Two-way MoE partitioning requires hidden_size and "
            "intermediate_size divisible by two"
        )
    if partition_alignment_elements <= 0 or pointer_alignment_bytes <= 0:
        raise ValueError("Alignment requirements must be positive")
    if (intermediate_size // 2) % partition_alignment_elements:
        raise ValueError(f"I/2 must be {partition_alignment_elements}-element aligned")
    hidden_partition = hidden_size // 2 if strategy == "tp_n_tp_n" else hidden_size
    if hidden_partition % partition_alignment_elements:
        raise ValueError(
            f"FC2 output rows must be {partition_alignment_elements}-element aligned"
        )

    stream_device = torch.device(resources.streams[0].device)
    if torch.device(resources.streams[1].device) != stream_device:
        raise ValueError("Partition streams must be on the same device")
    for partition_id in range(_PARTITION_COUNT):
        fc1_rows, fc2_rows, fc2_k = _partition_geometry(
            strategy,
            partition_id,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        tensors = (
            ("fc1_weight", fc1_weights[partition_id]),
            ("fc2_weight", fc2_weights[partition_id]),
            (
                "fc1_scale",
                fc1_scales[partition_id] if fc1_scales is not None else None,
            ),
            (
                "fc2_scale",
                fc2_scales[partition_id] if fc2_scales is not None else None,
            ),
        )
        for name, tensor in tensors:
            if tensor is None:
                continue
            if tensor.device != stream_device:
                raise ValueError(
                    f"Partition {partition_id} {name} is on {tensor.device}, "
                    f"expected stream device {stream_device}"
                )
            if tensor.data_ptr() % pointer_alignment_bytes:
                raise ValueError(
                    f"Partition {partition_id} {name} pointer must be "
                    f"{pointer_alignment_bytes}-byte aligned"
                )
            if not tensor.is_contiguous():
                raise ValueError(f"Partition {partition_id} {name} must be contiguous")

        _validate_prepared_weight(
            f"partition {partition_id} fc1_weight",
            fc1_weights[partition_id],
            num_experts=num_experts,
            logical_rows=fc1_rows[1] - fc1_rows[0],
            logical_k=hidden_size,
            layout=fc1_layout,
            pointer_alignment_bytes=pointer_alignment_bytes,
        )
        _validate_prepared_weight(
            f"partition {partition_id} fc2_weight",
            fc2_weights[partition_id],
            num_experts=num_experts,
            logical_rows=fc2_rows[1] - fc2_rows[0],
            logical_k=fc2_k[1] - fc2_k[0],
            layout=fc2_layout,
            pointer_alignment_bytes=pointer_alignment_bytes,
        )

        for name, scale, layout, rows, logical_k in (
            (
                "fc1_scale",
                fc1_scales[partition_id] if fc1_scales is not None else None,
                fc1_layout,
                fc1_rows[1] - fc1_rows[0],
                hidden_size,
            ),
            (
                "fc2_scale",
                fc2_scales[partition_id] if fc2_scales is not None else None,
                fc2_layout,
                fc2_rows[1] - fc2_rows[0],
                fc2_k[1] - fc2_k[0],
            ),
        ):
            if scale is None:
                continue
            if layout.scale_dtype is not None and scale.dtype != layout.scale_dtype:
                raise ValueError(
                    f"{name} must have dtype {layout.scale_dtype}, got {scale.dtype}"
                )
            if layout.scale_layout is None:
                continue
            k_group_size = layout.scale_k_group_size
            if k_group_size is None:
                raise ValueError(f"{name} block-scale metadata is incomplete")
            scale_rows = _scale_rows_per_block(layout.scale_layout)
            row_partitioned = name == "fc1_scale" or strategy == "tp_n_tp_n"
            if row_partitioned and rows % scale_rows:
                raise ValueError(
                    f"{name} row shard has {rows} rows per expert; "
                    f"{layout.scale_layout} requires {scale_rows}-row alignment"
                )
            expected = _scale_storage_numel(
                num_experts=num_experts,
                logical_rows=rows,
                logical_k=logical_k,
                layout=layout.scale_layout,
                k_group_size=k_group_size,
            )
            if scale.element_size() != 1:
                raise ValueError(
                    "Prepared block-scale tensors must use one-byte elements"
                )
            scale_alignment = max(
                pointer_alignment_bytes, _SCALE_POINTER_ALIGNMENT_BYTES
            )
            if scale.data_ptr() % scale_alignment:
                raise ValueError(
                    f"{name} data pointer must be {scale_alignment}-byte aligned"
                )
            if scale.numel() != expected:
                raise ValueError(
                    "Prepared block-scale tensor must be contiguous flattened MMA "
                    f"storage with {expected} elements, got shape {tuple(scale.shape)}"
                )


@functools.cache
def validate_partition_tactic(
    *,
    strategy: PartitionStrategy,
    hidden_size: int,
    intermediate_size: int,
    fc1_tile_m: int,
    fc2_tile_m: int,
    fc1_epilogue_tile_m: int,
    fc2_epilogue_tile_m: int,
    fc1_is_swap_ab: bool,
    fc2_is_swap_ab: bool,
    fc1_partition_width: int | None = None,
    fc1_output_sf_block_size: int | None = None,
) -> None:
    """Validate and memoize tactic-dependent partition geometry."""

    if strategy not in _SUPPORTED_STRATEGIES:
        raise ValueError(f"Unsupported MoE partition strategy {strategy!r}")

    if not fc1_is_swap_ab or not fc2_is_swap_ab:
        raise ValueError("Partitioned MoE requires swapAB Prims-TS tactics")
    if fc1_tile_m <= 0 or fc2_tile_m <= 0:
        raise ValueError("Partitioned MoE tile sizes must be positive")
    if intermediate_size % fc1_tile_m:
        raise ValueError(
            f"partitioned FC1 width must be divisible by tile_m={fc1_tile_m}"
        )
    local_fc2_width = (
        hidden_size // _PARTITION_COUNT if strategy == "tp_n_tp_n" else hidden_size
    )
    if local_fc2_width % fc2_tile_m:
        raise ValueError(
            f"partitioned FC2 width must be divisible by tile_m={fc2_tile_m}"
        )
    if strategy == "tp_n_tp_n" and fc1_output_sf_block_size is not None:
        sf_block_width = int(fc1_output_sf_block_size)
        if fc1_partition_width is None or fc1_partition_width % sf_block_width:
            raise ValueError(
                "partitioned FC1 output offset must be divisible by its "
                f"SF block width {sf_block_width}"
            )

    fc1_shuffle = _shuffle_block_size(fc1_epilogue_tile_m)
    if intermediate_size % fc1_shuffle:
        raise ValueError(
            f"FC1 shard rows must align to {fc1_shuffle}-row shuffle blocks"
        )
    if strategy == "tp_n_tp_n":
        fc2_shuffle = _shuffle_block_size(fc2_epilogue_tile_m)
        if local_fc2_width % fc2_shuffle:
            raise ValueError(
                f"FC2 shard rows must align to {fc2_shuffle}-row shuffle blocks"
            )

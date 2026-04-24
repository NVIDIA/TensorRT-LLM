# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ADPiecewiseRunner: manages warmup -> capture -> replay for a single static CUDA graph segment.

Each static submodule in a piecewise-split model is wrapped in an ADPiecewiseRunner.
The runner's behavior is controlled by two class-level contexts set by the orchestrator
(PiecewiseCapturedGraph) before each split_gm forward pass:

  - `_current_phase`: determines execution mode ("warmup", "capture", or "replay")
  - `_current_num_tokens`: identifies which bucket entry to use

Phase semantics:
  1. WARMUP: Run the submodule eagerly (multiple iterations to stabilize allocator state).
  2. CAPTURE: Capture the submodule as a CUDA graph.  If this runner is linked to
     following dynamic ops (via set_dynamic_out_info), also allocate each dynamic
     op's output buffer *inside* the graph capture so it gets a deterministic
     address from the shared graph pool.
  3. REPLAY: Replay the captured CUDA graph.  Each DynamicOpWrapper retrieves its
     own pre-allocated output buffer via get_dynamic_out_buf(nt, submod_id).

MetadataWrapper: wraps metadata-prep dynamic ops whose outputs flow into
  subsequent CUDA-graph-captured static segments.  These ops (e.g.
  mamba_ssm_prepare_metadata) run eagerly every time and allocate fresh output
  tensors.  After torch.cuda.empty_cache() between capture and runtime, the
  allocator hands out different addresses, breaking the CUDA graph's recorded
  pointers.  The wrapper clones outputs during capture and copy_()s new data
  into those stable buffers on replay, preserving addresses at negligible cost
  (metadata tensors are tiny, e.g. two int32 [1024] ≈ 8 KB).

Memory:
  All model-level inputs (input_ids, position_ids, etc.) come from SequenceInfo's
  InputBuffer, which provides stable addresses across calls.  Dynamic op outputs are
  pre-allocated inside graph captures from a shared pool, giving deterministic addresses.
  Both static_output and dynamic_out_bufs are weak-ref'd via make_weak_ref, allowing the
  graph pool to recycle memory across layers.  No copy-back mechanism needed.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten

from .._compat import make_weak_ref
from ..utils.logger import ad_logger


@dataclass
class OutputInfo:
    """Metadata about a dynamic op's output, discovered during warmup."""

    shape: torch.Size
    dtype: torch.dtype
    device: torch.device


@dataclass
class SegmentEntry:
    """State for a single (num_tokens) configuration of a segment."""

    cuda_graph: Optional[torch.cuda.CUDAGraph] = None
    static_output: Any = None
    # Keyed by dynamic submodule index → pre-allocated output buffer
    dynamic_out_bufs: Dict[int, torch.Tensor] = field(default_factory=dict)
    input_addresses: List[Optional[int]] = field(default_factory=list)


_METADATA_PAD_ALIGN = 64
_MAMBA_METADATA_OP_NAME = "mamba_ssm_prepare_metadata"


def _contains_mamba_metadata_prep(submodule: nn.Module) -> bool:
    graph = getattr(submodule, "graph", None)
    if graph is None:
        return False
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        target = node.target
        name = target.name() if hasattr(target, "name") else str(target)
        if _MAMBA_METADATA_OP_NAME in name:
            return True
    return False


def _align_up(value: int, align: int) -> int:
    if align <= 1:
        return value
    return ((value + align - 1) // align) * align


def _alloc_stable_metadata_tensor(
    tensor: torch.Tensor,
    num_tokens: int,
    max_batch_size: int,
    pad_for_mamba_metadata: bool,
) -> torch.Tensor:
    """Allocate one stable metadata tensor with optional Mamba headroom."""
    if tensor.ndim == 0:
        return tensor.clone()

    # Only Mamba metadata prep has known runtime growth beyond capture shape
    # due to misaligned sequence boundaries.
    extra = max(0, max_batch_size - 1) if pad_for_mamba_metadata else 0

    if tensor.ndim == 1:
        padded_len = _align_up(tensor.shape[0] + extra, _METADATA_PAD_ALIGN)
        buf = torch.zeros((padded_len,), dtype=tensor.dtype, device=tensor.device)
        buf[: tensor.shape[0]].copy_(tensor)
        return buf

    # Common metadata shape: [1, N] (e.g., seq_idx_prefill). Grow only token dim.
    if tensor.ndim == 2 and tensor.shape[0] == 1:
        padded_len = min(
            _align_up(tensor.shape[1] + extra, _METADATA_PAD_ALIGN),
            _align_up(max(num_tokens, tensor.shape[1]), _METADATA_PAD_ALIGN),
        )
        buf = torch.zeros((1, padded_len), dtype=tensor.dtype, device=tensor.device)
        buf[:, : tensor.shape[1]].copy_(tensor)
        return buf

    # Fallback: only grow the leading dimension.
    padded_shape = list(tensor.shape)
    padded_shape[0] = _align_up(tensor.shape[0] + extra, _METADATA_PAD_ALIGN)
    buf = torch.zeros(tuple(padded_shape), dtype=tensor.dtype, device=tensor.device)
    slices = (slice(0, tensor.shape[0]),) + tuple(slice(None) for _ in range(tensor.ndim - 1))
    buf[slices].copy_(tensor)
    return buf


def _alloc_stable_metadata_result(
    result: Any,
    num_tokens: int,
    max_batch_size: int,
    pad_for_mamba_metadata: bool,
) -> Tuple[torch.Tensor, ...]:
    if isinstance(result, torch.Tensor):
        return (
            _alloc_stable_metadata_tensor(
                result, num_tokens, max_batch_size, pad_for_mamba_metadata
            ),
        )
    if isinstance(result, (tuple, list)):
        return tuple(
            _alloc_stable_metadata_tensor(t, num_tokens, max_batch_size, pad_for_mamba_metadata)
            if isinstance(t, torch.Tensor)
            else t
            for t in result
        )
    return ()


class MetadataWrapper(nn.Module):
    """Wraps a metadata-prep dynamic op to give its outputs stable addresses.

    Metadata-prep ops (e.g. mamba_ssm_prepare_metadata) run eagerly and return
    freshly allocated tensors.  After torch.cuda.empty_cache(), those addresses
    change, which breaks any downstream CUDA graph that captured them.

    Fix: on the capture pass we allocate stable buffers larger than the observed
    capture output when wrapping Mamba metadata prep
    (auto_deploy::mamba_ssm_prepare_metadata). For this op, runtime output can
    exceed capture output by up to (max_batch_size - 1) due to extra logical
    chunks from misaligned sequence boundaries in multi-sequence batches.
    On replay we copy_() the real results into those stable buffers and return
    the full padded tensors — same address and shape as capture time.
    Downstream dynamic ops determine how much to read from batch metadata
    (cu_seqlens, batch_info_host), so extra zeros are never accessed.
    """

    def __init__(self, submodule: nn.Module, max_batch_size: Optional[int] = None):
        super().__init__()
        self.submodule = submodule
        self.max_batch_size = (
            max_batch_size if max_batch_size is not None and max_batch_size > 0 else 1
        )
        self._pad_for_mamba_metadata = _contains_mamba_metadata_prep(submodule)
        # {num_tokens: tuple of stable output tensors (sized to bucket upper bound)}
        self._stable_outputs: Dict[int, Tuple[torch.Tensor, ...]] = {}

    def forward(self, *args, **kwargs) -> Any:
        nt = ADPiecewiseRunner._current_num_tokens
        phase = ADPiecewiseRunner._current_phase

        result = self.submodule(*args, **kwargs)

        if nt is None or phase == "warmup":
            return result

        if phase == "capture":
            padded = self._alloc_padded(result, nt)
            self._stable_outputs[nt] = padded
            return self._rebuild_result(result, padded)

        # replay
        saved = self._stable_outputs.get(nt)
        if saved is None:
            return result
        self._copy_into_saved(result, saved)
        return self._rebuild_result(result, saved)

    def _alloc_padded(self, result: Any, num_tokens: int) -> Tuple[torch.Tensor, ...]:
        return _alloc_stable_metadata_result(
            result=result,
            num_tokens=num_tokens,
            max_batch_size=self.max_batch_size,
            pad_for_mamba_metadata=self._pad_for_mamba_metadata,
        )

    @staticmethod
    def _copy_into_saved(result: Any, saved: Tuple[torch.Tensor, ...]) -> None:
        items = (result,) if isinstance(result, torch.Tensor) else result
        for real, stable in zip(items, saved):
            if not isinstance(real, torch.Tensor) or not isinstance(stable, torch.Tensor):
                continue
            if real.shape == stable.shape:
                stable.copy_(real)
            else:
                stable.zero_()
                slices = tuple(slice(0, s) for s in real.shape)
                stable[slices].copy_(real)

    @staticmethod
    def _rebuild_result(original: Any, saved: Tuple[torch.Tensor, ...]) -> Any:
        """Return saved tensors in the same container type as the original.

        The saved tensors are returned at their full (padded) shape — no
        truncation.  Downstream static CUDA graph segments record this padded
        shape at capture time and replay with the same shape.  Dynamic ops
        (SSM kernels) determine how much to read from batch_info / cu_seqlens,
        so extra zero-padded elements are never accessed.
        """
        if isinstance(original, torch.Tensor):
            return saved[0]
        if isinstance(original, tuple):
            return tuple(saved)
        if isinstance(original, list):
            return list(saved)
        return saved


class ADPiecewiseRunner(nn.Module):
    """Wraps a static submodule and manages its CUDA graph capture/replay.

    Behavior is controlled by two class-level contexts set by the orchestrator:
      - `_current_phase`: "warmup" (eager), "capture" (CUDA graph capture),
        or "replay" (graph replay / eager fallback at runtime)
      - `_current_num_tokens`: identifies which bucket entry to use

    If `num_tokens` doesn't match any captured bucket, falls back to eager.

    No copy-back is needed because:
      - Model-level inputs come from InputBuffer (stable addresses).
      - Dynamic op outputs are pre-allocated in the shared graph pool
        (deterministic addresses on replay).
      - Weights are inherently at fixed addresses.
    """

    _current_num_tokens: Optional[int] = None
    _current_phase: str = "replay"

    @classmethod
    def set_current_num_tokens(cls, num_tokens: Optional[int]) -> None:
        cls._current_num_tokens = num_tokens

    @classmethod
    def set_current_phase(cls, phase: str) -> None:
        assert phase in ("warmup", "capture", "replay"), f"Invalid phase: {phase}"
        cls._current_phase = phase

    def __init__(
        self,
        submodule: nn.Module,
        graph_pool: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.submodule = submodule
        self._graph_pool = graph_pool

        self._weight_ptrs: Set[int] = set()
        for p in submodule.parameters():
            self._weight_ptrs.add(p.data_ptr())
        for b in submodule.buffers():
            self._weight_ptrs.add(b.data_ptr())

        self.entries: Dict[int, SegmentEntry] = {}
        # Keyed by dynamic submodule index → OutputInfo discovered during warmup
        self._next_dynamic_out_infos: Dict[int, OutputInfo] = {}

    def set_dynamic_out_info(self, dynamic_submod_id: int, info: OutputInfo) -> None:
        """Set the output shape/dtype for a dynamic op that follows this runner.

        Called by the orchestrator after shape discovery.  The runner will allocate
        a buffer of this shape inside torch.cuda.graph() during capture.

        Args:
            dynamic_submod_id: Index of the dynamic submodule in the split graph.
            info: Output metadata (shape, dtype, device) discovered during warmup.
        """
        self._next_dynamic_out_infos[dynamic_submod_id] = info

    def get_dynamic_out_buf(
        self, num_tokens: int, dynamic_submod_id: int
    ) -> Optional[torch.Tensor]:
        """Retrieve the pre-allocated output buffer for a linked dynamic op.

        Returns the weak-ref'd buffer allocated during graph capture, or None
        if shape discovery failed or this runner has no linked dynamic op.

        Args:
            num_tokens: Bucket size identifying the SegmentEntry.
            dynamic_submod_id: Index of the dynamic submodule whose buffer to retrieve.
        """
        entry = self.entries.get(num_tokens)
        if entry is None:
            ad_logger.warning(
                "ADPiecewiseRunner.get_dynamic_out_buf: no entry for "
                "num_tokens=%d, dynamic_submod_id=%d.",
                num_tokens,
                dynamic_submod_id,
            )
            return None
        buf = entry.dynamic_out_bufs.get(dynamic_submod_id)
        if buf is None:
            ad_logger.warning(
                "ADPiecewiseRunner.get_dynamic_out_buf: no buffer for "
                "num_tokens=%d, dynamic_submod_id=%d. Shape discovery may "
                "have failed or this runner has no linked dynamic op.",
                num_tokens,
                dynamic_submod_id,
            )
            return None
        return buf

    def forward(self, *args, **kwargs) -> Any:
        num_tokens = ADPiecewiseRunner._current_num_tokens
        phase = ADPiecewiseRunner._current_phase

        if num_tokens is None:
            return self.submodule(*args, **kwargs)

        # --- WARMUP ---
        if phase == "warmup":
            return self.submodule(*args, **kwargs)

        entry = self.entries.get(num_tokens)
        if entry is None:
            entry = SegmentEntry()
            self.entries[num_tokens] = entry

        # --- CAPTURE ---
        if phase == "capture":
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()

            dynamic_out_bufs: Dict[int, torch.Tensor] = {}
            with torch.cuda.graph(graph, pool=self._graph_pool):
                output = self.submodule(*args, **kwargs)
                for submod_id, info in self._next_dynamic_out_infos.items():
                    dynamic_out_bufs[submod_id] = torch.empty(
                        info.shape, dtype=info.dtype, device=info.device
                    )

            torch.cuda.synchronize()

            if self._graph_pool is None:
                self._graph_pool = graph.pool()

            entry.cuda_graph = graph
            entry.static_output = make_weak_ref(output)
            for submod_id, buf in dynamic_out_bufs.items():
                entry.dynamic_out_bufs[submod_id] = make_weak_ref(buf)

            # Debug: record input addresses for assertion in replay
            flat_args, _ = tree_flatten((args, kwargs))
            entry.input_addresses = [
                a.data_ptr() if isinstance(a, torch.Tensor) else None for a in flat_args
            ]

            return output

        # --- REPLAY ---
        if entry.cuda_graph is None:
            return self.submodule(*args, **kwargs)

        if entry.input_addresses and not getattr(entry, "_address_verified", False):
            flat_args, _ = tree_flatten((args, kwargs))
            mismatches = []
            for i, (cap, cur_arg) in enumerate(zip(entry.input_addresses, flat_args)):
                cur = cur_arg.data_ptr() if isinstance(cur_arg, torch.Tensor) else None
                if cap != cur:
                    desc = (
                        f"shape={cur_arg.shape}, dtype={cur_arg.dtype}"
                        if isinstance(cur_arg, torch.Tensor)
                        else type(cur_arg).__name__
                    )
                    mismatches.append(f"  arg[{i}]: captured=0x{cap:x}, runtime=0x{cur:x} ({desc})")
            if mismatches:
                ad_logger.error(
                    "ADPiecewiseRunner ADDRESS MISMATCH for nt=%d! %d/%d inputs changed:\n%s",
                    num_tokens,
                    len(mismatches),
                    len(entry.input_addresses),
                    "\n".join(mismatches),
                )
            entry._address_verified = True

        entry.cuda_graph.replay()
        return entry.static_output

    @property
    def graph_pool(self):
        return self._graph_pool

    @graph_pool.setter
    def graph_pool(self, pool):
        self._graph_pool = pool


class DynamicOpWrapper(nn.Module):
    """Wraps a non-inplace dynamic op to pass out= from the preceding runner's graph pool.

    During warmup, forwards without out= (normal allocation).
    During capture/replay, retrieves the pre-allocated buffer from the linked
    ADPiecewiseRunner and passes it as the ``out`` kwarg.
    """

    def __init__(
        self,
        submodule: nn.Module,
        preceding_runner: ADPiecewiseRunner,
        dynamic_submod_id: int,
    ):
        super().__init__()
        self.submodule = submodule
        self.preceding_runner = preceding_runner
        self.dynamic_submod_id = dynamic_submod_id

    def forward(self, *args, **kwargs) -> Any:
        phase = ADPiecewiseRunner._current_phase
        if phase == "warmup":
            return self.submodule(*args, **kwargs)

        nt = ADPiecewiseRunner._current_num_tokens
        if nt is not None:
            out_buf = self.preceding_runner.get_dynamic_out_buf(nt, self.dynamic_submod_id)
            assert out_buf is not None, (
                f"DynamicOpWrapper(submod_{self.dynamic_submod_id}): "
                f"no pre-allocated out buffer for nt={nt}. "
                f"Shape discovery may have failed — downstream static runners "
                f"require stable output addresses."
            )
            kwargs["out"] = out_buf
        return self.submodule(*args, **kwargs)

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
  2. CAPTURE: Capture the submodule as a CUDA graph.  If this runner is linked to a
     following dynamic op (via set_dynamic_out_info), also allocate the dynamic op's
     output buffer *inside* the graph capture so it gets a deterministic address from
     the shared graph pool.
  3. REPLAY: Replay the captured CUDA graph.  The DynamicOpWrapper retrieves the
     pre-allocated output buffer via get_dynamic_out_buf().

MetadataStabilizer: wraps metadata-prep dynamic ops whose outputs flow into
  subsequent CUDA-graph-captured static segments.  These ops (e.g.
  mamba_ssm_prepare_metadata) run eagerly every time and allocate fresh output
  tensors.  After torch.cuda.empty_cache() between capture and runtime, the
  allocator hands out different addresses, breaking the CUDA graph's recorded
  pointers.  The stabilizer clones outputs during capture and copy_()s new data
  into those stable buffers on replay, preserving addresses at negligible cost
  (metadata tensors are tiny, e.g. two int32 [1024] ≈ 8 KB).

Memory:
  All model-level inputs (input_ids, position_ids, etc.) come from SequenceInfo's
  InputBuffer, which provides stable addresses across calls.  Dynamic op outputs are
  pre-allocated inside graph captures from a shared pool, giving deterministic addresses.
  Both static_output and dynamic_out_buf are weak-ref'd via make_weak_ref, allowing the
  graph pool to recycle memory across layers.  No copy-back mechanism needed.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten

from tensorrt_llm._torch.utils import make_weak_ref

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
    dynamic_out_buf: Optional[torch.Tensor] = None
    input_addresses: List[Optional[int]] = field(default_factory=list)


class MetadataStabilizer(nn.Module):
    """Wraps a metadata-prep dynamic op to give its outputs stable addresses.

    Metadata-prep ops (e.g. mamba_ssm_prepare_metadata) run eagerly and return
    freshly allocated tensors.  After torch.cuda.empty_cache(), those addresses
    change, which breaks any downstream CUDA graph that captured them.

    Fix: on the capture pass we clone the outputs (tiny int32 metadata tensors)
    and return the clones so the downstream CUDA graph records *clone* addresses.
    On replay we run the op eagerly, copy_() the real results into the clones,
    and return the clones — same addresses, correct data.
    """

    def __init__(self, submodule: nn.Module):
        super().__init__()
        self.submodule = submodule
        # {num_tokens: tuple of cloned output tensors}
        self._stable_outputs: Dict[int, Tuple[torch.Tensor, ...]] = {}

    def forward(self, *args, **kwargs) -> Any:
        nt = ADPiecewiseRunner._current_num_tokens
        phase = ADPiecewiseRunner._current_phase

        result = self.submodule(*args, **kwargs)

        if nt is None or phase == "warmup":
            return result

        if phase == "capture":
            clones = self._clone_result(result)
            self._stable_outputs[nt] = clones
            return self._rebuild_result(result, clones)

        # replay
        saved = self._stable_outputs.get(nt)
        if saved is None:
            return result
        self._copy_into_saved(result, saved)
        return self._rebuild_result(result, saved)

    @staticmethod
    def _clone_result(result: Any) -> Tuple[torch.Tensor, ...]:
        if isinstance(result, torch.Tensor):
            return (result.clone(),)
        if isinstance(result, (tuple, list)):
            return tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in result)
        return ()

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
        """Return saved tensors in the same container type as the original."""
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
        self._next_dynamic_out_info: Optional[OutputInfo] = None

    def set_dynamic_out_info(self, info: OutputInfo) -> None:
        """Set the output shape/dtype for the dynamic op that follows this runner.

        Called by the orchestrator after shape discovery.  The runner will allocate
        a buffer of this shape inside torch.cuda.graph() during capture.
        """
        self._next_dynamic_out_info = info

    def get_dynamic_out_buf(self, num_tokens: int) -> Optional[torch.Tensor]:
        """Retrieve the pre-allocated output buffer for the linked dynamic op.

        Returns the weak-ref'd buffer allocated during graph capture, or None
        if shape discovery failed or this runner has no linked dynamic op.
        """
        entry = self.entries.get(num_tokens)
        if entry is None or entry.dynamic_out_buf is None:
            ad_logger.warning(
                "ADPiecewiseRunner.get_dynamic_out_buf: no buffer for "
                "num_tokens=%d. Shape discovery may have failed or this "
                "runner has no linked dynamic op.",
                num_tokens,
            )
            return None
        return entry.dynamic_out_buf

    def forward(self, *args, **kwargs) -> Any:
        num_tokens = ADPiecewiseRunner._current_num_tokens
        phase = ADPiecewiseRunner._current_phase

        if num_tokens is None:
            return self.submodule(*args, **kwargs)

        entry = self.entries.get(num_tokens)
        if entry is None:
            entry = SegmentEntry()
            self.entries[num_tokens] = entry

        # --- WARMUP ---
        if phase == "warmup":
            return self.submodule(*args, **kwargs)

        # --- CAPTURE ---
        if phase == "capture":
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()

            dynamic_out_buf = None
            with torch.cuda.graph(graph, pool=self._graph_pool):
                output = self.submodule(*args, **kwargs)
                if self._next_dynamic_out_info is not None:
                    info = self._next_dynamic_out_info
                    dynamic_out_buf = torch.empty(info.shape, dtype=info.dtype, device=info.device)

            torch.cuda.synchronize()

            if self._graph_pool is None:
                self._graph_pool = graph.pool()

            entry.cuda_graph = graph
            entry.static_output = make_weak_ref(output)
            if dynamic_out_buf is not None:
                entry.dynamic_out_buf = make_weak_ref(dynamic_out_buf)

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

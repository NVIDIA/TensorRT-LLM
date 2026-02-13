"""ADPiecewiseRunner: manages warmup → capture → replay for a single static CUDA graph segment.

Each static submodule in a piecewise-split model is wrapped in an ADPiecewiseRunner.
The runner's behavior is controlled by two class-level contexts set by the orchestrator
(PiecewiseCapturedGraph) before each split_gm forward pass:

  - `_current_phase`: determines execution mode ("warmup", "capture", or "replay")
  - `_current_num_tokens`: identifies which bucket entry to use

Phase semantics:
  1. WARMUP: Run the submodule eagerly. (Data-ptr tracking runs but is NOT relied on
     for correctness — see note on dynamic-index identification below.)
  2. CAPTURE: Capture the submodule as a CUDA graph. All non-weight tensor args are
     treated as dynamic. For those that came from a previous static runner (found in
     the _static_output_registry), we reuse the same buffer (zero-copy). Others
     (model inputs, dynamic-segment outputs) are referenced directly and refreshed
     via _prepare_replay_inputs during replay.
  3. REPLAY: Copy only dynamic inputs into the static buffers, then replay the
     captured graph.

Dynamic-index identification:
  We do NOT rely on data_ptr() change detection during warmup, because PyTorch's
  caching allocator can reuse the same address for activation tensors across warmup
  iterations, making them falsely appear "static." Instead, we mark ALL non-weight
  tensor args as dynamic. Weights/buffers are identified by matching against
  data_ptrs collected from `submodule.parameters()` and `submodule.buffers()`.

Each runner maintains entries keyed by `num_tokens`.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from ..utils.logger import ad_logger


@dataclass
class SegmentEntry:
    """State for a single (num_tokens) configuration of a segment."""

    cuda_graph: Optional[torch.cuda.CUDAGraph] = None
    # Static input list — each element is a direct reference to a tensor at a fixed address.
    # During replay, _prepare_replay_inputs refreshes activation buffers as needed.
    #
    # Three categories:
    # - Weight tensors: referenced directly (already at fixed addresses, never change).
    # - Activation tensors from a previous static runner's output: reused from the
    #   static output registry. During replay the previous runner's CUDA graph writes to
    #   the same address, so _prepare_replay_inputs skips the copy (zero-copy).
    # - Activation tensors from model inputs or dynamic segment outputs: referenced
    #   directly from the capture iteration. During replay, the dynamic segment produces
    #   output at a new address, so _prepare_replay_inputs copies into this buffer.
    static_inputs: Optional[List[Any]] = None
    # Indices of dynamic (activation) tensor args that need copy during replay
    dynamic_indices: Optional[Set[int]] = None
    # Static output — the output tensor(s) produced during capture.
    # During replay, the CUDA graph writes to the same addresses, so returning
    # this object gives the caller the updated data.
    static_output: Any = None
    # Tracks data_ptr() of tensor args during warmup to identify static vs dynamic
    _warmup_data_ptrs: Optional[List[Optional[int]]] = None


class ADPiecewiseRunner(nn.Module):
    """Wraps a static submodule and manages its CUDA graph capture/replay.

    Behavior is controlled by two class-level contexts set by the orchestrator:
      - `_current_phase`: "warmup" (eager + ptr tracking), "capture" (CUDA graph
        capture), or "replay" (graph replay / eager fallback at runtime)
      - `_current_num_tokens`: identifies which bucket entry to use

    If `num_tokens` doesn't match any pre-configured bucket, falls back to eager.
    Bucket resolution (nearest bucket >= real token count) is handled upstream by
    DualModeCapturedGraph, so the runner always sees an exact bucket value.
    """

    # Class-level contexts: the orchestrator sets these before each split_gm forward pass
    # so ALL runners in the graph use the same correct num_tokens and phase.
    _current_num_tokens: Optional[int] = None
    _current_phase: str = "replay"  # "warmup", "capture", or "replay"

    # Class-level registry of output tensors produced during CUDA graph capture.
    # Key: (num_tokens, data_ptr) -> output tensor at a fixed address.
    # During capture, a runner checks if any of its activation inputs match a
    # registered output (by data_ptr). If so, it references that buffer directly —
    # enabling zero-copy during replay (the producer's graph writes, the consumer's
    # graph reads, same address).
    # Note: runners capture in sequential order, so all registry entries are from
    # earlier runners — no need to track runner_id.
    _static_output_registry: Dict[Tuple[int, int], torch.Tensor] = {}

    @classmethod
    def set_current_num_tokens(cls, num_tokens: Optional[int]) -> None:
        """Set the current num_tokens context for all runners.

        Called by PiecewiseCapturedGraph before each forward pass through the split graph.
        """
        cls._current_num_tokens = num_tokens

    @classmethod
    def set_current_phase(cls, phase: str) -> None:
        """Set the current execution phase for all runners.

        Called by PiecewiseCapturedGraph to control warmup → capture → replay transitions.
        Valid phases: "warmup", "capture", "replay".
        """
        assert phase in ("warmup", "capture", "replay"), f"Invalid phase: {phase}"
        cls._current_phase = phase

    @classmethod
    def clear_static_output_registry(cls) -> None:
        """Clear the static output registry.

        Called when switching between different graph configurations or resetting state.
        """
        cls._static_output_registry.clear()

    def __init__(
        self,
        submodule: nn.Module,
        piecewise_num_tokens: Optional[List[int]] = None,
        graph_pool: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.submodule = submodule
        self._graph_pool = graph_pool

        # Collect data_ptrs of all parameters and buffers in this submodule.
        # These are weight tensors with stable addresses that NEVER need copying.
        # Everything else that appears in flat_args is a cross-partition activation
        # (from a previous static runner or a dynamic segment) and must be treated
        # as dynamic for correctness during CUDA graph replay.
        self._weight_ptrs: Set[int] = set()
        for p in submodule.parameters():
            self._weight_ptrs.add(p.data_ptr())
        for b in submodule.buffers():
            self._weight_ptrs.add(b.data_ptr())

        # Pre-populate entries for each bucket size
        self.entries: Dict[int, SegmentEntry] = {}
        if piecewise_num_tokens:
            for nt in piecewise_num_tokens:
                self.entries[nt] = SegmentEntry()

    def _find_entry(self, num_tokens: int) -> Optional[SegmentEntry]:
        """Find the SegmentEntry for the given num_tokens.

        Expects an exact match — bucket resolution (nearest bucket >= real token count)
        is handled upstream by DualModeCapturedGraph._find_nearest_bucket before
        num_tokens reaches the runner.

        Returns None if num_tokens doesn't match any pre-configured bucket (eager fallback).
        """
        return self.entries.get(num_tokens)

    def _track_warmup_ptrs(self, entry: SegmentEntry, flat_args: List[Any]) -> None:
        """Track data_ptr() during warmup to identify static (weight) vs dynamic (activation) args.

        On the first warmup call, record all data_ptrs. On subsequent calls, mark args whose
        data_ptr changed as "dynamic" (by setting their tracked ptr to None).
        """
        if entry._warmup_data_ptrs is None:
            # First warmup: record all data_ptrs
            entry._warmup_data_ptrs = [
                a.data_ptr() if isinstance(a, torch.Tensor) else None for a in flat_args
            ]
        else:
            # Subsequent warmup: check for changes
            for i, a in enumerate(flat_args):
                if isinstance(a, torch.Tensor):
                    if (
                        entry._warmup_data_ptrs[i] is not None
                        and a.data_ptr() != entry._warmup_data_ptrs[i]
                    ):
                        # data_ptr changed → this is a dynamic (activation) tensor
                        entry._warmup_data_ptrs[i] = None

    def _identify_dynamic_indices(self, entry: SegmentEntry, flat_args: List[Any]) -> Set[int]:
        """Mark all non-weight tensor args as dynamic.

        Weight/buffer tensors (matched via _weight_ptrs) are static.
        Everything else is dynamic — the capture code will further check
        _static_output_registry for zero-copy reuse where possible.
        """
        dynamic_indices: Set[int] = set()
        for i, a in enumerate(flat_args):
            if not isinstance(a, torch.Tensor):
                continue
            if a.data_ptr() in self._weight_ptrs:
                continue  # Weight/buffer — stable address, no copy needed
            dynamic_indices.add(i)
        return dynamic_indices

    def _prepare_replay_inputs(self, entry: SegmentEntry, flat_inputs: List[Any]) -> None:
        """Refresh dynamic activation buffers before CUDA graph replay.

        For each dynamic tensor input, this copies runtime data into the captured
        static buffer unless both tensors already share the same data_ptr() (no-copy
        fast path, common for static segment chaining).

        When runtime input is smaller than the bucketed static buffer (padding case),
        copy only the valid prefix and leave the remaining captured padding untouched.
        """
        for idx in entry.dynamic_indices:
            new_inp = flat_inputs[idx]
            static_inp = entry.static_inputs[idx]

            if not isinstance(new_inp, torch.Tensor) or not isinstance(static_inp, torch.Tensor):
                continue

            # Fast path: no copy needed when producer already wrote into the
            # captured static buffer (segment N output -> segment N+1 input).
            if new_inp.data_ptr() == static_inp.data_ptr():
                continue

            if static_inp.shape == new_inp.shape:
                static_inp.copy_(new_inp, non_blocking=True)
            elif (
                new_inp.shape[0] < static_inp.shape[0] and new_inp.shape[1:] == static_inp.shape[1:]
            ):
                # Padded case: runtime input is smaller along dim 0.
                # Copy real data into the front of the static buffer.
                n = new_inp.shape[0]
                static_inp[:n].copy_(new_inp, non_blocking=True)
            elif (
                new_inp.ndim >= 2
                and new_inp.shape[1] < static_inp.shape[1]
                and new_inp.shape[0] == static_inp.shape[0]
            ):
                # Padded case: runtime input is smaller along dim 1
                # (e.g., [1, real, D] vs [1, bucket, D]).
                n = new_inp.shape[1]
                static_inp[:, :n].copy_(new_inp, non_blocking=True)
            else:
                # Fallback: shapes are incompatible — this is a real error
                static_inp.copy_(new_inp, non_blocking=True)

    def forward(self, *args, **kwargs) -> Any:
        # Use the class-level contexts set by the orchestrator
        num_tokens = ADPiecewiseRunner._current_num_tokens
        phase = ADPiecewiseRunner._current_phase
        entry = self._find_entry(num_tokens) if num_tokens is not None else None

        if entry is None:
            # Unknown num_tokens or exceeds all buckets — fallback to eager
            return self.submodule(*args, **kwargs)

        # Flatten inputs once (used by all phases)
        flat_args, args_spec = tree_flatten((args, kwargs))

        # --- WARMUP PHASE ---
        if phase == "warmup":
            # Track data_ptr() to distinguish weights from activations
            self._track_warmup_ptrs(entry, flat_args)
            return self.submodule(*args, **kwargs)

        # --- CAPTURE PHASE ---
        if phase == "capture":
            ad_logger.debug(f"ADPiecewiseRunner: capturing CUDA graph for num_tokens={num_tokens}")

            # Identify which args are dynamic (activations) vs static (weights)
            entry.dynamic_indices = self._identify_dynamic_indices(entry, flat_args)

            # Build static_inputs list for this entry. Every element is a direct
            # reference (no cloning) — we just need each tensor at a persistent address.
            #
            # For activation tensors, we check the static output registry to find
            # outputs from previous static runners. During replay, those runners'
            # CUDA graphs write to the same address, so _prepare_replay_inputs can skip the
            # copy (zero-copy). All other activation tensors (model inputs, dynamic
            # segment outputs) are referenced directly from this capture iteration;
            # _prepare_replay_inputs will copy new data into them during replay.
            entry.static_inputs = []
            num_reused = 0
            num_referenced = 0
            for i, a in enumerate(flat_args):
                if isinstance(a, torch.Tensor) and i in entry.dynamic_indices:
                    # Check if this activation is a previous static runner's output
                    # (if so, record the registry reference for zero-copy during replay)
                    prev_output = ADPiecewiseRunner._static_output_registry.get(
                        (num_tokens, a.data_ptr())
                    )
                    if prev_output is not None:
                        entry.static_inputs.append(prev_output)
                        num_reused += 1
                    else:
                        # Model input or dynamic segment output — reference directly.
                        # During replay, _prepare_replay_inputs will copy new data into this buffer.
                        entry.static_inputs.append(a)
                else:
                    # Weight tensor — reference directly (fixed address, never changes)
                    entry.static_inputs.append(a)
                    if isinstance(a, torch.Tensor):
                        num_referenced += 1

            # Unflatten back to get the static args/kwargs
            static_args_kwargs = tree_unflatten(entry.static_inputs, args_spec)
            static_args, static_kwargs = static_args_kwargs

            # Capture
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._graph_pool):
                output = self.submodule(*static_args, **static_kwargs)

            torch.cuda.synchronize()

            # Fallback: if no pool was provided at construction time, store the
            # auto-created pool so subsequent captures within this runner reuse it.
            if self._graph_pool is None:
                self._graph_pool = graph.pool()

            entry.cuda_graph = graph
            entry.static_output = output

            # Register outputs in the static output registry so next runners can reuse them
            flat_output, _ = tree_flatten(output)
            for out_tensor in flat_output:
                if isinstance(out_tensor, torch.Tensor):
                    ADPiecewiseRunner._static_output_registry[
                        (num_tokens, out_tensor.data_ptr())
                    ] = out_tensor

            num_dynamic = len(entry.dynamic_indices) - num_reused
            ad_logger.debug(
                f"ADPiecewiseRunner: captured graph for num_tokens={num_tokens} — "
                f"{num_dynamic} dynamic activation buffers, "
                f"{num_reused} reused from previous static segments, "
                f"{num_referenced} weight tensors (zero-copy)"
            )

            return output

        # --- REPLAY PHASE ---
        # Copy only dynamic inputs into static buffers.
        # _prepare_replay_inputs skips copy if input is already at static buffer address
        # (common case: segment N's output is segment N+1's input, so addresses match)
        self._prepare_replay_inputs(entry, flat_args)

        # Replay the captured graph
        entry.cuda_graph.replay()

        return entry.static_output

    @property
    def graph_pool(self):
        """Return the CUDA graph memory pool (for sharing across runners)."""
        return self._graph_pool

    @graph_pool.setter
    def graph_pool(self, pool):
        self._graph_pool = pool

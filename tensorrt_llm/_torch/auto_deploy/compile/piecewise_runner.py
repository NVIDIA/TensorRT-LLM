"""ADPiecewiseRunner: manages warmup → capture → replay for a single static CUDA graph segment.

Each static submodule in a piecewise-split model is wrapped in an ADPiecewiseRunner.
The runner's behavior is controlled by two class-level contexts set by the orchestrator
(PiecewiseCapturedGraph) before each split_gm forward pass:

  - `_current_phase`: determines execution mode ("warmup", "capture", or "replay")
  - `_current_num_tokens`: identifies which bucket entry to use

Phase semantics:
  1. WARMUP: Run the submodule eagerly, tracking data_ptr() of each flattened tensor
     arg to identify which args are "static" (weights — same address every call) vs
     "dynamic" (activations — address changes each call).
  2. CAPTURE: Capture the submodule as a CUDA graph. Only dynamic (activation) tensors
     are cloned into static input buffers. Weight tensors are referenced directly
     (already at fixed addresses, zero-copy).
  3. REPLAY: Copy only dynamic inputs into the static buffers, then replay the
     captured graph.

Each runner maintains entries keyed by `num_tokens`.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from ..utils.logger import ad_logger


@dataclass
class SegmentEntry:
    """State for a single (num_tokens) configuration of a segment."""

    cuda_graph: Optional[torch.cuda.CUDAGraph] = None
    # Static input list — each element is a direct reference to a tensor at a fixed address.
    # During replay, _copy_inputs copies new data into activation buffers as needed.
    #
    # Three categories:
    # - Weight tensors: referenced directly (already at fixed addresses, never change).
    # - Activation tensors from a previous static runner's output: reused from the
    #   static output registry. During replay the previous runner's CUDA graph writes to
    #   the same address, so _copy_inputs skips the copy (zero-copy).
    # - Activation tensors from model inputs or dynamic segment outputs: referenced
    #   directly from the capture iteration. During replay, the dynamic segment produces
    #   output at a new address, so _copy_inputs copies into this buffer.
    static_inputs: Optional[List[Any]] = None
    # Indices of dynamic (activation) tensor args that need copy during replay
    dynamic_indices: Optional[List[int]] = None
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
    # Key: (runner_id, num_tokens, output_idx) -> output tensor at a fixed address.
    # During capture, a runner checks if any of its activation inputs match a
    # registered output (by data_ptr). If so, it references that buffer directly —
    # enabling zero-copy during replay (the producer's graph writes, the consumer's
    # graph reads, same address).
    _static_output_registry: Dict[Tuple[int, int, int], torch.Tensor] = {}
    _runner_counter: int = 0

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
        cls._runner_counter = 0

    def __init__(
        self,
        submodule: nn.Module,
        piecewise_num_tokens: Optional[List[int]] = None,
        graph_pool: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.submodule = submodule
        self._graph_pool = graph_pool

        # Assign unique runner ID for registry tracking
        self._runner_id = ADPiecewiseRunner._runner_counter
        ADPiecewiseRunner._runner_counter += 1

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

    def _identify_dynamic_indices(self, entry: SegmentEntry, flat_args: List[Any]) -> List[int]:
        """After warmup, determine which tensor args are dynamic (need cloning for CUDA graph).

        An arg is dynamic if:
        - It's a tensor AND its data_ptr() changed during warmup (tracked as None), OR
        - It's a tensor AND we have no warmup data (conservative: treat as dynamic)
        """
        dynamic_indices = []
        for i, a in enumerate(flat_args):
            if not isinstance(a, torch.Tensor):
                continue
            if entry._warmup_data_ptrs is None:
                # No warmup tracking — conservatively treat all tensors as dynamic
                dynamic_indices.append(i)
            elif entry._warmup_data_ptrs[i] is None:
                # data_ptr changed during warmup → dynamic activation tensor
                dynamic_indices.append(i)
            # else: data_ptr was stable → weight tensor, no clone needed
        return dynamic_indices

    def _copy_inputs(self, entry: SegmentEntry, flat_inputs: List[Any]) -> None:
        """Copy only dynamic (activation) input data into the static input buffers.

        When padding is active (runtime num_tokens < bucket size), dynamic ops may
        produce outputs sized to the real token count. In that case, we copy into the
        first N positions of the static buffer, leaving the remaining positions (zeros
        from capture) untouched.

        Skip copy if input is already at the static buffer address (common case:
        segment N's output is segment N+1's input, so addresses match).
        """
        for idx in entry.dynamic_indices:
            static_inp = entry.static_inputs[idx]
            new_inp = flat_inputs[idx]
            if isinstance(new_inp, torch.Tensor) and isinstance(static_inp, torch.Tensor):
                # Skip copy if input is already at the static buffer address
                # (segment N's output is segment N+1's input)
                if new_inp.data_ptr() == static_inp.data_ptr():
                    continue

                if static_inp.shape == new_inp.shape:
                    static_inp.copy_(new_inp, non_blocking=True)
                elif (
                    new_inp.shape[0] < static_inp.shape[0]
                    and new_inp.shape[1:] == static_inp.shape[1:]
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

        # --- WARMUP PHASE ---
        if phase == "warmup":
            # Track data_ptr() to distinguish weights from activations
            flat_args, _ = tree_flatten((args, kwargs))
            self._track_warmup_ptrs(entry, flat_args)
            return self.submodule(*args, **kwargs)

        # --- CAPTURE PHASE ---
        if phase == "capture":
            ad_logger.debug(f"ADPiecewiseRunner: capturing CUDA graph for num_tokens={num_tokens}")

            # Flatten inputs
            flat_args, args_spec = tree_flatten((args, kwargs))
            entry._args_spec = args_spec  # type: ignore[attr-defined]

            # Identify which args are dynamic (activations) vs static (weights)
            entry.dynamic_indices = self._identify_dynamic_indices(entry, flat_args)

            # Build static_inputs list for this entry. Every element is a direct
            # reference (no cloning) — we just need each tensor at a persistent address.
            #
            # For activation tensors, we check the static output registry to find
            # outputs from previous static runners. During replay, those runners'
            # CUDA graphs write to the same address, so _copy_inputs can skip the
            # copy (zero-copy). All other activation tensors (model inputs, dynamic
            # segment outputs) are referenced directly from this capture iteration;
            # _copy_inputs will copy new data into them during replay.
            entry.static_inputs = []
            dynamic_set = set(entry.dynamic_indices)
            num_reused = 0
            num_referenced = 0
            for i, a in enumerate(flat_args):
                if isinstance(a, torch.Tensor) and i in dynamic_set:
                    # Check if this activation is a previous static runner's output
                    # (if so, record the registry reference for zero-copy during replay)
                    input_ptr = a.data_ptr()
                    reused = False
                    for (
                        prev_runner_id,
                        prev_nt,
                        prev_out_idx,
                    ), prev_output in ADPiecewiseRunner._static_output_registry.items():
                        if (
                            prev_runner_id < self._runner_id
                            and prev_nt == num_tokens
                            and isinstance(prev_output, torch.Tensor)
                            and prev_output.data_ptr() == input_ptr
                            and prev_output.shape == a.shape
                        ):
                            entry.static_inputs.append(prev_output)
                            num_reused += 1
                            reused = True
                            break
                    if not reused:
                        # Model input or dynamic segment output — reference directly.
                        # During replay, _copy_inputs will copy new data into this buffer.
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

            # Store the graph pool for sharing across segments
            if self._graph_pool is None:
                self._graph_pool = graph.pool()

            entry.cuda_graph = graph
            entry.static_output = output

            # Register outputs in the static output registry so next runners can reuse them
            flat_output, _ = tree_flatten(output)
            for out_idx, out_tensor in enumerate(flat_output):
                if isinstance(out_tensor, torch.Tensor):
                    ADPiecewiseRunner._static_output_registry[
                        (self._runner_id, num_tokens, out_idx)
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
        # _copy_inputs skips copy if input is already at static buffer address
        # (common case: segment N's output is segment N+1's input, so addresses match)
        flat_args, _ = tree_flatten((args, kwargs))
        self._copy_inputs(entry, flat_args)

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

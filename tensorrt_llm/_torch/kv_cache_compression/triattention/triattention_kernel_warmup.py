# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""One home for every kernel warmup in the TriAttention eviction pipeline.

Triton and CuTE-DSL kernels JIT-compile on first call, and the eviction
kernels never run inside the engine warmup forwards, so each launcher here
compiles its kernels once at build time and then calls the compiled binaries
directly: no per-call dispatch work and no allocations on the eviction path.
Build one launcher per fixed buffer set; call it once per eviction round.
"""

from typing import Dict, Optional, Tuple

import torch

from .triattention_kernels import (
    _score_row_stats_kernel,
    _score_union_kernel,
    _settle_ties_after_topk_kernel,
)


class _FixedScoreStreamMismatch(RuntimeError):
    """Raised when fixed score staging buffers are used from another CUDA stream."""


class FrozenTritonKernelCall:
    """Call one Triton kernel frozen at build time.

    Triton's standard dispatch costs tens of microseconds of host work per
    call; eviction fires its kernels every round, so the grid, bound tensor
    set, and constexpr set are frozen once here. ``warmup`` (Triton's own
    API) JIT-compiles the kernel at build time and ``__call__`` runs the
    compiled binary directly. Constexpr values are passed positionally on
    each call, so their order is validated against the kernel's constexpr
    parameter declaration order at build time.
    """

    def __init__(
        self,
        triton_kernel,
        bound_tensors: Tuple[torch.Tensor, ...],
        constexpr_values: Dict[str, object],
        *,
        grid: Tuple[int, ...],
        num_warps: int,
    ) -> None:
        params = getattr(triton_kernel, "params", None)
        if params is not None:
            declared = [param.name for param in params if param.is_constexpr]
            if list(constexpr_values.keys()) != declared:
                raise ValueError(
                    f"constexpr order {list(constexpr_values.keys())} must match the "
                    f"kernel's declaration order {declared}: the frozen call passes "
                    "them positionally"
                )
        self.device = bound_tensors[0].device
        self.bound_tensors = tuple(bound_tensors)
        self.constexpr_values = dict(constexpr_values)
        with torch.cuda.device(self.device):
            self.build_stream = torch.cuda.current_stream(self.device)
            # warmup() then indexing the compiled cache by grid is the
            # documented-by-use Triton pattern for dispatch-free calls; if a
            # Triton upgrade changes it, this raises here at build time rather
            # than corrupting a later call.
            compiled = triton_kernel.warmup(
                *self.bound_tensors,
                **self.constexpr_values,
                num_warps=num_warps,
                grid=grid,
            )
            self.compiled_kernel_runner = compiled[grid]

    def __call__(self, *call_tensors: torch.Tensor) -> None:
        """Run the kernel; ``call_tensors``, if given, substitute the bound tensors."""
        current_stream = torch.cuda.current_stream(self.device)
        if (current_stream.device, current_stream.cuda_stream) != (
            self.build_stream.device,
            self.build_stream.cuda_stream,
        ):
            raise RuntimeError("a frozen Triton kernel call must run on the stream it was built on")
        self.compiled_kernel_runner(
            *(call_tensors if call_tensors else self.bound_tensors),
            *self.constexpr_values.values(),
            stream=self.build_stream.cuda_stream,
        )


class FrozenScoreCall:
    """Phase and trig-score launches frozen over one staging buffer set."""

    def __init__(self, staging, valid_widths: torch.Tensor, score_aggregation: str) -> None:
        from .triattention_kernels import _prepare_mean_phase_kernel, _tri_score_perhead_kernel

        if score_aggregation not in ("mean", "max"):
            raise ValueError(f"unsupported score aggregation: {score_aggregation}")
        group = staging.fused_group
        if (
            valid_widths.shape != (staging.max_requests,)
            or valid_widths.dtype != torch.int32
            or valid_widths.device != staging.device
            or not valid_widths.is_contiguous()
        ):
            raise ValueError("prepared score lengths do not match their fixed buffers")

        frequency_block = 1 << (group.num_freqs - 1).bit_length()
        phase_pointer_args = (
            staging.round_starts_device,
            staging.offsets,
            staging.omega,
            staging.mean_cos,
            staging.mean_sin,
        )
        phase_constants = (
            group.num_freqs,
            int(staging.offsets.numel()),
            frequency_block,
        )
        score_pointer_args = (
            *group.pointer_prefix,
            staging.valid_seq_lens_device,
            valid_widths,
            staging.round_starts_device,
            staging.token_starts_device,
            *group.pointer_middle,
            staging.mean_cos.view(-1),
            staging.mean_sin.view(-1),
            *group.pointer_tail,
            group.output,
        )
        score_geometry = (
            group.output_width,
            group.num_layers,
            *group.geometry_args,
        )
        score_constants = (
            score_aggregation == "max",
            group.token_block,
            frequency_block,
        )
        self._phase_args = (*phase_pointer_args, *phase_constants)
        self._score_args = (*score_pointer_args, *score_geometry, *score_constants)
        phase_grid = (staging.max_requests, 1, 1)
        score_segments = staging.max_requests * group.num_layers
        if score_segments > 65535:
            # Segments sit on the y grid axis (CUDA caps y/z at 65535) so the
            # unbounded x axis can hold the token tiles of long sequences.
            raise ValueError("request*layer segment count exceeds the CUDA grid limit")
        score_grid = (
            group.max_ntblk,
            score_segments,
            group.num_kv_heads,
        )
        self.device = staging.device
        self.output = group.output
        self._phase_runner = None
        with torch.cuda.device(self.device):
            self.stream = torch.cuda.current_stream(self.device)
            if score_aggregation == "mean":
                compiled = _prepare_mean_phase_kernel.warmup(
                    *phase_pointer_args,
                    NUM_FREQS=phase_constants[0],
                    NUM_OFFSETS=phase_constants[1],
                    F_BLOCK=phase_constants[2],
                    num_warps=1,
                    grid=phase_grid,
                )
                self._phase_runner = compiled[phase_grid]
            compiled = _tri_score_perhead_kernel.warmup(
                *score_pointer_args,
                *score_geometry,
                USE_MAX=score_constants[0],
                T_BLOCK=score_constants[1],
                F_BLOCK=score_constants[2],
                grid=score_grid,
            )
            self._score_runner = compiled[score_grid]

    def __call__(self) -> torch.Tensor:
        current_stream = torch.cuda.current_stream(self.device)
        if (current_stream.device, current_stream.cuda_stream) != (
            self.stream.device,
            self.stream.cuda_stream,
        ):
            raise _FixedScoreStreamMismatch(
                "TriAttention prepared score is bound to its staging CUDA stream"
            )
        if self._phase_runner is not None:
            self._phase_runner(*self._phase_args, stream=self.stream.cuda_stream)
        self._score_runner(*self._score_args, stream=self.stream.cuda_stream)
        return self.output


class _PreparedDeterministicTopK:
    """Deterministic top-k over fixed row-major buffers, built once.

    The CuTE top-k kernel is fast but breaks score ties arbitrarily and emits
    indices in arbitrary order, so a frozen Triton finalizer recomputes the
    threshold membership with lowest-index-wins ties, rebases each row by its
    prompt offset, and writes sorted ordinals. The CuTE kernel is compiled
    here once with owned scratch storage; every later call runs both kernels
    over the bound buffers with no dispatch work and no allocations.
    """

    def __init__(
        self,
        scores: torch.Tensor,
        seq_lens: torch.Tensor,
        prompt_offsets: torch.Tensor,
        provisional_indices: torch.Tensor,
        output_indices: torch.Tensor,
        keep_count: int,
    ) -> None:
        rows, width = scores.shape
        if not 1 <= keep_count <= width:
            raise ValueError("deterministic top-k requires 1 <= keep_count <= width")

        from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

        self.device = scores.device
        self.scores = scores
        self.seq_lens = seq_lens
        self.provisional_indices = provisional_indices
        with torch.cuda.device(self.device):
            self.stream = torch.cuda.current_stream(self.device)
            self.scratch = torch.empty((rows, 2, width), dtype=torch.int32, device=self.device)
            runner = cute_dsl_custom_ops.CuteDSLTopKDecodeSingleCTARunner
            key = (
                cute_dsl_custom_ops._TORCH_TO_CUTLASS_DTYPE[torch.float32],
                1 << (width - 1).bit_length(),
                keep_count,
                1,
                False,
                256,
                False,
                rows > cute_dsl_custom_ops._get_num_sms(),
            )
            runner._compile(*key)
            self.compiled_topk = runner.kernel_cache[key]
        self.frozen_settle_ties = FrozenTritonKernelCall(
            _settle_ties_after_topk_kernel,
            (scores, seq_lens, prompt_offsets, provisional_indices, output_indices),
            dict(
                WIDTH=width,
                KEEP_COUNT=keep_count,
                OUTPUT_WIDTH=keep_count,
                BLOCK=256,
            ),
            grid=(rows, 1, 1),
            num_warps=4,
        )

    def __call__(self) -> None:
        self.compiled_topk(
            self.scores,
            None,
            self.scratch,
            None,
            self.seq_lens,
            self.provisional_indices,
            None,
        )
        self.frozen_settle_ties()


class _PreparedUnionScores:
    """Launch fixed union score preparation without Triton JIT dispatch."""

    def __init__(
        self,
        scores: torch.Tensor,
        valid_widths: torch.Tensor,
        row_mean: torch.Tensor,
        row_inv_std: torch.Tensor,
        combined: torch.Tensor,
        *,
        normalize_scores: bool,
    ) -> None:
        if scores.ndim != 3:
            raise ValueError("prepared union scores require request-major rows")
        request_count, rows, width = scores.shape
        if (
            not scores.is_cuda
            or scores.dtype != torch.float32
            or not scores.is_contiguous()
            or valid_widths.shape != (request_count,)
            or valid_widths.dtype != torch.int32
            or valid_widths.device != scores.device
            or row_mean.shape != (request_count, rows, 1)
            or row_mean.dtype != torch.float32
            or row_mean.device != scores.device
            or row_inv_std.shape != row_mean.shape
            or row_inv_std.dtype != torch.float32
            or row_inv_std.device != scores.device
            or combined.shape != (request_count, width)
            or combined.dtype != torch.float32
            or combined.device != scores.device
            or not valid_widths.is_contiguous()
            or not row_mean.is_contiguous()
            or not row_inv_std.is_contiguous()
            or not combined.is_contiguous()
        ):
            raise ValueError("prepared union score tensors do not share one fixed geometry")

        normalize_scores = bool(normalize_scores)
        # Callers verify score-tensor identity and the normalize flag against
        # this launcher before dispatching to it.
        self.scores = scores
        self.normalize_scores = normalize_scores
        self._frozen_stats_call = None
        if normalize_scores:
            stats_grid = (request_count * rows, 1, 1)
            self._frozen_stats_call = FrozenTritonKernelCall(
                _score_row_stats_kernel,
                (scores, valid_widths, row_mean, row_inv_std),
                dict(ROWS=rows, WIDTH=width, BLOCK=256),
                grid=stats_grid,
                num_warps=4,
            )
        union_grid = (request_count, (width + 31) // 32, 1)
        self._frozen_union_call = FrozenTritonKernelCall(
            _score_union_kernel,
            (scores, valid_widths, row_mean, row_inv_std, combined),
            dict(ROWS=rows, WIDTH=width, NORMALIZE=normalize_scores, BLOCK=32),
            grid=union_grid,
            num_warps=1,
        )

    def __call__(self) -> None:
        if self._frozen_stats_call is not None:
            self._frozen_stats_call()
        self._frozen_union_call()


# Launch shape of the move-index packing kernel: tokens per program along the
# move axis, and its warp count.
_PACK_BLOCK_TOKENS = 256
_PACK_NUM_WARPS = 4


def frozen_move_index_pack(
    kept_token_ordinals: torch.Tensor,
    valid_sequence_lengths: torch.Tensor,
    move_source_offsets: torch.Tensor,
    move_source_indices: torch.Tensor,
    *,
    eviction_mode: str,
    decode_keep_count: int,
    num_dense_layers: int,
    num_kv_heads: int,
    max_protected_tail: int,
    swa_window: int,
    swa_move_source_offsets: Optional[torch.Tensor],
    swa_move_source_indices: Optional[torch.Tensor],
) -> FrozenTritonKernelCall:
    """Build one frozen call of the move-index packing kernel.

    The kernel reads the kept-token ordinals and each request's valid length
    and writes the packed per-(layer, head) move source indices consumed by
    the C++ compact launches. Only the caller-provided selection tensors are
    validated here; the move buffers are allocated by this module.
    """
    per_layer = eviction_mode == "per_layer_perhead"
    union = eviction_mode == "union"
    request_count = int(kept_token_ordinals.shape[0]) if kept_token_ordinals.ndim else 0
    if union:
        selection_rows = 1
    elif per_layer:
        selection_rows = num_dense_layers * num_kv_heads
    else:
        selection_rows = num_kv_heads
    selection_prefix = (request_count,) if union else (request_count, selection_rows)
    # Selection rows carry decode-only kept ordinals (already absolute), so
    # the rectangle is prompt-length independent.
    expected_selection = (*selection_prefix, decode_keep_count)
    if (
        request_count <= 0
        or tuple(kept_token_ordinals.shape) != expected_selection
        or valid_sequence_lengths.shape != (request_count,)
    ):
        raise ValueError(
            f"prepared compaction packing expects kept ordinals of shape "
            f"{expected_selection} and one valid length per request; got "
            f"{tuple(kept_token_ordinals.shape)} and "
            f"{tuple(valid_sequence_lengths.shape)}"
        )

    if swa_move_source_indices is not None:
        swa_offsets_arg = swa_move_source_offsets
        swa_indices_arg = swa_move_source_indices
        swa_total = int(swa_move_source_indices.shape[-1])
    else:
        # HAS_SWA specializes all corresponding loads and stores away.
        swa_offsets_arg = move_source_offsets
        swa_indices_arg = move_source_indices
        swa_total = 0

    from .triattention_kernels import _pack_compaction_sources_kernel

    max_move = decode_keep_count + max_protected_tail
    if swa_total:
        max_move = max(max_move, swa_window + max_protected_tail)
    packed_row_count = num_dense_layers * num_kv_heads if per_layer else num_kv_heads
    grid = (
        request_count,
        packed_row_count,
        (max_move + _PACK_BLOCK_TOKENS - 1) // _PACK_BLOCK_TOKENS,
    )
    bound_tensors = (
        kept_token_ordinals,
        valid_sequence_lengths,
        move_source_offsets,
        move_source_indices,
        swa_offsets_arg,
        swa_indices_arg,
    )
    # Ordered to match the kernel's constexpr parameter declaration: the
    # frozen call passes these by position.
    constexpr_values = dict(
        DENSE_TOTAL=int(move_source_indices.shape[-1]),
        SWA_TOTAL=swa_total,
        SELECTION_ROWS=selection_rows,
        SELECTION_STRIDE=decode_keep_count,
        KEEP_COUNT=decode_keep_count,
        NUM_KV_HEADS=num_kv_heads,
        SWA_WINDOW=swa_window,
        UNION=union,
        PER_LAYER=per_layer,
        HAS_SWA=swa_total > 0,
        BLOCK=_PACK_BLOCK_TOKENS,
    )
    return FrozenTritonKernelCall(
        _pack_compaction_sources_kernel,
        bound_tensors,
        constexpr_values,
        grid=grid,
        num_warps=_PACK_NUM_WARPS,
    )

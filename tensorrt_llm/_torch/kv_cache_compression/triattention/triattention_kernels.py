# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU kernels for the TriAttention KV-eviction pipeline.

The production path uses one fixed-shape trig-score launch across all dense
layers, CuTE-DSL TopK selection, and grouped C++ compaction. This module owns
the score launcher and its persistent metadata; scoring itself runs through
the compiled ``trtllm`` CUDA ops (coefficient fold + folded paged score) for
every supported geometry. The original Triton score kernel has been deleted;
the unit tests validate the CUDA ops against an independent PyTorch oracle.
Selection and compaction live in their respective runtime modules. An
optional SM100 CuTe-DSL specialization (``triattention_cute_score.py``,
default off behind ``TRTLLM_TRIATTENTION_CUTE_SCORE=1``) can take over the
mean-aggregation score launch for one exactly-validated geometry.

House rules honored throughout:
  * fp32 math (loads up-cast to fp32, fp32 accumulators, fp32 score output).
  * int64 for every flat buffer offset that can exceed 2^31.
  * mask ragged valid-width tails (and frequency tails) in every load and store.
  * the kernels are vendored in this module (no lazy-load hub).
"""

from __future__ import annotations

import os
import warnings
from typing import List

import torch
import triton
import triton.language as tl

# --------------------------------------------------------------------------- #
# Scoring: trig-score every cached token across all dense layers.             #
# --------------------------------------------------------------------------- #


@triton.jit
def _prepare_mean_phase_kernel(
    round_starts,
    offsets,
    omega,
    mean_cos,
    mean_sin,
    NUM_FREQS: tl.constexpr,
    NUM_OFFSETS: tl.constexpr,
    F_BLOCK: tl.constexpr,
):
    """Collapse all offset phases for one request into reusable frequency means."""
    request = tl.program_id(0)
    frequency = tl.arange(0, F_BLOCK)
    frequency_mask = frequency < NUM_FREQS
    round_start = tl.load(round_starts + request)
    angular_frequency = tl.load(omega + frequency, mask=frequency_mask, other=0.0)
    cos_sum = tl.zeros((F_BLOCK,), tl.float32)
    sin_sum = tl.zeros((F_BLOCK,), tl.float32)
    for offset_index in tl.static_range(0, NUM_OFFSETS):
        offset = tl.load(offsets + offset_index)
        phase = (round_start + offset) * angular_frequency
        cos_sum += tl.cos(phase)
        sin_sum += tl.sin(phase)
    output_offset = request * NUM_FREQS + frequency
    scale = 1.0 / NUM_OFFSETS
    tl.store(mean_cos + output_offset, cos_sum * scale, mask=frequency_mask)
    tl.store(mean_sin + output_offset, sin_sum * scale, mask=frequency_mask)


def prepare_mean_phase(
    round_starts: torch.Tensor,
    offsets: torch.Tensor,
    omega: torch.Tensor,
    mean_cos: torch.Tensor,
    mean_sin: torch.Tensor,
    request_count: int,
) -> None:
    """Prepare mean score phases in one launch without intermediate tensors."""
    request_count = int(request_count)
    if request_count <= 0 or request_count > round_starts.numel():
        raise ValueError("phase preparation request count is outside its fixed buffers")
    num_freqs = int(omega.numel())
    num_offsets = int(offsets.numel())
    if (
        num_freqs <= 0
        or num_offsets <= 0
        or mean_cos.ndim != 2
        or mean_cos.shape[0] < request_count
        or mean_cos.shape[1] != num_freqs
        or mean_sin.shape != mean_cos.shape
        or any(
            tensor.device != round_starts.device for tensor in (offsets, omega, mean_cos, mean_sin)
        )
        or round_starts.dtype != torch.int32
        or any(tensor.dtype != torch.float32 for tensor in (offsets, omega, mean_cos, mean_sin))
    ):
        raise ValueError("phase preparation tensors do not share one valid FP32 geometry")
    _prepare_mean_phase_kernel[(request_count,)](
        round_starts,
        offsets,
        omega,
        mean_cos,
        mean_sin,
        NUM_FREQS=num_freqs,
        NUM_OFFSETS=num_offsets,
        F_BLOCK=triton.next_power_of_2(num_freqs),
        num_warps=1,
    )


def _launch_tri_score_perhead(
    group: "_FixedScoreGroup",
    request_count: int,
    num_segments: int,
    valid_seq_lens: torch.Tensor,
    valid_widths: torch.Tensor,
    round_starts_device: torch.Tensor,
    token_starts_device: torch.Tensor,
    mean_cos: torch.Tensor,
    mean_sin: torch.Tensor,
    *,
    score_aggregation: str,
) -> None:
    """Fold the per-round coefficients, then score paged KV via the C++ ops.

    The compiled ``trtllm`` score ops are THE implementation for every
    geometry this launcher accepts; unsupported inputs fail loudly inside the
    ops (TORCH_CHECK) instead of routing to another kernel. The unit tests
    validate them against an independent PyTorch oracle.
    """
    if score_aggregation not in ("mean", "max"):
        raise ValueError(f"unsupported score aggregation: {score_aggregation}")
    if not (
        hasattr(torch.ops.trtllm, "tri_attention_fold_score_coefficients")
        and hasattr(torch.ops.trtllm, "tri_attention_paged_score")
    ):
        raise RuntimeError(
            "this TensorRT-LLM build is missing the TriAttention score ops; rebuild the C++ "
            "th_common extension (there is deliberately no Triton fallback: a loud failure "
            "here beats silently scoring through a slower path)"
        )
    use_max = score_aggregation == "max"
    (
        num_q_heads,
        num_kv_heads,
        num_freqs,
        tokens_per_block,
        kv_factor,
        num_offsets,
        s_page,
        s_kv_head,
        s_slot,
        s_dim,
    ) = group.geometry_args
    # Mean aggregation collapses all offsets into mean_cos/mean_sin (one
    # coefficient plane); max keeps one c_re/c_im plane per offset because
    # max does not commute through the frequency sum.
    offset_planes = num_offsets if use_max else 1
    c_re, c_im, c_mlr = group._fold_coefficient_buffers(offset_planes)
    q_real, q_imag, mlr_coef = group.pointer_middle
    freq_scale_sq, omega, offsets = group.pointer_tail
    torch.ops.trtllm.tri_attention_fold_score_coefficients(
        c_re,
        c_im,
        c_mlr,
        q_real,
        q_imag,
        mlr_coef,
        freq_scale_sq,
        None if use_max else mean_cos.view(-1),
        None if use_max else mean_sin.view(-1),
        omega if use_max else None,
        offsets if use_max else None,
        round_starts_device if use_max else None,
        request_count,
        group._num_calibrated_layers,
        num_q_heads,
        num_freqs,
        offset_planes,
        use_max,
        # Per-layer dequant scales (quantized pools only, else None): the fold
        # multiplies them into the coefficient tables so the score op below
        # reads raw quantized elements at zero hot-loop cost.
        group._kv_scales,
    )
    pool_anchor, layer_base_addrs, block_offsets, seg_page_off, seg_req, seg_layer = (
        group.pointer_prefix
    )
    torch.ops.trtllm.tri_attention_paged_score(
        pool_anchor,
        layer_base_addrs,
        block_offsets,
        seg_page_off,
        seg_req,
        seg_layer,
        valid_seq_lens,
        valid_widths,
        token_starts_device,
        c_re,
        c_im,
        c_mlr,
        group.output,
        group.output_width,
        group.num_layers,
        request_count,
        group._num_calibrated_layers,
        num_q_heads,
        num_kv_heads,
        num_freqs,
        tokens_per_block,
        kv_factor,
        offset_planes,
        s_page,
        s_kv_head,
        s_slot,
        s_dim,
        num_segments,
        use_max,
        group._use_vectorized,
        # Validation-only here (presence must match the pool dtype; the fold
        # op above already consumed the values).
        group._kv_scales,
    )


class _FixedScoreGroup:
    """Persistent score metadata/output for one fixed geometry.

    Since the per-layer absolute-address ABI, ONE group can span dense layers
    living in DISTINCT storages with DISTINCT block tables. ``block_offsets``
    uses the native TRT-LLM attention layout and ``page_table_slots`` maps each
    scored layer to its V2 pool slot.

    LIFETIME CONTRACT: the group captures the scored layer pools as raw device
    addresses (``layer_base_addrs``) and keeps a reference only to the anchor
    pool (the score op's dtype witness). The caller owns ``layer_pools`` and must
    keep every scored pool alive for as long as it launches through this group
    (in production the V2 KV-cache manager does); a dropped pool leaves its
    address dangling and scores read allocator-recycled memory.
    """

    def __init__(
        self,
        layer_pools: List[torch.Tensor],
        layer_indices: List[int],
        max_requests: int,
        page_count: int,
        seq_len: int,
        num_q_heads: int,
        block_offsets: torch.Tensor,  # [num_pools, max_requests, 2, copied_blocks] int32
        page_table_slots: List[int],  # per scored layer: pool slot into block_offsets
        q_real_LHF: torch.Tensor,
        q_imag_LHF: torch.Tensor,
        mlr_coef_LHF: torch.Tensor,
        freq_scale_sq: torch.Tensor,
        omega: torch.Tensor,
        offsets: torch.Tensor,
        output_width: int,
        kv_scales: torch.Tensor | None = None,
    ) -> None:
        if not layer_indices or min(max_requests, page_count, seq_len) <= 0:
            raise ValueError("fixed score group requires non-empty positive geometry")
        if output_width <= 0 or output_width > seq_len:
            raise ValueError("fixed score group requires a decode width within its capacity")
        if len(page_table_slots) != len(layer_indices):
            raise ValueError("page_table_slots must align with layer_indices")
        self.max_requests = max_requests
        # Prompt lengths are per-request kernel inputs; this capacity only
        # sizes the widest possible decode window of the output buffer.
        self.output_width = int(output_width)
        self.num_layers = len(layer_indices)
        p0 = layer_pools[layer_indices[0]]
        if p0.ndim != 5:
            raise ValueError("fixed score group requires HND pools")
        device = p0.device
        q_real_LHF = q_real_LHF.to(device=device, dtype=torch.float32).contiguous()
        q_imag_LHF = q_imag_LHF.to(device=device, dtype=torch.float32).contiguous()
        mlr_coef_LHF = mlr_coef_LHF.to(device=device, dtype=torch.float32).contiguous()
        freq_scale_sq = freq_scale_sq.to(device=device, dtype=torch.float32).contiguous()
        omega = omega.to(device=device, dtype=torch.float32).contiguous()
        offsets = offsets.to(device=device, dtype=torch.float32).contiguous()
        _, kv_factor, num_kv_heads, tokens_per_block, head_dim = p0.shape
        if num_q_heads % num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        self.num_freqs = head_dim // 2
        strides = tuple(int(value) for value in p0.stride())
        self.geometry_args = (
            num_q_heads,
            num_kv_heads,
            self.num_freqs,
            tokens_per_block,
            kv_factor,
            int(offsets.numel()),
            strides[0],
            strides[2],
            strides[3],
            strides[4],
        )
        # Per-layer ABSOLUTE base addresses. Layers may live in distinct
        # storages (V2 TensorWrapper-per-layer); only geometry must be uniform.
        element_size = p0.element_size()
        bases_16b_aligned = True
        layer_base_addrs = torch.zeros(len(layer_pools), dtype=torch.int64, device=device)
        for layer in layer_indices:
            pool = layer_pools[layer]
            if (
                tuple(pool.shape[1:]) != tuple(p0.shape[1:])
                or tuple(pool.stride()) != strides
                or pool.dtype != p0.dtype
            ):
                raise ValueError("fixed score layers must share one uniform geometry")
            address = int(pool.data_ptr())
            if address % element_size:
                raise ValueError("fixed score layer base is not element-aligned")
            bases_16b_aligned &= address % 16 == 0
            layer_base_addrs[layer] = address
        # The score op runs 16-byte 8-frequency K loads when the fixed layout
        # guarantees aligned rows, and its strided scalar path otherwise.
        # Audited ONCE here: bases and strides never change for this group.
        strides_16b_aligned = all(
            (element_size * stride) % 16 == 0 for stride in (strides[0], strides[2], strides[3])
        )
        self._use_vectorized = (
            p0.dtype in (torch.bfloat16, torch.float16)
            and self.num_freqs % 8 == 0
            and strides[4] == 1
            and bases_16b_aligned
            and strides_16b_aligned
        )
        # Quantized (fp8/int8) pools are FUNCTIONAL-ONLY and scalar-path-only
        # (the dtype gate above already excludes them from the vectorized
        # path). Their per-layer dequantization scale is folded into the
        # score coefficients at launch time, so it must be present up front;
        # conversely, scales alongside a float pool would double-scale the
        # coefficients, so that pairing is rejected just as loudly.
        quantized_pool = p0.dtype in (torch.float8_e4m3fn, torch.int8)
        if quantized_pool and kv_scales is None:
            raise ValueError("quantized (fp8/int8) KV pools require per-layer kv_scales")
        if not quantized_pool and kv_scales is not None:
            raise ValueError("kv_scales are only valid for quantized (fp8/int8) KV pools")
        self._kv_scales = (
            None
            if kv_scales is None
            else kv_scales.to(device=device, dtype=torch.float32).contiguous().view(-1)
        )
        # Calibration tables span every model layer; segments index them by
        # ABSOLUTE layer id, so the fold covers the full calibrated extent.
        self._num_calibrated_layers = q_real_LHF.numel() // (int(num_q_heads) * self.num_freqs)
        # Segment layer ids index the fold tables ON DEVICE where they cannot
        # be range-checked; validate the extent once here, loudly.
        if min(layer_indices) < 0 or max(layer_indices) >= self._num_calibrated_layers:
            raise ValueError("scored layer index exceeds the calibrated layer extent")
        # Folded per-round coefficient tables, allocated on first launch and
        # keyed by plane count so switching aggregation (mean: one plane;
        # max: one plane per offset) re-shapes without churn.
        self._fold_buffers: dict = {}
        # The anchor pool is passed to the CUDA score op ONLY as its dtype
        # witness: the op recovers the pool element type from it and never
        # reads data through it.
        seg_req = torch.arange(max_requests, dtype=torch.int32, device=device).repeat_interleave(
            self.num_layers
        )
        seg_layer = torch.tensor(layer_indices, dtype=torch.int32, device=device).repeat(
            max_requests
        )
        # Each segment reads the K plane for one request from the same native
        # block-offset buffer used by TRT-LLM attention metadata preparation.
        if (
            block_offsets.ndim != 4
            or tuple(block_offsets.shape[1:3]) != (max_requests, 2)
            or block_offsets.shape[3] < page_count
            or block_offsets.dtype != torch.int32
            or block_offsets.device != device
        ):
            raise ValueError("block offsets do not match fixed score geometry")
        if not block_offsets.is_contiguous():
            raise ValueError("fixed score block offsets must be contiguous")
        slots_t = torch.tensor(page_table_slots, dtype=torch.int64, device=device)
        if int(slots_t.max()) >= int(block_offsets.shape[0]):
            raise ValueError("page table slot exceeds staged page-id planes")
        req_idx = torch.arange(max_requests, dtype=torch.int64, device=device).repeat_interleave(
            self.num_layers
        )
        slot_idx = slots_t.repeat(max_requests)
        seg_page_off = slot_idx * block_offsets.stride(0) + req_idx * block_offsets.stride(1)
        self.output = torch.empty(
            max_requests,
            self.num_layers,
            num_q_heads,
            self.output_width,
            dtype=torch.float32,
            device=device,
        )
        self.pointer_prefix = (
            p0,
            layer_base_addrs,
            block_offsets.view(-1),
            seg_page_off,
            seg_req,
            seg_layer,
        )
        self.pointer_middle = (
            q_real_LHF.view(-1),
            q_imag_LHF.view(-1),
            mlr_coef_LHF.view(-1),
        )
        self.pointer_tail = (freq_scale_sq, omega, offsets)
        # Optional SM100 CuTe mean-score specialization (see
        # triattention_cute_score.py), compiled by the first
        # ``prepare_cute_score`` call and default OFF behind the
        # TRTLLM_TRIATTENTION_CUTE_SCORE environment knob (read once here).
        # The CuTe runner encodes TMA descriptors from the actual pool
        # tensors, so pool references are retained ONLY when the knob is on;
        # the default path keeps the raw-address-only lifetime contract
        # documented above.
        self.seq_len = int(seq_len)
        self._cute_score_runner = None
        self._cute_score_attempted = False
        cute_score_enabled = os.environ.get("TRTLLM_TRIATTENTION_CUTE_SCORE", "0") == "1"
        self._cute_layer_pools = list(layer_pools) if cute_score_enabled else None
        self._cute_layer_indices = [int(layer) for layer in layer_indices]

    def prepare_cute_score(self, mean_cos: torch.Tensor, mean_sin: torch.Tensor) -> None:
        """Compile the optional SM100 CuTe mean-score specialization once.

        Call this outside CUDA graph capture: compilation allocates memory
        and synchronizes. With the environment knob unset (the default) or on
        any unsupported geometry this returns without importing the CuTe
        module, and every launch keeps using the compiled C++ score ops.

        Geometry reality check: the supported contract below (SM100 exactly,
        BF16 pools, 128-token pages, 64-element K rows / 32 frequencies,
        8 query heads per KV head) matches none of the current production
        models (Qwen3 uses 128-element K rows, 32-token pages, and 4 query
        heads per KV head; GPT-OSS uses 32-token pages), so today the kernel
        fires only on the synthetic unit-test geometry. This wiring exists to
        validate the kernel end to end while wider geometry support lands.
        """
        if self._cute_score_attempted:
            return
        self._cute_score_attempted = True
        if self._cute_layer_pools is None:
            return
        anchor = self.pointer_prefix[0]
        num_q_heads, num_kv_heads, num_freqs, tokens_per_block, kv_factor = self.geometry_args[:5]
        max_segments = self.max_requests * self.num_layers
        supported = (
            torch.cuda.get_device_capability(anchor.device) == (10, 0)
            and anchor.dtype == torch.bfloat16
            and kv_factor == 2
            and tokens_per_block == 128
            and num_freqs == 32
            and num_q_heads == num_kv_heads * 8
            and int(anchor.stride(-1)) == 1
            # The kernel computes flat score offsets in 32-bit arithmetic.
            and num_q_heads * max_segments * self.seq_len < 2**31
        )
        if not supported:
            return
        device = anchor.device
        try:
            from .triattention_cute_score import TriAttentionCuteScoreRunner

            # The kernel scores the FULL sequence from physical token zero
            # into its own head-major scratch (row = query head, column =
            # segment * seq_len + token); ``launch`` gathers each request's
            # decode window from that scratch into ``self.output``. All
            # buffers below are persistent because the compiled kernel
            # captures their device pointers.
            scratch = torch.empty(
                num_q_heads * max_segments * self.seq_len,
                dtype=torch.float32,
                device=device,
            )
            seg_seq_len = torch.zeros(max_segments, dtype=torch.int32, device=device)
            seg_out_offset = (
                torch.arange(max_segments, dtype=torch.int64, device=device) * self.seq_len
            ).to(torch.int32)
            gather_columns = torch.arange(self.output_width, dtype=torch.int64, device=device)
            self._cute_score_runner = TriAttentionCuteScoreRunner(
                layer_pools=self._cute_layer_pools,
                layer_indices=self._cute_layer_indices,
                max_requests=self.max_requests,
                num_layers=self.num_layers,
                seq_len=self.seq_len,
                # Always score from physical token zero: per-request prompt
                # windows are applied by the gather in ``launch`` instead of
                # one global page-aligned start.
                score_start=0,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                num_freqs=num_freqs,
                tokens_per_block=tokens_per_block,
                page_ids=self.pointer_prefix[2],
                seg_page_off=self.pointer_prefix[3],
                seg_req_id=self.pointer_prefix[4],
                seg_layer_id=self.pointer_prefix[5],
                seg_seq_len=seg_seq_len,
                seg_out_offset=seg_out_offset,
                q_real=self.pointer_middle[0],
                q_imag=self.pointer_middle[1],
                mlr_coef=self.pointer_middle[2],
                mean_cos=mean_cos,
                mean_sin=mean_sin,
                freq_scale_sq=self.pointer_tail[0],
                output=scratch,
            )
            self._cute_scratch = scratch
            self._cute_seg_seq_len = seg_seq_len
            self._cute_gather_columns = gather_columns.view(1, 1, 1, -1)
        except (ImportError, RuntimeError, ValueError, AssertionError) as error:
            warnings.warn(
                f"TriAttention CuTe score setup failed; using the C++ score ops: {error}",
                RuntimeWarning,
                stacklevel=2,
            )

    def _fold_coefficient_buffers(
        self, offset_planes: int
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        """Return (c_re, c_im, c_mlr) fold tables for one plane count.

        Sized on ``max_requests`` so any launch's active ``request_count``
        fits without reallocation (each launch folds only its active rows);
        c_mlr is offset independent so it never grows planes.
        """
        buffers = self._fold_buffers.get(offset_planes)
        if buffers is None:
            elements = (
                self.max_requests
                * self._num_calibrated_layers
                * int(self.geometry_args[0])
                * self.num_freqs
            )
            device = self.output.device
            c_re = torch.empty(offset_planes * elements, dtype=torch.float32, device=device)
            c_im = torch.empty_like(c_re)
            c_mlr = torch.empty(elements, dtype=torch.float32, device=device)
            buffers = (c_re, c_im, c_mlr)
            self._fold_buffers[offset_planes] = buffers
        return buffers

    def launch(
        self,
        request_count: int,
        valid_seq_lens: torch.Tensor,
        valid_widths: torch.Tensor,
        round_starts_device: torch.Tensor,
        token_starts_device: torch.Tensor,
        mean_cos: torch.Tensor,
        mean_sin: torch.Tensor,
        score_aggregation: str,
    ) -> torch.Tensor:
        """Return decode-only scores as ``[request, layer, head, token]``."""
        if request_count <= 0 or request_count > self.max_requests:
            raise ValueError("request count exceeds fixed score capacity")
        if (
            valid_widths.ndim != 1
            or valid_widths.numel() < request_count
            or valid_widths.dtype != torch.int32
            or valid_widths.device != self.output.device
        ):
            raise ValueError("score output lengths do not fit the keep-set selector")
        num_segments = request_count * self.num_layers
        if num_segments > 65535:
            # Segments sit on the y grid axis (CUDA caps y/z at 65535) so the
            # unbounded x axis can hold the token tiles of long sequences.
            raise ValueError("request*layer segment count exceeds the CUDA grid limit")
        output = self.output[:request_count]
        if score_aggregation == "mean":
            # Lazy compile covers groups used without their owning workspace
            # (unit tests); production compiles in the workspace constructor,
            # outside CUDA graph capture. Default off: one attribute check.
            self.prepare_cute_score(mean_cos, mean_sin)
            runner = self._cute_score_runner
            if runner is not None and runner.supports(request_count):
                # Stage per-segment valid lengths (segment = request x layer).
                torch.index_select(
                    valid_seq_lens,
                    0,
                    self.pointer_prefix[4][:num_segments],
                    out=self._cute_seg_seq_len[:num_segments],
                )
                runner.launch(request_count, mean_cos, mean_sin)
                # The kernel wrote full-sequence scores from physical token
                # zero into its head-major scratch. Gather each request's
                # decode window (starting at its pinned prompt length) into
                # the group output so callers see exactly the layout the C++
                # score ops produce. This costs one extra read+write of the
                # score volume per round, only on this opt-in path; columns
                # past a request's valid width carry unscored scratch data,
                # matching the C++ op, whose consumers mask by valid width.
                num_q_heads = int(self.geometry_args[0])
                source = (
                    self._cute_scratch[: num_q_heads * num_segments * self.seq_len]
                    .view(num_q_heads, request_count, self.num_layers, self.seq_len)
                    .permute(1, 2, 0, 3)
                )
                columns = (
                    token_starts_device[:request_count].to(torch.int64).view(-1, 1, 1, 1)
                    + self._cute_gather_columns
                )
                columns = columns.clamp_(max=self.seq_len - 1).expand(
                    request_count, self.num_layers, num_q_heads, self.output_width
                )
                torch.gather(source, 3, columns, out=output)
                return output
        _launch_tri_score_perhead(
            self,
            request_count,
            num_segments,
            valid_seq_lens,
            valid_widths,
            round_starts_device,
            token_starts_device,
            mean_cos,
            mean_sin,
            score_aggregation=score_aggregation,
        )
        return output


# --------------------------------------------------------------------------- #
# Selection: combine scores per mode, then finalize the top-k set.            #
# --------------------------------------------------------------------------- #


@triton.jit
def _score_row_stats_kernel(
    scores,
    valid_widths,
    row_mean,
    row_inv_std,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Compute one valid-prefix mean and inverse standard deviation per score row."""
    flat_row = tl.program_id(0)
    request = flat_row // ROWS
    valid_width = tl.load(valid_widths + request)
    score_row = scores + flat_row * WIDTH
    lane = tl.arange(0, BLOCK)
    score_sum = 0.0
    for start in tl.static_range(0, WIDTH, BLOCK):
        token = start + lane
        valid = token < valid_width
        value = tl.load(score_row + token, mask=valid, other=0.0).to(tl.float32)
        score_sum += tl.sum(value, axis=0)
    mean = score_sum / valid_width
    square_sum = 0.0
    for start in tl.static_range(0, WIDTH, BLOCK):
        token = start + lane
        valid = token < valid_width
        value = tl.load(score_row + token, mask=valid, other=0.0).to(tl.float32)
        centered = tl.where(valid, value - mean, 0.0)
        square_sum += tl.sum(centered * centered, axis=0)
    std = tl.sqrt(square_sum / valid_width)
    tl.store(row_mean + flat_row, mean)
    tl.store(row_inv_std + flat_row, 1.0 / tl.maximum(std, 1e-6))


@triton.jit
def _score_union_kernel(
    scores,
    valid_widths,
    row_mean,
    row_inv_std,
    combined,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    NORMALIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Normalize score rows and reduce them directly to one request-level union."""
    request = tl.program_id(0)
    token = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid_width = tl.load(valid_widths + request)
    valid_token = token < valid_width
    union_max = tl.full((BLOCK,), -float("inf"), tl.float32)
    for row in tl.range(0, ROWS):
        flat_row = request * ROWS + row
        value = tl.load(
            scores + flat_row * WIDTH + token,
            mask=valid_token,
            other=-float("inf"),
        ).to(tl.float32)
        if NORMALIZE:
            mean = tl.load(row_mean + flat_row)
            inv_std = tl.load(row_inv_std + flat_row)
            value = tl.where(valid_token, (value - mean) * inv_std, -float("inf"))
        union_max = tl.maximum(union_max, value)
    tl.store(combined + request * WIDTH + token, union_max, mask=token < WIDTH)


def prepare_union_scores(
    scores: torch.Tensor,
    valid_widths: torch.Tensor,
    row_mean: torch.Tensor,
    row_inv_std: torch.Tensor,
    combined: torch.Tensor,
    request_count: int,
    *,
    normalize_scores: bool,
) -> None:
    """Mask, normalize, and union-reduce score rows in two or three launches."""
    request_count = int(request_count)
    if not scores.is_cuda or scores.ndim != 3 or scores.dtype != torch.float32:
        raise ValueError("union score preparation requires contiguous CUDA FP32 rows")
    if not scores.is_contiguous() or request_count != scores.shape[0]:
        raise ValueError("union score preparation request geometry does not match")
    _, rows, width = scores.shape
    if (
        valid_widths.shape != (request_count,)
        or valid_widths.dtype != torch.int32
        or valid_widths.device != scores.device
        or row_mean.numel() < request_count * rows
        or row_inv_std.shape != row_mean.shape
        or combined.shape != (request_count, width)
    ):
        raise ValueError("union score preparation buffers do not match")
    stats_block = 256
    if normalize_scores:
        _score_row_stats_kernel[(request_count * rows,)](
            scores,
            valid_widths,
            row_mean,
            row_inv_std,
            ROWS=rows,
            WIDTH=width,
            BLOCK=stats_block,
            num_warps=4,
        )
    union_block = 32
    _score_union_kernel[(request_count, triton.cdiv(width, union_block))](
        scores,
        valid_widths,
        row_mean,
        row_inv_std,
        combined,
        ROWS=rows,
        WIDTH=width,
        NORMALIZE=normalize_scores,
        BLOCK=union_block,
        num_warps=1,
    )


@triton.jit
def _score_per_head_reduce_kernel(
    scores,
    valid_widths,
    row_mean,
    row_inv_std,
    selection_scores,
    selection_seq_lens,
    NUM_LAYERS: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    SELECTION_ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    PER_LAYER: tl.constexpr,
    NORMALIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Reduce query-head score rows into one selector row per KV-head domain."""
    request = tl.program_id(0)
    selection_row = tl.program_id(1)
    token_block = tl.program_id(2)
    token = token_block * BLOCK + tl.arange(0, BLOCK)
    valid_width = tl.load(valid_widths + request)
    valid_token = token < valid_width

    if token_block == 0:
        tl.store(
            selection_seq_lens + request * SELECTION_ROWS + selection_row,
            valid_width,
        )

    kv_head = selection_row % NUM_KV_HEADS
    if PER_LAYER:
        layer = selection_row // NUM_KV_HEADS
        reduced = tl.full((BLOCK,), -float("inf"), tl.float32)
        for query_in_group in tl.static_range(0, QUERY_GROUP_SIZE):
            query_head = kv_head * QUERY_GROUP_SIZE + query_in_group
            flat_row = (request * NUM_LAYERS + layer) * NUM_QUERY_HEADS + query_head
            value = tl.load(
                scores + flat_row * WIDTH + token,
                mask=valid_token,
                other=-float("inf"),
            ).to(tl.float32)
            if NORMALIZE:
                mean = tl.load(row_mean + flat_row)
                inv_std = tl.load(row_inv_std + flat_row)
                value = tl.where(valid_token, (value - mean) * inv_std, -float("inf"))
            reduced = tl.maximum(reduced, value)
    else:
        reduced = tl.zeros((BLOCK,), tl.float32)
        for layer in tl.static_range(0, NUM_LAYERS):
            layer_max = tl.full((BLOCK,), -float("inf"), tl.float32)
            for query_in_group in tl.static_range(0, QUERY_GROUP_SIZE):
                query_head = kv_head * QUERY_GROUP_SIZE + query_in_group
                flat_row = (request * NUM_LAYERS + layer) * NUM_QUERY_HEADS + query_head
                value = tl.load(
                    scores + flat_row * WIDTH + token,
                    mask=valid_token,
                    other=-float("inf"),
                ).to(tl.float32)
                if NORMALIZE:
                    mean = tl.load(row_mean + flat_row)
                    inv_std = tl.load(row_inv_std + flat_row)
                    value = tl.where(valid_token, (value - mean) * inv_std, -float("inf"))
                layer_max = tl.maximum(layer_max, value)
            reduced += layer_max
        reduced /= NUM_LAYERS

    output = (request * SELECTION_ROWS + selection_row) * WIDTH + token
    tl.store(selection_scores + output, reduced, mask=token < WIDTH)


def prepare_per_head_scores(
    scores: torch.Tensor,
    valid_widths: torch.Tensor,
    row_mean: torch.Tensor,
    row_inv_std: torch.Tensor,
    selection_scores: torch.Tensor,
    selection_seq_lens: torch.Tensor,
    request_count: int,
    *,
    num_kv_heads: int,
    per_layer: bool,
    normalize_scores: bool,
) -> None:
    """Normalize and reduce score rows for either per-head eviction mode."""
    request_count = int(request_count)
    num_kv_heads = int(num_kv_heads)
    if not scores.is_cuda or scores.ndim != 4 or scores.dtype != torch.float32:
        raise ValueError("per-head score preparation requires CUDA FP32 rows")
    if not scores.is_contiguous() or request_count != scores.shape[0]:
        raise ValueError("per-head score preparation request geometry does not match")
    _, num_layers, num_query_heads, width = scores.shape
    if num_kv_heads <= 0 or num_query_heads % num_kv_heads:
        raise ValueError("per-head score preparation requires valid GQA geometry")
    selection_rows = num_layers * num_kv_heads if per_layer else num_kv_heads
    if (
        valid_widths.shape != (request_count,)
        or valid_widths.dtype != torch.int32
        or valid_widths.device != scores.device
        or row_mean.numel() < request_count * num_layers * num_query_heads
        or row_inv_std.shape != row_mean.shape
        or selection_scores.shape != (request_count, selection_rows, width)
        or selection_scores.dtype != torch.float32
        or selection_scores.device != scores.device
        or selection_seq_lens.shape != (request_count, selection_rows)
        or selection_seq_lens.dtype != torch.int32
        or selection_seq_lens.device != scores.device
    ):
        raise ValueError("per-head score preparation buffers do not match")

    stats_block = 256
    rows = num_layers * num_query_heads
    if normalize_scores:
        _score_row_stats_kernel[(request_count * rows,)](
            scores,
            valid_widths,
            row_mean,
            row_inv_std,
            ROWS=rows,
            WIDTH=width,
            BLOCK=stats_block,
            num_warps=4,
        )
    reduction_block = 256
    _score_per_head_reduce_kernel[
        (request_count, selection_rows, triton.cdiv(width, reduction_block))
    ](
        scores,
        valid_widths,
        row_mean,
        row_inv_std,
        selection_scores,
        selection_seq_lens,
        NUM_LAYERS=num_layers,
        NUM_QUERY_HEADS=num_query_heads,
        NUM_KV_HEADS=num_kv_heads,
        QUERY_GROUP_SIZE=num_query_heads // num_kv_heads,
        SELECTION_ROWS=selection_rows,
        WIDTH=width,
        PER_LAYER=per_layer,
        NORMALIZE=normalize_scores,
        BLOCK=reduction_block,
        num_warps=4,
    )


@triton.jit
def _settle_ties_after_topk_kernel(
    scores,
    seq_lens,
    prompt_offsets,
    provisional_indices,
    output_indices,
    WIDTH: tl.constexpr,
    KEEP_COUNT: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Resolve boundary ties and emit increasing physical token indices."""
    row = tl.program_id(0)
    row_scores = scores + row * WIDTH
    row_selected = provisional_indices + row * KEEP_COUNT
    row_output = output_indices + row * OUTPUT_WIDTH
    # Scores are decode-relative; this row's pinned prompt length rebases the
    # emitted ordinals to absolute positions (per row, so one launch may mix
    # prompt lengths).
    prompt_len = tl.load(prompt_offsets + row)

    threshold = float("inf")
    for start in tl.static_range(0, KEEP_COUNT, BLOCK):
        selected_offset = start + tl.arange(0, BLOCK)
        selected_mask = selected_offset < KEEP_COUNT
        token_index = tl.load(
            row_selected + selected_offset,
            mask=selected_mask,
            other=0,
        )
        selected_score = tl.load(
            row_scores + token_index,
            mask=selected_mask,
            other=float("inf"),
        ).to(tl.float32)
        threshold = tl.minimum(threshold, tl.min(selected_score, axis=0))

    seq_len = tl.load(seq_lens + row)
    greater_count = 0
    for start in tl.static_range(0, WIDTH, BLOCK):
        token_index = start + tl.arange(0, BLOCK)
        valid = (token_index < WIDTH) & (token_index < seq_len)
        score = tl.load(
            row_scores + token_index,
            mask=valid,
            other=float("-inf"),
        ).to(tl.float32)
        greater_count += tl.sum((valid & (score > threshold)).to(tl.int32))

    tie_quota = KEEP_COUNT - greater_count
    output_count = 0
    ties_seen = 0
    for start in tl.static_range(0, WIDTH, BLOCK):
        token_index = start + tl.arange(0, BLOCK)
        valid = (token_index < WIDTH) & (token_index < seq_len)
        score = tl.load(
            row_scores + token_index,
            mask=valid,
            other=float("-inf"),
        ).to(tl.float32)
        greater = valid & (score > threshold)
        tied = valid & (score == threshold)
        tied_i32 = tied.to(tl.int32)
        tie_rank = ties_seen + tl.cumsum(tied_i32, axis=0) - tied_i32
        selected = greater | (tied & (tie_rank < tie_quota))
        selected_i32 = selected.to(tl.int32)
        write_offset = output_count + tl.cumsum(selected_i32, axis=0) - selected_i32
        tl.store(
            row_output + write_offset,
            token_index + prompt_len,
            mask=selected,
        )
        output_count += tl.sum(selected_i32)
        ties_seen += tl.sum(tied_i32)


# --------------------------------------------------------------------------- #
# Compaction: pack the kept ordinals into per-request move indices.           #
# --------------------------------------------------------------------------- #


@triton.jit
def _pack_compaction_sources_kernel(
    selected_indices,
    valid_seq_lens,
    dense_offsets,
    dense_indices,
    swa_offsets,
    swa_indices,
    DENSE_TOTAL: tl.constexpr,
    SWA_TOTAL: tl.constexpr,
    SELECTION_ROWS: tl.constexpr,
    SELECTION_STRIDE: tl.constexpr,
    KEEP_COUNT: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    SWA_WINDOW: tl.constexpr,
    UNION: tl.constexpr,
    PER_LAYER: tl.constexpr,
    HAS_SWA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Pack selected decode ordinals and protected tails for the C++ updater."""
    request = tl.program_id(0)
    domain = tl.program_id(1)
    move = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)

    dense_begin = tl.load(dense_offsets + request)
    dense_end = tl.load(dense_offsets + request + 1)
    dense_count = dense_end - dense_begin
    seq_len = tl.load(valid_seq_lens + request)

    if UNION:
        selection_domain = 0
    else:
        selection_domain = domain
    # Selection rows carry decode-only kept ordinals (already absolute), so
    # rows are prompt-length independent and one cohort may mix prompt sizes.
    selection_row = request * SELECTION_ROWS + selection_domain
    selected = tl.load(
        selected_indices + selection_row.to(tl.int64) * SELECTION_STRIDE + move,
        mask=move < KEEP_COUNT,
        other=0,
    )
    dense_source = tl.where(move < KEEP_COUNT, selected, seq_len + move - KEEP_COUNT)
    dense_output = domain.to(tl.int64) * DENSE_TOTAL + dense_begin.to(tl.int64) + move
    tl.store(dense_indices + dense_output, dense_source, mask=move < dense_count)

    if HAS_SWA:
        # Per-layer selection has one dense domain per (layer, head). SWA uses
        # one shared source row per head, so only the first layer writes it.
        if PER_LAYER:
            write_swa = domain < NUM_KV_HEADS
        else:
            write_swa = move >= 0
        swa_begin = tl.load(swa_offsets + request)
        swa_end = tl.load(swa_offsets + request + 1)
        swa_count = swa_end - swa_begin
        head = domain % NUM_KV_HEADS
        swa_output = head.to(tl.int64) * SWA_TOTAL + swa_begin.to(tl.int64) + move
        swa_source = seq_len - SWA_WINDOW + move
        tl.store(
            swa_indices + swa_output,
            swa_source,
            mask=write_swa & (move < swa_count),
        )

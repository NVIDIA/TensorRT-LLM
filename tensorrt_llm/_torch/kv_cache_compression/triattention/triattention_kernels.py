# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU kernels for the TriAttention KV-eviction pipeline.

The production path uses one fixed-shape trig-score launch across all dense
layers, CuTE-DSL TopK selection, and grouped C++ compaction. This module owns
the score kernel and its persistent launcher; selection and compaction live in
their respective runtime modules.

House rules honored throughout:
  * fp32 math (loads up-cast to fp32, fp32 accumulators, fp32 score output).
  * int64 for every page/stride offset that can exceed 2^31 (paged-pool reads).
  * mask seq tails not divisible by ``tokens_per_block`` (and freq/dim tails).
  * the kernels are vendored in this module (no lazy-load hub).
"""

from __future__ import annotations

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


@triton.jit
def _tri_score_perhead_kernel(
    pool_anchor_ptr,  # typed pool pointer; used ONLY to infer the element type
    #   for the int->pointer cast below (its data is never read through it).
    layer_base_addrs,  # [num_layers] int64: ABSOLUTE device address of each
    #   scored layer's HND base. Layers do NOT need to share one storage:
    #   each segment casts its own layer's address back to a typed pointer, so
    #   all address arithmetic stays inside that layer's own allocation.
    block_offsets_ptr,  # Native V2 [pool, request, K/V, block] int32 offsets.
    seg_page_off,  # [nseg] int64: offset of this segment's page table into
    #   block_offsets_ptr.
    # per-SEGMENT metadata (seg = req_slot*L_scored + layer_slot), idx by pid(0):
    seg_req_id,  # [nseg] int32: request slot (round_start / mean phase lookup)
    seg_layer_id,  # [nseg] int32: ABSOLUTE layer id (indexes layer_base_addrs + calib)
    req_seq_len,  # [num_requests] int32
    req_valid_width_out,  # [num_requests] int32: decode-only length for selection
    req_round_start,  # [num_requests] int32 logical token position
    req_token_start,  # [num_requests] int32: pinned prompt length; scoring
    #   starts at this decode-region origin, so prompt lengths may differ
    #   across the cohort.
    # per-LAYER calibration, [L,H,F] flattened layer-major:
    q_real_ptr,  # [L*H*F] fp32
    q_imag_ptr,  # [L*H*F] fp32
    mlr_coef_ptr,  # [L*H*F] fp32
    # per-REQUEST offset-collapsed phase ('mean' path), [num_requests,F] flattened:
    mean_cos_ptr,  # [num_requests*F] fp32
    mean_sin_ptr,  # [num_requests*F] fp32
    # shared freq vectors:
    freq_scale_sq_ptr,  # [F] fp32
    omega_ptr,  # [F] fp32 ('max' path only)
    offsets_ptr,  # [O] fp32 ('max' path only)
    out_ptr,  # [request, layer, query_head, decode_token] fp32
    output_width,
    # scalars uniform across the batch:
    num_layers,
    num_q_heads,
    num_kv_heads,
    num_freqs,  # F = head_dim // 2
    head_dim,
    tokens_per_block,
    kv_factor,
    num_offsets,  # O ('max' path only)
    # per-layer HND element strides (uniform across scored layers):
    s_page,
    s_kv_head,
    s_slot,
    s_dim,
    USE_MAX: tl.constexpr,
    T_BLOCK: tl.constexpr,
    F_BLOCK: tl.constexpr,
):
    # Token tiles ride the fastest grid axis: adjacent programs then walk
    # consecutive K pages of one (request, layer, head) and reuse its
    # calibration/phase rows in L2 (~2% faster than segment-major order).
    seg = tl.program_id(1)
    t_blk = tl.program_id(0)
    # KV heads are grid-parallel (axis 2): iterations of the former kv_head
    # loop shared NO data (each KV head reads its own K and writes its own
    # output rows), so hoisting it onto the grid multiplies parallelism with
    # zero extra HBM traffic. The q-in-group loop below stays inside the
    # program because it REUSES this head's K from registers (GQA dedup).
    kv_head = tl.program_id(2)

    req_id = tl.load(seg_req_id + seg)
    seq_len = tl.load(req_seq_len + req_id)
    token_start = tl.load(req_token_start + req_id)
    if (seg % num_layers == 0) & (t_blk == 0) & (kv_head == 0):
        tl.store(req_valid_width_out + req_id, seq_len - token_start)
    # Derive the ragged launch bound in the score program instead of staging
    # one replicated length and block count for every request/layer segment.
    n_tblk = (seq_len - token_start + T_BLOCK - 1) // T_BLOCK
    if t_blk >= n_tblk:
        return

    layer_id = tl.load(seg_layer_id + seg)
    rstart = tl.load(req_round_start + req_id)
    # This segment's layer base: an absolute address cast back to a pool-typed
    # pointer. TRT-LLM V2 exposes every layer as its own TensorWrapper storage,
    # so "element offset relative to one shared storage" does not exist; the
    # per-layer absolute address is the same device-pointer-array pattern the
    # C++ backends use (KVBlockArray / grouped-GEMM pointer arrays).
    layer_ptr = tl.load(layer_base_addrs + layer_id).to(
        tl.pointer_type(pool_anchor_ptr.dtype.element_ty)
    )
    page_off = tl.load(seg_page_off + seg)

    f = tl.arange(0, F_BLOCK)
    f_mask = f < num_freqs
    f64 = f.to(tl.int64)

    # ---- token tile of THIS segment ----
    t = t_blk * T_BLOCK + tl.arange(0, T_BLOCK)
    absolute_t = t + token_start
    t_mask = absolute_t < seq_len
    blk_in_seq = absolute_t // tokens_per_block
    slot = (absolute_t % tokens_per_block).to(tl.int64)
    # The native attention page-table copy encodes K offsets in units of the
    # underlying K/V role pages. Convert that value to the HND pool page inline
    # instead of materializing a second page table before scoring.
    encoded_page = tl.load(
        block_offsets_ptr + page_off + blk_in_seq,
        mask=t_mask,
        other=0,
    )
    phys_page = (encoded_page // kv_factor).to(tl.int64)

    # element offset into THIS layer's pool for (page, KEY=0, *, slot).
    # KEY half is kv_factor index 0 -> its stride term is 0 (matches reference).
    tok_base = phys_page * s_page + slot * s_slot  # [T_BLOCK] int64

    # per-request 'mean'-path phase + shared freq scale.
    mcos = tl.load(mean_cos_ptr + req_id * num_freqs + f, mask=f_mask, other=0.0)
    msin = tl.load(mean_sin_ptr + req_id * num_freqs + f, mask=f_mask, other=0.0)
    fss = tl.load(freq_scale_sq_ptr + f, mask=f_mask, other=0.0)

    # ---- PER-HEAD (position + mlr), GQA-deduped, NO head reduction ----
    # This program scores ONE KV head's token tile for the group_size q-heads
    # that share it. K (and |K|) is loaded ONCE and reused across the group;
    # h = kv_head*group_size + qg keeps query-head order 0..num_q_heads-1, so
    # every head's math is bit-for-bit identical to the looped variant.
    group_size = num_q_heads // num_kv_heads
    load_mask = t_mask[:, None] & f_mask[None, :]
    off_re = f64[None, :] * s_dim
    off_im = (num_freqs + f64[None, :]) * s_dim

    base = tok_base + kv_head.to(tl.int64) * s_kv_head  # [T_BLOCK]
    # paged K loaded ONCE for this KV head (shared by group_size q-heads).
    k_re = tl.load(layer_ptr + base[:, None] + off_re, mask=load_mask, other=0.0).to(tl.float32)
    k_im = tl.load(layer_ptr + base[:, None] + off_im, mask=load_mask, other=0.0).to(tl.float32)
    kmag = tl.sqrt(k_re * k_re + k_im * k_im)  # once per KV head

    qg = 0
    while qg < group_size:
        h = kv_head * group_size + qg
        calib_off = (layer_id.to(tl.int64) * num_q_heads + h) * num_freqs
        qre = tl.load(q_real_ptr + calib_off + f, mask=f_mask, other=0.0)
        qim = tl.load(q_imag_ptr + calib_off + f, mask=f_mask, other=0.0)
        mlrc = tl.load(mlr_coef_ptr + calib_off + f, mask=f_mask, other=0.0)

        # complex product Q . conj(K) -- the trig importance score.
        prod_real = qre[None, :] * k_re + qim[None, :] * k_im
        prod_imag = qim[None, :] * k_re - qre[None, :] * k_im

        if USE_MAX:
            # max over O offsets does NOT commute through the freq-sum;
            # explicit O loop reducing max over the per-offset F-sum.
            score = tl.full((T_BLOCK,), -float("inf"), tl.float32)
            o = 0
            while o < num_offsets:
                off = tl.load(offsets_ptr + o)
                om = tl.load(omega_ptr + f, mask=f_mask, other=0.0)
                phase = (rstart + off) * om
                cphase = tl.cos(phase)
                sphase = tl.sin(phase)
                per_f = fss[None, :] * (prod_real * cphase[None, :] - prod_imag * sphase[None, :])
                offset_score = tl.sum(tl.where(f_mask[None, :], per_f, 0.0), axis=1)
                score = tl.maximum(score, offset_score)
                o += 1
        else:
            # 'mean': offset loop collapsed into mean_cos/mean_sin.
            per_f = fss[None, :] * (prod_real * mcos[None, :] - prod_imag * msin[None, :])
            score = tl.sum(tl.where(f_mask[None, :], per_f, 0.0), axis=1)

        # position-INDEPENDENT MLR term (reuses the per-KV-head |K|).
        mlr_f = kmag * mlrc[None, :] * fss[None, :]
        mlr = tl.sum(tl.where(f_mask[None, :], mlr_f, 0.0), axis=1)

        # Segments are request-major then layer-major. Write the decode-only
        # score directly in the selector's [request, layer, head, token] layout.
        out_offset = (seg.to(tl.int64) * num_q_heads + h) * output_width + t
        tl.store(out_ptr + out_offset, score + mlr, mask=t_mask)
        qg += 1


def _launch_tri_score_perhead(
    grid: tuple,
    pointer_args: tuple,
    geometry_args: tuple,
    *,
    score_aggregation: str,
    token_block: int,
    num_freqs: int,
) -> None:
    """Launch the shared score ABI for eager and fixed metadata owners."""
    if score_aggregation not in ("mean", "max"):
        raise ValueError(f"unsupported score aggregation: {score_aggregation}")
    _tri_score_perhead_kernel[grid](
        *pointer_args,
        *geometry_args,
        USE_MAX=(score_aggregation == "max"),
        T_BLOCK=token_block,
        F_BLOCK=triton.next_power_of_2(num_freqs),
    )


class _FixedScoreGroup:
    """Persistent score metadata/output for one fixed geometry.

    Since the per-layer absolute-address ABI, ONE group can span dense layers
    living in DISTINCT storages with DISTINCT block tables. ``block_offsets``
    uses the native TRT-LLM attention layout and ``page_table_slots`` maps each
    scored layer to its V2 pool slot.
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
        output_width: int | None = None,
    ) -> None:
        if not layer_indices or min(max_requests, page_count, seq_len) <= 0:
            raise ValueError("fixed score group requires non-empty positive geometry")
        # Default: the whole sequence capacity is scorable.
        if output_width is None:
            output_width = int(seq_len)
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
        self.num_kv_heads = int(num_kv_heads)
        self.num_freqs = head_dim // 2
        strides = tuple(int(value) for value in p0.stride())
        self.geometry_args = (
            num_q_heads,
            num_kv_heads,
            self.num_freqs,
            head_dim,
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
            layer_base_addrs[layer] = address
        # The anchor pool is passed as a typed kernel argument ONLY so the
        # kernel can recover the element type for the int->pointer cast.
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
        self.token_block = 64
        self.max_ntblk = (self.output_width + self.token_block - 1) // self.token_block
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
        _launch_tri_score_perhead(
            (self.max_ntblk, num_segments, self.num_kv_heads),
            (
                *self.pointer_prefix,
                valid_seq_lens,
                valid_widths,
                round_starts_device,
                token_starts_device,
                *self.pointer_middle,
                mean_cos.view(-1),
                mean_sin.view(-1),
                *self.pointer_tail,
                output,
            ),
            (self.output_width, self.num_layers, *self.geometry_args),
            score_aggregation=score_aggregation,
            token_block=self.token_block,
            num_freqs=self.num_freqs,
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

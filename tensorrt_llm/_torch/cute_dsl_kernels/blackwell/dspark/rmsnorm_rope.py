# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused DSpark RMSNorm and adjacent-pair RoPE for Blackwell."""

from typing import Optional

import cutlass
import cutlass.cute as cute

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda


class DSparkRMSNormRoPEKernel:
    """Apply optional RMSNorm and last-dimension RoPE with one warp per row."""

    num_threads = cute.arch.WARP_SIZE
    rows_per_cta = 4

    def __init__(
        self,
        hidden_dim: int,
        rope_dim: int,
        num_heads: int,
        eps: float,
        apply_weight: bool,
        apply_rmsnorm: bool,
        inverse_rope: bool,
        norm_dim: int | None = None,
        out_rope_offset: int | None = None,
        write_nope: bool = True,
    ):
        if hidden_dim % self.num_threads != 0:
            raise ValueError(
                f"hidden_dim must be divisible by {self.num_threads}; got {hidden_dim}"
            )
        if rope_dim < 0 or rope_dim > hidden_dim or rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be even and in [0, {hidden_dim}]; got {rope_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive; got {num_heads}")
        self.hidden_dim = hidden_dim
        self.rope_dim = rope_dim
        self.nope_dim = hidden_dim - rope_dim
        self.rope_pairs = rope_dim // 2
        self.num_heads = num_heads
        self.eps = eps
        self.apply_weight = apply_weight
        self.apply_rmsnorm = apply_rmsnorm
        self.inverse_rope = inverse_rope
        # Which prefix the RMS is taken over, and how far the norm scale and the
        # weight reach. DSpark normalizes the whole row (the default, bit-identical
        # to before this knob existed); DeepSeek-style MLA normalizes only the
        # kv_lora_rank latent and leaves k_pe raw, which is nope_dim here.
        self.norm_dim = hidden_dim if norm_dim is None else norm_dim
        if self.norm_dim not in (hidden_dim, self.nope_dim):
            raise ValueError(
                f"norm_dim must be hidden_dim ({hidden_dim}) or nope_dim "
                f"({self.nope_dim}); got {self.norm_dim}"
            )
        self.norm_covers_rope = self.norm_dim == hidden_dim
        # Where the rotated pairs land in `output`. None means "same column as in
        # the input", which is every caller that rewrites a row in place. The MLA
        # drafter instead reads a 192-wide q row and writes into the rope half of
        # a 576-wide fused query, so its input and output offsets differ and it
        # wants the nope half left to bmm_out.
        self.out_rope_offset = self.nope_dim if out_rope_offset is None else out_rope_offset
        if self.out_rope_offset < 0 or self.out_rope_offset % 2 != 0:
            raise ValueError(
                f"out_rope_offset must be even and non-negative; got {self.out_rope_offset}"
            )
        self.write_nope = write_nope
        if not self.write_nope and self.norm_covers_rope:
            # The rope lanes would need the RMS scale computed over a row whose
            # nope half is never written -- readable, but no caller wants it and
            # it would silently pair a raw-passthrough offset with a scaled rope.
            raise ValueError("write_nope=False requires norm_dim == nope_dim")
        if self.norm_dim % self.num_threads != 0:
            raise ValueError(
                f"norm_dim must be divisible by {self.num_threads}; got {self.norm_dim}"
            )
        self.norm_elements_per_thread = self.norm_dim // self.num_threads
        if self.nope_dim % self.num_threads != 0:
            raise ValueError(
                f"nope_dim must be divisible by {self.num_threads}; got {self.nope_dim}"
            )
        if self.rope_pairs % self.num_threads != 0:
            raise ValueError(
                f"rope pairs must be divisible by {self.num_threads}; got {self.rope_pairs}"
            )
        self.elements_per_thread = hidden_dim // self.num_threads
        self.nope_elements_per_thread = self.nope_dim // self.num_threads
        self.pairs_per_thread = self.rope_pairs // self.num_threads
        self.loaded_elements_per_thread = max(
            self.norm_elements_per_thread if apply_rmsnorm else 0,
            self.nope_elements_per_thread if write_nope else 0,
        )

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: Optional[cute.Tensor],
        freqs: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """weight is None exactly when apply_weight is False.

        Both reads sit behind const_expr(self.apply_weight), so with the flag
        off the operand never reaches the generated signature -- which is the
        point: a rope-only caller has no weight and should not have to
        materialize one. The compile wrappers derive both from the same flag,
        so they cannot disagree.
        """
        self.kernel(x, weight, freqs, output).launch(
            grid=[(x.shape[0] + self.rows_per_cta - 1) // self.rows_per_cta, 1, 1],
            block=[self.num_threads * self.rows_per_cta, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: Optional[cute.Tensor],
        freqs: cute.Tensor,
        output: cute.Tensor,
    ):
        thread, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        tidx = thread % self.num_threads
        row = cta * self.rows_per_cta + thread // self.num_threads
        if row < x.shape[0]:
            # Issue every load before the first store: output may alias x, so no
            # load could otherwise move above an earlier store.
            values = []
            for item in cutlass.range_constexpr(self.loaded_elements_per_thread):
                values.append(cutlass.Float32(x[row, tidx + item * self.num_threads]))
            nope_weights = []
            if cutlass.const_expr(self.write_nope and self.apply_weight):
                for item in cutlass.range_constexpr(self.nope_elements_per_thread):
                    nope_weights.append(cutlass.Float32(weight[tidx + item * self.num_threads]))
            pairs = []
            if cutlass.const_expr(self.rope_pairs > 0):
                freq_row = row // self.num_heads
                for item in cutlass.range_constexpr(self.pairs_per_thread):
                    pair = tidx + item * self.num_threads
                    real_dim = self.nope_dim + pair * 2
                    imag_dim = real_dim + 1
                    real_weight = imag_weight = None
                    if cutlass.const_expr(self.norm_covers_rope and self.apply_weight):
                        real_weight = cutlass.Float32(weight[real_dim])
                        imag_weight = cutlass.Float32(weight[imag_dim])
                    pairs.append(
                        (
                            cutlass.Float32(x[row, real_dim]),
                            cutlass.Float32(x[row, imag_dim]),
                            real_weight,
                            imag_weight,
                            cutlass.Float32(freqs[freq_row, pair, 0]),
                            cutlass.Float32(freqs[freq_row, pair, 1]),
                        )
                    )

            inverse_rms = cutlass.Float32(1.0)
            if cutlass.const_expr(self.apply_rmsnorm):
                sum_sq = cutlass.Float32(0.0)
                for item in cutlass.range_constexpr(self.norm_elements_per_thread):
                    value = values[item]
                    sum_sq += value * value
                sum_sq = cute.arch.warp_reduction_sum(sum_sq)
                inverse_rms = cute.math.rsqrt(sum_sq / self.norm_dim + self.eps)

            if cutlass.const_expr(self.write_nope):
                for item in cutlass.range_constexpr(self.nope_elements_per_thread):
                    dim = tidx + item * self.num_threads
                    value = values[item] * inverse_rms
                    if cutlass.const_expr(self.apply_weight):
                        value *= nope_weights[item]
                    output[row, dim] = value.to(output.element_type)

            for item in cutlass.range_constexpr(len(pairs)):
                real, imag, real_weight, imag_weight, cos, sin = pairs[item]
                out_real = self.out_rope_offset + (tidx + item * self.num_threads) * 2
                out_imag = out_real + 1
                # Outside norm_dim the rope lanes are passed through raw: no RMS
                # scale, no weight. weight is only norm_dim long in that case.
                if cutlass.const_expr(self.norm_covers_rope):
                    real *= inverse_rms
                    imag *= inverse_rms
                    if cutlass.const_expr(self.apply_weight):
                        real *= real_weight
                        imag *= imag_weight
                if cutlass.const_expr(self.inverse_rope):
                    sin = -sin
                output[row, out_real] = (real * cos - imag * sin).to(output.element_type)
                output[row, out_imag] = (imag * cos + real * sin).to(output.element_type)

    @cute.jit
    def store_normalized_row(
        self,
        x: cute.Tensor,
        row: cutlass.Int32,
        weight: cute.Tensor,
        freqs: cute.Tensor,
        freq_row: cutlass.Int32,
        dst: cute.Tensor,
    ):
        """Store the weighted RMSNorm and forward RoPE of ``x[row]`` into the row ``dst``."""
        tidx = cute.arch.thread_idx()[0] % self.num_threads
        # All loads precede the stores, which may alias them.
        values = []
        for item in cutlass.range_constexpr(self.elements_per_thread):
            values.append(cutlass.Float32(x[row, tidx + item * self.num_threads]))
        nope_weights = []
        for item in cutlass.range_constexpr(self.nope_elements_per_thread):
            nope_weights.append(cutlass.Float32(weight[tidx + item * self.num_threads]))
        pairs = []
        for item in cutlass.range_constexpr(self.pairs_per_thread):
            pair = tidx + item * self.num_threads
            real_dim = self.nope_dim + pair * 2
            imag_dim = real_dim + 1
            pairs.append(
                (
                    cutlass.Float32(x[row, real_dim]),
                    cutlass.Float32(x[row, imag_dim]),
                    cutlass.Float32(weight[real_dim]),
                    cutlass.Float32(weight[imag_dim]),
                    cutlass.Float32(freqs[freq_row, pair, 0]),
                    cutlass.Float32(freqs[freq_row, pair, 1]),
                )
            )

        sum_sq = cutlass.Float32(0.0)
        for item in cutlass.range_constexpr(self.elements_per_thread):
            value = values[item]
            sum_sq += value * value
        sum_sq = cute.arch.warp_reduction_sum(sum_sq)
        inverse_rms = cute.math.rsqrt(sum_sq / self.hidden_dim + self.eps)

        for item in cutlass.range_constexpr(self.nope_elements_per_thread):
            value = values[item] * inverse_rms
            value *= nope_weights[item]
            dst[tidx + item * self.num_threads] = value.to(dst.element_type)

        for item in cutlass.range_constexpr(self.pairs_per_thread):
            real, imag, real_weight, imag_weight, cos, sin = pairs[item]
            real_dim = self.nope_dim + (tidx + item * self.num_threads) * 2
            real = real * inverse_rms
            imag = imag * inverse_rms
            real *= real_weight
            imag *= imag_weight
            dst[real_dim] = (real * cos - imag * sin).to(dst.element_type)
            dst[real_dim + 1] = (imag * cos + real * sin).to(dst.element_type)


class DSparkRMSNormRoPECacheWriteKernel(DSparkRMSNormRoPEKernel):
    """Apply RMSNorm/RoPE and scatter each row into the rolling KV cache."""

    def __init__(self, hidden_dim: int, rope_dim: int, eps: float, window_size: int):
        super().__init__(hidden_dim, rope_dim, 1, eps, True, True, False)
        self.window_size = window_size

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        freqs: cute.Tensor,
        kv_cache: cute.Tensor,
        slots: cute.Tensor,
        start_pos: cute.Tensor,
        slots_i32: cute.Tensor,
        cache_seqs: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            x,
            weight,
            freqs,
            kv_cache,
            slots,
            start_pos,
            slots_i32,
            cache_seqs,
        ).launch(
            grid=[x.shape[0], 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        freqs: cute.Tensor,
        kv_cache: cute.Tensor,
        slots: cute.Tensor,
        start_pos: cute.Tensor,
        slots_i32: cute.Tensor,
        cache_seqs: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        slot = cutlass.Int32(slots[row])
        cache_seq = cutlass.Int32(start_pos[row])
        cache_pos = cache_seq % self.window_size

        self.store_normalized_row(x, row, weight, freqs, row, kv_cache[slot, cache_pos, None])
        if tidx == 0:
            slots_i32[row] = slot
            cache_seqs[row] = cache_seq


class DSparkRMSNormRoPEDraftBlockKernel(DSparkRMSNormRoPEKernel):
    """Apply RMSNorm/RoPE and materialize a fixed-size zero-padded draft block."""

    def __init__(
        self,
        hidden_dim: int,
        rope_dim: int,
        eps: float,
        block_size: int,
        storage_size: int,
    ):
        super().__init__(hidden_dim, rope_dim, 1, eps, True, True, False)
        self.block_size = block_size
        self.storage_size = storage_size

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        freqs: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        # One CTA per real row, plus one that zero-fills the padding rows.
        self.kernel(x, weight, freqs, output).launch(
            grid=[output.shape[0], min(self.block_size + 1, self.storage_size), 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        freqs: cute.Tensor,
        output: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        batch, page_row, _ = cute.arch.block_idx()

        if page_row < self.block_size:
            row = batch * self.block_size + page_row
            self.store_normalized_row(x, row, weight, freqs, row, output[batch, page_row, None])
        else:
            for pad_row in cutlass.range_constexpr(self.block_size, self.storage_size):
                for item in cutlass.range_constexpr(self.elements_per_thread):
                    dim = tidx + item * self.num_threads
                    output[batch, pad_row, dim] = cutlass.BFloat16(0.0)


class DSparkBackfillKernel(DSparkRMSNormRoPEKernel):
    """Write each valid ``(interim token, stage)`` RMSNorm/RoPE row into its rolling-window slot."""

    def __init__(
        self, hidden_dim: int, rope_dim: int, eps: float, window_size: int, num_stages: int
    ):
        super().__init__(hidden_dim, rope_dim, 1, eps, True, True, False)
        self.window_size = window_size
        self.num_stages = num_stages

    @cute.jit
    def __call__(
        self,
        kv: cute.Tensor,
        weight: cute.Tensor,
        freqs: cute.Tensor,
        mask: cute.Tensor,
        positions: cute.Tensor,
        slots: cute.Tensor,
        windows: cute.Tensor,
        interim: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(kv, weight, freqs, mask, positions, slots, windows, interim).launch(
            grid=[mask.shape[0], self.num_stages, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        kv: cute.Tensor,
        weight: cute.Tensor,
        freqs: cute.Tensor,
        mask: cute.Tensor,
        positions: cute.Tensor,
        slots: cute.Tensor,
        windows: cute.Tensor,
        interim: cutlass.Int32,
    ):
        token, stage, _ = cute.arch.block_idx()
        if mask[token] != 0:
            position = cutlass.Int32(positions[token])
            slot = cutlass.Int32(slots[token // interim])
            self.store_normalized_row(
                kv,
                token * self.num_stages + stage,
                weight[stage, None],
                freqs,
                position,
                windows[slot, stage, position % self.window_size, None],
            )

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


@cute.jit
def _load_norm_items(
    src: cute.Tensor,
    base,
    weight: cute.Tensor,
    lane,
    values: cute.Tensor,
    scales: cute.Tensor,
    items: cutlass.Constexpr,
):
    """Lane ``lane``'s columns ``lane + 32 i`` of the normalized prefix of row ``src`` (from column
    ``base``) and of ``weight`` into registers."""
    for item in cutlass.range_constexpr(items):
        dim = lane + item * cute.arch.WARP_SIZE
        values[item] = cutlass.Float32(src[base + dim])
        scales[item] = cutlass.Float32(weight[dim])


@cute.jit
def _load_rope_pairs(
    src: cute.Tensor,
    base,
    freqs: cute.Tensor,
    position,
    lane,
    reals: cute.Tensor,
    imags: cute.Tensor,
    coss: cute.Tensor,
    sins: cute.Tensor,
    items: cutlass.Constexpr,
):
    """Lane ``lane``'s adjacent pairs ``lane + 32 i`` of row ``src`` from column ``base`` and their
    ``freqs[position]`` entries into registers."""
    for item in cutlass.range_constexpr(items):
        pair = lane + item * cute.arch.WARP_SIZE
        reals[item] = cutlass.Float32(src[base + pair * 2])
        imags[item] = cutlass.Float32(src[base + pair * 2 + 1])
        coss[item] = cutlass.Float32(freqs[position, pair, 0])
        sins[item] = cutlass.Float32(freqs[position, pair, 1])


@cute.jit
def _store_rms_normalized(
    dst: cute.Tensor,
    lane,
    values: cute.Tensor,
    scales: cute.Tensor,
    norm_dim: cutlass.Constexpr,
    eps: cutlass.Constexpr,
    items: cutlass.Constexpr,
):
    """RMSNorm of the warp's ``norm_dim`` loaded values (squares summed in item order, then the
    warp reduction), scaled, into columns ``[0, norm_dim)`` of row ``dst``."""
    sum_sq = cutlass.Float32(0.0)
    for item in cutlass.range_constexpr(items):
        value = values[item]
        sum_sq += value * value
    sum_sq = cute.arch.warp_reduction_sum(sum_sq)
    inverse_rms = cute.math.rsqrt(sum_sq / norm_dim + eps)
    for item in cutlass.range_constexpr(items):
        dim = lane + item * cute.arch.WARP_SIZE
        value = values[item] * inverse_rms
        value *= scales[item]
        dst[dim] = value.to(dst.element_type)


@cute.jit
def _store_rotated_pairs(
    dst: cute.Tensor,
    base,
    lane,
    reals: cute.Tensor,
    imags: cute.Tensor,
    coss: cute.Tensor,
    sins: cute.Tensor,
    items: cutlass.Constexpr,
):
    """The loaded pairs rotated (``real * cos - imag * sin``, ``imag * cos + real * sin``) into
    row ``dst`` from column ``base``."""
    for item in cutlass.range_constexpr(items):
        pair = lane + item * cute.arch.WARP_SIZE
        real = reals[item]
        imag = imags[item]
        cos = coss[item]
        sin = sins[item]
        dst[base + pair * 2] = (real * cos - imag * sin).to(dst.element_type)
        dst[base + pair * 2 + 1] = (imag * cos + real * sin).to(dst.element_type)


class DSparkRMSNormRoPEKernel:
    """Apply optional RMSNorm and last-dimension RoPE with one warp per row."""

    num_threads = cute.arch.WARP_SIZE

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
            grid=[x.shape[0], 1, 1],
            block=[self.num_threads, 1, 1],
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
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()

        inverse_rms = cutlass.Float32(1.0)
        if cutlass.const_expr(self.apply_rmsnorm):
            sum_sq = cutlass.Float32(0.0)
            for item in cutlass.range_constexpr(self.norm_elements_per_thread):
                dim = tidx + item * self.num_threads
                value = cutlass.Float32(x[row, dim])
                sum_sq += value * value
            sum_sq = cute.arch.warp_reduction_sum(sum_sq)
            inverse_rms = cute.math.rsqrt(sum_sq / self.norm_dim + self.eps)

        if cutlass.const_expr(self.write_nope):
            for item in cutlass.range_constexpr(self.nope_elements_per_thread):
                dim = tidx + item * self.num_threads
                value = cutlass.Float32(x[row, dim]) * inverse_rms
                if cutlass.const_expr(self.apply_weight):
                    value *= cutlass.Float32(weight[dim])
                output[row, dim] = value.to(output.element_type)

        if cutlass.const_expr(self.rope_pairs > 0):
            freq_row = row // self.num_heads
            for item in cutlass.range_constexpr(self.pairs_per_thread):
                pair = tidx + item * self.num_threads
                real_dim = self.nope_dim + pair * 2
                imag_dim = real_dim + 1
                out_real = self.out_rope_offset + pair * 2
                out_imag = out_real + 1
                real = cutlass.Float32(x[row, real_dim])
                imag = cutlass.Float32(x[row, imag_dim])
                # Outside norm_dim the rope lanes are passed through raw: no RMS
                # scale, no weight. weight is only norm_dim long in that case.
                if cutlass.const_expr(self.norm_covers_rope):
                    real *= inverse_rms
                    imag *= inverse_rms
                    if cutlass.const_expr(self.apply_weight):
                        real *= cutlass.Float32(weight[real_dim])
                        imag *= cutlass.Float32(weight[imag_dim])
                cos = cutlass.Float32(freqs[freq_row, pair, 0])
                sin = cutlass.Float32(freqs[freq_row, pair, 1])
                if cutlass.const_expr(self.inverse_rope):
                    sin = -sin
                output[row, out_real] = (real * cos - imag * sin).to(output.element_type)
                output[row, out_imag] = (imag * cos + real * sin).to(output.element_type)


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

        if tidx == 0:
            slots_i32[row] = slot
            cache_seqs[row] = cache_seq

        sum_sq = cutlass.Float32(0.0)
        for item in cutlass.range_constexpr(self.elements_per_thread):
            dim = tidx + item * self.num_threads
            value = cutlass.Float32(x[row, dim])
            sum_sq += value * value
        sum_sq = cute.arch.warp_reduction_sum(sum_sq)
        inverse_rms = cute.math.rsqrt(sum_sq / self.hidden_dim + self.eps)

        for item in cutlass.range_constexpr(self.nope_elements_per_thread):
            dim = tidx + item * self.num_threads
            value = cutlass.Float32(x[row, dim]) * inverse_rms
            value *= cutlass.Float32(weight[dim])
            kv_cache[slot, cache_pos, dim] = value.to(kv_cache.element_type)

        for item in cutlass.range_constexpr(self.pairs_per_thread):
            pair = tidx + item * self.num_threads
            real_dim = self.nope_dim + pair * 2
            imag_dim = real_dim + 1
            real = cutlass.Float32(x[row, real_dim]) * inverse_rms
            imag = cutlass.Float32(x[row, imag_dim]) * inverse_rms
            real *= cutlass.Float32(weight[real_dim])
            imag *= cutlass.Float32(weight[imag_dim])
            cos = cutlass.Float32(freqs[row, pair, 0])
            sin = cutlass.Float32(freqs[row, pair, 1])
            kv_cache[slot, cache_pos, real_dim] = (real * cos - imag * sin).to(
                kv_cache.element_type
            )
            kv_cache[slot, cache_pos, imag_dim] = (imag * cos + real * sin).to(
                kv_cache.element_type
            )


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
        self.kernel(x, weight, freqs, output).launch(
            grid=[output.shape[0], self.storage_size, 1],
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
            sum_sq = cutlass.Float32(0.0)
            for item in cutlass.range_constexpr(self.elements_per_thread):
                dim = tidx + item * self.num_threads
                value = cutlass.Float32(x[row, dim])
                sum_sq += value * value
            sum_sq = cute.arch.warp_reduction_sum(sum_sq)
            inverse_rms = cute.math.rsqrt(sum_sq / self.hidden_dim + self.eps)

            for item in cutlass.range_constexpr(self.nope_elements_per_thread):
                dim = tidx + item * self.num_threads
                value = cutlass.Float32(x[row, dim]) * inverse_rms
                value *= cutlass.Float32(weight[dim])
                output[batch, page_row, dim] = value.to(output.element_type)

            for item in cutlass.range_constexpr(self.pairs_per_thread):
                pair = tidx + item * self.num_threads
                real_dim = self.nope_dim + pair * 2
                imag_dim = real_dim + 1
                real = cutlass.Float32(x[row, real_dim]) * inverse_rms
                imag = cutlass.Float32(x[row, imag_dim]) * inverse_rms
                real *= cutlass.Float32(weight[real_dim])
                imag *= cutlass.Float32(weight[imag_dim])
                cos = cutlass.Float32(freqs[row, pair, 0])
                sin = cutlass.Float32(freqs[row, pair, 1])
                output[batch, page_row, real_dim] = (real * cos - imag * sin).to(
                    output.element_type
                )
                output[batch, page_row, imag_dim] = (imag * cos + real * sin).to(
                    output.element_type
                )
        else:
            for item in cutlass.range_constexpr(self.elements_per_thread):
                dim = tidx + item * self.num_threads
                output[batch, page_row, dim] = cutlass.BFloat16(0.0)


class DSparkRMSNormFP8QuantKernel:
    """Apply RMSNorm and emit packed 1x128 MXFP8 activations."""

    num_threads = cute.arch.WARP_SIZE
    quant_block = 128

    def __init__(self, hidden_dim: int, eps: float):
        if hidden_dim % self.quant_block != 0:
            raise ValueError(
                f"hidden_dim must be divisible by {self.quant_block}; got {hidden_dim}"
            )
        self.hidden_dim = hidden_dim
        self.num_quant_blocks = hidden_dim // self.quant_block
        self.values_per_thread = self.quant_block // self.num_threads
        self.eps = eps

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        output_scale: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(x, weight, output, output_scale).launch(
            grid=[x.shape[0], 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        output_scale: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()

        if cutlass.dynamic_expr(row < x.shape[0]):
            sum_sq = cutlass.Float32(0.0)
            for quant_block in cutlass.range_constexpr(self.num_quant_blocks):
                block_base = quant_block * self.quant_block
                for item in cutlass.range_constexpr(self.values_per_thread):
                    dim = block_base + tidx * self.values_per_thread + item
                    value = cutlass.Float32(x[row, dim])
                    sum_sq += value * value
            sum_sq = cute.arch.warp_reduction_sum(sum_sq)
            inverse_rms = cute.math.rsqrt(sum_sq / self.hidden_dim + self.eps)

            for quant_block in cutlass.range_constexpr(self.num_quant_blocks):
                block_base = quant_block * self.quant_block
                values = cute.make_rmem_tensor((self.values_per_thread,), cutlass.BFloat16)
                local_absmax = cutlass.Float32(0.0)
                for item in cutlass.range_constexpr(self.values_per_thread):
                    dim = block_base + tidx * self.values_per_thread + item
                    value = (
                        cutlass.Float32(x[row, dim]) * inverse_rms * cutlass.Float32(weight[dim])
                    ).to(cutlass.BFloat16)
                    values[item] = value
                    value_f32 = cutlass.Float32(value)
                    local_absmax = cute.arch.fmax(
                        local_absmax,
                        cute.arch.fmax(value_f32, -value_f32),
                    )

                block_absmax = cute.arch.warp_reduction_max(local_absmax)
                block_absmax = cute.arch.fmax(block_absmax, cutlass.Float32(1e-10))
                scale = (block_absmax * cutlass.Float32(1.0 / 448.0)).to(cutlass.Float8E8M0FNU)
                quant_scale = cute.arch.rcp_approx(cutlass.Float32(scale))

                for item in cutlass.range_constexpr(self.values_per_thread):
                    dim = block_base + tidx * self.values_per_thread + item
                    output[row, dim] = (cutlass.Float32(values[item]) * quant_scale).to(
                        output.element_type
                    )

                if tidx == 0:
                    scale_offset = (
                        (row // 128) * self.num_quant_blocks * 512
                        + quant_block * 512
                        + (row % 32) * 16
                        + ((row % 128) // 32) * 4
                    )
                    for replica in cutlass.range_constexpr(4):
                        output_scale[scale_offset + replica] = scale


class DSparkConcatFP8QuantKernel:
    """Concatenate two row ranges while producing packed 1x128 MXFP8."""

    num_threads = 4 * cute.arch.WARP_SIZE
    num_warps = 4
    quant_block = 128
    values_per_thread = quant_block // cute.arch.WARP_SIZE

    def __init__(self, hidden_dim: int):
        if hidden_dim % self.quant_block != 0:
            raise ValueError(
                f"hidden_dim must be divisible by {self.quant_block}; got {hidden_dim}"
            )
        self.hidden_dim = hidden_dim
        self.num_quant_blocks = hidden_dim // self.quant_block

    @cute.jit
    def __call__(
        self,
        first: cute.Tensor,
        second: cute.Tensor,
        output: cute.Tensor,
        output_scale: cute.Tensor,
        stream: cuda.CUstream,
    ):
        total_rows = first.shape[0] + second.shape[0]
        self.kernel(first, second, output, output_scale).launch(
            grid=[self.num_quant_blocks, cute.ceil_div(total_rows, self.num_warps), 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        first: cute.Tensor,
        second: cute.Tensor,
        output: cute.Tensor,
        output_scale: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        quant_block, row_block, _ = cute.arch.block_idx()
        warp = tidx // cute.arch.WARP_SIZE
        lane = tidx % cute.arch.WARP_SIZE
        row = row_block * self.num_warps + warp
        total_rows = first.shape[0] + second.shape[0]
        scale = cutlass.Float8E8M0FNU(0.0)

        if cutlass.dynamic_expr(row < total_rows):
            values = cute.make_rmem_tensor((self.values_per_thread,), cutlass.BFloat16)
            local_absmax = cutlass.Float32(0.0)
            block_base = quant_block * self.quant_block
            if cutlass.dynamic_expr(row < first.shape[0]):
                for item in cutlass.range_constexpr(self.values_per_thread):
                    dim = block_base + lane * self.values_per_thread + item
                    value = first[row, dim]
                    values[item] = value
                    value_f32 = cutlass.Float32(value)
                    local_absmax = cute.arch.fmax(
                        local_absmax,
                        cute.arch.fmax(value_f32, -value_f32),
                    )
            else:
                second_row = row - first.shape[0]
                for item in cutlass.range_constexpr(self.values_per_thread):
                    dim = block_base + lane * self.values_per_thread + item
                    value = second[second_row, dim]
                    values[item] = value
                    value_f32 = cutlass.Float32(value)
                    local_absmax = cute.arch.fmax(
                        local_absmax,
                        cute.arch.fmax(value_f32, -value_f32),
                    )

            block_absmax = cute.arch.warp_reduction_max(local_absmax)
            block_absmax = cute.arch.fmax(block_absmax, cutlass.Float32(1e-10))
            scale = (block_absmax * cutlass.Float32(1.0 / 448.0)).to(cutlass.Float8E8M0FNU)
            quant_scale = cute.arch.rcp_approx(cutlass.Float32(scale))
            for item in cutlass.range_constexpr(self.values_per_thread):
                dim = block_base + lane * self.values_per_thread + item
                output[row, dim] = (cutlass.Float32(values[item]) * quant_scale).to(
                    output.element_type
                )

            if lane == 0:
                scale_offset = (
                    (row // 128) * self.num_quant_blocks * 512
                    + quant_block * 512
                    + (row % 32) * 16
                    + ((row % 128) // 32) * 4
                )
                for replica in cutlass.range_constexpr(4):
                    output_scale[scale_offset + replica] = scale


class DSparkRMSNormRoPEPageWriteKernel(DSparkRMSNormRoPEKernel):
    """Apply RMSNorm/RoPE and write each valid row directly into its managed page."""

    def __init__(self, hidden_dim: int, rope_dim: int, eps: float, page_size: int):
        super().__init__(hidden_dim, rope_dim, 1, eps, True, True, False)
        self.page_size = page_size

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        freqs: cute.Tensor,
        kv_cache: cute.Tensor,
        block_tables: cute.Tensor,
        start_pos: cute.Tensor,
        capacities: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            x,
            weight,
            freqs,
            kv_cache,
            block_tables,
            start_pos,
            capacities,
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
        block_tables: cute.Tensor,
        start_pos: cute.Tensor,
        capacities: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        position = start_pos[row] - 1
        page = cutlass.Int32(-1)
        if position >= 0 and position < capacities[row]:
            logical_page = position // self.page_size
            if logical_page < block_tables.shape[1]:
                page = block_tables[row, logical_page]
        if page >= 0 and page < kv_cache.shape[0]:
            cache_pos = position % self.page_size

            sum_sq = cutlass.Float32(0.0)
            for item in cutlass.range_constexpr(self.elements_per_thread):
                dim = tidx + item * self.num_threads
                value = cutlass.Float32(x[row, dim])
                sum_sq += value * value
            sum_sq = cute.arch.warp_reduction_sum(sum_sq)
            inverse_rms = cute.math.rsqrt(sum_sq / self.hidden_dim + self.eps)

            for item in cutlass.range_constexpr(self.nope_elements_per_thread):
                dim = tidx + item * self.num_threads
                value = cutlass.Float32(x[row, dim]) * inverse_rms
                value *= cutlass.Float32(weight[dim])
                kv_cache[cutlass.Int64(page), cache_pos, dim] = value.to(kv_cache.element_type)

            for item in cutlass.range_constexpr(self.pairs_per_thread):
                pair = tidx + item * self.num_threads
                real_dim = self.nope_dim + pair * 2
                imag_dim = real_dim + 1
                real = cutlass.Float32(x[row, real_dim]) * inverse_rms
                imag = cutlass.Float32(x[row, imag_dim]) * inverse_rms
                real *= cutlass.Float32(weight[real_dim])
                imag *= cutlass.Float32(weight[imag_dim])
                cos = cutlass.Float32(freqs[row, pair, 0])
                sin = cutlass.Float32(freqs[row, pair, 1])
                kv_cache[cutlass.Int64(page), cache_pos, real_dim] = (real * cos - imag * sin).to(
                    kv_cache.element_type
                )
                kv_cache[cutlass.Int64(page), cache_pos, imag_dim] = (imag * cos + real * sin).to(
                    kv_cache.element_type
                )


class DSparkMlaCtxNormRoPEKernel:
    """MLA context latents of every drafter layer in one launch: RMSNorm + RoPE.

    Row ``r`` of the output is layer ``r // N``, token ``r % N``: the
    ``kv_lora_rank`` latent of ``x[token, layer]`` RMS-normalized and scaled by
    ``weight[layer]``, then the ``rope_dim`` tail rotated at
    ``freqs[positions[token]]`` (adjacent pairs, no norm, no weight).

    One warp per row, and per row the arithmetic of
    ``DSparkRMSNormRoPEKernel(kv_lora_rank + rope_dim, rope_dim, 1, eps, True,
    True, False, norm_dim=kv_lora_rank)``: lane ``l`` owns columns ``l + 32 i``,
    accumulates their squares in ``i`` order, the warp reduction and ``rsqrt``
    are the same calls, and each output is the same chain of FP32 products, so
    the results are bitwise that kernel's. Every load of a row is issued before
    its first store, and the RoPE table is indexed by position in the kernel.
    """

    num_warps = 4
    num_threads = num_warps * cute.arch.WARP_SIZE

    def __init__(self, num_layers: int, kv_lora_rank: int, rope_dim: int, eps: float):
        warp = cute.arch.WARP_SIZE
        if kv_lora_rank % warp != 0 or (rope_dim // 2) % warp != 0 or rope_dim % 2 != 0:
            raise ValueError(
                f"kv_lora_rank ({kv_lora_rank}) and rope_dim / 2 ({rope_dim // 2}) must be "
                f"multiples of {warp}"
            )
        self.num_layers = num_layers
        self.norm_dim = kv_lora_rank
        self.head_dim = kv_lora_rank + rope_dim
        self.eps = eps
        self.norm_items = kv_lora_rank // warp
        self.pair_items = rope_dim // 2 // warp

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        positions: cute.Tensor,
        freqs: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        rows = x.shape[0] * self.num_layers
        self.kernel(x, weight, positions, freqs, output).launch(
            grid=[cute.ceil_div(rows, self.num_warps), 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        positions: cute.Tensor,
        freqs: cute.Tensor,
        output: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % cute.arch.WARP_SIZE
        row = bidx * self.num_warps + tidx // cute.arch.WARP_SIZE
        num_tokens = x.shape[0]
        if row < num_tokens * self.num_layers:
            layer = row // num_tokens
            token = row - layer * num_tokens
            col0 = layer * self.head_dim
            values = cute.make_rmem_tensor((self.norm_items,), cutlass.Float32)
            scales = cute.make_rmem_tensor((self.norm_items,), cutlass.Float32)
            reals = cute.make_rmem_tensor((self.pair_items,), cutlass.Float32)
            imags = cute.make_rmem_tensor((self.pair_items,), cutlass.Float32)
            coss = cute.make_rmem_tensor((self.pair_items,), cutlass.Float32)
            sins = cute.make_rmem_tensor((self.pair_items,), cutlass.Float32)
            position = positions[token]
            src = x[token, None]
            _load_norm_items(src, col0, weight[layer, None], lane, values, scales, self.norm_items)
            _load_rope_pairs(
                src,
                col0 + self.norm_dim,
                freqs,
                position,
                lane,
                reals,
                imags,
                coss,
                sins,
                self.pair_items,
            )
            dst = output[row, None]
            _store_rms_normalized(
                dst, lane, values, scales, self.norm_dim, self.eps, self.norm_items
            )
            _store_rotated_pairs(
                dst, self.norm_dim, lane, reals, imags, coss, sins, self.pair_items
            )


class DSparkMlaBlockRoPEKernel:
    """One drafter layer's block RoPE work in one launch, all loads up front (the row helpers of
    :class:`DSparkMlaCtxNormRoPEKernel`).

    Rows ``[0, T * H)`` rotate the ``rope_dim`` tail of the 192-wide query row
    ``q[r]`` at ``freqs[positions[r // H]]`` into ``query[r, kv_lora_rank:]``
    (the rope half of the 576-wide absorbed query; the latent half is left to
    the absorption GEMM). Rows ``[T * H, T * H + T)`` RMS-normalize the latent of
    the block token ``t`` (``kv[t, :kv_lora_rank]``, scaled by ``weight``),
    rotate its ``k_pe`` at ``freqs[positions[t]]`` and store the 576-wide row
    straight into the paged pool at ``pool[pages[t], offsets[t]]``.

    Per row the arithmetic is that of ``cute_dsl_dspark_rope_into`` (query rows)
    and of ``cute_dsl_dspark_rmsnorm_rope(norm_dim=kv_lora_rank)`` followed by
    the pool scatter (latent rows), so the values written are bitwise theirs.
    """

    num_warps = 4
    num_threads = num_warps * cute.arch.WARP_SIZE

    def __init__(
        self, num_heads: int, qk_nope_dim: int, kv_lora_rank: int, rope_dim: int, eps: float
    ):
        warp = cute.arch.WARP_SIZE
        if kv_lora_rank % warp != 0 or (rope_dim // 2) % warp != 0 or rope_dim % 2 != 0:
            raise ValueError(
                f"kv_lora_rank ({kv_lora_rank}) and rope_dim / 2 ({rope_dim // 2}) must be "
                f"multiples of {warp}"
            )
        self.num_heads = num_heads
        self.qk_nope_dim = qk_nope_dim
        self.norm_dim = kv_lora_rank
        self.eps = eps
        self.norm_items = kv_lora_rank // warp
        self.pair_items = rope_dim // 2 // warp

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        query: cute.Tensor,
        kv: cute.Tensor,
        weight: cute.Tensor,
        positions: cute.Tensor,
        freqs: cute.Tensor,
        pool: cute.Tensor,
        pages: cute.Tensor,
        offsets: cute.Tensor,
        stream: cuda.CUstream,
    ):
        rows = q.shape[0] + kv.shape[0]
        self.kernel(q, query, kv, weight, positions, freqs, pool, pages, offsets).launch(
            grid=[cute.ceil_div(rows, self.num_warps), 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        query: cute.Tensor,
        kv: cute.Tensor,
        weight: cute.Tensor,
        positions: cute.Tensor,
        freqs: cute.Tensor,
        pool: cute.Tensor,
        pages: cute.Tensor,
        offsets: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % cute.arch.WARP_SIZE
        row = bidx * self.num_warps + tidx // cute.arch.WARP_SIZE
        num_q_rows = q.shape[0]
        reals = cute.make_rmem_tensor((self.pair_items,), cutlass.Float32)
        imags = cute.make_rmem_tensor((self.pair_items,), cutlass.Float32)
        coss = cute.make_rmem_tensor((self.pair_items,), cutlass.Float32)
        sins = cute.make_rmem_tensor((self.pair_items,), cutlass.Float32)
        if row < num_q_rows:
            position = positions[row // self.num_heads]
            _load_rope_pairs(
                q[row, None],
                self.qk_nope_dim,
                freqs,
                position,
                lane,
                reals,
                imags,
                coss,
                sins,
                self.pair_items,
            )
            _store_rotated_pairs(
                query[row, None], self.norm_dim, lane, reals, imags, coss, sins, self.pair_items
            )
        elif row < num_q_rows + kv.shape[0]:
            token = row - num_q_rows
            values = cute.make_rmem_tensor((self.norm_items,), cutlass.Float32)
            scales = cute.make_rmem_tensor((self.norm_items,), cutlass.Float32)
            position = positions[token]
            page = pages[token]
            offset = offsets[token]
            src = kv[token, None]
            _load_norm_items(src, 0, weight, lane, values, scales, self.norm_items)
            _load_rope_pairs(
                src,
                self.norm_dim,
                freqs,
                position,
                lane,
                reals,
                imags,
                coss,
                sins,
                self.pair_items,
            )
            dst = pool[page, offset, None]
            _store_rms_normalized(
                dst, lane, values, scales, self.norm_dim, self.eps, self.norm_items
            )
            _store_rotated_pairs(
                dst, self.norm_dim, lane, reals, imags, coss, sins, self.pair_items
            )

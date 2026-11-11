# ruff: noqa: ANN001, ANN201, ERA001, N803, N806
try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except (ImportError, OSError):
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def fused_add_round_kernel(
        x_ptr,
        output_ptr,  # contents will be added to the output
        seed,
        n_elements,
        EXPONENT_BIAS,
        MANTISSA_BITS,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        A kernel to upcast 8bit quantized weights to bfloat16 with stochastic rounding
        and add them to bfloat16 output weights. Might be used to upcast original model weights
        and to further add them to precalculated deltas coming from LoRAs.
        """
        # Get program ID and compute offsets
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load data
        x = tl.load(x_ptr + offsets, mask=mask)
        rand_vals = tl.rand(seed, offsets) - 0.5

        x = tl.cast(x, tl.float16)
        delta = tl.load(output_ptr + offsets, mask=mask)
        delta = tl.cast(delta, tl.float16)
        x = x + delta

        x_bits = tl.cast(x, tl.int16, bitcast=True)

        # Calculate the exponent. Unbiased fp16 exponent is ((x_bits & 0x7C00) >> 10) - 15 for
        # normal numbers and -14 for subnormals.
        fp16_exponent_bits = (x_bits & 0x7C00) >> 10
        fp16_normals = fp16_exponent_bits > 0
        fp16_exponent = tl.where(fp16_normals, fp16_exponent_bits - 15, -14)

        # Add the target dtype's exponent bias and clamp to the target dtype's exponent range.
        exponent = fp16_exponent + EXPONENT_BIAS
        MAX_EXPONENT = 2 * EXPONENT_BIAS + 1
        exponent = tl.where(exponent > MAX_EXPONENT, MAX_EXPONENT, exponent)
        exponent = tl.where(exponent < 0, 0, exponent)

        # Normal ULP exponent, expressed as an fp16 exponent field:
        # (exponent - EXPONENT_BIAS - MANTISSA_BITS) + 15
        # Simplifies to: fp16_exponent - MANTISSA_BITS + 15
        # See https://en.wikipedia.org/wiki/Unit_in_the_last_place
        eps_exp = tl.maximum(0, tl.minimum(31, exponent - EXPONENT_BIAS - MANTISSA_BITS + 15))

        # Calculate epsilon in the target dtype
        eps_normal = tl.cast(tl.cast(eps_exp << 10, tl.int16), tl.float16, bitcast=True)

        # Subnormal ULP: 2^(1 - EXPONENT_BIAS - MANTISSA_BITS) ->
        # fp16 exponent bits: (1 - EXPONENT_BIAS - MANTISSA_BITS) + 15 =
        # 16 - EXPONENT_BIAS - MANTISSA_BITS
        eps_subnormal = tl.cast(tl.cast((16 - EXPONENT_BIAS - MANTISSA_BITS) << 10, tl.int16), tl.float16, bitcast=True)
        eps = tl.where(exponent > 0, eps_normal, eps_subnormal)

        # Apply zero mask to epsilon
        eps = tl.where(x == 0, 0.0, eps)

        # Apply stochastic rounding
        output = tl.cast(x + rand_vals * eps, tl.bfloat16)

        # Store the result
        tl.store(output_ptr + offsets, output, mask=mask)

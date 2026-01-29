"""
Fused split and transpose kernel for mamba2_mixer optimization.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _extract_transpose_prefill_kernel(
    src_ptr,
    dst_ptr,
    num_prefill_tokens,
    d_in_proj,
    d_inner,
    conv_dim,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_CONV: tl.constexpr,
):
    pid_seq = tl.program_id(0)
    pid_conv = tl.program_id(1)

    seq_start = pid_seq * BLOCK_SEQ
    conv_start = pid_conv * BLOCK_CONV

    seq_offsets = seq_start + tl.arange(0, BLOCK_SEQ)
    conv_offsets = conv_start + tl.arange(0, BLOCK_CONV)

    seq_mask = seq_offsets < num_prefill_tokens
    conv_mask = conv_offsets < conv_dim

    src_offsets = seq_offsets[:, None] * d_in_proj + (d_inner + conv_offsets[None, :])
    mask = seq_mask[:, None] & conv_mask[None, :]

    data = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)

    dst_offsets = conv_offsets[:, None] * num_prefill_tokens + seq_offsets[None, :]
    tl.store(dst_ptr + dst_offsets, tl.trans(data), mask=conv_mask[:, None] & seq_mask[None, :])


def extract_transpose_xbc_prefill(
    zxbcdt: torch.Tensor,
    num_prefill_tokens: int,
    d_inner: int,
    conv_dim: int,
) -> torch.Tensor:
    out = torch.empty(conv_dim, num_prefill_tokens, dtype=zxbcdt.dtype, device=zxbcdt.device)
    d_in_proj = zxbcdt.shape[1]

    BLOCK_SEQ = 64
    BLOCK_CONV = 64

    grid = (triton.cdiv(num_prefill_tokens, BLOCK_SEQ), triton.cdiv(conv_dim, BLOCK_CONV))

    _extract_transpose_prefill_kernel[grid](
        zxbcdt,
        out,
        num_prefill_tokens,
        d_in_proj,
        d_inner,
        conv_dim,
        BLOCK_SEQ,
        BLOCK_CONV,
    )

    return out


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SEQ": 32, "BLOCK_CONV": 32}),
        triton.Config({"BLOCK_SEQ": 64, "BLOCK_CONV": 32}),
        triton.Config({"BLOCK_SEQ": 32, "BLOCK_CONV": 64}),
        triton.Config({"BLOCK_SEQ": 64, "BLOCK_CONV": 64}),
        triton.Config({"BLOCK_SEQ": 128, "BLOCK_CONV": 32}),
        triton.Config({"BLOCK_SEQ": 32, "BLOCK_CONV": 128}),
        triton.Config({"BLOCK_SEQ": 128, "BLOCK_CONV": 64}),
        triton.Config({"BLOCK_SEQ": 64, "BLOCK_CONV": 128}),
    ],
    key=["num_prefill_tokens", "conv_dim"],
)
@triton.jit
def _extract_transpose_prefill_kernel_autotuned(
    src_ptr,
    dst_ptr,
    num_prefill_tokens,
    d_in_proj,
    d_inner,
    conv_dim,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_CONV: tl.constexpr,
):
    pid_seq = tl.program_id(0)
    pid_conv = tl.program_id(1)

    seq_start = pid_seq * BLOCK_SEQ
    conv_start = pid_conv * BLOCK_CONV

    seq_offsets = seq_start + tl.arange(0, BLOCK_SEQ)
    conv_offsets = conv_start + tl.arange(0, BLOCK_CONV)

    seq_mask = seq_offsets < num_prefill_tokens
    conv_mask = conv_offsets < conv_dim

    src_offsets = seq_offsets[:, None] * d_in_proj + (d_inner + conv_offsets[None, :])
    mask = seq_mask[:, None] & conv_mask[None, :]

    data = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)

    dst_offsets = conv_offsets[:, None] * num_prefill_tokens + seq_offsets[None, :]
    tl.store(dst_ptr + dst_offsets, tl.trans(data), mask=conv_mask[:, None] & seq_mask[None, :])


def extract_transpose_xbc_prefill_autotuned(
    zxbcdt: torch.Tensor,
    num_prefill_tokens: int,
    d_inner: int,
    conv_dim: int,
) -> torch.Tensor:
    """
    Autotuned version of extract_transpose_xbc_prefill that dynamically selects optimal block sizes.
    """
    out = torch.empty(conv_dim, num_prefill_tokens, dtype=zxbcdt.dtype, device=zxbcdt.device)
    d_in_proj = zxbcdt.shape[1]

    def grid(meta):
        return (
            triton.cdiv(num_prefill_tokens, meta["BLOCK_SEQ"]),
            triton.cdiv(conv_dim, meta["BLOCK_CONV"]),
        )

    _extract_transpose_prefill_kernel_autotuned[grid](
        zxbcdt,
        out,
        num_prefill_tokens,
        d_in_proj,
        d_inner,
        conv_dim,
    )

    return out


def extract_transpose_xbc_prefill_smart(
    zxbcdt: torch.Tensor,
    num_prefill_tokens: int,
    d_inner: int,
    conv_dim: int,
    use_autotune: bool = True,
    autotune_threshold: int = 1024,
) -> torch.Tensor:
    """
    Smart wrapper that chooses between fixed-size and autotuned kernels.

    Args:
        zxbcdt: Input tensor
        num_prefill_tokens: Number of prefill tokens
        d_inner: Inner dimension offset
        conv_dim: Convolution dimension size
        use_autotune: Whether to use autotuning when appropriate
        autotune_threshold: Minimum size (tokens * conv_dim) to trigger autotuning

    Returns:
        Transposed tensor [conv_dim, num_prefill_tokens]
    """
    # For small inputs, fixed block sizes are often more efficient
    # For large inputs, autotuning can find better configurations
    if use_autotune and num_prefill_tokens * conv_dim >= autotune_threshold:
        return extract_transpose_xbc_prefill_autotuned(
            zxbcdt, num_prefill_tokens, d_inner, conv_dim
        )
    else:
        return extract_transpose_xbc_prefill(zxbcdt, num_prefill_tokens, d_inner, conv_dim)

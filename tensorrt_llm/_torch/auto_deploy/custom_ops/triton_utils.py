"""Triton utility operations for auto_deploy."""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_gather_scatter_kernel(
    ungathered_ptr,  # *T
    gather_ids_ptr,  # *int64
    mask_indices_ptr,  # *int64
    out_ptr,  # *T
    n_elements,  # int32
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for fused gather and scatter operation.

    This kernel gathers values from `ungathered_ptr` using indices from `gather_ids_ptr`
    and scatters them to `out_ptr` at positions specified by `mask_indices_ptr`.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # load source indices
    src_idx = tl.load(gather_ids_ptr + offs, mask=mask, other=0)
    # load values from ungathered
    vals = tl.load(ungathered_ptr + src_idx, mask=mask, other=0)

    # load destination indices (into flattened output)
    dst_idx = tl.load(mask_indices_ptr + offs, mask=mask, other=0)

    # scatter values to output
    tl.store(out_ptr + dst_idx, vals, mask=mask)


@torch.library.custom_op("auto_deploy::triton_utils_fused_gather_scatter", mutates_args=("out",))
def fused_gather_scatter(
    ungathered_input: torch.Tensor,
    gather_ids: torch.Tensor,
    mask_indices: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Fused gather and scatter operation using Triton.

    This operation gathers values from `ungathered_input` at indices specified by
    `gather_ids` and scatters the gathered values to `out` at positions specified
    by `mask_indices`.

    This is useful for efficiently rearranging input_ids in overlap scheduling
    scenarios where tokens need to be reordered based on scheduling decisions.

    Args:
        ungathered_input: Source tensor from which to gather values.
        gather_ids: Indices into `ungathered_input` specifying which values to gather.
        mask_indices: Destination indices in `out` where gathered values should be scattered.
        out: Output tensor where gathered values will be scattered.

    Note:
        This operation mutates `out` in-place.
    """
    n = gather_ids.numel()

    BLOCK_SIZE = 256
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _fused_gather_scatter_kernel[grid](
        ungathered_input,  # ungathered_ptr
        gather_ids,  # gather_ids_ptr
        mask_indices,  # mask_indices_ptr
        out,  # out_ptr
        n,  # n_elements
        BLOCK_SIZE=BLOCK_SIZE,
    )


@fused_gather_scatter.register_fake
def fused_gather_scatter_fake(
    ungathered_input: torch.Tensor,
    gather_ids: torch.Tensor,
    mask_indices: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Fake implementation for torch.compile / graph tracing."""
    pass

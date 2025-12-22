import torch


@torch.library.custom_op("trtllm::copy_to_nccl_window", mutates_args=())
def copy_to_nccl_window(a: torch.Tensor, group: list[int]) -> torch.Tensor:
    """Copy tensor to NCCL window buffer.

    Args:
        a: Input tensor to copy
        group: List of ranks in the tensor parallel group

    Returns:
        Tensor backed by NCCL window buffer
    """
    nccl_tensor = torch.ops.trtllm.create_nccl_window_tensor(group, a.shape, a.dtype)
    return nccl_tensor.copy_(a, non_blocking=True)


@copy_to_nccl_window.register_fake
def _(a, group) -> torch.Tensor:
    return torch.empty_like(a)


# Custom ops below are used to avoid in-place operation added to torch fx graph.
# According to test, exporting in-place operation leads to additional copy
# and may block other ops using the in-place op from being optimized.


@torch.library.custom_op("trtllm::add_to_nccl_window", mutates_args=())
def add_to_nccl_window(a: torch.Tensor, b: torch.Tensor, group: list[int]) -> torch.Tensor:
    """Add two tensors, allocating result into NCCL window buffer.

    Args:
        a: First tensor
        b: Second tensor
        group: List of ranks in the tensor parallel group

    Returns:
        Tensor backed by NCCL window buffer containing a + b
    """
    shape = [max(i, j) for i, j in zip(a.shape, b.shape)]
    nccl_tensor = torch.ops.trtllm.create_nccl_window_tensor(group, shape, a.dtype)
    return torch.add(a, b, out=nccl_tensor)


@add_to_nccl_window.register_fake
def _(a, b, group) -> torch.Tensor:
    shape = [max(i, j) for i, j in zip(a.shape, b.shape)]
    return a.new_empty(shape, dtype=a.dtype)


@torch.library.custom_op("trtllm::matmul_to_nccl_window", mutates_args=())
def matmul_to_nccl_window(a: torch.Tensor, b: torch.Tensor, group: list[int]) -> torch.Tensor:
    """Matmul two tensors, allocating result into NCCL window buffer.

    Args:
        a: First tensor of shape (..., m, k)
        b: Second tensor of shape (..., k, n)
        group: List of ranks in the tensor parallel group

    Returns:
        Tensor backed by NCCL window buffer containing a @ b
    """
    shape = list(a.shape)
    shape[-1] = b.shape[-1]
    nccl_tensor = torch.ops.trtllm.create_nccl_window_tensor(group, shape, a.dtype)
    return torch.matmul(a, b, out=nccl_tensor)


@matmul_to_nccl_window.register_fake
def _(a, b, group) -> torch.Tensor:
    shape = list(a.shape)
    shape[-1] = b.shape[-1]
    return a.new_empty(shape, dtype=a.dtype)

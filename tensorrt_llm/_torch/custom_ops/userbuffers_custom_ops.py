import torch


@torch.library.custom_op("trtllm::copy_to_userbuffers", mutates_args=())
def copy_to_userbuffers(a: torch.Tensor) -> torch.Tensor:
    ub_tensor = torch.ops.trtllm.create_userbuffers_tensor(a.shape, a.dtype)
    return ub_tensor.copy_(a, non_blocking=True)


@copy_to_userbuffers.register_fake
def _(a) -> torch.Tensor:
    return torch.empty_like(a)


# Custom ops below are used to avoid in-place operation added to torch fx graph.
# According to test, exporting in-place operation leads to additional copy and may block ops using the in-place op from being optimized.


@torch.library.custom_op("trtllm::add_to_ub", mutates_args=())
def add_to_ub(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    shape = [max(i, j) for i, j in zip(a.shape, b.shape)]
    ub_tensor = torch.ops.trtllm.create_userbuffers_tensor(shape, a.dtype)
    return torch.add(a, b, out=ub_tensor)


@add_to_ub.register_fake
def _(a, b) -> torch.Tensor:
    shape = [max(i, j) for i, j in zip(a.shape, b.shape)]
    return a.new_empty(shape, dtype=a.dtype)


@torch.library.custom_op("trtllm::matmul_to_ub", mutates_args=())
def matmul_to_ub(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    shape = list(a.shape)
    shape[-1] = b.shape[-1]
    ub_tensor = torch.ops.trtllm.create_userbuffers_tensor(shape, a.dtype)
    return torch.matmul(a, b, out=ub_tensor)


@matmul_to_ub.register_fake
def _(a, b) -> torch.Tensor:
    shape = list(a.shape)
    shape[-1] = b.shape[-1]
    return a.new_empty(shape, dtype=a.dtype)

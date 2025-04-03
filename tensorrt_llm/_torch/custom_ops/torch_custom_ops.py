import torch


# Used to WAR an issue in torch.bmm that it would break the graph when the out is not contiguous.
@torch.library.custom_op("trtllm::bmm_out", mutates_args=("out", ))
def bmm_out(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    torch.bmm(a, b, out=out)


# Used to WAR an issue that torch.add not allowed to use out in target pattern
@torch.library.custom_op("trtllm::add_out", mutates_args=("out", ))
def add_out(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    torch.add(a, b, out=out)

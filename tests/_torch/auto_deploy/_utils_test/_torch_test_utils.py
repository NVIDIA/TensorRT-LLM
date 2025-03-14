from typing import Tuple, Union

import torch


def all_close(
    t1: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    t2: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    atol: float = 1e-3,
    rtol: float = 1e-3,
):
    if isinstance(t1, torch.Tensor):
        t1 = (t1,)
    if isinstance(t2, torch.Tensor):
        t2 = (t2,)
    return all(torch.allclose(a, b, atol=atol, rtol=rtol) for a, b in zip(t1, t2))


def reset_parameters(model: torch.nn.Module):
    for p in model.parameters():
        p.data = torch.randn_like(p.data, dtype=torch.float16).to(p)


def fp8_compatible():
    return torch.cuda.get_device_capability(0) >= (8, 9)


def fp4_compatible():
    return torch.cuda.get_device_capability(0) >= (10, 0)


def trtllm_ops_available():
    return hasattr(torch.ops, "trtllm")

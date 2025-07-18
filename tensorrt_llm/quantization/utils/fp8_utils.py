from typing import Tuple

import torch

from tensorrt_llm._utils import nvtx_range


def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
        x: the dividend.
        y: the divisor.

    Returns:
        The result of the ceiling division.
    """
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


@nvtx_range("[DG] quantization")
def per_token_cast_to_fp8_e8m0(
        x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.dim() == 2:
        assert x.size(1) % 128 == 0
        m, n = x.shape
        x_view = x.view(m, -1, 128)
        x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
        sf = ceil_to_ue8m0(x_amax / 448.0)
        return (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(
            m, n), sf
    else:
        assert x.size(2) % 128 == 0
        g, m, n = x.shape
        x_view = x.view(g, m, -1, 128)
        x_amax = x_view.abs().float().amax(dim=3).view(g, m, -1).clamp(1e-4)
        sf = ceil_to_ue8m0(x_amax / 448.0)
        return (x_view * (1.0 / sf.unsqueeze(3))).to(torch.float8_e4m3fn).view(
            g, m, n), sf


def per_block_cast_to_fp8_e8m0(
        x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.dim() == 2:
        m, n = x.shape
        x_padded = torch.zeros((align(m, 128), align(n, 128)),
                               dtype=x.dtype,
                               device=x.device)
        x_padded[:m, :n] = x
        x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
        x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
        sf = ceil_to_ue8m0(x_amax / 448.0)
        x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
        return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(
            x_view.size(0), x_view.size(2))
    else:
        g, m, n = x.shape
        x_padded = torch.zeros((g, align(m, 128), align(n, 128)),
                               dtype=x.dtype,
                               device=x.device)
        x_padded[:, :m, :n] = x
        x_view = x_padded.view(g, -1, 128, x_padded.size(-1) // 128, 128)
        x_amax = x_view.abs().float().amax(dim=(2, 4), keepdim=True).clamp(1e-4)
        sf = ceil_to_ue8m0(x_amax / 448.0)
        x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
        return x_scaled.view_as(x_padded)[:, :m, :n].contiguous(), sf.view(
            x_view.size(0), x_view.size(1), x_view.size(3))


def resmooth_to_fp8_e8m0(weight: torch.Tensor,
                         sf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if weight.dim() == 2:
        x = weight.float() * sf.repeat_interleave(128, dim=0).repeat_interleave(
            128, dim=1)[:weight.shape[0], :weight.shape[1]]
    else:
        x = weight.float() * sf.repeat_interleave(128, dim=1).repeat_interleave(
            128, dim=2)[:weight.shape[0], :weight.shape[1], :weight.shape[2]]
    return per_block_cast_to_fp8_e8m0(x)

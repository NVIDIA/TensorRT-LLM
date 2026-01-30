"""Custom ops corresponding to l2norm."""

import torch

from tensorrt_llm._torch.modules.fla.l2norm import l2norm_fwd


@torch.library.custom_op("auto_deploy::torch_l2norm", mutates_args=())
def _torch_l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x: (..., D)
    returns:
      y: (..., D)  # normalized
      rstd: (...,) # 1/sqrt(sum(x^2)+eps) along `dim`
    """
    x_f32 = x.float()
    s = (x_f32 * x_f32).sum(dim=-1, keepdim=True)  # (..., 1)
    rstd = torch.rsqrt(s + eps)  # (..., 1)
    y = (x_f32 * rstd).to(x.dtype)  # cast back
    return y


@_torch_l2norm.register_fake
def _torch_l2norm_fake(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("auto_deploy::fla_l2norm", mutates_args=())
def fla_l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y = l2norm_fwd(x, eps)
    return y


@fla_l2norm.register_fake
def fla_l2norm_fake(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.empty_like(x)

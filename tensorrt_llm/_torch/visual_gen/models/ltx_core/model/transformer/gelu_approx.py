import torch


class GELUApprox(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.proj(x), approximate="tanh")

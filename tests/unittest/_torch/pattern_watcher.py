import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import (PatternPrettyPrinter, fwd_only,
                                             gen_pattern)

import tensorrt_llm  #isort:skip


@torch.library.custom_op("trtllm::custom_add", mutates_args=("a", ))
def custom_add(a: torch.Tensor, b: torch.Tensor) -> None:
    a.add_(b)


def source_pattern(x: torch.Tensor, residual: torch.Tensor,
                   weight: torch.Tensor, eps: float):
    at = auto_functionalized(
        torch.ops.trtllm.flashinfer_fused_add_rmsnorm.default,
        input=x,
        residual=residual,
        weight=weight,
        eps=eps)
    return at[1], at[2]


p = PatternPrettyPrinter()

x = torch.empty((5, 3)).cuda().half()
res = x.clone()
weight = torch.empty((3, )).cuda().half()
eps = 1e-5

pattern = gen_pattern(source_pattern, [x, res, weight, eps], fwd_only)

print(PatternPrettyPrinter.run(pattern))

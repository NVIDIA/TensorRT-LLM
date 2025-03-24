import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import (PatternPrettyPrinter, fwd_only,
                                             gen_pattern)

import tensorrt_llm
import tensorrt_llm._torch
import tensorrt_llm._torch.modules
import tensorrt_llm._torch.modules.rms_norm

norm = tensorrt_llm._torch.modules.rms_norm.RMSNorm(hidden_size=3,
                                                    eps=1e-5,
                                                    dtype=torch.float16).cuda()


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

x = torch.empty((1, 3)).cuda().half()
res = x.clone()
weight = torch.empty((3, )).cuda().half()
eps = 1e-5

pattern = gen_pattern(source_pattern, [x, res, weight, eps], fwd_only)

print(PatternPrettyPrinter.run(pattern))

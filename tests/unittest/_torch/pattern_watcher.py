import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import (PatternPrettyPrinter, fwd_only,
                                             gen_pattern)
from torch.fx import GraphModule

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
    output_shape = (x.shape[0], 2 * 3)
    output = x.new_empty(output_shape, dtype=x.dtype)
    return at[1], at[2], output * output


p = PatternPrettyPrinter()

x = torch.empty((5, 3)).cuda().half()
res = x.clone()
weight = torch.empty((3, )).cuda().half()
eps = 1e-5

pattern = gen_pattern(source_pattern, [x, res, weight, eps], fwd_only)

print(PatternPrettyPrinter.run(pattern))

torch._dynamo.mark_dynamic(x, 0)


def print_aten(gm: GraphModule, _):
    print("asdas", type(gm))
    gm.graph.print_tabular()
    return gm


func = torch.compile(source_pattern, backend=print_aten)

# func = aot_function(source_pattern, fw_compiler=print_aten)
func(x, res, weight, eps)

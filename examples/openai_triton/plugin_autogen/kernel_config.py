import os

import torch

from tensorrt_llm.tools.plugin_gen.core import *

openai_triton_example_root = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "manual_plugin")


def get_fmha_kernel_meta_data():
    block_size = 128
    num_stages = 2 if torch.cuda.get_device_capability() >= (8, 0) else 1

    return KernelMetaData(
        kernel_name='fused_attention_kernel',
        ios=[
            # outputs
            OutputArg('Out', Type('tensor[fp16]'), hints=['16', '16']),
            OutputArg('L', Type('tensor[fp32]'), hints=['16', '16']),
            OutputArg('M', Type('tensor[fp32]'), hints=['16', '16']),
            # inputs
            InputArg('Q', Type('tensor[fp16]'), hints=['16', '16']),
            InputArg('K', Type('tensor[fp16]'), hints=['16', '16']),
            InputArg('V', Type('tensor[fp16]'), hints=['16', '16']),
            ParamArg('sm_scale', Type('fp32')),
            DimSizeArg('batch_size'),
            ParamArg('num_heads', Type('i32')),
            DimSizeArg('seq_len', hints=['', '16']),
            # constexprs
            Constexpr(block_size),
            Constexpr(64),
            Constexpr(block_size),
        ],
        shape_infer_rules=[
            # The following rules helps to deduce the shapes of the output tensors
            "Q[*] -> Out[*]",
            "Q[m,n,k,*] -> L[m,n,k]",
            "Q[m,n,k,*] -> M[m,n,k]",

            # The following rules helps to deduce both DimSizeArgs: batch_size and seq_len
            "Q[m,n,k,*] : m -> batch_size",
            "Q[m,n,k,*] : k -> seq_len",
        ],
        version=0,
        kernel_file=f'{openai_triton_example_root}/fmha_triton.py',
        num_warps=4,
        num_stages=num_stages,
        grid_dims=(f"(seq_len + {block_size-1}) / {block_size}",
                   "batch_size * num_heads", "1"))


KERNELS = [
    get_fmha_kernel_meta_data(),
]

from typing import Sequence

import triton
import triton.language as tl

from tensorrt_llm import PluginBase
from tensorrt_llm._utils import TensorWrapper
from tensorrt_llm.python_plugin import SymTensor, trtllm_plugin


@trtllm_plugin("DummyPlugin")
class DummyPlugin(PluginBase):

    def __init__(self):
        super().__init__()

    def shape_dtype_inference(self, inputs: Sequence[SymTensor]) -> SymTensor:
        return inputs[0]

    def forward(self, inputs: Sequence[TensorWrapper],
                outputs: Sequence[TensorWrapper]):
        pass


@triton.jit
def lookup_kernel(X, Y, Z, vocab_size, hidden_size, token_num):
    pid = tl.program_id(axis=0)
    while pid < token_num * hidden_size:
        row_idx = pid // hidden_size
        col_idx = pid % hidden_size
        word_idx = tl.load(X + row_idx)
        embedding = tl.load(
            Y + word_idx * hidden_size + col_idx,
            mask=word_idx < vocab_size,
        )
        tl.store(Z + pid, embedding)
        pid += tl.num_programs(0)

# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence

import torch

from tensorrt_llm import PluginBase
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.python_plugin import SymTensor, trtllm_plugin

from .lookup_kernel import lookup_kernel


@trtllm_plugin("TritonLookUp")
class LookUpPlugin(PluginBase):

    def __init__(self, use_torch_tensor, fp32_output):
        super().__init__()
        self.use_torch_tensor = use_torch_tensor
        self.fp32_output = fp32_output

    def shape_dtype_inference(self, inputs: Sequence[SymTensor]) -> SymTensor:
        shape = inputs[1].shape
        shape[0] = inputs[0].shape[0] + inputs[1].shape[0] - inputs[1].shape[0]
        return SymTensor(
            inputs[1].dtype if not self.fp32_output else torch.float32, shape)

    def forward(self, inputs: Sequence[TensorWrapper],
                outputs: Sequence[TensorWrapper]):
        assert len(inputs) == 2
        assert inputs[0].dtype in [torch.int32 or torch.int64]
        assert inputs[1].dtype in [torch.float32, torch.float16, torch.bfloat16]
        assert (self.fp32_output and outputs[0].dtype
                == torch.float32) or outputs[0].dtype == inputs[1].dtype

        x = inputs[0]
        y = inputs[1]
        z = outputs[0]
        if self.use_torch_tensor:
            x = convert_to_torch_tensor(x)
            y = convert_to_torch_tensor(y)
            z = convert_to_torch_tensor(z)
        MAX_BLOCK_NUM = 65536
        MAX_BLOCK_SIZE = 512
        grid = lambda meta: (min(MAX_BLOCK_NUM, x.shape[0]) * min(
            MAX_BLOCK_SIZE, y.shape[1]), )
        lookup_kernel[grid](x, y, z, y.shape[0], y.shape[1], x.shape[0])

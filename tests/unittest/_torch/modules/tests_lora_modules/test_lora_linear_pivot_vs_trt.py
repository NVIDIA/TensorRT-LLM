import os
import sys
import unittest

import numpy as np
import tensorrt as trt
import torch

import tensorrt_llm
from tensorrt_llm._torch.modules.linear import Linear as PivotLinear
from tensorrt_llm._torch.peft.lora.layer import LoraModuleType
from tensorrt_llm._utils import torch_to_numpy
from tensorrt_llm.functional import Tensor
from tensorrt_llm.layers.linear import Linear
from tensorrt_llm.layers.lora import Lora, LoraRuntimeParams
from tensorrt_llm.runtime.session import Session

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
from utils.util import create_session, run_session


class TestLoraLinearPivotVsTRT(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dtype = "float16"
        cls.batch_size = 4
        cls.seq_len = 8
        cls.hidden_size = 1024
        cls.lora_rank = 8

        # Create input tensors
        cls.input_tensor = torch.randn(cls.batch_size,
                                       cls.seq_len,
                                       cls.hidden_size,
                                       dtype=tensorrt_llm.str_dtype_to_torch(
                                           cls.dtype),
                                       device="cuda")

        cls.weight = torch.randn(cls.hidden_size,
                                 cls.hidden_size,
                                 dtype=tensorrt_llm.str_dtype_to_torch(
                                     cls.dtype),
                                 device="cuda")

        cls.A = torch.randn(cls.lora_rank,
                            cls.hidden_size,
                            dtype=tensorrt_llm.str_dtype_to_torch(cls.dtype),
                            device="cuda")
        cls.B = torch.randn(cls.hidden_size,
                            cls.lora_rank,
                            dtype=tensorrt_llm.str_dtype_to_torch(cls.dtype),
                            device="cuda")

    def _create_linear_lora_trt_session(self) -> Session:
        builder = tensorrt_llm.Builder()
        network = builder.create_network()

        with tensorrt_llm.net_guard(network):
            network.plugin_config.lora_plugin = self.dtype
            network.plugin_config.remove_input_padding = False

            linear = Linear(in_features=self.hidden_size,
                            out_features=self.hidden_size,
                            dtype=self.dtype,
                            bias=False)
            linear.lora = Lora(in_hidden_size=self.hidden_size,
                               out_hidden_sizes=[self.hidden_size],
                               max_low_rank=self.lora_rank)

            linear.weight.value = np.ascontiguousarray(
                torch_to_numpy(self.weight.cpu()))

            inp = Tensor(
                name="input_tensor",
                shape=[self.batch_size, self.seq_len, self.hidden_size],
                dtype=tensorrt_llm.str_dtype_to_trt(self.dtype))

            lora_weights_pointers = Tensor(name="lora_weights_pointers",
                                           shape=[self.batch_size, 3],
                                           dtype=trt.int64)

            host_request_types = Tensor(name="host_request_types",
                                        shape=[self.batch_size],
                                        dtype=trt.int32)

            lora_ranks = Tensor(name="lora_ranks",
                                shape=(self.batch_size, ),
                                dtype=trt.int32)

            lora_params = LoraRuntimeParams(
                lora_ranks=[lora_ranks],
                lora_weights_pointers=[lora_weights_pointers],
                host_request_types=host_request_types,
                weight_index=0)

            output = linear(inp, lora_runtime_params=lora_params)
            output.mark_output("output", self.dtype)

        return create_session(builder, network, precision=self.dtype)

    def _create_trt_inputs(self):
        host_request_types = torch.zeros(self.batch_size, dtype=torch.int32)
        magnitude_dora = torch.zeros(self.hidden_size,
                                     dtype=tensorrt_llm.str_dtype_to_torch(
                                         self.dtype),
                                     device="cuda")

        inputs = {
            "input_tensor": self.input_tensor,
            "host_request_types": host_request_types
        }

        # Create LoRA weight pointers
        weights_ptrs = torch.tensor(
            [[self.A.data_ptr(),
              self.B.data_ptr(),
              magnitude_dora.data_ptr()] for _ in range(self.batch_size)],
            dtype=torch.int64)
        inputs["lora_weights_pointers"] = weights_ptrs
        inputs["lora_ranks"] = torch.tensor([self.lora_rank] * self.batch_size,
                                            dtype=torch.int32)

        return inputs

    def _setup_pivot_linear(self):
        pivot_linear = PivotLinear(in_features=self.hidden_size,
                                   out_features=self.hidden_size,
                                   bias=False,
                                   dtype=tensorrt_llm.str_dtype_to_torch(
                                       self.dtype),
                                   layer_idx=0)

        pivot_linear.weight.data = self.weight
        return pivot_linear

    def test_lora_linear_layer(self):
        session = self._create_linear_lora_trt_session()

        inputs = self._create_trt_inputs()
        outputs = run_session(session, inputs)
        torch.cuda.synchronize()

        pivot_linear = self._setup_pivot_linear()

        lora_params = {
            'num_seqs':
            self.batch_size,
            'host_request_types':
            inputs["host_request_types"],
            'prompt_lens_cpu':
            torch.tensor([self.seq_len] * self.batch_size, dtype=torch.int32),
            0: {  # layer_idx
                LoraModuleType.DENSE: {  # module_type
                    'adapter_size': inputs["lora_ranks"],
                    'weight_pointers': inputs["lora_weights_pointers"],
                    'is_dora': False,
                }
            }
        }

        outputs_pivot = pivot_linear(self.input_tensor, lora_params=lora_params)

        print(f"outputs: {outputs['output']}")
        print(f"outputs_pivot: {outputs_pivot}")

        torch.testing.assert_close(outputs["output"],
                                   outputs_pivot,
                                   atol=2e-3,
                                   rtol=0)


if __name__ == "__main__":
    unittest.main()
    # x = 0
    # for i in range(100):
    #     x += 1
    # print(x)

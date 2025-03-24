import os
import sys

import numpy as np
import tensorrt as trt
import torch

import tensorrt_llm
from tensorrt_llm._torch.modules.linear import Linear as PivotLinear
from tensorrt_llm._utils import torch_to_numpy
from tensorrt_llm.functional import Tensor
from tensorrt_llm.layers.linear import Linear
from tensorrt_llm.layers.lora import Lora, LoraRuntimeParams
from tensorrt_llm.runtime.session import Session

# Add the unittest directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from utils.util import create_session, run_session


class TestLoraLinearPivotVsTrt:
    """Test class for comparing LoRA linear layer implementations."""

    def setup_method(self, method):
        """Set up test parameters and resources."""
        # Test parameters
        self.dtype = "float16"
        self.batch_size = 16
        self.seq_len = 32
        self.hidden_size = 1024
        self.lora_rank = 8

        # Create input tensors
        self.activations = torch.randn(self.batch_size,
                                       self.seq_len,
                                       self.hidden_size,
                                       dtype=tensorrt_llm.str_dtype_to_torch(
                                           self.dtype),
                                       device="cuda")

        # Create weights
        self.weight = torch.randn(self.hidden_size,
                                  self.hidden_size,
                                  dtype=tensorrt_llm.str_dtype_to_torch(
                                      self.dtype),
                                  device="cuda")

        # Create LoRA weights
        self.A = torch.randn(self.lora_rank,
                             self.hidden_size,
                             dtype=tensorrt_llm.str_dtype_to_torch(self.dtype),
                             device="cuda")
        self.B = torch.randn(self.hidden_size,
                             self.lora_rank,
                             dtype=tensorrt_llm.str_dtype_to_torch(self.dtype),
                             device="cuda")

    def _create_linear_lora_trt_session(self) -> Session:
        """Create a TensorRT session for the LoRA linear layer."""
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

            # Define input tensors
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

            # Create LoRA runtime parameters
            lora_params = LoraRuntimeParams(
                lora_ranks=[lora_ranks],
                lora_weights_pointers=[lora_weights_pointers],
                host_request_types=host_request_types,
                weight_index=0)

            # Define output
            output = linear(inp, lora_runtime_params=lora_params)
            output.mark_output("output", self.dtype)

        return create_session(builder, network, precision=self.dtype)

    def _create_trt_inputs(self):
        """Create input dictionary for TensorRT session."""
        host_request_types = torch.zeros(self.batch_size, dtype=torch.int32)
        magnitude_dora = torch.zeros(self.hidden_size,
                                     dtype=tensorrt_llm.str_dtype_to_torch(
                                         self.dtype),
                                     device="cuda")

        inputs = {
            "input_tensor": self.activations,
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
        """Set up the pivot linear layer with weights."""
        pivot_linear = PivotLinear(in_features=self.hidden_size,
                                   out_features=self.hidden_size)
        pivot_linear = pivot_linear.to(
            dtype=tensorrt_llm.str_dtype_to_torch(self.dtype))
        pivot_linear.weight.data = self.weight
        return pivot_linear

    def test_lora_linear_layer(self):
        """Test LoRA linear layer implementation."""
        # Create TensorRT session
        session = self._create_linear_lora_trt_session()

        # Create input dictionary
        inputs = self._create_trt_inputs()

        # Run TensorRT inference
        outputs = run_session(session, inputs)
        torch.cuda.synchronize()

        # Set up and run pivot implementation
        pivot_linear = self._setup_pivot_linear()
        lora_params = {"lora_weight_ins": self.A, "lora_weight_outs": self.B}
        outputs_pivot = pivot_linear(self.activations, lora_params=lora_params)

        # Compare outputs
        torch.testing.assert_close(outputs["output"], outputs_pivot)
        print("Test passed!")


if __name__ == "__main__":
    test = TestLoraLinearPivotVsTrt()
    test.setup_method(None)  # None is passed as method parameter
    test.test_lora_linear_layer()

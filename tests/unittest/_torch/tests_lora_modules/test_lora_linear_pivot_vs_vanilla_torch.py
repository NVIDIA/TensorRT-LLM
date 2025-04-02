import unittest

import torch
from torch.nn import Linear

from tensorrt_llm._torch.modules.linear import Linear as PivotLinear
from tensorrt_llm._torch.peft.lora.layer import LoraModuleType


class TestLoraLinearPivotVsVanilla(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.in_dim = 64
        cls.out_dim = 64
        cls.input_length = 10
        cls.batch_size = 4
        cls.device = "cuda"
        cls.input_tensor = torch.randn(cls.batch_size,
                                       cls.input_length,
                                       cls.in_dim,
                                       device=cls.device)

    def _get_lora_params(self):
        lora_rank = 8
        lora_weight_ins = torch.randn(self.in_dim,
                                      lora_rank,
                                      device=self.device)
        lora_weight_outs = torch.randn(lora_rank,
                                       self.out_dim,
                                       device=self.device)

        lora_params = {
            'num_seqs': self.batch_size,
            'host_request_types': torch.zeros(self.batch_size,
                                              dtype=torch.int32),
            'prompt_lens_cpu':
            torch.tensor([self.input_length] * self.batch_size),
            0: {  # layer_idx
                LoraModuleType.DENSE: {  # module_type
                    'adapter_size':
                    torch.tensor([lora_rank]),
                    'weight_pointers':
                    torch.tensor([
                        lora_weight_ins.data_ptr(),
                        lora_weight_outs.data_ptr()
                    ]),
                    'weight_tensors': [lora_weight_ins, lora_weight_outs],
                    'is_dora':
                    False
                }
            }
        }
        return lora_params

    def _setup_linear_layers(self):
        torch_linear = Linear(self.in_dim, self.out_dim).to(self.device)
        pivot_linear = PivotLinear(in_features=self.in_dim,
                                   out_features=self.out_dim,
                                   layer_idx=0)

        # Initialize pivot linear with same weights as torch linear
        pivot_linear.weight.data = torch_linear.weight.data
        pivot_linear.bias.data = torch_linear.bias.data

        return torch_linear, pivot_linear

    def test_compare_linear_torch_pivot_lora(self):
        lora_params = self._get_lora_params()
        torch_linear, pivot_linear = self._setup_linear_layers()

        lora_weight_ins = lora_params[0][
            LoraModuleType.DENSE]['weight_tensors'][0]
        lora_weight_outs = lora_params[0][
            LoraModuleType.DENSE]['weight_tensors'][1]
        lora_output = (
            self.input_tensor @ lora_weight_outs.T) @ lora_weight_ins.T

        torch_output = torch_linear(self.input_tensor)
        torch_output = torch_output + lora_output

        pivot_output = pivot_linear(self.input_tensor, lora_params=lora_params)

        self.assertTrue(torch.allclose(torch_output, pivot_output))

    def test_compare_linear_torch_pivot(self):
        torch_linear, pivot_linear = self._setup_linear_layers()

        torch_output = torch_linear(self.input_tensor)
        pivot_output = pivot_linear(self.input_tensor)

        torch.testing.assert_close(torch_output, pivot_output)

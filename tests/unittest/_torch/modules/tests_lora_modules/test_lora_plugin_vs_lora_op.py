import os
import sys
import unittest

import torch

import tensorrt_llm
from tensorrt_llm import Tensor

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
from utils.util import create_session, run_session


class TestLoraPluginVsLayer(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('info')
        torch.random.manual_seed(0)
        self.dtype = 'float16'
        self.torch_dtype = torch.float16
        self.device = 'cuda'
        self.batch_size = 4
        self.seq_len = 8
        self.hidden_size = 1024
        self.lora_rank = 8
        self.is_remove_input_padding = False
        self.weight_index = 0
        self.transA = False
        self.transB = True

    def _create_input_tensors(self, batch_size, seq_len, hidden_size,
                              lora_ranks_list):
        input_tensor = torch.randn(batch_size,
                                   seq_len,
                                   hidden_size,
                                   dtype=self.torch_dtype,
                                   device=self.device) * 0.1

        lora_weight_ins = [
            torch.randn(hidden_size, lora_rank, device=self.device).to(
                self.torch_dtype) * 0.1 for lora_rank in lora_ranks_list
        ]
        lora_weight_outs = [
            torch.randn(lora_rank, hidden_size, device=self.device).to(
                self.torch_dtype) * 0.1 for lora_rank in lora_ranks_list
        ]

        lora_weight_ins = [tmp.contiguous() for tmp in lora_weight_ins]
        lora_weight_outs = [
            tmp.transpose(1, 0).contiguous() for tmp in lora_weight_outs
        ]

        # Create LoRA weight pointers
        lora_weights_pointers = []
        for in_ptr, out_ptr in zip(lora_weight_ins, lora_weight_outs):
            lora_weights_pointers.append(in_ptr.data_ptr())
            lora_weights_pointers.append(out_ptr.data_ptr())
            # null dora scale
            lora_weights_pointers.append(0)

        lora_weights_pointers = torch.LongTensor(lora_weights_pointers).to(
            torch.int64).reshape([batch_size, 3])

        # Create other tensors
        host_context_lengths = torch.Tensor(
            [seq_len for _ in range(batch_size)]).to(torch.int32)
        lora_ranks = torch.Tensor(lora_ranks_list).to(torch.int32)
        host_request_types = torch.zeros_like(host_context_lengths,
                                              device='cpu').int()

        return {
            'input_tensor': input_tensor,
            'lora_weight_ins': lora_weight_ins,
            'lora_weight_outs': lora_weight_outs,
            'lora_weights_pointers': lora_weights_pointers,
            'host_context_lengths': host_context_lengths,
            'lora_ranks': lora_ranks,
            'host_request_types': host_request_types,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'hidden_size': hidden_size,
            'max_lora_rank': max(max(lora_ranks_list), 8)
        }

    def _create_lora_plugin_session(self, tensors):
        # Construct TensorRT network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        network.plugin_config.set_lora_plugin(self.dtype)
        network.plugin_config.remove_input_padding = self.is_remove_input_padding

        with tensorrt_llm.net_guard(network):
            input_tensor = Tensor(name='input_tensor',
                                  shape=[
                                      tensors['batch_size'], tensors['seq_len'],
                                      tensors['hidden_size']
                                  ],
                                  dtype=tensorrt_llm.str_dtype_to_trt(
                                      self.dtype))
            host_request_types_tensor = Tensor(
                name='host_request_types',
                shape=[tensors['batch_size']],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            host_context_lengths_tensor = Tensor(
                name='host_context_lengths',
                shape=[tensors['batch_size']],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            lora_ranks_tensor = Tensor(
                name='lora_ranks',
                shape=[tensors['batch_size']],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            lora_weights_pointers_tensor = Tensor(
                name='lora_weights_pointers',
                shape=[tensors['batch_size'], 3],
                dtype=tensorrt_llm.str_dtype_to_trt('int64'))

            output = tensorrt_llm.functional.lora_plugin(
                input_tensor,
                tensors['hidden_size'],
                [tensors['hidden_size']],
                host_request_types_tensor,
                self.transA,  # transA
                self.transB,  # transB
                host_context_lengths_tensor,
                tensors['max_lora_rank'],
                [lora_ranks_tensor],
                [lora_weights_pointers_tensor],
                weight_index=self.weight_index,
            )
            output.mark_output('output')

        return create_session(builder, network, precision=self.dtype)

    def _run_lora_grouped_gemm(self, tensors):
        """Run the lora_grouped_gemm operation directly"""
        # Prepare parameters for lora_grouped_gemm
        x = tensors['input_tensor']
        host_request_types = tensors[
            'host_request_types'][:tensors['batch_size']]
        lora_ranks = tensors['lora_ranks']
        lora_weight_pointers = tensors['lora_weights_pointers']
        prompt_lens_cpu = tensors[
            'host_context_lengths'][:tensors['batch_size']]
        output_hidden_sizes = [tensors['hidden_size']]
        transA = self.transA
        transB = self.transB
        max_rank = max([r.item() for r in lora_ranks])
        weight_index = self.weight_index
        is_remove_input_padding = self.is_remove_input_padding

        lora_outputs = torch.ops.trtllm.lora_grouped_gemm(
            x, host_request_types, [lora_ranks], [lora_weight_pointers],
            prompt_lens_cpu, output_hidden_sizes, transA, transB, max_rank,
            weight_index, is_remove_input_padding)

        return lora_outputs[0]

    def test_lora_plugin_vs_lora_op(self):
        lora_ranks_list = [self.lora_rank] * self.batch_size

        tensors = self._create_input_tensors(self.batch_size, self.seq_len,
                                             self.hidden_size, lora_ranks_list)

        session = self._create_lora_plugin_session(tensors)
        inputs = {
            'input_tensor': tensors['input_tensor'],
            'host_request_types': tensors['host_request_types'],
            'host_context_lengths': tensors['host_context_lengths'],
            'lora_ranks': tensors['lora_ranks'],
            'lora_weights_pointers': tensors['lora_weights_pointers'],
        }
        outputs = run_session(session, inputs)
        torch.cuda.synchronize()

        lora_outputs = self._run_lora_grouped_gemm(tensors)

        torch.testing.assert_close(outputs['output'],
                                   lora_outputs,
                                   atol=5e-3,
                                   rtol=0.3)

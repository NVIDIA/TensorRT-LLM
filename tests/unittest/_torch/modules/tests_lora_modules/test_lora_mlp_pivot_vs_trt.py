import unittest

import torch

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm._torch.modules.mlp import MLP as PivotMLP
from tensorrt_llm._torch.peft.lora.layer import LoraModuleType
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.layers import MLP as TRTMLP
from tensorrt_llm.layers.lora import Lora, LoraParams


class TestLoraMLPPivotVsTRT(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 1
        cls.seq_len = 16
        cls.hidden_size = 64
        cls.intermediate_size = cls.hidden_size * 4
        cls.num_hidden_layers = 1
        cls.dtype = 'float16'
        cls.torch_dtype = str_dtype_to_torch(cls.dtype)
        cls.device = torch.device('cuda')

    def _create_mlp_inputs(self):
        hidden_states = torch.empty(
            size=[self.batch_size, self.seq_len, self.hidden_size],
            dtype=self.torch_dtype,
            device='cuda')
        hidden_states.normal_(0.0, 0.02)

        return hidden_states

    def _get_lora_params(self, in_dim, out_dim):
        lora_rank = 8
        A = torch.randn(in_dim,
                        lora_rank,
                        device=self.device,
                        dtype=self.torch_dtype)
        B = torch.randn(lora_rank,
                        out_dim,
                        device=self.device,
                        dtype=self.torch_dtype)
        return A, B

    def _create_lora_params(self):
        lora_ranks_list = [8]

        host_context_lengths = torch.Tensor(
            [self.seq_len for _ in range(self.batch_size)]).to(torch.int32)

        lora_ranks = torch.tensor(lora_ranks_list * self.batch_size,
                                  dtype=torch.int32)

        host_request_types = torch.zeros_like(host_context_lengths,
                                              device='cpu').int()

        # Create weights for up projection
        lora_weight_ins_up = [
            torch.randn(self.hidden_size, lora_rank, device=self.device).to(
                self.torch_dtype) * 0.1 for lora_rank in lora_ranks_list
        ]
        lora_weight_outs_up = [
            torch.randn(lora_rank, self.intermediate_size,
                        device=self.device).to(self.torch_dtype) * 0.1
            for lora_rank in lora_ranks_list
        ]

        # Create weights for down projection
        lora_weight_ins_down = [
            torch.randn(self.intermediate_size, lora_rank,
                        device=self.device).to(self.torch_dtype) * 0.1
            for lora_rank in lora_ranks_list
        ]
        lora_weight_outs_down = [
            torch.randn(lora_rank, self.hidden_size, device=self.device).to(
                self.torch_dtype) * 0.1 for lora_rank in lora_ranks_list
        ]

        lora_weight_ins_up = [tmp.contiguous() for tmp in lora_weight_ins_up]
        lora_weight_outs_up = [
            tmp.transpose(1, 0).contiguous() for tmp in lora_weight_outs_up
        ]
        lora_weight_ins_down = [
            tmp.contiguous() for tmp in lora_weight_ins_down
        ]
        lora_weight_outs_down = [
            tmp.transpose(1, 0).contiguous() for tmp in lora_weight_outs_down
        ]

        # Create weight pointers for TensorRT
        lora_weights_pointers_up = []
        for in_ptr, out_ptr in zip(lora_weight_ins_up, lora_weight_outs_up):
            lora_weights_pointers_up.append(in_ptr.data_ptr())
            lora_weights_pointers_up.append(out_ptr.data_ptr())

        lora_weights_pointers_down = []
        for in_ptr, out_ptr in zip(lora_weight_ins_down, lora_weight_outs_down):
            lora_weights_pointers_down.append(in_ptr.data_ptr())
            lora_weights_pointers_down.append(out_ptr.data_ptr())

        lora_weights_pointers_up = torch.LongTensor(
            lora_weights_pointers_up).to(torch.int64).reshape(
                [self.batch_size, 2])
        lora_weights_pointers_down = torch.LongTensor(
            lora_weights_pointers_down).to(torch.int64).reshape(
                [self.batch_size, 2])

        return {
            'lora_ranks': lora_ranks,
            'host_context_lengths': host_context_lengths,
            'host_request_types': host_request_types,
            'lora_weights_pointers_up': lora_weights_pointers_up,
            'lora_weights_pointers_down': lora_weights_pointers_down,
            'lora_weight_ins_up': lora_weight_ins_up,
            'lora_weight_outs_up': lora_weight_outs_up,
            'lora_weight_ins_down': lora_weight_ins_down,
            'lora_weight_outs_down': lora_weight_outs_down
        }

    def _setup_mlp_module(self):

        mlp_module = PivotMLP(hidden_size=self.hidden_size,
                              intermediate_size=self.intermediate_size,
                              bias=True,
                              activation=torch.nn.functional.silu,
                              dtype=self.torch_dtype,
                              layer_idx=0).to(self.device)
        return mlp_module

    def _setup_trt_network(self, hidden_states, mlp_module, lora_params):
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        net.plugin_config.to_legacy_setting()
        net.plugin_config.lora_plugin = self.dtype
        net.plugin_config.remove_input_padding = False

        with tensorrt_llm.net_guard(net):
            trt_hidden_states = Tensor(name='hidden_states',
                                       shape=hidden_states.shape,
                                       dtype=tensorrt_llm.str_dtype_to_trt(
                                           self.dtype))

            host_request_types = Tensor(
                name='host_request_types',
                shape=[lora_params['host_request_types'].shape[0]],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            host_context_lengths = Tensor(
                name='host_context_lengths',
                shape=[lora_params['host_context_lengths'].shape[0]],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            lora_ranks = Tensor(name='lora_ranks',
                                shape=(lora_params['lora_ranks'].shape[0], ),
                                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            lora_weights_pointers_up = Tensor(
                name='lora_weights_pointers_up',
                shape=lora_params['lora_weights_pointers_up'].shape,
                dtype=tensorrt_llm.str_dtype_to_trt('int64'))
            lora_weights_pointers_down = Tensor(
                name='lora_weights_pointers_down',
                shape=lora_params['lora_weights_pointers_down'].shape,
                dtype=tensorrt_llm.str_dtype_to_trt('int64'))

            mlp_layer = TRTMLP(
                hidden_size=self.hidden_size,
                ffn_hidden_size=self.intermediate_size,
                hidden_act='silu',  # not tested gated activations
                bias=True,
                dtype=self.dtype)

            # Create LoRA layers for both linear projections
            mlp_layer.fc.lora = Lora(
                in_hidden_size=self.hidden_size,
                out_hidden_sizes=[self.intermediate_size],
                max_low_rank=8,
            )
            mlp_layer.proj.lora = Lora(
                in_hidden_size=self.intermediate_size,
                out_hidden_sizes=[self.hidden_size],
                max_low_rank=8,
            )

            # Set weights
            mlp_layer.fc.weight.value = mlp_module.up_proj.weight.data
            mlp_layer.fc.bias.value = mlp_module.up_proj.bias.data

            mlp_layer.proj.weight.value = mlp_module.down_proj.weight.data
            mlp_layer.proj.bias.value = mlp_module.down_proj.bias.data

            # Create LoRA parameters for TensorRT
            trt_lora_params = LoraParams(
                lora_ranks=[{
                    "mlp_h_to_4h_lora_ranks": lora_ranks,
                    "mlp_4h_to_h_lora_ranks": lora_ranks,
                }],
                lora_weights_pointers=[{
                    "mlp_h_to_4h_lora_weights_pointers":
                    lora_weights_pointers_up,
                    "mlp_4h_to_h_lora_weights_pointers":
                    lora_weights_pointers_down,
                }],
                host_context_lengths=host_context_lengths,
                host_request_types=host_request_types)

            output = mlp_layer(trt_hidden_states,
                               lora_layer_params=trt_lora_params)
            output.mark_output('output',
                               tensorrt_llm.str_dtype_to_trt(self.dtype))

        return builder, net

    def _run_trt_inference(self, builder, net, hidden_states, lora_params):
        builder_config = builder.create_builder_config(name='mlp',
                                                       precision=self.dtype)
        engine_buffer = builder.build_engine(net, builder_config)
        session = tensorrt_llm.runtime.Session.from_serialized_engine(
            engine_buffer)

        stream = torch.cuda.current_stream().cuda_stream
        inputs = {
            'hidden_states': hidden_states,
            'host_request_types': lora_params['host_request_types'],
            'host_context_lengths': lora_params['host_context_lengths'],
            'lora_ranks': lora_params['lora_ranks'],
            'lora_weights_pointers_up': lora_params['lora_weights_pointers_up'],
            'lora_weights_pointers_down':
            lora_params['lora_weights_pointers_down'],
        }

        outputs = {
            'output':
            torch.empty(hidden_states.shape,
                        dtype=tensorrt_llm._utils.str_dtype_to_torch(
                            self.dtype),
                        device='cuda'),
        }

        session.run(inputs=inputs, outputs=outputs, stream=stream)
        torch.cuda.synchronize()

        return outputs['output']

    def test_mlp(self):
        hidden_states = self._create_mlp_inputs()
        lora_params = self._create_lora_params()

        mlp_module = self._setup_mlp_module()

        builder, net = self._setup_trt_network(hidden_states, mlp_module,
                                               lora_params)
        trt_output = self._run_trt_inference(builder, net, hidden_states,
                                             lora_params)

        # Create LoRA parameters for PyTorch MLP
        lora_params_pivot = {
            'num_seqs': self.batch_size,
            'host_request_types': lora_params['host_request_types'],
            'prompt_lens_cpu': lora_params['host_context_lengths'],
            0: {
                LoraModuleType.MLP_H_TO_4H: {
                    'adapter_size': lora_params['lora_ranks'],
                    'weight_pointers': lora_params['lora_weights_pointers_up'],
                    'is_dora': False,
                },
                LoraModuleType.MLP_4H_TO_H: {
                    'adapter_size': lora_params['lora_ranks'],
                    'weight_pointers':
                    lora_params['lora_weights_pointers_down'],
                    'is_dora': False,
                }
            }
        }

        pivot_output = mlp_module(hidden_states, lora_params_pivot)

        torch.testing.assert_close(pivot_output, trt_output, atol=2e-3, rtol=0)


if __name__ == "__main__":
    unittest.main()

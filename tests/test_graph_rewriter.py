# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import unittest

import numpy as np

# isort: off
import torch
import tensorrt as trt
# isort: on
from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

import tensorrt_llm
import tensorrt_llm as tllm
from tensorrt_llm import Tensor, net_guard
from tensorrt_llm._utils import torch_to_numpy
from tensorrt_llm.functional import PositionEmbeddingType, gpt_attention
from tensorrt_llm.graph_rewriting import (AnalysisPatternManager,
                                          FLayerInfoMemo,
                                          FuseAttentionWithBiasPass, Layer,
                                          PatternAnalyzer, PatternRewriter,
                                          RewritePatternManager)
from tensorrt_llm.layers import Linear
from tensorrt_llm.quantization import QuantMode


# Borrowed from test_gpt_attention.py, and pruned unnecessary logics.
def create_gpt_attention_network(attention_type='gpt2_attention',
                                 dtype='float32') -> tensorrt_llm.Network:

    def _construct_execution(
        shape_dict,
        weight,
        bias,
        num_heads,
        hidden_size,
        dtype,
        enable_multi_query_attention,
        in_len,
    ):

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        net.plugin_config.set_gpt_attention_plugin(dtype)
        net.plugin_config.remove_input_padding = True

        head_size = hidden_size // num_heads
        with tensorrt_llm.net_guard(net):
            x_tensor = Tensor(name='input',
                              shape=shape_dict['input'],
                              dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            past_key_value_tensor = Tensor(
                name='past_key_value',
                shape=tuple(shape_dict['past_key_value']),
                dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            sequence_length_tensor = Tensor(
                name='sequence_length',
                shape=shape_dict['sequence_length'],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            host_past_key_value_lengths_tensor = Tensor(
                name='host_past_key_value_lengths',
                shape=shape_dict['host_past_key_value_lengths'],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            host_max_attention_window_sizes_tensor = Tensor(
                name='host_max_attention_window_sizes',
                shape=shape_dict['host_max_attention_window_sizes'],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            host_sink_token_length_tensor = Tensor(
                name='host_sink_token_length',
                shape=shape_dict['host_sink_token_length'],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            context_lengths_tensor = Tensor(
                name='context_lengths',
                shape=shape_dict['context_lengths'],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            host_request_types_tensor = Tensor(
                name='host_request_types',
                shape=shape_dict['host_request_types'],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            host_context_lengths_tensor = None
            if net.plugin_config.remove_input_padding:
                host_context_lengths_tensor = Tensor(
                    name='host_context_lengths',
                    shape=shape_dict['host_context_lengths'],
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            cache_indirection_tensor = Tensor(
                name='cache_indirection',
                shape=shape_dict['cache_indirection'],
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            linear = Linear(hidden_size,
                            weight.size()[-1],
                            bias=attention_type in [
                                'gpt2_attention', 'llama_attention',
                                'gpt_bigcode_attention'
                            ])
            linear.weight.value = np.ascontiguousarray(
                torch_to_numpy(weight.T.cpu()))
            if attention_type in [
                    'gpt2_attention', 'llama_attention', 'gpt_bigcode_attention'
            ]:
                linear.bias.value = torch_to_numpy(bias.cpu())
            qkv = linear(x_tensor)

            rotary_embedding_dim = head_size if attention_type in [
                'llama_attention', 'gptj_attention'
            ] else 0

            if attention_type == 'llama_attention':
                position_embedding_type = PositionEmbeddingType.rope_gpt_neox
            elif attention_type == 'gptj_attention':
                position_embedding_type = PositionEmbeddingType.rope_gptj
            else:
                position_embedding_type = PositionEmbeddingType.learned_absolute
            outputs = tensorrt_llm.functional.gpt_attention(
                qkv=qkv,
                past_key_value=past_key_value_tensor,
                sequence_length=sequence_length_tensor,
                host_past_key_value_lengths=host_past_key_value_lengths_tensor,
                host_max_attention_window_sizes=
                host_max_attention_window_sizes_tensor,
                host_sink_token_length=host_sink_token_length_tensor,
                context_lengths=context_lengths_tensor,
                cache_indirection=cache_indirection_tensor,
                host_request_types=host_request_types_tensor,
                num_heads=num_heads,
                num_kv_heads=1 if enable_multi_query_attention else num_heads,
                hidden_size_per_head=head_size,
                q_scaling=1.0,
                max_context_length=in_len,
                rotary_embedding_dim=rotary_embedding_dim,
                position_embedding_type=position_embedding_type,
                kv_orig_quant_scale=None,
                kv_quant_orig_scale=None,
                kv_cache_quant_mode=QuantMode.from_description(
                    use_int8_kv_cache=False),
                kv_cache_block_pointers=None,
                host_context_lengths=host_context_lengths_tensor)

            net._mark_output(outputs[0],
                             'output',
                             dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            net._mark_output(outputs[1],
                             'present_key_value',
                             dtype=tensorrt_llm.str_dtype_to_trt('float16'))

        return net

    enable_multi_query_attention = False
    batch_size = 4
    in_len = 128
    max_seq_len = 148
    hidden_size = 1024
    num_heads = 16
    head_size = hidden_size // num_heads
    kv_num_heads = 1 if enable_multi_query_attention and attention_type == 'gpt_bigcode_attention' else num_heads
    qkv_hidden_size = hidden_size + 2 * kv_num_heads * head_size
    shape_dict = {
        'weight': (hidden_size, qkv_hidden_size),
        'bias': (qkv_hidden_size, ),
        'past_key_value': (batch_size, 2, kv_num_heads, max_seq_len, head_size),
        'present_key_value':
        (batch_size, 2, kv_num_heads, max_seq_len, head_size),
        'sequence_length': (batch_size, ),
        'context_lengths': (batch_size, ),
        'host_context_lengths': (batch_size, ),
        'host_request_types': (batch_size, ),
        'max_input_sequence_length': (in_len, ),
        'cache_indirection': (batch_size, 1, max_seq_len),
        'input': (batch_size, in_len, hidden_size),
        'output': (batch_size, in_len, hidden_size),
        'host_past_key_value_lengths': (batch_size, ),
        'host_max_attention_window_sizes': (1, ),
        'host_sink_token_length': (1, )
    }

    weight = torch.randn(shape_dict['weight'],
                         dtype=tllm._utils.str_dtype_to_torch(dtype),
                         device='cuda') * 1e-3
    bias = torch.randn(shape_dict['bias'],
                       dtype=tllm._utils.str_dtype_to_torch(dtype),
                       device='cuda') * 1e-2

    configuration = GPT2Config(
        hidden_size=hidden_size,
        n_layer=1,
        n_head=num_heads,
        vocab_size=51200,
        use_cache=True,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
        hidden_act='gelu',
        torch_dtype=dtype,
    )
    attention = GPT2Attention(configuration).cuda().eval()

    attention.c_attn.weight = torch.nn.parameter.Parameter(
        data=weight.clone().detach(), requires_grad=False)
    attention.c_attn.bias = torch.nn.parameter.Parameter(
        data=bias.clone().detach(), requires_grad=False)
    attention.c_proj.weight = torch.nn.parameter.Parameter(
        data=torch.eye(hidden_size,
                       dtype=tllm._utils.str_dtype_to_torch(dtype),
                       device='cuda'),
        requires_grad=False)
    attention.c_proj.bias = torch.nn.parameter.Parameter(data=torch.zeros(
        (hidden_size, ),
        dtype=tllm._utils.str_dtype_to_torch(dtype),
        device='cuda'),
                                                         requires_grad=False)

    return _construct_execution(
        shape_dict=shape_dict,
        weight=weight,
        bias=bias,
        num_heads=num_heads,
        hidden_size=hidden_size,
        dtype='float16',
        enable_multi_query_attention=enable_multi_query_attention,
        in_len=in_len)


class TestNetworkForGraphRewrite(unittest.TestCase):

    def setUp(self) -> None:
        self.network = create_gpt_attention_network('gpt2_attention', 'float32')

    def test_num_inputs(self):
        num_inputs = self.network.trt_network.num_inputs
        self.assertEqual(num_inputs, len(self.network.get_inputs()))

    def test_is_input(self):
        for x in self.network.get_inputs():
            self.assertTrue(self.network.is_input(x))

    def test_is_output(self):
        for x in self.network.get_outputs():
            self.assertTrue(self.network.is_output(x))

    def test_get_layer_by_name(self):
        for layer in self.network.get_layers():
            self.assertEqual(layer, self.network.get_layer_by_name(layer.name))

    def test_get_tensor_parent(self):
        for layer in self.network.get_layers():
            for tensor in layer.get_outputs():
                self.assertEqual(layer, tensor.get_parent())

    def test_get_tensor_users(self):
        for layer in self.network.get_layers():
            for tensor in layer.get_inputs():
                found_this_user = False
                for user in tensor.get_users():
                    if user == layer:
                        found_this_user = True
                        break
                self.assertTrue(found_this_user)

    def test_to_dot(self):
        dot_code = self.network.to_dot()
        self.assertTrue('digraph' in dot_code)


class TestLayer(unittest.TestCase):

    def setUp(self) -> None:
        self.network = create_gpt_attention_network('gpt2_attention', 'float32')

    def test_as_layer(self):
        for layer in self.network.get_layers():
            self.assertTrue(layer.as_layer())


class NaivePatternRewriter_ReplaceAddWithSub(PatternRewriter):

    def __init__(self):
        super().__init__('replace_add_with_sub',
                         root_layer={trt.LayerType.ELEMENTWISE},
                         seperate_match_rewrite=True)
        self.rewrite_count = 0

    def match(self, layer: Layer):
        # the rewriter will stop at the first matched layer, and the Rewriter will enter rewrite() to do the rewriting.
        return layer.as_layer().op == trt.ElementWiseOperation.SUM

    def rewrite(self, layer: Layer) -> None:
        # here just a statis for testing, free to add arbitrary logic here.
        self.rewrite_count += 1

        # the layer here should be an elementwise_SUM layer
        with net_guard(layer.network):
            # There are several stages to replace some subgraph with another subgraph:

            # Step 1: Get the input tensors and output tensors of the subgraph to replace.
            # - For elementwise_SUM, there are two inputs and one output.
            a, b = layer.get_inputs(0, 1)
            o = layer.get_outputs(0)[0]

            # Step 2: Create a new subgraph that takes the old one's inputs
            # - here we insert an elementwise_SUB layer, and c is the output
            c = a - b

            # Step 3: Redirect all the layers depending on the outputs of the old subgraph to the new subgraph's.
            # - After this, the SUM become dangling, and will be pruned by TensorRT when building the engine.
            # - Note that, there is no API in TensorRT python to remove a layer explicitly, the `replace_all_uses_with` is the only way to "remove" a layer.
            o.replace_all_uses_with(c)

            # Step 4: Mark the all the layers in the old subgraph as removed.
            # - This helps the PatternRewriter to skip the removed layers
            layer.mark_as_removed()


class NaivePatternAnalyzer_CountAdd(PatternAnalyzer):

    def __init__(self) -> None:
        super().__init__('count_add', root_layer={trt.LayerType.ELEMENTWISE})
        self.count = 0

    def match(self, layer: Layer) -> bool:
        return layer.as_layer().op == trt.ElementWiseOperation.SUM

    def analyze(self, layer: Layer) -> None:
        self.count += 1


class TestPatternRewriter(unittest.TestCase):

    def setUp(self) -> None:
        self.network = create_gpt_attention_network('gpt2_attention', 'float32')

    def test_pattern_rewriter(self):
        patterns = RewritePatternManager()
        patterns.add(
            label='replace_add_to_sub',
            pattern=NaivePatternRewriter_ReplaceAddWithSub(),
        )

        def get_input_key(layer) -> str:
            inputs = layer.get_inputs(0, 1)
            return '-'.join([x.name for x in inputs])

        add_input_to_users = {}
        for layer in self.network.get_layers():
            if layer.as_layer(
            ).type == trt.LayerType.ELEMENTWISE and layer.as_layer(
            ).op == trt.ElementWiseOperation.SUM:
                output_uses = list(layer.get_outputs(0)[0].get_users())
                add_input_to_users[get_input_key(layer)] = output_uses

        patterns.rewrite(self.network)

        # Only one layer hit this pattern: Linear/ELEMENTWISE_SUM_0
        self.assertEqual(patterns.get('replace_add_to_sub').rewrite_count, 1)

        # check all the users of the output tensors of the removed layers are removed, and the danling layer will be finally removed by TensorRT.
        for layer in self.network.get_layers():
            if layer.as_layer(
            ).type == trt.LayerType.ELEMENTWISE and layer.as_layer(
            ).op == trt.ElementWiseOperation.SUM:
                for out in layer.get_outputs():
                    self.assertEqual(len(list(out.get_users())), 0)

        # check the add layers are replaced by sub layers with same input and output consumers
        for layer in self.network.get_layers():
            if layer.as_layer(
            ).type == trt.LayerType.ELEMENTWISE and layer.as_layer(
            ).op == trt.ElementWiseOperation.SUB:
                output_uses = layer.get_outputs(0)[0].get_users()
                key = get_input_key(layer)
                if key in add_input_to_users:
                    src = [x.name for x in add_input_to_users[key]]
                    dst = [x.name for x in output_uses]
                    self.assertEqual(src, dst)


class TestPatternAnalyzer(unittest.TestCase):

    def setUp(self) -> None:
        self.network = create_gpt_attention_network('gpt2_attention', 'float32')

    def test_pattern_analyzer(self):
        patterns = AnalysisPatternManager()
        patterns.add(
            label='count_add',
            pattern=NaivePatternAnalyzer_CountAdd(),
        )

        patterns.analyze(self.network)
        # There is only one layer hit this pattern: ELEMENTWISE_SUM_0
        self.assertEqual(patterns.get('count_add').count, 1)


class GPTAttentionPluginRemovePaddingRewritePass(PatternRewriter):
    '''
    This is just a demo for altering the gpt_attention plugin to enable remove_padding by replacing the inputs:
        - tensor, shape from [batch_size, in_len, hidden_size] to [1, batch_size * in_len, hidden_size]
    '''

    def __init__(self):
        super().__init__('gpt_attention_plugin_remove_padding',
                         root_layer={trt.LayerType.PLUGIN_V2})

        self.count = 0

    def match_and_rewrite(self, layer: Layer) -> bool:
        if layer.as_layer().type != trt.LayerType.PLUGIN_V2 or \
                layer.as_layer().plugin.plugin_namespace != 'tensorrt_llm' or\
                layer.as_layer().plugin.plugin_type != 'GPTAttention':
            return False

        flayer = FLayerInfoMemo.instance().get(layer.name)
        assert flayer
        tensor_input: Tensor = flayer.get_input('qkv')
        if tensor_input.shape[0] == 1:  # already on remove-padding mode
            return False

        self.log_info(f'hit gpt_attention plugin: {layer.name}')

        assert self.args is not None, "args should be passed in from RewritePatternManager.rewrite()"
        batch_size, in_len, hidden_size = self.args['batch_size'], self.args[
            'in_len'], self.args['hidden_size']

        # record the times of rewriting
        self.count += 1

        new_inputs = flayer.clone_inputs()
        with net_guard(layer.network):
            # Step 1: create new inputs and repalce the original arglist
            input = Tensor(
                name='qkv',
                dtype=trt.float16,
                shape=(1, batch_size * in_len, hidden_size),
            )
            new_inputs['qkv'] = input

            # Step 2: create a new plugin instance
            new_outs = gpt_attention(**new_inputs)

            # Step 3: deprive all the users of the old plugin instance
            flayer.replace_outputs_uses_with(layer.network, new_outs)

            # Step 4: remove the old plugin instance
            layer.mark_as_removed()

        return True


class TestGPTAttentionPluginRemovePaddingRewritePass(unittest.TestCase):

    def setUp(self) -> None:
        self.network = create_gpt_attention_network('gpt2_attention', 'float32')

    def test_gpt_attention_plugin_remove_padding_rewrite_pass(self):
        with net_guard(self.network):
            for layer in FLayerInfoMemo.instance().data.values():
                print(layer.raw_inputs)

            RewritePatternManager().instance().add(
                "gpt_attention remove padding",
                GPTAttentionPluginRemovePaddingRewritePass())

            RewritePatternManager().instance().rewrite(self.network,
                                                       args=dict(
                                                           batch_size=4,
                                                           in_len=128,
                                                           hidden_size=1024))

            self.assertEqual(
                RewritePatternManager().instance().get(
                    "gpt_attention remove padding").count, 1)


class TestFuseAttentionWithBiasPass(unittest.TestCase):

    def setUp(self) -> None:
        self.network = create_gpt_attention_network('gpt2_attention', 'float32')

    def test_as_layer(self):
        res = False
        rewriter = FuseAttentionWithBiasPass()
        for layer in self.network.get_layers():
            res = res or rewriter.match_and_rewrite(layer)
        self.assertEqual(res, True)


if __name__ == '__main__':
    unittest.main()

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

from parameterized import parameterized

from tensorrt_llm.layers import (Attention, ColumnLinear, GatedMLP, RmsNorm,
                                 RowLinear)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import (GPTForCausalLM, LLaMAConfig, LLaMAForCausalLM,
                                 PretrainedConfig)
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo

# isort: off
from tensorrt_llm.quantization.layers import (
    Fp8RowwiseAttention, Fp8RowwiseGatedMLP, Fp8RowwiseRmsNorm,
    SmoothQuantAttention, SmoothQuantLayerNorm, SmoothQuantMLP,
    WeightOnlyQuantColumnLinear, WeightOnlyQuantRowLinear)
# isort: on
from utils.util import unittest_name_func

from tensorrt_llm.quantization.quantize import quantize


class TestQuant(unittest.TestCase):

    def test_weight_only_quant(self):
        quant_algo = QuantAlgo.W8A16
        config = {
            'architecture': 'GPTForCausalLM',
            'dtype': 'float16',
            'num_hidden_layers': 2,
            'num_attention_heads': 12,
            'hidden_size': 768,
            'vocab_size': 51200,
            'max_position_embeddings': 1024,
            'hidden_act': 'relu',
        }
        config = PretrainedConfig.from_dict(config)
        model = GPTForCausalLM(config)

        quant_model = quantize(model, QuantConfig(quant_algo))

        self.assertTrue(hasattr(quant_model, 'quant_mode'))

        self.assertTrue(
            isinstance(quant_model.transformer.layers[0].attention.qkv,
                       WeightOnlyQuantColumnLinear))
        self.assertTrue(
            isinstance(quant_model.transformer.layers[0].attention.dense,
                       WeightOnlyQuantRowLinear))
        self.assertTrue(
            isinstance(quant_model.transformer.layers[0].mlp.fc,
                       WeightOnlyQuantColumnLinear))
        self.assertTrue(
            isinstance(quant_model.transformer.layers[0].mlp.proj,
                       WeightOnlyQuantRowLinear))

        self.assertTrue(
            isinstance(quant_model.transformer.layers[1].attention.qkv,
                       WeightOnlyQuantColumnLinear))
        self.assertTrue(
            isinstance(quant_model.transformer.layers[1].attention.dense,
                       WeightOnlyQuantRowLinear))
        self.assertTrue(
            isinstance(quant_model.transformer.layers[1].mlp.fc,
                       WeightOnlyQuantColumnLinear))
        self.assertTrue(
            isinstance(quant_model.transformer.layers[1].mlp.proj,
                       WeightOnlyQuantRowLinear))

        self.assertTrue(isinstance(quant_model.lm_head, ColumnLinear))

    def test_weight_only_quant_exclude_modules(self):
        quant_algo = QuantAlgo.W8A16
        config = {
            'architecture': 'GPTForCausalLM',
            'dtype': 'float16',
            'num_hidden_layers': 1,
            'num_attention_heads': 12,
            'hidden_size': 768,
            'vocab_size': 51200,
            'max_position_embeddings': 1024,
            'hidden_act': 'relu',
        }
        config = PretrainedConfig.from_dict(config)
        model = GPTForCausalLM(config)

        quant_model = quantize(
            model,
            QuantConfig(quant_algo,
                        exclude_modules=[
                            'fc', 'dense', 'vocab_embedding',
                            'position_embedding', 'block_embedding'
                        ]))

        self.assertTrue(hasattr(quant_model, 'quant_mode'))

        self.assertTrue(
            isinstance(quant_model.transformer.layers[0].attention.qkv,
                       WeightOnlyQuantColumnLinear))
        self.assertTrue(
            isinstance(quant_model.transformer.layers[0].attention.dense,
                       RowLinear))
        self.assertTrue(
            isinstance(quant_model.transformer.layers[0].mlp.fc, ColumnLinear))
        self.assertTrue(
            isinstance(quant_model.transformer.layers[0].mlp.proj,
                       WeightOnlyQuantRowLinear))
        self.assertTrue(
            isinstance(quant_model.lm_head, WeightOnlyQuantColumnLinear))

    def test_convert_GPT_to_smooth_quant(self):
        config = {
            'architecture': 'GPTForCausalLM',
            'dtype': 'float16',
            'num_hidden_layers': 1,
            'num_attention_heads': 1,
            'hidden_size': 128,
            'vocab_size': 1024,
            'max_position_embeddings': 256,
            'hidden_act': 'gelu',
        }
        config = PretrainedConfig.from_dict(config)
        model = GPTForCausalLM(config)

        quant_algo = QuantAlgo.W8A8_SQ_PER_TENSOR_PLUGIN
        quant_config = QuantConfig(quant_algo)
        quant_model = quantize(model, quant_config)
        for layer in quant_model.transformer.layers:
            assert isinstance(layer.input_layernorm, SmoothQuantLayerNorm)
            assert isinstance(layer.post_layernorm, SmoothQuantLayerNorm)
            assert isinstance(layer.mlp, SmoothQuantMLP)
            assert isinstance(layer.attention, SmoothQuantAttention)

        assert quant_model.quant_mode == quant_config.quant_mode

    @parameterized.expand([(False, 1, 0), (True, 1, 0), (False, 2, 0),
                           (False, 2, 1), (True, 2, 0), (True, 2, 1)],
                          name_func=unittest_name_func)
    def test_fp8_rowwise_quant(self, use_meta_recipe: bool, world_size: int,
                               rank: int):
        mapping = Mapping(rank=rank, pp_size=world_size, world_size=world_size)
        config = LLaMAConfig(architecture='LlamaForCausalLM',
                             dtype='float16',
                             hidden_size=128,
                             num_hidden_layers=16,
                             num_attention_heads=16,
                             vocab_size=1024,
                             hidden_act='silu',
                             position_embedding_type='rope_gpt_neox',
                             max_position_embeddings=256,
                             mapping=mapping)
        model = LLaMAForCausalLM(config)

        quant_algo = QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN
        quant_config = QuantConfig(quant_algo, use_meta_recipe=use_meta_recipe)

        quant_model = quantize(model, quant_config)
        local_num_hidden_layers = len(quant_model.transformer.layers)
        for local_layer_idx, layer in enumerate(quant_model.transformer.layers):
            assert layer.layer_idx == local_layer_idx + rank * local_num_hidden_layers
            if use_meta_recipe and (layer.layer_idx == 0
                                    or layer.layer_idx == 15):
                assert isinstance(layer.input_layernorm, RmsNorm)
                assert isinstance(layer.attention, Attention)
                assert isinstance(layer.post_layernorm, RmsNorm)
                assert isinstance(layer.mlp, GatedMLP)
            elif use_meta_recipe:
                assert isinstance(layer.input_layernorm, RmsNorm)
                assert isinstance(layer.attention, Attention)
                assert isinstance(layer.post_layernorm, Fp8RowwiseRmsNorm)
                assert isinstance(layer.mlp, Fp8RowwiseGatedMLP)
            else:
                assert isinstance(layer.input_layernorm, Fp8RowwiseRmsNorm)
                assert isinstance(layer.attention, Fp8RowwiseAttention)
                assert isinstance(layer.post_layernorm, Fp8RowwiseRmsNorm)
                assert isinstance(layer.mlp, Fp8RowwiseGatedMLP)

        assert quant_model.quant_mode == quant_config.quant_mode


if __name__ == '__main__':
    unittest.main()

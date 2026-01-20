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
import argparse
import json
import os
import time

import numpy as np
import torch
from safetensors.torch import save_file

import tensorrt_llm
from tensorrt_llm.functional import LayerNormPositionType, LayerNormType
from tensorrt_llm.models.convert_utils import weight_only_quantize_dict
from tensorrt_llm.quantization import QuantAlgo


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="assets")
    parser.add_argument('--quant_ckpt_path', type=str, default=None)
    parser.add_argument('--model_name',
                        type=str,
                        default="large-v3",
                        choices=[
                            "large-v3-turbo",
                            "large-v3",
                            "large-v2",
                            "medium",
                            "small",
                            "base",
                            "tiny",
                            "medium.en",
                            "small.en",
                            "base.en",
                            "tiny.en",
                            "distil-large-v3",
                            "distil-large-v2",
                            "distil-medium.en",
                            "distil-small.en",
                        ])
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--logits_dtype',
                        type=str,
                        default='float16',
                        choices=['float16', 'float32'])
    parser.add_argument('--output_dir',
                        type=str,
                        default='tllm_checkpoint',
                        help='The path to save the TensorRT LLM checkpoint')
    parser.add_argument(
        '--use_weight_only',
        default=False,
        action="store_true",
        help='Quantize weights for the various GEMMs to INT4/INT8.'
        'See --weight_only_precision to set the precision')
    parser.add_argument(
        '--weight_only_precision',
        const='int8',
        type=str,
        nargs='?',
        default='int8',
        choices=['int8', 'int4'],
        help=
        'Define the precision for the weights when using weight-only quantization.'
        'You must also use --use_weight_only for that argument to have an impact.'
    )
    args = parser.parse_args()
    return args


def get_encoder_config(model_metadata: dict, dtype: str,
                       quant_algo: QuantAlgo) -> dict:
    model_is_multilingual = (model_metadata['n_vocab'] >= 51865)
    num_languages = model_metadata['n_vocab'] - 51765 - int(
        model_is_multilingual)
    return {
        'architecture': "WhisperEncoder",
        'dtype': dtype,
        'num_hidden_layers': model_metadata['n_audio_layer'],
        'num_attention_heads': model_metadata['n_audio_head'],
        'hidden_size': model_metadata['n_audio_state'],
        'max_position_embeddings': model_metadata['n_audio_ctx'],
        'has_position_embedding': True,
        'n_mels': model_metadata['n_mels'],
        'vocab_size': model_metadata['n_vocab'],
        'hidden_act': "gelu",
        'num_languages': num_languages,
        'quantization': {
            'quant_algo': quant_algo
        },
    }


def get_decoder_config(model_metadata: dict, dtype: str, logits_dtype: str,
                       quant_algo: QuantAlgo) -> dict:
    return {
        'architecture': "DecoderModel",
        'dtype': dtype,
        'logits_dtype': logits_dtype,
        'num_hidden_layers': model_metadata['n_text_layer'],
        'num_attention_heads': model_metadata['n_text_head'],
        'hidden_size': model_metadata['n_text_state'],
        'norm_epsilon': 1e-5,
        'vocab_size': model_metadata['n_vocab'],
        'hidden_act': "gelu",
        'use_parallel_embedding': False,
        'embedding_sharding_dim': 0,
        'max_position_embeddings': model_metadata['n_text_ctx'],
        'use_prompt_tuning': False,
        'head_size':
        model_metadata['n_text_state'] // model_metadata['n_text_head'],
        'has_position_embedding': True,
        'layernorm_type': LayerNormType.LayerNorm,
        'has_attention_qkvo_bias': True,
        'has_mlp_bias': True,
        'has_model_final_layernorm': True,
        'has_embedding_layernorm': False,
        'has_embedding_scale': False,
        'ffn_hidden_size': 4 * model_metadata['n_text_state'],
        'q_scaling': 1.0,
        'layernorm_position': LayerNormPositionType.pre_layernorm,
        'relative_attention': False,
        'max_distance': 0,
        'num_buckets': 0,
        'model_type': 'whisper',
        'rescale_before_lm_head': False,
        'encoder_hidden_size': model_metadata['n_text_state'],
        'encoder_num_heads': model_metadata['n_text_head'],
        'encoder_head_size': None,
        'skip_cross_kv': False,
        'quantization': {
            'quant_algo': quant_algo
        },
    }


def convert_openai_whisper_encoder(
    model_metadata: dict,
    model_params: dict,
    quant_algo: str = None,
):
    weights = {}

    def sinusoids(length, channels, max_timescale=10000):
        """Returns sinusoids for positional embedding"""
        assert channels % 2 == 0
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment *
                                   torch.arange(channels // 2))
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[
            np.newaxis, :]
        return torch.cat([torch.sin(scaled_time),
                          torch.cos(scaled_time)],
                         dim=1)

    weights['transformer.position_embedding.weight'] = sinusoids(
        model_metadata['n_audio_ctx'],
        model_metadata['n_audio_state']).contiguous()

    weights['transformer.conv1.weight'] = torch.unsqueeze(
        model_params['encoder.conv1.weight'], -1).contiguous()
    weights['transformer.conv1.bias'] = model_params[
        'encoder.conv1.bias'].contiguous()
    weights['transformer.conv2.weight'] = torch.unsqueeze(
        model_params['encoder.conv2.weight'], -1).contiguous()
    weights['transformer.conv2.bias'] = model_params[
        'encoder.conv2.bias'].contiguous()

    for i in range(model_metadata['n_audio_layer']):
        trtllm_layer_name_prefix = f'transformer.layers.{i}'

        weights[
            f'{trtllm_layer_name_prefix}.attention_layernorm.weight'] = model_params[
                'encoder.blocks.' + str(i) + '.attn_ln.weight'].contiguous()
        weights[
            f'{trtllm_layer_name_prefix}.attention_layernorm.bias'] = model_params[
                'encoder.blocks.' + str(i) + '.attn_ln.bias'].contiguous()

        t = torch.cat([
            model_params['encoder.blocks.' + str(i) + '.attn.query.weight'],
            model_params['encoder.blocks.' + str(i) + '.attn.key.weight'],
            model_params['encoder.blocks.' + str(i) + '.attn.value.weight']
        ],
                      dim=0).contiguous()

        weights[f'{trtllm_layer_name_prefix}.attention.qkv.weight'] = t

        bias_shape = model_params['encoder.blocks.' + str(i) +
                                  '.attn.query.bias'].shape
        dtype = model_params['encoder.blocks.' + str(i) +
                             '.attn.query.bias'].dtype
        fused_bias = torch.cat([
            model_params['encoder.blocks.' + str(i) + '.attn.query.bias'],
            torch.zeros([*bias_shape], dtype=dtype),
            model_params['encoder.blocks.' + str(i) + '.attn.value.bias']
        ],
                               dim=0).contiguous()

        weights[f'{trtllm_layer_name_prefix}.attention.qkv.bias'] = fused_bias

        t = model_params['encoder.blocks.' + str(i) +
                         '.attn.out.weight'].contiguous()

        weights[f'{trtllm_layer_name_prefix}.attention.dense.weight'] = t
        weights[
            f'{trtllm_layer_name_prefix}.attention.dense.bias'] = model_params[
                'encoder.blocks.' + str(i) + '.attn.out.bias'].contiguous()

        weights[
            f'{trtllm_layer_name_prefix}.mlp_layernorm.weight'] = model_params[
                'encoder.blocks.' + str(i) + '.mlp_ln.weight'].contiguous()
        weights[
            f'{trtllm_layer_name_prefix}.mlp_layernorm.bias'] = model_params[
                'encoder.blocks.' + str(i) + '.mlp_ln.bias'].contiguous()

        t = model_params['encoder.blocks.' + str(i) +
                         '.mlp.0.weight'].contiguous()
        weights[f'{trtllm_layer_name_prefix}.mlp.fc.weight'] = t

        weights[f'{trtllm_layer_name_prefix}.mlp.fc.bias'] = model_params[
            'encoder.blocks.' + str(i) + '.mlp.0.bias'].contiguous()

        t = model_params['encoder.blocks.' + str(i) +
                         '.mlp.2.weight'].contiguous()
        weights[f'{trtllm_layer_name_prefix}.mlp.proj.weight'] = t

        weights[f'{trtllm_layer_name_prefix}.mlp.proj.bias'] = model_params[
            'encoder.blocks.' + str(i) + '.mlp.2.bias'].contiguous()

    weights['transformer.ln_f.weight'] = model_params[
        'encoder.ln_post.weight'].contiguous()
    weights['transformer.ln_f.bias'] = model_params[
        'encoder.ln_post.bias'].contiguous()

    return weight_only_quantize_dict(weights,
                                     quant_algo=quant_algo,
                                     plugin=True)


def convert_openai_whisper_decoder(model_metadata: dict,
                                   model_params: dict,
                                   quant_algo: str = None):

    weights = {}

    weights['transformer.vocab_embedding.weight'] = model_params[
        'decoder.token_embedding.weight']
    weights['transformer.position_embedding.weight'] = model_params[
        'decoder.positional_embedding']
    weights['lm_head.weight'] = model_params[
        'decoder.token_embedding.weight'].clone()

    for i in range(model_metadata['n_text_layer']):
        trtllm_layer_name_prefix = f'transformer.layers.{i}'

        t = torch.cat([
            model_params['decoder.blocks.' + str(i) + '.attn.query.weight'],
            model_params['decoder.blocks.' + str(i) + '.attn.key.weight'],
            model_params['decoder.blocks.' + str(i) + '.attn.value.weight']
        ],
                      dim=0)
        weights[f'{trtllm_layer_name_prefix}.self_attention.qkv.weight'] = t

        t = model_params['decoder.blocks.' + str(i) +
                         '.attn.out.weight'].contiguous()
        weights[f'{trtllm_layer_name_prefix}.self_attention.dense.weight'] = t

        bias_shape = model_params['decoder.blocks.' + str(i) +
                                  '.attn.query.bias'].shape
        dtype = model_params['decoder.blocks.' + str(i) +
                             '.attn.query.bias'].dtype
        weights[
            f'{trtllm_layer_name_prefix}.self_attention.qkv.bias'] = torch.cat(
                [
                    model_params['decoder.blocks.' + str(i) +
                                 '.attn.query.bias'],
                    torch.zeros([*bias_shape], dtype=dtype),
                    model_params['decoder.blocks.' + str(i) +
                                 '.attn.value.bias']
                ],
                dim=0)
        weights[
            f'{trtllm_layer_name_prefix}.self_attention.dense.bias'] = model_params[
                'decoder.blocks.' + str(i) + '.attn.out.bias']

        weights[
            f'{trtllm_layer_name_prefix}.self_attention_layernorm.weight'] = model_params[
                'decoder.blocks.' + str(i) + '.attn_ln.weight']
        weights[
            f'{trtllm_layer_name_prefix}.self_attention_layernorm.bias'] = model_params[
                'decoder.blocks.' + str(i) + '.attn_ln.bias']

        t = torch.cat([
            model_params['decoder.blocks.' + str(i) +
                         '.cross_attn.query.weight'],
            model_params['decoder.blocks.' + str(i) + '.cross_attn.key.weight'],
            model_params['decoder.blocks.' + str(i) +
                         '.cross_attn.value.weight']
        ],
                      dim=0)
        weights[f'{trtllm_layer_name_prefix}.cross_attention.qkv.weight'] = t

        t = model_params['decoder.blocks.' + str(i) +
                         '.cross_attn.out.weight'].contiguous()
        weights[f'{trtllm_layer_name_prefix}.cross_attention.dense.weight'] = t

        bias_shape = model_params['decoder.blocks.' + str(i) +
                                  '.cross_attn.query.bias'].shape
        dtype = model_params['decoder.blocks.' + str(i) +
                             '.cross_attn.query.bias'].dtype
        cross_attn_qkv_bias = torch.cat([
            model_params['decoder.blocks.' + str(i) + '.cross_attn.query.bias'],
            torch.zeros([*bias_shape], dtype=dtype),
            model_params['decoder.blocks.' + str(i) + '.cross_attn.value.bias']
        ],
                                        dim=0)

        weights[
            f'{trtllm_layer_name_prefix}.cross_attention.qkv.bias'] = cross_attn_qkv_bias

        weights[
            f'{trtllm_layer_name_prefix}.cross_attention.dense.bias'] = model_params[
                'decoder.blocks.' + str(i) + '.cross_attn.out.bias']

        weights[
            f'{trtllm_layer_name_prefix}.cross_attention_layernorm.weight'] = model_params[
                'decoder.blocks.' + str(i) + '.cross_attn_ln.weight']
        weights[
            f'{trtllm_layer_name_prefix}.cross_attention_layernorm.bias'] = model_params[
                'decoder.blocks.' + str(i) + '.cross_attn_ln.bias']

        t = model_params['decoder.blocks.' + str(i) +
                         '.mlp.0.weight'].contiguous()
        weights[f'{trtllm_layer_name_prefix}.mlp.fc.weight'] = t

        t = model_params['decoder.blocks.' + str(i) +
                         '.mlp.2.weight'].contiguous()
        weights[f'{trtllm_layer_name_prefix}.mlp.proj.weight'] = t

        weights[f'{trtllm_layer_name_prefix}.mlp.fc.bias'] = model_params[
            'decoder.blocks.' + str(i) + '.mlp.0.bias']
        weights[f'{trtllm_layer_name_prefix}.mlp.proj.bias'] = model_params[
            'decoder.blocks.' + str(i) + '.mlp.2.bias']

        weights[
            f'{trtllm_layer_name_prefix}.mlp_layernorm.weight'] = model_params[
                'decoder.blocks.' + str(i) + '.mlp_ln.weight']
        weights[
            f'{trtllm_layer_name_prefix}.mlp_layernorm.bias'] = model_params[
                'decoder.blocks.' + str(i) + '.mlp_ln.bias']

    weights['transformer.ln_f.weight'] = model_params['decoder.ln.weight']
    weights['transformer.ln_f.bias'] = model_params['decoder.ln.bias']

    return weight_only_quantize_dict(weights,
                                     quant_algo=quant_algo,
                                     plugin=True)


if __name__ == '__main__':
    # TODO(qijun): Currently, the convert script depends on a torch op:
    # torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix,
    # which is included in tensorrt_llm Python package. Otherwise, the convert
    # script does not need to import tensorrt_llm. Will remove it after reimplementing
    # the op with PyTorch.
    print(tensorrt_llm.__version__)
    args = parse_arguments()
    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    quant_algo = None
    plugin_weight_only_quant_type = None
    if args.use_weight_only and args.weight_only_precision == 'int8':
        plugin_weight_only_quant_type = torch.int8
        quant_algo = QuantAlgo.W8A16
    elif args.use_weight_only and args.weight_only_precision == 'int4':
        plugin_weight_only_quant_type = torch.quint4x2
        quant_algo = QuantAlgo.W4A16
    elif args.use_weight_only and args.weight_only_precision == 'int4_gptq':
        quant_algo = QuantAlgo.W4A16_GPTQ

    model_path = os.path.join(args.model_dir, args.model_name + '.pt')
    assert os.path.exists(model_path), f"Model {model_path} does not exist."

    model = torch.load(model_path, map_location='cpu')
    print(f"Loaded model from {model_path}")
    model_metadata = model['dims']
    model_state_dict = model['model_state_dict']
    for param_tensor in model_state_dict:
        model_state_dict[param_tensor] = model_state_dict[param_tensor].half()

    def convert_and_save(component: str = "encoder"):
        # call get_encoder_config or get_decoder_config according to component
        if component == "encoder":
            config = get_encoder_config(model_metadata, args.dtype, quant_algo)
        else:
            config = get_decoder_config(model_metadata, args.dtype,
                                        args.logits_dtype, quant_algo)

        if args.use_weight_only and args.weight_only_precision == 'int4_gptq':
            config['quantization'].update({
                'has_zero_point': True,
            })

        component_save_dir = os.path.join(args.output_dir, component)
        if not os.path.exists(component_save_dir):
            os.makedirs(component_save_dir)

        with open(os.path.join(component_save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

        if component == "encoder":
            weights = convert_openai_whisper_encoder(model_metadata,
                                                     model_state_dict,
                                                     quant_algo=quant_algo)
        else:
            assert component == "decoder"
            weights = convert_openai_whisper_decoder(model_metadata,
                                                     model_state_dict,
                                                     quant_algo=quant_algo)

        save_file(weights, os.path.join(component_save_dir,
                                        f'rank0.safetensors'))

    print("Converting encoder checkpoints...")
    convert_and_save("encoder")
    print("Converting decoder checkpoints...")
    convert_and_save("decoder")

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')

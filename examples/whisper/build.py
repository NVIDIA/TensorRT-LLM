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
import os
import time

import torch
from weight import load_decoder_weight, load_encoder_weight

import tensorrt_llm
from tensorrt_llm import str_dtype_to_torch, str_dtype_to_trt
from tensorrt_llm.builder import Builder
from tensorrt_llm.functional import LayerNormPositionType, LayerNormType
from tensorrt_llm.logger import logger
from tensorrt_llm.models import quantize_model
from tensorrt_llm.network import net_guard
from tensorrt_llm.quantization import QuantMode

MODEL_ENCODER_NAME = "whisper_encoder"
MODEL_DECODER_NAME = "whisper_decoder"


def get_engine_name(model, dtype, tp_size=1, rank=0):
    return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)


def serialize_engine(engine, path):
    logger.info(f'Serializing engine to {path}...')
    tik = time.time()
    with open(path, 'wb') as f:
        f.write(engine)
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Engine serialized. Total time: {t}')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size',
                        type=int,
                        default=1,
                        help='world size, only support tensor parallelism now')
    parser.add_argument('--model_dir', type=str, default="assets")
    parser.add_argument('--model_name',
                        type=str,
                        default="large-v3",
                        choices=[
                            "large-v3",
                            "large-v2",
                        ])
    parser.add_argument('--quantize_dir', type=str, default="quantize/1-gpu")
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float16'])
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--max_batch_size', type=int, default=8)
    parser.add_argument('--max_input_len', type=int, default=14)
    parser.add_argument('--max_output_len', type=int, default=100)
    parser.add_argument('--max_beam_width', type=int, default=4)
    parser.add_argument(
        '--use_gpt_attention_plugin',
        nargs='?',
        const=None,
        type=str,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help=
        "Activates attention plugin. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument(
        '--use_bert_attention_plugin',
        nargs='?',
        const=None,
        type=str,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help=
        "Activates BERT attention plugin. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument(
        '--use_gemm_plugin',
        nargs='?',
        const=None,
        type=str,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help=
        "Activates GEMM plugin. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument(
        '--use_layernorm_plugin',
        nargs='?',
        const=None,
        type=str,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help=
        "Activates layernorm plugin. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument('--remove_input_padding',
                        default=False,
                        action='store_true')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='whisper_outputs',
        help=
        'The path to save the serialized engine files, timing cache file and model configs'
    )
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
    parser.add_argument(
        '--int8_kv_cache',
        default=False,
        action="store_true",
        help=
        'By default, we use dtype for KV cache. int8_kv_cache chooses int8 quantization for KV'
    )
    parser.add_argument('--debug_mode', action='store_true')
    args = parser.parse_args()
    logger.set_level(args.log_level)

    plugins_args = [
        'use_gemm_plugin', 'use_layernorm_plugin', 'use_gpt_attention_plugin',
        'use_bert_attention_plugin'
    ]
    for plugin_arg in plugins_args:
        if getattr(args, plugin_arg) is None:
            logger.info(
                f"plugin_arg is None, setting it as {args.dtype} automatically."
            )
            setattr(args, plugin_arg, args.dtype)

    if args.use_weight_only:
        args.quant_mode = QuantMode.from_description(
            quantize_weights=True,
            quantize_activations=False,
            use_int4_weights="int4" in args.weight_only_precision)
    else:
        args.quant_mode = QuantMode(0)

    if args.int8_kv_cache:
        args.quant_mode = args.quant_mode.set_int8_kv_cache()

    return args


def build_encoder(model, args):
    model_metadata = model['dims']
    model_params = model['model_state_dict']

    # cast params according dtype
    for k, v in model_params.items():
        model_params[k] = v.to(str_dtype_to_torch(args.dtype))

    builder = Builder()

    max_batch_size = args.max_batch_size
    hidden_states = model_metadata['n_audio_state']
    num_heads = model_metadata['n_audio_head']
    num_layers = model_metadata['n_audio_layer']

    model_is_multilingual = (model_metadata['n_vocab'] >= 51865)

    builder_config = builder.create_builder_config(
        name=MODEL_ENCODER_NAME,
        precision=args.dtype,
        tensor_parallel=1,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_size=hidden_states,
        max_batch_size=max_batch_size,
        int8=args.quant_mode.has_act_or_weight_quant(),
        n_mels=model_metadata['n_mels'],
        num_languages=model_metadata['n_vocab'] - 51765 -
        int(model_is_multilingual),
    )

    tensorrt_llm_whisper_encoder = tensorrt_llm.models.WhisperEncoder(
        model_metadata['n_mels'], model_metadata['n_audio_ctx'],
        model_metadata['n_audio_state'], model_metadata['n_audio_head'],
        model_metadata['n_audio_layer'], str_dtype_to_trt(args.dtype))

    if args.use_weight_only:
        tensorrt_llm_whisper_encoder = quantize_model(
            tensorrt_llm_whisper_encoder, args.quant_mode)

    load_encoder_weight(tensorrt_llm_whisper_encoder, model_metadata,
                        model_params, model_metadata['n_audio_layer'])

    network = builder.create_network()

    if args.use_gemm_plugin:
        network.plugin_config.set_gemm_plugin(dtype=args.use_gemm_plugin)
    if args.use_layernorm_plugin:
        network.plugin_config.set_layernorm_plugin(
            dtype=args.use_layernorm_plugin)
    if args.use_bert_attention_plugin:
        network.plugin_config.set_bert_attention_plugin(
            dtype=args.use_bert_attention_plugin)
    if args.remove_input_padding:
        network.plugin_config.enable_remove_input_padding()
    if args.use_weight_only:
        network.plugin_config.set_weight_only_quant_matmul_plugin(
            dtype=args.dtype)

    with net_guard(network):

        inputs = tensorrt_llm_whisper_encoder.prepare_inputs(
            args.max_batch_size)

        tensorrt_llm_whisper_encoder(*inputs)

        if args.debug_mode:
            for k, v in tensorrt_llm_whisper_encoder.named_network_outputs():
                network._mark_output(v, k, str_dtype_to_trt(args.dtype))

    engine = None
    engine_name = get_engine_name(MODEL_ENCODER_NAME, args.dtype, 1, 0)

    engine = builder.build_engine(network, builder_config)

    config_path = os.path.join(args.output_dir, 'encoder_config.json')
    builder.save_config(builder_config, config_path)

    serialize_engine(engine, os.path.join(args.output_dir, engine_name))


def build_decoder(model, args):

    model_metadata = model['dims']
    model_params = model['model_state_dict']

    # cast params according dtype
    for k, v in model_params.items():
        model_params[k] = v.to(str_dtype_to_torch(args.dtype))

    builder = Builder()

    timing_cache_file = os.path.join(args.output_dir, 'decoder_model.cache')
    builder_config = builder.create_builder_config(
        name=MODEL_DECODER_NAME,
        precision=args.dtype,
        timing_cache=timing_cache_file,
        tensor_parallel=args.world_size,
        num_layers=model_metadata['n_text_layer'],
        num_heads=model_metadata['n_text_head'],
        hidden_size=model_metadata['n_text_state'],
        vocab_size=model_metadata['n_vocab'],
        hidden_act="gelu",
        max_position_embeddings=model_metadata['n_text_ctx'],
        apply_query_key_layer_scaling=False,
        max_batch_size=args.max_batch_size,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        opt_level=None,
        cross_attention=True,
        has_position_embedding=True,
        has_token_type_embedding=False,
        int8=args.quant_mode.has_act_or_weight_quant(),
    )

    tensorrt_llm_whisper_decoder = tensorrt_llm.models.DecoderModel(
        num_layers=model_metadata['n_text_layer'],
        num_heads=model_metadata['n_text_head'],
        hidden_size=model_metadata['n_text_state'],
        ffn_hidden_size=4 * model_metadata['n_text_state'],
        encoder_hidden_size=model_metadata['n_text_state'],
        encoder_num_heads=model_metadata['n_text_head'],
        vocab_size=model_metadata['n_vocab'],
        head_size=model_metadata['n_text_state'] //
        model_metadata['n_text_head'],
        max_position_embeddings=model_metadata['n_text_ctx'],
        has_position_embedding=True,
        relative_attention=False,
        max_distance=0,
        num_buckets=0,
        has_embedding_layernorm=False,
        has_embedding_scale=False,
        q_scaling=1.0,
        has_attention_qkvo_bias=True,
        has_mlp_bias=True,
        has_model_final_layernorm=True,
        layernorm_eps=1e-5,
        layernorm_position=LayerNormPositionType.pre_layernorm,
        layernorm_type=LayerNormType.LayerNorm,
        hidden_act="gelu",
        rescale_before_lm_head=False,
        dtype=str_dtype_to_trt(args.dtype),
        logits_dtype=str_dtype_to_trt(args.dtype))

    if args.use_weight_only:
        tensorrt_llm_whisper_decoder = quantize_model(
            tensorrt_llm_whisper_decoder, args.quant_mode)

    load_decoder_weight(
        tensorrt_llm_whisper_decoder,
        model_params,
    )

    network = builder.create_network()

    if args.use_gemm_plugin:
        network.plugin_config.set_gemm_plugin(dtype=args.use_gemm_plugin)
    if args.use_layernorm_plugin:
        network.plugin_config.set_layernorm_plugin(
            dtype=args.use_layernorm_plugin)
    if args.use_gpt_attention_plugin:
        network.plugin_config.set_gpt_attention_plugin(
            dtype=args.use_gpt_attention_plugin)
    if args.remove_input_padding:
        network.plugin_config.enable_remove_input_padding()

    with net_guard(network):
        inputs = tensorrt_llm_whisper_decoder.prepare_inputs(
            args.max_batch_size,
            args.max_beam_width,
            args.max_input_len,
            args.max_output_len,
            model_metadata['n_audio_ctx'],
        )

        tensorrt_llm_whisper_decoder(*inputs)

        if args.debug_mode:
            for k, v in tensorrt_llm_whisper_decoder.named_network_outputs():
                network._mark_output(v, k, str_dtype_to_trt(args.dtype))

    engine = None
    engine_name = get_engine_name(MODEL_DECODER_NAME, args.dtype, 1, 0)

    engine = builder.build_engine(network, builder_config)

    config_path = os.path.join(args.output_dir, 'decoder_config.json')
    builder.save_config(builder_config, config_path)

    serialize_engine(engine, os.path.join(args.output_dir, engine_name))


def run_build(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model_path = os.path.join(args.model_dir, args.model_name + '.pt')
    model = torch.load(model_path)
    build_encoder(model, args)
    build_decoder(model, args)


if __name__ == '__main__':
    args = parse_arguments()
    run_build(args)

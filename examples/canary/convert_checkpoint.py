# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import logging
import os
import time
import shutil

import nemo.collections.asr.models as nemo_asr
import onnx
import torch
import numpy as np
from omegaconf import OmegaConf
from safetensors.torch import save_file
import onnx_graphsurgeon as gs

import tensorrt_llm
from tensorrt_llm.functional import LayerNormPositionType, LayerNormType
from tensorrt_llm.network import set_plugin_info
from tensorrt_llm.quantization import QuantAlgo

TORCH_DTYPES = {
    'float32': torch.float32,
    'float64': torch.float64,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quant_ckpt_path', type=str, default=None)
    parser.add_argument('--model_name',
                        type=str,
                        default="nvidia/canary-1b-flash",
                        choices=[
                            "nvidia/canary-1b",
                            "nvidia/canary-1b-flash",
                            "nvidia/canary-180m-flash",
                        ])
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
    )
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
                        help='The path to save the TensorRT-LLM checkpoint')
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
    parser.add_argument('engine_dir')
    args = parser.parse_args()
    return args


class UnsupportedModel(Exception):

    def __init__(self, *args, model_name):
        super().__init__(*args)
        self.model_name = model_name


class CudaOOMInExportOfASRWithMaxDim(Exception):

    def __init__(self, *args, max_dim=None):
        super().__init__(*args)
        self.max_dim = max_dim


class CanaryModel:

    def __init__(self, args):
        self.args = args
        quant_algo = None
        self.plugin_weight_only_quant_type = None
        if args.use_weight_only and args.weight_only_precision == 'int8':
            self.plugin_weight_only_quant_type = torch.int8
            quant_algo = QuantAlgo.W8A16
        elif args.use_weight_only and args.weight_only_precision == 'int4':
            self.plugin_weight_only_quant_type = torch.quint4x2
            quant_algo = QuantAlgo.W4A16
        elif args.use_weight_only and args.weight_only_precision == 'int4_gptq':
            quant_algo = QuantAlgo.W4A16_GPTQ

        self.dtype = args.dtype
        self.logits_dtype = args.logits_dtype
        self.use_weight_only = args.use_weight_only
        self.weight_only_precision = args.weight_only_precision

        self.quant_algo = quant_algo
        self.engine_dir = args.engine_dir

        if args.model_path is not None:
            if self.is_supported(args.model_name):
                self.model_name = args.model_name
                try:
                    with torch.inference_mode():
                        # Restore instance from .nemo file using generic model restore_from
                        self.model = nemo_asr.EncDecMultiTaskModel.restore_from(
                            args.model_path).to('cpu')

                except Exception as e:
                    logging.error(
                        f"Failed to restore model from NeMo file : {args.model_path}. "
                    )
                    raise e
            else:
                raise UnsupportedModel(model_name=args.model_name)
        else:
            if args.model_name == 'nvidia/canary-1b' or args.model_name == 'nvidia/canary-1b-flash' or args.model_name == 'nvidia/canary-180m-flash':
                self.model = nemo_asr.EncDecMultiTaskModel.from_pretrained(
                    args.model_name).to(device='cpu')
            else:
                raise UnsupportedModel(model_name=args.model_name)
        self.model.freeze()
        self.model_config = OmegaConf.to_container(self.model.cfg)

        self.config = self.make_config()

    def __str__(self):
        return "trtllm.canary"

    @staticmethod
    def is_supported(model_type: str):
        supported_models = [
            'nvidia/canary-1b', 'nvidia/canary-1b-flash', 'nemo.canary',
            'nemo/canary', 'canary'
        ]

        if model_type in supported_models:
            return True

        return False

    def export_encoder(self):
        encoder_path = f"{self.args.output_dir}/encoder"
        preprocessor_path = f"{self.args.engine_dir}/preprocessor"
        encoder_config = self.model_config['encoder']
        encoder_config['dtype'] = self.dtype

        try:
            autocast = torch.cuda.amp.autocast(enabled=True,
                                               cache_enabled=False,
                                               dtype=torch.float32)
            with autocast, torch.no_grad(), torch.inference_mode():
                logging.info(f"Exporting model {self.model.__class__.__name__}")



                encoder_filename = 'encoder.onnx'
                tmp_encoder_path = f"{encoder_path}.tmp"

                os.makedirs(tmp_encoder_path, exist_ok=True)
                tmp_export_file = os.path.join(tmp_encoder_path, "encoder.onnx")

                self.model.encoder.export(tmp_export_file, onnx_opset_version=17)
                save_as_external_data = len(os.listdir(tmp_encoder_path)) > 1
                print(f"Loading encoder from {encoder_path} for GS")
                enc_graph = gs.import_onnx(onnx.load(tmp_export_file))
                enc_outputs = enc_graph.outputs[0]
                enc_len = enc_graph.outputs[1]
                Y = gs.Variable(name='encoded_outputs', dtype=np.float32, shape=(None, None, 1024))

                if isinstance(self.model.encoder_decoder_proj, torch.nn.Linear):
                    proj_w=self.model.encoder_decoder_proj.weight.data.clone()
                    proj_b=self.model.encoder_decoder_proj.bias.data.clone()
                    x = gs.Variable(name='x')
                    w = gs.Constant(name='w', values=proj_w.transpose(1, 0).cpu().numpy())
                    b = gs.Constant(name='b', values=proj_b.cpu().numpy())
                    mul_out = gs.Variable(name="mul_out")
                    enc_graph.nodes.append(
                        gs.Node(op="Transpose", inputs=[enc_outputs], outputs=[x], attrs={"perm": [0, 2, 1]}))
                    enc_graph.nodes.append(gs.Node(op="MatMul", inputs=[x, w], outputs=[mul_out]))
                    enc_graph.nodes.append(gs.Node(op="Add", inputs=[mul_out, b], outputs=[Y]))

                elif isinstance(self.model.encoder_decoder_proj, torch.nn.Identity):
                    enc_graph.nodes.append(
                        gs.Node(op="Transpose", inputs=[enc_outputs], outputs=[Y], attrs={"perm": [0, 2, 1]}))
                else:
                    raise AssertionError(f"Projection layer {type(self.model.encoder_decoder_proj)} is not supported.")

                enc_graph.outputs = [Y, enc_len]
                print(f"exporting encoder from  GS")

                model = gs.export_onnx(enc_graph)
                print(f"Saving encoder from  GS")
    
                os.makedirs(encoder_path, exist_ok=True)
                export_file = os.path.join(encoder_path, "encoder.onnx")

                onnx.save(model, export_file, save_as_external_data=save_as_external_data)
                shutil.rmtree(tmp_encoder_path)


                with open(os.path.join(encoder_path, "config.json"),
                          'w') as encoder_config_file:
                    json.dump(encoder_config, encoder_config_file)

                mel_basis_file = os.path.join(preprocessor_path, "mel_basis.pt")
                os.makedirs(preprocessor_path, exist_ok=True)

                torch.save(self.model.preprocessor.featurizer.filter_banks,
                           mel_basis_file)


                with open(os.path.join(preprocessor_path, "config.json"),
                          'w') as feat_config:
                    json.dump(self.model_config['preprocessor'], feat_config)

        except Exception as e:
            raise e

        return export_file, encoder_filename

    def make_config(self):
        keys_required = [
            'beam_search',
            'encoder',
            'head',
            'model_defaults',
            'prompt_format',
            'sample_rate',
            'target',
            'preprocessor',
        ]

        if 'beam_search' not in self.model_config and 'decoding' in self.model_config:
            self.model_config['beam_search'] = self.model_config[
                'decoding'].get('beam', {
                    'beam_size': 1,
                    'len_pen': 0.0,
                    'max_generation_delta': 50
                })

        enc_dec_hidden_size = self.model_config['model_defaults'][
            'asr_enc_hidden']
        if type(self.model.encoder_decoder_proj
                ) == torch.nn.modules.linear.Linear:
            if self.model.encoder_decoder_proj.out_features != self.model.encoder_decoder_proj.in_features:
                enc_dec_hidden_size = self.model.encoder_decoder_proj.out_features

        model_metadata = {
            "decoder_layers":
            self.model_config['transf_decoder']['config_dict']
            ['num_layers'],  # 24,
            "num_attention_heads":
            self.model_config['transf_decoder']['config_dict']
            ['num_attention_heads'],  # 8,
            "hidden_size":
            self.model_config['transf_decoder']['config_dict']
            ['hidden_size'],  # 1024,
            "vocab_size":
            self.model_config['head']['num_classes'],
            "max_sequence_length":
            self.model_config['transf_decoder']['config_dict']
            ['max_sequence_length'],  # 512,
            'hidden_act':
            self.model_config['transf_decoder']['config_dict']['hidden_act'],
            'max_position_embeddings':
            self.model_config['encoder']['pos_emb_max_len'],
            'ff_expansion_factor':
            self.model_config['encoder']['ff_expansion_factor'],
            'd_model':
            self.model_config['encoder']['d_model'],
            'enc_hidden_size':
            enc_dec_hidden_size,
            'enc_heads':
            self.model_config['encoder']['n_heads'],
            'vocab':
            self.export_vocab(),
            'prompt_format':
            self.model_config['prompt_format'],
        }

        return model_metadata

    def convert_decoder(self):
        self.model.transf_decoder.freeze()


        try:
            weights = {}
            self.model.transf_decoder.to(dtype=TORCH_DTYPES[self.dtype])
            model_params = self.model.transf_decoder.state_dict()
            lm_head = self.model.log_softmax.state_dict()

            assert torch.equal(
                lm_head['mlp.layer0.weight'],
                model_params['_embedding.token_embedding.weight'])

            weights['embedding.vocab_embedding.weight'] = model_params[
                '_embedding.token_embedding.weight'].contiguous().clone()
            weights['lm_head.weight'] = lm_head['mlp.layer0.weight'].contiguous(
            )
            weights['lm_head.bias'] = lm_head['mlp.layer0.bias'].contiguous()
            weights['embedding.position_embedding.weight'] = model_params[
                '_embedding.position_embedding.pos_enc'].contiguous()
            weights['embedding.embedding_layernorm.weight'] = model_params[
                '_embedding.layer_norm.weight'].contiguous()
            weights['embedding.embedding_layernorm.bias'] = model_params[
                '_embedding.layer_norm.bias'].contiguous()

            for i in range(self.config['decoder_layers']):
                trtllm_layer_name_prefix = f'decoder_layers.{i}'
                #layer_norm_1 aka self_attention_layernorm
                weights[f'{trtllm_layer_name_prefix}.self_attention_layernorm.weight'] =  \
                    model_params[f'_decoder.layers.{i}.layer_norm_1.weight'].contiguous()
                weights[
                    f'{trtllm_layer_name_prefix}.self_attention_layernorm.bias'] = \
                    model_params[f'_decoder.layers.{i}.layer_norm_1.bias'].contiguous()

                #first_sub_layer
                t = torch.cat(
                    [
                        model_params[
                            f'_decoder.layers.{i}.first_sub_layer.query_net.weight'],
                        model_params[
                            f'_decoder.layers.{i}.first_sub_layer.key_net.weight'],
                        model_params[
                            f'_decoder.layers.{i}.first_sub_layer.value_net.weight'],
                    ],
                    dim=0,
                ).contiguous()
                dst = weights[
                    f'{trtllm_layer_name_prefix}.self_attention.qkv.weight'] = t
                t = model_params[
                    f'_decoder.layers.{i}.first_sub_layer.out_projection.weight'].contiguous(
                    )
                dst = weights[
                    f'{trtllm_layer_name_prefix}.self_attention.dense.weight'] = t

                weights[f'{trtllm_layer_name_prefix}.self_attention.qkv.bias'] = torch.cat(
                    [
                        model_params[
                            f'_decoder.layers.{i}.first_sub_layer.query_net.bias'],
                        model_params[
                            f'_decoder.layers.{i}.first_sub_layer.key_net.bias'],
                        model_params[
                            f'_decoder.layers.{i}.first_sub_layer.value_net.bias'],
                    ],
                    dim=0).contiguous()

                weights[f'{trtllm_layer_name_prefix}.self_attention.dense.bias'] =  \
                    model_params[f'_decoder.layers.{i}.first_sub_layer.out_projection.bias'].contiguous()

                #layer_norm_2 aka cross_attention_layernorm
                weights[f'{trtllm_layer_name_prefix}.cross_attention_layernorm.weight'] = \
                    model_params[f'_decoder.layers.{i}.layer_norm_2.weight'].contiguous()
                weights[
                    f'{trtllm_layer_name_prefix}.cross_attention_layernorm.bias'] = \
                    model_params[f'_decoder.layers.{i}.layer_norm_2.bias'].contiguous()

                #second_sub_layer
                t = torch.cat(
                    [
                        model_params[
                            f'_decoder.layers.{i}.second_sub_layer.query_net.weight'],
                        model_params[
                            f'_decoder.layers.{i}.second_sub_layer.key_net.weight'],
                        model_params[
                            f'_decoder.layers.{i}.second_sub_layer.value_net.weight'],
                    ],
                    dim=0,
                ).contiguous()

                dst = weights[
                    f'{trtllm_layer_name_prefix}.cross_attention.qkv.weight'] = t

                t = model_params[
                    f'_decoder.layers.{i}.second_sub_layer.out_projection.weight'].contiguous(
                    )

                dst = weights[
                    f'{trtllm_layer_name_prefix}.cross_attention.dense.weight'] = t

                cross_attn_qkv_bias = torch.cat([
                    model_params[
                        f'_decoder.layers.{i}.second_sub_layer.query_net.bias'],
                    model_params[
                        f'_decoder.layers.{i}.second_sub_layer.key_net.bias'],
                    model_params[
                        f'_decoder.layers.{i}.second_sub_layer.value_net.bias'],
                ],
                                                dim=0).contiguous()
                weights[
                    f'{trtllm_layer_name_prefix}.cross_attention.qkv.bias'] = cross_attn_qkv_bias
                weights[f'{trtllm_layer_name_prefix}.cross_attention.dense.bias'] = \
                    model_params[f'_decoder.layers.{i}.second_sub_layer.out_projection.bias'].contiguous()

                #layer_norm_3
                weights[f'{trtllm_layer_name_prefix}.mlp_layernorm.weight'] = \
                    model_params[f'_decoder.layers.{i}.layer_norm_3.weight'].contiguous()
                weights[f'{trtllm_layer_name_prefix}.mlp_layernorm.bias'] = \
                    model_params[f'_decoder.layers.{i}.layer_norm_3.bias'].contiguous()

                #third_sub_layer
                t = model_params[
                    f'_decoder.layers.{i}.third_sub_layer.dense_in.weight'].contiguous(
                    )
                weights[f'{trtllm_layer_name_prefix}.mlp.fc.weight'] = t
                t = model_params[
                    f'_decoder.layers.{i}.third_sub_layer.dense_out.weight'].contiguous(
                    )
                weights[f'{trtllm_layer_name_prefix}.mlp.proj.weight'] = t

                weights[f'{trtllm_layer_name_prefix}.mlp.fc.bias'] = \
                    model_params[f'_decoder.layers.{i}.third_sub_layer.dense_in.bias'].contiguous()
                weights[f'{trtllm_layer_name_prefix}.mlp.proj.bias'] = \
                    model_params[f'_decoder.layers.{i}.third_sub_layer.dense_out.bias'].contiguous()

            weights['final_layernorm.weight'] = model_params[
                '_decoder.final_layer_norm.weight'].contiguous()
            weights['final_layernorm.bias'] = model_params[
                '_decoder.final_layer_norm.bias'].contiguous()

        except Exception as e:
            raise e
        component_save_dir = os.path.join(args.output_dir, "decoder")
        vocab_dir = os.path.join(args.engine_dir, 'decoder')
        os.makedirs(component_save_dir, exist_ok=True)
        os.makedirs(vocab_dir, exist_ok=True)

        # weights = weight_only_quantize_dict(weights,quant_algo=self.quant_algo, plugin=True)

        save_file(weights, os.path.join(component_save_dir,
                                        f'rank0.safetensors'))

        with open(os.path.join(component_save_dir, 'config.json'), 'w') as f:
            json.dump(self.get_decoder_config(), f, indent=4)
        with open(os.path.join(vocab_dir, 'vocab.json'), 'w') as f:
            json.dump(self.config['vocab'], f, indent=4)

    def export_vocab(self):

        tokenizer_vocab = {
            'tokens': {},
            'offsets': self.model.tokenizer.token_id_offset
        }
        for lang in self.model.tokenizer.langs:
            tokenizer_vocab['tokens'][lang] = {}
        tokenizer_vocab['size'] = self.model.tokenizer.vocab_size

        try:
            tokenizer_vocab['bos_id'] = self.model.tokenizer.bos_id
        except Exception:
            logging.warning(
                f"Tokenizer is missing bos_id. Could affect accuracy")

        try:
            tokenizer_vocab['eos_id'] = self.model.tokenizer.eos_id
        except Exception:
            logging.warning(
                f"Tokenizer is missing eos_id. Could affect accuracy")
        try:
            tokenizer_vocab['nospeech_id'] = self.model.tokenizer.nospeech_id
        except Exception:
            logging.warning(
                f"Tokenizer is missing nospeech_id. Could affect accuracy")
        try:
            tokenizer_vocab['pad_id'] = self.model.tokenizer.pad_id
        except Exception:
            logging.warning(
                f"Tokenizer is missing pad_id. Could affect accuracy")

        for t_id in range(0, self.model.tokenizer.vocab_size):
            lang = self.model.tokenizer.ids_to_lang([t_id])
            tokenizer_vocab['tokens'][lang][
                t_id] = self.model.tokenizer.ids_to_tokens([t_id])[0]

        return tokenizer_vocab

    def get_decoder_config(self):
        return {
            'architecture':
            "DecoderModel",
            'dtype':
            self.dtype,
            'logits_dtype':
            self.logits_dtype,
            'num_hidden_layers':
            self.config['decoder_layers'],
            'num_attention_heads':
            self.config['num_attention_heads'],
            'hidden_size':
            self.config['hidden_size'],
            'norm_epsilon':
            1e-5,
            'vocab_size':
            self.config['vocab_size'],
            'hidden_act':
            self.config['hidden_act'],
            'use_parallel_embedding':
            False,
            'embedding_sharding_dim':
            0,
            'max_position_embeddings':
            self.config['max_sequence_length'],
            'use_prompt_tuning':
            False,
            'prompt_format':
            self.config['prompt_format'],
            'head_size':
            self.config['hidden_size'] // self.config['num_attention_heads'],
            'has_position_embedding':
            True,
            'layernorm_type':
            LayerNormType.LayerNorm,
            'layernorm_position':
            LayerNormPositionType.pre_layernorm,
            'has_attention_qkvo_bias':
            True,
            'has_mlp_bias':
            True,
            'has_lm_head_bias':
            True,
            'has_model_final_layernorm':
            True,
            'has_embedding_layernorm':
            True,
            'has_embedding_scale':
            False,
            'ffn_hidden_size':
            self.config['ff_expansion_factor'] * self.config['d_model'],
            'q_scaling':
            1.0,
            'relative_attention':
            False,
            'max_distance':
            0,
            'num_buckets':
            0,  #1 in riva implementation
            'model_type':
            'canary',
            'rescale_before_lm_head':
            False,
            'encoder_hidden_size':
            self.config['enc_hidden_size'],
            'encoder_num_heads':
            self.config['enc_heads'],
            'encoder_head_size':
            None,
            'skip_cross_qkv':
            False,
            'quantization': {
                'quant_algo': self.quant_algo
            },
        }


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

    canary_model = CanaryModel(args)

    print("Converting encoder checkpoints...")
    canary_model.export_encoder()
    canary_model.convert_decoder()

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')

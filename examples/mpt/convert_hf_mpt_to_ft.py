# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2022, MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert MPT model checkpoint to FT format.

It's a modified version of
https://github.com/NVIDIA/FasterTransformer/blob/main/examples/pytorch/gpt/utils/huggingface_gpt_convert.py
"""

import argparse
import configparser
import os
from typing import Any, Dict, List

import numpy as np
import torch
import transformers

from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy

# numpy doesn't know bfloat16, define abstract binary type instead
np_bfloat16 = np.dtype('V2', metadata={"dtype": "bfloat16"})


def write_zero_bias(weight_name: str, weight_file_path: str,
                    bias_shape: List[int], data_type: torch.dtype) -> None:
    """Write zeros for bias.

    MPT model might not have bias while FT expects bias.

    Args:
        weight_name (str): Name of the weight tensor.
        weight_file_path (str): Output path for storing the weight (NOT zero bias).
        bias_shape (List[int]): Shape of the bias array.
    """
    if 'weight' not in weight_file_path:
        raise RuntimeError(
            f'Cannot write zero bias for {weight_name}. Input is not a weight tensor'
        )
    print(f'zero bias for weight: {weight_name}')
    bias_file_path = weight_file_path.replace('.weight', '.bias')
    bias = torch_to_numpy(torch.zeros(bias_shape, dtype=data_type))
    bias.tofile(bias_file_path)


def convert_weight_to_ft_each(out_dir: str, tensor_parallelism: int,
                              tensor_name: str, config: Dict[str, Any],
                              data: np.ndarray, data_type: torch.dtype):
    """Convert an MPT checkpoint to a FasterTransformer compatible format.

    Args:
        out_dir (str): Path of the directory to save the weight in FT format. The directory must already exist.
        tensor_parallelism (int): The number of gpus you are planning to use for inference.
        tensor_name (str): Name of the weight tensor. Used in naming the output file.
        config (Dict[str, Any]): Configuration for the model. This is used in getting model specific parameters.
        data (np.ndarray): Tensor data in np.ndarray format.

    Returns:
        None: Writes to a file in `out_dir`. File name is based on the `tensor_name`
    """
    if tensor_name.find('input_layernorm.weight') != -1 or tensor_name.find('input_layernorm.bias') != -1 or \
        tensor_name.find('attention.dense.bias') != -1 or tensor_name.find('post_attention_layernorm.weight') != -1 or \
        tensor_name.find('post_attention_layernorm.bias') != -1 or tensor_name.find('mlp.dense_4h_to_h.bias') != -1 or \
        tensor_name.find('final_layernorm.weight') != -1 or tensor_name.find('final_layernorm.bias') != -1:

        save_path = os.path.join(out_dir, f'model.{tensor_name}.bin')
        data.tofile(save_path)
        if 'weight' in tensor_name and config['no_bias']:
            write_zero_bias(tensor_name, save_path, data.shape[-1], data_type)

    elif tensor_name.find('attention.dense.weight') != -1:
        assert data.shape == (
            config['d_model'],
            config['d_model']), f'unexpected dim for {tensor_name}'
        # nn.Linear weights are transposed
        data = data.T
        split_vals = np.split(data, tensor_parallelism, axis=0)
        for j in range(tensor_parallelism):
            save_path = os.path.join(out_dir, f'model.{tensor_name}.{j}.bin')
            split_vals[j].tofile(save_path)
        if config['no_bias']:
            fake_weight_path = os.path.join(out_dir, f'model.{tensor_name}.bin')
            write_zero_bias(tensor_name, fake_weight_path, data.shape[-1],
                            data_type)

    elif tensor_name.find('mlp.dense_4h_to_h.weight') != -1:
        assert data.shape == (
            config['d_model'], config['expansion_ratio'] *
            config['d_model']), f'unexpected dim for {tensor_name}'
        # nn.Linear weights are transposed
        data = data.T
        split_vals = np.split(data, tensor_parallelism, axis=0)
        for j in range(tensor_parallelism):
            save_path = os.path.join(out_dir, f'model.{tensor_name}.{j}.bin')
            split_vals[j].tofile(save_path)
        if config['no_bias']:
            fake_weight_path = os.path.join(out_dir, f'model.{tensor_name}.bin')
            write_zero_bias(tensor_name, fake_weight_path, data.shape[-1],
                            data_type)

    elif tensor_name.find('mlp.dense_h_to_4h.weight') != -1:
        assert data.shape == (
            config['expansion_ratio'] * config['d_model'],
            config['d_model']), f'unexpected dim for {tensor_name}'
        # nn.Linear weights are transposed
        data = data.T

        split_vals = np.split(data, tensor_parallelism, axis=-1)
        for j in range(tensor_parallelism):
            save_path = os.path.join(out_dir, f'model.{tensor_name}.{j}.bin')
            split_vals[j].tofile(save_path)
            if config['no_bias']:
                write_zero_bias(tensor_name, save_path, split_vals[j].shape[-1],
                                data_type)

    elif tensor_name.find('mlp.dense_h_to_4h.bias') != -1:
        assert data.shape == (
            config['expansion_ratio'] *
            config['d_model'], ), f'unexpected dim for {tensor_name}'
        split_vals = np.split(data, tensor_parallelism, axis=-1)
        for j in range(tensor_parallelism):
            save_path = os.path.join(out_dir + f'model.{tensor_name}.{j}.bin')
            split_vals[j].tofile(save_path)

    elif tensor_name.find('attention.query_key_value.bias') != -1:
        assert data.shape == (
            3 * config['d_model'], ), f'unexpected dim for {tensor_name}'

        data = data.reshape(3, config['d_model'])

        split_vals = np.split(data, tensor_parallelism, axis=-1)

        for j in range(tensor_parallelism):
            save_path = os.path.join(out_dir, f'model.{tensor_name}.{j}.bin')
            split_vals[j].tofile(save_path)

    elif tensor_name.find('attention.query_key_value.weight') != -1:
        assert data.shape == (
            3 * config['d_model'],
            config['d_model']), f'unexpected dim for {tensor_name}'
        # nn.Linear weights are transposed
        data = data.T

        data = data.reshape(config['d_model'], 3, config['d_model'])
        split_vals = np.split(data, tensor_parallelism, axis=-1)

        for j in range(tensor_parallelism):
            save_path = os.path.join(out_dir, f'model.{tensor_name}.{j}.bin')
            split_vals[j].tofile(save_path)
            if config['no_bias']:
                write_zero_bias(tensor_name, save_path,
                                (3, split_vals[j].shape[-1]), data_type)

    else:
        raise RuntimeError(f'Tensor with name {tensor_name} is not handled')


def convert_mpt_to_ft(model_name_or_path: str,
                      output_dir: str,
                      tensor_parallelism: int = 1,
                      data_type: str = 'float16',
                      force: bool = False) -> None:
    """Convert an MPT checkpoint to a FasterTransformer compatible format.

    Args:
        model_name_or_path (str): The HF hub name of the model (e.g., mosaicml/mpt-7b) or the path of a directory
            containing an MPT checkpoint in a local dir.
        output_dir (str): Path of the directory to save the checkpoint in FT format. The directory must not already exist.
        tensor_parallelism (int): The number of gpus you are planning to use for inference.
        data_type (str): Data type of the weights in the input checkpoint.
        force (bool): force conversion even with unsupported features in FT.
    """
    out_dir = os.path.join(output_dir, f'{tensor_parallelism}-gpu')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # do conversion on cpu
    torch_device = 'cpu'

    torch_data_type = str_dtype_to_torch(data_type)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path, trust_remote_code=True,
        torch_dtype=torch_data_type).to(torch_device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True)

    hf_config = vars(model.config)
    print(hf_config)

    config = configparser.ConfigParser()
    config['gpt'] = {}
    try:
        config['gpt']['model_name'] = 'mpt' if hf_config[
            '_name_or_path'] == '' else hf_config['_name_or_path']
        config['gpt']['n_head'] = str(hf_config['n_heads'])
        n_embd = hf_config['d_model']
        config['gpt']['n_embd'] = str(n_embd)
        # config['gpt']['size_per_head'] = str(n_embd // hf_config['n_heads'])
        config['gpt']['n_inner'] = str(n_embd * hf_config['expansion_ratio'])
        config['gpt']['n_positions'] = str(hf_config['max_seq_len'])
        config['gpt']['n_layer'] = str(hf_config['n_layers'])
        config['gpt']['vocab_size'] = str(hf_config['vocab_size'])
        config['gpt']['bos_token_id'] = str(
            hf_config['bos_token_id']
        ) if hf_config['bos_token_id'] != None else str(tokenizer.bos_token_id)
        config['gpt']['eos_token_id'] = str(
            hf_config['eos_token_id']
        ) if hf_config['eos_token_id'] != None else str(tokenizer.eos_token_id)
        config['gpt']['storage_dtype'] = data_type  # == 'fp32' else 'float16'
        config['gpt']['tensor_parallelism'] = str(tensor_parallelism)
        config['gpt']['activation_function'] = str("gelu")
        config['gpt']['calibrate_kv_cache'] = 'False'
        config['gpt']['smoothquant'] = 'None'
        # nn.LayerNorm default eps is 1e-5
        config['gpt']['layer_norm_epsilon'] = str(1e-5)
        if hf_config['attn_config']['alibi']:
            config['gpt']['position_embedding_type'] = str("alibi")
            # config['gpt']['has_positional_encoding'] = str(False)
            # config['gpt']['use_attention_linear_bias'] = str(True)
        if hf_config['attn_config']['clip_qkv'] and not force:
            raise RuntimeError(
                'clip_qkv is enabled for this MPT model. This may not work as expected in FT. Use --force to force a conversion.'
            )
        if hf_config['attn_config']['qk_ln'] and not force:
            raise RuntimeError(
                'qk_ln is enabled for this MPT model. This may not work as expected in FT. Use --force to force a conversion.'
            )

        with open(os.path.join(out_dir, 'config.ini'), 'w') as configfile:
            config.write(configfile)
    except:
        print(f'Failed to save the config in config.ini.')
        raise

    param_remapping = {
        'norm_1.bias': 'input_layernorm.bias',
        'norm_1.weight': 'input_layernorm.weight',
        'attn.Wqkv.bias': 'attention.query_key_value.bias',
        'attn.Wqkv.weight': 'attention.query_key_value.weight',
        'attn.out_proj.bias': 'attention.dense.bias',
        'attn.out_proj.weight': 'attention.dense.weight',
        'norm_2.bias': 'post_attention_layernorm.bias',
        'norm_2.weight': 'post_attention_layernorm.weight',
        'ffn.up_proj.bias': 'mlp.dense_h_to_4h.bias',
        'ffn.up_proj.weight': 'mlp.dense_h_to_4h.weight',
        'ffn.down_proj.bias': 'mlp.dense_4h_to_h.bias',
        'ffn.down_proj.weight': 'mlp.dense_4h_to_h.weight',
    }

    for name, param in model.named_parameters():
        print(f'Working on parameter {name} ...')
        data = torch_to_numpy(param.to(torch_data_type).detach().cpu(
        ))  #param.detach().cpu().numpy().astype(np_weight_data_type)
        if name.find('weight') == -1 and name.find('bias') == -1:
            print(f'found a parameter name that is not handled: {name}')
            continue
        if name == 'transformer.wpe.weight':
            assert data.shape == (
                hf_config['max_seq_len'],
                hf_config['d_model']), f'unexpected dim for {name}'
            data.tofile(os.path.join(out_dir, 'model.wpe.bin'))
        elif name == 'transformer.wte.weight':
            assert data.shape == (
                hf_config['vocab_size'],
                hf_config['d_model']), f'unexpected dim for {name}'
            data.tofile(os.path.join(out_dir, 'model.wte.bin'))
        elif name == 'transformer.norm_f.bias':
            assert data.shape == (
                hf_config['d_model'], ), f'unexpected dim for {name}'
            data.tofile(os.path.join(out_dir, 'model.final_layernorm.bias.bin'))
        elif name == 'transformer.norm_f.weight':
            assert data.shape == (
                hf_config['d_model'], ), f'unexpected dim for {name}'
            save_path = os.path.join(out_dir,
                                     'model.final_layernorm.weight.bin')
            data.tofile(save_path)
            if hf_config['no_bias']:
                write_zero_bias(name, save_path, data.shape[-1],
                                torch_data_type)
        elif name == 'transformer.lm_head.weight':
            data.tofile(os.path.join(out_dir, 'model.lm_head.weight.bin'))
        else:
            for mpt_pattern, ft_pattern in param_remapping.items():
                if name.find(mpt_pattern) != -1:
                    new_name = name.replace('transformer.blocks.',
                                            'layers.').replace(
                                                mpt_pattern, ft_pattern)
                    convert_weight_to_ft_each(out_dir, tensor_parallelism,
                                              new_name, hf_config, data,
                                              torch_data_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--out_dir',
                        '-o',
                        type=str,
                        help='Directory to save converted checkpoint in',
                        required=True)
    parser.add_argument(
        '--in_file',
        '-i',
        type=str,
        help=
        'HF hub Model name (e.g., mosaicml/mpt-7b) or local dir path to load checkpoint from',
        required=True)
    parser.add_argument(
        '--tensor_parallelism',
        '-i_g',
        type=int,
        default=1,
        help='How many gpus for inference?',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help=
        'Force conversion to FT even if some features may not work as expected in FT'
    )
    parser.add_argument('--storage_type',
                        '-t',
                        type=str,
                        help='Data type of weights in the input checkpoint',
                        default='float16',
                        choices=['float32', 'float16', 'bfloat16'])

    args = parser.parse_args()
    print('\n=============== Argument ===============')
    for key in vars(args):
        print('{}: {}'.format(key, vars(args)[key]))
    print('========================================')

    convert_mpt_to_ft(args.in_file, args.out_dir, args.tensor_parallelism,
                      args.storage_type, args.force)

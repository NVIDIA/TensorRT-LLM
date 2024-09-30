import argparse
import copy
import json
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Union

import safetensors.torch
import torch
from transformers import AutoConfig, AutoModelForCausalLM

import tensorrt_llm
from tensorrt_llm import logger
from tensorrt_llm.models.convert_utils import (iterate_shard_files,
                                               load_state_dict)


class CheckpointType(str, Enum):
    mistral_inference = "mistral_inference"
    state_spaces = "state_spaces"
    hf = "hf"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_type",
                        type=CheckpointType,
                        choices=list(CheckpointType),
                        default=CheckpointType.hf,
                        help='Checkpoint type')
    parser.add_argument('--model_dir', type=Path, default=None)
    parser.add_argument("--world_size",
                        type=int,
                        default=1,
                        help="world size, only support tensor parallelism now")
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument(
        '--output_dir',
        type=Path,
        default='mamba_tllm_checkpoint',
        help='The path to save the mamba TensorRT-LLM checkpoint')
    parser.add_argument('--log_level', type=str, default='info')
    args = parser.parse_args()
    return args


def get_weight(config, prefix, dtype):
    return config[prefix + '.weight'].to(dtype).detach()


def get_bias(config, prefix, dtype):
    if (prefix + '.bias') in config:
        return config[prefix + '.bias'].to(dtype).detach()
    return None


def get_weight_and_bias(config, prefix, dtype_w, dtype_b):
    return get_weight(config, prefix,
                      dtype_w), get_bias(config, prefix, dtype_b)


def get_tllm_linear_weight(weight, prefix, bias=None):
    results = {}
    results[prefix + 'weight'] = weight.contiguous()
    if bias is not None:
        results[prefix + 'bias'] = bias
    return results


def split(v, tp_size, idx, dim=0):
    assert v.shape[dim] % tp_size == 0
    split_size = v.shape[dim] // tp_size
    if tp_size == 1:
        return v
    return torch.split(v, split_size, dim=dim)[idx]


def convert_hf_mamba(hf_mamba,
                     rank=0,
                     dtype='float32',
                     mamba_version: str = 'Mamba1'):
    weights = {}
    tik = time.time()

    model_params = dict(hf_mamba.named_parameters())
    dtype = getattr(torch, dtype)

    # Parameter names in mamba block
    for l in range(hf_mamba.config.num_hidden_layers):
        # ssm layer
        prefix = f'backbone.layers.{l}.mixer.'
        tllm_prex = f'backbone.layers.{l}.ssm.'
        for layer in ['conv1d', 'x_proj', 'dt_proj', 'out_proj']:
            dtype_b = torch.float32 if layer == 'dt_proj' else dtype
            weight, bias = get_weight_and_bias(model_params, prefix + layer,
                                               dtype, dtype_b)
            if layer == 'conv1d':
                weight = weight.unsqueeze(3)
            tllm_weight_name = tllm_prex + layer + '.weight'
            tllm_bias_name = tllm_prex + ('dt_bias' if layer == 'dt_proj' else
                                          layer + '.bias')
            weights[tllm_weight_name] = weight
            if bias is not None:
                weights[tllm_bias_name] = bias
        # in_proj
        weight, bias = get_weight_and_bias(model_params, prefix + 'in_proj',
                                           dtype, dtype)
        in_proj_weights = torch.split(weight, weight.size(0) // 2, dim=0)
        tllm_weight_name = tllm_prex + 'in_proj.weight'
        weights[tllm_weight_name.replace('proj', 'proj_x')] = in_proj_weights[0]
        weights[tllm_weight_name.replace('proj', 'proj_z')] = in_proj_weights[1]
        if bias is not None:
            in_proj_biases = torch.split(bias, bias.size(0) // 2, dim=0)
            tllm_bias_name = tllm_prex + 'in_proj.bias'
            weights[tllm_bias_name.replace('proj',
                                           'proj_x')] = in_proj_biases[0]
            weights[tllm_bias_name.replace('proj',
                                           'proj_x')] = in_proj_biases[1]

        # A and D
        Aparam = model_params[prefix + 'A_log'].float().detach()
        Aparam = Aparam.permute(1, 0).contiguous()
        weights[tllm_prex + 'A'] = -torch.exp(Aparam)
        weights[tllm_prex + 'D'] = model_params[prefix + 'D'].float().detach()
        # norm
        prefix = f'backbone.layers.{l}.norm'
        tllm_prex = f'backbone.layers.{l}.input_layernorm.'
        weight, bias = get_weight_and_bias(model_params, prefix, dtype, dtype)
        weights[tllm_prex + 'weight'] = weight
        if bias is not None:
            weights[tllm_prex + 'bias'] = bias

    # others
    for layer in ['backbone.embeddings', 'backbone.norm_f']:
        weight, bias = get_weight_and_bias(model_params, layer, dtype, dtype)
        layer = layer.replace('embeddings', 'vocab_embedding')
        layer = layer.replace('norm_f', 'ln_f')
        weights[layer + '.weight'] = weight
        if bias is not None:
            weights[layer + '.bias'] = bias
    weights['lm_head.weight'], _ = get_weight_and_bias(model_params,
                                                       'backbone.embeddings',
                                                       dtype, dtype)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights


def rename_hf_to_tllm(name: str):
    """ Rename a HF parameter name by the corresponding TRT-LLM style name. """
    # remove model
    if 'model.' in name:
        name = name.replace('model.', '')

    # change layer name
    if 'embeddings.' in name:
        name = name.replace('embeddings', 'vocab_embedding')
    elif 'embedding.' in name:
        name = name.replace('embedding', 'vocab_embedding')
    norm_pattern = r'\d\.norm\.'
    if 'mixer.' in name:
        name = name.replace('mixer.', 'ssm.')
    elif re.search(norm_pattern, name):
        name = name.replace('norm.', 'input_layernorm.')
    elif 'norm_f.' in name:
        name = name.replace('norm_f.', 'ln_f.')

    # Parameter names in ssm layers
    if 'A_log' in name:
        name = name.replace('A_log', 'A')
    elif 'dt_proj.bias' in name:
        name = name.replace('dt_proj.bias', 'dt_bias')
    return name


def convert_from_hf_checkpoint(mamba_config: dict,
                               model_dir: Union[str, Path],
                               rank=0,
                               dtype: Union[str, torch.dtype] = torch.float32,
                               mamba_version: str = 'Mamba1'):
    logger.info('Loading weights from HF Mamba...')
    tik = time.time()

    tp_rank = rank
    tp_size = mamba_config['mapping']['tp_size']
    d_inner = mamba_config['rnn_hidden_size']
    d_state = mamba_config['state_size']
    weights = {}
    if isinstance(dtype, str):
        dtype = tensorrt_llm.str_dtype_to_torch(dtype)

    for model_file in iterate_shard_files(model_dir, 0):
        logger.debug(f'Loading file {str(model_file)}...')
        model_params = load_state_dict(model_file, dtype=dtype)
        for name, param in model_params.items():
            logger.debug(f'Converting weight {name}...')
            tllm_name = rename_hf_to_tllm(name)
            param = param.detach().cpu()
            if 'A_log' in name:
                param = -torch.exp(param.float())
                if mamba_version == 'Mamba1':
                    param = param.permute(1, 0).contiguous()
            elif 'D' in name:
                param = param.float()
            elif 'dt_proj.bias' in name:
                param = param.float()
            elif 'dt_bias' in name:
                param = param.float()
            elif 'conv1d.weight' in name:
                param = param.unsqueeze(3)

            # split in_proj in Mamba1
            if 'in_proj' in name and mamba_version == 'Mamba1':
                in_proj_params = torch.split(param, param.size(0) // 2, dim=0)
                weights[tllm_name.replace('proj', 'proj_x')] = in_proj_params[0]
                weights[tllm_name.replace('proj', 'proj_z')] = in_proj_params[1]
            elif 'in_proj' in name and mamba_version == 'Mamba2':
                nheads = d_inner // mamba_config['rnn_head_size']
                ngroups = mamba_config['ngroups']
                in_proj_z, in_proj_x, in_proj_b, in_proj_c, in_proj_dt = torch.split(
                    param, [
                        d_inner, d_inner, ngroups * d_state, ngroups * d_state,
                        nheads
                    ],
                    dim=0)
                in_proj_z = split(in_proj_z, tp_size, tp_rank, dim=0)
                in_proj_x = split(in_proj_x, tp_size, tp_rank, dim=0)
                in_proj_b = split(in_proj_b, tp_size, tp_rank, dim=0)
                in_proj_c = split(in_proj_c, tp_size, tp_rank, dim=0)
                in_proj_dt = split(in_proj_dt, tp_size, tp_rank, dim=0)
                in_proj = torch.concat(
                    [in_proj_z, in_proj_x, in_proj_b, in_proj_c, in_proj_dt])
                weights[tllm_name] = in_proj.contiguous()
            elif 'conv1d' in name and mamba_version == 'Mamba2':
                ngroups = mamba_config['ngroups']
                conv_x, conv_b, conv_c = torch.split(
                    param, [d_inner, ngroups * d_state, ngroups * d_state],
                    dim=0)
                conv_x = split(conv_x, tp_size, tp_rank, dim=0)
                conv_b = split(conv_b, tp_size, tp_rank, dim=0)
                conv_c = split(conv_c, tp_size, tp_rank, dim=0)
                conv = torch.concat([conv_x, conv_b, conv_c])
                weights[tllm_name] = conv.contiguous()
            elif any(keyword in name for keyword in (
                    'mixer.norm.weight',
                    'A_log',
                    'D',
                    'dt_proj.bias',
                    'dt_bias',
            )) and mamba_version == 'Mamba2':
                weights[tllm_name] = split(param, tp_size, tp_rank, dim=0)
            elif 'out_proj' in name and mamba_version == 'Mamba2':
                weights[tllm_name] = split(param, tp_size, tp_rank,
                                           dim=1).contiguous()
            else:
                weights[tllm_name] = param
        del model_params

    # lm_head
    emb = weights['backbone.vocab_embedding.weight']
    if 'lm_head.weight' not in weights or weights['lm_head.weight'].data_ptr(
    ) == emb.data_ptr():
        weights['lm_head.weight'] = copy.deepcopy(emb)
    if mamba_version == 'Mamba2':
        weights['lm_head.weight'] = split(weights['lm_head.weight'],
                                          tp_size,
                                          tp_rank,
                                          dim=0)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
    return weights


def do_convert_from_ckpt(args):
    return args.model_dir.exists()


def convert(worker_rank, args, convert_args):
    convert_from_ckpt = do_convert_from_ckpt(args)
    for rank in range(worker_rank, args.world_size):
        if convert_from_ckpt:
            weights = convert_from_hf_checkpoint(rank=rank, **convert_args)
        else:
            weights = convert_hf_mamba(rank=rank, **convert_args)
        safetensors.torch.save_file(weights,
                                    args.output_dir / f'rank{rank}.safetensors')


@dataclass
class MambaConfig:

    architectures: List[str] = field(
        default_factory=lambda: ['MambaForCausalLM'])
    d_intermediate: int = 0
    vocab_size: int = 50277
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    pad_vocab_size_multiple: int = 8
    hidden_size: int = 2560
    num_hidden_layers: int = 64
    intermediate_size: int = 0
    state_size: int = 128
    conv_kernel: int = 4
    use_bias: bool = False
    head_dim: int = 64
    n_groups: int = 1
    chunk_size: int = 256
    ssm_rmsnorm: bool = True

    def update(self, data_dict):
        self.__dict__.update(data_dict)


def load_config_hf(model_name, ckpt_type):
    if ckpt_type == CheckpointType.hf:  # transformer compatible models
        hf_config = AutoConfig.from_pretrained(model_name,
                                               trust_remote_code=True)
        mamba_version = 'Mamba2' if hf_config.model_type == 'mamba2' else 'Mamba1'
    elif ckpt_type == CheckpointType.state_spaces:  # state-spaces/mamba models
        config = json.load(open(os.path.join(model_name, 'config.json')))
        ssm_cfg = config.pop('ssm_cfg')
        cfg_to_mamba_cfg = {
            'd_model': 'hidden_size',
            'n_layer': 'num_hidden_layers',
            'fused_add_norm': None,
            'tie_embeddings': None,
        }
        ssm_cfg_to_mamba_cfg = {
            'd_state': 'state_size',
            'd_conv': 'conv_kernel',
            'bias': 'use_bias',
            'headdim': 'head_dim',
            'ngroups': 'n_groups',
            'chunk_size': 'chunk_size',
            'rmsnorm': 'ssm_rmsnorm',
        }
        for k in cfg_to_mamba_cfg:
            if k in config:
                v = config.pop(k)
                if cfg_to_mamba_cfg[k] is not None:
                    config[cfg_to_mamba_cfg[k]] = v
        for k in ssm_cfg_to_mamba_cfg:
            if k in ssm_cfg and ssm_cfg_to_mamba_cfg[k] is not None:
                config[ssm_cfg_to_mamba_cfg[k]] = ssm_cfg[k]
        hf_config = MambaConfig(**config)
        if 'expand' in ssm_cfg:
            expand = ssm_cfg['expand']
            hf_config.intermediate_size = expand * hf_config.hidden_size
        else:
            hf_config.intermediate_size = 2 * hf_config.hidden_size
        mamba_version = ssm_cfg.pop("layer", "Mamba1")
    elif ckpt_type == CheckpointType.mistral_inference:  # mistral inference format
        config = json.load(open(os.path.join(model_name, 'params.json')))
        cfg_to_mamba_cfg = {
            'dim': 'hidden_size',
            'n_layers': 'num_hidden_layers',
            'n_groups': 'n_groups',
            'fused_add_norm': None,
            'tie_embeddings': None,
            'model_type': None,
        }
        for k in cfg_to_mamba_cfg:
            if k in config:
                v = config.pop(k)
                if cfg_to_mamba_cfg[k] is not None:
                    config[cfg_to_mamba_cfg[k]] = v
        hf_config = MambaConfig(**config)
        if 'expand' in config:
            expand = config['expand']
            hf_config.intermediate_size = expand * hf_config.hidden_size
        else:
            hf_config.intermediate_size = 2 * hf_config.hidden_size
        mamba_version = 'Mamba2'
    return hf_config, mamba_version


def main():
    print(tensorrt_llm.__version__)

    args = parse_arguments()
    logger.set_level(args.log_level)
    tik = time.time()

    args.output_dir.mkdir(exist_ok=True, parents=True)

    hf_config, mamba_version = load_config_hf(args.model_dir, args.ckpt_type)

    vocab_size = hf_config.vocab_size
    pad_vocab_size_multiple = getattr(hf_config, "pad_vocab_size_multiple", 1)
    if vocab_size % pad_vocab_size_multiple != 0:
        vocab_size += pad_vocab_size_multiple - (vocab_size %
                                                 pad_vocab_size_multiple)

    config = {
        'architecture': 'MambaForCausalLM',
        'dtype': args.dtype,
        'logits_dtype': 'float32',
        'hidden_size': hf_config.hidden_size,
        'num_hidden_layers': hf_config.num_hidden_layers,
        'layer_types': ['recurrent'],
        'vocab_size': vocab_size,
        'rms_norm': hf_config.rms_norm,
        'residual_in_fp32': hf_config.residual_in_fp32,
        'pad_vocab_size_multiple': pad_vocab_size_multiple,
        'hidden_act': 'silu',
        'num_attention_heads': args.world_size,
        'rnn_hidden_size': hf_config.intermediate_size,
        'rnn_conv_dim_size': hf_config.intermediate_size,
        'state_size': hf_config.state_size,
        'conv_kernel': hf_config.conv_kernel,
        'use_bias': hf_config.use_bias,
        'mamba_version': mamba_version,
        'mapping': {
            'world_size': args.world_size,
            'tp_size': args.world_size,
            'pp_size': 1
        },
    }
    if mamba_version == 'Mamba2':
        conv_dim = hf_config.intermediate_size + 2 * hf_config.n_groups * hf_config.state_size
        ssm_rmsnorm = getattr(hf_config, "ssm_rmsnorm", hf_config.rms_norm)
        mamba2_cfg = {
            'rnn_head_size': hf_config.head_dim,
            'rnn_conv_dim_size': conv_dim,
            'ngroups': hf_config.n_groups,
            'chunk_size': hf_config.chunk_size,
            'ssm_rmsnorm': ssm_rmsnorm,
        }
        config.update(mamba2_cfg)

    with (args.output_dir / 'config.json').open('w') as f:
        json.dump(config, f, indent=4)

    convert_from_ckpt = do_convert_from_ckpt(args)
    # TODO: Add convert_hf_mamba support for Mamba2 when transformers can support Mamba2 models
    assert convert_from_ckpt or mamba_version == 'Mamba2', "Mamba2 can only support convert from checkpoints."
    assert args.world_size == 1 or mamba_version == 'Mamba2', "Mamba1 can not support tensor parallelism."
    if not convert_from_ckpt:
        logger.info(f'Convert by using model')
        hf_mamba = AutoModelForCausalLM.from_pretrained(args.model_dir,
                                                        device_map="auto",
                                                        torch_dtype="auto",
                                                        trust_remote_code=True)
    else:
        logger.info(f'Convert by using checkpoint')
        hf_mamba = None

    convert_args = dict(dtype=args.dtype, )

    if convert_from_ckpt:
        convert_args['model_dir'] = args.model_dir
    else:
        convert_args['hf_mamba'] = hf_mamba
    convert_args['mamba_version'] = mamba_version
    convert_args['mamba_config'] = config

    convert(0, args, convert_args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')


if __name__ == '__main__':
    main()

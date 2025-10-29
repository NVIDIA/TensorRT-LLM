import argparse
import copy
import functools
import json
import os
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Tuple

import safetensors
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D

import tensorrt_llm
from tensorrt_llm._utils import release_gc
from tensorrt_llm.layers import MoeConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.convert_utils import (generate_int8,
                                               load_calib_dataset, split)
from tensorrt_llm.quantization import QuantAlgo


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--tp_size',
                        type=int,
                        default=1,
                        help='N-way tensor parallelism size')
    parser.add_argument('--pp_size',
                        type=int,
                        default=1,
                        help='N-way pipeline parallelism size')
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--logits_dtype',
                        type=str,
                        default='float32',
                        choices=['float16', 'float32'])
    parser.add_argument(
        '--per_channel',
        default=False,
        action="store_true",
        help=
        'By default, we use a single static scaling factor for the GEMM\'s result. '
        'per_channel instead uses a different static scaling factor for each channel. '
        'The latter is usually more accurate, but a little slower.')
    parser.add_argument(
        '--calib_dataset',
        type=str,
        default='ccdv/cnn_dailymail',
        help=
        "The huggingface dataset name or the local directory of the dataset for calibration."
    )
    parser.add_argument("--dataset_cache_dir",
                        type=str,
                        default=None,
                        help="cache dir to load the hugging face dataset")
    parser.add_argument(
        '--int8_kv_cache',
        default=False,
        action="store_true",
        help=
        'By default, we use dtype for KV cache. int8_kv_cache chooses int8 quantization for KV'
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
    parser.add_argument('--output_dir',
                        type=str,
                        default='tllm_checkpoint',
                        help='The path to save the TensorRT LLM checkpoint')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='The number of workers for converting checkpoint in parallel')
    parser.add_argument('--rotary_base', type=float, default=10000.0)
    parser.add_argument('--rotary_scaling', nargs=2, type=str, default=None)
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--n_positions', type=int, default=2048)
    parser.add_argument('--n_layer', type=int, default=32)
    parser.add_argument('--n_head', type=int, default=32)
    parser.add_argument('--n_kv_head', type=int, default=None)
    parser.add_argument('--n_embd', type=int, default=4096)
    parser.add_argument('--inter_size', type=int, default=11008)
    parser.add_argument('--max_seq_len', type=int, default=4096)
    parser.add_argument('--clip_qkv', type=int, default=None)
    parser.add_argument('--hidden_act',
                        type=str,
                        default='gelu',
                        help='Set to swiglu to use GLU in MoEs')
    parser.add_argument(
        '--moe_num_experts',
        default=0,
        type=int,
        help='Specify the number of experts to use for MOE layers')
    parser.add_argument(
        '--moe_top_k',
        default=0,
        type=int,
        help=
        'Specify the top_k value to use for MOE layers. Default to 1 if --moe_num_experts is set'
    )
    parser.add_argument(
        '--moe_tp_size',
        type=int,
        default=-1,
        help=
        'N-way tensor parallelism size for MOE, default is tp_size, which will do tp-only for MoE'
    )
    parser.add_argument(
        '--moe_ep_size',
        type=int,
        default=-1,
        help=
        'N-way expert parallelism size for MOE, default is 1, which will do tp-only for MoE'
    )
    parser.add_argument(
        '--moe_renorm_mode',
        default=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE,
        type=int,
        help=
        'Controls renormalization after gate logits. Check layers/moe.py for accepted values',
    )
    parser.add_argument(
        '--disable_weight_only_quant_plugin',
        default=False,
        action="store_true",
        help=
        'By default, using plugin implementation for weight quantization. Enabling disable_weight_only_quant_plugin flag will use ootb implementation instead of plugin.'
        'You must also use --use_weight_only for that argument to have an impact.'
    )
    parser.add_argument(
        '--dense_context_fmha',
        default=False,
        action='store_true',
        help=
        'Enable dense fmha in context phase, otherwise sliding window attention.'
        'If dense_context_fmha=False, the sliding window size is the max attention window size.'
    )
    parser.add_argument(
        '--use_parallel_embedding',
        action="store_true",
        default=False,
        help=
        'By default embedding parallelism is disabled. By setting this flag, embedding parallelism is enabled'
    )
    parser.add_argument(
        '--embedding_sharding_dim',
        type=int,
        default=0,
        choices=[0, 1],
        help=
        'By default the embedding lookup table is sharded along vocab dimension (embedding_sharding_dim=0). '
        'To shard it along hidden dimension, set embedding_sharding_dim=1'
        'Note: embedding sharing is only enabled when embedding_sharding_dim = 0'
    )
    parser.add_argument('--use_prompt_tuning',
                        action="store_true",
                        default=False)
    args = parser.parse_args()

    return args


def args_to_build_options(args):
    return {
        'use_parallel_embedding': args.use_parallel_embedding,
        'embedding_sharding_dim': args.embedding_sharding_dim,
        'disable_weight_only_quant_plugin':
        args.disable_weight_only_quant_plugin
    }


def get_weight(params: Dict[str, torch.Tensor], prefix: str,
               dtype: torch.dtype) -> torch.Tensor:
    if f'{prefix}' in params:
        return params[f'{prefix}'].to(dtype).detach().cpu()
    elif f'{prefix}.weight' not in params:
        return None
    return params[f'{prefix}.weight'].to(dtype).detach().cpu()


def get_bias(params: Dict[str, torch.Tensor], prefix: str,
             dtype: torch.dtype) -> torch.Tensor:
    if f'{prefix}.bias' not in params:
        return None
    return params[f'{prefix}.bias'].to(dtype).detach().cpu()


def get_weight_and_bias(params: Dict[str, torch.Tensor], prefix: str,
                        dtype: torch.dtype) -> Tuple[torch.Tensor]:
    return get_weight(params, prefix, dtype), get_bias(params, prefix, dtype)


@torch.no_grad()
def capture_activation_range(model,
                             tokenizer,
                             dataset,
                             num_samples=1,
                             seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    act_scales = defaultdict(lambda: {"x": None, "y": None, "w": None})

    tokenizer.pad_token = tokenizer.eos_token

    def stat_tensor(name, tensor, act_scales, key):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float()

        if act_scales[name][key] is None:
            act_scales[name][key] = comming_max
        else:
            act_scales[name][key] = torch.max(act_scales[name][key],
                                              comming_max)

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x, act_scales, "x")
        stat_tensor(name, y, act_scales, "y")

        if act_scales[name]["w"] is None:
            act_scales[name]["w"] = m.weight.abs().clip(
                1e-8, None).max(dim=1)[0].float()

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) or isinstance(m, Conv1D):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples), desc="calibrating model"):
        datapoint = dataset[i:i + 1]
        line = copy.copy(datapoint)
        line[0] = line[0] + ' TL;DR: '
        line[0] = line[0].strip()
        line[0] = line[0].replace(" n't", "n't")
        input_ids = tokenizer(line,
                              return_tensors="pt",
                              max_length=seq_len,
                              padding=True,
                              truncation=True).input_ids.to(device)
        model(input_ids)

    for h in hooks:
        h.remove()

    return act_scales


def split_qkv_tp(qkv, n_head, n_kv_heads, n_hidden, tensor_parallel, rank):
    """
    Splits the QKV matrix according to tensor parallelism
    """
    kv_head_size = n_kv_heads * (n_hidden // n_head)
    q, k, v = torch.split(qkv, [n_hidden, kv_head_size, kv_head_size], dim=0)
    q = split(q, tensor_parallel, rank, dim=0)
    k = split(k, tensor_parallel, rank, dim=0)
    v = split(v, tensor_parallel, rank, dim=0)
    return torch.concatenate([q, k, v], dim=0).contiguous()


def split_matrix(weight: torch.Tensor, tp_size: int, rank: int,
                 dim: int) -> torch.Tensor:
    return split(weight, tp_size, rank, dim=dim)


def get_tllm_linear_weight(
        weight: torch.Tensor,
        prefix: str,
        bias: Optional[torch.Tensor] = None,
        use_weight_only: bool = False,
        plugin_weight_only_quant_type: torch.dtype = torch.int8,
        postfix='weight',
        quant_scale_name=None) -> Dict[str, torch.Tensor]:
    results = {}
    if use_weight_only:
        if weight.dim() > 2:
            v = weight.transpose(1, 2).contiguous().clone()
        else:
            v = weight.t().contiguous().clone()
        processed_torch_weights, torch_weight_scales = \
            torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                v.cpu(), plugin_weight_only_quant_type)
        results[prefix + postfix] = processed_torch_weights
        if quant_scale_name is not None:
            results[quant_scale_name] = torch_weight_scales
        else:
            results[prefix + 'per_channel_scale'] = torch_weight_scales
    else:
        results[prefix + postfix] = weight.contiguous()

    if bias is not None:
        results[f'{prefix}bias'] = bias

    return results


def convert_hf_dbrx(model_params: dict,
                    hf_config: AutoConfig,
                    mapping: Mapping,
                    dtype: str = 'float32',
                    use_weight_only: bool = False,
                    plugin_weight_only_quant_type: torch.dtype = torch.int8,
                    moe_config: MoeConfig = None,
                    int8_kv_cache=False,
                    act_range=[]):

    weights = {}
    tik = time.time()

    dtype = getattr(torch, dtype)
    num_hidden_layers = hf_config.n_layers
    num_head = hf_config.n_heads
    num_kv_heads = hf_config.attn_config.kv_n_heads
    num_hidden = hf_config.d_model
    mlp_hidden_size = hf_config.ffn_config.ffn_hidden_size
    layers_range = mapping.pp_layers(num_hidden_layers)
    multi_query_mode = (num_kv_heads != num_head)

    for l in layers_range:
        prefix = f'transformer.blocks.{l}'
        tllm_prex = f'transformer.layers.{l-layers_range[0]}'
        # Attention QKV (no bias)
        qkv_w = get_weight(model_params, f'{prefix}.norm_attn_norm.attn.Wqkv',
                           dtype)
        qkv_w = split_qkv_tp(qkv_w, num_head, num_kv_heads, num_hidden,
                             mapping.tp_size, mapping.tp_rank)
        weights.update(
            get_tllm_linear_weight(qkv_w, f'{tllm_prex}.attention.qkv.', None,
                                   use_weight_only,
                                   plugin_weight_only_quant_type))
        # Attention dense (no bias)
        attn_dense_weight = get_weight(
            model_params, f'{prefix}.norm_attn_norm.attn.out_proj', dtype)
        attn_dense_w = split_matrix(attn_dense_weight,
                                    mapping.tp_size,
                                    mapping.tp_rank,
                                    dim=1)
        weights.update(
            get_tllm_linear_weight(attn_dense_w,
                                   f'{tllm_prex}.attention.dense.', None,
                                   use_weight_only,
                                   plugin_weight_only_quant_type))

        if int8_kv_cache:
            qkv_weight = get_weight(model_params,
                                    f'{prefix}.norm_attn_norm.attn.Wqkv', dtype)
            qkv_weight = qkv_weight.t()
            if not multi_query_mode:
                qkv_weight = qkv_weight.reshape(num_hidden, 3, num_hidden)
            int8_weights = generate_int8(
                qkv_weight,
                act_range.get(f'{prefix}.norm_attn_norm.attn.Wqkv'),
                is_qkv=True,
                multi_query_mode=multi_query_mode)
            weights[
                f'{tllm_prex}.attention.kv_cache_scaling_factor'] = int8_weights[
                    'scale_y_quant_orig'].contiguous()

        # input layer_norm
        input_ln_weight = get_weight(model_params,
                                     f'{prefix}.norm_attn_norm.norm_1', dtype)
        weights[f'{tllm_prex}.input_layernorm.weight'] = input_ln_weight

        # post layer_norm
        post_ln_weight = get_weight(model_params,
                                    f'{prefix}.norm_attn_norm.norm_2', dtype)
        weights[f'{tllm_prex}.post_layernorm.weight'] = post_ln_weight

        if moe_config and moe_config.has_moe():
            # experts mlp w1 -> mlp gate
            mlp_gate_weight = get_weight(model_params,
                                         f'{prefix}.ffn.experts.mlp.w1', dtype)
            mlp_gate_weight = mlp_gate_weight.reshape(-1, mlp_hidden_size,
                                                      num_hidden)
            # moe expert parallel
            mlp_gate_weight = split_matrix(mlp_gate_weight,
                                           mapping.moe_ep_size,
                                           mapping.moe_ep_rank,
                                           dim=0)
            # moe tensor parallel
            mlp_gate_w = split_matrix(mlp_gate_weight,
                                      mapping.moe_tp_size,
                                      mapping.moe_tp_rank,
                                      dim=1)

            # experts mlp v1 -> mlp fc
            mlp_fc_weight = get_weight(model_params,
                                       f'{prefix}.ffn.experts.mlp.v1', dtype)
            mlp_fc_weight = mlp_fc_weight.reshape(-1, mlp_hidden_size,
                                                  num_hidden)
            # moe expert parallel
            mlp_fc_weight = split_matrix(mlp_fc_weight,
                                         mapping.moe_ep_size,
                                         mapping.moe_ep_rank,
                                         dim=0)
            # moe tensor parallel
            mlp_fc_w = split_matrix(mlp_fc_weight,
                                    mapping.moe_tp_size,
                                    mapping.moe_tp_rank,
                                    dim=1)
            mlp_fc_w = torch.concat([mlp_fc_w, mlp_gate_w], dim=-2)
            weights.update(
                get_tllm_linear_weight(mlp_fc_w, f'{tllm_prex}.mlp.fc.', None,
                                       use_weight_only,
                                       plugin_weight_only_quant_type))

            # experts mlp w2 -> mlp proj
            mlp_proj_weight = get_weight(model_params,
                                         f'{prefix}.ffn.experts.mlp.w2', dtype)
            mlp_proj_weight = mlp_proj_weight.reshape(-1, mlp_hidden_size,
                                                      num_hidden).transpose(
                                                          1, 2)
            # moe expert parallel
            mlp_proj_weight = split_matrix(mlp_proj_weight,
                                           mapping.moe_ep_size,
                                           mapping.moe_ep_rank,
                                           dim=0)
            # moe tensor parallel
            mlp_proj_w = split_matrix(mlp_proj_weight,
                                      mapping.moe_tp_size,
                                      mapping.moe_tp_rank,
                                      dim=2)
            weights.update(
                get_tllm_linear_weight(mlp_proj_w, f'{tllm_prex}.mlp.proj.',
                                       None, use_weight_only,
                                       plugin_weight_only_quant_type))

            # router mlp
            router_weights = get_weight(model_params,
                                        f'{prefix}.ffn.router.layer',
                                        torch.float32)
            weights[f'{tllm_prex}.mlp.router.weight'] = router_weights

    embed_w = get_weight(model_params, 'transformer.wte', dtype)
    lm_head = get_weight(model_params, 'lm_head', dtype)
    if mapping.is_first_pp_rank():
        # Embedding
        weights['transformer.vocab_embedding.weight'] = embed_w
    if mapping.is_last_pp_rank():
        if lm_head is None:
            lm_head = embed_w.clone()
        ln_f_w = get_weight(model_params, 'transformer.norm_f', dtype)
        # ln_f weight and bias
        weights['transformer.ln_f.weight'] = ln_f_w
        weights['lm_head.weight'] = split_matrix(lm_head,
                                                 mapping.tp_size,
                                                 mapping.tp_rank,
                                                 dim=0)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights


def execute(workers, func, hf_model):
    if workers == 1:
        for rank, f in enumerate(func):
            f(hf_model, rank)
    else:
        with ThreadPoolExecutor(max_workers=workers) as p:
            futures = [
                p.submit(f, hf_model, rank) for rank, f in enumerate(func)
            ]
            exceptions = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    exceptions.append(e)
            assert len(
                exceptions
            ) == 0, "Checkpoint conversion failed, please check error log."


if __name__ == '__main__':
    print(tensorrt_llm.__version__)
    args = parse_arguments()
    world_size = args.tp_size * args.pp_size
    if (args.moe_tp_size == -1 and args.moe_ep_size == -1):
        # moe default to tp-only
        args.moe_tp_size = args.tp_size
        args.moe_ep_size = 1
    elif (args.moe_tp_size == -1):
        args.moe_tp_size = args.tp_size // args.moe_ep_size
    elif (args.moe_ep_size == -1):
        args.moe_ep_size = args.tp_size // args.moe_tp_size
    assert (args.moe_tp_size * args.moe_ep_size == args.tp_size
            ), "moe_tp_size * moe_ep_size must equal to tp_size"

    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    quant_algo = None
    kv_cache_quant_algo = None
    plugin_weight_only_quant_type = None
    if args.use_weight_only:
        if args.weight_only_precision == 'int8':
            plugin_weight_only_quant_type = torch.int8
            quant_algo = QuantAlgo.W8A16
        elif args.weight_only_precision == 'int4':
            plugin_weight_only_quant_type = torch.quint4x2
            quant_algo = QuantAlgo.W4A16

    if args.int8_kv_cache:
        kv_cache_quant_algo = QuantAlgo.INT8

    hf_config = None

    if args.model_dir is not None:
        hf_config = AutoConfig.from_pretrained(args.model_dir,
                                               trust_remote_code=True)
        args.n_kv_head = hf_config.attn_config.kv_n_heads
        args.n_layer = hf_config.n_layers
        args.n_head = hf_config.n_heads
        args.vocab_size = hf_config.vocab_size
        args.n_embd = hf_config.d_model
        args.inter_size = hf_config.ffn_config.ffn_hidden_size
        args.max_seq_len = hf_config.max_seq_len
        args.moe_num_experts = getattr(hf_config.ffn_config, "moe_num_experts",
                                       0)
        args.moe_top_k = getattr(hf_config.ffn_config, "moe_top_k", 0)
        if args.moe_num_experts and args.moe_top_k == 0:
            args.moe_top_k = 1
        args.clip_qkv = hf_config.attn_config.clip_qkv
        args.hidden_act = 'swiglu'
        args.rotary_base = hf_config.attn_config.rope_theta
    args.moe_config = MoeConfig(
        num_experts=args.moe_num_experts,
        top_k=args.moe_top_k,
        normalization_mode=args.moe_renorm_mode).validate()
    config = {
        'architecture': hf_config.architectures[0],
        'dtype': args.dtype,
        'logits_dtype': args.logits_dtype,
        'vocab_size': args.vocab_size,
        'hidden_size': args.n_embd,
        'intermediate_size': args.inter_size,
        'num_hidden_layers': args.n_layer,
        'num_attention_heads': args.n_head,
        'num_key_value_heads': args.n_kv_head,
        'max_position_embeddings': args.max_seq_len,
        'norm_epsilon': 1e-5,
        'position_embedding_type': 'rope_gpt_neox',
        'hidden_act': args.hidden_act,
        'rotary_base': args.rotary_base,
        'rotary_scaling': args.rotary_scaling,
        'quantization': {
            'quant_algo': quant_algo,
            'kv_cache_quant_algo': kv_cache_quant_algo,
        },
        'moe': {
            "num_experts": args.moe_num_experts,
            "top_k": args.moe_top_k,
            "normalization_mode": args.moe_renorm_mode
        },
        'mapping': {
            'world_size': world_size,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size,
            'moe_tp_size': args.moe_tp_size,
            'moe_ep_size': args.moe_ep_size,
        },
        'clip_qkv': args.clip_qkv,
        'dense_context_fmha': args.dense_context_fmha,
    }

    config.update(args_to_build_options(args))

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    def load_from_hf(model_dir):
        hf_model = AutoModelForCausalLM.from_pretrained(model_dir,
                                                        trust_remote_code=True,
                                                        device_map="auto",
                                                        dtype=getattr(
                                                            torch, args.dtype),
                                                        config=hf_config)
        return hf_model

    def convert_and_save(hf_model, rank):
        mapping = Mapping(world_size=world_size,
                          rank=rank,
                          tp_size=args.tp_size,
                          pp_size=args.pp_size,
                          moe_tp_size=args.moe_tp_size,
                          moe_ep_size=args.moe_ep_size)
        act_range = {}
        if args.int8_kv_cache:
            tokenizer = AutoTokenizer.from_pretrained(args.model_dir,
                                                      padding_side='left',
                                                      trust_remote_code=True)
            dataset = load_calib_dataset(args.calib_dataset,
                                         cache_dir=args.dataset_cache_dir)
            act_range = capture_activation_range(hf_model, tokenizer, dataset)

        hf_model = dict(hf_model.named_parameters())
        weights = convert_hf_dbrx(
            hf_model,
            hf_config,
            mapping,
            dtype=args.dtype,
            use_weight_only=args.use_weight_only,
            plugin_weight_only_quant_type=plugin_weight_only_quant_type,
            moe_config=args.moe_config,
            int8_kv_cache=args.int8_kv_cache,
            act_range=act_range)

        safetensors.torch.save_file(
            weights, os.path.join(args.output_dir, f'rank{rank}.safetensors'))
        del weights
        release_gc()

    if args.model_dir:
        hf_model = load_from_hf(args.model_dir)
        execute(args.workers, [convert_and_save] * world_size, hf_model)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')

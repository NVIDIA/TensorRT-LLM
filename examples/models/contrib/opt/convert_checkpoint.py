import argparse
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import safetensors
import torch
from transformers import AutoModelForCausalLM, Blip2ForConditionalGeneration

import tensorrt_llm
from tensorrt_llm._utils import pad_vocab_size
from tensorrt_llm.models.convert_utils import (get_weight, get_weight_and_bias,
                                               split, split_matrix_tp,
                                               split_qkv_bias_tp, split_qkv_tp)
from tensorrt_llm.quantization import QuantAlgo


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument(
        '--model_type',
        type=str,
        default='opt',
        choices=['opt', 'blip2'],
        help=
        'Multimodal type when this script is used for multimodal conversion.')
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
    parser.add_argument('--output_dir',
                        type=str,
                        default='tllm_checkpoint',
                        help='The path to save the TensorRT-LLM checkpoint')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='The number of workers for converting checkpoint in parallel')
    args = parser.parse_args()

    return args


def split_embedding(
    param: torch.Tensor,
    tp_size: int,
    tp_rank: int,
    use_parallel_embedding: bool = False,
    sharding_dim: int = 0,
) -> torch.Tensor:
    if param is None:
        return None
    if not use_parallel_embedding:
        return param

    vocab_size, hidden_size = param.size()
    if sharding_dim == 0:
        if vocab_size % tp_size != 0:
            vocab_size_padded = pad_vocab_size(vocab_size, tp_size)
            pad_width = vocab_size_padded - vocab_size
            param = torch.nn.functional.pad(param, (0, 0, 0, pad_width),
                                            value=0)
        else:
            assert hidden_size % tp_size == 0
    return split(param, tp_size, tp_rank, dim=sharding_dim)


def get_tllm_linear_weight(weight,
                           prefix,
                           bias=None,
                           use_weight_only=False,
                           plugin_weight_only_quant_type=torch.int8):
    results = {}
    if use_weight_only:
        v = weight.t().contiguous()
        processed_torch_weights, torch_weight_scales = \
            torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                v, plugin_weight_only_quant_type)
        results[prefix + 'weight'] = processed_torch_weights
        results[prefix + 'per_channel_scale'] = torch_weight_scales
    else:
        results[prefix + 'weight'] = weight.contiguous()

    if bias is not None:
        results[prefix + 'bias'] = bias

    return results


def convert_hf_opt(hf_model,
                   rank=0,
                   tensor_parallel=1,
                   dtype='float32',
                   use_parallel_embedding=False,
                   sharding_dim=0,
                   use_weight_only=False,
                   plugin_weight_only_quant_type=torch.int8):

    weights = {}
    tik = time.time()

    model_params = dict(hf_model.named_parameters())
    dtype = getattr(torch, dtype)
    do_layer_norm_before = hf_model.config.do_layer_norm_before
    num_attention_heads = hf_model.config.num_attention_heads
    hidden_size = hf_model.config.hidden_size

    for l in range(hf_model.config.num_hidden_layers):
        prefix = f'model.decoder.layers.{l}.'
        tllm_prex = f'transformer.layers.{l}.'

        q_weight, q_bias = get_weight_and_bias(model_params,
                                               prefix + 'self_attn.q_proj',
                                               dtype)
        k_weight, k_bias = get_weight_and_bias(model_params,
                                               prefix + 'self_attn.k_proj',
                                               dtype)
        v_weight, v_bias = get_weight_and_bias(model_params,
                                               prefix + 'self_attn.v_proj',
                                               dtype)
        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        split_v = split_qkv_tp(qkv_weight, num_attention_heads, hidden_size,
                               tensor_parallel, rank)
        qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
        bias = split_qkv_bias_tp(qkv_bias, num_attention_heads, hidden_size,
                                 tensor_parallel, rank)
        weights.update(
            get_tllm_linear_weight(split_v, tllm_prex + 'attention.qkv.', bias,
                                   use_weight_only,
                                   plugin_weight_only_quant_type))

        attn_dense_weight, attn_dense_bias = get_weight_and_bias(
            model_params, prefix + 'self_attn.out_proj', dtype)
        split_v = split_matrix_tp(attn_dense_weight,
                                  tensor_parallel,
                                  rank,
                                  dim=1)
        weights.update(
            get_tllm_linear_weight(split_v, tllm_prex + 'attention.dense.',
                                   attn_dense_bias, use_weight_only,
                                   plugin_weight_only_quant_type))

        mlp_fc_weight, mlp_fc_bias = get_weight_and_bias(
            model_params, prefix + 'fc1', dtype)
        split_v = split_matrix_tp(mlp_fc_weight, tensor_parallel, rank, dim=0)
        bias = split_matrix_tp(mlp_fc_bias, tensor_parallel, rank, dim=0)
        weights.update(
            get_tllm_linear_weight(split_v, tllm_prex + 'mlp.fc.', bias,
                                   use_weight_only,
                                   plugin_weight_only_quant_type))

        mlp_proj_weight, mlp_proj_bias = get_weight_and_bias(
            model_params, prefix + 'fc2', dtype)
        split_v = split_matrix_tp(mlp_proj_weight, tensor_parallel, rank, dim=1)
        weights.update(
            get_tllm_linear_weight(split_v, tllm_prex + 'mlp.proj.',
                                   mlp_proj_bias, use_weight_only,
                                   plugin_weight_only_quant_type))

        # Layer norms do not use tensor parallelism
        input_ln_weight, input_ln_bias = get_weight_and_bias(
            model_params, prefix + 'self_attn_layer_norm', dtype)
        weights[tllm_prex + 'input_layernorm.weight'] = input_ln_weight
        weights[tllm_prex + 'input_layernorm.bias'] = input_ln_bias

        post_ln_weight, post_ln_bias = get_weight_and_bias(
            model_params, prefix + 'final_layer_norm', dtype)
        weights[tllm_prex + 'post_layernorm.weight'] = post_ln_weight
        weights[tllm_prex + 'post_layernorm.bias'] = post_ln_bias

    embed_w = get_weight(model_params, 'model.decoder.embed_tokens', dtype)
    if 'model.decoder.project_in.weight' in model_params.keys():
        project_in = get_weight(model_params, 'model.decoder.project_in', dtype)
        project_out = get_weight(model_params, 'model.decoder.project_out',
                                 dtype)
        lm_head_w = torch.matmul(embed_w.float(), project_out.float()).to(dtype)
        embed_w = torch.matmul(embed_w.float(),
                               project_in.t().float()).to(dtype)
    elif 'lm_head.weight' in model_params.keys():
        lm_head_w = get_weight(model_params, 'lm_head', dtype)
    else:
        lm_head_w = embed_w.clone()

    weights['lm_head.weight'] = split_matrix_tp(lm_head_w,
                                                tensor_parallel,
                                                rank,
                                                dim=0)

    weights['transformer.vocab_embedding.weight'] = split_embedding(
        embed_w,
        tp_size=tensor_parallel,
        tp_rank=rank,
        use_parallel_embedding=use_parallel_embedding,
        sharding_dim=sharding_dim)

    embed_p = get_weight(model_params, 'model.decoder.embed_positions', dtype)
    weights['transformer.position_embedding.weight'] = split_embedding(
        embed_p[2:, :],
        tp_size=tensor_parallel,
        tp_rank=rank,
        use_parallel_embedding=use_parallel_embedding,
        sharding_dim=sharding_dim)

    if do_layer_norm_before:
        ln_f_w, ln_f_b = get_weight_and_bias(model_params,
                                             'model.decoder.final_layer_norm',
                                             dtype)
        weights['transformer.ln_f.weight'] = ln_f_w
        weights['transformer.ln_f.bias'] = ln_f_b

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights


if __name__ == '__main__':
    # TODO(qijun): Currently, the convert script depends on a torch op:
    # torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix,
    # which is included in tensorrt_llm Python package. Otherwise, the convert
    # script does not need to import tensorrt_llm. Will remove it after reimplementing
    # the op with PyTorch.
    print(tensorrt_llm.__version__)
    args = parse_arguments()
    world_size = args.tp_size * args.pp_size
    assert args.pp_size == 1, "Pipeline parallelism is not supported."

    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.model_type == 'opt':
        hf_model = AutoModelForCausalLM.from_pretrained(args.model_dir,
                                                        torch_dtype="auto")
    elif args.model_type == 'blip2':
        hf_model = Blip2ForConditionalGeneration.from_pretrained(
            args.model_dir, torch_dtype="auto").language_model

    hf_config = hf_model.config
    if hf_config.hidden_size != hf_config.word_embed_proj_dim:
        args.use_parallel_embedding = False

    quant_algo = None
    plugin_weight_only_quant_type = None
    if args.use_weight_only and args.weight_only_precision == 'int8':
        plugin_weight_only_quant_type = torch.int8
        quant_algo = QuantAlgo.W8A16
    elif args.use_weight_only and args.weight_only_precision == 'int4':
        plugin_weight_only_quant_type = torch.quint4x2
        quant_algo = QuantAlgo.W4A16

    config = {
        'architecture': hf_config.architectures[0],
        'dtype': args.dtype,
        'num_hidden_layers': hf_config.num_hidden_layers,
        'num_attention_heads': hf_config.num_attention_heads,
        'hidden_size': hf_config.hidden_size,
        'vocab_size': hf_config.vocab_size,
        'position_embedding_type': 'learned_absolute',
        'max_position_embeddings': hf_config.max_position_embeddings,
        'hidden_act': hf_config.activation_function,
        'quantization': {
            'quant_algo': quant_algo
        },
        'mapping': {
            'world_size': world_size,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size,
        },
        'use_parallel_embedding': args.use_parallel_embedding,
        'embedding_sharding_dim': args.embedding_sharding_dim,
        'do_layer_norm_before': hf_config.do_layer_norm_before,
    }

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    def covert_and_save(rank):
        weights = convert_hf_opt(
            hf_model,
            rank,
            world_size,
            dtype=args.dtype,
            use_weight_only=args.use_weight_only,
            plugin_weight_only_quant_type=plugin_weight_only_quant_type,
            use_parallel_embedding=args.use_parallel_embedding,
            sharding_dim=args.embedding_sharding_dim)
        safetensors.torch.save_file(
            weights, os.path.join(args.output_dir, f'rank{rank}.safetensors'))

    if args.workers == 1:
        for rank in range(world_size):
            covert_and_save(rank)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as p:
            futures = [
                p.submit(covert_and_save, rank) for rank in range(world_size)
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

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')

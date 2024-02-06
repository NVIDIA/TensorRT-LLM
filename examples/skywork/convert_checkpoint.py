import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, wait

import safetensors
import torch
from torch.nn.functional import pad
from transformers import AutoModelForCausalLM

import tensorrt_llm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--world_size',
                        type=int,
                        default=1,
                        help='world size, only support tensor parallelism now')
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float32', 'bfloat16', 'float16'])
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


def pad_vocab_size(vocab_size: int, tp_size: int):
    return int(math.ceil(vocab_size / tp_size) * tp_size)


def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return torch.chunk(v, tp_size)[idx].contiguous()
    else:
        return torch.chunk(v, tp_size, dim=dim)[idx].contiguous()


def split_qkv_tp(v, n_head, n_hidden, tensor_parallel, rank):
    """
    Splits the QKV matrix according to tensor parallelism
    """
    v = v.reshape(3, n_hidden, n_hidden)
    split_v = split(v, tensor_parallel, rank, dim=1)
    split_v = split_v.reshape(3 * (n_hidden // tensor_parallel), n_hidden)
    return split_v.contiguous()


def split_qkv_bias_tp(v, n_head, n_hidden, tensor_parallel, rank):
    """
    Splits the QKV bias according to tensor parallelism
    """
    v = v.reshape(3, n_hidden)
    split_v = split(v, tensor_parallel, rank, dim=1)
    split_v = split_v.reshape(3 * (n_hidden // tensor_parallel))
    return split_v.contiguous()


def split_matrix_tp(v, tensor_parallel, rank, dim):
    return split(v, tensor_parallel, rank, dim=dim)


def get_weight(config, prefix, dtype):
    return config[prefix + '.weight'].to(dtype).detach()


def get_bias(config, prefix, dtype):
    return config[prefix + '.bias'].to(dtype).detach()


def get_weight_and_bias(config, prefix, dtype):
    return get_weight(config, prefix, dtype), get_bias(config, prefix, dtype)


def get_tllm_linear_weight(weight,
                           prefix,
                           bias=None,
                           use_weight_only=False,
                           plugin_weight_only_quant_type=torch.int8):
    results = {}
    if use_weight_only:
        v = weight.t().contiguous()
        processed_torch_weights, torch_weight_scales = \
            torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                v, plugin_weight_only_quant_type)
        results[prefix + 'weight'] = processed_torch_weights
        results[prefix + 'per_channel_scale'] = torch_weight_scales
    else:
        results[prefix + 'weight'] = weight.contiguous()

    if bias is not None:
        results[prefix + 'bias'] = bias

    return results


def convert_hf_skywork(hf_model, rank=0, tensor_parallel=1, dtype='float32'):
    # TODO: add parallel embedding support
    weights = {}
    tik = time.time()

    model_params = dict(hf_model.named_parameters())
    dtype = getattr(torch, dtype)
    num_attention_heads = hf_model.config.num_attention_heads
    hidden_size = hf_model.config.hidden_size

    for l in range(hf_model.config.num_hidden_layers):
        prefix = f'model.layers.{l}.'
        tllm_prex = f'transformer.layers.{l}.'
        qkv_weight = torch.cat([
            get_weight(model_params, prefix + f"self_attn.{c}_proj", dtype)
            for c in "qkv"
        ],
                               dim=0)
        split_v = split_qkv_tp(qkv_weight, num_attention_heads, hidden_size,
                               tensor_parallel, rank)
        weights.update(
            get_tllm_linear_weight(split_v, tllm_prex + 'attention.qkv.', None))
        attn_dense_weight = get_weight(model_params,
                                       prefix + 'self_attn.o_proj', dtype)
        split_v = split_matrix_tp(attn_dense_weight,
                                  tensor_parallel,
                                  rank,
                                  dim=1)
        weights.update(
            get_tllm_linear_weight(
                split_v,
                tllm_prex + 'attention.dense.',
                None,
            ))
        mlp_gate_weight = get_weight(model_params, prefix + "mlp.gate_proj",
                                     dtype)
        split_v = split_matrix_tp(mlp_gate_weight, tensor_parallel, rank, dim=0)
        weights.update(
            get_tllm_linear_weight(
                split_v,
                tllm_prex + 'mlp.fc.',
                None,
            ))

        mlp_up_weight = get_weight(model_params, prefix + "mlp.up_proj", dtype)
        split_v = split_matrix_tp(mlp_up_weight, tensor_parallel, rank, dim=0)
        weights.update(
            get_tllm_linear_weight(
                split_v,
                tllm_prex + 'mlp.gate.',
                None,
            ))
        mlp_down_weight = get_weight(model_params, prefix + "mlp.down_proj",
                                     dtype)
        split_v = split_matrix_tp(mlp_down_weight, tensor_parallel, rank, dim=1)
        weights.update(
            get_tllm_linear_weight(split_v, tllm_prex + 'mlp.proj.', None))
        input_ln_weight = get_weight(model_params, prefix + "input_layernorm",
                                     dtype)
        weights[tllm_prex + "input_layernorm.weight"] = input_ln_weight
        post_ln_weight = get_weight(model_params,
                                    prefix + "post_attention_layernorm", dtype)
        weights[tllm_prex + "post_layernorm.weight"] = post_ln_weight

    embed_weight = get_weight(model_params, "model.embed_tokens", dtype)
    ## NOTE: vocab embedding is not sharded
    weights["transformer.vocab_embedding.weight"] = embed_weight
    ln_weight = get_weight(model_params, "model.norm", dtype)
    weights["transformer.ln_f.weight"] = ln_weight
    lm_head_weight = get_weight(model_params, "lm_head", dtype)
    vocab_size = lm_head_weight.shape[0]
    pad_width = pad_vocab_size(vocab_size, tensor_parallel) - vocab_size
    lm_head_weight = pad(lm_head_weight, (0, 0, 0, pad_width), "constant", 0)
    weights["lm_head.weight"] = split_matrix_tp(lm_head_weight,
                                                tensor_parallel,
                                                rank,
                                                dim=0)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights


def main():
    print(tensorrt_llm.__version__)
    args = parse_arguments()
    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    hf_model = AutoModelForCausalLM.from_pretrained(args.model_dir,
                                                    torch_dtype="auto",
                                                    trust_remote_code=True)
    hf_config = hf_model.config

    config = {
        'architecture': hf_config.architectures[0],
        'dtype': args.dtype,
        'num_hidden_layers': hf_config.num_hidden_layers,
        'num_attention_heads': hf_config.num_attention_heads,
        'num_key_value_heads': hf_config.num_key_value_heads,
        'hidden_size': hf_config.hidden_size,
        'mlp_hidden_size': hf_config.intermediate_size,
        'vocab_size': hf_config.vocab_size,
        'position_embedding_type': 'rope_gpt_neox',
        'max_position_embeddings': hf_config.max_position_embeddings,
        'hidden_act': hf_config.hidden_act,
        'rotary_base': hf_config.rope_theta,
        'rope_scaling': hf_config.rope_scaling,
        'norm_eps': hf_config.rms_norm_eps,
        'mapping': {
            'world_size': args.world_size,
            'tp_size': args.world_size,
        },
    }

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    def covert_and_save(rank):
        weights = convert_hf_skywork(hf_model,
                                     rank,
                                     args.world_size,
                                     dtype=args.dtype)
        safetensors.torch.save_file(
            weights, os.path.join(args.output_dir, f'rank{rank}.safetensors'))

    if args.workers == 1:
        for rank in range(args.world_size):
            covert_and_save(rank)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as p:
            futures = [
                p.submit(covert_and_save, rank)
                for rank in range(args.world_size)
            ]
            wait(futures)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')


if __name__ == "__main__":
    main()

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

import numpy as np
import safetensors
import torch
from einops import rearrange
from transformers import AutoConfig, AutoModelForCausalLM

import tensorrt_llm
from tensorrt_llm._utils import release_gc
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.llama import convert


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
    parser.add_argument('--log_level', type=str, default='info')
    args = parser.parse_args()

    tensorrt_llm.logger.set_level(args.log_level)
    return args


def get_qkv_weight(weight: torch.Tensor,
                   hidden_size: int,
                   num_heads: int,
                   tp_size: int,
                   tp_rank: int,
                   is_bias: bool,
                   num_kv_heads: Optional[int] = None) -> torch.Tensor:
    """ Splits the QKV matrix according to tensor parallelism """
    head_size = hidden_size // num_heads
    num_kv_groups = num_heads // num_kv_heads
    mha_mode = num_kv_heads == num_heads
    weight = rearrange(weight,
                       '(h gs d) dim -> h gs d dim',
                       gs=2 + num_kv_groups,
                       d=head_size)
    q_w, k_w, v_w = torch.split(weight, [num_kv_groups, 1, 1], dim=1)
    if is_bias:
        q_w = q_w.ravel()
        k_w = k_w.ravel()
        v_w = v_w.ravel()
        qkv_w = torch.cat((q_w, k_w, v_w))
        qkv_w = convert.split_qkv_bias_tp(qkv_w, num_heads, hidden_size,
                                          tp_size, tp_rank)
    else:
        q_w = rearrange(q_w, 'h gs d dim -> (h gs d) dim')
        k_w = rearrange(k_w, 'h gs d dim -> (h gs d) dim')
        v_w = rearrange(v_w, 'h gs d dim -> (h gs d) dim')
        if not mha_mode:
            if num_kv_heads < tp_size:
                k_w = convert.dup_kv_weight(k_w, num_kv_heads, tp_size)
                v_w = convert.dup_kv_weight(v_w, num_kv_heads, tp_size)
            assert (k_w.shape[0] % (tp_size * head_size)) == 0
            assert (v_w.shape[0] % (tp_size * head_size)) == 0
            wq = convert.split(q_w, tp_size, tp_rank)
            wk = convert.split(k_w, tp_size, tp_rank)
            wv = convert.split(v_w, tp_size, tp_rank)
            qkv_w = torch.concat((wq, wk, wv))

        else:
            qkv_w = torch.cat([q_w, k_w, v_w], dim=0)

            qkv_w = convert.split_qkv_tp(qkv_w, num_heads, hidden_size, tp_size,
                                         tp_rank)
    return qkv_w


def get_tllm_linear_weight(
    weight: torch.Tensor,
    prefix: str,
    bias: Optional[torch.Tensor] = None,
    use_weight_only: bool = False,
    plugin_weight_only_quant_type: torch.dtype = torch.int8
) -> Dict[str, torch.Tensor]:
    results = {}
    if use_weight_only:
        v = weight.t().contiguous()
        processed_torch_weights, torch_weight_scales = \
            torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                v, plugin_weight_only_quant_type)
        results[f'{prefix}.weight'] = processed_torch_weights
        results[f'{prefix}.per_channel_scale'] = torch_weight_scales
    else:
        results[f'{prefix}.weight'] = weight.contiguous()

    if bias is not None:
        results[f'{prefix}.bias'] = bias

    return results


def convert_from_hf(hf_model,
                    hf_config,
                    mapping: Mapping,
                    dtype: str = 'float32',
                    use_parallel_embedding: bool = False,
                    sharding_dim: int = 0,
                    use_weight_only: bool = False,
                    plugin_weight_only_quant_type: torch.dtype = torch.int8):
    weights = {}
    tik = time.time()

    model_params = dict(hf_model.named_parameters())
    #This is for InternVL2
    if hf_config.architectures[0] == 'InternLM2ForCausalLM':
        keys_to_rename = [
            key for key in model_params.keys() if 'language_model.' in key
        ]
        keys_to_delete = [
            key for key in model_params.keys() if 'vision_model.' in key
        ]
        for key in keys_to_rename:
            keys_rename = key.replace('language_model.', '')
            model_params[keys_rename] = model_params[key]
            del model_params[key]
        for key in keys_to_delete:
            del model_params[key]

    dtype = getattr(torch, dtype)
    num_attention_heads = hf_config.num_attention_heads
    hidden_size = hf_config.hidden_size
    vocab_size = hf_config.vocab_size
    num_kv_heads = hf_config.num_key_value_heads
    num_hidden_layers = hf_config.num_hidden_layers
    layers_range = mapping.pp_layers(num_hidden_layers)
    is_xcomposer2 = (
        hf_config.architectures[0] == 'InternLMXComposer2ForCausalLM')
    lora_weights = {}
    for l in layers_range:
        prefix = f'model.layers.{l}'
        tllm_prex = f'transformer.layers.{l - layers_range[0]}'

        qkv_weight = convert.get_weight(model_params,
                                        f'{prefix}.attention.wqkv', dtype)
        qkv_w = get_qkv_weight(qkv_weight,
                               hidden_size,
                               num_attention_heads,
                               mapping.tp_size,
                               mapping.tp_rank,
                               is_bias=False,
                               num_kv_heads=num_kv_heads)

        if is_xcomposer2:
            lora_prefix = f'base_model.model.model.layers.{l}'
            assert num_attention_heads % num_kv_heads == 0
            num_key_value_groups = num_attention_heads // num_kv_heads

            qkv_loraA = convert.get_weight(model_params,
                                           prefix + '.attention.wqkv.Plora_A',
                                           dtype)
            qkv_loraB = convert.get_weight(model_params,
                                           prefix + '.attention.wqkv.Plora_B',
                                           dtype)
            q_lora_name = f"{lora_prefix}.self_attn.q_proj"
            k_lora_name = f"{lora_prefix}.self_attn.k_proj"
            v_lora_name = f"{lora_prefix}.self_attn.v_proj"

            #save qkv_loraA to be (q/k/v)_loraA
            lora_weights[f"{q_lora_name}.lora_A.weight"] = qkv_loraA
            lora_weights[f"{k_lora_name}.lora_A.weight"] = qkv_loraA
            lora_weights[f"{v_lora_name}.lora_A.weight"] = qkv_loraA

            qkv_lora_rank = qkv_loraB.shape[-1]
            head_size = hidden_size // num_attention_heads

            qkv_loraB = qkv_loraB.reshape(-1, num_key_value_groups + 2,
                                          head_size, qkv_lora_rank)
            q_loraB = qkv_loraB[:, :num_key_value_groups, :, :].reshape(
                -1, qkv_lora_rank).contiguous()
            k_loraB = qkv_loraB[:,
                                -2, :, :].reshape(-1,
                                                  qkv_lora_rank).contiguous()
            v_loraB = qkv_loraB[:,
                                -1, :, :].reshape(-1,
                                                  qkv_lora_rank).contiguous()

            #save (q/k/v)_loraB
            lora_weights[f"{q_lora_name}.lora_B.weight"] = q_loraB
            lora_weights[f"{k_lora_name}.lora_B.weight"] = k_loraB
            lora_weights[f"{v_lora_name}.lora_B.weight"] = v_loraB

            wo_loraA = convert.get_weight(model_params,
                                          prefix + '.attention.wo.Plora_A',
                                          dtype)
            wo_loraB = convert.get_weight(model_params,
                                          prefix + '.attention.wo.Plora_B',
                                          dtype)

            lora_weights[
                f"{lora_prefix}.self_attn.o_proj.lora_A.weight"] = wo_loraA
            lora_weights[
                f"{lora_prefix}.self_attn.o_proj.lora_B.weight"] = wo_loraB

            mlp_gate_loraA = convert.get_weight(
                model_params, prefix + '.feed_forward.w3.Plora_A', dtype)
            mlp_gate_loraB = convert.get_weight(
                model_params, prefix + '.feed_forward.w3.Plora_B', dtype)
            lora_weights[
                f"{lora_prefix}.mlp.up_proj.lora_A.weight"] = mlp_gate_loraA
            lora_weights[
                f"{lora_prefix}.mlp.up_proj.lora_B.weight"] = mlp_gate_loraB

            mlp_fc_loraA = convert.get_weight(
                model_params, prefix + '.feed_forward.w1.Plora_A', dtype)
            mlp_fc_loraB = convert.get_weight(
                model_params, prefix + '.feed_forward.w1.Plora_B', dtype)
            lora_weights[
                f"{lora_prefix}.mlp.gate_proj.lora_A.weight"] = mlp_fc_loraA
            lora_weights[
                f"{lora_prefix}.mlp.gate_proj.lora_B.weight"] = mlp_fc_loraB

            mlp_proj_loraA = convert.get_weight(
                model_params, prefix + '.feed_forward.w2.Plora_A', dtype)
            mlp_proj_loraB = convert.get_weight(
                model_params, prefix + '.feed_forward.w2.Plora_B', dtype)
            lora_weights[
                f"{lora_prefix}.mlp.down_proj.lora_A.weight"] = mlp_proj_loraA
            lora_weights[
                f"{lora_prefix}.mlp.down_proj.lora_B.weight"] = mlp_proj_loraB

        qkv_bias = None
        if f'{prefix}.attention.wqkv.bias' in model_params:
            qkv_bias = convert.get_bias(model_params,
                                        f'{prefix}.attention.wqkv', dtype)
        if qkv_bias is None:
            qkv_b = None
        else:
            qkv_b = get_qkv_weight(qkv_bias,
                                   hidden_size,
                                   num_attention_heads,
                                   mapping.tp_size,
                                   mapping.tp_rank,
                                   is_bias=True,
                                   num_kv_heads=num_kv_heads)
        weights.update(
            get_tllm_linear_weight(
                qkv_w,
                f'{tllm_prex}.attention.qkv',
                qkv_b,
                use_weight_only,
                plugin_weight_only_quant_type,
            ))

        attn_dense_weight = convert.get_weight(model_params,
                                               f'{prefix}.attention.wo', dtype)
        attn_dense_w = convert.split_matrix_tp(attn_dense_weight,
                                               mapping.tp_size,
                                               mapping.tp_rank,
                                               dim=1)
        attn_dense_bias = None
        if f'{prefix}.attention.wo.bias' in model_params:
            attn_dense_bias = convert.get_bias(model_params,
                                               f'{prefix}.attention.wo', dtype)

        weights.update(
            get_tllm_linear_weight(
                attn_dense_w,
                f'{tllm_prex}.attention.dense',
                attn_dense_bias,
                use_weight_only,
                plugin_weight_only_quant_type,
            ))

        mlp_fc_weight = convert.get_weight(model_params,
                                           f'{prefix}.feed_forward.w1', dtype)
        mlp_fc_w = convert.split_matrix_tp(mlp_fc_weight,
                                           mapping.tp_size,
                                           mapping.tp_rank,
                                           dim=0)
        mlp_fc_b = None
        weights.update(
            get_tllm_linear_weight(mlp_fc_w, f'{tllm_prex}.mlp.fc', mlp_fc_b,
                                   use_weight_only,
                                   plugin_weight_only_quant_type))

        mlp_proj_weight = convert.get_weight(model_params,
                                             f'{prefix}.feed_forward.w2', dtype)
        mlp_proj_w = convert.split_matrix_tp(mlp_proj_weight,
                                             mapping.tp_size,
                                             mapping.tp_rank,
                                             dim=1)
        mlp_proj_bias = None
        weights.update(
            get_tllm_linear_weight(mlp_proj_w, f'{tllm_prex}.mlp.proj',
                                   mlp_proj_bias, use_weight_only,
                                   plugin_weight_only_quant_type))

        mlp_gate_weight = convert.get_weight(model_params,
                                             f'{prefix}.feed_forward.w3', dtype)
        mlp_gate_w = convert.split_matrix_tp(mlp_gate_weight,
                                             mapping.tp_size,
                                             mapping.tp_rank,
                                             dim=0)
        mlp_gate_bias = None
        weights.update(
            get_tllm_linear_weight(mlp_gate_w, f'{tllm_prex}.mlp.gate',
                                   mlp_gate_bias, use_weight_only,
                                   plugin_weight_only_quant_type))

        # Layer norms do not use tensor parallelism
        input_ln_weight = convert.get_weight(model_params,
                                             f'{prefix}.attention_norm', dtype)
        weights[f'{tllm_prex}.input_layernorm.weight'] = input_ln_weight

        post_ln_weight = convert.get_weight(model_params, f'{prefix}.ffn_norm',
                                            dtype)
        weights[f'{tllm_prex}.post_layernorm.weight'] = post_ln_weight

        release_gc()

    if is_xcomposer2:
        torch.save(lora_weights, 'adapter_model.bin')
        adapter_config = {
            "base_model_name_or_path":
            "Internlm-xcomposer-2-7b-vl-hf",
            "bias":
            "none",
            "enable_lora":
            None,
            "fan_in_fan_out":
            False,
            "inference_mode":
            True,
            "lora_alpha":
            256.0,
            "lora_dropout":
            0.05,
            "merge_weights":
            False,
            "modules_to_save":
            None,
            "peft_type":
            "LORA",
            "r":
            256,
            "target_modules": [
                "q_proj", "v_proj", "k_proj", "o_proj", "gate_proj",
                "down_proj", "up_proj"
            ],
            "task_type":
            "CAUSAL_LM"
        }
        with open(os.path.join('adapter_config.json'), 'w') as f:
            json.dump(adapter_config, f, indent=4)

    embed_w = convert.get_weight(model_params, 'model.tok_embeddings', dtype)
    if use_parallel_embedding:
        embed_w = convert.split_matrix_tp(embed_w,
                                          mapping.tp_size,
                                          mapping.tp_rank,
                                          dim=sharding_dim)
    if mapping.is_first_pp_rank():
        weights['transformer.vocab_embedding.weight'] = embed_w
    lm_head_weights = convert.get_weight(model_params, 'output', dtype)
    if mapping.is_last_pp_rank():
        if vocab_size % mapping.tp_size != 0:
            # padding
            vocab_size_padded = convert.pad_vocab_size(vocab_size,
                                                       mapping.tp_size)
            pad_width = vocab_size_padded - vocab_size

            lm_head_weights = torch.from_numpy(
                np.pad(lm_head_weights.detach().cpu().numpy(),
                       ((0, pad_width), (0, 0)),
                       'constant',
                       constant_values=0))
        weights['lm_head.weight'] = convert.split_matrix_tp(lm_head_weights,
                                                            mapping.tp_size,
                                                            mapping.tp_rank,
                                                            dim=0)
        ln_f_w = convert.get_weight(model_params, 'model.norm', dtype)
        weights['transformer.ln_f.weight'] = ln_f_w

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights


if __name__ == '__main__':
    args = parse_arguments()
    world_size = args.tp_size * args.pp_size

    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    quant_algo = None
    plugin_weight_only_quant_type = None
    if args.use_weight_only and args.weight_only_precision == 'int8':
        plugin_weight_only_quant_type = torch.int8
        quant_algo = 'W8A16'
    elif args.use_weight_only and args.weight_only_precision == 'int4':
        plugin_weight_only_quant_type = torch.quint4x2
        quant_algo = 'W4A16'

    hf_config = AutoConfig.from_pretrained(args.model_dir,
                                           trust_remote_code=True)
    #This is for InternVL2
    if hasattr(hf_config, 'llm_config'):
        hf_config = hf_config.llm_config

    config = {
        'architecture': hf_config.architectures[0],
        'dtype': args.dtype,
        'logits_dtype': 'float32',
        'num_hidden_layers': hf_config.num_hidden_layers,
        'num_attention_heads': hf_config.num_attention_heads,
        'num_key_value_heads': hf_config.num_key_value_heads,
        'hidden_size': hf_config.hidden_size,
        'intermediate_size': hf_config.intermediate_size,
        'norm_epsilon': hf_config.rms_norm_eps,
        'vocab_size': hf_config.vocab_size,
        'position_embedding_type': 'rope_gpt_neox',
        'rotary_base': hf_config.rope_theta,
        'max_position_embeddings': hf_config.max_position_embeddings,
        'hidden_act': hf_config.hidden_act,
        'use_parallel_embedding': args.use_parallel_embedding,
        'embedding_sharding_dim': args.embedding_sharding_dim,
        'quantization': {
            'quant_algo': quant_algo,
        },
        'mapping': {
            'world_size': world_size,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size,
        },
        'has_partial_lora_mask':
        hf_config.architectures[0] == 'InternLMXComposer2ForCausalLM',
        'attn_bias': getattr(hf_config, 'bias', False),
        'rotary_scaling': getattr(hf_config, "rope_scaling", None)
    }

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    def covert_and_save(rank):
        mapping = Mapping(world_size=world_size,
                          rank=rank,
                          tp_size=args.tp_size,
                          pp_size=args.pp_size)

        hf_model = AutoModelForCausalLM.from_pretrained(args.model_dir,
                                                        trust_remote_code=True,
                                                        dtype="auto")
        weights = convert_from_hf(
            hf_model,
            hf_config,
            mapping,
            dtype=args.dtype,
            use_parallel_embedding=args.use_parallel_embedding,
            sharding_dim=args.embedding_sharding_dim,
            use_weight_only=args.use_weight_only,
            plugin_weight_only_quant_type=plugin_weight_only_quant_type)
        del hf_model
        save_file = os.path.join(args.output_dir, f'rank{rank}.safetensors')
        print(f'Saving to {save_file}')
        safetensors.torch.save_file(weights, save_file)

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

import argparse
import configparser
import copy
import json
import logging
import os
import types
from ast import literal_eval
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import safetensors
from helper import (convert_weight_to_dtype, fairseq_sin_pos_embedding,
                    fuse_qkv_one_layer, reshape, split)
from transformers import (AutoModelForSeq2SeqLM, Blip2ForConditionalGeneration,
                          MBartForConditionalGeneration, NougatProcessor,
                          Pix2StructForConditionalGeneration,
                          T5ForConditionalGeneration, VisionEncoderDecoderModel)

from tensorrt_llm._utils import pad_vocab_size
from tensorrt_llm.functional import (LayerNormPositionType, LayerNormType,
                                     MLPType)
from tensorrt_llm.layers import LanguageAdapterConfig
from tensorrt_llm.models import PretrainedConfig

dir_path = os.path.dirname(os.path.realpath(__file__))
LOGGER = logging.getLogger(__name__)

layernorm_type_map = {i.name: i.value for i in LayerNormType}
layernorm_position_map = {i.name: i.value for i in LayerNormPositionType}
mlp_type_map = {i.name: i.value for i in MLPType}

# Constants for specific model configurations
ECLAIR_RADIO_MAX_POSITION_EMBEDDINGS = 20000


def copy_args_to_component_config(component_config, args):
    for arg in vars(args):
        setattr(component_config, arg, getattr(args, arg))
    return component_config


def parse_t5_config(args, hf_model):
    config = configparser.ConfigParser()

    config["encoder"] = {}
    for key, val in hf_model.encoder.config.to_dict().items():
        config["encoder"][key] = f"{val}"

    # manually set q_scaling to offset attention scaling's effect.
    # TODO: modify kernels to control whether to disable attention scaling
    def get_offset_q_scaling(config):
        scaling = 1 / config.head_size**.5
        return scaling

    config["decoder"] = {}
    for key, val in hf_model.decoder.config.to_dict().items():
        config["decoder"][key] = f"{val}"

    config["structure"] = dict()
    config["structure"]["t5_with_bias"] = "false"
    config["structure"]["use_gated_activation"] = str(
        hf_model.encoder.config.is_gated_act)
    config["structure"]["position_embedding_type"] = "relative"
    config["structure"]["model_type"] = args.model_type

    def parse_t5_config_by_component(config, component, args):
        component_config = types.SimpleNamespace()
        component_config = copy_args_to_component_config(component_config, args)
        component_config.n_head = config.getint(component, 'num_heads')
        component_config.head_size = config.getint(component, 'd_kv')
        component_config.hidden_size = config.getint(component, 'd_model')
        component_config.ffn_hidden_size = config.getint(component, 'd_ff')
        component_config.vocab_size = config.getint(component, 'vocab_size')
        component_config.n_positions = config.getint(component,
                                                     'n_positions',
                                                     fallback=512)
        component_config.has_position_embedding = config.getboolean(
            component, 'has_position_embedding',
            fallback=False)  # TODO: hardcoded here

        component_config.has_token_type_embedding = config.getboolean(
            component, 'has_token_type_embedding', fallback=False)
        component_config.has_embedding_layernorm = config.getboolean(
            component, 'has_embedding_layernorm', fallback=False)
        component_config.has_embedding_scale = config.getboolean(
            component, 'has_embedding_scale', fallback=False)
        component_config.q_scaling = get_offset_q_scaling(component_config)
        component_config.has_attention_qkvo_bias = config.getboolean(
            component, 'has_attention_qkvo_bias',
            fallback=False)  # TODO: hardcoded here
        component_config.has_mlp_bias = config.getboolean(component,
                                                          'has_mlp_bias',
                                                          fallback=False)
        component_config.has_model_final_layernorm = config.getboolean(
            component, 'has_model_final_layernorm', fallback=True)
        component_config.layernorm_eps = config.getfloat(
            component, 'layer_norm_epsilon')
        component_config.layernorm_position = layernorm_position_map[config.get(
            component, 'layernorm_position',
            fallback='pre_layernorm')]  # TODO: hardcoded here
        component_config.layernorm_type = layernorm_type_map[config.get(
            component, 'layernorm_type', fallback='RmsNorm')]
        component_config.hidden_act = config.get(component, 'dense_act_fn')
        component_config.gated_act = config.getboolean(component,
                                                       'is_gated_act')
        component_config.mlp_type = mlp_type_map['GatedMLP' if component_config.
                                                 gated_act else 'MLP']
        component_config.num_buckets = config.getint(
            component, 'relative_attention_num_buckets')
        component_config.max_distance = config.getint(
            component, 'relative_attention_max_distance')
        component_config.position_embedding_type = config.get(
            'structure', 'position_embedding_type')
        component_config.logits_dtype = config.get(component,
                                                   'logits_dtype',
                                                   fallback='float32')

        if component == 'encoder':
            component_config.n_layer = config.getint(component, 'num_layers')

            component_config.relative_attention = config.get(
                'structure', 'position_embedding_type') == 'relative'

        elif component == 'decoder':
            component_config.n_layer = config.getint(component,
                                                     'num_decoder_layers')
            component_config.has_lm_head_bias = config.getboolean(
                component,  # TODO: T5 with bias
                'has_lm_head_bias',
                fallback=False)
            component_config.relative_attention = config.getboolean(
                component, 'relative_attention', fallback=True)
            component_config.rescale_before_lm_head = config.getboolean(
                component, 'tie_word_embeddings'
            )  # default is True (for T5), but False for Flan-T5
            component_config.encoder_hidden_size = config.getint(
                'encoder', 'd_model')
            component_config.encoder_num_heads = config.getint(
                'encoder', 'num_heads')
            component_config.encoder_head_size = config.getint(
                'encoder', 'd_kv')
            component_config.decoder_start_token_id = config.getint(
                'decoder', 'decoder_start_token_id')
            component_config.eos_token_id = config.getint(
                'decoder', 'eos_token_id')
            bos_token_id = config.get('decoder', 'bos_token_id')
            # T5 does not have bos_token_id
            component_config.bos_token_id = int(
                bos_token_id) if bos_token_id != "None" else None
            component_config.pad_token_id = config.getint(
                'decoder', 'pad_token_id')

        else:
            assert False, 'Unsupported component!'

        return component_config

    encoder_config = parse_t5_config_by_component(config, "encoder", args)
    decoder_config = parse_t5_config_by_component(config, "decoder", args)

    return encoder_config, decoder_config


def convert_t5_weights_to_tllm_safetensors(config, component, params):
    weights = {}

    mapping = config.mapping

    convert_weight_to_dtype(params, config.dtype)
    hidden_size = config.hidden_size
    ffn_hidden_size = config.intermediate_size
    num_layers = config.num_hidden_layers
    n_head = config.num_attention_heads
    head_size = config.head_size
    attention_hidden_size = n_head * head_size  # head size * num_heads not necessarily equals hidden_dim, such as Flan-T5

    hf_param_prefix = f'{component}'
    trtllm_layer_name = f'transformer.layers'
    trtllm_attn_layer_name = 'attention' if component == 'encoder' else 'self_attention'
    trtllm_attn_layernorm_name = 'self_attention_layernorm' if component == 'decoder' else 'attention_layernorm'
    hf_component_idx = 1 if component == 'encoder' else 2

    def get_attn_module_name(component, block, layer, attn_type):
        return f'{component}.block.{int(block)}.layer.{int(layer)}.{attn_type}'

    weights['transformer.vocab_embedding.weight'] = reshape(
        params['shared.weight'].clone(),
        None) if not config.use_parallel_embedding else reshape(
            split(params['shared.weight'].clone(), mapping.tp_size,
                  mapping.tp_rank, 0), None)

    layers_range = mapping.pp_layers(num_layers)
    for layer_idx in layers_range:
        local_layer_idx = layer_idx - layers_range[0]
        trtllm_layer_name_prefix = f'{trtllm_layer_name}.{local_layer_idx}'
        hf_layer_name_prefix = f'{hf_param_prefix}.block.{layer_idx}'

        hidden_layer_name_split = {
            f'{hf_layer_name_prefix}.layer.0.SelfAttention.o.weight': {
                "name":
                f'{trtllm_layer_name_prefix}.{trtllm_attn_layer_name}.dense.weight',
                "shape":
                (hidden_size, attention_hidden_size // mapping.tp_size),
                "split_dim": -1
            },
            f'{hf_layer_name_prefix}.layer.{hf_component_idx}.DenseReluDense.wo.weight':
            {
                "name": f'{trtllm_layer_name_prefix}.mlp.proj.weight',
                "shape": (hidden_size, ffn_hidden_size // mapping.tp_size),
                "split_dim": -1
            },
            f'{hf_layer_name_prefix}.layer.{hf_component_idx}.DenseReluDense.wi.weight':
            {
                "name": f'{trtllm_layer_name_prefix}.mlp.fc.weight',
                "shape": (ffn_hidden_size // mapping.tp_size, hidden_size),
                "split_dim": 0
            },
            f'{hf_layer_name_prefix}.layer.{hf_component_idx}.DenseReluDense.wi_0.weight':
            {
                "name": f'{trtllm_layer_name_prefix}.mlp.fc.weight',
                "shape": (ffn_hidden_size // mapping.tp_size, hidden_size),
                "split_dim": 0
            },
        }

        hidden_layer_name_no_split = {
            f'{hf_layer_name_prefix}.layer.0.layer_norm.weight': {
                "name":
                f'{trtllm_layer_name_prefix}.{trtllm_attn_layernorm_name}.weight',
                "shape": None
            },
            f'{hf_layer_name_prefix}.layer.{hf_component_idx}.layer_norm.weight':
            {
                "name": f'{trtllm_layer_name_prefix}.mlp_layernorm.weight',
                "shape": None
            },
        }

        if config.gated_act:
            hidden_layer_name_split.update({
                f'{hf_layer_name_prefix}.layer.{hf_component_idx}.DenseReluDense.wi2.weight':
                {
                    "name": f'{trtllm_layer_name_prefix}.mlp.gate.weight',
                    "shape": (ffn_hidden_size // mapping.tp_size, hidden_size),
                    "split_dim": 0
                },
                f'{hf_layer_name_prefix}.layer.{hf_component_idx}.DenseReluDense.wi_1.weight':
                {
                    "name": f'{trtllm_layer_name_prefix}.mlp.gate.weight',
                    "shape": (ffn_hidden_size // mapping.tp_size, hidden_size),
                    "split_dim": 0
                },
            })

        if component == 'decoder':
            hidden_layer_name_split.update({
                f'{hf_layer_name_prefix}.layer.1.EncDecAttention.o.weight': {
                    "name":
                    f'{trtllm_layer_name_prefix}.cross_attention.dense.weight',
                    "shape":
                    (hidden_size, attention_hidden_size // mapping.tp_size),
                    "split_dim": -1
                },
            })
            hidden_layer_name_no_split.update({
                f'{hf_layer_name_prefix}.layer.1.layer_norm.weight': {
                    "name":
                    f'{trtllm_layer_name_prefix}.cross_attention_layernorm.weight',
                    "shape": None
                },
            })
            self_attn_module_name = get_attn_module_name(
                component, layer_idx, "1", 'EncDecAttention')
            weights.update(
                fuse_qkv_one_layer(
                    params, self_attn_module_name,
                    f'{trtllm_layer_name_prefix}.cross_attention',
                    mapping.tp_size, mapping.tp_rank, config.model_type,
                    (attention_hidden_size * 3 // mapping.tp_size, hidden_size),
                    None))

        self_attn_module_name = get_attn_module_name(component, layer_idx, "0",
                                                     'SelfAttention')
        weights.update(
            fuse_qkv_one_layer(
                params, self_attn_module_name,
                f'{trtllm_layer_name_prefix}.{trtllm_attn_layer_name}',
                mapping.tp_size, mapping.tp_rank, config.model_type,
                (attention_hidden_size * 3 // mapping.tp_size, hidden_size),
                None))

        weights[
            f'{trtllm_layer_name_prefix}.{trtllm_attn_layer_name}.rel_attn_table'] = reshape(
                split(
                    params[
                        f'{component}.block.0.layer.0.SelfAttention.relative_attention_bias.weight']
                    .T, mapping.tp_size, mapping.tp_rank, 0),
                (n_head // mapping.tp_size, config.num_buckets))

        for hf_weight_name, weight_info in hidden_layer_name_split.items():
            if hf_weight_name in params.keys():
                weights[weight_info["name"]] = reshape(
                    split(params[hf_weight_name],
                          mapping.tp_size,
                          mapping.tp_rank,
                          dim=weight_info["split_dim"]), weight_info["shape"])
        for hf_weight_name, weight_info in hidden_layer_name_no_split.items():
            if hf_weight_name in params.keys():
                weights[weight_info["name"]] = reshape(
                    params[hf_weight_name].clone(), shape=weight_info["shape"])

    weights['transformer.ln_f.weight'] = reshape(
        params[f'{component}.final_layer_norm.weight'].clone(), None)

    if component == 'decoder':
        weights['lm_head.weight'] = reshape(
            split(params['lm_head.weight'],
                  mapping.tp_size,
                  mapping.tp_rank,
                  dim=0), (config.vocab_size // mapping.tp_size, hidden_size))
        if not config.use_implicit_relative_attention:
            weights['rel_attn_table'] = reshape(
                split(
                    params[
                        f'{component}.block.0.layer.0.SelfAttention.relative_attention_bias.weight']
                    .T, mapping.tp_size, mapping.tp_rank, 0),
                (n_head // mapping.tp_size, config.num_buckets))

    return weights


convert_blip2_weights_to_tllm_safetensors = convert_t5_weights_to_tllm_safetensors  # func alias


def parse_nmt_config_by_component(config, component, args):
    assert component in ('encoder', 'decoder'), 'Unsupported component!'
    component_config = types.SimpleNamespace()
    component_config = copy_args_to_component_config(component_config, args)
    component_config.n_layer = config.getint(component, f'{component}_layers')
    component_config.n_head = config.getint(component,
                                            f'{component}_attention_heads')
    component_config.hidden_size = config.getint(
        component, f'{component}_embed_dim')  # fairseq naming
    component_config.head_size = config.getint(
        component,
        'd_kv',
        fallback=component_config.hidden_size // component_config.n_head)
    component_config.ffn_hidden_size = config.getint(
        component, f'{component}_ffn_embed_dim')  # fairseq naming
    component_config.vocab_size = config.getint(component, 'vocab_size')
    component_config.n_positions = config.getint(
        component, 'max_source_positions')  # fairseq naming
    component_config.has_position_embedding = not config.getboolean(
        component, 'no_token_positional_embeddings',
        fallback=False)  # fairseq naming
    component_config.has_token_type_embedding = config.getboolean(
        component, 'has_token_type_embedding', fallback=False)
    component_config.has_embedding_layernorm = config.getboolean(
        component, 'layernorm_embedding', fallback=True)  # fairseq naming
    component_config.has_embedding_scale = not config.getboolean(
        component, 'no_scale_embedding')  # fairseq naming
    component_config.q_scaling = config.getfloat(component,
                                                 'q_scaling',
                                                 fallback=1.0)
    component_config.has_attention_qkvo_bias = config.getboolean('structure',
                                                                 't5_with_bias',
                                                                 fallback=True)
    component_config.has_mlp_bias = config.getboolean('structure',
                                                      't5_with_bias',
                                                      fallback=True)
    component_config.has_model_final_layernorm = config.getboolean(
        component, 'has_model_final_layernorm')
    component_config.layernorm_eps = config.getfloat(
        component, 'layer_norm_epsilon', fallback=1e-5)  # fairseq naming

    normalize_before = config.getboolean(
        component, f'{component}_normalize_before')  # fairseq naming
    component_config.layernorm_position = layernorm_position_map[
        'pre_layernorm' if normalize_before else 'post_layernorm']

    component_config.layernorm_type = layernorm_type_map[config.get(
        component, 'layernorm_type', fallback='LayerNorm')]
    component_config.hidden_act = config.get(component,
                                             'activation_fn')  # fairseq naming
    component_config.gated_act = config.getboolean(component,
                                                   'is_gated_act',
                                                   fallback=False)
    component_config.mlp_type = mlp_type_map['GatedMLP' if component_config.
                                             gated_act else 'MLP']
    component_config.relative_attention = config.get(
        'structure', 'position_embedding_type') == 'relative'

    component_config.num_buckets = config.getint(
        component, 'relative_attention_num_buckets', fallback=0)
    component_config.max_distance = config.getint(
        component, 'relative_attention_max_distance', fallback=0)
    component_config.position_embedding_type = config.get(
        'structure', 'position_embedding_type')
    component_config.logits_dtype = config.get(component,
                                               'logits_dtype',
                                               fallback='float32')
    if component == 'decoder':
        component_config.rescale_before_lm_head = config.getboolean(
            component, 'rescale_before_lm_head')

        component_config.encoder_hidden_size = config.getint(
            'encoder', 'encoder_embed_dim')  # fairseq naming
        component_config.encoder_num_heads = config.getint(
            'encoder', 'encoder_attention_heads')
        component_config.encoder_head_size = config.getint(
            'encoder',
            'd_kv',
            fallback=component_config.encoder_hidden_size //
            component_config.encoder_num_heads)
        component_config.decoder_start_token_id = None
        component_config.eos_token_id = None
        component_config.bos_token_id = None
        component_config.pad_token_id = None

    return component_config


def parse_nmt_config(args, model):
    config = configparser.ConfigParser()
    fairseq_config = vars(model.cfg.model)  # Namespace --> dict

    config['encoder'] = dict()
    for key, val in fairseq_config.items():
        config["encoder"][key] = f"{val}"
    config["encoder"]["q_scaling"] = '1'
    # NMT has final layernorm for pre-norm model architecture.
    config['encoder']['has_model_final_layernorm'] = config['encoder'][
        'encoder_normalize_before']
    config['encoder']['vocab_size'] = str(len(model.src_dict))  # fairseq naming

    config['decoder'] = dict()
    for key, val in fairseq_config.items():
        config["decoder"][key] = f"{val}"
    config["decoder"]["q_scaling"] = '1'
    config["decoder"]["rescale_before_lm_head"] = 'false'
    config['decoder']['has_model_final_layernorm'] = str(
        config['decoder'].getboolean('decoder_normalize_before', False)
        and not config['decoder'].getboolean('no_decoder_final_norm', False))
    config['decoder']['vocab_size'] = str(len(model.tgt_dict))  # fairseq naming

    config["structure"] = dict()
    config["structure"]["t5_with_bias"] = "true"
    config["structure"]["use_gated_activation"] = "false"
    config["structure"][
        "position_embedding_type"] = "learned_absolute"  # "sinusoid"
    config["structure"]["model_type"] = args.model_type

    encoder_config = parse_nmt_config_by_component(config, "encoder", args)
    decoder_config = parse_nmt_config_by_component(config, "decoder", args)

    return encoder_config, decoder_config


def convert_nmt_weights_to_tllm_safetensors(config, component, params,
                                            sin_pos_embedding):
    weights = {}

    mapping = config.mapping

    hidden_size = config.hidden_size

    convert_weight_to_dtype(params, config.dtype)
    ffn_hidden_size = config.intermediate_size
    vocab_size = config.vocab_size

    hf_param_prefix = f'models.0.{component}'
    trtllm_layer_name = f'transformer.layers'
    trtllm_attn_layer_name = 'attention' if component == 'encoder' else 'self_attention'
    trtllm_attn_layernorm_name = 'self_attention_layernorm' if component == 'decoder' else 'attention_layernorm'

    hidden_layer_name_split = {
        'self_attn.out_proj.weight': {
            "name": f'{trtllm_attn_layer_name}.dense.weight',
            "shape": (hidden_size, hidden_size // mapping.tp_size),
            "split_dim": -1
        },
        'fc1.weight': {
            "name": 'mlp.fc.weight',
            "shape": (ffn_hidden_size // mapping.tp_size, hidden_size),
            "split_dim": 0
        },
        'fc1.bias': {
            "name": 'mlp.fc.bias',
            "shape": (ffn_hidden_size // mapping.tp_size),
            "split_dim": 0
        },
        'fc2.weight': {
            "name": 'mlp.proj.weight',
            "shape": (hidden_size, ffn_hidden_size // mapping.tp_size),
            "split_dim": -1
        },
    }

    hidden_layer_name_no_split = {
        'self_attn.out_proj.bias': {
            "name": f'{trtllm_attn_layer_name}.dense.bias',
            "shape": (hidden_size)
        },
        'self_attn_layer_norm.weight': {
            "name": f'{trtllm_attn_layernorm_name}.weight',
            "shape": None
        },
        'self_attn_layer_norm.bias': {
            "name": f'{trtllm_attn_layernorm_name}.bias',
            "shape": None
        },
        'fc2.bias': {
            "name": 'mlp.proj.bias',
            "shape": (hidden_size)
        },
        'final_layer_norm.weight': {
            "name": 'mlp_layernorm.weight',
            "shape": None
        },
        'final_layer_norm.bias': {
            "name": 'mlp_layernorm.bias',
            "shape": None
        },
    }

    if component == "decoder":
        hidden_layer_name_split.update({
            'encoder_attn.out_proj.weight': {
                "name": 'cross_attention.dense.weight',
                "shape": (hidden_size, hidden_size // mapping.tp_size),
                "split_dim": -1
            },
        })
        hidden_layer_name_no_split.update({
            'encoder_attn.out_proj.bias': {
                "name": 'cross_attention.dense.bias',
                "shape": (hidden_size)
            },
            'encoder_attn_layer_norm.weight': {
                "name": 'cross_attention_layernorm.weight',
                "shape": None,
            },
            'encoder_attn_layer_norm.bias': {
                "name": 'cross_attention_layernorm.bias',
                "shape": None
            },
        })

    def get_attn_module_name(component, layer, attn_type):
        return f'models.0.{component}.layers.{int(layer)}.{attn_type}'

    weights["transformer.vocab_embedding.weight"] = reshape(
        params[f'{hf_param_prefix}.embed_tokens.weight'].clone(),
        (vocab_size, -1)) if not config.use_parallel_embedding else reshape(
            split(params[f'{hf_param_prefix}.embed_tokens.weight'].clone(),
                  mapping.tp_size, mapping.tp_rank, 0),
            (vocab_size // mapping.tp_size, -1))
    weights["transformer.position_embedding.weight"] = reshape(
        sin_pos_embedding, (config.max_position_embeddings, hidden_size))

    num_layers = config.num_hidden_layers

    layers_range = mapping.pp_layers(num_layers)
    for layer_idx in layers_range:
        local_layer_idx = layer_idx - layers_range[0]
        hf_layer_name_prefix = f'{hf_param_prefix}.layers.{layer_idx}'
        trtllm_layer_name_prefix = f'{trtllm_layer_name}.{local_layer_idx}'

        for hf_weight_name, weight_info in hidden_layer_name_split.items():
            weights[
                f'{trtllm_layer_name_prefix}.{weight_info["name"]}'] = reshape(
                    split(params[f'{hf_layer_name_prefix}.{hf_weight_name}'],
                          mapping.tp_size,
                          mapping.tp_rank,
                          dim=weight_info["split_dim"]), weight_info["shape"])

        for hf_weight_name, weight_info in hidden_layer_name_no_split.items():
            trtllm_layer_fullname = f'{trtllm_layer_name_prefix}.{weight_info["name"]}'
            hf_layer_fullname = f'{hf_layer_name_prefix}.{hf_weight_name}'
            weights[trtllm_layer_fullname] = reshape(
                params[hf_layer_fullname].clone(), shape=weight_info["shape"])

        self_attn_module_name = get_attn_module_name(component, layer_idx,
                                                     'self_attn')
        weights.update(
            fuse_qkv_one_layer(
                params, self_attn_module_name,
                f'{trtllm_layer_name_prefix}.{trtllm_attn_layer_name}',
                mapping.tp_size, mapping.tp_rank, config.model_type,
                (hidden_size * 3 // mapping.tp_size, hidden_size),
                (hidden_size * 3 // mapping.tp_size)))
        if component == 'decoder':
            cross_attn_module_name = get_attn_module_name(
                component, layer_idx, 'encoder_attn')
            weights.update(
                fuse_qkv_one_layer(
                    params, cross_attn_module_name,
                    f'{trtllm_layer_name_prefix}.cross_attention',
                    mapping.tp_size, mapping.tp_rank, config.model_type,
                    (hidden_size * 3 // mapping.tp_size, hidden_size),
                    (hidden_size * 3 // mapping.tp_size)))

    if component == 'decoder':
        weights['lm_head.weight'] = reshape(
            split(params[f'{hf_param_prefix}.output_projection.weight'],
                  mapping.tp_size,
                  mapping.tp_rank,
                  dim=0), (config.vocab_size // mapping.tp_size, hidden_size))

    if config.has_model_final_layernorm:
        weights['transformer.ln_f.weight'] = params[
            f'{hf_param_prefix}.layer_norm.weight'].clone()
        weights['transformer.ln_f.bias'] = params[
            f'{hf_param_prefix}.layer_norm.bias'].clone()

    return weights


def parse_bart_config(args, hf_model):

    config = configparser.ConfigParser()

    config['decoder'] = dict()
    if args.eclair_radio:
        for key, val in hf_model.config.to_dict().items():
            config["decoder"][key] = f"{val}"
    else:
        for key, val in hf_model.model.decoder.config.to_dict().items():
            config["decoder"][key] = f"{val}"
    config["decoder"]["q_scaling"] = '1'
    config["decoder"]["rescale_before_lm_head"] = str(False)
    config['decoder']['has_model_final_layernorm'] = str(
        args.nougat or args.eclair_radio
        or isinstance(hf_model, MBartForConditionalGeneration))

    if args.nougat or args.eclair_radio:
        # These flags are true for mbart decoders, but missing in HF config
        config['decoder']['normalize_before'] = str(True)
        config['decoder']['normalize_embeddings'] = str(True)

        config['encoder'] = dict()
        # Init few encoder configs, needed by build, from decoder config
        encoder_config_keys = [
            "encoder_ffn_dim", "encoder_layers", "encoder_attention_heads",
            "encoder_layerdrop", "d_model"
        ]
        for key in encoder_config_keys:
            config['encoder'][key] = config['decoder'][key]
    else:
        config['encoder'] = dict()
        for key, val in hf_model.model.encoder.config.to_dict().items():
            config["encoder"][key] = f"{val}"
        config["encoder"]["q_scaling"] = '1'

        # mBART has final layernorm, BART does not
        config['encoder']['has_model_final_layernorm'] = str(
            isinstance(hf_model, MBartForConditionalGeneration))

    config["structure"] = dict()
    config["structure"]["t5_with_bias"] = "true"
    config["structure"]["use_gated_activation"] = "false"
    config["structure"]["position_embedding_type"] = "learned_absolute"
    config["structure"]["model_type"] = args.model_type

    def parse_bart_config_by_component(config, component, args):
        assert component in ('encoder', 'decoder'), 'Unsupported component!'
        component_config = types.SimpleNamespace()
        component_config = copy_args_to_component_config(component_config, args)
        component_config.n_layer = config.getint(component,
                                                 f'{component}_layers')
        component_config.n_head = config.getint(component,
                                                f'{component}_attention_heads')
        component_config.hidden_size = config.getint(component, 'd_model')
        component_config.head_size = config.getint(
            component,
            'd_kv',
            fallback=component_config.hidden_size // component_config.n_head)
        component_config.ffn_hidden_size = config.getint(
            component, f'{component}_ffn_dim')
        component_config.vocab_size = config.getint(component, 'vocab_size')
        component_config.n_positions = config.getint(component,
                                                     'max_position_embeddings')
        component_config.has_position_embedding = config.getboolean(
            component, 'has_position_embedding',
            fallback=True)  # TODO: hardcoded here
        component_config.has_token_type_embedding = config.getboolean(
            component, 'has_token_type_embedding', fallback=False)
        component_config.has_embedding_layernorm = config.getboolean(
            component, 'has_embedding_layernorm', fallback=True)
        component_config.has_embedding_scale = config.getboolean(
            component, 'scale_embedding')
        component_config.q_scaling = config.getfloat(component,
                                                     'q_scaling',
                                                     fallback=1.0)
        component_config.has_attention_qkvo_bias = config.getboolean(
            'structure', 't5_with_bias', fallback=True)
        component_config.has_mlp_bias = config.getboolean('structure',
                                                          't5_with_bias',
                                                          fallback=True)
        component_config.has_model_final_layernorm = config.getboolean(
            component, 'has_model_final_layernorm')
        component_config.layernorm_eps = config.getfloat(component,
                                                         'layer_norm_epsilon',
                                                         fallback=False)

        normalize_before = config.getboolean(component, 'normalize_before')
        component_config.layernorm_position = layernorm_position_map[
            'pre_layernorm' if normalize_before else 'post_layernorm']

        component_config.layernorm_type = layernorm_type_map[config.get(
            component, 'layernorm_type', fallback='LayerNorm')]
        component_config.hidden_act = config.get(component,
                                                 'activation_function')
        component_config.gated_act = config.getboolean(component,
                                                       'is_gated_act',
                                                       fallback=False)
        component_config.mlp_type = mlp_type_map['GatedMLP' if component_config.
                                                 gated_act else 'MLP']
        component_config.relative_attention = config.get(
            'structure', 'position_embedding_type') == 'relative'

        component_config.num_buckets = config.getint(
            component, 'relative_attention_num_buckets', fallback=0)
        component_config.max_distance = config.getint(
            component, 'relative_attention_max_distance', fallback=0)
        component_config.max_lora_rank = config.getint(component,
                                                       'max_lora_rank',
                                                       fallback=0)
        component_config.lora_target_modules = literal_eval(
            config.get(component, 'lora_target_modules', fallback="[]"))
        component_config.hf_modules_to_trtllm_modules = literal_eval(
            config.get(component, 'hf_modules_to_trtllm_modules',
                       fallback="{}"))
        component_config.trtllm_modules_to_hf_modules = literal_eval(
            config.get(component, 'trtllm_modules_to_hf_modules',
                       fallback="{}"))
        component_config.logits_dtype = config.get(component,
                                                   'logits_dtype',
                                                   fallback='float32')
        component_config.position_embedding_type = config.get(
            'structure', 'position_embedding_type')

        if component == 'decoder':
            component_config.rescale_before_lm_head = config.getboolean(
                component, 'rescale_before_lm_head')

            component_config.encoder_hidden_size = config.getint(
                'encoder', 'd_model')
            component_config.encoder_num_heads = config.getint(
                'encoder', 'encoder_attention_heads')
            component_config.encoder_head_size = config.getint(
                'encoder',
                'd_kv',
                fallback=component_config.encoder_hidden_size //
                component_config.encoder_num_heads)

            # nougat has decoder_start_token_id = None, special handling
            decoder_start_token_id = config.get('decoder',
                                                'decoder_start_token_id')
            component_config.decoder_start_token_id = int(
                decoder_start_token_id
            ) if decoder_start_token_id != "None" else None
            component_config.eos_token_id = config.getint(
                'decoder', 'eos_token_id')
            component_config.bos_token_id = config.getint(
                'decoder', 'bos_token_id')
            component_config.pad_token_id = config.getint(
                'decoder', 'pad_token_id')

        return component_config

    encoder_config = None
    if not (args.nougat or args.eclair_radio):
        encoder_config = parse_bart_config_by_component(config, "encoder", args)
    decoder_config = parse_bart_config_by_component(config, "decoder", args)

    # Override n_positions for eclair_radio model
    if args.eclair_radio:
        decoder_config.n_positions = ECLAIR_RADIO_MAX_POSITION_EMBEDDINGS

    return encoder_config, decoder_config


def convert_bart_weights_to_tllm_safetensors(config, component, params):
    weights = {}

    mapping = config.mapping

    hidden_size = config.hidden_size

    convert_weight_to_dtype(params, config.dtype)
    ffn_hidden_size = config.intermediate_size
    vocab_size = config.vocab_size

    hf_param_prefix = f'model.{component}'
    trtllm_layer_name = f'transformer.layers'
    trtllm_attn_layer_name = 'attention' if component == 'encoder' else 'self_attention'
    trtllm_attn_layernorm_name = 'self_attention_layernorm' if component == 'decoder' else 'attention_layernorm'
    embedding_layer_names = {
        'embed_tokens.weight': {
            "name": 'transformer.vocab_embedding.weight',
            "shape": (vocab_size, -1)
        },
        'embed_positions.weight': {
            "name": 'transformer.position_embedding.weight',
            "shape": (config.max_position_embeddings, hidden_size)
        },
        'layernorm_embedding.weight': {
            "name": 'transformer.ln_embed.weight',
            "shape": None
        },
        'layernorm_embedding.bias': {
            "name": 'transformer.ln_embed.bias',
            "shape": None
        },
    }

    hidden_layer_name_split = {
        'self_attn.out_proj.weight': {
            "name": f'{trtllm_attn_layer_name}.dense.weight',
            "shape": (hidden_size, hidden_size // mapping.tp_size),
            "split_dim": -1
        },
        'fc1.weight': {
            "name": 'mlp.fc.weight',
            "shape": (ffn_hidden_size // mapping.tp_size, hidden_size),
            "split_dim": 0
        },
        'fc1.bias': {
            "name": 'mlp.fc.bias',
            "shape": (ffn_hidden_size // mapping.tp_size),
            "split_dim": 0
        },
        'fc2.weight': {
            "name": 'mlp.proj.weight',
            "shape": (hidden_size, ffn_hidden_size // mapping.tp_size),
            "split_dim": -1
        },
    }

    hidden_layer_name_no_split = {
        'self_attn.out_proj.bias': {
            "name": f'{trtllm_attn_layer_name}.dense.bias',
            "shape": (hidden_size)
        },
        'self_attn_layer_norm.weight': {
            "name": f'{trtllm_attn_layernorm_name}.weight',
            "shape": None
        },
        'self_attn_layer_norm.bias': {
            "name": f'{trtllm_attn_layernorm_name}.bias',
            "shape": None
        },
        'fc2.bias': {
            "name": 'mlp.proj.bias',
            "shape": (hidden_size)
        },
        'final_layer_norm.weight': {
            "name": 'mlp_layernorm.weight',
            "shape": None
        },
        'final_layer_norm.bias': {
            "name": 'mlp_layernorm.bias',
            "shape": None
        },
    }

    if config.model_type == 'mbart':
        hidden_layer_name_split['layer_norm.weight'] = {
            "name": 'transformer.ln_f.weight',
            "shape": None,
            "split_dim": 0
        }
        hidden_layer_name_no_split['layer_norm.bias'] = {
            "name": 'transformer.ln_f.bias',
            "shape": None,
            "split_dim": 0
        }

    if component == "decoder":
        hidden_layer_name_split.update({
            'encoder_attn.out_proj.weight': {
                "name": 'cross_attention.dense.weight',
                "shape": (hidden_size, hidden_size // mapping.tp_size),
                "split_dim": -1
            }
        })
        hidden_layer_name_no_split.update({
            'encoder_attn.out_proj.bias': {
                "name": 'cross_attention.dense.bias',
                "shape": (hidden_size)
            },
            'encoder_attn_layer_norm.weight': {
                "name": 'cross_attention_layernorm.weight',
                "shape": None
            },
            'encoder_attn_layer_norm.bias': {
                "name": 'cross_attention_layernorm.bias',
                "shape": None
            },
        })

    def get_attn_module_name(component, layer, attn_type):
        return f'model.{component}.layers.{int(layer)}.{attn_type}'

    for hf_weight_name, weight_info in embedding_layer_names.items():
        if 'position' in hf_weight_name:
            weights[weight_info["name"]] = params[
                f'{hf_param_prefix}.{hf_weight_name}'][2:].clone()
        else:
            weights[weight_info["name"]] = params[
                f'{hf_param_prefix}.{hf_weight_name}'].clone()
        weights[weight_info["name"]] = reshape(weights[weight_info["name"]],
                                               weight_info["shape"])

    weights["embedding.vocab_embedding.weight"] = reshape(
        params[f'{hf_param_prefix}.embed_tokens.weight'].clone(),
        (vocab_size, -1)) if not config.use_parallel_embedding else reshape(
            split(params[f'{hf_param_prefix}.embed_tokens.weight'].clone(),
                  mapping.tp_size, mapping.tp_rank, 0),
            (vocab_size // mapping.tp_size, -1))

    num_layers = config.num_hidden_layers

    layers_range = mapping.pp_layers(num_layers)
    for layer_idx in layers_range:
        local_layer_idx = layer_idx - layers_range[0]
        hf_layer_name_prefix = f'{hf_param_prefix}.layers.{layer_idx}'
        trtllm_layer_name_prefix = f'{trtllm_layer_name}.{local_layer_idx}'

        for hf_weight_name, weight_info in hidden_layer_name_split.items():
            weights[
                f'{trtllm_layer_name_prefix}.{weight_info["name"]}'] = reshape(
                    split(params[f'{hf_layer_name_prefix}.{hf_weight_name}'],
                          mapping.tp_size,
                          mapping.tp_rank,
                          dim=weight_info["split_dim"]), weight_info["shape"])

        for hf_weight_name, weight_info in hidden_layer_name_no_split.items():
            trtllm_layer_fullname = f'{trtllm_layer_name_prefix}.{weight_info["name"]}'
            hf_layer_fullname = f'{hf_layer_name_prefix}.{hf_weight_name}'
            weights[trtllm_layer_fullname] = reshape(
                params[hf_layer_fullname].clone(), shape=weight_info["shape"])

        self_attn_module_name = get_attn_module_name(component, layer_idx,
                                                     'self_attn')
        weights.update(
            fuse_qkv_one_layer(
                params, self_attn_module_name,
                f'{trtllm_layer_name_prefix}.{trtllm_attn_layer_name}',
                mapping.tp_size, mapping.tp_rank, config.model_type,
                (hidden_size * 3 // mapping.tp_size, hidden_size),
                (hidden_size * 3 // mapping.tp_size)))
        if component == 'decoder':
            cross_attn_module_name = get_attn_module_name(
                component, layer_idx, 'encoder_attn')
            weights.update(
                fuse_qkv_one_layer(
                    params, cross_attn_module_name,
                    f'{trtllm_layer_name_prefix}.cross_attention',
                    mapping.tp_size, mapping.tp_rank, config.model_type,
                    (hidden_size * 3 // mapping.tp_size, hidden_size),
                    (hidden_size * 3 // mapping.tp_size)))

    if component == 'decoder':
        import torch
        lm_head_weights = params['lm_head.weight'].clone().detach()
        vocab_size = config.vocab_size
        if params['lm_head.weight'].shape[0] % mapping.tp_size != 0:
            vocab_size_padded = pad_vocab_size(config.vocab_size,
                                               mapping.tp_size)
            pad_width = vocab_size_padded - config.vocab_size

            lm_head_weights = torch.nn.functional.pad(lm_head_weights,
                                                      (0, 0, 0, pad_width),
                                                      'constant',
                                                      value=0)
            vocab_size = vocab_size_padded
        weights['lm_head.weight'] = reshape(
            split(lm_head_weights, mapping.tp_size, mapping.tp_rank, dim=0),
            (vocab_size // mapping.tp_size, hidden_size))

    if config.has_model_final_layernorm:
        weights['transformer.ln_f.weight'] = params[
            f'{hf_param_prefix}.layer_norm.weight'].clone()
        weights['transformer.ln_f.bias'] = params[
            f'{hf_param_prefix}.layer_norm.bias'].clone()

    return weights


def parse_pix2struct_config(args, hf_model):
    # manually set q_scaling to offset attention scaling's effect.
    # TODO: modify kernels to control whether to disable attention scaling
    config = configparser.ConfigParser()

    def get_offset_q_scaling(config) -> str:
        d_model = config.hidden_size
        num_heads = config.num_heads
        head_size = d_model / num_heads
        scaling = 1 / head_size**.5
        return str(scaling)

    config["decoder"] = {}
    for key, val in hf_model.decoder.config.to_dict().items():
        config["decoder"][key] = f"{val}"

    config["decoder"]["q_scaling"] = get_offset_q_scaling(
        hf_model.decoder.config)

    config["structure"] = dict()
    config["structure"]["pix2struct_with_bias"] = "false"
    config["structure"]["use_gated_activation"] = "false"
    config["structure"]["position_embedding_type"] = "relative"
    config["structure"]["model_type"] = args.model_type

    def parse_pix2struct_config_by_component(config, component, args):
        if component == 'decoder':
            args.n_layer = config.getint(component, 'num_layers')
            args.n_head = config.getint(component, 'num_heads')
            args.head_size = config.getint(component, 'd_kv')
            args.hidden_size = config.getint(component, 'hidden_size')
            args.ffn_hidden_size = config.getint(component, 'd_ff')
            args.vocab_size = config.getint(component, 'vocab_size')
            args.n_positions = config.getint(component,
                                             'n_positions',
                                             fallback=512)
            args.has_position_embedding = config.getboolean(
                component, 'has_position_embedding',
                fallback=False)  # TODO: hardcoded here
            args.has_token_type_embedding = config.getboolean(
                component, 'has_token_type_embedding', fallback=False)
            args.has_embedding_layernorm = config.getboolean(
                component, 'has_embedding_layernorm', fallback=False)
            args.has_embedding_scale = config.getboolean(component,
                                                         'has_embedding_scale',
                                                         fallback=False)
            args.q_scaling = config.getfloat(component,
                                             'q_scaling',
                                             fallback=1.0)
            args.has_attention_qkvo_bias = config.getboolean(
                component, 'has_attention_qkvo_bias', fallback=False)
            args.has_mlp_bias = config.getboolean(component,
                                                  'has_mlp_bias',
                                                  fallback=False)
            args.has_model_final_layernorm = config.getboolean(
                component, 'has_model_final_layernorm', fallback=True)
            args.layernorm_eps = config.getfloat(component,
                                                 'layer_norm_epsilon')
            args.layernorm_position = layernorm_position_map[config.get(
                component, 'layernorm_position',
                fallback='pre_layernorm')]  # TODO: hardcoded here
            args.layernorm_type = layernorm_type_map[config.get(
                component, 'layernorm_type', fallback='RmsNorm')]
            args.hidden_act = config.get(component, 'dense_act_fn')
            args.gated_act = True
            args.mlp_type = mlp_type_map['GatedMLP' if args.
                                         gated_act else 'MLP']
            args.has_lm_head_bias = config.getboolean(
                component,  # TODO: T5 with bias
                'has_lm_head_bias',
                fallback=False)
            args.relative_attention = config.getboolean(component,
                                                        'relative_attention',
                                                        fallback=True)
            args.num_buckets = config.getint(component,
                                             'relative_attention_num_buckets')
            args.max_distance = config.getint(
                component, 'relative_attention_max_distance')
            args.logits_dtype = config.get(component,
                                           'logits_dtype',
                                           fallback='float32')
            args.rescale_before_lm_head = config.getboolean(
                component, 'tie_word_embeddings'
            )  # default is True (for T5), but False for Flan-T5
            args.encoder_hidden_size = config.getint('decoder', 'hidden_size')
            args.encoder_num_heads = config.getint('decoder', 'num_heads')
            args.encoder_head_size = config.getint('decoder', 'd_kv')
            args.position_embedding_type = config.get(
                'structure', 'position_embedding_type')
            args.decoder_start_token_id = config.getint(
                'decoder', 'decoder_start_token_id')
            args.eos_token_id = config.getint('decoder', 'eos_token_id')
            bos_token_id = config.get('decoder', 'bos_token_id')
            # pix2struct does not have bos_token_id
            args.bos_token_id = int(
                bos_token_id) if bos_token_id != "None" else None
            args.pad_token_id = config.getint('decoder', 'pad_token_id')

        else:
            assert False, 'Unsupported component!'
        return args

    decoder_args = parse_pix2struct_config_by_component(config, "decoder", args)
    return None, decoder_args


def convert_pix2struct_weights_to_tllm_safetensors(config, component, params):
    weights = {}

    mapping = config.mapping

    convert_weight_to_dtype(params, config.dtype)
    hidden_size = config.hidden_size
    ffn_hidden_size = config.intermediate_size
    num_layers = config.num_hidden_layers
    n_head = config.num_attention_heads
    head_size = config.head_size
    attention_hidden_size = n_head * head_size  # head size * num_heads not necessarily equals hidden_dim, such as Flan-T5

    hf_param_prefix = f'{component}'
    trtllm_layer_name = f'transformer.layers'
    trtllm_attn_layer_name = 'self_attention'
    trtllm_attn_layernorm_name = 'self_attention_layernorm'

    def get_attn_module_name(component, layer, attn_type):
        return f'{component}.layer.{int(layer)}.{attn_type}.attention'

    weights['transformer.vocab_embedding.weight'] = reshape(
        params[f'{hf_param_prefix}.embed_tokens.weight'].clone(),
        None) if not config.use_parallel_embedding else reshape(
            split(params[f'{hf_param_prefix}.embed_tokens.weight'].clone(),
                  mapping.tp_size, mapping.tp_rank, 0), None)

    layers_range = mapping.pp_layers(num_layers)
    for layer_idx in layers_range:
        local_layer_idx = layer_idx - layers_range[0]
        trtllm_layer_name_prefix = f'{trtllm_layer_name}.{local_layer_idx}'
        hf_layer_name_prefix = f'{hf_param_prefix}.layer.{layer_idx}'

        hidden_layer_name_split = {
            f'{hf_layer_name_prefix}.self_attention.attention.output.weight': {
                "name":
                f'{trtllm_layer_name_prefix}.{trtllm_attn_layer_name}.dense.weight',
                "shape":
                (hidden_size, attention_hidden_size // mapping.tp_size),
                "split_dim": -1
            },
            f'{hf_layer_name_prefix}.mlp.DenseReluDense.wo.weight': {
                "name": f'{trtllm_layer_name_prefix}.mlp.proj.weight',
                "shape": (hidden_size, ffn_hidden_size // mapping.tp_size),
                "split_dim": -1
            },
            f'{hf_layer_name_prefix}.mlp.DenseReluDense.wi_0.weight': {
                "name": f'{trtllm_layer_name_prefix}.mlp.fc.weight',
                "shape": (ffn_hidden_size // mapping.tp_size, hidden_size),
                "split_dim": 0
            },
        }

        hidden_layer_name_no_split = {
            f'{hf_layer_name_prefix}.self_attention.layer_norm.weight': {
                "name":
                f'{trtllm_layer_name_prefix}.{trtllm_attn_layernorm_name}.weight',
                "shape": None
            },
            f'{hf_layer_name_prefix}.mlp.layer_norm.weight': {
                "name": f'{trtllm_layer_name_prefix}.mlp_layernorm.weight',
                "shape": None
            },
        }

        if config.gated_act:
            hidden_layer_name_split.update({
                f'{hf_layer_name_prefix}.mlp.DenseReluDense.wi_1.weight': {
                    "name": f'{trtllm_layer_name_prefix}.mlp.gate.weight',
                    "shape": (ffn_hidden_size // mapping.tp_size, hidden_size),
                    "split_dim": 0
                },
            })

        hidden_layer_name_split.update({
            f'{hf_layer_name_prefix}.encoder_decoder_attention.attention.output.weight':
            {
                "name":
                f'{trtllm_layer_name_prefix}.cross_attention.dense.weight',
                "shape":
                (hidden_size, attention_hidden_size // mapping.tp_size),
                "split_dim": -1
            },
        })
        hidden_layer_name_no_split.update({
            f'{hf_layer_name_prefix}.encoder_decoder_attention.layer_norm.weight':
            {
                "name":
                f'{trtllm_layer_name_prefix}.cross_attention_layernorm.weight',
                "shape": None
            },
        })
        self_attn_module_name = get_attn_module_name(
            component, layer_idx, 'encoder_decoder_attention')
        weights.update(
            fuse_qkv_one_layer(
                params, self_attn_module_name,
                f'{trtllm_layer_name_prefix}.cross_attention', mapping.tp_size,
                mapping.tp_rank, config.model_type,
                (attention_hidden_size * 3 // mapping.tp_size, hidden_size),
                None))

        self_attn_module_name = get_attn_module_name(component, layer_idx,
                                                     'self_attention')
        weights.update(
            fuse_qkv_one_layer(
                params, self_attn_module_name,
                f'{trtllm_layer_name_prefix}.{trtllm_attn_layer_name}',
                mapping.tp_size, mapping.tp_rank, config.model_type,
                (attention_hidden_size * 3 // mapping.tp_size, hidden_size),
                None))

        weights[
            f'{trtllm_layer_name_prefix}.{trtllm_attn_layer_name}.rel_attn_table'] = reshape(
                split(
                    params[
                        f'{component}.layer.0.self_attention.attention.relative_attention_bias.weight']
                    .T, mapping.tp_size, mapping.tp_rank, 0),
                (n_head // mapping.tp_size, config.num_buckets))

        for hf_weight_name, weight_info in hidden_layer_name_split.items():
            if hf_weight_name in params.keys():
                weights[weight_info["name"]] = reshape(
                    split(params[hf_weight_name],
                          mapping.tp_size,
                          mapping.tp_rank,
                          dim=weight_info["split_dim"]), weight_info["shape"])
        for hf_weight_name, weight_info in hidden_layer_name_no_split.items():
            if hf_weight_name in params.keys():
                weights[weight_info["name"]] = reshape(
                    params[hf_weight_name].clone(), shape=weight_info["shape"])

    weights[f'transformer.ln_f.weight'] = reshape(
        params[f'{component}.final_layer_norm.weight'].clone(), None)

    weights['lm_head.weight'] = reshape(
        split(params[f'{component}.lm_head.weight'],
              mapping.tp_size,
              mapping.tp_rank,
              dim=0), (config.vocab_size // mapping.tp_size, hidden_size))
    if not config.use_implicit_relative_attention:
        weights[f'rel_attn_table'] = reshape(
            split(
                params[
                    f'{component}.layer.0.self_attention.attention.relative_attention_bias.weight']
                .T, mapping.tp_size, mapping.tp_rank, 0),
            (n_head // mapping.tp_size, config.num_buckets))

    return weights


def parse_language_adapter_config(args, model):
    config = configparser.ConfigParser()
    config.read(args.model_dir + "/config.ini")
    # rename from "sinusoid" to "learned_absolute"
    config["structure"]["position_embedding_type"] = "learned_absolute"

    locale_list = ['de_DE', 'en_US', 'es_ES', 'fr_FR']  # to be changed by user

    encoder_config = parse_nmt_config_by_component(config, "encoder", args)
    encoder_config.residual_scaling = config.getfloat("encoder",
                                                      'residual_scaling',
                                                      fallback=1.0)
    encoder_config.language_adapter_config = LanguageAdapterConfig(
        num_languages=config.getint("encoder", 'adapter_langs', fallback=None),
        ffn_hidden_size=config.getint("encoder",
                                      'encoder_adapter_embed_dim',
                                      fallback=None),
        language_list=locale_list).to_dict()

    decoder_config = parse_nmt_config_by_component(config, "decoder", args)
    decoder_config.residual_scaling = config.getfloat("decoder",
                                                      'residual_scaling',
                                                      fallback=1.0)
    decoder_config.language_adapter_config = LanguageAdapterConfig(
        num_languages=config.getint("decoder", 'adapter_langs', fallback=None),
        ffn_hidden_size=config.getint("decoder",
                                      'decoder_adapter_embed_dim',
                                      fallback=None),
        language_list=locale_list).to_dict()

    decoder_config.decoder_start_token_id = 2
    decoder_config.eos_token_id = 2
    decoder_config.bos_token_id = 2
    decoder_config.pad_token_id = 0

    return encoder_config, decoder_config


def convert_language_adapter_weights_to_tllm_safetensors(
        config, component, params):
    weights = {}

    mapping = config.mapping

    convert_weight_to_dtype(params, config.dtype)

    param_prefix = f'{component}'
    trtllm_layer_name = f'transformer.layers'
    trtllm_attn_layer_name = 'attention' if component == 'encoder' else 'self_attention'
    trtllm_attn_layernorm_name = 'self_attention_layernorm' if component == 'decoder' else 'attention_layernorm'
    mlp_param_prefix = '' if f'{param_prefix}.0.fc1.weight' in params else 'mlp.'

    hidden_size = params[
        f'{param_prefix}.layers.0.self_attn.out_proj.weight'].shape[0]
    ffn_hidden_size = params[
        f'{param_prefix}.layers.0.{mlp_param_prefix}fc1.weight'].shape[0]

    hidden_layer_name_split = {
        'self_attn.out_proj.weight': {
            "name": f'{trtllm_attn_layer_name}.dense.weight',
            "shape": (hidden_size, hidden_size // mapping.tp_size),
            "split_dim": -1
        },
        f'{mlp_param_prefix}fc1.weight': {
            "name": 'mlp.fc.weight',
            "shape": (ffn_hidden_size // mapping.tp_size, hidden_size),
            "split_dim": 0
        },
        f'{mlp_param_prefix}fc1.bias': {
            "name": 'mlp.fc.bias',
            "shape": (ffn_hidden_size // mapping.tp_size),
            "split_dim": 0
        },
        f'{mlp_param_prefix}fc2.weight': {
            "name": 'mlp.proj.weight',
            "shape": (hidden_size, ffn_hidden_size // mapping.tp_size),
            "split_dim": -1
        },
    }

    hidden_layer_name_no_split = {
        'self_attn.out_proj.bias': {
            "name": f'{trtllm_attn_layer_name}.dense.bias',
            "shape": (hidden_size)
        },
        'self_attn_layer_norm.weight': {
            "name": f'{trtllm_attn_layernorm_name}.weight',
            "shape": None
        },
        'self_attn_layer_norm.bias': {
            "name": f'{trtllm_attn_layernorm_name}.bias',
            "shape": None
        },
        f'{mlp_param_prefix}fc2.bias': {
            "name": 'mlp.proj.bias',
            "shape": (hidden_size)
        },
        'final_layer_norm.weight': {
            "name": 'mlp_layernorm.weight',
            "shape": None
        },
        'final_layer_norm.bias': {
            "name": 'mlp_layernorm.bias',
            "shape": None
        },
    }

    if component == "decoder":
        hidden_layer_name_split.update({
            'encoder_attn.out_proj.weight': {
                "name": 'cross_attention.dense.weight',
                "shape": (hidden_size, hidden_size // mapping.tp_size),
                "split_dim": -1
            },
        })
        hidden_layer_name_no_split.update({
            'encoder_attn.out_proj.bias': {
                "name": 'cross_attention.dense.bias',
                "shape": (hidden_size)
            },
            'encoder_attn_layer_norm.weight': {
                "name": 'cross_attention_layernorm.weight',
                "shape": None,
            },
            'encoder_attn_layer_norm.bias': {
                "name": 'cross_attention_layernorm.bias',
                "shape": None
            },
        })

    def get_attn_module_name(layer, attn_type):
        return f'{param_prefix}.layers.{int(layer)}.{attn_type}'

    # support MostlyFreezedEmbedding in 5.5B model
    embed_tokens_weight_name = f'{param_prefix}.embed_tokens.weight'
    if embed_tokens_weight_name not in params:
        embed_tokens_weight_name = f'{param_prefix}.embed_tokens.weight_'

    weights['transformer.vocab_embedding.weight'] = reshape(
        params[embed_tokens_weight_name].clone(),
        None) if not config.use_parallel_embedding else reshape(
            split(params[embed_tokens_weight_name].clone(), mapping.tp_size,
                  mapping.tp_rank, 0), None)

    weights[
        'transformer.position_embedding.weight'] = fairseq_sin_pos_embedding(
            config.max_position_embeddings,
            params[embed_tokens_weight_name].shape[1])

    num_layers = config.num_hidden_layers
    layers_range = mapping.pp_layers(num_layers)

    for layer_idx in layers_range:
        local_layer_idx = layer_idx - layers_range[0]
        hf_layer_name_prefix = f'{param_prefix}.layers.{layer_idx}'
        trtllm_layer_name_prefix = f'{trtllm_layer_name}.{local_layer_idx}'

        for hf_weight_name, weight_info in hidden_layer_name_split.items():
            weights[
                f'{trtllm_layer_name_prefix}.{weight_info["name"]}'] = reshape(
                    split(params[f'{hf_layer_name_prefix}.{hf_weight_name}'],
                          mapping.tp_size,
                          mapping.tp_rank,
                          dim=weight_info["split_dim"]), weight_info["shape"])

        for hf_weight_name, weight_info in hidden_layer_name_no_split.items():
            trtllm_layer_fullname = f'{trtllm_layer_name_prefix}.{weight_info["name"]}'
            hf_layer_fullname = f'{hf_layer_name_prefix}.{hf_weight_name}'
            weights[trtllm_layer_fullname] = reshape(
                params[hf_layer_fullname].clone(), shape=weight_info["shape"])

        self_attn_module_name = get_attn_module_name(layer_idx, 'self_attn')
        weights.update(
            fuse_qkv_one_layer(
                params, self_attn_module_name,
                f'{trtllm_layer_name_prefix}.{trtllm_attn_layer_name}',
                mapping.tp_size, mapping.tp_rank, config.model_type,
                (hidden_size * 3 // mapping.tp_size, hidden_size),
                (hidden_size * 3 // mapping.tp_size)))

        if component == 'decoder':
            cross_attn_module_name = get_attn_module_name(
                layer_idx, 'encoder_attn')
            weights.update(
                fuse_qkv_one_layer(
                    params, cross_attn_module_name,
                    f'{trtllm_layer_name_prefix}.cross_attention',
                    mapping.tp_size, mapping.tp_rank, config.model_type,
                    (hidden_size * 3 // mapping.tp_size, hidden_size),
                    (hidden_size * 3 // mapping.tp_size)))
        assert len(config.language_adapter_config['language_list']) > 0

        language_adapter_weights = defaultdict(list)
        language_adapter_weight_info = {
            'fc1.weight': {
                "name": f'{trtllm_layer_name_prefix}.adapter.layers.fc.weight',
                "shape": None
            },
            'fc1.bias': {
                "name": f'{trtllm_layer_name_prefix}.adapter.layers.fc.bias',
                "shape": None
            },
            'fc2.weight': {
                "name":
                f'{trtllm_layer_name_prefix}.adapter.layers.proj.weight',
                "shape": None
            },
            'fc2.bias': {
                "name": f'{trtllm_layer_name_prefix}.adapter.layers.proj.bias',
                "shape": None
            },
        }

        for language in config.language_adapter_config['language_list']:
            for key in language_adapter_weight_info.keys():
                language_adapter_weights[key].append(params[
                    f'{param_prefix}.layers.{layer_idx}.adapter.{language}.{key}']
                                                     .unsqueeze(0))

        import torch
        for key, weight_info in language_adapter_weight_info.items():
            weights[weight_info["name"]] = torch.cat(
                language_adapter_weights[key], dim=0)

        weights[
            f'{trtllm_layer_name_prefix}.adapter_layer_norm.weight'] = params[
                f'{param_prefix}.layers.{layer_idx}.adapter_layer_norm.weight']
        weights[f'{trtllm_layer_name_prefix}.adapter_layer_norm.bias'] = params[
            f'{param_prefix}.layers.{layer_idx}.adapter_layer_norm.bias']

    if component == 'decoder':

        # share_decoder_input_output_embed=True, output_proj = embed_tokens.transpose()
        lm_head_weight_name = f'{param_prefix}.output_projection.weight'
        if lm_head_weight_name not in params:
            lm_head_weight_name = embed_tokens_weight_name
        weights['lm_head.weight'] = reshape(
            split(params[lm_head_weight_name],
                  mapping.tp_size,
                  mapping.tp_rank,
                  dim=0), (config.vocab_size // mapping.tp_size, hidden_size))

    return weights


def get_model(args):
    if args.model_type == "t5":
        model = T5ForConditionalGeneration.from_pretrained(args.model_dir)
    elif args.model_type == "nmt":
        from fairseq.models.transformer import TransformerModel
        model = TransformerModel.from_pretrained(args.model_dir)
    elif args.model_type == "bart":
        if args.nougat:
            model = VisionEncoderDecoderModel.from_pretrained(args.model_dir)
            model = model.get_decoder()
        elif args.eclair_radio:
            import torch

            class RadioWithNeck(torch.nn.Module):

                def __init__(self):
                    super().__init__()

                    self.model_encoder = torch.hub.load("NVlabs/RADIO",
                                                        "radio_model",
                                                        version="radio_v2.5-h")
                    self.model_encoder.summary_idxs = torch.tensor(4)

                    self.conv1 = torch.nn.Conv1d(1280, 1024, 1)
                    self.layer_norm1 = torch.nn.LayerNorm(
                        1024, eps=1e-6, elementwise_affine=True)
                    self.conv2 = torch.nn.Conv2d(1024,
                                                 1024,
                                                 kernel_size=(1, 4),
                                                 stride=(1, 4),
                                                 padding=0,
                                                 bias=False)
                    self.layer_norm2 = torch.nn.LayerNorm(
                        1024, eps=1e-6, elementwise_affine=True)

                def forward(self, pixel_values):
                    _, feature = self.model_encoder(pixel_values)
                    output = self.conv1(feature.permute(0, 2,
                                                        1)).permute(0, 2, 1)
                    output = self.layer_norm1(output).permute(0, 2, 1)

                    b, d, _ = output.shape
                    h = pixel_values.shape[-2] // 16
                    w = pixel_values.shape[-1] // 16
                    output = self.conv2(output.reshape(b, d, h, w))
                    output = output.flatten(-2, -1).permute(0, 2, 1)
                    output = self.layer_norm2(output)
                    return output

            def get_processor():
                processor = NougatProcessor.from_pretrained(
                    "facebook/nougat-base")

                special_tokens = {
                    "output_plain_index": "<output_plain>",
                    "output_markdown_index": "<output_markdown>",
                    "output_no_text_index": "<output_no_text>",
                    "output_ocr_index": "<output_ocr>",
                    "predict_bbox_index": "<predict_bbox>",
                    "no_bbox_index": "<no_bbox>",
                    "bbox_start_index": "<bbox>",  # not used but can keep
                    # "bbox_end_index": "</bbox>",  # not used but can keep
                    "no_class_index": "<no_classes>",
                    "predict_classes_index": "<predict_classes>",
                }
                for key, special_t in special_tokens.items():
                    processor.tokenizer.add_special_tokens(
                        {"additional_special_tokens": [special_t]})
                    setattr(processor.tokenizer, key,
                            processor.tokenizer.encode(special_t)[1])

                # Add regular tokens for boxes
                processor.tokenizer.add_tokens(
                    [f"<x_{x_i}>" for x_i in range(1024)])
                processor.tokenizer.add_tokens(
                    [f"<y_{y_i}>" for y_i in range(1280)])
                # Add regular tokens for classes
                #"<class_{class_i}>"
                possible_classes = [
                    "Text", "Title", "Section-header", "List-item", "TOC",
                    "Bibliography", "Footnote", "Page-header", "Page-footer",
                    "Picture", "Formula", "Page-number", "Table", "Caption"
                ]
                processor.tokenizer.add_tokens(
                    [f"<class_{cls}>" for cls in possible_classes])
                return processor

            processor = get_processor()
            model = VisionEncoderDecoderModel.from_pretrained(
                "facebook/nougat-base")
            model.encoder = RadioWithNeck()
            model.decoder.resize_token_embeddings(len(processor.tokenizer),
                                                  pad_to_multiple_of=64)
            model.config.decoder_start_token_id = processor.tokenizer.eos_token_id  # 2
            model.config.pad_token_id = processor.tokenizer.pad_token_id  # 1
            from transformers.models.mbart.modeling_mbart import \
                MBartLearnedPositionalEmbedding
            _, d_model = model.device, model.config.decoder.d_model

            with torch.inference_mode():
                # Inspect checkpoint shapes
                safetensors.torch.load_model(model,
                                             os.path.join(
                                                 args.model_dir,
                                                 "model.safetensors"),
                                             strict=False)
            model.decoder.model.decoder.embed_positions = MBartLearnedPositionalEmbedding(
                ECLAIR_RADIO_MAX_POSITION_EMBEDDINGS, d_model)
            model.decoder.model.decoder.embed_positions.weight.data.zero_()
            model.decoder.model.decoder.embed_positions.weight.requires_grad_(
                True)
            model.decoder.lm_head.weight = model.decoder.get_input_embeddings(
            ).weight

            model.eval()
            model = model.get_decoder()

        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    elif args.model_type == "pix2struct":
        model = Pix2StructForConditionalGeneration.from_pretrained(
            args.model_dir)
    elif args.model_type == "blip2":
        model = Blip2ForConditionalGeneration.from_pretrained(
            args.model_dir).language_model
    elif args.model_type == "language_adapter":
        import torch

        class DummyTorchModel:

            def __init__(self, model) -> None:
                self.model = model

            def state_dict(self):
                return self.model['model']

        model = torch.load(args.model_dir + "/model.pt", weights_only=False)
        return DummyTorchModel(model)

    return model


def convert_checkpoint(args):

    model = get_model(args)

    saved_dir = Path(args.output_dir)
    saved_dir.mkdir(parents=True, exist_ok=True)

    encoder_saved_dir = saved_dir / "encoder"
    encoder_saved_dir.mkdir(parents=True, exist_ok=True)
    decoder_saved_dir = saved_dir / "decoder"
    decoder_saved_dir.mkdir(parents=True, exist_ok=True)

    world_size = args.tp_size * args.pp_size

    kv_cache_quant_algo = None
    quant_algo = None

    model_type = args.model_type if args.model_type != "blip2" else "t5"
    parse_config_mapper = {
        't5': parse_t5_config,
        'pix2struct': parse_pix2struct_config,
        'blip2': parse_t5_config,  # blip2 uses t5 config parser
        'language_adapter': parse_language_adapter_config,
        'nmt': parse_nmt_config,
        'bart': parse_bart_config,
    }
    encoder_config, decoder_config = parse_config_mapper[model_type](args,
                                                                     model)

    additional_settings = ["gated_act"]
    if model_type == 'language_adapter':
        additional_settings += ["residual_scaling", "language_adapter_config"]

    if not (args.nougat
            or args.eclair_radio) and args.model_type != "pix2struct":
        tllm_encoder_config = {
            'architecture': "EncoderModel",
            'dtype': args.dtype,
            'logits_dtype': encoder_config.logits_dtype,
            'num_hidden_layers': encoder_config.n_layer,
            'num_attention_heads': encoder_config.n_head,
            'hidden_size': encoder_config.hidden_size,
            'norm_epsilon': encoder_config.layernorm_eps,
            'vocab_size': encoder_config.vocab_size,
            'position_embedding_type': encoder_config.position_embedding_type,
            'hidden_act': encoder_config.hidden_act,
            'quantization': {
                'quant_algo': quant_algo,
                'kv_cache_quant_algo': kv_cache_quant_algo,
            },
            'mapping': {
                'world_size': world_size,
                'tp_size': args.tp_size,
                'pp_size': args.pp_size,
            },
            'use_parallel_embedding': args.use_parallel_embedding,
            'embedding_sharding_dim': args.embedding_sharding_dim,
            'max_position_embeddings': encoder_config.n_positions,
            'num_key_value_heads': encoder_config.n_head,
            'head_size': encoder_config.head_size,
            'has_position_embedding': encoder_config.has_position_embedding,
            'layernorm_type': encoder_config.layernorm_type,
            'has_attention_qkvo_bias': encoder_config.has_attention_qkvo_bias,
            'has_mlp_bias': encoder_config.has_mlp_bias,
            'has_model_final_layernorm':
            encoder_config.has_model_final_layernorm,
            'has_embedding_layernorm': encoder_config.has_embedding_layernorm,
            'has_embedding_scale': encoder_config.has_embedding_scale,
            'intermediate_size': encoder_config.ffn_hidden_size,
            'q_scaling': encoder_config.q_scaling,
            'layernorm_position': encoder_config.layernorm_position,
            'mlp_type': encoder_config.mlp_type,
            'relative_attention': encoder_config.relative_attention,
            'max_distance': encoder_config.max_distance,
            'num_buckets': encoder_config.num_buckets,
            'model_type': encoder_config.model_type,
        }

        for additional_setting in additional_settings:
            if hasattr(encoder_config, additional_setting):
                tllm_encoder_config.update({
                    additional_setting:
                    getattr(encoder_config, additional_setting)
                })

        with (encoder_saved_dir / "config.json").open('w') as f:
            json.dump(tllm_encoder_config, f, indent=4)

        encoder_convert_args = dict(params=model.state_dict(),
                                    component="encoder")
    tllm_decoder_config = {
        'architecture': "DecoderModel",
        'dtype': args.dtype,
        'logits_dtype': decoder_config.logits_dtype,
        'num_hidden_layers': decoder_config.n_layer,
        'num_attention_heads': decoder_config.n_head,
        'hidden_size': decoder_config.hidden_size,
        'norm_epsilon': decoder_config.layernorm_eps,
        'vocab_size': decoder_config.vocab_size,
        'position_embedding_type': decoder_config.position_embedding_type,
        'hidden_act': decoder_config.hidden_act,
        'quantization': {
            'quant_algo': quant_algo,
            'kv_cache_quant_algo': kv_cache_quant_algo,
        },
        'mapping': {
            'world_size': world_size,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size,
        },
        'use_parallel_embedding': args.use_parallel_embedding,
        'embedding_sharding_dim': args.embedding_sharding_dim,
        'max_position_embeddings': decoder_config.n_positions,
        'head_size': decoder_config.head_size,
        'has_position_embedding': decoder_config.has_position_embedding,
        'layernorm_type': decoder_config.layernorm_type,
        'has_attention_qkvo_bias': decoder_config.has_attention_qkvo_bias,
        'has_mlp_bias': decoder_config.has_mlp_bias,
        'has_model_final_layernorm': decoder_config.has_model_final_layernorm,
        'has_embedding_layernorm': decoder_config.has_embedding_layernorm,
        'has_embedding_scale': decoder_config.has_embedding_scale,
        'intermediate_size': decoder_config.ffn_hidden_size,
        'q_scaling': decoder_config.q_scaling,
        'layernorm_position': decoder_config.layernorm_position,
        'mlp_type': decoder_config.mlp_type,
        'relative_attention': decoder_config.relative_attention,
        'max_distance': decoder_config.max_distance,
        'num_buckets': decoder_config.num_buckets,
        'model_type': decoder_config.model_type,
        'rescale_before_lm_head': decoder_config.rescale_before_lm_head,
        'encoder_hidden_size': decoder_config.encoder_hidden_size,
        'encoder_num_heads': decoder_config.encoder_num_heads,
        'encoder_head_size': decoder_config.encoder_head_size,
        'skip_cross_kv': args.skip_cross_kv,
        'use_implicit_relative_attention': args.use_implicit_relative_attention,
        'decoder_start_token_id': decoder_config.decoder_start_token_id,
        'eos_token_id': decoder_config.eos_token_id,
        'bos_token_id': decoder_config.bos_token_id,
        'pad_token_id': decoder_config.pad_token_id,
    }
    for additional_setting in additional_settings:
        if hasattr(decoder_config, additional_setting):
            tllm_decoder_config.update({
                additional_setting:
                getattr(decoder_config, additional_setting)
            })

    with (decoder_saved_dir / "config.json").open('w') as f:
        json.dump(tllm_decoder_config, f, indent=4)

    decoder_convert_args = dict(params=model.state_dict(), component="decoder")

    if args.model_type == "nmt":
        fairseq_config = vars(model.cfg.model)  # Namespace --> dict
        num_embeddings = fairseq_config['max_source_positions']
        embedding_dim = fairseq_config['encoder_embed_dim']
        padding_idx = model.models[0].encoder.embed_tokens.padding_idx  # 1

        sin_pos_embedding = model.models[
            0].encoder.embed_positions.get_embedding(
                padding_idx + 1 + num_embeddings,
                embedding_dim,
                padding_idx=padding_idx)  # [2 + num_embeddings, embed_dim]
        sin_pos_embedding = sin_pos_embedding[2:, :]  # remove offset embeddings

        encoder_convert_args["sin_pos_embedding"] = sin_pos_embedding
        decoder_convert_args["sin_pos_embedding"] = sin_pos_embedding

    if args.workers == 1:
        if not (args.nougat
                or args.eclair_radio) and args.model_type != "pix2struct":
            convert(0, world_size, args, tllm_encoder_config,
                    encoder_convert_args, encoder_saved_dir)
        convert(0, world_size, args, tllm_decoder_config, decoder_convert_args,
                decoder_saved_dir)
    else:
        if args.workers > world_size:
            args.workers = world_size
        LOGGER.info(f'Convert checkpoint using {args.workers} workers.')
        import torch.multiprocessing as mp
        if not (args.nougat
                or args.eclair_radio) and args.model_type != "pix2struct":
            mp.spawn(convert,
                     nprocs=args.workers,
                     args=(world_size, args, tllm_encoder_config,
                           encoder_convert_args, encoder_saved_dir))
        mp.spawn(convert,
                 nprocs=args.workers,
                 args=(world_size, args, tllm_decoder_config,
                       decoder_convert_args, decoder_saved_dir))


def convert(worker_rank, world_size, args, model_config, convert_args,
            saved_dir):
    for rank in range(worker_rank, world_size, args.workers):
        rank_config = copy.deepcopy(PretrainedConfig.from_dict(model_config))
        rank_config.set_rank(rank)
        weights = globals(
        )[f'convert_{rank_config.model_type}_weights_to_tllm_safetensors'](
            config=rank_config, **convert_args)
        safetensors.torch.save_file(weights,
                                    f'{saved_dir}/rank{rank}.safetensors')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '--model_type',
        type=str,
        default='t5',
        choices=[
            't5', 'nmt', 'bart', 'pix2struct', 'blip2', 'language_adapter'
        ],
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
    parser.add_argument("--model_dir",
                        "-i",
                        type=str,
                        help="Path to the framework checkpoint file",
                        required=True)
    parser.add_argument("--output_dir",
                        "-o",
                        type=str,
                        help="Path to the converted TRT-LLM model weight file",
                        required=True)
    parser.add_argument(
        "--workers",
        type=int,
        help="How many workers to spawn for conversion (default: 4)",
        default=4)
    parser.add_argument("--nougat",
                        action="store_true",
                        help="Model which uses vision encoder + mbart decoder")
    parser.add_argument("--eclair_radio",
                        action="store_true",
                        help="Model which uses vision encoder + mbart decoder")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Provide verbose messages")
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
        'Note: embedding sharding is only enabled when embedding_sharding_dim = 0'
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
        '--dtype',
        type=str,
        default='float16',
        choices=['float16', 'float32', 'bfloat16'],
        help=
        'Target inference dtype. Weights and Computation will be in this dtype, no matter what original dtype the weight checkpoint has.'
    )
    parser.add_argument(
        '--skip_cross_kv',
        action='store_true',
        help=
        'Skip redundant cross qkv computation by using TensorRT IfConditional switch (experimental).'
    )
    parser.add_argument(
        '--use_implicit_relative_attention',
        action='store_true',
        help=
        'Compute relative attention bias on the fly instead of pre-compute a relative attention bias table.'
    )
    args = parser.parse_args()
    log_format = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format=log_format)
    LOGGER.info("\n=============== Argument ===============")
    for key in vars(args):
        LOGGER.info(f"{key}: {vars(args)[key]}")
    LOGGER.info("========================================")

    start_time = datetime.now()
    convert_checkpoint(args)
    stop_time = datetime.now()
    run_time = (stop_time - start_time)
    LOGGER.info("Spend {} (h:m:s) to convert the model".format(run_time))

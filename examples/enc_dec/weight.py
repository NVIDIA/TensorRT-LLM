import configparser

import numpy as np
import torch

from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy
from tensorrt_llm.functional import LayerNormPositionType, LayerNormType

layernorm_type_map = {i.name: i.value for i in LayerNormType}
layernorm_position_map = {i.name: i.value for i in LayerNormPositionType}


def parse_config(ini_file, component, args):
    config = configparser.ConfigParser()
    config.read(ini_file)
    if component == 'encoder':
        args.n_layer = config.getint(component, 'n_layer')
        args.n_head = config.getint(component, 'n_head')
        args.hidden_size = config.getint(component, 'hidden_size')
        args.ffn_hidden_size = config.getint(component, 'ffn_hidden_size')
        args.vocab_size = config.getint(component, 'vocab_size')
        args.n_positions = config.getint(component, 'n_positions')
        args.has_position_embedding = config.getboolean(
            component, 'has_position_embedding', fallback=False)
        args.has_token_type_embedding = config.getboolean(
            component, 'has_token_type_embedding', fallback=False)
        args.has_embedding_layernorm = config.getboolean(
            component, 'has_embedding_layernorm', fallback=False)
        args.has_embedding_scale = config.getboolean(component,
                                                     'has_embedding_scale',
                                                     fallback=False)
        args.q_scaling = config.getfloat(component, 'q_scaling', fallback=1.0)
        args.has_attention_qkvo_bias = config.getboolean(
            component, 'has_attention_qkvo_bias', fallback=False)
        args.has_mlp_bias = config.getboolean(component,
                                              'has_mlp_bias',
                                              fallback=False)
        args.has_model_final_layernorm = config.getboolean(
            component, 'has_model_final_layernorm', fallback=False)
        args.layernorm_eps = config.getfloat(component,
                                             'layernorm_eps',
                                             fallback=1e-5)
        args.layernorm_position = layernorm_position_map[config.get(
            component, 'layernorm_position')]
        args.layernorm_type = layernorm_type_map[config.get(
            component, 'layernorm_type')]
        args.hidden_act = config.get(component, 'hidden_act')
        args.relative_attention = config.getboolean(component,
                                                    'relative_attention',
                                                    fallback=False)
        args.num_buckets = config.getint(component, 'num_buckets')
        args.max_distance = config.getint(component, 'max_distance')

    elif component == 'decoder':
        args.n_layer = config.getint(component, 'n_layer')
        args.n_head = config.getint(component, 'n_head')
        args.hidden_size = config.getint(component, 'hidden_size')
        args.ffn_hidden_size = config.getint(component, 'ffn_hidden_size')
        args.vocab_size = config.getint(component, 'vocab_size')
        args.n_positions = config.getint(component, 'n_positions')
        args.has_position_embedding = config.getboolean(
            component, 'has_position_embedding', fallback=False)
        args.has_token_type_embedding = config.getboolean(
            component, 'has_token_type_embedding', fallback=False)
        args.has_embedding_layernorm = config.getboolean(
            component, 'has_embedding_layernorm', fallback=False)
        args.has_embedding_scale = config.getboolean(component,
                                                     'has_embedding_scale',
                                                     fallback=False)
        args.q_scaling = config.getfloat(component, 'q_scaling', fallback=1.0)
        args.has_attention_qkvo_bias = config.getboolean(
            component, 'has_attention_qkvo_bias', fallback=False)
        args.has_mlp_bias = config.getboolean(component,
                                              'has_mlp_bias',
                                              fallback=False)
        args.has_model_final_layernorm = config.getboolean(
            component, 'has_model_final_layernorm', fallback=False)
        args.layernorm_eps = config.getfloat(component,
                                             'layernorm_eps',
                                             fallback=1e-5)
        args.layernorm_position = layernorm_position_map[config.get(
            component, 'layernorm_position')]
        args.layernorm_type = layernorm_type_map[config.get(
            component, 'layernorm_type')]
        args.hidden_act = config.get(component, 'hidden_act')
        args.has_lm_head_bias = config.getboolean(component,
                                                  'has_lm_head_bias',
                                                  fallback=False)
        args.relative_attention = config.getboolean(component,
                                                    'relative_attention',
                                                    fallback=False)
        args.num_buckets = config.getint(component, 'num_buckets')
        args.max_distance = config.getint(component, 'max_distance')
        args.logits_dtype = config.get(component,
                                       'logits_dtype',
                                       fallback='float32')
        args.encoder_hidden_size = config.getint('encoder', 'hidden_size')
        args.encoder_num_heads = config.getint('encoder', 'n_head')

    else:
        assert False, 'Unsupported component!'

    return args


def fuse_qkv(q, k, v):
    qkv_weight = np.concatenate((q, k, v))
    return qkv_weight


def load_t5_from_pytorch(tllm_model,
                         pytorch_ckpt_path,
                         component,
                         dtype="float32"):
    torch_dtype = str_dtype_to_torch(dtype)

    pytorch_ckpt = torch.load(pytorch_ckpt_path + '/t5_small.ckpt')
    pytorch_model = {
        key: torch_to_numpy(value.to(torch_dtype))
        for key, value in pytorch_ckpt.items()
    }

    if component == "encoder":
        tllm_model.embedding.vocab_embedding.weight.value = np.ascontiguousarray(
            pytorch_model['shared.weight'])
        if tllm_model.embedding.position_embedding:
            tllm_model.embedding.position_embedding.weight.value = pytorch_model[
                'encoder.embed_positions.weight']
        if tllm_model.embedding.token_type_embedding:
            tllm_model.embedding.token_type_embedding.weight.value = pytorch_model[
                'encoder.embed_token_type.weight']

        # all layers use 1st layer's attn table
        # transpose from [num_buckets, num_heads] -> [num_heads, num_buckets]
        # ascontiguousarray is very important! otherwise TRT always receives the original layout
        relative_attention_table = np.ascontiguousarray(pytorch_model[
            f'encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight']
                                                        .T)

        for i in range(tllm_model.num_layers):
            layer = tllm_model.encoder_layers[i]
            layer_prefix = f'encoder.block.{i}.'

            # attention table for all layers
            layer.attention.rel_attn_table.value = relative_attention_table

            layer.attention.qkv.weight.value = fuse_qkv(
                pytorch_model[f'{layer_prefix}layer.0.SelfAttention.q.weight'],
                pytorch_model[f'{layer_prefix}layer.0.SelfAttention.k.weight'],
                pytorch_model[f'{layer_prefix}layer.0.SelfAttention.v.weight'])
            layer.attention.dense.weight.value = pytorch_model[
                f'{layer_prefix}layer.0.SelfAttention.o.weight']

            if tllm_model.has_attention_qkvo_bias:
                layer.attention.qkv.bias.value = fuse_qkv(
                    pytorch_model[
                        f'{layer_prefix}layer.0.SelfAttention.q.bias'],
                    pytorch_model[
                        f'{layer_prefix}layer.0.SelfAttention.k.bias'],
                    pytorch_model[f'{layer_prefix}layer.0.SelfAttention.v.bias']
                )
                layer.attention.dense.bias.value = pytorch_model[
                    f'{layer_prefix}layer.0.SelfAttention.o.bias']

            layer.attention_layernorm.weight.value = pytorch_model[
                f'{layer_prefix}layer.0.layer_norm.weight']
            if tllm_model.layernorm_type != LayerNormType.RmsNorm:
                layer.attention_layernorm.bias.value = pytorch_model[
                    f'{layer_prefix}layer.0.layer_norm.bias']

            layer.mlp.fc.weight.value = pytorch_model[
                f'{layer_prefix}layer.1.DenseReluDense.wi.weight']
            layer.mlp.proj.weight.value = pytorch_model[
                f'{layer_prefix}layer.1.DenseReluDense.wo.weight']

            if tllm_model.has_mlp_bias:
                layer.mlp.fc.bias.value = pytorch_model[
                    f'{layer_prefix}layer.1.DenseReluDense.wi.bias']
                layer.mlp.proj.bias.value = pytorch_model[
                    f'{layer_prefix}layer.1.DenseReluDense.wo.bias']

            layer.mlp_layernorm.weight.value = pytorch_model[
                f'{layer_prefix}layer.1.layer_norm.weight']
            if tllm_model.layernorm_type != LayerNormType.RmsNorm:
                layer.mlp_layernorm.bias.value = pytorch_model[
                    f'{layer_prefix}layer.1.layer_norm.bias']

        if tllm_model.final_layernorm:
            tllm_model.final_layernorm.weight.value = pytorch_model[
                'encoder.final_layer_norm.weight']
            if tllm_model.layernorm_type != LayerNormType.RmsNorm:
                tllm_model.final_layernorm.bias.value = pytorch_model[
                    'encoder.final_layer_norm.bias']

    if component == "decoder":
        tllm_model.embedding.vocab_embedding.weight.value = pytorch_model[
            'shared.weight']
        if tllm_model.embedding.position_embedding:
            tllm_model.embedding.position_embedding.weight.value = pytorch_model[
                'decoder.embed_positions.weight']
        if tllm_model.embedding.token_type_embedding:
            tllm_model.embedding.token_type_embedding.weight.value = pytorch_model[
                'decoder.embed_token_type.weight']

        # all layers use 1st layer's attn table
        # transpose from [num_buckets, num_heads] --> [num_heads, num_buckets]
        # ascontiguousarray is very important! otherwise TRT always receives the original layout
        relative_attention_table = np.ascontiguousarray(pytorch_model[
            f'decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight']
                                                        .T)

        for i in range(tllm_model.num_layers):
            layer = tllm_model.decoder_layers[i]
            layer_prefix = f'decoder.block.{i}.'

            # attention table for all layers
            layer.self_attention.rel_attn_table.value = relative_attention_table

            # self attn
            layer.self_attention.qkv.weight.value = fuse_qkv(
                pytorch_model[f'{layer_prefix}layer.0.SelfAttention.q.weight'],
                pytorch_model[f'{layer_prefix}layer.0.SelfAttention.k.weight'],
                pytorch_model[f'{layer_prefix}layer.0.SelfAttention.v.weight'])
            layer.self_attention.dense.weight.value = pytorch_model[
                f'{layer_prefix}layer.0.SelfAttention.o.weight']

            if tllm_model.has_attention_qkvo_bias:
                layer.self_attention.qkv.bias.value = fuse_qkv(
                    pytorch_model[
                        f'{layer_prefix}layer.0.SelfAttention.q.bias'],
                    pytorch_model[
                        f'{layer_prefix}layer.0.SelfAttention.k.bias'],
                    pytorch_model[f'{layer_prefix}layer.0.SelfAttention.v.bias']
                )
                layer.self_attention.dense.bias.value = pytorch_model[
                    f'{layer_prefix}layer.0.SelfAttention.o.bias']

            layer.self_attention_layernorm.weight.value = pytorch_model[
                f'{layer_prefix}layer.0.layer_norm.weight']
            if tllm_model.layernorm_type != LayerNormType.RmsNorm:
                layer.self_attention_layernorm.bias.value = pytorch_model[
                    f'{layer_prefix}layer.0.layer_norm.bias']

            # cross attn
            layer.cross_attention.qkv.weight.value = fuse_qkv(
                pytorch_model[
                    f'{layer_prefix}layer.1.EncDecAttention.q.weight'],
                pytorch_model[
                    f'{layer_prefix}layer.1.EncDecAttention.k.weight'],
                pytorch_model[f'{layer_prefix}layer.1.EncDecAttention.v.weight']
            )
            layer.cross_attention.dense.weight.value = pytorch_model[
                f'{layer_prefix}layer.1.EncDecAttention.o.weight']

            if tllm_model.has_attention_qkvo_bias:
                layer.cross_attention.qkv.bias.value = fuse_qkv(
                    pytorch_model[
                        f'{layer_prefix}layer.1.EncDecAttention.q.bias'],
                    pytorch_model[
                        f'{layer_prefix}layer.1.EncDecAttention.k.bias'],
                    pytorch_model[
                        f'{layer_prefix}layer.1.EncDecAttention.v.bias'])
                layer.cross_attention.dense.bias.value = pytorch_model[
                    f'{layer_prefix}layer.1.EncDecAttention.o.bias']

            layer.cross_attention_layernorm.weight.value = pytorch_model[
                f'{layer_prefix}layer.1.layer_norm.weight']
            if tllm_model.layernorm_type != LayerNormType.RmsNorm:
                layer.cross_attention_layernorm.bias.value = pytorch_model[
                    f'{layer_prefix}layer.1.layer_norm.bias']

            layer.mlp.fc.weight.value = pytorch_model[
                f'{layer_prefix}layer.2.DenseReluDense.wi.weight']
            layer.mlp.proj.weight.value = pytorch_model[
                f'{layer_prefix}layer.2.DenseReluDense.wo.weight']

            if tllm_model.has_mlp_bias:
                layer.mlp.fc.bias.value = pytorch_model[
                    f'{layer_prefix}layer.2.DenseReluDense.wi.bias']
                layer.mlp.proj.bias.value = pytorch_model[
                    f'{layer_prefix}layer.2.DenseReluDense.wo.bias']

            layer.mlp_layernorm.weight.value = pytorch_model[
                f'{layer_prefix}layer.2.layer_norm.weight']
            if tllm_model.layernorm_type != LayerNormType.RmsNorm:
                layer.mlp_layernorm.bias.value = pytorch_model[
                    f'{layer_prefix}layer.2.layer_norm.bias']

        if tllm_model.final_layernorm:
            tllm_model.final_layernorm.weight.value = pytorch_model[
                'decoder.final_layer_norm.weight']
            if tllm_model.layernorm_type != LayerNormType.RmsNorm:
                tllm_model.final_layernorm.bias.value = pytorch_model[
                    'decoder.final_layer_norm.bias']

        tllm_model.lm_head.weight.value = pytorch_model['lm_head.weight']

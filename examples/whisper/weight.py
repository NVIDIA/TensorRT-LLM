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
    print(config)
    if component == 'encoder':
        args.n_layer = config.getint(component, 'encoder_layers')
        args.n_head = config.getint(component, 'encoder_attention_heads')
        args.hidden_size = config.getint(component, 'd_model')
        args.n_ctx = config.getint(component, 'max_source_positions')
        args.ffn_hidden_size = config.getint(component, 'encoder_ffn_dim')
        args.vocab_size = config.getint(component, 'vocab_size')
        args.n_positions = config.getint(component, 'max_source_positions')
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
        args.has_attention_qkvo_bias = True
        args.has_mlp_bias = True
        args.has_model_final_layernorm = True
        args.layernorm_eps = config.getfloat(component,
                                             'layernorm_eps',
                                             fallback=1e-5)
        args.layernorm_position = layernorm_position_map['pre_layernorm']
        args.layernorm_type = layernorm_type_map["LayerNorm"]
        args.hidden_act = config.get(component, 'activation_function')
        args.relative_attention = False
        args.head_size = args.hidden_size // config.getint(component, 'encoder_attention_heads')
        args.ckpt_weight_dtype = config.get(component, 'weight_data_type')
        
    elif component == 'decoder':
        args.n_layer = config.getint(component, 'decoder_layers')
        args.n_head = config.getint(component, 'decoder_attention_heads')
        args.hidden_size = config.getint(component, 'd_model')
        args.head_size = args.hidden_size // config.getint(component, 'decoder_attention_heads')
        
        args.ffn_hidden_size = config.getint(component, 'decoder_ffn_dim')
        args.vocab_size = config.getint(component, 'vocab_size')
        args.n_positions = config.getint(component, 'max_target_positions')
        args.has_position_embedding = True
        args.has_token_type_embedding = config.getboolean(
            component, 'has_token_type_embedding', fallback=False)
        args.has_embedding_layernorm = config.getboolean(
            component, 'has_embedding_layernorm', fallback=False)
        args.has_embedding_scale = config.getboolean(component,
                                                     'has_embedding_scale',
                                                     fallback=False)
        args.q_scaling = config.getfloat(component, 'q_scaling', fallback=1.0)
        args.has_attention_qkvo_bias = True
        args.has_mlp_bias = True
        args.has_model_final_layernorm = True
        args.layernorm_eps = config.getfloat(component,
                                             'layernorm_eps',
                                             fallback=1e-5)
        args.layernorm_position = layernorm_position_map['pre_layernorm']
        args.layernorm_type = layernorm_type_map["LayerNorm"]
        args.hidden_act = config.get(component, 'activation_function')
        args.gated_act = False
        args.mlp_type = 'MLP'
        args.has_lm_head_bias = False
        args.relative_attention = False
        args.logits_dtype = config.get(component,
                                       'logits_dtype',
                                       fallback='float32')
        args.encoder_hidden_size = config.getint('encoder', 'd_model')
        args.encoder_num_heads = config.getint('encoder', 'encoder_attention_heads')
        args.encoder_head_size = config.getint('encoder', 'd_model') // config.getint(component, 'encoder_attention_heads')
        args.ckpt_weight_dtype = config.get(component, 'weight_data_type')

    else:
        assert False, 'Unsupported component!'
    return args

def fuse_qkv(q, k, v):
    qkv_weight = np.concatenate((q, k, v))
    return qkv_weight

def load_whisper_from_pytorch(tllm_model,
                         pytorch_ckpt_path,
                         component,
                         model_size="tiny",
                         multilingual=False,
                         dtype="float32"):
    torch_dtype = str_dtype_to_torch(dtype)
    model_name = "whisper-" + f"{model_size}"
    if not multilingual:
        model_name = model_name + ".en"

    pytorch_ckpt = torch.load(pytorch_ckpt_path + f"{model_name}.ckpt")
    pytorch_model = {
        key: torch_to_numpy(value.to(torch_dtype))
        for key, value in pytorch_ckpt.items()
    }

    if component == "encoder":
        # set conv1d
        tllm_model.conv1.weight.value = np.expand_dims(pytorch_model['model.encoder.conv1.weight'], axis=3)
        tllm_model.conv1.bias.value = pytorch_model['model.encoder.conv1.bias']
        tllm_model.conv2.weight.value = np.expand_dims(pytorch_model['model.encoder.conv2.weight'], axis=3)
        tllm_model.conv2.bias.value = pytorch_model['model.encoder.conv2.bias']
       
        for i in range(tllm_model.num_layers):
            layer = tllm_model.blocks[i]
            layer_prefix = f'model.encoder.layers.{i}.'

            # attention table for all layers
            # layer.attention.rel_attn_table.value = relative_attention_table

            layer.attention.qkv.weight.value = fuse_qkv(
                pytorch_model[f'{layer_prefix}self_attn.q_proj.weight'],
                pytorch_model[f'{layer_prefix}self_attn.k_proj.weight'],
                pytorch_model[f'{layer_prefix}self_attn.v_proj.weight'])
            layer.attention.dense.weight.value = pytorch_model[
                f'{layer_prefix}self_attn.out_proj.weight']

            # if tllm_model.has_attention_qkvo_bias:
            layer.attention.qkv.bias.value = fuse_qkv(
                pytorch_model[f'{layer_prefix}self_attn.q_proj.bias'],
                torch.zeros(pytorch_model[f'{layer_prefix}self_attn.q_proj.bias'].shape), # no bias for k
                pytorch_model[f'{layer_prefix}self_attn.v_proj.bias']
            )
            layer.attention.dense.bias.value = pytorch_model[
                f'{layer_prefix}self_attn.out_proj.bias']

            layer.attention_layernorm.weight.value = pytorch_model[
                f'{layer_prefix}self_attn_layer_norm.weight']
            # if tllm_model.layernorm_type != LayerNormType.RmsNorm:
            layer.attention_layernorm.bias.value = pytorch_model[
                f'{layer_prefix}self_attn_layer_norm.bias']

            layer.mlp.fc.weight.value = pytorch_model[
                f'{layer_prefix}fc1.weight']
            layer.mlp.proj.weight.value = pytorch_model[
                f'{layer_prefix}fc2.weight']

            # if tllm_model.has_mlp_bias:
            layer.mlp.fc.bias.value = pytorch_model[
                f'{layer_prefix}fc1.bias']
            layer.mlp.proj.bias.value = pytorch_model[
                f'{layer_prefix}fc2.bias']

            layer.mlp_layernorm.weight.value = pytorch_model[
                f'{layer_prefix}final_layer_norm.weight']
            # if tllm_model.layernorm_type != LayerNormType.RmsNorm:
            layer.mlp_layernorm.bias.value = pytorch_model[
                f'{layer_prefix}final_layer_norm.bias']

        # if tllm_model.final_layernorm:
        tllm_model.ln_post.weight.value = pytorch_model[
            'model.encoder.layer_norm.weight']
        # if tllm_model.layernorm_type != LayerNormType.RmsNorm:
        tllm_model.ln_post.bias.value = pytorch_model[
            'model.encoder.layer_norm.bias']

    if component == "decoder":
        tllm_model.embedding.vocab_embedding.weight.value = pytorch_model[
            'model.decoder.embed_tokens.weight']
        if tllm_model.embedding.position_embedding:
            tllm_model.embedding.position_embedding.weight.value = pytorch_model[
                'model.decoder.embed_positions.weight']
        if tllm_model.embedding.token_type_embedding:
            tllm_model.embedding.token_type_embedding.weight.value = pytorch_model[
                'decoder.embed_token_type.weight']

        for i in range(tllm_model.num_layers):
            layer = tllm_model.decoder_layers[i]
            layer_prefix = f'model.decoder.layers.{i}.'

            # self attn
            layer.self_attention.qkv.weight.value = fuse_qkv(
                pytorch_model[f'{layer_prefix}self_attn.q_proj.weight'],
                pytorch_model[f'{layer_prefix}self_attn.k_proj.weight'],
                pytorch_model[f'{layer_prefix}self_attn.v_proj.weight'])
            layer.self_attention.dense.weight.value = pytorch_model[
                f'{layer_prefix}self_attn.out_proj.weight']

            if tllm_model.has_attention_qkvo_bias:
                layer.self_attention.qkv.bias.value = fuse_qkv(
                    pytorch_model[
                        f'{layer_prefix}self_attn.q_proj.bias'],
                    torch.zeros(
                        pytorch_model[f'{layer_prefix}self_attn.q_proj.bias'].shape),
                    pytorch_model[
                        f'{layer_prefix}self_attn.v_proj.bias']
                )
                layer.self_attention.dense.bias.value = pytorch_model[
                    f'{layer_prefix}self_attn.out_proj.bias']

            layer.self_attention_layernorm.weight.value = pytorch_model[
                f'{layer_prefix}self_attn_layer_norm.weight']
            if tllm_model.layernorm_type != LayerNormType.RmsNorm:
                layer.self_attention_layernorm.bias.value = pytorch_model[
                    f'{layer_prefix}self_attn_layer_norm.bias']

            # cross attn
            layer.cross_attention.qkv.weight.value = fuse_qkv(
                pytorch_model[
                    f'{layer_prefix}encoder_attn.q_proj.weight'],
                pytorch_model[
                    f'{layer_prefix}encoder_attn.k_proj.weight'],
                pytorch_model[
                    f'{layer_prefix}encoder_attn.v_proj.weight']
            )
            layer.cross_attention.dense.weight.value = pytorch_model[
                f'{layer_prefix}encoder_attn.out_proj.weight']

            if tllm_model.has_attention_qkvo_bias:
                layer.cross_attention.qkv.bias.value = fuse_qkv(
                    pytorch_model[
                        f'{layer_prefix}encoder_attn.q_proj.bias'],
                    torch.zeros(pytorch_model[
                        f'{layer_prefix}encoder_attn.q_proj.bias'].shape),
                    pytorch_model[
                        f'{layer_prefix}encoder_attn.v_proj.bias'])
                layer.cross_attention.dense.bias.value = pytorch_model[
                    f'{layer_prefix}encoder_attn.out_proj.bias']

            layer.cross_attention_layernorm.weight.value = pytorch_model[
                f'{layer_prefix}encoder_attn_layer_norm.weight']
            if tllm_model.layernorm_type != LayerNormType.RmsNorm:
                layer.cross_attention_layernorm.bias.value = pytorch_model[
                    f'{layer_prefix}encoder_attn_layer_norm.bias']

            layer.mlp.fc.weight.value = pytorch_model[
                f'{layer_prefix}fc1.weight']
            layer.mlp.proj.weight.value = pytorch_model[
                f'{layer_prefix}fc2.weight']

            if tllm_model.has_mlp_bias:
                layer.mlp.fc.bias.value = pytorch_model[
                    f'{layer_prefix}fc1.bias']
                layer.mlp.proj.bias.value = pytorch_model[
                    f'{layer_prefix}fc2.bias']

            layer.mlp_layernorm.weight.value = pytorch_model[
                f'{layer_prefix}final_layer_norm.weight']
            if tllm_model.layernorm_type != LayerNormType.RmsNorm:
                layer.mlp_layernorm.bias.value = pytorch_model[
                    f'{layer_prefix}final_layer_norm.bias']

        if tllm_model.final_layernorm:
            tllm_model.final_layernorm.weight.value = pytorch_model[
                'model.decoder.layer_norm.weight']
            if tllm_model.layernorm_type != LayerNormType.RmsNorm:
                tllm_model.final_layernorm.bias.value = pytorch_model[
                    'model.decoder.layer_norm.bias']

        tllm_model.lm_head.weight.value = pytorch_model['proj_out.weight']
import time
from os import path
from pathlib import Path
from typing import Optional, Union

import numpy as np

from tensorrt_llm import logger
from tensorrt_llm._utils import numpy_to_dtype, str_dtype_to_np
from tensorrt_llm.functional import (LayerNormPositionType, LayerNormType,
                                     MLPType)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import (  # TODO: probably need to change model name to distinguish from other models
    DecoderModel, EncoderModel)

layernorm_type_map = {i.name: i.value for i in LayerNormType}
layernorm_position_map = {i.name: i.value for i in LayerNormPositionType}
mlp_type_map = {i.name: i.value for i in MLPType}


def parse_bart_config(config, component, args):
    assert component in ('encoder', 'decoder'), 'Unsupported component!'
    args.n_layer = config.getint(component, f'{component}_layers')
    args.n_head = config.getint(component, f'{component}_attention_heads')
    args.hidden_size = config.getint(component, 'd_model')
    args.head_size = config.getint(component,
                                   'd_kv',
                                   fallback=args.hidden_size // args.n_head)
    args.ffn_hidden_size = config.getint(component, f'{component}_ffn_dim')
    args.vocab_size = config.getint(component, 'vocab_size')
    args.n_positions = config.getint(component, 'max_position_embeddings')
    args.has_position_embedding = config.getboolean(
        component, 'has_position_embedding',
        fallback=True)  # TODO: hardcoded here
    args.has_token_type_embedding = config.getboolean(
        component, 'has_token_type_embedding', fallback=False)
    args.has_embedding_layernorm = config.getboolean(component,
                                                     'has_embedding_layernorm',
                                                     fallback=True)
    args.has_embedding_scale = config.getboolean(component, 'scale_embedding')
    args.q_scaling = config.getfloat(component, 'q_scaling', fallback=1.0)
    args.has_attention_qkvo_bias = config.getboolean('structure',
                                                     't5_with_bias',
                                                     fallback=True)
    args.has_mlp_bias = config.getboolean('structure',
                                          't5_with_bias',
                                          fallback=True)
    args.has_model_final_layernorm = config.getboolean(
        component, 'has_model_final_layernorm')
    args.layernorm_eps = config.getfloat(component,
                                         'layer_norm_epsilon',
                                         fallback=False)

    normalize_before = config.getboolean(component, 'normalize_before')
    args.layernorm_position = layernorm_position_map[
        'pre_layernorm' if normalize_before else 'post_layernorm']

    args.layernorm_type = layernorm_type_map[config.get(component,
                                                        'layernorm_type',
                                                        fallback='LayerNorm')]
    args.hidden_act = config.get(component, 'activation_function')
    args.gated_act = config.getboolean(component,
                                       'is_gated_act',
                                       fallback=False)
    args.mlp_type = mlp_type_map['GatedMLP' if args.gated_act else 'MLP']
    args.relative_attention = config.get(
        'structure', 'position_embedding_type') == 'relative'

    args.num_buckets = config.getint(component,
                                     'relative_attention_num_buckets',
                                     fallback=0)
    args.max_distance = config.getint(component,
                                      'relative_attention_max_distance',
                                      fallback=0)
    args.ckpt_weight_dtype = config.get(component, 'weight_data_type')

    if component == 'decoder':
        args.rescale_before_lm_head = config.getboolean(
            component, 'rescale_before_lm_head')
        args.logits_dtype = config.get(component,
                                       'logits_dtype',
                                       fallback='float32')
        args.encoder_hidden_size = config.getint('encoder', 'd_model')
        args.encoder_num_heads = config.getint('encoder',
                                               'encoder_attention_heads')
        args.encoder_head_size = config.getint(
            'encoder',
            'd_kv',
            fallback=args.encoder_hidden_size // args.encoder_num_heads)

    return args


def load_from_binary_bart(tllm_model: Union[DecoderModel, EncoderModel],
                          dir_path,
                          args,
                          mapping: Optional[Mapping] = None,
                          dtype='float32',
                          use_parallel_embedding=False,
                          sharding_dim=0,
                          share_embedding_table=False,
                          scaling_factors=None):

    logger.info('Loading weights from binary...')
    tik = time.time()

    if mapping is None:
        mapping = Mapping()

    ckpt_np_dtype = str_dtype_to_np(args.ckpt_weight_dtype)

    def fromfile(name, split=True, shape=None) -> Optional[np.ndarray]:
        p = path.join(
            dir_path,
            f'{name}.{mapping.tp_rank}.bin' if split else f'{name}.bin')
        if Path(p).exists():
            t = np.fromfile(p, dtype=ckpt_np_dtype)
            t = numpy_to_dtype(t, dtype)
            if shape is not None:
                t = t.reshape(shape)
            t = np.ascontiguousarray(t)
            return t
        return None

    component = 'encoder' if isinstance(tllm_model, EncoderModel) else 'decoder'

    # only load word / pos emb and emb layernorm to first PP rank
    if mapping.is_first_pp_rank():
        wte = fromfile(f'model.{component}.embed_tokens.weight',
                       shape=[args.vocab_size, -1],
                       split=False)

        # word embedding
        tllm_model.embedding.vocab_embedding.weight.value = wte

        # positional embedding
        wpe = fromfile(f'model.{component}.embed_positions.weight',
                       shape=[args.n_positions, args.hidden_size],
                       split=False)
        tllm_model.embedding.position_embedding.weight.value = wpe

        # Embedding layer norm
        tllm_model.embedding.embedding_layernorm.weight.value = fromfile(
            f'model.{component}.layernorm_embedding.weight', split=False)
        tllm_model.embedding.embedding_layernorm.bias.value = fromfile(
            f'model.{component}.layernorm_embedding.bias', split=False)

    local_num_layers = tllm_model.num_layers

    for local_idx, global_idx in enumerate(
            range(mapping.pp_rank * local_num_layers,
                  (mapping.pp_rank + 1) * local_num_layers)
    ):  # TODO: does this load the correct layers for PP?
        layer = getattr(tllm_model, f'{component}_layers')[local_idx]
        layer_prefix = f'model.{component}.layers.{global_idx}'

        self_attention_layer = getattr(
            layer, 'attention' if component == 'encoder' else 'self_attention')

        # self attention
        self_attention_layer.qkv.weight.value = fromfile(
            f'{layer_prefix}.self_attn.qkv_proj.weight',
            shape=[args.hidden_size * 3 // mapping.tp_size, args.hidden_size])
        self_attention_layer.qkv.bias.value = fromfile(
            f'{layer_prefix}.self_attn.qkv_proj.bias',
            shape=[args.hidden_size * 3 // mapping.tp_size])

        self_attention_layer.dense.weight.value = fromfile(
            f'{layer_prefix}.self_attn.out_proj.weight',
            shape=[args.hidden_size, args.hidden_size // mapping.tp_size])
        self_attention_layer.dense.bias.value = fromfile(
            f'{layer_prefix}.self_attn.out_proj.bias',
            shape=[args.hidden_size],
            split=False)

        self_attention_layernorm = getattr(
            layer, 'self_attention_layernorm'
            if component == 'decoder' else 'attention_layernorm')
        self_attention_layernorm.weight.value = fromfile(
            f'{layer_prefix}.self_attn_layer_norm.weight', split=False)
        self_attention_layernorm.bias.value = fromfile(
            f'{layer_prefix}.self_attn_layer_norm.bias', split=False)

        # cross attention
        if component == 'decoder':
            layer.cross_attention.qkv.weight.value = fromfile(
                f'{layer_prefix}.encoder_attn.qkv_proj.weight',
                shape=[
                    args.hidden_size * 3 // mapping.tp_size, args.hidden_size
                ])
            layer.cross_attention.qkv.bias.value = fromfile(
                f'{layer_prefix}.encoder_attn.qkv_proj.bias',
                shape=[args.hidden_size * 3 // mapping.tp_size])

            layer.cross_attention.dense.weight.value = fromfile(
                f'{layer_prefix}.encoder_attn.out_proj.weight',
                shape=[args.hidden_size, args.hidden_size // mapping.tp_size])
            layer.cross_attention.dense.bias.value = fromfile(
                f'{layer_prefix}.encoder_attn.out_proj.bias',
                shape=[args.hidden_size],
                split=False)

            layer.cross_attention_layernorm.weight.value = fromfile(
                f'{layer_prefix}.encoder_attn_layer_norm.weight', split=False)
            layer.cross_attention_layernorm.bias.value = fromfile(
                f'{layer_prefix}.encoder_attn_layer_norm.bias', split=False)

        layer.mlp.fc.weight.value = fromfile(
            f'{layer_prefix}.fc1.weight',
            shape=[args.ffn_hidden_size // mapping.tp_size, args.hidden_size])
        layer.mlp.fc.bias.value = fromfile(
            f'{layer_prefix}.fc1.bias',
            shape=[args.ffn_hidden_size // mapping.tp_size])
        layer.mlp.proj.weight.value = fromfile(
            f'{layer_prefix}.fc2.weight',
            shape=[args.hidden_size, args.ffn_hidden_size // mapping.tp_size])
        layer.mlp.proj.bias.value = fromfile(f'{layer_prefix}.fc2.bias',
                                             shape=[args.hidden_size],
                                             split=False)

        layer.mlp_layernorm.weight.value = fromfile(
            f'{layer_prefix}.final_layer_norm.weight', split=False)
        layer.mlp_layernorm.bias.value = fromfile(
            f'{layer_prefix}.final_layer_norm.bias', split=False)

    if mapping.is_last_pp_rank():
        if tllm_model.has_model_final_layernorm:  # mBART true BART false
            tllm_model.final_layernorm.weight.value = fromfile(
                f'model.{component}.layer_norm.weight', split=False)
            tllm_model.final_layernorm.bias.value = fromfile(
                f'model.{component}.layer_norm.bias', split=False)
        if component == 'decoder':
            tllm_model.lm_head.weight.value = fromfile(
                'lm_head.weight',
                shape=[args.vocab_size // mapping.tp_size, args.hidden_size])

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Weights loaded. Total time: {t}')

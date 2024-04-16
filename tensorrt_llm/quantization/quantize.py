from ..layers import MLP, ColumnLinear, GatedMLP, LayerNorm, RmsNorm, RowLinear
from ..models.modeling_utils import QuantConfig
from ..parameter import Parameter
from .layers import (FP8Linear, FP8RowLinear, Int8SmoothQuantLinear,
                     Int8SmoothQuantRowLinear, SmoothQuantAttention,
                     SmoothQuantGatedMLP, SmoothQuantLayerNorm, SmoothQuantMLP,
                     SmoothQuantRmsNorm, WeightOnlyGroupwiseQuantColumnLinear,
                     WeightOnlyGroupwiseQuantRowLinear,
                     WeightOnlyQuantColumnLinear, WeightOnlyQuantRowLinear)
from .mode import W8A8_SQ_PLUGIN_LIST, QuantAlgo


def weight_only_quantize(model,
                         quant_config: QuantConfig,
                         current_key_name=None):
    assert quant_config.quant_mode.is_weight_only()

    exclude_modules = quant_config.exclude_modules or ['lm_head']

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if len(list(module.children())) > 0:
            weight_only_quantize(module, quant_config, current_key_name)

        if isinstance(module, ColumnLinear) and name not in exclude_modules:
            if not any(key in '.'.join(current_key_name)
                       for key in exclude_modules):
                model._modules[name] = WeightOnlyQuantColumnLinear(
                    in_features=module.in_features,
                    out_features=module.out_features * module.tp_size,
                    bias=module.bias is not None,
                    dtype=module.dtype,
                    tp_group=module.tp_group,
                    tp_size=module.tp_size,
                    gather_output=module.gather_output,
                    quant_mode=quant_config.quant_mode)
        elif isinstance(module, RowLinear) and name not in exclude_modules:
            if not any(key in '.'.join(current_key_name)
                       for key in exclude_modules):
                model._modules[name] = WeightOnlyQuantRowLinear(
                    in_features=module.in_features * module.tp_size,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    dtype=module.dtype,
                    tp_group=module.tp_group,
                    tp_size=module.tp_size,
                    quant_mode=quant_config.quant_mode)

        current_key_name.pop(-1)

    setattr(model, 'quant_mode', quant_config.quant_mode)
    return model


def weight_only_groupwise_quantize(model,
                                   quant_config: QuantConfig,
                                   current_key_name=None):
    assert quant_config.quant_mode.is_weight_only()

    exclude_modules = quant_config.exclude_modules or ['lm_head']

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if len(list(module.children())) > 0:
            weight_only_groupwise_quantize(module, quant_config,
                                           current_key_name)

        if isinstance(module, ColumnLinear) and name not in exclude_modules:
            if not any(key in '.'.join(current_key_name)
                       for key in exclude_modules):
                model._modules[name] = WeightOnlyGroupwiseQuantColumnLinear(
                    in_features=module.in_features,
                    out_features=module.out_features * module.tp_size,
                    group_size=quant_config.group_size,
                    pre_quant_scale=quant_config.pre_quant_scale,
                    zero=quant_config.has_zero_point,
                    bias=module.bias is not None,
                    use_w4a8_awq=quant_config.quant_algo == QuantAlgo.W4A8_AWQ,
                    dtype=module.dtype,
                    tp_group=module.tp_group,
                    tp_size=module.tp_size,
                    gather_output=module.gather_output)
        elif isinstance(module, RowLinear) and name not in exclude_modules:
            if not any(key in '.'.join(current_key_name)
                       for key in exclude_modules):
                model._modules[name] = WeightOnlyGroupwiseQuantRowLinear(
                    in_features=module.in_features * module.tp_size,
                    out_features=module.out_features,
                    group_size=quant_config.group_size,
                    pre_quant_scale=quant_config.pre_quant_scale,
                    zero=quant_config.has_zero_point,
                    bias=module.bias is not None,
                    use_w4a8_awq=quant_config.quant_algo == QuantAlgo.W4A8_AWQ,
                    dtype=module.dtype,
                    tp_group=module.tp_group,
                    tp_size=module.tp_size)

        current_key_name.pop(-1)

    setattr(model, 'quant_mode', quant_config.quant_mode)
    return model


def smooth_quantize_ootb(
    model,
    quant_config: QuantConfig,
    current_key_name=None,
):
    exclude_modules = quant_config.exclude_modules or ['lm_head']

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if len(list(module.children())) > 0:
            smooth_quantize_ootb(module, quant_config, current_key_name)

        if isinstance(module, ColumnLinear) and name not in exclude_modules:
            if not any(key in '.'.join(current_key_name)
                       for key in exclude_modules):
                model._modules[name] = Int8SmoothQuantLinear(
                    module.in_features, module.out_features * module.tp_size,
                    module.bias, module.dtype, module.tp_group, module.tp_size,
                    module.gather_output)
        elif isinstance(module, RowLinear) and name not in exclude_modules:
            if not any(key in '.'.join(current_key_name)
                       for key in exclude_modules):
                model._modules[name] = Int8SmoothQuantRowLinear(
                    module.in_features * module.tp_size, module.out_features,
                    module.bias, module.dtype, module.tp_group, module.tp_size)

        current_key_name.pop(-1)

    setattr(model, 'quant_mode', quant_config.quant_mode)
    return model


def smooth_quantize_plugin(model, quant_mode):
    for layer_idx, layer in enumerate(model.transformer.layers):
        config = layer.config

        assert hasattr(layer,
                       "input_layernorm"), "The layer has no input_layernorm"
        quant_norm_cls = None
        if isinstance(layer.input_layernorm, RmsNorm):
            quant_norm_cls = SmoothQuantRmsNorm
        elif isinstance(layer.input_layernorm, LayerNorm):
            quant_norm_cls = SmoothQuantLayerNorm
        assert quant_norm_cls is not None
        layer.input_layernorm = quant_norm_cls(
            normalized_shape=config.hidden_size,
            eps=config.norm_epsilon,
            dtype=config.dtype,
            quant_mode=quant_mode)

        assert hasattr(layer, "attention"), "The layer has no attention"
        qkv_bias = layer.attention.qkv.bias is not None
        dense_bias = layer.attention.dense.bias is not None
        head_size = config.head_size if hasattr(config, 'head_size') else None
        layer.attention = SmoothQuantAttention(
            layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            attention_head_size=head_size,
            max_position_embeddings=config.max_position_embeddings,
            num_layers=config.num_hidden_layers,
            dtype=config.dtype,
            attention_mask_type=layer.attention.attention_mask_type,
            position_embedding_type=layer.attention.position_embedding_type,
            rotary_embedding_base=layer.attention.rotary_embedding_base,
            rotary_embedding_scaling=layer.attention.rotary_embedding_scaling,
            rotary_embedding_percentage=layer.attention.
            rotary_embedding_percentage,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            tp_rank=config.mapping.tp_rank,
            quant_mode=quant_mode,
            bias=(qkv_bias and dense_bias),
            qkv_bias_only=(qkv_bias and not dense_bias))

        assert hasattr(layer, "mlp"), "The layer has no mlp"

        mlp_norm_cls = None
        if isinstance(layer.mlp, GatedMLP):
            mlp_norm_cls = SmoothQuantGatedMLP
        elif isinstance(layer.mlp, MLP):
            mlp_norm_cls = SmoothQuantMLP

        mlp_hidden_size = config.hidden_size * 4 if config.intermediate_size is None else config.intermediate_size
        layer.mlp = mlp_norm_cls(hidden_size=config.hidden_size,
                                 ffn_hidden_size=mlp_hidden_size,
                                 hidden_act=config.hidden_act,
                                 dtype=config.dtype,
                                 tp_group=config.mapping.tp_group,
                                 tp_size=config.mapping.tp_size,
                                 quant_mode=quant_mode,
                                 bias=layer.mlp.bias)
        assert hasattr(layer,
                       "post_layernorm"), "The layer has no post_layernorm"

        quant_norm_cls = None
        if isinstance(layer.post_layernorm, RmsNorm):
            quant_norm_cls = SmoothQuantRmsNorm
        elif isinstance(layer.post_layernorm, LayerNorm):
            quant_norm_cls = SmoothQuantLayerNorm
        assert quant_norm_cls is not None

        layer.post_layernorm = quant_norm_cls(
            normalized_shape=config.hidden_size,
            eps=config.norm_epsilon,
            dtype=config.dtype,
            quant_mode=quant_mode)

    setattr(model, 'quant_mode', quant_mode)
    return model


def smooth_quantize(model, quant_config: QuantConfig):
    assert quant_config.quant_mode.has_act_and_weight_quant()
    if quant_config.quant_algo in W8A8_SQ_PLUGIN_LIST:
        return smooth_quantize_plugin(model, quant_config.quant_mode)
    else:
        return smooth_quantize_ootb(model, quant_config)


def fp8_quantize(model, quant_config: QuantConfig, current_key_name=None):
    assert quant_config.quant_mode.has_fp8_qdq()

    exclude_modules = quant_config.exclude_modules or ['lm_head']
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if len(list(module.children())) > 0:
            fp8_quantize(module, quant_config, current_key_name)

        if isinstance(module, ColumnLinear) and name not in exclude_modules:
            if not any(key in '.'.join(current_key_name)
                       for key in exclude_modules):
                model._modules[name] = FP8Linear(
                    in_features=module.in_features,
                    out_features=module.out_features * module.tp_size,
                    bias=module.bias is not None,
                    dtype=module.dtype,
                    tp_group=module.tp_group,
                    tp_size=module.tp_size,
                    gather_output=module.gather_output)
        elif isinstance(module, RowLinear) and name not in exclude_modules:
            if not any(key in '.'.join(current_key_name)
                       for key in exclude_modules):
                model._modules[name] = FP8RowLinear(
                    in_features=module.in_features * module.tp_size,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    dtype=module.dtype,
                    tp_group=module.tp_group,
                    tp_size=module.tp_size)

        current_key_name.pop(-1)

    setattr(model, 'quant_mode', quant_config.quant_mode)
    return model


def kv_cache_quantize(model, quant_config: QuantConfig):
    assert quant_config.quant_mode.has_kv_cache_quant()

    for layer in model.transformer.layers:
        layer.attention.kv_cache_scaling_factor = Parameter(shape=(1, ),
                                                            dtype='float32')


def quantize(model, quant_config: QuantConfig):
    quant_mode = quant_config.quant_mode

    if quant_mode.has_fp8_qdq():
        model = fp8_quantize(model, quant_config)
    elif quant_mode.has_act_and_weight_quant():
        model = smooth_quantize(model, quant_config)
    elif quant_mode.is_weight_only():
        if quant_mode.has_per_group_scaling():
            model = weight_only_groupwise_quantize(model, quant_config)
        else:
            model = weight_only_quantize(model, quant_config)

    if quant_mode.has_kv_cache_quant():
        model = kv_cache_quantize(model, quant_config)

    return model

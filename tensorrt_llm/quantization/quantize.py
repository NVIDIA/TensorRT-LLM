import fnmatch
from typing import Union

import torch

from .._utils import get_init_params
from ..layers import (MLP, Attention, ColumnLinear, Embedding, GatedMLP,
                      LayerNorm, RmsNorm, RowLinear)
from ..layers.moe import MixtureOfExperts
from ..models.modeling_utils import LayerQuantConfig, QuantConfig
from ..parameter import Parameter

# isort: off
from .layers import (
    FP4Linear, FP4RowLinear, FP8Linear, FP8RowLinear, Fp8RowwiseAttention,
    Fp8RowwiseGatedMLP, Fp8RowwiseLayerNorm, Fp8RowwiseMLP, Fp8RowwiseRmsNorm,
    Int8SmoothQuantLinear, Int8SmoothQuantRowLinear, QServeAttention,
    QServeGatedMLP, QServeMLP, QServeRmsNorm, SmoothQuantAttention,
    SmoothQuantGatedMLP, SmoothQuantLayerNorm, SmoothQuantMLP,
    SmoothQuantRmsNorm, WeightOnlyGroupwiseQuantColumnLinear,
    WeightOnlyGroupwiseQuantRowLinear, WeightOnlyQuantColumnLinear,
    WeightOnlyQuantEmbedding, WeightOnlyQuantRowLinear)
# isort: on
from .mode import W8A8_SQ_PLUGIN_LIST, QuantAlgo, QuantMode


def quantize_layers(
    model,
    quant_config: QuantConfig,
    quant_map,
    preprocess_init_params=None,
):
    exclude_modules = quant_config.exclude_modules
    if exclude_modules is None:
        exclude_modules = [
            '*lm_head',
            '*router',
            '*vocab_embedding',
            '*position_embedding',
            '*block_embedding',
            '*shared_expert_gate',
        ]

    for name, module, parent in model.named_modules_with_parent():
        module_name = name.rsplit('.', 1)[-1]
        is_excluded = False
        quant_cls = None

        # handle exclusion
        for exclude_module in exclude_modules:
            if fnmatch.fnmatchcase(name, exclude_module):
                is_excluded = True
                break

        # MoE modules are quantized on their constructor, so they must always
        # be re-created with the appropriate quant_mode. When excluded,
        # re-create with quant_mode 0.
        # We need to handle it specially, we may want to redesign MoE implementation
        if isinstance(module, MixtureOfExperts):
            quant_cls = type(module)
        elif not is_excluded:
            for cls in quant_map:
                if isinstance(module, cls):
                    quant_cls = quant_map[cls]
                    break

        if quant_cls:
            init_params = get_init_params(module, quant_cls)
            if isinstance(module, MixtureOfExperts):
                if is_excluded:
                    quant_mode = QuantMode(0)
                else:
                    quant_mode = quant_config.quant_mode
                init_params["quant_mode"] = quant_mode
            if "bias" in init_params and not isinstance(module,
                                                        MixtureOfExperts):
                init_params["bias"] = init_params["bias"] is not None
            if isinstance(module, ColumnLinear):
                init_params[
                    "out_features"] = module.out_features * module.tp_size
            elif isinstance(module, RowLinear):
                init_params["in_features"] = module.in_features * module.tp_size
            if preprocess_init_params is not None:
                preprocess_init_params(init_params, name, module)
            quant_layer = quant_cls(**init_params)
            if parent is not None:
                setattr(parent, module_name, quant_layer)
            else:
                model = quant_layer

    setattr(model, 'quant_mode', quant_config.quant_mode)
    return model


def weight_only_quantize(model, quant_config: QuantConfig, model_config=None):
    assert quant_config.quant_mode.is_weight_only()

    try:
        model_cfg = model.config
    except AttributeError:
        model_cfg = model_config

    quant_map = {
        ColumnLinear: WeightOnlyQuantColumnLinear,
        RowLinear: WeightOnlyQuantRowLinear,
        Embedding: WeightOnlyQuantEmbedding,
    }

    def preprocess_init_params(init_params, name, module):
        init_params["quant_mode"] = quant_config.quant_mode
        if isinstance(module, ColumnLinear):
            module_name = name.rsplit('.', 1)[-1]
            init_params["transb"] = module_name == "lm_head"
        if "tp_rank" in init_params:
            init_params["tp_rank"] = model_cfg.mapping.tp_rank

    model = quantize_layers(
        model,
        quant_config,
        quant_map,
        preprocess_init_params,
    )
    return model


def weight_only_groupwise_quantize(model,
                                   quant_config: QuantConfig,
                                   model_config=None):
    assert quant_config.quant_mode.is_weight_only()

    try:
        model_cfg = model.config
    except AttributeError:
        model_cfg = model_config

    quant_map = {
        ColumnLinear: WeightOnlyGroupwiseQuantColumnLinear,
        RowLinear: WeightOnlyGroupwiseQuantRowLinear,
        MixtureOfExperts: MixtureOfExperts,
    }

    def preprocess_init_params(init_params, name, module):
        init_params["group_size"] = quant_config.group_size
        init_params["pre_quant_scale"] = quant_config.pre_quant_scale
        init_params["zero"] = quant_config.has_zero_point
        init_params[
            "use_w4a8_awq"] = quant_config.quant_algo == QuantAlgo.W4A8_AWQ
        init_params[
            "use_int8_weight"] = quant_config.quant_algo == QuantAlgo.W8A16_GPTQ
        if "tp_rank" in init_params:
            init_params["tp_rank"] = model_cfg.mapping.tp_rank

    model = quantize_layers(
        model,
        quant_config,
        quant_map,
        preprocess_init_params,
    )
    return model


def smooth_quantize_ootb(
    model,
    quant_config: QuantConfig,
):
    quant_map = {
        ColumnLinear: Int8SmoothQuantLinear,
        RowLinear: Int8SmoothQuantRowLinear,
    }

    model = quantize_layers(
        model,
        quant_config,
        quant_map,
    )
    return model


def smooth_quantize_plugin(model, quant_mode):
    quant_map = {
        RmsNorm: SmoothQuantRmsNorm,
        LayerNorm: SmoothQuantLayerNorm,
        GatedMLP: SmoothQuantGatedMLP,
        MLP: SmoothQuantMLP,
        Attention: SmoothQuantAttention,
    }
    for name, layer, parent in model.named_modules_with_parent():
        layer_name = name.rsplit('.', 1)[-1]
        if layer_name in ['ln_f', 'ln_embed']:
            continue

        quant_cls = None
        for cls in quant_map:
            if isinstance(layer, cls):
                quant_cls = quant_map[cls]
                break

        if quant_cls is None:
            continue

        init_params = get_init_params(layer, quant_cls)
        init_params["quant_mode"] = quant_mode
        if isinstance(layer, Attention):
            init_params[
                "num_attention_heads"] = layer.num_attention_heads * layer.tp_size
        quant_layer = quant_cls(**init_params)
        if parent is not None:
            setattr(parent, layer_name, quant_layer)
        else:
            model = quant_layer

    setattr(model, 'quant_mode', quant_mode)
    return model


def smooth_quantize(model, quant_config: QuantConfig):
    assert quant_config.quant_mode.has_act_and_weight_quant()
    if quant_config.quant_algo in W8A8_SQ_PLUGIN_LIST:
        return smooth_quantize_plugin(model, quant_config.quant_mode)
    else:
        return smooth_quantize_ootb(model, quant_config)


def fp8_quantize(model, quant_config: QuantConfig):
    assert quant_config.quant_mode.has_fp8_qdq()

    quant_map = {
        ColumnLinear: FP8Linear,
        RowLinear: FP8RowLinear,
        MixtureOfExperts: MixtureOfExperts,
    }

    model = quantize_layers(
        model,
        quant_config,
        quant_map,
    )
    return model


def fp8_rowwise_quantize(model, quant_config: QuantConfig):
    assert quant_config.quant_mode.has_fp8_rowwise()

    quant_cls_map = {
        RmsNorm: Fp8RowwiseRmsNorm,
        LayerNorm: Fp8RowwiseLayerNorm,
        GatedMLP: Fp8RowwiseGatedMLP,
        MLP: Fp8RowwiseMLP,
        Attention: Fp8RowwiseAttention,
    }

    exclude_modules = quant_config.exclude_modules
    if exclude_modules is None:
        exclude_modules = []
    # Always exclude these modules for FP8 rowwise
    exclude_modules = list(
        set(exclude_modules + ['*ln_f', '*ln_embed', '*lm_head']))

    def extract_layer_idx(name):
        ss = name.split('.')
        for s in ss:
            if s.isdigit():
                return int(s)
        return None

    # Meta's LLaMA 3.1 recipe:
    # (1) Skip quantization for the first and last Transformer layers
    # (2) Skip quantization for the Attention layers
    if quant_config.use_meta_recipe:
        exclude_modules.extend(['*input_layernorm', '*attention'])

    for name, layer, parent in model.named_modules_with_parent():
        module_name = name.rsplit('.', 1)[-1]

        if quant_config.use_meta_recipe:
            local_layer_idx = extract_layer_idx(name)
            mapping = model.config.mapping
            layers_range = mapping.pp_layers(model.config.num_hidden_layers)
            if mapping.is_first_pp_rank() and local_layer_idx == 0:
                continue
            if mapping.is_last_pp_rank(
            ) and local_layer_idx == len(layers_range) - 1:
                continue

        quant_cls = None
        for cls in quant_cls_map:
            if isinstance(layer, cls):
                quant_cls = quant_cls_map[cls]
                break
        if quant_cls is None:
            continue

        is_excluded = False
        for exclude_module in exclude_modules:
            if fnmatch.fnmatchcase(name, exclude_module):
                is_excluded = True
                break
        if is_excluded:
            continue

        init_params = get_init_params(layer, quant_cls)
        init_params["quant_mode"] = quant_config.quant_mode
        if isinstance(layer, Attention):
            init_params[
                "num_attention_heads"] = layer.num_attention_heads * layer.tp_size
        quant_layer = quant_cls(**init_params, clamp_val=quant_config.clamp_val)
        if parent is not None:
            setattr(parent, module_name, quant_layer)
        else:
            model = quant_layer

    setattr(model, 'quant_mode', quant_config.quant_mode)
    return model


# TODO: These functions should be moved to ModelOpt.
def qserve_quantize_weight_per_group(linear_weight: torch.HalfTensor,
                                     s1_scales: torch.FloatTensor,
                                     s2_scales: torch.FloatTensor,
                                     s2_szeros: torch.FloatTensor,
                                     group_size: int) -> torch.CharTensor:
    out_features = linear_weight.shape[0]
    in_features = linear_weight.shape[1]

    # Step 1: Quantize the weights to int8
    linear_weight = linear_weight.div(
        s1_scales.reshape(out_features, 1).to(linear_weight.device))
    linear_weight = linear_weight.round()
    # assert linear_weight.min() >= -119 and linear_weight.max() <= 119, "Stage 1: Quantized weight out of range" # 119 is the "magic" number
    assert (linear_weight.min() >= -128 and linear_weight.max()
            <= 127), "Stage 1: Quantized weight out of range"

    # Step 2: Quantize the weights to int4
    linear_weight = linear_weight.reshape(out_features,
                                          in_features // group_size, group_size)
    s2_szeros = s2_szeros.reshape(out_features, in_features // group_size,
                                  1).to(torch.float16).to(linear_weight.device)
    s2_scales = s2_scales.reshape(out_features, in_features // group_size,
                                  1).to(torch.float16).to(linear_weight.device)
    linear_weight = linear_weight.add(s2_szeros).div(s2_scales).round()
    assert (linear_weight.min() >= 0 and linear_weight.max()
            <= 15), "Stage 2: Quantized weight out of range"

    qweight = linear_weight.reshape(out_features, in_features).to(torch.int8)
    return qweight


def qserve_quantize_weight_per_channel(
        linear_weight: torch.HalfTensor, s1_scales: torch.FloatTensor,
        s1_szeros: torch.FloatTensor) -> torch.CharTensor:
    out_features = linear_weight.shape[0]
    in_features = linear_weight.shape[1]

    # Step 1: Quantize the weights to int4
    s1_scales = s1_scales.reshape(out_features, 1).to(linear_weight.device)
    s1_szeros = s1_szeros.reshape(out_features, 1).to(linear_weight.device)

    qweight = linear_weight.add(s1_szeros).div(s1_scales).round()
    assert (qweight.min() >= 0
            and qweight.max() <= 15), "Quantized weight out of range"

    return qweight.reshape(out_features, in_features).to(torch.int8)


# Pack the quantized weights, scales and zeros and apply the reordering required by QServe kernels.
# Return: processed [qweight, s1_scales, s2_scales, s2_zeros]
def qserve_pack_reorder_per_group(qweight: torch.CharTensor,
                                  s1_scales: torch.FloatTensor,
                                  s2_scales: torch.FloatTensor,
                                  s2_szeros: torch.FloatTensor, group_size):
    out_features = qweight.shape[0]
    in_features = qweight.shape[1]

    outputs = []

    s1_scales = s1_scales.reshape(out_features).to(torch.float16)
    s2_szeros = s2_szeros.reshape(out_features,
                                  in_features // group_size).to(torch.int8)
    s2_scales = s2_scales.reshape(out_features,
                                  in_features // group_size).to(torch.int8)

    # Step 3: Pack the quantized weights to real quantized weights
    # ---- Repack the weight ---- #
    assert qweight.dtype == torch.int8
    # pack to M // 32, K // 32, (8, 4), ([2], 2, 2, 4)
    W_unpack_reorder = (qweight.reshape(
        out_features // 32,
        2,
        2,
        8,
        in_features // 32,
        2,
        4,
        4,
    ).permute(0, 4, 3, 6, 1, 5, 2, 7).contiguous())
    W_unpack_reorder = (W_unpack_reorder.permute(0, 1, 2, 3, 5, 6, 7,
                                                 4).contiguous().to(torch.int8))
    # B_fp16_reorder = B_fp16_reorder[:, :, :, :, :, :, [3, 2, 1, 0]].contiguous()
    # [16, 0, 17, 1, ...]
    W_unpack_repacked = (W_unpack_reorder[..., 1] << 4) + W_unpack_reorder[...,
                                                                           0]
    W_unpack_repacked = W_unpack_repacked.reshape(out_features // 32,
                                                  in_features // 32, 32, 16)
    W_unpack_repacked = W_unpack_repacked.reshape(out_features,
                                                  in_features // 2)

    outputs.append(W_unpack_repacked)

    # for the last dimension, organize as 0, 8, 16, 24, 1, 9, 17, 25, ... following the requirement of tensor core gemm
    # ---- Pack the scales ---- #
    outputs.append(s1_scales.reshape(out_features))

    s2_scales = (s2_scales.reshape(out_features, in_features //
                                   group_size).transpose(0, 1).contiguous())
    s2_scales = s2_scales.reshape(in_features // group_size, out_features // 32,
                                  32)
    s2_scales = (s2_scales.reshape(in_features // group_size,
                                   out_features // 32, 4,
                                   8).transpose(-2, -1).contiguous())
    s2_scales = s2_scales.reshape(in_features // group_size,
                                  out_features).contiguous()
    outputs.append(s2_scales)

    # ---- Pack the zeros ---- #
    s2_szeros = (s2_szeros.reshape(out_features, in_features //
                                   group_size).transpose(0, 1).contiguous())
    s2_szeros = s2_szeros.reshape(in_features // group_size, out_features // 32,
                                  32)
    s2_szeros = (s2_szeros.reshape(in_features // group_size,
                                   out_features // 32, 4,
                                   8).transpose(-2, -1).contiguous())
    s2_szeros = (s2_szeros.reshape(in_features // group_size,
                                   out_features).contiguous())

    # (q - s2_zeros) * s2_scales = q * s2_scales - s2_zeros * s2_scales,
    # We convert the s2_zeros -> -s2_zeros * s2_scales
    s2_szeros = (-s2_szeros).int()  # It has been pre-scaled in DeepCompressor
    s2_szeros = s2_szeros.to(torch.int8)

    outputs.append(s2_szeros)

    return outputs


def qserve_pack_reorder_per_channel(qweight: torch.CharTensor,
                                    s1_scales: torch.FloatTensor,
                                    s1_szeros: torch.FloatTensor):
    out_features = qweight.shape[0]
    in_features = qweight.shape[1]

    outputs = []

    # ---- Repack the weight ---- #
    assert qweight.dtype == torch.int8
    # pack to M // 32, K // 32, (8, 4), ([2], 2, 2, 4)
    W_unpack_reorder = (qweight.reshape(
        out_features // 32,
        2,
        2,
        8,
        in_features // 32,
        2,
        4,
        4,
    ).permute(0, 4, 3, 6, 1, 5, 2, 7).contiguous())
    W_unpack_reorder = (W_unpack_reorder.permute(0, 1, 2, 3, 5, 6, 7,
                                                 4).contiguous())
    # B_fp16_reorder = B_fp16_reorder[:, :, :, :, :, :, [3, 2, 1, 0]].contiguous()
    # [16, 0, 17, 1, ...]
    W_unpack_repacked = (W_unpack_reorder[..., 1] << 4) + W_unpack_reorder[...,
                                                                           0]
    W_unpack_repacked = W_unpack_repacked.reshape(out_features // 32,
                                                  in_features // 32, 32, 16)
    W_unpack_repacked = W_unpack_repacked.reshape(out_features, in_features //
                                                  2).contiguous()

    outputs.append(W_unpack_repacked)

    # ---- Pack the scales and zeros ---- #
    s1_scales = s1_scales.reshape(out_features).contiguous()
    outputs.append(s1_scales.half())

    s1_szeros = s1_szeros.reshape(out_features).contiguous().half()
    outputs.append(s1_szeros)

    return outputs


# TODO: Duplicates smooth_quantize and quantize_layers
def qserve_quantize(model, quant_config: QuantConfig):
    quant_mode = quant_config.quant_mode
    assert quant_config.quant_mode.is_qserve_w4a8()

    quant_map = {
        RmsNorm: QServeRmsNorm,
        LayerNorm: QServeRmsNorm,
        GatedMLP: QServeGatedMLP,
        MLP: QServeMLP,
        Attention: QServeAttention,
    }

    for name, layer, parent in model.named_modules_with_parent():
        layer_name = name.rsplit('.', 1)[-1]
        if layer_name in ['ln_f', 'ln_embed']:
            continue

        quant_cls = None
        for cls in quant_map:
            if isinstance(layer, cls):
                quant_cls = quant_map[cls]
                break

        if quant_cls is None:
            continue

        init_params = get_init_params(layer, quant_cls)
        init_params["quant_mode"] = quant_mode
        if isinstance(layer, Attention):
            init_params[
                "num_attention_heads"] = layer.num_attention_heads * layer.tp_size
        quant_layer = quant_cls(**init_params)
        if parent is not None:
            setattr(parent, layer_name, quant_layer)
        else:
            model = quant_layer

    setattr(model, 'quant_mode', quant_mode)
    return model


def fp4_quantize(model, quant_config: QuantConfig):
    assert quant_config.quant_mode.has_nvfp4()
    quant_map = {
        ColumnLinear: FP4Linear,
        RowLinear: FP4RowLinear,
        MixtureOfExperts: MixtureOfExperts,
    }

    model = quantize_layers(
        model,
        quant_config,
        quant_map,
    )
    return model


# Now consider the kv cache is enabled for all layers
def kv_cache_quantize(model):
    for name, module in model.named_modules():
        if isinstance(module,
                      (Attention, SmoothQuantAttention, Fp8RowwiseAttention)):
            # for dequant
            module.kv_cache_scaling_factor = Parameter(shape=(1, ),
                                                       dtype='float32')
            # for quant
            module.kv_cache_rcp_scaling_factor = Parameter(shape=(1, ),
                                                           dtype='float32')
    return model


def quantize(model, quant_config: Union[QuantConfig, LayerQuantConfig]):

    for name, module, parent in model.named_modules_with_parent():

        if quant_config.quant_algo == QuantAlgo.MIXED_PRECISION:
            layer_quant_mode = quant_config.layer_quant_mode(name)
        else:
            layer_quant_mode = quant_config.layer_quant_mode
        if layer_quant_mode == QuantMode(0):
            continue

        layer_quant_cfg = quant_config._get_quant_cfg(name)

        if layer_quant_mode.has_fp8_qdq():
            module = fp8_quantize(module, layer_quant_cfg)
        elif layer_quant_mode.has_fp8_rowwise():
            module = fp8_rowwise_quantize(module, layer_quant_cfg)
        elif layer_quant_mode.is_qserve_w4a8():
            module = qserve_quantize(module, quant_config)
        elif layer_quant_mode.has_nvfp4():
            module = fp4_quantize(module, layer_quant_cfg)
        elif layer_quant_mode.has_act_and_weight_quant():
            module = smooth_quantize(module, layer_quant_cfg)
        elif layer_quant_mode.is_weight_only():
            if layer_quant_mode.has_per_group_scaling():
                module = weight_only_groupwise_quantize(module, layer_quant_cfg,
                                                        model.config)
            else:
                module = weight_only_quantize(module, layer_quant_cfg,
                                              model.config)

        if parent is not None:  # for per layer
            module_name = name.rsplit('.', 1)[-1]
            setattr(parent, module_name, module)
        else:  # for all layer
            model = module
            break

    if quant_config.quant_mode.has_kv_cache_quant():
        model = kv_cache_quantize(model)

    setattr(model, 'quant_mode', quant_config.quant_mode)
    return model

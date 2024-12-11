import torch

from ..._utils import pad_vocab_size, str_dtype_to_torch


def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return torch.chunk(v, tp_size)[idx].contiguous()
    else:
        return torch.chunk(v, tp_size, dim=dim)[idx].contiguous()


def load_weights_from_hf_model(hf_model, config):
    torch_dtype = str_dtype_to_torch(config.dtype)
    hf_state_dict = hf_model.state_dict()
    weights = {}
    is_weight_only = config.quant_mode.is_weight_only()
    if config.quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif config.quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    # replace key name
    for key, value in hf_state_dict.items():
        # Decoder Layers
        if "model.layers." in key:
            key = key.replace("model.layers.", "transformer.layers.")
            key = key.replace("self_attn.", "attention.")
            key = key.replace("mlp.fc1.", "mlp.fc.")
            key = key.replace("mlp.fc2.", "mlp.proj.")
        # Embedding
        key = key.replace("model.embed_tokens.weight",
                          "transformer.vocab_embedding.weight")
        # Final Layer norm
        key = key.replace("model.final_layernorm.", "transformer.ln_f.")

        weights[key] = value.to(torch_dtype).cpu()

    # merge qkv weights
    qkv_keys = ["q_proj", "k_proj", "v_proj"]

    scales = {}
    for key in hf_state_dict.keys():
        if 'self_attn.q_proj.weight' in key:
            prefix = key.split('self_attn')[0].replace("model.layers.",
                                                       "transformer.layers.")

            # [(num_heads x q)|(num_heads x k)|(num_heads x v), hidden_size]
            qkv_weights = []
            qkv_bias = []

            for k in qkv_keys:
                split_w = split(weights.pop(f"{prefix}attention.{k}.weight"),
                                config.mapping.tp_size, config.mapping.tp_rank)

                qkv_weights.append(split_w)
                split_b = split(weights.pop(f"{prefix}attention.{k}.bias"),
                                config.mapping.tp_size, config.mapping.tp_rank)
                qkv_bias.append(split_b)
            v = torch.cat(qkv_weights, dim=0)
            if is_weight_only:
                processed_torch_weights, torch_weight_scales = \
                    torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                        v.t().contiguous().cpu(), plugin_weight_only_quant_type)
                weights[
                    f"{prefix}attention.qkv.weight"] = processed_torch_weights
                scales[
                    f"{prefix}attention.qkv.per_channel_scale"] = torch_weight_scales
            else:
                weights[f"{prefix}attention.qkv.weight"] = v
            weights[f"{prefix}attention.qkv.bias"] = torch.cat(qkv_bias, dim=0)

    tp_rank = config.mapping.tp_rank
    for weight_name in weights:
        loaded_weight = weights[weight_name]
        if "attention.dense.weight" in weight_name or "mlp.proj.weight" in weight_name:  # RowLinear
            v = split(loaded_weight,
                      config.mapping.tp_size,
                      config.mapping.tp_rank,
                      dim=1)
            if is_weight_only:
                processed_torch_weights, torch_weight_scales = \
                    torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                        v.t().contiguous().cpu(), plugin_weight_only_quant_type)
                weights[weight_name] = processed_torch_weights
                scales[weight_name.replace(
                    '.weight', '.per_channel_scale')] = torch_weight_scales
            else:
                weights[weight_name] = v
        elif "mlp.fc." in weight_name:

            v = split(loaded_weight, config.mapping.tp_size,
                      config.mapping.tp_rank)
            if is_weight_only and "mlp.fc.weight" in weight_name:
                processed_torch_weights, torch_weight_scales = \
                    torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                        v.t().contiguous().cpu(), plugin_weight_only_quant_type)
                weights[weight_name] = processed_torch_weights
                scales[weight_name.replace(
                    '.weight', '.per_channel_scale')] = torch_weight_scales
            else:
                weights[weight_name] = v
        elif "lm_head." in weight_name:
            output_dim = 0
            shard_size = loaded_weight.shape[output_dim]
            tp_rank * shard_size
            vocab_size = loaded_weight.shape[output_dim]

            if shard_size % config.mapping.tp_size != 0:
                pad_width = pad_vocab_size(vocab_size,
                                           config.mapping.tp_size) - vocab_size
                loaded_weight = torch.nn.functional.pad(loaded_weight,
                                                        (0, 0, 0, pad_width),
                                                        'constant',
                                                        value=0)
            weights[weight_name] = split(loaded_weight, config.mapping.tp_size,
                                         config.mapping.tp_rank)

    weights.update(scales)

    return weights


def convert_hf_config(hf_config, dtype, args):
    config = {
        'architecture': hf_config.architectures[0],
        'dtype': dtype,
        'num_hidden_layers': hf_config.num_hidden_layers,
        'num_attention_heads': hf_config.num_key_value_heads,
        'rotary_pct': hf_config.partial_rotary_factor,
        'rope_theta': hf_config.rope_theta,
        'hidden_size': hf_config.hidden_size,
        'intermediate_size': hf_config.intermediate_size,
        'vocab_size': hf_config.vocab_size,
        'position_embedding_type': 'rope_gpt_neox',
        'max_position_embeddings': hf_config.max_position_embeddings,
        'hidden_act': hf_config.hidden_act,
        'mapping': {
            'world_size': args.tp_size * args.pp_size,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size,
        }
    }
    return config

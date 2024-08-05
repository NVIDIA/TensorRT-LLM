import torch

from ..._utils import str_dtype_to_torch


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

    for key in hf_state_dict.keys():
        if 'self_attn.q_proj.weight' in key:
            prefix = key.split('self_attn')[0].replace("model.layers.",
                                                       "transformer.layers.")

            # [(num_heads x q)|(num_heads x k)|(num_heads x v), hidden_size]
            qkv_weights = []
            qkv_bias = []
            for k in qkv_keys:
                qkv_weights.append(weights.pop(f"{prefix}attention.{k}.weight"))
                qkv_bias.append(weights.pop(f"{prefix}attention.{k}.bias"))

            weights[f"{prefix}attention.qkv.weight"] = torch.cat(qkv_weights,
                                                                 dim=0)
            weights[f"{prefix}attention.qkv.bias"] = torch.cat(qkv_bias, dim=0)
    if is_weight_only:
        kw_list = [
            'attention.dense.weight', 'attention.qkv.weight', 'mlp.fc.weight',
            'mlp.proj.weight'
        ]
        for key in [
                weight_name for kw in kw_list for weight_name in weights
                if kw in weight_name
        ]:
            v = weights[key].t().contiguous().cpu()
            processed_torch_weights, torch_weight_scales = \
                torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                    v, plugin_weight_only_quant_type)
            weights[key] = processed_torch_weights
            weights[key.replace('.weight',
                                '.per_channel_scale')] = torch_weight_scales

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
        'share_embedding_table': False,
        'mapping': {
            'world_size': args.tp_size * args.pp_size,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size,
        }
    }
    return config

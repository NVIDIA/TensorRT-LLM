import torch

from ..._utils import str_dtype_to_torch


def convert_hf_weights(hf_model, dtype, **kwargs):
    torch_dtype = str_dtype_to_torch(dtype)
    hf_state_dict = hf_model.state_dict()
    weights = {}

    # replace key name
    for key, value in hf_state_dict.items():
        # Decoder Layers
        orig_key = key
        if "model.layers." in key:
            key = key.replace("model.layers.", "transformer.layers.")
            #Attention
            key = key.replace("self_attn.", "attention.")
            key = key.replace("Wqkv.weight", "qkv.weight")
            key = key.replace("qkv_proj.", "qkv.")  #128k
            #MLP
            key = key.replace("mlp.fc1.", "mlp.fc.")
            key = key.replace("mlp.fc2.", "mlp.proj.")
            key = key.replace("mlp.gate_up_proj.", "mlp.fc.")
            key = key.replace("mlp.up_proj.", "mlp.gate.")  #128k
            key = key.replace("mlp.down_proj.", "mlp.proj.")  #128k
            key = key.replace("mlp.gate_proj.", "mlp.fc.")  #128k
            key = key.replace("o_proj.", "dense.")  #128k
            #Layer norm
            key = key.replace("post_attention_layernorm.",
                              "post_layernorm.")  #128k

        # Embedding
        key = key.replace("model.embed_tokens.weight",
                          "transformer.vocab_embedding.weight")
        # Final Layer norm
        key = key.replace("model.final_layernorm.", "transformer.ln_f.")
        key = key.replace("model.norm.", "transformer.ln_f.")  #128k

        if "mlp.gate_up_proj." in orig_key:  #4k
            original_weights = value.contiguous().clone()
            half_split = original_weights.shape[0] // 2
            first_half, second_half = original_weights[:
                                                       half_split, :], original_weights[
                                                           half_split:, :]
            # Swap the halves
            value = torch.cat((second_half, first_half), dim=0)

        if "q_proj" in key:  #128k
            q_param = value
            k_param = hf_state_dict[orig_key.replace("q_proj", "k_proj")]
            v_param = hf_state_dict[orig_key.replace("q_proj", "v_proj")]
            value = torch.cat([q_param, k_param, v_param], dim=0)
            key = key.replace("q_proj.weight", "qkv.weight")
        elif "k_proj" in key or "v_proj" in key:
            continue
        weights[key] = value.to(torch_dtype).cpu()

    return weights


def convert_hf_config(hf_config, dtype, **kwargs):
    config = {
        'architecture': "Phi3ForCausalLM",
        'dtype': dtype,
        'num_hidden_layers': hf_config.num_hidden_layers,
        'num_attention_heads': hf_config.num_key_value_heads,
        'rope_theta': hf_config.rope_theta,
        'hidden_size': hf_config.hidden_size,
        'intermediate_size': hf_config.intermediate_size,
        'vocab_size': hf_config.vocab_size,
        'max_position_embeddings': hf_config.max_position_embeddings,
        'hidden_act': hf_config.hidden_act,
        'share_embedding_table': False,
        'layer_norm_eps': hf_config.rms_norm_eps,
    }
    if hf_config.max_position_embeddings >= 128000:
        config.update({
            'original_max_position_embeddings':
            hf_config.original_max_position_embeddings,
            'longrope_scaling_short_factors':
            hf_config.rope_scaling["short_factor"],
            'longrope_scaling_long_factors':
            hf_config.rope_scaling["long_factor"]
        })
    if config["hidden_act"] == "silu":
        config["hidden_act"] = "swiglu"
    return config

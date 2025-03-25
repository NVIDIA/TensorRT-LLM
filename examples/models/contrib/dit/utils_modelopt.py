import os
import re
from collections import OrderedDict

import modelopt.torch.opt as mto
import torch
from diffusers import DiTPipeline
from modelopt.torch.export.layer_utils import (get_activation_scaling_factor,
                                               get_weight_scaling_factor)
from modelopt.torch.export.model_config_utils import to_quantized_weight
from torchvision.datasets.utils import download_url

HUGGINGFACE_TO_FACEBOOK_DIT_NAME_MAPPING = {
    "^transformer_blocks.(\d+).norm1.emb.class_embedder.embedding_table.weight$":
    "y_embedder.embedding_table.weight",
    "^transformer_blocks.(\d+).norm1.emb.timestep_embedder.linear_1.weight$":
    "t_embedder.mlp.0.weight",
    "^transformer_blocks.(\d+).norm1.emb.timestep_embedder.linear_1.bias$":
    "t_embedder.mlp.0.bias",
    "^transformer_blocks.(\d+).norm1.emb.timestep_embedder.linear_2.weight$":
    "t_embedder.mlp.2.weight",
    "^transformer_blocks.(\d+).norm1.emb.timestep_embedder.linear_2.bias$":
    "t_embedder.mlp.2.bias",
    "^pos_embed.proj.weight$":
    "x_embedder.proj.weight",
    "^pos_embed.proj.bias$":
    "x_embedder.proj.bias",
    "^transformer_blocks.(\d+).attn1.to_qkv.weight$":
    "blocks.*.attn.qkv.weight",
    "^transformer_blocks.(\d+).attn1.to_qkv.bias$":
    "blocks.*.attn.qkv.bias",
    "^transformer_blocks.(\d+).attn1.to_out.0.weight$":
    "blocks.*.attn.proj.weight",
    "^transformer_blocks.(\d+).attn1.to_out.0.bias$":
    "blocks.*.attn.proj.bias",
    "^transformer_blocks.(\d+).ff.net.0.proj.weight$":
    "blocks.*.mlp.fc1.weight",
    "^transformer_blocks.(\d+).ff.net.0.proj.bias$":
    "blocks.*.mlp.fc1.bias",
    "^transformer_blocks.(\d+).ff.net.2.weight$":
    "blocks.*.mlp.fc2.weight",
    "^transformer_blocks.(\d+).ff.net.2.bias$":
    "blocks.*.mlp.fc2.bias",
    "^transformer_blocks.(\d+).norm1.linear.weight$":
    "blocks.*.adaLN_modulation.1.weight",
    "^transformer_blocks.(\d+).norm1.linear.bias$":
    "blocks.*.adaLN_modulation.1.bias",
    "^proj_out_2.weight$":
    "final_layer.linear.weight",
    "^proj_out_2.bias$":
    "final_layer.linear.bias",
    "^proj_out_1.weight$":
    "final_layer.adaLN_modulation.1.weight",
    "^proj_out_1.bias$":
    "final_layer.adaLN_modulation.1.bias",
    "^transformer_blocks.(\d+).norm1.emb.timestep_embedder.linear_1.weights_scaling_factor$":
    "t_embedder.mlp.0.weights_scaling_factor",
    "^transformer_blocks.(\d+).norm1.emb.timestep_embedder.linear_1.activation_scaling_factor":
    "t_embedder.mlp.0.activation_scaling_factor",
    "^transformer_blocks.(\d+).norm1.emb.timestep_embedder.linear_2.weights_scaling_factor":
    "t_embedder.mlp.2.weights_scaling_factor",
    "^transformer_blocks.(\d+).norm1.emb.timestep_embedder.linear_2.activation_scaling_factor":
    "t_embedder.mlp.2.activation_scaling_factor",
    "^transformer_blocks.(\d+).attn1.to_qkv.weights_scaling_factor$":
    "blocks.*.attn.qkv.weights_scaling_factor",
    "^transformer_blocks.(\d+).attn1.to_qkv.activation_scaling_factor$":
    "blocks.*.attn.qkv.activation_scaling_factor",
    "^transformer_blocks.(\d+).attn1.to_out.0.weights_scaling_factor$":
    "blocks.*.attn.proj.weights_scaling_factor",
    "^transformer_blocks.(\d+).attn1.to_out.0.activation_scaling_factor$":
    "blocks.*.attn.proj.activation_scaling_factor",
    "^transformer_blocks.(\d+).ff.net.0.proj.weights_scaling_factor$":
    "blocks.*.mlp.fc1.weights_scaling_factor",
    "^transformer_blocks.(\d+).ff.net.0.proj.activation_scaling_factor$":
    "blocks.*.mlp.fc1.activation_scaling_factor",
    "^transformer_blocks.(\d+).ff.net.2.weights_scaling_factor$":
    "blocks.*.mlp.fc2.weights_scaling_factor",
    "^transformer_blocks.(\d+).ff.net.2.activation_scaling_factor$":
    "blocks.*.mlp.fc2.activation_scaling_factor",
    "^transformer_blocks.(\d+).norm1.linear.weights_scaling_factor$":
    "blocks.*.adaLN_modulation.1.weights_scaling_factor",
    "^transformer_blocks.(\d+).norm1.linear.activation_scaling_factor$":
    "blocks.*.adaLN_modulation.1.activation_scaling_factor",
    "^proj_out_2.weights_scaling_factor$":
    "final_layer.linear.weights_scaling_factor",
    "^proj_out_2.activation_scaling_factor$":
    "final_layer.linear.activation_scaling_factor",
    "^proj_out_1.weights_scaling_factor$":
    "final_layer.adaLN_modulation.1.weights_scaling_factor",
    "^proj_out_1.activation_scaling_factor$":
    "final_layer.adaLN_modulation.1.activation_scaling_factor",
}


def convert_amax_to_scaling_factor(model, state_dict):
    ret_dict = state_dict.copy()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            activation_scaling_factor = get_activation_scaling_factor(module)
            weight_scaling_factor = get_weight_scaling_factor(module)
            if activation_scaling_factor:
                ret_dict[
                    f'{name}.activation_scaling_factor'] = activation_scaling_factor
            if weight_scaling_factor:
                ret_dict[
                    f'{name}.weights_scaling_factor'] = weight_scaling_factor

            weight = module.weight.detach().cpu()
            if weight_scaling_factor:
                # only module with valid weight scaling factor
                weight = to_quantized_weight(
                    weight=weight,
                    weights_scaling_factor=weight_scaling_factor,
                    quantization="fp8",
                )
            # replace the quantized weight
            ret_dict[f'{name}.weight'] = weight
    return ret_dict


def get_weights_map(state_dict):

    def _get_fb_dit_name(dit_name):
        for k, v in HUGGINGFACE_TO_FACEBOOK_DIT_NAME_MAPPING.items():
            m = re.match(k, dit_name)
            if m is not None:
                if "*" in v:
                    v = v.replace("*", m.groups()[0])
                return v
        return dit_name

    weights_map = OrderedDict()
    for key, value in state_dict.items():
        if ("to_q." in key) or ("to_k." in key) or ("to_v." in key):
            continue
        if _get_fb_dit_name(key) in weights_map:
            continue
        else:
            weights_map[_get_fb_dit_name(key)] = value
    return weights_map


def download_model(ckpt="DiT-XL-2-512x512.pt"):
    """
    Downloads a pre-trained DiT model from the web.
    """
    if not os.path.isfile(ckpt):
        os.makedirs('pretrained_models', exist_ok=True)
        web_path = f"https://dl.fbaipublicfiles.com/DiT/models/{ckpt}"
        download_url(web_path, '.')
    model = torch.load(ckpt, map_location=lambda storage, loc: storage)
    return model


def remap_model(model_name="facebook/DiT-XL-2-512",
                quantized_ckpt="dit.quantized.pt",
                output_ckpt="dit.converted.pt",
                fuse_qkv=True,
                dtype=torch.float32):
    pipe = DiTPipeline.from_pretrained(model_name, torch_dtype=dtype)
    transformer = pipe.transformer

    # fuse qkv gemm
    if fuse_qkv:
        from diffusers.models.attention_processor import (Attention,
                                                          AttnProcessor2_0)
        for module in transformer.modules():
            if isinstance(module, Attention):
                assert (isinstance(module.processor, AttnProcessor2_0))
                module.fuse_projections(fuse=True)
    pretrained_state_dict = transformer.state_dict()

    mto.restore(transformer, quantized_ckpt)
    quantized_dict = convert_amax_to_scaling_factor(transformer,
                                                    pretrained_state_dict)
    assert set(pretrained_state_dict.keys()).issubset(set(
        quantized_dict.keys()))

    remapped_dict = get_weights_map(quantized_dict)
    # TODO: currently we use weights of pos_embed and x_embedder from official DiT because
    # in the implementation by HuggingFace, there are some modifications for these modules.
    stored_params = download_model(ckpt="DiT-XL-2-512x512.pt")
    for name in ["pos_embed", "x_embedder.proj.weight", "x_embedder.proj.bias"]:
        remapped_dict[name] = stored_params[name]

    torch.save(remapped_dict, output_ckpt)

#!/usr/bin/env python3
import argparse
import json
import logging
import math
import os
import pathlib
import re
import time
import typing

import flax.traverse_util
import h5py
import numpy as np
import safetensors.numpy
import safetensors.torch
import sentencepiece as sp
import torch
import utils.params
import utils.transformer
from easydict import EasyDict
from transformers import AutoConfig, AutoModelForCausalLM

import tensorrt_llm
from tensorrt_llm._utils import (np_bfloat16, numpy_to_torch,
                                 str_dtype_to_torch, torch_to_numpy)
from tensorrt_llm.models.convert_utils import load_calib_dataset
from tensorrt_llm.models.gemma.smoothquant import *
from tensorrt_llm.models.gemma.weight import (dummy_weights_awq,
                                              load_from_fp8_gemma,
                                              quantize_fp8_weights)
from tensorrt_llm.quantization import QuantAlgo

LOGGER = logging.getLogger("convert_checkpoint")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-type",
                        type=str,
                        choices=["jax", "keras", "torch", "hf"])
    parser.add_argument("--model-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-model-dir", type=pathlib.Path, required=True)
    parser.add_argument("--world-size",
                        type=int,
                        default=1,
                        help="world size, only support tensor parallelism now")
    parser.add_argument(
        "--use-weight-only-with-precision",
        choices=["int8", "int4", "w4a8_awq", "w4a16_awq"],
        help=
        "help='Quantize weights for the various GEMMs to INT4/INT8. Define the precision for the weights.",
    )
    parser.add_argument(
        "--use-int8-weight-only-embedding",
        action="store_true",
        help=
        "Use weight only on embedding table and lm_head. (Only supported on Hopper GPU)",
    )
    parser.add_argument("--dtype",
                        type=str,
                        choices=["float32", "bfloat16", "float16"])
    parser.add_argument(
        "--enable_fp8",
        action="store_true",
        help="Use FP8 Linear layer for Attention QKV/Dense and MLP.")
    parser.add_argument(
        "--fp8_kv_cache",
        action="store_true",
        help=
        "By default, we use dtype for KV cache. fp8_kv_cache chooses fp8 quantization for KV",
    )
    parser.add_argument(
        "--quant_ckpt_path",
        default=None,
        help=
        "Path of a directory to quantized model checkpoints in .safetensors format or \
              path of a quantized model checkpoint in .npz format")
    parser.add_argument(
        '--calib_dataset',
        type=str,
        default='ccdv/cnn_dailymail',
        help=
        "The huggingface dataset name or the local directory of the dataset for calibration."
    )
    parser.add_argument('--use_smooth_quant',
                        default=False,
                        action="store_true",
                        help="Use smooth quant.")
    parser.add_argument(
        "--calibrate_kv_cache",
        "-kv",
        action="store_true",
        help=
        "Generate scaling factors for KV cache. Used for storing KV cache in int8."
    )
    parser.add_argument(
        '--per_channel',
        default=False,
        action="store_true",
        help=
        'By default, we use a single static scaling factor for the GEMM\'s result. '
        'per_channel instead uses a different static scaling factor for each channel. '
        'The latter is usually more accurate, but a little slower.')
    parser.add_argument(
        '--per_token',
        default=False,
        action="store_true",
        help=
        'By default, we use a single static scaling factor to scale activations in the int8 range. '
        'per_token chooses at run time, and for each token, a custom scaling factor. '
        'The latter is usually more accurate, but a little slower.')
    parser.add_argument(
        "--use_smooth_quant_plugin",
        "-sq",
        type=float,
        default=None,
        help="Set the α parameter (see https://arxiv.org/pdf/2211.10438.pdf)"
        " to Smoothquant the model, and output int8 weights."
        " A good first try is 0.5. Must be in [0, 1]")
    parser.add_argument(
        '--tokenizer_dir',
        default=None,
        help='tokenizer path; defaults to jax_model_dir if left unspecified')

    args = parser.parse_args()
    args.use_embedding_sharing = True
    return args


class JAXParser:

    def load_parameters(self, checkpoint_path: pathlib.Path):
        checkpoint_path = checkpoint_path.absolute()
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        return utils.params.nest_params(
            utils.params.param_remapper(
                utils.params.load_params(checkpoint_path)))

    def embedding_weights(self, ckpt_params):
        return ckpt_params["transformer"]["embedder"]["input_embedding"]

    def get_config(self, checkpoint_path, ckpt_params, num_embed):
        return utils.transformer.TransformerConfig.from_params(
            ckpt_params, num_embed=num_embed)

    def rename_to_trt_llm(self, name: str):
        """Rename a gemma parameter name by the corresponding TRT-LLM style name."""
        prefix, name = name.split(".", maxsplit=1)
        assert prefix == "transformer"
        sub_patterns = (
            (r"embedder.input_embedding", r"vocab_embedding.weight"),
            (r"layer_(\d+).pre_attention_norm.scale",
             r"layers.\1.input_layernorm.weight"),
            (r"layer_(\d+).attn.q_einsum.w", r"layers.\1.attention.qkv.weight"),
            (r"layer_(\d+).attn.kv_einsum.w",
             None),  # drop as kv will be concatenated with q
            (r"layer_(\d+).attn.qkv_einsum.w",
             r"layers.\1.attention.qkv.weight"),
            (r"layer_(\d+).attn.attn_vec_einsum.w",
             r"layers.\1.attention.dense.weight"),
            (r"layer_(\d+).mlp.gating_einsum", r"layers.\1.mlp.fc.weight"),
            (r"layer_(\d+).mlp.linear", r"layers.\1.mlp.proj.weight"),
            (r"layer_(\d+).pre_ffw_norm.scale",
             r"layers.\1.post_layernorm.weight"),
            (r"final_norm.scale", r"ln_f.weight"),
        )

        for source, target in sub_patterns:
            if re.match(source, name):
                if target is None:
                    return target
                else:
                    name = re.sub(source, target, name)
                    return ".".join((prefix, name))
        else:
            raise ValueError(f"Don't know how to rename {prefix}.{name}")

    def flatten_params(self, params):
        new_params = flax.traverse_util.flatten_dict(params, sep=".")
        # if the dtype is bfloat16, cast to float32
        for k in new_params:
            if new_params[k].dtype != np.float32 and new_params[
                    k].dtype != np.float16:
                new_params[k] = new_params[k].astype(np.float32)
        return new_params


class KerasParser:

    def load_parameters(self, checkpoint_path: pathlib.Path):
        checkpoint_path = checkpoint_path.absolute()
        config_file = "config.json"
        weights_file = json.load(open(checkpoint_path / config_file))["weights"]
        h5_path = checkpoint_path / weights_file
        return h5py.File(h5_path, "r")

    def embedding_weights(self, ckpt_params):
        return np.array(ckpt_params["layers/reversible_embedding/vars/0"])

    def get_config(self, checkpoint_path, ckpt_params, num_embed):
        checkpoint_path = checkpoint_path.absolute()
        config_file = "config.json"
        config_old = json.load(open(checkpoint_path / config_file))["config"]
        config_new = {}
        config_new["num_layers"] = config_old["num_layers"]
        config_new["num_embed"] = config_old["vocabulary_size"]
        config_new["embed_dim"] = config_old["hidden_dim"]
        config_new["hidden_dim"] = config_old["intermediate_dim"] // 2
        config_new["num_heads"] = config_old["num_query_heads"]
        config_new["head_dim"] = config_old["head_dim"]
        config_new["num_kv_heads"] = config_old["num_key_value_heads"]
        return EasyDict(config_new)

    def rename_to_trt_llm(self, name: str):
        """Rename a gemma parameter name by the corresponding TRT-LLM style name."""
        prefix = "transformer"
        name = name.replace("/gemma_decoder_block/", "/gemma_decoder_block_0/")
        sub_patterns = (
            (r"layers/reversible_embedding/vars/0", r"vocab_embedding.weight"),
            (r"layers/gemma_decoder_block_(\d+)/pre_attention_norm/vars/0",
             r"layers.\1.input_layernorm.weight"),
            (r"layers/gemma_decoder_block_(\d+)/attention/query_dense/vars/0",
             r"layers.\1.attention.qkv.weight"),
            (r"layers/gemma_decoder_block_(\d+)/attention/key_dense/vars/0",
             None),  # drop as k will be concatenated with q
            (r"layers/gemma_decoder_block_(\d+)/attention/value_dense/vars/0",
             None),  # drop as v will be concatenated with q
            (r"layers/gemma_decoder_block_(\d+)/attention/output_dense/vars/0",
             r"layers.\1.attention.dense.weight"),
            (r"layers/gemma_decoder_block_(\d+)/gating_ffw/vars/0",
             r"layers.\1.mlp.fc.weight"),
            (r"layers/gemma_decoder_block_(\d+)/gating_ffw_2/vars/0",
             None),  # merged with above
            (r"layers/gemma_decoder_block_(\d+)/ffw_linear/vars/0",
             r"layers.\1.mlp.proj.weight"),
            (r"layers/gemma_decoder_block_(\d+)/pre_ffw_norm/vars/0",
             r"layers.\1.post_layernorm.weight"),
            (r"layers/rms_normalization/vars/0", r"ln_f.weight"),
            (r"optimizer/vars/(\d+)", None),  # Not used
        )

        for source, target in sub_patterns:
            if re.match(source, name):
                if target is None:
                    return target
                else:
                    name = re.sub(source, target, name)
                    return ".".join((prefix, name))
        else:
            raise ValueError(f"Don't know how to rename {prefix}.{name}")

    def flatten_params(self, params):
        f_params = {}

        def walk(name, obj):
            if isinstance(obj, h5py.Dataset):
                if obj.dtype == "|V2":
                    # bfloat16 case
                    f_params[name] = torch_to_numpy(
                        numpy_to_torch(np.array(obj).astype(np_bfloat16)).to(
                            torch.float32))
                else:
                    f_params[name] = np.array(obj)

        params.visititems(walk)
        return f_params


class TorchParser:

    def load_parameters(self, checkpoint_path: pathlib.Path):
        ckpt_path = list(checkpoint_path.glob('*.ckpt'))[0]
        model_params = torch.load(ckpt_path)['model_state_dict']
        model_params.pop('freqs_cis')
        return model_params

    def embedding_weights(self, ckpt_params):
        return ckpt_params["embedder.weight"]

    def get_config(self, checkpoint_path, ckpt_params, num_embed):
        checkpoint_path = checkpoint_path.absolute()
        config_file = "config.json"
        with open(checkpoint_path / config_file, 'r') as f:
            json_str = f.read()
            json_str = json_str.replace("'", "\"")
            json_str = json_str.replace(",\n}", "\n}")
            config_old = json.loads(json_str)
        config_new = {}
        config_new["num_layers"] = config_old["num_hidden_layers"]
        config_new["num_embed"] = config_old["vocab_size"]
        config_new["embed_dim"] = config_old["hidden_size"]
        config_new["hidden_dim"] = config_old["intermediate_size"]
        config_new["num_heads"] = config_old["num_attention_heads"]
        config_new["head_dim"] = config_old["head_dim"]
        config_new["num_kv_heads"] = config_old["num_key_value_heads"]
        return EasyDict(config_new)

    def rename_to_trt_llm(self, name: str):
        """Rename a gemma parameter name by the corresponding TRT-LLM style name."""
        prefix = "transformer"
        sub_patterns = (
            (r"embedder.weight", r"vocab_embedding.weight"),
            (r"model.layers.(\d+).input_layernorm.weight",
             r"layers.\1.input_layernorm.weight"),
            (r"model.layers.(\d+).self_attn.qkv_proj.weight",
             r"layers.\1.attention.qkv.weight"),
            (r"model.layers.(\d+).self_attn.o_proj.weight",
             r"layers.\1.attention.dense.weight"),
            (r"model.layers.(\d+).mlp.gate_proj.weight",
             r"layers.\1.mlp.fc.weight"),
            (r"model.layers.(\d+).mlp.up_proj.weight",
             None),  # merged with above
            (r"model.layers.(\d+).mlp.down_proj.weight",
             r"layers.\1.mlp.proj.weight"),
            (r"model.layers.(\d+).post_attention_layernorm.weight",
             r"layers.\1.post_layernorm.weight"),
            (r"model.norm.weight", r"ln_f.weight"),
        )

        for source, target in sub_patterns:
            if re.match(source, name):
                if target is None:
                    return target
                else:
                    name = re.sub(source, target, name)
                    return ".".join((prefix, name))
        else:
            raise ValueError(f"Don't know how to rename {name}")

    def flatten_params(self, params):
        f_params = {}
        for k, v in params.items():
            if v.dtype == torch.bfloat16:
                v = v.float()
            f_params[k] = torch_to_numpy(v)
        return f_params


class HfParser:

    def load_parameters(self, checkpoint_path: pathlib.Path):
        hf_model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            device_map='auto',
            torch_dtype='auto',
            trust_remote_code=True,
        )
        model_params = dict(hf_model.named_parameters())
        return model_params

    def embedding_weights(self, ckpt_params):
        return ckpt_params['model.embed_tokens.weight']

    def get_config(self, checkpoint_path, ckpt_params, num_embed):
        hf_config = AutoConfig.from_pretrained(
            checkpoint_path, trust_remote_code=True).to_dict()
        config_new = {}
        config_new["num_layers"] = hf_config["num_hidden_layers"]
        config_new["num_embed"] = hf_config["vocab_size"]
        config_new["embed_dim"] = hf_config["hidden_size"]
        config_new["hidden_dim"] = hf_config["intermediate_size"]
        config_new["num_heads"] = hf_config["num_attention_heads"]
        config_new["head_dim"] = hf_config["head_dim"]
        config_new["num_kv_heads"] = hf_config["num_key_value_heads"]
        return EasyDict(config_new)

    def rename_to_trt_llm(self, name: str):
        """Rename a gemma parameter name by the corresponding TRT-LLM style name."""
        prefix = "transformer"
        sub_patterns = (
            (r"model.embed_tokens.weight", r"vocab_embedding.weight"),
            (r"model.layers.(\d+).input_layernorm.weight",
             r"layers.\1.input_layernorm.weight"),
            (r"model.layers.(\d+).self_attn.q_proj.weight",
             r"layers.\1.attention.qkv.weight"),
            (r"model.layers.(\d+).self_attn.k_proj.weight",
             None),  # merged with above
            (r"model.layers.(\d+).self_attn.v_proj.weight",
             None),  # merged with above
            (r"model.layers.(\d+).self_attn.o_proj.weight",
             r"layers.\1.attention.dense.weight"),
            (r"model.layers.(\d+).mlp.gate_proj.weight",
             r"layers.\1.mlp.fc.weight"),
            (r"model.layers.(\d+).mlp.up_proj.weight",
             None),  # merged with above
            (r"model.layers.(\d+).mlp.down_proj.weight",
             r"layers.\1.mlp.proj.weight"),
            (r"model.layers.(\d+).post_attention_layernorm.weight",
             r"layers.\1.post_layernorm.weight"),
            (r"model.norm.weight", r"ln_f.weight"),
        )

        for source, target in sub_patterns:
            if re.match(source, name):
                if target is None:
                    return target
                else:
                    name = re.sub(source, target, name)
                    return ".".join((prefix, name))
        else:
            raise ValueError(f"Don't know how to rename {prefix}.{name}")

    def flatten_params(self, params):
        f_params = {}
        for k, v in params.items():
            if v.dtype == torch.bfloat16:
                v = v.float()
            f_params[k] = torch_to_numpy(v)
        return f_params


CKPT_PARSER = {
    'jax': JAXParser,
    'keras': KerasParser,
    'torch': TorchParser,
    'hf': HfParser
}


def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    return np.split(v, tp_size, axis=dim)[idx]


def split_matrix_tp(v, tensor_parallel, rank, dim):
    return split(v, tensor_parallel, rank, dim=dim)


def add_trt_llm_weight(weights: typing.Dict[str, np.ndarray],
                       name: str,
                       param: np.ndarray,
                       dtype: typing.Optional[np.dtype] = None):
    assert name not in weights, f"{name} is already added."
    param = numpy_to_torch(param)
    if dtype is not None:
        assert isinstance(dtype,
                          str), f"dtype must be str, but get type {type(dtype)}"
        param = param.to(str_dtype_to_torch(dtype))
    weights[name] = param.contiguous()


def quantize(param: np.ndarray,
             quant_mode: tensorrt_llm.quantization.QuantMode):
    if quant_mode.is_int8_weight_only():
        quant_dtype = torch.int8
    elif quant_mode.is_int4_weight_only():
        quant_dtype = torch.quint4x2
    else:
        raise ValueError(f"Invalid configuration got quant_mode={quant_mode}")

    if param.dtype == np.dtype("bfloat16"):
        param = torch.from_numpy(param.astype(np.float32)).to(torch.bfloat16)
    else:
        param = torch.from_numpy(param)
    param = param.t().contiguous()

    # previously this fn was available in torch.ops.fastertransformer namespace
    (
        quantized_weights,
        scales,
    ) = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
        param, quant_dtype)

    if scales.dtype == torch.bfloat16:
        scales = scales.to(torch.float32).numpy().astype("bfloat16")
    else:
        scales = scales.numpy()
    return quantized_weights.numpy(), scales


def convert_from_checkpoint(
    trt_llm_config: tensorrt_llm.models.modeling_utils.PretrainedConfig,
    model_dir: typing.Union[str, pathlib.Path],
    ckpt_parser,
    rank=0,
):
    print("Loading weights...")
    tik = time.time()

    tp_rank = rank
    tp_size = trt_llm_config.mapping.tp_size
    hidden_size = trt_llm_config.hidden_size
    head_dim = trt_llm_config.head_size

    weights = {}
    for model_file in [model_dir]:
        LOGGER.debug(f"Loading directory {str(model_file)}...")
        model_params = ckpt_parser.load_parameters(model_file)
        model_params = ckpt_parser.flatten_params(model_params)

        for name, param in model_params.items():
            LOGGER.debug(f"Converting weight {name}...")
            trt_llm_name = ckpt_parser.rename_to_trt_llm(name)
            if trt_llm_name is None:  # omit as used with other params
                continue

            if "attn.q_einsum" in name:
                gqa_mode = trt_llm_config.num_attention_heads != trt_llm_config.num_key_value_heads
                assert gqa_mode

                # initial shape: (num_q_heads, hidden_size, head_dim)
                q_param = param.transpose(1, 0, 2)
                q_param = split_matrix_tp(q_param, tp_size, tp_rank, dim=1)

                # initial shape: (2, num_kv_heads, hidden_size, head_dim)
                kv_name = name.replace("q_einsum", "kv_einsum")
                kv_param = model_params[kv_name]
                kv_param = kv_param.reshape(
                    trt_llm_config.num_key_value_heads * 2,
                    hidden_size,
                    head_dim,
                ).transpose(1, 0, 2)

                # -> (hidden_size, num_q_heads / tp_size + 2, head_dim)
                qkv_param = np.concatenate([q_param, kv_param], axis=1)
                qkv_param = qkv_param.reshape(qkv_param.shape[0], -1)
                qkv_param = qkv_param.transpose(1, 0)

                # If int8 kv enabled, weight-only quantization will be done later.
                if trt_llm_config.quant_mode.is_weight_only() and not trt_llm_config.quant_mode.has_per_group_scaling() and \
                    not trt_llm_config.quant_mode.has_int8_kv_cache():
                    qkv_param_quantized, qkv_param_scales = quantize(
                        qkv_param, trt_llm_config.quant_mode)
                    add_trt_llm_weight(weights, trt_llm_name,
                                       qkv_param_quantized)
                    add_trt_llm_weight(
                        weights,
                        trt_llm_name.replace(".weight", ".per_channel_scale"),
                        qkv_param_scales,
                        trt_llm_config.dtype,
                    )
                else:
                    add_trt_llm_weight(weights, trt_llm_name, qkv_param,
                                       trt_llm_config.dtype)
            elif "self_attn.qkv_proj" in name:
                q_param, k_param, v_param = np.split(param, [
                    trt_llm_config.num_attention_heads *
                    trt_llm_config.head_size,
                    trt_llm_config.num_attention_heads *
                    trt_llm_config.head_size +
                    trt_llm_config.num_key_value_heads *
                    trt_llm_config.head_size
                ],
                                                     axis=0)
                gqa_mode = trt_llm_config.num_attention_heads != trt_llm_config.num_key_value_heads

                q_param = split_matrix_tp(q_param, tp_size, tp_rank, dim=0)
                if not gqa_mode:
                    k_param = split_matrix_tp(k_param, tp_size, tp_rank, dim=0)
                    v_param = split_matrix_tp(v_param, tp_size, tp_rank, dim=0)

                qkv_param = np.concatenate([q_param, k_param, v_param], axis=0)
                if trt_llm_config.quant_mode.is_weight_only(
                ) and not trt_llm_config.quant_mode.has_per_group_scaling():
                    qkv_param_quantized, qkv_param_scales = quantize(
                        qkv_param, trt_llm_config.quant_mode)
                    add_trt_llm_weight(weights, trt_llm_name,
                                       qkv_param_quantized)
                    add_trt_llm_weight(
                        weights,
                        trt_llm_name.replace(".weight", ".per_channel_scale"),
                        qkv_param_scales,
                        trt_llm_config.dtype,
                    )
                else:
                    add_trt_llm_weight(weights, trt_llm_name, qkv_param,
                                       trt_llm_config.dtype)
            elif "attn.qkv_einsum" in name:
                gqa_mode = trt_llm_config.num_attention_heads != trt_llm_config.num_key_value_heads
                assert not gqa_mode
                # initial shape: [3, num_heads, hidden_size, head_dim] -> [3, num_heads, head_dim, hidden_size]
                qkv_param = param.transpose(0, 1, 3, 2)
                qkv_param = qkv_param.reshape(qkv_param.shape[0], -1,
                                              qkv_param.shape[3])
                qkv_param = split_matrix_tp(qkv_param, tp_size, tp_rank, dim=1)
                qkv_param = qkv_param.reshape(-1, qkv_param.shape[2])
                if trt_llm_config.quant_mode.is_weight_only() and not trt_llm_config.quant_mode.has_per_group_scaling() \
                    and not trt_llm_config.quant_mode.has_int8_kv_cache():
                    qkv_param_quantized, qkv_param_scales = quantize(
                        qkv_param, trt_llm_config.quant_mode)
                    add_trt_llm_weight(weights, trt_llm_name,
                                       qkv_param_quantized)
                    add_trt_llm_weight(
                        weights,
                        trt_llm_name.replace(".weight", ".per_channel_scale"),
                        qkv_param_scales,
                        trt_llm_config.dtype,
                    )
                else:
                    add_trt_llm_weight(weights, trt_llm_name, qkv_param,
                                       trt_llm_config.dtype)
            elif "attention/query_dense" in name:
                # Keras specific KQV convert
                gqa_mode = trt_llm_config.num_attention_heads != trt_llm_config.num_key_value_heads
                if gqa_mode:

                    # initial shape: (num_q_heads, hidden_size, head_dim)
                    q_param = param.transpose(1, 0, 2)
                    q_param = split_matrix_tp(q_param, tp_size, tp_rank, dim=1)

                    # initial shape: (2, num_kv_heads, hidden_size, head_dim)
                    k_name = name.replace("query", "key")
                    k_param = model_params[k_name]
                    v_name = name.replace("query", "value")
                    v_param = model_params[v_name]
                    kv_param = np.stack((k_param, v_param), axis=0)

                    kv_param = kv_param.reshape(
                        trt_llm_config.num_key_value_heads * 2,
                        hidden_size,
                        head_dim,
                    ).transpose(1, 0, 2)

                    # -> (hidden_size, num_q_heads / tp_size + 2, head_dim)
                    qkv_param = np.concatenate([q_param, kv_param], axis=1)
                    qkv_param = qkv_param.reshape(qkv_param.shape[0], -1)
                    qkv_param = qkv_param.transpose(1, 0)

                    if trt_llm_config.quant_mode.is_weight_only(
                    ) and not trt_llm_config.quant_mode.has_int8_kv_cache():
                        qkv_param_quantized, qkv_param_scales = quantize(
                            qkv_param, trt_llm_config.quant_mode)
                        add_trt_llm_weight(weights, trt_llm_name,
                                           qkv_param_quantized)
                        add_trt_llm_weight(
                            weights,
                            trt_llm_name.replace(".weight",
                                                 ".per_channel_scale"),
                            qkv_param_scales,
                            trt_llm_config.dtype,
                        )
                    else:
                        add_trt_llm_weight(weights, trt_llm_name, qkv_param,
                                           trt_llm_config.dtype)
                else:
                    q_param = param
                    k_name = name.replace("query", "key")
                    k_param = model_params[k_name]
                    v_name = name.replace("query", "value")
                    v_param = model_params[v_name]
                    # initial shape: [3, num_heads, hidden_size, head_dim] -> [3, num_heads, head_dim, hidden_size]
                    qkv_param = np.stack((q_param, k_param, v_param), axis=0)
                    qkv_param = qkv_param.transpose(0, 1, 3, 2)
                    qkv_param = qkv_param.reshape(qkv_param.shape[0], -1,
                                                  qkv_param.shape[3])
                    qkv_param = split_matrix_tp(qkv_param,
                                                tp_size,
                                                tp_rank,
                                                dim=1)
                    qkv_param = qkv_param.reshape(-1, qkv_param.shape[2])
                    if trt_llm_config.quant_mode.is_weight_only(
                    ) and not trt_llm_config.quant_mode.has_int8_kv_cache():
                        qkv_param_quantized, qkv_param_scales = quantize(
                            qkv_param, trt_llm_config.quant_mode)
                        add_trt_llm_weight(weights, trt_llm_name,
                                           qkv_param_quantized)
                        add_trt_llm_weight(
                            weights,
                            trt_llm_name.replace(".weight",
                                                 ".per_channel_scale"),
                            qkv_param_scales,
                            trt_llm_config.dtype,
                        )
                    else:
                        add_trt_llm_weight(weights, trt_llm_name, qkv_param,
                                           trt_llm_config.dtype)
            elif "q_proj" in name:
                gqa_mode = trt_llm_config.num_attention_heads != trt_llm_config.num_key_value_heads

                if gqa_mode:
                    # initial shape: (num_heads * head_dim, hidden_size)
                    q_param = param
                    q_param = split_matrix_tp(q_param, tp_size, tp_rank, dim=0)

                    k_name = name.replace("q_proj", "k_proj")
                    k_param = model_params[k_name]

                    v_name = name.replace("q_proj", "v_proj")
                    v_param = model_params[v_name]
                else:
                    # initial shape: (num_heads * head_dim, hidden_size)
                    q_param = param
                    q_param = split_matrix_tp(q_param, tp_size, tp_rank, dim=0)

                    k_name = name.replace("q_proj", "k_proj")
                    k_param = model_params[k_name]
                    k_param = split_matrix_tp(k_param, tp_size, tp_rank, dim=0)

                    v_name = name.replace("q_proj", "v_proj")
                    v_param = model_params[v_name]
                    v_param = split_matrix_tp(v_param, tp_size, tp_rank, dim=0)

                qkv_param = np.concatenate([q_param, k_param, v_param], axis=0)
                qkv_param = qkv_param.reshape(qkv_param.shape[0], -1)

                # If int8 kv enabled, weight-only quantization will be done later.
                if trt_llm_config.quant_mode.is_weight_only() and not trt_llm_config.quant_mode.has_per_group_scaling() and \
                    not trt_llm_config.quant_mode.has_int8_kv_cache():
                    qkv_param_quantized, qkv_param_scales = quantize(
                        qkv_param, trt_llm_config.quant_mode)
                    add_trt_llm_weight(weights, trt_llm_name,
                                       qkv_param_quantized)
                    add_trt_llm_weight(
                        weights,
                        trt_llm_name.replace(".weight", ".per_channel_scale"),
                        qkv_param_scales,
                        trt_llm_config.dtype,
                    )
                else:
                    add_trt_llm_weight(weights, trt_llm_name, qkv_param,
                                       trt_llm_config.dtype)
            elif "attention.dense.weight" in trt_llm_name:
                # initial shape: (num_heads, head_dim, hidden_size)
                if len(param.shape) == 3:
                    param = param.reshape(-1, param.shape[2])
                    param = param.transpose(
                        1, 0)  # (hidden_size, num_heads * head_dum)
                param = split_matrix_tp(param, tp_size, tp_rank, dim=1)
                if trt_llm_config.quant_mode.is_weight_only(
                ) and not trt_llm_config.quant_mode.has_int8_kv_cache():
                    param_quantized, param_scales = quantize(
                        param, trt_llm_config.quant_mode)
                    add_trt_llm_weight(weights, trt_llm_name, param_quantized)
                    add_trt_llm_weight(
                        weights,
                        trt_llm_name.replace(".weight", ".per_channel_scale"),
                        param_scales,
                        trt_llm_config.dtype,
                    )
                else:
                    add_trt_llm_weight(weights, trt_llm_name, param,
                                       trt_llm_config.dtype)
            elif "mlp.fc.weight" in trt_llm_name:
                if isinstance(ckpt_parser, KerasParser):
                    # initial shape: (hidden_size, intermediate_size)
                    fc_param, gate_param = param, model_params[name.replace(
                        "gating_ffw", "gating_ffw_2")]
                elif isinstance(ckpt_parser, TorchParser):
                    # initial shape: (intermediate_size, hidden_size)
                    fc_param, gate_param = param, model_params[name.replace(
                        "mlp.gate_proj", "mlp.up_proj")]
                    fc_param = fc_param.transpose(1, 0)
                    gate_param = gate_param.transpose(1, 0)
                elif isinstance(ckpt_parser, HfParser):
                    # initial shape: (intermediate_size, hidden_size)
                    fc_param, gate_param = param, model_params[name.replace(
                        "mlp.gate_proj", "mlp.up_proj")]
                    fc_param = fc_param.transpose(1, 0)
                    gate_param = gate_param.transpose(1, 0)
                else:
                    # initial shape: (2, hidden_size, intermediate_size)
                    fc_param, gate_param = param[0], param[1]
                fc_param = fc_param.transpose(1, 0)
                fc_param = split_matrix_tp(fc_param, tp_size, tp_rank, dim=0)
                if trt_llm_config.quant_mode.is_weight_only() and not trt_llm_config.quant_mode.has_per_group_scaling() and \
                    not trt_llm_config.quant_mode.has_int8_kv_cache():
                    fc_param_quantized, fc_param_scales = quantize(
                        fc_param, trt_llm_config.quant_mode)
                    add_trt_llm_weight(weights, trt_llm_name,
                                       fc_param_quantized)
                    add_trt_llm_weight(
                        weights,
                        trt_llm_name.replace(".weight", ".per_channel_scale"),
                        fc_param_scales,
                        trt_llm_config.dtype,
                    )
                else:
                    add_trt_llm_weight(weights, trt_llm_name, fc_param,
                                       trt_llm_config.dtype)

                gate_param = gate_param.transpose(1, 0)
                gate_param = split_matrix_tp(gate_param,
                                             tp_size,
                                             tp_rank,
                                             dim=0)
                trt_llm_name = trt_llm_name.replace("mlp.fc.weight",
                                                    "mlp.gate.weight")
                if trt_llm_config.quant_mode.is_weight_only() and not trt_llm_config.quant_mode.has_per_group_scaling() and \
                    not trt_llm_config.quant_mode.has_int8_kv_cache():
                    gate_param_quantized, gate_param_scales = quantize(
                        gate_param, trt_llm_config.quant_mode)
                    add_trt_llm_weight(weights, trt_llm_name,
                                       gate_param_quantized)
                    add_trt_llm_weight(
                        weights,
                        trt_llm_name.replace(".weight", ".per_channel_scale"),
                        gate_param_scales,
                        trt_llm_config.dtype,
                    )
                else:
                    add_trt_llm_weight(weights, trt_llm_name, gate_param,
                                       trt_llm_config.dtype)
            elif "mlp.proj.weight" in trt_llm_name:
                if not isinstance(ckpt_parser, TorchParser) and not isinstance(
                        ckpt_parser, HfParser):
                    # initial shape: (intermediate_size, hidden_size)
                    param = param.transpose(1, 0)
                param = split_matrix_tp(param, tp_size, tp_rank, dim=1)
                if trt_llm_config.quant_mode.is_weight_only() and not trt_llm_config.quant_mode.has_per_group_scaling() and \
                    not trt_llm_config.quant_mode.has_int8_kv_cache():
                    param_quantized, param_scales = quantize(
                        param, trt_llm_config.quant_mode)
                    add_trt_llm_weight(weights, trt_llm_name, param_quantized)
                    add_trt_llm_weight(
                        weights,
                        trt_llm_name.replace(".weight", ".per_channel_scale"),
                        param_scales,
                        trt_llm_config.dtype,
                    )
                else:
                    add_trt_llm_weight(weights, trt_llm_name, param,
                                       trt_llm_config.dtype)
            elif "embedder.input_embedding" in name or "reversible_embedding" in name or "embedder.weight" in name \
                    or "embed_tokens.weight" in name:
                if not trt_llm_config.share_embedding_table:
                    # TODO: safetensor doesn't allow to save a shared tensor.
                    # Currently, we clone the weight but to save the disk, it
                    # would be better to skip saving lm_head weights and
                    # handle it at the loading phase.
                    lm_head = split_matrix_tp(param, tp_size, tp_rank, dim=0)
                    add_trt_llm_weight(weights, "lm_head.weight",
                                       np.copy(lm_head), trt_llm_config.dtype)

                if trt_llm_config.use_parallel_embedding:
                    assert trt_llm_config.vocab_size % tp_size == 0
                    param = split_matrix_tp(
                        param,
                        tp_size,
                        tp_rank,
                        dim=trt_llm_config.embedding_sharding_dim,
                    )
                if trt_llm_config.quant_mode.is_int8_weight_only() and not trt_llm_config.quant_mode.has_per_group_scaling() and \
                    not trt_llm_config.quant_mode.has_int8_kv_cache() and trt_llm_config.quantization.exclude_modules is not None:

                    # shape of embedding table: [V, K], V: vocab size, K: embedding dim

                    # quantize will do following work:
                    # 1. transpose the input from [V, K] to [K, V]
                    # 2. compute V scales across dimension K
                    # 3. quantize the input to int8_weight
                    # 4. transpose the int8_weight to [V, K], but there is a bug in 'quantize' that the claimed shape is [K, V]
                    param_quantized, param_scales = quantize(
                        param, trt_llm_config.quant_mode)

                    # Reshape the [K, V] to [V, K] to match the real data layout.
                    param_quantized = np.ascontiguousarray(
                        param_quantized.reshape([
                            param_quantized.shape[1], param_quantized.shape[0]
                        ]))

                    add_trt_llm_weight(weights, trt_llm_name, param_quantized)
                    add_trt_llm_weight(
                        weights,
                        trt_llm_name.replace(".weight", ".per_token_scale"),
                        param_scales,
                        trt_llm_config.dtype,
                    )
                else:
                    add_trt_llm_weight(weights, trt_llm_name, param,
                                       trt_llm_config.dtype)
            elif any(keyword in name for keyword in (
                    "pre_attention_norm.scale",
                    "pre_ffw_norm.scale",
                    "final_norm.scale",
                    "pre_attention_norm/vars/0",
                    "pre_ffw_norm/vars/0",
                    "rms_normalization/vars/0",
                    "input_layernorm",
                    "post_attention_layernorm",
                    "model.norm.weight",
            )):
                param = param + 1.0  # upcasted to float32 in case of bfloat16
                add_trt_llm_weight(weights, trt_llm_name, param,
                                   trt_llm_config.dtype)
            else:
                raise RuntimeError(f"Unhandled {name} module weights")
        del model_params

    print(
        f"Weights loaded. Total time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - tik))}"
    )
    return weights


def convert(worker_rank, args, convert_kwargs):
    for rank in range(worker_rank, args.world_size):
        weights = convert_from_checkpoint(rank=rank, **convert_kwargs)
        trt_llm_config = convert_kwargs.get("trt_llm_config")
        if args.use_smooth_quant_plugin is not None or args.calibrate_kv_cache:
            qkv_para = {}
            smoother = {}
            dataset = load_calib_dataset(args.calib_dataset)
            assert args.tokenizer_dir is not None, "Must set tokenizer_dir to do calibration"
            tokenizer = sp.SentencePieceProcessor(model_file=args.tokenizer_dir)
            if "transformer.vocab_embedding.weight" in weights:
                # To use the HF to do SmoothQuant, we need to scale the embedding.
                weights["transformer.vocab_embedding.weight"] = torch.multiply(
                    weights["transformer.vocab_embedding.weight"].to(
                        torch.float32),
                    math.sqrt(trt_llm_config.hidden_size),
                )
            hf_model = create_model_from_config(trt_llm_config, weights)
            act_range = capture_activation_range(hf_model, tokenizer, dataset)
            if args.use_smooth_quant_plugin is not None:
                smooth_model(hf_model, act_range, args.use_smooth_quant_plugin,
                             qkv_para, smoother)
            weights = convert_hf_model(
                hf_model, trt_llm_config.mapping, trt_llm_config.vocab_size,
                args.dtype, False, 0,
                args.use_weight_only_with_precision != None,
                torch.int8 if args.use_weight_only_with_precision == 'int8' else
                torch.quint4x2, args.use_smooth_quant_plugin is not None,
                args.per_channel, args.per_token, args.calibrate_kv_cache,
                act_range, qkv_para, smoother)
            if "transformer.vocab_embedding.weight" in weights:
                # Revert the scaling of embedding
                weights["transformer.vocab_embedding.weight"] = torch.divide(
                    weights["transformer.vocab_embedding.weight"].to(
                        torch.float32),
                    math.sqrt(trt_llm_config.hidden_size),
                ).to(str_dtype_to_torch(args.dtype))
            if trt_llm_config.share_embedding_table and "lm_head.weight" in weights:
                # When share_embedding_table is enabled, we add lm_head into weights
                # to do quantization in HF. Remove lm_head before saving it in unified
                # checkpoint.
                del weights["lm_head.weight"]
            safetensors.torch.save_file(
                weights, args.output_model_dir / f"rank{rank}.safetensors")
            return

        use_awq = False
        if args.use_weight_only_with_precision:
            if args.use_weight_only_with_precision.endswith("awq"):
                use_awq = True
        if use_awq:
            weights = dummy_weights_awq(
                weights=weights,
                precision=args.use_weight_only_with_precision,
                trt_llm_config=trt_llm_config,
                group_size=128)
        elif args.enable_fp8 or args.fp8_kv_cache:
            weight_scales = quantize_fp8_weights(
                weights, trt_llm_config.num_hidden_layers,
                trt_llm_config.mapping)
            scales = load_from_fp8_gemma(args.quant_ckpt_path,
                                         trt_llm_config.num_hidden_layers,
                                         trt_llm_config.mapping,
                                         args.fp8_kv_cache, weight_scales)
            weights.update(scales)

        safetensors.torch.save_file(
            weights, args.output_model_dir / f"rank{rank}.safetensors")


def main():
    args = parse_arguments()

    tik = time.time()

    print(f"Loading source parameters from {args.model_dir.absolute()}")
    ckpt_parser = CKPT_PARSER[args.ckpt_type]()
    ckpt_params = ckpt_parser.load_parameters(args.model_dir)
    input_embedding_weights = ckpt_parser.embedding_weights(ckpt_params)
    num_embed, _ = input_embedding_weights.shape
    ckpt_params_dtype = str(
        input_embedding_weights.dtype).split(".")[-1]  # np.bfloat16 -> bfloat16
    ckpt_config = ckpt_parser.get_config(args.model_dir, ckpt_params, num_embed)
    # 2B TransformerConfig(num_layers=18, num_embed=256128, embed_dim=2048, hidden_dim=16384, num_heads=8, head_dim=256, num_kv_heads=1)
    # 7B TransformerConfig(...)

    print(f"Source configuration determined from parameters: {ckpt_config}")

    quant_kwargs = {}
    quant_algo = None
    kv_cache_quant_algo = None
    if args.use_weight_only_with_precision:
        quant_algo = {
            "int8": QuantAlgo.W8A16,
            "int4": QuantAlgo.W4A16,
            "w4a8_awq": QuantAlgo.W4A8_AWQ,
            "w4a16_awq": QuantAlgo.W4A16_AWQ,
        }[args.use_weight_only_with_precision]
    elif args.enable_fp8:
        quant_algo = QuantAlgo.FP8
    elif args.use_smooth_quant:
        quant_algo = QuantAlgo.W8A8_SQ_PER_CHANNEL

    if args.fp8_kv_cache:
        kv_cache_quant_algo = QuantAlgo.FP8
    if args.calibrate_kv_cache:
        kv_cache_quant_algo = QuantAlgo.INT8
    if args.use_smooth_quant:
        quant_algo = QuantAlgo.W8A8_SQ_PER_CHANNEL
    elif args.use_smooth_quant_plugin is not None:
        if args.per_token and args.per_channel:
            quant_algo = QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN
        elif not args.per_token and not args.per_channel:
            quant_algo = QuantAlgo.W8A8_SQ_PER_TENSOR_PLUGIN
        elif not args.per_token and args.per_channel:
            quant_algo = QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN
        elif args.per_token and not args.per_channel:
            quant_algo = QuantAlgo.W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN

    quant_kwargs.update(quant_algo=quant_algo,
                        kv_cache_quant_algo=kv_cache_quant_algo)
    if args.use_weight_only_with_precision:
        if args.use_weight_only_with_precision.endswith(
                "awq") or args.use_weight_only_with_precision.endswith(
                    "int4") or not args.use_int8_weight_only_embedding:
            quant_kwargs.update(has_zero_point=False, pre_quant_scale=True)
        else:
            quant_kwargs.update(exclude_modules=['router'])

    quant_config = tensorrt_llm.models.modeling_utils.QuantConfig()
    quant_config.quant_algo = quant_kwargs['quant_algo']
    quant_config.kv_cache_quant_algo = quant_kwargs['kv_cache_quant_algo']
    if args.use_weight_only_with_precision:
        quant_config.exclude_modules = quant_kwargs.get('exclude_modules')
        if args.use_weight_only_with_precision.endswith("awq"):
            quant_config.group_size = 128
            quant_config.has_zero_point = quant_kwargs['has_zero_point']
            quant_config.pre_quant_scale = quant_kwargs['pre_quant_scale']

    trt_llm_config = tensorrt_llm.models.modeling_utils.PretrainedConfig(
        architecture="GemmaForCausalLM",
        dtype=args.dtype or ckpt_params_dtype,
        logits_dtype="float32",
        vocab_size=ckpt_config.num_embed,
        max_position_embeddings=8192,
        hidden_size=ckpt_config.embed_dim,
        num_hidden_layers=ckpt_config.num_layers,
        num_attention_heads=ckpt_config.num_heads,
        num_key_value_heads=ckpt_config.num_kv_heads,
        head_size=ckpt_config.head_dim,
        hidden_act="gelu",
        intermediate_size=ckpt_config.hidden_dim,
        norm_epsilon=1e-6,  # hard-coded in RMSNorm from gemma/layers.py
        position_embedding_type="rope_gpt_neox",
        world_size=args.world_size,
        tp_size=args.world_size,
        pp_size=1,
        gpus_per_node=8,
        quantization=quant_config,
        share_embedding_table=args.use_embedding_sharing,
    )

    trt_llm_config_dict = trt_llm_config.to_dict()
    print(f"Determined TensorRT-LLM configuration {trt_llm_config_dict}")

    config_path = args.output_model_dir / "config.json"
    config_path.parent.mkdir(exist_ok=True, parents=True)
    LOGGER.debug(f"Saving TensorRT-LLM configuration to {config_path}")
    with config_path.open("w") as config_file:
        json.dump(trt_llm_config_dict, config_file, indent=4)

    convert_args = dict(trt_llm_config=trt_llm_config,
                        model_dir=args.model_dir,
                        ckpt_parser=ckpt_parser)
    convert(0, args, convert_args)

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - tik))
    print(f"Total time of converting checkpoints: {elapsed}")


if __name__ == "__main__":
    main()

import argparse
import json
import logging
import math
import re
import time
import typing
from pathlib import Path

# isort: off
import flax
import numpy as np
import orbax
import safetensors.torch
import torch
from recurrentgemma import jax as recurrentgemma_jax
from transformers import AutoConfig, AutoModelForCausalLM
#isort: on

import tensorrt_llm
from tensorrt_llm import logger
from tensorrt_llm._utils import (numpy_to_torch, str_dtype_to_torch,
                                 torch_to_numpy)

LOGGER = logging.getLogger("convert_checkpoint")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_type", type=str, choices=["jax", "hf"])
    parser.add_argument("--model_dir", type=Path, default=None)
    parser.add_argument("--world_size",
                        type=int,
                        default=1,
                        help="world size, only support tensor parallelism now")
    parser.add_argument("--dtype",
                        type=str,
                        default="float16",
                        choices=["float32", "bfloat16", "float16"])
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="recurrentgemma_tllm_checkpoint",
        help="The path to save the recurrentgemma TensorRT LLM checkpoint")
    parser.add_argument("--log_level", type=str, default="info")
    args = parser.parse_args()
    return args


class JAXParser:

    def load_parameters(self, checkpoint_path: Path):
        checkpoint_path = checkpoint_path.absolute()
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        params = checkpointer.restore(checkpoint_path)
        return params

    def embedding_weights(self, ckpt_params):
        return ckpt_params["embedder"]["input_embedding"]

    def get_config(self, checkpoint_path, ckpt_params):
        config = recurrentgemma_jax.GriffinConfig.from_flax_params_or_variables(
            ckpt_params,
            preset=recurrentgemma_jax.Preset.RECURRENT_GEMMA_2B_V1,
        )._asdict()
        if config["lru_width"] is None:
            config["lru_width"] = config["width"]
        layer_types = []
        for p in config["block_types"]:
            if p == recurrentgemma_jax.TemporalBlockType.ATTENTION:
                layer_types.append("attention")
            else:
                layer_types.append("recurrent")
        config["block_types"] = layer_types
        config["hidden_size"] = config.pop("width")
        config["num_attention_heads"] = config.pop("num_heads")
        config["intermediate_size"] = config.pop("mlp_expanded_width")
        config["num_hidden_layers"] = len(config["block_types"])
        return config

    def rename_to_trt_llm(self, name: str):
        """Rename a recurrentgemma parameter name by the corresponding TRT-LLM style name."""
        sub_patterns = (
            (r"embedder.input_embedding", r"vocab_embedding.weight"),
            (r"blocks.(\d+).channel_pre_norm.scale",
             r"layers.\1.post_layernorm.weight"),
            (r"blocks.(\d+).temporal_pre_norm.scale",
             r"layers.\1.input_layernorm.weight"),
            (r"blocks.(\d+).recurrent_block.conv_1d.w",
             r"layers.\1.recurrent.conv1d.weight"),
            (r"blocks.(\d+).recurrent_block.conv_1d.b",
             r"layers.\1.recurrent.conv1d.bias"),
            (r"blocks.(\d+).recurrent_block.linear_out.kernel",
             r"layers.\1.recurrent.linear_out.weight"),
            (r"blocks.(\d+).recurrent_block.linear_out.bias",
             r"layers.\1.recurrent.linear_out.bias"),
            (r"blocks.(\d+).recurrent_block.linear_x.kernel",
             r"layers.\1.recurrent.linear_x.weight"),
            (r"blocks.(\d+).recurrent_block.linear_x.bias",
             r"layers.\1.recurrent.linear_x.bias"),
            (r"blocks.(\d+).recurrent_block.linear_y.kernel",
             r"layers.\1.recurrent.linear_y.weight"),
            (r"blocks.(\d+).recurrent_block.linear_y.bias",
             r"layers.\1.recurrent.y_bias"),
            (r"blocks.(\d+).recurrent_block.rg_lru.a_gate.w",
             r"layers.\1.recurrent.rg_lru.recurrent_gate.weight"),
            (r"blocks.(\d+).recurrent_block.rg_lru.a_gate.b",
             r"layers.\1.recurrent.rg_lru.recurrent_gate.bias"),
            (r"blocks.(\d+).recurrent_block.rg_lru.input_gate.w",
             r"layers.\1.recurrent.rg_lru.input_gate.weight"),
            (r"blocks.(\d+).recurrent_block.rg_lru.input_gate.b",
             r"layers.\1.recurrent.rg_lru.input_gate.bias"),
            (r"blocks.(\d+).recurrent_block.rg_lru.a_param",
             r"layers.\1.recurrent.rg_lru.recurrent_param"),
            (r"blocks.(\d+).mlp_block.ffw_up.w", r"layers.\1.mlp.fc.weight"),
            (r"blocks.(\d+).mlp_block.ffw_up.b", None),
            (r"blocks.(\d+).mlp_block.ffw_down.kernel",
             r"layers.\1.mlp.proj.weight"),
            (r"blocks.(\d+).mlp_block.ffw_down.bias",
             r"layers.\1.mlp.proj.bias"),
            (r"blocks.(\d+).attention_block.proj_q.kernel",
             r"layers.\1.attention.qkv.weight"),
            (r"blocks.(\d+).attention_block.proj_k.kernel", None),
            (r"blocks.(\d+).attention_block.proj_v.kernel", None),
            (r"blocks.(\d+).attention_block.proj_final.kernel",
             r"layers.\1.attention.dense.weight"),
            (r"blocks.(\d+).attention_block.proj_final.bias",
             r"layers.\1.attention.dense.bias"),
            (r"final_norm.scale", r"ln_f.weight"),
        )

        for source, target in sub_patterns:
            if re.match(source, name):
                if target is None:
                    return target
                else:
                    name = re.sub(source, target, name)
                    return ".".join(("transformer", name))
        else:
            raise ValueError(f"Don't know how to rename {name}")

    def flatten_params(self, params):
        new_params = flax.traverse_util.flatten_dict(params, sep=".")
        # if the dtype is bfloat16, cast to float32
        for k in new_params:
            if new_params[k].dtype != np.float32 and new_params[
                    k].dtype != np.float16:
                new_params[k] = new_params[k].astype(np.float32)
        return new_params


class HfParser:

    def load_parameters(self, checkpoint_path: Path):
        hf_model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            device_map="auto",
            dtype="auto",
        )
        model_params = dict(hf_model.named_parameters())
        return model_params

    def embedding_weights(self, ckpt_params):
        return ckpt_params["model.embed_tokens.weight"]

    def get_config(self, checkpoint_path, ckpt_params):
        checkpoint_path = checkpoint_path.absolute()
        hf_config = AutoConfig.from_pretrained(
            checkpoint_path, trust_remote_code=True).to_dict()
        hf_config["block_types"] = hf_config.pop("_block_types")
        hf_config["intermediate_size"] = hf_config["intermediate_size"] // 2
        return hf_config

    def rename_to_trt_llm(self, name: str):
        """Rename a recurrentgemma parameter name by the corresponding TRT-LLM style name."""
        sub_patterns = (
            (r"model.embed_tokens.weight", r"vocab_embedding.weight"),
            (r"model.layers.(\d+).temporal_pre_norm.weight",
             r"layers.\1.input_layernorm.weight"),
            (r"model.layers.(\d+).channel_pre_norm.weight",
             r"layers.\1.post_layernorm.weight"),
            (r"model.layers.(\d+).temporal_block.conv_1d.weight",
             r"layers.\1.recurrent.conv1d.weight"),
            (r"model.layers.(\d+).temporal_block.conv_1d.bias",
             r"layers.\1.recurrent.conv1d.bias"),
            (r"model.layers.(\d+).temporal_block.linear_out.weight",
             r"layers.\1.recurrent.linear_out.weight"),
            (r"model.layers.(\d+).temporal_block.linear_out.bias",
             r"layers.\1.recurrent.linear_out.bias"),
            (r"model.layers.(\d+).temporal_block.linear_x.weight",
             r"layers.\1.recurrent.linear_x.weight"),
            (r"model.layers.(\d+).temporal_block.linear_x.bias",
             r"layers.\1.recurrent.linear_x.bias"),
            (r"model.layers.(\d+).temporal_block.linear_y.weight",
             r"layers.\1.recurrent.linear_y.weight"),
            (r"model.layers.(\d+).temporal_block.linear_y.bias",
             r"layers.\1.recurrent.y_bias"),
            (r"model.layers.(\d+).temporal_block.rg_lru.recurrent_gate_weight",
             r"layers.\1.recurrent.rg_lru.recurrent_gate.weight"),
            (r"model.layers.(\d+).temporal_block.rg_lru.recurrent_gate_bias",
             r"layers.\1.recurrent.rg_lru.recurrent_gate.bias"),
            (r"model.layers.(\d+).temporal_block.rg_lru.input_gate_weight",
             r"layers.\1.recurrent.rg_lru.input_gate.weight"),
            (r"model.layers.(\d+).temporal_block.rg_lru.input_gate_bias",
             r"layers.\1.recurrent.rg_lru.input_gate.bias"),
            (r"model.layers.(\d+).temporal_block.rg_lru.recurrent_param",
             r"layers.\1.recurrent.rg_lru.recurrent_param"),
            (r"model.layers.(\d+).mlp_block.up_proj.weight",
             r"layers.\1.mlp.gate.weight"),
            (r"model.layers.(\d+).mlp_block.up_proj.bias",
             r"layers.\1.mlp.gate.bias"),
            (r"model.layers.(\d+).mlp_block.gate_proj.weight",
             r"layers.\1.mlp.fc.weight"),
            (r"model.layers.(\d+).mlp_block.gate_proj.bias",
             r"layers.\1.mlp.fc.bias"),
            (r"model.layers.(\d+).mlp_block.down_proj.weight",
             r"layers.\1.mlp.proj.weight"),
            (r"model.layers.(\d+).mlp_block.down_proj.bias",
             r"layers.\1.mlp.proj.bias"),
            (r"model.layers.(\d+).temporal_block.q_proj.weight",
             r"layers.\1.attention.qkv.weight"),
            (r"model.layers.(\d+).temporal_block.k_proj.weight", None),
            (r"model.layers.(\d+).temporal_block.v_proj.weight", None),
            (r"model.layers.(\d+).temporal_block.o_proj.weight",
             r"layers.\1.attention.dense.weight"),
            (r"model.layers.(\d+).temporal_block.o_proj.bias",
             r"layers.\1.attention.dense.bias"),
            (r"model.final_norm.weight", r"ln_f.weight"),
        )

        for source, target in sub_patterns:
            if re.match(source, name):
                if target is None:
                    return target
                else:
                    name = re.sub(source, target, name)
                    return ".".join(("transformer", name))
        else:
            raise ValueError(f"Don't know how to rename {name}")

    def flatten_params(self, params):
        f_params = {}
        for k, v in params.items():
            if v.dtype == torch.bfloat16:
                v = v.float()
            f_params[k] = torch_to_numpy(v)
        return f_params


CKPT_PARSER = {
    "jax": JAXParser,
    "hf": HfParser,
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


def convert_from_checkpoint(
    trt_llm_config: tensorrt_llm.models.modeling_utils.PretrainedConfig,
    model_dir: typing.Union[str, Path],
    ckpt_parser,
    rank=0,
):
    print("Loading weights...")
    tik = time.time()

    tp_rank = rank
    tp_size = trt_llm_config.mapping.tp_size
    intermediate_size = trt_llm_config.intermediate_size
    rnn_hidden_size = trt_llm_config.rnn_hidden_size
    conv_kernel = trt_llm_config.conv_kernel

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

            if "proj_q" in name or "q_proj" in name:
                if isinstance(ckpt_parser, JAXParser):
                    k_name = name.replace("proj_q", "proj_k")
                    v_name = name.replace("proj_q", "proj_v")
                    q_param = param.transpose(1, 0)
                    k_param = model_params[k_name].transpose(1, 0)
                    v_param = model_params[v_name].transpose(1, 0)
                else:
                    k_name = name.replace("q_proj", "k_proj")
                    v_name = name.replace("q_proj", "v_proj")
                    q_param = param
                    k_param = model_params[k_name]
                    v_param = model_params[v_name]
                q_param = split_matrix_tp(q_param, tp_size, tp_rank, dim=0)
                qkv_param = np.concatenate([q_param, k_param, v_param], axis=0)
                add_trt_llm_weight(weights, trt_llm_name, qkv_param,
                                   trt_llm_config.dtype)
            elif "ffw_up.w" in name and isinstance(ckpt_parser, JAXParser):
                bias_name = name.replace("ffw_up.w", "ffw_up.b")
                fc_param, gate_param = param[
                    0,
                ].transpose(1, 0), param[
                    1,
                ].transpose(1, 0)
                fc_param = split_matrix_tp(fc_param, tp_size, tp_rank, dim=0)
                gate_param = split_matrix_tp(gate_param,
                                             tp_size,
                                             tp_rank,
                                             dim=0)
                fc_bias = model_params[bias_name][
                    0,
                ].reshape(intermediate_size)
                gate_bias = model_params[bias_name][
                    1,
                ].reshape(intermediate_size)
                fc_bias = split_matrix_tp(fc_bias, tp_size, tp_rank, dim=0)
                gate_bias = split_matrix_tp(gate_bias, tp_size, tp_rank, dim=0)
                trt_llm_fc_name = trt_llm_name
                trt_llm_gate_name = trt_llm_name.replace(
                    "fc.weight", "gate.weight")
                trt_llm_fc_b_name = trt_llm_name.replace("fc.weight", "fc.bias")
                trt_llm_gate_b_name = trt_llm_name.replace(
                    "fc.weight", "gate.bias")
                add_trt_llm_weight(weights, trt_llm_fc_name, fc_param,
                                   trt_llm_config.dtype)
                add_trt_llm_weight(weights, trt_llm_gate_name, gate_param,
                                   trt_llm_config.dtype)
                add_trt_llm_weight(weights, trt_llm_fc_b_name, fc_bias,
                                   trt_llm_config.dtype)
                add_trt_llm_weight(weights, trt_llm_gate_b_name, gate_bias,
                                   trt_llm_config.dtype)
            elif "conv_1d.w" in name:
                if isinstance(ckpt_parser, JAXParser):
                    param = param.transpose(1,
                                            0).reshape(rnn_hidden_size, 1,
                                                       conv_kernel, 1)
                else:
                    param = param.reshape(rnn_hidden_size, 1, conv_kernel, 1)
                param = split_matrix_tp(param, tp_size, tp_rank, dim=0)
                add_trt_llm_weight(weights, trt_llm_name, param,
                                   trt_llm_config.dtype)
            elif "embedder.input_embedding" in name or "model.embed_tokens" in name:
                lm_head = split_matrix_tp(param, tp_size, tp_rank, dim=0)
                add_trt_llm_weight(weights, "lm_head.weight", np.copy(lm_head),
                                   trt_llm_config.dtype)
                if trt_llm_config.emb_scale_by_sqrt_dim:
                    param = np.multiply(
                        param.astype(np.float32),
                        math.sqrt(trt_llm_config.hidden_size),
                    )
                if trt_llm_config.use_parallel_embedding:
                    assert trt_llm_config.vocab_size % tp_size == 0
                    param = split_matrix_tp(
                        param,
                        tp_size,
                        tp_rank,
                        dim=trt_llm_config.embedding_sharding_dim,
                    )
                add_trt_llm_weight(weights, trt_llm_name, param,
                                   trt_llm_config.dtype)
            elif any(keyword in name for keyword in (
                    "proj_final.kernel",
                    "ffw_down.kernel",
                    "linear_out.kernel",
                    "o_proj.weight",
                    "down_proj.weight",
                    "linear_out.weight",
            )):
                if isinstance(ckpt_parser, JAXParser):
                    param = param.transpose(1, 0)
                param = split_matrix_tp(param, tp_size, tp_rank, dim=1)
                add_trt_llm_weight(weights, trt_llm_name, param,
                                   trt_llm_config.dtype)
            elif any(keyword in name for keyword in (
                    "linear_x.kernel",
                    "linear_y.kernel",
                    "linear_x.weight",
                    "linear_y.weight",
                    "up_proj.weight",
                    "gate_proj.weight",
            )):
                if isinstance(ckpt_parser, JAXParser):
                    param = param.transpose(1, 0)
                param = split_matrix_tp(param, tp_size, tp_rank, dim=0)
                add_trt_llm_weight(weights, trt_llm_name, param,
                                   trt_llm_config.dtype)
            elif any(keyword in name for keyword in (
                    "linear_x.bias",
                    "linear_y.bias",
                    "rg_lru",
                    "conv_1d.b",
                    "gate_proj.bias",
                    "up_proj.bias",
            )):
                param = split_matrix_tp(param, tp_size, tp_rank, dim=0)
                add_trt_llm_weight(weights, trt_llm_name, param,
                                   trt_llm_config.dtype)
            elif any(keyword in name for keyword in (
                    "channel_pre_norm",
                    "temporal_pre_norm",
                    "final_norm",
            )):
                param = param + 1.0
                add_trt_llm_weight(weights, trt_llm_name, param,
                                   trt_llm_config.dtype)
            elif any(keyword in name for keyword in (
                    "proj_final.bias",
                    "ffw_down.bias",
                    "linear_out.bias",
                    "o_proj.bias",
                    "down_proj.bias",
            )):
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
        safetensors.torch.save_file(weights,
                                    args.output_dir / f"rank{rank}.safetensors")


def main():
    print(tensorrt_llm.__version__)

    args = parse_arguments()
    logger.set_level(args.log_level)
    tik = time.time()

    print(f"Loading source parameters from {args.model_dir.absolute()}")
    ckpt_parser = CKPT_PARSER[args.ckpt_type]()
    ckpt_params = ckpt_parser.load_parameters(args.model_dir)
    input_embedding_weights = ckpt_parser.embedding_weights(ckpt_params)
    ckpt_params_dtype = str(input_embedding_weights.dtype).split(".")[-1]
    ckpt_config = ckpt_parser.get_config(args.model_dir, ckpt_params)

    print(f"Source configuration determined from parameters: {ckpt_config}")

    quant_config = tensorrt_llm.models.modeling_utils.QuantConfig()
    trt_llm_config = tensorrt_llm.models.modeling_utils.PretrainedConfig(
        architecture="RecurrentGemmaForCausalLM",
        dtype=args.dtype or ckpt_params_dtype,
        logits_dtype="float32",
        vocab_size=ckpt_config["vocab_size"],
        # follow the setting of gemma models
        max_position_embeddings=8192,
        hidden_size=ckpt_config["hidden_size"],
        num_hidden_layers=ckpt_config["num_hidden_layers"],
        num_attention_heads=ckpt_config["num_attention_heads"],
        num_key_value_heads=1,
        head_size=ckpt_config["hidden_size"] //
        ckpt_config["num_attention_heads"],
        hidden_act="gelu",
        intermediate_size=ckpt_config["intermediate_size"],
        norm_epsilon=1e-6,
        position_embedding_type="rope_gpt_neox",
        mapping={
            'world_size': args.world_size,
            'tp_size': args.world_size,
            'pp_size': 1
        },
        gpus_per_node=8,
        quantization=quant_config,
        conv_kernel=4,
        state_size=1,
        state_dtype='float32',
        rotary_pct=0.5,
        layer_types=ckpt_config["block_types"],
        rnn_hidden_size=ckpt_config["lru_width"],
        logits_soft_cap=ckpt_config["logits_soft_cap"],
        emb_scale_by_sqrt_dim=ckpt_config["embeddings_scale_by_sqrt_dim"],
        rnn_conv_dim_size=ckpt_config["lru_width"],
    )

    trt_llm_config_dict = trt_llm_config.to_dict()
    print(f"Determined TensorRT LLM configuration {trt_llm_config_dict}")

    config_path = args.output_dir / "config.json"
    config_path.parent.mkdir(exist_ok=True, parents=True)
    LOGGER.debug(f"Saving TensorRT LLM configuration to {config_path}")
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

# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import json
import os
import traceback
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional

import safetensors
import torch
from transformers.models.auto import AutoModel

import tensorrt_llm.models.modeling_utils
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import PretrainedConfig
from tensorrt_llm.models.llama.convert import (get_weight, get_weight_and_bias,
                                               split)

BASE_MODEL_TLLM_WEIGHT_PREFIX = "base_model."
DRAFTER_TLLM_WEIGHT_PREFIX = "drafter."

# To add support for a new base model in ReDrafter:
# 1. Add the base model's tensorrt_llm class name mapping in `REDRAFTER_MAP` below
# 2. Create a new ReDrafter class in `tensorrt_llm/redrafter/models.py` by inheriting from `ReDrafterMixin`
# 3. Add the new ReDrafter class to the model registry in `tensorrt_llm/models/__init__.py`
#
# Example:
# REDRAFTER_MAP = {
#     "QWenForCausalLM": "ReDrafterForQWenLM",
#     "Qwen2ForCausalLM": "ReDrafterForQWenLM",
#     "LlamaForCausalLM": "ReDrafterForLLaMALM"
# }

REDRAFTER_MAP = {
    "QWenForCausalLM": "ReDrafterForQWenLM",
    "Qwen2ForCausalLM": "ReDrafterForQWenLM",
    "LlamaForCausalLM": "ReDrafterForLLaMALM"
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_checkpoint_dir",
                        type=str,
                        default=None,
                        required=True)
    parser.add_argument("--drafter_model_dir",
                        type=str,
                        default=None,
                        required=True)

    parser.add_argument("--tp_size",
                        type=int,
                        default=1,
                        help="N-way tensor parallelism size")
    parser.add_argument("--dtype",
                        type=str,
                        default="float16",
                        choices=["float32", "bfloat16", "float16"])

    parser.add_argument("--storage-type",
                        "-t",
                        type=str,
                        default="fp32",
                        choices=["fp32", "fp16"])
    parser.add_argument("--load_model_on_cpu", action="store_true")
    parser.add_argument(
        "--use_parallel_embedding",
        action="store_true",
        default=False,
        help="By default embedding parallelism is disabled.",
    )
    parser.add_argument(
        "--embedding_sharding_dim",
        type=int,
        default=0,
        choices=[0, 1],
        help=
        "By default the embedding lookup table is sharded along vocab dimension (=0). "
        "To shard it along hidden dimension, set embedding_sharding_dim=1"
        "Note: embedding sharing is only enabled when embedding_sharding_dim = 0",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tllm_checkpoint",
        help="The path to save the TensorRT LLM checkpoint",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="The number of workers for converting checkpoint in parallel",
    )
    parser.add_argument(
        "--dense_context_fmha",
        default=False,
        action="store_true",
        help=
        "Enable dense fmha in context phase, otherwise sliding window attention."
        "If dense_context_fmha=False, the sliding window size is the max attention window size.",
    )

    parser.add_argument(
        "--redrafter_draft_len_per_beam",
        type=int,
        default=5,
        help=
        "Number of times that the Recurrent Drafter runs the beam search to generate draft"
        "candidates. Note that this draft_len does not include the first true/guaranteed token.",
    )
    parser.add_argument(
        "--redrafter_num_beams",
        type=int,
        default=5,
        help="Number of beam search candidates to keep during the Recurrent"
        "Drafter beam search iterations.",
    )
    parser.add_argument(
        "--redrafter_no_greedy_search",
        action="store_false",
        default=True,
        dest="redrafter_greedy_search",
        help=
        "Whether Redrafter will use the token with the highest probability from lm_head"
        "output or randomly sampled from the probability distribution.",
    )

    return parser.parse_args()


def hf_drafter(
    hf_model: Namespace,  #DrafterModel, # TODO:
    mapping: Mapping,
    dtype: torch.dtype = torch.float32,
    additional_tllm_prefix: str = "",
) -> Dict[str, torch.Tensor]:
    """
    Possible tensor names for Drafter checkpoints:
        input_proj.weight
        input_proj.bias
        lm_head.0.linear.weight
        lm_head.0.linear.bias
        lm_head.1.linear.weight
        lm_head.1.linear.bias
        lm_head.2.weight
        rnn_u.weight
        rnn_u.bias
        rnn_w.weight

        OR

        input_projs.weight
        input_projs.bias
        lm_heads.0.linear.weight
        lm_heads.0.linear.bias
        lm_heads.1.linear.weight
        lm_heads.1.linear.bias
        lm_heads.2.weight

        OR

        0.0.linear.weight
        0.0.linear.bias
        0.1.linear.weight
        0.1.linear.bias
        0.2.weight

    """

    def get_weight_and_bias_with_multiple_possible_names(
            model_params, dtype, names_to_try, bias=True):
        w, b = None, None
        for name in names_to_try:
            try:
                if bias:
                    w, b = get_weight_and_bias(model_params, name, dtype)
                else:
                    w = get_weight(model_params, name, dtype)
                break
            except:
                pass
        if not bias:
            return w
        return w, b

    weights = {}
    # TODO: When ReDrafter is added to Transformers
    # model_params = dict(hf_model.named_parameters())
    model_params = dict(hf_model.named_parameters)

    if hf_model.config.hidden_size * 2 != hf_model.config.exit_dim:
        input_proj_weight, input_proj_bias = get_weight_and_bias_with_multiple_possible_names(
            model_params, dtype, ["input_proj", "input_projs"])
        weights[f"{additional_tllm_prefix}input_proj.weight"] = split(
            input_proj_weight, mapping.tp_size, mapping.tp_rank, dim=0)
        weights[f"{additional_tllm_prefix}input_proj.bias"] = split(
            input_proj_bias, mapping.tp_size, mapping.tp_rank, dim=0)

    for layer_idx in range(hf_model.config.num_draft_layers):
        layer_weight, layer_bias = get_weight_and_bias_with_multiple_possible_names(
            model_params, dtype, [
                f"lm_head.{layer_idx}.linear", f"lm_heads.{layer_idx}.linear",
                f"0.{layer_idx}.linear"
            ])
        weights[
            f"{additional_tllm_prefix}layers.{layer_idx}.linear.weight"] = split(
                layer_weight, mapping.tp_size, mapping.tp_rank, dim=0)
        weights[
            f"{additional_tllm_prefix}layers.{layer_idx}.linear.bias"] = split(
                layer_bias, mapping.tp_size, mapping.tp_rank, dim=0)

    last_layer_weight = get_weight_and_bias_with_multiple_possible_names(
        model_params,
        dtype, [
            f"lm_head.{hf_model.config.num_draft_layers}",
            f"lm_heads.{hf_model.config.num_draft_layers}",
            f"0.{hf_model.config.num_draft_layers}"
        ],
        bias=False)
    weights[f"{additional_tllm_prefix}lm_head.weight"] = split(
        last_layer_weight, mapping.tp_size, mapping.tp_rank, dim=0)

    if hf_model.config.rnn:
        # rnn_u has both weight and bias
        rnn_u_weight, rnn_u_bias = get_weight_and_bias(model_params, "rnn_u",
                                                       dtype)
        weights[f"{additional_tllm_prefix}rnn_u.weight"] = split(
            rnn_u_weight, mapping.tp_size, mapping.tp_rank, dim=0)
        weights[f"{additional_tllm_prefix}rnn_u.bias"] = split(rnn_u_bias,
                                                               mapping.tp_size,
                                                               mapping.tp_rank,
                                                               dim=0)

        # rnn_w only has weight
        rnn_w_weight = get_weight(model_params, "rnn_w", dtype)
        weights[f"{additional_tllm_prefix}rnn_w.weight"] = split(
            rnn_w_weight, mapping.tp_size, mapping.tp_rank, dim=0)

    return weights


def hf_redrafter_config(
    tllm_base_model_config: tensorrt_llm.models.modeling_utils.PretrainedConfig,
    drafter_config: Namespace,  # DrafterConfig
    redrafter_num_beams: int,
    redrafter_draft_len_per_beam: int,
    redrafter_greedy_search: bool,
) -> tensorrt_llm.models.modeling_utils.PretrainedConfig:
    tllm_config = copy.deepcopy(tllm_base_model_config)

    tllm_config.base_model_architecture = tllm_config.architecture
    tllm_config.architecture = REDRAFTER_MAP[tllm_config.architecture]
    setattr(tllm_config, "redrafter_num_layers",
            drafter_config.num_draft_layers)
    setattr(tllm_config, "redrafter_hidden_size", drafter_config.hidden_size)
    setattr(tllm_config, "redrafter_exit_dim", drafter_config.exit_dim)
    setattr(tllm_config, "redrafter_is_rnn", drafter_config.rnn)

    # These three configs look like runtime parameters. But for TensorRT-LLM
    # implementation, they are required to be provided at engine build time and
    # TensorRT needs to unroll loops with set number of loop iterations.
    setattr(tllm_config, "redrafter_num_beams", redrafter_num_beams)
    setattr(tllm_config, "redrafter_draft_len_per_beam",
            redrafter_draft_len_per_beam)
    setattr(tllm_config, "redrafter_greedy_search", redrafter_greedy_search)

    # Exclude the redrafter weights from any quantisation
    if hasattr(tllm_config,
               "quantization") and tllm_config.quantization is not None:
        # If quantization is an object/namespace, handle it accordingly
        if getattr(tllm_config.quantization, "exclude_modules",
                   None) is not None:
            redrafter_exclude_modules = [
                'drafter', 'drafter.layers', 'drafter.lm_head'
            ]
            num_redrafter_layers = tllm_config.redrafter_num_layers
            if tllm_config.redrafter_is_rnn:
                redrafter_exclude_modules += ['drafter.rnn_u', 'drafter.rnn_w']
            for lyrnum in range(num_redrafter_layers):
                redrafter_exclude_modules += [
                    f'drafter.layers.{lyrnum}',
                    f'drafter.layers.{lyrnum}.linear'
                ]
            tllm_config.quantization.exclude_modules += redrafter_exclude_modules

    return tllm_config


def convert_and_save(
    rank: int,
    tp_size: int,
    base_model_checkpoint_dir: str,
    hf_drafter_model: Optional[AutoModel],
    dtype: str,
    use_parallel_embedding: bool,
    embedding_sharding_dim: int,
    output_dir: str,
) -> None:

    mapping = Mapping(
        world_size=tp_size,
        rank=rank,
        tp_size=tp_size,
    )

    # Load and prepare weights
    stade_dict_path = os.path.join(base_model_checkpoint_dir,
                                   f'rank{rank}.safetensors')
    weights_safe = safetensors.safe_open(stade_dict_path, framework="pt")
    weights = {k: weights_safe.get_tensor(k) for k in weights_safe.keys()}

    if hf_drafter_model is not None:
        drafter_weights = hf_drafter(
            hf_drafter_model,
            mapping,
            dtype=str_dtype_to_torch(dtype),
            additional_tllm_prefix=(DRAFTER_TLLM_WEIGHT_PREFIX
                                    if hf_drafter_model is not None else ""),
        )
        weights.update(drafter_weights)

    safetensors.torch.save_file(
        weights, os.path.join(output_dir, f"rank{rank}.safetensors"))


def multi_worker_convert_and_save(
    workers: int,
    tp_size: int,
    base_model_checkpoint_dir: str,
    hf_drafter_model: Optional[AutoModel],
    dtype: str,
    use_parallel_embedding: bool,
    embedding_sharding_dim: int,
    output_dir: str,
) -> None:
    with ThreadPoolExecutor(max_workers=workers) as p:
        futures = [
            p.submit(
                convert_and_save,
                rank,
                tp_size,
                base_model_checkpoint_dir,
                hf_drafter_model,
                dtype,
                use_parallel_embedding,
                embedding_sharding_dim,
                output_dir,
            ) for rank in range(tp_size)
        ]
        exceptions = []
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                traceback.print_exc()
                exceptions.append(e)
        assert len(
            exceptions
        ) == 0, "Checkpoint conversion failed, please check error log."


def create_and_save_config(args):
    mapping = Mapping(
        world_size=args.tp_size,
        tp_size=args.tp_size,
        pp_size=1,
    )

    base_checkpoint_dir = args.base_model_checkpoint_dir
    config_path = os.path.join(base_checkpoint_dir, 'config.json')
    model_config = PretrainedConfig.from_json_file(config_path)
    tllm_model_config = copy.deepcopy(model_config)

    if args.drafter_model_dir:
        # TODO: When ReDrafter is added to Transformers
        # drafter_hf_config = AutoConfig.from_pretrained(args.drafter_model_dir)
        with open(Path(args.drafter_model_dir, "config.json")) as fp:
            drafter_hf_config = Namespace(**json.load(fp))
        tllm_model_config = hf_redrafter_config(
            tllm_base_model_config=tllm_model_config,
            drafter_config=drafter_hf_config,
            redrafter_num_beams=args.redrafter_num_beams,
            redrafter_draft_len_per_beam=args.redrafter_draft_len_per_beam,
            redrafter_greedy_search=args.redrafter_greedy_search,
        )

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(tllm_model_config.to_dict(), f, indent=4)
    return drafter_hf_config


def main():
    args = parse_arguments()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    drafter_hf_config = create_and_save_config(args)
    base_checkpoint_dir = args.base_model_checkpoint_dir

    hf_drafter_model: Optional[AutoModel] = None
    if args.drafter_model_dir:
        # TODO: When ReDrafter is added to Transformers
        # hf_drafter_model = AutoModel.from_pretrained(
        #     args.drafter_model_dir,
        #     dtype="auto",
        # )
        ckpt_file = Path(args.drafter_model_dir, "model.safetensors")
        if not Path.exists(ckpt_file):
            ckpt_file = Path(args.drafter_model_dir, "model.pt")
        print(f"Loading drafter from {ckpt_file}")
        if str(ckpt_file).endswith(".safetensors"):
            drafter_ckpt = {}
            with safetensors.safe_open(ckpt_file, framework="pt",
                                       device="cpu") as f:
                key: str = None
                for key in f.keys():
                    drafter_ckpt[key] = f.get_tensor(key)
        else:
            drafter_ckpt = torch.load(ckpt_file, map_location='cpu')
        hf_drafter_model = Namespace(**{
            "named_parameters": drafter_ckpt,
            "config": drafter_hf_config
        })

    multi_worker_convert_and_save(
        args.workers,
        args.tp_size,
        base_checkpoint_dir,
        hf_drafter_model,
        args.dtype,
        args.use_parallel_embedding,
        args.embedding_sharding_dim,
        args.output_dir,
    )


if __name__ == "__main__":
    main()

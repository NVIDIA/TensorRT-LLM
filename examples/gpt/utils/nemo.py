# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import configparser
import functools
import logging
import os
import shutil
import tarfile
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import yaml
from transformers import GPT2Config, GPT2Tokenizer, T5Tokenizer
from utils.convert import cpu_map_location, gpu_map_location

LOGGER = logging.getLogger(__name__)

# The field names are the same as in .nemo config file
# Defaults and their locations in NeMo code are given for commit 9c7926db4ae375b77dae7eb57656213de1dd76a5 in main branch
# The commit from main is used instead of a release because there are `rotary_base` commit was introduced recently.
NemoRotaryEmbeddingParameters = namedtuple(
    "NemoRotaryEmbeddingParameters",
    [
        "position_embedding_type", "rotary_percentage",
        "seq_len_interpolation_factor", "rotary_base"
    ],
    defaults=[
        # "position_embedding_type", the default is taken from
        # https://github.com/NVIDIA/NeMo/blob/9c7926db4ae375b77dae7eb57656213de1dd76a5/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L370
        "learned_absolute",
        # "rotary_percentage", the default is taken from
        # https://github.com/NVIDIA/NeMo/blob/9c7926db4ae375b77dae7eb57656213de1dd76a5/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L370
        1.0,
        # "seq_len_interpolation_factor", the default is take from
        # https://github.com/NVIDIA/NeMo/blob/9c7926db4ae375b77dae7eb57656213de1dd76a5/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L388
        None,
        # "rotary_base", the default is taken from
        # https://github.com/NVIDIA/NeMo/blob/9c7926db4ae375b77dae7eb57656213de1dd76a5/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L389
        10000,
    ])


def set_parameter_from_config(params: Dict[str, Any], nemo_config: Dict[str,
                                                                        Any],
                              param_name: str) -> None:
    if param_name in nemo_config:
        params[param_name] = nemo_config[param_name]
    else:
        LOGGER.debug(
            f"A parameter '{param_name}' is missing in nemo checkpoint. "
            f"The default value {repr(NemoRotaryEmbeddingParameters._field_defaults[param_name])} will be used."
        )


def extract_rotary_parameters_from_nemo_config(
        nemo_config: Dict[str, Any]) -> NemoRotaryEmbeddingParameters:
    params = {}
    set_parameter_from_config(params, nemo_config, "position_embedding_type")
    set_parameter_from_config(params, nemo_config, "rotary_percentage")
    set_parameter_from_config(params, nemo_config,
                              "seq_len_interpolation_factor")
    set_parameter_from_config(params, nemo_config, "rotary_base")
    return NemoRotaryEmbeddingParameters(**params)


def nemo_to_gpt_config(nemo_model_config, vocab_size, eos_id, bos_id):
    convertion_dict = {
        "activation_function": "activation",
        "layer_norm_epsilon": "layernorm_epsilon",
        "n_embd": "hidden_size",
        "n_head": "num_attention_heads",
        "n_layer": "num_layers",
        "n_positions": "max_position_embeddings",
        "rotary_pct": "rotary_percentage",
        "bias": "bias",
        "intermediate_size": "ffn_hidden_size",
    }

    kwargs = {
        key: nemo_model_config[value]
        for key, value in convertion_dict.items() if value in nemo_model_config
    }
    kwargs["vocab_size"] = vocab_size
    kwargs["eos_token_id"] = eos_id
    kwargs["bos_token_id"] = bos_id

    return GPT2Config(**kwargs)


def copy_tokenizer_files(config, out_dir):
    basenames = {
        "model": "tokenizer",
        "vocab_file": "vocab",
        "merge_file": "merges",
    }

    for key in basenames.keys():
        if config[key] is None:
            continue
        path = Path(config[key])
        if not path.exists():
            LOGGER.debug(f"Tokenizer {key}: {path} file not found")
            continue

        dst_path = out_dir / f"{basenames[key]}{path.suffix}"
        LOGGER.debug(f"Copy tokenizer {key}: {path}->{dst_path}")
        shutil.copy(path.as_posix(), dst_path.as_posix())


def add_rotary_parameters_to_ini_config(
        config: configparser.ConfigParser,
        rotary_parameters: NemoRotaryEmbeddingParameters) -> None:
    if rotary_parameters.position_embedding_type == "rope":
        if rotary_parameters.rotary_percentage > 1.0 or rotary_parameters.rotary_percentage <= 0.0:
            raise ValueError(
                f"Rotary percentage has to suffice 0.0 < rotary_percentage <= 1.0, whereas "
                f"rotary_percentage={rotary_parameters.rotary_percentage}")
        config["gpt"]["rotary_pct"] = str(rotary_parameters.rotary_percentage)
        config["gpt"]["rotary_base"] = str(rotary_parameters.rotary_base)
        if rotary_parameters.seq_len_interpolation_factor is not None:
            if rotary_parameters.seq_len_interpolation_factor <= 1.0:
                raise ValueError(
                    f"Rotary scaling is supported only for seq_len_interpolation_factor > 1.0. "
                    f"Got seq_len_interpolation_factor={rotary_parameters.seq_len_interpolation_factor}"
                )
            config["gpt"]["rotary_scaling_type"] = "linear"
            config["gpt"]["rotary_scaling_factor"] = str(
                float(rotary_parameters.seq_len_interpolation_factor))
    else:
        # As in HF rotary_pct > 0.0 triggers RoPE. Dislabe RoPE if different embedding type is used
        config["gpt"]["rotary_pct"] = "0.0"


def update_tokenizer_paths(tokenizer_config: Dict,
                           tokenizer_file_paths: Dict[str, Optional[str]]):
    for key, new_path in tokenizer_file_paths.items():
        old_path = tokenizer_config[key]
        if old_path is None:
            continue
        old_path = Path(old_path)
        if new_path:
            LOGGER.debug(f"Update tokenizer {key} {old_path} -> {new_path}")
            tokenizer_config[key] = new_path.as_posix()
        elif not old_path.exists():
            LOGGER.warning(
                f"Tokenizer {key}'s path {old_path} does not exists: set it to None"
            )
            tokenizer_config[key] = None
    return tokenizer_config


def build_tokenizer(tokenizer_config: Dict):
    if tokenizer_config["library"] == "sentencepiece":
        tokenizer = T5Tokenizer(tokenizer_config["model"], extra_ids=0)
    elif "GPT2" in tokenizer_config["type"]:
        tokenizer = GPT2Tokenizer(tokenizer_config["vocab_file"],
                                  tokenizer_config["merge_file"])
    else:
        raise ValueError(
            f'Tokenizer type {tokenizer_config["library"]} not handled')

    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": "<s>"})
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})

    return tokenizer


def get_eos_bos_ids_from_tokenizer_config(
        tokenizer_config: Dict[str, Any]) -> Tuple[int, int]:
    tokenizer = build_tokenizer(tokenizer_config)
    return tokenizer.eos_token_id, tokenizer.bos_token_id


def nemo_config_to_ini_config(
    nemo_model_config: Dict[str, Any],
    eos_id: int,
    bos_id: int,
    vocab_size: int,
    storage_type: str,
) -> configparser.ConfigParser:
    gpt_model_config = nemo_to_gpt_config(nemo_model_config, vocab_size, eos_id,
                                          bos_id)
    config = configparser.ConfigParser()
    config["gpt"] = {k: str(v) for k, v in vars(gpt_model_config).items()}
    config["gpt"]["storage_dtype"] = storage_type
    add_rotary_parameters_to_ini_config(
        config, extract_rotary_parameters_from_nemo_config(nemo_model_config))
    return config


def add_special_tokens_to_tokenizer(tokenizer):

    # Need to add cls, sep, mask tokens to the tokenizer if they don't exist.
    # If cls, sep and mask are not attributes of the tokenizer, add it.
    if not hasattr(tokenizer, 'cls_token'):
        tokenizer.add_special_tokens({'cls_token': '<cls>'})
    if not hasattr(tokenizer.tokenizer, 'sep_id'):
        tokenizer.add_special_tokens({'sep_token': '<sep>'})
    if not hasattr(tokenizer.tokenizer, 'mask_id'):
        tokenizer.add_special_tokens({'mask_token': '<mask>'})

    # bos, eos, pad and unk may be present in the provided spm .model file, if they are, use it.
    if not hasattr(tokenizer, 'pad_token'):
        if hasattr(tokenizer.tokenizer,
                   'pad_id') and tokenizer.tokenizer.pad_id() > 0:
            tokenizer.pad_token = tokenizer.tokenizer.id_to_piece(
                tokenizer.tokenizer.pad_id())
        else:
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
    else:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})

    if not hasattr(tokenizer, 'bos_token'):
        if hasattr(tokenizer.tokenizer,
                   'bos_id') and tokenizer.tokenizer.bos_id() > 0:
            tokenizer.bos_token = tokenizer.tokenizer.id_to_piece(
                tokenizer.tokenizer.bos_id())
        else:
            tokenizer.add_special_tokens({'bos_token': '<bos>'})
    else:
        tokenizer.add_special_tokens({'bos_token': '<s>'})

    if not hasattr(tokenizer, 'eos_token'):
        if hasattr(tokenizer.tokenizer,
                   'eos_id') and tokenizer.tokenizer.eos_id() > 0:
            tokenizer.eos_token = tokenizer.tokenizer.id_to_piece(
                tokenizer.tokenizer.eos_id())
        else:
            tokenizer.add_special_tokens({'eos_token': '<eos>'})
    else:
        tokenizer.add_special_tokens({'eos_token': '</s>'})


def unpack_nemo_ckpt(nemo_archive_path: Union[str, Path],
                     out_dir_path: Union[str, Path]):
    nemo_archive_path = Path(nemo_archive_path)
    if not nemo_archive_path.exists():
        raise FileNotFoundError(f"{nemo_archive_path} does not exist")

    for tar_mode in ["r:", "r:gz"]:
        try:
            with tarfile.open(nemo_archive_path, mode=tar_mode) as tar_file:

                def is_within_directory(directory, target):

                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)

                    prefix = os.path.commonprefix([abs_directory, abs_target])

                    return prefix == abs_directory

                def safe_members(tar_file):
                    members = []
                    for member in tar_file.getmembers():
                        member_path = os.path.join(out_dir_path, member.name)
                        if not is_within_directory(out_dir_path, member_path):
                            raise Exception(
                                "Attempted Path Traversal in Tar File")
                        members.append(member)
                    return members

                tar_file.extractall(out_dir_path,
                                    members=safe_members(tar_file),
                                    numeric_owner=False)

            return out_dir_path
        except tarfile.ReadError:
            pass

    raise RuntimeError(f"Could not unpack {nemo_archive_path}")


def extract_layers_with_prefix(model_, prefix):
    length_to_trim = len(prefix)
    model_state = model_.get("state_dict", model_)
    return {
        key[length_to_trim:]: model_state[key]
        for key in model_state.keys() if prefix in key
    }


class UnpackedNemoCheckpointDir:

    def __init__(self,
                 checkpoints_dir: Union[str, Path],
                 load_checkpoints_to_cpu: bool = False):
        self._checkpoints_dir = Path(checkpoints_dir)
        self._load_checkpoints_to_cpu = load_checkpoints_to_cpu

    @property
    @functools.lru_cache
    def model_config(self):
        model_config = None

        model_config_filename = "model_config.yaml"
        model_configs_paths = list(
            self._checkpoints_dir.rglob(model_config_filename))
        if model_configs_paths:
            if len(model_configs_paths) > 1:
                raise RuntimeError(
                    f"There are more than single {model_config_filename} "
                    f"in {self._checkpoints_dir}: {', '.join(map(lambda p: p.as_posix(), model_configs_paths))}"
                )
            model_config_path = model_configs_paths[0]
            LOGGER.debug("Loading model config from %s", model_config_path)
            with model_config_path.open("r") as model_config_file:
                model_config = yaml.load(model_config_file,
                                         Loader=yaml.SafeLoader)
        else:
            LOGGER.debug("Searching model config in checkpoints")
            # try to obtain from checkpoint
            checkpoint_name = self.checkpoint_name
            checkpoints_paths = sorted(
                self._checkpoints_dir.rglob(checkpoint_name))
            if checkpoints_paths:
                # assume that parallel ranks 0 checkpoint should have model config embedded
                checkpoint_path = checkpoints_paths[0]

                map_location_fn = cpu_map_location if self._load_checkpoints_to_cpu else gpu_map_location

                model_00 = torch.load(checkpoint_path,
                                      map_location=map_location_fn)
                if "hyper_parameters" in model_00 and "cfg" in model_00[
                        "hyper_parameters"]:
                    model_config = model_00["hyper_parameters"]["cfg"]
                    LOGGER.debug("Loaded model config from checkpoint %s",
                                 checkpoint_path)
                else:
                    LOGGER.debug("Could not find model config in checkpoint %s",
                                 checkpoint_path)

                del model_00

        if model_config is None:
            LOGGER.warning(
                "Could not find checkpoint with NeMo model config in %s",
                self._checkpoints_dir)

        LOGGER.debug("Loaded model config %s", model_config)

        return model_config

    @property
    def checkpoints_dir(self):
        return self._checkpoints_dir

    def get_checkpoints_paths(self,
                              tensor_model_parallel_size=1,
                              pipeline_model_parallel_size=1):
        """
        Injects tensor/pipeline model parallel ranks into the filepath.
        Does nothing if not using model parallelism.
        """

        checkpoint_path_without_rank = self.checkpoints_dir / self.checkpoint_name

        def _inject_parallel_ranks(tp_rank, pp_rank):
            if tensor_model_parallel_size > 1 or pipeline_model_parallel_size > 1:
                if pipeline_model_parallel_size is None or pipeline_model_parallel_size == 1:
                    checkpoint_path = (checkpoint_path_without_rank.parent /
                                       f"mp_rank_{tp_rank:02d}" /
                                       checkpoint_path_without_rank.name)
                else:
                    checkpoint_path = (
                        checkpoint_path_without_rank.parent /
                        f"tp_rank_{tp_rank:02d}_pp_rank_{pp_rank:03d}" /
                        checkpoint_path_without_rank.name)
                return checkpoint_path
            else:
                return checkpoint_path_without_rank

        return [[
            _inject_parallel_ranks(tp_rank=tp_rank, pp_rank=pp_rank)
            for pp_rank in range(pipeline_model_parallel_size)
        ] for tp_rank in range(tensor_model_parallel_size)]

    @property
    @functools.lru_cache
    def checkpoint_name(self):
        patterns = [
            "model_weights.ckpt",  # older megatron checkpoints
            "*last.ckpt",  # newer format of checkpoints
        ]
        for pattern in patterns:
            model_files = sorted(list(self._checkpoints_dir.rglob(pattern)))
            if model_files:
                return model_files[0].name

        raise ValueError(
            f"Could not find checkpoint files in {self._checkpoints_dir}")

    @functools.lru_cache
    def get_tokenizer_file_path(self, tokenizer_key, file_key,
                                default_filename_pattern):
        model_config = self.model_config
        file_property = None
        if tokenizer_key in model_config and file_key in model_config[
                tokenizer_key]:
            file_property = model_config[tokenizer_key][file_key]
        elif file_key in model_config:
            file_property = model_config[file_key]

        LOGGER.debug("model_config[%s][%s]=%s", tokenizer_key, file_key,
                     file_property)

        if file_property and file_property.startswith("nemo:"):
            filename = file_property.split("nemo:")[1]
            filename_pattern = f"*{filename}"
        elif file_property and file_property.startswith("/artifacts/"):
            filename = Path(file_property).name
            filename_pattern = f"*{filename}"
        elif file_property is None or file_property == "None":
            filename_pattern = None
        else:
            filename_pattern = default_filename_pattern
            LOGGER.warning(
                f"Tokenizer file from config: {tokenizer_key}.{file_key}={file_property} "
                f"looks like unsupported path. Pattern {filename_pattern} will be used."
            )

        file_path = None
        if filename_pattern is not None:
            files_paths = list(self._checkpoints_dir.glob(filename_pattern))
            if files_paths:
                assert len(files_paths) == 1
                file_path = files_paths[0]

        return file_path

    @functools.lru_cache
    def get_all_tokenizer_file_paths(self):
        return {
            "model":
            self.get_tokenizer_file_path("tokenizer", "model", "*.model"),
            "vocab_file":
            self.get_tokenizer_file_path("tokenizer", "vocab_file", "*vocab*"),
            "merge_file":
            self.get_tokenizer_file_path("tokenizer", "merge_file",
                                         "*merge*.txt"),
        }

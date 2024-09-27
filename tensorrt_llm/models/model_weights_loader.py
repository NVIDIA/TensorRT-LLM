import glob
import math
import os
import weakref
from enum import Enum
from typing import Callable, List

import torch
from safetensors import safe_open
from tqdm import tqdm

from tensorrt_llm.layers.moe import MOEWeightWrapper
from tensorrt_llm.quantization.layers import (
    WeightOnlyGroupwiseQuantColumnLinear, WeightOnlyGroupwiseQuantRowLinear)

from .._utils import trt_dtype_to_torch
from ..logger import logger


class ModelWeightsFormat(Enum):
    SAFETENSORS = "safetensors"
    BINARY = "bin"
    PYTORCH = "pth"


class ModelWeightsLoader:
    """Convert and load external checkpoint into a TensorRT-LLM model.

    Attributes:
        model_dir                 : Model directory or in-memory torch model.
        format                    : Checkpoint file format.
        shards                    : Shard pointer list (safetensors) or shard dict lists (other types)
        shard map                 : Dict of external checkpoints' keys -> shard index.
        tllm_to_externel_key_dict : Dict of TRT-LLM keywords -> External checkpoints' keywords, based on HF LLaMA.
        customized_key_dict       : Customized dict for updating the default tllm_to_externel_key_dict.
    """

    def __init__(self, model_dir, customized_key_dict: dict = {}) -> None:

        # Checkpoint file format information
        self.model_dir = model_dir
        self.format = None
        self.model = None
        self.shards = []
        self.shard_map = {}

        # Key translator vocabulary
        self.tllm_to_externel_key_dict = {
            "transformer": "model",
            "vocab_embedding": "embed_tokens",
            "lm_head": "lm_head",
            "ln_f": "norm",
            "attention": "self_attn",
            "qkv": ["q_proj", "k_proj", "v_proj"],
            "dense": "o_proj",
            "gate": "up_proj",
            "proj": "down_proj",
            "fc": "gate_proj",
            "input_layernorm": "input_layernorm",
            "post_layernorm": "post_attention_layernorm",
        }
        self.tllm_to_externel_key_dict.update(customized_key_dict)

        self.detect_format()
        self.preload()

    def translate_to_external_key(
            self, tllm_key: str,
            tllm_to_externel_key_dict: dict) -> str | List[str]:
        """Translate TRT-LLM key into HF key or HF key list (e.g. QKV/MoE/GPTQ)

        tllm_key will get translated into HF format section by section.
        If one section is responeded with multiple hf_keys in a list, \
        the translated keys will also get multiplied accordingly.
        tllm_key : "transformer.layers.0.attention.  qkv .weight"
                          |        |   |     |        |     |
        translated: ["  model  .layers.0.self_attn.q_proj.weight,
                     "  model  .layers.0.self_attn.k_proj.weight,
                     "  model  .layers.0.self_attn.v_proj.weight]

        Args:
            tllm_key (str): Input TRT-LLM key.
            tllm_to_externel_key_dict (dict): User specified dict with higher priority. \
            Generated from layer attributes automatically.

        Returns:
            hf_keys (str | list[str]) : Translated HF key(s).
        """
        tllm_key_split = tllm_key.split(".")
        hf_key_split = tllm_key_split.copy()

        for i, tllm_word in enumerate(tllm_key_split):
            if tllm_word in tllm_to_externel_key_dict:
                hf_key_split[i] = tllm_to_externel_key_dict[tllm_word]
            elif tllm_word in self.tllm_to_externel_key_dict:
                hf_key_split[i] = self.tllm_to_externel_key_dict[tllm_word]

        hf_keys = [""]
        for hf_word in hf_key_split:
            if isinstance(hf_word, list):
                hf_keys = [
                    item for item in hf_keys for _ in range(len(hf_word))
                ]
                hf_keys = [
                    hf_key + hf_word[i % len(hf_word)]
                    for i, hf_key in enumerate(hf_keys)
                ]
            else:
                if hf_word == "":
                    continue
                hf_keys = [hf_key + hf_word for hf_key in hf_keys]
            hf_keys = [hf_key + "." for hf_key in hf_keys]
        hf_keys = [hf_key[:-1] for hf_key in hf_keys]

        return hf_keys[0] if len(hf_keys) == 1 else hf_keys

    def detect_format(self):
        if os.path.isfile(self.model_dir):
            if self.model_dir.endswith(".safetensors"):
                self.format = ModelWeightsFormat.SAFETENSORS
            elif self.model_dir.endswith(".bin"):
                self.format = ModelWeightsFormat.BINARY
            elif self.model_dir.endswith(".pth"):
                self.format = ModelWeightsFormat.PYTORCH
            else:
                raise NotImplementedError(
                    "Only safetensors/pickle/binary files are supported.")
        elif os.path.isdir(self.model_dir):
            file_list = os.listdir(self.model_dir)
            if any([f.endswith(".safetensors") for f in file_list]):
                self.format = ModelWeightsFormat.SAFETENSORS
            elif any([f.endswith(".bin") for f in file_list]):
                self.format = ModelWeightsFormat.BINARY
            elif any([f.endswith(".pth") for f in file_list]):
                self.format = ModelWeightsFormat.PYTORCH
            else:
                raise NotImplementedError(
                    "Only safetensors/pickle/binary directories are supported.")
        else:
            raise NotImplementedError(
                "args.model_dir is Neither a directory nor a file!")

    def preload(self):
        # Initialize shards and load_func
        if os.path.isdir(self.model_dir):
            shard_files = glob.glob(self.model_dir + "/*." + self.format.value)
        elif os.path.isfile(self.model_dir):
            shard_files = [self.model_dir]
        else:
            raise NotImplementedError(
                "args.model_dir is Neither a directory nor a file!")
        shard_files.sort()
        if self.format == ModelWeightsFormat.SAFETENSORS:
            self.shards = [
                safe_open(f, framework="pt", device="cpu") for f in shard_files
            ]
        elif self.format == ModelWeightsFormat.BINARY or self.format == ModelWeightsFormat.PYTORCH:
            self.shards = [
                torch.load(f, weights_only=True, map_location="cpu", mmap=True)
                for f in shard_files
            ]
        else:
            raise NotImplementedError(
                "Only *.safetensors/*.pth/*.bin files are supported.")
        for idx, shard in enumerate(self.shards):
            self.shard_map.update({k: idx for k in shard.keys()})

    def load_tensor(self, key, tp_size=1, tp_dim=-1, tp_rank=0):
        # Retrieve shard index
        if key in self.shard_map:
            ptr_idx = self.shard_map[key]
        else:
            return None

        if self.format == ModelWeightsFormat.SAFETENSORS:
            tensor = self.shards[ptr_idx].get_slice(key)
            tensor_shape = tensor.get_shape()
            if tensor_shape == []:
                tensor = self.shards[ptr_idx].get_tensor(key).unsqueeze(0)
                tensor_shape = tensor.shape
        elif self.format == ModelWeightsFormat.BINARY or self.format == ModelWeightsFormat.PYTORCH:
            tensor = self.shards[ptr_idx][key]
            tensor_shape = tensor.shape

        if tp_size <= 1 or tp_dim < 0:
            return tensor[:]
        else:
            if len(tensor_shape) == 1 and (tp_dim > 0 or tensor_shape[0] == 1):
                return tensor[:]
            else:
                width = tensor_shape[tp_dim]
                if width == 1:
                    return tensor[:]
                slice_width = math.ceil(width / tp_size)
                slice_start = tp_rank * slice_width
                slice_end = min((tp_rank + 1) * slice_width, width)
                slice_obj = [slice(None)] * len(tensor_shape)
                slice_obj[tp_dim] = slice(slice_start, slice_end)
                res = tensor[tuple(slice_obj)]
                return res

    def load(self,
             tllm_key: str,
             preprocess: Callable[[int], None] = None,
             skip_tp: bool = False,
             custom_postprocess_kwargs: dict = {}):
        """Load tensor from shards

        This function contains following steps:
            1. Translate tllm_key into external key(s).
            2. Load tensor/tensors partially according to layer attributes.
            3. Call preprocess() if it is not None.
            4. Call layer's post processing function.
            5. Return the dict for updating weight dict.

        Args:
            tllm_key (str): TRT-LLM key from model iterators
            preprocess (function, Optional): Customized preprocess function for step 3.
            skip_tp (bool): Skip TP in case of the derived TP config is inappropriate.
        """
        tp_rank = self.model.config.mapping.tp_rank

        sub_module = self.model
        for attr in tllm_key.split(".")[:-1]:
            sub_module = getattr(sub_module, attr)
        param = self.model
        for attr in tllm_key.split("."):
            param = getattr(param, attr)
        if param.is_buffer:
            return {}
        assert sub_module is not None and param is not None, f"{tllm_key} got Nonetype for parameter or parent module."

        tllm_to_externel_key_dict = sub_module.tllm_to_externel_key_dict if hasattr(
            sub_module, "tllm_to_externel_key_dict") else {}
        tp_dim = sub_module.tp_dim if hasattr(sub_module, "tp_dim") else -1
        require_weight_transpose = (
            isinstance(sub_module, WeightOnlyGroupwiseQuantColumnLinear)
            or isinstance(sub_module, WeightOnlyGroupwiseQuantRowLinear)
        ) and tllm_key.endswith("weight")
        if tp_dim >= 0 and require_weight_transpose:
            tp_dim = 1 - tp_dim
        tp_size = sub_module.tp_size if hasattr(sub_module, "tp_size") else 1
        if skip_tp:
            tp_dim = -1
            tp_size = 1
        if isinstance(sub_module, MOEWeightWrapper):
            tp_rank = self.model.config.mapping.moe_tp_rank
        external_key = self.translate_to_external_key(
            tllm_key, tllm_to_externel_key_dict)

        if isinstance(external_key, list):
            v = [
                self.load_tensor(k, tp_size, tp_dim, tp_rank)
                for k in external_key
            ]
        else:
            v = self.load_tensor(external_key, tp_size, tp_dim, tp_rank)

        if preprocess is not None:
            v = preprocess(v)

        if not hasattr(sub_module, "postprocess"):
            if isinstance(v, list):
                raise ValueError(
                    f"Param {tllm_key} is translated into {external_key}, post-process function is required."
                )
            elif v is None:
                weight_dict = {}
            else:
                weight_dict = {tllm_key: v.to(trt_dtype_to_torch(param.dtype))}
        else:
            postprocess_kwargs = {"config": self.model.config}
            postprocess_kwargs.update(custom_postprocess_kwargs)
            v = sub_module.postprocess(tllm_key, v, **postprocess_kwargs)
            if isinstance(v, dict):
                weight_dict = v
            else:
                weight_dict = {tllm_key: v}

        for k, v in weight_dict.items():
            if not v.is_contiguous():
                weight_dict[k] = v.contiguous()

        return weight_dict

    def check_share_embedding(self, config):
        # TODO: Remove after --use_share_embedding is removed
        if not config.share_embedding_table:
            return

        from ..logger import logger
        lm_head_weights = self.load_tensor(
            self.translate_to_external_key("lm_head.weight",
                                           self.tllm_to_externel_key_dict))
        vocab_embed_weights = self.load_tensor(
            self.translate_to_external_key("transformer.vocab_embedding.weight",
                                           self.tllm_to_externel_key_dict))
        share_embedding_table = False
        if lm_head_weights is not None and vocab_embed_weights is not None:
            if lm_head_weights.shape == vocab_embed_weights.shape:
                if not (lm_head_weights - vocab_embed_weights).any():
                    share_embedding_table = True
        elif lm_head_weights is None and vocab_embed_weights is not None:
            self.tllm_to_externel_key_dict[
                'lm_head'] = self.tllm_to_externel_key_dict[
                    'transformer'] + '.' + self.tllm_to_externel_key_dict[
                        'vocab_embedding']
            share_embedding_table = True
        elif lm_head_weights is not None and vocab_embed_weights is None:
            self.tllm_to_externel_key_dict[
                'vocab_embedding'] = self.tllm_to_externel_key_dict['lm_head']
            share_embedding_table = True

        # Validation
        mapping = config.mapping
        if mapping.tp_size > 1:
            if (not config.use_parallel_embedding) or (
                    config.use_parallel_embedding
                    and config.embedding_sharding_dim == 1):
                share_embedding_table = False
        if mapping.pp_size > 1:
            share_embedding_table = False
        if mapping.cp_size > 1:
            share_embedding_table = False
        config.share_embedding_table = share_embedding_table

        if config.share_embedding_table:
            logger.info("share_embedding_table enabled.")

    def update_key_mapping(self, model):
        self.model = weakref.ref(model)()
        # Auto PP
        config = model.config
        if config.mapping.has_pp():
            pp_layers = config.mapping.pp_layers(config.num_hidden_layers)
            self.tllm_to_externel_key_dict.update({
                str(tllm_local_layer_idx): str(hf_global_layer_idx)
                for tllm_local_layer_idx, hf_global_layer_idx in enumerate(
                    pp_layers)
            })

        # Share embedding
        if self.tllm_to_externel_key_dict[
                'vocab_embedding'] == self.tllm_to_externel_key_dict['lm_head']:
            self.model.transformer.vocab_embedding.tllm_to_externel_key_dict = {
                self.tllm_to_externel_key_dict['transformer']: '',
            }

    def fill(self, weights):
        for tllm_key, param in self.model.named_parameters():
            if param.is_buffer:
                continue
            w_shape = weights[tllm_key].shape
            if w_shape != param.shape:
                logger.warning(
                    f'{tllm_key} has invalid shape {w_shape}. Expected {param.shape}.'
                )
                pad = torch.nn.functional.pad
                pad_dim = []
                for dim in range(weights[tllm_key].dim()):
                    current_dim = -1 - dim
                    pad_dim.append(0)
                    pad_dim.append(
                        max(0, param.shape[current_dim] - w_shape[current_dim]))
                try:
                    logger.warning(
                        f'{tllm_key} is going to be padded by {pad_dim}.')
                    weights[tllm_key] = pad(weights[tllm_key],
                                            tuple(pad_dim),
                                            value=0)
                    assert weights[tllm_key].shape == param.shape
                except:
                    raise ValueError(
                        f'Parameter {tllm_key} has invalid shape {weights[tllm_key].shape} compared with expected shape {param.shape}. Auto padding failed.'
                    )
            param.value = weights[tllm_key]

    def generate_tllm_weights(self,
                              model,
                              custom_postprocess_kwargs: dict = {}):
        # For customization, please copy this function and make changes inside the for loop.
        self.update_key_mapping(model)
        tllm_weights = {}
        for tllm_key, _ in tqdm(model.named_parameters()):
            tllm_weights.update(
                self.load(tllm_key,
                          custom_postprocess_kwargs=custom_postprocess_kwargs))
        self.fill(tllm_weights)

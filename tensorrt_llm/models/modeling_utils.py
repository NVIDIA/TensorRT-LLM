import argparse
import copy
import dataclasses
import fnmatch
import json
import os
import re
from enum import IntFlag, auto
from functools import cached_property
from pathlib import Path
from typing import (TYPE_CHECKING, Callable, Dict, Generator, List, Optional,
                    Union)

import numpy as np
import safetensors
import torch

from .._common import default_net
from .._utils import (QuantModeWrapper, get_init_params, numpy_to_torch,
                      release_gc, str_dtype_to_torch, str_dtype_to_trt,
                      trt_dtype_to_torch)
from ..bindings import KVCacheType
from ..bindings.executor import RuntimeDefaults
from ..functional import (PositionEmbeddingType, Tensor, allgather, constant,
                          cp_split_plugin, gather_last_token_logits,
                          index_select, tanh, view)
from ..layers import (MLP, AttentionParams, Embedding, FusedGatedMLP,
                      FusedRgLru, GatedMLP, KeyValueCacheParams, LoraParams,
                      PromptTuningEmbedding, RgLru)
from ..layers.attention import Attention, BertAttention
from ..layers.linear import ColumnLinear, Linear, RowLinear
from ..layers.lora import Dora, Lora
from ..layers.moe import MOE, MoeOOTB
from ..logger import logger
from ..mapping import Mapping
from ..module import Module, ModuleList
from ..parameter import Parameter
from ..plugin import init_all_reduce_helper
from ..quantization import QuantMode
from ..quantization.functional import preprocess_weights_for_mixed_gemm
from ..quantization.layers import (FP8Linear, Fp8RowwiseFusedGatedMLP,
                                   Fp8RowwiseGatedMLP,
                                   WeightOnlyGroupwiseQuantLinear,
                                   WeightOnlyGroupwiseQuantRowLinear,
                                   WeightOnlyQuantLinear,
                                   WeightOnlyQuantRowLinear)
from ..quantization.mode import (KV_CACHE_QUANT_ALGO_LIST, QUANT_ALGO_LIST,
                                 W8A8_SQ_PLUGIN_LIST, QuantAlgo)
from ..quantization.utils import fp4_utils
from ..top_model_mixin import TopModelMixin
from .convert_utils import weight_only_quantize_dict
from .generation_mixin import GenerationMixin


@dataclasses.dataclass(kw_only=True, frozen=True)
class Gemma2ConfigGroup:
    query_pre_attn_scalar: int
    final_logit_softcapping: Optional[float]
    attn_logit_softcapping: Optional[float]

    @classmethod
    def keys(cls):
        return {f.name for f in dataclasses.fields(cls)}


@dataclasses.dataclass(kw_only=True, frozen=True)
class Gemma3ConfigGroup:
    query_pre_attn_scalar: float
    final_logit_softcapping: Optional[float]
    _sliding_window_pattern: int
    rope_local_base_freq: int
    sliding_window: int

    @classmethod
    def keys(cls):
        return {f.name for f in dataclasses.fields(cls)}


if TYPE_CHECKING:
    from typing import Type, TypeVar

    from typing_extensions import Self

    ConfigGroups = Union[Gemma2ConfigGroup, Gemma3ConfigGroup]
    """Groupings of config where, if one of said properties exists, we assume all of the properties exist (even if they are `None`)"""
    CG = TypeVar("CG", bound=ConfigGroups)

    RuntimeDefaultsIn = Optional[Union[RuntimeDefaults, dict]]


class SpeculativeDecodingMode(IntFlag):
    # [WARNING] KEEP BELOW DEFINITION IN SYNC WITH cpp/tensorrt_llm/runtime/speculativeDecodingMode.h
    NONE = auto()
    DRAFT_TOKENS_EXTERNAL = auto()
    MEDUSA = auto()
    LOOKAHEAD_DECODING = auto()
    EXPLICIT_DRAFT_TOKENS = auto()
    EAGLE = auto()
    NGRAM = auto()
    USER_PROVIDED = auto()
    SAVE_HIDDEN_STATES = auto()
    AUTO = auto()

    @staticmethod
    def from_arguments(args: argparse.Namespace):
        if args.speculative_decoding_mode is None:
            return SpeculativeDecodingMode.NONE
        elif args.speculative_decoding_mode == "draft_tokens_external":
            return SpeculativeDecodingMode.DRAFT_TOKENS_EXTERNAL
        elif args.speculative_decoding_mode == "medusa":
            return SpeculativeDecodingMode.MEDUSA
        elif args.speculative_decoding_mode == "lookahead_decoding":
            return SpeculativeDecodingMode.LOOKAHEAD_DECODING
        elif args.speculative_decoding_mode == "explicit_draft_tokens":
            return SpeculativeDecodingMode.EXPLICIT_DRAFT_TOKENS
        elif args.speculative_decoding_mode == "eagle":
            return SpeculativeDecodingMode.EAGLE
        elif args.speculative_decoding_mode == "ngram":
            return SpeculativeDecodingMode.NGRAM
        elif args.speculative_decoding_mode == "user_provided":
            return SpeculativeDecodingMode.USER_PROVIDED
        elif args.speculative_decoding_mode == "auto":
            return SpeculativeDecodingMode.AUTO
        elif args.speculative_decoding_mode == "save_hidden_states":
            return SpeculativeDecodingMode.SAVE_HIDDEN_STATES
        else:
            assert False, "Unknown speculative_decoding_mode " + args.speculative_decoding_mode


@dataclasses.dataclass
class QuantConfig:
    """
    Serializable quantization configuration class, part of the PretrainedConfig.

    Args:
        quant_algo (tensorrt_llm.quantization.mode.QuantAlgo, optional): Quantization algorithm. Defaults to None.
        kv_cache_quant_algo (tensorrt_llm.quantization.mode.QuantAlgo, optional): KV cache quantization algorithm. Defaults to None.
        group_size (int): The group size for group-wise quantization. Defaults to 128.
        smoothquant_val (float): The smoothing parameter alpha used in smooth quant. Defaults to 0.5.
        clamp_val (List[float], optional): The clamp values used in FP8 rowwise quantization. Defaults to None.
        use_meta_recipe (bool): Whether to use Meta's recipe for FP8 rowwise quantization. Defaults to False.
        has_zero_point (bool): Whether to use zero point for quantization. Defaults to False.
        pre_quant_scale (bool): Whether to use pre-quant scale for quantization. Defaults to False.
        exclude_modules (List[str], optional): The module name patterns that are skipped in quantization. Defaults to None.
        mamba_ssm_cache_dtype (str, optional): The data type for mamba SSM cache. Defaults to None.
    """
    quant_algo: Optional[QuantAlgo] = None
    kv_cache_quant_algo: Optional[QuantAlgo] = None
    group_size: int = 128
    smoothquant_val: float = 0.5
    clamp_val: Optional[List[float]] = None
    use_meta_recipe: bool = False
    has_zero_point: bool = False
    pre_quant_scale: bool = False
    exclude_modules: Optional[List[str]] = None
    mamba_ssm_cache_dtype: Optional[str] = None

    @cached_property
    def quant_mode(self) -> QuantModeWrapper:
        quant_mode_list = [
            QuantMode.from_quant_algo(
                self.quant_algo,
                self.kv_cache_quant_algo,
            )
        ]
        return QuantModeWrapper(quant_mode_list)

    @cached_property
    def layer_quant_mode(self) -> QuantMode:
        return QuantMode.from_quant_algo(
            self.quant_algo,
            self.kv_cache_quant_algo,
        )

    @property
    def _use_plugin_sq(self):
        return self.quant_algo in W8A8_SQ_PLUGIN_LIST

    @property
    def _requires_calibration(self):
        return self.quant_algo in (set(QUANT_ALGO_LIST) - {
            QuantAlgo.W8A16, QuantAlgo.W4A16,
            QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN
        }) or self.kv_cache_quant_algo in KV_CACHE_QUANT_ALGO_LIST

    @property
    def _requires_modelopt_quantization(self):
        if self.quant_algo in [
                QuantAlgo.NVFP4, QuantAlgo.FP8, QuantAlgo.W4A16_AWQ,
                QuantAlgo.W4A8_AWQ, QuantAlgo.W8A8_SQ_PER_CHANNEL,
                QuantAlgo.MIXED_PRECISION
        ]:
            return True
        elif self.quant_algo is None and self.kv_cache_quant_algo == QuantAlgo.FP8:
            return True
        else:
            return False

    def _get_quant_cfg(self, module_name=None):
        if self.exclude_modules is not None:
            for exclude_module in self.exclude_modules:
                if exclude_module == module_name or (
                        exclude_module.endswith('*')
                        and module_name.startswith(exclude_module[:-1])):
                    return LayerQuantConfig(quant_algo=None,
                                            quantized_layers={})
        return self

    def _get_modelopt_qformat(self):
        algo_to_modelopt_map = {
            QuantAlgo.W8A16: "int8_wo",
            QuantAlgo.W4A16: "int4_wo",
            QuantAlgo.NVFP4: "nvfp4",
            QuantAlgo.FP8: "fp8",
            QuantAlgo.W4A16_AWQ: "int4_awq",
            QuantAlgo.W4A8_AWQ: "w4a8_awq",
            QuantAlgo.W8A8_SQ_PER_CHANNEL: "int8_sq",
        }
        assert self.quant_algo != QuantAlgo.MIXED_PRECISION, f"We don't support mixed precision in QuantConfig"
        if self.quant_algo is not None:
            assert self.quant_algo in algo_to_modelopt_map, f"We don't use Modelopt for quantization algorithm {self.quant_algo}, you probably shall not call this"
            return algo_to_modelopt_map[self.quant_algo]
        else:
            return 'full_prec'

    def _get_modelopt_kv_cache_dtype(self):
        algo_to_modelopt_map = {
            QuantAlgo.FP8: 'fp8',
            QuantAlgo.INT8: 'int8',
        }
        if self.kv_cache_quant_algo is not None:
            assert self.kv_cache_quant_algo in algo_to_modelopt_map, f"We don't use Modelopt for quantization algorithm {self.kv_cache_quant_algo}, you probably shall not call this"
            return algo_to_modelopt_map[self.kv_cache_quant_algo]
        else:
            return None

    def is_module_excluded_from_quantization(self, name: str) -> bool:
        """Check if the module is excluded from quantization.

        Args:
            name (str): The name of the module.

        Returns:
            bool: True if the module is excluded from quantization, False otherwise.
        """
        if self.exclude_modules is not None:
            for exclude_module in self.exclude_modules:
                if fnmatch.fnmatchcase(name, exclude_module):
                    return True
        return False

    @classmethod
    def from_dict(cls, config: dict) -> 'QuantConfig':
        """Create a QuantConfig instance from a dict.

        Args:
            config (dict): The dict used to create QuantConfig.

        Returns:
            tensorrt_llm.models.modeling_utils.QuantConfig: The QuantConfig created from dict.
        """
        obj = cls(**config)
        return obj

    def to_dict(self) -> dict:
        """Dump a QuantConfig instance to a dict.

        Returns:
            dict: The dict dumped from QuantConfig.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass
class LayerQuantConfig(QuantConfig):
    quant_algo: Optional[QuantConfig] = None
    kv_cache_quant_algo: Optional[QuantConfig] = None
    quantized_layers: Optional[Dict[str, QuantConfig]] = None

    def __init__(self,
                 *,
                 quant_algo: Optional[QuantConfig] = None,
                 kv_cache_quant_algo: Optional[QuantConfig] = None,
                 quantized_layers: Optional[Dict[str, QuantConfig]] = None,
                 **kwargs):
        self.quant_algo = quant_algo
        self.quantized_layers = quantized_layers
        self.kv_cache_quant_algo = kv_cache_quant_algo
        self.auto_quant_mode = {}
        for name, layer_config in self.quantized_layers.items():
            self.auto_quant_mode.update({
                name:
                QuantMode.from_quant_algo(
                    layer_config.quant_algo,
                    self.kv_cache_quant_algo,
                )
            })
        for key in kwargs:
            logger.warning(
                f"Warning: Unrecognized parameter '{key}' with value '{kwargs[key]}'"
            )

    @cached_property
    def quant_mode(self):
        quant_mode_list = list(set(self.auto_quant_mode.values()))
        return QuantModeWrapper(quant_mode_list)

    #@lru_cache(maxsize=None)
    def layer_quant_mode(self, layer_name) -> QuantMode:

        for name, quant_mode in self.auto_quant_mode.items():
            if fnmatch.fnmatch(layer_name, name):
                return quant_mode

        return QuantMode(0)

    @cached_property
    def auto_quant_list(self):
        quant_list = []
        for _, layer_config in self.quantized_layers.items():
            quant_list.append(layer_config.quant_algo)
        return list(set(quant_list))

    @classmethod
    def from_dict(cls, config: dict):
        quantized_layers = config.pop('quantized_layers', {})

        quantized_layers_dict = {
            layer_name: QuantConfig(**layer_config)
            for layer_name, layer_config in quantized_layers.items()
        }

        obj = cls(quantized_layers=quantized_layers_dict, **config)
        return obj

    #@lru_cache(maxsize=None)
    def _get_quant_cfg(self, module_name):
        quant_res = QuantConfig()

        for name, quant_cfg in self.quantized_layers.items():
            if fnmatch.fnmatch(module_name, name):
                quant_res = quant_cfg
                break
        return quant_res

    def _get_modelopt_qformat(self):
        algo_to_modelopt_map = {
            QuantAlgo.NVFP4: "nvfp4",
            QuantAlgo.FP8: "fp8",
            QuantAlgo.W4A16_AWQ: "int4_awq",
            QuantAlgo.W4A8_AWQ: "w4a8_awq",
            QuantAlgo.W8A8_SQ_PER_CHANNEL: "int8_sq",
        }
        assert self.quant_algo == QuantAlgo.MIXED_PRECISION, f"We only support mixed precision quantization in LayerQuantConfig"
        autoq_format = ','.join(
            [algo_to_modelopt_map[item] for item in self.auto_quant_list])
        return autoq_format

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output.pop('auto_quant_mode', None)
        output.pop('quant_mode', None)
        for name, per_layer_config in output['quantized_layers'].items():
            per_layer_config = per_layer_config.to_dict()
            output['quantized_layers'][name] = per_layer_config
        return output


class PretrainedConfig:

    def __init__(self,
                 *,
                 architecture: str,
                 dtype: str,
                 hidden_size: int,
                 num_hidden_layers: int,
                 num_attention_heads: int,
                 vocab_size: Optional[int] = None,
                 hidden_act: str = 'gelu',
                 logits_dtype: str = 'float32',
                 norm_epsilon: float = 1e-5,
                 position_embedding_type: Union[
                     PositionEmbeddingType,
                     str] = PositionEmbeddingType.learned_absolute,
                 max_position_embeddings: Optional[int] = None,
                 rotary_embedding_dim: Optional[int] = None,
                 num_key_value_heads: Optional[int] = None,
                 intermediate_size: Optional[int] = None,
                 mapping: Optional[Union[Mapping, dict]] = None,
                 quantization: Optional[Union[QuantConfig, dict]] = None,
                 use_parallel_embedding: bool = False,
                 embedding_sharding_dim: int = 0,
                 head_size: Optional[int] = None,
                 qk_layernorm: bool = False,
                 runtime_defaults: "RuntimeDefaultsIn" = None,
                 **kwargs):
        self.architecture = architecture
        self.dtype = dtype
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act

        self.logits_dtype = logits_dtype
        self.norm_epsilon = norm_epsilon

        self.runtime_defaults = self.create_runtime_defaults(runtime_defaults)

        if isinstance(position_embedding_type, str):
            position_embedding_type = PositionEmbeddingType.from_string(
                position_embedding_type)
        assert isinstance(position_embedding_type, PositionEmbeddingType)
        self.position_embedding_type = position_embedding_type

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        if intermediate_size is None:
            intermediate_size = hidden_size * 4
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings

        if mapping is None:
            mapping = Mapping()
        elif isinstance(mapping, dict):
            mapping = Mapping.from_dict(mapping)
        assert isinstance(mapping, Mapping)
        self.mapping = mapping

        if quantization is None:
            quantization = QuantConfig()
        elif isinstance(quantization, dict):
            quantization = QuantConfig.from_dict(quantization)
        assert isinstance(quantization, QuantConfig)
        self.quantization = quantization

        self.use_parallel_embedding = use_parallel_embedding
        self.embedding_sharding_dim = embedding_sharding_dim

        if head_size is None:
            head_size = hidden_size // num_attention_heads
        self.head_size = head_size
        self.qk_layernorm = qk_layernorm

        if rotary_embedding_dim is None:
            rotary_embedding_percentage = kwargs.get('rotary_pct', 1.0)
            rotary_embedding_dim = kwargs.get(
                'rotary_dim', int(head_size * rotary_embedding_percentage))
        self.rotary_embedding_dim = rotary_embedding_dim

        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
                logger.warning(
                    f"Implicitly setting {self.__class__.__name__}.{key} = {value}"
                )
            except AttributeError as err:
                raise err

    @staticmethod
    def create_runtime_defaults(
            defaults: "RuntimeDefaultsIn" = None) -> Optional[RuntimeDefaults]:
        if isinstance(defaults, dict):
            return RuntimeDefaults(**defaults)
        return defaults

    @property
    def kv_dtype(self):
        # TODO: need to align the kv dtype
        # now assume the kv cache is for all layers
        if self.quant_mode.has_int8_kv_cache():
            return 'int8'
        elif self.quant_mode.has_fp8_kv_cache():
            return 'fp8'
        elif self.quant_mode.has_fp4_kv_cache():
            return 'fp4'
        else:
            return self.dtype

    def set_if_not_exist(self, key, value):
        if not hasattr(self, key):
            setattr(self, key, value)

    @classmethod
    def from_dict(cls, config: dict):
        # Maybe we need AutoConfig for this
        from . import MODEL_MAP
        model_cls = MODEL_MAP[config['architecture']]
        config_cls = getattr(model_cls, 'config_class', cls)
        return config_cls(**config)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)

        output['position_embedding_type'] = str(self.position_embedding_type)
        output['mapping'] = self.mapping.to_dict()
        output['mapping'].pop('rank')
        output['quantization'] = self.quantization.to_dict()

        return output

    @classmethod
    def from_json_file(cls, config_file: str):
        with open(config_file) as f:
            config = json.load(f)
        obj = cls.from_dict(config)
        if obj.quantization.quant_algo == QuantAlgo.MIXED_PRECISION:
            try:
                layer_config_path = str(config_file).replace(
                    'config.json', 'quant_cfg.json')
                obj.to_layer_quant_config(layer_config_path)
            except Exception as e:
                raise RuntimeError(
                    f"Encounter error '{e}' for read quantization config '{layer_config_path}'"
                )
        return obj

    @classmethod
    def from_checkpoint(cls, ckpt_dir: str):
        return cls.from_json_file(os.path.join(ckpt_dir, 'config.json'))

    def to_json_file(self, config_file: str):
        with open(config_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    def to_layer_quant_config(self, config_file: str):
        with open(config_file) as f:
            config = json.load(f)

        if self.architecture == "MixtralForCausalLM":
            for layer_name in list(config["quantized_layers"].keys()):
                quant_cfg = config["quantized_layers"][layer_name]
                if "mlp.fc" in layer_name or "mlp.proj" in layer_name:
                    moe_name, _ = layer_name.rsplit('.', 1)
                    if moe_name not in config["quantized_layers"]:
                        config["quantized_layers"][moe_name] = quant_cfg
                    else:
                        assert quant_cfg == config["quantized_layers"][
                            moe_name], "MoE module needs to have the same quantization format for non-rounter sub-modules"

        self.quantization = LayerQuantConfig.from_dict(config)

    @property
    def quant_mode(self):
        return self.quantization.quant_mode

    @property
    def quant_algo(self):
        return self.quantization.quant_algo

    def _get_quant_cfg(self, module_name: str):
        return self.quantization._get_quant_cfg(module_name)

    def set_rank(self, rank: int):
        self.mapping.rank = rank

    def get_config_group(self, group_cls: "Type[CG]") -> "CG":
        cfg = {k: v for k, v in self.to_dict().items() if k in group_cls.keys()}
        return group_cls(**cfg)

    def has_config_group(self, group_cls: "Type[CG]") -> "bool":
        return all(hasattr(self, key) for key in group_cls.keys())

    def for_each_rank(self) -> "Generator[Self, None, None]":
        for rank in range(self.mapping.world_size):
            config_copy = copy.deepcopy(self)
            config_copy.set_rank(rank)
            yield config_copy


class DecoderLayerList(ModuleList):

    def __init__(self, cls, config):
        self.num_hidden_layers = config.num_hidden_layers
        self.layer_list = config.mapping.pp_layers(config.num_hidden_layers)
        self.quant_mode = config.quant_mode
        super().__init__([cls(config, idx) for idx in self.layer_list])

    def forward(self,
                hidden_states,
                use_cache=False,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                mrope_params=None,
                position_ids=None,
                lora_params=None,
                spec_decoding_params=None,
                vision_token_mask=None):
        kv_cache_params.fill_none_tensor_list(len(self.layer_list))

        if use_cache:
            presents = []

        for layer_idx, (layer, past) in enumerate(
                zip(self, kv_cache_params.past_key_value)):

            lora_layer_params = None
            if lora_params is not None and lora_params.lora_ranks is not None:
                lora_layer_params = lora_params.get_layer_params(layer_idx)

            kwargs = {}
            if position_ids is not None:
                kwargs['position_ids'] = position_ids
            if vision_token_mask is not None:
                kwargs['vision_token_mask'] = vision_token_mask
            if lora_layer_params is not None:
                kwargs['lora_layer_params'] = lora_layer_params
            if spec_decoding_params is not None:
                kwargs['spec_decoding_params'] = spec_decoding_params
            if mrope_params is not None:
                kwargs['mrope_params'] = mrope_params

            if default_net().plugin_config.reduce_fusion:
                if layer_idx + self.layer_list[0] < self.layer_list[-1]:
                    qkv_activation_scaling_factor = None
                    if default_net().plugin_config.user_buffer:
                        qkv_linear = self[layer_idx + 1].attention.qkv
                        if self.quant_mode.has_fp8_qdq():
                            qkv_activation_scaling_factor = constant(
                                qkv_linear.activation_scaling_factor.raw_value.
                                copy())
                        elif self.quant_mode.has_nvfp4():
                            qkv_activation_scaling_factor = constant(
                                qkv_linear.activation_global_scaling_factor.
                                raw_value.copy())
                    kwargs['next_layer_input_layernorm_args'] = (
                        self[layer_idx + 1].input_layernorm.weight.value,
                        self[layer_idx + 1].input_layernorm.eps,
                        qkv_activation_scaling_factor)
                else:
                    kwargs['next_layer_input_layernorm_args'] = None
            elif default_net().plugin_config.norm_quant_fusion:
                if layer_idx < self.layer_list[-1] - self.layer_list[0]:
                    try:
                        activation_scaling_factor = constant(
                            self[layer_idx + 1].attention.qkv.
                            activation_global_scaling_factor.raw_value.copy())
                    except:
                        activation_scaling_factor = None
                    kwargs['next_layer_input_layernorm_args'] = (
                        self[layer_idx + 1].input_layernorm.weight.value,
                        self[layer_idx + 1].input_layernorm.eps,
                        activation_scaling_factor)
                else:
                    kwargs['next_layer_input_layernorm_args'] = None

            hidden_states = layer(
                hidden_states,
                use_cache=use_cache,
                attention_mask=attention_mask,
                kv_cache_params=KeyValueCacheParams(
                    past_key_value=[past],
                    host_past_key_value_lengths=kv_cache_params.
                    host_past_key_value_lengths,
                    host_max_attention_window_sizes=kv_cache_params.
                    host_max_attention_window_sizes,
                    host_sink_token_length=kv_cache_params.
                    host_sink_token_length,
                    kv_cache_block_offsets=kv_cache_params.
                    kv_cache_block_offsets,
                    host_kv_cache_block_offsets=kv_cache_params.
                    host_kv_cache_block_offsets,
                    host_kv_cache_pool_pointers=kv_cache_params.
                    host_kv_cache_pool_pointers,
                    host_kv_cache_pool_mapping=kv_cache_params.
                    host_kv_cache_pool_mapping,
                    cache_indirection=kv_cache_params.cache_indirection),
                attention_params=attention_params,
                **kwargs)

            if use_cache:
                presents.append(hidden_states[1])
                hidden_states = hidden_states[0]

        if use_cache:
            return hidden_states, presents
        return hidden_states


class PostInitCaller(type):

    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__()
        return obj


class PretrainedModel(Module,
                      GenerationMixin,
                      TopModelMixin,
                      metaclass=PostInitCaller):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        init_all_reduce_helper()
        self.config = config

    def __post_init__(self):
        from ..quantization.quantize import quantize
        quantize(self, self.config.quantization)

        # Currently, use_parallel_embedding must be enabled before weight loading;
        # otherwise, the model will be inconsistent with the weights loaded from checkpoint.
        optimize_model(
            self, use_parallel_embedding=self.config.use_parallel_embedding)

    def release(self):
        release_gc()

    def __del__(self):
        self.release()

    def check_config(self, config):
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @classmethod
    def from_config(cls, config: PretrainedConfig):
        return cls(config)

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_dir: str,
        rank: Optional[int] = None,
        config: Optional[PretrainedConfig] = None,
        *,
        preprocess_weights_hook: Optional[Callable[[Dict[str, Tensor]],
                                                   Dict[str, Tensor]]] = None):
        if config is None:
            config = PretrainedConfig.from_json_file(
                os.path.join(ckpt_dir, 'config.json'))

        if rank is not None:
            config.set_rank(rank)

        rank = config.mapping.rank
        if config.mapping.cp_size > 1:
            # tp_cp_pp rank -> tp_pp rank: because different cp ranks share the same ckpt
            tp_size = config.mapping.tp_size
            cp_size = config.mapping.cp_size
            rank = rank % tp_size + rank // (tp_size * cp_size) * tp_size
        weights_path = os.path.join(ckpt_dir, f'rank{rank}.safetensors')

        assert os.path.isfile(weights_path)
        weights = safetensors.torch.load_file(weights_path)
        is_checkpoint_pruned = getattr(config, 'is_pruned', False)

        if preprocess_weights_hook is not None:
            weights = preprocess_weights_hook(weights)

        weights = preprocess_weights(weights,
                                     config,
                                     from_pruned=is_checkpoint_pruned)
        model = cls(config)
        model.load(weights, from_pruned=is_checkpoint_pruned)
        return model

    def load(self, weights, from_pruned=False):
        required_names = set()
        for name, param in self.named_parameters():
            if param.is_inited():
                continue
            if name not in weights:
                # Exemption for embedding sharing
                if name.endswith('lm_head.weight') and any(
                        k.endswith('vocab_embedding.weight')
                        for k in weights.keys()):
                    continue
                if name.endswith('lm_head.per_channel_scale') and any(
                        k.endswith('vocab_embedding.per_channel_scale')
                        for k in weights.keys()):
                    continue
            required_names.add(name)

        provided_names = set(weights.keys())

        if not required_names.issubset(provided_names):
            raise RuntimeError(
                f"Required but not provided tensors:{required_names.difference(provided_names)}"
            )
        if not provided_names.issubset(required_names):
            logger.warning(
                f"Provided but not required tensors: {provided_names.difference(required_names)}"
            )

        for name, param in self.named_parameters():
            if name in provided_names:
                if not from_pruned:
                    try:
                        param.value = weights[name]
                    except Exception as e:
                        raise RuntimeError(
                            f"Encounter error '{e}' for parameter '{name}'")
                else:
                    param.set_value_or_dummy(weights[name])

    def save_checkpoint(self, output_dir, save_config=True):
        # multiple ranks could share same config.json, so adding a save_config parameter to let user avoiding writing config.json in all ranks
        rank = self.config.mapping.rank
        weights = {
            name: numpy_to_torch(param.raw_value)
            for name, param in self.named_parameters()
        }
        # If there are some tensors share memory, this will lead to error when we call "save_file". So, for repeated tensors, we
        # clone the tensors to prevent this issue.
        data_ptrs = set()
        for name, param in weights.items():
            if param.data_ptr() in data_ptrs:
                weights[name] = param.clone()
            data_ptrs.add(weights[name].data_ptr())
        safetensors.torch.save_file(
            weights, os.path.join(output_dir, f'rank{rank}.safetensors'))
        if save_config:
            self.config.to_json_file(os.path.join(output_dir, 'config.json'))

    def prepare_inputs(
        self,
        max_batch_size,
        max_input_len,
        max_seq_len,
        max_num_tokens,
        use_cache,
        max_beam_width: int = 1,
        opt_num_tokens: int = None,
        prompt_embedding_table_size: int = 0,
        position_encoding_2d: bool = False,
        max_draft_len: int = 0,
        speculative_decoding_draft_tokens_external: bool = False,
        spec_decoding_is_generation_length_variable: bool = False,
        gather_context_logits: bool = False,
        lora_target_modules: List[str] = None,
        opt_batch_size: int = 0,
        num_hidden_layers: int = None,
        mrope_rotary_cos_sin_size: int = None,
    ):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''

        # Prepare inputs
        remove_input_padding = default_net().plugin_config.remove_input_padding
        use_gpt_attention_plugin = default_net(
        ).plugin_config.gpt_attention_plugin
        use_gemm_plugin = default_net().plugin_config.gemm_plugin
        paged_kv_cache = default_net().plugin_config.paged_kv_cache
        tokens_per_block = default_net().plugin_config.tokens_per_block
        use_lora_plugin = default_net().plugin_config.lora_plugin
        multiple_profiles = default_net().plugin_config.multiple_profiles
        streamingllm = default_net().plugin_config.streamingllm
        pp_reduce_scatter = default_net().plugin_config.pp_reduce_scatter

        kv_cache_type = None
        if not use_cache:
            kv_cache_type = KVCacheType.DISABLED
        else:
            if paged_kv_cache:
                kv_cache_type = KVCacheType.PAGED
            else:
                kv_cache_type = KVCacheType.CONTINUOUS

        model_inputs = self.prepare_basic_inputs(
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_input_len=max_input_len,
            max_seq_len=max_seq_len,
            hidden_size=self.config.hidden_size,
            num_kv_heads=self.config.num_key_value_heads,
            head_size=self.config.head_size,
            num_layers=num_hidden_layers
            if num_hidden_layers is not None else self.config.num_hidden_layers,
            kv_dtype=str_dtype_to_trt(self.config.kv_dtype),
            remove_input_padding=remove_input_padding,
            use_gpt_attention_plugin=use_gpt_attention_plugin,
            use_gemm_plugin=use_gemm_plugin,
            kv_cache_type=kv_cache_type,
            tokens_per_block=tokens_per_block,
            num_heads=self.config.num_attention_heads,
            max_num_tokens=max_num_tokens,
            opt_num_tokens=opt_num_tokens,
            dtype=str_dtype_to_trt(self.config.dtype),
            prompt_embedding_table_size=prompt_embedding_table_size,
            position_encoding_2d=position_encoding_2d,
            mapping=self.config.mapping,
            gather_context_logits=gather_context_logits,
            use_lora_plugin=use_lora_plugin,
            max_draft_len=max_draft_len,
            speculative_decoding_draft_tokens_external=
            speculative_decoding_draft_tokens_external,
            spec_decoding_is_generation_length_variable=
            spec_decoding_is_generation_length_variable,
            lora_target_modules=lora_target_modules,
            multiple_profiles=multiple_profiles,
            streamingllm=streamingllm,
            opt_batch_size=opt_batch_size,
            pp_reduce_scatter=pp_reduce_scatter,
            mrope_rotary_cos_sin_size=mrope_rotary_cos_sin_size)

        result = {
            'input_ids':
            model_inputs['input_ids'],
            'position_ids':
            model_inputs['position_ids'],
            'use_cache':
            kv_cache_type != KVCacheType.DISABLED,
            'last_token_ids':
            model_inputs['last_token_ids'],
            'attention_mask':
            model_inputs['attention_mask'],
            'kv_cache_params':
            KeyValueCacheParams(
                past_key_value=model_inputs['past_key_value'],
                host_past_key_value_lengths=model_inputs[
                    'host_past_key_value_lengths'],
                host_max_attention_window_sizes=model_inputs[
                    'host_max_attention_window_sizes'],
                host_sink_token_length=model_inputs['host_sink_token_length'],
                kv_cache_block_offsets=model_inputs['kv_cache_block_offsets'],
                host_kv_cache_block_offsets=model_inputs[
                    'host_kv_cache_block_offsets'],
                host_kv_cache_pool_pointers=model_inputs[
                    'host_kv_cache_pool_pointers'],
                host_kv_cache_pool_mapping=model_inputs[
                    'host_kv_cache_pool_mapping'],
                cache_indirection=model_inputs['cache_indirection'],
            ),
            'attention_params':
            AttentionParams(
                sequence_length=model_inputs['sequence_length'],
                context_lengths=model_inputs['context_lengths'],
                host_context_lengths=model_inputs['host_context_lengths'],
                max_context_length=max_input_len,
                host_request_types=model_inputs['host_request_types'],
                host_runtime_perf_knobs=model_inputs['host_runtime_perf_knobs'],
                host_context_progress=model_inputs['host_context_progress'],
            )
        }

        if prompt_embedding_table_size > 0:
            result['prompt_embedding_table'] = model_inputs[
                'prompt_embedding_table']
            result['prompt_tasks'] = model_inputs['tasks']
            result['prompt_vocab_size'] = model_inputs['prompt_vocab_size']
        if model_inputs['hidden_states_input'] is not None:
            result['hidden_states'] = model_inputs['hidden_states_input']
        if use_lora_plugin:
            result['lora_params'] = LoraParams(
                model_inputs['lora_ranks'],
                model_inputs['lora_weights_pointers'],
                host_context_lengths=model_inputs['host_context_lengths'],
                host_request_types=model_inputs['host_request_types'])
        if model_inputs['spec_decoding_params'] is not None:
            result['spec_decoding_params'] = model_inputs[
                'spec_decoding_params']
        if model_inputs['mrope_params'] is not None:
            result['mrope_params'] = model_inputs['mrope_params']

        return result

    @classmethod
    def quantize(
        cls,
        hf_model_dir: str,
        output_dir: str,
        dtype: str = 'auto',
        mapping: Optional[Mapping] = None,
        quant_config: Optional[QuantConfig] = None,
        *,
        device: str = 'cuda',
        calib_dataset: str = 'cnn_dailymail',
        calib_batches: int = 512,
        calib_batch_size: int = 1,
        calib_max_seq_length: int = 512,
        random_seed: int = 1234,
        tokenizer_max_seq_length: int = 2048,
        **kwargs,
    ):
        config_cls = getattr(cls, 'config_class', None)
        if config_cls is None:
            raise NotImplementedError(
                f"{cls.__name__} has not implemented corresponding config class, which is needed for correct config parsing."
            )
        config: PretrainedConfig = config_cls.from_hugging_face(
            hf_model_dir,
            dtype=dtype,
            mapping=mapping,
            quant_config=quant_config,
            **kwargs)
        if config.mapping.moe_ep_size > 1:
            raise NotImplementedError(
                "Quantization for expert parallelism is not supported")
        if not config.quantization._requires_modelopt_quantization:
            raise ValueError(
                f"The quant_config ({quant_config}) should not call modelopt quantization"
            )

        from ..quantization import quantize_and_export
        quantize_and_export(
            model_dir=str(hf_model_dir),
            device=device,
            calib_dataset=calib_dataset,
            dtype=config.dtype,
            qformat=config.quantization._get_modelopt_qformat(),
            kv_cache_dtype=config.quantization._get_modelopt_kv_cache_dtype(),
            calib_size=calib_batches,
            batch_size=calib_batch_size,
            calib_max_seq_length=calib_max_seq_length,
            awq_block_size=config.quantization.group_size,
            output_dir=output_dir,
            tp_size=config.mapping.tp_size,
            pp_size=config.mapping.pp_size,
            cp_size=config.mapping.cp_size,
            seed=random_seed,
            tokenizer_max_seq_length=tokenizer_max_seq_length,
        )


class DecoderModelForCausalLM(PretrainedModel):

    def __init__(self, config: PretrainedConfig, transformer, lm_head):
        super().__init__(config)
        self.transformer = transformer
        self.lm_head = lm_head
        self.mup_width_multiplier = getattr(config, 'mup_width_multiplier',
                                            None)
        # Create constant attention parameters to be reused by all layers.
        Attention.create_attention_const_params(self, config)
        self.position_embedding_type = config.position_embedding_type

    def forward(self,
                input_ids: Tensor,
                position_ids=None,
                use_cache=False,
                last_token_ids=None,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                mrope_params=None,
                hidden_states=None,
                prompt_embedding_table: Optional[Tensor] = None,
                prompt_tasks: Optional[Tensor] = None,
                prompt_vocab_size: Optional[Tensor] = None,
                lora_params=None,
                spec_decoding_params=None):

        # fill attention params.
        attention_params = Attention.fill_attention_params(
            self, attention_params)

        # split the sequence for context parallelism
        if self.config.mapping.cp_size > 1:
            if len(input_ids.shape) == 1:
                # input shape is [-1]
                input_ids, cp_join_index = cp_split_plugin(
                    input_ids,
                    attention_params.host_request_types,
                    attention_params.host_context_lengths,
                    self.config.mapping.cp_size,
                    self.config.mapping.cp_rank,
                )
            else:
                assert False, "Context parallelism with non-remove-padding is not supported yet."

        is_gemma_2_cg = self.config.has_config_group(Gemma2ConfigGroup)
        is_gemma_3_cg = self.config.has_config_group(Gemma3ConfigGroup)

        kwargs = {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'use_cache': use_cache,
            'attention_mask': attention_mask,
            'kv_cache_params': kv_cache_params,
            'attention_params': attention_params,
        }
        if lora_params is not None:
            kwargs['lora_params'] = lora_params
        if hidden_states is not None:
            kwargs['hidden_states'] = hidden_states
        if prompt_embedding_table is not None:
            kwargs['prompt_embedding_table'] = prompt_embedding_table
        if prompt_tasks is not None:
            kwargs['prompt_tasks'] = prompt_tasks
        if prompt_vocab_size is not None:
            kwargs['prompt_vocab_size'] = prompt_vocab_size

        if spec_decoding_params is not None:
            kwargs['spec_decoding_params'] = spec_decoding_params
        if mrope_params is not None:
            kwargs['mrope_params'] = mrope_params

        hidden_states = self.transformer.forward(**kwargs)

        if use_cache:
            hidden_states, presents = hidden_states

        # All gather and rebuild sequence after transformer layer for context parallelism
        if self.config.mapping.cp_size > 1:
            if len(hidden_states.shape) == 2:
                hidden_states = allgather(hidden_states,
                                          self.config.mapping.cp_group,
                                          gather_dim=0)
                hidden_states = view(hidden_states,
                                     [-1, hidden_states.shape[-1]])
                hidden_states = index_select(hidden_states, 0, cp_join_index)
            else:
                assert False, "Context parallelism with non-remove-padding is not supported yet."

        if self.config.mapping.is_last_pp_rank():
            all_hidden_states = hidden_states
            hidden_states = gather_last_token_logits(
                hidden_states, last_token_ids,
                default_net().plugin_config.remove_input_padding)

            # [batch_size, hidden_size] -> [batch_size, vocab_size]
            lm_logits = self.lm_head(hidden_states)
            if hasattr(self.config, 'output_multiplier_scale'):
                lm_logits *= getattr(self.config, 'output_multiplier_scale', 1)
            if self.mup_width_multiplier is not None:
                lm_logits = lm_logits / self.mup_width_multiplier
            if is_gemma_2_cg or is_gemma_3_cg:
                softcap = self.config.get_config_group(
                    Gemma2ConfigGroup if not is_gemma_3_cg else
                    Gemma3ConfigGroup).final_logit_softcapping
                if softcap:
                    lm_logits = lm_logits * float(1 / softcap)
                    lm_logits = tanh(lm_logits) * float(softcap)
            lm_logits.mark_output('logits', self.config.logits_dtype)
        else:
            hidden_states.mark_output('hidden_states_output', self.config.dtype)

        if use_cache and not default_net().plugin_config.paged_kv_cache:
            for i, present in zip(
                    self.config.mapping.pp_layers(
                        self.config.num_hidden_layers), presents):
                present.mark_output(f'present_key_value_{i}',
                                    self.config.kv_dtype)
            if self.config.mapping.is_last_pp_rank():
                return (lm_logits, presents, hidden_states)
            return (hidden_states, presents)
        else:
            if self.config.mapping.is_last_pp_rank():
                return lm_logits, hidden_states, all_hidden_states
            return hidden_states


def fuse_gate_mlp(
    model: PretrainedModel,
    gemm_swiglu_plugin_dtype: Optional[str] = None,
    low_latency_gemm_swiglu_plugin_dtype: Optional[str] = None,
) -> PretrainedModel:
    from ..quantization.quantize import fp8_quantize

    for name, mlp, layer in model.named_modules_with_parent():
        if isinstance(mlp, GatedMLP):
            init_params = get_init_params(mlp)

            hidden_act = init_params["hidden_act"]
            if hidden_act not in ["silu", "gelu"]:
                logger.warning(
                    f"fuse_gate_mlp cannot be done for {name} due to unsupported activation {hidden_act}. Skipping."
                )
                continue

            init_params["inner_layernorm"] = mlp.inner_layernorm is not None
            fused_layer = FusedGatedMLP(**init_params)

            fc_name = name + '.fc'
            layer_quant_cfg = model.config._get_quant_cfg(fc_name)
            layer_quant_algo = layer_quant_cfg.quant_algo
            if layer_quant_algo != QuantAlgo.FP8 and layer_quant_algo is not None:
                continue

            if isinstance(model.config.quantization.exclude_modules, list) \
                    and fc_name in model.config.quantization.exclude_modules:
                layer_quant_algo = None

            if layer_quant_algo == QuantAlgo.FP8:
                fused_layer = fp8_quantize(fused_layer, layer_quant_cfg)

                if isinstance(mlp.dtype, str):
                    dtype = str_dtype_to_torch(mlp.dtype)
                else:
                    dtype = trt_dtype_to_torch(mlp.dtype)

                gate_weight = numpy_to_torch(mlp.gate.weight.raw_value)
                fc_weight = numpy_to_torch(mlp.fc.weight.raw_value)
                assert gate_weight.dtype == fc_weight.dtype
                need_qdq = gate_weight.dtype == torch.float8_e4m3fn

                gate_weight = gate_weight.to(dtype)
                fc_weight = fc_weight.to(dtype)
                # dequantize if needed
                if need_qdq:
                    gate_weight = gate_weight.to(dtype) * numpy_to_torch(
                        mlp.gate.weights_scaling_factor.raw_value)
                    fc_weight = fc_weight.to(dtype) * numpy_to_torch(
                        mlp.fc.weights_scaling_factor.raw_value)

                # concat
                fused_weight = torch.cat([gate_weight, fc_weight], dim=0)

                fused_weight_scaling_factor = numpy_to_torch(
                    max(
                        mlp.gate.weights_scaling_factor.raw_value,
                        mlp.fc.weights_scaling_factor.raw_value,
                    ))
                # quantize if needed
                if need_qdq:
                    fused_weight = (fused_weight /
                                    fused_weight_scaling_factor).to(
                                        torch.float8_e4m3fn)

                if gemm_swiglu_plugin_dtype == 'fp8' or low_latency_gemm_swiglu_plugin_dtype == 'fp8':
                    # gemm_swiglu_plugin needs (k, n) weights
                    # but weights should still be k-major for fp8
                    fused_layer.fused_fc.weight = Parameter(
                        shape=(fused_layer.fused_fc.in_features,
                               fused_layer.fused_fc.out_features),
                        dtype='fp8')
                    fused_layer.fused_fc.weight.value = fused_weight.view(
                        fused_layer.fused_fc.in_features,
                        fused_layer.fused_fc.out_features)
                else:
                    fused_layer.fused_fc.weight.value = fused_weight
                fused_layer.fused_fc.weights_scaling_factor.value = fused_weight_scaling_factor

                fused_layer.fused_fc.activation_scaling_factor.value = max(
                    mlp.gate.activation_scaling_factor.raw_value,
                    mlp.fc.activation_scaling_factor.raw_value,
                )

                if mlp.bias:
                    fused_layer.fused_fc.bias.value = np.concatenate(
                        [mlp.gate.bias.raw_value, mlp.fc.bias.raw_value],
                        axis=0)
            elif layer_quant_algo is None:
                fused_layer.fused_fc.weight.value = np.concatenate(
                    [
                        mlp.gate.weight.raw_value,
                        mlp.fc.weight.raw_value,
                    ],
                    axis=0,
                )
                if mlp.bias:
                    fused_layer.fused_fc.bias.value = np.concatenate(
                        [mlp.gate.bias.raw_value, mlp.fc.bias.raw_value],
                        axis=0)
            else:
                raise ValueError(f'Unsupported quant algo: {layer_quant_algo}')

            fused_layer.proj = mlp.proj
            fused_layer.inner_layernorm = mlp.inner_layernorm

            _, mlp_name = name.rsplit('.', 1)
            setattr(layer, mlp_name, fused_layer)

        elif isinstance(mlp, Fp8RowwiseGatedMLP):
            init_params = get_init_params(mlp)

            hidden_act = init_params["hidden_act"]
            if hidden_act not in ["silu", "gelu"]:
                logger.warning(
                    f"fuse_gate_mlp cannot be done for {name} due to unsupported activation {hidden_act}. Skipping."
                )
                continue

            if mlp.clamp_val is not None:
                init_params["clamp_val"] = mlp.clamp_val.raw_value.tolist()
            fused_layer = Fp8RowwiseFusedGatedMLP(**init_params)
            fused_layer.fused_fc.weight.value = np.concatenate(
                [
                    mlp.gate.weight.raw_value,
                    mlp.fc.weight.raw_value,
                ],
                axis=0,
            )
            fused_layer.fused_fc.per_channel_scale.value = np.concatenate(
                [
                    mlp.gate.per_channel_scale.raw_value,
                    mlp.fc.per_channel_scale.raw_value,
                ],
                axis=0,
            )
            if mlp.bias:
                fused_layer.fused_fc.bias.value = np.concatenate(
                    [mlp.gate.bias.raw_value, mlp.fc.bias.raw_value], axis=0)

            fused_layer.proj = mlp.proj
            _, mlp_name = name.rsplit('.', 1)
            setattr(layer, mlp_name, fused_layer)

    return model


def unfuse_qkv_gemm(model: PretrainedModel) -> PretrainedModel:
    '''Split all the models' Attention layer's QKV GEMM into 3 GEMMs layer.q layer.k, layer.v and return the changed model
    '''
    from ..quantization.quantize import quantize

    for name, layer in model.named_modules():
        if isinstance(layer, Attention) and not layer.cross_attention:
            assert layer.tp_size == 1, "unfuse_qkv_gemm requires tp_size == 1"
            if layer.qkv is None:
                continue
            qkv_params = get_init_params(layer.qkv, ColumnLinear)
            qkv_params["bias"] = qkv_params["bias"] is not None
            qkv_params["strict_dtype"] = qkv_params.get(
                "strict_dtype") is not None
            q = ColumnLinear(
                **{
                    **qkv_params,
                    "out_features":
                    layer.tp_size * layer.num_attention_heads *
                    layer.attention_head_size,
                })
            k = ColumnLinear(
                **{
                    **qkv_params,
                    "out_features":
                    layer.tp_size * layer.num_attention_kv_heads *
                    layer.attention_head_size,
                })
            v = ColumnLinear(
                **{
                    **qkv_params,
                    "out_features":
                    layer.tp_size * layer.num_attention_kv_heads *
                    layer.attention_head_size,
                })
            layer_quant_cfg = model.config._get_quant_cfg(name + '.qkv')
            q = quantize(q, layer_quant_cfg)
            k = quantize(k, layer_quant_cfg)
            v = quantize(v, layer_quant_cfg)
            out_features = q.out_features + k.out_features + v.out_features
            if isinstance(layer.qkv, (
                    WeightOnlyQuantLinear,
                    WeightOnlyQuantRowLinear,
                    WeightOnlyGroupwiseQuantLinear,
                    WeightOnlyGroupwiseQuantRowLinear,
            )):
                out_dim = 1
            else:
                out_dim = 0
            if layer.qkv.weight.is_inited():
                qkv_weight = layer.qkv.weight.raw_value
                weights = np.split(qkv_weight, [
                    qkv_weight.shape[out_dim] * q.out_features // out_features,
                    qkv_weight.shape[out_dim] *
                    (q.out_features + k.out_features) // out_features,
                ],
                                   axis=out_dim)
                for gemm, weight in zip([q, k, v], weights):
                    gemm.weight.value = weight
            if layer.qkv.bias is not None and layer.qkv.bias.is_inited():
                qkv_bias = layer.qkv.bias.raw_value
                biases = np.split(qkv_bias, [
                    qkv_bias.shape[out_dim] * q.out_features // out_features,
                    qkv_bias.shape[out_dim] *
                    (q.out_features + k.out_features) // out_features,
                ],
                                  axis=out_dim)
                for gemm, bias in zip([q, k, v], biases):
                    gemm.bias.value = bias
            for name, parameter in layer.qkv._parameters.items():
                if name not in ["weight", "bias"]:
                    for gemm in [q, k, v]:
                        setattr(gemm, name, parameter)
            layer.q = q
            layer.k = k
            layer.v = v
            layer.qkv = None
    return model


def fuse_rg_lru(model: PretrainedModel) -> PretrainedModel:
    for name, rg_lru, parent in model.named_modules_with_parent():
        if isinstance(rg_lru, RgLru):
            fused_layer = FusedRgLru(**get_init_params(rg_lru))
            fused_layer.gate.weight.value = np.concatenate(
                [
                    rg_lru.input_gate.weight.raw_value,
                    rg_lru.recurrent_gate.weight.raw_value,
                ],
                axis=-1,
            )
            fused_layer.gate.bias.value = np.concatenate(
                [
                    rg_lru.input_gate.bias.raw_value,
                    rg_lru.recurrent_gate.bias.raw_value,
                ],
                axis=-1,
            )
            fused_layer.recurrent_param.value = rg_lru.recurrent_param.raw_value
            rg_lru_name = name.rsplit('.', 1)[-1]
            setattr(parent, rg_lru_name, fused_layer)
    return model


def set_prompt_tuning(model: PretrainedModel) -> PretrainedModel:
    '''Replace the given models embedding layer with a PromptTuningEmbedding layer in-place, return the changed model
       Pre-conditions: vocab_embedding exists
       Post-conditions: isinstance(vocab_embedding, PromptTuningEmbedding)

    '''
    for name, embedding, parent in model.named_modules_with_parent():
        layer_name = name.rsplit('.', 1)[-1]
        if layer_name == "vocab_embedding" and isinstance(embedding, Embedding):
            ptuning_embedding = PromptTuningEmbedding(
                **get_init_params(embedding))
            ptuning_embedding.weight.value = embedding.weight.raw_value
            parent.vocab_embedding = ptuning_embedding
    return model


def add_lora(model: PretrainedModel,
             max_lora_rank: Optional[int],
             with_dora: bool = False) -> PretrainedModel:
    ''' Add lora layers to the Attention/BertAttention/Linear/RowLinear/FusedGatedMLP layers to the given model, return the changed model
    '''
    for name, layer in model.named_modules():
        max_rank = max_lora_rank
        if isinstance(layer, (Attention, BertAttention)):
            if max_rank is None:
                max_rank = min(
                    layer.hidden_size,
                    layer.num_attention_heads * layer.attention_head_size,
                    layer.num_attention_kv_heads * layer.attention_head_size)
            layer.qkv_lora = Lora(
                in_hidden_size=layer.hidden_size,
                out_hidden_sizes=[
                    layer.num_attention_heads * layer.attention_head_size,
                    layer.num_attention_kv_heads * layer.attention_head_size,
                    layer.num_attention_kv_heads * layer.attention_head_size
                ],
                max_low_rank=max_rank,
            )

            if with_dora:
                layer.qkv_dora = Dora(out_hidden_sizes=[
                    layer.num_attention_heads * layer.attention_head_size,
                    layer.num_attention_kv_heads * layer.attention_head_size,
                    layer.num_attention_kv_heads * layer.attention_head_size
                ], )

        if isinstance(layer, (Linear, RowLinear)):
            if max_rank is None:
                max_rank = min(layer.in_features, layer.out_features)
            layer.lora = Lora(
                in_hidden_size=layer.in_features,
                out_hidden_sizes=[layer.out_features],
                max_low_rank=max_rank,
            )
            if with_dora:
                layer.dora = Dora(out_hidden_sizes=[layer.out_features])

        if isinstance(layer, (MLP, FusedGatedMLP)):
            if max_rank is None:
                max_rank = min(layer.hidden_size,
                               layer.ffn_hidden_size // layer.tp_size)
            layer.lora = Lora(
                in_hidden_size=layer.hidden_size,
                out_hidden_sizes=[
                    layer.ffn_hidden_size // layer.tp_size,
                    layer.ffn_hidden_size // layer.tp_size
                ],
                max_low_rank=max_rank,
            )

            if isinstance(layer, FusedGatedMLP):
                layer.fused_gate_up_lora = Lora(
                    in_hidden_size=layer.hidden_size,
                    out_hidden_sizes=[
                        layer.ffn_hidden_size * 2 // layer.tp_size
                    ],
                    max_low_rank=max_rank,
                )

            if with_dora:
                layer.dora = Dora(out_hidden_sizes=[
                    layer.ffn_hidden_size // layer.tp_size,
                    layer.ffn_hidden_size // layer.tp_size
                ], )

                if isinstance(layer, FusedGatedMLP):
                    layer.fused_gate_up_dora = Dora(out_hidden_sizes=[
                        layer.ffn_hidden_size * 2 // layer.tp_size
                    ], )

        if isinstance(layer, MOE):
            if max_rank is None:
                max_rank = min(layer.hidden_size,
                               layer.ffn_hidden_size // layer.tp_size)
            layer.max_low_rank = max_rank
    return model


def to_ootb_moe(model: PretrainedModel) -> PretrainedModel:
    ''' Use OOTB MoE instead of MoE plugin, return the changed model
    '''
    for name, layer, parent in model.named_modules_with_parent():
        if isinstance(layer, MOE):
            layer_name = name.rsplit('.', 1)[-1]
            ootb_layer = layer.to(MoeOOTB, model.config.quantization)
            setattr(parent, layer_name, ootb_layer)
    return model


def parallelize_embedding(model: PretrainedModel) -> PretrainedModel:
    for name, embedding, parent in model.named_modules_with_parent():
        layer_name = name.rsplit('.', 1)[-1]
        if isinstance(embedding, Embedding) and embedding.tp_group is None:
            init_params = get_init_params(embedding)
            init_params["tp_group"] = model.config.mapping.tp_group
            init_params["tp_size"] = model.config.mapping.tp_size
            init_params["tp_rank"] = model.config.mapping.tp_rank
            init_params["sharding_dim"] = model.config.embedding_sharding_dim
            new_embedding = embedding.__class__(**init_params)
            setattr(parent, layer_name, new_embedding)
    return model


def share_embedding(model: PretrainedModel) -> PretrainedModel:
    lm_head = None
    vocab_embedding = None
    for name, layer in model.named_modules():
        layer_name = name.rsplit('.', 1)[-1]
        if layer_name == "lm_head":
            lm_head = layer
        if layer_name == "vocab_embedding":
            vocab_embedding = layer
        if lm_head is not None and vocab_embedding is not None:
            break

    # Cannot find either lm_head or vocab_embedding, e.g., pipeline parallel
    if lm_head is None or vocab_embedding is None:
        return model

    # lm_head and vocab_embedding have different shapes, e.g., tensor parallel without embedding parallel
    if lm_head.weight.shape != vocab_embedding.weight.shape:
        return model

    # lm_head can have a different type if quantized
    if lm_head.weight.dtype != vocab_embedding.weight.dtype:
        return model

    # Don't assume weight can be shared if vocab_embedding is not initialized, e.g., dummy weights
    if not vocab_embedding.weight.is_inited():
        return model

    if lm_head.weight.is_inited():
        lm_head_weight = numpy_to_torch(lm_head.weight.raw_value)
        vocab_embed_weight = numpy_to_torch(vocab_embedding.weight.raw_value)
        # The lm_head and vocab_embedding have different weights
        if (lm_head_weight - vocab_embed_weight).abs().max().item() > 1e-6:
            return model

    lm_head.weight = vocab_embedding.weight
    if getattr(lm_head, 'per_channel_scale', None) and getattr(
            vocab_embedding, 'per_channel_scale', None):
        lm_head.per_channel_scale = vocab_embedding.per_token_scale
    return model


def set_fp8_context_fhma(model: PretrainedModel) -> PretrainedModel:
    for name, layer in model.named_modules():
        if isinstance(layer, Attention) and hasattr(
                layer.dense, 'activation_scaling_factor'):
            scale = [1.0] / layer.dense.activation_scaling_factor.raw_value
            layer.attention_output_orig_quant_scale = Parameter(
                value=scale.astype(np.float32), dtype='float32')
        elif isinstance(layer, Attention) and hasattr(
                layer.dense, 'activation_global_scaling_factor'):
            scale = [1.0
                     ] / layer.dense.activation_global_scaling_factor.raw_value
            layer.attention_output_orig_quant_scale = Parameter(
                value=scale.astype(np.float32), dtype='float32')

    return model


def set_fuse_fp4_quant(model: PretrainedModel) -> PretrainedModel:
    for name, layer in model.named_modules():
        if isinstance(layer, Attention) and hasattr(
                layer.dense, 'activation_global_scaling_factor'):
            scale = [1.0
                     ] / layer.dense.activation_global_scaling_factor.raw_value
            layer.attention_output_sf_scale = Parameter(value=scale.astype(
                np.float32),
                                                        dtype='float32')

    return model


def optimize_model(
    model: PretrainedModel,
    use_parallel_embedding: bool = False,
    share_embedding_table: bool = False,
    use_ootb_moe: bool = False,
    use_fused_mlp: bool = False,
    gemm_swiglu_plugin_dtype: Optional[str] = None,
    low_latency_gemm_swiglu_plugin_dtype: Optional[str] = None,
    use_fused_rg_lru: bool = False,
    use_unfused_qkv_gemm: bool = False,
    use_prompt_tuning: bool = False,
    use_lora: bool = False,
    max_lora_rank: Optional[int] = None,
    use_fp8_context_fmha: bool = False,
    fuse_fp4_quant: bool = False,
    use_optimize_cross_qkv: bool = False,
    use_dora: bool = False,
) -> PretrainedModel:
    """
    Run optimization passes on model.
    There are dependencies between some passes,
    so we always run passes in the order of arguments to guarantee the execution order.
    """
    # before weight loading
    if use_parallel_embedding:
        model = parallelize_embedding(model)

    if share_embedding_table:
        # if share_embedding_table is enabled, only one copy of the embedding table is store in converted ckpt
        # this pass is required to make lm_head.weight and vocab_embedding.weight point to the same tensor
        # however even if share_embedding_table is not enabled, trt would still only keep one copy of the table if the weights are identical
        model = share_embedding(model)

    # After weight loading
    if use_ootb_moe:
        model = to_ootb_moe(model)
    if use_fused_mlp:
        model = fuse_gate_mlp(model, gemm_swiglu_plugin_dtype,
                              low_latency_gemm_swiglu_plugin_dtype)
    if use_fused_rg_lru:
        model = fuse_rg_lru(model)
    if use_unfused_qkv_gemm:
        model = unfuse_qkv_gemm(model)
    if use_prompt_tuning:
        model = set_prompt_tuning(model)
    if use_lora:
        model = add_lora(model, max_lora_rank, with_dora=use_dora)
    if use_fp8_context_fmha:
        model = set_fp8_context_fhma(model)
    if fuse_fp4_quant:
        model = set_fuse_fp4_quant(model)
    if not use_lora and use_optimize_cross_qkv is True:
        # This optimization is not supported when we use lora
        model = optimize_cross_qkv(model)

    return model


def optimize_cross_qkv(model):
    """
    For cross attention layer, we can skip computing the query of encoder_output.
    So, add a new attribute 'kv' in the cross_attention layer. This might lead to
    additional memory cost on model size, but save the memory usage on runtime.

    Currently, this function only detect the ColumnLinear and FP8Linear. It does not supports
    other quantization now.
    """
    for name, attn, layer in model.named_modules_with_parent():
        if isinstance(attn, Attention) and attn.cross_attention and \
        (type(attn.qkv) == ColumnLinear or type(attn.qkv) == FP8Linear):
            old_qkv = attn.qkv
            linear_class = type(old_qkv)
            new_kv = linear_class(
                in_features=attn.hidden_size,
                out_features=2 * attn.tp_size * attn.num_attention_kv_heads *
                attn.attention_head_size,
                bias=old_qkv.bias,
                dtype=old_qkv.dtype,
                tp_group=old_qkv.tp_group,
                tp_size=old_qkv.tp_size,
                gather_output=old_qkv.gather_output,
                prefer_managed_weight=old_qkv.prefer_managed_weight,
                is_qkv=old_qkv.is_qkv,
            )

            old_qkv_weight_value = old_qkv.weight.raw_value
            if (old_qkv_weight_value.shape == np.asarray([
                (attn.num_attention_heads + 2 * attn.num_attention_kv_heads) *
                    attn.attention_head_size, attn.hidden_size
            ])).all():

                q_weight, kv_weight = np.array_split(
                    old_qkv_weight_value.reshape(
                        attn.num_attention_heads +
                        2 * attn.num_attention_kv_heads,
                        attn.attention_head_size, attn.hidden_size),
                    [attn.num_attention_heads],
                    axis=0)
                new_kv.weight.value = kv_weight.reshape([
                    2 * attn.num_attention_kv_heads * attn.attention_head_size,
                    attn.hidden_size
                ])
            elif (old_qkv_weight_value.shape == np.asarray([
                    attn.hidden_size,
                (attn.num_attention_heads + 2 * attn.num_attention_kv_heads) *
                    attn.attention_head_size
            ])).all():
                q_weight, kv_weight = np.array_split(
                    old_qkv_weight_value.reshape(
                        attn.hidden_size, attn.num_attention_heads +
                        2 * attn.num_attention_kv_heads,
                        attn.attention_head_size), [attn.num_attention_heads],
                    axis=1)
                new_kv.weight.value = kv_weight.reshape([
                    attn.hidden_size,
                    2 * attn.num_attention_kv_heads * attn.attention_head_size
                ])
            else:
                assert False

            if isinstance(attn.qkv, FP8Linear):
                new_kv.activation_scaling_factor.value = old_qkv.activation_scaling_factor.raw_value
                new_kv.weights_scaling_factor.value = old_qkv.weights_scaling_factor.raw_value

            if old_qkv.bias:
                q_bias, kv_bias = np.array_split(old_qkv.bias.raw_value.reshape(
                    attn.num_attention_heads + 2 * attn.num_attention_kv_heads,
                    attn.attention_head_size), [attn.num_attention_heads],
                                                 axis=0)
                new_kv.bias.value = kv_bias.reshape([
                    2 * attn.num_attention_kv_heads * attn.attention_head_size
                ])
            setattr(attn, "kv", new_kv)

    return model


def preprocess_perlayer_weights(weights,
                                model_config,
                                quant_algo,
                                from_pruned=False):
    exclude_modules = model_config.quantization.exclude_modules

    # INT4_AWQ
    if quant_algo == QuantAlgo.W4A8_AWQ or quant_algo == QuantAlgo.W4A16_AWQ:
        preprocessor = preprocess_weights_for_mixed_gemm
        if quant_algo == QuantAlgo.W4A8_AWQ:
            activation_type = torch.float8_e4m3fn
        elif quant_algo == QuantAlgo.W4A16_AWQ:
            activation_type = torch.float16
        for name, param in weights.items():
            if from_pruned and param.numel() == 0:
                continue
            if name.endswith('weight') and param.dtype == torch.int8:
                dtype = torch.float16
                if model_config.dtype == "bfloat16":
                    dtype = torch.bfloat16
                weights[name] = preprocessor(param.transpose(-1, -2),
                                             torch.quint4x2,
                                             activation_type).view(dtype)
            if name.endswith('weights_scaling_factor'):
                weights[name] = param.transpose(-1, -2).contiguous().to(
                    str_dtype_to_torch(model_config.dtype))
            if name.endswith('prequant_scaling_factor'):
                if len(weights[name].shape) == 2:
                    # MoE experts share the same scaling factor.
                    param = param[0, :]
                weights[name] = param.reshape(1, -1)
            if model_config.mapping.tp_rank > 0:
                if name.endswith('attention.dense.bias') or name.endswith(
                        'mlp.proj.bias'):
                    weights[name] = torch.zeros_like(param)

        if quant_algo == QuantAlgo.W4A8_AWQ:
            for name in list(weights):
                if name.endswith('weights_scaling_factor'):
                    activation_scaling_factor = weights.pop(
                        name.replace('weights_scaling_factor',
                                     'activation_scaling_factor'))
                    weights_scaling_factor_2 = weights.pop(
                        name.replace('weights_scaling_factor',
                                     'weights_scaling_factor_2'))
                    weights[name] /= weights_scaling_factor_2
                    weights[name] = weights[name].to(torch.float16).view(
                        str_dtype_to_torch(model_config.dtype))
                    weights[name.replace(
                        'weights_scaling_factor',
                        'prequant_scaling_factor')] /= activation_scaling_factor
                    weights[name.replace(
                        'weights_scaling_factor', 'alpha'
                    )] = activation_scaling_factor * weights_scaling_factor_2
                    weights[name.replace('weights_scaling_factor',
                                         'activation_scaling_factor'
                                         )] = activation_scaling_factor

    # FP8
    elif quant_algo == QuantAlgo.FP8:
        for name, param in weights.items():
            if name.endswith('weight') and param.dtype == torch.int8:
                weights[name] = param.view(torch.float8_e4m3fn)
        # lm_head is not always quantized to FP8
        if "lm_head.weight" in weights and weights[
                'lm_head.weight'].dtype is not torch.float8_e4m3fn:
            weights.pop('lm_head.weights_scaling_factor', None)
            weights.pop('lm_head.activation_scaling_factor', None)
    elif quant_algo == QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN:
        for name, param in weights.items():
            if name.endswith('weight') and param.dtype == torch.int8:
                weights[name] = param.view(torch.float8_e4m3fn)
        # lm_head is not quantized to FP8
        if "lm_head.weight" in weights:
            assert weights['lm_head.weight'].dtype == str_dtype_to_torch(
                model_config.dtype)
            weights.pop('lm_head.weights_scaling_factor', None)
            weights.pop('lm_head.activation_scaling_factor', None)
    # FP4
    elif quant_algo == QuantAlgo.NVFP4:
        # Interleave block scale for NVFP4 plugin.
        for name in list(weights):
            if name.endswith('weights_scaling_factor'):
                out_features, in_features = weights[name].shape
                nrows = fp4_utils.pad_up(out_features, 128)
                ncols = fp4_utils.pad_up(in_features, 4)
                new_name = name.replace('weights_scaling_factor',
                                        'weights_block_scaling_factor')
                weights[new_name] = weights[name]
                weights[
                    new_name +
                    "_interleaved"] = torch.ops.trtllm.block_scale_interleave(
                        weights[name].view(fp4_utils.float4_sf_dtype).cpu(
                        ).contiguous()).reshape(nrows, ncols).view(
                            fp4_utils.float4_sf_dtype)
                weights.pop(name)
            if name.endswith('weights_scaling_factor_2'):
                new_name = name.replace('weights_scaling_factor_2',
                                        'weights_global_scaling_factor')
                weights[new_name] = weights[name]
                weights.pop(name)
            if name.endswith('activation_scaling_factor'):
                new_name = name.replace('activation_scaling_factor',
                                        'activation_global_scaling_factor')
                weights[new_name] = weights[name]
                weights.pop(name)
        for name in list(weights):
            if name.endswith('weights_global_scaling_factor'):
                weight_global_sf = weights[name]
                act_global_sf = weights[name.replace(
                    'weights_global_scaling_factor',
                    'activation_global_scaling_factor')]
                weights[name.replace(
                    'weights_global_scaling_factor',
                    'alpha')] = act_global_sf * weight_global_sf
    elif quant_algo in [QuantAlgo.W4A16, QuantAlgo.W8A16]:
        weights = weight_only_quantize_dict(weights=weights,
                                            quant_algo=quant_algo,
                                            exclude_modules=exclude_modules,
                                            plugin=True)


def preprocess_weights(weights: Dict[str, torch.Tensor],
                       model_config: PretrainedConfig,
                       from_pruned=False) -> None:
    """This function in-place modifies weights and model_config, making them compatible with each other.

    Note: Typically, it should be called before model creation and weight loading. For example,
        preprocess_weights(weights, model_config)
        model = XXXForCausalLM(model_config)
        model.load(weights)
    """
    quant_config = model_config.quantization
    quant_algo = quant_config.quant_algo

    pattern_info = ['fc', 'gate', 'proj', 'qkv', 'dense']

    def process_kv_scaling_factor(weights: Dict[str, torch.Tensor]):
        new_entries = {}
        names_to_delete = set()

        # If k, v cache scaling factors are stored separately, combine them into kv cache scaling factor.
        for name, param in weights.items():
            if name.endswith('.k_cache_scaling_factor'):
                v_name = name.replace('k_cache_scaling_factor',
                                      'v_cache_scaling_factor')
                assert v_name in weights, f"{v_name} not found"
                kv_name = name.replace('k_cache_scaling_factor',
                                       'kv_cache_scaling_factor')
                new_entries[kv_name] = torch.max(weights[name], weights[v_name])
                names_to_delete.update([name, v_name])
        weights.update(new_entries)
        for k in names_to_delete:
            del weights[k]

        new_entries = []
        # The unified converter generate_tllm_weights() already generates these rcp weights, but legacy
        # converters do not. Handle it here.
        for name, param in weights.items():
            if name.endswith('.kv_cache_scaling_factor'):
                rcp_name = name.replace('kv_cache_scaling_factor',
                                        'kv_cache_rcp_scaling_factor')
                if rcp_name not in weights:
                    new_entries.append((rcp_name, torch.reciprocal(param)))
        weights.update(new_entries)

    process_kv_scaling_factor(weights)

    per_layer_weights = {}

    for name, param in weights.items():
        in_mode = False
        for info in pattern_info:
            pattern = rf'(.*?{info}.*?)'
            pattern_match = re.match(pattern, name)
            if pattern_match:
                base_name = pattern_match.group(1)
                if base_name not in per_layer_weights.keys():
                    per_layer_weights[base_name] = {}
                per_layer_weights[base_name][name] = param
                in_mode = True
                break
        if not in_mode:
            # [lm_head.weight, ln_f.weight, vocab_embedding.weight]
            base_name = name.rsplit('.', 1)[0]
            if base_name not in per_layer_weights.keys():
                per_layer_weights[base_name] = {}
            per_layer_weights[base_name][name] = param

    new_weights = {}
    for base_name, layer_weights in per_layer_weights.items():
        if quant_algo != QuantAlgo.MIXED_PRECISION:
            layer_quant_algo = quant_algo
        else:
            quant_cfg = quant_config._get_quant_cfg(base_name)
            if not quant_cfg.quant_algo:
                new_weights.update(layer_weights)
                continue

            layer_quant_algo = quant_cfg.quant_algo

        preprocess_perlayer_weights(layer_weights, model_config,
                                    layer_quant_algo, from_pruned)
        new_weights.update(layer_weights)

    weights = new_weights
    for name, param in weights.items():
        if model_config.architecture == 'GPTJForCausalLM':
            if model_config.mapping.tp_rank > 0:
                if 'attention.dense.bias' in name or 'mlp.proj.bias' in name:
                    weights[name] = torch.zeros_like(param)

    return weights


def get_kv_cache_type_from_legacy(use_cache: bool,
                                  paged_kv_cache: bool) -> KVCacheType:
    if use_cache:
        if paged_kv_cache:
            return KVCacheType.PAGED
        else:
            return KVCacheType.CONTINUOUS
    else:
        return KVCacheType.DISABLED


def save_config(config: PretrainedConfig, *, output_dir: str,
                log: bool) -> None:
    config_path = Path(output_dir) / "config.json"
    if log:
        logger.debug(f"Saving TensorRT LLM configuration to {config_path}")
    config_path.parent.mkdir(exist_ok=True, parents=True)
    config_path.write_text(json.dumps(config.to_dict(), indent=4))


def save_checkpoint(*, output_dir: str, weights: dict, rank: int) -> None:
    """ Checkpoint saver for weight loader."""
    safetensors.torch.save_file(
        weights, os.path.join(output_dir, f'rank{rank}.safetensors'))

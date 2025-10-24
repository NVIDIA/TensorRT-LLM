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
import copy
import dataclasses
import json
import math
import os
import shutil
import time
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import tensorrt as trt

from ._common import _is_building, check_max_num_tokens, serialize_engine
from ._utils import (get_sm_version, np_bfloat16, np_float8, str_dtype_to_trt,
                     to_json_file, trt_gte)
from .bindings import KVCacheType
from .functional import PositionEmbeddingType
from .graph_rewriting import optimize
from .logger import logger
from .lora_helper import LoraConfig
from .models import PretrainedConfig, PretrainedModel
from .models.modeling_utils import SpeculativeDecodingMode, optimize_model
from .network import Network, net_guard
from .plugin import PluginConfig
from .quantization import QuantAlgo, QuantMode
from .version import __version__


class ConfigEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, KVCacheType):
            # For KVCacheType, convert it to string by split of 'KVCacheType.PAGED'.
            return obj.__str__().split('.')[-1]
        elif hasattr(obj, 'model_dump'):
            # Handle Pydantic models (including DecodingBaseConfig and subclasses)
            return obj.model_dump(mode='json')
        else:
            return super().default(obj)


class BuilderConfig(object):

    def __init__(self, **kwargs):
        # intentionally use **kwargs, user should never call this ctor directly,
        # use Builder.create_builder_config() instead
        pass

    def _init(self, trt_builder_config, **kwargs):
        self._trt_builder_config = trt_builder_config
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    @property
    def trt_builder_config(self) -> trt.IBuilderConfig:
        return self._trt_builder_config

    def to_dict(self) -> Dict:
        '''return a dict with keys
        {
            "builder_config": {
                # all key values set by the _init function
            },
            "plugin_config": {
                # the network plugin_config (if any) attached to this BuilderConfig object
                # inside the Builder.build_engine
            }
        }
        '''
        config = {'builder_config': {}}
        for k in self.__dict__.keys():
            if k not in ['_trt_builder_config', 'plugin_config']:
                config['builder_config'][k] = self.__getattribute__(k)
        if hasattr(self, 'plugin_config'):
            assert isinstance(self.plugin_config, PluginConfig), \
                f"Found unexpected plugin_config object with type: {type(self.plugin_config)}"
            config['plugin_config'] = self.plugin_config.model_dump(mode="json")
        return config


class Builder():

    _ALLOWED_PRECISIONS = [
        'float32', 'float16', 'bfloat16', trt.DataType.HALF, trt.DataType.FLOAT,
        trt.DataType.BF16
    ]

    def __init__(self):
        super().__init__()
        self._trt_builder = trt.Builder(logger.trt_logger)
        self.strongly_typed = True

    @property
    def trt_builder(self) -> trt.Builder:
        return self._trt_builder

    def create_network(self) -> Network:
        explicit_batch_flag = 0
        # Explicit batch flag will be deprecated in TRT 10
        if "EXPLICIT_BATCH" in trt.NetworkDefinitionCreationFlag.__members__.keys(
        ):
            explicit_batch_flag = 1 << int(
                trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        if self.strongly_typed:
            return Network()._init(
                self.trt_builder.create_network(
                    explicit_batch_flag
                    | (1 << int(
                        trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))))
        else:
            return Network()._init(
                self.trt_builder.create_network(explicit_batch_flag))

    def create_builder_config(self,
                              precision: Union[str, trt.DataType],
                              timing_cache: Union[str, Path,
                                                  trt.ITimingCache] = None,
                              tensor_parallel: int = 1,
                              use_refit: bool = False,
                              int8: bool = False,
                              strongly_typed: bool = True,
                              force_num_profiles: Optional[int] = None,
                              profiling_verbosity: str = "layer_names_only",
                              use_strip_plan: bool = False,
                              weight_streaming: bool = False,
                              precision_constraints: Optional[str] = "obey",
                              **kwargs) -> BuilderConfig:
        ''' @brief Create a builder config with given precisions and timing cache
            @param precision: one of allowed precisions, defined in Builder._ALLOWED_PRECISIONS
            @param timing_cache: a timing cache object or a path to a timing cache file
            @param tensor_parallel: number of GPUs used for tensor parallel
            @param kwargs: any other arguments users would like to attach to the config object as attributes
            @param refit: set to accelerate multi-gpu building, build engine for 1 gpu and refit for the others
            @param int8: whether to build with int8 enabled or not. Can't be used together with refit option
            @return: A BuilderConfig object, return None if failed
        '''
        self.strongly_typed = self.strongly_typed and strongly_typed

        quant_mode = kwargs.get("quant_mode", QuantMode(0))
        if not strongly_typed and precision not in self._ALLOWED_PRECISIONS:
            logger.error(
                f"precision should be one of {self._ALLOWED_PRECISIONS}")

        config = self.trt_builder.create_builder_config()
        if weight_streaming:
            config.set_flag(trt.BuilderFlag.WEIGHT_STREAMING)
        if not self.strongly_typed:
            fp8 = quant_mode.has_fp8_qdq() or quant_mode.has_fp8_kv_cache()
            if precision == 'float16' or precision == trt.DataType.HALF:
                config.set_flag(trt.BuilderFlag.FP16)
                if precision_constraints == 'obey':
                    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            elif precision == 'bfloat16' or precision == trt.DataType.BF16:
                config.set_flag(trt.BuilderFlag.BF16)
                if precision_constraints == 'obey':
                    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            if int8:
                config.set_flag(trt.BuilderFlag.INT8)
            if fp8:
                config.set_flag(trt.BuilderFlag.FP8)
                if precision_constraints == 'obey':
                    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

        if use_refit:
            config.set_flag(trt.BuilderFlag.REFIT)

        # Use fine-grained refit when strip plan is enabled in TRT10.2+.
        if use_strip_plan:
            config.set_flag(trt.BuilderFlag.REFIT_INDIVIDUAL)

        if use_strip_plan:
            config.set_flag(trt.BuilderFlag.STRIP_PLAN)

        # Set TRT Engine profiling verbosity
        if profiling_verbosity == "detailed":
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        elif profiling_verbosity == "none":
            config.profiling_verbosity = trt.ProfilingVerbosity.NONE
        else:
            config.profiling_verbosity = trt.ProfilingVerbosity.LAYER_NAMES_ONLY

        # set timing cache
        cache = None
        if timing_cache is not None:
            # use given cache
            if isinstance(timing_cache, trt.ITimingCache):
                cache = timing_cache
            # read cache from file
            elif isinstance(timing_cache,
                            (str, Path)) and os.path.exists(timing_cache):
                with open(timing_cache, "rb") as f:
                    cache = config.create_timing_cache(f.read())
            else:
                logger.warning(
                    "Invalid timing cache, using freshly created one")
        if cache is None:
            cache = config.create_timing_cache(b"")
        # When user does not given any existing cache, internally always created one
        # so the cache should never None here
        assert cache is not None and isinstance(cache, trt.ITimingCache)
        config.set_timing_cache(cache, ignore_mismatch=False)

        # set weight sparsity
        weight_sparsity = kwargs.get("weight_sparsity", False)
        if weight_sparsity:
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

        # TODO: remove this constraint after trt 10.6 is integrated
        if trt_gte(10, 6):
            # set monitor memory
            monitor_memory = kwargs.get("monitor_memory", False)
            if monitor_memory:
                config.set_flag(trt.BuilderFlag.MONITOR_MEMORY)

        return BuilderConfig()._init(config,
                                     precision=precision,
                                     tensor_parallel=tensor_parallel,
                                     use_refit=use_refit,
                                     int8=int8,
                                     force_num_profiles=force_num_profiles,
                                     strongly_typed=self.strongly_typed,
                                     use_strip_plan=use_strip_plan,
                                     **kwargs)

    def _add_optimization_profile(self, network: Network,
                                  builder_config: BuilderConfig):
        assert isinstance(builder_config, BuilderConfig)
        assert isinstance(network, Network)
        input_tensors = network._inputs
        if len(input_tensors) == 0:
            logger.warning("There are no inputs in the network!")
            return
        num_profiles = len(list(input_tensors.values())[0].profiles)
        force_num_profiles = getattr(builder_config, "force_num_profiles", None)
        for i in range(num_profiles):
            logger.debug(f'Adding optimization profile {i+1}/{num_profiles}')
            profile = self.trt_builder.create_optimization_profile()
            for input_name in input_tensors.keys():
                if len(input_tensors[input_name].profiles) == 0:
                    continue
                shape_profile = input_tensors[input_name].profiles[i]
                min_shape = [*shape_profile.min]
                opt_shape = [*shape_profile.opt]
                max_shape = [*shape_profile.max]
                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                logger.debug(
                    f'{input_name}, min: {min_shape}, opt: {opt_shape}, max: {max_shape}, dimension names: {shape_profile.dimension_names}'
                )
            ret = builder_config.trt_builder_config.add_optimization_profile(
                profile)
            logger.debug(f"Added optimization profile: #{ret}")
            if force_num_profiles is not None and (
                    i + 1
            ) == force_num_profiles and force_num_profiles < num_profiles:
                logger.warning(
                    f"Only adding {force_num_profiles} profiles instead of {num_profiles}."
                )
                break
        assert self._validate_named_dimensions(
            network, builder_config
        ), "Validation of the tensor dimension ranges failed, please check the dimension ranges, find the offensive tensor and dimension name in above the error log"

    def _validate_named_dimensions(self, network: Network,
                                   builder_config) -> bool:
        '''
            For each profile, validate that the named dimensions of different input tensors in this profile all have same range.
            TRT will validate the same condition, validate it earlier to make sure the modeling in TensorRT LLM are correct and
            makes the error msg more user friendly.
        '''
        valid = True
        for profile_idx in range(
                builder_config.trt_builder_config.num_optimization_profiles):
            dimension_to_range = {}
            for input_name, input_tensor in network._inputs.items():
                # it's legal that a Tensor does not have dim_range?
                if len(input_tensor.profiles) != 0:
                    profile = input_tensor.profiles[profile_idx]
                    for dim_idx, dim_name in enumerate(profile.dimension_names):
                        if dim_name not in dimension_to_range:
                            dimension_to_range[dim_name] = []
                        min, opt, max = profile.min[dim_idx], profile.opt[
                            dim_idx], profile.max[dim_idx]
                        dimension_to_range[dim_name].append(
                            (input_name, (min, opt, max)))
            for dim, ranges in dimension_to_range.items():
                unique_ranges = set([r[1] for r in ranges])
                logger.debug(
                    f"Validating dimension:{dim}, ranges for this dim are:{unique_ranges}"
                )
                if len(unique_ranges) != 1:
                    logger.error(
                        f"Found illegal dimension setting for profile {profile_idx}, dimension name is: {dim}"
                    )
                    logger.error(
                        "Offensive tensors which have this dimension are:\n" +
                        "\n".join([f"{r[1]} {dim} {r[0]}" for r in ranges]))
                    valid = False
        return valid

    @_is_building
    def refit_engine(self, network: Network, engine_buffer) -> trt.IHostMemory:
        '''
            @brief: Refit one TensorRT engine using weights from the network,
                user should guarantee that the engine is built with REFIT flag, and the network has the same structure with the engine.
            @param engine_buffer: A serialized TensorRT engine.
            @param network: Network object.
            @return: A serialized TRT engine if refit successfully, None otherwise
        '''
        assert isinstance(network, Network)
        logger.info('Refit TRT engine')
        runtime = trt.Runtime(logger.trt_logger)
        engine = runtime.deserialize_cuda_engine(engine_buffer)

        tik = time.time()

        # Refit engine
        refitter = trt.Refitter(engine, logger.trt_logger)
        if network.named_parameters is not None:
            for name, param in network.named_parameters:
                if param._get_weights(
                ) is None or not refitter.set_named_weights(
                        name, param._get_weights()):
                    logger.error(f'Failed to refit weight: {name}')
                    return None
        else:
            logger.error(
                'Please set named parameters before building multiple engines.')
            return None

        if not refitter.refit_cuda_engine():
            logger.error('Failed to refit engine.')
            return None

        tok = time.time()
        t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
        logger.info(f'Total time of refitting {engine.name}: {t}')
        serialized_engine = engine.serialize()
        return serialized_engine

    @_is_building
    def build_engine(self,
                     network: Network,
                     builder_config: BuilderConfig,
                     managed_weights: dict = None) -> trt.IHostMemory:
        '''
            @brief: Build one TensorRT engine from the network.
            @param network: Network object.
            @param builder_config: BuilderConfig object.
            @return: A serialized TRT engine.
        '''
        assert isinstance(network, Network)
        builder_config.plugin_config = network.plugin_config
        if builder_config.trt_builder_config.num_optimization_profiles == 0:
            self._add_optimization_profile(network, builder_config)
        logger.info(
            f"Total optimization profiles added: {builder_config.trt_builder_config.num_optimization_profiles}"
        )
        engine = None

        tik = time.time()
        # Rename weights
        if network.named_parameters is not None:
            managed_parameters = []
            for name, param in network.named_parameters:
                if param.is_managed(network):
                    assert managed_weights is not None, "managed_weights should be provided when enabled"
                    managed_parameters.append(param)
                    param.set_name(name, network)
                    continue
                if param._get_weights(network) is None:
                    if not param.is_buffer:
                        logger.debug(
                            f"Parameter {name} {param.raw_value.shape} {param.raw_value.dtype} was created"
                            " but unused in forward method, so not materialized to TRT network"
                        )
                    continue
                if not param.set_name(name, network):
                    raise RuntimeError(f'Failed to set weight: {name}')
                # This mark_weights_refittable has no side effect when refit_individual is not enabled.
                network.trt_network.mark_weights_refittable(name)

        network._fill_weights()
        tok = time.time()
        t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
        logger.info(
            f'Total time to initialize the weights in network {network.trt_network.name}: {t}'
        )

        # Build engine
        logger.info(f'Build TensorRT engine {network.trt_network.name}')
        tik = time.time()
        engine = self.trt_builder.build_serialized_network(
            network.trt_network, builder_config.trt_builder_config)
        assert engine is not None, 'Engine building failed, please check the error log.'

        tok = time.time()
        t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
        logger.info(f'Total time of building {network.trt_network.name}: {t}')

        if managed_weights is not None and network.named_parameters is not None:
            for param in managed_parameters:
                name = param.name
                value: np.ndarray = param._value
                if value is None:
                    logger.error(f'Failed to get weight: {name}')
                    continue
                if param.need_transpose:
                    # MOE has ndim=3 and uses plugin, no need to transpose
                    value = value.transpose(1, 0)  # WAR for bug 4641821
                managed_weights[name] = value

        return engine

    @staticmethod
    def save_timing_cache(builder_config: BuilderConfig, out_path: str) -> bool:
        '''Serialize timing cache of given builder config to file specified by out_path
            return True if the cache is successfully serialized, False otherwise
        '''
        cache = builder_config.trt_builder_config.get_timing_cache()
        if cache is None:
            logger.warning(
                'No timing cache found in the given builder config, skip saving.'
            )
            return False
        with cache.serialize() as buffer:
            with open(out_path, "wb") as f:
                f.write(buffer)
                f.flush()
                os.fsync(f)
        logger.info(f'Timing cache serialized to {out_path}')
        return True

    @staticmethod
    def save_config(builder_config: BuilderConfig, config_path: str):
        config = builder_config.to_dict()
        to_json_file(config, config_path)
        logger.info(f'Config saved to {config_path}.')


@dataclass
class BuildConfig:
    """Configuration class for TensorRT LLM engine building parameters.

    This class contains all the configuration parameters needed to build a TensorRT LLM engine,
    including sequence length limits, batch sizes, optimization settings, and various features.

    Args:
        max_input_len (int): Maximum length of input sequences. Defaults to 1024.
        max_seq_len (int, optional): The maximum possible sequence length for a single request, including both input and generated output tokens. Defaults to None.
        opt_batch_size (int): Optimal batch size for engine optimization. Defaults to 8.
        max_batch_size (int): Maximum batch size the engine can handle. Defaults to 2048.
        max_beam_width (int): Maximum beam width for beam search decoding. Defaults to 1.
        max_num_tokens (int): Maximum number of batched input tokens after padding is removed in each batch. Defaults to 8192.
        opt_num_tokens (int, optional): Optimal number of batched input tokens for engine optimization. Defaults to None.
        max_prompt_embedding_table_size (int): Maximum size of prompt embedding table for prompt tuning. Defaults to 0.
        kv_cache_type (KVCacheType, optional): Type of KV cache to use (CONTINUOUS or PAGED). If None, defaults to PAGED. Defaults to None.
        gather_context_logits (int): Whether to gather logits during context phase. Defaults to False.
        gather_generation_logits (int): Whether to gather logits during generation phase. Defaults to False.
        strongly_typed (bool): Whether to use strongly_typed. Defaults to True.
        force_num_profiles (int, optional): Force a specific number of optimization profiles. If None, auto-determined. Defaults to None.
        profiling_verbosity (str): Verbosity level for TensorRT profiling ('layer_names_only', 'detailed', 'none'). Defaults to 'layer_names_only'.
        enable_debug_output (bool): Whether to enable debug output during building. Defaults to False.
        max_draft_len (int): Maximum length of draft tokens for speculative decoding. Defaults to 0.
        speculative_decoding_mode (SpeculativeDecodingMode): Mode for speculative decoding (NONE, MEDUSA, EAGLE, etc.). Defaults to SpeculativeDecodingMode.NONE.
        use_refit (bool): Whether to enable engine refitting capabilities. Defaults to False.
        input_timing_cache (str, optional): Path to input timing cache file. If None, no input cache used. Defaults to None.
        output_timing_cache (str): Path to output timing cache file. Defaults to 'model.cache'.
        lora_config (LoraConfig): Configuration for LoRA (Low-Rank Adaptation) fine-tuning. Defaults to default LoraConfig.
        weight_sparsity (bool): Whether to enable weight sparsity optimization. Defaults to False.
        weight_streaming (bool): Whether to enable weight streaming for large models. Defaults to False.
        plugin_config (PluginConfig): Configuration for TensorRT LLM plugins. Defaults to default PluginConfig.
        use_strip_plan (bool): Whether to use stripped plan for engine building. Defaults to False.
        max_encoder_input_len (int): Maximum encoder input length for encoder-decoder models. Defaults to 1024.
        dry_run (bool): Whether to perform a dry run without actually building the engine. Defaults to False.
        visualize_network (str, optional): Path to save network visualization. If None, no visualization generated. Defaults to None.
        monitor_memory (bool): Whether to monitor memory usage during building. Defaults to False.
        use_mrope (bool): Whether to use Multi-RoPE (Rotary Position Embedding) optimization. Defaults to False.
    """
    max_input_len: int = 1024
    max_seq_len: int = None
    opt_batch_size: int = 8
    max_batch_size: int = 2048
    max_beam_width: int = 1
    max_num_tokens: int = 8192
    opt_num_tokens: Optional[int] = None
    max_prompt_embedding_table_size: int = 0
    kv_cache_type: KVCacheType = None
    gather_context_logits: int = False
    gather_generation_logits: int = False
    strongly_typed: bool = True
    force_num_profiles: Optional[int] = None
    profiling_verbosity: str = 'layer_names_only'
    enable_debug_output: bool = False
    max_draft_len: int = 0
    speculative_decoding_mode: SpeculativeDecodingMode = SpeculativeDecodingMode.NONE
    use_refit: bool = False
    input_timing_cache: str = None
    output_timing_cache: str = 'model.cache'
    lora_config: LoraConfig = field(default_factory=LoraConfig)
    weight_sparsity: bool = False
    weight_streaming: bool = False
    plugin_config: PluginConfig = field(default_factory=PluginConfig)
    use_strip_plan: bool = False
    max_encoder_input_len: int = 1024  # for enc-dec DecoderModel
    dry_run: bool = False
    visualize_network: str = None
    monitor_memory: bool = False
    use_mrope: bool = False

    # Since we have some overlapping between kv_cache_type, paged_kv_cache, and paged_state (later two will be deprecated in the future),
    # we need to handle it given model architecture.
    def update_kv_cache_type(self, model_architecture: str):
        paged_kv_cache_attr = 'paged_state' if model_architecture in [
            'MambaForCausalLM', 'RecurrentGemmaForCausalLM'
        ] else 'paged_kv_cache'
        assert self.plugin_config is not None
        paged_kv_cache_val = getattr(self.plugin_config, paged_kv_cache_attr)

        if self.kv_cache_type is not None:
            if paged_kv_cache_val is not None:
                assert (paged_kv_cache_val == True
                        and self.kv_cache_type == KVCacheType.PAGED) or (
                            paged_kv_cache_val == False
                            and self.kv_cache_type != KVCacheType.PAGED)
            else:
                setattr(self.plugin_config, paged_kv_cache_attr,
                        self.kv_cache_type == KVCacheType.PAGED)
        else:
            if paged_kv_cache_val is not None:
                self.kv_cache_type = KVCacheType.PAGED if paged_kv_cache_val else KVCacheType.CONTINUOUS
            else:
                self.kv_cache_type = KVCacheType.PAGED
                setattr(self.plugin_config, paged_kv_cache_attr,
                        self.kv_cache_type == KVCacheType.PAGED)

        assert self.kv_cache_type is not None and getattr(
            self.plugin_config, paged_kv_cache_attr) is not None

        def override_attri(attr_name, value):
            val = getattr(self.plugin_config, attr_name)
            if val is not None and val != value:
                logger.warning(f'Overriding {attr_name} to {value}')
            setattr(self.plugin_config, attr_name, value)

        # Init other paged kvcache attri to false. For RecurrentGemma, we only support paged_state and paged_kv_cache have
        # the same values. All other models should only consume either of the value and set other to False.
        is_recurrent_gemma = model_architecture == 'RecurrentGemmaForCausalLM'

        if paged_kv_cache_attr == 'paged_state':
            override_attri(
                'paged_kv_cache',
                getattr(self.plugin_config, paged_kv_cache_attr)
                if is_recurrent_gemma else False)
        else:
            override_attri('paged_state', False)

    @classmethod
    @cache
    def get_build_config_defaults(cls):
        return {
            field.name: field.default
            for field in dataclasses.fields(cls)
            if field.default is not dataclasses.MISSING
        }

    @classmethod
    def from_dict(cls, config, plugin_config=None):
        config = copy.deepcopy(
            config
        )  # it just does not make sense to change the input arg `config`

        defaults = cls.get_build_config_defaults()
        max_input_len = config.pop('max_input_len',
                                   defaults.get('max_input_len'))
        max_seq_len = config.pop('max_seq_len', defaults.get('max_seq_len'))
        max_batch_size = config.pop('max_batch_size',
                                    defaults.get('max_batch_size'))
        max_beam_width = config.pop('max_beam_width',
                                    defaults.get('max_beam_width'))
        max_num_tokens = config.pop('max_num_tokens',
                                    defaults.get('max_num_tokens'))
        opt_num_tokens = config.pop('opt_num_tokens',
                                    defaults.get('opt_num_tokens'))
        opt_batch_size = config.pop('opt_batch_size',
                                    defaults.get('opt_batch_size'))
        max_prompt_embedding_table_size = config.pop(
            'max_prompt_embedding_table_size',
            defaults.get('max_prompt_embedding_table_size'))

        if "kv_cache_type" in config and config["kv_cache_type"] is not None:
            kv_cache_type = KVCacheType.from_string(config.pop('kv_cache_type'))
        else:
            kv_cache_type = None
        gather_context_logits = config.pop(
            'gather_context_logits', defaults.get('gather_context_logits'))
        gather_generation_logits = config.pop(
            'gather_generation_logits',
            defaults.get('gather_generation_logits'))
        strongly_typed = config.pop('strongly_typed',
                                    defaults.get('strongly_typed'))
        force_num_profiles = config.pop('force_num_profiles',
                                        defaults.get('force_num_profiles'))
        weight_sparsity = config.pop('weight_sparsity',
                                     defaults.get('weight_sparsity'))
        profiling_verbosity = config.pop('profiling_verbosity',
                                         defaults.get('profiling_verbosity'))
        enable_debug_output = config.pop('enable_debug_output',
                                         defaults.get('enable_debug_output'))
        max_draft_len = config.pop('max_draft_len',
                                   defaults.get('max_draft_len'))
        speculative_decoding_mode = config.pop(
            'speculative_decoding_mode',
            defaults.get('speculative_decoding_mode'))
        use_refit = config.pop('use_refit', defaults.get('use_refit'))
        input_timing_cache = config.pop('input_timing_cache',
                                        defaults.get('input_timing_cache'))
        output_timing_cache = config.pop('output_timing_cache',
                                         defaults.get('output_timing_cache'))
        lora_config = LoraConfig(**config.get('lora_config', {}))
        max_encoder_input_len = config.pop(
            'max_encoder_input_len', defaults.get('max_encoder_input_len'))
        weight_streaming = config.pop('weight_streaming',
                                      defaults.get('weight_streaming'))
        use_strip_plan = config.pop('use_strip_plan',
                                    defaults.get('use_strip_plan'))

        if plugin_config is None:
            plugin_config = PluginConfig()
        if "plugin_config" in config.keys():
            plugin_config = plugin_config.model_copy(
                update=config["plugin_config"], deep=True)

        dry_run = config.pop('dry_run', defaults.get('dry_run'))
        visualize_network = config.pop('visualize_network',
                                       defaults.get('visualize_network'))
        monitor_memory = config.pop('monitor_memory',
                                    defaults.get('monitor_memory'))
        use_mrope = config.pop('use_mrope', defaults.get('use_mrope'))

        return cls(
            max_input_len=max_input_len,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_num_tokens=max_num_tokens,
            opt_num_tokens=opt_num_tokens,
            opt_batch_size=opt_batch_size,
            max_prompt_embedding_table_size=max_prompt_embedding_table_size,
            kv_cache_type=kv_cache_type,
            gather_context_logits=gather_context_logits,
            gather_generation_logits=gather_generation_logits,
            strongly_typed=strongly_typed,
            force_num_profiles=force_num_profiles,
            profiling_verbosity=profiling_verbosity,
            enable_debug_output=enable_debug_output,
            max_draft_len=max_draft_len,
            speculative_decoding_mode=speculative_decoding_mode,
            use_refit=use_refit,
            input_timing_cache=input_timing_cache,
            output_timing_cache=output_timing_cache,
            lora_config=lora_config,
            use_strip_plan=use_strip_plan,
            max_encoder_input_len=max_encoder_input_len,
            weight_sparsity=weight_sparsity,
            weight_streaming=weight_streaming,
            plugin_config=plugin_config,
            dry_run=dry_run,
            visualize_network=visualize_network,
            monitor_memory=monitor_memory,
            use_mrope=use_mrope)

    @classmethod
    def from_json_file(cls, config_file, plugin_config=None):
        with open(config_file) as f:
            config = json.load(f)
            return BuildConfig.from_dict(config, plugin_config=plugin_config)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        # the enum KVCacheType cannot be converted automatically
        if output.get('kv_cache_type', None) is not None:
            output['kv_cache_type'] = str(output['kv_cache_type'].name)
        output['plugin_config'] = output['plugin_config'].model_dump()
        output['lora_config'] = output['lora_config'].model_dump()
        return output

    def update_from_dict(self, config: dict):
        for name, value in config.items():
            if not hasattr(self, name):
                raise AttributeError(
                    f"{self.__class__} object has no attribute {name}")
            setattr(self, name, value)

    def update(self, **kwargs):
        self.update_from_dict(kwargs)


class EngineConfig:

    def __init__(self, pretrained_config: 'PretrainedConfig',
                 build_config: 'BuildConfig', version: str):
        self.pretrained_config = pretrained_config
        self.build_config = build_config
        self.version = version

    @classmethod
    def from_json_file(cls, config_file):
        with open(config_file) as f:
            return cls.from_json_str(f.read())

    @classmethod
    def from_json_str(cls, config_str):
        config = json.loads(config_str)
        return cls(PretrainedConfig.from_dict(config['pretrained_config']),
                   BuildConfig.from_dict(config['build_config']),
                   config['version'])

    def to_dict(self):
        build_config = self.build_config.to_dict()
        build_config.pop('dry_run', None)  # Not an Engine Characteristic
        build_config.pop('visualize_network',
                         None)  # Not an Engine Characteristic
        return {
            'version': self.version,
            'pretrained_config': self.pretrained_config.to_dict(),
            'build_config': build_config,
        }


class Engine:

    def __init__(
        self,
        config: EngineConfig,
        engine: Union[trt.IHostMemory, None],
        managed_weights: dict[str, np.ndarray] = {},
    ):
        self.config = config
        self.engine = engine
        self.managed_weights = managed_weights
        if self.managed_weights is None:
            self.managed_weights = {}
        for name, value in self.managed_weights.items():
            if not value.flags['C_CONTIGUOUS']:
                self.managed_weights[name] = np.ascontiguousarray(value)

    def save(self, engine_dir: str):
        os.makedirs(engine_dir, exist_ok=True)
        lora_config = self.config.build_config.lora_config
        lora_dirs = lora_config.lora_dir
        root_lora_dir = os.path.join(engine_dir, 'lora')
        if len(lora_dirs) > 0:
            os.makedirs(root_lora_dir, exist_ok=True)
            for index, lora_dir in enumerate(lora_dirs):
                if lora_config.lora_ckpt_source == 'hf':
                    target_lora_dir = f"{root_lora_dir}/{index}"
                    os.makedirs(target_lora_dir, exist_ok=True)
                    shutil.copy2(os.path.join(lora_dir, 'adapter_config.json'),
                                 target_lora_dir)
                    weight_file = os.path.join(lora_dir, 'adapter_model.bin')
                    if os.path.exists(weight_file):
                        shutil.copy2(weight_file, target_lora_dir)
                    weight_file = os.path.join(lora_dir,
                                               'adapter_model.safetensors')
                    if os.path.exists(weight_file):
                        shutil.copy2(weight_file, target_lora_dir)
                    lora_config.lora_dir[index] = f"lora/{index}"
                elif lora_config.lora_ckpt_source == 'nemo':
                    target_lora_file = f"{root_lora_dir}/{index}.nemo"
                    shutil.copyfile(lora_dir, target_lora_file)
                    lora_config.lora_dir[index] = f"lora/{index}.nemo"
        else:
            if os.path.exists(root_lora_dir) and os.path.isdir(root_lora_dir):
                shutil.rmtree(root_lora_dir)
        if self.config.pretrained_config.mapping.rank == 0:
            config_dict = self.config.to_dict()
            if self.config.pretrained_config.quant_algo == QuantAlgo.MIXED_PRECISION:
                quant_dict = {
                    'version': self.config.version,
                }
                quant_dict.update(
                    config_dict['pretrained_config']['quantization'])
                config_dict['pretrained_config']['quantization'].pop(
                    'quantized_layers', None)
                with open(os.path.join(engine_dir, 'quant_cfg.json'),
                          "w",
                          encoding="utf-8") as f:
                    json.dump(quant_dict, f, indent=4, cls=ConfigEncoder)

            with open(os.path.join(engine_dir, 'config.json'),
                      "w",
                      encoding="utf-8") as f:
                json.dump(config_dict, f, indent=4, cls=ConfigEncoder)
        if self.engine is not None:
            serialize_engine(
                self.engine,
                os.path.join(
                    engine_dir,
                    f'rank{self.config.pretrained_config.mapping.rank}.engine'))
        if self.managed_weights is not None and len(self.managed_weights) > 0:
            fn = os.path.join(
                engine_dir,
                f'rank{self.config.pretrained_config.mapping.rank}_managed_weights.safetensors'
            )
            serialize_managed_weights(self.managed_weights, fn)

    @classmethod
    def from_dir(cls, engine_dir: str, rank: int = 0):
        with open(os.path.join(engine_dir, f'rank{rank}.engine'), 'rb') as f:
            engine_buffer = f.read()

        mw_path = os.path.join(engine_dir,
                               f'rank{rank}_managed_weights.safetensors')
        managed_weights = deserialize_managed_weights(
            mw_path) if os.path.exists(mw_path) else None

        config = EngineConfig.from_json_file(
            os.path.join(engine_dir, 'config.json'))
        config.pretrained_config.set_rank(rank)

        return cls(config, engine_buffer, managed_weights)

    @classmethod
    def from_buffer(cls,
                    engine_buffer: Union[trt.IHostMemory, bytes],
                    json_config_str: str,
                    rank: int = 0):
        config = EngineConfig.from_json_str(json_config_str)
        config.pretrained_config.set_rank(rank)
        return cls(config, engine_buffer)


def get_engine_version(engine_dir: str) -> Union[None, str]:
    engine_dir = Path(engine_dir)
    config_path = engine_dir / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    if 'version' not in config:
        return None

    return config['version']


def optimize_model_with_config(model: PretrainedModel,
                               build_config: BuildConfig):
    gemm_swiglu_plugin = build_config.plugin_config.gemm_swiglu_plugin
    low_latency_gemm_swiglu_plugin = build_config.plugin_config.low_latency_gemm_swiglu_plugin
    if gemm_swiglu_plugin or low_latency_gemm_swiglu_plugin:
        if not build_config.plugin_config.use_fused_mlp:
            raise RuntimeError(
                "GemmSwiGLU plugin requires --use_fused_mlp flag")
        if gemm_swiglu_plugin not in [
                "fp8"
        ] and low_latency_gemm_swiglu_plugin not in ["fp8"]:
            raise RuntimeError(
                f"GemmSwiGLU plugin currently has limited support: fp8 only, "
                f"got: {gemm_swiglu_plugin}"
                f"got: {low_latency_gemm_swiglu_plugin}")

    if build_config.plugin_config.lora_plugin is not None:
        model.use_lora(build_config.lora_config)

    is_enc_dec = model.config.architecture in ["EncoderModel", "DecoderModel"]
    # FusedMLP does not support RecurrentGemma FP8 currently.
    is_recurrent_gemma = model.config.architecture in [
        "RecurrentGemmaForCausalLM"
    ]
    is_fp8 = model.config.quantization.quant_algo == QuantAlgo.FP8
    model = optimize_model(
        model,
        share_embedding_table=True,
        use_ootb_moe=build_config.plugin_config.moe_plugin is None,
        use_fused_mlp=(build_config.plugin_config.use_fused_mlp
                       and not is_enc_dec
                       and not (is_recurrent_gemma and is_fp8)),
        gemm_swiglu_plugin_dtype=gemm_swiglu_plugin,
        low_latency_gemm_swiglu_plugin_dtype=low_latency_gemm_swiglu_plugin,
        use_fused_rg_lru=is_recurrent_gemma,
        use_unfused_qkv_gemm=False,
        use_prompt_tuning=(build_config.max_prompt_embedding_table_size > 0),
        use_lora=build_config.plugin_config.lora_plugin is not None,
        max_lora_rank=build_config.lora_config.max_lora_rank,
        use_fp8_context_fmha=(model.config.quantization.quant_algo in [
            QuantAlgo.FP8, QuantAlgo.W4A8_AWQ, QuantAlgo.NVFP4
        ] and build_config.plugin_config.use_fp8_context_fmha),
        fuse_fp4_quant=build_config.plugin_config.fuse_fp4_quant,
        use_optimize_cross_qkv=True,
        use_dora=build_config.plugin_config.dora_plugin)

    if is_enc_dec:
        model.precompute_relative_attention_bias(build_config)
    return model


def _init_max_seq_len(model_config, build_config):
    """
    If max_seq_len is not specified, set it to max_position_embeddings * rotary_factor
    Additional checks to ensure max_seq_len, max_input_len, and max_num_tokens have valid values.
    """
    # Extract rotary scaling which will be used for checks and default value of max_seq_len
    rotary_scaling = getattr(model_config, "rotary_scaling", None)
    if rotary_scaling is not None:
        rotary_type = rotary_scaling.get('type',
                                         rotary_scaling.get('rope_type'))
        rotary_factor = rotary_scaling.get(
            'factor', 1.0) if rotary_type not in ("su", "longrope",
                                                  "llama3") else 1
    else:
        rotary_factor = 1

    if model_config.architecture == "EncoderModel":
        if build_config.max_seq_len is None:
            build_config.max_seq_len = build_config.max_input_len
            logger.info(
                f'max_seq_len is not specified for EncoderModel, using --max_input_len.'
            )
        assert build_config.max_input_len == build_config.max_seq_len, f"EncoderModel should have same --max_input_len ({build_config.max_input_len}) and --max_seq_len ({build_config.max_seq_len})."

    if build_config.max_seq_len is None:
        # Step 1: Find the upper bound of max_seq_len
        deduced_max_seq_len = 2048
        if model_config.max_position_embeddings is not None:
            deduced_max_seq_len = model_config.max_position_embeddings

        # Step 2: Scale max_seq_len with rotary scaling
        if rotary_factor != 1:
            deduced_max_seq_len = math.ceil(deduced_max_seq_len * rotary_factor)
            logger.warning(
                f'max_seq_len is scaled to {deduced_max_seq_len} by rotary scaling {rotary_factor}'
            )

        # Step 3: Assign the new max_seq_len
        build_config.max_seq_len = int(deduced_max_seq_len)
        logger.info(
            f'max_seq_len is not specified, using deduced value {deduced_max_seq_len}'
        )
    else:
        if not build_config.plugin_config.streamingllm and model_config.max_position_embeddings is not None \
            and model_config.position_embedding_type != PositionEmbeddingType.relative:
            if build_config.max_seq_len > model_config.max_position_embeddings * rotary_factor:
                logger.warning(
                    f'max_seq_len {build_config.max_seq_len} is larger than max_position_embeddings {model_config.max_position_embeddings} * rotary scaling {rotary_factor}, '
                    'the model accuracy might be affected')

    if build_config.max_input_len > build_config.max_seq_len:
        logger.warning(
            f'max_input_len is {build_config.max_input_len} is larger than max_seq_len {build_config.max_seq_len}, clipping it to max_seq_len'
        )
        build_config.max_input_len = build_config.max_seq_len

    # Check and may modify max_num_tokens and opt_num_tokens (need to happen after max_seq_len is deduced)
    max_num_tokens, opt_num_tokens = check_max_num_tokens(
        max_num_tokens=build_config.max_num_tokens,
        opt_num_tokens=build_config.opt_num_tokens,
        max_batch_size=build_config.max_batch_size,
        max_input_len=build_config.max_input_len,
        max_seq_len=build_config.max_seq_len,
        max_beam_width=build_config.max_beam_width,
        remove_input_padding=build_config.plugin_config.remove_input_padding,
        enable_context_fmha=build_config.plugin_config.context_fmha,
        tokens_per_block=build_config.plugin_config.tokens_per_block,
        multiple_profiles=build_config.plugin_config.multiple_profiles,
    )
    build_config.max_num_tokens, build_config.opt_num_tokens = max_num_tokens, opt_num_tokens

    if build_config.plugin_config.remove_input_padding and build_config.plugin_config.context_fmha:
        if build_config.max_input_len:
            logger.warning(
                'padding removal and fMHA are both enabled, max_input_len is not required and will be ignored'
            )
    else:
        assert build_config.max_input_len is not None, 'padding removal and fMHA aren\'t both enabled, max_input_len is required'
        if build_config.max_seq_len:
            assert build_config.max_input_len <= build_config.max_seq_len, 'max_input_len should not be larger than max_seq_len'


def serialize_managed_weights(managed_weights: dict[str, np.ndarray],
                              path: str | Path,
                              metadata=None) -> None:
    header = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    begin = 0
    for name, value in managed_weights.items():
        size = value.size * value.itemsize
        if value.dtype == np.float32:
            dtype = "F32"
        elif value.dtype == np.float16:
            dtype = "F16"
        elif value.dtype == np_bfloat16:
            dtype = "BF16"
        elif value.dtype == np_float8:
            dtype = "F8_E4M3"
        elif value.dtype == np.int64:
            dtype = "I64"
        elif value.dtype == np.int32:
            dtype = "I32"
        elif value.dtype == np.int8:
            dtype = "I8"
        else:
            raise RuntimeError(f"Unsupported dtype: {value.dtype}")
        header[name] = {
            "dtype": dtype,
            "shape": value.shape,
            "data_offsets": [begin, begin + size],
        }
        begin += size

    header_json = json.dumps(header)
    header_json_len = len(header_json)
    with open(path, "wb") as f:
        logger.info(
            f"Serializing {len(managed_weights)} managed weights to {path}...")
        f.write(header_json_len.to_bytes(8, byteorder="little"))
        f.write(header_json.encode())
        for name, value in managed_weights.items():
            logger.debug(f"Serializing managed weight: {name}")
            buf = value.data
            f.write(buf)


def deserialize_managed_weights(path: str | Path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        header_json_len = int.from_bytes(f.read(8), byteorder="little")
        header_json = f.read(header_json_len).decode()
        header = json.loads(header_json)

        managed_weights = {}
        for name, info in header.items():
            dtype = info["dtype"]
            shape = info["shape"]
            data_offsets = info["data_offsets"]
            if dtype == "F32":
                dtype = np.float32
            elif dtype == "F16":
                dtype = np.float16
            elif dtype == "BF16":
                dtype = np_bfloat16
            elif dtype == "F8_E4M3":
                dtype = np_float8
            elif dtype == "I64":
                dtype = np.int64
            elif dtype == "I32":
                dtype = np.int32
            else:
                raise RuntimeError(f"Unsupported dtype: {dtype}")

            f.seek(data_offsets[0] + header_json_len + 8)
            buf = f.read(data_offsets[1] - data_offsets[0])
            value = np.frombuffer(buf, dtype=dtype).reshape(shape)
            managed_weights[name] = value

    return managed_weights


def build(model: PretrainedModel, build_config: BuildConfig) -> Engine:
    '''Build engine from given model and optimization options specified in the build_config
       WARNING: this function may change the given model object state in some optimization passes
       to avoid cloning a model since normally the LLM models consumes large memory.
       Create a new fresh model object if you need to build with different options.
    '''
    tic = time.time()
    # avoid changing the input config
    build_config = copy.deepcopy(build_config)
    build_config.plugin_config.dtype = model.config.dtype
    build_config.update_kv_cache_type(model.config.architecture)

    _init_max_seq_len(model.config, build_config)

    if build_config.plugin_config.streamingllm:
        build_config.plugin_config.use_paged_context_fmha = False
        logger.warning(
            "Paged Context FMHA is disabled because StreamingLLM is not supported when enabling paged KV context FMHA."
        )
    if build_config.plugin_config.reduce_fusion and (
            model.config.mapping.tp_size == 1 or
        (model.config.architecture != "LlamaForCausalLM"
         and model.config.architecture != "Gemma2ForCausalLM"
         and model.config.architecture != "MedusaForCausalLM")):
        logger.warning('Overriding reduce_fusion to False')
        build_config.plugin_config.reduce_fusion = False
    if build_config.plugin_config.user_buffer and not build_config.plugin_config.reduce_fusion:
        logger.warning('Overriding user_buffer to False')
        build_config.plugin_config.user_buffer = False
    if build_config.plugin_config.norm_quant_fusion and (
            build_config.plugin_config.reduce_fusion
            or model.config.architecture != "LlamaForCausalLM"
            or model.config.quantization.quant_algo != QuantAlgo.NVFP4):
        logger.warning('Overriding norm_quant_fusion to False')
        build_config.plugin_config.norm_quant_fusion = False

    if model.config.quantization.quant_algo == QuantAlgo.FP8 or \
            model.config.quantization.kv_cache_quant_algo == QuantAlgo.FP8:
        build_config.strongly_typed = True

    if hasattr(model.config, 'max_draft_len'):
        # If model.config has 'max_draft_len' but build_config not specified,
        # use the value of model.config.max_draft_len to set the value of build_config.max_draft_len
        if build_config.max_draft_len == 0:
            build_config.max_draft_len = model.config.max_draft_len

    if hasattr(model.config, 'redrafter_num_beams') and hasattr(
            model.config, 'redrafter_draft_len_per_beam'):
        build_config.max_draft_len = model.config.redrafter_num_beams * model.config.redrafter_draft_len_per_beam
        if build_config.speculative_decoding_mode != SpeculativeDecodingMode.EXPLICIT_DRAFT_TOKENS:
            logger.warning(
                'speculative_decoding_mode is not EXPLICIT_DRAFT_TOKENS for ReDrafter model. Overwriting speculative_decoding_mode'
            )
        build_config.speculative_decoding_mode = SpeculativeDecodingMode.EXPLICIT_DRAFT_TOKENS

    if build_config.speculative_decoding_mode != SpeculativeDecodingMode.NONE:
        logger.info(
            f'Increasing max_seq_len ({build_config.max_seq_len}) '
            f'by max_draft_len ({build_config.max_draft_len}) '
            'to account for speculative decoding implementation specifics. '
            'Maximum number of generated tokens remains the same. '
            f'New max_seq_len is set to {build_config.max_seq_len + build_config.max_draft_len}'
        )
        build_config.max_seq_len += build_config.max_draft_len

    if build_config.speculative_decoding_mode == SpeculativeDecodingMode.EAGLE:
        assert hasattr(model.config, 'num_eagle_layers')
        num_eagle_layers = model.config.num_eagle_layers
        logger.info(
            f'Increasing max_seq_len ({build_config.max_seq_len}) '
            f'by num_eagle_layers ({num_eagle_layers}) '
            'to account for EAGLE implementation specifics. '
            'Maximum number of generated tokens remains the same. '
            f'New max_seq_len is set to {build_config.max_seq_len + num_eagle_layers}'
        )
        build_config.max_seq_len += num_eagle_layers

    if build_config.speculative_decoding_mode != SpeculativeDecodingMode.NONE:
        num_tokens = build_config.max_batch_size * (build_config.max_draft_len +
                                                    1)
        if build_config.max_num_tokens < num_tokens:
            logger.info(
                f'max_num_tokens ({build_config.max_num_tokens}) is smaller than '
                'max_batch_size * (max_draft_len + 1) = '
                f'({build_config.max_batch_size} * ({build_config.max_draft_len} + 1)). '
                f'New max_num_tokens is set to {num_tokens}.')
            build_config.max_num_tokens = num_tokens

    # Logics to control paged_context_fmha and fp8_context_fmha
    if not build_config.plugin_config.context_fmha:
        build_config.plugin_config.use_fp8_context_fmha = False
        build_config.plugin_config.use_paged_context_fmha = False
        logger.warning(
            "Context FMHA is disabled, FP8 Context FMHA and Paged Context FMHA are disabled."
        )
    elif not model.config.quantization.quant_algo in [
            QuantAlgo.FP8, QuantAlgo.W4A8_AWQ, QuantAlgo.NVFP4
    ]:
        if build_config.plugin_config.use_fp8_context_fmha:
            build_config.plugin_config.use_fp8_context_fmha = False
            logger.warning(
                "FP8 Context FMHA is disabled because it must be used together with the fp8 quantization workflow."
            )
        if build_config.plugin_config.use_paged_context_fmha and model.config.quant_mode.has_fp8_kv_cache(
        ):
            build_config.plugin_config.use_paged_context_fmha = False
            logger.warning(
                "FP8 Paged Context FMHA is disabled because FP8 context FMHA is disabled."
            )
    elif get_sm_version() < 89:
        build_config.plugin_config.use_fp8_context_fmha = False
        logger.warning(
            "FP8 context FMHA is disabled because it is only supported on Ada and Hopper Arch."
        )
        if build_config.plugin_config.use_paged_context_fmha and model.config.quant_mode.has_fp8_kv_cache(
        ):
            build_config.plugin_config.use_paged_context_fmha = False
            logger.warning(
                "FP8 Paged Context FMHA is disabled because FP8 context FMHA is disabled."
            )
    elif build_config.plugin_config.use_paged_context_fmha:
        if not model.config.quant_mode.has_fp8_kv_cache(
        ) and build_config.plugin_config.use_fp8_context_fmha:
            build_config.plugin_config.use_fp8_context_fmha = False
            logger.warning(
                "FP8 Paged Context FMHA is disabled because it must be used together with fp8 KV Cache."
            )
        elif model.config.quant_mode.has_fp8_kv_cache(
        ) and not build_config.plugin_config.use_fp8_context_fmha:
            build_config.plugin_config.use_fp8_context_fmha = True
            logger.warning(
                "FP8 Context FMHA is enabled to support FP8 Paged Context FMHA."
            )

    if build_config.plugin_config.use_paged_context_fmha and model.config.quant_mode.has_int8_kv_cache(
    ):
        build_config.plugin_config.use_paged_context_fmha = False
        logger.warning(
            "Paged Context FMHA is disabled because it doesn't work with int8 kv cache currently."
        )

    if get_sm_version() >= 100 and get_sm_version() < 120:
        if model.config.quant_mode.is_int8_weight_only(
        ) or model.config.quant_mode.is_int4_weight_only(
        ) or model.config.quant_mode.has_int8_kv_cache():
            raise RuntimeError(
                "INT8/INT4 quantization is not supported on SM>=100.")
        if model.config.quant_mode.has_act_and_weight_quant():
            raise RuntimeError("SmoothQuant is not supported on SM>=100.")
        if model.config.quant_mode.has_per_channel_scaling(
        ) or model.config.quant_mode.has_per_token_dynamic_scaling():
            raise RuntimeError(
                "Per-channel or per-token scaling is not supported on SM>=100.")

    model = optimize_model_with_config(model, build_config)

    builder = Builder()
    builder_config = builder.create_builder_config(
        precision=model.config.dtype,
        use_refit=build_config.use_refit,
        timing_cache=build_config.input_timing_cache,
        int8=(model.config.quant_mode.has_act_or_weight_quant()
              and not model.config.quant_mode.has_per_group_scaling())
        or model.config.quant_mode.has_int8_kv_cache(),
        strongly_typed=build_config.strongly_typed,
        force_num_profiles=build_config.force_num_profiles,
        profiling_verbosity=build_config.profiling_verbosity,
        quant_mode=model.config.quant_mode,
        use_strip_plan=build_config.use_strip_plan,
        weight_sparsity=build_config.weight_sparsity,
        weight_streaming=build_config.weight_streaming,
        monitor_memory=build_config.monitor_memory,
    )

    network = builder.create_network()
    network.plugin_config = build_config.plugin_config

    use_weight_only = model.config.quant_mode.is_weight_only()
    per_group = model.config.quant_mode.has_per_group_scaling()
    use_smooth_quant = model.config.quant_mode.has_act_and_weight_quant()
    use_qserve = model.config.quant_mode.is_qserve_w4a8()
    use_fp8_rowwise = model.config.quant_mode.has_fp8_rowwise()
    disable_weight_only_quant_plugin = model.config.disable_weight_only_quant_plugin if hasattr(
        model.config, 'disable_weight_only_quant_plugin') else False
    use_fp8_rowwise = model.config.quant_mode.has_fp8_rowwise()
    use_fp4_gemm = model.config.quant_mode.has_nvfp4()
    if use_fp4_gemm and network.plugin_config._explicitly_disable_gemm_plugin is False:
        logger.info(
            'NVFP4 quantization detected, by default enabling NVFP4 GEMM plugin. To use OOTB GEMM, please explicitly set gemm_plugin to "disable"'
        )
        network.plugin_config.gemm_plugin = "nvfp4"

    if build_config.plugin_config.manage_weights:
        if use_weight_only and disable_weight_only_quant_plugin:
            raise RuntimeError(
                "Manage weights of weight only quant works only with plugin currently."
            )

    if use_weight_only and not disable_weight_only_quant_plugin:
        if per_group:
            network.plugin_config.weight_only_groupwise_quant_matmul_plugin = model.config.dtype
        else:
            network.plugin_config.weight_only_quant_matmul_plugin = model.config.dtype
    if use_smooth_quant and model.config.quantization._use_plugin_sq and build_config.plugin_config.smooth_quant_plugins:
        network.plugin_config.set_smooth_quant_plugins(model.config.dtype)
    if use_qserve:
        network.plugin_config.set_qserve_plugins(model.config.dtype)
    if use_fp8_rowwise:
        network.plugin_config.set_fp8_rowwise_quant_plugins(model.config.dtype)
    nccl_plugin = model.config.dtype if model.config.mapping.world_size > 1 else None
    network.plugin_config.set_nccl_plugin(nccl_plugin)

    with net_guard(network):
        # Prepare
        network.set_named_parameters(model.named_parameters())

        # Forward
        prepare_input_args = {
            "max_batch_size":
            build_config.max_batch_size,
            "max_input_len":
            build_config.max_input_len,
            "max_seq_len":
            build_config.max_seq_len,
            "use_cache":
            build_config.kv_cache_type != KVCacheType.DISABLED,
            "max_beam_width":
            build_config.max_beam_width,
            "max_num_tokens":
            build_config.max_num_tokens,
            "opt_num_tokens":
            build_config.opt_num_tokens,
            "prompt_embedding_table_size":
            build_config.max_prompt_embedding_table_size,
            "max_draft_len":
            build_config.max_draft_len,
            "speculative_decoding_draft_tokens_external":
            build_config.speculative_decoding_mode ==
            SpeculativeDecodingMode.DRAFT_TOKENS_EXTERNAL,
            "gather_context_logits":
            build_config.gather_context_logits,
            "lora_target_modules":
            build_config.lora_config.lora_target_modules
        }

        if model.config.architecture == "DecoderModel" or "mllama" in model.config.architecture.lower(
        ):
            prepare_input_args["max_seq_len"] = build_config.max_seq_len
            prepare_input_args[
                "max_decoder_input_len"] = build_config.max_input_len
            prepare_input_args[
                "max_encoder_input_len"] = build_config.max_encoder_input_len

        if model.config.architecture == "WhisperEncoder":

            prepare_input_args = {
                "max_batch_size": build_config.max_batch_size,
            }

        if build_config.speculative_decoding_mode == SpeculativeDecodingMode.EAGLE:
            prepare_input_args[
                "spec_decoding_is_generation_length_variable"] = True
            assert build_config.max_batch_size <= 512, "Max batch size > 512 is not supported for EAGLE"
            assert build_config.max_draft_len <= 256, "Max draft len > 256 is not supported for EAGLE"

        if build_config.speculative_decoding_mode == SpeculativeDecodingMode.LOOKAHEAD_DECODING:
            prepare_input_args[
                "spec_decoding_is_generation_length_variable"] = True
        if model.config.architecture == "Qwen2VLForConditionalGeneration" or model.config.architecture == "Qwen2VLModel":
            prepare_input_args[
                'mrope_rotary_cos_sin_size'] = model.config.max_position_embeddings * model.config.rotary_embedding_dim
        if build_config.speculative_decoding_mode == SpeculativeDecodingMode.EAGLE and not build_config.plugin_config.use_paged_context_fmha:
            logger.warning(
                "Paged Context FMHA is required for EAGLE. Turning it on")
            build_config.plugin_config.use_paged_context_fmha = True

        inputs = model.prepare_inputs(**prepare_input_args)
        model(**inputs)

        if build_config.enable_debug_output:
            for k, v in model.named_network_outputs():
                network._mark_output(v, k, str_dtype_to_trt(model.config.dtype))

    if model.config.architecture != "DecoderModel":
        optimize(network)

    if build_config.visualize_network is not None:
        with net_guard(network):
            network.to_onnx(build_config.visualize_network)

    # Network -> Engine
    logger.info(
        f"Total time of constructing network from module object {time.time()-tic} seconds"
    )
    managed_weights = {} if network.plugin_config.manage_weights else None
    engine = None if build_config.dry_run else builder.build_engine(
        network, builder_config, managed_weights)
    engine_config = EngineConfig(model.config, build_config, __version__)

    if build_config.output_timing_cache is not None and model.config.mapping.rank == 0:
        ok = builder.save_timing_cache(builder_config,
                                       build_config.output_timing_cache)
        assert ok, "Failed to save timing cache."

    import psutil

    # Get the current process
    current_process = psutil.Process()
    # Get resource usage for the current process (self)
    rusage_s = current_process.memory_info()
    # Get resource usage for all child processes
    children = current_process.children(recursive=True)
    rusage_c = [child.memory_info() for child in children]
    logger.info(
        f"Build phase peak memory: {rusage_s.rss / 1024 / 1024:.2f} MB, children: {sum([ru.rss for ru in rusage_c]) / 1024 / 1024:.2f} MB"
    )

    return Engine(engine_config, engine, managed_weights)

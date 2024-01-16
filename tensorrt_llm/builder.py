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
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import tensorrt as trt
from packaging import version

from ._common import _is_building
from ._utils import to_dict, to_json_file, trt_version
from .logger import logger
from .network import Network
from .plugin import PluginConfig
from .plugin.plugin import ContextFMHAType
from .quantization import QuantMode


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
            if k != '_trt_builder_config' and k != 'plugin_config':
                config['builder_config'][k] = self.__getattribute__(k)
        if hasattr(self, 'plugin_config'):
            assert isinstance(self.plugin_config, PluginConfig), \
                f"Found unexpected plugin_config object with type: {type(self.plugin_config)}"
            config['plugin_config'] = to_dict(self.plugin_config)
        return config


class Builder():

    _ALLOWED_PRECISIONS = ['float32', 'float16', 'bfloat16']

    def __init__(self):
        super().__init__()
        self._trt_builder = trt.Builder(logger.trt_logger)
        self.strongly_typed = False

    @property
    def trt_builder(self) -> trt.Builder:
        return self._trt_builder

    def create_network(self) -> Network:
        explicit_batch_flag = 0
        if "EXPLICIT_BATCH" in trt.NetworkDefinitionCreationFlag.__members__.keys(
        ):
            # Explicit batch flag will be deprecated in TRT 10
            explicit_batch_flag = 1 << int(
                trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        if version.parse(trt_version()) >= version.parse(
                "9.1.0") and self.strongly_typed:
            return Network()._init(
                self.trt_builder.create_network(
                    explicit_batch_flag
                    | (1 << int(
                        trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))))
        else:
            return Network()._init(
                self.trt_builder.create_network(explicit_batch_flag))

    def create_builder_config(self,
                              precision: str,
                              timing_cache: Union[str, Path,
                                                  trt.ITimingCache] = None,
                              tensor_parallel: int = 1,
                              use_refit: bool = False,
                              int8: bool = False,
                              strongly_typed: bool = False,
                              opt_level: Optional[int] = None,
                              profiling_verbosity: str = "layer_names_only",
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
        self.strongly_typed = strongly_typed

        quant_mode = kwargs.get("quant_mode", QuantMode(0))
        if not strongly_typed and precision not in self._ALLOWED_PRECISIONS:
            logger.error(
                f"precision should be one of {self._ALLOWED_PRECISIONS}")

        if use_refit and int8:
            # TRT folds weights into Myelin graph because network contains int8 tensor or Q/DQ nodes
            # These folded weights can not be refitted
            logger.error(f"can't use refit and int8 mode at the same time")

        config = self.trt_builder.create_builder_config()
        if not strongly_typed:
            fp8 = quant_mode.has_fp8_qdq() or quant_mode.has_fp8_kv_cache()
            if precision == 'float16' or precision == trt.DataType.HALF:
                config.set_flag(trt.BuilderFlag.FP16)
                config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            elif precision == 'bfloat16' or precision == trt.DataType.BF16:
                config.set_flag(trt.BuilderFlag.BF16)
                config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            if int8:
                config.set_flag(trt.BuilderFlag.INT8)

            if fp8:
                config.set_flag(trt.BuilderFlag.FP8)
                config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

        config.set_preview_feature(trt.PreviewFeature.PROFILE_SHARING_0806,
                                   True)

        if use_refit:
            config.set_flag(trt.BuilderFlag.REFIT)

        if opt_level is not None:
            config.builder_optimization_level = opt_level

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

        return BuilderConfig()._init(config,
                                     precision=precision,
                                     tensor_parallel=tensor_parallel,
                                     use_refit=use_refit,
                                     int8=int8,
                                     **kwargs)

    def _add_optimization_profile(self, network: Network,
                                  builder_config: BuilderConfig):
        assert isinstance(builder_config, BuilderConfig)
        assert isinstance(network, Network)
        input_tensors = network._inputs
        num_profiles = len(list(input_tensors.items())[0][1].profiles)
        for i in range(num_profiles):
            logger.debug(f'Adding optimization profile {i+1}/{num_profiles}')
            profile = self.trt_builder.create_optimization_profile()
            for input_name in input_tensors.keys():
                shape_profile = input_tensors[input_name].profiles[i]
                min_shape = [*shape_profile.min]
                opt_shape = [*shape_profile.opt]
                max_shape = [*shape_profile.max]
                if network._autopp_config is not None:
                    io_shards = network._autopp_config["io_shards"]
                    if input_name in io_shards:
                        shards = io_shards[input_name]
                        for dim, shard_num in shards.items():
                            min_shape[dim] = int(
                                math.floor(min_shape[dim] / shard_num))
                            opt_shape[dim] = int(
                                round(opt_shape[dim] / shard_num))
                            max_shape[dim] = int(
                                math.ceil(max_shape[dim] / shard_num))
                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                logger.debug(
                    f'{input_name}, min: {min_shape}, opt: {opt_shape}, max: {max_shape}, dimension names: {shape_profile.dimension_names}'
                )
            builder_config.trt_builder_config.add_optimization_profile(profile)
        assert self._validate_named_dimensions(
            network, builder_config
        ), "Validation of the tensor dimension ranges failed, please check the dimension ranges, find the offensive tensor and dimension name in above the error log"

    def _validate_named_dimensions(self, network: Network,
                                   builder_config) -> bool:
        '''
            For each profile, validate that the named dimensions of different input tensors in this profile all have same range.
            TRT will validate the same condition, validate it earlier to make sure the modeling in TensorRT-LLM are correct and
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
                        f"Offensive tensors which have this dimension are:\n" +
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
        logger.info(f'Refit TRT engine')
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
                f'Please set named parameters before building multiple engines.'
            )
            return None

        if not refitter.refit_cuda_engine():
            logger.error(f'Failed to refit engine.')
            return None

        tok = time.time()
        t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
        logger.info(f'Total time of refitting {engine.name}: {t}')
        serialized_engine = engine.serialize()
        return serialized_engine

    @_is_building
    def build_engine(self, network: Network,
                     builder_config: BuilderConfig) -> trt.IHostMemory:
        '''
            @brief: Build one TensorRT engine from the network.
            @param network: Network object.
            @param builder_config: BuilderConfig object.
            @return: A serialized TRT engine.
        '''
        assert isinstance(network, Network)
        builder_config.plugin_config = network.plugin_config
        builder_config.autopp_config = network.autopp_config
        if builder_config.trt_builder_config.num_optimization_profiles == 0:
            self._add_optimization_profile(network, builder_config)
        engine = None

        # Rename weights
        if network.named_parameters is not None:
            for name, param in network.named_parameters:
                if param._get_weights(
                ) is None or not network.trt_network.set_weights_name(
                        param._get_weights(), name):
                    raise RuntimeError(f'Failed to set weight: {name}')

        network._fill_weights()
        # Build engine
        logger.info(f'Build TensorRT engine {network.trt_network.name}')
        tik = time.time()
        engine = self.trt_builder.build_serialized_network(
            network.trt_network, builder_config.trt_builder_config)
        if engine is None:
            logger.error('Engine building failed, please check the error log.')
            return None

        tok = time.time()
        t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
        logger.info(f'Total time of building {network.trt_network.name}: {t}')

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
    max_input_len: int = 256
    max_output_len: int = 256
    max_batch_size: int = 8
    max_beam_width: int = 1
    max_num_tokens: Optional[int] = None
    max_prompt_embedding_table_size: int = 0
    gather_context_logits: int = False
    gather_generation_logits: int = False
    strongly_typed: bool = False
    plugin_config: PluginConfig = PluginConfig()

    @classmethod
    def from_dict(cls, config):
        max_input_len = config.pop('max_input_len')
        max_output_len = config.pop('max_output_len')
        max_batch_size = config.pop('max_batch_size')
        max_beam_width = config.pop('max_beam_width')
        max_num_tokens = config.pop('max_num_tokens')
        max_prompt_embedding_table_size = config.pop(
            'max_prompt_embedding_table_size', 0)
        gather_context_logits = config.pop('gather_context_logits', False)
        gather_generation_logits = config.pop('gather_generation_logits', False)
        strongly_typed = config.pop('strongly_typed', False)

        plugin_config = PluginConfig()
        if 'plugin_config' not in config:
            return cls(
                max_input_len=max_input_len,
                max_output_len=max_output_len,
                max_batch_size=max_batch_size,
                max_beam_width=max_beam_width,
                max_num_tokens=max_num_tokens,
                max_prompt_embedding_table_size=max_prompt_embedding_table_size,
                gather_context_logits=gather_context_logits,
                gather_generation_logits=gather_generation_logits,
                plugin_config=plugin_config)

        config = config['plugin_config']
        gpt_attention_plugin = config.pop('gpt_attention_plugin', False)
        if gpt_attention_plugin:
            plugin_config.set_gpt_attention_plugin(dtype=gpt_attention_plugin)

        gemm_plugin = config.pop('gemm_plugin', False)
        if gemm_plugin:
            plugin_config.set_gemm_plugin(dtype=gemm_plugin)

        lookup_plugin = config.pop('lookup_plugin', False)
        if lookup_plugin:
            plugin_config.set_lookup_plugin(dtype=lookup_plugin)

        enable_context_fmha = config.pop('enable_context_fmha', False)
        enable_context_fmha_fp32_acc = config.pop(
            'enable_context_fmha_fp32_acc', False)
        assert not (enable_context_fmha and enable_context_fmha_fp32_acc)
        if enable_context_fmha:
            plugin_config.set_context_fmha(ContextFMHAType.enabled)
        if enable_context_fmha_fp32_acc:
            plugin_config.set_context_fmha(
                ContextFMHAType.enabled_with_fp32_acc)

        remove_input_padding = config.pop('remove_input_padding', False)
        if remove_input_padding:
            plugin_config.enable_remove_input_padding()

        paged_kv_cache = config.pop('paged_kv_cache', False)
        tokens_per_block = config.pop('tokens_per_block', 64)
        if paged_kv_cache:
            plugin_config.enable_paged_kv_cache(tokens_per_block)

        use_custom_all_reduce = config.pop('use_custom_all_reduce', False)
        plugin_config.use_custom_all_reduce = use_custom_all_reduce

        return cls(
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_num_tokens=max_num_tokens,
            max_prompt_embedding_table_size=max_prompt_embedding_table_size,
            gather_context_logits=gather_context_logits,
            gather_generation_logits=gather_generation_logits,
            strongly_typed=strongly_typed,
            plugin_config=plugin_config)

    @classmethod
    def from_json_file(cls, config_file):
        with open(config_file) as f:
            config = json.load(f)
            return BuildConfig.from_dict(config)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        plugin_config = output.pop('plugin_config')
        plugin_config_dict = copy.deepcopy(plugin_config.__dict__)
        output['plugin_config'] = plugin_config_dict
        return output

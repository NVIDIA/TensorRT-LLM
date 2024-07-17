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
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

import tensorrt as trt

from ._common import _is_building, check_max_num_tokens, serialize_engine
from ._utils import (str_dtype_to_trt, support_strongly_type, to_json_file,
                     trt_gte_10)
from .auto_parallel import auto_parallel
from .auto_parallel.config import AutoParallelConfig
from .graph_rewriting import optimize
from .logger import logger
from .lora_manager import LoraConfig
from .models import PretrainedConfig, PretrainedModel
from .models.modeling_utils import SpeculativeDecodingMode, optimize_model
from .network import Network, net_guard
from .plugin import PluginConfig
from .quantization import QuantAlgo, QuantMode
from .version import __version__


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
            },
            "auto_parallel_config": {
                # the network auto_parallel_config (if any) attached to this BuilderConfig object
                # inside the Builder.build_engine
            }
        }
        '''
        config = {'builder_config': {}}
        for k in self.__dict__.keys():
            if k not in [
                    '_trt_builder_config', 'plugin_config',
                    'auto_parallel_config'
            ]:
                config['builder_config'][k] = self.__getattribute__(k)
        if hasattr(self, 'plugin_config'):
            assert isinstance(self.plugin_config, PluginConfig), \
                f"Found unexpected plugin_config object with type: {type(self.plugin_config)}"
            config['plugin_config'] = self.plugin_config.to_dict()
        return config


class Builder():

    _ALLOWED_PRECISIONS = [
        'float32', 'float16', 'bfloat16', trt.DataType.HALF, trt.DataType.FLOAT,
        trt.DataType.BF16
    ]

    def __init__(self):
        super().__init__()
        self._trt_builder = trt.Builder(logger.trt_logger)
        # TODO: Enable strongly_typed on by default in TRT 10.0
        self.strongly_typed = False

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

        if support_strongly_type() and self.strongly_typed:
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
                              strongly_typed: bool = False,
                              opt_level: Optional[int] = None,
                              profiling_verbosity: str = "layer_names_only",
                              use_strip_plan: bool = False,
                              weight_streaming: bool = False,
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
        if strongly_typed and not support_strongly_type():
            logger.warning(
                "TRT version does not support strongly_type. strongly_typed flag is ignored."
            )

        # In TRT 10.0, enable strongly_typed by default.
        self.strongly_typed = self.strongly_typed or (strongly_typed and
                                                      support_strongly_type())

        quant_mode = kwargs.get("quant_mode", QuantMode(0))
        if not strongly_typed and precision not in self._ALLOWED_PRECISIONS:
            logger.error(
                f"precision should be one of {self._ALLOWED_PRECISIONS}")

        if use_strip_plan and not trt_gte_10():
            logger.error(
                "cannot use --strip_plan with tensorrt version 9.x or below")

        if (use_refit or use_strip_plan) and int8 and not trt_gte_10():
            # TRT folds weights into Myelin graph because network contains int8 tensor or Q/DQ nodes
            # These folded weights can not be refitted
            logger.error(
                "can't use refit/strip_plan and int8 mode at the same time before tensorrt 10.0"
            )

        config = self.trt_builder.create_builder_config()
        if weight_streaming:
            assert trt_gte_10(), \
                  "Weight streaming is only supported by TensorRT 10.0 or later."
            config.set_flag(trt.BuilderFlag.WEIGHT_STREAMING)
        if not self.strongly_typed:
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

        if use_strip_plan:
            config.set_flag(trt.BuilderFlag.STRIP_PLAN)

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

        # set weight sparsity
        weight_sparsity = kwargs.get("weight_sparsity", False)
        if weight_sparsity:
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

        return BuilderConfig()._init(config,
                                     precision=precision,
                                     tensor_parallel=tensor_parallel,
                                     use_refit=use_refit,
                                     int8=int8,
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
        for i in range(num_profiles):
            logger.debug(f'Adding optimization profile {i+1}/{num_profiles}')
            profile = self.trt_builder.create_optimization_profile()
            for input_name in input_tensors.keys():
                shape_profile = input_tensors[input_name].profiles[i]
                min_shape = [*shape_profile.min]
                opt_shape = [*shape_profile.opt]
                max_shape = [*shape_profile.max]
                if network._auto_parallel_config is not None:
                    io_shards = network._auto_parallel_config["io_shards"]
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
        builder_config.auto_parallel_config = network.auto_parallel_config
        if builder_config.auto_parallel_config is not None:
            mapping = builder_config.auto_parallel_config["mapping"]
            builder_config.tensor_parallel = mapping.tp_size
            builder_config.pipeline_parallel = mapping.pp_size
            builder_config.moe_tensor_parallel = mapping.moe_tp_size
            builder_config.moe_expert_parallel = mapping.moe_ep_size
        if builder_config.trt_builder_config.num_optimization_profiles == 0:
            self._add_optimization_profile(network, builder_config)
        engine = None

        # Rename weights
        if network.named_parameters is not None:
            for name, param in network.named_parameters:
                if param._get_weights() is None:
                    if not param.is_buffer:
                        logger.info(
                            f"Parameter {name} {param.raw_value.shape} {param.raw_value.dtype} was created"
                            " but unused in forward method, so not materialized to TRT network"
                        )
                    continue
                if not network.trt_network.set_weights_name(
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
    max_seq_len: int = 512
    opt_batch_size: int = 8
    max_batch_size: int = 8
    max_beam_width: int = 1
    max_num_tokens: Optional[int] = None
    opt_num_tokens: Optional[int] = None
    max_prompt_embedding_table_size: int = 0
    gather_context_logits: int = False
    gather_generation_logits: int = False
    strongly_typed: bool = False
    builder_opt: Optional[int] = None
    profiling_verbosity: str = 'layer_names_only'
    enable_debug_output: bool = False
    max_draft_len: int = 0
    speculative_decoding_mode: SpeculativeDecodingMode = SpeculativeDecodingMode.NONE
    use_refit: bool = False
    input_timing_cache: str = None
    output_timing_cache: str = None
    lora_config: LoraConfig = LoraConfig()
    auto_parallel_config: AutoParallelConfig = field(
        default_factory=AutoParallelConfig)
    weight_sparsity: bool = False
    weight_streaming: bool = False
    plugin_config: PluginConfig = field(default_factory=PluginConfig)
    use_strip_plan: bool = False
    max_encoder_input_len: int = 1  # for enc-dec DecoderModel
    use_fused_mlp: bool = False
    dry_run: bool = False
    visualize_network: bool = False

    def __post_init__(self):
        """
        Check and may modify max_num_tokens and opt_num_tokens after instantiation
        """
        max_num_tokens, opt_num_tokens = check_max_num_tokens(
            max_num_tokens=self.max_num_tokens,
            opt_num_tokens=self.opt_num_tokens,
            max_batch_size=self.max_batch_size,
            max_input_len=self.max_input_len,
            max_seq_len=self.max_seq_len,
            max_beam_width=self.max_beam_width,
            remove_input_padding=self.plugin_config.remove_input_padding,
            enable_context_fmha=self.plugin_config.context_fmha,
            tokens_per_block=self.plugin_config.tokens_per_block,
            multiple_profiles=self.plugin_config.multiple_profiles,
        )
        self.max_num_tokens, self.opt_num_tokens = max_num_tokens, opt_num_tokens

        if self.plugin_config.remove_input_padding and self.plugin_config.context_fmha:
            if self.max_input_len:
                logger.warning(
                    'padding removal and fMHA are both enabled, max_input_len is not required and will be ignored'
                )
        else:
            assert self.max_input_len is not None, 'padding removal and fMHA aren\'t both enabled, max_input_len is required'
            if self.max_seq_len:
                assert self.max_input_len <= self.max_seq_len, 'max_input_len should not be larger than max_seq_len'

    @classmethod
    def from_dict(cls, config, plugin_config=None):
        max_input_len = config.pop('max_input_len')
        max_seq_len = config.pop('max_seq_len')
        max_batch_size = config.pop('max_batch_size')
        max_beam_width = config.pop('max_beam_width')
        max_num_tokens = config.pop('max_num_tokens')
        opt_num_tokens = config.pop('opt_num_tokens')
        opt_batch_size = config.pop('opt_batch_size', None)
        max_prompt_embedding_table_size = config.pop(
            'max_prompt_embedding_table_size', 0)
        gather_context_logits = config.pop('gather_context_logits', False)
        gather_generation_logits = config.pop('gather_generation_logits', False)
        strongly_typed = config.pop('strongly_typed', False)
        builder_opt = config.pop('builder_opt', None)
        weight_sparsity = config.pop('weight_sparsity', False)
        profiling_verbosity = config.pop('profiling_verbosity',
                                         'layer_names_only')
        enable_debug_output = config.pop('enable_debug_output', False)
        max_draft_len = config.pop('max_draft_len', 0)
        speculative_decoding_mode = config.pop('speculative_decoding_mode',
                                               SpeculativeDecodingMode.NONE)
        use_refit = config.pop('use_refit', False)
        input_timing_cache = config.pop('input_timing_cache', None)
        output_timing_cache = config.pop('output_timing_cache', None)
        lora_config = LoraConfig.from_dict(config.get('lora_config', {}))
        auto_parallel_config = AutoParallelConfig.from_dict(
            config.get('auto_parallel_config', {}))
        max_encoder_input_len = config.pop('max_encoder_input_len', 1024)
        weight_streaming = config.pop('weight_streaming', False)

        use_strip_plan = config.pop('use_strip_plan', False)

        if plugin_config is None:
            plugin_config = PluginConfig()
        if "plugin_config" in config.keys():
            plugin_config.update_from_dict(config["plugin_config"])

        dry_run = config.pop('dry_run', False)
        visualize_network = config.pop('visualize_network', False)

        return cls(
            max_input_len=max_input_len,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_num_tokens=max_num_tokens,
            opt_num_tokens=opt_num_tokens,
            opt_batch_size=opt_batch_size,
            max_prompt_embedding_table_size=max_prompt_embedding_table_size,
            gather_context_logits=gather_context_logits,
            gather_generation_logits=gather_generation_logits,
            strongly_typed=strongly_typed,
            builder_opt=builder_opt,
            profiling_verbosity=profiling_verbosity,
            enable_debug_output=enable_debug_output,
            max_draft_len=max_draft_len,
            speculative_decoding_mode=speculative_decoding_mode,
            use_refit=use_refit,
            input_timing_cache=input_timing_cache,
            output_timing_cache=output_timing_cache,
            lora_config=lora_config,
            auto_parallel_config=auto_parallel_config,
            use_strip_plan=use_strip_plan,
            max_encoder_input_len=max_encoder_input_len,
            weight_sparsity=weight_sparsity,
            weight_streaming=weight_streaming,
            plugin_config=plugin_config,
            dry_run=dry_run,
            visualize_network=visualize_network)

    @classmethod
    def from_json_file(cls, config_file, plugin_config=None):
        with open(config_file) as f:
            config = json.load(f)
            return BuildConfig.from_dict(config, plugin_config=plugin_config)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output['plugin_config'] = output['plugin_config'].to_dict()
        output['lora_config'] = output['lora_config'].to_dict()
        output['auto_parallel_config'] = output['auto_parallel_config'].to_dict(
        )
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
            config = json.load(f)
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

    def __init__(self, config: EngineConfig, engine: Union[trt.IHostMemory,
                                                           None]):
        self.config = config
        self.engine = engine

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
            with open(os.path.join(engine_dir, 'config.json'),
                      "w",
                      encoding="utf-8") as f:
                json.dump(self.config.to_dict(), f, indent=4)
        if self.engine is not None:
            serialize_engine(
                self.engine,
                os.path.join(
                    engine_dir,
                    f'rank{self.config.pretrained_config.mapping.rank}.engine'))

    @classmethod
    def from_dir(cls, engine_dir: str, rank: int = 0):
        with open(os.path.join(engine_dir, f'rank{rank}.engine'), 'rb') as f:
            engine_buffer = f.read()

        config = EngineConfig.from_json_file(
            os.path.join(engine_dir, 'config.json'))
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
    use_auto_parallel = build_config.auto_parallel_config.enabled
    gemm_swiglu_plugin = build_config.plugin_config.gemm_swiglu_plugin
    if gemm_swiglu_plugin:
        if not build_config.use_fused_mlp:
            raise RuntimeError(
                "GemmSwiGLU plugin requires --use_fused_mlp flag")
        if gemm_swiglu_plugin not in ["fp8"]:
            raise RuntimeError(
                f"GemmSwiGLU plugin currently has limited support: fp8 only, "
                f"got: {gemm_swiglu_plugin}")

    if build_config.plugin_config.lora_plugin is not None:
        model.use_lora(build_config.lora_config)

    is_enc_dec = model.config.architecture in ["EncoderModel", "DecoderModel"]
    model = optimize_model(
        model,
        use_ootb_moe=build_config.plugin_config.moe_plugin is None,
        use_fused_mlp=(build_config.use_fused_mlp and not is_enc_dec
                       and not use_auto_parallel),
        gemm_swiglu_plugin_dtype=gemm_swiglu_plugin,
        use_fused_rg_lru=model.config.architecture
        in ["RecurrentGemmaForCausalLM"],
        use_unfused_qkv_gemm=use_auto_parallel,
        use_prompt_tuning=(build_config.max_prompt_embedding_table_size > 0),
        use_lora=build_config.plugin_config.lora_plugin is not None,
        max_lora_rank=build_config.lora_config.max_lora_rank,
        use_fp8_context_fmha=(
            model.config.quantization.quant_algo == QuantAlgo.FP8
            and build_config.plugin_config.use_fp8_context_fmha),
    )

    if is_enc_dec:
        model.precompute_relative_attention_bias(build_config)
    return model


def build(model: PretrainedModel,
          build_config: BuildConfig,
          return_build_config: bool = False) -> Engine | BuildConfig:
    '''Build engine from given model and optimization options specified in the build_config
       WARNING: this function may change the given \p model object state in some optimization passes
       to avoid cloning a model since normally the LLM models consumes large memory.
       Create a new fresh model object if you need to build with different options.

    '''
    # avoid changing the input config
    build_config = copy.deepcopy(build_config)
    build_config.plugin_config.dtype = model.config.dtype

    if model.config.quantization.quant_algo == QuantAlgo.FP8 or \
            model.config.quantization.kv_cache_quant_algo == QuantAlgo.FP8:
        build_config.strongly_typed = True

    if hasattr(model.config, 'max_draft_len'):
        build_config.max_draft_len = model.config.max_draft_len
        if build_config.speculative_decoding_mode != SpeculativeDecodingMode.MEDUSA:
            logger.warning(
                'speculative_decoding_mode is not Medusa for Medusa model. Overwriting speculative_decoding_mode'
            )
        build_config.speculative_decoding_mode = SpeculativeDecodingMode.MEDUSA

    if build_config.speculative_decoding_mode != SpeculativeDecodingMode.NONE:
        logger.info(
            f'Increasing max_seq_len ({build_config.max_seq_len}) '
            f'by max_draft_len ({build_config.max_draft_len}) '
            'to account for speculative decoding implementation specifics. '
            'Maximum number of generated tokens remains the same. '
            f'New max_seq_len is set to {build_config.max_seq_len + build_config.max_draft_len}'
        )
        build_config.max_seq_len += build_config.max_draft_len

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

    if build_config.plugin_config.use_paged_context_fmha:
        if (model.config.quant_mode.has_fp8_kv_cache()
                and not model.config.quant_mode.has_fp8_qdq()):
            raise RuntimeError(
                "FP8 Paged Context FMHA only works with fp8 quantization workflow currently."
            )
        if (model.config.quant_mode.has_fp8_kv_cache()
                and not build_config.plugin_config.use_fp8_context_fmha):
            build_config.plugin_config.use_fp8_context_fmha = True
            logger.warning(
                "FP8 Context FMHA is enabled by default to support FP8 Paged Context FMHA."
            )
        if model.config.quant_mode.has_int8_kv_cache():
            raise RuntimeError(
                "Paged Context FMHA doesn't work with int8 kv cache currently.")

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
        opt_level=build_config.builder_opt,
        profiling_verbosity=build_config.profiling_verbosity,
        quant_mode=model.config.quant_mode,
        use_strip_plan=build_config.use_strip_plan,
        weight_sparsity=build_config.weight_sparsity,
        weight_streaming=build_config.weight_streaming,
    )

    network = builder.create_network()
    network.plugin_config = build_config.plugin_config

    use_auto_parallel = build_config.auto_parallel_config.enabled
    use_weight_only = model.config.quant_mode.is_weight_only()
    per_group = model.config.quant_mode.has_per_group_scaling()
    use_smooth_quant = model.config.quant_mode.has_act_and_weight_quant()
    disable_weight_only_quant_plugin = model.config.disable_weight_only_quant_plugin if hasattr(
        model.config, 'disable_weight_only_quant_plugin') else False

    if use_weight_only and not disable_weight_only_quant_plugin:
        if per_group:
            network.plugin_config.weight_only_groupwise_quant_matmul_plugin = model.config.dtype
        else:
            network.plugin_config.weight_only_quant_matmul_plugin = model.config.dtype
    if use_smooth_quant and model.config.quantization.use_plugin_sq:
        network.plugin_config.set_smooth_quant_plugins(model.config.dtype)
    nccl_plugin = model.config.dtype if model.config.mapping.world_size > 1 else None
    network.plugin_config.set_nccl_plugin(
        nccl_plugin, network.plugin_config.use_custom_all_reduce)

    # NOTE: Please never change the build_config object after this point!
    if return_build_config:
        # Get an modified build_config that is the same as the one in the final engine dir
        return build_config

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
            True,
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
            "gather_generation_logits":
            build_config.gather_generation_logits,
            "lora_target_modules":
            build_config.lora_config.lora_target_modules
        }

        if model.config.architecture == "DecoderModel":
            prepare_input_args["max_seq_len"] = build_config.max_seq_len
            prepare_input_args[
                "max_decoder_input_len"] = build_config.max_input_len
            prepare_input_args[
                "max_encoder_input_len"] = build_config.max_encoder_input_len

        if model.config.architecture == "WhisperEncoder":

            prepare_input_args = {
                "max_batch_size": build_config.max_batch_size,
            }

        inputs = model.prepare_inputs(**prepare_input_args)
        model(**inputs)

        if build_config.enable_debug_output:
            for k, v in model.named_network_outputs():
                network._mark_output(v, k, str_dtype_to_trt(model.config.dtype))

    if model.config.architecture != "DecoderModel":
        optimize(network)

    if use_auto_parallel:
        config = build_config.auto_parallel_config
        config.builder_flags = builder_config.trt_builder_config.flags
        sharded_networks = auto_parallel(network, config)
        network = sharded_networks[model.config.mapping.rank]
        if not build_config.auto_parallel_config.debug_mode:
            mapping = network.auto_parallel_config["mapping"]
            model.config.mapping = mapping

    if build_config.visualize_network:
        network.to_dot(f'rank{model.config.mapping.rank}.dot')

    # Network -> Engine
    engine = None if build_config.dry_run else builder.build_engine(
        network, builder_config)
    engine_config = EngineConfig(model.config, build_config, __version__)

    if build_config.output_timing_cache is not None and model.config.mapping.rank == 0:
        ok = builder.save_timing_cache(builder_config,
                                       build_config.output_timing_cache)
        assert ok, "Failed to save timing cache."

    return Engine(engine_config, engine)

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
import os
from os.path import join
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from .. import profiler
from .._utils import mpi_broadcast
from ..bindings import (DataType, GptJsonConfig, KVCacheType, ModelConfig,
                        WorldConfig)
from ..bindings import executor as trtllm
from ..bindings.executor import (DecodingMode, ExternalDraftTokensConfig,
                                 OrchestratorConfig, ParallelConfig)
from ..builder import EngineConfig
from ..layers import MropeParams
from ..logger import logger
from ..mapping import Mapping
from .generation import LogitsProcessor, LoraManager
from .generation import ModelConfig as ModelConfigPython
from .generation import SamplingConfig, StoppingCriteria
from .model_runner import ModelRunnerMixin, _engine_config_to_model_config

_bindings_dtype_to_torch_dtype_dict = {
    DataType.FLOAT: torch.float,
    DataType.HALF: torch.half,
    DataType.INT8: torch.int8,
    DataType.INT32: torch.int32,
    DataType.BOOL: torch.bool,
    DataType.UINT8: torch.uint8,
    DataType.BF16: torch.bfloat16,
    DataType.INT64: torch.int64
}

SamplingConfigType = Union[SamplingConfig, trtllm.SamplingConfig]


def _world_config_to_mapping(world_config: WorldConfig):
    return Mapping(world_size=world_config.size,
                   rank=world_config.rank,
                   gpus_per_node=world_config.gpus_per_node,
                   tp_size=world_config.tensor_parallelism,
                   pp_size=world_config.pipeline_parallelism,
                   cp_size=world_config.context_parallelism)


class ModelRunnerCpp(ModelRunnerMixin):
    """
    An interface class that wraps Executor and provides generation methods.
    """

    def __init__(self,
                 executor: trtllm.Executor,
                 max_batch_size: int,
                 max_input_len: int,
                 max_seq_len: int,
                 max_beam_width: int,
                 model_config: ModelConfig,
                 world_config: WorldConfig,
                 use_kv_cache: bool,
                 lora_manager: Optional[LoraManager] = None) -> None:
        self.session = executor
        self.max_batch_size = max_batch_size
        self.max_input_len = max_input_len
        self.max_seq_len = max_seq_len
        self.max_beam_width = max_beam_width
        self.model_config = model_config
        self.mapping = _world_config_to_mapping(world_config)
        self.world_config = world_config
        self.use_kv_cache = use_kv_cache
        self.lora_manager = lora_manager

    @classmethod
    def from_dir(
        cls,
        engine_dir: str,
        *,
        lora_dir: Optional[str] = None,
        rank: int = 0,
        max_batch_size: Optional[int] = None,
        max_input_len: Optional[int] = None,
        max_output_len: Optional[int] = None,
        max_beam_width: Optional[int] = None,
        max_attention_window_size: Optional[list[int]] = None,
        sink_token_length: Optional[int] = None,
        kv_cache_free_gpu_memory_fraction: Optional[float] = None,
        cross_kv_cache_fraction: Optional[float] = None,
        medusa_choices: list[list[int]] | None = None,
        eagle_choices: list[list[int]] | None = None,
        eagle_posterior_threshold: float | None = None,
        eagle_use_dynamic_tree: bool = False,
        eagle_dynamic_tree_max_top_k: Optional[int] = None,
        lookahead_config: list[int] | None = None,
        debug_mode: bool = False,
        lora_ckpt_source: str = "hf",
        use_gpu_direct_storage: bool = False,
        gpu_weights_percent: float = 1,
        max_tokens_in_paged_kv_cache: int | None = None,
        kv_cache_enable_block_reuse: bool = False,
        enable_chunked_context: bool = False,
        is_enc_dec: bool = False,
        multi_block_mode: bool = True,
        enable_context_fmha_fp32_acc: Optional[bool] = None,
        cuda_graph_mode: Optional[bool] = None,
        logits_processor_map: Optional[Dict[str, LogitsProcessor]] = None,
        device_ids: List[int] | None = None,
        is_orchestrator_mode: bool = False,
        use_runtime_defaults: bool = True,
        gather_generation_logits: bool = False,
        use_variable_beam_width_search: bool = False,
        mm_embedding_offloading: bool = False,
        fail_fast_on_attention_window_too_large: bool = False,
    ) -> 'ModelRunnerCpp':
        """
        Create a ModelRunnerCpp instance from an engine directory.

        Args:
            engine_dir (str):
                The directory that contains the serialized engine files and config files.
            lora_dir (str):
                The directory that contains LoRA weights.
            rank (int):
                The runtime rank id.
            max_batch_size (int):
                The runtime batch size limit. If max_batch_size is not None, it should not
                be larger than the engine's max_batch_size; otherwise, the engine's max_batch_size
                will be used.
            max_input_len (int):
                The runtime input length limit. If max_input_len is not None, it should not
                be larger than the engine's max_input_len; otherwise, the engine's max_input_len
                will be used.
            max_output_len (int):
                The runtime output length limit. If max_output_len is not None, it should not
                be larger than the engine's max_output_len; otherwise, the engine's max_output_len
                will be used.
            max_beam_width (int):
                The runtime beam width limit. If max_beam_width is not None, it should not
                be larger than the engine's max_beam_width; otherwise, the engine's max_beam_width
                will be used.
            max_attention_window_size (List[int]):
                The attention window size that controls the sliding window attention / cyclic kv cache behavior.
            sink_token_length (int) :
                The sink token length, default=0.
            kv_cache_free_gpu_memory_fraction (float) :
                Free GPU memory fraction that KV cache used.
            cross_kv_cache_fraction (float) :
                KV Cache fraction reserved for cross attention, should only be used with enc-dec models.
            debug_mode (bool):
                Whether or not to turn on the debug mode.
            medusa_choices (List[List[int]]):
                Medusa choices to use when in Medusa decoding.
            eagle_choices (List[List[int]]):
                Eagle choices to use when in Eagle-1 decoding.
            eagle_posterior_threshold float:
                Minimum token probability threshold for typical acceptance.
                Value different from None enables typical acceptance in Eagle.
            eagle_use_dynamic_tree bool:
                Whether to use Eagle-2, which is dynamic tree.
            eagle_dynamic_tree_max_top_k int:
                The maximum number of draft tokens to expand for each node in Eagle-2.
            lora_ckpt_source (str):
                Source of checkpoint. Should be one of ['hf', 'nemo'].
            max_tokens_in_paged_kv_cache (int):
                Maximum amount of tokens configured in kv cache.
            kv_cache_enable_block_reuse (bool):
                Enables block reuse in kv cache.
            enable_chunked_context (bool):
                Enables chunked context.
            is_enc_dec (bool):
                Whether the model is encoder-decoder architecture.
            multi_block_mode (bool):
                Whether to distribute the work across multiple CUDA thread-blocks on the GPU for masked MHA kernel.
            enable_context_fmha_fp32_acc (bool):
                Enable FMHA runner FP32 accumulation.
            cuda_graph_mode (bool):
                Whether to use cuda graph for inference.
            logits_processor_map (Dict[str, LogitsProcessor])
                A map of logits processor functions indexed by names. A name can be provided later to
                the generate() function to specify which logits processor to run.
            device_ids (List[int]):
                Device indices to run the Executor on.
            is_orchestrator_mode (bool):
                The mode to run the model-runner, Leader mode by default.
            gather_generation_logits (bool):
                Enable gathering generation logits.
            fail_fast_on_attention_window_too_large (bool):
                Whether to fail fast if the attention window(s) are too large to fit even a single sequence in the KVCache.
        Returns:
            ModelRunnerCpp: An instance of ModelRunnerCpp.
        """
        extended_runtime_perf_knob_config = trtllm.ExtendedRuntimePerfKnobConfig(
        )
        if multi_block_mode is not None:
            extended_runtime_perf_knob_config.multi_block_mode = multi_block_mode
        if enable_context_fmha_fp32_acc is not None:
            extended_runtime_perf_knob_config.enable_context_fmha_fp32_acc = enable_context_fmha_fp32_acc
        if cuda_graph_mode is not None:
            extended_runtime_perf_knob_config.cuda_graph_mode = cuda_graph_mode

        model_config = None
        is_multimodal = {'vision', 'llm'}.issubset({
            name
            for name in os.listdir(engine_dir)
            if os.path.isdir(os.path.join(engine_dir, name))
        })
        encoder_path = None
        decoder_path = None

        if is_enc_dec:
            if is_multimodal:
                encoder_path = join(engine_dir, 'vision')
                decoder_path = join(engine_dir, 'llm')
            else:
                encoder_path = join(engine_dir, 'encoder')
                decoder_path = join(engine_dir, 'decoder')

            encoder_config_path = Path(encoder_path) / "config.json"
            encoder_json_config = GptJsonConfig.parse_file(encoder_config_path)
            encoder_model_config = encoder_json_config.model_config
            decoder_config_path = Path(decoder_path) / "config.json"
            decoder_json_config = GptJsonConfig.parse_file(decoder_config_path)
            decoder_model_config = decoder_json_config.model_config

            json_config = decoder_json_config
            model_config = decoder_model_config
            engine_dir = decoder_path

            if max_input_len is None and not is_multimodal:
                max_input_len = encoder_model_config.max_input_len
        else:
            config_path = Path(engine_dir) / "config.json"
            json_config = GptJsonConfig.parse_file(config_path)
            model_config = json_config.model_config

        use_kv_cache = model_config.kv_cache_type != KVCacheType.DISABLED
        if not model_config.use_cross_attention:
            assert cross_kv_cache_fraction is None, "cross_kv_cache_fraction should only be used with enc-dec models."

        if not use_kv_cache:
            assert max_output_len == 1 or max_output_len is None, 'Disabled KV cache is intended for context phase only now.'

        # Note: Parallel configuration will be fetched automatically from trtllm.Executor constructor
        # by inspecting the json file. These lines serve the purpose of serving vocab_size_padded and
        # num_layers properties.
        # MPI world size must be 1 in Orchestrator mode
        if is_orchestrator_mode:
            tp_size = 1
            pp_size = 1
            cp_size = 1
            # Check the count of devices equal to tp_size of engine
            # assert len(device_ids) == json_config.tensor_parallelism
        else:
            tp_size = json_config.tensor_parallelism
            pp_size = json_config.pipeline_parallelism
            cp_size = json_config.context_parallelism
        gpus_per_node = json_config.gpus_per_node
        world_config = WorldConfig.mpi(tensor_parallelism=tp_size,
                                       pipeline_parallelism=pp_size,
                                       context_parallelism=cp_size,
                                       gpus_per_node=gpus_per_node)
        assert rank == world_config.rank

        engine_config = EngineConfig.from_json_file(f"{engine_dir}/config.json")
        if model_config.use_lora_plugin and rank == 0:
            mapping = _world_config_to_mapping(world_config)
            lora_manager = LoraManager(
                mapping=mapping,
                model_config=ModelConfigPython.from_model_config_cpp(
                    model_config))
            if lora_dir is None:
                config_lora_dir = engine_config.build_config.lora_config.lora_dir
                if len(config_lora_dir) > 0:
                    lora_dir = [
                        f"{engine_dir}/{dir}" for dir in config_lora_dir
                    ]
                    lora_ckpt_source = engine_config.build_config.lora_config.lora_ckpt_source

            if lora_dir is not None:
                runtime_model_config = _engine_config_to_model_config(
                    engine_config, gpu_weights_percent=gpu_weights_percent)
                # For Executor, only rank 0 can enqueue requests, and should hold all lora weights
                lora_manager.load_from_ckpt(lora_dir,
                                            model_config=runtime_model_config,
                                            ckpt_source=lora_ckpt_source)
            else:
                raise RuntimeError(
                    f"LoRA weights are unspecified and also unavailable in the engine_dir ({engine_dir})."
                )

            max_lora_rank = engine_config.build_config.lora_config.max_lora_rank
            num_lora_modules = engine_config.pretrained_config.num_hidden_layers * \
                len(lora_manager.lora_target_modules + lora_manager.missing_qkv_modules)
            num_lora_adapters = min(lora_manager.num_lora_adapters, 8)
            peft_cache_config = trtllm.PeftCacheConfig(
                num_device_module_layer=max_lora_rank * num_lora_modules *
                num_lora_adapters,
                num_host_module_layer=max_lora_rank * num_lora_modules *
                num_lora_adapters,
            )
        else:
            lora_manager = None
            peft_cache_config = trtllm.PeftCacheConfig()

        if world_config.size > 1:
            peft_cache_config = mpi_broadcast(peft_cache_config, 0)

        profiler.start('load tensorrt_llm engine')

        kv_cache_config = trtllm.KvCacheConfig(
            free_gpu_memory_fraction=kv_cache_free_gpu_memory_fraction,
            max_attention_window=max_attention_window_size,
            sink_token_length=sink_token_length,
            max_tokens=max_tokens_in_paged_kv_cache,
            enable_block_reuse=kv_cache_enable_block_reuse,
            cross_kv_cache_fraction=cross_kv_cache_fraction,
            runtime_defaults=json_config.runtime_defaults
            if use_runtime_defaults else None,
        )

        decoding_config = trtllm.DecodingConfig()
        if medusa_choices is not None:
            decoding_config.medusa_choices = medusa_choices
            if multi_block_mode is not None:
                multi_block_mode = False  # Medusa doesn't support multi-block mode.

        if eagle_choices is not None or eagle_posterior_threshold is not None or eagle_use_dynamic_tree:
            greedy_sampling = eagle_posterior_threshold is None
            decoding_config.eagle_config = trtllm.EagleConfig(
                eagle_choices, greedy_sampling, eagle_posterior_threshold,
                eagle_use_dynamic_tree, eagle_dynamic_tree_max_top_k)
            if multi_block_mode is not None:
                logger.warning(
                    f'Multi block mode is not supported for EAGLE. Disabling it.'
                )
                multi_block_mode = False  # Eagle doesn't support multi-block mode.

        if lookahead_config is not None:
            [w, n, g] = lookahead_config
            decoding_config.lookahead_decoding_config = trtllm.LookaheadDecodingConfig(
                w, n, g)

        if use_variable_beam_width_search:
            decoding_config.decoding_mode = DecodingMode.BeamSearch(
            ).useVariableBeamWidthSearch(True)

        if max_batch_size is None:
            max_batch_size = model_config.max_batch_size
        else:
            assert max_batch_size <= model_config.max_batch_size
        if max_input_len is None:
            max_input_len = model_config.max_input_len
        # NOTE: remove assertion here for temp fix,
        # model_config.max_input_len is not the upper bound of input length.
        # If runtime max_input_len is not properly set,
        # C++ runtime will throw an error when fetching new requests
        if max_output_len is None or is_enc_dec:
            max_seq_len = model_config.max_seq_len
        else:
            max_seq_len = max_input_len + max_output_len
            assert max_seq_len <= model_config.max_seq_len
        if max_beam_width is None:
            max_beam_width = model_config.max_beam_width
        else:
            assert max_beam_width <= model_config.max_beam_width

        debug_config = None
        if debug_mode:
            # To debug specific tensors, add tensor names in the following list
            #   if none provided, all input and output tensors will be dumped
            #   if not none, it will disable all input/output dump
            debug_tensor_names: List[str] = [
            ]  # modify this list for specific tensor dump
            debug_config = trtllm.DebugConfig(
                debug_input_tensors=True,
                debug_output_tensors=True,
                debug_tensor_names=debug_tensor_names)

        trtllm_config = trtllm.ExecutorConfig(
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            kv_cache_config=kv_cache_config,
            decoding_config=decoding_config,
            peft_cache_config=peft_cache_config,
            debug_config=debug_config,
            use_gpu_direct_storage=use_gpu_direct_storage,
            gpu_weights_percent=gpu_weights_percent,
            gather_generation_logits=gather_generation_logits,
        )
        trtllm_config.enable_chunked_context = enable_chunked_context
        trtllm_config.extended_runtime_perf_knob_config = extended_runtime_perf_knob_config
        trtllm_config.mm_embedding_offloading = mm_embedding_offloading
        trtllm_config.fail_fast_on_attention_window_too_large = fail_fast_on_attention_window_too_large
        if is_orchestrator_mode:
            communication_mode = trtllm.CommunicationMode.ORCHESTRATOR
            path = str(Path(__file__).parent.parent / 'bin' / 'executorWorker')
            orchestrator_config = OrchestratorConfig(True, path)
        else:
            communication_mode = trtllm.CommunicationMode.LEADER
            orchestrator_config = None

        trtllm_config.parallel_config = ParallelConfig(
            trtllm.CommunicationType.MPI,
            communication_mode,
            device_ids=device_ids,
            orchestrator_config=orchestrator_config)

        # LogitsPostProcessor in Orchestrator mode is not supported yet.
        if not is_orchestrator_mode:
            logits_proc_config = trtllm.LogitsPostProcessorConfig()
            if logits_processor_map is not None:
                logits_proc_config.processor_map = logits_processor_map
            trtllm_config.logits_post_processor_config = logits_proc_config

        if is_enc_dec:
            executor = trtllm.Executor(encoder_path, decoder_path,
                                       trtllm.ModelType.ENCODER_DECODER,
                                       trtllm_config)
        else:
            executor = trtllm.Executor(Path(engine_dir),
                                       trtllm.ModelType.DECODER_ONLY,
                                       trtllm_config)

        profiler.stop('load tensorrt_llm engine')

        loading_time = profiler.elapsed_time_in_sec("load tensorrt_llm engine")
        logger.info(f'Load engine takes: {loading_time} sec')

        return cls(executor,
                   max_batch_size=max_batch_size,
                   max_input_len=max_input_len,
                   max_seq_len=max_seq_len,
                   max_beam_width=max_beam_width,
                   model_config=model_config,
                   world_config=world_config,
                   use_kv_cache=use_kv_cache,
                   lora_manager=lora_manager)

    def _check_inputs(self, batch_input_ids: List[List[int]],
                      encoder_input_ids: Optional[List[List[int]]],
                      sampling_config: trtllm.SamplingConfig, max_new_tokens):
        batch_size = len(encoder_input_ids) if encoder_input_ids else len(
            batch_input_ids)
        if batch_size > self.max_batch_size:
            raise RuntimeError(
                f"Input batch size ({batch_size}) exceeds the engine or specified limit ({self.max_batch_size})"
            )
        input_lengths = [
            len(x) for x in encoder_input_ids
        ] if encoder_input_ids else [len(x) for x in batch_input_ids]
        max_length = max(input_lengths)
        if max_length > self.max_input_len:
            raise RuntimeError(
                f"Maximum input length ({max_length}) exceeds the engine or specified limit ({self.max_input_len})"
            )
        if encoder_input_ids:
            decoder_max_length = max([len(x) for x in batch_input_ids])
            if decoder_max_length + max_new_tokens > self.max_seq_len:
                raise RuntimeError(
                    f"Decoder prefix tokens ({decoder_max_length}) + maximum new tokens ({max_new_tokens}) exceeds the engine or specified limit ({self.max_seq_len})"
                )
        else:
            if max_length + max_new_tokens > self.max_seq_len:
                raise RuntimeError(
                    f"Maximum input length ({max_length}) + maximum new tokens ({max_new_tokens}) exceeds the engine or specified limit ({self.max_seq_len})"
                )
        if sampling_config.beam_width > self.max_beam_width:
            raise RuntimeError(
                f"Num beams ({sampling_config.beam_width}) exceeds the engine or specified limit ({self.max_beam_width})"
            )

    @property
    def dtype(self) -> torch.dtype:
        bindings_dtype = self.model_config.data_type
        return _bindings_dtype_to_torch_dtype_dict[bindings_dtype]

    @property
    def vocab_size(self) -> int:
        return self.model_config.vocab_size

    @property
    def vocab_size_padded(self) -> int:
        return self.model_config.vocab_size_padded(self.world_config.size)

    @property
    def hidden_size(self) -> int:
        return self.model_config.hidden_size

    @property
    def num_heads(self) -> int:
        return self.model_config.num_heads

    @property
    def num_layers(self) -> int:
        return self.model_config.num_layers(
            self.world_config.pipeline_parallelism,
            self.world_config.pipeline_parallel_rank,
        )

    @property
    def max_sequence_length(self) -> int:
        return self.max_seq_len

    @property
    def remove_input_padding(self) -> bool:
        return self.model_config.use_packed_input

    @property
    def max_prompt_embedding_table_size(self) -> int:
        return self.model_config.max_prompt_embedding_table_size

    @property
    def gather_context_logits(self) -> bool:
        return self.model_config.compute_context_logits

    @property
    def gather_generation_logits(self) -> bool:
        return self.model_config.compute_generation_logits

    def generate(
            self,
            batch_input_ids: List[torch.Tensor],
            *,
            position_ids: List[torch.Tensor] = None,
            encoder_input_ids: List[torch.Tensor] = None,
            encoder_input_features: List[
                torch.Tensor] = None,  # TODO: add to doc string
            encoder_output_lengths: List[int] = None,
            cross_attention_masks: List[
                torch.Tensor] = None,  # TODO: add to doc string
            mrope_params: Optional[MropeParams] = None,
            sampling_config: Optional[SamplingConfig] = None,
            lora_uids: Optional[list] = None,
            lookahead_config: list[int] | None = None,
            streaming: bool = False,
            stopping_criteria: Optional[StoppingCriteria] = None,
            logits_processor_names: list[str] | None = None,
            max_new_tokens: int = 1,
            end_id: int | None = None,
            pad_id: int | None = None,
            bad_words_list: list[list[int]] | None = None,
            stop_words_list: list[list[int]] | None = None,
            return_dict: bool = False,
            output_sequence_lengths: bool = False,
            output_generation_logits: bool = False,
            output_log_probs: bool = False,
            output_cum_log_probs: bool = False,
            prompt_table: Optional[Union[str, torch.Tensor]] = None,
            prompt_tasks: Optional[str] = None,
            input_token_extra_ids: List[List[int]] = None,
            return_all_generated_tokens: bool = False,
            language_adapter_uids: Optional[List[int]] = None,
            mm_embedding_offloading: bool = False,
            **kwargs) -> Union[torch.Tensor, dict]:
        """
        Generates sequences of token ids.
        The generation-controlling parameters are set in the sampling_config; it will be set to a default one if not passed.
        You can override any sampling_config's attributes by passing corresponding parameters.

        Args:
            batch_input_ids (List[torch.Tensor]):
                A list of input id tensors. Each tensor is of shape (sequence_length, ).
            position_ids (List[torch.Tensor]):
                A list of position id tensors. Each tensor is of shape (sequence_length, ).
            encoder_input_ids (List[torch.Tensor]):
                A list of encoder input id tensors for encoder-decoder models (optional). Each tensor is of shape (sequence_length, ).
            encoder_input_features: (List[torch.Tensor]):
                A list of encoder input feature tensors for multimodal encoder-decoder models (optional). Each tensor is of shape (sequence_length, feature_dim).
            encoder_output_lengths: (List[int]):
                A list of encoder output lengths (optional) if encoder output has different length from encoder input (due to convolution down-sampling, etc.)
            sampling_config (SamplingConfig):
                The sampling configuration to be used as base parametrization for the generation call.
                The passed **kwargs matching the sampling_config's attributes will override them.
                If the sampling_config is not provided, a default will be used.
            prompt_table (str or torch.Tensor):
                The file path of prompt table (.npy format, exported by nemo_prompt_convert.py) or the prompt table itself.
            prompt_tasks (str):
                The prompt tuning task ids for the input batch, in format of comma-separated list (e.g., 0,3,1,0).
            input_token_extra_ids (List[List[int]]):
                Input token extra ids for using p-tuning and KV Cache reuse together
            lora_uids (list):
                The uids of LoRA weights for the input batch. Use -1 to disable the LoRA module.
            streaming (bool):
                Whether or not to use streaming mode for generation.
            stopping_criteria (StoppingCriteria):
                Custom stopping criteria.
            logits_processor_names (List[str]):
                Custom logits processor names.
            return_all_generated_tokens (bool):
                Whether the full output is returned at each streaming step
            kwargs (Dict[str, Any]:
                Ad hoc parametrization of sampling_config.
                The passed **kwargs matching the sampling_config's attributes will override them.
        Returns:
            torch.Tensor or dict:
                If return_dict=False, the method returns generated output_ids.
                If return_dict=True, the method returns a dict of output_ids,
                sequence_lengths (if sampling_config.output_sequence_lengths=True),
                context_logits and generation_logits (if self.gather_context_logits=True and
                self.gather_generation_logits=True, respectively).
        """
        # TODO: Check if these can be supported now and support them
        if stopping_criteria is not None:
            raise RuntimeError(
                "Stopping criteria is not supported in C++ session.")

        if not self.use_kv_cache and max_new_tokens > 1:
            raise RuntimeError(
                'Disabled KV cache is intended for context phase only now.')

        # If we are in a multi-gpu scenario, only rank 0 continues
        if not self.session.can_enqueue_requests():
            return []

        # Convert tensor input to plain lists
        batch_input_ids_list = [a.tolist() for a in batch_input_ids]
        encoder_input_ids_list = [a.tolist() for a in encoder_input_ids
                                  ] if encoder_input_ids else None

        if sampling_config is None:
            # Convert from old API of SamplingConfig
            # Note: Due to a Python3.10 bug one cannot use inspect on it currently
            accepted_parameters = [
                "num_beams",
                "top_k",
                "top_p",
                "top_p_min",
                "top_p_reset_ids",
                "top_p_decay",
                "temperature",
                "min_tokens",
                "beam_search_diversity_rate",
                "repetition_penalty",
                "presence_penalty",
                "frequency_penalty",
                "length_penalty",
                "early_stopping",
                "no_repeat_ngram_size",
                "random_seed",
                "num_return_sequences",
                "min_p",
                "beam_width_array",
            ]
            rename_params = {"num_beams": "beam_width", "random_seed": "seed"}
            sampling_params = {
                k: v
                for k, v in kwargs.items() if k in accepted_parameters
            }
            for k, v in rename_params.items():
                if k in sampling_params:
                    sampling_params[v] = sampling_params.pop(k)
            if "top_p" in sampling_params and sampling_params["top_p"] == 0.0:
                sampling_params["top_p"] = None

            # TODO: improve usage of SamplingConfig. For example,
            # construct SamplingConfig for each request, rather than one for the whole batch.
            # Here we use beam width array for each request for Variable-Beam-Width-Search.
            batch_size = len(batch_input_ids)
            use_sampling_config_for_each_request = False
            # Just placeholder for non-Variable-Beam-Width-Search
            sampling_config_list = [None] * batch_size
            if "beam_width_array" in sampling_params and sampling_params[
                    "beam_width_array"] is not None and len(
                        sampling_params["beam_width_array"]) == batch_size:
                use_sampling_config_for_each_request = True
                sp_copy = copy.deepcopy(sampling_params)
                for i in range(batch_size):
                    bwa = sampling_params["beam_width_array"][i]
                    sp_copy["beam_width_array"] = bwa
                    sp_copy["beam_width"] = max(bwa)
                    sampling_config_list[i] = trtllm.SamplingConfig(**sp_copy)
                # Just placeholder for Variable-Beam-Width-Search and for `self._check_inputs`
                max_beam_width = max(sc.beam_width
                                     for sc in sampling_config_list)
                sampling_params["beam_width"] = max_beam_width
                sampling_params["beam_width_array"] = [max_beam_width] * 8
            sampling_config = trtllm.SamplingConfig(**sampling_params)
        else:
            sampling_config = copy.deepcopy(sampling_config)

        self._check_inputs(batch_input_ids_list, encoder_input_ids_list,
                           sampling_config, max_new_tokens)

        output_config = trtllm.OutputConfig(
            return_context_logits=self.gather_context_logits,
            return_generation_logits=self.gather_generation_logits
            or output_generation_logits,
            return_log_probs=output_log_probs,
        )

        prompt_tuning_configs = self._prepare_ptuning_executor(
            batch_input_ids_list,
            prompt_table,
            prompt_tasks,
            input_token_extra_ids,
            mm_embedding_offloading=mm_embedding_offloading)
        mrope_configs = self._prepare_mrope_executor(batch_input_ids_list,
                                                     mrope_params)

        stop_words_list = self._prepare_words_list(stop_words_list,
                                                   len(batch_input_ids_list))
        bad_words_list = self._prepare_words_list(bad_words_list,
                                                  len(batch_input_ids_list))
        logits_processor_names = self._prepare_names_list(
            logits_processor_names, len(batch_input_ids_list))

        lora_configs = self._prepare_lora_configs(lora_uids,
                                                  len(batch_input_ids_list))
        request_lookahead_config = None
        if lookahead_config is not None:
            [w, n, g] = lookahead_config
            request_lookahead_config = trtllm.LookaheadDecodingConfig(w, n, g)
        skip_cross_attn_blocks = kwargs.get('skip_cross_attn_blocks', None)

        # Draft-Target-Model speculative decoding
        if "draft_tokens_list" in kwargs.keys() and kwargs[
                "draft_tokens_list"] is not None and "draft_logits_list" in kwargs.keys(
                ) and kwargs["draft_logits_list"] is not None:
            # Use logits to accept
            external_draft_tokens_configs = [
                ExternalDraftTokensConfig(draft_tokens, draft_logits)
                for draft_tokens, draft_logits in zip(
                    kwargs["draft_tokens_list"], kwargs["draft_logits_list"])
            ]
            is_draft_target_model = True
        elif "draft_tokens_list" in kwargs.keys(
        ) and kwargs["draft_tokens_list"] is not None:
            # Use tokens to accept
            external_draft_tokens_configs = [
                ExternalDraftTokensConfig(draft_tokens)
                for draft_tokens in kwargs["draft_tokens_list"]
            ]
            is_draft_target_model = True
        else:
            external_draft_tokens_configs = [None] * len(batch_input_ids_list)
            is_draft_target_model = False

        if language_adapter_uids is None:
            language_adapter_uids = [None] * len(batch_input_ids_list)

        requests = [
            trtllm.Request(
                input_token_ids=input_ids,
                encoder_input_token_ids=encoder_input_ids_list[i]
                if encoder_input_ids is not None else None,
                encoder_output_length=encoder_output_lengths[i]
                if encoder_output_lengths is not None else None,
                encoder_input_features=encoder_input_features[i].contiguous()
                if encoder_input_features is not None else None,
                position_ids=position_ids[i].tolist()
                if position_ids is not None else None,
                cross_attention_mask=cross_attention_masks[i].contiguous() if
                (cross_attention_masks is not None
                 and cross_attention_masks[i] is not None) else None,
                max_tokens=max_new_tokens,
                pad_id=pad_id,
                end_id=end_id,
                stop_words=stop_words,
                bad_words=bad_words,
                sampling_config=(sampling_config_each_request
                                 if use_sampling_config_for_each_request else
                                 sampling_config),
                lookahead_config=request_lookahead_config,
                streaming=streaming,
                output_config=output_config,
                prompt_tuning_config=prompt_tuning_config,
                mrope_config=mrope_config,
                lora_config=lora_config,
                return_all_generated_tokens=return_all_generated_tokens,
                logits_post_processor_name=logits_post_processor_name,
                external_draft_tokens_config=external_draft_tokens_config,
                skip_cross_attn_blocks=skip_cross_attn_blocks,
                language_adapter_uid=language_adapter_uid,
            ) for i,
            (input_ids, stop_words, bad_words, prompt_tuning_config,
             mrope_config, lora_config, logits_post_processor_name,
             external_draft_tokens_config, language_adapter_uid,
             sampling_config_each_request) in enumerate(
                 zip(batch_input_ids_list, stop_words_list, bad_words_list,
                     prompt_tuning_configs, mrope_configs, lora_configs,
                     logits_processor_names, external_draft_tokens_configs,
                     language_adapter_uids, sampling_config_list))
        ]

        request_ids = self.session.enqueue_requests(requests)
        if not streaming:
            return self._initialize_and_fill_output(
                request_ids=request_ids,
                end_id=end_id,
                return_dict=return_dict,
                output_sequence_lengths=output_sequence_lengths,
                output_generation_logits=output_generation_logits,
                output_log_probs=output_log_probs,
                output_cum_log_probs=output_cum_log_probs,
                batch_input_ids=batch_input_ids,
                streaming=streaming,
                sampling_config=sampling_config,
                is_draft_target_model=is_draft_target_model,
            )
        else:
            return self._stream(
                request_ids=request_ids,
                end_id=end_id,
                return_dict=return_dict,
                output_sequence_lengths=output_sequence_lengths,
                output_generation_logits=output_generation_logits,
                output_log_probs=output_log_probs,
                output_cum_log_probs=output_cum_log_probs,
                batch_input_ids=batch_input_ids,
                batch_input_ids_list=batch_input_ids_list,
                streaming=streaming,
                return_all_generated_tokens=return_all_generated_tokens,
                sampling_config=sampling_config,
                is_draft_target_model=is_draft_target_model,
            )

    def _prepare_words_list(self, words_list: List[List[List[int]]],
                            batch_size: int):
        if words_list is None:
            return [None] * batch_size
        return words_list

    def _prepare_names_list(self, names_list: List[str], batch_size: int):
        if names_list is None:
            return [None] * batch_size
        return names_list

    def _prepare_ptuning_executor(self, batch_input_ids_list, prompt_table,
                                  prompt_tasks, input_token_extra_ids,
                                  mm_embedding_offloading):
        if input_token_extra_ids:
            assert len(batch_input_ids_list) == len(input_token_extra_ids), \
                f"Batch size of input_token_extra_ids ({len(input_token_extra_ids)}) must be the same as input batch size ({len(batch_input_ids_list)})"
        prompt_tuning_configs = len(batch_input_ids_list) * [None]
        if prompt_table is not None:
            if mm_embedding_offloading:
                # CUDA Stream Overlapping Requirements:
                # 1. Both memory copy stream and kernel execution stream must be non-default streams
                # 2. For host<->device transfers (H2D/D2H), host memory MUST be page-locked (pinned)
                prompt_table_data = self._prepare_embedding_table(
                    prompt_table).pin_memory()
            else:
                prompt_table_data = self._prepare_embedding_table(
                    prompt_table).cuda()
            if prompt_tasks is not None:
                task_indices = [int(t) for t in prompt_tasks.split(',')]
                assert len(task_indices) == len(batch_input_ids_list), \
                    f"Number of supplied tasks ({len(task_indices)}) must match input batch size ({len(batch_input_ids_list)})"
                prompt_tuning_configs = [
                    trtllm.PromptTuningConfig(
                        embedding_table=prompt_table_data[task_indices[i]],
                        input_token_extra_ids=input_token_extra_ids[i]
                        if input_token_extra_ids else None)
                    for i in range(len(batch_input_ids_list))
                ]
            else:
                prompt_tuning_configs = [
                    trtllm.PromptTuningConfig(
                        embedding_table=prompt_table_data[0],
                        input_token_extra_ids=input_token_extra_ids[i]
                        if input_token_extra_ids else None)
                    for i in range(len(batch_input_ids_list))
                ]
        return prompt_tuning_configs

    # TODO: add multimodal input for TRT engine backend

    def _prepare_mrope_executor(self, batch_input_ids_list, mrope: MropeParams):
        mrope_configs = len(batch_input_ids_list) * [None]
        if mrope != None:
            mrope_rotary_cos_sin = mrope.mrope_rotary_cos_sin
            assert isinstance(
                mrope_rotary_cos_sin,
                torch.Tensor), "mrope_rotary_cos_sin should be torch.Tensor"
            mrope_rotary_cos_sin_data = mrope_rotary_cos_sin.to(
                dtype=torch.float32)

            mrope_position_deltas = mrope.mrope_position_deltas
            assert isinstance(
                mrope_position_deltas,
                torch.Tensor), "mrope_position_deltas should be torch.Tensor"
            mrope_position_deltas_data = mrope_position_deltas.to(
                dtype=torch.int32)

            mrope_configs = [
                trtllm.MropeConfig(
                    mrope_rotary_cos_sin=mrope_rotary_cos_sin_data[i],
                    mrope_position_deltas=mrope_position_deltas_data[i])
                for i in range(len(batch_input_ids_list))
            ]
        return mrope_configs

    def _prepare_lora_configs(self, lora_uids, batch_size):
        if lora_uids is None:
            return [None] * batch_size
        assert len(lora_uids) == batch_size
        return [
            trtllm.LoraConfig(task_id=int(uid),
                              weights=self.lora_manager.cpp_lora_weights[uid],
                              config=self.lora_manager.cpp_lora_config[uid])
            if int(uid) >= 0 else None for uid in lora_uids
        ]

    def _get_num_sequences(self, sampling_config: SamplingConfigType):
        num_beams = sampling_config.num_beams if isinstance(
            sampling_config, SamplingConfig) else sampling_config.beam_width
        num_sequences = sampling_config.num_return_sequences or num_beams
        assert num_beams == 1 or num_sequences <= num_beams
        return num_sequences

    def _initialize_and_fill_output(
        self,
        *,
        request_ids,
        end_id,
        return_dict,
        output_sequence_lengths,
        output_generation_logits,
        output_log_probs,
        output_cum_log_probs,
        batch_input_ids,
        streaming,
        sampling_config: SamplingConfigType,
        is_draft_target_model: bool = False,
    ):
        num_sequences = self._get_num_sequences(sampling_config)
        # (batch_size, num_sequences, sequence_len)
        output_ids = [[[] for _ in range(num_sequences)]
                      for _ in range(len(request_ids))]

        all_responses = []
        finished_request_ids = set()
        while finished_request_ids != set(request_ids):
            responses = self.session.await_responses()
            for response in responses:
                if response.result.is_final:
                    finished_request_ids.add(response.request_id)
            all_responses.extend(responses)

        return self._fill_output(
            responses=all_responses,
            output_ids=output_ids,
            end_id=end_id,
            return_dict=return_dict,
            output_sequence_lengths=output_sequence_lengths,
            output_generation_logits=output_generation_logits,
            output_log_probs=output_log_probs,
            output_cum_log_probs=output_cum_log_probs,
            batch_input_ids=batch_input_ids,
            batch_input_ids_list=[],
            streaming=streaming,
            request_ids=request_ids,
            return_all_generated_tokens=False,
            sampling_config=sampling_config,
            is_draft_target_model=is_draft_target_model,
        )

    def _stream(
        self,
        *,
        request_ids,
        end_id,
        return_dict,
        output_sequence_lengths,
        output_generation_logits,
        output_log_probs,
        output_cum_log_probs,
        batch_input_ids,
        batch_input_ids_list,
        streaming,
        return_all_generated_tokens,
        sampling_config: SamplingConfigType,
        is_draft_target_model: bool = False,
    ):
        num_sequences = self._get_num_sequences(sampling_config)
        # (batch_size, num_sequences, sequence_len)
        output_ids = [[
            copy.deepcopy(batch_input_ids_list[batch_idx])
            for _ in range(num_sequences)
        ] for batch_idx in range(len(request_ids))]

        finished_request_ids = set()
        while finished_request_ids != set(request_ids):
            responses = self.session.await_responses()
            for response in responses:
                if response.result.is_final:
                    finished_request_ids.add(response.request_id)

            yield self._fill_output(
                responses=responses,
                output_ids=output_ids,
                end_id=end_id,
                return_dict=return_dict,
                output_sequence_lengths=output_sequence_lengths,
                output_generation_logits=output_generation_logits,
                output_log_probs=output_log_probs,
                output_cum_log_probs=output_cum_log_probs,
                batch_input_ids=batch_input_ids,
                batch_input_ids_list=batch_input_ids_list,
                streaming=streaming,
                request_ids=request_ids,
                return_all_generated_tokens=return_all_generated_tokens,
                sampling_config=sampling_config,
                is_draft_target_model=is_draft_target_model,
            )

    def _fill_output(
        self,
        *,
        responses,
        output_ids,
        end_id,
        return_dict,
        output_sequence_lengths,
        output_generation_logits,
        output_log_probs,
        output_cum_log_probs,
        batch_input_ids,
        batch_input_ids_list,
        streaming,
        request_ids,
        return_all_generated_tokens,
        sampling_config: SamplingConfigType,
        is_draft_target_model: bool,
    ):
        cuda_device = torch.device("cuda")

        batch_size = len(batch_input_ids)
        num_sequences = len(output_ids[0])
        beam_width = getattr(sampling_config, 'num_beams',
                             getattr(sampling_config, 'beam_width'))
        is_beam_search = beam_width > 1

        def fill_output_ids(result_token_ids, batch_idx, seq_idx):
            # Return shape = (batch_size, num_sequences, seq_len)
            if return_all_generated_tokens:
                output_ids[batch_idx][seq_idx] = (
                    batch_input_ids_list[batch_idx] + result_token_ids)
            else:
                output_ids[batch_idx][seq_idx] += result_token_ids

        for response in responses:
            if response.has_error():
                raise RuntimeError(response.error_msg)

            result = response.result
            batch_idx = request_ids.index(response.request_id)
            if is_beam_search:
                for beam, output_tokens in enumerate(result.output_token_ids):
                    fill_output_ids(output_tokens, batch_idx, beam)
            else:
                fill_output_ids(result.output_token_ids[0], batch_idx,
                                result.sequence_index)

        if output_sequence_lengths:
            sequence_lengths = [[len(token_ids) for token_ids in beams]
                                for beams in output_ids]

        if streaming:
            output_ids = copy.deepcopy(output_ids)

        # Pad by end_id tokens (batch, num_sequences, max_seq_len).
        for beams in output_ids:
            for token_ids in beams:
                token_ids += [end_id] * (self.max_seq_len - len(token_ids))
        output_ids = torch.tensor(output_ids,
                                  dtype=torch.int32,
                                  device=cuda_device)

        if return_dict:
            outputs = {'output_ids': output_ids}

            input_lengths = torch.tensor([x.size(0) for x in batch_input_ids],
                                         dtype=torch.int32,
                                         device=cuda_device)

            if output_sequence_lengths:
                outputs['sequence_lengths'] = torch.tensor(sequence_lengths,
                                                           dtype=torch.int32,
                                                           device=cuda_device)

            if self.gather_context_logits:
                context_logits = None
                max_input_len = input_lengths.max()
                for response in responses:
                    result = response.result
                    logits = result.context_logits
                    if logits is None:
                        continue
                    input_len, vocab_size = logits.shape
                    if context_logits is None:
                        context_logits = torch.zeros(
                            (batch_size, max_input_len, vocab_size),
                            dtype=logits.dtype,
                            device=cuda_device)
                    if result.sequence_index == 0:
                        batch_idx = request_ids.index(response.request_id)
                        context_logits[batch_idx, :input_len, :] = logits
                assert context_logits is not None
                outputs['context_logits'] = context_logits

            if self.gather_generation_logits or output_generation_logits:
                gen_logits = None
                if is_draft_target_model:
                    # Put the outputs in a list rather than a tensor since their
                    # length may vary among requests in a batch
                    gen_logits = [
                        a.result.generation_logits.cuda() for a in responses
                        if a.result.generation_logits is not None
                    ]
                else:
                    # The shape of generation logits
                    #   (num_sequences, seq_len, vocab_size) in non-streaming
                    #   (seq_len, num_sequences, vocab_size) in streaming
                    seq_dim = 0 if streaming else 1
                    max_out_len = max(
                        response.result.generation_logits.size(seq_dim)
                        for response in responses
                        if response.result.generation_logits is not None)
                    vocab_size = responses[0].result.generation_logits.size(-1)
                    if not streaming:
                        gen_shape = (num_sequences, max_out_len, vocab_size)
                    elif streaming and return_all_generated_tokens:
                        gen_shape = (max_out_len, num_sequences, vocab_size)
                    else:
                        # streaming and not return_all_generated_tokens
                        gen_shape = (1, num_sequences, vocab_size)
                    logits_dtype = responses[0].result.generation_logits.dtype
                    gen_logits = torch.zeros((batch_size, *gen_shape),
                                             dtype=logits_dtype,
                                             device=cuda_device)

                    for response in responses:
                        logits = response.result.generation_logits
                        if logits is None:
                            continue
                        seq_len = logits.size(seq_dim)

                        batch_idx = request_ids.index(response.request_id)
                        seq_idx = response.result.sequence_index
                        if streaming:
                            if is_beam_search:
                                # WAR: gen_logits contains all beams, clipping
                                # the first n beams as a postprocessing.
                                gen_logits[batch_idx, :seq_len,
                                           ...] = logits[:, :num_sequences, :]
                            else:
                                gen_logits[batch_idx, :seq_len, seq_idx,
                                           ...] = logits[:, 0, :]
                        else:
                            if is_beam_search:
                                gen_logits[batch_idx, :, :seq_len, ...] = logits
                            else:
                                gen_logits[batch_idx, seq_idx, :seq_len,
                                           ...] = logits[0]
                outputs['generation_logits'] = gen_logits

            if output_log_probs:
                max_log_probs_len = max(
                    len(lprobs) for response in responses
                    for lprobs in response.result.log_probs)
                log_probs = torch.zeros(
                    (batch_size, num_sequences, max_log_probs_len),
                    dtype=torch.float32)
                for response in responses:
                    batch_idx = request_ids.index(response.request_id)
                    if is_beam_search:
                        for beam_idx, lprobs in enumerate(
                                response.result.log_probs):
                            log_probs[batch_idx,
                                      beam_idx, :len(lprobs)] = torch.tensor(
                                          lprobs)
                    else:
                        seq_idx = response.result.sequence_index
                        lprobs = response.result.log_probs[0]
                        log_probs[batch_idx,
                                  seq_idx, :len(lprobs)] = torch.tensor(lprobs)
                assert isinstance(log_probs, torch.Tensor)
                outputs['log_probs'] = log_probs.to(cuda_device)

            if output_cum_log_probs:
                cum_log_probs = torch.zeros((batch_size, num_sequences),
                                            dtype=torch.float32)
                for response in responses:
                    if response.result.cum_log_probs is None:
                        continue
                    batch_idx = request_ids.index(response.request_id)
                    clprobs = torch.tensor(response.result.cum_log_probs)
                    if is_beam_search:
                        cum_log_probs[batch_idx, :] = clprobs
                    else:
                        seq_idx = response.result.sequence_index
                        cum_log_probs[batch_idx, seq_idx] = clprobs
                outputs['cum_log_probs'] = cum_log_probs.to(cuda_device)

            outputs = self._prepare_outputs(outputs, input_lengths)
        else:
            outputs = output_ids

        return outputs

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
from pathlib import Path
from typing import List, Optional, Union

import torch

from .. import profiler
from ..bindings import (DataType, GptJsonConfig, KVCacheType, ModelConfig,
                        WorldConfig)
from ..bindings import executor as trtllm
from ..builder import EngineConfig
from ..logger import logger
from ..mapping import Mapping
from .generation import (LogitsProcessor, LoraManager, SamplingConfig,
                         StoppingCriteria)
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
        self.mapping = Mapping(world_size=world_config.size,
                               rank=world_config.rank,
                               gpus_per_node=world_config.gpus_per_node,
                               tp_size=world_config.tensor_parallelism,
                               pp_size=world_config.pipeline_parallelism)
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
        medusa_choices: list[list[int]] | None = None,
        lookahead_config: list[int] | None = None,
        debug_mode: bool = False,
        lora_ckpt_source: str = "hf",
        gpu_weights_percent: float = 1,
        max_tokens_in_paged_kv_cache: int | None = None,
        kv_cache_enable_block_reuse: bool = False,
        enable_chunked_context: bool = False,
        is_enc_dec: bool = False,
        multi_block_mode: bool = True,
        enable_context_fmha_fp32_acc: Optional[bool] = None
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
            debug_mode (bool):
                Whether or not to turn on the debug mode.
            medusa_choices (List[List[int]]):
                Medusa choices to use when in Medusa decoding.
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
        Returns:
            ModelRunnerCpp: An instance of ModelRunnerCpp.
        """
        extended_runtime_perf_knob_config = trtllm.ExtendedRuntimePerfKnobConfig(
        )
        if multi_block_mode is not None:
            extended_runtime_perf_knob_config.multi_block_mode = multi_block_mode
        if enable_context_fmha_fp32_acc is not None:
            extended_runtime_perf_knob_config.enable_context_fmha_fp32_acc = enable_context_fmha_fp32_acc

        if is_enc_dec:
            encoder_config_path = Path(engine_dir) / "encoder" / "config.json"
            encoder_json_config = GptJsonConfig.parse_file(encoder_config_path)
            encoder_json_config.model_config
            decoder_config_path = Path(engine_dir) / "decoder" / "config.json"
            decoder_json_config = GptJsonConfig.parse_file(decoder_config_path)
            decoder_model_config = decoder_json_config.model_config
            use_kv_cache = decoder_model_config.kv_cache_type != KVCacheType.DISABLED

            if not use_kv_cache:
                assert max_output_len == 1 or max_output_len is None, 'Disabled KV cache is intended for context phase only now.'

            tp_size = decoder_json_config.tensor_parallelism
            pp_size = decoder_json_config.pipeline_parallelism
            gpus_per_node = decoder_json_config.gpus_per_node
            world_config = WorldConfig.mpi(tensor_parallelism=tp_size,
                                           pipeline_parallelism=pp_size,
                                           gpus_per_node=gpus_per_node)
            assert rank == world_config.rank

            profiler.start('load tensorrt_llm engine')

            kv_cache_config = trtllm.KvCacheConfig(
                free_gpu_memory_fraction=kv_cache_free_gpu_memory_fraction /
                2,  # hardcoded as half self kv & half cross kv for now
                max_attention_window=max_attention_window_size,
                sink_token_length=sink_token_length)

            executor = trtllm.Executor(
                Path(engine_dir) / "encoder",
                Path(engine_dir) / "decoder", trtllm.ModelType.ENCODER_DECODER,
                trtllm.ExecutorConfig(max_beam_width=max_beam_width,
                                      kv_cache_config=kv_cache_config,
                                      gpu_weights_percent=gpu_weights_percent,
                                      extended_runtime_perf_knob_config=
                                      extended_runtime_perf_knob_config))

            profiler.stop('load tensorrt_llm engine')

            loading_time = profiler.elapsed_time_in_sec(
                "load tensorrt_llm engine")
            logger.info(f'Load engine takes: {loading_time} sec')

            return cls(executor,
                       max_batch_size=max_batch_size,
                       max_input_len=max_input_len,
                       max_seq_len=max_input_len + max_output_len,
                       max_beam_width=max_beam_width,
                       model_config=decoder_model_config,
                       world_config=world_config,
                       use_kv_cache=use_kv_cache)

        config_path = Path(engine_dir) / "config.json"
        json_config = GptJsonConfig.parse_file(config_path)
        model_config = json_config.model_config
        use_kv_cache = model_config.kv_cache_type != KVCacheType.DISABLED

        if not use_kv_cache:
            assert max_output_len == 1 or max_output_len is None, 'Disabled KV cache is intended for context phase only now.'

        # Note: Parallel configuration will be fetched automatically from trtllm.Executor constructor
        # by inspecting the json file. These lines serve the purpose of serving vocab_size_padded and
        # num_layers properties.
        tp_size = json_config.tensor_parallelism
        pp_size = json_config.pipeline_parallelism
        gpus_per_node = json_config.gpus_per_node
        world_config = WorldConfig.mpi(tensor_parallelism=tp_size,
                                       pipeline_parallelism=pp_size,
                                       gpus_per_node=gpus_per_node)
        assert rank == world_config.rank

        if model_config.use_lora_plugin and rank == 0:
            engine_config = EngineConfig.from_json_file(
                f"{engine_dir}/config.json")
            lora_manager = LoraManager()
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
                                            runtime_mapping=None,
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

        profiler.start('load tensorrt_llm engine')

        kv_cache_config = trtllm.KvCacheConfig(
            free_gpu_memory_fraction=kv_cache_free_gpu_memory_fraction,
            max_attention_window=max_attention_window_size,
            sink_token_length=sink_token_length,
            max_tokens=max_tokens_in_paged_kv_cache,
            enable_block_reuse=kv_cache_enable_block_reuse)

        decoding_config = trtllm.DecodingConfig()
        if medusa_choices is not None:
            decoding_config.medusa_choices = medusa_choices
            if multi_block_mode is not None:
                multi_block_mode = False  # Medusa doesn't support multi-block mode.

        if lookahead_config is not None:
            [w, n, g] = lookahead_config
            decoding_config.lookahead_decoding_config = trtllm.LookaheadDecodingConfig(
                w, n, g)

        if max_batch_size is None:
            max_batch_size = model_config.max_batch_size
        else:
            assert max_batch_size <= model_config.max_batch_size
        if max_input_len is None:
            max_input_len = model_config.max_input_len
        # NOTE{pengyunl}: remove assertion here for temp fix,
        # model_config.max_input_len is not the upper bound of input length.
        # If runtime max_input_len is not properly set,
        # C++ runtime will throw an error when fetching new requests
        if max_output_len is None:
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
            debug_tensor_names: List[
                str] = None  # modify this list for specific tensor dump
            debug_config = trtllm.DebugConfig(
                debug_input_tensors=True,
                debug_output_tensors=True,
                debug_tensor_names=debug_tensor_names)

        trtllm_config = trtllm.ExecutorConfig(
            max_beam_width=max_beam_width,
            kv_cache_config=kv_cache_config,
            decoding_config=decoding_config,
            peft_cache_config=peft_cache_config,
            debug_config=debug_config,
            gpu_weights_percent=gpu_weights_percent)
        trtllm_config.enable_chunked_context = enable_chunked_context
        trtllm_config.extended_runtime_perf_knob_config = extended_runtime_perf_knob_config

        executor = trtllm.Executor(engine_dir, trtllm.ModelType.DECODER_ONLY,
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
                      sampling_config: trtllm.SamplingConfig, max_new_tokens):
        batch_size = len(batch_input_ids)
        if batch_size > self.max_batch_size:
            raise RuntimeError(
                f"Input batch size ({batch_size}) exceeds the engine or specified limit ({self.max_batch_size})"
            )
        input_lengths = [len(x) for x in batch_input_ids]
        max_length = max(input_lengths)
        if max_length > self.max_input_len:
            raise RuntimeError(
                f"Maximum input length ({max_length}) exceeds the engine or specified limit ({self.max_input_len})"
            )
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
            self.world_config.pipeline_parallelism)

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
            sampling_config: Optional[SamplingConfig] = None,
            lora_uids: Optional[list] = None,
            lookahead_config: list[int] | None = None,
            streaming: bool = False,
            stopping_criteria: Optional[StoppingCriteria] = None,
            logits_processor: Optional[LogitsProcessor] = None,
            max_new_tokens: int = 1,
            end_id: int | None = None,
            pad_id: int | None = None,
            bad_words_list: list[list[int]] | None = None,
            stop_words_list: list[list[int]] | None = None,
            return_dict: bool = False,
            output_sequence_lengths: bool = False,
            output_log_probs: bool = False,
            output_cum_log_probs: bool = False,
            prompt_table: Optional[Union[str, torch.Tensor]] = None,
            prompt_tasks: Optional[str] = None,
            input_token_extra_ids: List[List[int]] = None,
            return_all_generated_tokens: bool = False,
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
            logits_processor (LogitsProcessor):
                Custom logits processors.
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
        if logits_processor is not None:
            raise RuntimeError(
                "Logits processor is not supported in C++ session.")

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
                "num_beams", "top_k", "top_p", "top_p_min", "top_p_reset_ids",
                "top_p_decay", "temperature", "min_tokens",
                "beam_search_diversity_rate", "repetition_penalty",
                "presence_penalty", "frequency_penalty", "length_penalty",
                "early_stopping", "no_repeat_ngram_size", "random_seed"
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

            sampling_config = trtllm.SamplingConfig(**sampling_params)
        else:
            sampling_config = copy.deepcopy(sampling_config)

        self._check_inputs(
            encoder_input_ids_list if encoder_input_ids else
            batch_input_ids_list, sampling_config, max_new_tokens)

        output_config = trtllm.OutputConfig(
            return_context_logits=self.gather_context_logits,
            return_generation_logits=self.gather_generation_logits,
            return_log_probs=output_log_probs,
        )

        prompt_tuning_configs = self._prepare_ptuning_executor(
            batch_input_ids_list, prompt_table, prompt_tasks,
            input_token_extra_ids)

        stop_words_list = self._prepare_words_list(stop_words_list,
                                                   len(batch_input_ids_list))
        bad_words_list = self._prepare_words_list(bad_words_list,
                                                  len(batch_input_ids_list))

        lora_configs = self._prepare_lora_configs(lora_uids,
                                                  len(batch_input_ids_list))
        request_lookahead_config = None
        if lookahead_config is not None:
            [w, n, g] = lookahead_config
            request_lookahead_config = trtllm.LookaheadDecodingConfig(w, n, g)

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
                max_tokens=max_new_tokens,
                pad_id=pad_id,
                end_id=end_id,
                stop_words=stop_words,
                bad_words=bad_words,
                sampling_config=sampling_config,
                lookahead_config=request_lookahead_config,
                streaming=streaming,
                output_config=output_config,
                prompt_tuning_config=prompt_tuning_config,
                lora_config=lora_config,
                return_all_generated_tokens=return_all_generated_tokens) for i,
            (input_ids, stop_words, bad_words, prompt_tuning_config,
             lora_config) in enumerate(
                 zip(batch_input_ids_list, stop_words_list, bad_words_list,
                     prompt_tuning_configs, lora_configs))
        ]

        request_ids = self.session.enqueue_requests(requests)
        if not streaming:
            return self._initialize_and_fill_output(request_ids, end_id,
                                                    return_dict,
                                                    output_sequence_lengths,
                                                    output_log_probs,
                                                    output_cum_log_probs,
                                                    batch_input_ids, streaming)
        else:
            return self._stream(request_ids, end_id, return_dict,
                                output_sequence_lengths, output_log_probs,
                                output_cum_log_probs, batch_input_ids,
                                batch_input_ids_list, streaming,
                                return_all_generated_tokens)

    def _prepare_words_list(self, words_list: List[List[List[int]]],
                            batch_size: int):
        if words_list is None:
            return [None] * batch_size
        return words_list

    def _prepare_ptuning_executor(self, batch_input_ids_list, prompt_table,
                                  prompt_tasks, input_token_extra_ids):
        if input_token_extra_ids:
            assert len(batch_input_ids_list) == len(input_token_extra_ids), \
                f"Batch size of input_token_extra_ids ({len(input_token_extra_ids)}) must be the same as input batch size ({len(batch_input_ids_list)})"
        prompt_tuning_configs = len(batch_input_ids_list) * [None]
        if prompt_table is not None:
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

    def _initialize_and_fill_output(self, request_ids, end_id, return_dict,
                                    output_sequence_lengths, output_log_probs,
                                    output_cum_log_probs, batch_input_ids,
                                    streaming):
        output_ids = [[] for _ in range(len(request_ids))]
        for reqid_pos in range(len(request_ids)):
            output_ids[reqid_pos] = [[] for _ in range(self.max_beam_width)]

        multi_responses = self.session.await_responses(request_ids)
        responses = [
            response for responses in multi_responses for response in responses
        ]

        return self._fill_output(responses, output_ids, end_id, return_dict,
                                 output_sequence_lengths, output_log_probs,
                                 output_cum_log_probs, batch_input_ids, [],
                                 streaming, request_ids, False)

    def _stream(self, request_ids, end_id, return_dict, output_sequence_lengths,
                output_log_probs, output_cum_log_probs, batch_input_ids,
                batch_input_ids_list, streaming, return_all_generated_tokens):
        output_ids = [[] for _ in range(len(request_ids))]
        for reqid_pos in range(len(request_ids)):
            output_ids[reqid_pos] = [
                copy.deepcopy(batch_input_ids_list[reqid_pos])
                for _ in range(self.max_beam_width)
            ]

        finished_reqs = 0
        while finished_reqs < len(request_ids):
            responses = self.session.await_responses()

            for response in responses:
                if response.result.is_final:
                    finished_reqs += 1

            yield self._fill_output(responses, output_ids, end_id, return_dict,
                                    output_sequence_lengths, output_log_probs,
                                    output_cum_log_probs, batch_input_ids,
                                    batch_input_ids_list, streaming,
                                    request_ids, return_all_generated_tokens)

    def _fill_output(self, responses, output_ids, end_id, return_dict,
                     output_sequence_lengths, output_log_probs,
                     output_cum_log_probs, batch_input_ids,
                     batch_input_ids_list, streaming, request_ids,
                     return_all_generated_tokens):
        cuda_device = torch.device("cuda")

        for response in responses:
            if response.has_error():
                raise RuntimeError(response.error_msg)

            reqid_pos = request_ids.index(response.request_id)
            for beam, output_tokens in enumerate(
                    response.result.output_token_ids):
                if return_all_generated_tokens:
                    output_ids[reqid_pos][
                        beam] = batch_input_ids_list[reqid_pos] + output_tokens
                else:
                    output_ids[reqid_pos][beam] += output_tokens

        sequence_lengths = []
        for output in output_ids:
            sequence_lengths.append([len(a) for a in output])

        if streaming:
            output_ids = copy.deepcopy(output_ids)

        for beam in output_ids:
            for output_tokens in beam:
                output_tokens += (self.max_seq_len -
                                  len(output_tokens)) * [end_id]

        output_ids = torch.tensor(output_ids,
                                  dtype=torch.int32,
                                  device=cuda_device)

        if return_dict:
            outputs = {'output_ids': output_ids}
            if output_sequence_lengths:
                outputs['sequence_lengths'] = torch.tensor(sequence_lengths,
                                                           dtype=torch.int32,
                                                           device=cuda_device)
            if self.gather_context_logits:
                outputs['context_logits'] = [
                    a.result.context_logits.cuda() for a in responses
                    if a.result.context_logits is not None
                ]
                # Pad context_logits into a rectangle
                max_input_length = max(a.shape[0]
                                       for a in outputs['context_logits'])
                for i, a in enumerate(outputs['context_logits']):
                    pad_length = max_input_length - a.shape[0]
                    outputs['context_logits'][i] = torch.nn.functional.pad(
                        a, [0, 0, 0, pad_length])
                outputs['context_logits'] = torch.stack(
                    outputs['context_logits'])
            if self.gather_generation_logits:
                outputs['generation_logits'] = [
                    a.result.generation_logits.cuda() for a in responses
                    if a.result.generation_logits is not None
                ]
                outputs['generation_logits'] = torch.stack(
                    outputs['generation_logits'])
            if output_log_probs:
                outputs['log_probs'] = [
                    a.result.log_probs for a in responses
                    if a.result.log_probs is not None
                ]
                # Pad log_probs into a rectangle
                max_seq_len = max(
                    len(a) for beam_list in outputs['log_probs']
                    for a in beam_list)
                for i, a in enumerate(outputs['log_probs']):
                    for j, b in enumerate(a):
                        pad_length = max_seq_len - len(b)
                        outputs['log_probs'][i][j] = b + [0.0] * pad_length
                outputs['log_probs'] = torch.tensor(outputs['log_probs'],
                                                    device=cuda_device)
            if output_cum_log_probs:
                outputs['cum_log_probs'] = [
                    a.result.cum_log_probs for a in responses
                    if a.result.cum_log_probs is not None
                ]
                outputs['cum_log_probs'] = torch.tensor(
                    outputs['cum_log_probs'], device=cuda_device)
            input_lengths = torch.tensor([x.size(0) for x in batch_input_ids],
                                         dtype=torch.int32,
                                         device=cuda_device)
            outputs = self._prepare_outputs(outputs, input_lengths)
        else:
            outputs = output_ids
        return outputs

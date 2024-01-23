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
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorrt as trt
import torch

from .. import profiler
from .._utils import mpi_world_size
from ..logger import logger
from ..mapping import Mapping
from ..quantization import QuantMode
from .engine import Engine, get_engine_version
from .generation import (ChatGLMGenerationSession, GenerationSession,
                         LogitsProcessor, LoraManager,
                         MambaLMHeadModelGenerationSession, ModelConfig,
                         QWenForCausalLMGenerationSession, SamplingConfig,
                         StoppingCriteria)


def get_engine_name(model: str, dtype: str, tp_size: int, pp_size: int,
                    rank: int) -> str:
    """
    Get the serialized engine file name.

    Args:
        model (str):
            Model name, e.g., bloom, gpt.
        dtype (str):
            Data type, e.g., float32, float16, bfloat16,
        tp_size (int):
            The size of tensor parallel.
        pp_size (int):
            The size of pipeline parallel.
        rank (int):
            The rank id.

    Returns:
        str: The serialized engine file name.
    """
    if pp_size == 1:
        return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)
    return '{}_{}_tp{}_pp{}_rank{}.engine'.format(model, dtype, tp_size,
                                                  pp_size, rank)


def read_config(config_path: Path) -> Tuple[ModelConfig, dict]:
    """
    Read the engine config file and create a ModelConfig instance, return the ModelConfig instance
    and other config fields in a dict.

    Args:
        config_path (Path):
            The path of engine config file.

    Returns:
        Tuple[ModelConfig, dict]: A ModelConfig instance and other config fields.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
        return _builder_to_model_config(config)


def _builder_to_model_config(config: dict) -> Tuple[ModelConfig, dict]:
    builder_config = config['builder_config']
    model_name = builder_config['name']
    dtype = builder_config['precision']
    tp_size = builder_config['tensor_parallel']
    pp_size = builder_config.get('pipeline_parallel', 1)
    world_size = tp_size * pp_size
    assert world_size == mpi_world_size(), \
        f'Engine world size ({tp_size} * {pp_size}) != Runtime world size ({mpi_world_size()})'

    num_heads = builder_config['num_heads']
    assert num_heads % tp_size == 0, \
        f"The number of heads ({num_heads}) is not a multiple of tp_size ({tp_size})"
    num_kv_heads = builder_config.get('num_kv_heads', num_heads)
    # TODO: multi_query_mode should be removed
    multi_query_mode = builder_config.get('multi_query_mode', False)
    if multi_query_mode:
        logger.warning(
            "`multi_query_mode` config is deprecated. Please rebuild the engine."
        )
    # num_kv_heads, if exists in config, should override multi_query_mode
    if multi_query_mode and ('num_kv_heads' not in builder_config):
        num_kv_heads = 1
    num_heads = num_heads // tp_size
    num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size

    hidden_size = builder_config['hidden_size'] // tp_size
    vocab_size = builder_config['vocab_size']
    num_layers = builder_config['num_layers']

    cross_attention = builder_config.get('cross_attention', False)
    has_position_embedding = builder_config.get('has_position_embedding', True)
    has_token_type_embedding = builder_config.get('has_token_type_embedding',
                                                  False)
    gather_context_logits = builder_config.get('gather_context_logits', False)
    gather_generation_logits = builder_config.get('gather_generation_logits',
                                                  False)
    max_prompt_embedding_table_size = builder_config.get(
        'max_prompt_embedding_table_size', 0)
    quant_mode = QuantMode(builder_config.get('quant_mode', 0))
    lora_target_modules = builder_config.get('lora_target_modules')
    lora_hf_modules_to_trtllm_modules = builder_config.get(
        'hf_modules_to_trtllm_modules')
    lora_trtllm_modules_to_hf_modules = builder_config.get(
        'trtllm_modules_to_hf_modules')
    max_medusa_token_len = builder_config.get('max_medusa_token_len', 0)
    num_medusa_heads = builder_config.get('num_medusa_heads', 0)

    plugin_config = config['plugin_config']
    use_gpt_attention_plugin = bool(plugin_config['gpt_attention_plugin'])
    remove_input_padding = plugin_config['remove_input_padding']
    paged_kv_cache = plugin_config['paged_kv_cache']
    tokens_per_block = plugin_config['tokens_per_block']
    use_custom_all_reduce = plugin_config.get('use_custom_all_reduce', False)
    lora_plugin = plugin_config.get('lora_plugin')
    use_context_fmha_for_generation = plugin_config.get(
        'use_context_fmha_for_generation')

    model_config = ModelConfig(
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        gpt_attention_plugin=use_gpt_attention_plugin,
        remove_input_padding=remove_input_padding,
        model_name=model_name,
        paged_kv_cache=paged_kv_cache,
        cross_attention=cross_attention,
        has_position_embedding=has_position_embedding,
        has_token_type_embedding=has_token_type_embedding,
        tokens_per_block=tokens_per_block,
        max_prompt_embedding_table_size=max_prompt_embedding_table_size,
        quant_mode=quant_mode,
        gather_context_logits=gather_context_logits,
        gather_generation_logits=gather_generation_logits,
        dtype=dtype,
        use_custom_all_reduce=use_custom_all_reduce,
        lora_plugin=lora_plugin,
        lora_target_modules=lora_target_modules,
        use_context_fmha_for_generation=use_context_fmha_for_generation,
        hf_modules_to_trtllm_modules=lora_hf_modules_to_trtllm_modules,
        trtllm_modules_to_hf_modules=lora_trtllm_modules_to_hf_modules,
        num_medusa_heads=num_medusa_heads,
        max_medusa_tokens=max_medusa_token_len,
    )

    other_config = {
        'world_size': world_size,
        'tp_size': tp_size,
        'pp_size': pp_size,
        'max_batch_size': builder_config['max_batch_size'],
        'max_input_len': builder_config['max_input_len'],
        'max_output_len': builder_config['max_output_len'],
        'max_beam_width': builder_config['max_beam_width']
    }
    return model_config, other_config


class ModelRunnerMixin:

    def _check_inputs(self, batch_input_ids: List[torch.Tensor],
                      sampling_config: SamplingConfig):
        batch_size = len(batch_input_ids)
        if batch_size > self.max_batch_size:
            raise RuntimeError(
                f"Input batch size ({batch_size}) exceeds the engine or specified limit ({self.max_batch_size})"
            )
        input_lengths = [x.size(0) for x in batch_input_ids]
        max_length = max(input_lengths)
        if max_length > self.max_input_len:
            raise RuntimeError(
                f"Maximum input length ({max_length}) exceeds the engine or specified limit ({self.max_input_len})"
            )
        if max_length + sampling_config.max_new_tokens > self.max_seq_len:
            raise RuntimeError(
                f"Maximum input length ({max_length}) + maximum new tokens ({sampling_config.max_new_tokens}) exceeds the engine or specified limit ({self.max_seq_len})"
            )
        if sampling_config.num_beams > self.max_beam_width:
            raise RuntimeError(
                f"Num beams ({sampling_config.num_beams}) exceeds the engine or specified limit ({self.max_beam_width})"
            )

    def _prepare_inputs(self, batch_input_ids: List[torch.Tensor],
                        pad_id: int) -> Tuple[torch.Tensor]:
        # Cast to int32
        batch_input_ids = [x.type(torch.int32) for x in batch_input_ids]
        input_lengths = [x.size(0) for x in batch_input_ids]
        max_length = max(input_lengths)

        if self.remove_input_padding:
            batch_input_ids = torch.concat(batch_input_ids)
        else:
            # Right padding for trt-llm
            paddings = [
                torch.ones(max_length - l, dtype=torch.int32) * pad_id
                for l in input_lengths
            ]
            batch_input_ids = [
                torch.cat([x, pad]) for x, pad in zip(batch_input_ids, paddings)
            ]
            batch_input_ids = torch.stack(batch_input_ids)
        input_lengths = torch.tensor(input_lengths, dtype=torch.int32)
        return batch_input_ids, input_lengths

    def _prepare_outputs(self, outputs: Optional[dict],
                         input_lengths: torch.Tensor) -> dict:
        if outputs is not None:
            batch_size = input_lengths.size(0)
            if 'context_logits' in outputs:
                context_logits = outputs['context_logits']
                if self.remove_input_padding:
                    context_logits = context_logits.flatten(end_dim=-2)

                    seg_points = [0] + input_lengths.cumsum(dim=0).tolist()
                    context_logits = [
                        context_logits[s:e]
                        for s, e in zip(seg_points[:-1], seg_points[1:])
                    ]
                else:
                    context_logits = [
                        context_logits[bidx, :input_lengths[bidx]]
                        for bidx in range(batch_size)
                    ]
                outputs['context_logits'] = context_logits

            if 'generation_logits' in outputs and isinstance(
                    self.session, GenerationSession):
                generation_logits = torch.stack(outputs['generation_logits'],
                                                dim=1)
                batch_x_beam, max_gen_len, voc_size = generation_logits.size()
                num_beams = batch_x_beam // batch_size
                generation_logits = generation_logits.view(
                    batch_size, num_beams, max_gen_len, voc_size)
                outputs['generation_logits'] = generation_logits

        return outputs

    def _prepare_ptuning(self, prompt_table_path: str, tasks: str,
                         batch_size: int):
        if self.max_prompt_embedding_table_size == 0:
            return {}

        if prompt_table_path is not None:
            prompt_table = torch.from_numpy(
                np.load(prompt_table_path)).to(dtype=self.dtype)
            _, task_vocab_size, hidden_size = prompt_table.size()
            task_vocab_size = torch.tensor([task_vocab_size], dtype=torch.int32)
            prompt_table = prompt_table.view(-1, hidden_size)
        else:
            prompt_table = torch.empty([1, self.hidden_size], dtype=self.dtype)
            task_vocab_size = torch.zeros([1], dtype=torch.int32)

        if tasks is not None:
            tasks = torch.tensor([int(t) for t in tasks.split(',')],
                                 dtype=torch.int32)
            assert tasks.size(0) == batch_size, \
                f"Number of supplied tasks ({tasks.size(0)}) must match input batch size ({batch_size})"
        else:
            tasks = torch.zeros([batch_size], dtype=torch.int32)

        if isinstance(self.session, GenerationSession):
            return {
                'prompt_embedding_table': prompt_table.cuda(),
                'tasks': tasks.cuda(),
                'prompt_vocab_size': task_vocab_size.cuda()
            }
        else:
            return {
                'embedding_table': prompt_table.cuda(),
                'tasks': tasks.cuda(),
                'vocab_size': task_vocab_size.cuda()
            }


class ModelRunner(ModelRunnerMixin):
    """
    An interface class that wraps GenerationSession and provides generation methods.
    """

    def __init__(self,
                 session: GenerationSession,
                 max_batch_size: int,
                 max_input_len: int,
                 max_seq_len: int,
                 max_beam_width: int,
                 lora_manager: Optional[LoraManager] = None) -> None:
        """
        Create a ModelRunner instance.
        You are recommended to use the from_dir method to load the engine and create a ModelRunner instance.

        Args:
            session (GenerationSession):
                The TensorRT session created from an engine.
            max_batch_size (int):
                The maximum batch size allowed for the input.
            max_input_len (int):
                The maximum input length allowed for the input.
            max_seq_len (int):
                The maximum sequence length (input + new tokens).
            max_beam_width (int):
                The maximum beam width.
            lora_manager (LoraManager):
                The LoRA manager to handle LoRA weights.
        """
        self.session = session
        self.max_batch_size = max_batch_size
        self.max_input_len = max_input_len
        self.max_seq_len = max_seq_len
        self.max_beam_width = max_beam_width
        self.lora_manager = lora_manager

    @classmethod
    def from_dir(cls,
                 engine_dir: str,
                 lora_dir: Optional[str] = None,
                 rank: int = 0,
                 debug_mode: bool = False,
                 lora_ckpt_source: str = "hf",
                 medusa_choices: List[List[int]] = None) -> 'ModelRunner':
        """
        Create a ModelRunner instance from an engine directory.

        Args:
            engine_dir (str):
                The directory that contains the serialized engine files and config files.
            lora_dir (str):
                The directory that contains LoRA weights.
            rank (int):
                The runtime rank id.
            debug_mode (bool):
                Whether or not to turn on the debug mode.
            medusa_choices (List[List[int]]):
                Medusa choices to use when in Medusa decoding
        Returns:
            ModelRunner: An instance of ModelRunner.
        """
        profiler.start('load tensorrt_llm engine')

        engine_version = get_engine_version(engine_dir)
        # the old engine format
        if engine_version is None:
            engine_dir = Path(engine_dir)
            config_path = engine_dir / "config.json"
            model_config, other_config = read_config(config_path)
            world_size = other_config.pop('world_size')
            tp_size = other_config.pop('tp_size')
            pp_size = other_config.pop('pp_size')
            max_batch_size = other_config.pop('max_batch_size')
            max_input_len = other_config.pop('max_input_len')
            max_output_len = other_config.pop('max_output_len')
            max_beam_width = other_config.pop('max_beam_width')
            runtime_mapping = Mapping(world_size=world_size,
                                      rank=rank,
                                      tp_size=tp_size,
                                      pp_size=pp_size)

            engine_name = get_engine_name(model_config.model_name,
                                          model_config.dtype, tp_size, pp_size,
                                          rank)
            serialize_path = engine_dir / engine_name

            with open(serialize_path, 'rb') as f:
                engine_buffer = f.read()

            if model_config.model_name in ('chatglm_6b', 'glm_10b'):
                session_cls = ChatGLMGenerationSession
            elif model_config.model_name == 'qwen':
                session_cls = QWenForCausalLMGenerationSession
            else:
                session_cls = GenerationSession
        else:
            # the new engine format
            engine = Engine.from_dir(engine_dir, rank)
            pretrained_config = engine.config.pretrained_config
            build_config = engine.config.build_config

            tp_size = pretrained_config.mapping.tp_size
            num_heads = pretrained_config.num_attention_heads // tp_size
            num_kv_heads = pretrained_config.num_key_value_heads
            num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size
            hidden_size = pretrained_config.hidden_size // tp_size

            if pretrained_config.architecture == 'MambaLMHeadModel':
                mamba_d_state = pretrained_config.ssm_cfg['d_state']
                mamba_expand = pretrained_config.ssm_cfg['expand']
                mamba_d_conv = pretrained_config.ssm_cfg['d_conv']
            else:
                mamba_d_state, mamba_expand, mamba_d_conv = 0, 0, 0

            model_config = ModelConfig(
                vocab_size=pretrained_config.vocab_size,
                num_layers=pretrained_config.num_hidden_layers,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                hidden_size=hidden_size,
                gpt_attention_plugin=bool(
                    build_config.plugin_config.gpt_attention_plugin),
                remove_input_padding=build_config.plugin_config.
                remove_input_padding,
                paged_kv_cache=build_config.plugin_config.paged_kv_cache,
                tokens_per_block=build_config.plugin_config.tokens_per_block,
                quant_mode=pretrained_config.quant_mode,
                gather_context_logits=build_config.gather_context_logits,
                gather_generation_logits=build_config.gather_generation_logits,
                dtype=pretrained_config.dtype,
                max_prompt_embedding_table_size=build_config.
                max_prompt_embedding_table_size,
                mamba_d_state=mamba_d_state,
                mamba_expand=mamba_expand,
                mamba_d_conv=mamba_d_conv,
            )
            max_batch_size = build_config.max_batch_size
            max_input_len = build_config.max_input_len
            max_output_len = build_config.max_output_len
            max_beam_width = build_config.max_beam_width
            if pretrained_config.architecture == 'MambaLMHeadModel':
                session_cls = MambaLMHeadModelGenerationSession
            else:
                session_cls = GenerationSession
            engine_buffer = engine.engine
            runtime_mapping = pretrained_config.mapping

        if medusa_choices is not None:
            assert session_cls == GenerationSession, "Medusa is only supported by GenerationSession"
            assert model_config.max_medusa_tokens > 0, \
                "medusa_chioce is specified but model_config.max_medusa_tokens is 0."

        torch.cuda.set_device(rank % runtime_mapping.gpus_per_node)
        session = session_cls(model_config,
                              engine_buffer,
                              runtime_mapping,
                              debug_mode=debug_mode)
        profiler.stop('load tensorrt_llm engine')
        loading_time = profiler.elapsed_time_in_sec("load tensorrt_llm engine")
        logger.info(f'Load engine takes: {loading_time} sec')

        if session.use_lora_plugin:
            lora_manager = LoraManager()
            if lora_dir is not None:
                lora_manager.load_from_ckpt(model_dir=lora_dir,
                                            model_config=model_config,
                                            runtime_mapping=runtime_mapping,
                                            ckpt_source=lora_ckpt_source)
        else:
            lora_manager = None

        return cls(session,
                   max_batch_size,
                   max_input_len,
                   max_input_len + max_output_len,
                   max_beam_width,
                   lora_manager=lora_manager)

    @property
    def dtype(self) -> torch.dtype:
        return self.session.dtype

    @property
    def vocab_size(self) -> int:
        return self.session.vocab_size

    @property
    def vocab_size_padded(self) -> int:
        return self.session.vocab_size_padded

    @property
    def hidden_size(self) -> int:
        return self.session.hidden_size

    @property
    def num_heads(self) -> int:
        return self.session.num_heads

    @property
    def num_layers(self) -> int:
        return self.session.num_layers

    @property
    def max_sequence_length(self) -> int:
        return self.max_seq_len

    @property
    def remove_input_padding(self) -> bool:
        return self.session.remove_input_padding

    @property
    def use_lora_plugin(self) -> bool:
        return self.session.use_lora_plugin

    @property
    def max_prompt_embedding_table_size(self) -> int:
        return self.session.max_prompt_embedding_table_size

    @property
    def gather_context_logits(self) -> bool:
        return self.session.gather_context_logits

    @property
    def gather_generation_logits(self) -> bool:
        return self.session.gather_generation_logits

    def generate(self,
                 batch_input_ids: List[torch.Tensor],
                 sampling_config: Optional[SamplingConfig] = None,
                 prompt_table_path: Optional[str] = None,
                 prompt_tasks: Optional[str] = None,
                 lora_uids: Optional[list] = None,
                 streaming: bool = False,
                 stopping_criteria: Optional[StoppingCriteria] = None,
                 logits_processor: Optional[LogitsProcessor] = None,
                 medusa_choices: Optional[List[List[int]]] = None,
                 **kwargs) -> Union[torch.Tensor, dict]:
        """
        Generates sequences of token ids.
        The generation-controlling parameters are set in the sampling_config; it will be set to a default one if not passed.
        You can override any sampling_config's attributes by passing corresponding parameters.

        Args:
            batch_input_ids (List[torch.Tensor]):
                A list of input id tensors. Each tensor is of shape (sequence_length, ).
            sampling_config (SamplingConfig):
                The sampling configuration to be used as base parametrization for the generation call.
                The passed **kwargs matching the sampling_config's attributes will override them.
                If the sampling_config is not provided, a default will be used.
            prompt_table_path (str):
                The file path of prompt table (.npy format, exported by nemo_prompt_convert.py).
            prompt_tasks (str):
                The prompt tuning task ids for the input batch, in format of comma-separated list (e.g., 0,3,1,0).
            lora_uids (list):
                The uids of LoRA weights for the input batch. Use -1 to disable the LoRA module.
            streaming (bool):
                Whether or not to use streaming mode for generation.
            stopping_criteria (StoppingCriteria):
                Custom stopping criteria.
            logits_processor (LogitsProcessor):
                Custom logits processors.
            medusa_choices (List[List[int]]):
                Medusa decoding choices.
            kwargs (Dict[str, Any]:
                Ad hoc parametrization of sampling_config.
                The passed **kwargs matching the sampling_config's attributes will override them.
        Returns:
            torch.Tensor or dict:
                If return_dict=False, the method returns generated output_ids.
                If return_dict=True, the method returns a dict of output_ids,
                sequence_lengths (if sampling_config.output_sequence_lengths=True),
                context_logits and generation_logits (if self.gather_context_logits=True
                and self.gather_generation_logits=True, respectively).
        """
        # Use sampling_config like HF's generation_config
        if sampling_config is None:
            sampling_config = SamplingConfig(end_id=None, pad_id=None)
        else:
            sampling_config = copy.deepcopy(sampling_config)
        sampling_config.update(**kwargs)
        self._check_inputs(batch_input_ids, sampling_config)

        batch_size = len(batch_input_ids)
        batch_input_ids, input_lengths = self._prepare_inputs(
            batch_input_ids, sampling_config.pad_id)

        self.session.setup(
            batch_size=batch_size,
            max_context_length=input_lengths.max().item(),
            max_new_tokens=sampling_config.max_new_tokens,
            beam_width=sampling_config.num_beams,
            max_attention_window_size=sampling_config.max_attention_window_size,
            sink_token_length=sampling_config.sink_token_length,
            lora_manager=self.lora_manager,
            lora_uids=lora_uids,
            medusa_choices=medusa_choices)

        batch_input_ids = batch_input_ids.cuda()
        input_lengths = input_lengths.cuda()
        ptuning_kwargs = self._prepare_ptuning(prompt_table_path, prompt_tasks,
                                               batch_size)
        outputs = self.session.decode(
            batch_input_ids,
            input_lengths,
            sampling_config,
            stop_words_list=sampling_config.stop_words_list,
            bad_words_list=sampling_config.bad_words_list,
            output_sequence_lengths=sampling_config.output_sequence_lengths,
            return_dict=sampling_config.return_dict,
            streaming=streaming,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            **ptuning_kwargs)
        if sampling_config.return_dict:
            if streaming:
                outputs = (self._prepare_outputs(curr_outputs, input_lengths)
                           for curr_outputs in outputs)
            else:
                outputs = self._prepare_outputs(outputs, input_lengths)
        return outputs

    def serialize_engine(self) -> trt.IHostMemory:
        """
        Serialize the engine.

        Returns:
            bytes: The serialized engine.
        """
        return self.session.runtime._serialize_engine()

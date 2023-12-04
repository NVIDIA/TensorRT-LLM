# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import (ChatGLMGenerationSession, GenerationSession,
                                  ModelConfig, SamplingConfig)


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

    builder_config = config['builder_config']
    model_name = builder_config['name']
    dtype = builder_config['precision']
    tp_size = builder_config['tensor_parallel']
    pp_size = builder_config.get('pipeline_parallel', 1)
    world_size = tp_size * pp_size
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({tp_size} * {pp_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'

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
    gather_all_token_logits = builder_config.get('gather_all_token_logits',
                                                 False)
    max_prompt_embedding_table_size = builder_config.get(
        'max_prompt_embedding_table_size', 0)
    quant_mode = QuantMode(builder_config.get('quant_mode', 0))

    plugin_config = config['plugin_config']
    use_gpt_attention_plugin = bool(plugin_config['gpt_attention_plugin'])
    remove_input_padding = plugin_config['remove_input_padding']
    paged_kv_cache = plugin_config['paged_kv_cache']
    tokens_per_block = plugin_config['tokens_per_block']
    use_custom_all_reduce = plugin_config.get('use_custom_all_reduce', False)

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
        gather_all_token_logits=gather_all_token_logits,
        dtype=dtype,
        use_custom_all_reduce=use_custom_all_reduce)

    other_config = {
        'world_size': world_size,
        'tp_size': tp_size,
        'pp_size': pp_size,
        'max_batch_size': builder_config['max_batch_size'],
        'max_input_len': builder_config['max_input_len']
    }
    return model_config, other_config


class ModelRunner:
    """
    An interface class that wraps GenerationSession and provides generation methods.
    """

    def __init__(self, session: GenerationSession, max_batch_size: int,
                 max_input_len: int) -> None:
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
        """
        self.session = session
        self.max_batch_size = max_batch_size
        self.max_input_len = max_input_len

    @classmethod
    def from_dir(cls,
                 engine_dir: str,
                 rank: int = 0,
                 debug_mode: bool = False) -> 'ModelRunner':
        """
        Create a ModelRunner instance from an engine directory.

        Args:
            engine_dir (str):
                The directory that contains the serialized engine files and config files.
            rank (int):
                The runtime rank id.
            debug_mode (int):
                Whether or not to turn on the debug mode.
        Returns:
            ModelRunner: An instance of ModelRunner.
        """
        # session setup
        engine_dir = Path(engine_dir)
        config_path = engine_dir / "config.json"
        model_config, other_config = read_config(config_path)
        world_size = other_config.pop('world_size')
        tp_size = other_config.pop('tp_size')
        pp_size = other_config.pop('pp_size')
        runtime_mapping = tensorrt_llm.Mapping(world_size=world_size,
                                               rank=rank,
                                               tp_size=tp_size,
                                               pp_size=pp_size)
        torch.cuda.set_device(rank % runtime_mapping.gpus_per_node)

        engine_name = get_engine_name(model_config.model_name,
                                      model_config.dtype, tp_size, pp_size,
                                      rank)
        serialize_path = engine_dir / engine_name

        profiler.start('load tensorrt_llm engine')
        with open(serialize_path, 'rb') as f:
            engine_buffer = f.read()

        if model_config.model_name in ('chatglm_6b', 'glm_10b'):
            session_cls = ChatGLMGenerationSession
        else:
            session_cls = GenerationSession
        session = session_cls(model_config,
                              engine_buffer,
                              runtime_mapping,
                              debug_mode=debug_mode)
        profiler.stop('load tensorrt_llm engine')
        loading_time = profiler.elapsed_time_in_sec("load tensorrt_llm engine")
        logger.info(f'Load engine takes: {loading_time} sec')

        return cls(session, **other_config)

    @property
    def remove_input_padding(self) -> bool:
        return self.session.remove_input_padding

    def _prepare_inputs(self, batch_input_ids: List[torch.Tensor],
                        pad_id: int) -> Tuple[torch.Tensor]:
        # Remove potential additional dim, cast to int32
        batch_input_ids = [
            x.flatten().type(torch.int32) for x in batch_input_ids
        ]
        input_lengths = [x.size(0) for x in batch_input_ids]
        max_length = max(input_lengths)
        if max_length > self.max_input_len:
            raise RuntimeError(
                f"Maximum input length ({max_length}) exceeds the engine limit ({self.max_input_len})"
            )
        batch_size = len(batch_input_ids)
        if batch_size > self.max_batch_size:
            raise RuntimeError(
                f"Input batch size ({batch_size}) exceeds the engine limit ({self.max_batch_size})"
            )

        if self.remove_input_padding:
            batch_input_ids = torch.concat(batch_input_ids).unsqueeze(0)
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
        if outputs is not None and 'context_logits' in outputs:
            batch_size = input_lengths.size(0)
            context_logits = outputs['context_logits']
            if self.remove_input_padding:
                context_logits = context_logits.flatten(end_dim=1)

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

        return outputs

    def generate(self,
                 batch_input_ids: List[torch.Tensor],
                 sampling_config: Optional[SamplingConfig] = None,
                 **kwargs) -> Union[torch.Tensor, dict]:
        """
        Generates sequences of token ids.
        The generation-controlling parameters are set in the sampling_config; it will be set to a default one if not passed.
        You can override any sampling_config's attributes by passing corresponding parameters.

        Args:
            batch_input_ids (List[torch.Tensor]):
                A list of input id tensors. Each tensor is of shape (sequence_length, ).
            sampling_config (Optional[SamplingConfig]):
                The sampling configuration to be used as base parametrization for the generation call.
                The passed **kwargs matching the sampling_config's attributes will override them.
                If the sampling_config is not provided, a default will be used.
            kwargs (Dict[str, Any]:
                Ad hoc parametrization of sampling_config.
                The passed **kwargs matching the sampling_config's attributes will override them.
        Returns:
            torch.Tensor or dict:
                If return_dict=False, the method returns generated output_ids.
                If return_dict=True, the method returns a dict of output_ids,
                sequence_lengths (if sampling_config.output_sequence_lengths=True),
                context_logits and generation_logits (if self.session.gather_all_token_logits=True).
        """
        # Use sampling_config like HF's generation_config
        if sampling_config is None:
            sampling_config = SamplingConfig(end_id=None, pad_id=None)
        else:
            sampling_config = copy.deepcopy(sampling_config)
        sampling_config.update(**kwargs)

        batch_size = len(batch_input_ids)
        batch_input_ids, input_lengths = self._prepare_inputs(
            batch_input_ids, sampling_config.pad_id)

        self.session.setup(
            batch_size=batch_size,
            max_context_length=input_lengths.max().item(),
            max_new_tokens=sampling_config.max_new_tokens,
            beam_width=sampling_config.num_beams,
            max_kv_cache_length=sampling_config.max_kv_cache_length)

        batch_input_ids = batch_input_ids.cuda()
        input_lengths = input_lengths.cuda()
        outputs = self.session.decode(
            batch_input_ids,
            input_lengths,
            sampling_config,
            output_sequence_lengths=sampling_config.output_sequence_lengths,
            return_dict=sampling_config.return_dict)
        if sampling_config.return_dict:
            outputs = self._prepare_outputs(outputs, input_lengths)
        return outputs

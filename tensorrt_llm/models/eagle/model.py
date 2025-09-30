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

from collections import OrderedDict
from typing import Optional, Union

import numpy as np
import tensorrt as trt
from tqdm import tqdm

from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.generation_mixin import GenerationMixin
from tensorrt_llm.models.llama.model import LLaMAForCausalLM, LLaMAModel
from tensorrt_llm.models.model_weights_loader import ModelWeightsLoader

from ..._common import default_net, default_trtnet
from ..._utils import pad_vocab_size
from ...bindings import KVCacheType
from ...functional import (Tensor, _create_tensor, cast, concat,
                           gather_last_token_logits, index_select, shape)
from ...layers import AttentionParams, ColumnLinear, SpecDecodingParams
from ...module import Module, ModuleList
from ...plugin import TRT_LLM_PLUGIN_NAMESPACE
from ..modeling_utils import QuantConfig
from .config import EagleConfig


class TreeParams(object):

    def __init__(self, paths: Tensor = None):
        self.paths = paths  # on GPU


def eagle_sample_and_accept_draft_plugin(lm_logits: Tensor = None,
                                         draft_tokens: Tensor = None,
                                         draft_lens: Tensor = None,
                                         eagle_temperature: Tensor = None,
                                         rand_data_validation: Tensor = None,
                                         posterior_alpha: Tensor = None,
                                         posterior_threshold: Tensor = None,
                                         tree_params: TreeParams = None,
                                         greedy_sampling: Tensor = None,
                                         use_dynamic_tree: Tensor = None):
    '''
    Takes input logits and samples golden token + predictions from draft tokens.
    Runs acceptance algorithm to accept draft tokens.
    When greedy_sampling is True, all decoding is done using Top1 and token equality is used
    for acceptance. Otherwise, typical acceptance and multinomial samplings are used.

    Visit tests/model/eagle/test_sample_accept_draft_tokens.py for input/output examples.

    Parameters:
        lm_logits : Tensor
            [num_tokens, vocab_size]
            Logits produced by the base model.

        draft_tokens : Tensor
            [batch_size, max_decoding_draft_tokens]
            Input draft tokens. Only the first draft_lens[bi] tokens are relevant for bi'th row.

        draft_lens : Tensor
            [batch_size]
            Lengths of the draft_tokens. 0 for context request. Actual draft length for generation requests.

        eagle_temperature : Tensor
            [batch_size]
            Temperature of the decoding.

        rand_data_validation : Tensor
            [batch_size, max_decoding_tokens]
            Random data for multinomial sampling.

        posterior_alpha : Tensor
            [batch_size]
            Delta in typical acceptance in https://arxiv.org/pdf/2401.10774.

        posterior_threshold : Tensor
            [batch_size]
            Minimum probability threshold.
            Epsilon in typical acceptance in https://arxiv.org/pdf/2401.10774.

        tree_params : TreeParams
            Tree params of the input draft tokens.

        greedy_sampling : Tensor
            Whether to do greedy or non-greedy sampling.

        use_dynamic_tree: Tensor
            Whether to use dynamic tree (i.e., Eagle-2)


    Return:
        accepted_tokens : Tensor
            [batch_size, max_path_len]
            Accepted token ids. Only the first num_accepted_tokens[bi] tokens are relevant for bi'th row,

        num_accepted_tokens : Tensor
            [batch_size]
            Number of accepted tokens per request. Each entry is >= 1.

        accepted_paths : Tensor
            [batch_size]
            Indices of the accepted path per request of the input paths in tree_params.paths.

        next_draft_tokens : Tensor
            [batch_size, max_decoding_draft_tokens]
            Empty tensor used to allocate space for the next draft tokens.

        next_draft_lens : Tensor
            [batch_size]
            Empty tensor used to allocate space for lens of the next draft tokens.

        next_draft_paths : Tensor
            [batch_size, max_decoding_len, max_path_len]
            For EAGLE-1 just a copy of input path.

        hidden_size_batch_level_starts : Tensor
            [max_draft_path_len * batch_size + 1]
            Empty tensor used to allocate space for eagle_prepare_drafter_inputs_plugin.
    '''

    plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'EagleSampleAndAcceptDraftTokens', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plg_creator is not None

    pf_type = trt.PluginField("type_id",
                              np.array([int(lm_logits.dtype)], np.int32),
                              trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([pf_type])
    plugin = plg_creator.create_plugin("eagle_sample_and_accept_draft_plugin",
                                       pfc)

    plug_inputs = [
        lm_logits, draft_tokens, draft_lens, eagle_temperature,
        rand_data_validation, posterior_alpha, posterior_threshold,
        tree_params.paths, greedy_sampling, use_dynamic_tree
    ]

    plug_inputs = [i.trt_tensor for i in plug_inputs]
    layer = default_trtnet().add_plugin_v2(plug_inputs, plugin)

    accepted_tokens = _create_tensor(layer.get_output(0), layer)
    num_accepted_tokens = _create_tensor(layer.get_output(1), layer)
    accepted_paths = _create_tensor(layer.get_output(2), layer)
    next_draft_tokens = _create_tensor(layer.get_output(3), layer)
    next_draft_lens = _create_tensor(layer.get_output(4), layer)
    next_draft_paths = _create_tensor(layer.get_output(5), layer)
    hidden_size_batch_level_starts = _create_tensor(layer.get_output(6), layer)
    return tuple([
        accepted_tokens, num_accepted_tokens, accepted_paths, next_draft_tokens,
        next_draft_lens, next_draft_paths, hidden_size_batch_level_starts
    ])


def eagle_draft_decoder_plugin(
        layer_idx: int, num_eagle_layers: int, top_k_sampling: bool,
        logits: Tensor, num_last_token_indices: Tensor, input_paths: Tensor,
        use_dynamic_tree: Tensor, dynamic_tree_max_topK: Tensor,
        input_draft_token_ids: Tensor, input_draft_lens: Tensor,
        input_prev_scores: Tensor, input_current_expand_indices: Tensor,
        input_all_layers_scores: Tensor,
        input_all_layers_draft_token_ids: Tensor,
        input_all_layers_draft_token_ids_predecessor: Tensor):
    '''
    Parameters:
        layer_idx : int
            The index of the EagleNet.

        num_eagle_layers: int
            The total number of eagle layers.

        top_k_sampling: bool
            Whether to use top K sampling. Otherwise, use multinomial sampling.

        logits : Tensor
            [num_logits, vocab_size]
            Input logits.

        num_last_token_indices : Tensor
            [1]
            Number of valid logits in logits.

        input_paths: Tensor
            [batch_size, max_decoding_tokens, max_path_len]
            Input paths

        use_dynamic_tree: Tensor
            [1]
            Whether use dynamic tree (i.e., Eagle-2)

        dynamic_tree_max_topK: Tensor
            [1]
            Number of draft tokens expand in Eagle-2.

        input_draft_token_ids: Tensor
            [batch_size, max_decoding_draft_tokens]
            Draft tokens generated by previous EagleNets.

        input_draft_lens: Tensor
            [batch_size]
            Number of draft tokens for each request.

        input_prev_scores: Tensor
            [batch_size, max_decoding_draft_tokens]
            Last layer's scores

        input_current_expand_indices: Tensor
            [batch_size, max_decoding_draft_tokens]
            The indices of the nodes that expand in this layer.
            The index is related to the final output tree, which has max_decoding_draft_tokens draft tokens.

        input_all_layers_scores: Tensor
            [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]
            For Eagle-2, record scores from all EagleNets

        input_all_layers_draft_token_ids: Tensor
            [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]
            For Eagle-2, record all draft tokens from all EagleNets

        input_all_layers_draft_token_ids_predecessor: Tensor
            [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]
            For Eagle-2, record all draft tokens' predecessor

    Return:
        output_draft_token_ids: Tensor
            [batch_size, max_decoding_draft_tokens]
            Draft tokens generated by this EagleNets, also include the previous draft tokens.

        output_draft_draft_lens: Tensor
            [batch_size]
            Number of draft tokens for each request.

        output_paths: Tensor
            [batch_size, max_decoding_draft_tokens, max_path_len]
            The latest path.

        output_current_scores: Tensor
            [batch_size, max_decoding_draft_tokens]
            This layer's scores, which will be used in next layer.

        output_next_expand_indices:
            [batch_size, max_decoding_draft_tokens]
            The indices of the nodes that expand in next layer.
            The index is related to the final output tree, which has max_decoding_draft_tokens draft tokens.

        output_all_layers_scores:
            [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]
            For Eagle-2, record scores from all EagleNets

        output_all_layers_draft_token_ids: Tensor
            [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]
            For Eagle-2, record all draft tokens from all EagleNets

        output_all_layers_draft_token_ids_predecessor: Tensor
            [batch_size, num_eagle_layers, max_decoding_draft_tokens x max_decoding_draft_tokens]
            For Eagle-2, record all draft tokens' predecessor

    '''

    plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'EagleDecodeDraftTokens', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plg_creator is not None

    pf_type = trt.PluginField("type_id", np.array([int(logits.dtype)],
                                                  np.int32),
                              trt.PluginFieldType.INT32)

    layer_idx_t = trt.PluginField("layer_idx",
                                  np.array(layer_idx, dtype=np.int32),
                                  trt.PluginFieldType.INT32)

    num_eagle_layers_t = trt.PluginField(
        "num_eagle_layers", np.array(num_eagle_layers, dtype=np.int32),
        trt.PluginFieldType.INT32)

    top_k_sampling_t = 1 if top_k_sampling else 0
    top_k_sampling_t = trt.PluginField(
        "top_k_sampling", np.array(top_k_sampling_t, dtype=np.int32),
        trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection(
        [pf_type, layer_idx_t, num_eagle_layers_t, top_k_sampling_t])
    plugin = plg_creator.create_plugin("eagle_draft_decoder_plugin", pfc)

    plug_inputs = [
        logits, input_paths, num_last_token_indices, use_dynamic_tree,
        dynamic_tree_max_topK, input_draft_token_ids, input_draft_lens,
        input_prev_scores, input_current_expand_indices,
        input_all_layers_scores, input_all_layers_draft_token_ids,
        input_all_layers_draft_token_ids_predecessor
    ]

    plug_inputs = [i.trt_tensor for i in plug_inputs]
    layer = default_trtnet().add_plugin_v2(plug_inputs, plugin)

    output_draft_token_ids = _create_tensor(layer.get_output(0), layer)
    output_draft_lens = _create_tensor(layer.get_output(1), layer)
    output_paths = _create_tensor(layer.get_output(2), layer)
    output_current_scores = _create_tensor(layer.get_output(3), layer)
    output_next_expand_indices = _create_tensor(layer.get_output(4), layer)
    output_all_layers_scores = _create_tensor(layer.get_output(5), layer)
    output_all_layers_draft_token_ids = _create_tensor(layer.get_output(6),
                                                       layer)
    output_all_layers_draft_token_ids_predecessor = _create_tensor(
        layer.get_output(7), layer)
    return tuple([
        output_draft_token_ids, output_draft_lens, output_paths,
        output_current_scores, output_next_expand_indices,
        output_all_layers_scores, output_all_layers_draft_token_ids,
        output_all_layers_draft_token_ids_predecessor
    ])


def eagle_prepare_drafter_inputs_plugin(
        layer_idx: int, num_layers: int, max_non_leaves_per_layer: int,
        attention_params: AttentionParams, input_ids: Tensor,
        chunked_context_next_tokens: Tensor, accepted_token_ids: Tensor,
        accepted_lens: Tensor, accepted_path_ids: Tensor,
        next_draft_tokens: Tensor, next_draft_lens: Tensor,
        next_draft_paths: Tensor, prev_draft_lens: Tensor,
        prev_draft_paths: Tensor, hidden_size_batch_level_starts: Tensor,
        input_gen_tokens: Tensor,
        input_spec_decoding_generation_lengths: Tensor):
    '''
    Prepares inputs for the EagleNet inference.

    Visit tests/model/eagle/test_prepare_drafter_inputs.py for input/output examples.

    Parameters:
        layer_idx : int
            Index of the EagleNet. 0 means context phase EagleNet or EagleNet0,
            > 0 means EagleNetX or generation phase of EagleNet

        num_layers : int
            Number of Eagle layers.

        max_non_leaves_per_layer : int
            Number of nodes that can be non leaf in the tree at each level of the tree.

        attention_params : AttentionParams

        input_ids : Tensor
            [num_tokens]
            Tokens ids, inputs to the base model.

        chunked_context_next_tokens : Tensor
            [batch_size]
            The first token of the next chunk in chunked context.
            -1 if current chunk is the last chunk or requests is in the gen phase.

        accepted_token_ids : Tensor
            [batch_size, max_path_len]
            Accepted tokens ids.

        accepted_lens : Tensor
            [batch_size]
            Number of accepted tokens.

        accepted_path_ids : Tensor
            [batch_size]
            Idx of the accepted path in prev_draft_paths.

        next_draft_tokens : Tensor
            [batch_size, max_decoding_draft_tokens]
            Tokens ids of the draft tokens being drafted by EagleNet

        next_draft_lens : Tensor
            [batch_size]
            Number of drafted tokens in next_draft_tokens

        next_draft_paths : Tensor
            [batch_size, max_decoding_tokens, max_path_len]
            Paths of the draft tokens for the next iteration. In EAGLE-1 is the same as prev_draft_paths

        prev_draft_lens : Tensor
            [batch_size]
            Number of draft tokens, inputs to the base model.
            0 for ctx requests and actual draft len for gen requests.

        prev_draft_paths : Tensor
            [batch_size, max_decoding_tokens, max_path_len]
            Paths of the draft tokens inputs to the base model.

        hidden_size_batch_level_starts : Tensor
            [max_draft_path_len * batch_size + 1]
            Exclusive sum of the starts of the segments of the hidden states in the concatenated array.
            Hidden states shape is (flattened and w/o padding)
            [max_draft_path_len, batch_size, num_output_tokens_i_j], where num_output_tokens_i_j
            depends on the path of request j at level i.

        input_gen_tokens : Tensor
            [num_gen_tokens]
            Only needed to infer number of generation tokens from its shape. The content is irrelevant

        input_spec_decoding_generation_lengths : Tensor
            [num_gen_requests]
            Number of tokens for the base model. Only used to infer num_gen_requests from its shape, the content is irrelevant.

    Return:
        sequence_length : Tensor
            [batch_size]
            Sequence length of the next EagleNet iteration.
            For EagleNet0 equals to the (prompt_len + num_generated_tokens + accepted_lens).
            For EagleNetX (X > 0) (seq_len_eagle_net_0 + spec_decoding_generation_lengths).

        context_length : Tensor
            [batch_size]
            Context length of the next EagleNet iteration.
            For EagleNet0 it is either the actual context length of the request (for ctx requests)
            or the number of accepted tokens in this iteration. EagleNet0's attn does chunked context attn.
            For EagleNetX (X > 0), context length equals to the sequence length of the EagleNet0.

        spec_decoding_generation_lengths : Tensor
            [batch_size]
            Only relevant for EagleNetX (X > 0).
            Number of draft tokens.

        spec_decoding_position_offsets : Tensor
            [batch_size, max_decoding_tokens]
            Only relevant for EagleNetX (X > 0).
            Position offsets of the selected tokens from output_ids.

        spec_decoding_packed_mask : Tensor
            [batch_size, max_decoding_tokens, ceil(max_decoding_tokens / 32)]
            Only relevant for EagleNetX (X > 0).
            uint32_t packed masks.

        output_ids : Tensor
            [batch_size * max_non_leaves_per_layer * layer_idx] for layer_idx > 0
            [num_tokens - num_gen_tokens + num_gen_requests * (num_layers + 1)] for layer_idx == 0
            Token ids selected for the EagleNet iteration.
            Tensor's actual size is larger than num_output_tokens.

        position_ids : Tensor
            [batch_size] for layer_idx > 0
            [num_tokens - num_gen_tokens + num_gen_requests * (num_layers + 1)] for layer_idx == 0
            Position ids of the tokens selected for the EagleNet iteration.
            Tensor's actual size is larger than num_output_tokens.

        hidden_states_indices : Tensor
            [batch_size * max_non_leaves_per_layer * layer_idx] for layer_idx > 0
            [num_tokens - num_gen_tokens + num_gen_requests * (num_layers + 1)] for layer_idx == 0
            Indices of the hidden states to be selected from aggregated hidden states for the next iteration.
            Tensor's actual size is larger than num_output_tokens.

        last_token_indices : Tensor
            [batch_size * max_non_leaves_per_layer]
            Indices of the hidden states to be converted to logits after the next EagleNet iteration.
            Tensor's actual size is larger than num_output_tokens.

        num_last_token_indices : Tensor
            []
            Number of logits selected after the next EagleNet iteration.
            Tensors containing size of the outputs of V3 plugins. 0-D tensor.

        out_hidden_size_batch_level_starts : Tensor
            [max_draft_path_len * batch_size + 1]
            Same as hidden_size_batch_level_starts, but with updated path lens for the next level.
    '''

    plg_creator = trt.get_plugin_registry().get_creator(
        'EaglePrepareDrafterInputs', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plg_creator is not None

    layer_idx = trt.PluginField("layer_idx", np.array(layer_idx,
                                                      dtype=np.int32),
                                trt.PluginFieldType.INT32)

    num_layers = trt.PluginField("num_layers",
                                 np.array(num_layers, dtype=np.int32),
                                 trt.PluginFieldType.INT32)

    max_non_leaves_per_layer = trt.PluginField(
        "max_non_leaves_per_layer",
        np.array(max_non_leaves_per_layer, dtype=np.int32),
        trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection(
        [layer_idx, num_layers, max_non_leaves_per_layer])
    plugin = plg_creator.create_plugin("eagle_prepare_drafter_inputs_plugin",
                                       pfc, trt.TensorRTPhase.BUILD)

    plug_inputs = [
        attention_params.sequence_length, attention_params.context_lengths,
        input_ids, chunked_context_next_tokens, accepted_token_ids,
        accepted_lens, accepted_path_ids, next_draft_tokens, next_draft_lens,
        next_draft_paths, prev_draft_lens, prev_draft_paths,
        hidden_size_batch_level_starts, input_gen_tokens,
        input_spec_decoding_generation_lengths
    ]

    plug_inputs = [i.trt_tensor for i in plug_inputs]
    shape_inputs = []
    layer = default_trtnet().add_plugin_v3(plug_inputs, shape_inputs, plugin)

    sequence_length = _create_tensor(layer.get_output(0), layer)
    context_length = _create_tensor(layer.get_output(1), layer)
    spec_decoding_generation_lengths = _create_tensor(layer.get_output(2),
                                                      layer)
    spec_decoding_position_offsets = _create_tensor(layer.get_output(3), layer)
    spec_decoding_packed_mask = _create_tensor(layer.get_output(4), layer)
    output_ids = _create_tensor(layer.get_output(5), layer)
    position_ids = _create_tensor(layer.get_output(6), layer)
    hidden_states_indices = _create_tensor(layer.get_output(7), layer)
    last_token_indices = _create_tensor(layer.get_output(8), layer)
    num_last_token_indices = _create_tensor(layer.get_output(9), layer)
    out_hidden_size_batch_level_starts = _create_tensor(layer.get_output(10),
                                                        layer)
    return tuple([
        sequence_length, context_length, spec_decoding_generation_lengths,
        spec_decoding_position_offsets, spec_decoding_packed_mask, output_ids,
        position_ids, hidden_states_indices, last_token_indices,
        num_last_token_indices, out_hidden_size_batch_level_starts
    ])


class EagleNet(Module):

    def __init__(self, config, logits_dtype):
        super().__init__()
        self.drafter = LLaMAModel(config)
        self.config = config
        self.logits_dtype = logits_dtype

        vocab_size_padded = pad_vocab_size(config.vocab_size,
                                           config.mapping.tp_size)
        if config.mapping.is_last_pp_rank():
            self.lm_head = ColumnLinear(config.hidden_size,
                                        vocab_size_padded,
                                        bias=False,
                                        dtype=config.dtype,
                                        tp_group=config.mapping.tp_group,
                                        tp_size=config.mapping.tp_size,
                                        gather_output=True)
        else:
            self.lm_head = None

    def forward(self,
                input_ids,
                position_ids=None,
                hidden_states=None,
                last_token_indices=None,
                spec_decoding_params=None,
                kv_cache_params=None,
                attention_params=None):
        hidden_states, cache = self.drafter(
            input_ids,
            position_ids=position_ids,
            use_cache=True,
            spec_decoding_params=spec_decoding_params,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            hidden_states_for_embed=hidden_states)

        if self.config.mapping.is_last_pp_rank():
            hidden_states = gather_last_token_logits(
                hidden_states, last_token_indices,
                default_net().plugin_config.remove_input_padding)
            return cast(self.lm_head(hidden_states),
                        dtype=self.logits_dtype), hidden_states, cache

        return None, hidden_states, cache


class EagleForCausalLM(LLaMAForCausalLM):
    config_class = EagleConfig

    def __init__(self, config: EagleConfig):

        super().__init__(config)

        self.num_eagle_layers = config.num_eagle_layers
        self.max_non_leaves_per_layer = config.max_non_leaves_per_layer
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        vocab_size_padded = pad_vocab_size(self.vocab_size,
                                           config.mapping.tp_size)
        eagle_net_config = config.eagle_net_config
        eagle_net_config.mapping = Mapping(world_size=config.mapping.world_size,
                                           rank=config.mapping.rank,
                                           cp_size=1,
                                           tp_size=config.mapping.world_size,
                                           pp_size=1)

        eagle_net_config.fc_after_embed = True
        eagle_net_config.use_input_layernorm_in_first_layer = False
        eagle_net_config.use_last_layernorm = False
        eagle_net_config.layer_idx_offset = config.num_hidden_layers
        if self.mapping.is_last_pp_rank():
            self.eagle_nets = ModuleList([
                EagleNet(config=eagle_net_config,
                         logits_dtype=config.logits_dtype)
                for _ in range(self.num_eagle_layers)
            ])
        self.max_draft_len = config.max_draft_len

    def _prepare_drafter_inputs(
            self, layer_idx, input_ids, chunked_context_next_tokens,
            accepted_token_ids, accepted_lens, accepted_path_ids,
            next_draft_tokens, next_draft_lens, next_draft_paths,
            prev_draft_lens, prev_draft_paths, input_attention_params,
            input_kv_cache_params, hidden_states,
            host_ctx_eagle_net_request_types,
            host_ctx_eagle_net_context_lengths,
            host_ctx_eagle_net_past_key_value_lengths,
            host_gen_eagle_net_request_types,
            host_gen_eagle_net_context_lengths,
            host_gen_eagle_net_past_key_value_lengths,
            hidden_size_batch_level_starts, input_gen_tokens,
            input_spec_decoding_generation_lengths, spec_decoding_use):

        drafter_inputs = eagle_prepare_drafter_inputs_plugin(
            layer_idx, self.num_eagle_layers, self.max_non_leaves_per_layer,
            input_attention_params, input_ids, chunked_context_next_tokens,
            accepted_token_ids, accepted_lens, accepted_path_ids,
            next_draft_tokens, next_draft_lens, next_draft_paths,
            prev_draft_lens, prev_draft_paths, hidden_size_batch_level_starts,
            input_gen_tokens, input_spec_decoding_generation_lengths)

        sequence_length, context_lengths, \
            spec_decoding_generation_lengths, spec_decoding_position_offsets, \
            spec_decoding_packed_mask, output_ids, position_ids, hidden_states_indices, \
            last_token_indices, num_last_token_indices, out_hidden_size_batch_level_starts \
            = drafter_inputs

        attention_params = input_attention_params
        kv_cache_params = input_kv_cache_params
        attention_params.sequence_length = sequence_length
        attention_params.context_lengths = context_lengths

        if layer_idx == 0:
            attention_params.host_request_types = host_ctx_eagle_net_request_types
            attention_params.host_context_lengths = host_ctx_eagle_net_context_lengths
            kv_cache_params.host_past_key_value_lengths = host_ctx_eagle_net_past_key_value_lengths
        else:
            attention_params.host_request_types = host_gen_eagle_net_request_types
            attention_params.host_context_lengths = host_gen_eagle_net_context_lengths
            kv_cache_params.host_past_key_value_lengths = host_gen_eagle_net_past_key_value_lengths

        spec_decoding_params = None
        if layer_idx > 0:
            spec_decoding_params = SpecDecodingParams(
                True, self.max_draft_len, spec_decoding_generation_lengths,
                spec_decoding_position_offsets, spec_decoding_packed_mask,
                spec_decoding_use)

        # Get hidden states for accepted ids
        hidden_states = self._slice_hidden_states(hidden_states,
                                                  hidden_states_indices)

        eagle_net_inputs = {}
        eagle_net_inputs["input_ids"] = output_ids
        eagle_net_inputs["position_ids"] = position_ids
        eagle_net_inputs["last_token_indices"] = last_token_indices
        eagle_net_inputs["attention_params"] = attention_params
        eagle_net_inputs["kv_cache_params"] = kv_cache_params
        eagle_net_inputs["spec_decoding_params"] = spec_decoding_params
        eagle_net_inputs["hidden_states"] = hidden_states
        return eagle_net_inputs, out_hidden_size_batch_level_starts, num_last_token_indices

    def _slice_hidden_states(self, hidden_states, indices):
        hidden_states = index_select(hidden_states, 0, indices)

        hidden_states = hidden_states.view(concat(
            [shape(indices, 0), shape(hidden_states, 1)]),
                                           zero_is_placeholder=False)
        return hidden_states

    def _eagle_fwd_helper(self, lm_logits, hidden_states, *args, **kwargs):
        '''
        EAGLE inference can be viewed as
        TRT_Engine(Target -> Draft0 -> Draft1 -> .. -> DraftK-1) -> Runtime -> TRT_Engine(..) -> ..
        Target is Base model and Draft is EagleNet.

        Each EagleNet call can be viewed as call to Draft LLM in TensorRT-LLM.
        We have to
            1. prepare input tensors before the EagleNet call (like in the the runtime),
            2. call EagleNet,
            3. decode draft tokens after the EagleNet.
        The only difference with normal execution of the Draft model is that in EAGLE,
        all these 3 things happen inside of the TensorRT engine execution.
        We do 1 and 3 inside of the plugins.
        For 1. We call eagle_prepare_drafter_inputs_plugin and for 3. eagle_draft_decoder_plugin.

        The first call to the EagleNet (Draft0 == EagleNet0) is the context phase.
        For context request we populate the KV cache of the EagleNet.
        For generation request that have accepted tokens we emulate KV cache reuse by doing chunked attention,
        where chunk is the newly accepted tokens -- all previous tokens are already in the KV cache.

        The following calls to the EagleNet (EagleNetX (X > 0)) are generation phase.
        For each EagleNetX we select tokens based on the current path which are going to be used for the generation.

        Let's consider an example: prompt ABCD. EAGLE-1, i.e tree is fixed for the iteration.
        Tree:
                ┌───┐
                │ 0 │
                └─┬─┘
            ┌─────┴─────┐
          ┌─┴─┐ ┌─┴─┐ ┌─┴─┐
          │ 1 │ │ 2 │ │ 3 │
          └─┬─┘ └─┬─┘ └───┘
          ┌─┴─┐ ┌─┴─┐
          │ 4 │ │ 5 │
          └─┬─┘ └─┬─┘
          ┌─┴─┐ ┌─┴─┐
          │ 6 │ │ 7 │
          └───┘ └───┘

        First iteration of the TRT engine. Request is context request:
        1. Base model is called for [ABCD] tokens produces token E.
        2. Draft0 is called for tokens [BCDE] and produces
           three possibilities F, G and H for positions 1, 2 and 3 respectively.
        3. Since H (position 3) is a leaf, it is not chosen as the input to Draft1 inference.
        4. Draft1 is called for tokens [FG] with appropriate mask of:
             |F|G
            F|1|0
            G|0|1
            It produces tokens I and J for positions 4 and 5.
        6. Draft2 is called for inputs [FGIJ] with mask of
             |F|G|I|J
            F|1|0|0|0
            G|0|1|0|0
            I|1|0|1|0
            J|0|1|0|1
            Note that we could've stored FG in KV cache and provide only IJ tokens here
            with mask for past KV cache, but it is not supported in TensorRT LLM attention at the moment.

            Draft2 produces tokens K and L at positions 6 and 7.
        7. Resulting outputs are:
            7.1 accepted_ids [E]
            7.2 next_draft_tokens [FGHIJKL]

        Second iteration of the TRT engine. Request is the generation request.
        1. Base model is called for [EFGHIJKL]. Let's assume that it accepts [FIKM], i.e. the left-most path in the tree.
        2. Draft0 is called as context phase for [FIKM] -- to append to kv cache of the existing [BCDE].
           It produces tokens N, O and P for positions 1, 2 and 3.
        3. Draft1 is called as generation phase for [NO] tokens.
        etc.

        '''
        input_tree_params = kwargs["tree_params"]

        draft_tokens = kwargs['draft_tokens']
        draft_lens = kwargs['draft_lens']
        eagle_temperature = kwargs['eagle_temperature']
        rand_data_validation = kwargs['rand_data_validation']
        posterior_alpha = kwargs['posterior_alpha']
        posterior_threshold = kwargs['posterior_threshold']
        input_ids = kwargs['input_ids']
        chunked_context_next_tokens = kwargs['chunked_context_next_tokens']
        host_ctx_eagle_net_request_types = kwargs[
            'host_ctx_eagle_net_request_types']
        host_ctx_eagle_net_context_lengths = kwargs[
            'host_ctx_eagle_net_context_lengths']
        host_ctx_eagle_net_past_key_value_lengths = kwargs[
            'host_ctx_eagle_net_past_key_value_lengths']
        host_gen_eagle_net_request_types = kwargs[
            'host_gen_eagle_net_request_types']
        host_gen_eagle_net_context_lengths = kwargs[
            'host_gen_eagle_net_context_lengths']
        host_gen_eagle_net_past_key_value_lengths = kwargs[
            'host_gen_eagle_net_past_key_value_lengths']
        input_gen_tokens = kwargs["input_gen_tokens"]
        greedy_sampling = kwargs["greedy_sampling"]

        # Eagle-2
        use_dynamic_tree = kwargs['use_dynamic_tree']
        dynamic_tree_max_topK = kwargs['dynamic_tree_max_topK']
        prev_scores = kwargs['prev_scores']
        current_expand_indices = kwargs['current_expand_indices']
        all_layers_scores = kwargs['all_layers_scores']
        all_layers_draft_token_ids = kwargs['all_layers_draft_token_ids']
        all_layers_draft_token_ids_predecessor = kwargs[
            'all_layers_draft_token_ids_predecessor']

        # Sample target tokens and accept them
        # next_draft_tokens, next_draft_lens, hidden_size_batch_level_starts are outputted here just to
        # reserve the tensor with max size, which eagle_draft_decoder_plugin and
        # eagle_prepare_drafter_inputs_plugin are going to directly write to
        output = eagle_sample_and_accept_draft_plugin(
            lm_logits, draft_tokens, draft_lens, eagle_temperature,
            rand_data_validation, posterior_alpha, posterior_threshold,
            input_tree_params, greedy_sampling, use_dynamic_tree)
        accepted_tokens, num_accepted_tokens, accepted_paths, next_draft_tokens, \
            next_draft_lens, next_draft_paths, hidden_size_batch_level_starts = output

        attention_params = kwargs["attention_params"]
        kv_cache_params = kwargs["kv_cache_params"]
        spec_decoding_params = kwargs["spec_decoding_params"]

        input_hidden_states = hidden_states

        # Run EAGLE nets
        for li in range(self.num_eagle_layers):
            # Prepare EAGLE Net inputs.
            eagle_net_inputs, hidden_size_batch_level_starts, num_last_token_indices = self._prepare_drafter_inputs(
                layer_idx=li,
                input_ids=input_ids,
                chunked_context_next_tokens=chunked_context_next_tokens,
                accepted_token_ids=accepted_tokens,
                accepted_lens=num_accepted_tokens,
                accepted_path_ids=accepted_paths,
                next_draft_tokens=next_draft_tokens,
                next_draft_lens=next_draft_lens,
                next_draft_paths=next_draft_paths,
                prev_draft_lens=draft_lens,
                prev_draft_paths=input_tree_params.paths,
                input_attention_params=attention_params,
                input_kv_cache_params=kv_cache_params,
                hidden_states=input_hidden_states,
                host_ctx_eagle_net_request_types=
                host_ctx_eagle_net_request_types,
                host_ctx_eagle_net_context_lengths=
                host_ctx_eagle_net_context_lengths,
                host_ctx_eagle_net_past_key_value_lengths=
                host_ctx_eagle_net_past_key_value_lengths,
                host_gen_eagle_net_request_types=
                host_gen_eagle_net_request_types,
                host_gen_eagle_net_context_lengths=
                host_gen_eagle_net_context_lengths,
                host_gen_eagle_net_past_key_value_lengths=
                host_gen_eagle_net_past_key_value_lengths,
                hidden_size_batch_level_starts=hidden_size_batch_level_starts,
                input_gen_tokens=input_gen_tokens,
                input_spec_decoding_generation_lengths=spec_decoding_params.
                spec_decoding_generation_lengths,
                spec_decoding_use=spec_decoding_params.spec_decoding_use)

            def single_eagle_net_iter(next_draft_tokens, next_draft_lens,
                                      next_draft_paths, prev_scores,
                                      current_expand_indices, all_layers_scores,
                                      all_layers_draft_token_ids,
                                      all_layers_draft_token_ids_predecessor):
                # Run EAGLE Net
                # NOTE: handle base net kv cache and eagle net kv cache are in the same tensor.
                # EagleNet's kv cache is located starting at numBaseNetHiddenLayers in the kv tensor.
                logits, hidden_states, _ = self.eagle_nets[li](
                    **eagle_net_inputs)

                # FIXME We need to take top_k_sampling as an input
                top_k_sampling = True

                # Decode draft tokens
                next_draft_tokens, next_draft_lens, next_draft_paths, prev_scores, current_expand_indices, all_layers_scores, all_layers_draft_token_ids, all_layers_draft_token_ids_predecessor = eagle_draft_decoder_plugin(
                    li, self.num_eagle_layers, top_k_sampling, logits,
                    num_last_token_indices, next_draft_paths, use_dynamic_tree,
                    dynamic_tree_max_topK, next_draft_tokens, next_draft_lens,
                    prev_scores, current_expand_indices, all_layers_scores,
                    all_layers_draft_token_ids,
                    all_layers_draft_token_ids_predecessor)

                return next_draft_tokens, next_draft_lens, hidden_states, next_draft_paths, prev_scores, current_expand_indices, all_layers_scores, all_layers_draft_token_ids, all_layers_draft_token_ids_predecessor


            next_draft_tokens, next_draft_lens, hidden_states, next_draft_paths, prev_scores, \
                current_expand_indices, all_layers_scores, all_layers_draft_token_ids, all_layers_draft_token_ids_predecessor \
                    = single_eagle_net_iter(next_draft_tokens, next_draft_lens, next_draft_paths, \
                                          prev_scores, current_expand_indices, all_layers_scores, \
                                          all_layers_draft_token_ids, all_layers_draft_token_ids_predecessor)

            # Update params
            if li == 0:
                eagle_net_0_sequence_length = eagle_net_inputs[
                    "attention_params"].sequence_length
                input_hidden_states = hidden_states
            else:
                input_hidden_states = concat(
                    [input_hidden_states, hidden_states])

            kv_cache_params = eagle_net_inputs["kv_cache_params"]
            attention_params = eagle_net_inputs["attention_params"]
            attention_params.context_lengths = eagle_net_0_sequence_length
            attention_params.sequence_length = eagle_net_0_sequence_length

        # Mark tensors as output
        accepted_tokens.mark_output('accepted_tokens')
        num_accepted_tokens.mark_output('num_accepted_tokens')
        accepted_paths.mark_output('accepted_paths')
        next_draft_tokens.mark_output('next_draft_tokens')
        next_draft_lens.mark_output('next_draft_lens')
        next_draft_paths.mark_output('next_draft_paths')

        return next_draft_tokens

    def forward(self, *args, **kwargs):
        extra_args = [
            "draft_tokens", "draft_lens", "eagle_temperature",
            "rand_data_validation", "tree_params",
            "host_ctx_eagle_net_request_types",
            "host_ctx_eagle_net_context_lengths",
            "host_ctx_eagle_net_past_key_value_lengths",
            "host_gen_eagle_net_request_types",
            "host_gen_eagle_net_context_lengths",
            "host_gen_eagle_net_past_key_value_lengths", "input_gen_tokens",
            "chunked_context_next_tokens", "posterior_alpha",
            "posterior_threshold", "greedy_sampling", "use_dynamic_tree",
            "dynamic_tree_max_topK", "prev_scores", "current_expand_indices",
            "all_layers_scores", "all_layers_draft_token_ids",
            "all_layers_draft_token_ids_predecessor"
        ]

        base_kwargs = {k: v for k, v in kwargs.items() if k not in extra_args}

        # Base model forward
        hidden_states = super().forward(*args, **base_kwargs)

        if self.mapping.is_last_pp_rank():
            extra_args = ["hidden_states"]
            kwargs = {k: v for k, v in kwargs.items() if k not in extra_args}

        assert kwargs['use_cache'] and default_net(
        ).plugin_config.paged_kv_cache

        if self.mapping.is_last_pp_rank():
            lm_logits, hidden_states, all_hidden_states = hidden_states
            lm_logits = cast(lm_logits, self.config.logits_dtype)
            # Call eagle logic to accept prev draft tokens and predict next draft tokens
            next_draft_tokens = self._eagle_fwd_helper(lm_logits,
                                                       all_hidden_states, *args,
                                                       **kwargs)
        else:
            hidden_states.mark_output('hidden_states_output', self.config.dtype)

        if self.mapping.is_last_pp_rank():
            return next_draft_tokens
        return hidden_states

    def prepare_inputs(self, *args, **kwargs):
        """
        Inputs needed:
            device_request_types: [bs]
            draft_tokens: [bs, max_draft_len]
            draft_lens: [bs]
            spec_decoding_generation_lengths: [bs]
            spec_decoding_position_offsets: [bs, max_gen_tokens]
            spec_decoding_packed_mask: [bs, max_draft_len, packed_length] **
            eagle_temperature: [bs]
            rand_data_validation: [bs, max_draft_len]

            ** The mask is tricky since the boolean mask will need to be
               packed in runtime. So, the last dim will be:
                    packed_length = ceil((max_draft_len+1)/32)
        """
        default_range = GenerationMixin.default_range
        remove_input_padding = default_net().plugin_config.remove_input_padding
        use_gpt_attention_plugin = default_net(
        ).plugin_config.gpt_attention_plugin
        use_gemm_plugin = default_net().plugin_config.gemm_plugin
        paged_kv_cache = default_net().plugin_config.paged_kv_cache
        multiple_profiles = default_net().plugin_config.multiple_profiles
        max_batch_size = kwargs['max_batch_size']
        assert max_batch_size is not None
        gt_range = default_range(max_batch_size * (self.max_draft_len + 1),
                                 min_range=0,
                                 opt_offset=1)

        kwargs['speculative_decoding_draft_tokens_external'] = False
        kwargs['max_draft_len'] = self.max_draft_len
        kwargs['spec_decoding_is_generation_length_variable'] = True
        kwargs[
            'num_hidden_layers'] = self.config.num_hidden_layers + self.config.eagle_net_config.num_hidden_layers

        # Call base class prepare inputs
        inputs = super().prepare_inputs(*args, **kwargs)

        assert inputs['spec_decoding_params'] is not None

        kv_cache_type = KVCacheType.PAGED if paged_kv_cache else KVCacheType.CONTINUOUS
        enable_ctx_gen_opt_profiles = GenerationMixin.has_ctx_gen_opt_profiles(
            use_gpt_attention_plugin=use_gpt_attention_plugin,
            use_gemm_plugin=use_gemm_plugin,
            remove_input_padding=remove_input_padding,
            kv_cache_type=kv_cache_type)

        num_profiles, ranges = GenerationMixin.get_profiles_ranges(
            max_batch_size=max_batch_size,
            max_beam_width=kwargs['max_beam_width'],
            max_input_len=kwargs['max_input_len'],
            max_num_tokens=kwargs['max_num_tokens'],
            max_draft_len=self.max_draft_len,
            opt_batch_size=None
            if 'opt_batch_size' not in kwargs else kwargs['opt_batch_size'],
            opt_num_tokens=None
            if 'opt_num_tokens' not in kwargs else kwargs['opt_num_tokens'],
            enable_ctx_gen_opt_profiles=enable_ctx_gen_opt_profiles,
            multiple_profiles=multiple_profiles,
            kv_cache_type=kv_cache_type)

        bb_range = ranges['bb_range']

        draft_len_range = [self.max_draft_len for _ in range(len(bb_range))]
        decoding_len_range = [(self.max_draft_len + 1)
                              for _ in range(len(bb_range))]
        path_len_range = [(self.num_eagle_layers + 1)
                          for _ in range(len(bb_range))]
        gen_tokens_range = [gt_range for _ in range(len(bb_range))]
        num_eagle_layers_range = [
            self.num_eagle_layers for _ in range(len(bb_range))
        ]
        draft_len_square_range = [(self.max_draft_len * self.max_draft_len)
                                  for _ in range(len(bb_range))]

        draft_tokens = Tensor(name='draft_tokens',
                              dtype=trt.int32,
                              shape=[-1, self.max_draft_len],
                              dim_range=OrderedDict([
                                  ('batch_size', bb_range),
                                  ('draft_len', draft_len_range),
                              ]))
        draft_lens = Tensor(name='draft_lens',
                            dtype=trt.int32,
                            shape=[-1],
                            dim_range=OrderedDict([
                                ('batch_size', bb_range),
                            ]))
        eagle_temperature = Tensor(name='eagle_temperature',
                                   dtype=trt.float32,
                                   shape=[-1],
                                   dim_range=OrderedDict([
                                       ("batch_size", bb_range),
                                   ]))
        rand_data_validation = Tensor(name='rand_data_validation',
                                      dtype=trt.float32,
                                      shape=[-1, self.max_draft_len + 1],
                                      dim_range=OrderedDict([
                                          ('batch_size', bb_range),
                                          ('decoding_len', decoding_len_range),
                                      ]))
        posterior_alpha = Tensor(name='posterior_alpha',
                                 dtype=trt.float32,
                                 shape=[-1],
                                 dim_range=OrderedDict([
                                     ("batch_size", bb_range),
                                 ]))
        posterior_threshold = Tensor(name='posterior_threshold',
                                     dtype=trt.float32,
                                     shape=[-1],
                                     dim_range=OrderedDict([
                                         ("batch_size", bb_range),
                                     ]))
        draft_paths = Tensor(
            name='draft_paths',
            dtype=trt.int32,
            shape=[-1, self.max_draft_len + 1, self.num_eagle_layers + 1],
            dim_range=OrderedDict([
                ('batch_size', bb_range),
                ('decoding_len', decoding_len_range),
                ('path_len', path_len_range),
            ]))

        host_ctx_eagle_net_request_types = Tensor(
            name='host_ctx_eagle_net_request_types',
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([
                ('batch_size', bb_range),
            ]))
        host_ctx_eagle_net_context_lengths = Tensor(
            name='host_ctx_eagle_net_context_lengths',
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([
                ('batch_size', bb_range),
            ]))
        host_ctx_eagle_net_past_key_value_lengths = Tensor(
            name='host_ctx_eagle_net_past_key_value_lengths',
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([
                ('batch_size', bb_range),
            ]))
        host_gen_eagle_net_request_types = Tensor(
            name='host_gen_eagle_net_request_types',
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([
                ('batch_size', bb_range),
            ]))
        host_gen_eagle_net_context_lengths = Tensor(
            name='host_gen_eagle_net_context_lengths',
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([
                ('batch_size', bb_range),
            ]))
        host_gen_eagle_net_past_key_value_lengths = Tensor(
            name='host_gen_eagle_net_past_key_value_lengths',
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([
                ('batch_size', bb_range),
            ]))

        input_gen_tokens = Tensor(name='input_gen_tokens',
                                  dtype=trt.int32,
                                  shape=[-1],
                                  dim_range=OrderedDict([
                                      ('gen_tokens', gen_tokens_range),
                                  ]))
        chunked_context_next_tokens = Tensor(name='chunked_context_next_tokens',
                                             dtype=trt.int32,
                                             shape=[-1],
                                             dim_range=OrderedDict([
                                                 ('batch_size', bb_range),
                                             ]))
        greedy_sampling = Tensor(name='greedy_sampling',
                                 dtype=trt.int32,
                                 shape=[1])

        use_dynamic_tree = Tensor(name='use_dynamic_tree',
                                  dtype=trt.int32,
                                  shape=[1])

        dynamic_tree_max_topK = Tensor(name='dynamic_tree_max_topK',
                                       dtype=trt.int32,
                                       shape=[1])

        prev_scores = Tensor(name='prev_scores',
                             dtype=trt.float32,
                             shape=[-1, self.max_draft_len],
                             dim_range=OrderedDict([
                                 ('batch_size', bb_range),
                                 ('draft_len', draft_len_range),
                             ]))

        current_expand_indices = Tensor(name='current_expand_indices',
                                        dtype=trt.int32,
                                        shape=[-1, self.max_draft_len],
                                        dim_range=OrderedDict([
                                            ('batch_size', bb_range),
                                            ('draft_len', draft_len_range),
                                        ]))

        all_layers_scores = Tensor(name='all_layers_scores',
                                   dtype=trt.float32,
                                   shape=[
                                       -1, self.num_eagle_layers,
                                       self.max_draft_len * self.max_draft_len
                                   ],
                                   dim_range=OrderedDict([
                                       ('batch_size', bb_range),
                                       ('num_eagle_layers',
                                        num_eagle_layers_range),
                                       ('draft_len_square',
                                        draft_len_square_range),
                                   ]))

        all_layers_draft_token_ids = Tensor(
            name='all_layers_draft_token_ids',
            dtype=trt.int32,
            shape=[
                -1, self.num_eagle_layers,
                self.max_draft_len * self.max_draft_len
            ],
            dim_range=OrderedDict([
                ('batch_size', bb_range),
                ('num_eagle_layers', num_eagle_layers_range),
                ('draft_len_square', draft_len_square_range),
            ]))

        all_layers_draft_token_ids_predecessor = Tensor(
            name='all_layers_draft_token_ids_predecessor',
            dtype=trt.int32,
            shape=[
                -1, self.num_eagle_layers,
                self.max_draft_len * self.max_draft_len
            ],
            dim_range=OrderedDict([
                ('batch_size', bb_range),
                ('num_eagle_layers', num_eagle_layers_range),
                ('draft_len_square', draft_len_square_range),
            ]))

        tree_params = TreeParams(paths=draft_paths)

        inputs['draft_tokens'] = draft_tokens
        inputs['draft_lens'] = draft_lens
        inputs['eagle_temperature'] = eagle_temperature
        inputs['posterior_alpha'] = posterior_alpha
        inputs['posterior_threshold'] = posterior_threshold
        inputs['rand_data_validation'] = rand_data_validation
        inputs['tree_params'] = tree_params
        inputs[
            'host_ctx_eagle_net_request_types'] = host_ctx_eagle_net_request_types
        inputs[
            'host_ctx_eagle_net_context_lengths'] = host_ctx_eagle_net_context_lengths
        inputs[
            'host_ctx_eagle_net_past_key_value_lengths'] = host_ctx_eagle_net_past_key_value_lengths
        inputs[
            'host_gen_eagle_net_request_types'] = host_gen_eagle_net_request_types
        inputs[
            'host_gen_eagle_net_context_lengths'] = host_gen_eagle_net_context_lengths
        inputs[
            'host_gen_eagle_net_past_key_value_lengths'] = host_gen_eagle_net_past_key_value_lengths
        inputs['input_gen_tokens'] = input_gen_tokens
        inputs['chunked_context_next_tokens'] = chunked_context_next_tokens
        inputs['greedy_sampling'] = greedy_sampling
        inputs['use_dynamic_tree'] = use_dynamic_tree
        inputs['dynamic_tree_max_topK'] = dynamic_tree_max_topK
        inputs['prev_scores'] = prev_scores
        inputs['current_expand_indices'] = current_expand_indices
        inputs['all_layers_scores'] = all_layers_scores
        inputs['all_layers_draft_token_ids'] = all_layers_draft_token_ids
        inputs[
            'all_layers_draft_token_ids_predecessor'] = all_layers_draft_token_ids_predecessor

        return inputs

    @classmethod
    def from_hugging_face(
            cls,
            hf_model_or_dir: Union[str, 'transformers.PreTrainedModel'],
            dtype: str = 'auto',
            mapping: Optional[Mapping] = None,
            quant_config: Optional[QuantConfig] = None,
            **kwargs):
        assert hf_model_or_dir is not None
        speculative_model_dir = kwargs.get('speculative_model_dir', None)
        tllm_config = EagleConfig.from_hugging_face(hf_model_or_dir,
                                                    dtype=dtype,
                                                    mapping=mapping,
                                                    quant_config=quant_config,
                                                    **kwargs)

        # for rank in range(mapping.world_size):
        tllm_config.mapping = Mapping(world_size=mapping.world_size,
                                      rank=mapping.rank,
                                      cp_size=1,
                                      tp_size=mapping.tp_size,
                                      pp_size=mapping.pp_size)

        model = EagleForCausalLM(tllm_config)

        def check_and_update(module, dict):
            if hasattr(module, 'tllm_to_externel_key_dict'):
                module.tllm_to_externel_key_dict.update(dict)
            else:
                module.tllm_to_externel_key_dict = dict

        def copy(tensors):
            if isinstance(tensors, list):
                if None in tensors:
                    return tensors
                else:
                    return [tensor.clone() for tensor in tensors]
            elif tensors is None:
                return tensors
            else:
                return tensors.clone()

        shared_weight_prefixs = []
        tllm_weights = {}
        customized_dict = {"drafter": ""}
        if speculative_model_dir is None:
            # Single checkpoint for ModelOpt
            for idx, eagle_net in enumerate(model.eagle_nets):
                check_and_update(eagle_net.drafter.fc, {"fc": "fc"})
                check_and_update(eagle_net.drafter.vocab_embedding,
                                 {f"eagle_nets.{idx}": "model"})
                check_and_update(eagle_net.lm_head, {f"eagle_nets.{idx}": ""})
                shared_weight_prefixs.append(f"eagle_nets.{idx}")
                customized_dict[f'eagle_nets.{idx}'] = 'eagle_module'
            loader = ModelWeightsLoader(speculative_model_dir, customized_dict)
            loader.update_key_mapping(model)
            for tllm_key, _ in tqdm(model.named_parameters()):
                if any([
                        tllm_key.startswith(prefix)
                        for prefix in shared_weight_prefixs
                ]):
                    tllm_weights.update(loader.load(tllm_key, preprocess=copy))
                else:
                    tllm_weights.update(loader.load(tllm_key))
            loader.fill(tllm_weights)
        else:
            # Double checkpoint for HF
            for idx, eagle_net in enumerate(model.eagle_nets):
                check_and_update(eagle_net.drafter.fc, {"fc": "fc"})
                check_and_update(eagle_net.drafter.vocab_embedding,
                                 {f"eagle_nets.{idx}": ""})
                check_and_update(eagle_net.lm_head, {f"eagle_nets.{idx}": ""})
                shared_weight_prefixs.append(f"eagle_nets.{idx}")
                customized_dict[f'eagle_nets.{idx}'] = ''

            # Load base model
            base_loader = ModelWeightsLoader(hf_model_or_dir)
            base_loader.update_key_mapping(model)
            for tllm_key, _ in tqdm(model.transformer.named_parameters()):
                tllm_weights.update(base_loader.load("transformer." + tllm_key))
            tllm_weights.update(base_loader.load("lm_head.weight"))
            # for idx in range(args.num_eagle_layers):
            for idx in range(4):
                tllm_weights.update(
                    base_loader.load(f"eagle_nets.{idx}.lm_head.weight",
                                     preprocess=copy))

            # Load eagle model
            eagle_loader = ModelWeightsLoader(str(speculative_model_dir),
                                              customized_dict)
            eagle_loader.update_key_mapping(model)
            for tllm_key, _ in tqdm(model.eagle_nets.named_parameters()):
                if not tllm_key.endswith("lm_head.weight"):
                    if any([
                            tllm_key.startswith(prefix)
                            for prefix in shared_weight_prefixs
                    ]):
                        tllm_weights.update(
                            eagle_loader.load("eagle_nets." + tllm_key,
                                              preprocess=copy))
                    else:
                        tllm_weights.update(
                            eagle_loader.load("eagle_nets." + tllm_key))
            base_loader.fill(tllm_weights)

        return model

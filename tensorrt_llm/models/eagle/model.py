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

import numpy as np
import tensorrt as trt

from tensorrt_llm.models.generation_mixin import GenerationMixin
from tensorrt_llm.models.llama.model import LLaMAForCausalLM, LLaMAModel

from ..._common import default_net, default_trtnet
from ..._utils import pad_vocab_size
from ...bindings import KVCacheType
from ...functional import Tensor, _create_tensor, concat, index_select, shape
from ...layers import (AttentionParams, ColumnLinear, KeyValueCacheParams,
                       SpecDecodingParams)
from ...module import Module, ModuleList
from ...plugin import TRT_LLM_PLUGIN_NAMESPACE
from .config import EagleConfig


class TreeParams(object):

    def __init__(self, paths: Tensor = None):
        self.paths = paths


def eagle_sample_and_accept_draft_plugin(lm_logits: Tensor = None,
                                         draft_tokens: Tensor = None,
                                         draft_lens: Tensor = None,
                                         eagle_temperature: Tensor = None,
                                         rand_data_validation: Tensor = None,
                                         tree_params: TreeParams = None,
                                         greedy_sampling: bool = True):
    # TODO
    '''
    Parameters:
        lm_logits : Tensor (On GPU)

        draft_tokens: Tensor

        draft_lens: Tensor

        eagle_temperature: Tensor

        rand_data_validation: Tensor

        tree_params : TreeParams

        greedy_sampling : bool

    Return:
        accepted_tokens, num_accepted_tokens, accepted_paths, last_accepted_tokens,
        cum_sum_last_accepted_indices, next_draft_tokens, next_draft_lens

    '''

    plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'EagleSampleAndAcceptDraftTokens', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plg_creator is not None

    pf_type = trt.PluginField("type_id",
                              np.array([int(lm_logits.dtype)], np.int32),
                              trt.PluginFieldType.INT32)

    greedy_sampling = 1 if greedy_sampling else 0
    greedy_sampling = trt.PluginField("greedy_sampling",
                                      np.array(greedy_sampling, dtype=np.int32),
                                      trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([pf_type, greedy_sampling])
    plugin = plg_creator.create_plugin("eagle_sample_and_accept_draft_plugin",
                                       pfc)

    plug_inputs = [
        lm_logits, draft_tokens, draft_lens, eagle_temperature,
        rand_data_validation, tree_params.paths
    ]

    plug_inputs = [i.trt_tensor for i in plug_inputs]
    layer = default_trtnet().add_plugin_v2(plug_inputs, plugin)

    accepted_tokens = _create_tensor(layer.get_output(0), layer)
    num_accepted_tokens = _create_tensor(layer.get_output(1), layer)
    accepted_paths = _create_tensor(layer.get_output(2), layer)
    last_accepted_tokens = _create_tensor(layer.get_output(3), layer)
    cum_sum_last_accepted_indices = _create_tensor(layer.get_output(4), layer)
    next_draft_tokens = _create_tensor(layer.get_output(5), layer)
    next_draft_lens = _create_tensor(layer.get_output(6), layer)
    return tuple([
        accepted_tokens, num_accepted_tokens, accepted_paths,
        last_accepted_tokens, cum_sum_last_accepted_indices, next_draft_tokens,
        next_draft_lens
    ])


def eagle_draft_decoder_plugin(layer_idx: int, logits: Tensor,
                               next_draft_tokens: Tensor,
                               next_draft_lens: Tensor,
                               rand_data_sample: Tensor,
                               tree_params: TreeParams):
    # TODO
    '''
    Parameters:
        layer_idx : int

        logits : Tensor

        next_draft_tokens : Tensor

        next_draft_lens : Tensor

        rand_data_sample : Tensor

        tree_params : TreeParams

    Return:
        output_next_draft_tokens, output_next_draft_lens

    '''

    plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'EagleDecodeDraftTokens', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plg_creator is not None

    pf_type = trt.PluginField("type_id", np.array([int(logits.dtype)],
                                                  np.int32),
                              trt.PluginFieldType.INT32)

    layer_idx = trt.PluginField("layer_idx", np.array(layer_idx,
                                                      dtype=np.int32),
                                trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([pf_type, layer_idx])
    plugin = plg_creator.create_plugin("eagle_draft_decoder_plugin", pfc)

    plug_inputs = [
        logits, next_draft_tokens, next_draft_lens, rand_data_sample,
        tree_params.paths
    ]

    plug_inputs = [i.trt_tensor for i in plug_inputs]
    layer = default_trtnet().add_plugin_v2(plug_inputs, plugin)

    output_next_draft_tokens = _create_tensor(layer.get_output(0), layer)
    output_next_draft_lens = _create_tensor(layer.get_output(1), layer)
    return tuple([output_next_draft_tokens, output_next_draft_lens])


def eagle_prepare_drafter_inputs_plugin(layer_idx: int,
                                        attention_params: AttentionParams,
                                        kv_cache_params: KeyValueCacheParams,
                                        tree_params: TreeParams,
                                        hidden_states: Tensor):
    # TODO
    '''
    Parameters:
        layer_idx : int

        attention_params : AttentionParams

        kv_cache_params : KeyValueCacheParams

        tree_params : TreeParams

        hidden_states: Tensor

    Return:
        sequence_length, host_request_types, host_past_key_value_lengths,
        past_key_value_length, spec_decoding_generation_lengths, spec_decoding_position_offsets,
        spec_decoding_packed_mask, input_ids, position_ids, hidden_states

    '''

    plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'EaglePrepareDrafterInputs', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plg_creator is not None

    pf_type = trt.PluginField("type_id",
                              np.array([int(hidden_states.dtype)], np.int32),
                              trt.PluginFieldType.INT32)

    layer_idx = trt.PluginField("layer_idx", np.array(layer_idx,
                                                      dtype=np.int32),
                                trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([pf_type, layer_idx])
    plugin = plg_creator.create_plugin("eagle_prepare_drafter_inputs_plugin",
                                       pfc)

    plug_inputs = [
        attention_params.sequence_length, attention_params.host_request_types,
        kv_cache_params.host_past_key_value_lengths,
        kv_cache_params.host_kv_cache_pool_pointers,
        kv_cache_params.kv_cache_block_offsets, tree_params.paths, hidden_states
    ]

    plug_inputs = [i.trt_tensor for i in plug_inputs]
    layer = default_trtnet().add_plugin_v2(plug_inputs, plugin)

    sequence_length = _create_tensor(layer.get_output(0), layer)
    host_request_types = _create_tensor(layer.get_output(1), layer)
    host_past_key_value_lengths = _create_tensor(layer.get_output(2), layer)
    spec_decoding_generation_lengths = _create_tensor(layer.get_output(3),
                                                      layer)
    spec_decoding_position_offsets = _create_tensor(layer.get_output(4), layer)
    spec_decoding_packed_mask = _create_tensor(layer.get_output(5), layer)
    input_ids = _create_tensor(layer.get_output(6), layer)
    position_ids = _create_tensor(layer.get_output(7), layer)
    hidden_states = _create_tensor(layer.get_output(8), layer)
    return tuple([
        sequence_length, host_request_types, host_past_key_value_lengths,
        spec_decoding_generation_lengths, spec_decoding_position_offsets,
        spec_decoding_packed_mask, input_ids, position_ids, hidden_states
    ])


class EagleNet(Module):

    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.drafter = LLaMAModel(config)

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

    def forward(self, hidden_states, input_ids, position_ids,
                spec_decoding_params, kv_cache_params, attention_params):
        hidden_states, cache = self.drafter(
            input_ids,
            position_ids=position_ids,
            use_cache=True,
            spec_decoding_params=spec_decoding_params,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            hidden_states_for_embed=hidden_states)

        return self.lm_head(hidden_states), hidden_states, cache


class EagleForCausalLM(LLaMAForCausalLM):
    config_class = EagleConfig

    def __init__(self, config: EagleConfig):

        super().__init__(config)
        self.num_eagle_layers = config.num_eagle_layers
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        vocab_size_padded = pad_vocab_size(self.vocab_size,
                                           config.mapping.tp_size)
        eagle_net_config = config.eagle_net_config
        eagle_net_config.fc_after_embed = True
        eagle_net_config.use_input_layernorm_in_first_layer = False
        self.eagle_nets = ModuleList([
            EagleNet(config=eagle_net_config)
            for _ in range(self.num_eagle_layers)
        ])
        self.max_draft_len = config.max_draft_len

    def _prepare_drafter_inputs(self, layer_idx, input_attention_params,
                                input_kv_cache_params, input_tree_params,
                                hidden_states):

        drafter_inputs = eagle_prepare_drafter_inputs_plugin(
            layer_idx, input_attention_params, input_kv_cache_params,
            input_tree_params, hidden_states)

        sequence_length, host_request_types, host_past_key_value_lengths, \
            spec_decoding_generation_lengths, spec_decoding_position_offsets, \
            spec_decoding_packed_mask, input_ids, position_ids, hidden_states = drafter_inputs

        attention_params = input_attention_params
        attention_params.sequence_length = sequence_length
        attention_params.host_request_types = host_request_types

        kv_cache_params = input_kv_cache_params
        kv_cache_params.host_past_key_value_lengths = host_past_key_value_lengths

        spec_decoding_params = SpecDecodingParams(
            True, self.max_draft_len, spec_decoding_generation_lengths,
            spec_decoding_position_offsets, spec_decoding_packed_mask)

        eagle_net_inputs = {}
        eagle_net_inputs["input_ids"] = input_ids
        eagle_net_inputs["position_ids"] = position_ids
        eagle_net_inputs["attention_params"] = attention_params
        eagle_net_inputs["kv_cache_params"] = kv_cache_params
        eagle_net_inputs["spec_decoding_params"] = spec_decoding_params
        eagle_net_inputs["hidden_states"] = hidden_states
        return eagle_net_inputs

    def _slice_hidden_states(self, hidden_states, last_ids):
        hidden_states = index_select(hidden_states, 0,
                                     last_ids - 1)  # [seq_len, hidden]

        hidden_states = hidden_states.view(
            concat([shape(last_ids, 0),
                    shape(hidden_states, 1)]))
        return hidden_states

    def _eagle_fwd_helper(self, lm_logits, hidden_states, *args, **kwargs):
        # TODO what to do for context

        # FIXME Once it is clear what we need from Tree, either get them from runtime or assemble dynamically in the TRT
        # Most likely the latter, as EAGLE-2 gets dynamic tree.
        input_tree_params = kwargs["tree_params"]

        draft_tokens = kwargs['draft_tokens']
        draft_lens = kwargs['draft_lens']
        eagle_temperature = kwargs['eagle_temperature']
        rand_data_validation = kwargs['rand_data_validation']
        rand_data_sample = kwargs['rand_data_sample']

        # Sample target tokens and accept them
        # next_draft_tokens, next_draft_lens, next_paths are outputted here just to
        # reserve the tensor with max size, which eagle_draft_decoder_plugin is going to directly write to
        output = eagle_sample_and_accept_draft_plugin(lm_logits, draft_tokens,
                                                      draft_lens,
                                                      eagle_temperature,
                                                      rand_data_validation,
                                                      input_tree_params)
        accepted_tokens, num_accepted_tokens, accepted_paths, \
            last_accepted_tokens, cum_sum_last_accepted_indices, next_draft_tokens, next_draft_lens = output

        # Get hidden states for accepted ids
        hidden_states = self._slice_hidden_states(
            hidden_states, cum_sum_last_accepted_indices)

        attention_params = kwargs["attention_params"]
        kv_cache_params = kwargs["kv_cache_params"]

        # Run EAGLE nets
        for li in range(self.num_eagle_layers):

            # FIXME: what to do with appending KV cache in the decoder
            # We won't append more than max_draft_len + max_path_len
            # TODO handle context requests in a special way
            # TODO rewind KV cache
            # For the 1st layer, rewind kv cache for the accepted tokens, prepare EAGLE Net inputs
            eagle_net_inputs = self._prepare_drafter_inputs(
                li, attention_params, kv_cache_params, input_tree_params,
                hidden_states)

            # Run EAGLE Net
            logits, hidden_states, _ = self.eagle_nets[li](**eagle_net_inputs)

            # Decode draft tokens
            next_draft_tokens, next_draft_lens = eagle_draft_decoder_plugin(
                li, logits, next_draft_tokens, next_draft_lens,
                rand_data_sample, input_tree_params)

            # Update params
            attention_params = eagle_net_inputs["attention_params"]
            kv_cache_params = eagle_net_inputs["kv_cache_params"]

        # Mark tensors as output
        accepted_tokens.mark_output('accepted_tokens')
        num_accepted_tokens.mark_output('num_accepted_tokens')
        accepted_paths.mark_output('accepted_paths')
        next_draft_tokens.mark_output('next_draft_tokens')
        next_draft_lens.mark_output('next_draft_lens')

        return next_draft_tokens

    def forward(self, *args, **kwargs):
        extra_args = [
            "draft_tokens", "draft_lens", "eagle_temperature",
            "rand_data_validation", "rand_data_sample", "tree_params"
        ]

        base_kwargs = {k: v for k, v in kwargs.items() if k not in extra_args}

        # Base model forward
        hidden_states = super().forward(*args, **base_kwargs)

        assert kwargs['use_cache'] and default_net(
        ).plugin_config.paged_kv_cache

        lm_logits, hidden_states = hidden_states

        if self.mapping.is_last_pp_rank():
            # Call eagle logic to accept prev draft tokens and predict next draft tokens
            next_draft_tokens = self._eagle_fwd_helper(lm_logits, hidden_states,
                                                       *args, **kwargs)
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
            rand_data_sample: [bs]
            rand_data_validation: [bs, max_draft_tokens]

            ** The mask is tricky since the boolean mask will need to be
               packed in runtime. So, the last dim will be:
                    packed_length = ceil(max_draft_tokens/32)
        """
        default_range = GenerationMixin.default_range
        remove_input_padding = default_net().plugin_config.remove_input_padding
        use_gpt_attention_plugin = default_net(
        ).plugin_config.gpt_attention_plugin
        use_gemm_plugin = default_net().plugin_config.gemm_plugin
        paged_kv_cache = default_net().plugin_config.paged_kv_cache
        max_batch_size = kwargs['max_batch_size']
        assert max_batch_size is not None
        bb_range = default_range(max_batch_size)
        bb0_range = default_range(max_batch_size, min_range=0, opt_offset=1)

        kwargs['speculative_decoding_draft_tokens_external'] = False
        kwargs['max_draft_len'] = self.max_draft_len
        kwargs['spec_decoding_is_generation_length_variable'] = True

        # Call base class prepare inputs
        inputs = super().prepare_inputs(*args, **kwargs)

        assert inputs['spec_decoding_params'] is not None

        enable_two_optimization_profiles = GenerationMixin.has_ctx_gen_opt_profiles(
            use_gpt_attention_plugin=use_gpt_attention_plugin,
            use_gemm_plugin=use_gemm_plugin,
            remove_input_padding=remove_input_padding,
            kv_cache_type=KVCacheType.PAGED
            if paged_kv_cache else KVCacheType.CONTINUOUS)
        if enable_two_optimization_profiles:
            bb_range = [bb_range, bb_range]
            bb0_range = [bb0_range, bb0_range]
            draft_len_range = [self.max_draft_len]
            path_len_range = [self.num_eagle_layers + 1]
        else:
            bb_range = [bb_range]
            bb0_range = [bb0_range]
            draft_len_range = [self.max_draft_len]
            path_len_range = [self.num_eagle_layers + 1]

        draft_tokens = Tensor(name='draft_tokens',
                              dtype=trt.int32,
                              shape=[-1, self.max_draft_len],
                              dim_range=OrderedDict([
                                  ('batch_size_wt0', bb0_range),
                                  ('draft_len', draft_len_range),
                              ]))
        draft_lens = Tensor(name='draft_lens',
                            dtype=trt.int32,
                            shape=[-1],
                            dim_range=OrderedDict([
                                ('batch_size_wt0', bb0_range),
                            ]))
        eagle_temperature = Tensor(name='eagle_temperature',
                                   dtype=trt.float32,
                                   shape=[-1],
                                   dim_range=OrderedDict([
                                       ("batch_size", bb_range),
                                   ]))
        rand_data_validation = Tensor(name='rand_data_validation',
                                      dtype=trt.float32,
                                      shape=[-1, self.max_draft_len],
                                      dim_range=OrderedDict([
                                          ('batch_size_wt0', bb0_range),
                                          ('draft_len', draft_len_range),
                                      ]))
        rand_data_sample = Tensor(name='rand_data_sample',
                                  dtype=trt.float32,
                                  shape=[-1],
                                  dim_range=OrderedDict([
                                      ('batch_size', bb_range),
                                  ]))
        tree_paths = Tensor(
            name='tree_paths',
            dtype=trt.int32,
            # FIXME max_accepted len is not necessary self.num_eagle_layers + 1. Only True for EAGLE-1
            shape=[-1, self.max_draft_len, self.num_eagle_layers + 1],
            dim_range=OrderedDict([
                ('batch_size', bb_range),
                ('draft_len', draft_len_range),
                ('path_len', path_len_range),
            ]))

        tree_params = TreeParams(paths=tree_paths)

        inputs['draft_tokens'] = draft_tokens
        inputs['draft_lens'] = draft_lens
        inputs['eagle_temperature'] = eagle_temperature
        inputs['rand_data_validation'] = rand_data_validation
        inputs['rand_data_sample'] = rand_data_sample
        inputs['tree_params'] = tree_params
        return inputs

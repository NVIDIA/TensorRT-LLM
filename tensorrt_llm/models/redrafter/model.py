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

import tensorrt as trt

from tensorrt_llm._common import default_net
from tensorrt_llm.bindings import KVCacheType
from tensorrt_llm.functional import Tensor, cast, categorical_sample
from tensorrt_llm.models import LLaMAForCausalLM, QWenForCausalLM
from tensorrt_llm.models.generation_mixin import GenerationMixin

from ..._utils import pad_vocab_size, str_dtype_to_trt
from .drafter import Drafter
from .redrafter_helper import (_beam_search_candidates, _beams2tree,
                               _process_logits_and_hidden_states)


class ReDrafterMixin:

    def __init__(self, config):

        super().__init__(config)
        self.dtype = str_dtype_to_trt(config.dtype)
        self.vocab_size = config.vocab_size
        vocab_size_padded = pad_vocab_size(self.vocab_size,
                                           config.mapping.tp_size)
        self.drafter = Drafter.from_config(config, vocab_size_padded)
        self.num_beams = config.redrafter_num_beams
        self.beam_candidate_length = config.redrafter_draft_len_per_beam
        self.beam_length = self.beam_candidate_length + 1  # including true token
        self.greedy_search = config.redrafter_greedy_search
        self.is_rnn = config.redrafter_is_rnn
        assert self.dtype == self.drafter.dtype, f"{self.dtype} != {self.drafter.dtype}"

    def _fwd_helper(self, hidden_states, lm_logits, embedding, drafter,
                    kwargs: dict):
        '''
        Must enable remove_input_padding:
            hidden_states [total_tokens, H]
            lm_logits [total_tokens, V]
        1. process_logits: context vs gen
            a. Context: just return the last hidden states, and logits/probs
            b. Gen:
                i. verify: use lm_logits, draft_probs, draft_indices, draft_tokens
                ii. select hidden state and update probs
        3. Sample token based on probs
        4. Generate candidates using hidden_states, sampled token
        5. Using beams, generate validation buffers, mark them as output
        6. Mark all the outputs
        '''

        num_beams = self.num_beams
        beam_length = self.beam_length

        # Get the inputs needed
        rand_data_sample = kwargs['rand_data_sample']
        position_ids_base = kwargs['position_ids_base']

        # Step 1: Process logits and hidden states
        # process the base model output (verify for gen-phase)
        probs, draft_input, num_accepted_tokens, \
            accepted_beam_index = _process_logits_and_hidden_states(
                self, lm_logits, hidden_states, kwargs)
        # NOTE: num_accepted_tokens doesn't include true token so add 1 here
        num_accepted_tokens = num_accepted_tokens + 1

        # At this point:
        #  probs : [bs, V]
        #  hidden_states : [bs, H]

        # Step 2: Sample token
        next_token = categorical_sample(probs, rand_data_sample)

        # Step 3: beam search
        new_draft_tokens, new_draft_logits = _beam_search_candidates(
            draft_input, next_token, embedding, drafter, self.num_beams,
            self.beam_length, self.is_rnn)

        # Step 4: tree processing
        active_tokens_flattened, new_draft_token_indices, new_mask, \
            new_position_offsets, packed_position_ids, next_num_gen_tokens, max_gen_token, \
            total_gen_token = _beams2tree(new_draft_tokens, num_beams, beam_length,
                                          position_ids_base + num_accepted_tokens)

        # Step 5: mark all the tensors we need
        num_accepted_tokens.mark_output('num_accepted_tokens')
        accepted_beam_index.mark_output('accepted_beam_index')
        max_gen_token.mark_output('max_gen_token')
        total_gen_token.mark_output('total_gen_token')
        next_num_gen_tokens.mark_output('next_spec_decoding_generation_lengths')
        active_tokens_flattened.mark_output('next_flat_tokens')
        new_draft_tokens.mark_output('next_draft_tokens')
        new_draft_logits.mark_output('next_draft_probs')
        new_draft_token_indices.mark_output('next_draft_indices')
        new_mask.mark_output('spec_decoding_mask')
        new_position_offsets.mark_output('next_spec_decoding_position_offsets')
        packed_position_ids.mark_output('packed_position_ids')

        return next_token, probs, draft_input

    def forward(self, *args, **kwargs):
        """
        0. run base model, get logits, hidden_states
        """

        extra_args = [
            'draft_tokens',
            'draft_indices',
            'draft_probs',
            'device_request_types',
            'redrafter_inverted_temperature',
            'rand_data_validation',
            'rand_data_sample',
            'position_ids_base',
        ]
        use_cache = True
        base_kwargs = {k: v for k, v in kwargs.items() if k not in extra_args}
        if use_cache and default_net().plugin_config.paged_kv_cache is False:
            lm_logits, presents, hidden_states = super().forward(
                *args, **base_kwargs)
        else:
            lm_logits, hidden_states, _ = super().forward(*args, **base_kwargs)

        # lm_logits could be in fp32
        lm_logits_cast = cast(lm_logits, self.dtype)  # no-op if same type
        self.register_network_output("hidden_states",
                                     hidden_states)  # debugging

        new_draft_tokens, new_draft_logits, probs = self._fwd_helper(
            hidden_states,
            lm_logits_cast,
            self.transformer.vocab_embedding,
            self.drafter,
            kwargs=kwargs)

        return new_draft_tokens, new_draft_logits, probs

    def prepare_inputs(self, *args, **kwargs):
        """
        Inputs needed:
            Assuming, max_gen_tokens = 1 + nb*(bl - 1), counting true token
            device_request_types: [bs]
            draft_tokens: [bs, nb, bl]
            draft_indices: [bs, nb, bl]
            draft_probs: [bs, nb, bl-1, V]
            spec_decoding_generation_lengths: [bs]
            spec_decoding_position_offsets: [bs, max_gen_tokens]
            spec_decoding_packed_mask: [bs, max_gen_tokens, packed_length] **
            redrafter_inverted_temperature: [bs]
            rand_data_sample: [bs]
            rand_data_validation: [bs, nb, bl-1]

            ** The mask is tricky since the boolean mask will need to be
               packed in runtime. So, the last dim will be:
                    packed_length = ceil(max_gen_tokens/32)
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
        num_beam_tokens = self.num_beams * self.beam_length
        max_draft_len = num_beam_tokens - self.num_beams  # ignore the true token
        max_gen_token_len = 1 + max_draft_len  # for the true token
        max_gen_token_len_range = default_range(max_gen_token_len)
        bb_max_gen_token_len_range = default_range(max_gen_token_len *
                                                   max_batch_size,
                                                   min_range=0)

        kwargs['speculative_decoding_draft_tokens_external'] = False
        kwargs['max_draft_len'] = max_draft_len
        kwargs['spec_decoding_is_generation_length_variable'] = True
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
            max_gen_token_len_range = [
                max_gen_token_len_range, max_gen_token_len_range
            ]
            bb_max_gen_token_len_range = [
                bb_max_gen_token_len_range, bb_max_gen_token_len_range
            ]
            num_beams_range = [self.num_beams, self.num_beams]
            beam_length_range = [self.beam_length, self.beam_length]
            candidate_length_range = [
                self.beam_candidate_length, self.beam_candidate_length
            ]
            vocab_size_range = [self.vocab_size, self.vocab_size]
        else:
            bb_range = [bb_range]
            bb0_range = [bb0_range]
            max_gen_token_len_range = [max_gen_token_len_range]
            bb_max_gen_token_len_range = [bb_max_gen_token_len_range]
            num_beams_range = [self.num_beams]
            beam_length_range = [self.beam_length]
            candidate_length_range = [self.beam_candidate_length]
            vocab_size_range = [self.vocab_size]

        device_request_types = Tensor(name='device_request_types',
                                      dtype=trt.int32,
                                      shape=[-1],
                                      dim_range=OrderedDict([
                                          ('batch_size', bb_range),
                                      ]))
        draft_tokens = Tensor(name='draft_tokens',
                              dtype=trt.int32,
                              shape=[-1, self.num_beams, self.beam_length],
                              dim_range=OrderedDict([
                                  ('batch_size_wt0', bb0_range),
                                  ('num_beams', num_beams_range),
                                  ('beam_length', beam_length_range),
                              ]))
        draft_indices = Tensor(name='draft_indices',
                               dtype=trt.int32,
                               shape=[-1, self.num_beams, self.beam_length],
                               dim_range=OrderedDict([
                                   ('batch_size_wt0', bb0_range),
                                   ('num_beams', num_beams_range),
                                   ('beam_length', beam_length_range),
                               ]))
        draft_probs = Tensor(
            name='draft_probs',
            dtype=self.dtype,
            shape=[-1, self.num_beams, self.beam_length - 1, self.vocab_size],
            dim_range=OrderedDict([
                ('batch_size_wt0', bb0_range),
                ('num_beams', num_beams_range),
                ('candidate_length', candidate_length_range),
                ('vocab_size', vocab_size_range),
            ]))
        redrafter_inverted_temperature = Tensor(
            name='redrafter_inverted_temperature',
            dtype=self.dtype,
            shape=[-1],
            dim_range=OrderedDict([
                ("batch_size", bb_range),
            ]))
        rand_data_validation = Tensor(
            name='rand_data_validation',
            dtype=self.dtype,
            shape=[-1, self.num_beams, self.beam_length - 1],
            dim_range=OrderedDict([
                ('batch_size_wt0', bb0_range),
                ('num_beams', num_beams_range),
                ('candidate_length', candidate_length_range),
            ]))
        rand_data_sample = Tensor(name='rand_data_sample',
                                  dtype=self.dtype,
                                  shape=[-1],
                                  dim_range=OrderedDict([
                                      ('batch_size', bb_range),
                                  ]))
        position_ids_base = Tensor(
            name="position_ids_base",
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([
                ("batch_size", bb_range),
            ]),
        )

        inputs[
            'device_request_types'] = device_request_types  # needed by process_logits
        inputs['draft_tokens'] = draft_tokens
        inputs['draft_indices'] = draft_indices
        inputs['draft_probs'] = draft_probs
        inputs[
            'redrafter_inverted_temperature'] = redrafter_inverted_temperature
        inputs['rand_data_validation'] = rand_data_validation
        inputs['rand_data_sample'] = rand_data_sample
        inputs['position_ids_base'] = position_ids_base
        return inputs


class ReDrafterForQWenLM(ReDrafterMixin, QWenForCausalLM):
    """ReDrafter implementation for QWen models.

    Combines:
    - Base QWen model functionality from QWenForCausalLM
    - Drafting/speculative decoding logic from ReDrafterMixin
    """


class ReDrafterForLLaMALM(ReDrafterMixin, LLaMAForCausalLM):
    """ReDrafter implementation for LLaMA models.

    Combines:
    - Base LLaMA model functionality from LLaMAForCausalLM
    - Drafting/speculative decoding logic from ReDrafterMixin
    """

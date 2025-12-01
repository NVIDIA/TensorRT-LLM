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
import os
from collections import OrderedDict
from typing import List, Optional, Union

import tensorrt as trt
from transformers import AutoModelForCausalLM

from ..._common import default_net
from ..._utils import str_dtype_to_trt
from ...functional import (Tensor, arange, cast, concat, expand,
                           gather_last_token_logits, shape, unsqueeze)
from ...layers import ColumnLinear, Embedding, LayerNorm, Mamba, Mamba2, RmsNorm
from ...mapping import Mapping
from ...module import Module, ModuleList
from ...plugin import current_all_reduce_helper
from ..generation_mixin import GenerationMixin
from ..modeling_utils import PretrainedConfig, PretrainedModel, QuantConfig
from .config import MambaConfig
from .convert import convert_from_hf_checkpoint, convert_hf_mamba


class MambaLayer(Module):

    def __init__(self, config: PretrainedConfig, layer_idx: int):
        super().__init__()
        self.dtype = config.dtype
        self.residual_in_fp32 = config.residual_in_fp32
        n_layer = config.num_hidden_layers
        self.last_layer = layer_idx == n_layer - 1

        if config.mamba_version == 'Mamba1':
            assert config.mapping.tp_size == 1, "Mamba1 can not support tensor parallelism."
            self.ssm = Mamba(config.hidden_size,
                             config.rnn_hidden_size,
                             d_state=config.state_size,
                             d_conv=config.conv_kernel,
                             bias=config.use_bias,
                             dtype=config.dtype)
        elif config.mamba_version == 'Mamba2':
            self.ssm = Mamba2(config.hidden_size,
                              config.rnn_hidden_size,
                              d_state=config.state_size,
                              d_conv=config.conv_kernel,
                              headdim=config.rnn_head_size,
                              ngroups=config.ngroups,
                              chunk_size=config.chunk_size,
                              bias=config.use_bias,
                              rmsnorm=config.ssm_rmsnorm,
                              dtype=config.dtype,
                              tp_group=config.mapping.tp_group,
                              tp_size=config.mapping.tp_size)
        if config.rms_norm:
            self.input_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                           eps=config.norm_epsilon,
                                           dtype=config.dtype)
        else:
            self.input_layernorm = LayerNorm(
                normalized_shape=config.hidden_size,
                eps=config.norm_epsilon,
                dtype=config.dtype)

    def forward(self,
                hidden_states: Tensor,
                residual: Tensor,
                conv_state: Tensor,
                ssm_state: Tensor,
                host_request_types: Tensor,
                last_token_ids: Tensor,
                host_context_lengths: Optional[Tensor] = None,
                slot_mapping: Optional[Tensor] = None,
                conv_indices: Optional[Tensor] = None):

        hidden_states = self.input_layernorm(hidden_states)

        ssm_out, present_conv, present_ssm = self.ssm(
            hidden_states,
            conv_state=conv_state,
            ssm_state=ssm_state,
            host_request_types=host_request_types,
            last_token_ids=last_token_ids,
            host_context_lengths=host_context_lengths,
            slot_mapping=slot_mapping,
            conv_indices=conv_indices)
        if self.residual_in_fp32:
            residual = residual + cast(ssm_out, 'float32')
            hidden_states = cast(residual, self.dtype)
        else:
            residual = residual + ssm_out
            hidden_states = residual

        if self.last_layer:
            return hidden_states, None, present_conv, present_ssm
        else:
            return hidden_states, residual, present_conv, present_ssm


class MambaModel(Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.d_conv = config.conv_kernel
        self.d_inner = config.rnn_hidden_size // config.mapping.tp_size
        n_layer = config.num_hidden_layers
        self.residual_in_fp32 = config.residual_in_fp32
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple)
        self.vocab_embedding = Embedding(config.vocab_size,
                                         config.hidden_size,
                                         dtype=config.dtype)
        self.layers = ModuleList(
            [MambaLayer(config, i) for i in range(n_layer)])
        if config.rms_norm:
            self.ln_f = RmsNorm(normalized_shape=config.hidden_size,
                                eps=config.norm_epsilon,
                                dtype=config.dtype)
        else:
            self.ln_f = LayerNorm(normalized_shape=config.hidden_size,
                                  eps=config.norm_epsilon,
                                  dtype=config.dtype)

    def forward(self,
                input_ids,
                conv_states,
                ssm_states,
                host_request_types,
                last_token_ids,
                host_context_lengths,
                slot_mapping: Optional[Tensor] = None):
        hidden_states = self.vocab_embedding(input_ids)

        # Get conv state indices
        indices = None
        if not default_net().plugin_config.mamba_conv1d_plugin:
            batch_size = shape(input_ids, 0)
            indices = expand(
                unsqueeze(arange(0, self.d_conv - 1, dtype='int32'), 0),
                concat([batch_size, self.d_conv - 1]))
            offsets = expand(unsqueeze(last_token_ids, 1),
                             concat([batch_size, self.d_conv - 1]))
            indices = unsqueeze(indices + offsets, 1)
            indices = expand(
                indices, concat([batch_size, self.d_inner, self.d_conv - 1]))

        residual = cast(hidden_states,
                        'float32') if self.residual_in_fp32 else hidden_states
        hidden_values = [hidden_states, residual]
        present_convs, present_ssms = [], []
        for layer, past_conv, past_ssm in zip(self.layers, conv_states,
                                              ssm_states):
            hidden_values = layer(hidden_values[0], hidden_values[1], past_conv,
                                  past_ssm, host_request_types, last_token_ids,
                                  host_context_lengths, slot_mapping, indices)
            present_convs.append(hidden_values[2])
            present_ssms.append(hidden_values[3])
        hidden_states = hidden_values[0]
        hidden_states = self.ln_f(hidden_states)
        return hidden_states, tuple(present_convs), tuple(present_ssms)


class MambaForCausalLM(PretrainedModel):
    config_class = MambaConfig

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        dtype = config.dtype
        logits_dtype = config.logits_dtype
        if isinstance(dtype, str):
            self.dtype = str_dtype_to_trt(dtype)
        else:
            assert isinstance(dtype, trt.DataType)
            self.dtype = dtype

        self.config = config
        self.mamba_version = config.mamba_version
        self.d_inner = config.rnn_hidden_size // config.mapping.tp_size
        self.d_conv = config.conv_kernel
        self.d_state = config.state_size
        self.conv_dim = config.rnn_conv_dim_size // config.mapping.tp_size
        self.gather_context_logits = False

        if isinstance(logits_dtype, str):
            self._logits_dtype = str_dtype_to_trt(logits_dtype)
        else:
            assert isinstance(logits_dtype, trt.DataType)
            self._logits_dtype = logits_dtype

        self.backbone = MambaModel(config)
        self.lm_head = ColumnLinear(config.hidden_size,
                                    config.vocab_size,
                                    bias=False,
                                    dtype=dtype,
                                    tp_group=config.mapping.tp_group,
                                    tp_size=config.mapping.tp_size,
                                    gather_output=True)

    def __post_init__(self):
        return

    def forward(self,
                input_ids,
                conv_states,
                ssm_states,
                host_request_types,
                last_token_ids,
                last_token_ids_for_logits,
                host_context_lengths,
                slot_mapping: Optional[Tensor] = None):
        hidden_states, present_convs, present_ssms = self.backbone(
            input_ids, conv_states, ssm_states, host_request_types,
            last_token_ids, host_context_lengths, slot_mapping)

        if not self.gather_context_logits:
            hidden_states = gather_last_token_logits(
                hidden_states, last_token_ids_for_logits,
                default_net().plugin_config.remove_input_padding)

        lm_logits = self.lm_head(hidden_states)
        lm_logits.mark_output('logits', self._logits_dtype)
        if not default_net().plugin_config.paged_state:
            for i, present_conv in enumerate(present_convs):
                present_conv.mark_output(f'present_conv_state_{i}', self.dtype)
            for i, present_ssm in enumerate(present_ssms):
                present_ssm.mark_output(f'present_rnn_state_{i}', self.dtype)

        return (lm_logits, present_convs, present_ssms)

    def prepare_inputs(
            self,
            max_batch_size,
            max_input_len,
            max_seq_len,
            max_num_tokens,
            use_cache,
            max_beam_width: int = 1,
            opt_num_tokens: int = None,
            opt_batch_size: int = 0,
            prompt_embedding_table_size: int = 0,
            max_draft_len: int = 0,
            gather_context_logits: bool = False,
            lora_target_modules: List[str] = None,
            speculative_decoding_draft_tokens_external: bool = False):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''
        assert speculative_decoding_draft_tokens_external == False, "Speculative decoding is not supported in Mamba"
        assert max_beam_width == 1, "We don't support beam search for the Mamba model."

        remove_input_padding = default_net().plugin_config.remove_input_padding
        use_gemm_plugin = default_net().plugin_config.gemm_plugin
        paged_state = default_net().plugin_config.paged_state
        multiple_profiles = default_net().plugin_config.multiple_profiles
        use_mamba_conv1d_plugin = default_net(
        ).plugin_config.mamba_conv1d_plugin

        self.gather_context_logits = gather_context_logits
        mapping = self.config.mapping

        # basic inputs
        enable_ctx_gen_opt_profiles = GenerationMixin.has_ctx_gen_opt_profiles(
            use_gemm_plugin=use_gemm_plugin,
            use_mamba_conv1d_plugin=use_mamba_conv1d_plugin,
            remove_input_padding=remove_input_padding,
            paged_state=paged_state)

        num_profiles, ranges = GenerationMixin.get_profiles_ranges(
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_input_len=max_input_len,
            max_num_tokens=max_num_tokens,
            max_draft_len=max_draft_len,
            opt_batch_size=opt_batch_size,
            opt_num_tokens=opt_num_tokens,
            enable_ctx_gen_opt_profiles=enable_ctx_gen_opt_profiles,
            multiple_profiles=multiple_profiles)

        if remove_input_padding:
            assert use_mamba_conv1d_plugin, "mamba_conv1d_plugin is needed to support remove_input_padding"
            input_ids = Tensor(name='input_ids',
                               dtype=trt.int32,
                               shape=[-1],
                               dim_range=OrderedDict([
                                   ('num_tokens', ranges['num_tokens_range']),
                               ]))
        else:
            input_ids = Tensor(name='input_ids',
                               dtype=trt.int32,
                               shape=[-1, -1],
                               dim_range=OrderedDict([
                                   ('batch_size_beam_width',
                                    ranges['bb_range']),
                                   ('input_len', ranges['inlen_range']),
                               ]))
        if mapping.tp_size > 1:
            current_all_reduce_helper().set_workspace_tensor(
                mapping, num_profiles)

        # recurrent inputs
        conv_states = []
        ssm_states = []
        if use_mamba_conv1d_plugin:
            conv_state_dim_range = OrderedDict([
                ('batch_size', ranges['bb_range']),
                ('kernel_size', [self.d_conv - 1] * num_profiles),
                ('dim_size', [self.conv_dim] * num_profiles),
            ])
        else:
            conv_state_dim_range = OrderedDict([
                ('batch_size', ranges['bb_range']),
                ('dim_size', [self.conv_dim] * num_profiles),
                ('kernel_size', [self.d_conv - 1] * num_profiles),
            ])

        if self.mamba_version == 'Mamba2':
            headdim = self.config.rnn_head_size
            nheads = self.d_inner // headdim
            ssm_state_dim_range = OrderedDict([
                ('batch_size', ranges['bb_range']),
                ('head_size', [nheads] * num_profiles),
                ('state_size', [self.d_state] * num_profiles),
                ('headdim_size', [headdim] * num_profiles),
            ])
            ssm_state_shape = [-1, nheads, self.d_state, headdim]
        else:
            ssm_state_dim_range = OrderedDict([
                ('batch_size', ranges['bb_range']),
                ('state_size', [self.d_state] * num_profiles),
                ('dim_size', [self.d_inner] * num_profiles),
            ])
            ssm_state_shape = [-1, self.d_state, self.d_inner]
        one_dim_range = OrderedDict([
            ('buffer_count', [1] * num_profiles),
        ])

        for i in range(self.config.num_hidden_layers):
            if default_net().plugin_config.paged_state:
                conv_state = Tensor(name=f'conv_state_ptr_{i}',
                                    dtype=str_dtype_to_trt('int64'),
                                    shape=[1],
                                    dim_range=one_dim_range)

                ssm_state = Tensor(name=f'rnn_state_ptr_{i}',
                                   dtype=str_dtype_to_trt('int64'),
                                   shape=[1],
                                   dim_range=one_dim_range)
            else:
                if use_mamba_conv1d_plugin:
                    conv_state = Tensor(
                        name=f'past_conv_state_{i}',
                        dtype=self.dtype,
                        shape=[-1, self.d_conv - 1, self.conv_dim],
                        dim_range=conv_state_dim_range)
                else:
                    conv_state = Tensor(
                        name=f'past_conv_state_{i}',
                        dtype=self.dtype,
                        shape=[-1, self.conv_dim, self.d_conv - 1],
                        dim_range=conv_state_dim_range)

                ssm_state = Tensor(name=f'past_rnn_state_{i}',
                                   dtype=self.dtype,
                                   shape=ssm_state_shape,
                                   dim_range=ssm_state_dim_range)

            conv_states.append(conv_state)
            ssm_states.append(ssm_state)

        host_request_types = Tensor(
            name='host_request_types',
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([('batch_size', ranges['bb_range'])]),
        )

        if remove_input_padding:
            host_context_lengths = Tensor(
                name='host_context_lengths',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('batch_size', ranges['bb_range'])]),
            )
        else:
            host_context_lengths = None

        last_token_ids = Tensor(
            name='last_token_ids',
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([
                ('batch_size', ranges['bbd_range']),
            ]),
        )
        last_token_ids_for_logits = None
        if not gather_context_logits:
            last_token_ids_for_logits = last_token_ids

        return_dict = {
            'input_ids': input_ids,
            'conv_states': conv_states,
            'ssm_states': ssm_states,
            'host_request_types': host_request_types,
            'last_token_ids': last_token_ids,
            'last_token_ids_for_logits': last_token_ids_for_logits,
            'host_context_lengths': host_context_lengths,
        }

        if default_net().plugin_config.paged_state:
            slot_mapping = Tensor(
                name='slot_mapping',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('batch_size', ranges['bb_range'])]),
            )
            return_dict['slot_mapping'] = slot_mapping

        return return_dict

    @classmethod
    def from_hugging_face(
            cls,
            hf_model_or_dir: Union[str, 'transformers.PreTrainedModel'],
            dtype: str = 'auto',
            mapping: Optional[Mapping] = None,
            quant_config: Optional[QuantConfig] = None,
            **kwargs):
        import transformers

        assert hf_model_or_dir is not None
        use_preloading = isinstance(hf_model_or_dir,
                                    transformers.PreTrainedModel)
        if use_preloading:
            hf_model = hf_model_or_dir
            hf_config_or_dir = hf_model.config
        else:
            hf_model_dir = hf_model_or_dir
            hf_config_or_dir = hf_model_or_dir
        config = MambaConfig.from_hugging_face(hf_config_or_dir,
                                               dtype=dtype,
                                               mapping=mapping,
                                               quant_config=quant_config,
                                               **kwargs)

        if not os.path.exists(hf_model_dir):
            hf_model = AutoModelForCausalLM.from_pretrained(
                hf_model_dir, dtype="auto", trust_remote_code=True)

            assert isinstance(hf_model, transformers.PreTrainedModel)
            weights = convert_hf_mamba(hf_model, dtype)
        else:
            weights = convert_from_hf_checkpoint(config, hf_model_dir)

        model = cls(config)
        model.load(weights)

        return model

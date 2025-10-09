# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Any, Dict, List, Optional

from ..._utils import str_dtype_to_torch
from ...functional import (Tensor, allgather, chunk, concat, einsum, pad, shape,
                           unsqueeze)
from ...layers import LayerNorm, Linear
from ...layers.attention import DiffusersAttention
from ...layers.embedding import (CombinedTimestepTextProjEmbeddings,
                                 SD3PatchEmbed)
from ...layers.mlp import (LinearActivation, LinearApproximateGELU, LinearGEGLU,
                           LinearGELU, LinearSwiGLU)
from ...layers.normalization import (AdaLayerNormContinuous, AdaLayerNormZero,
                                     SD35AdaLayerNormZeroX)
from ...logger import logger
from ...mapping import Mapping
from ...module import Module, ModuleList
from ..model_weights_loader import ModelWeightsLoader
from ..modeling_utils import PretrainedModel
from .config import SD3Transformer2DModelConfig


class FeedForward(Module):

    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            mult: int = 4,
            activation_fn: str = "geglu",
            inner_dim=None,
            bias: bool = True,
            mapping=Mapping(),
            dtype=None,
    ):
        super().__init__()

        self.mapping = mapping
        self.dtype = dtype

        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            raise NotImplementedError('GELU only support tanh now.')
        if activation_fn == "gelu-approximate":
            act_fn = LinearGELU(dim,
                                inner_dim,
                                approximate="tanh",
                                bias=bias,
                                mapping=mapping,
                                dtype=dtype)
        elif activation_fn == "geglu":
            act_fn = LinearGEGLU(dim,
                                 inner_dim,
                                 approximate="tanh",
                                 bias=bias,
                                 mapping=mapping,
                                 dtype=dtype)
        elif activation_fn == "geglu-approximate":
            act_fn = LinearApproximateGELU(dim,
                                           inner_dim,
                                           bias=bias,
                                           mapping=mapping,
                                           dtype=dtype)
        elif activation_fn == "swiglu":
            act_fn = LinearSwiGLU(dim,
                                  inner_dim,
                                  bias=bias,
                                  mapping=mapping,
                                  dtype=dtype)
        elif activation_fn == "linear-silu":
            act_fn = LinearActivation(dim,
                                      inner_dim,
                                      bias=bias,
                                      activation="silu",
                                      mapping=mapping,
                                      dtype=dtype)

        self.net = ModuleList([
            act_fn,
            Linear(inner_dim,
                   dim_out,
                   bias=bias,
                   tp_group=self.mapping.tp_group,
                   tp_size=self.mapping.tp_size,
                   dtype=self.dtype)
        ])

    def forward(self, hidden_states: Tensor):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class JointTransformerBlock(Module):

    def __init__(self,
                 dim: int,
                 num_attention_heads: int,
                 attention_head_dim: int,
                 context_pre_only: bool = False,
                 qk_norm: Optional[str] = None,
                 use_dual_attention: bool = False,
                 mapping=Mapping(),
                 dtype=None):
        super().__init__()

        self.use_dual_attention = use_dual_attention
        self.context_pre_only = context_pre_only
        context_norm_type = "ada_norm_continous" if context_pre_only else "ada_norm_zero"

        if use_dual_attention:
            self.norm1 = SD35AdaLayerNormZeroX(dim,
                                               mapping=mapping,
                                               dtype=dtype)
        else:
            self.norm1 = AdaLayerNormZero(dim, mapping=mapping, dtype=dtype)

        if context_norm_type == "ada_norm_continous":
            self.norm1_context = AdaLayerNormContinuous(
                dim,
                dim,
                elementwise_affine=False,
                eps=1e-6,
                bias=True,
                norm_type="layer_norm",
                dtype=dtype)
        elif context_norm_type == "ada_norm_zero":
            self.norm1_context = AdaLayerNormZero(dim, dtype=dtype)
        else:
            raise ValueError(
                f"Unknown context_norm_type: {context_norm_type}, currently only support `ada_norm_continous`, `ada_norm_zero`"
            )

        self.attn = DiffusersAttention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=context_pre_only,
            bias=True,
            qk_norm=qk_norm,
            eps=1e-6,
            mapping=mapping,
            dtype=dtype,
        )

        if use_dual_attention:
            self.attn2 = DiffusersAttention(
                query_dim=dim,
                cross_attention_dim=None,
                dim_head=attention_head_dim,
                heads=num_attention_heads,
                out_dim=dim,
                bias=True,
                qk_norm=qk_norm,
                eps=1e-6,
                mapping=mapping,
                dtype=dtype,
            )
        else:
            self.attn2 = None

        self.norm2 = LayerNorm(dim,
                               elementwise_affine=False,
                               eps=1e-6,
                               dtype=dtype)
        self.ff = FeedForward(dim=dim,
                              dim_out=dim,
                              activation_fn="gelu-approximate",
                              mapping=mapping,
                              dtype=dtype)

        if not context_pre_only:
            self.norm2_context = LayerNorm(dim,
                                           elementwise_affine=False,
                                           eps=1e-6,
                                           dtype=dtype)
            self.ff_context = FeedForward(dim=dim,
                                          dim_out=dim,
                                          activation_fn="gelu-approximate",
                                          mapping=mapping,
                                          dtype=dtype)
        else:
            self.norm2_context = None
            self.ff_context = None

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self,
                               chunk_size: Optional[int] = None,
                               dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    @staticmethod
    def _chunked_feed_forward(ff: Module, hidden_states: Tensor, chunk_dim: int,
                              chunk_size: int):
        # "feed_forward_chunk_size" can be used to save memory
        if hidden_states.shape[chunk_dim] % chunk_size != 0:
            raise ValueError(
                f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
            )

        num_chunks = hidden_states.shape[chunk_dim] // chunk_size
        ff_output = concat(
            [
                ff(hid_slice)
                for hid_slice in chunk(hidden_states, num_chunks, dim=chunk_dim)
            ],
            dim=chunk_dim,
        )
        return ff_output

    def forward(self,
                hidden_states: Tensor,
                encoder_hidden_states: Tensor,
                temb: Tensor,
                joint_attention_kwargs: Optional[Dict[str, Any]] = None,
                *args,
                **kwargs):
        joint_attention_kwargs = joint_attention_kwargs or {}
        if self.use_dual_attention:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2 = self.norm1(
                hidden_states, emb=temb)
        else:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, emb=temb)

        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(
                encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb)

        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            **joint_attention_kwargs,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = unsqueeze(gate_msa, 1) * attn_output
        hidden_states = hidden_states + attn_output

        if self.use_dual_attention:
            attn_output2 = self.attn2(hidden_states=norm_hidden_states2,
                                      **joint_attention_kwargs)
            attn_output2 = unsqueeze(gate_msa2, 1) * attn_output2
            hidden_states = hidden_states + attn_output2

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (
            1 + unsqueeze(scale_mlp, 1)) + unsqueeze(shift_mlp, 1)

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = self._chunked_feed_forward(self.ff, norm_hidden_states,
                                                   self._chunk_dim,
                                                   self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)
        ff_output = unsqueeze(gate_mlp, 1) * ff_output
        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = unsqueeze(c_gate_msa, 1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(
                encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (
                1 + unsqueeze(c_scale_mlp, 1)) + unsqueeze(c_shift_mlp, 1)
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output = self._chunked_feed_forward(
                    self.ff_context, norm_encoder_hidden_states,
                    self._chunk_dim, self._chunk_size)
            else:
                context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + unsqueeze(
                c_gate_mlp, 1) * context_ff_output

        return encoder_hidden_states, hidden_states


class SD3Transformer2DModel(PretrainedModel):
    config_class = SD3Transformer2DModelConfig

    def __init__(self, config: SD3Transformer2DModelConfig):
        super().__init__(config)
        self.quant_mode = config.quant_mode
        self.mapping = config.mapping
        self.dtype = config.dtype

        self.in_channels = config.in_channels
        default_out_channels = config.in_channels
        self.out_channels = config.out_channels if config.out_channels is not None else default_out_channels
        self.inner_dim = config.num_attention_heads * config.attention_head_dim

        self.pos_embed = SD3PatchEmbed(
            height=config.sample_size,
            width=config.sample_size,
            patch_size=config.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=config.
            pos_embed_max_size,  # hard-code as HF implementation
            dtype=self.dtype)
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=config.pooled_projection_dim,
            mapping=self.mapping,
            dtype=self.dtype)
        self.context_embedder = Linear(config.joint_attention_dim,
                                       config.caption_projection_dim,
                                       tp_group=self.mapping.tp_group,
                                       tp_size=self.mapping.tp_size,
                                       dtype=self.dtype)

        self.transformer_blocks = ModuleList([
            JointTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=config.num_attention_heads,
                attention_head_dim=config.attention_head_dim,
                context_pre_only=(i == config.num_layers - 1),
                qk_norm=config.qk_norm,
                use_dual_attention=True
                if i in config.dual_attention_layers else False,
                mapping=self.mapping,
                dtype=self.dtype) for i in range(config.num_layers)
        ])

        self.norm_out = AdaLayerNormContinuous(self.inner_dim,
                                               self.inner_dim,
                                               elementwise_affine=False,
                                               eps=1e-6,
                                               dtype=self.dtype)
        self.proj_out = Linear(self.inner_dim,
                               config.patch_size * config.patch_size *
                               self.out_channels,
                               bias=True,
                               tp_group=self.mapping.tp_group,
                               tp_size=self.mapping.tp_size,
                               dtype=self.dtype)

        self.skip_layers = config.skip_layers
        self.use_pretrained_pos_emb = config.use_pretrained_pos_emb
        self.config = config

    def forward(self,
                hidden_states: Tensor,
                encoder_hidden_states: Optional[Tensor] = None,
                pooled_projections: Optional[Tensor] = None,
                timestep: Optional[Tensor] = None,
                block_controlnet_hidden_states: List[Tensor] = None,
                joint_attention_kwargs: Optional[Dict[str, Any]] = None):
        height, width = hidden_states.shape[-2:]
        hidden_states = self.pos_embed(
            hidden_states)  # takes care of adding positional embeddings too.
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if self.mapping.cp_size > 1:
            hidden_states = chunk(hidden_states,
                                  chunks=self.mapping.cp_size,
                                  dim=1)[self.mapping.cp_rank]
            encoder_redundant = encoder_hidden_states.shape[
                1] % self.mapping.cp_size
            encoder_padding_index = tuple(
                [0, 0] * (encoder_hidden_states.ndim() - 2) +
                [0, self.mapping.cp_size - encoder_redundant])
            if encoder_redundant != 0:
                encoder_hidden_states = pad(encoder_hidden_states,
                                            pad=encoder_padding_index)
            encoder_hidden_states = chunk(encoder_hidden_states,
                                          chunks=self.mapping.cp_size,
                                          dim=1)[self.mapping.cp_rank]
        for index_block, block in enumerate(self.transformer_blocks):
            # Skip specified layers
            is_skip = True if self.skip_layers is not None and index_block in self.skip_layers else False

            if not is_skip:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                interval_control = len(self.transformer_blocks) / len(
                    block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[
                    int(index_block / interval_control)]

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)
        if self.mapping.cp_size > 1:
            hidden_states = allgather(hidden_states,
                                      group=self.mapping.cp_group,
                                      gather_dim=1)

        # unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.view(
            concat([
                shape(hidden_states, 0), height, width, patch_size, patch_size,
                self.out_channels
            ]))
        hidden_states = einsum("nhwpqc->nchpwq", [hidden_states])
        output = hidden_states.view(
            concat([
                shape(hidden_states, 0), self.out_channels, height * patch_size,
                width * patch_size
            ]))

        output.mark_output("output")
        return output

    def prepare_inputs(self, max_batch_size, **kwargs):

        def sd3_default_range(max_batch_size):
            return [1, max(1, (max_batch_size + 1) // 2), max_batch_size]

        default_range = sd3_default_range
        prompt_embeds_len = 256 + 77  # [NOTE] tokenizer_max_length = 77; max_sequence_length = 256

        hidden_states = Tensor(name='hidden_states',
                               dtype=self.dtype,
                               shape=[
                                   -1, self.in_channels,
                                   self.config.sample_size,
                                   self.config.sample_size
                               ],
                               dim_range=OrderedDict([
                                   ('batch_size',
                                    [default_range(max_batch_size)]),
                                   ('in_channels', [[self.in_channels] * 3]),
                                   ('height', [[self.config.sample_size] * 3]),
                                   ('width', [[self.config.sample_size] * 3]),
                               ]))
        encoder_hidden_states = Tensor(
            name='encoder_hidden_states',
            dtype=self.dtype,
            shape=[-1, prompt_embeds_len, self.config.joint_attention_dim],
            dim_range=OrderedDict([
                ('batch_size', [default_range(max_batch_size)]),
                ('txt_len', [[prompt_embeds_len] * 3]),
                ('joint_attention_dim', [[self.config.joint_attention_dim] * 3
                                         ]),
            ]))
        pooled_projections = Tensor(
            name='pooled_projections',
            dtype=self.dtype,
            shape=[-1, self.config.pooled_projection_dim],
            dim_range=OrderedDict([
                ('batch_size', [default_range(max_batch_size)]),
                ('pooled_projection_dim',
                 [[self.config.pooled_projection_dim] * 3]),
            ]))
        timestep = Tensor(name='timestep',
                          dtype=self.dtype,
                          shape=[-1],
                          dim_range=OrderedDict([
                              ('batch_size', [default_range(max_batch_size)]),
                          ]))
        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_projections,
            "timestep": timestep,
        }

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path: str,
                        dtype='float16',
                        mapping=Mapping(),
                        **kwargs):
        quant_ckpt_path = kwargs.pop('quant_ckpt_path', None)

        from diffusers import StableDiffusion3Pipeline

        transformer = StableDiffusion3Pipeline.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=str_dtype_to_torch(dtype)).transformer

        config = SD3Transformer2DModelConfig.from_hugging_face_config(
            transformer.config, dtype=dtype, mapping=mapping, **kwargs)

        hf_model_dir = transformer.config._name_or_path
        custom_dict = {}
        if quant_ckpt_path is not None:
            hf_model_dir = quant_ckpt_path

        loader = SD3ModelWeightsLoader(hf_model_dir, custom_dict)
        model = cls(config)
        loader.generate_tllm_weights(model)
        return model

    def load(self, weights, from_pruned=False):
        required_names = set()
        for name, param in self.named_parameters():
            if self.use_pretrained_pos_emb and 'pos_embed' in name:
                required_names.add(name)
                continue
            if param.is_inited():
                continue
            if name not in weights:
                # Exemption for embedding sharing
                if name.endswith('lm_head.weight') and any(
                        k.endswith('vocab_embedding.weight')
                        for k in weights.keys()):
                    continue
                if name.endswith('lm_head.per_channel_scale') and any(
                        k.endswith('vocab_embedding.per_channel_scale')
                        for k in weights.keys()):
                    continue
            required_names.add(name)

        provided_names = set(weights.keys())
        if not required_names.issubset(provided_names):
            raise RuntimeError(
                f"Required but not provided tensors:{required_names.difference(provided_names)}"
            )
        if not provided_names.issubset(required_names):
            logger.warning(
                f"Provided but not required tensors: {provided_names.difference(required_names)}"
            )

        for name, param in self.named_parameters():
            if name in provided_names:
                if not from_pruned:
                    try:
                        param.value = weights[name]
                    except Exception as e:
                        raise RuntimeError(
                            f"Encounter error '{e}' for parameter '{name}'")
                else:
                    param.set_value_or_dummy(weights[name])

    def enable_forward_chunking(self,
                                chunk_size: Optional[int] = None,
                                dim: int = 0):
        raise NotImplementedError()

    def disable_forward_chunking(self):
        raise NotImplementedError()

    @property
    def attn_processors(self):
        return None

    def set_attn_processor(self, processor):
        raise NotImplementedError()

    def fuse_qkv_projections(self):
        raise NotImplementedError()

    def unfuse_qkv_projections(self):
        raise NotImplementedError()

    def _set_gradient_checkpointing(self, module, value=False):
        raise NotImplementedError()


class SD3ModelWeightsLoader(ModelWeightsLoader):

    def translate_to_external_key(self, tllm_key: str,
                                  tllm_to_externel_key_dict: dict):
        """Convert and load external checkpoint into a TensorRT LLM model.
        """
        trtllm_to_hf_name = {
            r"transformer_blocks.(\d+).ff(\w*).net.1.weight":
            "transformer_blocks.*.ff*.net.2.weight",
            r"transformer_blocks.(\d+).ff(\w*).net.1.bias":
            "transformer_blocks.*.ff*.net.2.bias",
        }
        import re
        for k, v in trtllm_to_hf_name.items():
            m = re.match(k, tllm_key)
            if m is not None:
                matched_pos = m.groups()
                placeholders = v.count('*')
                assert len(matched_pos) == placeholders
                for i in range(len(matched_pos)):
                    v = v.replace('*', matched_pos[i], 1)
                return v
        return tllm_key

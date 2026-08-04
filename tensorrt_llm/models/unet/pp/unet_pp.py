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
from ....functional import allgather, concat, slice, stack
from ....layers import Conv2d, GroupNorm
from ....mapping import Mapping
from ....module import Module
from ..attention import CrossAttention, SelfAttention
from ..unet_2d_condition import UNet2DConditionModel
from .attention import DistriCrossAttentionPP, DistriSelfAttentionPP
from .conv2d import DistriConv2dPP
from .groupnorm import DistriGroupNorm


class DistriUNetPP(Module):

    def __init__(self,
                 model: UNet2DConditionModel,
                 mapping: Mapping = Mapping()):
        super().__init__()
        self.mapping = mapping
        self.model = model
        if mapping.tp_size > 1:
            for name, module in model.named_modules():
                if isinstance(module, DistriConv2dPP) or isinstance(module, DistriSelfAttentionPP) \
                        or isinstance(module, DistriCrossAttentionPP) or isinstance(module, DistriGroupNorm):
                    continue
                for subname, submodule in module.named_children():
                    if isinstance(submodule, Conv2d):
                        kernel_size = submodule.kernel_size
                        if kernel_size == (1, 1) or kernel_size == 1:
                            continue
                        wrapped_submodule = DistriConv2dPP(
                            submodule,
                            mapping,
                            is_first_layer=subname == "conv_in")
                        setattr(module, subname, wrapped_submodule)
                    elif isinstance(submodule, SelfAttention):
                        wrapped_submodule = DistriSelfAttentionPP(
                            submodule, mapping)
                        setattr(module, subname, wrapped_submodule)
                    elif isinstance(submodule, CrossAttention):
                        wrapped_submodule = DistriCrossAttentionPP(
                            submodule, mapping)
                        setattr(module, subname, wrapped_submodule)
                    elif isinstance(submodule, GroupNorm):
                        wrapped_submodule = DistriGroupNorm(submodule, mapping)
                        setattr(module, subname, wrapped_submodule)

    def forward(self,
                sample,
                timesteps,
                encoder_hidden_states,
                text_embeds=None,
                time_ids=None):
        mapping = self.mapping
        b, c, h, w = sample.shape

        if mapping.world_size == 1:
            output = self.model(
                sample,
                timesteps,
                encoder_hidden_states,
                text_embeds=text_embeds,
                time_ids=time_ids,
            )
        elif mapping.pp_size > 1:
            assert b == 2 and mapping.pp_size == 2
            batch_idx = mapping.pp_rank
            # sample[batch_idx : batch_idx + 1]
            sample = slice(sample, [batch_idx, 0, 0, 0], [1, c, h, w])
            e_shape = encoder_hidden_states.shape
            encoder_hidden_states = slice(
                encoder_hidden_states, [batch_idx, 0, 0],
                [1, e_shape[1], e_shape[2]
                 ])  # encoder_hidden_states[batch_idx : batch_idx + 1]
            if text_embeds:
                t_shape = text_embeds.shape
                # text_embeds[batch_idx : batch_idx + 1]
                text_embeds = slice(text_embeds, [batch_idx, 0],
                                    [1, t_shape[1]])
            if time_ids:
                t_shape = time_ids.shape
                # time_ids[batch_idx : batch_idx + 1]
                time_ids = slice(time_ids, [batch_idx, 0], [1, t_shape[1]])
            output = self.model(
                sample,
                timesteps,
                encoder_hidden_states,
                text_embeds=text_embeds,
                time_ids=time_ids,
            )
            output = allgather(
                output,
                [i for i in range(mapping.world_size)],
            )
            patch_list = []
            for i in range(mapping.tp_size):
                patch_list.append(output.select(dim=0, index=i))
            b1 = concat(patch_list, dim=1)
            patch_list = []
            for i in range(mapping.tp_size, mapping.world_size):
                patch_list.append(output.select(dim=0, index=i))
            b2 = concat(patch_list, dim=1)
            output = stack([b1, b2], dim=0)
        else:
            output = self.model(
                sample,
                timesteps,
                encoder_hidden_states,
                text_embeds=text_embeds,
                time_ids=time_ids,
            )
            output = allgather(output, mapping.tp_group, 2)

        return output

    @property
    def add_embedding(self):
        return self.model.add_embedding

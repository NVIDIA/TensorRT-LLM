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
import time

import numpy as np

from ...logger import logger


def update_timestep_weight(src, dst):
    dst.linear_1.update_parameters(src.linear_1)
    dst.linear_2.update_parameters(src.linear_2)


def update_crossattn_downblock_2d_weight(src, dst):
    for index, value in enumerate(src.resnets):
        update_resnet_block_weight(value, dst.resnets[index])

    for index, value in enumerate(src.attentions):
        update_transformer_2d_model_weight(dst.attentions[index], value)

    if src.downsamplers:
        for index, value in enumerate(src.downsamplers):
            dst.downsamplers[index].conv.update_parameters(value.conv)


def update_transformer_2d_model_weight(gm, m):
    gm.norm.update_parameters(m.norm)
    gm.proj_in.update_parameters(m.proj_in)
    for i in range(len(gm.transformer_blocks)):
        gm.transformer_blocks[i].attn1.to_qkv.weight.value = np.concatenate(
            (m.transformer_blocks[i].attn1.to_q.weight.detach().cpu().numpy(),
             m.transformer_blocks[i].attn1.to_k.weight.detach().cpu().numpy(),
             m.transformer_blocks[i].attn1.to_v.weight.detach().cpu().numpy()))
        gm.transformer_blocks[i].attn1.to_out.update_parameters(
            m.transformer_blocks[i].attn1.to_out[0])

        gm.transformer_blocks[i].attn2.to_q.update_parameters(
            m.transformer_blocks[i].attn2.to_q)
        gm.transformer_blocks[i].attn2.to_kv.weight.value = np.concatenate(
            (m.transformer_blocks[i].attn2.to_k.weight.detach().cpu().numpy(),
             m.transformer_blocks[i].attn2.to_v.weight.detach().cpu().numpy()))
        gm.transformer_blocks[i].attn2.to_out.update_parameters(
            m.transformer_blocks[i].attn2.to_out[0])

        gm.transformer_blocks[i].ff.proj_in.update_parameters(
            m.transformer_blocks[i].ff.net[0].proj)
        gm.transformer_blocks[i].ff.proj_out.update_parameters(
            m.transformer_blocks[i].ff.net[2])

        gm.transformer_blocks[i].norm1.update_parameters(
            m.transformer_blocks[i].norm1)
        gm.transformer_blocks[i].norm2.update_parameters(
            m.transformer_blocks[i].norm2)
        gm.transformer_blocks[i].norm3.update_parameters(
            m.transformer_blocks[i].norm3)

    gm.proj_out.update_parameters(m.proj_out)


def update_upblock_2d_weight(src, dst):
    if src.upsamplers:
        for index, value in enumerate(src.upsamplers):
            dst.upsamplers[index].conv.update_parameters(value.conv)

    for index, value in enumerate(src.resnets):
        dst.resnets[index].norm1.update_parameters(value.norm1)
        dst.resnets[index].conv1.update_parameters(value.conv1)

        dst.resnets[index].norm2.update_parameters(value.norm2)
        dst.resnets[index].conv2.update_parameters(value.conv2)

        if value.conv_shortcut:
            dst.resnets[index].conv_shortcut.update_parameters(
                value.conv_shortcut)

        dst.resnets[index].time_emb_proj.update_parameters(value.time_emb_proj)


def update_downblock_2d_weight(src, dst):
    if src.downsamplers:
        for index, value in enumerate(src.downsamplers):
            dst.downsamplers[index].conv.update_parameters(value.conv)

    for index, value in enumerate(src.resnets):
        dst.resnets[index].norm1.update_parameters(value.norm1)
        dst.resnets[index].conv1.update_parameters(value.conv1)

        dst.resnets[index].norm2.update_parameters(value.norm2)
        dst.resnets[index].conv2.update_parameters(value.conv2)

        if value.conv_shortcut:
            dst.resnets[index].conv_shortcut.update_parameters(
                value.conv_shortcut)

        dst.resnets[index].time_emb_proj.update_parameters(value.time_emb_proj)


def update_unet_mid_block_2d_weight(src, dst):
    for index, value in enumerate(src.resnets):
        update_resnet_block_weight(value, dst.resnets[index])

    for index, value in enumerate(src.attentions):
        update_transformer_2d_model_weight(dst.attentions[index], value)


def update_crossattn_upblock_2d_weight(src, dst):
    for index, value in enumerate(src.resnets):
        update_resnet_block_weight(value, dst.resnets[index])

    for index, value in enumerate(src.attentions):
        update_transformer_2d_model_weight(dst.attentions[index], value)
    if src.upsamplers:
        for index, value in enumerate(src.upsamplers):
            dst.upsamplers[index].conv.update_parameters(value.conv)


def update_resnet_block_weight(src, dst):
    dst.norm1.update_parameters(src.norm1)
    dst.conv1.update_parameters(src.conv1)

    dst.norm2.update_parameters(src.norm2)
    dst.conv2.update_parameters(src.conv2)

    dst.time_emb_proj.update_parameters(src.time_emb_proj)
    if src.conv_shortcut:
        dst.conv_shortcut.update_parameters(src.conv_shortcut)


def update_unetmidblock_2d_weight(src, dst):
    for index, value in enumerate(src.attentions):
        dst.attentions[index].group_norm.update_parameters(value.group_norm)
        dst.attentions[index].proj_attn.update_parameters(value.proj_attn)

        dst.attentions[index].qkv.weight.value = np.concatenate(
            (value.query.weight.detach().cpu().numpy(),
             value.key.weight.detach().cpu().numpy(),
             value.value.weight.detach().cpu().numpy()))
        dst.attentions[index].qkv.bias.value = np.concatenate(
            (value.query.bias.detach().cpu().numpy(),
             value.key.bias.detach().cpu().numpy(),
             value.value.bias.detach().cpu().numpy()))

    for index, value in enumerate(src.resnets):
        update_resnet_block_weight(value, dst.resnets[index])


def update_unet_2d_condition_model_weights(src, dst):
    dst.conv_in.update_parameters(src.conv_in)

    dst.time_embedding.update_parameters(src.time_embedding)
    if src.config.addition_embed_type:
        dst.add_embedding.update_parameters(src.add_embedding)

    for index, type in enumerate(src.config.down_block_types):
        if type == 'CrossAttnDownBlock2D':
            update_crossattn_downblock_2d_weight(src.down_blocks[index],
                                                 dst.down_blocks[index])
        elif type == 'DownBlock2D':
            update_downblock_2d_weight(src.down_blocks[index],
                                       dst.down_blocks[index])

    update_unet_mid_block_2d_weight(src.mid_block, dst.mid_block)

    for index, type in enumerate(src.config.up_block_types):
        if type == 'CrossAttnUpBlock2D':
            update_crossattn_upblock_2d_weight(src.up_blocks[index],
                                               dst.up_blocks[index])
        elif type == 'UpBlock2D':
            update_upblock_2d_weight(src.up_blocks[index], dst.up_blocks[index])

    dst.conv_norm_out.update_parameters(src.conv_norm_out)

    dst.conv_out.update_parameters(src.conv_out)


def load_from_hf_unet(src, dst):
    logger.info('Loading weights from HF Unet...')
    tik = time.time()

    update_unet_2d_condition_model_weights(src, dst)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Weights loaded. Total time: {t}')

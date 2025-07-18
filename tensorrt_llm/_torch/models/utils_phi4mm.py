# coding=utf-8
# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" PyTorch Phi-4-MM model."""
import math
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig

from .vision_siglip_navit import get_siglip_vision_model
from .speech_conformer_encoder import ConformerEncoder


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "TBA"
_CONFIG_FOR_DOC = "Phi4MMConfig"

# Special token ids
_IMAGE_SPECIAL_TOKEN_ID = 200010  # '<|endoftext10|>', or we can better name it (in `tokenizer_config.json`)
_AUDIO_SPECIAL_TOKEN_ID = 200011  # '<|endoftext11|>'
_COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE = [-9999, -1]  # For backward compatibility
_COMPATIBLE_AUDIO_SPECIAL_TOKEN_ID_RANGE = [float('-inf'), -10000]  # For backward compatibility


class Phi4MMImageEmbedding(nn.Module):
    """Image embedding."""

    def __init__(self, config: PretrainedConfig, **kwargs) -> None:
        super().__init__()

        # n_embed or hidden_size
        hidden_size = config.n_embd if hasattr(config, 'n_embd') else config.hidden_size
        if hasattr(config, 'embd_pdrop') or hasattr(config, 'embed_pdrop'):
            embd_drop = config.embd_pdrop if hasattr(config, 'embd_pdrop') else config.embed_pdrop
            self.drop = nn.Dropout(embd_drop)
        else:
            self.drop = None

        logger.info(f"create image tower {config.img_processor}")
        enable_gradient_checkpointing = kwargs.get('enable_gradient_checkpointing', False)

        # Load SigLIP model
        self.img_processor = get_siglip_vision_model(
            _flash_attn_2_enabled=config._attn_implementation == 'flash_attention_2'
        )

        pe_weight = self.img_processor.embeddings.position_embedding.weight
        L, D = pe_weight.size()
        H = int(math.sqrt(L))
        assert H**2 == L
        if H % 2 != 0: #and kwargs.get('image_token_compression_cls', None) is None:
            self.img_processor_padding = nn.ReflectionPad2d((0, 1, 0, 1))
            H += 1
        image_dim_out = D
        # ((448/14)//2)**2
        self.num_img_tokens = (H//2)**2
        self.base_feat_height_target = H

        if enable_gradient_checkpointing:
            self.img_processor.encoder.gradient_checkpointing = True

        self.image_dim_out = image_dim_out
        self.img_sizes = None
        self.image_attention_mask = None

        # global_gn and sub_gn for hd transform, serves as line separator
        self.use_hd_transform = kwargs.get('use_hd_transform', False)
        self.with_learnable_separator = kwargs.get('with_learnable_separator', False)
        self.hd_transform_order = kwargs.get('hd_transform_order', 'glb_sub')
        self.freeze_img_processor = kwargs.get('freeze_img_processor', False)
        self.crop_size = kwargs.get('crop_size', 336)
        logger.info(f'freeze_img_processor = {self.freeze_img_processor}')

        # image token compression
        self.image_token_compression_cls = kwargs.get('image_token_compression_cls', None)
        if self.image_token_compression_cls == 'avg_pool_2d':
            self.image_token_compression = nn.AvgPool2d(kernel_size=2, stride=2)
            self.base_feat_height_reduction = 1
            self.base_feat_height_target = self.base_feat_height_target // 2
        elif self.image_token_compression_cls is None:
            self.image_token_compression = None
            self.base_feat_height_reduction = 2
        else:
            raise NotImplementedError(f'image_token_compression_cls = {self.image_token_compression_cls}, not implemented')

        # with_hd_transform and with_learnable_separator should have same value
        assert self.use_hd_transform == self.with_learnable_separator, 'use_hd_transform and with_learnable_separator should have same value'
        if self.with_learnable_separator:
            assert self.use_hd_transform, 'learnable separator is only for hd transform'
            # 1024 * 4, merge spatial to channel dimension
            self.glb_GN = nn.Parameter(torch.zeros([1, 1, self.image_dim_out * self.base_feat_height_reduction**2]))
            self.sub_GN = nn.Parameter(torch.zeros([1, 1, 1, self.image_dim_out * self.base_feat_height_reduction**2]))
            logger.info(f'learnable separator enabled for hd transform, hd_transform_order = {self.hd_transform_order}')

        projection_cls = kwargs.get('projection_cls', 'linear')
        if projection_cls == 'linear':
            self.img_projection = nn.Linear(image_dim_out, hidden_size)
        elif projection_cls == 'mlp' and self.use_hd_transform:
            dim_projection = hidden_size
            depth = 2
            layers = [nn.Linear(image_dim_out * self.base_feat_height_reduction**2, dim_projection)]
            for _ in range(1, depth):
                layers.extend([nn.GELU(),
                                nn.Linear(dim_projection, dim_projection)])
            self.img_projection = nn.Sequential(*layers)
        elif projection_cls == 'mlp':
            # follow llava-v1.5's implementation
            # (do not use image_projection and image_proj_norm)
            dim_projection = hidden_size
            depth = 2
            layers = [nn.Linear(image_dim_out, dim_projection)]
            for _ in range(1, depth):
                layers.extend([nn.GELU(),
                                nn.Linear(dim_projection, dim_projection)])
            self.img_projection = nn.Sequential(*layers)
        else:
            raise NotImplementedError(f'projection_cls = {projection_cls}, not implemented')

        self.vocab_size = config.vocab_size
        self.img_features = None

        if isinstance(config.img_processor, dict):
            self.layer_idx = config.img_processor.get('layer_idx', -2)
            self.type_feature = config.img_processor.get('type_feature', 'patch')
        else:
            self.layer_idx = -2
            self.type_feature = 'patch'

    def set_img_features(self, img_features: torch.FloatTensor) -> None:
        self.img_features = img_features

    def set_img_sizes(self, img_sizes: torch.LongTensor) -> None:
        self.img_sizes = img_sizes

    def set_img_attn_mask(self, image_attention_mask: torch.FloatTensor) -> None:
        self.image_attention_mask = image_attention_mask

    def get_img_features(self, img_embeds: torch.FloatTensor, attention_mask=None) -> torch.FloatTensor:
        LAYER_IDX = self.layer_idx
        TYPE_FEATURE = self.type_feature

        if self.freeze_img_processor:
            with torch.no_grad():
                if attention_mask is not None:
                    img_processor_output = self.img_processor(img_embeds, output_hidden_states=True, patch_attention_mask=attention_mask)
                else:
                    img_processor_output = self.img_processor(img_embeds, output_hidden_states=True)
                img_feature = img_processor_output.hidden_states[LAYER_IDX]
        else:
            if attention_mask is not None:
                img_processor_output = self.img_processor(img_embeds, output_hidden_states=True, patch_attention_mask=attention_mask)
            else:
                img_processor_output = self.img_processor(img_embeds, output_hidden_states=True)
            img_feature = img_processor_output.hidden_states[LAYER_IDX]

        if TYPE_FEATURE == "patch":
            patch_feature = img_feature
            if self.image_token_compression is not None:
                # reshape to 2D tensor
                width = int(math.sqrt(patch_feature.size(1)))
                patch_feature = patch_feature.view(-1, width, width, patch_feature.size(-1))
                # convert to NCHW
                patch_feature = patch_feature.permute(0, 3, 1, 2)
                if getattr(self, 'img_processor_padding', None) is not None:
                    patch_feature = self.img_processor_padding(patch_feature)
                patch_feature = self.image_token_compression(patch_feature)
                # convert to NHWC
                patch_feature = patch_feature.permute(0, 2, 3, 1)
                patch_feature = patch_feature.view(-1, patch_feature.size(1) * patch_feature.size(2), patch_feature.size(-1))
            elif getattr(self, 'img_processor_padding', None) is not None:
                width = int(math.sqrt(patch_feature.size(1)))
                patch_feature = patch_feature.view(-1, width, width, patch_feature.size(-1))
                # convert to NCHW
                patch_feature = patch_feature.permute(0, 3, 1, 2)
                patch_feature = self.img_processor_padding(patch_feature)
                # convert to NHWC
                patch_feature = patch_feature.permute(0, 2, 3, 1)
                patch_feature = patch_feature.view(-1, patch_feature.size(1) * patch_feature.size(2), patch_feature.size(-1))
            return patch_feature

        if TYPE_FEATURE == "cls_patch":
            if self.image_token_compression is not None:
                # reshape to 2D tensor
                patch_feature = img_feature[:, 1:]
                cls_feature = img_feature[:, 0]
                width = math.sqrt(patch_feature.size(1))
                patch_feature = patch_feature.view(-1, width, width, patch_feature.size(-1))
                patch_feature = self.image_token_compression(patch_feature)
                patch_feature = patch_feature.view(-1, patch_feature.size(-2) * patch_feature.size(-1))
                img_feature = torch.cat([cls_feature, patch_feature], dim=1)
            return img_feature

        logger.info(f'processed img feature size = {img_feature.size()}')
        raise NotImplementedError

    def spatiotemporal_pool(self, x, num_img_tokens, batch_size=1, T=1):

        if self.image_pos_embed is not None:
            x = x.view(batch_size * T, -1, x.shape[-1])
            num_tokens = x.shape[-2]
            h, w = int(num_tokens ** 0.5), int(num_tokens ** 0.5)
            assert h * w == num_tokens, 'only support square feature maps for now'
            x = x.view(batch_size * T, h, w, x.shape[-1])
            pos_embed = self.image_pos_embed(x)
            x = x + pos_embed
            x = x.view(batch_size, T * h * w, x.shape[-1])

        if self.visual_temporal_embed is not None:
            visual_temporal_embed = self.visual_temporal_embed(x.view(batch_size, T, -1, x.shape[-1])[:, :, 0])
            x = x.view(batch_size, T, -1, x.shape[-1]) + visual_temporal_embed.view(1, T, 1, x.shape[-1])

        new_x = []
        # [bsz, T * H' * W', C] -> [bsz, T, C]
        spatial_avg_pool_x = x.view(batch_size, T, -1, x.shape[-1]).mean(dim=2)
        new_x.append(spatial_avg_pool_x)

        # [bsz, T * H' * W', C] -> [bsz, H'*W', C]
        temporal_avg_pool_x = x.view(batch_size, T, -1, x.shape[-1]).mean(dim=1)
        new_x.append(temporal_avg_pool_x)

        x = torch.cat(new_x, dim=1).view(-1, self.image_dim_out)
        num_img_tokens += T
        return x, num_img_tokens

    def forward(self, input_ids: torch.LongTensor, input_embeds: torch.FloatTensor, image_sizes=None, **kwargs) -> torch.FloatTensor:

        if isinstance(input_ids, tuple):
            # # pipeline parallel
            input_ids, input_embeds = input_ids

        img_embeds = input_embeds
        if image_sizes is None and 'image_sizes' in kwargs:
            image_sizes = kwargs['image_sizes']
        img_sizes = image_sizes

        if self.img_features is not None:
            img_embeds = self.img_features.clone()
            self.img_features = None

        if self.img_sizes is not None:
            img_sizes = self.img_sizes

        dtype = self.img_processor.embeddings.patch_embedding.weight.dtype
        if img_embeds is not None:
            # convert to bf16
            img_embeds = img_embeds.to(dtype)

        if self.image_attention_mask is not None:
            image_attention_mask = self.image_attention_mask.clone()
            self.image_attention_mask = None
        elif 'image_attention_mask' in kwargs:
            image_attention_mask = kwargs['image_attention_mask']
        else:
            image_attention_mask = None
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        with torch.no_grad():
            positions = torch.nonzero(input_ids == _IMAGE_SPECIAL_TOKEN_ID, as_tuple=False)
            positions_tuple = torch.nonzero(input_ids == _IMAGE_SPECIAL_TOKEN_ID, as_tuple=True)

        # logger.info(f'position size: {positions.size()} ...')
        fake_image_forward = False
        select = False
        hd_transform = False

        if isinstance(self.img_projection, nn.Sequential):
            target_device = self.img_projection[0].bias.device
            target_dtype = self.img_projection[0].bias.dtype
        else:  # It's a single nn.Linear layer
            target_device = self.img_projection.bias.device
            target_dtype = self.img_projection.bias.dtype

        num_img_tokens = self.num_img_tokens
        if len(positions.tolist()) > 0:
            if self.use_hd_transform and img_sizes is not None and len(img_sizes):
                hd_transform = True
                assert img_embeds.ndim == 5, f'(branch 1) img_embeds size: {img_embeds.size()}, expect 5D tensor for hd transform'
                # img_embeds: (num_images, max_num_crops, 3, H, W)
                # img_sizes: (num_images, 2).view(1, -1)

                bs = img_embeds.shape[0]
                # Nx(HW)xC
                if image_attention_mask is not None and len(image_attention_mask) > 0:
                    img_features = self.get_img_features(img_embeds.flatten(0, 1), attention_mask=image_attention_mask.type(torch.BoolTensor).flatten(0,1).to(target_device))
                else:
                    img_features = self.get_img_features(img_embeds.flatten(0, 1))

                base_feat_height_target = self.base_feat_height_target
                base_resolution = self.crop_size
                base_feat_height_reduction = self.base_feat_height_reduction

                base_feat_height = base_feat_width = int(np.sqrt(img_features.shape[1]))

                assert base_feat_height == base_feat_height_target and base_feat_width == base_feat_height_target, f'base_feat_height: {base_feat_height}, base_feat_width: {base_feat_width}, expect {base_feat_height_target} features for hd transform'

                # bs x max_num_crops x (24x24) x C
                img_features = img_features.view(bs, -1, base_feat_height * base_feat_width, self.image_dim_out)
                C = self.image_dim_out
                H = base_feat_height

                output_imgs = []
                output_len = []
                # training is tensor, inference is list
                if isinstance(img_sizes, torch.Tensor):
                    img_sizes = img_sizes.view(-1, 2)
                for _bs in range(bs):
                    h, w = img_sizes[_bs]
                    h = h // base_resolution
                    w = w // base_resolution
                    B_ = h * w

                    # 1 x (24x24) x 1024
                    global_img_feature = img_features[_bs, :1]

                    # 1 x 12 x 12 x 4096
                    glb_img = global_img_feature.reshape(1,H,H,C).reshape(1,H//base_feat_height_reduction,base_feat_height_reduction,H//base_feat_height_reduction,base_feat_height_reduction,C).contiguous().permute(0,1,3,2,4,5).reshape(1,H//base_feat_height_reduction,H//base_feat_height_reduction,base_feat_height_reduction*base_feat_height_reduction*C).contiguous()
                    temp_glb_GN = self.sub_GN.repeat(1, H//base_feat_height_reduction, 1, 1)

                    # 1 x 156 x 4096
                    glb_img = torch.cat([glb_img, temp_glb_GN], dim=2).reshape(1,-1,base_feat_height_reduction*base_feat_height_reduction*C)

                    # (max_num_crops-1) x (12x12) x C
                    sub_img = img_features[_bs, 1:]
                    # 16x574x1024
                    # get rid of padding sub_img
                    sub_img = sub_img[:B_]

                    # (num_crops, 12, 2, 12, 2, 1024) -> (num_crops, 12, 12, 2, 2, 1024) -> (num_crops, 12*12, 4*1024)
                    sub_img = sub_img.reshape(B_,H,H,C).reshape(B_,H//base_feat_height_reduction,base_feat_height_reduction,H//base_feat_height_reduction,base_feat_height_reduction,C).contiguous().permute(0,1,3,2,4,5).reshape(B_,-1,base_feat_height_reduction*base_feat_height_reduction*C).contiguous()
                    sub_img = sub_img.reshape(1, h, w, base_feat_height // base_feat_height_reduction, base_feat_width // base_feat_height_reduction, -1).permute(0,1,3,2,4,5).reshape(1,h*base_feat_height//base_feat_height_reduction,w*base_feat_width//base_feat_height_reduction,base_feat_height_reduction*base_feat_height_reduction*C)

                    if image_attention_mask is not None and len(image_attention_mask) > 0:
                        reshaped_image_attention_mask = image_attention_mask[_bs,1:B_+1,0::2,0::2].reshape(1, h, w, base_feat_height // base_feat_height_reduction, base_feat_width // base_feat_height_reduction).permute(0,1,3,2,4).reshape(1,h*base_feat_height//base_feat_height_reduction,w*base_feat_width//base_feat_height_reduction)
                        useful_height = int(reshaped_image_attention_mask[0,:,0].sum().item())
                        useful_width = int(reshaped_image_attention_mask[0,0,:].sum().item())
                        sub_img = sub_img[:,:useful_height, :useful_width]
                        temp_sub_GN = self.sub_GN.repeat(1, useful_height, 1, 1)
                        temp_len = int(image_attention_mask[_bs,:B_+1,0::2,0::2].sum().item()) + (useful_height+1) + base_feat_height//base_feat_height_reduction
                    else:
                        temp_sub_GN = self.sub_GN.repeat(1, h*base_feat_height//base_feat_height_reduction, 1, 1)
                        temp_len = int((h*w+1)*self.num_img_tokens+ 1 + (h+1)*base_feat_height//base_feat_height_reduction)

                    sub_img = torch.cat([sub_img, temp_sub_GN], dim=2).reshape(1,-1,base_feat_height_reduction*base_feat_height_reduction*C)
                    # (1, num_img_tokens, 1024*4)

                    # glb + sub
                    if self.hd_transform_order == 'glb_sub':
                        output_imgs.append(torch.cat([glb_img, self.glb_GN, sub_img], dim=1))
                    elif self.hd_transform_order == 'sub_glb':
                        output_imgs.append(torch.cat([sub_img, self.glb_GN, glb_img], dim=1))
                    else:
                        raise NotImplementedError(f'hd_transform_order = {self.hd_transform_order}, not implemented')

                    #temp_len = int((h*w+1)*144 + 1 + (h+1)*12)
                    assert temp_len == output_imgs[-1].shape[1], f'temp_len: {temp_len}, output_imgs[-1].shape[1]: {output_imgs[-1].shape[1]}'
                    output_len.append(temp_len)

                num_img_tokens = output_len
                img_set_tensor = []
                for _output_img in output_imgs:
                    img_feature_proj = self.img_projection(_output_img.to(target_device).to(target_dtype))
                    img_set_tensor.append(img_feature_proj)
                #logger.info(f'img_embeds size: {img_embeds.size()}, image sizes: {img_sizes} loading time {datetime.now() - start_time}')
                #assert sum(num_img_tokens) == len(g_values), f'(branch 1) sum(num_img_tokens): {sum(num_img_tokens)}, g_values size: {len(g_values)}, g_values {g_values}'

            else:
                raise NotImplementedError
            select = True
        else:
            # # create a fake image tensor
            # # TODO: need define image size for different vision model
            if self.training:
                img_embeds = torch.zeros(1, 3, self.crop_size, self.crop_size, dtype=target_dtype, device=input_ids.device)

                tt = (
                    self.get_img_features(img_embeds)
                    .to(target_device)
                    .to(target_dtype)
                    .reshape(-1, 1024)
                )
                if self.use_hd_transform:
                    img_set_tensor = self.img_projection(tt.reshape(-1, self.image_dim_out*self.base_feat_height_reduction**2) * self.glb_GN[0] * self.sub_GN[0, 0])
                else:
                    img_set_tensor = self.img_projection(tt)  # adapted visual features.
                fake_image_forward = True

        # we use the token embedding layer from the huggingface model, this is REQUIRED to make sure we are using the loaded weights.
        hidden_states = kwargs['wte'](input_ids)

        if select:
            if hd_transform:
                # new implementation without in-place operation
                # Ref: https://huggingface.co/microsoft/Phi-3.5-vision-instruct/blob/4a0d683eba9f1d0cbfb6151705d1ee73c25a80ca/modeling_phi3_v.py#L233
                # Ref: https://pytorch.org/docs/stable/generated/torch.Tensor.index_put.html
                # Ref: https://pytorch.org/docs/stable/generated/torch.Tensor.index_put_.html#torch.Tensor.index_put_
                # img_set_tensor: a list of tensors, each tensor has shape (1, N_tokens, C)
                assert all([_img_set_tensor.shape[0] == 1 for _img_set_tensor in img_set_tensor]), 'img_set_tensor should have shape (1, N_tokens, C)'
                # Shape: (merged_N_tokens, C)
                merged_img_set_tensor = torch.cat(img_set_tensor, dim=1).squeeze(0)
                merged_img_set_tensor = merged_img_set_tensor.to(hidden_states.dtype).to(hidden_states.device)
                # Temporarily disable autocast to avoid issue on bf16 tensors
                # Ref: https://github.com/pytorch/pytorch/issues/132715
                with torch.autocast(device_type=hidden_states.device.type, enabled=False):
                    new_hidden_states = hidden_states.index_put(
                        indices=positions_tuple,
                        values=merged_img_set_tensor,
                        accumulate=False
                    )
                hidden_states = new_hidden_states
            else:
                raise NotImplementedError

        if fake_image_forward and self.training:
            hidden_states = hidden_states + (0 * img_set_tensor[0].to(hidden_states.dtype).to(hidden_states.device)).sum()

        if self.drop is not None:
            hidden_states = self.drop(hidden_states)

        return hidden_states


class Phi4MMAudioEmbedding(nn.Module):
    """Audio embedding."""

    def __init__(self, config: PretrainedConfig, **kwargs) -> None:
        super().__init__()
        self.config = config
        # n_embed or hidden_size for text LM
        hidden_size = config.n_embd if hasattr(config, 'n_embd') else config.hidden_size

        if hasattr(config, 'embd_pdrop') or hasattr(config, 'embed_pdrop'):
            embd_drop = config.embd_pdrop if hasattr(config, 'embd_pdrop') else config.embed_pdrop
            self.drop = nn.Dropout(embd_drop)
        else:
            self.drop = None

        audio_dim_out = None # Set this variable according to the actual audio processor
        logger.info(f"create audio processor {config.audio_processor}")
        self.layer_idx = -2

        if isinstance(config.audio_processor, dict) and config.audio_processor.get('name', None) == "cascades":
            encoder_config = config.audio_processor.get("config", None)
            assert encoder_config is not None
            self.encoder = ConformerEncoder(**encoder_config)

            # fake initialization, create encoder_embedding layer only so that
            # in decoding, all parameters can be loaded in from_pretrained_function
            # in training, we do post init after from_pretrained function to make sure the correct initialization
            self.encoder.post_init({})

            audio_dim_out = encoder_config["attention_dim"]
            n_mels = encoder_config["input_size"]
        else:
            raise NotImplementedError

        assert audio_dim_out is not None, "Remember to set values for audio_dim_out"
        self.audio_dim_out = audio_dim_out
        self.audio_dim_in = n_mels

        self.freeze_audio_processor = kwargs.get('freeze_audio_processor', False)
        logger.info(f'freeze_audio_processor = {self.freeze_audio_processor}')

        self.downsample_rate = kwargs.get('downsample_rate', 1)

        enable_gradient_checkpointing = kwargs.get('enable_gradient_checkpointing', False)
        if enable_gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
            logger.info(f'gradient checkpointing enabled for audio processor')

        projection_cls = kwargs.get('projection_cls', 'linear')
        if projection_cls == 'linear':
            self.audio_projection = nn.Linear(audio_dim_out, hidden_size)
        elif projection_cls == 'mlp':
            # follow llava-v1.5's implementation
            # (do not use image_projection and image_proj_norm)
            dim_projection = hidden_size
            depth = 2
            self.linear_downsample_rate = self.downsample_rate

            layers_for_speech = [nn.Linear(audio_dim_out * self.linear_downsample_rate, dim_projection)]
            for _ in range(1, depth):
                layers_for_speech.extend([nn.GELU(), nn.Linear(dim_projection, dim_projection)])
            audio_projection_for_speech = nn.Sequential(*layers_for_speech)

            layers_for_vision = [nn.Linear(audio_dim_out * self.linear_downsample_rate, dim_projection)]
            for _ in range(1, depth):
                layers_for_vision.extend([nn.GELU(), nn.Linear(dim_projection, dim_projection)])
            audio_projection_for_vision = nn.Sequential(*layers_for_vision)

            self.audio_projection = nn.ModuleDict({
                'speech': audio_projection_for_speech,
                'vision': audio_projection_for_vision
            })
        else:
            raise NotImplementedError(f'projection_cls = {projection_cls}, not implemented')

        self.vocab_size = config.vocab_size
        self.input_embeds = None
        self.audio_embed_sizes = None

    def post_init(self, audio_config):
        # execute after the from_pretrained() initialization of the phi4mm model
        if audio_config.get('name', None) == "cascades":
            init_model_config = audio_config.get("init_model", {})
            self.encoder.post_init(init_model_config)
            # remove the init model in config so it is not saved in the config.
            # This might affect the model loading in resuming training and decoding.
            if "init_model" in audio_config:
                audio_config.pop("init_model")

    def set_audio_embeds(self, input_embeds: torch.FloatTensor) -> None:
        self.input_embeds = input_embeds

    def set_audio_embed_sizes(self, audio_embed_sizes: torch.LongTensor) -> None:
        self.audio_embed_sizes = audio_embed_sizes

    def get_audio_features(self, input_embeds: torch.FloatTensor, audio_attention_mask: torch.Tensor, audio_projection_mode: str='speech'):

        if self.freeze_audio_processor:
            with torch.no_grad():
                audio_features, masks = self.encoder(input_embeds, audio_attention_mask)
        else:
            audio_features, masks = self.encoder(input_embeds, audio_attention_mask)

        if isinstance(self.audio_projection, nn.Sequential):
            audio_set_tensor = self.audio_projection(audio_features)
        elif isinstance(self.audio_projection, nn.ModuleDict):
            audio_set_tensor = self.audio_projection[audio_projection_mode](audio_features)
        else:
            raise NotImplementedError

        return audio_set_tensor

    def forward(self, input_ids: torch.LongTensor, input_embeds: torch.FloatTensor, audio_embed_sizes=None, audio_attention_mask=None, audio_projection_mode='speech', **kwargs) -> torch.FloatTensor:
        '''
        arguments:
            input_ids: input text ids (B, U)
            input_embeds: audio features (B, T, D)  B: num audios in a sequence
        '''
        if self.input_embeds is not None:
            input_embeds = self.input_embeds.clone()
        if self.audio_embed_sizes is not None:
            audio_embed_sizes = self.audio_embed_sizes.clone()

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        MAX_INPUT_ID = int(1e9)

        with torch.no_grad():
            positions = torch.nonzero(input_ids == _AUDIO_SPECIAL_TOKEN_ID, as_tuple=False)
            positions_tuple = torch.nonzero(input_ids == _AUDIO_SPECIAL_TOKEN_ID, as_tuple=True)

        if isinstance(self.audio_projection, nn.Sequential):
            target_device = self.audio_projection[0].bias.device
            target_dtype = self.audio_projection[0].bias.dtype
        elif isinstance(self.audio_projection, nn.ModuleDict):
            target_device = self.audio_projection[audio_projection_mode][0].bias.device
            target_dtype = self.audio_projection[audio_projection_mode][0].bias.dtype
        else:  # It's a single nn.Linear layer
            target_device = self.audio_projection.bias.device
            target_dtype = self.audio_projection.bias.dtype

        if input_embeds is not None:
            input_embeds = input_embeds.to(target_device).to(target_dtype)

        if len(positions.tolist()) > 0:
            audio_set_tensor = self.get_audio_features(input_embeds, audio_attention_mask, audio_projection_mode)
        else:
            # # create an audio tensor
            # To do: not sure if this is required for text only input
            if self.training:
                audio_embeds = torch.zeros(1, 500, self.audio_dim_in).to(target_device).to(target_dtype)
                audio_attention_mask = audio_embeds.new_ones(audio_embeds.size()[:2]).long()
                audio_set_tensor = self.get_audio_features(audio_embeds, audio_attention_mask, audio_projection_mode)

        hidden_states = kwargs['wte'](input_ids)

        if len(positions.tolist()) > 0:

            assert audio_embed_sizes.sum().item() == len(positions), \
                f"please ensure the encoder outputs have the same length as defined in input_ids! \n audio_embed_sizes.sum().item(): {audio_embed_sizes.sum().item()} \n len(positions): {len(positions)} \n audio_embed_sizes: {audio_embed_sizes} \n positions: {positions} \n input_ids.shape \n {input_ids.shape}"

            # new implementation without in-place operation
            # Ref: https://huggingface.co/microsoft/Phi-3.5-vision-instruct/blob/4a0d683eba9f1d0cbfb6151705d1ee73c25a80ca/modeling_phi3_v.py#L233
            # Ref: https://pytorch.org/docs/stable/generated/torch.Tensor.index_put.html
            # Ref: https://pytorch.org/docs/stable/generated/torch.Tensor.index_put_.html#torch.Tensor.index_put_
            # audio_set_tensor: shape (N_audios, N_padded_tokens, C)
            # Shape: (merged_N_tokens, C)
            merged_audio_set_tensor = torch.cat([
                audio_set_tensor[i, :audio_embed_sizes[i], :]
                for i in range(len(audio_embed_sizes))
            ], dim=0)
            merged_audio_set_tensor = merged_audio_set_tensor.to(hidden_states.dtype).to(hidden_states.device)
            # Temporarily disable autocast to avoid issue on bf16 tensors
            # Ref: https://github.com/pytorch/pytorch/issues/132715
            with torch.autocast(device_type=hidden_states.device.type, enabled=False):
                new_hidden_states = hidden_states.index_put(
                    indices=positions_tuple,
                    values=merged_audio_set_tensor,
                    accumulate=False
                )
            hidden_states = new_hidden_states
        else:
            if self.training:
                hidden_states  = hidden_states + (0 * audio_set_tensor[:,0].to(hidden_states.dtype).to(hidden_states.device)).sum()

        if self.drop is not None:
            hidden_states = self.drop(hidden_states)

        return hidden_states

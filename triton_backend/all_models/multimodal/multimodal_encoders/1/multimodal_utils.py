from typing import Optional, Tuple

import torch


class LlavaOnevisionUtils:
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_onevision/modeling_llava_onevision.py

    def __init__(self, config, newline):
        self.config = config
        self.image_newline = newline

    def postprocess_image(self, image_features, image_sizes, image_num_patches):

        import math

        from torch import nn
        from transformers.models.llava_onevision.modeling_llava_onevision import (
            get_anyres_image_grid_shape, unpad_image)
        image_features = torch.split(image_features, image_num_patches, dim=0)
        vision_aspect_ratio = self.config.vision_aspect_ratio

        # LlavaOnevisionForConditionalGeneration.pack_image_features
        new_image_features = []
        feature_lens = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size
                if height * width != base_image_feature.shape[0]:
                    raise ValueError(
                        "The number of patches is not consistent with the image size."
                    )
                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                image_feature = image_feature.view(num_patch_height,
                                                   num_patch_width, height,
                                                   width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1,
                                                      3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature,
                                            image_sizes[image_idx])
                max_num_patches = int(vision_aspect_ratio.strip("anyres_max_"))
                channels, curr_height, curr_width = image_feature.shape
                ratio = math.sqrt(curr_height * curr_width /
                                  (max_num_patches * height**2))
                if ratio > 1.1:
                    image_feature = image_feature[None]
                    image_feature = nn.functional.interpolate(
                        image_feature,
                        [int(curr_height // ratio),
                         int(curr_width // ratio)],
                        mode="bilinear")[0]
                if self.image_newline is not None:
                    image_feature = torch.cat(
                        (
                            image_feature,
                            self.image_newline[:, None, None].expand(
                                *image_feature.shape[:-1], 1).to(
                                    image_feature.device, image_feature.dtype),
                        ),
                        dim=-1,
                    )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, image_feature),
                                          dim=0)
            else:
                image_feature = image_feature[0]
                if self.image_newline is not None:
                    image_feature = torch.cat(
                        (image_feature,
                         self.image_newline[None].to(image_feature)),
                        dim=0)
            new_image_features.append(image_feature)
            feature_lens.append(image_feature.size(0))

        max_tokens = max(x.shape[0] for x in new_image_features)
        new_image_features = [
            torch.nn.functional.pad(table,
                                    (0, 0, 0, max_tokens - table.shape[0]),
                                    mode='constant')
            for table in new_image_features
        ]
        image_features = torch.stack(new_image_features, dim=0)
        return image_features

    def postprocess_video(self, image_features, batch_size, frames):

        # LlavaOnevisionForConditionalGeneration.apply_pooling
        import math

        from torch import nn
        height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size
        batch_frames, seq_len, dim = image_features.shape
        image_features = image_features.view(batch_frames, height, width, -1)
        image_features = image_features.permute(0, 3, 1, 2).contiguous()

        height, width = image_features.shape[2:]
        scaled_shape = [math.ceil(height / 2), math.ceil(width / 2)]
        image_features = nn.functional.interpolate(image_features,
                                                   size=scaled_shape,
                                                   mode="bilinear")

        image_features = image_features.permute(0, 2, 3, 1)
        image_features = image_features.view(batch_frames, -1, dim)

        video_features = image_features.reshape(
            batch_size, frames * image_features.shape[1], -1)
        image_newline = self.image_newline[None, None, :].repeat(
            batch_size, 1, 1).to(video_features.device)
        video_features = torch.cat((video_features, image_newline), dim=1)
        return video_features


class Qwen2VLUtils:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py

    def __init__(self, config):
        self.config = config
        self.image_token_id = config.image_token_id
        self.video_token_id = config.video_token_id
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.vision_start_token_id = config.vision_start_token_id
        self.max_position_embeddings = config.max_position_embeddings
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.rope_theta = config.rope_theta

    def get_rope_index(
        self,
        input_ids: torch.IntTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.IntTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.IntTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.spatial_merge_size
        image_token_id = self.image_token_id
        video_token_id = self.video_token_id
        vision_start_token_id = self.vision_start_token_id
        mrope_position_deltas = []
        input_ids = input_ids.to(torch.int64).to('cuda')
        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids
            position_ids = torch.ones(3,
                                      input_ids.shape[0],
                                      input_ids.shape[1],
                                      dtype=input_ids.dtype,
                                      device=input_ids.device)
            image_index, video_index = 0, 0
            for idx, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = input_ids[attention_mask[idx] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(
                    input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                        llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) +
                        st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(
                        -1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(
                        llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(
                        llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len +
                        st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                        llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) +
                        st_idx)

                llm_positions = torch.cat(llm_pos_ids_list,
                                          dim=1).reshape(3, -1)
                position_ids[..., idx,
                             attention_mask[idx] == 1] = llm_positions.to(
                                 position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 -
                                             len(total_input_ids[idx]))
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(
                    input_ids.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                    -1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[
                    -1]
            else:
                position_ids = (torch.arange(input_ids.shape[1],
                                             device=input_ids.device).view(
                                                 1, 1, -1).expand(
                                                     3, input_ids.shape[0], -1))
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )
            return position_ids, mrope_position_deltas

    def compute_mrope(self, input_ids, image_grid_thw, attention_mask):
        pass

        from tensorrt_llm.functional import (RopeEmbeddingUtils,
                                             RotaryScalingType)
        mrope_position_ids, mrope_position_deltas = self.get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )
        mrope_position_deltas = mrope_position_deltas.to(torch.device('cpu'))
        if not hasattr(self, 'rotary_cos_sin'):
            inv_freq, rotary_cos_sin = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
                num_pos=self.max_position_embeddings,
                dim=int(self.hidden_size / self.num_attention_heads),
                theta=float(self.rope_theta),
                scale_type=RotaryScalingType.mrope)

            self.rotary_cos_sin = torch.from_numpy(rotary_cos_sin).to("cuda")
            self.rotary_cos_sin = self.rotary_cos_sin.reshape(
                self.max_position_embeddings,
                int(self.hidden_size / self.num_attention_heads / 2), 2)
            self.cos_ori = self.rotary_cos_sin[:, :, 0]
            self.sin_ori = self.rotary_cos_sin[:, :, 1]

        mrope_position_ids = mrope_position_ids.transpose(1, 0)
        mrope_position_ids_padding = torch.zeros(
            mrope_position_ids.shape[:-1] + (self.max_position_embeddings, ),
            dtype=torch.int32,
            device="cuda")
        mrope_position_ids_padding[:, :, :mrope_position_ids.
                                   shape[-1]] = mrope_position_ids
        cos = self.cos_ori[mrope_position_ids_padding]
        sin = self.sin_ori[mrope_position_ids_padding]

        mrope_section = [16, 24, 24]
        cos = torch.cat([
            m[:, i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))
        ],
                        dim=-1).unsqueeze(-1)
        sin = torch.cat([
            m[:, i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))
        ],
                        dim=-1).unsqueeze(-1)
        concat_cos_sin = torch.concatenate((cos, sin), axis=-1)
        concat_cos_sin = concat_cos_sin.reshape(concat_cos_sin.shape[0], -1)
        return concat_cos_sin, mrope_position_deltas

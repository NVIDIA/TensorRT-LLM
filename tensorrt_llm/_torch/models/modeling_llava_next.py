import copy
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoConfig, AutoModel, AutoProcessor, AutoTokenizer,
                          LlavaNextConfig, PretrainedConfig, PreTrainedModel)
from transformers.modeling_utils import load_sharded_checkpoint
from transformers.models.llava_next.modeling_llava_next import (
    LlavaNextMultiModalProjector, get_anyres_image_grid_shape,
    image_size_to_num_patches, unpad_image)

from tensorrt_llm.inputs.multimodal import MultimodalParams

from ...inputs import (ExtraProcessedInputs, InputProcessor,
                       MultimodalPlaceholderMetadata,
                       MultimodalPlaceholderPlacement, TextPrompt,
                       register_input_processor)
from ...llmapi.utils import download_hf_model
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from .modeling_auto import AutoModelForCausalLM
from .modeling_clip import CLIPVisionModel
from .modeling_multimodal_utils import fuse_input_embeds
from .modeling_utils import ModelConfig, filter_weights, register_auto_model

DISAGG = os.getenv('TLLM_MULTIMODAL_DISAGGREGATED', '0') == '1'


class LlavaNextInputProcessor(InputProcessor):

    def __init__(self,
                 model_path: str,
                 model_config: PretrainedConfig,
                 tokenizer: AutoTokenizer,
                 trust_remote_code: bool = True):
        self.tokenizer = tokenizer
        self.use_fast = True
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                use_fast=self.use_fast)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            use_fast=self.use_fast)
        self.model_config = model_config

        self.image_token_index = model_config.image_token_index
        self.vocab_size = model_config.vocab_size

    @torch.inference_mode()
    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        text_prompt, mm_data = inputs.get("prompt"), inputs.get(
            "multi_modal_data", {})
        # Preprocess
        images = mm_data.get('image', [])
        if not images:
            return self.processor.tokenizer(
                text_prompt,
                return_tensors="pt").input_ids[0].to(torch.int32).tolist(), {}

        processed_values = self.processor(
            text=text_prompt,
            images=images,
            do_rescale=not (images and isinstance(images[0], torch.Tensor)),
            return_tensors="pt")
        # Postprocess
        fused_input_ids = processed_values['input_ids'][0]
        fused_input_ids[fused_input_ids ==
                        self.image_token_index] = self.vocab_size + 1

        multimodal_data = {}
        multimodal_data["image"] = {
            "pixel_values": processed_values['pixel_values'],
            "image_sizes": processed_values['image_sizes'],
        }
        return fused_input_ids.to(torch.int32).tolist(), {
            "multimodal_data": multimodal_data
        }


class LlavaNextVisionModel(nn.Module):

    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args,
                 **kwargs) -> None:
        super().__init__()
        self.pretrained_config = model_config.pretrained_config
        self.device = f"cuda:{model_config.mapping.rank}"
        model_path = self.pretrained_config._name_or_path

        # Determine the actual local path for model files
        if os.path.isdir(model_path):
            local_model_path = model_path
        else:
            local_model_path = download_hf_model(model_path)

        # Partially load the model to reduce memory usage(Vision tower and multi-modal projector)
        hf_model_config = AutoConfig.from_pretrained(local_model_path)
        self.dtype = hf_model_config.text_config.torch_dtype
        module_dict = nn.ModuleDict({
            "vision_tower":
            AutoModel.from_config(hf_model_config.vision_config),
            "multi_modal_projector":
            LlavaNextMultiModalProjector(hf_model_config)
        })
        module_dict.register_parameter(
            "image_newline",
            nn.Parameter(torch.empty(hf_model_config.text_config.hidden_size)))

        missing_keys, _ = load_sharded_checkpoint(module_dict,
                                                  local_model_path,
                                                  strict=False)
        assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"
        hf_vision_tower = module_dict["vision_tower"].to(self.dtype)
        hf_mm_projector = module_dict["multi_modal_projector"].to(
            self.dtype).to(self.device)
        hf_image_newline = module_dict.image_newline.to(self.dtype).to(
            self.device)

        # For A100 GPU, fallback to HF vision tower due to accuracy issue in TRT-LLM CLIPAttention
        # Otherwise, use TRTLLM vision tower(CLIPVisionModel)
        prop = torch.cuda.get_device_properties(0)
        sm_version = prop.major * 10 + prop.minor
        self.use_hf_vision_tower = sm_version == 80
        if self.use_hf_vision_tower:
            self.vision_tower = hf_vision_tower.to(self.device)
        else:
            vision_model_config = ModelConfig(
                pretrained_config=model_config.pretrained_config.vision_config,
                attn_backend="TRTLLM")
            self.vision_tower = CLIPVisionModel(vision_model_config).to(
                self.device).to(self.dtype)
            self.vision_tower.load_weights(hf_vision_tower.state_dict())

        # Use HF multi-modal projector
        self.mm_projector = hf_mm_projector
        self.image_newline = hf_image_newline
        self.vision_feature_select_strategy = getattr(
            model_config.pretrained_config, "vision_feature_select_strategy",
            "default")

    # Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_next/modeling_llava_next.py#L284
    def pack_image_features(self,
                            image_features,
                            image_sizes,
                            vision_feature_select_strategy,
                            image_newline=None):
        new_image_features = []
        feature_lens = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = self.pretrained_config.vision_config.image_size // self.pretrained_config.vision_config.patch_size

                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.pretrained_config.image_grid_pinpoints,
                    self.pretrained_config.vision_config.image_size,
                )

                if (np.prod(image_feature.shape) %
                    (num_patch_height * num_patch_width * height * width) != 0
                        and vision_feature_select_strategy == "default"):
                    logger.warning_once(
                        "Image feature shape does not line up with the provided patch size. "
                        "You may be using the `default` vision_feature_select_strategy with a"
                        " visual encoder that does not have CLS.")

                image_feature = image_feature.view(num_patch_height,
                                                   num_patch_width, height,
                                                   width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1,
                                                      3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature,
                                            image_sizes[image_idx])
                if image_newline is not None:
                    image_feature = torch.cat(
                        (
                            image_feature,
                            image_newline[:, None, None].expand(
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
                if image_newline is not None:
                    image_feature = torch.cat(
                        (image_feature, image_newline[None].to(image_feature)),
                        dim=0)
            new_image_features.append(image_feature)
            feature_lens.append(image_feature.size(0))
        feature_lens = torch.tensor(feature_lens,
                                    dtype=torch.long,
                                    device=image_features[0].device)
        return new_image_features, feature_lens

    def _pad_for_batching(
        self,
        pixel_values: list[torch.Tensor],
    ):
        """
        Pads images on the `num_of_patches` dimension with zeros to form a batch of same number of patches.
        Optimized for CUDA tensors using torch.nn.functional.pad.

        Args:
            pixel_values (`list[torch.Tensor]`):
                A list of pixel value tensors, each of shape (`batch`, `num_patches`, `height`, `width`, `channels`)

        Returns:
            list[`torch.Tensor`]: The padded image tensors.
        """
        if not pixel_values:
            return pixel_values

        # Find max patches across all images
        max_patches = max(tensor.shape[1] for tensor in pixel_values)

        # Pad each tensor efficiently using torch.nn.functional.pad
        padded_values = []

        for pixel_value in pixel_values:
            current_patches = pixel_value.shape[1]
            if current_patches < max_patches:
                # Pad the second dimension (num_patches at index 1)
                # torch.nn.functional.pad expects padding in reverse order of dimensions
                # For shape (batch, num_patches, height, width, channels), we pad the second dim:
                # padding format: (channels_before, channels_after, width_before, width_after, height_before, height_after, patches_before, patches_after, batch_before, batch_after)
                padding = (0, 0, 0, 0, 0, 0, 0, max_patches - current_patches,
                           0, 0)
                padded_pixel_value = F.pad(pixel_value,
                                           padding,
                                           mode='constant',
                                           value=0.0)
            else:
                padded_pixel_value = pixel_value

            padded_values.append(padded_pixel_value)

        return padded_values

    @torch.inference_mode()
    def forward(self, multimodal_params: List[MultimodalParams]):
        pixel_values = [
            multimodal_param.multimodal_data["image"]["pixel_values"]
            for multimodal_param in multimodal_params
        ]
        image_sizes = [
            multimodal_param.multimodal_data["image"]["image_sizes"]
            for multimodal_param in multimodal_params
        ]
        pixel_values = self._pad_for_batching(pixel_values)

        pixel_values = torch.cat(pixel_values, dim=0)
        image_sizes = torch.cat(image_sizes, dim=0)

        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=self.pretrained_config.image_grid_pinpoints,
                patch_size=self.pretrained_config.vision_config.image_size,
            ) for imsize in image_sizes
        ]

        if pixel_values.dim() == 5:
            # stacked if input is (batch_size, num_patches, num_channels, height, width)
            _pixel_values_list = [
                pix_val[:num_patch]
                for pix_val, num_patch in zip(pixel_values, image_num_patches)
            ]
            pixel_values = torch.cat(_pixel_values_list, dim=0)

        if self.use_hf_vision_tower:
            image_features = self.vision_tower(
                pixel_values, output_hidden_states=True).hidden_states
        else:
            attn_metadata = self.vision_tower.prepare_attn_metadata(
                pixel_values.shape[0])
            image_features = self.vision_tower(
                pixel_values,
                attn_metadata=attn_metadata,
            )
        selected_image_feature = image_features[-2][:, 1:]
        image_features = self.mm_projector(selected_image_feature)

        image_features = torch.split(image_features, image_num_patches, dim=0)

        # NOTE: 'pack_image_features' is directly copied from the HF's code
        image_features, feature_lens = self.pack_image_features(
            image_features,
            image_sizes,
            vision_feature_select_strategy=self.vision_feature_select_strategy,
            image_newline=self.image_newline,
        )
        image_features = torch.cat(image_features, dim=0)
        return [image_features]


@register_auto_model("LlavaNextForConditionalGeneration")
@register_input_processor(
    LlavaNextInputProcessor,
    model_type="llava_next",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={"image": "<image>"},
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
    ))
class LlavaNextModel(PreTrainedModel):
    config_class = LlavaNextConfig

    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args,
                 **kwargs) -> None:
        config = model_config.pretrained_config
        super().__init__(config)
        if hasattr(self, "llm"):
            return
        if not DISAGG:
            self.mm_encoder = LlavaNextVisionModel(model_config)

        llm_model_config = copy.deepcopy(model_config)
        llm_model_config.pretrained_config = model_config.pretrained_config.text_config

        # TODO Remove these when MistralConfig is natively supported
        llm_model_config.pretrained_config.attention_bias = False
        llm_model_config.pretrained_config.rope_scaling = None
        llm_model_config.pretrained_config.mlp_bias = False

        self.llm = AutoModelForCausalLM.from_config(llm_model_config)

        self.model_config = model_config
        self.vocab_size = config.vocab_size
        self.model_dtype = getattr(config.text_config, "torch_dtype",
                                   torch.float16)
        logger.info(f"{self.dtype=} {self.model_dtype=}")

        self.post_config()
        self.is_loaded = True

    def load_weights(self, weights):
        weights = filter_weights("language_model", weights)
        self.llm.load_weights(weights)

    def post_config(self):
        self.config = self.llm.config
        self.model_config.pretrained_config = self.llm.config

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        num_context_requests, num_generation_requests = attn_metadata.num_contexts, attn_metadata.num_generations
        logger.debug(f"{num_context_requests=}, {num_generation_requests=}")

        multimodal_params = kwargs.get("multimodal_params", [])
        mm_embeds = []
        if len(multimodal_params) > 0:
            if not DISAGG:
                mm_embeds = self.mm_encoder.forward(multimodal_params)
            else:
                mm_embeds = [
                    multimodal_param.multimodal_data["multimodal_embedding"]
                    for multimodal_param in multimodal_params
                ]
        input_ids, inputs_embeds = fuse_input_embeds(
            self.llm.model.embed_tokens, input_ids, mm_embeds)

        logits = self.llm.forward(attn_metadata, input_ids, position_ids,
                                  inputs_embeds, return_context_logits)
        return logits

import copy
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import transformers
from PIL import Image

from tensorrt_llm._torch.models import modeling_utils
from tensorrt_llm._torch.models.checkpoints import NemotronHHfWeightMapper
from tensorrt_llm.inputs.multimodal import MultimodalParams

from ...inputs import (ExtraProcessedInputs, InputProcessor,
                       MultimodalPlaceholderMetadata,
                       MultimodalPlaceholderPlacement, TextPrompt,
                       register_input_processor)
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from .modeling_auto import AutoModelForCausalLM
from .modeling_multimodal_utils import fuse_input_embeds
from .modeling_radio import RADIOVisionModel
from .modeling_utils import register_auto_model


# Make this a runtime lookup rather than a module-wide constant for easier unit testing.
def _is_disagg() -> bool:
    return os.getenv("TLLM_MULTIMODAL_DISAGGREGATED", "0") == "1"


IMAGE_TOKEN_ID = 131072


class SquaredReLU(nn.Module):

    def forward(self, x):
        return torch.pow(torch.nn.functional.relu(x), 2)


class RMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)


class NanoV2VLVisionEncoder(transformers.PreTrainedModel,
                            transformers.generation.GenerationMixin):

    def __init__(self,
                 model_config: ModelConfig[transformers.PretrainedConfig]):
        config = model_config.pretrained_config
        super().__init__(config)
        self.image_size = config.force_image_size
        self.patch_size = config.patch_size
        # self.template = config.template
        self.num_image_token = int((self.image_size // self.patch_size)**2 *
                                   (config.downsample_ratio**2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version  # Pixel shuffle version.

        # Construct the vision projection.
        self.vit_hidden_size = config.vit_hidden_size
        self.vision_projection_hidden_size = config.projector_hidden_size
        self.llm_hidden_size = config.llm_config.hidden_size
        self.mlp1 = nn.Sequential(
            RMSNorm(self.vit_hidden_size * int(1 / self.downsample_ratio)**2,
                    eps=1e-5),
            nn.Linear(self.vit_hidden_size * int(1 / self.downsample_ratio)**2,
                      self.vision_projection_hidden_size,
                      bias=False), SquaredReLU(),
            nn.Linear(self.vision_projection_hidden_size,
                      self.llm_hidden_size,
                      bias=False))
        self.mlp1 = self.mlp1.to(config.torch_dtype)

        # Construct the vision encoder.
        self.with_hf_codes = os.getenv("WITH_HF_CODES", "0") == "1"
        if self.with_hf_codes:
            self.vision_model = transformers.AutoModel.from_config(
                config.vision_config, trust_remote_code=True)
            # set input_condition as Identity module.
            self.vision_model.radio_model.make_preprocessor_external()
            self.vision_model.to(config.torch_dtype)
        else:
            vision_model_config = copy.deepcopy(model_config)
            vision_model_config.pretrained_config = vision_model_config.pretrained_config.vision_config
            self.vision_model = RADIOVisionModel(vision_model_config)
            self.vision_model.to(config.torch_dtype)

    @torch.compile
    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            logger.warning(
                "In ps_version 'v1', the height and width have not been swapped back, "
                'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.with_hf_codes:
            vit_embeds = self.vision_model(pixel_values).features
        else:
            vit_embeds = self.vision_model(pixel_values)
        vit_embeds = vit_embeds.to(dtype=torch.bfloat16)
        # Down-sampling and projection.
        h = w = int(vit_embeds.shape[1]**0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds,
                                        scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1,
                                        vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def forward(self, multimodal_params: List[MultimodalParams]):
        mm_embedding = []

        BATCH_INFERENCE = True
        if BATCH_INFERENCE:
            # Batch data.
            batched_pixel_values = torch.cat([
                multimodal_param.multimodal_data["pixel_values"]
                for multimodal_param in multimodal_params
            ],
                                             dim=0)
            # -> [num_patches, channel, height, width]
            batched_num_patches = torch.cat([
                multimodal_param.multimodal_data["num_patches"]
                for multimodal_param in multimodal_params
            ],
                                            dim=0).tolist()
            # -> list of[num_patches1, num_patches2, ...]
            batched_image_embeds = self.extract_feature(batched_pixel_values)
            # -> [num_patches, num_image_token, hidden_size]
            mm_embedding = torch.split(batched_image_embeds,
                                       batched_num_patches,
                                       dim=0)
            mm_embedding = [
                m.reshape(-1, self.llm_hidden_size) for m in mm_embedding
            ]
            # -> list of [num_patches*num_image_token, hidden_size]
        else:
            # Inference per sample.
            for multimodal_param in multimodal_params:
                pixel_values = multimodal_param.multimodal_data["pixel_values"]
                image_embeds = self.extract_feature(pixel_values)
                # -> [num_patches, num_image_token, hidden_size]
                image_embeds = image_embeds.reshape(-1, self.llm_hidden_size)
                # -> [num_patches*num_image_token, hidden_size]
                mm_embedding.append(image_embeds)
        return mm_embedding


class NanoV2VLInputProcessor(InputProcessor):

    def __init__(self,
                 model_path: str,
                 model_config: transformers.PretrainedConfig,
                 tokenizer: transformers.AutoTokenizer,
                 trust_remote_code: bool = True):
        if not trust_remote_code:
            raise ValueError("trust_remote_code must be True for NanoV2VL")

        self.model_config = model_config
        self.image_size = model_config.force_image_size
        self.patch_size = model_config.patch_size
        self.downsample_ratio = model_config.downsample_ratio
        self.num_image_token = int((self.image_size // self.patch_size)**2 *
                                   (self.downsample_ratio**2))

        self.device = 'cpu'

        self.tokenizer = tokenizer
        self.use_fast = True
        if self.tokenizer is None:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, use_fast=self.use_fast)

        self.image_processor = transformers.AutoImageProcessor.from_pretrained(
            model_path, trust_remote_code=True, use_fast=self.use_fast)
        self.img_context_token = "<image>"
        self.img_start_token = "<img>"
        self.img_end_token = "</img>"
        self.dtype = model_config.torch_dtype

    @torch.inference_mode()
    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        text_prompt, mm_data = inputs.get("prompt"), inputs.get(
            "multi_modal_data", {})
        images = mm_data.get("image", None)

        if images is not None:
            if isinstance(images[0], torch.Tensor):
                # NanoV2VL can only support PIL images. Convert normalized tensors (0-1) to PIL images (0-255).
                images = [
                    Image.fromarray((image.permute(1, 2, 0) * 255).to(
                        torch.uint8).cpu().numpy()) for image in images
                ]

        # Processing for multimodal data.
        processed_images = self.image_processor(images=images,
                                                return_tensors='pt').to(
                                                    self.device)

        # Insert enough special tokens for image embedding.
        parts = text_prompt.split(self.img_context_token)
        if len(parts) - 1 != len(processed_images['num_patches']):
            raise ValueError(
                f"Number of {self.img_context_token} tokens ({len(parts) - 1}) doesn't match num_patches_list length ({len(processed_images['num_patches'])})"
            )
        processed_query = parts[0]
        for num_patches, part in zip(processed_images['num_patches'],
                                     parts[1:]):
            feature_size = num_patches * self.num_image_token
            image_repl = self.img_start_token + self.img_context_token * feature_size + self.img_end_token
            processed_query += image_repl + part
        input_ids = self.tokenizer.encode(processed_query,
                                          add_special_tokens=False,
                                          return_tensors="pt")

        # Will package inputs for language model forward in AGGREGATE mode.
        multimodal_data = {}
        multimodal_data['pixel_values'] = processed_images['pixel_values'].to(
            self.dtype)
        multimodal_data['num_patches'] = processed_images['num_patches']
        return input_ids[0].to(torch.int32).tolist(), {
            "multimodal_data": multimodal_data,
        }


@register_auto_model("NemotronH_Nano_VL_V2")
@register_input_processor(
    NanoV2VLInputProcessor,
    model_type="NemotronH_Nano_VL_V2",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={
            "image": "<image>\n",
        },
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
        placeholders_separator="",
    ))
class NemotronH_Nano_VL_V2(transformers.PreTrainedModel):

    _supports_flash_attn_2 = True

    def __init__(self, model_config: ModelConfig):
        if _is_disagg():
            raise ValueError(
                "NanoV2VL does not support disaggregated inference yet.")

        config = model_config.pretrained_config
        super().__init__(config)

        self.model_config = model_config
        if hasattr(self, "llm"):
            return

        if not _is_disagg():
            self.vision_encoder = NanoV2VLVisionEncoder(model_config).eval()
            self.vision_encoder.to(config.torch_dtype)

        llm_model_config = copy.deepcopy(model_config)
        llm_model_config.pretrained_config = llm_model_config.pretrained_config.llm_config
        self.llm = AutoModelForCausalLM.from_config(llm_model_config)

        self.vocab_size = llm_model_config.pretrained_config.vocab_size
        self.model_dtype = getattr(config, "torch_dtype", torch.float16)
        logger.info(f"{self.dtype=} {self.model_dtype=}")
        self.post_config()
        self.is_loaded = True

    def load_weights(self, weights):
        # TODO: move vision encoder weights loading to vision encoder class.

        # Load vision encoder weights for pytorch modules.
        filter_weights = {
            k: v
            for k, v in weights.items()
            if k.startswith('vision') or k.startswith('mlp1')
        }
        missing_keys, unexpected_keys = self.vision_encoder.load_state_dict(
            filter_weights, strict=False)
        try:
            missing_keys.remove("vision_model.radio_model.summary_idxs")
        except ValueError:
            pass

        unexpected_keys.remove(
            "vision_model.radio_model.input_conditioner.norm_mean")
        unexpected_keys.remove(
            "vision_model.radio_model.input_conditioner.norm_std")
        for m in missing_keys:
            if not m.startswith('vision_model.radio_model.model.blocks.'):
                raise ValueError(f"Missing key: {m}")
        for u in unexpected_keys:
            if not u.startswith('vision_model.radio_model.model.blocks.'):
                raise ValueError(f"Unexpected key: {u}")

        if len(unexpected_keys) > 0 or len(missing_keys) > 0:
            # Load weights for vision transformer module.
            model_weights = {
                k.replace('vision_model.radio_model.model.', ''): v
                for k, v in weights.items()
                if k.startswith('vision_model.radio_model.model.')
            }
            converted_weights = dict()
            for name in model_weights:
                # Handle with weights and bias for vision transformer's qkv projection.
                if "attn.qkv." in name:
                    q_name = name.replace("attn.qkv.", "attn.q_proj.")
                    k_name = name.replace("attn.qkv.", "attn.k_proj.")
                    v_name = name.replace("attn.qkv.", "attn.v_proj.")
                    dim_shape = model_weights[name].shape[0] // 3
                    converted_weights[q_name] = model_weights[name][:dim_shape]
                    converted_weights[k_name] = model_weights[name][
                        dim_shape:2 * dim_shape]
                    converted_weights[v_name] = model_weights[name][2 *
                                                                    dim_shape:]
                else:
                    converted_weights[name] = model_weights[name]
            pattern_mapping = {
                r'(.*?)attn.proj.(.*)': r'\1attn.o_proj.\2',
                r'(.*?)mlp.fc1.(.*)': r'\1mlp.up_proj.\2',
                r'(.*?)mlp.fc2.(.*)': r'\1mlp.down_proj.\2',
            }
            modeling_utils._load_weights_impl(
                self.vision_encoder.vision_model.radio_model.model,
                converted_weights,
                params_map=pattern_mapping)

        # Load language model weights.
        filtered_weights = {
            k.replace('language_model.', ''): v
            for k, v in weights.items() if k.startswith('language_model.')
        }
        weight_mapper = NemotronHHfWeightMapper()
        weight_mapper.init_model_and_config(self.llm, self.model_config)
        self.llm.load_weights(filtered_weights, weight_mapper=weight_mapper)

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    def post_config(self):
        # use llm.config as config for pytorch model engine
        self.config = self.llm.config
        self.model_config.pretrained_config = self.llm.config

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        VLM forward logic with inflight batching support.
        """
        num_context_requests, num_generation_requests = attn_metadata.num_contexts, attn_metadata.num_generations
        logger.debug(
            f"num_context_requests: {num_context_requests}, num_generation_requests: {num_generation_requests}"
        )

        multimodal_params = kwargs.get("multimodal_params", [])
        mm_embedding = []
        if len(multimodal_params) > 0:
            if not _is_disagg():
                mm_embedding = self.vision_encoder(multimodal_params)
            else:
                # Directly fetch the multimodal embedding for DISAGG mode.
                # This path is not functional now. `multimodal_params` will be prepared in PyExecutor.
                mm_embedding = [
                    multimodal_param.multimodal_data["multimodal_embedding"]
                    for multimodal_param in multimodal_params
                ]
        input_ids, input_embeds = fuse_input_embeds(
            self.llm.model.embed_tokens,
            input_ids,
            mm_embedding,
            mm_token_ids=torch.tensor([IMAGE_TOKEN_ID], dtype=torch.int32),
        )
        output_prob = self.llm.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
            return_context_logits=return_context_logits,
            lora_params=kwargs.get("lora_params", None),
        )

        logger.debug(f'output shape: {output_prob.shape}')
        return output_prob

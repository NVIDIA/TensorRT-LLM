import copy
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import (AutoConfig, AutoModel, AutoProcessor, LlavaNextConfig,
                          PretrainedConfig, PreTrainedModel)
from transformers.modeling_utils import load_sharded_checkpoint
from transformers.models.llava_next.modeling_llava_next import \
    LlavaNextMultiModalProjector

from ..._utils import nvtx_range
from ...inputs import (ExtraProcessedInputs, InputProcessor, TextPrompt,
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
from .modeling_utils import ModelConfig, register_auto_model
from ...executor.request import MultimodalParams

class CLIPEncoderInfo:

    def __init__(self, hf_config):
        self.vision_config = hf_config.vision_config

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        return self.get_patch_grid_length()**2 + 1

    def get_image_size(self) -> int:
        return self.vision_config.image_size

    def get_patch_size(self) -> int:
        return self.vision_config.patch_size

    def get_patch_grid_length(self) -> int:
        image_size, patch_size = self.get_image_size(), self.get_patch_size()
        assert image_size % patch_size == 0
        return image_size // patch_size

class LlavaNextInputProcessor(InputProcessor):

    def __init__(self, model_path, model_config, tokenizer):
        self.tokenizer = tokenizer
        self.processor = AutoProcessor.from_pretrained(model_path,
                                                       use_fast=True)
        self.model_config = model_config

        self.device = 'cuda'

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
        missing_keys, _ = load_sharded_checkpoint(module_dict,
                                                  local_model_path,
                                                  strict=False)
        assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"
        hf_vision_tower = module_dict["vision_tower"].to(self.dtype)
        hf_mm_projector = module_dict["multi_modal_projector"].to(
            self.dtype).to(self.device)

        # Use TRTLLM vision tower(CLIPVisionModel)
        vision_model_config = ModelConfig(
            pretrained_config=model_config.vision_config, attn_backend="TRTLLM")
        self.vision_tower = CLIPVisionModel(vision_model_config).to(
            self.device).to(self.dtype)
        self.vision_tower.load_weights(hf_vision_tower.state_dict())

        # Use HF multi-modal projector
        self.mm_projector = hf_mm_projector
        self.hf_model_config = hf_model_config

    def image_size_to_num_patches(self, image_size, grid_pinpoints = None, patch_size: int = None):
        """
        Calculate the number of patches after the preprocessing for images of any resolution.

        Args:
            image_size (`torch.LongTensor` or `np.ndarray` or `Tuple[int, int]`):
                The size of the input image in the format (height, width). ?
            grid_pinpoints (`List`):
                A list containing possible resolutions. Each item in the list should be a tuple or list
                of the form `(height, width)`.
            patch_size (`int`):
                The size of each image patch.

        Returns:
            int: the number of patches
        """
        def select_best_resolution(original_size: tuple, possible_resolutions: list) -> tuple:
            """
            Selects the best resolution from a list of possible resolutions based on the original size.

            This is done by calculating the effective and wasted resolution for each possible resolution.

            The best fit resolution is the one that maximizes the effective resolution and minimizes the wasted resolution.

            Args:
                original_size (tuple):
                    The original size of the image in the format (height, width).
                possible_resolutions (list):
                    A list of possible resolutions in the format [(height1, width1), (height2, width2), ...].

            Returns:
                tuple: The best fit resolution in the format (height, width).
            """
            original_height, original_width = original_size
            best_fit = None
            max_effective_resolution = 0
            min_wasted_resolution = float("inf")

            for height, width in possible_resolutions:
                scale = min(width / original_width, height / original_height)
                downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
                effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
                wasted_resolution = (width * height) - effective_resolution

                if effective_resolution > max_effective_resolution or (
                    effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
                ):
                    max_effective_resolution = effective_resolution
                    min_wasted_resolution = wasted_resolution
                    best_fit = (height, width)

            return best_fit

        if grid_pinpoints is None:
            grid_pinpoints = self.hf_model_config.image_grid_pinpoints
        if patch_size is None:
            patch_size = self.hf_model_config.vision_config.image_size

        if not isinstance(grid_pinpoints, list):
            raise TypeError("grid_pinpoints should be a list of tuples or lists")

        # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
        if not isinstance(image_size, (list, tuple)):
            if not isinstance(image_size, (torch.Tensor, np.ndarray)):
                raise TypeError(f"image_size invalid type {type(image_size)} with value {image_size}")
            image_size = image_size.tolist()

        best_resolution = select_best_resolution(image_size, grid_pinpoints)
        height, width = best_resolution
        num_patches = 0
        # consider change to ceil(height/patch_size)*ceil(width/patch_size) + 1
        for i in range(0, height, patch_size):
            for j in range(0, width, patch_size):
                num_patches += 1
        # add the base patch
        num_patches += 1
        return num_patches

    def image_size_to_num_tokens(self, image_size):
        num_patches = self.image_size_to_num_patches(image_size)
        vision_encoder_info = CLIPEncoderInfo(self.hf_model_config)
        base_feature_size = vision_encoder_info.get_num_image_tokens(
            image_width=image_size[1],
            image_height=image_size[0],
        )
        if self.hf_model_config.vision_feature_select_strategy == "default":
            base_feature_size = base_feature_size - 1
        return num_patches * base_feature_size

    @nvtx_range("[Vision] preprocess")
    def _preprocess(self, images):
        return [
            self.processor(text="dummy",
                           images=image,
                           do_rescale=not isinstance(image, torch.Tensor),
                           return_tensors="pt",
                           device=self.device)['pixel_values'][0].to(
                               self.device) for image in images
        ]

    @nvtx_range("[Vision] process")
    def _process(self, pixel_values):
        attn_metadata = self.vision_tower.prepare_attn_metadata(
            pixel_values.shape[0])
        image_features: Tuple[torch.Tensor] = self.vision_tower(
            pixel_values,
            attn_metadata=attn_metadata,
        )
        selected_image_feature = image_features[-2][:, 1:]
        image_features = self.mm_projector(selected_image_feature)
        return image_features.reshape(-1, image_features.shape[-1])

    @nvtx_range("[Vision] postprocess")
    def _postprocess(self, input_ids, mm_features):
        # Define model specific variables here before shared logic
        mm_tokens = torch.tensor([self.model_config.image_token_index
                                  ]).to(input_ids.device)
        model_hidden_size = self.model_config.text_config.hidden_size
        vocab_size = self.model_config.text_config.vocab_size
        start_len = end_len = 0  # for llava, need not append start/end token around each image token
        # End model specific variables

        ## find mm token positions in input_ids
        mm_token_positions = torch.where(torch.isin(input_ids, mm_tokens))[0]
        num_medias = num_mm_tokens = len(mm_token_positions)
        if num_medias > 1 and isinstance(mm_features, torch.Tensor):
            mm_features = list(
                mm_features.split(mm_features.shape[0] // num_medias))

        if isinstance(mm_features, torch.Tensor):
            # 1 prompt + 1 media
            # "split" means what a single mm_token in the input_ids should represent
            # image: one split --> one frame
            # video: one split --> N frames
            num_frames, mm_feature_length, mm_hidden_dim = mm_features.shape
            mm_lengths_per_split = [mm_feature_length * num_frames]
            mm_lengths_per_frame = [mm_feature_length]
        elif isinstance(mm_features, list):
            # 1 prompt + N media
            num_frames = len(mm_features) if mm_features[0].dim() == 2 else sum(
                [f.shape[0] for f in mm_features])
            mm_lengths_per_split = [
                f.shape[0] if f.dim() == 2 else f.shape[0] * f.shape[1]
                for f in mm_features
            ]
            mm_lengths_per_frame = [
                f.shape[0] if f.dim() == 2 else f.shape[1] for f in mm_features
            ]
            mm_hidden_dim = mm_features[0].shape[-1]
            mm_features = torch.cat(mm_features, dim=0)
        else:
            raise ValueError(
                f"Invalid multimodal features type: {type(mm_features)}")
        mm_total_length = sum(mm_lengths_per_split)
        assert mm_hidden_dim == model_hidden_size, "Multimodal embedding_dim must match model hidden_size"

        ## split input_ids into segments by isolating mm tokens
        mm_split_positions = torch.cat(
            [mm_token_positions, mm_token_positions + 1]).unique()
        input_ids_splits = list(input_ids.tensor_split(mm_split_positions.cpu(
        )))  # len(input_ids_splits) = num_segments after mm tokens are isolated
        mm_ids_splits = list(
            torch.arange(vocab_size,
                         vocab_size + mm_total_length,
                         device=input_ids.device).split(mm_lengths_per_split)
        )  # len(mm_ids_splits) = num_mm_segments

        for i, mm_ids in enumerate(mm_ids_splits):
            mm_ids = mm_ids.reshape(-1, mm_lengths_per_frame[i])
            mm_ids_splits[i] = mm_ids.flatten()

        ## replace mm token ids with the expanded out-of-vocab ids
        mm_split_idx = 0
        for i, split in enumerate(input_ids_splits):
            if torch.isin(split, mm_tokens).any().item():
                input_ids_splits[i] = mm_ids_splits[mm_split_idx]
                mm_split_idx += 1
        assert mm_split_idx == len(
            mm_ids_splits), "All mm_ids_splits should be consumed"

        ## concat text & mm input_ids, wrap mm feature in prompt tuning config
        fused_input_ids = torch.cat(input_ids_splits).to(
            device=input_ids.device)
        fused_length = len(input_ids) + mm_total_length + num_frames * (
            start_len + end_len) - num_medias
        assert len(
            fused_input_ids
        ) == fused_length, f"Fused input_ids length {len(fused_input_ids)} should match the sum of text and multimodal embedding lengths {fused_length}"

        # [num_frames, feature_length, hidden_dim] -> [num_frames * feature_length, hidden_dim]
        mm_features = mm_features.view(-1, mm_features.shape[-1])
        return fused_input_ids, mm_features

    @torch.inference_mode()
    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        text_prompt, mm_data = inputs.get("prompt"), inputs.get(
            "multi_modal_data", {})

        input_ids = self.tokenizer(
            text_prompt, return_tensors="pt").input_ids[0].to(self.device)

        if 'image' not in mm_data:
            return input_ids.to(torch.int32).tolist(), {}

        mm_tensor = self._preprocess(mm_data['image'])
        mm_features = torch.stack(
            [self._process(tensor) for tensor in mm_tensor])
        fused_input_ids, mm_features = self._postprocess(input_ids, mm_features)
        return fused_input_ids.to(torch.int32).tolist(), {
            "mm_embedding": mm_features
        }


    @nvtx_range("[Vision] postprocess")
    def _postprocess_ids_only(self, input_ids, total_mm_tokens):
        # Define model specific variables here before shared logic
        mm_tokens = torch.tensor([self.model_config.image_token_index
                                  ]).to(input_ids.device)
        vocab_size = self.model_config.text_config.vocab_size
        start_len = end_len = 0  # for llava, need not append start/end token around each image token
        # End model specific variables

        ## find mm token positions in input_ids
        mm_token_positions = torch.where(torch.isin(input_ids, mm_tokens))[0]
        num_medias = len(mm_token_positions)
        mm_tokens_per_media = total_mm_tokens // num_medias
        assert mm_tokens_per_media > 0, "Number of multimodal tokens per media must be greater than 0"

        # TODO: 1 prompt + N media (N>=1)  only one frame per media (image only)
        mm_lengths_per_frame = [mm_tokens_per_media] * num_medias
        mm_lengths_per_split = [mm_tokens_per_media] * num_medias
        mm_total_length = sum(mm_lengths_per_split)


        ## split input_ids into segments by isolating mm tokens
        mm_split_positions = torch.cat(
            [mm_token_positions, mm_token_positions + 1]).unique()
        input_ids_splits = list(input_ids.tensor_split(mm_split_positions.cpu(
        )))  # len(input_ids_splits) = num_segments after mm tokens are isolated
        mm_ids_splits = list(
            torch.arange(vocab_size,
                         vocab_size + mm_total_length,
                         device=input_ids.device).split(mm_lengths_per_split)
        )  # len(mm_ids_splits) = num_mm_segments

        for i, mm_ids in enumerate(mm_ids_splits):
            mm_ids = mm_ids.reshape(-1, mm_lengths_per_frame[i])
            mm_ids_splits[i] = mm_ids.flatten()

        ## replace mm token ids with the expanded out-of-vocab ids
        mm_split_idx = 0
        for i, split in enumerate(input_ids_splits):
            if torch.isin(split, mm_tokens).any().item():
                input_ids_splits[i] = mm_ids_splits[mm_split_idx]
                mm_split_idx += 1
        assert mm_split_idx == len(
            mm_ids_splits), "All mm_ids_splits should be consumed"

        ## concat text & mm input_ids, wrap mm feature in prompt tuning config
        fused_input_ids = torch.cat(input_ids_splits).to(
            device=input_ids.device)
        assert len(fused_input_ids) == len(input_ids) + mm_total_length - num_medias, "Fused input_ids length should match the sum of text and multimodal embedding lengths"
        #fused_length = len(input_ids) + mm_total_length + num_frames * (
        #    start_len + end_len) - num_medias

        return fused_input_ids

    @torch.inference_mode()
    def postprocess(
        self, inputs: TextPrompt, sampling_params: SamplingParams, disagg_mm_params: MultimodalParams,
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        text_prompt, mm_data = inputs.get("prompt"), inputs.get(
            "multi_modal_data", {})
        assert 'image' in mm_data
        model_hidden_size = self.model_config.text_config.hidden_size

        input_ids = self.tokenizer(
            text_prompt, return_tensors="pt").input_ids[0].to(self.device)
        assert len(disagg_mm_params.embeddings) == 1, "Only one fused multimodal embedding is supported"
        mm_handle = disagg_mm_params.embeddings[0]
        total_mm_tokens = mm_handle['tensor_size'][0]
        hidden_size = mm_handle['tensor_size'][-1]
        assert model_hidden_size == hidden_size, "Multimodal embedding hidden size must match model hidden size"

        fused_input_ids = self._postprocess_ids_only(input_ids, total_mm_tokens)
        return fused_input_ids.to(torch.int32).tolist()

@register_auto_model("LlavaNextForConditionalGeneration")
@register_input_processor(LlavaNextInputProcessor, model_type="llava_next")
class LlavaNextModel(PreTrainedModel):
    config_class = LlavaNextConfig

    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args,
                 **kwargs) -> None:
        config = model_config.pretrained_config
        super().__init__(config)
        if hasattr(self, "llm"):
            return

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
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        num_context_requests, num_generation_requests = attn_metadata.num_contexts, attn_metadata.num_generations
        logger.debug(f"{num_context_requests=}, {num_generation_requests=}")

        mm_embed = kwargs.get("multi_modal_data", [])
        assert mm_embed == [] or len(
            mm_embed
        ) == num_context_requests, "Number of multimodal features (if provided) should be equal to number of context requests"

        input_ids, inputs_embeds = fuse_input_embeds(
            self.llm.model.embed_tokens, input_ids, mm_embed)
        logits = self.llm.forward(attn_metadata, input_ids, position_ids,
                                  inputs_embeds, return_context_logits)
        return logits


AutoModel.register(LlavaNextConfig, LlavaNextModel)

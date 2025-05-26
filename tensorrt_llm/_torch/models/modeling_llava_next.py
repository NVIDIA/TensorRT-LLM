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
from .modeling_multimodal_utils import fuse_input_embeds, prepare_multimodal_ifb
from .modeling_utils import ModelConfig, filter_weights, register_auto_model

DISAGG = os.getenv('TLLM_MULTIMODAL_DISAGGREGATED', '0') == '1'


class LlavaNextInputProcessor(InputProcessor):

    def __init__(self, model_path, model_config, tokenizer):
        self.tokenizer = tokenizer

        if DISAGG:
            self.mm_encoder = LlavaNextVisionModel(model_path, model_config)

    @torch.inference_mode()
    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        # text input - tokenization
        text_prompt = inputs.get("prompt")
        input_ids = self.tokenizer(text_prompt,
                                   return_tensors="pt").input_ids[0]

        # multimodal input - either mm encoder or plain concatenation the raw media tensors
        mm_data = inputs.get("multi_modal_data", None)
        assert mm_data is not None and 'image' in mm_data, "Multimodal inputs must be a dictionary with 'image' field"
        if DISAGG:
            # disaggregated batching mode: multimodal encoder runs as a per-request input processor, always BS=1
            # returning expanded input_ids & multimodal embedding
            mm_embeds, mm_embed_lengths = self.mm_encoder.forward(
                mm_data['image'])
            fused_input_ids, mm_embeds = self.mm_encoder._postprocess(
                input_ids.to(mm_embeds.device), mm_embeds, mm_embed_lengths)
            return fused_input_ids.tolist(), {"mm_embedding": mm_embeds}
        else:
            # inflight batching mode: multimodal encoder runs as part of the model forward
            # returning raw input_ids & raw media
            # within each request, input images/frames are usually of the same size; across requests, the sizes may vary
            # InputProcessor is per-request call, so we can stack all images/frames as [num_images, C, H, W]. still on CPU.
            mm_data = torch.stack(mm_data['image'])
            return input_ids.tolist(), {"mm_embedding": mm_data}


class LlavaNextVisionModel:

    def __init__(self, model_path, model_config):
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

        self.max_batch_size = 8

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
    def _postprocess(self, input_ids, mm_features, mm_feature_lengths):
        # Define model specific variables here before shared logic
        mm_tokens = torch.tensor([self.model_config.image_token_index
                                  ]).to(input_ids.device)
        model_hidden_size = self.model_config.text_config.hidden_size
        vocab_size = self.model_config.text_config.vocab_size
        start_len = end_len = 0  # for llava, need not append start/end token around each image token
        # End model specific variables

        mm_features = mm_features.reshape(len(mm_feature_lengths), -1,
                                          mm_features.shape[-1])

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

        ## concat text & mm input_ids
        fused_input_ids = torch.cat(input_ids_splits).to(
            device=input_ids.device)
        fused_length = len(input_ids) + mm_total_length + num_frames * (
            start_len + end_len) - num_medias
        assert len(
            fused_input_ids
        ) == fused_length, f"Fused input_ids length {len(fused_input_ids)} should match the sum of text and multimodal embedding lengths {fused_length}"

        # [num_frames, feature_length, hidden_dim] -> [num_frames * feature_length, hidden_dim]
        mm_features = mm_features.view(-1, mm_features.shape[-1])
        return fused_input_ids.to(torch.int32), mm_features

    @torch.inference_mode()
    def forward(self,
                images: List[torch.Tensor]) -> Tuple[torch.Tensor, List[int]]:

        # List[raw image tensors (CPU)] for N requests, could have different sizes --> CPU compute-vision ops e.g. resize, crop, etc. --> List[preprocessed tensors (GPU)] for N requests
        mm_preprocessed = self._preprocess(images)
        mm_preprocessed_lengths = [t.size(0) for t in mm_preprocessed]

        # concatenated preprocessed tensors (GPU) --> GPU model forward --> mm embedding tensors (GPU), [total_num_media_tokens, hidden_dim]
        mm_embeds = self._process(torch.cat(mm_preprocessed))
        mm_embed_lengths = [
            mm_embeds.shape[0] // sum(mm_preprocessed_lengths) * l
            for l in mm_preprocessed_lengths
        ]  # calculate num_media_tokens for each request, which is proportional to the preprocessed length

        return mm_embeds, mm_embed_lengths


@register_auto_model("LlavaNextForConditionalGeneration")
@register_input_processor(LlavaNextInputProcessor)
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

        if not DISAGG:
            self.mm_encoder = LlavaNextVisionModel(config.checkpoint_dir,
                                                   config)
            self.mm_tokens = torch.tensor([config.image_token_index],
                                          device='cuda')

        self.post_config()

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

        mm_data = kwargs.get("multi_modal_data", [])
        if DISAGG:
            # disaggregated batching mode: multimodal context phase is decoupled from LLM forward. mm_data is processed multimodal embedding
            logger.warning(
                "No multimodal encoder found in model definition. You might be in disaggregated inference mode. Skipping multimodal processing. It's expected that a expanded input_ids and mm_embed are provided as inputs."
            )
            mm_embed = mm_data
            assert mm_embed == [] or len(
                mm_embed
            ) == num_context_requests, f"Number of multimodal tensors ({len(mm_data)}) should be equal to number of context requests ({num_context_requests}) in the batch."
            input_ids, inputs_embeds = fuse_input_embeds(
                self.llm.model.embed_tokens, input_ids, mm_embed)
        else:
            # inflight batching mode: multimodal context phase is fused in LLM forward. mm_data is raw media tensors
            mm_embeds, mm_embed_lengths = None, None
            if len(mm_data) > 0:
                assert len(
                    mm_data
                ) == num_context_requests, f"Number of multimodal tensors ({len(mm_data)}) should be equal to number of context requests ({num_context_requests}) in the batch."
                mm_embeds, mm_embed_lengths = self.mm_encoder.forward(mm_data)

            attn_metadata, input_ids, position_ids, inputs_embeds = prepare_multimodal_ifb(
                self.llm.model.embed_tokens,
                attn_metadata,
                input_ids,
                position_ids,
                self.mm_tokens,
                mm_embeds,
                mm_embed_lengths,
                prepare_position_ids=self.llm.model.layers[0].self_attn.
                apply_rotary_emb
            )  # position_ids is only relevant when RoPE needs to be explicitly applied outside the fused attention op.

        logits = self.llm.forward(attn_metadata, input_ids, position_ids,
                                  inputs_embeds, return_context_logits)
        return logits


AutoModel.register(LlavaNextConfig, LlavaNextModel)

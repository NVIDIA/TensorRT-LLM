# Plan for phi4-mm model support.
# (done) step 1: support legacy inference pipeline for phi4-mm model.
# (todo) step 2: refactor the inference pipeline to use AGGREGATE mode (https://github.com/NVIDIA/TensorRT-LLM/pull/5522).

import copy
from typing import List, Optional, Tuple

import torch
import transformers
from PIL import Image

from ...executor.request import LoRARequest
from ...inputs import (ExtraProcessedInputs, InputProcessor,
                       MultimodalPlaceholderMetadata,
                       MultimodalPlaceholderPlacement, TextPrompt,
                       register_input_processor)
from ...logger import logger
from ...lora_helper import LoraConfig
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from .modeling_auto import AutoModelForCausalLM
from .modeling_multimodal_utils import fuse_input_embeds
from .modeling_utils import register_auto_model

# Special tokens
_IMAGE_SPECIAL_TOKEN_ID = 200010  # '<|endoftext10|>'
_AUDIO_SPECIAL_TOKEN_ID = 200011  # '<|endoftext11|>'


# Create a PreTrainedModel class for transformers=4.53.1 upgrade.
# Core idea is to provide `prepare_inputs_for_generation` method from `GenerationMixin`.
class NewPreTrainedModel(transformers.modeling_utils.PreTrainedModel,
                         transformers.generation.GenerationMixin):
    pass


class Phi4MMInputProcessor(InputProcessor):

    def __init__(self,
                 model_path: str,
                 model_config: transformers.PretrainedConfig,
                 tokenizer: transformers.AutoTokenizer,
                 trust_remote_code: bool = True):
        assert trust_remote_code, "trust_remote_code must be True for Phi4MM"

        self.model_config = model_config
        self.device = 'cuda'

        self.tokenizer = tokenizer
        self.use_fast = True
        if self.tokenizer is None:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                use_fast=self.use_fast)

        self.processor = transformers.AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            use_fast=self.use_fast)

        # Build pure-pytorch model architecture for multimodal encoder.
        # Model weights are also loaded here.
        OldPreTrainedModel = transformers.modeling_utils.PreTrainedModel
        transformers.modeling_utils.PreTrainedModel = NewPreTrainedModel
        # TODO: Make separate Phi4VisionEncoder and Phi4AudioEncoder, and move them to LLM-side.
        ref_phi4mm_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            # Flash_attn_2 only supports bf16 or fp16 and set in HF config.
            torch_dtype='auto',
            _attn_implementation='flash_attention_2',
        ).eval()
        transformers.modeling_utils.PreTrainedModel = OldPreTrainedModel
        self.phi4mm_modal_encoder = ref_phi4mm_model.model.embed_tokens_extend.to(
            self.device)
        # Required by Phi4MMImageAudioEmbedding.
        # See link: https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/modeling_phi4mm.py#L701
        self.phi4mm_wte = ref_phi4mm_model.model.embed_tokens.to(self.device)

    @torch.inference_mode()
    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        text_prompt, mm_data, mm_processor_kwargs = inputs.get("prompt"), \
                        inputs.get("multi_modal_data", {}), inputs.get("mm_processor_kwargs", {})
        images = mm_data.get("image", None)
        audios = mm_data.get("audio", None)

        if images is not None:
            if isinstance(images[0], torch.Tensor):
                # Convert normalized tensors (0-1) to PIL images (0-255).
                images = [
                    Image.fromarray((image.permute(1, 2, 0) * 255).to(
                        torch.uint8).cpu().numpy()) for image in images
                ]

        # Preprocessing for multimodal data.
        inputs = self.processor(text=[text_prompt],
                                images=images,
                                audios=audios,
                                return_tensors='pt').to(self.device)

        # Set audio_projection_mode according to the modality.
        # Ref: https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/modeling_phi4mm.py#L2103
        if images is not None:
            audio_projection_mode = 'vision'
        elif audios is not None:
            audio_projection_mode = 'speech'
        else:
            audio_projection_mode = 'speech'

        # Processing with Phi4MMImageAudioEmbedding.
        mm_features = self.phi4mm_modal_encoder(
            input_ids=inputs['input_ids'],
            input_embeds=None,
            input_image_embeds=inputs['input_image_embeds'],
            input_audio_embeds=inputs['input_audio_embeds'],
            image_sizes=inputs['image_sizes'],
            image_attention_mask=inputs['image_attention_mask'],
            audio_embed_sizes=inputs['audio_embed_sizes'],
            audio_attention_mask=inputs['audio_attention_mask'],
            audio_projection_mode=audio_projection_mode,
            wte=self.phi4mm_wte,
        )

        # Postprocessing to get multimodal-only embeddings.
        image_token_mask = inputs['input_ids'] == _IMAGE_SPECIAL_TOKEN_ID
        audio_token_mask = inputs['input_ids'] == _AUDIO_SPECIAL_TOKEN_ID
        mm_token_mask = image_token_mask | audio_token_mask
        mm_features = mm_features[mm_token_mask]

        multimodal_data = {}
        multimodal_data["multimodal_embedding"] = mm_features

        return inputs['input_ids'][0].to(torch.int32).tolist(), {
            "multimodal_data": multimodal_data,
        }


@register_auto_model("Phi4MMForCausalLM")
@register_input_processor(
    Phi4MMInputProcessor,
    model_type="phi4mm",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={
            "image": "<|image_{0}|>",
            "audio": "<|audio_{0}|>",
        },
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
        placeholders_separator="",
    ))
class Phi4MMForCausalLM(transformers.PreTrainedModel):

    _supports_flash_attn_2 = True
    MM_TOKEN_IDS = torch.tensor(
        [_IMAGE_SPECIAL_TOKEN_ID, _AUDIO_SPECIAL_TOKEN_ID])

    def __init__(self, model_config: ModelConfig):

        config = model_config.pretrained_config
        super().__init__(config)

        self.model_config = model_config
        if hasattr(self, "llm"):
            return

        # We use Phi3ForCausalLM as the language model.
        llm_model_config = copy.deepcopy(model_config)
        llm_model_config.pretrained_config.architectures = ["Phi3ForCausalLM"]
        # Only build the language model architecture without loading weights.
        self.llm = AutoModelForCausalLM.from_config(llm_model_config)

        self.vocab_size = config.vocab_size
        self.model_dtype = getattr(config, "torch_dtype", torch.float16)
        logger.info(f"{self.dtype=} {self.model_dtype=}")
        self.post_config()
        self.is_loaded = True

    def load_weights(self, weights):
        # Filter out non-language model weights.
        weights = {
            k: v
            for k, v in weights.items()
            if not k.startswith('model.embed_tokens_extend')
        }
        # Filter out LoRA weights.
        # LoRA weights will be loaded by LoraManager.
        weights = {k: v for k, v in weights.items() if '.lora_' not in k}
        # Rename base layer weights.
        updated_weights = {}
        for k in weights.keys():
            if 'base_layer.weight' in k:
                new_k = k.replace('base_layer.weight', 'weight')
                updated_weights[new_k] = weights[k]
            else:
                updated_weights[k] = weights[k]
        weights = updated_weights

        self.llm.load_weights(weights)

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
        mm_embeds = []
        if len(multimodal_params) > 0:
            mm_embeds = [
                multimodal_param.multimodal_data["multimodal_embedding"]
                for multimodal_param in multimodal_params
            ]
        input_ids, input_embeds = fuse_input_embeds(
            self.llm.model.embed_tokens,
            input_ids,
            mm_embeds,
            mm_token_ids=self.MM_TOKEN_IDS,
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

    @staticmethod
    def lora_config(model_dir: str):
        _lora_config = LoraConfig(
            lora_target_modules=[
                "attn_qkv",
                "attn_dense",
                "mlp_gate_up",
                "mlp_4h_to_h",
            ],
            trtllm_modules_to_hf_modules={
                "attn_qkv": "qkv_proj",
                "attn_dense": "o_proj",
                "mlp_gate_up": "gate_up_proj",
                "mlp_4h_to_h": "down_proj",
            },
            max_lora_rank=320,  # Max rank for Phi4MM.
            swap_gate_up_proj_lora_b_weight=
            False,  # Disable swap gate_up_proj.lora_B.weight for Phi4MM.
        )
        return _lora_config

    @staticmethod
    def lora_request(num_requests: int, modality: str, base_model_dir: str):
        # Prepare LoRA requests for different modalities.
        # Ref: https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/modeling_phi4mm.py#L2103
        lora_request = None
        if modality == "image" or modality == "image_audio":
            lora_request = [
                LoRARequest(
                    lora_name="vision-lora",
                    lora_int_id=0,
                    lora_path=f"{base_model_dir}/vision-lora",
                ) for i in range(num_requests)
            ]
        elif modality == "audio":
            lora_request = [
                LoRARequest(
                    lora_name="speech-lora",
                    lora_int_id=1,
                    lora_path=f"{base_model_dir}/speech-lora",
                ) for i in range(num_requests)
            ]

        return lora_request

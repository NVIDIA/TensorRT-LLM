# Plan for phi4-mm model support.
# (done) step 1: support legacy inference pipeline for phi4-mm model.
# (todo) step 2: refactor the inference pipeline to use AGGREGATE mode (https://github.com/NVIDIA/TensorRT-LLM/pull/5522).

import copy
from typing import List, Optional, Tuple

import torch
import transformers
from PIL import Image

from ...executor.request import LoRARequest
from ...inputs import (ExtraProcessedInputs, InputProcessor, TextPrompt,
                       register_input_processor)
from ...logger import logger
from ...lora_manager import LoraConfig
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from .modeling_auto import AutoModelForCausalLM
from .modeling_multimodal_utils import fuse_input_embeds
from .modeling_utils import register_auto_model

from .utils_phi4mm import Phi4MMImageEmbedding, Phi4MMAudioEmbedding
from .configuration_phi4mm import Phi4MMConfig

# Special tokens
_IMAGE_SPECIAL_TOKEN_ID = 200010  # '<|endoftext10|>'
_AUDIO_SPECIAL_TOKEN_ID = 200011  # '<|endoftext11|>'
_COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE = [-9999, -1]  # For backward compatibility
_COMPATIBLE_AUDIO_SPECIAL_TOKEN_ID_RANGE = [float('-inf'), -10000]  # For backward compatibility


class HFPhi4MultimodalEncoder(transformers.PreTrainedModel, transformers.generation.GenerationMixin):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Phi4MMDecoderLayer`]

    Args:
        config: Phi4MMConfig
    """

    config_class = Phi4MMConfig
    base_model_prefix = "model"
    _tied_weights_keys = ["lm_head.weight"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def __init__(self, config: transformers.PretrainedConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id

        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Initialize embed_tokens_extend as a container to match HF model weights.
        self.embed_tokens_extend = torch.nn.Module()

        self._attn_implementation = config._attn_implementation
        self.gradient_checkpointing = False

        self.vocab_size = config.vocab_size

        embedding_config = {
            'embedding_cls': config.embd_layer['embedding_cls'],
            **config.embd_layer
        }
        self.image_input_id = embedding_config.get('image_input_id', -1)
        self.audio_input_id = embedding_config.get('audio_input_id', -10000)
        assert self.image_input_id != self.audio_input_id, 'image_input_id and audio_input_id should be different'

        self.image_embd_layer_kwargs = embedding_config['image_embd_layer']
        self.embed_tokens_extend.image_embed = Phi4MMImageEmbedding(config, **self.image_embd_layer_kwargs)

        self.audio_embd_layer_kwargs = embedding_config['audio_embd_layer']
        self.embed_tokens_extend.audio_embed = Phi4MMAudioEmbedding(config, **self.audio_embd_layer_kwargs)

        self.input_image_embeds = None
        self.image_sizes = None
        self.image_attention_mask = None
        self.input_audio_embeds = None
        self.audio_embed_sizes = None

    def post_init(self, audio_config):
        # post init for audio embedding
        # ref: model.model.embed_tokens_extend.post_init(audio_config) in phyagi/getters/model.py
        self.embed_tokens_extend.audio_embed.post_init(audio_config)

    def set_input_image_embeds(self, input_image_embeds: torch.FloatTensor) -> None:
        self.input_image_embeds = input_image_embeds

    def set_image_sizes(self, image_sizes: torch.LongTensor) -> None:
        self.image_sizes = image_sizes

    def set_img_attn_mask(self, image_attention_mask: torch.FloatTensor) -> None:
        self.image_attention_mask = image_attention_mask

    def set_input_audio_embeds(self, input_audio_embeds: torch.FloatTensor) -> None:
        self.input_audio_embeds = input_audio_embeds

    def set_audio_embed_sizes(self, audio_embed_sizes: torch.LongTensor) -> None:
        self.audio_embed_sizes = audio_embed_sizes

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeds,
        input_image_embeds: Optional[torch.FloatTensor]=None,
        input_audio_embeds: Optional[torch.FloatTensor]=None,
        image_sizes=None,
        image_attention_mask=None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        audio_projection_mode='speech',
        wte=None,
    ) -> torch.FloatTensor:
        MAX_INPUT_ID = int(1e9)
        assert -MAX_INPUT_ID < self.audio_input_id < self.image_input_id

        # override image and audio embeddings and sizes from object itself
        # this is for inference
        # ref: phyagi/eval/utils/text_generation_vision_audio_pipeline.py
        if self.input_image_embeds is not None:
            assert input_image_embeds is None
            input_image_embeds = self.input_image_embeds.clone()
            # NOTE weijian: set input_image_embeds to None after first call in for eval stage
            #               during evaluation, it will call model's forward() multiple times
            #               the first time input_ids contains the prompt (including <|image_{}|>) and input_embeds exists
            #               from the second time, the input_ids will only contain the generated text
            #               thus, the input_image_embeds is no longer needed
            self.input_image_embeds = None

        if self.image_sizes is not None:
            assert image_sizes is None
            image_sizes = self.image_sizes

        if self.input_audio_embeds is not None:
            assert input_audio_embeds is None
            input_audio_embeds = self.input_audio_embeds.clone()
            self.input_audio_embeds = None

        if self.audio_embed_sizes is not None:
            assert audio_embed_sizes is None
            audio_embed_sizes = self.audio_embed_sizes.clone()

        if self.image_attention_mask is not None:
            assert image_attention_mask is None
            image_attention_mask = self.image_attention_mask.clone()
            self.image_attention_mask = None

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        # backward compatibility
        with torch.no_grad():
            new_input_ids = input_ids.clone()
            new_input_ids[(input_ids >= _COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE[0]) &
                        (input_ids <= _COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE[1])] = _IMAGE_SPECIAL_TOKEN_ID
            new_input_ids[(input_ids >= _COMPATIBLE_AUDIO_SPECIAL_TOKEN_ID_RANGE[0]) &
                        (input_ids <= _COMPATIBLE_AUDIO_SPECIAL_TOKEN_ID_RANGE[1])] = _AUDIO_SPECIAL_TOKEN_ID
            input_ids = new_input_ids

        with torch.no_grad():
            image_position_mask = input_ids == _IMAGE_SPECIAL_TOKEN_ID
            non_image_position_mask = ~image_position_mask

        assert input_embeds is None
        if self.training:
            assert input_image_embeds is not None or input_audio_embeds is not None

        image_hidden_states = None
        audio_hidden_states = None

        if input_image_embeds is not None:
            image_hidden_states = self.embed_tokens_extend.image_embed(
                input_ids=input_ids,
                input_embeds=input_image_embeds,
                image_sizes=image_sizes,
                wte=wte,
                image_attention_mask=image_attention_mask
            )
        if input_audio_embeds is not None:
            audio_hidden_states = self.embed_tokens_extend.audio_embed(
                input_ids=input_ids,
                input_embeds=input_audio_embeds,
                audio_embed_sizes=audio_embed_sizes,
                audio_attention_mask=audio_attention_mask,
                wte=wte,
                audio_projection_mode=audio_projection_mode,
            )

        # merge image and audio hidden states
        # NOTE weijian: for non-image-audio tokens, here we use audio hidden states
        #               actually, in the debug code above, the non-image-audio tokens from image_hidden_states and audio_hidden_states should be the same
        if input_image_embeds is not None and input_audio_embeds is not None:
            dtype = image_hidden_states.dtype
            hidden_states = image_hidden_states * image_position_mask.to(dtype).unsqueeze(-1) + audio_hidden_states * non_image_position_mask.to(dtype).unsqueeze(-1)
        elif input_image_embeds is not None:
            hidden_states = image_hidden_states
        elif input_audio_embeds is not None:
            hidden_states = audio_hidden_states
        else:
            assert wte is not None
            hidden_states = wte(input_ids)

        return hidden_states


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

        self.hf_phi4mm_model = HFPhi4MultimodalEncoder.from_pretrained(
            model_path,
            trust_remote_code=False,
            # Flash_attn_2 only supports bf16 or fp16 and set in HF config.
            torch_dtype='auto',
            _attn_implementation='flash_attention_2',
        ).eval()
        # Move the model to device and set dtype
        self.hf_phi4mm_model = self.hf_phi4mm_model.to(self.device)

        # Required by HFPhi4MultimodalEncoder.
        self.phi4mm_wte = self.hf_phi4mm_model.embed_tokens.to(self.device)

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
        mm_features = self.hf_phi4mm_model.forward(
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
@register_input_processor(Phi4MMInputProcessor, model_type="phi4mm")
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
        mm_embedding = [
            multimodal_param.multimodal_data["multimodal_embedding"]
            for multimodal_param in multimodal_params
        ]
        input_ids, input_embeds = fuse_input_embeds(
            self.llm.model.embed_tokens,
            input_ids,
            mm_embedding,
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
            lora_dir=[
                f"{model_dir}/vision-lora",
                f"{model_dir}/speech-lora",
            ],
            lora_target_modules=[
                "attn_qkv",
                "attn_dense",
                "mlp_h_to_4h",
                "mlp_4h_to_h",
            ],
            trtllm_modules_to_hf_modules={
                "attn_qkv": "qkv_proj",
                "attn_dense": "o_proj",
                "mlp_h_to_4h": "gate_up_proj",
                "mlp_4h_to_h": "down_proj",
            },
            max_lora_rank=320,  # Max rank for Phi4MM.
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
                    lora_name=f"vision-lora-{i}",
                    lora_int_id=i,
                    lora_path=f"{base_model_dir}/vision-lora",
                ) for i in range(num_requests)
            ]
        elif modality == "audio":
            lora_request = [
                LoRARequest(
                    lora_name=f"speech-lora-{i}",
                    lora_int_id=i,
                    lora_path=f"{base_model_dir}/speech-lora",
                ) for i in range(num_requests)
            ]

        return lora_request

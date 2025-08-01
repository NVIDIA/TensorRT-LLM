# Plan for phi4-mm model support.
# (done) step 1: support legacy inference pipeline for phi4-mm model.
# (done) step 2: refactor the inference pipeline to use AGGREGATE mode (https://github.com/NVIDIA/TensorRT-LLM/pull/5522).
# (todo) step 3: optimization
#   * use TRTLLM-attention to replace original pytorch attention in vision/audio encoders.
#   * use data parallel to accelerate inference.
#   * batch inference for encoder.

import copy
import importlib
import os
import sys
from pathlib import Path
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

# Special token ids from the original Phi-4-multimodal-instruct implementation
_IMAGE_SPECIAL_TOKEN_ID = 200010  # '<|endoftext10|>'
_AUDIO_SPECIAL_TOKEN_ID = 200011  # '<|endoftext11|>'
_COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE = [-9999, -1]
_COMPATIBLE_AUDIO_SPECIAL_TOKEN_ID_RANGE = [float('-inf'), -10000]
_MM_TOKEN_IDS = torch.tensor([_IMAGE_SPECIAL_TOKEN_ID, _AUDIO_SPECIAL_TOKEN_ID])

# Below classes will be loaded from HuggingFace codes, rather than using transformers version,
# since transformers version is not compatible with checkpoints and configs from `microsoft/Phi-4-multimodal-instruct`.
Phi4MMAudioEmbedding = None
Phi4MMImageEmbedding = None
Phi4MMConfig = None


# Make this a runtime lookup rather than a module-wide constant for easier unit testing.
def _is_disagg() -> bool:
    return os.getenv("TLLM_MULTIMODAL_DISAGGREGATED", "0") == "1"


def _load_phi4mm_classes(local_path):
    """Load Phi4MM classes from the specified local path."""
    global Phi4MMAudioEmbedding, Phi4MMImageEmbedding, Phi4MMConfig
    if Phi4MMAudioEmbedding is not None and Phi4MMImageEmbedding is not None and Phi4MMConfig is not None:
        return

    # Add parent folder to sys.path to enable relative import.
    original_sys_path = sys.path.copy()
    package_folder = Path(local_path)
    parent_folder = str(package_folder.parent)
    if parent_folder not in sys.path:
        sys.path.insert(0, parent_folder)

    try:
        # Import Phi4MMConfig from configuration_phi4mm.py.
        config_path = os.path.join(local_path, 'configuration_phi4mm.py')
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"configuration_phi4mm.py not found at {local_path}.")
        spec = importlib.util.spec_from_file_location("hf_config", config_path)
        hf_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hf_config)
        Phi4MMConfig = hf_config.Phi4MMConfig

        # Import Phi4MMAudioEmbedding and Phi4MMImageEmbedding from modeling_phi4mm.py.
        modeling_phi4mm_path = os.path.join(local_path, 'modeling_phi4mm.py')
        if not os.path.exists(modeling_phi4mm_path):
            raise FileNotFoundError(
                f"modeling_phi4mm.py not found at {local_path}.")
        # `Phi-4-multimodal-instruct` as the package name to avoid relative import errors.
        # `hf_modeling_phi4mm` as the module name to avoid name conflicts.
        spec = importlib.util.spec_from_file_location(
            "Phi-4-multimodal-instruct.hf_modeling_phi4mm",
            modeling_phi4mm_path)
        hf_modeling_phi4mm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hf_modeling_phi4mm)
        Phi4MMAudioEmbedding = hf_modeling_phi4mm.Phi4MMAudioEmbedding
        Phi4MMImageEmbedding = hf_modeling_phi4mm.Phi4MMImageEmbedding
    finally:
        sys.path = original_sys_path


class HFPhi4MultimodalEncoder(transformers.PreTrainedModel,
                              transformers.generation.GenerationMixin):

    # Copy and modify from https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/modeling_phi4mm.py::Phi4MMImageAudioEmbedding
    # Note: the HF implementation here will cause duplicated encoders on all GPUs for TP>1 scenario.
    # TODO: use TRTLLM-attention to replace original pytorch Flash_attn_2 in HFPhi4MultimodalEncoder.
    config_class = Phi4MMConfig
    base_model_prefix = "model"
    _tied_weights_keys = ["lm_head.weight"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def __init__(self, config: transformers.PretrainedConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.padding_idx = config.pad_token_id

        self.embed_tokens = torch.nn.Embedding(config.vocab_size,
                                               config.hidden_size,
                                               self.padding_idx)

        self._attn_implementation = config._attn_implementation

        self.vocab_size = config.vocab_size

        embedding_config = {
            'embedding_cls': config.embd_layer['embedding_cls'],
            **config.embd_layer
        }
        # The default values are from HuggingFace Phi-4-multimodal-instruct codes.
        self.image_input_id = embedding_config.get('image_input_id', -1)
        self.audio_input_id = embedding_config.get('audio_input_id', -10000)
        if self.image_input_id == self.audio_input_id:
            raise ValueError(
                'image_input_id and audio_input_id should be different')

        self.image_embd_layer_kwargs = embedding_config['image_embd_layer']
        self.image_embed = Phi4MMImageEmbedding(config,
                                                **self.image_embd_layer_kwargs)

        self.audio_embd_layer_kwargs = embedding_config['audio_embd_layer']
        self.audio_embed = Phi4MMAudioEmbedding(config,
                                                **self.audio_embd_layer_kwargs)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.LongTensor,
        input_image_embeds: Optional[torch.FloatTensor] = None,
        input_audio_embeds: Optional[torch.FloatTensor] = None,
        image_sizes=None,
        image_attention_mask=None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        audio_projection_mode='speech',
        wte=None,
    ) -> torch.FloatTensor:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        # Inplace-replacement for special token ids.
        torch.where(
            (input_ids >= _COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE[0])
            & (input_ids <= _COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE[1]),
            torch.tensor(_IMAGE_SPECIAL_TOKEN_ID),
            input_ids,
            out=input_ids,
        )
        torch.where(
            (input_ids >= _COMPATIBLE_AUDIO_SPECIAL_TOKEN_ID_RANGE[0])
            & (input_ids <= _COMPATIBLE_AUDIO_SPECIAL_TOKEN_ID_RANGE[1]),
            torch.tensor(_AUDIO_SPECIAL_TOKEN_ID),
            input_ids,
            out=input_ids,
        )
        image_position_mask = input_ids == _IMAGE_SPECIAL_TOKEN_ID
        non_image_position_mask = ~image_position_mask

        image_hidden_states = None
        if input_image_embeds is not None:
            image_hidden_states = self.image_embed(
                input_ids=input_ids,
                input_embeds=input_image_embeds,
                image_sizes=image_sizes,
                wte=wte,
                image_attention_mask=image_attention_mask,
            )
        audio_hidden_states = None
        if input_audio_embeds is not None:
            audio_hidden_states = self.audio_embed(
                input_ids=input_ids,
                input_embeds=input_audio_embeds,
                audio_embed_sizes=audio_embed_sizes,
                audio_attention_mask=audio_attention_mask,
                wte=wte,
                audio_projection_mode=audio_projection_mode,
            )

        if input_image_embeds is not None and input_audio_embeds is not None:
            dtype = image_hidden_states.dtype
            hidden_states = image_hidden_states * image_position_mask.to(
                dtype).unsqueeze(
                    -1) + audio_hidden_states * non_image_position_mask.to(
                        dtype).unsqueeze(-1)
        elif input_image_embeds is not None:
            hidden_states = image_hidden_states
        elif input_audio_embeds is not None:
            hidden_states = audio_hidden_states
        else:
            if wte is None:
                raise ValueError(
                    "wte is required for Phi4MM if no image/audio data is provided"
                )
            hidden_states = wte(input_ids)

        # Postprocessing to get multimodal-only embeddings.
        mm_token_mask = torch.isin(input_ids,
                                   _MM_TOKEN_IDS.to(input_ids.device))
        hidden_states = hidden_states[mm_token_mask]

        return hidden_states


class Phi4MMInputProcessor(InputProcessor):

    def __init__(self,
                 model_path: str,
                 model_config: transformers.PretrainedConfig,
                 tokenizer: transformers.AutoTokenizer,
                 trust_remote_code: bool = True):
        if not trust_remote_code:
            raise ValueError("trust_remote_code must be True for Phi4MM")

        self.model_config = model_config
        self.device = 'cpu'

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

    @torch.inference_mode()
    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        text_prompt, mm_data = inputs.get("prompt"), inputs.get(
            "multi_modal_data", {})
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

        # Will package inputs for language model forward in AGGREGATE mode.
        multimodal_data = {}
        multimodal_data['input_ids'] = inputs['input_ids']
        multimodal_data['input_image_embeds'] = inputs['input_image_embeds']
        multimodal_data['image_sizes'] = inputs['image_sizes']
        multimodal_data['image_attention_mask'] = inputs['image_attention_mask']
        multimodal_data['input_audio_embeds'] = inputs['input_audio_embeds']
        multimodal_data['audio_embed_sizes'] = inputs['audio_embed_sizes']
        multimodal_data['audio_attention_mask'] = inputs['audio_attention_mask']
        multimodal_data['audio_projection_mode'] = audio_projection_mode
        return inputs['input_ids'][0].to(torch.int32).tolist(), {
            "multimodal_data": multimodal_data,
        }


@register_auto_model("Phi4MMForCausalLM")
@register_input_processor(Phi4MMInputProcessor, model_type="phi4mm")
class Phi4MMForCausalLM(transformers.PreTrainedModel):

    _supports_flash_attn_2 = True

    def __init__(self, model_config: ModelConfig):
        if _is_disagg():
            raise ValueError(
                "Phi4MM does not support disaggregated inference yet.")

        config = model_config.pretrained_config
        super().__init__(config)

        self.model_config = model_config
        if hasattr(self, "llm"):
            return

        if not _is_disagg():
            _load_phi4mm_classes(config._name_or_path)

            # Setup HFPhi4MultimodalEncoder in AGGREGATE mode.
            self.hf_phi4mm_model = HFPhi4MultimodalEncoder(config).eval()
            self.hf_phi4mm_model.to(config.torch_dtype)
            # Required by HFPhi4MultimodalEncoder.
            self.phi4mm_wte = self.hf_phi4mm_model.embed_tokens

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
        # Load weights into HFPhi4MultimodalEncoder.
        if not _is_disagg():
            filtered_weights = {}
            for k, v in weights.items():
                if k.startswith("model.embed_tokens."):
                    new_k = k.replace("model.embed_tokens.", "embed_tokens.")
                    filtered_weights[new_k] = v
                elif k.startswith("model.embed_tokens_extend."):
                    new_k = k.replace("model.embed_tokens_extend.", "")
                    filtered_weights[new_k] = v
            self.hf_phi4mm_model.load_state_dict(filtered_weights, strict=True)

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
        mm_embedding = []
        if len(multimodal_params) > 0:
            if not _is_disagg():
                # Forward the multimodal data to HFPhi4MultimodalEncoder in AGGREGATE mode.
                # Note: HFPhi4MultimodalEncoder does not support data parallel or batching.
                for i in range(len(multimodal_params)):
                    mm_features = \
                        self.hf_phi4mm_model(
                            input_ids=multimodal_params[i].multimodal_data["input_ids"],
                            input_image_embeds=multimodal_params[i].multimodal_data["input_image_embeds"],
                            input_audio_embeds=multimodal_params[i].multimodal_data["input_audio_embeds"],
                            image_sizes=multimodal_params[i].multimodal_data["image_sizes"],
                            image_attention_mask=multimodal_params[i].multimodal_data["image_attention_mask"],
                            audio_embed_sizes=multimodal_params[i].multimodal_data["audio_embed_sizes"],
                            audio_attention_mask=multimodal_params[i].multimodal_data["audio_attention_mask"],
                            audio_projection_mode=multimodal_params[i].multimodal_data["audio_projection_mode"],
                            wte=self.phi4mm_wte,
                        )
                    mm_embedding.append(mm_features)
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
            mm_token_ids=_MM_TOKEN_IDS,
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

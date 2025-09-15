# Plan for phi4-mm model support.
# (done) step 1: support legacy inference pipeline for phi4-mm model.
# (done) step 2: refactor the inference pipeline to use AGGREGATE mode (https://github.com/NVIDIA/TensorRT-LLM/pull/5522).
# (todo) step 3: optimization
#   * use TRTLLM-attention to replace original pytorch attention in vision/audio encoders.
#   * use data parallel to accelerate inference.

import copy
import importlib
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import transformers
from PIL import Image

from tensorrt_llm.inputs.multimodal import MultimodalParams

from ...executor.request import LoRARequest
from ...inputs import (BaseMultimodalInputProcessor, ExtraProcessedInputs,
                       InputProcessor, MultimodalPlaceholderMetadata,
                       MultimodalPlaceholderPlacement, TextPrompt,
                       register_input_processor)
from ...logger import logger
from ...lora_helper import LoraConfig
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from .modeling_auto import AutoModelForCausalLM
from .modeling_multimodal_utils import find_input_mm_embeds, fuse_input_embeds
from .modeling_utils import register_auto_model

# Special token ids from the original Phi-4-multimodal-instruct implementation
_IMAGE_SPECIAL_TOKEN_ID = 200010  # '<|endoftext10|>' from HF `modeling_phi4mm.py`
_AUDIO_SPECIAL_TOKEN_ID = 200011  # '<|endoftext11|>' from HF `modeling_phi4mm.py`
_PAD_TOKEN_ID = 199999  # '<|endoftext|>' from HF `special_tokens_map.json`
_COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE = [-9999,
                                            -1]  # from HF `modeling_phi4mm.py`
_COMPATIBLE_AUDIO_SPECIAL_TOKEN_ID_RANGE = [float('-inf'), -10000
                                            ]  # from HF `modeling_phi4mm.py`

# Below classes will be loaded from HuggingFace codes, rather than using transformers version,
# since transformers version is not compatible with checkpoints and configs from `microsoft/Phi-4-multimodal-instruct`.
Phi4MMAudioEmbedding = None
Phi4MMImageEmbedding = None
Phi4MMConfig = None


# Make this a runtime lookup rather than a module-wide constant for easier unit testing.
def _is_disagg() -> bool:
    return os.getenv("TLLM_MULTIMODAL_DISAGGREGATED", "0") == "1"


# Load the Phi4MM classes from HuggingFace Phi-4-multimodal-instruct repo.
# Remove this function by using the transformers version of Phi4Multimodal when weights/configs are converted to transformers format.
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

    def _replace_special_token_ids(self,
                                   input_ids: torch.Tensor) -> torch.Tensor:
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
        return input_ids

    def _batch_infer_image_embeds(
            self, batched_input_ids: torch.Tensor,
            multimodal_params: List[MultimodalParams]) -> torch.Tensor:
        # Batch image inputs and attention mask with padding along dim=1 (patch num).
        input_image_embeds_list, input_image_attn_mask_list, input_image_sizes_list = [], [], []
        for mm_param in multimodal_params:
            mm_data = mm_param.multimodal_data
            input_image_embeds = mm_data["input_image_embeds"]
            if input_image_embeds is not None and input_image_embeds.numel(
            ) > 0:
                input_image_embeds_list.append(input_image_embeds)
                input_image_attn_mask_list.append(
                    mm_data["image_attention_mask"])
                input_image_sizes_list.append(mm_data["image_sizes"])
        batched_image_hidden_states = None
        if len(input_image_embeds_list) > 0:
            # Padding image embeds/attn_masks along dim=1 (patch dimension).
            b_list = [x.shape[0] for x in input_image_embeds_list]
            p_list = [x.shape[1] for x in input_image_embeds_list]
            c_i, h_i, w_i = input_image_embeds_list[0].shape[2:5]
            h_i_attn, w_i_attn = input_image_attn_mask_list[0].shape[2:4]
            total_b = sum(b_list)
            max_p = max(p_list)
            batched_image_embeds = torch.zeros(
                (total_b, max_p, c_i, h_i, w_i),
                dtype=input_image_embeds_list[0].dtype,
                device=input_image_embeds_list[0].device)
            batched_image_attn_mask = torch.zeros(
                (total_b, max_p, h_i_attn, w_i_attn),
                dtype=input_image_embeds_list[0].dtype,
                device=input_image_embeds_list[0].device)
            b_offset = 0
            for i, tensor in enumerate(input_image_embeds_list):
                b, p = tensor.shape[:2]
                batched_image_embeds[b_offset:b_offset + b, :p] = tensor
                if input_image_attn_mask_list[i] is not None:
                    batched_image_attn_mask[
                        b_offset:b_offset +
                        b, :p] = input_image_attn_mask_list[i]
                else:
                    batched_image_attn_mask[b_offset:b_offset + b, :p] = 1
                b_offset += b
            batched_image_sizes = torch.cat(input_image_sizes_list, dim=0)
            # Forward image encoder with batched image embeds.
            batched_image_hidden_states = self.image_embed(
                input_ids=batched_input_ids,
                input_embeds=batched_image_embeds,
                image_sizes=batched_image_sizes,
                image_attention_mask=batched_image_attn_mask,
                wte=self.embed_tokens,
            )
        return batched_image_hidden_states

    def _batch_infer_audio_embeds(
            self, batched_input_ids: torch.Tensor,
            multimodal_params: List[MultimodalParams]) -> torch.Tensor:
        # Batch audio inputs and attention mask with padding along dim=1 (patch num).
        input_audio_embeds_list, input_audio_attn_mask_list, input_audio_sizes_list = [], [], []
        for mm_param in multimodal_params:
            mm_data = mm_param.multimodal_data
            input_audio_embeds = mm_data["input_audio_embeds"]
            if input_audio_embeds is not None and input_audio_embeds.numel(
            ) > 0:
                input_audio_embeds_list.append(input_audio_embeds)
                input_audio_attn_mask_list.append(
                    mm_data["audio_attention_mask"])
                input_audio_sizes_list.append(mm_data["audio_embed_sizes"])
        batched_audio_hidden_states = None
        if len(input_audio_embeds_list) > 0:
            b_list = [x.shape[0] for x in input_audio_embeds_list]
            p_list = [x.shape[1] for x in input_audio_embeds_list]
            d_a = input_audio_embeds_list[0].shape[2]
            total_b = sum(b_list)
            max_p = max(p_list)
            batched_audio_embeds = torch.zeros(
                (total_b, max_p, d_a),
                dtype=input_audio_embeds_list[0].dtype,
                device=input_audio_embeds_list[0].device)
            batched_audio_attn_mask = torch.zeros(
                (total_b, max_p),
                dtype=input_audio_embeds_list[0].dtype,
                device=input_audio_embeds_list[0].device)
            b_offset = 0
            for i, tensor in enumerate(input_audio_embeds_list):
                b, p = tensor.shape[:2]
                batched_audio_embeds[b_offset:b_offset + b, :p] = tensor
                if input_audio_attn_mask_list[i] is not None:
                    batched_audio_attn_mask[
                        b_offset:b_offset +
                        b, :p] = input_audio_attn_mask_list[i]
                else:
                    batched_audio_attn_mask[b_offset:b_offset + b, :p] = 1
                b_offset += b
            batched_audio_sizes = torch.cat(input_audio_sizes_list, dim=0)
            # Forward audio encoder with batched audio embeds.
            batched_audio_hidden_states = self.audio_embed(
                input_ids=batched_input_ids,
                input_embeds=batched_audio_embeds,
                audio_embed_sizes=batched_audio_sizes,
                audio_attention_mask=batched_audio_attn_mask,
                wte=self.embed_tokens,
            )
        return batched_audio_hidden_states

    def _encoding_per_request(
            self, multimodal_params: List[MultimodalParams],
            mm_token_ids: torch.Tensor) -> List[torch.FloatTensor]:
        # Loop implementation.
        mm_embeddings = []
        for i in range(len(multimodal_params)):
            input_ids = multimodal_params[i].multimodal_data["input_ids"]
            input_image_embeds = multimodal_params[i].multimodal_data[
                "input_image_embeds"]
            input_audio_embeds = multimodal_params[i].multimodal_data[
                "input_audio_embeds"]
            image_sizes = multimodal_params[i].multimodal_data["image_sizes"]
            image_attention_mask = multimodal_params[i].multimodal_data[
                "image_attention_mask"]
            audio_embed_sizes = multimodal_params[i].multimodal_data[
                "audio_embed_sizes"]
            audio_attention_mask = multimodal_params[i].multimodal_data[
                "audio_attention_mask"]
            audio_projection_mode = multimodal_params[i].multimodal_data[
                "audio_projection_mode"]

            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            input_ids = self._replace_special_token_ids(input_ids)
            image_position_mask = input_ids == _IMAGE_SPECIAL_TOKEN_ID
            non_image_position_mask = ~image_position_mask

            image_hidden_states = None
            if input_image_embeds is not None:
                image_hidden_states = self.image_embed(
                    input_ids=input_ids,
                    input_embeds=input_image_embeds,
                    image_sizes=image_sizes,
                    wte=self.embed_tokens,
                    image_attention_mask=image_attention_mask,
                )
            audio_hidden_states = None
            if input_audio_embeds is not None:
                audio_hidden_states = self.audio_embed(
                    input_ids=input_ids,
                    input_embeds=input_audio_embeds,
                    audio_embed_sizes=audio_embed_sizes,
                    audio_attention_mask=audio_attention_mask,
                    wte=self.embed_tokens,
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
                hidden_states = self.embed_tokens(input_ids)

            # Postprocessing to get multimodal-only embeddings.
            mm_token_mask = torch.isin(input_ids, mm_token_ids)
            hidden_states = hidden_states[mm_token_mask]

            mm_embeddings.append(hidden_states)
        return mm_embeddings

    def _encoding_batch_request(
            self, multimodal_params: List[MultimodalParams],
            mm_token_ids: torch.Tensor) -> List[torch.FloatTensor]:
        # Batch input_ids.
        input_ids_list = [
            multimodal_params[i].multimodal_data["input_ids"]
            for i in range(len(multimodal_params))
        ]
        max_input_ids_len = max(
            [input_ids.shape[1] for input_ids in input_ids_list])
        batched_input_ids = torch.full(
            (len(multimodal_params), max_input_ids_len),
            _PAD_TOKEN_ID,
            device=input_ids_list[0].device)
        for i, input_ids in enumerate(input_ids_list):
            batched_input_ids[i, :input_ids.shape[1]] = input_ids
        batched_input_ids = batched_input_ids.view(-1, max_input_ids_len)
        batched_input_ids = self._replace_special_token_ids(batched_input_ids)
        image_position_mask = batched_input_ids == _IMAGE_SPECIAL_TOKEN_ID
        non_image_position_mask = ~image_position_mask

        # Batch inference for image and audio embeds.
        batched_image_hidden_states = self._batch_infer_image_embeds(
            batched_input_ids, multimodal_params)
        batched_audio_hidden_states = self._batch_infer_audio_embeds(
            batched_input_ids, multimodal_params)

        # Combine different modalities into one.
        if batched_image_hidden_states is not None and batched_audio_hidden_states is not None:
            batched_hidden_states = batched_image_hidden_states * image_position_mask.unsqueeze(
                -1
            ) + batched_audio_hidden_states * non_image_position_mask.unsqueeze(
                -1)
        elif batched_image_hidden_states is not None:
            batched_hidden_states = batched_image_hidden_states
        elif batched_audio_hidden_states is not None:
            batched_hidden_states = batched_audio_hidden_states
        else:
            batched_hidden_states = self.embed_tokens(batched_input_ids)

        # Postprocessing to get multimodal-only embeddings.
        mm_token_mask = torch.isin(batched_input_ids, mm_token_ids)
        batched_hidden_states = batched_hidden_states[mm_token_mask]
        batched_hidden_states = [batched_hidden_states]
        return batched_hidden_states

    @torch.inference_mode()
    def forward(self, multimodal_params: List[MultimodalParams],
                mm_token_ids: torch.Tensor) -> List[torch.FloatTensor]:
        if os.getenv("PHI4_MM_PER_REQUEST_INFER", "0") == "1":
            # Reference code path to check correctness of batch inference and further dev.
            # (TODO) Remove this path after accuracy bench and data parallelism are supported.
            return self._encoding_per_request(multimodal_params, mm_token_ids)
        else:
            # Batch inference as default path.
            return self._encoding_batch_request(multimodal_params, mm_token_ids)


class Phi4MMInputProcessor(BaseMultimodalInputProcessor, InputProcessor):

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

    def get_mm_token_ids(self) -> Optional[torch.Tensor]:
        return torch.tensor([_IMAGE_SPECIAL_TOKEN_ID, _AUDIO_SPECIAL_TOKEN_ID],
                            dtype=torch.int32,
                            device=self.device)

    def get_num_tokens_per_image(
        self,
        *,
        image: Image.Image,
        **kwargs,
    ):
        data = self.processor.image_processor.preprocess(image)
        return data["num_img_tokens"][0]

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
                # HF Phi4MM can only support PIL images. Convert normalized tensors (0-1) to PIL images (0-255).
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

        # Move mm_token_ids to the correct device.
        self.mm_token_ids = torch.tensor(
            [_IMAGE_SPECIAL_TOKEN_ID, _AUDIO_SPECIAL_TOKEN_ID],
            device=self.device)

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
                mm_embedding = self.hf_phi4mm_model(multimodal_params,
                                                    self.mm_token_ids)
            else:
                # Directly fetch the multimodal embedding for DISAGG mode.
                # This path is not functional now. `multimodal_params` will be prepared in PyExecutor.
                mm_embedding = [
                    multimodal_param.multimodal_data["multimodal_embedding"]
                    for multimodal_param in multimodal_params
                ]
            mm_embedding = find_input_mm_embeds(
                mm_embedding, multimodal_params[:num_context_requests])

        input_ids, input_embeds = fuse_input_embeds(
            self.llm.model.embed_tokens,
            input_ids,
            mm_embedding,
            mm_token_ids=self.mm_token_ids,
            **kwargs,
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

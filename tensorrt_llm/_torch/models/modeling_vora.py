# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VoRA (Vision as LoRA) model implementation for PyTorch backend."""

import copy
import dataclasses
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (AutoConfig, AutoImageProcessor, AutoProcessor,
                          PretrainedConfig, PreTrainedModel, Qwen2Config)

from ..._utils import nvtx_range, str_dtype_to_torch
from ...inputs import ExtraProcessedInputs, TextPrompt
from ...inputs.registry import register_input_processor, InputProcessor
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..modules.embedding import Embedding, LMHead
from ..modules.linear import Linear, TensorParallelMode
from ..modules.rms_norm import RMSNorm
from .modeling_qwen import QwenAttention, QwenDecoderLayer, QwenModel
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             ModelConfig, register_auto_model,
                             _load_weights_impl)
from .modeling_multimodal_utils import fuse_input_embeds


# PureRMSNorm removed - using standard RMSNorm with bfloat16 support




class VoRAConfig(Qwen2Config):
    """Configuration for VoRA models."""
    model_type = "vora"
    
    # VoRA specific parameters
    vision_embedding_type: str = "AIMv2"
    patch_size: int = 14
    image_size: int = 448
    vision_attention_mask: str = "bidirectional"
    vision_embedding_intermediate_size: int = 1024
    layer_types: Optional[List[str]] = None
    lora: Optional[Dict] = None
    
    def __init__(self, **kwargs):
        # Set default layer_types if not provided
        if 'layer_types' not in kwargs:
            num_layers = kwargs.get('num_hidden_layers', 28)
            kwargs['layer_types'] = ["full_attention"] * num_layers
        
        # Force model_type for input processor registration
        kwargs['model_type'] = 'vora'
        
        super().__init__(**kwargs)
        
        # Set VoRA specific attributes
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class VoRAVisionEmbedding(nn.Module):
    """Vision embedding layer for VoRA."""
    
    def __init__(self, config: VoRAConfig, hidden_size: int):
        super().__init__()
        self.patch_size = config.patch_size
        self.hidden_size = hidden_size
        # Use intermediate size from config (1536 for VoRA)
        self.embed_dim = config.vision_embedding_intermediate_size
        
        # Get dtype from config (fundamental solution)
        dtype = torch.bfloat16  # Default
        if hasattr(config, 'torch_dtype'):
            if isinstance(config.torch_dtype, str):
                from tensorrt_llm.models.convert_utils import str_dtype_to_torch
                dtype = str_dtype_to_torch(config.torch_dtype)
            elif isinstance(config.torch_dtype, torch.dtype):
                dtype = config.torch_dtype
        
        # Patchifier: Conv2d to extract patches with correct dtype
        self.patchifier_proj = nn.Conv2d(
            3,
            self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size),
            bias=True,
            dtype=dtype
        )
        # Use standard RMSNorm (bfloat16 is supported)
        self.patchifier_norm = RMSNorm(hidden_size=self.embed_dim, eps=1e-05, dtype=dtype)
        
        # Output projection: embed_dim -> hidden_size with correct dtype
        self.out_proj = nn.Linear(self.embed_dim, hidden_size, bias=False, dtype=dtype)
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Convert to model dtype
        target_dtype = self.patchifier_proj.weight.dtype
        pixel_values = pixel_values.to(target_dtype)
        
        _, _, H, W = pixel_values.shape
        # Step 1: Patchify and normalize
        tokens = self.patchifier_proj(pixel_values).flatten(2).transpose(1, 2)
        # Ensure tensor is contiguous for flashinfer operations
        tokens = tokens.contiguous()
        
        # Apply RMSNorm (flashinfer requires 2D tensor)
        batch_size, seq_len, hidden_dim = tokens.shape
        tokens_2d = tokens.view(-1, hidden_dim)
        tokens_2d = self.patchifier_norm(tokens_2d)
        tokens = tokens_2d.view(batch_size, seq_len, hidden_dim)
        
        # Step 2: Add position embeddings
        pos_embed = self._get_sincos_pos_embed(
            H // self.patch_size, W // self.patch_size, 
            embed_dim=self.embed_dim, device=tokens.device
        )
        # Ensure pos_embed has same dtype as tokens
        tokens = tokens + pos_embed.to(tokens.dtype)
        
        # Step 3: Project to hidden size
        tokens = self.out_proj(tokens)
        # Ensure output is in correct dtype
        tokens = tokens.to(target_dtype)
        return tokens
    
    def _get_sincos_pos_embed(self, h: int, w: int, embed_dim: int, device: torch.device) -> torch.Tensor:
        """Generate sinusoidal position embeddings."""
        assert embed_dim % 2 == 0, embed_dim
        grid_h = torch.arange(h).float().to(device)
        grid_w = torch.arange(w).float().to(device)
        grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
        grid = torch.stack(grid, dim=0).to(device)
        grid = grid.reshape([2, 1, h, w])
        
        # Get 1D embeddings
        emb_h = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0], device)
        emb_w = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1], device)
        
        pos_embed = torch.cat([emb_h, emb_w], dim=1)  # (H * W, D)
        return pos_embed
    
    def _get_1d_sincos_pos_embed_from_grid(self, embed_dim: int, pos: torch.Tensor, device: torch.device) -> torch.Tensor:
        omega = torch.arange(embed_dim // 2).float().to(device)
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega  # (D / 2,)
        pos = pos.reshape(-1)  # (M,)
        out = pos[:, None] * omega[None, :]  # (M, D / 2), outer product
        emb_sin, emb_cos = torch.sin(out).to(device), torch.cos(out).to(device)  # (M, D / 2)
        emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
        return emb


class VoRAAttention(QwenAttention):
    """VoRA attention with hybrid bidirectional/causal masking."""
    
    def __init__(self, model_config: ModelConfig[VoRAConfig], layer_idx: Optional[int] = None):
        super().__init__(model_config, layer_idx)
    
    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        mrope_config: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        # For now, use parent class implementation
        # In production, this would handle hybrid attention masking
        return super().forward(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            mrope_config=mrope_config,
            **kwargs
        )


class VoRADecoderLayer(QwenDecoderLayer):
    """VoRA decoder layer with potential for hybrid attention."""
    
    def __init__(self, model_config: ModelConfig[VoRAConfig], layer_idx: int):
        super().__init__(model_config, layer_idx)
        # Replace attention with VoRA attention
        self.self_attn = VoRAAttention(model_config, layer_idx)
        
        # Get dtype from config
        config = model_config.pretrained_config
        dtype = torch.bfloat16  # Default
        if hasattr(config, 'torch_dtype'):
            if isinstance(config.torch_dtype, str):
                from tensorrt_llm.models.convert_utils import str_dtype_to_torch
                dtype = str_dtype_to_torch(config.torch_dtype)
            elif isinstance(config.torch_dtype, torch.dtype):
                dtype = config.torch_dtype
        
        # Use standard RMSNorm (bfloat16 is supported)
        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        


class VoRAModel(QwenModel):
    """VoRA model implementation."""
    
    def __init__(self, model_config: ModelConfig[VoRAConfig]):
        # Initialize parent
        super().__init__(model_config)
        
        # Initialize vision embedding
        self.vision_embedding = VoRAVisionEmbedding(
            model_config.pretrained_config,
            model_config.pretrained_config.hidden_size
        )
        
        # Fundamental solution: Convert entire model to config dtype
        config = model_config.pretrained_config
        if hasattr(config, 'torch_dtype'):
            if isinstance(config.torch_dtype, str):
                from tensorrt_llm.models.convert_utils import str_dtype_to_torch
                target_dtype = str_dtype_to_torch(config.torch_dtype)
            elif isinstance(config.torch_dtype, torch.dtype):
                target_dtype = config.torch_dtype
            else:
                target_dtype = torch.bfloat16
            
            # Convert entire model hierarchy to target dtype
            self.to(target_dtype)
            
            # Explicitly convert all submodules
            for name, module in self.named_modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    module.weight.data = module.weight.data.to(target_dtype)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data = module.bias.data.to(target_dtype)
                    
            # CRITICAL: Ensure Conv2d bias is converted
            if hasattr(self.vision_embedding, 'patchifier_proj'):
                conv = self.vision_embedding.patchifier_proj
                if conv.bias is not None and conv.bias.dtype != target_dtype:
                    conv.bias.data = conv.bias.data.to(target_dtype)
        
        # Replace layers with VoRA layers
        self.layers = nn.ModuleList([
            VoRADecoderLayer(model_config, layer_idx)
            for layer_idx in range(model_config.pretrained_config.num_hidden_layers)
        ])
        
        # Get dtype from config
        config = model_config.pretrained_config
        dtype = torch.bfloat16  # Default
        if hasattr(config, 'torch_dtype'):
            if isinstance(config.torch_dtype, str):
                from tensorrt_llm.models.convert_utils import str_dtype_to_torch
                dtype = str_dtype_to_torch(config.torch_dtype)
            elif isinstance(config.torch_dtype, torch.dtype):
                dtype = config.torch_dtype
                
        # Use standard RMSNorm (bfloat16 is supported)
        self.norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        
    
    def forward(
        self,
        attn_metadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs
    ):
        """Forward method - multimodal processing now handled by VoRAForCausalLM."""
        # Call parent forward (QwenModel)
        return super().forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs
        )


class VoRAInputProcessor(InputProcessor):
    """Input processor for VoRA models."""
    
    def __init__(self, 
                 model_path: str, 
                 model_config: AutoConfig, 
                 tokenizer, 
                 trust_remote_code: bool = True):
        # Store standard parameters
        self.model_path = model_path
        self.model_config = model_config
        self.config = model_config  # Alias for compatibility
        
        # Load processor from HuggingFace
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code
        )
        
        # Handle tokenizer safely
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            # Load tokenizer separately if not provided
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code
            )
        
        self.vision_placeholder_token = -200  # VoRA's special token
    
    def get_num_tokens_per_image(self, **kwargs) -> int:
        """Get number of tokens per image for multimodal hashing."""
        # VoRA calculates tokens based on image size and patch size
        image_size = getattr(self.config, 'image_size', 448)
        patch_size = getattr(self.config, 'patch_size', 14)
        
        # Calculate number of patches per dimension
        patches_per_dim = image_size // patch_size
        
        # Total number of vision tokens
        num_tokens = patches_per_dim * patches_per_dim
        
        logger.info(f"VoRA calculated {num_tokens} tokens per image (image_size={image_size}, patch_size={patch_size})")
        return num_tokens
    
    def __call__(
        self, 
        inputs, 
        sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        """Process inputs according to InputProcessor protocol."""
        # Handle both TextPrompt objects and dict inputs
        if hasattr(inputs, 'prompt'):
            # TextPrompt object
            prompt = inputs.prompt
            multi_modal_data = inputs.multi_modal_data
        else:
            # Dict input (fallback case)
            prompt = inputs.get('prompt', '')
            multi_modal_data = inputs.get('multi_modal_data', None)
        
        # Extract images from multimodal data
        images = None
        if multi_modal_data and "image" in multi_modal_data:
            images = multi_modal_data["image"]
            logger.info(f"VoRAInputProcessor: Received image type: {type(images)}")
            if not isinstance(images, list):
                images = [images]
            
            # Convert images to PIL format if needed (VoRA custom logic preserved)
            from PIL import Image
            processed_images = []
            for i, img in enumerate(images):
                logger.info(f"VoRAInputProcessor: Processing image {i} of type: {type(img)}")
                if isinstance(img, str):
                    # Load from path
                    processed_images.append(Image.open(img))
                elif hasattr(img, 'convert'):  # Already PIL Image
                    processed_images.append(img)
                else:
                    # Try to convert to PIL
                    logger.warning(f"Unknown image type: {type(img)}, attempting conversion")
                    try:
                        processed_images.append(Image.fromarray(img))
                    except:
                        logger.error(f"Failed to convert image of type {type(img)}")
                        processed_images.append(img)
            images = processed_images
        
        # Tokenize the prompt if we have images, otherwise use simple tokenization
        if images:
            # Create conversation format for VoRA (custom logic preserved)
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": images[0]},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            # Apply chat template (VoRA custom logic preserved)
            model_inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors='pt',
                return_dict=True
            )
            
            input_ids = model_inputs['input_ids'][0].tolist()
            logger.info(f"VoRAInputProcessor: apply_chat_template result - input_ids[:20]: {input_ids[:20]}")
            logger.info(f"VoRAInputProcessor: Found -200 in input_ids? {-200 in input_ids}")
            
            # TensorRT-LLM standard: return multimodal data in extra_processed_inputs format
            extra_processed_inputs = None
            if 'frames' in model_inputs:
                # Find positions of -200 tokens
                placeholder_positions = [i for i, token_id in enumerate(input_ids) if token_id == -200]
                logger.info(f"VoRAInputProcessor: Found {len(placeholder_positions)} placeholder positions: {placeholder_positions}")
                
                extra_processed_inputs = {
                    "multimodal_data": {
                        "image": {
                            "frames": model_inputs['frames'],
                            "pixel_values": model_inputs['frames'],  # Alias for compatibility
                            "placeholder_positions": placeholder_positions,
                            "vision_placeholder_index": -200
                        }
                    }
                }
                logger.info(f"VoRAInputProcessor: Created multimodal_data with frames shape {model_inputs['frames'].shape}")
            
            logger.info(f"VoRAInputProcessor: Returning extra_processed_inputs keys: {list(extra_processed_inputs.keys()) if extra_processed_inputs else None}")
            return input_ids, extra_processed_inputs
        else:
            # Text-only tokenization
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            return input_ids, None
    
    def get_extra_inputs(
        self,
        prompt: Union[str, List[str]],
        images: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        **kwargs
    ) -> ExtraProcessedInputs:
        """Process inputs for VoRA model."""
        # Handle batch or single inputs
        if isinstance(prompt, str):
            prompts = [prompt]
            images = [images] if images is not None else None
        else:
            prompts = prompt
        
        # Process with VoRA processor
        if images is not None:
            # Ensure images are in correct format
            from PIL import Image
            processed_images = []
            
            # Handle case where images might be nested lists
            flat_images = []
            for img in images:
                if isinstance(img, list):
                    # Flatten nested list
                    flat_images.extend(img)
                else:
                    flat_images.append(img)
            
            for img in flat_images:
                if isinstance(img, str):
                    # Load from path
                    processed_images.append(Image.open(img))
                elif hasattr(img, 'convert'):  # Already PIL Image
                    processed_images.append(img)
                else:
                    # Try to convert to PIL
                    logger.warning(f"get_extra_inputs: Unknown image type: {type(img)}, attempting conversion")
                    try:
                        processed_images.append(Image.fromarray(img))
                    except:
                        logger.error(f"get_extra_inputs: Failed to convert image of type {type(img)}")
                        # Skip this image
                        continue
            
            if not processed_images:
                logger.warning("get_extra_inputs: No valid images after processing")
                images = None
            else:
                images = processed_images
        
        # Use VoRA's original processing method
        if images is not None:
            # Ensure prompts include <image> token
            processed_prompts = []
            for prompt in prompts:
                if '<image>' not in prompt:
                    # Add <image> at the beginning if not present
                    processed_prompts.append('<image> ' + prompt)
                else:
                    processed_prompts.append(prompt)
            
            # VoRA expects images in nested list format: [[image1], [image2], ...]
            nested_images = [[img] for img in images]
            
            # Use VoRA's original processor
            model_inputs = self.processor(
                images=nested_images,
                text=processed_prompts
            )
            
            logger.info(f"VoRAInputProcessor: Processed with VoRA method, keys: {list(model_inputs.keys())}")
            if 'input_ids' in model_inputs:
                logger.info(f"VoRAInputProcessor: input_ids shape: {model_inputs['input_ids'].shape}")
                logger.info(f"VoRAInputProcessor: input_ids sample: {model_inputs['input_ids'][0][:10].tolist()}")
        else:
            # Text-only processing
            model_inputs = self.processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
        
        # Extract vision-related metadata
        extra_inputs = {}
        
        if 'frames' in model_inputs:
            extra_inputs['vision_embeddings'] = model_inputs['frames']
            logger.info(f"VoRAInputProcessor: Found vision embeddings with shape {model_inputs['frames'].shape}")
        else:
            logger.warning("VoRAInputProcessor: No 'frames' found in model_inputs")
            logger.info(f"VoRAInputProcessor: Available keys: {list(model_inputs.keys())}")
        
        if 'vision_placeholder_index' in model_inputs:
            extra_inputs['vision_placeholder_positions'] = model_inputs['vision_placeholder_index']
        
        return extra_inputs


@register_input_processor(VoRAInputProcessor, model_type="vora")
@register_auto_model("VoRAForCausalLM")
class VoRAForCausalLM(PreTrainedModel):
    """VoRA multimodal model for causal language modeling."""
    
    config_class = VoRAConfig
    
    def __init__(
        self,
        model_config: ModelConfig[VoRAConfig],
    ):
        config = model_config.pretrained_config
        super().__init__(config)
        
        self.model_config = model_config
        
        # Initialize LLM component (VoRAModel contains the LLM)
        self.model = VoRAModel(model_config)
        
        # Ultimate solution: Ensure ALL model components use config dtype
        if hasattr(config, 'torch_dtype'):
            if isinstance(config.torch_dtype, str):
                from tensorrt_llm.models.convert_utils import str_dtype_to_torch
                target_dtype = str_dtype_to_torch(config.torch_dtype)
            elif isinstance(config.torch_dtype, torch.dtype):
                target_dtype = config.torch_dtype
            else:
                target_dtype = torch.bfloat16
            
            # Convert ENTIRE model including all inherited components
            self.to(target_dtype)
        
        # Apply LoRA if configured
        if config.lora:
            self._apply_lora(config.lora)
    
    def load_weights(self, weights: dict, *args, **kwargs):
        """Load weights into the model."""
        # Delegate to the internal VoRAModel's load_weights method
        self.model.load_weights(weights, *args, **kwargs)
    
    def infer_max_seq_len(self) -> int:
        """Infer max sequence length."""
        return getattr(self.model_config.pretrained_config, 'max_position_embeddings', 32768)
    
    @torch.inference_mode()
    def forward(
        self,
        attn_metadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Forward method for multimodal VLM."""
        num_context_requests, num_generation_requests = attn_metadata.num_contexts, attn_metadata.num_generations
        logger.info(f"VoRAForCausalLM.forward: {num_context_requests=}, {num_generation_requests=}")
        
        multimodal_params = kwargs.get("multimodal_params", [])
        logger.info(f"VoRAForCausalLM.forward: multimodal_params length: {len(multimodal_params)}")
        
        if multimodal_params:
            logger.info(f"VoRAForCausalLM.forward: Processing {len(multimodal_params)} multimodal params")
            # Extract vision features from multimodal_params
            vision_features = []
            
            for mm_param in multimodal_params:
                if hasattr(mm_param, 'multimodal_data') and 'image' in mm_param.multimodal_data:
                    image_data = mm_param.multimodal_data['image']
                    logger.info(f"VoRAForCausalLM.forward: Processing vision data with keys {list(image_data.keys())}")
                    
                    # VoRA expects raw pixel values for vision embedding
                    if 'frames' in image_data:
                        pixel_values = image_data['frames']
                    elif 'pixel_values' in image_data:
                        pixel_values = image_data['pixel_values'] 
                    else:
                        logger.warning(f"VoRAForCausalLM.forward: No frames or pixel_values found in image_data: {list(image_data.keys())}")
                        continue
                        
                    logger.info(f"VoRAForCausalLM.forward: Processing vision embeddings with shape {pixel_values.shape}")
                    # VoRA custom vision embedding processing (preserved)
                    vision_embeds = self.model.vision_embedding(pixel_values)
                    vision_features.append(vision_embeds)
                    logger.info(f"VoRAForCausalLM.forward: Vision embeds processed to shape {vision_embeds.shape}")
            
            if vision_features:
                # VoRA special handling: expand single placeholder token to multiple vision tokens
                vision_embed = vision_features[0]  # [1024, 3584]
                batch_size, num_vision_tokens, hidden_dim = vision_embed.shape
                
                # Find VoRA placeholder tokens (-200) in input_ids
                placeholder_mask = (input_ids == -200)
                placeholder_indices = torch.where(placeholder_mask)[0]
                
                if len(placeholder_indices) > 0:
                    # Create expanded input_ids to accommodate all vision tokens
                    # Each placeholder (-200) gets replaced by num_vision_tokens tokens
                    expanded_length = input_ids.shape[0] - len(placeholder_indices) + len(placeholder_indices) * num_vision_tokens
                    expanded_input_ids = torch.full((expanded_length,), -200, device=input_ids.device, dtype=input_ids.dtype)
                    
                    # Create expanded embeddings tensor - ensure dtype consistency
                    text_embed = self.model.embed_tokens(input_ids[~placeholder_mask])
                    # Make sure vision and text embeddings have the same dtype
                    if text_embed.dtype != vision_embed.dtype:
                        vision_embed = vision_embed.to(text_embed.dtype)
                    
                    expanded_embeds = torch.zeros(expanded_length, hidden_dim, device=vision_embed.device, dtype=text_embed.dtype)
                    
                    # Fill in text embeddings and vision embeddings
                    expanded_pos = 0
                    text_pos = 0
                    
                    for i, token_id in enumerate(input_ids):
                        if token_id == -200:
                            # Insert all vision tokens for this placeholder
                            expanded_embeds[expanded_pos:expanded_pos + num_vision_tokens] = vision_embed[0]  # [1024, 3584]
                            expanded_pos += num_vision_tokens
                        else:
                            # Insert text embedding
                            expanded_embeds[expanded_pos] = text_embed[text_pos]
                            expanded_pos += 1
                            text_pos += 1
                    
                    input_ids = None
                    inputs_embeds = expanded_embeds
                    logger.info(f"VoRAForCausalLM.forward: Expanded {len(placeholder_indices)} placeholders to {num_vision_tokens} vision tokens each")
                    logger.info(f"VoRAForCausalLM.forward: Final inputs_embeds shape: {inputs_embeds.shape}")
                else:
                    logger.warning("VoRAForCausalLM.forward: No vision placeholder tokens found in input_ids")
        
        # Call the underlying model (simplified - no multimodal handling in VoRAModel.forward now)
        return self.model.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **{k: v for k, v in kwargs.items() if k != 'multimodal_params'}
        )
    
    def _apply_lora(self, lora_config: Dict):
        """Apply LoRA configuration to the model."""
        # This would implement LoRA application
        # For now, just log
        logger.info(f"LoRA configuration: {lora_config}")
    
    def load_weights(self, weights: dict, strict: bool = False):
        """Load weights from a dictionary."""
        # Map HuggingFace parameter names to TRT-LLM names
        remapped_weights = {}
        
        for name, weight in weights.items():
            new_name = name
            
            # Handle vision embedding parameters
            if name.startswith('vision_embedding.'):
                # Map to our vision embedding structure
                if 'patchifier.proj' in name:
                    # vision_embedding.patchifier.proj.weight -> model.vision_embedding.patchifier_proj.weight
                    new_name = name.replace('patchifier.proj', 'patchifier_proj')
                elif 'patchifier.norm' in name:
                    # vision_embedding.patchifier.norm.weight -> model.vision_embedding.patchifier_norm.weight  
                    new_name = name.replace('patchifier.norm', 'patchifier_norm')
                elif 'pos_embed' in name:
                    # Skip pos_embed as it's computed dynamically
                    logger.info(f"Skipping {name} (computed dynamically)")
                    continue
                elif 'out_proj' in name:
                    # vision_embedding.out_proj.weight -> model.vision_embedding.out_proj.weight
                    new_name = name  # Keep as is
                new_name = 'model.' + new_name
            
            # Handle LLM parameters
            elif name.startswith('llm.'):
                # Remove 'llm.' prefix
                new_name = name[4:]
            
            remapped_weights[new_name] = weight
        
        # Load weights using the common implementation
        _load_weights_impl(
            self,
            remapped_weights
        )
        
        logger.info(f"Loaded {len(remapped_weights)} weights from {len(weights)} total")
    
    @classmethod
    def from_hugging_face(
        cls,
        hf_model_dir: str,
        **kwargs
    ):
        """Load VoRA model from HuggingFace checkpoint."""
        # Load configuration
        config = AutoConfig.from_pretrained(
            hf_model_dir,
            trust_remote_code=True
        )
        
        # Ensure it's a VoRA config
        if not isinstance(config, VoRAConfig):
            # Convert to VoRA config
            config_dict = config.to_dict()
            vora_config = VoRAConfig(**config_dict)
        else:
            vora_config = config
            
        # Convert torch_dtype from string to torch.dtype if needed
        if hasattr(vora_config, 'torch_dtype') and isinstance(vora_config.torch_dtype, str):
            vora_config.torch_dtype = str_dtype_to_torch(vora_config.torch_dtype)
        
        # Create model config
        model_config = ModelConfig(
            pretrained_config=vora_config,
            quant_config=kwargs.get('quant_config', None)
        )
        
        # Initialize model
        model = cls(model_config)
        
        # Load weights from checkpoint
        import os
        from pathlib import Path
        from safetensors.torch import load_file
        from ...llmapi.utils import download_hf_model
        
        # Handle HuggingFace model ID or local path
        model_path = hf_model_dir
        if not os.path.isdir(model_path):
            # Download from HuggingFace Hub
            logger.info(f"Downloading model from HuggingFace Hub: {model_path}")
            model_path = str(download_hf_model(model_path))
            logger.info(f"Model downloaded to: {model_path}")
        
        weights = {}
        safetensors_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
        
        if not safetensors_files:
            # Try loading from pytorch files
            pt_files = [f for f in os.listdir(model_path) if f.endswith('.bin')]
            if pt_files:
                logger.info("Loading from PyTorch checkpoint files")
                for filename in sorted(pt_files):
                    filepath = os.path.join(model_path, filename)
                    logger.info(f"Loading weights from {filename}")
                    file_weights = torch.load(filepath, map_location='cpu')
                    weights.update(file_weights)
            else:
                raise ValueError(f"No safetensors or .bin files found in {model_path}")
        else:
            for filename in sorted(safetensors_files):
                filepath = os.path.join(hf_model_dir, filename)
                logger.info(f"Loading weights from {filename}")
                file_weights = load_file(filepath)
                weights.update(file_weights)
        
        # Use TensorRT-LLM standard dtype inference
        from tensorrt_llm.models.convert_utils import infer_dtype, str_dtype_to_torch
        
        # Standard TRT-LLM pattern for dtype handling
        inferred_dtype_str = infer_dtype(
            kwargs.get('dtype', 'auto'), 
            getattr(vora_config, 'torch_dtype', None)
        )
        target_dtype = str_dtype_to_torch(inferred_dtype_str)
        logger.info(f"Inferred dtype: {inferred_dtype_str} ({target_dtype})")
        
        # Convert weights to target dtype
        logger.info(f"Converting all weights to {target_dtype}")
        weights = {k: v.to(target_dtype) for k, v in weights.items()}
        
        # Load weights into model
        model.load_weights(weights)
        
        # Convert entire model to target dtype (TensorRT-LLM multimodal standard)
        model = model.to(target_dtype)
        logger.info(f"Model converted to {target_dtype}")
        
        # Move model to device if specified
        device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        if device != 'cpu':
            model = model.to(device)
            logger.info(f"Model moved to {device}")
        
        logger.info(f"Successfully loaded VoRA model from {hf_model_dir}")
        
        return model
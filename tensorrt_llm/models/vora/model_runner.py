# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.multimodal_model_runner import MultimodalModelRunner


class VoRAModelRunner:
    """PyTorch-based runner for VoRA models.
    
    This runner loads and executes VoRA models directly using PyTorch,
    without converting to TensorRT engines.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        trust_remote_code: bool = True,
        max_batch_size: int = 1,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        
        logger.info(f"Loading VoRA model from {model_path}")
        
        # Load processor (handles both image and text)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code
        )
        
        # Load config and add layer_types if missing
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if not hasattr(config, 'layer_types'):
            config.layer_types = ["full_attention"] * getattr(config, 'num_hidden_layers', 28)
            logger.info(f"Added layer_types to config: {len(config.layer_types)} layers")
        
        # Load model with custom VoRA architecture
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            device_map=device
        )
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Get model configuration
        self.config = self.model.config
        
        # Vision configuration
        self.patch_size = getattr(self.config, 'patch_size', 14)
        self.image_size = getattr(self.config, 'image_size', 448)
        self.vision_attention_mask = getattr(self.config, 'vision_attention_mask', 'bidirectional')
        
        logger.info(f"VoRA model loaded successfully")
        logger.info(f"Vision config: patch_size={self.patch_size}, image_size={self.image_size}")
        logger.info(f"Vision attention mask: {self.vision_attention_mask}")
    
    def preprocess_images(
        self,
        images: Union[List[Image.Image], List[str]],
    ) -> torch.Tensor:
        """Preprocess images for VoRA model."""
        processed_images = []
        
        for img in images:
            if isinstance(img, str):
                # Load image from path or URL
                if img.startswith('http'):
                    import requests
                    from io import BytesIO
                    response = requests.get(img)
                    img = Image.open(BytesIO(response.content))
                else:
                    img = Image.open(img)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            processed_images.append(img)
        
        return processed_images
    
    def create_attention_mask_for_vora(
        self,
        input_ids: torch.Tensor,
        vision_token_positions: List[List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create hybrid attention mask for VoRA.
        
        Returns:
            attention_mask: Standard attention mask
            vision_token_mask: Binary mask indicating vision token positions
        """
        batch_size, seq_len = input_ids.shape
        
        # Create vision token mask
        vision_token_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        for batch_idx, positions in enumerate(vision_token_positions):
            if positions:
                start_pos, end_pos = positions[0], positions[-1] + 1
                vision_token_mask[batch_idx, start_pos:end_pos] = True
        
        # Standard attention mask (all ones for valid tokens)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        
        return attention_mask.to(self.device), vision_token_mask.to(self.device)
    
    @torch.no_grad()
    def generate(
        self,
        prompts: Union[str, List[str]],
        images: Optional[Union[Image.Image, List[Image.Image], str, List[str]]] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> List[str]:
        """Generate text from prompts and optional images.
        
        Args:
            prompts: Text prompts
            images: Optional images (can be PIL Images or paths/URLs)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters
        
        Returns:
            List of generated texts
        """
        # Handle single prompt/image
        if isinstance(prompts, str):
            prompts = [prompts]
        if images is not None and not isinstance(images, list):
            images = [images]
        
        batch_size = len(prompts)
        if batch_size > self.max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds maximum {self.max_batch_size}")
        
        # Prepare inputs
        if images:
            # Preprocess images
            processed_images = self.preprocess_images(images)
            
            # Create conversations with images
            conversations = []
            for i, (prompt, image) in enumerate(zip(prompts, processed_images)):
                conversation = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"<image> {prompt}"}
                    ]
                }]
                conversations.append(conversation)
            
            # Apply chat template with images directly (following VoRA's original approach)
            # This should handle both text and image processing
            if len(conversations) == 1:
                model_inputs = self.processor.apply_chat_template(
                    conversations[0], 
                    add_generation_prompt=True, 
                    tokenize=True, 
                    return_tensors='pt', 
                    return_dict=True
                )
            else:
                # TODO: Handle batch processing
                raise NotImplementedError("Batch multimodal processing not yet implemented")
        else:
            # Text-only processing
            conversations = []
            for prompt in prompts:
                conversation = [{
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }]
                conversations.append(conversation)
            
            texts = [
                self.processor.apply_chat_template(conv, add_generation_prompt=True)
                for conv in conversations
            ]
            
            model_inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
        
        # Move inputs to device
        model_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in model_inputs.items()}
        
        # Generate
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
            **kwargs
        }
        
        # VoRA expects a batch dictionary
        # The processor already returns the correct format!
        batch = model_inputs
        
        # Add keys that VoRA's original model expects for debugging
        # These are used in assert error messages, not for actual computation
        if 'prompt' not in batch:
            # For single prompt or batch processing
            batch['prompt'] = prompts if isinstance(prompts, list) else [prompts]
        if 'gt' not in batch:
            # Empty ground truth for inference (not training)
            batch['gt'] = [''] * (len(batch['prompt']) if isinstance(batch['prompt'], list) else 1)
        
        # Debug: print batch keys and values
        logger.info(f"Batch keys: {batch.keys()}")
        if 'vision_placeholder_index' in batch:
            logger.info(f"vision_placeholder_index: {batch['vision_placeholder_index']}")
        if 'frames' in batch:
            logger.info(f"frames shape: {batch['frames'].shape if hasattr(batch['frames'], 'shape') else 'N/A'}")
            logger.info(f"frames dtype: {batch['frames'].dtype if hasattr(batch['frames'], 'dtype') else 'N/A'}")
        if 'n_frames' in batch:
            logger.info(f"n_frames: {batch['n_frames']}")
        if 'input_ids' in batch:
            logger.info(f"input_ids shape: {batch['input_ids'].shape}")
            logger.info(f"input_ids unique values: {torch.unique(batch['input_ids'])[:20]}")  # First 20 unique values
            logger.info(f"Contains -200?: {-200 in batch['input_ids']}")
            # Count -200 occurrences
            logger.info(f"Number of -200 tokens: {(batch['input_ids'] == -200).sum().item()}")
        
        # VoRA requires special handling of 4D attention masks during generation
        # We need to restore the vision_placeholder_index and call VoRA's generate method
        
        # Extract the essential inputs from batch
        vision_placeholder_index = batch.pop("vision_placeholder_index", None)
        frames = batch.get("frames", None)
        n_frames = batch.get("n_frames", None)
        
        # Debug: check if VoRA's generate method exists
        logger.info(f"VoRA model has generate method: {hasattr(self.model, 'generate')}")
        logger.info(f"vision_placeholder_index: {vision_placeholder_index}")
        
        # Use VoRA's custom generation method if available
        if hasattr(self.model, 'generate') and vision_placeholder_index is not None:
            # Restore vision_placeholder_index for VoRA's generate method
            batch["vision_placeholder_index"] = vision_placeholder_index
            
            # Apply VoRA's custom attention implementation
            try:
                # Verify VoRA's generation utilities are properly loaded
                if hasattr(self.model.llm, 'prepare_inputs_for_generation'):
                    logger.info("VoRA generation utilities are loaded")
                    
                    # Replace attention modules with VoRA-specific implementation
                    from .vora_attention import replace_attention_with_vora
                    
                    logger.info("Replacing attention modules with VoRAAttention")
                    self.model.llm = replace_attention_with_vora(self.model.llm)
                    
                    # Call VoRA's generate method with custom attention
                    outputs = self.model.generate(
                        batch,
                        **generation_config
                    )
                    
                else:
                    logger.warning("VoRA generation utilities not found, falling back to standard generation")
                    # Standard generation for text-only or fallback
                    outputs = self.model.llm.generate(
                        **batch,
                        **generation_config
                    )
            except Exception as e:
                logger.error(f"VoRA generation failed: {e}")
                logger.error(f"Error type: {type(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                # For debugging: let's raise the original error to understand the problem better
                raise e
        else:
            # Standard generation for text-only or fallback
            outputs = self.model.llm.generate(
                **batch,
                **generation_config
            )
        
        # Decode outputs
        generated_texts = self.processor.tokenizer.batch_decode(
            outputs[:, model_inputs['input_ids'].shape[1]:],  # Skip input tokens
            skip_special_tokens=True
        )
        
        return generated_texts
    
    def process_batch(
        self,
        batch: Dict[str, Union[List[str], List[Image.Image]]],
        **generation_kwargs
    ) -> List[str]:
        """Process a batch of multimodal inputs.
        
        Args:
            batch: Dictionary with 'prompts' and optional 'images' keys
            **generation_kwargs: Generation parameters
        
        Returns:
            List of generated texts
        """
        prompts = batch['prompts']
        images = batch.get('images', None)
        
        return self.generate(
            prompts=prompts,
            images=images,
            **generation_kwargs
        )


def create_vora_runner(
    model_path: str,
    device: str = "cuda",
    dtype: str = "float16",
    **kwargs
) -> VoRAModelRunner:
    """Factory function to create VoRA model runner.
    
    Args:
        model_path: Path to VoRA model
        device: Device to run on
        dtype: Model data type
        **kwargs: Additional arguments for VoRAModelRunner
    
    Returns:
        VoRAModelRunner instance
    """
    # Convert string dtype to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)
    
    return VoRAModelRunner(
        model_path=model_path,
        device=device,
        dtype=torch_dtype,
        **kwargs
    )
# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration of VoRA with TensorRT-LLM's high-level LLM API."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoConfig

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import SamplingParams
from tensorrt_llm.logger import logger

from .model_runner import VoRAModelRunner


class VoRALLM:
    """High-level API for VoRA models using PyTorch backend.
    
    This class provides a compatible interface with TensorRT-LLM's LLM class
    while using PyTorch for execution.
    """
    
    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tensor_parallel_size: int = 1,
        dtype: str = "float16",
        trust_remote_code: bool = True,
        **kwargs
    ):
        """Initialize VoRA LLM.
        
        Args:
            model: Path to VoRA model or HuggingFace model ID
            tokenizer: Path to tokenizer (defaults to model path)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            dtype: Model data type
            trust_remote_code: Whether to trust remote code
            **kwargs: Additional arguments
        """
        self.model_path = model
        self.tokenizer_path = tokenizer or model
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        
        # Check if model is VoRA
        config = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
        if getattr(config, 'model_type', None) != 'vora':
            raise ValueError(f"Model type {config.model_type} is not VoRA")
        
        # Create PyTorch runner
        self.runner = VoRAModelRunner(
            model_path=model,
            device="cuda",
            dtype=self._get_torch_dtype(dtype),
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        
        logger.info(f"VoRA LLM initialized with PyTorch backend")
    
    def _get_torch_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(dtype_str, torch.float16)
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
        images: Optional[Union[Any, List[Any]]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate text from prompts with optional images.
        
        Args:
            prompts: Input prompts
            sampling_params: Sampling parameters
            images: Optional images
            **kwargs: Additional generation parameters
        
        Returns:
            List of generation results
        """
        if sampling_params is None:
            sampling_params = SamplingParams()
        
        # Convert sampling params to generation kwargs
        gen_kwargs = {
            "max_new_tokens": sampling_params.max_tokens or 1024,
            "temperature": sampling_params.temperature or 1.0,
            "top_p": sampling_params.top_p or 1.0,
            "do_sample": sampling_params.temperature > 0,
        }
        
        # Add any additional kwargs
        gen_kwargs.update(kwargs)
        
        # Generate
        outputs = self.runner.generate(
            prompts=prompts,
            images=images,
            **gen_kwargs
        )
        
        # Format outputs to match LLM API
        results = []
        if isinstance(prompts, str):
            prompts = [prompts]
            
        for prompt, output in zip(prompts, outputs):
            results.append({
                "prompt": prompt,
                "generated_text": output,
                "text": output,  # For compatibility
            })
        
        return results
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        sampling_params: Optional[SamplingParams] = None,
        **kwargs
    ) -> str:
        """Chat interface for VoRA.
        
        Args:
            messages: List of message dictionaries
            sampling_params: Sampling parameters
            **kwargs: Additional parameters
        
        Returns:
            Generated response
        """
        # Extract images and text from messages
        images = []
        prompts = []
        
        for message in messages:
            if message["role"] == "user":
                content = message["content"]
                if isinstance(content, list):
                    # Multimodal content
                    text_parts = []
                    for item in content:
                        if item["type"] == "text":
                            text_parts.append(item["text"])
                        elif item["type"] == "image":
                            if "url" in item:
                                images.append(item["url"])
                            elif "image" in item:
                                images.append(item["image"])
                    prompts.append(" ".join(text_parts))
                else:
                    # Text-only content
                    prompts.append(content)
        
        # Generate response
        results = self.generate(
            prompts=prompts[-1],  # Use last user message
            images=images[-1] if images else None,
            sampling_params=sampling_params,
            **kwargs
        )
        
        return results[0]["generated_text"]


def register_vora_in_llm_api():
    """Register VoRA model support in TensorRT-LLM's LLM API.
    
    This allows using VoRA models with the standard LLM interface.
    """
    # Add VoRA to supported models
    if hasattr(LLM, '_MODEL_CLASSES'):
        LLM._MODEL_CLASSES['vora'] = VoRALLM
    
    # Register model type mapping
    if hasattr(LLM, '_get_model_type'):
        original_get_model_type = LLM._get_model_type
        
        def _get_model_type_with_vora(model_path: str, trust_remote_code: bool = True):
            try:
                config = AutoConfig.from_pretrained(
                    model_path, 
                    trust_remote_code=trust_remote_code
                )
                if getattr(config, 'model_type', None) == 'vora':
                    return 'vora'
            except:
                pass
            return original_get_model_type(model_path, trust_remote_code)
        
        LLM._get_model_type = staticmethod(_get_model_type_with_vora)
    
    logger.info("VoRA model support registered in LLM API")


# Auto-register when module is imported
register_vora_in_llm_api()
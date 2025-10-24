# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for VoRA models."""

from typing import Dict, List, Optional

from transformers import Qwen2Config


class VoRAConfig(Qwen2Config):
    """Configuration for VoRA (Vision as LoRA) models.
    
    VoRA is a multimodal model based on Qwen2 architecture with vision capabilities.
    """
    
    model_type = "vora"
    model_architecture = "VoRAForCausalLM"
    
    def __init__(
        self,
        vision_embedding_type: str = "AIMv2",
        patch_size: int = 14,
        image_size: int = 448,
        vision_attention_mask: str = "bidirectional",
        vision_embedding_intermediate_size: int = 1024,
        layer_types: Optional[List[str]] = None,
        lora: Optional[Dict] = None,
        **kwargs
    ):
        """Initialize VoRA configuration.
        
        Args:
            vision_embedding_type: Type of vision embedding to use
            patch_size: Size of image patches
            image_size: Input image size
            vision_attention_mask: Type of attention mask for vision tokens
            vision_embedding_intermediate_size: Hidden size for vision embeddings
            layer_types: Types of attention layers
            lora: LoRA configuration
            **kwargs: Additional Qwen2 configuration parameters
        """
        super().__init__(**kwargs)
        
        self.vision_embedding_type = vision_embedding_type
        self.patch_size = patch_size
        self.image_size = image_size
        self.vision_attention_mask = vision_attention_mask
        self.vision_embedding_intermediate_size = vision_embedding_intermediate_size
        
        # Set default layer_types if not provided
        if layer_types is None:
            num_layers = kwargs.get('num_hidden_layers', self.num_hidden_layers)
            self.layer_types = ["full_attention"] * num_layers
        else:
            self.layer_types = layer_types
        
        self.lora = lora
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load configuration from a pretrained model."""
        config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # Ensure it's a VoRA config
        if not isinstance(config, cls):
            # Convert to VoRA config
            config_dict = config.to_dict()
            return cls(**config_dict)
        
        return config
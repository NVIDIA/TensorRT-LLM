# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional

from tensorrt_llm.models.qwen.config import QWenConfig


class VoRAConfig(QWenConfig):
    """Configuration for Vision as LoRA (VoRA) models.
    
    VoRA is based on Qwen2.5 architecture with vision capabilities through
    merged LoRA adapters and lightweight vision embeddings.
    """

    def __init__(self,
                 # Vision-specific parameters
                 vision_patch_size: int = 14,
                 vision_image_size: int = 448,
                 vision_embedding_intermediate_size: int = 1536,
                 vision_embedding_type: str = "AIMv2",
                 vision_attention_mask: str = "bidirectional",
                 vision_max_patches: int = 1024,
                 # Base Qwen2 parameters
                 **kwargs):
        
        # Set default layer_types if not provided
        if 'layer_types' not in kwargs:
            # Get num_hidden_layers from kwargs or use default
            num_layers = kwargs.get('num_hidden_layers', 28)
            kwargs['layer_types'] = ["full_attention"] * num_layers
        
        super().__init__(**kwargs)
        
        # Vision embedding parameters
        self.vision_patch_size = vision_patch_size
        self.vision_image_size = vision_image_size
        self.vision_embedding_intermediate_size = vision_embedding_intermediate_size
        self.vision_embedding_type = vision_embedding_type
        self.vision_attention_mask = vision_attention_mask
        self.vision_max_patches = vision_max_patches
        
        # Model type identifier
        self.model_type = "vora"

    @classmethod
    def from_hugging_face(cls,
                         hf_config_or_dir: str,
                         dtype: str = 'auto',
                         mapping: Optional[Dict[str, Any]] = None,
                         quant_config: Optional[Dict[str, Any]] = None,
                         **kwargs):
        """Create VoRAConfig from HuggingFace model configuration."""
        import json
        import os
        from pathlib import Path

        if isinstance(hf_config_or_dir, str):
            config_path = Path(hf_config_or_dir)
            if config_path.is_dir():
                config_path = config_path / "config.json"
            
            with open(config_path, 'r') as f:
                hf_config = json.load(f)
        else:
            hf_config = hf_config_or_dir

        # Extract base Qwen2 config parameters
        base_config = super().from_hugging_face(
            hf_config_or_dir=hf_config,
            dtype=dtype,
            mapping=mapping,
            quant_config=quant_config,
            **kwargs
        )

        # Extract VoRA-specific parameters from HF config
        config_dict = base_config.to_dict()
        config_dict.update({
            'vision_patch_size': hf_config.get('patch_size', 14),
            'vision_image_size': hf_config.get('image_size', 448),
            'vision_embedding_intermediate_size': hf_config.get('vision_embedding_intermediate_size', 1536),
            'vision_embedding_type': hf_config.get('vision_embedding_type', 'AIMv2'),
            'vision_attention_mask': hf_config.get('vision_attention_mask', 'bidirectional'),
        })
        
        return cls(**config_dict)

    def to_dict(self):
        """Convert config to dictionary."""
        output = super().to_dict()
        
        # Add VoRA-specific fields
        output.update({
            'vision_patch_size': self.vision_patch_size,
            'vision_image_size': self.vision_image_size,
            'vision_embedding_intermediate_size': self.vision_embedding_intermediate_size,
            'vision_embedding_type': self.vision_embedding_type,
            'vision_attention_mask': self.vision_attention_mask,
            'vision_max_patches': self.vision_max_patches,
            'model_type': self.model_type,
        })
        
        return output
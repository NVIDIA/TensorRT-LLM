# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import numpy as np
import torch

from tensorrt_llm.functional import (ACT2FN, AllReduceFusionOp, PositionEmbeddingType, 
                                      concat, constant, embedding, expand_dims_like, 
                                      interpolate, matmul, permute, pow, view, 
                                      select, shape, slice, split, unsqueeze, where)
from tensorrt_llm.layers import Conv2d, Linear, LayerNorm, RmsNorm
from tensorrt_llm.module import Module
from tensorrt_llm.parameter import Parameter


def get_sincos_pos_embed_weights(h: int, w: int, embed_dim: int) -> np.ndarray:
    """Generate 2D sinusoidal position embeddings as numpy array for initialization."""
    assert embed_dim % 2 == 0, embed_dim
    
    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h, indexing="xy")
    grid = np.stack(grid, axis=0)
    grid = grid.view([2, 1, h, w])
    
    # Generate embeddings
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    
    # Compute embeddings for height
    pos_h = grid[0].view(-1)
    out_h = pos_h[:, None] @ omega[None, :]
    emb_sin_h = np.sin(out_h)
    emb_cos_h = np.cos(out_h)
    emb_h = np.concatenate([emb_sin_h, emb_cos_h], axis=1)
    
    # Compute embeddings for width  
    pos_w = grid[1].view(-1)
    out_w = pos_w[:, None] @ omega[None, :]
    emb_sin_w = np.sin(out_w)
    emb_cos_w = np.cos(out_w)
    emb_w = np.concatenate([emb_sin_w, emb_cos_w], axis=1)
    
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)
    return pos_embed.astype(np.float32)


class VisionEmbedding(Module):
    """Basic vision embedding layer for VoRA."""
    
    def __init__(self,
                 config,
                 hidden_size: int = 3584,
                 dtype=None):
        super().__init__()
        self.patch_size = config.vision_patch_size
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.max_image_size = config.vision_image_size
        
        # Patch embedding via Conv2D
        self.proj = Conv2d(
            in_channels=3,
            out_channels=hidden_size,
            kernel_size=(self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size),
            bias=True,
            dtype=dtype
        )
        
        # RMS normalization
        self.norm = RmsNorm(
            normalized_shape=hidden_size,
            eps=1e-05,
            dtype=dtype
        )
        
        # Pre-compute sincos embeddings for maximum size
        max_patches_h = self.max_image_size // self.patch_size
        max_patches_w = self.max_image_size // self.patch_size
        pos_embed_weights = get_sincos_pos_embed_weights(
            max_patches_h, max_patches_w, hidden_size
        )
        
        # Store as parameter for dynamic slicing
        self.pos_embed = Parameter(
            shape=pos_embed_weights.shape,
            dtype=dtype
        )
        self.pos_embed.value = pos_embed_weights
    
    def forward(self, pixel_values):
        # pixel_values: [batch_size, 3, height, width]
        # Apply conv2d to extract patches
        tokens = self.proj(pixel_values)  # [batch_size, hidden_size, h_patches, w_patches]
        
        # Get shapes
        batch_size = shape(tokens, 0)
        hidden_size = shape(tokens, 1)
        h_patches = shape(tokens, 2)
        w_patches = shape(tokens, 3)
        num_patches = h_patches * w_patches
        
        # Reshape to [batch_size, num_patches, hidden_size]
        tokens = view(tokens, [batch_size, hidden_size, num_patches])
        tokens = permute(tokens, [0, 2, 1])  # [batch_size, num_patches, hidden_size]
        
        # Apply normalization
        tokens = self.norm(tokens)
        
        # Add position embeddings (slice from pre-computed embeddings)
        pos_embed = slice(self.pos_embed.value, [0, 0], [num_patches, hidden_size])
        pos_embed = unsqueeze(pos_embed, 0)  # [1, num_patches, hidden_size]
        pos_embed = expand_dims_like(pos_embed, tokens, [0])  # [batch_size, num_patches, hidden_size]
        
        tokens = tokens + pos_embed
        
        return tokens


class AIMv2PatchEmbed(Module):
    """AIMv2-style patch embedding layer."""
    
    def __init__(self,
                 config,
                 dtype=None):
        super().__init__()
        self.patch_size = config.vision_patch_size
        
        self.proj = Conv2d(
            in_channels=3,
            out_channels=config.vision_embedding_intermediate_size,
            kernel_size=(self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size),
            bias=False,
            dtype=dtype
        )
        
        self.norm = RmsNorm(
            normalized_shape=config.vision_embedding_intermediate_size,
            eps=config.rms_norm_eps,
            dtype=dtype
        )
    
    def forward(self, x):
        # x: [batch_size, 3, height, width]
        x = self.proj(x)  # [batch_size, intermediate_size, h_patches, w_patches]
        
        # Get shapes
        batch_size = shape(x, 0)
        intermediate_size = shape(x, 1)
        h_patches = shape(x, 2)
        w_patches = shape(x, 3)
        num_patches = h_patches * w_patches
        
        # Reshape to [batch_size, num_patches, intermediate_size]
        x = view(x, [batch_size, intermediate_size, num_patches])
        x = permute(x, [0, 2, 1])  # [batch_size, num_patches, intermediate_size]
        
        x = self.norm(x)
        return x, h_patches, w_patches


class AIMv2ViTPreprocessor(Module):
    """AIMv2 Vision Transformer preprocessor for VoRA."""
    
    def __init__(self,
                 config,
                 hidden_size: int,
                 dtype=None):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.patch_size = config.vision_patch_size
        
        # Calculate maximum number of patches
        max_patches = (config.vision_image_size // config.vision_patch_size) ** 2
        self.max_patches_per_side = config.vision_image_size // config.vision_patch_size
        
        # Patch embedding layer
        self.patchifier = AIMv2PatchEmbed(config, dtype=dtype)
        
        # Learnable position embeddings
        self.pos_embed = Parameter(
            shape=(1, max_patches, config.vision_embedding_intermediate_size),
            dtype=dtype
        )
        
        # Output projection to match LLM hidden size
        self.out_proj = Linear(
            in_features=config.vision_embedding_intermediate_size,
            out_features=hidden_size,
            bias=False,
            dtype=dtype
        )
    
    def forward(self, x):
        # x: [batch_size, 3, height, width]
        batch_size = shape(x, 0)
        
        # Extract patches
        tokens, h_patches, w_patches = self.patchifier(x)
        # tokens: [batch_size, num_patches, intermediate_size]
        
        num_patches = shape(tokens, 1)
        intermediate_size = shape(tokens, 2)
        
        # Handle position embeddings with interpolation for variable resolution
        pos_embed = self.pos_embed.value  # [1, max_patches, intermediate_size]
        
        # Check if interpolation is needed
        max_num_patches = shape(pos_embed, 1)
        
        # If current patches exceed max, we need interpolation
        # In TensorRT-LLM, we handle this by reshaping and using interpolate
        if num_patches != max_num_patches:
            # Reshape pos_embed to 2D grid
            pos_embed_2d = view(
                pos_embed, 
                [1, self.max_patches_per_side, self.max_patches_per_side, intermediate_size]
            )
            # Permute to [1, intermediate_size, H, W] for interpolation
            pos_embed_2d = permute(pos_embed_2d, [0, 3, 1, 2])
            
            # Interpolate to match current patch grid
            pos_embed_2d = interpolate(
                pos_embed_2d,
                size=[h_patches, w_patches],
                mode='bilinear',
                align_corners=False
            )
            
            # Permute back and view to [1, num_patches, intermediate_size]
            pos_embed_2d = permute(pos_embed_2d, [0, 2, 3, 1])
            pos_embed = view(pos_embed_2d, [1, num_patches, intermediate_size])
        else:
            # Direct slicing for smaller images
            pos_embed = slice(pos_embed, [0, 0, 0], [1, num_patches, intermediate_size])
        
        # Expand for batch size
        pos_embed = expand_dims_like(pos_embed, tokens, [0])
        
        # Add position embeddings
        tokens = tokens + pos_embed
        
        # Project to LLM hidden size
        tokens = self.out_proj(tokens)
        
        return tokens


def build_vision_embedding(config, hidden_size, dtype=None):
    """Factory function to build vision embedding layer based on config."""
    if config.vision_embedding_type == "AIMv2":
        return AIMv2ViTPreprocessor(config, hidden_size, dtype)
    else:
        return VisionEmbedding(config, hidden_size, dtype)
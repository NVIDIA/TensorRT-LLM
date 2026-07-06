# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Self-contained config classes for MiniCPM-V 4.6.

The composite ``minicpmv4_6`` HF config is only registered in
``transformers>=5.7.0``.  TRT-LLM must load the model on older transformers
releases too, so we ship lightweight ``PretrainedConfig`` subclasses that mirror
the HF fields we consume, and build them from ``config.json`` inside
``pyexecutor.config_utils.load_pretrained_config`` (bypassing ``AutoConfig``).

The inner text config is normalized separately into a ``Qwen3NextConfig`` (the
runtime model that backs the Qwen3.5 dense text tower) via the shared Qwen3.5
compatibility shim; it is passed in already constructed.
"""

from transformers import PretrainedConfig

__all__ = ["MiniCPMV4_6Config", "MiniCPMV4_6VisionConfig"]


class MiniCPMV4_6VisionConfig(PretrainedConfig):
    """SigLIP2-style variable-resolution ViT config for MiniCPM-V 4.6."""

    model_type = "minicpmv4_6_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        num_hidden_layers: int = 27,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        image_size: int = 980,
        patch_size: int = 14,
        hidden_act: str = "gelu_pytorch_tanh",
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        insert_layer_id: int = 6,
        window_kernel_size=(2, 2),
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.insert_layer_id = insert_layer_id
        self.window_kernel_size = tuple(window_kernel_size)
        super().__init__(**kwargs)

    @property
    def window_hidden_size(self) -> int:
        return self.hidden_size * self.window_kernel_size[0] * self.window_kernel_size[1]

    @property
    def window_intermediate_size(self) -> int:
        return (self.intermediate_size * self.window_kernel_size[0] *
                self.window_kernel_size[1])


class MiniCPMV4_6Config(PretrainedConfig):
    """Composite config: SigLIP2 vision tower + Qwen3.5 dense text tower."""

    model_type = "minicpmv4_6"
    sub_configs = {"vision_config": MiniCPMV4_6VisionConfig}

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        insert_layer_id: int = 6,
        image_size: int = 448,
        drop_vision_last_layer: bool = False,
        image_token_id=None,
        video_token_id=None,
        downsample_mode: str = "16x",
        merge_kernel_size=(2, 2),
        merger_times: int = 1,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            vision_config = dict(vision_config)
            vision_config.pop("model_type", None)
            vision_config = MiniCPMV4_6VisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = MiniCPMV4_6VisionConfig()
        self.vision_config = vision_config
        self.vision_config.insert_layer_id = insert_layer_id

        # The loader passes an already-constructed Qwen3NextConfig. When
        # AutoConfig builds this class from a raw dict (e.g. AutoTokenizer),
        # keep the dict as-is; only the loader path needs a real text config.
        self.text_config = text_config

        self.insert_layer_id = insert_layer_id
        self.image_size = image_size
        self.drop_vision_last_layer = drop_vision_last_layer
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.downsample_mode = downsample_mode
        self.merge_kernel_size = tuple(merge_kernel_size)
        self.merger_times = merger_times
        self.patch_size = self.vision_config.patch_size
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

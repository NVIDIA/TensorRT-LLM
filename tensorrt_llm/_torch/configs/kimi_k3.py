# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-tree composite config for Kimi K3 ("kimi_k3") multimodal checkpoints.

Mirrors the checkpoint-shipped ``configuration_kimi_k3.KimiK3Config`` /
``KimiK3VisionConfig`` so TRT-LLM can parse the released Kimi K3 VLM checkpoint
without ``trust_remote_code`` for the config. The composite ``kimi_k3`` config
nests the in-tree text config (``KimiLinearConfig``) as ``text_config`` and a
flat ``KimiK3VisionConfig`` as ``vision_config``.

``pyexecutor.config_utils.load_pretrained_config`` keeps this composite config
(architectures ``KimiK3ForConditionalGeneration``) when the checkpoint ships both
``text_config`` and ``vision_config`` and multimodal is not disabled; otherwise
it flattens to the text config (``KimiLinearForCausalLM``) as before.
"""

from typing import Optional, Union

from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.configs.kimi_linear import KimiLinearConfig


class KimiK3VisionConfig(PretrainedConfig):
    """Vision-tower + projector sub-config for Kimi K3.

    Flat fields mirroring the checkpoint-shipped ``KimiK3VisionConfig``. The
    vision tower is a MoonViT-3D encoder (``vt_*`` fields) and the projector is a
    ``patchmergerv2`` MLP (``mm_*`` / ``projector_*`` fields).
    """

    model_type = "kimi_k3_vision"

    def __init__(
        self,
        patch_size: int = 14,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        init_pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
        vt_num_attention_heads: int = 12,
        vt_num_hidden_layers: int = 27,
        vt_hidden_size: int = 1024,
        vt_intermediate_size: int = 4096,
        merge_kernel_size: tuple = (2, 2),
        merge_type: str = "sd2_tpool",
        _attn_implementation: str = "flash_attention_2",
        # MM Projector parameters
        mm_projector_type: str = "patchmergerv2",
        mm_hidden_size: Optional[int] = None,
        projector_hidden_act: str = "gelu",
        projector_ln_eps: float = 1e-5,
        # vision tower parameters
        qkv_hidden_size: int = 1536,
        norm_type: str = "rmsnorm",
        attn_bias: bool = False,
        patch_embed_proj_bias: bool = False,
        mlp_type: str = "mlp2",
        linear_bias: bool = False,
        activation_func: str = "gelu_pytorch_tanh",
        pos_emb_interpolation_mode: str = "bilinear",
        # Other parameters
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 0,
        text_hidden_size: int = 7168,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.init_pos_emb_time = init_pos_emb_time
        self.pos_emb_type = pos_emb_type
        self.vt_num_attention_heads = vt_num_attention_heads
        self.vt_num_hidden_layers = vt_num_hidden_layers
        self.vt_hidden_size = vt_hidden_size
        self.vt_intermediate_size = vt_intermediate_size
        self.merge_kernel_size = tuple(merge_kernel_size)
        self.merge_type = merge_type

        # MM Projector config
        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = mm_hidden_size if mm_hidden_size is not None else vt_hidden_size
        self.projector_hidden_act = projector_hidden_act
        self.projector_ln_eps = projector_ln_eps
        self.text_hidden_size = text_hidden_size

        # vision tower parameters
        self.qkv_hidden_size = qkv_hidden_size
        self.norm_type = norm_type
        self.attn_bias = attn_bias
        self.patch_embed_proj_bias = patch_embed_proj_bias
        self.mlp_type = mlp_type
        self.linear_bias = linear_bias
        self.activation_func = activation_func
        self.pos_emb_interpolation_mode = pos_emb_interpolation_mode

        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id

        # transformers v5 PretrainedConfig.__init__ assigns
        # `attn_implementation` (default None) over any `_attn_implementation`
        # set beforehand, so route the default through the kwarg instead of
        # assigning the private attribute directly. An explicit
        # `attn_implementation` passed by the caller still wins.
        kwargs.setdefault("attn_implementation", _attn_implementation)
        super().__init__(pad_token_id=pad_token_id, **kwargs)


class KimiK3Config(PretrainedConfig):
    """Top-level composite config for the Kimi K3 multimodal model.

    ``text_config`` is the in-tree :class:`KimiLinearConfig` (the already
    brought-up text core); ``vision_config`` is :class:`KimiK3VisionConfig`.
    Sub-configs arrive as nested dicts from ``AutoConfig.from_pretrained`` and are
    rebuilt with the in-tree classes here so no ``trust_remote_code`` is needed.
    """

    model_type = "kimi_k3"

    def __init__(
        self,
        text_config: Optional[Union[dict, KimiLinearConfig]] = None,
        vision_config: Optional[Union[dict, KimiK3VisionConfig]] = None,
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 0,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config = KimiLinearConfig(**text_config)
        if isinstance(vision_config, dict):
            vision_config = KimiK3VisionConfig(**vision_config)
        self.text_config = text_config
        self.vision_config = vision_config

        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id
        # The routed-expert MXFP4 quantization lives on the text config; surface
        # it at the top level so TRT-LLM's quant-config extraction finds it.
        if getattr(self.text_config, "quantization_config", None) is not None:
            self.quantization_config = self.text_config.quantization_config

        super().__init__(pad_token_id=pad_token_id, **kwargs)

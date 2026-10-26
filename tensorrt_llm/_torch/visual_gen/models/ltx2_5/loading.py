# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2x
"""Native single-file LTX-2.5 VisualGen pipeline."""

from __future__ import annotations

import json

import torch
from safetensors import safe_open

from tensorrt_llm._torch.visual_gen.runtime_lora import _normalize_common_name
from tensorrt_llm.logger import logger

from ..ltx2.ltx2_core.rope import LTXRopeType
from ..ltx2.transformer_ltx2 import LTXModelType
from .transformer_ltx2_5 import LTX25Model


def _build_ltx25_transformer(pipeline_config) -> LTX25Model:
    model_config = pipeline_config.model_configs["transformer"]
    cfg = model_config.pretrained_config
    expected = {
        "_class_name": "AVTransformer3DModel",
        "cross_attention_adaln": True,
        "use_keyframes_abs_pos_embedding": True,
        "ff_bias": False,
        "use_audio_video_cross_attention": True,
        "text_encoder_norm_type": "PER_TOKEN_RMS",
    }
    for name, value in expected.items():
        if getattr(cfg, name, None) != value:
            raise ValueError(f"Unsupported LTX-2.5 transformer config: {name} must be {value!r}")
    rope_type = LTXRopeType(getattr(cfg, "rope_type", "interleaved"))
    double_precision_rope = getattr(cfg, "frequencies_precision", False) == "float64"
    apply_gated_attention = getattr(cfg, "apply_gated_attention", False)

    logger.info(
        f"LTX-2.5 transformer config: rope_type={rope_type.value}, "
        f"double_precision_rope={double_precision_rope}, "
        f"apply_gated_attention={apply_gated_attention} (AudioVideo)"
    )
    transformer = LTX25Model(
        model_type=LTXModelType.AudioVideo,
        use_keyframes_abs_pos_embedding=getattr(cfg, "use_keyframes_abs_pos_embedding", True),
        num_attention_heads=getattr(cfg, "num_attention_heads", 32),
        attention_head_dim=getattr(cfg, "attention_head_dim", 128),
        in_channels=getattr(cfg, "in_channels", 128),
        out_channels=getattr(cfg, "out_channels", 128),
        num_layers=getattr(cfg, "num_layers", 48),
        cross_attention_dim=getattr(cfg, "cross_attention_dim", 4096),
        audio_num_attention_heads=getattr(cfg, "audio_num_attention_heads", 32),
        audio_attention_head_dim=getattr(cfg, "audio_attention_head_dim", 64),
        audio_in_channels=getattr(cfg, "audio_in_channels", 128),
        audio_out_channels=getattr(cfg, "audio_out_channels", 128),
        audio_cross_attention_dim=getattr(cfg, "audio_cross_attention_dim", 2048),
        audio_positional_embedding_max_pos=getattr(cfg, "audio_positional_embedding_max_pos", [20]),
        norm_eps=float(getattr(cfg, "norm_eps", 1e-6)),
        caption_channels=getattr(cfg, "caption_channels", 3840),
        positional_embedding_theta=float(getattr(cfg, "positional_embedding_theta", 10000.0)),
        positional_embedding_max_pos=getattr(cfg, "positional_embedding_max_pos", [20, 2048, 2048]),
        timestep_scale_multiplier=getattr(cfg, "timestep_scale_multiplier", 1000),
        av_ca_timestep_scale_multiplier=getattr(cfg, "av_ca_timestep_scale_multiplier", 1),
        use_middle_indices_grid=getattr(cfg, "use_middle_indices_grid", True),
        rope_type=rope_type,
        double_precision_rope=double_precision_rope,
        apply_gated_attention=apply_gated_attention,
        model_config=model_config,
    )
    transformer._transformer_config = vars(cfg)
    return transformer


def _checkpoint_gemma_version(paths: list[str]) -> str:
    versions = set()
    for path in paths:
        with safe_open(path, framework="pt") as checkpoint:
            source = (checkpoint.metadata() or {}).get("gemma_source_checkpoint")
        if source is not None:
            versions.add(json.loads(source).get("gemma_version"))
    if len(versions) != 1 or not all(isinstance(version, str) and version for version in versions):
        raise ValueError(
            "LTX-2.5 checkpoint must declare one gemma_source_checkpoint.gemma_version."
        )
    return versions.pop()


def _load_component_strict(paths: list[str], module: torch.nn.Module, prefix: str) -> None:
    weights = {}
    for path in paths:
        with safe_open(path, framework="pt") as checkpoint:
            for key in checkpoint.keys():
                if key.startswith(prefix):
                    target = key.removeprefix(prefix)
                    if target in weights:
                        raise ValueError(f"Duplicate LTX-2.5 component weight: {key}")
                    weights[target] = checkpoint.get_tensor(key)
    # In particular, do not silently use randomly initialized readout projections.
    module.load_state_dict(weights, strict=True)


def _has_component(paths: list[str], prefix: str) -> bool:
    for path in paths:
        with safe_open(path, framework="pt") as checkpoint:
            if any(key.startswith(prefix) for key in checkpoint.keys()):
                return True
    return False


def validate_transformer_weights(module: torch.nn.Module, weights: dict[str, torch.Tensor]) -> None:
    """Reject incomplete BF16 checkpoints before the shared permissive loader runs."""
    normalized = [_normalize_common_name(key) for key in weights]
    if len(normalized) != len(set(normalized)):
        raise ValueError("Duplicate normalized LTX-2.5 transformer weight keys")
    expected = set(dict(module.named_parameters()))
    actual = set(normalized)
    missing, unexpected = expected - actual, actual - expected
    if missing or unexpected:
        raise ValueError(
            f"LTX-2.5 transformer checkpoint mismatch: missing={sorted(missing)[:20]}, "
            f"unexpected={sorted(unexpected)[:20]}"
        )

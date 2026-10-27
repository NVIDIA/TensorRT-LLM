# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2x
"""Gemma4 loading and tokenization for native LTX-2.5 text conditioning.

This module does not depend on the external LTX repository. Gemma4 support is
resolved lazily so older Transformers installations can still run LTX-2.
"""

import json
from pathlib import Path

import torch
import transformers
from packaging.version import Version
from safetensors import safe_open
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)


def _check_transformers_version() -> None:
    """Require the model-specific version qualified for BF16 hidden states."""
    if Version(transformers.__version__) != Version("5.14.1"):
        raise RuntimeError(
            "LTX-2.5 requires transformers==5.14.1 in its model-specific environment; "
            f"found {transformers.__version__}. Install the LTX-2.5 requirements after "
            "TensorRT-LLM; see examples/visual_gen/models/README.ltx2_5.md."
        )


def load_gemma4_config(path: str, expected_version: str) -> PretrainedConfig:
    """Validate the LTX-tuned encoder before loading any model weights."""
    _check_transformers_version()
    if not path:
        raise ValueError(
            "LTX-2.5 requires pipeline_config.text_encoder_path for its Gemma4 encoder."
        )
    if Path(path).is_file():
        with safe_open(path, framework="pt") as checkpoint:
            raw = (checkpoint.metadata() or {}).get("gemma_config")
        if raw is None:
            raise ValueError("Packed LTX-2.5 text encoder is missing gemma_config metadata.")
        config_dict = json.loads(raw)
        model_type = config_dict.pop("model_type", None)
        if model_type != "gemma4_unified":
            raise ValueError(f"LTX-2.5 requires gemma4_unified, got {model_type!r}.")
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        if model_type not in CONFIG_MAPPING:
            raise RuntimeError(
                "LTX-2.5 requires Transformers with Gemma4 unified support "
                "(validated with 5.14.1). Use a compatible LTX-2.5 environment; "
                "See examples/visual_gen/models/README.ltx2_5.md for model-specific installation."
            )
        config = AutoConfig.for_model(model_type, **config_dict)
    else:
        config = AutoConfig.from_pretrained(path, trust_remote_code=False)
    if config.model_type != "gemma4_unified":
        raise ValueError(f"LTX-2.5 requires gemma4_unified, got {config.model_type!r}.")
    actual_version = getattr(config, "gemma_version", None)
    if not expected_version or actual_version != expected_version:
        raise ValueError(
            f"LTX-2.5 Gemma version mismatch: checkpoint expects {expected_version!r}, "
            f"text encoder declares {actual_version!r}."
        )
    return config


def load_gemma4_tokenizer(path: str) -> PreTrainedTokenizerBase:
    """Load either an HF tokenizer directory or the embedded packed tokenizer."""
    if Path(path).is_file():
        from tokenizers import Tokenizer

        with safe_open(path, framework="pt") as checkpoint:
            tokenizer_json = checkpoint.get_tensor("tokenizer_json").numpy().tobytes()
            config_key = "hf_asset__tokenizer_config.json"
            if config_key in checkpoint.keys():
                raw_config = checkpoint.get_tensor(config_key).numpy().tobytes()
            else:
                raw_config = (checkpoint.metadata() or {}).get("tokenizer_config.json", "{}")
        config = json.loads(raw_config)
        # These are from_pretrained reconstruction options, not tokenizer kwargs.
        excluded = {
            "tokenizer_class",
            "auto_map",
            "model_max_length",
            "backend",
            "is_local",
            "local_files_only",
            "processor_class",
            "added_tokens_decoder",
        }
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer.from_buffer(tokenizer_json),
            **{key: value for key, value in config.items() if key not in excluded},
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("LTX-2.5 tokenizer requires BOS and padding token IDs.")
    return tokenizer


def _gemma4_weight_key(key: str) -> str | None:
    """Map Comfy-flat/older HF packs to current HF conditional-generation keys."""
    replacements = (
        ("model.layers.", "model.language_model.layers."),
        ("model.embed_tokens.", "model.language_model.embed_tokens."),
        ("model.norm.", "model.language_model.norm."),
        ("model.vision_embedder.", "model.embed_vision."),
        (
            "model.embed_vision.embedding_projection.",
            "model.embed_vision.multimodal_embedder.embedding_projection.",
        ),
        ("vision_model.", "model.embed_vision."),
        (
            "multi_modal_projector.embedding_projection.",
            "model.embed_vision.multimodal_embedder.embedding_projection.",
        ),
        ("audio_projector.", "model.embed_audio."),
    )
    for source, destination in replacements:
        if key.startswith(source):
            return destination + key.removeprefix(source)
    return key if key.startswith(("model.", "lm_head.")) else None


def load_gemma4_model(
    path: str, config: PretrainedConfig, dtype: torch.dtype, device: torch.device
) -> torch.nn.Module:
    """Load a complete HF model, validating weights, and retain its multimodal backbone.

    Using from_pretrained also initializes non-persistent RoPE/embedding buffers;
    a bare meta-model load_state_dict(assign=True) would leave those uninitialized.
    """
    _check_transformers_version()
    # Resolve the family lazily, without imposing Gemma4 on other LTX pipelines.
    # The concrete class also supports from_pretrained(None, state_dict=...),
    # unlike AutoModel versions that treat None as a repository path.
    from transformers import Gemma4UnifiedForConditionalGeneration

    kwargs = {"config": config, "torch_dtype": dtype, "output_loading_info": True}
    if Path(path).is_file():
        weights = {}
        with safe_open(path, framework="pt") as checkpoint:
            for key in checkpoint.keys():
                target = _gemma4_weight_key(key)
                if target is None:
                    continue
                if target in weights:
                    raise ValueError(f"Duplicate Gemma4 weight after remapping: {target}")
                weights[target] = checkpoint.get_tensor(key)
        if config.tie_word_embeddings and "lm_head.weight" not in weights:
            weights["lm_head.weight"] = weights["model.language_model.embed_tokens.weight"]
        model, info = Gemma4UnifiedForConditionalGeneration.from_pretrained(
            None, state_dict=weights, **kwargs
        )
    else:
        model, info = Gemma4UnifiedForConditionalGeneration.from_pretrained(path, **kwargs)
    if any(
        info.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError(f"Incomplete or incompatible Gemma4 text encoder weights: {info}")
    # Only hidden states are needed; do not compute the large vocabulary logits.
    # from_pretrained already casts parameters. Preserve FP32 RoPE buffers:
    # casting the whole module to BF16 changes text-conditioning numerics.
    # HF constructs embed_scale using its loading default dtype. Recreate it
    # in FP32, matching LTX's buffer initialization without BF16 rounding.
    model.model.language_model.embed_tokens.register_buffer(
        "embed_scale",
        torch.tensor(config.text_config.hidden_size**0.5, dtype=torch.float32),
        persistent=False,
    )
    return model.model.to(device=device).eval()


def tokenize_gemma4(
    tokenizer: PreTrainedTokenizerBase, prompts: list[str], max_length: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match LTX encoding: strip whitespace, ensure one leading BOS, then left-pad."""
    if max_length < 1:
        raise ValueError("max_sequence_length must be positive")
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        raise ValueError("LTX-2.5 tokenizer requires a BOS token ID.")
    token_ids = []
    for prompt in prompts:
        ids = tokenizer(prompt.strip(), padding=False, truncation=True, max_length=max_length)[
            "input_ids"
        ]
        if not ids or ids[0] != bos_id:
            ids = [bos_id, *ids][:max_length]
        token_ids.append(ids)
    encoded = tokenizer.pad(
        {"input_ids": token_ids},
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True,
    )
    return encoded.input_ids.to(device), encoded.attention_mask.to(device)


def pack_gemma4_hidden_states(
    hidden_states: tuple[torch.Tensor, ...], attention_mask: torch.Tensor
) -> torch.Tensor:
    """Apply the LTX-2.5 per-token RMS recipe and zero padding before readout.

    Preserve the checkpoint computation dtype, including its rounding order.
    LTX-2.5's existing FP32 packing path is deliberately left unchanged.
    """
    encoded = torch.stack(hidden_states, dim=-1)
    variance = encoded.square().mean(dim=2, keepdim=True)
    normalized = (encoded * torch.rsqrt(variance + 1e-6)).flatten(2, 3)
    return normalized.masked_fill(~attention_mask.bool().unsqueeze(-1), 0)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model setup shared by every executor: capacity, compilation and model resources."""

import math
import os

import torch

from tensorrt_llm.logger import logger


def _infer_max_seq_len_from_config(model: torch.nn.Module) -> int:
    if hasattr(model, "model_config") and model.model_config:
        model_config = model.model_config.pretrained_config
        rope_scaling = getattr(model_config, "rope_scaling", None)
        rope_factor = 1
        if rope_scaling is not None:
            rope_type = rope_scaling.get("type", rope_scaling.get("rope_type"))
            if rope_type not in ("su", "longrope", "llama3", "yarn"):
                rope_factor = rope_scaling.get("factor", 1.0)

        # Step 1: Find the upper bound of max_seq_len
        inferred_max_seq_len = 2048
        max_position_embeddings = getattr(model_config, "max_position_embeddings", None)
        if max_position_embeddings is None and hasattr(model_config, "text_config"):
            max_position_embeddings = getattr(
                model_config.text_config, "max_position_embeddings", None
            )
        if max_position_embeddings is not None:
            inferred_max_seq_len = max_position_embeddings

        # Step 2: Scale max_seq_len with rotary scaling
        if rope_factor != 1:
            inferred_max_seq_len = int(math.ceil(inferred_max_seq_len * rope_factor))
            logger.warning(
                f"max_seq_len is scaled to {inferred_max_seq_len} by rope scaling {rope_factor}"
            )

        return inferred_max_seq_len

    default_max_seq_len = 8192
    logger.warning(
        f"Could not infer max_seq_len from model config, using default value: {default_max_seq_len}"
    )
    return default_max_seq_len


def infer_max_seq_len(model: torch.nn.Module) -> int:
    """The longest sequence the loaded model supports."""
    # Vision encoders may expose their own sequence-length inference.
    if hasattr(model, "infer_max_seq_len"):
        return model.infer_max_seq_len()
    return _infer_max_seq_len_from_config(model)


def resolve_max_seq_len(model: torch.nn.Module, max_seq_len: int | None) -> int:
    """Clamp the requested max_seq_len to what the loaded model supports."""
    # Allow user to override the inferred max_seq_len with a warning.
    allow_long_max_model_len = os.getenv("TLLM_ALLOW_LONG_MAX_MODEL_LEN", "0").lower() in [
        "1",
        "true",
        "yes",
        "y",
    ]
    inferred_max_seq_len = infer_max_seq_len(model)

    if max_seq_len is None:
        logger.info(f"max_seq_len is not specified, using inferred value {inferred_max_seq_len}")
        return inferred_max_seq_len
    if inferred_max_seq_len < max_seq_len:
        if allow_long_max_model_len:
            logger.warning(
                f"User specified max_seq_len is larger than the config in the model config file "
                f"({inferred_max_seq_len}). Setting max_seq_len to user's specified value {max_seq_len}. "
            )
        else:
            # NOTE: py_executor_creator makes sure that the executor uses this
            # smaller value as its max_seq_len too.
            logger.warning(
                f"Specified {max_seq_len=} is larger than what the model can support "
                f"({inferred_max_seq_len}). Setting max_seq_len to {inferred_max_seq_len}. "
            )
            return inferred_max_seq_len
    return max_seq_len


def resolve_max_num_tokens(
    max_num_tokens: int | None, *, max_seq_len: int, max_batch_size: int
) -> int:
    """Clamp the requested max_num_tokens to max_seq_len * max_batch_size."""
    # Modified from tensorrt_llm/_bootstrap.py check_max_num_tokens
    if max_num_tokens is None:
        max_num_tokens = max_seq_len * max_batch_size
    if max_num_tokens > max_seq_len * max_batch_size:
        logger.warning(
            f"max_num_tokens ({max_num_tokens}) shouldn't be greater than "
            f"max_seq_len * max_batch_size ({max_seq_len * max_batch_size}), "
            f"specifying to max_seq_len * max_batch_size ({max_seq_len * max_batch_size})."
        )
        max_num_tokens = max_seq_len * max_batch_size
    return max_num_tokens

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint helpers of the Kimi K3 text-only targets.

The checkpoint is the vision-language wrapper's: ``language_model.*`` holds the text model,
``vision_tower.*`` and ``mm_projector.*`` the vision tower and its projector. The SM107 targets
are text only: they build no vision tower, and its tensors are never read (the engine hands the
model lazily mapped safetensors, so a key nobody reads costs no I/O). A step carrying
multimodal input is refused.

* :func:`text_model_weights`: the text model's share of the checkpoint, prefix stripped.
* :func:`check_text_only`: refuses multimodal input.
* :func:`filter_weights` / :func:`run_concurrently`: the prefix filter and the thread-pool loop
  of the checkpoint loader.
"""

from concurrent import futures
from typing import Any, Callable, Dict, Iterable, List, Optional

from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
from tensorrt_llm.logger import logger

__all__ = [
    "LANGUAGE_MODEL_PREFIX",
    "VISION_KEY_PREFIXES",
    "check_text_only",
    "filter_weights",
    "run_concurrently",
    "text_model_weights",
]

LANGUAGE_MODEL_PREFIX = "language_model."
#: The vision tower's and the projector's key families: in the checkpoint, never read.
VISION_KEY_PREFIXES = ("vision_tower.", "mm_projector.")


def filter_weights(prefix: str, weights: Dict) -> Dict:
    """The entries of ``weights`` under ``prefix``, with ``prefix + "."`` stripped."""
    result = {}
    for k, v in weights.items():
        if k.startswith(prefix):
            new_k = k[len(prefix) + 1 :]
            result[new_k] = v
    return result


def run_concurrently(
    func: Callable,
    args_list: Iterable[tuple],
    reduce_func: Optional[Callable] = None,
    pbar: Any = None,
    num_workers: Optional[int] = None,
) -> None:
    """``func(*args)`` for every ``args`` of ``args_list`` on a pool of ``num_workers`` threads;
    each result goes to ``reduce_func`` as it completes, and the first exception is re-raised."""
    with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_args = {executor.submit(func, *args): args for args in args_list}
        for result in futures.as_completed(future_to_args):
            args = future_to_args[result]
            try:
                part = result.result()
                if reduce_func:
                    reduce_func(part)
                if pbar:
                    pbar.update(1)
            except Exception as e:
                logger.error(f"Error executing {func.__name__} with args {args}: {str(e)}")
                raise


def text_model_weights(weights) -> Any:
    """The weights the text model loads: the ``language_model.*`` entries with the prefix
    stripped, as a ``ConsumableWeightsDict`` that keeps the checkpoint directory and prefix (the
    expert loader re-opens shard files by name); ``weights`` itself when no key has the prefix.
    The vision tower's and projector's entries are left unread."""
    if not any(k.startswith(LANGUAGE_MODEL_PREFIX) for k in weights):
        return weights
    skipped = sum(1 for k in weights if k.startswith(VISION_KEY_PREFIXES))
    if skipped:
        logger.info(
            f"Kimi K3 (text only): not loading the {skipped} vision tower / projector tensors"
        )
    lm_weights = ConsumableWeightsDict(filter_weights(LANGUAGE_MODEL_PREFIX[:-1], weights))
    checkpoint_dir = getattr(weights, "checkpoint_dir", None)
    if checkpoint_dir is not None:
        lm_weights.checkpoint_dir = checkpoint_dir
    lm_weights.checkpoint_prefix = LANGUAGE_MODEL_PREFIX
    return lm_weights


def check_text_only(multimodal_params: Optional[List[Any]]) -> None:
    """Refuse a step carrying multimodal input (the engine passes only requests with content)."""
    if multimodal_params:
        raise ValueError(
            "this Kimi K3 target is text only: it builds no vision tower, and a request carried "
            "multimodal input"
        )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the weight-loading hooks that models and their modules define.

`call_load_weights` copies checkpoint tensors into a model through its
``load_weights()``. The walks then call one post-load hook, such as
``post_load_weights()``, on every module that defines it. Modules opt in by
defining the hook; no base class is needed. The post-load walks skip a module
whose ``_weights_removed`` flag is set, because its weights were freed on
purpose. A draft model may declare the checkpoint key prefixes it reads as
``checkpoint_key_prefixes``; the draft read then gets only those keys.
"""

import inspect
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Optional

from torch import nn

if TYPE_CHECKING:
    from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper


def call_load_weights(
    load_method: Callable[..., None],
    weights: dict[str, Any],
    weight_mapper: Optional["BaseWeightMapper"],
    allow_partial_loading: bool = False,
) -> None:
    """Call a model's weight-loading method with the arguments it accepts.

    Models differ in what ``load_weights()`` or ``load_draft_weights()`` take:
    some accept ``weight_mapper``, some accept ``allow_partial_loading``. Only
    the arguments the method declares are passed.

    Args:
        load_method: bound method to call, e.g. ``model.load_weights``
        weights: checkpoint tensors by checkpoint name
        weight_mapper: weight mapper of the model, passed only if accepted
        allow_partial_loading: whether missing tensors are allowed

    Raises:
        AssertionError: ``allow_partial_loading`` is True but the method
            does not accept it.
    """
    args = inspect.getfullargspec(load_method).args
    kwargs = {}
    if "weight_mapper" in args:
        kwargs["weight_mapper"] = weight_mapper
    if "allow_partial_loading" in args:
        kwargs["allow_partial_loading"] = allow_partial_loading
    else:
        assert allow_partial_loading is False, (
            "allow_partial_loading is not supported for this model"
        )
    load_method(weights, **kwargs)


def draft_checkpoint_load_kwargs(model: nn.Module) -> dict[str, Any]:
    """Name the checkpoint key prefixes the draft reads, so loaders may skip files without them."""
    key_prefixes = getattr(getattr(model, "draft_model", None), "checkpoint_key_prefixes", None)
    if isinstance(key_prefixes, str):
        key_prefixes = (key_prefixes,)
    if isinstance(key_prefixes, Sequence) and key_prefixes:
        return {"key_prefixes": tuple(key_prefixes)}
    return {}


def select_draft_weights(
    weights: dict[str, Any], key_prefixes: tuple[str, ...] | None
) -> dict[str, Any]:
    """Keep only the draft's keys, so it gets the same keys whichever files were read."""
    if not key_prefixes:
        return weights
    return {name: value for name, value in weights.items() if name.startswith(key_prefixes)}


def setup_aliases(model: nn.Module) -> None:
    """Call ``setup_aliases()`` on every module, to create tensors that share storage."""
    for module in model.modules():
        module_setup_aliases: Optional[Callable[[], None]] = getattr(module, "setup_aliases", None)
        if module_setup_aliases is not None and not getattr(module, "_weights_removed", False):
            module_setup_aliases()


def prepare_post_transform_receiver(model: nn.Module) -> None:
    """Prepare a staged post-transform receiver before its weights arrive.

    Runs `setup_aliases()` on every module, then
    `setup_post_transform_receiver_aliases()` on every module that defines it.
    The second hook restores aliases that a model's weight loader creates,
    because a staged receiver gets its weights without running that loader.
    Only the MX receiver of a qualified root calls this; every other load path,
    including GMS read-only peers, runs `setup_aliases()` alone.
    """
    setup_aliases(model)
    for module in model.modules():
        receiver_aliases: Optional[Callable[[], None]] = getattr(
            module, "setup_post_transform_receiver_aliases", None
        )
        if receiver_aliases is not None and not getattr(module, "_weights_removed", False):
            receiver_aliases()


def mark_weights_transformed(model: nn.Module) -> None:
    """Mark modules with a transform guard as already transformed.

    Post-transform sharing skips ``transform_weights()``, because the incoming
    bytes already have the final runtime layout. The flag keeps later code
    from treating them as raw checkpoint bytes.
    """
    for module in model.modules():
        if hasattr(module, "_weights_transformed") and not getattr(
            module, "_weights_removed", False
        ):
            module._weights_transformed = True


def walk_cache_state(model: nn.Module) -> None:
    """Call ``cache_derived_state()`` on every module.

    This recomputes Python-side state from weights that already have their
    final layout, without transforming them again.
    """
    for module in model.modules():
        cache_derived_state: Optional[Callable[[], None]] = getattr(
            module, "cache_derived_state", None
        )
        if cache_derived_state is not None and not getattr(module, "_weights_removed", False):
            cache_derived_state()


def walk_full_post_load(model: nn.Module) -> None:
    """Call ``post_load_weights()``, the full post-load hook, on every module."""
    for module in model.modules():
        post_load_weights: Optional[Callable[[], None]] = getattr(module, "post_load_weights", None)
        if post_load_weights is not None and not getattr(module, "_weights_removed", False):
            post_load_weights()


def reset_weights_transformed(model: nn.Module) -> None:
    """Mark transformed modules as needing a new transform pass.

    Called before fresh, untransformed weights replace the current ones. Only
    modules that already carry the flag are touched.
    """
    for module in model.modules():
        if hasattr(module, "_weights_transformed"):
            module._weights_transformed = False

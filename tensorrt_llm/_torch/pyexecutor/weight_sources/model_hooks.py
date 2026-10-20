# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Call the weight-loading and post-load hooks that models and modules define.

`ModelLoader` and the weight sources fill a model in two stages. First the
checkpoint tensors go into the model through ``model.load_weights()`` (see
`call_load_weights()`). Then every module that defines post-load hooks gets a
chance to finish its weights, for example by fusing Q/K/V projections or
re-quantizing scales. Modules opt in by defining any of these methods:

* ``setup_aliases()`` creates tensors that share storage with other tensors.
* ``transform_weights()`` changes weight layouts once, after loading.
* ``cache_derived_state()`` recomputes Python-side values from the weights.
* ``post_load_weights()`` runs all of the above for the standard load path.

The walks below call one hook on every module that defines it. They skip a
module whose ``_weights_removed`` flag is set, because its weights were freed
on purpose (for example, layers that this rank does not run).
"""

import inspect
from typing import Callable, Optional

from torch import nn


def call_load_weights(
    load_method: Callable,
    weights: dict,
    weight_mapper,
    allow_partial_loading: bool = False,
) -> None:
    """Call a model's weight-loading method with the arguments it accepts.

    Models differ in what ``load_weights()`` (or ``load_draft_weights()``)
    takes: some accept a ``weight_mapper`` that renames checkpoint tensors, and
    some accept ``allow_partial_loading`` for incremental updates. This looks
    at the method's signature and passes only the arguments it declares.

    Called by the weight sources for the initial load and by
    `ModelLoader.reload()` for weight updates.

    Args:
        load_method: The bound method to call, such as ``model.load_weights``.
        weights: Checkpoint tensors keyed by checkpoint name.
        weight_mapper: The mapper for this model, passed only if accepted.
        allow_partial_loading: Whether missing tensors are allowed.

    Raises:
        AssertionError: If ``allow_partial_loading`` is True but the method
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


def mark_weights_transformed(model: nn.Module) -> None:
    """Mark modules with transform guards as already transformed.

    Post-transform sharing paths skip `transform_weights()` because the
    incoming bytes already use the final runtime layout. Preserve that
    lifecycle state on modules that participate in the transform guard
    protocol so a later orchestrator/refactor does not treat them as raw
    checkpoint bytes.
    """
    for module in model.modules():
        if hasattr(module, "_weights_transformed") and not getattr(
            module, "_weights_removed", False
        ):
            module._weights_transformed = True


def setup_aliases(model: nn.Module) -> None:
    """Run structural alias setup on eligible modules.

    The walk is duck-typed so modules can opt in without inheriting a
    shared base class. Modules whose weights were removed are skipped,
    matching the legacy full post-load walk.

    Args:
        model: Root decoder model whose module tree should be visited.

    Returns:
        None.
    """
    for module in model.modules():
        setup: Optional[Callable[[], None]] = getattr(module, "setup_aliases", None)
        if setup is not None and not getattr(module, "_weights_removed", False):
            setup()


def walk_transform(model: nn.Module) -> None:
    """Run one-shot weight transforms on eligible modules.

    The walk is duck-typed so modules can opt in without inheriting a shared
    base class. Modules whose weights were removed are skipped, and modules
    already marked `_weights_transformed` are left untouched until an
    orchestrator resets the flag after rebinding fresh weight bytes.

    Args:
        model: Root decoder model whose module tree should be visited.

    Returns:
        None.
    """
    for module in model.modules():
        transform_weights: Optional[Callable[[], None]] = getattr(module, "transform_weights", None)
        if (
            transform_weights is not None
            and not getattr(module, "_weights_removed", False)
            and not getattr(module, "_weights_transformed", False)
        ):
            transform_weights()


def walk_cache_state(model: nn.Module) -> None:
    """Recompute derived Python-side state on eligible modules.

    This walk is separate from weight transforms so callers that already
    have transformed weight bytes can refresh local Python state without
    mutating tensor layouts again.

    Args:
        model: Root decoder model whose module tree should be visited.

    Returns:
        None.
    """
    for module in model.modules():
        cache_derived_state: Optional[Callable[[], None]] = getattr(
            module, "cache_derived_state", None
        )
        if cache_derived_state is not None and not getattr(module, "_weights_removed", False):
            cache_derived_state()


def walk_full_post_load(model: nn.Module) -> None:
    """Run the backward-compatible post-load hook on eligible modules.

    This preserves the previous `ModelLoader` behavior for standard load
    paths while staged-hook migration proceeds incrementally.

    Args:
        model: Root decoder model whose module tree should be visited.

    Returns:
        None.
    """
    for module in model.modules():
        post_load_weights: Optional[Callable[[], None]] = getattr(module, "post_load_weights", None)
        if post_load_weights is not None and not getattr(module, "_weights_removed", False):
            post_load_weights()


def reset_weights_transformed(model: nn.Module) -> None:
    """Mark transformed modules as needing a new transform pass.

    Orchestrators call this before rebinding fresh, untransformed weights.
    The reset only touches modules that already carry the flag so unrelated
    modules do not grow staged-hook state eagerly.

    Args:
        model: Root decoder model whose module tree should be visited.

    Returns:
        None.
    """
    for module in model.modules():
        if hasattr(module, "_weights_transformed"):
            module._weights_transformed = False

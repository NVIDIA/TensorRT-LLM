# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The multimodal contract every target of a multimodal checkpoint meets.

A checkpoint is multimodal when its input processor accepts media
(``BaseMultimodalInputProcessor``). A target for such a checkpoint does one of two things:

* It serves media through the shared multimodal runtime: it inherits
  ``MultimodalRuntimeMixin`` (``_torch/multimodal/model.py``) and implements its required
  members. The executor then gives it encoder-output caching, chunk reuse, prefetch and the
  other runtime features.
* It serves the checkpoint's language model only: it inherits :class:`TextOnlyTarget`,
  builds no encoder, and the marker refuses requests carrying media at admission.

``modeling_v2_resolve`` checks this for every target a ``route`` returns, under ``auto`` as
well as ``require``: a target that meets neither is a bug in the target, not a deployment it
declines.

Like the router, this module imports nothing at load time: the registries it reads are
imported when a target is checked.
"""

#: ``multimodal_data`` entries that carry media for an encoder, or encoder output in place of it.
_MEDIA_KEYS = frozenset({"audio", "image", "video", "multimodal_embedding"})


class TextOnlyTarget:
    """Marker for a target that serves only the language model of a multimodal checkpoint.

    The target builds no multimodal encoder. Requests that carry media are refused at
    admission, so each fails on its own instead of failing every request in its batch.
    """

    def validate_multimodal_request_data(self, mm_data: dict) -> None:
        """Refuse a request whose multimodal payload carries media; accept position-only data."""
        media = sorted(key for key in _MEDIA_KEYS if mm_data.get(key) is not None)
        if media:
            raise ValueError(
                f"{type(self).__name__} serves this checkpoint's language model only and "
                f"does not accept multimodal input; the request carries {', '.join(media)}"
            )


def checkpoint_accepts_media(architecture: str) -> bool:
    """Whether the input processor registered for ``architecture`` accepts media."""
    from tensorrt_llm._torch.models.modeling_utils import get_registered_model_class
    from tensorrt_llm.inputs.registry import INPUT_PROCESSOR_REGISTRY, BaseMultimodalInputProcessor

    model_cls = get_registered_model_class(architecture)
    if model_cls is None:
        return False
    processor_cls = INPUT_PROCESSOR_REGISTRY._input_processors_cls_by_model_type.get(model_cls)
    return processor_cls is not None and issubclass(processor_cls, BaseMultimodalInputProcessor)


def check_target_multimodal_contract(architecture: str, target: type) -> None:
    """Raise ``MultimodalModelContractError`` unless ``target`` meets the contract above."""
    from tensorrt_llm._torch.multimodal.model import (
        MultimodalModelContractError,
        MultimodalRuntimeMixin,
    )

    runs_runtime = issubclass(target, MultimodalRuntimeMixin)
    text_only = issubclass(target, TextOnlyTarget)
    if runs_runtime and text_only:
        raise MultimodalModelContractError(
            f"{target.__qualname__} inherits both MultimodalRuntimeMixin and TextOnlyTarget"
        )
    if runs_runtime:
        target.check_multimodal_class_contract()
        return
    if text_only or not checkpoint_accepts_media(architecture):
        return
    raise MultimodalModelContractError(
        f"{target.__qualname__} is a target for the multimodal architecture {architecture!r}; "
        "it must inherit MultimodalRuntimeMixin to serve media, or TextOnlyTarget to serve "
        "the language model only"
    )

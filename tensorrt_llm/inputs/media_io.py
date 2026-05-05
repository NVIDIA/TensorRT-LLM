# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Per-modality merge rules for `media_io_kwargs`.

One subclass per modality; default is a shallow merge with runtime keys
winning. Subclasses override `merge_kwargs` for kwargs with semantic
interactions (e.g. video `fps` / `num_frames`). Loading itself still
lives in the free functions in `tensorrt_llm/inputs/utils.py`.
"""

from typing import Any, Dict, Optional, Type


class BaseMediaIO:
    """Default merge semantics for a media modality.

    Subclass per modality and override `merge_kwargs` to encode kwarg
    interactions specific to that modality.
    """

    @classmethod
    def merge_kwargs(
        cls,
        default_kwargs: Optional[Dict[str, Any]],
        runtime_kwargs: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Shallow-merge default and runtime kwargs; runtime wins per key."""
        merged: Dict[str, Any] = dict(default_kwargs or {})
        if runtime_kwargs:
            merged.update(runtime_kwargs)
        return merged


class ImageMediaIO(BaseMediaIO):
    """Image kwargs (`format`, `device`) are independent; default merge."""


class AudioMediaIO(BaseMediaIO):
    """Audio kwargs (`format`, `device`) are independent; default merge."""


class VideoMediaIO(BaseMediaIO):
    """Video kwargs include `fps` and `num_frames`, which interact."""

    @classmethod
    def merge_kwargs(
        cls,
        default_kwargs: Optional[Dict[str, Any]],
        runtime_kwargs: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        merged = super().merge_kwargs(default_kwargs, runtime_kwargs)
        # `fps` and `num_frames` together determine the time window of the
        # sampled clip. If a request overrides one without the other, keeping
        # the server's value for the unmentioned key produces a clip that
        # likely does not match the client's intent. Drop the unmentioned
        # default so the loader falls back to its built-in for that key.
        if runtime_kwargs:
            if "num_frames" in runtime_kwargs and "fps" not in runtime_kwargs:
                merged.pop("fps", None)
            elif "fps" in runtime_kwargs and "num_frames" not in runtime_kwargs:
                merged.pop("num_frames", None)
        return merged


MEDIA_IO_REGISTRY: Dict[str, Type[BaseMediaIO]] = {
    "image": ImageMediaIO,
    "video": VideoMediaIO,
    "audio": AudioMediaIO,
}

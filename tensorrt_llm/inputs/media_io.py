# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Generic I/O interfaces for the supported multimodal modalities."""

from types import MappingProxyType
from typing import Any, Dict, Literal, Mapping, Optional, Type

# Canonical set of supported media modalities for Pydantic field validation.
MediaModality = Literal["image", "video", "audio"]


class BaseMediaIO:
    """Per-modality I/O interface.

    Subclass per modality and override methods to customize behavior.
    Today the interface exposes only `merge_kwargs`; future I/O hooks
    (e.g. fetch/decode) belong on this base class.
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
    """I/O for the image modality; uses default merge (kwargs are independent)."""


class AudioMediaIO(BaseMediaIO):
    """I/O for the audio modality; uses default merge (kwargs are independent)."""


class VideoMediaIO(BaseMediaIO):
    """I/O for the video modality; customizes merge for `fps`/`num_frames` coupling."""

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


MEDIA_IO_REGISTRY: Mapping[MediaModality, Type[BaseMediaIO]] = MappingProxyType(
    {
        "image": ImageMediaIO,
        "video": VideoMediaIO,
        "audio": AudioMediaIO,
    }
)

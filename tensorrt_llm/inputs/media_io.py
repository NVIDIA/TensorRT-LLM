# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Generic I/O interfaces for the supported multimodal modalities."""

import asyncio
import base64
import tempfile
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Generic, Literal, Mapping, Optional, Tuple, Type, TypeVar, Union
from urllib.parse import urlparse

import numpy as np
import soundfile
import torch
from PIL import Image

from tensorrt_llm.inputs.multimodal_data import VideoData
from tensorrt_llm.inputs.utils import (
    _get_aiohttp_session,
    _load_and_convert_image,
    _load_video_by_cv2,
    _normalize_file_uri,
    _safe_aiohttp_get,
)
from tensorrt_llm.logger import logger

# Canonical set of supported media modalities for Pydantic field validation.
MediaModality = Literal["image", "video", "audio"]

# Output representations supported by `ImageMediaIO` and `VideoMediaIO`:
# `"pt"` → `torch.Tensor`, `"pil"` → `PIL.Image.Image` (per frame for video).
_SUPPORTED_IMAGE_FORMATS = ("pt", "pil")

_MediaT = TypeVar("_MediaT")


class BaseMediaIO(ABC, Generic[_MediaT]):
    """Per-modality I/O interface.

    Subclass per modality and override `load_bytes`, `load_base64`, and
    `load_file` to implement modality-specific decoding. URL scheme dispatch
    and async coordination live here on the base class and are shared by all
    modalities.
    """

    @classmethod
    def create(
        cls,
        default_kwargs: Optional[Dict[str, Any]],
        runtime_kwargs: Optional[Dict[str, Any]],
    ) -> "BaseMediaIO[_MediaT]":
        """Merge per-modality kwargs and return a configured instance."""
        merged = cls.merge_kwargs(default_kwargs, runtime_kwargs)
        logger.debug("effective %s kwargs keys: %s", cls.__name__, sorted(merged))
        return cls(**merged)

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

    @abstractmethod
    def load_bytes(self, data: bytes) -> _MediaT:
        """Decode media from raw bytes."""
        ...

    @abstractmethod
    def load_base64(self, media_type: str, data: str) -> _MediaT:
        """Decode media from a base64-encoded string."""
        ...

    @abstractmethod
    def load_file(self, url: str) -> _MediaT:
        """Load media from a local file URL or bare path.

        Receives the original URL string (not a parsed path) so that
        subclasses can apply their own normalization — e.g. unquoting
        `file://` URIs, or passing the string through verbatim to a
        decoder that handles bare paths.
        """
        ...

    async def async_load(self, url: str) -> _MediaT:
        """Fetch and decode media from a URL.

        Dispatches on scheme: http/https (remote fetch), data: (inline
        base64), or file:// / bare path (local file).
        """
        parsed = urlparse(url)
        if parsed.scheme in ("http", "https"):
            session = await _get_aiohttp_session()
            data = await _safe_aiohttp_get(url, session=session)
            return await asyncio.to_thread(self.load_bytes, data)
        elif parsed.scheme == "data":
            data_spec, b64_data = parsed.path.split(",", 1)
            parts = data_spec.split(";", 1)
            media_type = parts[0]
            encoding = parts[1] if len(parts) > 1 else ""
            if encoding != "base64":
                raise NotImplementedError("Only base64 data URLs are supported for now.")
            return await asyncio.to_thread(self.load_base64, media_type, b64_data)
        elif parsed.scheme in ("", "file"):
            return await asyncio.to_thread(self.load_file, url)
        else:
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r}")


class ImageMediaIO(BaseMediaIO[Union[Image.Image, torch.Tensor]]):
    """I/O for the image modality."""

    def __init__(self, format: str = "pt", device: str = "cpu") -> None:
        if format not in _SUPPORTED_IMAGE_FORMATS:
            raise ValueError(f"format must be one of {_SUPPORTED_IMAGE_FORMATS}, got {format!r}")
        self._format = format
        self._device = device

    def _postprocess(self, image: Image.Image) -> Union[Image.Image, torch.Tensor]:
        if self._format == "pt":
            from torchvision.transforms import ToTensor

            return ToTensor()(image).to(device=self._device)
        return image

    def load_bytes(self, data: bytes) -> Union[Image.Image, torch.Tensor]:
        return self._postprocess(_load_and_convert_image(BytesIO(data)))

    def load_base64(self, media_type: str, data: str) -> Union[Image.Image, torch.Tensor]:
        return self._postprocess(_load_and_convert_image(BytesIO(base64.b64decode(data))))

    def load_file(self, url: str) -> Union[Image.Image, torch.Tensor]:
        # Hand the parsed path (no unquoting) to PIL.
        parsed = urlparse(url)
        return self._postprocess(_load_and_convert_image(Path(parsed.path)))


class AudioMediaIO(BaseMediaIO[Tuple[np.ndarray, int]]):
    """I/O for the audio modality."""

    def load_bytes(self, data: bytes) -> Tuple[np.ndarray, int]:
        return soundfile.read(BytesIO(data))

    def load_base64(self, media_type: str, data: str) -> Tuple[np.ndarray, int]:
        return soundfile.read(BytesIO(base64.b64decode(data)))

    def load_file(self, url: str) -> Tuple[np.ndarray, int]:
        # Strip `file://` and unquote the path for `file:` URIs; pass
        # empty-scheme URLs through verbatim.
        parsed = urlparse(url)
        path = _normalize_file_uri(url) if parsed.scheme == "file" else url
        return soundfile.read(path)


class VideoMediaIO(BaseMediaIO[VideoData]):
    """I/O for the video modality; customizes merge for `fps`/`num_frames` coupling."""

    def __init__(
        self,
        num_frames: int = 10,
        fps: int = 30,
        format: str = "pt",
        device: str = "cpu",
        extract_audio: bool = False,
    ) -> None:
        if format not in _SUPPORTED_IMAGE_FORMATS:
            raise ValueError(f"format must be one of {_SUPPORTED_IMAGE_FORMATS}, got {format!r}")
        self._num_frames = num_frames
        self._fps = fps
        self._format = format
        self._device = device
        self._extract_audio = extract_audio

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

    def load_bytes(self, data: bytes) -> VideoData:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as f:
            f.write(data)
            f.flush()
            return _load_video_by_cv2(
                f.name,
                self._num_frames,
                self._fps,
                self._format,
                self._device,
                extract_audio=self._extract_audio,
            )

    def load_base64(self, media_type: str, data: str) -> VideoData:
        return self.load_bytes(base64.b64decode(data))

    def load_file(self, url: str) -> VideoData:
        # Pass the URL/path string straight to cv2 — it handles bare
        # paths but not `file://` URIs.
        return _load_video_by_cv2(
            url,
            self._num_frames,
            self._fps,
            self._format,
            self._device,
            extract_audio=self._extract_audio,
        )


MEDIA_IO_REGISTRY: Mapping[MediaModality, Type[BaseMediaIO]] = MappingProxyType(
    {
        "image": ImageMediaIO,
        "video": VideoMediaIO,
        "audio": AudioMediaIO,
    }
)

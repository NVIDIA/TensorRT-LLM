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
from PIL import Image

from tensorrt_llm.inputs.utils import (
    VideoData,
    _get_aiohttp_session,
    _load_and_convert_image,
    _load_video_by_cv2,
    _normalize_file_uri,
    _safe_aiohttp_get,
)

# Canonical set of supported media modalities for Pydantic field validation.
MediaModality = Literal["image", "video", "audio"]

_MediaT = TypeVar("_MediaT")


class BaseMediaIO(ABC, Generic[_MediaT]):
    """Per-modality I/O interface.

    Subclass per modality and override `load_bytes`, `load_base64`, and
    `load_file` to implement modality-specific decoding. URL scheme dispatch
    and async coordination live here on the base class and are shared by all
    modalities.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

    @classmethod
    def create(
        cls,
        default_kwargs: Optional[Dict[str, Any]],
        runtime_kwargs: Optional[Dict[str, Any]],
    ) -> "BaseMediaIO[_MediaT]":
        """Merge per-modality kwargs and return a configured instance."""
        merged = cls.merge_kwargs(default_kwargs, runtime_kwargs)
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
        subclasses can match the modality-specific normalization done by the
        legacy free functions.
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

    async def load_base64_async(self, media_type: str, data: str) -> _MediaT:
        """Async wrapper for `load_base64`.

        For use when the caller already has raw base64 data rather than a URL
        (e.g. `input_audio` parts).
        """
        return await asyncio.to_thread(self.load_base64, media_type, data)


class ImageMediaIO(BaseMediaIO[Union[Image.Image, Any]]):
    """I/O for the image modality."""

    def __init__(self, format: str = "pt", device: str = "cpu", **kwargs: Any) -> None:
        assert format in ["pt", "pil"], "format must be either Pytorch or PIL"
        super().__init__(format=format, device=device, **kwargs)
        self.format = format
        self.device = device

    def _postprocess(self, image: Image.Image) -> Union[Image.Image, Any]:
        if self.format == "pt":
            from torchvision.transforms import ToTensor

            return ToTensor()(image).to(device=self.device)
        return image

    def load_bytes(self, data: bytes) -> Union[Image.Image, Any]:
        return self._postprocess(_load_and_convert_image(BytesIO(data)))

    def load_base64(self, media_type: str, data: str) -> Union[Image.Image, Any]:
        return self._postprocess(_load_and_convert_image(BytesIO(base64.b64decode(data))))

    def load_file(self, url: str) -> Union[Image.Image, Any]:
        # Mirrors `async_load_image` for file/empty scheme: build a Path from
        # the parsed path (no unquoting) and hand it to PIL.
        parsed = urlparse(url)
        return self._postprocess(_load_and_convert_image(Path(parsed.path)))


class AudioMediaIO(BaseMediaIO[Tuple[np.ndarray, int]]):
    """I/O for the audio modality."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    async def async_load(self, url: str) -> Tuple[np.ndarray, int]:
        # Override to match `async_load_audio`, which does not accept `data:`
        # URLs (raw base64 audio goes through `load_base64_async` instead).
        parsed = urlparse(url)
        if parsed.scheme in ("http", "https"):
            session = await _get_aiohttp_session()
            data = await _safe_aiohttp_get(url, session=session)
            return await asyncio.to_thread(self.load_bytes, data)
        elif parsed.scheme in ("", "file"):
            return await asyncio.to_thread(self.load_file, url)
        else:
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r}")

    def load_bytes(self, data: bytes) -> Tuple[np.ndarray, int]:
        return soundfile.read(BytesIO(data))

    def load_base64(self, media_type: str, data: str) -> Tuple[np.ndarray, int]:
        return soundfile.read(BytesIO(base64.b64decode(data)))

    def load_file(self, url: str) -> Tuple[np.ndarray, int]:
        # Mirrors `async_load_audio`: strip `file://` and unquote the path for
        # `file:` URIs; pass empty-scheme URLs through verbatim.
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
        **kwargs: Any,
    ) -> None:
        assert format in ["pt", "pil"], "format must be either Pytorch or PIL"
        super().__init__(
            num_frames=num_frames,
            fps=fps,
            format=format,
            device=device,
            extract_audio=extract_audio,
            **kwargs,
        )
        self.num_frames = num_frames
        self.fps = fps
        self.format = format
        self.device = device
        self.extract_audio = extract_audio

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
                self.num_frames,
                self.fps,
                self.format,
                self.device,
                extract_audio=self.extract_audio,
            )

    def load_base64(self, media_type: str, data: str) -> VideoData:
        return self.load_bytes(base64.b64decode(data))

    def load_file(self, url: str) -> VideoData:
        # Mirrors `async_load_video`: pass the original URL/path string
        # directly to cv2 (cv2 handles bare paths but not `file://` URIs).
        return _load_video_by_cv2(
            url,
            self.num_frames,
            self.fps,
            self.format,
            self.device,
            extract_audio=self.extract_audio,
        )


MEDIA_IO_REGISTRY: Mapping[MediaModality, Type[BaseMediaIO]] = MappingProxyType(
    {
        "image": ImageMediaIO,
        "video": VideoMediaIO,
        "audio": AudioMediaIO,
    }
)

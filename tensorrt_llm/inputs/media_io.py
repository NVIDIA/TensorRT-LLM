# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Generic I/O interfaces for the supported multimodal modalities."""

import asyncio
import base64
import ipaddress
import math
import os
import socket
import tempfile
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Generic, List, Literal, Mapping, Optional, Tuple, Type, TypeVar, Union
from urllib.parse import unquote, urljoin, urlparse

import aiohttp
import numpy as np
import requests
import soundfile
import torch
from PIL import Image

from tensorrt_llm.inputs.multimodal_data import AudioData, VideoData
from tensorrt_llm.logger import logger


def rgba_to_rgb(
    image: Image.Image, background_color: Union[tuple[int, int, int], list[int]] = (255, 255, 255)
) -> Image.Image:
    """Convert an RGBA image to RGB with filled background color.

    Uses white (255, 255, 255) as the default background color because:
    1. It's the most neutral and commonly expected background for images
    2. Maintains backward compatibility with existing code
    """
    if image.mode != "RGBA":
        raise ValueError(f"Expected image mode to be 'RGBA', but got '{image.mode}'")
    converted = Image.new("RGB", image.size, background_color)
    converted.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return converted


def convert_image_mode(image: Image.Image, to_mode: str) -> Image.Image:
    """Convert image to specified mode with proper handling of RGBA to RGB conversion."""
    if image.mode == to_mode:
        return image
    elif image.mode == "RGBA" and to_mode == "RGB":
        return rgba_to_rgb(image)
    else:
        return image.convert(to_mode)


# Canonical set of supported media modalities for Pydantic field validation.
MediaModality = Literal["image", "video", "audio"]

# Output representations supported by `ImageMediaIO` and `VideoMediaIO`:
# `"pt"` → `torch.Tensor`, `"pil"` → `PIL.Image.Image` (per frame for video).
_SUPPORTED_IMAGE_FORMATS = ("pt", "pil")

# Module-level aiohttp session shared across all media fetch calls.
# Created lazily on first use inside the async event loop, then reused so that
# TCP connections are kept alive and not re-established per request.
_global_aiohttp_session: aiohttp.ClientSession | None = None


async def _get_aiohttp_session() -> aiohttp.ClientSession:
    """Return the shared aiohttp.ClientSession, creating it on first call."""
    global _global_aiohttp_session
    if _global_aiohttp_session is None or _global_aiohttp_session.closed:
        _global_aiohttp_session = aiohttp.ClientSession()
    return _global_aiohttp_session


# Maximum allowed response size for remote fetches (200 MB).
_MAX_RESPONSE_BYTES = 200 * 1024 * 1024

# Chunk size used while enforcing the response size cap.
_RESPONSE_CHUNK_BYTES = 1024 * 1024

# Maximum number of redirects allowed for remote fetches.
_MAX_REDIRECTS = 5

_REDIRECT_STATUSES = (301, 302, 303, 307, 308)


def _validate_url(url: str) -> None:
    """Validate that *url* points to a public, non-internal HTTP(S) resource.

    Raises ``RuntimeError`` for URLs that target non-global addresses or that
    use a scheme other than http / https.

    Note: validation is performed at DNS-resolution time. A DNS-rebinding
    attack (TTL=0, resolves to a public IP during validation then a private IP
    during the actual TCP connect) could bypass this check. For strict
    isolation, supplement with network-level egress filtering that blocks
    non-global address ranges at the host firewall.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise RuntimeError(f"Only http and https URLs are allowed, got: {parsed.scheme!r}")

    hostname = parsed.hostname
    if not hostname:
        raise RuntimeError("URL has no hostname")

    # Resolve to IP and check address range.
    try:
        infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise RuntimeError(f"Could not resolve hostname {hostname!r}") from exc

    for _family, _type, _proto, _canon, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if not ip.is_global or ip.is_multicast:
            raise RuntimeError(f"URL resolves to a non-public address ({ip})")


def _buffer_requests_response(resp: "requests.Response") -> "requests.Response":
    """Read a requests response with a hard size limit."""
    content = BytesIO()
    total = 0
    try:
        for chunk in resp.iter_content(chunk_size=_RESPONSE_CHUNK_BYTES):
            if not chunk:
                continue
            total += len(chunk)
            if total > _MAX_RESPONSE_BYTES:
                raise RuntimeError("Response exceeds maximum allowed size")
            content.write(chunk)
    except Exception:
        resp.close()
        raise

    data = content.getvalue()
    resp._content = data
    resp.raw = BytesIO(data)
    return resp


def _safe_request_get(url: str, *, stream: bool = False, timeout: int = 30) -> "requests.Response":
    """``requests.get`` wrapper that validates URLs and bounds response size."""
    del stream  # Kept for API compatibility; responses are always bounded.

    current_url = url
    _validate_url(current_url)

    for redirect_count in range(_MAX_REDIRECTS + 1):
        resp = requests.get(
            current_url,
            stream=True,
            timeout=timeout,
            allow_redirects=False,
        )
        if resp.status_code not in _REDIRECT_STATUSES:
            try:
                resp.raise_for_status()
            except Exception:
                resp.close()
                raise
            return _buffer_requests_response(resp)

        if redirect_count == _MAX_REDIRECTS:
            resp.close()
            raise RuntimeError("Too many redirects")

        redirect_url = resp.headers.get("Location", "")
        next_url = urljoin(current_url, redirect_url)
        resp.close()
        _validate_url(next_url)
        current_url = next_url

    raise RuntimeError("Too many redirects")


async def _read_aiohttp_content(response: aiohttp.ClientResponse) -> bytes:
    """Read an aiohttp response to EOF with a hard size limit."""
    content = BytesIO()
    total = 0
    while True:
        chunk = await response.content.read(_RESPONSE_CHUNK_BYTES)
        if not chunk:
            return content.getvalue()
        total += len(chunk)
        if total > _MAX_RESPONSE_BYTES:
            raise RuntimeError("Response exceeds maximum allowed size")
        content.write(chunk)


async def _safe_aiohttp_get(
    url: str, timeout_sec: int = 30, session: Optional[aiohttp.ClientSession] = None
) -> bytes:
    """Aiohttp GET wrapper that validates every redirect hop before following."""
    await asyncio.to_thread(_validate_url, url)
    timeout = aiohttp.ClientTimeout(total=timeout_sec)

    async def _fetch(fetch_session: aiohttp.ClientSession) -> bytes:
        current_url = url
        for _ in range(_MAX_REDIRECTS + 1):
            async with fetch_session.get(
                current_url, timeout=timeout, allow_redirects=False
            ) as response:
                if response.status in _REDIRECT_STATUSES:
                    redirect_url = response.headers.get("Location", "")
                    current_url = urljoin(current_url, redirect_url)
                    await asyncio.to_thread(_validate_url, current_url)
                    continue
                maybe_coro = response.raise_for_status()
                if asyncio.iscoroutine(maybe_coro):
                    await maybe_coro
                return await _read_aiohttp_content(response)
        raise RuntimeError("Too many redirects")

    if session is not None:
        return await _fetch(session)

    async with aiohttp.ClientSession(timeout=timeout) as owned_session:
        return await _fetch(owned_session)


def _load_and_convert_image(image):
    image = Image.open(image)
    image.load()
    return convert_image_mode(image, "RGB")


def _audio_frame_to_array(frame, mono: bool) -> np.ndarray:
    """Convert a PyAV audio frame to a NumPy array, averaging channels if mono."""
    chunk = frame.to_ndarray()
    if mono and chunk.ndim > 1:
        chunk = chunk.mean(axis=0)
    return chunk


# TODO(TRTLLM-12001): Add unit tests for this.
def extract_audio_from_video(
    source: Union[str, BytesIO],
    *,
    sr: Optional[float] = None,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """Extract the audio track from a video file using PyAV.

    Args:
        source: File path, URL, or BytesIO containing the video.
        sr: Target sample rate. If `None`, the native sample rate is kept.
        mono: If `True` (default), average channels to produce a mono waveform.

    Returns:
        `(waveform, sample_rate)` where *waveform* is a 1-D float32
        NumPy array and *sample_rate* is an integer in Hz.

    Raises:
        ValueError: If the video has no audio stream or the data is corrupt.
    """
    if os.environ.get("TRTLLM_ENABLE_PYAV", "0") != "1":
        raise RuntimeError(
            "PyAV is required for audio extraction from video. "
            "Set the environment variable TRTLLM_ENABLE_PYAV=1 to enable it."
        )
    try:
        import av
    except ImportError:
        raise ImportError("PyAV is required for audio extraction from video but is not installed.")

    try:
        with av.open(source) as container:
            if not container.streams.audio:
                raise ValueError("No audio stream found in the video.")
            stream = container.streams.audio[0]
            stream.thread_type = "AUTO"
            native_sr = stream.rate
            target_sr = int(sr) if sr is not None else native_sr

            chunks: List[np.ndarray] = []
            target_layout = "mono" if mono else stream.layout.name
            needs_resampling = (
                not math.isclose(float(target_sr), float(native_sr), rel_tol=0.0, abs_tol=1e-6)
                or stream.format.name != "fltp"
                or stream.layout.name != target_layout
            )
            resampler = (
                av.AudioResampler(
                    format="fltp",
                    layout=target_layout,
                    rate=target_sr,
                )
                if needs_resampling
                else None
            )
            for frame in container.decode(stream):
                if needs_resampling:
                    for out_frame in resampler.resample(frame):
                        chunks.append(_audio_frame_to_array(out_frame, mono))
                else:
                    chunks.append(_audio_frame_to_array(frame, mono))
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            "Invalid or corrupted video data when extracting audio. "
            "Ensure the input is a valid video file."
        ) from e

    if not chunks:
        raise ValueError("No audio frames decoded from the video.")

    audio = np.concatenate(chunks, axis=-1).astype(np.float32, copy=False)

    return audio, target_sr


def _load_video_by_cv2(
    video: str,
    num_frames: int = 10,
    fps: int = 30,
    format: str = "pt",
    device: str = "cpu",
    extract_audio: bool = False,
) -> VideoData:
    # Keep this import local to avoid importing cv2 if not needed
    import cv2

    assert format in ["pt", "pil"], "format must be either Pytorch or PIL"

    vidcap = cv2.VideoCapture(video)

    try:
        if not vidcap.isOpened():
            raise ValueError(
                f"Video '{video}' could not be opened. Make sure opencv is installed with video support."
            )

        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = vidcap.get(cv2.CAP_PROP_FPS)

        if frame_count <= 0 or original_fps <= 0:
            raise ValueError("Video has no frames or invalid FPS.")

        duration = frame_count / original_fps
        num_frames_to_sample = frame_count
        if num_frames > 0:
            num_frames_to_sample = min(num_frames, frame_count)
        if fps > 0:
            num_frames_to_sample = min(num_frames_to_sample, math.floor(duration * fps))
        num_frames_to_sample = max(1, num_frames_to_sample)  # at least one sample

        indices = np.linspace(0, frame_count - 1, num_frames_to_sample, dtype=int).tolist()

        # Sequential forward scan — grab() without per-frame seek
        target_set = set(indices)
        max_idx = indices[-1]
        raw_frames: dict[int, np.ndarray] = {}
        frame_idx = 0
        while frame_idx <= max_idx:
            grab_succeeded = vidcap.grab()
            if not grab_succeeded:
                break
            if frame_idx in target_set:
                # cv2 decodes frames in BGR order; convert to RGB for downstream use
                retrieve_succeeded, bgr_frame = vidcap.retrieve()
                if retrieve_succeeded:
                    raw_frames[frame_idx] = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            frame_idx += 1
        vidcap.release()

        if not raw_frames:
            raise ValueError("Video has no readable frames.")

        valid_indices = [i for i in indices if i in raw_frames]
        if format == "pt":
            # Bypass PIL: direct numpy HWC uint8 -> torch CHW float32
            loaded_frames = [
                torch.from_numpy(raw_frames[i])
                .permute(2, 0, 1)
                .float()
                .div_(255.0)
                .to(device=device)
                for i in valid_indices
            ]
        else:
            loaded_frames = [Image.fromarray(raw_frames[i]) for i in valid_indices]

        metadata = {
            "total_num_frames": frame_count,
            "fps": original_fps,
            "duration": duration,
            "frames_indices": valid_indices,
        }
    finally:
        # Release the OpenCV handle before any downstream re-open (e.g. PyAV
        # audio extraction on Windows) and to avoid leaking descriptors.
        vidcap.release()

    audio = None
    if extract_audio:
        try:
            audio_samples, audio_sample_rate = extract_audio_from_video(video)
            audio = AudioData(samples=audio_samples, sample_rate=audio_sample_rate)
        except ValueError as e:
            if "No audio stream found" in str(e):
                logger.warning("Video has no audio track, skipping audio extraction.")
            else:
                raise

    return VideoData(frames=loaded_frames, metadata=metadata, audio=audio)


def _normalize_file_uri(uri: str) -> str:
    """Strip the file:// scheme and unquote percent-encoded characters."""
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return unquote(parsed.path)
    return uri


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
        return self._postprocess(_load_and_convert_image(Path(_normalize_file_uri(url))))


class AudioMediaIO(BaseMediaIO[Tuple[np.ndarray, int]]):
    """I/O for the audio modality."""

    def __init__(self, **kwargs) -> None:
        # AudioMediaIO has no configurable parameters. An explicit __init__
        # is needed so that BaseMediaIO.create() does not crash when kwargs
        # are present (e.g. via --media_io_kwargs or the per-request API).
        if kwargs:
            logger.warning(
                "AudioMediaIO received unexpected kwargs %s — audio loading "
                "has no configurable parameters; kwargs will be ignored.",
                sorted(kwargs),
            )

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
        return _load_video_by_cv2(
            _normalize_file_uri(url),
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

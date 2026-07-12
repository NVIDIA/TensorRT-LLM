# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Generic I/O interfaces for the supported multimodal modalities."""

import asyncio
import base64
import functools
import ipaddress
import math
import os
import socket
import tempfile
from abc import ABC, abstractmethod
from concurrent.futures import Executor
from io import BytesIO
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    ClassVar,
    Dict,
    Generic,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from urllib.parse import unquote, urljoin, urlparse

import aiohttp
import numpy as np
import requests
import soundfile
import torch
from packaging.version import Version
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

# Output representations supported by `ImageMediaIO`:
# `"pt"` -> `torch.Tensor`, `"pil"` -> `PIL.Image.Image`.
_SUPPORTED_IMAGE_FORMATS = ("pt", "pil")

# Output representations supported by `VideoMediaIO`. See
# `_load_video_by_cv2` for the per-format contract.
_SUPPORTED_VIDEO_FORMATS = ("pt", "np", "pil")

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


def _get_cv2():
    """Import OpenCV on demand for the optional cv2-backed video decode path."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV (cv2) is required for video decoding but is not installed. "
            "Install it with `pip install opencv-python-headless`."
        ) from exc
    return cv2


def _select_cv2_stream_buffered_backend() -> Optional[int]:
    """Return a VideoCapture backend that can read from a Python `BytesIO`.

    Returns `None` if no such backend is available in this OpenCV build —
    the caller falls back to the tempfile path. The `videoio_registry`
    stream-buffered API was introduced in OpenCV 4.13.0, so older builds
    are gated out by a version check.

    Plugin-provided backends must implement the stream-buffered API at
    version 1.2+; older plugins exist that load fine but crash on stream
    open. Built-in backends (the FFMPEG path in the PyPI wheels) are always
    safe to use.
    """
    cv2 = _get_cv2()

    # The stream-buffered API was introduced in OpenCV 4.13.0. Older builds
    # don't have `cv2.videoio_registry.getStreamBufferedBackends`, so signal
    # "no usable backend" and let the caller fall back to the tempfile path.
    if Version(cv2.__version__) < Version("4.13.0"):
        return None

    vr = cv2.videoio_registry

    # `getStreamBufferedBackends()` enumerates every backend in this OpenCV
    # build that *claims* to support stream-buffered reads. We filter that
    # list down to one that's safe to use:
    #   1. `hasBackend(backend)` — the backend's shared library is actually
    #      loadable in this process (a backend can be enumerated by the
    #      registry but missing at link time on a stripped build).
    #   2. For plugin (non-built-in) backends, check the plugin's announced
    #      stream-buffered API/ABI version. Plugins reporting ABI<1 or
    #      (ABI==1, API<2) load fine but segfault on stream open in the
    #      wild — skip them. Built-in backends (FFMPEG in the PyPI wheels)
    #      have no plugin-version concept and are always safe.
    # The first surviving backend is returned. Order is OpenCV's preference.
    for backend in vr.getStreamBufferedBackends():
        if not vr.hasBackend(backend):
            continue
        if not vr.isBackendBuiltIn(backend):
            _, abi, api = vr.getStreamBufferedBackendPluginVersion(backend)
            if abi < 1 or (abi == 1 and api < 2):
                continue
        return backend
    return None


# Prefer /dev/shm (tmpfs, RAM-backed) over the system tempdir to avoid a disk
# round-trip; falls back to `None` (system default) when unavailable. Only
# consumed as the `dir=` argument to `tempfile.NamedTemporaryFile`, so the
# B108 predictable-path concern does not apply.
_VIDEO_TEMPFILE_DIR: Optional[str] = (  # nosec B108
    "/dev/shm"  # nosec B108
    if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK)  # nosec B108
    else None
)


def _load_video_by_cv2(
    video: Union[str, bytes],
    num_frames: int = 10,
    fps: int = 30,
    format: str = "pt",
    device: str = "cpu",
    extract_audio: bool = False,
    cv2_backend: Optional[int] = None,
) -> VideoData:
    """Decode a video and return sampled frames as a list.

    `video` is either a file path / URL (`str`) or raw mp4 bytes. When
    bytes are passed the caller must also pass `cv2_backend` (a
    stream-buffered backend id from `_select_cv2_stream_buffered_backend`);
    cv2.VideoCapture is then opened over a `BytesIO` via that backend, no
    tempfile required. Callers that have no stream-buffered backend
    available should spill to a tempfile themselves and pass a path.

    `format` controls the per-frame return type:
      `"pt"`    - list[torch.Tensor], dtype=float32, shape=(C, H, W), range
                  [0, 1]; rescaled and permuted to CHW here.
      `"np"` - list[np.ndarray], dtype=uint8, shape=(H, W, 3); returned as
                  decoded, leaving rescale/permute to the HF processor.
      `"pil"`   - list[PIL.Image], one per sampled frame.
    """
    assert format in ("pt", "np", "pil"), "format must be one of 'pt', 'np', 'pil'"

    cv2 = _get_cv2()

    # Open the source. Two cases:
    #   (a) `video` is a file path / URL str -> hand it straight to cv2.
    #   (b) `video` is mp4 bytes -> feed cv2 from an in-memory BytesIO via
    #       the caller-supplied stream-buffered backend. The buffer is held
    #       alive in `video_buf` because cv2 keeps a non-owning view into it.
    video_buf: Optional[BytesIO] = None
    if isinstance(video, (bytes, bytearray, memoryview)):
        if cv2_backend is None:
            raise ValueError(
                "cv2_backend must be provided when `video` is bytes; "
                "callers without a stream-buffered backend should spill "
                "the bytes to a tempfile and pass the path instead."
            )
        video_buf = BytesIO(bytes(video))
        vidcap = cv2.VideoCapture(video_buf, cv2_backend, [])
    else:
        vidcap = cv2.VideoCapture(video)

    try:
        if not vidcap.isOpened():
            src_repr = (
                f"<{len(video)} bytes>"
                if isinstance(video, (bytes, bytearray, memoryview))
                else f"'{video}'"
            )
            raise ValueError(
                f"Video {src_repr} could not be opened. Make sure opencv is installed with video support."
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
            # uint8 -> float32 + /255 rescale done once on the stacked buffer
            # so the dtype conversion is a single memory pass and there's one
            # Python torch call instead of one per frame.
            stacked_uint8 = np.stack([raw_frames[i] for i in valid_indices])
            stacked_f32 = stacked_uint8.astype(np.float32) * (1.0 / 255.0)
            tensor_nchw = torch.from_numpy(stacked_f32).permute(0, 3, 1, 2).contiguous()
            if device != "cpu":
                tensor_nchw = tensor_nchw.to(device)
            loaded_frames = list(torch.unbind(tensor_nchw, dim=0))
        elif format == "np":
            # uint8 HWC frames as-is; the HF processor rescales/permutes.
            loaded_frames = [raw_frames[i] for i in valid_indices]
        else:  # "pil"
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
        # extract_audio_from_video accepts Union[str, BytesIO]. When the
        # in-memory cv2 path was taken, `video` is still the original
        # bytes parameter — wrap in a fresh BytesIO (the one consumed by
        # cv2 is at end-of-stream). When the tempfile fallback was used,
        # `video` is the file path string and is passed through as-is.
        audio_source = (
            BytesIO(bytes(video)) if isinstance(video, (bytes, bytearray, memoryview)) else video
        )
        try:
            audio_samples, audio_sample_rate = extract_audio_from_video(audio_source)
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

    # Executor used by `async_load` to run blocking decode work off the
    # event loop. `None` selects the asyncio loop's default executor.
    # Servers can publish a dedicated pool via `set_executor` so
    # media decoding does not contend with unrelated `to_thread` callers.
    _executor: ClassVar[Optional[Executor]] = None

    @classmethod
    def set_executor(cls, executor: Executor) -> None:
        """Publish a shared executor for blocking decode work.

        Affects every existing and future subclass instance. Calling more
        than once replaces the previously configured executor.
        """
        BaseMediaIO._executor = executor

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
        base64), or file:// / bare path (local file). Blocking decode work
        runs on the executor published via `set_executor`, falling
        back to the asyncio loop's default executor when none is set.
        """
        parsed = urlparse(url)
        if parsed.scheme in ("http", "https"):
            session = await _get_aiohttp_session()
            data = await _safe_aiohttp_get(url, session=session)
            return await self._run_in_executor(self.load_bytes, data)
        elif parsed.scheme == "data":
            data_spec, b64_data = parsed.path.split(",", 1)
            parts = data_spec.split(";", 1)
            media_type = parts[0]
            encoding = parts[1] if len(parts) > 1 else ""
            if encoding != "base64":
                raise NotImplementedError("Only base64 data URLs are supported for now.")
            return await self._run_in_executor(self.load_base64, media_type, b64_data)
        elif parsed.scheme in ("", "file"):
            return await self._run_in_executor(self.load_file, url)
        else:
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r}")

    @staticmethod
    async def _run_in_executor(fn, *args, **kwargs):
        """Run a blocking decode callable on the configured executor."""
        if kwargs:
            fn = functools.partial(fn, **kwargs)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(BaseMediaIO._executor, fn, *args)


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
        # `"pt"` is the safe default every video model handles; models tuned
        # for the uint8-HWC path opt into `"np"` via
        # InputProcessor.get_preferred_media_io_kwargs().
        if format not in _SUPPORTED_VIDEO_FORMATS:
            raise ValueError(f"format must be one of {_SUPPORTED_VIDEO_FORMATS}, got {format!r}")
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
        # In-memory fast path when cv2 has a stream-buffered backend; spill
        # to a tempfile otherwise so cv2 can open it by path. The tempfile
        # is unlinked on context exit; the inode survives until cv2 closes
        # its own fd (Linux semantics), so the decode inside the `with`
        # block reads safely.
        cv2_backend = _select_cv2_stream_buffered_backend()
        if cv2_backend is not None:
            return _load_video_by_cv2(
                data,
                self._num_frames,
                self._fps,
                self._format,
                self._device,
                extract_audio=self._extract_audio,
                cv2_backend=cv2_backend,
            )
        with tempfile.NamedTemporaryFile(suffix=".mp4", dir=_VIDEO_TEMPFILE_DIR) as tmp:
            tmp.write(data)
            tmp.flush()
            return _load_video_by_cv2(
                tmp.name,
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

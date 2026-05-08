import asyncio
import base64
import ipaddress
import math
import os
import socket
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Coroutine, Dict, List, Optional, Tuple, TypedDict, Union
from urllib.parse import unquote, urljoin, urlparse

import aiohttp
import numpy as np
import requests
import soundfile
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from transformers import AutoProcessor, ProcessorMixin
from transformers.utils import logging

from tensorrt_llm.inputs.content_format import (ContentFormat,
                                                detect_content_format)
from tensorrt_llm.inputs.multimodal import (MultimodalServerConfig,
                                            default_hasher)
from tensorrt_llm.inputs.registry import (MULTIMODAL_PLACEHOLDER_REGISTRY,
                                          MultimodalPlaceholderPlacement)
from tensorrt_llm.llmapi.llm_utils import ModelLoader
from tensorrt_llm.tokenizer import TokenizerBase, TransformersTokenizer
from tensorrt_llm.tokenizer.deepseek_v32 import DeepseekV32Tokenizer

logger = logging.get_logger(__name__)

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


@dataclass
class BaseModalityData:
    """Base class for modality-specific data.

    This class serves as the foundation for all modality data types (image, video, audio, etc.),
    providing a common interface for modality-specific data structures.

    Subclasses should define their own attributes based on the specific needs of each modality.
    """


@dataclass
class VideoData(BaseModalityData):
    """Data class for video loading results.

    Attributes:
        frames: List of video frames, either as PIL Images or PyTorch tensors.
        metadata: Dictionary containing video metadata including:
            - total_num_frames: Total number of frames in the video
            - fps: Original frames per second of the video
            - duration: Duration of the video in seconds
            - frames_indices: List of indices of the sampled frames
    """
    frames: Union[List[Image.Image], List[torch.Tensor]]
    """The loaded video frames, either as PIL Images or PyTorch tensors."""

    metadata: Dict[str, Any]
    """Metadata associated with the video (e.g., fps, duration, frame indices)."""

    def __post_init__(self):
        """Validate that frames list is not empty."""
        if not self.frames:
            raise ValueError("frames list cannot be empty")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")


def rgba_to_rgb(
    image: Image.Image,
    background_color: Union[tuple[int, int, int], list[int]] = (255, 255, 255)
) -> Image.Image:
    """Convert an RGBA image to RGB with filled background color.

    Uses white (255, 255, 255) as the default background color because:
    1. It's the most neutral and commonly expected background for images
    2. Maintains backward compatibility with existing code
    """
    if image.mode != "RGBA":
        raise ValueError(
            f"Expected image mode to be 'RGBA', but got '{image.mode}'")
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
        raise RuntimeError(
            f"Only http and https URLs are allowed, got: {parsed.scheme!r}")

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


def _safe_request_get(url: str,
                      *,
                      stream: bool = False,
                      timeout: int = 30) -> "requests.Response":
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
        url: str,
        timeout_sec: int = 30,
        session: Optional[aiohttp.ClientSession] = None) -> bytes:
    """Aiohttp GET wrapper that validates every redirect hop before following."""
    await asyncio.to_thread(_validate_url, url)
    timeout = aiohttp.ClientTimeout(total=timeout_sec)

    async def _fetch(fetch_session: aiohttp.ClientSession) -> bytes:
        current_url = url
        for _ in range(_MAX_REDIRECTS + 1):
            async with fetch_session.get(current_url,
                                         timeout=timeout,
                                         allow_redirects=False) as response:
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


def load_base64_image(parsed_url: str) -> Image.Image:
    data_spec, data = parsed_url.path.split(",", 1)
    media_type, data_type = data_spec.split(";", 1)

    if data_type != "base64":
        msg = "Only base64 data URLs are supported for now."
        raise NotImplementedError(msg)

    content = base64.b64decode(data)
    image = _load_and_convert_image(BytesIO(content))
    return image


def load_base64_image_embeds(str_content: str) -> torch.Tensor:
    content_bytes = base64.b64decode(str_content)
    with BytesIO(content_bytes) as buf:
        image_data: torch.Tensor = torch.load(buf,
                                              weights_only=True,
                                              map_location="cpu")
    return image_data


def load_image(image: Union[str, Image.Image],
               format: str = "pt",
               device: str = "cpu") -> Union[Image.Image, torch.Tensor]:
    assert format in ["pt", "pil"], "format must be either Pytorch or PIL"

    if isinstance(image, Image.Image):
        return image.convert('RGB')

    parsed_url = urlparse(image)

    if parsed_url.scheme in ["http", "https"]:
        resp = _safe_request_get(image, stream=True)
        image = _load_and_convert_image(resp.raw)
    elif parsed_url.scheme == "data":
        image = load_base64_image(parsed_url)
    elif parsed_url.scheme in ("", "file"):
        image = _load_and_convert_image(image)
    else:
        raise ValueError(f"Unsupported URL scheme: {parsed_url.scheme!r}")

    if format == "pt":
        return ToTensor()(image).to(device=device)
    else:
        return image


async def async_load_image(
        image: Union[str, Image.Image],
        format: str = "pt",
        device: str = "cpu") -> Union[Image.Image, torch.Tensor]:
    assert format in ["pt", "pil"], "format must be either Pytorch or PIL"

    if isinstance(image, Image.Image):
        return image.convert('RGB')

    parsed_url = urlparse(image)

    if parsed_url.scheme in ["http", "https"]:
        session = await _get_aiohttp_session()
        content = await _safe_aiohttp_get(image, session=session)
        image = await asyncio.to_thread(_load_and_convert_image,
                                        BytesIO(content))
    elif parsed_url.scheme == "data":
        image = await asyncio.to_thread(load_base64_image, parsed_url)
    elif parsed_url.scheme in ("", "file"):
        image = await asyncio.to_thread(_load_and_convert_image,
                                        Path(parsed_url.path))
    else:
        raise ValueError(f"Unsupported URL scheme: {parsed_url.scheme!r}")

    if format == "pt":
        return await asyncio.to_thread(lambda: ToTensor()
                                       (image).to(device=device))
    else:
        return image


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
            "Set the environment variable TRTLLM_ENABLE_PYAV=1 to enable it.")
    try:
        import av
    except ImportError:
        raise ImportError(
            "PyAV is required for audio extraction from video but is not installed."
        )

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
            needs_resampling = (not math.isclose(
                float(target_sr), float(native_sr), rel_tol=0.0, abs_tol=1e-6)
                                or stream.format.name != "fltp"
                                or stream.layout.name != target_layout)
            resampler = (av.AudioResampler(
                format="fltp",
                layout=target_layout,
                rate=target_sr,
            ) if needs_resampling else None)
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
            "Ensure the input is a valid video file.") from e

    if not chunks:
        raise ValueError("No audio frames decoded from the video.")

    audio = np.concatenate(chunks, axis=-1).astype(np.float32, copy=False)

    return audio, target_sr


def _load_video_by_cv2(video: str,
                       num_frames: int = 10,
                       fps: int = 30,
                       format: str = "pt",
                       device: str = "cpu",
                       extract_audio: bool = False) -> VideoData:
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
            num_frames_to_sample = min(num_frames_to_sample,
                                       math.floor(duration * fps))
        num_frames_to_sample = max(1,
                                   num_frames_to_sample)  # at least one sample

        indices = np.linspace(0,
                              frame_count - 1,
                              num_frames_to_sample,
                              dtype=int).tolist()

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
                    raw_frames[frame_idx] = cv2.cvtColor(
                        bgr_frame, cv2.COLOR_BGR2RGB)
            frame_idx += 1
        vidcap.release()

        if not raw_frames:
            raise ValueError("Video has no readable frames.")

        valid_indices = [i for i in indices if i in raw_frames]
        if format == "pt":
            # Bypass PIL: direct numpy HWC uint8 -> torch CHW float32
            loaded_frames = [
                torch.from_numpy(raw_frames[i]).permute(
                    2, 0, 1).float().div_(255.0).to(device=device)
                for i in valid_indices
            ]
        else:
            loaded_frames = [
                Image.fromarray(raw_frames[i]) for i in valid_indices
            ]

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

    if extract_audio:
        try:
            audio_samples, audio_sample_rate = extract_audio_from_video(video)
            metadata["audio_samples"] = audio_samples
            metadata["audio_sample_rate"] = audio_sample_rate
        except ValueError as e:
            if "No audio stream found" in str(e):
                logger.warning(
                    "Video has no audio track, skipping audio extraction.")
            else:
                raise

    return VideoData(frames=loaded_frames, metadata=metadata)


def load_base64_video(video: str) -> BytesIO:
    parsed_url = urlparse(video)
    data_spec, data = parsed_url.path.split(",", 1)
    media_type, data_type = data_spec.split(";", 1)

    if data_type != "base64":
        msg = "Only base64 data URLs are supported for now."
        raise NotImplementedError(msg)

    content = base64.b64decode(data)
    return content


def load_video(video: str,
               num_frames: int = 10,
               fps: int = 30,
               format: str = "pt",
               device: str = "cpu",
               extract_audio: bool = False) -> VideoData:
    parsed_url = urlparse(video)
    if parsed_url.scheme in ["http", "https"]:
        resp = _safe_request_get(video, stream=False)
        with tempfile.NamedTemporaryFile(delete=True,
                                         suffix=".mp4") as tmp_file:
            tmp_file.write(resp.content)
            tmp_file.flush()
            return _load_video_by_cv2(tmp_file.name,
                                      num_frames,
                                      fps,
                                      format,
                                      device,
                                      extract_audio=extract_audio)
    elif parsed_url.scheme in ("", "file"):
        return _load_video_by_cv2(video,
                                  num_frames,
                                  fps,
                                  format,
                                  device,
                                  extract_audio=extract_audio)
    elif parsed_url.scheme == "data":
        decoded_video = load_base64_video(video)
        with tempfile.NamedTemporaryFile(delete=True,
                                         suffix='.mp4') as tmp_file:
            tmp_file.write(decoded_video)
            tmp_file.flush()
            return _load_video_by_cv2(tmp_file.name,
                                      num_frames,
                                      fps,
                                      format,
                                      device,
                                      extract_audio=extract_audio)
    else:
        raise ValueError(f"Unsupported video scheme: {parsed_url.scheme}")


async def async_load_video(video: str,
                           num_frames: int = 10,
                           fps: int = 30,
                           format: str = "pt",
                           device: str = "cpu",
                           extract_audio: bool = False) -> VideoData:
    assert format in ["pt", "pil"], "format must be either Pytorch or PIL"

    parsed_url = urlparse(video)

    def _load_from_bytes(data: bytes) -> VideoData:
        with tempfile.NamedTemporaryFile(delete=True, suffix='.mp4') as tmp:
            tmp.write(data)
            tmp.flush()
            return _load_video_by_cv2(tmp.name,
                                      num_frames,
                                      fps,
                                      format,
                                      device,
                                      extract_audio=extract_audio)

    if parsed_url.scheme in ["http", "https"]:
        session = await _get_aiohttp_session()
        video_data = await _safe_aiohttp_get(video, session=session)
        return await asyncio.to_thread(_load_from_bytes, video_data)
    elif parsed_url.scheme == "data":
        decoded_video = await asyncio.to_thread(load_base64_video, video)
        return await asyncio.to_thread(_load_from_bytes, decoded_video)
    elif parsed_url.scheme in ("", "file"):
        return await asyncio.to_thread(_load_video_by_cv2,
                                       video,
                                       num_frames,
                                       fps,
                                       format,
                                       device,
                                       extract_audio=extract_audio)
    else:
        raise ValueError(f"Unsupported URL scheme: {parsed_url.scheme!r}")


def _normalize_file_uri(uri: str) -> str:
    """Strip the file:// scheme and unquote percent-encoded characters."""
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return unquote(parsed.path)
    return uri


def load_audio(
    audio: str,
    format: str = "pt",
    device: str = "cpu",
) -> Tuple[np.ndarray, int]:
    parsed_url = urlparse(audio)
    if parsed_url.scheme in ["http", "https"]:
        resp = _safe_request_get(audio, stream=False)
        audio = BytesIO(resp.content)
    elif parsed_url.scheme in ("", "file"):
        audio = _normalize_file_uri(
            audio) if parsed_url.scheme == "file" else audio
    else:
        raise ValueError(f"Unsupported URL scheme: {parsed_url.scheme!r}")

    audio = soundfile.read(audio)
    return audio


async def async_load_audio(
    audio: str,
    format: str = "pt",
    device: str = "cpu",
    is_base64: bool = False,
) -> Tuple[np.ndarray, int]:
    if is_base64:
        raw_bytes = base64.b64decode(audio)
        return soundfile.read(BytesIO(raw_bytes))

    parsed_url = urlparse(audio)

    if parsed_url.scheme in ["http", "https"]:
        session = await _get_aiohttp_session()
        audio_data = await _safe_aiohttp_get(audio, session=session)
        audio = BytesIO(audio_data)
    elif parsed_url.scheme in ("", "file"):
        audio = _normalize_file_uri(
            audio) if parsed_url.scheme == "file" else audio
    else:
        raise ValueError(f"Unsupported URL scheme: {parsed_url.scheme!r}")

    return await asyncio.to_thread(soundfile.read, audio)


def encode_base64_content_from_url(content_url: str) -> str:
    """Encode a content retrieved from a remote url to base64 format."""

    resp = _safe_request_get(content_url, stream=False)
    result = base64.b64encode(resp.content).decode('utf-8')

    return result


def encode_base64_image(
    media: Image.Image,
    *,
    image_format: str = "JPEG",
) -> str:
    image = media

    with BytesIO() as buffer:
        image = convert_image_mode(image, "RGB")
        image.save(buffer, image_format)
        data = buffer.getvalue()

    return base64.b64encode(data).decode("utf-8")


# Helpers to always get the latest supported multimodal model types from the registry
def ALL_SUPPORTED_MULTIMODAL_MODELS():
    return MULTIMODAL_PLACEHOLDER_REGISTRY.get_registered_model_types()


def ALL_SUPPORTED_IMAGE_MODELS():
    return MULTIMODAL_PLACEHOLDER_REGISTRY.get_registered_image_model_types()


def ALL_SUPPORTED_VIDEO_MODELS():
    return MULTIMODAL_PLACEHOLDER_REGISTRY.get_registered_video_model_types()


def ALL_SUPPORTED_AUDIO_MODELS():
    return MULTIMODAL_PLACEHOLDER_REGISTRY.get_registered_audio_model_types()


def retrieve_multimodal_placeholder(model_type: str, modality: str,
                                    current_count: int) -> Optional[str]:
    """
        Get the appropriate placeholder for a given modality and model type.

        Args:
            model_type: The type of the multimodal model.
            modality: The modality of the data.
            current_count: The number of multimodal data already added.

    """
    if MULTIMODAL_PLACEHOLDER_REGISTRY.is_valid(model_type, modality):
        """
        The placeholder is a string with a single placeholder for the current count.
            - For example, if the placeholder is "<|image_{0}|>", and the current count is 1,
              the placeholder will be "<|image_1|>".
            - However, if the placeholder is "<|image|>", the current count would be ignored.
              In this case, the placeholder would be "<|image|>".
        """
        return MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholder(
            model_type, modality).format(current_count)
    raise TypeError(f"Unknown modality: {modality}")


class MultimodalData(TypedDict):
    """Type definition for multimodal data structure."""
    modality: str
    data: Any
    is_embedding: bool


class ConversationMessage(TypedDict, total=False):
    """Type definition for conversation message structure.

    Attributes:
        role: The message role (e.g. "user", "assistant", "system").
        content: Flattened text content (all text parts joined by newlines).
        media: List of multimodal data items attached to this message.
        content_parts: Ordered list preserving the interleaved positions of text and media as the
            user originally sent them. Only present when the message contains media.

            Each element is either:
            - A `str` for a text segment, or
            - A `dict` of the form `{"type": "<modality>", "media_index": <int>}`
              marking where a media item (image/video/audio) appeared.
              `media_index` is a 0-based index into `media`.

            This is used by `interleave_mm_placeholders` to insert multimodal placeholders at the
            correct positions, and to reconstruct the OpenAI-style content list for templates that
            handle media natively.
    """
    role: str
    content: str
    media: List[MultimodalData]
    content_parts: List[Union[str, dict]]


class MultimodalDataTracker:
    """Tracks and manages multimodal data for both sync and async processing."""

    def __init__(
            self,
            model_type: str,
            multimodal_server_config: Optional[MultimodalServerConfig] = None):
        self._model_type = model_type
        self._data = defaultdict[str, list](list)
        self._embeddings = defaultdict[str, list](list)
        self._placeholder_counts = defaultdict[str, int](int)
        self._placeholder_to_modality: dict[str, str] = {}
        self._multimodal_server_config = multimodal_server_config if multimodal_server_config is not None else MultimodalServerConfig(
        )

    async def retrieve_all_async(
        self
    ) -> tuple[Optional[Dict[str, List[Any]]], Optional[Dict[str, List[Any]]]]:
        """Retrieve all collected multimodal data and embeddings.

        All coroutines across all modalities (and across _data/_embeddings) are
        gathered concurrently in a single asyncio.gather call, so e.g. image
        downloads and video downloads overlap instead of running sequentially.
        """

        async def _retrieve(
                data: Optional[dict[str,
                                    list]]) -> Optional[Dict[str, List[Any]]]:
            if not data:
                return None
            pairs = [(modality, item) for modality, items in data.items()
                     if items for item in items]
            if not pairs:
                return None
            modality_keys, coroutines = zip(*pairs)
            results = await asyncio.gather(*coroutines)
            out: dict[str, list] = defaultdict(list)
            for modality, result in zip(modality_keys, results):
                out[modality].append(result)
            return dict(out)

        # _data and _embeddings also gathered concurrently
        data_result, embed_result = await asyncio.gather(
            _retrieve(self._data), _retrieve(self._embeddings))
        return data_result, embed_result

    def retrieve_all_sync(
        self
    ) -> tuple[Optional[Dict[str, List[Any]]], Optional[Dict[str, List[Any]]]]:
        """Retrieve all collected multimodal data and embeddings."""

        def _retrieve(
                data: Optional[dict[str,
                                    list]]) -> Optional[Dict[str, List[Any]]]:
            if not data:
                return None
            return {
                modality: items
                for modality, items in data.items() if items
            }

        return _retrieve(self._data), _retrieve(self._embeddings)

    def add_data(self,
                 media_type: str,
                 data: Union[Coroutine, Any],
                 *,
                 is_embedding: bool = False) -> Optional[str]:
        current_count = len(self._data[media_type]) + len(
            self._embeddings[media_type]) + 1
        placeholder = retrieve_multimodal_placeholder(self._model_type,
                                                      media_type, current_count)
        (self._embeddings
         if is_embedding else self._data)[media_type].append(data)
        if placeholder:
            self._placeholder_counts[placeholder] += 1
            self._placeholder_to_modality[placeholder] = media_type
        return placeholder

    def placeholder_counts(self) -> Dict[str, int]:
        """Get the count of multimodal placeholders."""
        return dict(self._placeholder_counts)

    def placeholder_modalities(self) -> Dict[str, str]:
        """Get the mapping from placeholder string to modality name."""
        return dict(self._placeholder_to_modality)


def add_multimodal_placeholders(model_type: str, text_prompt: str,
                                mm_placeholder_counts: dict[str, int]) -> str:
    """Add multimodal placeholders to the text prompt.

    Placeholders that already exist in the text are counted and subtracted
    from the requested count to avoid double-insertion (e.g. when the
    client already embeds ``<image>`` in the prompt text).
    """
    placeholders = []
    for placeholder, count in mm_placeholder_counts.items():
        existing = text_prompt.count(placeholder)
        needed = max(0, count - existing)
        placeholders.extend([placeholder] * needed)
    if not placeholders:
        return text_prompt
    parts = []
    match MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholder_placement(model_type):
        case MultimodalPlaceholderPlacement.BEFORE_TEXT:
            parts.extend(placeholders)
            parts.append(text_prompt)
        case MultimodalPlaceholderPlacement.AFTER_TEXT:
            parts.append(text_prompt)
            parts.extend(placeholders)
    return MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholders_separator(
        model_type).join(parts)


def interleave_mm_placeholders(
    model_type: str,
    content_parts: list[Union[str, dict]],
    mm_placeholder_counts: dict[str, int],
    placeholder_modalities: Dict[str, str],
) -> str:
    """Build a prompt string with placeholders interleaved at media positions.

    When `content_parts` preserves the original ordering of text and media
    items from the user's request, this function inserts the correct
    placeholder at each media position instead of bulk-prepending/appending.

    Args:
        model_type: The model type string (used to look up placeholder info).
        content_parts: Ordered list of text strings and media position dicts.
        mm_placeholder_counts: Mapping of placeholder -> expected count.
        placeholder_modalities: Mapping of placeholder string to modality
            name (e.g. `{"<image>": "image"}`).

    Returns:
        A single string with placeholders inserted at the correct positions.
    """
    if not content_parts:
        return add_multimodal_placeholders(model_type, "",
                                           mm_placeholder_counts)

    # Build a per-modality queue of placeholder strings (expanded by count).
    # This handles both shared placeholders (e.g. "<image>" with count=3)
    # and unique per-item placeholders (e.g. "<|image_1|>", "<|image_2|>").
    modality_placeholders: dict[str, list[str]] = {}
    for placeholder, count in mm_placeholder_counts.items():
        if placeholder not in placeholder_modalities:
            raise KeyError(
                f"Placeholder '{placeholder}' not found in "
                f"placeholder_modalities mapping. Known placeholders: "
                f"{list(placeholder_modalities.keys())}")
        modality = placeholder_modalities[placeholder]
        modality_placeholders.setdefault(modality,
                                         []).extend([placeholder] * count)

    parts: list[str] = []
    separator = MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholders_separator(
        model_type)
    # Track how many placeholders have been consumed per modality
    modality_cursor: dict[str, int] = {}

    for part in content_parts:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict):
            media_type = part.get("type", "image")
            queue = modality_placeholders.get(media_type)
            if not queue:
                continue
            cursor = modality_cursor.get(media_type, 0)
            if cursor < len(queue):
                parts.append(queue[cursor])
                modality_cursor[media_type] = cursor + 1

    return separator.join(parts)


def resolve_hf_chat_template(
    tokenizer: TokenizerBase,
    processor: ProcessorMixin,
    chat_template: Optional[str],
    tools: Optional[list[dict[str, Any]]],
) -> Optional[str]:
    """Resolve the appropriate chat template to use."""

    # 1. If chat_template is not None, return it
    if chat_template is not None:
        return chat_template

    # 2. If tool is not provided, use the processor's default chat template
    if not tools and processor and hasattr(processor, 'chat_template'):
        return processor.chat_template

    # 3. If tool is provided, use the tool
    try:
        return tokenizer.get_chat_template(chat_template, tools=tools)
    except Exception:
        logger.warning(
            "Failed to load AutoTokenizer chat template for %s",
            getattr(tokenizer, "name_or_path",
                    type(tokenizer).__name__))
    return None


def _resolve_content_format(model_type: str,
                            chat_template: Optional[str]) -> ContentFormat:
    """Determine the content format for the given model and template.

    Resolution order:
    1. Registry override (explicit per-model annotation).
    2. Jinja AST auto-detection (if a template string is available).
    3. Default to STRING.
    """
    # 1. Check registry for an explicit override
    registry_format = MULTIMODAL_PLACEHOLDER_REGISTRY.get_content_format(
        model_type)
    if registry_format is not None:
        return registry_format

    # 2. Auto-detect from template AST
    if chat_template is not None:
        return detect_content_format(chat_template)

    # 3. Default
    return ContentFormat.STRING


def _build_openai_content(
        conv: ConversationMessage,
        mm_placeholder_count: dict[str, int]) -> list[dict[str, Any]]:
    """Reconstruct OpenAI-style content list from a ConversationMessage.

    Uses `content_parts` (preserving media position) when available, otherwise falls back to placing
    text first then media items.
    """
    content_list: list[dict[str, Any]] = []
    content_parts = conv.get("content_parts")

    if content_parts:
        for part in content_parts:
            if isinstance(part, str):
                content_list.append({"type": "text", "text": part})
            elif isinstance(part, dict):
                media_type = part.get("type", "image")
                content_list.append({"type": media_type})
    else:
        # Fallback: text first, then media placeholders
        text = conv.get("content", "")
        if text:
            content_list.append({"type": "text", "text": text})
        for placeholder, count in mm_placeholder_count.items():
            # Infer modality from placeholder (e.g. "<image>" -> "image")
            modality = "image"
            if "video" in placeholder.lower():
                modality = "video"
            elif "audio" in placeholder.lower(
            ) or "so_embedding" in placeholder.lower():
                modality = "audio"
            for _ in range(count):
                content_list.append({"type": modality})

    return content_list


def apply_chat_template(
    *,
    model_type: str,
    tokenizer: Union[TransformersTokenizer, TokenizerBase],
    processor: ProcessorMixin,
    conversation: list[ConversationMessage],
    add_generation_prompt: bool,
    mm_placeholder_counts: list[dict[str, int]],
    tools: Optional[list[dict[str, Any]]] = None,
    documents: Optional[list[dict[str, str]]] = None,
    chat_template: Optional[str] = None,
    chat_template_kwargs: Optional[dict[str, Any]] = None,
    enable_tokenize: bool = False,
) -> (str | List[str]):
    """Apply chat template to the conversation.

    Uses content-format-driven dispatch:
    - PASSTHROUGH: skip template rendering, just concatenate content strings
    - OPENAI: reconstructs content as list of dicts for the template to handle
    - STRING: keeps flattened text with pre-inserted placeholders
    """

    # Handle DeepSeek V32 tokenizer with custom chat template
    if isinstance(tokenizer, DeepseekV32Tokenizer):
        prompt = tokenizer.apply_chat_template(
            messages=conversation,
            tools=tools,
            **(chat_template_kwargs or {}),
        )
        if enable_tokenize:
            return tokenizer.encode(prompt)
        return prompt

    # Check for PASSTHROUGH early — before we need tokenizer/processor/template.
    # The registry may already know this model skips chat templates entirely.
    registry_format = MULTIMODAL_PLACEHOLDER_REGISTRY.get_content_format(
        model_type)
    if registry_format == ContentFormat.PASSTHROUGH:
        return "".join([conv["content"] for conv in conversation])

    if isinstance(tokenizer, TransformersTokenizer):
        tokenizer = tokenizer.tokenizer  # we need the TokenizerBase for apply_chat_template

    hf_chat_template = resolve_hf_chat_template(tokenizer, processor,
                                                chat_template, tools)
    if hf_chat_template is None:
        raise ValueError(
            "No chat template found for the given tokenizer and tools.")

    # Determine content format and prepare conversation accordingly
    content_format = _resolve_content_format(model_type, hf_chat_template)

    if content_format == ContentFormat.OPENAI:
        # Path OPENAI: reconstruct content as list of dicts for the template
        for conv, mm_placeholder_count in zip(conversation,
                                              mm_placeholder_counts):
            if mm_placeholder_count:
                conv["content"] = _build_openai_content(conv,
                                                        mm_placeholder_count)
    # STRING path: placeholders already inserted in content by caller

    result = tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=enable_tokenize,
        return_dict=False,
        add_generation_prompt=add_generation_prompt,
        tools=tools,
        documents=documents,
        chat_template=hf_chat_template,
        **(chat_template_kwargs or {}),
    )

    return result


def default_multimodal_input_loader(
    *,
    tokenizer: Optional[Union[TransformersTokenizer, TokenizerBase]],
    model_dir: str,
    model_type: str,
    modality: str,
    prompts: List[str],
    media: Optional[Union[List[str], List[List[str]]]] = None,
    image_data_format: str = "pt",
    num_frames: int = 8,
    mm_embeddings: Optional[Union[List[torch.Tensor],
                                  List[List[torch.Tensor]]]] = None,
    device: str = "cpu",
    extract_audio: bool = False,
) -> List[dict[str, Union[str, torch.Tensor]]]:

    def convert_to_conversation_message(
        prompt: str,
        media: Union[Any, List[Any]],
        modality: str,
        is_embedding: bool = False,
    ) -> ConversationMessage:
        if isinstance(media, str):
            media = [media]
        if modality in ["image", "multiple_image"]:
            if is_embedding:
                _load = lambda mm: mm

                # each mm_embedding corresponds to each image placeholder
                if not isinstance(media, list):
                    media = [media]
            else:
                _load = lambda mm: load_image(
                    mm, format=image_data_format, device=device)

            mm_data = [
                MultimodalData(modality=modality,
                               data=_load(mm),
                               is_embedding=is_embedding) for mm in media
            ]
        elif modality == "video":
            if is_embedding:
                raise ValueError(
                    "External embedding is not supported for video modality yet."
                )
            mm_data = [
                MultimodalData(
                    modality=modality,
                    data=load_video(i,
                                    num_frames,
                                    format=image_data_format,
                                    device=device,
                                    extract_audio=extract_audio),
                    is_embedding=False,
                ) for i in media
            ]
        elif modality == "audio":
            if is_embedding:
                raise ValueError(
                    "External embedding is not supported for audio modality yet."
                )
            mm_data = [
                MultimodalData(
                    modality=modality,
                    data=load_audio(i, device=device),
                    is_embedding=False,
                ) for i in media
            ]
        elif modality == "image_audio":
            if is_embedding:
                raise ValueError(
                    "External embedding is not supported for image_audio modality yet."
                )
            # Use different load_xxx functions to match the modality.
            mm_data = []
            for m in media:
                data = None
                _modal = None
                if _modal is None:
                    try:
                        data = load_image(m,
                                          format=image_data_format,
                                          device=device)
                        _modal = "image"
                    except Exception:
                        pass
                if _modal is None:
                    try:
                        data = load_audio(m, device=device)
                        _modal = "audio"
                    except Exception:
                        pass
                if _modal is None:
                    raise ValueError(f"Unknown matching modality: {modality}")
                mm_data.append(
                    MultimodalData(modality=_modal,
                                   data=data,
                                   is_embedding=False))
        elif modality == "mixture_text_image":
            mm_data = []
            for m in media:
                if m:
                    mm_data.append(
                        MultimodalData(
                            modality="image",
                            data=load_image(m,
                                            format=image_data_format,
                                            device=device),
                            is_embedding=False,
                        ))
        else:
            raise ValueError(f"Unknown modality: {modality}")
        return ConversationMessage(role="user", content=prompt, media=mm_data)

    assert media is not None or mm_embeddings is not None, "Either media or mm_embeddings must be provided."
    assert media is None or mm_embeddings is None, "Either media or mm_embeddings must be provided, not both."
    media_or_embeddings = media if media is not None else mm_embeddings
    is_embedding = mm_embeddings is not None

    if len(media_or_embeddings) > len(prompts) and len(prompts) == 1:
        # 1 prompt + N media
        assert not isinstance(
            media_or_embeddings[0],
            list)  # media cannot be a list of lists in this case
        media_or_embeddings = [media_or_embeddings]
    assert len(media_or_embeddings) == len(prompts)

    is_passthrough = (MULTIMODAL_PLACEHOLDER_REGISTRY.get_content_format(
        model_type) == ContentFormat.PASSTHROUGH)

    if tokenizer is None and not is_passthrough:
        tokenizer = ModelLoader.load_hf_tokenizer(model_dir, use_fast=True)

    processor = None
    if not is_passthrough:
        processor = AutoProcessor.from_pretrained(model_dir,
                                                  use_fast=True,
                                                  trust_remote_code=True)

    inputs = []
    for prompt_idx, (prompt,
                     media) in enumerate(zip(prompts, media_or_embeddings)):
        conv = convert_to_conversation_message(prompt, media, modality,
                                               is_embedding)
        mm_data_tracker = MultimodalDataTracker(model_type)
        for mdata in conv["media"]:
            mdata_modality = mdata["modality"]
            if modality == "multiple_image":
                mdata_modality = "image"
            mm_data_tracker.add_data(mdata_modality,
                                     mdata["data"],
                                     is_embedding=is_embedding)
        mm_placeholder_counts = mm_data_tracker.placeholder_counts()
        prompt = conv["content"]
        if mm_placeholder_counts:
            # Resolve content format to decide whether to pre-insert
            # placeholders.  OPENAI templates handle media natively (e.g.
            # numbered <image N> tags), so we must NOT pre-insert or the
            # template's dedup guard will suppress its own output.
            hf_chat_template = resolve_hf_chat_template(tokenizer, processor,
                                                        None, None)
            content_format = _resolve_content_format(model_type,
                                                     hf_chat_template)
            if content_format != ContentFormat.OPENAI:
                conv["content"] = add_multimodal_placeholders(
                    model_type, conv["content"], mm_placeholder_counts)
        prompt = apply_chat_template(
            model_type=model_type,
            tokenizer=tokenizer,
            processor=processor,
            conversation=[conv],
            add_generation_prompt=True,
            mm_placeholder_counts=[mm_placeholder_counts])
        input = {"prompt": prompt}

        if mm_placeholder_counts:
            if mm_embeddings is not None:
                _, input[
                    "multi_modal_embeddings"] = mm_data_tracker.retrieve_all_sync(
                    )
            else:
                input[
                    "multi_modal_data"], _ = mm_data_tracker.retrieve_all_sync(
                    )
        inputs.append(input)

    return inputs


def get_cache_salt_id(cache_salt: str) -> int:
    b = cache_salt.encode("utf-8")
    h = default_hasher(b).digest(length=8)
    cache_salt_id = int.from_bytes(h, "little", signed=False)
    if cache_salt_id < 0 or cache_salt_id >= (1 << 64):
        raise ValueError(
            f"cache_salt_id must be in [0, 2**64 - 1], got {cache_salt_id}.")

    return cache_salt_id

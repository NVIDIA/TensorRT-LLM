# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import binascii
# The decoder walks pickle opcodes; nothing is unpickled.
import pickletools  # nosec B403
import struct
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import (Any, Collection, Coroutine, Dict, List, NamedTuple,
                    Optional, Tuple, TypedDict, Union)
from urllib.parse import urlparse

import numpy as np
import soundfile
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from transformers import AutoProcessor, PreTrainedTokenizerBase, ProcessorMixin
from transformers.utils import logging

from tensorrt_llm.inputs.chat_template_guard import \
    validate_chat_template_kwargs
from tensorrt_llm.inputs.content_format import (ContentFormat,
                                                detect_content_format)
from tensorrt_llm.inputs.data import prompt_inputs
from tensorrt_llm.inputs.media_io import (MEDIA_IO_REGISTRY, VideoMediaIO,
                                          _decode_base64_with_limit,
                                          _load_and_convert_image,
                                          _normalize_file_uri,
                                          _safe_aiohttp_get, _safe_request_get)
from tensorrt_llm.inputs.media_io import \
    convert_image_mode as convert_image_mode
from tensorrt_llm.inputs.multimodal import MultimodalServerConfig
from tensorrt_llm.inputs.multimodal_data import \
    BaseModalityData as BaseModalityData
from tensorrt_llm.inputs.multimodal_data import VideoData as VideoData
from tensorrt_llm.inputs.registry import (MULTIMODAL_PLACEHOLDER_REGISTRY,
                                          MultimodalPlaceholderPlacement)
from tensorrt_llm.llmapi.llm_utils import ModelLoader
from tensorrt_llm.metrics.perf_utils import _time_input_processing
from tensorrt_llm.tokenizer import TokenizerBase, TransformersTokenizer
from tensorrt_llm.tokenizer.deepseek_v4 import DeepseekV4Tokenizer
from tensorrt_llm.tokenizer.deepseek_v32 import DeepseekV32Tokenizer
from tensorrt_llm.tokenizer.deepseek_v41 import DeepseekV41Tokenizer

logger = logging.get_logger(__name__)


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


# A ``torch.save`` pickle of one tensor is a single call
# ``_rebuild_tensor_v2(storage, offset, size, stride, requires_grad,
# OrderedDict())`` whose storage is a persistent id. Client payloads are never
# unpickled or passed to ``torch.load``: the pickle's stack is replayed
# symbolically, only that shape is accepted, and the tensor is built from the
# replayed fields over the storage record that ``zipfile`` read.
_TENSOR_REBUILD = "torch._utils _rebuild_tensor_v2"
_TENSOR_HOOKS = "collections OrderedDict"
_TENSOR_STORAGE_DTYPES = {
    "torch FloatStorage": torch.float32,
    "torch BFloat16Storage": torch.bfloat16,
    "torch HalfStorage": torch.float16,
}
_TENSOR_PICKLE_OPS = frozenset({
    "PROTO", "FRAME", "GLOBAL", "STACK_GLOBAL", "BINPUT", "LONG_BINPUT",
    "MEMOIZE", "MARK", "BINUNICODE", "SHORT_BINUNICODE", "BININT", "BININT1",
    "BININT2", "LONG1", "TUPLE", "TUPLE1", "TUPLE2", "TUPLE3", "EMPTY_TUPLE",
    "NEWFALSE", "NEWTRUE", "BINPERSID", "REDUCE", "STOP"
})
# A single tensor's pickle is about 160 bytes.
_TENSOR_PICKLE_MAX_BYTES = 4096
# torch.save writes seven records for one tensor, about 600 bytes of central
# directory; zipfile parses all of it up front.
_TENSOR_ARCHIVE_MAX_DIRECTORY_BYTES = 4096
_ZIP_EOCD = struct.Struct("<4s4x2H2IH")
_ZIP64_EOCD = struct.Struct("<4s36xQQ")
_ZIP64_LOCATOR = struct.Struct("<4s4xQ4x")
_BASE64_WHITESPACE = str.maketrans("", "", " \t\r\n")
# Base64 is decoded in slices of this many characters (a multiple of 4): each
# call holds the GIL, so slices bound the event-loop stall of a worker thread.
_BASE64_SLICE_CHARS = 1 << 18
_TENSOR_ARCHIVE_RECORDS = frozenset({
    "data.pkl", "byteorder", "version", ".format_version", ".storage_alignment",
    ".data/version", ".data/serialization_id"
})


# Dataclasses, unlike named tuples, never compare equal to a tuple the
# pickle builds itself.
@dataclass(frozen=True)
class _PickleGlobal:
    name: str


@dataclass(frozen=True)
class _PickleCall:
    func: object
    args: tuple


@dataclass(frozen=True)
class _PicklePersistentId:
    pid: object


class _TensorSpec(NamedTuple):
    """The ``_rebuild_tensor_v2`` fields of a checked tensor pickle."""
    key: str
    dtype: torch.dtype
    storage_numel: int
    offset: int
    size: tuple[int, ...]
    stride: tuple[int, ...]


_PICKLE_MARK = object()


def _replay_tensor_pickle(pickled: bytes) -> object:
    """Replay the pickle's stack without calling anything; return its result."""
    try:
        ops = list(pickletools.genops(pickled))
    except ValueError as e:
        raise ValueError(f"Tensor payload pickle is malformed: {e}") from e
    if not ops or ops[0][0].name != "PROTO":
        raise ValueError("Tensor payload pickle is malformed.")
    stack: list[object] = []
    frame_end = 0
    # Each opcode, arguments included, ends where the next one starts.
    ends = [pos for _, _, pos in ops[1:]] + [ops[-1][2] + 1]
    try:
        for (opcode, arg, pos), end in zip(ops, ends):
            name = opcode.name
            if name not in _TENSOR_PICKLE_OPS:
                raise ValueError(
                    f"Tensor payload uses unsupported pickle opcode {name}.")
            if name == "STOP":
                break
            # torch.save starts the pickle with one PROTO of 2 to 5; a frame
            # starts after the previous one and ends within the pickle (a
            # FRAME opcode is 9 bytes), and no opcode crosses a frame's end.
            if (name == "PROTO" and (pos or not 2 <= arg <= 5)
                    or pos < frame_end < end or name == "FRAME" and
                (pos < frame_end or pos + 9 + arg > len(pickled))):
                raise ValueError("Tensor payload pickle is malformed.")
            if name == "FRAME":
                frame_end = pos + 9 + arg
            if name in ("BINPUT", "LONG_BINPUT", "MEMOIZE"):
                # The unpickler memoizes the value on top, above any mark.
                if not stack or stack[-1] is _PICKLE_MARK:
                    raise ValueError("Tensor payload pickle is malformed.")
                continue
            if name in ("PROTO", "FRAME"):
                continue
            if name == "MARK":
                stack.append(_PICKLE_MARK)
            elif name in ("GLOBAL", "STACK_GLOBAL"):
                if name == "STACK_GLOBAL":
                    qualname, module = stack.pop(), stack.pop()
                    if not (isinstance(module, str)
                            and isinstance(qualname, str)):
                        raise ValueError("Tensor payload pickle is malformed.")
                    arg = f"{module} {qualname}"
                # genops decodes backslash escapes that the unpickler keeps.
                elif pickled[pos + 1:end] != (arg.replace(" ", "\n", 1) +
                                              "\n").encode():
                    raise ValueError("Tensor payload pickle is malformed.")
                if (arg not in (_TENSOR_REBUILD, _TENSOR_HOOKS)
                        and arg not in _TENSOR_STORAGE_DTYPES):
                    raise ValueError(
                        f"Tensor payload references {arg!r}; only one "
                        "float16, bfloat16 or float32 tensor is accepted "
                        "(unsupported dtype or type).")
                stack.append(_PickleGlobal(arg))
            elif name == "NEWFALSE" or name == "NEWTRUE":
                stack.append(name == "NEWTRUE")
            elif name == "EMPTY_TUPLE":
                stack.append(())
            elif name in ("TUPLE1", "TUPLE2", "TUPLE3", "TUPLE"):
                start = (len(stack) -
                         int(name[-1]) if name != "TUPLE" else len(stack) -
                         stack[::-1].index(_PICKLE_MARK))
                items = tuple(stack[start:])
                if start < 0 or _PICKLE_MARK in items:
                    raise ValueError("Tensor payload pickle is malformed.")
                del stack[start - (name == "TUPLE"):]
                stack.append(items)
            elif name == "BINPERSID":
                stack.append(_PicklePersistentId(stack.pop()))
            elif name == "REDUCE":
                args = stack.pop()
                if not isinstance(args, tuple):
                    raise ValueError("Tensor payload pickle is malformed.")
                stack.append(_PickleCall(stack.pop(), args))
            elif arg is not None:
                stack.append(arg)
    except (IndexError, ValueError) as e:
        if isinstance(e, ValueError) and str(e).startswith("Tensor payload"):
            raise
        raise ValueError("Tensor payload pickle is malformed.") from e
    if len(stack) != 1:
        raise ValueError(
            "Tensor payload pickle must rebuild exactly one tensor.")
    return stack[0]


def _is_int_tuple(value: object) -> bool:
    return isinstance(value, tuple) and all(
        type(v) is int and v >= 0 for v in value)


def _check_tensor_pickle(pickled: bytes, max_bytes: int) -> _TensorSpec:
    """Check that a pickle rebuilds one bounded tensor and return its fields."""
    if len(pickled) > _TENSOR_PICKLE_MAX_BYTES:
        raise ValueError(
            f"Tensor payload pickle is {len(pickled)} bytes; one tensor needs "
            f"far less than {_TENSOR_PICKLE_MAX_BYTES}.")
    root = _replay_tensor_pickle(pickled)
    malformed = ValueError(
        "Tensor payload pickle must rebuild exactly one tensor.")
    if not (isinstance(root, _PickleCall) and root.func
            == _PickleGlobal(_TENSOR_REBUILD) and len(root.args) == 6):
        raise malformed
    storage, offset, size, stride, requires_grad, hooks = root.args
    if hooks != _PickleCall(_PickleGlobal(_TENSOR_HOOKS), ()):
        raise malformed
    pid = storage.pid if isinstance(storage, _PicklePersistentId) else None
    if not (isinstance(pid, tuple) and len(pid) == 5 and pid[0] == "storage"
            and isinstance(pid[1], _PickleGlobal) and isinstance(pid[2], str)
            and isinstance(pid[3], str) and type(pid[4]) is int):
        raise malformed
    dtype = _TENSOR_STORAGE_DTYPES.get(pid[1].name)
    if dtype is None:
        raise malformed
    element_size = dtype.itemsize
    if not (type(offset) is int and offset >= 0 and _is_int_tuple(size)
            and _is_int_tuple(stride) and len(size) == len(stride)
            and isinstance(requires_grad, bool) and pid[4] >= 0):
        raise malformed
    # A zero-sized dim or a size-1 dim leaves other values out of the extent
    # check below; bound them so as_strided never sees an overflowing value.
    if max((offset, *size, *stride)) > max_bytes:
        raise ValueError(
            f"Tensor payload size, stride or offset exceeds {max_bytes}.")
    storage_bytes = pid[4] * element_size
    logical_bytes = element_size
    span = 1
    for dim in size:
        logical_bytes *= dim
        span *= max(dim, 1)
    if max(logical_bytes, storage_bytes) > max_bytes:
        raise ValueError(
            f"Tensor payload describes {max(logical_bytes, storage_bytes)} "
            f"bytes, over the {max_bytes}-byte limit.")
    # torch multiplies the sizes in order, so a zero-sized dim does not stop
    # the others' product from overflowing.
    if span > max_bytes:
        raise ValueError(
            f"Tensor payload sizes multiply to {span}, over {max_bytes}.")
    # A view repeating its storage (expand, overlapping strides) would let a
    # small archive decode to a large tensor.
    if logical_bytes > storage_bytes:
        raise ValueError("Tensor payload view repeats storage elements; save a "
                         "contiguous tensor.")
    if logical_bytes and offset + sum(
        (d - 1) * s for d, s in zip(size, stride)) >= pid[4]:
        raise ValueError("Tensor payload view reaches past its storage.")
    return _TensorSpec(pid[2], dtype, pid[4], offset, size, stride)


def _check_central_directory_size(content: bytes) -> None:
    """Bound the central directory zipfile will parse, as torch.save writes it.

    torch.save ends the archive with a comment-free end-of-central-directory
    record, preceded by a zip64 one. Like zipfile, the directory size is taken
    from the zip64 record when its locator is present (large archives store a
    placeholder in the plain record).
    """
    tail = content[-_ZIP_EOCD.size:]
    if len(tail) < _ZIP_EOCD.size:
        raise ValueError("Tensor payload must be a torch.save zip archive.")
    signature, _, _, size, _, comment = _ZIP_EOCD.unpack(tail)
    if signature != b"PK\x05\x06" or comment:
        raise ValueError("Tensor payload must end with a torch.save zip "
                         "directory record and no comment.")
    locator = len(content) - _ZIP_EOCD.size - _ZIP64_LOCATOR.size
    if locator >= 0 and content[locator:locator + 4] == b"PK\x06\x07":
        # zipfile reads the zip64 record where the locator points (newer
        # versions) or right before the locator (older ones); torch.save
        # writes it right before, so require both to be the same record.
        if (_ZIP64_LOCATOR.unpack_from(content, locator)[1]
                != locator - _ZIP64_EOCD.size):
            raise ValueError("Tensor payload zip64 directory record must "
                             "directly precede its locator.")
        record = content[max(locator - _ZIP64_EOCD.size, 0):locator]
        if (len(record) != _ZIP64_EOCD.size
                or _ZIP64_EOCD.unpack(record)[0] != b"PK\x06\x06"):
            raise ValueError("Tensor payload zip64 directory record is "
                             "malformed.")
        size = _ZIP64_EOCD.unpack(record)[1]
    if size > _TENSOR_ARCHIVE_MAX_DIRECTORY_BYTES:
        raise ValueError(f"Tensor payload archive's central directory is over "
                         f"{_TENSOR_ARCHIVE_MAX_DIRECTORY_BYTES} bytes.")


def _read_tensor_archive(content: bytes,
                         max_bytes: int) -> tuple[_TensorSpec, torch.Tensor]:
    """Check a ``torch.save`` archive of one tensor and read its storage.

    Only the zip format with the records ``torch.save`` writes for one tensor,
    all uncompressed and little-endian, is accepted, and the pickle may only
    rebuild one float16, bfloat16 or float32 tensor.

    Returns:
        The tensor's fields and its storage record's bytes as a uint8 tensor.
    """
    _check_central_directory_size(content)
    try:
        with zipfile.ZipFile(BytesIO(content)) as archive:
            members = archive.infolist()
            # zipfile renames a member by a Unicode path field or at a NUL;
            # torch.load uses the stored name.
            if any(m.filename != m.orig_filename for m in members):
                raise ValueError(
                    "Tensor payload record names must be stored as is.")
            names = [m.filename for m in members]
            prefix = names[0].split("/", 1)[0] + "/" if names else ""
            records = [n[len(prefix):] for n in names if n.startswith(prefix)]
            data = [r for r in records if r.startswith("data/")]
            if (len(records) != len(names) or len(set(names)) != len(names)
                    or "data.pkl" not in records or len(data) > 1
                    or any(r not in _TENSOR_ARCHIVE_RECORDS
                           for r in records if not r.startswith("data/"))):
                raise ValueError(
                    "Tensor payload archive must hold only the records "
                    "torch.save writes for one tensor.")
            total = 0
            for member in members:
                # zipfile reads compress_size bytes even for a stored member,
                # so the two sizes must agree for file_size to bound reads.
                if (member.compress_type != zipfile.ZIP_STORED
                        or member.compress_size != member.file_size
                        or member.flag_bits & 0x1):
                    raise ValueError("Tensor payload members must be stored "
                                     "uncompressed and unencrypted.")
                # A zip64 offset past the archive would overflow the seek.
                if not 0 <= member.header_offset <= len(content) - 30:
                    raise ValueError(
                        "Tensor payload record lies outside the archive.")
                total += member.file_size
            if total > max_bytes:
                raise ValueError(
                    f"Tensor payload expands to {total} bytes, over the "
                    f"{max_bytes}-byte limit.")
            # Metadata records are sized before they are read, and all are
            # read so that zipfile checks their CRCs.
            sized = [(m.filename[len(prefix):], m) for m in members
                     if not m.filename[len(prefix):].startswith("data/")]
            for record, member in sized:
                if record == "byteorder" and member.file_size != len(b"little"):
                    raise ValueError("Tensor payload must be little-endian.")
                if member.file_size > _TENSOR_PICKLE_MAX_BYTES:
                    raise ValueError(
                        f"Tensor payload record {record} is {member.file_size} "
                        f"bytes; one tensor needs far less than "
                        f"{_TENSOR_PICKLE_MAX_BYTES}.")
            metadata = {r: archive.read(m) for r, m in sized}
            if metadata.get("byteorder", b"little") != b"little":
                raise ValueError("Tensor payload must be little-endian.")
            spec = _check_tensor_pickle(metadata["data.pkl"], max_bytes)
            storage_bytes = spec.storage_numel * spec.dtype.itemsize
            # The archive must hold the storage bytes before they are
            # allocated; the declared sizes alone are the client's to choose.
            if (data != [f"data/{spec.key}"] or
                    archive.getinfo(prefix + data[0]).file_size != storage_bytes
                    or storage_bytes > len(content)):
                raise ValueError(
                    "Tensor payload storage record does not match its tensor.")
            # Copy in chunks into the tensor's buffer, so no second full copy
            # exists; zipfile checks the record's CRC once all of it is read.
            # The buffer is not zero-filled, since every byte is overwritten.
            storage = torch.empty(storage_bytes,
                                  dtype=torch.uint8,
                                  device="cpu")
            view, filled = memoryview(storage.numpy()), 0
            with archive.open(prefix + data[0]) as record:
                while chunk := record.read(1 << 20):
                    if filled + len(chunk) > storage_bytes:
                        break
                    view[filled:filled + len(chunk)] = chunk
                    filled += len(chunk)
            if filled != storage_bytes:
                raise ValueError("Tensor payload storage record is truncated.")
    except (zipfile.BadZipFile, zipfile.LargeZipFile, EOFError, OSError,
            NotImplementedError, KeyError, struct.error) as e:
        # zipfile raises these on malformed archives.
        raise ValueError(
            f"Tensor payload must be a torch.save zip archive: {e}") from e
    return spec, storage


def _b64decode_strict(text: str, strip_whitespace: bool = False) -> bytes:
    """Decode standard base64, rejecting characters outside its alphabet.

    The text is decoded a slice at a time, carrying incomplete quartets over,
    so no call holds the GIL for the whole payload. Padding may only end it.
    """
    if sys.version_info >= (3, 11):
        # Unlike base64.b64decode, this does not first copy the text to bytes.
        def decode(piece: str) -> bytes:
            return binascii.a2b_base64(piece, strict_mode=True)
    else:

        def decode(piece: str) -> bytes:
            # Before 3.11, b64decode ignores characters after the padding.
            if len(piece) % 4:
                raise binascii.Error("Incorrect padding")
            return base64.b64decode(piece, validate=True)

    parts, carry, padded = [], "", False
    for start in range(0, len(text), _BASE64_SLICE_CHARS):
        piece = text[start:start + _BASE64_SLICE_CHARS]
        if strip_whitespace:
            piece = piece.translate(_BASE64_WHITESPACE)
        if padded:
            # Only stripped whitespace may follow the padding.
            if piece:
                raise ValueError("data after padding")
            continue
        piece = carry + piece
        if start + _BASE64_SLICE_CHARS < len(text):
            cut = len(piece) - len(piece) % 4
            piece, carry = piece[:cut], piece[cut:]
            if "=" in piece:
                if carry:
                    raise ValueError("data after padding")
                padded = True
        else:
            carry = ""
        parts.append(decode(piece))
    return b"".join(parts)


def load_base64_tensor(data: str, *, max_bytes: int) -> torch.Tensor:
    """Decode one dense tensor from base64-encoded ``torch.save`` bytes.

    The payload is never unpickled: its archive and pickle are checked and the
    tensor is built from the checked fields. The tensor's logical size is at
    most ``max_bytes``, and peak allocation here is about twice the decoded
    payload (the payload plus the tensor's storage). Only little-endian
    float16, bfloat16 and float32 tensors load, without conjugate or negative
    bits (save ``tensor.resolve_conj().resolve_neg()``). Callers check shape.

    Args:
        data: Standard base64 text of a ``torch.save`` file. Spaces, tabs and
            line breaks are ignored, up to two characters per 76 (MIME line
            wrapping).
        max_bytes: Upper bound on the decoded payload size.

    Returns:
        The decoded tensor on CPU.
    """
    if not isinstance(data, str):
        raise TypeError(f"Tensor payload must be a base64 string, got "
                        f"{type(data).__name__}.")
    encoded_limit = 4 * ((max_bytes + 2) // 3)
    # Line-wrapped (MIME style) text adds up to two characters per 76.
    if len(data) > encoded_limit + encoded_limit // 38 + 2:
        raise ValueError(
            f"Tensor payload is {len(data)} base64 characters, over the "
            f"{max_bytes}-byte limit.")
    # Base64 is ASCII; checked first because translate() is slow on other text.
    if not data.isascii():
        raise ValueError("Tensor payload is not valid base64: non-ASCII text.")
    content, error = None, ""
    try:
        content = _b64decode_strict(data)
    except ValueError as e:
        error = str(e)
    if content is None:
        # Line-wrapped base64 is accepted; only then is whitespace stripped.
        # This runs outside the handler, whose traceback would keep the
        # failed attempt's decoded slices alive.
        try:
            content = _b64decode_strict(data, strip_whitespace=True)
        except ValueError:
            raise ValueError(
                f"Tensor payload is not valid base64: {error}") from None
    if len(content) > max_bytes:
        raise ValueError(f"Tensor payload is {len(content)} bytes, over the "
                         f"{max_bytes}-byte limit.")
    spec, storage = _read_tensor_archive(content, max_bytes)
    return storage.view(spec.dtype).as_strided(spec.size, spec.stride,
                                               spec.offset)


async def async_load_base64_tensor(data: str, *,
                                   max_bytes: int) -> torch.Tensor:
    """Run :func:`load_base64_tensor` in a worker thread, off the event loop."""
    return await asyncio.to_thread(load_base64_tensor,
                                   data,
                                   max_bytes=max_bytes)


def load_image(image: Union[str, Image.Image],
               format: str = "pil",
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
        image = _load_and_convert_image(Path(_normalize_file_uri(image)))
    else:
        raise ValueError(f"Unsupported URL scheme: {parsed_url.scheme!r}")

    if format == "pt":
        return ToTensor()(image).to(device=device)
    else:
        return image


async def async_load_image(
        image: Union[str, Image.Image],
        format: str = "pil",
        device: str = "cpu") -> Union[Image.Image, torch.Tensor]:
    assert format in ["pt", "pil"], "format must be either Pytorch or PIL"

    if isinstance(image, Image.Image):
        return image.convert('RGB')

    parsed_url = urlparse(image)

    if parsed_url.scheme in ["http", "https"]:
        content = await _safe_aiohttp_get(image)
        image = await asyncio.to_thread(_load_and_convert_image,
                                        BytesIO(content))
    elif parsed_url.scheme == "data":
        image = await asyncio.to_thread(load_base64_image, parsed_url)
    elif parsed_url.scheme in ("", "file"):
        image = await asyncio.to_thread(_load_and_convert_image,
                                        Path(_normalize_file_uri(image)))
    else:
        raise ValueError(f"Unsupported URL scheme: {parsed_url.scheme!r}")

    if format == "pt":
        return await asyncio.to_thread(lambda: ToTensor()
                                       (image).to(device=device))
    else:
        return image


def load_base64_video(video: str) -> BytesIO:
    parsed_url = urlparse(video)
    data_spec, data = parsed_url.path.split(",", 1)
    media_type, data_type = data_spec.split(";", 1)

    if data_type != "base64":
        msg = "Only base64 data URLs are supported for now."
        raise NotImplementedError(msg)

    return _decode_base64_with_limit(data)


def load_video(video: str,
               num_frames: int = 10,
               fps: int = 30,
               format: str = "np",
               device: str = "cpu",
               extract_audio: bool = False) -> VideoData:
    """Load video frames with the shared source-byte cache identity.

    Encoded video inputs share the serving loader's size limit. Local files
    are read fully into memory during decoding. Local-file I/O failures raise
    ValueError, with the original OSError as their cause.
    """
    # Decode through `VideoMediaIO`, the loader `trtllm-serve` uses, so the
    # same video gets the same source-byte anchor (`raw_bytes_hash`) and
    # therefore the same multimodal hash offline and in serving.
    video_io = VideoMediaIO(num_frames=num_frames,
                            fps=fps,
                            format=format,
                            device=device,
                            extract_audio=extract_audio)
    parsed_url = urlparse(video)
    if parsed_url.scheme in ["http", "https"]:
        resp = _safe_request_get(video, stream=False)
        return video_io.load_bytes(resp.content)
    elif parsed_url.scheme in ("", "file"):
        try:
            return video_io.load_file(video)
        except OSError as error:
            raise ValueError(f"Video '{video}' could not be opened.") from error
    elif parsed_url.scheme == "data":
        return video_io.load_bytes(load_base64_video(video))
    else:
        raise ValueError(f"Unsupported video scheme: {parsed_url.scheme}")


async def async_load_video(video: str,
                           num_frames: int = 10,
                           fps: int = 30,
                           format: str = "np",
                           device: str = "cpu",
                           extract_audio: bool = False) -> VideoData:
    """Load video frames without blocking the event loop.

    Encoded video inputs share the serving loader's size limit. Local files
    are read fully into memory during decoding. Local-file I/O failures raise
    ValueError, with the original OSError as their cause.
    """
    assert format in ("pt", "np",
                      "pil"), "format must be one of 'pt', 'np', 'pil'"

    # See `load_video`: share the serving decode path so hashes agree.
    video_io = VideoMediaIO(num_frames=num_frames,
                            fps=fps,
                            format=format,
                            device=device,
                            extract_audio=extract_audio)
    parsed_url = urlparse(video)
    if parsed_url.scheme in ["http", "https"]:
        video_data = await _safe_aiohttp_get(video)
        return await asyncio.to_thread(video_io.load_bytes, video_data)
    elif parsed_url.scheme == "data":
        decoded_video = await asyncio.to_thread(load_base64_video, video)
        return await asyncio.to_thread(video_io.load_bytes, decoded_video)
    elif parsed_url.scheme in ("", "file"):
        try:
            return await asyncio.to_thread(video_io.load_file, video)
        except OSError as error:
            raise ValueError(f"Video '{video}' could not be opened.") from error
    else:
        raise ValueError(f"Unsupported URL scheme: {parsed_url.scheme!r}")


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
        audio_data = await _safe_aiohttp_get(audio)
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
        tools: Message-level (dynamic) tool declarations carried on system messages. Only
            populated for models whose python-renderer chat template consumes them (kimi_k3);
            absent for other models.
    """
    role: str
    content: str
    media: List[MultimodalData]
    content_parts: List[Union[str, dict]]
    tools: List[Dict[str, Any]]


class MultimodalDataTracker:
    """Tracks and manages multimodal data for both sync and async processing."""

    def __init__(
        self,
        model_type: str,
        multimodal_server_config: Optional[MultimodalServerConfig] = None,
        request_media_io_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self._model_type = model_type
        self._data = defaultdict[str, list](list)
        self._embeddings = defaultdict[str, list](list)
        self._placeholder_counts = defaultdict[str, int](int)
        self._placeholder_to_modality: dict[str, str] = {}
        # Prompt-order manifest of data-backed items. Each entry is
        # `{"modality": <str>, "index": <int>, "placeholder": <str>}`:
        # `modality` is the modality name, `index` is the item's position in
        # `multi_modal_data[modality]`, and `placeholder` is the exact string
        # the input processor splices this item's embedding into. Populated by
        # `add_data` (skipping `is_embedding=True` items — the interleave
        # manifest addresses raw payload only).
        self._item_order: list[dict[str, Union[str, int]]] = []
        self._multimodal_server_config = multimodal_server_config if multimodal_server_config is not None else MultimodalServerConfig(
        )
        # Per-request override merged with the server default at media-load
        # time; see `BaseMediaIO.merge_kwargs` in `inputs/media_io.py`.
        self._request_media_io_kwargs = request_media_io_kwargs

    @property
    def request_media_io_kwargs(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Per-request `media_io_kwargs` override, or `None` if not set."""
        return self._request_media_io_kwargs

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

        if not self._data and not self._embeddings:
            return None, None
        # _data and _embeddings also gathered concurrently
        with _time_input_processing("media_load"):
            data_result, embed_result = await asyncio.gather(
                _retrieve(self._data), _retrieve(self._embeddings))
        return data_result, embed_result

    async def discard_all_async(
        self
    ) -> tuple[Optional[Dict[str, List[Any]]], Optional[Dict[str, List[Any]]]]:
        """Return `(None, None)`, closing the pending loads instead of running them.

        Use instead of `retrieve_all_async` when the caller will not use the
        media: the per-item loads are coroutines `add_data` parked and never
        started, and closing them is what keeps Python from reporting them as
        never awaited. Exactly one of the two may be awaited per tracker.

        Placeholder counts and `item_order` are unaffected; `add_data` builds
        both synchronously.
        """
        for pending in (self._data, self._embeddings):
            for items in pending.values():
                for item in items:
                    if asyncio.iscoroutine(item):
                        item.close()
            pending.clear()
        return None, None

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
        if not is_embedding:
            self._item_order.append({
                "modality": media_type,
                "index": len(self._data[media_type]),
                "placeholder": placeholder,
            })
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

    def item_order(self) -> List[Dict[str, Union[str, int]]]:
        """Prompt-order manifest of data-backed items.

        Each entry is `{"modality": <str>, "index": <int>, "placeholder": <str>}`.
        `index` is the item's position in `multi_modal_data[modality]`;
        `placeholder` is the exact string the input processor will look
        for when splicing this item's encoder embedding.

        Items are recorded in chat-part send order. When the model's chat
        template regroups placeholders by modality (declared via
        `MultimodalPlaceholderMetadata.prompt_modality_order`), sort by
        that priority so the returned list matches what the template
        actually emits — preserving per-modality send order as the tie
        break.
        """
        prompt_order = MULTIMODAL_PLACEHOLDER_REGISTRY.get_prompt_modality_order(
            self._model_type)
        if not prompt_order:
            return list(self._item_order)
        rank = {m: i for i, m in enumerate(prompt_order)}
        return sorted(
            self._item_order,
            key=lambda e:
            (rank.get(e["modality"], len(prompt_order)), e["index"]),
        )


def add_multimodal_placeholders(
    model_type: str,
    text_prompt: str,
    mm_placeholder_counts: dict[str, int],
    item_order: Optional[List[Dict[str, Union[str, int]]]] = None,
) -> str:
    """Add multimodal placeholders to the text prompt.

    Placeholders already in the text are counted and subtracted to
    avoid double-insertion (e.g. when the client already embeds
    `<image>` in the prompt text). When `item_order` is supplied,
    placeholders are emitted in prompt-arrival order (needed for mixed
    modality); otherwise they follow `mm_placeholder_counts` iteration
    order.
    """
    if item_order:
        wanted = [e["placeholder"] for e in item_order]
    else:
        wanted = [
            ph for ph, n in mm_placeholder_counts.items() for _ in range(n)
        ]

    remaining = {ph: text_prompt.count(ph) for ph in set(wanted)}
    placeholders = []
    for ph in wanted:
        if remaining[ph] > 0:
            remaining[ph] -= 1
        else:
            placeholders.append(ph)

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


def _has_python_chat_template(tokenizer: PreTrainedTokenizerBase) -> bool:
    """Return whether a tokenizer overrides Hugging Face's Jinja renderer."""
    apply_chat_template_method = getattr(type(tokenizer), "apply_chat_template",
                                         None)
    return (apply_chat_template_method is not None
            and apply_chat_template_method
            is not PreTrainedTokenizerBase.apply_chat_template)


def uses_tokenizer_native_renderer(tokenizer: Union[TransformersTokenizer,
                                                    TokenizerBase],
                                   processor: ProcessorMixin,
                                   chat_template: Optional[str]) -> bool:
    """Whether `apply_chat_template` renders with the tokenizer's own renderer.

    It does when no Jinja template is declared anywhere (the call, the
    processor or the tokenizer) and the tokenizer overrides
    `apply_chat_template` in Python, as the kimi_k3 checkpoint tokenizer does.
    """
    if isinstance(tokenizer, TransformersTokenizer):
        tokenizer = tokenizer.tokenizer
    return (chat_template is None
            and getattr(processor, "chat_template", None) is None
            and getattr(tokenizer, "chat_template", None) is None
            and _has_python_chat_template(tokenizer))


def resolve_hf_chat_template(
    tokenizer: TokenizerBase,
    processor: ProcessorMixin,
    chat_template: Optional[str],
    tools: Optional[list[dict[str, Any]]],
) -> Optional[str]:
    """Resolve the appropriate chat template to use."""

    # 1. If chat_template is not None, return it
    if chat_template is not None:
        templates = getattr(tokenizer, "chat_template", None)
        if isinstance(templates, dict) and chat_template in templates:
            return templates[chat_template]
        return chat_template

    # 2. Without tools, prefer the processor's chat template when it has one.
    #    A processor without one (some keep the template on their tokenizer
    #    only) falls through to the tokenizer below.
    if not tools and processor:
        processor_template = getattr(processor, "chat_template", None)
        if processor_template is not None:
            return processor_template

    # 3. Use the tokenizer's chat template, which accounts for the tools
    try:
        return tokenizer.get_chat_template(chat_template, tools=tools)
    except Exception:
        logger.warning(
            "Failed to load AutoTokenizer chat template for %s",
            getattr(tokenizer, "name_or_path",
                    type(tokenizer).__name__))
    return None


# Model types whose chats mistral-common renders: the checkpoints the
# Mistral-native format (checkpoint_format "mistral" or "mistral_large_3")
# loads. Their tokenizer is a MistralTokenizer, and the renderer returns
# token ids.
NATIVE_MISTRAL_MODEL_TYPES = frozenset(("mistral_common", ))


def renders_from_loaded_media(model_type: Optional[str]) -> bool:
    """Whether the chat renderer of `model_type` needs the media loaded first.

    Other renderers render the chat while the media loads. mistral-common
    derives each image's token count from the decoded image, so it is handed
    the loaded `mm_data`, and cannot take precomputed multimodal embeddings.
    """
    return model_type in NATIVE_MISTRAL_MODEL_TYPES


class ChatRejectedError(ValueError):
    """The model's own chat renderer rejected the chat as an invalid request.

    Raised for the chats mistral-common renders (Mistral-native checkpoints),
    for example with a reasoning effort the tokenizer file does not declare. A
    distinct type so that a caller that only needs the chat's length (the
    KV-cache-aware router) can leave the request to the server, without
    treating every ValueError raised during rendering the same way.
    """


class _RenderDispatch(NamedTuple):
    """The renderer `apply_chat_template` uses for a set of inputs.

    `kind` is `"mistral_native"` (mistral-common renders the chat),
    `"deepseek_v41"`, `"deepseek"` (the other DeepSeek
    tokenizers), `"passthrough"` (the model skips chat templates),
    `"native"` (a tokenizer-native python template) or `"jinja"`.
    `tokenizer` is the object that renderer is called on, and `template`
    is the resolved source for `"jinja"` (None when no template resolves).
    """
    kind: str
    tokenizer: Any
    template: Optional[str] = None


def _resolve_render_dispatch(
    *,
    model_type: Optional[str],
    tokenizer: Union[TransformersTokenizer, TokenizerBase],
    processor: ProcessorMixin,
    chat_template: Optional[str],
    tools: Optional[list[dict[str, Any]]],
) -> _RenderDispatch:
    """Pick the renderer, in the order `apply_chat_template` checks them."""
    if model_type in NATIVE_MISTRAL_MODEL_TYPES:
        return _RenderDispatch("mistral_native", tokenizer)
    if isinstance(tokenizer, DeepseekV41Tokenizer):
        return _RenderDispatch("deepseek_v41", tokenizer)
    if isinstance(tokenizer, (DeepseekV32Tokenizer, DeepseekV4Tokenizer)):
        return _RenderDispatch("deepseek", tokenizer)
    # Before the tokenizer, processor and template are needed: the registry
    # may already know this model skips chat templates entirely.
    if model_type and MULTIMODAL_PLACEHOLDER_REGISTRY.get_content_format(
            model_type) == ContentFormat.PASSTHROUGH:
        return _RenderDispatch("passthrough", tokenizer)
    if isinstance(tokenizer, TransformersTokenizer):
        # The TokenizerBase renders the template.
        tokenizer = tokenizer.tokenizer
    if uses_tokenizer_native_renderer(tokenizer, processor, chat_template):
        return _RenderDispatch("native", tokenizer)
    return _RenderDispatch(
        "jinja", tokenizer,
        resolve_hf_chat_template(tokenizer, processor, chat_template, tools))


def rendered_hf_chat_template_source(
    *,
    tokenizer: Union[TransformersTokenizer, TokenizerBase],
    processor: ProcessorMixin,
    chat_template: Optional[str] = None,
    tools: Optional[list[dict[str, Any]]] = None,
    model_type: Optional[str] = None,
) -> Optional[str]:
    """The Jinja source `apply_chat_template` will render for these inputs.

    Returns `None` when `apply_chat_template` dispatches to a renderer
    with no Jinja source to analyze (mistral-common for Mistral-native checkpoints,
    the DeepSeek custom tokenizers, a model that skips chat templates, a
    tokenizer-native python template) or when no
    template resolves. Callers that construct chat_template_kwargs use this
    to run the same referenced-kwargs analysis the render-site guard applies,
    so they can avoid injecting a kwarg the active template never reads.
    """
    dispatch = _resolve_render_dispatch(model_type=model_type,
                                        tokenizer=tokenizer,
                                        processor=processor,
                                        chat_template=chat_template,
                                        tools=tools)
    if dispatch.kind != "jinja" or not isinstance(dispatch.template, str):
        return None
    return dispatch.template


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


def _openai_chat_messages(
    conversation: list[ConversationMessage],
    mm_placeholder_counts: list[dict[str, int]],
) -> list[dict[str, Any]]:
    """The conversation as OpenAI messages, each media part in its place."""
    messages = []
    for conv, counts in zip(conversation, mm_placeholder_counts):
        message = {
            key: value
            for key, value in conv.items()
            if key not in ("media", "content_parts")
        }
        if counts or conv.get("content_parts"):
            message["content"] = _build_openai_content(conv, counts)
        messages.append(message)
    return messages


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
    injected_chat_template_kwargs: Optional[Collection[str]] = None,
    mm_data: Optional[Dict[str, Any]] = None,
) -> (str | List[str] | List[int]):
    """Apply chat template to the conversation.

    `injected_chat_template_kwargs` names the keys of `chat_template_kwargs`
    the server derived from API-level fields rather than the caller; the
    unused-kwargs guard exempts them (see `validate_chat_template_kwargs`).

    Uses content-format-driven dispatch:
    - PASSTHROUGH: skip template rendering, just concatenate content strings
    - PYTHON: use a tokenizer-native renderer when no Jinja template is declared
    - OPENAI: reconstructs content as list of dicts for the template to handle
    - STRING: keeps flattened text with pre-inserted placeholders
    """

    dispatch = _resolve_render_dispatch(model_type=model_type,
                                        tokenizer=tokenizer,
                                        processor=processor,
                                        chat_template=chat_template,
                                        tools=tools)

    if dispatch.kind == "mistral_native":
        from mistral_common.exceptions import MistralCommonException

        from tensorrt_llm._torch.models.checkpoints.mistral.tokenizer import \
            MistralTokenizer
        if not isinstance(tokenizer, MistralTokenizer):
            raise ValueError("The Mistral-native format renders chats with "
                             "mistral-common's MistralTokenizer, not "
                             f"{type(tokenizer).__name__}")
        try:
            return tokenizer.render_chat(
                _openai_chat_messages(conversation, mm_placeholder_counts),
                mm_data,
                add_generation_prompt=add_generation_prompt,
                tools=tools,
                documents=documents,
                chat_template=chat_template,
                chat_template_kwargs=chat_template_kwargs)
        except MistralCommonException as error:
            raise ChatRejectedError(str(error)) from error

    if dispatch.kind == "deepseek_v41":
        # The native renderer deep-copies messages; omit pending media coroutines
        # and retain flattened content with its interleaved image placeholders.
        messages = [{
            key: value
            for key, value in message.items()
            if key not in ("media", "content_parts")
        } for message in conversation]
        native_kwargs = dict(chat_template_kwargs or {})
        native_kwargs.update(tokenize=enable_tokenize,
                             add_generation_prompt=add_generation_prompt)
        return tokenizer.apply_chat_template(messages=messages,
                                             tools=tools,
                                             **native_kwargs)

    # Handle DeepSeek tokenizers with custom chat templates.
    if dispatch.kind == "deepseek":
        prompt = tokenizer.apply_chat_template(
            messages=conversation,
            tools=tools,
            **(chat_template_kwargs or {}),
        )
        if enable_tokenize:
            return tokenizer.encode(prompt)
        return prompt

    if dispatch.kind == "passthrough":
        return "".join([conv["content"] for conv in conversation])

    tokenizer = dispatch.tokenizer
    if dispatch.kind == "native":
        native_kwargs = dict(chat_template_kwargs or {})
        if documents is not None:
            native_kwargs["documents"] = documents
        return tokenizer.apply_chat_template(
            conversation,
            tools=tools,
            tokenize=enable_tokenize,
            add_generation_prompt=add_generation_prompt,
            **native_kwargs,
        )

    hf_chat_template = dispatch.template
    if hf_chat_template is None:
        raise ValueError(
            "No chat template found for the given tokenizer and tools.")

    validate_chat_template_kwargs(hf_chat_template,
                                  chat_template_kwargs,
                                  injected_keys=injected_chat_template_kwargs)

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

    template_renderer = tokenizer.apply_chat_template
    if (model_type in ("kimi_k3", "kimi_linear")
            and isinstance(tokenizer, PreTrainedTokenizerBase)
            and callable(getattr(tokenizer, "_encode_chat_segments", None))):
        # Kimi-K3's segment renderer (also used by its kimi_linear text model)
        # ignores Jinja templates. Other overrides, e.g. Kimi-K2.5's, delegate
        # to Jinja after preprocessing tools and must remain in the call path.
        template_renderer = partial(PreTrainedTokenizerBase.apply_chat_template,
                                    tokenizer)

    result = template_renderer(
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


async def async_apply_chat_template(
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
    injected_chat_template_kwargs: Optional[Collection[str]] = None,
    mm_data: Optional[Dict[str, Any]] = None,
) -> (str | List[str] | List[int]):
    """Apply chat template without blocking the event loop."""
    with _time_input_processing("chat_template"):
        return await asyncio.to_thread(
            apply_chat_template,
            model_type=model_type,
            tokenizer=tokenizer,
            processor=processor,
            conversation=conversation,
            add_generation_prompt=add_generation_prompt,
            mm_placeholder_counts=mm_placeholder_counts,
            tools=tools,
            documents=documents,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
            enable_tokenize=enable_tokenize,
            injected_chat_template_kwargs=injected_chat_template_kwargs,
            mm_data=mm_data,
        )


async def async_apply_chat_template_with_mm(mm_coroutines, **kwargs):
    """Resolve native image geometry before encoding its chat token stream."""
    if renders_from_loaded_media(kwargs["model_type"]):
        mm_data, mm_embeddings = await mm_coroutines
        if mm_embeddings:
            raise ValueError(
                "Native Mistral chat requires images, not external embeddings")
        prompt = await async_apply_chat_template(**kwargs, mm_data=mm_data)
        return prompt, (mm_data, mm_embeddings)
    return await asyncio.gather(async_apply_chat_template(**kwargs),
                                mm_coroutines)


def apply_mm_placeholders(
    model_type: str,
    message: ConversationMessage,
    placeholder_counts: Dict[str, int],
    mm_data_tracker: "MultimodalDataTracker",
    *,
    item_order_start: int = 0,
    content_format: Optional[ContentFormat] = None,
) -> None:
    """Insert this message's multimodal placeholders into its text, in place.

    Only `ContentFormat.STRING` templates need them. For `OPENAI`,
    `apply_chat_template` rebuilds `content` from `content_parts` via
    `_build_openai_content`, so pre-inserting here would render every media
    item twice: once as text and once as the template's own content part.

    When the model opts into interleaving and `content_parts` is available the
    placeholders keep their original positions; otherwise they are placed in
    bulk according to the registered placement.
    """
    if not placeholder_counts:
        return

    if content_format is None:
        registry_format = MULTIMODAL_PLACEHOLDER_REGISTRY.get_content_format(
            model_type)
        content_format = (registry_format if registry_format is not None else
                          ContentFormat.STRING)
    if content_format != ContentFormat.STRING:
        return

    content_parts = message.get("content_parts")
    interleave = MULTIMODAL_PLACEHOLDER_REGISTRY.get_interleave_placeholders(
        model_type)
    if content_parts and interleave:
        message["content"] = interleave_mm_placeholders(
            model_type, content_parts, placeholder_counts,
            mm_data_tracker.placeholder_modalities())
    else:
        message["content"] = add_multimodal_placeholders(
            model_type,
            message["content"],
            placeholder_counts,
            item_order=mm_data_tracker.item_order()[item_order_start:],
        )


async def async_build_multimodal_prompt(
    *,
    model_type: str,
    tokenizer: Union[TransformersTokenizer, TokenizerBase],
    processor: ProcessorMixin,
    prompt: str,
    media: List[str],
    modality: str = "image",
    add_generation_prompt: bool = True,
    chat_template: Optional[str] = None,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
    media_io_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Package one text prompt plus media URLs into engine inputs.

    For callers whose request format is a plain prompt and a list of media
    references rather than OpenAI chat messages -- the Triton `llmapi` backend,
    for example. Runs the same steps `trtllm-serve` runs for
    `v1/chat/completions`: resolve placeholders for the model's content format,
    render the chat template, and attach the fetched media.

    The media fetches and the template rendering are awaited together, so
    several items on one request are loaded concurrently.

    Args:
        model_type: Top-level HF `model_type`, the multimodal registry key.
        prompt: The user's text, without any placeholder tokens.
        media: URLs, `data:` URIs or local paths, one per item.
        modality: Registered media modality, e.g. `"image"`.

    Returns:
        A `TextPrompt` carrying `multi_modal_data`, ready for
        `LLM.generate_async`.
    """
    media_io_cls = MEDIA_IO_REGISTRY.get(modality)
    if media_io_cls is None:
        raise ValueError(f"Unsupported modality {modality!r}. "
                         f"Registered modalities: {list(MEDIA_IO_REGISTRY)}")
    media_io = media_io_cls.create((media_io_kwargs or {}).get(modality), None)

    mm_data_tracker = MultimodalDataTracker(model_type)
    content_parts: List[Union[str, Dict[str, Any]]] = []
    for index, item in enumerate(media):
        # `add_data` takes the un-awaited fetch; they run concurrently below.
        mm_data_tracker.add_data(modality, media_io.async_load(item))
        content_parts.append({"type": modality, "media_index": index})
    content_parts.append(prompt)

    message = ConversationMessage(role="user",
                                  content=prompt,
                                  media=[],
                                  content_parts=content_parts)
    placeholder_counts = mm_data_tracker.placeholder_counts()
    apply_mm_placeholders(model_type, message, placeholder_counts,
                          mm_data_tracker)

    rendered, (mm_data, _) = await async_apply_chat_template_with_mm(
        mm_data_tracker.retrieve_all_async(),
        model_type=model_type,
        tokenizer=tokenizer,
        processor=processor,
        conversation=[message],
        add_generation_prompt=add_generation_prompt,
        mm_placeholder_counts=[placeholder_counts],
        chat_template=chat_template,
        chat_template_kwargs=chat_template_kwargs,
    )

    inputs = prompt_inputs(rendered)
    if mm_data:
        inputs["multi_modal_data"] = mm_data
        inputs["chat_template_applied"] = True
        item_order = mm_data_tracker.item_order()
        if item_order:
            inputs["mm_item_order"] = item_order
    return inputs


def default_multimodal_input_loader(
    *,
    tokenizer: Optional[Union[TransformersTokenizer, TokenizerBase]],
    model_dir: str,
    model_type: str,
    modality: str,
    prompts: List[str],
    media: Optional[Union[List[str], List[List[str]]]] = None,
    image_data_format: Optional[str] = None,
    num_frames: int = 8,
    mm_embeddings: Optional[Union[List[torch.Tensor],
                                  List[List[torch.Tensor]]]] = None,
    device: str = "cpu",
    extract_audio: bool = False,
    trust_remote_code: bool = True,
) -> List[dict[str, Union[str, torch.Tensor]]]:

    format_kwargs = {
        "format": image_data_format
    } if image_data_format is not None else {}

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
                    mm, device=device, **format_kwargs)

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
                                    device=device,
                                    extract_audio=extract_audio,
                                    **format_kwargs),
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
                        data = load_image(m, device=device, **format_kwargs)
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
                            data=load_image(m, device=device, **format_kwargs),
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

    if tokenizer is None and model_type in NATIVE_MISTRAL_MODEL_TYPES:
        from tensorrt_llm._torch.models.checkpoints.mistral.tokenizer import \
            MistralTokenizer
        tokenizer = MistralTokenizer.from_pretrained(model_dir)
    elif tokenizer is None and not is_passthrough:
        tokenizer = ModelLoader.load_hf_tokenizer(
            model_dir, trust_remote_code=trust_remote_code, use_fast=True)

    processor = None
    if not is_passthrough and model_type not in NATIVE_MISTRAL_MODEL_TYPES:
        processor = AutoProcessor.from_pretrained(
            model_dir, use_fast=True, trust_remote_code=trust_remote_code)

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
        if mm_placeholder_counts and model_type not in NATIVE_MISTRAL_MODEL_TYPES:
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
        native_mm_data = None
        if renders_from_loaded_media(model_type):
            if is_embedding:
                raise ValueError(
                    "Native Mistral chat requires images, not external embeddings"
                )
            native_mm_data, _ = mm_data_tracker.retrieve_all_sync()
            # As the input processor does for a text prompt, the images go
            # where the prompt has their placeholders, one per image, or
            # after it when it has none.
            for placeholder, count in mm_placeholder_counts.items():
                if placeholder in conv["content"]:
                    parts = conv["content"].split(placeholder)
                    if len(parts) != count + 1:
                        raise ValueError(
                            f"The prompt has {len(parts) - 1} {placeholder} "
                            f"placeholders for {count} images.")
                    content_parts = []
                    for index, part in enumerate(parts):
                        if part:
                            content_parts.append(part)
                        if index < count:
                            content_parts.append({
                                "type": "image",
                                "media_index": index
                            })
                    conv["content"] = "".join(parts)
                    conv["content_parts"] = content_parts
        prompt = apply_chat_template(
            model_type=model_type,
            tokenizer=tokenizer,
            processor=processor,
            conversation=[conv],
            add_generation_prompt=True,
            mm_placeholder_counts=[mm_placeholder_counts],
            mm_data=native_mm_data)
        input = prompt_inputs(prompt)

        if mm_placeholder_counts:
            input["chat_template_applied"] = True
            if mm_embeddings is not None:
                _, input[
                    "multi_modal_embeddings"] = mm_data_tracker.retrieve_all_sync(
                    )
            else:
                if native_mm_data is not None:
                    input["multi_modal_data"] = native_mm_data
                else:
                    input[
                        "multi_modal_data"], _ = mm_data_tracker.retrieve_all_sync(
                        )
            item_order = mm_data_tracker.item_order()
            if item_order:
                input["mm_item_order"] = item_order
        inputs.append(input)

    return inputs

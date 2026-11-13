# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multipart parsing and audio decoding for short Whisper transcriptions."""

from __future__ import annotations

import asyncio
import io
import math
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.datastructures import UploadFile

if TYPE_CHECKING:
    from transformers import WhisperConfig, WhisperProcessor

    from tensorrt_llm.inputs.data import TextPrompt

MAX_AUDIO_UPLOAD_BYTES = 25 * 1024 * 1024


class TranscriptionOptions(BaseModel):
    """Supported form fields for the short-clip transcription endpoint."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, description="Served Whisper model identifier.")
    language: Optional[str] = Field(
        default=None, description="Spoken language; omitted uses the checkpoint's English default."
    )
    temperature: float = Field(
        default=0.0, ge=0.0, le=1.0, allow_inf_nan=False, description="Sampling temperature."
    )
    response_format: Literal["json", "text"] = Field(
        default="json", description="Transcription response format."
    )
    prompt: str = Field(default="", description="Only an empty prompt is currently supported.")
    stream: bool = Field(default=False, description="Only non-streaming output is supported.")

    @field_validator("prompt")
    @classmethod
    def _reject_prompt(cls, value: str) -> str:
        if value:
            raise ValueError("Prompt conditioning is not supported for audio transcriptions yet.")
        return value

    @field_validator("stream")
    @classmethod
    def _reject_streaming(cls, value: bool) -> bool:
        if value:
            raise ValueError("Streaming audio transcriptions are not supported yet.")
        return value


def _decode_audio(data: bytes, sampling_rate: int, max_samples: int) -> NDArray[np.float32]:
    # Importing audio codecs/scipy is unnecessary for servers that never transcribe.
    import soundfile as sf
    from scipy.signal import resample_poly

    try:
        with sf.SoundFile(io.BytesIO(data)) as audio_file:
            source_rate = audio_file.samplerate
            if audio_file.frames == 0:
                raise ValueError("The audio file contains no samples.")
            if audio_file.frames * sampling_rate > max_samples * source_rate:
                raise ValueError(
                    f"Audio duration exceeds the model's {max_samples / sampling_rate:g}s window. "
                    "Chunk the input on the client side."
                )
            waveform = audio_file.read(dtype="float32", always_2d=True).mean(axis=1)
    except (sf.LibsndfileError, sf.SoundFileRuntimeError) as exc:
        raise ValueError("Cannot decode audio: upload a supported, nonempty audio file.") from exc
    if not np.isfinite(waveform).all():
        raise ValueError("The audio file contains nonfinite samples.")
    if source_rate != sampling_rate:
        divisor = math.gcd(source_rate, sampling_rate)
        waveform = resample_poly(waveform, sampling_rate // divisor, source_rate // divisor)
    return np.ascontiguousarray(waveform, dtype=np.float32)


async def parse_transcription_request(
    request: Request, *, served_model: str, sampling_rate: int, max_samples: int
) -> tuple[TranscriptionOptions, NDArray[np.float32]]:
    """Validate one multipart upload and decode mono audio at the model's rate.

    Decoding runs off the event loop. The model window is checked from file
    metadata before allocating the decoded waveform.
    """
    if request.headers.get("content-type", "").split(";", 1)[0] != "multipart/form-data":
        raise ValueError("Audio transcriptions require multipart/form-data with a file upload.")
    async with request.form(max_files=1, max_fields=16) as form:
        fields: dict[str, str] = {}
        upload = None
        for name, value in form.multi_items():
            if name == "file":
                if not isinstance(value, UploadFile):
                    raise ValueError("'file' must be an uploaded audio file.")
                upload = value
            else:
                if name in fields or not isinstance(value, str):
                    raise ValueError(f"Expected one text value for {name!r}.")
                fields[name] = value
        options = TranscriptionOptions.model_validate(fields)
        if options.model != served_model:
            raise HTTPException(status_code=404, detail=f"Model {options.model!r} is not served.")
        if upload is None:
            raise ValueError("A 'file' audio upload is required.")
        data = await upload.read(MAX_AUDIO_UPLOAD_BYTES + 1)
        if len(data) > MAX_AUDIO_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Audio upload exceeds the 25 MiB limit.")
    waveform = await asyncio.to_thread(_decode_audio, data, sampling_rate, max_samples)
    return options, waveform


def transcription_response(text: str, response_format: Literal["json", "text"]) -> Response:
    """Render decoded text in the requested OpenAI-compatible response format."""
    if response_format == "text":
        return PlainTextResponse(text)
    return JSONResponse({"text": text})


async def create_transcription_response(
    request: Request,
    *,
    served_model: str,
    model_config: Optional[WhisperConfig],
    processor: Optional[WhisperProcessor],
    generate: Callable[[TextPrompt, float, Request], Awaitable[str]],
) -> Response:
    """Convert one short audio request into Whisper inputs and render the result.

    ``generate`` submits the inputs through the serving engine. Keeping this
    conversion independent of the engine permits CPU request-contract tests.
    """
    if model_config is None or model_config.model_type != "whisper":
        raise ValueError("Audio transcriptions require a Whisper model.")
    if processor is None:
        raise RuntimeError("The Whisper processor is unavailable.")
    extractor = processor.feature_extractor
    options, waveform = await parse_transcription_request(
        request,
        served_model=served_model,
        sampling_rate=int(extractor.sampling_rate),
        max_samples=int(extractor.n_samples),
    )
    prompt_text = ""
    if options.language is not None:
        try:
            forced = processor.get_decoder_prompt_ids(
                language=options.language, task="transcribe", no_timestamps=True
            )
        except ValueError:
            # English-only checkpoints have no language/task tokens.
            if options.language != "en":
                raise
            forced = processor.get_decoder_prompt_ids(no_timestamps=True)
        prefix = [int(model_config.decoder_start_token_id)]
        prefix.extend(int(token) for _, token in sorted(forced))
        prompt_text = processor.tokenizer.decode(prefix, skip_special_tokens=False)
    inputs: TextPrompt = {
        "prompt": prompt_text,
        "multi_modal_data": {"audio": [(waveform, int(extractor.sampling_rate))]},
    }
    text = await generate(inputs, options.temperature, request)
    return transcription_response(text, options.response_format)

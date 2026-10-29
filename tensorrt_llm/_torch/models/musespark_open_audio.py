# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The MuseSparkOpen waveform ("wav_ae") audio tower and its token accounting.

Plain PyTorch on purpose, and importable without the rest of `tensorrt_llm`:
the module hierarchy and parameter names mirror the checkpoint suffixes under
`modalities.soft_audio_perception.interaction_encoder.` one for one
(`stages.{s}.0.linear` / `stages.{s}.0.norm` for the patchifiers,
`stages.{s}.{1,2}.{conv.conv,norm,linear1,linear2}` for the conv/MLP blocks,
then `norm` / `o_proj` / `dual_norm`), so the weight mapper can walk
`state_dict()` and prepend the checkpoint prefix. The geometry is fixed in
code rather than read from the config (the config's
`audio_perception_modality_args` names only the encoder family and the input
mode), so a checkpoint with different patchify factors fails at load with a
shape mismatch rather than adapting.

The tower consumes the RAW mono 24 kHz float waveform: no mel, no STFT, no
amplitude normalization. Four cascaded patchify stages divide the time axis by
16 * 6 * 5 * 4 = 1920, so one latent token covers 1920 samples = 80 ms, and the
tail is zero-padded up to a multiple of 1920 -- which is why the token count is
`ceil(num_samples / 1920)`, computed on the resampled, mono waveform.

The output is already in decoder embedding width: `o_proj` maps
2048 -> hidden_size (8192) inside the tower and there is no separate audio
projector. Rows come out of `dual_norm` un-normed by the decoder's
`embed_norm`; the decoder applies that weightless norm to the merged
embedding tensor exactly once, text and audio alike. RMSNorm is per-row, so
this equals norming the audio rows before the merge.

QUANTIZATION: this tower is BF16 and is never quantized. The checkpoint stores
all 72 audio tensors in BF16 with no quantization scales, but
`hf_quant_config.json`'s `exclude_modules` does not name any audio prefix,
so a quantization plan derived from the checkpoint metadata alone would put
the tower in scope for NVFP4. The exemption is structural instead: every layer
is a plain `nn.Linear` / `nn.Conv1d` built without a quant_config, which
TRT-LLM's quantization plumbing (applied through its own Linear modules) does
not reach.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "AUDIO_SAMPLES_PER_TOKEN",
    "AUDIO_TARGET_SAMPLE_RATE",
    "AUDIO_TOKEN_IDS",
    "AUDIO_TOKEN_NAMES",
    "MuseSparkOpenAudioEncoder",
    "audio_placeholder_ids",
    "num_audio_tokens",
    "resampled_num_samples",
    "to_mono_24k",
    "verify_audio_token_ids",
]

#: The tower's fixed input rate. Everything is resampled to this first; the
#: token count is defined on the resampled waveform.
AUDIO_TARGET_SAMPLE_RATE = 24_000

#: One latent (decoder) token per this many samples at 24 kHz = 80 ms. Equal to
#: the product of the patchify factors (16 * 6 * 5 * 4), verified against the
#: checkpoint's patchifier shapes.
AUDIO_SAMPLES_PER_TOKEN = 1_920

#: Token names the expansion is defined in terms of. The ids are resolved by
#: vocab name lookup, never from the config: the `next_audio_token` /
#: `empty_audio_token` ids in `config.json` take no part in inference.
AUDIO_TOKEN_NAMES: Dict[str, str] = {
    "audio": "<|audio|>",
    "soft_audio_start": "<|soft_audio_start|>",
    "soft_audio_end": "<|soft_audio_end|>",
}

#: The name lookups above, resolved against the shipped tokenizer and
#: pinned here in the same style as `VISION_TOKEN_IDS`. `<|audio|>` is the
#: repeated placeholder whose positions the merge overwrites; the two
#: delimiters keep their learned text embeddings and are never overwritten.
#: :func:`verify_audio_token_ids` re-derives these from a live vocab so a
#: tokenizer drift fails loudly instead of splicing rows onto wrong positions.
AUDIO_TOKEN_IDS: Dict[str, int] = {
    "audio": 200038,
    "soft_audio_start": 201463,
    "soft_audio_end": 201464,
}


def num_audio_tokens(num_samples: int) -> int:
    """Decoder tokens one clip occupies: `ceil(num_samples / 1920)`.

    `num_samples` is measured AFTER the resample to 24 kHz and AFTER the
    mono downmix. Rounds up: the encoder zero-pads the tail to a multiple of
    1920, so a 1-sample clip is 1 token and 1921 samples are 2. Excludes the
    two delimiter tokens, which carry no embeddings.
    """
    if num_samples <= 0:
        raise ValueError(f"audio clip has {num_samples} samples; an empty clip cannot be encoded")
    return math.ceil(num_samples / AUDIO_SAMPLES_PER_TOKEN)


def resampled_num_samples(num_samples: int, sample_rate: int) -> int:
    """Length a clip will have after the resample to 24 kHz.

    The output length of `librosa.resample(..., fix=True)`, which is what
    :func:`to_mono_24k` uses, computed with librosa's own float operations:
    the ratio first, then `ceil(n * ratio)`. Multiplying before dividing
    differs at some lengths (88,200 samples at 44.1 kHz resample to 48,001,
    not 48,000), and then the placeholder count would disagree with the
    waveform the encoder sees.
    """
    if sample_rate == AUDIO_TARGET_SAMPLE_RATE:
        return num_samples
    ratio = float(AUDIO_TARGET_SAMPLE_RATE) / sample_rate
    return math.ceil(num_samples * ratio)


def audio_placeholder_ids(count: int) -> List[int]:
    """The id run one clip expands to: start + `count` fillers + end.

    Only the interior `<|audio|>` ids are overwritten with encoder rows; the
    delimiters stay ordinary text embeddings.
    """
    if count <= 0:
        raise ValueError(f"an audio expansion needs at least one token, got {count}")
    return (
        [AUDIO_TOKEN_IDS["soft_audio_start"]]
        + [AUDIO_TOKEN_IDS["audio"]] * count
        + [AUDIO_TOKEN_IDS["soft_audio_end"]]
    )


def tokenizer_vocab(tokenizer: Any) -> Dict[str, int]:
    """Vocab of a live tokenizer, unwrapping TRT-LLM's tokenizer wrapper.

    The served path hands the input processor a `TransformersTokenizer`
    (or its musespark_open subclass, via `--custom_tokenizer`), which
    inherits `PreTrainedTokenizerBase.get_vocab()` -- a stub that raises
    `NotImplementedError`. The real vocab lives on the wrapped HF
    tokenizer at `.tokenizer`. Try the object itself first (an offline caller
    passes the HF tokenizer directly), then the wrapped one.
    """
    inner = getattr(tokenizer, "tokenizer", None)
    for obj in (tokenizer, inner):
        if obj is None or not hasattr(obj, "get_vocab"):
            continue
        try:
            return obj.get_vocab()
        except NotImplementedError:
            continue
    raise TypeError(
        f"cannot resolve a vocab from tokenizer {type(tokenizer).__name__}: "
        "get_vocab() is unimplemented on it and on its wrapped tokenizer"
    )


def verify_audio_token_ids(vocab: Dict[str, int]) -> None:
    """Check the pinned ids against a live tokenizer vocab, by token name.

    Raises when a name is missing (the published OSS tokenizer leaves the
    soft-audio delimiters as reserved placeholders under other names) or when
    the vocab disagrees with the pinned id -- either way, splicing would put
    encoder rows on the wrong positions, so this must be an error, not a log.
    """
    for role, name in AUDIO_TOKEN_NAMES.items():
        resolved = vocab.get(name)
        if resolved is None:
            raise ValueError(
                f"tokenizer has no {name!r} token: this tokenizer cannot serve audio "
                "(the soft-audio delimiters are reserved-but-unnamed in the published "
                "OSS tokenizer; serve the checkpoint's own tokenizer)"
            )
        if int(resolved) != AUDIO_TOKEN_IDS[role]:
            raise ValueError(
                f"tokenizer resolves {name!r} to id {resolved}, but the audio path "
                f"is pinned to {AUDIO_TOKEN_IDS[role]}; refusing to splice audio "
                "rows onto positions the tokenizer disagrees about"
            )


def to_mono_24k(audio: Any, sample_rate: Optional[int] = None) -> np.ndarray:
    """Normalize one audio item to a 1-D float32 mono waveform at 24 kHz.

    Casts to float32, downmixes multi-channel audio by the channel mean, then
    resamples. A bare array with no rate is taken to be 24 kHz already, as the
    Nemotron audio path treats bare arrays; a wrong rate there produces
    wrong-duration token counts, so callers that know the rate must pass it.

    The channel axis is taken to be the shorter axis of a 2-D input: soundfile
    (the loader behind `AudioMediaIO`) returns `(frames, channels)` while
    other callers pass channels-first `(channels, frames)`, and any real clip
    is longer than its channel count. Ties (a clip as short as its channel
    count) resolve channels-first.
    """
    if isinstance(audio, tuple) and len(audio) == 2:
        audio, sample_rate = audio
    elif isinstance(audio, dict):
        sample_rate = audio.get("sampling_rate", audio.get("sample_rate", sample_rate))
        audio = audio.get("array", audio.get("audio", audio.get("waveform")))
        if audio is None:
            raise ValueError("audio dict carries no 'array'/'audio'/'waveform' entry")

    if isinstance(audio, torch.Tensor):
        array = audio.detach().cpu().to(torch.float32).numpy()
    else:
        array = np.asarray(audio, dtype=np.float32)

    if array.ndim == 2:
        channel_axis = 0 if array.shape[0] <= array.shape[1] else 1
        array = array.mean(axis=channel_axis)
    elif array.ndim != 1:
        raise ValueError(f"audio must be 1-D or 2-D, got shape {tuple(array.shape)}")
    array = np.ascontiguousarray(array, dtype=np.float32)

    if array.size == 0:
        raise ValueError("audio clip is empty; an empty clip cannot be encoded")

    if sample_rate is not None and int(sample_rate) != AUDIO_TARGET_SAMPLE_RATE:
        import librosa  # noqa: PLC0415 -- optional dep, only needed off-rate

        # fix=True guarantees the output length is ceil(n * target / orig),
        # keeping this in lockstep with resampled_num_samples().
        array = librosa.resample(
            array,
            orig_sr=int(sample_rate),
            target_sr=AUDIO_TARGET_SAMPLE_RATE,
            fix=True,
        ).astype(np.float32)

    return array


class MuseSparkOpenAudioRMSNorm(nn.Module):
    """Weighted RMSNorm with eps 1e-6: normalize in fp32, downcast, then scale.

    The eps is the audio tower's training value, not the decoder's
    `rms_norm_eps` (1e-5). The ordering matches the reference outputs:
    upcast to fp32, normalize, downcast to the input dtype, then multiply by
    the (bf16) weight. `F.rms_norm` multiplies the weight in fp32 before the
    downcast instead; the two differ by at most 1 bf16 ulp.
    """

    def __init__(
        self, hidden_size: int, eps: float = 1e-6, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        orig_dtype = value.dtype
        value = value.to(torch.float32)
        value = value * torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return value.to(orig_dtype) * self.weight


def _patchify(value: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Group `patch_size` consecutive timesteps into the channel axis."""
    batch, length, channels = value.shape
    if length % patch_size:
        raise ValueError(f"audio length {length} is not divisible by {patch_size}")
    return value.reshape(batch, length // patch_size, patch_size * channels)


class _Patchifier(nn.Module):
    """Patchify -> Linear -> RMSNorm. Checkpoint children: `linear`, `norm`."""

    def __init__(
        self, patch_size: int, input_size: int, output_size: int, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.linear = nn.Linear(input_size * patch_size, output_size, bias=True, dtype=dtype)
        self.norm = MuseSparkOpenAudioRMSNorm(output_size, dtype=dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(_patchify(value, self.patch_size)))


class _DepthwiseCausalConv(nn.Module):
    """Depthwise causal conv1d; causal via an explicit left zero-pad.

    Checkpoint child: `conv` (hence the `conv.conv` suffix). Offline
    encoding only: there is no streaming cache.
    """

    def __init__(self, hidden_size: int, kernel_size: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=True, dtype=dtype
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.transpose(1, 2)  # [B, T, D] -> [B, D, T]
        value = F.pad(value, (self.kernel_size - 1, 0))
        # `.contiguous()` is numeric, not cosmetic: the RMSNorm that follows
        # reduces over the last axis, and its accumulation order differs on a
        # non-contiguous transpose view by 1-2 bf16 ulp per block, which
        # compounds across the tower.
        return self.conv(value).transpose(1, 2).contiguous()


class _Block(nn.Module):
    """conv -> RMSNorm -> Linear -> GELU -> Linear, with a residual.

    Checkpoint children: `conv`, `norm`, `linear1`, `linear2`. The
    norm sits after the conv and before the MLP, and the residual spans the
    whole block.
    """

    def __init__(
        self, hidden_size: int, intermediate_size: int, kernel_size: int, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.conv = _DepthwiseCausalConv(hidden_size, kernel_size, dtype)
        self.norm = MuseSparkOpenAudioRMSNorm(hidden_size, dtype=dtype)
        self.linear1 = nn.Linear(hidden_size, intermediate_size, bias=True, dtype=dtype)
        self.linear2 = nn.Linear(intermediate_size, hidden_size, bias=True, dtype=dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.conv(value)
        value = self.linear2(F.gelu(self.linear1(self.norm(value))))
        return value + residual


class MuseSparkOpenAudioEncoder(nn.Module):
    """The `wav_ae` tower: mono 24 kHz waveform -> one 8192-wide row / 1920 samples.

    Geometry is fixed (see the module docstring); only the output width -- the
    decoder hidden size -- comes from the config. Built in BF16 by default,
    matching the checkpoint's storage dtype; see the module docstring for why
    it is exempt from quantization. Every layer is created in its dtype rather
    than converted after construction, so the model can be initialized on the
    meta device.
    """

    def __init__(
        self,
        output_size: int,
        patch_size: int = 1920,
        kernel_size: int = 7,
        input_channels: int = 1,
        patchify_sizes: Sequence[int] = (16, 6, 5, 4),
        hidden_sizes: Sequence[int] = (256, 512, 1024, 2048),
        layers_per_stage: Sequence[int] = (2, 2, 2, 2),
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if not (len(patchify_sizes) == len(hidden_sizes) == len(layers_per_stage)):
            raise ValueError("audio stage settings must have equal lengths")
        if patch_size != math.prod(patchify_sizes):
            raise ValueError("patch_size must equal the product of patchify_sizes")

        self.output_size = output_size
        self.patch_size = patch_size
        self.input_channels = input_channels

        stages: List[nn.Sequential] = []
        stage_input = input_channels
        for patch, hidden, layer_count in zip(patchify_sizes, hidden_sizes, layers_per_stage):
            modules: List[nn.Module] = [_Patchifier(patch, stage_input, hidden, dtype)]
            # The 4x intermediate width matches the checkpoint's linear1 shapes.
            modules.extend(
                _Block(hidden, hidden * 4, kernel_size, dtype) for _ in range(layer_count)
            )
            stages.append(nn.Sequential(*modules))
            stage_input = hidden
        self.stages = nn.Sequential(*stages)
        self.norm = MuseSparkOpenAudioRMSNorm(stage_input, dtype=dtype)
        self.o_proj = nn.Linear(stage_input, output_size, bias=True, dtype=dtype)
        self.dual_norm = MuseSparkOpenAudioRMSNorm(output_size, dtype=dtype)

    @property
    def dtype(self) -> torch.dtype:
        return self.o_proj.weight.dtype

    def _encode(self, value: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            for module in stage:
                value = module(value)
        return self.dual_norm(self.o_proj(self.norm(value)))

    def forward(self, waveforms: List[torch.Tensor]) -> List[torch.Tensor]:
        """Encode a list of 1-D waveforms; each output is `[ceil(N/1920), output_size]`.

        A per-clip loop: clips have unequal lengths and each is padded
        independently to its own multiple of 1920.
        The float32 waveform is cast to the tower's dtype (BF16) here.
        """
        outputs: List[torch.Tensor] = []
        for waveform in waveforms:
            waveform = waveform.to(dtype=self.dtype).reshape(-1, self.input_channels)
            padding = (-waveform.shape[0]) % self.patch_size
            waveform = F.pad(waveform, (0, 0, 0, padding)).unsqueeze(0)  # [1, T, C]
            outputs.append(self._encode(waveform).squeeze(0))
        return outputs


def audio_state_dict_to_checkpoint_keys(
    encoder: MuseSparkOpenAudioEncoder, prefix: str
) -> List[Tuple[str, str]]:
    """`(parameter_path, checkpoint_key)` pairs for every tower tensor.

    The module tree mirrors the checkpoint suffixes exactly, so the mapping is
    a prefix concatenation; kept as a function so that every user of the
    mapping derives it the same way.
    """
    return [(name, f"{prefix}{name}") for name in encoder.state_dict()]

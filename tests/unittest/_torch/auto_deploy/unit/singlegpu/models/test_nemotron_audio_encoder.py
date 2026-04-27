# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor

from tensorrt_llm._torch.auto_deploy.models.custom.modeling_nemotron_audio_encoder import (
    NemotronAudioEncoder,
)
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_nemotron_nano_omni import (
    NemotronNanoOmniADInputProcessor,
    NemotronNanoOmniForConditionalGeneration,
)
from tensorrt_llm._torch.models.modeling_parakeet import ProjectedParakeet
from tensorrt_llm.inputs.content_format import ContentFormat
from tensorrt_llm.sampling_params import SamplingParams

MODEL_ID = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning"
PROMPT_TEXT = "Transcribe this audio verbatim. Output only the transcription."
REPO_ROOT = Path(__file__).resolve().parents[7]
AUDIO_PATH = REPO_ROOT / "en-US-GuyNeural.mp3"


def _small_sound_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        conv_kernel_size=3,
        subsampling_factor=8,
        subsampling_conv_channels=32,
        subsampling_conv_kernel_size=3,
        subsampling_conv_stride=2,
        num_mel_bins=8,
        projection_hidden_size=48,
    )


class _BaseProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = SimpleNamespace(tokenizer=tokenizer)
        self.content_format = ContentFormat.OPENAI

    def __call__(self, inputs, sampling_params):
        del sampling_params
        prompt = inputs.get("prompt", "")
        return self.tokenizer.tokenizer.encode(prompt, add_special_tokens=False), {}


def _native_encode_audio_data(
    sound_encoder: ProjectedParakeet,
    input_audio_features: torch.Tensor,
    feature_attention_mask: torch.Tensor,
    audio_num_clips: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror native Nemotron `_encode_audio_data(...)` semantics."""
    sound_embeds = sound_encoder(input_audio_features, feature_attention_mask)
    valid_input_lens = feature_attention_mask.sum(dim=1)
    valid_output_lens = sound_encoder.encoder._get_subsampling_output_length(valid_input_lens)

    truncated = []
    for i in range(sound_embeds.shape[0]):
        valid_len = int(valid_output_lens[i].item())
        truncated.append(sound_embeds[i, :valid_len])

    if audio_num_clips is None:
        flat = torch.cat(truncated, dim=0)
        return sound_embeds, valid_output_lens.to(torch.int32), flat

    per_audio = []
    clip_offset = 0
    for num_clips in audio_num_clips.tolist():
        num_clips = int(num_clips)
        per_audio.append(torch.cat(truncated[clip_offset : clip_offset + num_clips], dim=0))
        clip_offset += num_clips

    flat = torch.cat(per_audio, dim=0)
    return sound_embeds, valid_output_lens.to(torch.int32), flat


def _load_real_sound_weights(snapshot_dir: Path) -> dict[str, torch.Tensor]:
    index = json.loads((snapshot_dir / "model.safetensors.index.json").read_text())
    needed: dict[str, list[str]] = {}
    for full_key, shard in index["weight_map"].items():
        if full_key.startswith("sound_encoder.") or full_key.startswith("sound_projection."):
            needed.setdefault(shard, []).append(full_key)

    state_dict = {}
    for shard, keys in needed.items():
        with safe_open(snapshot_dir / shard, framework="pt", device="cpu") as handle:
            for full_key in keys:
                state_dict[full_key] = handle.get_tensor(full_key)
    return state_dict


def _build_real_audio_inputs(config, snapshot_dir: Path):
    processor = AutoProcessor.from_pretrained(
        snapshot_dir,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    base = _BaseProcessor(processor.tokenizer)
    input_processor = NemotronNanoOmniADInputProcessor(base, processor, config)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(AUDIO_PATH)},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }
    ]
    _, mm_inputs = input_processor({"messages": messages}, SamplingParams(max_tokens=8))
    mm_data = mm_inputs["multimodal_data"]
    return (
        mm_data["input_audio_features"],
        mm_data["feature_attention_mask"],
        mm_data["audio_num_clips"],
    )


def _maybe_get_real_snapshot_dir() -> Path | None:
    try:
        return Path(snapshot_download(MODEL_ID, local_files_only=True))
    except LocalEntryNotFoundError:
        return None


def _load_real_encoder_modules(device: torch.device, dtype: torch.dtype):
    if not AUDIO_PATH.is_file():
        pytest.skip(f"Real audio fixture not found: {AUDIO_PATH}")

    snapshot_dir = _maybe_get_real_snapshot_dir()
    if snapshot_dir is None:
        pytest.skip(f"Local snapshot for {MODEL_ID} is not available")

    config = AutoConfig.from_pretrained(snapshot_dir, trust_remote_code=True)
    sound_weights = _load_real_sound_weights(snapshot_dir)

    native = (
        ProjectedParakeet(
            config.sound_config,
            llm_hidden_size=config.llm_config.hidden_size,
            dtype=dtype,
        )
        .to(device)
        .eval()
    )
    native.load_weights(sound_weights)

    remapped_weights = dict(sound_weights)
    NemotronNanoOmniForConditionalGeneration._remap_sound_weight_keys(remapped_weights, prefix="")
    remap_state_dict = {
        key[len("sound_encoder.") :]: value
        for key, value in remapped_weights.items()
        if key.startswith("sound_encoder.")
    }
    ad_encoder = (
        NemotronAudioEncoder(
            config.sound_config,
            llm_hidden_size=config.llm_config.hidden_size,
            dtype=dtype,
        )
        .to(device)
        .eval()
    )
    missing, unexpected = ad_encoder.load_state_dict(remap_state_dict, strict=False)
    filtered_missing = [
        key
        for key in missing
        if not key.startswith("encoder.feature_extractor.")
        and not ("_conv" in key and key.endswith(".bias"))
        and not (".conv." in key and key.endswith(".bias"))
    ]
    if filtered_missing or unexpected:
        raise RuntimeError(
            f"Unexpected load mismatch: missing={filtered_missing[:16]}, unexpected={unexpected[:16]}"
        )

    input_audio_features, feature_attention_mask, audio_num_clips = _build_real_audio_inputs(
        config,
        snapshot_dir,
    )
    return (
        native,
        ad_encoder,
        input_audio_features.to(device=device, dtype=dtype),
        feature_attention_mask.to(device=device),
        audio_num_clips.to(device=device),
    )


@torch.no_grad()
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for ProjectedParakeet RMSNorm"
)
def test_eager_nemotron_audio_encoder_matches_real_weights_and_inputs():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    (
        native_encoder,
        ad_encoder,
        input_audio_features,
        feature_attention_mask,
        audio_num_clips,
    ) = _load_real_encoder_modules(device, dtype)

    ref_padded, ref_valid_lens, ref_flat = _native_encode_audio_data(
        native_encoder,
        input_audio_features,
        feature_attention_mask,
        audio_num_clips=audio_num_clips,
    )
    test_padded, test_valid_lens = ad_encoder(
        input_audio_features=input_audio_features,
        feature_attention_mask=feature_attention_mask,
    )
    test_flat = NemotronAudioEncoder.flatten_valid_outputs(test_padded, test_valid_lens)

    assert not torch.isnan(ref_padded).any()
    assert not torch.isnan(test_padded).any()
    torch.testing.assert_close(test_padded, ref_padded)
    torch.testing.assert_close(test_valid_lens, ref_valid_lens)
    torch.testing.assert_close(test_flat, ref_flat)


@torch.no_grad()
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for ProjectedParakeet RMSNorm"
)
def test_eager_nemotron_audio_encoder_matches_native_forward():
    torch.manual_seed(0)

    sound_config = _small_sound_config()
    llm_hidden_size = 40
    device = torch.device("cuda")
    dtype = torch.float16

    native_encoder = (
        ProjectedParakeet(
            sound_config=sound_config,
            llm_hidden_size=llm_hidden_size,
            dtype=dtype,
        )
        .to(device)
        .eval()
    )
    ad_encoder = (
        NemotronAudioEncoder(
            sound_config=sound_config,
            llm_hidden_size=llm_hidden_size,
            dtype=dtype,
        )
        .to(device)
        .eval()
    )
    ad_encoder.load_state_dict(native_encoder.state_dict())

    input_audio_features = torch.randn(
        3,
        40,
        sound_config.num_mel_bins,
        device=device,
        dtype=dtype,
    )
    feature_attention_mask = torch.tensor(
        [
            [1] * 40,
            [1] * 40,
            [1] * 40,
        ],
        device=device,
        dtype=torch.int32,
    )

    ref_padded, ref_valid_lens, ref_flat = _native_encode_audio_data(
        native_encoder,
        input_audio_features,
        feature_attention_mask,
    )
    test_padded, test_valid_lens = ad_encoder(
        input_audio_features=input_audio_features,
        feature_attention_mask=feature_attention_mask,
    )

    torch.testing.assert_close(test_padded, ref_padded)
    torch.testing.assert_close(test_valid_lens, ref_valid_lens)
    torch.testing.assert_close(
        NemotronAudioEncoder.flatten_valid_outputs(test_padded, test_valid_lens),
        ref_flat,
    )


def test_flatten_valid_outputs_matches_native_truncation():
    sound_embeds = torch.tensor(
        [
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
            [[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]],
            [[12.0, 13.0], [14.0, 15.0], [16.0, 17.0]],
        ],
        dtype=torch.float32,
    )
    valid_output_lens = torch.tensor([3, 1, 2], dtype=torch.int32)

    expected = torch.tensor(
        [
            [0.0, 1.0],
            [2.0, 3.0],
            [4.0, 5.0],
            [6.0, 7.0],
            [12.0, 13.0],
            [14.0, 15.0],
        ],
        dtype=torch.float32,
    )

    actual = NemotronAudioEncoder.flatten_valid_outputs(
        sound_embeds,
        valid_output_lens,
    )

    torch.testing.assert_close(actual, expected)

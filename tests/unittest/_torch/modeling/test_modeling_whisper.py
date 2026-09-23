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

"""Unit tests for modeling_whisper.py.

CPU-only parity tests pinning ``WhisperLogMelFrontend`` (the engine-side GPU
log-mel front-end) to the HF reference (``_torch_extract_fbank_features``);
drift here silently corrupts transcripts.
"""

import numpy as np
import pytest
import torch
from transformers import WhisperConfig, WhisperFeatureExtractor

from tensorrt_llm._torch.models.modeling_whisper import WhisperInputProcessor, WhisperLogMelFrontend


def test_whisper_input_processor_requires_encoder_features():
    assert WhisperInputProcessor.requires_encoder_features


def _synthetic_waveform_batch(n_samples: int, seed: int = 1234) -> np.ndarray:
    """[3, n_samples] fp32 batch: full window, ~1/3 window, and near-silence,
    zero-padded to a common length (the request contract)."""
    rng = np.random.default_rng(seed)
    time = np.arange(n_samples, dtype=np.float32)
    batch = np.zeros((3, n_samples), dtype=np.float32)

    full = 0.4 * np.sin(2 * np.pi * 440.0 / 16000.0 * time)
    full += 0.2 * np.sin(2 * np.pi * 1333.0 / 16000.0 * time)
    full += 0.05 * rng.standard_normal(n_samples)
    batch[0] = full.astype(np.float32)

    short = n_samples // 3
    batch[1, :short] = (0.3 * rng.standard_normal(short)).astype(np.float32)

    batch[2, :1600] = 1e-4  # hard-zero tail exercises the log floor
    return batch


def _reference_log_mel(extractor: WhisperFeatureExtractor, batch: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(extractor._torch_extract_fbank_features(batch, device="cpu"))


@pytest.mark.parametrize("num_mel_bins", [80, 128])
def test_log_mel_frontend_matches_hf(num_mel_bins):
    """Default-parameter parity (no preprocessor config: whisper-tiny=80,
    large-v3=128 mel bins)."""
    config = WhisperConfig(num_mel_bins=num_mel_bins)
    frontend = WhisperLogMelFrontend(config)
    extractor = WhisperFeatureExtractor(feature_size=num_mel_bins)

    batch = _synthetic_waveform_batch(extractor.n_samples)
    ours = frontend(torch.from_numpy(batch))
    reference = _reference_log_mel(extractor, batch)

    assert ours.shape == (3, num_mel_bins, extractor.nb_max_frames)
    torch.testing.assert_close(ours, reference, atol=1e-4, rtol=1e-4)


def test_log_mel_frontend_reads_preprocessor_config(tmp_path):
    """Non-default STFT parameters (hop_length) and dither must be read from
    the checkpoint's preprocessor_config.json, not assumed."""
    extractor = WhisperFeatureExtractor(feature_size=80, hop_length=320, dither=0.02)
    extractor.save_pretrained(tmp_path)

    config = WhisperConfig(num_mel_bins=80)
    config._name_or_path = str(tmp_path)
    frontend = WhisperLogMelFrontend(config)

    assert frontend.hop_length == 320
    assert frontend.dither == pytest.approx(0.02)

    # Both implementations draw the dither noise from torch's default
    # generator with an identical shape, so seeding both sides identically
    # makes the (random) dither path exactly comparable.
    batch = _synthetic_waveform_batch(extractor.n_samples)
    torch.manual_seed(0)
    ours = frontend(torch.from_numpy(batch))
    torch.manual_seed(0)
    reference = _reference_log_mel(extractor, batch)

    torch.testing.assert_close(ours, reference, atol=1e-4, rtol=1e-4)


def test_log_mel_frontend_feature_size_mismatch_falls_back(tmp_path):
    """A preprocessor config contradicting config.num_mel_bins (broken
    checkpoint) must not win over the model config: the conv stem's input
    channel count comes from num_mel_bins."""
    WhisperFeatureExtractor(feature_size=80).save_pretrained(tmp_path)

    config = WhisperConfig(num_mel_bins=128)
    config._name_or_path = str(tmp_path)
    frontend = WhisperLogMelFrontend(config)

    assert frontend._mel_filters_np.shape[1] == 128
    batch = _synthetic_waveform_batch(WhisperFeatureExtractor().n_samples)
    assert frontend(torch.from_numpy(batch)).shape[1] == 128


def test_log_mel_frontend_does_not_mutate_input():
    """The waveform buffer belongs to the request; dither must not be added
    in place."""
    extractor = WhisperFeatureExtractor(feature_size=80)
    config = WhisperConfig(num_mel_bins=80)
    frontend = WhisperLogMelFrontend(config)
    frontend.dither = 0.02

    batch = torch.from_numpy(_synthetic_waveform_batch(extractor.n_samples))
    snapshot = batch.clone()
    frontend(batch)
    torch.testing.assert_close(batch, snapshot, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_whisper_for_conditional_generation_construction_and_forward():
    """Verify Whisper registration, construction, and forward parity with Hugging Face."""
    from transformers import WhisperForConditionalGeneration as HFWhisper

    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_utils import get_registered_model_class
    from tensorrt_llm._torch.models.modeling_whisper import WhisperForConditionalGeneration

    assert (
        get_registered_model_class("WhisperForConditionalGeneration")
        is WhisperForConditionalGeneration
    )

    config = WhisperConfig(
        vocab_size=32,
        d_model=64,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=1,
        decoder_attention_heads=1,
        encoder_ffn_dim=128,
        decoder_ffn_dim=128,
        max_source_positions=4,
        max_target_positions=8,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
        suppress_tokens=[],
        begin_suppress_tokens=[],
        torch_dtype=torch.float16,
    )
    config._attn_implementation = "eager"

    def make_metadata(encoder_length=None):
        """Build context attention metadata for decoder self-attention or cross-attention."""
        metadata = TrtllmAttentionMetadata(
            max_num_requests=1,
            max_num_tokens=3,
            kv_cache_manager=None,
            request_ids=[0],
        )
        metadata.seq_lens = torch.tensor([3], dtype=torch.int32)
        metadata.num_contexts = 1
        if encoder_length is not None:
            metadata.seq_lens_kv = torch.tensor([encoder_length], dtype=torch.int32)
        metadata.max_seq_len = encoder_length if encoder_length is not None else 3
        metadata.prepare()
        return metadata

    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(1234)
        reference = HFWhisper(config).half().cuda().eval()
        with torch.device("cuda"):
            model = WhisperForConditionalGeneration(
                ModelConfig(
                    pretrained_config=config,
                    attn_backend="TRTLLM",
                    is_encoder_decoder=True,
                )
            ).eval()
        model.load_weights(reference.state_dict())

        assert len(model.model.encoder.layers) == 1
        assert len(model.model.decoder.layers) == 1

        input_ids = torch.tensor([1, 5, 9], dtype=torch.long, device="cuda")
        position_ids = torch.arange(3, dtype=torch.long, device="cuda")
        encoder_states = torch.randn(4, 64, dtype=torch.float16, device="cuda")
        metadata = make_metadata()
        cross_metadata = make_metadata(encoder_length=4)

        with torch.inference_mode():
            logits = model(
                attn_metadata=metadata,
                input_ids=input_ids,
                position_ids=position_ids,
                encoder_hidden_states=encoder_states,
                cross_attn_metadata=cross_metadata,
                return_context_logits=True,
            )
            reference_output = reference(
                encoder_outputs=(encoder_states.unsqueeze(0),),
                decoder_input_ids=input_ids.unsqueeze(0),
                use_cache=False,
                return_dict=True,
            )
        expected_logits = reference_output.logits.squeeze(0).float()
        assert logits.shape == (3, config.vocab_size)
        assert logits.dtype == torch.float32
        assert torch.isfinite(logits).all()
        torch.testing.assert_close(logits, expected_logits, atol=1e-3, rtol=1e-3)

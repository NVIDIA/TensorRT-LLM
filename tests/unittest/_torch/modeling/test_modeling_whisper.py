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

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import WhisperConfig, WhisperFeatureExtractor

from tensorrt_llm._torch.models.modeling_whisper import WhisperInputProcessor, WhisperLogMelFrontend
from tensorrt_llm.sampling_params import SamplingParams


def test_whisper_input_processor_declares_input_capabilities():
    assert WhisperInputProcessor.requires_encoder_features
    assert WhisperInputProcessor.supports_token_ids_input is False


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


def _input_processor(num_mel_bins: int = 80):
    """A WhisperInputProcessor over a bare config, without a checkpoint on disk."""
    extractor = WhisperFeatureExtractor(feature_size=num_mel_bins)
    processor = object.__new__(WhisperInputProcessor)
    processor.config = WhisperConfig(num_mel_bins=num_mel_bins)
    processor.processor = SimpleNamespace(feature_extractor=extractor)
    processor._configure_audio_contract(extractor)
    processor._decoder_prompt = [50258, 50259, 50359, 50363]
    return processor, extractor


def _waveform_request(waveform, sampling_rate):
    return {"prompt": "", "multi_modal_data": {"audio": (waveform, sampling_rate)}}


def _mel_request(mel):
    return {"prompt": "", "multi_modal_data": {"audio": {"input_features": mel}}}


def _features(extra):
    return extra["multimodal_data"]["audio"]["encoder_input_features"]


def test_input_processor_forwards_a_precomputed_log_mel():
    """The extractor's output is accepted as is, with or without its batch axis,
    as numpy or torch, and reaches the encoder as fp32 [1, num_mel_bins, frames]."""
    processor, extractor = _input_processor()
    waveform = _synthetic_waveform_batch(extractor.n_samples)[0]
    mel = extractor(waveform, sampling_rate=extractor.sampling_rate, return_tensors="np")[
        "input_features"
    ]
    assert mel.shape == (1, 80, extractor.nb_max_frames)

    reversed_view = np.ascontiguousarray(mel[0][:, ::-1])[:, ::-1]  # negative strides
    with_grad = torch.from_numpy(mel[0]).clone().requires_grad_(True)
    for item in (
        mel,
        mel[0],
        torch.from_numpy(mel),
        mel[0].astype(np.float64),
        reversed_view,
        with_grad,
    ):
        prompt, extra = processor(_mel_request(item), SamplingParams())
        features = _features(extra)
        # A copy: the client may reuse its buffer for the next clip.
        source = (
            item.detach()
            if isinstance(item, torch.Tensor)
            else torch.from_numpy(np.ascontiguousarray(item))
        )
        assert features.data_ptr() != source.data_ptr()
        audio = extra["multimodal_data"]["audio"]
        assert prompt == processor._decoder_prompt
        assert audio["encoder_output_len"] == processor.config.max_source_positions
        assert features.shape == (1, 80, extractor.nb_max_frames)
        assert features.dtype == torch.float32 and features.is_contiguous()
        assert features.device.type == "cpu" and not features.requires_grad
        torch.testing.assert_close(features[0], torch.from_numpy(mel[0]).float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_a_cuda_log_mel_stays_on_its_device():
    """A spectrogram computed on the GPU ships as a device-local copy, not a CPU round trip."""
    processor, extractor = _input_processor()
    mel = torch.randn(1, 80, extractor.nb_max_frames, device="cuda", dtype=torch.float16)
    mel.requires_grad_(True)

    _, extra = processor(_mel_request(mel), SamplingParams())
    features = _features(extra)

    assert features.device == mel.device
    assert features.dtype == torch.float32 and features.is_contiguous()
    assert not features.requires_grad and features.data_ptr() != mel.data_ptr()
    torch.testing.assert_close(features, mel.detach().float())


def test_precomputed_log_mel_equals_the_engine_frontend():
    """Sending the spectrogram or the waveform gives the encoder the same input."""
    processor, extractor = _input_processor()
    waveform = _synthetic_waveform_batch(extractor.n_samples)[0][: extractor.n_samples // 2]
    mel = extractor(waveform, sampling_rate=extractor.sampling_rate, return_tensors="np")[
        "input_features"
    ]
    _, from_mel = processor(_mel_request(mel), SamplingParams())
    _, from_wave = processor(_waveform_request(waveform, extractor.sampling_rate), SamplingParams())
    frontend = WhisperLogMelFrontend(processor.config)

    torch.testing.assert_close(
        _features(from_mel), frontend(_features(from_wave)), atol=1e-4, rtol=1e-4
    )


@pytest.mark.parametrize(
    "shape",
    [(128, 3000), (80, 1500), (2, 80, 3000), (3000,)],
    ids=["wrong_bins", "wrong_frames", "batch_of_two", "one_dim"],
)
def test_input_processor_rejects_a_log_mel_of_the_wrong_shape(shape):
    processor, _ = _input_processor()
    with pytest.raises(ValueError, match="input_features"):
        processor(_mel_request(np.zeros(shape, dtype=np.float32)), SamplingParams())


@pytest.mark.parametrize(
    "bad",
    [np.zeros((80, 3000), dtype=np.int16), torch.zeros(80, 3000, dtype=torch.int32), [[object()]]],
)
def test_input_processor_rejects_a_non_float_log_mel(bad):
    processor, _ = _input_processor()
    with pytest.raises(ValueError, match="floating-point"):
        processor(_mel_request(bad), SamplingParams())


def test_requests_choose_their_audio_form_per_item():
    """One processor serves waveform items and spectrogram items alike, request by request."""
    processor, extractor = _input_processor()
    waveform = _synthetic_waveform_batch(extractor.n_samples)[0]
    rate = extractor.sampling_rate
    batch_feature = extractor(waveform, sampling_rate=rate, return_tensors="np")

    _, from_tuple = processor(_waveform_request(waveform, rate), SamplingParams())
    _, from_mapping = processor(
        {"prompt": "", "multi_modal_data": {"audio": {"array": waveform, "sampling_rate": rate}}},
        SamplingParams(),
    )
    _, from_mel = processor(_mel_request(batch_feature["input_features"]), SamplingParams())
    # The extractor's own BatchFeature (a UserDict, not a dict) is accepted as the item.
    _, from_batch_feature = processor(
        {"prompt": "", "multi_modal_data": {"audio": batch_feature}}, SamplingParams()
    )

    assert _features(from_tuple).shape == _features(from_mapping).shape == (1, extractor.n_samples)
    assert _features(from_mel).shape == _features(from_batch_feature).shape == (1, 80, 3000)
    with pytest.raises(TypeError, match="input_features"):
        processor({"prompt": "", "multi_modal_data": {"audio": object()}}, SamplingParams())


def test_encoder_graph_spec_is_the_spectrogram():
    from tensorrt_llm._torch.models.modeling_whisper import WhisperForConditionalGeneration

    model = object.__new__(WhisperForConditionalGeneration)
    model.model_config = SimpleNamespace(pretrained_config=WhisperConfig(num_mel_bins=80))
    model.model = SimpleNamespace(
        encoder=SimpleNamespace(log_mel=SimpleNamespace(n_samples=480000, nb_max_frames=3000))
    )

    assert model.encoder_graph_spec() == ((80, 3000), torch.float32, 1500)


def test_normalize_encoder_features_converts_only_the_waveform_rows():
    """A mixed batch reaches the stem as spectrogram rows: waveforms through the
    log-mel front-end in one upload, spectrograms untouched and in place."""
    from tensorrt_llm._torch.models.modeling_whisper import WhisperForConditionalGeneration

    config = WhisperConfig(num_mel_bins=80)
    frontend = WhisperLogMelFrontend(config)
    model = object.__new__(WhisperForConditionalGeneration)
    model.model = SimpleNamespace(encoder=SimpleNamespace(log_mel=frontend))
    waveforms = torch.from_numpy(_synthetic_waveform_batch(WhisperFeatureExtractor().n_samples)[:2])
    mel = torch.randn(1, 80, 3000)
    uploads = []

    def upload(rows):
        uploads.append([row.data_ptr() for row in rows])
        return torch.cat(rows, dim=0)

    normalized = model.normalize_encoder_features([waveforms[:1], mel, waveforms[1:2]], upload)

    assert [tuple(row.shape) for row in normalized] == [(1, 80, 3000)] * 3
    assert normalized[1] is mel
    expected = frontend(waveforms)
    torch.testing.assert_close(normalized[0], expected[:1])
    torch.testing.assert_close(normalized[2], expected[1:2])
    assert uploads == [[waveforms[:1].data_ptr(), waveforms[1:2].data_ptr()]]

    untouched = model.normalize_encoder_features([mel], upload)
    assert len(untouched) == 1 and untouched[0] is mel and len(uploads) == 1


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

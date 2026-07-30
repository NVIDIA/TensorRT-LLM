# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage-6 / Goal 6.1 focused tests for the Inkling dMel audio path.

All functional -- NO ASR-WER, no scored benchmark, no parity gate (human
feedback #22 TASK 2). Covers:
  * dMel preprocessing produces correctly-shaped, in-range, non-empty features
    with exactly one audio token per frame (and correct multi-clip concat);
  * :class:`InklingAudioModel` loads the real NVFP4-checkpoint audio weights
    (bf16 ``encoder`` [1280,6144] + ``final_norm`` [6144]) and forwards on
    CUDA to one ``decoder_dmodel`` row per frame with no NaN/Inf and no dtype
    drift / checkpoint upcast;
  * the tower forward equals the reference codebook-offset embedding-sum + final
    RMSNorm (port correctness -- catches a transpose / offset bug);
  * the input processor expands one ``<audio>`` placeholder into one token per
    dMel frame and FAILS LOUDLY on any placeholder/media/feature-row mismatch.

Run in the Slurm container (imports ``tensorrt_llm``); the CUDA forward uses one
GPU. See ``workspace/.../inkling_audio_tower.sbatch``.
"""
import json
import os

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.models.modeling_inkling_vision import (
    DEFAULT_AUDIO_TOKEN_ID,
    DEFAULT_IMAGE_TOKEN_ID,
    InklingAudioModel,
    InklingAudioPreprocessor,
    InklingImagePreprocessor,
    InklingInputProcessor,
    InklingVisionRMSNorm,
)

_DEFAULT_CKPT = (
    "/lustre/fsw/coreai_comparch_trtllm/kleinc/hf_data/hf_home/hub/"
    "models--thinkingmachines--Inkling-NVFP4/snapshots/"
    "95e51a54d9486020a80d49ae4f9103fb2b3f9686"
)
CKPT = os.environ.get("INKLING_CKPT", _DEFAULT_CKPT)
_HAVE_CKPT = os.path.isfile(os.path.join(CKPT, "config.json"))


def _synth_waveform(seconds: float = 0.6, sr: int = 16000) -> np.ndarray:
    """A short deterministic multi-tone clip (a valid 'short clip', no file I/O)."""
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    wav = 0.1 * np.sin(2 * np.pi * 440.0 * t) + 0.05 * np.sin(2 * np.pi * 1320.0 * t)
    return wav.astype(np.float32)


def _audio_config_from_ckpt():
    cfg = json.load(open(os.path.join(CKPT, "config.json")))["audio_config"]

    class _AC:
        pass

    ac = _AC()
    for k, v in cfg.items():
        setattr(ac, k, v)
    return ac


def _load_audio_weights():
    from safetensors import safe_open

    idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
    keys = ["model.audio.encoder.weight", "model.audio.final_norm.weight"]
    handles, out = {}, {}
    for k in keys:
        shard = idx[k]
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(CKPT, shard), framework="pt")
        out[k] = handles[shard].get_tensor(k)
    return out


# --------------------------------------------------------------------------- #
# dMel preprocessing (no checkpoint / GPU needed)
# --------------------------------------------------------------------------- #
def test_dmel_preprocess_shape_and_range():
    feat = InklingAudioPreprocessor().preprocess(_synth_waveform(0.6))
    db = feat["dmel_bins"]
    assert db.dtype == torch.int32
    assert db.ndim == 2 and db.shape[1] == 80, db.shape
    assert db.shape[0] >= 8, f"expected ~12 frames for 0.6s, got {db.shape[0]}"
    assert int(db.min()) >= 0 and int(db.max()) < 16, (int(db.min()), int(db.max()))
    assert feat["num_frames"] == [int(db.shape[0])]
    assert feat["num_tokens"] == feat["num_frames"]  # one audio token per frame


def test_dmel_multi_clip_concat():
    pre = InklingAudioPreprocessor()
    a, b = _synth_waveform(0.5), _synth_waveform(0.3)
    fa = pre.preprocess(a)["num_frames"][0]
    fb = pre.preprocess(b)["num_frames"][0]
    feat = pre.preprocess([a, b])
    assert feat["num_frames"] == [fa, fb]
    assert feat["dmel_bins"].shape[0] == fa + fb


def test_dmel_empty_clip():
    db = InklingAudioPreprocessor().encode_one(np.zeros(0, dtype=np.float32))
    assert db.shape == (0, 80) and db.dtype == torch.int32


# --------------------------------------------------------------------------- #
# Audio tower math (small deterministic module -- port correctness, no ckpt)
# --------------------------------------------------------------------------- #
def test_audio_tower_matches_reference_math():
    class _AC:
        decoder_dmodel = 32
        n_mel_bins = 4
        mel_vocab_size = 8
        use_audio_norm = True
        audio_mode = "dmel"

    torch.manual_seed(0)
    tower = InklingAudioModel(_AC())
    with torch.no_grad():
        tower.encoder.weight.normal_()
        tower.final_norm.weight.normal_()
    frames = torch.randint(0, 8, (5, 4), dtype=torch.int32)
    out = tower(frames)
    # Reference: bin m occupies codebook rows [m*V, (m+1)*V); sum over bins; norm.
    offs = torch.arange(4) * 8
    idx = offs.unsqueeze(0) + frames.long()
    ref = tower.final_norm(tower.encoder(idx.reshape(-1)).reshape(5, 4, 32).sum(dim=1))
    assert out.shape == (5, 32)
    assert torch.allclose(out, ref, atol=1e-5)


def test_audio_tower_no_norm_variant():
    class _AC:
        decoder_dmodel = 16
        n_mel_bins = 3
        mel_vocab_size = 4
        use_audio_norm = False
        audio_mode = "dmel"

    tower = InklingAudioModel(_AC())
    assert tower.final_norm is None
    out = tower(torch.zeros((2, 3), dtype=torch.int32))
    assert out.shape == (2, 16)


def test_audio_tower_rejects_wrong_bin_count():
    class _AC:
        decoder_dmodel = 8
        n_mel_bins = 4
        mel_vocab_size = 4
        use_audio_norm = True
        audio_mode = "dmel"

    tower = InklingAudioModel(_AC())
    with pytest.raises(ValueError):
        tower(torch.zeros((2, 5), dtype=torch.int32))  # 5 != n_mel_bins=4


# --------------------------------------------------------------------------- #
# Audio tower with REAL checkpoint weights, forwarded on CUDA
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAVE_CKPT, reason=f"checkpoint not found at {CKPT}")
def test_audio_tower_real_weights_cuda_forward():
    ac = _audio_config_from_ckpt()
    assert int(ac.n_mel_bins) == 80 and int(ac.mel_vocab_size) == 16
    assert int(ac.decoder_dmodel) == 6144 and bool(ac.use_audio_norm)

    tower = InklingAudioModel(ac).to(torch.bfloat16)
    tower.load_weights(_load_audio_weights())  # strict: exactly encoder + final_norm
    assert tuple(tower.encoder.weight.shape) == (1280, 6144)
    assert tower.encoder.weight.dtype == torch.bfloat16  # no upcast from NVFP4 base

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tower = tower.to(dev)
    db = InklingAudioPreprocessor().preprocess(_synth_waveform(0.8))["dmel_bins"].to(dev)
    out = tower(db)

    assert out.shape == (db.shape[0], 6144)
    assert out.dtype == torch.bfloat16  # no dtype drift
    assert torch.isfinite(out.float()).all()  # no NaN/Inf
    assert float(out.float().abs().sum()) > 0.0  # non-empty / non-degenerate
    print(
        f"AUDIO_TOWER_CUDA_OK dev={dev} n_frames={int(db.shape[0])} "
        f"out={tuple(out.shape)} dtype={out.dtype} finite=True"
    )


# --------------------------------------------------------------------------- #
# Input-processor audio placeholder expansion + fail-loud contract
# --------------------------------------------------------------------------- #
def _bare_processor():
    """An InklingInputProcessor with only the fields ``assemble`` needs, so the
    pure expansion/validation logic is testable without a tokenizer/model load."""
    ip = InklingInputProcessor.__new__(InklingInputProcessor)
    ip.image_token_id = DEFAULT_IMAGE_TOKEN_ID
    ip.audio_token_id = DEFAULT_AUDIO_TOKEN_ID
    ip._preprocessor = InklingImagePreprocessor()
    ip._audio_preprocessor = InklingAudioPreprocessor()
    return ip


def test_assemble_audio_expand():
    ip = _bare_processor()
    wav = _synth_waveform(0.6)
    n_frames = ip._audio_preprocessor.preprocess(wav)["num_frames"][0]
    ids = [10, 11, ip.audio_token_id, 12]
    out_ids, mm = ip.assemble(ids, image_data=None, audio_data=[wav])
    assert out_ids.count(ip.audio_token_id) == n_frames
    assert len(out_ids) == (len(ids) - 1) + n_frames
    assert "audio" in mm and "image" not in mm
    assert mm["audio"]["dmel_bins"].shape[0] == n_frames
    assert mm["audio"]["num_frames"] == [n_frames]
    assert mm["audio"]["offsets"] == [(2, 2 + n_frames - 1)]


def test_assemble_text_only_passthrough():
    ip = _bare_processor()
    assert ip.assemble([1, 2, 3]) == ([1, 2, 3], {})


def test_assemble_audio_failloud():
    ip = _bare_processor()
    wav = _synth_waveform(0.4)
    # placeholder present but no clip
    with pytest.raises(ValueError):
        ip.assemble([ip.audio_token_id], image_data=None, audio_data=None)
    # clip present but no placeholder
    with pytest.raises(ValueError):
        ip.assemble([1, 2, 3], image_data=None, audio_data=[wav])
    # two clips but one placeholder
    with pytest.raises(ValueError):
        ip.assemble([ip.audio_token_id], image_data=None, audio_data=[wav, wav])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Stage-transition failure recovery and reference preprocessing contracts."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.ltx2_5 import pipeline_ltx2_5 as impl
from tensorrt_llm._torch.visual_gen.models.ltx2_5.conditioning import video_state
from tensorrt_llm._torch.visual_gen.models.ltx2_5.lora import StageLoRAWeights, _StageWeight
from tensorrt_llm.visual_gen.params import MediaRef, VisualGenParams

pytestmark = pytest.mark.cpu_only


class _FailOnceTarget:
    def __init__(self, value: torch.Tensor) -> None:
        self.value = value
        self.fail = False

    def copy_(self, source: torch.Tensor) -> None:
        if self.fail:
            self.fail = False
            raise RuntimeError("injected copy failure")
        self.value.copy_(source)


def _stage_bank() -> StageLoRAWeights:
    bank = StageLoRAWeights.__new__(StageLoRAWeights)
    bank._weights = [
        _StageWeight(_FailOnceTarget(torch.zeros(2)), torch.zeros(2), torch.ones(2)),
        _StageWeight(_FailOnceTarget(torch.zeros(2)), torch.zeros(2), torch.ones(2)),
    ]
    bank._stage_one = False
    return bank


@pytest.mark.parametrize("stage_one", [True, False])
@pytest.mark.parametrize("retry_same_stage", [True, False])
def test_partial_switch_never_short_circuits_retry(stage_one, retry_same_stage) -> None:
    bank = _stage_bank()
    bank.select(not stage_one)
    bank._weights[1].target.fail = True
    with pytest.raises(RuntimeError, match="injected copy"):
        bank.select(stage_one)
    assert bank._stage_one is None
    desired = stage_one if retry_same_stage else not stage_one
    bank.select(desired)
    assert bank._stage_one is desired
    for entry in bank._weights:
        assert torch.equal(entry.target.value, entry.conditioned if desired else entry.base)


@pytest.mark.parametrize("failure", ["select", "denoise", "restore"])
def test_infer_restores_after_stage_one_failure(monkeypatch, failure) -> None:
    bank = _stage_bank()
    timer = SimpleNamespace(mark_pre_start=lambda: None, mark_denoise_start=lambda: None)
    monkeypatch.setattr(impl, "CudaPhaseTimer", lambda: timer)
    if failure == "select":
        bank._weights[1].target.fail = True

    def denoise(*args):
        assert bank._stage_one is True
        if failure == "restore":
            bank._weights[1].target.fail = True
        raise RuntimeError("injected denoise failure")

    pipeline = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        reference_downscale_factor=1,
        reference_temporal_scale_factor=1,
        _encode_prompt=lambda *a, **k: (None, None),
        _process_connectors=lambda *a: (None, None, None),
        _read_reference=lambda *a: torch.zeros(1, 128, 2, 1, 1, dtype=torch.bfloat16),
        transformer=SimpleNamespace(configure_audio_ulysses=lambda *a: None),
        _audio_state=lambda latent: video_state(latent.unsqueeze(-1), 24),
        _set_stage=bank.select,
        _denoise=denoise,
    )
    params = VisualGenParams(
        height=256,
        width=256,
        num_frames=9,
        frame_rate=24,
        seed=42,
        num_inference_steps=8,
        guidance_scale=1,
        video_reference=MediaRef(content=b"video", format="bytes"),
    )
    with pytest.raises(RuntimeError, match="injected") as error:
        impl.LTX25ICLoraPipeline.infer(pipeline, SimpleNamespace(params=params, prompt="test"))
    if failure == "restore":
        assert bank._stage_one is None
        assert "denoise failure" in str(error.value.__context__)
        bank.select(False)
    assert bank._stage_one is False
    for entry in bank._weights:
        assert torch.equal(entry.target.value, entry.base)
    # A failed request does not disable subsequent full stage transitions.
    bank.select(True)
    bank.select(False)
    assert all(torch.equal(e.target.value, e.base) for e in bank._weights)


def test_reference_uses_first_frames_and_preserves_normalization(monkeypatch) -> None:
    frames = np.stack([np.full((4, 6, 3), value, dtype=np.uint8) for value in (0, 64, 128, 255)])
    loader = Mock()
    loader.return_value.load_bytes.return_value.frames = frames
    monkeypatch.setattr(impl, "VideoMediaIO", loader)
    pipeline = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        reference_downscale_factor=1,
        reference_temporal_scale_factor=1,
        video_encoder=lambda video: video,
    )
    actual = impl.LTX25ICLoraPipeline._read_reference(pipeline, b"video", 2, 2, 3)
    loader.assert_called_once_with(num_frames=-1, fps=-1, format="np")
    assert actual.shape == (1, 3, 3, 2, 2)
    expected = (torch.tensor([0.0, 64.0, 128.0]) / 127.5 - 1).bfloat16()
    assert torch.equal(actual[0, 0, :, 0, 0], expected)
    with pytest.raises(ValueError, match="at least 5 frames"):
        impl.LTX25ICLoraPipeline._read_reference(pipeline, b"video", 2, 2, 5)

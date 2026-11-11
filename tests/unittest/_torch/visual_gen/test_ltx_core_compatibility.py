# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Boundary regressions for the November 4 LTX core migration."""

import pytest
import torch

pytestmark = pytest.mark.cpu_only


def test_connector_accepts_left_and_right_padding():
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.connector import Embeddings1DConnector

    torch.manual_seed(42)
    connector = Embeddings1DConnector(
        attention_head_dim=16, num_attention_heads=2, num_layers=1, num_learnable_registers=4
    )
    valid = torch.randn(2, 5, 32)
    padding = torch.zeros(2, 3, 32)
    left = torch.cat((padding, valid), dim=1)
    right = torch.cat((valid, padding), dim=1)
    left_mask = torch.zeros(2, 1, 1, 8)
    left_mask[..., :3] = -10000
    right_mask = torch.zeros_like(left_mask)
    right_mask[..., 5:] = -10000
    with torch.no_grad():
        actual_left, mask_left = connector(left, attention_mask=left_mask)
        actual_right, mask_right = connector(right, attention_mask=right_mask)
    torch.testing.assert_close(actual_left, actual_right, rtol=0, atol=0)
    assert torch.count_nonzero(mask_left) == torch.count_nonzero(mask_right) == 0


def test_split_rope_preserves_fused_kernel_layout_and_norm():
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.rope import (
        LTXRopeType,
        apply_rotary_emb,
        precompute_freqs_cis,
    )

    torch.manual_seed(42)
    positions = torch.rand(2, 3, 11, 2)
    freqs = precompute_freqs_cis(
        positions, 128, torch.float32, num_attention_heads=4, rope_type=LTXRopeType.SPLIT
    )
    for frequency in freqs:
        assert frequency.shape == (2, 11, 4, 32)
        torch.testing.assert_close(frequency[..., :16], frequency[..., 16:], rtol=0, atol=0)
    values = torch.randn(2, 11, 128)
    output = apply_rotary_emb(values, freqs, LTXRopeType.SPLIT)
    assert output.shape == values.shape
    torch.testing.assert_close(output.square().sum(-1), values.square().sum(-1))


def test_linear_factory_preserves_checkpoint_names():
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.adaln import AdaLayerNormSingle

    class RuntimeLinear(torch.nn.Linear):
        pass

    model = AdaLayerNormSingle(64, make_linear=RuntimeLinear)
    assert isinstance(model.linear, RuntimeLinear)
    assert isinstance(model.emb.timestep_embedder.linear_1, RuntimeLinear)
    assert isinstance(model.emb.timestep_embedder.linear_2, RuntimeLinear)
    assert set(model.state_dict()) == {
        "linear.weight",
        "linear.bias",
        "emb.timestep_embedder.linear_1.weight",
        "emb.timestep_embedder.linear_1.bias",
        "emb.timestep_embedder.linear_2.weight",
        "emb.timestep_embedder.linear_2.bias",
    }


@pytest.mark.parametrize("batch", [1, 2])
def test_audio_decoder_preserves_batch_axis(batch):
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.audio_vae.audio_vae import (
        decode_audio,
    )

    class Vocoder(torch.nn.Identity):
        output_sampling_rate = 24000

    samples = torch.randn(batch, 2, 32)
    actual = decode_audio(samples, torch.nn.Identity(), Vocoder())
    torch.testing.assert_close(actual, samples, rtol=0, atol=0)


def test_upsampler_casts_after_unnormalization():
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.upsampler import upsample_video

    class Statistics:
        def un_normalize(self, x):
            assert x.dtype == torch.float32
            return x + 0.1

        def normalize(self, x):
            return x - 0.1

    layer = torch.nn.Linear(4, 4).double()
    value = torch.randn(1, 4)
    expected = layer((value + 0.1).double()) - 0.1
    actual = upsample_video(value, Statistics(), layer)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

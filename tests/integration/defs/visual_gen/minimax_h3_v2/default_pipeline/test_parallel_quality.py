# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from defs.visual_gen import minimax_h3_quality_cases as parallel


@pytest.mark.parametrize("world_size", [2, 4])
def test_minimax_h3_text_encoder_tp_lpips(tmp_path, monkeypatch, world_size):
    parallel.check_text_encoder_tp_lpips(tmp_path, monkeypatch, world_size)


@pytest.mark.parametrize(
    "ulysses_size,vae_size,backend",
    [(2, 2, "VANILLA"), (4, 3, "VANILLA"), (2, 2, "FA4"), (2, 2, "TRTLLM"), (2, 2, "FLASHINFER")],
)
def test_minimax_h3_ulysses_parallel_vae_lpips(tmp_path, ulysses_size, vae_size, backend):
    parallel.check_ulysses_parallel_vae_lpips(tmp_path, ulysses_size, vae_size, backend)


def test_minimax_h3_tp2_example_lpips(tmp_path):
    parallel.check_tp2_example_lpips(tmp_path)

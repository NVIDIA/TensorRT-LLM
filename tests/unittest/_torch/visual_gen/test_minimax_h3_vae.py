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

"""Checkpoint-compatibility guard for the MiniMax-H3 autoencoders.

The autoencoders themselves come from Diffusers, so their numerics are covered
upstream.  What is *not* covered upstream is the pairing this pipeline depends
on: that the installed Diffusers' module definitions still produce exactly the
parameter set stored in the released MiniMax-H3 checkpoints.  A Diffusers
release that renames, adds, or drops a VAE submodule would deserialize into a
silently wrong model here, so the key sets are pinned.
"""

import hashlib

import torch
from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio


def _state_dict_key_digest(model: torch.nn.Module) -> tuple[int, str]:
    keys = sorted(model.state_dict())
    digest = hashlib.sha256("\n".join(keys).encode()).hexdigest()
    return len(keys), digest


def test_minimax_h3_vae_state_dict_keys_match_converted_checkpoint_manifests() -> None:
    # Expected counts and digests come from the sorted keys in the official converted
    # video/audio VAE safetensors headers. Meta construction keeps this test CPU-memory cheap.
    with torch.device("meta"):
        video_vae = AutoencoderKLMiniMaxH3()
        audio_vae = AutoencoderKLMiniMaxH3Audio()

    assert _state_dict_key_digest(video_vae) == (
        703,
        "ee4f6076997b8b88be09a7d90f20c2f35af4e0783119e8d1a4a6c4173956c5a4",
    )
    assert _state_dict_key_digest(audio_vae) == (
        1087,
        "88311a6ddd74e9b42c33f85a41635c8ef5d20ef029983de12b4016cf98493f1c",
    )

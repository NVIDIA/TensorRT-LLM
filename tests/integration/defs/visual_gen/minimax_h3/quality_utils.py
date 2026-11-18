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

"""Video quality metrics and shared parameters for MiniMax-H3 comparisons."""

import os
from pathlib import Path

import lpips as lpips_package
import torch
from test_common.llm_data import get_checkpoint

from defs.examples.visual_gen.visual_gen_test_utils import _cleanup_cuda

MINIMAX_H3_PROMPT = "\n".join(
    (
        " ".join(
            (
                "integrated_multimodal_description: [Shot 1] Cinematic, medium wide shot, pushing in",
                "slowly. In the cavernous, dimly lit bridge of a starship, sleek metallic consoles with",
                "glowing amber displays flank a massive, curved observation window. A female captain, in",
                "her late 40s with an athletic build and short silver-streaked black hair, stands in the",
                "center midground. She wears a structured, high-collared dark navy military tunic with",
                "silver chest insignias. Her back is to the camera, silhouetted against the cool, ambient",
                "starlight pouring through the thick glass. She stands perfectly still with her hands",
                "clasped tightly behind her back. Outside the window, a massive armada of jagged, dark",
                "grey dreadnoughts hovers in tight formation against a deep purple space nebula. The",
                "fleet's massive rear thrusters begin to glow with an intense, escalating bright blue",
                "light. [Shot 2] At 00:04.500, the camera cuts to a close-up of the captain's face and",
                "shakes strongly. The brilliant blue-white light from the fleet's gathering energy",
                "reflects vividly in her dark eyes. Suddenly, a blinding white flash floods through the",
                "window, completely washing out the background as the fleet jumps to hyperspace. The",
                "sheer spatial force violently jolts the bridge, causing the captain from Shot 1 to",
                "stagger slightly forward, her shoulders tensing as she visibly braces herself against",
                "the physical tremors. As the intense white light fades abruptly, leaving only the dim,",
                "empty expanse of the purple nebula reflected on her starkly lit skin, her jaw clenches,",
                "and she slowly closes her eyes in the newly emptied space.",
            )
        ),
        " ".join(
            (
                "overall_soundscape: A low, resonant hum of the ship's ambient life support systems",
                "serves as the baseline, soon drowned out by an audible, escalating, high-pitched",
                "electronic whine as the fleet outside charges its hyperdrives. A massive, deafening,",
                "bass-heavy boom and sharp crackle erupts during the blinding flash, accompanied by the",
                "loud metallic creaking, rattling, and deep thuds of the bridge's bulkheads vibrating",
                "under immense physical stress. The intense roaring impact then cuts abruptly back to a",
                "hollow, echoing room tone, leaving only the faint, steady hum of the isolated bridge.",
            )
        ),
        " ".join(
            (
                "non_diegetic_music: Cinematic space-opera orchestral score, slow tempo, featuring a",
                "solitary, mournful French horn melody over deep, sustained string dissonances that build",
                "rapidly in volume and intensity, swelling to a massive orchestral peak before snapping",
                "immediately into silence right after the jump.",
            )
        ),
    )
)
MINIMAX_H3_QUALITY_HEIGHT = 128
MINIMAX_H3_QUALITY_WIDTH = 128
MINIMAX_H3_SMOKE_NUM_FRAMES = 124
MINIMAX_H3_QUALITY_NUM_FRAMES = 124
MINIMAX_H3_NUM_INFERENCE_STEPS = 50
MINIMAX_H3_SEED = 0
# Keep explicit quality bounds; report measured distances for every run.
MINIMAX_H3_LPIPS_THRESHOLD = 0.15
# Provisional regression tolerance allowing quantization drift, not a calibrated
# perceptual acceptance score. Its scale is independent of LPIPS.
MINIMAX_H3_AUDIO_LOG_STFT_THRESHOLD = 0.10
MINIMAX_H3_LPIPS_BATCH_SIZE = 16


def _minimax_h3_checkpoint_path() -> str:
    override = os.environ.get("MINIMAX_H3_CHECKPOINT")
    if override and not Path(override).is_dir():
        raise FileNotFoundError(override)
    return override or get_checkpoint("MiniMax-H3")


def _mean_lpips_distance(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    if actual.dtype != torch.uint8:
        raise AssertionError(f"Generated video must be uint8, got {actual.dtype}.")
    actual = actual.permute(0, 1, 4, 2, 3).float().div(255.0)
    if actual.shape != expected.shape:
        raise AssertionError(
            f"Video shape mismatch: generated {tuple(actual.shape)}, "
            f"golden {tuple(expected.shape)}."
        )

    actual = actual.flatten(0, 1)
    expected = expected.flatten(0, 1)
    metric = lpips_package.LPIPS(net="alex", verbose=False).cuda().eval()
    scores: list[torch.Tensor] = []
    try:
        with torch.inference_mode():
            for start in range(0, actual.shape[0], MINIMAX_H3_LPIPS_BATCH_SIZE):
                end = start + MINIMAX_H3_LPIPS_BATCH_SIZE
                scores.append(
                    metric(
                        expected[start:end].cuda().mul(2).sub(1),
                        actual[start:end].cuda().mul(2).sub(1),
                    ).cpu()
                )
        return float(torch.cat(scores).mean())
    finally:
        del metric
        _cleanup_cuda()

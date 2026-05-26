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

from __future__ import annotations

from typing import Any

import torch

from tensorrt_llm.logger import logger

GUARDRAIL_HF_REPO = "nvidia/Cosmos-1.0-Guardrail"
GUARDRAIL_REVISION = "cf03c0395fac8c4de386c0bdab12cc4fc8d66362"


def download_guardrail_checkpoint() -> str:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    try:
        return snapshot_download(
            GUARDRAIL_HF_REPO,
            revision=GUARDRAIL_REVISION,
            local_files_only=True,
        )
    except FileNotFoundError:
        logger.warning(f"Guardrail checkpoint not found, downloading from {GUARDRAIL_HF_REPO}")
        try:
            return snapshot_download(
                GUARDRAIL_HF_REPO,
                revision=GUARDRAIL_REVISION,
            )
        except GatedRepoError:
            raise ValueError(
                "Cosmos Guardrail checkpoint not found. "
                "Please ensure "
                "a) you have accepted the terms of use (https://huggingface.co/nvidia/Cosmos-1.0-Guardrail) "
                "b) you have set a valid HF_TOKEN environment variable"
            )


def check_video_safety(video_tensor: torch.Tensor, safety_checker: Any) -> torch.Tensor | None:
    v = video_tensor.detach().cpu()
    was_batched = v.dim() == 5
    if was_batched:
        v = v[0]
    frames_np = v.numpy()
    frames_np = safety_checker.check_video_safety(frames_np)
    if frames_np is None:
        return None

    result = torch.from_numpy(frames_np)
    if was_batched:
        result = result.unsqueeze(0)
    return result.to(video_tensor.device)

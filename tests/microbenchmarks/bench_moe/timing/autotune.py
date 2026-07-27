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

"""Untimed autotune pre-pass."""

from __future__ import annotations

import os
import tempfile
from typing import List

import torch

from tensorrt_llm._torch.autotuner import AutoTuner, autotune


def _run_autotune(
    moe,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    all_rank_num_tokens: List[int],
    fast_autotune: bool,
) -> str:
    """One untimed forward pass under ``autotune(...)`` to populate kernel caches.

    Returns an autotune status string, one of:
      - ``"success"``       : ran with the project default tuner settings
      - ``"success:fast"``  : ran with ``--fast_autotune`` overrides (lower quality)
      - ``"failed:<reason>"``: the autotune pass raised; caller decides whether to
                              trust the subsequent timings

    The function always restores ``AutoTuner`` singleton state on exit so that
    ``--fast_autotune`` set for one case does not leak into the next.
    """
    tuner = AutoTuner.get()
    saved_warmup = tuner.warmup
    saved_repeat = tuner.repeat
    saved_stream_delay = tuner.stream_delay_micro_secs
    if fast_autotune:
        tuner.warmup = 0
        tuner.repeat = 1
        tuner.stream_delay_micro_secs = 10

    cache_path = os.path.join(tempfile.gettempdir(), "bench_moe_autotuner_cache.json")
    try:
        with torch.inference_mode(), autotune(cache_path=cache_path):
            moe.forward(x, router_logits, all_rank_num_tokens=all_rank_num_tokens)
        torch.cuda.synchronize()
        return "success:fast" if fast_autotune else "success"
    finally:
        tuner.warmup = saved_warmup
        tuner.repeat = saved_repeat
        tuner.stream_delay_micro_secs = saved_stream_delay

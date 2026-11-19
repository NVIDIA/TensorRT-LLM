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

"""The DFlash drafter's FP8 E4M3 context latents, which the engine's ``speculative.dflash_attention`` does not
carry: their per-tensor scale and the saturating cast the sm_107 targets store them with."""

import torch

# FP8 E4M3 storage of an MLA drafter's context latent: one per-tensor scale of
# 1.0 for the latent and for the query that reads it -- the scheme of the
# target MLA's FP8 KV cache without calibrated kv scales (quant_scale_kv and
# quant_scale_q default to 1, bmm1_scale = softmax_scale * dq_q * dq_kv,
# bmm2_scale = dq_kv). Values saturate at the format's largest finite value, as
# the target's FP8 cache writes do, instead of overflowing to NaN.
DRAFT_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
DRAFT_FP8_KV_SCALE = 1.0


def to_draft_fp8(x: torch.Tensor) -> torch.Tensor:
    """``x`` as saturated FP8 E4M3 at the drafter's unit per-tensor scale."""
    return x.clamp(-DRAFT_FP8_MAX, DRAFT_FP8_MAX).to(torch.float8_e4m3fn)

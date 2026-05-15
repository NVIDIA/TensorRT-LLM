# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/delta_rule/chunk.py

from typing import Optional

import torch

from tensorrt_llm._torch.modules.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from tensorrt_llm._torch.modules.fla.chunk_o import chunk_fwd_o

from .wy_fast import prepare_wy_repr_fwd


def chunk_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    # obtain WY representation. u is actually the new v.
    w, u, A = prepare_wy_repr_fwd(
        k=k,
        v=v,
        beta=beta,
        cu_seqlens=cu_seqlens,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=None,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )

    o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=None, scale=scale, cu_seqlens=cu_seqlens)

    return o, A, final_state

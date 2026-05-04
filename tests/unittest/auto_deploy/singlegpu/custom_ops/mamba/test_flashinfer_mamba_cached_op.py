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
import pytest
import torch
from test_triton_mamba_cached_op import _random_params

import tensorrt_llm._torch.auto_deploy  # noqa: F401
from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import BatchInfo


@pytest.fixture
def mamba_env():
    device = "cuda"
    dtype = torch.bfloat16
    atol = 1e-3
    rtol = 1e-3
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    return {"device": device, "dtype": dtype, "atol": atol, "rtol": rtol}


def test_flashinfer_decode_matches_triton(mamba_env):
    device = mamba_env["device"]
    dtype = mamba_env["dtype"]
    atol = mamba_env["atol"]
    rtol = mamba_env["rtol"]

    batch, seq = 2, 1
    num_heads, head_dim = 2, 64
    n_groups, ssm_state_size = 2, 64
    (hidden_states, A, B, C, D, dt, dt_bias, time_step_limit, chunk_size) = _random_params(
        device, dtype, batch, seq, num_heads, head_dim, n_groups, ssm_state_size
    )

    max_batch_size = 4
    slot_idx = torch.tensor([0, 2], device=device, dtype=torch.int32)
    ssm_state_cache_triton = torch.randn(
        max_batch_size, num_heads, head_dim, ssm_state_size, device=device, dtype=dtype
    )
    ssm_state_cache_flashinfer = ssm_state_cache_triton.clone()

    _bi = BatchInfo()
    _bi.update([0, 0, 0, 0, batch, batch])
    batch_info_host = _bi.serialize()
    cu_seqlen = torch.zeros(batch + 1, device=device, dtype=torch.int32)
    use_initial_states = torch.zeros(batch, device=device, dtype=torch.bool)

    any_prefill_use_initial_states_host = torch.tensor([False], device=device, dtype=torch.bool)
    y_triton = torch.ops.auto_deploy.triton_cached_ssm(
        hidden_states,
        A,
        B,
        C,
        D,
        dt,
        dt_bias,
        # STANDARD METADATA
        batch_info_host,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        any_prefill_use_initial_states_host,
        # EXTRA METADATA
        None,  # chunk indices
        None,  # chunk offsets
        None,  # seq_idx_prefill
        # CACHES
        ssm_state_cache_triton,
        None,
        # CONSTANTS
        time_step_limit,
        chunk_size,
    )

    y_flashinfer = torch.ops.auto_deploy.flashinfer_cached_ssm(
        hidden_states,
        A,
        B,
        C,
        D,
        dt,
        dt_bias,
        # STANDARD METADATA
        batch_info_host,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        any_prefill_use_initial_states_host,
        # EXTRA METADATA
        None,  # chunk indices
        None,  # chunk offsets
        None,  # seq_idx_prefill
        # CACHES
        ssm_state_cache_flashinfer,
        # CONSTANTS
        time_step_limit,
        chunk_size,
    )

    assert y_triton.shape == hidden_states.shape
    assert y_flashinfer.shape == hidden_states.shape
    assert torch.isfinite(y_flashinfer).all()
    assert torch.allclose(y_flashinfer, y_triton.to(y_flashinfer.dtype), atol=atol, rtol=rtol)

    after_triton = ssm_state_cache_triton.index_select(0, slot_idx)
    after_flashinfer = ssm_state_cache_flashinfer.index_select(0, slot_idx)
    assert torch.allclose(
        after_flashinfer.to(after_triton.dtype), after_triton, atol=atol, rtol=rtol
    )

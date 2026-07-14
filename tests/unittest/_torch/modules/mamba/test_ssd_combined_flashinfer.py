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
"""FlashInfer SSD equivalence tests for varlen and non-varlen modes."""

import pytest
import torch
from utils.torch_ref import ssd_chunk_scan_combined_ref

from tensorrt_llm._torch.modules.mamba.mamba2_metadata import (
    cu_seqlens_to_chunk_indices_offsets_triton,
)
from tensorrt_llm._torch.modules.mamba.ssd_combined import (
    _flashinfer_ssd_supported,
    _get_flashinfer_ssd_kernel,
    mamba_chunk_scan_combined,
)
from tensorrt_llm._utils import is_sm_100f


def _flashinfer_available():
    try:
        import flashinfer  # noqa: F401

        return True
    except ImportError:
        return False


skip_no_flashinfer = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_sm_100f() and _flashinfer_available()),
    reason="FlashInfer SSD requires SM100+ with flashinfer installed",
)

# Configurations that flashinfer SSD currently lowers cleanly.
_VALID_SHAPES = [
    # (chunk_size, nheads, headdim, ngroups, dstate)
    (128, 4, 64, 1, 128),
    (128, 8, 64, 1, 128),
    (128, 8, 64, 2, 128),
]

_BF16_RTOL = 1e-2
_BF16_ATOL = 2e-1


def _make_inputs(batch, seqlen, nheads, headdim, ngroups, dstate, device):
    torch.manual_seed(0)
    x = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.bfloat16, device=device)
    dt = torch.randn(batch, seqlen, nheads, dtype=torch.bfloat16, device=device)
    A = -torch.rand(nheads, dtype=torch.float32, device=device) - 1.0
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=torch.bfloat16, device=device)
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=torch.bfloat16, device=device)
    D = torch.randn(nheads, dtype=torch.float32, device=device)
    dt_bias = torch.rand(nheads, dtype=torch.float32, device=device) - 4.0
    return x, dt, A, B, C, D, dt_bias


@skip_no_flashinfer
@pytest.mark.parametrize("chunk_size,nheads,headdim,ngroups,dstate", _VALID_SHAPES)
@pytest.mark.parametrize("seqlens", [[256, 192], [512], [128, 64, 192]])
def test_flashinfer_varlen(chunk_size, nheads, headdim, ngroups, dstate, seqlens):
    assert _flashinfer_ssd_supported(chunk_size, dstate, headdim)
    device = torch.device("cuda")
    total = sum(seqlens)
    num_seqs = len(seqlens)

    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seqlens), 0)),
        dtype=torch.int32,
        device=device,
    )
    seq_idx = torch.cat(
        [torch.full((sl,), i, dtype=torch.int32, device=device) for i, sl in enumerate(seqlens)]
    ).unsqueeze(0)
    chunk_indices, chunk_offsets = cu_seqlens_to_chunk_indices_offsets_triton(
        cu_seqlens, chunk_size, total_seqlens=total
    )

    x, dt, A, B, C, D, dt_bias = _make_inputs(1, total, nheads, headdim, ngroups, dstate, device)

    out = torch.empty_like(x)
    varlen_states = mamba_chunk_scan_combined(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=D,
        z=None,
        dt_bias=dt_bias,
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        cu_seqlens=cu_seqlens,
        dt_softplus=True,
        out=out,
        return_varlen_states=True,
        return_final_states=False,
    )

    # Reference: process each sequence independently with the PyTorch ref.
    out_ref = torch.empty_like(out)
    state_ref = torch.empty(num_seqs, nheads, dstate, headdim, dtype=torch.bfloat16, device=device)
    for i in range(num_seqs):
        s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        part_out, part_state = ssd_chunk_scan_combined_ref(
            x[:, s:e],
            dt[:, s:e],
            A,
            B[:, s:e],
            C[:, s:e],
            chunk_size,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
        )
        out_ref[0, s:e] = part_out.squeeze(0)
        state_ref[i] = part_state.squeeze(0)

    # ssd_chunk_scan_combined_ref returns final_states as (..., dstate, headdim);
    # flashinfer returns (..., headdim, dstate).
    torch.testing.assert_close(out, out_ref, rtol=_BF16_RTOL, atol=_BF16_ATOL)
    torch.testing.assert_close(
        varlen_states.to(torch.bfloat16),
        state_ref.permute(0, 1, 3, 2).contiguous(),
        rtol=_BF16_RTOL,
        atol=_BF16_ATOL,
    )


@skip_no_flashinfer
@pytest.mark.parametrize("chunk_size,nheads,headdim,ngroups,dstate", _VALID_SHAPES)
@pytest.mark.parametrize("batch,seqlen", [(1, 512), (2, 256), (3, 384)])
def test_flashinfer_non_varlen(chunk_size, nheads, headdim, ngroups, dstate, batch, seqlen):
    assert _flashinfer_ssd_supported(chunk_size, dstate, headdim)
    device = torch.device("cuda")

    x, dt, A, B, C, D, dt_bias = _make_inputs(
        batch, seqlen, nheads, headdim, ngroups, dstate, device
    )

    out = torch.empty_like(x)
    final_states = mamba_chunk_scan_combined(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=D,
        z=None,
        dt_bias=dt_bias,
        seq_idx=None,
        cu_seqlens=None,
        dt_softplus=True,
        out=out,
        return_varlen_states=False,
        return_final_states=True,
    )

    out_ref, state_ref = ssd_chunk_scan_combined_ref(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
    )

    torch.testing.assert_close(out, out_ref, rtol=_BF16_RTOL, atol=_BF16_ATOL)
    torch.testing.assert_close(
        final_states.to(torch.bfloat16),
        state_ref.permute(0, 1, 3, 2).contiguous(),
        rtol=_BF16_RTOL,
        atol=_BF16_ATOL,
    )


@skip_no_flashinfer
def test_flashinfer_kernel_cache_distinguishes_modes():
    # Sharing a cache entry across modes would silently use the wrong kernel.
    chunk_size, nheads, headdim, ngroups, dstate = 128, 4, 64, 1, 128
    k_v = _get_flashinfer_ssd_kernel(chunk_size, nheads, headdim, dstate, ngroups, True)
    k_n = _get_flashinfer_ssd_kernel(chunk_size, nheads, headdim, dstate, ngroups, False)
    assert k_v is not k_n
    assert k_v is _get_flashinfer_ssd_kernel(chunk_size, nheads, headdim, dstate, ngroups, True)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gate of the SM107 in-place output of the MiniMax-M3 NVFP4 sparse prefill (msa_prefill)."""

import types

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha import msa_prefill


def _sparse(takes_out: bool):
    if takes_out:

        def fn(q, *, schedule=None, out=None):
            return out

    else:

        def fn(q, *, schedule=None):
            return q

    return types.SimpleNamespace(sparse_atten_nvfp4_kv_func=fn)


@pytest.fixture(autouse=True)
def _fresh_signature_cache():
    msa_prefill._nvfp4_sparse_takes_out.__dict__.pop("_cached", None)
    yield
    msa_prefill._nvfp4_sparse_takes_out.__dict__.pop("_cached", None)


@pytest.mark.parametrize(
    "sm_version,enabled,takes_out,dtype,contiguous,expected",
    [
        (107, True, True, torch.bfloat16, True, True),
        (103, True, True, torch.bfloat16, True, False),
        (107, False, True, torch.bfloat16, True, False),
        (107, True, False, torch.bfloat16, True, False),
        (107, True, True, torch.float16, True, False),
        (107, True, True, torch.bfloat16, False, False),
    ],
)
def test_out_in_place_gate(
    monkeypatch, sm_version, enabled, takes_out, dtype, contiguous, expected
):
    monkeypatch.setattr(msa_prefill, "get_sm_version", lambda: sm_version)
    monkeypatch.setattr(msa_prefill, "_SM107_NVFP4_SPARSE_OUT_IN_PLACE", enabled)
    out = torch.zeros(8, 4, 16, dtype=dtype)
    if not contiguous:
        out = torch.zeros(4, 8, 16, dtype=dtype).transpose(0, 1)
    kwargs = msa_prefill._nvfp4_sparse_out_kwargs(_sparse(takes_out), out)
    assert ("out" in kwargs) is expected
    if expected:
        assert kwargs["out"] is out

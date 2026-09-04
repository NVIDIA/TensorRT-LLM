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
"""Tests for the packed GDN causal-conv1d-update fast path.

The fast path is opt-in (``TRTLLM_GDN_CONV_PACKED=1``) and only engages on the exact
contract it was written for. These tests pin both halves of that: the guard
predicate must reject everything outside the contract, and inside the contract the
fast path must agree with the shipped Triton kernel on all three results (``out``
plus the two in-place mutated state tensors).
"""

import pytest
import torch

from tensorrt_llm._torch.modules.mamba import causal_conv1d_triton as ccu

# Production GDN decode shape for Qwen3.6-35B-A3B at TP2.
DIM = 4096
SEQLEN = 3
STATE_LEN = 3
WIDTH = 4

skip_unsupported = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA",
)


def _make(batch, device="cuda", dim=DIM, seqlen=SEQLEN, state_len=STATE_LEN, width=WIDTH):
    """Build inputs with production's exact layouts.

    ``x`` is a transposed view of a ``(batch, seqlen, dim)`` GEMM output, so the
    feature axis is unit-stride; ``conv_state_indices`` is a scattered permutation
    of pool slots, never identity.
    """
    g = torch.Generator(device=device).manual_seed(1234 + batch)
    num_cache_lines = batch + 1
    x = (
        torch.randn(batch, seqlen, dim, generator=g, device=device, dtype=torch.float32)
        .to(torch.bfloat16)
        .transpose(1, 2)
    )
    conv_state = torch.randn(
        num_cache_lines, dim, state_len, generator=g, device=device, dtype=torch.float32
    ).to(torch.bfloat16)
    weight = (torch.randn(dim, width, generator=g, device=device, dtype=torch.float32) * 0.5).to(
        torch.bfloat16
    )
    bias = (torch.randn(dim, generator=g, device=device, dtype=torch.float32) * 0.1).to(
        torch.bfloat16
    )
    idx = torch.randperm(num_cache_lines, generator=g, device=device)[:batch]
    return dict(
        x=x,
        conv_state=conv_state,
        weight=weight,
        bias=bias,
        conv_state_indices=idx.to(torch.int32),
        intermediate_conv_window=torch.zeros(
            batch, seqlen, dim, state_len, device=device, dtype=torch.bfloat16
        ),
        intermediate_state_indices=torch.arange(batch, device=device, dtype=torch.int32),
    )


def _run(t, enabled):
    t = {k: v.clone() for k, v in t.items()}
    prev = ccu._PACKED_CONV_ENABLED
    ccu._PACKED_CONV_ENABLED = enabled
    try:
        out = ccu.causal_conv1d_update(
            t["x"],
            t["conv_state"],
            t["weight"],
            t["bias"],
            "silu",
            conv_state_indices=t["conv_state_indices"],
            intermediate_conv_window=t["intermediate_conv_window"],
            intermediate_state_indices=t["intermediate_state_indices"],
        )
    finally:
        ccu._PACKED_CONV_ENABLED = prev
    return out, t["conv_state"], t["intermediate_conv_window"]


@skip_unsupported
class TestGdnConvKfFastPath:
    """The fast path must be indistinguishable from the shipped kernel."""

    @pytest.mark.parametrize("batch", [1, 2, 8, 64, 128, 129, 255, 256, 511, 512])
    def test_matches_shipped_kernel(self, batch):
        t = _make(batch)
        ref_out, ref_state, ref_inter = _run(t, enabled=False)
        packed_out, packed_state, packed_inter = _run(t, enabled=True)

        # The two state tensors are pure data movement -- they must be bit-exact.
        torch.testing.assert_close(packed_state, ref_state, rtol=0, atol=0)
        torch.testing.assert_close(packed_inter, ref_inter, rtol=0, atol=0)
        # `out` differs only by bf16 rounding order: the shipped kernel rounds each
        # tap product to bf16, the fast path accumulates in fp32.
        torch.testing.assert_close(packed_out.float(), ref_out.float(), rtol=2e-2, atol=2e-2)

    def test_out_layout_is_preserved(self):
        """Callers view the result as (batch*seqlen, dim); the strides must allow it."""
        t = _make(8)
        out, _, _ = _run(t, enabled=True)
        assert out.shape == (8, DIM, SEQLEN)
        assert out.stride() == (SEQLEN * DIM, 1, DIM)
        # The free transpose every caller performs must not copy.
        assert out.transpose(1, 2).is_contiguous()


@skip_unsupported
class TestGdnConvKfGuard:
    """Anything outside the supported contract must fall through to the shipped kernel."""

    def test_accepts_production_contract(self):
        t = _make(512)
        assert ccu._packed_conv_applicable(
            t["x"],
            t["conv_state"],
            t["weight"],
            t["bias"],
            "silu",
            512,
            DIM,
            SEQLEN,
            STATE_LEN,
            WIDTH,
            None,
            t["conv_state_indices"],
            None,
            t["intermediate_conv_window"],
            t["intermediate_state_indices"],
            None,
        )

    @pytest.mark.parametrize(
        "override",
        [
            pytest.param(dict(dim=2048), id="wrong_dim"),
            pytest.param(dict(seqlen=1), id="wrong_seqlen"),
            pytest.param(dict(width=3, state_len=2), id="wrong_width"),
        ],
    )
    def test_rejects_off_contract_shapes(self, override):
        kw = dict(dim=DIM, seqlen=SEQLEN, state_len=STATE_LEN, width=WIDTH)
        kw.update(override)
        t = _make(8, **kw)
        assert not ccu._packed_conv_applicable(
            t["x"],
            t["conv_state"],
            t["weight"],
            t["bias"],
            "silu",
            8,
            kw["dim"],
            kw["seqlen"],
            kw["state_len"],
            kw["width"],
            None,
            t["conv_state_indices"],
            None,
            t["intermediate_conv_window"],
            t["intermediate_state_indices"],
            None,
        )

    def test_rejects_unsupported_features(self):
        """Circular buffer, spec-decode offset, and eagle-tree are all out of contract."""
        t = _make(8)
        base = (
            t["x"],
            t["conv_state"],
            t["weight"],
            t["bias"],
            "silu",
            8,
            DIM,
            SEQLEN,
            STATE_LEN,
            WIDTH,
        )
        tail = (t["intermediate_conv_window"], t["intermediate_state_indices"])
        sentinel = torch.zeros(8, device="cuda", dtype=torch.int32)
        # cache_seqlens (circular buffer)
        assert not ccu._packed_conv_applicable(
            *base, sentinel, t["conv_state_indices"], None, *tail, None
        )
        # num_accepted_tokens (IS_SPEC_DECODING state-roll offset)
        assert not ccu._packed_conv_applicable(
            *base, None, t["conv_state_indices"], sentinel, *tail, None
        )
        # retrieve_next_token (eagle tree)
        assert not ccu._packed_conv_applicable(
            *base, None, t["conv_state_indices"], None, *tail, sentinel
        )
        # no intermediate-window capture
        assert not ccu._packed_conv_applicable(
            *base, None, t["conv_state_indices"], None, None, t["intermediate_state_indices"], None
        )
        # no continuous batching
        assert not ccu._packed_conv_applicable(*base, None, None, None, *tail, None)

    def test_rejects_non_silu(self):
        t = _make(8)
        assert not ccu._packed_conv_applicable(
            t["x"],
            t["conv_state"],
            t["weight"],
            t["bias"],
            None,
            8,
            DIM,
            SEQLEN,
            STATE_LEN,
            WIDTH,
            None,
            t["conv_state_indices"],
            None,
            t["intermediate_conv_window"],
            t["intermediate_state_indices"],
            None,
        )

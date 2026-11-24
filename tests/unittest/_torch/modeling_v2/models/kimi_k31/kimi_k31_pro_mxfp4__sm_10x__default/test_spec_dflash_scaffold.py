# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 DFlash hidden-state capture taps.

CPU-only (no checkpoint): checks that K31TextModel and K31DecoderLayer hand the drafter the pre-norm attn_res
mixture, including the no-successor tail.
"""

from types import MethodType, SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm.rms_norm import (
    K31RMSNorm,
)

# A norm module stand-in for the identity: the code under test reads its weight and epsilon and the tests replace
# the RMSNorm op with the identity.
_IDENTITY_NORM = SimpleNamespace(weight=None, variance_epsilon=0.0)


def test_capture_taps_the_next_layers_aggregated_stream(monkeypatch):
    """The tap reads layer j+1's mixture, not layer j's prefix sum.

    A drafter distilled on the mixture accepts fewer drafts on the prefix sum, and the
    mistake degrades silently: capturing the prefix sum instead is
    still a valid tensor of the right shape, so nothing raises and only
    acceptance moves.

    Layers here are stubs that reproduce the one contract the real layer has
    with this loop -- compute the attention-side mixture, and when handed a
    ``capture`` tuple, record it under the layer id in that tuple. Giving
    each stub its own proj/norm weights is what makes the assertions able to
    tell "layer j+1's mixture" from "layer j's", and the model's output-side
    weights from either. Also covers the no-successor tail branch, which uses
    output_attn_res_proj/norm and no layer can produce.
    """
    pytest.importorskip("fla")
    from types import SimpleNamespace

    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default import (
        modeling as mkl,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default.modeling import (  # noqa: E501
        K31TextModel,
        _apply_attn_res,
    )

    # The model applies its norms through the catalog RMSNorm op with the module's weight and epsilon.
    monkeypatch.setattr(mkl, "flashinfer_rmsnorm", lambda h, weight, eps: h)

    torch.manual_seed(0)
    num_layers, num_tokens, hidden = 4, 3, 8

    def _weights(scale):
        proj = torch.nn.Linear(hidden, 1, bias=False, dtype=torch.float32)
        norm = K31RMSNorm(hidden, eps=1e-6, dtype=torch.float32)
        with torch.no_grad():
            proj.weight.copy_(torch.randn(1, hidden) * scale)
            norm.weight.copy_(torch.randn(hidden) * scale + 1.0)
        return proj, norm

    layer_w = [_weights(0.5 + i) for i in range(num_layers)]
    out_proj, out_norm = _weights(9.0)

    seen = {}

    class _StubLayer:
        def __init__(self, idx):
            self.layer_idx = idx
            self.proj, self.norm = layer_w[idx]

        def __call__(
            self,
            hidden_states,
            block_residual,
            num_snapshots,
            attn_metadata,
            capture,
        ):
            # What the real layer computes on the way into its attention, and
            # hands to the tap: the aggregated stream the previous layer's
            # consumer sees.
            mixture = (
                _apply_attn_res(hidden_states, block_residual[:num_snapshots], self.proj, self.norm)
                if num_snapshots > 0
                else hidden_states
            )
            if capture is not None:
                spec_md, layer_id = capture
                spec_md.maybe_capture_hidden_states(layer_id, mixture, None)
            block_residual[num_snapshots] = hidden_states * (self.layer_idx + 1)
            # A prefix sum that is never equal to the mixture above.
            return hidden_states + (self.layer_idx + 1), num_snapshots + 1

    layers = [_StubLayer(i) for i in range(num_layers)]
    last_idx = num_layers - 1
    spec_md = SimpleNamespace(
        _capture_layer_set=frozenset({0, last_idx}),
        maybe_capture_hidden_states=lambda lid, h, r: seen.__setitem__(lid, h.clone()),
    )
    embeds = torch.randn(num_tokens, hidden, dtype=torch.float32)
    fake = SimpleNamespace(
        embed_tokens=lambda ids: embeds,
        layers=layers,
        norm=_IDENTITY_NORM,
        output_attn_res_proj=out_proj,
        output_attn_res_norm=out_norm,
        num_attn_res_snapshots=num_layers,
    )
    fake._forward_layers = MethodType(K31TextModel._forward_layers, fake)

    K31TextModel.forward(
        fake,
        attn_metadata=SimpleNamespace(num_tokens=num_tokens),
        input_ids=torch.zeros(num_tokens, dtype=torch.int32),
        spec_metadata=spec_md,
    )

    assert set(seen) == {0, last_idx}, "capture_set not honoured"

    # Replay layer 0 to rebuild what each candidate tensor would have been.
    br = torch.empty(num_layers, num_tokens, hidden)
    br[0] = embeds * 1
    after_l0 = embeds + 1

    proj1, norm1 = layer_w[1]
    torch.testing.assert_close(seen[0], _apply_attn_res(after_l0, br[:1], proj1, norm1))
    # The two ways this regresses, both silent:
    proj0, norm0 = layer_w[0]
    assert not torch.allclose(seen[0], _apply_attn_res(after_l0, br[:1], proj0, norm0)), (
        "tap used layer j's own weights instead of its successor's"
    )
    assert not torch.allclose(seen[0], after_l0), "tap captured the raw prefix sum"

    # Tail: no successor exists, so it must use the model's output-side
    # weights -- not the last layer's, and not the bare prefix sum.
    h = embeds
    for i in range(num_layers):
        br[i] = h * (i + 1)
        h = h + (i + 1)
    torch.testing.assert_close(
        seen[last_idx], _apply_attn_res(h, br[:num_layers], out_proj, out_norm)
    )
    proj_last, norm_last = layer_w[last_idx]
    assert not torch.allclose(
        seen[last_idx], _apply_attn_res(h, br[:num_layers], proj_last, norm_last)
    ), "tail used the last layer's weights instead of the output-side ones"
    assert not torch.allclose(seen[last_idx], h), "tail captured the raw prefix sum"


def test_aux_capture_taps_the_mixture(monkeypatch):
    """The real layer forward taps the pre-norm mixture, not the prefix sum.

    A tap of the other value only lowers acceptance -- nothing raises. The test
    above stubs the layer and so never runs the tap; this one calls the real
    layer forward.
    """
    pytest.importorskip("fla")
    from types import SimpleNamespace

    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default import (
        modeling as mkl,
    )

    monkeypatch.setattr(mkl, "flashinfer_rmsnorm", lambda h, weight, eps: h)

    torch.manual_seed(0)
    num_tokens, hidden, k_max, num_snapshots = 3, 8, 4, 2
    proj = torch.nn.Linear(hidden, 1, bias=False, dtype=torch.float32)
    norm = K31RMSNorm(hidden, eps=1e-6, dtype=torch.float32)
    with torch.no_grad():
        proj.weight.copy_(torch.randn(1, hidden))
        norm.weight.copy_(torch.randn(hidden) + 1.0)

    prefix_sum = torch.randn(num_tokens, hidden)
    block_residual = torch.randn(k_max, num_tokens, hidden)

    seen = {}
    layer = SimpleNamespace(
        self_attention_res_proj=proj,
        self_attention_res_norm=norm,
        # Not a block boundary, so prefix_sum survives to the end of forward.
        layer_idx=1,
        attn_res_block_size=4,
        input_layernorm=_IDENTITY_NORM,
        is_kda=True,
        linear_attn=lambda h, md: torch.zeros_like(h),
        mlp_res_proj=proj,
        mlp_res_norm=norm,
        post_attention_layernorm=_IDENTITY_NORM,
        is_moe=False,
        mlp=lambda h: torch.zeros_like(h),
    )
    # The layer taps through its own helper; bind the real one.
    layer._capture_hidden_states = MethodType(mkl.K31DecoderLayer._capture_hidden_states, layer)
    spec_md = SimpleNamespace(
        maybe_capture_hidden_states=lambda lid, h, r: seen.__setitem__(lid, h.clone())
    )

    mkl.K31DecoderLayer.forward(
        layer,
        prefix_sum.clone(),
        block_residual.clone(),
        num_snapshots,
        attn_metadata=SimpleNamespace(),
        capture=(spec_md, 0),
    )

    mixture = mkl._apply_attn_res(prefix_sum, block_residual[:num_snapshots], proj, norm)
    torch.testing.assert_close(seen[0], mixture)
    assert not torch.allclose(seen[0], prefix_sum), "the tap captured the raw prefix sum"


def test_aux_capture_tail_taps_the_mixture(monkeypatch):
    """The final layer has no successor, so the tail recomputes the mixture.

    A tail that captured the prefix sum would hand the last captured layer the
    wrong tensor -- the one inconsistency the in-loop test cannot see, because
    the loop never reaches the final layer's tap.
    """
    pytest.importorskip("fla")
    from types import SimpleNamespace

    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default import (
        modeling as mkl,
    )

    monkeypatch.setattr(mkl, "flashinfer_rmsnorm", lambda h, weight, eps: h)

    torch.manual_seed(1)
    num_layers, num_tokens, hidden = 3, 2, 8
    out_proj = torch.nn.Linear(hidden, 1, bias=False, dtype=torch.float32)
    out_norm = K31RMSNorm(hidden, eps=1e-6, dtype=torch.float32)
    with torch.no_grad():
        out_proj.weight.copy_(torch.randn(1, hidden) * 3.0)
        out_norm.weight.copy_(torch.randn(hidden) + 1.0)

    class _Layer:
        def __init__(self, idx):
            self.layer_idx = idx

        def __call__(
            self,
            hidden_states,
            block_residual,
            num_snapshots,
            attn_metadata,
            capture,
        ):
            block_residual[num_snapshots] = hidden_states * (self.layer_idx + 1)
            return hidden_states + (self.layer_idx + 1), num_snapshots + 1

    layers = [_Layer(i) for i in range(num_layers)]
    last_idx = num_layers - 1
    seen = {}
    # Only the final layer: the in-loop tap fires for layers[i-1], so this set
    # leaves the tail as the single capture site.
    spec_md = SimpleNamespace(
        _capture_layer_set=frozenset({last_idx}),
        maybe_capture_hidden_states=lambda lid, h, r: seen.__setitem__(lid, h.clone()),
    )
    embeds = torch.randn(num_tokens, hidden, dtype=torch.float32)
    fake = SimpleNamespace(
        embed_tokens=lambda ids: embeds,
        layers=layers,
        norm=_IDENTITY_NORM,
        output_attn_res_proj=out_proj,
        output_attn_res_norm=out_norm,
        num_attn_res_snapshots=num_layers,
    )
    fake._forward_layers = MethodType(mkl.K31TextModel._forward_layers, fake)

    mkl.K31TextModel.forward(
        fake,
        attn_metadata=SimpleNamespace(num_tokens=num_tokens),
        input_ids=torch.zeros(num_tokens, dtype=torch.int32),
        spec_metadata=spec_md,
    )

    br = torch.empty(num_layers, num_tokens, hidden)
    h = embeds
    for i in range(num_layers):
        br[i] = h * (i + 1)
        h = h + (i + 1)
    mixture = mkl._apply_attn_res(h, br[:num_layers], out_proj, out_norm)
    torch.testing.assert_close(seen[last_idx], mixture)
    assert not torch.allclose(seen[last_idx], h), "the tail captured the raw prefix sum"

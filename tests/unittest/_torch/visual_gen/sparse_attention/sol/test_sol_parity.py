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

"""Numerical parity of the VisualGen SOL backends.

The TRTLLM backend (Triton predictor plus the PrimTS block-sparse FMHA) and the
CuTeDSL backend (vendored fused kernel) are checked against dense attention,
against the exact-token/proxy-summary contract of the routes the predictor
emits, against an independent fp32 re-derivation of the SOL routing and
block-mean approximation, and against each other. Every test here needs an
SM100-family GPU.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha.prims_ts_block_sparse import PrimsTSBlockSparseFmha
from tensorrt_llm._torch.visual_gen.attention_backend.sparse.sol import predictor as sol_predictor
from tensorrt_llm._torch.visual_gen.attention_backend.sparse.sol.backend import (
    SOLCuTeDSLAttention,
    SOLTrtllmAttention,
)
from tensorrt_llm._torch.visual_gen.attention_backend.sparse.sol.predictor import (
    SolPredictorOutputs,
)
from tensorrt_llm._torch.visual_gen.attention_backend.utils import create_attention
from tensorrt_llm._torch.visual_gen.config import create_attention_metadata_state
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell import sol_attn_backend
from tensorrt_llm.visual_gen import SolAttentionConfig
from tensorrt_llm.visual_gen.args import AttentionConfig

_REQUIRES_SM100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="SOL requires SM100 or SM103",
)
_BLOCK = 64
_HEAD_DIM = 128
_THRESH_TYPES = pytest.mark.parametrize("thresh_type", ["diag", "exact"])


# --------------------------------------------------------------------------- references
def _dense_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * (_HEAD_DIM**-0.5)
    return torch.einsum("bhqk,bkhd->bqhd", scores.softmax(dim=-1), v.float()).to(q.dtype)


def _mixed_proxy_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    outputs: SolPredictorOutputs,
) -> torch.Tensor:
    """Evaluate the exact-token/proxy-summary attention contract of a predicted route."""

    num_blocks = math.ceil(k.shape[1] / _BLOCK)
    exact_words = outputs.exact_block_bits.detach().cpu().to(torch.int64)
    reference = torch.empty_like(q)
    scale = q.shape[-1] ** -0.5
    for batch_idx in range(q.shape[0]):
        for head_idx in range(q.shape[2]):
            for q_block_idx in range(math.ceil(q.shape[1] / _BLOCK)):
                q_begin = q_block_idx * _BLOCK
                q_end = min(q_begin + _BLOCK, q.shape[1])
                exact_blocks = [
                    block_idx
                    for block_idx in range(num_blocks)
                    if int(exact_words[batch_idx, head_idx, q_block_idx, block_idx // 32])
                    & (1 << (block_idx % 32))
                ]
                proxy_blocks = [
                    block_idx for block_idx in range(num_blocks) if block_idx not in exact_blocks
                ]
                exact_tokens = torch.cat(
                    [
                        torch.arange(
                            block_idx * _BLOCK,
                            min((block_idx + 1) * _BLOCK, k.shape[1]),
                            device=q.device,
                        )
                        for block_idx in exact_blocks
                    ]
                )
                q_rows = q[batch_idx, q_begin:q_end, head_idx].float()
                exact_logits = (q_rows @ k[batch_idx, exact_tokens, head_idx].float().T) * scale
                proxy_logits = (
                    q_rows @ outputs.k_summary[batch_idx, proxy_blocks, head_idx].float().T
                ) * scale
                logits = torch.cat((exact_logits, proxy_logits), dim=1)
                weights = torch.exp(logits - logits.amax(dim=1, keepdim=True))
                exact_weights = weights[:, : exact_tokens.numel()]
                proxy_weights = weights[:, exact_tokens.numel() :]
                numerator = exact_weights @ v[batch_idx, exact_tokens, head_idx].float()
                if proxy_blocks:
                    numerator += (
                        proxy_weights @ outputs.v_summary[batch_idx, proxy_blocks, head_idx].float()
                    )
                denominator = exact_weights.sum(dim=1, keepdim=True)
                for proxy_offset, block_idx in enumerate(proxy_blocks):
                    tokens_in_block = min(_BLOCK, k.shape[1] - block_idx * _BLOCK)
                    denominator += proxy_weights[:, proxy_offset : proxy_offset + 1] * (
                        tokens_in_block
                    )
                reference[batch_idx, q_begin:q_end, head_idx] = (numerator / denominator).to(
                    q.dtype
                )
    return reference


def _sol_reference(q, k, v, *, tau, scale, block=_BLOCK):
    """fp32 reference of the SOL ``diag`` routing and block-mean approximation.

    Mirrors the vendored kernel (``preprocess.py``, ``common/selector.py``,
    ``sm100/mainloop.py``):

    * ``kc`` is the per-block mean of K, ``vc`` the per-block sum of V.
    * Per query block the threshold is ``mean + tau * std`` in the log2 domain,
      where mean and variance come from the query centroid against the global
      K mean and the per-channel K variance.
    * A KV block is exact for a query block when the column-mean route score
      exceeds that threshold or when it lies within one block of the diagonal.
    * Non-exact blocks contribute ``exp(q . kc) * vc`` to the numerator and
      ``exp(q . kc) * block_len`` to the denominator: every key in the block is
      treated as its mean.

    Returns the output and the exact-routing mask ``[B, nb_q, nb_kv, H]``.
    """

    q, k, v = (t.float() for t in (q, k, v))
    B, S, H, D = q.shape
    nb = -(-S // block)
    log2e = math.log2(math.e)
    log2_scale = scale * log2e
    dev = q.device
    idx = torch.arange(S, device=dev) // block
    blen = torch.bincount(idx, minlength=nb).float()

    def _block_sum(t):
        return torch.zeros(B, nb, H, D, device=dev).index_add_(1, idx, t)

    kc = _block_sum(k) / blen[None, :, None, None]
    vc = _block_sum(v)
    qc = _block_sum(q) / blen[None, :, None, None]
    kmean = k.mean(dim=1)
    kvar = ((k * k).mean(dim=1) - kmean * kmean).clamp(min=0)
    mean = (qc * kmean[:, None]).sum(-1) * log2_scale
    var = (qc * qc * kvar[:, None]).sum(-1) * log2_scale**2
    thr = mean + tau * torch.sqrt(var.clamp(min=0) + 1e-6)  # [B, nb_q, H]

    s2 = torch.einsum("bshd,bjhd->bsjh", q, kc) * log2_scale  # [B, S, nb_kv, H]
    col_mean = (
        torch.zeros(B, nb, nb, H, device=dev).index_add_(1, idx, s2) / blen[None, :, None, None]
    )
    ar = torch.arange(nb, device=dev)
    near_diag = (ar[:, None] - ar[None, :]).abs() <= 1
    exact = (col_mean > thr[:, :, None, :]) | near_diag[None, :, :, None]  # [B, nb_q, nb_kv, H]

    exact_rows = exact[:, idx]  # [B, S, nb_kv, H]
    exact_pair = exact_rows[:, :, idx, :]  # [B, S, T, H]
    scores = torch.einsum("bshd,bthd->bsth", q, k) * scale
    approx = s2 / log2e
    neg = torch.finfo(torch.float32).min
    m = torch.maximum(
        scores.masked_fill(~exact_pair, neg).amax(2),
        approx.masked_fill(exact_rows, neg).amax(2),
    )
    pe = torch.exp(scores - m[:, :, None]) * exact_pair
    pa = torch.exp(approx - m[:, :, None]) * (~exact_rows)
    num = torch.einsum("bsth,bthd->bshd", pe, v) + torch.einsum("bsjh,bjhd->bshd", pa, vc)
    den = pe.sum(2) + (pa * blen[None, None, :, None]).sum(2)
    return num / den[..., None], exact


# --------------------------------------------------------------------------- backends
def _sol_config(*, tau: float, thresh_type: str = "diag", **kwargs) -> SolAttentionConfig:
    return SolAttentionConfig(tau=tau, thresh_type=thresh_type, **kwargs)


def _trtllm_backend(sparse_config: SolAttentionConfig, *, num_heads: int) -> SOLTrtllmAttention:
    attention_config = AttentionConfig(backend="TRTLLM", sparse_attention_config=sparse_config)
    backend = create_attention(
        backend="TRTLLM",
        layer_idx=1,
        num_heads=num_heads,
        head_dim=_HEAD_DIM,
        dtype=torch.bfloat16,
        attention_config=attention_config,
        attention_metadata_state=create_attention_metadata_state(),
    )
    assert isinstance(backend, SOLTrtllmAttention)
    assert any(isinstance(fmha, PrimsTSBlockSparseFmha) for fmha in backend._fmha_manager.fmha_libs)
    return backend


def _cutedsl_backend(sparse_config: SolAttentionConfig, *, num_heads: int) -> SOLCuTeDSLAttention:
    if not sol_attn_backend.sol_attn_supported(
        torch.empty(1, 8, num_heads, _HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    ):
        pytest.skip("no SOL CuTeDSL kernel for this device")
    attention_config = AttentionConfig(backend="CUTEDSL", sparse_attention_config=sparse_config)
    backend = create_attention(
        backend="CUTEDSL",
        layer_idx=1,
        num_heads=num_heads,
        head_dim=_HEAD_DIM,
        dtype=torch.bfloat16,
        attention_config=attention_config,
    )
    assert isinstance(backend, SOLCuTeDSLAttention)
    return backend


def _forward_trtllm(backend: SOLTrtllmAttention, q, k, v, *, timestep=None) -> torch.Tensor:
    batch_size, seq_len = q.shape[:2]
    return backend.forward(
        q=q,
        k=k,
        v=v,
        batch_size=batch_size,
        seq_len=seq_len,
        seq_len_kv=seq_len,
        timestep=timestep,
    ).view_as(q)


def _forward_cutedsl(backend: SOLCuTeDSLAttention, q, k, v) -> torch.Tensor:
    before = sol_attn_backend._SOL_STATS["kernel_calls"]
    out = backend.forward(q, k, v)
    torch.cuda.synchronize()
    assert sol_attn_backend._SOL_STATS["kernel_calls"] == before + 1, (
        "the CuTeDSL SOL kernel did not run; this would compare dense against dense"
    )
    return out


def _spy_primts_forward(monkeypatch) -> dict:
    """Record every PrimTS block-sparse FMHA launch; a dense fallback would leave it at zero."""

    calls = {"n": 0}
    original = PrimsTSBlockSparseFmha.forward

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(PrimsTSBlockSparseFmha, "forward", counted)
    return calls


def _packed_strided_inputs(generator, shape):
    """Q/K/V as non-contiguous split views of one packed projection, like the module."""

    packed = torch.randint(
        -2,
        3,
        (shape[0], shape[1], 3 * shape[2] * shape[3]),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    return tuple(tensor.view(shape) for tensor in packed.split(shape[2] * shape[3], dim=-1))


def _gaussian_inputs(seed: int, shape):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return tuple(
        torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    )


# --------------------------------------------------------------------------- TRTLLM backend
@_REQUIRES_SM100
@torch.no_grad()
def test_trtllm_sol_all_exact_routes_match_dense_under_cuda_graph() -> None:
    """A threshold no block can miss routes everything exact, so SOL equals dense attention."""

    backend = _trtllm_backend(_sol_config(tau=-1.0e6, disabled_until_timestep=0.6), num_heads=2)
    generator = torch.Generator(device="cuda").manual_seed(20260901)
    shape = (1, 257, 2, _HEAD_DIM)

    q, k, v = _packed_strided_inputs(generator, shape)
    timestep = torch.tensor(0.2, device="cuda")
    assert not any(tensor.is_contiguous() for tensor in (q, k, v))
    eager = _forward_trtllm(backend, q, k, v, timestep=timestep)
    torch.cuda.synchronize()
    torch.testing.assert_close(eager, _dense_reference(q, k, v), rtol=2e-2, atol=2e-2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _forward_trtllm(backend, q, k, v, timestep=timestep)

    next_q, next_k, next_v = _packed_strided_inputs(generator, shape)
    q.copy_(next_q)
    k.copy_(next_k)
    v.copy_(next_v)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(captured, _dense_reference(q, k, v), rtol=2e-2, atol=2e-2)


@_REQUIRES_SM100
@torch.no_grad()
@pytest.mark.parametrize("seq_len", [256, 257])
def test_trtllm_sol_mixed_proxy_routes_match_reference_under_cuda_graph(seq_len: int) -> None:
    """A threshold most blocks miss exercises the proxy path; the output follows the
    exact-token/proxy-summary contract of the predicted route, also after replay."""

    backend = _trtllm_backend(_sol_config(tau=1.0e6), num_heads=2)
    generator = torch.Generator(device="cuda").manual_seed(20260903)
    shape = (1, seq_len, 2, _HEAD_DIM)

    q, k, v = _packed_strided_inputs(generator, shape)
    eager = _forward_trtllm(backend, q, k, v)
    predictor_outputs = sol_predictor.predict(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        tau=1.0e6,
        sm_scale=_HEAD_DIM**-0.5,
    )
    torch.cuda.synchronize()
    exact_bits = predictor_outputs.exact_block_bits
    num_blocks = math.ceil(seq_len / _BLOCK)
    num_exact = sum(
        int(
            (exact_bits[..., block_idx // 32].to(torch.int64) >> (block_idx % 32))
            .bitwise_and(1)
            .sum()
            .item()
        )
        for block_idx in range(num_blocks)
    )
    assert 0 < num_exact < math.prod(exact_bits.shape[:3]) * num_blocks
    torch.testing.assert_close(
        eager, _mixed_proxy_reference(q, k, v, predictor_outputs), rtol=2e-2, atol=2e-2
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _forward_trtllm(backend, q, k, v)

    next_q, next_k, next_v = _packed_strided_inputs(generator, shape)
    q.copy_(next_q)
    k.copy_(next_k)
    v.copy_(next_v)
    graph.replay()
    torch.cuda.synchronize()

    # The captured graph re-predicts from the refreshed inputs; the reference
    # needs the summaries of those inputs too.
    replayed_outputs = sol_predictor.predict(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        tau=1.0e6,
        sm_scale=_HEAD_DIM**-0.5,
    )
    torch.testing.assert_close(
        captured, _mixed_proxy_reference(q, k, v, replayed_outputs), rtol=2e-2, atol=2e-2
    )


# --------------------------------------------------------------------------- CuTeDSL kernel
@_REQUIRES_SM100
def test_cutedsl_sol_kernel_matches_dense_on_a_single_block() -> None:
    """With ``tokens <= BLOCK`` there is exactly one KV block, so nothing can be routed
    away whatever ``tau`` says and the kernel must reproduce dense attention.

    ``kernel_calls`` is asserted to have advanced: without that, a fallback to
    dense would make this pass by comparing dense against itself.
    """

    if not sol_attn_backend.sol_attn_supported(
        torch.empty(1, 8, 8, _HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    ):
        pytest.skip("no SOL CuTeDSL kernel for this device")

    torch.manual_seed(0)
    shape = (1, _BLOCK, 4, _HEAD_DIM)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float()
    ).transpose(1, 2)

    before = sol_attn_backend._SOL_STATS["kernel_calls"]
    out = sol_attn_backend._run_sol_attn_bthd(q, k, v, tau=2.0, thresh_type="diag", kv_splits=1)
    torch.cuda.synchronize()
    assert sol_attn_backend._SOL_STATS["kernel_calls"] == before + 1, (
        "the kernel did not run; this comparison would be dense against dense"
    )

    torch.testing.assert_close(out.float(), reference, rtol=2e-2, atol=2e-2)


@_REQUIRES_SM100
def test_cutedsl_sol_kernel_matches_reference_with_approximated_blocks() -> None:
    """The kernel agrees with the fp32 reference where blocks are approximated.

    Plain Gaussian inputs are the right stimulus: every KV block carries
    comparable softmax mass and no block's column-mean score clears the
    ``mean + tau*std`` threshold, so every block two or more away from the
    diagonal is approximated, and because keys vary within a block, treating 64
    keys as their mean is measurably different from dense attention. The test
    asserts that some blocks were approximated and that the result differs
    from dense, so neither an accidental all-exact run nor a mass-free
    approximation can pass.
    """

    if not sol_attn_backend.sol_attn_supported(
        torch.empty(1, 8, 8, _HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    ):
        pytest.skip("no SOL CuTeDSL kernel for this device")

    torch.manual_seed(0)
    shape = (1, 256, 2, _HEAD_DIM)  # 4 KV blocks
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    scale = _HEAD_DIM**-0.5
    tau = 2.0

    ref, exact = _sol_reference(q, k, v, tau=tau, scale=scale)
    assert (~exact).any(), "inputs did not force any block onto the approximation path"
    dense = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float()
    ).transpose(1, 2)
    # Calibrated on B200: max |ref - dense| = 0.45 for this stimulus. A tight
    # margin here would let a mass-free approximation pass vacuously.
    assert (ref - dense).abs().max() > 1e-1, "approximation is not measurably different from dense"

    before = sol_attn_backend._SOL_STATS["kernel_calls"]
    out = sol_attn_backend._run_sol_attn_bthd(q, k, v, tau=tau, thresh_type="diag", kv_splits=1)
    torch.cuda.synchronize()
    assert sol_attn_backend._SOL_STATS["kernel_calls"] == before + 1, "kernel did not run"

    # Calibrated on B200: max |kernel - ref| = 1.7e-3 (mean 2e-4); 5e-3 leaves
    # ~3x headroom for bf16 rounding while still catching a routing or
    # approximation-formula mismatch, which shows up at 1e-1 or worse.
    torch.testing.assert_close(out.float(), ref, rtol=5e-3, atol=5e-3)


# --------------------------------------------------------------------------- TRTLLM vs CuTeDSL vs dense
@_REQUIRES_SM100
@_THRESH_TYPES
@torch.no_grad()
def test_sol_backends_match_dense_when_every_block_is_exact(monkeypatch, thresh_type) -> None:
    """Both backends reproduce dense attention when the threshold routes every block exact."""

    num_heads = 2
    q, k, v = _gaussian_inputs(20260921, (2, 320, num_heads, _HEAD_DIM))
    dense = _dense_reference(q, k, v)
    primts_calls = _spy_primts_forward(monkeypatch)
    config = _sol_config(tau=-1.0e6, thresh_type=thresh_type)

    trtllm = _forward_trtllm(_trtllm_backend(config, num_heads=num_heads), q, k, v)
    cutedsl = _forward_cutedsl(_cutedsl_backend(config, num_heads=num_heads), q, k, v)
    torch.cuda.synchronize()

    assert primts_calls["n"] == 1
    torch.testing.assert_close(trtllm, dense, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(cutedsl, dense, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(trtllm, cutedsl, rtol=2e-2, atol=2e-2)


def _predicted_routes(q, k, v, *, tau: float, thresh_type: str) -> SolPredictorOutputs:
    outputs = sol_predictor.predict(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        tau=tau,
        sm_scale=_HEAD_DIM**-0.5,
        thresh_type=thresh_type,
    )
    torch.cuda.synchronize()
    return outputs


def _exact_block_ratio(outputs: SolPredictorOutputs, num_blocks: int) -> float:
    bits = outputs.exact_block_bits.to(torch.int64)
    exact = sum(
        int(((bits[..., block // 32] >> (block % 32)) & 1).sum().item())
        for block in range(num_blocks)
    )
    return exact / (math.prod(bits.shape[:3]) * num_blocks)


@_REQUIRES_SM100
@_THRESH_TYPES
# From a threshold few blocks reach, through one inside the score distribution of this
# stimulus, to one most blocks clear.
@pytest.mark.parametrize("tau", [2.0, 1.0, 0.0])
@torch.no_grad()
def test_sol_backends_follow_the_same_routes(monkeypatch, thresh_type, tau) -> None:
    """Both backends execute the routes the predictor emits, at every threshold.

    The oracle is the exact-token/proxy-summary contract evaluated on the predicted
    routes, so a backend that routed a borderline block differently from the
    predictor fails with a localized error even though a global similarity to
    dense attention would barely move. A threshold sitting inside the score
    distribution (``tau=1`` here) is exactly the case an independent fp32
    re-derivation of the routing cannot arbitrate, and this test covers it.
    """

    num_heads, num_blocks = 2, 5
    q, k, v = _gaussian_inputs(20260922, (2, num_blocks * _BLOCK, num_heads, _HEAD_DIM))
    dense = _dense_reference(q, k, v).float()
    routes = _predicted_routes(q, k, v, tau=tau, thresh_type=thresh_type)
    ratio = _exact_block_ratio(routes, num_blocks)
    assert 0.0 < ratio < 1.0, f"tau={tau} left no block on the proxy path ({ratio:.2f} exact)"
    reference = _mixed_proxy_reference(q, k, v, routes).float()
    primts_calls = _spy_primts_forward(monkeypatch)
    config = _sol_config(tau=tau, thresh_type=thresh_type)

    trtllm = _forward_trtllm(_trtllm_backend(config, num_heads=num_heads), q, k, v).float()
    cutedsl = _forward_cutedsl(_cutedsl_backend(config, num_heads=num_heads), q, k, v).float()
    torch.cuda.synchronize()

    assert primts_calls["n"] == 1
    torch.testing.assert_close(trtllm, cutedsl, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(trtllm, reference, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(cutedsl, reference, rtol=2e-2, atol=2e-2)
    # The approximation is active: the shared output is measurably not dense attention.
    assert (trtllm - dense).abs().max() > 1e-1


@_REQUIRES_SM100
@torch.no_grad()
def test_sol_backends_match_independent_reference_away_from_the_threshold(monkeypatch) -> None:
    """Both backends reproduce an independent fp32 re-derivation of the ``diag`` routing
    and the block-mean approximation.

    ``tau=2`` on Gaussian inputs leaves only the near-diagonal blocks exact, far from
    the threshold. Right at the threshold the fp32 re-derivation and the bf16 kernels
    can legitimately route a block differently, so borderline thresholds are covered
    by ``test_sol_backends_follow_the_same_routes`` against the predicted routes.
    """

    num_heads = 2
    q, k, v = _gaussian_inputs(20260922, (2, 320, num_heads, _HEAD_DIM))
    reference, exact = _sol_reference(q, k, v, tau=2.0, scale=_HEAD_DIM**-0.5)
    assert (~exact).any()
    dense = _dense_reference(q, k, v).float()
    assert (reference - dense).abs().max() > 1e-1
    primts_calls = _spy_primts_forward(monkeypatch)
    config = _sol_config(tau=2.0, thresh_type="diag")

    trtllm = _forward_trtllm(_trtllm_backend(config, num_heads=num_heads), q, k, v).float()
    cutedsl = _forward_cutedsl(_cutedsl_backend(config, num_heads=num_heads), q, k, v).float()
    torch.cuda.synchronize()

    assert primts_calls["n"] == 1
    torch.testing.assert_close(trtllm, reference, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(cutedsl, reference, rtol=1e-2, atol=1e-2)

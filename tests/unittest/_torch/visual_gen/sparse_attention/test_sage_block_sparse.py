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
"""Sage quantization of VSA and SOL block-sparse attention on the TRTLLM backend.

Backends come from the VisualGen factory with a validated ``AttentionConfig``.
Lossy checks compare against the same sparse backend with Sage disabled on
identical BF16 inputs. CUDA Graph replay and masking checks compare against
eager or unquantized execution with near-exact tolerances.
"""

from typing import Literal

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention.backends.fmha.prims_ts_block_sparse import PrimsTSBlockSparseFmha
from tensorrt_llm._torch.attention.backends.prims_ts import BlockSparseTSWrapper
from tensorrt_llm._torch.attention.backends.sparse.params import (
    BlockSparseForwardInputs,
    SparseBackendForwardArgs,
)
from tensorrt_llm._torch.visual_gen.attention.backends.sparse.sol import predictor as sol_predictor
from tensorrt_llm._torch.visual_gen.attention.backends.sparse.vsa.metadata import (
    VSAMetadataBuilder,
    set_vsa_forward_context,
)
from tensorrt_llm._torch.visual_gen.attention.backends.trtllm import TrtllmAttention
from tensorrt_llm._torch.visual_gen.attention.backends.utils import create_attention
from tensorrt_llm._torch.visual_gen.config import create_attention_metadata_state
from tensorrt_llm.visual_gen.args import (
    AttentionConfig,
    QuantAttentionConfig,
    SolAttentionConfig,
    VideoSparseAttentionConfig,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Sage block-sparse attention requires SM100 or SM103",
)

_QuantDtype = Literal["int8", "fp8"]
_SolPhase = Literal["proxy", "exact", "dense"]
_HEADS = 2
_HEAD_DIM = 128
_BATCH_SIZE = 2
# B200, BF16 inputs: INT8 worst relL2/cosine/maxAbs = .03618/.999403/.140625;
# FP8 = .04979/.998761/.167969. The bounds leave about 25-35% headroom.
_LOSSY_BOUNDS = {
    "int8": {
        "relative_l2": 0.045,
        "cosine": 0.9991,
        "max_abs": 0.18,
        "p99_abs": 0.035,
        "channel_mean_shift": 0.0075,
    },
    "fp8": {
        "relative_l2": 0.065,
        "cosine": 0.9982,
        "max_abs": 0.22,
        "p99_abs": 0.05,
        "channel_mean_shift": 0.0095,
    },
}


def _backend(
    qk_dtype: _QuantDtype | None,
    sparse_config: SolAttentionConfig | VideoSparseAttentionConfig | None = None,
    *,
    k_block_size: int | None = None,
    smooth_v: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> TrtllmAttention:
    quant_config = None
    if qk_dtype is not None:
        if qk_dtype == "int8" and torch.cuda.get_device_capability() == (10, 3):
            pytest.skip("SM103 does not have INT8 Tensor Cores")
        if k_block_size is None:
            k_block_size = 16 if qk_dtype == "int8" else 4
        quant_config = QuantAttentionConfig(
            qk_dtype=qk_dtype,
            v_dtype="fp8",
            q_block_size=1,
            k_block_size=k_block_size,
            v_block_size=1,
            smooth_v=smooth_v,
        )
    return create_attention(
        backend="TRTLLM",
        layer_idx=1,
        num_heads=_HEADS,
        head_dim=_HEAD_DIM,
        dtype=dtype,
        attention_config=AttentionConfig(
            backend="TRTLLM",
            quant_attention_config=quant_config,
            sparse_attention_config=sparse_config,
        ),
        attention_metadata_state=create_attention_metadata_state(),
    )


def _sol(
    qk_dtype: _QuantDtype | None,
    *,
    phase: _SolPhase = "proxy",
    k_block_size: int | None = None,
    smooth_v: bool = False,
) -> TrtllmAttention:
    sparse_config = SolAttentionConfig(
        tau=-1.0e6 if phase == "exact" else 1.0e6, disabled_until_timestep=0.6
    )
    return _backend(qk_dtype, sparse_config, k_block_size=k_block_size, smooth_v=smooth_v)


def _inputs(seq_len: int, *, seed: int = 20260921) -> tuple[torch.Tensor, ...]:
    """Strided projection views with batch-dependent scales and a nonzero V mean."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    packed = torch.randn(
        _BATCH_SIZE,
        seq_len,
        3,
        _HEADS,
        _HEAD_DIM,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    batch_scales = torch.tensor([0.8, 1.2], device="cuda", dtype=packed.dtype)
    channel_bias = torch.linspace(-0.2, 0.2, _HEAD_DIM, device="cuda", dtype=packed.dtype)
    packed = packed * batch_scales.view(_BATCH_SIZE, 1, 1, 1, 1) + channel_bias
    return packed.unbind(dim=2)


def _forward_sol(
    backend: TrtllmAttention,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    phase: _SolPhase = "proxy",
) -> torch.Tensor:
    return backend.forward(
        q=q,
        k=k,
        v=v,
        batch_size=q.shape[0],
        seq_len=q.shape[1],
        seq_len_kv=q.shape[1],
        timestep=0.8 if phase == "dense" else 0.2,
    ).view_as(q)


def _sparse_wrappers(backend: TrtllmAttention) -> list[BlockSparseTSWrapper]:
    """Block-sparse wrappers planned in the backend's shared metadata plan caches."""
    return [
        wrapper
        for entry in backend.metadata._metadata_cache.values()
        for wrapper in entry["metadata"]
        .fmha_plan_caches.get(PrimsTSBlockSparseFmha.PLAN_CACHE_KEY, {})
        .values()
    ]


def _assert_sparse_sage_active(
    backend: TrtllmAttention,
    *,
    proxy: bool,
    token_mask: bool,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    wrappers = _sparse_wrappers(backend)
    assert wrappers, "Sage silently bypassed the PrimTS block-sparse FMHA"
    quant_config = backend.quant_attention_config
    qk_dtype = torch.int8 if quant_config.qk_dtype == "int8" else torch.float8_e4m3fn
    for wrapper in wrappers:
        state = wrapper._plan_state
        assert state is not None and state.sage is not None
        assert state.use_block_sparse
        assert state.use_proxy_routes == proxy
        assert state.use_kv_valid_bits == token_mask
        assert state.q_dtype == qk_dtype
        assert state.v_dtype == torch.float8_e4m3fn
        assert state.output_dtype == dtype
        assert state.sage.q_block_size == quant_config.q_block_size
        assert state.sage.k_block_size == quant_config.k_block_size
        assert state.sage.v_mean == quant_config.smooth_v


def _parity_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual, expected = actual.float(), expected.float()
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    delta = actual - expected
    return {
        "relative_l2": (
            torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(expected)
        ).item(),
        "cosine": F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item(),
        "max_abs": delta.abs().max().item(),
        "p99_abs": torch.quantile(delta.abs().flatten(), 0.99).item(),
        "channel_mean_shift": delta.mean(dim=(0, 1)).abs().max().item(),
    }


def _assert_parity(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
    qk_dtype: _QuantDtype | None = None,
) -> None:
    """Check lossy Sage bounds for ``qk_dtype``, or near-exact agreement without one."""
    stats = _parity_stats(actual, expected)
    report = f"{label}, shape={tuple(actual.shape)}, dtype={actual.dtype}: {stats}"
    print(report)
    if qk_dtype is None:
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3, msg=report)
        assert stats["relative_l2"] <= 1e-2, report
        assert stats["cosine"] >= 0.9999, report
        return
    bounds = _LOSSY_BOUNDS[qk_dtype]
    assert stats["cosine"] > bounds["cosine"], report
    for name in ("relative_l2", "max_abs", "p99_abs", "channel_mean_shift"):
        assert stats[name] < bounds[name], report
    assert stats["max_abs"] > 0.0, "Sage quantization had no observable effect"


@pytest.mark.parametrize("qk_dtype", ["int8", "fp8"])
@torch.no_grad()
def test_vsa_sage_matches_unquantized_sparse_attention(qk_dtype: _QuantDtype) -> None:
    """Quantize the sparse fine stage while keeping VSA padding and both gates."""
    metadata = VSAMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=(3, 7, 9),
        patch_size=(1, 1, 1),
        vsa_sparsity=0.5,
        device=torch.device("cuda"),
    )
    q, k, v = _inputs(3 * 7 * 9)
    assert metadata.padded_seq_length > q.shape[1]
    assert not q.is_contiguous()
    gate_compress = 0.25 + 0.1 * q.sigmoid()
    gate_fine = 0.7 + 0.2 * k.sigmoid()
    vsa_config = VideoSparseAttentionConfig(vsa_sparsity=0.5)
    reference, candidate = _backend(None, vsa_config), _backend(qk_dtype, vsa_config)
    with set_vsa_forward_context(metadata):
        kwargs = dict(
            q=q,
            k=k,
            v=v,
            batch_size=q.shape[0],
            seq_len=q.shape[1],
            gate_compress=gate_compress,
            gate_fine=gate_fine,
        )
        expected = reference.forward(**kwargs).view_as(q)
        actual = candidate.forward(**kwargs).view_as(q)

    _assert_sparse_sage_active(candidate, proxy=False, token_mask=True)
    assert _sparse_wrappers(reference)
    _assert_parity(
        actual,
        expected,
        label=f"VSA Sage {qk_dtype}, padded coarse+fine",
        qk_dtype=qk_dtype,
    )


@pytest.mark.parametrize(
    ("qk_dtype", "phase", "k_block_size"),
    [
        ("int8", "proxy", 16),
        ("fp8", "proxy", 1),
        ("fp8", "exact", 4),
        ("fp8", "dense", 4),
    ],
)
@torch.no_grad()
def test_sol_sage_matches_unquantized_attention(
    qk_dtype: _QuantDtype, phase: _SolPhase, k_block_size: int
) -> None:
    """Identical SOL routes and means, including tail blocks and the dense timestep prefix."""
    q, k, v = _inputs(131)
    reference = _sol(None, phase=phase)
    candidate = _sol(qk_dtype, phase=phase, k_block_size=k_block_size)
    expected = _forward_sol(reference, q, k, v, phase=phase)
    actual = _forward_sol(candidate, q, k, v, phase=phase)

    if phase == "dense":
        assert not _sparse_wrappers(candidate), "the dense timestep unexpectedly ran SOL"
        assert not _sparse_wrappers(reference)
    else:
        _assert_sparse_sage_active(candidate, proxy=True, token_mask=False)
        outputs = sol_predictor.predict(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            tau=candidate.sol_params.tau,
            sm_scale=_HEAD_DIM**-0.5,
        )
        exact_bits = outputs.exact_block_bits.to(torch.int64)
        num_blocks = outputs.k_summary.shape[1]
        selected = torch.stack(
            [(exact_bits[..., idx // 32] >> (idx % 32)) & 1 for idx in range(num_blocks)]
        )
        if phase == "exact":
            assert selected.all(), "the all-exact SOL policy retained proxy routes"
        else:
            assert selected.any() and not selected.all(), "SOL did not mix exact and proxy routes"
    _assert_parity(
        actual,
        expected,
        label=f"SOL Sage {qk_dtype}, Kblock={k_block_size}, {phase}",
        qk_dtype=qk_dtype,
    )


@pytest.mark.parametrize("case", ["sol_proxy", "sol_dense_timestep", "dense_model"])
@torch.no_grad()
def test_sage_smooth_v_matches_unquantized_attention(case: str) -> None:
    """V smoothing on mixed SOL routes, and on dense calls, which then run on a PrimTS dense plan."""
    q, k, v = _inputs(131)
    if case == "dense_model":
        reference, candidate = _backend(None), _backend("fp8", smooth_v=True)
        expected, actual = (
            backend.forward(q=q, k=k, v=v, batch_size=q.shape[0], seq_len=q.shape[1]).view_as(q)
            for backend in (reference, candidate)
        )
    else:
        phase = "dense" if case == "sol_dense_timestep" else "proxy"
        reference, candidate = _sol(None, phase=phase), _sol("fp8", phase=phase, smooth_v=True)
        expected = _forward_sol(reference, q, k, v, phase=phase)
        actual = _forward_sol(candidate, q, k, v, phase=phase)

    wrappers = _sparse_wrappers(candidate)
    assert wrappers, "V smoothing bypassed the PrimTS kernels"
    for wrapper in wrappers:
        state = wrapper._plan_state
        assert state.sage is not None and state.sage.v_mean
        assert state.use_block_sparse == (case == "sol_proxy")
    _assert_parity(actual, expected, label=f"Sage fp8 smooth_v, {case}", qk_dtype="fp8")


@torch.no_grad()
def test_sol_sage_cuda_graph_refreshes_inputs_and_scales() -> None:
    """A captured mixed-proxy Sage layer consumes new Q/K/V and scales on replay."""
    backend = _sol("fp8")
    q, k, v = _inputs(131)
    previous = _forward_sol(backend, q, k, v).clone()
    _assert_sparse_sage_active(backend, proxy=True, token_mask=False)
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        _forward_sol(backend, q, k, v)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _forward_sol(backend, q, k, v)
    for destination, source in zip((q, k, v), _inputs(131, seed=20260922)):
        destination.copy_(source)
    graph.replay()
    replayed = captured.clone()
    expected = _forward_sol(backend, q, k, v)
    assert _parity_stats(expected, previous)["relative_l2"] > 0.1
    _assert_parity(replayed, expected, label="SOL Sage CUDA Graph replay")


def _masked_zero_score_inputs(
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, BlockSparseForwardInputs]:
    shape = (_BATCH_SIZE, 131, _HEADS, _HEAD_DIM)
    magnitudes = torch.tensor([0.0, 1.0e-8], device="cuda", dtype=dtype)
    q = torch.ones(shape, device="cuda", dtype=dtype) * magnitudes[:, None, None, None]
    tokens = torch.arange(shape[1], device=q.device)
    visible = (tokens >= 68) & (tokens < 76)
    # Masked V=4 exposes leaked probability; the eight visible tokens have V=1.
    v = torch.where(visible, 1.0, 4.0)[None, :, None, None].expand(shape).to(q.dtype).contiguous()
    # Query blocks select a fully masked block, a partly visible block, then no block.
    bits = torch.tensor([1, 2, 0], device=q.device, dtype=torch.uint32)
    valid_words = torch.tensor([0, 0, 0xFF0, 0, 0], device=q.device, dtype=torch.uint32)
    routes = BlockSparseForwardInputs(
        q_block_size=64,
        kv_block_size=64,
        exact_block_bits=bits.view(1, 1, 3, 1).expand(_BATCH_SIZE, _HEADS, 3, 1).contiguous(),
        kv_valid_bits=valid_words[None, :].expand(_BATCH_SIZE, -1).contiguous(),
    )
    return q, -q, v, routes


@pytest.mark.parametrize(
    ("dtype", "qk_dtype"),
    [(torch.bfloat16, "int8"), (torch.float16, "fp8")],
    ids=["bf16-int8", "fp16-fp8"],
)
@torch.no_grad()
def test_sage_zero_scores_preserve_token_masks_and_empty_rows(
    qk_dtype: _QuantDtype, dtype: torch.dtype
) -> None:
    """Zero and tiny Q/K keep token masking and empty rows exact through quantization."""
    q, k, v, routes = _masked_zero_score_inputs(dtype)
    backends = (_backend(None, dtype=dtype), _backend(qk_dtype, dtype=dtype))
    expected, actual = (
        backend.forward(
            q,
            k,
            v,
            batch_size=q.shape[0],
            seq_len=q.shape[1],
            sparse_backend_args=SparseBackendForwardArgs(block_sparse_inputs=routes),
        ).view_as(q)
        for backend in backends
    )
    _assert_sparse_sage_active(backends[1], proxy=False, token_mask=True, dtype=dtype)
    _assert_parity(actual, expected, label=f"{qk_dtype} zero/tiny scores with masked rows")
    assert torch.count_nonzero(actual[:, :64]).item() == 0, "fully masked rows must be zero"
    assert torch.count_nonzero(actual[:, 128:]).item() == 0, "empty route rows must be zero"
    torch.testing.assert_close(actual[:, 64:128], torch.ones_like(actual[:, 64:128]))


@torch.no_grad()
def test_sol_sage_zero_values_preserve_zero_output() -> None:
    """Exact and proxy SOL contributions stay zero through the V-scale floor.

    The inputs and tau match the mixed-route proxy cases above, so both route
    kinds contribute.
    """
    q, k, v = _inputs(131)
    zero_v = torch.zeros_like(v)
    candidate = _sol("fp8")
    actual = _forward_sol(candidate, q, k, zero_v)
    _assert_sparse_sage_active(candidate, proxy=True, token_mask=False)
    assert torch.count_nonzero(actual).item() == 0, "zero exact values and proxy means stay zero"

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

"""Behavior tests for the VisualGen SOL backends.

The TRTLLM backend predicts SOL routes inside the core prediction hook and
executes them through the PrimTS block-sparse FMHA; the CuTeDSL backend runs
the vendored fused kernel. This module covers dispatch, the public config, the
timestep schedule, CUDA Graph phase keys and the guards of both backends.
Numerical checks against dense attention, the fp32 SOL reference and each
other live in ``test_sol_parity.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from pydantic import ValidationError

from tensorrt_llm._torch.attention.backends.fmha.prims_ts_block_sparse import PrimsTSBlockSparseFmha
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention.backends.sparse.hooks import prepare_sparse_runtime_params
from tensorrt_llm._torch.attention.backends.sparse.params import BlockSparseForwardInputs
from tensorrt_llm._torch.attention.backends.sparse.timestep_phase import graph_phase_for_timestep
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention as CoreTrtllmAttention
from tensorrt_llm._torch.visual_gen.attention_backend import CuTeDSLAttention
from tensorrt_llm._torch.visual_gen.attention_backend.sparse.sol import backend as sol_backend
from tensorrt_llm._torch.visual_gen.attention_backend.sparse.sol import predictor as sol_predictor
from tensorrt_llm._torch.visual_gen.attention_backend.sparse.sol.backend import (
    SOLCuTeDSLAttention,
    SOLTrtllmAttention,
)
from tensorrt_llm._torch.visual_gen.attention_backend.sparse.sol.params import SolParams
from tensorrt_llm._torch.visual_gen.attention_backend.sparse.sol.predictor import (
    SolPredictorOutputs,
)
from tensorrt_llm._torch.visual_gen.attention_backend.trtllm import (
    TrtllmAttention,
    TrtllmAttentionMetadata,
)
from tensorrt_llm._torch.visual_gen.attention_backend.utils import create_attention
from tensorrt_llm._torch.visual_gen.config import (
    DiffusionModelConfig,
    create_attention_metadata_state,
)
from tensorrt_llm._torch.visual_gen.cuda_graph_runner import (
    CUDAGraphRunner,
    CUDAGraphRunnerConfig,
    resolved_extra_key,
    resolved_extra_keys_scope,
)
from tensorrt_llm._torch.visual_gen.models.modeling import BaseDiffusionModel
from tensorrt_llm._torch.visual_gen.modules import attention as attention_module
from tensorrt_llm._torch.visual_gen.modules.attention import Attention, QKVMode
from tensorrt_llm.visual_gen import SolAttentionConfig
from tensorrt_llm.visual_gen.args import AttentionConfig, QuantAttentionConfig

_CPU_ONLY = pytest.mark.cpu_only
_REQUIRES_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
_BACKENDS = pytest.mark.parametrize("backend", ["TRTLLM", "CUTEDSL"])


# --------------------------------------------------------------------------- TRTLLM helpers
def _make_backend(params: SolParams, *, layer_idx: int = 1) -> SOLTrtllmAttention:
    backend = object.__new__(SOLTrtllmAttention)
    backend.layer_idx = layer_idx
    backend.num_heads = 2
    backend.num_kv_heads = 2
    backend.head_dim = 128
    backend.q_scaling = 1.0
    backend.quant_attention_config = None
    backend.sparse_params = None
    backend._fmha_manager = SimpleNamespace(fmha_libs=[object.__new__(PrimsTSBlockSparseFmha)])
    backend.sol_params = params
    backend.metadata = TrtllmAttentionMetadata(
        device=torch.device("cpu"), attention_metadata_state={}
    )
    return backend


def _flatten(tensor: torch.Tensor | None) -> torch.Tensor | None:
    """Convert a BSHD tensor into the flattened ``[B*S, H*D]`` core layout."""

    if tensor is None:
        return None
    return tensor.reshape(tensor.shape[0] * tensor.shape[1], -1)


def _core_metadata(batch_size: int, seq_len: int) -> SimpleNamespace:
    return SimpleNamespace(num_seqs=batch_size, max_seq_len=seq_len)


def _predict(
    backend: SOLTrtllmAttention,
    q: torch.Tensor,
    k: torch.Tensor | None,
    v: torch.Tensor | None,
    *,
    attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
    timestep: object = None,
) -> BlockSparseForwardInputs | None:
    """Invoke the core prediction hook the way the core forward does."""

    return backend.block_sparse_attn_predict(
        _flatten(q),
        _flatten(k),
        _flatten(v),
        _core_metadata(q.shape[0], q.shape[1]),
        AttentionForwardArgs(attention_mask=attention_mask, timestep=timestep),
    )


def _forward(
    backend: SOLTrtllmAttention,
    q: torch.Tensor,
    k: torch.Tensor | None,
    v: torch.Tensor | None,
    **kwargs,
) -> torch.Tensor:
    seq_len_kv = kwargs.pop("seq_len_kv", q.shape[1])
    return backend.forward(
        q=q,
        k=k,
        v=v,
        batch_size=q.shape[0],
        seq_len=q.shape[1],
        seq_len_kv=seq_len_kv,
        **kwargs,
    )


def _stub_core_forward(monkeypatch) -> dict:
    """Replace metadata preparation and the core forward with a recorder that
    still runs the backend's sparse prediction."""

    captured = {}
    monkeypatch.setattr(
        TrtllmAttention,
        "_prepare_metadata",
        lambda self, batch_size, seq_len: _core_metadata(batch_size, seq_len),
    )

    def _core_forward(self, q, k, v, metadata, forward_args=None, **kwargs):
        forward_args.sparse_runtime_params = prepare_sparse_runtime_params(
            self, q, k, v, metadata, forward_args
        )
        captured.update(q=q, k=k, v=v, metadata=metadata, forward_args=forward_args)
        return q

    monkeypatch.setattr(CoreTrtllmAttention, "forward", _core_forward)
    return captured


def _predictor_outputs(*, batch_size: int, seq_len: int, num_heads: int) -> SolPredictorOutputs:
    num_blocks = (seq_len + 63) // 64
    return SolPredictorOutputs(
        exact_block_bits=torch.zeros(
            batch_size,
            num_heads,
            num_blocks,
            (num_blocks + 31) // 32,
            dtype=torch.uint32,
        ),
        k_summary=torch.zeros(batch_size, num_blocks, num_heads, 128, dtype=torch.bfloat16),
        v_summary=torch.zeros(batch_size, num_blocks, num_heads, 128, dtype=torch.bfloat16),
    )


def _bshd(seq_len: int = 64, num_heads: int = 2) -> torch.Tensor:
    return torch.zeros(1, seq_len, num_heads, 128, dtype=torch.bfloat16)


def _stub_backend(
    monkeypatch,
    params: SolParams | None = None,
    *,
    seq_len: int = 64,
    unsupported_reason: str | None = None,
) -> tuple[SOLTrtllmAttention, SimpleNamespace]:
    """Backend whose predictor functions are recorded mocks."""
    predictor = SimpleNamespace(
        support_reason=Mock(return_value=unsupported_reason),
        predict=Mock(return_value=_predictor_outputs(batch_size=1, seq_len=seq_len, num_heads=2)),
    )
    monkeypatch.setattr(sol_predictor, "support_reason", predictor.support_reason)
    monkeypatch.setattr(sol_predictor, "predict", predictor.predict)
    return _make_backend(params or SolParams(tau=1.0)), predictor


# --------------------------------------------------------------------------- TRTLLM backend
@_CPU_ONLY
def test_sol_backend_reuses_prepared_timestep_during_cuda_graph_capture(monkeypatch) -> None:
    backend, _predictor = _stub_backend(
        monkeypatch, SolParams(tau=1.0, disabled_until_timestep=0.6)
    )
    # Per-token timesteps reduce to the largest live value.
    assert backend.resolve_timestep(torch.tensor([0.0, 0.2])) == pytest.approx(0.2)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    prepared = backend.resolve_timestep(torch.tensor(0.8))

    assert prepared == pytest.approx(0.2)
    assert backend.should_use_sparse(prepared)


@_CPU_ONLY
def test_sol_backend_warmup_prepares_dense_phase_for_capture(monkeypatch) -> None:
    q = _bshd()
    backend, predictor = _stub_backend(monkeypatch, SolParams(tau=1.0, disabled_until_timestep=0.6))

    assert _predict(backend, q, q, q, timestep=backend.resolve_timestep(torch.tensor(0.8))) is None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert _predict(backend, q, q, q, timestep=backend.resolve_timestep(torch.tensor(0.8))) is None
    predictor.predict.assert_not_called()


@_CPU_ONLY
def test_sol_backend_rejects_cutoff_capture_without_warmup(monkeypatch) -> None:
    backend, predictor = _stub_backend(monkeypatch, SolParams(tau=1.0, disabled_until_timestep=0.6))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="prepared before CUDA Graph capture"):
        backend.resolve_timestep(torch.tensor(0.2))

    predictor.predict.assert_not_called()


@_CPU_ONLY
def test_sol_backend_without_timestep_runs_sparse_like_skip_softmax(monkeypatch) -> None:
    q = _bshd()
    backend, predictor = _stub_backend(monkeypatch, SolParams(tau=1.0, disabled_until_timestep=0.6))

    assert _predict(backend, q, q, q, timestep=backend.resolve_timestep(None)) is not None
    predictor.predict.assert_called_once()


@_CPU_ONLY
def test_sol_phase_waits_until_all_token_timesteps_are_below_cutoff(monkeypatch) -> None:
    backend, _predictor = _stub_backend(
        monkeypatch, SolParams(tau=1.0, disabled_until_timestep=0.6)
    )

    assert not backend.should_use_sparse(backend.resolve_timestep(torch.tensor([0.0, 0.8])))
    assert backend.should_use_sparse(backend.resolve_timestep(torch.tensor([0.0, 0.2])))


@_CPU_ONLY
def test_sol_config_lowers_and_factory_initializes_backend(monkeypatch) -> None:
    base_kwargs = {}

    def _base_init(self, **kwargs) -> None:
        base_kwargs.update(kwargs)
        self.layer_idx = kwargs["layer_idx"]
        self.head_dim = kwargs["head_dim"]
        self.q_scaling = 1.0

    monkeypatch.setattr(TrtllmAttention, "__init__", _base_init)
    attention_config = AttentionConfig(
        backend="TRTLLM",
        sparse_attention_config={
            "algorithm": "sol_attn",
            "tau": -0.25,
            "disabled_until_timestep": 0.6,
            "dense_layers": [0, 2, 4, 3],
        },
    )
    config = attention_config.sparse_attention_config
    assert isinstance(config, SolAttentionConfig)
    params = config.to_sparse_params()

    backend = create_attention(
        backend="TRTLLM",
        layer_idx=3,
        num_heads=4,
        head_dim=128,
        attention_config=attention_config,
        sparse_params=params,
        attention_metadata_state=create_attention_metadata_state(),
    )

    assert config.algorithm == "sol_attn"
    assert params.tau == -0.25
    assert params.disabled_until_timestep == 0.6
    assert params.dense_layers == frozenset({0, 2, 3, 4})
    assert isinstance(backend, SOLTrtllmAttention)
    assert backend.sol_params is params
    assert not hasattr(backend, "predictor")
    assert base_kwargs["sparse_params"] is None
    assert "_enable_sparse_workflow" not in SOLTrtllmAttention.__dict__
    assert "_should_use_sparse_workflow" not in SOLTrtllmAttention.__dict__
    assert not backend.support_fused_qkv()
    assert "forward" not in SOLTrtllmAttention.__dict__
    assert "block_sparse_attn_predict" in SOLTrtllmAttention.__dict__


@_CPU_ONLY
def test_sol_backend_sparse_phase_emits_proxy_bitmask_carrier(monkeypatch) -> None:
    batch_size, seq_len, num_heads = 1, 65, 2
    q, k, v = (_bshd(seq_len, num_heads) for _ in range(3))
    predictor_outputs = _predictor_outputs(
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
    )
    backend, predictor = _stub_backend(monkeypatch, SolParams(tau=0.75), seq_len=seq_len)
    predictor.predict.return_value = predictor_outputs
    monkeypatch.setattr(sol_backend, "get_bmm1_scale", lambda attn: 0.375)

    carrier = _predict(backend, q, k, v, timestep=0.2)

    predicted_q, predicted_k, predicted_v = predictor.predict.call_args.args
    assert predictor.predict.call_args.kwargs == {
        "tau": 0.75,
        "sm_scale": 0.375,
        "thresh_type": "diag",
    }
    predictor.support_reason.assert_called_once_with(predicted_q, predicted_k, predicted_v)
    for predicted, source in zip((predicted_q, predicted_k, predicted_v), (q, k, v), strict=True):
        assert predicted.shape == (batch_size, seq_len, num_heads, 128)
        assert predicted.is_contiguous()
        assert predicted.data_ptr() == source.data_ptr()
    assert (
        carrier.q_block_size,
        carrier.kv_block_size,
        carrier.max_blocks_per_row,
        carrier.block_indptr,
        carrier.block_indices,
        carrier.kv_valid_bits,
    ) == (64, 64, None, None, None, None)
    assert carrier.exact_block_bits is predictor_outputs.exact_block_bits
    assert carrier.k_summary is predictor_outputs.k_summary
    assert carrier.v_summary is predictor_outputs.v_summary
    assert carrier.sparse_format == "bitmask"
    assert carrier.use_proxy_routes


@_CPU_ONLY
def test_sol_wrapper_compacts_separate_qkv_and_predicts_inside_core(monkeypatch) -> None:
    batch_size, seq_len, num_heads = 1, 65, 2
    packed_qkv = torch.zeros(batch_size, seq_len, 3 * num_heads * 128, dtype=torch.bfloat16)
    q, k, v = (
        tensor.view(batch_size, seq_len, num_heads, 128)
        for tensor in packed_qkv.split(num_heads * 128, dim=-1)
    )
    predictor_outputs = _predictor_outputs(
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
    )
    backend, predictor = _stub_backend(monkeypatch, SolParams(tau=0.75), seq_len=seq_len)
    predictor.predict.return_value = predictor_outputs
    monkeypatch.setattr(sol_backend, "get_bmm1_scale", lambda attn: 0.375)
    captured = _stub_core_forward(monkeypatch)

    output = _forward(backend, q, k, v, attention_mask=PredefinedAttentionMask.FULL, timestep=0.2)

    assert output.shape == (batch_size, seq_len, num_heads * 128)
    assert all(
        tensor.is_contiguous() and tensor.shape == (batch_size * seq_len, num_heads * 128)
        for tensor in (captured["q"], captured["k"], captured["v"])
    )
    forward_args = captured["forward_args"]
    assert forward_args.timestep == 0.2
    assert forward_args.sparse_backend_args is None
    carrier = forward_args.sparse_runtime_params.block_sparse_inputs
    assert carrier.exact_block_bits is predictor_outputs.exact_block_bits
    predicted_q = predictor.predict.call_args.args[0]
    assert predicted_q.data_ptr() == captured["q"].data_ptr()


@_CPU_ONLY
@pytest.mark.parametrize(
    ("params", "layer_idx", "timestep"),
    (
        (SolParams(tau=1.0, dense_layers=frozenset({3})), 3, None),
        (SolParams(tau=1.0, disabled_until_timestep=0.6), 1, torch.tensor([0.9])),
    ),
    ids=("dense_layer", "dense_phase"),
)
def test_sol_wrapper_fuses_qkv_for_dense_calls(monkeypatch, params, layer_idx, timestep) -> None:
    """A dense layer or a dense-phase step predicts nothing, so the wrapper hands
    the core fused QKV: the only dense self-attention layout the TRTLLM kernel serves."""
    batch_size, seq_len, num_heads = 1, 64, 2
    q, k, v = (
        torch.zeros(batch_size, seq_len, num_heads, 128, dtype=torch.bfloat16) for _ in range(3)
    )
    backend, predictor = _stub_backend(monkeypatch, params, seq_len=seq_len)
    backend.layer_idx = layer_idx
    captured = _stub_core_forward(monkeypatch)

    _forward(backend, q, k, v, attention_mask=PredefinedAttentionMask.FULL, timestep=timestep)

    assert captured["k"] is None and captured["v"] is None
    assert captured["q"].shape == (batch_size * seq_len, 3 * num_heads * 128)
    runtime_params = captured["forward_args"].sparse_runtime_params
    assert getattr(runtime_params, "block_sparse_inputs", None) is None
    predictor.predict.assert_not_called()


@_CPU_ONLY
@pytest.mark.parametrize(
    ("k", "v", "attention_mask", "message"),
    (
        (None, None, PredefinedAttentionMask.FULL, "separate q, k, and v"),
        (_bshd(32), _bshd(32), PredefinedAttentionMask.FULL, "self-attention"),
        (_bshd(), _bshd(), PredefinedAttentionMask.CAUSAL, "full attention mask"),
    ),
)
def test_sol_backend_rejects_non_sol_sparse_calls(
    monkeypatch,
    k: torch.Tensor | None,
    v: torch.Tensor | None,
    attention_mask: PredefinedAttentionMask,
    message: str,
) -> None:
    q = _bshd()
    backend, predictor = _stub_backend(monkeypatch)

    with pytest.raises(ValueError, match=message):
        _predict(backend, q, k, v, attention_mask=attention_mask)

    predictor.predict.assert_not_called()


@_CPU_ONLY
def test_sol_wrapper_rejects_fused_qkv_before_core(monkeypatch) -> None:
    q = _bshd()
    backend, predictor = _stub_backend(monkeypatch)
    prepare_metadata = Mock(return_value=object())
    monkeypatch.setattr(TrtllmAttention, "_prepare_metadata", prepare_metadata)

    with pytest.raises(ValueError, match="separate q, k, and v"):
        _forward(backend, q, None, None)

    prepare_metadata.assert_not_called()
    predictor.predict.assert_not_called()


@_CPU_ONLY
def test_sol_backend_surfaces_predictor_support_reason_before_execution(monkeypatch) -> None:
    q = _bshd()
    reason = "SOL predictor requires compact BSHD q/k/v"
    backend, predictor = _stub_backend(monkeypatch, unsupported_reason=reason)

    with pytest.raises(ValueError, match=reason):
        _predict(backend, q, q, q)

    predictor.predict.assert_not_called()


@_CPU_ONLY
@pytest.mark.parametrize(
    ("params", "layer_idx", "timestep"),
    (
        (SolParams(dense_layers=frozenset({1})), 1, None),
        (SolParams(disabled_until_timestep=0.6), 1, 0.8),
    ),
)
def test_sol_dense_policy_returns_no_routes_without_predicting(
    monkeypatch,
    params: SolParams,
    layer_idx: int,
    timestep: float | None,
) -> None:
    q = _bshd()
    backend, predictor = _stub_backend(monkeypatch, params)
    backend.layer_idx = layer_idx

    assert _predict(backend, q, q, q, timestep=timestep) is None
    predictor.support_reason.assert_not_called()
    predictor.predict.assert_not_called()


@_CPU_ONLY
def test_sol_sparse_phase_without_primts_fails_closed(monkeypatch) -> None:
    q = _bshd()
    backend, predictor = _stub_backend(monkeypatch)
    backend._fmha_manager = SimpleNamespace(fmha_libs=[])

    with pytest.raises(RuntimeError, match="requires PrimTS block-sparse FMHA"):
        _predict(backend, q, q, q)

    predictor.support_reason.assert_not_called()
    predictor.predict.assert_not_called()


@_CPU_ONLY
def test_sol_sparse_phase_with_quantization_fails_closed(monkeypatch) -> None:
    q = _bshd()
    backend, predictor = _stub_backend(monkeypatch)
    backend.quant_attention_config = object()

    with pytest.raises(ValueError, match="does not support quant_attention_config"):
        _predict(backend, q, q, q)

    predictor.support_reason.assert_not_called()
    predictor.predict.assert_not_called()


# --------------------------------------------------------------------------- CuTeDSL helpers
def _kernel_wrapper():
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell import sol_attn_backend

    return sol_attn_backend


def _cutedsl_backend(
    *, layer_idx: int = 1, num_heads: int = 2, head_dim: int = 128
) -> SOLCuTeDSLAttention:
    """CuTeDSL SOL backend with the dense prefix disabled unless a test sets it."""

    backend = SOLCuTeDSLAttention(layer_idx=layer_idx, num_heads=num_heads, head_dim=head_dim)
    backend.disabled_until_timestep = None
    return backend


def _cutedsl_module_config(
    hidden_size: int, num_heads: int, head_dim: int, *, tau: float
) -> DiffusionModelConfig:
    """Minimal DiffusionModelConfig for one CuTeDSL SOL attention module."""

    return DiffusionModelConfig(
        pretrained_config=SimpleNamespace(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            attention_head_dim=head_dim,
            eps=1e-6,
        ),
        attention=AttentionConfig(
            backend="CUTEDSL", sparse_attention_config=SolAttentionConfig(tau=tau)
        ),
        skip_create_weights_in_init=False,
    )


def _fake_cuda_q(shape, dtype=torch.bfloat16) -> SimpleNamespace:
    """A tensor-like that reports ``is_cuda=True`` without needing a GPU.

    ``sol_attn_ineligible_reason`` checks ``is_cuda`` first and returns early, so
    a CPU tensor can never reach the rank, head_dim, or dtype branches. These
    stubs pass that first gate so each later reason is actually exercised; the
    architecture check comes after them and is not reached.
    """

    return SimpleNamespace(is_cuda=True, ndim=len(shape), shape=shape, dtype=dtype)


def _is_dynamo_disabled(fn) -> bool:
    """True if ``fn`` is wrapped by torch.compiler.disable / torch._dynamo.disable."""

    target = getattr(fn, "__func__", fn)
    return bool(getattr(target, "_torchdynamo_disable", False))


def _masked_sdpa_reference(q, k, v, *, key_padding_mask=None, is_causal=False) -> torch.Tensor:
    attn_mask = (
        None if key_padding_mask is None else key_padding_mask.to(torch.bool)[:, None, None, :]
    )
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2).float(),
        k.transpose(1, 2).float(),
        v.transpose(1, 2).float(),
        attn_mask=attn_mask,
        is_causal=is_causal and attn_mask is None,
    ).transpose(1, 2)


# --------------------------------------------------------------------------- CuTeDSL backend
@_CPU_ONLY
def test_cutedsl_factory_dispatches_dense_and_sol() -> None:
    dense_attention = create_attention(
        backend="CUTEDSL",
        layer_idx=0,
        num_heads=8,
        head_dim=128,
        attention_config=AttentionConfig(backend="CUTEDSL"),
    )
    sol_attention = create_attention(
        backend="CUTEDSL",
        layer_idx=0,
        num_heads=8,
        head_dim=128,
        attention_config=AttentionConfig(
            backend="CUTEDSL",
            sparse_attention_config=SolAttentionConfig(tau=2.0, disabled_until_timestep=0.9545),
        ),
    )

    assert isinstance(dense_attention, CuTeDSLAttention)
    assert isinstance(sol_attention, SOLCuTeDSLAttention)
    assert sol_attention.tau == 2.0
    assert sol_attention.disabled_until_timestep == 0.9545


@_REQUIRES_CUDA
def test_cutedsl_sol_cross_attention_delegates_per_call() -> None:
    """Cross-attention is delegated to the dense backend, decided per call.

    SOL is self-attention only. It is still the module's backend for a
    cross-attention module and delegates at ``forward`` because ``_can_serve``
    sees ``k.shape[1] != q.shape[1]``. Deciding this per call rather than at
    construction matters: ``qkv_mode`` describes how Q/K/V are projected, not
    whether K/V come from another sequence, so a construction-time rule keyed
    on SEPARATE_QKV would strip the configured backend from self-attention
    modules that use that mode for unrelated reasons (Qwen-Image; WAN attn1
    under async Ulysses).
    """

    device = torch.device("cuda")
    cross_attn = (
        Attention(
            64,
            4,
            qkv_mode=QKVMode.SEPARATE_QKV,
            config=_cutedsl_module_config(64, 4, 16, tau=1.0),
        )
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )
    assert cross_attn.attn_backend == "CUTEDSL"
    assert isinstance(cross_attn.attn, SOLCuTeDSLAttention)

    q = torch.randn(1, 32, 4, 16, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, 77, 4, 16, device=device, dtype=torch.bfloat16)
    assert not cross_attn.attn._can_serve(q, k), (
        "differing q/k sequence lengths must be delegated, not routed to the sparse kernel"
    )


@_CPU_ONLY
def test_cutedsl_sol_serves_separate_qkv_self_attention() -> None:
    """SEPARATE_QKV self-attention keeps SOL: WAN attn1 under async Ulysses and Qwen-Image."""

    backend = _cutedsl_backend()
    # CPU tensors on purpose: `_can_serve` compares shapes and the layer index
    # and never touches the device, so this runs on CPU-only hosts too.
    q = k = torch.randn(1, 64, 2, 128, dtype=torch.bfloat16)
    assert backend._can_serve(q, k), "equal q/k sequence lengths must reach the sparse kernel"


@_CPU_ONLY
def test_cutedsl_sol_rejects_gqa_mqa() -> None:
    with pytest.raises(ValueError, match="MHA-only"):
        SOLCuTeDSLAttention(layer_idx=0, num_heads=8, head_dim=128, num_kv_heads=2)


@_CPU_ONLY
def test_cutedsl_sol_dense_prefix_skips_kernel(monkeypatch) -> None:
    """Inside the dense prefix the sparse kernel must not be invoked at all.

    CPU tensors, so ``_delegate`` takes its SDPA branch here; that the dense
    path routes to the CuTe kernel on CUDA is covered by
    ``test_cutedsl_sol_dense_paths_use_cutedsl_dense_kernel``.
    """

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("kernel must not run inside the dense prefix")

    monkeypatch.setattr(sol_backend, "_sol_attn_run", _fail_if_called)
    backend = SOLCuTeDSLAttention(layer_idx=0, num_heads=2, head_dim=16)
    backend.disabled_until_timestep = 0.9
    q = k = v = torch.randn(1, 4, 2, 16)

    out = backend.forward(q, k, v, timestep=torch.tensor(0.95))

    assert out.shape == q.shape
    assert torch.isfinite(out).all()


@_CPU_ONLY
def test_cutedsl_sol_missing_timestep_fails_open_to_sparse(monkeypatch) -> None:
    """Without a timestep the prefix cannot be applied; run sparse, do not raise."""

    called = {"n": 0}

    def _record(*args, **kwargs):
        called["n"] += 1
        return args[0]

    monkeypatch.setattr(sol_backend, "_sol_attn_run", _record)
    backend = SOLCuTeDSLAttention(layer_idx=0, num_heads=2, head_dim=16)
    backend.disabled_until_timestep = 0.9
    q = k = v = torch.randn(1, 4, 2, 16)

    backend.forward(q, k, v)

    assert called["n"] == 1, "expected the sparse kernel, not a silent dense fallback"


@_CPU_ONLY
def test_cutedsl_sol_dense_layers_guard_skips_kernel(monkeypatch) -> None:
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("kernel must not be invoked for a dense_layers-forced layer")

    monkeypatch.setattr(sol_backend, "_sol_attn_run", _fail_if_called)
    backend = SOLCuTeDSLAttention(layer_idx=3, num_heads=2, head_dim=16)
    backend.dense_layers = frozenset({3})
    q = k = v = torch.randn(1, 4, 2, 16)

    out = backend.forward(q, k, v)

    assert out.shape == q.shape
    assert torch.isfinite(out).all()


@_CPU_ONLY
def test_cutedsl_sol_dense_by_step_prefers_runner_resolved_phase(monkeypatch) -> None:
    """Inside a runner-driven call the phase comes from the runner, not the tensor.

    CUDA Graph capture forbids device-to-host syncs, and reading the timestep is
    a ``.item()``. The runner resolves the phase host-side to build the graph
    key and republishes it during capture; ``_dense_by_step`` must use that and
    never touch the tensor.
    """

    import tensorrt_llm._torch.attention.backends.sparse.timestep_phase as timestep_phase

    reads = {"n": 0}
    real = timestep_phase.timestep_to_float

    def spy(value):
        reads["n"] += 1
        return real(value)

    monkeypatch.setattr(timestep_phase, "timestep_to_float", spy)
    backend = _cutedsl_backend()
    backend.disabled_until_timestep = 0.9

    # The resolved phase wins even when the tensor says otherwise: phase 0 is
    # the dense prefix, phase 1 the sparse phase.
    with resolved_extra_keys_scope({"sparse_attn_phase": 0}):
        assert backend._dense_by_step(torch.tensor(0.1)) is True
    with resolved_extra_keys_scope({"sparse_attn_phase": 1}):
        assert backend._dense_by_step(torch.tensor(0.99)) is False
    assert reads["n"] == 0, "timestep tensor was read despite a runner-resolved phase"

    # Outside a runner-driven call the tensor is the only source of truth.
    assert backend._dense_by_step(torch.tensor(0.99)) is True
    assert reads["n"] == 1


@_REQUIRES_CUDA
def test_cutedsl_sol_dense_paths_use_cutedsl_dense_kernel(monkeypatch) -> None:
    """Every path SOL cannot serve must reach the configured dense kernel.

    SOL runs dense attention on the ``dense_layers`` guard, the
    ``disabled_until_timestep`` prefix and the kernel-ineligibility fallback. If
    those ran torch SDPA instead of the CuTe DSL FMHA, a ``backend: CUTEDSL``
    run would differ from a ``backend: CUTEDSL`` dense baseline on those steps
    and an A/B against that baseline would measure a backend swap rather than
    sparsity.
    """

    if not sol_backend._cute_dense_available():
        pytest.skip("no CuTe DSL dense kernel for this device")

    device = torch.device("cuda")
    q = k = v = torch.randn(1, 64, 2, 128, device=device, dtype=torch.bfloat16)

    def _make() -> tuple[SOLCuTeDSLAttention, dict]:
        backend = _cutedsl_backend()
        calls = {"n": 0}
        real = backend._inner.forward

        def _spy(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(backend._inner, "forward", _spy)
        return backend, calls

    backend, calls = _make()
    backend.disabled_until_timestep = 0.9
    backend.forward(q, k, v, timestep=torch.tensor(0.95))
    assert calls["n"] == 1, "dense prefix was not delegated to the CuTeDSL dense kernel"

    backend, calls = _make()
    backend.dense_layers = frozenset({backend.layer_idx})
    backend.forward(q, k, v)
    assert calls["n"] == 1, "dense_layers guard was not delegated to the CuTeDSL dense kernel"

    backend, calls = _make()
    monkeypatch.setattr(
        sol_backend,
        "_sol_attn_run",
        lambda *args, **kw: kw["dense_fn"](*args[:3]),
    )
    backend.forward(q, k, v)
    assert calls["n"] == 1, "dense_fn did not route the fallback to the CuTeDSL dense kernel"


@_CPU_ONLY
@pytest.mark.parametrize(
    ("make", "expect"),
    [
        (lambda: torch.randn(1, 4, 2, 128), "not a CUDA tensor"),
        (lambda: _fake_cuda_q((1, 4, 128)), "must be 4-D"),
        (lambda: _fake_cuda_q((1, 4, 2, 64)), "head_dim must be 128"),
        (lambda: _fake_cuda_q((1, 4, 2, 128), dtype=torch.float16), "dtype must be bfloat16"),
    ],
    ids=["cpu-tensor", "wrong-rank", "wrong-head-dim", "wrong-dtype"],
)
def test_cutedsl_ineligible_reason_is_reported(make, expect) -> None:
    """Ineligibility must name the specific reason, never fail silently."""

    wrapper = _kernel_wrapper()
    reason = wrapper.sol_attn_ineligible_reason(make())
    assert reason is not None and expect in reason
    assert not wrapper.sol_attn_supported(make())


@_CPU_ONLY
def test_cutedsl_strict_raises_on_ineligible_input(monkeypatch) -> None:
    """TRTLLM_SOL_ATTN_STRICT=1 turns an unservable input into an error."""

    wrapper = _kernel_wrapper()
    monkeypatch.setenv("TRTLLM_SOL_ATTN_STRICT", "1")
    q = k = v = torch.randn(1, 4, 2, 128)  # CPU -> ineligible
    with pytest.raises(RuntimeError, match="cannot run the CuTe kernel"):
        wrapper._run_sol_attn_bthd(q, k, v)


@_CPU_ONLY
def test_cutedsl_ineligible_falls_back_to_dense_and_counts(monkeypatch) -> None:
    """Without STRICT the same input degrades to dense and increments the counter."""

    wrapper = _kernel_wrapper()
    monkeypatch.delenv("TRTLLM_SOL_ATTN_STRICT", raising=False)
    wrapper.reset_sol_attn_stats()
    q = k = v = torch.randn(1, 4, 2, 128)

    out = wrapper._run_sol_attn_bthd(q, k, v)

    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)
    assert torch.allclose(out, ref), "dense fallback must be plain SDPA"
    assert wrapper.get_sol_attn_stats()["dense_fallback_calls"] == 1
    assert wrapper.get_sol_attn_stats()["kernel_calls"] == 0


@_CPU_ONLY
def test_cutedsl_supported_archs_match_kernel_dispatch_map() -> None:
    """SUPPORTED_ARCHS mirrors the kernel dispatch map; drift would reject a served arch."""

    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.sol_attn import interface

    assert _kernel_wrapper().SUPPORTED_ARCHS == frozenset(
        major * 10 + minor for major, minor in interface._CUTE_BACKENDS
    )


@_CPU_ONLY
def test_cutedsl_datacenter_blackwell_archs_are_supported() -> None:
    """Both datacenter Blackwell steppings dispatch to the vendored kernel."""

    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.sol_attn import interface

    # Resolve, don't just look up: a malformed value would satisfy a key check
    # and then fail at dispatch.
    for arch in ((10, 0), (10, 3)):
        kernel = interface._backend_for_arch(arch, cute_available=True)
        assert kernel in interface._CUTE_BACKENDS.values()
        assert isinstance(kernel, str) and kernel


@_CPU_ONLY
def test_cutedsl_no_arch_literal_outside_the_dispatch_map() -> None:
    """The guard in ``_sol_attn_cute`` must key off ``_CUTE_BACKENDS``, not a literal.

    ``test_cutedsl_supported_archs_match_kernel_dispatch_map`` keeps
    SUPPORTED_ARCHS and _CUTE_BACKENDS in step, so widening both would leave a
    hardcoded literal here as the only thing still rejecting a new arch, with
    every test green. Assert on the source so the coupling cannot regress.
    """

    import ast
    import inspect
    import textwrap

    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.sol_attn import interface

    src = textwrap.dedent(inspect.getsource(interface._sol_attn_cute))
    assert "arch not in _CUTE_BACKENDS" in src, (
        "the guard in _sol_attn_cute must be keyed off _CUTE_BACKENDS"
    )

    offenders = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        names = {n.id for n in operands if isinstance(n, ast.Name)}
        if "arch" not in names:
            continue
        for operand in operands:
            if isinstance(operand, ast.Tuple) and all(
                isinstance(e, ast.Constant) and isinstance(e.value, int) for e in operand.elts
            ):
                offenders.append(ast.unparse(node))
    assert not offenders, (
        "hardcoded architecture literal(s) compared against `arch` in "
        f"_sol_attn_cute: {offenders}; key the guard off _CUTE_BACKENDS instead"
    )


@_CPU_ONLY
def test_cutedsl_kernel_launch_is_opaque_to_dynamo() -> None:
    """The CuTe DSL launch boundary must be ``torch.compiler.disable``d.

    Without it Dynamo traces into the CuTe DSL JIT builder and retraces on every
    call, which is silently two orders of magnitude slower.
    """

    assert _is_dynamo_disabled(_kernel_wrapper()._run_sol_attn_bthd), (
        "_run_sol_attn_bthd must be decorated with @torch.compiler.disable"
    )


@_CPU_ONLY
def test_cutedsl_timestep_scalar_read_is_opaque_to_dynamo() -> None:
    """The dense-prefix ``.item()`` must stay in eager, or it graph-breaks per layer."""

    assert _is_dynamo_disabled(SOLCuTeDSLAttention._dense_by_step), (
        "SOLCuTeDSLAttention._dense_by_step must be decorated with @torch.compiler.disable"
    )


@_CPU_ONLY
def test_cutedsl_sol_key_padding_mask_routes_to_vanilla_and_is_honored(monkeypatch) -> None:
    """Masked self-attention must never reach the mask-blind sparse kernel.

    Equal Q/K lengths do not mean unmasked: HunyuanVideo 1.5 and GLM-Image pass
    a ``[B, S]`` ``key_padding_mask`` on self-attention. The sparse kernel takes
    no mask and CuTeDSL's dense ``_fwd(**kwargs)`` swallows it, so the only
    correct destination is VANILLA. The output is checked against a masked SDPA
    reference so a mask that was routed but then dropped still fails.
    """

    monkeypatch.setattr(
        sol_backend,
        "_sol_attn_run",
        lambda *a, **k: pytest.fail("sparse kernel ran on a masked call"),
    )
    backend = _cutedsl_backend()
    calls = {"vanilla": 0}
    real = backend._vanilla.forward

    def spy(*a, **k):
        calls["vanilla"] += 1
        return real(*a, **k)

    monkeypatch.setattr(backend._vanilla, "forward", spy)

    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 64, 2, 128, dtype=torch.float32) for _ in range(3))
    mask = torch.ones(2, 64, dtype=torch.bool)
    mask[0, 48:] = False  # pad the tail of sample 0
    mask[1, :16] = False  # pad the head of sample 1

    assert not backend._can_serve(q, k, key_padding_mask=mask)
    out = backend.forward(q, k, v, key_padding_mask=mask)
    assert calls["vanilla"] == 1
    ref = _masked_sdpa_reference(q, k, v, key_padding_mask=mask)
    torch.testing.assert_close(out.float(), ref, rtol=1e-4, atol=1e-4)
    # And the mask actually mattered: unmasked attention gives a different answer.
    assert (ref - _masked_sdpa_reference(q, k, v)).abs().max() > 1e-3


@_CPU_ONLY
def test_cutedsl_sol_causal_mask_routes_to_dense_and_is_honored(monkeypatch) -> None:
    """CAUSAL disqualifies the noncausal sparse kernel and is honored downstream."""

    monkeypatch.setattr(
        sol_backend,
        "_sol_attn_run",
        lambda *a, **k: pytest.fail("sparse kernel ran on a causal call"),
    )
    backend = _cutedsl_backend()
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 64, 2, 128, dtype=torch.float32) for _ in range(3))

    assert not backend._can_serve(q, k, attention_mask=PredefinedAttentionMask.CAUSAL)
    out = backend.forward(q, k, v, attention_mask=PredefinedAttentionMask.CAUSAL)
    ref = _masked_sdpa_reference(q, k, v, is_causal=True)
    torch.testing.assert_close(out.float(), ref, rtol=1e-4, atol=1e-4)
    assert (ref - _masked_sdpa_reference(q, k, v)).abs().max() > 1e-3


@_CPU_ONLY
def test_cutedsl_sol_unmasked_self_attention_is_served() -> None:
    """The mask routing must not touch the measured path: no mask, sparse."""

    backend = _cutedsl_backend()
    q = k = torch.randn(1, 64, 2, 128, dtype=torch.bfloat16)
    assert backend._can_serve(q, k)
    assert backend._can_serve(
        q, k, attention_mask=PredefinedAttentionMask.FULL, key_padding_mask=None
    )


# --------------------------------------------------------------------------- public config
@_CPU_ONLY
@pytest.mark.parametrize(
    "config_kwargs",
    (
        {"tau": 1.0e100},
        {"disabled_until_timestep": 0.0},
        {"dense_layers": [-1]},
    ),
)
def test_sol_public_config_rejects_invalid_policy(config_kwargs) -> None:
    with pytest.raises((ValidationError, ValueError)):
        SolAttentionConfig(**config_kwargs).to_sparse_params()


@_CPU_ONLY
def test_sol_public_config_defaults_to_no_dense_prefix() -> None:
    """``None``, not ``0.0``, means no dense prefix; ``0.0`` would run dense on every step."""

    assert SolAttentionConfig(tau=2.0).disabled_until_timestep is None


@_CPU_ONLY
def test_sol_public_config_requires_supported_backend() -> None:
    with pytest.raises(ValidationError, match="requires backend"):
        AttentionConfig(
            backend="VANILLA",
            sparse_attention_config=SolAttentionConfig(),
        )


@_CPU_ONLY
def test_sol_exact_threshold_lowers_for_both_backends() -> None:
    for backend in ("TRTLLM", "CUTEDSL"):
        config = AttentionConfig(
            backend=backend,
            sparse_attention_config=SolAttentionConfig(thresh_type="exact"),
        )
        assert config.sparse_attention_config.to_sparse_params().thresh_type == "exact"


@_CPU_ONLY
@pytest.mark.parametrize(
    ("backend", "quant_recipe"),
    [
        ("TRTLLM", {"qk_dtype": "fp8", "q_block_size": 1, "k_block_size": 1, "v_block_size": 1}),
        ("CUTEDSL", {"qk_dtype": "bf16", "v_dtype": "fp8"}),
    ],
    ids=["TRTLLM", "CUTEDSL"],
)
def test_sol_and_attention_quantization_are_mutually_exclusive(
    backend: str, quant_recipe: dict
) -> None:
    """A recipe the backend accepts on its own is still rejected next to SOL."""

    AttentionConfig(backend=backend, quant_attention_config=QuantAttentionConfig(**quant_recipe))
    with pytest.raises(ValidationError, match="SOL and quant_attention_config"):
        AttentionConfig(
            backend=backend,
            quant_attention_config=QuantAttentionConfig(**quant_recipe),
            sparse_attention_config=SolAttentionConfig(),
        )


@_CPU_ONLY
@pytest.mark.parametrize(
    "layers",
    [[-1], [0, -2], ["x"], "0,2", [1.5]],
    ids=["negative", "negative_in_list", "non_numeric", "string_spec", "float"],
)
def test_sol_dense_layers_rejects_invalid_values(layers) -> None:
    """``dense_layers`` is a list of non-negative layer indices; anything else fails at
    config time rather than at attention construction."""

    with pytest.raises(ValueError):
        SolAttentionConfig(tau=2.0, dense_layers=layers)


@_CPU_ONLY
@pytest.mark.parametrize(
    ("layers", "expected"),
    [(None, None), ([0], [0]), ([0, 2, 4], [0, 2, 4]), ([4, 2, 2, 0], [0, 2, 4])],
    ids=["none", "single", "list", "unsorted_with_duplicates"],
)
def test_sol_dense_layers_normalizes_valid_values(layers, expected) -> None:
    assert SolAttentionConfig(tau=2.0, dense_layers=layers).dense_layers == expected


@_CPU_ONLY
def test_sol_config_has_no_kv_splits_knob() -> None:
    """Only one KV split exists on the shipped kernels, so the config rejects the knob."""

    assert "kv_splits" not in SolAttentionConfig.model_fields
    with pytest.raises(ValueError):
        SolAttentionConfig(tau=2.0, kv_splits="1")


# --------------------------------------------------------------------------- attention module
def _sol_model_config(*, backend: str = "TRTLLM", cp_size: int = 1) -> DiffusionModelConfig:
    config = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(),
        attention=AttentionConfig(
            backend=backend,
            sparse_attention_config=SolAttentionConfig(
                tau=0.75,
                disabled_until_timestep=0.6,
                dense_layers=[0, 2, 3],
            ),
        ),
        skip_create_weights_in_init=True,
        attention_metadata_state=create_attention_metadata_state(),
    )
    if cp_size > 1:
        config.visual_gen_mapping = SimpleNamespace(
            ring_size=cp_size,
            ring_group=None,
            ulysses_size=1,
            ulysses_group=None,
            attn2d_row_size=1,
            attn2d_col_size=1,
            attn2d_row_group=None,
            attn2d_col_group=None,
            cp_size=cp_size,
        )
    return config


class _SolModel(BaseDiffusionModel):
    def __init__(self, backends: tuple[SOLTrtllmAttention, ...]) -> None:
        super().__init__(_sol_model_config())
        self.backends = backends

    def forward(self, q: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        for backend in self.backends:
            q = backend.forward(
                q=q,
                k=q,
                v=q,
                batch_size=q.shape[0],
                seq_len=q.shape[1],
                seq_len_kv=q.shape[1],
                timestep=timestep,
            )
        return q


@_CPU_ONLY
@pytest.mark.parametrize(
    ("is_self_attention", "expected_backend", "expects_sol_params"),
    ((True, "TRTLLM", True), (False, "VANILLA", False)),
    ids=("self", "cross"),
)
def test_sol_attention_module_dispatches_by_attention_role(
    monkeypatch,
    is_self_attention: bool,
    expected_backend: str,
    expects_sol_params: bool,
) -> None:
    captured = {}

    def _create_attention(*, backend, **kwargs):
        captured.update(backend=backend, **kwargs)
        return SimpleNamespace(preferred_layout=None)

    monkeypatch.setattr(attention_module, "create_attention", _create_attention)

    attention = Attention(
        hidden_size=256,
        num_attention_heads=2,
        head_dim=128,
        qkv_mode=QKVMode.SEPARATE_QKV,
        qk_norm=False,
        config=_sol_model_config(),
        separate_qkv_is_self_attention=is_self_attention,
    )

    assert attention.attn_backend == expected_backend
    if expects_sol_params:
        assert isinstance(attention.sparse_params, SolParams)
        assert captured["sparse_params"] is attention.sparse_params
    else:
        assert attention.sparse_params is None
        assert captured["sparse_params"] is None


@_CPU_ONLY
@_BACKENDS
def test_sol_attention_rejects_context_parallelism(backend: str) -> None:
    """SOL needs the full sequence per rank, so Ring/Attention2D fail at construction."""

    with pytest.raises(ValueError, match="SOL.*incompatible with context parallelism"):
        Attention(
            hidden_size=256,
            num_attention_heads=2,
            head_dim=128,
            qk_norm=False,
            config=_sol_model_config(backend=backend, cp_size=2),
        )


# --------------------------------------------------------------------------- CUDA Graph phase keys
def _sol_model(*, disabled_until_timestep=None, dense_layers=None) -> BaseDiffusionModel:
    """Minimal BaseDiffusionModel carrying a SOL sparse config."""

    config = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(
            hidden_size=64, num_attention_heads=4, attention_head_dim=16, eps=1e-6
        ),
        attention=AttentionConfig(
            backend="CUTEDSL",
            sparse_attention_config=SolAttentionConfig(
                tau=2.0,
                disabled_until_timestep=disabled_until_timestep,
                dense_layers=dense_layers,
            ),
        ),
        skip_create_weights_in_init=False,
    )
    return BaseDiffusionModel(config)


def _graph_runner() -> CUDAGraphRunner:
    return CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))


@_CPU_ONLY
def test_sol_cuda_graph_phase_is_keyed_without_model_scope(monkeypatch) -> None:
    q = _bshd()
    backend, predictor = _stub_backend(monkeypatch, SolParams(tau=1.0, disabled_until_timestep=0.6))
    model = _SolModel((backend,))
    runner = _graph_runner()
    model.register_cuda_graph_extra_key_fns(runner)
    _stub_core_forward(monkeypatch)
    monkeypatch.setattr(sol_backend, "get_bmm1_scale", lambda attn: 0.125)
    capturing = False
    captured_outputs = {}
    captured_keys = []

    def _capture(key, fn, args, kwargs):
        nonlocal capturing
        captured_outputs[key] = fn(*args, **kwargs)
        capturing = True
        try:
            captured_outputs[key] = fn(*args, **kwargs)
            captured_keys.append(key)
        finally:
            capturing = False

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(runner, "capture", _capture)
    monkeypatch.setattr(runner, "replay", lambda key, args, kwargs: captured_outputs[key])
    model.forward = runner.wrap(model.forward)

    # The stubbed core echoes its q: fused QKV in the dense prefix, Q alone in
    # the sparse phase.
    assert model(q, timestep=torch.tensor(0.8)).shape == (1, 64, 3 * 256)
    assert model(q, timestep=torch.tensor(0.2)).shape == (1, 64, 256)
    assert ("sparse_attn_phase", 0) in captured_keys[0]
    assert ("sparse_attn_phase", 1) in captured_keys[1]
    assert captured_keys[0] != captured_keys[1]
    assert predictor.predict.call_count == 2


@_CPU_ONLY
def test_sol_cuda_graph_key_separates_dense_prefix_from_sparse_phase() -> None:
    """The prefix swaps kernels without changing any tensor shape, so a graph
    captured in the dense prefix must not be replayed for the sparse phase."""

    model = _sol_model(disabled_until_timestep=0.9)
    runner = _graph_runner()
    model.register_cuda_graph_extra_key_fns(runner)

    base = {"hidden_states": torch.empty(1, 8, 64)}
    key_dense = runner.get_graph_key(**base, timestep=torch.empty(1).fill_(0.95))
    key_sparse = runner.get_graph_key(**base, timestep=torch.empty(1).fill_(0.10))

    assert key_dense != key_sparse, (
        "dense-prefix and sparse phases share a CUDA graph key despite running "
        "different kernels; a graph captured in one phase would be replayed in "
        "the other"
    )


@_CPU_ONLY
def test_sol_cuda_graph_key_unregistered_without_prefix() -> None:
    """``dense_layers`` alone is fixed per layer, so it needs no graph key."""

    model = _sol_model(disabled_until_timestep=None, dense_layers=[0, 2])
    runner = _graph_runner()
    model.register_cuda_graph_extra_key_fns(runner)

    base = {"hidden_states": torch.empty(1, 8, 64)}
    key_a = runner.get_graph_key(**base, timestep=torch.empty(1).fill_(0.95))
    key_b = runner.get_graph_key(**base, timestep=torch.empty(1).fill_(0.10))
    assert key_a == key_b, "no phase key should be registered without a dense prefix"


# --------------------------------------------------------------------------- CUDA Graph capture
def _base_runner() -> CUDAGraphRunner:
    return CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))


def _ltx2_runner():
    from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import _LTX2CUDAGraphRunner

    return _LTX2CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))


def _ltx2_two_stage_runner():
    from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_two_stages import (
        _LTX2TwoStageCUDAGraphRunner,
    )

    return _LTX2TwoStageCUDAGraphRunner(
        CUDAGraphRunnerConfig(use_cuda_graph=True), lambda: "original", lambda: "default"
    )


# Every runner that overrides ``capture`` must keep the phase-scope contract;
# LTX-2's runners re-implement capture for ``Modality`` inputs.
_RUNNER_FACTORIES = pytest.mark.parametrize(
    "make_runner",
    [_base_runner, _ltx2_runner, _ltx2_two_stage_runner],
    ids=["base", "ltx2", "ltx2_two_stage"],
)


@_REQUIRES_CUDA
@_RUNNER_FACTORIES
def test_cuda_graph_runner_publishes_resolved_extra_keys_during_capture(make_runner) -> None:
    """The runner exposes its host-resolved extra keys to the captured callee."""

    seen = []

    def fn(x):
        seen.append(resolved_extra_key("probe"))
        return x * 2

    runner = make_runner()
    runner.register_extra_key_fn("probe", lambda *args, **kwargs: 7)
    wrapped = runner.wrap(fn)
    x = torch.ones(4, device="cuda")
    out = wrapped(x)
    torch.cuda.synchronize()
    assert torch.equal(out, x * 2)
    assert seen and all(v == 7 for v in seen), seen  # warmup + capture passes
    assert resolved_extra_key("probe") is None, "scope must close after capture"


@_REQUIRES_CUDA
@_RUNNER_FACTORIES
def test_cutedsl_sol_dense_prefix_survives_cuda_graph_capture(make_runner) -> None:
    """Capture and replay a graph on each side of the cutoff without a sync.

    The dense-prefix decision must be baked into each captured graph via the
    runner's phase key, and no ``.item()`` may run inside capture.
    ``kernel_calls`` proves the phase-0 graph never launched the sparse kernel
    and the phase-1 graph did.
    """

    wrapper = _kernel_wrapper()
    if not wrapper.sol_attn_supported(
        torch.empty(1, 8, 8, 128, device="cuda", dtype=torch.bfloat16)
    ):
        pytest.skip("no SOL kernel for this device")

    cutoff = 0.9
    backend = _cutedsl_backend()
    backend.disabled_until_timestep = cutoff
    runner = make_runner()
    runner.register_extra_key_fn(
        "sparse_attn_phase",
        lambda *args, **kwargs: graph_phase_for_timestep(
            kwargs.get("timestep"), disabled_until_timestep=cutoff
        ),
    )

    def fwd(q, k, v, *, timestep):
        return backend.forward(q, k, v, timestep=timestep)

    wrapped = runner.wrap(fwd)
    torch.manual_seed(0)
    q = k = v = torch.randn(1, 64, 2, 128, device="cuda", dtype=torch.bfloat16)
    t_dense = torch.tensor(0.95, device="cuda")
    t_sparse = torch.tensor(0.5, device="cuda")

    before = wrapper._SOL_STATS["kernel_calls"]
    wrapped(q, k, v, timestep=t_dense)
    torch.cuda.synchronize()
    after_dense = wrapper._SOL_STATS["kernel_calls"]
    wrapped(q, k, v, timestep=t_sparse)
    torch.cuda.synchronize()
    after_sparse = wrapper._SOL_STATS["kernel_calls"]

    assert len(runner.graphs) == 2, "one graph per phase"
    assert after_dense == before, "dense prefix must not launch the sparse kernel"
    assert after_sparse > after_dense, "sparse phase must launch the kernel"
    # Replays must not error either.
    wrapped(q, k, v, timestep=t_dense)
    wrapped(q, k, v, timestep=t_sparse)
    torch.cuda.synchronize()

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

"""Unit tests for the VisualGen SOL TRTLLM backend."""

from __future__ import annotations

import math
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
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention as CoreTrtllmAttention
from tensorrt_llm._torch.visual_gen.attention_backend.sparse.sol import backend as sol_backend
from tensorrt_llm._torch.visual_gen.attention_backend.sparse.sol.backend import SOLTrtllmAttention
from tensorrt_llm._torch.visual_gen.attention_backend.sparse.sol.params import SolParams
from tensorrt_llm._torch.visual_gen.attention_backend.sparse.sol.predictor import (
    SolPredictorOutputs,
    SOLSparsePredictor,
)
from tensorrt_llm._torch.visual_gen.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._torch.visual_gen.attention_backend.utils import create_attention
from tensorrt_llm._torch.visual_gen.config import (
    DiffusionModelConfig,
    create_attention_metadata_state,
)
from tensorrt_llm._torch.visual_gen.cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig
from tensorrt_llm._torch.visual_gen.models.modeling import BaseDiffusionModel
from tensorrt_llm._torch.visual_gen.modules import attention as attention_module
from tensorrt_llm._torch.visual_gen.modules.attention import Attention, QKVMode
from tensorrt_llm.visual_gen import SolAttentionConfig
from tensorrt_llm.visual_gen.args import AttentionConfig, QuantAttentionConfig

_REQUIRES_SM100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="SOL requires SM100 or SM103",
)
_CPU_ONLY = pytest.mark.cpu_only


def _make_backend(
    params: SolParams,
    predictor: Mock,
    *,
    layer_idx: int = 1,
) -> SOLTrtllmAttention:
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
    backend._prepared_graph_phase = None
    backend.predictor = predictor
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
):
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
    params: SolParams | None = None,
    *,
    seq_len: int = 64,
    unsupported_reason: str | None = None,
) -> tuple[SOLTrtllmAttention, Mock]:
    predictor = Mock(spec=SOLSparsePredictor)
    predictor.support_reason.return_value = unsupported_reason
    predictor.predict.return_value = _predictor_outputs(batch_size=1, seq_len=seq_len, num_heads=2)
    return _make_backend(params or SolParams(tau=1.0), predictor), predictor


def _dense_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * (128**-0.5)
    return torch.einsum("bhqk,bkhd->bqhd", scores.softmax(dim=-1), v.float()).to(q.dtype)


def _mixed_proxy_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    outputs: SolPredictorOutputs,
) -> torch.Tensor:
    """Evaluate the exact-token/proxy-summary attention contract."""

    block_size = 64
    num_blocks = math.ceil(k.shape[1] / block_size)
    exact_words = outputs.exact_block_bits.detach().cpu().to(torch.int64)
    reference = torch.empty_like(q)
    scale = q.shape[-1] ** -0.5
    for batch_idx in range(q.shape[0]):
        for head_idx in range(q.shape[2]):
            for q_block_idx in range(math.ceil(q.shape[1] / block_size)):
                q_begin = q_block_idx * block_size
                q_end = min(q_begin + block_size, q.shape[1])
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
                            block_idx * block_size,
                            min((block_idx + 1) * block_size, k.shape[1]),
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
                    tokens_in_block = min(block_size, k.shape[1] - block_idx * block_size)
                    denominator += proxy_weights[:, proxy_offset : proxy_offset + 1] * (
                        tokens_in_block
                    )
                reference[batch_idx, q_begin:q_end, head_idx] = (numerator / denominator).to(
                    q.dtype
                )
    return reference


@_CPU_ONLY
def test_sol_params_requires_precomputed_phase_during_cuda_graph_capture(monkeypatch) -> None:
    params = SolParams(tau=1.0, disabled_until_timestep=0.6)
    timestep = torch.tensor(0.2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="precomputed"):
        params.should_use_sparse(layer_idx=1, timestep=timestep)

    assert params.should_use_sparse(layer_idx=1, timestep=timestep, graph_phase=1)
    assert not params.should_use_sparse(layer_idx=1, timestep=timestep, graph_phase=0)


@_CPU_ONLY
def test_sol_backend_warmup_prepares_phase_for_capture(monkeypatch) -> None:
    params = SolParams(tau=1.0, disabled_until_timestep=0.6)
    q = _bshd()
    backend, predictor = _stub_backend(params)

    assert _predict(backend, q, q, q, timestep=torch.tensor(0.8)) is None
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert _predict(backend, q, q, q, timestep=torch.tensor(0.8)) is None
    predictor.predict.assert_not_called()


@_CPU_ONLY
def test_sol_backend_rejects_cutoff_capture_without_warmup(monkeypatch) -> None:
    params = SolParams(tau=1.0, disabled_until_timestep=0.6)
    q = _bshd()
    backend, predictor = _stub_backend(params)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="prepared before CUDA Graph capture"):
        _predict(backend, q, q, q, timestep=torch.tensor(0.2))

    predictor.predict.assert_not_called()


@_CPU_ONLY
def test_sol_backend_cutoff_requires_timestep_during_eager_forward() -> None:
    params = SolParams(tau=1.0, disabled_until_timestep=0.6)
    q = _bshd()
    backend, predictor = _stub_backend(params)

    with pytest.raises(ValueError, match="timestep is required"):
        _predict(backend, q, q, q)

    predictor.predict.assert_not_called()


@_CPU_ONLY
def test_sol_phase_waits_until_all_token_timesteps_are_below_cutoff() -> None:
    params = SolParams(tau=1.0, disabled_until_timestep=0.6)

    assert not params.should_use_sparse(layer_idx=1, timestep=torch.tensor([0.0, 0.8]))
    assert params.should_use_sparse(layer_idx=1, timestep=torch.tensor([0.0, 0.2]))


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
            "dense_layers": "0,2-4,3",
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
    assert isinstance(backend.predictor, SOLSparsePredictor)
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
    backend, predictor = _stub_backend(SolParams(tau=0.75), seq_len=seq_len)
    predictor.predict.return_value = predictor_outputs
    monkeypatch.setattr(sol_backend, "get_bmm1_scale", lambda attn: 0.375)

    carrier = _predict(backend, q, k, v, timestep=0.2)

    predicted_q, predicted_k, predicted_v = predictor.predict.call_args.args
    assert predictor.predict.call_args.kwargs == {"tau": 0.75, "sm_scale": 0.375}
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
    backend, predictor = _stub_backend(SolParams(tau=0.75), seq_len=seq_len)
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
    ("k", "v", "attention_mask", "message"),
    (
        (None, None, PredefinedAttentionMask.FULL, "separate q, k, and v"),
        (_bshd(32), _bshd(32), PredefinedAttentionMask.FULL, "self-attention"),
        (_bshd(), _bshd(), PredefinedAttentionMask.CAUSAL, "full attention mask"),
    ),
)
def test_sol_backend_rejects_non_sol_sparse_calls(
    k: torch.Tensor | None,
    v: torch.Tensor | None,
    attention_mask: PredefinedAttentionMask,
    message: str,
) -> None:
    q = _bshd()
    backend, predictor = _stub_backend()

    with pytest.raises(ValueError, match=message):
        _predict(backend, q, k, v, attention_mask=attention_mask)

    predictor.predict.assert_not_called()


@_CPU_ONLY
def test_sol_wrapper_rejects_fused_qkv_before_core(monkeypatch) -> None:
    q = _bshd()
    backend, predictor = _stub_backend()
    prepare_metadata = Mock(return_value=object())
    monkeypatch.setattr(TrtllmAttention, "_prepare_metadata", prepare_metadata)

    with pytest.raises(ValueError, match="separate q, k, and v"):
        _forward(backend, q, None, None)

    prepare_metadata.assert_not_called()
    predictor.predict.assert_not_called()


@_CPU_ONLY
def test_sol_backend_surfaces_predictor_support_reason_before_execution() -> None:
    q = _bshd()
    reason = "SOL predictor requires compact BSHD q/k/v"
    backend, predictor = _stub_backend(unsupported_reason=reason)

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
    params: SolParams,
    layer_idx: int,
    timestep: float | None,
) -> None:
    q = _bshd()
    backend, predictor = _stub_backend(params)
    backend.layer_idx = layer_idx

    assert _predict(backend, q, q, q, timestep=timestep) is None
    predictor.support_reason.assert_not_called()
    predictor.predict.assert_not_called()


@_CPU_ONLY
def test_sol_sparse_phase_without_primts_fails_closed() -> None:
    q = _bshd()
    backend, predictor = _stub_backend()
    backend._fmha_manager = SimpleNamespace(fmha_libs=[])

    with pytest.raises(RuntimeError, match="requires PrimTS block-sparse FMHA"):
        _predict(backend, q, q, q)

    predictor.support_reason.assert_not_called()
    predictor.predict.assert_not_called()


@_CPU_ONLY
def test_sol_sparse_phase_with_quantization_fails_closed() -> None:
    q = _bshd()
    backend, predictor = _stub_backend()
    backend.quant_attention_config = object()

    with pytest.raises(ValueError, match="does not support quant_attention_config"):
        _predict(backend, q, q, q)

    predictor.support_reason.assert_not_called()
    predictor.predict.assert_not_called()


@_CPU_ONLY
@pytest.mark.parametrize(
    "config_kwargs",
    (
        {"tau": 1.0e100},
        {"disabled_until_timestep": 0.0},
        {"dense_layers": "2-1"},
    ),
)
def test_sol_public_config_rejects_invalid_policy(config_kwargs) -> None:
    with pytest.raises((ValidationError, ValueError)):
        SolAttentionConfig(**config_kwargs).to_sparse_params()


@_CPU_ONLY
def test_sol_public_config_requires_trtllm_backend() -> None:
    with pytest.raises(ValidationError, match="requires backend"):
        AttentionConfig(
            backend="VANILLA",
            sparse_attention_config=SolAttentionConfig(),
        )


@_CPU_ONLY
def test_sol_and_attention_quantization_are_mutually_exclusive() -> None:
    with pytest.raises(ValidationError, match="SOL and quant_attention_config"):
        AttentionConfig(
            backend="TRTLLM",
            quant_attention_config=QuantAttentionConfig(
                qk_dtype="fp8",
                q_block_size=1,
                k_block_size=1,
                v_block_size=1,
            ),
            sparse_attention_config=SolAttentionConfig(),
        )


def _sol_model_config(*, cp_size: int = 1) -> DiffusionModelConfig:
    config = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(),
        attention=AttentionConfig(
            backend="TRTLLM",
            sparse_attention_config=SolAttentionConfig(
                tau=0.75,
                disabled_until_timestep=0.6,
                dense_layers="0,2-3",
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
def test_sol_attention_rejects_context_parallelism() -> None:
    with pytest.raises(ValueError, match="SOL.*incompatible with context parallelism"):
        Attention(
            hidden_size=256,
            num_attention_heads=2,
            head_dim=128,
            qk_norm=False,
            config=_sol_model_config(cp_size=2),
        )


@_CPU_ONLY
def test_sol_cuda_graph_phase_is_keyed_without_model_scope(monkeypatch) -> None:
    q = _bshd()
    backend, predictor = _stub_backend(SolParams(tau=1.0, disabled_until_timestep=0.6))
    model = _SolModel((backend,))
    runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
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

    assert model(q, timestep=torch.tensor(0.8)).shape == (1, 64, 256)
    assert model(q, timestep=torch.tensor(0.2)).shape == (1, 64, 256)
    assert ("sol_attn_phase", 0) in captured_keys[0]
    assert ("sol_attn_phase", 1) in captured_keys[1]
    assert captured_keys[0] != captured_keys[1]
    assert predictor.predict.call_count == 2


@_REQUIRES_SM100
@torch.no_grad()
def test_real_b200_sol_backend_cuda_graph_matches_dense_reference() -> None:
    attention_config = AttentionConfig(
        backend="TRTLLM",
        sparse_attention_config=SolAttentionConfig(
            tau=-1.0e6,
            disabled_until_timestep=0.6,
        ),
    )
    sparse_config = attention_config.sparse_attention_config
    assert isinstance(sparse_config, SolAttentionConfig)
    backend = create_attention(
        backend="TRTLLM",
        layer_idx=1,
        num_heads=2,
        head_dim=128,
        dtype=torch.bfloat16,
        attention_config=attention_config,
        attention_metadata_state=create_attention_metadata_state(),
        sparse_params=sparse_config.to_sparse_params(),
    )
    assert isinstance(backend, SOLTrtllmAttention)
    assert any(isinstance(fmha, PrimsTSBlockSparseFmha) for fmha in backend._fmha_manager.fmha_libs)

    generator = torch.Generator(device="cuda").manual_seed(20260901)
    shape = (1, 257, 2, 128)

    def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        packed = torch.randint(
            -2,
            3,
            (shape[0], shape[1], 3 * shape[2] * shape[3]),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        return tuple(tensor.view(shape) for tensor in packed.split(shape[2] * shape[3], dim=-1))

    q, k, v = _inputs()
    timestep = torch.tensor(0.2, device="cuda")
    assert not any(tensor.is_contiguous() for tensor in (q, k, v))
    eager = backend.forward(
        q=q,
        k=k,
        v=v,
        batch_size=1,
        seq_len=257,
        seq_len_kv=257,
        timestep=timestep,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        eager.view_as(q),
        _dense_reference(q, k, v),
        rtol=2e-2,
        atol=2e-2,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = backend.forward(
            q=q,
            k=k,
            v=v,
            batch_size=1,
            seq_len=257,
            seq_len_kv=257,
            timestep=timestep,
        ).view_as(q)

    next_q, next_k, next_v = _inputs()
    q.copy_(next_q)
    k.copy_(next_k)
    v.copy_(next_v)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        captured,
        _dense_reference(q, k, v),
        rtol=2e-2,
        atol=2e-2,
    )
    assert backend.predictor.num_plans == 1


@_REQUIRES_SM100
@torch.no_grad()
@pytest.mark.parametrize("seq_len", [256, 257])
def test_real_b200_sol_backend_mixed_proxy_cuda_graph_matches_reference(seq_len: int) -> None:
    attention_config = AttentionConfig(
        backend="TRTLLM",
        sparse_attention_config=SolAttentionConfig(tau=1.0e6),
    )
    sparse_config = attention_config.sparse_attention_config
    assert isinstance(sparse_config, SolAttentionConfig)
    backend = create_attention(
        backend="TRTLLM",
        layer_idx=1,
        num_heads=2,
        head_dim=128,
        dtype=torch.bfloat16,
        attention_config=attention_config,
        attention_metadata_state=create_attention_metadata_state(),
        sparse_params=sparse_config.to_sparse_params(),
    )
    assert isinstance(backend, SOLTrtllmAttention)

    generator = torch.Generator(device="cuda").manual_seed(20260903)
    shape = (1, seq_len, 2, 128)

    def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        packed = torch.randint(
            -2,
            3,
            (shape[0], shape[1], 3 * shape[2] * shape[3]),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        return tuple(tensor.view(shape) for tensor in packed.split(shape[2] * shape[3], dim=-1))

    q, k, v = _inputs()
    eager = backend.forward(q=q, k=k, v=v, batch_size=1, seq_len=seq_len, seq_len_kv=seq_len)
    predictor_outputs = backend.predictor.predict(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        tau=1.0e6,
        sm_scale=128**-0.5,
    )
    torch.cuda.synchronize()
    exact_bits = predictor_outputs.exact_block_bits
    num_blocks = math.ceil(seq_len / 64)
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
        eager.view_as(q),
        _mixed_proxy_reference(q, k, v, predictor_outputs),
        rtol=2e-2,
        atol=2e-2,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = backend.forward(
            q=q,
            k=k,
            v=v,
            batch_size=1,
            seq_len=seq_len,
            seq_len_kv=seq_len,
        ).view_as(q)

    next_q, next_k, next_v = _inputs()
    q.copy_(next_q)
    k.copy_(next_k)
    v.copy_(next_v)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        captured,
        _mixed_proxy_reference(q, k, v, predictor_outputs),
        rtol=2e-2,
        atol=2e-2,
    )

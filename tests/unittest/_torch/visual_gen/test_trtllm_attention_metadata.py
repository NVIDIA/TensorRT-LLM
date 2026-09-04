# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
from dataclasses import FrozenInstanceError, dataclass, replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.attention_backend.block_sparse import BlockSparseForwardInputs
from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask
from tensorrt_llm._torch.attention_backend.sparse.params import SparseRuntimeParams
from tensorrt_llm._torch.visual_gen.attention_backend import trtllm as visual_trtllm
from tensorrt_llm._torch.visual_gen.config import create_attention_metadata_state


class _FakeBaseTrtllmAttentionMetadata:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.prepare_calls = 0
        self.seq_lens = None
        self.num_contexts = None
        self.max_seq_len = None
        self.request_ids = None

    def prepare(self):
        self.prepare_calls += 1


class _RecordingSparseAttention(visual_trtllm.TrtllmAttention):
    def block_sparse_attn_predict(
        self,
        q,
        k,
        v,
        *,
        batch_size,
        seq_len,
        seq_len_kv,
        attention_mask,
        forward_kwargs,
    ):
        self.lifecycle_events.append("predict")
        self.predict_inputs = {
            "q": q,
            "k": k,
            "v": v,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "seq_len_kv": seq_len_kv,
            "attention_mask": attention_mask,
            "forward_kwargs": forward_kwargs,
        }
        self.prediction = visual_trtllm.SparseForwardInputs(
            q=q + 1,
            k=k,
            v=v,
            batch_size=batch_size,
            seq_len=seq_len,
            seq_len_kv=seq_len_kv,
            attention_mask=attention_mask,
            sparse_runtime_params=SparseRuntimeParams(block_sparse_inputs=self.predicted_carrier),
            forward_kwargs=forward_kwargs,
        )
        return self.prediction

    def _forward_impl(
        self,
        q,
        k,
        v,
        batch_size,
        seq_len,
        attention_mask=PredefinedAttentionMask.FULL,
        seq_len_kv=None,
        sparse_runtime_params=None,
        **kwargs,
    ):
        self.lifecycle_events.append("forward_impl")
        self.impl_inputs = {
            "q": q,
            "k": k,
            "v": v,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "attention_mask": attention_mask,
            "seq_len_kv": seq_len_kv,
            "sparse_runtime_params": sparse_runtime_params,
            "kwargs": kwargs,
        }
        return q.reshape(batch_size, seq_len, -1)

    def sparse_post_process(self, output, prediction):
        self.lifecycle_events.append("post_process")
        self.post_process_prediction = prediction
        return output + 2


class _NoPredictionSparseAttention(_RecordingSparseAttention):
    def block_sparse_attn_predict(self, *args, forward_kwargs, **kwargs):
        del args, kwargs
        self.lifecycle_events.append("predict")
        forward_kwargs.pop("private_sparse_phase", None)
        return None


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class _ExtendedSparseForwardInputs(visual_trtllm.SparseForwardInputs):
    post_marker: str


def _make_block_sparse_inputs():
    return BlockSparseForwardInputs(
        q_block_size=64,
        kv_block_size=64,
        max_blocks_per_row=1,
        block_indptr=torch.tensor([[[0, 1]]], dtype=torch.int32),
        block_indices=torch.tensor([0], dtype=torch.int32),
    )


def test_sparse_forward_inputs_owns_only_aggregate_sparse_runtime_params():
    runtime_params = SparseRuntimeParams(
        block_sparse_inputs=_make_block_sparse_inputs(),
    )
    inputs = visual_trtllm.SparseForwardInputs(
        q=torch.empty((1, 4, 2, 8)),
        k=None,
        v=None,
        batch_size=1,
        seq_len=4,
        seq_len_kv=4,
        attention_mask=PredefinedAttentionMask.FULL,
        sparse_runtime_params=runtime_params,
    )

    fields = visual_trtllm.SparseForwardInputs.__dataclass_fields__
    assert inputs.sparse_runtime_params is runtime_params
    assert fields["sparse_runtime_params"].type is SparseRuntimeParams
    assert "block_sparse_inputs" not in fields


class _StopAtFmhaDispatch(Exception):
    pass


def _make_core_forward_metadata():
    metadata = object.__new__(visual_trtllm.BaseTrtllmAttentionMetadata)
    seq_lens = torch.tensor([4], dtype=torch.int32)
    metadata._seq_lens = seq_lens
    metadata._seq_lens_kv = seq_lens
    metadata._seq_lens_cuda = None
    metadata.kv_cache_manager = None
    metadata._max_seq_len_storage = 4
    metadata.use_paged_context_fmha = False
    metadata.cu_q_seqlens = None
    metadata.cu_kv_seqlens = None
    metadata.enable_flash_mla = False
    metadata.spec_bl_tree_first_sparse_mask_offset_kv = None
    metadata.spec_decoding_bl_tree_mask = None
    metadata.kv_lens_cuda_runtime = torch.tensor([4], dtype=torch.int32)
    metadata.kv_lens_runtime = torch.tensor([4], dtype=torch.int32)
    metadata.prompt_lens_cuda_runtime = torch.tensor([4], dtype=torch.int32)
    metadata.prompt_lens_cpu_runtime = torch.tensor([4], dtype=torch.int32)
    metadata.host_request_types_runtime = torch.tensor([0], dtype=torch.int32)
    metadata.max_context_q_len_override = None
    return metadata


def test_trtllm_attention_metadata_caches_distinct_seq_lens(monkeypatch):
    monkeypatch.setattr(
        visual_trtllm,
        "BaseTrtllmAttentionMetadata",
        _FakeBaseTrtllmAttentionMetadata,
    )
    attention_metadata_state = {}
    metadata = visual_trtllm.TrtllmAttentionMetadata(
        device=torch.device("cpu"),
        attention_metadata_state=attention_metadata_state,
    )

    first_seq_lens = torch.tensor([64], dtype=torch.int32)
    first_metadata = metadata.prepare(batch_size=1, seq_lens=first_seq_lens)
    first_seq_lens.fill_(999)

    second_metadata = metadata.prepare(batch_size=1, seq_lens=torch.tensor([96], dtype=torch.int32))
    first_metadata_again = metadata.prepare(
        batch_size=1,
        seq_lens=torch.tensor([64], dtype=torch.int32),
    )

    assert first_metadata is first_metadata_again
    assert first_metadata is not second_metadata
    assert first_metadata.prepare_calls == 1
    assert second_metadata.prepare_calls == 1

    metadata_cache = attention_metadata_state["metadata_cache"]
    assert set(metadata_cache) == {
        (1, (64,)),
        (1, (96,)),
    }
    assert metadata_cache[(1, (64,))]["metadata"] is first_metadata
    assert metadata_cache[(1, (96,))]["metadata"] is second_metadata

    first_cached_seq_lens = metadata_cache[(1, (64,))]["seq_lens"]
    second_cached_seq_lens = metadata_cache[(1, (96,))]["seq_lens"]
    assert torch.equal(first_cached_seq_lens, torch.tensor([64], dtype=torch.int32))
    assert torch.equal(second_cached_seq_lens, torch.tensor([96], dtype=torch.int32))
    assert first_cached_seq_lens is not second_cached_seq_lens
    assert first_cached_seq_lens.data_ptr() != second_cached_seq_lens.data_ptr()
    assert first_metadata.seq_lens is first_cached_seq_lens
    assert second_metadata.seq_lens is second_cached_seq_lens


def test_trtllm_attention_layers_share_block_sparse_plan_cache(monkeypatch):
    from tensorrt_llm._torch.attention_backend.fmha import prims_ts_block_sparse

    def _base_update_quant_config(self, new_quant_config):
        del new_quant_config
        self._fmha_manager = SimpleNamespace(
            fmha_libs=[prims_ts_block_sparse.PrimsTSBlockSparseFmha(self)]
        )

    def _base_init(self, **kwargs):
        del kwargs
        self.is_mla_enable = False
        self.kv_lora_rank = None
        self.v_head_dim = None
        self.head_dim = 64
        self.update_quant_config(None)

    monkeypatch.setattr(
        visual_trtllm.BaseTrtllmAttention,
        "update_quant_config",
        _base_update_quant_config,
    )
    monkeypatch.setattr(visual_trtllm.BaseTrtllmAttention, "__init__", _base_init)
    attention_metadata_state = create_attention_metadata_state()
    assert "block_sparse_fmha_cache" not in attention_metadata_state

    first = visual_trtllm.TrtllmAttention(
        attention_metadata_state=attention_metadata_state,
    )
    second = visual_trtllm.TrtllmAttention(
        attention_metadata_state=attention_metadata_state,
    )

    assert not hasattr(first, "_block_sparse_fmha_cache_state")
    assert not hasattr(second, "_block_sparse_fmha_cache_state")
    first_fmha = first._fmha_manager.fmha_libs[0]
    second_fmha = second._fmha_manager.fmha_libs[0]
    assert first_fmha._contiguous_wrappers is second_fmha._contiguous_wrappers
    assert first_fmha._paged_wrappers is second_fmha._paged_wrappers

    first.update_quant_config(None)
    first_fmha = first._fmha_manager.fmha_libs[0]
    assert first_fmha._contiguous_wrappers is second_fmha._contiguous_wrappers
    assert first_fmha._paged_wrappers is second_fmha._paged_wrappers
    assert attention_metadata_state["fmha_caches"]["prims_ts_block_sparse"] == {
        "contiguous_wrappers": {},
        "paged_wrappers": {},
    }

    other = visual_trtllm.TrtllmAttention(
        attention_metadata_state=create_attention_metadata_state(),
    )
    other_fmha = other._fmha_manager.fmha_libs[0]
    assert first_fmha._contiguous_wrappers is not other_fmha._contiguous_wrappers
    assert first_fmha._paged_wrappers is not other_fmha._paged_wrappers


def test_sparse_forward_inputs_are_immutable_and_copy_forward_kwargs():
    q = torch.randn(1, 4, 2, 8)
    kwargs = {"timestep": torch.tensor([12]), "sparse_phase": "sparse"}
    prediction = visual_trtllm.SparseForwardInputs(
        q=q,
        k=None,
        v=None,
        batch_size=1,
        seq_len=4,
        seq_len_kv=4,
        attention_mask=PredefinedAttentionMask.FULL,
        sparse_runtime_params=SparseRuntimeParams(),
        forward_kwargs=kwargs,
    )
    kwargs["sparse_phase"] = "dense"

    assert prediction.q is q
    assert prediction.forward_kwargs["sparse_phase"] == "sparse"
    with pytest.raises(FrozenInstanceError):
        prediction.seq_len = 8
    with pytest.raises(TypeError):
        prediction.forward_kwargs["sparse_phase"] = "dense"


def test_sparse_forward_inputs_use_identity_equality_and_compact_repr():
    prediction = visual_trtllm.SparseForwardInputs(
        q=torch.randn(1, 4, 2, 8),
        k=None,
        v=None,
        batch_size=1,
        seq_len=4,
        seq_len_kv=4,
        attention_mask=PredefinedAttentionMask.FULL,
        sparse_runtime_params=SparseRuntimeParams(),
    )

    copied_prediction = replace(prediction)

    assert copied_prediction is not prediction
    assert copied_prediction != prediction
    assert "tensor(" not in repr(prediction)


def test_sparse_lifecycle_accepts_typed_prediction_context():
    class _TypedSparseAttention(visual_trtllm.TrtllmAttention):
        def block_sparse_attn_predict(
            self,
            q,
            k,
            v,
            *,
            batch_size,
            seq_len,
            seq_len_kv,
            attention_mask,
            forward_kwargs,
        ) -> _ExtendedSparseForwardInputs:
            return _ExtendedSparseForwardInputs(
                q=q,
                k=k,
                v=v,
                post_marker="vsa",
                batch_size=batch_size,
                seq_len=seq_len,
                seq_len_kv=seq_len_kv,
                attention_mask=attention_mask,
                sparse_runtime_params=SparseRuntimeParams(),
                forward_kwargs=forward_kwargs,
            )

        def _forward_impl(self, q, *args, **kwargs):
            return q.reshape(1, 4, -1)

        def sparse_post_process(self, output, sparse_inputs):
            self.post_inputs = sparse_inputs
            return output

    attention = object.__new__(_TypedSparseAttention)
    q = torch.randn(1, 4, 2, 8)

    output = attention.forward(q, None, None, batch_size=1, seq_len=4)

    assert isinstance(attention.post_inputs, _ExtendedSparseForwardInputs)
    assert attention.post_inputs.post_marker == "vsa"
    assert output.shape == (1, 4, 16)


def test_sparse_lifecycle_does_not_require_generic_parameterization():
    assert getattr(visual_trtllm.TrtllmAttention, "__parameters__", ()) == ()


def test_sparse_lifecycle_default_predictor_returns_none():
    attention = object.__new__(visual_trtllm.TrtllmAttention)
    q = torch.randn(1, 4, 2, 8)

    assert "block_sparse_attn_predict" in visual_trtllm.TrtllmAttention.__dict__
    assert "sparse_attn_predict" not in visual_trtllm.TrtllmAttention.__dict__
    assert "sparse_predict" not in visual_trtllm.TrtllmAttention.__dict__
    assert (
        attention.block_sparse_attn_predict(
            q,
            None,
            None,
            batch_size=1,
            seq_len=4,
            seq_len_kv=4,
            attention_mask=PredefinedAttentionMask.FULL,
            forward_kwargs={},
        )
        is None
    )
    assert "_enable_sparse_workflow" not in visual_trtllm.TrtllmAttention.__dict__
    assert "_should_use_sparse_workflow" not in visual_trtllm.TrtllmAttention.__dict__


def test_sparse_lifecycle_predicts_block_inputs_then_runs_normal_forward():
    assert "sparse_preprocess" not in visual_trtllm.TrtllmAttention.__dict__
    attention = object.__new__(_RecordingSparseAttention)
    attention.lifecycle_events = []
    attention.predicted_carrier = _make_block_sparse_inputs()
    q = torch.randn(1, 4, 2, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    timestep = torch.tensor([12])
    gate_compress = torch.randn(1, 4, 1)

    output = attention.forward(
        q,
        k,
        v,
        batch_size=1,
        seq_len=4,
        attention_mask=PredefinedAttentionMask.FULL,
        timestep=timestep,
        gate_compress=gate_compress,
    )

    assert attention.lifecycle_events == ["predict", "forward_impl", "post_process"]
    assert attention.predict_inputs["q"] is q
    assert attention.impl_inputs["q"] is attention.prediction.q
    assert attention.impl_inputs["k"] is k
    assert attention.impl_inputs["v"] is v
    core_runtime_params = attention.impl_inputs["sparse_runtime_params"]
    assert isinstance(core_runtime_params, SparseRuntimeParams)
    assert core_runtime_params.block_sparse_inputs is attention.predicted_carrier
    assert attention.impl_inputs["kwargs"] == {
        "timestep": timestep,
        "gate_compress": gate_compress,
    }
    assert attention.post_process_prediction is attention.prediction
    torch.testing.assert_close(output, (q + 1).reshape(1, 4, -1) + 2)


def test_sparse_lifecycle_dense_fallback_still_hands_off_prediction_sentinel():
    attention = object.__new__(_RecordingSparseAttention)
    attention.lifecycle_events = []
    attention.predicted_carrier = None
    q = torch.randn(1, 4, 2, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    output = attention.forward(q, k, v, batch_size=1, seq_len=4)

    assert attention.lifecycle_events == ["predict", "forward_impl", "post_process"]
    assert attention.impl_inputs["q"] is attention.prediction.q
    assert attention.impl_inputs["k"] is k
    assert attention.impl_inputs["v"] is v
    core_runtime_params = attention.impl_inputs["sparse_runtime_params"]
    assert isinstance(core_runtime_params, SparseRuntimeParams)
    assert core_runtime_params.block_sparse_inputs is None
    assert attention.post_process_prediction is attention.prediction
    torch.testing.assert_close(output, (q + 1).reshape(1, 4, -1) + 2)


def test_sparse_lifecycle_none_prediction_uses_normal_forward_and_cleans_kwargs():
    attention = object.__new__(_NoPredictionSparseAttention)
    attention.lifecycle_events = []
    attention.predicted_carrier = _make_block_sparse_inputs()
    q = torch.randn(1, 4, 2, 8)
    timestep = torch.tensor([12])

    output = attention.forward(
        q,
        None,
        None,
        batch_size=1,
        seq_len=4,
        timestep=timestep,
        private_sparse_phase="dense",
    )

    assert attention.lifecycle_events == ["predict", "forward_impl"]
    assert "private_sparse_phase" not in attention.impl_inputs["kwargs"]
    torch.testing.assert_close(output, q.reshape(1, 4, -1))


def test_sparse_lifecycle_exposes_only_aggregate_runtime_params():
    assert (
        "block_sparse_inputs"
        not in inspect.signature(visual_trtllm.TrtllmAttention._forward_impl).parameters
    )
    assert (
        "block_sparse_inputs"
        not in inspect.signature(visual_trtllm.TrtllmAttention.block_sparse_attn_predict).parameters
    )
    assert (
        "sparse_runtime_params"
        in inspect.signature(visual_trtllm.TrtllmAttention._forward_impl).parameters
    )


@pytest.mark.parametrize("legacy_kwarg", ["block_sparse_inputs", "sparse_runtime_params"])
def test_sparse_lifecycle_rejects_legacy_sparse_input_kwargs(legacy_kwarg):
    attention = object.__new__(visual_trtllm.TrtllmAttention)
    q = torch.randn(1, 4, 2, 8)

    with pytest.raises(TypeError, match=r"block_sparse_attn_predict.*SparseRuntimeParams"):
        attention.forward(
            q,
            None,
            None,
            batch_size=1,
            seq_len=4,
            **{legacy_kwarg: object()},
        )


def test_dense_lifecycle_rejects_unexpected_kwargs_before_metadata_or_core(monkeypatch):
    prepare_metadata = Mock(return_value=object())
    core_forward = Mock(return_value=torch.empty(4, 16))
    monkeypatch.setattr(
        visual_trtllm.TrtllmAttention,
        "_prepare_metadata",
        prepare_metadata,
    )
    monkeypatch.setattr(
        visual_trtllm.BaseTrtllmAttention,
        "forward",
        core_forward,
    )
    attention = object.__new__(visual_trtllm.TrtllmAttention)
    attention.quant_attention_config = None

    with pytest.raises(TypeError) as exc_info:
        attention.forward(
            torch.randn(1, 4, 2, 8),
            None,
            None,
            batch_size=1,
            seq_len=4,
            attention_maks=PredefinedAttentionMask.FULL,
            timstep=torch.tensor([12]),
        )

    assert str(exc_info.value) == (
        "Unexpected TRTLLM attention forward keyword arguments: attention_maks, timstep"
    )
    prepare_metadata.assert_not_called()
    core_forward.assert_not_called()


def test_predicted_lifecycle_rejects_unexpected_forward_kwargs_before_setup(monkeypatch):
    q = torch.randn(1, 4, 2, 8)
    timestep = torch.tensor([12])
    sparse_inputs = visual_trtllm.SparseForwardInputs(
        q=q,
        k=None,
        v=None,
        batch_size=1,
        seq_len=4,
        seq_len_kv=4,
        attention_mask=PredefinedAttentionMask.FULL,
        sparse_runtime_params=SparseRuntimeParams(),
        forward_kwargs={
            "timestep": timestep,
            "gate_fnne": torch.randn_like(q),
            "route_modde": "sparse",
        },
    )
    prepare_metadata = Mock(return_value=object())
    core_forward = Mock(return_value=torch.empty(4, 16))
    monkeypatch.setattr(
        visual_trtllm.TrtllmAttention,
        "block_sparse_attn_predict",
        lambda *args, **kwargs: sparse_inputs,
    )
    monkeypatch.setattr(
        visual_trtllm.TrtllmAttention,
        "_prepare_metadata",
        prepare_metadata,
    )
    monkeypatch.setattr(
        visual_trtllm.BaseTrtllmAttention,
        "forward",
        core_forward,
    )
    attention = object.__new__(visual_trtllm.TrtllmAttention)
    attention.quant_attention_config = None

    with pytest.raises(TypeError) as exc_info:
        attention.forward(q, None, None, batch_size=1, seq_len=4)

    assert str(exc_info.value) == (
        "Unexpected TRTLLM attention forward keyword arguments: gate_fnne, route_modde"
    )
    prepare_metadata.assert_not_called()
    core_forward.assert_not_called()


def test_dense_trtllm_attention_runs_noop_predictor_without_post_process(monkeypatch):
    captured = {}
    events = []
    q = torch.randn(1, 4, 2, 8)

    def _no_prediction(*args, **kwargs):
        del args, kwargs
        events.append("predict")
        return None

    def _fail_post_process(*args, **kwargs):
        raise AssertionError(f"no-op prediction invoked post-process: {args}, {kwargs}")

    def _capture_impl(self, *args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return q.reshape(1, 4, -1)

    monkeypatch.setattr(
        visual_trtllm.TrtllmAttention,
        "block_sparse_attn_predict",
        _no_prediction,
        raising=False,
    )
    monkeypatch.setattr(
        visual_trtllm.TrtllmAttention,
        "sparse_post_process",
        _fail_post_process,
        raising=False,
    )
    monkeypatch.setattr(
        visual_trtllm.TrtllmAttention, "_forward_impl", _capture_impl, raising=False
    )

    attention = object.__new__(visual_trtllm.TrtllmAttention)
    output = attention.forward(q, None, None, batch_size=1, seq_len=4)

    assert events == ["predict"]
    assert captured["args"][:5] == (q, None, None, 1, 4)
    assert output.shape == (1, 4, 16)


def test_plain_trtllm_attention_impl_forwards_precomputed_sparse_runtime_params(monkeypatch):
    captured = {}
    prepared_metadata = object()

    monkeypatch.setattr(
        visual_trtllm.TrtllmAttention,
        "_prepare_metadata",
        lambda self, batch_size, seq_len: prepared_metadata,
    )

    def _capture_base_forward(self, q, k, v, metadata, forward_args=None, **kwargs):
        captured.update(
            q=q,
            k=k,
            v=v,
            metadata=metadata,
            forward_args=forward_args,
            kwargs=kwargs,
        )
        return q

    monkeypatch.setattr(
        visual_trtllm.BaseTrtllmAttention,
        "forward",
        _capture_base_forward,
    )

    attention = object.__new__(visual_trtllm.TrtllmAttention)
    attention.quant_attention_config = None
    runtime_params = SparseRuntimeParams(block_sparse_inputs=_make_block_sparse_inputs())
    q = torch.randn(1, 4, 2, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    output = attention._forward_impl(
        q,
        k,
        v,
        batch_size=1,
        seq_len=4,
        sparse_runtime_params=runtime_params,
    )

    assert output.shape == (1, 4, 16)
    assert captured["q"].shape == (4, 16)
    assert captured["k"].shape == (4, 16)
    assert captured["v"].shape == (4, 16)
    assert captured["metadata"] is prepared_metadata
    assert captured["forward_args"].sparse_runtime_params is runtime_params
    assert captured["kwargs"] == {}


@pytest.mark.parametrize("has_block_sparse_inputs", [False, True])
def test_sparse_lifecycle_passes_only_typed_prediction_to_core(
    monkeypatch,
    has_block_sparse_inputs,
):
    captured = {}
    prepared_metadata = object()

    class _CoreHandoffSparseAttention(visual_trtllm.TrtllmAttention):
        def block_sparse_attn_predict(
            self,
            q,
            k,
            v,
            *,
            batch_size,
            seq_len,
            seq_len_kv,
            attention_mask,
            forward_kwargs,
        ):
            self.prediction = visual_trtllm.SparseForwardInputs(
                q=q + 1,
                k=k,
                v=v,
                batch_size=batch_size,
                seq_len=seq_len,
                seq_len_kv=seq_len_kv,
                attention_mask=attention_mask,
                sparse_runtime_params=SparseRuntimeParams(
                    block_sparse_inputs=self.predicted_carrier
                ),
                forward_kwargs=forward_kwargs,
            )
            return self.prediction

        def sparse_post_process(self, output, sparse_inputs):
            self.post_process_prediction = sparse_inputs
            return output + 2

    monkeypatch.setattr(
        _CoreHandoffSparseAttention,
        "_prepare_metadata",
        lambda self, batch_size, seq_len: prepared_metadata,
    )
    monkeypatch.setattr(
        _CoreHandoffSparseAttention,
        "_concat_qkv",
        lambda self, q, k, v, batch_size, seq_len, kv_seq_len: torch.cat(
            [
                q.reshape(batch_size * seq_len, -1),
                k.reshape(batch_size * kv_seq_len, -1),
                v.reshape(batch_size * kv_seq_len, -1),
            ],
            dim=-1,
        ),
    )

    def _capture_base_forward(self, q, k, v, metadata, forward_args=None, **kwargs):
        captured.update(
            q=q,
            k=k,
            v=v,
            metadata=metadata,
            forward_args=forward_args,
            kwargs=kwargs,
        )
        return q[:, :16]

    monkeypatch.setattr(
        visual_trtllm.BaseTrtllmAttention,
        "forward",
        _capture_base_forward,
    )

    attention = object.__new__(_CoreHandoffSparseAttention)
    attention.quant_attention_config = None
    attention.predicted_carrier = _make_block_sparse_inputs() if has_block_sparse_inputs else None
    q = torch.randn(1, 4, 2, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    output = attention.forward(q, k, v, batch_size=1, seq_len=4)

    forward_args = captured["forward_args"]
    assert isinstance(forward_args.sparse_runtime_params, SparseRuntimeParams)
    assert forward_args.sparse_runtime_params.block_sparse_inputs is attention.predicted_carrier
    assert captured["metadata"] is prepared_metadata
    if has_block_sparse_inputs:
        torch.testing.assert_close(captured["q"], (q + 1).reshape(4, 16))
        assert captured["k"].data_ptr() == k.data_ptr()
        assert captured["v"].data_ptr() == v.data_ptr()
    else:
        torch.testing.assert_close(captured["q"][:, :16], (q + 1).reshape(4, 16))
        assert captured["k"] is None
        assert captured["v"] is None
    assert captured["kwargs"] == {}
    assert attention.post_process_prediction is attention.prediction
    torch.testing.assert_close(output, (q + 1).reshape(1, 4, 16) + 2)


@pytest.mark.parametrize("has_block_sparse_inputs", [False, True])
def test_sparse_lifecycle_reaches_core_fmha_with_precomputed_prediction(
    monkeypatch,
    has_block_sparse_inputs,
):
    class _CoreHandoffSparseAttention(visual_trtllm.TrtllmAttention):
        def block_sparse_attn_predict(
            self,
            q,
            k,
            v,
            *,
            batch_size,
            seq_len,
            seq_len_kv,
            attention_mask,
            forward_kwargs,
        ):
            return visual_trtllm.SparseForwardInputs(
                q=q,
                k=k,
                v=v,
                batch_size=batch_size,
                seq_len=seq_len,
                seq_len_kv=seq_len_kv,
                attention_mask=attention_mask,
                sparse_runtime_params=SparseRuntimeParams(
                    block_sparse_inputs=self.predicted_carrier
                ),
                forward_kwargs=forward_kwargs,
            )

    metadata = _make_core_forward_metadata()
    monkeypatch.setattr(
        _CoreHandoffSparseAttention,
        "_prepare_metadata",
        lambda self, batch_size, seq_len: metadata,
    )

    attention = object.__new__(_CoreHandoffSparseAttention)
    attention.quant_attention_config = None
    attention.sparse_params = None
    attention.is_mla_enable = False
    attention.num_heads = 2
    attention.num_kv_heads = 2
    attention.head_dim = 8
    attention.get_local_layer_idx = Mock(return_value=0)
    attention._ensure_rope_table_size = Mock()
    attention.print_skip_softmax_stat = False
    attention.kv_scale_orig_quant = None
    attention.kv_scale_quant_orig = None
    attention.predict_sparse_attention = Mock()
    select_fmha = Mock(side_effect=_StopAtFmhaDispatch)
    attention._fmha_manager = SimpleNamespace(
        fmha_libs=[object()],
        select=select_fmha,
    )
    attention.predicted_carrier = _make_block_sparse_inputs() if has_block_sparse_inputs else None
    q = torch.randn(1, 4, 2, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    with pytest.raises(_StopAtFmhaDispatch):
        attention.forward(q, k, v, batch_size=1, seq_len=4)

    attention.predict_sparse_attention.assert_not_called()
    select_fmha.assert_called_once()
    core_forward_args = select_fmha.call_args.args[5]
    runtime_params = core_forward_args.sparse_runtime_params
    assert isinstance(runtime_params, SparseRuntimeParams)
    assert runtime_params.block_sparse_inputs is attention.predicted_carrier

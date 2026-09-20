# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.params import (
    BlockSparseForwardInputs,
    SparseBackendForwardArgs,
    SparseRuntimeParams,
)
from tensorrt_llm._torch.attention.backends.sparse.skip_softmax import (
    SkipSoftmaxParams,
    SkipSoftmaxScheduler,
)
from tensorrt_llm._torch.visual_gen.attention_backend import trtllm as visual_trtllm
from tensorrt_llm._torch.visual_gen.config import create_attention_metadata_state

_REQUIRES_SM100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the VisualGen TRTLLM FMHA path is exercised on SM100 or SM103",
)


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


def _make_block_sparse_inputs():
    return BlockSparseForwardInputs(
        q_block_size=64,
        kv_block_size=64,
        max_blocks_per_row=1,
        block_indptr=torch.tensor([[[0, 1]]], dtype=torch.int32),
        block_indices=torch.tensor([0], dtype=torch.int32),
    )


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


def _make_wrapper(cls=visual_trtllm.TrtllmAttention, *, quant_attention_config=None):
    attention = object.__new__(cls)
    attention.quant_attention_config = quant_attention_config
    attention.layer_idx = 0
    attention.sparse_params = None
    attention.metadata = visual_trtllm.TrtllmAttentionMetadata(
        device=torch.device("cpu"), attention_metadata_state={}
    )
    return attention


def _capture_core_forward(monkeypatch, captured: dict):
    prepared_metadata = object()
    monkeypatch.setattr(
        visual_trtllm.TrtllmAttention,
        "_prepare_metadata",
        lambda self, batch_size, seq_len: prepared_metadata,
    )
    monkeypatch.setattr(
        visual_trtllm.TrtllmAttention,
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

    monkeypatch.setattr(visual_trtllm.BaseTrtllmAttention, "forward", _capture_base_forward)
    return prepared_metadata


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
    from tensorrt_llm._torch.attention.backends.fmha import prims_ts_block_sparse

    def _base_update_quant_config(self, new_quant_config):
        del new_quant_config
        self._fmha_manager = SimpleNamespace(
            fmha_libs=[prims_ts_block_sparse.PrimsTSBlockSparseFmha(self)]
        )

    def _base_init(self, **kwargs):
        fmha_state = kwargs.get("fmha_state")
        self.fmha_state = {} if fmha_state is None else fmha_state
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
    assert "fmha_caches" not in attention_metadata_state

    first = visual_trtllm.TrtllmAttention(
        attention_metadata_state=attention_metadata_state,
    )
    second = visual_trtllm.TrtllmAttention(
        attention_metadata_state=attention_metadata_state,
    )

    assert first.fmha_state is attention_metadata_state["fmha_caches"]
    assert second.fmha_state is first.fmha_state
    assert "update_quant_config" not in visual_trtllm.TrtllmAttention.__dict__
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


def test_visual_gen_wrapper_owns_only_the_timestep_schedule():
    assert not hasattr(visual_trtllm, "SparseForwardInputs")
    for name in ("block_sparse_attn_predict", "sparse_post_process", "_forward_impl"):
        assert name not in visual_trtllm.TrtllmAttention.__dict__
    for name in ("resolve_timestep", "should_use_sparse"):
        assert name in visual_trtllm.TrtllmAttention.__dict__
    assert getattr(visual_trtllm.TrtllmAttention, "__parameters__", ()) == ()


def test_wrapper_schedule_follows_cutoff_and_dense_layers():
    attention = _make_wrapper()
    assert attention.should_use_sparse(0.8)
    assert attention.should_use_sparse(None)

    attention.layer_idx = 1
    attention.sparse_params = SimpleNamespace(
        disabled_until_timestep=0.6, dense_layers=frozenset({3})
    )
    assert not attention.should_use_sparse(0.8)
    assert attention.should_use_sparse(0.2)
    assert attention.should_use_sparse(None)
    attention.layer_idx = 3
    assert not attention.should_use_sparse(0.2)


def test_wrapper_passes_timestep_through_without_a_cutoff():
    attention = _make_wrapper()
    timestep = torch.tensor([0.2])

    assert attention.resolve_timestep(timestep) is timestep
    assert attention.resolve_timestep(None) is None


def test_wrapper_prepares_timestep_for_cuda_graph_capture(monkeypatch):
    attention = _make_wrapper()
    attention.sparse_params = SimpleNamespace(disabled_until_timestep=0.6)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="prepared before CUDA Graph capture"):
        attention.resolve_timestep(torch.tensor([0.8]))

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    # Per-token timesteps reduce to the largest live value.
    assert attention.resolve_timestep(torch.tensor([0.0, 0.8])) == pytest.approx(0.8)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert attention.resolve_timestep(torch.tensor([0.2])) == pytest.approx(0.8)


def test_trtllm_attention_metadata_prepares_timestep_for_capture(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    state = {}
    metadata = visual_trtllm.TrtllmAttentionMetadata(
        device=torch.device("cpu"), attention_metadata_state=state
    )
    with pytest.raises(RuntimeError, match="prepared before CUDA Graph capture"):
        metadata.prepare_timestep(torch.tensor([0.8]))

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert metadata.prepare_timestep(torch.tensor([0.0, 0.8])) == pytest.approx(0.8)
    assert state["timestep"] == pytest.approx(0.8)
    assert metadata.prepare_timestep(None) is None
    assert state["timestep"] is None

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    state["timestep"] = 0.4
    assert metadata.prepare_timestep(torch.tensor([0.9])) == pytest.approx(0.4)


def test_forward_hands_the_prepared_timestep_to_the_core(monkeypatch):
    captured: dict = {}
    _capture_core_forward(monkeypatch, captured)
    attention = _make_wrapper()
    attention.sparse_params = SimpleNamespace(disabled_until_timestep=0.6)
    monkeypatch.setattr(attention, "support_fused_qkv", lambda: True, raising=False)
    q = torch.randn(1, 4, 2, 8)

    attention.forward(q, q, q, batch_size=1, seq_len=4, timestep=torch.tensor([0.0, 0.8]))

    assert captured["forward_args"].timestep == pytest.approx(0.8)


def test_forward_ignores_backend_specific_kwargs_like_other_backends(monkeypatch):
    captured: dict = {}
    _capture_core_forward(monkeypatch, captured)
    attention = _make_wrapper()
    monkeypatch.setattr(attention, "support_fused_qkv", lambda: True, raising=False)
    q = torch.randn(1, 4, 2, 8)

    output = attention.forward(
        q,
        q,
        q,
        batch_size=1,
        seq_len=4,
        key_padding_mask=torch.ones(1, 4, dtype=torch.bool),
        gate_compress=torch.ones(1, 4, 2, 8),
    )

    assert output.shape == (1, 4, 16)
    assert captured["kwargs"] == {}
    assert not hasattr(captured["forward_args"], "key_padding_mask")


def test_forward_flattens_fused_qkv_without_copy(monkeypatch):
    captured = {}
    prepared_metadata = _capture_core_forward(monkeypatch, captured)
    attention = _make_wrapper()
    qkv = torch.randn(1, 4, 6, 8)
    timestep = torch.tensor([12])

    output = attention.forward(qkv, None, None, batch_size=1, seq_len=4, timestep=timestep)

    assert output.shape == (1, 4, 16)
    assert captured["q"].shape == (4, 48)
    assert captured["q"].data_ptr() == qkv.data_ptr()
    assert captured["k"] is None and captured["v"] is None
    assert captured["metadata"] is prepared_metadata
    assert captured["forward_args"].timestep is timestep
    assert captured["forward_args"].sparse_backend_args is None
    assert captured["forward_args"].sparse_runtime_params == SparseRuntimeParams()
    assert captured["kwargs"] == {}


def test_forward_fuses_separate_qkv_without_sparse_backend_args(monkeypatch):
    captured = {}
    _capture_core_forward(monkeypatch, captured)
    attention = _make_wrapper()
    q = torch.randn(1, 4, 2, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    attention.forward(q, k, v, batch_size=1, seq_len=4)

    assert captured["q"].shape == (4, 48)
    torch.testing.assert_close(captured["q"][:, :16], q.reshape(4, 16))
    assert captured["k"] is None and captured["v"] is None
    assert captured["forward_args"].sparse_backend_args is None


def test_forward_hands_separate_qkv_and_backend_args_to_core_for_block_sparse_routes(
    monkeypatch,
):
    captured = {}
    _capture_core_forward(monkeypatch, captured)
    attention = _make_wrapper()
    q = torch.randn(1, 4, 2, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    backend_args = SparseBackendForwardArgs(block_sparse_inputs=_make_block_sparse_inputs())

    output = attention.forward(
        q,
        k,
        v,
        batch_size=1,
        seq_len=4,
        sparse_backend_args=backend_args,
    )

    assert output.shape == (1, 4, 16)
    assert captured["q"].data_ptr() == q.data_ptr()
    assert captured["k"].data_ptr() == k.data_ptr()
    assert captured["v"].data_ptr() == v.data_ptr()
    assert captured["q"].shape == captured["k"].shape == captured["v"].shape == (4, 16)
    assert captured["forward_args"].sparse_backend_args is backend_args
    assert captured["forward_args"].sparse_runtime_params == SparseRuntimeParams()


def test_forward_hands_separate_qkv_to_core_when_backend_rejects_fused_qkv(monkeypatch):
    class _SeparateQkvAttention(visual_trtllm.TrtllmAttention):
        @classmethod
        def support_fused_qkv(cls) -> bool:
            return False

    captured = {}
    _capture_core_forward(monkeypatch, captured)
    attention = _make_wrapper(_SeparateQkvAttention)
    q = torch.randn(1, 4, 2, 8)

    attention.forward(q, q, q, batch_size=1, seq_len=4)

    assert captured["k"] is not None and captured["v"] is not None
    assert captured["q"].shape == (4, 16)
    assert captured["forward_args"].sparse_backend_args is None


def test_forward_applies_sage_quantization_to_separate_qkv(monkeypatch):
    captured = {}
    _capture_core_forward(monkeypatch, captured)
    quant_cfg = SimpleNamespace(q_block_size=1, k_block_size=2, v_block_size=3, qk_dtype="int8")
    attention = _make_wrapper(quant_attention_config=quant_cfg)
    q = torch.randn(1, 4, 2, 8)

    attention.forward(q, q, q, batch_size=1, seq_len=4)

    forward_args = captured["forward_args"]
    assert captured["k"] is not None and captured["v"] is not None
    assert forward_args.sage_attn_num_elts_per_blk_q == 1
    assert forward_args.sage_attn_num_elts_per_blk_k == 2
    assert forward_args.sage_attn_num_elts_per_blk_v == 3
    assert forward_args.sage_attn_qk_int8 is True


def test_forward_requires_separate_qkv_for_block_sparse_routes(monkeypatch):
    prepare_metadata = Mock(return_value=object())
    monkeypatch.setattr(visual_trtllm.TrtllmAttention, "_prepare_metadata", prepare_metadata)
    attention = _make_wrapper()
    backend_args = SparseBackendForwardArgs(block_sparse_inputs=_make_block_sparse_inputs())

    with pytest.raises(ValueError, match="separate q, k, and v"):
        attention.forward(
            torch.randn(1, 4, 6, 8),
            None,
            None,
            batch_size=1,
            seq_len=4,
            sparse_backend_args=backend_args,
        )

    prepare_metadata.assert_not_called()


def test_forward_rejects_block_sparse_routes_with_quant_config(monkeypatch):
    prepare_metadata = Mock(return_value=object())
    monkeypatch.setattr(visual_trtllm.TrtllmAttention, "_prepare_metadata", prepare_metadata)
    attention = _make_wrapper(quant_attention_config=object())
    q = torch.randn(1, 4, 2, 8)
    backend_args = SparseBackendForwardArgs(block_sparse_inputs=_make_block_sparse_inputs())

    with pytest.raises(ValueError, match="quant_attention_config"):
        attention.forward(q, q, q, batch_size=1, seq_len=4, sparse_backend_args=backend_args)

    prepare_metadata.assert_not_called()


@pytest.mark.parametrize("has_block_sparse_inputs", [False, True])
def test_forward_reaches_core_fmha_with_module_predicted_routes(
    monkeypatch,
    has_block_sparse_inputs,
):
    metadata = _make_core_forward_metadata()
    monkeypatch.setattr(
        visual_trtllm.TrtllmAttention,
        "_prepare_metadata",
        lambda self, batch_size, seq_len: metadata,
    )

    attention = _make_wrapper()
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
    attention.sparse_kv_predict = Mock(return_value=(None, None))
    attention.sparse_attn_predict = Mock(return_value=(None, None))
    select_fmha = Mock(side_effect=_StopAtFmhaDispatch)
    attention._fmha_manager = SimpleNamespace(
        fmha_libs=[object()],
        select=select_fmha,
    )
    carrier = _make_block_sparse_inputs() if has_block_sparse_inputs else None
    backend_args = SparseBackendForwardArgs(block_sparse_inputs=carrier)
    q = torch.randn(1, 4, 2, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    with pytest.raises(_StopAtFmhaDispatch):
        attention.forward(q, k, v, batch_size=1, seq_len=4, sparse_backend_args=backend_args)

    select_fmha.assert_called_once()
    attention.sparse_kv_predict.assert_called_once()
    attention.sparse_attn_predict.assert_called_once()
    core_forward_args = select_fmha.call_args.args[5]
    assert core_forward_args.sparse_backend_args is backend_args
    runtime_params = core_forward_args.sparse_runtime_params
    assert isinstance(runtime_params, SparseRuntimeParams)
    assert runtime_params.block_sparse_inputs is carrier
    assert runtime_params.sparse_attn_indices_block_size == 0


@_REQUIRES_SM100
def test_trtllm_skip_softmax_cutoff_is_capturable_with_a_device_timestep() -> None:
    """The wrapper prepares the timestep during warmup so capture never reads the tensor."""
    params = SkipSoftmaxParams(
        scheduler=SkipSoftmaxScheduler(
            threshold_scale_factor_prefill=5000.0, disabled_until_timestep=0.6
        )
    )
    attention = visual_trtllm.TrtllmAttention(
        layer_idx=0,
        num_heads=2,
        head_dim=128,
        dtype=torch.bfloat16,
        attention_metadata_state=create_attention_metadata_state(),
        sparse_params=params,
    )
    batch_size, seq_len = 1, 256
    q = torch.randn(batch_size, seq_len, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    timestep = torch.tensor([0.8], device="cuda")

    def forward() -> torch.Tensor:
        return attention.forward(q, k, v, batch_size=batch_size, seq_len=seq_len, timestep=timestep)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            eager = forward()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward()
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(captured, eager)

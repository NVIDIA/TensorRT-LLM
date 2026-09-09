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
"""VSA correctness tests: backend dispatch, preprocessing, and kernel behavior.

Module-level dense-equivalence and finite-output checks live in
test_attention_integration.py.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention.backends.interface import PredefinedAttentionMask
from tensorrt_llm._torch.attention.backends.sparse.params import BlockSparseForwardInputs
from tensorrt_llm._torch.visual_gen.attention_backend.cute_dsl import CuTeDSLAttention
from tensorrt_llm._torch.visual_gen.attention_backend.interface import AttentionTensorLayout
from tensorrt_llm._torch.visual_gen.attention_backend.sparse.vsa import backend as vsa_backend
from tensorrt_llm._torch.visual_gen.attention_backend.sparse.vsa import kernels as vsa_kernels
from tensorrt_llm._torch.visual_gen.attention_backend.sparse.vsa import predictor as vsa_predictor
from tensorrt_llm._torch.visual_gen.attention_backend.sparse.vsa.backend import (
    VSACuTeDSLAttention,
    VSATrtllmAttention,
)
from tensorrt_llm._torch.visual_gen.attention_backend.sparse.vsa.kernels import tile_and_pool_cubes
from tensorrt_llm._torch.visual_gen.attention_backend.sparse.vsa.metadata import (
    VSA_BLOCK_SIZE,
    VSAMetadataBuilder,
    set_vsa_forward_context,
)
from tensorrt_llm._torch.visual_gen.attention_backend.sparse.vsa.predictor import (
    VSAForwardInputs,
    VSAPredictor,
)
from tensorrt_llm._torch.visual_gen.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._torch.visual_gen.attention_backend.utils import create_attention
from tensorrt_llm._torch.visual_gen.attention_backend.vanilla import VanillaAttention
from tensorrt_llm._torch.visual_gen.config import (
    DiffusionModelConfig,
    create_attention_metadata_state,
)
from tensorrt_llm._torch.visual_gen.modules import attention as attention_module
from tensorrt_llm._torch.visual_gen.modules.attention import Attention, QKVMode
from tensorrt_llm.visual_gen.args import AttentionConfig, VideoSparseAttentionConfig


def test_cute_vsa_backend_preserves_sparse_backend_contract() -> None:
    attention = VSACuTeDSLAttention(
        num_heads=4,
        head_dim=128,
    )

    assert isinstance(attention, CuTeDSLAttention)
    assert attention.preferred_layout == AttentionTensorLayout.NHD
    assert not attention.support_lse()
    with pytest.raises(NotImplementedError, match="VSA does not support LSE"):
        attention.forward_with_lse(torch.empty(0), torch.empty(0), torch.empty(0))


def _make_vsa_metadata(*, sparsity: float = 0.0):
    return VSAMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=(5, 4, 4),
        patch_size=(1, 1, 1),
        vsa_sparsity=sparsity,
        device=torch.device("cpu"),
    )


def test_vsa_trtllm_overrides_only_forward_around_the_core() -> None:
    assert "forward" in VSATrtllmAttention.__dict__
    for name in (
        "block_sparse_attn_predict",
        "sparse_predict",
        "sparse_post_process",
        "_enable_sparse_workflow",
    ):
        assert name not in VSATrtllmAttention.__dict__


def _capture_wrapper_forward(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace the VisualGen wrapper forward with a recorder returning the fine input."""

    captured = {}

    def _forward(
        self,
        q,
        k,
        v,
        batch_size,
        seq_len,
        attention_mask=PredefinedAttentionMask.FULL,
        seq_len_kv=None,
        sparse_backend_args=None,
        **kwargs,
    ):
        captured.update(
            q=q,
            k=k,
            v=v,
            batch_size=batch_size,
            seq_len=seq_len,
            attention_mask=attention_mask,
            seq_len_kv=seq_len_kv,
            sparse_backend_args=sparse_backend_args,
            kwargs=kwargs,
        )
        return q.reshape(batch_size, seq_len, -1)

    monkeypatch.setattr(TrtllmAttention, "forward", _forward)
    return captured


def test_vsa_backends_share_one_predictor_implementation() -> None:
    trtllm_attention = object.__new__(VSATrtllmAttention)
    cute_attention = object.__new__(VSACuTeDSLAttention)
    trtllm_attention.predictor = VSAPredictor(num_heads=1)
    cute_attention.predictor = VSAPredictor(num_heads=1)

    assert type(trtllm_attention.predictor) is type(cute_attention.predictor) is VSAPredictor
    assert set(vsa_backend.__all__) >= {"VSATrtllmAttention", "VSACuTeDSLAttention"}


def test_vsa_predictor_produces_sorted_block_inputs_and_effective_tiled_qkv() -> None:
    predictor = VSAPredictor(num_heads=1)
    metadata = _make_vsa_metadata()
    q = torch.randn(1, 80, 1, 8)

    inputs = predictor.predict(
        q,
        q,
        q,
        batch_size=1,
        seq_len=80,
        seq_len_kv=80,
        attention_mask=PredefinedAttentionMask.FULL,
        gate_compress=torch.zeros_like(q),
        gate_fine=None,
        use_sparse_fine=True,
        produce_block_sparse_inputs=True,
        metadata=metadata,
    )

    assert isinstance(inputs, VSAForwardInputs)
    assert inputs.q.shape == inputs.k.shape == inputs.v.shape == (1, 128, 1, 8)
    assert inputs.seq_len == 128
    block_sparse_inputs = inputs.block_sparse_inputs
    assert isinstance(block_sparse_inputs, BlockSparseForwardInputs)
    assert block_sparse_inputs.block_indptr.tolist() == [[[0, 2, 4]]]
    assert block_sparse_inputs.block_indices.tolist() == [0, 1, 0, 1]
    assert block_sparse_inputs.kv_valid_bits.dtype == torch.uint32
    assert block_sparse_inputs.kv_valid_bits.tolist() == [[0xFFFFFFFF, 0xFFFFFFFF, 0xFFFF, 0]]


def test_vsa_predictor_dense_fallback_keeps_compact_qkv_and_no_block_inputs() -> None:
    predictor = VSAPredictor(num_heads=1)
    metadata = _make_vsa_metadata(sparsity=0.5)
    q = torch.randn(1, 80, 1, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    inputs = predictor.predict(
        q,
        k,
        v,
        batch_size=1,
        seq_len=80,
        seq_len_kv=80,
        attention_mask=PredefinedAttentionMask.FULL,
        gate_compress=torch.zeros_like(q),
        gate_fine=None,
        use_sparse_fine=False,
        produce_block_sparse_inputs=False,
        metadata=metadata,
    )

    assert inputs.q is q
    assert inputs.k is k
    assert inputs.v is v
    assert inputs.seq_len == 80
    assert inputs.block_sparse_inputs is None
    assert not inputs.post_context.fine_is_tiled


def test_vsa_shared_post_process_restores_shape_and_applies_gates() -> None:
    predictor = VSAPredictor(num_heads=1)
    metadata = _make_vsa_metadata(sparsity=0.5)
    q = torch.randn(1, 80, 1, 8)
    gate_compress = torch.full_like(q, 2.0)
    gate_fine = torch.full_like(q, 0.5)
    inputs = predictor.predict(
        q,
        q,
        q,
        batch_size=1,
        seq_len=80,
        seq_len_kv=80,
        attention_mask=PredefinedAttentionMask.FULL,
        gate_compress=gate_compress,
        gate_fine=gate_fine,
        use_sparse_fine=False,
        produce_block_sparse_inputs=False,
        metadata=metadata,
    )
    fine_output = torch.randn_like(q)

    output = vsa_backend.vsa_post_process(fine_output, inputs)

    coarse_per_token = inputs.post_context.coarse_output.index_select(
        1, metadata.untile_idx // VSA_BLOCK_SIZE
    )
    expected = 2.0 * coarse_per_token + 0.5 * fine_output
    assert inputs.post_context.coarse_output.shape == (1, metadata.num_cubes, 1, 8)
    assert output.shape == q.shape
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("backend", ["CUTEDSL", "TRTLLM"])
def test_factory_composes_vsa_with_attention_backend(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    class _Backend:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    backend_name = "VSACuTeDSLAttention" if backend == "CUTEDSL" else "VSATrtllmAttention"
    monkeypatch.setattr(vsa_backend, backend_name, _Backend)
    sparse_config = VideoSparseAttentionConfig(vsa_sparsity=0.9)
    attention = create_attention(
        backend=backend,
        layer_idx=0,
        num_heads=8,
        head_dim=128,
        attention_config=AttentionConfig(
            backend=backend,
            sparse_attention_config=sparse_config,
        ),
        attention_metadata_state=(
            create_attention_metadata_state() if backend == "TRTLLM" else None
        ),
    )

    assert isinstance(attention, _Backend)
    assert "sparse_params" not in attention.kwargs


def test_factory_preserves_local_vanilla_fallback_for_vsa() -> None:
    attention = create_attention(
        backend="VANILLA",
        layer_idx=0,
        num_heads=8,
        head_dim=128,
        attention_config=AttentionConfig(
            backend="CUTEDSL",
            sparse_attention_config=VideoSparseAttentionConfig(vsa_sparsity=0.9),
        ),
    )

    assert isinstance(attention, VanillaAttention)


def _make_dense_fallback_vsa_attention() -> VSATrtllmAttention:
    attention = object.__new__(VSATrtllmAttention)
    attention.predictor = VSAPredictor(num_heads=1)
    attention._fmha_manager = SimpleNamespace(fmha_libs=[])
    attention.quant_attention_config = None
    return attention


def test_trtllm_vsa_dense_fallback_runs_compact_inputs_through_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_wrapper_forward(monkeypatch)
    attention = _make_dense_fallback_vsa_attention()
    q = torch.randn(1, 80, 1, 8)

    with set_vsa_forward_context(_make_vsa_metadata(sparsity=0.5)):
        output = attention.forward(
            q,
            q,
            q,
            batch_size=1,
            seq_len=80,
            gate_compress=torch.zeros_like(q),
        )

    assert captured["q"] is q
    assert captured["k"] is q and captured["v"] is q
    assert (captured["batch_size"], captured["seq_len"], captured["seq_len_kv"]) == (1, 80, 80)
    assert captured["sparse_backend_args"] is None
    assert output.shape == (1, 80, 8)


def test_trtllm_vsa_hands_predicted_routes_to_core_via_sparse_backend_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_wrapper_forward(monkeypatch)
    attention = _make_dense_fallback_vsa_attention()
    attention._fmha_manager = SimpleNamespace(
        fmha_libs=[object.__new__(vsa_backend.PrimsTSBlockSparseFmha)]
    )
    monkeypatch.setattr(vsa_backend, "_get_unsupported_primts_reason", lambda *args: None)
    q = torch.randn(1, 80, 1, 8)

    with set_vsa_forward_context(_make_vsa_metadata()):
        output = attention.forward(
            q,
            q,
            q,
            batch_size=1,
            seq_len=80,
            gate_compress=torch.zeros_like(q),
        )

    assert captured["q"].shape == captured["k"].shape == captured["v"].shape == (1, 128, 1, 8)
    assert (captured["seq_len"], captured["seq_len_kv"]) == (128, 128)
    block_sparse_inputs = captured["sparse_backend_args"].block_sparse_inputs
    assert isinstance(block_sparse_inputs, BlockSparseForwardInputs)
    assert block_sparse_inputs.kv_valid_bits is not None
    assert output.shape == (1, 80, 8)


def test_trtllm_vsa_accepts_packed_qkv_through_shared_predictor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_wrapper_forward(monkeypatch)
    attention = _make_dense_fallback_vsa_attention()
    qkv = tuple(torch.randn(1, 80, 1, 8) for _ in range(3))

    with set_vsa_forward_context(_make_vsa_metadata(sparsity=0.5)):
        attention.forward(
            torch.stack(qkv, dim=2),
            None,
            None,
            batch_size=1,
            seq_len=80,
            gate_compress=torch.zeros_like(qkv[0]),
        )

    for actual, expected in zip((captured["q"], captured["k"], captured["v"]), qkv):
        torch.testing.assert_close(actual, expected)


def test_trtllm_vsa_consumes_gates_and_forwards_only_timestep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_wrapper_forward(monkeypatch)
    attention = _make_dense_fallback_vsa_attention()
    q = torch.randn(1, 80, 1, 8)
    gate_compress = torch.full_like(q, 2.0)
    gate_fine = torch.full_like(q, 0.5)
    timestep = torch.tensor([12])

    with set_vsa_forward_context(_make_vsa_metadata(sparsity=0.5)):
        output = attention.forward(
            q,
            q,
            q,
            batch_size=1,
            seq_len=80,
            gate_compress=gate_compress,
            gate_fine=gate_fine,
            timestep=timestep,
        )

    assert captured["kwargs"] == {"timestep": timestep}
    assert output.shape == (1, 80, 8)
    assert torch.isfinite(output).all()


def test_cutedsl_vsa_rejects_unexpected_forward_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = object.__new__(VSACuTeDSLAttention)
    attention.predictor = VSAPredictor(num_heads=1)
    q = torch.randn(1, 80, 1, 8)
    monkeypatch.setattr(vsa_backend, "_vsa_import_error", RuntimeError("disabled for test"))

    with set_vsa_forward_context(_make_vsa_metadata(sparsity=0.5)):
        with pytest.raises(TypeError, match="gate_fnne"):
            attention.forward(
                q,
                q,
                q,
                gate_compress=torch.zeros_like(q),
                gate_fnne=torch.zeros_like(q),
            )


def _make_config(
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    backend: str,
    vsa_sparsity: "float | None" = None,
) -> DiffusionModelConfig:
    """Minimal DiffusionModelConfig for one Attention module."""
    pretrained_config = SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        eps=1e-6,
    )
    sparse_attention_config = (
        VideoSparseAttentionConfig(vsa_sparsity=vsa_sparsity) if vsa_sparsity is not None else None
    )
    config = DiffusionModelConfig(
        pretrained_config=pretrained_config,
        attention=AttentionConfig(backend=backend, sparse_attention_config=sparse_attention_config),
        skip_create_weights_in_init=False,
    )
    config.attention_metadata_state = (
        create_attention_metadata_state() if backend == "TRTLLM" else None
    )
    return config


@pytest.mark.parametrize("backend", ["CUTEDSL", "TRTLLM"])
@pytest.mark.parametrize(
    ("is_self_attention", "expected_backend"),
    [(False, "VANILLA"), (True, None)],
    ids=["cross", "self"],
)
def test_vsa_separate_qkv_dispatches_by_attention_role(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    is_self_attention: bool,
    expected_backend: str | None,
) -> None:
    monkeypatch.setattr(
        attention_module,
        "create_attention",
        lambda *, backend, **kwargs: SimpleNamespace(backend=backend, kwargs=kwargs),
    )
    cfg = _make_config(
        hidden_size=64,
        num_heads=4,
        head_dim=16,
        backend=backend,
        vsa_sparsity=0.5,
    )
    attention = Attention(
        64,
        4,
        qkv_mode=QKVMode.SEPARATE_QKV,
        config=cfg,
        separate_qkv_is_self_attention=is_self_attention,
    )

    assert attention.attn_backend == (expected_backend or backend)


def test_plain_trtllm_separate_qkv_self_attention_keeps_vanilla_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        attention_module,
        "create_attention",
        lambda *, backend, **kwargs: SimpleNamespace(backend=backend, kwargs=kwargs),
    )
    cfg = _make_config(
        hidden_size=64,
        num_heads=4,
        head_dim=16,
        backend="TRTLLM",
    )

    attention = Attention(
        64,
        4,
        qkv_mode=QKVMode.SEPARATE_QKV,
        config=cfg,
        separate_qkv_is_self_attention=True,
    )

    assert attention.attn_backend == "VANILLA"


def test_vsa_with_attn2d_raises():
    """VSA + Attention2D must error at construction (VSA needs the full sequence per rank)."""
    pretrained_config = SimpleNamespace(
        hidden_size=64,
        num_attention_heads=4,
        attention_head_dim=16,
        eps=1e-6,
    )
    cfg = DiffusionModelConfig(
        pretrained_config=pretrained_config,
        attention=AttentionConfig(
            backend="CUTEDSL",
            sparse_attention_config=VideoSparseAttentionConfig(vsa_sparsity=0.0),
        ),
        skip_create_weights_in_init=False,
    )
    cfg.visual_gen_mapping = SimpleNamespace(
        ring_size=1,
        ring_group=None,
        ulysses_size=1,
        ulysses_group=None,
        attn2d_row_size=2,
        attn2d_col_size=2,
        attn2d_row_group=None,
        attn2d_col_group=None,
        cp_size=4,
    )
    with pytest.raises(ValueError, match="incompatible with context parallelism"):
        Attention(64, 4, qkv_mode=QKVMode.FUSE_QKV, config=cfg)


def test_vsa_metadata_builder_reuses_shape_tensors_with_live_step_policy() -> None:
    builder = VSAMetadataBuilder()
    build_args = {
        "raw_latent_shape": (9, 9, 9),
        "patch_size": (1, 1, 1),
        "device": torch.device("cpu"),
    }

    first = builder.build(current_timestep=3, vsa_sparsity=0.25, **build_args)
    second = builder.build(current_timestep=4, vsa_sparsity=0.75, **build_args)

    assert first is not second
    assert (first.current_timestep, first.vsa_sparsity) == (3, 0.25)
    assert (second.current_timestep, second.vsa_sparsity) == (4, 0.75)
    assert second.tile_source_index is first.tile_source_index
    assert first.num_cubes == 27

    builder.clear()

    rebuilt = builder.build(current_timestep=5, vsa_sparsity=0.5, **build_args)
    assert rebuilt.tile_source_index is not first.tile_source_index


def test_vsa_metadata_exposes_tile_source_index_and_packed_kv_words() -> None:
    metadata = _make_vsa_metadata()

    source = metadata.tile_source_index
    assert source.shape == (metadata.padded_seq_length,)
    assert int((source >= 0).sum()) == 80
    assert torch.equal(source[metadata.untile_idx], torch.arange(80))
    assert metadata.kv_valid_words.dtype == torch.uint32
    assert metadata.kv_valid_words.tolist() == [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFF, 0]


def test_vsa_graph_stable_caches_bound_shape_profiles() -> None:
    builder = VSAMetadataBuilder(max_cached_shapes=1)
    build_args = {
        "current_timestep": 0,
        "patch_size": (1, 1, 1),
        "vsa_sparsity": 0.5,
        "device": torch.device("cpu"),
    }
    builder.build(raw_latent_shape=(4, 4, 4), **build_args)
    with pytest.raises(RuntimeError, match="metadata cache reached its 1-shape limit"):
        builder.build(raw_latent_shape=(8, 4, 4), **build_args)

    route_builder = VSAPredictor(num_heads=1, max_cached_shapes=1)._route_builder
    kv_valid_words = torch.ones((1,), dtype=torch.uint32)
    route_builder.from_selected_blocks(torch.zeros((1, 1, 1, 1), dtype=torch.int32), kv_valid_words)
    with pytest.raises(RuntimeError, match="route cache reached its 1-shape limit"):
        route_builder.from_selected_blocks(
            torch.zeros((1, 1, 2, 1), dtype=torch.int32), kv_valid_words
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="VSA needs CUDA")
@pytest.mark.parametrize(
    "latent_shape",
    [
        (8, 8, 8),
        (9, 9, 9),
        (21, 45, 80),
    ],
    ids=["clean_8x8x8", "ragged_9x9x9", "wan720p_21x45x80"],
)
def test_vsa_tile_untile_roundtrip(latent_shape):
    """Tiling then untiling must reproduce the input, and pooled cubes must be token means."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)

    B, H, D = 2, 4, 32
    seq_len = latent_shape[0] * latent_shape[1] * latent_shape[2]

    builder = VSAMetadataBuilder()
    meta = builder.build(
        current_timestep=0,
        raw_latent_shape=latent_shape,
        patch_size=(1, 1, 1),
        vsa_sparsity=0.0,
        device=device,
    )

    x = torch.randn(B, seq_len, H, D, device=device, dtype=dtype)

    x_tiled, x_pooled = tile_and_pool_cubes(
        x,
        meta.tile_source_index,
        meta.variable_block_sizes,
        cube_size=VSA_BLOCK_SIZE,
    )

    pad_mask = meta.tile_source_index < 0
    if pad_mask.any():
        assert x_tiled[:, pad_mask, :, :].abs().max().item() == 0.0, (
            "tiling must zero-fill padded positions"
        )

    x_roundtrip = x_tiled.index_select(1, meta.untile_idx)

    assert x_roundtrip.shape == x.shape, (
        f"shape mismatch after tile/untile: {x_roundtrip.shape} vs {x.shape}"
    )
    assert torch.equal(x_roundtrip, x), (
        f"tile/untile round-trip is not lossless for latent_shape={latent_shape}: "
        f"max_diff={(x_roundtrip - x).abs().max().item():.3e}"
    )

    expected_pooled = x_tiled.view(B, meta.num_cubes, VSA_BLOCK_SIZE, H, D).float().sum(dim=2)
    expected_pooled = expected_pooled / meta.variable_block_sizes.view(1, -1, 1, 1).float()
    torch.testing.assert_close(x_pooled, expected_pooled.to(dtype), rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="VSA needs CUDA")
def test_vsa_predictor_kernels_match_torch_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Triton path and the PyTorch fallback must produce the same envelope and output."""
    device = torch.device("cuda")
    metadata = VSAMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=(9, 9, 9),
        patch_size=(1, 1, 1),
        vsa_sparsity=0.75,
        device=device,
    )
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 729, 4, 32, device=device) for _ in range(3))
    gate = torch.randn_like(q)
    fine_output = torch.randn(2, metadata.padded_seq_length, 4, 32, device=device)
    call_args = {
        "batch_size": 2,
        "seq_len": 729,
        "seq_len_kv": 729,
        "attention_mask": PredefinedAttentionMask.FULL,
        "gate_compress": gate,
        "gate_fine": gate,
        "use_sparse_fine": True,
        "produce_block_sparse_inputs": True,
        "metadata": metadata,
    }

    with_kernels = VSAPredictor(num_heads=4).predict(q, k, v, **call_args)
    output_with_kernels = vsa_backend.vsa_post_process(fine_output, with_kernels)

    monkeypatch.setattr(
        vsa_predictor, "tile_and_pool_cubes", vsa_kernels._tile_and_pool_cubes_torch
    )
    monkeypatch.setattr(vsa_predictor, "sort_last_dim", vsa_kernels._sort_last_dim_torch)
    monkeypatch.setattr(vsa_predictor, "blend_coarse_fine", vsa_kernels._blend_coarse_fine_torch)
    fallback = VSAPredictor(num_heads=4).predict(q, k, v, **call_args)
    output_fallback = vsa_backend.vsa_post_process(fine_output, fallback)

    for name in ("q", "k", "v"):
        assert torch.equal(getattr(with_kernels, name), getattr(fallback, name)), name
    assert torch.equal(
        with_kernels.block_sparse_inputs.block_indices, fallback.block_sparse_inputs.block_indices
    )
    torch.testing.assert_close(
        with_kernels.post_context.coarse_output, fallback.post_context.coarse_output
    )
    torch.testing.assert_close(output_with_kernels, output_fallback)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="VSA needs CUDA")
def test_vsa_predictor_replays_inside_cuda_graph() -> None:
    device = torch.device("cuda")
    metadata = VSAMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=(8, 8, 8),
        patch_size=(1, 1, 1),
        vsa_sparsity=0.5,
        device=device,
    )
    predictor = VSAPredictor(num_heads=2)
    torch.manual_seed(0)
    q = torch.randn(1, 512, 2, 16, device=device, dtype=torch.bfloat16)
    gate = torch.randn_like(q)

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        inputs = predictor.predict(
            q,
            q,
            q,
            batch_size=1,
            seq_len=512,
            seq_len_kv=512,
            attention_mask=PredefinedAttentionMask.FULL,
            gate_compress=gate,
            gate_fine=None,
            use_sparse_fine=True,
            produce_block_sparse_inputs=True,
            metadata=metadata,
        )
        return vsa_backend.vsa_post_process(
            inputs.q, inputs
        ), inputs.block_sparse_inputs.block_indices

    eager_output, eager_routes = run()
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        run()
    torch.cuda.current_stream().wait_stream(side_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output, graph_routes = run()
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(graph_output, eager_output)
    assert torch.equal(graph_routes, eager_routes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel test needs CUDA")
def test_cute_kernel_matches_dense_at_full_topk():
    """CuTe block-sparse kernel matches dense SDPA when every cube is selected."""
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.video_sparse_attention import (
        CUTE_AVAILABLE,
        block_sparse_attn_from_indices_cute,
        is_cute_supported,
    )

    if not CUTE_AVAILABLE:
        pytest.skip("cuda-bindings or cutlass-dsl not importable")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)

    B, H, num_cubes, D = 1, 4, 4, 128
    block_size = 64
    seq_len = num_cubes * block_size

    q = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)
    k = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)
    v = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)

    if not is_cute_supported(q):
        pytest.skip("CuTe path needs sm_100+ Blackwell (current device unsupported)")

    topk = num_cubes
    q2k_idx = (
        torch.arange(num_cubes, device=device, dtype=torch.int32)
        .view(1, 1, 1, num_cubes)
        .expand(B, H, num_cubes, topk)
        .contiguous()
    )
    q2k_num = torch.full((B, H, num_cubes), topk, dtype=torch.int32, device=device)
    variable_block_sizes = torch.full((num_cubes,), block_size, dtype=torch.int32, device=device)

    out_kernel, _lse = block_sparse_attn_from_indices_cute(
        q, k, v, q2k_idx, q2k_num, variable_block_sizes
    )
    out_ref = F.scaled_dot_product_attention(q, k, v)

    max_diff = (out_kernel - out_ref).abs().max().item()
    mean_diff = (out_kernel - out_ref).abs().mean().item()

    rtol, atol = 1e-2, 1e-2
    assert torch.allclose(out_kernel, out_ref, rtol=rtol, atol=atol), (
        f"CuTe block-sparse kernel deviates from dense SDPA at full top-K: "
        f"max_diff={max_diff:.3e}, mean_diff={mean_diff:.3e} (rtol={rtol}, atol={atol})"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel test needs CUDA")
def test_cute_kernel_matches_ref_with_independent_indices():
    """CuTe kernel: paired Q-blocks (2i, 2i+1) attend to independent KV index lists."""
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.video_sparse_attention import (
        CUTE_AVAILABLE,
        block_sparse_attn_from_indices_cute,
        is_cute_supported,
    )

    if not CUTE_AVAILABLE:
        pytest.skip("cuda-bindings or cutlass-dsl not importable")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(42)

    B, H, num_cubes, D = 2, 4, 16, 128
    block_size = 64
    topk = num_cubes // 2
    seq_len = num_cubes * block_size

    q = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)
    k = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)
    v = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)

    if not is_cute_supported(q):
        pytest.skip("CuTe path needs sm_100+ Blackwell (current device unsupported)")

    q2k_idx = (
        torch.stack(
            [
                torch.randperm(num_cubes, device=device, dtype=torch.int32)[:topk]
                for _ in range(B * H * num_cubes)
            ]
        )
        .view(B, H, num_cubes, topk)
        .contiguous()
    )

    paired = q2k_idx.view(B, H, num_cubes // 2, 2, topk).sort(dim=-1).values
    pair_mismatch = (paired[..., 0, :] != paired[..., 1, :]).sum().item()
    assert pair_mismatch > 0, (
        "Pre-condition failed: random permutations matched across every pair; "
        "re-seed or raise num_cubes."
    )

    q2k_num = torch.full((B, H, num_cubes), topk, dtype=torch.int32, device=device)
    variable_block_sizes = torch.full((num_cubes,), block_size, dtype=torch.int32, device=device)

    attn_mask = torch.full(
        (B, H, seq_len, seq_len), float("-inf"), device=device, dtype=torch.float32
    )
    for b in range(B):
        for h in range(H):
            for q_blk in range(num_cubes):
                for ki in range(topk):
                    k_blk = q2k_idx[b, h, q_blk, ki].item()
                    qs = q_blk * block_size
                    ks = k_blk * block_size
                    attn_mask[b, h, qs : qs + block_size, ks : ks + block_size] = 0.0

    out_kernel, _lse = block_sparse_attn_from_indices_cute(
        q, k, v, q2k_idx, q2k_num, variable_block_sizes
    )

    scale = 1.0 / (D**0.5)
    scores = (q.float() @ k.float().transpose(-2, -1)) * scale
    scores = scores + attn_mask
    probs = torch.softmax(scores, dim=-1)
    out_ref = (probs @ v.float()).to(dtype)

    abs_diff = (out_kernel.float() - out_ref.float()).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    rtol, atol = 1e-2, 1e-2
    assert torch.allclose(out_kernel, out_ref, rtol=rtol, atol=atol), (
        f"CuTe kernel with independent per-Q-block indices deviated from masked fp32 "
        f"reference: max_diff={max_diff:.3e}, mean_diff={mean_diff:.3e} "
        f"(rtol={rtol}, atol={atol}, pair_mismatch={pair_mismatch})"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel test needs CUDA")
def test_cute_kernel_50pct_sparsity_quality_vs_dense():
    """50% sparse CuTe kernel with score-based topk stays close to dense SDPA."""
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.video_sparse_attention import (
        CUTE_AVAILABLE,
        block_sparse_attn_from_indices_cute,
        is_cute_supported,
    )

    if not CUTE_AVAILABLE:
        pytest.skip("cuda-bindings or cutlass-dsl not importable")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)

    batch_size, num_heads, num_cubes, head_dim = 1, 4, 16, 128
    block_size = 64
    topk = num_cubes // 2
    seq_len = num_cubes * block_size

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    if not is_cute_supported(q):
        pytest.skip("CuTe path needs sm_100+ Blackwell (current device unsupported)")

    q_blocks = q.reshape(batch_size, num_heads, num_cubes, block_size, head_dim).mean(dim=3)
    k_blocks = k.reshape(batch_size, num_heads, num_cubes, block_size, head_dim).mean(dim=3)
    block_scores = torch.einsum(
        "bhqd,bhkd->bhqk",
        q_blocks.float(),
        k_blocks.float(),
    ) * (head_dim**-0.5)
    q2k_idx = block_scores.topk(topk, dim=-1).indices.to(torch.int32).contiguous()
    q2k_num = torch.full(
        (batch_size, num_heads, num_cubes),
        topk,
        dtype=torch.int32,
        device=device,
    )
    variable_block_sizes = torch.full(
        (num_cubes,),
        block_size,
        dtype=torch.int32,
        device=device,
    )

    out_sparse, _lse = block_sparse_attn_from_indices_cute(
        q,
        k,
        v,
        q2k_idx,
        q2k_num,
        variable_block_sizes,
    )
    out_dense = F.scaled_dot_product_attention(q, k, v)

    cos_sim = F.cosine_similarity(
        out_sparse.float().reshape(-1),
        out_dense.float().reshape(-1),
        dim=0,
    ).item()
    assert cos_sim >= 0.65, (
        f"50% sparse CuTe kernel deviated too far from dense SDPA: cos_sim={cos_sim:.4f} < 0.65"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel test needs CUDA")
@pytest.mark.parametrize(
    "num_cubes",
    [1, 3, 9],
    ids=["1cube_odd", "3cubes_odd", "9cubes_odd"],
)
def test_cute_kernel_odd_num_cubes_correctness(num_cubes):
    """CuTe kernel supports a final Q block that has no paired neighbor."""
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.video_sparse_attention import (
        CUTE_AVAILABLE,
        block_sparse_attn_from_indices_cute,
        is_cute_supported,
    )

    if not CUTE_AVAILABLE:
        pytest.skip("cuda-bindings or cutlass-dsl not importable")

    assert num_cubes % 2 == 1
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)

    batch_size, num_heads, head_dim = 1, 4, 128
    block_size = 64
    seq_len = num_cubes * block_size
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    if not is_cute_supported(q):
        pytest.skip("CuTe path needs sm_100+ Blackwell (current device unsupported)")

    q2k_idx = (
        torch.arange(num_cubes, device=device, dtype=torch.int32)
        .view(1, 1, 1, num_cubes)
        .expand(batch_size, num_heads, num_cubes, num_cubes)
        .contiguous()
    )
    q2k_num = torch.full(
        (batch_size, num_heads, num_cubes),
        num_cubes,
        dtype=torch.int32,
        device=device,
    )
    variable_block_sizes = torch.full(
        (num_cubes,),
        block_size,
        dtype=torch.int32,
        device=device,
    )

    out_kernel, _lse = block_sparse_attn_from_indices_cute(
        q,
        k,
        v,
        q2k_idx,
        q2k_num,
        variable_block_sizes,
    )
    out_ref = F.scaled_dot_product_attention(q, k, v)

    assert torch.isfinite(out_kernel).all()
    torch.testing.assert_close(out_kernel, out_ref, rtol=1e-2, atol=1e-2)

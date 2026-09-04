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

from tensorrt_llm._torch.attention_backend.block_sparse import BlockSparseForwardInputs
from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask
from tensorrt_llm._torch.attention_backend.sparse.params import SparseRuntimeParams
from tensorrt_llm._torch.visual_gen.attention_backend.cute_dsl import CuTeDSLAttention
from tensorrt_llm._torch.visual_gen.attention_backend.interface import AttentionTensorLayout
from tensorrt_llm._torch.visual_gen.attention_backend.sparse.vsa import backend as vsa_backend
from tensorrt_llm._torch.visual_gen.attention_backend.sparse.vsa.backend import (
    VSACuTeDSLAttention,
    VSATrtllmAttention,
)
from tensorrt_llm._torch.visual_gen.attention_backend.sparse.vsa.metadata import (
    VSAMetadataBuilder,
    set_vsa_forward_context,
)
from tensorrt_llm._torch.visual_gen.attention_backend.sparse.vsa.predictor import (
    VSAForwardInputs,
    VSAPredictor,
    VSAPreprocessor,
)
from tensorrt_llm._torch.visual_gen.attention_backend.trtllm import SparseForwardInputs
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


def test_vsa_trtllm_uses_generic_lifecycle_without_forward_override() -> None:
    assert "forward" not in VSATrtllmAttention.__dict__
    assert "block_sparse_attn_predict" in VSATrtllmAttention.__dict__
    assert "sparse_predict" not in VSATrtllmAttention.__dict__
    assert "sparse_post_process" in VSATrtllmAttention.__dict__
    assert "_enable_sparse_workflow" not in VSATrtllmAttention.__dict__


def test_vsa_backends_share_one_predictor_implementation() -> None:
    trtllm_attention = object.__new__(VSATrtllmAttention)
    cute_attention = object.__new__(VSACuTeDSLAttention)
    trtllm_attention.predictor = VSAPredictor(num_heads=1)
    cute_attention.predictor = VSAPredictor(num_heads=1)

    assert type(trtllm_attention.predictor) is type(cute_attention.predictor) is VSAPredictor
    assert set(vsa_backend.__all__) >= {"VSATrtllmAttention", "VSACuTeDSLAttention"}


def test_vsa_trtllm_layers_share_model_scoped_predictor() -> None:
    attention_metadata_state = create_attention_metadata_state()
    first = VSATrtllmAttention(
        layer_idx=0,
        num_heads=2,
        num_kv_heads=2,
        head_dim=128,
        attention_metadata_state=attention_metadata_state,
    )
    second = VSATrtllmAttention(
        layer_idx=1,
        num_heads=2,
        num_kv_heads=2,
        head_dim=128,
        attention_metadata_state=attention_metadata_state,
    )

    assert first.predictor is second.predictor


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
        forward_kwargs={},
        metadata=metadata,
    )

    assert isinstance(inputs, VSAForwardInputs)
    assert inputs.q.shape == inputs.k.shape == inputs.v.shape == (1, 128, 1, 8)
    assert inputs.seq_len == inputs.seq_len_kv == 128
    block_sparse_inputs = inputs.sparse_runtime_params.block_sparse_inputs
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
        forward_kwargs={},
        metadata=metadata,
    )

    assert inputs.q is q
    assert inputs.k is k
    assert inputs.v is v
    assert inputs.seq_len == inputs.seq_len_kv == 80
    assert isinstance(inputs.sparse_runtime_params, SparseRuntimeParams)
    assert inputs.sparse_runtime_params.block_sparse_inputs is None
    assert inputs.post_context.untile_idx is None


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
        forward_kwargs={},
        metadata=metadata,
    )
    fine_output = torch.randn_like(q)

    output = vsa_backend.vsa_post_process(fine_output, inputs)

    expected = 2.0 * inputs.post_context.coarse_output + 0.5 * fine_output
    assert output.shape == q.shape
    torch.testing.assert_close(output, expected)


def test_vsa_post_process_rejects_non_vsa_inputs() -> None:
    attention = object.__new__(VSATrtllmAttention)
    output = torch.randn(1, 4, 8)
    sparse_inputs = SparseForwardInputs(
        q=torch.randn(1, 4, 1, 8),
        k=None,
        v=None,
        batch_size=1,
        seq_len=4,
        seq_len_kv=4,
        attention_mask=PredefinedAttentionMask.FULL,
        sparse_runtime_params=SparseRuntimeParams(),
    )

    with pytest.raises(TypeError, match="VSAForwardInputs"):
        attention.sparse_post_process(output, sparse_inputs)


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


def test_trtllm_vsa_dense_fallback_predicts_compact_inputs() -> None:
    attention = object.__new__(VSATrtllmAttention)
    attention.predictor = VSAPredictor(num_heads=1)
    attention._fmha_manager = SimpleNamespace(fmha_libs=[])
    attention.quant_attention_config = None
    q = torch.randn(1, 80, 1, 8)

    with set_vsa_forward_context(_make_vsa_metadata(sparsity=0.5)):
        inputs = attention.block_sparse_attn_predict(
            q,
            q,
            q,
            batch_size=1,
            seq_len=80,
            seq_len_kv=80,
            attention_mask=PredefinedAttentionMask.FULL,
            forward_kwargs={"gate_compress": torch.zeros_like(q)},
        )

    assert inputs.q is q
    assert isinstance(inputs.sparse_runtime_params, SparseRuntimeParams)
    assert inputs.sparse_runtime_params.block_sparse_inputs is None
    assert inputs.seq_len == 80


def test_trtllm_vsa_accepts_packed_qkv_through_shared_predictor() -> None:
    attention = object.__new__(VSATrtllmAttention)
    attention.predictor = VSAPredictor(num_heads=1)
    attention._fmha_manager = SimpleNamespace(fmha_libs=[])
    attention.quant_attention_config = None
    qkv = tuple(torch.randn(1, 80, 1, 8) for _ in range(3))

    with set_vsa_forward_context(_make_vsa_metadata(sparsity=0.5)):
        inputs = attention.block_sparse_attn_predict(
            torch.stack(qkv, dim=2),
            None,
            None,
            batch_size=1,
            seq_len=80,
            seq_len_kv=80,
            attention_mask=PredefinedAttentionMask.FULL,
            forward_kwargs={"gate_compress": torch.zeros_like(qkv[0])},
        )

    for actual, expected in zip((inputs.q, inputs.k, inputs.v), qkv):
        torch.testing.assert_close(actual, expected)


def test_trtllm_vsa_consumes_gates_and_forwards_only_timestep() -> None:
    attention = object.__new__(VSATrtllmAttention)
    attention.predictor = VSAPredictor(num_heads=1)
    attention._fmha_manager = SimpleNamespace(fmha_libs=[])
    attention.quant_attention_config = None
    q = torch.randn(1, 80, 1, 8)
    gate_compress = torch.zeros_like(q)
    gate_fine = torch.ones_like(q)
    timestep = torch.tensor([12])

    with set_vsa_forward_context(_make_vsa_metadata(sparsity=0.5)):
        inputs = attention.block_sparse_attn_predict(
            q,
            q,
            q,
            batch_size=1,
            seq_len=80,
            seq_len_kv=80,
            attention_mask=PredefinedAttentionMask.FULL,
            forward_kwargs={
                "gate_compress": gate_compress,
                "gate_fine": gate_fine,
                "timestep": timestep,
            },
        )

    assert dict(inputs.forward_kwargs) == {"timestep": timestep}
    assert inputs.post_context.gate_compress is gate_compress
    assert inputs.post_context.gate_fine is gate_fine


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
    assert second.gather_idx is first.gather_idx
    assert first.num_cubes == 27

    builder.clear()

    rebuilt = builder.build(current_timestep=5, vsa_sparsity=0.5, **build_args)
    assert rebuilt.gather_idx is not first.gather_idx


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
    kv_valid_bits = torch.ones((1, 1), dtype=torch.uint32)
    route_builder.from_selected_blocks(torch.zeros((1, 1, 1, 1), dtype=torch.int32), kv_valid_bits)
    with pytest.raises(RuntimeError, match="route cache reached its 1-shape limit"):
        route_builder.from_selected_blocks(
            torch.zeros((1, 1, 2, 1), dtype=torch.int32), kv_valid_bits
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
    """VSAPreprocessor.tile then .untile must losslessly reproduce the input."""
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

    x_tiled = VSAPreprocessor.tile(
        x,
        meta.non_pad_index,
        meta.gather_idx,
        meta.padded_seq_length,
    )

    pad_mask = torch.ones(meta.padded_seq_length, dtype=torch.bool, device=device)
    pad_mask[meta.non_pad_index] = False
    if pad_mask.any():
        assert x_tiled[:, pad_mask, :, :].abs().max().item() == 0.0, (
            "tile() must zero-fill padded positions"
        )

    x_roundtrip = VSAPreprocessor.untile(
        x_tiled,
        meta.untile_idx,
    )

    assert x_roundtrip.shape == x.shape, (
        f"shape mismatch after tile/untile: {x_roundtrip.shape} vs {x.shape}"
    )
    assert torch.equal(x_roundtrip, x), (
        f"tile/untile round-trip is not lossless for latent_shape={latent_shape}: "
        f"max_diff={(x_roundtrip - x).abs().max().item():.3e}"
    )


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

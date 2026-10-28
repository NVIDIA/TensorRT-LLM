# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""qwen4_exp derives its KV storage from the engine's `kv_cache_quant_algo`.

`None` keeps the checkpoint's bf16 K/V with the attention op's `quant_mode` 0 and no KV
scales; `FP8` stores K/V as e4m3 under `QM_FP8_KV_CACHE` with unit fp32 `[1]` scales; `NVFP4`
stores e2m1 pairs as uint8 half a head wide under `QM_NVFP4_KV_CACHE` with unit fp32 `[3]`
scales; any other algorithm is refused at construction. The core is built on the meta device from the
checkpoint's text and vision config values, so nothing here touches a device; the
`post_load_weights` cases stub out everything that reads weight values.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

__extra_import_path__ = ["../../.."]  # noqa: F841 -- repo's file-scoped import hook

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1 import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch.configs import Qwen4ExpTextConfig, Qwen4ExpVisionConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.llmapi import QSASparseAttentionConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

NUM_LAYERS = 48

# Qwen3.8-Flash-Next's text config.
_TEXT = dict(
    architectures=["Qwen4ExpForCausalLM"],
    attention_bias=False,
    attention_dropout=0.0,
    bos_token_id=248044,
    dtype="bfloat16",
    eos_token_id=248044,
    full_attention_interval=4,
    hc_count=4,
    hc_lowrank=320,
    head_dim=256,
    heads_per_ngram=8,
    hidden_act="silu",
    hidden_size=2560,
    indexer_budget=2048,
    indexer_compress_ratio=4,
    indexer_head_dim=128,
    indexer_kv_heads=1,
    indexer_n_heads=4,
    initializer_range=0.02,
    layer_types=["full_attention" if i % 4 == 3 else "linear_attention" for i in range(NUM_LAYERS)],
    linear_conv_kernel_dim=4,
    linear_key_head_dim=128,
    linear_num_key_heads=16,
    linear_num_value_heads=48,
    linear_value_head_dim=128,
    make_ngram_vocab_size_divisible_by=128,
    mamba_ssm_dtype="float32",
    max_position_embeddings=262144,
    moe_intermediate_size=640,
    mtp=dict(
        hybrid=True,
        layer_types=["full_attention"],
        mtp_use_hidden_state_from_layer=None,
        num_hidden_layers=1,
        rope_theta=10000000,
    ),
    mtp_num_hidden_layers=1,
    mtp_use_dedicated_embeddings=False,
    ngram_size=3,
    ngram_vocab_size_base=20000000,
    norm_topk_prob=True,
    num_attention_heads=24,
    num_experts=512,
    num_experts_per_tok=10,
    num_hidden_layers=NUM_LAYERS,
    num_key_value_heads=2,
    number_of_conv_states=3,
    output_gate_type="sigmoid",
    output_router_logits=False,
    pad_token_id=None,
    partial_rotary_factor=0.25,
    ple_conv_kernel_size=4,
    ple_embed_dim=2560,
    ple_layer_ids=[2],
    rms_norm_eps=1e-06,
    rope_parameters=dict(
        mrope_interleaved=True,
        mrope_section=[11, 11, 10],
        partial_rotary_factor=0.25,
        rope_theta=10000000,
        rope_type="default",
    ),
    router_aux_loss_coef=0.001,
    seed=1234,
    shared_expert_intermediate_size=640,
    split_ngram_parts=128,
    tie_word_embeddings=False,
    use_cache=True,
    vocab_size=248320,
)
_VISION = dict(
    deepstack_visual_indexes=[],
    depth=27,
    dtype="bfloat16",
    hidden_act="gelu_pytorch_tanh",
    hidden_size=1152,
    in_channels=3,
    intermediate_size=4304,
    num_heads=16,
    num_position_embeddings=2304,
    out_hidden_size=2560,
    patch_size=16,
    spatial_merge_size=2,
    temporal_patch_size=2,
)


def _core(kv_cache_quant_algo):
    """The core of a Qwen3.8-Flash-Next engine whose resolved `kv_cache_quant_algo` is
    `kv_cache_quant_algo`, with every parameter on the meta device."""
    model_config = ModelConfig(
        pretrained_config=Qwen4ExpTextConfig(**_TEXT),
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        quant_config=QuantConfig(
            quant_algo=QuantAlgo.MIXED_PRECISION, kv_cache_quant_algo=kv_cache_quant_algo
        ),
        moe_backend="CUTLASS",
        moe_backend_requested="TRTLLM",
        sparse_attention_config=QSASparseAttentionConfig(),
    )
    composite = SimpleNamespace(
        vision_config=Qwen4ExpVisionConfig(**_VISION), image_token_id=248056, video_token_id=248057
    )
    # The core asserts the device it runs on; the weights it declares stay on meta.
    with (
        mock.patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)),
        torch.device("meta"),
    ):
        return M.Qwen4ExpTp1ModelingV2(model_config, composite)


def _load(core, monkeypatch) -> None:
    """`post_load_weights` on a core with the weights it reads replaced by CPU stubs.

    Everything that reads weight values (the expert scalars, the n-gram table's driver
    residency, the per-step targets) is replaced; what is left is the derivation under test
    and the shape checks around it. The derived scales land on `embed`'s device, here CPU,
    so their values can be read.
    """
    table_rows = core.ngram_vocab_divisor
    stubs = dict(
        embed=torch.empty(1),
        ple_table=torch.empty(table_rows, 1),
        ple_offs=torch.zeros(1, dtype=torch.int64),
        ple_vocab=torch.full((1,), table_rows, dtype=torch.int64),
    )
    for key, value in stubs.items():
        core.w[key] = torch.nn.Parameter(value, requires_grad=False)
    monkeypatch.setattr(M, "_moe_layer_scalars", lambda w, i, backend: {})
    for target in ("PrefillTarget", "DecodeTarget", "VisionTower"):
        monkeypatch.setattr(M, target, lambda core: None)
    monkeypatch.setattr(core, "_report_ple_residency", lambda: None)
    core.post_load_weights()


def test_unquantized_kv_cache_keeps_the_checkpoints_bf16(monkeypatch) -> None:
    core = _core(None)
    assert core.quant_mode == 0
    assert core.kv_pool_dtype == torch.bfloat16
    geometry = core._pool_geometry(32)
    assert geometry.kv_dtype == torch.bfloat16
    assert geometry.kv_width == core.head_dim
    assert core.kv_scale_orig_quant is None and core.kv_scale_quant_orig is None
    _load(core, monkeypatch)
    assert core.kv_scale_orig_quant is None and core.kv_scale_quant_orig is None


def test_fp8_kv_cache_stores_e4m3_with_unit_fp32_single_element_scales(monkeypatch) -> None:
    """The generation kernels read `kvCacheScale[0]` and the QSA store and sparse GQA take
    no scale, so the scale is a real tensor holding exactly 1.0: the dense and the sparse
    path then agree on what a cache slot holds."""
    core = _core(QuantAlgo.FP8)
    assert core.quant_mode == M.QM_FP8_KV_CACHE
    assert core.kv_pool_dtype == torch.float8_e4m3fn
    # The pool contract holds every manager's K/V pool to this dtype; the index and position
    # planes keep the activation dtype.
    geometry = core._pool_geometry(32)
    assert geometry.kv_dtype == torch.float8_e4m3fn
    assert geometry.kv_width == core.head_dim
    assert geometry.dtype == torch.bfloat16
    _load(core, monkeypatch)
    for scale in (core.kv_scale_orig_quant, core.kv_scale_quant_orig):
        assert scale is not None
        assert scale.dtype == torch.float32 and tuple(scale.shape) == (1,)
        assert torch.equal(scale, torch.ones(1))


def test_nvfp4_kv_cache_stores_e2m1_pairs_with_unit_fp32_scale_triples(monkeypatch) -> None:
    """The attention op reads the q, K and V second-level scales out of one fp32 `[3]` tensor,
    and the QSA NVFP4 store and sparse GQA take the same tensor, so every writer and reader
    applies the one value, 1.0 (the checkpoint has no KV scales)."""
    core = _core(QuantAlgo.NVFP4)
    assert core.quant_mode == M.QM_NVFP4_KV_CACHE
    # An NVFP4 byte holds two values: the pool is uint8, half a head wide.
    assert core.kv_pool_dtype == torch.uint8
    geometry = core._pool_geometry(32)
    assert geometry.kv_dtype == torch.uint8
    assert geometry.kv_width == core.head_dim // 2
    assert geometry.dtype == torch.bfloat16
    _load(core, monkeypatch)
    for scale in (core.kv_scale_orig_quant, core.kv_scale_quant_orig):
        assert scale is not None
        assert scale.dtype == torch.float32 and tuple(scale.shape) == (3,)
        assert torch.equal(scale, torch.ones(3))


def test_other_kv_cache_algorithms_are_refused() -> None:
    """The QSA backend serves its sparse path over half, bf16, fp8 and NVFP4 pools and falls
    back to dense for any other, so a pool of another storage must not be accepted."""
    with pytest.raises(AssertionError, match="kv_cache_quant_algo"):
        _core(QuantAlgo.INT8)

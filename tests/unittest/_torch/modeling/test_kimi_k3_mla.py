# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare complete K3 MLA outputs with the pinned Hugging Face implementation."""

import math
import weakref
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from _torch.attention.multi_gpu.helix_test_utils import CACHE_TYPE_SELFKONLY, setup_kv_and_metadata
from _torch.modeling.kimi_k3_mla_reference import KimiMLAAttention
from _torch.modeling.test_kimi_k3_checkpoint import load_mla_checkpoint
from utils.util import skip_pre_blackwell

import tensorrt_llm.bindings
from tensorrt_llm._torch.attention.backends.fp4_mla import (
    FP4_MLA_TOKENS_PER_BLOCK,
    _cutedsl_backend_available,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.cache_manager import Fp4MlaKVCacheManagerV2
from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures, KVCacheParams
from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.configs.kimi_linear import KimiLinearConfig
from tensorrt_llm._torch.kimi_k3_cache_policy import KIMI_K3_FP8_KV_LAYERS_ENV
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_kimi_linear import KimiLinearForCausalLM
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.utils import model_extra_attrs
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def _checkpoint_weights(reference: KimiMLAAttention, fp8: bool) -> dict:
    projections = {}
    for name, parameter in reference.named_parameters():
        if parameter.ndim != 2:
            continue
        weight = parameter.detach().clone()
        pair = {"weight": weight}
        if fp8:
            rows, cols = weight.shape
            padded = F.pad(weight.float(), (0, -cols % 128, 0, -rows % 128))
            blocks = padded.view(math.ceil(rows / 128), 128, math.ceil(cols / 128), 128)
            scale = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-8) / 448.0
            expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:rows, :cols]
            pair = {
                "weight": (weight.float() / expanded).to(torch.float8_e4m3fn),
                "weight_scale": scale,
            }
            # Both implementations see the same stored weights. Only TRT-LLM
            # quantizes activations and uses FP8 GEMM/BMM kernels.
            with torch.no_grad():
                parameter.copy_((pair["weight"].float() * expanded).to(parameter.dtype))
        projections[name.removesuffix(".weight")] = pair
    return projections


def _assert_matches_reference(actual: torch.Tensor, expected: torch.Tensor, fp8: bool) -> None:
    # DeepGEMM weight resmoothing and activation rounding compound across the
    # A/B projections and absorption (about 7% relative L2 on this fixture).
    # Bound both elementwise outliers and the overall error relative to the signal.
    atol, rtol = (0.1, 0.1) if fp8 else (0.005, 0.03)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    relative_l2 = (
        torch.linalg.vector_norm(actual.float() - expected.float())
        / torch.linalg.vector_norm(expected.float()).clamp_min(1e-8)
    ).item()
    assert relative_l2 < (0.08 if fp8 else 0.01), f"relative L2 error: {relative_l2}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "fp8", [False, pytest.param(True, marks=skip_pre_blackwell)], ids=["bf16", "fp8"]
)
@pytest.mark.parametrize("gate", [False, True], ids=["no_gate", "gate"])
def test_k3_mla_matches_huggingface(fp8: bool, gate: bool) -> None:
    torch.manual_seed(42)
    mapping = Mapping(world_size=1, rank=0, tp_size=1)
    config = KimiLinearConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=1,
        attn_res_block_size=1,
        torch_dtype=torch.bfloat16,
        linear_attn_config={"kda_layers": [], "full_attn_layers": [1], "num_heads": 32},
        num_attention_heads=32,
        num_key_value_heads=32,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rms_norm_eps=1e-6,
        mla_use_nope=True,
        mla_use_output_gate=gate,
        max_position_embeddings=512,
    )
    config._attn_implementation = "eager"
    reference = KimiMLAAttention(config, layer_idx=0).bfloat16().eval()
    weights = _checkpoint_weights(reference, fp8)
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=mapping,
        quant_config=QuantConfig(),
        quant_config_dict={
            f"model.layers.0.self_attn.{name}": QuantConfig(
                quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128
            )
            for name in weights
        }
        if fp8
        else None,
        use_cute_dsl_blockscaling_mm=True,
        use_cute_dsl_blockscaling_bmm=True,
    )
    with torch.device("cuda"):
        model = KimiLinearForCausalLM(model_config).eval()
    projections = {
        f"{name}.{suffix}": value
        for name, pair in weights.items()
        for suffix, value in pair.items()
    }
    projections.update(
        {name: value for name, value in reference.named_parameters() if value.ndim == 1}
    )
    load_mla_checkpoint(model, projections, "")
    runtime = model.model.layers[0].self_attn
    for module in runtime.modules():
        if callable(getattr(module, "post_load_weights", None)):
            module.post_load_weights()
    reference.cuda()

    batch, context_length = 2, 128
    generator = torch.Generator(device="cuda").manual_seed(1234)
    hidden = torch.randn(
        batch,
        context_length + 1,
        config.hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    causal_mask = torch.full(
        (context_length + 1, context_length + 1), float("-inf"), device="cuda"
    ).triu(1)[None, None]
    expected = reference(hidden, attention_mask=causal_mask)

    context = hidden[:, :context_length].reshape(-1, config.hidden_size).contiguous()
    scenario = SimpleNamespace(
        ctx_len=context_length,
        batch=batch,
        kv_cache_tokens_per_block=64,
        num_layers=1,
        kv_cache_dtype=torch.bfloat16,
    )
    manager, metadata = setup_kv_and_metadata(
        scenario,
        mapping,
        cache_type=CACHE_TYPE_SELFKONLY,
        num_kv_heads=1,
        head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
    )
    try:
        extra_attrs = {
            "attention_metadata": weakref.ref(metadata),
            "mla_layers": {runtime.mixer.layer_idx_str: weakref.ref(runtime.mixer)},
        }
        with model_extra_attrs(extra_attrs):
            actual = runtime(context, metadata).view(batch, context_length, -1)
            _assert_matches_reference(actual, expected[:, :context_length], fp8)

        for request_id in range(batch):
            assert manager.kv_cache_map[request_id].capacity >= context_length + 1
        metadata = get_attention_backend("TRTLLM").Metadata(
            seq_lens=torch.ones(batch, dtype=torch.int),
            request_ids=list(range(batch)),
            max_num_requests=batch,
            num_contexts=0,
            prompt_lens=[context_length] * batch,
            max_num_tokens=batch,
            kv_cache_manager=manager,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=[context_length] * batch
            ),
            mapping=mapping,
            enable_context_mla_with_cached_kv=True,
        )
        metadata.prepare()
        extra_attrs["attention_metadata"] = weakref.ref(metadata)
        with model_extra_attrs(extra_attrs):
            actual = runtime(hidden[:, -1].contiguous(), metadata)
        _assert_matches_reference(actual, expected[:, -1], fp8)
    finally:
        manager.shutdown()


def _k3_mla_test_config(gate: bool, num_mla_layers: int = 1) -> KimiLinearConfig:
    config = KimiLinearConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=num_mla_layers,
        attn_res_block_size=1,
        torch_dtype=torch.bfloat16,
        linear_attn_config={
            "kda_layers": [],
            "full_attn_layers": list(range(1, num_mla_layers + 1)),
            "num_heads": 32,
        },
        num_attention_heads=32,
        num_key_value_heads=32,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rms_norm_eps=1e-6,
        mla_use_nope=True,
        mla_use_output_gate=gate,
        max_position_embeddings=512,
    )
    config._attn_implementation = "eager"
    return config


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (
        torch.linalg.vector_norm(actual.float() - expected.float())
        / torch.linalg.vector_norm(expected.float()).clamp_min(1e-8)
    ).item()


def _load_mla_checkpoint_every_layer(
    model: KimiLinearForCausalLM, projections: dict[str, torch.Tensor], num_layers: int
) -> None:
    """Load the same synthetic MLA projections into every layer, zeros elsewhere."""
    name_map, _, _ = model.checkpoint_name_plan("")
    config = model.model_config.pretrained_config
    params = dict(model.named_parameters())
    weights = {}
    for name, key in name_map.items():
        param = params[name]
        if name.endswith("gate_up_proj.weight"):
            for source in ("gate_proj", "up_proj"):
                weights[key.replace("gate_up_proj", source)] = torch.zeros(
                    config.intermediate_size, config.hidden_size, dtype=param.dtype
                )
        else:
            shape = param.shape
            if name == "lm_head.weight":
                shape = (config.vocab_size, config.hidden_size)
            elif name.endswith(".mlp.down_proj.weight"):
                shape = (config.hidden_size, config.intermediate_size)
            weights[key] = torch.zeros(shape, dtype=param.dtype)
    for layer_idx in range(num_layers):
        weights.update(
            {
                f"model.layers.{layer_idx}.self_attn.{name}": value
                for name, value in projections.items()
            }
        )
    model.load_weights(weights)


def _k3_mla_nvfp4_kv_errors(
    gate: bool, fp8: bool, num_mla_layers: int = 1, cached_prefix: int = 0
) -> tuple[float, float]:
    """Run K3 MLA layer 0 on an NVFP4-KV cache manager; return (context, generation) errors.

    With ``num_mla_layers`` > 1 the cache manager holds every MLA layer, so an
    FP8-fallback selection can put layer 0 in the FP8 pool of the FP4 MLA cache.
    With ``cached_prefix`` > 0 the context runs as two chunks; the second one
    attends the first from the cache through chunked-prefill partitions.
    """
    torch.manual_seed(42)
    mapping = Mapping(world_size=1, rank=0, tp_size=1)
    config = _k3_mla_test_config(gate, num_mla_layers)
    reference = KimiMLAAttention(config, layer_idx=0).bfloat16().eval()
    weights = _checkpoint_weights(reference, fp8)
    projection_algo = QuantAlgo.FP8_BLOCK_SCALES if fp8 else None
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=mapping,
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.NVFP4),
        quant_config_dict={
            f"model.layers.0.self_attn.{name}": QuantConfig(
                quant_algo=projection_algo,
                group_size=128 if fp8 else None,
                kv_cache_quant_algo=QuantAlgo.NVFP4,
            )
            for name in weights
        },
    )
    with torch.device("cuda"):
        model = KimiLinearForCausalLM(model_config).eval()
    projections = {
        f"{name}.{suffix}": value
        for name, pair in weights.items()
        for suffix, value in pair.items()
    }
    projections.update(
        {name: value for name, value in reference.named_parameters() if value.ndim == 1}
    )
    if num_mla_layers == 1:
        load_mla_checkpoint(model, projections, "")
    else:
        _load_mla_checkpoint_every_layer(model, projections, num_mla_layers)
    runtime = model.model.layers[0].self_attn
    expected_absorption_dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    assert runtime.mixer.k_b_proj_trans.dtype == expected_absorption_dtype
    for module in runtime.modules():
        if callable(getattr(module, "post_load_weights", None)):
            module.post_load_weights()
    reference.cuda()

    batch, context_length = 2, 128
    generator = torch.Generator(device="cuda").manual_seed(1234)
    hidden = torch.randn(
        batch,
        context_length + 1,
        config.hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    causal_mask = torch.full(
        (context_length + 1, context_length + 1), float("-inf"), device="cuda"
    ).triu(1)[None, None]
    expected = reference(hidden, attention_mask=causal_mask)

    page = FP4_MLA_TOKENS_PER_BLOCK
    max_tokens = batch * 4 * page
    manager = Fp4MlaKVCacheManagerV2(
        KvCacheConfig(
            max_tokens=max_tokens, dtype="nvfp4", enable_block_reuse=False, host_cache_size=0
        ),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=num_mla_layers,
        num_kv_heads=1,
        head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
        tokens_per_block=page,
        max_seq_len=4 * page,
        max_batch_size=batch,
        mapping=mapping,
        dtype=tensorrt_llm.bindings.DataType.NVFP4,
        max_num_tokens=max_tokens,
        pretrained_config=config,
    )
    try:
        requests = manager.add_dummy_requests(list(range(batch)), [context_length + 1] * batch)
        assert requests is not None
        for req in requests:
            req.state = LlmRequestState.GENERATION_IN_PROGRESS
            req.prompt_len = context_length
            req.py_prompt_len = context_length
        backend = get_attention_backend("TRTLLM")
        extra_attrs = {
            "mla_layers": {runtime.mixer.layer_idx_str: weakref.ref(runtime.mixer)},
        }
        chunk_bounds = [0, cached_prefix, context_length] if cached_prefix else [0, context_length]
        context_outputs = []
        for start, end in zip(chunk_bounds[:-1], chunk_bounds[1:]):
            metadata = backend.Metadata(
                seq_lens=torch.tensor([end - start] * batch, dtype=torch.int),
                request_ids=list(range(batch)),
                max_num_requests=batch,
                num_contexts=batch,
                prompt_lens=[context_length] * batch,
                max_num_tokens=batch * context_length,
                kv_cache_manager=manager,
                kv_cache_params=KVCacheParams(
                    use_cache=True, num_cached_tokens_per_seq=[start] * batch
                ),
                mapping=mapping,
                enable_context_mla_with_cached_kv=bool(cached_prefix),
                runtime_features=AttentionRuntimeFeatures(
                    chunked_prefill=bool(cached_prefix), chunk_size=context_length
                ),
            )
            metadata.prepare()
            chunk = hidden[:, start:end].reshape(-1, config.hidden_size).contiguous()
            extra_attrs["attention_metadata"] = weakref.ref(metadata)
            with model_extra_attrs(extra_attrs):
                context_outputs.append(runtime(chunk, metadata).view(batch, end - start, -1))
        actual = torch.cat(context_outputs, dim=1)
        context_error = _relative_l2(actual, expected[:, :context_length])

        metadata = backend.Metadata(
            seq_lens=torch.ones(batch, dtype=torch.int),
            request_ids=list(range(batch)),
            max_num_requests=batch,
            num_contexts=0,
            prompt_lens=[context_length] * batch,
            max_num_tokens=batch,
            kv_cache_manager=manager,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=[context_length] * batch
            ),
            mapping=mapping,
            enable_context_mla_with_cached_kv=True,
        )
        metadata.prepare()
        extra_attrs["attention_metadata"] = weakref.ref(metadata)
        with model_extra_attrs(extra_attrs):
            actual = runtime(hidden[:, -1].contiguous(), metadata)
        generation_error = _relative_l2(actual, expected[:, -1])
    finally:
        manager.shutdown()
    return context_error, generation_error


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 7)
    or not _cutedsl_backend_available(),
    reason="FP4 MLA requires Rubin SM107 and the CuTe DSL runtime",
)
@pytest.mark.parametrize("gate", [False, True], ids=["no_gate", "gate"])
def test_k3_mla_nvfp4_kv_with_fp8_kv_b_matches_huggingface(
    monkeypatch: pytest.MonkeyPatch, gate: bool
) -> None:
    """FP4 MLA context and generation with FP8 block-scaled absorption weights.

    K3 checkpoints store kv_b_proj as FP8 with 128x128 block scales, so the
    absorption weights are FP8. The layer must build on an NVFP4 KV cache and
    stay as close to the reference as the same layer with BF16 weights, whose
    error is the NVFP4 KV quantization alone.
    """
    # Keep the only MLA layer on NVFP4; the FP8 tail fallback would move it.
    monkeypatch.setenv(KIMI_K3_FP8_KV_LAYERS_ENV, "")
    bf16_context, bf16_generation = _k3_mla_nvfp4_kv_errors(gate, fp8=False)
    fp8_context, fp8_generation = _k3_mla_nvfp4_kv_errors(gate, fp8=True)
    errors = (
        f"relative L2 with BF16 weights: context {bf16_context:.4f}, generation "
        f"{bf16_generation:.4f}; with FP8 weights: context {fp8_context:.4f}, "
        f"generation {fp8_generation:.4f}"
    )
    # Context attends the current chunk through FP8 partitions, so its error is
    # dominated by the FP8 projections (same bound as the BF16-KV FP8 case).
    assert bf16_context < 0.05, errors
    assert fp8_context < 0.1, errors
    # Generation runs the FP4 no-dequant decode; FP8 absorption weights must
    # not add materially to the NVFP4 quantization error of the BF16 layer.
    assert bf16_generation < 0.2, errors
    assert fp8_generation < bf16_generation + 0.02, errors


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 7)
    or not _cutedsl_backend_available(),
    reason="FP4 MLA requires Rubin SM107 and the CuTe DSL runtime",
)
def test_k3_mla_fp8_fallback_layer_on_nvfp4_cache_matches_huggingface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An FP8-fallback MLA layer runs context and generation on the FP4 MLA cache.

    The FP4 MLA cache publishes NVFP4-shaped pool pointers for every pool, so
    the FP8 layer's attention and RoPE ops must read its pool from them.
    """
    monkeypatch.setenv(KIMI_K3_FP8_KV_LAYERS_ENV, "0")
    context_error, generation_error = _k3_mla_nvfp4_kv_errors(
        gate=False, fp8=True, num_mla_layers=2
    )
    errors = f"relative L2: context {context_error:.4f}, generation {generation_error:.4f}"
    # FP8 KV keeps generation well inside the NVFP4 no-dequant decode error
    # (about 0.15 on this fixture), so the bound also shows the layer ran on FP8.
    assert context_error < 0.1, errors
    assert generation_error < 0.1, errors


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 7)
    or not _cutedsl_backend_available(),
    reason="FP4 MLA requires Rubin SM107 and the CuTe DSL runtime",
)
def test_k3_mla_nvfp4_kv_chunked_prefill_with_cached_prefix_matches_huggingface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second context chunk attends the first through FP4 chunked-prefill partitions."""
    monkeypatch.setenv(KIMI_K3_FP8_KV_LAYERS_ENV, "")
    context_error, generation_error = _k3_mla_nvfp4_kv_errors(
        gate=False, fp8=True, cached_prefix=64
    )
    errors = f"relative L2: context {context_error:.4f}, generation {generation_error:.4f}"
    assert context_error < 0.1, errors
    assert generation_error < 0.2, errors

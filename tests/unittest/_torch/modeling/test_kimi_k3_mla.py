# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare complete K3 MLA outputs with the pinned Hugging Face implementation."""

import math
import weakref
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from _torch.modeling.kimi_k3_mla_reference import KimiMLAAttention
from _torch.modeling.test_kimi_k3_checkpoint import load_mla_checkpoint
from _torch.modules.helix_test_utils import CACHE_TYPE_SELFKONLY, setup_kv_and_metadata
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.attention_backend.interface import KVCacheParams
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.configs.kimi_linear import KimiLinearConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_kimi_linear import KimiLinearForCausalLM
from tensorrt_llm._torch.utils import model_extra_attrs
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


def _k3_mla_case(fp8: bool, gate: bool):
    """The Hugging Face reference module and a one-layer TRT-LLM K3 MLA loaded from it."""
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
    return mapping, config, reference, runtime


def _prefill_then_decode(mapping, config, runtime, hidden: torch.Tensor, context_length: int):
    """Run ``runtime`` on the first ``context_length`` tokens, then decode the last one."""
    batch = hidden.shape[0]
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
            prefill = runtime(context, metadata).view(batch, context_length, -1)

        for request_id in range(batch):
            manager.impl.add_token(request_id)
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
            decode = runtime(hidden[:, -1].contiguous(), metadata)
    finally:
        manager.shutdown()
    return prefill, decode


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "fp8", [False, pytest.param(True, marks=skip_pre_blackwell)], ids=["bf16", "fp8"]
)
@pytest.mark.parametrize("gate", [False, True], ids=["no_gate", "gate"])
def test_k3_mla_matches_huggingface(fp8: bool, gate: bool) -> None:
    mapping, config, reference, runtime = _k3_mla_case(fp8, gate)
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
    prefill, decode = _prefill_then_decode(mapping, config, runtime, hidden, context_length)
    _assert_matches_reference(prefill, expected[:, :context_length], fp8)
    _assert_matches_reference(decode, expected[:, -1], fp8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@skip_pre_blackwell
def test_k3_mla_fused_gate_a_projection_is_bitwise(monkeypatch) -> None:
    """TRTLLM_K3_MLA_FUSE_GATE_A: one [g_proj; kv_a_proj_with_mqa] activation quant + GEMM
    reproduces the two projections bitwise, in prefill and in decode, and the two
    projections become row views of the fused weight."""
    outputs = {}
    for fuse in (False, True):
        monkeypatch.setenv("TRTLLM_K3_MLA_FUSE_GATE_A", "1" if fuse else "0")
        mapping, config, _, runtime = _k3_mla_case(fp8=True, gate=True)
        mixer = runtime.mixer
        assert (mixer.g_fused_a_proj is not None) == fuse
        if fuse:
            fused_storage = mixer.g_fused_a_proj.weight.untyped_storage().data_ptr()
            for name in ("g_proj", "kv_a_proj_with_mqa"):
                assert getattr(mixer, name).weight.untyped_storage().data_ptr() == fused_storage
        generator = torch.Generator(device="cuda").manual_seed(1234)
        hidden = torch.randn(
            2, 129, config.hidden_size, dtype=torch.bfloat16, device="cuda", generator=generator
        )
        outputs[fuse] = _prefill_then_decode(mapping, config, runtime, hidden, 128)
    for unfused, fused in zip(outputs[False], outputs[True]):
        assert torch.equal(fused, unfused)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@skip_pre_blackwell
@pytest.mark.parametrize("overlap", [False, True], ids=["serial_gate", "overlap_gate"])
@pytest.mark.parametrize("gate_fp8", [False, True], ids=["bf16_gate", "fp8_gate"])
@pytest.mark.parametrize("fuse", [False, True], ids=["g_proj", "fused_gate_a"])
def test_k3_mla_gate_flags_compose_bitwise(
    fuse: bool, gate_fp8: bool, overlap: bool, monkeypatch
) -> None:
    """TRTLLM_K3_MLA_FUSE_GATE_A and TRTLLM_K3_MLA_GATE_FP8 / _GATE_OVERLAP: every
    allowed combination reproduces the all-off outputs bitwise in prefill and decode, and each
    switch takes its own path; FUSE_GATE_A with GATE_OVERLAP is rejected when the module is
    built. The overlap arms run with multi-stream on, as under CUDA graphs, so g_proj does run
    on the side stream; their all-off reference runs with multi-stream on as well."""
    from tensorrt_llm._torch.models import modeling_kimi_linear as mkl
    from tensorrt_llm._torch.modules.kimi_k3_mla import kimi_k3_mla_attention as k3mla
    from tensorrt_llm._torch.modules.kimi_k3_mla import output_gate_fp8
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    def build(fuse_on: bool, fp8_on: bool, overlap_on: bool):
        # The model reads FUSE_GATE_A at weight load; the module reads the other switches at import.
        monkeypatch.setenv(mkl._KIMI_K3_MLA_FUSE_GATE_A_ENV, "1" if fuse_on else "0")
        monkeypatch.setattr(k3mla, "_K3_MLA_GATE_FP8", fp8_on)
        monkeypatch.setattr(k3mla, "_K3_MLA_GATE_OVERLAP", overlap_on)
        return _k3_mla_case(fp8=True, gate=True)

    if fuse and overlap:
        with pytest.raises(ValueError, match="are alternatives"):
            build(fuse, gate_fp8, overlap)
        return
    if gate_fp8 and torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("TRTLLM_K3_MLA_GATE_FP8 acts through o_proj's prequantized entry (SM107 only)")

    gate_strides = []
    gate_kernel = output_gate_fp8.gate_mul_fp8_r128c4

    def gate_spy(attn_out: torch.Tensor, gate: torch.Tensor):
        gate_strides.append(gate.stride(0))
        return gate_kernel(attn_out, gate)

    monkeypatch.setattr(output_gate_fp8, "gate_mul_fp8_r128c4", gate_spy)

    def run(case):
        mapping, config, _, runtime = case
        generator = torch.Generator(device="cuda").manual_seed(1234)
        hidden = torch.randn(
            2, 129, config.hidden_size, dtype=torch.bfloat16, device="cuda", generator=generator
        )
        with with_multi_stream(overlap):
            return _prefill_then_decode(mapping, config, runtime, hidden, 128)

    expected = run(build(False, False, False))
    assert not gate_strides
    case = build(fuse, gate_fp8, overlap)
    mixer = case[3].mixer
    assert (mixer.g_fused_a_proj is not None) == fuse
    assert (mixer._gate_stream is not None) == overlap
    g_proj_streams = []
    g_proj_forward = mixer.g_proj.forward

    def g_proj_spy(x: torch.Tensor) -> torch.Tensor:
        g_proj_streams.append(torch.cuda.current_stream())
        return g_proj_forward(x)

    monkeypatch.setattr(mixer.g_proj, "forward", g_proj_spy)
    main_stream = torch.cuda.current_stream()
    actual = run(case)
    for name, out, ref in zip(("prefill", "decode"), actual, expected):
        assert torch.equal(out, ref), name
    if fuse:
        # The gate came out of the fused [g_proj; kv_a] GEMM: g_proj itself never ran.
        assert not g_proj_streams
    else:
        # Once in prefill and once in decode: on the side stream with GATE_OVERLAP.
        gate_stream = mixer._gate_stream if overlap else main_stream
        assert len(g_proj_streams) == 2
        assert all(stream == gate_stream for stream in g_proj_streams)
    if gate_fp8:
        # The FP8 epilogue ran in prefill and decode (o_proj's prequantized entry).
        assert len(gate_strides) == 2
        if fuse:
            # On the gate columns of the fused [g_proj; kv_a] output, not a contiguous copy.
            assert all(stride > mixer.g_proj.out_features for stride in gate_strides)
    else:
        assert not gate_strides


@pytest.mark.parametrize("overlap", [False, True], ids=["serial_gate", "overlap_gate"])
@pytest.mark.parametrize("fuse", [None, "0", "1", "true"], ids=["unset", "0", "1", "true"])
def test_k3_mla_gate_flag_guard(fuse, overlap: bool, monkeypatch) -> None:
    """_check_gate_flags (no GPU needed) rejects only FUSE_GATE_A on with GATE_OVERLAP on. It
    reads the variable the model reads, through the same parser (get_env_flag)."""
    from tensorrt_llm._torch.models import modeling_kimi_linear as mkl
    from tensorrt_llm._torch.modules.kimi_k3_mla import kimi_k3_mla_attention as k3mla

    assert k3mla._K3_MLA_FUSE_GATE_A_ENV == mkl._KIMI_K3_MLA_FUSE_GATE_A_ENV
    monkeypatch.setattr(k3mla, "_K3_MLA_GATE_OVERLAP", overlap)
    if fuse is None:
        monkeypatch.delenv(k3mla._K3_MLA_FUSE_GATE_A_ENV, raising=False)
    else:
        monkeypatch.setenv(k3mla._K3_MLA_FUSE_GATE_A_ENV, fuse)
    if overlap and fuse in ("1", "true"):
        with pytest.raises(ValueError, match="are alternatives"):
            k3mla._check_gate_flags()
    else:
        k3mla._check_gate_flags()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@skip_pre_blackwell
def test_k3_mla_fp8_kv_decode_q_absorb() -> None:
    """FP8 KV cache decode: the FP8 query built in one kernel
    (``fp8_nope_q_absorb``) matches the BF16 absorption BMM + RoPE-kernel
    quantization path (q_pe bitwise, the absorbed q_nope to one FP8 step), and
    the KV append and BMM scales of the KV-only RoPE kernel are unchanged."""
    torch.manual_seed(7)
    mapping = Mapping(world_size=1, rank=0, tp_size=1)
    config = KimiLinearConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=1,
        attn_res_block_size=1,
        torch_dtype=torch.bfloat16,
        linear_attn_config={"kda_layers": [], "full_attn_layers": [1], "num_heads": 32},
        num_attention_heads=96,
        num_key_value_heads=96,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rms_norm_eps=1e-6,
        mla_use_nope=True,
        mla_use_output_gate=True,
        max_position_embeddings=512,
    )
    config._attn_implementation = "eager"
    reference = KimiMLAAttention(config, layer_idx=0).bfloat16().eval()
    weights = _checkpoint_weights(reference, fp8=False)
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=mapping,
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8),
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
    mla = runtime.mixer

    batch, context_length = 3, 128
    scenario = SimpleNamespace(
        ctx_len=context_length,
        batch=batch,
        kv_cache_tokens_per_block=64,
        num_layers=1,
        kv_cache_dtype=torch.float8_e4m3fn,
    )
    manager, metadata = setup_kv_and_metadata(
        scenario,
        mapping,
        cache_type=CACHE_TYPE_SELFKONLY,
        num_kv_heads=1,
        head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
    )
    generator = torch.Generator(device="cuda").manual_seed(99)
    hidden = torch.randn(
        batch * (context_length + 1),
        config.hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    captured = {}

    def capture_attention(attn_backend, q, k, v, position_ids, attn_metadata, **kwargs):
        # Stop at the decode kernel: record what the Q/KV preparation produced.
        captured["q"] = kwargs["quant_q_buffer"].clone()
        captured["bmm1"] = kwargs["mla_bmm1_scale"].clone()
        captured["bmm2"] = kwargs["mla_bmm2_scale"].clone()
        return q.new_zeros(q.shape[0], mla.num_heads_tp * mla.kv_lora_rank)

    try:
        extra_attrs = {
            "attention_metadata": weakref.ref(metadata),
            "mla_layers": {mla.layer_idx_str: weakref.ref(mla)},
        }
        context = hidden.view(batch, context_length + 1, -1)[:, :context_length]
        with model_extra_attrs(extra_attrs):
            runtime(context.reshape(-1, config.hidden_size).contiguous(), metadata)

        for request_id in range(batch):
            manager.impl.add_token(request_id)
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
        decode_in = hidden.view(batch, context_length + 1, -1)[:, -1].contiguous()
        kv_pool = manager.get_buffers(0)
        results = {}
        mla._attn_forward_gen = capture_attention
        for fused in (False, True):
            mla.fp8_nope_q_absorb = fused
            kv_pool.view(torch.uint8).zero_()
            with model_extra_attrs(extra_attrs):
                runtime(decode_in, metadata)
            torch.cuda.synchronize()
            results[fused] = dict(captured, kv=kv_pool.clone())
    finally:
        manager.shutdown()

    old, new = results[False], results[True]
    head_dim = config.kv_lora_rank + config.qk_rope_head_dim
    q_old = old["q"].view(torch.float8_e4m3fn).view(batch, 96, head_dim).float()
    q_new = new["q"].view(torch.float8_e4m3fn).view(batch, 96, head_dim).float()
    assert torch.equal(q_new[..., 512:], q_old[..., 512:])
    nope_old, nope_new = q_old[..., :512], q_new[..., :512]
    step = torch.exp2(torch.floor(torch.log2(nope_old.abs().clamp_min(2.0**-6))) - 3)
    assert ((nope_new - nope_old).abs() <= step * 1.0001).all()
    # double rounding (FP32 -> BF16 -> FP8) moves ~3% of values by one step
    assert (nope_new != nope_old).float().mean().item() < 0.1
    assert torch.equal(new["bmm1"], old["bmm1"]) and torch.equal(new["bmm2"], old["bmm2"])
    assert torch.equal(new["kv"].view(torch.uint8), old["kv"].view(torch.uint8))
    assert new["kv"].view(torch.uint8).any()

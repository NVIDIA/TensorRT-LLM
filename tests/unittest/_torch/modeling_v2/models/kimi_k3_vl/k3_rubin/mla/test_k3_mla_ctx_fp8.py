# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 MLA cached-KV context with FP8 Q / K / V into the attention op (k3_rubin/mla/ctx_fp8.py).

CPU: switch gating per KV-cache kind and the base-hook plumbing. GPU: the Triton FP8 kernels
against the C++ static FP8 quantization, and one K3 FP8-KV MLA layer whose FP8 K / V and output
are byte-identical to the default path (TRTLLM_K3_MLA_CTX_FP8_KV_FUSED)."""

import math
import weakref
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.attention.mla as mla_module
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.mla import ctx_fp8
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.mla.ctx_fp8 import (
    KimiK3MlaContextFp8Mixin,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionInputType
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention
from tensorrt_llm._torch.attention.mla import MLA

_FUSED = "mla_ctx_fp8_kv_fused"
_QUANT = "mla_ctx_triton_quant"

NOPE, ROPE, V, LORA = 128, 64, 128, 512

requires_fp8_cvt = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="E4M3 conversion instructions require SM89+",
)


# ---------------------------------------------------------------------------------------------
# CPU: gating and plumbing
# ---------------------------------------------------------------------------------------------


def _attention(*, fp4: bool, fp8: bool, sparse=None) -> TrtllmAttention:
    attention = object.__new__(TrtllmAttention)
    attention.is_mla_enable = True
    attention.has_fp4_kv_cache = fp4
    attention.has_fp8_kv_cache = fp8
    attention.sparse_params = sparse
    return attention


class _BaseHooks:
    def _cached_kv_context_inputs(self, q):
        return None, "base"


class _Layer(KimiK3MlaContextFp8Mixin, _BaseHooks):
    def __init__(self, mha, *, helix: bool = False):
        self.mha = mha
        self.mapping = SimpleNamespace(has_cp_helix=lambda: helix)
        self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim = NOPE, ROPE, V
        self.kv_lora_rank = LORA
        self.num_heads_tp = 2
        self.qk_head_dim = NOPE + ROPE


@pytest.fixture
def cpu_gates(monkeypatch, k3_features):
    monkeypatch.setattr(ctx_fp8, "get_sm_version", lambda: 107)
    monkeypatch.setattr(ctx_fp8, "attention_op_accepts_fp8_kv", lambda: True)
    k3_features(**{_FUSED: False, _QUANT: False})
    return monkeypatch


def _q(dtype=torch.bfloat16):
    return torch.zeros(4, 2 * (NOPE + ROPE), dtype=dtype)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "fp4,fp8,fused,quant,expected",
    [
        (False, True, "0", "0", False),
        (False, True, "1", "0", True),
        (False, True, "0", "1", False),  # TRITON_QUANT covers only NVFP4-cache layers
        (True, False, "0", "1", True),
        (True, False, "1", "0", False),  # FP8_KV_FUSED covers only FP8-cache layers
        (False, False, "1", "1", False),  # BF16 cache: no FP8 context FMHA
    ],
)
def test_switch_selects_layers_by_kv_cache_kind(
    cpu_gates, k3_features, fp4, fp8, fused, quant, expected
):
    k3_features(**{_FUSED: fused == "1", _QUANT: quant == "1"})
    layer = _Layer(_attention(fp4=fp4, fp8=fp8))
    assert layer._use_fp8_kv_attention_inputs(_q()) is expected


@pytest.mark.cpu_only
def test_fp8_inputs_need_supported_layer_and_op(cpu_gates, k3_features):
    k3_features(**{_FUSED: True})
    fp8_layer = _Layer(_attention(fp4=False, fp8=True))
    assert fp8_layer._use_fp8_kv_attention_inputs(_q())
    assert not fp8_layer._use_fp8_kv_attention_inputs(_q(torch.float16))
    assert not fp8_layer._use_fp8_kv_attention_inputs(_q()[:, ::2])
    assert not _Layer(_attention(fp4=False, fp8=True), helix=True)._use_fp8_kv_attention_inputs(
        _q()
    )
    assert not _Layer(
        _attention(fp4=False, fp8=True, sparse=object())
    )._use_fp8_kv_attention_inputs(_q())
    narrow = _Layer(_attention(fp4=False, fp8=True))
    narrow.v_head_dim = 64
    assert not narrow._use_fp8_kv_attention_inputs(_q())
    cpu_gates.setattr(ctx_fp8, "get_sm_version", lambda: 89)
    assert not fp8_layer._use_fp8_kv_attention_inputs(_q())
    cpu_gates.setattr(ctx_fp8, "get_sm_version", lambda: 107)
    # A C++ library without FP8 K / V input keeps the BF16 path.
    cpu_gates.setattr(ctx_fp8, "attention_op_accepts_fp8_kv", lambda: False)
    assert not fp8_layer._use_fp8_kv_attention_inputs(_q())
    assert fp8_layer._cached_kv_context_inputs(_q()) == (None, "base")


@pytest.mark.cpu_only
def test_hook_returns_fp8_q_and_fp8_expansion(cpu_gates, k3_features):
    k3_features(**{_FUSED: True})
    layer = _Layer(_attention(fp4=False, fp8=True))
    layer._quantize_q_fp8 = lambda q: ("q_fp8", q.shape)
    q_fp8, expand_kv = layer._cached_kv_context_inputs(_q())
    assert q_fp8 == ("q_fp8", _q().shape)
    assert expand_kv == layer._expand_kv_to_fp8_k_v
    k3_features(**{_FUSED: False})
    assert layer._cached_kv_context_inputs(_q()) == (None, "base")


@pytest.mark.cpu_only
@pytest.mark.parametrize("heads,per_program", [(96, 4), (12, 4), (6, 2), (3, 1)])
def test_fp8_expansion_heads_per_program_divides_rank_heads(monkeypatch, heads, per_program):
    """Any per-rank head count (e.g. 96 heads over TP16 = 6) gets a program width that divides it."""
    seen = {}

    def mla_kv_to_fp8(kv, k_pe, num_heads, nope, v_dim, *, heads_per_program):
        seen.update(num_heads=num_heads, heads_per_program=heads_per_program, k_pe=k_pe.shape)
        return "k8", "v8"

    monkeypatch.setattr(ctx_fp8.fp8_static_quant, "mla_kv_to_fp8", mla_kv_to_fp8)
    layer = _Layer(_attention(fp4=False, fp8=True))
    layer.num_heads_tp = heads
    layer.kv_b_proj = lambda x: torch.zeros(x.shape[0], heads * (NOPE + V))
    assert layer._expand_kv_to_fp8_k_v(torch.zeros(5, LORA), torch.zeros(5, 1, ROPE)) == (
        "k8",
        "v8",
    )
    assert seen == dict(num_heads=heads, heads_per_program=per_program, k_pe=(5, ROPE))


@pytest.mark.cpu_only
def test_cached_kv_path_passes_hook_inputs_to_the_op():
    """The base forward_context_with_cached_kv expands with the hook's function and hands its
    Q buffer to the attention op."""
    num_tokens = 6
    seen = {}
    q_fp8 = torch.ones(1)

    def expand(compressed_kv, k_pe):
        seen["expand"] = (compressed_kv.shape[0], k_pe.shape[0])
        return torch.zeros(num_tokens, 2 * (NOPE + ROPE)), torch.zeros(num_tokens, 2 * V)

    def forward(q, k, v, metadata, forward_args):
        assert forward_args.attention_input_type == AttentionInputType.context_only
        assert forward_args.latent_cache is None
        seen["quant_q_buffer"] = forward_args.quant_q_buffer
        return forward_args.output

    stub = SimpleNamespace(
        mha=SimpleNamespace(
            mla_rope_append_paged_kv_assign_q=lambda *args: None,
            load_paged_kv_cache_for_mla=lambda metadata, dtype: (
                torch.zeros(num_tokens, LORA),
                torch.zeros(num_tokens, ROPE),
            ),
            forward=forward,
        ),
        _cached_kv_context_inputs=lambda q: (q_fp8, expand),
        kv_lora_rank=LORA,
        qk_rope_head_dim=ROPE,
        out_scale=None,
    )
    metadata = SimpleNamespace(num_ctx_cached_tokens=4, num_ctx_tokens=2)
    output = torch.empty(2, 2 * V)
    q = torch.zeros(2, 2 * (NOPE + ROPE))
    assert MLA.forward_context_with_cached_kv(stub, q, torch.zeros(2, 1), metadata, output) is (
        output
    )
    assert seen == {"expand": (num_tokens, num_tokens), "quant_q_buffer": q_fp8}


@pytest.mark.cpu_only
def test_base_hook_keeps_bf16_concat(monkeypatch):
    """Without the mixin the cached-KV paths keep main's kv_b_proj + concat expansion."""
    monkeypatch.setattr(mla_module, "maybe_compiled_cat", torch.cat)
    heads, nope, rope, v_dim, tokens = 2, 4, 2, 3, 5
    stub = SimpleNamespace(
        num_heads_tp=heads,
        qk_nope_head_dim=nope,
        qk_rope_head_dim=rope,
        qk_head_dim=nope + rope,
        v_head_dim=v_dim,
        kv_b_proj=lambda x: torch.arange(tokens * heads * (nope + v_dim), dtype=torch.float32).view(
            tokens, -1
        ),
    )
    stub._expand_kv_to_k_v = lambda *args: MLA._expand_kv_to_k_v(stub, *args)
    q_fp8, expand_kv = MLA._cached_kv_context_inputs(stub, torch.zeros(1))
    assert q_fp8 is None
    k_pe = torch.full((tokens, rope), -1.0)
    k, v = expand_kv(torch.zeros(tokens, 1), k_pe)
    kv = stub.kv_b_proj(None)
    expected_k = torch.cat(
        (kv[:, : heads * nope].view(tokens, heads, nope), k_pe[:, None].expand(-1, heads, -1)), -1
    ).view(tokens, -1)
    assert torch.equal(k, expected_k)
    assert torch.equal(v, kv[:, heads * nope :])


# ---------------------------------------------------------------------------------------------
# GPU: Triton FP8 kernels against the C++ static FP8 quantization
# ---------------------------------------------------------------------------------------------


def _cpp_quant(x, scale):
    return torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x.contiguous(), scale)[0]


def _quant_test_input(rows, cols, seed):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    x = torch.randn(rows, cols, generator=g, device="cuda") * 64
    flat = x.view(-1)
    special = [
        float("inf"),
        float("-inf"),
        float("nan"),
        1e30,
        -1e30,
        1e-30,
        -0.0,
        0.0,
        447.0,
        448.0,
        456.0,
        464.0,
        465.0,
        480.0,
        -464.0,
        -465.0,
        2.0**-9,
        2.0**-10,
    ]
    flat[: len(special)] = torch.tensor(special, device="cuda")
    n = min(flat.numel() - len(special), 8192)
    flat[len(special) : len(special) + n] = torch.logspace(-12, 12, n, device="cuda") * torch.sign(
        torch.randn(n, generator=g, device="cuda")
    )
    return x.to(torch.bfloat16)


@requires_fp8_cvt
@pytest.mark.parametrize("shape", [(1, 18432), (257, 18432), (4096, 192), (3, 6)])
@pytest.mark.parametrize("scale", [1.0, 0.5, 3.0, 0.37])
def test_triton_static_quant_matches_cpp(shape, scale):
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.kernels import (
        fp8_static_quant,
    )

    x = _quant_test_input(*shape, seed=shape[0])
    s = torch.tensor([scale], dtype=torch.float32, device="cuda")
    y = fp8_static_quant.static_quantize_e4m3(x, s)
    assert y.shape == x.shape and y.dtype == torch.float8_e4m3fn
    assert torch.equal(y.view(torch.uint8), _cpp_quant(x, s).view(torch.uint8))


@requires_fp8_cvt
@pytest.mark.parametrize("rows", [1, 63, 64, 1000, 16384])
@pytest.mark.parametrize("scale", [1.0, 0.37])
def test_triton_static_quant_rows_matches_cpp(rows, scale):
    """Row-strided views (the K-pe columns of latent_cache) quantize to the C++ bytes."""
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.kernels import (
        fp8_static_quant,
    )

    latent = _quant_test_input(rows, LORA + ROPE, seed=rows)
    s = torch.tensor([scale], dtype=torch.float32, device="cuda")
    for view in (latent[:, LORA:], latent[:, LORA:].contiguous(), latent[:, :ROPE]):
        assert fp8_static_quant.supports_rows(view)
        y = fp8_static_quant.static_quantize_e4m3_rows(view, s)
        assert y.is_contiguous() and y.shape == view.shape
        assert torch.equal(y.view(torch.uint8), _cpp_quant(view, s).view(torch.uint8))
    # Odd storage offsets, non-power-of-two widths and non-BF16 inputs are not supported.
    for view in (latent[:, 511:575], latent[:, LORA : LORA + 48], latent[:, LORA:].float()):
        assert not fp8_static_quant.supports_rows(view)
        with pytest.raises(ValueError):
            fp8_static_quant.static_quantize_e4m3_rows(view, s)


@requires_fp8_cvt
@pytest.mark.parametrize("tokens,heads", [(1, 96), (31, 96), (1000, 96), (4099, 96), (65553, 8)])
@pytest.mark.parametrize("strided_pe", [False, True], ids=["contiguous_pe", "latent_cache_pe"])
def test_mla_kv_to_fp8_matches_cat_and_cpp_quant(tokens, heads, strided_pe):
    """FP8 K / V of mla_kv_to_fp8 are the C++ unit-scale FP8 bytes of the BF16 [T, H, 192]
    concat of k_nope and k_pe and of the V columns (special values and saturation included)."""
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.kernels import (
        fp8_static_quant,
    )

    kv = _quant_test_input(tokens, heads * (NOPE + V), seed=tokens)
    latent = _quant_test_input(tokens, LORA + ROPE, seed=tokens + 1).flip(1).contiguous()
    k_pe = latent[:, LORA:] if strided_pe else latent[:, LORA:].contiguous()
    k8, v8 = fp8_static_quant.mla_kv_to_fp8(kv, k_pe, heads, NOPE, V)
    k_bf16 = torch.cat(
        (kv[:, : heads * NOPE].view(-1, heads, NOPE), k_pe[:, None].expand(-1, heads, -1)), -1
    ).view(tokens, -1)
    unit = torch.ones(1, dtype=torch.float32, device="cuda")
    for x8, x in ((k8, k_bf16), (v8, kv[:, heads * NOPE :])):
        assert x8.dtype == torch.float8_e4m3fn and x8.is_contiguous() and x8.shape == x.shape
        assert torch.equal(x8.view(torch.uint8), _cpp_quant(x, unit).view(torch.uint8))


@requires_fp8_cvt
def test_mla_kv_to_fp8_rejects_bad_layouts():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.kernels import (
        fp8_static_quant,
    )

    heads, tokens = 4, 16
    kv = torch.randn(tokens, heads * (NOPE + V), device="cuda").to(torch.bfloat16)
    latent = torch.randn(tokens, LORA + ROPE, device="cuda").to(torch.bfloat16)
    k, v = fp8_static_quant.mla_kv_to_fp8(kv[:0], latent[:0, LORA:], heads, NOPE, V)
    assert k.shape == (0, heads * (NOPE + ROPE)) and v.shape == (0, heads * V)
    bad_calls = (
        (kv[:, 1:], latent[:, LORA:], heads),  # kv width
        (kv.float(), latent[:, LORA:], heads),  # kv dtype
        (kv, latent[:, LORA - 1 : -1], heads),  # odd k_pe offset
        (kv, latent[:, LORA : LORA + 48], heads),  # k_pe width not a power of two
        (kv[:-1], latent[:, LORA:], heads),  # token counts differ
    )
    for kv_arg, pe_arg, h in bad_calls:
        with pytest.raises(ValueError):
            fp8_static_quant.mla_kv_to_fp8(kv_arg, pe_arg, h, NOPE, V)
    with pytest.raises(ValueError):
        fp8_static_quant.mla_kv_to_fp8(kv, latent[:, LORA:], heads, NOPE, V, heads_per_program=3)


# ---------------------------------------------------------------------------------------------
# GPU: one K3 FP8-KV MLA layer, default path vs FP8 Q / K / V into the attention op
# ---------------------------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="needs the FP8 context MLA FMHA of SM100 / SM103 / SM107",
)
@pytest.mark.parametrize(
    "cached,new,chunk_size,buffer_batch",
    [
        ([256], [128], 64, 1),
        ([256, 200], [128, 184], 64, 1),
        ([1000, 3000, 37], [100, 64, 300], 512, 2),
        ([256, 200], [128, 184], None, None),
    ],
    ids=["one_request", "two_requests", "three_requests_1024_key_slices", "cached_kv"],
)
def test_k3_mla_ctx_fp8_kv_fused_matches_default(
    cached, new, chunk_size, buffer_batch, monkeypatch, k3_features
) -> None:
    """``features.mla_ctx_fp8_kv_fused`` hands the attention op FP8 Q / K / V. The FP8 K and V of
    every cached-KV partition and of the uncached tokens are the unit-scale FP8 bytes of the
    default path's BF16 K and V, and the layer output is bit-identical, through the
    chunked-prefill loop (partitions within and across requests) and the plain cached-KV path."""
    from _torch.attention.multi_gpu.helix_test_utils import (
        CACHE_TYPE_SELFKONLY,
        setup_kv_and_metadata,
    )
    from _torch.modeling.kimi_k3_mla_reference import KimiMLAAttention
    from _torch.modeling.test_kimi_k3_checkpoint import load_mla_checkpoint
    from _torch.modeling.test_kimi_k3_mla import _checkpoint_weights

    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.text_model import (
        KimiLinearForCausalLM,
    )
    from tensorrt_llm._torch.attention.backends.interface import (
        AttentionRuntimeFeatures,
        KVCacheParams,
    )
    from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
    from tensorrt_llm._torch.configs.kimi_linear import KimiLinearConfig
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.utils import model_extra_attrs
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    if not ctx_fp8.attention_op_accepts_fp8_kv():
        pytest.skip("the C++ attention op of this build has no FP8 K / V input")
    torch.manual_seed(13)
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
        max_position_embeddings=4096,
    )
    config._attn_implementation = "eager"
    reference = KimiMLAAttention(config, layer_idx=0).bfloat16().eval()
    weights = _checkpoint_weights(reference, fp8=True)
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=mapping,
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8),
        quant_config_dict={
            f"model.layers.0.self_attn.{name}": QuantConfig(
                quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128
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
    load_mla_checkpoint(model, projections, "")
    runtime = model.model.layers[0].self_attn
    for module in runtime.modules():
        if callable(getattr(module, "post_load_weights", None)):
            module.post_load_weights()
    mla = runtime.mixer
    assert isinstance(mla, KimiK3MlaContextFp8Mixin)
    del reference

    batch = len(cached)
    total = [c + n for c, n in zip(cached, new)]
    scenario = SimpleNamespace(
        ctx_len=max(total),
        batch=batch,
        kv_cache_tokens_per_block=64,
        num_layers=1,
        kv_cache_dtype=torch.float8_e4m3fn,
    )
    manager, _ = setup_kv_and_metadata(
        scenario,
        mapping,
        cache_type=CACHE_TYPE_SELFKONLY,
        num_kv_heads=1,
        head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
    )
    generator = torch.Generator(device="cuda").manual_seed(3)
    hidden = [
        torch.randn(t, config.hidden_size, dtype=torch.bfloat16, device="cuda", generator=generator)
        for t in total
    ]

    def context_metadata(seq_lens, cached_lens, runtime_features=None):
        kwargs = {} if runtime_features is None else {"runtime_features": runtime_features}
        metadata = get_attention_backend("TRTLLM").Metadata(
            seq_lens=torch.tensor(seq_lens, dtype=torch.int),
            request_ids=list(range(batch)),
            max_num_requests=batch,
            num_contexts=batch,
            prompt_lens=list(seq_lens),
            max_num_tokens=sum(seq_lens),
            kv_cache_manager=manager,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=cached_lens),
            mapping=mapping,
            enable_context_mla_with_cached_kv=True,
            **kwargs,
        )
        metadata.prepare()
        return metadata

    # Record what reaches the attention op and the K / V of both expansions.
    op_inputs, bf16_kv, fp8_kv = [], [], []
    mha_forward = mla.mha.forward
    expand_bf16 = mla._expand_kv_to_k_v
    expand_fp8 = mla._expand_kv_to_fp8_k_v

    def recording_forward(q, k, v, metadata, forward_args, **kwargs):
        op_inputs.append((k.dtype, v.dtype, forward_args.quant_q_buffer is not None))
        return mha_forward(q, k, v, metadata, forward_args=forward_args, **kwargs)

    def recording_bf16(*args):
        k, v = expand_bf16(*args)
        bf16_kv.append((k.clone(), v.clone()))
        return k, v

    def recording_fp8(*args):
        k, v = expand_fp8(*args)
        fp8_kv.append((k.clone(), v.clone()))
        return k, v

    monkeypatch.setattr(mla.mha, "forward", recording_forward)
    monkeypatch.setattr(mla, "_expand_kv_to_k_v", recording_bf16)
    monkeypatch.setattr(mla, "_expand_kv_to_fp8_k_v", recording_fp8)
    outputs, op_calls = {}, {}
    try:
        # The first chunk fills the FP8 cache with the first cached[i] tokens of each request.
        k3_features(**{_FUSED: False})
        first = context_metadata(cached, [0] * batch)
        extra_attrs = {
            "attention_metadata": weakref.ref(first),
            "mla_layers": {mla.layer_idx_str: weakref.ref(mla)},
        }
        with model_extra_attrs(extra_attrs):
            runtime(torch.cat([hidden[i][: cached[i]] for i in range(batch)]), first)
        second_input = torch.cat([hidden[i][cached[i] :] for i in range(batch)])
        features = (
            None
            if chunk_size is None
            else AttentionRuntimeFeatures(
                chunked_prefill=True,
                chunk_size=chunk_size,
                chunked_prefill_buffer_batch_size=buffer_batch,
            )
        )
        for flag in ("0", "1"):
            k3_features(**{_FUSED: flag == "1"})
            op_inputs.clear()
            second = context_metadata(new, cached, features)
            extra_attrs["attention_metadata"] = weakref.ref(second)
            with model_extra_attrs(extra_attrs):
                outputs[flag] = runtime(second_input, second).clone()
            torch.cuda.synchronize()
            op_calls[flag] = list(op_inputs)
        assert mla._use_fp8_kv_attention_inputs(second_input.new_empty(1, 1))
    finally:
        manager.shutdown()

    partitions = 1
    if chunk_size is not None:
        partitions += math.ceil(sum(cached) / (chunk_size * buffer_batch))
    assert op_calls["0"] == [(torch.bfloat16, torch.bfloat16, False)] * partitions
    assert op_calls["1"] == [(torch.float8_e4m3fn, torch.float8_e4m3fn, True)] * partitions
    assert len(fp8_kv) == partitions
    assert len(bf16_kv) == partitions
    unit = torch.ones(1, dtype=torch.float32, device="cuda")
    for (k, v), (k8, v8) in zip(bf16_kv, fp8_kv):
        for x, x8 in ((k, k8), (v, v8)):
            expected = _cpp_quant(x, unit)
            assert x8.is_contiguous() and x8.shape == x.shape
            assert torch.equal(x8.view(torch.uint8), expected.view(torch.uint8))
    assert outputs["0"].abs().sum() > 0
    assert torch.equal(outputs["1"].view(torch.int16), outputs["0"].view(torch.int16))

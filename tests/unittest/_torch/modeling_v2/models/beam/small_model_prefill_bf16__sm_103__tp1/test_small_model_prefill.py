# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the Beam six-block small model: bf16 / sm_103 / tp1.

Configuration under test, fixed below and nowhere else: the first six logical layers of the
checkpoint's schedule -- dense sliding-attention blocks 0-2, sparse sliding-attention blocks
3-4 and the sparse full-attention block 5 without position embedding -- at hidden 5120 with
48 q / 8 kv heads of 128, 256 experts of 2560 per sparse block, vocabulary 151936, untied
embedding and head and a final-logit softcap of 30; bf16 weights and activations with fp32
routers; one sm_103 GPU; eager, batch 1, one fresh context per call on a six-layer
KVCacheManagerV2; generated weights; seeds {17, 41} x lengths {1, 17, 129}.

What is checked. Construction, the 2394-tensor loading contract and its rejections, the
input and cache contracts (rejected before any op runs), and on every case the composition
itself, observed read-only at the catalog ops of the forward: the six blocks run once each
in order, each on its predecessor's output, with its own residual coefficients, attention
kind, window and cache slot; layer 5's q takes the query scale and its k is the normalized
k unrotated; the final norm, head and softcap run once each, and the returned logits are
exactly the softcap of the head output. Numerical agreement with the source model is
established separately; these tests generate everything at runtime and need no model files.

Invocation (fp32 steps need the TF32 overrides removed before torch is imported)::

    env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE -u NVIDIA_TF32_OVERRIDE python -m pytest \\
        tests/unittest/_torch/modeling_v2/models/beam/small_model_prefill_bf16__sm_103__tp1

Dependencies: a TensorRT-LLM build for sm_103 with flashinfer importable and about 200 GiB
of free device memory: the generated state and the loaded model hold 62 GiB each, and one
test constructs a second model. No model files.
"""

from __future__ import annotations

import contextlib
import gc
import math
import types
import zlib
from collections import OrderedDict

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.beam_logit_softcap import (  # noqa: E501
    beam_logit_softcap,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.flashinfer_silu_and_mul import (  # noqa: E501
    flashinfer_silu_and_mul,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.swiglu_fp32 import (  # noqa: E501
    swiglu_fp32,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.beam_neox_rope import (
    beam_neox_rope,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.beam_nope_query_scale import (  # noqa: E501
    beam_nope_query_scale,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.beam_output_gate import (
    beam_output_gate,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    thop_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.beam_dispatch_combine import (  # noqa: E501
    beam_dispatch_combine_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.beam_grouped_topk_router import (  # noqa: E501
    beam_grouped_topk_router,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.residual.beam_depth_residual import (
    beam_depth_residual,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.small_model_prefill_bf16__sm_103__tp1 import (  # noqa: E501
    model as model_module,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.small_model_prefill_bf16__sm_103__tp1.model import (  # noqa: E501
    BeamSmallModelPrefillBf16Sm103Tp1,
    prefill_context,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

assert torch.cuda.is_available(), "the Beam six-block model requires a CUDA device"

pytestmark = pytest.mark.high_cuda_memory

# -------------------------------------------------- the fixed configuration

#: The checkpoint's configuration reduced to its first six layers; only these fields are read.
CONFIG = types.MappingProxyType(
    dict(
        num_hidden_layers=6,
        hidden_size=5120,
        intermediate_size=20480,
        moe_intermediate_size=2560,
        num_attention_heads=48,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=151936,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        sliding_window=512,
        layer_types=("sliding_attention",) * 5 + ("full_attention",),
        mlp_layer_types=("dense",) * 3 + ("sparse",) * 3,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": None,
        },
        num_experts=256,
        num_experts_per_tok=8,
        n_group=8,
        topk_group=4,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        tie_word_embeddings=False,
        final_logit_softcapping=30.0,
        max_position_embeddings=1048576,
    )
)
SEEDS = (17, 41)
LENGTHS = (1, 17, 129)
CASES = tuple((seed, length) for seed in SEEDS for length in LENGTHS)
CASE_IDS = [f"seed{seed}-T{length}" for seed, length in CASES]
VOCAB = CONFIG["vocab_size"]
HIDDEN = CONFIG["hidden_size"]

#: Every catalog entry the forward can call.
OPS = (
    flashinfer_rmsnorm,
    cublas_mm,
    beam_neox_rope,
    beam_nope_query_scale,
    thop_attention,
    beam_output_gate,
    beam_depth_residual,
    flashinfer_silu_and_mul,
    beam_grouped_topk_router,
    beam_dispatch_combine_op,
    swiglu_fp32,
    beam_logit_softcap,
)


def model_config(**overrides) -> types.SimpleNamespace:
    fields = dict(CONFIG)
    fields["layer_types"] = list(fields["layer_types"])
    fields["mlp_layer_types"] = list(fields["mlp_layer_types"])
    fields["rope_parameters"] = {
        k: (dict(v) if v else v) for k, v in fields["rope_parameters"].items()
    }
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def depth_scales(layer_idx: int, is_attention: bool) -> tuple[float, float]:
    """The source model's depth residual coefficients: the n-th add of the residual stream
    scales the branch by 1 / sqrt(n) and the stream by sqrt(n - 1) / sqrt(n)."""
    n = 2 * (layer_idx + 1) + (0 if is_attention else 1)
    return 1.0 / math.sqrt(n), math.sqrt(n - 1) / math.sqrt(n)


# -------------------------------------------------- generated state and inputs


def _half_width(name: str, shape: tuple) -> tuple[float, float]:
    """(half width, offset) of the uniform draw for one tensor: unit-variance projections,
    embeddings of std 1, norm weights 1 +- 0.4, gate bias std 0.75, correction bias std 0.05."""
    if name == model_module.EMBED:
        return math.sqrt(3.0), 0.0
    if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
        return 0.4, 1.0
    if name.endswith("gate_bias"):
        return 0.75 * math.sqrt(3.0), 0.0
    if name.endswith("e_score_correction_bias"):
        return 0.05 * math.sqrt(3.0), 0.0
    return math.sqrt(3.0 / shape[-1]), 0.0


def generate_state(seed: int) -> "OrderedDict[str, torch.Tensor]":
    """All 2394 tensors, each drawn from its own generator keyed by seed and name, fp32 then
    rounded once to the stored dtype."""
    state = OrderedDict()
    schema = model_module.weight_schema(model_config())
    for name, (shape, dtype) in schema.items():
        g = torch.Generator(device="cuda").manual_seed(zlib.crc32(f"{seed}|{name}".encode()))
        half, offset = _half_width(name, shape)
        t = torch.rand(shape, generator=g, device="cuda", dtype=torch.float32)
        state[name] = (t.mul_(2.0 * half).sub_(half).add_(offset)).to(dtype)
    return state


def generate_ids(seed: int, length: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(0, VOCAB, (1, max(LENGTHS)), generator=g, device="cuda")[:, :length]


_CACHE: dict = {}


@pytest.fixture(scope="module", autouse=True)
def device_memory():
    """Release the generated weights and model before the next module runs."""
    yield
    _CACHE.clear()
    gc.collect()
    torch.cuda.empty_cache()


def loaded(seed: int) -> tuple[BeamSmallModelPrefillBf16Sm103Tp1, dict]:
    """The seed's model, loaded with the seed's generated state; one seed at a time."""
    if _CACHE.get("seed") != seed:
        _CACHE.clear()
        torch.cuda.empty_cache()
        state = generate_state(seed)
        model = BeamSmallModelPrefillBf16Sm103Tp1(model_config(), device="cuda")
        model.load_weights(state)
        _CACHE.update(seed=seed, state=state, model=model)
    return _CACHE["model"], _CACHE["state"]


def positions_of(length: int) -> torch.Tensor:
    return torch.arange(length, device="cuda").view(1, length)


def run(model, ids: torch.Tensor) -> torch.Tensor:
    length = ids.shape[1]
    with prefill_context(length) as md:
        out = model(ids, positions_of(length), md)
        torch.cuda.synchronize()
    return out


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Same dtype, shape and bits (so -0.0 differs from +0.0)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    view = {torch.bfloat16: torch.int16, torch.float32: torch.int32}.get(a.dtype)
    if view is None:
        return bool(torch.equal(a, b))
    return bool(torch.equal(a.contiguous().view(view), b.contiguous().view(view)))


# -------------------------------------------------- op-path instrumentation


@contextlib.contextmanager
def recording(model):
    """Log every catalog op call of the forward in order, with the block it ran in, its
    activation arguments and its output; and every block's entry and exit."""
    calls: list[dict] = []
    events: list = []
    current = [None]
    saved = []
    weight_args_from = {"CublasMm": 1, "FlashinferRmsnorm": 1, "BeamGroupedTopkRouter": 1}
    for w in OPS:
        cls = type(w)
        original = cls.__call__
        saved.append((cls, original))

        def logged(self, *args, _original=original, _name=cls.__name__, **kwargs):
            keep = weight_args_from.get(_name, len(args))
            entry = {
                "op": _name,
                "block": current[0],
                "args": [a if j < keep else None for j, a in enumerate(args)],
                "kwargs": kwargs,
            }
            if isinstance(kwargs.get("q"), torch.Tensor):
                entry["qkv"] = kwargs["q"].clone()  # before the op appends from it
            calls.append(entry)
            entry["output"] = _original(self, *args, **kwargs)
            return entry["output"]

        cls.__call__ = logged
    handles = []
    for i, block in enumerate(model.layers):

        def pre(module, args, _i=i):
            current[0] = _i
            events.append(("enter", _i, args[0]))

        def post(module, args, out, _i=i):
            events.append(("exit", _i, out))
            current[0] = None

        handles.append(block.register_forward_pre_hook(pre))
        handles.append(block.register_forward_hook(post))
    try:
        yield calls, events
    finally:
        for h in handles:
            h.remove()
        for cls, original in reversed(saved):
            cls.__call__ = original


@contextlib.contextmanager
def no_dispatch():
    """Fail on any op call or embedding lookup: rejected inputs are rejected before either."""
    saved = []
    for w in OPS:
        cls = type(w)
        saved.append((cls, cls.__call__))

        def refuse(self, *args, _name=cls.__name__, **kwargs):
            raise AssertionError(f"{_name} dispatched before the input was rejected")

        cls.__call__ = refuse
    embedding = torch.nn.functional.embedding

    def refuse_embedding(*args, **kwargs):
        raise AssertionError("the embedding ran before the input was rejected")

    torch.nn.functional.embedding = refuse_embedding
    try:
        yield
    finally:
        torch.nn.functional.embedding = embedding
        for cls, original in reversed(saved):
            cls.__call__ = original


@contextlib.contextmanager
def manager_context(length: int, **overrides):
    """A KVCacheManagerV2 other than the model's own, with prepared metadata for one request."""
    windows = overrides.pop("max_attention_window", list(model_module.ATTENTION_WINDOWS))
    args = dict(
        num_layers=6,
        num_kv_heads=8,
        head_dim=128,
        tokens_per_block=32,
        max_seq_len=1024,
        max_batch_size=1,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.BF16,
    )
    args.update(overrides)
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
    from tensorrt_llm._torch.metadata import KVCacheParams

    mgr = KVCacheManagerV2(
        KvCacheConfig(max_tokens=1024, enable_block_reuse=False, max_attention_window=windows),
        CacheTypeCpp.SELF,
        **args,
    )
    try:
        mgr.add_dummy_requests([0], token_nums=[length])
        md = TrtllmAttentionMetadata(
            max_num_requests=1,
            max_num_tokens=model_module.MAX_TOKENS,
            kv_cache_manager=mgr,
            num_contexts=1,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0]),
            seq_lens=torch.tensor([length], dtype=torch.int),
            request_ids=[0],
            prompt_lens=[length],
        )
        md.prepare()
        yield md
    finally:
        mgr.shutdown()


# -------------------------------------------------- tests: configuration and construction


def test_configuration_is_the_fixed_scope() -> None:
    assert model_module.NUM_LAYERS == 6
    assert model_module.LAYER_TYPES == tuple(CONFIG["layer_types"])
    assert model_module.MLP_LAYER_TYPES == tuple(CONFIG["mlp_layer_types"])
    assert model_module.ATTENTION_WINDOWS == (512,) * 5 + (1024,)
    assert model_module.MAX_TOKENS == model_module.KV_CACHE["max_seq_len"] == 1024
    for field, value in model_module.SUPPORTED_CONFIG.items():
        assert CONFIG[field] == value, field
    schema = model_module.weight_schema(model_config())
    assert len(schema) == 2394
    assert sum(1 for n in schema if ".mlp.experts." in n) == 3 * 256 * 3
    fp32 = sorted(n for n, (_, dtype) in schema.items() if dtype == torch.float32)
    assert fp32 == sorted(
        f"model.layers.{i}.mlp.gate.{t}"
        for i in (3, 4, 5)
        for t in ("weight", "e_score_correction_bias")
    )


@pytest.mark.parametrize(
    "overrides,match",
    [
        (dict(num_hidden_layers=52), "num_hidden_layers"),
        (dict(tie_word_embeddings=True), "tie_word_embeddings"),
        (dict(final_logit_softcapping=None), "final_logit_softcapping"),
        (dict(vocab_size=151552), "vocab_size"),
        (dict(layer_types=["sliding_attention"] * 6), "layer_types"),
        (dict(mlp_layer_types=["dense"] * 4 + ["sparse"] * 2), "mlp_layer_types"),
        (
            dict(
                rope_parameters={"sliding_attention": {"rope_type": "default", "rope_theta": 1e4}}
            ),
            "full_attention",
        ),
    ],
)
def test_rejects_unsupported_configuration(overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        BeamSmallModelPrefillBf16Sm103Tp1(model_config(**overrides), device="cuda")


def test_constructs_six_distinct_blocks() -> None:
    model, _ = loaded(17)
    assert [type(b).__name__ for b in model.layers] == ["BeamDenseBlockBf16Sm103Tp1"] * 3 + [
        "BeamSparseBlockBf16Sm103Tp1"
    ] * 3
    assert [b.layer_idx for b in model.layers] == list(range(6))
    for i, block in enumerate(model.layers):
        assert block.attention_scales == depth_scales(i, True)
        assert block.feed_forward_scales == depth_scales(i, False)
    assert [getattr(b, "is_sliding", True) for b in model.layers] == [True] * 5 + [False]
    assert [b.rope_theta for b in model.layers] == [10000.0] * 5 + [None]
    for i in (3, 4, 5):
        assert len(model.layers[i].mlp.experts) == 256
        assert model.layers[i].mlp.gate.weight.dtype == torch.float32
    tensors = model.checkpoint_tensors()
    assert len(tensors) == 2394
    spans = sorted(
        (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in tensors.values()
    )
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), "two tensors share storage"


def test_unloaded_model_refuses_to_run() -> None:
    loaded(17)
    fresh = BeamSmallModelPrefillBf16Sm103Tp1(model_config(), device="cuda")
    with prefill_context(1) as md, no_dispatch(), pytest.raises(ValueError, match="load_weights"):
        fresh(generate_ids(17, 1), positions_of(1), md)
    del fresh
    torch.cuda.empty_cache()


# -------------------------------------------------- tests: loading


def test_load_weights_maps_every_tensor() -> None:
    """Every tensor lands under its own name, value, dtype and [out, in] orientation."""
    model, state = loaded(17)
    tensors = model.checkpoint_tensors()
    assert list(tensors) == list(state)
    for name, value in state.items():
        assert bitwise_equal(tensors[name], value), name
    assert tensors[model_module.EMBED].data_ptr() != tensors[model_module.HEAD].data_ptr()
    assert not torch.equal(tensors[model_module.EMBED], tensors[model_module.HEAD])


_BAD_LOADS = ["missing_head", "unexpected_name", "late_expert_fp32", "late_expert_transposed"]
_BAD_LOADS += ["router_bf16", "head_not_a_tensor"]


@pytest.mark.parametrize("case", _BAD_LOADS)
def test_rejected_load_changes_nothing(case) -> None:
    """A malformed mapping is rejected before any tensor is copied, even when only its last
    tensors are wrong, and the model keeps running on what it held."""
    model, state = loaded(17)
    before = run(model, generate_ids(17, 1))
    versions = {n: t._version for n, t in model.checkpoint_tensors().items()}
    bad = OrderedDict(state)
    late = "model.layers.5.mlp.experts.255.down_proj.weight"
    if case == "missing_head":
        del bad[model_module.HEAD]
    elif case == "unexpected_name":
        bad["model.layers.6.input_layernorm.weight"] = state[model_module.NORM]
    elif case == "late_expert_fp32":
        bad[late] = state[late].float()
    elif case == "late_expert_transposed":
        bad[late] = state[late].t()
    elif case == "router_bf16":
        bad["model.layers.5.mlp.gate.weight"] = state["model.layers.5.mlp.gate.weight"].bfloat16()
    else:
        bad[model_module.HEAD] = state[model_module.HEAD].tolist
    with pytest.raises((ValueError, TypeError)):
        model.load_weights(bad)
    after = {n: t._version for n, t in model.checkpoint_tensors().items()}
    assert after == versions
    assert bitwise_equal(run(model, generate_ids(17, 1)), before)


def test_failed_copy_leaves_the_model_unavailable(monkeypatch) -> None:
    model, state = loaded(17)
    before = run(model, generate_ids(17, 17))

    def broken(weights):
        raise RuntimeError("copy failed")

    monkeypatch.setattr(model.layers[4], "load_weights", broken)
    with pytest.raises(RuntimeError, match="copy failed"):
        model.load_weights(state)
    with prefill_context(1) as md, no_dispatch(), pytest.raises(ValueError, match="load_weights"):
        model(generate_ids(17, 1), positions_of(1), md)
    monkeypatch.undo()
    model.load_weights(state)
    assert bitwise_equal(run(model, generate_ids(17, 17)), before)


# -------------------------------------------------- tests: input and cache contracts

_BAD_INPUTS = [
    "float_ids",
    "bool_ids",
    "int16_ids",
    "negative_id",
    "id_vocab_size",
    "batch_2",
    "flat_ids",
    "empty",
    "too_long",
    "host_ids",
    "non_contiguous_ids",
    "positions_offset",
    "float_positions",
    "positions_shape",
    "host_positions",
    "metadata_type",
    "cached_prefix",
    "context_length_mismatch",
    "one_layer_cache",
    "seven_layer_cache",
    "duplicate_slots",
    "slot_out_of_range",
    "missing_layer",
    "all_sliding_windows",
    "no_sliding_windows",
    "fp16_cache",
    "four_kv_heads",
]
#: Host pool tables replaced after the manager built them: rows sharing a slot's storage or
#: swapped between slots, an offset outside its pool, a fractional, int64 or misshapen
#: mapping, and pointer rows other than the manager's.
_BAD_POOL_TABLES = [
    "aliased_pool_rows",
    "swapped_pool_offsets",
    "pool_offset_out_of_range",
    "fractional_pool_offsets",
    "int64_pool_mapping",
    "extra_pool_mapping_row",
    "aliased_pool_pointers",
    "secondary_pool_pointer",
]
_BAD_INPUTS += _BAD_POOL_TABLES + ["swa_scratch_cache"]


@pytest.mark.parametrize("case", _BAD_INPUTS)
def test_rejects_unsupported_inputs_before_dispatch(case, monkeypatch) -> None:
    """Each declared restriction raises ValueError before any op or the embedding runs."""
    model, _ = loaded(17)
    length = 17
    ids = generate_ids(17, length)
    positions = positions_of(length)
    context = lambda: prefill_context(length)  # noqa: E731
    if case == "float_ids":
        ids = ids.float()
    elif case == "bool_ids":
        ids = ids > 0
    elif case == "int16_ids":
        ids = ids.to(torch.int16)
    elif case == "negative_id":
        ids = ids.clone()
        ids[0, 5] = -1
    elif case == "id_vocab_size":
        ids = ids.clone()
        ids[0, -1] = VOCAB
    elif case == "batch_2":
        ids = torch.cat((ids, ids))
        positions = torch.cat((positions, positions))
    elif case == "flat_ids":
        ids = ids.view(length)
    elif case == "empty":
        ids, positions = ids[:, :0], positions[:, :0]
    elif case == "too_long":
        ids = generate_ids(17, 129).repeat(1, 8)[:, : model_module.MAX_TOKENS + 1].contiguous()
        positions = positions_of(model_module.MAX_TOKENS + 1)
    elif case == "host_ids":
        ids = ids.cpu()
    elif case == "non_contiguous_ids":
        ids = generate_ids(17, 129).repeat(1, 2)[:, : 2 * length : 2]
    elif case == "positions_offset":
        positions = positions + 1
    elif case == "float_positions":
        positions = positions.float()
    elif case == "positions_shape":
        positions = positions.view(length)
    elif case == "host_positions":
        positions = positions.cpu()
    elif case == "cached_prefix":
        context = lambda: manager_context(length, max_seq_len=1024)  # noqa: E731
    elif case == "context_length_mismatch":
        context = lambda: prefill_context(length - 1)  # noqa: E731
    elif case == "one_layer_cache":
        context = lambda: manager_context(length, num_layers=1, max_attention_window=[512])  # noqa: E731
    elif case == "seven_layer_cache":
        context = lambda: manager_context(  # noqa: E731
            length, num_layers=7, max_attention_window=[512] * 5 + [1024, 512]
        )
    elif case == "all_sliding_windows":
        context = lambda: manager_context(length, max_attention_window=[512] * 6)  # noqa: E731
    elif case == "no_sliding_windows":
        context = lambda: manager_context(length, max_attention_window=[1024] * 6)  # noqa: E731
    elif case == "fp16_cache":
        context = lambda: manager_context(length, dtype=DataType.HALF)  # noqa: E731
    elif case == "four_kv_heads":
        context = lambda: manager_context(length, num_kv_heads=4)  # noqa: E731
    with context() as md:
        if case == "metadata_type":
            md = object()
        elif case == "cached_prefix":
            md.kv_lens_runtime[0] = length + 8
        elif case in ("duplicate_slots", "slot_out_of_range", "missing_layer"):
            offsets = dict(md.kv_cache_manager.layer_offsets)
            if case == "duplicate_slots":
                offsets[5] = offsets[4]
            elif case == "slot_out_of_range":
                offsets[5] = 6
            else:
                del offsets[5]
            monkeypatch.setattr(md.kv_cache_manager, "layer_offsets", offsets)
        elif case in _BAD_POOL_TABLES:
            mgr = md.kv_cache_manager
            mapping = mgr.kv_cache_pool_mapping.clone()
            pointers = mgr.kv_cache_pool_pointers.clone()
            if case == "aliased_pool_rows":
                mapping[4] = mapping[3]
            elif case == "swapped_pool_offsets":
                mapping[[0, 1]] = mapping[[1, 0]]
            elif case == "pool_offset_out_of_range":
                mapping[5, 1] = 1 << 20
            elif case == "fractional_pool_offsets":
                mapping = mapping.float()
                mapping[5, 1] = 0.5
            elif case == "int64_pool_mapping":
                mapping = mapping.long()
            elif case == "extra_pool_mapping_row":
                mapping = torch.cat((mapping, mapping[5:]))
            elif case == "aliased_pool_pointers":
                pointers[1, 0] = pointers[0, 0]
            else:
                pointers[0, 1] = pointers[0, 0]
            monkeypatch.setattr(mgr, "kv_cache_pool_mapping", mapping)
            monkeypatch.setattr(mgr, "kv_cache_pool_pointers", pointers)
        elif case == "swa_scratch_cache":
            monkeypatch.setattr(md.kv_cache_manager, "enable_swa_scratch_reuse", True)
        with no_dispatch(), pytest.raises(ValueError):
            model(ids, positions, md)


def test_cache_slots_are_the_managers_own_buffers() -> None:
    """The prepared cache passes check_cache, and each layer's pool row addresses the buffer
    the manager itself reports for that layer: six distinct buffers, as allocated."""
    with prefill_context(17) as md:
        model_module.check_cache(md, 17)
        mgr = md.kv_cache_manager
        mapping, pointers = md.host_kv_cache_pool_mapping, md.host_kv_cache_pool_pointers
        assert mapping.dtype == torch.int32 and pointers.dtype == torch.int64
        page = 32 * 8 * 128 * 2
        addresses = []
        for layer_idx in range(6):
            pool, offset = mapping[mgr.layer_offsets[layer_idx]].tolist()
            addresses.append(int(pointers[pool, 0]) + offset * 2 * page)
            assert addresses[-1] == mgr.get_buffers(layer_idx).data_ptr(), layer_idx
        assert len(set(addresses)) == 6


# -------------------------------------------------- tests: the composed forward


@pytest.mark.parametrize("seed,length", CASES, ids=CASE_IDS)
def test_forward_composes_the_six_blocks(seed, length) -> None:
    """One plain call and one observed call: the same logits, every element of them, and the
    composition the observation shows."""
    model, _ = loaded(seed)
    ids = generate_ids(seed, length)
    ids_before = ids.clone()
    versions = {n: t._version for n, t in model.checkpoint_tensors().items()}
    plain = run(model, ids)
    assert plain.shape == (1, length, VOCAB) and plain.dtype == torch.float32
    assert plain.device == model.device and bool(torch.isfinite(plain).all())
    assert float(plain.abs().max()) <= 30.0
    with prefill_context(length) as md, recording(model) as (calls, events):
        observed = model(ids, positions_of(length), md)
        torch.cuda.synchronize()
    assert bitwise_equal(observed, plain)
    assert torch.equal(ids, ids_before)
    assert {n: t._version for n, t in model.checkpoint_tensors().items()} == versions
    # Blocks 0..5 once each, in order, each on its predecessor's output.
    assert [(kind, i) for kind, i, _ in events] == [
        (kind, i) for i in range(6) for kind in ("enter", "exit")
    ]
    for i in range(1, 6):
        assert bitwise_equal(events[2 * i][2], events[2 * i - 1][2])
    embedded = events[0][2]
    assert embedded.shape == (1, length, HIDDEN) and embedded.dtype == torch.bfloat16
    assert bitwise_equal(embedded[0], model.embed_tokens[ids[0]])
    for i in range(6):
        block = [c for c in calls if c["block"] == i and c["op"] != "expert"]
        attention = [c for c in block if c["op"] == "ThopAttention"]
        (att,) = attention
        assert att["kwargs"]["local_layer_idx"] == i
        assert att["kwargs"]["attention_window_size"] == (512 if i < 5 else 1024)
        residuals = [c for c in block if c["op"] == "BeamDepthResidual"]
        assert [tuple(c["args"][2:4]) for c in residuals] == [
            depth_scales(i, True),
            depth_scales(i, False),
        ]
        qkv = att["qkv"]
        k_norm = [c for c in block if c["op"] == "FlashinferRmsnorm"][2]["output"]
        q_norm = [c for c in block if c["op"] == "FlashinferRmsnorm"][1]["output"]
        packed_q = qkv[:, : 48 * 128].reshape(length, 48, 128)
        packed_k = qkv[:, 48 * 128 : 56 * 128].reshape(length, 8, 128)
        if i < 5:
            assert [c["op"] for c in block].count("BeamNeoxRope") == 1
            assert "BeamNopeQueryScale" not in [c["op"] for c in block]
        else:
            assert "BeamNeoxRope" not in [c["op"] for c in block]
            assert bitwise_equal(packed_k, k_norm.view(length, 8, 128))
            assert bitwise_equal(
                packed_q,
                beam_nope_query_scale(q_norm.view(length, 48, 128), positions_of(length)[0]),
            )
    top = [c for c in calls if c["block"] is None]
    assert [c["op"] for c in top] == ["FlashinferRmsnorm", "CublasMm", "BeamLogitSoftcap"]
    norm, head, softcap = top
    assert bitwise_equal(norm["args"][0], events[-1][2].view(length, HIDDEN))
    assert bitwise_equal(head["args"][0], norm["output"])
    assert head["output"].dtype == torch.bfloat16 and head["output"].shape == (length, VOCAB)
    # The softcap, exactly once: the returned logits are this expression of the head output.
    once = (head["output"].float() / 30.0).tanh() * 30.0
    assert bitwise_equal(plain.view(length, VOCAB), once)
    assert bitwise_equal(softcap["args"][0], head["output"]) and softcap["args"][1] == 30.0


def test_layer5_query_scale_is_visible() -> None:
    """At 129 tokens the query scale changes block 5's q at later positions, so an omitted
    scale would be a different call."""
    model, _ = loaded(17)
    with prefill_context(129) as md, recording(model) as (calls, _):
        model(generate_ids(17, 129), positions_of(129), md)
        torch.cuda.synchronize()
    block = [c for c in calls if c["block"] == 5]
    q_norm = [c for c in block if c["op"] == "FlashinferRmsnorm"][1]["output"].view(129, 48, 128)
    (att,) = [c for c in block if c["op"] == "ThopAttention"]
    packed_q = att["qkv"][:, : 48 * 128].reshape(129, 48, 128)
    changed = (packed_q != q_norm).flatten(1).any(dim=1)
    assert not bool(changed[0]) and bool(changed[-1])


def test_causal_prefix_and_request_isolation() -> None:
    """Changing the tokens after the first 17 leaves the dense blocks' prefix rows and block
    3's input to its feed-forward unchanged, bit for bit; and a request between two identical
    requests changes nothing of the second."""
    model, _ = loaded(17)
    full = generate_ids(17, 129)
    other = full.clone()
    other[:, 17:] = generate_ids(41, 129)[:, 17:]

    def observe(ids):
        with prefill_context(ids.shape[1]) as md, recording(model) as (calls, events):
            out = model(ids, positions_of(ids.shape[1]), md)
            torch.cuda.synchronize()
        pre_ff = [c["output"] for c in calls if c["block"] == 3 and c["op"] == "FlashinferRmsnorm"][
            4
        ]
        return out, [x for kind, _, x in events if kind == "exit"], pre_ff

    out_a, exits_a, pre_ff_a = observe(full)
    out_b, exits_b, pre_ff_b = observe(other)
    for i in range(3):
        assert bitwise_equal(exits_a[i][:, :17], exits_b[i][:, :17])
    assert bitwise_equal(pre_ff_a[:17], pre_ff_b[:17])
    assert not torch.equal(out_a[:, 17:], out_b[:, 17:])
    first = run(model, full[:, :17])
    run(model, generate_ids(41, 17))
    assert bitwise_equal(run(model, full[:, :17]), first)

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""TurboQuant cold-page codec: distortion, unbiasedness, layouts, and the codec contract.

Every test drives the provider through the KVCM-facing interface (build_codec_state, configure,
encode_cold_pages, decode_cold_pages) with raw addresses, pinned host cold pages, and device page
indices, exactly as KVCacheManagerV2 calls it.
"""

import math
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.kv_cache_compression.quantization_for_cold_page import (
    turboquant_math as tq_math,
)
from tensorrt_llm._torch.kv_cache_compression.quantization_for_cold_page.turboquant_quantization import (
    TurboQuantColdPageQuantizationCompression,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import DataType
from tensorrt_llm.llmapi.llm_args import ColdPageQuantizationCompressionConfig
from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig, BufferConfig

# Theorem 1 (normalized MSE) and Theorem 2 (inner-product variance times d) at d = 128.
PAPER_MSE = {2: 0.117, 3: 0.034, 4: 0.0094}
PAPER_PROD_VARIANCE = {2: 0.56, 3: 0.18, 4: 0.047}
_DTYPES = {
    "bf16": (torch.bfloat16, DataType.BF16),
    "fp16": (torch.float16, DataType.HALF),
    "fp8": (torch.float8_e4m3fn, DataType.FP8),
}

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _provider(bits=4, mode="mse", seed=0, *, pretrained_config=None, skip_rope=False):
    config = ColdPageQuantizationCompressionConfig(
        quant="turboquant",
        turboquant_bits=bits,
        turboquant_mode=mode,
        turboquant_seed=seed,
        skip_rope_quantization=skip_rope,
    )
    return TurboQuantColdPageQuantizationCompression(
        config, pretrained_config=pretrained_config or SimpleNamespace(model_type="llama")
    )


class _Codec:
    """Hot slots, pinned cold pages, and one configured lifecycle of a TurboQuant provider.

    ``layers`` maps a layer id to ``{role: (heads, tokens, elements)}``; roles other than key and
    value are stored as raw bytes by the codec. Every buffer gets ``slots`` hot slots.
    """

    def __init__(
        self,
        provider,
        layers,
        *,
        runtime="bf16",
        slots=8,
        pages=8,
        cache_config=None,
        num_kv_heads=None,
        head_dim=None,
        pp_layers=None,
        is_draft=False,
    ):
        self.provider = provider
        self.dtype, data_type = _DTYPES[runtime]
        self.hot = {
            (layer_id, role): torch.zeros((slots, *shape), dtype=self.dtype, device="cuda")
            for layer_id, buffers in layers.items()
            for role, shape in buffers.items()
        }
        self.slot_bytes = {
            key: value[0].numel() * value.element_size() for key, value in self.hot.items()
        }
        if cache_config is None:
            cache_config = SimpleNamespace(
                tokens_per_block=next(iter(next(iter(layers.values())).values()))[1],
                layers=tuple(
                    AttentionLayerConfig(
                        layer_id=layer_id,
                        buffers=[
                            BufferConfig(role=role, size=self.slot_bytes[(layer_id, role)])
                            for role in buffers
                        ],
                    )
                    for layer_id, buffers in layers.items()
                ),
            )
        num_layers = max(layers) + 1
        first = {lid: next(iter(b.values())) for lid, b in layers.items()}
        self.state = provider.build_codec_state(
            cache_config,
            runtime_dtype=data_type,
            pp_layers=pp_layers or tuple(range(num_layers)),
            num_kv_heads_per_layer=num_kv_heads
            or tuple(first.get(lid, (1,))[0] for lid in range(num_layers)),
            head_dim_per_layer=head_dim
            or tuple(first.get(lid, (1, 1, 1))[2] for lid in range(num_layers)),
            is_draft=is_draft,
        )
        lifecycle = SimpleNamespace(
            layers={
                layer_id: {
                    role: SimpleNamespace(
                        raw_base=self.hot[(layer_id, role)].data_ptr(),
                        raw_slot_bytes=self.slot_bytes[(layer_id, role)],
                        raw_bytes=self.slot_bytes[(layer_id, role)],
                    )
                    for role in buffers
                }
                for layer_id, buffers in layers.items()
                if layer_id in self.state.layer_ids
            }
        )
        (properties,) = provider.configure(self.state, [lifecycle])
        self.cold_page_bytes = int(properties.cold_page_bytes)
        self.location = properties.page_index_location
        self.cold = torch.zeros(pages * self.cold_page_bytes, dtype=torch.uint8, pin_memory=True)
        self.stream = torch.cuda.Stream()

    def _pairs(self, dst, src):
        return torch.tensor(list(zip(dst, src)), dtype=torch.int32, device="cuda")

    def encode(self, pages, slots):
        pairs = self._pairs(pages, slots)
        self.provider.encode_cold_pages(
            self.state,
            0,
            self.cold.data_ptr(),
            pairs.data_ptr(),
            len(pairs),
            self.stream.cuda_stream,
        )
        self.stream.synchronize()

    def decode(self, slots, pages):
        pairs = self._pairs(slots, pages)
        self.provider.decode_cold_pages(
            self.state,
            0,
            self.cold.data_ptr(),
            pairs.data_ptr(),
            len(pairs),
            self.stream.cuda_stream,
        )
        self.stream.synchronize()


def _vectors(kind, shape, generator):
    """Inputs that are not Gaussian: TurboQuant must be oblivious to their structure."""

    dim = shape[-1]
    gaussian = torch.randn(shape, generator=generator, device="cuda")
    if kind == "gaussian":
        return gaussian
    if kind == "outlier_channels":  # like real K: a few channels much larger than the rest
        gaussian[..., 3] *= 25.0
        gaussian[..., dim // 2 + 7] *= 10.0
        return gaussian
    if kind == "sparse":  # four non-zeros per vector at random positions
        keep = torch.rand(shape, generator=generator, device="cuda").argsort(-1) < 4
        return gaussian * keep
    if kind == "one_hot":
        index = torch.randint(0, dim, shape[:-1], generator=generator, device="cuda")
        return torch.nn.functional.one_hot(index, dim).float() * 3.0
    if kind == "constant":
        return torch.full(shape, 0.75, device="cuda")
    if kind == "mixed_norms":  # norms from 1e-6 to 1e6 within one Page
        exponents = torch.randint(-6, 7, (*shape[:-1], 1), generator=generator, device="cuda")
        return gaussian * torch.pow(10.0, exponents.float())
    raise ValueError(kind)


def _nmse(original, restored):
    x, y = original.double().flatten(0, -2), restored.double().flatten(0, -2)
    norms = (x * x).sum(-1)
    keep = norms > 0
    return float((((x - y) ** 2).sum(-1)[keep] / norms[keep]).mean())


def _round_trip(codec, values, *, pages=None):
    """Write ``values[layer, role]`` into slots 0..P-1, encode to pages, decode into P..2P-1."""

    count = next(iter(values.values())).shape[0]
    for key, value in values.items():
        codec.hot[key][:count] = value.to(codec.dtype)
    pages = list(range(count)) if pages is None else pages
    codec.encode(pages, list(range(count)))
    codec.decode(list(range(count, 2 * count)), pages)
    return {key: codec.hot[key][count : 2 * count].clone() for key in values}


def _kv_codec(provider, *, heads=2, tokens=32, dim=128, layers=2, runtime="bf16", slots=8):
    return _Codec(
        provider,
        {
            layer: {"key": (heads, tokens, dim), "value": (heads, tokens, dim)}
            for layer in range(layers)
        },
        runtime=runtime,
        slots=slots,
        pages=slots,
    )


def test_codebooks_and_matrices_match_the_paper_and_the_seed():
    for bits, paper in PAPER_MSE.items():
        assert tq_math.codebook_distortion(128, bits) == pytest.approx(paper, rel=0.05)
    rotation = tq_math.haar_rotation(0, 576)
    assert torch.allclose(rotation @ rotation.T, torch.eye(576, dtype=torch.float64), atol=1e-10)
    assert torch.equal(rotation, tq_math.haar_rotation.__wrapped__(0, 576))
    assert not torch.equal(rotation, tq_math.haar_rotation.__wrapped__(1, 576))
    centroids, thresholds = tq_math.lloyd_max_codebook(128, 3)
    assert torch.allclose(centroids, -centroids.flip(0), atol=1e-9)
    assert torch.all(thresholds[1:] > thresholds[:-1])


@requires_cuda
@pytest.mark.parametrize("bits", (2, 3, 4))
@pytest.mark.parametrize(
    "kind", ("gaussian", "outlier_channels", "sparse", "one_hot", "constant", "mixed_norms")
)
def test_mse_distortion_is_data_oblivious(bits, kind):
    codec = _kv_codec(_provider(bits))
    generator = torch.Generator(device="cuda").manual_seed(bits)
    values = {key: _vectors(kind, (4, 2, 32, 128), generator) for key in codec.hot}
    restored = _round_trip(codec, values)
    error = sum(_nmse(values[key].to(codec.dtype), restored[key]) for key in values) / len(values)
    # The constant vector has a single direction, so its error is one sample of the distribution.
    tolerance = 1.6 if kind == "constant" else 1.15
    assert error < tolerance * PAPER_MSE[bits], error


@requires_cuda
@pytest.mark.parametrize("runtime", ("bf16", "fp16", "fp8"))
def test_zero_tiny_and_huge_vectors_stay_finite(runtime):
    codec = _kv_codec(_provider(4), runtime=runtime)
    generator = torch.Generator(device="cuda").manual_seed(0)
    scales = {"bf16": (0.0, 1e-30, 1e30), "fp16": (0.0, 1e-4, 6e3), "fp8": (0.0, 1e-2, 1e2)}[
        runtime
    ]
    for scale in scales:
        values = {
            key: torch.randn((4, 2, 32, 128), generator=generator, device="cuda") * scale
            for key in codec.hot
        }
        if runtime == "fp8":
            values = {key: value.clamp(-448, 448) for key, value in values.items()}
        restored = _round_trip(codec, values)
        for key, value in values.items():
            original = value.to(codec.dtype)
            if scale == 0.0:
                assert torch.equal(restored[key].float(), torch.zeros_like(value))
            else:
                assert torch.isfinite(restored[key].float()).all()
                assert _nmse(original, restored[key]) < 1.2 * PAPER_MSE[4] + (
                    0.003 if runtime == "fp8" else 0
                )


@requires_cuda
@pytest.mark.parametrize("bits", (2, 3, 4))
def test_prod_inner_products_are_unbiased(bits):
    dim, count = 128, 16 * 2 * 32
    codec = _kv_codec(_provider(bits, "prod"), slots=32, layers=1)
    generator = torch.Generator(device="cuda").manual_seed(10 + bits)
    x = _vectors("outlier_channels", (16, 2, 32, dim), generator)
    x = x / x.norm(dim=-1, keepdim=True)
    restored = _round_trip(codec, {(0, "key"): x, (0, "value"): x})[(0, "key")].double()
    y = torch.randn((16, 2, 32, dim), generator=generator, device="cuda", dtype=torch.float64)
    y = y / y.norm(dim=-1, keepdim=True)
    error = ((restored - x.to(torch.bfloat16).double()) * y).sum(-1).flatten()
    assert abs(float(error.mean())) < 4 * float(error.std()) / math.sqrt(count)
    variance = float(error.var()) * dim
    assert 0.6 * PAPER_PROD_VARIANCE[bits] < variance < 1.4 * PAPER_PROD_VARIANCE[bits], variance


@requires_cuda
def test_prod_is_unbiased_over_seeds_for_one_pair():
    """E over the random rotation and projection of <y, x_hat> equals <y, x> (Theorem 2)."""

    generator = torch.Generator(device="cuda").manual_seed(5)
    x = _vectors("outlier_channels", (1, 1, 32, 128), generator)
    x = (x / x.norm(dim=-1, keepdim=True)).expand(2, 1, 32, 128).contiguous()
    y = torch.randn(128, generator=generator, device="cuda", dtype=torch.float64)
    y /= y.norm()
    errors = []
    for seed in range(24):
        codec = _Codec(_provider(2, "prod", seed), {0: {"key": (1, 32, 128)}}, slots=4, pages=2)
        restored = _round_trip(codec, {(0, "key"): x})[(0, "key")].double()
        errors.append(((restored - x.to(torch.bfloat16).double()) @ y)[0, 0])
    errors = torch.cat(errors)  # 24 seeds x 32 identical vectors
    per_seed = errors.view(24, 32).mean(1)
    assert abs(float(per_seed.mean())) < 4 * float(per_seed.std()) / math.sqrt(24)


@requires_cuda
@pytest.mark.parametrize("bits", (2.5, 3.5))
@pytest.mark.parametrize("kind", ("gaussian", "outlier_channels"))
def test_fractional_bits_lie_between_their_integer_neighbours(bits, kind):
    errors = {}
    for setting in (math.floor(bits), bits, math.ceil(bits)):
        codec = _kv_codec(_provider(setting))
        generator = torch.Generator(device="cuda").manual_seed(3)
        values = {key: _vectors(kind, (4, 2, 32, 128), generator) for key in codec.hot}
        restored = _round_trip(codec, values)
        errors[setting] = sum(_nmse(values[k].to(codec.dtype), restored[k]) for k in values) / len(
            values
        )
    assert errors[math.ceil(bits)] < errors[bits] < errors[math.floor(bits)], errors
    if kind == "outlier_channels":  # the large channels land in the high-bit group
        assert errors[bits] < 0.5 * (errors[math.ceil(bits)] + errors[math.floor(bits)])


@requires_cuda
@pytest.mark.parametrize(
    "bits,mode,runtime,limit",
    (
        (4, "mse", "bf16", 1e-3),
        (3, "mse", "fp16", 1e-3),
        (2, "mse", "bf16", 1e-3),
        (4, "mse", "fp8", 2e-2),
        (3.5, "mse", "bf16", 0.1),
        (2.5, "mse", "bf16", 0.1),
        (4, "prod", "bf16", None),
    ),
)
def test_repeated_round_trips(bits, mode, runtime, limit):
    """Re-encoding a decoded Page: integer-bit mse is stable; prod accumulates fresh QJL noise."""

    codec = _kv_codec(_provider(bits, mode), runtime=runtime)
    generator = torch.Generator(device="cuda").manual_seed(4)
    values = {key: _vectors("outlier_channels", (4, 2, 32, 128), generator) for key in codec.hot}
    for key, value in values.items():
        codec.hot[key][:4] = value.to(codec.dtype)
    errors = []
    for trip in range(4):
        codec.encode([0, 1, 2, 3], [0, 1, 2, 3] if trip == 0 else [4, 5, 6, 7])
        codec.decode([4, 5, 6, 7], [0, 1, 2, 3])
        errors.append(sum(_nmse(v.to(codec.dtype), codec.hot[k][4:]) for k, v in values.items()))
    if limit is None:
        assert errors[1] > errors[0]  # documented: prod is unbiased per round trip, not idempotent
    else:
        assert max(errors) <= errors[0] * (1 + limit), errors


@requires_cuda
@pytest.mark.parametrize(
    "bits,mode,code_bits",
    (
        (2, "mse", 2),
        (3, "mse", 3),
        (4, "mse", 4),
        (2, "prod", 2),
        (3, "prod", 3),
        (4, "prod", 4),
        (2.5, "mse", 2.5),
        (3.5, "mse", 3.5),
    ),
)
def test_cold_page_bytes_stay_within_the_bit_budget(bits, mode, code_bits):
    codec = _kv_codec(_provider(bits, mode), heads=8, layers=2)
    vectors, elements = 2 * 2 * 8 * 32, 2 * 2 * 8 * 32 * 128
    side_bits = (codec.cold_page_bytes * 8 - code_bits * elements) / vectors
    assert 0 < side_bits <= 64, side_bits


@requires_cuda
def test_encoding_is_deterministic_and_seeded():
    generator = torch.Generator(device="cuda").manual_seed(6)
    values = {
        key: _vectors("outlier_channels", (4, 2, 32, 128), generator)
        for key in ((0, "key"), (0, "value"), (1, "key"), (1, "value"))
    }
    blobs = []
    for seed in (0, 0, 1):
        codec = _kv_codec(_provider(3, seed=seed))
        _round_trip(codec, values)
        blobs.append(codec.cold.clone())
        codec.encode([0, 1, 2, 3], [0, 1, 2, 3])
        assert torch.equal(codec.cold, blobs[-1])
    assert torch.equal(blobs[0], blobs[1])
    assert not torch.equal(blobs[0], blobs[2])


@requires_cuda
@pytest.mark.parametrize("bits,mode", ((4, "mse"), (2.5, "mse"), (3, "prod")))
def test_mla_latent_with_a_lossless_auxiliary_buffer(bits, mode):
    """A 576-wide key-only latent (not a power of two) plus raw DSA index keys."""

    layers = {layer: {"key": (1, 32, 576), "index_key": (1, 32, 132)} for layer in range(3)}
    codec = _Codec(_provider(bits, mode), layers, num_kv_heads=(1,) * 3, head_dim=(576,) * 3)
    generator = torch.Generator(device="cuda").manual_seed(7)
    values = {
        key: _vectors("outlier_channels", (4, *codec.hot[key].shape[1:]), generator)
        for key in codec.hot
    }
    restored = _round_trip(codec, values)
    bound = {4: 1.15 * PAPER_MSE[4], 2.5: PAPER_MSE[2], 3: 0.25}[bits]
    for (layer, role), value in values.items():
        if role == "index_key":
            assert torch.equal(
                restored[(layer, role)].view(torch.uint8), value.to(codec.dtype).view(torch.uint8)
            )
        else:
            assert _nmse(value.to(codec.dtype), restored[(layer, role)]) < bound


@requires_cuda
@pytest.mark.parametrize(
    "pretrained_config,roles,head_dim,rope",
    (
        (
            dict(model_type="glm_moe_dsa", kv_lora_rank=512, qk_rope_head_dim=64),
            ("key",),
            576,
            slice(512, 576),
        ),
        (
            dict(model_type="qwen3_5", head_dim=256, partial_rotary_factor=0.25),
            ("key", "value"),
            256,
            slice(0, 64),
        ),
    ),
)
def test_skip_rope_quantization_keeps_rope_bytes(pretrained_config, roles, head_dim, rope):
    model_type = pretrained_config.pop("model_type")
    config = PretrainedConfig(hidden_size=2048, num_attention_heads=8, **pretrained_config)
    config.model_type = model_type
    provider = _provider(4, pretrained_config=config, skip_rope=True)
    codec = _Codec(
        provider,
        {0: {role: (1, 32, head_dim) for role in roles}},
        num_kv_heads=(1,),
        head_dim=(head_dim,),
    )
    generator = torch.Generator(device="cuda").manual_seed(8)
    values = {key: _vectors("gaussian", (2, 1, 32, head_dim), generator) for key in codec.hot}
    restored = _round_trip(codec, values)
    key = values[(0, "key")].to(codec.dtype)
    assert torch.equal(
        restored[(0, "key")][..., rope].view(torch.uint8), key[..., rope].view(torch.uint8)
    )
    nope = torch.ones(head_dim, dtype=torch.bool)
    nope[rope] = False
    assert 0 < _nmse(key[..., nope], restored[(0, "key")][..., nope]) < 1.15 * PAPER_MSE[4]
    if "value" in roles:  # V carries no position: always quantized whole
        assert (
            _nmse(values[(0, "value")].to(codec.dtype), restored[(0, "value")])
            < 1.15 * PAPER_MSE[4]
        )


@requires_cuda
@pytest.mark.parametrize("bits", (4, 3.5))
def test_deepseek_v4_csa_rows_quantize_nope_and_keep_rope_and_indexer(bits):
    """448 NoPE values of each CSA row are quantized; RoPE, indexer and SWA bytes are not."""

    cache_config = SimpleNamespace(
        tokens_per_block=128,
        layers=(
            AttentionLayerConfig(
                layer_id=0, buffers=[BufferConfig(role="deepseek_v4_swa", size=128 * 512 * 2)]
            ),
            AttentionLayerConfig(
                layer_id=1,
                buffers=[
                    BufferConfig(role="deepseek_v4_compress", size=32 * 512 * 2),
                    BufferConfig(role="deepseek_v4_indexer_compress", size=32 * 68),
                ],
            ),
        ),
    )
    provider = _provider(bits, pretrained_config=SimpleNamespace(model_type="deepseek_v4"))
    layers = {
        1: {"deepseek_v4_compress": (1, 32, 512), "deepseek_v4_indexer_compress": (1, 32, 34)}
    }
    codec = _Codec(
        provider,
        layers,
        cache_config=cache_config,
        num_kv_heads=(1, 1),
        head_dim=(512, 512),
        pp_layers=(0,),
    )
    assert codec.state.layer_ids == (1,)  # the SWA lifecycle keeps the lossless default codec
    generator = torch.Generator(device="cuda").manual_seed(9)
    values = {
        key: _vectors("outlier_channels", (4, *codec.hot[key].shape[1:]), generator)
        for key in codec.hot
    }
    restored = _round_trip(codec, values)
    row = values[(1, "deepseek_v4_compress")].to(codec.dtype)
    out = restored[(1, "deepseek_v4_compress")]
    assert torch.equal(out[..., 448:].view(torch.uint8), row[..., 448:].view(torch.uint8))
    assert _nmse(row[..., :448], out[..., :448]) < 1.15 * PAPER_MSE[3]
    indexer = values[(1, "deepseek_v4_indexer_compress")].to(codec.dtype)
    assert torch.equal(
        restored[(1, "deepseek_v4_indexer_compress")].view(torch.uint8), indexer.view(torch.uint8)
    )


@requires_cuda
@pytest.mark.parametrize("runtime", ("fp16", "fp8"))
def test_mixed_head_dims_and_runtime_dtypes(runtime):
    """One lifecycle with layers of two head dims decodes into the same runtime dtype."""

    layers = {
        0: {"key": (2, 32, 128), "value": (2, 32, 128)},
        1: {"key": (4, 32, 64), "value": (4, 32, 64)},
    }
    codec = _Codec(_provider(4), layers, runtime=runtime, num_kv_heads=(2, 4), head_dim=(128, 64))
    generator = torch.Generator(device="cuda").manual_seed(11)
    values = {
        key: _vectors("outlier_channels", (4, *codec.hot[key].shape[1:]), generator).clamp(
            -400, 400
        )
        for key in codec.hot
    }
    restored = _round_trip(codec, values)
    extra = 0.004 if runtime == "fp8" else 0.0  # FP8 rounding of the decoded values
    for key, value in values.items():
        assert restored[key].dtype == codec.dtype
        assert _nmse(value.to(codec.dtype), restored[key]) < 1.15 * PAPER_MSE[4] + extra


@requires_cuda
@pytest.mark.parametrize("bits", (4, 2.5))
def test_partial_page_garbage_does_not_leak_into_valid_tokens(bits):
    codec = _kv_codec(_provider(bits))
    generator = torch.Generator(device="cuda").manual_seed(12)
    values = {key: _vectors("outlier_channels", (4, 2, 32, 128), generator) for key in codec.hot}
    for value in values.values():  # tokens 20..31 of every Page were never written
        value[:, :, 20:24] = float("nan")
        value[:, :, 24:28] = float("inf")
        value[:, :, 28:] = 0.0
    restored = _round_trip(codec, values)
    for key, value in values.items():
        error = _nmse(value[:, :, :20].to(codec.dtype), restored[key][:, :, :20])
        assert error < (1.15 * PAPER_MSE[4] if bits == 4 else 0.5 * PAPER_MSE[2]), error
        assert torch.equal(restored[key][:, :, 28:].float(), torch.zeros_like(value[:, :, 28:]))


@requires_cuda
def test_large_shuffled_batch_matches_page_by_page_encoding():
    pages = 4096
    codec = _Codec(_provider(3), {0: {"key": (1, 32, 128)}}, slots=2 * pages, pages=pages)
    generator = torch.Generator(device="cuda").manual_seed(13)
    codec.hot[(0, "key")].copy_(
        _vectors("outlier_channels", codec.hot[(0, "key")].shape, generator)
    )
    shuffle = torch.randperm(2 * pages, generator=torch.Generator().manual_seed(1))[:pages].tolist()
    destination = torch.randperm(pages, generator=torch.Generator().manual_seed(2)).tolist()
    codec.encode(destination, shuffle)
    batched = codec.cold.clone()
    codec.cold.zero_()
    for chunk in range(0, pages, 512):
        codec.encode(destination[chunk : chunk + 512], shuffle[chunk : chunk + 512])
    assert torch.equal(codec.cold, batched)


@requires_cuda
def test_writes_only_target_pages_and_slots():
    codec = _kv_codec(_provider(4), slots=8)
    generator = torch.Generator(device="cuda").manual_seed(14)
    for value in codec.hot.values():
        value.copy_(torch.randn(value.shape, generator=generator, device="cuda"))
    codec.cold.fill_(0xA5)
    before_hot = {key: value.clone() for key, value in codec.hot.items()}
    codec.encode([5, 2], [3, 6])
    page = codec.cold.view(8, codec.cold_page_bytes)
    untouched = [i for i in range(8) if i not in (5, 2)]
    assert torch.all(page[untouched] == 0xA5)
    codec.decode([1, 7], [5, 2])
    for key, value in codec.hot.items():
        changed = [i for i in range(8) if not torch.equal(value[i], before_hot[key][i])]
        assert changed == [1, 7]


@requires_cuda
def test_codec_enqueues_on_the_given_stream_without_host_synchronization():
    codec = _kv_codec(_provider(4))
    assert codec.location == type(codec.location).DEVICE  # no host index array is read
    pairs = codec._pairs([0, 1], [0, 1])
    torch.cuda.synchronize()
    with torch.cuda.stream(codec.stream):
        torch.cuda._sleep(200_000_000)  # keep the stream busy far longer than the call takes
        codec.provider.encode_cold_pages(
            codec.state, 0, codec.cold.data_ptr(), pairs.data_ptr(), 2, codec.stream.cuda_stream
        )
        assert not codec.stream.query()
    codec.stream.synchronize()

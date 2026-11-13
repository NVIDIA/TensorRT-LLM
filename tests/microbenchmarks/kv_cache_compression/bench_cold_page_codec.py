# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Encode/decode time of the cold-page KV codecs, against NVFP4 and the GPU's speed of light.

A cold-page quantization provider (``ColdPageQuantizationCompressionConfig``) encodes KV Pages
from the GPU hot cache into cold pages in pinned host memory when KVCM offloads them, and
decodes them back when KVCM onboards them. This benchmark times TurboQuant
(``quant="turboquant"``) and, as the reference, NVFP4 (``quant="nvfp4"``) on these KVCM
lifecycles (no model weights needed):

    case                       KV    TurboQuant  pages/call  geometry
    glm5-bf16-mse4             bf16  mse 4-bit   256         GLM-5 MLA: 78 x {K} x 1 x 32 x 576
    glm5-bf16-mse3             bf16  mse 3-bit   256         as glm5
    glm5-skiprope-bf16-mse4    bf16  mse 4-bit   256         as glm5, skip_rope_quantization
    mla-fp8-mse4               fp8   mse 4-bit   256         DeepSeek-V3.2 / Kimi-K3 MLA: 61 x {K} x 1 x 32 x 576
    dsv4-csa-bf16-mse4         bf16  mse 4-bit   256         DeepSeek-V4 CSA: 18 x {compressed, indexer} x 1 x 32
                                                             x {512, 34}, 128-token Pages
    dsv4-csa-skiprope-bf16-mse4  bf16  mse 4-bit   256       as dsv4-csa, skip_rope_quantization
    llama8b-bf16-mse4          bf16  mse 4-bit   256         Llama-3.1-8B GQA: 32 x {K, V} x 8 x 32 x 128
    llama8b-bf16-prod4         bf16  prod 4-bit  256         as llama8b
    gqa-hd64-bf16-mse4         bf16  mse 4-bit   256         GQA: 36 x {K, V} x 8 x 32 x 64
    glm5-bf16-mse4-8pages      bf16  mse 4-bit   8           as glm5; a latency case

Geometry is layers x buffers x KV heads x rows per Page x elements per row. The MLA latent is 512
NoPE + 64 RoPE elements; with skip_rope_quantization both formats copy the 64 RoPE elements. A
DeepSeek-V4 CSA Page holds one compressed row per 4 of its 128 tokens: 448 NoPE elements, which
both formats quantize, and 64 RoPE elements, which TurboQuant always copies (NVFP4 quantizes them
unless skip_rope_quantization is set, so dsv4-csa-skiprope compares like with like); the 68-byte FP4
indexer row (34 BF16 elements) is copied too. Each CSA layer follows its SWA layer, which another
lifecycle stores, so only the CSA layers are timed. The latency case is fixed-cost bound and is
compared with NVFP4.

Method. Each format builds its provider through the public interface
(``create_kv_cache_compression_manager`` -> ``build_codec_state`` -> ``configure``) over its own
hot buffers: one ``[2N, KV heads, tokens, elements]`` GPU tensor per (layer, buffer), for N pages
per call. Slots 0..N-1 hold fixed-seed Gaussian data with a few 10-20x channels, like real K
(clamped to +-448 for FP8). Encode packs them, in a fixed shuffled order, into N pinned host cold
pages; decode unpacks the cold pages into slots N..2N-1. The ``(dst, src)`` page index pairs
live where ``configure`` asks (host or device). Each call is timed alone on a non-default stream
between two CUDA events recorded around the Python call, so its host overhead counts: 3
warmups, then the median of 20. The formats alternate call by call, so clock drift (clocks are
not locked) affects them alike.

Speed of light (SOL). A Triton copy kernel over 1 GiB measures the HBM read and write bandwidth
(bytes read / time and bytes written / time of a device copy) and the bandwidth of the GPU
reading and writing pinned host memory directly (zero-copy). Encode SOL = max(hot bytes / HBM
read, cold bytes / host write); decode SOL = max(cold bytes / host read, hot bytes / HBM write);
cold bytes are N x the format's own ``cold_page_bytes``.

Correctness. After timing, each decoded slot is compared with its source slot: nMSE is
||x - x_hat||^2 / ||x||^2, averaged over the K/V vectors of the case and of each (layer, buffer).
A format FAILS when an nMSE is not finite or one buffer's nMSE is above four times the case-wide
nMSE (a layer that decodes to zeros has nMSE 1, which a case-wide mean over many layers would
hide); TurboQuant mse also FAILS when its case-wide nMSE is above twice the paper's distortion
(0.0094, 0.034, 0.117 at 4, 3, 2 bits). Results that fail do not count. Cases with an FP8
active KV cache are skipped on GPUs below compute capability 8.9, where TurboQuant does not admit
it. Non-finite values are written to --json as null.

Run on one GPU, from the repository root::

    python3 tests/microbenchmarks/kv_cache_compression/bench_cold_page_codec.py \
        [--cases glm5-bf16-mse4,dsv4-csa-bf16-mse4] [--formats turboquant,nvfp4] \
        [--warmup 3] [--reps 20] [--json results.json] [--profile]

``--profile`` also lists the GPU kernels and copies of one encode and one decode per format and
case, to confirm which kernels the default path runs. A format that cannot be built or run for a
case prints its exception and the benchmark moves on; the exit status is 1 if any requested
measurement errored or failed its check.
"""

import argparse
import itertools
import json
import math
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import torch
import triton
import triton.language as tl
from transformers import PretrainedConfig

from tensorrt_llm._torch.pyexecutor._util import create_kv_cache_compression_manager
from tensorrt_llm._torch.pyexecutor.resource_manager import DataType
from tensorrt_llm.bindings.internal import kv_cache_compression as native
from tensorrt_llm.llmapi.llm_args import ColdPageQuantizationCompressionConfig, KvCacheConfig
from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig, BufferConfig

TOKENS_PER_BLOCK = 32
FORMATS = ("turboquant", "nvfp4")
OPS = ("encode", "decode")
SEED = 0
PAPER_MSE = {2: 0.117, 3: 0.034, 4: 0.0094}  # TurboQuant Theorem 1 distortion, d = 128
SOL_BYTES = 1 << 30
COPY_BLOCK = 4096
RUNTIMES = {"bf16": (torch.bfloat16, DataType.BF16), "fp8": (torch.float8_e4m3fn, DataType.FP8)}
WORST_BUFFER_FACTOR = 4.0  # no (layer, buffer) may exceed this multiple of the case-wide nMSE


@dataclass(frozen=True)
class Case:
    """One KVCM lifecycle: page geometry, runtime KV dtype, TurboQuant setting, pages per call."""

    name: str
    model: str
    model_type: str
    layers: int
    roles: tuple[str, ...]
    kv_heads: int
    head_dim: int
    runtime: str
    mode: str
    bits: int
    pages: int
    latency: bool = False
    skip_rope: bool = False
    tokens_per_block: int = TOKENS_PER_BLOCK

    @property
    def deepseek_v4(self) -> bool:
        return self.model_type == "deepseek_v4"

    def slot_shape(self, role: str) -> tuple[int, int, int]:
        """[KV heads, rows, elements] of one Page of ``role``."""
        if self.deepseek_v4:
            elements = self.head_dim if role == _DSV4_COMPRESS else _DSV4_INDEXER_ELEMENTS
            return (1, self.tokens_per_block // 4, elements)
        return (self.kv_heads, self.tokens_per_block, self.head_dim)

    def slot_bytes(self, role: str) -> int:
        return math.prod(self.slot_shape(role)) * RUNTIMES[self.runtime][0].itemsize

    @property
    def page_elements(self) -> int:
        return self.layers * sum(math.prod(self.slot_shape(role)) for role in self.roles)

    @property
    def hot_page_bytes(self) -> int:
        return self.layers * sum(self.slot_bytes(role) for role in self.roles)


_DSV4_SWA = "deepseek_v4_swa"
_DSV4_COMPRESS = "deepseek_v4_compress"
_DSV4_INDEXER = "deepseek_v4_indexer_compress"
_DSV4_INDEXER_ELEMENTS = 34  # the FP4 indexer row: 64 bytes of codes and 4 bytes of scales
_GLM_5 = {
    "model": "GLM-5 MLA latent (512 NoPE + 64 RoPE)",
    "model_type": "glm_moe_dsa",
    "layers": 78,
    "roles": ("key",),
    "kv_heads": 1,
    "head_dim": 576,
}
_DSV4_CSA = {
    "model": "DeepSeek-V4 CSA rows (448 NoPE + 64 RoPE) and indexer rows",
    "model_type": "deepseek_v4",
    "layers": 18,
    "roles": (_DSV4_COMPRESS, _DSV4_INDEXER),
    "kv_heads": 1,
    "head_dim": 512,
    "tokens_per_block": 128,
}
_LLAMA_8B = {
    "model": "Llama-3.1-8B",
    "model_type": "llama",
    "layers": 32,
    "roles": ("key", "value"),
    "kv_heads": 8,
    "head_dim": 128,
}
_MLA = {
    "model": "DeepSeek-V3.2 / Kimi-K3 MLA latent (512 NoPE + 64 RoPE)",
    "model_type": "deepseek_v3",
    "layers": 61,
    "roles": ("key",),
    "kv_heads": 1,
    "head_dim": 576,
}
_GQA_64 = {
    "model": "GQA with head_dim 64",
    "model_type": "llama",
    "layers": 36,
    "roles": ("key", "value"),
    "kv_heads": 8,
    "head_dim": 64,
}
CASES = {
    case.name: case
    for case in (
        Case("glm5-bf16-mse4", **_GLM_5, runtime="bf16", mode="mse", bits=4, pages=256),
        Case("glm5-bf16-mse3", **_GLM_5, runtime="bf16", mode="mse", bits=3, pages=256),
        Case(
            "glm5-skiprope-bf16-mse4",
            **_GLM_5,
            runtime="bf16",
            mode="mse",
            bits=4,
            pages=256,
            skip_rope=True,
        ),
        Case("mla-fp8-mse4", **_MLA, runtime="fp8", mode="mse", bits=4, pages=256),
        Case("dsv4-csa-bf16-mse4", **_DSV4_CSA, runtime="bf16", mode="mse", bits=4, pages=256),
        Case(
            "dsv4-csa-skiprope-bf16-mse4",
            **_DSV4_CSA,
            runtime="bf16",
            mode="mse",
            bits=4,
            pages=256,
            skip_rope=True,
        ),
        Case("llama8b-bf16-mse4", **_LLAMA_8B, runtime="bf16", mode="mse", bits=4, pages=256),
        Case("llama8b-bf16-prod4", **_LLAMA_8B, runtime="bf16", mode="prod", bits=4, pages=256),
        Case("gqa-hd64-bf16-mse4", **_GQA_64, runtime="bf16", mode="mse", bits=4, pages=256),
        Case(
            "glm5-bf16-mse4-8pages",
            **_GLM_5,
            runtime="bf16",
            mode="mse",
            bits=4,
            pages=8,
            latency=True,
        ),
    )
}


def _model_engine(case: Case) -> SimpleNamespace:
    """A stand-in model engine with the fields ``create_kv_cache_compression_manager`` reads."""
    if case.model_type in ("deepseek_v3", "glm_moe_dsa"):
        fields = {"kv_lora_rank": 512, "qk_rope_head_dim": 64}
    elif case.deepseek_v4:
        fields = {}
    else:
        fields = {"num_key_value_heads": case.kv_heads, "head_dim": case.head_dim}
    pretrained_config = PretrainedConfig(num_hidden_layers=case.layers, **fields)
    pretrained_config.model_type = case.model_type
    return SimpleNamespace(
        mapping=SimpleNamespace(has_cp_helix=lambda: False),
        cuda_graph_batch_sizes=[],
        cuda_graph_padding_enabled=False,
        spec_config=None,
        model=SimpleNamespace(
            model_config=SimpleNamespace(quant_config=None, pretrained_config=pretrained_config)
        ),
    )


def _layer_ids(case: Case) -> range:
    """KVCM layer ids of the timed lifecycle: every layer, or the CSA layers after each SWA layer."""
    return range(1, 2 * case.layers, 2) if case.deepseek_v4 else range(case.layers)


def _make_source(case: Case) -> dict[tuple[int, str], torch.Tensor]:
    """Fixed-seed source slots ``[pages, KV heads, tokens, elements]`` per (layer, role)."""
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    dtype = RUNTIMES[case.runtime][0]
    source = {}
    for layer in _layer_ids(case):
        for role in case.roles:
            shape = case.slot_shape(role)
            outliers = {3: 20.0, shape[-1] // 2 + 1: 15.0, shape[-1] - 2: 10.0}
            values = torch.randn((case.pages, *shape), generator=generator, device="cuda")
            for channel, scale in outliers.items():
                values[..., channel] *= scale
            if dtype == torch.float8_e4m3fn:
                values = values.clamp(-448.0, 448.0)
            source[(layer, role)] = values.to(dtype)
    return source


@dataclass
class _Codec:
    """One format's provider, configured over its own hot buffers of a case."""

    manager: object
    state: object
    hot: dict[tuple[int, str], torch.Tensor]
    cold: torch.Tensor
    pairs: dict[str, torch.Tensor]
    cold_page_bytes: int
    index_location: str

    def run(self, op: str, stream: torch.cuda.Stream) -> None:
        """Encode the source slots into the cold pages, or decode those into slots N..2N-1."""
        call = self.manager.encode_cold_pages if op == "encode" else self.manager.decode_cold_pages
        pairs = self.pairs[op]
        call(self.state, 0, self.cold.data_ptr(), pairs.data_ptr(), len(pairs), stream.cuda_stream)


def _build_codec(case: Case, fmt: str, source: dict, perm: torch.Tensor) -> _Codec:
    """Build and configure ``fmt``'s provider over fresh hot buffers holding ``source``."""
    options = {"turboquant_bits": case.bits, "turboquant_mode": case.mode}
    manager = create_kv_cache_compression_manager(
        ColdPageQuantizationCompressionConfig(
            quant=fmt,
            skip_rope_quantization=case.skip_rope,
            **(options if fmt == "turboquant" else {}),
        ),
        model_engine=_model_engine(case),
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=True,
            host_cache_size=1 << 30,
            dtype="fp8" if case.runtime == "fp8" else "auto",  # admission sees the measured dtype
        ),
    )
    if not getattr(manager, "provides_cold_page_codec", False):
        raise RuntimeError(f"the factory returned {manager!r}, not a cold-page codec provider")

    pages = case.pages
    hot = {}
    for key, values in source.items():
        # NaN in the decode half exposes any slot that decode leaves unwritten.
        hot[key] = torch.full(
            (2 * pages, *values.shape[1:]), math.nan, dtype=values.dtype, device="cuda"
        )
        hot[key][:pages] = values
    layers = [
        AttentionLayerConfig(
            layer_id=layer,
            buffers=[BufferConfig(role=role, size=case.slot_bytes(role)) for role in case.roles],
        )
        for layer in _layer_ids(case)
    ]
    kvcm_layers = len(layers)
    if case.deepseek_v4:
        # Every model layer starts with its SWA layer, which is stored by another lifecycle.
        swa_bytes = case.tokens_per_block * case.head_dim * RUNTIMES[case.runtime][0].itemsize
        swa = [BufferConfig(role=_DSV4_SWA, size=swa_bytes)]
        layers = [
            entry
            for layer in layers
            for entry in (AttentionLayerConfig(layer_id=layer.layer_id - 1, buffers=swa), layer)
        ]
        kvcm_layers = 2 * case.layers
    cache_config = SimpleNamespace(tokens_per_block=case.tokens_per_block, layers=tuple(layers))
    state = manager.build_codec_state(
        cache_config,
        runtime_dtype=RUNTIMES[case.runtime][1],
        pp_layers=tuple(range(case.layers)),
        num_kv_heads_per_layer=(case.kv_heads,) * kvcm_layers,
        head_dim_per_layer=(case.head_dim,) * kvcm_layers,
        is_draft=False,
    )
    lifecycle = SimpleNamespace(
        layers={
            layer: {
                role: SimpleNamespace(
                    raw_base=hot[(layer, role)].data_ptr(),
                    raw_slot_bytes=case.slot_bytes(role),
                    raw_bytes=case.slot_bytes(role),
                )
                for role in case.roles
            }
            for layer in _layer_ids(case)
        }
    )
    (properties,) = manager.configure(state, [lifecycle])
    cold_page_bytes = int(properties.cold_page_bytes)
    location = properties.page_index_location
    host, device = native.ColdPageIndexLocation.HOST, native.ColdPageIndexLocation.DEVICE
    if cold_page_bytes <= 0 or location not in (host, device):
        raise RuntimeError(f"configure returned cold_page_bytes={cold_page_bytes}, {location}")
    slots = torch.arange(pages, dtype=torch.int32)
    index_device = "cpu" if location == host else "cuda"
    pairs = {
        "encode": torch.stack([slots, perm], 1).to(index_device),  # cold page i <- slot perm[i]
        "decode": torch.stack([slots + pages, slots], 1).to(index_device),  # slot N+i <- page i
    }
    cold = torch.empty(pages * cold_page_bytes, dtype=torch.uint8, pin_memory=True)
    cold.fill_(0xFF)  # stale bytes would hide an encode that leaves cold pages unwritten
    index_location = "host" if location == host else "device"
    return _Codec(manager, state, hot, cold, pairs, cold_page_bytes, index_location)


def _time_us(fn: Callable[[], None], stream: torch.cuda.Stream) -> float:
    """Microseconds between CUDA events recorded on ``stream`` just before and after ``fn()``."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    fn()
    end.record(stream)
    end.synchronize()
    return start.elapsed_time(end) * 1e3


@triton.jit
def _copy_kernel(src_ptr, dst_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    tl.store(dst_ptr + offsets, tl.load(src_ptr + offsets))


def _measure_bandwidth(stream: torch.cuda.Stream, warmup: int, reps: int) -> dict[str, float]:
    """GB/s of a Triton copy kernel: HBM read and write, the GPU reading and writing pinned host."""
    numel = SOL_BYTES // 4
    device_src = torch.empty(numel, dtype=torch.int32, device="cuda")
    device_dst = torch.empty_like(device_src)
    host = torch.empty(numel, dtype=torch.int32, pin_memory=True)

    def gbps(src: torch.Tensor, dst: torch.Tensor) -> float:
        grid = (numel // COPY_BLOCK,)
        samples = [
            _time_us(lambda: _copy_kernel[grid](src, dst, BLOCK=COPY_BLOCK), stream)
            for _ in range(warmup + reps)
        ]
        return SOL_BYTES / statistics.median(samples[warmup:]) / 1e3

    host_read, host_write = gbps(host, device_dst), gbps(device_src, host)
    # Last, once the GPU has left its idle clocks; a copy reads and writes the same bytes.
    hbm = gbps(device_src, device_dst)
    return {"hbm_read": hbm, "hbm_write": hbm, "host_read": host_read, "host_write": host_write}


def _nmse(
    codec: _Codec, source: dict, perm: torch.Tensor, pages: int
) -> tuple[float, dict[str, float]]:
    """Mean over K/V vectors of ||x - x_hat||^2 / ||x||^2, for the case and per buffer.

    x_hat is x after the round trip; the per-buffer means are keyed ``"layer/buffer"``.
    """
    total = torch.zeros((), dtype=torch.float64, device="cuda")
    count = torch.zeros((), dtype=torch.int64, device="cuda")
    per_buffer = {}
    for (layer, role), values in source.items():
        reference = values.float()[perm].flatten(0, -2)
        decoded = codec.hot[(layer, role)][pages:].float().flatten(0, -2)
        errors, norms = (decoded - reference).square().sum(-1), reference.square().sum(-1)
        keep = norms > 0
        buffer_total = (errors[keep] / norms[keep]).sum(dtype=torch.float64)
        total += buffer_total
        count += keep.sum()
        per_buffer[f"{layer}/{role}"] = buffer_total / keep.sum().clamp(min=1)
    return float(total / count), {key: float(value) for key, value in per_buffer.items()}


def _json_safe(value: object) -> object:
    """``value`` with every non-finite float replaced by None, so the report stays valid JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _gpu_activities(fn: Callable[[], None]) -> list[str]:
    """Names of the GPU kernels and copies that ``fn()`` launches, in order, with repeat counts."""
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
        fn()
        torch.cuda.synchronize()
    cuda = torch.autograd.DeviceType.CUDA
    events = [event for event in profiler.events() if event.device_type == cuda]
    names = (event.name for event in sorted(events, key=lambda event: event.time_range.start))
    return [f"{name} x{len(list(group))}" for name, group in itertools.groupby(names)]


def _error(fmt: str, stage: str, error: Exception) -> dict:
    message = " ".join(f"{type(error).__name__}: {error}".split())
    print(f"  {fmt} {stage} ERROR: {message}")
    return {"error": f"{stage}: {message}"}


def _run_case(
    case: Case, args: argparse.Namespace, bandwidth: dict, stream: torch.cuda.Stream
) -> dict:
    """Time every format on one case, check its round trip, print and return the results."""
    buffers = ", ".join(
        f"{role} {'x'.join(map(str, case.slot_shape(role)))}" for role in case.roles
    )
    print(
        f"\n{case.name}: {case.model}, {case.layers} layers x {{{buffers}}} {case.runtime}"
        f" ({case.hot_page_bytes / 2**20:.2f} MiB/page), {case.pages} pages/call;"
        f" TurboQuant {case.mode} {case.bits}-bit"
        + (", skip_rope_quantization" if case.skip_rope else "")
    )
    source = _make_source(case)
    perm = torch.randperm(case.pages, generator=torch.Generator().manual_seed(SEED)).to(torch.int32)
    codecs, results = {}, {}
    for fmt in args.formats:
        try:
            codecs[fmt] = _build_codec(case, fmt, source, perm)
        except Exception as error:  # any provider failure: report it, measure the other formats
            results[fmt] = _error(fmt, "setup", error)

    torch.cuda.synchronize()
    samples = {fmt: {op: [] for op in OPS} for fmt in codecs}
    for rep in range(args.warmup + args.reps):
        for op in OPS:
            for fmt, codec in list(codecs.items()):
                try:
                    elapsed = _time_us(lambda: codec.run(op, stream), stream)
                except Exception as error:  # any provider failure: report it, measure the others
                    results[fmt] = _error(fmt, op, error)
                    del codecs[fmt]
                    continue
                if rep >= args.warmup:
                    samples[fmt][op].append(elapsed)
    sm_clock = torch.cuda.clock_rate()

    hot_bytes = case.pages * case.hot_page_bytes
    perm_device = perm.to("cuda", torch.int64)
    for fmt, codec in codecs.items():
        cold_bytes = case.pages * codec.cold_page_bytes
        sol_ns = {  # bytes / (GB/s) is nanoseconds
            "encode": max(hot_bytes / bandwidth["hbm_read"], cold_bytes / bandwidth["host_write"]),
            "decode": max(cold_bytes / bandwidth["host_read"], hot_bytes / bandwidth["hbm_write"]),
        }
        nmse, per_buffer = _nmse(codec, source, perm_device, case.pages)
        bound = 2 * PAPER_MSE[case.bits] if fmt == "turboquant" and case.mode == "mse" else None
        finite = all(math.isfinite(value) for value in (nmse, *per_buffer.values()))
        worst = max(  # a non-finite buffer first
            per_buffer,
            key=lambda key: per_buffer[key] if math.isfinite(per_buffer[key]) else math.inf,
        )
        result = results[fmt] = {
            "cold_page_bytes": codec.cold_page_bytes,
            "bits_per_element": 8 * codec.cold_page_bytes / case.page_elements,
            "page_index_location": codec.index_location,
            "nmse": nmse,
            "nmse_bound": bound,
            "worst_buffer": worst,
            "worst_buffer_nmse": per_buffer[worst],
            # One corrupted layer barely moves the case-wide mean; check every buffer.
            "passed": finite
            and (bound is None or nmse <= bound)
            and per_buffer[worst] <= WORST_BUFFER_FACTOR * nmse,
        }
        for op in OPS:
            median, sol_us = statistics.median(samples[fmt][op]), sol_ns[op] / 1e3
            result[op] = {
                "median_us": median,
                "samples_us": samples[fmt][op],
                "cold_gbps": cold_bytes / median / 1e3,
                "hot_gbps": hot_bytes / median / 1e3,
                "sol_us": sol_us,
                "sol_percent": 100 * sol_us / median,
            }
        if args.profile:
            result["kernels"] = {op: _gpu_activities(lambda: codec.run(op, stream)) for op in OPS}
    if "turboquant" in codecs and "nvfp4" in codecs:
        for op in OPS:
            ratio = results["turboquant"][op]["median_us"] / results["nvfp4"][op]["median_us"]
            results["turboquant"][op]["vs_nvfp4"] = ratio

    _print_case(case, results, sm_clock)
    return {
        **asdict(case),
        "hot_page_bytes": case.hot_page_bytes,
        "sm_clock_mhz_after_timing": sm_clock,
        "formats": {fmt: results[fmt] for fmt in args.formats},
    }


_COLUMNS = {  # header: alignment and width
    "format": "<10",
    "cold B/page": ">11",
    "bit/elem": ">8",
    "index": "<6",
    "nMSE": ">8",
    "check": "<6",
    "op": "<6",
    "median us": ">9",
    "cold GB/s": ">9",
    "hot GB/s": ">9",
    "SOL us": ">8",
    "%SOL": ">5",
    "TQ/NVFP4": ">8",
}
_ROW = "  " + "  ".join(f"{{:{spec}}}" for spec in _COLUMNS.values())


def _print_case(case: Case, results: dict, sm_clock: int) -> None:
    measured = {fmt: result for fmt, result in results.items() if "error" not in result}
    if not measured:
        return
    print(_ROW.format(*_COLUMNS))
    for fmt, result in measured.items():
        for op in OPS:
            stats = result[op]
            print(
                _ROW.format(
                    fmt,
                    result["cold_page_bytes"],
                    f"{result['bits_per_element']:.3f}",
                    result["page_index_location"],
                    f"{result['nmse']:.5f}",
                    "ok" if result["passed"] else "FAILED",
                    op,
                    f"{stats['median_us']:.1f}",
                    f"{stats['cold_gbps']:.1f}",
                    f"{stats['hot_gbps']:.1f}",
                    f"{stats['sol_us']:.1f}",
                    f"{stats['sol_percent']:.1f}",
                    f"{stats['vs_nvfp4']:.3f}x" if "vs_nvfp4" in stats else "",
                ).rstrip()
            )
    turboquant = measured.get("turboquant", {})
    if turboquant.get("nmse_bound") is not None:
        bound = turboquant["nmse_bound"]
        print(f"  TurboQuant nMSE bound, 2x the paper's {case.bits}-bit distortion: {bound:.4f}")
    if case.latency and "vs_nvfp4" in turboquant.get("encode", {}):
        ratios = ", ".join(f"{op} {turboquant[op]['vs_nvfp4']:.3f}x" for op in OPS)
        print(f"  ** latency case, fixed costs dominate: judge TurboQuant/NVFP4 time, {ratios} **")
    for fmt, result in measured.items():
        for op, names in result.get("kernels", {}).items():
            print(f"  {fmt} {op} launches: {'; '.join(names) or 'nothing on the GPU'}")
    print(f"  SM clock after timing: {sm_clock} MHz")


def _csv(allowed) -> Callable[[str], list[str]]:
    def parse(text: str) -> list[str]:
        values = list(dict.fromkeys(text.split(",")))
        unknown = sorted(set(values) - set(allowed))
        if unknown:
            raise argparse.ArgumentTypeError(f"unknown {unknown}; choose from {','.join(allowed)}")
        return values

    return parse


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cases", type=_csv(CASES), default=list(CASES), help="default: all")
    parser.add_argument("--formats", type=_csv(FORMATS), default=list(FORMATS), help="default: all")
    parser.add_argument("--warmup", type=int, default=3, help="untimed calls first (default: 3)")
    parser.add_argument("--reps", type=int, default=20, help="timed calls (default: 20)")
    parser.add_argument("--json", help="also write all results, samples included, to this file")
    parser.add_argument(
        "--profile", action="store_true", help="list the GPU kernels of each encode and decode"
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.reps < 1:
        parser.error("--warmup must be >= 0 and --reps >= 1")
    # TurboQuant admits an FP8 active KV cache from compute capability 8.9 (Triton FP8) only.
    skipped = [
        name
        for name in args.cases
        if CASES[name].runtime == "fp8" and torch.cuda.get_device_capability() < (8, 9)
    ]
    for name in skipped:
        print(f"Skipping {name}: an FP8 active KV cache needs compute capability 8.9 or newer")
    args.cases = [name for name in args.cases if name not in skipped]

    properties = torch.cuda.get_device_properties()
    gpu = f"{properties.name}, sm_{properties.major}{properties.minor}"
    gpu += f", {properties.multi_processor_count} SMs"
    sm_clock = torch.cuda.clock_rate()
    print(f"GPU {gpu}; SM clock now {sm_clock} MHz (clocks are not locked)")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        bandwidth = _measure_bandwidth(stream, args.warmup, args.reps)
        print(
            f"Triton copy kernel over {SOL_BYTES >> 20} MiB, median GB/s: HBM read"
            f" {bandwidth['hbm_read']:.1f} / write {bandwidth['hbm_write']:.1f} (device copy),"
            f" GPU reads pinned host {bandwidth['host_read']:.1f},"
            f" GPU writes pinned host {bandwidth['host_write']:.1f}"
        )
        reports = [_run_case(CASES[name], args, bandwidth, stream) for name in args.cases]

    problems = [
        f"{report['name']} {fmt}: {result.get('error', 'nMSE check FAILED')}"
        for report in reports
        for fmt, result in report["formats"].items()
        if not result.get("passed", False)
    ]
    print("\nNot counted:" if problems else "\nAll requested measurements passed their checks.")
    for problem in problems:
        print(f"  {problem}")
    if args.json:
        summary = {
            "gpu": gpu,
            "sm_clock_mhz_at_start": sm_clock,
            "warmup": args.warmup,
            "reps": args.reps,
            "bandwidth_gbps": bandwidth,
            "skipped_cases": skipped,
            "cases": reports,
        }
        with open(args.json, "w") as file:
            json.dump(_json_safe(summary), file, indent=2, allow_nan=False)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

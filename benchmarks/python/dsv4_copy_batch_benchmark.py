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

"""Benchmark DeepSeek-V4 KV-cache copy-batch table construction."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch

from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import DeepseekV4CacheManager
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.cache_manager import (
    DSV4_ENABLE_SWA_SCRATCH_REUSE_ENV,
)
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.deepseek_v4 import (
    DEEPSEEK_V4_SLIDING_ATTENTION,
    DeepseekV4AttentionType,
    compress_ratio_has_attention,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping

DEFAULT_MODEL_CONFIG = Path(
    "/home/scratch.jiaganc_gpu/workspace/trtllm/dsv4/DeepSeek-V4-Flash/config.json"
)
DEFAULT_HEAD_DIM = 512
DEFAULT_INDEX_HEAD_DIM = 128
DEFAULT_WINDOW_SIZE = 128
DEFAULT_VOCAB_SIZE = 129280
DEFAULT_NUM_KV_HEADS = 1
DEFAULT_MAX_GPU_TOTAL_BYTES = 80_000_000_000
DEFAULT_WARMUP = 2
DEFAULT_ITERATIONS = 10
SUPPORTED_COMPRESS_RATIOS = {0, 1, 4, 128}
METHODS = ("block_offsets", "sliding", "compress", "indexer")


@dataclass(frozen=True)
class BenchmarkConfig:
    model_config_path: Path
    batch_size: int
    seq_len: int
    max_seq_len: int
    tokens_per_block: int
    head_dim: int
    index_head_dim: int
    window_size: int
    vocab_size: int
    num_kv_heads: int
    max_gpu_total_bytes: int
    compress_ratios: list[int]
    dtype: DataType
    warmup: int
    iterations: int
    phases: list[str]
    methods: list[str]
    scratch_reuse_modes: list[str]
    sync_cuda: bool
    include_device_copy: bool


@dataclass(frozen=True)
class BenchmarkTarget:
    name: str
    phase: str
    uses_cuda: bool
    fn: Callable[[], None]


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    phase: str
    scratch_reuse: str
    iterations: int
    min_ms: float
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    max_ms: float
    mean_ms_per_seq: float


@contextmanager
def _scratch_reuse_env(mode: str):
    original = os.environ.get(DSV4_ENABLE_SWA_SCRATCH_REUSE_ENV)
    if mode == "enabled":
        os.environ[DSV4_ENABLE_SWA_SCRATCH_REUSE_ENV] = "1"
    elif mode == "disabled":
        os.environ.pop(DSV4_ENABLE_SWA_SCRATCH_REUSE_ENV, None)
    else:
        raise ValueError(f"unsupported scratch reuse mode: {mode}")
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(DSV4_ENABLE_SWA_SCRATCH_REUSE_ENV, None)
        else:
            os.environ[DSV4_ENABLE_SWA_SCRATCH_REUSE_ENV] = original


def _normalize_compress_ratios(values: list[int]) -> list[int]:
    unsupported = sorted(set(values) - SUPPORTED_COMPRESS_RATIOS)
    if unsupported:
        raise argparse.ArgumentTypeError(f"unsupported DeepSeek-V4 compress ratios: {unsupported}")
    return [1 if value == 0 else value for value in values]


def _parse_int_list(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid integer list: {value}") from error
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    return _normalize_compress_ratios(values)


def _load_model_config(path: Path) -> dict[str, object]:
    try:
        with path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)
    except OSError as error:
        raise SystemExit(f"failed to read --model-config {path}: {error}") from error
    if not isinstance(config, dict):
        raise SystemExit(f"--model-config must contain a JSON object: {path}")
    return config


def _get_int(model_config: dict[str, object], name: str, default: int) -> int:
    value = model_config.get(name, default)
    if not isinstance(value, int):
        raise SystemExit(f"model config field {name!r} must be an integer")
    return value


def _get_window_size(model_config: dict[str, object]) -> int:
    value = model_config.get(
        "window_size",
        model_config.get("sliding_window", DEFAULT_WINDOW_SIZE),
    )
    if not isinstance(value, int):
        raise SystemExit("model config field 'window_size' or 'sliding_window' must be an integer")
    return value


def _get_model_compress_ratios(model_config: dict[str, object]) -> list[int]:
    ratios = model_config.get("compress_ratios")
    if not isinstance(ratios, list) or not all(isinstance(ratio, int) for ratio in ratios):
        raise SystemExit("model config field 'compress_ratios' must be a list of integers")
    return _normalize_compress_ratios(ratios)


def _get_model_dtype(model_config: dict[str, object]) -> str:
    torch_dtype = str(model_config.get("torch_dtype", "bfloat16")).lower()
    if torch_dtype in ("bfloat16", "bf16"):
        return "bf16"
    if torch_dtype in ("float8", "fp8", "float8_e4m3fn"):
        return "fp8"
    raise SystemExit(f"unsupported model config torch_dtype: {torch_dtype}")


def _expand_compress_ratios(pattern: list[int], num_layers: int | None) -> list[int]:
    if num_layers is None:
        return pattern
    if num_layers <= 0:
        raise ValueError("--num-layers must be positive")
    return [pattern[idx % len(pattern)] for idx in range(num_layers)]


def _parse_choices(value: str, choices: Iterable[str]) -> list[str]:
    valid_choices = set(choices)
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    if "all" in values:
        return [choice for choice in choices if choice != "all"]
    invalid = sorted(set(values) - valid_choices)
    if invalid:
        raise argparse.ArgumentTypeError(f"invalid choices: {invalid}")
    return values


def _create_request(request_id: int, prompt_len: int) -> LlmRequest:
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=1024,
        input_tokens=list(range(prompt_len)),
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )


def _create_cache_manager(config: BenchmarkConfig) -> DeepseekV4CacheManager:
    sparse_attn_config = DeepSeekV4SparseAttentionConfig(
        index_head_dim=config.index_head_dim,
        window_size=config.window_size,
        compress_ratios=config.compress_ratios,
    )
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=False,
        max_gpu_total_bytes=config.max_gpu_total_bytes,
        event_buffer_max_size=0,
    )
    return DeepseekV4CacheManager(
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=len(config.compress_ratios),
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        tokens_per_block=config.tokens_per_block,
        max_seq_len=config.max_seq_len,
        max_batch_size=config.batch_size,
        max_input_len=config.seq_len,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=config.dtype,
        compressor_dtype=DataType.FLOAT,
        vocab_size=config.vocab_size,
        sparse_attn_config=sparse_attn_config,
    )


def _prepare_context_requests(
    cache_manager: DeepseekV4CacheManager,
    batch_size: int,
    seq_len: int,
) -> list[LlmRequest]:
    requests = [_create_request(request_id, seq_len) for request_id in range(batch_size)]
    for request in requests:
        if not cache_manager.prepare_context(request):
            raise RuntimeError(f"prepare_context failed for request {request.py_request_id}")
        if not cache_manager.resize_context(request, request.context_chunk_size):
            raise RuntimeError(f"resize_context failed for request {request.py_request_id}")
    return requests


def _promote_to_generation(
    cache_manager: DeepseekV4CacheManager,
    requests: list[LlmRequest],
    seq_len: int,
) -> None:
    scheduled_batch = ScheduledRequests()
    scheduled_batch.context_requests_last_chunk = requests
    for request in requests:
        request.context_current_position = seq_len
        request.add_new_token(seq_len, 0)
        if request.context_remaining_length != 0:
            raise RuntimeError(
                f"request {request.py_request_id} still has "
                f"{request.context_remaining_length} context tokens remaining"
            )
    cache_manager.update_context_resources(scheduled_batch)
    for request in requests:
        kv_cache = cache_manager.kv_cache_map[request.py_request_id]
        if kv_cache.enable_swa_scratch_reuse:
            raise RuntimeError(
                f"scratch reuse is still enabled for generation request {request.py_request_id}"
            )
    cache_manager.update_resources(scheduled_batch)
    for request in requests:
        if not cache_manager.try_allocate_generation(request):
            raise RuntimeError(
                f"try_allocate_generation failed for request {request.py_request_id}"
            )


def _allocate_targets(
    cache_manager: DeepseekV4CacheManager,
    requests: list[LlmRequest],
    phase: str,
    methods: list[str],
    include_device_copy: bool,
) -> list[BenchmarkTarget]:
    request_ids = [request.py_request_id for request in requests]
    num_seqs = len(request_ids)
    num_contexts = num_seqs if phase == "context" else 0
    max_blocks = cache_manager.max_blocks_per_seq
    targets: list[BenchmarkTarget] = []
    target_kwargs = (
        {"device": "cuda"}
        if include_device_copy
        else {"device": "cpu", "pin_memory": prefer_pinned()}
    )

    if "block_offsets" in methods:
        block_offsets_num_seqs = (
            cache_manager._host_attention_op_block_offsets_staging.shape[1]
            if include_device_copy
            else num_seqs
        )
        block_offsets = torch.empty(
            cache_manager.num_attention_op_pools,
            block_offsets_num_seqs,
            2,
            max_blocks,
            dtype=torch.int32,
            **target_kwargs,
        )
        targets.append(
            BenchmarkTarget(
                name="copy_batch_block_offsets",
                phase=phase,
                uses_cuda=include_device_copy,
                fn=lambda: cache_manager.copy_batch_block_offsets(
                    block_offsets,
                    request_ids,
                    beam_width=1,
                    num_contexts=num_contexts,
                    num_seqs=num_seqs,
                ),
            )
        )

    if "sliding" in methods:
        sliding_tables = torch.empty(
            cache_manager.num_local_layers,
            len(DEEPSEEK_V4_SLIDING_ATTENTION),
            num_seqs,
            max_blocks,
            dtype=torch.int32,
            **target_kwargs,
        )
        targets.append(
            BenchmarkTarget(
                name="copy_batch_sliding_block_tables",
                phase=phase,
                uses_cuda=include_device_copy,
                fn=lambda: cache_manager.copy_batch_sliding_block_tables(
                    sliding_tables,
                    request_ids,
                    num_contexts=num_contexts,
                    num_seqs=num_seqs,
                ),
            )
        )

    if "compress" in methods:
        compress_ratios = sorted(
            {
                ratio
                for ratio in cache_manager._compress_ratios
                if compress_ratio_has_attention(ratio, DeepseekV4AttentionType.COMPRESS)
            }
        )
        for compress_ratio in compress_ratios:
            compress_table = torch.empty(
                num_seqs,
                max_blocks,
                dtype=torch.int32,
                **target_kwargs,
            )

            def _copy_compress(compress_ratio=compress_ratio, compress_table=compress_table):
                cache_manager.copy_batch_compress_block_tables(
                    compress_table,
                    request_ids,
                    compress_ratio=compress_ratio,
                    beam_width=1,
                    num_contexts=num_contexts,
                    num_seqs=num_seqs,
                )

            targets.append(
                BenchmarkTarget(
                    name=f"copy_batch_compress_block_tables:{compress_ratio}",
                    phase=phase,
                    uses_cuda=include_device_copy,
                    fn=_copy_compress,
                )
            )

    if "indexer" in methods and any(
        compress_ratio_has_attention(ratio, DeepseekV4AttentionType.INDEXER_COMPRESS)
        for ratio in cache_manager._compress_ratios
    ):
        indexer_table = torch.empty(
            num_seqs,
            max_blocks,
            dtype=torch.int32,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        targets.append(
            BenchmarkTarget(
                name="copy_batch_indexer_compress_block_tables",
                phase=phase,
                uses_cuda=False,
                fn=lambda: cache_manager.copy_batch_indexer_compress_block_tables(
                    indexer_table,
                    request_ids,
                    beam_width=1,
                    num_contexts=num_contexts,
                    num_seqs=num_seqs,
                ),
            )
        )

    return targets


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _benchmark_target(
    target: BenchmarkTarget,
    iterations: int,
    warmup: int,
    sync_cuda: bool,
    batch_size: int,
    scratch_reuse: str,
) -> BenchmarkResult:
    should_sync = sync_cuda and target.uses_cuda
    for _ in range(warmup):
        target.fn()
        if should_sync:
            torch.cuda.synchronize()

    elapsed_ms = []
    for _ in range(iterations):
        if should_sync:
            torch.cuda.synchronize()
        start_ns = time.perf_counter_ns()
        target.fn()
        if should_sync:
            torch.cuda.synchronize()
        elapsed_ms.append((time.perf_counter_ns() - start_ns) / 1_000_000.0)

    sorted_ms = sorted(elapsed_ms)
    mean_ms = statistics.fmean(elapsed_ms)
    return BenchmarkResult(
        name=target.name,
        phase=target.phase,
        scratch_reuse=scratch_reuse,
        iterations=iterations,
        min_ms=sorted_ms[0],
        mean_ms=mean_ms,
        p50_ms=_percentile(sorted_ms, 0.50),
        p90_ms=_percentile(sorted_ms, 0.90),
        p99_ms=_percentile(sorted_ms, 0.99),
        max_ms=sorted_ms[-1],
        mean_ms_per_seq=mean_ms / batch_size,
    )


def _print_results(results: list[BenchmarkResult], config: BenchmarkConfig) -> None:
    print(f"Model config: {config.model_config_path}")
    print(
        "Config: "
        f"batch={config.batch_size}, seq_len={config.seq_len}, max_seq_len={config.max_seq_len}, "
        f"tokens_per_block={config.tokens_per_block}, layers={len(config.compress_ratios)}, "
        f"max_gpu_total_bytes={config.max_gpu_total_bytes}, sync_cuda={config.sync_cuda}, "
        f"include_device_copy={config.include_device_copy}, "
        f"scratch_reuse_modes={config.scratch_reuse_modes}"
    )
    print(
        "Model fields: "
        f"head_dim={config.head_dim}, index_head_dim={config.index_head_dim}, "
        f"window_size={config.window_size}, num_kv_heads={config.num_kv_heads}, "
        f"vocab_size={config.vocab_size}"
    )
    print(f"Compress ratios: {config.compress_ratios}")
    print()
    header = (
        "phase",
        "scratch_reuse",
        "target",
        "iters",
        "min_ms",
        "mean_ms",
        "p50_ms",
        "p90_ms",
        "p99_ms",
        "max_ms",
        "mean_ms_per_seq",
    )
    print(" ".join(f"{column:>18}" for column in header))
    for result in results:
        print(
            f"{result.phase:>18} "
            f"{result.scratch_reuse:>18} "
            f"{result.name:>18} "
            f"{result.iterations:>18d} "
            f"{result.min_ms:>18.3f} "
            f"{result.mean_ms:>18.3f} "
            f"{result.p50_ms:>18.3f} "
            f"{result.p90_ms:>18.3f} "
            f"{result.p99_ms:>18.3f} "
            f"{result.max_ms:>18.3f} "
            f"{result.mean_ms_per_seq:>18.3f}"
        )


def _write_csv(results: list[BenchmarkResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(BenchmarkResult.__dataclass_fields__))
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def _run_phase(config: BenchmarkConfig, phase: str, scratch_reuse: str) -> list[BenchmarkResult]:
    with _scratch_reuse_env(scratch_reuse):
        cache_manager = _create_cache_manager(config)
    requests: list[LlmRequest] = []
    try:
        if cache_manager.enable_swa_scratch_reuse != (scratch_reuse == "enabled"):
            raise RuntimeError(
                f"expected scratch reuse {scratch_reuse}, got "
                f"{cache_manager.enable_swa_scratch_reuse}"
            )
        requests = _prepare_context_requests(cache_manager, config.batch_size, config.seq_len)
        if phase == "generation":
            _promote_to_generation(cache_manager, requests, config.seq_len)
        targets = _allocate_targets(
            cache_manager,
            requests,
            phase,
            config.methods,
            config.include_device_copy,
        )
        return [
            _benchmark_target(
                target,
                iterations=config.iterations,
                warmup=config.warmup,
                sync_cuda=config.sync_cuda,
                batch_size=config.batch_size,
                scratch_reuse=scratch_reuse,
            )
            for target in targets
        ]
    finally:
        for request in requests:
            if request.py_request_id in cache_manager.kv_cache_map:
                cache_manager.free_resources(request)
        cache_manager.shutdown()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark DeepseekV4CacheManager copy_batch_* helpers using the same setup pattern "
            "as tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py."
        )
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help="DeepSeek-V4 model config JSON used for model dimensions and default compress ratios.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--tokens-per-block", type=int, choices=(128, 256), default=128)
    parser.add_argument(
        "--max-gpu-total-bytes",
        type=int,
        default=DEFAULT_MAX_GPU_TOTAL_BYTES,
        help="GPU KV-cache byte quota. Defaults to 80,000,000,000 bytes.",
    )
    parser.add_argument(
        "--compress-ratios",
        type=_parse_int_list,
        default=None,
        help="Override comma-separated ratios. Ratio 0 is normalized to 1.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="If set, repeat/truncate the selected compress-ratio pattern to this many layers.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp8"),
        default=None,
        help="Override KV cache dtype; default comes from --model-config torch_dtype.",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument(
        "--phase",
        type=lambda value: _parse_choices(value, ("all", "context", "generation")),
        default=["context", "generation"],
        help="Comma-separated phase list: context,generation,all.",
    )
    parser.add_argument(
        "--methods",
        type=lambda value: _parse_choices(value, ("all", *METHODS)),
        default=list(METHODS),
        help="Comma-separated method list: block_offsets,sliding,compress,indexer,all.",
    )
    parser.add_argument(
        "--scratch-reuse",
        type=lambda value: _parse_choices(value, ("all", "disabled", "enabled")),
        default=["disabled", "enabled"],
        help="Comma-separated scratch reuse modes: disabled,enabled,all.",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Do not synchronize CUDA after device-copy helpers; measures host enqueue overhead only.",
    )
    parser.add_argument(
        "--include-device-copy",
        action="store_true",
        help=(
            "Include the final CPU-to-GPU copy into destination tensors. By default "
            "destinations stay on CPU so timings exclude that transfer."
        ),
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--csv", type=Path, default=None)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for DeepSeek-V4 cache-manager benchmarking.")
    torch.cuda.set_device(args.device)
    model_config = _load_model_config(args.model_config)

    max_seq_len = args.max_seq_len if args.max_seq_len is not None else args.seq_len + 1
    if max_seq_len < args.seq_len + 1:
        raise SystemExit("--max-seq-len must be at least --seq-len + 1")

    compress_ratios = (
        args.compress_ratios
        if args.compress_ratios is not None
        else _get_model_compress_ratios(model_config)
    )
    dtype = args.dtype if args.dtype is not None else _get_model_dtype(model_config)
    config = BenchmarkConfig(
        model_config_path=args.model_config,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_seq_len=max_seq_len,
        tokens_per_block=args.tokens_per_block,
        head_dim=_get_int(model_config, "head_dim", DEFAULT_HEAD_DIM),
        index_head_dim=_get_int(model_config, "index_head_dim", DEFAULT_INDEX_HEAD_DIM),
        window_size=_get_window_size(model_config),
        vocab_size=_get_int(model_config, "vocab_size", DEFAULT_VOCAB_SIZE),
        num_kv_heads=_get_int(model_config, "num_key_value_heads", DEFAULT_NUM_KV_HEADS),
        max_gpu_total_bytes=args.max_gpu_total_bytes,
        compress_ratios=_expand_compress_ratios(compress_ratios, args.num_layers),
        dtype={"bf16": DataType.BF16, "fp8": DataType.FP8}[dtype],
        warmup=args.warmup,
        iterations=args.iterations,
        phases=args.phase,
        methods=args.methods,
        scratch_reuse_modes=args.scratch_reuse,
        sync_cuda=not args.no_sync,
        include_device_copy=args.include_device_copy,
    )

    all_results: list[BenchmarkResult] = []
    for scratch_reuse in config.scratch_reuse_modes:
        for phase in config.phases:
            all_results.extend(_run_phase(config, phase, scratch_reuse))

    _print_results(all_results, config)
    if args.csv is not None:
        _write_csv(all_results, args.csv)


if __name__ == "__main__":
    main()

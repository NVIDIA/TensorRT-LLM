#!/usr/bin/env python3
#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Skip-softmax FMHA prefill benchmark via TrtllmAttentionWrapper.run().

The default torch-op backend drives the production attention kernel through
TrtllmAttentionWrapper with no KV cache (fused-QKV context attention) and times
each (config, threshold) via flashinfer.testing.bench_gpu_time using CUDA Event
+ CUDA Graph mode. A torch.profiler verification pass asserts that exactly one
FMHA kernel runs per timed iteration; surrounding QKV preprocessing kernels
(applyBiasRopeUpdateKVCacheV2, computeSeqAndPaddingOffsets) are reported as a
warning and contribute constant overhead independent of the skip-softmax
threshold.

Decode configs (seq_len_q != seq_len_kv) and e4m3 are skipped by this
microbenchmark. Use the standalone bench_skip_softmax_fmha_exe.py script only
when direct fmha.exe cross-validation is explicitly needed; this benchmark is
intended to stay attention-op based.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Callable

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests.microbenchmarks.skip_softmax.diffusion_configs import diffusion_configs  # noqa: E402
from tests.microbenchmarks.skip_softmax.llm_configs import (  # noqa: E402
    FmhaConfig,
    llm_configs,
    threshold_sweep_from_multipliers,
)

PASS1_COLUMNS = [
    "config",
    "dtype",
    "batch",
    "num_heads_q",
    "num_heads_kv",
    "head_size",
    "seq_len_q",
    "seq_len_kv",
    "mask",
    "threshold_scale_factor",
    "skipped_blocks",
    "total_blocks",
    "achieved_sparsity",
]
PASS2_COLUMNS = [
    "config",
    "dtype",
    "threshold_scale_factor",
    "elapsed_us_median",
    "speedup",
]
TORCH_SUPPORTED_DTYPES = ("bf16", "fp16")
DRY_RUN_ITERS = 5
REPEAT_ITERS = 20
FMHA_RANDOM_SEED = 1234
FMHA_Q_RANGE = 5
FMHA_K_RANGE = 3
FMHA_V_RANGE = 5
FMHA_Q_SCALE = 1.0
FMHA_K_SCALE = 1.0
FMHA_BF16_FP16_V_SCALE = 0.125


def _load_configs(
    config: str,
    threshold_sweep: list[float] | None,
    threshold_multipliers: list[float] | None,
    config_filter: set[str] | None,
    batch_size: int | None,
) -> list[FmhaConfig]:
    cfgs: list[FmhaConfig] = []
    if config in ("llm", "both"):
        cfgs.extend(llm_configs())
    if config in ("diffusion", "both"):
        cfgs.extend(diffusion_configs())

    if config_filter is not None:
        available = {cfg.name for cfg in cfgs}
        missing = sorted(config_filter - available)
        if missing:
            _warn(f"--config-filter ignored unknown configs: {missing}")
        cfgs = [cfg for cfg in cfgs if cfg.name in config_filter]

    if threshold_sweep is None and threshold_multipliers is None and batch_size is None:
        return cfgs

    updated_cfgs: list[FmhaConfig] = []
    for cfg in cfgs:
        updates: dict[str, object] = {}
        if threshold_sweep is not None:
            updates["threshold_sweep"] = list(threshold_sweep)
        if threshold_multipliers is not None:
            updates["threshold_sweep"] = threshold_sweep_from_multipliers(
                cfg, threshold_multipliers
            )
        if batch_size is not None:
            updates["batch"] = batch_size
        updated_cfgs.append(replace(cfg, **updates))
    return updated_cfgs


def _parse_threshold_sweep(values: list[str] | None) -> list[float] | None:
    if values is None:
        return None
    thresholds: list[float] = []
    for value in values:
        thresholds.extend(float(part) for part in value.split(",") if part)
    return thresholds


def _parse_config_filter(value: str | None) -> set[str] | None:
    if value is None:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def _warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def _torch_op_configs(cfgs: list[FmhaConfig]) -> list[FmhaConfig]:
    selected: list[FmhaConfig] = []
    for cfg in cfgs:
        if cfg.seq_len_q != cfg.seq_len_kv:
            _warn(
                f"skipping {cfg.name}: torch-op backend is prefill-only "
                f"(seq_len_q={cfg.seq_len_q}, seq_len_kv={cfg.seq_len_kv})"
            )
            continue
        if cfg.dtype not in TORCH_SUPPORTED_DTYPES:
            _warn(
                f"skipping {cfg.name}: torch-op backend supports bf16/fp16 "
                f"inputs, not dtype={cfg.dtype!r}"
            )
            continue
        selected.append(cfg)
    return selected


def _import_torch() -> ModuleType:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("the torch-op backend requires PyTorch") from exc
    return torch


def _import_trtllm_attention() -> tuple[type, object, object, Callable[[], int]]:
    try:
        from tensorrt_llm._torch.attention_backend import trtllm as trtllm_module
        from tensorrt_llm._torch.attention_backend.interface import (
            AttentionInputType,
            PredefinedAttentionMask,
        )
        from tensorrt_llm._utils import get_sm_version
    except ImportError as exc:
        raise RuntimeError(
            "The torch-op backend requires the TensorRT-LLM Python attention "
            "stack and its compiled thop bindings."
        ) from exc

    trtllm_module._TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION = False
    return (
        trtllm_module.TrtllmAttentionWrapper,
        AttentionInputType,
        PredefinedAttentionMask,
        get_sm_version,
    )


def _fill_fmha_random_(tensor: object, value_range: int, scale: float, generator: object) -> None:
    tensor.random_(0, value_range, generator=generator)
    tensor.sub_(value_range // 2)
    if scale != 1.0:
        tensor.mul_(scale)


class TorchOpFmhaRunner:
    def __init__(self, cfg: FmhaConfig, init_mode: str) -> None:
        torch = _import_torch()
        wrapper_cls, attention_input_type, predefined_mask, get_sm_version = (
            _import_trtllm_attention()
        )
        self.cfg = cfg
        self.torch = torch
        self.attention_input_type = attention_input_type
        self.attention_mask = (
            predefined_mask.CAUSAL if cfg.mask == "causal" else predefined_mask.FULL
        )
        self.use_paged_context_fmha = get_sm_version() == 90
        self.num_tokens = cfg.batch * cfg.seq_len_q
        self.qkv_hidden_size = (cfg.num_heads_q + 2 * cfg.num_heads_kv) * cfg.head_size
        self.output_hidden_size = cfg.num_heads_q * cfg.head_size
        dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[cfg.dtype]

        self.qkv = torch.empty((self.num_tokens, self.qkv_hidden_size), dtype=dtype, device="cuda")
        if init_mode == "fmha-random":
            self._init_qkv_fmha_random(dtype)
        elif init_mode == "torch-normal":
            self.qkv.normal_(mean=0.0, std=0.02)
        else:
            raise ValueError(f"Unknown init mode: {init_mode}")
        self.output = torch.empty(
            (self.num_tokens, self.output_hidden_size), dtype=dtype, device="cuda"
        )
        self.workspace = torch.empty((0,), dtype=torch.int8, device="cuda")

        self.sequence_length = torch.full(
            (cfg.batch,), cfg.seq_len_q, dtype=torch.int32, device="cuda"
        )
        self.context_lengths = self.sequence_length.clone()
        self.host_context_lengths = torch.full(
            (cfg.batch,), cfg.seq_len_q, dtype=torch.int32, device="cpu"
        )
        self.host_past_key_value_lengths = self.host_context_lengths.clone()
        self.host_total_kv_lens = torch.tensor(
            [self.num_tokens, 0], dtype=torch.int32, device="cpu"
        )
        self.host_request_types = torch.zeros((cfg.batch,), dtype=torch.int32, device="cpu")

        self.wrapper = wrapper_cls(cfg.num_heads_q, cfg.head_size, num_kv_heads=cfg.num_heads_kv)
        self.wrapper.update_quant_config(None)

    def _init_qkv_fmha_random(self, dtype: object) -> None:
        """Use the same discrete Q/K/V value ranges as fmha.exe."""
        torch = self.torch
        cfg = self.cfg
        head_size = cfg.head_size
        q_size = cfg.num_heads_q * head_size
        k_size = cfg.num_heads_kv * head_size
        q = self.qkv[:, :q_size].view(self.num_tokens, cfg.num_heads_q, head_size)
        k = self.qkv[:, q_size : q_size + k_size].view(self.num_tokens, cfg.num_heads_kv, head_size)
        v = self.qkv[:, q_size + k_size :].view(self.num_tokens, cfg.num_heads_kv, head_size)

        generator = torch.Generator(device="cuda")
        generator.manual_seed(FMHA_RANDOM_SEED)

        _fill_fmha_random_(q, FMHA_Q_RANGE, FMHA_Q_SCALE, generator)
        if cfg.num_heads_kv == cfg.num_heads_q:
            _fill_fmha_random_(k, FMHA_K_RANGE, FMHA_K_SCALE, generator)
            _fill_fmha_random_(v, FMHA_V_RANGE, FMHA_BF16_FP16_V_SCALE, generator)
            return

        full_k = torch.empty(
            (self.num_tokens, cfg.num_heads_q, head_size), dtype=dtype, device="cuda"
        )
        full_v = torch.empty_like(full_k)
        _fill_fmha_random_(full_k, FMHA_K_RANGE, FMHA_K_SCALE, generator)
        _fill_fmha_random_(full_v, FMHA_V_RANGE, FMHA_BF16_FP16_V_SCALE, generator)
        heads_per_group = cfg.num_heads_q // cfg.num_heads_kv
        selected_heads = torch.arange(
            0, cfg.num_heads_q, heads_per_group, dtype=torch.long, device="cuda"
        )
        k.copy_(full_k.index_select(1, selected_heads))
        v.copy_(full_v.index_select(1, selected_heads))

    def plan(self, threshold: float, skip_softmax_stat: object | None = None) -> None:
        threshold_arg = float(threshold) if threshold > 0 else None
        self.wrapper.plan(
            layer_idx=0,
            tokens_per_block=None,
            max_num_requests=self.cfg.batch,
            max_sequence_length=self.cfg.seq_len_q,
            max_context_length=self.cfg.seq_len_q,
            attention_window_size=self.cfg.seq_len_q,
            sink_token_length=0,
            beam_width=1,
            sequence_length=self.sequence_length,
            host_past_key_value_lengths=self.host_past_key_value_lengths,
            host_total_kv_lens=self.host_total_kv_lens,
            context_lengths=self.context_lengths,
            host_context_lengths=self.host_context_lengths,
            host_request_types=self.host_request_types,
            workspace=self.workspace,
            use_paged_context_fmha=self.use_paged_context_fmha,
            attention_input_type=self.attention_input_type.context_only,
            sparse_attention_config=None,
            skip_softmax_threshold_scale_factor_prefill=threshold_arg,
            skip_softmax_threshold_scale_factor_decode=None,
            quant_config=None,
            kv_cache_manager=None,
        )
        self.wrapper.skip_softmax_stat = skip_softmax_stat
        self.wrapper.skip_softmax_stat_enabled = skip_softmax_stat is not None

    def run_kernel(self) -> None:
        self.wrapper.run(
            self.qkv,
            self.output,
            is_fused_qkv=True,
            update_kv_cache=True,
            attention_mask=self.attention_mask,
            num_contexts=self.cfg.batch,
            num_ctx_tokens=self.num_tokens,
        )

    def run(self, threshold: float, skip_softmax_stat: object | None = None) -> None:
        self.plan(threshold, skip_softmax_stat=skip_softmax_stat)
        self.run_kernel()


def _stat_row(cfg: FmhaConfig, threshold: float, total: int, skipped: int) -> dict[str, object]:
    return {
        "config": cfg.name,
        "dtype": cfg.dtype,
        "batch": cfg.batch,
        "num_heads_q": cfg.num_heads_q,
        "num_heads_kv": cfg.num_heads_kv,
        "head_size": cfg.head_size,
        "seq_len_q": cfg.seq_len_q,
        "seq_len_kv": cfg.seq_len_kv,
        "mask": cfg.mask,
        "threshold_scale_factor": threshold,
        "skipped_blocks": skipped,
        "total_blocks": total,
        "achieved_sparsity": skipped / total if total else None,
    }


def _get_bench_gpu_time() -> Callable[..., list[float]]:
    try:
        from flashinfer.testing import bench_gpu_time
    except ImportError as exc:
        raise RuntimeError(
            "The torch-op backend requires flashinfer.testing.bench_gpu_time. "
            "Install flashinfer's Python dependencies before running pass 2."
        ) from exc
    return bench_gpu_time


def _verify_single_fmha_kernel(runner: TorchOpFmhaRunner, threshold: float) -> None:
    torch = runner.torch
    for _ in range(DRY_RUN_ITERS):
        runner.run(threshold, skip_softmax_stat=None)
    torch.cuda.synchronize()

    with (
        torch.inference_mode(),
        torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler,
    ):
        runner.run(threshold, skip_softmax_stat=None)
    torch.cuda.synchronize()

    cuda_events = [
        event
        for event in profiler.events()
        if getattr(event, "device_type", None) == torch.autograd.DeviceType.CUDA
    ]
    event_names = [str(getattr(event, "name", getattr(event, "key", ""))) for event in cuda_events]
    fmha_events = [name for name in event_names if "fmha" in name.lower()]
    if len(fmha_events) != 1:
        raise AssertionError(
            f"expected exactly one FMHA CUDA event for {runner.cfg.name}, "
            f"got {len(fmha_events)} from events: {event_names}"
        )

    other_events = sorted(
        {
            name
            for name in event_names
            if "fmha" not in name.lower() and "memset" not in name.lower()
        }
    )
    if other_events:
        _warn(
            f"{runner.cfg.name}: profiler saw non-FMHA CUDA events besides memset: {other_events}"
        )


def _make_timed_fn(runner: TorchOpFmhaRunner, threshold: float) -> Callable[[], None]:
    """Returns a callable that invokes plan() + wrapper.run() per iter.

    wrapper.run() calls self.plan() with no args at exit (line 890 in
    trtllm.py) to drop tensor refs, so plan must be re-applied before every
    run. The plan() CPU work is amortized when the function is captured into
    a CUDA graph; the only kernels in the timed region are the FMHA prefill
    kernel plus QKV preprocessing (applyBiasRopeUpdateKVCache,
    computeSeqAndPaddingOffsets), which are constant overhead independent of
    the skip-softmax threshold.
    """

    def fn() -> None:
        runner.run(threshold, skip_softmax_stat=None)

    return fn


def pass1_sparsity(cfgs: list[FmhaConfig], init_mode: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cfg in cfgs:
        runner = TorchOpFmhaRunner(cfg, init_mode=init_mode)
        torch = runner.torch
        for threshold in cfg.threshold_sweep:
            print(f"[pass1] {cfg.name} threshold={threshold:g} ...", flush=True)
            stat = torch.zeros(2, dtype=torch.uint32, device="cuda")
            runner.run(threshold, skip_softmax_stat=stat)
            total, skipped = (int(value) for value in stat.cpu().tolist())
            if threshold > 0 and total == 0:
                raise RuntimeError(
                    f"{cfg.name} threshold={threshold:g}: skip-softmax stat "
                    f"produced zero total blocks, stat={[total, skipped]}"
                )
            rows.append(_stat_row(cfg, threshold, total, skipped))
        del runner
        torch.cuda.empty_cache()
    return rows


def pass2_speedup(cfgs: list[FmhaConfig], init_mode: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not cfgs:
        return rows
    bench_gpu_time = _get_bench_gpu_time()
    for cfg in cfgs:
        runner = TorchOpFmhaRunner(cfg, init_mode=init_mode)
        torch = runner.torch
        verify_threshold = cfg.threshold_sweep[0] if cfg.threshold_sweep else 0.0
        _verify_single_fmha_kernel(runner, verify_threshold)

        baseline_us: float | None = None
        for threshold in cfg.threshold_sweep:
            # CUDA Event + Graph mode: amortizes launch overhead via graph
            # capture, times via CUDA events. Cross-checks within ~5% of
            # fmha.exe on H200 prefill bf16. CUPTI mode (enable_cupti=True)
            # over-counts at low-sparsity baselines for reasons we have not
            # tracked down; not used here.
            times_ms = bench_gpu_time(
                _make_timed_fn(runner, threshold),
                dry_run_iters=DRY_RUN_ITERS,
                repeat_iters=REPEAT_ITERS,
                enable_cupti=False,
                use_cuda_graph=True,
                num_iters_within_graph=10,
            )
            elapsed_us = statistics.median(float(t) for t in times_ms) * 1000.0
            if threshold == 0:
                baseline_us = elapsed_us
            speedup = (baseline_us / elapsed_us) if baseline_us and elapsed_us else None
            print(
                f"[pass2] {cfg.name} threshold={threshold:g} "
                f"{elapsed_us:.2f}us speedup={speedup:.3f}x"
                if speedup is not None
                else f"[pass2] {cfg.name} threshold={threshold:g} {elapsed_us:.2f}us baseline",
                flush=True,
            )
            rows.append(
                {
                    "config": cfg.name,
                    "dtype": cfg.dtype,
                    "threshold_scale_factor": threshold,
                    "elapsed_us_median": elapsed_us,
                    "speedup": speedup,
                }
            )
        del runner
        torch.cuda.empty_cache()
    return rows


def write_csv(rows: list[dict[str, object]], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def _run_torch_op_backend(args: argparse.Namespace, cfgs: list[FmhaConfig]) -> int:
    cfgs = _torch_op_configs(cfgs)
    if not cfgs:
        _warn("no torch-op-compatible prefill configs selected")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if not cfgs:
        if not args.skip_pass1:
            write_csv([], args.out_dir / "pass1_sparsity.csv", PASS1_COLUMNS)
        if not args.skip_pass2:
            write_csv([], args.out_dir / "pass2_speedup.csv", PASS2_COLUMNS)
        print(f"done in {time.time() - started:.1f}s")
        return 0

    torch = _import_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("the torch-op backend requires a CUDA device")

    if not args.skip_pass1:
        rows1 = pass1_sparsity(cfgs, init_mode=args.init_mode)
        write_csv(rows1, args.out_dir / "pass1_sparsity.csv", PASS1_COLUMNS)
    if not args.skip_pass2:
        rows2 = pass2_speedup(cfgs, init_mode=args.init_mode)
        write_csv(rows2, args.out_dir / "pass2_speedup.csv", PASS2_COLUMNS)
    print(f"done in {time.time() - started:.1f}s")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=("llm", "diffusion", "both"), default="both")
    parser.add_argument(
        "--backend",
        choices=("torch-op",),
        default="torch-op",
        help=(
            "Benchmark backend. The old fmha-exe backend was removed from this "
            "attention-op benchmark; use bench_skip_softmax_fmha_exe.py for "
            "direct fmha.exe cross-validation."
        ),
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    threshold_group = parser.add_mutually_exclusive_group()
    threshold_group.add_argument(
        "--threshold-sweep", nargs="+", help="space- or comma-separated threshold values"
    )
    threshold_group.add_argument(
        "--threshold-multipliers",
        nargs="+",
        help="space- or comma-separated multipliers applied to each config's seq_len_kv",
    )
    parser.add_argument(
        "--config-filter",
        help="comma-separated config names to keep after loading the selected config family",
    )
    parser.add_argument(
        "--init-mode",
        choices=("fmha-random", "torch-normal"),
        default="fmha-random",
        help="torch-op input initializer; fmha-random matches fmha.exe value ranges",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--skip-pass1", action="store_true")
    parser.add_argument("--skip-pass2", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    threshold_sweep = _parse_threshold_sweep(args.threshold_sweep)
    threshold_multipliers = _parse_threshold_sweep(args.threshold_multipliers)
    config_filter = _parse_config_filter(args.config_filter)
    cfgs = _load_configs(
        args.config, threshold_sweep, threshold_multipliers, config_filter, args.batch_size
    )
    return _run_torch_op_backend(args, cfgs)


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real CUTLASS regression tests; requires a native build matching this checkout.

Also provides a manual operator benchmark (not model throughput):
    python test_workspace_cuda.py --benchmark --output /tmp/moe-workspace.json
Run separately with PYTORCH_CUDA_ALLOC_CONF=backend:native and
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync. Never compare allocator changes
as reclamation gains. No empty_cache or manual workspace resizing is used.
"""

import argparse
import json
import statistics
import time
from contextlib import nullcontext
from pathlib import Path
from threading import Event, Thread

import pytest
import torch

from tensorrt_llm._torch.custom_ops import torch_custom_ops as ops
from tensorrt_llm._torch.moe.workspace import CutlassWorkspaceReclaimer


class CutlassWorkload:
    """Fixed inputs shared by enabled/disabled arms, three calls per forward."""

    def __init__(self, dtype: torch.dtype = torch.bfloat16) -> None:
        generator = torch.Generator(device="cuda").manual_seed(19051)
        self.x = torch.randn(1024, 256, device="cuda", dtype=dtype, generator=generator)
        self.w1 = torch.randn(8, 512, 256, device="cuda", dtype=dtype, generator=generator) / 16
        self.w2 = torch.randn(8, 256, 256, device="cuda", dtype=dtype, generator=generator) / 16
        self.experts = torch.stack(
            (torch.arange(1024, device="cuda") % 8, (torch.arange(1024, device="cuda") + 1) % 8),
            dim=1,
        ).int()
        self.scales = torch.full((1024, 2), 0.5, device="cuda")

    def forward(
        self, tokens: int, scope: CutlassWorkspaceReclaimer | None, *, warmup: bool = False
    ) -> list[torch.Tensor]:
        with (
            scope.forward(warmup=warmup, num_tokens=tokens) if scope is not None else nullcontext()
        ):
            return [
                ops.fused_moe(
                    self.x[:n],
                    self.experts[:n],
                    self.scales[:n],
                    self.w1,
                    None,
                    self.w2,
                    None,
                    self.x.dtype,
                    [],
                    tune_max_num_tokens=1024,
                )[0]
                for n in (tokens, max(1, tokens // 2), max(1, tokens // 4))
            ]


def memory() -> dict[str, int]:
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "allocated_peak": torch.cuda.max_memory_allocated(),
        "reserved_peak": torch.cuda.max_memory_reserved(),
        # Phase-boundary device usage, deliberately not labelled device peak.
        "device_used_at_boundary": total - free,
    }


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cutlass_fifty_growth_reclaim_cycles(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype
) -> None:
    workload = CutlassWorkload(dtype)
    monkeypatch.setattr(ops.MoERunner, "runner_dict", {})
    references = {
        n: [output.cpu() for output in workload.forward(n, None)] for n in (32, 1024, 16, 8, 4)
    }
    # Use a separate native owner, preserving identical inputs/tactics. This
    # also prevents sharing with other tests from silently disabling reclaim.
    monkeypatch.setattr(ops.MoERunner, "runner_dict", {})
    scope = CutlassWorkspaceReclaimer()
    workload.forward(32, scope, warmup=True)
    baseline = memory()["allocated"]
    for _ in range(50):
        peak = 0
        for step, n in enumerate((1024, 16, 8, 4)):
            outputs = workload.forward(n, scope)
            for output, expected in zip(outputs, references[n]):
                torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
            del outputs, output
            retained = memory()["allocated"]
            if step == 0:
                peak = retained
                assert peak > baseline, "Real kernel sizing must grow beyond the initial allocation"
            elif step < 3:
                assert retained == peak, "Count complete forwards, not the three layer calls"
            else:
                assert retained <= baseline, "Third underfilled forward must release excess backing"
                assert retained < peak


def test_cutlass_reclaims_maximum_shape_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    workload = CutlassWorkload()
    monkeypatch.setattr(ops.MoERunner, "runner_dict", {})
    references = {n: [x.cpu() for x in workload.forward(n, None)] for n in (16, 8, 4)}
    monkeypatch.setattr(ops.MoERunner, "runner_dict", {})
    scope = CutlassWorkspaceReclaimer()
    workload.forward(1024, scope, warmup=True)
    peak = memory()["allocated"]
    for step, n in enumerate((16, 8, 4)):
        outputs = workload.forward(n, scope)
        for output, expected in zip(outputs, references[n]):
            torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        del outputs, output
        retained = memory()["allocated"]
        if step < 2:
            assert retained == peak
        else:
            assert retained < peak, "Normal maximum-shape warmup must not pin eager scratch"


def test_cutlass_graph_replay_after_eager_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    workload = CutlassWorkload()
    monkeypatch.setattr(ops.MoERunner, "runner_dict", {})
    scope = CutlassWorkspaceReclaimer()
    workload.forward(32, scope, warmup=True)
    expected = [x.cpu() for x in workload.forward(1024, scope)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        workload.forward(1024, None)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = workload.forward(1024, scope)
    for _ in range(5):
        workload.forward(4, scope)
    graph.replay()
    for output, reference in zip(outputs, expected):
        torch.testing.assert_close(output.cpu(), reference, rtol=0, atol=0)


def memory_sequence(workload: CutlassWorkload, enabled: bool) -> dict[str, object]:
    """Untimed native growth plus a synthetic later-buffer phase.

    NVML samples exclude process startup and warmup. The sequence is short, so
    its sampled peak is a lower bound; phase-boundary usage is also retained.
    """
    import pynvml

    ops.MoERunner.runner_dict = {}
    scope = CutlassWorkspaceReclaimer() if enabled else None
    workload.forward(32, scope, warmup=True)
    torch.cuda.synchronize()
    pynvml.nvmlInit()
    uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
    handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
    samples = []
    sampling_errors = []
    stop = Event()

    def sample() -> None:
        try:
            while not stop.is_set():
                samples.append(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
                stop.wait(0.005)
        except pynvml.NVMLError as error:
            sampling_errors.append(error)
            stop.set()

    sampler = Thread(target=sample, daemon=True)
    torch.cuda.reset_peak_memory_stats()

    def snapshot() -> dict[str, int]:
        stats = memory()
        stats["nvml_at_boundary"] = pynvml.nvmlDeviceGetMemoryInfo(handle).used
        samples.append(stats["nvml_at_boundary"])
        return stats

    phases = {"warmup": snapshot()}
    sampler.start()
    try:
        for label, tokens in (("burst", 1024), ("short1", 16), ("short2", 8), ("short3", 4)):
            workload.forward(tokens, scope)
            phases[label] = snapshot()
        baseline = phases["warmup"]["allocated"]
        peak = phases["burst"]["allocated"]
        assert peak > baseline, "The workload must actually grow native scratch"
        for label in ("short1", "short2"):
            assert phases[label]["allocated"] == peak
        if enabled:
            assert phases["short3"]["allocated"] <= baseline
            assert phases["short3"]["allocated"] < peak
        else:
            assert phases["short3"]["allocated"] == peak
        subsequent = torch.empty(64 * 1024 * 1024, dtype=torch.int8, device="cuda")
        subsequent.fill_(1)
        phases["synthetic_later_buffer"] = snapshot()
        del subsequent
        workload.forward(1024, scope)
        phases["regrowth"] = snapshot()
        assert phases["regrowth"]["allocated"] == peak
    finally:
        stop.set()
        sampler.join(timeout=5)
        pynvml.nvmlShutdown()
    if sampling_errors:
        raise sampling_errors[0]
    return {
        "phases": phases,
        "sampled_nvml_peak": max(samples) if samples else None,
        "nvml_samples": len(samples),
        "nvml_interval_ms": 5,
        "nvml_boundaries_included": True,
        "scope": "post-warmup only; synthetic later buffer; samples plus boundaries may miss transients",
    }


def benchmark(output: Path, runs: int, cycles: int) -> None:
    """Paired operator timings; correctness/memory checks stay outside timing."""
    workload = CutlassWorkload()
    original = ops.MoERunner.runner_dict
    records = []
    try:
        # Prime library/kernel initialization outside measured arms.
        ops.MoERunner.runner_dict = {}
        for tokens in (32, 1024, 16, 8, 4):
            workload.forward(tokens, None)
        torch.cuda.synchronize()
        for run in range(runs):
            for name, short_count in (("none", 0), ("low", 100), ("medium", 10), ("high", 3)):
                pair = []
                reference = None
                # Reverse order across independent runner pairs to limit drift.
                for enabled in (False, True) if run % 2 == 0 else (True, False):
                    ops.MoERunner.runner_dict = {}
                    scope = CutlassWorkspaceReclaimer() if enabled else None
                    workload.forward(16 if name == "none" else 32, scope, warmup=True)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start_memory = memory()
                    schedule = [16] * 400 if name == "none" else [1024] + [4] * short_count
                    start = time.perf_counter()
                    for _ in range(cycles):
                        for tokens in schedule:
                            workload.forward(tokens, scope)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    end_memory = memory()
                    # Compare full output tensors in an untimed phase.
                    actual = [x.cpu() for x in workload.forward(4, scope)]
                    if reference is None:
                        reference = actual
                    else:
                        for a, b in zip(actual, reference):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                    pair.append(
                        {
                            "enabled": enabled,
                            "elapsed_seconds": elapsed,
                            "forwards": cycles * len(schedule),
                            "forward_per_second": cycles * len(schedule) / elapsed,
                            "after_warmup": start_memory,
                            "after_workload": end_memory,
                        }
                    )
                records.append({"run": run, "frequency": name, "arms": pair})
        summary = {}
        for name in ("none", "low", "medium", "high"):
            changes = []
            for row in records:
                if row["frequency"] == name:
                    arms = {a["enabled"]: a for a in row["arms"]}
                    changes.append(
                        100 * (arms[True]["elapsed_seconds"] / arms[False]["elapsed_seconds"] - 1)
                    )
            summary[name] = {
                "paired_latency_change_percent": changes,
                "mean_percent": statistics.mean(changes),
            }
        output.write_text(
            json.dumps(
                {
                    "scope": "CUTLASS operator; not model throughput or production traffic",
                    "gpu": torch.cuda.get_device_name(),
                    "torch": torch.__version__,
                    "allocator": torch.cuda.get_allocator_backend(),
                    "short_forwards": {"low": 100, "medium": 10, "high": 3},
                    "high_frequency_note": "Three underfilled forwards per cycle; not three model requests.",
                    "device_measurement": "Timing arms use boundaries only; run --memory-only in fresh processes.",
                    "records": records,
                    "summary": summary,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        ops.MoERunner.runner_dict = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--benchmark", action="store_true")
    mode.add_argument("--memory-only", action="store_true")
    parser.add_argument("--enabled", type=int, choices=(0, 1), default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--cycles", type=int, default=10)
    args = parser.parse_args()
    if args.runs < 1 or args.cycles < 1:
        parser.error("runs and cycles must be positive")
    if args.memory_only:
        report = memory_sequence(CutlassWorkload(), bool(args.enabled))
        report.update(
            {
                "enabled": bool(args.enabled),
                "allocator": torch.cuda.get_allocator_backend(),
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
            }
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    else:
        benchmark(args.output, args.runs, args.cycles)

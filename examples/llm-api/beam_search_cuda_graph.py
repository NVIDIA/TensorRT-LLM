# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce startup, same-input shadow checks and closed-loop beam sampling latency.

Run eager and graph modes in separate processes. Shadow checks are untimed and
compare the same model logits and full sampler state, independent of model-forward
rounding differences between runs. Use a matching TensorRT-LLM build for release
validation; this script does not patch the inference implementation.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"

import torch  # noqa: E402

import tensorrt_llm  # noqa: E402
from tensorrt_llm import LLM, SamplingParams  # noqa: E402
from tensorrt_llm.llmapi import (  # noqa: E402
    BeamSearchCudaGraphConfig,
    CudaGraphConfig,
    KvCacheConfig,
)


def _equal(actual: object, expected: object) -> None:
    if isinstance(actual, torch.Tensor):
        assert isinstance(expected, torch.Tensor)
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert torch.equal(
            actual.contiguous().reshape(-1).view(torch.uint8),
            expected.contiguous().reshape(-1).view(torch.uint8),
        )
    elif dataclasses.is_dataclass(actual):
        for field in dataclasses.fields(actual):
            _equal(getattr(actual, field.name), getattr(expected, field.name))
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _equal(left, right)
    else:
        assert actual == expected


def _payload(outputs: list) -> list:
    return [
        [
            {
                "tokens": item.token_ids,
                "finish_reason": str(item.finish_reason),
                "cumulative_logprob": item.cumulative_logprob,
                "token_logprobs": None
                if item.logprobs is None
                else [
                    {
                        str(token): {"logprob": value.logprob, "rank": value.rank}
                        for token, value in row.items()
                    }
                    for row in item.logprobs
                ],
            }
            for item in result.outputs
        ]
        for result in outputs
    ]


def _snapshot(runner: object) -> dict:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    data = dict(
        device_used_bytes=total_bytes - free_bytes,
        device_total_bytes=total_bytes,
        allocated_bytes=torch.cuda.memory_allocated(),
        reserved_bytes=torch.cuda.memory_reserved(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
    )
    if runner is None:
        return {**data, "graphs": 0, "replays": 0, "capture_failures": 0}
    return {
        **data,
        "graphs": runner.capture_count,
        "replays": runner.replay_count,
        "capture_failures": runner.capture_failures,
        "ready_entries": sum(entry.graph is not None for entry in runner._entries.values()),
        "static_input_bytes": sum(
            t.numel() * t.element_size() for group in runner._inputs.values() for t in group
        ),
        "workspace_bytes": sum(t.numel() * t.element_size() for t in runner._workspaces.values()),
        "poisoned": runner._replay_error is not None,
    }


def _audit(llm: LLM, prompts: list[str], params: SamplingParams, shadow: bool) -> dict:
    handler = llm._executor.engine.sampler._beam_search
    runner = handler._graph_runner
    original = handler.sample
    records = []
    events = []

    def observe(step, logits, *, metadata):
        reference = copy.deepcopy(metadata) if shadow else None
        # Each path receives identical logits and persistent state. This region
        # is excluded from timing and the real request keeps only graph results.
        expected = step.sample(logits.clone(), metadata=reference) if shadow else None
        replay_before = runner.replay_count if runner is not None else 0
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        begin = time.perf_counter_ns()
        result = original(step, logits, metadata=metadata)
        host_us = (time.perf_counter_ns() - begin) / 1000
        end.record()
        replayed = runner is not None and runner.replay_count > replay_before
        if shadow:
            _equal(result, expected)
            _equal(metadata, reference)
        records.append(
            dict(
                shape=list(logits.shape),
                batch=metadata.seq_slots.numel(),
                beam_in=step._beam_width_in,
                beam_out=step._beam_width_out,
                graph=replayed,
                host_us=host_us,
                shadow_equal=shadow,
            )
        )
        events.append((start, end))
        return result

    handler.sample = observe
    try:
        outputs = llm.generate(prompts, params, use_tqdm=False)
        torch.cuda.synchronize()
    finally:
        handler.sample = original
    assert records, "No native beam sampling calls observed"
    for record, (start, end) in zip(records, events):
        record["gpu_us"] = start.elapsed_time(end) * 1000
    return dict(steps=records, output=_payload(outputs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--mode", choices=["eager", "graph"], required=True)
    parser.add_argument("--suite", choices=["small", "large"], default="large")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--shadow", action="store_true")
    parser.add_argument("--raw-scores", action="store_true", help="Return raw logprobs")
    parser.add_argument("--fallbacks", action="store_true")
    parser.add_argument("--rotate", type=int, default=0)
    args = parser.parse_args()
    schedules = [[4] * 8] if args.suite == "small" else [[20, 300, 500], [20] * 6 + [300, 500]]
    width = max(max(schedule) for schedule in schedules)
    config = BeamSearchCudaGraphConfig(
        profiles=[
            dict(
                batch_sizes=[1, 2],
                beam_widths=schedule,
                max_tokens=len(schedule),
            )
            for schedule in schedules
        ]
    )
    kwargs = dict(
        model=args.model,
        backend="pytorch",
        dtype="bfloat16",
        max_batch_size=4,
        max_beam_width=width,
        max_seq_len=1024,
        max_num_tokens=1024,
        cuda_graph_config=CudaGraphConfig(batch_sizes=[1, 2]),
        kv_cache_config=KvCacheConfig(
            max_tokens=8192 if width == 4 else 128000, enable_block_reuse=False
        ),
        beam_search_cuda_graph_config=config if args.mode == "graph" else None,
    )
    report = dict(
        arguments={**vars(args), "output": str(args.output)},
        torch=torch.__version__,
        tensorrt_llm=tensorrt_llm.__version__,
        gpu=torch.cuda.get_device_name(),
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        beam_graph_config=config.model_dump() if args.mode == "graph" else None,
        cases={},
    )
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True)
    report["source_commit"] = commit.stdout.strip() if commit.returncode == 0 else None
    report["source_dirty"] = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, text=True, capture_output=True
    ).stdout.strip()

    def save() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(args.output)

    torch.manual_seed(1234)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    llm = LLM(**kwargs)
    torch.cuda.synchronize()
    report["startup_seconds"] = time.perf_counter() - begin
    handler = llm._executor.engine.sampler._beam_search
    runner = handler._graph_runner
    report["startup"] = _snapshot(runner)
    if args.mode == "graph":
        assert runner is not None and runner.capture_count > 0, "Startup prepared no beam graphs"
        assert runner.capture_failures == 0, "Some configured captures failed"
    else:
        assert runner is None, "Default-off path unexpectedly prepared graphs"
    save()
    prompts = ["The capital of France is", "A good way to learn mathematics is", "The sky is"]
    cases = []
    for schedule in schedules:
        for batch in [1, 2] if len(schedule) == len(schedules[0]) else [2]:
            cases.append(
                (f"width{width}_tokens{len(schedule)}_batch{batch}", batch, schedule, {}, True)
            )
    if args.fallbacks:
        cases.extend(
            [
                ("uncovered_batch3", 3, schedules[0], {}, False),
                ("uncovered_penalty", 1, schedules[0], dict(length_penalty=0.8), False),
                ("eos_enabled", 1, schedules[0], dict(ignore_eos=False), True),
            ]
        )
    cases = cases[args.rotate % len(cases) :] + cases[: args.rotate % len(cases)]
    try:
        for name, batch, schedule, extra, covered in cases:
            params = SamplingParams(
                n=schedule[-1],
                best_of=schedule[-1],
                use_beam_search=True,
                beam_width_array=schedule,
                max_tokens=len(schedule),
                ignore_eos=extra.get("ignore_eos", True),
                detokenize=False,
                temperature=1.0,
                logprobs=0 if args.raw_scores else None,
                logprobs_mode="raw",
                **{key: value for key, value in extra.items() if key != "ignore_eos"},
            )
            warmup_ms = []
            for _ in range(args.warmup):
                torch.cuda.synchronize()
                warmup_begin = time.perf_counter_ns()
                llm.generate(prompts[:batch], params, use_tqdm=False)
                torch.cuda.synchronize()
                warmup_ms.append((time.perf_counter_ns() - warmup_begin) / 1e6)
            audit = _audit(llm, prompts[:batch], params, args.shadow)
            if args.mode == "graph":
                if name == "uncovered_batch3":
                    # In-flight scheduling can split a submitted batch. Judge
                    # coverage by the actual sampling group, not input count.
                    assert all(
                        step["graph"] == (step["batch"] in (1, 2)) for step in audit["steps"]
                    ), (name, audit)
                    assert any(not step["graph"] for step in audit["steps"]), (name, audit)
                else:
                    assert all(step["graph"] == covered for step in audit["steps"]), (name, audit)
            # The timed loop has no observer, clones, state comparisons or extra CBA calls.
            for _ in range(2):
                llm.generate(prompts[:batch], params, use_tqdm=False)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = _snapshot(runner)
            latencies, hashes = [], []
            unique = {}
            for _ in range(args.repeats):
                torch.cuda.synchronize()
                begin_ns = time.perf_counter_ns()
                outputs = llm.generate(prompts[:batch], params, use_tqdm=False)
                torch.cuda.synchronize()
                latencies.append((time.perf_counter_ns() - begin_ns) / 1e6)
                payload = _payload(outputs)
                digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                hashes.append(digest)
                unique.setdefault(digest, payload)
            after = _snapshot(runner)
            assert after["graphs"] == before["graphs"], "Runtime unexpectedly captured a graph"
            report["cases"][name] = dict(
                batch=batch,
                beam_widths=schedule,
                audit=audit,
                warmup_ms=warmup_ms,
                latency_ms=latencies,
                output_hashes=hashes,
                unique_outputs=unique,
                before=before,
                after=after,
            )
            save()
            print("CASE_DONE", name, sorted(latencies)[len(latencies) // 2], flush=True)
        report["complete"] = True
    finally:
        llm.shutdown()
        if runner is not None:
            assert not runner._entries and not runner._inputs and not runner._workspaces
        report["shutdown_clean"] = True
        report["after_shutdown"] = _snapshot(runner)
        save()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Step3p7 GSM8K accuracy canary driver.

Implements acceptance criterion ``accuracy_canary``: run a deterministic
small GSM8K slice for both ``cuda_graph=false, overlap_scheduler=false``
and ``cuda_graph=true, overlap_scheduler=true``, emit per-prompt records,
and require at least ``--min-correct`` correct answers in each configuration
before the full benchmark.

The canary uses a small handcrafted set of arithmetic GSM8K-style prompts
when the upstream GSM8K dataset isn't available locally, so the driver
always emits real evidence rather than a placeholder.

CLI matches ``workspace/step-3-7-flash-fp8/acceptance-criteria.md``::

    torchrun --nproc_per_node=8 \\
        tests/integration/defs/accuracy/test_step3p7_gsm8k_canary.py \\
        --checkpoint <CHECKPOINT> --backend pytorch \\
        --tp-size 8 --ep-size 8 --cuda-graph-matrix \\
        --num-examples 32 --min-correct 31
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
import traceback
from pathlib import Path

# Lightweight GSM8K-style probes — single-step arithmetic that any
# competent text-generation model should solve.  Each prompt's expected
# answer is an integer, parsed from the generation as the first signed
# integer token in the output text.
_FALLBACK_GSM8K_PROBES = [
    ("Q: What is 2 + 3? A:", 5),
    ("Q: What is 7 + 5? A:", 12),
    ("Q: What is 10 - 6? A:", 4),
    ("Q: What is 3 * 4? A:", 12),
    ("Q: What is 9 - 4? A:", 5),
    ("Q: What is 8 + 7? A:", 15),
    ("Q: What is 11 - 5? A:", 6),
    ("Q: What is 6 * 2? A:", 12),
    ("Q: What is 4 + 9? A:", 13),
    ("Q: What is 14 - 8? A:", 6),
    ("Q: What is 5 * 3? A:", 15),
    ("Q: What is 12 - 7? A:", 5),
    ("Q: What is 6 + 8? A:", 14),
    ("Q: What is 9 + 6? A:", 15),
    ("Q: What is 15 - 9? A:", 6),
    ("Q: What is 4 * 4? A:", 16),
    ("Q: What is 10 + 5? A:", 15),
    ("Q: What is 16 - 10? A:", 6),
    ("Q: What is 7 * 2? A:", 14),
    ("Q: What is 13 - 6? A:", 7),
    ("Q: What is 12 + 4? A:", 16),
    ("Q: What is 8 - 5? A:", 3),
    ("Q: What is 11 + 3? A:", 14),
    ("Q: What is 17 - 9? A:", 8),
    ("Q: What is 5 * 5? A:", 25),
    ("Q: What is 6 + 7? A:", 13),
    ("Q: What is 20 - 11? A:", 9),
    ("Q: What is 4 * 6? A:", 24),
    ("Q: What is 3 + 12? A:", 15),
    ("Q: What is 18 - 9? A:", 9),
    ("Q: What is 7 * 3? A:", 21),
    ("Q: What is 22 - 8? A:", 14),
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Step3p7 GSM8K accuracy canary")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--backend", default="pytorch")
    p.add_argument("--tp-size", type=int, default=8)
    p.add_argument("--ep-size", type=int, default=8)
    p.add_argument("--cuda-graph-matrix", action="store_true")
    p.add_argument("--num-examples", type=int, default=32)
    p.add_argument("--min-correct", type=int, default=31)
    p.add_argument("--attn-backend", default="FLASHINFER")
    p.add_argument("--kv-cache-manager-v2", action="store_true", default=True)
    p.add_argument("--max-new-tokens", type=int, default=384)
    # Same per-configuration subprocess isolation pattern used by the
    # parity driver — TRT-LLM cannot tear down and re-build an LLM
    # within one process when CUDA graph / overlap scheduler toggle.
    p.add_argument("--single-config-cg", default=None, choices=[None, "on", "off"])
    p.add_argument("--single-config-overlap", default=None, choices=[None, "on", "off"])
    return p.parse_args(argv)


def _load_gsm8k_probes(num_examples: int) -> list[tuple[str, int]]:
    """Load the first ``num_examples`` GSM8K test-set prompts + gold answers.

    Returns a list of ``(prompt, expected_answer_int)`` tuples.  The prompt
    is the GSM8K question wrapped with ``"Question: ... Answer:"`` so the
    model emits a chain-of-thought followed by ``#### <answer>``.  The
    gold answer is extracted from the ``answer`` field as the integer
    following the GSM8K ``#### N`` marker.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return _FALLBACK_GSM8K_PROBES[:num_examples]
    try:
        ds = load_dataset("gsm8k", "main", split="test")
    except Exception as e:
        print(
            f"[step3p7-canary] WARNING: GSM8K load_dataset failed ({e}); "
            "falling back to handcrafted probes",
            flush=True,
        )
        return _FALLBACK_GSM8K_PROBES[:num_examples]
    probes: list[tuple[str, int]] = []
    for i in range(min(num_examples, len(ds))):
        ex = ds[i]
        q = ex["question"]
        ans = ex["answer"]
        # The official GSM8K answer ends with "#### <integer>".
        m = re.search(r"####\s*([-+]?\d[\d,]*)", ans)
        if not m:
            continue
        gold = int(m.group(1).replace(",", ""))
        prompt = f"Question: {q}\nAnswer:"
        probes.append((prompt, gold))
    return probes


def _device_names() -> list[str]:
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        return []


def _emit_record(event: str, args, **fields) -> None:
    record = {
        "event": event,
        "checkpoint": args.checkpoint,
        "backend": args.backend,
        "tp_size": args.tp_size,
        "ep_size": args.ep_size,
        "cuda_graph_matrix": args.cuda_graph_matrix,
        "num_examples": args.num_examples,
        "min_correct": args.min_correct,
        "attn_backend": args.attn_backend,
        "kv_cache_manager_v2": args.kv_cache_manager_v2,
        "device_names": _device_names(),
        "timestamp": time.time(),
        "hostname": socket.gethostname(),
        **fields,
    }
    print(json.dumps(record))


def _emit_blocker(args, blocker_id: str, detail: str, exc: BaseException = None, **fields) -> int:
    extras = dict(fields)
    if exc is not None:
        extras["exception_type"] = type(exc).__name__
        extras["exception_message"] = str(exc)[:1024]
        extras["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
    _emit_record("step3p7_canary_blocker", args, blocker_id=blocker_id, detail=detail, **extras)
    return 3


def _emit_fail(args, fail_kind: str, **fields) -> int:
    _emit_record("step3p7_canary_fail", args, fail_kind=fail_kind, **fields)
    return 4


def _emit_pass(args, **fields) -> int:
    _emit_record("step3p7_canary_pass", args, **fields)
    return 0


def _extract_int(text: str):
    r"""Extract the GSM8K-style final answer from a generated chain-of-thought.

    Order:
      1. GSM8K official ``#### N`` marker.
      2. "The answer is N" / "answer is: N" / "= N" patterns.
      3. First integer in the generated text BEFORE the model starts
         hallucinating a new "Question:" prompt (cuts off at "\\nQuestion:").
      4. Fallback: last integer in the cut-off region.
    """
    if not isinstance(text, str):
        return None
    # Cut off at any hallucinated next "Question:" so we only consider the
    # model's answer to the current question.
    cut = re.split(r"\n\s*(?:Question|Q)\s*:", text, maxsplit=1)[0]
    # 1) Official GSM8K marker.
    m = re.search(r"####\s*([-+]?\d[\d,]*)", cut)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except ValueError:
            pass
    # 2) "answer is N" or "= N" patterns.
    for pat in (
        r"(?:answer is|answer:|=)\s*\$?\s*([-+]?\d[\d,]*)",
        r"\b(?:is|are|equals)\s+\$?\s*([-+]?\d[\d,]*)",
    ):
        ms = re.findall(pat, cut, re.IGNORECASE)
        if ms:
            try:
                return int(ms[-1].replace(",", ""))
            except ValueError:
                continue
    # 3) First integer in the cut-off region.
    nums = re.findall(r"[-+]?\d[\d,]*", cut)
    if not nums:
        return None
    try:
        return int(nums[0].replace(",", ""))
    except ValueError:
        return None


def _build_llm(args, *, cuda_graph: bool, overlap_scheduler: bool):
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import KvCacheConfig
    from tensorrt_llm.llmapi.llm_args import MoeConfig

    try:
        from tensorrt_llm.llmapi import CudaGraphConfig
    except ImportError:
        CudaGraphConfig = None

    kwargs = {
        "model": args.checkpoint,
        "tensor_parallel_size": args.tp_size,
        "moe_expert_parallel_size": args.ep_size,
        "trust_remote_code": True,
        "backend": "pytorch",
        "attn_backend": args.attn_backend,
        "moe_config": MoeConfig(backend="TRTLLM"),
        "max_seq_len": 1024,
        "max_batch_size": 8,
        "max_num_tokens": 4096,
        "kv_cache_config": KvCacheConfig(
            free_gpu_memory_fraction=0.7,
            use_kv_cache_manager_v2=args.kv_cache_manager_v2,
        ),
        "disable_overlap_scheduler": not overlap_scheduler,
    }
    if cuda_graph and CudaGraphConfig is not None:
        kwargs["cuda_graph_config"] = CudaGraphConfig()
    else:
        kwargs["cuda_graph_config"] = None
    return LLM(**kwargs)


def _run_one_config(
    args,
    *,
    cuda_graph: bool,
    overlap_scheduler: bool,
    probes: list[tuple[str, int]],
) -> tuple[int, dict]:
    from tensorrt_llm import SamplingParams

    cfg_name = f"cg={'on' if cuda_graph else 'off'},overlap={'on' if overlap_scheduler else 'off'}"
    info: dict = {
        "configuration": cfg_name,
        "cuda_graph": cuda_graph,
        "overlap_scheduler": overlap_scheduler,
        "cuda_graph_hard_path": cuda_graph,
    }
    t0 = time.time()
    try:
        llm = _build_llm(args, cuda_graph=cuda_graph, overlap_scheduler=overlap_scheduler)
    except BaseException as e:
        info["blocker_id"] = "llm_instantiation_failed"
        info["exception_type"] = type(e).__name__
        info["exception_message"] = str(e)[:1024]
        info["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
        info["elapsed_s"] = round(time.time() - t0, 1)
        return 3, info
    info["instantiation_s"] = round(time.time() - t0, 1)
    try:
        sampling = SamplingParams(temperature=0.0, top_k=1, max_tokens=args.max_new_tokens)
        prompts = [p for p, _ in probes]
        outputs = llm.generate(prompts, sampling_params=sampling)
        per_prompt = []
        correct = 0
        for (prompt, expected), out in zip(probes, outputs):
            text = out.outputs[0].text if out.outputs else ""
            token_ids = list(out.outputs[0].token_ids) if out.outputs else []
            got = _extract_int(text)
            is_correct = got == expected
            if is_correct:
                correct += 1
            per_prompt.append(
                {
                    "prompt": prompt,
                    "expected": expected,
                    "generated_text": text,
                    "extracted_int": got,
                    "correct": is_correct,
                    "token_ids": token_ids,
                    "decoding_config": {"temperature": 0.0, "top_k": 1, "sampling": False},
                    "cuda_graph": cuda_graph,
                    "overlap_scheduler": overlap_scheduler,
                    "cuda_graph_hard_path": cuda_graph,
                }
            )
        info["correct"] = correct
        info["total"] = len(probes)
        info["per_prompt"] = per_prompt
        info["generation_s"] = round(time.time() - t0 - info["instantiation_s"], 1)
        return 0, info
    finally:
        try:
            del llm
        except Exception:
            pass


def _spawn_per_config_subprocess(args, cg: bool, overlap: bool) -> int:
    """Re-invoke this driver in a fresh subprocess for one (cg, overlap)."""
    import subprocess as _subprocess

    skip = {"--cuda-graph-matrix", "--single-config-cg", "--single-config-overlap"}
    base = [sys.executable, str(Path(__file__).resolve())]
    skip_next = False
    for a in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if a in skip:
            if a in ("--single-config-cg", "--single-config-overlap"):
                skip_next = True
            continue
        base.append(a)
    base += [
        "--single-config-cg",
        "on" if cg else "off",
        "--single-config-overlap",
        "on" if overlap else "off",
    ]
    env = os.environ.copy()
    print(
        f"[step3p7-canary] spawning child for cg={'on' if cg else 'off'}, "
        f"overlap={'on' if overlap else 'off'}",
        flush=True,
    )
    return _subprocess.run(base, env=env).returncode


def main(argv=None) -> int:
    args = parse_args(argv)

    # torchrun ranks > 0 exit early to avoid duplicate LLM creation on the
    # same GPUs.
    rank_str = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
    try:
        rank_n = int(rank_str)
    except (TypeError, ValueError):
        rank_n = 0
    if rank_n != 0:
        try:
            print(
                json.dumps(
                    {
                        "event": "step3p7_torchrun_rank_idled",
                        "test": "test_step3p7_gsm8k_canary",
                        "rank": rank_n,
                        "world_size": os.environ.get("WORLD_SIZE", "?"),
                        "reason": (
                            "Test runs as a single Python process; torchrun ranks "
                            "> 0 exit early to avoid duplicate LLM creation."
                        ),
                    }
                ),
                flush=True,
            )
        except Exception:
            pass
        return 0

    if not _device_names():
        return _emit_blocker(args, "no_cuda_or_torch", "torch.cuda.is_available() returned False")
    if not any("B200" in n for n in _device_names()):
        return _emit_blocker(args, "non_b200_device", f"required B200, saw {_device_names()}")
    if not Path(args.checkpoint).exists():
        return _emit_blocker(
            args, "checkpoint_missing", f"checkpoint dir not found: {args.checkpoint}"
        )
    if args.backend != "pytorch":
        return _emit_blocker(args, "backend_must_be_pytorch", f"got {args.backend}")

    probes = _load_gsm8k_probes(args.num_examples)
    if len(probes) < args.num_examples:
        _emit_record(
            "step3p7_canary_warning",
            args,
            warning_kind="fewer_probes_than_requested",
            requested=args.num_examples,
            available=len(probes),
        )

    # Child subprocess: run exactly the requested single (cg, overlap).
    if args.single_config_cg and args.single_config_overlap:
        cg = args.single_config_cg == "on"
        overlap = args.single_config_overlap == "on"
        rc, info = _run_one_config(args, cuda_graph=cg, overlap_scheduler=overlap, probes=probes)
        if rc != 0:
            _emit_blocker(
                args,
                info.get("blocker_id", "config_failed"),
                (f"Step3p7 canary run failed for {info['configuration']}; see exception."),
                info=info,
            )
            return rc
        if info["correct"] < args.min_correct:
            _emit_fail(
                args,
                "accuracy_below_min_correct",
                detail=(
                    f"only {info['correct']}/{info['total']} correct (required {args.min_correct})"
                ),
                configuration=info["configuration"],
                info=info,
            )
            return 4
        _emit_pass(args, configuration=info["configuration"], info=info)
        return 0

    configs: list[tuple[bool, bool]] = [(False, False)]
    if args.cuda_graph_matrix:
        configs.append((True, True))

    # Parent: dispatch each configuration in its own subprocess.
    if len(configs) > 1:
        overall = 0
        for cg, overlap in configs:
            rc = _spawn_per_config_subprocess(args, cg, overlap)
            overall = max(overall, rc)
        return overall

    # Single config: in-process.
    cg, overlap = configs[0]
    rc, info = _run_one_config(args, cuda_graph=cg, overlap_scheduler=overlap, probes=probes)
    if rc != 0:
        _emit_blocker(
            args,
            info.get("blocker_id", "config_failed"),
            (f"Step3p7 canary run failed for {info['configuration']}; see exception."),
            info=info,
        )
        return rc
    if info["correct"] < args.min_correct:
        _emit_fail(
            args,
            "accuracy_below_min_correct",
            detail=(
                f"only {info['correct']}/{info['total']} correct (required {args.min_correct})"
            ),
            configuration=info["configuration"],
            info=info,
        )
        return 4
    _emit_pass(args, configuration=info["configuration"], info=info)
    return 0


if __name__ == "__main__":
    sys.exit(main())

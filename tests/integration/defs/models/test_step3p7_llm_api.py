#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step3p7 LLM API smoke-test driver.

Implements acceptance criterion ``real_runtime`` LLM API smoke for Step3p7
text generation.  Loads the production TRT-LLM PyTorch backend with
``KVCacheManagerV2``, ``FLASHINFER`` attention, and the selected FP8 MoE
backend, and runs a short context+decode for both the baseline
(``cuda_graph=false, overlap_scheduler=false``) and enabled
(``cuda_graph=true, overlap_scheduler=true``) configurations.

When any pass-critical step fails, the driver emits a structured blocker
record with the concrete failure mode (CUDA error, missing weight, etc.)
so the next iteration's Coder can resolve a specific code path rather than
a generic 'pending' marker.

CLI matches ``workspace/step-3-7-flash-fp8/acceptance-criteria.md``::

    torchrun --nproc_per_node=8 tests/integration/defs/models/test_step3p7_llm_api.py \
        --checkpoint <CHECKPOINT> --kv-cache-manager-v2 \
        --attn-backend FLASHINFER --cuda-graph-matrix
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Step3p7 LLM API smoke")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--kv-cache-manager-v2", action="store_true")
    p.add_argument("--attn-backend", default="FLASHINFER")
    p.add_argument("--cuda-graph-matrix", action="store_true")
    p.add_argument("--tp-size", type=int, default=8)
    p.add_argument("--ep-size", type=int, default=8)
    p.add_argument("--max-batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=32)
    # TRT-LLM cannot tear down and re-construct an LLM inside the same
    # Python process when CUDA graph / overlap scheduler are toggled between
    # runs; the MPI / executor worker state persists and the second LLM()
    # construction fails with ``RuntimeError: Executor worker returned error``.
    # The parent process therefore re-invokes itself once per configuration
    # with ``--single-config-cg`` / ``--single-config-overlap`` set so each
    # configuration owns its own subprocess.
    p.add_argument("--single-config-cg", default=None, choices=[None, "on", "off"])
    p.add_argument("--single-config-overlap", default=None, choices=[None, "on", "off"])
    return p.parse_args(argv)


def _emit_pending(reason: str, args: argparse.Namespace, extra: dict = None) -> int:
    record = {
        "event": "step3p7_llm_api_pending",
        "checkpoint": args.checkpoint,
        "kv_cache_manager_v2": args.kv_cache_manager_v2,
        "attn_backend": args.attn_backend,
        "cuda_graph_matrix": args.cuda_graph_matrix,
        "reason": reason,
        "evidence_required": True,
        "timestamp": time.time(),
        "hostname": socket.gethostname(),
    }
    if extra:
        record.update(extra)
    print(json.dumps(record))
    return 2


def _emit_blocker(
    args: argparse.Namespace,
    blocker_id: str,
    detail: str,
    exc: BaseException = None,
    config: str = None,
) -> int:
    record = {
        "event": "step3p7_llm_api_blocker",
        "checkpoint": args.checkpoint,
        "blocker_id": blocker_id,
        "detail": detail,
        "configuration": config,
        "timestamp": time.time(),
    }
    if exc is not None:
        record["exception_type"] = type(exc).__name__
        record["exception_message"] = str(exc)[:1024]
        record["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
    print(json.dumps(record))
    return 3


def _run_config(args: argparse.Namespace, cuda_graph: bool, overlap_scheduler: bool) -> int:
    """Run a single (cuda_graph, overlap_scheduler) configuration."""
    config_name = (
        f"cg={'on' if cuda_graph else 'off'},overlap={'on' if overlap_scheduler else 'off'}"
    )
    try:
        from tensorrt_llm import LLM, SamplingParams
        from tensorrt_llm.llmapi import KvCacheConfig
        from tensorrt_llm.llmapi.llm_args import MoeConfig

        try:
            from tensorrt_llm.llmapi import CudaGraphConfig
        except ImportError:
            CudaGraphConfig = None
    except Exception as e:
        return _emit_blocker(
            args,
            "trtllm_import_failed",
            "tensorrt_llm could not be imported",
            exc=e,
            config=config_name,
        )

    kwargs = {
        "model": args.checkpoint,
        "tensor_parallel_size": args.tp_size,
        "moe_expert_parallel_size": args.ep_size,
        "trust_remote_code": True,
        "backend": "pytorch",
        "attn_backend": args.attn_backend,
        "moe_config": MoeConfig(backend="TRTLLM"),
        "max_seq_len": 4096,
        "max_batch_size": args.max_batch_size,
        "max_num_tokens": 8192,
        "kv_cache_config": KvCacheConfig(
            free_gpu_memory_fraction=0.6,
            use_kv_cache_manager_v2=args.kv_cache_manager_v2,
        ),
        "disable_overlap_scheduler": not overlap_scheduler,
    }
    if cuda_graph and CudaGraphConfig is not None:
        # Non-default CudaGraphConfig instantiates the hard-path capture/replay
        # path; passing None here would set ``cuda_graph=false`` for the runtime.
        kwargs["cuda_graph_config"] = CudaGraphConfig()
    else:
        kwargs["cuda_graph_config"] = None

    try:
        llm = LLM(**kwargs)
    except BaseException as e:
        return _emit_blocker(
            args,
            "llm_instantiation_failed",
            (
                "LLM API failed to construct Step3p7 / load weights. Investigate"
                " the exception traceback below."
            ),
            exc=e,
            config=config_name,
        )

    prompts = [
        "Q: What is 2 + 2? A:",
        "Q: What is the capital of France? A:",
    ]
    sampling = SamplingParams(temperature=0.0, top_k=1, max_tokens=args.max_new_tokens)
    try:
        outputs = llm.generate(prompts, sampling_params=sampling)
        per_prompt = []
        for prompt, o in zip(prompts, outputs):
            text = o.outputs[0].text if o.outputs else ""
            token_ids = list(o.outputs[0].token_ids) if o.outputs else []
            per_prompt.append({"prompt": prompt, "text": text, "token_ids": token_ids})
        # Detect the iter-3/iter-4 math bug: every prompt yields the same
        # tokens (or each prompt collapses to a single repeated token).
        unique_token_sequences = {tuple(p["token_ids"]) for p in per_prompt}
        record = {
            "event": "step3p7_llm_api_smoke",
            "configuration": config_name,
            "checkpoint": args.checkpoint,
            "outputs": per_prompt,
            "kv_cache_manager_v2": args.kv_cache_manager_v2,
            "attn_backend": args.attn_backend,
            "device_names": _device_names(),
            "timestamp": time.time(),
            "cuda_graph_hard_path": cuda_graph and CudaGraphConfig is not None,
            "unique_token_sequence_count": len(unique_token_sequences),
        }
        if len(unique_token_sequences) <= 1 and len(per_prompt) > 1:
            record["event"] = "step3p7_llm_api_smoke_fail"
            record["fail_kind"] = "all_prompts_emitted_identical_tokens"
            print(json.dumps(record))
            return 4
        record["event"] = "step3p7_llm_api_smoke_ok"
        print(json.dumps(record))
        return 0
    except BaseException as e:
        return _emit_blocker(
            args,
            "generation_failed",
            "generate() raised during context or decode",
            exc=e,
            config=config_name,
        )
    finally:
        try:
            del llm
        except Exception:
            pass


def _device_names():
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        return []


def main(argv=None) -> int:
    args = parse_args(argv)

    # See `test_step3p7_parity.py::_torchrun_rank_gate` — duplicate Python
    # processes (one per torchrun rank) would each create their own LLM with
    # internal MPI and OOM the box.  Idle ranks >0 here.
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
                        "test": "test_step3p7_llm_api",
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

    try:
        import torch

        if not torch.cuda.is_available():
            return _emit_pending("cuda_unavailable", args)
        device_name = torch.cuda.get_device_name(0)
        if "B200" not in device_name:
            return _emit_pending(f"non_b200_device:{device_name}", args)
    except ImportError:
        return _emit_pending("torch_unimportable", args)

    if not Path(args.checkpoint).exists():
        return _emit_pending(f"checkpoint_missing:{args.checkpoint}", args)
    if not args.kv_cache_manager_v2:
        return _emit_pending("kv_cache_manager_v2_required", args)
    if args.attn_backend.upper() != "FLASHINFER":
        return _emit_pending(f"attn_backend_must_be_FLASHINFER_got_{args.attn_backend}", args)

    # Child subprocess invocation: do exactly the requested single config.
    if args.single_config_cg and args.single_config_overlap:
        return _run_config(
            args,
            cuda_graph=(args.single_config_cg == "on"),
            overlap_scheduler=(args.single_config_overlap == "on"),
        )

    configs = [(False, False)]
    if args.cuda_graph_matrix:
        configs.append((True, True))

    overall = 0
    if len(configs) == 1:
        cg, overlap = configs[0]
        return _run_config(args, cuda_graph=cg, overlap_scheduler=overlap)
    # Multi-config matrix: spawn a fresh subprocess per configuration.
    import subprocess as _subprocess

    skip = {"--single-config-cg", "--single-config-overlap", "--cuda-graph-matrix"}
    base_argv = [sys.executable, str(Path(__file__).resolve())]
    for a in sys.argv[1:]:
        if a in skip:
            continue
        base_argv.append(a)
    env = os.environ.copy()
    for cg, overlap in configs:
        child_argv = base_argv + [
            "--single-config-cg",
            "on" if cg else "off",
            "--single-config-overlap",
            "on" if overlap else "off",
        ]
        print(
            f"[step3p7-llm-api] spawning child for cg={'on' if cg else 'off'},"
            f" overlap={'on' if overlap else 'off'}",
            flush=True,
        )
        proc = _subprocess.run(child_argv, env=env)
        if proc.returncode != 0:
            overall = proc.returncode
    return overall


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workload for profile_deepseek_v4_offload.py; run that launcher directly."""

import argparse
import json
from pathlib import Path
from time import perf_counter


def _build_engine_config(settings: dict) -> dict:
    """Resolve the experiment without importing CUDA libraries in the launcher."""
    model = Path(settings["model"]).expanduser().resolve(strict=True)
    checkpoint = json.loads((model / "config.json").read_text())
    if "DeepseekV4ForCausalLM" not in checkpoint.get("architectures", []):
        raise ValueError("--model must contain a DeepseekV4ForCausalLM checkpoint")
    if checkpoint.get("vocab_size", 0) < 3:
        raise ValueError("The checkpoint must declare vocab_size >= 3")
    config = {}
    if settings.get("config"):
        import yaml

        config = yaml.safe_load(Path(settings["config"]).expanduser().read_text())
        if not isinstance(config, dict):
            raise ValueError("--config must contain a mapping of LLM API options")

    for key in ("batch_size", "input_len", "output_len", "host_cache_mib"):
        if settings[key] <= 0:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    if settings["warmup_runs"] < 0:
        raise ValueError("--warmup-runs must be nonnegative")
    if not 0 < settings["kv_cache_fraction"] < 1:
        raise ValueError("--kv-cache-fraction must be between 0 and 1")
    page_size = settings["tokens_per_block"]
    if settings["input_len"] < page_size:
        raise ValueError("--input-len must cover at least one complete KV page")
    # Only output_len - 1 generated tokens have their KV written by completion.
    next_boundary = page_size - settings["input_len"] % page_size
    if settings["output_len"] < next_boundary + 2:
        raise ValueError(
            f"--output-len must be >= {next_boundary + 2} to copy a new page during decode"
        )
    steps = settings.get("profile_steps")
    if steps is not None:
        if settings["capture"] != "range":
            raise ValueError("--profile-steps requires --capture range")
        if steps < next_boundary + 3 or steps > settings["output_len"] - 3:
            raise ValueError(
                "--profile-steps must include the next page boundary and leave at least "
                f"three output iterations for verification: {next_boundary + 3} <= steps "
                f"<= {settings['output_len'] - 3}"
            )

    for key in ("tensor_parallel_size", "moe_expert_parallel_size"):
        option = "tp_size" if key == "tensor_parallel_size" else "ep_size"
        if settings.get(option) is not None:
            config[key] = settings[option]
    config.setdefault("tensor_parallel_size", 8)
    if config.get("moe_expert_parallel_size") is None:
        config["moe_expert_parallel_size"] = config["tensor_parallel_size"]
    tp, ep = config["tensor_parallel_size"], config["moe_expert_parallel_size"]
    if tp <= 0 or ep <= 0 or tp % ep:
        raise ValueError("TP and EP must be positive, and EP must divide TP")
    if config.get("pipeline_parallel_size", 1) != 1 or config.get("context_parallel_size", 1) != 1:
        raise ValueError("This profiling launcher supports one-node TP/EP with PP=CP=1")
    for key in ("cache_transceiver_config", "kv_connector_config"):
        if config.get(key) is not None:
            raise ValueError(f"{key} is unsupported by this copy-only experiment")

    sparse = dict(config.get("sparse_attention_config") or {})
    ratios = sparse.get("compress_ratios", checkpoint.get("compress_ratios"))
    if not ratios or len(ratios) < checkpoint["num_hidden_layers"] or 4 not in ratios:
        raise ValueError("The checkpoint/config must specify per-layer ratios including ratio 4")
    sparse.update(algorithm="deepseek_v4", compress_ratios=ratios, enable_kv_cache_offload=True)
    if "window_size" not in sparse:
        sparse["window_size"] = checkpoint.get("sliding_window", checkpoint.get("window_size", 128))
    config["sparse_attention_config"] = sparse
    kv = dict(config.get("kv_cache_config") or {})
    kv.update(
        dtype=settings["kv_cache_dtype"],
        tokens_per_block=page_size,
        host_cache_size=settings["host_cache_mib"] * 1024**2,
        free_gpu_memory_fraction=settings["kv_cache_fraction"],
        enable_block_reuse=False,
        use_kv_cache_manager_v2=True,
    )
    config.update(
        model=str(model),
        backend="pytorch",
        skip_tokenizer_init=True,
        speculative_config=None,
        enable_chunked_prefill=False,
        max_batch_size=settings["batch_size"],
        max_seq_len=settings["input_len"] + settings["output_len"] + 1,
        max_num_tokens=settings["input_len"] * settings["batch_size"],
        kv_cache_config=kv,
        cuda_graph_config={
            "batch_sizes": sorted({1, settings["batch_size"]}),
            "enable_padding": True,
        },
    )
    return config


def _run(run_dir: Path) -> None:
    import nvtx
    import torch

    import tensorrt_llm
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.bindings.internal import batch_manager

    manifest = json.loads((run_dir / "run.json").read_text())
    settings, config = manifest["settings"], manifest["llm_args"]
    if not hasattr(batch_manager, "kv_cache_manager_v2"):
        raise RuntimeError("Rebuild the native library and bindings with KV cache manager v2")
    from tensorrt_llm.runtime.kv_cache_manager_v2 import KVCacheManagerConfig

    if not hasattr(KVCacheManagerConfig, "sparse_offload_copy_only"):
        raise RuntimeError("Use the TensorRT-LLM build containing the copy-only diagnostic changes")
    (run_dir / "runtime.json").write_text(
        json.dumps(
            {
                "tensorrt_llm_version": tensorrt_llm.__version__,
                "tensorrt_llm_path": tensorrt_llm.__file__,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            },
            indent=2,
        )
        + "\n"
    )
    checkpoint = json.loads((Path(config["model"]) / "config.json").read_text())
    vocabulary = checkpoint["vocab_size"] - 2
    end_id = checkpoint.get("eos_token_id")
    if isinstance(end_id, list):
        end_id = end_id[0]
    if end_id is None:
        end_id = 1
    pad_id = checkpoint.get("pad_token_id")
    if pad_id is None:
        pad_id = end_id

    def generate(llm: LLM, wave: int, output_len: int) -> list[int]:
        prompts = [
            {
                "prompt_token_ids": [
                    2 + (wave * 997 + request * 257 + token) % vocabulary
                    for token in range(settings["input_len"])
                ]
            }
            for request in range(settings["batch_size"])
        ]
        outputs = llm.generate(
            prompts,
            SamplingParams(
                max_tokens=output_len,
                ignore_eos=True,
                temperature=0,
                detokenize=False,
                add_special_tokens=False,
                end_id=end_id,
                pad_id=pad_id,
            ),
            use_tqdm=False,
        )
        lengths = [len(output.outputs[0].token_ids) for output in outputs]
        if lengths != [output_len] * settings["batch_size"]:
            raise RuntimeError(f"Generation did not complete the requested workload: {lengths}")
        return lengths

    with LLM(**config) as llm:
        for wave in range(settings["warmup_runs"]):
            with nvtx.annotate("dsv4_profile_warmup", domain="TensorRT-LLM"):
                generate(llm, wave, min(settings["output_len"], 16))
        ranged = settings["capture"] == "range"
        if ranged:
            llm.start_profile(
                output_dir=str(run_dir),
                activities=["CUDA_PROFILER"],
                start_step=0,
                num_steps=settings.get("profile_steps"),
            )
        started = perf_counter()
        try:
            with nvtx.annotate("dsv4_profile_measured_generate", domain="TensorRT-LLM"):
                lengths = generate(llm, settings["warmup_runs"], settings["output_len"])
        finally:
            if ranged:
                llm.stop_profile()
        result = {"generated_tokens": lengths, "elapsed_seconds": perf_counter() - started}
    (run_dir / "workload-result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    _run(parser.parse_args().run_dir)

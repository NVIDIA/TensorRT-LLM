# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build kernels the warmup needs while the weights load.

``DecoderWarmup`` starts with steps that only build kernels: flashinfer's
sampling module and, for MiniMax-M3 with MSA sparse attention, the indexer's
KV-split proxy variants. They depend on the configuration alone, so the model
loader starts them on a background thread as soon as it has read the
configuration. Once it has allocated the model's weights, it also starts
building the GEMM kernels of the model's Linear layers
(``modules/linear_prebuild.py``), which the warmup's forwards would build
otherwise. The warmup waits for these threads before its first step, which then
finds the kernels built. ``TLLM_OVERLAP_PREWARM=0`` leaves all of it to the
warmup, ``TLLM_OVERLAP_PREWARM_LINEAR=0`` the Linear GEMMs.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import Any, Callable, List, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

_ENV = "TLLM_OVERLAP_PREWARM"
_LINEAR_ENV = "TLLM_OVERLAP_PREWARM_LINEAR"
# nvcc runs with the GIL released, so DeepGEMM compiles in parallel threads.
_DEEP_GEMM_THREADS_ENV = "TLLM_OVERLAP_PREWARM_DEEP_GEMM_THREADS"
# Free device memory left to the loader next to the dummy operands.
_LOADER_RESERVE_BYTES = 8 << 30
_lock = threading.Lock()
_thread: Optional[threading.Thread] = None
_linear_groups: List["_Group"] = []
_linear_threads: List[threading.Thread] = []
_released = False


def overlap_enabled() -> bool:
    return os.environ.get(_ENV, "1") != "0"


def _msa_indexer_params(
    model_config: Any, llm_args: Any, mapping: Any
) -> Optional[Tuple[torch.dtype, int, int]]:
    """The arguments ``DecoderWarmup`` passes to ``prewarm_split_proxy_variants``.

    Derived from the configuration the KV cache manager is later built from;
    None when that warmup step does not run (no MSA, or no KV cache manager V2).
    """
    from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig

    sparse = getattr(llm_args, "sparse_attention_config", None)
    if not isinstance(sparse, MiniMaxM3SparseAttentionConfig) or sparse.implementation != "msa":
        return None
    kv_cache_config = getattr(llm_args, "kv_cache_config", None)
    if not getattr(kv_cache_config, "use_kv_cache_manager_v2", False):
        return None
    try:
        params = sparse.to_sparse_metadata_params(pretrained_config=model_config.pretrained_config)
        num_index_heads = params.sharded_index_head_count(mapping)
    except Exception as e:  # noqa: BLE001 - the warmup runs the step instead
        logger.warning(
            f"Overlapped MSA indexer prewarm skipped; the warmup runs it. {type(e).__name__}: {e}"
        )
        return None
    dtype = (
        torch.float8_e4m3fn
        if str(getattr(sparse, "indexer_kv_dtype", "bf16")) == "fp8"
        else torch.bfloat16
    )
    return dtype, num_index_heads, kv_cache_config.tokens_per_block


def _run(device: int, msa_args: Optional[Tuple[torch.dtype, int, int]]) -> None:
    # The current CUDA device is per thread.
    torch.cuda.set_device(device)
    from tensorrt_llm._torch.pyexecutor.sampler.ops.flashinfer import warmup_sampling_module

    t0 = time.perf_counter()
    warmup_sampling_module()  # best effort, warns on failure
    t1 = time.perf_counter()
    done = f"sampling_module {t1 - t0:.1f}s"
    if msa_args is not None:
        try:
            from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_indexer import (
                prewarm_split_proxy_variants,
            )

            # A side stream keeps the probe launches off the weight loader's stream.
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream), torch.inference_mode():
                prewarm_split_proxy_variants(*msa_args)
            stream.synchronize()
            done += f", msa_indexer {time.perf_counter() - t1:.1f}s"
        except Exception as e:  # noqa: BLE001 - the warmup runs the step instead
            logger.warning(
                "Overlapped MSA indexer prewarm failed; the warmup runs it. "
                f"{type(e).__name__}: {e}"
            )
    logger.info(f"[warmup] overlapped prewarm done: {done}")


def start(model_config: Any, llm_args: Any, mapping: Any) -> None:
    """Start the background prewarm once per process; a no-op when disabled."""
    global _thread
    if not overlap_enabled() or not torch.cuda.is_available():
        return
    with _lock:
        if _thread is not None:
            return
        msa_args = _msa_indexer_params(model_config, llm_args, mapping)
        _thread = threading.Thread(
            target=_run,
            args=(torch.cuda.current_device(), msa_args),
            name="tllm-overlapped-prewarm",
            daemon=True,
        )
        _thread.start()
        logger.info("[warmup] overlapped prewarm started next to weight loading")


class _Group:
    """Builds one kind of kernel from a queue of shapes on ``threads`` threads, each on
    its own stream, and logs its time once the last thread is done."""

    def __init__(self, name: str, items: List[Any], build: Callable[[Any], int], threads: int):
        self.name, self.build = name, build
        self.items: queue.Queue = queue.Queue()
        for item in items:
            self.items.put(item)
        self.count, self.launches, self.failed = len(items), 0, 0
        self.running = min(threads, len(items))
        self.mutex = threading.Lock()
        self.t0 = time.perf_counter()

    def run(self, device: int) -> None:
        torch.cuda.set_device(device)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), torch.inference_mode():
            while True:
                try:
                    item = self.items.get_nowait()
                except queue.Empty:
                    break
                try:
                    launches = self.build(item)
                    with self.mutex:
                        self.launches += launches
                except Exception as e:  # noqa: BLE001 - the warmup builds what is missing
                    with self.mutex:
                        self.failed += 1
                    logger.warning(
                        f"Linear prebuild of {self.name} {item} failed; the warmup builds it. "
                        f"{type(e).__name__}: {e}"
                    )
        stream.synchronize()
        with self.mutex:
            self.running -= 1
            last = self.running == 0
        if last:
            logger.info(
                f"[warmup] linear prebuild done: {self.name} {time.perf_counter() - self.t0:.1f}s "
                f"({self.count} shapes, {self.launches} builds, {self.failed} failed)"
            )


def _deep_gemm_threads(plan: Any) -> int:
    """Threads for the DeepGEMM builds: the requested count, as far as their dummy
    operands fit next to the loader in free device memory (0: none fit)."""
    from tensorrt_llm._torch.modules import linear_prebuild as lp

    threads = max(1, int(os.environ.get(_DEEP_GEMM_THREADS_ENV, "4")))
    largest = max((lp.deep_gemm_footprint(item) for item in plan.fp8), default=0)
    if largest == 0:
        return threads
    free_bytes, _ = torch.cuda.mem_get_info()
    return max(0, min(threads, (free_bytes - _LOADER_RESERVE_BYTES) // largest))


def start_linear_prebuild(model: torch.nn.Module, llm_args: Any, mapping: Any) -> None:
    """Start building the GEMM kernels of ``model``'s Linear layers, once per process.

    Called with the model's weights allocated and not yet loaded; a no-op when
    disabled.
    """
    if (
        not overlap_enabled()
        or os.environ.get(_LINEAR_ENV, "1") == "0"
        or not torch.cuda.is_available()
    ):
        return
    from tensorrt_llm._torch.modules import linear_prebuild as lp

    with _lock:
        if _linear_threads:
            return
        try:
            plan = lp.collect(model, llm_args, mapping)
        except Exception as e:  # noqa: BLE001 - the warmup builds them instead
            logger.warning(f"Linear prebuild skipped. {type(e).__name__}: {e}")
            return
        if plan.empty():
            return
        device = torch.cuda.current_device()
        dev = torch.device("cuda", device)
        deep_gemm_threads = _deep_gemm_threads(plan)
        if plan.fp8 and deep_gemm_threads == 0:
            logger.info("[warmup] linear prebuild: no free memory for DeepGEMM's operands")
        groups = [
            _Group(
                "deep_gemm",
                plan.fp8 if deep_gemm_threads else [],
                lambda item: lp.build_deep_gemm(item, dev),
                deep_gemm_threads,
            ),
            _Group(
                "flashinfer_mxfp8",
                ["module"] if plan.flashinfer_mxfp8 else [],
                lambda _: lp.build_flashinfer_mxfp8() or 1,
                1,
            ),
        ]
        _linear_groups.extend(groups)
        for group in groups:
            for i in range(group.running):
                thread = threading.Thread(
                    target=group.run,
                    args=(device,),
                    name=f"tllm-linear-prebuild-{group.name}-{i}",
                    daemon=True,
                )
                thread.start()
                _linear_threads.append(thread)
        logger.info(f"[warmup] linear prebuild started next to weight loading: {plan.summary()}")


def wait() -> None:
    """Block until the background builds, if any, have finished."""
    global _released
    threads = [t for t in (_thread, *_linear_threads) if t is not None and t.is_alive()]
    t0 = time.perf_counter()
    for thread in threads:
        thread.join()
    if _linear_threads and not _released:
        # Return the dummy operands' memory before the warmup measures any.
        _released = True
        torch.cuda.empty_cache()
    if threads:
        logger.info(f"[warmup] overlapped prewarm: waited {time.perf_counter() - t0:.1f}s")

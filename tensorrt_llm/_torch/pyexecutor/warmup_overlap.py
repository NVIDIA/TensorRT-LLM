# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build, while the weights load, the kernels the decoder warmup builds without a model forward.

``warmup_overlap`` wraps the model loader's weight loading. Next to it, on background
threads, it runs the steps ``DecoderWarmup`` starts with that only build kernels
(flashinfer's sampling modules and, for MiniMax-M3 with MSA sparse attention, the
indexer's KV-split proxy variants) and builds the GEMM kernels of the model's Linear
layers (``modules/linear_prebuild.py``), which the warmup's forwards would build
otherwise. It returns once the threads are done; the warmup then runs as before and finds
the kernels built. ``TLLM_OVERLAP_PREWARM=0`` leaves all of it to the warmup,
``TLLM_OVERLAP_PREWARM_LINEAR=0`` the Linear GEMMs.
"""

from __future__ import annotations

import contextlib
import os
import queue
import threading
import time
from typing import Any, Callable, Iterator, List, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

_ENV = "TLLM_OVERLAP_PREWARM"
_LINEAR_ENV = "TLLM_OVERLAP_PREWARM_LINEAR"
# nvcc runs with the GIL released, so DeepGEMM compiles in parallel threads.
_DEEP_GEMM_THREADS_ENV = "TLLM_OVERLAP_PREWARM_DEEP_GEMM_THREADS"
# Free device memory left to the loader next to the dummy operands.
_LOADER_RESERVE_BYTES = 8 << 30


def overlap_enabled() -> bool:
    return os.environ.get(_ENV, "1") != "0"


class _Group:
    """Builds one kind of kernel from a queue of items on ``threads`` threads, each on its
    own stream, and logs its time once the last thread is done."""

    def __init__(self, name: str, items: List[Any], build: Callable[[Any], int], threads: int):
        self.name, self.build = name, build
        self.items: queue.Queue = queue.Queue()
        for item in items:
            self.items.put(item)
        self.count, self.launches, self.failed = len(items), 0, 0
        self.running = min(threads, len(items))
        self.mutex = threading.Lock()
        self.stop = threading.Event()
        self.threads: List[threading.Thread] = []
        self.t0 = time.perf_counter()

    def start(self, device: int) -> None:
        for i in range(self.running):
            thread = threading.Thread(
                target=self._run, args=(device,), name=f"tllm-overlap-{self.name}-{i}", daemon=True
            )
            thread.start()
            self.threads.append(thread)

    def join(self) -> None:
        for thread in self.threads:
            thread.join()

    def _run(self, device: int) -> None:
        torch.cuda.set_device(device)
        # A side stream keeps the launches off the weight loader's stream.
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), torch.inference_mode():
            while not self.stop.is_set():
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
                        f"Overlapped {self.name} build of {item} failed; the warmup builds it. "
                        f"{type(e).__name__}: {e}"
                    )
        stream.synchronize()
        with self.mutex:
            self.running -= 1
            last = self.running == 0
        if last:
            logger.info(
                f"[warmup] overlap done: {self.name} {time.perf_counter() - self.t0:.1f}s "
                f"({self.count} items, {self.launches} builds, {self.failed} failed)"
            )


def _msa_indexer_params(
    model_config: Any, llm_args: Any, mapping: Any
) -> Optional[Tuple[torch.dtype, int, int]]:
    """The arguments ``DecoderWarmup`` passes to ``prewarm_split_proxy_variants``.

    Derived from the configuration the KV cache manager is built from; None when that
    warmup step does not run (no MSA, or no KV cache manager V2).
    """
    from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig

    sparse = getattr(llm_args, "sparse_attention_config", None)
    if not isinstance(sparse, MiniMaxM3SparseAttentionConfig) or sparse.implementation != "msa":
        return None
    kv_cache_config = getattr(llm_args, "kv_cache_config", None)
    if not getattr(kv_cache_config, "use_kv_cache_manager_v2", False):
        return None
    params = sparse.to_sparse_metadata_params(pretrained_config=model_config.pretrained_config)
    dtype = (
        torch.float8_e4m3fn
        if str(getattr(sparse, "indexer_kv_dtype", "bf16")) == "fp8"
        else torch.bfloat16
    )
    return dtype, params.sharded_index_head_count(mapping), kv_cache_config.tokens_per_block


def _warmup_steps(model_config: Any, llm_args: Any, mapping: Any) -> List[Tuple[str, Callable]]:
    """The steps of ``DecoderWarmup``'s ``sampling_module_prewarm`` that need no KV cache."""
    from tensorrt_llm._torch.pyexecutor.sampler.ops import flashinfer as sampler_ops

    steps = [("sampling_module", sampler_ops.warmup_sampling_module)]
    if getattr(llm_args, "max_beam_width", 1) > 1:
        steps.append(("topk_module", sampler_ops.warmup_topk_module))
    msa_args = _msa_indexer_params(model_config, llm_args, mapping)
    if msa_args is not None:
        from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_indexer import (
            prewarm_split_proxy_variants,
        )

        steps.append(("msa_indexer", lambda: prewarm_split_proxy_variants(*msa_args)))
    return steps


def _run_step(step: Tuple[str, Callable]) -> int:
    name, call = step
    t0 = time.perf_counter()
    call()
    logger.info(f"[warmup] overlapped {name}: {time.perf_counter() - t0:.1f}s")
    return 1


def _deep_gemm_threads(plan: Any) -> int:
    """Threads for the DeepGEMM builds: the requested count, as far as their dummy operands
    fit next to the loader in free device memory (0: none fit)."""
    from tensorrt_llm._torch.modules import linear_prebuild as lp

    threads = max(1, int(os.environ.get(_DEEP_GEMM_THREADS_ENV, "4")))
    largest = max((lp.deep_gemm_footprint(item) for item in plan.fp8), default=0)
    if largest == 0:
        return threads
    free_bytes, _ = torch.cuda.mem_get_info()
    return max(0, min(threads, (free_bytes - _LOADER_RESERVE_BYTES) // largest))


def _linear_groups(model: torch.nn.Module, llm_args: Any, mapping: Any) -> List[_Group]:
    """The builds of the GEMM kernels of ``model``'s Linear layers."""
    from tensorrt_llm._torch.modules import linear_prebuild as lp

    plan = lp.collect(model, llm_args, mapping)
    device = torch.device("cuda", torch.cuda.current_device())
    deep_gemm_threads = _deep_gemm_threads(plan)
    if plan.fp8 and deep_gemm_threads == 0:
        logger.info("[warmup] overlap: no free memory for DeepGEMM's operands")
    logger.info(f"[warmup] overlap plan: {plan.summary()}")
    return [
        _Group(
            "deep_gemm",
            plan.fp8 if deep_gemm_threads else [],
            lambda item: lp.build_deep_gemm(item, device),
            deep_gemm_threads,
        ),
        _Group(
            "flashinfer_mxfp8",
            ["module"] if plan.flashinfer_mxfp8 else [],
            lambda _: lp.build_flashinfer_mxfp8() or 1,
            1,
        ),
    ]


@contextlib.contextmanager
def warmup_overlap(
    model: torch.nn.Module, model_config: Any, llm_args: Any, mapping: Any
) -> Iterator[List[_Group]]:
    """While the body of the ``with`` loads ``model``'s allocated weights, build on
    background threads the kernels the decoder warmup builds without a model forward;
    leave the ``with`` once they are built. Yields the build groups."""
    groups: List[_Group] = []
    if overlap_enabled() and torch.cuda.is_available():
        try:
            steps = _warmup_steps(model_config, llm_args, mapping)
        except Exception as e:  # noqa: BLE001 - the warmup runs them instead
            logger.warning(f"Overlapped warmup steps skipped. {type(e).__name__}: {e}")
            steps = []
        # Each step builds in subprocesses (ninja), so the steps run in parallel.
        groups.append(_Group("warmup_steps", steps, _run_step, len(steps)))
        if os.environ.get(_LINEAR_ENV, "1") != "0":
            try:
                groups += _linear_groups(model, llm_args, mapping)
            except Exception as e:  # noqa: BLE001 - the warmup builds them instead
                logger.warning(f"Overlapped Linear GEMM builds skipped. {type(e).__name__}: {e}")
        device = torch.cuda.current_device()
        for group in groups:
            group.start(device)
    try:
        yield groups
    except BaseException:
        # Leave the rest to the warmup rather than hold up the error.
        for group in groups:
            group.stop.set()
        raise
    t0 = time.perf_counter()
    for group in groups:
        group.join()
    if groups:
        # Return the dummy operands' memory before anything measures free memory.
        torch.cuda.empty_cache()
        logger.info(f"[warmup] overlap: waited {time.perf_counter() - t0:.1f}s for the builds")

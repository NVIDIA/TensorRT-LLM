# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent connector instrumentation imported by spawned integration workers."""

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import torch
from llm_kv_cache_connector import PersistentKvCacheConnectorMetadata
from llm_kv_cache_connector_adp import (
    PersistentKvCacheConnectorLeader,
    PersistentKvCacheConnectorWorker,
)

from tensorrt_llm import mpi_rank
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import (
    KvCacheConnectorScheduler,
    KvCacheConnectorWorker,
    SchedulerOutput,
)
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_layout import KvCacheLayout
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs


def _record(event: str, count: int) -> None:
    folder = Path(os.environ["TRTLLM_TEST_CONNECTOR_RECORDS"])
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"rank-{mpi_rank()}.jsonl").open("a") as stream:
        stream.write(json.dumps({"event": event, "count": count}) + "\n")


class RecordingConnectorWorker(PersistentKvCacheConnectorWorker):
    """Record completed copies, and optionally delay async completion reporting."""

    def __init__(self, llm_args: TorchLlmArgs) -> None:
        super().__init__(llm_args)
        self._pending_load_ids: set[int] = set()
        self._pending_prefix_load_ids: list[int] = []

    def register_kv_cache_layout(self, layout: KvCacheLayout) -> None:
        super().register_kv_cache_layout(layout)
        _record("v2_layout", 1)

    def start_load_kv(self, stream: torch.cuda.Stream) -> None:
        super().start_load_kv(stream)
        # Record only after the example's blocking copies have completed.
        if self._metadata.load:
            for path, block_id in self._metadata.load:
                torch.testing.assert_close(
                    self.kv_cache_tensor[block_id].cpu(),
                    torch.load(path, map_location="cpu", weights_only=True),
                    rtol=0,
                    atol=0,
                )
            _record("loaded_blocks", len(self._metadata.load))

    def get_finished_prefix_loads(self) -> list[int]:
        finished = super().get_finished_prefix_loads()
        if os.environ.get("TRTLLM_TEST_CONNECTOR_STALL") == "1":
            if finished:
                _record("withheld_prefix_load", len(finished))
            return []
        if os.environ.get("TRTLLM_TEST_CONNECTOR_ASYNC") == "1":
            ready = self._pending_prefix_load_ids
            self._pending_prefix_load_ids = finished
            if ready:
                _record("async_finished", len(ready))
            return ready
        return finished

    def get_finished(
        self, finished_gen_req_ids: list[int], started_loading_req_ids: list[int]
    ) -> tuple[list[int], list[int]]:
        # Complete on the next poll, so the initial async batch is emptied
        # even though the filesystem copies themselves run synchronously.
        finished_loading = list(self._pending_load_ids)
        self._pending_load_ids = set(started_loading_req_ids)
        if finished_loading:
            _record("async_finished", len(finished_loading))
        return [], finished_loading


class RecordingConnectorScheduler(PersistentKvCacheConnectorLeader):
    """Optionally force an async hit on owner zero and a cache miss on its peer."""

    def __init__(self, llm_args: TorchLlmArgs) -> None:
        # MPI forwards TRTLLM-prefixed overrides; the filesystem example reads
        # CONNECTOR_CACHE_FOLDER when its scheduler is constructed.
        os.environ["CONNECTOR_CACHE_FOLDER"] = os.environ["TRTLLM_TEST_CONNECTOR_CACHE_FOLDER"]
        super().__init__(llm_args)
        self._async_ids: set[int] = set()
        self._async_loads: list[tuple[Path, int]] = []

    def reserve_prefix(
        self, request: LlmRequest, num_computed_tokens: int, reservation_id: int
    ) -> tuple[int, bool]:
        asynchronous = os.environ.get("TRTLLM_TEST_CONNECTOR_ASYNC") == "1"
        if asynchronous and mpi_rank() != 0:
            return 0, False
        matched, _ = super().reserve_prefix(request, num_computed_tokens, reservation_id)
        return matched, bool(matched and asynchronous)

    def get_num_new_matched_tokens(
        self, request: LlmRequest, num_computed_tokens: int
    ) -> tuple[int, bool]:
        asynchronous = os.environ.get("TRTLLM_TEST_CONNECTOR_ASYNC") == "1"
        if asynchronous and mpi_rank() != 0:
            self.pending_loads[request.request_id] = []
            return 0, False
        matched, _ = super().get_num_new_matched_tokens(request, num_computed_tokens)
        if asynchronous and matched:
            self._async_ids.add(request.request_id)
            return matched, True
        return matched, False

    def update_state_after_alloc(self, request: LlmRequest, block_ids: list[int]) -> None:
        _record("allocated_requests", 1)
        if request.request_id not in self._async_ids:
            return
        self._async_ids.remove(request.request_id)
        paths = self.pending_loads.pop(request.request_id)
        first_block = request.context_current_position // self.block_size - len(paths)
        self._async_loads.extend(zip(paths, block_ids[first_block:]))

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> PersistentKvCacheConnectorMetadata:
        metadata = super().build_connector_meta(scheduler_output)
        # Async requests are absent from scheduler_output until they can
        # compute; allocation feedback supplies their destination block IDs.
        metadata.load.extend(self._async_loads)
        self._async_loads = []
        return metadata


def _run_timeout_consumer() -> None:
    """One disposable process owns the executor that is expected to fail."""
    import sys
    import traceback

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig, KvCacheConnectorConfig

    options = json.loads(sys.argv[1])
    options["kv_cache_config"] = KvCacheConfig(**options["kv_cache_config"])
    options["kv_connector_config"] = KvCacheConnectorConfig(**options["kv_connector_config"])
    prompt = sys.argv[2]
    try:
        # A fatal transfer error requires immediate process exit. A context
        # manager would try graceful shutdown before reaching the handler.
        print("ADP timeout consumer: initializing executor", flush=True)
        llm = LLM(**options)
        print("ADP timeout consumer: starting stalled generation", flush=True)
        llm.generate([prompt] * 2, SamplingParams(max_tokens=8, ignore_eos=True))
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        # Fatal connector failures require process exit; never construct a
        # replacement executor or reuse a potentially damaged worker here.
        os._exit(86)
    llm.shutdown()
    raise AssertionError("The stalled connector unexpectedly completed")


if __name__ == "__main__":
    _run_timeout_consumer()


@dataclass
class FailedLoadMetadata:
    """Allocation owners and save intents for injected prefix-load failures."""

    loads: dict[int, list[int]] = field(default_factory=dict)
    saves: set[int] = field(default_factory=set)


class FailedLoadConnectorScheduler(KvCacheConnectorScheduler):
    """Offer one block, then honor the runtime's lifetime failure marker."""

    supports_attention_dp = True

    def __init__(self, llm_args: TorchLlmArgs) -> None:
        super().__init__(llm_args)
        self._block_size = llm_args.kv_cache_config.tokens_per_block
        self._offered: set[int] = set()
        self._loads: dict[int, list[int]] = {}

    def get_num_new_matched_tokens(
        self, request: LlmRequest, num_computed_tokens: int
    ) -> tuple[int, bool]:
        if request.py_connector_load_failed:
            _record("declined_after_failure", 1)
            return 0, False
        if os.environ.get("TRTLLM_TEST_CONNECTOR_FAIL") != "1" or mpi_rank() != 0:
            return 0, False
        assert num_computed_tokens == 0
        assert request.prompt_len > self._block_size
        self._offered.add(request.request_id)
        return self._block_size, os.environ.get("TRTLLM_TEST_CONNECTOR_ASYNC") == "1"

    def update_state_after_alloc(self, request: LlmRequest, block_ids: list[int]) -> None:
        if request.request_id in self._offered:
            self._offered.remove(request.request_id)
            self._loads[request.request_id] = block_ids[:1]

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> FailedLoadMetadata:
        # Async requests are parked outside the compute batch, so retain their
        # allocation feedback as well as this pass's synchronous save intents.
        loads, self._loads = self._loads, {}
        saves = {req.request_id for req in scheduler_output.new_requests}
        saves.update(loads)
        return FailedLoadMetadata(loads, saves)

    def request_finished(self, request: LlmRequest, cache_block_ids: list[int]) -> bool:
        return False

    def request_reset(self, request: LlmRequest) -> None:
        self._offered.discard(request.request_id)
        self._loads.pop(request.request_id, None)
        _record("reset", 1)


class FailedLoadConnectorWorker(KvCacheConnectorWorker):
    """Poison a prefix page and report failure only after the write is drained.

    This injects a failed transfer into a real model/allocator execution. It
    does not emulate network transport or claim RDMA completion coverage.
    """

    supports_attention_dp = True
    reports_failed_loads = True
    requires_block_aligned_reuse = True

    def __init__(self, llm_args: TorchLlmArgs) -> None:
        super().__init__(llm_args)
        self._failed: set[int] = set()
        self._async_failed: set[int] = set()
        self._pending: set[int] = set()
        self._invalid_saves: set[int] = set()

    def register_kv_caches(self, kv_cache_tensor: torch.Tensor) -> None:
        self._kv = kv_cache_tensor

    def start_load_kv(self, stream: torch.cuda.Stream) -> None:
        metadata = self.get_connector_meta()
        assert isinstance(metadata, FailedLoadMetadata)
        with torch.cuda.stream(stream):
            for pages in metadata.loads.values():
                for page in pages:
                    self._kv[page].fill_(float("nan"))
        stream.synchronize()
        failed = set(metadata.loads)
        self._invalid_saves.update(failed)
        if os.environ.get("TRTLLM_TEST_CONNECTOR_ASYNC") == "1":
            self._async_failed.update(failed)
        else:
            self._failed.update(failed)
        if failed:
            _record("poisoned_load", len(failed))

    def take_failed_load_requests(self) -> set[int]:
        failed, self._failed = self._failed, set()
        return failed

    def get_finished(
        self, finished_gen_req_ids: list[int], started_loading_req_ids: list[int]
    ) -> tuple[list[int], list[int]]:
        finished = list(self._pending)
        self._pending = set(started_loading_req_ids)
        self._finished = set(finished)
        return [], finished

    def take_failed_async_load_requests(self) -> set[int]:
        failed = self._async_failed.intersection(self._finished)
        self._async_failed.difference_update(failed)
        if failed:
            _record("async_failed", len(failed))
        return failed

    def drop_bound_saves(self, request_ids: Iterable[int]) -> None:
        metadata = self.get_connector_meta()
        assert isinstance(metadata, FailedLoadMetadata)
        ids = set(request_ids)
        metadata.saves.difference_update(ids)
        self._invalid_saves.difference_update(ids)
        _record("dropped_saves", len(ids))

    def wait_for_layer_load(self, layer_idx: int, stream: torch.cuda.Stream) -> None:
        pass

    def save_kv_layer(self, layer_idx: int, stream: torch.cuda.Stream) -> None:
        pass

    def wait_for_save(self, stream: torch.cuda.Stream) -> None:
        metadata = self.get_connector_meta()
        assert isinstance(metadata, FailedLoadMetadata)
        # Async requests are not computed yet; their save intents are not
        # publishable until their completion/recovery poll has run.
        assert not (metadata.saves & self._invalid_saves - self._async_failed)

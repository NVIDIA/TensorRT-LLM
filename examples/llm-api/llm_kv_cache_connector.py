# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

### :title KV Cache Connector
### :order 6
### :section Customization
'''
This script demonstrates the KV cache connector feature in TensorRT-LLM, which enables
custom persistence and reuse of KV cache blocks across different LLM instances.

**Scenario:**
The script implements a persistent KV cache connector that saves computed KV cache blocks
to disk and loads them back in subsequent runs, eliminating redundant computation for
recurring prompts.

**What is a KV Cache Connector?**

A KV cache connector is a customizable interface that allows you to:
1.  **Save KV Cache:** Persist computed KV cache blocks to an external storage
    (disk, database, distributed cache, etc.)
2.  **Load KV Cache:** Retrieve previously computed cache blocks instead of recomputing them
3.  **Share Cache Across Instances:** Reuse cache blocks across different LLM instances
    or sessions, unlike regular block reuse which is limited to a single instance

**How It Works:**

This example implements a `PersistentKvCacheConnector` with two key components:

* **PersistentKvCacheConnectorLeader (Scheduler):**
    - Hashes token sequences to create unique identifiers for each cache block
    - Checks if cached blocks exist on disk for incoming requests
    - Schedules load operations for cache hits
    - Schedules save operations for newly computed blocks

* **PersistentKvCacheConnectorWorker:**
    - Executes the actual load/save operations between GPU and disk
    - Loads cached blocks from disk files into GPU memory
    - Saves newly computed blocks from GPU to disk files

**Demonstration:**

The script processes the same prompt twice using two separate LLM instances:

1.  **First Run (Instance 1):**
    - The LLM computes the KV cache for the input prompt
    - The connector saves the computed cache blocks to disk (as .pt files)
    - The generation completes and the LLM instance is destroyed

2.  **Second Run (Instance 2):**
    - A new LLM instance is created with the same connector configuration
    - When processing the same prompt, the connector finds matching cache blocks on disk
    - The cache is loaded from disk instead of being recomputed
    - **Expected Outcome:** Faster prefill as cache blocks are loaded rather than computed
    - Both outputs should be identical, demonstrating deterministic cache reuse

**Key Benefits:**

- **Cross-Instance Cache Sharing:** Share computed caches across multiple LLM instances
- **Persistent Storage:** Cache survives beyond the lifetime of a single LLM instance
- **Custom Storage Backends:** Implement any storage mechanism (shown here: disk files)
- **Reduced Computation:** Eliminate redundant KV cache computation for repeated prompts

**How to Run:**

```bash
python llm_kv_cache_connector.py <model_path>
```

Example:
```bash
python llm_kv_cache_connector.py meta-llama/Llama-3.1-8B-Instruct
```

**Implementation Notes:**

- This example uses content-based hashing to identify cache blocks
- Cache files are stored in a temporary directory (cleaned up after the demo)
- The implementation is simplified and not optimized for production use
- Does not support chunked prefill in this example
- See `tensorrt_llm/_torch/pyexecutor/connectors/kv_cache_connector.py` for the full connector interface

**NOTE:** This example connector implementation is designed for demonstration purposes
and is NOT suitable for production use without additional optimizations and error handling.
'''

import hashlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Optional

import click
import torch

from tensorrt_llm import LLM, SamplingParams, logger
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import (
    KvCacheConnectorScheduler, KvCacheConnectorWorker, SchedulerOutput)
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_layout import \
    valid_page_slots
from tensorrt_llm.bindings.internal.batch_manager import LlmRequest
from tensorrt_llm.llmapi.llm_args import KvCacheConnectorConfig, TorchLlmArgs

CONNECTOR_CACHE_FOLDER_KEY = "CONNECTOR_CACHE_FOLDER"


@dataclass
class PersistentKvCacheConnectorMetadata:
    load: list[tuple[str, int]] = field(default_factory=list)
    save: list[tuple[str, int]] = field(default_factory=list)
    prefix_load_ids: list[int] = field(default_factory=list)


class PersistentKvCacheConnectorWorker(KvCacheConnectorWorker):

    def __init__(self, llm_args: TorchLlmArgs):
        super().__init__(llm_args)

        self.kv_cache_tensor = None
        self._finished_prefix_loads: list[int] = []

    def register_kv_caches(self, kv_cache_tensor: torch.Tensor):
        # This is the only registration hook this connector needs. A cache that
        # describes itself as a layout instead still arrives here, through
        # `register_kv_cache_layout`'s default, as long as one tensor can
        # describe it. See llm_kv_cache_connector_vswa.py for the case where it
        # cannot -- one attention window size per layer group.
        assert self.kv_cache_tensor is None, "KV cache tensor already registered"
        self.kv_cache_tensor = kv_cache_tensor

    def start_load_kv(self, stream: torch.cuda.Stream):
        # Do all loads synchronously, and blockwise.
        for path, block_id in self._metadata.load:
            cpu_tensor = torch.load(path, map_location="cpu")

            # Copy into the device block.
            self.kv_cache_tensor[block_id].copy_(cpu_tensor, non_blocking=False)

        self._finished_prefix_loads.extend(self._metadata.prefix_load_ids)

    def get_finished_prefix_loads(self) -> list[int]:
        finished, self._finished_prefix_loads = self._finished_prefix_loads, []
        return finished

    def wait_for_layer_load(self, layer_idx: int, stream: torch.cuda.Stream):
        pass

    def save_kv_layer(self, layer_idx: int, stream: torch.cuda.Stream):
        pass

    def wait_for_save(self, stream: torch.cuda.Stream):

        # Make sure the forward pass is complete before beginning our save.
        stream.synchronize()

        for path, block_id in self._metadata.save:
            cpu_tensor = self.kv_cache_tensor[block_id].cpu()

            # Don't write anything if this specific block already exists.
            if Path(path).exists():
                continue

            # Publish complete immutable files so a reservation can retain an
            # inode while another process replaces or removes its cache key.
            with NamedTemporaryFile(dir=Path(path).parent) as staging:
                torch.save(cpu_tensor, staging.name)
                try:
                    os.link(staging.name, path)
                except FileExistsError:
                    pass

    def get_finished(
            self, finished_gen_req_ids: list[int],
            started_loading_req_ids: list[int]) -> tuple[list[int], list[int]]:

        return [], []


class PersistentKvCacheConnectorLeader(KvCacheConnectorScheduler):

    def __init__(self, llm_args: TorchLlmArgs):
        super().__init__(llm_args)

        self.block_size = self._llm_args.kv_cache_config.tokens_per_block
        self.pending_loads = {}

        self.cache_folder = os.environ.get(CONNECTOR_CACHE_FOLDER_KEY,
                                           "./connector_cache")

        os.makedirs(self.cache_folder, exist_ok=True)
        self._reservation_folder = TemporaryDirectory(prefix=".reservations-",
                                                      dir=self.cache_folder)
        self._reserved_files: dict[int, dict[int, Path]] = {}
        self._reserved_ranges: dict[int, list[tuple[int, int]]] = {}

    def build_connector_meta(self, scheduler_output: SchedulerOutput):
        # NOTE: This is a simplified implementation, and does not work with chunked prefill.

        metadata = PersistentKvCacheConnectorMetadata()

        accepted_ends = {}
        for load in scheduler_output.prefix_loads:
            if len(load.block_ids_by_layer_group) != 1:
                raise ValueError(
                    "Persistent connector requires one layer group")
            if load.start % self.block_size or load.end % self.block_size:
                raise ValueError("Persistent connector loads complete blocks")
            block_ids = load.block_ids_by_layer_group[0]
            if load.end // self.block_size > len(block_ids):
                raise ValueError(
                    "Confirmed prefix exceeds destination allocation")
            slots = dict(valid_page_slots(block_ids))
            files = self._reserved_files[load.reservation_id]
            for ordinal in range(load.start // self.block_size,
                                 load.end // self.block_size):
                if ordinal in slots:
                    metadata.load.append((str(files[ordinal]), slots[ordinal]))
            metadata.prefix_load_ids.append(load.reservation_id)
            accepted_ends[load.request_id] = load.end

        for req in scheduler_output.new_requests:
            num_computed_blocks = req.computed_position // self.block_size
            slots = dict(valid_page_slots(req.new_block_ids))
            pending_load = self.pending_loads.get(req.request_id, [])
            for ordinal, path in enumerate(pending_load, num_computed_blocks):
                if ordinal in slots:
                    metadata.load.append((str(path), slots[ordinal]))

            loaded_end = accepted_ends.get(
                req.request_id,
                (num_computed_blocks + len(pending_load)) * self.block_size)
            computed_end = max(req.computed_position,
                               loaded_end) + req.num_scheduled_tokens
            for ordinal, slot in slots.items():
                end = (ordinal + 1) * self.block_size
                if end <= loaded_end or end > min(len(req.new_tokens),
                                                  computed_end):
                    continue
                key = self._hash_tokens(req.new_tokens[:end], req.cache_salt)
                metadata.save.append((str(self._file_path(key)), slot))

        self.pending_loads = {}
        return metadata

    def _hash_tokens(self, tokens: list[int], cache_salt: Optional[str]) -> str:
        # KV depends on every preceding token, including those in other blocks.
        key = repr((cache_salt, tuple(tokens))).encode("utf-8")
        return hashlib.sha256(key).hexdigest()

    def _file_path(self, hash_value: str) -> Path:
        return Path(self.cache_folder) / f"{hash_value}.pt"

    def get_num_new_matched_tokens(
            self, request: LlmRequest,
            num_computed_tokens: int) -> tuple[int, bool]:
        self.pending_loads[request.request_id] = []

        # Don't bother with sequences with partial matches.
        if (num_computed_tokens % self.block_size) != 0:
            return 0, False

        tokens = request.get_tokens(0)
        for end in range(num_computed_tokens + self.block_size,
                         len(tokens) + 1, self.block_size):
            key = self._hash_tokens(tokens[:end], request.cache_salt)
            file_path = self._file_path(key)
            if not file_path.exists():
                break
            self.pending_loads[request.request_id].append(file_path)

        logger.info(
            f"KV CONNECTOR: Matched {len(self.pending_loads[request.request_id])} blocks for request {request.request_id}"
        )

        return len(
            self.pending_loads[request.request_id]) * self.block_size, False

    def reserve_prefix(self, request: LlmRequest, num_computed_tokens: int,
                       reservation_id: int) -> tuple[int, bool]:
        """Protect immutable source files without reading their KV data."""
        if num_computed_tokens % self.block_size:
            return 0, False
        tokens = request.get_tokens(0)
        folder = Path(self._reservation_folder.name) / str(reservation_id)
        folder.mkdir()
        files = {}
        for end in range(num_computed_tokens + self.block_size,
                         len(tokens) + 1, self.block_size):
            ordinal = end // self.block_size - 1
            key = self._hash_tokens(tokens[:end], request.cache_salt)
            protected = folder / f"{ordinal}.pt"
            try:
                os.link(self._file_path(key), protected)
            except FileNotFoundError:
                break
            files[ordinal] = protected
        count = len(files) * self.block_size
        if count:
            self._reserved_files[reservation_id] = files
            self._reserved_ranges[reservation_id] = [
                (num_computed_tokens, num_computed_tokens + count)
            ]
        else:
            folder.rmdir()
        return count, False

    def release_prefix_reservation(self, request: LlmRequest,
                                   reservation_id: int, start: int,
                                   end: int) -> None:
        """Release only the named range; overlapping reservations retain it."""
        remaining = []
        for left, right in self._reserved_ranges[reservation_id]:
            if right <= start or left >= end:
                remaining.append((left, right))
            else:
                if left < start:
                    remaining.append((left, start))
                if end < right:
                    remaining.append((end, right))
        files = self._reserved_files[reservation_id]
        for ordinal, path in list(files.items()):
            block_start = ordinal * self.block_size
            block_end = block_start + self.block_size
            if not any(left < block_end and right > block_start
                       for left, right in remaining):
                path.unlink()
                del files[ordinal]
        if remaining:
            self._reserved_ranges[reservation_id] = remaining
        else:
            del self._reserved_ranges[reservation_id]
            del self._reserved_files[reservation_id]
            (Path(self._reservation_folder.name) / str(reservation_id)).rmdir()

    def request_finished(self, request: LlmRequest,
                         cache_block_ids: list[int]) -> bool:
        # We don't do any asynchronous saving, so always return False
        return False

    def update_state_after_alloc(self, request: LlmRequest,
                                 block_ids: list[int]):
        pass


@click.command()
@click.argument("model", type=str)
def main(model: str):
    sys.path.append(os.path.join(
        os.path.dirname(__file__),
        "..",
    ))

    this_module = __file__[__file__.rfind("/") + 1:__file__.rfind(".py")]

    # --- KV Cache Connector Config ---
    kv_connector_config = KvCacheConnectorConfig(
        connector_module=this_module,
        connector_scheduler_class="PersistentKvCacheConnectorLeader",
        connector_worker_class="PersistentKvCacheConnectorWorker",
    )

    connector_cache_dir = TemporaryDirectory()
    os.environ[CONNECTOR_CACHE_FOLDER_KEY] = connector_cache_dir.name

    # Create LLM instance with KV Cache Connector
    llm = LLM(model=model,
              backend="pytorch",
              cuda_graph_config=None,
              kv_connector_config=kv_connector_config)

    test_text = (
        "Nvidia Corporation is an American technology company headquartered in Santa Clara, California."
        "Founded in 1993 by Jensen Huang, Chris Malachowsky, and Curtis Priem, it develops graphics processing units (GPUs), "
        "system on a chips (SoCs), and application programming interfaces (APIs) for data science, high-performance computing, "
        "and mobile and automotive applications. Tell me about the company.")

    sampling_params = SamplingParams(max_tokens=32)

    # Generate text with the first LLM instance and save the kv cache blocks by the connector.
    output = llm.generate([test_text], sampling_params)
    text0 = output[0].outputs[0].text

    print("First output: ", text0)
    print("Loading new LLM instance...")

    del llm

    # Create a new LLM instance with the same connector configuration
    llm = LLM(model=model,
              backend="pytorch",
              cuda_graph_config=None,
              kv_connector_config=kv_connector_config)

    # Generate text with the second LLM instance and it should reuse the kv cache blocks from the connector.
    output = llm.generate([test_text], sampling_params)
    text1 = output[0].outputs[0].text

    print("Second output (using connector cache): ", text1)

    # Verify that the two outputs are identical
    assert text0 == text1

    connector_cache_dir.cleanup()


if __name__ == "__main__":
    main()

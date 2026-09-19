# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""V2 ADP filesystem connector: stable prefix keys and atomic publication.

Set both connector classes below and enable_attention_dp=True. Every owner
must see the same CONNECTOR_CACHE_FOLDER. Use a separate folder per model,
KV representation and block size. This example requires one full-attention
layer group and does not support chunked prefill or attention TP sharding.
"""

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

import torch
from llm_kv_cache_connector import PersistentKvCacheConnectorLeader as BaseConnectorLeader
from llm_kv_cache_connector import PersistentKvCacheConnectorMetadata
from llm_kv_cache_connector import PersistentKvCacheConnectorWorker as BaseConnectorWorker

from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import SchedulerOutput
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_layout import valid_page_slots
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.logger import logger


class PersistentKvCacheConnectorWorker(BaseConnectorWorker):
    supports_attention_dp = True

    def __init__(self, llm_args: TorchLlmArgs) -> None:
        super().__init__(llm_args)

        if llm_args.tensor_parallel_size > 1 and not llm_args.enable_attention_dp:
            raise NotImplementedError("The example requires unsharded attention KV cache.")

        self.kv_cache_tensor = None

    def wait_for_save(self, stream: torch.cuda.Stream) -> None:
        # Make sure the forward pass is complete before beginning our save.
        stream.synchronize()

        for path, block_id in self._metadata.save:
            cpu_tensor = self.kv_cache_tensor[block_id].cpu()

            # Don't write anything if this specific block already exists.
            if Path(path).exists():
                continue

            # Publish only complete files. Multiple ADP owners can save the
            # same prefix concurrently, while another owner starts a restore.
            with NamedTemporaryFile(dir=Path(path).parent, delete=False) as tmp:
                temporary_path = Path(tmp.name)
            try:
                torch.save(cpu_tensor, temporary_path)
                os.replace(temporary_path, path)
            finally:
                temporary_path.unlink(missing_ok=True)


class PersistentKvCacheConnectorLeader(BaseConnectorLeader):
    supports_attention_dp = True

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> PersistentKvCacheConnectorMetadata:
        # NOTE: This is a simplified implementation, and does not work with chunked prefill.

        metadata = PersistentKvCacheConnectorMetadata()

        for req in scheduler_output.new_requests:
            # If we don't have any pending loads for this request, we can skip it.
            if req.request_id not in self.pending_loads:
                continue

            num_computed_blocks = req.computed_position // self.block_size
            block_ids = req.new_block_ids

            pending_load = self.pending_loads[req.request_id]

            # Ordinal -> page slot for the blocks that have a page. Blocks with
            # none keep their ordinal in `block_ids` so that entry `i` always
            # describes the same token range; they are dropped here so no
            # transfer can be built against one.
            slots = dict(valid_page_slots(block_ids))

            for file_path, block_pos in zip(
                pending_load, range(num_computed_blocks, len(block_ids))
            ):
                slot = slots.get(block_pos)
                if slot is None:
                    continue
                metadata.load.append((file_path, slot))

            # Break up the remainder of the token sequence into chunks.
            chunks = self._chunk_tokens(req.new_tokens)

            # For each chunk that isn't already on device, and isn't in our connector cache, we need to save it.
            for block_pos in range(num_computed_blocks + len(pending_load), len(block_ids)):
                slot = slots.get(block_pos)
                if slot is None:
                    continue
                if len(chunks[block_pos]) == self.block_size:
                    # KV for a block depends on the entire preceding prefix.
                    prefix = req.new_tokens[: (block_pos + 1) * self.block_size]
                    hashed_tokens = self._hash_tokens(prefix, req.cache_salt)

                    file_path = self._file_path(hashed_tokens)

                    metadata.save.append((file_path, slot))

        self.pending_loads = {}

        return metadata

    def _hash_tokens(self, tokens: list[int], cache_salt: Optional[str]) -> str:
        # cache_salt must participate in the hash so that requests carrying
        # different salts (or no salt) cannot collide on the same cache file.
        # Python's hash is randomized independently in different processes.
        content = json.dumps([cache_salt, tokens], separators=(",", ":"))
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _file_path(self, hash_value: str) -> Path:
        return Path(self.cache_folder) / f"{hash_value}.pt"

    def get_num_new_matched_tokens(
        self, request: LlmRequest, num_computed_tokens: int
    ) -> tuple[int, bool]:
        self.pending_loads[request.request_id] = []

        # Don't bother with sequences with partial matches.
        if (num_computed_tokens % self.block_size) != 0:
            return 0, False

        computed_blocks = num_computed_tokens // self.block_size

        # Get all the tokens that don't have a cache hit on device.
        tokens = request.get_tokens(0)
        # Leave the final prompt token for computing the first output logits.
        remaining_tokens = tokens[computed_blocks * self.block_size : -1]

        remaining_chunks = self._chunk_tokens(remaining_tokens)

        # For each chunk, check if it exists in our cache.
        for block_offset, chunk in enumerate(remaining_chunks):
            # Only do full blocks.
            if len(chunk) == self.block_size:
                prefix_end = (computed_blocks + block_offset + 1) * self.block_size
                hashed_tokens = self._hash_tokens(tokens[:prefix_end], request.cache_salt)

                file_path = self._file_path(hashed_tokens)

                # If we get a cache hit, we want to load it into device.
                # Otherwise, we can stop looking.
                if file_path.exists():
                    self.pending_loads[request.request_id].append(file_path)
                else:
                    break

        logger.info(
            f"KV CONNECTOR: Matched {len(self.pending_loads[request.request_id])} "
            f"blocks for request {request.request_id}"
        )

        return len(self.pending_loads[request.request_id]) * self.block_size, False

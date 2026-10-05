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
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs


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
        # The base implements accepted prefix loads and bounds saves to the
        # computed range. Keep the ADP content-key format below for reuse.
        return super().build_connector_meta(scheduler_output)

    def _hash_tokens(self, tokens: list[int], cache_salt: Optional[str]) -> str:
        # cache_salt must participate in the hash so that requests carrying
        # different salts (or no salt) cannot collide on the same cache file.
        # Python's hash is randomized independently in different processes.
        content = json.dumps([cache_salt, tokens], separators=(",", ":"))
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _file_path(self, hash_value: str) -> Path:
        return Path(self.cache_folder) / f"{hash_value}.pt"

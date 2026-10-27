# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
### :title Attention-DP KV Cache Connector
### :order 7
### :section Customization
"""V2 ADP filesystem connector: stable prefix keys and atomic publication.

Set both connector classes below and enable_attention_dp=True. Every owner
must see the same CONNECTOR_CACHE_FOLDER. Use a separate folder per model,
KV representation and block size. This example requires one full-attention
layer group and does not support chunked prefill or attention TP sharding.
Cache keys and file contents use the base filesystem example's format.
"""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from llm_kv_cache_connector import PersistentKvCacheConnectorLeader as BaseConnectorLeader
from llm_kv_cache_connector import PersistentKvCacheConnectorWorker as BaseConnectorWorker

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

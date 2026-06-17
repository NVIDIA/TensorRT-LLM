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
### :title Semantic KV Cache Connector
### :order 8
### :section Customization
"""Demonstrates semantic donor discovery with safe KV cache materialization.

The example is intentionally conservative:

* Semantic donor discovery is delegated to a provider.
* Non-identical semantic hits are recorded as discovery-only and do not change
  TensorRT-LLM cache state.
* The connector reports matched tokens only when the selected donor is
  exact-equivalent to the request prefix and the worker can load those KV blocks.

This gives semantic routing and telemetry a real connector lifecycle while
preserving TensorRT-LLM's exact KV cache semantics.  Non-identical semantic KV
materialization should use a request-local approximate reuse path before it is
allowed to report matched tokens.

How to run:

```bash
python llm_semantic_kv_cache_connector.py <model_path>
```
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import click
import torch

from tensorrt_llm import LLM, SamplingParams, logger
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import (
    KvCacheConnectorScheduler,
    KvCacheConnectorWorker,
    SchedulerOutput,
)
from tensorrt_llm._torch.pyexecutor.connectors.semantic_kv_cache import (
    LocalSemanticKvProvider,
    SemanticKvDonor,
    SemanticKvLookupRequest,
    SemanticKvMaterializationKind,
)
from tensorrt_llm.bindings.internal.batch_manager import LlmRequest
from tensorrt_llm.llmapi.llm_args import KvCacheConnectorConfig, TorchLlmArgs

SEMANTIC_CONNECTOR_CACHE_FOLDER_KEY = "SEMANTIC_CONNECTOR_CACHE_FOLDER"
SEMANTIC_CONNECTOR_MODEL_ID_KEY = "SEMANTIC_CONNECTOR_MODEL_ID"
SEMANTIC_CONNECTOR_MIN_SIMILARITY_KEY = "SEMANTIC_CONNECTOR_MIN_SIMILARITY"


@dataclass(frozen=True)
class PendingSemanticLoad:
    donor_id: str
    namespace: str
    file_paths: list[str]
    start_block: int
    token_count: int


@dataclass
class SemanticPersistentKvCacheConnectorMetadata:
    load: list[tuple[str, int]] = field(default_factory=list)
    save: list[tuple[str, int]] = field(default_factory=list)


class SemanticPersistentKvCacheConnectorWorker(KvCacheConnectorWorker):
    """Worker-side block copy implementation for semantic connector metadata."""

    def __init__(self, llm_args: TorchLlmArgs):
        super().__init__(llm_args)
        self.kv_cache_tensor = None

    def register_kv_caches(self, kv_cache_tensor: torch.Tensor):
        assert self.kv_cache_tensor is None, "KV cache tensor already registered"
        self.kv_cache_tensor = kv_cache_tensor

    def start_load_kv(self, stream: torch.cuda.Stream):
        if self._metadata is None:
            return
        for path, block_id in self._metadata.load:
            cpu_tensor = torch.load(path, map_location="cpu")
            self.kv_cache_tensor[block_id].copy_(cpu_tensor, non_blocking=False)

    def wait_for_layer_load(self, layer_idx: int, stream: torch.cuda.Stream):
        pass

    def save_kv_layer(self, layer_idx: int, stream: torch.cuda.Stream):
        pass

    def wait_for_save(self, stream: torch.cuda.Stream):
        if self._metadata is None:
            return
        if stream is not None:
            stream.synchronize()
        for path, block_id in self._metadata.save:
            path = Path(path)
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            cpu_tensor = self.kv_cache_tensor[block_id].cpu()
            torch.save(cpu_tensor, path)

    def get_finished(
            self,
            finished_gen_req_ids: list[int],
            started_loading_req_ids: list[int]) -> tuple[list[int], list[int]]:
        return [], []


class SemanticPersistentKvCacheConnectorLeader(KvCacheConnectorScheduler):
    """Scheduler-side semantic lookup and exact-equivalent load planning."""

    def __init__(self, llm_args: TorchLlmArgs):
        super().__init__(llm_args)
        self.block_size = self._llm_args.kv_cache_config.tokens_per_block
        self.cache_folder = Path(
            os.environ.get(SEMANTIC_CONNECTOR_CACHE_FOLDER_KEY,
                           "./semantic_connector_cache"))
        self.cache_folder.mkdir(parents=True, exist_ok=True)
        self.model_id = os.environ.get(SEMANTIC_CONNECTOR_MODEL_ID_KEY,
                                       self._infer_model_id())
        self.provider = LocalSemanticKvProvider(
            min_similarity=float(
                os.environ.get(SEMANTIC_CONNECTOR_MIN_SIMILARITY_KEY, "0.70")))
        self.pending_loads: dict[str, PendingSemanticLoad] = {}
        self._donor_records: dict[str, dict[str, Any]] = {}
        self.stats = {
            "semantic_hits": 0,
            "discovery_only_hits": 0,
            "exact_loads_advertised": 0,
            "materialization_rejected": 0,
            "donors_registered": 0,
        }
        self._load_manifest()

    def build_connector_meta(self, scheduler_output: SchedulerOutput):
        metadata = SemanticPersistentKvCacheConnectorMetadata()

        for req in scheduler_output.new_requests:
            request_id = str(req.request_id)
            pending = self.pending_loads.get(request_id)
            if pending is not None:
                for offset, file_path in enumerate(pending.file_paths):
                    block_pos = pending.start_block + offset
                    if block_pos < len(req.new_block_ids):
                        metadata.load.append(
                            (file_path, req.new_block_ids[block_pos]))

            namespace = self._namespace(req.cache_salt)
            chunks = self._chunk_tokens(req.new_tokens)
            for block_pos, chunk in enumerate(chunks):
                if len(chunk) != self.block_size:
                    continue
                file_path = self._file_path(request_id, namespace, block_pos)
                if block_pos < len(req.new_block_ids):
                    metadata.save.append((str(file_path),
                                          req.new_block_ids[block_pos]))

        self.pending_loads = {}
        return metadata

    def get_num_new_matched_tokens(
            self, request: LlmRequest,
            num_computed_tokens: int) -> tuple[int, bool]:
        request_id = str(request.request_id)
        self.pending_loads.pop(request_id, None)

        if num_computed_tokens % self.block_size != 0:
            return 0, False

        token_ids = list(request.get_tokens(0))
        namespace = self._namespace(request.cache_salt)
        result = self.provider.lookup(
            SemanticKvLookupRequest(
                request_id=request_id,
                token_ids=token_ids,
                prompt_text=None,
                model_id=self.model_id,
                namespace=namespace,
                already_computed_tokens=num_computed_tokens,
            ))

        if result is None:
            return 0, False

        self.stats["semantic_hits"] += 1
        if result.materialization_kind == SemanticKvMaterializationKind.DISCOVERY_ONLY:
            self.stats["discovery_only_hits"] += 1
            logger.info(
                "SEMANTIC KV CONNECTOR: discovery-only hit request=%s "
                "donor=%s similarity=%.4f reason=%s",
                request_id, result.donor_id, result.similarity, result.reason)
            return 0, False

        if result.materialization_kind != SemanticKvMaterializationKind.EXACT_PREFIX:
            self.stats["materialization_rejected"] += 1
            return 0, False

        donor_tokens = list(result.donor_token_ids or [])
        if donor_tokens != token_ids:
            self.stats["materialization_rejected"] += 1
            return 0, False

        record = self._donor_records.get(self._record_key(
            result.donor_id, namespace))
        if record is None:
            self.stats["materialization_rejected"] += 1
            return 0, False

        reusable_tokens = min(int(result.reusable_token_count),
                              int(record.get("token_count", 0)),
                              len(token_ids))
        reusable_tokens = self._block_align(reusable_tokens)
        if reusable_tokens <= num_computed_tokens:
            return 0, False

        start_block = num_computed_tokens // self.block_size
        end_block = reusable_tokens // self.block_size
        file_paths = []
        for block_pos in range(start_block, end_block):
            file_path = self._file_path(result.donor_id, namespace, block_pos)
            if not file_path.exists():
                break
            file_paths.append(str(file_path))

        matched_tokens = len(file_paths) * self.block_size
        if matched_tokens == 0:
            self.stats["materialization_rejected"] += 1
            return 0, False

        self.pending_loads[request_id] = PendingSemanticLoad(
            donor_id=result.donor_id,
            namespace=namespace,
            file_paths=file_paths,
            start_block=start_block,
            token_count=matched_tokens,
        )
        self.stats["exact_loads_advertised"] += 1
        logger.info(
            "SEMANTIC KV CONNECTOR: exact-equivalent load request=%s donor=%s tokens=%d",
            request_id, result.donor_id, matched_tokens)
        return matched_tokens, False

    def request_finished(self, request: LlmRequest,
                         cache_block_ids: list[int]) -> bool:
        token_ids = self._prompt_tokens(request)
        token_count = self._block_align(len(token_ids))
        if token_count <= 0:
            return False

        donor_id = str(request.request_id)
        namespace = self._namespace(request.cache_salt)
        record = {
            "donor_id": donor_id,
            "token_ids": token_ids,
            "token_count": token_count,
            "model_id": self.model_id,
            "namespace": namespace,
            "block_size": self.block_size,
            "num_blocks": len(cache_block_ids),
        }
        self._donor_records[self._record_key(donor_id, namespace)] = record
        self.provider.register_donor(
            SemanticKvDonor(
                donor_id=donor_id,
                token_ids=token_ids,
                prompt_text=None,
                model_id=self.model_id,
                namespace=namespace,
                metadata={"num_blocks": len(cache_block_ids)},
            ))
        self._write_manifest()
        self.stats["donors_registered"] += 1
        return False

    def update_state_after_alloc(self, request: LlmRequest,
                                 block_ids: list[int]):
        pass

    def _infer_model_id(self) -> str:
        for attr in ("model", "model_dir", "checkpoint_dir"):
            value = getattr(self._llm_args, attr, None)
            if value:
                return str(value)
        return "unknown-model"

    def _prompt_tokens(self, request: LlmRequest) -> list[int]:
        token_ids = list(request.get_tokens(0))
        for attr in ("py_prompt_len", "prompt_len", "py_orig_prompt_len",
                     "orig_prompt_len"):
            prompt_len = getattr(request, attr, None)
            if prompt_len is None:
                continue
            try:
                prompt_len = int(prompt_len)
            except (TypeError, ValueError):
                continue
            if prompt_len > 0:
                return token_ids[:min(prompt_len, len(token_ids))]
        return token_ids

    def _namespace(self, cache_salt: str | None) -> str:
        salt = cache_salt or "no-salt"
        raw = f"{self.model_id}:block={self.block_size}:salt={salt}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _record_key(self, donor_id: str, namespace: str) -> str:
        return f"{namespace}:{donor_id}"

    def _donor_dir(self, donor_id: str, namespace: str) -> Path:
        raw = f"{namespace}:{donor_id}".encode("utf-8")
        return self.cache_folder / hashlib.sha256(raw).hexdigest()

    def _file_path(self, donor_id: str, namespace: str, block_pos: int) -> Path:
        return self._donor_dir(donor_id, namespace) / f"block_{block_pos}.pt"

    def _manifest_path(self) -> Path:
        return self.cache_folder / "semantic_donors.json"

    def _load_manifest(self) -> None:
        try:
            with self._manifest_path().open(encoding="utf-8") as f:
                records = json.load(f)
        except (OSError, json.JSONDecodeError):
            records = []

        for record in records:
            if record.get("model_id") != self.model_id:
                continue
            if record.get("block_size") != self.block_size:
                continue
            donor_id = str(record["donor_id"])
            namespace = str(record["namespace"])
            token_ids = list(record["token_ids"])
            self._donor_records[self._record_key(donor_id, namespace)] = record
            self.provider.register_donor(
                SemanticKvDonor(
                    donor_id=donor_id,
                    token_ids=token_ids,
                    prompt_text=None,
                    model_id=self.model_id,
                    namespace=namespace,
                    metadata={"num_blocks": record.get("num_blocks", 0)},
                ))

    def _write_manifest(self) -> None:
        records = list(self._donor_records.values())
        tmp_path = self._manifest_path().with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(records, f)
        tmp_path.replace(self._manifest_path())

    def _chunk_tokens(self, tokens: list[int]) -> list[list[int]]:
        return [
            tokens[i:i + self.block_size]
            for i in range(0, len(tokens), self.block_size)
        ]

    def _block_align(self, num_tokens: int) -> int:
        return (num_tokens // self.block_size) * self.block_size


@click.command()
@click.argument("model", type=str)
def main(model: str):
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    this_module = __file__[__file__.rfind("/") + 1:__file__.rfind(".py")]

    kv_connector_config = KvCacheConnectorConfig(
        connector_module=this_module,
        connector_scheduler_class="SemanticPersistentKvCacheConnectorLeader",
        connector_worker_class="SemanticPersistentKvCacheConnectorWorker",
    )

    connector_cache_dir = TemporaryDirectory()
    os.environ[SEMANTIC_CONNECTOR_CACHE_FOLDER_KEY] = connector_cache_dir.name
    os.environ[SEMANTIC_CONNECTOR_MODEL_ID_KEY] = model

    prompt = (
        "Nvidia Corporation is an American technology company headquartered in "
        "Santa Clara, California. Founded in 1993 by Jensen Huang, Chris "
        "Malachowsky, and Curtis Priem, it develops graphics processing units "
        "(GPUs), system on a chips (SoCs), and application programming "
        "interfaces (APIs) for data science, high-performance computing, and "
        "mobile and automotive applications. Tell me about the company.")
    sampling_params = SamplingParams(max_tokens=32)

    llm = LLM(model=model,
              backend="pytorch",
              cuda_graph_config=None,
              kv_connector_config=kv_connector_config)
    output = llm.generate([prompt], sampling_params)
    text0 = output[0].outputs[0].text
    del llm

    llm = LLM(model=model,
              backend="pytorch",
              cuda_graph_config=None,
              kv_connector_config=kv_connector_config)
    output = llm.generate([prompt], sampling_params)
    text1 = output[0].outputs[0].text

    assert text0 == text1
    print("OK: exact-equivalent semantic donor discovery loaded KV blocks.")

    connector_cache_dir.cleanup()


if __name__ == "__main__":
    main()

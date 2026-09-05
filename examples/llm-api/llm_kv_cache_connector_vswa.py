# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""KV connector for a cache with one attention window size per layer group.

`llm_kv_cache_connector.py` is the connector to start from. It works whenever a
single tensor can describe the whole KV cache, which is every model with one
attention window size. This one covers what changes when that stops being true
-- variable sliding-window attention (VSWA), where the cache allocates one pool
per window size.

Five things differ, and each is marked `VSWA:` below.

1. Pages are addressed per layer group. A page index is scoped to a group, so
   the flat `block_ids` list does not exist and `register_kv_caches` is never
   called; `register_kv_cache_layout` is the entry point.
2. Every block-id callback switches to its `*_by_layer_group` form.
3. A cache key must include the layer group. The same token range lives in every
   group holding *different* KV, so a key derived from tokens alone collides
   across groups and one group's bytes overwrite another's.
4. A sliding group offers only its live window to save. Blocks the window has
   passed report no page and hold no readable KV, so `valid_page_slots` drops
   them and a connector persists at most `window_size` tokens per sequence for
   such a group -- not `prompt_len`. Size the store for that.
5. A block is only servable when *every* group holds it. The full-attention
   group keeps the whole prompt while the sliding group keeps a tail, so the
   prefix this connector can serve back is bounded by the smallest window. The
   lookup below intersects across groups and stops at the first ordinal any
   group misses.

Run with a VSWA model, for example:

    python llm_kv_cache_connector_vswa.py --model <path-to-gemma-3> \
        --max-attention-window 1024 1024 1024 1024 1024 32768
"""

import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import torch

from tensorrt_llm import LLM, SamplingParams, logger
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import (
    KvCacheConnectorScheduler,
    KvCacheConnectorWorker,
    SchedulerOutput,
)
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_layout import (
    KvCacheRegion,
    valid_page_slots,
)
from tensorrt_llm.bindings.internal.batch_manager import LlmRequest
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, KvCacheConnectorConfig, TorchLlmArgs

CONNECTOR_CACHE_FOLDER_KEY = "CONNECTOR_CACHE_FOLDER"


@dataclass
class VswaConnectorMetadata:
    # (path, layer_group_id, page_slot) -- the group is part of every entry
    # because a page slot only means something inside its own group.
    load: List[Tuple[str, int, int]] = field(default_factory=list)
    save: List[Tuple[str, int, int]] = field(default_factory=list)


class VswaKvCacheConnectorWorker(KvCacheConnectorWorker):
    def __init__(self, llm_args: TorchLlmArgs):
        super().__init__(llm_args)
        # VSWA (1): one region per layer group, not one tensor for the whole
        # cache. The region is kept rather than a `[num_slots, bytes]` view of
        # it, because `slot_tensor` checks the page slot before addressing it
        # and a plain view accepts any subscript.
        self.group_regions: Dict[int, KvCacheRegion] = {}
        self.layer_to_group: Dict[int, int] = {}

    def register_kv_cache_layout(self, layout) -> None:
        # VSWA (1): a group's buffers may coalesce into several regions, so a
        # region is addressed per (group, region). `region.slot_tensor(i)` is the
        # bytes of page slot `i` of that group.
        for group in layout.groups:
            if len(group.regions) != 1:
                raise NotImplementedError(
                    f"layer group {group.layer_group_id} has "
                    f"{len(group.regions)} regions; this example handles one. "
                    "Address `group.regions` individually to support more."
                )
            self.group_regions[group.layer_group_id] = group.regions[0]
            for layer_id in group.layer_ids:
                self.layer_to_group[layer_id] = group.layer_group_id
            logger.info(
                f"layer group {group.layer_group_id}: window={group.window_size}, "
                f"{len(group.layer_ids)} layers, {group.bytes_per_page} bytes per page"
            )

    def register_kv_caches(self, kv_cache_tensor: torch.Tensor):
        # Still abstract on the ABC, so it has to be defined even though
        # overriding `register_kv_cache_layout` means nothing calls it: the
        # default that forwards a single-pool cache here is what got replaced.
        raise NotImplementedError(
            "This connector registers through register_kv_cache_layout, which "
            "describes one tensor per layer group; a VSWA cache has several and "
            "no single tensor describes it."
        )

    def start_load_kv(self, stream: torch.cuda.Stream):
        for path, group_id, slot in self._metadata.load:
            cpu_tensor = torch.load(path, map_location="cpu")
            self.group_regions[group_id].slot_tensor(slot).copy_(cpu_tensor, non_blocking=False)

    def wait_for_layer_load(self, layer_idx: int, stream: torch.cuda.Stream):
        # `layer_to_group` is what turns a per-layer hook into the group whose
        # pages that layer reads. A connector that overlapped the transfer with
        # compute would wait here only on the group this layer belongs to.
        pass

    def save_kv_layer(self, layer_idx: int, stream: torch.cuda.Stream):
        pass

    def wait_for_save(self, stream: torch.cuda.Stream):
        stream.synchronize()
        for path, group_id, slot in self._metadata.save:
            if Path(path).exists():
                continue
            torch.save(self.group_regions[group_id].slot_tensor(slot).cpu(), path)

    def get_finished(
        self, finished_gen_req_ids: List[int], started_loading_req_ids: List[int]
    ) -> Tuple[List[int], List[int]]:
        return [], []


class VswaKvCacheConnectorLeader(KvCacheConnectorScheduler):
    def __init__(self, llm_args: TorchLlmArgs):
        super().__init__(llm_args)
        self.block_size = self._llm_args.kv_cache_config.tokens_per_block
        # VSWA (5): the lookup has to check every group, so the leader needs the
        # group count before the first request. The cache allocates one group
        # per distinct attention window, so the configured window list gives it
        # without waiting for a worker-side layout.
        windows = self._llm_args.kv_cache_config.max_attention_window or [None]
        self.num_layer_groups = len(set(windows))
        # request_id -> list of per-group file paths, one entry per matched block
        # ordinal, in ordinal order starting at the first locally uncomputed one.
        self.pending_loads: Dict[int, List[List[str]]] = {}
        self.cache_folder = os.environ.get(CONNECTOR_CACHE_FOLDER_KEY, "./connector_cache")
        os.makedirs(self.cache_folder, exist_ok=True)

    # VSWA (3): the group id goes into the key. Without it, group 0 and group 1
    # hash the same tokens to the same file and overwrite each other's KV.
    def _file_path(self, tokens: List[int], layer_group_id: int, salt: Optional[str]) -> str:
        digest = hashlib.sha256(repr((tokens, layer_group_id, salt)).encode()).hexdigest()
        return os.path.join(self.cache_folder, f"{digest}.pt")

    def _chunk_tokens(self, tokens: List[int]) -> List[List[int]]:
        return [tokens[i : i + self.block_size] for i in range(0, len(tokens), self.block_size)]

    def get_num_new_matched_tokens(
        self, request: LlmRequest, num_computed_tokens: int
    ) -> Tuple[int, bool]:
        self.pending_loads[request.request_id] = []

        # Partial blocks are not stored, so a partial local match has nothing
        # to append to.
        if num_computed_tokens % self.block_size != 0:
            return 0, False

        computed_blocks = num_computed_tokens // self.block_size
        remaining = request.get_tokens(0)[computed_blocks * self.block_size :]

        for chunk in self._chunk_tokens(remaining):
            if len(chunk) != self.block_size:
                break
            paths = [
                self._file_path(chunk, group_id, request.cache_salt)
                for group_id in range(self.num_layer_groups)
            ]
            # VSWA (5): every group or none. A block the sliding group dropped
            # is unservable even though the full-attention group still has it,
            # because the sliding layers would then attend to KV that was never
            # written.
            if not all(Path(path).exists() for path in paths):
                break
            self.pending_loads[request.request_id].append(paths)

        matched = len(self.pending_loads[request.request_id]) * self.block_size
        logger.info(
            f"VSWA KV CONNECTOR: matched {matched} tokens "
            f"({len(self.pending_loads[request.request_id])} blocks x "
            f"{self.num_layer_groups} groups) for request {request.request_id}"
        )
        return matched, False

    def update_state_after_alloc(self, request: LlmRequest, block_ids: List[int]):
        # VSWA (2): still abstract on the ABC, so it has to be defined even
        # though overriding the per-layer-group form means nothing calls it --
        # the base default that folds a single group back to here is replaced.
        raise NotImplementedError(
            "This connector is per layer group; use update_state_after_alloc_by_layer_group."
        )

    def update_state_after_alloc_by_layer_group(
        self, request: LlmRequest, block_ids_by_layer_group: List[List[int]]
    ) -> None:
        # VSWA (2): the flat `update_state_after_alloc` is never called here.
        pass

    def build_connector_meta(self, scheduler_output: SchedulerOutput):
        metadata = VswaConnectorMetadata()

        for req in scheduler_output.new_requests:
            pending_load = self.pending_loads.pop(req.request_id, [])
            by_group = req.new_block_ids_by_layer_group
            # VSWA (2): the flat list is empty with several groups.
            if len(by_group) != self.num_layer_groups:
                raise RuntimeError(
                    f"expected {self.num_layer_groups} layer groups from the "
                    f"cache, got {len(by_group)}. The window list this leader "
                    "derived its group count from does not describe the cache."
                )

            # `computed_position` excludes what the connector said it would
            # serve, so this is where the locally computed prefix ends and the
            # matched blocks begin.
            num_computed_blocks = req.computed_position // self.block_size

            # VSWA (4): a sliding group holds pages for its live window only, so
            # its list reports no page for the ordinals the window has passed.
            # Those ordinals stay in place to keep entry `i` describing the same
            # token range, and drop out here so no transfer targets them.
            valid_by_group = [dict(valid_page_slots(slots)) for slots in by_group]

            for offset, paths in enumerate(pending_load):
                ordinal = num_computed_blocks + offset
                for group_id, path in enumerate(paths):
                    slot = valid_by_group[group_id].get(ordinal)
                    if slot is None:
                        continue
                    metadata.load.append((path, group_id, slot))

            chunks = self._chunk_tokens(req.new_tokens)
            for ordinal in range(num_computed_blocks + len(pending_load), len(chunks)):
                if len(chunks[ordinal]) != self.block_size:
                    continue
                for group_id, valid_slots in enumerate(valid_by_group):
                    slot = valid_slots.get(ordinal)
                    if slot is None:
                        continue
                    path = self._file_path(chunks[ordinal], group_id, req.cache_salt)
                    metadata.save.append((path, group_id, slot))

        return metadata

    def request_finished(self, request: LlmRequest, cache_block_ids: List[int]) -> bool:
        # VSWA (2): as above -- abstract on the ABC, unreachable here because
        # `request_finished_by_layer_group` is overridden.
        raise NotImplementedError(
            "This connector is per layer group; use request_finished_by_layer_group."
        )

    def request_finished_by_layer_group(
        self, request: LlmRequest, cache_block_ids_by_layer_group: List[List[int]]
    ) -> bool:
        # VSWA (2) and (4): per group, and a sliding group's list covers its live
        # window only -- everything older is -1 and holds no readable KV.
        self.pending_loads.pop(request.request_id, None)
        return False


def build_llm(
    model: str,
    max_attention_window: List[int],
    max_seq_len: Optional[int] = None,
    free_gpu_memory_fraction: float = 0.5,
    use_kv_cache_manager_v2: "bool | str" = True,
    enable_block_reuse: bool = True,
):
    """An `LLM` wired to this connector. Shared by `main` and the e2e test."""
    connector_config = KvCacheConnectorConfig(
        connector_module=__name__,
        connector_scheduler_class="VswaKvCacheConnectorLeader",
        connector_worker_class="VswaKvCacheConnectorWorker",
    )
    return LLM(
        model=model,
        backend="pytorch",
        cuda_graph_config=None,
        disable_overlap_scheduler=True,
        max_seq_len=max_seq_len,
        kv_cache_config=KvCacheConfig(
            free_gpu_memory_fraction=free_gpu_memory_fraction,
            # One window size per layer, repeated cyclically. More than one
            # distinct value is what makes the cache allocate a layer group
            # per window -- and what this example exists to demonstrate.
            max_attention_window=list(max_attention_window),
            # VSWA needs the layout-describing registration path, which only
            # KVCacheManagerV2 provides. "auto" is enough for a model that
            # declares the preference itself, which Gemma-3 does.
            use_kv_cache_manager_v2=use_kv_cache_manager_v2,
            enable_block_reuse=enable_block_reuse,
        ),
        kv_connector_config=connector_config,
    )


@click.command()
@click.option("--model", type=str, required=True)
@click.option("--max-attention-window", type=int, multiple=True, required=True)
def main(model: str, max_attention_window: Tuple[int, ...]):
    with tempfile.TemporaryDirectory() as cache_folder:
        os.environ[CONNECTOR_CACHE_FOLDER_KEY] = cache_folder
        prompt = "The future of AI is"
        params = SamplingParams(max_tokens=16, ignore_eos=True)

        # Cold: nothing is cached, so every full block is saved.
        llm = build_llm(model, list(max_attention_window))
        try:
            print("cold:", llm.generate([prompt], params)[0].outputs[0].text)
        finally:
            llm.shutdown()

        # Warm: a fresh instance shares only the disk cache, so anything served
        # back came through the connector.
        llm = build_llm(model, list(max_attention_window))
        try:
            print("warm:", llm.generate([prompt], params)[0].outputs[0].text)
        finally:
            llm.shutdown()


if __name__ == "__main__":
    main()

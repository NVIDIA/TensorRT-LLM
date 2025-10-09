### :title KV Cache Connector
### :order 6
### :section Customization

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import click
import torch

from tensorrt_llm import LLM, SamplingParams, logger
from tensorrt_llm._torch.pyexecutor.kv_cache_connector import (
    KvCacheConnectorScheduler, KvCacheConnectorWorker, SchedulerOutput)
from tensorrt_llm.bindings.internal.batch_manager import LlmRequest
from tensorrt_llm.llmapi.llm_args import KvCacheConnectorConfig, TorchLlmArgs

# This is a simple example of the use of the KV cache connector.
# It persists KV cache contents into a folder, and can load them back on subsequent runs.
# See tensorrt_llm/_torch/pyexecutor/connector.py for details about the KV cache connector interface.
# NOTE: This example connector implementation is NOT suitable for production use.

CONNECTOR_CACHE_FOLDER_KEY = "CONNECTOR_CACHE_FOLDER"


@dataclass
class PersistentKvCacheConnectorMetadata:
    load: list[tuple[str, int]] = field(default_factory=list)
    save: list[tuple[str, int]] = field(default_factory=list)


class PersistentKvCacheConnectorWorker(KvCacheConnectorWorker):

    def __init__(self, llm_args: TorchLlmArgs):
        super().__init__(llm_args)

        self.kv_cache_tensor = None

    def register_kv_caches(self, kv_cache_tensor: torch.Tensor):
        assert self.kv_cache_tensor is None, "KV cache tensor already registered"
        self.kv_cache_tensor = kv_cache_tensor

    def start_load_kv(self, stream: torch.cuda.Stream):
        # Do all loads synchronously, and blockwise.
        for path, block_id in self._metadata.load:
            cpu_tensor = torch.load(path, map_location="cpu")

            # Copy into the device block.
            self.kv_cache_tensor[block_id].copy_(cpu_tensor, non_blocking=False)

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

            # Do a blocking save to the file. This way, we only return once all saves are complete.
            torch.save(cpu_tensor, path)

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

    def build_connector_meta(self, scheduler_output: SchedulerOutput):
        # NOTE: This is a simplified implementation, and does not work with chunked prefill.

        metadata = PersistentKvCacheConnectorMetadata()

        for req in scheduler_output.new_requests:
            # If we don't have any pending loads for this request, we can skip it.
            if req.request_id not in self.pending_loads:
                continue

            num_computed_blocks = req.computed_position // self.block_size
            block_ids = req.new_block_ids

            pending_load = self.pending_loads[req.request_id]

            for file_path, block_pos in zip(
                    pending_load, range(num_computed_blocks, len(block_ids))):
                metadata.load.append((file_path, block_ids[block_pos]))

            # Break up the remainder of the token sequence into chunks.
            chunks = self._chunk_tokens(req.new_tokens)

            # For each chunk that isn't already on device, and isn't in our connector cache, we need to save it.
            for block_pos in range(num_computed_blocks + len(pending_load),
                                   len(block_ids)):
                if len(chunks[block_pos]) == self.block_size:
                    hashed_tokens = self._hash_tokens(chunks[block_pos])

                    file_path = self._file_path(hashed_tokens)

                    metadata.save.append((file_path, block_ids[block_pos]))

        self.pending_loads = {}

        return metadata

    def _hash_tokens(self, tokens: list[int]) -> int:
        return abs(hash(tuple(tokens)))

    def _file_path(self, hash_value: int) -> Path:
        return Path(self.cache_folder) / f"{hash_value}.pt"

    def _chunk_tokens(self, tokens: list[int]) -> list[list[int]]:
        return [
            tokens[i:i + self.block_size]
            for i in range(0, len(tokens), self.block_size)
        ]

    def get_num_new_matched_tokens(
            self, request: LlmRequest,
            num_computed_tokens: int) -> tuple[int, bool]:
        self.pending_loads[request.request_id] = []

        # Don't bother with sequences with partial matches.
        if (num_computed_tokens % self.block_size) != 0:
            return 0, False

        computed_blocks = num_computed_tokens // self.block_size

        # Get all the tokens that don't have a cache hit on device.
        remaining_tokens = request.get_tokens(0)[computed_blocks *
                                                 self.block_size:]

        remaining_chunks = self._chunk_tokens(remaining_tokens)

        # For each chunk, check if it exists in our cache.
        for chunk in remaining_chunks:
            # Only do full blocks.
            if len(chunk) == self.block_size:
                hashed_tokens = self._hash_tokens(chunk)

                file_path = self._file_path(hashed_tokens)

                # If we get a cache hit, we want to load it into device.
                # Otherwise, we can stop looking.
                if file_path.exists():
                    self.pending_loads[request.request_id].append(file_path)
                else:
                    break

        logger.info(
            f"KV CONNECTOR: Matched {len(self.pending_loads[request.request_id])} blocks for request {request.request_id}"
        )

        return len(
            self.pending_loads[request.request_id]) * self.block_size, False

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

    kv_connector_config = KvCacheConnectorConfig(
        connector_module=this_module,
        connector_scheduler_class="PersistentKvCacheConnectorLeader",
        connector_worker_class="PersistentKvCacheConnectorWorker",
    )

    connector_cache_dir = TemporaryDirectory()
    os.environ[CONNECTOR_CACHE_FOLDER_KEY] = connector_cache_dir.name

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

    output = llm.generate([test_text], sampling_params)
    text0 = output[0].outputs[0].text

    print("First output: ", text0)
    print("Loading new LLM instance...")

    del llm

    llm = LLM(model=model,
              backend="pytorch",
              cuda_graph_config=None,
              kv_connector_config=kv_connector_config)

    output = llm.generate([test_text], sampling_params)
    text1 = output[0].outputs[0].text

    print("Second output (using connector cache): ", text1)

    assert text0 == text1

    connector_cache_dir.cleanup()


if __name__ == "__main__":
    main()

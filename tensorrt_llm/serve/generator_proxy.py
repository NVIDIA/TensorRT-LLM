# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""OpenAI-server adapter for a generator hosted behind Unix-domain IPC."""

from __future__ import annotations

import asyncio
from collections import deque
from types import SimpleNamespace
from typing import Any, AsyncIterator, Optional

from tensorrt_llm.inputs import create_input_processor
from tensorrt_llm.llmapi.llm import BaseLLM
from tensorrt_llm.llmapi.llm_utils import ModelLoader
from tensorrt_llm.serve.encode_batcher import EncodeBatcher
from tensorrt_llm.serve.generator_ipc import GeneratorIpcClient
from tensorrt_llm.serve.responses_utils import ConversationHistoryStore
from tensorrt_llm.serve.visual_gen_utils import AsyncDictStore


def _generator_kind(generator: Any) -> str:
    from tensorrt_llm import MultimodalEncoder
    from tensorrt_llm.visual_gen import VisualGen

    if isinstance(generator, VisualGen):
        return "visual_gen"
    if isinstance(generator, MultimodalEncoder):
        return "mm_encoder"
    if getattr(getattr(generator, "args", None), "encode_only", False):
        return "embedding"
    return "llm"


class GeneratorService:
    """Own generator state that must be shared by all HTTP frontends."""

    def __init__(
        self,
        generator: Any,
        *,
        embedding_max_queue_delay: float = 0.005,
        embedding_max_queue_size: int = 2048,
    ) -> None:
        self.generator = generator
        self.conversation_store = ConversationHistoryStore()
        self.video_store = AsyncDictStore()
        max_stats = getattr(getattr(generator, "args", None), "iter_stats_max_iterations", 1000)
        if not max_stats:
            max_stats = 1000
        self._iteration_stats: deque[Any] = deque(maxlen=None if max_stats < 0 else max_stats)
        self._stats_lock = asyncio.Lock()
        self._kv_cache_events_lock = asyncio.Lock()
        self._embedding_batcher: Optional[EncodeBatcher] = None
        self._embedding_batcher_started = False
        if _generator_kind(generator) == "embedding":
            engine = generator._encoder_executor.model_engine
            self._embedding_batcher = EncodeBatcher(
                generator.encode,
                max_batch_size=engine.batch_size,
                max_queue_delay=embedding_max_queue_delay,
                max_queue_size=embedding_max_queue_size,
                max_num_tokens=engine.max_num_tokens,
                max_seq_len=engine.max_seq_len,
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.generator, name)

    def snapshot(self) -> dict[str, Any]:
        """Return immutable frontend bootstrap data."""
        kind = _generator_kind(self.generator)
        args = getattr(self.generator, "args", None)
        if args is not None and hasattr(args, "model_copy"):
            args = args.model_copy(update={"tokenizer": None})
        tokenizer = getattr(self.generator, "tokenizer", None)
        tokenizer_path = getattr(getattr(tokenizer, "tokenizer", tokenizer), "name_or_path", None)
        executor = getattr(self.generator, "_executor", None)
        snapshot = {
            "generator_kind": kind,
            "args": args,
            "hf_model_dir": getattr(self.generator, "_hf_model_dir", None),
            "tokenizer_path": tokenizer_path,
            "llm_id": getattr(self.generator, "llm_id", None),
            "model": getattr(self.generator, "model", None),
            "disaggregated_params": getattr(self.generator, "disaggregated_params", None),
            "supports_collective_rpc": hasattr(self.generator, "collective_rpc"),
            "has_resource_governor": getattr(executor, "resource_governor_queue", None) is not None,
        }
        if kind == "visual_gen":
            executor = getattr(self.generator, "executor", None)
            snapshot.update(
                {
                    "extra_param_specs": getattr(self.generator, "extra_param_specs", {}),
                    "default_generation_params": getattr(executor, "default_generation_params", {}),
                    "executor_extra_param_specs": getattr(executor, "extra_param_specs", {}),
                }
            )
        if kind == "embedding":
            engine = self.generator._encoder_executor.model_engine
            snapshot["encoder_limits"] = {
                "max_seq_len": engine.max_seq_len,
                "max_num_tokens": engine.max_num_tokens,
                "batch_size": engine.batch_size,
            }
        return snapshot

    async def drain_stats(self, timeout: Optional[float]) -> list[Any]:
        async with self._stats_lock:
            return await self._drain_stats(timeout)

    async def _drain_stats(self, timeout: Optional[float]) -> list[Any]:
        stats = []
        async for stat in self.generator.get_stats_async(timeout):
            stats.append(stat)
            self._iteration_stats.append(stat)
        return stats

    async def drain_kv_cache_events(self, timeout: Optional[float]) -> list[Any]:
        events = []
        async with self._kv_cache_events_lock:
            async for event in self.generator.get_kv_cache_events_async(timeout):
                events.append(event)
        return events

    async def take_iteration_stats(self, timeout: Optional[float]) -> list[Any]:
        async with self._stats_lock:
            if not self._iteration_stats:
                await self._drain_stats(timeout)
            stats = list(self._iteration_stats)
            self._iteration_stats.clear()
            return stats

    def health_status(self) -> dict[str, Any]:
        executor = getattr(self.generator, "_executor", None)
        fatal_error = getattr(executor, "_fatal_error", None)
        health_check = getattr(self.generator, "_check_health", None)
        ready = bool(health_check()) if health_check is not None else fatal_error is None
        return {
            "ready": ready,
            "fatal_error": str(fatal_error) if fatal_error is not None else None,
        }

    def fatal_error(self) -> Optional[str]:
        executor = getattr(self.generator, "_executor", None)
        error = getattr(executor, "_fatal_error", None)
        return str(error) if error is not None else None

    async def conversation_call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        return await getattr(self.conversation_store, method)(*args, **kwargs)

    async def video_store_call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        return await getattr(self.video_store, method)(*args, **kwargs)

    async def encode(self, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        if self._embedding_batcher is None:
            return await asyncio.to_thread(self.generator.encode, inputs, *args, **kwargs)
        if args or kwargs:
            raise ValueError("Embedding IPC encode does not accept additional arguments")
        if not self._embedding_batcher_started:
            await self._embedding_batcher.start()
            self._embedding_batcher_started = True
        is_batch = bool(inputs) and isinstance(inputs[0], list)
        batch = inputs if is_batch else [inputs]
        results = await asyncio.gather(*(self._embedding_batcher.submit(item) for item in batch))
        return results if is_batch else results[0]

    def resource_governor_put(self, value: Any) -> None:
        self.generator._executor.resource_governor_queue.put(value)


class _RemoteStore:
    def __init__(self, client: GeneratorIpcClient, service_method: str) -> None:
        self._client = client
        self._service_method = service_method

    def __getattr__(self, method: str):
        async def call(*args: Any, **kwargs: Any) -> Any:
            return await self._client.call_async(self._service_method, method, *args, **kwargs)

        return call


class _RemoteResourceGovernorQueue:
    def __init__(self, client: GeneratorIpcClient) -> None:
        self._client = client

    def put(self, value: Any) -> None:
        self._client.call("resource_governor_put", value)


class GeneratorProxy(BaseLLM):
    """Generator-compatible facade used by each FastAPI process."""

    generator_kind: str
    owns_lifecycle: bool

    @property
    def llm_id(self) -> Optional[str]:
        return self._llm_id

    @property
    def disaggregated_params(self) -> Any:
        return self._disaggregated_params or {}

    def __init__(self, address: tuple[str, bytes], *, owns_lifecycle: bool = False) -> None:
        self._client = GeneratorIpcClient(address)
        snapshot = self._client.call("snapshot")
        self.generator_kind = snapshot["generator_kind"]
        self.owns_lifecycle = owns_lifecycle
        self.args = snapshot["args"]
        self._hf_model_dir = snapshot["hf_model_dir"]
        self._tokenizer_path = snapshot["tokenizer_path"]
        self._llm_id = snapshot["llm_id"]
        self.model = snapshot["model"]
        self._disaggregated_params = snapshot["disaggregated_params"]
        self.supports_collective_rpc = snapshot["supports_collective_rpc"]
        self.conversation_store = _RemoteStore(self._client, "conversation_call")
        self.video_store = _RemoteStore(self._client, "video_store_call")
        resource_governor_queue = (
            _RemoteResourceGovernorQueue(self._client)
            if snapshot["has_resource_governor"]
            else None
        )
        self._executor = SimpleNamespace(
            resource_governor_queue=resource_governor_queue,
            _fatal_error=None,
            doing_shutdown=False,
        )
        self.processor = None
        self.tokenizer = None
        self.input_processor = None
        self._hf_model_config = None
        self._generation_config = None

        if self.generator_kind == "visual_gen":
            self.extra_param_specs = snapshot["extra_param_specs"]
            self.executor = SimpleNamespace(
                default_generation_params=snapshot["default_generation_params"],
                extra_param_specs=snapshot["executor_extra_param_specs"],
            )
        else:
            self._load_input_processor()

        if self.generator_kind == "embedding":
            limits = snapshot["encoder_limits"]
            model_engine = SimpleNamespace(**limits)
            self._encoder_executor = SimpleNamespace(model_engine=model_engine)

    def _load_input_processor(self) -> None:
        self._hf_model_config = ModelLoader.load_hf_model_config(
            self.args.model,
            trust_remote_code=getattr(self.args, "trust_remote_code", False),
        )
        self._generation_config = ModelLoader.load_hf_generation_config(self.args.model)
        if getattr(self.args, "skip_tokenizer_init", False):
            return
        tokenizer_path = self._tokenizer_path or self.args.model
        custom_tokenizer = getattr(self.args, "custom_tokenizer", None)
        if custom_tokenizer:
            from importlib import import_module

            from tensorrt_llm.llmapi.llm_args import TOKENIZER_ALIASES

            tokenizer_class_path = TOKENIZER_ALIASES.get(custom_tokenizer, custom_tokenizer)
            module_path, class_name = tokenizer_class_path.rsplit(".", 1)
            tokenizer_class = getattr(import_module(module_path), class_name)
            tokenizer = tokenizer_class.from_pretrained(
                tokenizer_path,
                trust_remote_code=getattr(self.args, "trust_remote_code", False),
                use_fast=getattr(self.args, "tokenizer_mode", "auto") != "slow",
            )
        else:
            tokenizer = ModelLoader.load_hf_tokenizer(
                tokenizer_path,
                trust_remote_code=getattr(self.args, "trust_remote_code", False),
                use_fast=getattr(self.args, "tokenizer_mode", "auto") != "slow",
            )
        input_processor_kwargs = {}
        multimodal_config = getattr(self.args, "multimodal_config", None)
        video_pruning_rate = getattr(multimodal_config, "video_pruning_rate", None)
        if video_pruning_rate is not None:
            input_processor_kwargs["video_pruning_rate"] = video_pruning_rate
        self.input_processor = create_input_processor(
            self._hf_model_dir or tokenizer_path,
            tokenizer,
            getattr(self.args, "checkpoint_format", None),
            trust_remote_code=getattr(self.args, "trust_remote_code", False),
            **input_processor_kwargs,
        )
        self.tokenizer = self.input_processor.tokenizer

    def generate_async(self, *args: Any, **kwargs: Any):
        return self._client.generate_async(*args, **kwargs)

    def register_generation_alias(self, alias: str, promise: Any) -> None:
        self._client.register_alias(alias, promise)

    async def cancel_generation(self, alias: str) -> bool:
        return await self._client.cancel_alias(alias)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self._client.call("generate", *args, **kwargs)

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        return self._client.call("encode", *args, **kwargs)

    async def get_stats_async(self, timeout: Optional[float]) -> AsyncIterator[Any]:
        for stat in await self._client.call_async("drain_stats", timeout):
            yield stat

    async def take_iteration_stats(self, timeout: Optional[float]) -> list[Any]:
        return await self._client.call_async("take_iteration_stats", timeout)

    async def get_kv_cache_events_async(self, timeout: Optional[float]) -> AsyncIterator[Any]:
        for event in await self._client.call_async("drain_kv_cache_events", timeout):
            yield event

    async def collective_rpc(self, method: str, args: tuple[Any, ...]) -> Any:
        return await self._client.call_async("collective_rpc", method, args)

    def _check_health(self) -> bool:
        status = self._client.health_status()
        self._executor._fatal_error = status.get("fatal_error")
        return bool(status["ready"])

    def shutdown(self) -> None:
        client = getattr(self, "_client", None)
        if client is not None:
            client.close()

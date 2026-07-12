# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy, logical LoRA registry for the OpenEngine control plane."""

import asyncio
import json
from pathlib import Path

from openengine.v1 import lora_pb2

from tensorrt_llm.executor.request import LoRARequest


class LoraRegistry:
    """Validate adapters now and let TRT-LLM load weights on first use."""

    def __init__(self, reserved_identities: dict[str, int] | None = None) -> None:
        reserved_identities = dict(reserved_identities or {})
        if any(not name for name in reserved_identities):
            raise ValueError("Reserved LoRA names must not be empty")
        if any(adapter_id < 0 for adapter_id in reserved_identities.values()):
            raise ValueError("Reserved LoRA IDs must be non-negative")
        if len(set(reserved_identities.values())) != len(reserved_identities):
            raise ValueError("Reserved LoRA IDs must be unique")
        self._by_name: dict[str, lora_pb2.LoraAdapter] = {}
        self._identities_by_name: dict[str, lora_pb2.LoraAdapter] = {}
        self._names_by_id: dict[int, str] = {}
        self._names_by_path: dict[str, str] = {}
        self._reserved_by_name = reserved_identities
        self._reserved_by_id = {
            adapter_id: name for name, adapter_id in reserved_identities.items()
        }
        self._lock = asyncio.Lock()

    @staticmethod
    def _validated(adapter: lora_pb2.LoraAdapter) -> lora_pb2.LoraAdapter:
        if not adapter.lora_name:
            raise ValueError("lora_name must not be empty")
        if adapter.lora_id < 0:
            raise ValueError("lora_id must be non-negative")
        path = Path(adapter.source_path).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"LoRA source path is not a directory: {path}")
        if not (path / "adapter_config.json").is_file():
            raise ValueError(f"LoRA directory is missing adapter_config.json: {path}")
        try:
            config = json.loads((path / "adapter_config.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"LoRA adapter_config.json is invalid: {path}") from error
        if not isinstance(config, dict):
            raise ValueError(f"LoRA adapter_config.json must contain an object: {path}")
        if not any(
            (path / filename).is_file()
            for filename in ("adapter_model.safetensors", "adapter_model.bin")
        ):
            raise ValueError(f"LoRA directory is missing adapter weights: {path}")
        return lora_pb2.LoraAdapter(
            lora_id=adapter.lora_id, lora_name=adapter.lora_name, source_path=str(path)
        )

    async def load(self, adapter: lora_pb2.LoraAdapter) -> tuple[lora_pb2.LoraAdapter, bool]:
        validated = self._validated(adapter)
        async with self._lock:
            reserved_id = self._reserved_by_name.get(validated.lora_name)
            if reserved_id is not None:
                raise ValueError(
                    f"LoRA name {validated.lora_name!r} is reserved for a model-owned adapter"
                )
            reserved_name = self._reserved_by_id.get(validated.lora_id)
            if reserved_name is not None:
                raise ValueError(
                    f"LoRA ID {validated.lora_id} is reserved for model-owned adapter "
                    f"{reserved_name!r}"
                )
            identity = self._identities_by_name.get(validated.lora_name)
            if identity is not None:
                if identity != validated:
                    raise ValueError(
                        f"LoRA name {validated.lora_name!r} is permanently bound to different "
                        "attributes"
                    )
                if validated.lora_name in self._by_name:
                    return identity, True
                self._by_name[validated.lora_name] = identity
                return identity, False
            registered_name = self._names_by_id.get(validated.lora_id)
            if registered_name is not None:
                raise ValueError(
                    f"LoRA ID {validated.lora_id} is permanently bound to {registered_name!r}"
                )
            registered_name = self._names_by_path.get(validated.source_path)
            if registered_name is not None:
                raise ValueError(
                    f"LoRA path {validated.source_path!r} is permanently bound to "
                    f"{registered_name!r}"
                )
            self._by_name[validated.lora_name] = validated
            self._identities_by_name[validated.lora_name] = validated
            self._names_by_id[validated.lora_id] = validated.lora_name
            self._names_by_path[validated.source_path] = validated.lora_name
            return validated, False

    async def unload(self, name: str) -> lora_pb2.LoraAdapter:
        async with self._lock:
            adapter = self._by_name.pop(name, None)
        if adapter is None:
            raise KeyError(name)
        return adapter

    async def list(self) -> list[lora_pb2.LoraAdapter]:
        async with self._lock:
            return [self._by_name[name] for name in sorted(self._by_name)]

    async def request(self, name: str) -> LoRARequest:
        async with self._lock:
            adapter = self._by_name.get(name)
        if adapter is None:
            raise KeyError(name)
        return LoRARequest(adapter.lora_name, adapter.lora_id, adapter.source_path)

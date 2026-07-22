# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path

import pytest
from openengine.v1 import lora_pb2

from tensorrt_llm.openengine.lora_registry import LoraRegistry


def _adapter(path: Path, name: str = "adapter", adapter_id: int = 7) -> lora_pb2.LoraAdapter:
    path.mkdir()
    (path / "adapter_config.json").write_text("{}")
    (path / "adapter_model.safetensors").write_bytes(b"weights")
    return lora_pb2.LoraAdapter(lora_name=name, lora_id=adapter_id, source_path=str(path))


@pytest.mark.asyncio
async def test_lora_registry_is_lazy_idempotent_and_logically_unloads(
    tmp_path: Path,
) -> None:
    registry = LoraRegistry()
    adapter = _adapter(tmp_path / "adapter")

    registered, already_loaded = await registry.load(adapter)
    assert not already_loaded
    repeated, already_loaded = await registry.load(adapter)
    assert already_loaded
    assert repeated == registered

    request = await registry.request("adapter")
    assert request.lora_int_id == 7
    assert request.lora_path == str((tmp_path / "adapter").resolve())

    await registry.unload("adapter")
    with pytest.raises(KeyError):
        await registry.request("adapter")

    reloaded, already_loaded = await registry.load(adapter)
    assert not already_loaded
    assert reloaded == registered


@pytest.mark.asyncio
async def test_lora_registry_rejects_conflicting_identity(tmp_path: Path) -> None:
    registry = LoraRegistry()
    await registry.load(_adapter(tmp_path / "one", "one", 1))
    with pytest.raises(ValueError, match="ID 1"):
        await registry.load(_adapter(tmp_path / "two", "two", 1))


@pytest.mark.asyncio
async def test_lora_registry_tombstones_name_id_and_path_after_unload(tmp_path: Path) -> None:
    registry = LoraRegistry()
    original = _adapter(tmp_path / "one", "one", 1)
    await registry.load(original)
    await registry.unload("one")

    with pytest.raises(ValueError, match="name 'one'.*permanently"):
        await registry.load(_adapter(tmp_path / "renamed-path", "one", 2))
    with pytest.raises(ValueError, match="ID 1.*permanently"):
        await registry.load(_adapter(tmp_path / "reused-id", "two", 1))
    with pytest.raises(ValueError, match="path.*permanently"):
        await registry.load(
            lora_pb2.LoraAdapter(lora_name="two", lora_id=2, source_path=original.source_path)
        )


@pytest.mark.asyncio
async def test_lora_registry_reserves_model_owned_names_and_ids(tmp_path: Path) -> None:
    registry = LoraRegistry({"vision-lora": 0, "speech-lora": 1})

    with pytest.raises(ValueError, match="name 'vision-lora'.*model-owned"):
        await registry.load(_adapter(tmp_path / "reserved-name", "vision-lora", 7))
    with pytest.raises(ValueError, match="ID 0.*model-owned"):
        await registry.load(_adapter(tmp_path / "reserved-id", "user-adapter", 0))


@pytest.mark.asyncio
async def test_lora_registry_serializes_concurrent_idempotent_loads(tmp_path: Path) -> None:
    registry = LoraRegistry()
    adapter = _adapter(tmp_path / "adapter")

    results = await asyncio.gather(registry.load(adapter), registry.load(adapter))

    assert sorted(already_loaded for _, already_loaded in results) == [False, True]
    assert len(await registry.list()) == 1

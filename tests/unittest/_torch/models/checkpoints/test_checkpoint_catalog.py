# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import struct

import pytest
import torch
from safetensors.torch import save_file

from tensorrt_llm._torch.models.checkpoints.checkpoint_catalog import (
    CheckpointCatalog,
    CheckpointExtent,
    CheckpointObject,
    CheckpointTensor,
)
from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_catalog import (
    build_safetensors_checkpoint_catalog,
)
from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import HfCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import HfWeightLoader

pytestmark = pytest.mark.cpu_only


def test_name_only_catalog_exposes_names_without_byte_ranges() -> None:
    catalog = CheckpointCatalog(
        objects=(),
        tensors=(CheckpointTensor("model.bias"), CheckpointTensor("model.weight")),
    )

    assert catalog.tensor_names == frozenset({"model.bias", "model.weight"})
    assert not catalog.has_complete_byte_ranges
    assert catalog.get_tensor("model.weight").extents == ()


def test_safetensors_catalog_records_exact_payload_ranges(tmp_path) -> None:
    path = tmp_path / "model.safetensors"
    save_file(
        {
            "model.bias": torch.tensor([3.5], dtype=torch.float32),
            "model.weight": torch.tensor([1.25, -2.5], dtype=torch.float32),
        },
        path,
    )

    catalog = build_safetensors_checkpoint_catalog([str(path)])

    assert catalog.has_complete_byte_ranges
    assert catalog.tensor_names == frozenset({"model.bias", "model.weight"})
    weight = catalog.get_tensor("model.weight")
    assert weight.dtype == "F32"
    assert weight.shape == (2,)
    assert len(weight.extents) == 1
    extent = weight.extents[0]
    assert extent.object_id == "model.safetensors"
    assert extent.length_bytes == 8
    with path.open("rb") as checkpoint_file:
        checkpoint_file.seek(extent.offset_bytes)
        assert struct.unpack("<2f", checkpoint_file.read(extent.length_bytes)) == pytest.approx(
            (1.25, -2.5)
        )


def test_catalog_id_is_independent_of_root_path_and_input_order(tmp_path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    for root in (first_root, second_root):
        save_file({"a": torch.ones(2)}, root / "a.safetensors")
        save_file({"b": torch.ones(3)}, root / "b.safetensors")

    first = build_safetensors_checkpoint_catalog(
        [str(first_root / "b.safetensors"), str(first_root / "a.safetensors")]
    )
    second = build_safetensors_checkpoint_catalog(
        [str(second_root / "a.safetensors"), str(second_root / "b.safetensors")]
    )

    assert first.catalog_id == second.catalog_id
    assert [obj.object_id for obj in first.objects] == ["a.safetensors", "b.safetensors"]


def test_catalog_allows_shared_extents_but_rejects_out_of_bounds() -> None:
    checkpoint_object = CheckpointObject("shard", 16)
    catalog = CheckpointCatalog(
        objects=(checkpoint_object,),
        tensors=(
            CheckpointTensor("alias", extents=(CheckpointExtent("shard", 0, 8),)),
            CheckpointTensor("view", extents=(CheckpointExtent("shard", 4, 8),)),
        ),
    )

    assert catalog.get_tensor("alias").extents[0].offset_bytes == 0
    assert catalog.get_tensor("view").extents[0].offset_bytes == 4

    with pytest.raises(ValueError, match="extends beyond"):
        CheckpointCatalog(
            objects=(checkpoint_object,),
            tensors=(CheckpointTensor("a", extents=(CheckpointExtent("shard", 12, 8),)),),
        )


def test_safetensors_adapter_rejects_overlapping_entries(tmp_path) -> None:
    path = tmp_path / "overlap.safetensors"
    header = json.dumps(
        {
            "a": {"dtype": "U8", "shape": [8], "data_offsets": [0, 8]},
            "b": {"dtype": "U8", "shape": [8], "data_offsets": [4, 12]},
        }
    ).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(12))

    with pytest.raises(ValueError, match="SafeTensors entries.*overlap"):
        build_safetensors_checkpoint_catalog([str(path)])


def test_catalog_select_tensors_preserves_catalog_order() -> None:
    catalog = CheckpointCatalog(
        objects=(),
        tensors=(CheckpointTensor("b"), CheckpointTensor("a"), CheckpointTensor("c")),
    )

    assert [tensor.name for tensor in catalog.select_tensors({"a", "b"})] == ["b", "a"]
    with pytest.raises(KeyError, match="unknown checkpoint tensors"):
        catalog.select_tensors({"missing"})


def test_catalog_indexes_large_out_of_order_tensor_sets() -> None:
    tensors = tuple(CheckpointTensor(f"tensor.{index:05d}") for index in reversed(range(2_000)))
    catalog = CheckpointCatalog(objects=(), tensors=tensors)

    assert catalog.get_tensor("tensor.00000").name == "tensor.00000"
    assert catalog.get_tensor("tensor.01999").name == "tensor.01999"
    assert [
        tensor.name
        for tensor in catalog.select_tensors({"tensor.00000", "tensor.01000", "tensor.01999"})
    ] == ["tensor.01999", "tensor.01000", "tensor.00000"]


def test_hf_catalog_is_unavailable_for_non_safetensors_and_partial_loads(tmp_path) -> None:
    (tmp_path / "pytorch_model.bin").write_bytes(b"not inspected")
    assert HfWeightLoader().build_checkpoint_catalog(str(tmp_path)) is None

    path = tmp_path / "model.safetensors"
    save_file({"model.weight": torch.ones(1)}, path)
    loader = HfWeightLoader(partial_model_loading=True)
    assert loader.build_checkpoint_catalog(str(tmp_path)) is None


def test_hf_catalog_is_unavailable_for_lazy_or_custom_loader_paths(tmp_path) -> None:
    save_file({"model.weight": torch.ones(1)}, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "kimi_k3"}))
    assert HfWeightLoader().build_checkpoint_catalog(str(tmp_path)) is None

    class _CustomHfCheckpointLoader(HfCheckpointLoader):
        pass

    assert _CustomHfCheckpointLoader().build_checkpoint_catalog(str(tmp_path)) is None

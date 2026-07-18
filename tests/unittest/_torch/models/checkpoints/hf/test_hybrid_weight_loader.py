# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from safetensors.torch import save_file

from tensorrt_llm._torch.models.checkpoints import (
    HfCheckpointLoader,
    HfWeightLoader,
    MistralCheckpointLoader,
    MistralLarge3CheckpointLoader,
)
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import BaseWeightLoader
from tensorrt_llm._torch.models.checkpoints.hf import weight_loader as weight_loader_module
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

_MODEL_MODULES = {
    "LlamaForCausalLM": "tensorrt_llm._torch.models.modeling_llama",
    "Qwen2ForCausalLM": "tensorrt_llm._torch.models.modeling_qwen",
    "Qwen3ForCausalLM": "tensorrt_llm._torch.models.modeling_qwen3",
}


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setattr(weight_loader_module, "ENABLE_MULTI_DEVICE", False)


def _model(
    model_type: str = "LlamaForCausalLM",
    *,
    module_name: str | None = None,
    quant_algo=None,
    lora_config=None,
    force_dynamic_quantization=False,
):
    model_cls = type(
        model_type, (), {"__module__": module_name or _MODEL_MODULES.get(model_type, __name__)}
    )
    model = model_cls()
    model.model_config = SimpleNamespace(
        quant_config=SimpleNamespace(quant_algo=quant_algo),
        quant_config_dict=None,
        spec_config=None,
        lora_config=lora_config,
        force_dynamic_quantization=force_dynamic_quantization,
    )
    return model


def test_hybrid_loader_uses_mmap_path_without_full_prefetch(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    reference = {
        "model.layers.0.self_attn.q_proj.weight": torch.arange(32).reshape(8, 4),
        "model.norm.weight": torch.arange(4),
    }
    save_file(reference, str(checkpoint))

    loader = HfWeightLoader(safetensors_load_mode="hybrid")
    with (
        mock.patch.object(loader, "_get_hybrid_prefetch_policy", return_value=(0, False)),
        mock.patch.object(loader, "prefetch_file_chunks") as chunk_prefetch,
    ):
        weights = loader.load_weights(str(tmp_path), mapping=Mapping(), model=_model())

    chunk_prefetch.assert_not_called()
    loaded_weight = weights["model.layers.0.self_attn.q_proj.weight"]
    assert isinstance(loaded_weight, torch.Tensor)
    torch.testing.assert_close(loaded_weight, reference["model.layers.0.self_attn.q_proj.weight"])


def test_hybrid_loader_rejects_duplicate_keys_before_consumption(tmp_path):
    save_file({"duplicate.weight": torch.ones(2)}, str(tmp_path / "a.safetensors"))
    save_file({"duplicate.weight": torch.zeros(2)}, str(tmp_path / "b.safetensors"))

    loader = HfWeightLoader(safetensors_load_mode="hybrid")
    with (
        mock.patch.object(loader, "_get_hybrid_prefetch_policy", return_value=(0, False)),
        pytest.raises(RuntimeError, match="Duplicate SafeTensors key"),
    ):
        loader.load_weights(str(tmp_path), mapping=Mapping(), model=_model())


@pytest.mark.parametrize(
    "model,mapping,reason",
    [
        (None, Mapping(), "initialized model was not provided"),
        (_model("UnsupportedForCausalLM"), Mapping(), "model type"),
        (_model(quant_algo="FP8"), Mapping(), "quantized checkpoints"),
        (_model(lora_config=object()), Mapping(), "LoRA-enabled models"),
        (_model(force_dynamic_quantization=True), Mapping(), "dynamic quantization"),
        (_model(), Mapping(world_size=2, cp_size=2), "context parallelism"),
    ],
)
def test_hybrid_loader_falls_back_before_loading(tmp_path, model, mapping, reason):
    (tmp_path / "model.safetensors").touch()
    eager_weights = {"weight": object()}
    loader = HfWeightLoader(safetensors_load_mode="hybrid")

    with (
        mock.patch.object(loader, "_prefetch_and_load", return_value=eager_weights) as eager_load,
        mock.patch.object(loader, "_prefetch_and_load_hybrid") as hybrid_load,
        mock.patch.object(logger, "warning") as warning,
    ):
        weights = loader.load_weights(
            str(tmp_path), mapping=mapping, model=model, _load_format="AUTO"
        )

    assert weights is eager_weights
    eager_load.assert_called_once()
    hybrid_load.assert_not_called()
    warning.assert_called_once()
    assert reason in warning.call_args.args[0]


def test_same_name_custom_model_is_not_eligible():
    model = _model("LlamaForCausalLM", module_name=__name__)

    reason = HfWeightLoader._hybrid_ineligibility_reason(
        model, Mapping(), checkpoint_format="HF", uses_custom_weight_mapper=False
    )

    assert reason is not None
    assert "model type" in reason


def test_distributed_non_mpi_loader_is_not_eligible(monkeypatch):
    monkeypatch.setattr(weight_loader_module, "mpi_disabled", lambda: True)

    reason = HfWeightLoader._hybrid_ineligibility_reason(
        _model(),
        Mapping(world_size=2, tp_size=2),
        checkpoint_format="HF",
        uses_custom_weight_mapper=False,
    )

    assert reason == "distributed hybrid loading requires MPI-launched ranks"


def test_distributed_direct_weight_loader_requires_explicit_auto_format():
    reason = HfWeightLoader._hybrid_coordination_error(
        Mapping(world_size=2, tp_size=2),
        "HF",
        None,
    )

    assert reason == ("distributed hybrid loading requires an explicit AUTO load format")


def test_hybrid_chunk_plan_is_disjoint_and_complete_across_local_ranks(tmp_path):
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    first.write_bytes(b"a" * 10)
    second.write_bytes(b"b" * 5)
    loader = HfWeightLoader(safetensors_load_mode="hybrid", hybrid_prefetch_chunk_size=4)

    rank_chunks = []
    for rank in range(3):
        with mock.patch.object(loader, "_get_local_rank_and_size", return_value=(rank, 3)):
            rank_chunks.append(loader._local_prefetch_chunks([str(second), str(first)]))

    flattened = [chunk for chunks in rank_chunks for chunk in chunks]
    assert len(flattened) == len(set(flattened))
    assert sorted(flattened) == [
        (str(first), 0, 4),
        (str(first), 4, 4),
        (str(first), 8, 2),
        (str(second), 0, 4),
        (str(second), 4, 1),
    ]


def test_hybrid_prefetch_reads_only_assigned_chunks(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"x" * 13)
    loader = HfWeightLoader(
        safetensors_load_mode="hybrid",
        hybrid_prefetch_chunk_size=4,
        hybrid_prefetch_workers_per_rank=2,
    )

    with (
        mock.patch.object(loader, "_get_local_rank_and_size", return_value=(1, 2)),
        mock.patch.object(loader, "_prefetch_one_chunk") as prefetch,
    ):
        loader.prefetch_file_chunks([str(checkpoint)])

    assert sorted(call.args for call in prefetch.call_args_list) == [
        (str(checkpoint), 4, 4),
        (str(checkpoint), 12, 1),
    ]


def test_mismatched_rank_load_modes_fail_before_loading(monkeypatch):
    class FakeWorldCommunicator:
        @staticmethod
        def Get_size():
            return 2

        @staticmethod
        def allgather(selection):
            assert selection == ("hybrid", "AUTO")
            return [("hybrid", "AUTO"), ("eager", "AUTO")]

    monkeypatch.setattr(weight_loader_module, "ENABLE_MULTI_DEVICE", True)
    monkeypatch.setattr(weight_loader_module, "mpi_disabled", lambda: False)
    monkeypatch.setattr(weight_loader_module, "mpi_comm", lambda: FakeWorldCommunicator())

    loader = HfWeightLoader(safetensors_load_mode="hybrid")
    with pytest.raises(RuntimeError, match="must match across all MPI ranks"):
        loader._get_coordinated_safetensors_load_mode(
            Mapping(world_size=2, tp_size=2), "HF", "AUTO"
        )


def test_active_communicator_must_match_mapping(monkeypatch):
    communicator = mock.Mock()
    communicator.Get_size.return_value = 4
    monkeypatch.setattr(weight_loader_module, "ENABLE_MULTI_DEVICE", True)
    monkeypatch.setattr(weight_loader_module, "mpi_disabled", lambda: False)
    monkeypatch.setattr(weight_loader_module, "mpi_comm", lambda: communicator)

    loader = HfWeightLoader(safetensors_load_mode="hybrid")
    load_mode, reason = loader._get_coordinated_safetensors_load_mode(
        Mapping(world_size=2, tp_size=2), "HF", "AUTO"
    )

    assert load_mode == "hybrid"
    assert reason is not None
    assert "active MPI communicator size (4)" in reason
    communicator.allgather.assert_not_called()


def test_rank_local_mapping_rejects_larger_active_communicator(monkeypatch):
    communicator = mock.Mock()
    communicator.Get_size.return_value = 2
    monkeypatch.setattr(weight_loader_module, "ENABLE_MULTI_DEVICE", True)
    monkeypatch.setattr(weight_loader_module, "mpi_disabled", lambda: False)
    monkeypatch.setattr(weight_loader_module, "mpi_comm", lambda: communicator)

    loader = HfWeightLoader(safetensors_load_mode="hybrid")
    load_mode, reason = loader._get_coordinated_safetensors_load_mode(Mapping(), "HF", None)

    assert load_mode == "hybrid"
    assert reason is not None
    assert "active MPI communicator size (2)" in reason
    communicator.allgather.assert_not_called()


def test_hybrid_node_communicator_is_derived_from_active_communicator(monkeypatch):
    node_communicator = object()
    active_communicator = mock.Mock()
    active_communicator.Split_type.return_value = node_communicator
    monkeypatch.setattr(weight_loader_module, "ENABLE_MULTI_DEVICE", True)
    monkeypatch.setattr(weight_loader_module, "mpi_disabled", lambda: False)
    monkeypatch.setattr(weight_loader_module, "mpi_comm", lambda: active_communicator)

    assert HfWeightLoader._get_hybrid_node_communicator() is node_communicator
    active_communicator.Split_type.assert_called_once_with(
        weight_loader_module._MPI.COMM_TYPE_SHARED
    )


def test_hybrid_node_communicator_is_freed_after_success():
    loader = HfWeightLoader(safetensors_load_mode="hybrid")
    node_communicator = mock.Mock()
    expected_weights = {"weight": object()}

    with (
        mock.patch.object(loader, "_get_hybrid_node_communicator", return_value=node_communicator),
        mock.patch.object(
            loader,
            "_prefetch_and_load_hybrid_with_communicator",
            return_value=expected_weights,
        ) as load_with_communicator,
    ):
        weights = loader._prefetch_and_load_hybrid(["model.safetensors"])

    assert weights is expected_weights
    load_with_communicator.assert_called_once_with(["model.safetensors"], node_communicator)
    node_communicator.Free.assert_called_once_with()


def test_remote_prefetch_error_is_raised_on_a_healthy_rank(monkeypatch):
    class FakeWorldCommunicator:
        @staticmethod
        def allgather(error_message):
            assert error_message is None
            return [None, "OSError: remote read failed"]

    monkeypatch.setattr(weight_loader_module, "ENABLE_MULTI_DEVICE", True)
    monkeypatch.setattr(weight_loader_module, "mpi_disabled", lambda: False)
    monkeypatch.setattr(weight_loader_module, "mpi_comm", lambda: FakeWorldCommunicator())

    with pytest.raises(RuntimeError, match="Rank 1 failed during hybrid checkpoint prefetch"):
        HfWeightLoader._raise_on_rank_error("hybrid checkpoint prefetch", None)


def test_local_prefetch_error_is_coordinated_before_barrier(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    loader = HfWeightLoader(safetensors_load_mode="hybrid")

    class FakeWorldCommunicator:
        @staticmethod
        def allgather(error_message):
            return [error_message]

    local_communicator = mock.Mock()
    local_communicator.Get_rank.return_value = 0
    local_communicator.Get_size.return_value = 1
    monkeypatch.setattr(weight_loader_module, "ENABLE_MULTI_DEVICE", True)
    monkeypatch.setattr(weight_loader_module, "mpi_disabled", lambda: False)
    monkeypatch.setattr(weight_loader_module, "mpi_comm", lambda: FakeWorldCommunicator())

    with (
        mock.patch.object(loader, "_get_hybrid_node_communicator", return_value=local_communicator),
        mock.patch.object(loader, "_get_hybrid_prefetch_policy", return_value=(1, True)),
        mock.patch.object(loader, "prefetch_file_chunks", side_effect=OSError("read failed")),
        pytest.raises(RuntimeError, match="hybrid checkpoint prefetch"),
    ):
        loader._prefetch_and_load_hybrid([str(checkpoint)])

    local_communicator.Barrier.assert_not_called()
    local_communicator.Free.assert_called_once_with()


def test_remote_policy_error_prevents_local_memory_allreduce(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    loader = HfWeightLoader(safetensors_load_mode="hybrid")

    class FakeWorldCommunicator:
        @staticmethod
        def allgather(error_message):
            assert error_message is None
            return [None, "OSError: remote stat failed"]

    local_communicator = mock.Mock()
    monkeypatch.setattr(weight_loader_module, "ENABLE_MULTI_DEVICE", True)
    monkeypatch.setattr(weight_loader_module, "mpi_disabled", lambda: False)
    monkeypatch.setattr(weight_loader_module, "mpi_comm", lambda: FakeWorldCommunicator())
    with pytest.raises(RuntimeError, match="remote stat failed"):
        loader._get_hybrid_prefetch_policy([str(checkpoint)], local_communicator)

    local_communicator.allreduce.assert_not_called()


def test_mapper_assigned_after_construction_forces_eager_fallback(tmp_path):
    (tmp_path / "model.safetensors").touch()
    eager_weights = {"weight": object()}
    weight_loader = HfWeightLoader(safetensors_load_mode="hybrid")
    checkpoint_loader = HfCheckpointLoader(weight_loader=weight_loader)
    checkpoint_loader.weight_mapper = mock.Mock()

    with (
        mock.patch.object(
            weight_loader, "_prefetch_and_load", return_value=eager_weights
        ) as eager_load,
        mock.patch.object(weight_loader, "_prefetch_and_load_hybrid") as hybrid_load,
        mock.patch.object(logger, "warning"),
    ):
        weights = checkpoint_loader.load_weights(str(tmp_path), Mapping(), model=_model())

    assert weights is eager_weights
    eager_load.assert_called_once()
    hybrid_load.assert_not_called()


@pytest.mark.parametrize(
    "load_kwargs,expected_load_format",
    [({}, "AUTO"), ({"_load_format": "GMS"}, "GMS")],
)
def test_hf_checkpoint_loader_forwards_hybrid_policy_metadata(load_kwargs, expected_load_format):
    weight_loader = HfWeightLoader()
    checkpoint_loader = HfCheckpointLoader(weight_loader=weight_loader)
    expected_weights = {"weight": object()}

    with mock.patch.object(
        weight_loader, "load_weights", return_value=expected_weights
    ) as load_weights:
        weights = checkpoint_loader.load_weights("checkpoint", Mapping(), **load_kwargs)

    assert weights is expected_weights
    _args, kwargs = load_weights.call_args
    assert kwargs["_checkpoint_format"] == "HF"
    assert kwargs["_uses_custom_weight_mapper"] is False
    assert kwargs["_load_format"] == expected_load_format


@pytest.mark.parametrize(
    "loader_type,checkpoint_format",
    [
        (MistralCheckpointLoader, "mistral"),
        (MistralLarge3CheckpointLoader, "mistral_large_3"),
    ],
)
def test_mistral_disk_fallback_forwards_checkpoint_format(loader_type, checkpoint_format):
    checkpoint_loader = loader_type()
    weight_loader = checkpoint_loader.weight_loader

    with mock.patch.object(weight_loader, "load_weights", return_value={}) as load_weights:
        assert checkpoint_loader.load_weights("checkpoint", mapping=Mapping()) == {}

    _args, kwargs = load_weights.call_args
    assert kwargs["use_consolidated"] is True
    assert kwargs["_checkpoint_format"] == checkpoint_format
    assert kwargs["_uses_custom_weight_mapper"] is False
    assert "_load_format" not in kwargs


def test_hf_checkpoint_loader_preserves_strict_custom_loader_signature():
    expected_weights = {"weight": object()}

    class StrictCustomWeightLoader(BaseWeightLoader):
        def load_weights(self, checkpoint_dir: str, mapping: Mapping):
            assert checkpoint_dir == "checkpoint"
            assert mapping.world_size == 1
            return expected_weights

    loader = HfCheckpointLoader(weight_loader=StrictCustomWeightLoader())

    assert loader.load_weights("checkpoint", Mapping(), _load_format="AUTO") is expected_weights


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"safetensors_load_mode": "unknown"}, "Unsupported SafeTensors load mode"),
        (
            {
                "safetensors_load_mode": "hybrid",
                "hybrid_prefetch_chunk_size": 0,
            },
            "chunk_size must be positive",
        ),
        (
            {
                "safetensors_load_mode": "hybrid",
                "hybrid_prefetch_workers_per_rank": 0,
            },
            "workers_per_rank must be positive",
        ),
    ],
)
def test_hybrid_options_are_validated(kwargs, match):
    with pytest.raises(ValueError, match=match):
        HfWeightLoader(**kwargs)

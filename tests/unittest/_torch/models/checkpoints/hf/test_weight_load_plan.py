# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for selectable HF SafeTensors weight-load policies."""

import threading
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from safetensors.torch import save_file

from tensorrt_llm._torch.models.checkpoints import (
    DEFAULT_WEIGHT_LOAD_PLAN,
    HfCheckpointLoader,
    HfWeightLoader,
    MistralCheckpointLoader,
    MistralLarge3CheckpointLoader,
    WeightLoadPolicy,
)
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import (
    BaseWeightLoader,
    BorrowedWeightStorageRetentionError,
    WeightBatch,
    WeightBatchLease,
    WeightBatchStream,
    WeightGroup,
    WeightSegment,
)
from tensorrt_llm._torch.models.checkpoints.hf import (
    shared_host_stream as shared_host_stream_module,
)
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
    monkeypatch.delenv("TRTLLM_HF_WEIGHT_LOAD_PLAN", raising=False)
    monkeypatch.delenv("TRTLLM_HF_WEIGHT_CACHE", raising=False)
    monkeypatch.delenv("TLLM_OVERRIDE_LAYER_NUM", raising=False)


def _enable_single_rank_mpi(monkeypatch):
    node_communicator = mock.Mock()
    node_communicator.Get_rank.return_value = 0
    node_communicator.Get_size.return_value = 1
    world_communicator = mock.Mock()
    world_communicator.Get_rank.return_value = 0
    world_communicator.Get_size.return_value = 1
    world_communicator.allgather.side_effect = lambda value: [value]
    world_communicator.Split_type.return_value = node_communicator
    monkeypatch.setattr(weight_loader_module, "ENABLE_MULTI_DEVICE", True)
    monkeypatch.setattr(weight_loader_module, "mpi_disabled", lambda: False)
    monkeypatch.setattr(weight_loader_module, "mpi_comm", lambda: world_communicator)
    return world_communicator, node_communicator


def _model(
    model_type: str = "LlamaForCausalLM",
    *,
    module_name: str | None = None,
    quant_algo=None,
    quant_config_dict=None,
    spec_config=None,
    lora_config=None,
    force_dynamic_quantization=False,
    moe_load_balancer=None,
    enable_min_latency=False,
):
    model_cls = type(
        model_type, (), {"__module__": module_name or _MODEL_MODULES.get(model_type, __name__)}
    )
    model = model_cls()
    model.model_config = SimpleNamespace(
        quant_config=SimpleNamespace(quant_algo=quant_algo),
        quant_config_dict=quant_config_dict,
        spec_config=spec_config,
        lora_config=lora_config,
        force_dynamic_quantization=force_dynamic_quantization,
        moe_load_balancer=moe_load_balancer,
        enable_min_latency=enable_min_latency,
    )
    return model


class _TestWeightLease(WeightBatchLease):
    def __init__(self, batch, payload, *, direct_buffer_enabled=False):
        self._batch = batch
        self._payload = payload
        self._direct_buffer_enabled = direct_buffer_enabled
        self.released = False

    @property
    def batch(self):
        return self._batch

    def view(self, segment):
        assert not self.released
        start = segment.payload_offset
        return memoryview(self._payload)[start : start + segment.nbytes]

    def borrow_direct_buffer(self, segment):
        if not self._direct_buffer_enabled:
            return None
        assert not self.released
        start = segment.payload_offset
        return memoryview(self._payload)[start : start + segment.nbytes]

    def release(self):
        self.released = True


class _TestWeightStream(WeightBatchStream):
    def __init__(self, groups, batches_and_payloads, *, direct_buffer_enabled=False):
        self._groups = tuple(groups)
        self._leases = [
            _TestWeightLease(batch, payload, direct_buffer_enabled=direct_buffer_enabled)
            for batch, payload in batches_and_payloads
        ]
        self.all_leases = tuple(self._leases)
        self.completed = []
        self.started = []
        self.aborted = []
        self.materializations = []
        self.finalized = False

    @property
    def groups(self):
        return self._groups

    def begin_next(self):
        if not self._leases:
            return None
        return self._leases.pop(0)

    def start(self, error=None):
        self.started.append(error)
        if error is not None:
            raise RuntimeError(f"start consensus: {type(error).__name__}: {error}")

    def complete(self, lease, error=None):
        assert lease.released
        self.completed.append((lease.batch.sequence, error))
        if error is not None:
            if isinstance(error, BorrowedWeightStorageRetentionError):
                raise error
            raise RuntimeError(f"consensus: {type(error).__name__}: {error}")

    def record_materialization(self, *, direct, nbytes):
        self.materializations.append((direct, nbytes))

    def abort(self, error):
        self.aborted.append(error)

    def finalize(self):
        self.finalized = True


class _TestIncrementalMapper:
    borrowed_source_tensors_safe = False

    def __init__(self):
        self.events = []

    def begin_incremental_load(self, groups):
        self.events.append(("begin", tuple(groups)))

    def record_incremental_group_loaded(self, group_id):
        self.events.append(("record", group_id))

    def finalize_incremental_load(self):
        self.events.append(("finalize",))

    def abort_incremental_load(self):
        self.events.append(("abort",))


def _stream_batch(
    sequence,
    tensor_offset,
    payload_nbytes,
    *,
    complete,
    group_id="model",
    key="model.weight",
):
    segment = WeightSegment(
        key=key,
        dtype="F32",
        shape=(4,),
        tensor_nbytes=16,
        tensor_offset=tensor_offset,
        payload_offset=0,
        nbytes=payload_nbytes,
    )
    return WeightBatch(
        sequence=sequence,
        slot=sequence % 2,
        group_id=group_id,
        group_keys=(key,),
        group_complete=complete,
        segments=(segment,),
        payload_nbytes=payload_nbytes,
    )


def test_model_loader_materializes_only_complete_stream_groups():
    from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader

    reference = torch.arange(4, dtype=torch.float32)
    payload = reference.numpy().tobytes()
    stream = _TestWeightStream(
        [WeightGroup("model", ("model.weight",))],
        [
            (_stream_batch(0, 0, 8, complete=False), payload[:8]),
            (_stream_batch(1, 8, 8, complete=True), payload[8:]),
        ],
        direct_buffer_enabled=True,
    )
    mapper = _TestIncrementalMapper()
    mapper.borrowed_source_tensors_safe = True
    loaded = []

    def load_weights(weights, weight_mapper, allow_partial_loading=False):
        assert weight_mapper is mapper
        assert allow_partial_loading
        loaded.append(weights["model.weight"].clone())

    loader = object.__new__(ModelLoader)
    loader._load_weight_stream(load_weights, stream, mapper, model=torch.nn.Module())

    assert len(loaded) == 1
    torch.testing.assert_close(loaded[0], reference)
    assert stream.completed == [(0, None), (1, None)]
    assert mapper.events == [
        ("begin", stream.groups),
        ("record", "model"),
        ("finalize",),
    ]
    assert stream.materializations == [(False, 16)]


def test_model_loader_borrows_complete_registered_group_through_h2d_sync():
    from tensorrt_llm._torch.pyexecutor import model_loader as model_loader_module
    from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader

    class BorrowSafeMapper(_TestIncrementalMapper):
        borrowed_source_tensors_safe = True

    reference = torch.arange(4, dtype=torch.float32)
    payload = bytearray(reference.numpy().tobytes())
    stream = _TestWeightStream(
        [WeightGroup("model", ("model.weight",))],
        [(_stream_batch(0, 0, 16, complete=True), payload)],
        direct_buffer_enabled=True,
    )
    lease = stream.all_leases[0]
    mapper = BorrowSafeMapper()
    source_ptr = torch.frombuffer(payload, dtype=torch.float32).data_ptr()
    loaded = []

    def load_weights(weights, allow_partial_loading=False):
        assert allow_partial_loading
        assert weights["model.weight"].data_ptr() == source_ptr
        loaded.append(weights["model.weight"].clone())

    def synchronize():
        assert not lease.released

    loader = object.__new__(ModelLoader)
    with (
        mock.patch.object(
            model_loader_module._StagedStreamTensor,
            "allocate",
            side_effect=AssertionError("unexpected staging"),
        ),
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.cuda, "current_stream") as current_stream,
    ):
        current_stream.return_value.synchronize.side_effect = synchronize
        loader._load_weight_stream(load_weights, stream, mapper, model=torch.nn.Module())

    assert lease.released
    assert len(loaded) == 1
    torch.testing.assert_close(loaded[0], reference)
    assert stream.materializations == [(True, 16)]
    current_stream.return_value.synchronize.assert_called_once_with()


def test_model_loader_stages_when_direct_buffer_is_unavailable():
    from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader

    class BorrowSafeMapper(_TestIncrementalMapper):
        borrowed_source_tensors_safe = True

    reference = torch.arange(4, dtype=torch.float32)
    payload = bytearray(reference.numpy().tobytes())
    stream = _TestWeightStream(
        [WeightGroup("model", ("model.weight",))],
        [(_stream_batch(0, 0, 16, complete=True), payload)],
        direct_buffer_enabled=False,
    )
    mapper = BorrowSafeMapper()
    source_ptr = torch.frombuffer(payload, dtype=torch.float32).data_ptr()
    loaded_ptrs = []

    def load_weights(weights, allow_partial_loading=False):
        assert allow_partial_loading
        loaded_ptrs.append(weights["model.weight"].data_ptr())

    loader = object.__new__(ModelLoader)
    loader._load_weight_stream(load_weights, stream, mapper, model=torch.nn.Module())

    assert loaded_ptrs != [source_ptr]
    assert stream.materializations == [(False, 16)]


def test_model_loader_rejects_direct_source_mutation():
    from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader

    class BorrowSafeMapper(_TestIncrementalMapper):
        borrowed_source_tensors_safe = True

    reference = torch.arange(4, dtype=torch.float32)
    stream = _TestWeightStream(
        [WeightGroup("model", ("model.weight",))],
        [(_stream_batch(0, 0, 16, complete=True), bytearray(reference.numpy().tobytes()))],
        direct_buffer_enabled=True,
    )
    mapper = BorrowSafeMapper()

    def load_weights(weights, allow_partial_loading=False):
        assert allow_partial_loading
        weights["model.weight"].add_(1)

    loader = object.__new__(ModelLoader)
    with pytest.raises(RuntimeError, match="borrowed checkpoint tensor.*immutable"):
        loader._load_weight_stream(load_weights, stream, mapper, model=torch.nn.Module())

    assert isinstance(stream.completed[0][1], RuntimeError)
    assert mapper.events[-1] == ("abort",)


def test_model_loader_rejects_retained_direct_source_tensor():
    from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader

    class BorrowSafeMapper(_TestIncrementalMapper):
        borrowed_source_tensors_safe = True

    reference = torch.arange(4, dtype=torch.float32)
    stream = _TestWeightStream(
        [WeightGroup("model", ("model.weight",))],
        [(_stream_batch(0, 0, 16, complete=True), bytearray(reference.numpy().tobytes()))],
        direct_buffer_enabled=True,
    )
    mapper = BorrowSafeMapper()
    retained = []

    def load_weights(weights, allow_partial_loading=False):
        assert allow_partial_loading
        retained.append(weights["model.weight"])

    loader = object.__new__(ModelLoader)
    with pytest.raises(
        BorrowedWeightStorageRetentionError, match="retained borrowed checkpoint tensor"
    ):
        loader._load_weight_stream(load_weights, stream, mapper, model=torch.nn.Module())

    retained.clear()
    assert isinstance(stream.completed[0][1], BorrowedWeightStorageRetentionError)
    assert mapper.events[-1] == ("abort",)


def test_model_loader_reports_materialization_error_through_batch_consensus():
    from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader

    reference = torch.arange(4, dtype=torch.float32)
    stream = _TestWeightStream(
        [WeightGroup("model", ("model.weight",))],
        [(_stream_batch(0, 0, 16, complete=True), reference.numpy().tobytes())],
    )
    mapper = _TestIncrementalMapper()

    def load_weights(_weights, allow_partial_loading=False):
        assert allow_partial_loading
        raise ValueError("materialization failed")

    loader = object.__new__(ModelLoader)
    with pytest.raises(RuntimeError, match="consensus: ValueError"):
        loader._load_weight_stream(load_weights, stream, mapper, model=torch.nn.Module())

    assert len(stream.completed) == 1
    assert isinstance(stream.completed[0][1], ValueError)
    assert mapper.events[-1] == ("abort",)


def test_model_loader_reports_final_validation_through_final_batch_consensus():
    from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader

    class FinalizeFailingMapper(_TestIncrementalMapper):
        def finalize_incremental_load(self):
            super().finalize_incremental_load()
            raise ValueError("manifest validation failed")

    reference = torch.arange(4, dtype=torch.float32)
    stream = _TestWeightStream(
        [WeightGroup("model", ("model.weight",))],
        [(_stream_batch(0, 0, 16, complete=True), reference.numpy().tobytes())],
    )
    mapper = FinalizeFailingMapper()

    def load_weights(_weights, allow_partial_loading=False):
        assert allow_partial_loading

    loader = object.__new__(ModelLoader)
    with pytest.raises(RuntimeError, match="consensus: ValueError"):
        loader._load_weight_stream(load_weights, stream, mapper, model=torch.nn.Module())

    assert isinstance(stream.completed[0][1], ValueError)
    assert mapper.events[-1] == ("abort",)


def test_model_loader_finalizes_partial_modules_once_before_post_load():
    from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader

    events = []

    class RecordingMapper(_TestIncrementalMapper):
        def finalize_incremental_load(self):
            super().finalize_incremental_load()
            events.append("mapper_finalize")

    class Backend(torch.nn.Module):
        def process_weights_after_loading(self):
            events.append("backend_process")

        def post_load_weights(self):
            events.append("backend_post_load")

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backend = Backend()

        def process_weights_after_loading(self):
            events.append("wrapper_process")
            self.backend.process_weights_after_loading()

    class Leaf(torch.nn.Module):
        def process_weights_after_loading(self):
            events.append("leaf_process")

        def post_load_weights(self):
            events.append("leaf_post_load")

    model = torch.nn.Module()
    model.wrapper = Wrapper()
    model.leaf = Leaf()
    reference = torch.arange(4, dtype=torch.float32)
    stream = _TestWeightStream(
        [
            WeightGroup("model.0", ("model.0.weight",)),
            WeightGroup("model.1", ("model.1.weight",)),
        ],
        [
            (
                _stream_batch(
                    0,
                    0,
                    16,
                    complete=True,
                    group_id="model.0",
                    key="model.0.weight",
                ),
                reference.numpy().tobytes(),
            ),
            (
                _stream_batch(
                    1,
                    0,
                    16,
                    complete=True,
                    group_id="model.1",
                    key="model.1.weight",
                ),
                reference.numpy().tobytes(),
            ),
        ],
    )
    mapper = RecordingMapper()

    def load_weights(_weights, allow_partial_loading=False):
        assert allow_partial_loading
        events.append("load")

    loader = object.__new__(ModelLoader)
    loader._load_weight_stream(load_weights, stream, mapper, model=model)
    loader._walk_full_post_load(model)

    assert events[:6] == [
        "load",
        "load",
        "mapper_finalize",
        "wrapper_process",
        "backend_process",
        "leaf_process",
    ]
    assert events.count("backend_process") == 1
    assert events.index("leaf_process") < events.index("leaf_post_load")
    assert mapper.events == [
        ("begin", stream.groups),
        ("record", "model.0"),
        ("record", "model.1"),
        ("finalize",),
    ]


def test_model_loader_reports_processing_error_through_final_batch_consensus():
    from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader

    class FailingModule(torch.nn.Module):
        def process_weights_after_loading(self):
            raise ValueError("deferred weight processing failed")

    model = torch.nn.Module()
    model.failing = FailingModule()
    reference = torch.arange(4, dtype=torch.float32)
    stream = _TestWeightStream(
        [WeightGroup("model", ("model.weight",))],
        [(_stream_batch(0, 0, 16, complete=True), reference.numpy().tobytes())],
    )
    mapper = _TestIncrementalMapper()

    def load_weights(_weights, allow_partial_loading=False):
        assert allow_partial_loading

    loader = object.__new__(ModelLoader)
    with pytest.raises(RuntimeError, match="consensus: ValueError"):
        loader._load_weight_stream(load_weights, stream, mapper, model=model)

    assert isinstance(stream.completed[0][1], ValueError)
    assert mapper.events[-2:] == [("finalize",), ("abort",)]


def test_model_loader_reports_mapper_begin_error_through_start_consensus():
    from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader

    class BeginFailingMapper(_TestIncrementalMapper):
        def begin_incremental_load(self, groups):
            super().begin_incremental_load(groups)
            raise ValueError("mapper initialization failed")

    stream = _TestWeightStream([WeightGroup("model", ("model.weight",))], [])
    mapper = BeginFailingMapper()
    loader = object.__new__(ModelLoader)

    with pytest.raises(RuntimeError, match="start consensus: ValueError"):
        loader._load_weight_stream(lambda _weights: None, stream, mapper, model=torch.nn.Module())

    assert isinstance(stream.started[0], ValueError)
    assert mapper.events[-1] == ("abort",)


def test_partial_loading_capability_accepts_keyword_only_argument():
    from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader

    def load_weights(_weights, *, allow_partial_loading=False):
        del allow_partial_loading

    assert ModelLoader._supports_partial_weight_loading(load_weights)


@pytest.mark.parametrize(
    "method_name, expected",
    [
        ("UnquantizedLinearMethod", True),
        ("FP8QDQLinearMethod", True),
        ("FP8BlockScalesLinearMethod", True),
        ("NVFP4LinearMethod", True),
        ("WeightOnlyQuantLinearMethod", False),
    ],
)
def test_linear_partial_weight_loading_capability_matches_runtime_guard(method_name, expected):
    from tensorrt_llm._torch.modules import linear as linear_module

    linear = linear_module.Linear.__new__(linear_module.Linear)
    torch.nn.Module.__init__(linear)
    linear._weights_created = True
    linear.quant_method = getattr(linear_module, method_name)()

    assert linear.supports_partial_weight_loading is expected


def test_fused_moe_partial_weight_loading_capability_matrix():
    from tensorrt_llm._torch.modules.fused_moe.quantization import (
        DeepSeekFP8BlockScalesFusedMoEMethod,
        FP8QDQFusedMoEMethod,
        FusedMoEMethodBase,
        NVFP4FusedMoEMethod,
        UnquantizedFusedMoEMethod,
        W4A8MXFP4MXFP8MegaMoEDeepGemmMethod,
    )

    assert UnquantizedFusedMoEMethod.supports_partial_weight_loading
    assert FP8QDQFusedMoEMethod.supports_partial_weight_loading
    assert DeepSeekFP8BlockScalesFusedMoEMethod.supports_partial_weight_loading
    assert NVFP4FusedMoEMethod.supports_partial_weight_loading
    assert not FusedMoEMethodBase.supports_partial_weight_loading
    assert not W4A8MXFP4MXFP8MegaMoEDeepGemmMethod.supports_partial_weight_loading


def test_moe_modules_publish_or_delegate_partial_weight_loading_capability():
    from tensorrt_llm._torch.modules.fused_moe.configurable_moe import ConfigurableMoE
    from tensorrt_llm._torch.modules.fused_moe.fused_moe_triton import TritonFusedMoE
    from tensorrt_llm._torch.modules.fused_moe.fused_moe_vanilla import VanillaMoE
    from tensorrt_llm._torch.modules.fused_moe.mega_moe import MegaMoEDeepGemm
    from tensorrt_llm._torch.modules.fused_moe.quantization import (
        W4A8MXFP4MXFP8MegaMoEDeepGemmMethod,
    )

    backend = torch.nn.Module()
    backend.supports_partial_weight_loading = True
    configurable_moe = ConfigurableMoE.__new__(ConfigurableMoE)
    torch.nn.Module.__init__(configurable_moe)
    configurable_moe.backend = backend

    assert configurable_moe.supports_partial_weight_loading
    backend.supports_partial_weight_loading = False
    assert not configurable_moe.supports_partial_weight_loading
    assert not VanillaMoE.__new__(VanillaMoE).supports_partial_weight_loading
    assert not TritonFusedMoE.__new__(TritonFusedMoE).supports_partial_weight_loading

    mega_deep_gemm = MegaMoEDeepGemm.__new__(MegaMoEDeepGemm)
    torch.nn.Module.__init__(mega_deep_gemm)
    mega_deep_gemm.quant_method = W4A8MXFP4MXFP8MegaMoEDeepGemmMethod()
    assert not mega_deep_gemm.supports_partial_weight_loading


def test_direct_rank_read_uses_mmap_path_without_full_prefetch(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    reference = {
        "model.layers.0.self_attn.q_proj.weight": torch.arange(32).reshape(8, 4),
        "model.norm.weight": torch.arange(4),
    }
    save_file(reference, str(checkpoint))

    loader = HfWeightLoader(weight_load_plan="direct_rank_read")
    with (
        mock.patch.object(loader, "_get_cooperative_prefetch_policy", return_value=(0, False)),
        mock.patch.object(loader, "prefetch_file_chunks") as chunk_prefetch,
    ):
        weights = loader.load_weights(str(tmp_path), mapping=Mapping(), model=_model())

    chunk_prefetch.assert_not_called()
    loaded_weight = weights["model.layers.0.self_attn.q_proj.weight"]
    assert isinstance(loaded_weight, torch.Tensor)
    torch.testing.assert_close(loaded_weight, reference["model.layers.0.self_attn.q_proj.weight"])


def test_direct_rank_read_rejects_duplicate_keys_before_consumption(tmp_path):
    save_file({"duplicate.weight": torch.ones(2)}, str(tmp_path / "a.safetensors"))
    save_file({"duplicate.weight": torch.zeros(2)}, str(tmp_path / "b.safetensors"))

    loader = HfWeightLoader(weight_load_plan="direct_rank_read")
    with (
        mock.patch.object(loader, "_get_cooperative_prefetch_policy", return_value=(0, False)),
        pytest.raises(RuntimeError, match="Duplicate SafeTensors key"),
    ):
        loader.load_weights(str(tmp_path), mapping=Mapping(), model=_model())


@pytest.mark.parametrize(
    "model,mapping",
    [
        (
            _model(
                "Qwen3_5MoeForCausalLM",
                module_name="tensorrt_llm._torch.models.modeling_qwen3_5",
                quant_algo="FP8",
                quant_config_dict={"experts": "NVFP4"},
            ),
            Mapping(),
        ),
        (
            _model(
                "DeepseekV4ForCausalLM",
                module_name="tensorrt_llm._torch.models.modeling_deepseekv4",
                spec_config=object(),
            ),
            Mapping(
                world_size=8,
                tp_size=4,
                pp_size=2,
                moe_tp_size=1,
                moe_ep_size=4,
                enable_attention_dp=True,
            ),
        ),
        (
            _model(
                "Llama4ForConditionalGeneration",
                module_name="tensorrt_llm._torch.models.modeling_llama",
                lora_config=object(),
                force_dynamic_quantization=True,
            ),
            Mapping(world_size=4, tp_size=2, cp_size=2),
        ),
    ],
)
def test_cooperative_read_ahead_accepts_flagship_quant_parallel_and_mtp_configs(
    model, mapping, monkeypatch
):
    monkeypatch.setattr(weight_loader_module, "mpi_disabled", lambda: False)

    reason = HfWeightLoader._cooperative_ineligibility_reason(
        model,
        mapping,
        checkpoint_format="HF",
        uses_custom_weight_mapper=True,
        load_format="AUTO",
    )

    assert reason is None


def test_cooperative_read_ahead_is_not_gated_by_model_class():
    reason = HfWeightLoader._cooperative_ineligibility_reason(
        _model("FutureForCausalLM"),
        Mapping(),
        checkpoint_format="HF",
        uses_custom_weight_mapper=False,
    )

    assert reason is None


def test_distributed_non_mpi_loader_is_not_eligible(monkeypatch):
    monkeypatch.setattr(weight_loader_module, "mpi_disabled", lambda: True)

    reason = HfWeightLoader._cooperative_ineligibility_reason(
        _model(),
        Mapping(world_size=2, tp_size=2),
        checkpoint_format="HF",
        uses_custom_weight_mapper=False,
    )

    assert reason == "distributed cooperative loading requires MPI-launched ranks"


def test_distributed_direct_weight_loader_requires_explicit_auto_format():
    reason = HfWeightLoader._cooperative_coordination_error(
        Mapping(world_size=2, tp_size=2),
        "HF",
        None,
    )

    assert reason == ("distributed cooperative loading requires an explicit AUTO load format")


def test_direct_rank_chunk_plan_is_disjoint_and_complete_across_local_ranks(tmp_path):
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    first.write_bytes(b"a" * 10)
    second.write_bytes(b"b" * 5)
    loader = HfWeightLoader(weight_load_plan="direct_rank_read", prefetch_chunk_size=4)

    rank_chunks = []
    for rank in range(3):
        with mock.patch.object(loader, "_get_local_rank_and_size", return_value=(rank, 3)):
            rank_chunks.append(
                loader._local_prefetch_chunks(
                    [str(second), str(first)], WeightLoadPolicy.DIRECT_RANK_READ
                )
            )

    flattened = [chunk for chunks in rank_chunks for chunk in chunks]
    assert len(flattened) == len(set(flattened))
    assert sorted(flattened) == [
        (str(first), 0, 4),
        (str(first), 4, 4),
        (str(first), 8, 2),
        (str(second), 0, 4),
        (str(second), 4, 1),
    ]


def test_direct_rank_prefetch_reads_only_assigned_chunks(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"x" * 13)
    loader = HfWeightLoader(
        weight_load_plan="direct_rank_read",
        prefetch_chunk_size=4,
        prefetch_workers_per_rank=2,
    )

    with (
        mock.patch.object(loader, "_get_local_rank_and_size", return_value=(1, 2)),
        mock.patch.object(loader, "_prefetch_one_chunk") as prefetch,
    ):
        loader.prefetch_file_chunks([str(checkpoint)], WeightLoadPolicy.DIRECT_RANK_READ)

    assert sorted(call.args for call in prefetch.call_args_list) == [
        (str(checkpoint), 4, 4),
        (str(checkpoint), 12, 1),
    ]


def test_single_producer_page_cache_prefetch_assigns_chunks_to_rank_zero(
    tmp_path,
):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"x" * 10)
    loader = HfWeightLoader(
        weight_load_plan="single_producer_page_cache_prefetch",
        prefetch_chunk_size=4,
    )

    with mock.patch.object(loader, "_get_local_rank_and_size", return_value=(0, 3)):
        producer_chunks = loader._local_prefetch_chunks(
            [str(checkpoint)],
            WeightLoadPolicy.SINGLE_PRODUCER_PAGE_CACHE_PREFETCH,
        )
    with mock.patch.object(loader, "_get_local_rank_and_size", return_value=(1, 3)):
        consumer_chunks = loader._local_prefetch_chunks(
            [str(checkpoint)],
            WeightLoadPolicy.SINGLE_PRODUCER_PAGE_CACHE_PREFETCH,
        )

    assert producer_chunks == [
        (str(checkpoint), 0, 4),
        (str(checkpoint), 4, 4),
        (str(checkpoint), 8, 2),
    ]
    assert consumer_chunks == []


def test_single_producer_page_cache_prefetch_then_mmaps(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    reference = {"model.norm.weight": torch.arange(4)}
    save_file(reference, str(checkpoint))
    loader = HfWeightLoader(weight_load_plan="single_producer_page_cache_prefetch")

    with mock.patch.object(
        loader, "prefetch_file_chunks", wraps=loader.prefetch_file_chunks
    ) as prefetch:
        weights = loader.load_weights(str(tmp_path), mapping=Mapping(), model=_model())

    prefetch.assert_called_once_with(
        [str(checkpoint)],
        WeightLoadPolicy.SINGLE_PRODUCER_PAGE_CACHE_PREFETCH,
        None,
    )
    torch.testing.assert_close(weights["model.norm.weight"], reference["model.norm.weight"])


def test_strict_shared_host_fails_preflight_without_safe_mapper_manifest(tmp_path, monkeypatch):
    _enable_single_rank_mpi(monkeypatch)
    checkpoint = tmp_path / "model.safetensors"
    save_file({"model.norm.weight": torch.arange(4)}, str(checkpoint))

    class UnsupportedMapper:
        single_tensor_groups_safe = False

        @staticmethod
        def get_weight_groups(_keys):
            return None

    loader = HfWeightLoader(weight_load_plan="shared_host_producer")
    with (
        mock.patch.object(
            shared_host_stream_module, "open_shared_host_weight_stream"
        ) as open_stream,
        pytest.raises(RuntimeError, match="did not declare a safe incremental"),
    ):
        with loader.open_weight_session(
            str(tmp_path),
            Mapping(),
            model=_model(),
            _weight_mapper=UnsupportedMapper(),
            _model_supports_partial_loading=True,
        ):
            pytest.fail("strict preflight must fail before session entry")

    open_stream.assert_not_called()


def test_strict_shared_host_requires_mpi_before_header_preflight(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    save_file({"model.norm.weight": torch.arange(4)}, str(checkpoint))
    loader = HfWeightLoader(weight_load_plan="shared_host_producer")

    with (
        mock.patch.object(
            shared_host_stream_module, "prepare_shared_host_weight_stream"
        ) as prepare_stream,
        pytest.raises(RuntimeError, match="requires active MPI"),
    ):
        with loader.open_weight_session(
            str(tmp_path),
            Mapping(),
            model=_model(),
            _weight_mapper=mock.Mock(),
            _model_supports_partial_loading=True,
        ):
            pytest.fail("strict preflight must fail before session entry")

    prepare_stream.assert_not_called()


@pytest.mark.parametrize("policy", ("shared_host_producer", "rank_cooperative_stream"))
def test_strict_stream_policy_rejects_synchronous_load_weights(tmp_path, monkeypatch, policy):
    _enable_single_rank_mpi(monkeypatch)
    save_file(
        {"model.norm.weight": torch.arange(4)},
        str(tmp_path / "model.safetensors"),
    )
    loader = HfWeightLoader(weight_load_plan=policy)

    with (
        mock.patch.object(
            shared_host_stream_module, "prepare_shared_host_weight_stream"
        ) as prepare_stream,
        pytest.raises(RuntimeError, match="requires an open weight session"),
    ):
        loader.load_weights(
            str(tmp_path),
            Mapping(),
            model=_model(),
            _weight_mapper=mock.Mock(),
            _model_supports_partial_loading=True,
        )

    prepare_stream.assert_not_called()


@pytest.mark.parametrize(
    "model, expected_reason",
    [
        (
            _model(
                "Qwen3_5MoeForCausalLM",
                moe_load_balancer=SimpleNamespace(layer_updates_per_iter=1),
            ),
            "does not support dynamic MoE load balancing",
        ),
        (
            _model("Llama4ForConditionalGeneration", enable_min_latency=True),
            "does not support min-latency model loading",
        ),
    ],
)
def test_strict_shared_host_rejects_unbounded_or_eager_partial_profiles(
    tmp_path, monkeypatch, model, expected_reason
):
    _enable_single_rank_mpi(monkeypatch)
    checkpoint = tmp_path / "model.safetensors"
    save_file({"model.norm.weight": torch.arange(4)}, str(checkpoint))

    loader = HfWeightLoader(weight_load_plan="shared_host_producer")
    with (
        mock.patch.object(
            shared_host_stream_module, "prepare_shared_host_weight_stream"
        ) as prepare_stream,
        pytest.raises(RuntimeError, match=expected_reason),
    ):
        with loader.open_weight_session(
            str(tmp_path),
            Mapping(),
            model=model,
            _weight_mapper=mock.Mock(),
            _model_supports_partial_loading=True,
        ):
            pytest.fail("unsafe partial-load profile must fail before preflight")

    prepare_stream.assert_not_called()


def test_nested_partial_load_capability_ignores_unknown_modules():
    model = torch.nn.Module()
    model.unknown = torch.nn.Module()

    assert HfWeightLoader._nested_partial_load_ineligibility_reason(model) is None


def test_nested_partial_load_capability_failure_returns_ineligibility_reason():
    class BrokenCapabilityModule(torch.nn.Module):
        @property
        def supports_partial_weight_loading(self):
            raise RuntimeError("capability unavailable")

    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.broken_projection = BrokenCapabilityModule()

    reason = HfWeightLoader._nested_partial_load_ineligibility_reason(model)

    assert "decoder.broken_projection" in reason
    assert "RuntimeError: capability unavailable" in reason


@pytest.mark.parametrize("getter_raises", [False, True], ids=["unsupported", "check-failed"])
def test_ordered_shared_host_plan_falls_back_before_nested_module_preflight_io(
    tmp_path, monkeypatch, getter_raises
):
    class UnsupportedModule(torch.nn.Module):
        @property
        def supports_partial_weight_loading(self):
            if getter_raises:
                raise RuntimeError("capability unavailable")
            return False

    _enable_single_rank_mpi(monkeypatch)
    checkpoint = tmp_path / "model.safetensors"
    save_file({"model.norm.weight": torch.arange(4)}, str(checkpoint))
    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.unsupported_projection = UnsupportedModule()
    expected_weights = {"model.norm.weight": object()}
    loader = HfWeightLoader(weight_load_plan=("shared_host_producer", "legacy_fallback"))

    with (
        mock.patch.object(
            shared_host_stream_module, "prepare_shared_host_weight_stream"
        ) as prepare_stream,
        mock.patch.object(
            shared_host_stream_module, "open_shared_host_weight_stream"
        ) as open_stream,
        mock.patch.object(loader, "_load_legacy_safetensors", return_value=expected_weights),
    ):
        with loader.open_weight_session(
            str(tmp_path),
            Mapping(),
            model=model,
            _weight_mapper=mock.Mock(),
            _model_supports_partial_loading=True,
        ) as weights:
            assert weights is expected_weights

    reason = HfWeightLoader._nested_partial_load_ineligibility_reason(model)
    assert "decoder.unsupported_projection" in reason
    if getter_raises:
        assert "RuntimeError: capability unavailable" in reason
    prepare_stream.assert_not_called()
    open_stream.assert_not_called()


def test_ordered_shared_host_plan_falls_back_before_transport_allocation(tmp_path, monkeypatch):
    _enable_single_rank_mpi(monkeypatch)
    checkpoint = tmp_path / "model.safetensors"
    save_file({"model.norm.weight": torch.arange(4)}, str(checkpoint))

    class UnsupportedMapper:
        single_tensor_groups_safe = False

        @staticmethod
        def get_weight_groups(_keys):
            return None

    expected_weights = {"model.norm.weight": object()}
    loader = HfWeightLoader(weight_load_plan=("shared_host_producer", "legacy_fallback"))
    with (
        mock.patch.object(
            loader, "_load_legacy_safetensors", return_value=expected_weights
        ) as legacy_load,
        mock.patch.object(
            shared_host_stream_module, "open_shared_host_weight_stream"
        ) as open_stream,
    ):
        with loader.open_weight_session(
            str(tmp_path),
            Mapping(),
            model=_model(),
            _weight_mapper=UnsupportedMapper(),
            _model_supports_partial_loading=True,
        ) as weights:
            assert weights is expected_weights

    open_stream.assert_not_called()
    legacy_load.assert_called_once()


@pytest.mark.parametrize(
    "policy,producer_mode_name",
    [
        ("shared_host_producer", "SINGLE_PRODUCER"),
        ("rank_cooperative_stream", "RANK_COOPERATIVE"),
    ],
)
def test_shared_host_session_routes_selected_producer_mode(
    tmp_path, monkeypatch, policy, producer_mode_name
):
    _enable_single_rank_mpi(monkeypatch)
    checkpoint = tmp_path / "model.safetensors"
    save_file({"model.norm.weight": torch.arange(4)}, str(checkpoint))

    class ManifestMapper:
        single_tensor_groups_safe = False

        @staticmethod
        def get_weight_groups(keys):
            assert tuple(keys) == ("model.norm.weight",)
            return [WeightGroup("model.norm", ("model.norm.weight",))]

    expected_stream = _TestWeightStream([WeightGroup("model.norm", ("model.norm.weight",))], [])
    loader = HfWeightLoader(weight_load_plan=policy)
    with mock.patch.object(
        shared_host_stream_module,
        "open_shared_host_weight_stream",
        return_value=expected_stream,
    ) as open_stream:
        with loader.open_weight_session(
            str(tmp_path),
            Mapping(),
            model=_model(),
            _weight_mapper=ManifestMapper(),
            _model_supports_partial_loading=True,
        ) as weights:
            assert weights is expected_stream

    assert expected_stream.finalized
    assert open_stream.call_args.kwargs["group_manifest"] == expected_stream.groups
    assert open_stream.call_args.kwargs["buffer_budget_bytes"] == 64 * 1024 * 1024 * 1024
    assert open_stream.call_args.kwargs["producer_mode"] is getattr(
        shared_host_stream_module.SharedHostProducerMode, producer_mode_name
    )


def test_rank_cooperative_stream_can_fallback_to_single_producer_stream(tmp_path, monkeypatch):
    _enable_single_rank_mpi(monkeypatch)
    checkpoint = tmp_path / "model.safetensors"
    save_file({"model.norm.weight": torch.arange(4)}, str(checkpoint))

    class ManifestMapper:
        @staticmethod
        def get_weight_groups(keys):
            return [WeightGroup("model.norm", tuple(keys))]

    expected_stream = _TestWeightStream([WeightGroup("model.norm", ("model.norm.weight",))], [])
    loader = HfWeightLoader(
        weight_load_plan=(
            "rank_cooperative_stream",
            "shared_host_producer",
            "legacy_fallback",
        )
    )
    with mock.patch.object(
        shared_host_stream_module,
        "open_shared_host_weight_stream",
        side_effect=(None, expected_stream),
    ) as open_stream:
        with loader.open_weight_session(
            str(tmp_path),
            Mapping(),
            model=_model(),
            _weight_mapper=ManifestMapper(),
            _model_supports_partial_loading=True,
        ) as weights:
            assert weights is expected_stream

    producer_modes = [call.kwargs["producer_mode"] for call in open_stream.call_args_list]
    assert producer_modes == [
        shared_host_stream_module.SharedHostProducerMode.RANK_COOPERATIVE,
        shared_host_stream_module.SharedHostProducerMode.SINGLE_PRODUCER,
    ]


def test_direct_session_yields_before_read_ahead_finishes_and_exit_joins(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"x")
    loader = HfWeightLoader(weight_load_plan="direct_rank_read", prefetch_chunk_size=1)
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    expected_weights = {"weight": object()}

    def blocked_read(*_args):
        started.set()
        assert release.wait(timeout=5)
        finished.set()

    with (
        mock.patch.object(loader, "_get_cooperative_prefetch_policy", return_value=(1, True)),
        mock.patch.object(loader, "_prefetch_one_chunk", side_effect=blocked_read),
        mock.patch.object(loader, "_load_weights_in_parallel", return_value=expected_weights),
    ):
        with loader.open_weight_session(str(tmp_path), Mapping(), model=_model()) as weights:
            assert weights is expected_weights
            assert started.wait(timeout=5)
            assert not finished.is_set()
            release.set()

    assert finished.is_set()


def test_public_direct_load_remains_synchronous(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"x")
    loader = HfWeightLoader(weight_load_plan="direct_rank_read", prefetch_chunk_size=1)
    started = threading.Event()
    release = threading.Event()
    returned = threading.Event()
    errors = []

    def blocked_read(*_args):
        started.set()
        assert release.wait(timeout=5)

    def load():
        try:
            loader.load_weights(str(tmp_path), Mapping(), model=_model())
        except Exception as error:
            errors.append(error)
        finally:
            returned.set()

    with (
        mock.patch.object(loader, "_get_cooperative_prefetch_policy", return_value=(1, True)),
        mock.patch.object(loader, "_prefetch_one_chunk", side_effect=blocked_read),
        mock.patch.object(loader, "_load_weights_in_parallel", return_value={}),
    ):
        load_thread = threading.Thread(target=load)
        load_thread.start()
        assert started.wait(timeout=5)
        assert not returned.is_set()
        release.set()
        load_thread.join(timeout=5)

    assert not load_thread.is_alive()
    assert not errors


def test_direct_session_has_no_pre_consumption_node_barrier(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"x")
    loader = HfWeightLoader(weight_load_plan="direct_rank_read", prefetch_chunk_size=1)
    node_communicator = mock.Mock()
    node_communicator.Get_rank.return_value = 0
    node_communicator.Get_size.return_value = 1
    node_communicator.allreduce.side_effect = lambda value, op: value
    started = threading.Event()
    release = threading.Event()

    def blocked_read(*_args):
        started.set()
        assert release.wait(timeout=5)

    with (
        mock.patch.object(
            loader,
            "_get_active_node_communicator",
            return_value=node_communicator,
        ),
        mock.patch.object(loader, "_get_cooperative_prefetch_policy", return_value=(1, True)),
        mock.patch.object(loader, "_prefetch_one_chunk", side_effect=blocked_read),
        mock.patch.object(loader, "_load_weights_in_parallel", return_value={}),
    ):
        with loader.open_weight_session(str(tmp_path), Mapping(), model=_model()):
            assert started.wait(timeout=5)
            node_communicator.Barrier.assert_not_called()
            release.set()

    node_communicator.Barrier.assert_called_once_with()
    node_communicator.Free.assert_called_once_with()


def test_single_producer_page_cache_session_stays_synchronous(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"x")
    loader = HfWeightLoader(weight_load_plan="single_producer_page_cache_prefetch")
    prefetch_started = threading.Event()
    release = threading.Event()
    body_entered = threading.Event()
    errors = []

    def blocked_prefetch(*_args):
        prefetch_started.set()
        assert release.wait(timeout=5)

    def load():
        try:
            with loader.open_weight_session(str(tmp_path), Mapping(), model=_model()):
                body_entered.set()
        except Exception as error:
            errors.append(error)

    with (
        mock.patch.object(loader, "_get_cooperative_prefetch_policy", return_value=(1, True)),
        mock.patch.object(loader, "prefetch_file_chunks", side_effect=blocked_prefetch),
        mock.patch.object(loader, "_load_weights_in_parallel", return_value={}),
    ):
        load_thread = threading.Thread(target=load)
        load_thread.start()
        assert prefetch_started.wait(timeout=5)
        assert not body_entered.is_set()
        release.set()
        load_thread.join(timeout=5)

    assert not load_thread.is_alive()
    assert body_entered.is_set()
    assert not errors


def test_direct_session_preserves_model_body_exception_during_cleanup(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"x")
    loader = HfWeightLoader(weight_load_plan="direct_rank_read", prefetch_chunk_size=1)

    with (
        mock.patch.object(loader, "_get_cooperative_prefetch_policy", return_value=(1, True)),
        mock.patch.object(loader, "_prefetch_one_chunk", side_effect=OSError("read failed")),
        mock.patch.object(loader, "_load_weights_in_parallel", return_value={}),
        pytest.raises(ValueError, match="materialization failed"),
    ):
        with loader.open_weight_session(str(tmp_path), Mapping(), model=_model()):
            raise ValueError("materialization failed")


def test_direct_session_cancels_read_ahead_on_model_body_failure(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"x")
    loader = HfWeightLoader(weight_load_plan="direct_rank_read", prefetch_chunk_size=1)
    started = threading.Event()
    cancellation_observed = threading.Event()

    def cancellable_read(_file, _offset, _length, cancel_event):
        started.set()
        assert cancel_event.wait(timeout=5)
        cancellation_observed.set()

    with (
        mock.patch.object(loader, "_get_cooperative_prefetch_policy", return_value=(1, True)),
        mock.patch.object(loader, "_prefetch_one_chunk", side_effect=cancellable_read),
        mock.patch.object(loader, "_load_weights_in_parallel", return_value={}),
        pytest.raises(ValueError, match="materialization failed"),
    ):
        with loader.open_weight_session(str(tmp_path), Mapping(), model=_model()):
            assert started.wait(timeout=5)
            raise ValueError("materialization failed")

    assert cancellation_observed.is_set()


def test_healthy_rank_raises_coordinated_peer_materialization_error(monkeypatch):
    class FakeWorldCommunicator:
        calls = 0

        @classmethod
        def allgather(cls, error_message):
            cls.calls += 1
            if cls.calls == 1:
                assert error_message is None
                return [None, "ValueError: peer materialization failed"]
            assert error_message is None
            return [None, None]

    monkeypatch.setattr(weight_loader_module, "ENABLE_MULTI_DEVICE", True)
    monkeypatch.setattr(weight_loader_module, "mpi_disabled", lambda: False)
    monkeypatch.setattr(weight_loader_module, "mpi_comm", lambda: FakeWorldCommunicator())
    loader = HfWeightLoader(weight_load_plan="direct_rank_read")
    session = weight_loader_module._DirectReadAheadSession(
        loader,
        node_communicator=None,
        local_chunks=[],
        max_workers=0,
        local_rank=0,
        enabled=False,
    )
    session.start()

    class JoinProbe:
        joined = False

        def join(self):
            assert session._cancel_event.is_set()
            self.joined = True

    join_probe = JoinProbe()
    session._thread = join_probe

    with pytest.raises(
        RuntimeError,
        match="Rank 1 failed during direct_rank_read model materialization",
    ):
        session.finish()

    assert FakeWorldCommunicator.calls == 2
    assert join_probe.joined


def test_direct_start_failure_is_coordinated_before_mmap(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"x")
    loader = HfWeightLoader(weight_load_plan="direct_rank_read", prefetch_chunk_size=1)
    node_communicator = mock.Mock()
    node_communicator.Get_rank.return_value = 0
    node_communicator.Get_size.return_value = 1
    node_communicator.allreduce.side_effect = lambda value, op: value

    with (
        mock.patch.object(
            loader,
            "_get_active_node_communicator",
            return_value=node_communicator,
        ),
        mock.patch.object(loader, "_get_cooperative_prefetch_policy", return_value=(1, True)),
        mock.patch.object(threading.Thread, "start", side_effect=RuntimeError("cannot start")),
        mock.patch.object(loader, "_load_weights_in_parallel") as mmap_weights,
        mock.patch.object(
            loader, "_raise_on_rank_error", wraps=loader._raise_on_rank_error
        ) as rank_consensus,
        pytest.raises(RuntimeError, match="read-ahead start"),
    ):
        with loader.open_weight_session(str(tmp_path), Mapping(), model=_model()):
            pytest.fail("the session body must not run")

    phases = [call.args[0] for call in rank_consensus.call_args_list]
    assert phases[:2] == [
        "direct_rank_read read-ahead planning",
        "direct_rank_read read-ahead start",
    ]
    assert "direct_rank_read SafeTensors mmap setup" not in phases
    mmap_weights.assert_not_called()
    node_communicator.Free.assert_called_once_with()


def test_default_weight_load_plan_matches_policy_order(monkeypatch):
    monkeypatch.delenv("TRTLLM_HF_WEIGHT_LOAD_PLAN", raising=False)

    assert DEFAULT_WEIGHT_LOAD_PLAN == (
        WeightLoadPolicy.DIRECT_RANK_READ,
        WeightLoadPolicy.SHARED_HOST_PRODUCER,
        WeightLoadPolicy.GPU_BROADCAST,
        WeightLoadPolicy.LEGACY_FALLBACK,
    )
    assert WeightLoadPolicy.RANK_COOPERATIVE_STREAM not in DEFAULT_WEIGHT_LOAD_PLAN
    assert HfWeightLoader()._get_weight_load_plan() == DEFAULT_WEIGHT_LOAD_PLAN


@pytest.mark.parametrize(
    "plan,expected",
    [
        ("direct_rank_read", False),
        ("legacy_fallback", False),
        ("shared_host_producer", True),
        ("rank_cooperative_stream", True),
        (("gpu_broadcast", "shared_host_producer", "legacy_fallback"), True),
        (("gpu_broadcast", "rank_cooperative_stream", "legacy_fallback"), True),
    ],
)
def test_only_shared_first_plan_requires_mapper_before_session(plan, expected):
    checkpoint_loader = HfCheckpointLoader(weight_loader=HfWeightLoader(weight_load_plan=plan))

    assert checkpoint_loader.requires_initialized_mapper_for_session() is expected


def test_implicit_plan_preserves_explicit_raw_weight_cache(monkeypatch):
    monkeypatch.setenv("TRTLLM_HF_WEIGHT_CACHE", "1")

    assert HfWeightLoader()._get_weight_load_plan() == (WeightLoadPolicy.LEGACY_FALLBACK,)
    assert HfWeightLoader(weight_load_plan="direct_rank_read")._get_weight_load_plan() == (
        WeightLoadPolicy.DIRECT_RANK_READ,
    )


def test_default_plan_selects_direct_rank_read_for_qualified_model(tmp_path, monkeypatch):
    monkeypatch.delenv("TRTLLM_HF_WEIGHT_LOAD_PLAN", raising=False)
    (tmp_path / "model.safetensors").touch()
    expected_weights = {"weight": object()}
    loader = HfWeightLoader()

    with mock.patch.object(
        loader, "_load_cooperative_policy", return_value=expected_weights
    ) as cooperative_load:
        weights = loader.load_weights(
            str(tmp_path), mapping=Mapping(), model=_model(), _load_format="AUTO"
        )

    assert weights is expected_weights
    cooperative_load.assert_called_once_with(
        [str(tmp_path / "model.safetensors")], WeightLoadPolicy.DIRECT_RANK_READ
    )


def test_weight_load_plan_can_be_selected_from_environment(monkeypatch):
    monkeypatch.setenv(
        "TRTLLM_HF_WEIGHT_LOAD_PLAN",
        "rank_cooperative_stream,shared_host_producer,legacy_fallback",
    )

    assert HfWeightLoader()._get_weight_load_plan() == (
        WeightLoadPolicy.RANK_COOPERATIVE_STREAM,
        WeightLoadPolicy.SHARED_HOST_PRODUCER,
        WeightLoadPolicy.LEGACY_FALLBACK,
    )


def test_rank_cooperative_stream_cannot_be_repeated_in_ordered_plan():
    with pytest.raises(ValueError, match="must not contain duplicate policies"):
        HfWeightLoader(
            weight_load_plan=(
                "rank_cooperative_stream",
                "shared_host_producer",
                "rank_cooperative_stream",
            )
        )


def test_shared_host_buffer_budget_can_be_selected_from_environment(monkeypatch):
    monkeypatch.setenv(
        "TRTLLM_HF_SHARED_HOST_BUFFER_BUDGET_BYTES",
        str(2 * 1024 * 1024 * 1024),
    )

    assert HfWeightLoader()._get_shared_host_buffer_budget() == 2 * 1024 * 1024 * 1024


def test_gpu_broadcast_is_explicitly_unavailable(tmp_path):
    (tmp_path / "model.safetensors").touch()
    loader = HfWeightLoader(weight_load_plan="gpu_broadcast")

    with pytest.raises(RuntimeError, match="gpu_broadcast is not implemented"):
        loader.load_weights(str(tmp_path), mapping=Mapping(), model=_model())


def test_gpu_broadcast_can_preflight_fallback_to_legacy(tmp_path):
    (tmp_path / "model.safetensors").touch()
    expected_weights = {"weight": object()}
    loader = HfWeightLoader(weight_load_plan=("gpu_broadcast", "legacy_fallback"))

    with (
        mock.patch.object(
            loader, "_prefetch_and_load", return_value=expected_weights
        ) as legacy_load,
        mock.patch.object(logger, "info"),
    ):
        weights = loader.load_weights(str(tmp_path), mapping=Mapping(), model=_model())

    assert weights is expected_weights
    legacy_load.assert_called_once()


@pytest.mark.parametrize("suffix", ("bin", "pth"))
def test_non_safetensors_checkpoint_requires_legacy_policy(tmp_path, suffix):
    (tmp_path / f"model.{suffix}").touch()
    loader = HfWeightLoader(weight_load_plan="direct_rank_read")

    with pytest.raises(RuntimeError, match="require legacy_fallback"):
        loader.load_weights(str(tmp_path), mapping=Mapping(), model=_model())


def test_mismatched_rank_weight_load_plans_fail_before_loading(monkeypatch):
    class FakeWorldCommunicator:
        @staticmethod
        def Get_size():
            return 2

        @staticmethod
        def allgather(selection):
            assert selection == (("direct_rank_read",), "AUTO")
            return [
                (("direct_rank_read",), "AUTO"),
                (("legacy_fallback",), "AUTO"),
            ]

    monkeypatch.setattr(weight_loader_module, "ENABLE_MULTI_DEVICE", True)
    monkeypatch.setattr(weight_loader_module, "mpi_disabled", lambda: False)
    monkeypatch.setattr(weight_loader_module, "mpi_comm", lambda: FakeWorldCommunicator())

    loader = HfWeightLoader(weight_load_plan="direct_rank_read")
    with pytest.raises(RuntimeError, match="must match across all MPI ranks"):
        loader._get_coordinated_weight_load_plan(Mapping(world_size=2, tp_size=2), "HF", "AUTO")


def test_rank_divergent_checkpoint_discovery_fails_before_loading(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()

    class FakeWorldCommunicator:
        @staticmethod
        def Get_size():
            return 2

        @staticmethod
        def allgather(signature):
            return [signature, ("missing", ())]

    monkeypatch.setattr(weight_loader_module, "ENABLE_MULTI_DEVICE", True)
    monkeypatch.setattr(weight_loader_module, "mpi_disabled", lambda: False)
    monkeypatch.setattr(weight_loader_module, "mpi_comm", lambda: FakeWorldCommunicator())

    with pytest.raises(RuntimeError, match="file discovery must match"):
        HfWeightLoader._coordinate_checkpoint_discovery(
            [str(checkpoint)],
            "safetensors",
            Mapping(world_size=2, tp_size=2),
            "HF",
            "AUTO",
        )


def test_active_communicator_must_match_mapping(monkeypatch):
    communicator = mock.Mock()
    communicator.Get_size.return_value = 4
    monkeypatch.setattr(weight_loader_module, "ENABLE_MULTI_DEVICE", True)
    monkeypatch.setattr(weight_loader_module, "mpi_disabled", lambda: False)
    monkeypatch.setattr(weight_loader_module, "mpi_comm", lambda: communicator)

    loader = HfWeightLoader(weight_load_plan="direct_rank_read")
    plan, reason = loader._get_coordinated_weight_load_plan(
        Mapping(world_size=2, tp_size=2), "HF", "AUTO"
    )

    assert plan == (WeightLoadPolicy.DIRECT_RANK_READ,)
    assert reason is not None
    assert "active MPI communicator size (4)" in reason
    communicator.allgather.assert_not_called()


@pytest.mark.parametrize("plan", (None, "legacy_fallback"))
def test_communicator_mismatch_never_runs_legacy(tmp_path, monkeypatch, plan):
    (tmp_path / "model.safetensors").touch()
    communicator = mock.Mock()
    communicator.Get_size.return_value = 4
    monkeypatch.setattr(weight_loader_module, "ENABLE_MULTI_DEVICE", True)
    monkeypatch.setattr(weight_loader_module, "mpi_disabled", lambda: False)
    monkeypatch.setattr(weight_loader_module, "mpi_comm", lambda: communicator)
    loader = HfWeightLoader(weight_load_plan=plan)

    with (
        mock.patch.object(loader, "_prefetch_and_load") as legacy_load,
        pytest.raises(RuntimeError, match="cannot coordinate ranks"),
    ):
        loader.load_weights(
            str(tmp_path),
            Mapping(world_size=2, tp_size=2),
            model=_model(),
            _load_format="AUTO",
        )

    legacy_load.assert_not_called()
    communicator.Split_type.assert_not_called()


def test_rank_local_mapping_rejects_larger_active_communicator(monkeypatch):
    communicator = mock.Mock()
    communicator.Get_size.return_value = 2
    monkeypatch.setattr(weight_loader_module, "ENABLE_MULTI_DEVICE", True)
    monkeypatch.setattr(weight_loader_module, "mpi_disabled", lambda: False)
    monkeypatch.setattr(weight_loader_module, "mpi_comm", lambda: communicator)

    loader = HfWeightLoader(weight_load_plan="direct_rank_read")
    plan, reason = loader._get_coordinated_weight_load_plan(Mapping(), "HF", None)

    assert plan == (WeightLoadPolicy.DIRECT_RANK_READ,)
    assert reason is not None
    assert "active MPI communicator size (2)" in reason
    communicator.allgather.assert_not_called()


def test_cooperative_node_communicator_is_derived_from_active_communicator(monkeypatch):
    node_communicator = object()
    active_communicator = mock.Mock()
    active_communicator.Split_type.return_value = node_communicator
    monkeypatch.setattr(weight_loader_module, "ENABLE_MULTI_DEVICE", True)
    monkeypatch.setattr(weight_loader_module, "mpi_disabled", lambda: False)
    monkeypatch.setattr(weight_loader_module, "mpi_comm", lambda: active_communicator)

    assert HfWeightLoader._get_active_node_communicator() is node_communicator
    active_communicator.Split_type.assert_called_once_with(
        weight_loader_module._MPI.COMM_TYPE_SHARED
    )


def test_cooperative_node_communicator_is_freed_after_success():
    loader = HfWeightLoader(weight_load_plan="direct_rank_read")
    node_communicator = mock.Mock()
    expected_weights = {"weight": object()}

    with (
        mock.patch.object(loader, "_get_active_node_communicator", return_value=node_communicator),
        mock.patch.object(
            loader,
            "_load_cooperative_policy_with_communicator",
            return_value=expected_weights,
        ) as load_with_communicator,
    ):
        weights = loader._load_cooperative_policy(
            ["model.safetensors"], WeightLoadPolicy.DIRECT_RANK_READ
        )

    assert weights is expected_weights
    load_with_communicator.assert_called_once_with(
        ["model.safetensors"], WeightLoadPolicy.DIRECT_RANK_READ, node_communicator
    )
    node_communicator.Free.assert_called_once_with()


def test_legacy_fallback_uses_and_frees_active_node_communicator(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    loader = HfWeightLoader(weight_load_plan="legacy_fallback")
    node_communicator = mock.Mock()
    expected_weights = {"weight": object()}

    with (
        mock.patch.object(loader, "_get_active_node_communicator", return_value=node_communicator),
        mock.patch.object(
            loader, "_prefetch_and_load", return_value=expected_weights
        ) as legacy_load,
    ):
        weights = loader.load_weights(str(tmp_path), Mapping(), model=_model())

    assert weights is expected_weights
    legacy_load.assert_called_once_with([str(checkpoint)], node_communicator)
    node_communicator.Free.assert_called_once_with()


@pytest.mark.parametrize(
    "metadata",
    ({"_checkpoint_format": "MX"}, {"_load_format": "GMS"}),
)
def test_format_specific_fallback_does_not_split_active_communicator(tmp_path, metadata):
    (tmp_path / "model.safetensors").touch()
    loader = HfWeightLoader(weight_load_plan="legacy_fallback")
    expected_weights = {"weight": object()}

    with (
        mock.patch.object(loader, "_get_active_node_communicator") as get_node,
        mock.patch.object(loader, "_load_weights_in_parallel", return_value=expected_weights),
        mock.patch.object(weight_loader_module, "local_mpi_comm") as local_comm,
        mock.patch.object(weight_loader_module, "local_mpi_barrier") as local_barrier,
    ):
        weights = loader.load_weights(str(tmp_path), Mapping(), model=_model(), **metadata)

    assert weights is expected_weights
    get_node.assert_not_called()
    local_comm.assert_not_called()
    local_barrier.assert_not_called()


def test_remote_prefetch_error_is_raised_on_a_healthy_rank(monkeypatch):
    class FakeWorldCommunicator:
        @staticmethod
        def allgather(error_message):
            assert error_message is None
            return [None, "OSError: remote read failed"]

    monkeypatch.setattr(weight_loader_module, "ENABLE_MULTI_DEVICE", True)
    monkeypatch.setattr(weight_loader_module, "mpi_disabled", lambda: False)
    monkeypatch.setattr(weight_loader_module, "mpi_comm", lambda: FakeWorldCommunicator())

    with pytest.raises(
        RuntimeError, match="Rank 1 failed during direct_rank_read checkpoint prefetch"
    ):
        HfWeightLoader._raise_on_rank_error("direct_rank_read checkpoint prefetch", None)


def test_local_prefetch_error_is_coordinated_before_barrier(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    loader = HfWeightLoader(weight_load_plan="direct_rank_read")

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
        mock.patch.object(loader, "_get_active_node_communicator", return_value=local_communicator),
        mock.patch.object(loader, "_get_cooperative_prefetch_policy", return_value=(1, True)),
        mock.patch.object(loader, "prefetch_file_chunks", side_effect=OSError("read failed")),
        pytest.raises(RuntimeError, match="direct_rank_read checkpoint prefetch"),
    ):
        loader._load_cooperative_policy([str(checkpoint)], WeightLoadPolicy.DIRECT_RANK_READ)

    local_communicator.Barrier.assert_not_called()
    local_communicator.Free.assert_called_once_with()


def test_remote_policy_error_prevents_local_memory_allreduce(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    loader = HfWeightLoader(weight_load_plan="direct_rank_read")

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
        loader._get_cooperative_prefetch_policy([str(checkpoint)], local_communicator)

    local_communicator.allreduce.assert_not_called()


def test_different_node_local_backing_files_are_rejected(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()

    class FakeWorldCommunicator:
        @staticmethod
        def allgather(value):
            return [value]

    local_communicator = mock.Mock()
    local_communicator.allgather.side_effect = lambda signature: [
        signature,
        (("different-backing-file",),),
    ]
    monkeypatch.setattr(weight_loader_module, "ENABLE_MULTI_DEVICE", True)
    monkeypatch.setattr(weight_loader_module, "mpi_disabled", lambda: False)
    monkeypatch.setattr(weight_loader_module, "mpi_comm", lambda: FakeWorldCommunicator())
    loader = HfWeightLoader(weight_load_plan="shared_host_producer")

    with pytest.raises(RuntimeError, match="different backing files"):
        loader._get_cooperative_prefetch_policy([str(checkpoint)], local_communicator)

    local_communicator.allreduce.assert_not_called()


def test_mapper_assigned_after_construction_keeps_direct_read_ahead_eligible(tmp_path):
    (tmp_path / "model.safetensors").touch()
    expected_weights = {"weight": object()}
    weight_loader = HfWeightLoader(weight_load_plan=("direct_rank_read", "legacy_fallback"))
    checkpoint_loader = HfCheckpointLoader(weight_loader=weight_loader)
    checkpoint_loader.weight_mapper = mock.Mock()

    with (
        mock.patch.object(
            weight_loader, "_load_cooperative_policy", return_value=expected_weights
        ) as cooperative_load,
    ):
        weights = checkpoint_loader.load_weights(str(tmp_path), Mapping(), model=_model())

    assert weights is expected_weights
    cooperative_load.assert_called_once_with(
        [str(tmp_path / "model.safetensors")], WeightLoadPolicy.DIRECT_RANK_READ
    )


@pytest.mark.parametrize(
    "load_kwargs,expected_load_format",
    [({}, "AUTO"), ({"_load_format": "GMS"}, "GMS")],
)
def test_hf_checkpoint_loader_forwards_weight_policy_metadata(load_kwargs, expected_load_format):
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


def test_format_specific_session_calls_polymorphic_checkpoint_loader():
    checkpoint_loader = MistralCheckpointLoader()
    expected_weights = {"weight": object()}

    with mock.patch.object(
        checkpoint_loader, "load_weights", return_value=expected_weights
    ) as load_weights:
        with checkpoint_loader.open_weight_session("checkpoint", mapping=Mapping()) as weights:
            assert weights is expected_weights

    load_weights.assert_called_once_with("checkpoint", mapping=mock.ANY)


def test_model_loader_helper_uses_class_session_and_duck_fallback():
    from tensorrt_llm._torch.pyexecutor.model_loader import _open_checkpoint_weight_session

    events = []

    class SessionCheckpointLoader:
        @contextmanager
        def open_weight_session(self, checkpoint_dir, **kwargs):
            events.append("enter")
            try:
                yield {"weight": object()}
            finally:
                events.append("exit")

        def load_weights(self, *_args, **_kwargs):
            pytest.fail("class-defined session should be used")

    with _open_checkpoint_weight_session(
        SessionCheckpointLoader(), "checkpoint", mapping=Mapping()
    ) as weights:
        assert weights
        events.append("materialize")

    assert events == ["enter", "materialize", "exit"]

    duck_loader = mock.MagicMock()
    duck_loader.load_weights.return_value = {"duck": object()}
    with _open_checkpoint_weight_session(duck_loader, "checkpoint", mapping=Mapping()) as weights:
        assert weights == duck_loader.load_weights.return_value
    duck_loader.load_weights.assert_called_once_with("checkpoint", mapping=mock.ANY)


def test_hf_checkpoint_loader_preserves_strict_custom_loader_signature():
    expected_weights = {"weight": object()}

    class StrictCustomWeightLoader(BaseWeightLoader):
        def load_weights(self, checkpoint_dir: str, mapping: Mapping):
            assert checkpoint_dir == "checkpoint"
            assert mapping.world_size == 1
            return expected_weights

    loader = HfCheckpointLoader(weight_loader=StrictCustomWeightLoader())

    assert loader.load_weights("checkpoint", Mapping(), _load_format="AUTO") is expected_weights


def test_hf_session_preserves_hf_weight_loader_subclass_override():
    expected_weights = {"weight": object()}

    class OverriddenHfWeightLoader(HfWeightLoader):
        def load_weights(self, checkpoint_dir: str, mapping: Mapping, **kwargs):
            assert checkpoint_dir == "checkpoint"
            assert kwargs["_checkpoint_format"] == "HF"
            return expected_weights

    checkpoint_loader = HfCheckpointLoader(weight_loader=OverriddenHfWeightLoader())

    with checkpoint_loader.open_weight_session("checkpoint", Mapping()) as weights:
        assert weights is expected_weights


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"weight_load_plan": "unknown"}, "Unsupported weight-load policy"),
        (
            {
                "weight_load_plan": (
                    "direct_rank_read",
                    "direct_rank_read",
                )
            },
            "must not contain duplicate policies",
        ),
        (
            {
                "weight_load_plan": "direct_rank_read",
                "prefetch_chunk_size": 0,
            },
            "chunk_size must be positive",
        ),
        (
            {
                "weight_load_plan": "direct_rank_read",
                "prefetch_workers_per_rank": 0,
            },
            "workers_per_rank must be positive",
        ),
        (
            {
                "weight_load_plan": "shared_host_producer",
                "shared_host_buffer_budget": 0,
            },
            "shared_host_buffer_budget must be positive",
        ),
        (
            {
                "weight_load_plan": "shared_host_producer",
                "shared_host_buffer_budget": 256 * 1024 * 1024,
            },
            "must hold two prefetch chunks",
        ),
        (
            {
                "weight_load_plan": "rank_cooperative_stream",
                "shared_host_buffer_budget": 256 * 1024 * 1024,
            },
            "must hold two prefetch chunks",
        ),
    ],
)
def test_weight_load_options_are_validated(kwargs, match):
    with pytest.raises(ValueError, match=match):
        HfWeightLoader(**kwargs)

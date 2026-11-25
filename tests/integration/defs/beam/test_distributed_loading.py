# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Manual four-GPU load-error qualification with one 32-byte generated tensor.

Run in a matching installed/native TRT-LLM environment on one node::

    env -u TLLM_DISABLE_MPI timeout 120s mpirun -n 4 python -m pytest -q \\
        tests/integration/defs/beam/test_distributed_loading.py

This uses real MPI error exchange, catalog diagnostic agreement and CUDA copies.
Only model geometry/quantization metadata are reduced; no checkpoint, inference,
NVFP4 kernel or full-model qualification is involved. Not scheduled in L0.
The outer timeout bounds a broken collective, including pytest/process startup.
"""

import json
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("packed", [False, True], ids=["bf16", "nvfp4"])
def test_distributed_loading(tmp_path, monkeypatch, packed):
    from tensorrt_llm import mpi_rank, mpi_world_size
    from tensorrt_llm._utils import local_mpi_rank, mpi_disabled

    if mpi_disabled() or mpi_world_size() != 4 or torch.cuda.device_count() < 4:
        pytest.skip("requires MPI with four ranks and four visible GPUs on one node")

    from safetensors.torch import save_file

    from tensorrt_llm._torch.distributed.communicator import Distributed
    from tensorrt_llm._torch.modeling_v2.models.beam._tp4 import component_bf16 as bf16
    from tensorrt_llm._torch.modeling_v2.models.beam._tp4 import layout_bf16 as layout
    from tensorrt_llm._torch.modeling_v2.models.beam._tp4 import loader_bf16 as loader
    from tensorrt_llm._torch.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (
        component as nvfp4,
    )
    from tensorrt_llm.mapping import Mapping

    rank = mpi_rank()
    torch.cuda.set_device(local_mpi_rank())
    device = torch.device("cuda", local_mpi_rank())
    mapping = Mapping(world_size=4, rank=rank, tp_size=4)
    cls = (
        nvfp4.BeamCheckpointLayoutNvfp4Sm103Tp4 if packed else bf16.BeamCheckpointLayoutBf16Sm103Tp4
    )
    obj = cls.__new__(cls)
    torch.nn.Module.__init__(obj)
    source = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    spec = layout.TensorSpec("weight", (4, 2), layout.F32, "dense", None, None, None)
    obj.config, obj.specs = {"test": True}, {spec.name: spec}
    obj.mapping, obj.rank, obj.device = mapping, rank, device
    obj.layers, obj.expert_names, obj._shared_fc1 = (), [], {}
    obj.schedule = SimpleNamespace(is_sparse=lambda layer: False)
    obj.usable, obj.load_record, obj.diagnostics = False, None, False
    obj._init_collectives()
    obj.register_parameter(
        bf16._attr(spec.name),
        torch.nn.Parameter(torch.zeros_like(source, device=device), requires_grad=False),
    )
    monkeypatch.setattr(cls, "_tensor_specs", staticmethod(lambda *args: obj.specs))
    if packed:
        monkeypatch.setattr(nvfp4, "tensor_specs", lambda *args: obj.specs)
        monkeypatch.setattr(nvfp4, "read_quant_config", lambda *args: {})
        monkeypatch.setattr(nvfp4, "check_recipe", lambda *args: {})
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(obj.config))
    save_file({"weight": source}, str(tmp_path / "weights.safetensors"))
    (tmp_path / loader.INDEX_FILE).write_text(
        json.dumps({"weight_map": {"weight": "weights.safetensors"}})
    )
    # Keep staging tiny as well as the generated tensor; retain real CUDA copies.
    reader_type = loader.TileReader
    monkeypatch.setattr(bf16, "TileReader", lambda: reader_type(tile_bytes=16))
    original_read = bf16.read_slice_into

    def fail_read(*args, **kwargs):
        if rank == 2:
            raise OSError("injected rank-two read failure")
        return original_read(*args, **kwargs)

    obj.load(str(tmp_path), record_digests=False)
    assert obj.usable and torch.equal(obj.tensor("weight").cpu(), source)
    previous = obj.load_record
    if rank == 2:
        config_path.write_text("{}")
    with pytest.raises(bf16.LoadError, match="rank 2: CheckpointFormatError"):
        obj.load(str(tmp_path), record_digests=False)
    assert obj.usable and obj.load_record is previous  # No writes during validation.
    config_path.write_text(json.dumps(obj.config))

    with monkeypatch.context() as patch:
        patch.setattr(bf16, "read_slice_into", fail_read)
        with pytest.raises(bf16.LoadError, match="rank 2: OSError.*injected rank-two read failure"):
            obj.load(str(tmp_path), record_digests=False)
    assert not obj.usable and obj.load_record is None
    if packed:
        assert obj._shared_fc1 == {}

    obj.load(str(tmp_path), record_digests=False)
    assert obj.usable and torch.equal(obj.tensor("weight").cpu(), source)
    assert Distributed.get(mapping).allgather(obj.usable) == [True] * 4


def test_catalog_collectives() -> None:
    """Manual TP4 Beam shapes, no checkpoint; exact sums and rank-ordered vocabulary."""
    from tensorrt_llm import mpi_rank, mpi_world_size
    from tensorrt_llm._utils import local_mpi_rank, mpi_disabled

    if mpi_disabled() or mpi_world_size() != 4 or torch.cuda.device_count() < 4:
        pytest.skip("requires MPI with four ranks and four visible GPUs on one node")

    from tensorrt_llm._torch.modeling_v2.models.beam._tp4 import component_bf16 as bf16
    from tensorrt_llm.mapping import Mapping

    rank = mpi_rank()
    torch.cuda.set_device(local_mpi_rank())
    device = torch.device("cuda", local_mpi_rank())
    cls = bf16.BeamCheckpointLayoutBf16Sm103Tp4
    obj = cls.__new__(cls)
    torch.nn.Module.__init__(obj)
    obj.mapping = Mapping(world_size=4, rank=rank, tp_size=4)
    obj.rank, obj.device = rank, device
    obj._init_collectives()
    # Dyadic values keep the four-rank sum exactly representable in both dtypes.
    for dtype in (torch.bfloat16, torch.float32):
        for rows in (1, 17, 129, 512):
            base = (torch.arange(rows * 5120, device=device) % 31).reshape(rows, 5120)
            local = (base.float() / 32 + rank / 8).to(dtype)
            result = obj.all_reduce(local)
            expected = (base.float() / 8 + 0.75).to(dtype)
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                local, (base.float() / 32 + rank / 8).to(dtype), rtol=0, atol=0
            )
    for rows in (1, 17, 129):
        base = (torch.arange(rows * 37984, device=device) % 31).reshape(rows, 37984)
        local = (base.float() / 32 + rank / 8).to(torch.bfloat16)
        gathered = obj._allgather(local.t().contiguous()).t().contiguous()
        expected = torch.cat(
            [(base.float() / 32 + peer / 8).to(local.dtype) for peer in range(4)], dim=1
        )
        torch.testing.assert_close(gathered, expected, rtol=0, atol=0)
    # Capture after warmup, then replay with changed inputs and an intervening eager call.
    static = torch.full((1, 5120), rank / 8, dtype=torch.bfloat16, device=device)
    obj.all_reduce(static)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = obj.all_reduce(static)
    static.fill_(rank / 8 + 0.5)
    obj.all_reduce(torch.zeros(17, 5120, dtype=static.dtype, device=device))
    graph.replay()
    torch.testing.assert_close(captured, torch.full_like(captured, 2.75), rtol=0, atol=0)
    # Diagnostic collectives use integer flags and byte digests, outside the hot path.
    obj.diagnostics = False
    obj._check_diagnostics_agree()
    assert len(obj._digests_agree([torch.ones(3, device=device)])) == 1
    obj.diagnostics = rank == 2
    with pytest.raises(bf16.LoadError, match="on for 1 of 4 ranks"):
        obj._check_diagnostics_agree()
    with pytest.raises(RuntimeError, match="replicated MoE routing differs"):
        obj._digests_agree([torch.tensor([rank], device=device)])

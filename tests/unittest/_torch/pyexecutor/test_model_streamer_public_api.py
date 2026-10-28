# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the installed Model Streamer wheel.

The CPU tests in ``test_model_loader_checkpoint_io.py`` use a fake streamer.
These tests check the public ``tensor_names`` API of the real one, and its
choice to share the reads of a node among the ranks of that node.
"""

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file

from tensorrt_llm._torch.distributed.initialization import model_streamer_torch_distributed
from tensorrt_llm._torch.models.checkpoints.model_streamer.model_loading import (
    ModelStreamerMetricNames,
    get_weight_source_map,
    stream_batched_tensors,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Model Streamer uses CUDA tensors")
def test_public_tensor_selection(tmp_path):
    pytest.importorskip("runai_model_streamer")
    path = tmp_path / "model.safetensors"
    save_file(
        {"needed": torch.arange(8), "later": torch.arange(2), "unused": torch.arange(4)}, str(path)
    )
    weight_map = get_weight_source_map(str(tmp_path))
    assert weight_map == dict.fromkeys(("needed", "later", "unused"), str(path))

    received = {}
    # Two batches in one streamer session, as run_model_streamer() streams with a batch_index.
    metrics = stream_batched_tensors(
        weight_map,
        [{"needed"}, {"later"}],
        lambda name, tensor: received.setdefault(name, tensor.cpu().clone()),
        distributed=False,
    )
    assert set(received) == {"needed", "later"}
    torch.testing.assert_close(received["needed"], torch.arange(8))
    torch.testing.assert_close(received["later"], torch.arange(2))
    assert metrics[ModelStreamerMetricNames.MODEL_STREAMER_NODE_TENSORS] == 2
    assert metrics[ModelStreamerMetricNames.MODEL_STREAMER_BATCHES] == 2


@pytest.mark.cpu_only
def test_mixed_backend_group_shares_reads(monkeypatch):
    pytest.importorskip("runai_model_streamer")
    from runai_model_streamer.distributed_streamer.distributed_streamer import DistributedStreamer

    # Ray workers and the CuteDSL MegaMoE bootstrap create this group before the load.
    monkeypatch.delenv("RUNAI_STREAMER_DIST", raising=False)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(dist, "get_backend", lambda group=None: "cuda:nccl,cpu:gloo")
    monkeypatch.setattr(dist, "get_backend_config", lambda group=None: "cuda:nccl,cpu:gloo")
    streamer = DistributedStreamer()

    # By default, each rank reads all tensors of its node.
    streamer.set_is_distributed(True, "cuda")
    assert not streamer.is_distributed
    with model_streamer_torch_distributed(0, 2):
        streamer.set_is_distributed(True, "cuda")
    assert streamer.is_distributed

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
"""The Cosmos3 Sim rollout on two GPUs.

Under Ulysses-2 it equals the single-rank rollout: two ranks build the same model
(same seed), run the same clean passes and one denoising forward through the K/V
cache with the sequence sharded between them, and every rank's gathered output must
match what one rank computes alone.

With the VAE on the second GPU, one process generates on GPU 0, decodes each
finished chunk on GPU 1 and returns the video of the latents it generated. The
decoder it puts there, WinoVae, gives on GPU 1 the frames it gives on GPU 0 while
GPU 0 stays the process's current device.
"""

import importlib.util
import os
import sys
from pathlib import Path

os.environ["TLLM_DISABLE_MPI"] = "1"

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping

WORLD = 2

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD,
    reason=f"needs {WORLD} GPUs",
)


@pytest.fixture(scope="module", autouse=True)
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


def _single_gpu_test_module(file_name: str, module_name: str):
    """A test module of the parent directory, loaded for its helpers."""
    path = Path(__file__).resolve().parent.parent / file_name
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sim_helpers():
    """The single-GPU Sim transformer test module: model builder, cache, rollout."""
    return _single_gpu_test_module("test_cosmos3_sim_transformer.py", "sim_transformer_test")


def _wino_vae_helpers():
    """The single-GPU WinoVae chunk-decode test module: tiny model, latents, clip decode."""
    return _single_gpu_test_module("test_wino_vae_stream_decode.py", "wino_vae_stream_test")


def _rollouts(helpers, model):
    first = helpers.frames(5, seed=3)
    second = helpers.frames(2, seed=4)
    chunking = [(0, 1), (1, 3), (3, 5)]
    return helpers.two_pass_rollout(model, chunking, first, second)


def _worker(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        helpers = _sim_helpers()
        reference = _rollouts(helpers, helpers.build_model("TRTLLM"))
        mapping = VisualGenMapping(world_size=world_size, rank=rank, ulysses_size=world_size)
        sharded_model = helpers.build_model("TRTLLM", mapping=mapping)
        out = _rollouts(helpers, sharded_model)
        helpers.assert_same_model_output(out, reference, label=f"rank {rank} video")
    finally:
        dist.destroy_process_group()


def test_sim_rollout_under_ulysses_matches_one_rank():
    from ._visual_gen_dist_utils import spawn_with_retry

    spawn_with_retry(lambda port: mp.spawn(_worker, args=(WORLD, port), nprocs=WORLD, join=True))


def _decode_worker(index):
    helpers = _sim_helpers()
    helpers.assert_rollout_decodes_its_chunks(helpers.build_model("TRTLLM"), decode_device="cuda:1")


def test_sim_rollout_decodes_on_the_second_gpu():
    """Runs in a process of its own, like the rank workers of this directory, so the
    test process holds no CUDA context."""
    mp.spawn(_decode_worker, nprocs=1, join=True)


def _wino_vae_worker(index):
    wino = _wino_vae_helpers()
    steps = [1, 2, 2, 1]
    latents = wino._latents(sum(steps))
    current = torch.cuda.current_device()
    vae = wino._model(identity_post_quant=True)
    stream = torch.cuda.Stream(device=torch.device("cuda:1"))
    second_vae = wino._model(identity_post_quant=True).to(stream.device)
    with torch.inference_mode():
        expected = wino._stream(vae, latents, steps)
        with torch.cuda.device(stream.device), torch.cuda.stream(stream):
            decoded = wino._stream(second_vae, latents.to(stream.device), steps)
            stream.synchronize()
            assert torch.cuda.current_device() == stream.device.index
        assert torch.cuda.current_device() == current
        repeated = wino._stream(vae, latents, steps)
    assert decoded.device == stream.device
    rel = wino._rel_l2(decoded.to(expected.device), expected)
    assert rel < wino.REL_L2_TOL, f"rel_l2={rel:.3e}"
    assert torch.equal(repeated, expected)


def test_wino_vae_decodes_on_the_second_gpu():
    """Two WinoVae instances of the same random weights decode one clip chunk by chunk:
    one on GPU 0, the process's current device, the other on GPU 1 the way the rollout's
    chunk decoder calls it (GPU 1 current for the call only, on a stream of GPU 1, the
    latents copied over). The frames agree within the rounding bound, the current device
    is where the caller left it during and after the clip, and GPU 0's instance then
    decodes the clip again bit for bit. In a process of its own, as above."""
    mp.spawn(_wino_vae_worker, nprocs=1, join=True)

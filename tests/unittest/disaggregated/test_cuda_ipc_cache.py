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
"""Bounded UCX CUDA IPC remote-handle cache for KV cache transfers."""

import os
import queue
import time

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.disaggregation.cuda_ipc_cache import (
    CUDA_IPC_CACHE_MAX_REGIONS_ENV,
    CUDA_IPC_CACHE_MAX_SIZE_ENV,
    DEFAULT_CUDA_IPC_CACHE_MAX_REGIONS,
    bound_cuda_ipc_cache,
)


@pytest.mark.cpu_only
def test_default_bound_applied_when_unset():
    env = {"UCX_TLS": "cuda_ipc,cuda_copy"}
    assert bound_cuda_ipc_cache(env) == str(DEFAULT_CUDA_IPC_CACHE_MAX_REGIONS)
    assert env[CUDA_IPC_CACHE_MAX_REGIONS_ENV] == str(DEFAULT_CUDA_IPC_CACHE_MAX_REGIONS)
    assert CUDA_IPC_CACHE_MAX_SIZE_ENV not in env


@pytest.mark.cpu_only
def test_default_bound_is_idempotent():
    env = {}
    bound_cuda_ipc_cache(env)
    # A second transceiver in the same process sees the limit as already configured.
    assert bound_cuda_ipc_cache(env) is None
    assert env[CUDA_IPC_CACHE_MAX_REGIONS_ENV] == str(DEFAULT_CUDA_IPC_CACHE_MAX_REGIONS)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "user_env",
    [
        {CUDA_IPC_CACHE_MAX_REGIONS_ENV: "inf"},
        {CUDA_IPC_CACHE_MAX_REGIONS_ENV: "64"},
        {CUDA_IPC_CACHE_MAX_SIZE_ENV: "8G"},
    ],
    ids=["regions_inf", "regions_64", "size_only"],
)
def test_user_limit_respected(user_env):
    env = dict(user_env)
    assert bound_cuda_ipc_cache(env) is None
    assert env == user_env


@pytest.mark.cpu_only
def test_process_environment_used_by_default(monkeypatch):
    # setenv first so monkeypatch restores the original state, including the
    # variable bound_cuda_ipc_cache() sets below.
    for name in (CUDA_IPC_CACHE_MAX_REGIONS_ENV, CUDA_IPC_CACHE_MAX_SIZE_ENV):
        monkeypatch.setenv(name, "0")
        monkeypatch.delenv(name)
    assert bound_cuda_ipc_cache() == str(DEFAULT_CUDA_IPC_CACHE_MAX_REGIONS)
    assert os.environ[CUDA_IPC_CACHE_MAX_REGIONS_ENV] == str(DEFAULT_CUDA_IPC_CACHE_MAX_REGIONS)


# --------------------------------------------------------------------------------------
# Two-process NIXL write into many peer VMM chunks, with and without the bound.
# --------------------------------------------------------------------------------------

_CHUNK_BYTES = 2 << 20
_DST_AGENT = "ipc_cache_dst"
_SRC_AGENT = "ipc_cache_src"


def _round_values(round_idx: int, num_chunks: int) -> np.ndarray:
    return ((round_idx * 31 + np.arange(num_chunks)) % 251).astype(np.uint8)


def _transfer_env():
    os.environ["UCX_TLS"] = "cuda_ipc,cuda_copy,sm,self,tcp"
    os.environ["TRTLLM_NIXL_NUM_THREADS"] = "1"


def _receiver(num_chunks: int, rounds: int, to_sender, from_sender, results):
    _transfer_env()
    from cuda.bindings import runtime as cudart

    from tensorrt_llm._torch.disaggregation.base.agent import RegMemoryDescs
    from tensorrt_llm._torch.disaggregation.nixl._agent_cpp import BindingsNixlTransferAgent
    from tensorrt_llm.runtime.kv_cache_manager_v2._introspection import (
        PooledPhysMemAllocator,
        VirtMem,
    )

    torch.cuda.set_device(1)
    torch.zeros(1, device="cuda")
    # A VMM region backed by one physical allocation per chunk, like a KV cache pool.
    allocator = PooledPhysMemAllocator(_CHUNK_BYTES)
    vm = VirtMem(num_chunks * _CHUNK_BYTES, allocator, init_num_phys_mem=num_chunks)
    base_ptr = int(vm.address)
    agent = BindingsNixlTransferAgent(_DST_AGENT, True, num_threads=1)
    reg = RegMemoryDescs(
        "VRAM", [(base_ptr, num_chunks * _CHUNK_BYTES, allocator.device_id, _DST_AGENT)]
    )
    agent.register_memory(reg)
    to_sender.put((bytes(agent.get_local_agent_desc()), base_ptr, allocator.device_id))
    host = np.empty(num_chunks * _CHUNK_BYTES, dtype=np.uint8)
    matches = []
    for round_idx in range(rounds):
        if from_sender.get(timeout=240) != round_idx:
            raise RuntimeError("receiver: rounds out of order")
        (err,) = cudart.cudaMemcpy(
            host.ctypes.data,
            base_ptr,
            host.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"receiver: cudaMemcpy failed: {err}")
        expected = np.repeat(_round_values(round_idx, num_chunks), _CHUNK_BYTES)
        match = bool(np.array_equal(host, expected))
        matches.append(match)
        to_sender.put(match)
    from_sender.get(timeout=240)  # sender is done with the region
    agent.deregister_memory(reg)
    agent.shutdown()
    vm.destroy()
    results.put(("receiver", matches))


def _sender(cache_limit, num_chunks: int, rounds: int, from_receiver, to_receiver, results):
    _transfer_env()
    os.environ.pop(CUDA_IPC_CACHE_MAX_REGIONS_ENV, None)
    os.environ.pop(CUDA_IPC_CACHE_MAX_SIZE_ENV, None)
    if cache_limit is not None:
        os.environ[CUDA_IPC_CACHE_MAX_REGIONS_ENV] = cache_limit
    applied = bound_cuda_ipc_cache()

    from tensorrt_llm._torch.disaggregation.base.agent import (
        MemoryDescs,
        MemoryType,
        RegMemoryDescs,
        TransferOp,
        TransferRequest,
    )
    from tensorrt_llm._torch.disaggregation.nixl._agent_cpp import BindingsNixlTransferAgent

    torch.cuda.set_device(0)
    agent = BindingsNixlTransferAgent(_SRC_AGENT, True, num_threads=1)
    src = torch.empty(num_chunks * _CHUNK_BYTES, dtype=torch.uint8, device="cuda")
    reg = RegMemoryDescs("VRAM", [(src.data_ptr(), src.numel(), 0, "src")])
    agent.register_memory(reg)
    remote_desc, dst_base, dst_dev = from_receiver.get(timeout=240)
    agent.load_remote_agent(_DST_AGENT, remote_desc)

    src_tuples = [(src.data_ptr() + i * _CHUNK_BYTES, _CHUNK_BYTES, 0) for i in range(num_chunks)]
    dst_tuples = [(dst_base + i * _CHUNK_BYTES, _CHUNK_BYTES, dst_dev) for i in range(num_chunks)]
    seconds = []
    matches = []
    for round_idx in range(rounds):
        values = torch.from_numpy(_round_values(round_idx, num_chunks)).cuda()
        src.view(num_chunks, _CHUNK_BYTES).copy_(values[:, None].expand(-1, _CHUNK_BYTES))
        torch.cuda.synchronize()
        request = TransferRequest(
            TransferOp.WRITE,
            MemoryDescs(MemoryType.VRAM, src_tuples),
            MemoryDescs(MemoryType.VRAM, dst_tuples),
            _DST_AGENT,
            None,
        )
        start = time.perf_counter()
        status = agent.submit_transfer_requests(request)
        if not status.wait(timeout_ms=120000):
            raise RuntimeError(f"sender: round {round_idx} transfer failed")
        seconds.append(time.perf_counter() - start)
        to_receiver.put(round_idx)
        matches.append(from_receiver.get(timeout=240))
    to_receiver.put(None)
    agent.invalidate_remote_agent(_DST_AGENT)
    agent.deregister_memory(reg)
    agent.shutdown()
    results.put(("sender", applied, os.environ.get(CUDA_IPC_CACHE_MAX_REGIONS_ENV), seconds))


def _run_pair(cache_limit, num_chunks: int, rounds: int):
    ctx = torch.multiprocessing.get_context("spawn")
    to_sender, to_receiver, results = ctx.Queue(), ctx.Queue(), ctx.Queue()
    receiver = ctx.Process(
        target=_receiver, args=(num_chunks, rounds, to_sender, to_receiver, results)
    )
    sender = ctx.Process(
        target=_sender, args=(cache_limit, num_chunks, rounds, to_sender, to_receiver, results)
    )
    receiver.start()
    sender.start()
    out = {}
    deadline = time.monotonic() + 600
    while len(out) < 2:
        try:
            item = results.get(timeout=5)
        except queue.Empty:
            for proc in (sender, receiver):
                if proc.exitcode not in (None, 0):
                    raise RuntimeError(f"{proc.name} exited with {proc.exitcode}")
            if time.monotonic() > deadline:
                raise TimeoutError("transfer processes did not finish")
            continue
        out[item[0]] = item[1:]
    sender.join(timeout=60)
    receiver.join(timeout=60)
    assert sender.exitcode == 0 and receiver.exitcode == 0
    return out


@pytest.mark.timeout(900)
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_bounded_cache_rewrites_evicted_chunks():
    """Writes stay correct when every round re-imports evicted peer chunks.

    The peer region has more chunks than the default per-peer bound, so with the
    bound each round evicts and re-imports chunks; with 'inf' every chunk stays
    mapped after the first round.
    """
    pytest.importorskip("tensorrt_llm.tensorrt_llm_transfer_agent_binding")
    num_chunks = DEFAULT_CUDA_IPC_CACHE_MAX_REGIONS + 128
    rounds = 4
    bounded = _run_pair(None, num_chunks, rounds)
    unbounded = _run_pair("inf", num_chunks, rounds)

    applied, env_value, bounded_s = bounded["sender"]
    assert applied == str(DEFAULT_CUDA_IPC_CACHE_MAX_REGIONS)
    assert env_value == str(DEFAULT_CUDA_IPC_CACHE_MAX_REGIONS)
    assert bounded["receiver"][0] == [True] * rounds

    applied, env_value, unbounded_s = unbounded["sender"]
    assert applied is None and env_value == "inf"
    assert unbounded["receiver"][0] == [True] * rounds

    # Rounds after the first show the cost of re-importing evicted chunks.
    later_bounded = sum(bounded_s[1:]) / (rounds - 1)
    later_unbounded = sum(unbounded_s[1:]) / (rounds - 1)
    print(
        f"[cuda_ipc_cache] {num_chunks} x {_CHUNK_BYTES >> 20} MiB chunks per round: "
        f"bounded({DEFAULT_CUDA_IPC_CACHE_MAX_REGIONS}) rounds={[f'{s * 1e3:.1f}' for s in bounded_s]} ms, "
        f"unbounded rounds={[f'{s * 1e3:.1f}' for s in unbounded_s]} ms, "
        f"re-import cost ~{(later_bounded - later_unbounded) / num_chunks * 1e6:.0f} us/chunk"
    )

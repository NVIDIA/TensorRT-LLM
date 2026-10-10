# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-node NIXL KV-payload benchmark with pinned host and GPU buffers.

This measures the physical transfer layer. It deliberately keeps buffers and
registrations alive until NIXL proves completion, including after logical cancel.
"""

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
import zmq

from tensorrt_llm._torch.disaggregation.base.agent import (
    MemoryDescs,
    MemoryType,
    RegMemoryDescs,
    TransferOp,
    TransferRequest,
)
from tensorrt_llm._torch.disaggregation.nixl.agent import NixlTransferAgent

MODES = {
    "gpu": ("VRAM", "VRAM"),
    "host": ("DRAM", "DRAM"),
    "host_to_gpu": ("DRAM", "VRAM"),
    "gpu_to_host": ("VRAM", "DRAM"),
}


def _buffer(size: int, kind: str) -> torch.Tensor:
    if kind == "DRAM":
        return torch.empty(size, dtype=torch.uint8, pin_memory=True)
    return torch.empty(size, dtype=torch.uint8, device="cuda")


def _pattern(size: int, rank: int, mode_index: int) -> torch.Tensor:
    values = (np.arange(size, dtype=np.uint32) * 17 + rank * 31 + mode_index * 47) % 251
    return torch.from_numpy(values.astype(np.uint8))


def _memory_desc(kind: str, ptr: int, size: int, device: int) -> MemoryDescs:
    return MemoryDescs.from_arrays_uniform_device(
        getattr(MemoryType, kind),
        np.asarray([ptr], dtype=np.int64),
        np.asarray([size], dtype=np.int64),
        device if kind == "VRAM" else 0,
    )


def _run_mode(
    role: str,
    rank: int,
    mode: str,
    mode_index: int,
    config: dict,
    socket: zmq.Socket,
    sweep: int,
) -> None:
    src_kind, dst_kind = MODES[mode]
    kind = src_kind if role == "ctx" else dst_kind
    size = int(config["bytes_per_transfer"])
    timeout_ms = int(config.get("timeout_ms", 30000))
    samples = int(config.get("samples", 8))
    warmup = int(config.get("warmup", 2))
    delay_ms = int(config.get("delayed_completion_ms", 100))
    tensor = _buffer(size, kind)
    gpu_endpoint = _buffer(size, "VRAM") if kind == "DRAM" else tensor
    expected = _pattern(size, rank, mode_index)
    if role == "ctx":
        gpu_endpoint.copy_(expected, non_blocking=False)
    else:
        tensor.zero_()
        if kind == "DRAM":
            gpu_endpoint.zero_()
    torch.cuda.synchronize()
    device = torch.cuda.current_device() if kind == "VRAM" else 0
    name = f"ctt_{role}_{rank}_{mode}_{sweep}"
    peer = f"ctt_{'gen' if role == 'ctx' else 'ctx'}_{rank}_{mode}_{sweep}"
    agent = NixlTransferAgent(name=name)
    registration = RegMemoryDescs(kind, [(tensor.data_ptr(), size, device, name)])
    registered = False
    submitted = False
    remote_active = False
    published = False
    settled = False
    try:
        agent.register_memory(registration)
        registered = True
        local = {
            "name": name,
            "ptr": tensor.data_ptr(),
            "kind": kind,
            "device": device,
            "agent": bytes(agent.get_local_agent_desc()),
        }
        if role == "gen":
            published = True
            socket.send_pyobj(local)
            remote = socket.recv_pyobj()
        else:
            remote = socket.recv_pyobj()
            published = True
            socket.send_pyobj(local)
        agent.load_remote_agent(peer, remote["agent"])
        results = []
        scenarios = [("throughput", i) for i in range(warmup + samples)]
        scenarios += [("delayed_completion", 0), ("logical_cancel", 0)]
        for scenario, index in scenarios:
            if role == "gen":
                tensor.zero_()
                if kind == "DRAM":
                    gpu_endpoint.zero_()
                torch.cuda.synchronize()
                socket.send_pyobj((scenario, index))
                remote_active = True
                result = socket.recv_pyobj()
                if scenario in ("logical_cancel", "delayed_completion"):
                    if result != {"pending": True, "logical_cancel": scenario == "logical_cancel"}:
                        raise RuntimeError(f"{mode} {scenario}: missing pending transfer notice")
                    time.sleep(delay_ms / 1000)
                    socket.send_pyobj(("drain", scenario, index))
                    result = socket.recv_pyobj()
                if result.get("error"):
                    raise RuntimeError(result["error"])
                destination_stage_seconds = 0.0
                if kind == "DRAM":
                    stage_start = time.perf_counter()
                    gpu_endpoint.copy_(tensor, non_blocking=True)
                    torch.cuda.synchronize()
                    destination_stage_seconds = time.perf_counter() - stage_start
                actual = gpu_endpoint.cpu()
                if not torch.equal(actual, expected):
                    mismatch = torch.nonzero(actual != expected).flatten()
                    raise AssertionError(
                        f"{mode} {scenario}: {len(mismatch)} wrong bytes; first={mismatch[0].item()}"
                    )
                socket.send_pyobj("ready")
                if socket.recv_pyobj() != "ready":
                    raise RuntimeError("missing sender GPU-ready acknowledgement")
                remote_active = False
                results.append(
                    {
                        "scenario": scenario,
                        "sample": index,
                        "warmup": scenario == "throughput" and index < warmup,
                        "seconds": result["seconds"],
                        "source_stage_seconds": result["source_stage_seconds"],
                        "destination_stage_seconds": destination_stage_seconds,
                        "gpu_ready_seconds": (
                            result["source_stage_seconds"]
                            + result["seconds"]
                            + destination_stage_seconds
                        ),
                        "bytes": size,
                        "byte_correct": True,
                        "logical_cancel": result["logical_cancel"],
                    }
                )
            else:
                received = socket.recv_pyobj()
                if received != (scenario, index):
                    raise RuntimeError(f"unexpected benchmark control message: {received!r}")
                source_stage_seconds = 0.0
                if kind == "DRAM":
                    stage_start = time.perf_counter()
                    tensor.copy_(gpu_endpoint, non_blocking=True)
                    torch.cuda.synchronize()
                    source_stage_seconds = time.perf_counter() - stage_start
                request = TransferRequest(
                    TransferOp.WRITE,
                    _memory_desc(src_kind, tensor.data_ptr(), size, device),
                    _memory_desc(dst_kind, remote["ptr"], size, remote["device"]),
                    peer,
                    None,
                )
                start = time.perf_counter()
                submitted = True
                status = agent.submit_transfer_requests(request)
                cancelled = scenario == "logical_cancel"
                if scenario in ("logical_cancel", "delayed_completion"):
                    socket.send_pyobj({"pending": True, "logical_cancel": cancelled})
                    if socket.recv_pyobj() != ("drain", scenario, index):
                        raise RuntimeError(f"{mode} {scenario}: missing drain request")
                    # The receiver has seen cancellation or deferred completion.
                    # Registered source and destination memory are still owned.
                complete = status.wait(timeout_ms=timeout_ms)
                elapsed = time.perf_counter() - start
                if not complete or not status.is_completed():
                    raise RuntimeError(f"{mode} {scenario}: physical completion unproven")
                socket.send_pyobj(
                    {
                        "seconds": elapsed,
                        "source_stage_seconds": source_stage_seconds,
                        "logical_cancel": cancelled,
                    }
                )
                if socket.recv_pyobj() != "ready":
                    raise RuntimeError("missing receiver GPU-ready acknowledgement")
                socket.send_pyobj("ready")
                submitted = False
        if role == "gen":
            socket.send_pyobj("done")
            socket.recv_pyobj()
            settled = True
            out = Path(config["work_dir"]) / "host_transfer"
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"sweep{sweep}_rank{rank}_{mode}.json"
            path.write_text(
                json.dumps(
                    {
                        "mode": mode,
                        "rank": rank,
                        "sweep": sweep,
                        "job_id": os.environ.get("SLURM_JOB_ID"),
                        "samples": results,
                    },
                    indent=2,
                )
            )
        else:
            if socket.recv_pyobj() != "done":
                raise RuntimeError("missing receiver completion acknowledgement")
            socket.send_pyobj("done")
            settled = True
    except BaseException:
        if submitted or remote_active or (published and not settled):
            # A failed or timed-out backend call gives no proof that remote access
            # stopped. Process death lets the transport own teardown order.
            traceback.print_exc()
            os.sys.stderr.flush()
            os._exit(2)
        raise
    finally:
        if not submitted and not remote_active and (not published or settled):
            if registered:
                agent.deregister_memory(registration)
            agent.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("ctx", "gen"), required=True)
    args = parser.parse_args()
    with open(os.environ["CTT_CONFIG"]) as stream:
        cfg = yaml.safe_load(stream)
    config = dict(cfg["host_transfer_benchmark"])
    config["work_dir"] = cfg["environment"]["work_dir"]
    if int(config["bytes_per_transfer"]) <= 0:
        raise ValueError("bytes_per_transfer must be positive")
    if int(config.get("samples", 8)) <= 0 or int(config.get("warmup", 2)) < 0:
        raise ValueError("samples must be positive and warmup must be non-negative")
    modes = config.get("modes", ["gpu", "host"])
    if "gpu" not in modes or "host" not in modes or len(set(modes)) != len(modes):
        raise ValueError("modes require unique gpu and host baselines")
    rank = int(os.environ["SLURM_LOCALID"])
    sweep = int(os.environ["CTT_SWEEP"])
    if args.role == "gen":
        out = Path(config["work_dir"]) / "host_transfer"
        out.mkdir(parents=True, exist_ok=True)
        for old in out.glob(f"sweep{sweep}_rank{rank}_*.json"):
            old.unlink()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    context = zmq.Context()
    socket = context.socket(zmq.REP if args.role == "ctx" else zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    timeout_ms = int(config.get("timeout_ms", 30000))
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    base_port = int(os.environ["ZMQ_PORT"]) + 100 + rank
    if args.role == "ctx":
        socket.bind(f"tcp://*:{base_port}")
    else:
        socket.connect(f"tcp://{os.environ['CTX_NODE']}:{base_port}")
    try:
        for index, mode in enumerate(modes):
            if mode not in MODES:
                raise ValueError(f"unknown transfer mode: {mode}")
            _run_mode(args.role, rank, mode, index, config, socket, sweep)
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()

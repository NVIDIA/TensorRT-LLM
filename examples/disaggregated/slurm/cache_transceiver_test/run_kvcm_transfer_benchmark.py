# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure NIXL writes between pages owned by native KV cache manager V2."""

import argparse
import ctypes
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
import zmq

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

from tensorrt_llm._torch.disaggregation.base.agent import (
    MemoryDescs,
    MemoryType,
    RegMemoryDescs,
    TransferOp,
    TransferRequest,
)
from tensorrt_llm._torch.disaggregation.nixl.agent import NixlTransferAgent
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    GpuCacheTierConfig,
    HostCacheTierConfig,
    KVCacheManager,
    KVCacheManagerConfig,
)

MODES = {
    "gpu": ("VRAM", "VRAM"),
    "host": ("DRAM", "DRAM"),
    "host_to_gpu": ("DRAM", "VRAM"),
    "gpu_to_host": ("VRAM", "DRAM"),
    "mixed": ("DRAM", "MIXED"),
}


def _pair_pages(source: list[dict], destination: list[dict]) -> list[tuple[dict, dict]]:
    src = {(page["group"], page["ordinal"], page["pool"]): page for page in source}
    dst = {(page["group"], page["ordinal"], page["pool"]): page for page in destination}
    if src.keys() != dst.keys() or len(src) != len(source) or len(dst) != len(destination):
        raise ValueError("source and destination page ordinals do not match")
    pairs = []
    for key in sorted(src):
        source_page, destination_page = src[key], dst[key]
        if (
            source_page["bytes"] != destination_page["bytes"]
            or source_page["valid_tokens"] != destination_page["valid_tokens"]
        ):
            raise ValueError(f"incompatible KV page layout at {key}")
        pairs.append((source_page, destination_page))
    return pairs


def _check_cuda(result: tuple, operation: str) -> None:
    if result[0] != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed: {result[0]}")


def _bytes_at(address: int, size: int, kind: str) -> bytes:
    if kind == "DRAM":
        return ctypes.string_at(address, size)
    host = (ctypes.c_ubyte * size)()
    _check_cuda(
        cudart.cudaMemcpy(
            ctypes.addressof(host),
            address,
            size,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        ),
        "cudaMemcpyDeviceToHost",
    )
    return bytes(host)


def _clear_pages(pages: list[dict], stream: int) -> None:
    for page in pages:
        if page["kind"] == "DRAM":
            ctypes.memset(page["address"], 0xFF, page["bytes"])
        else:
            _check_cuda(
                cudart.cudaMemsetAsync(page["address"], 0xFF, page["bytes"], stream),
                "cudaMemsetAsync",
            )
    torch.cuda.current_stream().synchronize()


def _pattern(rank: int, ordinal: int, mode_index: int) -> int:
    return (rank * 31 + ordinal * 17 + mode_index * 47 + 1) % 251


def _describe(pages: list, kind: str) -> list[dict]:
    expected_level = 1 if kind == "DRAM" else 0
    for page in pages:
        if page.cache_level != expected_level:
            raise RuntimeError(f"expected cache level {expected_level}, got {page.cache_level}")
        if (
            page.bytes <= 0
            or page.address < page.pool_base_address
            or page.address + page.bytes > page.pool_base_address + page.pool_bytes
        ):
            raise RuntimeError("native page descriptor is outside its pool")
    return [
        {
            "group": int(page.layer_group_id),
            "ordinal": int(page.ordinal),
            "pool": int(page.pool_index),
            "address": int(page.address),
            "bytes": int(page.bytes),
            "valid_tokens": int(page.valid_tokens),
            "pool_base": int(page.pool_base_address),
            "pool_bytes": int(page.pool_bytes),
            "kind": kind,
        }
        for page in pages
    ]


def _new_cache(config: dict, role: str, source_kind: str):
    page_bytes = int(config["page_bytes"])
    pages = int(config["pages"])
    tokens_per_block = int(config.get("tokens_per_block", 32))
    tokens = pages * tokens_per_block
    quota = max(4 * pages * page_bytes, 4 << 20)
    native_config = KVCacheManagerConfig(
        tokens_per_block=tokens_per_block,
        cache_tiers=[GpuCacheTierConfig(quota), HostCacheTierConfig(quota)],
        layers=[AttentionLayerConfig(0, [BufferConfig("key", page_bytes, is_sparse=True)])],
        enable_partial_reuse=False,
    )
    manager = KVCacheManager(native_config)
    cache = manager.create_kv_cache()
    try:
        stream = torch.cuda.current_stream().cuda_stream
        if not cache.resume(stream) or not cache.resize(tokens, tokens):
            raise RuntimeError("native benchmark cache could not allocate history")
        if role == "ctx":
            snapshot = cache.get_page_storage_snapshot(0)
            if len(snapshot.base_page_indices) != pages or any(
                level != 0 for level in snapshot.cache_levels
            ):
                raise RuntimeError("source history was not fully allocated on GPU")
            base = int(manager.get_mem_pool_base_address(0, "key"))
            stride = int(manager.get_page_stride(0, "key"))
            for ordinal, slot in enumerate(snapshot.base_page_indices):
                if slot < 0:
                    raise RuntimeError("source history has a missing page")
                _check_cuda(
                    cudart.cudaMemsetAsync(
                        base + int(slot) * stride,
                        _pattern(config["rank"], ordinal, config["mode_index"]),
                        page_bytes,
                        stream,
                    ),
                    "cudaMemsetAsync",
                )
            torch.cuda.current_stream().synchronize()
            if source_kind == "DRAM" and not cache.enter_decode():
                raise RuntimeError("native cache could not offload sparse history")
            access = cache.begin_external_read([0], tokens)
            cache.wait_external_access_ready(access)
            source_pages = cache.get_external_access_pages(access)
            if len(source_pages) != pages:
                raise RuntimeError("native source page count differs from requested history")
            return manager, cache, [access], _describe(source_pages, source_kind), stream

        cache.suspend()
        accesses = []
        destinations = []
        destination_kind = config["destination_kind"]
        for ordinal in range(pages):
            kind = (
                ("DRAM" if ordinal % 2 == 0 else "VRAM")
                if destination_kind == "MIXED"
                else destination_kind
            )
            access = cache.reserve_external_receive(0, ordinal, 1 if kind == "DRAM" else 0)
            cache.wait_external_access_ready(access)
            accesses.append(access)
            received = cache.get_external_access_pages(access)
            if len(received) != 1:
                raise RuntimeError("native receive reservation did not return one page")
            destinations.extend(_describe(received, kind))
        return manager, cache, accesses, destinations, stream
    except BaseException:
        cache.close()
        manager.shutdown()
        raise


def _register_pools(agent, pages: list[dict], device: int, name: str, registered: list) -> None:
    grouped = {"VRAM": {}, "DRAM": {}}
    for page in pages:
        key = (page["pool_base"], page["pool_bytes"])
        if key not in grouped[page["kind"]]:
            grouped[page["kind"]][key] = (
                page["pool_base"],
                page["pool_bytes"],
                device if page["kind"] == "VRAM" else 0,
                f"{name}_{page['kind']}_{len(grouped[page['kind']])}",
            )
    for kind, pools in grouped.items():
        if pools:
            desc = RegMemoryDescs(kind, list(pools.values()))
            agent.register_memory(desc)
            registered.append(desc)


def _requests(pairs: list[tuple[dict, dict]], peer: str, device: int) -> list:
    grouped = {}
    for src, dst in pairs:
        grouped.setdefault((src["kind"], dst["kind"]), []).append((src, dst))
    requests = []
    for (src_kind, dst_kind), group in grouped.items():

        def descriptors(which: int, kind: str) -> MemoryDescs:
            return MemoryDescs.from_arrays_uniform_device(
                getattr(MemoryType, kind),
                np.asarray([pair[which]["address"] for pair in group], dtype=np.int64),
                np.asarray([pair[which]["bytes"] for pair in group], dtype=np.int64),
                device if kind == "VRAM" else 0,
            )

        requests.append(
            TransferRequest(
                TransferOp.WRITE, descriptors(0, src_kind), descriptors(1, dst_kind), peer, None
            )
        )
    return requests


def _run_mode(
    role: str, rank: int, mode: str, mode_index: int, config: dict, socket: zmq.Socket, sweep: int
) -> None:
    source_kind, destination_kind = MODES[mode]
    local_config = dict(config, rank=rank, mode_index=mode_index, destination_kind=destination_kind)
    manager, cache, accesses, pages, stream = _new_cache(local_config, role, source_kind)
    name = f"kvcm_{role}_{rank}_{mode}_{sweep}"
    peer = f"kvcm_{'gen' if role == 'ctx' else 'ctx'}_{rank}_{mode}_{sweep}"
    agent = None
    registered = []
    pending = False
    remote_active = False
    published = False
    settled = False
    size = sum(page["bytes"] for page in pages)
    timeout_ms = int(config.get("timeout_ms", 30000))
    delay_ms = int(config.get("delayed_completion_ms", 100))
    try:
        agent = NixlTransferAgent(name=name)
        if role == "ctx":
            for page in pages:
                expected = _pattern(rank, page["ordinal"], mode_index)
                if (
                    _bytes_at(page["address"], page["bytes"], page["kind"])
                    != bytes([expected]) * page["bytes"]
                ):
                    raise AssertionError(
                        f"{mode}: source offload changed ordinal {page['ordinal']}"
                    )
        _register_pools(agent, pages, torch.cuda.current_device(), name, registered)
        for access in accesses:
            cache.expose_external_access(access)
        local = {"pages": pages, "agent": bytes(agent.get_local_agent_desc())}
        if role == "gen":
            published = True
            socket.send_pyobj(local)
            remote = socket.recv_pyobj()
        else:
            remote = socket.recv_pyobj()
            published = True
            socket.send_pyobj(local)
        agent.load_remote_agent(peer, remote["agent"])
        pairs = (
            _pair_pages(remote["pages"], pages)
            if role == "gen"
            else _pair_pages(pages, remote["pages"])
        )
        requests = _requests(pairs, peer, torch.cuda.current_device()) if role == "ctx" else []
        results = []
        warmup = int(config.get("warmup", 2))
        scenarios = [
            ("throughput", index) for index in range(warmup + int(config.get("samples", 8)))
        ]
        scenarios += [("delayed_completion", 0), ("logical_cancel", 0)]
        for scenario, index in scenarios:
            if role == "gen":
                _clear_pages(pages, stream)
                socket.send_pyobj((scenario, index))
                remote_active = True
                result = socket.recv_pyobj()
                if scenario != "throughput":
                    if result != {"pending": True, "logical_cancel": scenario == "logical_cancel"}:
                        raise RuntimeError("missing pending transfer notice")
                    time.sleep(delay_ms / 1000)
                    socket.send_pyobj(("drain", scenario, index))
                    result = socket.recv_pyobj()
                remote_active = False
                for src, dst in pairs:
                    expected = _pattern(rank, src["ordinal"], mode_index)
                    if (
                        _bytes_at(dst["address"], dst["bytes"], dst["kind"])
                        != bytes([expected]) * dst["bytes"]
                    ):
                        raise AssertionError(
                            f"{mode} {scenario}: byte mismatch at ordinal {src['ordinal']}"
                        )
                results.append(
                    {
                        "scenario": scenario,
                        "sample": index,
                        "warmup": scenario == "throughput" and index < warmup,
                        "seconds": result["seconds"],
                        "bytes": size,
                        "byte_correct": True,
                        "logical_cancel": result["logical_cancel"],
                    }
                )
            else:
                if socket.recv_pyobj() != (scenario, index):
                    raise RuntimeError("unexpected transfer request")
                start = time.perf_counter()
                statuses = []
                pending = True
                for request in requests:
                    statuses.append(agent.submit_transfer_requests(request))
                cancelled = scenario == "logical_cancel"
                if scenario != "throughput":
                    socket.send_pyobj({"pending": True, "logical_cancel": cancelled})
                    if socket.recv_pyobj() != ("drain", scenario, index):
                        raise RuntimeError("missing physical drain request")
                if not all(
                    status.wait(timeout_ms=timeout_ms) and status.is_completed()
                    for status in statuses
                ):
                    raise RuntimeError("physical completion unproven")
                pending = False
                socket.send_pyobj(
                    {"seconds": time.perf_counter() - start, "logical_cancel": cancelled}
                )
        if role == "gen":
            for access in accesses:
                descriptor = cache.get_external_access_pages(access)[0]
                cache.finalize_external_receive(access, descriptor.valid_tokens, stream)
            socket.send_pyobj("done")
            socket.recv_pyobj()
            settled = True
            out = Path(config["work_dir"]) / "kvcm_transfer"
            out.mkdir(parents=True, exist_ok=True)
            (out / f"sweep{sweep}_rank{rank}_{mode}.json").write_text(
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
        if pending or remote_active or (published and not settled):
            traceback.print_exc()
            os.sys.stderr.flush()
            os._exit(2)
        raise
    finally:
        if not pending and not remote_active and (not published or settled):
            for access in accesses:
                cache.end_external_access(access)
            cache.close()
            if agent is not None:
                for desc in registered:
                    agent.deregister_memory(desc)
                agent.shutdown()
            manager.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("ctx", "gen"), required=True)
    args = parser.parse_args()
    with open(os.environ["CTT_CONFIG"]) as stream:
        cfg = yaml.safe_load(stream)
    config = dict(cfg["kvcm_transfer_benchmark"])
    config["work_dir"] = cfg["environment"]["work_dir"]
    if int(config["page_bytes"]) <= 0 or int(config["pages"]) <= 0:
        raise ValueError("page_bytes and pages must be positive")
    if int(config.get("samples", 8)) <= 0 or int(config.get("warmup", 2)) < 0:
        raise ValueError("samples must be positive and warmup must be non-negative")
    modes = config.get("modes", ["gpu", "host"])
    if not {"gpu", "host"}.issubset(modes) or len(set(modes)) != len(modes):
        raise ValueError("modes require unique gpu and host baselines")
    rank = int(os.environ["SLURM_LOCALID"])
    sweep = int(os.environ["CTT_SWEEP"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    benchmark_stream = torch.cuda.Stream()
    torch.cuda.set_stream(benchmark_stream)
    if args.role == "gen":
        out = Path(config["work_dir"]) / "kvcm_transfer"
        out.mkdir(parents=True, exist_ok=True)
        for old in out.glob(f"sweep{sweep}_rank{rank}_*.json"):
            old.unlink()
    context = zmq.Context()
    socket = context.socket(zmq.REP if args.role == "ctx" else zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    timeout_ms = int(config.get("timeout_ms", 30000))
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    port = int(os.environ["ZMQ_PORT"]) + 200 + rank
    if args.role == "ctx":
        socket.bind(f"tcp://*:{port}")
    else:
        socket.connect(f"tcp://{os.environ['CTX_NODE']}:{port}")
    try:
        for index, mode in enumerate(modes):
            if mode not in MODES:
                raise ValueError(f"unknown mode: {mode}")
            _run_mode(args.role, rank, mode, index, config, socket, sweep)
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()

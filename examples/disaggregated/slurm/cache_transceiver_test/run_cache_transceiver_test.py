# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Per-rank driver for the 2-node KV cache transceiver bandwidth harness.

Each process runs ONE side (ctx or gen) inside its own MPI world of size N
(= gpus_per_node). The two srun steps (ctx on node A, gen on node B) talk only
over ZMQ (leader-to-leader) to hand the context connection info to the gen side;
the actual KV transfer goes over UCX/NIXL using the endpoint inside that info.

For every (UCX env set is fixed per process, set by launch.slurm) x transceiver
combination x request length, the ctx side fills a request's KV blocks with a
deterministic, rank-specific pattern, sends it, and the gen side verifies the
received blocks regenerate to the same pattern. Bandwidth is emitted by the
transceivers themselves into per-rank CSVs (parsed later by report.py):
  C++  -> TRTLLM_KVCACHE_TIME_OUTPUT_PATH (rank_*_send.csv / rank_*_recv.csv)
  Py   -> TLLM_KV_TRANSFER_PERF_LOG_FILE  (py_*_*.csv, throughput_mbs)

This driver mirrors the single-process test (tests/unittest/others/
test_kv_cache_transceiver.py) and the multi-process Python test
(tests/unittest/disaggregated/test_py_cache_transceiver_mp.py).
"""

import argparse
import gc
import json
import os
import pickle
import signal
import sys
from dataclasses import dataclass
from typing import List, Optional

import torch
import yaml
from mpi4py import MPI
from report import build_cases  # shared case enumeration (same dir on sys.path)

import tensorrt_llm
import tensorrt_llm.bindings
import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm._torch.pyexecutor.hang_detector import HangDetector
from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import create_kv_cache_transceiver
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState, LlmRequestType
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager, KVCacheManagerV2
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.sampling_params import SamplingParams

AttentionTypeCpp = tensorrt_llm.bindings.internal.batch_manager.AttentionType
CacheTypeCpp = tensorrt_llm.bindings.internal.batch_manager.CacheType
DataType = tensorrt_llm.bindings.DataType

# Must match report.py.
RID_COMBINATION_STRIDE = 1_000_000
RID_REQLEN_STRIDE = 10_000

DTYPE_MAP = {"FP8": DataType.FP8, "HALF": DataType.HALF, "BF16": DataType.BF16}


@dataclass
class KvCacheConfigV2:
    """KvCacheConfig wrapper for KVCacheManagerV2.

    Mirrors the single-process reference test_cache_transceiver_single_process.py.
    KVCacheManagerV2 reads these fields off the config object directly (no
    pydantic defaults are filled in for a bare dataclass), so every attribute it
    accesses MUST exist here -- a missing one surfaces as an AttributeError at
    transceiver setup time (seen with kv_cache_event_hash_algo / pool_ratio).
    Keep this in sync with the reference dataclass in
    tests/unittest/disaggregated/test_cache_transceiver_single_process.py.
    """

    max_tokens: Optional[int] = None
    enable_block_reuse: bool = False
    max_attention_window: Optional[List[int]] = None
    sink_token_length: Optional[int] = None
    free_gpu_memory_fraction: Optional[float] = None
    host_cache_size: Optional[int] = None
    onboard_blocks: bool = True
    cross_kv_cache_fraction: Optional[float] = None
    secondary_offload_min_priority: Optional[int] = None
    event_buffer_max_size: int = 0
    kv_cache_event_hash_algo: str = "auto"
    max_gpu_total_bytes: Optional[int] = None
    enable_partial_reuse: bool = False
    copy_on_partial_reuse: bool = False
    dtype: str = "auto"
    pool_ratio: Optional[List[float]] = None
    avg_seq_len: Optional[int] = None
    max_util_for_resume: float = 0.95


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


def make_rid(case_idx, reqlen_idx, r):
    return case_idx * RID_COMBINATION_STRIDE + reqlen_idx * RID_REQLEN_STRIDE + r


def seed_for(rid, rank, layer):
    # Distinct per (request, rank, layer); deterministic on both ctx and gen.
    return (rid * 1_000_003 + rank * 1009 + layer * 31) & 0x7FFFFFFF


def _layers_per_pp(num_layers, pp):
    base, extra = divmod(num_layers, pp)
    return [base + (1 if r < extra else 0) for r in range(pp)]


def local_layer_count(num_layers, pp, pp_rank):
    return _layers_per_pp(num_layers, pp)[pp_rank]


def build_kv_cache_manager(cfg_kv, mapping, use_v2):
    dtype = DTYPE_MAP[cfg_kv["dtype"].upper()]
    tpb = cfg_kv["tokens_per_block"]
    max_req = cfg_kv["_max_request_len"]
    max_seq_len = ((max_req + tpb - 1) // tpb) * tpb
    max_tokens = max_seq_len * 2  # headroom; sequences are freed after each request
    common = dict(
        num_layers=cfg_kv["num_layers"],
        num_kv_heads=cfg_kv["num_kv_heads"],
        head_dim=cfg_kv["head_dim"],
        tokens_per_block=tpb,
        max_seq_len=max_seq_len,
        # A few slots: V2's IndexMapper needs headroom (max_batch_size=1 yields 0
        # usable slots); we still transfer one request at a time and free it.
        max_batch_size=4,
        mapping=mapping,
        dtype=dtype,
    )
    if use_v2:
        return KVCacheManagerV2(
            KvCacheConfigV2(
                max_tokens=max_tokens, enable_block_reuse=False, max_attention_window=[max_seq_len]
            ),
            CacheTypeCpp.SELF,
            vocab_size=cfg_kv.get("vocab_size", 32000),
            **common,
        )
    return KVCacheManager(
        trtllm.KvCacheConfig(max_tokens=max_tokens, enable_block_reuse=False),
        CacheTypeCpp.SELF,
        **common,
    )


def add_sequence(mgr, req, prompt_len, use_v2):
    """Allocate KV blocks for the request. Returns a handle to close (V2) or None."""
    if use_v2:
        kv = mgr._create_kv_cache(req.py_request_id, None, None)
        ok = kv.resume(torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError(f"V2 resume failed for request {req.py_request_id}")
        kv.resize(prompt_len)
        return kv
    mgr.impl.add_sequence_batch([(req.py_request_id, prompt_len, 1)], [req])
    return None


def free_sequence(mgr, req, kv_handle, use_v2):
    if use_v2:
        # free_resources() closes the kv_cache AND releases the IndexMapper slot
        # (kv_handle.close() alone leaks the slot, exhausting them after a few
        # requests).
        torch.cuda.current_stream().synchronize()
        mgr.free_resources(req)
        return
    # block reuse is disabled here, so remove_sequence never reads the request's
    # context position (the store-for-reuse path is gated behind reuse); the
    # prefill-completion shim used by the unit tests is unnecessary.
    mgr.impl.remove_sequence(req.py_request_id, req, True)


def _seeded_like(view, seed):
    """Deterministic CPU-generated tensor matching view's shape/dtype.

    Generated on CPU so it is bit-identical across nodes/GPUs.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    rnd = torch.rand(view.shape, dtype=torch.float32, generator=g)
    return rnd.to(view.dtype)


def _request_block_views(mgr, rid, n_local_layers):
    """Yield (layer, buffer, valid_block_indices) for the request, per local layer.

    get_buffers() returns an aliasing view of the real KV pool (works for both V1
    and V2), so writes persist and reads see transferred data.
    """
    for layer in range(n_local_layers):
        blocks = mgr.get_batch_cache_indices([rid], layer)[0]
        valid = [b for b in blocks if b >= 0]
        if not valid:
            continue
        buf = mgr.get_buffers(layer, kv_layout="HND")
        yield layer, buf, valid


def fill_request(mgr, rid, rank, n_local_layers):
    for layer, buf, valid in _request_block_views(mgr, rid, n_local_layers):
        view = buf[valid]
        buf[valid] = _seeded_like(view, seed_for(rid, rank, layer)).to(view.device)


def verify_request(mgr, rid, rank, n_local_layers):
    for layer, buf, valid in _request_block_views(mgr, rid, n_local_layers):
        recv = buf[valid]
        exp = _seeded_like(recv, seed_for(rid, rank, layer)).to(recv.device)
        if not torch.equal(recv.float(), exp.float()):
            return False
    return True


def make_request(is_ctx, rid, req_len, runtime, ctx_params=None):
    """Build a ctx or gen LlmRequest.

    `ctx_params` (gen side only) is the ContextPhaseParams object produced by the
    ctx leader's respond_and_send_async(), shipped over ZMQ. Mirrors
    tests/unittest/others/test_kv_cache_transceiver.py:105-153.
    """
    sampling = SamplingParams()
    common = dict(
        request_id=rid,
        max_new_tokens=1,
        input_tokens=list(range(req_len)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(sampling._get_sampling_config()),
        is_streaming=False,
    )
    if is_ctx:
        req = LlmRequest(llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY, **common)
        if runtime == "PYTHON":
            req.py_disaggregated_params = DisaggregatedParams(
                request_type="context_only", disagg_request_id=rid
            )
        return req

    # gen side
    if runtime == "PYTHON":
        req = LlmRequest(llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY, **common)
        req.py_disaggregated_params = DisaggregatedParams(
            request_type="generation_only",
            disagg_request_id=rid,
            ctx_request_id=rid,
            ctx_dp_rank=ctx_params.ctx_dp_rank,
            ctx_info_endpoint=ctx_params.disagg_info_endpoint,
            first_gen_tokens=ctx_params.first_gen_tokens,
            draft_tokens=ctx_params.draft_tokens,
        )
    else:  # C++ transceiver: carry the ctx context phase params directly
        req = LlmRequest(
            llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
            context_phase_params=ctx_params,
            **common,
        )
    return req


class _TransferError(Exception):
    pass


def _wait_gen_complete(xcvr, req, runtime):
    """Block until this gen request's receive finishes (or errors).

    Block-all is the only safe wait here: returning while the receive is still
    in flight frees the request mid-transfer (gen hang + the ctx sender
    asserting on a freed session). How block-all is expressed depends on the
    transceiver:

    * PYTHON transceiver: check_gen_transfer_status(None) -> block_all.
    * C++ transceiver: the bound check_gen_transfer_status takes an int and
      returns as soon as >= N receives are *ready*; on a cold-start/slow link
      that can be BEFORE this request's transfer completes (some wheel builds
      also reject None outright with a TypeError). So poll the int API until the
      request reaches a terminal state. The per-cell signal.alarm and the hang
      detector bound this loop, so a genuinely stuck transfer is still caught.
    """
    if runtime == "PYTHON":
        xcvr.check_gen_transfer_status(None)  # block_all
        return
    import time

    terminal = (
        LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE,
        LlmRequestState.DISAGG_TRANS_ERROR,
    )
    while req.state not in terminal:
        xcvr.check_gen_transfer_status(1)
        if req.state in terminal:
            break
        time.sleep(0.001)


def run_one_request(
    role, comm, kvm, xcvr, runtime, use_v2, n_local_layers, rid, req_len, rank, zmq_sock
):
    """Transfer one request and verify it (gen side).

    The per-request ZMQ handshake is lockstep-safe: the ctx leader ALWAYS sends
    exactly one reply ("OK"+context_phase_params, or "ABORT"+reason) for each
    gen "go", so a ctx-side error never leaves the gen side blocked or the
    sockets out of sync. Errors (local exceptions or a transceiver-reported
    DISAGG_TRANS_ERROR) are raised so the caller records TRANSFER_ERROR.
    Returns the gen-side verification result (True/False), or None on ctx.
    """
    is_ctx = role == "ctx"
    is_leader = rank == 0

    if is_ctx:
        local_err = None
        req = kv_handle = None
        try:
            req = make_request(True, rid, req_len, runtime)
            kv_handle = add_sequence(kvm, req, req_len, use_v2)
            fill_request(kvm, req.py_request_id, rank, n_local_layers)
            tensorrt_llm.logger.info(
                f"[ctx r{rank}] rid={rid} len={req_len}: transfer START (send)"
            )
            xcvr.respond_and_send_async(req)
        except Exception as e:  # noqa: BLE001 - relay failure to gen, then raise
            local_err = e
        # Instance-wide consensus: every ctx rank must take the same branch at the
        # collective check below, even if only some ranks failed (e.g. a UCX
        # device error on a subset of GPUs). allreduce doubles as a barrier.
        any_failed = comm.allreduce(1 if local_err is not None else 0, op=MPI.MAX)
        if is_leader:
            zmq_sock.recv()  # gen leader's "go"
            if not any_failed:
                # context_phase_params is picklable and carries everything gen
                # needs (endpoint, ctx_dp_rank, first_gen/draft tokens).
                zmq_sock.send(pickle.dumps(("OK", req.context_phase_params)))
            else:
                reason = repr(local_err) if local_err is not None else "peer ctx rank failed"
                zmq_sock.send(pickle.dumps(("ABORT", reason)))
        if any_failed:
            if req is not None:
                try:
                    free_sequence(kvm, req, kv_handle, use_v2)
                except Exception:  # noqa: BLE001
                    pass
            raise local_err if local_err is not None else _TransferError("peer ctx rank failed")
        # Always block_all (None): with a request count, C++ only waits up to
        # kv_transfer_sender_future_timeout_ms (1000ms) and returns even if the
        # transfer is still in progress. NIXL/UCX cold-start connection setup can
        # exceed that, so the harness would free the request mid-transfer, leaving
        # the gen side hung and the ctx sender thread asserting on a freed session.
        xcvr.check_context_transfer_status(None)
        state = req.state
        tensorrt_llm.logger.info(f"[ctx r{rank}] rid={rid}: transfer DONE (send), state={state}")
        free_sequence(kvm, req, kv_handle, use_v2)
        if state == LlmRequestState.DISAGG_TRANS_ERROR:
            raise _TransferError("ctx transfer reported DISAGG_TRANS_ERROR")
        return None

    # gen side
    if is_leader:
        zmq_sock.send(b"go")
        status, payload = pickle.loads(zmq_sock.recv())
    else:
        status, payload = None, None
    status, payload = comm.bcast((status, payload), root=0)  # syncs gen ranks
    if status == "ABORT":
        raise _TransferError(f"ctx aborted: {payload}")
    ctx_params = payload

    local_err = None
    req = kv_handle = None
    try:
        req = make_request(False, rid, req_len, runtime, ctx_params=ctx_params)
        kv_handle = add_sequence(kvm, req, req_len, use_v2)
        tensorrt_llm.logger.info(f"[gen r{rank}] rid={rid} len={req_len}: transfer START (recv)")
        xcvr.request_and_receive_async(req)
    except Exception as e:  # noqa: BLE001
        local_err = e
    # Instance-wide consensus so all gen ranks take the same branch at check_gen.
    any_failed = comm.allreduce(1 if local_err is not None else 0, op=MPI.MAX)
    if any_failed:
        if req is not None:
            try:
                free_sequence(kvm, req, kv_handle, use_v2)
            except Exception:  # noqa: BLE001
                pass
        raise local_err if local_err is not None else _TransferError("peer gen rank failed")
    # Block until the receive actually completes (mirrors the ctx side) instead
    # of returning on the sender timeout. See _wait_gen_complete for why the C++
    # path polls an int rather than passing None.
    _wait_gen_complete(xcvr, req, runtime)
    # The receive may land on a side CUDA stream; sync before reading.
    torch.cuda.synchronize()

    state = req.state
    ok = state != LlmRequestState.DISAGG_TRANS_ERROR and verify_request(
        kvm, req.py_request_id, rank, n_local_layers
    )
    tensorrt_llm.logger.info(
        f"[gen r{rank}] rid={rid}: transfer DONE (recv), state={state}, "
        f"verify={'PASS' if ok else 'FAIL'}"
    )
    free_sequence(kvm, req, kv_handle, use_v2)
    if state == LlmRequestState.DISAGG_TRANS_ERROR:
        raise _TransferError("gen transfer reported DISAGG_TRANS_ERROR")
    return ok


def _preserve_cpp_csvs(csv_dir, ci, rank):
    """Rename THIS rank's C++ CSV files before the next combination truncates them.

    TRTLLM_KVCACHE_TIME_OUTPUT_PATH is cached by C++ on first read, so every C++
    transceiver instance writes the same filenames; one transceiver serves all
    request lengths of a combination and appends a row per request, so we move
    the whole combination's output aside (rid encodes req_len).

    Each rank touches ONLY its own files: all ranks share `csv_dir`, so a glob
    over `rank_*` would race -- multiple ranks renaming the same file, leaving
    some with FileNotFoundError, crashing those ranks and deadlocking the rest on
    the next case's collective KVCacheManager allreduce.
    """
    for tag in ("send", "recv"):
        path = os.path.join(csv_dir, f"rank_{rank}_{tag}.csv")
        if os.path.exists(path):
            os.replace(path, os.path.join(csv_dir, f"rank_{rank}_{tag}__c{ci}.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["ctx", "gen"])
    args = ap.parse_args()
    role = args.role
    is_ctx = role == "ctx"

    cfg_path = os.environ["CTT_CONFIG"]
    sweep = int(os.environ.get("CTT_SWEEP", "0"))
    sweep_name = os.environ.get("CTT_SWEEP_NAME", str(sweep))
    ctx_node = os.environ["CTX_NODE"]
    zmq_port = int(os.environ["ZMQ_PORT"])
    with open(cfg_path) as f:
        cfg = json.load(f) if cfg_path.endswith(".json") else yaml.safe_load(f)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n = comm.Get_size()
    n_cfg = int(cfg["hardware"]["gpus_per_node"])
    if n != n_cfg:
        raise RuntimeError(
            f"MPI world size {n} != gpus_per_node {n_cfg}; each srun step must be "
            f"its own MPI world of size N (see plan Risk #3)."
        )
    is_leader = rank == 0
    torch.cuda.set_device(rank % torch.cuda.device_count())

    tensorrt_llm.logger.set_level("info")
    ucx_env = {k: v for k, v in sorted(os.environ.items()) if k.startswith("UCX_")}
    ucx_env_str = " ".join(f"{k}={v}" for k, v in ucx_env.items()) or "<none>"
    print(f"[sweep={sweep_name} {role} rank={rank}] UCX env: {ucx_env_str}", flush=True)

    work_dir = cfg["environment"]["work_dir"]
    csv_dir = os.path.join(work_dir, "csv", str(sweep), role)
    os.makedirs(csv_dir, exist_ok=True)
    os.environ["TRTLLM_KVCACHE_TIME_OUTPUT_PATH"] = csv_dir
    # Python perf logging (singleton reads these once; C++ ignores them). Set at
    # startup so it is enabled regardless of combination ordering. Per-transceiver UUID
    # filenames (py_<uuid>_<rank>.csv) avoid cross-combination collisions.
    os.environ["TLLM_ENABLE_CACHE_TRANSFER_PERF_INFO"] = "1"
    os.environ["TLLM_KV_TRANSFER_PERF_LOG_FILE"] = os.path.join(csv_dir, "py")

    status_dir = os.path.join(work_dir, "status")
    os.makedirs(status_dir, exist_ok=True)
    status_path = os.path.join(status_dir, f"sweep{sweep}_{role}.jsonl")
    status_f = open(status_path, "a") if is_leader else None

    # ZMQ leader channel. A REQ/REP handshake interrupted by a timeout leaves the
    # socket unusable, so we reset it after any error to isolate the blast radius
    # to a single (combination, request_length) cell.
    zmq_ctx = None
    zmq_sock = None
    rcv_timeout_ms = cfg["run"]["timeout_per_cell_s"] * 1000

    def open_sock():
        if not is_leader:
            return None
        import time

        import zmq

        nonlocal zmq_ctx
        if zmq_ctx is None:
            zmq_ctx = zmq.Context.instance()
        if is_ctx:
            s = zmq_ctx.socket(zmq.REP)
            s.setsockopt(zmq.LINGER, 0)
            s.setsockopt(zmq.RCVTIMEO, rcv_timeout_ms)
            # Re-binding the same port right after a close can transiently hit
            # EADDRINUSE; retry briefly.
            for attempt in range(20):
                try:
                    s.bind(f"tcp://*:{zmq_port}")
                    break
                except zmq.error.ZMQError:
                    if attempt == 19:
                        raise
                    time.sleep(0.5)
        else:
            s = zmq_ctx.socket(zmq.REQ)
            s.setsockopt(zmq.LINGER, 0)
            s.setsockopt(zmq.RCVTIMEO, rcv_timeout_ms)
            s.connect(f"tcp://{ctx_node}:{zmq_port}")
        return s

    def reset_sock():
        nonlocal zmq_sock
        if not is_leader:
            return
        try:
            zmq_sock.close(linger=0)
        except Exception:  # noqa: BLE001
            pass
        zmq_sock = open_sock()

    zmq_sock = open_sock()

    cases = build_cases(cfg)
    req_lens = cfg["test_matrix"]["request_lengths"]
    warmup = cfg["test_matrix"]["warmup_requests"]
    num_req = cfg["test_matrix"]["num_requests_per_length"]
    cfg["kv_cache"]["_max_request_len"] = max(req_lens)

    tp = cfg["parallel"][f"{role}_tp"]
    pp = cfg["parallel"][f"{role}_pp"]
    mapping = Mapping(world_size=n, rank=rank, tp_size=tp, pp_size=pp, gpus_per_node=n)
    dist_obj = Distributed.get(mapping)
    n_local_layers = local_layer_count(cfg["kv_cache"]["num_layers"], pp, mapping.pp_rank)

    signal.signal(signal.SIGALRM, _alarm_handler)

    def record(combination_idx, reqlen_idx, status, reason=""):
        if not is_leader:
            return
        status_f.write(
            json.dumps(
                {
                    "combination_idx": combination_idx,
                    "reqlen_idx": reqlen_idx,
                    "status": status,
                    "reason": reason,
                }
            )
            + "\n"
        )
        status_f.flush()

    timeout_s = cfg["run"]["timeout_per_cell_s"]

    # In-process hang detector (TensorRT-LLM's HangDetector): a side thread runs
    # an asyncio timer reset per cell; on expiry it dumps all thread stacks
    # (print_all_stacks, great for seeing WHERE a UCX transfer wedged), records
    # TIMEOUT for the stuck cell, and SIGKILLs the process so `srun
    # --kill-on-bad-exit` tears down the sweep and the loop advances.
    #
    # LIMITATION: like any in-process timer, its callback needs the GIL, so it
    # CANNOT fire if the hang is in a native call that *holds* the GIL (e.g. the
    # UCX connection handshake inside respond_and_send_async /
    # request_and_receive_async, which have no gil_scoped_release). For those,
    # the GUARANTEED killer is the external `timeout -k <max_sweep_s>` around
    # each srun in launch.slurm. The detector only fires for GIL-released hangs
    # (e.g. check_*_transfer_status), where it also gives the stack dump.
    #
    # The deadline is clamped below max_sweep_s (the outer `timeout` cap) so the
    # detector, when it can fire, does so before the bash timeout and records
    # TIMEOUT status rather than the step being SIGKILLed with no record.
    max_sweep_s = int(cfg["run"].get("max_sweep_s", 300))
    watchdog_deadline = min(timeout_s + 30, max(30, max_sweep_s - 15))
    # Mutable holder so the fixed on_detected callback can attribute the hang to
    # the current cell. reqlen_idx=None marks all request lengths of the case
    # (used while building the transceiver).
    hang_cell = {"ci": 0, "li": None, "what": ""}

    def _on_hang():
        ci = hang_cell["ci"]
        targets = range(len(req_lens)) if hang_cell["li"] is None else [hang_cell["li"]]
        for li in targets:
            record(
                ci,
                li,
                "TIMEOUT",
                f"hang detected during {hang_cell['what']} (>{watchdog_deadline}s)",
            )
        sys.stderr.write(
            f"[{role} rank={rank}] WATCHDOG_KILL {hang_cell['what']} ci={ci} li={hang_cell['li']}\n"
        )
        sys.stderr.flush()
        os.kill(os.getpid(), signal.SIGKILL)

    hang_detector = HangDetector(timeout=watchdog_deadline, on_detected=_on_hang)
    hang_detector.start()

    def cancel_watchdog():
        hang_detector.cancel_task()

    def arm_watchdog(combination_idx, reqlen_idx, what):
        """(Re)arm the hang detector for one cell."""
        hang_cell["ci"] = combination_idx
        hang_cell["li"] = reqlen_idx
        hang_cell["what"] = what
        hang_detector.checkpoint()

    for ci, case in enumerate(cases):
        # Case boundary marker: report.py splits the per-rank logs on this to
        # attribute UCX_PROTO_INFO transport per (sweep, combination) instead of only
        # per sweep. One transceiver serves all request lengths of a case, so
        # transport is constant across req_len -- it varies only by sweep+combination.
        print(f"[CTT_CASE_BEGIN] ci={ci} label={case['label']}", flush=True)
        runtime = case["runtime"]
        backend = case["backend"]
        use_v2 = case["cache_manager"] == "V2"
        cache_cfg = CacheTransceiverConfig(
            backend=backend,
            transceiver_runtime=(None if runtime == "CPP" else "PYTHON"),
            max_tokens_in_buffer=cfg["kv_cache"]["max_tokens_in_buffer"],
        )

        # Build the cache manager + transceiver ONCE per case (the manager is
        # sized for the largest request length and serves all of them).
        kvm = xcvr = None
        setup_err = None
        try:
            signal.alarm(timeout_s)
            arm_watchdog(ci, None, f"setup {case['label']}")
            kvm = build_kv_cache_manager(cfg["kv_cache"], mapping, use_v2)
            xcvr = create_kv_cache_transceiver(
                mapping, dist_obj, kvm, AttentionTypeCpp.DEFAULT, cache_cfg
            )
            signal.alarm(0)
            cancel_watchdog()
        except Exception as e:  # noqa: BLE001 - setup failed for the whole case
            signal.alarm(0)
            cancel_watchdog()
            setup_err = e
            print(
                f"[{role} rank={rank}] SETUP ERROR {case['label']}: {e!r}",
                file=sys.stderr,
                flush=True,
            )
        # Instance-wide consensus: if ANY rank failed setup, every rank skips the
        # case together. Without this, a UCX init error on only some ranks/GPUs
        # would deadlock the instance's collectives (the real failure mode seen
        # with a bad UCX_TLS/UCX_NET_DEVICES on a subset of devices).
        if comm.allreduce(1 if setup_err is not None else 0, op=MPI.MAX):
            reason = (
                f"setup failed: {setup_err!r}"
                if setup_err is not None
                else "setup failed on another rank in the instance"
            )
            for li in range(len(req_lens)):
                record(ci, li, "TRANSFER_ERROR", reason)
            if xcvr is not None and hasattr(xcvr, "shutdown"):
                try:
                    xcvr.shutdown()
                except Exception:  # noqa: BLE001
                    pass
            del xcvr, kvm
            gc.collect()
            torch.cuda.empty_cache()
            continue

        case_timed_out = False
        for li, req_len in enumerate(req_lens):
            try:
                signal.alarm(timeout_s)
                arm_watchdog(ci, li, f"{case['label']} req_len={req_len}")
                all_ok = True
                for r in range(warmup + num_req):
                    rid = make_rid(ci, li, r)
                    ok = run_one_request(
                        role,
                        comm,
                        kvm,
                        xcvr,
                        runtime,
                        use_v2,
                        n_local_layers,
                        rid,
                        req_len,
                        rank,
                        zmq_sock,
                    )
                    if role == "gen" and r >= warmup and ok is False:
                        all_ok = False
                signal.alarm(0)
                cancel_watchdog()
                record(ci, li, "PASS" if (role != "gen" or all_ok) else "MISMATCH")
            except _Timeout:
                signal.alarm(0)
                cancel_watchdog()
                record(ci, li, "TIMEOUT", f"exceeded {timeout_s}s")
                for remaining_li in range(li + 1, len(req_lens)):
                    record(ci, remaining_li, "TIMEOUT", "skipped after timeout in earlier req_len")
                print(
                    f"[{role} rank={rank}] TIMEOUT {case['label']} req_len={req_len}",
                    file=sys.stderr,
                    flush=True,
                )
                reset_sock()
                case_timed_out = True
            except Exception as e:  # noqa: BLE001 - report any transceiver error
                signal.alarm(0)
                cancel_watchdog()
                record(ci, li, "TRANSFER_ERROR", repr(e))
                print(
                    f"[{role} rank={rank}] ERROR {case['label']} req_len={req_len}: {e!r}",
                    file=sys.stderr,
                    flush=True,
                )
            any_timed_out = comm.allreduce(1 if case_timed_out else 0, op=MPI.MAX)
            if any_timed_out:
                break

        # Tear down the case's transceiver and preserve its C++ CSVs.
        if hasattr(xcvr, "shutdown"):
            xcvr.shutdown()
        del xcvr, kvm
        gc.collect()
        torch.cuda.empty_cache()
        if runtime == "CPP":
            _preserve_cpp_csvs(csv_dir, ci, rank)
        # Resync all ranks before building the next case. The next case's
        # KVCacheManager does a collective MPI allreduce in its constructor; if
        # ranks entered it at different times (or one diverged) it would hang.
        comm.Barrier()

    cancel_watchdog()
    hang_detector.stop()
    if status_f:
        status_f.close()
    comm.Barrier()


if __name__ == "__main__":
    main()

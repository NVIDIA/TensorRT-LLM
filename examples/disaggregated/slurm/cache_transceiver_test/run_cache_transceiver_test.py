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
  C++  -> TRTLLM_KVCACHE_TIME_OUTPUT_PATH (<instanceId>_*_send.csv / <instanceId>_*_recv.csv)
  Py   -> same env var (PerfLogManager gives it top priority):
          <instanceUuid>_<rank>.csv, throughput_mbs column

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
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence

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
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import (
    create_kv_cache_transceiver,
    maybe_enable_fabric_memory_for_python_transceiver,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState, LlmRequestType
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.llmapi.llm_args import BlockReuseConfig, CacheTransceiverConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.sampling_params import SamplingParams

AttentionTypeCpp = tensorrt_llm.bindings.internal.batch_manager.AttentionType
CacheTypeCpp = tensorrt_llm.bindings.internal.batch_manager.CacheType
DataType = tensorrt_llm.bindings.DataType

# Must match report.py.
RID_COMBINATION_STRIDE = 1_000_000
RID_REQLEN_STRIDE = 10_000
TRANSFER_TIMEOUT_GRACE_SECONDS = 5
ABORT_COORDINATION_TIMEOUT_SECONDS = 2

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
    disk_cache_size: Optional[int] = None
    disk_cache_path: Optional[str] = None
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
    block_reuse_config: BlockReuseConfig = field(default_factory=BlockReuseConfig)
    enable_swa_scratch_reuse: bool = False
    disk_prefetch_num_reqs: int = 4
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
        if req.is_disagg_generation_init_state:
            ok = mgr.prepare_disagg_gen_init(req)
        else:
            ok = mgr.prepare_context(req) and mgr.resize_context(req, prompt_len)
        if not ok:
            raise RuntimeError(f"V2 KV cache allocation failed for request {req.py_request_id}")
        return None
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


class _FatalTransferError(_TransferError):
    """A transfer may still own KV pages, so this process must not reuse them."""


def _request_ids(values: Iterable[Any]) -> set[int]:
    return {value.py_request_id if hasattr(value, "py_request_id") else value for value in values}


def _context_completion_error(
    rid: int,
    completed: Iterable[Any],
    failed: Iterable[Any],
    state: Any,
    error_state: Any,
) -> Optional[str]:
    completed_rids = _request_ids(completed)
    failed_rids = _request_ids(failed)
    if rid in failed_rids:
        return f"ctx transfer failed for rid={rid}"
    if state == error_state:
        return f"ctx transfer reported DISAGG_TRANS_ERROR for rid={rid}"
    if rid not in completed_rids:
        return (
            f"ctx block-all returned without completing rid={rid}: "
            f"completed={sorted(completed_rids)} failed={sorted(failed_rids)}"
        )
    return None


def _gen_completion_error(
    rid: int,
    completed: Iterable[Any],
    failed: Iterable[Any],
    cancelled: Iterable[Any],
    state: Any,
    complete_state: Any,
    error_state: Any,
) -> Optional[str]:
    completed_rids = _request_ids(completed)
    failed_rids = _request_ids(failed)
    cancelled_rids = _request_ids(cancelled)
    if rid in failed_rids:
        return f"gen transfer failed for rid={rid}"
    if rid in cancelled_rids:
        return f"gen transfer cancelled for rid={rid}"
    if state == error_state:
        return f"gen transfer reported DISAGG_TRANS_ERROR for rid={rid}"
    if rid not in completed_rids:
        return (
            f"gen block-all returned without completing rid={rid}: "
            f"completed={sorted(completed_rids)} failed={sorted(failed_rids)} "
            f"cancelled={sorted(cancelled_rids)}"
        )
    if state != complete_state:
        return f"gen transfer returned rid={rid} as complete with nonterminal state={state}"
    return None


def _can_release_sequence(transfer_may_have_started: bool, transfer_completed: bool) -> bool:
    """Return whether the transceiver has relinquished its ownership of the pages."""
    return not transfer_may_have_started or transfer_completed


def _release_sequence_if_safe(
    mgr: Any,
    req: Any,
    kv_handle: Any,
    use_v2: bool,
    *,
    transfer_may_have_started: bool,
    transfer_completed: bool,
) -> bool:
    if req is None or not _can_release_sequence(transfer_may_have_started, transfer_completed):
        return False
    free_sequence(mgr, req, kv_handle, use_v2)
    return True


def _validate_context_completion(req: Any, status: Optional[Sequence[Any]]) -> None:
    if status is None or len(status) != 2:
        raise _FatalTransferError(
            f"ctx block-all returned invalid status for rid={req.py_request_id}"
        )
    completed, failed = status
    error = _context_completion_error(
        req.py_request_id,
        completed,
        failed,
        req.state,
        LlmRequestState.DISAGG_TRANS_ERROR,
    )
    if error is not None:
        raise _FatalTransferError(error)


def _validate_python_gen_completion(req: Any, status: Optional[Sequence[Any]]) -> None:
    if status is None or len(status) != 3:
        raise _FatalTransferError(
            f"gen block-all returned invalid status for rid={req.py_request_id}"
        )
    completed, failed, cancelled = status
    error = _gen_completion_error(
        req.py_request_id,
        completed,
        failed,
        cancelled,
        req.state,
        LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE,
        LlmRequestState.DISAGG_TRANS_ERROR,
    )
    if error is not None:
        raise _FatalTransferError(error)


def _first_reason(reasons: Iterable[Optional[str]], fallback: str) -> str:
    return next((reason for reason in reasons if reason), fallback)


def _exchange_release_decision(
    role: str,
    comm: Any,
    is_leader: bool,
    zmq_sock: Any,
    local_safe: bool,
    local_reason: str = "",
) -> tuple[bool, str]:
    """Agree across both MPI roles before either side releases KV pages."""
    role_safe = bool(comm.allreduce(1 if local_safe else 0, op=MPI.MIN))
    gathered_reasons = comm.gather(local_reason or None, root=0)
    decision = None
    if is_leader:
        role_reason = (
            ""
            if role_safe
            else _first_reason(
                gathered_reasons,
                f"{role} peer rank did not prove transfer completion",
            )
        )
        local_status = "COMPLETE" if role_safe else "FATAL"
        try:
            if role == "gen":
                zmq_sock.send(pickle.dumps((local_status, role_reason)))
            peer_status, peer_reason = pickle.loads(zmq_sock.recv())
            peer_role = "ctx" if role == "gen" else "gen"
            if peer_status not in ("COMPLETE", "FATAL"):
                combined_safe = False
                combined_reason = f"invalid {peer_role} peer release status: {peer_status!r}"
            elif role == "gen":
                combined_safe = role_safe and peer_status == "COMPLETE"
                combined_reason = "" if combined_safe else str(peer_reason or role_reason)
            else:
                combined_safe = role_safe and peer_status == "COMPLETE"
                combined_reason = (
                    ""
                    if combined_safe
                    else role_reason
                    if not role_safe
                    else str(peer_reason or "gen did not prove transfer completion")
                )
            if role != "gen":
                zmq_sock.send(
                    pickle.dumps(("COMPLETE" if combined_safe else "FATAL", combined_reason))
                )
        except _Timeout:
            raise
        except Exception as e:  # noqa: BLE001 - converted to process-fatal ownership outcome
            combined_safe = False
            combined_reason = f"final release handshake failed: {e!r}"
        decision = (combined_safe, combined_reason)
    return comm.bcast(decision, root=0)


def _hard_abort_process(comm: Any) -> None:
    """Terminate without running transceiver/KV-manager finalizers."""
    try:
        comm.Abort(137)
    except Exception:  # noqa: BLE001 - SIGKILL is the mandatory fallback
        pass
    os.kill(os.getpid(), signal.SIGKILL)
    raise RuntimeError("SIGKILL unexpectedly returned")


def _coordinate_abort_after_leader_flush(comm: Any) -> None:
    """Best-effort bounded rendezvous after the leader persists diagnostics."""
    try:
        request = comm.Ibarrier()
        deadline = time.monotonic() + ABORT_COORDINATION_TIMEOUT_SECONDS
        while not request.Test():
            if time.monotonic() >= deadline:
                return
            time.sleep(0.01)
    except Exception:  # noqa: BLE001 - abort must remain the fallback
        return


def _wait_gen_complete(xcvr: Any, req: LlmRequest, runtime: str) -> None:
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
        status = xcvr.check_gen_transfer_status(None)  # block_all
        _validate_python_gen_completion(req, status)
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
    if req.state == LlmRequestState.DISAGG_TRANS_ERROR:
        raise _FatalTransferError(
            f"gen transfer reported DISAGG_TRANS_ERROR for rid={req.py_request_id}"
        )


def run_one_request(
    role, comm, kvm, xcvr, runtime, use_v2, n_local_layers, rid, req_len, rank, zmq_sock
):
    """Transfer one request and verify it (gen side).

    The leaders exchange setup and final ownership decisions over ZMQ. Neither
    role releases its sequence until every local rank and the peer role report
    successful completion. A result without quiescence proof raises
    _FatalTransferError so the caller hard-aborts without running finalizers.
    Returns the gen-side verification result (True/False), or None on ctx.
    """
    is_ctx = role == "ctx"
    is_leader = rank == 0

    if is_ctx:
        local_err = None
        req = kv_handle = None
        transfer_may_have_started = False
        try:
            req = make_request(True, rid, req_len, runtime)
            kv_handle = add_sequence(kvm, req, req_len, use_v2)
            fill_request(kvm, req.py_request_id, rank, n_local_layers)
            tensorrt_llm.logger.info(
                f"[ctx r{rank}] rid={rid} len={req_len}: transfer START (send)"
            )
            # The call can dispatch KV work before raising, so ownership becomes
            # uncertain as soon as we enter it.
            transfer_may_have_started = True
            xcvr.respond_and_send_async(req)
        except _Timeout:
            raise
        except Exception as e:  # noqa: BLE001 - relayed to gen, then classified
            local_err = e

        any_failed = comm.allreduce(1 if local_err is not None else 0, op=MPI.MAX)
        any_started = comm.allreduce(1 if transfer_may_have_started else 0, op=MPI.MAX)
        handshake_error = None
        if is_leader:
            try:
                message = zmq_sock.recv()  # gen leader's "go"
                if message != b"go":
                    raise _TransferError(f"unexpected gen handshake message: {message!r}")
                if not any_failed:
                    # context_phase_params carries the endpoint and generation metadata.
                    zmq_sock.send(pickle.dumps(("OK", req.context_phase_params)))
                else:
                    reason = repr(local_err) if local_err is not None else "peer ctx rank failed"
                    status = "FATAL" if any_started else "ABORT"
                    zmq_sock.send(pickle.dumps((status, reason)))
            except _Timeout:
                raise
            except Exception as e:  # noqa: BLE001 - broadcast before raising
                handshake_error = repr(e)
        handshake_error = comm.bcast(handshake_error, root=0)
        if handshake_error is not None:
            raise _FatalTransferError(f"initial ctx/gen handshake failed: {handshake_error}")
        if any_failed:
            reason = repr(local_err) if local_err is not None else "peer ctx rank failed"
            if any_started:
                raise _FatalTransferError(reason)
            _release_sequence_if_safe(
                kvm,
                req,
                kv_handle,
                use_v2,
                transfer_may_have_started=False,
                transfer_completed=False,
            )
            raise _TransferError(reason)

        transfer_error = None
        try:
            # block_all retries short polling slices until completion or the
            # request-level KV-transfer deadline. Only the exact completed ID
            # proves that the sender has stopped reading this request's pages.
            status = xcvr.check_context_transfer_status(None)
            _validate_context_completion(req, status)
        except _Timeout:
            raise
        except Exception as e:  # noqa: BLE001 - becomes a cross-role fatal decision
            transfer_error = e

        safe_to_release, reason = _exchange_release_decision(
            role,
            comm,
            is_leader,
            zmq_sock,
            local_safe=transfer_error is None,
            local_reason=repr(transfer_error) if transfer_error is not None else "",
        )
        if not safe_to_release:
            raise _FatalTransferError(reason)
        state = req.state
        tensorrt_llm.logger.info(f"[ctx r{rank}] rid={rid}: transfer DONE (send), state={state}")
        free_sequence(kvm, req, kv_handle, use_v2)
        return None

    # gen side
    initial_error = None
    if is_leader:
        try:
            zmq_sock.send(b"go")
            status, payload = pickle.loads(zmq_sock.recv())
        except _Timeout:
            raise
        except Exception as e:  # noqa: BLE001 - ctx may already own in-flight pages
            status, payload = "FATAL", f"initial ctx/gen handshake failed: {e!r}"
            initial_error = repr(e)
    else:
        status, payload = None, None
    status, payload, initial_error = comm.bcast(
        (status, payload, initial_error), root=0
    )  # syncs gen ranks
    if initial_error is not None or status == "FATAL":
        raise _FatalTransferError(str(payload))
    if status == "ABORT":
        raise _TransferError(f"ctx aborted: {payload}")
    if status != "OK":
        raise _FatalTransferError(f"unexpected ctx handshake status: {status!r}")
    ctx_params = payload

    local_err = None
    req = kv_handle = None
    try:
        req = make_request(False, rid, req_len, runtime, ctx_params=ctx_params)
        kv_handle = add_sequence(kvm, req, req_len, use_v2)
        tensorrt_llm.logger.info(f"[gen r{rank}] rid={rid} len={req_len}: transfer START (recv)")
        # The context sender is already live, and this call may partially
        # dispatch before raising. Any setup error from here is process-fatal.
        xcvr.request_and_receive_async(req)
    except _Timeout:
        raise
    except Exception as e:  # noqa: BLE001
        local_err = e

    any_failed = comm.allreduce(1 if local_err is not None else 0, op=MPI.MAX)
    transfer_error = local_err
    if not any_failed:
        try:
            # See _wait_gen_complete for why the C++ path polls an int rather
            # than passing None. Python validates the exact returned request ID.
            _wait_gen_complete(xcvr, req, runtime)
        except _Timeout:
            raise
        except Exception as e:  # noqa: BLE001 - becomes a cross-role fatal decision
            transfer_error = e
    elif transfer_error is None:
        transfer_error = _FatalTransferError("peer gen rank failed during receive setup")

    if transfer_error is None:
        try:
            # A terminal transceiver result is not sufficient ownership proof:
            # wait for any side-stream work touching the received pages before
            # either role is allowed to release its sequence.
            torch.cuda.synchronize()
        except _Timeout:
            raise
        except Exception as e:  # noqa: BLE001 - quiescence remains unproven
            transfer_error = e

    safe_to_release, reason = _exchange_release_decision(
        role,
        comm,
        is_leader,
        zmq_sock,
        local_safe=transfer_error is None,
        local_reason=repr(transfer_error) if transfer_error is not None else "",
    )
    if not safe_to_release:
        raise _FatalTransferError(reason)

    # Transfer completion and CUDA synchronization have released transport
    # ownership. Verification errors are now safe mismatches: retain lockstep,
    # free the sequence, and keep testing.
    try:
        local_ok = verify_request(kvm, req.py_request_id, rank, n_local_layers)
    except _Timeout:
        raise
    except Exception as e:  # noqa: BLE001 - transfer is already proven quiescent
        tensorrt_llm.logger.error(f"[gen r{rank}] rid={rid}: verification error: {e!r}")
        local_ok = False
    ok = bool(comm.allreduce(1 if local_ok else 0, op=MPI.MIN))
    state = req.state
    tensorrt_llm.logger.info(
        f"[gen r{rank}] rid={rid}: transfer DONE (recv), state={state}, "
        f"verify={'PASS' if ok else 'FAIL'}"
    )
    free_sequence(kvm, req, kv_handle, use_v2)
    return ok


def _preserve_cpp_csvs(csv_dir, ci, rank):
    """Rename THIS rank's C++ CSV files before the next combination truncates them.

    TRTLLM_KVCACHE_TIME_OUTPUT_PATH is cached by C++ on first read, so every C++
    transceiver instance writes the same filenames; one transceiver serves all
    request lengths of a combination and appends a row per request, so we move
    the whole combination's output aside (rid encodes req_len).

    C++ names files "<instanceId>_<rank>_<tag>.csv" (instanceId is a runtime
    UUID). Each rank touches ONLY files carrying its own "_<rank>_<tag>.csv"
    suffix: all ranks share `csv_dir`, so matching a broader pattern would race
    -- multiple ranks renaming the same file, leaving some with
    FileNotFoundError, crashing those ranks and deadlocking the rest on the next
    case's collective KVCacheManager allreduce.
    """
    for tag in ("send", "recv"):
        suffix = f"_{rank}_{tag}.csv"
        for name in os.listdir(csv_dir):
            if name.endswith(suffix) and "__c" not in name:
                base = name[: -len(".csv")]
                os.replace(os.path.join(csv_dir, name), os.path.join(csv_dir, f"{base}__c{ci}.csv"))


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
    # One env var drives both transceivers: C++ caches it on first read, and
    # Python's PerfLogManager gives it top priority (enabling perf logging and
    # writing task CSVs as <instanceUuid>_<rank>.csv in csv_dir; the
    # per-transceiver UUID avoids cross-combination collisions, and report.py
    # identifies these files by header columns, not name).
    os.environ["TRTLLM_KVCACHE_TIME_OUTPUT_PATH"] = csv_dir

    status_dir = os.path.join(work_dir, "status")
    os.makedirs(status_dir, exist_ok=True)
    status_path = os.path.join(status_dir, f"sweep{sweep}_{role}.jsonl")
    status_f = open(status_path, "a") if is_leader else None
    timeout_s = int(cfg["run"]["timeout_per_cell_s"])
    if timeout_s <= TRANSFER_TIMEOUT_GRACE_SECONDS:
        raise ValueError(
            "run.timeout_per_cell_s must be greater than "
            f"{TRANSFER_TIMEOUT_GRACE_SECONDS}s to leave ownership-handshake headroom"
        )

    # ZMQ leader channel. Transfer-ownership failures hard-abort this sweep, so
    # an interrupted REQ/REP exchange is never reused in-process.
    zmq_ctx = None
    zmq_sock = None
    rcv_timeout_ms = timeout_s * 1000

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

    zmq_sock = open_sock()

    cases = build_cases(cfg)
    # The C++ fabric-memory env getter is cached on its first KV pool
    # allocation. Enable the default before any matrix case builds a pool, even
    # when a C++ transceiver case appears before Python+V1 in the matrix.
    python_v1_case = next(
        (case for case in cases if case["runtime"] == "PYTHON" and case["cache_manager"] == "V1"),
        None,
    )
    if python_v1_case is not None:
        maybe_enable_fabric_memory_for_python_transceiver(
            CacheTransceiverConfig(
                backend=python_v1_case["backend"],
                transceiver_runtime="PYTHON",
            ),
            KVCacheManager,
        )
        print(
            f"[{role} rank={rank}] PYTHON+V1 case in matrix: "
            "TRTLLM_KVCACHE_POOL_USE_FABRIC_MEMORY="
            f"{os.environ.get('TRTLLM_KVCACHE_POOL_USE_FABRIC_MEMORY')} "
            "applies to every case in this run, including C++ transceiver ones",
            flush=True,
        )
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

    def record_remaining(
        combination_idx: int,
        reqlen_idx: Optional[int],
        status: str,
        reason: str,
    ) -> None:
        """Record every unrun cell before a fatal sweep abort."""
        for remaining_ci in range(combination_idx, len(cases)):
            start_li = reqlen_idx if remaining_ci == combination_idx else 0
            if start_li is None:
                start_li = 0
            for remaining_li in range(start_li, len(req_lens)):
                cell_reason = (
                    reason
                    if remaining_ci == combination_idx
                    and (reqlen_idx is None or remaining_li == reqlen_idx)
                    else f"skipped after fatal transfer outcome: {reason}"
                )
                record(remaining_ci, remaining_li, status, cell_reason)

    def flush_status() -> None:
        if status_f is None or status_f.closed:
            return
        status_f.flush()
        try:
            os.fsync(status_f.fileno())
        except OSError:
            pass
        status_f.close()

    def hard_abort_sweep(
        combination_idx: int,
        reqlen_idx: Optional[int],
        status: str,
        reason: str,
        *,
        coordinated: bool = False,
    ) -> None:
        """Record the failure and exit without deregistering live transfer memory."""
        try:
            signal.alarm(0)
            cancel_watchdog()
        except Exception:  # noqa: BLE001 - abort must still happen
            pass
        if is_leader:
            try:
                record_remaining(combination_idx, reqlen_idx, status, reason)
                flush_status()
            except Exception:  # noqa: BLE001 - abort must still happen
                pass
        if coordinated:
            # All ranks reach this path after an intra-role consensus/broadcast.
            # The bounded collective prevents a nonleader from aborting before
            # the leader's status file is durable, without trusting consensus
            # in rank-local timeout/unknown-exception paths.
            _coordinate_abort_after_leader_flush(comm)
        elif not is_leader:
            # Rank-local alarms/exceptions cannot safely enter a collective.
            # Give the leader's independently armed deadline a short chance to
            # persist the role verdict before this rank aborts the srun step.
            time.sleep(ABORT_COORDINATION_TIMEOUT_SECONDS)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            _hard_abort_process(comm)

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
        reason = f"hang detected during {hang_cell['what']} (>{watchdog_deadline}s)"
        if is_leader:
            try:
                record_remaining(ci, hang_cell["li"], "TIMEOUT", reason)
                flush_status()
            except Exception:  # noqa: BLE001 - SIGKILL must still happen
                pass
        else:
            # All ranks arm the same cell deadline. Give the leader's watchdog
            # a bounded opportunity to persist the role status before this
            # rank's bad exit causes srun to terminate the whole step.
            time.sleep(ABORT_COORDINATION_TIMEOUT_SECONDS)
        try:
            sys.stderr.write(
                f"[{role} rank={rank}] WATCHDOG_KILL {hang_cell['what']} "
                f"ci={ci} li={hang_cell['li']}\n"
            )
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
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
            # PYTHON (V2) only; 0 keeps bounce off. With bounce on, the KV data
            # rides a fabric-VMM staging buffer (CU_MEM_HANDLE_TYPE_FABRIC), which
            # is what lets UCX pick cuda_ipc across NVL72 nodes -- direct
            # pool-to-pool transfers from non-fabric allocations fall back to
            # much slower host-staged tcp, so enable bounce for cross-node
            # transfers inside an NVLink domain.
            kv_cache_bounce_size_mb=int(cfg["kv_cache"].get("bounce_size_mb", 0)),
            # For the Python sender, leave deterministic headroom for the
            # signal handler and final CTX/GEN ownership handshake after its
            # request deadline. The cell alarm remains the C++ path's bound.
            kv_transfer_timeout_ms=(timeout_s - TRANSFER_TIMEOUT_GRACE_SECONDS) * 1000,
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
        except _Timeout:
            reason = f"setup exceeded {timeout_s}s with ownership state unknown"
            try:
                print(
                    f"[{role} rank={rank}] SETUP TIMEOUT {case['label']}",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception:  # noqa: BLE001 - abort must still happen
                pass
            hard_abort_sweep(ci, None, "TIMEOUT", reason)
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
            except _FatalTransferError as e:
                try:
                    print(
                        f"[{role} rank={rank}] FATAL {case['label']} req_len={req_len}: {e!r}",
                        file=sys.stderr,
                        flush=True,
                    )
                except Exception:  # noqa: BLE001 - abort must still happen
                    pass
                hard_abort_sweep(
                    ci,
                    li,
                    "TRANSFER_ERROR",
                    repr(e),
                    coordinated=True,
                )
            except _Timeout:
                reason = f"exceeded {timeout_s}s with transfer quiescence unproven"
                try:
                    print(
                        f"[{role} rank={rank}] TIMEOUT {case['label']} req_len={req_len}",
                        file=sys.stderr,
                        flush=True,
                    )
                except Exception:  # noqa: BLE001 - abort must still happen
                    pass
                hard_abort_sweep(ci, li, "TIMEOUT", reason)
            except _TransferError as e:
                signal.alarm(0)
                cancel_watchdog()
                record(ci, li, "TRANSFER_ERROR", repr(e))
                print(
                    f"[{role} rank={rank}] ERROR {case['label']} req_len={req_len}: {e!r}",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001 - unknown ownership state is fatal
                try:
                    print(
                        f"[{role} rank={rank}] UNEXPECTED {case['label']} req_len={req_len}: {e!r}",
                        file=sys.stderr,
                        flush=True,
                    )
                except Exception:  # noqa: BLE001 - abort must still happen
                    pass
                hard_abort_sweep(ci, li, "TRANSFER_ERROR", repr(e))

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

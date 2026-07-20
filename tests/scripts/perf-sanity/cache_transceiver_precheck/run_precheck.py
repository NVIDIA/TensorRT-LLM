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
"""Per-rank driver for the disagg perf-sanity cache-transceiver precheck.

Runs BEFORE the real disaggregated perf-sanity servers, with the SAME
instance layout (one MPI world per ctx/gen server, same node/GPU topology),
the SAME UCX environment (the launch script reuses the worker env prefix
verbatim), and the SAME `cache_transceiver_config` (built directly from the
test's disagg yaml). It allocates a KV cache shaped like the real model,
transfers deterministic data ctx -> gen through the transceiver for every
(ctx server, gen server) pair, verifies the received bytes, and fails the
stage with a specific error (TIMEOUT / TRANSFER_ERROR / MISMATCH / ...)
before any model is loaded, so network/UCX misconfiguration is caught in
minutes instead of after a full model bring-up.

Asymmetric parallelism (e.g. ctx dep4 -> gen dep16, ctx pp8 -> gen tp32) is
supported: the fill pattern is seeded per (request, GLOBAL layer) and is
constant along the KV-head axis, so any TP resharding or PP re-splitting on
the receiving side regenerates the identical expected bytes locally. (This
deliberately cannot detect head-permutation bugs -- it is a network
precheck, not a transceiver-correctness test.)

Derived from examples/disaggregated/slurm/cache_transceiver_test (the UCX
tuning harness), reduced to a single go/no-go sweep and extended to
asymmetric layouts, attention DP, and multi-instance pairing.

Rendezvous is file-based under --work-dir (a shared filesystem): each ctx
leader binds one ZMQ REP socket per gen peer and publishes host:port plus a
per-session HMAC key in rendezvous/ctx{ci}_gen{gj}.addr; gen leaders connect
with REQ sockets. Only tiny control payloads travel over ZMQ -- KV data goes
through the transceiver under test.

Control messages are JSON with an appended HMAC-SHA256 tag -- NEVER pickle:
the REP port is reachable from the cluster network, and unpickling
network-supplied bytes is arbitrary code execution. The key travels only via
the work-dir addr file (filesystem permissions = the job's trust domain), so
a network-only attacker can neither read it nor forge/tamper messages.
ContextPhaseParams crosses the wire as its primitive fields (opaque_state
base64-encoded), mirroring DisaggregatedParams <-> ContextPhaseParams in
tensorrt_llm/disaggregated_params.py and executor/result.py.
"""

import argparse
import base64
import dataclasses
import hashlib
import hmac
import json
import os
import secrets
import signal
import socket
import sys
import time

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

import precheck_config as pcfg  # noqa: E402

# Request-id strides (must keep rids unique across the whole precheck).
RID_PAIR_STRIDE = 1
RID_MAX_PAIRS = 4096
RID_REP_STRIDE = RID_MAX_PAIRS
RID_MAX_REPS = 64
RID_LEN_STRIDE = RID_REP_STRIDE * RID_MAX_REPS
RID_MAX_LENS = 64
RID_PEER_STRIDE = RID_LEN_STRIDE * RID_MAX_LENS


class _Timeout(Exception):
    pass


class _TransferError(Exception):
    pass


class _PeerAbort(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


def make_rid(ctx_idx, gen_idx, num_ctx, li, rep, pair):
    peer = gen_idx * num_ctx + ctx_idx
    return 1 + peer * RID_PEER_STRIDE + li * RID_LEN_STRIDE + rep * RID_REP_STRIDE + pair


def seed_for(rid, global_layer):
    # Per (request, GLOBAL layer); rank-independent so any receiving layout
    # can regenerate its local slice.
    return (rid * 1_000_003 + global_layer * 31) & 0x7FFFFFFF


# --------------------------------------------------------------------------- #
# Control-channel wire format: HMAC-SHA256-authenticated JSON (never pickle --
# the ZMQ port is reachable from the cluster network).
# --------------------------------------------------------------------------- #
_HMAC_TAG_LEN = hashlib.sha256().digest_size


def pack_msg(obj, key):
    """JSON-encode `obj` and append an HMAC-SHA256 tag."""
    data = json.dumps(obj, separators=(",", ":")).encode()
    return data + hmac.new(key, data, hashlib.sha256).digest()


def unpack_msg(raw, key):
    """Verify the HMAC tag, then JSON-decode. Raises _TransferError on forgery."""
    if len(raw) <= _HMAC_TAG_LEN:
        raise _TransferError(f"control frame too short ({len(raw)} bytes)")
    data, tag = raw[:-_HMAC_TAG_LEN], raw[-_HMAC_TAG_LEN:]
    if not hmac.compare_digest(hmac.new(key, data, hashlib.sha256).digest(), tag):
        raise _TransferError("control frame failed HMAC verification (tampered or wrong key)")
    return json.loads(data)


def params_to_wire(p):
    """ContextPhaseParams -> JSON-safe dict (fields per executor/result.py)."""
    return {
        "first_gen_tokens": list(p.first_gen_tokens or []),
        "req_id": p.req_id,
        "opaque_state": base64.b64encode(p.opaque_state or b"").decode(),
        "draft_tokens": list(p.draft_tokens) if p.draft_tokens is not None else None,
        "ctx_dp_rank": p.ctx_dp_rank,
        "ctx_info_endpoint": p.disagg_info_endpoint,
    }


def params_from_wire(d):
    """Inverse of params_to_wire (ctor per disaggregated_params.py)."""
    import tensorrt_llm.bindings.executor as tllme

    return tllme.ContextPhaseParams(
        list(d["first_gen_tokens"]),
        int(d["req_id"]),
        base64.b64decode(d["opaque_state"]),
        d["draft_tokens"],
        d["ctx_dp_rank"],
        d["ctx_info_endpoint"],
    )


# --------------------------------------------------------------------------- #
# KV fill / verify (heavy imports stay inside functions: --dry-run and unit
# tests must work without torch / tensorrt_llm installed)
# --------------------------------------------------------------------------- #
def _pattern_like(view, seed):
    """Deterministic tensor matching `view`, constant along the head axis.

    `view` is an HND block slice: [nblocks, kv_factor, heads, tokens, dim].
    Generated on CPU (bit-identical across nodes), head axis generated as 1
    and expanded, so ctx/gen sides with different local head counts (TP
    resharding) or different local layer sets (PP re-splitting) still agree.
    """
    import torch

    nb, kv, heads, tok, dim = view.shape
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    rnd = torch.rand((nb, kv, 1, tok, dim), dtype=torch.float32, generator=g)
    return rnd.to(view.dtype).expand(nb, kv, heads, tok, dim)


def _request_block_views(kvm, rid):
    """Yield (global_layer, buffer, valid_block_indices) for this rank."""
    for global_layer in kvm.pp_layers:
        blocks = kvm.get_batch_cache_indices([rid], layer_idx=global_layer)[0]
        valid = [b for b in blocks if b >= 0]
        if not valid:
            continue
        buf = kvm.get_buffers(global_layer, kv_layout="HND")
        yield global_layer, buf, valid


def fill_request(kvm, rid):
    for global_layer, buf, valid in _request_block_views(kvm, rid):
        view = buf[valid]
        buf[valid] = _pattern_like(view, seed_for(rid, global_layer)).to(view.device)


def verify_request(kvm, rid):
    """Returns (ok, detail) comparing received blocks to the expected pattern."""
    import torch

    for global_layer, buf, valid in _request_block_views(kvm, rid):
        recv = buf[valid]
        exp = _pattern_like(recv, seed_for(rid, global_layer)).to(recv.device)
        if not torch.equal(recv.float(), exp.float()):
            bad = (recv.float() != exp.float()).sum().item()
            return False, f"layer={global_layer} mismatched_elements={bad}/{recv.numel()}"
    return True, ""


@dataclasses.dataclass
class _KvCacheConfigV2:
    """KvCacheConfig stand-in for KVCacheManagerV2.

    Mirrors examples/disaggregated/slurm/cache_transceiver_test (and the
    reference dataclass in
    tests/unittest/disaggregated/test_cache_transceiver_single_process.py):
    KVCacheManagerV2 reads these fields off the config object directly, so
    every attribute it accesses must exist here.
    """

    max_tokens: "int | None" = None
    enable_block_reuse: bool = False
    max_attention_window: "list | None" = None
    sink_token_length: "int | None" = None
    free_gpu_memory_fraction: "float | None" = None
    host_cache_size: "int | None" = None
    disk_cache_size: "int | None" = None
    disk_cache_path: "str | None" = None
    onboard_blocks: bool = True
    cross_kv_cache_fraction: "float | None" = None
    secondary_offload_min_priority: "int | None" = None
    event_buffer_max_size: int = 0
    kv_cache_event_hash_algo: str = "auto"
    max_gpu_total_bytes: "int | None" = None
    enable_partial_reuse: bool = False
    copy_on_partial_reuse: bool = False
    dtype: str = "auto"
    pool_ratio: "list | None" = None
    avg_seq_len: "int | None" = None
    block_reuse_policy: str = "all_reusable"
    enable_swa_scratch_reuse: bool = False
    disk_prefetch_num_reqs: int = 4
    max_util_for_resume: float = 0.95


def _lookup_model_cls(model_dir):
    """Model class from config.json architectures, like serving's automodel path."""
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            hf_cfg = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return None, None
    archs = hf_cfg.get("architectures") or []
    hf_view = type("HFConfigView", (), hf_cfg)  # attribute access for the pref hook
    if not archs:
        return None, hf_view
    import tensorrt_llm._torch.models  # noqa: F401 - populates the registry

    from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING

    return MODEL_CLASS_MAPPING.get(archs[0]), hf_view


def resolve_model_prefs(model_dir, side, cache_cfg):
    """Mirror serving's model-preference resolution (PR #15823 semantics).

    - use_kv_cache_manager_v2 == "auto" (yaml absent): adopt the model
      class's get_model_defaults() value, default False
      (llm_utils._resolve_kv_cache_manager_v2_auto).
    - cache_cfg.transceiver_runtime == "auto": adopt
      model_cls.get_preferred_transceiver_runtime(), NIXL-gated, via the
      REAL llm_utils._resolve_transceiver_runtime_auto (mutates cache_cfg).

    Returns the effective use_v2 bool.
    """
    import types

    model_cls, hf_view = _lookup_model_cls(model_dir)

    setting = side["use_kv_cache_manager_v2"]
    if setting == "auto":
        defaults = {}
        if model_cls is not None:
            try:
                defaults = model_cls.get_model_defaults(None) or {}
            except Exception as e:  # noqa: BLE001 - model hooks may need llm_args
                print(f"[precheck] WARNING: get_model_defaults failed ({e!r}); "
                      f"assuming V1", flush=True)
        kv_defaults = defaults.get("kv_cache_config") or {}
        use_v2 = kv_defaults.get("use_kv_cache_manager_v2", False) is True
    else:
        use_v2 = bool(setting)

    if getattr(cache_cfg, "transceiver_runtime", None) == "auto":
        try:
            from tensorrt_llm.llmapi.llm_utils import _resolve_transceiver_runtime_auto

            shim = types.SimpleNamespace(cache_transceiver_config=cache_cfg)
            _resolve_transceiver_runtime_auto(shim, model_cls, hf_view)
        except Exception as e:  # noqa: BLE001 - fall back to the create() default (CPP)
            print(
                f"[precheck] WARNING: transceiver_runtime 'auto' resolution failed "
                f"({e!r}); create_kv_cache_transceiver will fall back to CPP",
                flush=True,
            )
    return use_v2


def build_kv_cache_manager(kv_shape, plan, side, mapping, max_req_len, use_v2):
    import tensorrt_llm.bindings
    import tensorrt_llm.bindings.executor as trtllm_executor
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager

    DataType = tensorrt_llm.bindings.DataType
    CacheTypeCpp = tensorrt_llm.bindings.internal.batch_manager.CacheType
    dtype_map = {
        "fp8": DataType.FP8,
        "fp16": DataType.HALF,
        "half": DataType.HALF,
        "bf16": DataType.BF16,
    }
    dtype_str = side["kv_dtype"].lower()
    dtype = dtype_map.get(dtype_str)
    if dtype is None:
        print(f"[precheck] kv dtype {dtype_str!r} not mapped, using BF16", flush=True)
        dtype = DataType.BF16

    spec_config = None
    if side["num_nextn_predict_layers"] > 0:
        from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

        spec_config = MTPDecodingConfig(num_nextn_predict_layers=side["num_nextn_predict_layers"])

    tpb = plan["tokens_per_block"]
    padded_len = ((max_req_len + tpb - 1) // tpb) * tpb
    owned = pcfg.max_owned_per_chunk(plan, side["role"])
    max_tokens = owned * padded_len + 2 * tpb  # concurrent pairs + headroom

    # Real MLA serving uses SELFKONLY (kv_factor=1: one latent plane, no V) —
    # see _torch/pyexecutor/_util.py; SELF would double the per-token bytes.
    cache_type = CacheTypeCpp.SELFKONLY if kv_shape["is_mla"] else CacheTypeCpp.SELF
    common = dict(
        num_layers=kv_shape["num_layers"],
        num_kv_heads=kv_shape["num_kv_heads"],
        head_dim=kv_shape["head_dim"],
        tokens_per_block=tpb,
        max_seq_len=padded_len,
        max_batch_size=max(4, owned + 1),
        mapping=mapping,
        dtype=dtype,
        spec_config=spec_config,
    )
    if use_v2:
        from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

        # is_disagg=True doubles the IndexMapper capacity so in-flight
        # transfers (TRANS_IN_PROGRESS) can hold slots, like real serving.
        return KVCacheManagerV2(
            _KvCacheConfigV2(
                max_tokens=max_tokens,
                enable_block_reuse=False,
                max_attention_window=[padded_len],
            ),
            cache_type,
            vocab_size=kv_shape.get("vocab_size") or 32000,
            is_disagg=True,
            **common,
        )
    return KVCacheManager(
        trtllm_executor.KvCacheConfig(max_tokens=max_tokens, enable_block_reuse=False),
        cache_type,
        **common,
    )


def make_request(is_ctx, rid, req_len, runtime, ctx_params=None):
    """Build a ctx or gen LlmRequest (mirrors the UCX-tuning harness)."""
    import tensorrt_llm.bindings
    from tensorrt_llm import DisaggregatedParams
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestType
    from tensorrt_llm.sampling_params import SamplingParams

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
        return req
    return LlmRequest(
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
        context_phase_params=ctx_params,
        **common,
    )


def add_sequence(kvm, req, prompt_len, use_v2):
    """Allocate KV blocks (mirrors the cache_transceiver_test harness)."""
    if use_v2:
        if req.is_disagg_generation_init_state:
            ok = kvm.prepare_disagg_gen_init(req)
        else:
            ok = kvm.prepare_context(req) and kvm.resize_context(req, prompt_len)
        if not ok:
            raise RuntimeError(f"V2 KV cache allocation failed for request {req.py_request_id}")
        return
    kvm.impl.add_sequence_batch([(req.py_request_id, prompt_len, 1)], [req])


def free_sequence(kvm, req, use_v2):
    if use_v2:
        import torch

        # free_resources() closes the kv_cache AND releases the IndexMapper
        # slot (closing the cache alone leaks slots).
        torch.cuda.current_stream().synchronize()
        kvm.free_resources(req)
        return
    kvm.impl.remove_sequence(req.py_request_id, req, True)


def _wait_gen_complete(xcvr, req, runtime, llm_request_state):
    """Block until this gen request's receive finishes (or errors).

    PYTHON transceiver: check_gen_transfer_status(None) blocks for all. C++:
    the int API can return before THIS request completes on a cold link, so
    poll for a terminal state (bounded by signal.alarm + hang detector).
    """
    if runtime == "PYTHON":
        xcvr.check_gen_transfer_status(None)
        return
    terminal = (
        llm_request_state.DISAGG_GENERATION_TRANS_COMPLETE,
        llm_request_state.DISAGG_TRANS_ERROR,
    )
    while req.state not in terminal:
        xcvr.check_gen_transfer_status(1)
        if req.state in terminal:
            break
        time.sleep(0.001)


# --------------------------------------------------------------------------- #
# Rendezvous
# --------------------------------------------------------------------------- #
def addr_path(work_dir, ctx_idx, gen_idx):
    return os.path.join(work_dir, "rendezvous", f"ctx{ctx_idx}_gen{gen_idx}.addr")


def run_token():
    """Identity of THIS run, stamped into addr files and checked by readers.

    A reused --work-dir (Slurm requeue reruns the batch script with the same
    directories; manual reruns) can hold addr files from a previous run --
    connecting to that stale host:port would block until the hello timeout
    and misreport TIMEOUT. Within one precheck all instances share
    SLURM_JOB_ID; empty (non-Slurm manual runs) disables the check.
    """
    return os.environ.get("SLURM_JOB_ID", "")


def write_addr(path, payload):
    """Atomically publish an addr file (clearing any stale one first). It
    carries the session HMAC key, so restrict it to the owning user before
    it becomes visible."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        os.remove(path)  # stale file from a previous run in a reused work dir
    except FileNotFoundError:
        pass
    payload = dict(payload, job=run_token())
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        os.fchmod(f.fileno(), 0o600)
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def wait_for_addr(path, timeout_s):
    """Wait for THIS run's addr file; files stamped with another run's job id
    are treated as stale and skipped (keep polling)."""
    expect_job = run_token()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    payload = json.load(f)
            except (OSError, json.JSONDecodeError):
                payload = None  # mid-rename/NFS staleness; retry
            if payload is not None:
                stamped = payload.get("job", "")
                if not expect_job or not stamped or stamped == expect_job:
                    return payload
                # Stale addr from a previous run in a reused work dir.
        time.sleep(1.0)
    raise _Timeout(f"rendezvous file {path} not published within {timeout_s}s")


# --------------------------------------------------------------------------- #
# Status recording
# --------------------------------------------------------------------------- #
class StatusRecorder:
    """Leader-side result sink.

    Rewritten on every record so a SIGKILL'd run still leaves the completed
    cases + the in-flight failure on disk.
    """

    def __init__(self, work_dir, role, server_idx, is_leader):
        self.role = role
        self.server_idx = server_idx
        self.is_leader = is_leader
        self.cases = []
        status_dir = os.path.join(work_dir, "status")
        if is_leader:
            os.makedirs(status_dir, exist_ok=True)
        self.json_path = os.path.join(status_dir, f"{role}_{server_idx}.json")
        self.text_path = os.path.join(status_dir, f"{role}_{server_idx}.status")
        env_keys = ("UCX_", "NIXL_", "TRTLLM_", "TLLM_")
        self.env = {k: v for k, v in sorted(os.environ.items()) if k.startswith(env_keys)}

    def record(self, peer, req_len, status, reason=""):
        if not self.is_leader:
            return
        self.cases.append({"peer": peer, "req_len": req_len, "status": status, "reason": reason})
        self._flush(final=False)

    def finalize(self, extra=None):
        if not self.is_leader:
            return
        self._flush(final=True, extra=extra)

    def failed_cases(self):
        return [c for c in self.cases if c["status"] not in ("PASS", "SKIP")]

    def _flush(self, final, extra=None):
        failed = self.failed_cases()
        overall = "PASS" if (final and not failed) else ("FAIL" if failed else "RUNNING")
        doc = {
            "role": self.role,
            "server_idx": self.server_idx,
            "overall": overall,
            "cases": self.cases,
            "env": self.env,
        }
        if extra:
            doc.update(extra)
        tmp = f"{self.json_path}.tmp"
        with open(tmp, "w") as f:
            json.dump(doc, f, indent=2)
        os.replace(tmp, self.json_path)
        with open(f"{self.text_path}.tmp", "w") as f:
            if failed:
                first = failed[0]
                # One-line summary for the launch-script console output: root
                # cause only (first line, bounded); the full reason incl. any
                # backtrace stays in the .json.
                reason = " | ".join(first["reason"].splitlines()[:2])[:400]
                f.write(
                    f"FAIL {self.role}_{self.server_idx}: {len(failed)} case(s) failed; "
                    f"first: peer={first['peer']} req_len={first['req_len']} "
                    f"{first['status']}: {reason}\n"
                )
            elif final:
                f.write(f"PASS {self.role}_{self.server_idx}: {len(self.cases)} case(s)\n")
            else:
                f.write(f"RUNNING {self.role}_{self.server_idx}\n")
        os.replace(f"{self.text_path}.tmp", self.text_path)


def parse_bandwidth_gbps(csv_dir, rank, tag="recv"):
    """Median per-request bandwidth in GB/s (bytes/1e9), best-effort.

    Parsed from the C++ transceiver CSV this rank wrote via
    TRTLLM_KVCACHE_TIME_OUTPUT_PATH.
    """
    import csv as csv_mod
    import statistics

    path = os.path.join(csv_dir, f"rank_{rank}_{tag}.csv")
    try:
        with open(path) as f:
            rows = list(csv_mod.DictReader(f))
        col = next((c for c in (rows[0] or {}) if "Bandwidth" in c), None)
        vals = [float(r[col]) / 8.0 for r in rows if col and r.get(col)]
        return statistics.median(vals) if vals else None
    except (OSError, ValueError, IndexError, StopIteration):
        return None


# --------------------------------------------------------------------------- #
# Chunk execution
# --------------------------------------------------------------------------- #
class PrecheckRunner:
    """One MPI world = one ctx or gen server instance."""

    def __init__(self, args, plan, side, comm):
        from mpi4py import MPI  # noqa: F401 - ensures MPI initialized

        self.args = args
        self.plan = plan
        self.side = side
        self.role = side["role"]
        self.is_ctx = self.role == "ctx"
        self.server_idx = args.server_idx
        self.comm = comm
        self.rank = comm.Get_rank()
        self.is_leader = self.rank == 0
        self.work_dir = args.work_dir
        self.recorder = StatusRecorder(self.work_dir, self.role, self.server_idx, self.is_leader)
        self.zmq_ctx = None
        self.kvm = None
        self.xcvr = None
        self.runtime = "CPP"
        # Resolved in setup(): "auto" needs the model class (get_model_defaults).
        self.use_v2 = False
        self.mapping = None
        self.llm_request_state = None
        self.csv_dir = os.path.join(self.work_dir, "csv", f"{self.role}_{self.server_idx}")

    # ---- consensus helpers -------------------------------------------------
    def _consensus_error(self, local_err):
        """All ranks agree whether anyone failed; returns the shared reason."""
        errs = self.comm.allgather("" if local_err is None else repr(local_err))
        bad = [(r, e) for r, e in enumerate(errs) if e]
        if not bad:
            return None
        ranks = [r for r, _ in bad]
        return f"rank(s) {ranks}: {bad[0][1]}"

    # ---- setup -------------------------------------------------------------
    def setup(self, kv_shape, max_req_len):
        import tensorrt_llm
        import tensorrt_llm.bindings
        from tensorrt_llm._torch.distributed import Distributed
        from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import create_kv_cache_transceiver
        from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
        from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig
        from tensorrt_llm.mapping import Mapping

        self.llm_request_state = LlmRequestState
        par = self.side["parallel"]
        self.mapping = Mapping(
            world_size=par["world_size"],
            rank=self.rank,
            gpus_per_node=self.plan["gpus_per_node"],
            tp_size=par["tp"],
            pp_size=par["pp"],
            cp_size=par["cp"],
            enable_attention_dp=par["enable_attention_dp"],
        )
        os.makedirs(self.csv_dir, exist_ok=True)
        os.environ["TRTLLM_KVCACHE_TIME_OUTPUT_PATH"] = self.csv_dir

        # Built VERBATIM from the disagg yaml's cache_transceiver_config so
        # backend/max_tokens_in_buffer/timeouts match the real test exactly.
        cache_cfg = CacheTransceiverConfig(**self.side["cache_transceiver_config"])
        # Yaml-absent settings resolve against the model's preferences, like
        # serving does (kv manager version + transceiver runtime).
        self.use_v2 = resolve_model_prefs(self.plan.get("_model_dir"), self.side, cache_cfg)
        # KVCacheManagerV2 only works with the Python transceiver (see
        # cache_transceiver_test/report.py); reject the pairing up front with
        # a clear INIT_ERROR instead of a C++ binding type error.
        if self.use_v2 and cache_cfg.transceiver_runtime != "PYTHON":
            raise RuntimeError(
                "KVCacheManagerV2 requires cache_transceiver_config."
                f"transceiver_runtime: PYTHON, got {cache_cfg.transceiver_runtime!r} "
                "(the C++ transceiver only supports the V1 manager)"
            )

        self.kvm = build_kv_cache_manager(
            kv_shape, self.plan, self.side, self.mapping, max_req_len, self.use_v2
        )
        AttentionTypeCpp = tensorrt_llm.bindings.internal.batch_manager.AttentionType
        attention_type = AttentionTypeCpp.MLA if kv_shape["is_mla"] else AttentionTypeCpp.DEFAULT
        dist_obj = Distributed.get(self.mapping)
        self.xcvr = create_kv_cache_transceiver(
            self.mapping, dist_obj, self.kvm, attention_type, cache_cfg
        )
        if self.xcvr is None:
            raise RuntimeError("cache transceiver disabled by config")
        # create_kv_cache_transceiver resolves 'auto' in-place (no model
        # preference on this path -> C++), so read the effective runtime back.
        self.runtime = cache_cfg.transceiver_runtime or "CPP"

    # ---- per-chunk transfer logic -------------------------------------------
    def _pair_rid(self, peer_idx, li, rep, pair):
        ctx_idx = self.server_idx if self.is_ctx else peer_idx
        gen_idx = peer_idx if self.is_ctx else self.server_idx
        return make_rid(ctx_idx, gen_idx, self.plan["num_ctx_servers"], li, rep, pair)

    def _owned(self, chunk):
        return pcfg.owned_pairs(self.plan, self.role, self.mapping.tp_rank, chunk)

    def ctx_run_chunk(self, peer_idx, li, req_len, rep, chunk):
        """Fill + send owned pairs.

        Returns {pair: context_phase_params} on the leader (params from each
        pair's owning dp rank, pp stage 0).
        """
        import tensorrt_llm

        owned = self._owned(chunk)
        reqs, local_err = {}, None
        try:
            for pair in owned:
                rid = self._pair_rid(peer_idx, li, rep, pair)
                req = make_request(True, rid, req_len, self.runtime)
                add_sequence(self.kvm, req, req_len, self.use_v2)
                fill_request(self.kvm, rid)
                tensorrt_llm.logger.info(
                    f"[ctx{self.server_idx} r{self.rank}] rid={rid} len={req_len}: send START"
                )
                self.xcvr.respond_and_send_async(req)
                reqs[pair] = req
        except Exception as e:  # noqa: BLE001 - relayed to gen, then raised
            local_err = e
        reason = self._consensus_error(local_err)

        # Params for pair k come from its owning dp rank at pp stage 0 with
        # attention DP; without DP every rank sends the same request, and the
        # instance leader's params are the ones the real server would return.
        contrib = {}
        if local_err is None:
            if self.side["parallel"]["enable_attention_dp"]:
                if self.mapping.pp_rank == 0:
                    contrib = {p: r.context_phase_params for p, r in reqs.items()}
            elif self.is_leader:
                contrib = {p: r.context_phase_params for p, r in reqs.items()}
        gathered = self.comm.gather(contrib, root=0)
        params_by_pair = {}
        if self.is_leader and reason is None:
            for d in gathered:
                params_by_pair.update(d or {})
            missing = [p for p in chunk if p not in params_by_pair]
            if missing:
                reason = f"missing context_phase_params for pairs {missing}"

        if reason is not None:
            self._free_all(reqs)
            raise _TransferError(f"ctx send setup failed: {reason}")
        return params_by_pair, reqs

    def ctx_finish_chunk(self, reqs):
        """Wait for all in-flight sends of this chunk, then free."""
        local_err = None
        try:
            self.xcvr.check_context_transfer_status(None)  # block-all
            bad = [
                p for p, r in reqs.items() if r.state == self.llm_request_state.DISAGG_TRANS_ERROR
            ]
            if bad:
                local_err = _TransferError(f"ctx DISAGG_TRANS_ERROR on pairs {bad}")
        except Exception as e:  # noqa: BLE001
            local_err = e
        finally:
            self._free_all(reqs)
        reason = self._consensus_error(local_err)
        if reason is not None:
            raise _TransferError(f"ctx transfer failed: {reason}")

    def gen_run_chunk(self, peer_idx, li, req_len, rep, chunk, params_by_pair):
        """Receive + verify owned pairs. Returns (ok, mismatch_detail)."""
        import torch

        import tensorrt_llm

        owned = self._owned(chunk)
        reqs, local_err = {}, None
        try:
            for pair in owned:
                rid = self._pair_rid(peer_idx, li, rep, pair)
                req = make_request(
                    False, rid, req_len, self.runtime, ctx_params=params_by_pair[pair]
                )
                add_sequence(self.kvm, req, req_len, self.use_v2)
                tensorrt_llm.logger.info(
                    f"[gen{self.server_idx} r{self.rank}] rid={rid} len={req_len}: recv START"
                )
                self.xcvr.request_and_receive_async(req)
                reqs[pair] = req
        except Exception as e:  # noqa: BLE001
            local_err = e
        reason = self._consensus_error(local_err)
        if reason is not None:
            self._free_all(reqs)
            raise _TransferError(f"gen receive setup failed: {reason}")

        mismatch = ""
        try:
            for pair, req in reqs.items():
                _wait_gen_complete(self.xcvr, req, self.runtime, self.llm_request_state)
            torch.cuda.synchronize()  # receive may land on a side stream
            bad = [
                p for p, r in reqs.items() if r.state == self.llm_request_state.DISAGG_TRANS_ERROR
            ]
            if bad:
                local_err = _TransferError(f"gen DISAGG_TRANS_ERROR on pairs {bad}")
            elif self.plan["verify_data"]:
                for pair, req in reqs.items():
                    ok, detail = verify_request(self.kvm, req.py_request_id)
                    if not ok:
                        mismatch = f"pair={pair} {detail}"
                        break
        except Exception as e:  # noqa: BLE001
            local_err = e
        finally:
            self._free_all(reqs)
        reason = self._consensus_error(local_err)
        if reason is not None:
            raise _TransferError(f"gen transfer failed: {reason}")
        mismatches = [m for m in self.comm.allgather(mismatch) if m]
        return (not mismatches, "; ".join(mismatches[:4]))

    def _free_all(self, reqs):
        for req in reqs.values():
            try:
                free_sequence(self.kvm, req, self.use_v2)
            except Exception:  # noqa: BLE001 - teardown best-effort
                pass

    # ---- ZMQ helpers ---------------------------------------------------------
    def _zmq(self):
        import zmq

        if self.zmq_ctx is None:
            self.zmq_ctx = zmq.Context.instance()
        return zmq, self.zmq_ctx

    def _leader_send_recv(self, sock, obj, key):
        """REQ round-trip on the gen leader; broadcast the reply to all ranks."""
        reply = None
        err = None
        if self.is_leader:
            try:
                sock.send(pack_msg(obj, key))
                reply = unpack_msg(sock.recv(), key)
            except Exception as e:  # noqa: BLE001
                err = repr(e)
        err, reply = self.comm.bcast((err, reply), root=0)
        if err:
            raise _TransferError(f"ZMQ control channel failed: {err}")
        return reply


def _schedule(plan):
    """Deterministic (li, req_len, rep, chunk) schedule both sides iterate."""
    out = []
    total_reps = plan["warmup_requests"] + plan["num_requests"]
    for li, req_len in enumerate(plan["request_lengths"]):
        for rep in range(total_reps):
            for chunk in pcfg.chunks(plan):
                out.append((li, req_len, rep, chunk))
    return out


def hello_timeout_s(plan, num_peers):
    """Timeout budget for session handshakes.

    Handshakes are serialized across peers (one gen talks to one ctx at a
    time), so waiting for a peer's hello/welcome can legitimately span other
    peers' full sessions -- budget rendezvous + per-peer slack.
    """
    return plan["rendezvous_timeout_s"] + num_peers * 300


# --------------------------------------------------------------------------- #
# ctx / gen session loops
# --------------------------------------------------------------------------- #
def ctx_serve_peer(runner, sock, peer_idx, arm, disarm, key):
    """Serve one gen peer's full schedule on a dedicated REP socket."""
    plan = runner.plan
    comm = runner.comm

    def leader_recv():
        msg, err = None, None
        if runner.is_leader:
            try:
                msg = unpack_msg(sock.recv(), key)
            except Exception as e:  # noqa: BLE001
                err = repr(e)
        err, msg = comm.bcast((err, msg), root=0)
        if err:
            raise _TransferError(f"ZMQ recv from gen_{peer_idx} failed: {err}")
        return msg

    def leader_reply(obj):
        if runner.is_leader:
            sock.send(pack_msg(obj, key))

    arm(f"hello gen_{peer_idx}", seconds=hello_timeout_s(plan, runner.side["num_peers"]))
    msg = leader_recv()
    if msg[0] != "hello" or msg[1].get("fingerprint") != plan["fingerprint"]:
        leader_reply(("abort", "plan fingerprint mismatch (ctx/gen yaml disagree)"))
        raise _TransferError(f"handshake with gen_{peer_idx} failed: {msg[:1]}")
    leader_reply(("welcome", {"fingerprint": plan["fingerprint"]}))

    for li, req_len, rep, chunk in _schedule(plan):
        arm(f"gen_{peer_idx} len={req_len} rep={rep}")
        msg = leader_recv()
        if msg[0] == "abort":
            raise _PeerAbort(f"gen_{peer_idx} aborted: {msg[1]}")
        if msg[0] != "go" or (msg[1]["li"], msg[1]["rep"]) != (li, rep):
            raise _TransferError(
                f"schedule desync with gen_{peer_idx}: expected li={li} rep={rep}, got {msg}"
            )
        try:
            params_by_pair, reqs = runner.ctx_run_chunk(peer_idx, li, req_len, rep, chunk)
        except _TransferError as e:
            leader_reply(("abort", str(e)))
            raise
        # JSON object keys are strings; the gen side converts back to int.
        leader_reply(("params", {str(p): params_to_wire(v) for p, v in params_by_pair.items()}))
        runner.ctx_finish_chunk(reqs)

    # The gen defers "done" until it has finished the schedules of ALL its
    # ctx peers, so every ctx instance stays alive for the whole precheck --
    # matching real serving, where ctx servers outlive the entire run. (An
    # early-exiting ctx leaves the gen's C++ transceiver holding connections
    # to a dead agent, a state the real test never produces.) The wait can
    # therefore span the gen's remaining sessions: budget like a handshake.
    arm(f"bye gen_{peer_idx}", seconds=hello_timeout_s(plan, runner.side["num_peers"]))
    msg = leader_recv()
    if msg[0] != "done":
        raise _TransferError(f"expected done from gen_{peer_idx}, got {msg[:1]}")
    leader_reply(("bye", {}))
    disarm()


def gen_run_peer(runner, peer_idx, arm, disarm):
    """Run the full schedule against ctx server `peer_idx`.

    Returns (sock, key) with the session STILL OPEN on success -- the caller
    sends the deferred "done" only after every ctx peer's schedule finished,
    keeping all ctx instances alive for the whole precheck (real-serving
    lifecycle; see ctx_serve_peer). On failure the socket is closed here.
    """
    plan = runner.plan
    comm = runner.comm
    hello_s = hello_timeout_s(plan, runner.side["num_peers"])

    # Rendezvous with instance-wide consensus: if only the leader raised here,
    # the other ranks would deadlock in the next bcast.
    sock, key, err = None, None, None
    arm(f"rendezvous ctx_{peer_idx}", seconds=hello_s)
    if runner.is_leader:
        try:
            addr = wait_for_addr(
                addr_path(runner.work_dir, peer_idx, runner.server_idx),
                plan["rendezvous_timeout_s"],
            )
            # Session HMAC key: shared only through the work-dir addr file,
            # never over the network.
            key = bytes.fromhex(addr["key"])
            zmq, zctx = runner._zmq()
            sock = zctx.socket(zmq.REQ)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.RCVTIMEO, hello_s * 1000)
            sock.connect(f"tcp://{addr['host']}:{addr['port']}")
        except Exception as e:  # noqa: BLE001 - shared via bcast below
            err = repr(e)
    err = comm.bcast(err, root=0)
    if err:
        raise _TransferError(f"rendezvous with ctx_{peer_idx} failed: {err}")

    try:
        arm(f"hello ctx_{peer_idx}", seconds=hello_s)
        reply = runner._leader_send_recv(
            sock,
            ("hello", {"gen_idx": runner.server_idx, "fingerprint": plan["fingerprint"]}),
            key,
        )
        if reply[0] == "abort":
            raise _TransferError(f"ctx_{peer_idx} aborted handshake: {reply[1]}")
        if reply[0] != "welcome":
            raise _TransferError(f"unexpected handshake reply from ctx_{peer_idx}: {reply[:1]}")
        # Established sessions are dedicated: chunk replies are prompt.
        if runner.is_leader:
            import zmq as zmq_mod

            sock.setsockopt(zmq_mod.RCVTIMEO, (plan["chunk_timeout_s"] + 30) * 1000)

        case_ok = {}
        for li, req_len, rep, chunk in _schedule(plan):
            arm(f"ctx_{peer_idx} len={req_len} rep={rep}")
            reply = runner._leader_send_recv(
                sock, ("go", {"li": li, "rep": rep, "chunk": chunk[0]}), key
            )
            if reply[0] == "abort":
                raise _TransferError(f"ctx_{peer_idx} aborted: {reply[1]}")
            params_by_pair = {int(p): params_from_wire(v) for p, v in reply[1].items()}
            ok, detail = runner.gen_run_chunk(peer_idx, li, req_len, rep, chunk, params_by_pair)
            if rep >= plan["warmup_requests"]:
                prev_ok, prev_detail = case_ok.get(req_len, (True, ""))
                case_ok[req_len] = (prev_ok and ok, prev_detail or detail)
        disarm()

        for req_len, (ok, detail) in case_ok.items():
            runner.recorder.record(
                f"ctx_{peer_idx}",
                req_len,
                "PASS" if ok else "MISMATCH",
                "" if ok else detail,
            )
        return sock, key
    except BaseException:
        if sock is not None:
            sock.close(linger=0)
        raise


def gen_release_peer(runner, peer_idx, sock, key, arm, disarm):
    """Deferred session teardown: send "done" and close (best-effort)."""
    try:
        arm(f"bye ctx_{peer_idx}")
        runner._leader_send_recv(sock, ("done", {}), key)
        disarm()
    finally:
        if sock is not None:
            sock.close(linger=0)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--role", required=True, choices=["ctx", "gen"])
    ap.add_argument("--server-idx", type=int, required=True)
    ap.add_argument("--config", required=True, help="disagg perf-sanity yaml path")
    ap.add_argument("--work-dir", required=True, help="shared dir for rendezvous/status")
    ap.add_argument("--benchmark-mode", default="e2e", choices=["e2e", "gen_only"])
    ap.add_argument("--llm-src", default="", help="repo root (model path dict lookup)")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved plan and exit")
    return ap.parse_args(argv)


def load_plan(args):
    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    plan = pcfg.resolve_plan(cfg, benchmark_mode=args.benchmark_mode)
    if plan.get("skip"):
        return plan, None
    model_dir = pcfg.resolve_model_dir(cfg, llm_src=args.llm_src or None)
    role_side = pcfg.side_plan(plan, args.role) if not args.dry_run else None
    kv_shape = pcfg.model_kv_shape(model_dir)
    plan["_kv_shape"] = kv_shape
    plan["_model_dir"] = model_dir
    return plan, role_side


def main(argv=None):
    args = parse_args(argv)
    plan, side = load_plan(args)

    if args.dry_run:
        print(json.dumps({k: v for k, v in plan.items()}, indent=2, default=str))
        return 0
    if plan.get("skip"):
        print(f"[precheck] SKIP: {plan['skip_reason']}", flush=True)
        return 0

    # UCX_PROTO_INFO=used is log-only (does not change transport selection):
    # it makes UCX >= 1.21 print the chosen GPU<->GPU protocol table, which the
    # failure summary uses to spot host-staged tcp fallbacks.
    os.environ.setdefault("UCX_PROTO_INFO", "used")

    import torch
    from mpi4py import MPI

    import tensorrt_llm

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world = comm.Get_size()
    expected_world = side["parallel"]["world_size"]
    if world != expected_world:
        raise RuntimeError(
            f"MPI world size {world} != {args.role} world size {expected_world}; "
            f"the precheck srun must use the same topology as the real "
            f"{args.role} server step."
        )
    torch.cuda.set_device(rank % torch.cuda.device_count())
    tensorrt_llm.logger.set_level("info")

    ucx_env = " ".join(f"{k}={v}" for k, v in sorted(os.environ.items()) if k.startswith("UCX_"))
    print(
        f"[precheck {args.role}_{args.server_idx} r{rank}] UCX env: {ucx_env or '<none>'}",
        flush=True,
    )

    runner = PrecheckRunner(args, plan, side, comm)
    kv_shape = plan["_kv_shape"]
    if runner.is_leader:
        print(
            f"[precheck {args.role}_{args.server_idx}] kv_shape={kv_shape} "
            f"model_dir={plan['_model_dir']} pairs={plan['n_pairs']} "
            f"req_lens={plan['request_lengths']}",
            flush=True,
        )

    # --- watchdog: signal.alarm for Python-level stalls + HangDetector for
    # GIL-released native hangs (dumps stacks, records TIMEOUT, SIGKILLs so
    # `srun --kill-on-bad-exit` tears the step down; the external `timeout`
    # around the srun is the guaranteed backstop for GIL-held hangs).
    signal.signal(signal.SIGALRM, _alarm_handler)
    from tensorrt_llm._torch.pyexecutor.hang_detector import HangDetector

    current_cell = {"what": "startup"}

    def _on_hang():
        runner.recorder.record("-", 0, "TIMEOUT", f"hang detected during {current_cell['what']}")
        runner.recorder.finalize()
        sys.stderr.write(
            f"[precheck {args.role}_{args.server_idx} r{rank}] WATCHDOG_KILL "
            f"{current_cell['what']}\n"
        )
        sys.stderr.flush()
        os.kill(os.getpid(), signal.SIGKILL)

    # The detector must outlast the LONGEST legitimate wait (peer handshakes
    # are serialized across sessions); per-cell alarms below are the tighter
    # bound for actual transfer work.
    hang_detector = HangDetector(
        timeout=hello_timeout_s(plan, side["num_peers"]) + plan["chunk_timeout_s"] + 60,
        on_detected=_on_hang,
    )
    hang_detector.start()

    def arm(what, seconds=None):
        current_cell["what"] = what
        signal.alarm(seconds or plan["chunk_timeout_s"])
        hang_detector.checkpoint()

    def disarm():
        signal.alarm(0)
        hang_detector.cancel_task()

    # --- setup: KV pool + transceiver (same config as the real test) ---------
    setup_err = None
    try:
        arm("kv pool + transceiver setup")
        runner.setup(kv_shape, max_req_len=max(plan["request_lengths"]))
        disarm()
    except Exception as e:  # noqa: BLE001 - recorded and gated below
        disarm()
        setup_err = e
    reason = runner._consensus_error(setup_err)
    if reason is not None:
        runner.recorder.record("-", 0, "INIT_ERROR", f"transceiver setup failed: {reason}")
        runner.recorder.finalize()
        print(
            f"[precheck {args.role}_{args.server_idx} r{rank}] INIT_ERROR: {reason}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    if runner.is_leader:
        # Effective values after model-preference resolution — what serving
        # would actually run with (PR #15823 semantics).
        print(
            f"[precheck {args.role}_{args.server_idx}] "
            f"kv_cache_manager={'V2' if runner.use_v2 else 'V1'} "
            f"transceiver_runtime={runner.runtime}",
            flush=True,
        )

    # --- sessions -------------------------------------------------------------
    num_peers = side["num_peers"]
    try:
        if args.role == "ctx":
            # One dedicated REP socket per gen peer (avoids REQ interleaving
            # across sessions on a shared socket), published via addr files.
            # Each session gets a fresh HMAC key, shared only through the
            # work-dir addr file (0600).
            socks, keys = {}, {}
            if runner.is_leader:
                zmq, zctx = runner._zmq()
                host = os.environ.get("SLURMD_NODENAME") or socket.gethostname()
                for gj in range(num_peers):
                    s = zctx.socket(zmq.REP)
                    s.setsockopt(zmq.LINGER, 0)
                    # Generous: gen peers are serialized across ctx servers.
                    s.setsockopt(zmq.RCVTIMEO, hello_timeout_s(plan, num_peers) * 1000)
                    port = s.bind_to_random_port("tcp://*")
                    keys[gj] = secrets.token_bytes(32)
                    write_addr(
                        addr_path(runner.work_dir, args.server_idx, gj),
                        {"host": host, "port": port, "key": keys[gj].hex()},
                    )
                    socks[gj] = s
            for gj in range(num_peers):
                try:
                    ctx_serve_peer(
                        runner,
                        socks.get(gj) if runner.is_leader else None,
                        gj,
                        arm,
                        disarm,
                        keys.get(gj),
                    )
                    runner.recorder.record(f"gen_{gj}", 0, "PASS", "served all transfers")
                except _Timeout:
                    disarm()
                    runner.recorder.record(
                        f"gen_{gj}",
                        0,
                        "TIMEOUT",
                        f"exceeded {plan['chunk_timeout_s']}s during {current_cell['what']}",
                    )
                except _PeerAbort as e:
                    disarm()
                    runner.recorder.record(f"gen_{gj}", 0, "TRANSFER_ERROR", str(e))
                except Exception as e:  # noqa: BLE001 - per-peer isolation
                    disarm()
                    runner.recorder.record(f"gen_{gj}", 0, "TRANSFER_ERROR", repr(e))
        else:
            open_sessions = []
            for ci in range(num_peers):
                try:
                    sock, sess_key = gen_run_peer(runner, ci, arm, disarm)
                    open_sessions.append((ci, sock, sess_key))
                except _Timeout:
                    disarm()
                    runner.recorder.record(
                        f"ctx_{ci}",
                        0,
                        "TIMEOUT",
                        f"exceeded {plan['chunk_timeout_s']}s during {current_cell['what']}",
                    )
                except Exception as e:  # noqa: BLE001 - per-peer isolation
                    disarm()
                    runner.recorder.record(f"ctx_{ci}", 0, "TRANSFER_ERROR", repr(e))
            # Deferred "done": released only after EVERY ctx peer's schedule
            # finished, so all ctx instances stay alive for the whole
            # precheck (real-serving lifecycle -- no dead-agent connections
            # in anyone's transceiver while transfers are still running).
            for ci, sock, sess_key in open_sessions:
                try:
                    gen_release_peer(runner, ci, sock, sess_key, arm, disarm)
                except _Timeout:
                    disarm()
                    runner.recorder.record(
                        f"ctx_{ci}", 0, "TIMEOUT", "releasing session (deferred done)"
                    )
                except Exception as e:  # noqa: BLE001 - best-effort release
                    disarm()
                    runner.recorder.record(f"ctx_{ci}", 0, "TRANSFER_ERROR", repr(e))
    finally:
        disarm()
        hang_detector.stop()

    # --- teardown + result ------------------------------------------------------
    bw = None
    if args.role == "gen":
        local_bw = parse_bandwidth_gbps(runner.csv_dir, rank)
        bws = [b for b in comm.gather(local_bw, root=0) or [] if b]
        if runner.is_leader and bws:
            bw = sorted(bws)[len(bws) // 2]
    if runner.xcvr is not None and hasattr(runner.xcvr, "shutdown"):
        try:
            runner.xcvr.shutdown()
        except Exception:  # noqa: BLE001 - teardown best-effort
            pass

    failed_local = 1 if runner.recorder.failed_cases() else 0
    failed = comm.allreduce(failed_local, op=MPI.MAX)
    extra = {
        "kv_cache_manager": "V2" if runner.use_v2 else "V1",
        "transceiver_runtime": runner.runtime,
    }
    if bw:
        extra["per_gpu_bw_gbps"] = bw
    runner.recorder.finalize(extra=extra)
    comm.Barrier()
    if runner.is_leader:
        verdict = "FAIL" if failed else "PASS"
        bw_note = f" per-GPU BW ~{bw:.1f} GB/s" if bw else ""
        print(f"[precheck {args.role}_{args.server_idx}] {verdict}{bw_note}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

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

Vocabulary: a PAIR is one dp-rank-to-dp-rank transfer pairing (n_pairs =
max of the two sides' dp sizes, so every dp rank is exercised); a REP is one
repetition of all pairs (1 warmup + num_requests measured); a WAVE is the
batch of pairs in flight at once (at most max_concurrent_pairs). "Wave" is
used instead of "chunk" on purpose -- it would otherwise collide with
chunked prefill (splitting a request's tokens), which is unrelated.

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

# Request-id scheme: rids must be unique across the whole precheck AND
# dense within a (ctx, gen) session -- the C++ transceiver derives its
# notification tag from the LOW 12 BITS of the request id (tagFromRequestId,
# dataTransceiver.cpp), and notifications are matched by (remote agent, tag).
# A dense per-session sequence keeps tags unique among any 4096 consecutive
# requests of a session; the peer stride only separates sessions, which talk
# to distinct agents and therefore cannot alias tags with each other.
RID_PEER_STRIDE = 1 << 24
CONTROL_POLL_INTERVAL_MS = 5_000
ABORT_COORDINATION_TIMEOUT_S = 2.0


class _Timeout(Exception):
    pass


class _TransferError(Exception):
    pass


class _FatalTransferError(_TransferError):
    """Transfer quiescence is unproven; finalizers must not release its memory."""


class _PeerAbort(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


def _coordinate_abort_after_leader_flush(comm):
    """Bounded best-effort rendezvous after the leader writes its verdict."""
    try:
        request = comm.Ibarrier()
        deadline = time.monotonic() + ABORT_COORDINATION_TIMEOUT_S
        while not request.Test():
            if time.monotonic() >= deadline:
                return
            time.sleep(0.01)
    except Exception:  # noqa: BLE001 - abort remains the mandatory fallback
        return


def _hard_abort_process(comm):
    """Terminate without running transceiver/KV-manager finalizers."""
    try:
        comm.Abort(137)
    except Exception:  # noqa: BLE001 - SIGKILL is the mandatory fallback
        pass
    os.kill(os.getpid(), signal.SIGKILL)
    raise RuntimeError("SIGKILL unexpectedly returned")


def make_rid(ctx_idx, gen_idx, num_ctx, seq):
    """Unique rid: peer-session base + dense in-session sequence number."""
    peer = gen_idx * num_ctx + ctx_idx
    return 1 + peer * RID_PEER_STRIDE + seq


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
    """Inverse of params_to_wire.

    Uses the maintained DisaggregatedParams converter (keeps us off the raw
    nanobind ctor signature).
    """
    return (
        load_internal_apis()
        .DisaggregatedParams(
            ctx_request_id=int(d["req_id"]),
            first_gen_tokens=list(d["first_gen_tokens"]),
            opaque_state=base64.b64decode(d["opaque_state"]),
            draft_tokens=d["draft_tokens"],
            ctx_dp_rank=d["ctx_dp_rank"],
            ctx_info_endpoint=d["ctx_info_endpoint"],
        )
        .get_context_phase_params()
    )


# --------------------------------------------------------------------------- #
# TRT-LLM internal API surface (single owner)
# --------------------------------------------------------------------------- #
_INTERNAL_APIS = None


def load_internal_apis():
    """Every tensorrt_llm symbol the precheck touches, imported in ONE place.

    The precheck bypasses the serving stack and drives internal APIs directly
    (_torch.pyexecutor.*, bindings.internal.*, private llm_utils resolvers),
    none of which carry a stability promise. Centralizing the imports means an
    upstream rename breaks exactly here, and the contract test
    (tests/unittest/others/test_cache_transceiver_precheck_run.py) fails in
    the refactorer's pre-merge CI instead of aborting the SLURM disagg perf
    pipeline at runtime.

    Deliberately lazy (NOT at module import): --dry-run and the pure-logic
    unit tests must work without torch / tensorrt_llm installed.
    """
    global _INTERNAL_APIS
    if _INTERNAL_APIS is not None:
        return _INTERNAL_APIS
    import types

    import tensorrt_llm
    import tensorrt_llm.bindings
    import tensorrt_llm.bindings.executor as trtllm_executor
    from tensorrt_llm import DisaggregatedParams
    from tensorrt_llm._torch.distributed import Distributed
    from tensorrt_llm._torch.models.modeling_utils import get_registered_model_class
    from tensorrt_llm._torch.pyexecutor.hang_detector import HangDetector
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import (
        create_kv_cache_transceiver,
        maybe_enable_fabric_memory_for_python_transceiver,
    )
    from tensorrt_llm._torch.pyexecutor.llm_request import (
        LlmRequest,
        LlmRequestState,
        LlmRequestType,
    )
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
    from tensorrt_llm.llmapi.llm_args import (
        CacheTransceiverConfig,
        KvCacheConfig,
        MTPDecodingConfig,
        TorchLlmArgs,
    )
    from tensorrt_llm.llmapi.llm_utils import (
        _resolve_kv_cache_manager_v2_auto,
        _resolve_transceiver_runtime_auto,
    )
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.sampling_params import SamplingParams

    _INTERNAL_APIS = types.SimpleNamespace(
        tensorrt_llm=tensorrt_llm,
        DataType=tensorrt_llm.bindings.DataType,
        SamplingConfigCpp=tensorrt_llm.bindings.SamplingConfig,
        CacheTypeCpp=tensorrt_llm.bindings.internal.batch_manager.CacheType,
        AttentionTypeCpp=tensorrt_llm.bindings.internal.batch_manager.AttentionType,
        KvCacheConfigCpp=trtllm_executor.KvCacheConfig,
        DisaggregatedParams=DisaggregatedParams,
        Distributed=Distributed,
        get_registered_model_class=get_registered_model_class,
        HangDetector=HangDetector,
        KVCacheManager=KVCacheManager,
        KVCacheManagerV2=KVCacheManagerV2,
        create_kv_cache_transceiver=create_kv_cache_transceiver,
        maybe_enable_fabric_memory_for_python_transceiver=(
            maybe_enable_fabric_memory_for_python_transceiver
        ),
        LlmRequest=LlmRequest,
        LlmRequestState=LlmRequestState,
        LlmRequestType=LlmRequestType,
        CacheTransceiverConfig=CacheTransceiverConfig,
        KvCacheConfig=KvCacheConfig,
        MTPDecodingConfig=MTPDecodingConfig,
        TorchLlmArgs=TorchLlmArgs,
        resolve_kv_cache_manager_v2_auto=_resolve_kv_cache_manager_v2_auto,
        resolve_transceiver_runtime_auto=_resolve_transceiver_runtime_auto,
        Mapping=Mapping,
        SamplingParams=SamplingParams,
    )
    return _INTERNAL_APIS


# --------------------------------------------------------------------------- #
# KV fill / verify (heavy imports stay inside functions: --dry-run and unit
# tests must work without torch / tensorrt_llm installed)
# --------------------------------------------------------------------------- #
def _pattern_like(shape, dtype, device, seed):
    """Deterministic tensor of `shape`, constant along the head axis.

    `shape` is an HND block slice: [nblocks, kv_factor, heads, tokens, dim].
    Generated on CPU (bit-identical across nodes), head axis generated as 1
    and expanded ON DEVICE (transferring heads x the unique data would be
    pure waste), so ctx/gen sides with different local head counts (TP
    resharding) or different local layer sets (PP re-splitting) still agree.
    """
    import torch

    nb, kv, heads, tok, dim = shape
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    rnd = torch.rand((nb, kv, 1, tok, dim), dtype=torch.float32, generator=g)
    return rnd.to(dtype).to(device).expand(nb, kv, heads, tok, dim)


def _request_block_views(kvm, rid, prompt_len):
    """Yield the prompt blocks transferred for this request on this rank."""
    num_prompt_blocks = (prompt_len + kvm.tokens_per_block - 1) // kvm.tokens_per_block
    for global_layer in kvm.pp_layers:
        blocks = kvm.get_batch_cache_indices([rid], layer_idx=global_layer)[0]
        # V2 may reserve extra KV tokens for speculative decoding. At an exact
        # block boundary those tokens allocate an additional page, but the
        # transceiver intentionally trims its slice to prompt_len blocks.
        # Verify the same payload range instead of the untransferred page.
        valid = [b for b in blocks if b >= 0]
        if len(valid) < num_prompt_blocks:
            raise _TransferError(
                f"KV under-allocation for rid={rid} layer={global_layer}: "
                f"required={num_prompt_blocks} available={len(valid)}"
            )
        valid = valid[:num_prompt_blocks]
        if not valid:
            continue
        buf = kvm.get_buffers(global_layer, kv_layout="HND")
        yield global_layer, buf, valid


def fill_request(kvm, rid, prompt_len):
    for global_layer, buf, valid in _request_block_views(kvm, rid, prompt_len):
        shape = (len(valid), *buf.shape[1:])
        buf[valid] = _pattern_like(shape, buf.dtype, buf.device, seed_for(rid, global_layer))


def verify_request(kvm, rid, prompt_len):
    """Returns (ok, detail) comparing received blocks to the expected pattern."""
    import torch

    for global_layer, buf, valid in _request_block_views(kvm, rid, prompt_len):
        recv = buf[valid]
        exp = _pattern_like(recv.shape, recv.dtype, recv.device, seed_for(rid, global_layer))
        recv_f, exp_f = recv.float(), exp.float()  # fp8 lacks direct compare ops
        if not torch.equal(recv_f, exp_f):
            bad = (recv_f != exp_f).sum().item()
            return False, f"layer={global_layer} mismatched_elements={bad}/{recv.numel()}"
    return True, ""


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
    # Resolves the lazily imported provider on demand, like serving does.
    return load_internal_apis().get_registered_model_class(archs[0]), hf_view


def resolve_model_prefs(model_dir, side, cache_cfg):
    """Mirror serving's model-preference resolution (PR #15823 semantics).

    - use_kv_cache_manager_v2 == "auto" (yaml absent): require the model
      class and adopt its get_preferred_kv_cache_manager_version() value
      (llm_utils._resolve_kv_cache_manager_v2_auto).
    - cache_cfg.transceiver_runtime == "auto": adopt
      model_cls.get_preferred_transceiver_runtime(), NIXL-gated, via the
      REAL llm_utils._resolve_transceiver_runtime_auto (mutates cache_cfg).

    Resolution errors propagate so the precheck cannot silently exercise a
    different runtime/cache-manager combination from serving. Returns the
    effective use_v2 bool.
    """
    import types

    api = load_internal_apis()
    model_cls, hf_view = _lookup_model_cls(model_dir)
    setting = side["use_kv_cache_manager_v2"]
    if setting == "auto" and model_cls is None:
        raise RuntimeError(
            "use_kv_cache_manager_v2 is 'auto', but the precheck could not resolve "
            f"a registered model class from model_dir={model_dir!r}; refusing to assume V1"
        )

    # Runtime BEFORE V2, like serving: the V2 resolver's disagg gating reads
    # cache_cfg.transceiver_runtime and treats an unresolved "auto" as non-PYTHON.
    if getattr(cache_cfg, "transceiver_runtime", None) == "auto":
        try:
            shim = types.SimpleNamespace(cache_transceiver_config=cache_cfg)
            api.resolve_transceiver_runtime_auto(shim, model_cls, hf_view)
        except Exception as e:  # noqa: BLE001 - resolver spans model extension hooks
            raise RuntimeError(
                "transceiver_runtime 'auto' resolution failed; refusing to validate "
                "a runtime that may differ from serving"
            ) from e

    if setting == "auto":
        try:
            parallel = side["parallel"]
            llm_args_kwargs = {
                "model": model_dir,
                "tensor_parallel_size": parallel["tp"],
                "pipeline_parallel_size": parallel["pp"],
                "context_parallel_size": parallel["cp"],
                "kv_cache_config": {"use_kv_cache_manager_v2": setting},
                "cache_transceiver_config": cache_cfg,
            }
            num_nextn = int(side.get("num_nextn_predict_layers", 0) or 0)
            if num_nextn > 0:
                llm_args_kwargs["speculative_config"] = api.MTPDecodingConfig(
                    num_nextn_predict_layers=num_nextn
                )
            resolver_args = api.TorchLlmArgs(**llm_args_kwargs)
            # Use the real serving arguments so the resolver sees the same
            # disaggregation, parallelism, and speculative-decoding inputs.
            use_v2 = bool(api.resolve_kv_cache_manager_v2_auto(resolver_args, model_cls, hf_view))
        except Exception as e:  # noqa: BLE001 - resolver spans model extension hooks
            raise RuntimeError("V2 'auto' resolution failed; refusing to assume V1") from e
    else:
        use_v2 = bool(setting)
    return use_v2


def build_kv_cache_manager(kv_shape, plan, side, mapping, max_req_len, use_v2):
    api = load_internal_apis()
    dtype_map = {
        "fp8": api.DataType.FP8,
        "fp16": api.DataType.HALF,
        "half": api.DataType.HALF,
        "bf16": api.DataType.BF16,
    }
    dtype_str = side["kv_dtype"].lower()
    dtype = dtype_map.get(dtype_str)
    if dtype is None:
        print(f"[precheck] kv dtype {dtype_str!r} not mapped, using BF16", flush=True)
        dtype = api.DataType.BF16

    spec_config = None
    if side["num_nextn_predict_layers"] > 0:
        spec_config = api.MTPDecodingConfig(
            num_nextn_predict_layers=side["num_nextn_predict_layers"]
        )

    tpb = plan["tokens_per_block"]
    padded_len = ((max_req_len + tpb - 1) // tpb) * tpb
    owned = pcfg.max_owned_per_wave(plan, side["role"])
    max_tokens = owned * padded_len + 2 * tpb  # concurrent pairs + headroom

    # Real MLA serving uses SELFKONLY (kv_factor=1: one latent plane, no V) —
    # see _torch/pyexecutor/_util.py; SELF would double the per-token bytes.
    cache_type = api.CacheTypeCpp.SELFKONLY if kv_shape["is_mla"] else api.CacheTypeCpp.SELF
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
        # The REAL pydantic KvCacheConfig (what serving passes): partial reuse
        # is pinned off because its pydantic default is True and block reuse
        # is off here. is_disagg=True doubles the IndexMapper capacity so
        # in-flight transfers (TRANS_IN_PROGRESS) can hold slots.
        return api.KVCacheManagerV2(
            api.KvCacheConfig(
                max_tokens=max_tokens,
                enable_block_reuse=False,
                enable_partial_reuse=False,
                copy_on_partial_reuse=False,
                max_attention_window=[padded_len],
            ),
            cache_type,
            vocab_size=kv_shape.get("vocab_size") or 32000,
            is_disagg=True,
            **common,
        )
    return api.KVCacheManager(
        api.KvCacheConfigCpp(max_tokens=max_tokens, enable_block_reuse=False),
        cache_type,
        **common,
    )


def make_request(is_ctx, rid, req_len, runtime, ctx_params=None):
    """Build a ctx or gen LlmRequest (mirrors the UCX-tuning harness)."""
    api = load_internal_apis()
    LlmRequest, LlmRequestType = api.LlmRequest, api.LlmRequestType
    DisaggregatedParams = api.DisaggregatedParams

    sampling = api.SamplingParams()
    common = dict(
        request_id=rid,
        max_new_tokens=1,
        input_tokens=list(range(req_len)),
        sampling_config=api.SamplingConfigCpp(sampling._get_sampling_config()),
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
        # free_resources() closes the kv_cache AND releases the IndexMapper
        # slot (closing the cache alone leaks slots). Callers synchronize the
        # stream once per batch before freeing (see _free_all).
        kvm.free_resources(req)
        return
    kvm.impl.remove_sequence(req.py_request_id, req, True)


def _wait_gen_complete(xcvr, req, runtime, llm_request_state):
    """Block until this gen request's receive finishes (or errors).

    The Python transceiver is handled once per wave in gen_run_wave(), where
    its returned request IDs can be checked before releasing KV pages. For C++,
    the int API can return before THIS request completes on a cold link, so
    poll for a terminal state (bounded by signal.alarm + hang detector).
    Logs periodic progress so a stalled transfer shows WHICH request is stuck
    in WHICH state (the difference between "requests never matched" and
    "RDMA write never completed").
    """
    if runtime == "PYTHON":
        raise ValueError("Python generation waves must be checked as a batch")
    terminal = (
        llm_request_state.DISAGG_GENERATION_TRANS_COMPLETE,
        llm_request_state.DISAGG_TRANS_ERROR,
    )
    t0 = time.monotonic()
    next_report = t0 + 15.0
    while req.state not in terminal:
        xcvr.check_gen_transfer_status(1)
        now = time.monotonic()
        if now >= next_report:
            print(
                f"[precheck] rid={req.py_request_id} recv still waiting: "
                f"state={req.state} elapsed={now - t0:.0f}s",
                flush=True,
            )
            next_report = now + 15.0
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
    """Atomically publish an addr file.

    os.replace overwrites stale ones; readers reject wrong-job stamps. It
    carries the session HMAC key, so restrict it to the owning user before it
    becomes visible.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = dict(payload, job=run_token())
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        os.fchmod(f.fileno(), 0o600)
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def wait_for_addr(path, timeout_s):
    """Wait for THIS run's addr file.

    Files stamped with another run's job id are treated as stale and skipped
    (keep polling).
    """
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


def peer_progress_path(work_dir, role, server_idx):
    """Shared progress marker for one ctx or gen instance."""
    return os.path.join(work_dir, "progress", f"{role}_{server_idx}.json")


def publish_peer_progress(runner, phase):
    """Best-effort atomic phase marker used by queued peer instances.

    Only instance leaders write. A new sequence value means the target peer is
    advancing, so queued hello/bye waits may refresh their no-progress
    watchdog without budgeting earlier serialized sessions cumulatively.
    """
    if not runner.is_leader:
        return
    path = peer_progress_path(runner.work_dir, runner.role, runner.server_idx)
    try:
        runner._precheck_progress_seq = getattr(runner, "_precheck_progress_seq", 0) + 1
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(
                {
                    "job": run_token(),
                    "phase": str(phase)[:400],
                    "seq": runner._precheck_progress_seq,
                },
                f,
            )
        os.replace(tmp, path)
    except OSError:
        # Missing progress only removes watchdog refreshes; the normal bounded
        # timeout and external gate remain authoritative.
        pass


def read_peer_progress(work_dir, role, server_idx):
    """Return this run's peer progress sequence, or None if absent/stale."""
    try:
        with open(peer_progress_path(work_dir, role, server_idx)) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    expect_job = run_token()
    stamped = payload.get("job", "")
    if expect_job and stamped and stamped != expect_job:
        return None
    return payload.get("seq")


def abort_flag_path(work_dir):
    return os.path.join(work_dir, "precheck.abort")


def raise_abort_flag(work_dir, reason):
    """Fail-fast signal, shared across instances through the work dir.

    The first peer failure (in ANY ctx/gen instance) drops this file so the
    others stop starting new sessions instead of each re-discovering the dead
    fabric on its own. Best-effort and write-once. Stamped with the job id so
    a stale flag in a reused work dir (Slurm requeue) is ignored, like addr
    files.
    """
    path = abort_flag_path(work_dir)
    if abort_flag_reason(work_dir) is not None:
        return
    try:
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(
                {"reason": (reason or "peer failure").splitlines()[0][:400], "job": run_token()},
                f,
            )
        os.replace(tmp, path)
    except OSError:
        pass  # best-effort: a missed flag only costs the usual per-peer timeout


def abort_flag_reason(work_dir):
    """The reason recorded by the first failing peer of THIS run, or None.

    A flag stamped with another job id is stale (reused work dir) and ignored.
    """
    try:
        with open(abort_flag_path(work_dir)) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    expect_job = run_token()
    stamped = payload.get("job", "")
    if expect_job and stamped and stamped != expect_job:
        return None
    return payload.get("reason") or "peer failure"


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
        # NIXL_* is deliberately absent: the only such variable seen in
        # practice is NIXL_VERSION, a stale marker from the NGC base image's
        # bundled NIXL (not the library TRT-LLM links from
        # /opt/nvidia/nvda_nixl), which misstates the transport version.
        env_keys = ("UCX_", "TRTLLM_", "TLLM_")
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
    TRTLLM_KVCACHE_TIME_OUTPUT_PATH, named "<instanceId>_<rank>_<tag>.csv"
    (instanceId is a runtime UUID), so match by the "_<rank>_<tag>.csv"
    suffix. The suffix's leading "_" keeps rank 1 from matching rank 11.

    Each row repeats the Bandwidth(Gbps) column once per transmission, so use
    csv.reader (DictReader would keep only the last duplicate) and take the
    mean transmission bandwidth as that request's value, then the median
    across requests -- same semantics as the harness report.
    """
    import csv as csv_mod
    import statistics

    suffix = f"_{rank}_{tag}.csv"
    try:
        names = [n for n in os.listdir(csv_dir) if n.endswith(suffix)]
    except OSError:
        return None
    vals = []
    for name in names:
        try:
            with open(os.path.join(csv_dir, name)) as f:
                reader = csv_mod.reader(f)
                header = next(reader, None)
                if not header:
                    continue
                bw_cols = [i for i, c in enumerate(header) if "Bandwidth" in c]
                if not bw_cols:
                    continue
                for row in reader:
                    bws = []
                    for i in bw_cols:
                        if i < len(row) and row[i]:
                            try:
                                bws.append(float(row[i]) / 8.0)  # Gbps -> GB/s
                            except ValueError:
                                pass
                    if bws:
                        vals.append(sum(bws) / len(bws))
        except OSError:
            continue
    return statistics.median(vals) if vals else None


def parse_python_bandwidth_gbps(csv_dir):
    """Median KV-send throughput in GB/s from the Python transceiver's perf CSVs.

    Written by perf_logger.py, which gives TRTLLM_KVCACHE_TIME_OUTPUT_PATH top
    priority and names files "{dir}/{instanceUuid}_{rank}.csv" -- so identify
    the CSVs by their header columns (task_type + throughput_mbs) rather than
    by name; C++ send/recv CSVs have neither column. throughput_mbs (MiB/s) is
    on the SENDER (ctx) side, task_type=KVSendTask; receiver rows have no
    throughput. Best-effort -- returns None when no perf CSV exists.
    """
    import csv as csv_mod
    import glob
    import statistics

    vals = []
    for path in glob.glob(os.path.join(csv_dir, "*.csv")):
        try:
            with open(path) as f:
                reader = csv_mod.DictReader(f)
                fields = reader.fieldnames or []
                if "task_type" not in fields or "throughput_mbs" not in fields:
                    continue
                for r in reader:
                    if r.get("task_type") == "KVSendTask" and r.get("throughput_mbs"):
                        # MiB/s -> GB/s
                        vals.append(float(r["throughput_mbs"]) * 1024.0 * 1024.0 / 1e9)
        except (OSError, ValueError):
            continue
    return statistics.median(vals) if vals else None


# --------------------------------------------------------------------------- #
# Wave execution
# --------------------------------------------------------------------------- #
class PrecheckRunner:
    """One MPI world = one ctx or gen server instance."""

    def __init__(self, args, plan, side, comm):
        from mpi4py import MPI  # noqa: F401 - ensures MPI initialized

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
        # Resolved in setup(): "auto" needs the model preference hook.
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
        api = load_internal_apis()

        self.llm_request_state = api.LlmRequestState
        par = self.side["parallel"]
        self.mapping = api.Mapping(
            world_size=par["world_size"],
            rank=self.rank,
            gpus_per_node=self.plan["gpus_per_node"],
            tp_size=par["tp"],
            pp_size=par["pp"],
            cp_size=par["cp"],
            enable_attention_dp=par["enable_attention_dp"],
        )
        os.makedirs(self.csv_dir, exist_ok=True)
        # One env var drives both transceivers' bandwidth CSVs: C++ writes
        # per-rank "<instanceId>_<rank>_send/recv.csv", and Python's
        # PerfLogManager gives the same var top priority, writing task CSVs
        # as "<instanceUuid>_<rank>.csv" (KVSendTask throughput on the ctx
        # side).
        os.environ["TRTLLM_KVCACHE_TIME_OUTPUT_PATH"] = self.csv_dir

        # Built VERBATIM from the disagg yaml's cache_transceiver_config so
        # backend/max_tokens_in_buffer/timeouts match the real test exactly.
        cache_cfg = api.CacheTransceiverConfig(**self.side["cache_transceiver_config"])
        # Yaml-absent settings resolve against the model's preferences, like
        # serving does (kv manager version + transceiver runtime) -- this holds
        # even for the simplified stand-in pool: only the KV SHAPE is generic;
        # the V1/V2 manager version and the transceiver runtime must still
        # match what the real model runs (e.g. V4 -> V2 + Python).
        self.use_v2 = resolve_model_prefs(self.plan.get("_model_dir"), self.side, cache_cfg)
        if kv_shape.get("simplified") and self.is_leader:
            print(
                f"[precheck {self.role}_{self.server_idx}] SIMPLIFIED: {kv_shape['simplified']}",
                flush=True,
            )
        # KVCacheManagerV2 only works with the Python transceiver (see
        # cache_transceiver_test/report.py); reject the pairing up front with
        # a clear INIT_ERROR instead of a C++ binding type error.
        if self.use_v2 and cache_cfg.transceiver_runtime != "PYTHON":
            raise RuntimeError(
                "KVCacheManagerV2 requires cache_transceiver_config."
                f"transceiver_runtime: PYTHON, got {cache_cfg.transceiver_runtime!r} "
                "(the C++ transceiver only supports the V1 manager)"
            )

        manager_cls = api.KVCacheManagerV2 if self.use_v2 else api.KVCacheManager
        api.maybe_enable_fabric_memory_for_python_transceiver(cache_cfg, manager_cls)
        self.kvm = build_kv_cache_manager(
            kv_shape, self.plan, self.side, self.mapping, max_req_len, self.use_v2
        )
        AttentionTypeCpp = api.AttentionTypeCpp
        attention_type = AttentionTypeCpp.MLA if kv_shape["is_mla"] else AttentionTypeCpp.DEFAULT
        dist_obj = api.Distributed.get(self.mapping)
        self.xcvr = api.create_kv_cache_transceiver(
            self.mapping, dist_obj, self.kvm, attention_type, cache_cfg
        )
        if self.xcvr is None:
            raise RuntimeError("cache transceiver disabled by config")
        # create_kv_cache_transceiver resolves 'auto' in-place (no model
        # preference on this path -> C++), so read the effective runtime back.
        self.runtime = cache_cfg.transceiver_runtime or "CPP"

    # ---- per-wave transfer logic -------------------------------------------
    def _pair_rid(self, peer_idx, li, rep, pair):
        ctx_idx = self.server_idx if self.is_ctx else peer_idx
        gen_idx = peer_idx if self.is_ctx else self.server_idx
        total_reps = self.plan["warmup_requests"] + self.plan["num_requests"]
        seq = (li * total_reps + rep) * self.plan["n_pairs"] + pair
        return make_rid(ctx_idx, gen_idx, self.plan["num_ctx_servers"], seq)

    def _owned(self, wave):
        return pcfg.owned_pairs(self.plan, self.role, self.mapping.tp_rank, wave)

    def ctx_run_wave(self, peer_idx, li, req_len, rep, wave):
        """Fill + send owned pairs.

        Returns {pair: context_phase_params} on the leader (params from each
        pair's owning dp rank, pp stage 0).
        """
        import tensorrt_llm

        owned = self._owned(wave)
        reqs, local_err = {}, None
        try:
            for pair in owned:
                rid = self._pair_rid(peer_idx, li, rep, pair)
                req = make_request(True, rid, req_len, self.runtime)
                add_sequence(self.kvm, req, req_len, self.use_v2)
                # Track ownership as soon as allocation succeeds. A later
                # setup failure must retain the pages rather than free storage
                # that an asynchronously dispatched sender may still read.
                reqs[pair] = req
                fill_request(self.kvm, rid, req_len)
                tensorrt_llm.logger.info(
                    f"[ctx{self.server_idx} r{self.rank}] rid={rid} len={req_len}: send START"
                )
                self.xcvr.respond_and_send_async(req)
        except _Timeout:
            raise
        except Exception as e:  # noqa: BLE001 - relayed to gen, then raised
            local_err = e
        try:
            reason = self._consensus_error(local_err)

            # Params for pair k come from its owning dp rank at pp stage 0 with
            # attention DP; without DP every rank sends the same request, and the
            # instance leader's params are the ones the real server would return.
            if self.side["parallel"]["enable_attention_dp"]:
                contributes = self.mapping.pp_rank == 0
            else:
                contributes = self.is_leader
            contrib = (
                {p: r.context_phase_params for p, r in reqs.items()}
                if local_err is None and contributes
                else {}
            )
            gathered = self.comm.gather(contrib, root=0)
            params_by_pair = {}
            if self.is_leader:
                for d in gathered:
                    params_by_pair.update(d or {})
                if reason is None:
                    missing = [p for p in wave if p not in params_by_pair]
                    if missing:
                        reason = f"missing context_phase_params for pairs {missing}"
            # The missing-params check runs only on the leader (only it holds the
            # gathered params). Broadcast the verdict so EVERY rank raises together:
            # otherwise the leader raises here while the other ranks return and enter
            # ctx_finish_wave's collective, the collective sequence diverges, and the
            # step deadlocks until the watchdog SIGKILLs it (misreported as TIMEOUT).
            reason = self.comm.bcast(reason, root=0)
        except _Timeout:
            raise
        except Exception as e:  # noqa: BLE001 - sends may already be in flight
            raise _FatalTransferError(f"ctx send ownership consensus failed: {e!r}") from e

        if reason is not None:
            # Send setup can fail asymmetrically after another rank dispatched
            # work. A block-all collective is not safe from that state, and
            # orderly teardown can deregister memory while a worker still owns
            # it. Conservatively abort without running finalizers.
            raise _FatalTransferError(f"ctx send setup failed: {reason}")
        return params_by_pair, reqs

    def ctx_finish_wave(self, reqs):
        """Wait for all in-flight sends of this wave, then free."""
        import tensorrt_llm

        t0 = time.monotonic()
        local_err = None
        try:
            completed, failed = self.xcvr.check_context_transfer_status(None)  # block-all
            completed_rids = set(completed)
            failed_rids = set(failed)
            # Python's block-all wait is bounded by the kv_transfer_timeout
            # deadline; a request still nonterminal on return exceeded it.
            # The precheck payloads are far smaller than real requests, so the
            # gate classifies that as a transfer failure — while retaining the
            # KV pages, because deadline expiry does not prove the peer
            # quiesced (the fatal path below skips ordinary teardown).
            missing = [
                p
                for p, req in reqs.items()
                if req.py_request_id not in completed_rids | failed_rids
            ]
            if missing:
                raise _TransferError(
                    f"block-all returned before terminal status for pairs {missing}"
                )
            failed_pairs = [p for p, req in reqs.items() if req.py_request_id in failed_rids]
            if failed_pairs:
                raise _TransferError(f"ctx transfer failed for pairs {failed_pairs}")
            bad = [
                p for p, r in reqs.items() if r.state == self.llm_request_state.DISAGG_TRANS_ERROR
            ]
            if bad:
                raise _TransferError(f"ctx DISAGG_TRANS_ERROR on pairs {bad}")
        except _Timeout:
            raise
        except Exception as e:  # noqa: BLE001
            local_err = e
        try:
            states = {p: str(r.state) for p, r in reqs.items()}
            tensorrt_llm.logger.info(
                f"[ctx{self.server_idx} r{self.rank}] wave sends finished in "
                f"{time.monotonic() - t0:.1f}s states={states}"
            )
            reason = self._consensus_error(local_err)
        except _Timeout:
            raise
        except Exception as e:  # noqa: BLE001 - release proof is incomplete
            raise _FatalTransferError(f"ctx transfer ownership consensus failed: {e!r}") from e
        if reason is not None:
            # Failed/cancelled/missing results do not prove that every NIXL
            # reader has relinquished the source pages. Do not proceed to the
            # ordinary shutdown path, which deregisters those pages.
            raise _FatalTransferError(f"ctx transfer failed: {reason}")
        # Every rank proved exact completion before any rank recycles pages.
        self._free_all(reqs)

    def gen_run_wave(self, peer_idx, li, req_len, rep, wave, params_by_pair):
        """Receive + verify owned pairs. Returns (ok, mismatch_detail).

        Warmup reps skip the (CPU-heavy) byte verification: their result is
        discarded by the caller either way, transfer errors still raise.
        """
        import torch

        import tensorrt_llm

        owned = self._owned(wave)
        reqs, local_err = {}, None
        try:
            for pair in owned:
                rid = self._pair_rid(peer_idx, li, rep, pair)
                req = make_request(
                    False, rid, req_len, self.runtime, ctx_params=params_by_pair[pair]
                )
                add_sequence(self.kvm, req, req_len, self.use_v2)
                # Track every allocated sequence before receive dispatch. On
                # setup failure, retain all pages and bypass normal teardown.
                reqs[pair] = req
                tensorrt_llm.logger.info(
                    f"[gen{self.server_idx} r{self.rank}] rid={rid} len={req_len}: recv START"
                )
                self.xcvr.request_and_receive_async(req)
        except _Timeout:
            raise
        except Exception as e:  # noqa: BLE001
            local_err = e
        try:
            reason = self._consensus_error(local_err)
        except _Timeout:
            raise
        except Exception as e:  # noqa: BLE001 - ctx sender is already live
            raise _FatalTransferError(f"gen receive setup consensus failed: {e!r}") from e
        if reason is not None:
            # The context sender is already live, and receive setup can fail
            # after partially dispatching local work. Quiescence is unknown.
            raise _FatalTransferError(f"gen receive setup failed: {reason}")

        mismatch = ""
        t0 = time.monotonic()
        transfer_error = None
        try:
            if self.runtime == "PYTHON":
                completed, failed, cancelled = self.xcvr.check_gen_transfer_status(None)
                completed_rids = set(completed)
                failed_rids = set(failed)
                cancelled_rids = {req.py_request_id for req in cancelled}
                expected_rids = {req.py_request_id for req in reqs.values()}
                # A request still nonterminal after the block-all deadline
                # (missing) is a gate failure like failed/cancelled; the fatal
                # path retains its KV pages because the peer may not have
                # quiesced.
                missing_rids = expected_rids - completed_rids - failed_rids - cancelled_rids
                if failed_rids or cancelled_rids or missing_rids:
                    raise _TransferError(
                        "Python gen block-all did not complete every request: "
                        f"failed={sorted(failed_rids)} "
                        f"cancelled={sorted(cancelled_rids)} "
                        f"missing={sorted(missing_rids)}"
                    )
            else:
                for req in reqs.values():
                    _wait_gen_complete(self.xcvr, req, self.runtime, self.llm_request_state)
            if reqs:
                tensorrt_llm.logger.info(
                    f"[gen{self.server_idx} r{self.rank}] wave recvs finished in "
                    f"{time.monotonic() - t0:.1f}s"
                )
            torch.cuda.synchronize()  # receive may land on a side stream
            bad = [
                p for p, r in reqs.items() if r.state == self.llm_request_state.DISAGG_TRANS_ERROR
            ]
            if bad:
                raise _TransferError(f"gen DISAGG_TRANS_ERROR on pairs {bad}")
            incomplete = [
                p
                for p, r in reqs.items()
                if r.state != self.llm_request_state.DISAGG_GENERATION_TRANS_COMPLETE
            ]
            if incomplete:
                raise _TransferError(f"gen requests not complete for pairs {incomplete}")
        except _Timeout:
            raise
        except Exception as e:  # noqa: BLE001
            transfer_error = e
        try:
            reason = self._consensus_error(transfer_error)
        except _Timeout:
            raise
        except Exception as e:  # noqa: BLE001 - release proof is incomplete
            raise _FatalTransferError(f"gen transfer ownership consensus failed: {e!r}") from e
        if reason is not None:
            # Exact completion plus CUDA stream synchronization is the release
            # proof. Do not let a locally successful rank free while another
            # rank still has transport-owned pages.
            raise _FatalTransferError(f"gen transfer failed: {reason}")

        # Remote writes and local CUDA work are complete on every rank. Later
        # byte-verification failures do not invalidate the ownership proof.
        verification_error = None
        try:
            if self.plan["verify_data"] and rep >= self.plan["warmup_requests"]:
                for pair, req in reqs.items():
                    ok, detail = verify_request(self.kvm, req.py_request_id, req_len)
                    if not ok:
                        mismatch = f"pair={pair} {detail}"
                        break
        except _Timeout:
            raise
        except Exception as e:  # noqa: BLE001
            verification_error = e
        self._free_all(reqs)
        reason = self._consensus_error(verification_error)
        if reason is not None:
            raise _TransferError(f"gen verification failed: {reason}")
        mismatches = [m for m in self.comm.allgather(mismatch) if m]
        return (not mismatches, "; ".join(mismatches[:4]))

    def _free_all(self, reqs):
        if reqs and self.use_v2:
            import torch

            torch.cuda.current_stream().synchronize()  # V2 frees need quiesced stream
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
        timed_out = False
        if self.is_leader:
            try:
                sock.send(pack_msg(obj, key))
                reply = unpack_msg(sock.recv(), key)
            except _Timeout as e:
                timed_out = True
                err = repr(e)
            except Exception as e:  # noqa: BLE001
                err = repr(e)
        timed_out, err, reply = self.comm.bcast((timed_out, err, reply), root=0)
        if timed_out:
            raise _Timeout(err)
        if err:
            raise _TransferError(f"ZMQ control channel failed: {err}")
        return reply

    def _leader_send_recv_with_progress(
        self,
        sock,
        obj,
        key,
        *,
        peer_role,
        peer_idx,
        what,
        timeout_s,
        arm,
    ):
        """REQ round-trip refreshed by progress from the queued target peer."""
        timeout_s = int(timeout_s)
        arm(what, seconds=timeout_s, python_alarm=False)
        send_err = None
        if self.is_leader:
            try:
                sock.send(pack_msg(obj, key))
            except Exception as e:  # noqa: BLE001 - shared via bcast below
                send_err = repr(e)
        send_err = self.comm.bcast(send_err, root=0)
        if send_err:
            raise _TransferError(f"ZMQ control send failed: {send_err}")

        return _collective_recv_with_progress(
            runner=self,
            sock=sock,
            key=key,
            peer_role=peer_role,
            peer_idx=peer_idx,
            what=what,
            timeout_s=timeout_s,
            arm=arm,
            refresh_from_peer_progress=True,
        )


def _collective_recv_with_progress(
    runner,
    sock,
    key,
    peer_role,
    peer_idx,
    what,
    timeout_s,
    arm,
    *,
    refresh_from_peer_progress,
):
    """Collectively receive control; target-peer progress resets the deadline.

    The caller arms the phase watchdog before entering this loop. Requiring an
    explicit refresh policy keeps active transfer waves on a hard deadline.
    """
    timeout_s = int(timeout_s)
    deadline = time.monotonic() + timeout_s
    last_progress = (
        read_peer_progress(runner.work_dir, peer_role, peer_idx)
        if runner.is_leader and refresh_from_peer_progress
        else None
    )

    while True:
        event = None
        if runner.is_leader:
            zmq, _ = runner._zmq()
            try:
                event = ("message", unpack_msg(sock.recv(), key))
            except zmq.Again:
                progress = (
                    read_peer_progress(runner.work_dir, peer_role, peer_idx)
                    if refresh_from_peer_progress
                    else None
                )
                if progress is not None and progress != last_progress:
                    event = ("progress", progress)
                elif time.monotonic() >= deadline:
                    event = ("timeout", None)
                else:
                    event = ("poll", None)
            except Exception as e:  # noqa: BLE001 - shared via bcast below
                event = ("error", repr(e))

        kind, payload = runner.comm.bcast(event, root=0)
        if kind == "message":
            return payload
        if kind == "error":
            raise _TransferError(f"ZMQ recv from {peer_role}_{peer_idx} failed: {payload}")
        if kind == "timeout":
            raise _Timeout(f"{what} made no progress for {timeout_s}s")
        if kind == "progress":
            last_progress = payload
            deadline = time.monotonic() + timeout_s
            arm(
                what,
                seconds=timeout_s,
                python_alarm=False,
                publish_progress=False,
            )


def _schedule(plan):
    """Deterministic (li, req_len, rep, wave) schedule both sides iterate."""
    out = []
    total_reps = plan["warmup_requests"] + plan["num_requests"]
    for li, req_len in enumerate(plan["request_lengths"]):
        for rep in range(total_reps):
            for wave in pcfg.waves(plan):
                out.append((li, req_len, rep, wave))
    return out


def hello_timeout_s(plan):
    """No-progress budget for serialized hello/bye waits.

    The target peer's progress marker refreshes this wait after each
    phase/wave, so the budget covers one slow active phase rather than
    incorrectly using either side's local peer count or multiplying every
    earlier session.
    """
    return plan["peer_progress_timeout_s"]


def wave_timeout_s(plan, li, rep):
    """Per-wave budget.

    The first rep additionally pays the one-time NIXL agent wire-up (see
    PRECHECK_DEFAULTS['wireup_timeout_s']).
    """
    extra = plan["wireup_timeout_s"] if (li == 0 and rep == 0) else 0
    return plan["wave_timeout_s"] + extra


def _recv_ctx_control(
    runner,
    sock,
    key,
    peer_idx,
    what,
    timeout_s,
    arm,
    refresh_from_gen_progress=False,
):
    """Collectively receive one ctx-side control message with progress refresh.

    Only the leader owns the ZMQ socket; every poll result is broadcast so all
    ranks execute the same watchdog checkpoints and failure branch. For queued
    hello/bye waits, observed progress from the target gen resets the deadline.
    Active wave waits do not refresh from unrelated progress.
    """
    timeout_s = int(timeout_s)
    arm(what, seconds=timeout_s, python_alarm=False)
    return _collective_recv_with_progress(
        runner=runner,
        sock=sock,
        key=key,
        peer_role="gen",
        peer_idx=peer_idx,
        what=what,
        timeout_s=timeout_s,
        arm=arm,
        refresh_from_peer_progress=refresh_from_gen_progress,
    )


# --------------------------------------------------------------------------- #
# ctx / gen session loops
#
# Per (ctx, gen) pair, the leaders speak a lockstep REQ/REP protocol (all
# frames HMAC-JSON; KV bytes themselves go through the transceiver under
# test, never over ZMQ):
#
#   gen leader                                ctx leader
#    | -- hello {fingerprint} --------------> |  yaml mismatch -> abort
#    | <---------------- welcome ------------ |
#    | -- go {li, rep, wave} --------------> |  every rank posts its sends
#    | <----- params {pair: ctx_phase} ------ |  (from the owning dp ranks)
#    |   ...KV transfer + byte verification.. |
#    |        (repeat per schedule entry)     |
#    | -- done (deferred: after ALL peers) -> |  ctx exits only now
#    | <------------------ bye -------------- |
#
# A "wave" here is a batch of TRANSFER PAIRS in flight at once (at most
# max_concurrent_pairs of the n_pairs dp-rank pairings) -- NOT token/data
# chunking. It bounds the tiny synthetic KV pool and gives the per-wave
# alarm a precise target, while still exercising concurrent transfers.
# --------------------------------------------------------------------------- #
def ctx_serve_peer(runner, sock, peer_idx, arm, disarm, key):
    """Serve one gen peer's full schedule on a dedicated REP socket."""
    plan = runner.plan

    def leader_reply(obj):
        if runner.is_leader:
            sock.send(pack_msg(obj, key))

    msg = _recv_ctx_control(
        runner,
        sock,
        key,
        peer_idx,
        f"hello gen_{peer_idx}",
        hello_timeout_s(plan),
        arm,
        refresh_from_gen_progress=True,
    )
    if msg[0] != "hello" or msg[1].get("fingerprint") != plan["fingerprint"]:
        leader_reply(("abort", "plan fingerprint mismatch (ctx/gen yaml disagree)"))
        raise _TransferError(f"handshake with gen_{peer_idx} failed: {msg[:1]}")
    leader_reply(("welcome", {"fingerprint": plan["fingerprint"]}))

    for li, req_len, rep, wave in _schedule(plan):
        msg = _recv_ctx_control(
            runner,
            sock,
            key,
            peer_idx,
            f"gen_{peer_idx} len={req_len} rep={rep}",
            wave_timeout_s(plan, li, rep),
            arm,
        )
        if msg[0] == "abort":
            # Ack so the gen's REQ send/recv completes (fail-fast teardown
            # sends this in place of the schedule; see gen_abort_peer).
            leader_reply(("aborted", {}))
            raise _PeerAbort(f"gen_{peer_idx} aborted: {msg[1]}")
        if msg[0] != "go" or (msg[1]["li"], msg[1]["rep"]) != (li, rep):
            raise _TransferError(
                f"schedule desync with gen_{peer_idx}: expected li={li} rep={rep}, got {msg}"
            )
        try:
            params_by_pair, reqs = runner.ctx_run_wave(peer_idx, li, req_len, rep, wave)
        except _FatalTransferError as e:
            try:
                leader_reply(("abort", str(e)))
            except Exception:  # noqa: BLE001 - preserve ownership-fatal outcome
                pass
            raise
        except _TransferError as e:
            leader_reply(("abort", str(e)))
            raise
        except Exception as e:  # noqa: BLE001 - dispatch may have started
            fatal = _FatalTransferError(f"ctx send path failed with ownership unknown: {e!r}")
            try:
                leader_reply(("abort", str(fatal)))
            except Exception:  # noqa: BLE001 - preserve ownership-fatal outcome
                pass
            raise fatal from e
        # JSON object keys are strings; the gen side converts back to int.
        try:
            leader_reply(("params", {str(p): params_to_wire(v) for p, v in params_by_pair.items()}))
        except _Timeout:
            raise
        except Exception as e:  # noqa: BLE001 - sends may already be in flight
            raise _FatalTransferError(
                f"failed to publish params to gen_{peer_idx} after send dispatch: {e!r}"
            ) from e
        try:
            runner.ctx_finish_wave(reqs)
        except (_Timeout, _FatalTransferError, _TransferError):
            raise
        except Exception as e:  # noqa: BLE001 - send quiescence is unknown
            raise _FatalTransferError(
                f"ctx completion proof failed with ownership unknown: {e!r}"
            ) from e

    # The gen defers "done" until it has finished the schedules of ALL its
    # ctx peers, so every ctx instance stays alive for the whole precheck --
    # matching real serving, where ctx servers outlive the entire run. (An
    # early-exiting ctx leaves the gen's C++ transceiver holding connections
    # to a dead agent, a state the real test never produces.) The wait can
    # therefore span the gen's remaining sessions: use the same progress-aware
    # no-progress budget as the initial handshake.
    msg = _recv_ctx_control(
        runner,
        sock,
        key,
        peer_idx,
        f"bye gen_{peer_idx}",
        hello_timeout_s(plan),
        arm,
        refresh_from_gen_progress=True,
    )
    if msg[0] != "done":
        raise _TransferError(f"expected done from gen_{peer_idx}, got {msg[:1]}")
    leader_reply(("bye", {}))
    disarm()


def _gen_open_session(runner, peer_idx, arm):
    """Rendezvous with ctx server `peer_idx`; return (sock, key) on an OPEN REQ socket.

    Completes the hello/welcome handshake. Shared by gen_run_peer (runs the schedule) and gen_abort_peer (sends an
    early abort for fail-fast). Rendezvous reaches instance-wide consensus via
    bcast: if only the leader raised, the other ranks would deadlock in the
    next bcast. The socket is closed here if the handshake itself fails.
    """
    plan = runner.plan
    comm = runner.comm
    hello_s = hello_timeout_s(plan)

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
            # Poll while queued behind another gen session so ctx progress can
            # refresh the no-progress deadline.
            sock.setsockopt(zmq.RCVTIMEO, CONTROL_POLL_INTERVAL_MS)
            sock.connect(f"tcp://{addr['host']}:{addr['port']}")
        except Exception as e:  # noqa: BLE001 - shared via bcast below
            err = repr(e)
    err = comm.bcast(err, root=0)
    if err:
        raise _TransferError(f"rendezvous with ctx_{peer_idx} failed: {err}")

    try:
        reply = runner._leader_send_recv_with_progress(
            sock,
            ("hello", {"gen_idx": runner.server_idx, "fingerprint": plan["fingerprint"]}),
            key,
            peer_role="ctx",
            peer_idx=peer_idx,
            what=f"hello ctx_{peer_idx}",
            timeout_s=hello_s,
            arm=arm,
        )
        if reply[0] == "abort":
            raise _TransferError(f"ctx_{peer_idx} aborted handshake: {reply[1]}")
        if reply[0] != "welcome":
            raise _TransferError(f"unexpected handshake reply from ctx_{peer_idx}: {reply[:1]}")
        if runner.is_leader:
            zmq, _ = runner._zmq()
            sock.setsockopt(zmq.RCVTIMEO, hello_s * 1000)
        return sock, key
    except BaseException:
        if sock is not None:
            sock.close(linger=0)
        raise


def gen_run_peer(runner, peer_idx, arm, disarm):
    """Run the full schedule against ctx server `peer_idx`.

    Returns (sock, key) with the session STILL OPEN on success -- the caller
    sends the deferred "done" only after every ctx peer's schedule finished,
    keeping all ctx instances alive for the whole precheck (real-serving
    lifecycle; see ctx_serve_peer). On failure the socket is closed here.
    """
    plan = runner.plan
    sock, key = _gen_open_session(runner, peer_idx, arm)
    try:
        # Established sessions are dedicated: wave replies are prompt. The
        # ZMQ timeout is only a backstop under the per-wave alarm, so it
        # includes the first-rep wire-up allowance unconditionally.
        if runner.is_leader:
            zmq, _ = runner._zmq()
            sock.setsockopt(
                zmq.RCVTIMEO,
                (plan["wave_timeout_s"] + plan["wireup_timeout_s"] + 30) * 1000,
            )

        case_ok = {}
        for li, req_len, rep, wave in _schedule(plan):
            arm(f"ctx_{peer_idx} len={req_len} rep={rep}", seconds=wave_timeout_s(plan, li, rep))
            try:
                reply = runner._leader_send_recv(
                    sock, ("go", {"li": li, "rep": rep, "wave": wave[0]}), key
                )
                if reply[0] == "abort":
                    raise _FatalTransferError(f"ctx_{peer_idx} aborted: {reply[1]}")
                if reply[0] != "params":
                    raise _FatalTransferError(
                        f"unexpected wave reply from ctx_{peer_idx}: {reply[:1]}"
                    )
                params_by_pair = {int(p): params_from_wire(v) for p, v in reply[1].items()}
            except _Timeout:
                raise
            except _FatalTransferError:
                raise
            except Exception as e:  # noqa: BLE001 - ctx may have dispatched sends
                raise _FatalTransferError(
                    f"wave control failed after ctx_{peer_idx} may have dispatched sends: {e!r}"
                ) from e
            try:
                ok, detail = runner.gen_run_wave(peer_idx, li, req_len, rep, wave, params_by_pair)
            except (_Timeout, _FatalTransferError, _TransferError):
                raise
            except Exception as e:  # noqa: BLE001 - receive quiescence is unknown
                raise _FatalTransferError(
                    f"gen completion proof failed with ownership unknown: {e!r}"
                ) from e
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


def gen_abort_peer(runner, peer_idx, reason, arm, disarm):
    """Fail-fast teardown of a not-yet-run ctx peer.

    Open the session and tell it to abort (it is blocked awaiting our hello),
    so it stops promptly instead of waiting out the handshake alarm.
    Best-effort; always closes.
    """
    sock = None
    try:
        sock, key = _gen_open_session(runner, peer_idx, arm)
        arm(f"abort ctx_{peer_idx}", seconds=hello_timeout_s(runner.plan))
        runner._leader_send_recv(sock, ("abort", f"peer fail-fast: {reason}"), key)
        disarm()
    finally:
        if sock is not None:
            sock.close(linger=0)


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
    # e2e_time_breakdown is an e2e run that additionally uploads per-request
    # lifecycle spans; the KV transfer it prechecks is identical. It has to be
    # listed here even though resolve_plan only distinguishes gen_only, because
    # submit.py forwards the test's mode verbatim and argparse would reject it.
    ap.add_argument(
        "--benchmark-mode",
        default="e2e",
        choices=["e2e", "e2e_time_breakdown", "gen_only"],
    )
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


def _install_watchdog(runner, plan, rank):
    """Two-layer stall protection around every phase of the run.

    signal.alarm catches Python-level stalls; HangDetector catches
    GIL-released native hangs (dumps stacks, records TIMEOUT, SIGKILLs so
    `srun --kill-on-bad-exit` tears the step down). The external `timeout`
    around the srun is the guaranteed backstop for GIL-held hangs.

    Returns (arm, disarm, stop, current_cell): per-phase alarm control, the
    final shutdown, and the mutable "what phase are we in" marker used by
    failure messages.
    """
    signal.signal(signal.SIGALRM, _alarm_handler)
    current_cell = {"what": "startup"}

    def _on_hang():
        if runner.is_leader:
            try:
                runner.recorder.record(
                    "-", 0, "TIMEOUT", f"hang detected during {current_cell['what']}"
                )
                runner.recorder.finalize()
            except Exception:  # noqa: BLE001 - SIGKILL must still happen
                pass
        else:
            # Let the leader's concurrently armed watchdog replace its status
            # files before this rank's bad exit tears down the srun step.
            time.sleep(ABORT_COORDINATION_TIMEOUT_S)
        try:
            sys.stderr.write(
                f"[precheck {runner.role}_{runner.server_idx} r{rank}] WATCHDOG_KILL "
                f"{current_cell['what']}\n"
            )
            sys.stderr.flush()
        finally:
            os.kill(os.getpid(), signal.SIGKILL)

    # `arm` updates this timeout before each checkpoint. This initial value is
    # replaced before the first task is scheduled.
    hang_detector = load_internal_apis().HangDetector(
        timeout=plan["setup_timeout_s"] + pcfg.WATCHDOG_GRACE_S,
        on_detected=_on_hang,
    )
    hang_detector.start()

    def arm(what, seconds=None, python_alarm=True, publish_progress=True):
        current_cell["what"] = what
        phase_timeout_s = int(plan["wave_timeout_s"] if seconds is None else seconds)
        signal.alarm(phase_timeout_s if python_alarm else 0)
        # A Python alarm cannot interrupt a native extension. Re-arm the
        # thread-based detector with the same phase budget plus a small grace,
        # rather than one global timeout inflated by unrelated later phases.
        hang_detector.timeout = phase_timeout_s + pcfg.WATCHDOG_GRACE_S
        hang_detector.checkpoint()
        # Progress-derived watchdog refreshes must not echo a marker back to
        # the peer: reciprocal echoes could keep a genuinely stuck pair alive.
        if publish_progress:
            publish_peer_progress(runner, what)

    def disarm():
        signal.alarm(0)
        hang_detector.cancel_task()

    def stop():
        disarm()
        hang_detector.stop()

    return arm, disarm, stop, current_cell


def _make_peer_failure_recorder(runner, disarm, current_cell):
    """Exception -> verdict mapping shared by all per-peer loops.

    Recording a failure also drops the fail-fast flag so the remaining peers
    (here and in the other instances) are skipped instead of tested against a
    fabric already known bad -- see _drive_ctx_peers / raise_abort_flag.
    """

    def record_peer_failure(peer, exc):
        disarm()
        if isinstance(exc, _Timeout):
            status, reason = "TIMEOUT", f"exceeded the budget during {current_cell['what']}"
        elif isinstance(exc, _PeerAbort):
            status, reason = "TRANSFER_ERROR", str(exc)
        else:
            status, reason = "TRANSFER_ERROR", repr(exc)
        runner.recorder.record(peer, 0, status, reason)
        raise_abort_flag(runner.work_dir, f"{peer} {status}: {reason}")

    return record_peer_failure


def _hard_abort_unquiesced(runner, current_cell, exc):
    """Persist the ownership-fatal verdict, then exit without teardown."""
    try:
        signal.alarm(0)
    except Exception:  # noqa: BLE001 - abort must still happen
        pass
    if isinstance(exc, _Timeout):
        status = "TIMEOUT"
        reason = f"exceeded the budget during {current_cell['what']}"
    else:
        status = "TRANSFER_ERROR"
        reason = str(exc)
    if runner.is_leader:
        try:
            peer = current_cell["what"]
            runner.recorder.record(peer, 0, status, reason)
            raise_abort_flag(runner.work_dir, f"{peer} {status}: {reason}")
            runner.recorder.finalize(
                extra={
                    "kv_cache_manager": "V2" if runner.use_v2 else "V1",
                    "transceiver_runtime": runner.runtime,
                }
            )
        except Exception:  # noqa: BLE001 - abort must still happen
            pass
    # Fatal transfer paths first reach instance-wide consensus. This bounded
    # rendezvous lets the leader finish replacing the status files before a
    # nonleader aborts MPI, but does not rely on coordination for rank-local
    # signal timeouts.
    _coordinate_abort_after_leader_flush(runner.comm)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        _hard_abort_process(runner.comm)


def _consensus_abort_reason(runner):
    """Instance-wide agreed view of the fail-fast flag (leader reads, bcast).

    The flag file can appear at any moment (any instance's failure drops it),
    so per-rank reads can race it and disagree -- and the branches they select
    (gen_run_peer vs gen_abort_peer, failure vs SKIP verdict) issue different
    MPI collective sequences, deadlocking or cross-pairing the instance. Must
    be called collectively by every rank.
    """
    reason = abort_flag_reason(runner.work_dir) if runner.is_leader else None
    return runner.comm.bcast(reason, root=0)


def _serve_gen_peers(runner, plan, arm, disarm, record_peer_failure):
    """Ctx role: bind per-peer REP sockets, publish addrs, serve each schedule.

    One dedicated REP socket per gen peer avoids REQ interleaving across
    sessions on a shared socket. Each session gets a fresh HMAC key, shared
    only through the work-dir addr file (0600).
    """
    num_peers = runner.side["num_peers"]
    socks, keys = {}, {}
    if runner.is_leader:
        zmq, zctx = runner._zmq()
        host = os.environ.get("SLURMD_NODENAME") or socket.gethostname()
        for gj in range(num_peers):
            s = zctx.socket(zmq.REP)
            s.setsockopt(zmq.LINGER, 0)
            # Poll so queued ctx sessions can observe target-gen progress and
            # refresh a no-progress watchdog without cumulative peer budgets.
            s.setsockopt(zmq.RCVTIMEO, CONTROL_POLL_INTERVAL_MS)
            port = s.bind_to_random_port("tcp://*")
            keys[gj] = secrets.token_bytes(32)
            write_addr(
                addr_path(runner.work_dir, runner.server_idx, gj),
                {"host": host, "port": port, "key": keys[gj].hex()},
            )
            socks[gj] = s
    for gj in range(num_peers):
        try:
            ctx_serve_peer(runner, socks.get(gj), gj, arm, disarm, keys.get(gj))
            runner.recorder.record(f"gen_{gj}", 0, "PASS", "served all transfers")
        except (_FatalTransferError, _Timeout):
            raise
        except _PeerAbort as e:
            # A gen driver that failed elsewhere aborts our session as part of
            # fail-fast: record a (non-failing) SKIP, not our own failure --
            # the real failure is recorded by whoever hit it. Absent the flag,
            # a genuine peer abort is still a real failure. The consensus read
            # is collectively safe here: _PeerAbort is only raised after a
            # bcast (leader_recv), so every rank reaches this handler.
            if _consensus_abort_reason(runner) is not None:
                runner.recorder.record(f"gen_{gj}", 0, "SKIP", f"aborted by fail-fast: {e}")
            else:
                record_peer_failure(f"gen_{gj}", e)
        except Exception as e:  # noqa: BLE001 - per-peer isolation
            record_peer_failure(f"gen_{gj}", e)


def _drive_ctx_peers(runner, arm, disarm, record_peer_failure):
    """Gen role: run every ctx peer's schedule, then release all sessions.

    Fail-fast: once any pair has failed (this instance or another -- signalled
    through the work-dir abort flag), the remaining ctx peers are not tested;
    each is told to abort so it tears down promptly rather than waiting out its
    handshake alarm. Sessions that already succeeded still get their deferred
    "done" (below) so those ctx instances shut down cleanly.

    The release ("done") is deferred until EVERY driven peer's schedule
    finished, so those ctx instances stay alive for the whole precheck --
    matching real serving, where no transceiver ever holds connections to a
    dead agent while transfers are still running.
    """
    open_sessions = []
    for ci in range(runner.side["num_peers"]):
        reason = _consensus_abort_reason(runner)
        if reason is not None:
            try:
                gen_abort_peer(runner, ci, reason, arm, disarm)
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
            runner.recorder.record(f"ctx_{ci}", 0, "SKIP", f"fail-fast: {reason}")
            continue
        try:
            sock, sess_key = gen_run_peer(runner, ci, arm, disarm)
            open_sessions.append((ci, sock, sess_key))
        except (_FatalTransferError, _Timeout):
            raise
        except Exception as e:  # noqa: BLE001 - failure sets the fail-fast flag
            record_peer_failure(f"ctx_{ci}", e)
    for ci, sock, sess_key in open_sessions:
        try:
            gen_release_peer(runner, ci, sock, sess_key, arm, disarm)
        except Exception as e:  # noqa: BLE001 - best-effort release
            record_peer_failure(f"ctx_{ci}", e)


def main(argv=None):
    args = parse_args(argv)
    plan, side = load_plan(args)

    if args.dry_run:
        print(json.dumps(plan, indent=2, default=str))
        return 0
    if plan.get("skip"):
        print(f"[precheck] SKIP: {plan['skip_reason']}", flush=True)
        return 0

    # UCX_PROTO_INFO=used is log-only (does not change transport selection):
    # it makes UCX >= 1.21 print the chosen GPU<->GPU protocol table, which the
    # failure summary uses to spot host-staged tcp fallbacks.
    os.environ.setdefault("UCX_PROTO_INFO", "used")

    # PRECHECK_DEBUG=1: verbose C++/Python transceiver logs for stall
    # debugging. Must be set before importing tensorrt_llm (the C++ logger
    # reads TLLM_LOG_LEVEL at init).
    debug = os.environ.get("PRECHECK_DEBUG") == "1"
    if debug:
        os.environ.setdefault("TLLM_LOG_LEVEL", "DEBUG")

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
    tensorrt_llm.logger.set_level("debug" if debug else "info")

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

    arm, disarm, stop_watchdog, current_cell = _install_watchdog(runner, plan, rank)

    # --- setup: KV pool + transceiver (same config as the real test) ---------
    setup_err = None
    try:
        arm("kv pool + transceiver setup", seconds=plan["setup_timeout_s"])
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

    record_peer_failure = _make_peer_failure_recorder(runner, disarm, current_cell)
    try:
        if args.role == "ctx":
            _serve_gen_peers(runner, plan, arm, disarm, record_peer_failure)
        else:
            _drive_ctx_peers(runner, arm, disarm, record_peer_failure)
    except (_FatalTransferError, _Timeout) as e:
        try:
            print(
                f"[precheck {runner.role}_{runner.server_idx} r{rank}] OWNERSHIP_FATAL: {e}",
                file=sys.stderr,
                flush=True,
            )
        except Exception:  # noqa: BLE001 - abort must still happen
            pass
        _hard_abort_unquiesced(runner, current_cell, e)
    finally:
        stop_watchdog()

    # --- teardown + result ------------------------------------------------------
    # Bandwidth lives on different sides per transceiver: C++ records it on the
    # receiver (gen recv CSVs), the Python transceiver on the sender (ctx perf
    # CSVs). csv_dir is shared across the instance's ranks, so for the Python
    # path the leader alone medians over all ranks' perf files.
    bw = None
    if runner.runtime == "PYTHON":
        if args.role == "ctx" and runner.is_leader:
            bw = parse_python_bandwidth_gbps(runner.csv_dir)
    elif args.role == "gen":
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

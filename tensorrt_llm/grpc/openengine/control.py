# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenEngine Control service backed by the TensorRT-LLM LLM API.

Reported capabilities describe what ``Generate`` accepts, not how the engine was
built: a capability a client acts on and ``Generate`` then rejects turns
discovery into a per-request failure.

The LoRA lifecycle RPCs return ``UNIMPLEMENTED``: the LLM API has no runtime
adapter load/unload. KV-event discovery advertises the engine's streaming
publisher, which is off unless ``kv_cache_config.kv_events_config`` enables it.
Event delivery is direct ZMQ; ``SubscribeKvEvents`` remains unimplemented.
"""

import asyncio
import time
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Optional

import grpc
from openengine.v1 import (
    kv_pb2,
    lifecycle_pb2,
    lora_pb2,
    model_pb2,
    openengine_pb2_grpc,
    server_pb2,
)

from tensorrt_llm import __version__ as trtllm_version
from tensorrt_llm._torch.pyexecutor.kv_cache_events import HEARTBEAT_INTERVAL_SECONDS
from tensorrt_llm.logger import logger
from tensorrt_llm.sampling_params import MAX_TOP_LOGPROBS

from .capabilities import supported_guides
from .errors import AbortFailedError
from .kv_events import (
    KvEventsUnavailableError,
    data_parallel_size,
    events_config,
    resolve_advertise_host,
    resolve_sources,
    to_proto_source,
)

__all__ = ["OpenEngineControlServicer"]

_SCHEMA_REVISION = 1
_MINIMUM_CLIENT_REVISION = 1
# BSR commit of the openengine schema this server was generated against. Derived
# from the installed bindings rather than hand-copied: the version is
# "<v>+<bsr commit>", and a hardcoded copy would keep advertising a stale
# release after requirements-openengine.txt is bumped.
_SCHEMA_PACKAGE = "openengine-openengine-protocolbuffers-python"


def _schema_release() -> str:
    try:
        return version(_SCHEMA_PACKAGE).rpartition("+")[2]
    except PackageNotFoundError:
        logger.warning(f"OpenEngine schema package '{_SCHEMA_PACKAGE}' is not installed")
        return ""


_ENGINE_NAME = "tensorrt_llm"

# Translation only. Which guides a backend can build is defined once, in
# `servicer.GUIDE_SUPPORT_BY_BACKEND`, so the advertisement cannot drift from
# what Generate enforces.
_MODE_BY_GUIDE_FIELD = {
    "json_schema": model_pb2.GUIDED_DECODING_MODE_JSON_SCHEMA,
    "regex": model_pb2.GUIDED_DECODING_MODE_REGEX,
    "ebnf_grammar": model_pb2.GUIDED_DECODING_MODE_EBNF_GRAMMAR,
    "structural_tag": model_pb2.GUIDED_DECODING_MODE_STRUCTURAL_TAG,
    "json_object": model_pb2.GUIDED_DECODING_MODE_JSON_OBJECT,
}

_INFERENCE_PROBE_TIMEOUT_SECONDS = 30.0
_LOAD_SNAPSHOT_CACHE_SECONDS = 0.05


def _abort_quietly(handle: Any, reason: str) -> None:
    """Abort a probe request without letting cleanup mask the original outcome."""
    if handle is None:
        return
    try:
        handle.abort()
    except Exception as error:
        logger.warning(f"OpenEngine {reason}; aborting the probe request failed: {error}")


def _positive_int(value: Any) -> Optional[int]:
    """Return ``value`` as a positive int, else None.

    These limits are proto3 ``optional``: unset means "unknown", which is not
    the same claim as a limit of zero.
    """
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


class OpenEngineControlServicer(openengine_pb2_grpc.ControlServicer):
    """Serve OpenEngine Control RPCs for one TensorRT-LLM engine.

    Args:
        llm: Initialized TensorRT-LLM LLM instance.
        model: Model name this server advertises.
        inference: The paired inference servicer, whose in-flight request table
            backs ``Abort`` and ``GetLoad``.
        kv_transfer_backend: KV cache transfer backend name, when configured.
        bind_host: Interface the server listens on. Used to name a routable host
            for KV event sources when the publishers bind a wildcard.
    """

    def __init__(
        self,
        llm: Any,
        model: str,
        inference: Any,
        kv_transfer_backend: str = "",
        bind_host: str = "",
    ) -> None:
        self._llm = llm
        self._model = model
        self._inference = inference
        self._kv_transfer_backend = kv_transfer_backend
        self._advertise_host = resolve_advertise_host(bind_host)
        self._probe: Optional[asyncio.Future] = None
        self._load_snapshot: dict = {}
        self._load_snapshot_deadline = 0.0
        self._load_snapshot_lock = asyncio.Lock()
        self._load_reporting_enabled = bool(getattr(llm.args, "_enable_routing_load", False))

    async def _get_kv_cache_load(self) -> dict:
        """Coalesce concurrent control-plane polls into one engine query."""
        if not self._load_reporting_enabled:
            return {}
        now = time.monotonic()
        if now < self._load_snapshot_deadline:
            return self._load_snapshot
        async with self._load_snapshot_lock:
            now = time.monotonic()
            if now >= self._load_snapshot_deadline:
                executor = getattr(self._llm, "_executor", None)
                getter = getattr(executor, "get_kv_cache_load", None)
                self._load_snapshot = await asyncio.to_thread(getter) if callable(getter) else {}
                self._load_snapshot_deadline = now + _LOAD_SNAPSHOT_CACHE_SECONDS
        return self._load_snapshot

    def _engine_is_healthy(self) -> bool:
        """Readiness via the predicate the engine's own HTTP /health uses.

        `_check_health` is stronger than the `is_shutdown()` guard generate_async
        raises on: it drains the executor's error queue, so a fatal error queued
        but not yet promoted is caught, and the proxy polls MPI worker liveness.
        It also covers encode-only engines, where `_executor` is None.
        """
        check = getattr(self._llm, "_check_health", None)
        if callable(check):
            # Only the engine's own failure modes mean "not ready". A TypeError
            # or AttributeError here is _check_health's contract drifting, and
            # reporting that as an unhealthy engine is the worst answer a health
            # endpoint can give: it looks like the thing it is meant to detect.
            try:
                return bool(check())
            except (RuntimeError, ValueError, OSError) as error:
                logger.warning(f"OpenEngine health check reported a failure: {error}")
                return False
        executor = getattr(self._llm, "_executor", None)
        return executor is not None and not executor.is_shutdown()

    # -- Runtime metadata ---------------------------------------------------

    async def GetServerInfo(
        self,
        request: server_pb2.GetServerInfoRequest,
        context: grpc.aio.ServicerContext,
    ) -> server_pb2.ServerInfo:
        info = server_pb2.ServerInfo(
            engine_name=_ENGINE_NAME,
            engine_version=str(trtllm_version),
            # The same build serves either phase; the role is per request.
            engine_role=server_pb2.ENGINE_ROLE_UNSPECIFIED,
            instance_id=str(getattr(self._llm, "llm_id", "") or ""),
            supported_models=[self._model],
            schema_revision=_SCHEMA_REVISION,
            minimum_client_revision=_MINIMUM_CLIENT_REVISION,
            schema_release=_schema_release(),
        )

        parallelism = server_pb2.ParallelismInfo()
        args = self._llm.args
        for field, value in (
            ("tensor_parallel_size", args.tensor_parallel_size),
            ("pipeline_parallel_size", args.pipeline_parallel_size),
            ("decode_context_parallel_size", args.context_parallel_size),
            # One KV event publisher binds per attention-DP rank, so a client
            # subscribing to GetKvEventSources needs this to tell a complete set
            # of sources from a partial one.
            ("data_parallel_size", data_parallel_size(self._llm)),
        ):
            size = _positive_int(value)
            if size is not None:
                setattr(parallelism, field, size)
        info.parallelism.CopyFrom(parallelism)

        if self._kv_transfer_backend:
            info.kv_connector.CopyFrom(
                kv_pb2.KvConnectorInfo(
                    enabled=True,
                    transfer_backend=self._kv_transfer_backend,
                    supports_remote_prefill=True,
                    # KV moves via the context worker's opaque_state, not a
                    # separate decode-pull channel.
                    supports_decode_pull=False,
                    # Abort(kv_session) is UNIMPLEMENTED: KvSessionRef.session_id
                    # is the engine's context request id, not a Generate
                    # request_id, so it cannot be resolved to an in-flight
                    # request. Advertising cleanup a client cannot invoke would
                    # have it drop a prefill whose KV blocks stay pinned.
                    supports_abort_cleanup=False,
                    schema_version=_SCHEMA_REVISION,
                )
            )

        capacity = server_pb2.DeploymentCapacity()
        max_batch_size = _positive_int(self._llm.args.max_batch_size)
        if max_batch_size is not None:
            capacity.max_running_requests = max_batch_size
        max_num_tokens = _positive_int(self._llm.args.max_num_tokens)
        if max_num_tokens is not None:
            capacity.max_batched_tokens = max_num_tokens
        block_size = _positive_int(self._llm.args.kv_cache_config.tokens_per_block)
        if block_size is not None:
            capacity.kv_block_size = block_size
        executor = getattr(self._llm, "_executor", None)
        capacity_getter = getattr(executor, "get_kv_cache_capacity", None)
        if callable(capacity_getter):
            kv_capacity = await asyncio.to_thread(capacity_getter)
            node = kv_capacity.get("openEngineNode")
            if node is not None:
                info.extra.update({"trtllm_node": node})
            local_blocks = _positive_int(kv_capacity.get("maxNumBlocks"))
            if local_blocks is not None:
                capacity.total_kv_blocks = local_blocks * data_parallel_size(self._llm)
        info.capacity.CopyFrom(capacity)

        # Optional engine metadata, not a routing policy or a protocol extension.
        # The request handler still validates every strict DP-rank hint.
        info.extra.update({"trtllm_supports_dp_rank_targeting": True})
        if events_config(self._llm) is not None:
            info.extra.update(
                {"kv_event_heartbeat_interval_ms": int(HEARTBEAT_INTERVAL_SECONDS * 1000)}
            )
        if self._load_reporting_enabled:
            snapshot = await self._get_kv_cache_load()
            if snapshot.get("ranks"):
                info.capacity.total_kv_blocks = snapshot["totalKvBlocks"]

        return info

    async def GetModelInfo(
        self,
        request: model_pb2.GetModelInfoRequest,
        context: grpc.aio.ServicerContext,
    ) -> model_pb2.ModelInfo:
        # Single-model server: like Generate, any non-empty name resolves to the
        # loaded model.
        info = model_pb2.ModelInfo(
            model_id=self._model,
            served_model_name=self._model,
            # Without a tokenizer the input processor rejects every string prompt.
            supports_text_input=getattr(self._llm, "tokenizer", None) is not None,
            supports_token_ids_input=True,
            # Generate rejects `lora_name` even on an enable_lora build.
            supports_lora=False,
            supports_multimodal=False,
            reasoning_parser=str(self._llm.args.reasoning_parser or ""),
        )

        max_seq_len = _positive_int(self._llm.args.max_seq_len)
        if max_seq_len is not None:
            info.max_context_length = max_seq_len
        # max_output_tokens stays unset. max_seq_len is the combined input+output
        # window, not an output budget: the real ceiling is max_seq_len minus the
        # prompt, which is per request and cannot be advertised statically.
        # Reporting the window here would tell a client with a 4000-token prompt
        # that it may ask for 4096 more, and Generate would reject it -- the
        # discovery-becomes-a-per-request-failure case this service exists to
        # avoid. Unset means "unknown", which is the honest claim.

        generation = model_pb2.GenerationCapabilities()
        # The bound is enforced per request, so it is known rather than unknown.
        logprobs = model_pb2.LogprobCapabilities(
            supported=True,
            candidate_selection_modes=[model_pb2.CANDIDATE_TOKEN_SELECTION_MODE_TOP_N],
            max_top_n=MAX_TOP_LOGPROBS,
        )
        generation.output_logprobs.CopyFrom(logprobs)
        generation.prompt_logprobs.CopyFrom(logprobs)
        guided_backend = str(self._llm.args.guided_decoding_backend or "")
        generation.guided_decoding.CopyFrom(
            model_pb2.GuidedDecodingCapabilities(
                supported=bool(guided_backend),
                modes=sorted(
                    _MODE_BY_GUIDE_FIELD[field]
                    for field in supported_guides(guided_backend)
                    if field in _MODE_BY_GUIDE_FIELD
                )
                if guided_backend
                else [],
            )
        )
        # max_num_sequences stays unset: n > 1 is served by sampling, not beam
        # search, so max_beam_width is not its limit.
        generation.supports_priority = False  # Generate rejects openengine-priority
        generation.supports_stop_in_output = True
        generation.supports_cache_salt = True
        generation.supports_prefix_cache_bypass = False  # Generate rejects it
        info.generation.CopyFrom(generation)

        return info

    async def GetLoad(
        self,
        request: server_pb2.GetLoadRequest,
        context: grpc.aio.ServicerContext,
    ) -> server_pb2.LoadInfo:
        snapshot = await self._get_kv_cache_load()
        if self._load_reporting_enabled and not snapshot.get("ranks"):
            await context.abort(
                grpc.StatusCode.UNAVAILABLE,
                "KV occupancy is enabled but no load snapshot is available",
            )
        load = server_pb2.LoadInfo(instance_id=str(getattr(self._llm, "llm_id", "") or ""))
        # This timestamp describes when the RPC observed the snapshot. The
        # scheduler sample itself may remain unchanged while a healthy engine is
        # idle; exposing its age here would make routers reject healthy workers.
        load.timestamp_unix_nanos = time.time_ns()
        load.running_requests = self._inference.active_request_count()
        used_blocks = snapshot.get("usedKvBlocks")
        total_blocks = snapshot.get("totalKvBlocks")
        if isinstance(used_blocks, int) and used_blocks >= 0:
            load.used_kv_blocks = used_blocks
        if isinstance(total_blocks, int) and total_blocks > 0:
            load.total_kv_blocks = total_blocks
        if request.include_per_rank:
            for rank in snapshot.get("ranks", []):
                rank_load = load.ranks.add()
                rank_load.data_parallel_rank = int(rank["rank"])
                rank_used = rank.get("usedKvBlocks")
                rank_total = rank.get("totalKvBlocks")
                rank_running = rank.get("runningRequests")
                if data_parallel_size(self._llm) == 1:
                    rank_running = load.running_requests
                if isinstance(rank_used, int) and rank_used >= 0:
                    rank_load.used_kv_blocks = rank_used
                if isinstance(rank_total, int) and rank_total > 0:
                    rank_load.total_kv_blocks = rank_total
                if isinstance(rank_running, int) and rank_running >= 0:
                    rank_load.running_requests = rank_running
        return load

    # -- Health and lifecycle ----------------------------------------------

    async def Health(
        self,
        request: lifecycle_pb2.HealthRequest,
        context: grpc.aio.ServicerContext,
    ) -> lifecycle_pb2.HealthResponse:
        checks = [
            lifecycle_pb2.HealthCheck(
                name="grpc",
                state=lifecycle_pb2.HEALTH_STATE_READY,
                message="serving",
            )
        ]

        model_ready = self._engine_is_healthy()
        checks.append(
            lifecycle_pb2.HealthCheck(
                name="model",
                state=(
                    lifecycle_pb2.HEALTH_STATE_READY
                    if model_ready
                    else lifecycle_pb2.HEALTH_STATE_NOT_READY
                ),
                message=self._model if model_ready else "engine health check failed",
            )
        )

        if self._kv_transfer_backend:
            checks.append(
                lifecycle_pb2.HealthCheck(
                    name="kv_connector",
                    state=lifecycle_pb2.HEALTH_STATE_READY,
                    message=self._kv_transfer_backend,
                )
            )

        if request.include_inference_probe:
            checks.append(await self._inference_probe(request.model or self._model))

        state = lifecycle_pb2.HEALTH_STATE_READY
        for check in checks:
            if check.state != lifecycle_pb2.HEALTH_STATE_READY:
                state = check.state
                break
        return lifecycle_pb2.HealthResponse(state=state, checks=checks)

    async def _inference_probe(self, model: str) -> lifecycle_pb2.HealthCheck:
        """Run a bounded single-token generation and report it as a check.

        A stuck scheduler is what the probe exists to detect, so an unbounded
        wait would hang the RPC in the one case that matters; every non-success
        exit aborts the request so a repeating probe cannot leak one per attempt.

        Concurrent callers share one probe. A readiness loop polling faster than
        the timeout would otherwise stack a fresh engine request per attempt --
        against exactly the wedged engine that makes them slow -- and those
        requests occupy scheduler slots that GetLoad does not report, so a
        load-balancing router would send the engine more traffic, not less.
        """
        probe = self._probe
        if probe is None or probe.done():
            probe = asyncio.ensure_future(self._run_inference_probe(model))
            self._probe = probe
        return await asyncio.shield(probe)

    async def _run_inference_probe(self, model: str) -> lifecycle_pb2.HealthCheck:
        from tensorrt_llm.sampling_params import SamplingParams

        handle = None
        # Counted by GetLoad and reachable by Abort(all_requests): a probe
        # occupies a scheduler slot like any other request, and a router sizing
        # itself on running_requests must see it.
        probe_id = f"__openengine_health_probe__{time.time_ns()}"
        try:
            handle = self._llm.generate_async(
                [1],
                sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
                streaming=False,
            )
            self._inference.track_request(probe_id, handle)
            await asyncio.wait_for(handle.aresult(), timeout=_INFERENCE_PROBE_TIMEOUT_SECONDS)
            return lifecycle_pb2.HealthCheck(
                name="inference_probe",
                state=lifecycle_pb2.HEALTH_STATE_READY,
                message=model,
            )
        except asyncio.TimeoutError:
            _abort_quietly(handle, "health inference probe timed out")
            return lifecycle_pb2.HealthCheck(
                name="inference_probe",
                state=lifecycle_pb2.HEALTH_STATE_DEGRADED,
                message=(f"probe did not complete within {_INFERENCE_PROBE_TIMEOUT_SECONDS}s"),
            )
        # CancelledError is a BaseException, so `except Exception` misses it.
        except asyncio.CancelledError:
            _abort_quietly(handle, "health inference probe cancelled")
            raise
        except Exception as error:
            _abort_quietly(handle, "health inference probe failed")
            logger.warning(f"OpenEngine health inference probe failed: {error}")
            return lifecycle_pb2.HealthCheck(
                name="inference_probe",
                state=lifecycle_pb2.HEALTH_STATE_DEGRADED,
                message=str(error),
            )
        finally:
            self._inference.untrack_request(probe_id, handle)

    async def Abort(
        self,
        request: lifecycle_pb2.AbortRequest,
        context: grpc.aio.ServicerContext,
    ) -> lifecycle_pb2.AbortResponse:
        target = request.WhichOneof("target")

        if target == "request_id":
            # AbortStatus has no failure value, and reporting ALREADY_FINISHED
            # for a refused abort would make the caller stop tracking a request
            # that is still running and still holding KV blocks.
            try:
                aborted = self._inference.abort_request_by_id(request.request_id)
            except AbortFailedError as error:
                await context.abort(grpc.StatusCode.INTERNAL, str(error))
                raise  # context.abort raises; keep that explicit for the reader
            # An already-finished or never-seen request is not an error: the
            # caller's intent already holds.
            return lifecycle_pb2.AbortResponse(
                status=(
                    lifecycle_pb2.ABORT_STATUS_ABORTED
                    if aborted
                    else lifecycle_pb2.ABORT_STATUS_ALREADY_FINISHED
                ),
                message=request.request_id,
            )

        if target == "all_requests":
            aborted, failed = self._inference.abort_all_requests()
            if failed:
                await context.abort(
                    grpc.StatusCode.INTERNAL,
                    f"aborted {aborted} request(s); {failed} could not be aborted "
                    "and are still running",
                )
            return lifecycle_pb2.AbortResponse(
                status=(
                    lifecycle_pb2.ABORT_STATUS_ABORTED
                    if aborted
                    else lifecycle_pb2.ABORT_STATUS_ALREADY_FINISHED
                ),
                message=f"aborted {aborted} request(s)",
            )

        if target == "kv_session":
            # session_id is the engine's context request id, not a Generate
            # request_id, so it cannot be resolved to an in-flight request.
            await context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "aborting by kv_session is not supported; abort by request_id",
            )

        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "AbortRequest has no target set")

    # -- Not supported by this engine ---------------------------------------

    async def LoadLora(
        self,
        request: lora_pb2.LoadLoraRequest,
        context: grpc.aio.ServicerContext,
    ) -> lora_pb2.LoadLoraResponse:
        await context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            "runtime LoRA load is not supported; configure adapters at startup",
        )

    async def UnloadLora(
        self,
        request: lora_pb2.UnloadLoraRequest,
        context: grpc.aio.ServicerContext,
    ) -> lora_pb2.UnloadLoraResponse:
        await context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            "runtime LoRA unload is not supported; configure adapters at startup",
        )

    async def ListLoras(
        self,
        request: lora_pb2.ListLorasRequest,
        context: grpc.aio.ServicerContext,
    ) -> lora_pb2.ListLorasResponse:
        await context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            "runtime LoRA enumeration is not supported",
        )

    async def _requested_sources(
        self,
        ranks: Any,
        context: grpc.aio.ServicerContext,
    ) -> list:
        """Resolve the publishers for `ranks`, or every rank when it is empty.

        Returns an empty list when KV events are disabled: the RPC is
        implemented, there is simply nothing to publish, and a client can tell
        that from "not supported" by the absence of an UNIMPLEMENTED status.
        """
        try:
            rank_hosts = None
            capacity = {}
            if events_config(self._llm) is not None:
                executor = getattr(self._llm, "_executor", None)
                getter = getattr(executor, "get_kv_cache_capacity", None)
                if callable(getter):
                    capacity = await asyncio.to_thread(getter)
                    rank_hosts = capacity.get("kvEventRankHosts")
            if "kvEventSources" in capacity:
                sources = [kv_pb2.KvEventSource(**source) for source in capacity["kvEventSources"]]
            else:
                sources = resolve_sources(self._llm, self._advertise_host, rank_hosts)
        except KvEventsUnavailableError as error:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
        if not sources:
            return []
        if not ranks:
            return sources
        by_rank = {source.data_parallel_rank: source for source in sources}
        unknown = sorted(set(ranks) - by_rank.keys())
        if unknown:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"no KV event source for data-parallel rank(s) {unknown}; this engine "
                f"publishes ranks 0..{len(sources) - 1}",
            )
        # Deduplicated and ordered by rank: a client repeating a rank must not
        # get two subscriptions to the same socket.
        return [by_rank[rank] for rank in sorted(set(ranks))]

    async def GetKvEventSources(
        self,
        request: kv_pb2.GetKvEventSourcesRequest,
        context: grpc.aio.ServicerContext,
    ) -> kv_pb2.GetKvEventSourcesResponse:
        sources = await self._requested_sources(request.data_parallel_ranks, context)
        if not sources:
            return kv_pb2.GetKvEventSourcesResponse()
        config = events_config(self._llm)
        return kv_pb2.GetKvEventSourcesResponse(
            sources=[
                source
                if isinstance(source, kv_pb2.KvEventSource)
                else to_proto_source(source, config, _SCHEMA_REVISION)
                for source in sources
            ]
        )

    async def SubscribeKvEvents(
        self,
        request: kv_pb2.SubscribeKvEventsRequest,
        context: grpc.aio.ServicerContext,
    ):
        await context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            "SubscribeKvEvents is not implemented; use GetKvEventSources and subscribe directly over ZMQ",
        )

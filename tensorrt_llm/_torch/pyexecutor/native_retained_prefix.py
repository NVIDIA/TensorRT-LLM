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
"""Engine-owned retained HBM exports over the existing native transfer worker.

The small control service exchanges capabilities, not KV payload. Its thread
only queues admitted commands; all cache/session ownership changes run on the
executor thread. The configured peer map is the authority for network routes.
"""

from __future__ import annotations

import hashlib
import json
import queue
import secrets
import struct
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import zmq

from tensorrt_llm._torch.disaggregation.base import TokenRange
from tensorrt_llm._torch.disaggregation.native.extent import Chunk
from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
from tensorrt_llm._torch.disaggregation.native.retained import (
    _bind_provider_claim,
    _require_retirement,
    create_retained_rx_session,
    create_retained_tx_session,
    describe_retained_prefix,
)
from tensorrt_llm._torch.disaggregation.native.retained_coverage import (
    MAX_PLAN_BYTES_PER_CLAIM,
    VERSION,
    ClaimBinding,
    certify_destination,
    logical_layout_identity,
    preview_write_plan,
    snapshot_cache,
    snapshot_from_trusted_wire,
)
from tensorrt_llm._torch.disaggregation.native.retained_profile import make_retained_admission
from tensorrt_llm._torch.disaggregation.native.transfer import WaitResult
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.transceiver import _chunk_bytes_by_kind
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.logger import logger
from tensorrt_llm.retained_prefix import RetainedPrefixParams, parse_retained_prefix


def _hash_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class _RetainedImportReceipt:
    export_id: str
    native_claim_id: int
    source_generation: str
    destination_generation: str
    prefix_tokens: int


def record_retained_model_input(
    request, *, rank: int, tokens, positions, cached_tokens: int
) -> None:
    """Record the actual decoder packer result, never a planned suffix count."""
    receipt = request.py_retained_import_receipt
    if receipt is None:
        raise ValueError("retained model-input receipt has no verified native import")

    def digest(values):
        result = hashlib.sha256()
        for value in values:
            result.update(struct.pack("<q", int(value)))
        return result.hexdigest()

    logger.info(
        "retained_prefix "
        + json.dumps(
            {
                "event": "model_input",
                **asdict(receipt),
                "rank": rank,
                "request_id": request.py_request_id,
                "begin": request.context_current_position,
                "end": request.context_current_position + len(tokens),
                "actual_token_count": len(tokens),
                "token_ids_sha256": digest(tokens),
                "position_ids_sha256": digest(positions),
                "cached_tokens": cached_tokens,
                "hash_encoding": "little-endian-signed-int64",
            }
        )
    )


@dataclass
class _Export:
    capability: RetainedPrefixParams
    request: object
    chunk: Chunk
    expires: float
    claims: dict[str, tuple[object, object]] = field(default_factory=dict)
    closed_claims: set[str] = field(default_factory=set)
    locally_closed_claims: set[str] = field(default_factory=set)
    retired: bool = False
    expired: bool = False
    admitted: bool = False
    snapshot: object = None
    prepared: dict = field(default_factory=dict)
    finish_receipts: dict = field(default_factory=dict)
    local_outcomes: dict = field(default_factory=dict)


@dataclass
class _Command:
    message: dict
    done: threading.Event = field(default_factory=threading.Event)
    response: dict | None = None


class NativeRetainedPrefix:
    """An all-rank provider inside the framework's qualified native profile.

    This initial admission bound is independent of native mapper capabilities.
    Expanding it requires complete lifecycle and semantic qualification.
    """

    def __init__(self, executor, config) -> None:
        self.executor = executor
        self.config = config
        self.rank = executor.dist.rank
        self.world_size = executor.dist.world_size
        error = None
        try:
            self._initialize_local_prerequisites(executor, config)
        except BaseException as failure:
            error = f"{type(failure).__name__}: {failure}"
        self._agree(error)
        from .retained_identity import verify_retained_model_identity

        self.verified_identity = None
        error = None
        try:
            self.verified_identity = verify_retained_model_identity(executor, config)
        except BaseException as failure:
            error = f"{type(failure).__name__}: {failure}"
        self._agree(error)
        identities = self._gather(
            (
                config.peer_id,
                config.model_identity,
                self.verified_identity.effective_config_sha256,
                config.control_bind,
                tuple(sorted(config.trusted_peers.items())),
                config.lease_seconds,
                config.max_exports,
                config.max_claims_per_export,
            )
        )
        if any(other != identities[0] for other in identities):
            raise ValueError("retained loaded model identity differs across ranks")
        self.generation = self._broadcast(uuid.uuid4().hex if self.rank == 0 else None)
        self.layout = None
        self.exports: dict[str, _Export] = {}
        self.pending_imports: dict[str, dict] = {}
        self.retired_exports: dict[str, tuple[RetainedPrefixParams, frozenset[str], dict]] = {}
        self.lock = threading.Lock()
        self.commands: queue.Queue[_Command] = queue.Queue()
        self.stopping = threading.Event()
        self.quarantined: list[object] = []
        self.thread = None
        self.server_ready = threading.Event()
        self.server_error = None
        if config.control_bind is not None and self.rank == 0:
            try:
                self.thread = threading.Thread(
                    target=self._serve, name="retained-prefix-control", daemon=True
                )
                self.thread.start()
                if not self.server_ready.wait(10):
                    self.stopping.set()
                    self.server_error = "retained control service did not report startup"
            except BaseException as failure:
                self.stopping.set()
                self.server_error = f"{type(failure).__name__}: {failure}"
        errors = self._gather(self.server_error)
        if any(errors):
            self.stopping.set()
            raise RuntimeError(f"retained control service startup failed: {errors}")

    def _initialize_local_prerequisites(self, executor, config) -> None:
        """Local checks only; constructor votes every failure before proceeding."""
        if executor.model_engine.model.model_config.is_encoder_decoder:
            raise NotImplementedError(
                "retained prefixes do not qualify encoder/cross-attention auxiliary state"
            )
        if executor.llm_args.extra_resource_managers:
            raise NotImplementedError(
                "retained prefixes do not qualify custom auxiliary resource managers"
            )
        self.transceiver = executor.kv_cache_transceiver
        if self.transceiver is None or not hasattr(self.transceiver, "_transfer_worker"):
            raise NotImplementedError("retained prefixes require the Python native transceiver")
        self.worker = self.transceiver._transfer_worker
        manager = executor.kv_cache_manager
        if not manager.enable_block_reuse:
            raise NotImplementedError(
                "Native retained prefixes require qualified external-prefix resume semantics; "
                "the existing V2 block-reuse-disabled restored-KV correctness restriction is unresolved"
            )
        mapping = manager.mapping
        if mapping.rank != self.rank or mapping.world_size != self.world_size:
            raise ValueError("retained cache mapping differs from the executor collective")
        if (
            mapping.world_size != mapping.tp_size
            or mapping.pp_size != 1
            or mapping.cp_size != 1
            or mapping.dp_size != 1
            or mapping.enable_attention_dp
        ):
            raise NotImplementedError(
                "retained prefix provider requires a native-qualified TP profile, PP=DP=CP=1"
            )
        # Reuse the framework's resolved-tier guard: a default/falsey host
        # size is not evidence that registered pages cannot migrate.
        executor._reject_non_gpu_cache_tiers(manager)
        if executor.kv_connector_manager is not None:
            raise NotImplementedError(
                "retained native prefixes cannot be combined with another connector"
            )
        if executor.draft_model_engine is not None or executor.llm_args.max_beam_width != 1:
            raise NotImplementedError(
                "retained prefixes require one beam and no speculative decoding"
            )
        _require_retirement(self.worker, sending=True)
        _require_retirement(self.worker, sending=False)
        if config.control_bind is not None and not config.control_bind.startswith("tcp://"):
            raise ValueError("retained control_bind must be an administrator TCP binding")

    def _gather(self, value):
        return [value] if self.world_size == 1 else self.executor.dist.allgather(value)

    def _broadcast(self, value):
        return value if self.world_size == 1 else self.executor.dist.broadcast(value, 0)

    def _agree(self, error: str | None) -> None:
        errors = self._gather(error)
        if any(errors):
            raise ValueError(f"retained all-rank admission failed: {errors}")

    @property
    def pending(self) -> bool:
        return bool(self.exports or self.quarantined or not self.commands.empty())

    def protects(self, request) -> bool:
        return any(item[0] is request for item in self.quarantined)

    def _identity(self, request, tokens: int) -> str:
        if request.lora_task_id is not None or request.py_multimodal_data:
            raise NotImplementedError("retained prefixes initially support unadapted text requests")
        ids = request.get_tokens(0)
        if len(ids) < tokens:
            raise ValueError("retained prefix exceeds actual request tokens")
        digest = hashlib.sha256()
        digest.update(self.config.model_identity.encode())
        digest.update(self.layout.encode())
        digest.update(_hash_json(request.cache_salt).encode())
        for token in ids[:tokens]:
            digest.update(struct.pack("<q", int(token)))
        return digest.hexdigest()

    def prepare_exports(self, scheduled_batch) -> None:
        """Freeze one replicated completed-generation frame before responses.

        Called only from the executor's all-rank V2-generation resource frame.
        No collective is entered from a response/leader callback.
        """
        requests = {
            request.py_request_id: request for request in scheduled_batch.generation_requests
        }
        frame = []
        for key, request in sorted(requests.items()):
            params = request.py_disaggregated_params
            frame.append(
                (
                    key,
                    params is not None and params.retain_prefix,
                    request.state == LlmRequestState.GENERATION_COMPLETE,
                    request.py_retained_export is not None,
                )
            )
        frames = self._gather(frame)
        if any(other != frame for other in frames):
            raise ValueError("retained completion frame differs across ranks")
        selected = [
            key
            for key, enabled, complete, exported in frame
            if enabled and complete and not exported
        ]
        if not selected:
            return
        local = {}
        error = None
        try:
            if self.config.control_bind is None:
                raise ValueError("retain_prefix requires an enabled source control binding")
            if len(self.exports) + len(selected) > self.config.max_exports:
                raise RuntimeError("retained source export budget exhausted")
            # In overlap execution the next batch may already be writing past
            # the committed boundary. Fence those actual writes as well before
            # detaching an index slot or publishing immutable source ranges.
            torch.cuda.synchronize()
            tpb = self.executor.kv_cache_manager.tokens_per_block
            for key in selected:
                request = requests[key]
                stable = (request.max_beam_num_tokens - 1) // tpb * tpb
                if stable <= 0:
                    raise ValueError("completed source has no stable full block")
                chunk = describe_retained_prefix(self.transceiver, request, stable)
                snapshot = snapshot_cache(
                    self.transceiver,
                    request,
                    chunk,
                    engine_generation=self.generation,
                    model_identity=self.config.model_identity,
                )
                layout = logical_layout_identity(snapshot)
                if self.layout is not None and self.layout != layout:
                    raise ValueError("retained logical cache layout changed after engine startup")
                self.layout = layout
                local[key] = (chunk, stable, self._identity(request, stable), snapshot)
        except BaseException as failure:
            error = str(failure)
        self._agree(error)
        receipt = [
            (key, value[1], value[2], self.layout, len(value[0].block_ids_per_layer_groups))
            for key, value in sorted(local.items())
        ]
        if any(other != receipt for other in self._gather(receipt)):
            raise ValueError("retained stable prefix/layout/group readiness differs across ranks")
        export_ids = self._broadcast(
            {key: secrets.token_hex(32) for key in selected} if self.rank == 0 else None
        )
        error = None
        try:
            for key in selected:
                request = requests[key]
                self.executor.async_transfer_manager.start_transfer(
                    request, preserve_request_state=True
                )
                self.executor.kv_cache_manager.release_index_slot(request.py_request_id)
        except BaseException as failure:
            error = str(failure)
        errors = self._gather(error)
        if any(errors):
            self.executor._kv_connector_fail(
                f"retained source ownership admission failed: {errors}"
            )
        for key in selected:
            request = requests[key]
            chunk, stable, identity, snapshot = local[key]
            capability = RetainedPrefixParams(
                1, self.config.peer_id, self.generation, export_ids[key], stable, identity
            )
            with self.lock:
                deadline = self._broadcast(
                    time.monotonic() + self.config.lease_seconds if self.rank == 0 else None
                )
                self.exports[capability.export_id] = _Export(
                    capability, request, chunk, deadline, snapshot=snapshot
                )
            request.py_retained_export = capability
            logger.info(
                "retained_prefix "
                + json.dumps(
                    {
                        "event": "export",
                        **asdict(capability),
                        "rank": self.rank,
                        "request_id": request.py_request_id,
                        "rank_source_clock_ns": time.monotonic_ns(),
                        "leader_expiry_clock_ns": int(
                            self.exports[capability.export_id].expires * 1e9
                        ),
                    }
                )
            )
        self._agree(None)

    def _serve(self) -> None:
        socket = zmq.Context.instance().socket(zmq.REP)
        try:
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.MAXMSGSIZE, 16 * 1024 * 1024)
            socket.bind(self.config.control_bind)
        except zmq.ZMQError as error:
            self.server_error = str(error)
            self.server_ready.set()
            socket.close()
            return
        self.server_ready.set()
        try:
            while not self.stopping.is_set():
                if not socket.poll(100):
                    continue
                try:
                    message = socket.recv_json()
                    self._validate_message(message)
                    nonce = message["nonce"]
                    capability = parse_retained_prefix(message["capability"])
                    if message["action"] == "prepared":
                        with self.lock:
                            prepared = self.pending_imports.get(nonce)
                            if (
                                prepared is None
                                or prepared["capability"] != asdict(capability)
                                or prepared["identity"] != message["identity"]
                            ):
                                raise ValueError("unknown prepared retained destination")
                            response = dict(prepared)
                        socket.send_json(response)
                        continue
                    command = None
                    completed_response = None
                    with self.lock:
                        known = self.exports.get(capability.export_id)
                        valid = known is not None and known.capability == capability
                        if message["action"] in ("claim", "admit"):
                            valid = valid and not known.retired and time.monotonic() < known.expires
                        tombstone = self.retired_exports.get(capability.export_id)
                        if (
                            not valid
                            and message["action"] == "finish"
                            and tombstone is not None
                            and tombstone[0] == capability
                            and nonce in tombstone[1]
                            and message["identity"] == capability.identity
                        ):
                            completed_response = tombstone[2][nonce]
                        elif valid:
                            command = _Command(message)
                            self.commands.put(command)
                    if completed_response is not None:
                        socket.send_json(completed_response)
                        continue
                    if not valid:
                        raise ValueError("unknown, stale or expired retained source")
                    # Valid exports hold all-rank async ownership, keeping the
                    # executor's polling frame awake without thread-side MPI.
                    while not command.done.wait(0.1):
                        if self.stopping.is_set():
                            raise RuntimeError("retained source is shutting down")
                    response = command.response
                except (KeyError, ValueError, RuntimeError, zmq.ZMQError) as error:
                    response = {"error": f"{type(error).__name__}: {error}"}
                socket.send_json(response)
        finally:
            # This thread creates, binds, polls, sends and closes its socket.
            socket.close()

    @staticmethod
    def _validate_message(message: dict) -> None:
        if type(message) is not dict:
            raise ValueError("invalid retained control message")
        common = {"action", "capability", "identity", "nonce", "protocol"}
        action = message.get("action")
        extra = {
            "claim": {"destination_peer"},
            "admit": {"certificates"},
            "prepared": set(),
            "finish": set(),
        }.get(action)
        if extra is None or set(message) != common | extra or message["protocol"] != VERSION:
            raise ValueError("retained control schema/version mismatch")
        nonce = message["nonce"]
        if (
            type(nonce) is not str
            or len(nonce) != 32
            or any(c not in "0123456789abcdef" for c in nonce)
        ):
            raise ValueError("invalid retained control nonce")
        if len(json.dumps(message).encode()) > 16 * 1024 * 1024:
            raise ValueError("retained control aggregate exceeds bounded capacity")

    def _check_claim(self, message: dict):
        self._validate_message(message)
        if message["action"] not in ("claim", "admit"):
            raise ValueError("invalid retained claim action")
        capability = parse_retained_prefix(message["capability"])
        export = self.exports.get(capability.export_id)
        if (
            export is None
            or export.capability != capability
            or capability.engine_generation != self.generation
            or message["identity"] != capability.identity
            or export.expired
            or export.retired
            or self.executor.is_shutdown
        ):
            raise ValueError("retained claim identity, generation or lifetime mismatch")
        nonce = message["nonce"]
        if (
            not isinstance(nonce, str)
            or len(nonce) != 32
            or any(c not in "0123456789abcdef" for c in nonce)
        ):
            raise ValueError("invalid retained claim nonce")
        if nonce in export.closed_claims:
            raise ValueError("retained claim already completed")
        if (
            nonce not in export.claims
            and nonce not in export.prepared
            and len(set(export.claims) | set(export.prepared) | export.closed_claims)
            >= self.config.max_claims_per_export
        ):
            raise RuntimeError("retained source claim budget exhausted")
        return capability, export, nonce

    def _prepare_claim(self, message: dict, native_id: int) -> dict:
        capability, export, nonce = self._check_claim(message)
        membership = (nonce in export.prepared, nonce in export.claims)
        if any(other != membership for other in self._gather(membership)):
            self.executor._kv_connector_fail("retained source preparation membership diverged")
        if nonce in export.prepared:
            prepared = export.prepared[nonce]
            if prepared["destination_peer"] != message["destination_peer"]:
                raise ValueError("retained nonce changed its trusted destination")
            return prepared["receipt"]
        endpoint = self.config.trusted_peers.get(message["destination_peer"])
        if endpoint is None or not endpoint.startswith("tcp://"):
            raise ValueError("retained destination is not an administrator trusted peer")
        query = {key: value for key, value in message.items() if key != "destination_peer"}
        query["action"] = "prepared"
        destination_reply = self._call_source(endpoint, query)
        if (
            destination_reply.get("protocol") != VERSION
            or destination_reply.get("nonce") != nonce
            or destination_reply.get("capability") != asdict(capability)
            or destination_reply.get("identity") != capability.identity
        ):
            raise ValueError("trusted prepared destination identity mismatch")
        # Gather the already-validated immutable local snapshots before any
        # fallible parsing/preview phase. Later failures rejoin the outer vote.
        sources_wire = self._gather(export.snapshot.wire())
        destinations = tuple(
            snapshot_from_trusted_wire(value) for value in destination_reply["snapshots"]
        )
        sources = tuple(snapshot_from_trusted_wire(value) for value in sources_wire)
        plans, certificates = self._coverage(native_id, nonce, sources, destinations)
        retained_plan_bytes = sum(
            len(plan.src_ptrs) + len(plan.dst_ptrs) + len(plan.sizes)
            for item in self.exports.values()
            for prepared in item.prepared.values()
            for row in prepared["plans"]
            for plan in row
        )
        new_plan_bytes = sum(
            len(plan.src_ptrs) + len(plan.dst_ptrs) + len(plan.sizes)
            for row in plans
            for plan in row
        )
        if retained_plan_bytes + new_plan_bytes > 64 * 1024 * 1024:
            raise RuntimeError("retained provider immutable plan budget exhausted")
        claim = _bind_provider_claim(
            disagg_request_id=native_id,
            donor_generation=self.generation,
            model_identity=self.config.model_identity,
            layout_identity=self.layout,
            token_identity=capability.identity,
            prefix_tokens=capability.prefix_tokens,
            ctx_info_endpoint=self.transceiver._context_info_endpoint,
        )
        receipt = {
            "protocol": VERSION,
            "claim": asdict(claim),
            "sources": [value.wire() for value in sources],
            "plans": [[plan.receipt() for plan in cohort] for cohort in plans],
            "certificates": certificates,
        }
        if len(json.dumps(receipt).encode()) > 16 * 1024 * 1024:
            raise ValueError("retained source receipt exceeds bounded capacity")
        export.prepared[nonce] = dict(
            destination_peer=message["destination_peer"],
            claim=claim,
            plans=plans,
            sources=sources,
            destinations=destinations,
            certificates=certificates,
            receipt=receipt,
        )
        return receipt

    @staticmethod
    def _coverage(native_id, nonce, sources, destinations):
        if not sources or not destinations:
            raise ValueError("retained coverage requires both complete rank cohorts")

        def cohort(values):
            infos = [value.rank_info for value in values]
            if (
                len({info.instance_name for info in infos}) != 1
                or [info.instance_rank for info in infos] != list(range(len(values)))
                or len({value.generation for value in values}) != 1
                or len({value.token_end for value in values}) != 1
                or any(info.tp_size != len(values) for info in infos)
            ):
                raise ValueError("retained native snapshot cohort is incomplete")
            if len({logical_layout_identity(value) for value in values}) != 1:
                raise ValueError("retained logical layout differs across ranks")

        cohort(sources)
        cohort(destinations)
        if logical_layout_identity(sources[0]) != logical_layout_identity(destinations[0]):
            raise ValueError("retained source/destination logical layout mismatch")
        source_map = {
            (value.rank_info.instance_name, value.rank_info.instance_rank): value
            for value in sources
        }
        plans, certificates = [], []
        for destination in destinations:
            dst = destination.rank_info
            registrar = PeerRegistrar(dst, KVRegionExtractorV1(dst.page_table))
            selected_ranks = set(registrar.get_peer_overlap(sources[0].rank_info, 0).ranks)
            writers = {(sources[0].rank_info.instance_name, rank) for rank in selected_ranks}
            current = []
            for source in sources:
                if source.rank_info.instance_rank not in selected_ranks:
                    continue
                src = source.rank_info
                binding = ClaimBinding(
                    native_id,
                    nonce,
                    source.model_identity,
                    source.generation,
                    src.instance_name,
                    src.instance_rank,
                    destination.generation,
                    dst.instance_name,
                    dst.instance_rank,
                    source.token_end,
                )
                current.append(preview_write_plan(binding, source, destination))
            plans.append(tuple(current))
            certificates.append(certify_destination(destination, current, source_map, writers))
        if (
            sum(
                len(plan.src_ptrs) + len(plan.dst_ptrs) + len(plan.sizes)
                for row in plans
                for plan in row
            )
            > MAX_PLAN_BYTES_PER_CLAIM
        ):
            raise ValueError("retained whole-claim immutable plan budget exceeded")
        return tuple(plans), certificates

    def _claim(self, message: dict, native_id: int | None = None) -> dict:
        self._refresh_expiry()
        capability, export, nonce = self._check_claim(message)
        prepared = export.prepared.get(nonce)
        membership = (
            prepared is not None,
            nonce in export.claims,
            None if prepared is None else _hash_json(prepared["certificates"]),
        )
        if any(other != membership for other in self._gather(membership)):
            self.executor._kv_connector_fail("retained source admission metadata diverged")
        if prepared is None or message.get("certificates") != prepared["certificates"]:
            raise ValueError("retained claim lacks destination whole-cohort agreement")
        if nonce not in export.claims:
            claim = prepared["claim"]
            admission = None
            error = None
            try:
                admission = make_retained_admission(
                    self.worker,
                    self.verified_identity,
                    direction="send",
                    local_snapshot=export.snapshot,
                    sources=prepared["sources"],
                    destinations=prepared["destinations"],
                    plans=prepared["plans"],
                    certificates=prepared["certificates"],
                )
            except BaseException as failure:
                error = str(failure)
            self._agree(error)
            # Certificate construction can consume lease time. Vote a fresh
            # leader clock after all-rank validation and before registration.
            self._refresh_expiry()
            if export.expired or export.retired:
                raise ValueError("retained source expired before native admission")
            # Every rank installs the same cohort before native construction;
            # partial constructor/submission failure retains all actual peers.
            export.claims[nonce] = (claim, None)
            session = create_retained_tx_session(
                self.worker,
                claim,
                local_request_id=export.request.py_request_id,
                donor_generation=self.generation,
                admission=admission,
            )
            # Retain the session before submission: even an exception may have
            # published work and may not permit freeing its source allocation.
            export.claims[nonce] = (claim, session)
            session.send(export.chunk)
            export.admitted = True
            logger.info(
                "retained_prefix "
                + json.dumps(
                    {
                        "event": "claim",
                        "export_id": capability.export_id,
                        "native_claim_id": claim.disagg_request_id,
                        "rank": self.rank,
                        "request_id": export.request.py_request_id,
                        "source_generation": self.generation,
                        "rank_source_clock_ns": time.monotonic_ns(),
                        "leader_expiry_clock_ns": int(export.expires * 1e9),
                    }
                )
            )
        claim, _ = export.claims[nonce]
        return {"claim": asdict(claim)}

    def _check_finish(self, message: dict):
        self._validate_message(message)
        if message["action"] != "finish":
            raise ValueError("invalid retained finish action")
        capability = parse_retained_prefix(message["capability"])
        nonce = message["nonce"]
        if (
            not isinstance(nonce, str)
            or len(nonce) != 32
            or any(c not in "0123456789abcdef" for c in nonce)
        ):
            raise ValueError("invalid retained finish nonce")
        export = self.exports.get(capability.export_id)
        if (
            export is None
            or export.capability != capability
            or capability.engine_generation != self.generation
            or message["identity"] != capability.identity
            or nonce not in set(export.claims) | export.closed_claims
        ):
            raise ValueError("retained finish does not identify an admitted claim")
        return export

    def _finish(self, message: dict) -> dict:
        self._refresh_expiry()
        export = self._check_finish(message)
        export.retired = True
        nonce = message["nonce"]
        if nonce not in export.finish_receipts:
            export.finish_receipts[nonce] = {
                "retired": True,
                "expired_used": export.expired and export.admitted,
                "engine_generation": self.generation,
            }
        return export.finish_receipts[nonce]

    def _refresh_expiry(self) -> None:
        # Only the source leader's monotonic clock defines this lease.
        expired = self._broadcast(
            [
                key
                for key, export in self.exports.items()
                if (
                    not export.retired
                    or (set(export.claims) | export.closed_claims) - set(export.finish_receipts)
                )
                and time.monotonic() >= export.expires
            ]
            if self.rank == 0
            else None
        )
        for key in expired:
            if key in self.exports:
                self.exports[key].expired = True

    def poll(self) -> None:
        self._refresh_expiry()
        commands = []
        if self.rank == 0:
            while True:
                try:
                    commands.append(self.commands.get_nowait())
                except queue.Empty:
                    break
        messages = self._broadcast(
            [command.message for command in commands] if self.rank == 0 else None
        )
        for index, message in enumerate(messages):
            # Earlier metadata preparation/RPC can span a lease boundary.
            self._refresh_expiry()
            error = None
            try:
                if message["action"] in ("claim", "admit"):
                    self._check_claim(message)
                else:
                    self._check_finish(message)
            except BaseException as failure:
                error = f"{type(failure).__name__}: {failure}"
            errors = self._gather(error)
            reply = None
            if any(errors):
                reply = {"error": f"retained all-rank source admission failed: {errors}"}
            else:
                if message["action"] == "finish":
                    reply = self._finish(message)
                    if self.rank == 0:
                        commands[index].response = reply
                        commands[index].done.set()
                    continue
                native_id = self._broadcast(
                    secrets.randbits(62) | (1 << 62) if self.rank == 0 else None
                )
                error = None
                try:
                    reply = (
                        self._prepare_claim(message, native_id)
                        if message["action"] == "claim"
                        else self._claim(message, native_id)
                    )
                except BaseException as failure:
                    error = f"{type(failure).__name__}: {failure}"
                errors = self._gather(error)
                if any(errors):
                    # Some ranks may already own a native sender. Abort its
                    # logical operation, but never release its source pages
                    # until every physical owner drains and closes below.
                    capability = parse_retained_prefix(message["capability"])
                    export = self.exports[capability.export_id]
                    if message["action"] == "claim":
                        # Preparation never registers a native session. Undo
                        # partial rank-local metadata before a caller retries.
                        export.prepared.pop(message["nonce"], None)
                    record = export.claims.get(message["nonce"])
                    cancel_error = None
                    try:
                        if record is not None and record[1] is not None:
                            record[1].cancel_local()
                    except BaseException as failure:
                        cancel_error = f"{type(failure).__name__}: {failure}"
                    cancel_errors = self._gather(cancel_error)
                    if any(cancel_errors):
                        self.executor._kv_connector_fail(
                            f"retained source cancellation failed: {cancel_errors}"
                        )
                    reply = {"error": f"retained all-rank native admission failed: {errors}"}
            if self.rank == 0:
                commands[index].response = reply
                commands[index].done.set()
        keys = sorted(self.exports)
        if any(other != keys for other in self._gather(keys)):
            self.executor._kv_connector_fail("retained source export membership diverged")
        for key in keys:
            export = self.exports[key]
            nonces = sorted(export.claims)
            if any(other != nonces for other in self._gather(nonces)):
                self.executor._kv_connector_fail("retained source claim membership diverged")
            for nonce in nonces:
                _, session = export.claims[nonce]
                locally_closed = nonce in export.locally_closed_claims or session is None
                error = None
                try:
                    if not locally_closed:
                        outcome = session.wait_complete(blocking=False)
                        if outcome is not None:
                            export.local_outcomes[nonce] = getattr(outcome, "name", str(outcome))
                        if self.executor.is_shutdown and outcome is None:
                            session.cancel_local()
                        locally_closed = (
                            outcome is not None and session.resources_drained() and session.close()
                        )
                        if locally_closed:
                            export.locally_closed_claims.add(nonce)
                except BaseException as failure:
                    error = f"{type(failure).__name__}: {failure}"
                errors = self._gather(error)
                if any(errors):
                    self.executor._kv_connector_fail(
                        f"retained source physical retirement failed: {errors}"
                    )
                if all(self._gather(locally_closed)):
                    claim, _ = export.claims[nonce]
                    logger.info(
                        "retained_prefix "
                        + json.dumps(
                            {
                                "event": "source_claim_closed",
                                "export_id": key,
                                "native_claim_id": getattr(claim, "disagg_request_id", None),
                                "source_generation": self.generation,
                                "rank": self.rank,
                                "request_id": getattr(export.request, "py_request_id", None),
                                "native_session_created": session is not None,
                                "native_session_not_started": session is None,
                                "physical_resources_drained": session is not None,
                                "native_session_closed": session is not None,
                                "native_outcome": export.local_outcomes.get(nonce),
                                "rank_source_clock_ns": time.monotonic_ns(),
                            }
                        )
                    )
                    del export.claims[nonce]
                    export.closed_claims.add(nonce)
                    export.locally_closed_claims.discard(nonce)
                    export.prepared.pop(nonce, None)
            with self.lock:
                # Keep the leader's lookup/enqueue lock across the release
                # decision. A racing valid command either keeps the export
                # awake on every rank or sees the immutable tombstone; it can
                # never leave only rank zero with an idle-breaking queue item.
                release = self._broadcast(
                    (
                        (
                            export.expired
                            or self.executor.is_shutdown
                            or (
                                export.retired
                                and export.closed_claims <= set(export.finish_receipts)
                            )
                        )
                        and not export.claims
                        and self.commands.empty()
                    )
                    if self.rank == 0
                    else None
                )
                if release:
                    error = None
                    try:
                        if self.executor.async_transfer_manager.end_transfer(export.request):
                            self.executor._terminate_request(export.request)
                    except BaseException as failure:
                        error = f"{type(failure).__name__}: {failure}"
                    errors = self._gather(error)
                    if any(errors):
                        # Physical native owners have drained, but allocator
                        # teardown was rank-local. Retain evidence/export and
                        # fail-stop every rank before another collective frame.
                        self.executor._kv_connector_fail(
                            f"retained source allocator release failed: {errors}"
                        )
                    del self.exports[key]
                    receipt = {
                        "retired": True,
                        "expired_used": export.expired and export.admitted,
                        "engine_generation": self.generation,
                    }
                    self.retired_exports[key] = (
                        export.capability,
                        frozenset(export.closed_claims),
                        {
                            nonce: export.finish_receipts.get(nonce, receipt)
                            for nonce in export.closed_claims
                        },
                    )
                    if (
                        len(self.retired_exports)
                        > self.config.max_exports * self.config.max_claims_per_export
                    ):
                        del self.retired_exports[next(iter(self.retired_exports))]
            if release:
                logger.info(
                    "retained_prefix "
                    + json.dumps(
                        {
                            "event": "release",
                            "export_id": key,
                            "rank": self.rank,
                            "request_id": getattr(export.request, "py_request_id", None),
                            "rank_source_clock_ns": time.monotonic_ns(),
                            "leader_clock_ns": time.monotonic_ns() if self.rank == 0 else None,
                            "explicitly_retired": export.retired,
                            "expired_admitted": export.expired and export.admitted,
                        }
                    )
                )
        if self.executor.is_shutdown and not self.pending:
            self.stopping.set()

    def _call_source(self, endpoint: str, message: dict) -> dict:
        reply = None
        if self.rank == 0:
            socket = zmq.Context.instance().socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SNDTIMEO, 10000)
            socket.setsockopt(zmq.RCVTIMEO, 10000)
            socket.setsockopt(zmq.MAXMSGSIZE, 16 * 1024 * 1024)
            try:
                socket.connect(endpoint)
                socket.send_json(message)
                reply = socket.recv_json()
                if not isinstance(reply, dict):
                    raise ValueError("retained source returned malformed metadata")
            except (ValueError, zmq.ZMQError) as failure:
                reply = {"error": f"{type(failure).__name__}: {failure}"}
            finally:
                socket.close()
        reply = self._broadcast(reply)
        if len(json.dumps(reply).encode()) > 16 * 1024 * 1024:
            raise ValueError("retained control reply exceeds bounded capacity")
        if "error" in reply:
            raise ValueError(reply["error"])
        return reply

    def import_batch(self, scheduled_batch) -> None:
        """Consume retained references only in the final all-rank resource frame."""
        requests = {request.py_request_id: request for request in scheduled_batch.context_requests}
        frame = []
        for key, request in sorted(requests.items()):
            params = request.py_disaggregated_params
            cap = None if params is None else params.retained_prefix
            frame.append(
                (
                    key,
                    None if cap is None else asdict(parse_retained_prefix(cap)),
                    request.is_first_context_chunk,
                    request.py_retained_loaded,
                )
            )
        if any(other != frame for other in self._gather(frame)):
            raise ValueError("retained context admission frame differs across ranks")
        changed = False
        for key, cap, first, loaded in frame:
            if cap is not None and first and not loaded:
                changed |= self.import_prefix(requests[key])
        if changed:
            # The existing connector path performs this same rebuild after a
            # prefix changes the final/chunking split. Model input preparation
            # follows ResourceManager.prepare_resources and reads the updated
            # request chunk lengths; it must not use a stale full-prompt split.
            scheduled_batch.reset_context_requests()

    def import_prefix(self, request) -> bool:
        capability = parse_retained_prefix(request.py_disaggregated_params.retained_prefix)
        manager = self.executor.kv_cache_manager
        cache = manager.kv_cache_map[request.py_request_id]
        local_hits = self._gather(cache.num_committed_tokens)
        if any(value != local_hits[0] for value in local_hits):
            raise ValueError("retained import requires rank-consistent local cache admission")
        if local_hits[0]:
            # Shared local radix pages must never become remote write targets.
            # This profile deliberately computes any missing decoder tail;
            # positive reverse reuse requires independently observed zero hits.
            request.py_retained_loaded = True
            logger.info(
                "retained_prefix "
                + json.dumps(
                    {
                        "event": "fallback_local_prefix",
                        "export_id": capability.export_id,
                        "source_generation": capability.engine_generation,
                        "destination_generation": self.generation,
                        "rank": self.rank,
                        "request_id": request.py_request_id,
                        "local_prefix_tokens": local_hits[0],
                        "retained_prefix_tokens": capability.prefix_tokens,
                        "reverse_imported_tokens": 0,
                        "reason": "shared_local_prefix_stitching_not_qualified",
                    }
                )
            )
            return False
        error = None
        endpoint = identity = chunk = None
        position = capability.prefix_tokens
        chunk_size = 0
        paged = state = 0
        try:
            if position % manager.tokens_per_block or position >= request.prompt_len:
                raise ValueError("retained prefix boundary cannot serve this prompt")
            endpoint = self.config.trusted_peers.get(capability.peer_id)
            if endpoint is None or not endpoint.startswith("tcp://"):
                raise ValueError("retained source is not a configured trusted peer")
            fitted = manager._reserve_connector_prefix(request, cache, 0, position)
            if fitted is None or fitted[0] != position:
                raise RuntimeError("retained destination cannot fit the complete admitted prefix")
            _, chunk_size = fitted
            # Allocate/fence before a remote claim; advance no request cursor
            # until every native writer has covered and drained its range.
            torch.cuda.synchronize()
            blocks = position // manager.tokens_per_block
            groups, kinds = [], []
            for idx, group in enumerate(self.transceiver._page_table.layer_groups):
                values = np.array(
                    self.transceiver._reuse_adapter.get_block_ordinals(request, idx, group)[
                        :blocks
                    ],
                    dtype=np.int64,
                    copy=True,
                )
                if len(values) != blocks or np.any(values < 0):
                    raise RuntimeError("retained destination lacks complete page groups")
                values.flags.writeable = False
                groups.append(values)
                kinds.append(group.kind)
            chunk = Chunk(
                block_ids_per_layer_groups=groups,
                kind_per_layer_group=kinds,
                token_range=TokenRange(start=0, end=position),
                is_last=True,
            )
            paged, state = _chunk_bytes_by_kind(self.transceiver._page_table, chunk)
            snapshot = snapshot_cache(
                self.transceiver,
                request,
                chunk,
                engine_generation=self.generation,
                model_identity=self.config.model_identity,
            )
            layout = logical_layout_identity(snapshot)
            if self.layout is not None and self.layout != layout:
                raise ValueError("retained logical cache layout changed")
            self.layout = layout
            identity = self._identity(request, position)
            if identity != capability.identity:
                raise ValueError("retained model/token/cache-salt/layout identity mismatch")
        except BaseException as failure:
            error = str(failure)
        self._agree(error)
        nonce = self._broadcast(uuid.uuid4().hex if self.rank == 0 else None)
        prepared_import = {
            "protocol": VERSION,
            "capability": asdict(capability),
            "identity": identity,
            "nonce": nonce,
            "snapshots": self._gather(snapshot.wire()),
        }
        if len(json.dumps(prepared_import).encode()) > 16 * 1024 * 1024:
            raise ValueError("retained prepared destination exceeds bounded capacity")
        with self.lock:
            self.pending_imports[nonce] = prepared_import
        message = {
            "action": "claim",
            "protocol": VERSION,
            "capability": asdict(capability),
            "identity": identity,
            "nonce": nonce,
            "destination_peer": self.config.peer_id,
        }
        try:
            reply = self._call_source(endpoint, message)
        except BaseException:
            # No session/publication exists until the separate admitted step.
            with self.lock:
                del self.pending_imports[nonce]
            raise
        # The route in this receipt comes from the configured trusted source,
        # never from public caller coordinates.
        error = None
        session = None
        self.quarantined.append((request, cache, chunk, None))
        try:
            claim = _bind_provider_claim(**reply["claim"])
            if (
                claim.donor_generation != capability.engine_generation
                or claim.token_identity != identity
                or claim.prefix_tokens != position
                or claim.model_identity != self.config.model_identity
                or claim.layout_identity != self.layout
            ):
                raise ValueError("trusted source returned an incompatible claim")
            sources = tuple(snapshot_from_trusted_wire(value) for value in reply["sources"])
            destinations = tuple(
                snapshot_from_trusted_wire(value) for value in prepared_import["snapshots"]
            )
            plans, certificates = self._coverage(
                claim.disagg_request_id, nonce, sources, destinations
            )
            if [[plan.receipt() for plan in cohort] for cohort in plans] != reply[
                "plans"
            ] or certificates != reply["certificates"]:
                raise ValueError(
                    "retained source preview differs from independent destination coverage"
                )
            admission = make_retained_admission(
                self.worker,
                self.verified_identity,
                direction="receive",
                local_snapshot=snapshot,
                sources=sources,
                destinations=destinations,
                plans=plans,
                certificates=certificates,
            )
            session = create_retained_rx_session(
                self.worker, claim, local_request_id=request.py_request_id, admission=admission
            )
            self.quarantined[-1] = (request, cache, chunk, session)
        except BaseException as failure:
            error = str(failure)
        errors = self._gather(error)
        if any(errors):
            self.executor._kv_connector_fail(
                f"retained destination session admission failed: {errors}"
            )
        message.pop("destination_peer")
        message.update(action="admit", certificates=certificates)
        try:
            admitted = self._call_source(endpoint, message)
        except BaseException as failure:
            # The source may have admitted despite a lost response. Keep the
            # actual allocated destination/session quarantined and fail-stop.
            self.executor._kv_connector_fail(
                f"retained source admission outcome is uncertain: {failure}"
            )
        if admitted.get("claim") != reply["claim"]:
            self.executor._kv_connector_fail(
                "retained native source admission changed its prepared claim"
            )
        error = None
        try:
            session.receive(chunk, expected_write_bytes=paged + state)
        except BaseException as failure:
            # Unexpected submission errors still need the same all-rank fatal
            # decision. This catch retains ownership; it never permits retry.
            error = f"{type(failure).__name__}: {failure}"
            self.executor._kv_connector_failed = True
        errors = self._gather(error)
        if any(errors):
            self.executor._kv_connector_fail(f"retained destination publication failed: {errors}")
        error = None
        try:
            outcome = session.wait_complete(blocking=True)
            if (
                outcome != WaitResult.COMPLETED
                or not session.kv_write_verified()
                or not session.resources_drained()
                or not session.close()
            ):
                raise RuntimeError("retained import did not prove complete native quiescence")
        except BaseException as failure:
            error = f"{type(failure).__name__}: {failure}"
            self.executor._kv_connector_failed = True
        errors = self._gather(error)
        if any(errors):
            self.executor._kv_connector_fail(
                f"retained all-rank destination completion failed: {errors}"
            )
        self.quarantined.pop()
        with self.lock:
            del self.pending_imports[nonce]
        message.pop("certificates")
        message["action"] = "finish"
        retired = self._call_source(endpoint, message)
        if (
            retired.get("retired") is not True
            or retired.get("expired_used") is not False
            or retired.get("engine_generation") != capability.engine_generation
        ):
            raise ValueError("retained donor expired before explicit verified completion")
        manager._record_connector_reuse(cache, 0, position)
        manager._mark_connector_prefix_populated(request, position, chunk_size)
        request.py_retained_loaded = True
        request.py_retained_imported_tokens = position
        request.py_retained_import_receipt = _RetainedImportReceipt(
            capability.export_id,
            claim.disagg_request_id,
            capability.engine_generation,
            self.generation,
            position,
        )
        logger.info(
            "retained_prefix "
            + json.dumps(
                {
                    "event": "import",
                    "export_id": capability.export_id,
                    "prefix_tokens": position,
                    "native_bytes": paged + state,
                    "rank": self.rank,
                    "request_id": request.py_request_id,
                    "native_claim_id": claim.disagg_request_id,
                    "source_generation": capability.engine_generation,
                    "destination_generation": self.generation,
                    "kv_write_verified": True,
                    "physical_resources_drained": True,
                    "native_session_closed": True,
                    "groups": len(chunk.block_ids_per_layer_groups),
                    "local_prefix_tokens": local_hits[0],
                    "pending_suffix_tokens": request.prompt_len - position,
                    "destination_clock_ns": time.monotonic_ns(),
                }
            )
        )
        return True

    def shutdown(self) -> None:
        if self.pending:
            raise RuntimeError("retained native ownership remains during shutdown")
        self.stopping.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                raise RuntimeError("retained control service did not close")

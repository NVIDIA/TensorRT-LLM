# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-use encoder handoffs using the native TRT-LLM NIXL transfer agent."""

import base64
import hashlib
import json
import math
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass

import grpc
import torch

from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.disaggregated_params import DisaggregatedParams

from .bindings import encoder_transfer_pb2, encoder_transfer_pb2_grpc
from .bindings import generation_pb2 as pb
from .encoder import _release, take_encoder_exports
from .errors import ENCODER_TRANSFER_REJECTED_METADATA, EncoderTransferRejected
from .preparation import _config_digest, _input_digest

BACKEND = "trtllm-nixl-encoder-v1"
_MAX_REFERENCE_BYTES = 64 * 1024
_MAX_TENSOR_BYTES = 64 * 1024 * 1024
_MAX_CACHE_BYTES = 256 * 1024 * 1024
_MAX_TENSORS = 64
_TTL_SECONDS = 300
_TIMEOUT_SECONDS = 30
_DTYPES = {
    str(dtype): dtype
    for dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.int64,
        torch.uint8,
    )
}
# An uncertain remote WRITE may still touch these buffers. Admission is stopped;
# retain registrations and tensors until process exit rather than recycle memory.
_QUARANTINED = []


def _encode(value):
    result = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    if len(result) > _MAX_REFERENCE_BYTES:
        raise ValueError("encoder transfer metadata exceeds the size limit")
    return result


def _decode(value):
    if not value or len(value) > _MAX_REFERENCE_BYTES:
        raise ValueError("invalid encoder transfer metadata size")
    try:
        result = json.loads(value)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("invalid encoder transfer metadata") from error
    if not isinstance(result, dict):
        raise ValueError("encoder transfer metadata must be an object")
    return result


def _native_agent(name):
    from tensorrt_llm._torch.disaggregation.nixl.agent import NixlTransferAgent

    return NixlTransferAgent(name)


def _descriptors(tensors, name):
    from tensorrt_llm._torch.disaggregation.base.agent import (
        MemoryDescs,
        MemoryType,
        RegMemoryDescs,
    )

    cuda = tensors[0].is_cuda
    if any(tensor.is_cuda != cuda for tensor in tensors):
        raise ValueError("encoder transfer tensors must share a memory type")
    kind = MemoryType.VRAM if cuda else MemoryType.DRAM
    descs = [(t.data_ptr(), t.numel() * t.element_size(), t.device.index or 0) for t in tensors]
    reg = RegMemoryDescs("VRAM" if cuda else "DRAM", [(*d, name) for d in descs])
    return reg, MemoryDescs(kind, descs), descs


@dataclass
class _Export:
    tensors: list
    metadata: dict
    created: float
    size: int
    discard_requested: bool = False


class NixlEncoderCache:
    backend = BACKEND

    def __init__(self, endpoint, model_revision, *, device="cuda:0", agent_factory=_native_agent):
        if (
            not isinstance(model_revision, str)
            or len(model_revision) not in (40, 64)
            or any(c not in "0123456789abcdef" for c in model_revision)
        ):
            raise ValueError("NIXL encoder transfer requires a pinned model revision")
        if endpoint and ("://" in endpoint or "/" in endpoint):
            raise ValueError("encoder endpoint must be a reachable host:port")
        self.endpoint = endpoint
        self.model_revision = model_revision
        self.device = torch.device(device)
        self.name = f"openengine-encoder-{uuid.uuid4().hex}"
        self._agent_factory = agent_factory
        # Fail startup before advertising a backend that cannot initialize.
        probe = agent_factory(f"{self.name}-probe")
        probe.shutdown()
        self._lock = threading.Condition()
        self._active = 0
        self._exports = {}
        self._claims = {}
        self._used = 0
        self.failed = False
        self._closed = False
        self._closing = False
        self._close_error = None

    @contextmanager
    def _operation(self, *, unavailable_error=RuntimeError):
        with self._lock:
            self._healthy(unavailable_error=unavailable_error)
            self._active += 1
        try:
            yield
        finally:
            with self._lock:
                self._active -= 1
                self._lock.notify_all()

    def _quarantine(self, *resources):
        with self._lock:
            self.failed = True
            _QUARANTINED.append(resources)

    def _model_digest(self, llm):
        # The immutable revision is deployment-provided, independent of host paths.
        return hashlib.sha256(_encode([_config_digest(llm), self.model_revision])).hexdigest()

    def _discard(self, item):
        self._used -= item.size

    def _collect(self):
        for identity, item in list(self._exports.items()):
            if time.monotonic() - item.created > _TTL_SECONDS:
                self._discard(item)
                del self._exports[identity]

    def _healthy(self, *, unavailable_error=RuntimeError):
        if self.failed or self._closed or self._closing:
            raise unavailable_error("encoder NIXL transport unavailable; restart the engine")

    def publish(self, result, request, llm):
        with self._operation():
            if not self.endpoint:
                raise ValueError("encoder producer requires an advertised endpoint")
            handles, mrope = take_encoder_exports(result.disaggregated_params)
            pending = [*handles, *(h for h in mrope if h is not None)]
            tensors = []
            reserved = False
            try:
                if not handles or (mrope[0] is None) != (mrope[1] is None):
                    raise ValueError("incomplete encoder output")
                if len(pending) > _MAX_TENSORS:
                    raise ValueError("too many encoder output tensors")
                views = []
                total = 0
                for handle in list(pending):
                    tensor = SharedTensorContainer.from_dict(handle).get_local_view()
                    pending.pop(0)
                    total += tensor.numel() * tensor.element_size()
                    if total > _MAX_TENSOR_BYTES:
                        raise ValueError("encoder output exceeds the tensor limit")
                    views.append(tensor)
                with self._lock:
                    self._healthy()
                    self._collect()
                    if self._used + total > _MAX_CACHE_BYTES:
                        raise RuntimeError("encoder NIXL cache is full")
                    self._used += total
                    reserved = True
                tensors = [view.detach().to(self.device).contiguous().clone() for view in views]
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                metadata = {
                    "input_digest": _input_digest(request),
                    "model_digest": self._model_digest(llm),
                    "hashes": result.disaggregated_params.multimodal_hashes,
                    "embedding_metadata": [h.get("metadata") for h in handles],
                    "has_mrope": mrope[0] is not None,
                    "tensors": [{"shape": list(t.shape), "dtype": str(t.dtype)} for t in tensors],
                }
                identity = uuid.uuid4().hex
                state = _encode(
                    {"version": 1, "endpoint": self.endpoint, "id": identity, "metadata": metadata}
                )
                reference = pb.EncoderHandoffRef(transfer_backend=BACKEND, opaque_state=state)
                with self._lock:
                    self._healthy()
                    self._exports[identity] = _Export(tensors, metadata, time.monotonic(), total)
                return reference
            except Exception:
                if reserved:
                    with self._lock:
                        self._used -= total
                raise
            finally:
                _release(pending)

    @staticmethod
    def _reference(reference):
        if reference.transfer_backend != BACKEND:
            raise ValueError("unsupported encoder transfer backend")
        value = _decode(reference.opaque_state)
        identity = value.get("id")
        if (
            type(value.get("version")) is not int
            or value["version"] != 1
            or not isinstance(identity, str)
            or len(identity) != 32
            or any(c not in "0123456789abcdef" for c in identity)
        ):
            raise ValueError("invalid encoder transfer reference")
        endpoint = value.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint or "://" in endpoint or "/" in endpoint:
            raise ValueError("invalid encoder endpoint")
        return value

    def discard(self, reference):
        value = self._reference(reference)
        with self._lock:
            if value["endpoint"] != self.endpoint:
                raise ValueError("encoder handoff belongs to another producer")
            identity = value["id"]
            claimed = identity in self._claims
            item = self._claims.get(identity) if claimed else self._exports.get(identity)
            if item is None:
                return False
            if value.get("metadata") != item.metadata:
                raise ValueError("encoder handoff metadata mismatch")
            if claimed:
                item.discard_requested = True
                return False
            del self._exports[identity]
            self._discard(item)
            return True

    def _claim(self, reference, destination):
        """Reserve a single-use export before importing any receiver metadata."""
        try:
            value = self._reference(reference)
            peer = _decode(destination)
            with self._lock:
                self._healthy(unavailable_error=EncoderTransferRejected)
                self._collect()
                item = self._exports.get(value["id"])
                if item is None:
                    raise ValueError("encoder handoff expired or was already consumed")
                if value.get("metadata") != item.metadata or value["endpoint"] != self.endpoint:
                    raise ValueError("encoder handoff metadata mismatch")
                descs = peer.get("descs")
                if not isinstance(descs, list) or len(descs) != len(item.tensors):
                    raise ValueError("invalid encoder receive descriptors")
                for desc, tensor in zip(descs, item.tensors):
                    if (
                        not isinstance(desc, list)
                        or len(desc) != 3
                        or any(type(n) is not int for n in desc)
                        or desc[0] <= 0
                        or desc[1] != tensor.numel() * tensor.element_size()
                        or desc[2] < 0
                    ):
                        raise ValueError("encoder receive buffer size mismatch")
                if (
                    peer.get("memory_type") not in ("VRAM", "DRAM")
                    or not isinstance(peer.get("name"), str)
                    or not peer["name"]
                ):
                    raise ValueError("invalid encoder receiver")
                try:
                    agent_desc = base64.b64decode(peer["agent"], validate=True)
                except (KeyError, ValueError, TypeError) as error:
                    raise ValueError("invalid encoder receiver agent") from error
                # Claimed exports stay charged to quota and cannot expire during preflight.
                del self._exports[value["id"]]
                self._claims[value["id"]] = item
            return value["id"], item, peer, agent_desc
        except ValueError as error:
            raise EncoderTransferRejected(str(error)) from error

    def transfer(self, reference, destination):
        """Write once with request-owned agent, registration, and completion status."""
        from tensorrt_llm._torch.disaggregation.base.agent import (
            MemoryDescs,
            MemoryType,
            TransferOp,
            TransferRequest,
        )

        with self._operation(unavailable_error=EncoderTransferRejected):
            identity, item, peer, agent_desc = self._claim(reference, destination)
            agent = None
            status = None
            registered = False
            submitted = False
            try:
                name = f"{self.name}-send-{uuid.uuid4().hex}"
                agent = self._agent_factory(name)
                registration, source, _ = _descriptors(item.tensors, name)
                agent.register_memory(registration)
                registered = True
                agent.load_remote_agent(peer["name"], agent_desc)
                target = MemoryDescs(
                    getattr(MemoryType, peer["memory_type"]), [tuple(d) for d in peer["descs"]]
                )
                if not agent.check_remote_descs(peer["name"], target):
                    raise ValueError("encoder receive buffers are not registered")
                transfer = TransferRequest(TransferOp.WRITE, source, target, peer["name"])
                # Submission can post a WRITE and then raise before returning status.
                submitted = True
                status = agent.submit_transfer_requests(transfer)
                if not status.wait(_TIMEOUT_SECONDS * 1000):
                    raise RuntimeError("encoder NIXL write did not complete")
                status = None
                agent.invalidate_remote_agent(peer["name"])
                agent.deregister_memory(registration)
                registered = False
                agent.shutdown()
                with self._lock:
                    self._claims.pop(identity)
                    self._discard(item)
            except Exception as error:
                if submitted:
                    self._quarantine(agent, item, status)
                    raise
                try:
                    if registered:
                        agent.deregister_memory(registration)
                    if agent is not None:
                        # Retire all imported peer names, including a partial failed import.
                        agent.shutdown()
                except Exception:
                    self._quarantine(agent, item)
                    raise
                with self._lock:
                    self._claims.pop(identity)
                    if item.discard_requested:
                        self._discard(item)
                    else:
                        self._exports[identity] = item
                raise EncoderTransferRejected(f"invalid encoder receiver: {error}") from error
            finally:
                with self._lock:
                    self._claims.pop(identity, None)

    def _receive(self, reference, value):
        metadata = value.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("missing encoder tensor metadata")
        specs = metadata.get("tensors")
        if not isinstance(specs, list) or not 0 < len(specs) <= _MAX_TENSORS:
            raise ValueError("invalid encoder tensor count")
        shapes = []
        total = 0
        for spec in specs:
            if not isinstance(spec, dict):
                raise ValueError("invalid encoder tensor specification")
            shape, dtype = spec.get("shape"), _DTYPES.get(spec.get("dtype"))
            if (
                dtype is None
                or not isinstance(shape, list)
                or not 0 < len(shape) <= 8
                or any(type(n) is not int or n <= 0 for n in shape)
            ):
                raise ValueError("invalid encoder tensor shape or dtype")
            total += math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            shapes.append((shape, dtype))
        with self._lock:
            self._healthy()
            if total > _MAX_TENSOR_BYTES or self._used + total > _MAX_CACHE_BYTES:
                raise ValueError("encoder receive exceeds the memory limit")
            self._used += total
        tensors = []
        agent = None
        registration = None
        registration_attempted = False
        registered = False
        attempted = False
        uncertain = False
        try:
            tensors = [
                torch.empty(shape, dtype=dtype, device=self.device) for shape, dtype in shapes
            ]
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            # NIXL metadata includes all registrations; isolate overlapping receives.
            name = f"{self.name}-receive-{uuid.uuid4().hex}"
            agent = self._agent_factory(name)
            registration, _, descs = _descriptors(tensors, name)
            registration_attempted = True
            agent.register_memory(registration)
            registered = True
            destination = _encode(
                {
                    "name": name,
                    "agent": base64.b64encode(agent.get_local_agent_desc()).decode(),
                    "memory_type": "VRAM" if self.device.type == "cuda" else "DRAM",
                    "descs": descs,
                }
            )
            with grpc.insecure_channel(value["endpoint"]) as channel:
                attempted = True
                encoder_transfer_pb2_grpc.EncoderTransferStub(channel).Transfer(
                    encoder_transfer_pb2.TransferEncoderCacheRequest(
                        cache=reference, destination=destination
                    ),
                    timeout=_TIMEOUT_SECONDS + 10,
                )
        except Exception as error:
            rejected = (
                isinstance(error, grpc.RpcError)
                and error.code() == grpc.StatusCode.INVALID_ARGUMENT
                and ENCODER_TRANSFER_REJECTED_METADATA in (error.trailing_metadata() or ())
            )
            uncertain = (attempted and not rejected) or (registration_attempted and not registered)
            if rejected:
                raise EncoderTransferRejected(error.details()) from error
            raise
        finally:
            if uncertain:
                self._quarantine(agent, tensors, registration)
            else:
                try:
                    if registered:
                        agent.deregister_memory(registration)
                    if agent is not None:
                        agent.shutdown()
                except Exception:
                    self._quarantine(agent, tensors, registration)
                    raise
                else:
                    with self._lock:
                        self._used -= total
        return tensors

    def prepare(self, reference, request, llm, sampling):
        value = self._reference(reference)
        metadata = value.get("metadata", {})
        if metadata.get("input_digest") != _input_digest(request):
            raise ValueError("encoder handoff does not match the request inputs")
        if metadata.get("model_digest") != self._model_digest(llm):
            raise ValueError("encoder handoff belongs to another model revision or configuration")
        embedding_metadata = metadata.get("embedding_metadata")
        count = len(embedding_metadata) if isinstance(embedding_metadata, list) else 0
        if not count or len(metadata.get("tensors", [])) != count + (
            2 if metadata.get("has_mrope") else 0
        ):
            raise ValueError("invalid encoder embedding layout")
        # Reject lossy decoding before consuming the single-use handoff.
        if request.WhichOneof("input") == "prompt":
            prompt = request.prompt
        else:
            ids = list(request.token_ids.ids)
            prompt = llm.tokenizer.decode(ids, skip_special_tokens=False)
            if llm.tokenizer.encode(prompt, add_special_tokens=False) != ids:
                raise ValueError("encoder handoff requires lossless prompt token decoding")
        with self._operation():
            tensors = self._receive(reference, value)
        exports = []
        try:
            for tensor in tensors:
                exports.append(SharedTensorContainer.from_tensor(tensor).dump_to_dict())
            for handle, layout in zip(exports[:count], embedding_metadata):
                if layout is not None:
                    handle["metadata"] = layout
            positions = {}
            if metadata.get("has_mrope"):
                positions = {
                    "mrope_position_ids_handle": exports[-2],
                    "mrope_position_deltas_handle": exports[-1],
                }
            params = DisaggregatedParams(
                request_type="context_and_generation",
                multimodal_embedding_handles=exports[:count],
                multimodal_hashes=metadata["hashes"],
                **positions,
            )
            inputs = llm.preprocess({"prompt": prompt}, sampling, params)
            exports.clear()
            return inputs, params
        finally:
            _release(exports)

    def close(self):
        with self._lock:
            if self._closing:
                self._lock.wait_for(lambda: self._closed)
            if self._closed:
                if self._close_error is not None:
                    raise RuntimeError("encoder NIXL shutdown failed") from self._close_error
                return
            self._closing = True
            try:
                if not self._lock.wait_for(lambda: not self._active, _TIMEOUT_SECONDS + 10):
                    self.failed = True
                    raise RuntimeError("encoder NIXL operations did not drain during shutdown")
                if self.failed:
                    return
                for item in self._exports.values():
                    self._discard(item)
                self._exports.clear()
            except Exception as error:
                self.failed = True
                self._close_error = error
                raise
            finally:
                if self.failed:
                    _QUARANTINED.append(self)
                self._closed = True
                self._lock.notify_all()

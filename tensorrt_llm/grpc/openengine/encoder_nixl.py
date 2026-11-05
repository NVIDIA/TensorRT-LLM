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
from dataclasses import dataclass
from typing import Any

import grpc
import torch

from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.disaggregated_params import DisaggregatedParams

from .bindings import generation_pb2 as pb
from .bindings import openengine_pb2_grpc
from .encoder import (
    _capture_encoder_decode_state,
    _encoder_input_params,
    _release,
    take_encoder_exports,
)
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
    except (ValueError, UnicodeError) as error:
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
    registration: Any
    descriptors: Any
    metadata: dict
    created: float
    size: int


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
        self.agent = agent_factory(self.name)
        self._lock = threading.RLock()
        self._exports = {}
        self._used = 0
        self.failed = False
        self._closed = False

    def _model_digest(self, llm):
        # The immutable revision is deployment-provided, independent of host paths.
        return hashlib.sha256(_encode([_config_digest(llm), self.model_revision])).hexdigest()

    def _discard(self, item):
        try:
            self.agent.deregister_memory(item.registration)
        except Exception:
            self.failed = True
            _QUARANTINED.append((self.agent, item))
            raise
        self._used -= item.size

    def _collect(self):
        for identity, item in list(self._exports.items()):
            if time.monotonic() - item.created > _TTL_SECONDS:
                self._discard(item)
                del self._exports[identity]

    def _healthy(self):
        if self.failed or self._closed:
            raise RuntimeError("encoder NIXL transport unavailable; restart the engine")

    def publish(self, result, request, llm):
        if not self.endpoint:
            raise ValueError("encoder producer requires an advertised endpoint")
        handles, mrope = take_encoder_exports(result.disaggregated_params)
        pending = [*handles, *(h for h in mrope if h is not None)]
        tensors = []
        try:
            if not handles or (mrope[0] is None) != (mrope[1] is None):
                raise ValueError("incomplete encoder output")
            if len(pending) > _MAX_TENSORS:
                raise ValueError("too many encoder output tensors")
            total = 0
            for handle in list(pending):
                tensor = SharedTensorContainer.from_dict(handle).get_local_view()
                pending.pop(0)
                total += tensor.numel() * tensor.element_size()
                if total > _MAX_TENSOR_BYTES:
                    raise ValueError("encoder output exceeds the tensor limit")
                tensors.append(tensor.detach().to(self.device).contiguous().clone())
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        finally:
            _release(pending)
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
        registration, descriptors, _ = _descriptors(tensors, self.name)
        with self._lock:
            self._healthy()
            self._collect()
            if self._used + total > _MAX_CACHE_BYTES:
                raise RuntimeError("encoder NIXL cache is full")
            try:
                self.agent.register_memory(registration)
            except Exception:
                self.failed = True
                _QUARANTINED.append((self.agent, tensors, registration))
                raise
            self._exports[identity] = _Export(
                tensors, registration, descriptors, metadata, time.monotonic(), total
            )
            self._used += total
        return pb.EncoderCacheRef(transfer_backend=BACKEND, opaque_state=state)

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

    def transfer(self, reference, destination):
        """Write an export once; an uncertain transfer poisons the transport."""
        from tensorrt_llm._torch.disaggregation.base.agent import (
            MemoryDescs,
            MemoryType,
            TransferOp,
            TransferRequest,
        )

        value = self._reference(reference)
        peer = _decode(destination)
        with self._lock:
            self._healthy()
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
            kind = peer.get("memory_type")
            if kind not in ("VRAM", "DRAM") or not isinstance(peer.get("name"), str):
                raise ValueError("invalid encoder receiver")
            try:
                agent_desc = base64.b64decode(peer["agent"], validate=True)
            except (KeyError, ValueError, TypeError) as error:
                raise ValueError("invalid encoder receiver agent") from error
            del self._exports[value["id"]]
            # No producer pointers leave this process: an expired/replayed
            # reference cannot issue a READ into recycled memory.
            try:
                self.agent.load_remote_agent(peer["name"], agent_desc)
                target = MemoryDescs(getattr(MemoryType, kind), [tuple(d) for d in descs])
                transfer = TransferRequest(TransferOp.WRITE, item.descriptors, target, peer["name"])
                status = self.agent.submit_transfer_requests(transfer)
                if not status.wait(_TIMEOUT_SECONDS * 1000):
                    raise RuntimeError("encoder NIXL write did not complete")
                self.agent.invalidate_remote_agent(peer["name"])
                self._discard(item)
            except Exception:
                self.failed = True
                _QUARANTINED.append((self.agent, item))
                raise

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
        if total > _MAX_TENSOR_BYTES or self._used + total > _MAX_CACHE_BYTES:
            raise ValueError("encoder receive exceeds the memory limit")
        tensors = [torch.empty(shape, dtype=dtype, device=self.device) for shape, dtype in shapes]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        registration, _, descs = _descriptors(tensors, self.name)
        attempted = False
        registered = False
        self._used += total
        try:
            self.agent.register_memory(registration)
            registered = True
            # Native agent metadata must include the just-registered buffers.
            destination = _encode(
                {
                    "name": self.name,
                    "agent": base64.b64encode(self.agent.get_local_agent_desc()).decode(),
                    "memory_type": "VRAM" if self.device.type == "cuda" else "DRAM",
                    "descs": descs,
                }
            )
            with grpc.insecure_channel(value["endpoint"]) as channel:
                attempted = True
                openengine_pb2_grpc.ControlStub(channel).TransferEncoderCache(
                    pb.TransferEncoderCacheRequest(cache=reference, destination=destination),
                    timeout=_TIMEOUT_SECONDS + 10,
                )
            self.agent.deregister_memory(registration)
            registered = False
            self._used -= total
            return tensors
        except Exception:
            if attempted or not registered:
                # A lost RPC reply does not prove the remote WRITE stopped.
                self.failed = True
                _QUARANTINED.append((self.agent, tensors, registration))
            else:
                try:
                    self.agent.deregister_memory(registration)
                    self._used -= total
                except Exception:
                    self.failed = True
                    _QUARANTINED.append((self.agent, tensors, registration))
            raise

    def prepare(
        self,
        reference,
        request,
        llm,
        sampling,
        disaggregated_params: DisaggregatedParams | None = None,
    ):
        params = _encoder_input_params(disaggregated_params)
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
        with self._lock:
            self._healthy()
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
            params.multimodal_embedding_handles = exports[:count]
            params.multimodal_hashes = metadata["hashes"]
            params.mrope_position_ids_handle = positions.get("mrope_position_ids_handle")
            params.mrope_position_deltas_handle = positions.get("mrope_position_deltas_handle")
            inputs = llm.preprocess({"prompt": prompt}, sampling, params)
            exports.clear()
            if params.request_type == "context_only":
                _capture_encoder_decode_state(
                    llm,
                    inputs,
                    {
                        "mrope_position_ids": tensors[-2],
                        "mrope_position_deltas": tensors[-1],
                    }
                    if metadata.get("has_mrope")
                    else {},
                )
            return inputs, params
        finally:
            _release(exports)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self.failed:
                _QUARANTINED.append(self)
                return
            for item in self._exports.values():
                self._discard(item)
            self._exports.clear()
            self.agent.shutdown()

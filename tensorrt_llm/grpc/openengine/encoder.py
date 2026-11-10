# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-use, same-host encoder handoffs through a bounded shared directory."""

import fcntl
import hashlib
import json
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file

from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.logger import logger

from .bindings import generation_pb2
from .errors import UnsupportedFeatureError
from .preparation import _config_digest, _input_digest

BACKEND = "trtllm-shared-safetensors-v1"
_MAX_TENSOR_BYTES = 64 * 1024 * 1024
_MAX_CACHE_BYTES = 256 * 1024 * 1024
_MAX_METADATA_BYTES = 64 * 1024
_TTL_SECONDS = 300
_EXPORT_LOCK = threading.Lock()


def _model_digest(llm: Any) -> str:
    checkpoint = getattr(llm, "_hf_model_dir", None)
    identity = [_config_digest(llm), str(Path(checkpoint).resolve()) if checkpoint else None]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def _release(handles: list[dict]) -> bool:
    succeeded = True
    for handle in handles:
        try:
            SharedTensorContainer.release_dict(handle)
        except Exception as error:
            succeeded = False
            logger.error(f"Failed to release abandoned encoder export: {error}")
    return succeeded


def take_encoder_exports(params: Any) -> tuple[list[dict], list[dict | None]]:
    with _EXPORT_LOCK:
        embeddings = params.multimodal_embedding_handles or []
        mrope = [params.mrope_position_ids_handle, params.mrope_position_deltas_handle]
        params.multimodal_embedding_handles = None
        params.mrope_position_ids_handle = params.mrope_position_deltas_handle = None
    return embeddings, mrope


def release_encoder_result(result: Any) -> None:
    params = getattr(result, "disaggregated_params", None)
    if params is not None:
        embeddings, mrope = take_encoder_exports(params)
        _release([*embeddings, *(handle for handle in mrope if handle is not None)])


class PrefillResultExports:
    """Reclaim native mRoPE exports omitted from portable prefill responses."""

    def __init__(self, params: DisaggregatedParams) -> None:
        # Request handles were consumed by the language worker, not abandoned.
        self._seen = {
            json.dumps(handle, sort_keys=True)
            for handle in (params.mrope_position_ids_handle, params.mrope_position_deltas_handle)
            if handle is not None
        }

    @staticmethod
    def has_exports(result: Any) -> bool:
        params = getattr(result, "disaggregated_params", None)
        return params is not None and (
            params.mrope_position_ids_handle is not None
            or params.mrope_position_deltas_handle is not None
        )

    def release(self, result: Any) -> bool:
        params = getattr(result, "disaggregated_params", None)
        if params is None:
            return True
        pending = []
        with _EXPORT_LOCK:
            for handle in (params.mrope_position_ids_handle, params.mrope_position_deltas_handle):
                if handle is None:
                    continue
                key = json.dumps(handle, sort_keys=True)
                if key not in self._seen:
                    self._seen.add(key)
                    pending.append(handle)
        return _release(pending)


def _encoder_input_params(params: DisaggregatedParams | None) -> DisaggregatedParams:
    if params is None:
        return DisaggregatedParams(request_type="context_and_generation")
    if params.request_type not in ("context_only", "context_and_generation"):
        raise UnsupportedFeatureError("encoder handoffs require a prefill or aggregated request")
    return params


def _encoder_output_params(
    params: DisaggregatedParams,
    handles: list[dict],
    hashes: list[list[int]] | None,
    mrope: dict[str, dict],
) -> DisaggregatedParams:
    """Apply native hash validation and the missing-hash fallback to a handoff."""
    try:
        return replace(
            params,
            multimodal_embedding_handles=handles,
            multimodal_hashes=hashes,
            mrope_position_ids_handle=mrope.get("mrope_position_ids_handle"),
            mrope_position_deltas_handle=mrope.get("mrope_position_deltas_handle"),
        )
    except (AssertionError, TypeError) as error:
        raise ValueError(f"invalid encoder handoff parameters: {error}") from error


def _capture_encoder_decode_state(llm: Any, inputs: Any, mrope: dict) -> None:
    capture = getattr(llm.input_processor, "_capture_prepared_decode", None)
    if not callable(capture) or not mrope:
        return
    try:
        # Native embedding preparation bypasses the processor's __call__ wrapper.
        capture((inputs.prompt_token_ids, {"multimodal_data": {"mrope_config": mrope}}))
    except Exception:
        params = getattr(inputs, "multimodal_params", None)
        if params is not None:
            params.release_handles()
        raise


class EncoderCache:
    backend = BACKEND

    def __init__(self, directory: str) -> None:
        self.root = Path(directory).resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.stat().st_mode & 0o022:
            raise ValueError("encoder cache directory must not be group- or world-writable")
        self.host = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        with self._lock():
            namespace = self.root / ".namespace"
            if not namespace.exists():
                namespace.write_text(uuid.uuid4().hex)
            self.namespace = namespace.read_text().strip()
            self._collect()

    @contextmanager
    def _lock(self):
        with (self.root / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _collect(self) -> int:
        used = 0
        for path in self.root.iterdir():
            if path.suffix not in (".safetensors", ".tmp"):
                continue
            try:
                stat = path.stat()
                if time.time() - stat.st_mtime > _TTL_SECONDS:
                    path.unlink(missing_ok=True)
                else:
                    used += stat.st_size
            except FileNotFoundError:
                continue
        return used

    def publish(self, result: Any, request: Any, llm: Any):
        params = result.disaggregated_params
        if params is None or not params.multimodal_embedding_handles:
            raise RuntimeError("encoder did not return native embedding handles")
        embeddings, mrope = take_encoder_exports(params)
        if not embeddings:
            raise RuntimeError("encoder exports were already reclaimed")
        pending = [*embeddings, *(handle for handle in mrope if handle is not None)]
        tensors = {}
        try:
            if (mrope[0] is None) != (mrope[1] is None):
                raise RuntimeError("encoder returned an incomplete mRoPE pair")
            names = [f"embedding_{i}" for i in range(len(embeddings))]
            names += [
                name
                for name, handle in zip(("mrope_ids", "mrope_deltas"), mrope)
                if handle is not None
            ]
            total = 0
            for name in names:
                handle = pending[0]
                tensor = SharedTensorContainer.from_dict(handle).get_local_view()
                pending.pop(0)
                total += tensor.numel() * tensor.element_size()
                if total > _MAX_TENSOR_BYTES:
                    raise ValueError("encoder output exceeds the shared-cache tensor limit")
                tensors[name] = tensor.detach().cpu().contiguous().clone()
        finally:
            _release(pending)
        metadata = json.dumps(
            {
                "input_digest": _input_digest(request),
                "config_digest": _model_digest(llm),
                "hashes": params.multimodal_hashes,
                "embedding_metadata": [handle.get("metadata") for handle in embeddings],
            },
            separators=(",", ":"),
            allow_nan=False,
        )
        if len(metadata.encode()) > _MAX_METADATA_BYTES:
            raise ValueError("encoder output metadata exceeds the size limit")
        identity = uuid.uuid4().hex
        path = self.root / f"{identity}.safetensors"
        temporary = path.with_suffix(".tmp")
        try:
            # CPU storage has no outstanding exported CUDA reference. An unused
            # handoff can expire even if its downstream process disappears.
            with self._lock():
                used = self._collect()
                save_file(tensors, str(temporary), metadata={"handoff": metadata})
                size = temporary.stat().st_size
                if size > _MAX_TENSOR_BYTES + _MAX_METADATA_BYTES:
                    raise ValueError("serialized encoder output exceeds the size limit")
                if used + size > _MAX_CACHE_BYTES:
                    raise RuntimeError("encoder shared cache is full")
                temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        reference = {"version": 1, "namespace": self.namespace, "host": self.host, "id": identity}
        return generation_pb2.EncoderHandoffRef(
            transfer_backend=BACKEND,
            opaque_state=json.dumps(reference, separators=(",", ":")).encode(),
        )

    def _path(self, reference: Any) -> Path:
        if reference.transfer_backend != BACKEND or len(reference.opaque_state) > 1024:
            raise ValueError("unsupported encoder handoff reference")
        try:
            value = json.loads(reference.opaque_state)
        except (ValueError, UnicodeError, RecursionError) as error:
            raise ValueError("invalid encoder handoff reference") from error
        if (
            not isinstance(value, dict)
            or type(value.get("version")) is not int
            or value["version"] != 1
        ):
            raise ValueError("unsupported encoder handoff reference version")
        if value.get("namespace") != self.namespace or value.get("host") != self.host:
            raise ValueError(
                "encoder and consumer require the same host and shared cache directory"
            )
        identity = value.get("id", "")
        if (
            not isinstance(identity, str)
            or len(identity) != 32
            or any(c not in "0123456789abcdef" for c in identity)
        ):
            raise ValueError("invalid encoder handoff id")
        return self.root / f"{identity}.safetensors"

    def discard(self, reference: Any) -> bool:
        path = self._path(reference)
        with self._lock():
            if path.is_symlink():
                raise ValueError("invalid encoder handoff file")
            try:
                path.unlink()
            except FileNotFoundError:
                return False
            return True

    def prepare(
        self,
        reference: Any,
        request: Any,
        llm: Any,
        sampling: Any,
        disaggregated_params: DisaggregatedParams | None = None,
    ):
        params = _encoder_input_params(disaggregated_params)
        path = self._path(reference)
        identity = path.stem
        exports = []
        claimed = self.root / f"{identity}-{uuid.uuid4().hex}.tmp"
        try:
            if request.WhichOneof("input") == "prompt":
                prompt = request.prompt
            else:
                ids = list(request.token_ids.ids)
                prompt = llm.tokenizer.decode(ids, skip_special_tokens=False)
                if llm.tokenizer.encode(prompt, add_special_tokens=False) != ids:
                    raise ValueError("encoder handoff requires lossless prompt token decoding")
            with self._lock():
                self._collect()
                if not path.is_file() or path.is_symlink():
                    raise ValueError("encoder handoff expired or was already consumed")
                if path.stat().st_size > _MAX_TENSOR_BYTES + _MAX_METADATA_BYTES:
                    raise ValueError("encoder handoff exceeds the size limit")
                with safe_open(str(path), framework="pt", device="cpu") as stored:
                    encoded_metadata = stored.metadata()["handoff"]
                    if len(encoded_metadata.encode()) > _MAX_METADATA_BYTES:
                        raise ValueError("encoder handoff metadata exceeds the size limit")
                    metadata = json.loads(encoded_metadata)
                    if metadata["input_digest"] != _input_digest(request):
                        raise ValueError("encoder handoff does not match the request inputs")
                    if metadata["config_digest"] != _model_digest(llm):
                        raise ValueError("encoder handoff belongs to another model configuration")
                path.rename(claimed)
                opened = safe_open(str(claimed), framework="pt", device="cpu")
            with opened as stored:
                handles = []
                for i, item_metadata in enumerate(metadata["embedding_metadata"]):
                    handle = SharedTensorContainer.from_tensor(
                        stored.get_tensor(f"embedding_{i}").clone()
                    ).dump_to_dict()
                    exports.append(handle)
                    if item_metadata is not None:
                        handle["metadata"] = item_metadata
                    handles.append(handle)
                mrope = {}
                mrope_tensors = {}
                for key, field in (
                    ("mrope_ids", "mrope_position_ids_handle"),
                    ("mrope_deltas", "mrope_position_deltas_handle"),
                ):
                    if key in stored.keys():
                        tensor = stored.get_tensor(key).clone()
                        mrope_tensors[field.removesuffix("_handle")] = tensor
                        handle = SharedTensorContainer.from_tensor(tensor).dump_to_dict()
                        exports.append(handle)
                        mrope[field] = handle
            params = _encoder_output_params(params, handles, metadata["hashes"], mrope)
            inputs = llm.preprocess({"prompt": prompt}, sampling, params)
            exports.clear()  # PreprocessedInputs now owns the worker exports.
            if params.request_type == "context_only":
                _capture_encoder_decode_state(llm, inputs, mrope_tensors)
            return inputs, params
        finally:
            _release(exports)
            claimed.unlink(missing_ok=True)


def encoder_role(llm: Any, cache: EncoderCache | None) -> str:
    if cache is None or not getattr(llm.input_processor, "support_mm_disagg", False):
        return ""
    if any(
        getattr(llm.args, name, 1) != 1
        for name in ("tensor_parallel_size", "pipeline_parallel_size", "context_parallel_size")
    ):
        return ""
    if getattr(llm.args, "mm_encoder_only", False):
        return "encode"
    if getattr(llm.args, "disable_mm_encoder", False):
        return "consume"
    return ""


def validate_encoder(request: Any, role: str) -> str:
    if not request.HasField("encoder"):
        if (
            role == "consume"
            and request.media
            and not (request.HasField("kv") and request.kv.HasField("session"))
        ):
            raise UnsupportedFeatureError("encoder consumer requires an encoder handoff for images")
        if role == "encode":
            raise UnsupportedFeatureError("encoder-only engines require encoder.encode_only")
        return ""
    mode = request.encoder.WhichOneof("mode")
    if mode is None:
        raise ValueError("set exactly one of encoder.encode_only or encoder.handoff")
    expected = "encode" if mode == "encode_only" else "consume"
    if role != expected:
        raise UnsupportedFeatureError(f"loaded engine does not support encoder role {expected}")
    if request.HasField("kv") and request.kv.HasField("session"):
        raise UnsupportedFeatureError("KV decode requests must not consume an encoder handoff")
    if expected == "encode":
        for name in ("sampling", "stopping", "guided", "response"):
            if getattr(request, name).ListFields():
                raise UnsupportedFeatureError(f"encoder-only requests do not support {name}")
        if request.lora_name:
            raise UnsupportedFeatureError("encoder-only requests do not support LoRA selection")
    if not request.media or any(
        item.modality != generation_pb2.MODALITY_IMAGE for item in request.media
    ):
        raise UnsupportedFeatureError("encoder handoffs require image inputs")
    return expected

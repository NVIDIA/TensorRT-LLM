# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request transport that maps large tensors on receiving ranks of the same host.

Requests reach every rank through hops. A hop is either a group broadcast
(all ranks, a TP group or a CP group) or a point-to-point send from one
pipeline stage to the next. On a hop whose receivers all run on the sender's
host, the sender pickles the payload with a hook that replaces each large
tensor with an index into a table of CPU shared-memory or CUDA IPC handles,
one per receiver. Receivers map every handle, then unpickle the payload with
the mapped views instead of copies of the tensor bytes.

The hook sees every tensor that pickling sees: tensors in Python containers
and dataclasses, and tensors that C++ request bindings hand to pickle through
their own state. A tensor referenced twice is sent once and maps to one view.
Payloads with stateful callables, tensor subclasses or custom tensor attributes
keep the byte transport, preserving their private state and object aliases.
Small tensors and tensors that cannot be shared are pickled inline.

Hops with a receiver on another host send bytes as before. A forwarding rank
(a CP group root, or a pipeline stage that sends on) exports the views it
mapped. CUDA views imported from another process cannot be exported again,
so the forwarder exports a local copy instead.

Lifetime: every export takes one reference. A receiver that maps a handle
releases that reference when its view is freed, through the normal request
teardown. One acknowledgement per hop that carries handles (an allgather on a
group, a reply message on a point-to-point hop) tells the sender which handles
were not mapped, and the sender releases those references itself.

Failure handling: an export or mapping error does not fail the request. The
hop is sent again as bytes, and that hop stays on bytes for the rest of the
run. A payload that cannot be pickled still raises, as it does on the byte
path.

Shared tensors are views of the same memory on every rank of a host, so
consumers must not modify request tensors in place.
"""

import io
import pickle  # nosec B403
import socket
from types import BuiltinFunctionType, FunctionType, ModuleType
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Tuple

import torch

from tensorrt_llm.logger import logger

from ..distributed.communicator import Distributed
from ..shared_tensor import SharedTensorContainer
from ..shared_tensor.shared_tensor import DTYPE_MAPPING

_SHAREABLE_DTYPES = frozenset(DTYPE_MAPPING.values())

# Smallest tensor storage sent as handles. Measured on CPU, end to end until
# every receiver holds a usable tensor, with 1, 3 and 7 receivers: bytes are
# faster at 128 KiB and handles are faster at 256 KiB for every receiver count.
_MIN_SHARED_TENSOR_BYTES = 256 * 1024

# Handles of one tensor, one per receiver of the hop.
_TensorHandles = Tuple[Dict[str, Any], ...]
# What a hop sends when sharing is enabled for it: (payload pickled with
# tensor indices, or None; the payload itself when it is sent as bytes, or
# None; the handles of each indexed tensor; the export error, if any).
_HopMessage = Tuple[Optional[bytes], Any, List[_TensorHandles], Optional[str]]


def _can_share(tensor: torch.Tensor) -> bool:
    """Whether a tensor is sent as handles; other tensors are pickled inline."""
    return (
        tensor.layout == torch.strided
        and tensor.device.type in ("cpu", "cuda")
        and tensor.dtype in _SHAREABLE_DTYPES
        and tensor.numel() > 0
        and not tensor.requires_grad
        and not tensor.is_neg()
        and not any(getattr(tensor, "names", ()))
        # Pickling a view copies its whole storage, and sharing maps it.
        and tensor.untyped_storage().nbytes() >= _MIN_SHARED_TENSOR_BYTES
    )


class _ExportError(RuntimeError):
    """A tensor could not be exported; the hop falls back to bytes."""


class _UseBytes(Exception):
    """The payload needs ordinary pickle to preserve its private object graph."""


class _Exporter(pickle.Pickler):
    """Pickler that replaces shareable tensors with indices into a handle table."""

    def __init__(self, file: io.BytesIO, receivers: int) -> None:
        super().__init__(file, protocol=pickle.HIGHEST_PROTOCOL)
        self.receivers = receivers
        self.indices: Dict[int, int] = {}
        # Every tensor seen during the dump stays referenced, so ``id`` values
        # stay unique even for temporary tensors returned by C++ bindings.
        self.tensors: List[torch.Tensor] = []
        # Local copies of imported CUDA tensors, alive until the ack.
        self.copies: List[torch.Tensor] = []
        # handles[tensor index][receiver index], and the matching exports.
        self.handles: List[_TensorHandles] = []
        self.exports: List[List[SharedTensorContainer]] = []

    def persistent_id(self, obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            if type(obj) is not torch.Tensor or obj.__dict__:
                raise _UseBytes
            if not _can_share(obj):
                return None
            index = self.indices.get(id(obj))
            if index is None:
                index = len(self.tensors)
                self.indices[id(obj)] = index
                self.tensors.append(obj)
                self.exports.append([])
                try:
                    self.handles.append(self._export(obj, self.exports[-1]))
                except Exception as exc:
                    raise _ExportError(repr(exc)) from exc
            return index
        if callable(obj) and not isinstance(obj, (type, FunctionType)):
            if not (
                isinstance(obj, BuiltinFunctionType)
                and (obj.__self__ is None or isinstance(obj.__self__, (ModuleType, type)))
            ):
                # Separate pickle streams would break aliases to callable
                # state elsewhere in the payload, including bound methods.
                raise _UseBytes
        return None

    def _export(
        self, tensor: torch.Tensor, containers: List[SharedTensorContainer]
    ) -> _TensorHandles:
        owner, handles = tensor, []
        for _ in range(self.receivers):
            try:
                container = SharedTensorContainer.from_tensor(owner)
            except RuntimeError:
                if owner is not tensor or not tensor.is_cuda:
                    raise
                # A CUDA allocation imported from another process cannot be
                # exported again, so export a local copy of its storage with
                # the same view on it.
                owner = torch.empty(0, dtype=tensor.dtype, device=tensor.device).set_(
                    tensor.untyped_storage().clone(),
                    tensor.storage_offset(),
                    tensor.size(),
                    tensor.stride(),
                )
                self.copies.append(owner)
                container = SharedTensorContainer.from_tensor(owner)
            # CUDA takes its export reference in from_tensor, CPU in dump_to_dict.
            if owner.is_cuda:
                containers.append(container)
            handle = container.dump_to_dict()
            if not owner.is_cuda:
                containers.append(container)
            handles.append(handle)
        return tuple(handles)

    def release(self, consumed: Optional[Sequence[Sequence[bool]]] = None) -> None:
        """Release the exports that no receiver mapped.

        Args:
            consumed: ``consumed[receiver][tensor]`` flags from the receivers'
                acknowledgements. None releases every export.
        """
        for index, containers in enumerate(self.exports):
            for receiver, container in enumerate(containers):
                if consumed is None or not consumed[receiver][index]:
                    container.release_unconsumed()
        self.exports.clear()


class _Importer(pickle.Unpickler):
    """Unpickler that resolves tensor indices to mapped views."""

    def __init__(self, data: bytes, views: List[torch.Tensor]) -> None:
        super().__init__(io.BytesIO(data))
        self.views = views

    def persistent_load(self, pid: Any) -> Any:
        return self.views[pid]


def _dump(payload: Any, receivers: int) -> Tuple[Optional[_Exporter], _HopMessage]:
    """Pickle ``payload`` with tensor handles; on failure, send it as bytes."""
    buffer = io.BytesIO()
    exporter = _Exporter(buffer, receivers)
    try:
        exporter.dump(payload)
    except _UseBytes:
        exporter.release()
        return None, (None, payload, [], None)
    except _ExportError as exc:
        exporter.release()
        return None, (None, payload, [], str(exc))
    except Exception:
        # The byte transport raises the same pickling error on every rank.
        exporter.release()
        return None, (None, payload, [], None)
    return exporter, (buffer.getvalue(), None, exporter.handles, None)


def _load(
    data: bytes, handles: List[_TensorHandles], receiver: int
) -> Tuple[Any, List[bool], List[str]]:
    """Map every handle addressed to ``receiver``, then unpickle the payload.

    Returns:
        The payload (None after a failure), whether each export reference
        was consumed, and the failures.
    """
    views, consumed, errors = [], [], []
    for tensor_handles in handles:
        view, opened, container = None, False, None
        try:
            container = SharedTensorContainer.from_dict(tensor_handles[receiver])
            opened = container.reference_consumed
            view = container.get_local_view()
            opened = True
        except Exception as exc:
            if container is not None:
                opened = container.reference_consumed
            errors.append(repr(exc))
        views.append(view)
        consumed.append(opened)
    if errors:
        return None, consumed, errors
    try:
        return _Importer(data, views).load(), consumed, errors
    except Exception as exc:
        if not handles:
            raise RuntimeError(f"Deserialization failed: {exc!r}") from exc
        return None, consumed, [repr(exc)]


class SameHostTensorSharing:
    """Send request payloads on hops, mapping large tensors on same-host receivers.

    Every method is a drop-in replacement for one existing transport call and
    must be called by the same ranks: ``broadcast`` for ``broadcast``,
    ``tp_broadcast`` and ``cp_broadcast``, ``send`` and ``recv`` for
    ``send_object`` and ``recv_object``. The sender gets its own payload back;
    receivers get the received payload with shared tensors mapped to views.

    Collectives: on the first use of each hop, one hostname exchange on that
    hop. Then, per call, the payload, plus one acknowledgement when the
    payload carries handles.
    """

    def __init__(self, dist: Distributed) -> None:
        self.dist = dist
        # Hop key -> whether the hop sends handles. Missing until first use.
        self.enabled: Dict[Hashable, bool] = {}

    def _disable(self, key: Hashable, reason: str) -> None:
        self.enabled[key] = False
        logger.warning(f"Same-host tensor sharing disabled for hop {key}, using bytes: {reason}")

    def _group(self, group: str) -> Tuple[Callable, Callable, List[int]]:
        dist = self.dist
        if group == "world":
            return dist.broadcast, dist.allgather, list(range(dist.world_size))
        if group == "tp":
            return dist.tp_broadcast, dist.tp_allgather, list(dist.mapping.tp_group)
        if group == "cp":
            return dist.cp_broadcast, dist.cp_allgather, list(dist.mapping.cp_group)
        raise ValueError(f"Unknown rank group: {group}")

    def broadcast(self, group: str, payload: Any, root: int = 0) -> Any:
        """Broadcast ``payload`` from the group member at index ``root``.

        Args:
            group: ``"world"``, ``"tp"`` or ``"cp"``.
            payload: The payload on the root. Ignored on other members.
            root: Index of the sender within the group.

        Raises:
            SerializationError: On the root, if the payload cannot be pickled.
            RuntimeError: On other members, in the same case.
        """
        bcast, allgather, members = self._group(group)
        index = members.index(self.dist.rank)
        sender = index == root
        key = (group, root)
        if key not in self.enabled:
            hosts = allgather(socket.gethostname())
            self.enabled[key] = all(host == hosts[root] for host in hosts)
            if sender:
                logger.info(f"Same-host tensor sharing for {group} broadcast: {self.enabled[key]}")
        if not self.enabled[key]:
            return bcast(payload, root=root)

        exporter, message = _dump(payload, len(members) - 1) if sender else (None, None)
        data, raw, handles, error = bcast(message, root=root)
        if error:
            self._disable(key, "export failed: " + error)
        if data is None:
            return payload if sender else raw

        consumed, errors, result = [], [], None
        if not sender:
            result, consumed, errors = _load(data, handles, index - (index > root))
        if not handles:
            return payload if sender else result
        acknowledgements = allgather((consumed, errors))
        if sender:
            del acknowledgements[root]
            exporter.release([flags for flags, _ in acknowledgements])
        failures = [failure for _, messages in acknowledgements for failure in messages]
        if failures:
            # Every member sees the same failures, so all take this branch.
            self._disable(key, "mapping failed: " + repr(failures))
            result = bcast(payload if sender else None, root=root)
        # Mapped storage now belongs to each request. Completion, cancellation
        # and error teardown drop it after the final use.
        return payload if sender else result

    def send(self, payload: Any, dest: int, tag: int, reply_tag: int) -> None:
        """Send ``payload`` to rank ``dest``, which calls ``recv``.

        Args:
            payload: The payload to send.
            dest: The receiving rank.
            tag: Tag of the payload messages.
            reply_tag: Tag of the receiver's replies; unused by other traffic
                between the two ranks.
        """
        dist = self.dist
        key = ("send", dest)
        if key not in self.enabled:
            # The receiver sends first, so neither side blocks on the other.
            self.enabled[key] = dist.recv_object(dest, reply_tag) == socket.gethostname()
            dist.send_object(socket.gethostname(), dest, tag)
            logger.debug(f"Same-host tensor sharing to rank {dest}: {self.enabled[key]}")
        if not self.enabled[key]:
            dist.send_object(payload, dest, tag)
            return

        exporter, message = _dump(payload, 1)
        dist.send_object(message, dest, tag)
        if message[3]:
            self._disable(key, "export failed: " + message[3])
        if message[0] is None or not message[2]:
            return
        consumed, errors = dist.recv_object(dest, reply_tag)
        exporter.release([consumed])
        if errors:
            self._disable(key, "mapping failed: " + repr(errors))
            dist.send_object(payload, dest, tag)

    def recv(self, src: int, tag: int, reply_tag: int) -> Any:
        """Receive a payload that rank ``src`` sends with ``send``."""
        dist = self.dist
        key = ("recv", src)
        if key not in self.enabled:
            dist.send_object(socket.gethostname(), src, reply_tag)
            self.enabled[key] = dist.recv_object(src, tag) == socket.gethostname()
        if not self.enabled[key]:
            return dist.recv_object(src, tag)

        data, raw, handles, error = dist.recv_object(src, tag)
        if error:
            self._disable(key, "export failed: " + error)
        if data is None:
            return raw
        result, consumed, errors = _load(data, handles, 0)
        if not handles:
            return result
        dist.send_object((consumed, errors), src, reply_tag)
        if errors:
            self._disable(key, "mapping failed: " + repr(errors))
            result = dist.recv_object(src, tag)
        return result

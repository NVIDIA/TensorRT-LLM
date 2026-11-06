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
"""Retention ownership for endpoint-private checkpoint producers.

This adapter does not submit transfers, own buffers, or infer CUDA/RDMA
completion. A producer must acquire before its first lookup/transfer, publish
its completion marker last, and report retirement only after *all* native export
copies and Store transfers have finished. Cancellation and timeout alone are
not retirement evidence. Buffer cleanup remains the producer's responsibility,
including when admission raises EndpointUnavailable.

Shared content-addressed attention pages must never enter a private manifest.
The attention-only connector cannot use this adapter until it has a private
checkpoint producer. No serving option is enabled by this module.
"""

from dataclasses import dataclass
from typing import Union

from .retention import CheckpointRetention, RetentionError, TransferLease
from .retention_rpc import RemoteCheckpointRetention, RemoteLease

Catalog = Union[CheckpointRetention, RemoteCheckpointRetention]
Lease = Union[TransferLease, RemoteLease]


@dataclass(frozen=True)
class CheckpointPublication:
    """Immutable private manifest and globally stable ownership identity.

    All owners of a conversation must use the same conversation and turn IDs.
    The limit counts first successful publication order, not logical turn order.
    An empty conversation conservatively disables reclamation of this endpoint.
    Namespace/endpoint key validation is performed by the catalog.
    """

    marker: str
    private_keys: tuple[str, ...]
    conversation: str
    turn: str

    def __post_init__(self) -> None:
        if not self.turn:
            raise ValueError("checkpoint publication requires a stable turn identity")
        if not isinstance(self.private_keys, tuple):
            raise TypeError("private_keys must be an immutable tuple")


class CheckpointTransfer:
    """Single-use, single-worker retention lease spanning lookup through completion.

    There is deliberately no context manager or destructor that releases a pin:
    leaving a scope or losing an object is not proof that transfers retired.
    After an uncertain completion the durable pin remains until the deployment
    is torn down; a late callback cannot silently reverse that decision.
    """

    def __init__(
        self,
        catalog: Catalog,
        lease: Lease,
        publication: CheckpointPublication | None,
    ) -> None:
        self._catalog = catalog
        self._lease = lease
        self._publication = publication
        self._finished = False

    @classmethod
    def begin_load(cls, catalog: Catalog, marker: str) -> "CheckpointTransfer":
        """Pin before checking the completion marker or submitting any GET."""
        return cls(catalog, catalog.acquire(marker), None)

    @classmethod
    def begin_save(
        cls, catalog: Catalog, publication: CheckpointPublication
    ) -> "CheckpointTransfer":
        """Pin and prepare reclamation metadata before submitting any Store PUT.

        Metadata failure releases only this Store lease (no Store I/O was
        submitted). The producer must still retire any native export copies
        before releasing their buffers.
        """
        lease = catalog.acquire(publication.marker)
        try:
            catalog.prepare_save(lease, publication.marker, publication.private_keys)
        except BaseException:
            lease.close(retired=True)
            raise
        return cls(catalog, lease, publication)

    def finish(self, *, retired: bool, success: bool) -> int:
        """Finish once; return the number of turns retired by a successful save.

        For saves, success asserts that all payload PUTs and the final completion
        marker PUT succeeded. A known-retired miss, capacity rejection, or
        cancellation uses success=False. An unknown transfer outcome must use
        retired=False and retain both the catalog pin and producer-owned buffers.
        A metadata failure keeps the pin and propagates to the caller.
        """
        if type(retired) is not bool or type(success) is not bool:
            raise TypeError("completion evidence must be boolean")
        if self._finished:
            raise RetentionError("checkpoint transfer already finished")
        if success and not retired:
            raise RetentionError("successful publication requires retired transfers")
        self._finished = True
        try:
            expired = 0
            if success and self._publication is not None:
                expired = self._catalog.publish(
                    self._lease, self._publication.conversation, self._publication.turn
                )
        except BaseException:
            self._lease.close(retired=False)
            raise
        self._lease.close(retired=retired)
        return expired

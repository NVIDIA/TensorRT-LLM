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

"""Allowlist for the peer endpoints that the native KV-transfer path dials.

Peer endpoints arrive from data a worker does not control -- ``ctx_info_endpoint``
on an incoming request, and ``sender_endpoints`` / ``self_endpoint`` /
``peer_endpoint`` inside peer replies -- and are connected to verbatim. When an
allowlist is configured, only endpoints whose host is a known peer are dialed.

Matching is on host only, since transfer endpoints bind an ephemeral port.
Disabled by default; a configured-but-unparsable value raises rather than
degrading to "allow everything".
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
from typing import Iterable, List, Optional, Sequence, Set

from tensorrt_llm import logger

ALLOWED_PEER_HOSTS_ENV = "TRTLLM_KV_TRANSFER_ALLOWED_PEER_HOSTS"

# Underscores are permitted because Kubernetes service names use them.
_HOSTNAME_RE = re.compile(
    r"^[a-z0-9_]([a-z0-9_-]*[a-z0-9_])?(\.[a-z0-9_]([a-z0-9_-]*[a-z0-9_])?)*$"
)

# Not urlparse: libzmq reads "tcp://<source>;<destination>" as "bind <source>,
# connect <destination>" while urlparse reports only <source>, so a permissive
# parse would allow "tcp://<allowed>:0;<attacker>:9999" and then dial <attacker>.
# \Z rather than $, which also matches before a trailing newline.
_ENDPOINT_RE = re.compile(r"\Atcp://(?P<host>\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9._-]+):(?:\*|\d{1,5})\Z")


class PeerEndpointNotAllowedError(ValueError):
    """Raised when a peer endpoint's host is not in the configured allowlist."""


def _normalize_host(host: str) -> str:
    host = host.strip().lower()
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if len(host) > 1 and host.endswith("."):
        host = host[:-1]
    return host


def _as_network(entry: str) -> Optional[ipaddress._BaseNetwork]:
    try:
        # strict=True keeps a mistyped "10.0.3.12/0" from widening to 0.0.0.0/0.
        return ipaddress.ip_network(entry, strict=True)
    except ValueError:
        return None


def _resolve(name: str) -> List[str]:
    try:
        infos = socket.getaddrinfo(name, None, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError, ValueError, OSError):
        logger.warning(
            f"Could not resolve {ALLOWED_PEER_HOSTS_ENV} entry {name!r}; it will "
            "only match endpoints that carry the name literally"
        )
        return []
    return sorted({info[4][0] for info in infos})


class PeerEndpointAllowlist:
    def __init__(self, entries: Iterable[str]):
        self._names: Set[str] = set()
        self._networks: List[ipaddress._BaseNetwork] = []
        provided = [entry for entry in map(_normalize_host, entries) if entry]
        rejected: List[str] = []
        for entry in provided:
            network = _as_network(entry)
            if network is not None:
                self._networks.append(network)
                continue
            if not _HOSTNAME_RE.match(entry):
                rejected.append(entry)
                continue
            # A hostname only matches an IP-based endpoint once resolved.
            self._names.add(entry)
            for address in _resolve(entry):
                resolved = _as_network(_normalize_host(address))
                if resolved is not None:
                    self._networks.append(resolved)
        if rejected:
            raise ValueError(
                f"{ALLOWED_PEER_HOSTS_ENV} contains entries that are neither a "
                f"hostname, an IP address, nor a CIDR block: {rejected}"
            )
        self._enabled = bool(provided)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def allows(self, endpoint: Optional[str]) -> bool:
        if not self._enabled:
            return True
        if not endpoint:
            return False
        match = _ENDPOINT_RE.match(endpoint)
        if match is None:
            return False
        host = _normalize_host(match.group("host"))
        if not host:
            return False
        if host in self._names:
            return True
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return False
        if address.version == 6 and address.ipv4_mapped is not None:
            address = address.ipv4_mapped
        return any(address in network for network in self._networks)

    def check(self, endpoint: Optional[str]) -> None:
        if not self.allows(endpoint):
            raise PeerEndpointNotAllowedError(
                f"Refusing to connect to peer endpoint {endpoint!r}: host is not "
                f"in {ALLOWED_PEER_HOSTS_ENV}"
            )


def _configured_entries() -> Sequence[str]:
    return os.getenv(ALLOWED_PEER_HOSTS_ENV, "").split(",")


_allowlist: Optional[PeerEndpointAllowlist] = None
_allowlist_error: Optional[ValueError] = None


def get_peer_endpoint_allowlist() -> PeerEndpointAllowlist:
    global _allowlist, _allowlist_error
    # Failures are cached too, so the dial path never re-parses or re-resolves.
    if _allowlist_error is not None:
        raise _allowlist_error
    if _allowlist is None:
        try:
            allowlist = PeerEndpointAllowlist(_configured_entries())
        except ValueError as error:
            _allowlist_error = error
            raise
        if allowlist.enabled:
            logger.info(f"KV transfer peer endpoint allowlist enabled via {ALLOWED_PEER_HOSTS_ENV}")
        _allowlist = allowlist
    return _allowlist


def reset_peer_endpoint_allowlist() -> None:
    global _allowlist, _allowlist_error
    _allowlist = None
    _allowlist_error = None


def check_peer_endpoint(endpoint: Optional[str]) -> None:
    get_peer_endpoint_allowlist().check(endpoint)

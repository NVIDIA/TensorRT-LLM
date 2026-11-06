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
"""Job-scoped JSON coordinator: hot metadata resides on local tmpfs, not Lustre.

No CUDA, native handles, or Store client on the service. Clients perform deletions
under exclusive server-held leases. Lost clients retain leases; there is no TTL.
A restarted service MUST get a fresh epoch/Store namespace. Bootstrap files are
exclusive-create, contain a random capability, and must not be printed.
"""

import argparse
import hmac
import http.client
import json
import os
import secrets
import socket
import threading
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .retention import CheckpointRetention, EndpointUnavailable, RetentionError, RetentionStore


class RetentionCoordinator:
    """Keep opaque lease IDs alive across independent worker RPC connections."""

    def __init__(self, directory: str, epoch: str) -> None:
        self.directory = directory
        self.epoch = epoch
        self._lock = threading.Lock()
        self._gc_lock = threading.Lock()
        self._catalogs = {}
        self._leases = {}
        self._claims = {}

    def dispatch(self, request: dict[str, Any]) -> Any:
        if request["epoch"] != self.epoch:
            raise RetentionError("coordinator epoch mismatch; never reconnect to a replacement")
        namespace = request["namespace"]
        max_turns = request["max_turns"]
        if type(max_turns) is not int or not 1 <= max_turns <= 100:
            raise RetentionError("invalid turn limit")
        if self.epoch not in namespace:
            raise RetentionError("namespace must contain the coordinator epoch")
        with self._lock:
            catalog = self._catalogs.get(namespace)
            if catalog is None:
                catalog = CheckpointRetention(self.directory, namespace, max_turns)
                self._catalogs[namespace] = catalog
            elif catalog.max_turns != max_turns:
                raise RetentionError("inconsistent turn limit")
        method, args = request["method"], request["args"]
        if method == "acquire":
            lease = catalog.acquire(args["marker"])
            token = uuid.uuid4().hex
            with self._lock:
                self._leases[token] = (namespace, lease)
            return token
        if method in ("prepare_save", "publish", "release"):
            if method == "release" and type(args["retired"]) is not bool:
                raise RetentionError("invalid retirement evidence")
            with self._lock:
                owner, lease = self._leases[args["token"]]
                if owner != namespace:
                    raise RetentionError("lease namespace mismatch")
                if method == "release":
                    del self._leases[args["token"]]
            if method == "release":
                lease.close(retired=args["retired"])
                return None
            if method == "prepare_save":
                catalog.prepare_save(lease, args["marker"], tuple(args["private_keys"]))
                return None
            return catalog.publish(lease, args["conversation"], args["turn"])
        if method == "claim":
            with self._gc_lock:
                claims, stats = catalog.claim(8)
                result = []
                with self._lock:
                    for claim in claims:
                        token = uuid.uuid4().hex
                        self._claims[token] = (namespace, claim)
                        result.append({"token": token, "manifest": claim[2]})
                return {"claims": result, "stats": stats}
        if method == "finish_claim":
            if type(args["success"]) is not bool:
                raise RetentionError("invalid deletion evidence")
            with self._lock:
                owner, claim = self._claims[args["token"]]
                if owner != namespace:
                    raise RetentionError("claim namespace mismatch")
            deferred = args.get("deferred", False)
            if type(deferred) is not bool or (deferred and args["success"]):
                raise RetentionError("invalid deferred deletion evidence")
            catalog.finish_claim(
                claim, success=args["success"], uncertain=not args["success"] and not deferred
            )
            with self._lock:
                del self._claims[args["token"]]
            return None
        raise RetentionError("unknown coordinator operation")


def make_server(
    address: tuple[str, int], coordinator: RetentionCoordinator, secret: str
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self) -> None:
            super().setup()
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        def log_message(self, *args: Any) -> None:
            pass  # Never print capability headers or cache identities.

        def do_POST(self) -> None:
            if not hmac.compare_digest(self.headers.get("Authorization", ""), secret):
                self.send_error(403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 2 * 1024 * 1024:
                    raise RetentionError("invalid metadata message size")
                request = json.loads(self.rfile.read(size))
                result = {"result": coordinator.dispatch(request)}
                status = 200
            except Exception as error:
                # Boundary: expose class only, never keys/paths/auth in error text.
                result = {"error": type(error).__name__}
                status = 409
            encoded = json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(address, Handler)
    server.daemon_threads = True
    return server


@dataclass
class RemoteLease:
    client: "RemoteCheckpointRetention"
    token: str

    def close(self, *, retired: bool) -> None:
        self.client._call("release", token=self.token, retired=retired)


class RemoteCheckpointRetention:
    """One HTTP connection per descriptor worker; calls are not retried implicitly."""

    def __init__(self, descriptor: dict[str, Any], namespace: str, max_turns: int = 5) -> None:
        self.descriptor = descriptor
        self.namespace = namespace
        self.max_turns = max_turns
        self._connection = http.client.HTTPConnection(
            descriptor["host"], descriptor["port"], timeout=10
        )

    def _call(self, method: str, **args: Any) -> Any:
        body = json.dumps(
            dict(
                epoch=self.descriptor["epoch"],
                namespace=self.namespace,
                max_turns=self.max_turns,
                method=method,
                args=args,
            )
        )
        try:
            self._connection.request(
                "POST",
                "/",
                body,
                {"Authorization": self.descriptor["token"], "Content-Type": "application/json"},
            )
            response = self._connection.getresponse()
            data = response.read()
            try:
                envelope = json.loads(data)
                if not isinstance(envelope, dict):
                    raise ValueError("invalid response envelope")
                if response.status == 409 and envelope == {"error": "EndpointUnavailable"}:
                    raise EndpointUnavailable("coordinator endpoint unavailable")
                if response.status != 200:
                    raise RetentionError("coordinator rejected metadata operation")
                return envelope["result"]
            except (ValueError, KeyError, TypeError) as error:
                raise RetentionError("invalid coordinator protocol response") from error
        except (OSError, http.client.HTTPException) as error:
            self._connection.close()
            raise RetentionError("coordinator unavailable; retain ownership and stop") from error

    def acquire(self, marker: str) -> RemoteLease:
        return RemoteLease(self, self._call("acquire", marker=marker))

    def prepare_save(self, lease: RemoteLease, marker: str, private_keys: tuple[str, ...]) -> None:
        self._call("prepare_save", token=lease.token, marker=marker, private_keys=private_keys)

    def publish(self, lease: RemoteLease, conversation: str, turn: str) -> int:
        return self._call("publish", token=lease.token, conversation=conversation, turn=turn)

    def collect(self, store: RetentionStore, limit: int = 8) -> dict[str, int]:
        response = self._call("claim")
        stats = response["stats"]
        for claim in response["claims"]:
            success = False
            deferred = False
            try:
                manifest = claim["manifest"]
                for key in [manifest["marker"], *manifest["private_keys"]]:
                    result = store.remove(key, False)
                    if type(result) is int and result == -706:
                        deferred = True
                        stats["gc_deferred_lease"] += 1
                        break
                    if type(result) is not int or result not in (0, -704):
                        raise RetentionError("non-forced Store deletion did not complete")
                    stats["gc_deleted_objects"] += int(result == 0)
                success = not deferred
                stats["gc_deleted_checkpoints"] += int(success)
            except (RuntimeError, OSError):
                stats["gc_errors"] += 1
            finally:
                self._call("finish_claim", token=claim["token"], success=success, deferred=deferred)
        return stats

    def close(self) -> None:
        self._connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptor", required=True)
    parser.add_argument(
        "--local-root", required=True, help="Node-local tmpfs path, e.g. /dev/shm/job"
    )
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--advertise", default=socket.gethostbyname(socket.gethostname()))
    args = parser.parse_args()
    epoch, secret = uuid.uuid4().hex, secrets.token_hex(32)
    coordinator = RetentionCoordinator(str(Path(args.local_root) / epoch), epoch)
    server = make_server((args.bind, 0), coordinator, secret)
    descriptor = dict(host=args.advertise, port=server.server_port, epoch=epoch, token=secret)
    # Never overwrite/reuse an epoch descriptor after service loss.
    target = Path(args.descriptor)
    temporary = target.with_name(target.name + "." + epoch + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as output:
            json.dump(descriptor, output)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, target)  # Atomic publication, fails if a descriptor exists.
    finally:
        temporary.unlink(missing_ok=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

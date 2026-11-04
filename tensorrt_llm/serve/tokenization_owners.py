# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenize each conversation in the frontend that caches it.

With ``num_serve_frontends > 1`` the kernel hands each connection to any
frontend, so consecutive turns of a conversation land in different processes,
each with its own prefix-tokenization cache. Every frontend therefore serves
its cache on a Unix socket in the multi-frontend ipc dir, and a prompt whose
conversation id hashes to another frontend is tokenized there, by the cache
that has seen the conversation's earlier turns. Only the prompt text and the
token ids cross the socket; the request itself stays where it arrived.
"""

import contextlib
import os
import socket
import socketserver
import struct
import threading
import zlib
from array import array
from typing import Any, Optional

from tensorrt_llm.inputs.prefix_token_cache import PrefixTokenCache, RouteFailed
from tensorrt_llm.logger import logger

SOCKET_NAME = "tokenize-{}.sock"
# Set to 1 to route tokenization by conversation id across serving frontends.
ENABLE_ENV_VAR = "TLLM_PREFIX_TOKEN_CACHE_ROUTE_BY_CONVERSATION"

_LENGTH = struct.Struct("!q")
# Sent in place of a token count when the owner could not tokenize the prompt.
_FAILED = -1
# Lone surrogates, which JSON can carry, survive the trip and fail in the
# tokenizer as they would locally.
_TEXT_ERRORS = "surrogatepass"


def routing_enabled() -> bool:
    """Whether tokenization is routed by conversation id (off unless enabled)."""
    return os.environ.get(ENABLE_ENV_VAR) == "1"


def owner_of(key: str, num_frontends: int) -> int:
    """The frontend that tokenizes conversation ``key``, the same in every process."""
    return zlib.crc32(key.encode()) % num_frontends


class TokenizationOwners:
    """Route ``cache``'s prompts to the frontend that owns their conversation.

    Serves this frontend's ``cache`` to the other frontends and installs
    ``cache.route``. A prompt the owner does not tokenize within ``timeout_s``,
    for example because it has not started or has died, is tokenized locally.
    """

    def __init__(
        self,
        ipc_dir: str,
        frontend_id: int,
        num_frontends: int,
        cache: PrefixTokenCache,
        tokenizer: Any,
        timeout_s: float = 5.0,
    ):
        self._ipc_dir = ipc_dir
        self._frontend_id = frontend_id
        self._num_frontends = num_frontends
        self._timeout_s = timeout_s
        # One connection per owner per tokenizing thread.
        self._connections = threading.local()
        self._all_connections: set[socket.socket] = set()
        self._lock = threading.Lock()

        path = os.path.join(ipc_dir, SOCKET_NAME.format(frontend_id))
        self._server = _Server(path, cache, tokenizer)
        threading.Thread(
            target=self._server.serve_forever, name="trtllm_tokenize_owner", daemon=True
        ).start()
        cache.route = self._route

    def close(self) -> None:
        """Stop serving and close every connection, in both directions."""
        self._server.shutdown()
        self._server.server_close()
        self._server.close_connections()
        with self._lock:
            for conn in self._all_connections:
                conn.close()
            self._all_connections.clear()

    def _route(self, key: str, text: str) -> Optional[list[int]]:
        owner = owner_of(key, self._num_frontends)
        if owner == self._frontend_id:
            return None
        try:
            return self._request(owner, text)
        except OSError as e:
            self._drop(owner)
            raise RouteFailed(f"frontend {owner} did not answer: {e!r}") from e

    def _request(self, owner: int, text: str) -> list[int]:
        conn = self._connection(owner)
        data = text.encode(errors=_TEXT_ERRORS)
        conn.sendall(_LENGTH.pack(len(data)) + data)
        (count,) = _LENGTH.unpack(_recv_exactly(conn, _LENGTH.size))
        if count == _FAILED:
            raise RouteFailed(f"frontend {owner} could not tokenize the prompt")
        ids = array("i")
        ids.frombytes(_recv_exactly(conn, count * ids.itemsize))
        return ids.tolist()

    def _connection(self, owner: int) -> socket.socket:
        conns = self._connections.__dict__
        conn = conns.get(owner)
        if conn is None:
            conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            conn.settimeout(self._timeout_s)
            try:
                conn.connect(os.path.join(self._ipc_dir, SOCKET_NAME.format(owner)))
            except OSError:
                conn.close()
                raise
            conns[owner] = conn
            with self._lock:
                self._all_connections.add(conn)
        return conn

    def _drop(self, owner: int) -> None:
        conn = self._connections.__dict__.pop(owner, None)
        if conn is not None:
            with self._lock:
                self._all_connections.discard(conn)
            conn.close()


class _Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    # Every tokenizing thread of every other frontend may connect at once, and a
    # Unix-socket connect to a full backlog fails at once instead of waiting.
    request_queue_size = socket.SOMAXCONN

    def __init__(self, path: str, cache: PrefixTokenCache, tokenizer: Any):
        self.cache = cache
        self.tokenizer = tokenizer
        # Accepted connections and the threads serving them.
        self.handlers: dict[socket.socket, threading.Thread] = {}
        self.handlers_lock = threading.Lock()
        if os.path.exists(path):
            os.unlink(path)
        super().__init__(path, _Handler)

    def close_connections(self, timeout_s: float = 5.0) -> None:
        """Wake every handler blocked on its peer and wait for it to exit."""
        with self.handlers_lock:
            handlers = list(self.handlers.items())
        for conn, _ in handlers:
            with contextlib.suppress(OSError):
                conn.shutdown(socket.SHUT_RDWR)
        for _, thread in handlers:
            thread.join(timeout_s)


class _Handler(socketserver.BaseRequestHandler):
    server: _Server

    def setup(self) -> None:
        with self.server.handlers_lock:
            self.server.handlers[self.request] = threading.current_thread()

    def finish(self) -> None:
        with self.server.handlers_lock:
            self.server.handlers.pop(self.request, None)

    def handle(self) -> None:
        # Ends when the peer closes the connection or close_connections() runs.
        with contextlib.suppress(OSError):
            while True:
                self._answer_one()

    def _answer_one(self) -> None:
        conn = self.request
        (size,) = _LENGTH.unpack(_recv_exactly(conn, _LENGTH.size))
        text = _recv_exactly(conn, size).decode(errors=_TEXT_ERRORS)
        try:
            ids = array("i", self.server.cache.encode(self.server.tokenizer, text))
        except Exception as e:
            # The asking frontend tokenizes the prompt itself and reports the
            # error with its request.
            logger.debug(f"Could not tokenize a prompt for another frontend: {e!r}")
            conn.sendall(_LENGTH.pack(_FAILED))
            return
        conn.sendall(_LENGTH.pack(len(ids)) + ids.tobytes())


def _recv_exactly(conn: socket.socket, size: int) -> bytearray:
    buf = bytearray(size)
    view = memoryview(buf)
    while view:
        n = conn.recv_into(view)
        if n == 0:
            raise ConnectionError("peer closed the connection")
        view = view[n:]
    return buf

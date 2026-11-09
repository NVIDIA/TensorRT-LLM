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
conversation id maps to another frontend is tokenized there, by the cache that
has seen the conversation's earlier turns. Only the prompt text and the token
ids cross the socket; the request itself stays where it arrived.

Each frontend keeps one connection to every other frontend, shared by its
tokenizing threads, so a server with K frontends holds 2 * (K - 1) routing
sockets and K - 1 serving threads per frontend.
"""

import contextlib
import os
import socket
import socketserver
import struct
import threading
import time
from array import array
from typing import Optional

from tensorrt_llm.inputs.prefix_token_cache import OffsetTokenizer, PrefixTokenCache, RouteFailed
from tensorrt_llm.inputs.registry import prefix_token_cache_of
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.multi_frontend import (
    MultiFrontendServing,
    frontend_of_conversation,
    tokenization_uds_path,
)

# Set to 1 to route tokenization by conversation id across serving frontends.
ENABLE_ENV_VAR = "TLLM_PREFIX_TOKEN_CACHE_ROUTE_BY_CONVERSATION"

_LENGTH = struct.Struct("!q")
# Sent in place of a token count when the owner could not tokenize the prompt.
_FAILED = -1
# Lone surrogates, which JSON can carry, survive the trip and fail in the
# tokenizer as they would locally.
_TEXT_ERRORS = "surrogatepass"


def routing_enabled() -> bool:
    """Whether tokenization is routed by conversation id (off unless enabled).

    Raises:
        ValueError: if the switch is set to anything but ``0`` or ``1``.
    """
    raw = os.environ.get(ENABLE_ENV_VAR)
    if raw is None or raw == "0":
        return False
    if raw == "1":
        return True
    raise ValueError(f"{ENABLE_ENV_VAR} must be 0 or 1, got {raw!r}")


def start_tokenization_owners(
    multi_frontend_serving: Optional[MultiFrontendServing], input_processor: object
) -> Optional["TokenizationOwners"]:
    """Route this frontend's prompts by conversation if the switch is on and can engage.

    Raises:
        ValueError: if the switch is malformed.
    """
    if not routing_enabled():
        return None
    num_frontends = 1 if multi_frontend_serving is None else multi_frontend_serving.num_frontends
    prefix_cache = prefix_token_cache_of(input_processor)
    if num_frontends <= 1 or prefix_cache is None:
        logger.warning(
            f"{ENABLE_ENV_VAR}=1 has no effect: it needs num_serve_frontends > 1 "
            f"(got {num_frontends}) and an enabled prefix token cache "
            f"(enable_tokenization_cache: {prefix_cache is not None})."
        )
        return None
    logger.info(
        f"Routing tokenization by conversation id: frontend "
        f"{multi_frontend_serving.frontend_id} of {num_frontends}."
    )
    return TokenizationOwners(
        os.path.dirname(multi_frontend_serving.launcher_uds),
        multi_frontend_serving.frontend_id,
        num_frontends,
        *prefix_cache,
    )


class _Peer:
    """This frontend's connection to one owner, shared by its tokenizing threads."""

    def __init__(self) -> None:
        # Held for a whole exchange, which keeps requests and replies paired.
        self.lock = threading.Lock()
        self.conn: Optional[socket.socket] = None
        # Monotonic time before which the owner, which timed out, is not asked.
        self.skip_until = 0.0


class TokenizationOwners:
    """Route ``cache``'s prompts to the frontend that owns their conversation.

    Serves this frontend's ``cache`` to the other frontends and installs its
    route. A prompt the owner does not tokenize within ``timeout_s``, for
    example because it has hung, is tokenized locally, and so is every prompt
    it owns for the next ``retry_after_s``. A prompt whose owner is not serving
    (not started yet, or dead) or cannot tokenize it is tokenized locally too.
    """

    def __init__(
        self,
        ipc_dir: str,
        frontend_id: int,
        num_frontends: int,
        cache: PrefixTokenCache,
        tokenizer: OffsetTokenizer,
        timeout_s: float = 5.0,
        retry_after_s: float = 30.0,
    ) -> None:
        self._ipc_dir = ipc_dir
        self._frontend_id = frontend_id
        self._num_frontends = num_frontends
        self._cache = cache
        self._timeout_s = timeout_s
        self._retry_after_s = retry_after_s
        self._peers = {owner: _Peer() for owner in range(num_frontends) if owner != frontend_id}

        self._server = _Server(tokenization_uds_path(ipc_dir, frontend_id), cache, tokenizer)
        threading.Thread(
            target=self._server.serve_forever, name="trtllm_tokenize_owner", daemon=True
        ).start()
        cache.set_route(self._route)

    def close(self, timeout_s: float = 5.0) -> None:
        """Stop routing and serving, and close every connection in both directions.

        Blocks for up to about ``timeout_s`` while handlers finish the prompt
        they are tokenizing; those still running then are left to exit with
        the process.
        """
        deadline = time.monotonic() + timeout_s
        self._cache.set_route(None)
        self._server.shutdown()
        self._server.server_close()
        self._server.close_connections(deadline)
        for peer in self._peers.values():
            conn = peer.conn
            if conn is not None:
                # Wakes a thread waiting on the owner, which then tokenizes
                # its prompt locally.
                with contextlib.suppress(OSError):
                    conn.shutdown(socket.SHUT_RDWR)
                conn.close()

    def _route(self, key: str, text: str) -> Optional[list[int]]:
        owner = frontend_of_conversation(key, self._num_frontends)
        if owner == self._frontend_id:
            return None
        peer = self._peers[owner]
        if time.monotonic() < peer.skip_until:
            raise RouteFailed(f"frontend {owner} timed out recently")
        if not peer.lock.acquire(timeout=self._timeout_s):
            raise RouteFailed(f"frontend {owner} is busy")
        try:
            return self._request(owner, peer, text)
        except RouteFailed:
            raise
        except TimeoutError as e:
            _drop(peer)
            peer.skip_until = time.monotonic() + self._retry_after_s
            logger.warning(
                f"Frontend {owner} did not tokenize a prompt within {self._timeout_s} s; "
                f"tokenizing its conversations locally for {self._retry_after_s} s."
            )
            raise RouteFailed(f"frontend {owner} timed out") from e
        except Exception as e:
            # Not serving (connect failed), died mid-exchange, or sent a
            # malformed reply: the connection cannot be reused.
            _drop(peer)
            raise RouteFailed(f"frontend {owner} did not answer: {e!r}") from e
        finally:
            peer.lock.release()

    def _request(self, owner: int, peer: _Peer, text: str) -> list[int]:
        if peer.conn is None:
            conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            conn.settimeout(self._timeout_s)
            try:
                conn.connect(tokenization_uds_path(self._ipc_dir, owner))
            except OSError:
                conn.close()
                raise
            peer.conn = conn
        data = text.encode(errors=_TEXT_ERRORS)
        peer.conn.sendall(_LENGTH.pack(len(data)) + data)
        (count,) = _LENGTH.unpack(_recv_exactly(peer.conn, _LENGTH.size))
        if count == _FAILED:
            raise RouteFailed(f"frontend {owner} could not tokenize the prompt")
        if count < 0:
            raise ValueError(f"invalid token count {count}")
        ids = array("i")
        ids.frombytes(_recv_exactly(peer.conn, count * ids.itemsize))
        return ids.tolist()


def _drop(peer: _Peer) -> None:
    """Caller holds ``peer.lock``."""
    if peer.conn is not None:
        peer.conn.close()
        peer.conn = None


class _Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    # Every other frontend may connect at once, and a Unix-socket connect to a
    # full backlog fails at once instead of waiting.
    request_queue_size = socket.SOMAXCONN

    def __init__(self, path: str, cache: PrefixTokenCache, tokenizer: OffsetTokenizer) -> None:
        self.cache = cache
        self.tokenizer = tokenizer
        # Accepted connections and the threads serving them.
        self.handlers: dict[socket.socket, threading.Thread] = {}
        self.handlers_lock = threading.Lock()
        if os.path.exists(path):
            os.unlink(path)
        super().__init__(path, _Handler)

    def close_connections(self, deadline: float) -> None:
        """Wake every handler blocked on its peer and wait for it to exit until ``deadline``."""
        with self.handlers_lock:
            handlers = list(self.handlers.items())
        for conn, _ in handlers:
            with contextlib.suppress(OSError):
                conn.shutdown(socket.SHUT_RDWR)
        for _, thread in handlers:
            thread.join(max(0.0, deadline - time.monotonic()))


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

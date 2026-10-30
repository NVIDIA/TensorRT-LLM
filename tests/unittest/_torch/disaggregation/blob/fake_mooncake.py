# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A fake ``mooncake.store.MooncakeDistributedStore`` that moves real bytes and can be scripted.

It implements only the SDK methods the Mooncake driver may call (plus ``setup`` and ``close`` for
the opening code), so a call to any other one, a copying API for instance, fails with
``AttributeError``. Several fakes share one ``FakePool``, as clients of one master do.

Unscripted calls follow SDK 0.3.13:

- ``batch_put_session_start`` reserves an object on the pool: -600 when this client already holds
  the key's session, -705 when the pool holds the key (committed or reserved).
  ``batch_put_from_multi_buffer_ranges`` copies into the reservation (-600 without a session).
  ``batch_put_session_end`` commits it and ``batch_put_session_revoke`` drops it; both answer -600
  without a session and keep the session when scripted to fail.
- ``batch_get_into`` answers each object's size after copying it (-704 absent, -703 reserved only,
  -600 when the object is longer than its buffer).
- A buffer outside every registration fails with TRANSFER_FAIL, as RDMA does.

``KeyGatedClient`` stops one method's calls naming a key at a gate, in front of any SDK client,
this fake or a real one.
"""

from __future__ import annotations

import ctypes
import threading
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

OBJECT_EXISTS, OBJECT_ABSENT = 1, 0
INTERNAL_ERROR = -1
INVALID_PARAMS = -600
REPLICA_IS_NOT_READY = -703
OBJECT_NOT_FOUND = -704
OBJECT_ALREADY_EXISTS = -705
TRANSFER_FAIL = -800
GATED_CALL_S = 50.0  # how long a blocked call waits for its gate: past any test's own wait


@dataclass
class FakePool:
    """The objects of one Mooncake pool: committed ones, and reservations by put sessions."""

    objects: dict[str, bytes] = field(default_factory=dict)
    reserved: dict[str, bytearray] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass(frozen=True)
class SdkCall:
    method: str
    thread: int
    args: tuple
    kwargs: dict
    answers: list = field(default_factory=list)  # filled in once the call answers


class Gate:
    """Holds one SDK call: ``entered`` is set when it arrives, it goes on once ``release`` is."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()


class KeyGatedClient:
    """An SDK client whose ``method`` calls naming ``key`` wait inside the call until ``release`` is
    set; every other call goes straight through."""

    def __init__(self, store, method: str, key: str) -> None:
        self.store = store
        self.method = method
        self.key = key
        self.entered = threading.Event()
        self.release = threading.Event()

    def __getattr__(self, name: str):
        call = getattr(self.store, name)
        if name != self.method:
            return call

        def gated(keys, *args):
            if self.key in keys:
                self.entered.set()
                # Past the suite's timeout: only the check's unblock lets it go.
                self.release.wait(GATED_CALL_S)
            return call(keys, *args)

        return gated


class FakeMooncakeStore:
    """See the module docstring. ``script`` and ``block`` steer the next calls of a method."""

    def __init__(self, pool: FakePool | None = None) -> None:
        self.pool = pool if pool is not None else FakePool()
        self.calls: list[SdkCall] = []
        self.registered: dict[int, int] = {}
        self.put_sessions: set[str] = set()
        self.setup_answer = 0
        self.closed = False
        self._lock = threading.Lock()
        self._steps: dict[str, deque] = defaultdict(deque)
        self._gates: dict[str, deque[Gate]] = defaultdict(deque)

    def script(self, method: str, step) -> None:
        """Steers the next call of ``method`` that has no step yet. A list answers per key,
        ``None`` meaning the real behaviour for that key; an exception is raised; a callable
        rewrites the real answers (to drop or add one, say)."""
        self._steps[method].append(step)

    def block(self, method: str) -> Gate:
        """Holds the next call of ``method``, before it does anything, until the gate opens."""
        gate = Gate()
        self._gates[method].append(gate)
        return gate

    def calls_to(self, method: str) -> list[SdkCall]:
        with self._lock:
            return [call for call in self.calls if call.method == method]

    def keys_answered(self, method: str, answer: int) -> list[str]:
        """Every key a call of ``method`` answered ``answer`` for, in call order."""
        with self._lock:
            return [
                key
                for call in self.calls
                if call.method == method
                for key, got in zip(call.args[0], call.answers)
                if got == answer
            ]

    # The SDK surface.

    def setup(self, *args, **kwargs) -> int:
        self._record("setup", args, kwargs)
        return self.setup_answer

    def close(self) -> int:
        self._record("close", (), {})
        self.closed = True
        return 0

    def register_buffer(self, buffer_ptr: int, size: int) -> int:
        args = (buffer_ptr, size)
        (answer,) = self._answer("register_buffer", args, {}, 1, lambda _: self._register(*args))
        return answer

    def unregister_buffer(self, buffer_ptr: int) -> int:
        args = (buffer_ptr,)
        (answer,) = self._answer(
            "unregister_buffer", args, {}, 1, lambda _: self._unregister(*args)
        )
        return answer

    def batch_put_session_start(self, *args, **kwargs) -> list[int]:
        keys, sizes = args
        return self._answer(
            "batch_put_session_start",
            args,
            kwargs,
            len(keys),
            lambda i: self._start_put(keys[i], sizes[i]),
        )

    def batch_put_from_multi_buffer_ranges(
        self, keys, all_buffer_ptrs, all_sizes, all_dst_offsets
    ) -> list[int]:
        args = (keys, all_buffer_ptrs, all_sizes, all_dst_offsets)
        return self._answer(
            "batch_put_from_multi_buffer_ranges",
            args,
            {},
            len(keys),
            lambda i: self._write(keys[i], all_buffer_ptrs[i], all_sizes[i], all_dst_offsets[i]),
        )

    def batch_put_session_end(self, keys: Sequence[str]) -> list[int]:
        return self._answer(
            "batch_put_session_end", (keys,), {}, len(keys), lambda i: self._end_put(keys[i])
        )

    def batch_put_session_revoke(self, keys: Sequence[str]) -> list[int]:
        return self._answer(
            "batch_put_session_revoke", (keys,), {}, len(keys), lambda i: self._revoke(keys[i])
        )

    def batch_is_exist(self, keys: Sequence[str]) -> list[int]:
        return self._answer(
            "batch_is_exist", (keys,), {}, len(keys), lambda i: self._exists(keys[i])
        )

    def batch_get_into(self, *args, **kwargs) -> list[int]:
        keys, pointers, sizes = args
        return self._answer(
            "batch_get_into",
            args,
            kwargs,
            len(keys),
            lambda i: self._read(keys[i], pointers[i], sizes[i]),
        )

    # Real behaviour, one key at a time.

    def _register(self, pointer: int, size: int) -> int:
        with self._lock:
            self.registered[pointer] = size
        return 0

    def _unregister(self, pointer: int) -> int:
        with self._lock:
            return 0 if self.registered.pop(pointer, None) is not None else INTERNAL_ERROR

    def _start_put(self, key: str, size: int) -> int:
        with self._lock, self.pool.lock:
            if key in self.put_sessions:
                return INVALID_PARAMS
            if key in self.pool.objects or key in self.pool.reserved:
                return OBJECT_ALREADY_EXISTS
            self.pool.reserved[key] = bytearray(size)
            self.put_sessions.add(key)
        return 0

    def _write(self, key: str, pointers: list[int], sizes: list[int], offsets: list[int]) -> int:
        with self._lock, self.pool.lock:
            target = self.pool.reserved.get(key) if key in self.put_sessions else None
        if target is None or any(o + n > len(target) for o, n in zip(offsets, sizes)):
            return INVALID_PARAMS
        if not all(self._is_registered(p, n) for p, n in zip(pointers, sizes)):
            return TRANSFER_FAIL
        for pointer, size, offset in zip(pointers, sizes, offsets):
            target[offset : offset + size] = ctypes.string_at(pointer, size)
        return sum(sizes)

    def _end_put(self, key: str) -> int:
        with self._lock, self.pool.lock:
            if key not in self.put_sessions:
                return INVALID_PARAMS
            self.put_sessions.discard(key)
            self.pool.objects[key] = bytes(self.pool.reserved.pop(key))
        return 0

    def _revoke(self, key: str) -> int:
        with self._lock, self.pool.lock:
            if key not in self.put_sessions:
                return INVALID_PARAMS
            self.put_sessions.discard(key)
            self.pool.reserved.pop(key, None)
        return 0

    def _exists(self, key: str) -> int:
        with self.pool.lock:
            return OBJECT_EXISTS if key in self.pool.objects else OBJECT_ABSENT

    def _read(self, key: str, pointer: int, size: int) -> int:
        with self.pool.lock:
            data, reserved = self.pool.objects.get(key), key in self.pool.reserved
        if data is None:
            return REPLICA_IS_NOT_READY if reserved else OBJECT_NOT_FOUND
        if len(data) > size:
            return INVALID_PARAMS
        if not self._is_registered(pointer, size):
            return TRANSFER_FAIL
        ctypes.memmove(pointer, data, len(data))
        return len(data)

    def _is_registered(self, pointer: int, size: int) -> bool:
        with self._lock:
            return any(
                start <= pointer and pointer + size <= start + length
                for start, length in self.registered.items()
            )

    # Scripting.

    def _record(self, method: str, args: tuple, kwargs: dict) -> SdkCall:
        call = SdkCall(method, threading.get_ident(), args, kwargs)
        with self._lock:
            self.calls.append(call)
        return call

    def _answer(self, method, args, kwargs, count, real: Callable[[int], int]) -> list[int]:
        call = self._record(method, args, kwargs)
        if self.closed and method.startswith("batch_"):
            return []  # a closed 0.3.13 client answers every batch call with no answer at all
        with self._lock:
            gate = self._gates[method].popleft() if self._gates[method] else None
            step = self._steps[method].popleft() if self._steps[method] else None
        if gate is not None:
            gate.entered.set()
            gate.release.wait(GATED_CALL_S)
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, list):
            assert len(step) == count, f"a scripted {method} answers {count} keys"
            answers = [real(i) if answer is None else answer for i, answer in enumerate(step)]
        else:
            answers = [real(i) for i in range(count)]
            answers = step(answers) if callable(step) else answers
        call.answers.extend(answers)
        return answers

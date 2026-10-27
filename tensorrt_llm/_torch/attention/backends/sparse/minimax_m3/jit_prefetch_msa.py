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
"""Triton variant provider for MiniMax-M3's MSA block-sparse decode.

``minimax_m3_sparse_attn_decode`` (``kernels/triton_sparse_decode.py``) runs
the producer kernel (``_gqa_sparse_decode_kernel``, or its query-pair form)
and, with more than one top-k chunk, the merge kernel, in every sparse layer
of a generation step. Its specialization depends on the batch through
``total_q`` (an integer argument, and the GB300-tuned tables that pick the
split-K factor, the producer, and its warps and stages) and through the
alignment of the strided query and top-k views it is given.

The launcher is wrapped while warmup runs (``start_recording`` /
``stop_recording``), recording one call per launch class: every tensor argument
described by its layout without its per-token or per-request dimension, and
the scalars. As soon as a launch class is recorded (``set_on_new_record``),
so that warmup's later launches of it find their variants built or being
built, the provider replays the record on ``meta`` tensors under
``shadow_launches`` for every generation request count up to the engine's
(times the decode span for the token count), with each row-sliced argument
whose alignment a row offset can change aligned and not: the launcher's own
code derives every Triton argument as the real call would.
Its split-K scratch comes from the persistent buffer arena, which the
replaying thread sees as meta allocations, so planning allocates no device
memory.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import itertools
import threading
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch

from .....jit_prefetch import KernelCall, shadow_launches

# Parameters whose leading dimension is the token: q and output
# ([total_q, heads, head_dim]) and the top-k table ([kv_heads, total_q, topk],
# token in dim 1). block_table / seq_lens are per request.
_TOKEN_DIM = {"q": 0, "output": 0, "topk_idx": 1}
_REQUEST_ARGS = ("block_table", "seq_lens")

# name -> (launcher, {param: spec}).
_RECORDS: Dict[tuple, Tuple[Any, Dict[str, tuple]]] = {}
_LOCK = threading.Lock()
_PATCHES: List[Tuple[Any, str, Any]] = []
_RECORDING = False
# Called after each new launch class is recorded (set_on_new_record).
_ON_NEW_RECORD: Optional[Callable[[], None]] = None


def _aligned(t: torch.Tensor) -> bool:
    return (t.storage_offset() * t.element_size()) % 16 == 0


def _spec(param: str, value: Any) -> tuple:
    """A description of one launcher argument independent of the batch: the
    per-token and per-request dimension left open, and the alignment left open
    (None) where a row offset can change it (rows not a multiple of 16
    bytes: the generation rows start after the context rows)."""
    if not isinstance(value, torch.Tensor):
        return ("value", value)
    shape, stride = list(value.shape), tuple(value.stride())
    aligned = _aligned(value)
    dim = _TOKEN_DIM.get(param)
    if dim is None and param in _REQUEST_ARGS:
        dim = 0
    if dim is None:
        return ("tensor", tuple(shape), stride, value.dtype, aligned, None)
    shape[dim] = None
    row_bytes = stride[dim] * value.element_size()
    kind = "tokens" if param in _TOKEN_DIM else "requests"
    return (kind, tuple(shape), stride, value.dtype, None if row_bytes % 16 else aligned, dim)


def _meta(spec: tuple, tokens: int, requests: int, aligned: bool) -> torch.Tensor:
    kind, shape, stride, dtype, _, dim = spec
    shape = list(shape)
    if kind == "tokens":
        shape[dim] = tokens
    elif kind == "requests":
        shape[dim] = requests
    offset = 0 if aligned else 1
    span = 1 + sum((s - 1) * st for s, st in zip(shape, stride) if s > 0)
    base = torch.empty(offset + max(span, 1), dtype=dtype, device="meta")
    return base.as_strided(tuple(shape), stride, offset)


def _record(launcher, sig: inspect.Signature, args, kwargs) -> None:
    bound = sig.bind_partial(*args, **kwargs)
    specs = {}
    for param, value in bound.arguments.items():
        if isinstance(value, torch.Tensor) and (value.is_meta or not value.is_cuda):
            return
        specs[param] = _spec(param, value)
    key = tuple(sorted(specs.items(), key=lambda kv: kv[0]))
    with _LOCK:
        new = key not in _RECORDS
        _RECORDS.setdefault(key, (launcher, specs))
    hook = _ON_NEW_RECORD
    if new and hook is not None:
        hook()


def set_on_new_record(hook: Optional[Callable[[], None]]) -> None:
    """Call ``hook`` after each launch class recorded from here on, until
    recording stops: JIT compile ahead plans the class's other batches then,
    while warmup still runs, rather than after warmup. The hook runs on the
    launching thread, before the launch, so it must not block."""
    global _ON_NEW_RECORD
    _ON_NEW_RECORD = hook


def start_recording() -> None:
    """Record the MSA decode launches of the forwards that follow (warmup)."""
    global _RECORDING
    from .kernels import triton_sparse_decode as tsd

    orig = tsd.minimax_m3_sparse_attn_decode
    if getattr(orig, "_jitp_launcher", None) is None:
        sig = inspect.signature(orig)

        @functools.wraps(orig)
        def wrapper(*args, **kwargs):
            if _RECORDING:
                try:
                    _record(orig, sig, args, kwargs)
                except Exception:  # noqa: BLE001 - recording must never break a launch
                    pass
            return orig(*args, **kwargs)

        wrapper._jitp_launcher = orig
        _PATCHES.append((tsd, "minimax_m3_sparse_attn_decode", orig))
        tsd.minimax_m3_sparse_attn_decode = wrapper
    _RECORDING = True


def stop_recording() -> None:
    """Stop recording, drop the new-record hook, and restore the launcher."""
    global _RECORDING, _ON_NEW_RECORD
    _RECORDING = False
    _ON_NEW_RECORD = None
    while _PATCHES:
        owner, attr, orig = _PATCHES.pop()
        setattr(owner, attr, orig)


class _MetaBuffers:
    """Stands in for the persistent buffer arena during a replay."""

    @staticmethod
    def get_buffer(tensor_shape, dtype, buffer_name=None, reserve_buffer=False):
        return torch.empty(tuple(tensor_shape), dtype=dtype, device="meta")


_CAPABILITY_LOCK = threading.Lock()
_CAPABILITY_SHIMMED = False


def _shim_meta_device_capability() -> None:
    """Let ``torch.cuda.get_device_capability(meta_tensor.device)`` (the
    launcher's NVFP4 support check) answer for the current device. Every other
    call behaves exactly as before."""
    global _CAPABILITY_SHIMMED
    with _CAPABILITY_LOCK:
        if _CAPABILITY_SHIMMED:
            return
        orig = torch.cuda.get_device_capability

        @functools.wraps(orig)
        def get_device_capability(device=None):
            if isinstance(device, torch.device) and device.type == "meta":
                device = None
            return orig(device)

        torch.cuda.get_device_capability = get_device_capability
        _CAPABILITY_SHIMMED = True


@contextlib.contextmanager
def _meta_buffers():
    """For a replay on this thread, the persistent buffer arena allocates on
    ``meta``; other threads (the executor may be serving) keep the real one."""
    from .kernels import triton_sparse_decode as tsd

    me = threading.get_ident()
    orig = tsd.get_memory_buffers

    def get_memory_buffers():
        return _MetaBuffers() if threading.get_ident() == me else orig()

    tsd.get_memory_buffers = get_memory_buffers
    try:
        yield
    finally:
        tsd.get_memory_buffers = orig


class MiniMaxM3MsaDecodeProvider:
    """Replays each recorded MSA decode launch for every reachable batch."""

    kernel_modules = (
        "tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.triton_sparse_decode",
    )
    # Enumerated again after every warmup (JitPrefetcher.start_background):
    # the final executor's pool and buffers differ from the first warmup's.
    plans_from_warmup = True

    def __init__(self, max_num_requests: int):
        self.max_num_requests = max(1, int(max_num_requests))
        self._planned: set = set()
        self._lock = threading.Lock()

    def __bool__(self):
        return bool(_RECORDS)

    def __call__(self, batch_ctx: Any, seen: Optional[set] = None) -> List[KernelCall]:
        if not getattr(batch_ctx, "minimax_m3_msa_all", False):
            return []
        return self.plan_all()

    def enumerate_batches(self) -> Iterator[Any]:
        if self:
            yield SimpleNamespace(minimax_m3_msa_all=True, ctx_chunk_lens=[])

    def plan_all(self) -> List[KernelCall]:
        from tensorrt_llm.logger import logger

        with _LOCK:
            records = [(k, r) for k, r in _RECORDS.items() if k not in self._planned]
            self._planned.update(k for k, _ in records)
        calls: List[KernelCall] = []
        failed = 0
        _shim_meta_device_capability()
        with self._lock, _meta_buffers():
            for _, (launcher, specs) in records:
                span = max(1, int(specs["decode_query_len"][1]))
                free = [p for p, s in specs.items() if s[0] != "value" and s[4] is None]
                for requests in range(1, self.max_num_requests + 1):
                    for bits in itertools.product((True, False), repeat=len(free)):
                        aligned = dict(zip(free, bits))
                        kwargs = {}
                        for p, s in specs.items():
                            if s[0] == "value":
                                kwargs[p] = s[1]
                            else:
                                a = aligned[p] if p in aligned else s[4]
                                kwargs[p] = _meta(s, requests * span, requests, a)
                        try:
                            with shadow_launches() as rec:
                                launcher(**kwargs)
                        except Exception as e:  # noqa: BLE001 - plans nothing
                            failed += 1
                            logger.warning_once(
                                f"[JIT prefetch] cannot replay the MSA decode: {e}",
                                key="jitp_msa_decode",
                            )
                            continue
                        calls.extend(rec)
        logger.info(
            f"[JIT prefetch] MiniMax-M3 MSA decode: {len(records)} launch class(es), "
            f"{len(calls)} launch(es) planned, {failed} failed replay(s)"
        )
        return calls


def recorded_launch_classes() -> int:
    with _LOCK:
        return len(_RECORDS)


def start_warmup_builds(prefetcher, engine) -> None:
    """Before warmup's first forward, register the MSA decode provider so it
    plans each decode launch class as warmup records it, and plan the classes
    the prewarms already recorded."""
    prefetcher.register("minimax_m3_msa", MiniMaxM3MsaDecodeProvider(engine.batch_size))
    # The first warmup's, which keeps the classes it planned.
    msa_provider = prefetcher.registered("minimax_m3_msa")
    set_on_new_record(lambda: prefetcher.plan_recorded(msa_provider))
    if msa_provider:
        # Launches recorded before the first forward (prewarms).
        prefetcher.plan_recorded(msa_provider)

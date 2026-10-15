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
"""Triton variant provider for the Mamba2 decode-side kernels.

Covers the launches whose Triton specialization depends on the decode batch
and that also run eagerly (mixed prefill + decode batches, or decode batches
no captured CUDA graph covers):

* the native ``selective_state_update`` (head dims FlashInfer does not
  support, e.g. Nemotron-Nano-9B-v2's 80);
* the MTP verify path: ``replay_selective_state_update`` (precompute and
  persistent main kernels) and the Triton ``causal_conv1d_update``;
* ``_promote_mamba_state_triton``, which commits the accepted MTP states.

What varies from batch to batch is the number of generation requests (an int
argument of every kernel, and the tuning bucket of the replay update) and the
16-byte alignment of the per-request inputs. Those are row slices of longer
buffers, and not all from the same row: some start after the context rows
(``state_indices[num_prefills:]``, ``num_accepted_tokens[num_contexts:]``),
others at row 0 (``intermediate_state_indices[:num_decodes]``), so in one
launch some can be aligned and others not.

The launchers are wrapped while warmup runs (``start_recording`` /
``stop_recording``): each real call is recorded once per launch class, with
every per-request tensor (``_BATCH_ARGS``) described independently of the
batch size and of its alignment. Recording stops at the end of warmup, so
serving pays nothing. A launch class also fixes the cache pool (its size is a
constexpr of the conv update and sets the promote kernel's strides), so the
classes the final executor's warmup records are planned after it, in addition
to those of the KV-cache estimation pass.

The provider then replays each recorded call on ``meta`` tensors under
``shadow_launches`` for every batch size up to the engine's and every
combination of alignments of the per-request tensors whose alignment a row
offset can change (rows that are not a multiple of 16 bytes); an argument the
real call passed as the same tensor as another stays that tensor. The
launcher's own code derives every Triton argument as the real call would.
"""

from __future__ import annotations

import functools
import inspect
import itertools
import threading
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch

from ...jit_prefetch import KernelCall, shadow_launches

# Per launcher: the parameters indexed by generation request (dim 0), and
# whether its specialization depends on more than the batch's integer class
# (the replay update picks its kernels and their constexpr knobs from a
# tuning table by batch size, so every batch size is planned).
_BATCH_ARGS: Dict[str, Tuple[Tuple[str, ...], bool]] = {
    "selective_state_update": (
        ("x", "dt", "B", "C", "z", "out", "state_batch_indices", "retrieve_parent_token",
         "intermediate_state_indices"),
        False,
    ),
    "replay_selective_state_update": (
        ("x", "dt", "B", "C", "z", "out", "replay_work_items", "state_batch_indices"),
        True,
    ),
    "causal_conv1d_update": (
        ("x", "cache_seqlens", "conv_state_indices", "num_accepted_tokens",
         "intermediate_state_indices", "retrieve_next_token", "retrieve_next_sibling",
         "retrieve_parent_token"),
        False,
    ),
    "_promote_mamba_state_triton": (
        ("src_state_indices", "accepted_position_source", "dst_state_indices",
         "num_accepted_tokens", "dummy_request_mask"),
        False,
    ),
}  # fmt: skip

# Batch sizes covering each of Triton's integer classes (1, divisible by 16,
# other) for launchers whose specialization depends on nothing else.
_INT_CLASS_BATCHES = (1, 2, 16, 17)
_MAX_ENUMERATED_BATCH = 256

# name -> {record key: (launcher, signature, {param: spec})}.
_RECORDS: Dict[str, Dict[tuple, tuple]] = {}
_LOCK = threading.Lock()
_PATCHES: List[Tuple[Any, str, Any]] = []
_RECORDING = False


def _spec(value: Any, batched: bool, same_as: Optional[str]) -> tuple:
    """A batch-size- and alignment-independent description of one launcher
    argument. ``same_as`` names an earlier parameter the call passed the same
    tensor as."""
    if not isinstance(value, torch.Tensor):
        return ("value", value)
    if same_as is not None:
        return ("same", same_as)
    if batched and value.dim() >= 1:
        # Whether a row offset can change the alignment: rows of a multiple
        # of 16 bytes keep the alignment of the buffer's first row.
        row_bytes = value.stride(0) * value.element_size()
        aligned = (value.storage_offset() * value.element_size()) % 16 == 0
        return ("rows", tuple(value.shape[1:]), tuple(value.stride()), value.dtype,
                None if row_bytes % 16 else aligned)  # fmt: skip
    aligned = (value.storage_offset() * value.element_size()) % 16 == 0
    return ("tensor", tuple(value.shape), tuple(value.stride()), value.dtype, aligned)


def _meta(spec: tuple, batch: int, aligned: bool) -> torch.Tensor:
    """Rebuild one tensor argument on ``meta`` (``batch`` rows for a
    per-request one), 16-byte aligned or not."""
    kind = spec[0]
    if kind == "rows":
        _, rest, stride, dtype, _ = spec
        shape = (batch, *rest)
    else:
        _, shape, stride, dtype, _ = spec
    offset = 0 if aligned else 1
    span = 1 + sum((s - 1) * st for s, st in zip(shape, stride) if s > 0)
    base_t = torch.empty(offset + max(span, 1), dtype=dtype, device="meta")
    return base_t.as_strided(tuple(shape), tuple(stride), offset)


def _record(name: str, launcher: Callable, sig: inspect.Signature, args, kwargs) -> None:
    batch_names, _ = _BATCH_ARGS[name]
    bound = sig.bind_partial(*args, **kwargs)
    specs = {}
    seen: Dict[int, str] = {}
    for param, value in bound.arguments.items():
        same_as = None
        if isinstance(value, torch.Tensor):
            if value.is_meta or not value.is_cuda:
                return
            same_as = seen.setdefault(id(value), param)
            same_as = None if same_as == param else same_as
        specs[param] = _spec(value, param in batch_names, same_as)
    key = tuple(sorted(specs.items(), key=lambda kv: kv[0]))
    with _LOCK:
        _RECORDS.setdefault(name, {}).setdefault(key, (launcher, sig, specs))


def _wrap(owner: Any, attr: str, name: str) -> None:
    """Make ``owner.attr`` (the launcher ``name``) record its real calls
    while recording is on."""
    orig = getattr(owner, attr, None)
    if orig is None or getattr(orig, "_jitp_launcher", None) is not None:
        return
    sig = inspect.signature(orig)

    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        if _RECORDING:
            try:
                _record(name, orig, sig, args, kwargs)
            except Exception:  # noqa: BLE001 - recording must never break a launch
                pass
        return orig(*args, **kwargs)

    wrapper._jitp_launcher = orig
    _PATCHES.append((owner, attr, orig))
    setattr(owner, attr, wrapper)


def start_recording(model: Optional[torch.nn.Module] = None) -> None:
    """Record the decode-side launches of the forwards that follow (warmup).

    Patches the names the callers use: the mixer and the cache manager bind
    the launchers at import time, and each ``Mamba2Mixer`` holds its state
    update function as an attribute.
    """
    global _RECORDING
    from ...pyexecutor.kv_cache import mamba_cache_manager as mcm
    from . import causal_conv1d_triton as conv
    from . import mamba2_mixer as mixer
    from . import replay_selective_state_update as replay
    from . import selective_state_update as ssu

    native = ssu.selective_state_update
    _wrap(ssu, "selective_state_update", "selective_state_update")
    _wrap(mixer, "selective_state_update_native", "selective_state_update")
    _wrap(replay, "replay_selective_state_update", "replay_selective_state_update")
    _wrap(mixer, "replay_selective_state_update", "replay_selective_state_update")
    _wrap(conv, "causal_conv1d_update", "causal_conv1d_update")
    _wrap(mixer, "causal_conv1d_update_triton", "causal_conv1d_update")
    _wrap(mcm, "_promote_mamba_state_triton", "_promote_mamba_state_triton")
    if model is not None:
        for mod in model.modules():
            if isinstance(mod, mixer.Mamba2Mixer) and mod.selective_state_update_func is native:
                _wrap(mod, "selective_state_update_func", "selective_state_update")
    _RECORDING = True


def stop_recording() -> None:
    """Stop recording and restore every patched launcher."""
    global _RECORDING
    _RECORDING = False
    while _PATCHES:
        owner, attr, orig = _PATCHES.pop()
        setattr(owner, attr, orig)


_DEVICE_PROPS_LOCK = threading.Lock()
_DEVICE_PROPS_SHIMMED = False


def _shim_meta_device_properties() -> None:
    """Let ``torch.cuda.get_device_properties(meta_tensor.device)`` (the
    replay update's SM count lookup) answer for the current device. Every
    other call behaves exactly as before."""
    global _DEVICE_PROPS_SHIMMED
    with _DEVICE_PROPS_LOCK:
        if _DEVICE_PROPS_SHIMMED:
            return
        orig = torch.cuda.get_device_properties

        @functools.wraps(orig)
        def get_device_properties(device=None):
            if isinstance(device, torch.device) and device.type == "meta":
                device = None
            return orig(device)

        torch.cuda.get_device_properties = get_device_properties
        _DEVICE_PROPS_SHIMMED = True


class MambaDecodeProvider:
    """Replays each recorded decode-side launch for every reachable batch."""

    kernel_modules = (
        "tensorrt_llm._torch.modules.mamba.selective_state_update",
        "tensorrt_llm._torch.modules.mamba.replay_selective_state_update",
        "tensorrt_llm._torch.modules.mamba.causal_conv1d_triton",
        "tensorrt_llm._torch.pyexecutor.kv_cache.mamba_state_kernels",
    )

    # Enumerated again after every warmup (JitPrefetcher.start_background):
    # the KV-cache estimation pass and the final executor warm up on
    # different cache pools, so each records launch classes of its own.
    plans_from_warmup = True

    def __init__(self, max_batch_size: int):
        self.max_batch_size = max(1, int(max_batch_size))
        self._planned: set = set()

    def __bool__(self):
        return bool(_RECORDS)

    def __call__(self, batch_ctx: Any, seen: Optional[set] = None) -> List[KernelCall]:
        # Every variant is planned by the background enumeration.
        if not getattr(batch_ctx, "mamba_decode_all", False):
            return []
        return self.plan_all()

    def enumerate_batches(self) -> Iterator[Any]:
        if self:
            yield SimpleNamespace(mamba_decode_all=True, ctx_chunk_lens=[])

    def _batches(self, every_size: bool) -> List[int]:
        top = min(self.max_batch_size, _MAX_ENUMERATED_BATCH)
        if every_size:
            return list(range(1, top + 1))
        return [b for b in _INT_CLASS_BATCHES if b <= top]

    @staticmethod
    def _alignments(specs: Dict[str, tuple]) -> List[Dict[str, bool]]:
        """Every combination of alignments of the per-request tensors whose
        alignment a row offset can change, with every other tensor at its
        recorded alignment."""
        fixed = {
            p: s[4] for p, s in specs.items() if s[0] in ("rows", "tensor") and s[4] is not None
        }
        free = [p for p, s in specs.items() if s[0] == "rows" and s[4] is None]
        combos = []
        for bits in itertools.product((True, False), repeat=len(free)):
            combos.append({**fixed, **dict(zip(free, bits))})
        return combos

    def plan_all(self) -> List[KernelCall]:
        """Plan every recorded launch class this provider has not planned
        before."""
        from tensorrt_llm.logger import logger

        _shim_meta_device_properties()
        with _LOCK:
            records = [
                (n, r)
                for n, per_name in _RECORDS.items()
                for key, r in per_name.items()
                if (n, key) not in self._planned
            ]
            self._planned.update((n, key) for n, per_name in _RECORDS.items() for key in per_name)
        calls: List[KernelCall] = []
        summary = {}
        for name, (launcher, sig, specs) in records:
            every_size = _BATCH_ARGS[name][1]
            alignments = self._alignments(specs)
            planned = failed = 0
            for batch in self._batches(every_size):
                for aligned in alignments:
                    rec, err = self._replay(launcher, specs, batch, aligned)
                    if err is not None:
                        failed += 1
                        logger.warning_once(
                            f"[JIT prefetch] cannot replay {name}: {err}",
                            key=f"jitp_mamba_decode_{name}",
                        )
                    planned += len(rec)
                    calls.extend(rec)
            done, fails = summary.get(name, (0, 0))
            summary[name] = (done + planned, fails + failed)
        logger.info(
            f"[JIT prefetch] Mamba decode launches planned (calls, failed replays): {summary}"
        )
        return calls

    @staticmethod
    def _replay(launcher, specs: Dict[str, tuple], batch: int, aligned: Dict[str, bool]):
        kwargs = {}
        for p, s in specs.items():
            if s[0] == "value":
                kwargs[p] = s[1]
            elif s[0] == "same":
                kwargs[p] = kwargs[s[1]]
            else:
                kwargs[p] = _meta(s, batch, aligned[p])
        try:
            with shadow_launches() as rec:
                launcher(**kwargs)
        except Exception as e:  # noqa: BLE001 - a replay that cannot run plans nothing
            return [], f"{type(e).__name__}: {e}"
        return list(rec), None


def recorded_launchers() -> Dict[str, int]:
    """Launch classes recorded per launcher name (for logs and tests)."""
    with _LOCK:
        return {name: len(per_name) for name, per_name in _RECORDS.items()}

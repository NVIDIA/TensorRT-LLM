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
"""Soft-pinned store puts, armed in-tree (port of the M3.1 `m31_softpin` host hook).

`TRTLLM_MOONCAKE_STORE_SOFTPIN=1` (default off). Every page a saving role puts is soft-pinned for
`TRTLLM_M31_SOFTPIN_TTL_MS` (default 86400000 = 24 h) through a Mooncake `ReplicateConfig` passed as the 4th
argument of `batch_put_from_multi_buffers`. The master's first eviction pass refills its shortfall in shard order
and so punches holes into prefixes still in use; soft-pinned objects are only taken by the second, age-ordered
pass. With `TRTLLM_M31_SOFTPIN_GROUP_BLOCKS=1` (default off, as in M3.1) every page of a block is put in one
eviction group `<namespace>/<model_key>/w<world>/<block hash>`.

Differences from the hook (peihengh `examples/disaggregated/m31_host_hooks/m31_softpin.py`, f1790970f6/d9db58795d):
- armed explicitly by the worker constructor, not by a `.pth` import hook (the recipes set `PYTHONNOUSERSITE=1`, and
  the hook's md5 pin to its own `worker.py` would never match ours); default off instead of on;
- HOLESTATS is not ported: it wraps `count_prefix_hit`, which our sliding-window `find_prefix_hit` path bypasses.
  Our `mooncake-store ... lookups:` line already reports blocks lost behind the full-attention hit.

Kept from the hook: the store proxy (the image's `put_with_retries` keeps calling the 3-argument put, so the
save-retry receipts keep counting), the fail-safe (until one pinned put has landed a page, a put whose every page
is refused is repeated once unpinned; if that lands anything, pinning is turned off for the process,
`M31_SOFTPIN_DISABLED reason=refused`), and a binding without `ReplicateConfig`/`SoftPinAction` leaving puts
unpinned (`M31_SOFTPIN_UNAVAILABLE`).

Changed from the hook: a refusal for want of space (`CAPACITY_CODES`, Mooncake NO_AVAILABLE_HANDLE = -200) never
arms the fail-safe. Before the first donor segment mounts the pool is empty and every put is refused -200; the
plain repeat of such a put lands only if a segment mounted in between, which says nothing about soft pin. The hook
then unpinned that one rank for good (lanes/mc/mcdrop: 3685966 w0 rank 2, -200 at 11:07:05.3568, first
mount_segment at 11:07:05.3572), so its shard became the master's first-pass eviction victim and, a TP lookup
needing every rank's pages, the worker's store hits fell to zero once the pool filled. A capacity refusal is now
returned as is (the save-retry path counts it) and pinning stays on until a pinned put lands. A real unpin is
logged at ERROR so a run holder can fail on it.

Receipt, once per saving rank: `M31_SOFTPIN_ARMED rank=R role=.. ttl_ms=.. softpin=on|off group_blocks=0|1 src=intree`.

Master hygiene that goes with it (srtctl `mooncake-master` service args, not code; M3.1 979029 values):
`--eviction_ratio=0.01 --eviction_high_watermark_ratio=0.97 --tenant_eviction_high_watermark_ratio=0.97`
(0.99 only with the sf4 allocator master). Soft-pinned objects stay evictable in the second pass only while the
master's `allow_evict_soft_pinned_objects` is true (Mooncake's default). With fewer holes, the RESAVE re-listing
(`TRTLLM_MOONCAKE_STORE_RESAVE_MIN_TOKENS`, 0 = off) can be raised or turned off to unload the save thread.
"""

import os
import sys
import threading
from typing import Callable, Optional, Sequence

from tensorrt_llm.logger import logger

__all__ = [
    "CAPACITY_CODES",
    "SOFTPIN_ENV",
    "SoftPinStore",
    "maybe_softpin_store",
    "replicate_config",
    "softpin_settings",
]

SOFTPIN_ENV = "TRTLLM_MOONCAKE_STORE_SOFTPIN"
TTL_ENV = "TRTLLM_M31_SOFTPIN_TTL_MS"
GROUP_BLOCKS_ENV = "TRTLLM_M31_SOFTPIN_GROUP_BLOCKS"
DEFAULT_TTL_MS = 86400000
#: Put statuses that mean "no room in the pool" (Mooncake ErrorCode::NO_AVAILABLE_HANDLE), not "soft pin refused".
CAPACITY_CODES = frozenset({-200})


def _emit(line: str, error: bool = False) -> None:
    """A receipt on stderr (grep target, as the hook printed it) and in the logger."""
    line = f"{line} pid={os.getpid()}"
    try:
        print(line, file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 - a receipt must never take the server down
        pass
    (logger.error if error else logger.warning)(line)


def softpin_settings(environ=None):
    """(enabled, ttl_ms, group_blocks) from the environment; None when the switch is off."""
    env = os.environ if environ is None else environ
    if env.get(SOFTPIN_ENV, "0") != "1":
        return None
    ttl = int(env.get(TTL_ENV, str(DEFAULT_TTL_MS)))
    if ttl <= 0:
        raise ValueError(f"{TTL_ENV} must be > 0, got {ttl}")
    return True, ttl, env.get(GROUP_BLOCKS_ENV, "0") == "1"


def replicate_config(ttl_ms: int, group_ids: Optional[Sequence[str]] = None, _mod=None):
    """A `mooncake.store.ReplicateConfig` soft-pinning for `ttl_ms`.

    Raises:
        ImportError: The binding lacks the soft-pin fields.
    """
    try:
        if _mod is None:
            import mooncake.store as _mod  # noqa: PLC0415
        config = _mod.ReplicateConfig()
        config.soft_pin_action = _mod.SoftPinAction.ENABLE
        config.soft_pin_ttl_ms = int(ttl_ms)
        if group_ids is not None:
            config.group_ids = list(group_ids)
    except (ImportError, AttributeError, TypeError) as exc:
        raise ImportError(
            "soft pin needs mooncake.store ReplicateConfig.soft_pin_action/soft_pin_ttl_ms and "
            f"SoftPinAction: {type(exc).__name__}: {exc}"
        ) from exc
    return config


def _ok(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


class SoftPinStore:
    """Passes every call through to the store handle, except that a three-argument
    `batch_put_from_multi_buffers` gets the soft-pin config appended."""

    #: Public attributes kept on the proxy. Any other attribute set goes to the store handle, and the pybind
    #: `MooncakeDistributedStore` raises AttributeError for a name it does not define, so every counter or flag
    #: this class adds must be listed here (private `_` names always stay local).
    _LOCAL_ATTRS = frozenset(
        {"pin", "calls", "keys", "pinned_keys", "failed_keys", "capacity_refusals"}
    )

    def __init__(
        self,
        inner,
        rank: int,
        ttl_ms: int,
        group_of: Optional[Callable[[str], str]] = None,
        config_factory: Callable = replicate_config,
    ):
        object.__setattr__(self, "_inner", inner)
        self._rank = rank
        self._ttl_ms = ttl_ms
        self._factory = config_factory
        self._group_of = group_of
        self._config = None
        self._lock = threading.Lock()
        self.pin = False
        self._confirmed = False
        self.calls = self.keys = self.pinned_keys = self.failed_keys = 0
        self.capacity_refusals = 0
        try:
            # A grouped probe too, so a binding without group_ids fails here, not mid-run.
            self._config = config_factory(ttl_ms, ["probe"] if group_of is not None else None)
            self.pin = True
        except ImportError as exc:
            _emit(f"M31_SOFTPIN_UNAVAILABLE rank={rank} error={exc}")

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def __setattr__(self, key, value):
        if key.startswith("_") or key in self._LOCAL_ATTRS:
            object.__setattr__(self, key, value)
        else:
            setattr(self._inner, key, value)

    def batch_put_from_multi_buffers(self, keys, addresses, sizes, *rest, **kwargs):
        if rest or kwargs or not self.pin:
            results = self._inner.batch_put_from_multi_buffers(
                keys, addresses, sizes, *rest, **kwargs
            )
            self._count(keys, results, pinned=False)
            return results
        if self._group_of is not None:
            config = self._factory(self._ttl_ms, [self._group_of(key) for key in keys])
        else:
            config = self._config
        try:
            results = self._inner.batch_put_from_multi_buffers(keys, addresses, sizes, config)
        except TypeError as exc:
            # The binding takes no config argument: unpin for good and put plainly.
            self.pin = False
            _emit(f"M31_SOFTPIN_DISABLED rank={self._rank} reason=typeerror error={exc}")
            results = self._inner.batch_put_from_multi_buffers(keys, addresses, sizes)
            self._count(keys, results, pinned=False)
            return results
        landed = isinstance(results, (list, tuple)) and any(_ok(r) for r in results)
        if not self._confirmed and keys and not landed:
            codes = (
                sorted({r for r in results if isinstance(r, int) and r < 0})
                if isinstance(results, (list, tuple))
                else results
            )
            if isinstance(codes, list) and codes and set(codes) <= CAPACITY_CODES:
                # The pool had no room (e.g. no donor segment mounted yet): not a verdict on soft pin. A plain
                # repeat could land only because a segment mounted in between, which would unpin this rank alone.
                with self._lock:
                    self.capacity_refusals += 1
                self._count(keys, results, pinned=True)
                return results
            plain = self._inner.batch_put_from_multi_buffers(keys, addresses, sizes)
            if isinstance(plain, (list, tuple)) and any(_ok(r) for r in plain):
                self.pin = False
                _emit(
                    f"M31_SOFTPIN_DISABLED rank={self._rank} reason=refused ttl_ms={self._ttl_ms} "
                    f"pinned_codes={codes}",
                    error=True,
                )
                self._count(keys, plain, pinned=False)
                return plain
            self._count(keys, results, pinned=True)
            return results
        if landed:
            self._confirmed = True
        self._count(keys, results, pinned=True)
        return results

    def _count(self, keys, results, pinned: bool) -> None:
        n = len(keys)
        if isinstance(results, (list, tuple)) and len(results) == n:
            ok = sum(1 for r in results if _ok(r))
        else:
            ok = 0
        with self._lock:
            self.calls += 1
            self.keys += n
            self.failed_keys += n - ok
            if pinned:
                self.pinned_keys += ok

    def put_summary(self) -> str:
        return (
            f"softpin calls={self.calls} keys={self.keys} pinned_keys={self.pinned_keys} "
            f"failed_keys={self.failed_keys} capacity_refusals={self.capacity_refusals} "
            f"softpin={'on' if self.pin else 'off'}"
        )


def _group_from_key(namespace: str, model_key: str, world_size: int) -> Callable[[str], str]:
    """Key = <namespace prefix>/<block hash hex>; group = <namespace>/<model_key>/w<world>/<block hash>."""
    prefix = f"{namespace}/{model_key}/w{world_size}/"

    def group_of(key: str) -> str:
        return prefix + key.rsplit("/", 1)[-1]

    return group_of


def maybe_softpin_store(
    store,
    *,
    rank: int,
    role: str,
    saves: bool,
    namespace: str,
    model_key: str,
    world_size: int,
    config_factory: Optional[Callable] = None,
    environ=None,
):
    """`store` wrapped in a `SoftPinStore` when the switch is on and this rank saves, else `store` unchanged."""
    settings = softpin_settings(environ)
    if settings is None or not saves:
        return store
    _, ttl, group = settings
    wrapped = SoftPinStore(
        store,
        rank,
        ttl,
        _group_from_key(namespace, model_key, world_size) if group else None,
        config_factory or replicate_config,
    )
    _emit(
        f"M31_SOFTPIN_ARMED rank={rank} role={role} ttl_ms={ttl} softpin={'on' if wrapped.pin else 'off'} "
        f"group_blocks={int(group)} src=intree"
    )
    return wrapped

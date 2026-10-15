# Model-owned disaggregated KV transfer planning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put every "which bytes pair with which" decision of the transfer v2 sender/receiver behind a `TransferPlanner` interface that a model class can override, with the current shared logic becoming `DefaultTransferPlanner`, and `native/transfer.py` no longer knowing about attention, Mamba, `CacheKind` or `MapperKind`.

**Architecture:** A new `TransferPlanner` ABC (`base/planner.py`) with `check_peer / targets / plan / receiver_payload_bytes / describe`. `DefaultTransferPlanner` (`native/default_planner.py`) receives the policy, mapper, overlap and pool-matching engine that lives in `PeerRegistrar` today plus the body of `Sender._build_kv_write_meta`, `Receiver._fanin_bounce_safe`, `Receiver._get_mamba_slot`. `PeerRegistrar` keeps only framework bookkeeping (peer `RankInfo` cache, extractors, aux layout cache, aux ownership) and holds the planner. `Sender`/`Receiver` call the planner through `self._planner`. The planner is produced by a factory the model class exposes (`build_transfer_planner`), plumbed `_util.py -> create_kv_cache_transceiver -> KvCacheTransceiverV2 -> TransferWorkerConfig`. Zero behavior change: every existing transfer plan is bit-identical.

**Tech Stack:** Python 3.12, numpy, msgpack (RankInfo wire), pytest. No GPU for the unit tasks; the final task runs the GPU disagg gate on a GB300 node.

**Spec:** `docs/superpowers/specs/2026-10-10-disagg-transfer-planner-design.md`

**Deviations from the spec (decided while planning, all in the direction of less churn):**
1. The framework bookkeeping class keeps its name `PeerRegistrar` (spec: `PeerRegistry`). ~10 test files mock `sender._registrar` / `receiver._registrar`; renaming buys nothing.
2. `RankInfo.attention` stays where it is and `DefaultTransferPlanner.describe()` returns `b""`. The spec moved `AttentionInfo` into the blob for cleanliness; that touches both policies, `tp_size_per_dp_group` and every RankInfo-building test for no functional gain. A custom planner does not need `attention`; it can leave it `None` and use `planner_blob`. Revisit only if a real planner needs it.
3. The model hook is a factory with the inputs the worker actually has: `build_transfer_planner(kv_cache_manager, self_rank_info, self_extractor) -> TransferPlanner`, not `build_transfer_planner(kv_cache_manager)`. `RankInfo` and the extractor are built by `TransferWorker` from the cache manager; the default planner consumes them, and duplicating that derivation in every planner would be wrong.
4. `PeerOverlap` moves with the engine into `default_planner.py` (it is the default planner's internal overlap record). Tests import it from there.
5. `TransferPlanner.forget_peer(peer_ri)` is added (default no-op) so `PeerRegistrar.unregister` can drop planner-side caches. It is cache hygiene the current `unregister` already does for `_kv_map_cache`/`_lg_pool_mapping_cache`.
6. The receiver's first-contact gate now runs the planner's full `check_peer` (attention
   geometry, both layout validators, layer-count and NHD warnings), where before it ran only
   the two layout validators and left geometry mismatches to the sender's listener thread,
   whose logged error left the generation request hanging until its timeout. A mismatched
   pair now fails fast on the generation side with a cached diagnostic. This is a deliberate
   behavior improvement, recorded here because Phase 1 is otherwise zero-change.

## Global Constraints

- NVIDIA SPDX copyright header (2026) on every new file; bump the year on modified files whose header is older.
- `git commit -s` (DCO); no AI attribution, no co-authors.
- Zero behavior change in Phase 1: every existing unit test under `tests/unittest/disaggregated/` keeps passing with only construction/import changes; the GPU gate `TestModelingV2GptOss120bSm103Tp1::test_gsm8k_disagg` and `TestGPTOSS::test_kv_cache_v2_nixl_python` must pass.
- After the refactor `native/transfer.py` contains none of the strings `mixers`, `CacheKind`, `MapperKind`, `MambaPolicy`, `AttentionPolicy` (grep gate, enforced by a test).
- Both peers must run the same planner class: `planner_id` mismatch -> `PeerIncompatibleError`, failing that request/peer, never the worker.
- Python-only change. Run unit tests inside the container with `PYTHONPATH=<worktree>` and main's venv first on `PATH` (see `/scratch/.../users/fredw/mv2-disagg/run_disagg_test.sh`); `srun --mpi=none` for anything that spawns MPI.

## Review Focus

1. **Empty plan from a non-owning rank.** A rank whose `plan()` returns three empty arrays (duplicate-head non-owner, PP stage with no overlap) must still produce a `WriteMeta` and complete the handshake, exactly like today's `_build_kv_write_meta` with no fragments. Pinned in Task 4 (`test_build_kv_write_meta_empty_plan_still_delivers`).
2. **`receiver_payload_bytes` returning `None`.** Receiver must disable bounce (not crash, not reserve a wrongly sized region) when the planner cannot size the extra payload. Pinned in Task 5 (`test_dispatch_task_payload_none_disables_bounce`).
3. **Gen-first ADP broadcast union.** With `ctx_dp_rank is None` the receiver unions `targets()` over all peer DP ranks and must set `uniform_writers=False` on the union so fan-in bounce is never attempted. Pinned by the existing ownership regression test that covers broadcast (adapted in Task 5) plus an assertion added there.
4. **planner_id mismatch on the receiver's first contact.** `_get_sender_info` must raise `PeerIncompatibleError` before `REGISTER_RANK_INFO` is sent, and cache the diagnosis. Pinned in Task 6 (`test_receiver_rejects_planner_id_mismatch`).
5. **Old peer wire format (no `planner_id`).** A `RankInfo` serialized without the two new fields must still deserialize (defaults `""`/`b""`) so a mixed-version pair fails with the explicit mismatch error rather than a msgpack `TypeError`. Pinned in Task 2 (`test_rank_info_from_bytes_without_planner_fields`).

---

### Task 1: `TransferPlanner` protocol module

**Files:**
- Create: `tensorrt_llm/_torch/disaggregation/base/planner.py`
- Modify: `tensorrt_llm/_torch/disaggregation/native/transfer.py:195-203` (replace the `PeerIncompatibleError` class body with an import)
- Test: `tests/unittest/disaggregated/test_transfer_planner.py` (new)

**Interfaces:**
- Produces:
  ```python
  class PeerIncompatibleError(ValueError)
  @dataclass class PeerTargets: ranks: list[int]; uniform_writers: bool
  @dataclass class TransferPlan: src_ptrs: np.ndarray; dst_ptrs: np.ndarray; sizes: np.ndarray
      @classmethod empty() -> TransferPlan
  class TransferPlanner(ABC):
      planner_id: str  (class attribute, must be overridden)
      def describe(self) -> bytes
      def check_peer(self, self_ri, peer_ri) -> None   # raises PeerIncompatibleError
      def targets(self, self_ri, peer_ri, peer_dp_rank: int) -> PeerTargets
      def plan(self, self_ri, peer_ri, src_block_ids_per_lg, dst_block_ids_per_lg,
               token_range, prompt_len, dst_start_token) -> TransferPlan
      def receiver_payload_bytes(self, self_ri, peer_ri, dst_block_ids_per_lg) -> Optional[int]
      def forget_peer(self, peer_ri) -> None   # default no-op
  ```

- [ ] **Step 1: Write the failing test**

```python
# tests/unittest/disaggregated/test_transfer_planner.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the model-owned transfer planner seam."""

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.base.planner import (
    PeerIncompatibleError,
    PeerTargets,
    TransferPlan,
    TransferPlanner,
)


def test_transfer_plan_empty_is_three_int64_arrays():
    plan = TransferPlan.empty()
    for arr in (plan.src_ptrs, plan.dst_ptrs, plan.sizes):
        assert arr.dtype == np.int64 and arr.size == 0


def test_transfer_planner_is_abstract():
    with pytest.raises(TypeError):
        TransferPlanner()  # type: ignore[abstract]


def test_peer_incompatible_error_is_value_error():
    assert issubclass(PeerIncompatibleError, ValueError)
    assert PeerTargets(ranks=[0], uniform_writers=True).ranks == [0]
```

- [ ] **Step 2: Run test to verify it fails**

Run (inside container, from `<worktree>/tests/unittest`): `python -m pytest disaggregated/test_transfer_planner.py -q`
Expected: FAIL with `ModuleNotFoundError: tensorrt_llm._torch.disaggregation.base.planner`

- [ ] **Step 3: Write the module**

```python
# tensorrt_llm/_torch/disaggregation/base/planner.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The seam between the transfer framework and a model's KV-transfer plan.

The framework (``native/transfer.py``) moves bytes and tracks transfers. Which
bytes of this rank pair with which bytes of a peer rank is a property of the
model's cache layout and the two parallel topologies, so a model class may
supply its own :class:`TransferPlanner`; ``native/default_planner.py`` is the
one every model gets unless it overrides ``build_transfer_planner``.

Both sides of a transfer must run the same planner class (``planner_id``):
the framework enforces that and nothing else about the plan.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from tensorrt_llm._torch.disaggregation.base.backend import TokenRange
    from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo


class PeerIncompatibleError(ValueError):
    """A peer cannot exchange KV with this rank. Fails the peer, never the worker."""


@dataclass
class PeerTargets:
    """Which peer ranks this rank exchanges with for one peer DP group."""

    ranks: List[int]
    # Every writer contributes an equal share of the receiver's region, so the
    # receiver may split one fan-in bounce reservation ``total // len(ranks)``.
    uniform_writers: bool = False


@dataclass
class TransferPlan:
    """Device addresses to copy, pairwise. ``sizes[i]`` bytes from ``src_ptrs[i]`` to ``dst_ptrs[i]``.

    An empty plan is valid: this rank writes nothing to this peer but still
    completes the handshake (a duplicate-head non-owner, a PP stage with no
    overlapping layers).
    """

    src_ptrs: np.ndarray
    dst_ptrs: np.ndarray
    sizes: np.ndarray

    @classmethod
    def empty(cls) -> "TransferPlan":
        e = np.array([], dtype=np.int64)
        return cls(src_ptrs=e, dst_ptrs=e.copy(), sizes=e.copy())


class TransferPlanner(ABC):
    """Model-owned KV transfer planning. One instance per rank, shared by Sender and Receiver.

    Threading is the framework's: ``plan`` runs where the sender builds write
    metadata, ``targets``/``receiver_payload_bytes``/``check_peer`` run on the
    receiver's dispatch path and both listeners. Keep per-peer caches keyed by
    ``(peer_ri.instance_name, peer_ri.instance_rank)`` and drop them in
    ``forget_peer``.
    """

    #: Both sides must agree. The framework compares it before ``check_peer``.
    planner_id: str = ""

    def describe(self) -> bytes:
        """Opaque self-description carried in ``RankInfo.planner_blob``; only this class decodes it."""
        return b""

    @abstractmethod
    def check_peer(self, self_ri: "RankInfo", peer_ri: "RankInfo") -> None:
        """Raise :class:`PeerIncompatibleError` with a field-level reason, else return."""

    @abstractmethod
    def targets(self, self_ri: "RankInfo", peer_ri: "RankInfo", peer_dp_rank: int) -> PeerTargets:
        """Peer flat ranks this rank exchanges with for the peer's DP group ``peer_dp_rank``."""

    @abstractmethod
    def plan(
        self,
        self_ri: "RankInfo",
        peer_ri: "RankInfo",
        src_block_ids_per_lg: List[np.ndarray],
        dst_block_ids_per_lg: List[np.ndarray],
        token_range: "TokenRange",
        prompt_len: int,
        dst_start_token: Optional[int],
    ) -> TransferPlan:
        """Sender side: the bytes this rank writes into ``peer_ri`` for one chunk."""

    @abstractmethod
    def receiver_payload_bytes(
        self, self_ri: "RankInfo", peer_ri: "RankInfo", dst_block_ids_per_lg: List[np.ndarray]
    ) -> Optional[int]:
        """Receiver side: bytes beyond the paged blocks that ride the same write.

        ``None`` means the planner cannot size them; the receiver then disables
        its bounce reservation for this request.
        """

    def forget_peer(self, peer_ri: "RankInfo") -> None:
        """Drop any per-peer cache. Called by ``PeerRegistrar.unregister``."""
        return None
```

In `native/transfer.py` replace the class

```python
class PeerIncompatibleError(ValueError):
    ...
```

with

```python
from tensorrt_llm._torch.disaggregation.base.planner import PeerIncompatibleError  # noqa: F401  (re-exported)
```

placed with the other imports (keep the name importable from `transfer` because tests and `transceiver.py` import it from there — check with `grep -rn "PeerIncompatibleError" tensorrt_llm tests`).

- [ ] **Step 4: Run tests**

Run: `python -m pytest disaggregated/test_transfer_planner.py -q` → PASS (3). Then `python -c "from tensorrt_llm._torch.disaggregation.native.transfer import PeerIncompatibleError"` → ok.

- [ ] **Step 5: Commit**

```bash
git add tensorrt_llm/_torch/disaggregation/base/planner.py tensorrt_llm/_torch/disaggregation/native/transfer.py tests/unittest/disaggregated/test_transfer_planner.py
git commit -s -m "[None][refactor] Add the TransferPlanner seam for disaggregated KV transfer"
```

---

### Task 2: `RankInfo` carries `planner_id` / `planner_blob`

**Files:**
- Modify: `tensorrt_llm/_torch/disaggregation/native/rank_info.py` (dataclass fields after `device_id`)
- Test: `tests/unittest/disaggregated/test_rank_info.py` (append)

**Interfaces:**
- Produces: `RankInfo.planner_id: str = ""`, `RankInfo.planner_blob: bytes = b""`; round-trips through `to_bytes`/`from_bytes`; absent on the wire -> defaults.

- [ ] **Step 1: Write the failing tests** (append to `test_rank_info.py`; reuse that file's existing RankInfo builder helper if there is one, else build a minimal `RankInfo` as below)

```python
def _minimal_rank_info(**kw):
    from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
    base = dict(
        instance_name="ctx", instance_rank=0, tp_size=1, tp_rank=0, pp_size=1, pp_rank=0,
        layer_num_per_pp=[2], sender_endpoints=[], self_endpoint="", transfer_engine_info=b"",
    )
    base.update(kw)
    return RankInfo(**base)


def test_rank_info_planner_fields_round_trip():
    from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
    ri = _minimal_rank_info(planner_id="toy", planner_blob=b"\x00\x01")
    back = RankInfo.from_bytes(ri.to_bytes())
    assert back.planner_id == "toy" and back.planner_blob == b"\x00\x01"


def test_rank_info_from_bytes_without_planner_fields():
    import msgpack
    from dataclasses import asdict
    from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
    data = asdict(_minimal_rank_info())
    for k in ("planner_id", "planner_blob", "attention", "aux_meta", "page_table"):
        data.pop(k, None)
    back = RankInfo.from_bytes(msgpack.packb(data))
    assert back.planner_id == "" and back.planner_blob == b""
```

- [ ] **Step 2: Run** `python -m pytest disaggregated/test_rank_info.py -q -k planner` → FAIL (`unexpected keyword argument 'planner_id'`).

- [ ] **Step 3: Implement** — in the `RankInfo` dataclass add after `device_id: int = 0`:

```python
    # Which TransferPlanner class produced this rank's plan, and that planner's
    # opaque self-description. The framework compares planner_id across peers;
    # only the planner decodes planner_blob.
    planner_id: str = ""
    planner_blob: bytes = b""
```

`to_bytes` (uses `asdict`) and `from_bytes` (`cls(**unpacked)`) need no change; msgpack encodes `bytes` natively.

- [ ] **Step 4: Run** `python -m pytest disaggregated/test_rank_info.py -q` → PASS.

- [ ] **Step 5: Commit** `git commit -s -m "[None][refactor] Carry planner_id and planner_blob on RankInfo"`

---

### Task 3: `DefaultTransferPlanner` owns the engine; `PeerRegistrar` keeps bookkeeping

**Files:**
- Create: `tensorrt_llm/_torch/disaggregation/native/default_planner.py`
- Modify: `tensorrt_llm/_torch/disaggregation/native/peer.py` (becomes ~120 lines)
- Modify tests (construction only): `tests/unittest/disaggregated/test_peer.py`, `test_pool_matching.py`, `test_extractor.py:350-392`, `test_kda_mamba_transfer.py:62,289`, `test_bounce.py:229-262`, `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py:33,1945`

**Interfaces:**
- Produces:
  ```python
  # native/default_planner.py
  LGPoolKey = Tuple[int, int]
  @dataclass class PeerOverlap  (unchanged fields)
  class DefaultTransferPlanner(TransferPlanner):
      planner_id = "trtllm.default"
      def __init__(self, self_rank_info: RankInfo, self_extractor: KVRegionExtractorV1)
      # engine (moved verbatim from PeerRegistrar):
      get_pool_mapping(peer_ri) -> Dict[LGPoolKey, LGPoolKey]
      get_kv_map(peer_ri, self_pool_key, peer_pool_key) -> RegionMapperBase
      get_peer_overlap(peer_ri, peer_dp_rank) -> PeerOverlap
      should_send_pool(peer_overlap, peer_ri, layer_group_id, pool_idx) -> bool
      _owns_tp_fan_in, _find_overlap, _get_buffers_per_layer, _get_policy, _POLICY_CLASSES
      peer_extractor(peer_ri) -> KVRegionExtractorV1   # planner-private cache
      # moved from transfer.py:
      @staticmethod _pair_ordinals(src_block_ids, dst_block_ids, dst_start_block=0)
      @staticmethod _fanin_bounce_safe(overlap, peer_ri, receiver_page_table) -> bool
      @staticmethod _get_mamba_slot(page_table, dst_block_ids_per_lg) -> Optional[int]
      # protocol:
      check_peer, targets, plan, receiver_payload_bytes, forget_peer
  # native/peer.py
  class PeerRegistrar:
      def __init__(self, self_rank_info, self_extractor, planner: TransferPlanner)
      planner (property), self_rank_info, self_extractor, register, unregister,
      get_peer_rank_info, peer_extractor(peer_name, peer_rank), get_aux_transfer_layout,
      cache_aux_transfer_layout, should_send_aux, _unique_key
  ```
- Consumes: Task 1 types.

- [ ] **Step 1: Create `default_planner.py` by moving code.** Start the file with the NVIDIA header and these imports (superset of what moves):

```python
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.base.backend import TokenRange
from tensorrt_llm._torch.disaggregation.base.planner import (
    PeerIncompatibleError,
    PeerTargets,
    TransferPlan,
    TransferPlanner,
)
from tensorrt_llm._torch.disaggregation.base.region import RegionMapperBase
from tensorrt_llm._torch.disaggregation.native.mixers.attention.peer import AttentionPolicy
from tensorrt_llm._torch.disaggregation.native.mixers.ssm.peer import (
    MambaPolicy,
    mamba_receiver_payload_bytes,
)
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.page import (
    CacheKind,
    KVCachePageTable,
    MapperKind,
    PoolView,
)
from tensorrt_llm._torch.disaggregation.resource.utils import (
    get_layer_byte_ranges,
    get_layer_to_layer_group,
    get_pool_view_global_layer_ids,
)
```

Then move, **unchanged in body**, from `peer.py`: `LGPoolKey`, `PeerOverlap`, and from `PeerRegistrar`: `_POLICY_CLASSES`, `get_pool_mapping`, `_get_policy`, `get_kv_map`, `_get_buffers_per_layer`, `_find_overlap`, `get_peer_overlap`, `_owns_tp_fan_in`, `should_send_pool`. Rename every `"PeerRegistrar.` prefix inside error strings to `"DefaultTransferPlanner.`. The class header and constructor:

```python
class DefaultTransferPlanner(TransferPlanner):
    """The planner every model gets unless it overrides ``build_transfer_planner``.

    Attention (PAGED) and recurrent-state (STATE) layer groups are planned by
    ``AttentionPolicy`` and ``MambaPolicy``; pools are matched by role and
    layer overlap; TP/PP/CP overlap decides which peer ranks receive what.
    """

    planner_id = "trtllm.default"

    _POLICY_CLASSES = {
        CacheKind.PAGED: AttentionPolicy,
        CacheKind.STATE: MambaPolicy,
    }

    def __init__(self, self_rank_info: RankInfo, self_extractor: KVRegionExtractorV1):
        self._ri = self_rank_info
        self._self_ext_cache = self_extractor
        self._policies: Dict[CacheKind, Union[AttentionPolicy, MambaPolicy]] = {}
        self._kv_map_cache: Dict[tuple, RegionMapperBase] = {}
        self._overlap_cache: Dict[str, PeerOverlap] = {}
        self._lg_pool_mapping_cache: Dict[str, Dict[LGPoolKey, LGPoolKey]] = {}
        self._peer_ext_cache: Dict[str, KVRegionExtractorV1] = {}

    @staticmethod
    def _unique_key(name: str, rank: int) -> str:
        return name + str(rank)

    def peer_extractor(self, peer_ri: RankInfo) -> KVRegionExtractorV1:
        key = self._unique_key(peer_ri.instance_name, peer_ri.instance_rank)
        ext = self._peer_ext_cache.get(key)
        if ext is None:
            ext = KVRegionExtractorV1(peer_ri.page_table)
            self._peer_ext_cache[key] = ext
        return ext

    def forget_peer(self, peer_ri: RankInfo) -> None:
        key = self._unique_key(peer_ri.instance_name, peer_ri.instance_rank)
        self._peer_ext_cache.pop(key, None)
        self._lg_pool_mapping_cache.pop(key, None)
        for k in [k for k in self._kv_map_cache if k[0] == key]:
            del self._kv_map_cache[k]
```

(`get_peer_overlap` caches under `_unique_key(instance_name, peer_dp_rank)`; leave that as is — it is per instance, not per rank.)

- [ ] **Step 2: Add the moved statics.** Move `Sender._pair_ordinals` (transfer.py:1683-1708) verbatim as `DefaultTransferPlanner._pair_ordinals`. Move `Receiver._fanin_bounce_safe` (3134-3170) verbatim as a `@staticmethod` with the same signature. Replace `Receiver._get_mamba_slot` with:

```python
    @staticmethod
    def _get_mamba_slot(
        page_table: Optional[KVCachePageTable], dst_block_ids_per_lg: List[np.ndarray]
    ) -> Optional[int]:
        """The receiver slot of the first STATE layer group, or None when there is none."""
        if page_table is None:
            return None
        for lg_idx, lg in enumerate(page_table.layer_groups):
            if lg.kind == CacheKind.STATE:
                block_ids = dst_block_ids_per_lg[lg_idx]
                if block_ids.size > 0:
                    return int(block_ids[0])
        return None
```

- [ ] **Step 3: Implement the protocol methods.**

```python
    # ---- TransferPlanner protocol -------------------------------------------------

    def check_peer(self, self_ri: RankInfo, peer_ri: RankInfo) -> None:
        if not self._get_policy(CacheKind.PAGED).check_peer_compatible(peer_ri):
            raise PeerIncompatibleError(
                "DefaultTransferPlanner.check_peer: attention geometry differs from the peer"
            )
        self_page_table = self._self_ext_cache.page_table
        try:
            MambaPolicy.validate_peer_compatible(
                self._ri, peer_ri, self_page_table, peer_ri.page_table
            )
            AttentionPolicy.validate_peer_compatible(self_page_table, peer_ri.page_table)
        except ValueError as e:
            raise PeerIncompatibleError(str(e)) from e

        self_layers = sum(self._ri.layer_num_per_pp)
        peer_layers = sum(peer_ri.layer_num_per_pp)
        if self_layers != peer_layers:
            # One side may hold speculative (MTP) layers the other lacks; pool
            # matching transfers only the layers both hold.
            logger.warning(
                "DefaultTransferPlanner: layer count differs "
                f"(local={self_layers}, peer={peer_layers}), allowing partial layer transfer."
            )
        self._warn_nhd_head_mismatch(peer_ri)

    def _warn_nhd_head_mismatch(self, peer_ri: RankInfo) -> None:
        # body = the NHD warning block of today's PeerRegistrar.register (peer.py:97-112),
        # using self._self_ext_cache.page_table and self.get_pool_mapping(peer_ri)
        ...

    def targets(self, self_ri: RankInfo, peer_ri: RankInfo, peer_dp_rank: int) -> PeerTargets:
        overlap = self.get_peer_overlap(peer_ri, peer_dp_rank)
        uniform = (
            self_ri.cp_size == 1
            and peer_ri.cp_size == 1
            and self._fanin_bounce_safe(overlap, peer_ri, self._self_ext_cache.page_table)
        )
        return PeerTargets(ranks=list(overlap.ranks), uniform_writers=uniform)

    def plan(
        self,
        self_ri: RankInfo,
        peer_ri: RankInfo,
        src_block_ids_per_lg: List[np.ndarray],
        dst_block_ids_per_lg: List[np.ndarray],
        token_range: TokenRange,
        prompt_len: int,
        dst_start_token: Optional[int],
    ) -> TransferPlan:
        is_partial = token_range.start > 0 or token_range.end < prompt_len
        if is_partial and (self_ri.cp_size > 1 or peer_ri.cp_size > 1):
            raise ValueError(
                "enable_pipelined_transfer is not supported with context parallelism "
                f"(sender cp_size={self_ri.cp_size}, receiver cp_size={peer_ri.cp_size})"
            )
        targets = self.get_peer_overlap(peer_ri, peer_ri.dp_rank)
        extractor = self._self_ext_cache
        peer_extractor = self.peer_extractor(peer_ri)
        pool_mapping = self.get_pool_mapping(peer_ri)

        src_frag_parts: list[np.ndarray] = []
        dst_frag_parts: list[np.ndarray] = []
        size_specs: list[tuple[int, int]] = []
        # <<< paste the per-pool loop of Sender._build_kv_write_meta (transfer.py:1750-1800)
        #     verbatim, with these renames:
        #       self._registrar.should_send_pool(...)  -> self.should_send_pool(...)
        #       self._registrar.self_rank_info          -> self_ri
        #       req_info.dst_start_token                -> dst_start_token
        #       src_block_ids_per_groups / dst_block_ids_per_groups -> src_block_ids_per_lg / dst_block_ids_per_lg
        #       Sender._pair_ordinals                   -> self._pair_ordinals
        #       self._registrar.get_kv_map(...)         -> self.get_kv_map(...)
        if not src_frag_parts:
            return TransferPlan.empty()
        counts, values = zip(*size_specs)
        return TransferPlan(
            src_ptrs=np.concatenate(src_frag_parts),
            dst_ptrs=np.concatenate(dst_frag_parts),
            sizes=np.repeat(np.array(values, dtype=np.int64), counts),
        )

    def receiver_payload_bytes(
        self, self_ri: RankInfo, peer_ri: RankInfo, dst_block_ids_per_lg: List[np.ndarray]
    ) -> Optional[int]:
        receiver_pt = self._self_ext_cache.page_table
        dst_slot = self._get_mamba_slot(receiver_pt, dst_block_ids_per_lg)
        if dst_slot is None:
            return 0
        if peer_ri.page_table is None:
            return None  # cannot size the recurrent-state payload
        return mamba_receiver_payload_bytes(
            sender_page_table=peer_ri.page_table,
            receiver_page_table=receiver_pt,
            dst_slot=dst_slot,
        )
```

Note on `plan` vs today: today `_build_kv_write_meta` also records perf-timer marks and `expected_transfers`; those stay in the framework (Task 4).

- [ ] **Step 4: Shrink `peer.py` to the bookkeeping class.**

```python
class PeerRegistrar:
    """Framework-side peer bookkeeping: who the peers are, their page-table extractors,
    the aux transfer layouts, and aux ownership. Planning is the planner's."""

    def __init__(
        self,
        self_rank_info: RankInfo,
        self_extractor: KVRegionExtractorV1,
        planner: TransferPlanner,
    ):
        self._ri = self_rank_info
        self._self_ext_cache = self_extractor
        self._planner = planner
        self._peer_ri_cache: Dict[str, RankInfo] = {}
        self._peer_ext_cache: Dict[str, KVRegionExtractorV1] = {}
        self._aux_transfer_layout_cache: Dict[str, AuxTransferLayout] = {}

    @property
    def planner(self) -> TransferPlanner:
        return self._planner

    def register(self, peer_name: str, peer_rank: int, peer_ri: RankInfo):
        if peer_ri.planner_id != self._planner.planner_id:
            raise PeerIncompatibleError(
                f"PeerRegistrar.register: peer {peer_name} (rank={peer_rank}) plans transfers "
                f"with '{peer_ri.planner_id}', this rank with '{self._planner.planner_id}'; "
                "both sides of a disaggregated deployment must run the same model planner"
            )
        self._planner.check_peer(self._ri, peer_ri)
        key = self._unique_key(peer_name, peer_rank)
        self._peer_ri_cache[key] = peer_ri
        self._aux_transfer_layout_cache.pop(key, None)
        self._peer_ext_cache[key] = KVRegionExtractorV1(peer_ri.page_table)

    def unregister(self, peer_name: str, peer_rank: int):
        key = self._unique_key(peer_name, peer_rank)
        peer_ri = self._peer_ri_cache.pop(key, None)
        self._peer_ext_cache.pop(key, None)
        self._aux_transfer_layout_cache.pop(key, None)
        if peer_ri is not None:
            self._planner.forget_peer(peer_ri)
```

Keep verbatim: `peer_extractor`, `self_extractor`, `get_peer_rank_info`, `get_aux_transfer_layout`, `cache_aux_transfer_layout`, `self_rank_info`, `_unique_key`, `should_send_aux`. Delete everything else (it moved). Imports left: `Dict`, `AuxTransferLayout`, `PeerIncompatibleError`, `TransferPlanner`, `RankInfo`, `KVRegionExtractorV1`. The old `register` raised `ValueError("... is incompatible with local rank.")` when `_check_peer_compatible` returned False; `check_peer` now raises `PeerIncompatibleError` (a `ValueError`) — callers that catch `ValueError` keep working.

- [ ] **Step 5: Adapt the direct tests (construction only).**
  - `test_peer.py:38`: `from tensorrt_llm._torch.disaggregation.native.default_planner import PeerOverlap, DefaultTransferPlanner`; replace `PeerRegistrar(` with `DefaultTransferPlanner(` and `PeerRegistrar._get_buffers_per_layer` with `DefaultTransferPlanner._get_buffers_per_layer`. Any test that calls `reg.register(...)` and expects `ValueError` keeps passing (`PeerIncompatibleError` is one) — if it calls `register` it needs `PeerRegistrar(ri, ext, DefaultTransferPlanner(ri, ext))`; grep `register(` in the file to find those.
  - `test_pool_matching.py:38,179,361`: same substitution.
  - `test_extractor.py:350,365`: same.
  - `test_kda_mamba_transfer.py:62,289`: `registrar = PeerRegistrar(ctx_ri, ext, DefaultTransferPlanner(ctx_ri, ext))` (this test exercises `register`).
  - `test_deepseek_v4_cache_manager.py:33,1945`: it calls `register` then `get_pool_mapping`/`get_kv_map`: build `planner = DefaultTransferPlanner(gen_rank_info, ext)`, `PeerRegistrar(gen_rank_info, ext, planner).register("ctx", 0, ctx_rank_info)`, then call `planner.get_pool_mapping(...)` / `planner.get_kv_map(...)`.
  - `test_bounce.py:239`: `safe = DefaultTransferPlanner._fanin_bounce_safe` (import from default_planner; drop the `tfr` importorskip if unused).
  - Also move the `PeerOverlap` import anywhere else: `grep -rn "PeerOverlap" tests tensorrt_llm`.

- [ ] **Step 6: Run** `python -m pytest disaggregated/test_peer.py disaggregated/test_pool_matching.py disaggregated/test_extractor.py disaggregated/test_bounce.py disaggregated/test_rank_info.py disaggregated/test_transfer_planner.py -q` → PASS. (`test_kda_mamba_transfer.py` and the deepseek_v4 test need a GPU; run them in Task 7.) `transfer.py` is still importing `PeerRegistrar` and calling the removed methods — it is not touched until Task 4, so do **not** run transfer-level tests yet.

- [ ] **Step 7: Commit** `git commit -s -m "[None][refactor] Move transfer planning out of PeerRegistrar into DefaultTransferPlanner"`

---

### Task 4: `Sender` plans through the planner

**Files:**
- Modify: `tensorrt_llm/_torch/disaggregation/native/transfer.py` — `Sender.__init__`, `_build_kv_write_meta`, `_build_aux_write_meta:1841`, readiness sites `1088`, `2228`, `2242`; delete `_pair_ordinals`.
- Modify tests: `tests/unittest/disaggregated/test_chunked_transfer.py:150-232`, `test_aux_buffer_registration.py:55-80`, `test_unsubmitted_session.py:45-70`, and any other `sender._registrar.get_peer_overlap` site (`grep -rln "get_peer_overlap" tests/unittest/disaggregated`).
- Test: `tests/unittest/disaggregated/test_transfer_planner.py` (add `test_build_kv_write_meta_uses_planner_plan`, `test_build_kv_write_meta_empty_plan_still_delivers`)

**Interfaces:**
- Consumes: `PeerRegistrar.planner`, `TransferPlanner.targets/plan`, `TransferPlan`.
- Produces: `Sender._planner` attribute; `Sender._expected_transfers(peer_ri) -> int`.

- [ ] **Step 1: Write the failing tests** (append to `test_transfer_planner.py`)

```python
from types import SimpleNamespace
from unittest.mock import Mock

from tensorrt_llm._torch.disaggregation.base.backend import Chunk, TokenRange
from tensorrt_llm._torch.disaggregation.native.transfer import KVSendTask, RecvReqInfo, Sender
from tensorrt_llm._torch.disaggregation.resource.page import CacheKind
from tensorrt_llm.llmapi.disagg_utils import DisaggregatedParams  # adjust to the module test_chunked_transfer.py uses


class _ToyPlanner(TransferPlanner):
    planner_id = "toy"

    def __init__(self, plan: TransferPlan, ranks=(0,)):
        self._plan, self._ranks, self.calls = plan, list(ranks), []

    def check_peer(self, self_ri, peer_ri):
        return None

    def targets(self, self_ri, peer_ri, peer_dp_rank):
        return PeerTargets(ranks=list(self._ranks), uniform_writers=True)

    def plan(self, self_ri, peer_ri, src, dst, token_range, prompt_len, dst_start_token):
        self.calls.append((src, dst, token_range, prompt_len, dst_start_token))
        return self._plan

    def receiver_payload_bytes(self, self_ri, peer_ri, dst):
        return 0


def _sender_with(planner):
    peer_ri = SimpleNamespace(
        instance_name="gen", instance_rank=3, dp_rank=0, device_id=1, cp_size=1,
        self_endpoint="tcp://gen:1", page_table=None,
    )
    sender = Sender.__new__(Sender)
    sender._registrar = Mock()
    sender._registrar.self_rank_info = SimpleNamespace(cp_size=1, instance_name="ctx", instance_rank=0)
    sender._registrar.get_peer_rank_info.return_value = peer_ri
    sender._planner = planner
    return sender


def _task_and_req():
    ids = np.array([5, 6], dtype=np.int64)
    task = KVSendTask(
        Chunk(block_ids_per_layer_groups=[ids], kind_per_layer_group=[CacheKind.PAGED],
              token_range=TokenRange(0, 64), is_last=True),
        DisaggregatedParams(),  # mirror test_chunked_transfer._make_params()
        slice_id=0, prompt_len=64,
    )
    req = RecvReqInfo(sender_req_id=1, instance_name="gen", instance_rank=3,
                      block_ids_per_layer_groups=[np.array([9, 10], dtype=np.int64)],
                      unique_rid=1, dst_start_token=32, slice_id=0)
    return task, req


def test_build_kv_write_meta_uses_planner_plan():
    plan = TransferPlan(src_ptrs=np.array([100, 200], dtype=np.int64),
                        dst_ptrs=np.array([300, 400], dtype=np.int64),
                        sizes=np.array([16, 16], dtype=np.int64))
    planner = _ToyPlanner(plan, ranks=(3, 4))
    sender = _sender_with(planner)
    task, req = _task_and_req()
    wm = sender._build_kv_write_meta(task, req)
    assert wm.src_ptrs is plan.src_ptrs and wm.dst_ptrs is plan.dst_ptrs and wm.sizes is plan.sizes
    assert wm.expected_transfers == 2 and task.expected_transfers == 2
    assert wm.dst_device_id == 1 and wm.peer_rank == 3
    (src, dst, tr, plen, dst_start), = planner.calls
    assert src is task._chunk.block_ids_per_layer_groups and dst is req.block_ids_per_layer_groups
    assert (tr.start, tr.end, plen, dst_start) == (0, 64, 64, 32)


def test_build_kv_write_meta_empty_plan_still_delivers():
    sender = _sender_with(_ToyPlanner(TransferPlan.empty()))
    task, req = _task_and_req()
    wm = sender._build_kv_write_meta(task, req)
    assert wm.sizes.size == 0 and wm.expected_transfers == 1 and wm.is_last_slice is True
```

(Use the same `_make_params()`/`TokenRange` helpers as `test_chunked_transfer.py` — copy them if they are module-private.)

- [ ] **Step 2: Run** `python -m pytest disaggregated/test_transfer_planner.py -q -k build_kv` → FAIL (`AttributeError: _planner` / old body calls `_registrar.get_peer_overlap`).

- [ ] **Step 3: Implement in `transfer.py`.**

`Sender.__init__`: after `self._registrar = peer_registrar` add `self._planner = peer_registrar.planner`. Add a helper right after `_get_session`:

```python
    def _expected_transfers(self, peer_ri: RankInfo) -> int:
        """How many writes the peer's DP group expects from this instance."""
        return len(
            self._planner.targets(self._registrar.self_rank_info, peer_ri, peer_ri.dp_rank).ranks
        )
```

Replace the four `len(self._registrar.get_peer_overlap(peer_ri, peer_ri.dp_rank).ranks)` with `self._expected_transfers(peer_ri)` (lines ~1088, 1841, 2228, 2242). Replace `_build_kv_write_meta` with:

```python
    @nvtx_range("_build_kv_write_meta")
    def _build_kv_write_meta(self, task: KVSendTask, req_info: RecvReqInfo) -> WriteMeta:
        peer_ri = self._registrar.get_peer_rank_info(req_info.instance_name, req_info.instance_rank)
        self_ri = self._registrar.self_rank_info
        timer = task._perf_timer
        if timer:
            timer.record_prepare_args_start(peer_ri.instance_rank)
        expected_transfers = self._expected_transfers(peer_ri)
        # Recorded on the task as well: the count on the write meta settles the task, while the
        # handle has to answer how many writes are still out after one of them has already failed.
        task.expected_transfers = expected_transfers

        plan = self._planner.plan(
            self_ri,
            peer_ri,
            task._chunk.block_ids_per_layer_groups,
            req_info.block_ids_per_layer_groups,
            task._chunk.token_range,
            task._prompt_len,
            req_info.dst_start_token,
        )

        if timer:
            timer.record_prepare_args_end(peer_ri.instance_rank)
            timer.record_transfer_sizes(
                peer_ri.instance_rank, int(plan.sizes.sum()), plan.dst_ptrs.size
            )

        return WriteMeta(
            task=task,
            src_ptrs=plan.src_ptrs,
            dst_ptrs=plan.dst_ptrs,
            sizes=plan.sizes,
            dst_device_id=peer_ri.device_id,
            expected_transfers=expected_transfers,
            peer_name=peer_ri.instance_name + str(peer_ri.instance_rank),
            peer_rank=peer_ri.instance_rank,
            peer_endpoint=peer_ri.self_endpoint,
            unique_rid=task._unique_rid,
            slice_id=task.slice_id,
            receiver_slice_id=req_info.slice_id if req_info.slice_id is not None else 0,
            is_last_slice=task._chunk.is_last,
            bounce_dst_base=req_info.bounce_dst_base,
        )
```

Delete `Sender._pair_ordinals`.

- [ ] **Step 4: Adapt sender-side mock fixtures.**
  - `test_chunked_transfer.py` `_make_projection_sender()`: keep `registrar` for `self_rank_info`/`get_peer_rank_info`; build the planner as the real engine with mocked internals so the ordinal-pairing tests still exercise `DefaultTransferPlanner.plan`:
    ```python
    planner = DefaultTransferPlanner.__new__(DefaultTransferPlanner)
    planner._self_ext_cache = extractor
    planner.peer_extractor = Mock(return_value=extractor)
    planner.get_peer_overlap = Mock(return_value=SimpleNamespace(ranks=[0]))
    planner.should_send_pool = Mock(return_value=True)
    planner.get_pool_mapping = Mock(return_value={(0, 0): (0, 0), (1, 0): (1, 0)})
    planner.get_kv_map = Mock(return_value=mapper)
    sender._planner = planner
    ```
    and replace `registrar.self_rank_info = SimpleNamespace(cp_size=1, cp_rank=0)` with one that also has `dp_rank=0`; `registrar.get_peer_overlap`/`should_send_kv`/`get_pool_mapping`/`peer_extractor`/`get_kv_map` lines go away. `expected_transfers` now comes from `planner.targets` → since `targets` is the real method calling the mocked `get_peer_overlap` and `_fanin_bounce_safe(...)` with `self_rank_info.cp_size` — give `peer_ri` a `cp_size=1` (it has one) and the SimpleNamespace overlap `duplicate_head_factor=1, overlap_pp_size=1`.
  - `test_aux_buffer_registration.py:69-71`: `sender = SimpleNamespace(_registrar=registrar, _planner=Mock(), _expected_transfers=lambda ri: 1)` — or set `sender._planner.targets.return_value = PeerTargets(ranks=[0])` and call `Sender._build_aux_write_meta(sender, ...)`; since `_expected_transfers` is a bound method on `Sender`, pass it explicitly as above.
  - `test_unsubmitted_session.py:63-64` and every other `sender._registrar.get_peer_overlap.return_value = SimpleNamespace(ranks=[0])`: replace with `sender._planner = Mock(); sender._planner.targets.return_value = PeerTargets(ranks=[0])`.
  - Find all: `grep -rn "get_peer_overlap\|should_send_kv\|get_pool_mapping\|get_kv_map" tests/unittest/disaggregated | grep -v test_peer.py | grep -v test_pool_matching.py`.

- [ ] **Step 5: Run** `python -m pytest disaggregated/test_transfer_planner.py disaggregated/test_chunked_transfer.py disaggregated/test_aux_buffer_registration.py disaggregated/test_unsubmitted_session.py -q` → PASS.

- [ ] **Step 6: Commit** `git commit -s -m "[None][refactor] Sender builds KV write metadata from the planner's TransferPlan"`

---

### Task 5: `Receiver` dispatches through the planner; `transfer.py` sheds model knowledge

**Files:**
- Modify: `tensorrt_llm/_torch/disaggregation/native/transfer.py` — imports (lines 65-70, 81), `Receiver.__init__`, delete `_fanin_bounce_safe` and `_get_mamba_slot`, rewrite the middle of `dispatch_task` (3215-3275), `_get_sender_info` (3381-3403).
- Modify tests: `tests/unittest/disaggregated/test_transfer_ownership_regressions.py` (3 receiver fixtures, lines ~602, ~699, ~872), any other `receiver._registrar.get_peer_overlap` site.
- Test: `test_transfer_planner.py` (add `test_transfer_module_has_no_model_knowledge`, `test_dispatch_task_payload_none_disables_bounce`)

**Interfaces:**
- Consumes: `PeerTargets.uniform_writers`, `TransferPlanner.receiver_payload_bytes/check_peer`, `RankInfo.planner_id`.
- Produces: `Receiver._planner`.

- [ ] **Step 1: Write the failing tests**

```python
import inspect
import tensorrt_llm._torch.disaggregation.native.transfer as transfer_mod


def test_transfer_module_has_no_model_knowledge():
    src = inspect.getsource(transfer_mod)
    for forbidden in ("mixers", "CacheKind", "MapperKind", "MambaPolicy", "AttentionPolicy"):
        assert forbidden not in src, f"native/transfer.py mentions {forbidden}"
```

For `test_dispatch_task_payload_none_disables_bounce`, copy the smallest receiver fixture from `test_transfer_ownership_regressions.py` (the ~860-900 one: single writer, `_enforce_physical_ownership=False`), give it `receiver._planner = Mock()` with `targets.return_value = PeerTargets(ranks=[0], uniform_writers=True)` and `receiver_payload_bytes.return_value = None`, a `receiver._bounce = Mock()`, and assert after `receiver.dispatch_task(task)` that `receiver._bounce.reserve.assert_not_called()` and `receiver._request_sender_data` was called once.

- [ ] **Step 2: Run** `python -m pytest disaggregated/test_transfer_planner.py -q -k "no_model_knowledge or payload_none"` → FAIL.

- [ ] **Step 3: Implement.**

Imports: delete `AttentionPolicy`, `MambaPolicy`, `mamba_receiver_payload_bytes`, `CacheKind`, `MapperKind` imports (keep `KVCachePageTable` only if still referenced — grep after the edit; add `from tensorrt_llm._torch.disaggregation.base.planner import PeerIncompatibleError, PeerTargets`). Delete `Receiver._fanin_bounce_safe` and `Receiver._get_mamba_slot`.

`Receiver.__init__`: after `self._registrar = peer_registrar` add `self._planner = peer_registrar.planner`.

`dispatch_task`, replace from `if sender_dp_rank is not None:` (first occurrence, ~3215) through the `bounced = ...` statement with:

```python
        self_ri = self._registrar.self_rank_info
        if sender_dp_rank is not None:
            # Normal path: ctx_dp_rank is known, send to overlapping ranks.
            peer_targets = self._planner.targets(self_ri, peer_infos, sender_dp_rank)
            dp0_targets = peer_targets
        else:
            # Gen-first with ADP: ctx_dp_rank unknown — broadcast REQUEST_DATA
            # to ALL ctx sender ranks so every DP group receives it. The real
            # writer count exceeds the per-group count, so the union is never
            # a uniform writer set.
            dp0_targets = self._planner.targets(self_ri, peer_infos, 0)
            all_ranks_set: set[int] = set(dp0_targets.ranks)
            for dp in range(1, peer_infos.dp_size):
                all_ranks_set.update(self._planner.targets(self_ri, peer_infos, dp).ranks)
            all_ranks = list(all_ranks_set)
            logger.debug(
                f"Receiver.dispatch_task: ADP broadcast path, dp_size={peer_infos.dp_size}, "
                f"all_ranks={all_ranks}"
            )
            peer_targets = PeerTargets(ranks=all_ranks, uniform_writers=False)

        # expected_transfers reflects one DP group even under broadcast: only
        # one group will actually process the context request.
        task.expected_transfers = len(dp0_targets.ranks)
        # TP fan-in splits ONE region equally, so allow it only when the planner says every
        # writer contributes the same share; never under ADP broadcast.
        allow_bounce = task.expected_transfers == 1 or (
            sender_dp_rank is not None and peer_targets.uniform_writers
        )
        # Bytes beyond the paged blocks that ride the same coalesced write
        # (recurrent state today) size the bounce region too; None = unknown.
        extra_bytes = self._planner.receiver_payload_bytes(
            self_ri, peer_infos, receiver_req.block_ids_per_layer_groups
        )
        if extra_bytes is None:
            allow_bounce = False
            extra_bytes = 0
        bounced = allow_bounce and self._bounce.reserve(
            receiver_req, task.expected_transfers, extra_bytes=extra_bytes
        )
```

and rename every later `peer_overlap.ranks` in the method to `peer_targets.ranks`.

`_get_sender_info`: replace the `try: MambaPolicy.validate_peer_compatible(...) ... except ValueError as e:` block with:

```python
            try:
                if sender_info.planner_id != self._planner.planner_id:
                    raise PeerIncompatibleError(
                        f"context peer plans transfers with '{sender_info.planner_id}', "
                        f"this rank with '{self._planner.planner_id}'"
                    )
                self._planner.check_peer(self._registrar.self_rank_info, sender_info)
            except PeerIncompatibleError as e:
                msg = (
                    f"context peer at '{info_endpoint}' is incompatible: {e} "
                    "(cached: all further requests to this context peer fail "
                    "fast; restart this generation worker to re-validate, "
                    "e.g. after redeploying a compatible server on the same "
                    "endpoint)"
                )
                self._incompatible_peers[info_endpoint] = msg
                raise PeerIncompatibleError(msg) from e
```

(Keep the explanatory comment above it, reworded: the sender-side gate is `PeerRegistrar.register` in the listener thread.)

- [ ] **Step 4: Adapt receiver fixtures.** In each `receiver._registrar = SimpleNamespace(get_peer_overlap=..., self_extractor=..., self_rank_info=...)`: drop `get_peer_overlap`/`self_extractor`, keep `self_rank_info`, and add `receiver._planner = SimpleNamespace(targets=Mock(side_effect=lambda _s, _p, dp: PeerTargets(ranks=overlaps[dp].ranks, uniform_writers=True)), receiver_payload_bytes=Mock(return_value=0))` (single-overlap fixtures: `Mock(return_value=PeerTargets(ranks=[...], uniform_writers=True))`). In the broadcast fixture (`overlaps = {0: ..., 1: ...}`) add the assertion that `receiver._bounce.reserve` was called with `expected_transfers == 2` only if it already asserts bounce behavior; otherwise assert `allow_bounce` indirectly via `reserve.assert_not_called()` when `ctx_dp_rank is None` (Review Focus 3).

- [ ] **Step 5: Run** `python -m pytest disaggregated -q -x --deselect disaggregated/test_kv_transfer.py --deselect disaggregated/test_kv_transfer_mp.py --ignore=disaggregated/test_kda_mamba_transfer.py` (GPU files excluded) → PASS, including `test_transfer_module_has_no_model_knowledge`.

- [ ] **Step 6: Commit** `git commit -s -m "[None][refactor] Receiver dispatches through the planner; transfer.py no longer knows cache kinds"`

---

### Task 6: Planner factory plumbing and the model mount point

**Files:**
- Modify: `tensorrt_llm/_torch/disaggregation/native/transfer.py` — `TransferWorkerConfig` (+`planner_factory`), `TransferWorker._setup_peer_infrastructure`
- Modify: `tensorrt_llm/_torch/disaggregation/transceiver.py:237-300` (`KvCacheTransceiverV2.__init__` +`planner_factory`)
- Modify: `tensorrt_llm/_torch/disaggregation/kv_cache_transceiver.py:144-220` (`create_kv_cache_transceiver` +`model=None`)
- Modify: `tensorrt_llm/_torch/pyexecutor/_util.py:4232` (pass `model=model_engine.model`)
- Modify: `tensorrt_llm/_torch/models/modeling_utils.py:486+` (`DecoderModelForCausalLM.build_transfer_planner`)
- Test: `test_transfer_planner.py` (add `test_registrar_rejects_planner_id_mismatch`, `test_receiver_rejects_planner_id_mismatch`, `test_worker_uses_model_planner_factory`)

**Interfaces:**
- Produces:
  ```python
  TransferPlannerFactory = Callable[[KVCacheManager, RankInfo, KVRegionExtractorV1], TransferPlanner]
  TransferWorkerConfig.planner_factory: Optional[TransferPlannerFactory] = None
  KvCacheTransceiverV2(mapping, dist, kv_cache_manager, cache_transceiver_config, planner_factory=None)
  create_kv_cache_transceiver(..., mamba_cache_manager=None, model=None)
  DecoderModelForCausalLM.build_transfer_planner(self, kv_cache_manager, self_rank_info, self_extractor) -> TransferPlanner
  ```

- [ ] **Step 1: Write the failing tests**

```python
def test_registrar_rejects_planner_id_mismatch():
    from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
    self_ri = _minimal_rank_info(planner_id="toy")   # reuse helper from test_rank_info or inline
    peer_ri = _minimal_rank_info(instance_name="gen", planner_id="trtllm.default")
    reg = PeerRegistrar(self_ri, Mock(), _ToyPlanner(TransferPlan.empty()))
    with pytest.raises(PeerIncompatibleError, match="toy"):
        reg.register("gen", 0, peer_ri)
    assert reg._peer_ri_cache == {}


def test_receiver_rejects_planner_id_mismatch(monkeypatch):
    from tensorrt_llm._torch.disaggregation.native.transfer import Receiver
    recv = Receiver.__new__(Receiver)
    recv._incompatible_peers = {}
    recv._sender_ep_instance_map = {}
    recv._registrar = SimpleNamespace(self_rank_info=_minimal_rank_info(planner_id="toy"))
    recv._planner = _ToyPlanner(TransferPlan.empty())
    recv._should_register_peer = lambda params: True
    sender_info = _minimal_rank_info(instance_name="ctx", planner_id="trtllm.default")
    fake_messenger = Mock(); fake_messenger.receive.return_value = [sender_info.to_bytes()]
    monkeypatch.setattr(transfer_mod, "ZMQMessenger", Mock(return_value=fake_messenger))
    recv._get_or_connect_dealer = Mock()
    params = SimpleNamespace(ctx_info_endpoint="tcp://ctx:9", ctx_dp_rank=0)  # match _extract_info_endpoint
    with pytest.raises(PeerIncompatibleError, match="trtllm.default"):
        recv._get_sender_info(params)
    recv._get_or_connect_dealer.assert_not_called()          # REGISTER_RANK_INFO never sent
    assert "tcp://ctx:9" in recv._incompatible_peers          # cached


def test_worker_uses_model_planner_factory():
    from tensorrt_llm._torch.disaggregation.native.transfer import TransferWorker
    worker = TransferWorker.__new__(TransferWorker)
    worker._rank_info = _minimal_rank_info()
    made = _ToyPlanner(TransferPlan.empty())
    seen = {}
    def factory(kvm, ri, ext):
        seen["args"] = (kvm, ri, ext); return made
    worker._config = SimpleNamespace(planner_factory=factory)
    kvm = Mock(); kvm.mapping.rank = 1
    monkeypatch_ext = Mock()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(transfer_mod, "KVRegionExtractorV1", Mock(return_value=monkeypatch_ext))
        mp.setattr(transfer_mod, "RankInfoServer", Mock())
        worker._setup_peer_infrastructure(kvm)
    assert worker._peer_registrar.planner is made
    assert seen["args"] == (kvm, worker._rank_info, monkeypatch_ext)
    assert worker._rank_info.planner_id == "toy" and worker._rank_info.planner_blob == b""
```

- [ ] **Step 2: Run** → FAIL (`TransferWorkerConfig` has no `planner_factory`; `_setup_peer_infrastructure` builds `PeerRegistrar(ri, ext)` with two args).

- [ ] **Step 3: Implement.**

`transfer.py`:

```python
TransferPlannerFactory = Callable[[KVCacheManager, RankInfo, KVRegionExtractorV1], TransferPlanner]

@dataclass
class TransferWorkerConfig:
    ...
    # Model-owned planner. None -> DefaultTransferPlanner.
    planner_factory: Optional[TransferPlannerFactory] = None
```

```python
    def _setup_peer_infrastructure(self, kvm: KVCacheManager):
        self._rank_info_server = RankInfoServer(self._rank_info) if kvm.mapping.rank == 0 else None
        self._kv_extractor = KVRegionExtractorV1(kvm)
        factory = self._config.planner_factory or (
            lambda _kvm, ri, ext: DefaultTransferPlanner(ri, ext)
        )
        self._planner = factory(kvm, self._rank_info, self._kv_extractor)
        # The envelope carries who planned and what they want the peer's planner to know.
        self._rank_info.planner_id = self._planner.planner_id
        self._rank_info.planner_blob = self._planner.describe()
        self._peer_registrar = PeerRegistrar(self._rank_info, self._kv_extractor, self._planner)
```

(`RankInfoServer` holds the object reference and serializes on request, as it already does for `self_endpoint`/`transfer_engine_info` set later.) Import `DefaultTransferPlanner` from `native.default_planner` and `TransferPlanner` from `base.planner`.

`transceiver.py`: `KvCacheTransceiverV2.__init__(..., cache_transceiver_config, planner_factory=None)` → `TransferWorkerConfig(..., planner_factory=planner_factory)`.

`kv_cache_transceiver.py`: `create_kv_cache_transceiver(..., mamba_cache_manager=None, model=None)`; in the Python branch:

```python
        planner_factory = getattr(model, "build_transfer_planner", None)
        return KvCacheTransceiverV2(mapping, dist, kv_cache_manager,
                                    cache_transceiver_config,
                                    planner_factory=planner_factory)
```

`_util.py:4232`: add `model=model_engine.model`.

`modeling_utils.py`, in `DecoderModelForCausalLM`:

```python
    def build_transfer_planner(self, kv_cache_manager, self_rank_info, self_extractor):
        """The disaggregated KV transfer plan for this model on this rank.

        A model whose cache layout the default planner does not understand
        returns its own :class:`TransferPlanner` here; both the context and
        the generation deployment of that model then run the same planner.
        """
        from tensorrt_llm._torch.disaggregation.native.default_planner import (
            DefaultTransferPlanner,
        )

        return DefaultTransferPlanner(self_rank_info, self_extractor)
```

(Local import: `modeling_utils` must not import the disaggregation package at module load.)

- [ ] **Step 4: Run** `python -m pytest disaggregated -q --ignore=disaggregated/test_kv_transfer.py --ignore=disaggregated/test_kv_transfer_mp.py --ignore=disaggregated/test_kda_mamba_transfer.py` → PASS; `python -m pytest llmapi/test_llm_args.py -q -k transceiver` and `api_stability` are unaffected (no LLM-arg change) — run `python -m pytest api_stability -q` once to be sure.

- [ ] **Step 5: Commit** `git commit -s -m "[None][feat] Let a model class supply its disaggregated TransferPlanner"`

---

### Task 7: GPU parity gate

**Files:** none (verification only).

- [ ] **Step 1: Holder job.** `cd /scratch/.../users/fredw/mv2-disagg && sbatch --parsable --qos=interactive alloc.sbatch` (writes `alloc.jobid`); wait for RUNNING.

- [ ] **Step 2: GPU unit tests** via `in_alloc.sh` with a runner that sets the same env as `run_disagg_test.sh` but `cd tests/unittest`:
  `python -m pytest disaggregated/test_kv_transfer.py disaggregated/test_kda_mamba_transfer.py disaggregated/test_kv_transfer_mp.py _torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py -q -x -p no:cacheprovider` → PASS (these exercise `TransferWorker` end-to-end with the default planner, including Mamba/KDA and the DeepSeek-V4 indexer pools).

- [ ] **Step 3: End-to-end gate**: `./in_alloc.sh run_disagg_test.sh "accuracy/test_modeling_v2_gpt_oss.py::TestModelingV2GptOss120bSm103Tp1::test_gsm8k_disagg"` → PASSED, GSM8K within tolerance of 90.3 (baseline on this branch before the refactor: 89.84). Then `"accuracy/test_disaggregated_serving.py::TestGPTOSS::test_kv_cache_v2_nixl_python"` (stock model, both cache-manager variants; needs 4 GPUs) → PASSED. Grep both logs for `planner_id` errors (none expected).

- [ ] **Step 4: Release the node** (`scancel $(cat alloc.jobid)`), record results in the PR description draft (not in code comments).

---

## Self-review notes

- Spec coverage: protocol (T1), wire fields (T2), DefaultTransferPlanner + registrar split (T3), sender (T4), receiver + grep gate (T5), factory/mount/plumbing + planner_id test (T6), parity (T7). Spec items consciously not done are listed under *Deviations* at the top.
- Names used consistently: `PeerTargets.ranks/uniform_writers`, `TransferPlan.src_ptrs/dst_ptrs/sizes`, `TransferPlanner.check_peer/targets/plan/receiver_payload_bytes/describe/forget_peer/planner_id`, `PeerRegistrar(self_rank_info, self_extractor, planner)` + `.planner`, `DefaultTransferPlanner(self_rank_info, self_extractor)`, `TransferWorkerConfig.planner_factory`, `build_transfer_planner(kv_cache_manager, self_rank_info, self_extractor)`.
- Review Focus items each have a pinned test: 1→T4, 2→T5, 3→T5 (broadcast fixture assertion), 4→T6, 5→T2.

# Disaggregated KV transfer: model-owned transfer planning

Date: 2026-10-10
Status: draft for review
Scope: `tensorrt_llm/_torch/disaggregation/native/` (transfer v2, Python/NIXL runtime)

## Problem

The transfer v2 sender decides *what bytes go where* in one shared code path:
`Sender._build_kv_write_meta` drives `PeerRegistrar`, which dispatches to two
hard-wired policies (`AttentionPolicy`, `MambaPolicy`) keyed by a closed
`CacheKind`/`MapperKind` enum set. Model knowledge leaks into the shared
module at several points:

| Site | What it knows |
|------|---------------|
| `native/transfer.py:1762-1777` | `CacheKind.STATE` branch, `MambaPolicy.is_paired`, helix CP strided slicing |
| `native/transfer.py:3134-3181` | `_fanin_bounce_safe` (duplicate-head, replicated views), `_get_mamba_slot` |
| `native/transfer.py:3262-3273` | `mamba_receiver_payload_bytes` for bounce sizing |
| `native/transfer.py:3381-3396` | receiver-side `validate_peer_compatible` of both policies |
| `native/peer.py` | `_POLICY_CLASSES` closed dict; `get_peer_overlap` computes attention `duplicate_head_factor` |
| `native/rank_info.py` | `AttentionInfo` is the only model-side description a rank can publish |
| `resource/page.py:37` | `MapperKind` closed enum |

A model whose cache has a new structure (a new recurrent state layout, a new
sharding of attention state, a sparse index pool with its own replication
rule) cannot support disaggregation without editing this shared module, and
every such edit has to preserve the invariants of every other model that
already flows through it.

## Idea

Apply the modeling_v2 principle: a target is one concrete implementation of
one concrete topology, so the target owns its transfer plan for that
topology. The framework keeps everything that is about *moving* bytes and
*tracking* a transfer; the model owns everything that is about *which*
bytes pair with which.

The current shared implementation is not deleted. It becomes
`DefaultTransferPlanner`, the planner every v1 model and every modeling_v2
target that does not override it uses. This mirrors the shared ops in the
modeling_v2 catalog.

## Boundary

### Model-owned: `TransferPlanner`

New module `tensorrt_llm/_torch/disaggregation/base/planner.py`.

```python
@dataclass
class PeerTargets:
    ranks: list[int]            # peer flat ranks this rank exchanges with
    uniform_writers: bool       # every writer contributes equal bytes (receiver fan-in bounce split)

@dataclass
class TransferPlan:
    src_ptrs: np.ndarray        # int64, device addresses on this rank
    dst_ptrs: np.ndarray        # int64, device addresses on the peer rank
    sizes: np.ndarray           # int64, bytes per (src, dst) pair
    # An empty plan is valid: this rank writes nothing to this peer but still
    # completes the handshake, as duplicate-head non-owners do today.

@dataclass(frozen=True)
class LayerGroupRows:
    """The rows one rank holds for a request in one pool view of one layer group.
    A row is a whole pool slot: block ordinal block_ordinals[i] starts at device
    address addresses[i] and is row_bytes long. span_blocks is how many ordinals
    the request covers on this side. A STATE group is one row, ordinal 0."""
    layer_group_index: int
    view_index: int
    kind: CacheKind
    block_ordinals: np.ndarray
    addresses: np.ndarray
    row_bytes: int
    span_blocks: int

class CacheDescriber(ABC):
    describer_id: str           # must match on both sides of a transfer

    @abstractmethod
    def rows(
        self,
        page_table: KVCachePageTable,
        slot_table_per_layer_group: Sequence[np.ndarray],   # positional, -1 = not held
        excluded_views: AbstractSet[tuple[int, int]],       # (layer group, view) left out
    ) -> list[LayerGroupRows]:
        """Single-sided: what this rank's cache holds for a request, off its own
        page table. Knows the model's cache layout, never the peer."""

class TransferPlanner(ABC):
    planner_id: str             # must match on both sides of a transfer

    @abstractmethod
    def describe(self) -> bytes:
        """Opaque self-description carried in RankInfo.planner_blob. Only this
        planner class decodes it (it runs on both sides)."""

    @abstractmethod
    def check_peer(self, self_ri: RankInfo, peer_ri: RankInfo) -> None:
        """Raise PeerIncompatibleError with a field-level reason. Replaces
        PeerRegistrar._check_peer_compatible and both validate_peer_compatible."""

    @abstractmethod
    def targets(self, self_ri: RankInfo, peer_ri: RankInfo, peer_dp_rank: int) -> PeerTargets:
        """Which peer ranks this rank exchanges with for one DP group of the peer.
        Replaces get_peer_overlap + should_send_pool + _fanin_bounce_safe."""

    @abstractmethod
    def pair(
        self,
        self_ri: RankInfo,
        peer_ri: RankInfo,
        self_rows: Sequence[LayerGroupRows],   # what this rank holds for the piece
        peer_rows: Sequence[LayerGroupRows],   # what the receiver holds
        *,
        token_range: TokenRange,               # the piece within the prompt
        prompt_len: int,
        dst_start_token: Optional[int],        # receiver accepts nothing below this
    ) -> TransferPlan:
        """Sender side, two-sided: pair the rows. Replaces the body of
        _build_kv_write_meta."""

    @abstractmethod
    def receiver_payload_bytes(
        self, self_ri: RankInfo, peer_ri: RankInfo, req_info: RecvReqInfo
    ) -> Optional[int]:
        """Receiver side, bytes beyond the paged blocks that ride the same write
        (recurrent state today). None = cannot size, receiver disables bounce.
        Replaces _get_mamba_slot + mamba_receiver_payload_bytes."""

def plan_transfer(describer, planner, self_ri, peer_ri, chunk, req_info, prompt_len) -> TransferPlan:
    """Framework composition for one piece: describe this side from chunk, describe
    the receiver from its wire message (req_info) against its published page table,
    then planner.pair(...)."""
```

The split follows where the knowledge lives. The describer looks at one side
only: it reads this rank's page table and a request's positional slot tables
(`Chunk.block_ids_per_layer_groups`, `RecvReqInfo.block_ids_per_layer_groups`)
and says which whole slots exist for the request and where. That is the model's
cache layout, nothing about the peer, so it is the half a KV cache manager
could export on its own. The planner looks at both sides: pool matching, head
re-sharding, who writes, and the ordinal intersection that decides which rows
move. It only ever consumes rows, so it never addresses memory itself and
never reads a slot table.

`RecvReqInfo` stays the receiver's wire message (its block tables,
`dst_start_token`, bounce address, aux slot); the sender rebuilds the
receiver's rows from it with the same describer. `receiver_payload_bytes`
still takes it whole. Both request types live in `base/backend.py`, so the
planner package depends on protocol types only, never on
`native/transfer.py`'s task objects.

The framework compares `describer_id` and `planner_id` across peers before
anything else; both travel on `RankInfo`. Today every model runs
`DefaultCacheDescriber` (`native/default_describer.py`): one row per held
slot at `base_address + slot * slot_stride_bytes`. There is no selection hook
for a describer yet because no model needs one; when one does, it is chosen
where the planner is chosen.

Threading contract is unchanged from today: `plan` is called where
`_build_kv_write_meta` is called now, `targets` and `receiver_payload_bytes`
where `dispatch_task` calls their predecessors. Planner-internal caches
(peer overlap, mappers) follow the same ownership as `PeerRegistrar`'s caches
do now.

### Framework-owned (unchanged responsibilities)

- `WriteMeta` assembly from a `TransferPlan` plus bookkeeping
  (`unique_rid`, `slice_id`, `bounce_dst_base`, `expected_transfers = len(targets.ranks)`).
- NIXL agent requests, bounce buffers (Python and C++ staging), aux transfer.
- Task lifecycle: `_LogicalOutcomes`, physical ownership, in-doubt polling,
  retirement, cancellation, ZMQ messaging, pipelined chunk numbering.
- Gen-first ADP broadcast: the union of `targets(...)` over all peer DP ranks
  stays in `Receiver.dispatch_task`; it is a protocol quirk, not a model one.
- Peer bookkeeping: peer `RankInfo` cache, `KVRegionExtractorV1` per peer,
  aux transfer layout cache, sender endpoints. These move out of
  `PeerRegistrar` into a framework `PeerRegistry`; `PeerRegistrar`'s
  remaining content (policies, mappers, overlap, pool mapping) becomes the
  internals of `DefaultTransferPlanner`.

### Wire format: `RankInfo`

`RankInfo` stays the envelope. Changes:

- Add `planner_id: str` and `planner_blob: bytes`.
- Add `enable_attention_dp: bool` as a top-level topology field; it is
  consumed by `tp_size_per_dp_group`, which is topology, not attention.
- Move `attention: AttentionInfo` into `DefaultTransferPlanner.describe()`.
  Nothing outside `AttentionPolicy` reads it once `enable_attention_dp` is
  lifted out.
- `page_table` stays top-level: the bounce layer and the region extractor
  consume it, and the default planner needs it too. A custom planner
  receives it like any other envelope field.
- `aux_meta` stays top-level (aux transfer is framework-owned).

Framework check at `PeerRegistry.register` and `Receiver._get_sender_info`:
`peer_ri.planner_id != self.planner.planner_id` raises
`PeerIncompatibleError` before `check_peer` runs. This is the one
cross-model invariant the framework enforces: both sides run the same
planner class. Blob format compatibility across wheel versions is the
planner's responsibility, exactly as `AttentionInfo` is today.

### Mount point

The planner comes from the model class, not the KV cache manager:

```python
class PretrainedModel/DecoderModelForCausalLM:
    def build_transfer_planner(self, kv_cache_manager) -> TransferPlanner:
        return DefaultTransferPlanner(kv_cache_manager)
```

The default is the method above. A modeling_v2 target does **not** override
it, and the framework does not pick a planner either: the user does.

- `TRTLLM_KV_TRANSFER_PLANNER=<planner_id>` names the planner. Unset, every
  model keeps the default plan.
- `disaggregation/planners/` holds every model-specific planner and
  `PLANNERS`, the map from `planner_id` to class. Each planner class answers
  `supports_deployment(architecture, mapping) -> bool`: the model architecture
  and the parallel topology are the whole input, so a built-in model and a
  modeling_v2 target of the same architecture are judged alike, and any
  width that satisfies the planner's conditions passes (dep4 may feed dep8).
- `AutoModelForCausalLM.from_config` calls `bind_transfer_planner`, which
  looks the name up, checks `supports_deployment` for the model just built
  (an unknown name or an unsupported deployment is a start-up error with the
  reason) and installs the planner's factory as the instance's
  `build_transfer_planner`. The framework reads the hook and nothing else;
  neither the model nor the disaggregation code names a planner.
- `describe()` is the base class's empty blob. The framework connects only
  ranks with equal `planner_id`, and the only way to hold a `planner_id` is
  to have passed the same `supports_deployment` at bind time, so the peer is
  already judged; `check_peer` compares the cache geometry both sides
  published (pool shape, layer count, `tokens_per_block`, `slot_bytes`).

Reason for the model over the cache manager still holds: a plan is a function
of the parallel topology *and* the cache layout, and the target is the one
object that already owns both. The cache manager builds the page table and
knows nothing about which peer topologies are supported.

Plumbing: `create_kv_cache_transceiver` (`pyexecutor/_util.py:4232`) gains a
`planner` argument built from `model_engine.model`;
`KvCacheTransceiverV2.__init__` passes it into `TransferWorkerConfig`;
`TransferWorker` hands it to `Sender`, `Receiver` and `PeerRegistry`.

## What moves where

| From | To |
|------|----|
| `Sender._build_kv_write_meta` body (`transfer.py:1710-1836`) | `DefaultTransferPlanner.plan`; `_build_kv_write_meta` shrinks to plan -> `WriteMeta` |
| `Sender._pair_ordinals` | `DefaultTransferPlanner` (a custom planner may import it as a helper) |
| `Receiver._fanin_bounce_safe`, `_get_mamba_slot`, `mamba_receiver_payload_bytes` call | `DefaultTransferPlanner.targets(...).uniform_writers` / `receiver_payload_bytes` |
| `Receiver._get_sender_info` policy validation (`3381-3396`) | `planner.check_peer` |
| `PeerRegistrar` policies, `_kv_map_cache`, `_overlap_cache`, `_lg_pool_mapping_cache`, `get_pool_mapping`, `get_kv_map`, `get_peer_overlap`, `should_send_pool`, `_owns_tp_fan_in` | `DefaultTransferPlanner` (file `native/default_planner.py`) |
| `PeerRegistrar` peer `RankInfo` cache, extractor cache, aux layout cache, `should_send_aux` | framework `PeerRegistry` (stays in `native/peer.py`) |
| `mixers/attention`, `mixers/ssm` | unchanged, now imported only by `default_planner.py` |

After the move, `native/transfer.py` imports nothing from `mixers/` and
mentions neither `CacheKind` nor `MapperKind`. That is the grep gate.

## Non-goals

- The C++ transceiver and the `UCX`/`MPI` backends.
- Aux (draft tokens, logits) transfer.
- Bounce internals and the pipelined-chunk protocol.
- Changing any mapper or policy behavior. Phase 1 is a pure extraction with
  bit-identical transfer plans.
- Writing the first target-owned planner. This spec creates the hook; the
  first custom planner is its own piece of work with its own ctx/gen target
  pair.

## Testing

- Existing `tests/unittest/disaggregated/test_peer.py`,
  `test_kda_mamba_transfer.py`, `test_kv_transfer.py` and `test_rank_info.py`
  keep passing with `PeerRegistrar(...)` construction retargeted to
  `DefaultTransferPlanner` / `PeerRegistry`. No behavior change means no
  golden change.
- One new contract test (`tests/unittest/disaggregated/test_transfer_planner.py`):
  a toy `TransferPlanner` (fixed ranks, fixed ptr arrays, `describe()`
  returning a marker) driven through `Sender._build_kv_write_meta` and
  `Receiver.dispatch_task` with a stub agent that records the `WriteMeta` it
  is handed. Asserts the plan arrives unmodified and that a `planner_id`
  mismatch fails the request with `PeerIncompatibleError` rather than the
  worker. The stub agent is new and lives in the test module; no existing
  fixture drives `Sender`/`Receiver` directly today.
- Grep gate on `native/transfer.py` (no `mixers`, `CacheKind`, `MapperKind`),
  enforced in the same test module.
- Integration: the existing disagg L0 tests on the native backend are the
  parity gate for Phase 1.

## Phasing

1. Extraction: `TransferPlanner` protocol, `DefaultTransferPlanner`,
   `PeerRegistry` split, `RankInfo` field changes, model mount point,
   plumbing. Zero behavior change.
2. First target-owned planner in a modeling_v2 target directory (separate
   spec, picks the model and the ctx/gen target pair).

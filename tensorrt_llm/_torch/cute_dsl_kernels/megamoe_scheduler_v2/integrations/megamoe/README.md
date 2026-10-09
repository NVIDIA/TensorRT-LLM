# MegaMoE host integration

`megamoe_scheduler.integrations.megamoe` owns the host connection between the
scheduler, in-switch weight copy and MegaMoE consumer. MegaMoE owns its kernels,
physical slots and helper READY gate. This package owns live-bank leases,
borrowed physical routes, copy submission and consumer-completion handoff.

The public names are `DynamicLoadBalanceBinding`, `MegaMoeWeightPlanes`,
`SamiLiveWeightBridge`, `TorchDistributedLiveBankLeaseProvider`,
`COMPLETION_CHECKED`, `STREAM_ORDERED`, `EACH_GENERATION` and `BIND_ONCE`.
Import these from this package; `next.sources` supplies no host compatibility
shim. The implementation lives in `direct_live_weight_bridge.py` and
`dynamic_load_balance.py` beneath this package.

## Construct one binding

The ordinary EP NVFP4 binding consumes one direct live bank. Before constructing
it, bind a fresh caller-supplied `InSwitchWeightCopy` endpoint to one scheduler,
its persistent outputs, and the exact framework producer stream. This package
does not construct or export that copy endpoint. The scheduler core exports
`CudaPhysicalSlotScheduler` and `CudaSchedulerOutputs`; `SchedulerOutputs` in
the adapter contracts refers to the bound outputs object, not another exported
class. The copy endpoint's SAMI backend
must expose system-scope remote visibility and the native terminal array.
`live_weight_views` below is the mapping of the seven allocator-owned tensors
registered with that live arena, using exactly these keys:

```text
mega_fc1_weight, mega_fc1_weight_sf, mega_fc2_weight, mega_fc2_weight_sf,
fc31_alpha, fc2_alpha, fc1_norm_const
```

The bridge validates their layout, capacity and storage identity. Copied or
repacked substitutes do not represent the registered live bank. The kernel
specialization must match the bound EP geometry, helper capacity, NVFP4 layout,
TopK shape and the 18-argument EP call signature. Pass the same kernel object
used to compile or load the callable; the callable needs no scheduler metadata.

```python
from megamoe_scheduler.integrations.megamoe import (
    COMPLETION_CHECKED,
    EACH_GENERATION,
    DynamicLoadBalanceBinding,
    MegaMoeWeightPlanes,
    SamiLiveWeightBridge,
    TorchDistributedLiveBankLeaseProvider,
)

# scheduler, copy_module and its seven live_weight_views are already bound.
producer_stream = copy_module.scheduler_stream
reuse_mode = COMPLETION_CHECKED
validation_mode = EACH_GENERATION

provider = TorchDistributedLiveBankLeaseProvider(
    copy_module,
    process_group=ep_process_group,
    reuse_mode=reuse_mode,
    validation_mode=validation_mode,
)
bridge = SamiLiveWeightBridge(
    weight_planes=MegaMoeWeightPlanes.from_mapping(live_weight_views),
    copy_module=copy_module,
    copy_stream=producer_stream,
    collective_live_bank_lease_provider=provider,
    reuse_mode=reuse_mode,
    validation_mode=validation_mode,
)
compiled = kernel.load_compiled(aot_path)
binding = DynamicLoadBalanceBinding(
    bridge, consumer_stream, compiled, kernel=kernel, validation_mode=validation_mode,
)

# Each generation: enqueue the scheduler before the bound consumer launch.
outputs = scheduler.submit(logical_routes, producer_stream.cuda_stream)
binding.launch(outputs, {
    "activation": activation,
    "activation_sf": activation_sf,
    "topk_scores": topk_scores,
    "output_activation": output,
    "local_workspace": local_workspace,
    "shared_workspace": shared_workspace,
    "peer_rank_ptr_mapper_host": peer_mapper,
})
```

`ep_process_group` is the already initialized process group for all participating
ranks; `None` selects the default group. The consumer stream must be on the same
CUDA device as the bound routes. The scheduler must return the same outputs
object that was bound at initialization. The example uses the core
`CudaPhysicalSlotScheduler.submit(routes, stream_handle)` no-Graph entry;
`run()` is its separate stage-and-launch path. `submit()` checks tensor
metadata on the CPU and consumes caller-owned route storage on the endpoint's
static stream. Inputs may have fewer rows than the configured capacity, including
zero rows on any rank; output storage retains its full configured shape and
unused routes become `-1`. Every rank still participates, including ranks with
no local tokens, since they may receive tokens for their resident experts.
Order input production and retain those inputs as described in
[HOST_SUBMISSION.md](../../../docs/HOST_SUBMISSION.md).

`binding.launch()` is the per-generation entry. It borrows the scheduler's physical
routes without allocating or copying a route tensor, submits the matching
weight-copy generation, acquires the consumer lease, forwards the routes, seven weights and READY pair to the native pipeline,
and records consumption on the bound stream. Callers supply exactly the seven
non-DLB operands in the example, rather than constructing partial launch
arguments or manipulating leases. The compiled router/body/optional TopKReduce
pipeline is invoked once. The API enqueues work; returning does not mean that
the output is ready for arbitrary host or other-stream access.

`bridge.physical_slot_ids` and the lease's `physical_slot_ids` reference the
exact `SchedulerOutputs.physical_slot_ids` tensor. The bridge property is read-only.
This applies to both reuse modes and both host-plan and GPU-direct copy handoffs.
No route snapshot allocation or D2D copy is enqueued. The route-ready event still orders
cross-stream consumers after HALO-Q, independently of helper-weight READY.
In both reuse modes, releasing the consumer enqueues a wait for its completion
on the producer stream. Enqueue the next scheduler only after `binding.launch()`
returns, on that same producer stream; do not overwrite or rebind the borrowed
routes until that dependency has completed. Retain the outputs for the binding's
lifetime.

## Reuse and validation modes

The defaults are `completion_checked` reuse and `each_generation` validation.
Before reusing a previously consumed bank, the default provider synchronizes its
completion event and performs a bounded collective barrier. All ranks must
participate consistently in this protocol.

`stream_ordered` removes that extra host completion check and collective reuse
barrier only under the following ordering contract:

1. Releasing the previous consumer records its completion event and makes the
   producer stream wait for that event.
2. The caller enqueues the next bound scheduler on that exact producer stream.
   The scheduler must exchange the new generation across all EP ranks before
   publishing its copy plan, as HALO-Q does.
3. `binding.launch()` records a route-ready event and queues in-switch copy
   after that scheduler. The generation exchange therefore precedes any
   new-bank writes and follows the previous consumers' completion on every participating rank.
4. The consumer waits for the route-ready event; MegaMoE's native helper gate still
   waits for the new copy's terminal publication before helper weight reads.

Construct both provider and bridge with `reuse_mode=STREAM_ORDERED` to select
this path. It requires the exact framework stream object, unchanged producer
stream identity and consistent generation order on every rank. A scheduler
without this generation-exchange guarantee does not satisfy the contract.
There is no `prepare_next_generation()` or `mark_stream_ordered_reuse_safe()`
callback. The remaining `mark_collective_reuse_safe()` is host bookkeeping by
the bound provider after scheduler enqueue; it adds no GPU signal or barrier.

`validation_mode=BIND_ONCE` captures storage contracts at construction. Select
it consistently on provider, bridge and binding only while the storage
identities and layouts of their weights, terminal array, borrowed routes and
outputs, together with the bound geometry, remain unchanged. Tensor contents
advance through the managed generation protocol.
`EACH_GENERATION` retains the per-generation validation. Neither option permits
reuse of memory still read by a consumer. Partial launch, copy or lease failures
poison the binding's bridge/provider; rebuild the affected group instead of
continuing with a half-installed generation.

## Scope

The example covers the ordinary helper-bearing EP NVFP4 pipeline with its native
18-argument AOT. Architecture-specific token-tile return modes are kernel
specialization choices; this host API preserves their arguments. LocalMegaMoE has
no EP token
communication or peer mapper and uses its own call interface. GenPhase also
has a separate kernel path; its helper support alone is not qualification of
this ordinary EP binding.

The host lifecycle described here preserves the existing direct-live behavior.
A package relocation does not establish a new correctness or performance
result. Qualification must bind the exact scheduler/integration and MegaMoE
commits, compiled artifacts and target toolchain. Delayed-consumer coverage
establishes the exercised ordering and final references; it does not by itself
prove actual overlap between a peer's next generation and a delayed consumer.


## Migrating route checks

Use `bridge.physical_slot_ids` (or `lease.physical_slot_ids`) instead of the removed
snapshot fields. A correctness probe should require
`bridge.physical_slot_ids is outputs.physical_slot_ids` and compare that tensor
with the scheduler oracle. The former non-aliasing check is no longer valid.
Custom test adapters now supply `stream_operations` with `record_event` and
`wait_event`; route allocation and copying are no longer part of the adapter.

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# The blob backend

A `Backend` of the KV sharing contract (`tensorrt_llm/_torch/disaggregation/base/backend.py`)
that moves KV rows between the lender's registered memory and a byte store, through a private
store driver. It is the reference path for any store that can hold a row as an object under a
key: the in-memory store the tests use, Mooncake, KVCR, or another SDK. The backend knows no
store; everything store-specific lives in a driver.

## Where it sits

```text
KV cache manager        lends regions; keeps the region hold
        │
        ▼
lease holder (runtime)  submits fetch / publish views; decides from the outcome; returns the
        │               lease once the attempt quiesced
        ▼
BlobBackend             this package: checks a view, cuts it into calls and hands each to the
        │               driver as one RowBatch; records what the driver tells of each call
        ▼
StoreDriver             one per store SDK: registers memory, runs the store's work on threads of
        │               its own, moves whole rows, answers two questions apart (result of a row,
        ▼               end of access to its memory) and tells of each change through on_change
store SDK
```

| Module | Holds |
|---|---|
| `backend.py` | `BlobBackend`, `BlobAttempt`, `BlobConfig`, `DriverCallLimit`, `RegionBinding`, `ConstructionRejected` |
| `store.py` | the driver boundary: `StoreDriver`, the optional `RegistersRegionsTogether`, `RowBatch`, `RowIO`, `OnChange`, `RowStatus`, `DriverCapabilities`, `DriverRejected`, `RegistrationError` |
| `keys.py` | the store key encoding |
| `mooncake.py` | `MooncakeDriver`, a `StoreDriver` over a set-up Mooncake 0.3.13 SDK client |
| `mooncake_setup.py` | `open_mooncake_driver`: reads a store config and opens the SDK client and the driver |
| `../testing/` | what a driver's own CI imports from the wheel: the conformance suite, the in-memory store and drivers, the broken drivers that prove every check bites |

## Running the tests

```bash
pytest tests/unittest/_torch/disaggregation/blob
```

All of it is CPU-only; `test_mooncake_real.py` also needs a Mooncake pool. The files, top down:

- `test_blob_example.py`: the backend end to end, each test the smallest example of one part
  of the picture: two endpoints sharing objects, a lease holder's decisions and leases, the
  region hold at shutdown, two backends in one process.
- `test_blob_backend.py`: the backend's behaviour against the in-memory driver, in every
  mode the test fixtures can stage (blocking, asynchronous, controlled step by step).
- `test_memory_driver.py`, `test_together_driver.py`: the conformance suite over the in-memory
  drivers, and the pairing meta-test that runs each check against a driver breaking exactly the
  rule it guards.
- `test_blob_format.py`: keys, spans, layouts, row batches, the object trailer.
- `test_testing_isolation.py`: no engine module imports the test tooling.
- `test_mooncake_driver.py`: the Mooncake driver over a fake SDK (`fake_mooncake.py`) that
  answers as 0.3.13 does, the conformance suite included.
- `test_mooncake_setup.py`: the store config, the SDK's setup and the refused settings and builds,
  and that each module imports only its layer.
- `test_mooncake_real.py`: the Mooncake driver and the backend over a real pool; runs only with
  `TRTLLM_MOONCAKE_REAL=1`.

## Writing a driver

The boundary is `store.py`; its docstrings are the specification. The rules that shape a driver:

1. **`DriverRejected` is the only raise that certifies nothing escaped.** It is raised before the
   store or the caller's memory was touched. A `put` or `get` that raises anything else once
   it was made may have touched its rows: the backend keeps them protected for good and sends
   the driver no more work. A `register` that fails says through `RegistrationError` what it
   left behind: `handle=None` certifies nothing, a handle names what to revoke; a failure the
   driver has no handle for escapes as the exception it is, and the memory stays protected.
2. **Two questions, answered apart.** `poll` gives each row's logical result; `access_ended`
   says whether the operation can still touch the rows' memory. Neither implies the other. A
   row's result may land long before its access ended, and the memory is reused only on the
   latter. `access_unprovable` is the driver's admission that it will never know. The driver
   tells the caller of each change through the `on_change` it was handed, after the change
   shows through those queries; the backend never polls.
3. **The driver owns the store's work.** `put` and `get` take a `RowBatch`, record and queue
   it, and return without waiting on the store; the driver's own threads move the bytes, in the
   order and at the concurrency its store wants. A batch's rows are laid out alike and never
   name a key twice, and every span lies inside one registered extent, so a driver checks memory
   once per extent. A driver that cannot take a call refuses it with `DriverRejected`.
4. **Only READY and FAULT are verdicts.** MISS, NOT_READY, DECLINED and CANCELLED say that
   nothing was delivered and nothing went wrong, each under a condition the store confirmed;
   an error the driver cannot place is FAULT.
5. **`drain` is one-way and never forces an end.** After it the driver accepts no new work;
   it returns True only when nothing can still touch memory. `close` is allowed only after a
   True drain and with every registration revoked.
6. **Registration comes in two shapes.** A store that takes memory a region at a time
   implements `register` and `unregister`. A store that takes all of its memory in one step,
   as one that opens its client with it, implements `RegistersRegionsTogether` instead:
   `register_regions(specs)` returns one handle for the batch and `unregister_regions(handle)`
   gives it back. No atomicity is promised; the failure shapes are the same as `register`'s.

### The conformance suite

```python
from tensorrt_llm._torch.disaggregation.testing.driver_conformance import DriverConformance


class TestMyDriverConformance(DriverConformance):
    def make_driver(self):
        return MyDriver(open_my_client())
```

- **Scenario hooks** stage what a check needs from the store: `make_blocked_get`,
  `make_held_key`, `make_full_store`, `make_failing_get`, `make_unprovable_get`, and the rest
  listed on `DriverConformance`. A hook that returns `None` skips its check; a driver's CI
  should implement every hook its store can honour.
- **Pool mode**: a driver whose memory is registered up front sets `registers_on_demand =
  False`, carves endpoints out of its pool in `allocate`, and revokes the pool in `shutdown`.
- **Batch mode**: a driver that registers regions together sets `registers_together = True`
  on top of pool mode and implements `make_together_registration`,
  `make_failing_together_registrations` and `make_busy_together_registration`.
- **Device memory**: override `allocate(nbytes, fill, memory_kind)` to allocate device memory
  when the kind is DEVICE, and declare `device_memory=True`. `allocate` returns a pair: the
  memory, an object implementing `Memory` (`address`, `nbytes`, `read`, `write`,
  `canaries_intact`), and its registration handle when the driver registered it up front, or
  `None` when the suite is to register it. The same checks then run over GPU memory; nothing
  else changes. A sketch over a CUDA tensor, with the protocol's signatures:

  ```python
  class DeviceBuffer:
      def __init__(self, nbytes: int, fill: int) -> None:
          self._tensor = torch.full((nbytes,), fill, dtype=torch.uint8, device="cuda")

      address = property(lambda self: self._tensor.data_ptr())
      nbytes = property(lambda self: self._tensor.numel())

      def read(self, offset: int = 0, nbytes: int | None = None) -> bytes:
          end = self.nbytes if nbytes is None else offset + nbytes
          return bytes(self._tensor[offset:end].cpu().numpy())

      def write(self, data: bytes, offset: int = 0) -> None:
          source = torch.frombuffer(bytearray(data), dtype=torch.uint8)
          self._tensor[offset : offset + len(data)].copy_(source)

      def canaries_intact(self) -> bool:
          return True  # a sketch: no guard bytes, so a write past the region goes unseen


  class TestMyDriverConformance(DriverConformance):
      def allocate(self, nbytes, fill, memory_kind):
          if memory_kind is MemoryKind.DEVICE:
              return DeviceBuffer(nbytes, fill), None  # the suite registers it
          return HostBuffer(nbytes, fill), None
  ```

- **Proving the checks bite**: `testing/broken_drivers.py` pairs each check with a driver that
  breaks exactly its rule; `test_memory_driver.py` runs every pair. A new check comes with a
  new pair.

## Assembling a backend

```python
driver = open_driver(store_config)  # whoever opens the driver closes it
binding = RegionBinding(regions, driver, hold)  # hold: the lender's region hold
binding.register()  # ConstructionRejected: nothing registered, the hold released
backend = BlobBackend(binding, wake, BlobConfig(key_prefix="trtllm-kv"))

...  # the lease holder fetches and publishes

labels = backend.shutdown(timeout)  # what is still accessible, if anything
if not binding.live_labels():
    binding.release_hold()  # on the manager's thread
    driver.close()  # after the drain and the revocation
```

`register` raises `ConstructionRejected` only when nothing is registered and the hold is
already released. Any other raise means a registration may be live: keep the binding, call
`try_unregister_all()` again later, and `release_hold()` once `live_labels()` is empty. The
same holds when `BlobBackend(...)` itself raises, as when its progress thread cannot start: the
binding its owner registered is still live, and that owner revokes and releases as after any
failed build.

Whoever opened the driver closes it, and owns it until then: after `shutdown` returned no
labels and `release_hold()` succeeded, the driver's `close()` is the last step. While anything
is still accessible, the owner keeps the driver, the binding and the hold alive; nothing frees
them on its own. A store that takes all of its memory in one step revokes it in one step too,
through `RegistersRegionsTogether`, and the same rule applies to the batch.

`shutdown` returns the labels of what the backend still holds: `attempt:<n>` for an attempt
whose access has not ended, `region:<i>` for a region still registered. It returns early once
nothing within its budget could change that, and stops nothing: the progress thread keeps
retrying the drain and the revocation, and the hold stays with the manager until
`release_hold()` succeeds. A non-empty set means "keep everything, ask again later", never
"free it".

## Two examples

1. **Publish and fetch between two endpoints**:
   `test_blob_example.py::test_two_endpoints_share_objects_through_the_store`. The writer's
   source bytes are untouched by its publish; the reader's miss leaves its slot alone; the
   store keeps its own copy, so the source may change once the publish quiesced.
2. **A lease holder's loop**: `FakeLeaseHolder` in `cpu_fixtures.py`, and the tests around it.
   Decide once, from the outcome or from nothing at the deadline; return the lease on
   `quiesced()` alone; release the region hold only when no region is registered.

## Decisions worth knowing

- **Batch registration**: `register_regions` registers a set of regions under one handle
  and promises no atomicity. `DriverRejected` or `RegistrationError(handle=None)` certify
  nothing is registered; `RegistrationError(handle=h)` says some of it may be and `h` revokes
  whatever is; any other raise means some of it may be registered with nothing to revoke it
  by, and the binding keeps the hold for good.
- **Who closes the driver**: the one who opened it, after `shutdown` returned no labels and the
  hold went back. The backend never calls `close`; a cleanup that did not finish keeps the owner,
  the driver and the memory, and is retried, never skipped.
- **`RegistrationError(handle=None)`** means the store certainly did not register the
  region. A driver that cannot tell gives the handle. A failure it has no handle for is not a
  `RegistrationError(None)`: the original exception escapes and the caller keeps the memory
  protected.
- **Hints** reach the driver only on a fetch and only when it declares `hints`; a hint-free
  fetch keeps the hint-free call shape.
- **Cancel** is asked of a driver that declares `cancel`, once per call of a stopped attempt
  whose rows have not all concluded, by the progress thread; it is never access-end evidence.
- **No queue of its own**: every call goes to the driver as it is submitted, so the backend
  bounds nothing and orders nothing. How many calls a store runs at once, in what order, and
  when it refuses more (`DriverRejected`, which fails that call) is the driver's; how much to
  submit at a time is the lease holder's admission policy. There is no backend-side cap on
  queued work.
- **One call per group, cut twice**: `BlobBackend.driver_call_limit(region_index)` tells a
  holder how many rows of a region one driver call carries (the driver's `max_rows_per_op`),
  so it can bound what a single call that never proves its access ended would pin; `None`
  means the driver declares no limit on the rows of one call. A group is also cut at a
  repeated object key: a view may name one key at two addresses, while a `RowBatch` carries no
  key twice. The rows keep their order, and the outcome's masks their positions.
- **A driver out of its protocol** (a raise from `put`/`get` that is not `DriverRejected`, no
  handle returned, a raise from a query, a `poll` with the wrong row count) is fenced: the call
  is held as unprovable unless its access end was recorded already, the calls of the same
  submission after it never reach the driver, and every later submission is rejected.

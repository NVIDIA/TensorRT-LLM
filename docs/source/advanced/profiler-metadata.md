<!-- Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

# Shared profiler metadata contract

TensorRT-LLM's Python and native profiling primitives attach a single string
input to a PyTorch `USER_SCOPE`. This document specifies the selector,
metadata envelope and ownership contract. The MPI wrappers below consume these
primitives; neither the primitives nor the wrappers start a profiler.

## Process-start selection

Set `TLLM_PROFILE_CUSTOM_REGIONS` before starting the process. Both languages
recognize `workflow`, `mpi`, `nccl`, `nanobind`, `kv` and `moe`. When unset,
the selector includes `workflow`, `mpi`, `nccl` and `nanobind`; `kv` and `moe`
require explicit selection. For example, `workflow,moe` selects those two
categories. `none` disables all custom scopes.

Tokens are case sensitive and may have surrounding ASCII spaces, tabs,
carriage returns or newlines. Empty tokens, empty values, duplicates, unknown
tokens, and any selection mixing `none` with other tokens are errors.

Python validates and caches the selector when importing `tensorrt_llm`.
Native applications must call `tensorrt_llm::common::initializeProfiler()`
at a startup boundary that can handle `std::invalid_argument`, before
installing instrumented consumers. Successful and invalid selections are
cached, and later environment changes do not reconfigure either language.
Native initialization is thread safe; probes stay inactive until it succeeds.
Do not initialize from a collective, profiler callback, ABI entry point or
shared-library static initializer.

`ENABLE_NCCL_PROFILING` and `ENABLE_NANOBIND_PROFILING` default to OFF.
Builds enabling either option validate native selection once at the Python
module's exception-handling startup boundary, before registering bindings.
Runtime selection cannot enable instrumentation omitted at build time.

## TRTLLM1 envelope

The encoding is exactly:

```text
TRTLLM1<decimal UTF-8 byte length>:<region><Python literal dictionary>
```

For example:

```text
TRTLLM18:workflow{'rank':-7,'ids':[]}
```

The region length counts bytes, not Unicode code points. It applies only to
the region, and no separator follows the region. UTF-8 is used for the complete
string. Whitespace and quote choices inside the dictionary may differ between
producers; consumers compare decoded values. Preserve the prefix and length
framing when extending payloads. Use a literal decoder such as `ast.literal_eval`,
never `eval`.

Python supports string keys and recursively nested values of the following
built-in types: strings, integers, finite floats, booleans, `None`, lists,
tuples and dictionaries with string keys. Sets, tensors, non-finite floats,
custom objects and cyclic containers are not supported. Python integers have
arbitrary precision. The encoder uses a Python literal dictionary, not JSON.

Native `ProfilerMetadata` supports a smaller subset: UTF-8 strings, signed
int64 values and vectors of int64 values, which decode as lists. String keys
and values escape backslashes, quotes, newlines, carriage returns, tabs and
ASCII control characters. Empty vectors decode as `[]`. Native consumers must
use unique field names and valid UTF-8. Both languages' tests decode the same
Unicode, escaping, signed-integer and empty-list golden payloads.

## Scope and ownership behavior

Python `record_function_metadata_scope` accepts a name, typed `ProfilerRegion`,
and a lazy factory returning either typed fields or an already encoded string.
It yields `None` and is decorated with `contextmanager`, so it can be used as
both a context manager and a function decorator. An already encoded string
must obey this envelope contract. Disabled or inactive scopes never evaluate
the factory. An active Python scope constructs metadata once.

Native `RecordFunctionWithMetadata::start` accepts an owned string name and
a metadata factory taking `ProfilerMetadata&`. `startWithLazyName` also makes
the name lazy. Disabled and inactive probes evaluate neither factory. An
active probe evaluates metadata only when its `RecordFunction` callbacks need
inputs, and starts at most once. `ProfilerMetadata::take() &&` transfers the
completed encoded string and consumes the builder once.

The native scope owns its name and `IValue` input storage until callbacks end;
it cannot be copied or moved. Create, start and destroy it on the same thread.
Destruction ends a started scope once. Metadata and profiler setup/cleanup
failures are observational: they must neither skip the operation nor replace
an exception from it. Native scope construction, start and destruction are
non-throwing; malformed selectors fail at explicit startup instead. Python
does not swallow process-control `BaseException` subclasses from factories.

The ATen-dependent implementation is excluded from the generic common-library
source glob, embedded in the PyTorch `th_common` library and compiled into the
CPU-capable native unit test target. `BUILD_PYT=OFF` does not build it. No new
runtime library or wheel artifact is required. Profiling has measurable costs,
including callbacks, scope creation and active metadata encoding.

## MPI operations

`MPIDist` records `trtllm::mpi_<method>` scopes in the `mpi` category. `TorchDist`
is not labeled as MPI. Coverage includes world broadcast/allgather/allreduce/barrier,
TP and CP broadcast/allgather, TP allreduce/barrier/gather, PP
broadcast/allgather/gather, and buffer/object send/isend/recv. Typed int64
allgather/broadcast paths are also covered. Combined operations record their
constituent communication steps, retaining CP-then-TP allgather and TP-then-CP
broadcast ordering. TP/CP `small_payload` fast paths remain native object
allgathers; chunked and typed-buffer protocols are unchanged. A nonblocking send
scope covers submission, not request completion.

Every scope includes `mpi.operation`, `mpi.communicator` (`world`, `tp`, `pp` or
`cp`), `mpi.kind` (`collective` or `p2p`), `mpi.rank`, and `mpi.group`. Rank and
ordered comma-separated group members use the active parent `mpi_comm()` rank
space, not necessarily `MPI.COMM_WORLD` or the mapping's world size/rank. This
includes thread-local communicator overrides. World membership is queried from
that parent's actual size. Uninitialized subgroups use their ordered mapping
members; cached subgroups translate their actual group into the parent's rank
space. Inspection is lazy and noncollective, frees temporary MPI groups, and
never creates a communicator solely for profiling. Unmappable members or inspection failures discard
the scope without changing the operation's result or exception.

`mpi.root` and `mpi.peer` are passed through in the operated communicator's
local rank space, **not** the parent rank space: for TP/PP/CP, a root of `1`
means the second member of `mpi.group`. World P2P peers and world roots are
already parent-local. `mpi.direction` is `send` or `recv`; `mpi.tag` preserves
the supplied tag, including MPI wildcard values. `mpi.reduce_op` is the numeric
`ReduceOp` value and `mpi.chunk_bytes` is the configured chunk-size bound, not
the measured transfer volume. Direct NumPy integers/floats/booleans and enums
are converted to the shared literal types; nested values still obey the shared
codec and are not recursively normalized by MPI helpers.

An explicitly supplied payload, including `None`, adds `mpi.payload.type`.
NumPy arrays of any shape/dtype add `shape`, `dtype`, `numel` and `bytes` under
`mpi.payload`; tensors also add `stride` and `device`. These describe only the
direct input: no values are read, no device copies/synchronizations or pickling
are introduced, and nested tensors are not discovered. Scalars and unknown
objects provide only their type. Strings/lists/tuples/dicts/sets provide their
top-level `length`; bytes/bytearray/memoryview provide their byte size (not the
memoryview's element count). Typed methods describe the input object before
their existing int64 conversion, except combined typed broadcasts describe each
step's existing buffer. Counts describe logical payload storage, not serialized
wire bytes or receive results. Object receives have no input payload metadata.

`record_mpi_function` supports an eager integer rank or a lazy rank callable;
the callable, group factory and payload inspection run only inside the selected,
active shared scope. Metadata is observational and capture-safe for direct
shape/stride inspection, but it does not trace tensor contents or asynchronous
communication completion. Profiling introduces scope and metadata overhead.

## Nanobind binding coverage and compatibility

Build with `ENABLE_NANOBIND_PROFILING=ON`; no NCCL profiling dependency is
required. `tensorrt_llm.bindings.BuildInfo.ENABLE_NANOBIND_PROFILING` is a Python
bool in both configurations. OFF builds retain the upstream dispatch paths.
ON scopes are named `trtllm::nanobind::<module>.<qualname>`. All four pinned
function vectorcall paths are covered, including rejected arguments and
exceptions. Bound methods and constructors delegate to those paths without
adding a second scope. Native startup and probes use the existing shared
`th_common` implementation, not a second embedded selector.

Names are extracted only for active callbacks. Direct Tensor positional and
keyword arguments are inspected only when callbacks request inputs; nested
containers are not traversed. Metadata uses region `nanobind` and the fields
`nanobind.tensor_count` (int64) and `nanobind.tensor.<index>.shape` (int64 list),
`.stride` (int64 list for strided tensors), `.dtype` and `.device` (strings).
Indices follow positional arguments then keyword values, including self when
it is a Tensor. Tensor handles are copied without reading data or synchronizing
devices; Python Tensor subclasses are recognized by their concrete C subtype.
Symbolic shape/stride failures can omit the observational scope, never the call.

The isolated private Tensor adapter is restricted to PyTorch 2.14 and the full,
GIL-enabled CPython API. Its checked ABI preparation used
`2.14.0a0+4fdf77b940.nv26.08` with pinned nanobind
`a0ed2587f1089ef7657e2ed49ad6756b01c74e9f`. Other Torch minors, stable Python ABI
and free-threaded Python are not supported by this instrumentation. Revalidate
the private headers, link against that installation's `torch_python`, and run
adapter/vectorcall regressions when updating the pin or Torch. OFF builds do
not include the private adapter. Source overrides bypass FetchContent patching:
apply the checked-in patch once to an owned pinned vendor copy before using an
override, and verify its contents.

Setup and cleanup preserve the current Python error type, value and traceback;
name extraction, metadata and profiler failures remain observational. The
scopes cover host binding dispatch, not GPU execution or graph replay. They do
not replace the runtime profiler manager or its executor-thread start/stop.

### Measured inactive binding cost

A standalone tiny-extension benchmark compared OFF and ON Release builds
(`-O3 -DNDEBUG`) with the checked Torch 2.14 build and nanobind pin above,
`TLLM_PROFILE_CUSTOM_REGIONS=nanobind`, and no active profiler. Five paired
fresh-process trials on `cpu-0036`, pinned to CPU 16, retained 400 block samples:
20 blocks of 1,000,000 calls per workload per process after 100,000 warmup calls.
The Python loop, inputs and single-thread settings were identical.

| Workload | OFF median (ns/call) | ON median (ns/call) | Mean paired increase (ns/call) |
| --- | ---: | ---: | ---: |
| Zero arguments | 43.579 | 138.235 | 94.601 (about 95) |
| One preallocated CPU Tensor | 44.858 | 135.917 | 91.038 (about 91) |

OFF/ON values are medians of the five process block medians; increases are
means of paired differences, not differences of those marginal medians.
Retained independent fixture artifacts `measurement.md`, `measurement.json`
and `timing.py` (job `7775138`, source revision `5b2b764a8e0be158e44a47712139d2035e28d310`)
provide the raw samples and provenance. This one-node, five-pair result includes
the Python loop and tiny binding dispatch; it is not an inference, GPU,
product-binding or active-metadata-serialization measurement, and does not
establish a platform-wide cost. ON is not zero-cost. Product binding timings
and the full supported OFF/ON native matrix remain separate release validation
gates, not inferred from this fixture.

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
carriage returns or newlines. Empty or whitespace-only values behave as unset.
Malformed nonempty values (unknown or duplicate tokens, empty comma-separated
tokens, or `none` mixed with other tokens) produce a startup warning and use the
default selection. Set `none` explicitly to disable custom regions.

Python and native code cache selection at startup; later environment changes do
not reconfigure either language. Native applications call
`tensorrt_llm::common::initializeProfilerRegions()` once and handle its optional diagnostic
at their startup boundary. Native initialization is thread safe; probes remain
inactive until initialization. Do not initialize from a collective, profiler
callback, ABI entry point or shared-library static initializer.

Native NCCL metadata scopes require `ENABLE_NCCL_PROFILING`, which defaults
to OFF. Runtime selection cannot enable instrumentation omitted at build time.
The capability is exposed by `tensorrt_llm.bindings.BuildInfo.ENABLE_NCCL_PROFILING`.
Native nanobind metadata scopes are not yet instrumented in this layer; the
`nanobind` selector token is reserved.

## Capture prerequisites and output formats

Custom Python scopes require a `torch.profiler.profile` session active on the
thread emitting the event. An `ExecutionTraceObserver` alone does not enable
them. To retain the metadata string input, attach an execution-trace observer
to the profiler. Shape recording is optional. The `TRTLLM1` fields are inputs in
the execution trace; `export_chrome_trace` does not export those fields. Chrome
traces still show scope names and CPU durations.

Profiler state is thread-local: a session on the executor thread does not enable
scopes on background send/receive threads. Use a profiler on each emitting thread
when that coverage is needed.

`ExecutionTraceObserver` registration is process-wide: only the first observer
installs a capture callback, and another registered observer can stop or
unregister that shared capture. Have one coordinator register and start an
observer before worker activity. Run each thread's profiler without an attached
observer; stop and unregister the coordinated observer only after all worker
profiling finishes. Sharing an observer between independently ending profiler
contexts also lets the first context to exit stop and unregister shared capture.

CUDA graph replay does not re-enter Python or native host scopes recorded at
capture time.

## TRTLLM1 envelope

The encoding is exactly:

```text
TRTLLM1:<decimal UTF-8 byte length>:<region><JSON object>
```

For example:

```text
TRTLLM1:8:workflow{"rank":-7,"ids":[]}
```

The region length counts bytes, not Unicode code points. It applies only to
the region, and no separator follows the region. UTF-8 is used for the complete
string. Whitespace and quote choices inside the dictionary may differ between
producers; consumers compare decoded values. Preserve the version and length
framing when extending payloads. Decode the payload with a JSON parser. The test
utilities also accept the earlier Python-literal envelope for existing traces;
new producers emit only the separated JSON format.

Python supports string keys and recursively nested values of the following
built-in types: strings, integers, finite floats, booleans, `None`, lists,
tuples and dictionaries with string keys. Sets, tensors, non-finite floats,
custom objects and cyclic containers are not supported. Python integers have
arbitrary precision. Tuples encode as JSON arrays and `None` encodes as `null`. JSON floats must be finite.

Native `ProfilerMetadata` supports a smaller subset: UTF-8 strings, signed
int64 values and vectors of int64 values, which decode as lists. The native encoder uses the existing nlohmann/json library for escaping. Empty vectors decode as `[]`. Native consumers must
use unique field names and valid UTF-8. Both languages' tests decode the same
Unicode, escaping, signed-integer and empty-list golden payloads.

## Scope and ownership behavior

Python `record_function_metadata_scope` accepts a name, typed `ProfilerRegion`,
and an optional lazy factory returning typed fields or an already encoded string.
Omitting the factory records a metadata-free scope with the same observational
error handling.
It yields `None` and is decorated with `contextmanager`, so it can be used as
both a context manager and a function decorator. An already encoded string
must obey this envelope contract. Disabled or inactive scopes never evaluate
the factory. An active Python scope constructs metadata once.

Native `RecordFunctionWithMetadata::startWithLazyName` accepts lazy name and
metadata factories; the metadata factory takes `ProfilerMetadata&`. Disabled and inactive probes evaluate neither factory. An
active probe evaluates metadata only when its `RecordFunction` callbacks need
inputs, and starts at most once. `ProfilerMetadata::take() &&` transfers the
completed encoded string and consumes the builder once.

The native scope owns its name and `IValue` input storage until callbacks end;
it cannot be copied or moved. Create, start and destroy it on the same thread.
Destruction ends a started scope once. Metadata and profiler setup/cleanup
failures are observational: they must neither skip the operation nor replace
an exception from it. Native scope construction, start and destruction are
non-throwing; selector diagnostics are handled at explicit startup. Python
isolates expected metadata/profiler failures (`AttributeError`, `LookupError`,
`TypeError`, `ValueError`, `RuntimeError`, and `OSError`) and emits a once-only
warning identifying the scope and failed phase. A Python metadata failure retains
the named scope without metadata, so trace boundaries remain visible. Custom factories must use these
error types for observational failures; other exceptions propagate. Process-control
`BaseException` subclasses also propagate.

The ATen-dependent implementation is owned once by the existing native shared
library when `BUILD_PYT` is enabled; Torch ops and bindings reuse that owner.
`BUILD_PYT=OFF` does not build it. Profiling adds scope, callback and active
metadata encoding costs.

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

`MPIDist` checks selection and the current profiler state before creating lazy
metadata callbacks or profiling scopes. Disabled or inactive calls use a reusable
no-op context. The same instance can emit scopes when profiling starts later.

Every scope includes `mpi.operation`, `mpi.communicator` (`world`, `tp`, `pp` or
`cp`), `mpi.kind` (`collective` or `p2p`), `mpi.rank`, and `mpi.group`. Rank and
ordered comma-separated group members use the active parent `mpi_comm()` rank
space, not necessarily `MPI.COMM_WORLD` or the mapping's world size/rank. This
uses the emitting thread's communicator override when that thread is profiled. World membership is queried from
that parent's actual size. Uninitialized subgroups use their ordered mapping
members. Subgroups created by `MPIDist` cache those known members with the
communicator and active parent identities. Externally supplied or replaced
subgroups translate their actual group once into the parent's rank space;
a changed communicator or parent invalidates that cached translation.
Inspection is lazy and noncollective, frees temporary MPI groups, and never
creates a communicator solely for profiling. Unmappable members or inspection
failures omit metadata while preserving the named scope and operation outcome.

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

Direct `_utils.mpi_*` helpers and background-thread calls without a profiler are
not covered by `MPIDist` annotations. A missing scope cannot establish that no
communication occurred.

## Nanobind binding coverage and compatibility

Build with `ENABLE_NANOBIND_PROFILING=ON`; no NCCL profiling dependency is
required. `tensorrt_llm.bindings.BuildInfo.ENABLE_NANOBIND_PROFILING` is a Python
bool in both configurations. OFF builds retain the upstream dispatch paths.
ON scopes are named `trtllm::nanobind::<module>.<qualname>`. All four pinned
function vectorcall paths are covered, including rejected arguments and
exceptions. Bound methods and constructors delegate to those paths without
adding a second scope. Native startup and probes use the existing native
shared-core implementation, not a second embedded selector.

Names are extracted only for active callbacks. Direct Tensor positional and
keyword arguments are inspected only when callbacks request inputs; nested
containers are not traversed. Metadata uses region `nanobind` and the fields
`nanobind.tensor_count` (int64) and `nanobind.tensor.<index>.shape` (int64 list),
`.stride` (int64 list for strided tensors), `.dtype` and `.device` (strings).
Indices follow positional arguments then keyword values, including self when
it is a Tensor. Tensor handles are copied without reading data or synchronizing
devices; Python Tensor subclasses are recognized by their concrete C subtype.
Symbolic shape/stride failures can omit the observational scope, never the call.

The private Tensor adapter supports PyTorch 2.14 and the full, GIL-enabled
CPython API. The pinned nanobind version is recorded in
[`3rdparty/fetch_content.json`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/3rdparty/fetch_content.json).
Other Torch minors, stable Python ABI and free-threaded Python are not supported
by this instrumentation. Revalidate private headers and adapter/vectorcall
behavior when updating either dependency. OFF builds do not include the private
adapter. Source overrides must include the checked-in dispatcher patch; configure
time checks reject incompatible overrides when profiling is enabled.

Setup and cleanup preserve the current Python error type, value and traceback.
Name extraction, metadata and profiler failures remain observational. The scopes
cover host binding dispatch, not GPU execution or graph replay, and do not replace
executor-thread profiler start/stop.

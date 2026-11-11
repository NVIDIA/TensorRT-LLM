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

Native NCCL and nanobind metadata scopes are not yet instrumented in this
layer. Their selector tokens are reserved; selecting them does not add native
probes.

## Capture prerequisites and output formats

Custom Python scopes require a `torch.profiler.profile` session active on the
thread emitting the event. An `ExecutionTraceObserver` alone does not enable
them. To retain the metadata string input, attach an execution-trace observer
to the profiler and enable shape recording. The `TRTLLM1` fields are inputs in
the execution trace; `export_chrome_trace` does not export those fields. Chrome
traces still show scope names and CPU durations.

Profiler state is thread-local: a session on the executor thread does not enable
scopes on background send/receive threads. Use a profiler on each emitting thread
when that coverage is needed. CUDA graph replay does not re-enter Python or native
host scopes recorded at capture time.

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

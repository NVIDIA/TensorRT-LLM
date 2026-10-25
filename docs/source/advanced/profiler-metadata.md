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
metadata envelope and ownership contract. These primitives do not add inference
instrumentation or start a profiler.

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

The native loader integration is intentionally deferred until native feature
consumers are added: NCCL and nanobind integrations must arrange safe startup
validation before installing their probes. `ENABLE_NCCL_PROFILING` and
`ENABLE_NANOBIND_PROFILING` are default-OFF build scaffolding. Runtime selection
cannot enable instrumentation omitted at build time.

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

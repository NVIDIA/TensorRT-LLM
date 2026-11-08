<!--
SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# NVTX API Reference

## Installation

```bash
pip install nvtx
```

## Modern API

**Decorator** — annotates every call to the function:

```python
import nvtx

@nvtx.annotate("training_step", color="blue")
def training_step():
    ...
```

If `message` is omitted, defaults to the function name:

```python
@nvtx.annotate()
def forward():  # NVTX range named "forward"
    ...
```

**Context manager** — annotates a code block:

```python
with nvtx.annotate("data_loading", color="green"):
    batch = next(dataloader)
```

## Domains

Domains provide namespace isolation. Use when your annotations might conflict with library-internal annotations:

```python
my_domain = nvtx.Domain("my_training")

@nvtx.annotate("step", domain=my_domain)
def step():
    ...
```

## Categories

Categories group annotations within a domain for filtering in profiler tools:

```python
@nvtx.annotate("forward", category=1)
def forward():
    ...

@nvtx.annotate("backward", category=2)
def backward():
    ...
```

## Payloads

Use payloads for per-call data (visible in nsys tooltips). Prefer payloads over f-string messages to avoid per-call string allocation:

```python
# WRONG: allocates a new string each call
with nvtx.annotate(f"batch_{batch_idx}"):
    ...

# RIGHT: use payload for variable data
with nvtx.annotate("batch", payload=batch_idx):
    ...
```

## Legacy API (Avoid)

The old push/pop API (`nvtx.range_push()` / `nvtx.range_pop()`) and PyTorch's `torch.cuda.nvtx.range_push()` still work but are error-prone (unbalanced push/pop). Prefer `@nvtx.annotate` or `with nvtx.annotate()`.

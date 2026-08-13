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

# Key Hot-Path Files

Based on the default profiling configuration in `host_profiler.py`, these are the critical files for host overhead analysis.

---

## PyExecutor Core (`py_executor.py`)
Main executor loop orchestrating inference requests.

| Method | Purpose | Common Hotspots |
|--------|---------|-----------------|
| `_prepare_and_schedule_batch` | Entry point for batch preparation | Request iteration |
| `_schedule` | Invoke scheduler for batch | Policy evaluation |
| `_forward_step` | Execute model forward pass | Model dispatch |
| `_sample_async` | Trigger async sampling | Sampler coordination |
| `_update_requests` | Update request states post-forward | State synchronization |
| `_update_request_states` | Process generation results | Token handling |
| `_fetch_and_activate_new_requests` | Pull new requests from queue | Queue operations |
| `_handle_responses` | Process completed requests | Response formatting |
| `_terminate_request` | Clean up finished request | Resource release |

---

## Model Engine (`model_engine.py`)
Input preparation and model forward dispatch. **Not profiled by default -- add to TLLM_LINE_PROFILER_FUNCTIONS.**

| Method | Purpose | Common Hotspots |
|--------|---------|-----------------|
| `PyTorchModelEngine.forward` | Prepare inputs + dispatch model | `_prepare_inputs` call |
| `PyTorchModelEngine._prepare_inputs` | Entry point for input preparation | Delegates to `_prepare_tp_inputs` |
| `PyTorchModelEngine._prepare_tp_inputs` | Core input tensor construction | PYLOOP over requests, REDUNDANT_ITER, DEAD_WORK |
| `PyTorchModelEngine._can_use_incremental_update` | Check if batch can be incrementally updated | List comparison |

---

## Sampler (`sampler.py`)
Token sampling and logits processing.

| Method | Purpose | Common Hotspots |
|--------|---------|-----------------|
| `TorchSampler.sample_async` | Async sampling entry | Tensor operations |
| `TorchSampler.update_requests` | Update request with samples | State sync |
| `TorchSampler._process_requests` | Core sampling logic | Strategy dispatch |
| `TorchSampler._select_generated_logits` | Extract logits for sampling | Tensor indexing |
| `TorchSampler._sample_batched_by_strategy` | Batch sampling by strategy | Group iteration |
| `_group_requests_by_strategy_key` (standalone) | Group requests by sampling params | Dict operations |

---

## Resource Manager (`resource_manager.py`)
KV cache and memory management.

| Method | Purpose | Common Hotspots |
|--------|---------|-----------------|
| `ResourceManager.prepare_resources` | Allocate resources for batch | Memory allocation |
| `ResourceManager.update_resources` | Update resource state | State tracking |
| `KVCacheManager.prepare_resources` | Allocate KV cache blocks | Block management |
| `KVCacheManager.update_resources` | Update cache metadata | Metadata sync |
| `KVCacheManager.free_resources` | Release KV cache blocks | Deallocation |

---

## Request Queue (`executor_request_queue.py`)
Request ingestion and preprocessing.

| Method | Purpose | Common Hotspots |
|--------|---------|-----------------|
| `ExecutorRequestQueue.fetch_new_requests` | Main fetch entry | Queue polling |
| `ExecutorRequestQueue._fetch_and_process_requests` | Fetch + preprocess | Tokenization |
| `ExecutorRequestQueue._merge_requests` | Combine requests for batching | List operations |

---

## Scheduler (`scheduler.py`)
Batch scheduling policy.

| Method | Purpose | Common Hotspots |
|--------|---------|-----------------|
| `RequestScheduler.schedule_request` | Scheduling decision | Policy evaluation |

---

## Common Drill-Down Targets

When a top-level profiled function shows >80% time in a single sub-call, drill down into these:

| Top-Level Function | Likely Sub-Function to Profile |
|---|---|
| `_forward_step` (98% in `forward()`) | `PyTorchModelEngine.forward`, `PyTorchModelEngine._prepare_tp_inputs` |
| `_sample_async` (95% in `sample_async`) | `TorchSampler._process_requests` |
| `_prepare_and_schedule_batch` (99% in `_fetch_and_activate_new_requests`) | `ExecutorRequestQueue.fetch_new_requests` |
| `_update_requests` (high % in `update_requests`) | `TorchSampler.update_requests` |

---

## Unit Test Mapping

| Modified File | Primary Test File | Key Tests |
|---|---|---|
| `model_engine.py` | `tests/unittest/_torch/executor/test_pytorch_model_engine.py` | `test_position_id_preparation`, `test_pad_generation_requests`, `test_warmup` |
| `sampler.py` | `tests/unittest/_torch/executor/test_sampler.py` | Sampler-related tests |
| `py_executor.py` | `tests/unittest/_torch/executor/test_py_executor.py` | Executor loop tests |
| `resource_manager.py` | `tests/unittest/_torch/executor/test_resource_manager.py` | KV cache tests |

<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

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

# Test stage resources

`stage_resources.json` configures the CPU and memory requests and limits of the `trt-llm`
container in Kubernetes test stages. Each top-level key is an exact stage name.
Keys that start with `_`, such as `_copyright`, are not stage names and are
skipped. CBTS can run a narrowed stage as `<stage>-cbts`; that stage uses the
entry for `<stage>` unless the file also has an entry for `<stage>-cbts`.
Values are positive integers: `cpu_m` is CPU millicores, and `memory_mi` is
memory in MiB. Requests and limits use the same values.

For example, this entry sets both requests and limits to 8 CPUs and 20 GiB:

```json
{
  "A10-PyTorch-1": {
    "cpu_m": 8000,
    "memory_mi": 20480
  }
}
```

The initial table is empty. Stages without an entry keep their existing requests
and limits. SLURM dispatcher pods keep their hardcoded resources, even if their
stage name appears in the file. JNLP and service containers keep their existing
resources; GPU and ephemeral-storage requests and limits are also unchanged.

`L0_Test.groovy` loads and validates the file once during Setup Environment. The
Jenkins controller reads only this file at the build's `gitlabCommit` from the
Artifactory GitHub proxy (`vcs-remote`) over HTTP. A git fetch is used only when
that request fails.
All Pod retries use the same resource settings.

Configured Pods get the Guaranteed QoS class because every container in them has
equal CPU and memory requests and limits. Kubernetes sets the QoS class from these
values; the `qosClass` line in the Pod template has no effect.

## Before you raise a value

Requests and limits use the same value, so Kubernetes reserves the full amount on
the node while the stage runs. Check these points:

- The value covers only the `trt-llm` container. The Pod also requests 2 CPUs and
  10 GiB for the JNLP container. ModelExpress stages also run `redis` (1 CPU,
  4 GiB) and `model-express-server` (4 CPUs, 8 GiB). The total must fit on one
  node of the stage's GPU type.
- Stay within the stage's share of the node. Several stages can run on the same
  multi-GPU node. By default, a single-GPU stage gets 12 CPUs and 96 GiB, and a
  multi-GPU stage gets 12 CPUs and 150 GiB per GPU. A stage that asks for more
  than its share can leave other GPUs on the node idle, because no other Pod
  fits next to it.
- For multi-GPU ModelExpress stages, the defaults subtract 6 CPUs and 16 GiB to
  leave room for the sidecars. An entry replaces the default, so subtract the
  same amount yourself.
- DGX Spark (`gb10x`) has 128 GB of memory that the CPU and the GPU share. The
  default gives 64 GiB to the CPU side and leaves the rest to the GPU. A larger
  value leaves less memory for the GPU.
- If no node has enough free CPU and memory, the Pod cannot be scheduled and the
  stage does not start.
- A PR that changes only this file does not run multi-GPU stages by default.
  Before merging, run `/bot run --stage-list "<changed stage names>"` and check
  that each changed stage starts and passes.

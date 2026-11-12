<!-- Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. -->

# Test stage resources

`stage_resources.json` configures the CPU and memory requests of the `trt-llm`
container in Kubernetes test stages. Entries in `stages` match the exact stage
name. Values are positive integers: `cpu_m` is CPU millicores, and `memory_mi`
is memory in MiB. Limits are 1.4 times the requests, rounded up to the next
millicore or MiB.

For example, this entry requests 8 CPUs and 20 GiB, with limits of 11.2 CPUs
and 28 GiB:

```json
{
  "stages": {
    "A10-PyTorch-1": {
      "cpu_m": 8000,
      "memory_mi": 20480
    }
  }
}
```

The initial table is empty. Stages without an entry keep their existing requests
and limits. SLURM dispatcher pods keep their hardcoded resources, even if their
stage name appears in the file. JNLP and service containers keep their existing
resources; GPU and ephemeral-storage requests and limits are also unchanged.

`L0_Test.groovy` loads and validates the file once during Setup Environment from
the build's `gitlabCommit`. All Pod retries use the same resource settings.
Configured Pods have Burstable QoS because their CPU and memory requests differ
from their limits.

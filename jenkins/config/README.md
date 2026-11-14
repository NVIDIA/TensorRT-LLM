<!-- Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. -->

# Test stage resources

`stage_resources.json` configures the CPU and memory requests and limits of the `trt-llm`
container in Kubernetes test stages. Entries in `stages` match the exact stage
name. Values are positive integers: `cpu_m` is CPU millicores, and `memory_mi`
is memory in MiB. Requests and limits use the same values.

For example, this entry sets both requests and limits to 8 CPUs and 20 GiB:

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
The template retains `qosClass: Guaranteed`. Kubernetes calculates the actual QoS
class from container resources. Configured test Pods meet the Guaranteed resource
requirements because all containers have equal CPU and memory requests and limits.

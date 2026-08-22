<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AWS EFA and LIBFABRIC for Disaggregated Serving

This guide is a deployment checklist for running TensorRT LLM disaggregated
serving on AWS EFA with NIXL using the LIBFABRIC backend. It complements the
main [Disaggregated Serving](disagg-serving.md) guide; start there for the
context/generation server topology and then apply the AWS-specific checks here.
The Kubernetes examples in this page are scoped to EKS-managed deployments;
SLURM or bare-metal deployments can reuse the container, LIBFABRIC plugin, and
runtime verification checks but should map device allocation to their scheduler.

For managed Dynamo deployments, also follow the Dynamo AWS EFA guide:
<https://docs.nvidia.com/dynamo/latest/kubernetes-deployment/cloud-provider-guides/aws/efa-rdma-over-aws-fabric>.

## When to use LIBFABRIC

Use LIBFABRIC when the context and generation workers run on different
EFA-enabled AWS nodes and KV-cache transfer should use EFA rather than TCP. In
TensorRT LLM, select NIXL at the cache-transceiver layer and select LIBFABRIC as
NIXL's transport backend:

```yaml
cache_transceiver_config:
  backend: NIXL
  max_tokens_in_buffer: 2048
```

```bash
export TRTLLM_NIXL_KVCACHE_BACKEND=LIBFABRIC
```

Keep the TensorRT LLM Python package, native libraries, NIXL plugin, and Dynamo
runtime (if used) from compatible container/image builds. Mixing Python from one
TensorRT LLM release with native libraries from another can surface as Python
signature errors or native ABI failures before KV-cache transfer starts.

## AWS and Kubernetes prerequisites

Before launching TensorRT LLM workers, verify the EFA layer independently:

- Use EFA-enabled instance types in subnets and Availability Zones where those
  instances are available.
- Install EFA host components on the nodes. EKS-optimized accelerated AMIs
  include the EFA host components; custom AMIs must install them explicitly.
- Install the EFA Kubernetes device plugin or use an EKS node group that deploys
  it automatically. The plugin advertises EFA devices as the
  `vpc.amazonaws.com/efa` extended resource.
- If the NVIDIA Kubernetes device plugin is also installed, disable its MOFED
  device mounting so the EFA plugin owns `/dev/infiniband` allocation.
- For custom AMIs, Bottlerocket, or any deployment that must explicitly align
  EFA and GPU locality, use the EFA DRA driver with the NVIDIA DRA driver and
  `matchAttribute` constraints, such as `resource.kubernetes.io/pcieRoot`,
  before enabling GPU Direct RDMA. Do not run the EFA DRA driver on nodes where
  the EFA device plugin is running.
- Configure huge pages for workloads that use EFA. AWS EFA nodes pre-allocate
  2 MiB huge pages, which can be requested from Pods.

Useful AWS references:

- EKS EFA device management:
  <https://docs.aws.amazon.com/eks/latest/userguide/device-management-efa.html>
- EKS EFA node group walkthrough:
  <https://docs.aws.amazon.com/eks/latest/userguide/node-efa.html>

## Container prerequisites

The worker image must contain the components needed by the selected transport:

- TensorRT LLM built with NIXL support.
- AWS EFA software and libfabric in the runtime image, normally under
  `/opt/amazon/efa`.
- The NIXL LIBFABRIC plugin, `libplugin_LIBFABRIC.so`, available to the process.
  If it is not in the default plugin search path, set `NIXL_PLUGINS_DIR` to the
  directory that contains the version-matched plugin. See the
  [LIBFABRIC Backend Setup](../../../examples/disaggregated/README.md#libfabric-backend-setup)
  notes for the two supported paths: rebuild NIXL with libfabric and hwloc, or
  provide a pre-compiled plugin that matches the NIXL version in the image.
- GDRCopy and GPU Direct RDMA support when GPU memory should be registered
  directly instead of bouncing through CPU memory.

If libfabric is installed in a non-standard location, also make the libraries
visible to the dynamic loader, for example:

```bash
export LD_LIBRARY_PATH=/opt/amazon/efa/lib64:${LD_LIBRARY_PATH}
```

## Pod configuration checklist

A minimal worker Pod needs both GPU and EFA resources. The exact values depend
on the instance type and parallelism plan.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: trtllm-disagg-worker
spec:
  containers:
    - name: worker
      image: <your-trtllm-efa-image>
      resources:
        limits:
          nvidia.com/gpu: 1
          vpc.amazonaws.com/efa: 1
          hugepages-2Mi: 512Mi
          memory: 64Gi
        requests:
          nvidia.com/gpu: 1
          vpc.amazonaws.com/efa: 1
          hugepages-2Mi: 512Mi
          memory: 64Gi
      securityContext:
        capabilities:
          add:
            - IPC_LOCK
            - SYS_RESOURCE
      env:
        - name: TRTLLM_NIXL_KVCACHE_BACKEND
          value: LIBFABRIC
        - name: FI_PROVIDER
          value: efa
        - name: FI_EFA_USE_DEVICE_RDMA
          value: "1"
        # Set only when the image stores libplugin_LIBFABRIC.so outside
        # NIXL's default plugin search path. The directory must contain the
        # version-matched LIBFABRIC plugin for this image.
        - name: NIXL_PLUGINS_DIR
          value: <directory-containing-libplugin_LIBFABRIC.so>
```

Production deployments normally add anti-affinity so context and generation
workers land on different EFA-capable nodes. This avoids validating only an
intra-node path when the target workload is cross-node disaggregated serving.

## Launch pattern

Use the same TensorRT LLM disaggregated-serving topology described in
[Disaggregated Serving](disagg-serving.md), with NIXL selected in both context
and generation worker configs. Example worker config:

```yaml
cache_transceiver_config:
  backend: NIXL
  max_tokens_in_buffer: 2048
```

Start context workers, generation workers, and the disaggregated orchestrator as
usual. For Dynamo, use the Dynamo TensorRT LLM backend configuration and set the
same environment variables on the prefill/decode workers.

## Verification

Do not rely on the environment variable alone. Verify that the runtime selected
LIBFABRIC and that EFA is visible inside the worker Pod.

```bash
kubectl describe node <node-name> | grep -A3 vpc.amazonaws.com/efa
kubectl exec <worker-pod> -- ls -l /dev/infiniband
kubectl logs <worker-pod> | grep "NixlTransferAgent::NixlTransferAgent using NIXL backend: LIBFABRIC"
```

The backend-selection log line is emitted at INFO level. If the worker logs show
`Unsupported NIXL backend: ..., fallback to UCX` or a final backend of UCX, the
backend value was unset, misspelled, used the wrong case, or was set after the
process started; fix `TRTLLM_NIXL_KVCACHE_BACKEND` and restart the worker.

For a loaded plugin check, inspect worker process mappings without first finding
the PID. Run the glob through the container shell so it expands inside the Pod:

```bash
kubectl exec <worker-pod> -- sh -c 'grep -l libplugin_LIBFABRIC /proc/*/maps'
```

If the worker aborts with `Failed to create NIXL backend: LIBFABRIC`, the
LIBFABRIC backend was selected but could not be created. Check
`NIXL_PLUGINS_DIR`, `LD_LIBRARY_PATH`, and that `libplugin_LIBFABRIC.so` was
built for the same NIXL version as the runtime image.

## Troubleshooting

| Symptom | Likely cause | Check |
| --- | --- | --- |
| Pod cannot schedule | EFA resource is not registered or requested count is too high | `kubectl describe node` and `vpc.amazonaws.com/efa` capacity |
| Worker uses UCX despite intended LIBFABRIC | Backend value is unset, misspelled, wrong-case, or set after process start | Worker log line `Unsupported NIXL backend: ..., fallback to UCX` and final backend-selection log |
| Worker aborts at startup with `Failed to create NIXL backend: LIBFABRIC` | LIBFABRIC plugin is missing, not discoverable, or version-mismatched | `NIXL_PLUGINS_DIR`, `LD_LIBRARY_PATH`, and plugin/NIXL version match |
| `/dev/infiniband` is missing | EFA device plugin is absent or another device plugin mounted devices incorrectly | EFA plugin DaemonSet and NVIDIA MOFED setting |
| Transfer uses CPU bounce buffers | GPU Direct RDMA path is unavailable | EFA/GDRCopy install, driver compatibility, and worker logs |
| Python signature or native ABI error at startup | Mixed TensorRT LLM Python/native library versions | Rebuild one image from a single TensorRT LLM release |

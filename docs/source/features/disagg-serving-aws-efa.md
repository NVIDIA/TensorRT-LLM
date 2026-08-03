# AWS EFA and LIBFABRIC for Disaggregated Serving

This guide is a deployment checklist for running TensorRT LLM disaggregated
serving on AWS EFA with NIXL using the LIBFABRIC backend. It complements the
main [Disaggregated Serving](disagg-serving.md) guide; start there for the
context/generation server topology and then apply the AWS-specific checks here.

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
  directory that contains it.
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
        - name: NIXL_PLUGINS_DIR
          value: /opt/nvidia/nvda_nixl/lib/plugins
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
kubectl logs <worker-pod> | grep -iE "NIXL.*backend|Backend.*instantiated|LIBFABRIC"
```

For a loaded plugin check, inspect the worker process mappings:

```bash
kubectl exec <worker-pod> -- bash -lc '
  grep libplugin_LIBFABRIC /proc/$(pgrep -f "trtllm|dynamo" | head -1)/maps
'
```

If the logs show UCX instead of LIBFABRIC, check `TRTLLM_NIXL_KVCACHE_BACKEND`,
`NIXL_PLUGINS_DIR`, and `LD_LIBRARY_PATH`, then restart the worker so NIXL is
initialized with the corrected environment.

## Troubleshooting

| Symptom | Likely cause | Check |
| --- | --- | --- |
| Pod cannot schedule | EFA resource is not registered or requested count is too high | `kubectl describe node` and `vpc.amazonaws.com/efa` capacity |
| Worker falls back to UCX | LIBFABRIC plugin not found or backend env var not set at process start | Worker logs and `NIXL_PLUGINS_DIR` |
| `/dev/infiniband` is missing | EFA device plugin is absent or another device plugin mounted devices incorrectly | EFA plugin DaemonSet and NVIDIA MOFED setting |
| Transfer uses CPU bounce buffers | GPU Direct RDMA path is unavailable | EFA/GDRCopy install, driver compatibility, and worker logs |
| Python signature or native ABI error at startup | Mixed TensorRT LLM Python/native library versions | Rebuild one image from a single TensorRT LLM release |

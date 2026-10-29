<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Disaggregated Serving

The execution method of disaggregated serving relies on the `trtllm-serve` command. Specifically, compared to the standard usage of `trtllm-serve`, serving requires running this command multiple times to separately start the router and workers (including context and generation) serving components. This document focuses on this approach and provides a detailed guide on how to use it.

Please note that disaggregated serving is currently an experimental feature, so the usage described in this document may change in the future.

## Startup Procedure

### Configuration File

Each worker takes its settings from a `--config` YAML. Disaggregated serving needs
`cache_transceiver_config.backend` set on every worker; without it the KV cache transceiver
is disabled and the worker rejects disaggregated requests.

These two files are the minimum that runs, and are used by the sections below:

```yaml
# ctx_config.yaml
disable_overlap_scheduler: True
cache_transceiver_config:
  backend: NIXL
```

```yaml
# gen_config.yaml
cache_transceiver_config:
  backend: NIXL
```

These examples omit `max_tokens_in_buffer` and leave the runtime at `auto`.
On the standard PyTorch model-loading path, `backend: DEFAULT` with runtime
`auto` normally selects NIXL/Python when no legacy backend selector or
incompatible configuration intervenes. Keep `DEFAULT`/`auto` when the test
or recipe should follow the supported defaults rather than exercise a
particular implementation.

Keep `cache_transceiver_config` identical in the two files. For the remaining fields, see
[Disaggregated Serving](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/disagg-serving.md).

## NIXL Backend Configuration

With `cache_transceiver_config.backend: NIXL`, the `TRTLLM_NIXL_KVCACHE_BACKEND` environment variable selects the underlying NIXL backend for KV cache exchange. The NIXL UCX backend is distinct from TRT-LLM's direct UCX backend, selected with `cache_transceiver_config.backend: UCX`.

**Supported NIXL backends:**
- **UCX** (default)
- **LIBFABRIC** (available from v0.16.0)

If an unsupported NIXL backend is specified, TRT-LLM selects the NIXL UCX backend.

### LIBFABRIC Backend Setup

**Note:** The NIXL-LIBFABRIC plugin is always built against libfabric 1.21.0, the same version used by NIXL upstream, to keep backward compatibility (see `docker/common/install_nixl.sh`). At runtime, containers built from NGC PyTorch base images use the AWS EFA installation at `/opt/amazon/efa`. Set `TRTLLM_NIXL_KVCACHE_BACKEND=LIBFABRIC` to select this backend. Other base images fall back to system libfabric detection if available. Use the installation options below if the plugin is missing or you need a custom libfabric installation.

#### Prerequisites

##### For LIBFABRIC Backend

**Required Dependencies:**

**Libfabric**
- Reuse the AWS EFA installation bundled in NGC PyTorch base images at `/opt/amazon/efa`.
- For a custom libfabric installation, see [https://ofiwg.github.io/libfabric/](https://ofiwg.github.io/libfabric/)
- **Minimum required version:** v1.21.0
- For EFA-enabled AWS instances using a non-bundled version, install through the [AWS EFA installer](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html) (recommend using the latest version)

**hwloc**
- NGC PyTorch container builds reuse the hwloc development files bundled with HPC-X under `/usr/local/mpi`. When the bundled EFA and hwloc development files are present, `docker/common/install_nixl.sh` prepends `/usr/local/mpi/lib/pkgconfig` to `PKG_CONFIG_PATH`; no extra `libhwloc-dev` installation is needed for this configuration.
- hwloc is used to understand the underlying architecture to optimize application performance
- For other base images or custom MPI installations, provide compatible hwloc development files and ensure their `hwloc.pc` is discoverable through `PKG_CONFIG_PATH`.

**Network Hardware Requirements:**
- AWS EFA (Elastic Fabric Adapter), with compatible KV cache settings and dependency versions as described above

##### For the NIXL UCX Backend

UCX is typically pre-installed in NVIDIA GPU containers. No additional installation is usually required.

#### Installation Options (missing plugin or custom libfabric)

Skip these steps if your container already includes the NIXL LIBFABRIC plugin for the libfabric installation you intend to use.

##### Option 1: Rebuild NIXL with a Custom LIBFABRIC Installation

1. **Install libfabric dependencies:**
   - Follow the installation instructions from the links above based on your system

2. **Provide hwloc development files:**
   - Reuse the hwloc bundled with your MPI installation when available, and ensure its `hwloc.pc` is discoverable through `PKG_CONFIG_PATH`. Otherwise, install a compatible hwloc development package before rebuilding NIXL.

3. **Reinstall NIXL after installing libfabric:**
   - After installing libfabric and hwloc, you must rebuild NIXL to generate the LIBFABRIC plugin against your custom path
   - You can base your installation on the TensorRT LLM NIXL installation script located at `docker/common/install_nixl.sh`, which uses libfabric v1.21.0 or leaves `-Dlibfabric_path` empty for system detection
   - For a custom installation, set `LIBFABRIC_INSTALL_PATH=/path/to/libfabric` immediately before the script's `meson setup` command, keeping the other Meson options unchanged
   - For more details, see the [NIXL LIBFABRIC Plugin documentation](https://github.com/ai-dynamo/nixl/tree/6ee64753605b3110f8ef96c7cfc2f1315675c9c7/src/plugins/libfabric#nixl-libfabric-plugin)

##### Option 2: Use Pre-compiled LIBFABRIC Plugin

If you have a pre-compiled `libplugin_LIBFABRIC.so` that matches your NIXL version:

1. Place the plugin file in a directory of your choice
2. Set the environment variable to point to the plugin directory:
   ```bash
   export NIXL_PLUGIN_DIR=/path/to/plugin/directory
   export TRTLLM_NIXL_KVCACHE_BACKEND=LIBFABRIC
   ```
3. Ensure the plugin was built with the same NIXL version as in your container

### NIXL Configuration Examples

To use NIXL for KV cache exchange, configure the `cache_transceiver_config` with `backend: NIXL`. The underlying NIXL backend (UCX or LIBFABRIC) is selected via the `TRTLLM_NIXL_KVCACHE_BACKEND` environment variable.

**Context server configuration:**
```yaml
# context_config_nixl.yml
disable_overlap_scheduler: True
cache_transceiver_config:
  backend: NIXL
```

**Generation server configuration:**
```yaml
# gen_config_nixl.yml
cache_transceiver_config:
  backend: NIXL
```

#### Example 1: Using NIXL with UCX backend (default)

```bash
# UCX is the default, but can be explicitly set
export TRTLLM_NIXL_KVCACHE_BACKEND=UCX  # Optional, UCX is default

# Start Context servers with NIXL using UCX
CUDA_VISIBLE_DEVICES=0 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --host localhost --port 8001 --backend pytorch \
  --config ./context_config_nixl.yml &> log_ctx_0 &

CUDA_VISIBLE_DEVICES=1 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --host localhost --port 8002 --backend pytorch \
  --config ./context_config_nixl.yml &> log_ctx_1 &

# Start Generation server with NIXL using UCX
CUDA_VISIBLE_DEVICES=2 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --host localhost --port 8003 --backend pytorch \
  --config ./gen_config_nixl.yml &> log_gen_0 &
```

#### Example 2: Using NIXL with LIBFABRIC backend

```bash
# Configure NIXL to use LIBFABRIC backend
export TRTLLM_NIXL_KVCACHE_BACKEND=LIBFABRIC

# If using pre-compiled plugin:
# export NIXL_PLUGIN_DIR=/path/to/plugin/directory

# For AWS EFA (optional):
# export FI_PROVIDER=efa
# export FI_EFA_USE_DEVICE_RDMA=1
# export FI_LOG_LEVEL=warn

# Start Context servers with NIXL using LIBFABRIC
CUDA_VISIBLE_DEVICES=0 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --host localhost --port 8001 --backend pytorch \
  --config ./context_config_nixl.yml &> log_ctx_0 &

CUDA_VISIBLE_DEVICES=1 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --host localhost --port 8002 --backend pytorch \
  --config ./context_config_nixl.yml &> log_ctx_1 &

# Start Generation server with NIXL using LIBFABRIC
CUDA_VISIBLE_DEVICES=2 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --host localhost --port 8003 --backend pytorch \
  --config ./gen_config_nixl.yml &> log_gen_0 &
```

### Environment Variables for NIXL Backends

**NIXL Backend Selection:**
- `TRTLLM_NIXL_KVCACHE_BACKEND`: Selects the underlying backend for NIXL. Valid options:
  - `UCX` (default)
  - `LIBFABRIC` (available from v0.16.0)
  - If an unsupported value is provided, NIXL automatically falls back to UCX

**Additional Environment Variables by Backend:**

**For UCX backend:**
- `UCX_MAX_RNDV_RAILS`: Maximum number of InfiniBand NIC devices per GPU. Setting to 1 can reduce contention in multi-GPU scenarios
- Standard UCX environment variables apply

**For LIBFABRIC backend:**
- `NIXL_PLUGIN_DIR`: Directory containing the NIXL LIBFABRIC plugin (`libplugin_LIBFABRIC.so`) if using pre-compiled plugin
- `FI_PROVIDER`: Specifies the libfabric provider to use (e.g., `efa` for AWS EFA)
- `FI_EFA_USE_DEVICE_RDMA`: Set to `1` to enable GPU Direct RDMA on AWS EFA (if supported)
- `FI_LOG_LEVEL`: Controls libfabric logging verbosity (e.g., `warn`, `info`, `debug`)

**Example configuration for AWS EFA with LIBFABRIC:**
```bash
export TRTLLM_NIXL_KVCACHE_BACKEND=LIBFABRIC
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export FI_LOG_LEVEL=warn
```

### Basic Usage

For non-SLURM clusters - particularly in single-node, multi-GPU setups, it is recommended to use standard mode. In such cases, the system does not enforce limits on process creation or termination.

Suppose we have three CUDA devices on the same machine. The first two devices are used to launch one context model each, and the third device is used to launch one generation model. In this case, the following commands need to be executed.

```bash
# Start context servers
CUDA_VISIBLE_DEVICES=0 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --host localhost --port 8001 \
    --config ./ctx_config.yaml &> log_ctx_0 &

CUDA_VISIBLE_DEVICES=1 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --host localhost --port 8002 \
    --config ./ctx_config.yaml &> log_ctx_1 &

# Start generation server
CUDA_VISIBLE_DEVICES=2 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --host localhost --port 8003 \
    --config ./gen_config.yaml &> log_gen_0 &
```

Once the context and generation servers are launched, you can launch the disaggregated
server, which will accept requests from clients and do the orchestration between context
and generation servers. The disaggregated server can be launched with:

```bash
# Start proxy
trtllm-serve disaggregated -c disagg_config.yaml
```

where `disagg_config.yaml` contains information about the context and generation servers. For the current example,
it would look like:

```yaml
# disagg_config.yaml

hostname: localhost
port: 8000
backend: pytorch
internal_request_auth_key: <shared-secret>
# Optional: HTTP keep-alive timeout in seconds (default: 10), applied to the
# client-facing listener and to the coordinator's when it runs in-process.
# Raise it when clients hold large idle connection pools and hit
# "Connection reset by peer" on a reused connection.
# server_keep_alive_timeout: 3600
context_servers:
  num_instances: 2
  urls:
      - "localhost:8001"
      - "localhost:8002"
generation_servers:
  num_instances: 1
  urls:
      - "localhost:8003"
```

Clients can then send requests to the disaggregated server at `localhost:8000`, which is an OpenAI API compatible endpoint.
Set `internal_request_auth_key` to the same non-empty secret in the disaggregated
server config and in each context/generation worker config. The proxy signs internal
generation requests with this key, and generation workers reject request-supplied
disaggregated metadata such as `ctx_info_endpoint` or `encoded_opaque_state` when
the internal handoff does not carry a valid signature.


#### Sending requests to the disaggregated server

Once the context, generation and disaggregated servers are launched, you can send requests to the disaggregated server using curl:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "prompt": "NVIDIA is a great company because",
        "max_tokens": 16,
        "temperature": 0
    }' -w "\n"
```

Or using the provided client parsing the prompts from a file and sending request to the disaggregated server specified in the `disagg_config.yaml` file at the `chat` endpoint:

```
python3 ./clients/disagg_client.py -c disagg_config.yaml -p ./clients/prompts.json -e chat
```

### Launching disaggregated servers on SLURM clusters

To simplify usage, TensorRT-LLM internally relies on MPI spawning processes. However, some clusters do not offer such process flexibility. In these cases, we provide the `trtllm-llmapi-launch` tool to launch all processes at once. Therefore, when using TensorRT-LLM on a Slurm cluster, please refer to the following method.

#### Single-Node Execution

After starting the node and entering interactive mode, you can run the following command to prevent process spawning.

```bash
# Start context servers
CUDA_VISIBLE_DEVICES=0 trtllm-llmapi-launch trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --host localhost --port 8001 \
    --config ./ctx_config.yaml &> log_ctx_0 &

CUDA_VISIBLE_DEVICES=1 trtllm-llmapi-launch trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --host localhost --port 8002 \
    --config ./ctx_config.yaml &> log_ctx_1 &

# Start generation server
CUDA_VISIBLE_DEVICES=2 trtllm-llmapi-launch trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --host localhost --port 8003 \
    --config ./gen_config.yaml &> log_gen_0 &

# Start proxy
trtllm-llmapi-launch trtllm-serve disaggregated -c disagg_config.yaml
```

#### Multi-Node Execution

If the model you are running cannot fit within a single node and requires multiple nodes,
we introduce the startup method using [srun](https://slurm.schedmd.com/srun.html) to run parallel jobs.

```bash
srun -A <account> -p <partition> -t <time> -N <num_nodes> --ntasks-per-node=<tasks_per_node> \
    --container-image=<container_image> \
    --container-mounts=<mount_paths> \
    --mpi=<mpi_type> \
    bash -c '<your_command>'
```

When using `srun`, the `-N` and `--ntasks-per-node` options are two critical parameters that
determine how your job is distributed across the cluster.

- `-N <num_nodes>`: Specifies how many physical nodes to use.
- `--ntasks-per-node=<num_tasks>`: Specifies how many tasks to run on each node.

Together, they define the total number of tasks your job will run:

$$
\text{Total tasks} = N \times \text{ntasks-per-node}
$$

Therefore, the command can be written as follows:

```bash
# The `container_image` must have the TensorRT-LLM wheel package pre-installed.
# Once the task is successfully launched, an API service will be available externally at http://host_ip:PORT.
# Launch a context with `tp_size=8` using two 4-GPU nodes.
srun -A <account> -p <partition> -t <time> \
    -N 2 --ntasks-per-node=4 \
    --container-image=<container_image> \
    --container-mounts=<mount_paths> \
    --mpi=pmix \
    bash -c "trtllm-llmapi-launch trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tp_size 8 --host 0.0.0.0 --port $PORT --config $WORK/ctx_config.yaml"

# Launch a generation with `tp_size=4` using one 4-GPU node.
srun -A <account> -p <partition> -t <time> \
    -N 1 --ntasks-per-node=4 \
    --container-image=<container_image> \
    --container-mounts=<mount_paths> \
    --mpi=pmix \
    bash -c "trtllm-llmapi-launch trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tp_size 4 --host 0.0.0.0 --port $PORT --config $WORK/gen_config.yaml"

# Launch a proxy.
# The above-mentioned value needs to be replaced with the IP address of the host machine accessible to external
# clients, and filled in the `disagg_config.yaml` file.
srun -A <account> -p <partition> -t <time> \
    -N 1 --ntasks-per-node=1 \
    --container-image=<container_image> \
    --container-mounts=<mount_paths> \
    --mpi=pmix \
    bash -c "trtllm-llmapi-launch trtllm-serve disaggregated -c $WORK/disagg_config.yaml"
```

Additionally, we offer a fully executable script—please refer to [Disaggregated SLURM Scripts](./slurm/simple_example/).

### Kubernetes Deployment with AWS EFA

LIBFABRIC backend is particularly useful for Kubernetes deployments on AWS with EFA (Elastic Fabric Adapter) for high-performance networking between pods in disaggregated serving.

#### Prerequisites

- Kubernetes cluster with GPU nodes and EFA support
- TensorRT-LLM container with wheel package pre-installed

#### Deployment Steps

##### 1. Configure Pod Resources

When deploying on Kubernetes with EFA, ensure proper resource allocation in your pod specification:

```yaml
resources:
  limits:
    nvidia.com/gpu: 2           # Number of GPUs for this pod
    vpc.amazonaws.com/efa: 4    # Number of EFA network interfaces
```

##### 2. Install EFA Libraries if Missing

Skip this step if the container already includes AWS EFA at `/opt/amazon/efa`. Otherwise, install the EFA libraries before rebuilding NIXL:

```bash
# Install AWS EFA library (required for LIBFABRIC with EFA)
curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz
tar -xf aws-efa-installer-latest.tar.gz
cd aws-efa-installer && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify
```

##### 3. Rebuild NIXL with EFA Support if Needed

Skip this step if the NIXL LIBFABRIC plugin was already built against the container's libfabric version.

##### 4. Configure and Launch Services

Use ConfigMaps to manage configurations for disaggregated serving:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: disagg-config
data:
  context.yaml: |
    disable_overlap_scheduler: true
    cache_transceiver_config:
      backend: NIXL
  generation.yaml: |
    cache_transceiver_config:
      backend: NIXL
```

Launch services:

```bash
# For context servers
TRTLLM_NIXL_KVCACHE_BACKEND=LIBFABRIC \
trtllm-serve <model> \
    --host localhost --port 8001 \
    --config /configs/context.yaml

# For generation servers
TRTLLM_NIXL_KVCACHE_BACKEND=LIBFABRIC \
trtllm-serve <model> \
    --host localhost --port 8002 \
    --config /configs/generation.yaml

# For disaggregated proxy server
trtllm-serve disaggregated -c disagg_config.yaml
```

## Mixed Precision Context and Generation

In disaggregated serving, the context workers and generation workers have different performance characteristics: context workers are compute-bound while generation workers are memory-bound. Therefore, it may be beneficial to run context workers and generation workers in different precisions.

### Prerequisites

To enable mixed precision serving, you will need:
1. A quantized checkpoint created with [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)
2. The original unquantized checkpoint (Can also be quantized)
3. Both checkpoints must use the same KV cache dtype to ensure compatibility during transfer

### Example (BF 16 Ctx, FP 8 Gen)

A quantized checkpoint can be created using `--kv_cache_qformat none`.

```bash
python $MODELOPT_ROOT/examples/llm_ptq/hf_ptq.py \
    --pyt_ckpt_path=meta-llama/Llama-3.1-8B-Instruct \
    --export_path=./weights/Llama-3.1-8B-Instruct-FP8-KV-BF16 \
    --sparsity_fmt=dense \
    --qformat=fp8 \
    --calib_size=512 \
    --batch_size=8 \
    --inference_tensor_parallel=1 \
    --inference_pipeline_parallel=1 \
    --kv_cache_qformat none \
    --export_fmt=hf
```

Verify both checkpoints have the same KV cache dtype by checking `hf_quant_config.json`.

```bash
# Start context servers with original BF16 checkpoint
CUDA_VISIBLE_DEVICES=0 trtllm-serve meta-llama/Llama-3.1-8B-Instruct \
    --host localhost --port 8001 \
    --server_role CONTEXT \
    --config ./ctx_config.yaml \
    --metadata_server_config_file ./metadata_config.yaml &> log_ctx_0 &

CUDA_VISIBLE_DEVICES=1 trtllm-serve meta-llama/Llama-3.1-8B-Instruct \
    --host localhost --port 8002 \
    --server_role CONTEXT \
    --config ./ctx_config.yaml \
    --metadata_server_config_file ./metadata_config.yaml &> log_ctx_1 &

# Start generation server with FP8 quantized checkpoint
CUDA_VISIBLE_DEVICES=2 trtllm-serve ./weights/Llama-3.1-8B-Instruct-FP8-KV-BF16 \
    --host localhost --port 8003 \
    --server_role GENERATION \
    --config ./gen_config.yaml \
    --metadata_server_config_file ./metadata_config.yaml &> log_gen_0 &

# Start disaggregated server
trtllm-serve disaggregated -c disagg_config.yaml -m ./metadata_config.yaml
```

You can also run FP8 for context and BF16 for generation, as long as the KV-cache dtype is consistent across all workers.

## Dynamic scaling 
  
### Service discovery method

Disaggregated server also supports dynamic service-discovery and auto-scaling of context/generation servers. This can be achieved by setting `disagg_cluster` section in the configurations of both context/generation servers and disagg-server. In this case, the context/generation servers must include an extra command line of `--server-role=[context|generation]`, also the `context/genration_servers` section of disaggregated server must be removed. You can simplify context/generation servers' config section by only passing `--disagg_cluster_uri=<disagg_cluster_uri>` in the command line (but disaggregated server's config must have this section). The omitted fields will use the defaults shown below. 

```yaml
disagg_cluster:
  cluster_uri: <your_cluster_uri>
  cluster_name: ""
  minimal_instances: 
    context_servers: 1
    generation_servers: 1
  heartbeat_interval_sec: 5
  inactive_interval_sec: 10
```
- `cluster_uri`: the http address of disagg-server like `http://<your-disagg-server-host>:<your-disagg-server-port>` or a pre-configured Etcd server address like `etcd://<your-etcd-host>:2379`.
- `cluster_name` : optional namespace to isolate multiple disagg-clusters in Etcd.
- `minimal_instances`: the equivalence of `num_instances` in the auto-scaling concept, disagg-server will reject requests when 
the active context/generation servers is below the corresponding threshold.
- `heartbeat_interval_sec`: frequency at which context/generation servers send heartbeats to the disagg-server.
- `inactive_interval_sec`: A server is marked inactive if no heartbeat is received within this interval (set higher than the heartbeat interval).

Note that the disaggregated server and all the context/generation servers should have the same `disagg_cluster` configuration values, or the disaggregated server may not be able to keep alive or detect inactivity the other servers properly. If `disagg_cluster` section is specified, 
the disaggregated server and all context/generation worker configs should also set the
same `internal_request_auth_key` value.

Additionally, we offer a fully executable script—please refer to [Disaggregated SLURM Scripts](./slurm/service_discovery_example/).

#### Dynamically adding servers

To add servers dynamically, you can start more context/generation workers with the same `disagg_cluster`, then the disaggregated server can discover the new servers and dispatch requests to them automatically. If a context/generation server becomes inactive, the disaggregated server will also detect this and stop routing requests to it.


### Metadata server method (Prototype)

Currently, trtllm supports dynamic addition and removal of servers by leveraging ETCD. To enable this feature, you should start the context and generation servers with an additional flag ```--metadata_server_config_file``` and ```--server_role```.
Before launching the context and generation servers, you should first start the ETCD server. By default, the ETCD server listens for client requests at ```localhost:2379```.

```bash
etcd
```

After this, you can enable the dynamic scaling feature for the use case above as follows:

```bash
# Context servers
CUDA_VISIBLE_DEVICES=0 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --port 8001  --server_role CONTEXT --config ./ctx_config.yaml --metadata_server_config_file ./metadata_config.yaml &> log_ctx_0 &
CUDA_VISIBLE_DEVICES=1 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --port 8002  --server_role CONTEXT --config ./ctx_config.yaml --metadata_server_config_file ./metadata_config.yaml &> log_ctx_1 &

# Generation servers
CUDA_VISIBLE_DEVICES=2 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --port 8003  --server_role GENERATION --config ./gen_config.yaml --metadata_server_config_file ./metadata_config.yaml &> log_gen_0 &
```

As for the disaggregated server, you should also specify the --metadata_server_config_file like the following

```bash
trtllm-serve disaggregated -c disagg_config.yaml -m ./metadata_config.yaml
```

The metadata_config file looks like
```yaml
hostname: "localhost"
port: 2379
health_check_timeout: 5.0
refersh_interval: 10.0
```

The ```hostname``` and ```port``` must match those used when starting the ETCD server. The ```health_check_timeout``` parameter specifies how long a server will be considered dead if no healthy response is received. By default, trtllm will perform two checks before marking a server as dead. The ```refresh_interval``` parameter determines how often the latest server list is fetched from the ETCD server.

#### Dynamically adding servers

Users can add servers by directly launching them with trtllm-serve. For example, you can start an additional generation server as follows:

```bash
CUDA_VISIBLE_DEVICES=3 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --host localhost --port 8004 \
     --server_role GENERATION \
    --config ./gen_config.yaml \
    --metadata_server_config_file ./metadata_config.yaml &> log_gen_0 &
```
TensorRT LLM will automatically register any newly launched server with the ETCD server, allowing the router to send new requests to the added server.

### Dynamically removing servers

When removing servers, special attention is required in the current version. You need to first remove the corresponding key from the ETCD server. After you see the log message "Server xxxx is removed," you can then safely shut down the server. This part will be improved soon.

## Troubleshooting

### NIXL LIBFABRIC Backend Issues

**Q: Why does NIXL fail to use LIBFABRIC backend even when `TRTLLM_NIXL_KVCACHE_BACKEND=LIBFABRIC` is set?**

A: Check that the container includes the NIXL LIBFABRIC plugin and the dependencies for your fabric:

1. **Missing plugin**: Older or custom images without libfabric development files at build time may not include the plugin. Follow [LIBFABRIC Backend Setup](#libfabric-backend-setup) to install it.
2. **EFA devices unavailable**: If using EFA, confirm the instance supports it and the devices are passed through to the container.
3. **Custom libfabric version mismatch**: If you rebuilt NIXL against a different libfabric installation, confirm it meets the minimum required version (v1.21.0) and that the plugin was built against that same libfabric.
4. **Using a pre-compiled plugin**: If you have a compatible `libplugin_LIBFABRIC.so`, set `NIXL_PLUGIN_DIR` to point to its directory.
5. **KV cache V2 with CUDA VMM**: A plugin can load successfully while transfers fail because of dependency limitations. See [LIBFABRIC Backend Setup](#libfabric-backend-setup) for the affected versions and upstream fixes.

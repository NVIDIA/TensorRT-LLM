# Disaggregated Serving

The execution method of disaggregated serving relies on the `trtllm-serve` command. Specifically, compared to the standard usage of `trtllm-serve`, serving requires running this command multiple times to separately start the router and workers (including context and generation) serving components. This document focuses on this approach and provides a detailed guide on how to use it.

Please note that disaggregated serving is currently an experimental feature, so the usage described in this document may change in the future.

## Startup Procedure

### Configuration File

The `trtllm-serve` command supports the `extra-llm-config.yaml` parameter. In the extra LLM configuration file, the `cache_transceiver_config` field is specifically used for disaggregated service. It is mainly used to specify additional parameters required for the KV cache transmission process.

```yaml
cache_transceiver_config:
  # KV cache transmission backend. Valid options include `DEFAULT` (i.e., NIXL), `UCX`, `NIXL`.
  backend: <str>
  # KV cache buffer size. Set it ≥ the maximum ISL (Input Sequence Length) for best performance.
  max_tokens_in_buffer: <int>
  # KV cache transfer timeout in milliseconds
  # For requests, if they do not send/receive the KV cache in time they are cancelled and cleaned up
  kv_transfer_timeout_ms: <int>
```

The following is an example, consisting of the `ctx_extra-llm-api-config.yaml` and `gen_extra-llm-api-config.yaml` files needed in the sections below.

```yaml
# ctx_extra-llm-api-config.yaml

# The overlap scheduler for context servers is currently disabled, as it is
# not yet supported in disaggregated context server architectures.
disable_overlap_scheduler: True
cache_transceiver_config:
  backend: UCX
  max_tokens_in_buffer: 2048
```

```yaml
# gen_extra-llm-api-config.yaml

cache_transceiver_config:
  backend: UCX
  max_tokens_in_buffer: 2048
```

### Basic Usage

For non-SLURM clusters - particularly in single-node, multi-GPU setups, it is recommended to use standard mode. In such cases, the system does not enforce limits on process creation or termination.

Suppose we have three CUDA devices on the same machine. The first two devices are used to launch one context model each, and the third device is used to launch one generation model. In this case, the following commands need to be executed.

```bash
# Start context servers
CUDA_VISIBLE_DEVICES=0 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --host localhost --port 8001 \
    --extra_llm_api_options ./ctx_extra-llm-api-config.yaml &> log_ctx_0 &

CUDA_VISIBLE_DEVICES=1 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --host localhost --port 8002 \
    --extra_llm_api_options ./ctx_extra-llm-api-config.yaml &> log_ctx_1 &

# Start generation server
CUDA_VISIBLE_DEVICES=2 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --host localhost --port 8003 \
    --extra_llm_api_options ./gen_extra-llm-api-config.yaml &> log_gen_0 &
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
    --extra_llm_api_options ./ctx_extra-llm-api-config.yaml &> log_ctx_0 &

CUDA_VISIBLE_DEVICES=1 trtllm-llmapi-launch trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --host localhost --port 8002 \
    --extra_llm_api_options ./ctx_extra-llm-api-config.yaml &> log_ctx_1 &

# Start generation server
CUDA_VISIBLE_DEVICES=2 trtllm-llmapi-launch trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --host localhost --port 8003 \
    --extra_llm_api_options ./gen_extra-llm-api-config.yaml &> log_gen_0 &

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
    bash -c "trtllm-llmapi-launch trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tp_size 8 --host 0.0.0.0 --port $PORT --extra_llm_api_options $WORK/ctx_extra-llm-api-config.yaml"

# Launch a generation with `tp_size=4` using one 4-GPU node.
srun -A <account> -p <partition> -t <time> \
    -N 1 --ntasks-per-node=4 \
    --container-image=<container_image> \
    --container-mounts=<mount_paths> \
    --mpi=pmix \
    bash -c "trtllm-llmapi-launch trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tp_size 4 --host 0.0.0.0 --port $PORT --extra_llm_api_options $WORK/gen_extra-llm-api-config.yaml"

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


## Dynamic scaling (Prototype)

Currently, trtllm supports dynamic addition and removal of servers by leveraging ETCD. To enable this feature, you should start the context and generation servers with an additional flag ```--metadata_server_config_file``` and ```--server_role```.
Before launching the context and generation servers, you should first start the ETCD server. By default, the ETCD server listens for client requests at ```localhost:2379```.

```bash
etcd
```

After this, you can enable the dynamic scaling feature for the use case above as follows:

```bash
# Context servers
CUDA_VISIBLE_DEVICES=0 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --port 8001  --server_role CONTEXT --extra_llm_api_options ./ctx_extra-llm-api-config.yaml --metadata_server_config_file ./metadata_config.yaml &> log_ctx_0 &
CUDA_VISIBLE_DEVICES=1 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --port 8002  --server_role CONTEXT --extra_llm_api_options ./ctx_extra-llm-api-config.yaml --metadata_server_config_file ./metadata_config.yaml &> log_ctx_1 &

# Generation servers
CUDA_VISIBLE_DEVICES=2 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --port 8003  --server_role GENERATION --extra_llm_api_options ./gen_extra-llm-api-config.yaml --metadata_server_config_file ./metadata_config.yaml &> log_gen_0 &
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

### Dynamically adding servers

Users can add servers by directly launching them with trtllm-serve. For example, you can start an additional generation server as follows:

```bash
CUDA_VISIBLE_DEVICES=3 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --host localhost --port 8004 \
     --server_role GENERATION \
    --extra_llm_api_options ./gen_extra-llm-api-config.yaml \
    --metadata_server_config_file ./metadata_config.yaml &> log_gen_0 &
```
TensorRT LLM will automatically register any newly launched server with the ETCD server, allowing the router to send new requests to the added server.

### Dynamically removing servers

When removing servers, special attention is required in the current version. You need to first remove the corresponding key from the ETCD server. After you see the log message "Server xxxx is removed," you can then safely shut down the server. This part will be improved soon.

## Startup Procedure with MPI Worker (Deprecated)

In the past, we used `disaggregated_mpi_worker` to allow context nodes and generation nodes to operate within the same MPI world. However, this approach conflicts with the dynamic node addition and removal functionality. As a result, disaggregated_mpi_worker has been marked as deprecated, and the corresponding examples will be gradually removed.

```bash
mpirun -n <total_num_ranks> trtllm-serve disaggregated_mpi_worker -c disagg_config.yaml
```
where `total_num_ranks` is the sum of `TP*PP` for all context and generation servers. For the example above, `total_num_ranks` is 3
since `TP` and `PP` is 1 for the two context and one generation server.

The `disagg_config.yaml` file must now contain the configuration parameters of the context and generation servers. For example,
it could look like:

```yaml
hostname: localhost
port: 8000
model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
backend: "pytorch"
disable_overlap_scheduler: True
context_servers:
  num_instances: 2
  tensor_parallel_size: 1
  pipeline_parallel_size: 1
  kv_cache_config:
    free_gpu_memory_fraction: 0.9
  cache_transceiver_config:
    backend: UCX
  urls:
      - "localhost:8001"
      - "localhost:8002"
generation_servers:
  num_instances: 1
  tensor_parallel_size: 1
  pipeline_parallel_size: 1
  cache_transceiver_config:
    backend: UCX
  urls:
      - "localhost:8003"
```

Once the context and generation servers are launched, you can again launch the disaggregated server with

```bash
trtllm-serve disaggregated -c disagg_config.yaml
```

The MPI communication backend for KV cache transfer has been deprecated and may not be supported in the future. When using the MPI backend, the environment variable `TRTLLM_USE_MPI_KVCACHE=1` should be set to avoid conflicts between mpi4py and KV cache transfer.

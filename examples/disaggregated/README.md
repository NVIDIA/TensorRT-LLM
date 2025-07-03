# TRT-LLM Disaggregated Serving

To run TRT-LLM in disaggregated mode, you must first launch context (prefill) and generation (decode) servers using `trtllm-serve`.
Depending on your deployment environment, this can be done in different ways.

## Launching context and generation servers using multiple independent `trtllm-serve` commands

You can use multiple `trtllm-serve` commands to launch the context and generation servers that will be used
for disaggregated serving. For example, you could launch two context servers and one generation servers as follows:

```
echo -e "disable_overlap_scheduler: True\ncache_transceiver_config:\n  max_num_tokens: 2048" > context_extra-llm-api-config.yml
echo -e "cache_transceiver_config:\n  max_num_tokens: 2048" > gen_extra-llm-api-config.yml

export TRTLLM_USE_UCX_KVCACHE=1
#Context servers
CUDA_VISIBLE_DEVICES=0 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --port 8001 --backend pytorch --extra_llm_api_options ./context_extra-llm-api-config.yml &> log_ctx_0 &
CUDA_VISIBLE_DEVICES=1 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --port 8002 --backend pytorch --extra_llm_api_options ./context_extra-llm-api-config.yml &> log_ctx_1 &
#Generation servers
CUDA_VISIBLE_DEVICES=2 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --port 8003 --backend pytorch --extra_llm_api_options ./gen_extra-llm-api-config.yml &> log_gen_0 &
```
Once the context and generation servers are launched, you can launch the disaggregated
server, which will accept requests from clients and do the orchestration between context
and generation servers. The disaggregated server can be launched with:

```
trtllm-serve disaggregated -c disagg_config.yaml
```
where `disagg_config.yaml` contains information about the context and generation servers. For the current example,
it would look like:
```
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

Clients can then send requests to the disaggregated server at `localhost:8000`, which is an OpenAI compatible endpoint.

## Launching context and generation servers using MPI

One can also launch all context and generation servers using MPI. This can be done by issuing the following command:
```
export TRTLLM_USE_MPI_KVCACHE=1
mpirun -n <total_num_ranks> trtllm-serve disaggregated_mpi_worker -c disagg_config.yaml
```
where `<total_num_ranks>` is the sum of `TP*PP` for all context and generation servers. For the example above, `total_num_ranks` is 3
since `TP` and `PP` is 1 for the two context and one generation server.

The `disagg_config.yaml` file must now contain the configuration parameters of the context and generation servers. For example,
it could look like:

```
hostname: localhost
port: 8000
model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
backend: "pytorch"
use_cuda_graph: False
disable_overlap_scheduler: True
context_servers:
  num_instances: 2
  tensor_parallel_size: 1
  pipeline_parallel_size: 1
  kv_cache_config:
    free_gpu_memory_fraction: 0.9
  urls:
      - "localhost:8001"
      - "localhost:8002"
generation_servers:
  num_instances: 1
  tensor_parallel_size: 1
  pipeline_parallel_size: 1
  urls:
      - "localhost:8003"
```

Once the context and generation servers are launched, you can again launch the disaggregated server with
```
trtllm-serve disaggregated -c disagg_config.yaml
```

## Sending requests to the disaggregated server

Once the context, generation and disaggregated servers are launched, you can send requests to the disaggregated server using curl:
```
curl http://localhost:8000/v1/completions     -H "Content-Type: application/json"     -d '{
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

## Dynamic scaling with ETCD (experimental)
Currently, trtllm supports dynamic addition and removal of servers by leveraging ETCD. To enable this feature, you should start the context and generation servers with an additional flag ```--metadata_server_config_file``` and ```--server_role```.
Before launching the context and generation servers, you should first start the ETCD server. By default, the ETCD server listens for client requests at ```localhost:2379```.
```
etcd
```
After this, you can enable the dynamic scaling feature for the use case above as follows:
```
export TRTLLM_USE_UCX_KVCACHE=1
#Context servers
CUDA_VISIBLE_DEVICES=0 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --port 8001 --backend pytorch --server_role CONTEXT --extra_llm_api_options ./context_extra-llm-api-config.yml --metadata_server_config_file ./metadata_config.yml &> log_ctx_0 &
CUDA_VISIBLE_DEVICES=1 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --port 8002 --backend pytorch --server_role CONTEXT --extra_llm_api_options ./context_extra-llm-api-config.yml --metadata_server_config_file ./metadata_config.yml &> log_ctx_1 &
#Generation servers
CUDA_VISIBLE_DEVICES=2 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --port 8003 --backend pytorch --server_role GENERATION --extra_llm_api_options ./gen_extra-llm-api-config.yml --metadata_server_config_file ./metadata_config.yml &> log_gen_0 &
```
As for the disaggregated server, you should also specify the --metadata_server_config_file like the following
```
trtllm-serve disaggregated -c disagg_config.yaml -m ./metadata_config.yml
```

The metadata_config file looks like
```
hostname: "localhost"
port: 2379
health_check_timeout: 5.0
refersh_interval: 10.0
```

The ```hostname``` and ```port``` must match those used when starting the ETCD server. The ```health_check_timeout``` parameter specifies how long a server will be considered dead if no healthy response is received. By default, trtllm will perform two checks before marking a server as dead. The ```refresh_interval``` parameter determines how often the latest server list is fetched from the ETCD server.

### Dynamically adding servers
Users can add servers by directly launching them with trtllm-serve. For example, you can start an additional generation server as follows:
```
CUDA_VISIBLE_DEVICES=3 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --port 8004 --backend pytorch --server_role GENERATION --extra_llm_api_options ./gen_extra-llm-api-config.yml --metadata_server_config_file ./metadata_config.yml &> log_gen_0 &
```
Trtllm will automatically register any newly launched server with the ETCD server, allowing the router to send new requests to the added server.

### Dynamically removing servers
When removing servers, special attention is required in the current version. You need to first remove the corresponding key from the ETCD server. After you see the log message "Server xxxx is removed," you can then safely shut down the server. This part will be improved soon.


To launch context and gen servers, use:

```
export TRTLLM_USE_MPI_KVCACHE=1
mpirun --allow-run-as-root -n 2 python3 launch_disaggregated_workers.py -c disagg_config.yaml &> output_workers &
```

Then, launch the disaggregated server which will do the orchestration between context and generation servers

```
python3 launch_disaggregated_server.py -c disagg_config.yaml  &> output_disagg &
```


Once ctx, gen and disagg servers are launched, one can send requests to disagg server using curl:
```
curl http://localhost:8000/v1/completions     -H "Content-Type: application/json"     -d '{
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "prompt": "NVIDIA is a great company because",
        "max_tokens": 16,
        "temperature": 0
    }' -w "\n"
```
Or using the provided client:
```
cd client
python3 disagg_client.py -c ../disagg_config.yaml -p prompts.json
```

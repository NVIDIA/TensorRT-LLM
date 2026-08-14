# Disaggregated Serving with Ray orchestrator
TensorRT-LLM supports a prototype [Ray orchestrator](../README.md) as an alternative to MPI.

Running disaggregated serving with Ray follows [the same workflow as in MPI](/examples/disaggregated/README.md), except that `orchestrator_type="ray"` must be set on the `LLM` class, and `CUDA_VISIBLE_DEVICES` can be omitted since Ray handles GPU placement.


## Quick Start
This script is a shorthand to launch a single-GPU context and generation server, as well as the disaggregated server within a single Ray cluster. Please see [this documentation](/examples/disaggregated/README.md) for details on adjusting parallel settings.

```bash
# requires a total of two GPUs
bash -e disagg_serving_local.sh
```

KV cache transfer between the context and generation servers uses the NIXL backend by default. Pass `--transceiver_backend UCX` to use UCX instead. The C++ cache-transceiver runtime is used by default; with NIXL, pass `--transceiver_runtime PYTHON` to use the Python runtime instead:
```bash
bash -e disagg_serving_local.sh --transceiver_backend UCX
bash -e disagg_serving_local.sh --transceiver_runtime PYTHON
```

Run `bash disagg_serving_local.sh --help` for the full list of options (executor backend, model, tensor-parallel size, etc.).

Once the disaggregated server is ready, you can send requests to the disaggregated server using curl:
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

## Disclaimer
The code is a prototype and subject to change. Currently, there are no guarantees regarding functionality, performance, or stability.

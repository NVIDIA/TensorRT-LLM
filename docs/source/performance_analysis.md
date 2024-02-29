# Performance Analysis of TensorRT-LLM

NVIDIA Nsight Systems reports at the application level are highly informative. Metric sampling capabilities have increased over generations and provide a clean middle-ground between timing analysis and kernel-level deep dives with NVIDIA Nsight Compute.

Given the potential long runtimes of Large Languages Models (LLMs) and the diversity of workloads a model may experience during a single inference pass or binary execution, we have added features to TensorRT-LLM to get the most out of Nsight Systems capabilities. This document outlines those features as well as provides examples of how to best utilize them to understand your application.

# Feature Descriptions

The main functionality here:
  * Relies on toggling the CUDA profiler runtime API on and off.
  * Provides a means to understand which regions a user may want to focus on.

Toggling the CUDA profiler runtime API on and off:
  * Allows users to know specifically what the profiled region corresponds to.
  * Results in smaller files to post-process (for metric extraction or similar).

## Usage

### Inference Time Command Line Options
  * `--log_iteration_data`, for use with gptManagerBenchmark. The runtime decides the specifics of each decoder iteration launch. This option prints to stdout metadata on each decoder iteration:
```
[TensorRT-LLM][INFO] {"Active Request Count":249,"Context Requests":8,"Free KV cache blocks":0,"Generation Requests":231,"Iteration Counter":90,"Max KV cache blocks":2448,"Max Request Count":256,"MicroBatch ID":0,"Runtime CPU Memory Usage":28784,"Runtime GPU Memory Usage":540173600,"Runtime Pinned Memory Usage":0,"Scheduled Requests":239,"Timestamp":"12-13-2023 14:55:14","Tokens per KV cache block":128,"Total Context Tokens":6904,"Used KV cache blocks":2448}
```
### Inference Time Environment Variables
  * `TLLM_GPTM_PROFILE_START_STOP`, a csv of iterations to trigger start/stop for gptManagerBenchmark (corresponds to "Iteration Counter" in output above
  * `TLLM_GPTS_PROFILE_START_STOP`, a csv of static batching iteration indexes to trigger start/stop for gptSessionBenchmark

## Coordinating with NVIDIA Nsight Systems Launch

Consult the Nsight Systems User Guide for full overview of options.

Say we want to profile the context phase and the first output token computation of a model with gptSessionBenchmark.

To profile just those iterations, in addition to setting `TLLM_GPTS_PROFILE_START_STOP="0,1"`:
  * We need to tell Nsight Systems to look for explicit API triggers to profile (`-c cudaProfilerApi`)
  * We need to tell Nsight Systems to keep profiling after seeing a profile stop API call (`--capture-range-end="repeat[]"`)

# Examples
Consult the Nsight Systems User Guide for full overview of MPI-related options.

## Profiling a single IFB iteration executing on a single rank of a multi-GPU model

Say we have run once using `--log_iteration_data` and want to analyze iterations 0, 63 and 127 based on the metadata output. We also want to capture metrics at an increased resolution. To do this we create a bash file as describe in the Nsight Systems User Guide:

```bash
#!/bin/bash

# Use $PMI_RANK for MPICH and $SLURM_PROCID with srun.
if [ $OMPI_COMM_WORLD_LOCAL_RANK -eq 0 ]; then
  nsys profile -e "NSYS_MPI_STORE_TEAMS_PER_RANK=1" -t cuda,nvtx --gpu-metrics-device=${OMPI_COMM_WORLD_LOCAL_RANK} -c cudaProfilerApi --capture-range-end="repeat[]" --gpu-metrics-frequency=100000 "$@"
else
  "$@"
fi
```

We name this file `profile_rank_0.bash` and then launch our application specifying the iterations to capture:
```bash
mpirun -n 2 env TLLM_GPTM_PROFILE_START_STOP="0,63,127" ./profile_rank_0.bash ./benchmarks/gptManagerBenchmark <benchmark/model options>
```

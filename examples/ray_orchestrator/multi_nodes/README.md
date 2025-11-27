# Multi-node inference with Ray orchestrator
TensorRT-LLM supports a prototype [Ray orchestrator](../README.md) as an alternative to MPI. The following example shows how to start a Ray cluster for multi-node inference.


## Quick Start

**Prerequisite:** a container image with TensorRT-LLM preinstalled (or suitable for installing it). The examples use Slurm and [Enroot](https://github.com/NVIDIA/enroot). If you use a different setup, adapt the following scripts and commands to your multi-node environment.

1. Allocate nodes and open a shell on the head node:

    ```shell
    # e.g., 2 nodes with 8 GPUs per node
    >> salloc -t 240 -N 2 -p interactive

    >> srun --pty -p interactive bash
    ```
2. Once on the head node, launch a multi-node Ray cluster:
    ```shell
    # Remember to set CONTAINER and MOUNTS env vars or variables inside the script to your path.
    # You can add the TensorRT-LLM installation command in this script if it is not preinstalled in your container.
    >> bash -e run_cluster.sh
    ```

3. Enter the head container and run your TensorRT-LLM driver script

    Note that this step requires TensorRT-LLM to be installed in the containers on all nodes. If it isn’t, install it manually inside each node’s container.

    ```shell
    # On the head node
    >> sacct

    # Grab the Slurm step ID with Job Name "ray-head"
    >> srun --jobid=<Your Step ID> --overlap  --pty bash

    >> enroot list -f # get process id
    >> enroot exec <process id> bash

    # You can change this script to a model and parallel settings effective for multi-node inference (e.g., TP8 or TP4PP4).
    >> python examples/ray_orchestrator/llm_inference_async_ray.py
    ```

## Disclaimer
The code is a prototype and subject to change. Currently, there are no guarantees regarding functionality, performance, or stability.

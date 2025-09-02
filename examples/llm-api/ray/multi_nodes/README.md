# Multi-node inference with Ray orchestrator
TensorRT-LLM supports [Ray](https://docs.ray.io/en/latest/index.html) as an orchestrator with [PyTorch Distributed](https://docs.pytorch.org/tutorials/beginner/dist_overview.html) as an alternative to MPI. This feature is currently experimental and under active development. 

**Prerequisite:** a container image with TensorRT-LLM preinstalled (or suitable for installing it). The examples use Slurm and [Enroot](https://github.com/NVIDIA/enroot); if you use a different setup, adapt the container options and launch commands to your multi-node environment.

## Run multi-node inference with Ray 

1. Allocate nodes and open a shell on the head node:

    ```shell
    # e.g., 2 nodes with 8 GPUs per node
    >> salloc -t 240 -N 2 -p interactive

    >> srun --pty -p interactive bash
    ```
2. Once on the head node, launch a multi-node Ray cluster:
    ```shell
    # Remember to set CONTAINER and MOUNTS env vars or variables inside the script to your path.
    >> bash -e run_cluster.sh
    ```

3. Enter the head container and run your TensorRT-LLM driver script
    ```shell
    # On the head node
    >> sacct

    # Grab the Slurm step ID with Job Name "ray-head"
    >> srun --jobid=<Your Step ID> --overlap  --pty bash

    >> enroot list -f # get process id
    >> enroot exec <process id> bash

    # Under your work directory:
    >> pip install -e . # if needed
    >> python examples/ray/llm_inference_async_ray.py
    ```

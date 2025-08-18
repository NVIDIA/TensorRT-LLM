
# Prerequisites
An example of the current dev workflow. It's easier to do all commands directly on draco or other multi-node cluster instead of computelab or local.

1. Prepare an TRT-LLM image
    - Setup `enroot`'s credentials following [doc](https://confluence.nvidia.com/display/DevtechCompute/Using+EOS). You need to add credentials for `urm.nvidia.com` for pulling dev image.
    - Create an enroot image from TRT-LLM dev image:
        ```shell
        # You can add XDG_CACHE_HOME=/home/scratch.<your_username> if running out of space.
        >> enroot import -o trtllm_ray.sqfs docker://urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm:pytorch-25.05-py3-x86_64-ubuntu24.04-trt10.11.0.33-skip-tritondevel-202506271620-5539
        ```  
    - [Optional] If you prefer pre-installing your local TRT-LLM
        ```shell
        # Unpacks the trtllm_ray.sqfs image and creates a named, writable container called trtllm_ray_dev.
        >>> enroot create --name trtllm_ray_dev trtllm_ray.sqfs

        # Start container and install your in-development TRT-LLM
        >> enroot start --mount /home/scratch.<your_username>:/scratch trtllm_ray_dev bash
        >> pip install  .

        # Exit container with `exit` and then export image
        >> exit
        >> enroot export --output trtllm_ray_dev.sqfs trtllm_ray_dev
        ```


# Run multi-node
Tested on Draco cluster.


1. Allocate nodes and get into a node as the head node:

```shell
# e.g., 2 nodes with 2 GPUs per node
>> salloc -t 240 -N 2 --gres gpu:2 -A coreai_dlalgo_llm -p interactive

>> srun --pty -A coreai_dlalgo_llm -p interactive bash
```

2. Once you're on the head node, launch a multi-node Ray cluster:
```shell
# Change CONTAINER and MOUNTS inside the script to your path
# This will only set up Ray cluster and connected nodes
>> bash -e run_cluster.sh
```

3. Step into head container and run driver
```shell
# on login node
>> sacct

# Grab the Slurm step ID with Job Name "ray-head"
# Example:  "5364209.1      ray-head            coreai_dl+          8    RUNNING      0:0"
>> srun --jobid=<Your Step ID> --overlap  --pty bash

# sanity check: you should only see GPUs you allocated
# >> nvidia-smi

>> enroot list -f # get process id
>> enroot exec <process id> bash

# Under <Your workdir>/tekit:
# Remember that worker container(s) need this installation as well
>> pip install -e .
>> python <your ray driver script>

```

4. (Alternative) launch multi-node cluster w/ TRTLLM driver command directly:
```shell
# Don't use this path yet
# >> bash -e run_cluster.sh -- python <file path to>/simple_ray_single_node.py
```

# Troubleshoots
## 1. Cluster not cleanup resource properly
During development, if you control-x kill driver script, Ray might not clean up actors and/or placement groups correctly. e.g., if you `ray list actors` on head container, you might see actors stuck at `PENDING_CREATION` or placement group at `CREATED` despite driver script has been killed.

Tentative WAR is `python ../utils/cleanup.py` so that you don't have to restart a Ray cluster in order to run driver script.

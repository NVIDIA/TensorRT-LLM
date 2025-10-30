# Layer-wise Benchmarks

## Generate profiles

### Run with MPI

**Step 1:** Start a container using Docker, Enroot or others. Please refer to `../../jenkins/current_image_tags.properties` for the Docker image URI.

**Step 2:** In the container, install `tensorrt_llm`:

```bash
pip install -e ../..
```

**Step 3:** In the container, run benchmarks and generate profiles:

```bash
# Run DeepSeek-R1
NP=4 ./mpi_launch.sh ./run_single.sh config_ctx.yaml
NP=4 ./mpi_launch.sh ./run_single.sh config_gen.yaml

# Run DeepSeek-V3.2-Exp
NP=4 ./mpi_launch.sh ./run_single.sh config_ctx.yaml --model deepseek-ai/DeepSeek-V3.2-Exp --tokens-per-block 64 --moe-backend DEEPGEMM
NP=4 ./mpi_launch.sh ./run_single.sh config_gen.yaml --model deepseek-ai/DeepSeek-V3.2-Exp --tokens-per-block 64 --moe-backend DEEPGEMM

# Run DeepSeek-V3.2-Exp with 32k context length
NP=4 ./mpi_launch.sh ./run_single.sh config_ctx.yaml --model deepseek-ai/DeepSeek-V3.2-Exp --tokens-per-block 64 --max-seq-len $((32768 + 1024 + 4)) --max-num-tokens $((32768 + 1024 + 4)) --moe-backend DEEPGEMM --batch-size 1 --seq-len-q 32769
NP=4 ./mpi_launch.sh ./run_single.sh config_gen.yaml --model deepseek-ai/DeepSeek-V3.2-Exp --tokens-per-block 64 --max-seq-len $((32768 + 1024 + 4)) --moe-backend DEEPGEMM --seq-len-kv-cache 32769

# Run with attention TP
NP=4 ./mpi_launch.sh ./run_single.sh config_gen.yaml --no-enable-attention-dp
NP=4 ./mpi_launch.sh ./run_single.sh config_ctx.yaml --no-enable-attention-dp

# Run with attention TP and TRTLLMGen
NP=4 TRTLLM_ENABLE_PDL=1 ./mpi_launch.sh ./run_single.sh config_ctx.yaml --no-enable-attention-dp --moe-backend TRTLLM
NP=4 TRTLLM_ENABLE_PDL=1 ./mpi_launch.sh ./run_single.sh config_gen.yaml --no-enable-attention-dp --moe-backend TRTLLM

# Run with MTP3
NP=4 ./mpi_launch.sh ./run_single.sh config_gen.yaml --batch-size 32 --seq-len-q 4

# Run 4 layers
NP=4 ./mpi_launch.sh ./run_single.sh config_ctx.yaml --layer-indices 5,6,7,8
NP=4 ./mpi_launch.sh ./run_single.sh config_gen.yaml --layer-indices 5,6,7,8

# Scale DEP=16 to 4 GPUs: reduce the number of experts, uses MNNVL A2A if applicable
NP=4 ./mpi_launch.sh ./run_single.sh config_gen.yaml --scaled-from 16 --moe-backend WIDEEP

# Scale TEP=16 to 4 GPUs: reduce the number of attention heads and experts
NP=4 ./mpi_launch.sh ./run_single.sh config_gen.yaml --scaled-from 16 --no-enable-attention-dp

# Run with DeepEP A2A
NP=4 TRTLLM_FORCE_ALLTOALL_METHOD=DeepEP ./mpi_launch.sh ./run_single.sh config_ctx.yaml --moe-backend WIDEEP
NP=4 TRTLLM_FORCE_ALLTOALL_METHOD=DeepEP ./mpi_launch.sh ./run_single.sh config_gen.yaml --moe-backend WIDEEP
```

### Run with Slurm

> Tips: If you have a running job with environment installed, please skip step 1 and 2 and go straight to step 3. In this case, your job must be run with `--container-name aaa`, and if the container name is not "layer_wise_benchmarks" please `export CONTAINER_NAME=aaa`.

**Step 1:** On the controller node, allocate one or multiple nodes, and record the `SLURM_JOB_ID`:

```bash
SLURM_JOB_ID=$(NODES=4 TIME=02:00:00 ./slurm_alloc.sh)
```

Please fill the variables in `./slurm_alloc.sh`.

**Step 2:** Start a container and install `tensorrt_llm`. Run the following command on the controller node:

```bash
SLURM_JOB_ID=$SLURM_JOB_ID ./slurm_init_containers.sh
```

It uses the image recorded in `../../jenkins/current_image_tags.properties`. The image will be downloaded to `../../enroot/` for once.

**Step 3:** Run benchmarks to generate profiles. Run the following command on the controller node, where `NODES` &le; the number of allocated nodes:

```bash
# Run DeepSeek-R1 with wide ep: uses MNNVL A2A if applicable
SLURM_JOB_ID=$SLURM_JOB_ID NODES=4 NP=16 ./slurm_launch.sh ./run_single.sh config_gen.yaml --moe-backend WIDEEP

# Run with attention TP and TRTLLMGen
SLURM_JOB_ID=$SLURM_JOB_ID NODES=4 NP=16 TRTLLM_ENABLE_PDL=1 ./slurm_launch.sh ./run_single.sh config_gen.yaml --no-enable-attention-dp --moe-backend TRTLLM

# Run with DeepEPLowLatency
SLURM_JOB_ID=$SLURM_JOB_ID NODES=4 NP=16 TRTLLM_FORCE_ALLTOALL_METHOD=DeepEPLowLatency ./slurm_launch.sh ./run_single.sh config_gen.yaml --moe-backend WIDEEP

# You can run 4-GPU and 8-GPU tasks without reallocate the slurm job
SLURM_JOB_ID=$SLURM_JOB_ID NODES=1 NP=4 ./slurm_launch.sh ./run_single.sh config_ctx.yaml
SLURM_JOB_ID=$SLURM_JOB_ID NODES=2 NP=8 ./slurm_launch.sh ./run_single.sh config_ctx.yaml
```

## Parse profiles

Coming soon.

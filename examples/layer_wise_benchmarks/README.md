# Layer-wise Benchmarks

## Generate profiles

### Run with OpenMPI

**Step 1:** Start a container using Docker, Enroot or others. Please refer to `../../jenkins/current_image_tags.properties` for the Docker image URI.

**Step 2:** In the container, install `tensorrt_llm`:

```bash
pip install -e ../..
```

**Step 3:** In the container, run benchmarks and generate profiles:

```bash
# Run DeepSeek-R1 NVFP4
NP=4 ./mpi_launch.sh ./run.sh config_ctx.yaml
NP=4 ./mpi_launch.sh ./run.sh config_gen.yaml

# Run DeepSeek-V3.2-Exp
NP=4 ./mpi_launch.sh ./run.sh config_ctx.yaml --model deepseek-ai/DeepSeek-V3.2-Exp --tokens-per-block 64 --moe-backend DEEPGEMM
NP=4 ./mpi_launch.sh ./run.sh config_gen.yaml --model deepseek-ai/DeepSeek-V3.2-Exp --tokens-per-block 64 --moe-backend DEEPGEMM

# Run DeepSeek-V3.2-Exp with 32k context length
NP=4 ./mpi_launch.sh ./run.sh config_ctx.yaml --model deepseek-ai/DeepSeek-V3.2-Exp --tokens-per-block 64 --max-seq-len $((32768 + 1024 + 4)) --moe-backend DEEPGEMM --batch-size 1 --seq-len-q 32769
NP=4 ./mpi_launch.sh ./run.sh config_gen.yaml --model deepseek-ai/DeepSeek-V3.2-Exp --tokens-per-block 64 --max-seq-len $((32768 + 1024 + 4)) --moe-backend DEEPGEMM --seq-len-kv-cache 32769

# Run with attention TP
NP=4 ./mpi_launch.sh ./run.sh config_ctx.yaml --no-enable-attention-dp
NP=4 ./mpi_launch.sh ./run.sh config_gen.yaml --no-enable-attention-dp

# Run with attention TP and TRTLLMGen
NP=4 ./mpi_launch.sh -x TRTLLM_ENABLE_PDL=1 ./run.sh config_ctx.yaml --no-enable-attention-dp --moe-backend TRTLLM --balance-method NotModified
NP=4 ./mpi_launch.sh -x TRTLLM_ENABLE_PDL=1 ./run.sh config_gen.yaml --no-enable-attention-dp --moe-backend TRTLLM --balance-method NotModified

# Run with MTP3
NP=4 ./mpi_launch.sh ./run.sh config_gen.yaml --batch-size 32 --seq-len-q 4

# Run 4 layers
NP=4 ./mpi_launch.sh ./run.sh config_ctx.yaml --layer-indices 5,6,7,8
NP=4 ./mpi_launch.sh ./run.sh config_gen.yaml --layer-indices 5,6,7,8

# Scale DEP=16 to 4 GPUs: reduce the number of experts, uses MNNVL A2A if applicable
NP=4 ./mpi_launch.sh ./run.sh config_gen.yaml --scaled-from 16 --moe-backend WIDEEP

# Scale TEP=16 to 4 GPUs: reduce the number of attention heads and experts
NP=4 ./mpi_launch.sh ./run.sh config_gen.yaml --scaled-from 16 --no-enable-attention-dp

# Run Qwen3-Next (balanced routing is not implemented)
NP=2 ./mpi_launch.sh ./run.sh config_ctx.yaml --model Qwen/Qwen3-Next-80B-A3B-Instruct --layer-indices 6,7 --no-enable-attention-dp --batch-size 4
NP=2 ./mpi_launch.sh ./run.sh config_gen.yaml --model Qwen/Qwen3-Next-80B-A3B-Instruct --layer-indices 6,7 --no-enable-attention-dp --batch-size 512

# Run with DeepEP A2A
NP=4 ./mpi_launch.sh -x TRTLLM_FORCE_ALLTOALL_METHOD=DeepEP ./run.sh config_ctx.yaml --moe-backend WIDEEP
NP=4 ./mpi_launch.sh -x TRTLLM_FORCE_ALLTOALL_METHOD=DeepEP ./run.sh config_gen.yaml --moe-backend WIDEEP

# Run with imbalanced ranks: except for activating all experts, a% of the tokens are sent to the 1st rank
# Note: if balance ratio is 0, ignore activating all experts
NP=4 ./mpi_launch.sh ./run.sh config_ctx.yaml --balance-method ImbalancedRanks --balance-ratio 0.5
NP=4 ./mpi_launch.sh ./run.sh config_gen.yaml --balance-method ImbalancedRanks --balance-ratio 0.5

# Run with imbalanced experts and balanced ranks: except for activating all experts, a% of the tokens are sent to the front experts on each rank
NP=4 ./mpi_launch.sh ./run.sh config_ctx.yaml --balance-method ImbalancedExperts --balance-ratio 0.5
NP=4 ./mpi_launch.sh ./run.sh config_gen.yaml --balance-method ImbalancedExperts --balance-ratio 0.5
```

### Run with Slurm

> Tips:
> 1. If you have a running Slurm job, please skip step 1 and go straight to step 2 and 3.
> 2. Further, if you have installed `tensorrt_llm` in the Slurm job, you can also skip step 2 and run step 3 with `export CONTAINER_NAME=aaa` specified. If you don't know the container name, run `export CONTAINER_NAME=$(SLURM_JOB_ID=$SLURM_JOB_ID ./slurm_query_container_name.sh)` to get it.

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

> Tips: If you want to change the image, no need to reallocate Slurm jobs. Just start another container by running step 2 with `export CONTAINER_NAME=aaa`, and step 3 will run in the container specified by the `CONTAINER_NAME` env.

**Step 3:** Run benchmarks to generate profiles. Run the following command on the controller node, where `NODES` &le; the number of allocated nodes:

```bash
# Run DeepSeek-R1 NVFP4 with wide ep: uses MNNVL A2A if applicable
SLURM_JOB_ID=$SLURM_JOB_ID NODES=4 NP=16 ./slurm_launch.sh ./run.sh config_gen.yaml --moe-backend WIDEEP

# Run with TRTLLMGen
SLURM_JOB_ID=$SLURM_JOB_ID NODES=4 NP=16 TRTLLM_ENABLE_PDL=1 ./slurm_launch.sh ./run.sh config_gen.yaml --moe-backend TRTLLM

# Run with DeepEPLowLatency
SLURM_JOB_ID=$SLURM_JOB_ID NODES=4 NP=16 TRTLLM_FORCE_ALLTOALL_METHOD=DeepEPLowLatency ./slurm_launch.sh ./run.sh config_gen.yaml --moe-backend WIDEEP

# You can run 4-GPU and 8-GPU tasks without reallocate the slurm job
SLURM_JOB_ID=$SLURM_JOB_ID NODES=1 NP=4 ./slurm_launch.sh ./run.sh config_ctx.yaml
SLURM_JOB_ID=$SLURM_JOB_ID NODES=2 NP=8 ./slurm_launch.sh ./run.sh config_gtx.yaml
```

### Batched run

By specifying a list for `--batch-size` on the command line (or `batch_size` in the YAML file), the script runs multiple configurations in a single process. This significantly reduces the total runtime because it avoids repeated library initialization and model initialization.

Supported list arguments:
- `--batch-size` (or `batch_size` in YAML)
- `--seq-len-q` (or `seq_len_q` in YAML)
- `--seq-len-kv-cache` (or `seq_len_kv_cache` in YAML)
- `--balance-ratio` (or `balance_ratio` in YAML)

Command line arguments are comma separated, for example, `--batch-size 1,2,4`. Configs in the YAML file are lists, for example, `batch_size: [1, 2, 4]`.

Run with OpenMPI:

```bash
NP=4 ./mpi_launch.sh ./run.sh config_ctx.yaml --batch-size 1,2,4 --seq-len-q 1024,8192
NP=4 ./mpi_launch.sh ./run.sh config_gen.yaml --scaled-from 16 --moe-backend WIDEEP --batch-size 32,64,128,256,512 --seq-len-q 1,2,3,4
```

## Parse profiles

Run the following command in the container:

```bash
python3 parse.py --world-size 4

# Specify the location of the .nsys-rep file
python3 parse.py --profile-dir ./profiles --world-size 4 --rank 0

# Parse a specific module. The module must appear exactly once in each run.
python3 parse.py --world-size 4 --module MoE
```

You will receive three reports, each containing kernel timing statistics grouped by module:
1. A printed report on stdout
2. A CSV report at `profiles/report_np4_rank0.csv`
3. An HTML report at `profiles/report_np4_rank0.html`

## Trouble shooting

1. Error `fp8 blockscale gemm only support Hopper` on Blackwell.

   The default MoE backend "CUTLASS" does not support FP8 weights. Please choose the same MoE backend as your end-to-end config. A typical choice is adding `--moe-backend DEEPGEMM`, `--moe-backend TRTLLM`, or `--moe-backend WIDEEP` option.

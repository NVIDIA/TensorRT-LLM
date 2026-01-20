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

# Run with weights loaded. Requires local model directory
NP=4 ./mpi_launch.sh ./run.sh config_ctx.yaml --model "$LLM_MODELS_ROOT/DeepSeek-R1/DeepSeek-R1-0528-FP4-v2" --load-format AUTO
NP=4 ./mpi_launch.sh ./run.sh config_gen.yaml --model "$LLM_MODELS_ROOT/DeepSeek-R1/DeepSeek-R1-0528-FP4-v2" --load-format AUTO

# Run DeepSeek-V3.2-Exp
NP=4 ./mpi_launch.sh ./run.sh config_ctx.yaml --model deepseek-ai/DeepSeek-V3.2-Exp --tokens-per-block 64 --moe-backend DEEPGEMM
NP=4 ./mpi_launch.sh ./run.sh config_gen.yaml --model deepseek-ai/DeepSeek-V3.2-Exp --tokens-per-block 64 --moe-backend DEEPGEMM --moe-backend-for-prefill DEEPGEMM

# Run DeepSeek-V3.2-Exp with 32k context length
NP=4 ./mpi_launch.sh ./run.sh config_ctx.yaml --model deepseek-ai/DeepSeek-V3.2-Exp --tokens-per-block 64 --moe-backend DEEPGEMM --batch-size 1 --seq-len-q 32769
NP=4 ./mpi_launch.sh ./run.sh config_gen.yaml --model deepseek-ai/DeepSeek-V3.2-Exp --tokens-per-block 64 --moe-backend DEEPGEMM --moe-backend-for-prefill DEEPGEMM --seq-len-kv-cache 32769

# Run with attention TP
NP=4 ./mpi_launch.sh ./run.sh config_ctx.yaml --no-enable-attention-dp
NP=4 ./mpi_launch.sh ./run.sh config_gen.yaml --no-enable-attention-dp

# Run with attention TP and TRTLLMGen
NP=4 ./mpi_launch.sh ./run.sh config_ctx.yaml --no-enable-attention-dp --moe-backend TRTLLM
NP=4 ./mpi_launch.sh ./run.sh config_gen.yaml --no-enable-attention-dp --moe-backend TRTLLM

# Run with MTP3
NP=4 ./mpi_launch.sh ./run.sh config_gen.yaml --batch-size 32 --seq-len-q 4

# Run 4 layers
NP=4 ./mpi_launch.sh ./run.sh config_ctx.yaml --layer-indices 5,6,7,8
NP=4 ./mpi_launch.sh ./run.sh config_gen.yaml --layer-indices 5,6,7,8

# Scale DEP=16 to 4 GPUs: reduce the number of experts, uses MNNVL A2A if applicable
NP=4 ./mpi_launch.sh ./run.sh config_gen.yaml --scaled-from 16 --moe-backend WIDEEP

# Scale TEP=16 to 4 GPUs: reduce the number of attention heads and experts
NP=4 ./mpi_launch.sh ./run.sh config_gen.yaml --scaled-from 16 --no-enable-attention-dp

# Run Nemotron-3-Nano
NP=1 ./mpi_launch.sh ./run.sh config_ctx.yaml --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --layer-indices 4,5,6 --mamba-ssm-cache-dtype float16
NP=1 ./mpi_launch.sh ./run.sh config_gen.yaml --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --layer-indices 4,5,6 --mamba-ssm-cache-dtype float16

# Run Qwen3-Next
NP=2 ./mpi_launch.sh ./run.sh config_ctx.yaml --model Qwen/Qwen3-Next-80B-A3B-Instruct --layer-indices 6,7 --no-enable-attention-dp --mamba-ssm-cache-dtype float16 --batch-size 4
NP=2 ./mpi_launch.sh ./run.sh config_gen.yaml --model Qwen/Qwen3-Next-80B-A3B-Instruct --layer-indices 6,7 --no-enable-attention-dp --mamba-ssm-cache-dtype float16 --batch-size 512

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
> 1. If you have a running Slurm job, you can set environment variable `export SLURM_JOB_ID=aaa` and skip step 1.
> 2. Further, if you have installed `tensorrt_llm` in the Slurm job, you can also skip step 2. Just run step 3 with `export CONTAINER_NAME=aaa` specified. If you don't know the container name, run `export CONTAINER_NAME=$(./slurm_query_container_name.sh)` to get it.

**Step 1:** On the controller node, allocate one or multiple nodes, and export the `SLURM_JOB_ID`:

```bash
export SLURM_JOB_ID=$(NODES=4 TIME=02:00:00 ./slurm_alloc.sh)
```

Please fill the variables in `./slurm_alloc.sh`.

**Step 2:** Start a container and install `tensorrt_llm`. Run the following command on the controller node:

```bash
./slurm_init_containers.sh
```

It uses the image recorded in `../../jenkins/current_image_tags.properties`. The image will be downloaded to `../../enroot/` for once.

> Tips: If you want to change the image, no need to reallocate Slurm jobs. Just start another container by running step 2 with `export CONTAINER_NAME=aaa`, and step 3 will run in the container specified by the `CONTAINER_NAME` env.

**(Optional) Get an interactive shell**

```bash
NODES=1 NP=1 ./slurm_launch.sh --overlap --pty middleware/exclude_slurm_envs bash
```

The `--overlap` option allows this shell to share the node with other jobs. The middleware enables nested MPI process spawning from within Slurm jobs.

You may compile C++ extensions in the interactive shell:

```bash
cd ../..
export CCACHE_DIR=$(realpath cpp/.ccache)
python3 scripts/build_wheel.py --cuda_architectures native --no-venv --skip_building_wheel -G Ninja --use_ccache --clean
```

**Step 3:** Run benchmarks to generate profiles. Run the following command on the controller node, where `NODES` &le; the number of allocated nodes:

```bash
# Run DeepSeek-R1 NVFP4 with wide ep: uses MNNVL A2A if applicable
NODES=4 NP=16 ./slurm_launch.sh ./run.sh config_gen.yaml --moe-backend WIDEEP

# Run with TRTLLMGen
NODES=4 NP=16 ./slurm_launch.sh ./run.sh config_gen.yaml --moe-backend TRTLLM

# Run with DeepEPLowLatency
NODES=4 NP=16 TRTLLM_FORCE_ALLTOALL_METHOD=DeepEPLowLatency ./slurm_launch.sh ./run.sh config_gen.yaml --moe-backend WIDEEP

# You can run 4-GPU and 8-GPU tasks without reallocating the slurm job
NODES=1 NP=4 ./slurm_launch.sh ./run.sh config_ctx.yaml
NODES=2 NP=8 ./slurm_launch.sh ./run.sh config_gen.yaml
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
# Parse the profile at the default directory
python3 parse.py --world-size 4

# Specify the file path
python3 parse.py --file-path profiles/report_np4_rank0.nsys-rep
python3 parse.py --profile-dir ./profiles --world-size 4 --rank 0

# Parse a specific module. The module must appear exactly once in each run.
python3 parse.py --world-size 4 --module MoE
```

You will receive three reports, each containing kernel timing statistics grouped by module:
1. A printed report on stdout
2. A CSV report at `profiles/report_np4_rank0.csv`
3. An HTML report at `profiles/report_np4_rank0.html`

## Performance calibrating

1. Run end-to-end serving in COLLECT mode, and capture the nsys profile. This generates a calibration file.

   The example runs `trtllm-bench` and captures the generation phase.

   > Note: All example commands in "Performance calibrating" section should be executed in a Docker container or Slurm container.

   ```bash
   model_path="$LLM_MODELS_ROOT/DeepSeek-R1/DeepSeek-R1-0528-FP4-v2/"
   NP=4
   BATCH_SIZE=32
   
   python3 ../../benchmarks/cpp/prepare_dataset.py \
       --tokenizer=${model_path} --stdout token-norm-dist --num-requests=$(($BATCH_SIZE*$NP)) \
       --input-mean=8192 --output-mean=1024 --input-stdev=0 --output-stdev=0 >dataset.jsonl
   
   cat <<EOF >config.yaml
   enable_attention_dp: true
   layerwise_benchmarks_config:
       calibration_mode: COLLECT
       calibration_file_path: profiles/calibration_data.json
   moe_config:
       backend: CUTLASS
   print_iter_log: true
   speculative_config:
       decoding_type: MTP
       num_nextn_predict_layers: 3
   EOF
   
   mkdir -p profiles
   
   TLLM_PROFILE_START_STOP=$(($BATCH_SIZE + 10))-$(($BATCH_SIZE + 35)) \
   NP=$NP ./mpi_launch.sh middleware/mpi_env_from_ompi \
   nsys profile \
       -t cuda,nvtx \
       --cpuctxsw none --cuda-event-trace false \
       --cuda-graph-trace node \
       -c cudaProfilerApi --capture-range-end stop \
       -o profiles/report_e2e_collect_rank%q{RANK}.nsys-rep \
       --force-overwrite true \
   trtllm-llmapi-launch \
   trtllm-bench \
       --model deepseek-ai/DeepSeek-V3 \
       --model_path ${model_path} \
       throughput \
       --tp $NP \
       --ep $NP \
       --warmup 0 \
       --dataset dataset.jsonl \
       --max_batch_size $BATCH_SIZE \
       --max_num_tokens 10240 \
       --disable_chunked_context \
       --num_requests $(($BATCH_SIZE*$NP)) \
       --concurrency $(($BATCH_SIZE*$NP)) \
       --config config.yaml
   ```

2. If the end-to-end serving uses CUDA Graphs, run it again in MARK mode without CUDA Graphs, and also capture the nsys profile.

   ```bash
   model_path="$LLM_MODELS_ROOT/DeepSeek-R1/DeepSeek-R1-0528-FP4-v2/"
   NP=4
   BATCH_SIZE=32
   
   cat <<EOF >config.yaml
   cuda_graph_config: null
   enable_attention_dp: true
   layerwise_benchmarks_config:
       calibration_mode: MARK
   moe_config:
       backend: CUTLASS
   print_iter_log: true
   speculative_config:
       decoding_type: MTP
       num_nextn_predict_layers: 3
   EOF
   
   TLLM_PROFILE_START_STOP=$(($BATCH_SIZE + 10))-$(($BATCH_SIZE + 35)) \
   NP=$NP ./mpi_launch.sh middleware/mpi_env_from_ompi \
   nsys profile \
       -t cuda,nvtx \
       --cpuctxsw none --cuda-event-trace false \
       --cuda-graph-trace node \
       -c cudaProfilerApi --capture-range-end stop \
       -o profiles/report_e2e_mark_rank%q{RANK}.nsys-rep \
       --force-overwrite true \
   trtllm-llmapi-launch \
   trtllm-bench \
       --model deepseek-ai/DeepSeek-V3 \
       --model_path ${model_path} \
       throughput \
       --tp $NP \
       --ep $NP \
       --warmup 0 \
       --dataset dataset.jsonl \
       --max_batch_size $BATCH_SIZE \
       --max_num_tokens 10240 \
       --disable_chunked_context \
       --num_requests $(($BATCH_SIZE*$NP)) \
       --concurrency $(($BATCH_SIZE*$NP)) \
       --config config.yaml
   ```

3. Run layer-wise benchmarks with the calibration file which is obtained by step 1.

   ```bash
   NP=4 ./mpi_launch.sh ./run.sh config_gen.yaml --model "$LLM_MODELS_ROOT/DeepSeek-R1/DeepSeek-R1-0528-FP4-v2" --load-format AUTO --layer-indices 5,6,7 --batch-size 32 --seq-len-q 4 --seq-len-kv-cache 8316 --balance-method NotModified --replay-file-path profiles/calibration_data.json --replay-start 47 --replay-stop 67
   ```

4. Parse end-to-end profiles with `parse_e2e.py`, and parse layer-wise benchmarks profiles with `parse.py`.

   ```bash
   seq 0 3 | xargs -I% python3 parse_e2e.py --eager-trace profiles/report_e2e_mark_rank%.nsys-rep --graph-trace profiles/report_e2e_collect_rank%.nsys-rep --layer-indices 5,6,7 --warmup-times 5 -o profiles/report_e2e_mark_rank%.json
   seq 0 3 | xargs -I% python3 parse.py --world-size 4 --rank %
   ```

5. Run `correlation.py` to generate the correlation report.

   ```bash
   python3 correlation.py --reference profiles/report_e2e_mark_rank0.json $(seq 1 3 | xargs -I% echo "--target profiles/report_e2e_mark_rank%.json") $(seq 0 3 | xargs -I% echo "--target profiles/report_np4_rank%.json") -o profiles/correlation.html
   ```

Limitations:

1. Pipeline parallelism is not supported.
2. MoE backend CUTLASS, and WIDEEP are supported.

## Developer utilities

1. Less startup time when debug a model
   1. Set autotuner cache or disable autotuner
      1. Set autotuner cache: add `TLLM_AUTOTUNER_CACHE_PATH=autotuner_cache/cache` environment variable. This is enabled at your own risk, and you may need to delete the cache if `NP` changes or the code changes
      2. Disable autotuner: add `--no-enable-autotuner` option
   2. Disable nsys profile: set `PROFILE=0` environment variable
2. Capture more information
   1. Enable GPU metrics: set `GPU_METRICS=1` environment variable
   2. Enable backtrace: set `BACKTRACE=1` environment variable

## Trouble shooting

1. Error `fp8 blockscale gemm only support Hopper` on Blackwell.

   The default MoE backend "CUTLASS" does not support FP8 weights. Please choose the same MoE backend as your end-to-end config. A typical choice is adding `--moe-backend DEEPGEMM` (or `TRTLLM`, `WIDEEP`) and `--moe-backend-for-prefill DEEPGEMM` (or `WIDEEP`) option.

2. Error `huggingface_hub.errors.HfHubHTTPError: 429 Client Error: Too Many Requests for url: https://huggingface.co/nvidia/DeepSeek-R1-0528-FP4-v2/resolve/main/config.json`.

   Please use a local model through the `--model` option, or follow Hugging Face's instructions: "We had to rate limit your IP. To continue using our service, create a HF account or login to your existing account, and make sure you pass a HF_TOKEN if you're using the API."

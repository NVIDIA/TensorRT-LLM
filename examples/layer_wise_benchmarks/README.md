# Layer-wise Benchmarks

This tool profiles individual layers of LLM models to help understand the performance characteristics of each layer and compare layer-wise benchmarks with end-to-end profiling results.

## Generate profiles

### Run with OpenMPI

**Step 1:** Start a container using Docker, Enroot, or other container runtimes. Please refer to `../../jenkins/current_image_tags.properties` for the Docker image URI.

**Step 2:** In the container, install `tensorrt_llm`:

```bash
pip install -e ../..
```

**Step 3:** In the container, run benchmarks and generate profiles:

```bash
# Run DeepSeek-R1 NVFP4
NP=4 ./mpi_launch.sh ./run.sh config_ctx.yaml
NP=4 ./mpi_launch.sh ./run.sh config_gen.yaml

# Run with weights loaded (requires a local model directory)
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

# Scale DEP=16 to 4 GPUs: reduces the number of experts; uses MNNVL A2A if applicable
NP=4 ./mpi_launch.sh ./run.sh config_gen.yaml --scaled-from 16 --moe-backend WIDEEP

# Scale TEP=16 to 4 GPUs: reduces the number of attention heads and experts
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

# Run with imbalanced ranks: in addition to activating all experts, the specified ratio of tokens is sent to rank 0
# Note: if balance ratio is 0, the "activate all experts" behavior is not applied
NP=4 ./mpi_launch.sh ./run.sh config_ctx.yaml --balance-method ImbalancedRanks --balance-ratio 0.5
NP=4 ./mpi_launch.sh ./run.sh config_gen.yaml --balance-method ImbalancedRanks --balance-ratio 0.5

# Run with imbalanced experts and balanced ranks: in addition to activating all experts, the specified ratio of tokens is sent to the front experts on each rank
NP=4 ./mpi_launch.sh ./run.sh config_ctx.yaml --balance-method ImbalancedExperts --balance-ratio 0.5
NP=4 ./mpi_launch.sh ./run.sh config_gen.yaml --balance-method ImbalancedExperts --balance-ratio 0.5
```

### Run with Slurm

> Tips:
> 1. If you have a running Slurm job, you can set the environment variable by running `export SLURM_JOB_ID=<job_id>` and skip Step 1.
> 2. Further, if you have already installed `tensorrt_llm` in the Slurm job, you can also skip Step 2. Just run Step 3 with `export CONTAINER_NAME=<name>` specified. If you don't know the container name, run `export CONTAINER_NAME=$(./slurm_query_container_name.sh)` to get it.

**Step 1:** On the controller node, allocate one or multiple nodes, and export the `SLURM_JOB_ID`:

```bash
export SLURM_JOB_ID=$(NODES=4 TIME=02:00:00 ./slurm_alloc.sh)
```

Please set the variables in `./slurm_alloc.sh` before running.

**Step 2:** Start a container and install `tensorrt_llm`. Run the following command on the controller node:

```bash
./slurm_init_containers.sh
```

This script uses the image recorded in `../../jenkins/current_image_tags.properties`. The image will be downloaded to `../../enroot/` once.

> Tip: If you want to change the image, there is no need to reallocate Slurm jobs. Just start another container by running Step 2 with `export CONTAINER_NAME=<new_name>`, and Step 3 will run in the container specified by the `CONTAINER_NAME` environment variable.

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
# Run DeepSeek-R1 NVFP4 with wide EP; uses MNNVL A2A if applicable
NODES=4 NP=16 ./slurm_launch.sh ./run.sh config_gen.yaml --moe-backend WIDEEP

# Run with TRTLLMGen
NODES=4 NP=16 ./slurm_launch.sh ./run.sh config_gen.yaml --moe-backend TRTLLM

# Run with DeepEPLowLatency
NODES=4 NP=16 TRTLLM_FORCE_ALLTOALL_METHOD=DeepEPLowLatency ./slurm_launch.sh ./run.sh config_gen.yaml --moe-backend WIDEEP

# You can run 4-GPU and 8-GPU tasks without reallocating the Slurm job
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

Command-line arguments are comma-separated, for example, `--batch-size 1,2,4`. Values in the YAML file are lists, for example, `batch_size: [1, 2, 4]`.

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

You will receive four reports, each containing kernel timing statistics grouped by module:
1. A printed report on stdout
2. A CSV report at `profiles/report_np4_rank0.csv`
3. An HTML report at `profiles/report_np4_rank0.html`
4. A JSON report at `profiles/report_np4_rank0.json` (for correlation analysis)

## Performance alignment between end-to-end performance and layer-wise benchmarks

A complete example can be found in `sample_performance_alignment.sh`. Below is an overview of the main steps.

1. Run end-to-end serving in **COLLECT** mode and capture nsys profiles. This step generates a calibration file.

   Requirements:

   1. Add the following fields to `config.yaml`.

      ```yaml
      layer_wise_benchmarks_config:
          calibration_mode: COLLECT
          calibration_file_path: profiles/calibration_data.json
      ```

   2. Set `TLLM_PROFILE_START_STOP` to a range that captures some iterations (typically tens of iterations) of the GEN phase. Ensure that every iteration has the same batch size. Capture 5 extra iterations at the beginning, because the first 5 iterations are treated as warm-ups and will be dropped by the parser by default.

   3. Capture per-rank nsys profiles; each rank should produce a separate file.

      Place `nsys profile` after `mpirun` or `srun`. To minimize profiling overhead and file size, there is no need to capture samples or GPU metrics.

      If you use `trtllm-serve` or `trtllm-bench`, use the following command order. If you use `examples/disaggregated/slurm/benchmark/submit.py`, setting `gen_profile_range` is sufficient.

      ```bash
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
          --model ...
      ```

   4. For more accurate results, set the same `TLLM_AUTOTUNER_CACHE_PATH` for all steps. The autotuner cache file should be generated in Step 1 and reused in Steps 2 and 3.

2. If the end-to-end serving uses CUDA Graphs, run Step 1 again in **MARK** mode without CUDA Graphs and capture nsys profiles.

   The differences from Step 1 are as follows:

   1. Add the following fields to `config.yaml`.

      ```yaml
      cuda_graph_config: null
      layer_wise_benchmarks_config:
          calibration_mode: MARK
      ```

   2. Change the paths of profiles. The recommended argument is `-o profiles/report_e2e_mark_rank%q{RANK}.nsys-rep`.

3. Run layer-wise benchmarks with the calibration file obtained by Step 1.

   ```bash
   NP=4 ./mpi_launch.sh ./run.sh config_gen.yaml \
       --model "$LLM_MODELS_ROOT/DeepSeek-R1/DeepSeek-R1-0528-FP4-v2" \
       --load-format AUTO \
       --layer-indices 5,6,7 \
       --batch-size 32 \
       --seq-len-q 1 \
       --seq-len-kv-cache 2090 \
       --balance-method NotModified \
       --replay-file-path profiles/calibration_data.json \
       --replay-start 47 \
       --replay-stop 67
   ```

   Argument explanations:

   | Argument/Parameter | Explanation |
   | ------------------ | ----------- |
   | `NP=4` | Should match the end-to-end run. |
   | `--load-format AUTO` | Instructs the benchmark to load model weights instead of using random weights. |
   | `--layer-indices 5,6,7` | A list of contiguous layers to calibrate. |
   | `--batch-size 32` | Should match the end-to-end run. |
   | `--seq-len-q 1` | Should match (1 + MTP) of the end-to-end run. |
   | `--seq-len-kv-cache 2090` | An estimate of the average context length for the captured iterations. The first 5 iterations should be excluded from this estimate because they will be dropped by the parser. |
   | `--replay-file-path` | The calibration file obtained from Step 1. |
   | `--replay-start` and `--replay-stop` | Should match the end-to-end `TLLM_PROFILE_START_STOP`. Do not replay the first 5 iterations because they will be dropped by the parser. |

4. Parse end-to-end profiles with `parse_e2e.py`, and parse layer-wise benchmarks profiles with `parse.py`.

   ```bash
   seq 0 $((NP - 1)) | xargs -I% python3 parse_e2e.py \
       --eager-trace profiles/report_e2e_mark_rank%.nsys-rep \
       --graph-trace profiles/report_e2e_collect_rank%.nsys-rep \
       --layer-indices 5,6,7 \
       --warmup-times 5 \
       -o profiles/report_e2e_collect_rank%.json
   seq 0 $((NP - 1)) | xargs -I% python3 parse.py \
       --world-size $NP \
       --rank %
   ```

5. Run `correlation.py` to generate the correlation report.

   ```bash
   python3 correlation.py \
       --reference profiles/report_e2e_collect_rank0.json \
       $(seq 1 $((NP - 1)) | xargs -I% echo "--target profiles/report_e2e_collect_rank%.json") \
       $(seq 0 $((NP - 1)) | xargs -I% echo "--target profiles/report_np${NP}_rank%.json") \
       -o profiles/correlation.html
   ```

   The report can be found at `profiles/correlation.html`.

Limitations:

1. Pipeline parallelism is not supported.
2. Only the CUTLASS and WIDEEP MoE backends are supported.
3. Only tested with the GEN phase and attention DP.

## Developer utilities

1. Reduce startup time when debugging a model
   1. Set autotuner cache or disable autotuner
      1. Set autotuner cache: set the `TLLM_AUTOTUNER_CACHE_PATH=autotuner_cache/cache` environment variable. Use this at your own risk; you may need to delete the cache if `NP` changes or the code changes
      2. Disable autotuner: add the `--no-enable-autotuner` option
   2. Disable nsys profiling: set the `PROFILE=0` environment variable
2. Capture more information
   1. Enable GPU metrics: set the `GPU_METRICS=1` environment variable
   2. Enable backtrace: set the `BACKTRACE=1` environment variable

## Troubleshooting

1. Error `fp8 blockscale gemm only support Hopper` on Blackwell.

   The default MoE backend "CUTLASS" does not support FP8 weights. Please choose the same MoE backend as your end-to-end config. A typical solution is to add the `--moe-backend DEEPGEMM` (or `TRTLLM`, `WIDEEP`) and `--moe-backend-for-prefill DEEPGEMM` (or `WIDEEP`) options.

2. Error `huggingface_hub.errors.HfHubHTTPError: 429 Client Error: Too Many Requests for url: https://huggingface.co/nvidia/DeepSeek-R1-0528-FP4-v2/resolve/main/config.json`.

   Please use a local model through the `--model` option, or follow Hugging Face's instructions: "We had to rate limit your IP. To continue using our service, create a HF account or login to your existing account, and make sure you pass a HF_TOKEN if you're using the API."

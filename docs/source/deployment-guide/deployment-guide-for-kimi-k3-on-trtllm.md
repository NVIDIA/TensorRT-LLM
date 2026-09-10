# Deployment Guide for Kimi K3 on TensorRT LLM - Blackwell

## Introduction

This deployment guide provides step-by-step instructions for running the Kimi K3 model using TensorRT LLM on NVIDIA Blackwell GPUs. The deployment configurations and results in this guide were validated on NVIDIA GB300 NVL GPUs. It covers the complete setup required; from preparing the model weights and building the software environment to configuring TensorRT LLM parameters, launching the multi-node server, and validating inference output and accuracy.

Kimi K3 is a large hybrid Mixture-of-Experts (MoE) model. Its 93 decoder layers interleave two attention families: most layers use Kimi Delta Attention (KDA), a linear-attention mixer that keeps a constant-size recurrent state per sequence, while every fourth layer uses full Multi-head Latent Attention (MLA). The MoE block routes each token to 16 of 896 experts; the routed expert weights are stored in MXFP4 (group size 32) while the attention, shared-expert, dense-MLP, and LM-head weights stay in BF16. In TensorRT LLM the model is served through the `KimiK3ForConditionalGeneration` / `KimiLinearForCausalLM` architectures (the text decoder path).

This guide uses Slurm and the `trtllm-llmapi-launch` multi-node launcher. The configuration walkthrough focuses on the **DEP16** high-throughput deployment: attention runs data-parallel across 16 ranks and the 896 experts shard across the expert-parallel group (`enable_attention_dp: true`, `moe_expert_parallel_size: 16`). The repository also provides the **TEP16** low-latency deployment and an 8-GPU deployment, **TEP8**.

## Prerequisites

* GPU: NVIDIA Blackwell GPUs. DEP16 and TEP16 use 16 GPUs; the TEP8 recipe uses 8 GPUs. These deployment recipes were validated on GB300 NVL GPUs. The repository's Slurm examples assume 4 GPUs per node. Per-GPU memory requirements differ per recipe, and are set by the attention layout rather than by the GPU count:

  | Recipe | Attention layout | Per-rank weights | Requires |
  | :-- | :-- | --: | :-- |
  | DEP16 | attention-DP (replicated) | 210 GB | GB300-class per-GPU memory |
  | TEP8 | attention-TP, EP8 | 213 GB | GB300-class per-GPU memory |
  | TEP16 | attention-TP, EP16 | 115 GB | validated on GB200 (`SM100`) |

  DEP16 replicates the BF16 non-expert weights on every rank (114 GB per rank) on top of the MXFP4 routed experts at 16-way expert parallelism (90 GB per rank). TEP16 shards those non-expert weights instead, which is what brings it within `SM100` per-GPU memory; TEP8 does not fit because its 8-way expert share alone is 181 GB per rank. On B200 (`SM100`), Kimi K3 is functionally supported at the kernel and module level and covered by unit tests in CI. Other GPU architectures are not supported.
* Multi-node launcher: Slurm with the pyxis/enroot container plugin (or an equivalent MPI launcher) to start one rank per GPU across the nodes.
* High-speed inter-node interconnect (e.g., NVLink/InfiniBand) for the expert-parallel traffic.
* Shared filesystem visible to all nodes for the repository, the model weights, and the configuration file.
* OS: Linux
* Drivers: CUDA Driver 580 or later
* Container runtime with NVIDIA GPU support on each node

## Models

* A complete Hugging Face-format Kimi K3 checkpoint and tokenizer, e.g. [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) downloaded from the Hugging Face Hub (`model_type: kimi_k3`, architecture `KimiK3ForConditionalGeneration`). The checkpoint ships the routed-expert weights pre-quantized to MXFP4; no additional quantization step is needed.

The checkpoint and the configuration file must live on a shared filesystem visible to all nodes. The examples below use `/path/to/kimi-k3-checkpoint` — replace it with your local path.

## Feature Support Notes

* **Blackwell only.** NVIDIA Blackwell GPUs are supported. The performance results in this guide were validated on NVIDIA GB300 NVL GPUs. Kimi K3 kernels and modules are also functional on B200 (`SM100`) and covered by unit tests in CI, and the TEP16 deployment is validated end-to-end on GB200 (`SM100`); DEP16 and TEP8 require GB300-class per-GPU memory (see Prerequisites). Support for other GPU architectures may be added in a future release.
* **High-throughput and low-latency deployments are provided.** DEP16 (`enable_attention_dp: true`, `moe_expert_parallel_size: 16`) is the high-throughput deployment. TEP16 (`enable_attention_dp: false`, `moe_expert_parallel_size: 16`) is the low-latency deployment. An 8-GPU deployment, TEP8 (`enable_attention_dp: false`, `moe_expert_parallel_size: 8`), is also provided. Select the deployment and concurrency appropriate for your workload.
* **CUDA graphs and the overlap scheduler are enabled.** The performance-sweep recipes set `disable_overlap_scheduler: false` and enable CUDA graphs. DEP16 additionally sets `cuda_graph_config.enable_padding: true`.
* **Chunked prefill is supported and enabled** (`enable_chunked_prefill: true`), so prompts longer than `max_num_tokens` are scheduled across multiple steps.
* **`kv_cache_config.tokens_per_block` must be `64`** — required by the MLA (576, 512) generation kernels.
* **Speculative decoding and disaggregated serving are not yet available** for Kimi K3; support is under development. See the "Current limitations" section of `examples/kimi_k3/README.md`.

## Deployment Steps

### Build TensorRT LLM from Source

Kimi K3 support currently requires TensorRT LLM built from source and installed in place. Inside the TensorRT LLM container, from the repository root:

```bash
python3 scripts/build_wheel.py --cuda_architectures 103-real --skip_building_wheel --yes
.venv-3.12/bin/python -m pip install -e .
```

`build_wheel.py` creates the virtual environment at the repository root, named after the container's Python version: `.venv-3.12` for the current containers (Python 3.12). If your container ships a different Python, substitute the matching `.venv-<major>.<minor>` path in the commands on this page. Adjust `--cuda_architectures` to the target GPUs (`103-real` for GB300, `100-real` for B200). A `103-real` build also runs on B200 (the Kimi K3 kernels compile for the `100f` family) but omits the `sm100a`-specific batched-GEMM kernels, so build with `100-real` when targeting B200. The multi-node jobs below run TensorRT LLM from this in-place environment, so build and install with the repository at the same path the jobs use.

Kimi K3 additionally depends on `fla` and `einops`, installed into the same in-place environment (these dependencies might be removed in future releases, replaced by other kernels):

```bash
.venv-3.12/bin/python -m pip install fla-core einops
```

For general build-from-source instructions see [https://nvidia.github.io/TensorRT-LLM/latest/installation/build-from-source.html](https://nvidia.github.io/TensorRT-LLM/latest/installation/build-from-source.html).

### Recommended Performance Settings

**Treat these as a starting point and tune the parameters for your workload.**

Create a YAML configuration file on the shared filesystem. The settings below are the tested DEP16 configuration, matching `examples/kimi_k3/eval_extra_llm_options.yaml`:

```shell
EXTRA_LLM_API_FILE=/path/to/kimi-k3-config.yml

cat << EOF > ${EXTRA_LLM_API_FILE}
# Kimi K3 DEP16 deployment: attention runs data-parallel and the 896
# experts shard across the expert-parallel group.
tensor_parallel_size: 16
enable_attention_dp: true
moe_expert_parallel_size: 16
max_batch_size: 32
max_num_tokens: 8192
max_seq_len: 8192
trust_remote_code: true
disable_overlap_scheduler: false
enable_chunked_prefill: true
cuda_graph_config:
  enable_padding: true
  max_batch_size: 32
moe_config:
  max_num_tokens: 33024
  use_low_precision_moe_combine: true
kv_cache_config:
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.25
  tokens_per_block: 64
EOF
```

Notes:

* The configuration uses `free_gpu_memory_fraction: 0.25` to leave runtime headroom. Each KDA layer also keeps a per-request recurrent state outside the paged KV pool, so reduce this value if the deployment runs out of memory.
* This YAML is specifically the DEP16 high-throughput configuration. For the TEP16 low-latency and TEP8 8-GPU configurations, use `examples/kimi_k3/perf_sweep/perf_sweep.sbatch`; changing only `enable_attention_dp` does not reproduce those recipes.

### Launch the TensorRT LLM Server

Kimi K3 is launched through the `trtllm-llmapi-launch` wrapper, which sets up the multi-rank (MPI/Slurm) environment that the parallel server requires. The wrapper is run once per rank by Slurm (`srun`), with one task (rank) per GPU. The example below launches the server across 4 nodes (`-N 4`), 4 GPUs per node (`--ntasks-per-node 4`, 16 ranks total), using the YAML file to drive parallelism, batching, and the KV-cache constraints:

```bash
MODEL=/path/to/kimi-k3-checkpoint
REPO=/path/to/TensorRT-LLM   # in-place build from the previous step

srun -N 4 \
    --ntasks 16 --ntasks-per-node 4 \
    --mpi=pmix --gpus-per-node=4 \
    --container-image=/path/to/tensorrt-llm-container.sqsh \
    --container-mount-home \
    --container-mounts=${REPO}:${REPO},${MODEL}:${MODEL},${EXTRA_LLM_API_FILE}:${EXTRA_LLM_API_FILE}:ro \
    bash -c "
        ulimit -n 65536

        # Node-local Triton cache: ~/.triton on shared NFS races across
        # ranks during autotune JIT (stale file handles).
        export TRITON_CACHE_DIR=/tmp/triton-cache-rank\${SLURM_PROCID:-0}
        mkdir -p \"\$TRITON_CACHE_DIR\"

        # Node-local flashinfer cache: the default (\$HOME) is shared NFS and
        # races across ranks during cubin download (stale file handles).
        export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer-rank\${SLURM_PROCID:-0}

        # Run from the in-place installation created by build_wheel.py.
        export PATH=\"${REPO}/.venv-3.12/bin:\$PATH\"

        exec ${REPO}/tensorrt_llm/llmapi/trtllm-llmapi-launch \
            trtllm-serve ${MODEL} \
                --host 0.0.0.0 --port 8000 \
                --config ${EXTRA_LLM_API_FILE}
    "
```

> [!NOTE]
> Adjust `-N`, `--ntasks`, `--ntasks-per-node`, and `--gpus-per-node` to match your cluster's GPUs-per-node; the total number of tasks (ranks) must equal `tensor_parallel_size` (`16`). Add the partition / account / time flags (`-p`, `-A`, `-t`) required by your Slurm setup, and ensure the repository, checkpoint, and YAML paths resolve identically on all nodes (shared filesystem).

TensorRT LLM will load weights and select the best kernels during startup. The server is successfully launched when the following log is shown:

```log
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:8000 (Press CTRL+C to quit)
```

### Quick Start Without a Server (LLM API)

For a first functional check without standing up a server, the repository ships a ready-made multi-node quick-start job that loads the model once, runs four prompts, and reports whether each response contains the expected text (a successful run reports `True` for all four checks):

```bash
sbatch examples/kimi_k3/quick_start_kimi_k3.sbatch \
    --model /path/to/kimi-k3-checkpoint \
    --image /path/to/tensorrt-llm-container.sqsh
```

See [`examples/kimi_k3/README.md`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/kimi_k3) for details, including how to override the Slurm partition, account, and time limit.

### Key YAML Options

These options are set within the YAML file passed to `trtllm-serve` via the `--config` argument. Only the Kimi-K3-relevant knobs are described here; see the [`TorchLlmArgs` class](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.TorchLlmArgs) for the full reference.

#### `tensor_parallel_size`

* **Description:** The width of the parallel group. Set it to `16` for DEP16 and TEP16, or `8` for TEP8.

#### `enable_attention_dp`

* **Description:** Set it to `true` for the DEP16 high-throughput deployment. The TEP16 low-latency and TEP8 8-GPU deployments set it to `false`.

#### `moe_expert_parallel_size`

* **Description:** The expert-parallel width for the MoE layers. Set it to `16` for DEP16 and TEP16, or `8` for TEP8.

#### `cuda_graph_config`

* **Description:** The standard DEP16, TEP16, and TEP8 performance recipes enable CUDA graphs. DEP16 also sets `enable_padding: true`, which pads generation batches to a captured size. Set the CUDA-graph `max_batch_size` to the same value as the top-level `max_batch_size`.

#### `disable_overlap_scheduler`

* **Description:** Set it to `false` to enable the overlap scheduler, as used by the standard DEP16, TEP16, and TEP8 performance recipes.

#### `moe_config`

* **Options:**

  * `max_num_tokens`: The DEP16 performance recipe uses `33024` to reduce serialized all-to-all chunk rounds during prefill.
  * `use_low_precision_moe_combine`: Set to `true` in the DEP16 performance recipe to use a low-precision expert-parallel combine payload.

#### `enable_chunked_prefill`

* **Description:** The recommended DEP16 configuration sets this option to `true`. Prompts longer than `max_num_tokens` are then scheduled across multiple prefill steps; this is verified on 16-GPU DEP16 at GSM8K parity with the unchunked baseline.

#### `kv_cache_config`

* **Options:**
  * `enable_block_reuse`: Off by default; set to `true` to enable prefix-cache reuse across requests.
  * `mamba_state_config.periodic_snapshot_interval`: With block reuse on, the KDA recurrent state is snapshotted every this many tokens so prefix hits can restore it (default `0` = snapshots off; hybrid models only expose reusable prefixes at snapshot boundaries, so set e.g. `256` for block reuse to engage; see `examples/kimi_k3/eval_extra_llm_options_reuse.yaml`).
  * `tokens_per_block`: Must be `64`, required by the MLA (576, 512) generation kernels.
  * `free_gpu_memory_fraction`: Fraction of free GPU memory reserved for the paged KV cache after model load. The configuration above uses `0.25` to leave runtime headroom. Lower it if you hit out-of-memory errors.

#### `trust_remote_code`

* **Description:** Required to load the Kimi K3 configuration and tokenizer code shipped with the checkpoint.

### Kimi-Specific API Behavior and Environment Variables

When the served model is Kimi K3, `trtllm-serve` applies Kimi/Moonshot API semantics on `/v1/chat/completions` (all of these are exercised by Moonshot's [Kimi Vendor Verifier](https://www.kimi.com/blog/kimi-vendor-verifier.html)):

* **Request extensions:** the `thinking` object (`{"type": "enabled"|"disabled", "keep": "all", "effort": "low"|"high"|"max"}`), `reasoning_effort` values `"low"`, `"high"`, `"max"`, and `"none"` (an explicit `thinking` object takes precedence), `tool_choice: "required"`, message-level (dynamic) tools declared on system messages, and `response_format` `json_object`/`json_schema` (the `json_schema` wrapper must carry a non-empty `name` and a `schema` object). These map onto the checkpoint chat template's native control messages; explicit `chat_template_kwargs` always win.
* **Streaming usage:** `usage` is reported in the final streaming chunk even when the client does not send `stream_options` (Kimi API parity).
* **Prompt-token accounting:** reported `usage.prompt_tokens` excludes the trailing 3-token generation channel opener, matching Kimi's reference accounting; the model still consumes the full rendered prompt.
* **`TRTLLM_KIMI_PARAM_POLICY`** (default `0`, off): when set to `1`, enforces Kimi's immutable sampling parameters — `top_p` pinned to 0.95 (unset or the OpenAI default `1.0` are coerced to 0.95; other values are rejected with HTTP 400), `presence_penalty`/`frequency_penalty` 0, `n` 1, and `temperature` bounded to [0, 1]. Off by default so existing deployments keep accepting the requests they accept today; a Kimi-Vendor-Verifier certification run must set it to `1` (the KVV params suite requires the rejections).
* **`TRTLLM_KIMI_K3_STRICT_TOOL_GRAMMAR`** (default `0`): opt-in constrained decoding for tools with `strict: true` (requires `guided_decoding_backend: xgrammar`). Disabled by default pending the investigation of a device-side assert observed under sustained concurrent guided load; strict tools otherwise fall back to warn-and-continue.

## Testing API Endpoint

The server (the OpenAI-compatible REST endpoint) runs on the rank-0 node, listening on port `8000`. Send requests to that node's hostname or IP; `localhost` only works from the rank-0 node itself.

### Health Check

```shell
curl -s -o /dev/null -w "Status: %{http_code}\n" "http://localhost:8000/health"
```

When the `Status: 200` code is returned, the server is ready for queries. Note that the very first query may take longer due to initialization and compilation.

### Basic Test

After the TensorRT LLM server is set up and shows `Application startup complete`, you can send requests to the server:

```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "/path/to/kimi-k3-checkpoint",
    "messages": [
        {
            "role": "user",
            "content": "Where is New York?"
        }
    ],
    "max_tokens": 64,
    "temperature": 0
}' -w "\n"
```

The response should contain a `choices[0].message.content` field completing the request, plus a `usage` section with the token counts.

## Running Evaluations to Verify Accuracy (Optional)

The repository ships a ready-made multi-node GSM8K evaluation job built on `trtllm-eval` with the tested DEP16 settings:

```bash
sbatch examples/kimi_k3/run_eval_kimi_k3.sbatch \
    --model /path/to/kimi-k3-checkpoint \
    --image /path/to/tensorrt-llm-container.sqsh
```

The job writes progress and results to `kimi-k3-eval-<job-id>.log` in the submission directory. If no local dataset path is configured, `trtllm-eval` downloads GSM8K from the Hugging Face Hub. The completed log contains a results table with the normalized GSM8K exact-match scores. With the tested checkpoint and the settings in this example, users should expect approximately:

| Filter | Exact match |
| :-- | --: |
| Flexible extract | 96.51 |
| Strict match | 96.44 |

The expected average accuracy is approximately 96.47. Small differences (roughly ±0.5 points) are possible with different checkpoint or dependency revisions.

### TEP16 on GB200

The same job runs the TEP16 layout with `--parallel tep`, which rewrites a per-job copy of the evaluation YAML with `enable_attention_dp: false` and raises `max_batch_size` from 32 to 128 (with attention-DP off every rank serves the same global batch instead of its own, so the batch size is raised to recover eval concurrency):

```bash
sbatch --account <account> --partition batch --qos <qos> --time 04:00:00 \
    examples/kimi_k3/run_eval_kimi_k3.sbatch \
    --model /path/to/kimi-k3-checkpoint \
    --image /path/to/tensorrt-llm-container.sqsh \
    --task gsm8k --parallel tep
```

The batch script declares `--nodes=4 --ntasks-per-node=4 --gpus-per-node=4`, and takes `--account`, `--partition` and `--qos` from the submitting command line. Export `KIMI_K3_ROUTER_BF16=0` before submitting: with attention-DP off the MoE router gate defaults to its BF16 fast path, which can flip borderline expert picks, so the reference scores above are only comparable with that path disabled. `KIMI_K3_FP8_WEIGHT_READ` defaults to `0`, which is the precision the reference scores were measured at.

Measured on 16 GB200 GPUs (4 nodes, `100-real` build, 184.31 GiB per GPU), with the checkpoint's native MXFP4 routed experts:

| Filter | Exact match |
| :-- | --: |
| Flexible extract | 96.82 |
| Strict match | 96.74 |

| Per-rank memory | Value |
| :-- | --: |
| Weights | 106.67 GiB |
| Non-torch (NCCL, CUDA graphs) | 15.32 GiB |
| Peak during profiling | 125.96 GiB |
| KV cache at `free_gpu_memory_fraction: 0.25` | 15.60 GiB |

## Benchmarking Performance

### Run the End-to-End Performance Sweep

Use `examples/kimi_k3/perf_sweep/perf_sweep.sbatch` to reproduce the performance measurements. The script generates the tuned serving configuration for the selected deployment mode, launches the server, waits for it to become healthy, runs a warmup, and measures each requested concurrency:

```bash
sbatch examples/kimi_k3/perf_sweep/perf_sweep.sbatch \
    --mode dep16 \
    --model /path/to/kimi-k3-checkpoint \
    --image /path/to/tensorrt-llm-container.sqsh \
    --isl 8192 \
    --osl 1024 \
    --concurrencies "64 128 256"
```

Set `--mode` to `dep16`, `tep16`, or `tep8`. The script derives `max_seq_len` from the requested input and output lengths as `ISL + OSL + 128`, and writes the server log, client log, generated YAML, and per-concurrency JSON results to the job's output directory.

### Benchmark an Existing Server

Launch the server with the deployment YAML that you want to measure, as described in [Launch the TensorRT LLM Server](#launch-the-tensorrt-llm-server). From a machine that can reach the rank-0 server, choose the input length, output length, and concurrency for your workload:

```bash
MODEL=/path/to/kimi-k3-checkpoint
HOST=localhost
PORT=8000
ISL=8192
OSL=1024
CONCURRENCIES="64 128 256"
NUM_ROUNDS=5
RESULT_DIR=/path/to/kimi-k3-results

mkdir -p "${RESULT_DIR}"

for concurrency in ${CONCURRENCIES}; do
    python3 -m tensorrt_llm.serve.scripts.benchmark_serving \
        --model "${MODEL}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --dataset-name random \
        --random-ids \
        --tokenize-on-client \
        --random-input-len "${ISL}" \
        --random-output-len "${OSL}" \
        --num-prompts "$((concurrency * NUM_ROUNDS))" \
        --max-concurrency "${concurrency}" \
        --ignore-eos \
        --trust-remote-code \
        --save-result \
        --result-dir "${RESULT_DIR}" \
        --result-filename "concurrency_${concurrency}.json"
done
```

Before running the benchmark, adjust `ISL` and `OSL` for the server configuration being measured. Their sum must not exceed the server's `max_seq_len`. To reproduce the 8K input / 1K output results below, use the performance-sweep script, which sets `max_seq_len: 9344` (`ISL + OSL + 128`).

### Sample Measured Results

The following results were measured on GB300 NVL GPUs with the 8K/1K sweep. All requests completed. `TPS/user` is `1000 / median TPOT`; it represents the median decode rate seen by one active user. Your results may vary with checkpoint, software, and cluster I/O revisions.

For the exact server settings and environment variables used to collect the results, refer to `examples/kimi_k3/perf_sweep/perf_sweep.sbatch`.

| Mode | GPUs | Concurrency | Output tok/s | Output tok/s/GPU | Median TTFT (ms) | Median TPOT (ms) | TPS/user |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| TEP16 | 16 | 4 | 201.2 | 12.6 | 1769.2 | 18.12 | 55.18 |
| DEP16 | 16 | 64 | 1669.1 | 104.3 | 4512.3 | 34.06 | 29.36 |
| DEP16 | 16 | 256 | 3823.9 | 239.0 | 4586.7 | 62.24 | 16.07 |

The TEP16 point illustrates the low-latency deployment at small concurrency. The DEP16 points illustrate aggregate-throughput scaling at higher concurrencies.

### Key Metrics

#### Time to First Token (TTFT)
  * The typical time elapsed from when a request is sent until the first output token is generated.

#### Time Per Output Token (TPOT) and Inter-Token Latency (ITL)
  * TPOT is the typical time required to generate each token *after* the first one.
  * ITL is the typical time delay between the completion of one token and the completion of the next.
  * Both TPOT and ITL ignore TTFT.

#### End-to-End (E2E) Latency
  * The typical total time from when a request is submitted until the final token of the response is received.

#### Total Token Throughput
  * The combined rate at which the system processes both input (prompt) tokens and output (generated) tokens.

## Troubleshooting Tips

* **Multi-node startup hangs or ranks can't find each other:** Verify that the repository, the checkpoint, and the YAML resolve identically on all nodes (shared filesystem), that the total Slurm task count equals `tensor_parallel_size` (`16`), and that the inter-node interconnect is healthy.
* **Stale file handle / cache errors during startup:** Point `TRITON_CACHE_DIR` and `FLASHINFER_WORKSPACE_BASE` at node-local storage (e.g., `/tmp`) as shown in the launch command; the shared-NFS defaults race across ranks during autotune JIT and cubin download.
* **CUDA out-of-memory errors:** Reduce `kv_cache_config.free_gpu_memory_fraction`, `max_batch_size`, or `max_seq_len`. Remember that the KDA layers keep per-request FP32 recurrent state outside the paged KV pool.
* **Job reaches its Slurm time limit before the model loads:** Weight loading for a checkpoint of this size is dominated by filesystem throughput; request a longer allocation (`--time=...`), particularly with cold caches or a busy filesystem.
* **Model fails to load:** Make sure `trust_remote_code: true` is set and that the checkpoint path is a complete Hugging Face-format Kimi K3 checkpoint with its tokenizer files.
* **`fla` / `einops` import errors:** Install the extra dependencies into the in-place environment: `.venv-3.12/bin/python -m pip install fla-core einops`.
* **Accuracy looks off:** Confirm `tokens_per_block: 64` (a hard constraint for Kimi K3), and that the remaining settings match the tested configuration above.
* **GPU utilization:** For performance issues, check GPU utilization with `nvidia-smi` on the compute nodes while the server is running.

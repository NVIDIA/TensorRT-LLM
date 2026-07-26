# Kimi K3

This example runs Kimi K3 with TensorRT-LLM. It includes an LLM API quick
start and configuration for GSM8K evaluation.

## Hardware support

Only NVIDIA Blackwell GPUs are currently supported and tested. Support for
other GPU architectures may be added in a future release.

## Prerequisites

- TensorRT-LLM built from this repository and installed. Inside
  the TensorRT-LLM container, from the repository root:

  ```bash
  python3 scripts/build_wheel.py --cuda_architectures 103-real
  .venv-3.12/bin/python -m pip install --no-deps -e .
  ```
  You can also build the wheel and then pip install the generated wheel.
  Using editable mode is recommended for development and testing, see [build-from-source](./docs/source/installation/build-from-source.md) for details.

  `build_wheel.py` creates the `.venv-3.12` virtual environment at the
  repository root (named after the container's Python version). Adjust
  `--cuda_architectures` to the target GPUs (`100-real` for GB200,
  `103-real` for GB300). 
- A complete Hugging Face-format Kimi K3 checkpoint and tokenizer.
- A Slurm cluster with 16 NVIDIA Blackwell GPUs and a TensorRT-LLM container
  image.
- The `fla-core` and `einops` packages, installed into the same in-place
  environment. Note: these dependencies might be removed in a future
  release, replaced by other kernels.

  ```bash
  .venv-3.12/bin/python -m pip install fla-core einops
  ```
- To use the optimized CuTeDSL MLA kernel, install the following FlashInfer
  revision into the same in-place environment after installing TensorRT-LLM:

  ```bash
  python -u -m pip install --force-reinstall --no-deps --no-build-isolation \
    "flashinfer-python[cu13] @ git+ssh://git@gitlab-master.nvidia.com:12051/jief/flashinfer.git@82eff087744246b16eab3b891edefd87e0d7b981"
  ```

  The TensorRT-LLM environment already provides FlashInfer's runtime
  dependencies; `--no-deps` prevents pip from replacing its pinned PyTorch,
  Triton, CUDA, and CuTeDSL packages. Install FlashInfer last: TensorRT-LLM
  currently pins `flashinfer-python==0.6.14`, so a later
  dependency-resolving TensorRT-LLM install can replace this source revision.

  > **Note:** This is a private repository; make sure a working,
  > authorized SSH key is available before running the install command.

## Run the model

Kimi K3 requires a multi-node launch. From the repository root, submit the
quick-start Slurm job with the checkpoint and container paths:

```bash
sbatch examples/kimi_k3/quick_start_kimi_k3.sbatch \
    --model /path/to/kimi-k3-checkpoint \
    --image /path/to/tensorrt-llm-container.sqsh
```

Slurm prints the submitted job ID immediately. After the job starts, its output
is written to `kimi-k3-quick-start-<job-id>.log` in the submission directory.
The model is loaded once, then the log shows four prompts, their generated
text, and whether each response contains the expected text. A successful run
should report `True` for all four checks.

For a full GSM8K evaluation, submit:

```bash
sbatch examples/kimi_k3/run_gsm8k_kimi_k3.sbatch \
    --model /path/to/kimi-k3-checkpoint \
    --image /path/to/tensorrt-llm-container.sqsh
```

This job writes progress and results to
`kimi-k3-gsm8k-<job-id>.log`. If no local dataset path is configured,
`trtllm-eval` downloads GSM8K from the Hugging Face Hub. The completed log
contains a results table with the normalized GSM8K exact-match scores. With
the tested checkpoint and the settings in this example, users should expect
approximately:

| Filter | Exact match |
| :-- | --: |
| Flexible extract | 96.51 |
| Strict match | 96.44 |

The expected average accuracy is approximately 96.47. Small differences are
possible with different checkpoint or dependency revisions.

To evaluate with suffix-automaton (SA) speculative decoding, add `--sa`:

```bash
sbatch examples/kimi_k3/run_gsm8k_kimi_k3.sbatch \
    --model /path/to/kimi-k3-checkpoint \
    --image /path/to/tensorrt-llm-container.sqsh \
    --sa
```

This selects `eval_extra_llm_options_sa.yaml` and the SA-required
`max_batch_size` of 8 (see Current limitations below for what SA runs
change). Scores should match the non-SA run within noise — SA speculative
decoding is lossless.

For serving performance, use the standard sweep under
`examples/kimi_k3/perf_sweep/` (this supersedes the older
`run_serving_benchmark_kimi_k3.sbatch` single-recipe benchmark). It submits
the 17-point 8K/1K sweep across the three tuned serving recipes — `tep16`
(latency, c 1–16), `tep8` (interactive, 8 GPUs, c 1–16), and `dep16`
(throughput, c 16–1024):

```bash
examples/kimi_k3/perf_sweep/submit_perf_sweep.sh \
    --model /path/to/kimi-k3-checkpoint \
    --image /path/to/tensorrt-llm-container.sqsh
```

All jobs of a comparison batch are submitted together on purpose: a weight
load overlapping another job's measurement window depresses DEP16 c>=128
points by ~30%. Use `--jobs "tep16 tep8 dep16-lo dep16-hi"` to select a
subset and `--dry-run` to inspect the sbatch commands.

Guard the same three recipes with the GSM8K accuracy sweep whenever the
serving configs or kernels change (expect ~96.5 +/- 0.5 per recipe):

```bash
examples/kimi_k3/perf_sweep/submit_acc_sweep.sh \
    --model /path/to/kimi-k3-checkpoint \
    --image /path/to/tensorrt-llm-container.sqsh
```

Results land in per-job `kimi-k3-sweep-<name>-<job-id>.log` files and result
JSONs in the submission directory (run from a fresh results folder).

Scheduler options must precede the script path; model and image arguments
follow it. The scripts default to the `batch` partition and the
`coreai_comparch_trtllm` account. Change the corresponding `#SBATCH` settings
to match your cluster, or override them when submitting the job. For example:

```bash
sbatch --partition=PARTITION --account=ACCOUNT --time=04:00:00 \
    SCRIPT --model MODEL --image IMAGE
```

## Troubleshooting

### The job reaches its time limit

The default time limits are intentionally aggressive to make the jobs easier
to schedule: 40 minutes for the quick start and two hours for GSM8K.
Depending on how fast your cluster's filesystem loads the weights, a job may
be terminated before producing its final results, particularly with cold
runtime caches or a busy filesystem.

Request a longer allocation if the time is not sufficient for your environment:

```bash
sbatch --time=02:00:00 examples/kimi_k3/quick_start_kimi_k3.sbatch \
    --model MODEL --image IMAGE

sbatch --time=04:00:00 examples/kimi_k3/run_gsm8k_kimi_k3.sbatch \
    --model MODEL --image IMAGE
```

## Chunked prefill and KV-cache block reuse

Chunked prefill is supported and enabled by default in this example
(`enable_chunked_prefill: true` in the quick start and in
`eval_extra_llm_options.yaml`).

KV-cache block reuse is supported as an opt-in: set
`kv_cache_config.enable_block_reuse: true`, or use the example flags —
`--enable-block-reuse` for the quick start and `--reuse` for the GSM8K
job (which selects `eval_extra_llm_options_reuse.yaml`):

```bash
sbatch examples/kimi_k3/quick_start_kimi_k3.sbatch \
    --model MODEL --image IMAGE --enable-block-reuse

sbatch examples/kimi_k3/run_gsm8k_kimi_k3.sbatch \
    --model MODEL --image IMAGE --reuse
```

Block reuse stays off by default because suffix-automaton speculative
decoding requires the default cache manager, which cannot reuse blocks.

## Current limitations

Pipeline parallelism and disaggregated serving are not supported.

Suffix-automaton (SA) speculative decoding is supported as an opt-in for
evaluation (see `eval_extra_llm_options_sa.yaml`). SA runs keep CUDA graphs
and the overlap scheduler off, need `max_batch_size` ≤ 8.
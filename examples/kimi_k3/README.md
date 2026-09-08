# Kimi K3

This example runs Kimi K3 with TensorRT-LLM. It includes an LLM API quick
start and configuration for GSM8K evaluation.

## Hardware support

Only NVIDIA Blackwell GPUs (`SM100` family) are supported. The performance
results in this example were measured on GB300 NVL (`SM103`). Per-GPU memory
requirements differ per recipe, and are set by the attention layout rather
than by the GPU count: DEP16 needs 210 GB per rank and TEP8 needs 213 GB per
rank, so both require GB300-class per-GPU memory, while TEP16 needs 115 GB
per rank and is validated end-to-end on GB200 (`SM100`) at 16 GPUs. B200
(`SM100`) is functionally supported at the kernel and module level and
covered by unit tests in CI. See Current limitations. Support for other GPU
architectures may be added in a future release.

## Prerequisites

- TensorRT-LLM built from this repository and installed. Inside
  the TensorRT-LLM container, from the repository root:

  ```bash
  python3 scripts/build_wheel.py --cuda_architectures 103-real --skip_building_wheel --yes
  .venv-3.12/bin/python -m pip install --no-deps -e .
  ```
  Using editable mode is recommended for development and testing; see
  [build from source](../../docs/source/installation/build-from-source.md)
  for details.

  `build_wheel.py` creates the `.venv-3.12` virtual environment at the
  repository root (named after the container's Python version). If your
  container ships a different Python, substitute `.venv-<major>.<minor>`
  for `.venv-3.12` in every command below and export
  `TRTLLM_VENV=/path/to/repo/.venv-<major>.<minor>` when submitting the
  Slurm jobs (they default to the repository-root `.venv-3.12`). Adjust
  `--cuda_architectures` to the target GPUs (`103-real` for GB300,
  `100-real` for B200).
- A complete Hugging Face-format Kimi K3 checkpoint and tokenizer, e.g.
  [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) downloaded
  from the Hugging Face Hub (the example scripts take a local filesystem
  path).
- A Slurm cluster with 16 NVIDIA Blackwell GPUs (GB300-class per-GPU memory
  for DEP16 and TEP8; see Hardware support) and a TensorRT-LLM container
  image. The image passed as `--image` below must already provide
  TensorRT-LLM's runtime dependencies, that is, a release-style TensorRT-LLM
  container; a build or devel image without them does not work. The Slurm
  scripts start a fresh container from that image. They mount the
  repository, including `.venv-3.12`, and the checkpoint, so packages installed in
  that virtual environment are available. Packages installed only in the preparation
  container's base environment are not:
  repository-root `.venv-3.12` is the build environment created by `build_wheel.py`
  (it contains Conan and pip, not PyTorch or Transformers), and `pip install
  --no-deps -e .` installs `tensorrt_llm` alone. With an image that lacks the dependencies, every rank
  fails with a `ModuleNotFoundError` that does not name the image, such as
  `No module named 'transformers'`.

  The scripts also export `PYTHONPATH="$REPO"` before launching the example.
  Keep that export when adapting them: `python3 <script>` puts the script's
  directory on `sys.path`, not the repository root, so without it the job
  fails with `No module named 'tensorrt_llm'` even when the image is correct.
- The `fla-core` and `einops` packages, installed into the same in-place
  environment. Note: these dependencies might be removed in a future
  release, replaced by other kernels.

  ```bash
  .venv-3.12/bin/python -m pip install fla-core einops
  ```
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
sbatch examples/kimi_k3/run_eval_kimi_k3.sbatch \
    --model /path/to/kimi-k3-checkpoint \
    --image /path/to/tensorrt-llm-container.sqsh
```

This job writes progress and results to
`kimi-k3-eval-<job-id>.log`. If no local dataset path is configured,
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
sbatch examples/kimi_k3/run_eval_kimi_k3.sbatch \
    --model /path/to/kimi-k3-checkpoint \
    --image /path/to/tensorrt-llm-container.sqsh \
    --sa
```

This selects `eval_extra_llm_options_sa.yaml` (see Current limitations
below for what SA changes) and logs a speculative-decoding acceptance
summary at the end of the run. SA is lossless, so the scores should match
the non-SA run within noise. Speedup is workload-dependent, proportional
to the n-gram repetition in the generated output.

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
follow it. The scripts default to the `batch` partition and use an `${account}`
placeholder. Change the corresponding `#SBATCH` settings
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

sbatch --time=04:00:00 examples/kimi_k3/run_eval_kimi_k3.sbatch \
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

sbatch examples/kimi_k3/run_eval_kimi_k3.sbatch \
    --model MODEL --image IMAGE --reuse
```

Block reuse stays off by default because suffix-automaton speculative
decoding requires the default cache manager, which cannot reuse blocks.

## Current limitations

- Pipeline parallelism is not supported.
- **DEP16 and TEP8 require GB300-class per-GPU memory.** Under attention-DP
  the BF16 non-expert weights are replicated on every rank (114 GB per rank),
  which together with the MXFP4 routed experts at 16-way expert parallelism
  (90 GB per rank) needs 210 GB per rank for DEP16; TEP8 needs 213 GB per
  rank because its 8-way expert share alone is 181 GB. Neither fits `SM100`
  per-GPU memory. Use TEP16, which shards the non-expert weights and needs
  115 GB per rank — validated end-to-end on GB200 (16 GPUs, 4 nodes), see the
  deployment guide. Note that the FP8 weight-read path (TRTLLM-14765) does
  not lift the DEP16 requirement: the conversion runs after the weights are
  already resident in BF16, so it lowers the steady-state footprint but
  leaves the load-time peak unchanged.
- **Known performance limitation at DEP16 saturation** (attention-DP +
  EP16 throughput recipe): the 8K/1K serving sweep loses several percent
  of output throughput at concurrency ≥ 128 (up to ~15% at concurrency
  1024) relative to earlier development measurements; concurrency ≤ 64
  and the TEP16/TEP8 latency recipes are unaffected. Tracked as
  TRTLLM-14904.
- FP8 KV cache (`kv_cache_config.dtype: fp8`) is not yet supported.
- Speculative decoding: suffix-automaton speculation is supported for aggregated serving (`speculative_config: {decoding_type: SA}` in the extra LLM API options). For evaluation, use `eval_extra_llm_options_sa.yaml` (the `--sa` flag of the GSM8K job): that configuration runs with the overlap scheduler off, `max_batch_size` 8, and a matching CUDA-graph `max_batch_size`. Suffix-automaton speculation also works under disaggregated serving; enable it on the generation server with `examples/kimi_k3/disagg/gen_config.yaml` (SA runs eager with `max_batch_size` ≤ 8; see `examples/kimi_k3/disagg/README.md`).

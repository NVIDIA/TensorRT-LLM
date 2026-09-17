# Accuracy Evaluation Tool `trtllm-eval`

We provide a CLI tool `trtllm-eval` for evaluating model accuracy. It shares the core evaluation logics with the [accuracy test suite](../../tests/integration/defs/accuracy) of TensorRT-LLM.

`trtllm-eval` is built on the offline API -- [LLM API](https://nvidia.github.io/TensorRT-LLM/llm-api/index.html). It provides developers a unified entrypoint for accuracy evaluation. Compared with the online API [`trtllm-serve`](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve.html), offline API provides clearer error messages and simplifies the debugging workflow.

`trtllm-eval` follows the CLI interface of [`trtllm-serve`](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve.html).

```bash
pip install -r requirements.txt

# Evaluate Llama-3.1-8B-Instruct on MMLU
trtllm-eval --model meta-llama/Llama-3.1-8B-Instruct mmlu

# Evaluate Llama-3.1-8B-Instruct on GSM8K
trtllm-eval --model meta-llama/Llama-3.1-8B-Instruct gsm8k

# Evaluate Llama-3.3-70B-Instruct on GPQA Diamond
trtllm-eval --model meta-llama/Llama-3.3-70B-Instruct gpqa_diamond
```

The `--model` argument accepts either a Hugging Face model ID or a local checkpoint path. By default, `trtllm-eval` runs the model with the PyTorch backend; pass `--backend tensorrt` to switch to the TensorRT backend. Alternatively, the `--model` argument also accepts a local path to pre-built TensorRT engines; in that case, please pass the Hugging Face tokenizer path to the `--tokenizer` argument.

See more details by `trtllm-eval --help`.

## NeMo-Skills benchmarks

`trtllm-eval` also exposes [NeMo-Skills](https://github.com/NVIDIA/NeMo-Skills)
benchmarks as **in-process** subcommands. The model is loaded once into the
in-process `LLM`, each bench generates through it, and NeMo-Skills' own graders
score the output (lm-eval style):

| Subcommand | Benchmark | Grader | LLM judge? |
|------------|-----------|--------|:----------:|
| `gpqa_ns`       | GPQA-diamond (multiple choice)              | `multichoice` answer extraction        | No  |
| `ifbench`       | instruction following                        | allenai/IFBench rule checker            | No  |
| `scicode_ns`    | SciCode (multi-step scientific code)         | code execution in a sandbox             | No  |
| `hle_aa`        | Humanity's Last Exam                         | LLM judge (`judge/hle`)                 | Yes |
| `aa_lcr`        | AA-LCR (long-context reading)                | LLM equality checker (`judge/aalcr`)    | Yes |
| `arena_hard_aa` | Arena-Hard (pairwise vs gpt-4-0314)          | pairwise LLM judge (`judge/arena`)      | Yes |

### 1. Install (once)

[`install_nemo_skills.sh`](install_nemo_skills.sh) pip-installs `nemo_skills`
plus the grader deps from [`requirements_nemo_skills.txt`](requirements_nemo_skills.txt)
into the current Python env. It installs **only the lib** — the datasets and
grader assets come from the shared, read-only `ns_acc_bench_infra` folder, so
there is no download / dataset-prepare step:

```bash
# Run inside the TensorRT-LLM container.
bash examples/trtllm-eval/install_nemo_skills.sh
```

### 2. Run

On first use the evaluators autowire the `ns_acc_bench_infra` folder (default
`<LLM_MODELS_ROOT>/datasets/ns_acc_bench_infra`, override with
`NS_ACC_BENCH_INFRA=<dir>`): they set the NeMo-Skills env, redirect the IFBench
grader and SciCode `test_data.h5` paths that NeMo-Skills hardcodes, and auto-start
the SciCode sandbox — so every bench just runs, and skips gracefully when the
infra folder is absent.

```bash
# Single GPU:
trtllm-eval --model <hf_or_path> gpqa_ns

# Multi-GPU (TP/EP) under SLURM -- one task per rank (ranks-as-workers, no MPI spawn):
srun --ntasks-per-node=4 --mpi=pmix \
     --container-image=<trtllm image> \
     --container-mounts=<repo>:<repo>,<models>:<models>,<ns_acc_bench_infra>:<ns_acc_bench_infra> \
     --container-workdir=<repo> \
  bash -c 'trtllm-llmapi-launch trtllm-eval --model <path> --tp_size 4 --ep_size 4 gpqa_ns'
```

The final score is logged as `[evaluate] NeMo-Skills <bench> results: ...`.

### 3. Grading & LLM judges

The three deterministic benches (`gpqa_ns`, `ifbench`, `scicode_ns`) need no
judge. The three free-form benches (`hle_aa`, `aa_lcr`, `arena_hard_aa`) need an
LLM judge and **self-judge** by default — the in-process answering model also
grades. Self-judging is biased and **not** leaderboard-comparable; it is fine as
a regression guard. Set `NS_JUDGE_MODEL` to route the judge pass to an external
OpenAI-compatible endpoint instead (answers still come from the in-process model;
requires `pip install openai`). Unset → self-judge, so the default path is
unchanged. The result log shows the judge used, e.g. `… (judge=gpt-4.1, …)`.
Either way the judge pass uses its own sampling settings (default temperature
`0.0` → deterministic grading), not the answer-generation `--temperature`.

```bash
export NS_JUDGE_MODEL=gpt-4.1     # enables the external judge
export NS_JUDGE_API_KEY=...       # or OPENAI_API_KEY (required to enable it)
export NS_JUDGE_BASE_URL=...      # endpoint (see below); default api.openai.com/v1
# optional: NS_JUDGE_MAX_TOKENS (4096), NS_JUDGE_TEMPERATURE (0.0), NS_JUDGE_CONCURRENCY (16)
#   MAX_TOKENS / TEMPERATURE also apply to the self-judge (default) path
```

Use a **non-reasoning instruct** judge (reasoning models bury the verdict and
break the parsers):

| Endpoint (`NS_JUDGE_BASE_URL`) | Recommended judge (`NS_JUDGE_MODEL`) |
|--------------------------------|--------------------------------------|
| `https://api.openai.com/v1` (default) | `gpt-4.1`, `o3-mini` (AA-official for `hle_aa`) |
| `https://integrate.api.nvidia.com/v1` | `meta/llama-3.3-70b-instruct` (verified); `gpt-4.1`/`o3-mini` are **not** hosted here |
| `http://<host>:<port>/v1` (self-hosted) | what you serve, e.g. `Qwen3-235B-A22B-Instruct-2507` (AA-LCR official) |

## Example: Nemotron-3-Super-120B-A12B-FP8

A worked example on `NVIDIA-Nemotron-3-Super-120B-A12B-FP8` (4×GPU, tp4/ep4),
matched to the model's serving recipe: attention data parallelism, chunked
prefill, CUTLASS MoE, fp8 KV cache, and the Mamba SSM fp16 stochastic-rounding
cache. Write the engine options to a YAML:

```yaml
# super_eval.yaml
enable_attention_dp: true
enable_chunked_prefill: true
stream_interval: 50
kv_cache_config:
  dtype: fp8
  enable_block_reuse: false
  mamba_ssm_cache_dtype: float16
  mamba_ssm_stochastic_rounding: true
  mamba_ssm_philox_rounds: 5
cuda_graph_config:
  enable_padding: true
  max_batch_size: 8
moe_config:
  backend: CUTLASS
```

Reasoning is kept **on** and generation **uncapped** (a small `--max_output_length`
truncates long reasoning and deflates the score):

```bash
MODEL=<path to NVIDIA-Nemotron-3-Super-120B-A12B-FP8>
COMMON="--tp_size 4 --ep_size 4 --kv_cache_free_gpu_memory_fraction 0.7 \
  --max_batch_size 8 --max_num_tokens 65536 --max_seq_len 262144 \
  --trust_remote_code --disable_kv_cache_reuse --extra_llm_api_options super_eval.yaml"
SAMPLING="--temperature 1.0 --top_p 0.95 --top_k 0 --min_p 0.0 \
  --chat_template_kwargs '{\"enable_thinking\": true}' --max_output_length 260000"

# GPQA-diamond (198 questions)
trtllm-eval --model $MODEL $COMMON gpqa_ns $SAMPLING

# IFBench (300 prompts)
trtllm-eval --model $MODEL $COMMON ifbench $SAMPLING

# SciCode -- runs code in a sandbox (needs scipy/h5py/matplotlib + test_data.h5).
# --split test_aai: AA 80-problem set (dev+test); default `test` is only 65.
# NS_SCICODE_CONCURRENCY: problems solved in parallel; NS_SCICODE_TIMEOUT: sandbox timeout (s).
# The sandbox auto-starts on 127.0.0.1:6000 (NEMO_SKILLS_SANDBOX_HOST/PORT). It runs
# model-generated code, so binding a non-loopback host needs NEMO_SKILLS_SANDBOX_ALLOW_REMOTE=1.
NS_SCICODE_CONCURRENCY=64 NS_SCICODE_TIMEOUT=120 \
  trtllm-eval --model $MODEL $COMMON scicode_ns --split test_aai $SAMPLING
```

For the leaderboard-style pass@1 avg-of-k, add `--num_repeats 8` to any bench
(needs `--temperature > 0`).

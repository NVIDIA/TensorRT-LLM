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

## NeMo-Skills benchmarks (`gpqa_ns`, `ifbench`, `scicode_ns`, `hle_aa`, `aa_lcr`, `arena_hard_aa`)

`trtllm-eval` also exposes [NeMo-Skills](https://github.com/NVIDIA/NeMo-Skills)
benchmarks as in-process subcommands — `gpqa_ns` (GPQA-diamond), `ifbench`
(instruction following), `scicode_ns` (SciCode, multi-step scientific code),
`hle_aa` (Humanity's Last Exam, self-judged), `aa_lcr` (AA-LCR long-context
reading, self-judged), and `arena_hard_aa` (Arena-Hard, self-judged pairwise
win-rate vs the gpt-4-0314 baseline). They work lm-eval style: **install the
lib, then run**.

### Install (once)

[`install_nemo_skills.sh`](install_nemo_skills.sh) pip-installs `nemo_skills`
plus the grader deps from [`requirements_nemo_skills.txt`](requirements_nemo_skills.txt)
into the current Python env. It installs **only the lib** — the datasets and
grader assets come from the shared, read-only `ns_acc_bench_infra` folder, so there
is no download / dataset-prepare step:

```bash
# Run inside the TensorRT-LLM container.
bash examples/trtllm-eval/install_nemo_skills.sh
```

### Run

On first use the evaluators autowire the `ns_acc_bench_infra` folder (default
`<LLM_MODELS_ROOT>/datasets/ns_acc_bench_infra`, override with
`NS_ACC_BENCH_INFRA=<dir>`): they set the NeMo-Skills env, redirect the IFBench
grader and SciCode `test_data.h5` paths that NeMo-Skills hardcodes, and auto-start
the SciCode sandbox — so every bench just runs.

```bash
# Single GPU (CLI):
trtllm-eval --model <hf_or_path> gpqa_ns

# Multi-GPU (TP/EP) under SLURM -- one task per rank (ranks-as-workers, no MPI spawn):
srun --ntasks-per-node=4 --mpi=pmix \
     --container-image=<trtllm image> \
     --container-mounts=<repo>:<repo>,<models>:<models>,<ns_acc_bench_infra>:<ns_acc_bench_infra> \
     --container-workdir=<repo> \
  bash -c 'trtllm-llmapi-launch trtllm-eval --model <path> --tp_size 4 --ep_size 4 gpqa_ns'
```

The final score is logged as `[evaluate] NeMo-Skills <bench> results: ...`. See
[`NEMO_SKILLS_EVAL.md`](../../tensorrt_llm/evaluate/NEMO_SKILLS_EVAL.md) for
per-bench notes and the manual/custom-infra fallback.

### Per-bench notes

#### Grading method — which benches need an LLM judge

Only the free-form benches need an LLM judge; the others grade deterministically
(no judge, no `NS_JUDGE_*` setup). For the judge-based ones the default is to
**self-judge** (the in-process answering model also grades — biased, not
leaderboard-comparable, fine as a regression guard); set `NS_JUDGE_MODEL` to use
an external judge instead (see below).

| Bench | Grader | LLM judge? | Recommended external judge (`NS_JUDGE_MODEL`) |
|-------|--------|------------|-----------------------------------------------|
| `gpqa_ns`       | multiple-choice answer extraction | No  | — (deterministic) |
| `ifbench`       | allenai/IFBench rule checker      | No  | — (deterministic) |
| `scicode_ns`    | code execution in a sandbox       | No  | — (deterministic) |
| `hle_aa`        | LLM judge (`judge/hle`)           | Yes | `o3-mini` (AA-official); else a strong general model (`gpt-4.1`) |
| `aa_lcr`        | LLM equality checker (`judge/aalcr`) | Yes | `Qwen3-235B-A22B-Instruct-2507`, non-reasoning (AA-LCR official); else `gpt-4.1` |
| `arena_hard_aa` | pairwise LLM judge (`judge/arena`)   | Yes | `gpt-4.1` (NeMo-Skills default) |

- Self-judging (used by `hle_aa`, `aa_lcr`, `arena_hard_aa`) is biased — the
  same model both answers and grades — so these numbers are **not**
  leaderboard-comparable; they are regression guards.

#### Optional external judge (instead of self-judge)

The judge-based benches (`hle_aa`, `aa_lcr`, `arena_hard_aa`) self-judge by
default. Set `NS_JUDGE_MODEL` to instead route the judge pass to an external
OpenAI-compatible endpoint (answers still come from the in-process model;
requires `pip install openai`). Unset → self-judge, so the default path is
unchanged. The result log shows the judge used, e.g. `… (judge=gpt-4.1, …)`.

```bash
export NS_JUDGE_MODEL=gpt-4.1     # enables the external judge
export NS_JUDGE_API_KEY=...       # or OPENAI_API_KEY (required to enable it)
export NS_JUDGE_BASE_URL=...      # endpoint (see below); default api.openai.com/v1
# optional: NS_JUDGE_MAX_TOKENS (4096), NS_JUDGE_TEMPERATURE (0.0), NS_JUDGE_CONCURRENCY (16)
```

Pick the endpoint via `NS_JUDGE_BASE_URL` + a matching key, and use a
**non-reasoning instruct** judge (reasoning models bury the verdict and break the
parsers):

| Endpoint (`NS_JUDGE_BASE_URL`) | Key | Judge models (`NS_JUDGE_MODEL`) |
|--------------------------------|-----|---------------------------------|
| `https://api.openai.com/v1` (default) | OpenAI `sk-...` | `gpt-4.1`, `o3-mini` (AA-official) |
| `https://integrate.api.nvidia.com/v1` | NVIDIA `nvapi-...` | `meta/llama-3.3-70b-instruct` (verified); `gpt-4.1`/`o3-mini` are **not** hosted here |
| `http://<host>:<port>/v1` (self-hosted) | server key, or any placeholder | what you serve, e.g. `Qwen3-235B-A22B-Instruct-2507` (AA-LCR official) |

# Pinned GSM8K evaluation over a chat endpoint

`examples/llm-api/eval_gsm8k.py` evaluates the first N questions (1 to 1,319) of the GSM8K test
split through an OpenAI-compatible `/v1/chat/completions` endpoint. One implementation of the
question selection, prompt and request construction, answer extraction and scoring serves both
manual use and the [Beam baseline accuracy gate](#beam-baseline-accuracy-gate).

lm-eval schedules chat requests and limits concurrency. The evaluator records each raw reply
so the gate can check the finish reason, token usage and final-answer channel before scoring.

## Requirements

- Existing-endpoint mode: Python 3.10 or later with `lm_eval[api]==0.4.10`, `requests` and `pyarrow`. It needs no GPU, no
  model checkpoint and no TensorRT LLM installation, and it never starts, stops or signals the
  server.
- Managed mode: a TensorRT LLM installation that provides `trtllm-serve`, the GPUs the serving
  configuration needs, and `PyYAML`.
- The GSM8K dataset `openai/gsm8k`, configuration `main`, at revision
  `cc7b047b6e5bb11b4f1af84efc572db110a51b3c`, staged locally with this layout:

  ```text
  <DATASET>/main/train-00000-of-00001.parquet
  <DATASET>/main/test-00000-of-00001.parquet
  ```

  For example, `huggingface-cli download openai/gsm8k --repo-type dataset --revision
  cc7b047b6e5bb11b4f1af84efc572db110a51b3c --local-dir <DATASET>`. The evaluator reads only the
  staged files and checks every row before sending a request.

## Commands

Existing endpoint (an aggregated or disaggregated `trtllm-serve`, or a compatible
frontend):

```bash
python examples/llm-api/eval_gsm8k.py \
  --endpoint http://<HOST>:<PORT> \
  --dataset-dir <DATASET> \
  --num-questions 100 \
  --output-dir <RESULTS>
```

Managed server: the evaluator launches `trtllm-serve <MODEL> --config <CONFIG>` bound to
`127.0.0.1` on a free port, evaluates, and stops every process of that launch (and only those)
when the run ends, fails, times out or is interrupted:

```bash
python examples/llm-api/eval_gsm8k.py \
  --model <MODEL> --config <CONFIG> \
  --chat-template <CHAT_TEMPLATE> \
  --server-env NAME=VALUE \
  --dataset-dir <DATASET> \
  --num-questions 100 \
  --output-dir <RESULTS>
```

Arguments:

| Argument | Meaning |
|---|---|
| `--endpoint URL` / `--model DIR` | Exactly one: an existing server's base URL, or a model for a managed launch. |
| `--served-model-name NAME` | Model id to request; by default the only id listed by `/v1/models`. Required only when the endpoint returns HTTP 404 for `/v1/models`. |
| `--config YAML` | Managed mode: the `trtllm-serve --config` file (required). |
| `--chat-template FILE` | Managed mode: passed as `trtllm-serve --chat_template`. |
| `--chat-template-kwargs JSON` | Optional JSON object passed to the model's chat template and recorded in the run identity. Omit it to use the template defaults. |
| `--server-env NAME=VALUE` | Managed mode: extra server environment variable (repeatable). |
| `--monitor-gpus 0,1,...` | Managed mode: GPU indices whose used memory is sampled every second. |
| `--num-questions N` | Evaluate test rows `0..N-1` in split order, `1 <= N <= 1319` (default 100). |
| `--concurrency N` | Requests in flight. Default: managed mode the config's `max_batch_size`, endpoint mode 64. |
| `--time-budget-s S` | Whole-run budget including server startup (default 2700). Raise it for large N. |
| `--request-timeout-s S` | Optional per-request read timeout. |
| `--max-seq-len N` | The server's `max_seq_len`; managed mode reads it from the config. |
| `--min-correct N` | Also require at least N correct answers (a regression gate). |

Invalid arguments, including a question count outside 1..1319, are rejected before any server is
started or any request is sent. A known `max_seq_len` must exceed the 65,536-token
output allowance; otherwise the run is refused before dataset loading or server startup.
Managed mode checks the YAML's limit as well as any `--max-seq-len` override. This is
only a lower-bound check: the endpoint owns tokenization and chat-template rendering,
so the evaluator cannot determine the exact prompt length before a reply. Each reply's
reported prompt-token count is still checked against the available context length.
The full 1,319-question evaluation is a manual run; no CI configuration selects it.

The `trtllm-serve disaggregated` router provides `/v1/models`, but may return HTTP 503
until a context worker registers. Both endpoint and managed modes retry that response
within the startup budget (at most one hour, also bounded by the remaining run budget).
Managed mode also waits for `/health`. Other model-discovery HTTP errors, malformed
replies and model-name mismatches fail immediately.

Exit status: 0 when every selected question completed without an error, timeout or truncation
(and, with `--min-correct`, enough answers are correct); 1 when the run finished but did not
pass; 2 when it was refused before asking (bad arguments, unusable dataset, a results directory
of a different run, or a server that serves a different model than the recorded replies came
from); 3 for an unexpected execution or result-storage failure (including a write
failure after requests have started); 130 when interrupted. A failure with exit 3 is
not an accuracy result. Completed records are retained where storage permits.

## Protocol

The Beam-specific protocol id is `beam-gsm8k-first100-public-v1`;
`gsm8k_eval/manifest.json` freezes its settings, including `reasoning_effort: 128`. To reproduce or resume a historical Beam run, pass
`--chat-template-kwargs '{"reasoning_effort":128}'`. Beam CI supplies this explicitly.
Other template settings, including the default with no kwargs, use `gsm8k-chat-v1`;
changing the request fields requires a fresh output directory.

- Selection: test rows `0..N-1` in the original split order; question ids are `test-0000` to
  `test-1318`. The first 100 rows, the ten exemplars and every prompt digest are pinned by the
  manifest; all 1,319 rows (question, answer, prompt and gold answer) are pinned by
  `TEST_ROWS_SHA256` in `gsm8k_eval/protocol.py`, so any changed, missing or reordered row stops
  the run before a request is sent.
- Prompt: one user message holding ten training exemplars (`random.Random(42).sample` over the
  training split, as frozen in the manifest) followed by the question:
  `Question: <question>\nAnswer: <answer>\n\n` per exemplar, then `Question: <question>\nAnswer:`.
- Request: `temperature 0`, `top_k 1`, `seed 42`, `max_tokens 65536`, `n 1`, `stream false`,
  no stop strings. Template kwargs are sent only when supplied. The server applies its own
  chat template, tokenizer and end-of-sequence handling and must return the reasoning in
  `reasoning_content` (for example through a reasoning parser); the evaluator never renders or
  tokenizes text itself.
- Scoring: only `choices[0].message.content` is scored, never the reasoning. The last number in
  the content (optional sign, comma thousands separators, optional decimals) is normalized and
  compared exactly with the number after the last `####` of the gold answer. Empty content or
  content without a number is a parse failure and counts as incorrect.
- Failures: a non-200 reply, a malformed reply, final content that still contains channel
  delimiters such as `<|channel|>`, a `length` finish below the 65,536-token allowance, a prompt
  that cannot fit the allowance within `max_seq_len`, and any finish reason other than `stop` or
  `length` are errors; a reply that used the whole allowance is a truncation; a request without a
  reply before the budget is a timeout. Errors, timeouts, truncations and unanswered questions
  each fail the run regardless of the score.

### Regenerating the benchmark pins

For an intentional benchmark change, run this from the repository root against the original
Parquet files at the dataset revision listed above. Update `PROTOCOL_ID` first if the benchmark
definition changes.

```bash
python examples/llm-api/gsm8k_eval/protocol.py "$LLM_MODELS_ROOT/datasets/openai/gsm8k" \
  --output examples/llm-api/gsm8k_eval/manifest.json
```

The command writes the first-100 manifest and prints the `TEST_ROWS_SHA256` assignment for
all 1,319 rows. Copy that assignment into `protocol.py` and review both changes together.
The CPU test `test_staged_gsm8k_matches_committed_pins` checks the staged data against those
pins and skips when neither split is staged. A partially staged dataset fails the check.

## Outputs and resume

The server inherits the caller's environment, with `--server-env NAME=VALUE` overriding
individual variables. The evaluator records only the sorted names of these overrides,
never their values or hashes. Rotating a credential under the same variable name does not
invalidate resume. Environment values are not compared: if a change affects model behavior,
use a fresh output directory. Older managed-run manifests that stored `server_env` values
also require a fresh directory; existing artifacts are not automatically redacted.

The output directory receives:

- `results.jsonl`: one line per question attempt, written and flushed as each reply arrives: the
  run identity digest, question id, prompt digest, the request fields (the served model, and
  the message text replaced by its digest), status, error, HTTP status, finish reason, final
  content, reasoning size and digest, token usage, parsed and gold answers, correctness and
  latency.
- `run_manifest.json`: the run identity (protocol, manifest digest, selection, request fields,
  transport, endpoint or launch settings) and one record per attempt with the served model it
  asked, its timings, the dataset file digests and, in managed mode, the server command and
  GPU-memory peaks.
- `summary.json`: per-question outcomes, the score, the served model the replies came from, the
  lists of parse failures, errors, timeouts, truncations and missing questions, latency and
  token statistics, and the verdict.
- Managed mode also keeps the server log, the reported address and the GPU-memory samples of
  each attempt.

GPU cost fields are `null` because the evaluator cannot verify the server's device
count. In managed mode, `timing.monitored_gpu_hours` estimates the listed monitored
devices' GPU-hours over the server lifetime; it is not server utilization or total
server cost. `cost.monitored_gpu_hours_all_attempts` is `null` if any attempt lacks
that estimate, including older manifests. GPU-poller shutdown waits at most 10 seconds
after SIGTERM and another 10 after SIGKILL, then warns and continues saving results.

Running again with the same output directory and the same settings resumes the run: completed
replies are kept and never asked again, failed attempts stay recorded and their questions are
asked again, and an interrupted last line is moved aside. A different protocol, selection,
endpoint URL or recorded managed-launch configuration is refused; use a fresh output
directory. Results recorded under
the previous `gsm8k-first100-public-v1` identifier must also use a fresh output directory
with this version; the renamed protocol preserves the same prompts, requests and scoring.

The replies of one run must all use the same served model name. Before its first request, each attempt
records the model it asks: the `--served-model-name`, or else the only model that `/v1/models`
lists. If the recorded replies were asked of a different model, the attempt is refused before
any question is asked. The refusal is recorded in `run_manifest.json`, and a managed
server is stopped. Completed records are preserved; loading the ledger can quarantine
and truncate an incomplete trailing line before this refusal is decided. A results directory whose replies come from more
than one model is never resumed. A run whose selected questions all have completed replies asks
nothing and contacts no server; its summary names the model those replies came from. The
evaluator identifies a model only by its served name, so it cannot detect different weights
served under an unchanged name.

**Endpoint resume trusts the operator.** The evaluator does not fingerprint the
server build, weights, tokenizer/chat template, precision or serving configuration.
An unchanged URL and model name do not prove that these are unchanged. If the server
is upgraded, reconfigured or restarted with different inputs, use a fresh output
directory; otherwise a resumed score may combine incompatible runs. Resume after a
restart only when you can establish that the model and evaluation-relevant settings
are unchanged. Managed mode records its launch settings, but also cannot detect
weights or a server installation replaced in place.

## Beam baseline accuracy gate

`TestModelingV2BeamFullModelNvfp4Sm103Tp4::test_gsm8k_first100` in
`tests/integration/defs/accuracy/modeling_v2/beam/full_model_nvfp4__sm_103__tp4/test_accuracy.py`
uses this evaluator through a managed `trtllm-serve` launch. It requires exactly four visible
GB300 GPUs and selects the NVFP4 Beam target with `modeling_v2: require`, TP4, BF16 KV cache,
no speculative decoding, CUDA graphs disabled and overlap scheduling disabled. Batch size and
request concurrency are both 1.

Stage an authorized target below `LLM_MODELS_ROOT` and set `TRTLLM_TEST_TARGET_RELATIVE_PATH`
to its relative directory. Stage the pinned dataset at `LLM_MODELS_ROOT/datasets/openai/gsm8k`.
A selected run fails if required inputs are missing. The gate evaluates the first 100 questions,
requires at least 95 correct answers and rejects execution errors, timeouts and truncations.
Its 45-minute test timeout reserves five minutes for cleanup, leaving at most 40 minutes for
server startup and evaluation. The generated `beam-baseline.yaml` and evaluator outputs are
written to the test's temporary output directory.

QA selects this gate through `tests/integration/test_lists/qa/functional/llm_function_core.txt`.
Full 1,319-question GSM8K evaluation remains manual-only.

## Comparing results

Compare runs using the same dataset revision and question selection, prompt and request
fields, scoring rules and protocol manifest. Also record the checkpoint, tokenizer/chat
template, serving configuration, precision and concurrency; sharing this evaluator alone
does not establish equivalent model accuracy. The baseline and an optimized serving
configuration are separate measurements. Re-scoring saved replies checks scoring consistency;
it does not verify that a different serving configuration generates equivalent replies.

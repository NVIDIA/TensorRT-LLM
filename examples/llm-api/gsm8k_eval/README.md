# Pinned GSM8K evaluation over a chat endpoint

`examples/llm-api/eval_gsm8k.py` evaluates the first N questions (1 to 1,319) of the GSM8K test
split through an OpenAI-compatible `/v1/chat/completions` endpoint. One implementation of the
question selection, prompt and request construction, answer extraction and scoring serves both
manual use and the first-100 CI job
(`tests/integration/defs/accuracy/test_nvfp4_serving_gsm8k.py`).

## Requirements

- Existing-endpoint mode: Python 3.10 or later with `requests` and `pyarrow`. It needs no GPU, no
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

Existing endpoint (an aggregated or disaggregated `trtllm-serve`, or any OpenAI-compatible
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
| `--served-model-name NAME` | Model id to request; by default the only id listed by `/v1/models`. Required for endpoints without `/v1/models`, such as the `trtllm-serve disaggregated` router. |
| `--config YAML` | Managed mode: the `trtllm-serve --config` file (required). |
| `--chat-template FILE` | Managed mode: passed as `trtllm-serve --chat_template`. |
| `--server-env NAME=VALUE` | Managed mode: extra server environment variable (repeatable). |
| `--monitor-gpus 0,1,...` | Managed mode: GPU indices whose used memory is sampled every second. |
| `--num-questions N` | Evaluate test rows `0..N-1` in split order, `1 <= N <= 1319` (default 100). |
| `--concurrency N` | Requests in flight. Default: managed mode the config's `max_batch_size`, endpoint mode 64. |
| `--time-budget-s S` | Whole-run budget including server startup (default 2700). Raise it for large N. |
| `--request-timeout-s S` | Optional per-request read timeout. |
| `--max-seq-len N` | The server's `max_seq_len`; managed mode reads it from the config. |
| `--min-correct N` | Also require at least N correct answers (a regression gate). |

Invalid arguments, including a question count outside 1..1319, are rejected before any server is
started or any request is sent. The full 1,319-question evaluation is a manual run; no CI
configuration selects it.

Exit status: 0 when every selected question completed without an error, timeout or truncation
(and, with `--min-correct`, enough answers are correct); 1 when the run finished but did not
pass; 2 when it was refused before asking (bad arguments, unusable dataset, a results directory
of a different run, or a server that serves a different model than the recorded replies came
from); 130 when interrupted.

## Protocol

The protocol id is `gsm8k-first100-public-v1`; `gsm8k_eval/manifest.json` freezes it.

- Selection: test rows `0..N-1` in the original split order; question ids are `test-0000` to
  `test-1318`. The first 100 rows, the ten exemplars and every prompt digest are pinned by the
  manifest; all 1,319 rows (question, answer, prompt and gold answer) are pinned by
  `TEST_ROWS_SHA256` in `gsm8k_eval/protocol.py`, so any changed, missing or reordered row stops
  the run before a request is sent.
- Prompt: one user message holding ten training exemplars (`random.Random(42).sample` over the
  training split, as frozen in the manifest) followed by the question:
  `Question: <question>\nAnswer: <answer>\n\n` per exemplar, then `Question: <question>\nAnswer:`.
- Request: `temperature 0`, `top_k 1`, `seed 42`, `max_tokens 65536`, `n 1`, `stream false`,
  `chat_template_kwargs: {"reasoning_effort": 128}`, no stop strings. The server applies its own
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

## Outputs and resume

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

Running again with the same output directory and the same settings resumes the run: completed
replies are kept and never asked again, failed attempts stay recorded and their questions are
asked again, and an interrupted last line is moved aside. A different protocol, selection,
endpoint or configuration is refused; use a fresh output directory.

The replies of one run all come from one served model. Before its first request, each attempt
records the model it asks: the `--served-model-name`, or else the only model that `/v1/models`
lists. If the recorded replies were asked of a different model, the attempt is refused before
any question is asked. The refusal is recorded in `run_manifest.json`, `results.jsonl` is left
unchanged, and a managed server is stopped. A results directory whose replies come from more
than one model is never resumed. A run whose selected questions all have completed replies asks
nothing and contacts no server; its summary names the model those replies came from. The
evaluator identifies a model only by its served name, so it cannot detect different weights
served under an unchanged name.

## Comparability

The prompt, request fields, scorer and manifest are those of the first-100 GSM8K job used
before this evaluator; first-100 results of the two are directly comparable. Replaying the
saved replies of earlier first-100 runs through this command line reproduces their question
ids, request records, statuses, parsed answers and scores exactly.

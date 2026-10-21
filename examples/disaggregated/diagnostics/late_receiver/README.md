<!-- Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. -->

# Late receiver lifecycle characterization

This temporary diagnostic forces a context-first request's receiver to arrive
after its context sender has retired. It tests an ordering hypothesis; it does
not fix the lifecycle, establish why a real workload delayed generation, or
replace the original stress test.

The baseline is source `da676ffb8b611a1a0a83c862aa094e736f9fe5aa`.
Use the matching published runtime `1.4.0rc1.dev202610140007`,
pinned to `nvcr.io/nvidia/tensorrt-llm/release@sha256:858d6e09f4f4de2d188ac2aaef12136a46c4ab19da4b3a18b6c07b2b30503071`.
For a serving experiment, apply only this diagnostic's Python changes to a
verified disposable runtime overlay, record those file hashes separately, and
verify the unchanged native libraries. Merely mounting the checkout beside an
installed wheel does not activate the hooks. No TensorRT-LLM rebuild is needed
for this Python-only diagnostic.

## First run the deterministic lifecycle test

In a compatible Linux TensorRT-LLM test environment:

```sh
python -m pytest tests/unittest/_torch/disaggregation/characterize_late_receiver_retirement.py -m cpu_only -v
```

The test uses the real coordinator, transfer manager, transceiver, TxSession,
and Sender. It substitutes time, memory descriptions, network endpoints and
backend completion. It needs native imports but no model or GPU computation.
A passing characterization means the unanswered-late-receiver behavior was
observed, not fixed. The early-receiver and boundary controls must also pass.

This explicitly named file is excluded from ordinary `test_*` discovery. It is
a local/QA characterization, not a permanent assertion requiring defective behavior.
Once a fix is selected, replace that characterization with the desired bounded
terminal outcome and add appropriate CI coverage.

## Prepare one dedicated serving experiment

Use one authorized node, the original GPT-OSS120B/Eagle setup, TP4 context and
TP4 generation with attention DP enabled, context-first Python NIXL, PP1/CP1,
and physical ownership disabled. Keep the original 60-second KV deadline.
Do not use the FP4 MLA ownership profile. Keep production behavior unchanged
when the diagnostic environment variable is absent.

Provide the same absolute diagnostic directory to every worker container,
router process and driver on the node. Use a new run directory on node-local
storage; retain its small evidence files to persistent storage while access is
valid. It must not be exposed to untrusted users or a public serving endpoint.

Before starting all server processes:

```sh
export TRTLLM_LATE_RECEIVER_DIAGNOSTIC_DIR=/path/to/new-run/late-receiver
export TRTLLM_DISAGG_NO_RETRY=1
```

Disable retries in both arms, before clients are constructed. Complete ordinary
readiness/warmup while the diagnostic is unarmed. Send no background traffic
during the experiment: each arm claims the first eligible request.

Use the existing no-rebuild launch environment, model paths and configs to start
the two workers and router. Do not invoke the 60,000-request stress selector for
this diagnostic. Confirm hooks are imported from the recorded overlay and
preserve full context, generation and router logs.

## Run a no-delay control and three delayed requests

Prepare a non-streaming completion or chat-completion JSON payload for the
original model. Do not supply disaggregated request parameters or IDs. Use the
same payload for both arms; the server must assign a fresh ID each time.

```sh
python examples/disaggregated/diagnostics/late_receiver/run.py \
  --diagnostic-dir /path/to/new-run/late-receiver \
  --url http://127.0.0.1:8000/v1/completions \
  --payload /path/to/request.json \
  --repetitions 3
```

The driver does not allocate compute or start servers. It runs one no-delay
control, then three delayed trials. Each delayed trial waits for an actual,
matched context sender retirement before allowing generation handoff; it does
not release merely because 60 seconds elapsed or a timeout warning appeared.
Attention DP means retirement is matched to the returned context DP owner,
not blindly to four ranks.

The default gate limit is 90 seconds and HTTP limit is 240 seconds. The
independent generation deadline must fit after the gate. Missing markers,
unsupported layouts, client timeouts, mismatched IDs and unrelated errors make
the experiment inconclusive. An inconclusive control or trial stops the driver;
there is no automatic retry.

Each trial has immutable lifecycle event files and a `client-result.json`.
The latter contains the HTTP response, which can contain generated text; retain
it with the same access controls as test logs. Lifecycle events contain IDs and
timing, not prompts or tokens.

Enabled tracing adds small node-local file writes, including one snapshot while
holding the sender session lock. This experiment changes timing and is not a
performance measurement; all hooks are inactive without the opt-in variable.

## Interpret the result

- `CONTROL_OK`: the request succeeded and receiver metadata found a live sender.
- `MECHANISM_REPRODUCED`: matched retirement precedes late receiver metadata,
  that metadata finds no live unfenced sender, and the request returns the
  product's KV-transfer timeout.
- `INCONCLUSIVE`: required evidence is missing, ambiguous, or shows another outcome.

Exit zero means all characterization arms met their expected observations.
It is not a product PASS or a validated fix. A successful delayed request or an
immediate explicit terminal failure is valuable contrary evidence; inspect the
trace rather than weakening the classifier.

Keep router release, generation receive publication and generation timeout
clock start distinct. A gate is not proof of scheduler admission. A failed-reply
attempt is not proof of network delivery. Compare monotonic timestamps only
when all events belong to the same host.

After reproducing, choose a focused lifecycle/admission or explicit-terminal
response fix. An expired sender must not silently resurrect. Test that fix with
the same injection and controls, then validate the original targeted stress
case. Do not use a longer timeout as evidence that the lifecycle was corrected.

## Local controller checks

These checks validate the diagnostic tooling only; they do not execute native
TensorRT-LLM lifecycle code:

```sh
python tests/unittest/disaggregated/test_late_receiver_diagnostic.py
python examples/disaggregated/diagnostics/late_receiver/test_run.py
```

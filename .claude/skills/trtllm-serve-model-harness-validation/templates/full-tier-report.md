<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Full-tier report: what the validator agent writes

The full tier is the only verdict that sets a cell's status (`SKILL.md` section 4a,
`references/validate.md` section 5). It runs in a different session from the one that produced the
evidence. It writes five files into `<snapshot>/full-tier/`. Copy the skeletons below and fill them
in.

## 1. Steps

1. **Verify the snapshot.** `sha256sum -c MANIFEST.sha256` in the snapshot. A mismatch stops the
   tier: the evidence is BLOCKED, not judged.
2. **Qualify the validator.** Use a fresh environment from `scripts/validate/requirements.txt`, then
   run `python scripts/validate/controls.py --out-dir <dir>`. Every case must pass. The receipt is
   bound to the validator tree it ran on.
3. **Run the verdict:**

   ```bash
   python scripts/validate/verdict.py --tier full --controls-receipt <dir>/controls.json \
       --harness <profile> --model <profile> --matrix <matrix> --population <kind> \
       --build-id <the recorder's id> [--raw <raw dir>] --out-dir <snapshot>/full-tier <trace>
   ```

   This writes `verdict.json`.
4. **Re-derive the key numbers with your own code** from the raw bytes: not with
   `scripts/validate/`, and not with the producing session's scripts. For each number, say how you
   got it.
5. **Write the five files** below. Never quote trace content: use locators
   `(file, line, trace_id)`, lengths and SHA-256.

## 2. `run-manifest.json`

```json
{
  "schema": "trtllm-serve-model-harness-validation/run-manifest@1",
  "cell": {"model": "<model id>", "harness": "<harness id>", "stage": "S4", "iteration": "<id>"},
  "snapshot": {"path": "<snapshot>", "manifest_sha256": "<sha256 of MANIFEST.sha256>",
               "files": 0, "bytes": 0, "verified": true},
  "build_id": "<version>+<commit>",
  "provenance_sha256": "<sha256 of the trial build's provenance file, or null on main>",
  "validator": {"tree_sha256": "<tree>", "controls_receipt_sha256": "<sha256>", "controls_passed": true,
                "sdk_versions": {"anthropic": "<x>", "openai": "<y>"}},
  "profiles": {"model_sha256": "<sha256>", "harness_sha256": "<sha256>"},
  "matrix_sha256": "<sha256>",
  "agent": "<who ran this tier: a session id or name, not the producing session>"
}
```

## 3. `validation-report.json`

Each claim, with what decided it and the re-derived numbers next to the verdict's:

```json
{
  "schema": "trtllm-serve-model-harness-validation/validation-report@1",
  "claims": {"inventory": "PASS", "conformance": "PASS", "runtime": "PASS",
             "model-behavior": "PASS", "harness-outcome": "BLOCKED"},
  "claim_blockers": {"harness-outcome": ["H-01:aborted: threshold not set in the matrix"]},
  "rederived": [
    {"what": "requests with a reply", "verdict": 0, "rederived": 0, "agree": true,
     "how": "joined requests and responses by trace_id in my own script"},
    {"what": "C-13 replies compared / equal", "verdict": [0, 0], "rederived": [0, 0], "agree": true,
     "how": "read the raw records with my own reader of the model's format"}
  ],
  "not_exercised": ["no request ran over length", "no call was abandoned"]
}
```

Any `"agree": false` is a finding. Until it is explained, the claim it feeds is BLOCKED.

## 4. `findings.jsonl`

One line per finding:

```json
{"id": "F01", "severity": "high|medium|low|info", "owner": "serving|model|harness|validator|evidence",
 "check": "C-13", "locator": {"file": "trace/responses-x.jsonl", "line": 0, "trace_id": "<id>"},
 "what": "<one sentence, no trace content>", "next": "<PR, validator fix, or the human's decision>"}
```

- A `validator` finding is fixed as a policy change: a separate commit, controls re-run, the cell
  re-judged.
- A `serving` finding gets a reproducer and a PR (`SKILL.md` section 4).

## 5. `SUMMARY.md`

For the human, in this order:

1. **Each claim's status**, with its tier and inputs: the population, the profiles, the build id,
   and whether `--raw` was given.
2. **The re-derived numbers that agree**, and every one that does not.
3. **Findings by severity**, each with its owner and next step.
4. **What the run did not exercise.**
5. **The cell's status from this tier:** PASS only when every claim the cell requires is PASS.

## 6. `verdict.json`

Written by `verdict.py --tier full` in step 3. The format is in `references/validate.md` section 6.

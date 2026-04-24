# Bucket Rules

## Bucketing Criteria

Group failures together only when ALL of these are true:
- same likely code owner and target repo
- same causal failure signature (failing symbol, op, assertion, stack frame, or config path)
- fixable by one coherent code change
- one PR can explain why the same fix covers every matched job

Split failures into different buckets when ANY of these are true:
- the first causal error differs even if the legacy category matches
- the same symptom comes from different repos or subsystems
- one failure is infrastructure noise and the other is a code bug
- the likely fixes would touch unrelated files
- the evidence is mixed or contradictory

When uncertain, split instead of merge.

## Evidence Priority

Use this order when bucketing:
1. First causal stack frame or assertion
2. Explicit failing symbol, op, layer, config key, or script
3. Repeated error snippet near the first failure
4. Repeated failure wording across matched traces
5. Job naming and workload metadata only as a weak tie-breaker

## Required Bucket Fields

Each bucket must have:
- a short name in the form `repo/component/failure-mode`
- one representative job
- a list of all matching jobs
- one root-cause hypothesis tied to code
- label: `actionable` or `issue-only`

## Infra and External Buckets

Every job must end in a bucket -- including infra/external cases. Example buckets:
- `infra/resource/oom`
- `infra/runtime/timeout-or-freeze`
- `infra/runtime/cancelled`
- `infra/filesystem/hf-lock-permission`
- `external/huggingface/access-forbidden`
- `external/huggingface/missing-revision`
- `external/huggingface/invalid-tokenizer-or-processor`
- `external/env/missing-python-package`
- `external/transformers/api-mismatch`

Do NOT assume `oom` or `timeout-or-freeze` are infra-only. In AutoDeploy pipelines they often reflect real TensorRT-LLM bugs. Classify as `infra/...` only when evidence points to cluster noise or non-code resource problem.

## Skip Rules (No PR)

Do NOT create a PR for a bucket when any of these are true:
- pure infrastructure noise (timeout, preemption, cancellation, log-access failure) without code evidence
- jobs do not share one plausible code fix
- evidence too weak to point at a concrete code path
- belongs to external infrastructure or dependency outside checked-out repos
- an open PR already addresses the same bucket
- only commonality is a broad status label or superficial wording

If the starting pipeline failed only because a bridge failed, do not treat the bridge as its own actionable bucket unless the downstream has no failing jobs or no accessible traces.

Infra and external buckets must still be reported explicitly -- usually as `issue-only`.

## Common Issue-Only Patterns

- Gated or forbidden Hugging Face repos (403)
- Missing or renamed Hugging Face revisions/models (404)
- Missing optional Python packages (timm, num2words, mamba_ssm, causal_conv1d, etc.)
- Filesystem permission on HF cache lock files
- Only clearly non-code resource failures after log review

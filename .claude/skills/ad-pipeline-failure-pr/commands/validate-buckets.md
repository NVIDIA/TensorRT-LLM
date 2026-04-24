# validate-buckets

## Trigger

After gather-evidence produces initial bucket assignments.

## Behavior

For every bucket:
1. Read the representative job log and isolate the first causal failure (not downstream fallout).
2. Read the relevant code, config, or script the failure points to.
3. Confirm the same hypothesis explains all jobs in the bucket.
4. If deeper AutoDeploy tracing is needed, use the `ad-debug-agent` workflow to inspect the failing code path.
5. If the representative log does not support the bucket hypothesis, split or discard the bucket.

## Gate

Do not proceed to create-fixes until each bucket has both:
- one representative log snippet
- one code-level hypothesis

## Output

Validated bucket list with representative evidence per bucket.

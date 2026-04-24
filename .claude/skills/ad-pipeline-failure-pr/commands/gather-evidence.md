# gather-evidence

## Trigger

After resolve-scope produces the terminal pipeline ID.

## Behavior

For each failed job, collect:
- pipeline ID, job ID, job URL, raw log URL
- workload name, model/benchmark configuration
- first causal error snippet from the raw trace
- whether the job came from a bridge-followed downstream path

## Trace-Reading Rules

- In `model-coverage` terminal pipelines, jobs often come in triplets: `[1 logs_before]`, `[2 <runner/stage>]`, `[3 logs_after]`. The primary failing workload is usually `[2 ...]`. Use `[1]` and `[3]` only as supplemental evidence.
- If the trace ends with generic wrapper failures (`RuntimeError: Executor worker returned error`, `RuntimeError: Executor worker died during initialization`, `Process exited with status 1`), scan upward for the earlier model/export/tokenizer/environment-specific exception.
- Prefer the first specific exception over later fallout from worker teardown, Slurm cleanup, or proxy startup.
- When the workload dumps its config, capture the resolved `model:` value and relevant `yaml_extra`/runtime hints.

## Bucketing

Apply rules from `references/bucket-rules.md`. Every job must end in exactly one bucket.

## Output

Per-job evidence records + initial bucket assignments.

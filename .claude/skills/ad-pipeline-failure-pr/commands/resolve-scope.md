# resolve-scope

## Trigger

Entry point. Run when user provides a pipeline ID/URL or asks to analyze the latest pipeline.

## Behavior

1. Default scope is `model-coverage`. Do not silently switch to benchmark pipelines.
2. If the user asks to analyze a benchmark pipeline, stop -- this skill does not support benchmark pipelines.
3. If the user gives a pipeline ID or URL, use it.
4. Treat a user-provided pipeline as potentially either:
   - an upstream AutoDeploy pipeline in `ftp/infra/autodeploy-dashboard`
   - a downstream triggered pipeline in `dl/jet/ci`

## Pipeline Resolution Order

1. Identify whether the pipeline belongs to the upstream dashboard project or downstream `dl/jet/ci`.
2. If upstream, inspect bridge jobs and select the failed `model-coverage` trigger path.
3. If the next pipeline contains only bridge jobs, keep following the failed trigger chain.
4. Stop at the first downstream pipeline with terminal failed `model-coverage` jobs with traces.
5. If no pipeline ID given, resolve the latest upstream AutoDeploy pipeline that ran `model-coverage`, then follow the same bridge chain.

## Output

Report both:
- the user-facing starting pipeline
- the terminal pipeline containing the actual failing jobs

Do not analyze only the bridge failure if a deeper downstream pipeline contains the real job traces.

---
name: ad-run-agent
description: Run AutoDeploy build and run command for a given model using gpu-shell
tools: Read, Grep, Glob, Bash, Write, Edit, gpu-shell
model: sonnet
---

Run a model with AutoDeploy using gpu-shell. If you are not given the model-id, config, and a short description of the run, ask the user/caller first.

## Setup

Before the first run, create the logs directory and worklog file in the user's PWD:
- Logs directory: `$PWD/ad-test-workspace/ad_run_logs/`
- Worklog file: `$PWD/ad-test-workspace/ad_run_logs/<MODEL_SHORT_NAME>_worklog.md` (e.g. `qwen3.5_moe_400b_worklog.md`)

Derive `<MODEL_SHORT_NAME>` from the config yaml filename (strip the `.yaml` extension).

If the worklog file already exists, append to it. Never overwrite previous entries.

## Workflow

### 1. Run the AD build and run command

Execute via gpu-shell since it requires GPU access:
```bash
AD_DUMP_GRAPHS_DIR=<AD_DUMP_GRAPHS_DIR> python examples/auto_deploy/build_and_run_ad.py \
  --model <MODEL_HF_ID> \
  --args.yaml-extra examples/auto_deploy/model_registry/configs/<CONFIG_YAML_FILE> \
  2>&1 | tee <LOG_FILE>
```
Where:
- `AD_DUMP_GRAPHS_DIR` — directory where graphs will be dumped (auto-created by the script)
- `<MODEL_HF_ID>` — HF model-id or local path to a model checkpoint
- `<CONFIG_YAML_FILE>` — configuration file for the model
- `<LOG_FILE>` — temporary log file to capture output (use a timestamped name like `ad_run_<MODEL_SHORT_NAME>_<YYYYMMDD_HHMMSS>.log`)

### 2. Evaluate the output

By default, if compilation succeeds, `build_and_run_ad.py` runs example prompts and generates text.

- **If compilation failed**: Extract the error from the log. Summarize it.
- **If compilation succeeded**: Check the generated text in the log output. Evaluate whether the generation is meaningful — i.e., it is coherent, on-topic, and not garbled/repetitive nonsense. Report your assessment.

### 3. Archive the log

After the run completes, move the log file into the logs directory with a descriptive name:
```
$PWD/ad-test-workspace/ad_run_logs/<MODEL_SHORT_NAME>_<YYYYMMDD_HHMMSS>_<STATUS>.log
```
Where `<STATUS>` is one of: `success`, `compile_error`, `runtime_error`, `bad_generation`.

### 4. Update the worklog

Append a new entry to the worklog markdown file (`$PWD/ad-test-workspace/ad_run_logs/<MODEL_SHORT_NAME>_worklog.md`) with the following format:

```markdown
## Run: <YYYY-MM-DD HH:MM:SS>

**Description:** <description provided by caller>
**Model:** <MODEL_HF_ID>
**Config:** <CONFIG_YAML_FILE>
**Log:** <path to archived log file>
**Status:** <success | compile_error | runtime_error | bad_generation>

### Summary
<Brief summary of what happened — if error, describe the error; if success, summarize the generation quality>
```

If the worklog file does not exist yet, create it with a header:
```markdown
# AutoDeploy Run Worklog: <MODEL_SHORT_NAME>

---

```
Then append the first entry.

## Notes

- Always ask for the run description from the caller before starting.
- If it failed, show the relevant error and suggest next steps.
- Remember to use your own tools — Read, Grep, Glob, Bash, Write, Edit, gpu-shell

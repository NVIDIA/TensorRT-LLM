<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TensorRT-LLM VisualGen Perf Sanity (`test_visual_gen_perf_sanity.py`)

Performance sanity testing for VisualGen models is implemented by
`tests/integration/defs/perf/test_visual_gen_perf_sanity.py`.

This document is the VisualGen counterpart to `README_test_perf_sanity.md`.
For the shared regression pipeline architecture, see
`README_perf_regression_system.md`.

## How `test_visual_gen_perf_sanity.py` Uses the Pipeline

`test_visual_gen_perf_sanity.py` is a test-specific frontend on top of the
shared perf regression pipeline. It is responsible for:

1. Parsing VisualGen YAML configs from `tests/scripts/perf-sanity/visual_gen/`
2. Expanding one or more named `server_configs`
3. Launching `trtllm-serve`
4. Running `benchmark_visual_gen.py` and validating the saved JSON result
5. Building `new_data_dict` and VisualGen-specific `match_keys`
6. Calling `process_and_upload_test_results()`

## Metric Definitions

| List | Contents |
|------|----------|
| `MAXIMIZE_METRICS` | `d_request_throughput`, `d_per_gpu_throughput` |
| `MINIMIZE_METRICS` | `d_mean_e2e_latency`, `d_median_e2e_latency`, `d_p90_e2e_latency`, `d_p99_e2e_latency` |
| `REGRESSION_METRICS` | `d_mean_e2e_latency` |

## Match Keys

VisualGen currently uses these match keys:

```
s_gpu_type
s_runtime
s_model_name
l_gpus
s_attn_backend
s_quant_algo
b_enable_teacache
b_enable_cuda_graph
b_enable_torch_compile
b_enable_two_stage
l_cfg_size
l_ulysses_size
l_parallel_vae_size
s_generation_mode
s_backend
s_size
l_num_frames
l_fps
l_num_inference_steps
l_max_concurrency
```

Two fields are derived instead of copied directly:

- `b_enable_two_stage` becomes true when the merged `server_config` contains
  `spatial_upsampler_path` or `distilled_lora_path`
- `s_generation_mode` comes from `client_configs[].generation_mode` when
  present, otherwise it is inferred from the backend and request body

## Config Files

### Location

- `tests/scripts/perf-sanity/visual_gen/`

The config folder can be overridden with:

- `VISUAL_GEN_PERF_SANITY_CONFIG_FOLDER`

### Current Naming Pattern

VisualGen family YAMLs currently use underscore-separated names with a hardware
suffix, for example:

- `wan21_t2v_14b_blackwell.yaml`
- `wan22_i2v_a14b_blackwell.yaml`
- `flux2_blackwell.yaml`
- `ltx2_blackwell.yaml`

### YAML Structure

Each file can hold one or more named `server_configs`. This mirrors the LLM
aggregated perf sanity layout and allows one family YAML to contain multiple
stable server recipes.

```yaml
metadata:
  model_name: Wan2.1-T2V-14B-Diffusers
  supported_gpus: [B200]

hardware:
  gpus_per_node: 8

environment:
  server_env_var: TLLM_LOG_LEVEL=info
  client_env_var: ""

server_configs:
  - name: wan21_14b_nvfp4_trtllm_cfg2_ulysses4_teacache_on
    model_path: Wan-AI/Wan2.1-T2V-14B-Diffusers
    extra_visual_gen_options_path: examples/visual_gen/serve/configs/foo.yml
    server_config:
      attention:
        backend: TRTLLM
      parallel:
        dit_cfg_size: 2
        dit_ulysses_size: 4
    client_configs:
      - name: 832x480_33f_50s_con1
        backend: openai-videos
        generation_mode: t2v
        size: 832x480
        num_frames: 33
        fps: 16
        num_inference_steps: 50
        max_concurrency: 1
```

### Schema Notes

- `server_configs` (list) is **required**; the file is rejected if it is absent
  or not a list
- `metadata.model_name` is required and acts as the default OpenSearch model key
  for all recipes in the file
- `server_configs[].name` is required and is used in pytest selectors and
  `s_test_case_name`
- `server_configs[].model_name` is optional; when set it overrides
  `metadata.model_name` as the OpenSearch key for that recipe only
- `server_configs[].model_path` is optional and allows the runtime serve path to
  differ from the stable OpenSearch `model_name`; defaults to `model_name` when
  omitted
- `extra_visual_gen_options_path` is optional; when present, it is merged with
  inline `server_config`, and the merged config is treated as the runtime truth
- `hardware.gpus_per_node` must match the GPU count derived from
  `server_config.parallel`
- `client_configs[].generation_mode` should be set explicitly for stable
  bucketing, especially for `i2v` and `t2v`
- `client_configs[].extra_body.input_reference` is the current way to express
  `i2v` requests in checked-in YAMLs

## Test Case Formats

VisualGen uses two prefixes:

- `vg_upload`: run the case and upload new results to OpenSearch
- `vg`: run the case without posting new results

`{config_base}` is the YAML filename without the `.yaml` extension (e.g.,
`flux2_blackwell`, `wan21_t2v_14b_blackwell`).

Both prefixes support two selector forms:

1. Run all server recipes in a family YAML:

```text
perf/test_visual_gen_perf_sanity.py::test_visual_gen_e2e[vg_upload-{config_base}]
```

2. Run one named server recipe:

```text
perf/test_visual_gen_perf_sanity.py::test_visual_gen_e2e[vg_upload-{config_base}-{server_name}]
```

Example:

```text
perf/test_visual_gen_perf_sanity.py::test_visual_gen_e2e[vg_upload-wan21_t2v_1p3b_blackwell-wan21_1p3b_fp8_fa4_cfg1_ulysses1_teacache_on]
```

## CI Integration

VisualGen perf sanity is wired through the normal test-db and Jenkins paths:

- Test-db:
  - `tests/integration/test_lists/test-db/l0_b200_visual_gen_perf_sanity.yml`
- Jenkins stage definitions:
  - `jenkins/L0_Test.groovy`

The current B200 test-db has a single `post_merge` block (8 GPUs). All
priority-0 VisualGen cases run as post-merge; there is no pre-merge VisualGen
perf stage today. Because the total case count is small and all cases share the
same hardware tier, they live in a single test-db file. When the case count
grows or pre-merge coverage is needed, add a new `pre_merge` block to the same
file or introduce a separate test-db file and a new Jenkins stage entry.

### Triggering CI

To run only the VisualGen perf sanity stage (without the full L0 suite):

```
/bot run --post-merge --stage-list "DGX_B200-8_GPUs-PyTorch-VisualGen-PerfSanity-Post-Merge-1"
```

To include the VisualGen perf sanity stage alongside the standard L0 suite:

```
/bot run --extra-stage "DGX_B200-8_GPUs-PyTorch-VisualGen-PerfSanity-Post-Merge-1"
```

Model assets are read from `LLM_MODELS_ROOT` (defaults to
`/home/scratch.trt_llm_data_ci/llm-models/`). The `model_path` in a YAML may
use the HuggingFace `org/model` format (e.g. `Wan-AI/Wan2.1-T2V-14B-Diffusers`)
— the resolver tries the full path under `LLM_MODELS_ROOT` first, then falls
back to stripping the org prefix, so CI layouts without subdirectories work
automatically.

Important:

- VisualGen perf sanity uses static pytest cases, like `test_perf_sanity.py`
- Do **not** add pytest `--perf` for VisualGen perf sanity runs
- VisualGen `PerfSanity` stages should not be routed through the legacy
  `perfMode` path used by `test_perf.py`

## How to Add a Test Case

1. **Locate (or create) the family YAML** under
   `tests/scripts/perf-sanity/visual_gen/`. Use an existing file if the model
   family already has one; otherwise create a new file following the naming
   pattern `<family>_<hardware>.yaml`.

2. **Add a `server_configs` entry.** Each entry needs a unique `name`, a
   `model_path` (or rely on `metadata.model_name`), a `server_config` block,
   and at least one `client_configs` entry. Set `generation_mode` explicitly in
   `client_configs` for stable bucketing.

3. **Verify `hardware.gpus_per_node`** matches the GPU count implied by
   `server_config.parallel` (cfg × ulysses).

4. **Register the case in the test-db.** Add a line to
   `tests/integration/test_lists/test-db/l0_b200_visual_gen_perf_sanity.yml`
   using the `vg_upload-{config_base}-{server_name}` selector form. Use the
   `post_merge` block unless the case is fast enough for pre-merge.

5. **Run locally** with the `vg` prefix (no upload) to verify the case passes
   end-to-end before pushing:

   ```bash
   pytest perf/test_visual_gen_perf_sanity.py::test_visual_gen_e2e[vg-{config_base}-{server_name}]
   ```

6. **Verify uploaded results** after a `vg_upload` run by opening the
   VisualGen OpenSearch data explorer:

   ```
   https://gpuwa.nvidia.com/os-dashboards/app/data-explorer/discover?security_tenant=TRT-LLM-Infra#?_a=(discover:(columns:!(_source),isDirty:!f,sort:!()),metadata:(indexPattern:'6b689e00-e52e-11f0-ae33-a57c903184a7',view:discover))&_g=(filters:!(),refreshInterval:(pause:!t,value:0),time:(from:'2026-04-24T04:00:00.000Z',to:'2026-04-25T04:00:00.000Z'))&_q=(filters:!(('$state':(store:appState),meta:(alias:!n,disabled:!f,index:'6b689e00-e52e-11f0-ae33-a57c903184a7',key:s_runtime,negate:!f,params:(query:visual_gen),type:phrase),query:(match_phrase:(s_runtime:visual_gen)))),query:(language:kuery,query:''))
   ```

   The link pre-filters to `s_runtime: visual_gen`. To find a specific run,
   search for the server config name (e.g. `s_server_name:
   wan21_14b_nvfp4_trtllm_cfg2_ulysses4_teacache_on`) or narrow the time range
   to match the CI run timestamp.

## How to Add or Change a Match Key

Match keys define the dimensions used to look up a historical baseline in
OpenSearch. Changes to `MATCH_KEYS` in `visual_gen_perf_utils.py` are
**breaking** for existing baselines: any document stored under the old key set
will not be found by the new query.

**Adding a new feature flag or config knob:**

1. Add the extraction logic to `build_visual_gen_db_entry` in
   `visual_gen_perf_utils.py` (follow the existing `b_enable_*` / `l_*` /
   `s_*` naming convention).
2. Add the field name to `MATCH_KEYS`.
3. Update the match-key table in this README.
4. Existing baselines will not be matched until a new baseline run is uploaded
   with the new field set; plan a controlled baseline refresh if needed.

**Renaming or removing a field** invalidates all stored baselines for that
dimension. Coordinate with the team before merging to avoid silent mismatches.

**Changing the value domain of an existing field** (e.g., the format of
`s_quant_algo` changes due to an API refactor) also invalidates existing
baselines. If the upstream API changes the string representation, update the
extraction in `build_visual_gen_db_entry` and treat it as a baseline reset.

## Local Run Examples

```bash
# Run a single named recipe
pytest perf/test_visual_gen_perf_sanity.py::test_visual_gen_e2e[vg_upload-wan21_t2v_14b_blackwell-wan21_14b_nvfp4_trtllm_cfg2_ulysses4_teacache_on]

# Run all server recipes from one family YAML without uploading
pytest perf/test_visual_gen_perf_sanity.py::test_visual_gen_e2e[vg-ltx2_blackwell]
```

Set `LLM_MODELS_ROOT` to the shared model cache before running locally:

```bash
export LLM_MODELS_ROOT=/home/scratch.trt_llm_data_ci/llm-models
```

## Quick Reference

| Resource | Path |
|----------|------|
| Main pytest entry | `tests/integration/defs/perf/test_visual_gen_perf_sanity.py` |
| VisualGen helpers | `tests/integration/defs/perf/visual_gen_perf_utils.py` |
| Family YAMLs | `tests/scripts/perf-sanity/visual_gen/` |
| B200 post-merge test-db | `tests/integration/test_lists/test-db/l0_b200_visual_gen_perf_sanity.yml` |
| Jenkins stage wiring | `jenkins/L0_Test.groovy` |

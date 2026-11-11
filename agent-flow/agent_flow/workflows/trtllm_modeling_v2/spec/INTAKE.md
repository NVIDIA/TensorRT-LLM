<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Intake: from a checkpoint to `task.yaml`

Intake writes `task.yaml` and the native TensorRT-LLM options files before a run starts. It reads files and starts no GPU job. Read first: the checkpoint's `config.json`, its safetensors index and tokenizer, and `tensorrt_llm/_torch/_experimental/modeling_v2/README.md`.

## Derive a first spec

```bash
trtllm-modeling-v2 intake derive --checkpoint <ckpt-dir> --repo <trtllm-dir> --out <spec-dir>
```

Pass `--site <site.yaml>` instead of `--repo` when the cluster has a site file (`trtllm_repo_path`, `environment`, `agents`, defaults). `derive` writes `task.derived.yaml` with one deployment, the first, and its options in `engines/<name>.yaml`. It prints what it left open (`open:` lines: `arch`, `agents.defaults.model`, `environment.python`, `accuracy.model_key`) and the spec's warnings; the dry run prints the same warnings again for the spec as you edited it:

- **first deployment too narrow:** the weights take more than 75% of the first deployment's devices (`environment.gpu_memory_gib` says how much one has).
- **no accuracy anchor:** the first deployment names no benchmark, or has no reference row and no declared anchor: `onboarded` can then only mean it boots.
- **accuracy without CUDA graphs:** a scored deployment states `cuda_graph_config: null`: a benchmark takes 10 to 50 times longer.
- **host memory:** no `kv_cache_config.host_cache_size` under a memory limit.
- **anchor under one percent** (from the dry run and every start; never from `derive`, which writes no anchor): a declared `anchor` below 1 with `unit: percent`.

Validate without a GPU: `trtllm-modeling-v2 --task <spec> --workspace <ws> --dry-run`.

## Deployments

A deployment is one configuration the checkpoint is served in: a name (`[a-z][a-z0-9_]{0,40}`) and a native options file, read as it is. The first is `core`: it defines `onboarded` and stays first. Each further one is independent; add them by hand:

- **Another parallel layout:** copy `engines/tp1.yaml` to `engines/tep4.yaml`, set `tensor_parallel_size: 4` and `moe_expert_parallel_size: 4`, and append `- {name: tep4, options: engines/tep4.yaml}`.
- **A KV-cache dtype:** copy the options, set `kv_cache_config: {dtype: fp8}`, and append `- {name: kv_fp8, options: engines/kv_fp8.yaml, relative_to: tp1}`. Where no reference row fits, `relative_to` judges its score against that deployment's own (`its score - 2 x sqrt(2) x SE`).
- **Disaggregated:** `- {name: pd, env: {UCX_TLS: "^ib"}, ctx: {options: engines/ctx.yaml}, gen: {options: engines/gen.yaml}, relative_to: tp1}`. Both options files state `cache_transceiver_config: {backend: DEFAULT}` (the load refuses a side without a backend) and `disable_overlap_scheduler: true`; `env` states the site's UCX transports.
- **Per deployment, optional:** `inputs: [text, image]` (the boot also sends an image), `tests: [<pytest node id>]` (the only check of a drafter's acceptance), `env`, `timeout`, and on a multi-node site `launch` and `stop` (README, "A multi-node site").

Every option is compared with what each rank's engine resolved. `DEFAULT` or `auto` leaves a choice to the engine; what it chose is in the verdict.

An older spec's shorthand keys still load: they are expanded into `deployments` once, at load, and `--dry-run` lists what they came to. A spec states one form or the other.

## Benchmarks and anchors

The top-level `accuracy` block gives defaults for every deployment (`benchmark`, `model_key`, `reference_yaml`, `discriminators`, `dataset_path`, `samples`, `score_pattern`, `boot`, `timeout`). Leave the block out to onboard on the boot alone. A deployment may list its own `benchmarks`:

```yaml
benchmarks:
  - name: gsm8k
  - name: mmmu
    args: ["--post_process_fn", "<fn>"]
  - name: aa_lcr
    anchor: 0.62
    unit: fraction
    source: "<where and how 0.62 was measured>"
    samples: 100
    env: {NS_JUDGE_MODEL: "<judge-model>"}
```

- The anchor of a benchmark is, in this order: the `anchor` it declares (with `source`), the tree's reference row for `model_key` with exactly the deployment's tags, the score of the deployment `relative_to` names, or nothing. It is frozen in `<ws>/anchors.yaml` at the first start; delete the file to freeze again.
- With no anchor the benchmark still runs: its score is in the report with no verdict.
- `unit` is `percent` (the default) or `fraction`; `samples` is the count behind the standard error when the benchmark is not one the flow knows.
- An aggregated deployment is scored by `trtllm-eval <name>` (behind its `launch`), a disaggregated one through its proxy, where only `gsm8k` is judged.
- Datasets are found on the host (`dataset_path`, `$LLM_MODELS_ROOT/datasets/...`, the HuggingFace cache); accuracy runs offline.

## Agents and notes

State `agents.defaults` (`backend`, `model`, and `reasoning_effort` if you want one); a role may differ under `agents.roles` (`surveyor`, `kernel_writer`, `assembler`). Keep `notes` to what a worker cannot read from the checkpoint: the architecture in a few lines and prior art in the tree.

# DWDP Reproduction

This directory provides a thin reproduction layer on top of
`examples/disaggregated/slurm/benchmark/submit_dwdp.py`.
It does not modify that launcher. Instead, it combines:

- `env.yaml`: cluster, container, model, and dataset inputs provided by the user
- `dwdp_reproduce.yaml`: the DWDP reproduction matrix
- `reproduce.py`: the script that merges both files, generates full benchmark
  configs, and forwards them to `submit_dwdp.py`

## Files

- `env.yaml`
  Holds environment-specific inputs such as Slurm settings, container image,
  mount list, model path, and dataset mapping.
- `dwdp_reproduce.yaml`
  Holds only experiment parameters such as `isl`, `osl`, `ctx_tp`, `gen_tp`,
  `batch`, `prefetch`, and DWDP settings.
- `generated/`
  Output directory for the generated full configs that are passed to
  `submit_dwdp.py`.

## How It Works

`reproduce.py` reads `env.yaml` and `dwdp_reproduce.yaml`, generates one full
benchmark config per experiment, writes the config by default to `generated/`, then
invokes:

```bash
python examples/disaggregated/slurm/benchmark/submit_dwdp.py -c <generated_config>
```

## Configure `env.yaml`

Update these sections before running:

- `slurm`
  Set `partition`, `account`, `time`, and any cluster-specific `extra_args`.
- `hardware`
  Set `gpus_per_node` for your cluster.
- `environment`
  Set `container_image`, `container_mount`, `model_path`, and usually
  `trtllm_repo`.
  Leave `log_dir` unset unless you intentionally want a fixed log location.
  When `log_dir` is omitted, `submit_dwdp.py` creates a unique per-run log
  directory automatically.
- `datasets`
  Map short dataset keys to concrete dataset files.

`environment.work_dir` is optional. If omitted, `reproduce.py` automatically
points it to `examples/disaggregated/slurm/benchmark`, which is what
`submit_dwdp.py` expects for locating the benchmark shell scripts.

## Configure `dwdp_reproduce.yaml`

This file can define both context-only and end-to-end reproduction experiments.

The reproduction matrix is split into:

- `experiment_defaults`
  Common fields shared across many experiments.
- `experiments`
  One entry per benchmark case.

Each experiment may reference datasets in two ways:

- `dataset_key`: resolves through `env.yaml -> datasets`
- `dataset_file`: directly provides the full dataset path

`dataset_key` is the preferred path when several experiments share the same
dataset file.

## Usage

Install required Python dependency first:

```bash
python3 -m pip install pyyaml
```


```bash
python3 examples/dwdp/reproduce.py \
  --env-config /path/to/env.yaml \
  --reproduce-config /path/to/dwdp_reproduce.yaml \
  --output-dir /path/to/generated
```

Before running, update `dwdp_reproduce.yaml` as needed so it includes the
reproduction experiments you want to launch.

## Generated Configs

Generated configs are written by default to `examples/dwdp/generated/`.
The filenames include both the experiment name and the generated benchmark
identifier so they can be inspected or reused directly with
`submit_dwdp.py`.

> **IMPORTANT:** Leave `environment.log_dir` unset by default. Logs are written
> under `examples/disaggregated/slurm/benchmark/logs/`.

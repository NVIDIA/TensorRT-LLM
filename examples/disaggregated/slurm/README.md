# Disaggregated Serving Launcher for SLURM

## Overview

A tool for running disaggregated serving benchmarks with TRT-LLM on SLURM.

## Usage

```bash
python3 launcher.py --account <account> --partition <partition> --time <time> --job-name <job-name> --container-image <container-image> --config-file <config-file> --experiment-path <experiment-path> --request-allocation --num-gpus <num-gpus>
```

### Configuration

The configuration file should be in the following format:

```yaml
exec:
  model_path: <model-path>
  # Determines the disaggregated serving configuration
  config:
    context:
      tp: <tp>
      ep: <ep>
      pp: <pp>
      max_batch_size: <max_batch_size>
      max_num_tokens: <max_num_tokens>
      max_seq_len: <max_seq_len>
      config:
        # Determines the context server PyTorch configuration
        print_iter_log: true
        disable_overlap_scheduler: true
        kv_cache_config:
          free_gpu_memory_fraction: 0.75
          enable_block_reuse: false
    generation:
      tp: <tp>
      ep: <ep>
      pp: <pp>
      max_batch_size: <max_batch_size>
      max_num_tokens: <max_num_tokens>
      max_seq_len: <max_seq_len>
      config:
        # Determines the generation server PyTorch configuration
        print_iter_log: true
        kv_cache_config:
          free_gpu_memory_fraction: 0.75
          enable_block_reuse: false

# Determines the profiling configuration
profile:
  isl: <isl>
  osl: <osl>
  use_benchmark_serving: true
  concurrency:
    - <concurrency>
```

Please refer to the [config.yaml](config.yaml) file for an example configuration.

The experiment results will be saved in the experiment path.
